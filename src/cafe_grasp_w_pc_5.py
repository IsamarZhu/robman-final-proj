from pathlib import Path
import time
import mpld3
import numpy as np
import trimesh

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Parser,
    DiagramBuilder,
    Concatenate,
    PointCloud,
    Simulator,
    StartMeshcat,
    SpatialVelocity,
    Rgba,
    RigidTransform,
    RotationMatrix,
    InverseKinematics,
    Solve,
)
from pydrake.common.eigen_geometry import Quaternion
from pydrake.systems.primitives import ConstantVectorSource
from pydrake.geometry import (
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    MakeRenderEngineVtk,
    RenderEngineVtkParams,
)

from pydrake.multibody.parsing import ProcessModelDirectives, ModelDirectives
from manipulation import running_as_notebook
from manipulation.station import LoadScenario
from manipulation.icp import IterativeClosestPoint

from perception import add_cameras, get_depth
from pid_controller import PIDController, StaticPositionController

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

SCENARIO_PATH = Path("/workspaces/robman-final-proj/src/scenario.yaml")
MUG_MESH_PATH = Path(
    "/workspaces/robman-final-proj/assets/mug/google_16k/textured.obj"
)
VOXEL_SIZE = 0.005
TABLE_Z_THRESHOLD = 1.0
N_SAMPLE_POINTS = 1500
MAX_ICP_ITERS = 25

if running_as_notebook:
    mpld3.enable_notebook()

meshcat = StartMeshcat()
scenario_file = SCENARIO_PATH

with open(scenario_file, "r") as f:
    scenario_yaml = f.read()

scenario = LoadScenario(data=scenario_yaml)

builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
parser = Parser(plant)
parser.package_map().Add(
    "manipulation",
    "/usr/local/lib/python3.12/dist-packages/manipulation/models",
)

model_directives = ModelDirectives(directives=scenario.directives)
ProcessModelDirectives(model_directives, parser)

robot_base_instance = plant.GetModelInstanceByName("robot_base")
iiwa_arm_instance = plant.GetModelInstanceByName("iiwa_arm")
iiwa_plate_instance = plant.GetModelInstanceByName("iiwa_plate")
wsg_arm_instance = plant.GetModelInstanceByName("wsg_arm")
wsg_plate_instance = plant.GetModelInstanceByName("wsg_plate")

plant.Finalize()

renderer_name = "renderer"
scene_graph.AddRenderer(renderer_name, MakeRenderEngineVtk(RenderEngineVtkParams()))

visualizer = MeshcatVisualizer.AddToBuilder(
    builder, scene_graph, meshcat, MeshcatVisualizerParams()
)

add_cameras(builder, plant, scene_graph, scenario)

# --------------------------------------------------------------------------- #
# Initial conditions and controllers
# --------------------------------------------------------------------------- #

initial_positions_arm = np.array([
    -1.57,  # joint 1
    0.9,    # joint 2
    0,      # joint 3
    -0.9,   # joint 4
    0,      # joint 5
    1.6,    # joint 6
    0       # joint 7
])

initial_positions_plate = np.array([
    -1.57,  # joint 1
    -1.5,   # joint 2
    0,      # joint 3
    0.9,    # joint 4
    0,      # joint 5
    -1.2,   # joint 6
    0.3     # joint 7
])

initial_base_pose = np.array([
    -1.0, 0.0, 0.0, 0.0,  # quaternion
    1.4, 0.0, 0.8         # position of robot_base
])

robot_body_initial = RigidTransform(
    RotationMatrix(Quaternion(wxyz=initial_base_pose[:4])),
    initial_base_pose[4:]
)

# PID Controller gains for the free arm
kp_arm = 200.0
kd_arm = 50.0
ki_arm = 2.0

# PID controller for iiwa_arm (this is the one we’ll move via q_desired)
q_desired_arm = initial_positions_arm.copy()
arm_controller = builder.AddSystem(
    PIDController(kp=kp_arm, kd=kd_arm, ki=ki_arm, q_desired=q_desired_arm)
)
builder.Connect(
    plant.get_state_output_port(iiwa_arm_instance),
    arm_controller.input_port,
)
builder.Connect(
    arm_controller.output_port,
    plant.get_actuation_input_port(iiwa_arm_instance),
)

# # ---- Plate arm: make it just “hold” a fixed pose (no PIDController) ----
# # plate_controller = builder.AddSystem(
# #     StaticPositionController(q_desired=initial_positions_plate)
# # )
# # builder.Connect(
# #     plant.get_state_output_port(iiwa_plate_instance),
# #     plate_controller.get_input_port(0),
# # )
# # builder.Connect(
# #     plate_controller.get_output_port(0),
# #     plant.get_actuation_input_port(iiwa_plate_instance),
# # )
# # PID controller for the plate arm (fights gravity, keeps tray steady)
# kp_plate = 300.0
# kd_plate = 120.0
# ki_plate = 100.0  # integral term handles gravity/steady load

# plate_controller = builder.AddSystem(
#     PIDController(
#         kp=kp_plate,
#         kd=kd_plate,
#         ki=ki_plate,
#         q_desired=initial_positions_plate.copy(),
#     )
# )

# builder.Connect(
#     plant.get_state_output_port(iiwa_plate_instance),
#     plate_controller.input_port,
# )
# builder.Connect(
#     plate_controller.output_port,
#     plant.get_actuation_input_port(iiwa_plate_instance),
# )


# WSG gripper on plate stays gripping
wsg_plate_source = builder.AddSystem(ConstantVectorSource([0.0, 0.0]))
builder.Connect(
    wsg_plate_source.get_output_port(),
    plant.get_actuation_input_port(wsg_plate_instance),
)

# For the free arm WSG we’ll command joint positions directly (no actuation input)
wsg_arm_source = builder.AddSystem(ConstantVectorSource([0.0, 0.0]))
builder.Connect(
    wsg_arm_source.get_output_port(),
    plant.get_actuation_input_port(wsg_arm_instance),
)

# def freeze_plate():
#     # Force the plate arm to stay at the initial joint config, with zero velocity
#     plant.SetPositions(plant_context, iiwa_plate_instance, initial_positions_plate)
#     plant.SetVelocities(plant_context, iiwa_plate_instance, np.zeros(7))


# --------------------------------------------------------------------------- #
# Build diagram and set initial state
# --------------------------------------------------------------------------- #

diagram = builder.Build()
simulator = Simulator(diagram)
context = simulator.get_mutable_context()
plant_context = plant.GetMyMutableContextFromRoot(context)

robot_base_body = plant.GetBodyByName("robot_base_link", robot_base_instance)
plant.SetFreeBodyPose(plant_context, robot_base_body, robot_body_initial)
zero_vel = SpatialVelocity(w=[0., 0., 0.], v=[0., 0., 0.])
plant.SetFreeBodySpatialVelocity(robot_base_body, zero_vel, plant_context)

plant.SetPositions(plant_context, iiwa_arm_instance, initial_positions_arm)
plant.SetPositions(plant_context, iiwa_plate_instance, initial_positions_plate)
plant.SetVelocities(plant_context, iiwa_arm_instance, np.zeros(7))
plant.SetVelocities(plant_context, iiwa_plate_instance, np.zeros(7))

# Lock all joints of the plate iiwa at their current (initial) configuration
for i in range(1, 8):
    joint = plant.GetJointByName(f"iiwa_joint_{i}", iiwa_plate_instance)
    joint.Lock(plant_context)


q_start = initial_positions_arm.copy()
arm_controller.set_desired_position(q_start)

# plate_controller.set_desired_position(initial_positions_plate.copy())

simulator.Initialize()
simulator.AdvanceTo(0.5)   # ~0.5 s is usually enough to let mugs settle
diagram.ForcedPublish(context)

if running_as_notebook:
    simulator.set_target_realtime_rate(1.0)

# --------------------------------------------------------------------------- #
# Point cloud helpers: remove tray & keep only mug rims
# --------------------------------------------------------------------------- #

def remove_table_points(pc: PointCloud,
                        z_thresh: float = TABLE_Z_THRESHOLD) -> PointCloud:
    """Keep only points with z > z_thresh."""
    xyz = pc.xyzs()
    if xyz.shape[1] == 0:
        return pc
    z = xyz[2, :]
    mask = (z > z_thresh)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return PointCloud(0)
    filt_xyz = xyz[:, idx]
    out = PointCloud(filt_xyz.shape[1])
    out.mutable_xyzs()[:] = filt_xyz
    return out


def keep_top_rim_band(pc: PointCloud, band_thickness: float = 0.02) -> PointCloud:
    """
    Keep only the top band_thickness meters near the global max z.
    This should capture just mug rims (and not the tray surface).
    """
    xyz = pc.xyzs()
    if xyz.shape[1] == 0:
        return pc
    z = xyz[2, :]
    z_max = float(np.max(z))
    mask = (z > (z_max - band_thickness))
    idx = np.where(mask)[0]
    if idx.size == 0:
        return pc
    rim_xyz = xyz[:, idx]
    out = PointCloud(rim_xyz.shape[1])
    out.mutable_xyzs()[:] = rim_xyz
    return out

def split_rim_into_mugs(rim_pc: PointCloud, k: int = 3):
    """
    Split rim point cloud into k clusters in (x, y).
    Returns a list of PointClouds, one per mug.
    """
    xyz = rim_pc.xyzs()
    n = xyz.shape[1]
    if n == 0:
        return []

    # Work in (x, y) only
    pts_xy = xyz[:2, :].T  # (N, 2)

    # Simple K-means implementation in numpy to avoid extra deps.
    # (Random init, few iterations is good enough here.)
    np.random.seed(0)
    # Initialize centroids by randomly picking k points
    indices = np.random.choice(n, k, replace=False)
    centroids = pts_xy[indices]

    for _ in range(15):
        # Assign points to nearest centroid
        # (N, k) distances
        diff = pts_xy[:, None, :] - centroids[None, :, :]
        dists = np.sum(diff**2, axis=2)
        labels = np.argmin(dists, axis=1)

        # Update centroids
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            mask = (labels == i)
            if np.any(mask):
                new_centroids[i] = pts_xy[mask].mean(axis=0)
            else:
                # If a cluster is empty, reinitialize it
                new_centroids[i] = pts_xy[np.random.randint(0, n)]
        centroids = new_centroids

    # Build separate PointClouds
    mug_pcs = []
    for i in range(k):
        mask = (labels == i)
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        mug_xyz = xyz[:, idx]
        pc = PointCloud(mug_xyz.shape[1])
        pc.mutable_xyzs()[:] = mug_xyz
        mug_pcs.append(pc)

    return mug_pcs



def build_rim_pointcloud(diagram, context) -> PointCloud:
    # Grab depth images (for debugging/plots if you want)
    get_depth(diagram, context)

    pc0 = diagram.GetOutputPort("camera_point_cloud0").Eval(context)
    pc1 = diagram.GetOutputPort("camera_point_cloud1").Eval(context)
    pc2 = diagram.GetOutputPort("camera_point_cloud2").Eval(context)

    concat_pc = Concatenate([pc0, pc1, pc2])
    down_pc = concat_pc.VoxelizedDownSample(VOXEL_SIZE)

    xyz_down = down_pc.xyzs()
    print("Downsampled cloud points:", xyz_down.shape[1])
    print("Downsampled z min/max:", float(np.min(xyz_down[2, :])),
          float(np.max(xyz_down[2, :])))

    obj_pc = remove_table_points(down_pc, z_thresh=TABLE_Z_THRESHOLD)
    if obj_pc.xyzs().shape[1] == 0:
        # relax if needed
        obj_pc = remove_table_points(down_pc, z_thresh=TABLE_Z_THRESHOLD - 0.1)
        if obj_pc.xyzs().shape[1] == 0:
            obj_pc = down_pc

    print("After table removal: num points =", obj_pc.xyzs().shape[1])

    rim_pc = keep_top_rim_band(obj_pc, band_thickness=0.02)
    if rim_pc.xyzs().shape[1] == 0:
        rim_pc = obj_pc

    xyz_rim = rim_pc.xyzs()
    print("Rim cloud: num points =", xyz_rim.shape[1])
    print("Rim z min/max:", float(np.min(xyz_rim[2, :])),
          float(np.max(xyz_rim[2, :])))

    meshcat.SetObject(
        "perception/mug_rim_cloud",
        rim_pc,
        point_size=0.05,
        rgba=Rgba(1, 0, 0, 0.4),
    )
    diagram.ForcedPublish(context)
    return rim_pc


# --------------------------------------------------------------------------- #
# ICP on rim cloud to estimate mug pose
# --------------------------------------------------------------------------- #

def estimate_mug_pose_icp(meshcat, rim_pc: PointCloud):
    p_Ws = rim_pc.xyzs()
    if p_Ws.shape[1] == 0:
        raise RuntimeError("estimate_mug_pose_icp: rim_pc has no points!")

    # Load mug model points in mug frame
    if MUG_MESH_PATH.is_file():
        mug_mesh = trimesh.load(str(MUG_MESH_PATH), force="mesh")
        pts = mug_mesh.sample(N_SAMPLE_POINTS)   # (N, 3)
        p_Om = pts.T                             # (3, N)
        print(f"Loaded mug mesh from {MUG_MESH_PATH}")
    else:
        print(f"WARNING: {MUG_MESH_PATH} not found; using cylindrical mug model.")
        np.random.seed(0)
        radius = 0.045
        height = 0.09
        theta = 2 * np.pi * np.random.rand(N_SAMPLE_POINTS)
        z = height * np.random.rand(N_SAMPLE_POINTS)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        p_Om = np.vstack([x, y, z])

    # --- Build an initial guess from rim + model geometry ---
    # World rim center and top
    center_xyz = np.mean(p_Ws, axis=1)      # (3,)
    center_x, center_y = center_xyz[0], center_xyz[1]
    z_rim_world = float(np.max(p_Ws[2, :]))

    # Mug model top (in its own frame)
    model_top_z    = float(np.max(p_Om[2, :]))
    model_bottom_z = float(np.min(p_Om[2, :]))

    # Translate model so its top aligns with rim z, and center x,y align
    initial_translation = [
        center_x,
        center_y,
        z_rim_world - model_top_z,
    ]

    initial_guess = RigidTransform(
        RotationMatrix(),   # assume mug is upright
        initial_translation,
    )

    # --- Run ICP ---
    X_WM_hat, cost = IterativeClosestPoint(
        p_Om=p_Om,
        p_Ws=p_Ws,
        X_Ohat=initial_guess,
        meshcat=meshcat,
        meshcat_scene_path="icp/mug",
        max_iterations=MAX_ICP_ITERS,
    )
    print("ICP cost:", cost)

    # Transform model points to world to get true mug top/bottom from ICP
    R_WM = X_WM_hat.rotation().matrix()
    p_WM = X_WM_hat.translation().reshape(3, 1)

    p_Wm = R_WM @ p_Om + p_WM
    z_vals = p_Wm[2, :]

    mug_bottom_z = float(np.min(z_vals))
    mug_top_z    = float(np.max(z_vals))

    return X_WM_hat, (mug_bottom_z, mug_top_z)



# --------------------------------------------------------------------------- #
# IK + pick-and-place sequence using the PID arm controller
# --------------------------------------------------------------------------- #

def run_pick_place(meshcat, simulator, diagram, plant, plant_context,
                   iiwa_arm_instance, wsg_arm_instance,
                   X_WM_hat: RigidTransform,
                   mug_bottom_z: float,
                   mug_top_z: float,
                   q_start=None):

    world_frame = plant.world_frame()

    # ----------------------------------------------------------------------- #
    # Frames + tray / table heights
    # ----------------------------------------------------------------------- #
    ee_frame = plant.GetFrameByName("body", wsg_arm_instance)

    # Table the mug will be placed on (on the ground)
    table_inst = plant.GetModelInstanceByName("table0")
    table_body = plant.GetBodyByName("table_body", table_inst)
    X_WT = plant.EvalBodyPoseInWorld(plant_context, table_body)
    p_WT = X_WT.translation()
    table_height = p_WT[2]
    print("Table height:", table_height)

    # Tray the mugs start on (attached to plate arm, in the air)
    tray_inst = plant.GetModelInstanceByName("tray")
    tray_body = plant.GetBodyByName("tray_link", tray_inst)
    X_WTray = plant.EvalBodyPoseInWorld(plant_context, tray_body)
    p_WTray = X_WTray.translation()
    tray_height = p_WTray[2]
    print("Tray height:", tray_height)

    # gripper orientation: -z aligned with world -z (top-down)
    R_WG_desired = RotationMatrix.MakeXRotation(-np.pi / 2.0)

    # mug position from ICP
    p_WM = X_WM_hat.translation()

    # Estimate mug bottom, but don't let it go below tray surface
    # mug_bottom_z = mug_top_z - 0.08          # assume ~8cm tall mug
    # mug_bottom_z = max(mug_bottom_z, tray_height + 0.005)

    # ----------------------------------------------------------------------- #
    # IK helper
    # ----------------------------------------------------------------------- #
    def ik_for_ee_position(p_W_target,
                       use_orientation: bool,
                       theta_bound: float = 1.0,
                       clamp_to_tray: bool = True):
        ik = InverseKinematics(plant, plant_context)
        q = ik.q()

        world_frame = plant.world_frame()
        ee_frame = plant.GetFrameByName("body", wsg_arm_instance)

        # lock base
        base_body = plant.GetBodyByName("robot_base_link", robot_base_instance)
        X_WB = plant.EvalBodyPoseInWorld(plant_context, base_body)
        p_WB = X_WB.translation()
        ik.AddPositionConstraint(
            base_body.body_frame(), [0, 0, 0],
            world_frame,
            p_WB, p_WB,
        )
        R_WB = X_WB.rotation()
        ik.AddOrientationConstraint(
            world_frame, R_WB,
            base_body.body_frame(), RotationMatrix(),
            0.0,
        )

        # --- position constraint box ---
        p = p_W_target.copy()
        if clamp_to_tray:
            # never let the lower z bound go below mug_bottom_z + margin
            p[2] = max(p[2], mug_bottom_z + 0.01)

        p_tol = 0.06

        lower = p.copy()
        upper = p.copy()

        # x,y ± p_tol
        lower[0:2] = p[0:2] - p_tol
        upper[0:2] = p[0:2] + p_tol

        if clamp_to_tray:
            # z ∈ [p[2], p[2] + p_tol]  (one-sided; keeps us above tray/mugs)
            lower[2] = p[2]
            upper[2] = p[2] + p_tol
        else:
            # symmetric z near table targets
            lower[2] = p[2] - p_tol
            upper[2] = p[2] + p_tol

        ik.AddPositionConstraint(
            ee_frame, [0, 0, 0],
            world_frame,
            lower,
            upper,
        )

        if use_orientation:
            ik.AddOrientationConstraint(
                world_frame,
                R_WG_desired,
                ee_frame,
                RotationMatrix(),
                theta_bound,
            )

        prog = ik.prog()
        q_seed = plant.GetPositions(plant_context)
        prog.SetInitialGuess(q, q_seed)

        result = Solve(prog)
        print(f"IK target (raw) {p_W_target}, (used) {p}, clamp_to_tray={clamp_to_tray}")

        if not result.is_success():
            raise RuntimeError(f"IK failed for target {p_W_target} (clamped {p})")

        q_sol = result.GetSolution(q)
        plant.SetPositions(plant_context, q_sol)
        return plant.GetPositions(plant_context, iiwa_arm_instance)


    # ----------------------------------------------------------------------- #
    # Waypoints in z
    # ----------------------------------------------------------------------- #
    # Waypoints in z (all derived from ICP + tray/table)
    safe_fly_height = max(tray_height, table_height) + 0.25  # always well above everything

    z_over_mug = max(mug_top_z + 0.15, safe_fly_height)
    z_grasp    = mug_top_z + 0.02   # just above rim, not in the mug
    z_lift     = max(mug_top_z + 0.25, safe_fly_height)



    p_over_mug = np.array([p_WM[0], p_WM[1], z_over_mug])
    p_grasp    = np.array([p_WM[0], p_WM[1], z_grasp])
    p_lift     = np.array([p_WM[0], p_WM[1], z_lift])


    table_height_for_place = table_height + 0.10
    p_place_xy = np.array([0.9, -0.05])

    p_over_table = np.array([
        p_place_xy[0],
        p_place_xy[1],
        max(table_height_for_place + 0.25, safe_fly_height),
    ])

    p_place = np.array([
        p_place_xy[0],
        p_place_xy[1],
        table_height_for_place + 0.02,   # closer to table for actual place
    ])

    p_retreat = p_over_table.copy()

    # ----------------------------------------------------------------------- #
    # IK solve for keyframes
    # ----------------------------------------------------------------------- #
    if q_start is None:
        q_start = plant.GetPositions(plant_context, iiwa_arm_instance)

    # 1) Enforce top-down orientation *above* the mug
    # Over mug / grasp / lift: clamp to tray (protect mugs)
    q_over_m = ik_for_ee_position(
        p_over_mug,
        use_orientation=True,
        theta_bound=1.0,
        clamp_to_tray=True,
    )

    # q_grasp = ik_for_ee_position(
    #     p_grasp,
    #     use_orientation=True,
    #     theta_bound=0.4,
    #     clamp_to_tray=True,
    # )
    
    q_grasp = ik_for_ee_position(
        p_grasp,
        use_orientation=False,  # <- important
        clamp_to_tray=True,
    )

    q_lift = ik_for_ee_position(
        p_lift,
        use_orientation=False,
        clamp_to_tray=True,
    )

    # Over table / place / retreat: allow going down to table
    q_over_t = ik_for_ee_position(
        p_over_table,
        use_orientation=False,
        clamp_to_tray=False,   # <---
    )
    q_place = ik_for_ee_position(
        p_place,
        use_orientation=False,
        clamp_to_tray=False,   # <---
    )
    q_retreat = ik_for_ee_position(
        p_retreat,
        use_orientation=False,
        clamp_to_tray=False,   # <---
    )
    
    print("tray_height:", tray_height, "table_height:", table_height)
    print("mug_top_z:", mug_top_z, "mug_bottom_z:", mug_bottom_z)
    print("p_over_mug:", p_over_mug)
    print("p_grasp:", p_grasp)
    print("p_over_table:", p_over_table, "p_place:", p_place)



    keyframes = [
        ("start",     q_start,   "open"),
        ("over_mug",  q_over_m,  "open"),
        ("grasp",     q_grasp,   "open_then_close"),
        ("lift",      q_lift,    "closed"),
        ("over_table",q_over_t,  "closed"),
        ("place",     q_place,   "open"),
        ("retreat",   q_retreat, "open"),
    ]

    # ----------------------------------------------------------------------- #
    # Initialize arm + gripper
    # ----------------------------------------------------------------------- #
    plant.SetPositions(plant_context, iiwa_arm_instance, q_start)
    plant.SetVelocities(plant_context, iiwa_arm_instance, np.zeros(7))
    arm_controller.set_desired_position(q_start)
    diagram.ForcedPublish(simulator.get_context())

    # WSG joint handles
    left_joint  = plant.GetJointByName("left_finger_sliding_joint",  wsg_arm_instance)
    right_joint = plant.GetJointByName("right_finger_sliding_joint", wsg_arm_instance)

    def set_wsg_width(width: float):
        left_joint.set_translation(plant_context,  +0.5 * width)
        right_joint.set_translation(plant_context, -0.5 * width)

    WSG_OPEN   = 0.12   # slightly > mug outer diameter
    WSG_CLOSED = 0.06   # enough to clamp the wall

    # ----------------------------------------------------------------------- #
    # Execute keyframes
    # ----------------------------------------------------------------------- #
    t = simulator.get_context().get_time()
    phase_dt = 2.0

    def move_to(q_desired, gripper_mode: str):
        nonlocal t

        # update PID desired position
        arm_controller.set_desired_position(q_desired)

        if gripper_mode == "open":
            set_wsg_width(WSG_OPEN)
        elif gripper_mode == "closed":
            set_wsg_width(WSG_CLOSED)
        elif gripper_mode == "open_then_close":
            set_wsg_width(WSG_OPEN)

        # advance full phase, with a mid-phase close if needed
        t_mid = t + phase_dt / 2.0
        if gripper_mode == "open_then_close":
            simulator.AdvanceTo(t_mid)
            set_wsg_width(WSG_CLOSED)
        simulator.AdvanceTo(t + phase_dt)
        t += phase_dt
        diagram.ForcedPublish(simulator.get_context())

    print("Executing pick-and-place sequence...")
    for name, q, mode in keyframes:
        print("  ->", name, "mode:", mode)
        move_to(q, mode)


# --------------------------------------------------------------------------- #
# Main flow
# --------------------------------------------------------------------------- #

print(f"Meshcat URL: {meshcat.web_url()}")

# Build rim-only cloud and ICP mug pose
# rim_pc = build_rim_pointcloud(diagram, context)
# X_WM_hat, mug_top_z = estimate_mug_pose_icp(meshcat, rim_pc)
# print("Estimated mug pose:", X_WM_hat)
# print("Estimated mug_top_z:", mug_top_z)


rim_pc = build_rim_pointcloud(diagram, context)

# Split into three mugs
mug_rim_pcs = split_rim_into_mugs(rim_pc, k=3)
print("Found", len(mug_rim_pcs), "rim clusters")

# For debugging: visualize each cluster in a different path
for i, pc in enumerate(mug_rim_pcs):
    meshcat.SetObject(
        f"perception/mug_rim_cloud_cluster_{i}",
        pc,
        point_size=0.05,
        rgba=Rgba(1, 0, 0, 0.4),
    )

# Pick which mug to grasp: e.g., the one closest to the robot in x
robot_base_body = plant.GetBodyByName("robot_base_link", robot_base_instance)
X_WR = plant.EvalBodyPoseInWorld(plant_context, robot_base_body)
p_WR = X_WR.translation()

def rim_center(pc: PointCloud):
    xyz = pc.xyzs()
    return np.mean(xyz, axis=1)

dists = [np.linalg.norm(rim_center(pc)[:2] - p_WR[:2]) for pc in mug_rim_pcs]
chosen_idx = int(np.argmin(dists))
chosen_rim_pc = mug_rim_pcs[chosen_idx]
print("Chose mug cluster", chosen_idx, "for ICP")

# Recompute mug_top_z from that one mug only
p_Ws_chosen = chosen_rim_pc.xyzs()
mug_top_z = float(np.max(p_Ws_chosen[2, :]))

# Run ICP only on that mug's rim
X_WM_hat, (mug_bottom_z, mug_top_z) = estimate_mug_pose_icp(meshcat, chosen_rim_pc)
print("Estimated mug pose:", X_WM_hat)
print("Estimated mug_bottom_z:", mug_bottom_z)
print("Estimated mug_top_z:", mug_top_z)

# (Optional) compare to ground truth mug pose as in step 1


# --- Debug: compare ICP pose to ground truth mug1 pose ---
mug1_instance = plant.GetModelInstanceByName("mug1")
mug1_body = plant.GetBodyByName("base_link", mug1_instance)
X_WM_true = plant.EvalBodyPoseInWorld(plant_context, mug1_body)
print("Ground truth mug1 pose:", X_WM_true)
print("ICP translation:", X_WM_hat.translation())
print("True translation:", X_WM_true.translation())


meshcat.StartRecording()

# Run the pick-and-place sequence with the free arm
run_pick_place(
    meshcat, simulator, diagram, plant, plant_context,
    iiwa_arm_instance, wsg_arm_instance,
    X_WM_hat, mug_bottom_z, mug_top_z,
    q_start,
)


meshcat.StopRecording()
meshcat.PublishRecording()
input("Simulation done. Press Enter to exit...")

print("Simulation complete!")
print("Final arm position:", plant.GetPositions(plant_context, iiwa_arm_instance))
print("Final plate position:", plant.GetPositions(plant_context, iiwa_plate_instance))
