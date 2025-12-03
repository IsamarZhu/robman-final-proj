# file: new_cafe_w_pc.py

from pathlib import Path
import time
import numpy as np
import trimesh

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    InverseKinematics,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    PointCloud,
    Rgba,
    RigidTransform,
    RotationMatrix,
    Simulator,
    Solve,
    SpatialVelocity,
    StartMeshcat,
)
from pydrake.geometry import (
    MakeRenderEngineVtk,
    RenderEngineVtkParams,
    Box,
)
from pydrake.multibody.parsing import Parser, ProcessModelDirectives, ModelDirectives

from pydrake.systems.primitives import (
    ConstantVectorSource,
    MatrixGain,
)

from manipulation import running_as_notebook
from manipulation.station import LoadScenario
from manipulation.icp import IterativeClosestPoint

from perception.perception import add_cameras, get_depth
from pid_controller import PIDController  # your custom controller

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

SCENARIO_PATH = Path("/workspaces/robman-final-proj/src/new_scenario.yaml")
MUG_MESH_PATH = Path(
    "/workspaces/robman-final-proj/assets/mug/google_16k/textured.obj"
)

VOXEL_SIZE = 0.005
BAND_THICKNESS = 0.02          # top band for rim
N_SAMPLE_POINTS = 1500
MAX_ICP_ITERS = 25

TABLE_Z_THRESH_MARGIN = 0.02   # how far above "table" z we keep points
SAFE_CLEARANCE = 0.25          # flying height above largest object
GRASP_BELOW_RIM = 0.02
PLACE_ABOVE_TABLE = 0.02

WSG_OPEN = 0.10
WSG_CLOSED = 0.02
PHASE_DT = 2.5                 # seconds per keyframe

if running_as_notebook:
    import mpld3
    mpld3.enable_notebook()


# --------------------------------------------------------------------------- #
# Point cloud helpers
# --------------------------------------------------------------------------- #

def downsample(pc: PointCloud, voxel_size: float) -> PointCloud:
    if pc.xyzs().shape[1] == 0:
        return pc
    return pc.VoxelizedDownSample(voxel_size)


def remove_below_z(pc: PointCloud, z_thresh: float) -> PointCloud:
    """Keep only points with z > z_thresh."""
    xyz = pc.xyzs()
    if xyz.shape[1] == 0:
        return pc
    z = xyz[2, :]
    mask = (z > z_thresh)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return PointCloud(0)
    sel = xyz[:, idx]
    out = PointCloud(sel.shape[1])
    out.mutable_xyzs()[:] = sel
    return out


def keep_top_band(pc: PointCloud, band_thickness: float) -> PointCloud:
    """Keep only the top band_thickness meters near the global max z."""
    xyz = pc.xyzs()
    if xyz.shape[1] == 0:
        return pc
    z = xyz[2, :]
    z_max = float(np.max(z))
    mask = (z > (z_max - band_thickness))
    idx = np.where(mask)[0]
    if idx.size == 0:
        return pc
    sel = xyz[:, idx]
    out = PointCloud(sel.shape[1])
    out.mutable_xyzs()[:] = sel
    return out


def build_rim_pointcloud(diagram, context, meshcat) -> PointCloud:
    """
    Uses camera_point_cloud{0,1,2}, removes everything below tray/table,
    and then keeps only the top band (mug rim).
    """

    # For debugging (optional):
    get_depth(diagram, context)

    # These ports are added by perception.add_cameras(...)
    pc0 = diagram.GetOutputPort("camera_point_cloud0").Eval(context)
    pc1 = diagram.GetOutputPort("camera_point_cloud1").Eval(context)
    pc2 = diagram.GetOutputPort("camera_point_cloud2").Eval(context)

    # Concatenate -> (3, N)
    xyz = np.concatenate(
        [pc0.xyzs(), pc1.xyzs(), pc2.xyzs()],
        axis=1,
    )
    concat_pc = PointCloud(xyz.shape[1])
    concat_pc.mutable_xyzs()[:] = xyz

    # Downsample a bit
    down_pc = downsample(concat_pc, VOXEL_SIZE)
    xyz_down = down_pc.xyzs()
    print("Downsampled N =", xyz_down.shape[1])
    print(
        "z min/max:",
        float(np.min(xyz_down[2, :])),
        float(np.max(xyz_down[2, :])),
    )

    # Estimate "support surface" height from the 10th percentile of z
    z_vals = xyz_down[2, :]
    if z_vals.size == 0:
        return down_pc

    z_surface = float(np.percentile(z_vals, 10))
    z_thresh = z_surface + TABLE_Z_THRESH_MARGIN
    print(
        "Estimated support surface z:",
        z_surface,
        " -> using thresh:",
        z_thresh,
    )

    obj_pc = remove_below_z(down_pc, z_thresh)
    print("After below-z removal: N =", obj_pc.xyzs().shape[1])

    rim_pc = keep_top_band(obj_pc, BAND_THICKNESS)
    if rim_pc.xyzs().shape[1] == 0:
        rim_pc = obj_pc

    xyz_rim = rim_pc.xyzs()
    print("Rim N =", xyz_rim.shape[1])
    print(
        "Rim z min/max:",
        float(np.min(xyz_rim[2, :])),
        float(np.max(xyz_rim[2, :])),
    )

    meshcat.SetObject(
        "perception/mug_rim_cloud",
        rim_pc,
        point_size=0.05,
        rgba=Rgba(1, 0, 0, 0.4),
    )
    diagram.ForcedPublish(context)
    return rim_pc


# --------------------------------------------------------------------------- #
# ICP on rim cloud
# --------------------------------------------------------------------------- #

def estimate_mug_pose_icp(meshcat, rim_pc: PointCloud):
    p_Ws = rim_pc.xyzs()
    if p_Ws.shape[1] == 0:
        raise RuntimeError("rim_pc has no points")

    # Load and sample mug mesh in its own frame
    mug_mesh = trimesh.load(str(MUG_MESH_PATH), force="mesh")
    pts = mug_mesh.sample(N_SAMPLE_POINTS)  # (N, 3)
    p_Om = pts.T                            # (3, N)

    # Initial guess: center of rim, mug upright
    center_xyz = np.mean(p_Ws, axis=1)
    z_rim_world = float(np.max(p_Ws[2, :]))
    model_top_z = float(np.max(p_Om[2, :]))

    initial_translation = [
        center_xyz[0],
        center_xyz[1],
        z_rim_world - model_top_z,
    ]
    initial_guess = RigidTransform(
        RotationMatrix(),  # assume mug is upright-ish
        initial_translation,
    )

    X_WM_hat, cost = IterativeClosestPoint(
        p_Om=p_Om,
        p_Ws=p_Ws,
        X_Ohat=initial_guess,
        meshcat=meshcat,
        meshcat_scene_path="icp/mug",
        max_iterations=MAX_ICP_ITERS,
    )
    print("ICP cost:", cost)

    # Use ICP result to compute mug top/bottom in world
    R_WM = X_WM_hat.rotation().matrix()
    p_WM = X_WM_hat.translation().reshape(3, 1)
    p_Wm = R_WM @ p_Om + p_WM
    z_vals = p_Wm[2, :]

    mug_bottom_z = float(np.min(z_vals))
    mug_top_z = float(np.max(z_vals))

    return X_WM_hat, mug_bottom_z, mug_top_z


# --------------------------------------------------------------------------- #
# IK + keyframe motion (using arm PID controller)
# --------------------------------------------------------------------------- #

def ik_for_ee_position(
    plant,
    plant_context,
    iiwa_model,
    wsg_model,
    p_W_target,
    R_WG_desired=None,
    theta_bound=0.8,   # looser than 0.3
):
    """
    Solve IK for the WSG 'body' frame to reach p_W_target, optionally with
    desired orientation R_WG_desired.

    - IK works in the full plant generalized position vector.
    - We lock the iiwa mobile base joints (iiwa_base_x/y/z) to their current
      values so that IK only moves the 7 arm joints.
    - We return the 10 positions for the iiwa model (3 base + 7 joints).
    """
    # Full seed for the entire plant (all models)
    q_seed_full = plant.GetPositions(plant_context)

    ik = InverseKinematics(plant, plant_context)
    q_decision = ik.q()          # full generalized positions (all models)
    world_frame = plant.world_frame()
    ee_frame = plant.GetFrameByName("body", wsg_model)

    # --- Lock iiwa base joints (keep base "fixed" for IK) ---
    for joint_name in ["iiwa_base_x", "iiwa_base_y", "iiwa_base_z"]:
        joint = plant.GetJointByName(joint_name, iiwa_model)
        idx = joint.position_start()   # index into the full q
        q_val = q_seed_full[idx]
        ik.prog().AddBoundingBoxConstraint(
            q_val, q_val, q_decision[idx:idx + 1]
        )

    # --- Position constraint around target (looser box now) ---
    p = np.copy(p_W_target)
    p_tol_xy = 0.08   # ~8cm slack in x/y
    p_tol_z  = 0.08   # ~8cm slack in z

    lower = np.array([p[0] - p_tol_xy, p[1] - p_tol_xy, p[2] - p_tol_z])
    upper = np.array([p[0] + p_tol_xy, p[1] + p_tol_xy, p[2] + p_tol_z])

    ik.AddPositionConstraint(
        ee_frame, [0, 0, 0],
        world_frame,
        lower,
        upper,
    )

    # --- Orientation constraint (gripper "pointing down") ---
    if R_WG_desired is not None:
        ik.AddOrientationConstraint(
            world_frame,
            R_WG_desired,
            ee_frame,
            RotationMatrix(),
            theta_bound,
        )

    prog = ik.prog()
    prog.SetInitialGuess(q_decision, q_seed_full)

    result = Solve(prog)
    if not result.is_success():
        print("IK FAILED for target:", p_W_target)
        print("  lower:", lower, "upper:", upper)
        raise RuntimeError(f"IK failed for target {p_W_target}")

    q_sol_full = result.GetSolution(q_decision)

    # Write full solution back to plant
    plant.SetPositions(plant_context, q_sol_full)

    # Extract only the iiwa model positions (10 DOFs: base+7 joints)
    q_iiwa = plant.GetPositions(plant_context, iiwa_model)
    return np.copy(q_iiwa)


def run_pick_place(
    meshcat,
    simulator,
    diagram,
    plant,
    plant_context,
    iiwa_model,
    wsg_model,
    arm_controller,
    X_WM_hat,
    mug_bottom_z,
    mug_top_z,
    q_start=None,
):
    """
    Full pick-and-place:
      1) Move above the mug with gripper pointing down (open).
      2) Descend until the fingers are around the rim.
      3) Close the gripper.
      4) Lift the mug.
      5) Move over the table.
      6) Place the mug on the table and retreat.
    """
    context = simulator.get_mutable_context()

    # Frames for environment
    table_inst = plant.GetModelInstanceByName("table0")
    table_body = plant.GetBodyByName("table_body", table_inst)
    X_WT = plant.EvalBodyPoseInWorld(plant_context, table_body)
    p_WT = X_WT.translation()
    table_height = p_WT[2]

    tray_inst = plant.GetModelInstanceByName("tray")
    tray_body = plant.GetBodyByName("tray_link", tray_inst)
    X_WTray = plant.EvalBodyPoseInWorld(plant_context, tray_body)
    p_WTray = X_WTray.translation()
    tray_height = p_WTray[2]

    print("table_height:", table_height, "tray_height:", tray_height)

    # Gripper orientation: "point down"
    R_WG_desired = RotationMatrix.MakeXRotation(-np.pi / 2.0)

    p_WM = X_WM_hat.translation()

    # ------------------ Pick-side keypoints (tray) ------------------ #
    # Safe hover above mug: relative to tray, guaranteed clearance.
    z_over_mug = max(
        mug_top_z + 0.15,      # at least 15cm above rim
        tray_height + 0.35     # and at least 35cm above tray
    )
    z_grasp = mug_top_z - GRASP_BELOW_RIM   # fingers around the rim
    z_lift = max(
        mug_top_z + 0.18,      # slightly above rim
        tray_height + 0.25     # and clearly above tray
    )

    p_over_mug = np.array([p_WM[0], p_WM[1], z_over_mug])
    p_grasp    = np.array([p_WM[0], p_WM[1], z_grasp])
    p_lift     = np.array([p_WM[0], p_WM[1], z_lift])

    print("p_over_mug:", p_over_mug)
    print("p_grasp   :", p_grasp)
    print("p_lift    :", p_lift)

    # Optional debug visuals for these targets
    meshcat.SetObject("debug/over_mug",
                      Box(0.03, 0.03, 0.03), Rgba(0, 1, 0, 0.4))
    meshcat.SetTransform("debug/over_mug",
                         RigidTransform(R_WG_desired, p_over_mug))
    meshcat.SetObject("debug/grasp",
                      Box(0.03, 0.03, 0.03), Rgba(1, 0, 0, 0.4))
    meshcat.SetTransform("debug/grasp",
                         RigidTransform(R_WG_desired, p_grasp))

    # ------------------ Place-side keypoints (table) ------------------ #
    # Place a bit toward the robot from table center.
    p_place_base = p_WT.copy()
    p_place_xy = p_place_base[:2] + np.array([0.0, -0.25])  # towards robot

    safe_over_table_z = table_height + SAFE_CLEARANCE

    p_over_table = np.array([
        p_place_xy[0],
        p_place_xy[1],
        safe_over_table_z,
    ])
    p_place = np.array([
        p_place_xy[0],
        p_place_xy[1],
        table_height + PLACE_ABOVE_TABLE,
    ])
    p_retreat = p_over_table.copy()

    print("p_over_table:", p_over_table)
    print("p_place     :", p_place)

    # ------------------ Initial configuration ------------------ #
    if q_start is None:
        q_start = plant.GetPositions(plant_context, iiwa_model)

    # ------------------ IK for keyframes ------------------ #
    # Over mug (open, pointing down)
    q_over_m = ik_for_ee_position(
        plant, plant_context, iiwa_model, wsg_model,
        p_over_mug, R_WG_desired=R_WG_desired, theta_bound=1.0
    )
    # Descend to rim (still pointing down)
    q_descend = ik_for_ee_position(
        plant, plant_context, iiwa_model, wsg_model,
        p_grasp, R_WG_desired=R_WG_desired, theta_bound=1.0
    )
    # Lift mug (orientation relaxed)
    q_lift = ik_for_ee_position(
        plant, plant_context, iiwa_model, wsg_model,
        p_lift, R_WG_desired=None
    )
    # Move above table
    q_over_t = ik_for_ee_position(
        plant, plant_context, iiwa_model, wsg_model,
        p_over_table, R_WG_desired=None
    )
    # Place on table
    q_place = ik_for_ee_position(
        plant, plant_context, iiwa_model, wsg_model,
        p_place, R_WG_desired=None
    )
    # Retreat back up from table
    q_retreat = ik_for_ee_position(
        plant, plant_context, iiwa_model, wsg_model,
        p_retreat, R_WG_desired=None
    )

    # ------------------ WSG "teleport" joints ------------------ #
    left_joint = plant.GetJointByName("left_finger_sliding_joint", wsg_model)
    right_joint = plant.GetJointByName("right_finger_sliding_joint", wsg_model)

    def set_wsg_width(width: float):
        left_joint.set_translation(plant_context, +0.5 * width)
        right_joint.set_translation(plant_context, -0.5 * width)

    # ------------------ Keyframe script ------------------ #
    keyframes = [
        ("start",      q_start,   "open"),    # initial
        ("over_mug",   q_over_m,  "open"),    # approach above mug
        ("descend",    q_descend, "open"),    # descend to rim
        ("close",      q_descend, "closed"),  # close at grasp pose
        ("lift",       q_lift,    "closed"),  # lift mug
        ("over_table", q_over_t,  "closed"),  # move above table
        ("place",      q_place,   "open"),    # lower + open to place
        ("retreat",    q_retreat, "open"),    # retreat up
    ]

    # ------------------ Initialize ------------------ #
    plant.SetPositions(plant_context, iiwa_model, q_start)
    plant.SetVelocities(plant_context, iiwa_model, np.zeros_like(q_start))
    arm_controller.set_desired_position(q_start[-7:])  # joints only
    set_wsg_width(WSG_OPEN)
    diagram.ForcedPublish(context)

    t = context.get_time()

    def move_to(q_desired_full, gripper_mode: str):
        nonlocal t

        # joints-only target for PID
        q_joints_des = q_desired_full[-7:]
        arm_controller.set_desired_position(q_joints_des)

        # Gripper
        if gripper_mode == "open":
            set_wsg_width(WSG_OPEN)
        elif gripper_mode == "closed":
            set_wsg_width(WSG_CLOSED)

        simulator.AdvanceTo(t + PHASE_DT)
        t += PHASE_DT
        diagram.ForcedPublish(context)

    print("Executing full pick-and-place sequence...")
    for name, q, mode in keyframes:
        print(" ->", name, "mode:", mode)
        move_to(q, mode)



# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    meshcat = StartMeshcat()
    print("Meshcat URL:", meshcat.web_url())

    # Load scenario and directives
    with open(SCENARIO_PATH, "r") as f:
        scenario_yaml = f.read()
    scenario = LoadScenario(data=scenario_yaml)

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)

    parser = Parser(plant)
    # Make sure the "manipulation" package is available for mobile_iiwa, tray, etc.
    parser.package_map().Add(
        "manipulation",
        "/usr/local/lib/python3.12/dist-packages/manipulation/models",
    )

    # Build models from scenario.directives
    model_directives = ModelDirectives(directives=scenario.directives)
    ProcessModelDirectives(model_directives, parser)

    plant.Finalize()

    # Renderer + visualizer
    renderer_name = "renderer"
    scene_graph.AddRenderer(
        renderer_name, MakeRenderEngineVtk(RenderEngineVtkParams())
    )
    MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat, MeshcatVisualizerParams()
    )

    # Add cameras (this creates camera_point_cloud0/1/2 ports)
    add_cameras(builder, plant, scene_graph, scenario)

    # Get model instances
    iiwa_model = plant.GetModelInstanceByName("iiwa_arm")
    wsg_model = plant.GetModelInstanceByName("wsg_arm")

    # ------------------------------------------------------------------ #
    # Arm PID controller: only for 7 joints, not mobile base
    # PIDController expects 14-dim [q_joints(7), v_joints(7)] input,
    # and outputs 7 joint torques.
    # iiwa_arm_state is 20-dim = [q_base(3), q_joints(7), v_base(3), v_joints(7)].
    # We:
    #   - select joint state for PID input
    #   - expand joint torques back to 10 actuations (zeros for base)
    # ------------------------------------------------------------------ #
    kp_arm = 200.0
    kd_arm = 80.0
    ki_arm = 0.0


    q_des_initial = np.zeros(7)
    arm_controller = builder.AddSystem(
        PIDController(kp=kp_arm, kd=kd_arm, ki=ki_arm, q_desired=q_des_initial)
    )

    # 20-dim state → 14-dim [q_joints(7), v_joints(7)]
    G = np.zeros((14, 20))

    num_positions = plant.num_positions(iiwa_model)   # should be 10
    base_pos_dofs = 3                                 # x, y, z
    joint_pos_start = base_pos_dofs                   # 3
    joint_vel_start = num_positions + base_pos_dofs   # 10 + 3 = 13

    # q_joints (rows 0..6) from positions 3..9
    for i in range(7):
        G[i, joint_pos_start + i] = 1.0

    # v_joints (rows 7..13) from velocities 13..19
    for i in range(7):
        G[7 + i, joint_vel_start + i] = 1.0


    state_selector = builder.AddSystem(MatrixGain(G))

    builder.Connect(
        plant.get_state_output_port(iiwa_model),
        state_selector.get_input_port(),
    )
    builder.Connect(
        state_selector.get_output_port(),
        arm_controller.input_port,
    )

    # 7 joint torques -> 10 actuations (3 base + 7 joints)
    H = np.zeros((10, 7))
    for i in range(7):
        H[3 + i, i] = 1.0   # leave base torques = 0

    torque_expander = builder.AddSystem(MatrixGain(H))

    builder.Connect(
        arm_controller.output_port,
        torque_expander.get_input_port(),
    )
    builder.Connect(
        torque_expander.get_output_port(),
        plant.get_actuation_input_port(iiwa_model),
    )

    # WSG actuation: constant zeros (we'll kinematically set finger joints)
    wsg_source = builder.AddSystem(ConstantVectorSource([0.0, 0.0]))
    builder.Connect(
        wsg_source.get_output_port(),
        plant.get_actuation_input_port(wsg_model),
    )

    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    # Set PID desired joints to the initial configuration BEFORE starting sim,
    # so there is no big initial error.
    q_start_full = plant.GetPositions(plant_context, iiwa_model)
    arm_controller.set_desired_position(q_start_full[-7:])

    # Let things settle and the mug fall onto the table/tray:
    #  - Start sim
    #  - Run to t = 1.0s for mug to settle
    simulator.Initialize()
    simulator.AdvanceTo(1.0)
    diagram.ForcedPublish(context)

    # Build rim cloud and run ICP (after the mug has settled ~1s)
    rim_pc = build_rim_pointcloud(diagram, context, meshcat)
    X_WM_hat, mug_bottom_z, mug_top_z = estimate_mug_pose_icp(meshcat, rim_pc)

    print("Estimated mug pose:", X_WM_hat)
    print("mug_bottom_z:", mug_bottom_z, "mug_top_z:", mug_top_z)

    # Optional: compare against ground truth (if mug1 exists)
    try:
        mug_instance = plant.GetModelInstanceByName("mug1")
        mug_body = plant.GetBodyByName("base_link", mug_instance)
        X_WM_true = plant.EvalBodyPoseInWorld(plant_context, mug_body)
        print("True mug pose:", X_WM_true)
    except RuntimeError:
        pass

    # Run hover-only motion
    meshcat.StartRecording()
    run_pick_place(
        meshcat,
        simulator,
        diagram,
        plant,
        plant_context,
        iiwa_model,
        wsg_model,
        arm_controller,
        X_WM_hat,
        mug_bottom_z,
        mug_top_z,
        q_start_full,
    )
    meshcat.StopRecording()
    meshcat.PublishRecording()

    print("Done; keeping Meshcat open.")
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
