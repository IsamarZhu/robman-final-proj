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
    Rgba,
    RigidTransform,
    RotationMatrix,
    InverseKinematics,
    Solve,
)
from pydrake.common.eigen_geometry import Quaternion
from pydrake.geometry import (
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    MakeRenderEngineVtk,
    RenderEngineVtkParams,
)
from pydrake.multibody.parsing import ProcessModelDirectives, ModelDirectives
from pydrake.systems.primitives import ConstantVectorSource

from manipulation import running_as_notebook
from manipulation.icp import IterativeClosestPoint
from manipulation.station import LoadScenario

# Your helper module
from perception import add_cameras, get_depth

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
SAVED_PC_PATH = "/workspaces/robman-final-proj/saved_pc.npy"
WAIT_BEFORE_MOVE = 1.0  # seconds at start between poses

# --------------------------------------------------------------------------- #
# Point cloud helpers
# --------------------------------------------------------------------------- #

def remove_table_points(point_cloud: PointCloud,
                        z_thresh: float = TABLE_Z_THRESHOLD) -> PointCloud:
    """Keep only points with z > z_thresh (throw away floor/table below)."""
    xyz = point_cloud.xyzs()
    z = xyz[2, :]
    mask = (z > z_thresh)
    keep_idx = np.where(mask)[0]

    filtered_xyz = xyz[:, keep_idx]
    pc = PointCloud(filtered_xyz.shape[1])
    pc.mutable_xyzs()[:] = filtered_xyz
    return pc


def keep_top_rim_band(pc: PointCloud,
                      band_thickness: float = 0.02) -> PointCloud:
    """
    Keep only the top band_thickness meters near the global max z.
    This gives you just the mug rims for a top-down grasp.
    """
    xyz = pc.xyzs()
    z = xyz[2, :]

    z_max = float(np.max(z))
    mask = (z > (z_max - band_thickness))

    idx = np.where(mask)[0]
    rim_xyz = xyz[:, idx]

    rim_pc = PointCloud(rim_xyz.shape[1])
    rim_pc.mutable_xyzs()[:] = rim_xyz
    return rim_pc

# --------------------------------------------------------------------------- #
# Scene (plant + scene_graph + cameras + visualizer)
# --------------------------------------------------------------------------- #

def build_scene(meshcat):
    """
    Build plant + scene_graph + cameras + visualizer using scenario.yaml.
    No MakeHardwareStation; just AddMultibodyPlantSceneGraph.
    """
    if running_as_notebook:
        mpld3.enable_notebook()

    scenario_yaml = SCENARIO_PATH.read_text()
    scenario = LoadScenario(data=scenario_yaml)

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    parser.package_map().Add(
        "manipulation",
        "/usr/local/lib/python3.12/dist-packages/manipulation/models",
    )

    # Load models via directives
    model_directives = ModelDirectives(directives=scenario.directives)
    ProcessModelDirectives(model_directives, parser)

    # Weld robot base like in your previous scripts
    robot_base_instance = plant.GetModelInstanceByName("robot_base")
    initial_base_pose = np.array([
        1.0, 0.0, 0.0, 0.0,  # identity quaternion
        1.4, 0.0, 0.8
    ])
    robot_body_initial = RigidTransform(
        RotationMatrix(Quaternion(wxyz=initial_base_pose[:4])),
        initial_base_pose[4:],
    )
    robot_base_body = plant.GetBodyByName("robot_base_link", robot_base_instance)
    plant.WeldFrames(
        plant.world_frame(),
        robot_base_body.body_frame(),
        robot_body_initial,
    )

    plant.Finalize()

    # Renderer + Meshcat visualizer
    renderer_name = "renderer"
    scene_graph.AddRenderer(renderer_name, MakeRenderEngineVtk(RenderEngineVtkParams()))
    MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat, MeshcatVisualizerParams()
    )

    # Add cameras (Rgbd + DepthImageToPointCloud) from scenario
    add_cameras(builder, plant, scene_graph, scenario)

    # Grab model instances
    iiwa_arm_instance   = plant.GetModelInstanceByName("iiwa_arm")
    iiwa_plate_instance = plant.GetModelInstanceByName("iiwa_plate")
    wsg_arm_instance    = plant.GetModelInstanceByName("wsg_arm")
    wsg_plate_instance  = plant.GetModelInstanceByName("wsg_plate")

    # Joint configs (same as in your scenario defaults)
    initial_positions_arm = [
        -1.57,
        0.1,
        0,
        -0.9,
        0,
        1.6,
        0,
    ]
    initial_positions_plate = [
        -1.57,
        -1.5,
        0,
        0.9,
        0,
        -1.2,
        0.3,
    ]

    # Simple "position-ish" actuation to hold the arms roughly in place
    position_arm_source = builder.AddSystem(ConstantVectorSource(initial_positions_arm))
    builder.Connect(
        position_arm_source.get_output_port(),
        plant.get_actuation_input_port(iiwa_arm_instance),
    )

    position_plate_source = builder.AddSystem(
        ConstantVectorSource(initial_positions_plate)
    )
    builder.Connect(
        position_plate_source.get_output_port(),
        plant.get_actuation_input_port(iiwa_plate_instance),
    )

    # Keep grippers roughly neutral torques
    wsg_arm_source = builder.AddSystem(ConstantVectorSource([0.0, 0.0]))
    builder.Connect(
        wsg_arm_source.get_output_port(),
        plant.get_actuation_input_port(wsg_arm_instance),
    )

    wsg_plate_source = builder.AddSystem(ConstantVectorSource([0.0, 0.0]))
    builder.Connect(
        wsg_plate_source.get_output_port(),
        plant.get_actuation_input_port(wsg_plate_instance),
    )

    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    # Make sure joints are at initial positions
    plant.SetPositions(plant_context, iiwa_arm_instance, initial_positions_arm)
    plant.SetPositions(plant_context, iiwa_plate_instance, initial_positions_plate)

    # Force one render for cameras/visuals
    diagram.ForcedPublish(context)

    return (
        diagram,
        simulator,
        context,
        plant,
        plant_context,
        iiwa_arm_instance,
        iiwa_plate_instance,
        wsg_arm_instance,
        wsg_plate_instance,
    )

# --------------------------------------------------------------------------- #
# Point cloud from cameras + ICP mug pose
# --------------------------------------------------------------------------- #

def build_rim_pointcloud(meshcat, diagram, context) -> PointCloud:
    """
    Build a rim-only point cloud using camera_point_cloud{0,1,2}.
    """
    # Optionally save RGB/depth images (your helper)
    get_depth(diagram, context)

    pc0 = diagram.GetOutputPort("camera_point_cloud0").Eval(context)
    pc1 = diagram.GetOutputPort("camera_point_cloud1").Eval(context)
    pc2 = diagram.GetOutputPort("camera_point_cloud2").Eval(context)

    concat_pc = Concatenate([pc0, pc1, pc2])
    down_pc = concat_pc.VoxelizedDownSample(VOXEL_SIZE)

    # Throw away big environment (floor/table) below some z
    obj_pc = remove_table_points(down_pc, z_thresh=TABLE_Z_THRESHOLD)

    # Keep only top band → mug rims
    obj_pc = keep_top_rim_band(obj_pc, band_thickness=0.02)

    # Save and visualize
    np.save(SAVED_PC_PATH, obj_pc.xyzs())
    print(f"Saved rim-only point cloud to {SAVED_PC_PATH}")

    meshcat.SetObject(
        "perception/mug_cloud",
        obj_pc,
        point_size=0.05,
        rgba=Rgba(1, 0, 0, 0.4),
    )
    diagram.ForcedPublish(context)

    return obj_pc


def estimate_mug_pose_icp(meshcat, object_pc: PointCloud) -> RigidTransform:
    """
    Run ICP between the mug mesh and the observed rim-only point cloud.
    Returns X_WM_hat (world-from-mug).
    """
    p_Ws = object_pc.xyzs()  # (3, M)

    # Model points in mug frame O
    if MUG_MESH_PATH.is_file():
        mug_mesh = trimesh.load(str(MUG_MESH_PATH), force="mesh")
        pts = mug_mesh.sample(N_SAMPLE_POINTS)  # (N, 3)
        p_Om = pts.T                             # (3, N)
        print(f"Loaded mug mesh from {MUG_MESH_PATH}")
    else:
        print(f"WARNING: {MUG_MESH_PATH} not found; using cylinder model.")
        np.random.seed(0)
        radius = 0.045
        height = 0.09
        theta = 2 * np.pi * np.random.rand(N_SAMPLE_POINTS)
        z = height * np.random.rand(N_SAMPLE_POINTS)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        p_Om = np.vstack([x, y, z])

    # Initial guess: near mug1's default pose from scenario.yaml
    initial_guess = RigidTransform(
        RotationMatrix.MakeZRotation(np.deg2rad(90.0)),
        [0.8, -0.1, 1.0],
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
    return X_WM_hat

# --------------------------------------------------------------------------- #
# IK + simple keyframe stepping on the SAME plant
# --------------------------------------------------------------------------- #

def run_pick_place_with_ik(
    meshcat,
    diagram,
    diagram_context,
    plant,
    plant_context,
    iiwa_arm_instance,
    wsg_arm_instance,
    X_WM_hat: RigidTransform,
    mug_top_z: float,
):
    """
    Use IK to plan keyframes and then step through them by directly
    setting joint positions and forcing publishes on the SAME plant.
    """

    world_frame = plant.world_frame()

    # Overwrite mug1 pose with ICP estimate
    mug_instance = plant.GetModelInstanceByName("mug1")
    mug_body = plant.GetBodyByName("base_link", mug_instance)
    plant.SetFreeBodyPose(plant_context, mug_body, X_WM_hat)
    diagram.ForcedPublish(diagram_context)

    # Frames
    ee_frame = plant.GetFrameByName("body", wsg_arm_instance)

    # Table0 (target, where we place)
    table_inst = plant.GetModelInstanceByName("table0")
    table_body = plant.GetBodyByName("table_body", table_inst)
    X_WT = plant.EvalBodyPoseInWorld(plant_context, table_body)
    p_WT = X_WT.translation()
    table_height = p_WT[2]

    # Initial config of free arm
    q_start = plant.GetPositions(plant_context, iiwa_arm_instance)

    # Object position from ICP
    p_WM = X_WM_hat.translation()

    # Top-down orientation: gripper -z aligned with world -z
    # This matches the weld where wsg "body" is rotated by [90, 0, 0]
    R_WG_desired = RotationMatrix.MakeXRotation(-np.pi / 2.0)

    # Estimate a "mug bottom" from the mug_top_z; we won't go below this
    mug_bottom_z = mug_top_z - 0.08  # assume ~8cm tall mug
    mug_bottom_z = max(mug_bottom_z, table_height + 0.01)

    def ik_for_ee_position(p_W_target, use_orientation: bool, theta_bound: float = 0.4):
        """
        Solve IK for a target EE position.
        Returns iiwa_arm joint positions, using full plant decision variable.
        """
        ik = InverseKinematics(plant, plant_context)
        q = ik.q()

        # Clamp z: never below mug_bottom_z
        p = p_W_target.copy()
        p[2] = max(p[2], mug_bottom_z + 0.01)  # margin above bottom / tray

        # Position constraint
        p_tol = 0.02  # 2 cm box
        ik.AddPositionConstraint(
            ee_frame, [0, 0, 0],
            world_frame,
            p - p_tol,
            p + p_tol,
        )

        # Optional orientation constraint
        if use_orientation:
            ik.AddOrientationConstraint(
                world_frame,
                R_WG_desired,
                ee_frame,
                RotationMatrix(),
                theta_bound,  # a bit looser (~23° if 0.4 rad)
            )

        prog = ik.prog()
        q_seed_full = plant.GetPositions(plant_context)
        prog.SetInitialGuess(q, q_seed_full)

        result = Solve(prog)
        if not result.is_success():
            raise RuntimeError(f"IK failed for target {p_W_target} (clamped to {p})")

        q_sol = result.GetSolution(q)

        # Write full solution back to plant context, then read just iiwa_arm
        plant.SetPositions(plant_context, q_sol)
        return plant.GetPositions(plant_context, iiwa_arm_instance)


    # -------------------------
    # Key points: TOP-DOWN pick
    # -------------------------
    z_over_mug = mug_top_z + 0.15   # hover above mug
    z_grasp    = mug_top_z - 0.02   # a little down into mug
    z_lift     = mug_top_z + 0.25   # lift higher

    p_over_mug = np.array([p_WM[0], p_WM[1], z_over_mug])
    p_grasp    = np.array([p_WM[0], p_WM[1], z_grasp])
    p_lift     = np.array([p_WM[0], p_WM[1], z_lift])

    # Place on table0, away from tray, toward robot
    table_height_for_place = table_height + 0.05
    p_place_xy = np.array([1.0, -0.05])

    p_over_table = np.array([p_place_xy[0], p_place_xy[1], table_height_for_place + 0.20])
    p_place      = np.array([p_place_xy[0], p_place_xy[1], table_height_for_place + 0.02])
    p_retreat    = p_over_table.copy()

    # Extra orientation waypoint slightly above mug
    p_orient = np.array([p_WM[0], p_WM[1], z_over_mug + 0.05])

    # Solve IK for each waypoint
    q_orient  = ik_for_ee_position(p_orient,      use_orientation=True)
    q_over_m  = ik_for_ee_position(p_over_mug,    use_orientation=True)
    q_grasp   = ik_for_ee_position(p_grasp,       use_orientation=True)
    q_lift    = ik_for_ee_position(p_lift,        use_orientation=True)
    q_over_t  = ik_for_ee_position(p_over_table,  use_orientation=True)
    q_place   = ik_for_ee_position(p_place,       use_orientation=True)
    q_retreat = ik_for_ee_position(p_retreat,     use_orientation=True)

    keyframes = [
        ("start",      q_start),
        ("orient",     q_orient),
        ("over_mug",   q_over_m),
        ("grasp",      q_grasp),
        ("lift",       q_lift),
        ("over_table", q_over_t),
        ("place",      q_place),
        ("retreat",    q_retreat),
    ]

    print("Stepping through keyframes...")
    t = 0.0
    dt = 1.0  # 1 second per pose

    # Explicit control of gripper finger joints (left/right slider)
    left_joint  = plant.GetJointByName("left_finger_sliding_joint",  wsg_arm_instance)
    right_joint = plant.GetJointByName("right_finger_sliding_joint", wsg_arm_instance)

    def set_wsg_width(width: float):
        """Set jaw opening by moving left/right sliders symmetrically."""
        left_joint.set_translation(plant_context,  +0.5 * width)
        right_joint.set_translation(plant_context, -0.5 * width)

    WSG_OPEN  = 0.10  # 10cm gap
    WSG_CLOSE = 0.03  # ~3cm gap

    # Start open
    set_wsg_width(WSG_OPEN)

    for name, q in keyframes:
        print(f"  -> {name}")
        plant.SetPositions(plant_context, iiwa_arm_instance, q)

        if name == "over_mug":
            # ensure open before going down
            set_wsg_width(WSG_OPEN)
        if name == "grasp":
            # close gripper
            set_wsg_width(WSG_CLOSE)
        if name == "place":
            # open to release
            set_wsg_width(WSG_OPEN)

        diagram_context.SetTime(t)
        diagram.ForcedPublish(diagram_context)
        t += dt
        time.sleep(dt)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    if running_as_notebook:
        mpld3.enable_notebook()

    meshcat = StartMeshcat()
    print("Meshcat URL:", meshcat.web_url())
    meshcat.StartRecording()

    # Build scene (single plant for perception + IK)
    (
        diagram,
        simulator,
        context,
        plant,
        plant_context,
        iiwa_arm_instance,
        iiwa_plate_instance,
        wsg_arm_instance,
        wsg_plate_instance,
    ) = build_scene(meshcat)

    # Point cloud and ICP
    object_pc = build_rim_pointcloud(meshcat, diagram, context)
    mug_top_z = float(np.max(object_pc.xyzs()[2, :]))
    print("Estimated mug_top_z from point cloud:", mug_top_z)

    X_WM_hat = estimate_mug_pose_icp(meshcat, object_pc)
    print("Estimated mug pose (X_WM_hat):")
    print(X_WM_hat)

    # IK + animation on the SAME diagram/plant
    run_pick_place_with_ik(
        meshcat,
        diagram,
        context,
        plant,
        plant_context,
        iiwa_arm_instance,
        wsg_arm_instance,
        X_WM_hat,
        mug_top_z,
    )
    simulator.AdvanceTo(5.0)
    meshcat.PublishRecording()
    input("Simulation done. Press Enter to exit...")
    print("Done.")
