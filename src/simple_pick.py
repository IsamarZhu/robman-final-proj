"""
Simplified mug picking using general object segmentation + ICP pose estimation.
This approach works for any object (mug, gelatin box, etc.), not just mugs.
"""

from pathlib import Path
import numpy as np

from pydrake.all import (
    DiagramBuilder,
    InverseKinematics,
    Rgba,
    RigidTransform,
    RotationMatrix,
    Simulator,
    Solve,
    StartMeshcat,
    BasicVector,
    LeafSystem,
)

from pydrake.geometry import (
    MakeRenderEngineVtk,
    RenderEngineVtkParams,
    Box,
)

from pydrake.systems.primitives import ConstantVectorSource

from manipulation.station import (
    LoadScenario,
    MakeHardwareStation,
    AddPointClouds,
)

# Import from your segmentation module
from perception.segmentation import (
    build_pointcloud,
    segment_objects_clustering,
    ObjectDetector,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

SCENARIO_PATH = Path("/workspaces/robman-final-proj/src/new_scenario.yaml")

# Motion parameters
APPROACH_HEIGHT = 0.15  # Height above object to approach from (meters)
LIFT_HEIGHT = 0.20      # Height to lift after grasp (meters)
GRASP_OFFSET = 0.00     # Offset from top of object (0 = grasp at top)

# Gripper settings
WSG_OPEN = 0.107
WSG_CLOSED = 0.035

# Timing
MOVE_TIME = 2.5         # Time for each motion phase (seconds)
GRASP_TIME = 2.0        # Time to close gripper (seconds)

# Segmentation parameters
DBSCAN_EPS = 0.03       # DBSCAN clustering epsilon
DBSCAN_MIN_SAMPLES = 50 # DBSCAN min samples per cluster


# --------------------------------------------------------------------------- #
# Command Source Systems
# --------------------------------------------------------------------------- #

class JointPositionCommandSource(LeafSystem):
    """Outputs [q_desired, v_desired] for iiwa_arm.desired_state"""

    def __init__(self, q_initial: np.ndarray):
        super().__init__()
        q_initial = np.copy(q_initial).reshape(-1)
        self._nq = q_initial.shape[0]
        self._q_des = q_initial

        self.DeclareVectorOutputPort(
            "iiwa_desired_state",
            BasicVector(2 * self._nq),
            self._DoCalcOutput,
        )

    def _DoCalcOutput(self, context, output: BasicVector):
        v_des = np.zeros(self._nq)
        desired_state = np.concatenate([self._q_des, v_des])
        output.SetFromVector(desired_state)

    def set_q_desired(self, q_des: np.ndarray):
        q_des = np.copy(q_des).reshape(-1)
        assert q_des.shape[0] == self._nq
        self._q_des = q_des


class WsgCommandSource(LeafSystem):
    """Outputs desired gripper width"""

    def __init__(self, initial_width: float):
        super().__init__()
        self._width = float(initial_width)
        self.DeclareVectorOutputPort(
            "wsg_position",
            BasicVector(1),
            self._DoCalcOutput,
        )

    def _DoCalcOutput(self, context, output: BasicVector):
        output.SetFromVector([self._width])

    def set_width(self, width: float):
        self._width = float(width)


# --------------------------------------------------------------------------- #
# Object Detection & Grasp Computation
# --------------------------------------------------------------------------- #

def detect_and_locate_object(diagram, context, meshcat, target_object="mug"):
    """
    Detect objects in scene and locate the target object using segmentation + ICP.

    Returns:
        X_WO: RigidTransform of object pose in world frame
        grasp_center_xyz: (x, y, z) position for grasp center
        object_top_z: z-coordinate of object's top surface
    """

    print("\n=== OBJECT DETECTION ===")

    # 1. Build point cloud from cameras
    print("Building point cloud from cameras...")
    pc = build_pointcloud(diagram, context)
    print(f"Total points after filtering: {pc.xyzs().shape[1]}")

    # 2. Segment into individual objects using DBSCAN
    print(f"Segmenting objects (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES})...")
    object_clouds = segment_objects_clustering(pc, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    print(f"Found {len(object_clouds)} object clusters")

    if len(object_clouds) == 0:
        raise RuntimeError("No objects detected in scene!")

    # 3. Match each cluster to known objects using ICP
    print(f"\nMatching objects to templates...")
    detector = ObjectDetector()

    best_match = None
    best_match_score = float('inf')
    best_match_pose = None
    best_match_cloud = None

    # Visualize detected clusters
    colors = [Rgba(1, 0, 0), Rgba(0, 1, 0), Rgba(0, 0, 1), Rgba(1, 1, 0)]

    for i, obj_cloud in enumerate(object_clouds):
        # Visualize this cluster
        meshcat.SetObject(
            f"detection/cluster_{i}",
            obj_cloud,
            point_size=0.01,
            rgba=colors[i % len(colors)],
        )

        # Match against templates
        print(f"\nCluster {i}:")
        name, pose, score = detector.match_object(obj_cloud)
        print(f"  Best match: {name} (score: {score:.6f})")

        # Check if this is our target object and has best score
        if name == target_object and score < best_match_score:
            best_match = name
            best_match_score = score
            best_match_pose = pose
            best_match_cloud = obj_cloud
            print(f"  ✓ New best {target_object} match!")

    if best_match is None:
        raise RuntimeError(f"Could not find {target_object} in scene!")

    print(f"\n✓ Found {target_object} with score {best_match_score:.6f}")

    # 4. Compute grasp pose from ICP result
    X_WO = best_match_pose  # Object pose in world frame

    # Get the matched object's point cloud in world frame
    # The ICP gives us X_WO, so we can transform template points to world
    # But simpler: just use the observed point cloud to find top surface

    obj_xyz = best_match_cloud.xyzs()  # (3, N) in world frame

    # Find top of object
    object_top_z = float(np.max(obj_xyz[2, :]))

    # Find center in x-y plane (use centroid)
    center_x = float(np.mean(obj_xyz[0, :]))
    center_y = float(np.mean(obj_xyz[1, :]))

    # Grasp position: centered above object, at top surface + offset
    grasp_center_xyz = np.array([center_x, center_y, object_top_z + GRASP_OFFSET])

    print(f"\nGrasp computation:")
    print(f"  Object center (x,y): ({center_x:.3f}, {center_y:.3f})")
    print(f"  Object top z: {object_top_z:.3f}")
    print(f"  Grasp position: {grasp_center_xyz}")

    # Visualize grasp target
    meshcat.SetObject(
        "grasp/target",
        Box(0.02, 0.02, 0.02),
        Rgba(1, 0, 1, 0.8),
    )
    meshcat.SetTransform(
        "grasp/target",
        RigidTransform(grasp_center_xyz),
    )

    return X_WO, grasp_center_xyz, object_top_z


# --------------------------------------------------------------------------- #
# Inverse Kinematics
# --------------------------------------------------------------------------- #

def solve_ik(plant, plant_context, iiwa_model, wsg_model,
             p_W_target, R_WG_desired=None, position_tolerance=0.01,
             lock_base=False, theta_bound=0.2, base_positions_to_lock=None):
    """
    Solve IK for gripper to reach target position (and optionally orientation).

    Args:
        plant: MultibodyPlant
        plant_context: plant context
        iiwa_model: iiwa model instance
        wsg_model: wsg gripper model instance
        p_W_target: target position in world frame (3,)
        R_WG_desired: desired gripper orientation (RotationMatrix) or None
        position_tolerance: tolerance for position constraint (meters)
        lock_base: if True, lock mobile base joints
        theta_bound: orientation tolerance (radians)
        base_positions_to_lock: explicit base positions to lock (dict or None)

    Returns:
        q_iiwa: joint positions for iiwa (10-dof for mobile base + arm)
    """

    q_seed = plant.GetPositions(plant_context)

    ik = InverseKinematics(plant, plant_context)
    q_decision = ik.q()
    world_frame = plant.world_frame()
    ee_frame = plant.GetFrameByName("body", wsg_model)

    # Lock mobile base joints AND first arm joint (to keep tray stable)
    joints_to_lock = ["iiwa_base_x", "iiwa_base_y", "iiwa_base_z", "iiwa_joint_1"]
    locked_values = {}

    if lock_base:
        for joint_name in joints_to_lock:
            try:
                joint = plant.GetJointByName(joint_name, iiwa_model)
                idx = joint.position_start()

                # Use explicitly provided position if available, otherwise use current
                if base_positions_to_lock and joint_name in base_positions_to_lock:
                    q_val = base_positions_to_lock[joint_name]
                else:
                    q_val = q_seed[idx]

                locked_values[joint_name] = q_val

                ik.prog().AddBoundingBoxConstraint(
                    q_val, q_val, q_decision[idx:idx+1]
                )
            except RuntimeError:
                continue

    # Position constraint (small box around target)
    tol = position_tolerance
    lower = p_W_target - tol
    upper = p_W_target + tol

    ik.AddPositionConstraint(
        ee_frame, [0, 0, 0],
        world_frame,
        lower,
        upper,
    )

    # Orientation constraint if specified
    if R_WG_desired is not None:
        ik.AddOrientationConstraint(
            world_frame,
            R_WG_desired,
            ee_frame,
            RotationMatrix(),
            theta_bound,
        )

    prog = ik.prog()
    prog.SetInitialGuess(q_decision, q_seed)

    result = Solve(prog)
    if not result.is_success():
        # Print detailed error info
        print(f"\n!!! IK FAILED !!!")
        print(f"  Target position: {p_W_target}")
        print(f"  Position tolerance: ±{position_tolerance}m")
        print(f"  Orientation constraint: {R_WG_desired is not None}")
        print(f"  Base locked: {lock_base}")
        if R_WG_desired is not None:
            print(f"  Theta bound: {theta_bound} rad")
        raise RuntimeError(f"IK failed for target {p_W_target}")

    q_sol_full = result.GetSolution(q_decision)

    # DON'T modify plant_context - keep it clean
    # Instead, create a temporary context to extract iiwa positions
    temp_context = plant.CreateDefaultContext()
    plant.SetPositions(temp_context, q_sol_full)
    q_iiwa = plant.GetPositions(temp_context, iiwa_model)

    return np.copy(q_iiwa)


# --------------------------------------------------------------------------- #
# Pick Motion Sequence
# --------------------------------------------------------------------------- #

def pick_object(meshcat, simulator, diagram, plant, plant_context,
                iiwa_model, wsg_model, cmd_source, wsg_cmd_source,
                grasp_center_xyz, object_top_z):
    """
    Execute simple 4-step pick sequence:
    1. Approach position (above object)
    2. Descend to grasp
    3. Close gripper
    4. Lift up
    """

    print("\n=== PICK SEQUENCE ===")

    context = simulator.get_mutable_context()

    # Gripper orientation: pointing down
    R_WG_down = RotationMatrix.MakeXRotation(-np.pi / 2.0)

    # Key positions
    x, y, z_grasp = grasp_center_xyz

    p_approach = np.array([x, y, z_grasp + APPROACH_HEIGHT])
    p_grasp = np.array([x, y, z_grasp])
    p_lift = np.array([x, y, z_grasp + LIFT_HEIGHT])

    print(f"Approach:  {p_approach}")
    print(f"Grasp:     {p_grasp}")
    print(f"Lift:      {p_lift}")

    # Visualize trajectory
    meshcat.SetObject("traj/approach", Box(0.015, 0.015, 0.015), Rgba(0, 1, 0, 0.5))
    meshcat.SetTransform("traj/approach", RigidTransform(p_approach))

    meshcat.SetObject("traj/grasp", Box(0.015, 0.015, 0.015), Rgba(1, 0, 0, 0.8))
    meshcat.SetTransform("traj/grasp", RigidTransform(p_grasp))

    meshcat.SetObject("traj/lift", Box(0.015, 0.015, 0.015), Rgba(0, 0, 1, 0.5))
    meshcat.SetTransform("traj/lift", RigidTransform(p_lift))

    # Solve IK for key poses
    print("\nSolving IK for key poses...")

    # Get FULL plant state to extract joint positions correctly
    q_plant_full = plant.GetPositions(plant_context)

    # Get joints to lock from FULL plant state
    base_x_joint = plant.GetJointByName("iiwa_base_x", iiwa_model)
    base_y_joint = plant.GetJointByName("iiwa_base_y", iiwa_model)
    base_z_joint = plant.GetJointByName("iiwa_base_z", iiwa_model)
    joint_1 = plant.GetJointByName("iiwa_joint_1", iiwa_model)  # Lock first arm joint too!

    base_positions_lock = {
        "iiwa_base_x": q_plant_full[base_x_joint.position_start()],  # Use FULL plant state!
        "iiwa_base_y": q_plant_full[base_y_joint.position_start()],
        "iiwa_base_z": q_plant_full[base_z_joint.position_start()],
        "iiwa_joint_1": q_plant_full[joint_1.position_start()],      # Lock first joint to keep tray stable
    }

    print(f"Locking base at: x={base_positions_lock['iiwa_base_x']:.3f}, "
          f"y={base_positions_lock['iiwa_base_y']:.3f}, "
          f"z={base_positions_lock['iiwa_base_z']:.3f}")
    print(f"Locking joint_1 at: {base_positions_lock['iiwa_joint_1']:.3f} rad")

    # Now get iiwa start configuration for IK seed
    q_start = plant.GetPositions(plant_context, iiwa_model)

    # Approach: Base locked, relaxed constraints
    q_approach = solve_ik(plant, plant_context, iiwa_model, wsg_model,
                          p_approach, R_WG_down,
                          position_tolerance=0.02,
                          lock_base=True,
                          theta_bound=0.3,
                          base_positions_to_lock=base_positions_lock)
    print("  ✓ Approach pose")

    # Grasp: Base locked, moderate constraints for accuracy
    q_grasp = solve_ik(plant, plant_context, iiwa_model, wsg_model,
                       p_grasp, R_WG_down,
                       position_tolerance=0.02,
                       lock_base=True,
                       theta_bound=0.3,
                       base_positions_to_lock=base_positions_lock)
    print("  ✓ Grasp pose")

    # Lift: Base locked, moderate tolerance
    q_lift = solve_ik(plant, plant_context, iiwa_model, wsg_model,
                      p_lift, R_WG_down,
                      position_tolerance=0.08,
                      lock_base=True,
                      theta_bound=0.8,
                      base_positions_to_lock=base_positions_lock)
    print("  ✓ Lift pose")

    # Motion helper
    def move_to(q_des, gripper_width, duration):
        """Move robot to joint configuration with specified gripper width"""
        # Debug: Print base positions
        base_x_idx = plant.GetJointByName("iiwa_base_x", iiwa_model).position_start()
        base_y_idx = plant.GetJointByName("iiwa_base_y", iiwa_model).position_start()
        base_z_idx = plant.GetJointByName("iiwa_base_z", iiwa_model).position_start()

        print(f"  Commanding base: x={q_des[base_x_idx]:.3f}, y={q_des[base_y_idx]:.3f}, z={q_des[base_z_idx]:.3f}")

        cmd_source.set_q_desired(q_des)
        wsg_cmd_source.set_width(gripper_width)
        t = simulator.get_context().get_time()
        simulator.AdvanceTo(t + duration)
        diagram.ForcedPublish(context)

    # Initialize
    plant.SetPositions(plant_context, iiwa_model, q_start)
    plant.SetVelocities(plant_context, iiwa_model, np.zeros_like(q_start))
    cmd_source.set_q_desired(q_start)
    wsg_cmd_source.set_width(WSG_OPEN)
    diagram.ForcedPublish(context)

    # Execute pick sequence
    print("\nExecuting motion...")

    print("1. Move to start position")
    move_to(q_start, WSG_OPEN, MOVE_TIME)

    print("2. Approach above object")
    move_to(q_approach, WSG_OPEN, MOVE_TIME)

    print("3. Descend to grasp position")
    move_to(q_grasp, WSG_OPEN, MOVE_TIME)

    print("4. Close gripper to grasp")
    move_to(q_grasp, WSG_CLOSED, GRASP_TIME)

    print("5. Lift object")
    move_to(q_lift, WSG_CLOSED, MOVE_TIME)

    print("\n✓ Pick sequence complete!")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    print("=" * 70)
    print("SIMPLIFIED OBJECT PICKING - Using Segmentation + ICP")
    print("=" * 70)

    meshcat = StartMeshcat()
    print(f"\nMeshcat: {meshcat.web_url()}")

    # Load scenario
    print("\nLoading scenario...")
    with open(SCENARIO_PATH, "r") as f:
        scenario_yaml = f.read()
    scenario = LoadScenario(data=scenario_yaml)

    # Build hardware station
    print("Building hardware station...")
    builder = DiagramBuilder()
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat=meshcat))
    plant = station.GetSubsystemByName("plant")
    scene_graph = station.GetSubsystemByName("scene_graph")

    # Add renderer
    renderer_name = "renderer"
    if not scene_graph.HasRenderer(renderer_name):
        scene_graph.AddRenderer(
            renderer_name, MakeRenderEngineVtk(RenderEngineVtkParams())
        )

    # Get robot models
    iiwa_model = plant.GetModelInstanceByName("iiwa_arm")
    wsg_model = plant.GetModelInstanceByName("wsg_arm")

    # Create command sources
    default_context = plant.CreateDefaultContext()
    q_start = plant.GetPositions(default_context, iiwa_model)

    cmd_source = builder.AddSystem(JointPositionCommandSource(q_start))
    builder.Connect(
        cmd_source.get_output_port(),
        station.GetInputPort("iiwa_arm.desired_state"),
    )

    wsg_cmd_source = builder.AddSystem(WsgCommandSource(WSG_OPEN))
    builder.Connect(
        wsg_cmd_source.get_output_port(),
        station.GetInputPort("wsg_arm.position"),
    )

    wsg_force_limit = builder.AddSystem(ConstantVectorSource([40.0]))
    builder.Connect(
        wsg_force_limit.get_output_port(),
        station.GetInputPort("wsg_arm.force_limit"),
    )

    # Add point cloud cameras
    print("Setting up cameras...")
    to_point_cloud = AddPointClouds(
        scenario=scenario,
        station=station,
        builder=builder,
        meshcat=None,  # Don't visualize raw clouds
    )

    for name, system in to_point_cloud.items():
        builder.ExportOutput(
            system.GetOutputPort("point_cloud"),
            f"{name}.point_cloud",
        )

    # Build and initialize simulator
    print("Building diagram...")
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    # Let scene settle
    print("\nLetting scene settle...")
    simulator.Initialize()
    simulator.AdvanceTo(1.0)
    diagram.ForcedPublish(context)
    print("✓ Scene ready")

    # Detect object and compute grasp
    X_WO, grasp_center_xyz, object_top_z = detect_and_locate_object(
        diagram, context, meshcat, target_object="mug"
    )

    # Start recording
    meshcat.StartRecording()

    # Execute pick
    pick_object(
        meshcat, simulator, diagram, plant, plant_context,
        iiwa_model, wsg_model, cmd_source, wsg_cmd_source,
        grasp_center_xyz, object_top_z
    )

    # Finish recording
    meshcat.StopRecording()
    meshcat.PublishRecording()

    print("\n" + "=" * 70)
    print("DONE! Check Meshcat for visualization.")
    print("=" * 70)
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
