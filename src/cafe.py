from pathlib import Path
import time
import mpld3
import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Parser,
    DiagramBuilder,
    Concatenate,
    PointCloud,
    Simulator,
    InverseKinematics,
    Solve,
    StartMeshcat,
    SpatialVelocity,
    Rgba,
    RigidTransform,
    RotationMatrix
)
from pydrake.common.eigen_geometry import Quaternion
from pydrake.systems.primitives import ConstantVectorSource
from pydrake.geometry import MeshcatVisualizer, MeshcatVisualizerParams, MakeRenderEngineVtk, RenderEngineVtkParams

from pydrake.multibody.parsing import ProcessModelDirectives, ModelDirectives
from manipulation import running_as_notebook
from manipulation.station import LoadScenario

from perception.top_level_perception import add_cameras, perceive_tables, remove_table_points
from pid_controller import PIDController
from pydrake.trajectories import PiecewisePolynomial
from pydrake.systems.primitives import TrajectorySource


# starting the meshcat and intializing the scenario information
if running_as_notebook: 
    mpld3.enable_notebook()
meshcat = StartMeshcat()
scenario_file = Path("/workspaces/robman-final-proj/src/scenario.yaml")
with open(scenario_file, "r") as f: 
    scenario_yaml = f.read()
scenario = LoadScenario(data=scenario_yaml)
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
parser = Parser(plant)
parser.package_map().Add("manipulation", "/usr/local/lib/python3.12/dist-packages/manipulation/models")
model_directives = ModelDirectives(directives=scenario.directives)
models = ProcessModelDirectives(model_directives, parser)


# obtaining all the model instance values
robot_base_instance = plant.GetModelInstanceByName("robot_base")
iiwa_arm_instance = plant.GetModelInstanceByName("iiwa_arm")
iiwa_plate_instance = plant.GetModelInstanceByName("iiwa_plate")
wsg_arm_instance = plant.GetModelInstanceByName("wsg_arm")
wsg_plate_instance = plant.GetModelInstanceByName("wsg_plate")
plant.Finalize()


# adding visualizers, rendering, and cameras
renderer_name = "renderer"
scene_graph.AddRenderer(renderer_name, MakeRenderEngineVtk(RenderEngineVtkParams()))
visualizer = MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, MeshcatVisualizerParams())
add_cameras(builder, plant, scene_graph, scenario)


# setting intial positions for the robot arms
initial_positions_arm = np.array([-1.2, 1.6, 0, -0.9, 0.6, 1.7, 0])
initial_positions_plate = np.array([-1.57, -1.5, 0, 0.9, 0, -1.2, 0.3])


# pid controller for the arm
kp_arm = 200.0
kd_arm = 50.0
ki_arm = 2.0
q_desired_arm = initial_positions_arm
arm_controller = builder.AddSystem(PIDController(kp=kp_arm, kd=kd_arm, ki=ki_arm, q_desired=q_desired_arm))
builder.Connect(plant.get_state_output_port(iiwa_arm_instance),arm_controller.input_port)
builder.Connect(arm_controller.output_port,plant.get_actuation_input_port(iiwa_arm_instance))




# buildling the diagram
diagram = builder.Build()
simulator = Simulator(diagram)
context = simulator.get_mutable_context()
plant_context = plant.GetMyMutableContextFromRoot(context)


# setting the initial pose and velocities of the robot
initial_base_pose = np.array([
    -1.0, 0.0, 0.0, 0.0,  # identity quaternion
    1.4, 0, 0.8         # position from scenario
])
robot_body_initial = RigidTransform(
    RotationMatrix(Quaternion(wxyz=initial_base_pose[:4])),
    initial_base_pose[4:]
)
robot_base_body = plant.GetBodyByName("robot_base_link", robot_base_instance)
plant.SetFreeBodyPose(plant_context, robot_base_body, robot_body_initial)
zero_vel = SpatialVelocity(w=[0., 0., 0.], v=[0., 0., 0.])
plant.SetFreeBodySpatialVelocity(robot_base_body, zero_vel, plant_context)
plant.SetPositions(plant_context, iiwa_arm_instance, initial_positions_arm)
plant.SetPositions(plant_context, iiwa_plate_instance, initial_positions_plate)
plant.SetVelocities(plant_context, iiwa_arm_instance, np.zeros(7))
plant.SetVelocities(plant_context, iiwa_plate_instance, np.zeros(7))
# plant.SetPositions(plant_context, wsg_arm_instance, np.array([opened, opened]))

for i in range(1, 8):
    joint = plant.GetJointByName(f"iiwa_joint_{i}", iiwa_plate_instance)
    joint.Lock(plant_context)
    
diagram.ForcedPublish(context)
tables = perceive_tables(diagram, context)

if running_as_notebook:
    simulator.set_target_realtime_rate(1.0)






######################## icp ########################
from perception.object_segmentation import build_rim_pointcloud, estimate_mug_pose_icp
rim_pc = build_rim_pointcloud(diagram, context)
# meshcat.SetObject("rim_pc", rim_pc, point_size=0.05, rgba=Rgba(1, 0, 0, 0.4))
X_WM_hat, (mug_bottom_z, mug_top_z) = estimate_mug_pose_icp(meshcat, rim_pc)
print("Estimated mug pose:", X_WM_hat)
# print("Estimated mug_bottom_z:", mug_bottom_z)
# print("Estimated mug_top_z:", mug_top_z)

# (Optional) compare to ground truth mug pose as in step 1


# --- Debug: compare ICP pose to ground truth mug1 pose ---
# mug1_instance = plant.GetModelInstanceByName("mug1")
# mug1_body = plant.GetBodyByName("base_link", mug1_instance)
# X_WM_true = plant.EvalBodyPoseInWorld(plant_context, mug1_body)
# # print("Ground truth mug1 pose:", X_WM_true)
# print("ICP translation:", X_WM_hat.translation())
# print("True translation:", X_WM_true.translation())
# # diagram.ForcedPublish(context)


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

    z_over_mug = max(mug_top_z + 0.18, safe_fly_height)
    z_grasp    = mug_top_z + 0.02   # just above rim, not in the mug
    z_lift     = max(mug_top_z + 0.25, safe_fly_height)



    p_over_mug = np.array([p_WM[0]-0.02, p_WM[1] + 0.02, z_over_mug])
    p_grasp    = np.array([p_over_mug[0], p_over_mug[1], p_over_mug[2]-0.1])
    # p_lift     = np.array([p_WM[0], p_WM[1], z_lift])


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
    q_grasp = ik_for_ee_position(
        p_grasp,
        use_orientation=False,  # <- important
        clamp_to_tray=True,
    )
    
    keyframes = [
        ("start",     q_start,   "open"),
        ("over_mug",  q_over_m,  "open"),
        ("grasp",     q_grasp,   "default"),
        ("grasp",     q_grasp,   "close"),
        # ("lift",      q_lift,    "closed"),
        # ("over_table",q_over_t,  "closed"),
        # ("place",     q_place,   "open"),
        # ("retreat",   q_retreat, "open"),
    ]

    # ----------------------------------------------------------------------- #
    # Initialize arm + gripper
    # ----------------------------------------------------------------------- #
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
        elif gripper_mode == "default":
            pass

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
        
q_start = initial_positions_arm.copy()
meshcat.StartRecording()

######################## icp ########################



# allowing the simulator to advance
# Monitor gripper contact forces
# tau_contact = plant.get_generalized_contact_forces_output_port(wsg_arm_instance).Eval(plant_context)
# total_force = np.sum(np.abs(tau_contact))
# print("Total gripper contact force:", total_force)

    
meshcat.StopRecording()
meshcat.PublishRecording()