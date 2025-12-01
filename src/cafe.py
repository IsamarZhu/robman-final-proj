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

from perception import add_cameras, get_depth, remove_table_points
from pid_controller import PIDController, StaticPositionController
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
initial_positions_arm = np.array([-1.1, 1.6, 0, -0.9, 0.6, 1.2, 0])
initial_positions_plate = np.array([-1.57, -1.5, 0, 0.9, 0, -1.2, 0.3])


# pid controller for the arm
kp_arm = 200.0
kd_arm = 50.0
ki_arm = 2.0
q_desired_arm = initial_positions_arm
arm_controller = builder.AddSystem(PIDController(kp=kp_arm, kd=kd_arm, ki=ki_arm, q_desired=q_desired_arm))
builder.Connect(plant.get_state_output_port(iiwa_arm_instance),arm_controller.input_port)
builder.Connect(arm_controller.output_port,plant.get_actuation_input_port(iiwa_arm_instance))

# pid controller for the plate arm (needs integral term to compensate for gravity + plate)
# kp_plate = 300.0
# kd_plate = 80.0
# ki_plate = 55.0 
# plate_controller = builder.AddSystem(PIDController(kp=kp_plate, kd=kd_plate, ki=ki_plate, q_desired=initial_positions_plate))
# # Connect plate controller
# builder.Connect(
#     plant.get_state_output_port(iiwa_plate_instance),
#     plate_controller.input_port
# )
# builder.Connect(
#     plate_controller.output_port,
#     plant.get_actuation_input_port(iiwa_plate_instance)
# )



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
tables = get_depth(diagram, context)

# generating the point cloud
camera0_point_cloud = diagram.GetOutputPort("camera_point_cloud0").Eval(context)
camera1_point_cloud = diagram.GetOutputPort("camera_point_cloud1").Eval(context)
camera2_point_cloud = diagram.GetOutputPort("camera_point_cloud2").Eval(context)
concatenated_pc = Concatenate([camera0_point_cloud, camera1_point_cloud, camera2_point_cloud])
voxel_size = 0.005  
downsampled_pc = concatenated_pc.VoxelizedDownSample(voxel_size)
letter_point_cloud = remove_table_points(downsampled_pc)
# meshcat.SetObject("letter_point_cloud", letter_point_cloud, point_size=0.05, rgba=Rgba(1, 0, 0))
diagram.ForcedPublish(context)

if running_as_notebook:
    simulator.set_target_realtime_rate(1.0)







# allowing the simulator to advance
# Monitor gripper contact forces
tau_contact = plant.get_generalized_contact_forces_output_port(wsg_arm_instance).Eval(plant_context)
total_force = np.sum(np.abs(tau_contact))
print("Total gripper contact force:", total_force)

print(f"Meshcat URL: {meshcat.web_url()}")
print("Running simulation...")
print(f"Arm starting at: {initial_positions_arm}")
print(f"Plate holding at: {initial_positions_plate}")

meshcat.StartRecording()

# # Option 1: Step through simulation and print at intervals
simulation_time = 5
print_interval = 0.5  # Print every 0.5 seconds
next_print_time = 0.0

while simulator.get_context().get_time() < simulation_time:
    # Advance to next print time
    simulator.AdvanceTo(min(next_print_time, simulation_time))
    
    # Get current contact forces
    tau_contact = plant.get_generalized_contact_forces_output_port(wsg_arm_instance).Eval(plant_context)
    total_force = np.sum(np.abs(tau_contact))
    
    current_time = simulator.get_context().get_time()
    print(f"Time: {current_time:.2f}s | Gripper contact force: {total_force:.4f}")
    
    next_print_time += print_interval

# print("Total gripper contact force:", total_force)
meshcat.StopRecording()
meshcat.PublishRecording()

# Example: How to change the arm's target position during runtime
# To use this, you would need to access the controller from the diagram
# and call: arm_controller.set_desired_position(new_target)