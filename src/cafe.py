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

from perception import add_cameras, get_depth
from pid_controller import PIDController, StaticPositionController


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

robot_base_instance = plant.GetModelInstanceByName("robot_base")
iiwa_arm_instance = plant.GetModelInstanceByName("iiwa_arm")
iiwa_plate_instance = plant.GetModelInstanceByName("iiwa_plate")
wsg_arm_instance = plant.GetModelInstanceByName("wsg_arm")
wsg_plate_instance = plant.GetModelInstanceByName("wsg_plate")

plant.Finalize()

renderer_name = "renderer"
scene_graph.AddRenderer(renderer_name, MakeRenderEngineVtk(RenderEngineVtkParams()))

visualizer = MeshcatVisualizer.AddToBuilder(
    builder, scene_graph, meshcat,
    MeshcatVisualizerParams()
)

add_cameras(builder, plant, scene_graph, scenario)

# Initial positions
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
    -1.0, 0.0, 0.0, 0.0,  # identity quaternion
    1.4, 0, 0.8         # position from scenario
])

robot_body_initial = RigidTransform(
    RotationMatrix(Quaternion(wxyz=initial_base_pose[:4])),
    initial_base_pose[4:]
)

# PID Controller gains for the arm (tuned for good performance)
kp_arm = 200.0
kd_arm = 50.0
ki_arm = 2.0

# Create PID controller for the arm
# You can change q_desired_arm to move the arm to different positions
q_desired_arm = initial_positions_arm  # Start at initial position
arm_controller = builder.AddSystem(
    PIDController(kp=kp_arm, kd=kd_arm, ki=ki_arm, q_desired=q_desired_arm)
)

# Connect arm controller
builder.Connect(
    plant.get_state_output_port(iiwa_arm_instance),
    arm_controller.input_port
)
builder.Connect(
    arm_controller.output_port,
    plant.get_actuation_input_port(iiwa_arm_instance)
)

# PID controller for the plate arm (needs integral term to compensate for gravity)
# Higher integral gain because it's fighting against the plate's weight
kp_plate = 300.0
kd_plate = 80.0
ki_plate = 55.0  # Higher Ki to compensate for steady gravitational load

plate_controller = builder.AddSystem(
    PIDController(kp=kp_plate, kd=kd_plate, ki=ki_plate, q_desired=initial_positions_plate)
)

# Connect plate controller
builder.Connect(
    plant.get_state_output_port(iiwa_plate_instance),
    plate_controller.input_port
)
builder.Connect(
    plate_controller.output_port,
    plant.get_actuation_input_port(iiwa_plate_instance)
)

# WSG gripper sources (constant for now)
wsg_arm_source = builder.AddSystem(ConstantVectorSource([0.1, 0.1]))  # open both fingers slightly
builder.Connect(
    wsg_arm_source.get_output_port(),
    plant.get_actuation_input_port(wsg_arm_instance)
)

wsg_plate_source = builder.AddSystem(ConstantVectorSource([0.0, 0.0]))  # grippers are gripping
builder.Connect(
    wsg_plate_source.get_output_port(),
    plant.get_actuation_input_port(wsg_plate_instance)
)

diagram = builder.Build()
simulator = Simulator(diagram)
context = simulator.get_mutable_context()
plant_context = plant.GetMyMutableContextFromRoot(context)

# Set initial conditions
robot_base_body = plant.GetBodyByName("robot_base_link", robot_base_instance)
X_initial = RigidTransform(
    RotationMatrix(Quaternion(wxyz=initial_base_pose[:4])),
    initial_base_pose[4:]
)
plant.SetFreeBodyPose(plant_context, robot_base_body, robot_body_initial)
zero_vel = SpatialVelocity(w=[0., 0., 0.], v=[0., 0., 0.])
plant.SetFreeBodySpatialVelocity(robot_base_body, zero_vel, plant_context)

plant.SetPositions(plant_context, iiwa_arm_instance, initial_positions_arm)
plant.SetPositions(plant_context, iiwa_plate_instance, initial_positions_plate)
# Set velocities to zero to start from rest
plant.SetVelocities(plant_context, iiwa_arm_instance, np.zeros(7))
plant.SetVelocities(plant_context, iiwa_plate_instance, np.zeros(7))

diagram.ForcedPublish(context)

tables = get_depth(diagram, context)

# Generating the pointcloud
camera0_point_cloud = diagram.GetOutputPort("camera_point_cloud0").Eval(context)
camera1_point_cloud = diagram.GetOutputPort("camera_point_cloud1").Eval(context)
camera2_point_cloud = diagram.GetOutputPort("camera_point_cloud2").Eval(context)
concatenated_pc = Concatenate([camera0_point_cloud, camera1_point_cloud, camera2_point_cloud])

voxel_size = 0.005  
downsampled_pc = concatenated_pc.VoxelizedDownSample(voxel_size)

def remove_table_points(point_cloud: PointCloud) -> PointCloud:
    xyz_points = point_cloud.xyzs()
    z_coordinates = xyz_points[2, :] 
    above_table_mask = z_coordinates > 1    
    keep_indices = np.where(above_table_mask)[0]    
    filtered_xyz = xyz_points[:, keep_indices]    
    filtered_point_cloud = PointCloud(filtered_xyz.shape[1])
    filtered_point_cloud.mutable_xyzs()[:] = filtered_xyz
    return filtered_point_cloud

letter_point_cloud = remove_table_points(downsampled_pc)

meshcat.SetObject(
    "letter_point_cloud", letter_point_cloud, point_size=0.05, rgba=Rgba(1, 0, 0)
)
diagram.ForcedPublish(context)

if running_as_notebook:
    simulator.set_target_realtime_rate(1.0)

# Monitor gripper contact forces
tau_contact = plant.get_generalized_contact_forces_output_port(wsg_arm_instance).Eval(plant_context)
total_force = np.sum(np.abs(tau_contact))
print("Total gripper contact force:", total_force)

print(f"Meshcat URL: {meshcat.web_url()}")
print("Running simulation...")
print(f"Arm starting at: {initial_positions_arm}")
print(f"Plate holding at: {initial_positions_plate}")

meshcat.StartRecording()
simulator.AdvanceTo(20.0)  # Run for 10 seconds to see the arm stabilize
meshcat.StopRecording()
meshcat.PublishRecording()

print("Simulation complete!")
print(f"Final arm position: {plant.GetPositions(plant_context, iiwa_arm_instance)}")
print(f"Final plate position: {plant.GetPositions(plant_context, iiwa_plate_instance)}")

# Example: How to change the arm's target position during runtime
# To use this, you would need to access the controller from the diagram
# and call: arm_controller.set_desired_position(new_target)