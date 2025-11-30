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

initial_positions_arm = [
    -1.57,  # joint 1
    0.9,    # joint 2
    0,      # joint 3
    -0.9,   # joint 4
    0,      # joint 5
    1.6,    # joint 6
    0       # joint 7
]
initial_positions_plate = [
    -1.57,  # joint 1
    -1.5,    # joint 2
    0,      # joint 3
    0.9,   # joint 4
    0,      # joint 5
    -1.2,    # joint 6
    0.3       # joint 7
]

initial_base_pose = np.array([
    -1.0, 0.0, 0.0, 0.0,  # identity quaternion
    -3.5, 3.5, 0.4         # position from scenario
])

robot_body_initial = RigidTransform(
    RotationMatrix(Quaternion(wxyz=initial_base_pose[:4])),
    initial_base_pose[4:]
)

# currently making both arms be held to the same initial position
position_arm_source = builder.AddSystem(ConstantVectorSource(initial_positions_arm))
builder.Connect(
    position_arm_source.get_output_port(),
    plant.get_actuation_input_port(iiwa_arm_instance)

)
position_plate_source = builder.AddSystem(ConstantVectorSource(initial_positions_plate))
builder.Connect(
    position_plate_source.get_output_port(),
    plant.get_actuation_input_port(iiwa_plate_instance)
)

wsg_arm_source = builder.AddSystem(ConstantVectorSource([0.1, 0.1])) # open both fingers slightly
builder.Connect(
    wsg_arm_source.get_output_port(),
    plant.get_actuation_input_port(wsg_arm_instance)
)

wsg_plate_source = builder.AddSystem(ConstantVectorSource([0.0, 0.0])) # the grippers are gripping
builder.Connect(
    wsg_plate_source.get_output_port(),
    plant.get_actuation_input_port(wsg_plate_instance)
)

diagram = builder.Build()
simulator = Simulator(diagram)
context = simulator.get_mutable_context()
plant_context = plant.GetMyMutableContextFromRoot(context)

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

diagram.ForcedPublish(context)

tables = get_depth(diagram, context)


# generating the pointcloud
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
# izzy this is what's generating the red section of what the pointclouds are seeing, the tan is what got filtered out
meshcat.SetObject(
    "letter_point_cloud", letter_point_cloud, point_size=0.05, rgba=Rgba(1, 0, 0)
)
diagram.ForcedPublish(context)

if running_as_notebook:
    simulator.set_target_realtime_rate(1.0)

meshcat.StartRecording()
# simulator.AdvanceTo(500.0)
time.sleep(10.0)
# meshcat.PublishRecording() #turning this on terminates or smthn, idk