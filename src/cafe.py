from pathlib import Path
import time
import mpld3
import numpy as np
from pydrake.all import (
    DiagramBuilder,
    Concatenate,
    PointCloud,
    Simulator,
    StartMeshcat,
    Rgba
)

from manipulation import running_as_notebook
from manipulation.station import LoadScenario, MakeHardwareStation, AddPointClouds


if running_as_notebook:
    mpld3.enable_notebook()

meshcat = StartMeshcat()
scenario_file = Path("/workspaces/robman-final-proj/src/scenario.yaml")

with open(scenario_file, "r") as f:
    scenario_yaml = f.read()

scenario = LoadScenario(data=scenario_yaml)

builder = DiagramBuilder()
station = MakeHardwareStation(scenario, meshcat=meshcat)
# Register the station with the builder before adding point-cloud systems so
# that AddPointClouds can connect to station's ports.
builder.AddSystem(station)
to_point_cloud = AddPointClouds(scenario=scenario, station=station, builder=builder, meshcat=None) # izzy make meshcat=meschat equal to 
builder.ExportOutput(to_point_cloud["camera0"].get_output_port(), "camera_point_cloud0")
builder.ExportOutput(to_point_cloud["camera1"].get_output_port(), "camera_point_cloud1")
builder.ExportOutput(to_point_cloud["camera2"].get_output_port(), "camera_point_cloud2")
plant = station.GetSubsystemByName("plant")



from pydrake.systems.primitives import ConstantVectorSource
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

# currently making both arms be held to the same initial position
position_arm_source = builder.AddSystem(ConstantVectorSource(initial_positions_arm))
builder.Connect(
    position_arm_source.get_output_port(),
    station.GetInputPort("iiwa_arm.position")
)
position_plate_source = builder.AddSystem(ConstantVectorSource(initial_positions_plate))
builder.Connect(
    position_plate_source.get_output_port(),
    station.GetInputPort("iiwa_plate.position")
)

wsg_arm_source = builder.AddSystem(ConstantVectorSource([0.1]))
builder.Connect(
    wsg_arm_source.get_output_port(),
    station.GetInputPort("wsg_arm.position")
)
wsg_plate_source = builder.AddSystem(ConstantVectorSource([0]))
builder.Connect(
    wsg_plate_source.get_output_port(),
    station.GetInputPort("wsg_plate.position")
)

diagram = builder.Build()
simulator = Simulator(diagram)
context = simulator.get_mutable_context()
station_context = station.GetMyContextFromRoot(context)


# generating the pointcloud
camera0_letter_point_cloud = diagram.GetOutputPort("camera_point_cloud0").Eval(context)
camera1_letter_point_cloud = diagram.GetOutputPort("camera_point_cloud1").Eval(context)
camera2_letter_point_cloud = diagram.GetOutputPort("camera_point_cloud2").Eval(context)
concatenated_pc = Concatenate([camera0_letter_point_cloud, camera1_letter_point_cloud, camera2_letter_point_cloud])

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
# meshcat.SetObject(
#     "letter_point_cloud", letter_point_cloud, point_size=0.05, rgba=Rgba(1, 0, 0)
# )
diagram.ForcedPublish(context)

if running_as_notebook:
    simulator.set_target_realtime_rate(1.0)

meshcat.StartRecording()
# simulator.AdvanceTo(500.0)
time.sleep(30.0)
# meshcat.PublishRecording() #turning this on terminates or smthn, idk