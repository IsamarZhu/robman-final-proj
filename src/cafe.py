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
from PIL import Image
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


# Save and crop the RGB image
topview_camera_rgb = station.GetOutputPort("topview_camera.rgb_image").Eval(station_context)
image_array = np.copy(topview_camera_rgb.data).reshape(
    (topview_camera_rgb.height(), topview_camera_rgb.width(), -1)
)

# Define crop region (x_start, y_start, x_end, y_end)
# Example: crop center 300x300 region
crop_x_start = 89  # left edge
crop_y_start = 0   # top edge
crop_x_end = 560    # right edge
crop_y_end = 480    # bottom edge

# Crop the RGB image
image_cropped = image_array[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

# Save full image
image_full = Image.fromarray(image_array.astype(np.uint8))
output_path_full = Path("/workspaces/robman-final-proj/topview_camera_image_full.png")
image_full.save(output_path_full)
print(f"Full image saved to {output_path_full} with shape {image_array.shape}")

# Save cropped image
image = Image.fromarray(image_cropped.astype(np.uint8))
output_path = Path("/workspaces/robman-final-proj/topview_camera_image.png")
image.save(output_path)
print(f"Cropped image saved to {output_path} with shape {image_cropped.shape}")

# Depth image processing with crop
topview_camera_depth = station.GetOutputPort("topview_camera.depth_image").Eval(station_context)
depth_array = np.copy(topview_camera_depth.data).reshape(
    (topview_camera_depth.height(), topview_camera_depth.width())
)

# Crop the depth array
depth_array_cropped = depth_array[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

# Debug: Print depth statistics for cropped region
print(f"Cropped depth array shape: {depth_array_cropped.shape}")
print(f"Depth min: {np.min(depth_array_cropped)}")
print(f"Depth max: {np.max(depth_array_cropped)}")
print(f"Number of inf values: {np.sum(np.isinf(depth_array_cropped))}")

# Normalize depth for visualization
depth_array_vis = np.copy(depth_array_cropped)
max_valid_depth = 15.0
depth_array_vis[np.isinf(depth_array_vis)] = max_valid_depth

depth_min = np.min(depth_array_vis)
depth_max = np.max(depth_array_vis)
print(f"Valid depth range: {depth_min} to {depth_max}")

if depth_max > depth_min:
    depth_normalized = 255 - ((depth_array_vis - depth_min) / (depth_max - depth_min) * 255)
else:
    depth_normalized = np.zeros_like(depth_array_vis)

depth_image = Image.fromarray(depth_normalized.astype(np.uint8))
depth_output_path = Path("/workspaces/robman-final-proj/topview_camera_depth.png")
depth_image.save(depth_output_path)
print(f"Cropped depth image saved to {depth_output_path}")

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
meshcat.SetObject(
    "letter_point_cloud", letter_point_cloud, point_size=0.05, rgba=Rgba(1, 0, 0)
)
diagram.ForcedPublish(context)

if running_as_notebook:
    simulator.set_target_realtime_rate(1.0)

meshcat.StartRecording()
# simulator.AdvanceTo(500.0)
time.sleep(30.0)
# meshcat.PublishRecording() #turning this on terminates or smthn, idk