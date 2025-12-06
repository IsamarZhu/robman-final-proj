from pydrake.systems.sensors import RgbdSensor, CameraInfo, PixelType
from pydrake.geometry import DepthRenderCamera, RenderCameraCore, ColorRenderCamera, ClippingRange, DepthRange
from pydrake.perception import DepthImageToPointCloud
from pydrake.math import RigidTransform
from pydrake.all import PointCloud
import numpy as np
from perception.obstacle_detection import detect_obstacles_from_img
from perception.table_detection import detect_tables_from_img
from perception.config import CROP_X_START, CROP_X_END, CROP_Y_START, CROP_Y_END


def remove_table_points(point_cloud: PointCloud) -> PointCloud:
    xyz_points = point_cloud.xyzs()
    z_coordinates = xyz_points[2, :] 
    above_table_mask = z_coordinates > 1    
    keep_indices = np.where(above_table_mask)[0]    
    filtered_xyz = xyz_points[:, keep_indices]    
    filtered_point_cloud = PointCloud(filtered_xyz.shape[1])
    filtered_point_cloud.mutable_xyzs()[:] = filtered_xyz
    return filtered_point_cloud

def add_cameras(builder, plant, scene_graph, scenario):
    rgbd_sensors = {}
    point_cloud_systems = {}

    for cam_name, cam_config in scenario.cameras.items():
        try:
            # Get the camera body from the plant
            cam_instance = plant.GetModelInstanceByName(cam_config.name)
            cam_body = plant.GetBodyByName("base", cam_instance)
            frame_id = plant.GetBodyFrameIdOrThrow(cam_body.index())
            
            # Create camera info from scenario configuration
            camera_info = CameraInfo(
                width=cam_config.width,
                height=cam_config.height,
                fov_y=cam_config.focal.y * np.pi / 180
            )
            
            # Create render cameras
            core = RenderCameraCore(
                "renderer", camera_info,
                ClippingRange(cam_config.z_near, cam_config.z_far),
                RigidTransform()
            )
            color_camera = ColorRenderCamera(core, show_window=False)
            depth_camera = DepthRenderCamera(core, DepthRange(cam_config.z_near, cam_config.z_far))
            
            # Create RGBD sensor
            sensor = builder.AddSystem(
                RgbdSensor(
                    parent_id=frame_id,
                    X_PB=RigidTransform(),
                    color_camera=color_camera,
                    depth_camera=depth_camera
                )
            )
            
            builder.Connect(
                scene_graph.get_query_output_port(),
                sensor.query_object_input_port()
            )
            
            builder.ExportOutput(sensor.color_image_output_port(), f"{cam_name}.rgb_image")
            builder.ExportOutput(sensor.depth_image_32F_output_port(), f"{cam_name}.depth_image")
            
            rgbd_sensors[cam_name] = sensor
            
            if cam_name in ["camera0", "camera1", "camera2"]:
                depth_to_cloud = builder.AddSystem(
                    DepthImageToPointCloud(
                        camera_info=camera_info,
                        pixel_type=PixelType.kDepth32F
                    )
                )
                
                builder.Connect(
                    sensor.depth_image_32F_output_port(),
                    depth_to_cloud.depth_image_input_port()
                )
                
                builder.Connect(
                    sensor.body_pose_in_world_output_port(),
                    depth_to_cloud.camera_pose_input_port()
                )
                
                cam_num = cam_name[-1]  # Get the number from "camera0", "camera1", etc.
                builder.ExportOutput(
                    depth_to_cloud.point_cloud_output_port(),
                    f"camera_point_cloud{cam_num}"
                )
                
                point_cloud_systems[cam_name] = depth_to_cloud
            
        except Exception as e:
            print(f"  Warning: Could not add camera {cam_name}: {e}")

def pixel_to_world_topview(pixel_x, pixel_y, depth_value, camera_z=12.0, camera_fov_y=np.pi/4):
    """
    pixel coords to world coords for topview camera
    """
    actual_pixel_x = pixel_x + CROP_X_START
    actual_pixel_y = pixel_y + CROP_Y_START
    
    full_img_width = 640 
    full_img_height = 480 
    
    focal_length_y = (full_img_height / 2.0) / np.tan(camera_fov_y / 2.0)
    focal_length_x = focal_length_y 
    
    cx = full_img_width / 2.0
    cy = full_img_height / 2.0
    
    x_cam = (actual_pixel_x - cx) * depth_value / focal_length_x
    y_cam = (actual_pixel_y - cy) * depth_value / focal_length_y
    z_cam = depth_value
    
    world_x = y_cam
    world_y = x_cam  
    world_z = camera_z - z_cam
    
    return (world_x, world_y, world_z)

def perceive_scene(station, station_context):
    """
    Perceive tables from the topview camera depth image.
    
    Args:
        station: The hardware station system
        station_context: The context for the station
    
    Returns:
        List of table dictionaries with both pixel and world coordinates
        List of obstacle dictionaries with both pixel and world coordinates
    """
    topview_camera_depth = station.GetOutputPort("topview_camera.depth_image").Eval(station_context)
    depth_array = np.copy(topview_camera_depth.data).reshape(
        (topview_camera_depth.height(), topview_camera_depth.width())
    )
    depth_array_cropped = depth_array[CROP_Y_START:CROP_Y_END, CROP_X_START:CROP_X_END]

    tables = detect_tables_from_img(depth_array_cropped)
    obstacles = detect_obstacles_from_img(depth_array_cropped)
    
    for table in tables:
        cx_pixel, cy_pixel = table['center']
        center_depth = depth_array_cropped[int(cy_pixel), int(cx_pixel)]
        table['center_world'] = pixel_to_world_topview(cx_pixel, cy_pixel, center_depth)
        w, h = table['size']
        table['corner_world'] = []
        for corner in table['box_corners']:
            # Clamp coordinates to valid range
            corner_x = int(np.clip(corner[0], 0, depth_array_cropped.shape[1] - 1))
            corner_y = int(np.clip(corner[1], 0, depth_array_cropped.shape[0] - 1))
            corner_depth = depth_array_cropped[corner_y, corner_x]
            corner_world = pixel_to_world_topview(corner[0], corner[1], corner_depth)
            table['corner_world'].append(corner_world)
        
        # Calculate waypoint world coordinates (midpoints between corners)
        table['waypoints_world'] = []
        corners_world = table['corner_world']
        for i in range(len(corners_world)):
            next_i = (i + 1) % len(corners_world)
            midpoint = (
                (corners_world[i][0] + corners_world[next_i][0]) / 2,
                (corners_world[i][1] + corners_world[next_i][1]) / 2,
                (corners_world[i][2] + corners_world[next_i][2]) / 2
            )
            table['waypoints_world'].append(midpoint)
        
        if w > h: 
            world_angle = np.pi / 2
        else: 
            world_angle = 0.0
        
        table['angle_radians'] = world_angle
    
    for obstacle in obstacles:
        cx_pixel, cy_pixel = obstacle['center']
        center_depth = depth_array_cropped[int(cy_pixel), int(cx_pixel)]
        obstacle['center_world'] = pixel_to_world_topview(cx_pixel, cy_pixel, center_depth)
        
        # Convert corners to world coordinates
        obstacle['corner_world'] = []
        for corner in obstacle['box_corners']:
            # Clamp coordinates to valid range
            corner_x = int(np.clip(corner[0], 0, depth_array_cropped.shape[1] - 1))
            corner_y = int(np.clip(corner[1], 0, depth_array_cropped.shape[0] - 1))
            corner_depth = depth_array_cropped[corner_y, corner_x]
            corner_world = pixel_to_world_topview(corner[0], corner[1], corner_depth)
            obstacle['corner_world'].append(corner_world)
        
        # Convert hull points to world coordinates for more accurate representation
        obstacle['hull_world'] = []
        for point in obstacle['hull']:
            px, py = point[0]
            # Clamp coordinates to valid range
            px_clamped = int(np.clip(px, 0, depth_array_cropped.shape[1] - 1))
            py_clamped = int(np.clip(py, 0, depth_array_cropped.shape[0] - 1))
            point_depth = depth_array_cropped[py_clamped, px_clamped]
            point_world = pixel_to_world_topview(px, py, point_depth)
            obstacle['hull_world'].append(point_world)

    return tables, obstacles

