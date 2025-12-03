from pydrake.systems.sensors import RgbdSensor, CameraInfo, PixelType
from pydrake.geometry import DepthRenderCamera, RenderCameraCore, ColorRenderCamera, ClippingRange, DepthRange
from pydrake.perception import DepthImageToPointCloud
from pydrake.math import RigidTransform
from pydrake.all import PointCloud
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import sys

TABLE_INTENSITY_MIN = 135
TABLE_INTENSITY_MAX = 140
SIZE_TOLERANCE = 0.07

# table size in pixels
TABLE_WIDTH = 45
TABLE_LENGTH = 66
TABLE_DEPTH = 11.248


CROP_X_START = 80  # left edge
CROP_Y_START = 0   # top edge
CROP_X_END = 560    # right edge
CROP_Y_END = 480    # bottom edge


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

def perceive_tables(station, station_context):
    """
    Perceive tables from the topview camera depth image.
    
    Args:
        station: The hardware station system
        station_context: The context for the station
    
    Returns:
        List of table dictionaries with both pixel and world coordinates
    """
    topview_camera_depth = station.GetOutputPort("topview_camera.depth_image").Eval(station_context)
    depth_array = np.copy(topview_camera_depth.data).reshape(
        (topview_camera_depth.height(), topview_camera_depth.width())
    )
    depth_array_cropped = depth_array[CROP_Y_START:CROP_Y_END, CROP_X_START:CROP_X_END]

    tables = detect_tables_from_img(depth_array_cropped)
    
    for table in tables:
        cx_pixel, cy_pixel = table['center']
        center_depth = depth_array_cropped[int(cy_pixel), int(cx_pixel)]
        table['center_world'] = pixel_to_world_topview(cx_pixel, cy_pixel, center_depth)
        w, h = table['size']
        table['corner_world'] = []
        for corner in table['box_corners']:
            corner_depth = depth_array_cropped[int(corner[1]), int(corner[0])]
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
    
    return tables

def detect_tables_from_img(depth_array):
    """
    filter by depth intensity and then by expected size
    """
    depth_array_vis = np.copy(depth_array)
    max_valid_depth = 15.0
    depth_array_vis[np.isinf(depth_array_vis)] = max_valid_depth

    depth_min = np.min(depth_array_vis)
    depth_max = np.max(depth_array_vis)
    if depth_max > depth_min:
        depth_normalized = 255 - ((depth_array_vis - depth_min) / (depth_max - depth_min) * 255)
    else:
        depth_normalized = np.zeros_like(depth_array_vis)

    # depth_image = Image.fromarray(depth_normalized.astype(np.uint8))
    # depth_output_path = Path("/workspaces/robman-final-proj/original_depth.png")
    # depth_image.save(depth_output_path)
    # print(f"Cropped depth image saved to {depth_output_path}")
    
    thresh = np.where((depth_normalized > TABLE_INTENSITY_MIN) & (depth_normalized <= TABLE_INTENSITY_MAX), 255, 0).astype(np.uint8)
    kernel_small = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # Save thresholded image for debugging
    thresh_output_path = Path("/workspaces/robman-final-proj/threshold_debug.png")
    cv2.imwrite(str(thresh_output_path), thresh)
    
    # Find contours of individual tables
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_features = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 2000:
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect
            dim1, dim2 = sorted([w, h])
            mask = np.zeros(depth_array.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], 0, 255, -1)
            mean_depth = np.mean(depth_array[mask > 0])
            
            contour_features.append({
                'contour': cnt,
                'area': area,
                'dim1': dim1,
                'dim2': dim2,
                'depth': mean_depth,
                'rect': rect
            })
    
    if not contour_features:
        print("No contours found!")
        table_contours = []
        
    # Filter: keep only objects that match table size
    table_contours = []
    
    
    for feat in contour_features:
        dim1_diff = abs(feat['dim1'] - TABLE_WIDTH) / TABLE_WIDTH
        dim2_diff = abs(feat['dim2'] - TABLE_LENGTH) / TABLE_LENGTH
        dim1_match = dim1_diff < SIZE_TOLERANCE
        dim2_match = dim2_diff < SIZE_TOLERANCE
        
        if dim1_match and dim2_match:
            table_contours.append(feat['contour'])

    # for debugging
    # result_img = cv2.cvtColor(depth_normalized.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    tables_info = []
    for i, contour in enumerate(table_contours):
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        (cx, cy), (w, h), angle = rect
        area = cv2.contourArea(contour)
        color = (0, 255, 0)

        waypoints_pixel = [
            (box[0] + box[1]) / 2,  # midpoint of edge 0-1
            (box[1] + box[2]) / 2,  # midpoint of edge 1-2
            (box[2] + box[3]) / 2,  # midpoint of edge 2-3
            (box[3] + box[0]) / 2   # midpoint of edge 3-0
        ]

        # for debugging
        # cv2.drawContours(result_img, [box], 0, color, 2)
        # cv2.drawContours(result_img, [contour], 0, (255, 0, 0), 1)
        
        # # label
        # label = f"Table {i+1}"
        # cv2.putText(result_img, label, (int(cx), int(cy)), 
        #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # for point in waypoints_pixel:
        #     cv2.circle(result_img, tuple(point.astype(int)), 3, (0, 255, 255), -1)
        
        tables_info.append({
            'id': i + 1,
            'center': (cx, cy),
            'size': (w, h),
            'angle': angle,
            'area': area,
            'box_corners': box,
            'waypoints': waypoints_pixel
        })
    
    # Table detection result
    # output_path = Path("/workspaces/robman-final-proj/table_detection.png")
    # cv2.imwrite(str(output_path), result_img)
    # print(f"\nDetection result saved to: {output_path}")
    
    return tables_info
