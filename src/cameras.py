from pydrake.systems.sensors import RgbdSensor, CameraInfo, PixelType
from pydrake.geometry import DepthRenderCamera, RenderCameraCore, ColorRenderCamera, ClippingRange, DepthRange
from pydrake.perception import DepthImageToPointCloud
from pydrake.math import RigidTransform
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

TABLE_INTENSITY_MIN = 130
TABLE_INTENSITY_MAX = 135
TABLE_WIDTH = 45
TABLE_LENGTH = 66
TABLE_DEPTH = 11.248
SIZE_TOLERANCE = 0.07


def add_cameras(builder, plant, scene_graph, scenario):
    """
    Adds RGB-D cameras to the simulation based on the provided camera configurations.

    Args:
        builder: The DiagramBuilder to which the cameras will be added.
        plant: The MultibodyPlant containing the robot and environment.
        scene_graph: The SceneGraph for geometry queries.
        camera_configs: A list of tuples (camera_name, camera_model) specifying
                        the cameras to be added.
    Returns:
        sensors: A dictionary mapping camera names to their RgbdSensor systems.
        depth_to_cloud_systems: A dictionary mapping camera names to their
                                DepthImageToPointCloud systems (if applicable).
    """
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
            
            # Connect to scene graph
            builder.Connect(
                scene_graph.get_query_output_port(),
                sensor.query_object_input_port()
            )
            
            # Export RGB and depth outputs
            builder.ExportOutput(sensor.color_image_output_port(), f"{cam_name}.rgb_image")
            builder.ExportOutput(sensor.depth_image_32F_output_port(), f"{cam_name}.depth_image")
            
            rgbd_sensors[cam_name] = sensor
            
            # Create point cloud converter for camera0, camera1, camera2 (not topview)
            if cam_name in ["camera0", "camera1", "camera2"]:
                depth_to_cloud = builder.AddSystem(
                    DepthImageToPointCloud(
                        camera_info=camera_info,
                        pixel_type=PixelType.kDepth32F
                    )
                )
                
                # Connect depth image to point cloud converter
                builder.Connect(
                    sensor.depth_image_32F_output_port(),
                    depth_to_cloud.depth_image_input_port()
                )
                
                # Connect camera pose
                builder.Connect(
                    sensor.body_pose_in_world_output_port(),
                    depth_to_cloud.camera_pose_input_port()
                )
                
                # Export point cloud output
                cam_num = cam_name[-1]  # Get the number from "camera0", "camera1", etc.
                builder.ExportOutput(
                    depth_to_cloud.point_cloud_output_port(),
                    f"camera_point_cloud{cam_num}"
                )
                
                point_cloud_systems[cam_name] = depth_to_cloud
                # print(f"  {cam_name}: RGBD sensor + point cloud converter")
            # else:
                # print(f"  {cam_name}: RGBD sensor only")
            
        except Exception as e:
            print(f"  Warning: Could not add camera {cam_name}: {e}")

def get_depth(diagram, context):
    # Save and crop the RGB image
    topview_camera_rgb = diagram.GetOutputPort("topview_camera.rgb_image").Eval(context)
    image_array = np.copy(topview_camera_rgb.data).reshape(
        (topview_camera_rgb.height(), topview_camera_rgb.width(), -1)
    )

    # Define crop region
    crop_x_start = 89  # left edge
    crop_y_start = 0   # top edge
    crop_x_end = 560    # right edge
    crop_y_end = 480    # bottom edge
    image_cropped = image_array[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

    # Depth image processing with crop
    topview_camera_depth = diagram.GetOutputPort("topview_camera.depth_image").Eval(context)
    depth_array = np.copy(topview_camera_depth.data).reshape(
        (topview_camera_depth.height(), topview_camera_depth.width())
    )
    depth_array_cropped = depth_array[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

    tables = detect_tables_from_depth(depth_array_cropped, image_cropped)


def detect_tables_from_depth(depth_array):
    """
    filter by depth intensity and then by expected size
    """
    #Normalize depth for visualization
    depth_array_vis = np.copy(depth_array)
    max_valid_depth = 15.0
    depth_array_vis[np.isinf(depth_array_vis)] = max_valid_depth

    depth_min = np.min(depth_array_vis)
    depth_max = np.max(depth_array_vis)
    if depth_max > depth_min:
        depth_normalized = 255 - ((depth_array_vis - depth_min) / (depth_max - depth_min) * 255)
    else:
        depth_normalized = np.zeros_like(depth_array_vis)

    depth_image = Image.fromarray(depth_normalized.astype(np.uint8))
    depth_output_path = Path("/workspaces/robman-final-proj/original_depth.png")
    depth_image.save(depth_output_path)
    print(f"Cropped depth image saved to {depth_output_path}")
    
    thresh = np.where((depth_normalized > TABLE_INTENSITY_MIN) & (depth_normalized <= TABLE_INTENSITY_MAX), 255, 0).astype(np.uint8)
    kernel_small = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # Save thresholded image for debugging
    # thresh_output_path = Path("/workspaces/robman-final-proj/threshold_debug.png")
    # cv2.imwrite(str(thresh_output_path), thresh)
    
    # Find contours of individual tables
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
        
        # Only accept if size AND depth match (more strict)
        if dim1_match and dim2_match:
            table_contours.append(feat['contour'])

    # Create visualization using the normalized depth image (not raw float32)
    result_img = cv2.cvtColor(depth_normalized.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
    tables_info = []
    for i, contour in enumerate(table_contours):
        # Fit minimum area rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        
        # Get rectangle info
        (cx, cy), (w, h), angle = rect
        area = cv2.contourArea(contour)
        
        # Draw contour and bounding box
        color = (0, 255, 0)
        cv2.drawContours(result_img, [box], 0, color, 2)
        cv2.drawContours(result_img, [contour], 0, (255, 0, 0), 1)
        
        # label
        label = f"Table {i+1}"
        cv2.putText(result_img, label, (int(cx), int(cy)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Calculate waypoints around the table perimeter
        waypoints_pixel = []
        num_waypoints = 8
        for j in range(num_waypoints):
            t = j / num_waypoints
            if t < 0.25:
                s = t / 0.25
                point = box[0] + s * (box[1] - box[0])
            elif t < 0.5:
                s = (t - 0.25) / 0.25
                point = box[1] + s * (box[2] - box[1])
            elif t < 0.75:
                s = (t - 0.5) / 0.25
                point = box[2] + s * (box[3] - box[2])
            else:
                s = (t - 0.75) / 0.25
                point = box[3] + s * (box[0] - box[3])
            
            waypoints_pixel.append(point)
            cv2.circle(result_img, tuple(point.astype(int)), 3, (0, 255, 255), -1)
        
        tables_info.append({
            'id': i + 1,
            'center': (cx, cy),
            'size': (w, h),
            'angle': angle,
            'area': area,
            'box_corners': box,
            'waypoints': waypoints_pixel
        })
        
        # print(f"\n   Table {i+1}:")
        # print(f"      Center: ({cx:.1f}, {cy:.1f}) pixels")
        # print(f"      Size: {w:.1f} x {h:.1f} pixels")
        # print(f"      Angle: {angle:.1f}°")
        # print(f"      Area: {area:.0f} pixels²")
    
    # Table detection result
    output_path = Path("/workspaces/robman-final-proj/table_detection.jpg")
    cv2.imwrite(str(output_path), result_img)
    print(f"\nDetection result saved to: {output_path}")
    
    return tables_info
