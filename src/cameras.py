from pydrake.systems.sensors import RgbdSensor, CameraInfo, PixelType
from pydrake.geometry import DepthRenderCamera, RenderCameraCore, ColorRenderCamera, ClippingRange, DepthRange
from pydrake.perception import DepthImageToPointCloud
from pydrake.math import RigidTransform
import numpy as np
from pathlib import Path
from PIL import Image

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

    # Save cropped image
    image = Image.fromarray(image_cropped.astype(np.uint8))
    output_path = Path("/workspaces/robman-final-proj/topview_camera_image.png")
    image.save(output_path)
    print(f"Cropped image saved to {output_path} with shape {image_cropped.shape}")

    # Depth image processing with crop
    topview_camera_depth = diagram.GetOutputPort("topview_camera.depth_image").Eval(context)
    depth_array = np.copy(topview_camera_depth.data).reshape(
        (topview_camera_depth.height(), topview_camera_depth.width())
    )
    depth_array_cropped = depth_array[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

    # Normalize depth for visualization
    depth_array_vis = np.copy(depth_array_cropped)
    max_valid_depth = 15.0
    depth_array_vis[np.isinf(depth_array_vis)] = max_valid_depth

    depth_min = np.min(depth_array_vis)
    depth_max = np.max(depth_array_vis)
    if depth_max > depth_min:
        depth_normalized = 255 - ((depth_array_vis - depth_min) / (depth_max - depth_min) * 255)
    else:
        depth_normalized = np.zeros_like(depth_array_vis)

    depth_image = Image.fromarray(depth_normalized.astype(np.uint8))
    depth_output_path = Path("/workspaces/robman-final-proj/topview_camera_depth.png")
    depth_image.save(depth_output_path)
    print(f"Cropped depth image saved to {depth_output_path}")
