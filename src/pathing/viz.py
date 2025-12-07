from pydrake.geometry import Box, Rgba
from pydrake.math import RigidTransform, RotationMatrix
import numpy as np
import cv2
from pathlib import Path
from perception.config import (
    CORNER_EXCLUSION_X,
    CORNER_EXCLUSION_Y,
    CROP_X_START,
    CROP_X_END,
    CROP_Y_START,
    CROP_Y_END
)

def visualize_perceived_tables(tables, meshcat):
    """
    Visualize perceived tables in meshcat as transparent colored boxes.
    
    Args:
        tables: List of table dictionaries from perceive_tables()
                Each table should have:
                - 'id': table identifier
                - 'center_world': (x, y, z) tuple of table center in world coordinates
                - 'angle_radians': rotation angle in radians
        meshcat: Meshcat visualizer instance
    """
    # Known table dimensions from the SDF files (collision box for table top)
    TABLE_LENGTH = 0.9  # meters (X dimension)
    TABLE_WIDTH = 1.3125  # meters (Y dimension)
    TABLE_HEIGHT = 0.7  # meters (Z dimension - visual height)
    
    # Define colors for different tables (RGBA with transparency)
    colors = [
        Rgba(1.0, 0.0, 0.0, 0.3),  # Red
        Rgba(0.0, 1.0, 0.0, 0.3),  # Green
        Rgba(0.0, 0.0, 1.0, 0.3),  # Blue
        Rgba(1.0, 1.0, 0.0, 0.3),  # Yellow
        Rgba(1.0, 0.0, 1.0, 0.3),  # Magenta
        Rgba(0.0, 1.0, 1.0, 0.3),  # Cyan
    ]
    
    for i, table in enumerate(tables):
        table_id = table['id']
        center_world = table['center_world']
        angle = table['angle_radians']
        
        box = Box(TABLE_LENGTH, TABLE_WIDTH, TABLE_HEIGHT)
        color = colors[i % len(colors)]
        path = f"perceived_tables/table_{table_id}"
        meshcat.SetObject(path, box, color)
        
        from pydrake.math import RigidTransform, RotationMatrix
        
        adjusted_z = TABLE_HEIGHT / 2.0
        position = [center_world[0], center_world[1], adjusted_z]
        
        rotation = RotationMatrix.MakeZRotation(angle)
        transform = RigidTransform(rotation, position)
        
        meshcat.SetTransform(path, transform)


def visualize_grid_in_meshcat(grid, meshcat, show_grid_lines=True):
    """
    Visualize the occupancy grid in meshcat with grid lines on the floor.
    
    Args:
        grid: The Grid object to visualize
        meshcat: Meshcat visualizer instance
        show_grid_lines: If True, draw grid lines on the floor
    """
    cell_size = grid.resolution
    cell_height = 0.2  # Very thin boxes
    line_height = 0.1  # Height for grid lines
    line_thickness = 0.005  # Thickness of grid lines
    
    # Draw grid lines first (on the floor)
    if show_grid_lines:
        grid_color = Rgba(0.5, 0.5, 0.5, 0.3)  # Light gray, semi-transparent
        
        # Vertical lines (along Y axis)
        for i in range(grid.width + 1):
            x = grid.x_min + i * cell_size
            y_center = (grid.y_min + grid.y_max) / 2
            length = grid.y_max - grid.y_min
            
            line = Box(line_thickness, length, line_height)
            path = f"grid/lines/vertical_{i}"
            meshcat.SetObject(path, line, grid_color)
            transform = RigidTransform([x, y_center, line_height / 2])
            meshcat.SetTransform(path, transform)
        
        # Horizontal lines (along X axis)
        for i in range(grid.height + 1):
            y = grid.y_min + i * cell_size
            x_center = (grid.x_min + grid.x_max) / 2
            length = grid.x_max - grid.x_min
            
            line = Box(length, line_thickness, line_height)
            path = f"grid/lines/horizontal_{i}"
            meshcat.SetObject(path, line, grid_color)
            transform = RigidTransform([x_center, y, line_height / 2])
            meshcat.SetTransform(path, transform)
        
        print(f"Drew {grid.width + 1 + grid.height + 1} grid lines")
    
    # Visualize occupied cells
    occupied_count = 0
    for gy in range(grid.height):
        for gx in range(grid.width):
            if grid.grid[gy, gx]:  # Occupied cell
                wx, wy = grid.grid_to_world(gx, gy)
                
                # Determine if it's a core obstacle or just padding
                if grid.core_obstacles[gy, gx]:
                    color = Rgba(0.3, 0.3, 0.3, 0.8)  # Dark gray for core obstacles
                    path = f"grid/obstacles/core_{gx}_{gy}"
                else:
                    color = Rgba(1.0, 0.5, 0.0, 0.4)  # Orange for padding
                    path = f"grid/obstacles/padding_{gx}_{gy}"
                
                box = Box(cell_size * 0.95, cell_size * 0.95, cell_height)  # Slightly smaller to see grid
                meshcat.SetObject(path, box, color)
                
                transform = RigidTransform([wx, wy, cell_height / 2])
                meshcat.SetTransform(path, transform)
                
                occupied_count += 1
    
    print(f"Visualized {occupied_count} occupied cells in meshcat")


def visualize_robot_config(meshcat, x, y, theta, label="robot", color=None):
    """
    Visualize a robot configuration (x, y, theta) in meshcat.
    
    Args:
        meshcat: Meshcat visualizer instance
        x, y: Position in world coordinates
        theta: Orientation in radians
        label: Name for this robot visualization
        color: Rgba color (default: cyan for robot, red for goal)
    """
    if color is None:
        color = Rgba(0.0, 1.0, 1.0, 0.7)  # Cyan
    
    # Robot footprint visualization (circular)
    from pydrake.geometry import Cylinder
    robot_radius = 0.35
    robot_height = 0.4
    
    # Draw robot body
    body = Cylinder(robot_radius, robot_height)
    path = f"{label}/body"
    meshcat.SetObject(path, body, color)
    
    position = [x, y, robot_height / 2]
    transform = RigidTransform(position)
    meshcat.SetTransform(path, transform)
    
    # Draw orientation arrow
    arrow_length = 0.4
    arrow_width = 0.05
    arrow_height = 0.1
    
    arrow = Box(arrow_length, arrow_width, arrow_height)
    path_arrow = f"{label}/arrow"
    arrow_color = Rgba(color.r(), color.g(), color.b(), 1.0)  # More opaque
    meshcat.SetObject(path_arrow, arrow, arrow_color)
    
    # Position arrow in front of robot, pointing forward
    arrow_x = x + (arrow_length / 2) * np.cos(theta)
    arrow_y = y + (arrow_length / 2) * np.sin(theta)
    arrow_position = [arrow_x, arrow_y, arrow_height / 2]
    
    rotation = RotationMatrix.MakeZRotation(theta)
    arrow_transform = RigidTransform(rotation, arrow_position)
    meshcat.SetTransform(path_arrow, arrow_transform)


def visualize_path(meshcat, path, color=None, label="path", show_orientations=False):
    """
    Visualize a path in meshcat as a line with waypoint markers.
    
    Args:
        meshcat: Meshcat visualizer instance
        path: List of (x, y) tuples representing waypoints
        color: Rgba color for the path (default: blue)
        label: Name for this path visualization
        show_orientations: If True, show orientation arrows at each waypoint
    """
    if color is None:
        color = Rgba(0.0, 0.0, 1.0, 0.8)  # Blue
    
    if len(path) < 2:
        return
    
    # Draw line segments between waypoints
    for i in range(len(path) - 1):
        x1, y1 = path[i]
        x2, y2 = path[i + 1]
        
        # Calculate segment properties
        dx = x2 - x1
        dy = y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        
        # Center of segment
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        
        # Draw segment as thin box
        segment_width = 0.03
        segment_height = 0.05
        segment = Box(length, segment_width, segment_height)
        
        path_name = f"{label}/segments/seg_{i}"
        meshcat.SetObject(path_name, segment, color)
        
        rotation = RotationMatrix.MakeZRotation(angle)
        position = [cx, cy, segment_height / 2]
        transform = RigidTransform(rotation, position)
        meshcat.SetTransform(path_name, transform)
    
    # Draw waypoint markers
    marker_radius = 0.05
    marker_height = 0.1
    from pydrake.geometry import Cylinder
    
    for i, (x, y) in enumerate(path):
        marker = Cylinder(marker_radius, marker_height)
        path_name = f"{label}/waypoints/wp_{i}"
        
        # Start and end get different colors
        if i == 0:
            marker_color = Rgba(0.0, 1.0, 0.0, 0.9)  # Green for start
        elif i == len(path) - 1:
            marker_color = Rgba(1.0, 0.0, 0.0, 0.9)  # Red for end
        else:
            marker_color = color
        
        meshcat.SetObject(path_name, marker, marker_color)
        
        position = [x, y, marker_height / 2]
        transform = RigidTransform(position)
        meshcat.SetTransform(path_name, transform)
        
        # Optionally show orientations
        if show_orientations and i < len(path) - 1:
            # Calculate orientation from current to next waypoint
            dx = path[i + 1][0] - x
            dy = path[i + 1][1] - y
            theta = np.arctan2(dy, dx)
            
            arrow_length = 0.15
            arrow_width = 0.03
            arrow_height = 0.08
            
            arrow = Box(arrow_length, arrow_width, arrow_height)
            arrow_path = f"{label}/orientations/arrow_{i}"
            meshcat.SetObject(arrow_path, arrow, Rgba(1.0, 1.0, 0.0, 0.8))
            
            arrow_x = x + (arrow_length / 2 + marker_radius) * np.cos(theta)
            arrow_y = y + (arrow_length / 2 + marker_radius) * np.sin(theta)
            arrow_position = [arrow_x, arrow_y, arrow_height / 2]
            
            rotation = RotationMatrix.MakeZRotation(theta)
            arrow_transform = RigidTransform(rotation, arrow_position)
            meshcat.SetTransform(arrow_path, arrow_transform)
    
    print(f"Visualized path '{label}' with {len(path)} waypoints")


def visualize_scene_detection(station, station_context, tables, obstacles):
    """
    Create visualization of both tables and obstacles.
    """
    topview_camera_depth = station.GetOutputPort("topview_camera.depth_image").Eval(station_context)
    depth_array = np.copy(topview_camera_depth.data).reshape(
        (topview_camera_depth.height(), topview_camera_depth.width())
    )
    depth_array_cropped = depth_array[CROP_Y_START:CROP_Y_END, CROP_X_START:CROP_X_END]
    
    # Normalize for visualization
    depth_array_vis = np.copy(depth_array_cropped)
    max_valid_depth = 15.0
    depth_array_vis[np.isinf(depth_array_vis)] = max_valid_depth
    depth_min = np.min(depth_array_vis)
    depth_max = np.max(depth_array_vis)
    if depth_max > depth_min:
        depth_normalized = 255 - ((depth_array_vis - depth_min) / (depth_max - depth_min) * 255)
    else:
        depth_normalized = np.zeros_like(depth_array_vis)
    
    result_img = cv2.cvtColor(depth_normalized.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
    # Draw exclusion zone in gray
    height, width = depth_array_cropped.shape
    cv2.rectangle(result_img, 
                    (width - CORNER_EXCLUSION_Y, 0),
                    (width, CORNER_EXCLUSION_X),
                  (128, 128, 128), 2)
    cv2.putText(result_img, "EXCLUDED", 
                (width-CORNER_EXCLUSION_Y+5, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
    
    # Draw tables in green
    for table in tables:
        box = table['box_corners']
        cv2.drawContours(result_img, [box], 0, (0, 255, 0), 2)
        cx, cy = table['center']
        cv2.putText(result_img, f"Table {table['id']}", (int(cx), int(cy)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw obstacles in red
    for obstacle in obstacles:
        # Draw convex hull for better visualization
        cv2.drawContours(result_img, [obstacle['hull']], 0, (0, 0, 255), 2)
        cx, cy = obstacle['center']
        cv2.circle(result_img, (int(cx), int(cy)), 5, (0, 0, 255), -1)
        cv2.putText(result_img, f"Obj {obstacle['id']}", (int(cx), int(cy) - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    output_path = Path("/workspaces/robman-final-proj/scene_detection.png")
    cv2.imwrite(str(output_path), result_img)
    print(f"\nScene detection saved to: {output_path}")