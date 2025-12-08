from typing import List, Tuple
import numpy as np
from pathing.grid import Grid

TABLE_LENGTH = 0.9
TABLE_WIDTH = 1.3125
# Increase padding so robot stops further from table edges for better camera visibility
ROBOT_RADIUS = 0.37

def add_obstacles(grid, tables, obstacles=None):
    """
    Add tables and obstacles to the grid.
    
    Args:
        grid: The Grid object for path planning
        tables: List of detected tables with world coordinates
        obstacles: List of detected obstacles with world coordinates (optional)
    """
    # Add tables
    for table in tables:
        x, y, _ = table['center_world']
        angle = table['angle_radians']
        
        table_padding = ROBOT_RADIUS + 0.3
        if angle == 0.0:
            grid.add_obstacle_rectangle(x, y, TABLE_LENGTH, TABLE_WIDTH, padding=table_padding)
        else:
            grid.add_obstacle_rectangle(x, y, TABLE_WIDTH, TABLE_LENGTH, padding=table_padding)
    
    # Add detected obstacles (unknown objects)
    if obstacles is not None:
        for obstacle in obstacles:
            # Use convex hull points for accurate representation
            if 'hull_world' in obstacle and len(obstacle['hull_world']) > 0:
                add_obstacle_polygon(grid, obstacle['hull_world'], padding=ROBOT_RADIUS)
            # Fallback to bounding box if hull not available
            elif 'center_world' in obstacle and 'size' in obstacle:
                x, y, _ = obstacle['center_world']
                w, h = obstacle['size']
                # Convert pixel dimensions to world dimensions (approximate)
                w_world = w * grid.resolution
                h_world = h * grid.resolution
                grid.add_obstacle_rectangle(x, y, w_world, h_world, padding=ROBOT_RADIUS)
    
    # Add static obstacle (robot base)
    grid.add_obstacle_circle(-0.35, 0.0, radius=0.3, padding=ROBOT_RADIUS)

def add_obstacle_polygon(grid: Grid, world_points: List[Tuple[float, float, float]], padding: float = 0.0):
    """
    Add a polygonal obstacle to the grid using its world coordinate points.
    
    Args:
        grid: The Grid object
        world_points: List of (x, y, z) world coordinates defining the polygon boundary
        padding: Additional padding around the obstacle for robot clearance
    """
    if len(world_points) < 3:
        return
    
    # Extract x, y coordinates (ignore z)
    polygon_2d = [(pt[0], pt[1]) for pt in world_points]
    
    # Find bounding box
    x_coords = [pt[0] for pt in polygon_2d]
    y_coords = [pt[1] for pt in polygon_2d]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # Convert to grid coordinates
    grid_x_min, grid_y_min = grid.world_to_grid(x_min - padding, y_min - padding)
    grid_x_max, grid_y_max = grid.world_to_grid(x_max + padding, y_max + padding)
    
    # Iterate over bounding box and check if each cell is inside polygon
    for gx in range(max(0, grid_x_min), min(grid.width, grid_x_max + 1)):
        for gy in range(max(0, grid_y_min), min(grid.height, grid_y_max + 1)):
            wx, wy = grid.grid_to_world(gx, gy)
            
            # Check if point is inside polygon (with padding)
            if point_in_polygon_with_padding(wx, wy, polygon_2d, padding):
                grid.set_occupied(gx, gy, True)

def point_in_polygon_with_padding(x: float, y: float, polygon: List[Tuple[float, float]], padding: float) -> bool:
    """
    Check if a point (x, y) is inside or within padding distance of a polygon.
    Uses ray casting algorithm for inside check and distance calculation for padding.
    """
    # First check if point is inside polygon
    inside = point_in_polygon(x, y, polygon)
    if inside:
        return True
    
    # If not inside, check if within padding distance of any edge
    if padding > 0:
        return point_near_polygon(x, y, polygon, padding)
    
    return False

def point_in_polygon(x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
    """
    Ray casting algorithm to check if point is inside polygon.
    """
    n = len(polygon)
    inside = False
    
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        
        j = i
    
    return inside

def point_near_polygon(x: float, y: float, polygon: List[Tuple[float, float]], max_dist: float) -> bool:
    """
    Check if point is within max_dist of any edge of the polygon.
    """
    n = len(polygon)
    
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        
        dist = point_to_segment_distance(x, y, p1[0], p1[1], p2[0], p2[1])
        if dist <= max_dist:
            return True
    
    return False

def point_to_segment_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """
    Calculate minimum distance from point (px, py) to line segment (x1, y1) - (x2, y2).
    """
    dx = x2 - x1
    dy = y2 - y1
    
    if dx == 0 and dy == 0:
        # Segment is a point
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    
    # Parameter t represents position along segment (0 to 1)
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    
    # Closest point on segment
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    
    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)