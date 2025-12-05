# woooo A*

import numpy as np
import heapq
from typing import List, Tuple, Set, Optional
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from pydrake.all import LeafSystem

from pathing.viz import visualize_grid_in_meshcat

TABLE_LENGTH = 0.9
TABLE_WIDTH = 1.3125
ROBOT_RADIUS = 0.35 
TRAY_OFFSET_THETA = 0

class Grid:
    """
    2D occupancy grid for A*
    """
    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float, resolution: float):
        """
            x_min, x_max: min and max x coordinates in world frame
            y_min, y_max: min and max y coordinates in world frame
            resolution: size of each grid cell (m)
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.resolution = resolution
        
        self.width = int(np.ceil((x_max - x_min) / resolution))
        self.height = int(np.ceil((y_max - y_min) / resolution))
        self.grid = np.zeros((self.height, self.width), dtype=bool)
        self.core_obstacles = np.zeros((self.height, self.width), dtype=bool)
        
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        grid_x = int((x - self.x_min) / self.resolution)
        grid_y = int((y - self.y_min) / self.resolution)
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x: int, grid_y: int) -> Tuple[float, float]:
        x = self.x_min + (grid_x + 0.5) * self.resolution
        y = self.y_min + (grid_y + 0.5) * self.resolution
        return x, y
    
    def is_valid(self, grid_x: int, grid_y: int) -> bool:
        return 0 <= grid_x < self.width and 0 <= grid_y < self.height
    
    def is_occupied(self, grid_x: int, grid_y: int) -> bool:
        if not self.is_valid(grid_x, grid_y):
            return True  # out of bounds is considered occupied
        return self.grid[grid_y, grid_x]
    
    def set_occupied(self, grid_x: int, grid_y: int, occupied: bool = True):
        if self.is_valid(grid_x, grid_y):
            self.grid[grid_y, grid_x] = occupied
    
    def line_of_sight(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        # Bresenham's line algorithm
        gx1, gy1 = self.world_to_grid(*p1)
        gx2, gy2 = self.world_to_grid(*p2)
        
        dx = abs(gx2 - gx1)
        dy = abs(gy2 - gy1)
        
        x = gx1
        y = gy1
        
        x_inc = 1 if gx2 > gx1 else -1
        y_inc = 1 if gy2 > gy1 else -1
        
        if dx > dy:
            error = dx / 2
            while x != gx2:
                if self.is_occupied(x, y):
                    return False
                error -= dy
                if error < 0:
                    y += y_inc
                    error += dx
                x += x_inc
        else:
            error = dy / 2
            while y != gy2:
                if self.is_occupied(x, y):
                    return False
                error -= dx
                if error < 0:
                    x += x_inc
                    error += dy
                y += y_inc
        
        if self.is_occupied(gx2, gy2):
            return False
        
        return True
    
    def add_obstacle_rectangle(self, x_center: float, y_center: float, 
                               width: float, height: float, padding: float = 0.0):
        x_min_core = x_center - width/2
        x_max_core = x_center + width/2
        y_min_core = y_center - height/2
        y_max_core = y_center + height/2
        
        grid_x_min_core, grid_y_min_core = self.world_to_grid(x_min_core, y_min_core)
        grid_x_max_core, grid_y_max_core = self.world_to_grid(x_max_core, y_max_core)
        
        for gx in range(max(0, grid_x_min_core), min(self.width, grid_x_max_core + 1)):
            for gy in range(max(0, grid_y_min_core), min(self.height, grid_y_max_core + 1)):
                self.core_obstacles[gy, gx] = True
        
        # add padding
        x_min = x_center - width/2 - padding
        x_max = x_center + width/2 + padding
        y_min = y_center - height/2 - padding
        y_max = y_center + height/2 + padding
        
        grid_x_min, grid_y_min = self.world_to_grid(x_min, y_min)
        grid_x_max, grid_y_max = self.world_to_grid(x_max, y_max)
        
        for gx in range(max(0, grid_x_min), min(self.width, grid_x_max + 1)):
            for gy in range(max(0, grid_y_min), min(self.height, grid_y_max + 1)):
                self.set_occupied(gx, gy, True)
    
    def add_obstacle_circle(self, x_center: float, y_center: float, 
                           radius: float, padding: float = 0.0):
        grid_x_center, grid_y_center = self.world_to_grid(x_center, y_center)
        grid_radius_core = int(np.ceil(radius / self.resolution))
        
        for gx in range(max(0, grid_x_center - grid_radius_core), 
                       min(self.width, grid_x_center + grid_radius_core + 1)):
            for gy in range(max(0, grid_y_center - grid_radius_core), 
                           min(self.height, grid_y_center + grid_radius_core + 1)):
                wx, wy = self.grid_to_world(gx, gy)
                dist = np.sqrt((wx - x_center)**2 + (wy - y_center)**2)
                if dist <= radius:
                    self.core_obstacles[gy, gx] = True
        
        # add padding
        total_radius = radius + padding
        grid_radius = int(np.ceil(total_radius / self.resolution))
        
        for gx in range(max(0, grid_x_center - grid_radius), 
                       min(self.width, grid_x_center + grid_radius + 1)):
            for gy in range(max(0, grid_y_center - grid_radius), 
                           min(self.height, grid_y_center + grid_radius + 1)):
                wx, wy = self.grid_to_world(gx, gy)
                dist = np.sqrt((wx - x_center)**2 + (wy - y_center)**2)
                if dist <= total_radius:
                    self.set_occupied(gx, gy, True)

class AStarPlanner:
    def __init__(self, grid: Grid):
        self.grid = grid
        
    def heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        # euclidean distance
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int, float]]:
        x, y = pos
        neighbors = []
        cardinal_directions = [
            (1, 0, 1.0), # right
            (-1, 0, 1.0),  # left
            (0, 1, 1.0), # up
            (0, -1, 1.0),  # down
        ]
        
        for dx, dy, cost in cardinal_directions:
            nx, ny = x + dx, y + dy
            if self.grid.is_valid(nx, ny) and not self.grid.is_occupied(nx, ny):
                neighbors.append((nx, ny, cost))
        
        # Diagonal directions + corner cutting check
        diagonal_directions = [
            (1, 1, 1.414, [(1, 0), (0, 1)]),  # upper right
            (1, -1, 1.414, [(1, 0), (0, -1)]),  # lower right
            (-1, 1, 1.414, [(-1, 0), (0, 1)]),  # upper left
            (-1, -1, 1.414, [(-1, 0), (0, -1)]) # lower left
        ]
        
        for dx, dy, cost, adjacent_cells in diagonal_directions:
            nx, ny = x + dx, y + dy
            
            if not self.grid.is_valid(nx, ny) or self.grid.is_occupied(nx, ny):
                continue
            
            corner_cut = False
            for adj_dx, adj_dy in adjacent_cells:
                adj_x, adj_y = x + adj_dx, y + adj_dy
                if not self.grid.is_valid(adj_x, adj_y) or self.grid.is_occupied(adj_x, adj_y):
                    corner_cut = True
                    break
            
            if not corner_cut:
                neighbors.append((nx, ny, cost))
        
        return neighbors
    
    def plan(self, start_world: Tuple[float, float], 
             goal_world: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        start_grid = self.grid.world_to_grid(*start_world)
        goal_grid = self.grid.world_to_grid(*goal_world)
        
        if self.grid.is_occupied(*start_grid):
            print(f"Start position {start_world} is occupied!")
            return None
        if self.grid.is_occupied(*goal_grid):
            print(f"Goal position {goal_world} is occupied!")
            return None
        
    
        open_set = [] 
        counter = 0 
        heapq.heappush(open_set, (0, counter, start_grid))
        
        came_from = {}  # For path reconstruction
        g_score = {start_grid: 0}  # Cost from start
        f_score = {start_grid: self.heuristic(start_grid, goal_grid)}  # Estimated total cost
        
        closed_set = set()
        
        while open_set:
            _, _, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            # Check if we reached the goal
            if current == goal_grid:
                path_grid = []
                while current in came_from:
                    path_grid.append(current)
                    current = came_from[current]
                path_grid.append(start_grid)
                path_grid.reverse()
                
                path_world = [self.grid.grid_to_world(gx, gy) for gx, gy in path_grid]
                return path_world
            
            # explore neighbors
            for neighbor, move_cost in [(n[:2], n[2]) for n in self.get_neighbors(current)]:
                if neighbor in closed_set:
                    continue
                
                tentative_g_score = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f = tentative_g_score + self.heuristic(neighbor, goal_grid)
                    f_score[neighbor] = f
                    
                    counter += 1
                    heapq.heappush(open_set, (f, counter, neighbor))
        
        print("No path found!")
        return None
    
    def smooth_path(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        1. Start with first waypoint
        2. Try to connect directly to farthest visible waypoint
        3. Use line-of-sight check to verify direct line is collision-free
        4. If yes, skip all waypoints in between
        5. Repeat from new waypoint
        """
        if len(path) <= 2:
            return path 
        
        smoothed = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            # Check from farthest to nearest (greedy approach)
            farthest_visible_idx = current_idx + 1
            for test_idx in range(len(path) - 1, current_idx, -1):
                if self.grid.line_of_sight(path[current_idx], path[test_idx]):
                    farthest_visible_idx = test_idx
                    break
            
            smoothed.append(path[farthest_visible_idx])
            current_idx = farthest_visible_idx
        
        return smoothed


class PathFollower(LeafSystem):
    def __init__(self, initial_position, arm_positions, paths=None, speed=0.5, start_theta=0.0, goal_theta=0.0, initial_delay=1.0, rotation_speed=1.0):
        LeafSystem.__init__(self)
        self.arm_positions = arm_positions
        self.initial_arm_positions = arm_positions
        self.speed = speed
        self.rotation_speed = rotation_speed  # radians per second
        self.start_theta = start_theta
        self.goal_theta = goal_theta
        self.previous_angle = start_theta  # Track previous angle to ensure smooth transitions
        self.initial_delay = initial_delay  # Time to wait before starting movement
        
        # Store all paths upfront
        self.all_paths = paths if paths is not None else []
        self.current_path_idx = 0
        self.current_path = None
        self.current_waypoint_times = None
        self.path_start_time = 0.0
        self.current_position = initial_position
        self.paths_initialized = False
        
        # desired state [10 positions + 10 velocities]
        self.DeclareVectorOutputPort("desired_state", 20, self.CalcOutput)
        
    def add_path(self, path: List[Tuple[float, float]]):
        # For backwards compatibility - add to the list
        if len(path) < 1:
            print("empty path, not adding")
            return
        
        self.all_paths.append(path)
    
    def set_paths(self, paths: List[List[Tuple[float, float]]]):
        """Set all paths at once."""
        self.all_paths = paths
        self.current_path_idx = 0
        self.paths_initialized = False
    
    def _calculate_waypoint_times(self, path: List[Tuple[float, float]], 
                                   start_pos: Tuple[float, float]) -> List[float]:
        times = [0.0]
        
        if len(path) > 0:
            dx = path[0][0] - start_pos[0]
            dy = path[0][1] - start_pos[1]
            distance = np.sqrt(dx**2 + dy**2)
            times.append(distance / self.speed)
        
        for i in range(1, len(path)):
            dx = path[i][0] - path[i-1][0]
            dy = path[i][1] - path[i-1][1]
            distance = np.sqrt(dx**2 + dy**2)
            times.append(times[-1] + distance / self.speed)
        
        return times
    
    def _start_next_path(self, current_time: float):
        if self.current_path_idx >= len(self.all_paths):
            self.current_path = None
            self.current_waypoint_times = None
            return
        
        self.current_path = self.all_paths[self.current_path_idx]
        self.current_path_idx += 1
        self.path_start_time = current_time
        self.current_waypoint_times = self._calculate_waypoint_times(
            self.current_path, self.current_position
        )
        
        total_time = self.current_waypoint_times[-1]
    
    def CalcOutput(self, context, output):
        t = context.get_time()
        
        # Wait for initial delay before starting
        if t < self.initial_delay:
            mobile_base_positions = [self.current_position[0], self.current_position[1], 0.1]
            all_positions = mobile_base_positions + self.arm_positions
            desired_state = all_positions + [0.0] * 10
            output.SetFromVector(desired_state)
            return
        
        # Initialize paths on first call after delay
        if not self.paths_initialized and len(self.all_paths) > 0:
            self._start_next_path(t)
            self.paths_initialized = True
        
        if self.current_path is None:
            mobile_base_positions = [self.current_position[0], self.current_position[1], 0.1]
            all_positions = mobile_base_positions + self.arm_positions
            desired_state = all_positions + [0.0] * 10
            output.SetFromVector(desired_state)
            return
        
        path_time = t - self.path_start_time
        total_path_time = self.current_waypoint_times[-1]
        
        if path_time >= total_path_time:
            self.current_position = self.current_path[-1]
            
            self._start_next_path(t)
            
            mobile_base_positions = [self.current_position[0], self.current_position[1], 0.1]
            all_positions = mobile_base_positions + self.arm_positions
            desired_state = all_positions + [0.0] * 10
            output.SetFromVector(desired_state)
            return
        
        segment_idx = 0
        for i in range(len(self.current_waypoint_times) - 1):
            if self.current_waypoint_times[i] <= path_time < self.current_waypoint_times[i + 1]:
                segment_idx = i
                break
        
        t0 = self.current_waypoint_times[segment_idx]
        t1 = self.current_waypoint_times[segment_idx + 1]
        alpha = (path_time - t0) / (t1 - t0) if t1 > t0 else 1.0
        
        if segment_idx == 0:
            x0, y0 = self.current_position
            x1, y1 = self.current_path[0]
        else:
            x0, y0 = self.current_path[segment_idx - 1]
            x1, y1 = self.current_path[segment_idx]
        
        current_x = x0 + alpha * (x1 - x0)
        current_y = y0 + alpha * (y1 - y0)
        
        vx = (x1 - x0) / (t1 - t0) if t1 > t0 else 0.0
        vy = (y1 - y0) / (t1 - t0) if t1 > t0 else 0.0

        # Set arm joint 0 based on current segment direction
        arm_positions = self.initial_arm_positions.copy()

        # Helper function to normalize angle difference to [-pi, pi]
        def angle_diff(target, current):
            diff = target - current
            # Wrap to [-pi, pi]
            while diff > np.pi:
                diff -= 2 * np.pi
            while diff < -np.pi:
                diff += 2 * np.pi
            return diff

        # Helper function to normalize angle to [-pi, pi]
        def normalize_angle(angle):
            while angle > np.pi:
                angle -= 2 * np.pi
            while angle < -np.pi:
                angle += 2 * np.pi
            return angle

        # Calculate the direction angle for the current segment
        dx = x1 - x0
        dy = y1 - y0

        if abs(dx) > 1e-6 or abs(dy) > 1e-6:
            # Use the direction of current segment with 180 degree offset
            segment_theta = np.arctan2(dy, dx) + TRAY_OFFSET_THETA
            segment_theta = normalize_angle(segment_theta)
        else:
            # No movement in this segment
            segment_theta = self.previous_angle
        
        # Check if we're in the last segment
        is_last_segment = (segment_idx == len(self.current_path) - 1)
        
        # Determine target angle
        if is_last_segment and alpha > 0.9:
            target_angle = self.goal_theta
        else:
            target_angle = segment_theta
        
        # Smoothly rotate towards target angle based on rotation_speed
        angle_difference = angle_diff(target_angle, self.previous_angle)
        
        # Calculate maximum rotation for this timestep
        # Estimate dt from velocity (rough approximation)
        dt = 0.01  # Assume small timestep for smooth control
        max_rotation = self.rotation_speed * dt
        
        # Limit the rotation to max_rotation per timestep
        if abs(angle_difference) < max_rotation:
            current_angle = target_angle
        else:
            current_angle = self.previous_angle + np.sign(angle_difference) * max_rotation
        
        arm_positions[0] = normalize_angle(current_angle)
        self.previous_angle = normalize_angle(current_angle)

        mobile_base_positions = [current_x, current_y, 0.1]
        all_positions = mobile_base_positions + arm_positions
        mobile_base_velocities = [vx, vy, 0.0]
        all_velocities = mobile_base_velocities + [0.0] * 7

        desired_state = all_positions + all_velocities
        output.SetFromVector(desired_state)


def add_obstacles(grid, tables):
    for table in tables:
        x, y, _ = table['center_world']
        angle = table['angle_radians']
        
        if angle == 0.0:
            grid.add_obstacle_rectangle(x, y, TABLE_LENGTH, TABLE_WIDTH, padding=ROBOT_RADIUS)
        else:
            grid.add_obstacle_rectangle(x, y, TABLE_WIDTH, TABLE_LENGTH, padding=ROBOT_RADIUS)
    

    grid.add_obstacle_circle(-0.35, 0.0, radius=0.3, padding=ROBOT_RADIUS)

def estimate_runtime(start_config, table_goals, all_paths):
     # Calculate total simulation time based on path distances and rotations
    total_distance = 0.0
    total_rotation = 0.0
    current_pos = (start_config[0], start_config[1])
    current_theta = start_config[2]
    
    for idx, (path, goal) in enumerate(zip(all_paths, table_goals)):
        # Distance from current position to first waypoint
        if len(path) > 0:
            dx = path[0][0] - current_pos[0]
            dy = path[0][1] - current_pos[1]
            total_distance += np.sqrt(dx**2 + dy**2)
            
            # Calculate rotation needed for this segment
            segment_angle = np.arctan2(dy, dx) + TRAY_OFFSET_THETA
            angle_diff = abs(segment_angle - current_theta)
            # Normalize to [0, pi]
            while angle_diff > np.pi:
                angle_diff = 2*np.pi - angle_diff
            total_rotation += angle_diff
            
            # Distance between waypoints in the path
            for i in range(1, len(path)):
                dx = path[i][0] - path[i-1][0]
                dy = path[i][1] - path[i-1][1]
                total_distance += np.sqrt(dx**2 + dy**2)
                
                # Rotation between segments
                next_segment_angle = np.arctan2(dy, dx) + TRAY_OFFSET_THETA
                angle_diff = abs(next_segment_angle - segment_angle)
                while angle_diff > np.pi:
                    angle_diff = 2*np.pi - angle_diff
                total_rotation += angle_diff
                segment_angle = next_segment_angle
            
            # Final rotation to goal orientation
            goal_theta = goal[2]
            angle_diff = abs(goal_theta - segment_angle)
            while angle_diff > np.pi:
                angle_diff = 2*np.pi - angle_diff
            total_rotation += angle_diff
            
            # Update current position and orientation
            current_pos = path[-1]
            current_theta = goal_theta
        
        return total_distance, total_rotation