# woooo A*

import numpy as np
from typing import List, Tuple
# import matplotlib.pyplot as plt
from pydrake.all import LeafSystem

TRAY_OFFSET_THETA = 0

class PathFollower(LeafSystem):
    def __init__(self, initial_position, arm_positions, paths=None, speed=0.5, start_theta=0.0, goal_theta=0.0, initial_delay=1.0, rotation_speed=1.0):
        LeafSystem.__init__(self)
        self.arm_positions = arm_positions
        self.initial_arm_positions = arm_positions
        self.speed = speed
        self.rotation_speed = rotation_speed  # radians per second
        self.approach_rotation_speed = approach_rotation_speed
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