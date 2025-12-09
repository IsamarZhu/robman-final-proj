import numpy as np
import gc
from typing import List, Tuple
from pydrake.all import RigidTransform, RotationMatrix, PiecewisePolynomial

from state_machine.states import CafeState
from state_machine.helpers import normalize_angle, find_nearest_table
from perception.object_detection import detect_and_locate_object
from grasping.motion_primitives import pick_object

# --------------------------------------------------------------------------- #
# Configuration (maybe incorporate back later?)
# --------------------------------------------------------------------------- #

# motion parameters
APPROACH_HEIGHT = 0.15
LIFT_HEIGHT = 0.2
GRASP_OFFSET = 0.00

# gripper settings
WSG_OPEN = 0.15
WSG_CLOSED = 0.015

# timing
MOVE_TIME = 2.5
GRASP_TIME = 2.0
LIFT_TIME = 4.0 

# segmentation parameters
DBSCAN_EPS = 0.03
DBSCAN_MIN_SAMPLES = 50


class CafeStateMachine:
    def __init__(
        self,
        env,
        approach_height=0.15,
        lift_height=0.20,
        grasp_offset=0.00,
        wsg_open=0.15,
        wsg_closed=0.015,
        move_time=2.5,
        grasp_time=2.0,
        lift_time=4.0,
        dbscan_eps=0.02,
        dbscan_min_samples=50,
        # navigation params
        paths=[],
        table_centers=[],
        rotation_speed=0.02,
        acceleration=0.03,
        deceleration=0.03,
        max_speed=0.5,
        
    ):
        self.env = env
        self.current_state = CafeState.IDLE
        self.scenario_number = env.scenario_number
        if self.scenario_number == "one":
            self.object_queue = ["mug", "gelatin_box", "apple"]
        elif self.scenario_number == "two":
            self.object_queue = ["potted_meat", "master_chef", "tuna"]
        elif self.scenario_number == "three":
            self.object_queue = ["pudding", "cupd"]
        self.current_object_index = 0

        self.approach_height = approach_height
        self.lift_height = lift_height
        self.grasp_offset = grasp_offset
        self.wsg_open = wsg_open
        self.wsg_closed = wsg_closed
        self.move_time = move_time
        self.grasp_time = grasp_time
        self.lift_time = lift_time
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples

        self.grasp_center_xyz = None
        self.object_top_z = None
        self.object_cloud = None
        self.baseline_force = 0.0

        self.q_approach = None
        self.q_grasp = None
        self.q_lift = None
        self.q_lift_higher = None
        self.q_place = None
        
        # navigation params
        self.paths = paths
        self.current_path_idx = 0
        self.current_waypoint_idx = 0
        self.table_centers = table_centers
        self.rotation_speed = rotation_speed
        self.rotation_tolerance = 0.01  # radians
        self.position_tolerance = 0.001  # meters
        self.current_position = None  # (x, y)
        self.current_theta = None  # orientation
        self.target_position = None  # (x, y)
        self.target_theta = None  # orientation
        self.movement_start_pos = None  # (x, y)
        self.movement_distance = 0.0
        self.current_speed = 0.0
        self.acceleration = acceleration
        self.deceleration = deceleration
        self.max_speed = max_speed
        self.slide_distance = 0.5
        self.state_start_time = 0.0
        self.dt = 0.01

        # hold arm configuration during base navigation to prevent droop due to grav
        self.nav_hold_q = None

    def run(self, dt=0.01, max_time=300.0):
        print("\n=== Starting Cafe State Machine ===")
        t_start = self.env.simulator.get_context().get_time()
        self.dt = dt
        while True:
            if self.current_position is None:
                plant = self.env.plant
                plant_context = self.env.plant_context
                iiwa_model = self.env.iiwa_model
                q_current = plant.GetPositions(plant_context, iiwa_model)
                self.current_position = (q_current[0], q_current[1])
                self.current_theta = q_current[3]     
            
            # Handle current state
            if len(self.all_paths) > 0 and self.current_state == CafeState.IDLE:
                self._start_navigation_sequence()  # Let next update handle the new state
        
            if self.current_state == CafeState.IDLE:
                if self.current_path_idx >= len(self.all_paths):
                    print("\n[COMPLETE - All paths and objects processed]")
                    break
        
            elif self.current_state == CafeState.NAVIGATE_ROTATE_TO_WAYPOINT:
                self.navigate_rotate_to_waypoint_state()
                
            elif self.current_state == CafeState.NAVIGATE_MOVE_TO_WAYPOINT:
                self.navigate_move_to_waypoint_state()
                
            elif self.current_state == CafeState.NAVIGATE_ROTATE_TO_TABLE:
                self.navigate_rotate_to_table_state()
                
            elif self.current_state == CafeState.NAVIGATE_SLIDE_LEFT:
                self.navigate_slide_left_state()
                
            elif self.current_state == CafeState.PERCEPTION:
                self.perception_state()
                
            elif self.current_state == CafeState.PICK:
                self.pick_state()
                
            # elif self.current_state == CafeState.PLACE:
            #     self.place_state()
                
            elif self.current_state == CafeState.NAVIGATE_SLIDE_RIGHT:
                self.navigate_slide_right_state()
            
            t_current = self.env.simulator.get_context().get_time()
            self.env.simulator.AdvanceTo(t_current + dt)

    def set_paths(self, paths: List[List[Tuple[float, float]]]):
        """Set all paths for navigation."""
        self.all_paths = paths
        self.current_path_idx = 0
        self.current_waypoint_idx = 0
    
    def set_table_centers(self, table_centers: List[Tuple[float, float, float]]):
        """Set table center positions for approach maneuvers."""
        self.table_centers = table_centers

    def perception_state(self):
        import psutil
        import os
        process = psutil.Process(os.getpid())
        print(f"Memory before settle: {process.memory_info().rss / 1024 / 1024:.1f} MB")

        # self.env.meshcat.StopRecording()
        self.env.settle_scene(duration=2.0)

        self.env.meshcat.Delete("detection")
        self.env.meshcat.Delete("grasp")
        self.env.meshcat.Delete("grid")
        self.env.meshcat.Delete("traj")
        gc.collect()

        print(f"Memory after cleanup: {process.memory_info().rss / 1024 / 1024:.1f} MB")

        # self.env.meshcat.StartRecording(frames_per_second=10)

        target_object = self.object_queue[self.current_object_index]
        print(f"\n[PERCEPTION]")
        print(f"Memory before detection: {process.memory_info().rss / 1024 / 1024:.1f} MB")

        detection_result = detect_and_locate_object(
            self.scenario_number,
            self.env.diagram,
            self.env.context,
            self.env.meshcat,
            target_object=target_object,
            dbscan_eps=self.dbscan_eps,
            dbscan_min_samples=self.dbscan_min_samples,
            grasp_offset=self.grasp_offset,
        )
        
        X_WO, self.grasp_center_xyz, self.object_top_z, self.object_cloud = detection_result

        self.current_state = CafeState.PICK

    def pick_state(self):
        print("\n[PICK]")

        pick_object(
            self.env.meshcat,
            self.env.simulator,
            self.env.diagram,
            self.env.plant,
            self.env.plant_context,
            self.env.iiwa_model,
            self.env.wsg_model,
            self.env.cmd_source,
            self.env.wsg_cmd_source,
            self.grasp_center_xyz,
            self.object_top_z,
            approach_height=self.approach_height,
            lift_height=self.lift_height,
            wsg_open=self.wsg_open,
            wsg_closed=self.wsg_closed,
            move_time=self.move_time,
            grasp_time=self.grasp_time,
            lift_time=self.lift_time,
            object_cloud=self.object_cloud,
        )
        self.env.meshcat.Delete("detection")
        self.env.meshcat.Delete("grasp")
        self.move_state()
        
    def move_state(self):
        print("\n[MOVE]")
        self.env.settle_scene(duration=2.0)

        # hold arm joints to current configuration for navigation to prevent slow droop due to gravity
        plant = self.env.plant
        plant_context = self.env.plant_context
        iiwa_model = self.env.iiwa_model
        self.nav_hold_q = plant.GetPositions(plant_context, iiwa_model).copy()

        self.current_object_index += 1
        self.current_path_idx += 1

        gc.collect()

        if self.current_object_index < len(self.object_queue):
            self.current_waypoint_idx = 0
            self._start_rotation_to_waypoint()
        else:
            print("\n[COMPLETE]")
            self._transition_to_state(CafeState.IDLE)

    def _transition_to_state(self, new_state: CafeState):
        """Transition to a new state."""
        t = self.env.simulator.get_context().get_time()
        print(f"[{t:.2f}s] State: {self.current_state.name} -> {new_state.name}")
        self.current_state = new_state
        self.state_start_time = t
    
    def _start_navigation_sequence(self):
        """Start navigating through paths."""
        print(f"\n=== Starting navigation with {len(self.all_paths)} paths ===")
        self._start_rotation_to_waypoint()
    
    def _start_rotation_to_waypoint(self):
        """Start rotating to face the next waypoint."""
        if self.current_path_idx >= len(self.all_paths):
            return
        path = self.all_paths[self.current_path_idx]
        if path is None or self.current_waypoint_idx >= len(path):
            return
        
        waypoint = path[self.current_waypoint_idx]
        dx = waypoint[0] - self.current_position[0]
        dy = waypoint[1] - self.current_position[1]
        
        self.target_theta = np.arctan2(dy, dx)
        self._transition_to_state(CafeState.NAVIGATE_ROTATE_TO_WAYPOINT)
        print(f"  Rotating to face waypoint {self.current_waypoint_idx}: ({waypoint[0]:.2f}, {waypoint[1]:.2f})")
    
    def navigate_rotate_to_waypoint_state(self):
        """Rotate to face next waypoint using shortest angle."""
        t = self.env.simulator.get_context().get_time()
        dt_elapsed = t - self.state_start_time
        
        # Get ACTUAL current position from plant
        plant = self.env.plant
        plant_context = self.env.plant_context
        iiwa_model = self.env.iiwa_model
        q_current = plant.GetPositions(plant_context, iiwa_model)
        actual_theta = q_current[3]
        
        # Calculate where we need to go
        delta = self.target_theta - actual_theta
        angle_diff = normalize_angle(delta)
        rotation_needed = abs(angle_diff)
        
        # Calculate how long rotation should take
        total_rotation_time = rotation_needed / self.rotation_speed
        
        # Check BOTH conditions: angle is close AND enough time has passed
        if rotation_needed < self.rotation_tolerance and dt_elapsed >= total_rotation_time:
            # Rotation complete - stop rotating
            self._update_robot_base_orientation(actual_theta, 0.0) # this used to go to target theta, supsicious fix, idgaf lmfao
            self._start_movement_to_waypoint()
            print(f"    Rotation complete: took {dt_elapsed:.2f}s")
        else:
            # Keep rotating
            rotation_sign = np.sign(angle_diff)
            omega = rotation_sign * self.rotation_speed
            
            lookahead_angle = omega * self.dt
            desired_theta = actual_theta + lookahead_angle
            
            self._update_robot_base_orientation(desired_theta, omega)
                
    def _start_movement_to_waypoint(self):
        """Start moving forward to the next waypoint."""
        if self.current_path_idx >= len(self.all_paths):
            return
        path = self.all_paths[self.current_path_idx]
        if path is None or self.current_waypoint_idx >= len(path):
            return
        
        self.target_position = path[self.current_waypoint_idx]
        self.movement_start_pos = self.current_position
        
        dx = self.target_position[0] - self.current_position[0]
        dy = self.target_position[1] - self.current_position[1]
        self.movement_distance = np.sqrt(dx**2 + dy**2)
        self.current_speed = 0.0
        
        self._transition_to_state(CafeState.NAVIGATE_MOVE_TO_WAYPOINT)
        print(f"  Moving to waypoint {self.current_waypoint_idx}: ({self.target_position[0]:.2f}, {self.target_position[1]:.2f}), distance: {self.movement_distance:.2f}m")
    
    def navigate_move_to_waypoint_state(self):
        """Move to waypoint with acceleration/deceleration profile."""
        t = self.env.simulator.get_context().get_time()
        dt = t - self.state_start_time

        distance_traveled = min(self.max_speed * dt, self.movement_distance)
    
        # Calculate current position
        progress = distance_traveled / self.movement_distance if self.movement_distance > 0 else 1.0
        dx_total = self.target_position[0] - self.movement_start_pos[0]
        dy_total = self.target_position[1] - self.movement_start_pos[1]
        
        current_x = self.movement_start_pos[0] + progress * dx_total
        current_y = self.movement_start_pos[1] + progress * dy_total
        
        # Velocity
        if distance_traveled >= self.movement_distance:
            vx, vy = 0.0, 0.0  # Stop
        else:
            direction_x = dx_total / self.movement_distance if self.movement_distance > 0 else 0
            direction_y = dy_total / self.movement_distance if self.movement_distance > 0 else 0
            vx = direction_x * self.max_speed
            vy = direction_y * self.max_speed
        
        self.current_position = (current_x, current_y)
        self._update_robot_base_position(current_x, current_y, vx, vy)
        
        if distance_traveled >= self.movement_distance:
            self.current_waypoint_idx += 1
            
            if self.current_path_idx >= len(self.all_paths):
                return
            path = self.all_paths[self.current_path_idx]
            if self.current_waypoint_idx < len(path):
                self._start_rotation_to_waypoint()
            else:
                print(f"  Path {self.current_path_idx + 1} complete")
                self._start_table_approach()        

    def _start_table_approach(self):
        """Start the table approach sequence."""
        self.nearest_table_center = find_nearest_table(self.table_centers, self.current_position)
        
        if self.nearest_table_center is None:
            self._advance_after_navigation()
            return
        
        dx = self.nearest_table_center[0] - self.current_position[0]
        dy = self.nearest_table_center[1] - self.current_position[1]
        angle_to_table = np.arctan2(dy, dx)
        self.table_facing_angle = normalize_angle(angle_to_table - np.pi / 2)
        self.target_theta = self.table_facing_angle 
        
        self._transition_to_state(CafeState.NAVIGATE_ROTATE_TO_TABLE)
        print(f"  Approaching table at ({self.nearest_table_center[0]:.2f}, {self.nearest_table_center[1]:.2f})")
        print(f"  Target orientation: {np.degrees(self.table_facing_angle):.1f}°")

    def navigate_rotate_to_table_state(self):
        """Rotate to face table center + 90 degrees."""
        t = self.env.simulator.get_context().get_time()
        dt_elapsed = t - self.state_start_time
        
        plant = self.env.plant
        plant_context = self.env.plant_context
        iiwa_model = self.env.iiwa_model
        q_current = plant.GetPositions(plant_context, iiwa_model)
        actual_theta = q_current[3]

        delta = self.target_theta - actual_theta
        angle_diff = normalize_angle(delta)

        rotation_sign = 1
        if angle_diff < 0:
            rotation_needed = 2 * np.pi + angle_diff
        else:
            rotation_needed = angle_diff

        if rotation_needed > 3 * np.pi / 2:
            rotation_sign = -1
            rotation_needed = 2 * np.pi - rotation_needed

        total_rotation_time = rotation_needed / self.rotation_speed

        if rotation_needed < self.rotation_tolerance and dt_elapsed >= total_rotation_time:
            self._update_robot_base_orientation(actual_theta, 0.0) # used to be target theta, sus fix, dgafery
            self._start_slide_left()
            print(f"    Rotation complete: took {dt_elapsed:.2f}s")
        else:
            omega = rotation_sign * self.rotation_speed
            
            lookahead_angle = omega * self.dt
            desired_theta = actual_theta + lookahead_angle
            
            self._update_robot_base_orientation(desired_theta, omega)
    
    def _start_slide_left(self):
        plant = self.env.plant
        plant_context = self.env.plant_context
        iiwa_model = self.env.iiwa_model
        q_current = plant.GetPositions(plant_context, iiwa_model)
        actual_theta = q_current[3]

        slide_angle = actual_theta + np.pi / 2
        self.slide_direction = (np.cos(slide_angle), np.sin(slide_angle))
        self.slide_start_pos = self.current_position

        self._transition_to_state(CafeState.NAVIGATE_SLIDE_LEFT)
        print(f"  Sliding left by {self.slide_distance}m")
    
    def navigate_slide_left_state(self):
        """Slide left perpendicular to current orientation."""
        t = self.env.simulator.get_context().get_time()
        dt = t - self.state_start_time
        distance_traveled = min(self.max_speed * dt, self.slide_distance)

        if distance_traveled >= self.slide_distance:
            vx, vy = 0.0, 0.0  # Stop
        else:
            vx = self.slide_direction[0] * self.max_speed
            vy = self.slide_direction[1] * self.max_speed

        current_x = self.slide_start_pos[0] + self.slide_direction[0] * distance_traveled
        current_y = self.slide_start_pos[1] + self.slide_direction[1] * distance_traveled
        
        self.current_position = (current_x, current_y)
        self._update_robot_base_position(current_x, current_y, vx, vy)
        
        if distance_traveled >= self.slide_distance:
            # Slide complete, transition to perception/pick
            self._transition_to_state(CafeState.PERCEPTION)
    
    def _start_slide_right(self):
        """Start sliding right to return to original position."""
        slide_angle = self.current_theta - np.pi / 2
        self.slide_direction = (np.cos(slide_angle), np.sin(slide_angle))
        self.slide_start_pos = self.current_position
        
        self._transition_to_state(CafeState.NAVIGATE_SLIDE_RIGHT)
        print(f"  Sliding right by {self.slide_distance}m")
    
    def navigate_slide_right_state(self):
        """Slide right to return to position before pick/place."""
        t = self.env.simulator.get_context().get_time()
        dt = t - self.state_start_time
        distance_traveled = min(self.max_speed * dt, self.slide_distance)
        
        current_x = self.slide_start_pos[0] + self.slide_direction[0] * distance_traveled
        current_y = self.slide_start_pos[1] + self.slide_direction[1] * distance_traveled
        
        self.current_position = (current_x, current_y)
        self._update_robot_base_position(current_x, current_y)
        
        if distance_traveled >= self.slide_distance:
            # Slide complete, move to next path
            self._advance_after_navigation()
    
    def _advance_after_navigation(self):
        """Move to next path after completing current one."""
        self.current_path_idx += 1
        self.current_waypoint_idx = 0
        
        if self.current_path_idx >= len(self.all_paths):
            self._transition_to_state(CafeState.IDLE)
            print(f"  All paths complete!")
        else:
            print(f"\n=== Starting path {self.current_path_idx + 1}/{len(self.all_paths)} ===")
            self._start_rotation_to_waypoint()
    
    def _update_robot_base_position(self, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
        """
        Update the robot's base position using desired state with velocity.
        
        Args:
            x, y: Target position
            vx, vy: Velocity (prevents teleporting!)
        """
        plant = self.env.plant
        plant_context = self.env.plant_context
        iiwa_model = self.env.iiwa_model
        cmd_source = self.env.cmd_source
        diagram = self.env.diagram
        context = self.env.context
        
        q_current = plant.GetPositions(plant_context, iiwa_model)
        
        # Build desired positions, holding arm joints fixed during navigation
        if self.nav_hold_q is not None:
            q_desired = self.nav_hold_q.copy()
            q_desired[3] = q_current[3]  # keep current yaw; updated elsewhere
        else:
            q_desired = q_current.copy()
        q_desired[0] = x
        q_desired[1] = y
        q_desired[2] = 0.1

        v_desired = np.zeros(10)
        v_desired[0] = vx
        v_desired[1] = vy

        cmd_source.set_q_desired(q_desired, v_desired)
        diagram.ForcedPublish(context)
    
    def _update_robot_base_orientation(self,  theta: float, omega: float = 0.0):
        """Update the robot's base orientation."""
        plant = self.env.plant
        plant_context = self.env.plant_context
        iiwa_model = self.env.iiwa_model
        cmd_source = self.env.cmd_source
        diagram = self.env.diagram
        context = self.env.context

        q_current = plant.GetPositions(plant_context, iiwa_model)
        if self.nav_hold_q is not None:
            q_desired = self.nav_hold_q.copy()
            q_desired[0] = q_current[0]
            q_desired[1] = q_current[1]
        else:
            q_desired = q_current.copy()
        q_desired[3] = theta

        v_desired = np.zeros(10)
        v_desired[3] = omega

        cmd_source.set_q_desired(q_desired, v_desired)
        diagram.ForcedPublish(context)

        self.current_theta = theta