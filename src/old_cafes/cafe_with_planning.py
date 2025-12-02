"""
Robot base motion planning with RRT.

Simulates the robot base moving from a random starting position to the target
position while avoiding obstacles (table, chair, person).

The simulation shows the robot box moving along the planned path in meshcat.
"""

from pathlib import Path
import time
import numpy as np
from pydrake.all import (
    DiagramBuilder,
    MultibodyPlant,
    Parser,
    PiecewisePolynomial,
    RigidTransform,
    RotationMatrix,
    Simulator,
    StartMeshcat,
    SceneGraph,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    AddMultibodyPlantSceneGraph,
    SpatialVelocity,
)
from pydrake.multibody.parsing import LoadModelDirectives, ProcessModelDirectives
from pydrake.systems.primitives import ConstantVectorSource

from manipulation import running_as_notebook


class RobotBaseMotionPlanner:
    """
    Simple RRT-based motion planner for the robot base to navigate
    from a random start position to a goal position while avoiding obstacles.
    """
    
    def __init__(self, obstacles, bounds):
        """
        Args:
            obstacles: List of obstacle dicts with 'center' and 'size' (half-extents)
            bounds: Dict with 'x_min', 'x_max', 'y_min', 'y_max' for workspace bounds
        """
        self.obstacles = obstacles
        self.bounds = bounds
        self.robot_size = np.array([0.25, 0.5])  # robot base half-extents: 0.5 x 1.0 x 0.5 full size from robot_base.sdf
        
    def is_collision_free(self, pos):
        """
        Check if robot base position (x, y) collides with any obstacle.
        Uses axis-aligned bounding box (AABB) collision detection.
        
        Args:
            pos: [x, y] position of robot base center
        """
        # Check workspace bounds
        if (pos[0] < self.bounds['x_min'] or pos[0] > self.bounds['x_max'] or
            pos[1] < self.bounds['y_min'] or pos[1] > self.bounds['y_max']):
            return False
            
        # Check collision with each obstacle
        for obs in self.obstacles:
            obs_center = obs['center'][:2]  # only x, y
            obs_size = obs['size'][:2]
            
            # AABB collision: check if rectangles overlap
            dist = np.abs(pos - obs_center)
            min_dist = self.robot_size + obs_size
            
            if np.all(dist < min_dist):
                return False  # collision detected
                
        return True
    
    def distance(self, pos1, pos2):
        """Euclidean distance between two positions."""
        return np.linalg.norm(pos1 - pos2)
    
    def sample_free_position(self):
        """Sample a random collision-free position in the workspace."""
        max_attempts = 1000
        for _ in range(max_attempts):
            x = np.random.uniform(self.bounds['x_min'], self.bounds['x_max'])
            y = np.random.uniform(self.bounds['y_min'], self.bounds['y_max'])
            pos = np.array([x, y])
            
            if self.is_collision_free(pos):
                return pos
                
        raise RuntimeError("Could not find collision-free position after max attempts")
    
    def steer(self, from_pos, to_pos, step_size=0.1):
        """
        Steer from from_pos toward to_pos by at most step_size.
        Returns the new position.
        """
        direction = to_pos - from_pos
        dist = np.linalg.norm(direction)
        
        if dist <= step_size:
            return to_pos
        else:
            return from_pos + (direction / dist) * step_size
    
    def plan_path(self, start, goal, max_iterations=2000, step_size=0.15):
        """
        RRT planner to find a path from start to goal.
        
        Returns:
            List of [x, y] waypoints from start to goal, or None if no path found
        """
        if not self.is_collision_free(start):
            raise ValueError("Start position is in collision!")
        if not self.is_collision_free(goal):
            raise ValueError("Goal position is in collision!")
        
        # RRT tree: each node stores (position, parent_index)
        tree = [(start, None)]
        
        for i in range(max_iterations):
            # Sample: bias toward goal 10% of the time
            if np.random.random() < 0.1:
                sample = goal
            else:
                sample = self.sample_free_position()
            
            nearest_idx = min(range(len(tree)), 
                            key=lambda idx: self.distance(tree[idx][0], sample))
            nearest_pos = tree[nearest_idx][0]
            
            new_pos = self.steer(nearest_pos, sample, step_size)
            
            if not self.is_collision_free(new_pos):
                continue
            
            edge_clear = True
            n_checks = int(self.distance(nearest_pos, new_pos) / 0.05) + 1
            for t in np.linspace(0, 1, n_checks):
                intermediate = nearest_pos + t * (new_pos - nearest_pos)
                if not self.is_collision_free(intermediate):
                    edge_clear = False
                    break
            
            if not edge_clear:
                continue
            
            tree.append((new_pos, nearest_idx))
            if self.distance(new_pos, goal) < step_size:
                tree.append((goal, len(tree) - 1))
                
                path = []
                idx = len(tree) - 1
                while idx is not None:
                    path.append(tree[idx][0])
                    idx = tree[idx][1]
                
                path.reverse()
                return path
        
        return None


def get_robot_base_position_from_scenario():
    """
    Extract the robot base position from the scenario file.
    This is the target position we want the robot to reach.
    """
    # From scenario.yaml: robot_base is welded to floor at [1.4, 0, 0.4]
    return np.array([1.4, 0.0, 0.4])


def get_obstacle_list():
    """
    Define all obstacles in the scene based on scenario_planning.yaml.
    Returns list of obstacle dicts with 'center' [x, y, z] and 'size' [half_x, half_y, half_z]
    
    NOTE: When you change positions in scenario_planning.yaml, update these values to match!
    """
    obstacles = [
        # Table - from scenario_planning.yaml translation: [0.6, 0, 0.75]
        # Dimensions from table.sdf: 0.9 x 1.3125 x 0.06 (collision box) at height 0.75
        # Half-extents: 0.45 x 0.65625 x (0.75 height + 0.03 half-thickness)
        {
            'name': 'table',
            'center': np.array([0.6, 0.0, 0.75]),  # Match YAML position
            'size': np.array([0.45, 0.66, 0.78]),  # half-extents (rounded y, full height box)
        },
        # Chair - from scenario_planning.yaml translation: [-0.05, 0, 0] (welded to floor)
        {
            'name': 'chair',
            'center': np.array([-0.05, 0.0, 0.3]),
            'size': np.array([0.3, 0.3, 0.5]),
        },
        # Person - from scenario_planning.yaml at [-0.1, 0, 0] relative to chair
        {
            'name': 'person',
            'center': np.array([-0.15, 0.0, 0.6]),
            'size': np.array([0.25, 0.25, 0.4]),
        },
        # Add more obstacles here as needed
    ]
    return obstacles


def create_base_pose_trajectory(waypoints, z_height=0.4, total_time=10.0):
    """
    Create a trajectory for the robot base that moves through waypoints.
    
    The robot_base in Drake has 7 DOF (quaternion + xyz translation).
    We keep z constant and rotation identity.
    
    Args:
        waypoints: List of [x, y] positions
        z_height: Fixed z position
        total_time: Total time for trajectory
    
    Returns:
        PiecewisePolynomial trajectory with 7 values: [qw, qx, qy, qz, x, y, z]
    """
    n_waypoints = len(waypoints)
    times = np.linspace(0, total_time, n_waypoints)
    
    # Build knots: each column is [qw, qx, qy, qz, x, y, z] for one waypoint
    knots = []
    for wp in waypoints:
        # Identity quaternion: w=1, x=0, y=0, z=0
        # Position: wp[0], wp[1], z_height
        knot = [1.0, 0.0, 0.0, 0.0, wp[0], wp[1], z_height]
        knots.append(knot)
    
    knots = np.array(knots).T  # Shape: (7, n_waypoints)
    
    traj = PiecewisePolynomial.FirstOrderHold(times, knots)
    return traj


def simulate_base_motion(all_paths, target_position, meshcat, bounds):
    """
    Simulate the robot base moving along multiple planned paths in Drake.
    The robot base slides frictionlessly along the ground.
    
    Args:
        all_paths: List of tuples (path, start_position) where path is list of [x, y] waypoints
        target_position: [x, y, z] target position
        meshcat: Meshcat instance for visualization
        bounds: Dict with 'x_min', 'x_max', 'y_min', 'y_max' for workspace bounds
    """
    print("\n" + "=" * 70)
    print("SIMULATING BASE MOTION IN DRAKE")
    print("=" * 70)
    
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    
    # Load models
    print("Loading models...")
    
    # Load all models from scenario file (without model_drivers section)
    scenario_file = "src/scenario_planning.yaml"
    directives = LoadModelDirectives(scenario_file)
    models = ProcessModelDirectives(directives, plant, parser)
    
    # Get model instances by name
    robot_base = plant.GetModelInstanceByName("robot_base")
    mug = plant.GetModelInstanceByName("mug")
    
    plant.Finalize()
    
    # Add visualizer
    visualizer = MeshcatVisualizer.AddToBuilder(
        builder, scene_graph, meshcat,
        MeshcatVisualizerParams()
    )
    
    # Build diagram
    diagram = builder.Build()
    
    # Create simulator
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(context)
    
    # Visualize workspace bounds as a wireframe box
    from pydrake.geometry import Box, Rgba
    x_center = (bounds['x_min'] + bounds['x_max']) / 2
    y_center = (bounds['y_min'] + bounds['y_max']) / 2
    x_size = bounds['x_max'] - bounds['x_min']
    y_size = bounds['y_max'] - bounds['y_min']
    z_height = 2.0  # Visualization height
    
    meshcat.SetObject(
        "workspace_bounds",
        Box(x_size, y_size, z_height),
        Rgba(0.3, 0.8, 0.3, 0.15)  # Green semi-transparent
    )
    meshcat.SetTransform(
        "workspace_bounds",
        RigidTransform([x_center, y_center, z_height / 2])
    )
    
    # Get robot base body and mug body
    robot_base_body = plant.GetBodyByName("table_link", robot_base)
    mug_body = plant.GetBodyByName("base_link", mug)
    
    # Initial publish
    diagram.ForcedPublish(context)
    
    if running_as_notebook:
        simulator.set_target_realtime_rate(1.0)
    
    print(f"\nStarting simulation for {len(all_paths)} runs...")
    
    meshcat.StartRecording()
    
    # Simulate each path sequentially
    time_per_run = 10.0  # seconds per run
    fps = 30  # frames per second
    
    current_time = 0.0
    
    for run_idx, (path, start_pos) in enumerate(all_paths):
        print(f"\n  Run {run_idx + 1}: {len(path)} waypoints from [{start_pos[0]:.2f}, {start_pos[1]:.2f}] to target")
        
        frames_per_run = int(time_per_run * fps)
        
        # Animate this path
        for frame in range(frames_per_run):
            t = frame / (frames_per_run - 1)  # 0 to 1
            
            # Update the context time
            context.SetTime(current_time)
            current_time += 1.0 / fps
            
            # Find which segment we're on
            segment_progress = t * (len(path) - 1)
            segment_idx = int(segment_progress)
            
            if segment_idx >= len(path) - 1:
                segment_idx = len(path) - 2
            
            # Interpolation factor within this segment
            local_t = segment_progress - segment_idx
            
            # Linear interpolation between waypoints
            p0 = path[segment_idx]
            p1 = path[segment_idx + 1]
            
            interpolated_pos = (1 - local_t) * p0 + local_t * p1
            
            # Set robot base pose at interpolated position
            X_base = RigidTransform(
                RotationMatrix(),
                [interpolated_pos[0], interpolated_pos[1], target_position[2]]
            )
            plant.SetFreeBodyPose(plant_context, robot_base_body, X_base)
            
            # Set velocities to zero
            zero_velocity = SpatialVelocity(w=[0., 0., 0.], v=[0., 0., 0.])
            plant.SetFreeBodySpatialVelocity(robot_base_body, zero_velocity, plant_context)
            
            # Position mug on tray (relative offset from robot base)
            X_mug = RigidTransform(
                RotationMatrix.MakeZRotation(np.pi/2),  # 90 deg rotation
                [interpolated_pos[0] + 0.0, interpolated_pos[1] + 0.35, target_position[2] + 0.95]
            )
            plant.SetFreeBodyPose(plant_context, mug_body, X_mug)
            plant.SetFreeBodySpatialVelocity(mug_body, zero_velocity, plant_context)
            
            # Publish the state for visualization with proper timestamp
            visualizer.ForcedPublish(visualizer.GetMyContextFromRoot(context))
            
            # Small delay for real-time playback
            time.sleep(1.0 / fps)
        
        # Pause briefly between runs
        if run_idx < len(all_paths) - 1:
            time.sleep(0.5)
    
    print("\nSimulation complete!")
    print("Publishing recording to meshcat...")
    time.sleep(2.0)
    meshcat.PublishRecording()
    
    print("\n" + "=" * 70)
    print("Meshcat URL:", meshcat.web_url())
    print("The animation will replay automatically.")
    print("=" * 70)


def main():
    """
    Main function: 
    1. Find robot base target position
    2. Generate random starting position
    3. Plan collision-free path
    4. Simulate motion in meshcat with replay
    5. Repeat 5 times with different random starts
    """
    
    print("=" * 70)
    print("ROBOT BASE MOTION PLANNING WITH RRT")
    print("=" * 70)
    
    # Start meshcat
    meshcat = StartMeshcat()
    print(f"\nMeshcat URL: {meshcat.web_url()}")
    print("   Open this link to view the simulation in your browser")
    print()
    
    # Get target position (where robot needs to be)
    target_position = get_robot_base_position_from_scenario()
    print(f"\nTarget robot base position: {target_position}")
    
    # Define obstacles
    obstacles = get_obstacle_list()
    print(f"Number of obstacles: {len(obstacles)}")
    for obs in obstacles:
        print(f"  - {obs['name']}: center={obs['center']}, size={obs['size']}")
    
    # Define workspace bounds
    bounds = {
        'x_min': -3.0,
        'x_max': 3.0,
        'y_min': -3.0,
        'y_max': 3.0,
    }
    
    # Create planner
    planner = RobotBaseMotionPlanner(obstacles, bounds)
    
    # Generate random starting position (collision-free)
    print("\nGenerating random starting position...")
    start_position = planner.sample_free_position()
    print(f"Start position: [{start_position[0]:.2f}, {start_position[1]:.2f}]")
    print(f"Target position: [{target_position[0]:.2f}, {target_position[1]:.2f}]")
    print(f"Distance: {planner.distance(start_position, target_position[:2]):.2f} m")
    
    # Plan path
    print("\nPlanning path with RRT...")
    goal_position = target_position[:2]  # only x, y for planning
    path = planner.plan_path(start_position, goal_position)
    
    if path is None:
        print("ERROR: No path found!")
        return None, None
    
    print(f"Path found with {len(path)} waypoints!")
    
    # Simulate the motion
    simulate_base_motion([(path, start_position)], target_position, meshcat, bounds)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Robot base moved from:")
    print(f"  Start:  [{start_position[0]:.3f}, {start_position[1]:.3f}, {target_position[2]:.3f}]")
    print(f"  Target: [{target_position[0]:.3f}, {target_position[1]:.3f}, {target_position[2]:.3f}]")
    print(f"  Path length: {len(path)} waypoints")
    print(f"  Obstacles avoided: {len(obstacles)}")
    print("=" * 70)
    
    input("\nPress Enter to exit...")
    
    return path, target_position


if __name__ == "__main__":
    path, target = main()
