from pathlib import Path
from time import sleep

from pydrake.all import (
    StartMeshcat,
    Simulator,
    DiagramBuilder,
    ConstantVectorSource,
)
from manipulation.station import LoadScenario, MakeHardwareStation
from perception.perception import perceive_tables
from find_path import Grid, AStarPlanner, PathFollower, add_obstacles, estimate_runtime

# for debugging
from pydrake.geometry import Rgba
from viz import (
    visualize_grid_in_meshcat, 
    visualize_robot_config, 
    visualize_path
)
import numpy as np

# Motion parameters
MOVEMENT_SPEED = 0.5  # m/s
ROTATION_SPEED = 0.1  # rad/s
INITIAL_DELAY = 1.0   # seconds

def simulate_scenario():
    meshcat = StartMeshcat()
    scenario_file = Path("/workspaces/robman-final-proj/src/new_scenario.yaml")
    scenario = LoadScenario(filename=str(scenario_file))
    
    # Initial base positions (x, y, z) + 7 arm joints = 10 total
    # do not manually change arm_positions[0]
    arm_positions = [0.0, 0.1, 0.0, -0.9, 0.6, 1.7, 0.0]  # 7 arm joints
    
    start_config = (1.4, 0.0, 0.0)
    end_config = (1.4, 0.0, 0.0)
    
    # Create builder and station
    builder = DiagramBuilder()
    station = builder.AddSystem(MakeHardwareStation(
        scenario=scenario,
        meshcat=meshcat,
    ))
    
    # Create path follower with empty paths initially
    path_follower = builder.AddSystem(
        PathFollower(start_config[:2], arm_positions, paths=[],
                     speed=MOVEMENT_SPEED, start_theta=start_config[2], goal_theta=end_config[2],
                     rotation_speed=ROTATION_SPEED, initial_delay=INITIAL_DELAY)
    )
    
    builder.Connect(
        path_follower.get_output_port(),
        station.GetInputPort("iiwa_arm.desired_state")
    )
    
    # Also need to control the gripper
    gripper_source = builder.AddSystem(ConstantVectorSource([0.1]))  # Gripper open
    builder.Connect(
        gripper_source.get_output_port(),
        station.GetInputPort("wsg_arm.position")
    )
    
    diagram = builder.Build()
    
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    context = simulator.get_mutable_context()
    station_context = station.GetMyContextFromRoot(context)
    
    tables = perceive_tables(station, station_context)
    table_goals = [
        # tables[0]['waypoints_world'][2],
        (4, 4, np.pi/2),
        (-4, 4, np.pi/2),
        end_config
    ]

    # grid for A*
    grid = Grid(
        x_min=-5.0, 
        x_max=5.0, 
        y_min=-5.0, 
        y_max=5.0, 
        resolution=0.1  # 10cm cells
    )
    add_obstacles(grid, tables)
    
    # Plan paths to visit each table in sequence
    planner = AStarPlanner(grid)
    all_paths = []
    
    current_pos = (start_config[0], start_config[1])
    
    for i, goal in enumerate(table_goals):
        goal_xy = (goal[0], goal[1])
        
        # Plan path from current position to this goal
        path = planner.plan(current_pos, goal_xy)
        
        if not path:
            print(f"No path found to table {i}!")
            return
        
        # Smooth the path
        smoothed_path = planner.smooth_path(path)
        all_paths.append(smoothed_path)
        
        # Update current position for next iteration
        current_pos = goal_xy
        
        print(f"Path {i+1}: {len(smoothed_path)} waypoints")
    
    # Set all paths in the path follower
    path_follower.set_paths(all_paths)
    
    total_distance, total_rotation = estimate_runtime(start_config, table_goals, all_paths)
    movement_time = total_distance / MOVEMENT_SPEED
    rotation_time = total_rotation / ROTATION_SPEED
    total_sim_time = INITIAL_DELAY + movement_time + rotation_time + 2.0  # +2s buffer

    # for debugging
    # visualize_grid_in_meshcat(grid, meshcat, show_grid_lines=True)
    # visualize_robot_config(meshcat, *start_config, label="start", 
    #                       color=Rgba(0.0, 1.0, 0.0, 0.6))
    # visualize_robot_config(meshcat, *goal_config, label="goal", 
    #                       color=Rgba(1.0, 0.0, 0.0, 0.6))
    # visualize_path(meshcat, path, color=Rgba(0.5, 0.5, 1.0, 0.5), 
    #               label="raw_path", show_orientations=False)
    # visualize_path(meshcat, smoothed_path, color=Rgba(0.0, 0.0, 1.0, 0.9), 
    #               label="smoothed_path", show_orientations=True)
    
    diagram.ForcedPublish(context)
    simulator.AdvanceTo(total_sim_time)
    
    input("Press Enter to exit...")


if __name__ == "__main__":
    simulate_scenario()