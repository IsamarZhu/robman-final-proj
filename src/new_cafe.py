from pathlib import Path
from time import sleep

from pydrake.all import (
    StartMeshcat,
    Simulator,
    DiagramBuilder,
    ConstantVectorSource,
)
from manipulation.station import LoadScenario, MakeHardwareStation
from temp_perception import perceive_tables
from find_path import Grid, AStarPlanner, PathFollower, add_obstacles

# for debugging
# from viz import (
#     visualize_grid_in_meshcat, 
#     visualize_robot_config, 
#     visualize_path
# )
import numpy as np

def simulate_scenario():
    meshcat = StartMeshcat()
    scenario_file = Path("/workspaces/robman-final-proj/src/new_scenario.yaml")
    scenario = LoadScenario(filename=str(scenario_file))
    builder = DiagramBuilder()
    station = builder.AddSystem(MakeHardwareStation(
        scenario=scenario,
        meshcat=meshcat,
    ))
    
    # Initial base positions (x, y, z) + 7 arm joints = 10 total
    initial_position = (1.4, 0.0)  
    arm_positions = [1.94, 0.1, 0.0, -0.9, 0.6, 1.7, 0.0]  # 7 arm joints

    path_follower = builder.AddSystem(
        PathFollower(initial_position, arm_positions, speed=0.5)
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
    
    # grid for A*
    grid = Grid(
        x_min=-5.0, 
        x_max=5.0, 
        y_min=-5.0, 
        y_max=5.0, 
        resolution=0.1  # 10cm cells
    )
    add_obstacles(grid, tables)
    
    # (x, y, theta)
    start_config = (1.4, 0.0, 0.0)
    goal_config = (-2.0, 0.0, np.pi/2)


    # # for debugging
    # visualize_grid_in_meshcat(grid, meshcat, show_grid_lines=True)
    # visualize_robot_config(meshcat, *start_config, label="start", 
    #                       color=Rgba(0.0, 1.0, 0.0, 0.6))
    # visualize_robot_config(meshcat, *goal_config, label="goal", 
    #                       color=Rgba(1.0, 0.0, 0.0, 0.6))
    
    # path using A*
    planner = AStarPlanner(grid)
    start_xy = (start_config[0], start_config[1])
    goal_xy = (goal_config[0], goal_config[1])
    
    path = planner.plan(start_xy, goal_xy)
    
    if path:
        smoothed_path = planner.smooth_path(path)
        path_follower.add_path(smoothed_path)
        # print(f"smooth path w {len(smoothed_path)} waypoints")
        
        # for debugging
        # visualize_path(meshcat, path, color=Rgba(0.5, 0.5, 1.0, 0.5), 
        #               label="raw_path", show_orientations=False)
        # visualize_path(meshcat, smoothed_path, color=Rgba(0.0, 0.0, 1.0, 0.9), 
        #               label="smoothed_path", show_orientations=True)
        
        
        total_sim_time = 30.0
        diagram.ForcedPublish(context)
        meshcat.StartRecording()
        simulator.AdvanceTo(total_sim_time)
        meshcat.StopRecording()
        meshcat.PublishRecording()
    else:
        print("No path found!")
        diagram.ForcedPublish(context)
        meshcat.StartRecording()
        simulator.AdvanceTo(10.0)
        meshcat.StopRecording()
        meshcat.PublishRecording()
    
    while True:
        sleep(1)


if __name__ == "__main__":
    simulate_scenario()