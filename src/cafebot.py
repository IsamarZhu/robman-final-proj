import argparse
import numpy as np
from setup import SimulationEnvironment
from pathing.find_path import plan_paths
from perception.top_level_perception import perceive_scene
from pydrake.geometry import Rgba
from pathing.viz import (
    visualize_robot_config,
    visualize_grid_in_meshcat,
    visualize_scene_detection
)
from state_machine.state_machine import CafeStateMachine

# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="specific pick and place simulation")
    parser.add_argument("scenario", choices=["one", "two", "three"], default="one")
    args = parser.parse_args()

    env = SimulationEnvironment(args.scenario, wsg_open=0.107)
    env.build()

    # for debugging
    env.meshcat.SetCameraPose(
        np.array([1, 8, 2]).reshape(3, 1),      # Camera position
        np.array([0, 0, 0.5]).reshape(3, 1)      # Look at slightly above ground
    )

    env.meshcat.StartRecording(frames_per_second = 15) # lowered fps because of memroy issue
    station = env.station
    station_context = station.GetMyContextFromRoot(env.context)
    
    tables, obstacles = perceive_scene(station, station_context)
    visualize_scene_detection(station, station_context, tables, obstacles)
    start_config = env.plant.GetPositions(env.plant_context, env.iiwa_model)
    table_centers = [table['center_world'] for table in tables]
    # print(args.scenario)
    # Define navigation goals (positions at table edges)
    if args.scenario == "one":
        to_visit = [
            tables[0]['waypoints_padded'][1],
            tables[1]['waypoints_padded'][2],
            tables[2]['waypoints_padded'][0], # not 1, 2
        ]
    elif args.scenario == "two":
        to_visit = [
            tables[2]['waypoints_padded'][2], #1 works, not 2, 0
            # tables[1]['waypoints_padded'][2],
            # tables[2]['waypoints_padded'][1],
        ]
    all_paths = plan_paths(tables, obstacles, start_config, to_visit, env.meshcat)
    state_machine = CafeStateMachine(
        env=env,
        # Pick/place parameters
        approach_height=0.15,
        lift_height=0.20,
        grasp_offset=0.00,
        wsg_open=0.107,
        wsg_closed=0.015,
        move_time=2.5,
        grasp_time=2.0,
        lift_time=4.0,
        # Navigation parameters
        paths=all_paths,
        table_centers=table_centers,
        max_speed=0.6,
        rotation_speed=0.5,
        acceleration=0.3,
        deceleration=0.3,
    )
     # Set paths explicitly (even though passed in constructor)
    state_machine.set_paths(all_paths)
    state_machine.set_table_centers(table_centers)
    state_machine.run()
    env.meshcat.StopRecording()
    env.meshcat.PublishRecording()


if __name__ == "__main__":
    main()
