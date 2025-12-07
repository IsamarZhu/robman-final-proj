    
from pathing.grid import Grid
from pathing.astar import AStarPlanner
from pathing.add_obstacles import add_obstacles

def plan_paths(tables, obstacles, start_config, to_visit):
    grid = Grid(
        x_min=-5.0,
        x_max=5.0,
        y_min=-5.0,
        y_max=5.0,
        resolution=0.1  # 10cm cells
    )
    add_obstacles(grid, tables, obstacles)
    # visualize_grid_in_meshcat(grid, env.meshcat, show_grid_lines=True)
    # Plan paths to each table
    planner = AStarPlanner(grid)
    all_paths = []
    
    current_pos = (start_config[0], start_config[1])
    for i, goal in enumerate(to_visit):
        goal_xy = (goal[0], goal[1])
        
        # Plan path
        path = planner.plan(current_pos, goal_xy)
        
        if not path:
            print(f"ERROR: No path found to table {i}!")
            return
        
        # Smooth the path
        smoothed_path = planner.smooth_path(path)
        all_paths.append(smoothed_path)
        
        # Update for next iteration
        current_pos = goal_xy
        
        print(f"Path {i+1}: {len(path)} waypoints (raw) -> {len(smoothed_path)} waypoints (smoothed)")
    
    return all_paths