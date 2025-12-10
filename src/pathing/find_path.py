    
from pathing.grid import Grid
from pathing.astar import AStarPlanner
from pathing.add_obstacles import add_obstacles
from pathing.viz import visualize_grid_in_meshcat, visualize_path_grid_cells, visualize_single_path
from pydrake.geometry import Rgba

def plan_paths(tables, obstacles, start_config, to_visit, meshcat=None):
    grid = Grid(
        x_min=-5.0,
        x_max=5.0,
        y_min=-5.0,
        y_max=5.0,
        resolution=0.1  # 10cm cells
    )
    add_obstacles(grid, tables, obstacles)
    # COMMENT OUT IF TOO SLOW (MAGGIE BING BONG)
    if meshcat is not None:
        visualize_grid_in_meshcat(grid, meshcat, show_grid_lines=True)
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
        
        # Visualize unsmoothed path as blue grid cells
        if meshcat is not None:
            visualize_path_grid_cells(meshcat, path, grid, Rgba(0.0, 0.0, 1.0, 0.6), f"path_raw_{i}")
        
        # Smooth the path
        smoothed_path = planner.smooth_path(path)
        
        # Visualize smoothed path as red line with green waypoints
        if meshcat is not None:
            visualize_single_path(meshcat, smoothed_path, Rgba(1.0, 0.0, 0.0, 0.9), f"path_smoothed_{i}", show_orientations=False)
        
        all_paths.append(smoothed_path)
        
        # Update for next iteration
        current_pos = goal_xy
        
        print(f"Path {i+1}: {len(path)} waypoints (raw) -> {len(smoothed_path)} waypoints (smoothed)")
    
    return all_paths