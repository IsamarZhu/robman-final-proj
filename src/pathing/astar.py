import heapq
import numpy as np
from typing import List, Tuple, Optional
from pathing.grid import Grid

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
    
    def _find_nearest_free_cell(self, center: Tuple[int, int], max_radius: int = 5) -> Optional[Tuple[int, int]]:
        if not self.grid.is_occupied(*center):
            return center

        for radius in range(1, max_radius + 1):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:
                        candidate = (center[0] + dx, center[1] + dy)
                        if self.grid.is_valid(*candidate) and not self.grid.is_occupied(*candidate):
                            return candidate
        return None

    def plan(self, start_world: Tuple[float, float],
             goal_world: Tuple[float, float]) -> Optional[List[Tuple[float, float]]]:
        start_grid = self.grid.world_to_grid(*start_world)
        goal_grid = self.grid.world_to_grid(*goal_world)

        if self.grid.is_occupied(*start_grid):
            start_grid = self._find_nearest_free_cell(start_grid)
            if start_grid is None:
                print(f"Start position {start_world} and nearby cells are all occupied!")
                return None

        if self.grid.is_occupied(*goal_grid):
            goal_grid = self._find_nearest_free_cell(goal_grid)
            if goal_grid is None:
                print(f"Goal position {goal_world} and nearby cells are all occupied!")
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

