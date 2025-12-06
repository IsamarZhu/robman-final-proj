import numpy as np
from typing import Tuple
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
