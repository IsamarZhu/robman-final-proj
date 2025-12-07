import numpy as np

def normalize_angle(angle):
    """Normalize angle to be within [-pi, pi] of current angle."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle

def find_nearest_table(table_centers, position):
        """Find the nearest table center to the given position."""
        if not table_centers:
            return None
        
        min_dist = float('inf')
        nearest_table = None
        
        for table_center in table_centers:
            dx = table_center[0] - position[0]
            dy = table_center[1] - position[1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist < min_dist:
                min_dist = dist
                nearest_table = table_center
        
        return nearest_table