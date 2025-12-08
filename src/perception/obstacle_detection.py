import cv2
import numpy as np
from pathlib import Path
import perception.config as config
def detect_obstacles_from_img(depth_array):
    """
    Detect obstacles (unknown objects) by finding regions that are above ground
    
    Returns:
        List of obstacle dictionaries with pixel and world coordinates
    """
    depth_array_vis = np.copy(depth_array)
    max_valid_depth = 15.0
    depth_array_vis[np.isinf(depth_array_vis)] = max_valid_depth

    depth_min = np.min(depth_array_vis)
    depth_max = np.max(depth_array_vis)
    print(f"DEPTH MAX: {depth_max}")
    if depth_max > depth_min:
        depth_normalized = 255 - ((depth_array_vis - depth_min) / (depth_max - depth_min) * 255)
    else:
        depth_normalized = np.zeros_like(depth_array_vis)
    
    obstacle_thresh = np.where(depth_normalized > 1, 255, 0).astype(np.uint8)

    exclusion_mask = np.ones(depth_array.shape, dtype=np.uint8) * 255
    height, width = depth_array.shape
    # Set upper right corner to 0 (exclude this region)

    # Draw the exclusion box (set it to 0/black to visualize)
    cv2.rectangle(exclusion_mask, 
                  (width - config.CORNER_EXCLUSION_Y, 0), 
                  (width, config.CORNER_EXCLUSION_X), 
                  0, 
                  -1)
    exclusion_mask[0:config.CORNER_EXCLUSION_X, width-config.CORNER_EXCLUSION_Y:width] = 0
    
    # Apply exclusion mask
    obstacle_thresh = cv2.bitwise_and(obstacle_thresh, exclusion_mask)
    
    # Morphological operations to clean noise and connect nearby regions
    kernel = np.ones((5, 5), np.uint8)
    obstacle_thresh = cv2.morphologyEx(obstacle_thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    obstacle_thresh = cv2.morphologyEx(obstacle_thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Save obstacle threshold for debugging
    obstacle_output_path = Path("/workspaces/robman-final-proj/obstacle_threshold_debug.png")
    cv2.imwrite(str(obstacle_output_path), obstacle_thresh)
    
    # Find contours of obstacles
    contours, _ = cv2.findContours(obstacle_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    obstacles_info = []
    MIN_OBSTACLE_AREA = 2  # Minimum pixel area to consider as obstacle
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area < MIN_OBSTACLE_AREA:
            continue
        
        # Get bounding information
        rect = cv2.minAreaRect(contour)
        (cx, cy), (w, h), angle = rect
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        
        # Calculate average depth for this obstacle
        mask = np.zeros(depth_array.shape, dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, -1)
        obstacle_depth_values = depth_array[mask > 0]
        mean_depth = np.mean(obstacle_depth_values[np.isfinite(obstacle_depth_values)])
        
        # Create convex hull for better representation
        hull = cv2.convexHull(contour)
        
        obstacles_info.append({
            'id': i + 1,
            'center': (cx, cy),
            'size': (w, h),
            'angle': angle,
            'area': area,
            'box_corners': box,
            'contour': contour,
            'hull': hull,
            'mean_depth': mean_depth
        })
    
    return obstacles_info

