import cv2
import numpy as np
from pathlib import Path
from PIL import Image


TABLE_INTENSITY_MIN = 90
TABLE_INTENSITY_MAX = 100
SIZE_TOLERANCE = 0.07

# table size in pixels
TABLE_WIDTH = 45
TABLE_LENGTH = 66
TABLE_DEPTH = 11.248

def detect_tables_from_img(depth_array):
    """
    filter by depth intensity and then by expected size
    """
    depth_array_vis = np.copy(depth_array)
    max_valid_depth = 15.0
    depth_array_vis[np.isinf(depth_array_vis)] = max_valid_depth

    depth_min = np.min(depth_array_vis)
    depth_max = np.max(depth_array_vis)
    if depth_max > depth_min:
        depth_normalized = 255 - ((depth_array_vis - depth_min) / (depth_max - depth_min) * 255)
    else:
        depth_normalized = np.zeros_like(depth_array_vis)

    depth_image = Image.fromarray(depth_normalized.astype(np.uint8))
    depth_output_path = Path("/workspaces/robman-final-proj/original_depth.png")
    depth_image.save(depth_output_path)
    print(f"Cropped depth image saved to {depth_output_path}")
    
    thresh = np.where((depth_normalized > TABLE_INTENSITY_MIN) & (depth_normalized <= TABLE_INTENSITY_MAX), 255, 0).astype(np.uint8)
    kernel_small = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # Save thresholded image for debugging
    thresh_output_path = Path("/workspaces/robman-final-proj/threshold_debug.png")
    cv2.imwrite(str(thresh_output_path), thresh)
    
    # Find contours of individual tables
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_features = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 2000:
            rect = cv2.minAreaRect(cnt)
            (cx, cy), (w, h), angle = rect
            dim1, dim2 = sorted([w, h])
            mask = np.zeros(depth_array.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], 0, 255, -1)
            mean_depth = np.mean(depth_array[mask > 0])
            
            contour_features.append({
                'contour': cnt,
                'area': area,
                'dim1': dim1,
                'dim2': dim2,
                'depth': mean_depth,
                'rect': rect
            })
    
    if not contour_features:
        print("No contours found!")
        table_contours = []
        
    # Filter: keep only objects that match table size
    table_contours = []
    
    
    for feat in contour_features:
        dim1_diff = abs(feat['dim1'] - TABLE_WIDTH) / TABLE_WIDTH
        dim2_diff = abs(feat['dim2'] - TABLE_LENGTH) / TABLE_LENGTH
        dim1_match = dim1_diff < SIZE_TOLERANCE
        dim2_match = dim2_diff < SIZE_TOLERANCE
        
        if dim1_match and dim2_match:
            table_contours.append(feat['contour'])

    # for debugging
    result_img = cv2.cvtColor(depth_normalized.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    ##############################
    # Draw both table and non-table contours
    ##############################
    debug_img = cv2.cvtColor(thresh.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    table_ids = {id(cnt) for cnt in table_contours}

    for cnt in contours:   # <--- use original contours
        # classify based on identity
        is_table = id(cnt) in table_ids
        color = (0, 255, 0) if is_table else (0, 0, 255)

        cv2.drawContours(debug_img, [cnt], -1, color, 2)


    labeled_output = Path("/workspaces/robman-final-proj/table_vs_non_table.png")
    cv2.imwrite(str(labeled_output), debug_img)

    print(f"Labeled contours saved to: {labeled_output}")

    tables_info = []
    for i, contour in enumerate(table_contours):
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        (cx, cy), (w, h), angle = rect
        area = cv2.contourArea(contour)
        color = (0, 255, 0)

        waypoints_pixel = [
            (box[0] + box[1]) / 2,  # midpoint of edge 0-1
            (box[1] + box[2]) / 2,  # midpoint of edge 1-2
            (box[2] + box[3]) / 2,  # midpoint of edge 2-3
            (box[3] + box[0]) / 2   # midpoint of edge 3-0
        ]

        # for debugging
        cv2.drawContours(result_img, [box], 0, color, 2)
        cv2.drawContours(result_img, [contour], 0, (255, 0, 0), 1)
        
        # label
        label = f"Table {i+1}"
        cv2.putText(result_img, label, (int(cx), int(cy)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        for point in waypoints_pixel:
            cv2.circle(result_img, tuple(point.astype(int)), 3, (0, 255, 255), -1)
        
        tables_info.append({
            'id': i + 1,
            'center': (cx, cy),
            'size': (w, h),
            'angle': angle,
            'area': area,
            'box_corners': box,
            'waypoints': waypoints_pixel
        })
    
    # Table detection result
    output_path = Path("/workspaces/robman-final-proj/table_detection.png")
    cv2.imwrite(str(output_path), result_img)
    print(f"\nDetection result saved to: {output_path}")
    
    return tables_info
