"""
Object detection and grasp pose computation using point cloud segmentation and ICP.
"""

import numpy as np
from pydrake.all import Rgba, RigidTransform
from pydrake.geometry import Box

from perception.object_segmentation import (
    build_pointcloud,
    segment_objects_clustering,
    ObjectDetector,
)


def detect_and_locate_object(
    diagram,
    context,
    meshcat,
    target_object="mug",
    dbscan_eps=0.03,
    dbscan_min_samples=50,
    grasp_offset=0.00,
):
    """
    Detect objects in scene and locate the target object using segmentation + ICP.

    Args:
        diagram: Drake diagram with camera point cloud outputs
        context: Diagram context
        meshcat: Meshcat visualizer
        target_object: Name of object to find ("mug", "gelatin_box", etc.)
        dbscan_eps: DBSCAN clustering epsilon parameter
        dbscan_min_samples: DBSCAN minimum samples per cluster
        grasp_offset: Z-offset from top of object for grasp position (meters)

    Returns:
        X_WO: RigidTransform of object pose in world frame
        grasp_center_xyz: (x, y, z) position for grasp center
        object_top_z: z-coordinate of object's top surface
    """

    print("\n=== OBJECT DETECTION ===")

    # 1. Build point cloud from cameras
    print("Building point cloud from cameras...")
    pc = build_pointcloud(diagram, context)
    print(f"Total points after filtering: {pc.xyzs().shape[1]}")

    # 2. Segment into individual objects using DBSCAN
    print(f"Segmenting objects (eps={dbscan_eps}, min_samples={dbscan_min_samples})...")
    object_clouds = segment_objects_clustering(
        pc, eps=dbscan_eps, min_samples=dbscan_min_samples
    )
    print(f"Found {len(object_clouds)} object clusters")

    if len(object_clouds) == 0:
        raise RuntimeError("No objects detected in scene!")

    # 3. Match each cluster to known objects using ICP
    print(f"\nMatching objects to templates...")
    detector = ObjectDetector()

    best_match = None
    best_match_score = float("inf")
    best_match_pose = None
    best_match_cloud = None

    # Visualize detected clusters
    colors = [Rgba(1, 0, 0), Rgba(0, 1, 0), Rgba(0, 0, 1), Rgba(1, 1, 0)]

    for i, obj_cloud in enumerate(object_clouds):
        # Optional: Visualize this cluster
        meshcat.SetObject(
            f"detection/cluster_{i}",
            obj_cloud,
            point_size=0.01,
            rgba=colors[i % len(colors)],
        )

        # Match against templates
        print(f"\nCluster {i}:")
        name, pose, score = detector.match_object(obj_cloud)
        print(f"  Best match: {name} (score: {score:.6f})")

        # Check if this is our target object and has best score
        if name == target_object and score < best_match_score:
            best_match = name
            best_match_score = score
            best_match_pose = pose
            best_match_cloud = obj_cloud
            print(f"  ✓ New best {target_object} match!")

    if best_match is None:
        raise RuntimeError(f"Could not find {target_object} in scene!")

    print(f"\n✓ Found {target_object} with score {best_match_score:.6f}")

    # 4. Compute grasp pose from ICP result
    X_WO = best_match_pose  # Object pose in world frame

    # Get the matched object's point cloud in world frame
    obj_xyz = best_match_cloud.xyzs()  # (3, N) in world frame

    # Find top of object
    object_top_z = float(np.max(obj_xyz[2, :]))

    # Find center in x-y plane (use centroid)
    center_x = float(np.mean(obj_xyz[0, :]))
    center_y = float(np.mean(obj_xyz[1, :]))

    # Grasp position: centered above object, at top surface + offset
    grasp_center_xyz = np.array([center_x, center_y, object_top_z + grasp_offset])

    print(f"\nGrasp computation:")
    print(f"  Object center (x,y): ({center_x:.3f}, {center_y:.3f})")
    print(f"  Object top z: {object_top_z:.3f}")
    print(f"  Grasp position: {grasp_center_xyz}")

    # Visualize grasp target
    meshcat.SetObject(
        "grasp/target",
        Box(0.02, 0.02, 0.02),
        Rgba(1, 0, 1, 0.8),
    )
    meshcat.SetTransform(
        "grasp/target",
        RigidTransform(grasp_center_xyz),
    )

    return X_WO, grasp_center_xyz, object_top_z
