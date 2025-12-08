import numpy as np
from pydrake.all import Rgba, RigidTransform
from pydrake.geometry import Box

from perception.object_segmentation import (
    build_pointcloud,
    segment_objects_clustering,
    ObjectDetector,
)


def detect_and_locate_object(
    scenario_number,
    diagram,
    context,
    meshcat,
    target_object="mug",
    dbscan_eps=0.01,
    dbscan_min_samples=50,
    grasp_offset=0.00,
):
    """
    detect objects in scene and locate the target object using segmentation + ICP
    and returns the transformation of the object in the world frame along with
    the computed grasp position and segmented point cloud (with normals for antipodal grasping)
    """

    pc = build_pointcloud(diagram, context)

    object_clouds = segment_objects_clustering(
        pc, eps=dbscan_eps, min_samples=dbscan_min_samples
    )
    print(f"found {len(object_clouds)} object clusters {dbscan_eps}")

    if len(object_clouds) == 0:
        raise RuntimeError("no objects detected in scene!")

    detector = ObjectDetector(scenario_number)

    best_match = None
    best_match_score = float("inf")
    best_match_pose = None
    best_match_cloud = None

    colors = [Rgba(1, 0, 0), Rgba(0, 1, 0), Rgba(0, 0, 1)]

    for i, obj_cloud in enumerate(object_clouds):
        meshcat.SetObject(
            f"detection/cluster_{i}",
            obj_cloud,
            point_size=0.01,
            # rgba=colors[i],
            rgba=colors[i % len(colors)],  # Wrap around if more than 6 clusters
        )

        print(f"\ncluster {i}:")
        name, pose, score = detector.match_object(obj_cloud)
        print(f"  best match: {name} (score: {score:.6f})")

        if name == target_object and score < best_match_score:
            best_match = name
            best_match_score = score
            best_match_pose = pose
            best_match_cloud = obj_cloud

    if best_match is None:
        raise RuntimeError(f"could not find {target_object} in scene")

    print(f"\nfound {target_object} with score {best_match_score:.6f}")

    X_WO = best_match_pose 

    obj_xyz = best_match_cloud.xyzs()
    object_top_z = float(np.max(obj_xyz[2, :]))
    object_bottom_z = float(np.min(obj_xyz[2, :]))

    center_x = float(np.mean(obj_xyz[0, :]))
    center_y = float(np.mean(obj_xyz[1, :]))
    grasp_center_xyz = np.array([center_x, center_y, object_top_z + grasp_offset])

    # debug logging
    print(f"  Object position: x={center_x:.3f}, y={center_y:.3f}, z_range=[{object_bottom_z:.3f}, {object_top_z:.3f}]")
    print(f"  Grasp center: {grasp_center_xyz}")

    meshcat.SetObject(
        "grasp/target",
        Box(0.02, 0.02, 0.02),
        Rgba(1, 0, 1, 0.8),
    )
    meshcat.SetTransform(
        "grasp/target",
        RigidTransform(grasp_center_xyz),
    )

    # Estimate normals for antipodal grasping
    best_match_cloud.EstimateNormals(radius=0.02, num_closest=30)

    return X_WO, grasp_center_xyz, object_top_z, best_match_cloud
