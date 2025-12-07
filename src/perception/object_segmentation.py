
from pathlib import Path
import numpy as np
import trimesh

from pydrake.all import (
    Concatenate,
    PointCloud,
    RigidTransform,
    RotationMatrix,
    InverseKinematics,
    Solve,
)

from manipulation.icp import IterativeClosestPoint
from pydrake.systems.sensors import RgbdSensor, CameraInfo, PixelType
from pydrake.geometry import DepthRenderCamera, RenderCameraCore, ColorRenderCamera, ClippingRange, DepthRange
from pydrake.perception import DepthImageToPointCloud
from pydrake.math import RigidTransform
from pydrake.all import PointCloud
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.cluster import DBSCAN
import numpy as np
from manipulation import running_as_notebook, FindResource


# mesh paths
MUG_MESH_PATH = Path("/workspaces/robman-final-proj/assets/mug/google_16k/textured.obj")

MAX_ICP_ITERS = 25
VOXEL_SIZE = 0.005
PLATE_HEIGHT = 0.6
N_SAMPLE_POINTS = 1500
            
# downsamples the point cloud
def downsample(pc: PointCloud, voxel_size: float) -> PointCloud:
    if pc.xyzs().shape[1] == 0:
        return pc
    return pc.VoxelizedDownSample(voxel_size)


# removes points below z threshold
def remove_below_z(pc: PointCloud, z_thresh: float) -> PointCloud:
    xyz = pc.xyzs()
    if xyz.shape[1] == 0:
        return pc
    z = xyz[2, :]
    mask = (z > z_thresh)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return PointCloud(0)
    sel = xyz[:, idx]
    out = PointCloud(sel.shape[1])
    out.mutable_xyzs()[:] = sel
    return out

# builds a pointcloud
def build_pointcloud(diagram, context) -> PointCloud:
    pc0 = diagram.GetOutputPort("camera0.point_cloud").Eval(context)
    pc1 = diagram.GetOutputPort("camera1.point_cloud").Eval(context)
    pc2 = diagram.GetOutputPort("camera2.point_cloud").Eval(context)
    # pc3 = diagram.GetOutputPort("camera3.point_cloud").Eval(context)

    # xyz = np.concatenate([pc0.xyzs(), pc1.xyzs(), pc2.xyzs(), pc3.xyzs()], axis=1,)
    xyz = np.concatenate([pc0.xyzs(), pc1.xyzs(), pc2.xyzs()], axis=1,)
    concat_pc = PointCloud(xyz.shape[1])
    concat_pc.mutable_xyzs()[:] = xyz

    down_pc = downsample(concat_pc, VOXEL_SIZE)
    obj_pc = remove_below_z(down_pc, PLATE_HEIGHT)
    return obj_pc

# segment point cloud into individual objects using DBSCAN clustering
def segment_objects_clustering(pc: PointCloud, eps=0.03, min_samples=50):
    """
    segment point cloud into individual objects using DBSCAN clustering, returning a
    list of PointCloud objects, one per detected object
    """
    xyz = pc.xyzs().T 
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz)
    labels = clustering.labels_    
    object_clouds = []
    unique_labels = set(labels)
    
    for label in unique_labels:
        if label == -1:
            continue
        
        mask = (labels == label)
        cluster_points = xyz[mask].T 
        
        obj_pc = PointCloud(cluster_points.shape[1])
        obj_pc.mutable_xyzs()[:] = cluster_points
        object_clouds.append(obj_pc)
    
    print(f"found {len(object_clouds)} objects")
    return object_clouds



# once we have the segmented point clouds, we want to figure out what object it is
class ObjectDetector:
    def __init__(self, scenario_number):
        """
        we preload templates for the known objects that can appear
        """
        
        if scenario_number == "one":
            self.mesh_templates = {
                "mug": Path("/workspaces/robman-final-proj/assets/mug/google_16k/textured.obj"),
                "gelatin_box": Path("/workspaces/robman-final-proj/assets/009_gelatin_box/google_16k/textured.obj"),
                "tomato_soup": Path("/workspaces/robman-final-proj/assets/005_tomato_soup_can/google_16k/textured.obj")
            }
        
        elif scenario_number == "two":
            self.mesh_templates = {
                "potted_meat": Path("/workspaces/robman-final-proj/assets/010_potted_meat_can/google_16k/textured.obj"),
                "apple": Path("/workspaces/robman-final-proj/assets/apple/google_16k/textured.obj"),
                "master_chef": Path("/workspaces/robman-final-proj/assets/master_chef/google_16k/textured.obj"),   
            }
        elif scenario_number == "three":
            self.mesh_templates = {
                "pudding": Path("/workspaces/robman-final-proj/assets/008_pudding_box/google_16k/textured.obj"),
                "tuna": Path("/workspaces/robman-final-proj/assets/007_tuna_fish_can/google_16k/textured.obj"),
            }
        
        
        self.templates = {}
        for name, mesh_path in self.mesh_templates.items():
            mesh = trimesh.load(mesh_path)

            # scale down mug mesh by 0.8
            if name == "mug":
                mesh.apply_scale(0.8)
            elif name == "pudding":
                mesh.apply_scale(0.75)


            cloud_points, _ = trimesh.sample.sample_surface(mesh, 1000)

            template_pc = PointCloud(cloud_points.shape[0])
            template_pc.mutable_xyzs()[:] = cloud_points.T

            self.templates[name] = {
                'mesh': mesh,
                'point_cloud': template_pc,
                'cloud_points': cloud_points
            }
        
    def match_object(self, observed_pc: PointCloud, max_iters=100):
        """
        attempt to match an observed point cloud segment against all templates
        works because we have a limited set of known objects
        """
        best_match = None
        best_score = float('inf')
        best_pose = None
        
        observed_np = observed_pc.xyzs().T
        
        for obj_name, template in self.templates.items():
            X_initial = self._get_initial_alignment(observed_np, template['cloud_points'])
            
            X_MS, cost = IterativeClosestPoint(
                p_Om=template['cloud_points'].T, 
                p_Ws=observed_np.T,
                X_Ohat=X_initial,
                max_iterations=max_iters
            )
            
            fitness = self._calculate_fitness(
                observed_np, 
                template['cloud_points'], 
                X_MS
            )
            
            print(f"  {obj_name}: fitness = {fitness:.6f}")
            
            if fitness < best_score:
                best_score = fitness
                best_match = obj_name
                best_pose = X_MS
        
        return best_match, best_pose, best_score

    def _get_initial_alignment(self, observed, template):
        """compute initial alignment by matching centroids"""
        obs_centroid = np.mean(observed, axis=0)
        temp_centroid = np.mean(template, axis=0)
        translation = obs_centroid - temp_centroid
        return RigidTransform(translation)
    
    def _calculate_fitness(self, observed, template, X_MS):
        """calculate mean squared distance after transformation"""
        # transform template to observed frame
        template_transformed = (X_MS.rotation().matrix() @ template.T).T + X_MS.translation()
        
        # find nearest neighbor distances
        from scipy.spatial import cKDTree
        tree = cKDTree(template_transformed)
        distances, _ = tree.query(observed, k=1)
        
        return np.mean(distances ** 2)
