
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

from pydrake.multibody.parsing import ProcessModelDirectives, ModelDirectives
from manipulation import running_as_notebook
from manipulation.station import LoadScenario
from manipulation.icp import IterativeClosestPoint

from perception import add_cameras, get_depth
from pid_controller import PIDController

SCENARIO_PATH = Path("/workspaces/robman-final-proj/src/scenario.yaml")
MUG_MESH_PATH = Path("/workspaces/robman-final-proj/assets/mug/google_16k/textured.obj")
VOXEL_SIZE = 0.005
TABLE_Z_THRESHOLD = 1.1
N_SAMPLE_POINTS = 1500
MAX_ICP_ITERS = 25

def keep_top_rim_band(pc: PointCloud, band_thickness: float = 0.02) -> PointCloud:
    """
    Keep only the top band_thickness meters near the global max z.
    This should capture just mug rims (and not the tray surface).
    """
    xyz = pc.xyzs()
    if xyz.shape[1] == 0:
        return pc
    z = xyz[2, :]
    z_max = float(np.max(z))
    mask = (z > (z_max - band_thickness))
    idx = np.where(mask)[0]
    if idx.size == 0:
        return pc
    rim_xyz = xyz[:, idx]
    out = PointCloud(rim_xyz.shape[1])
    out.mutable_xyzs()[:] = rim_xyz
    return out

def remove_table_points(pc: PointCloud,
                        z_thresh: float = TABLE_Z_THRESHOLD) -> PointCloud:
    """Keep only points with z > z_thresh."""
    xyz = pc.xyzs()
    if xyz.shape[1] == 0:
        return pc
    z = xyz[2, :]
    mask = (z > z_thresh)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return PointCloud(0)
    filt_xyz = xyz[:, idx]
    out = PointCloud(filt_xyz.shape[1])
    out.mutable_xyzs()[:] = filt_xyz
    return out

def build_rim_pointcloud(diagram, context) -> PointCloud:
    # Grab depth images (for debugging/plots if you want)
    get_depth(diagram, context)

    pc0 = diagram.GetOutputPort("camera_point_cloud0").Eval(context)
    pc1 = diagram.GetOutputPort("camera_point_cloud1").Eval(context)
    pc2 = diagram.GetOutputPort("camera_point_cloud2").Eval(context)

    concat_pc = Concatenate([pc0, pc1, pc2])
    down_pc = concat_pc.VoxelizedDownSample(VOXEL_SIZE)

    xyz_down = down_pc.xyzs()
    # print("Downsampled cloud points:", xyz_down.shape[1])
    # print("Downsampled z min/max:", float(np.min(xyz_down[2, :])),
    #       float(np.max(xyz_down[2, :])))

    obj_pc = remove_table_points(down_pc, z_thresh=TABLE_Z_THRESHOLD)
    if obj_pc.xyzs().shape[1] == 0:
        # relax if needed
        obj_pc = remove_table_points(down_pc, z_thresh=TABLE_Z_THRESHOLD - 0.1)
        if obj_pc.xyzs().shape[1] == 0:
            obj_pc = down_pc

    # print("After table removal: num points =", obj_pc.xyzs().shape[1])

    # rim_pc = keep_top_rim_band(obj_pc, band_thickness=0.02)
    # if rim_pc.xyzs().shape[1] == 0:
    #     rim_pc = obj_pc

    # xyz_rim = rim_pc.xyzs()
    # print("Rim cloud: num points =", xyz_rim.shape[1])
    # print("Rim z min/max:", float(np.min(xyz_rim[2, :])),
    #       float(np.max(xyz_rim[2, :])))

    diagram.ForcedPublish(context)
    return obj_pc


def estimate_mug_pose_icp(meshcat, rim_pc: PointCloud):
    p_Ws = rim_pc.xyzs()
    if p_Ws.shape[1] == 0:
        raise RuntimeError("estimate_mug_pose_icp: rim_pc has no points!")

    # Load mug model points in mug frame
    # if MUG_MESH_PATH.is_file():
    mug_mesh = trimesh.load(str(MUG_MESH_PATH), force="mesh")
    pts = mug_mesh.sample(N_SAMPLE_POINTS)   # (N, 3)
    p_Om = pts.T                             # (3, N)
    print(f"Loaded mug mesh from {MUG_MESH_PATH}")

    # --- Build an initial guess from rim + model geometry ---
    # World rim center and top
    center_xyz = np.mean(p_Ws, axis=1)      # (3,)
    center_x, center_y = center_xyz[0], center_xyz[1]
    z_rim_world = float(np.max(p_Ws[2, :]))

    # Mug model top (in its own frame)
    model_top_z    = float(np.max(p_Om[2, :]))
    model_bottom_z = float(np.min(p_Om[2, :]))

    # Translate model so its top aligns with rim z, and center x,y align
    initial_translation = [
        center_x,
        center_y,
        z_rim_world - model_top_z,
    ]

    initial_guess = RigidTransform(
        RotationMatrix(),   # assume mug is upright
        initial_translation,
    )

    # --- Run ICP ---
    X_WM_hat, cost = IterativeClosestPoint(
        p_Om=p_Om,
        p_Ws=p_Ws,
        X_Ohat=initial_guess,
        meshcat=meshcat,
        meshcat_scene_path="icp/mug",
        max_iterations=MAX_ICP_ITERS,
    )
    # print("ICP cost:", cost)

    # Transform model points to world to get true mug top/bottom from ICP
    R_WM = X_WM_hat.rotation().matrix()
    p_WM = X_WM_hat.translation().reshape(3, 1)

    p_Wm = R_WM @ p_Om + p_WM
    z_vals = p_Wm[2, :]

    mug_bottom_z = float(np.min(z_vals))
    mug_top_z    = float(np.max(z_vals))

    return X_WM_hat, (mug_bottom_z, mug_top_z)