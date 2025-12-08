"""
antipodal grasp generation
"""
import numpy as np
from pydrake.all import RigidTransform, RotationMatrix


def generate_antipodal_grasp_candidate(cloud, rng):
    """
    Generate one antipodal grasp candidate by
    - picking random point in cloud
    - aligning gripper x-axis with surface normal
    - trying different roll angles around the normal
    
    returns X_G (RigidTransform of grasp pose)
    """
    # unlike 5.12, gripper doesnt collision check but gripper always moves straight down and up, avoiding collisions 
    
    if cloud.size() == 0:
        print("  Point cloud is empty, cannot generate grasp")
        return None
        
    # pick random point
    index = rng.integers(0, cloud.size())
    
    # get point and normal
    p_WS = cloud.xyz(index)  # sample point in world
    n_WS = cloud.normal(index)  # surface normal
    
    # check normal is valid
    normal_magnitude = np.linalg.norm(n_WS)
    if not np.isclose(normal_magnitude, 1.0) or normal_magnitude < 0.9: # invalid norm length
        return None
    
    # align gripper x-axis with normal
    Gx = n_WS
    
    # make orthonormal y-axis aligned with world down
    y = np.array([0.0, 0.0, -1.0])
    if np.abs(np.dot(y, Gx)) > 0.99:  # normal pointing straight down/up
        return None
    
    Gy = y - np.dot(y, Gx) * Gx
    Gy = Gy / np.linalg.norm(Gy)
    Gz = np.cross(Gx, Gy)
    R_WG = RotationMatrix(np.vstack((Gx, Gy, Gz)).T)
    
    # offset from grasp point to gripper body frame origin
    # gripper fingers are at distance 0.10625 along y-axis in gripper frame
    # reduce offset to get gripper closer to actual contact point for better centering
    p_GS_G = np.array([0.025, -0.10625, 0])
    
    # try different roll angles (rotate around x-axis/normal)
    min_roll = -np.pi / 3.0 # -60 degrees
    max_roll = np.pi / 3.0 # 60 degrees
    # try center first, then spread out
    alphas = [0.5, 0.65, 0.35, 0.8, 0.2, 1.0, 0.0]
    
    for alpha in alphas:
        theta = min_roll + (max_roll - min_roll) * alpha
        
        # rotate around gripper x-axis (normal)
        R_WG_rotated = R_WG.multiply(RotationMatrix.MakeXRotation(theta))
        
        # calculate gripper position offset from contact point by gripper geometry
        p_SG_W = R_WG_rotated.multiply(p_GS_G)
        p_WG = p_WS - p_SG_W
        
        X_G = RigidTransform(R_WG_rotated, p_WG)
        
        # check gripper shouldn't be too low
        if p_WG[2] > 0.5:  # above ground
            return X_G
    
    return None


def sample_antipodal_grasps(cloud, rng, num_samples=100):
    """
    Sample multiple antipodal grasp candidates
    
    Returns list of RigidTransform grasp poses
    """
    grasps = []
    
    for i in range(num_samples):
        X_G = generate_antipodal_grasp_candidate(cloud, rng)
        if X_G is not None:
            grasps.append(X_G)
    
    return grasps


def score_grasp(X_G, cloud):
    """
    Score a grasp candidate (lower is better)
    uses 5.12 example notebook cost function with additional contact quality metrics
    - penalize deviation from vertical: 20.0 * R_G[2,1]
    - reward alignment with normals: -sum(n_x^2) where n_x is normal dot gripper_x
    - strongly penalize if too few contact points (need good grasp closure for round objects like soup can)
        otherwise object will rotate
    - for round objects, penalize imbalanced contacts (prefer symmetric grasps)
    """
    R_G = X_G.rotation().matrix()
    
    # penalize tilt from vertical
    cost = 20.0 * R_G[2, 1]
    
    # Transform cloud into gripper frame
    X_GW = X_G.inverse()
    p_GC = X_GW.multiply(cloud.xyzs())
    
    # crop to finger region so we only consider contact points near fingers
    crop_min = np.array([-0.05, 0.1, -0.00625])
    crop_max = np.array([0.05, 0.1125, 0.00625])
    
    indices = np.all(
        [
            crop_min[0] <= p_GC[0, :],
            p_GC[0, :] <= crop_max[0],
            crop_min[1] <= p_GC[1, :],
            p_GC[1, :] <= crop_max[1],
            crop_min[2] <= p_GC[2, :],
            p_GC[2, :] <= crop_max[2],
        ],
        axis=0,
    )
    
    num_contact_points = np.sum(indices)
    if num_contact_points < 20:  # discard if less than 20 points bc contact is bad
        return np.inf
    
    # get points and normals in gripper frame for contacts
    p_contact = p_GC[:, indices]
    n_GC = X_GW.rotation().multiply(cloud.normals()[:, indices])
    
    # reward alignment with gripper x-axis so normals oppose each other
    alignment_reward = np.sum(n_GC[0, :] ** 2)
    cost -= alignment_reward
    
    # bonus for having many contact points (more robust grasp)
    contact_bonus = min(num_contact_points / 100.0, 5.0)
    cost -= contact_bonus
    
    # for round objects: penalize if contacts are imbalanced (not symmetric around centerline)
    # want contact points on both sides of the gripper (z)
    z_positions = p_contact[2, :]
    z_positive = np.sum(z_positions > 0.001)
    z_negative = np.sum(z_positions < -0.001)
    
    # both sides should have significant contact
    if z_positive > 0 and z_negative > 0:
        # reward balanced contact distribution
        balance_ratio = min(z_positive, z_negative) / max(z_positive, z_negative)
        balance_bonus = balance_ratio * 3.0  # up to 3 point bonus
        cost -= balance_bonus
    else:
        # penalize one-sided grasps heavily
        cost += 10.0
    
    return cost


def get_best_antipodal_grasp(cloud, rng, num_samples=100):
    """
    Sample multiple grasps and return the best grasp RigidTransform
    """
    grasps = sample_antipodal_grasps(cloud, rng, num_samples)
    
    if len(grasps) == 0:
        return None
    
    # grasp scores
    scores = [score_grasp(X_G, cloud) for X_G in grasps]
    
    best_idx = np.argmin(scores) # min = best grasp
    best_cost = scores[best_idx]
    
    if np.isfinite(best_cost):
        print(f"  Found antipodal grasp with cost {best_cost:.3f} ({len(grasps)} candidates)")
        return grasps[best_idx]
    
    return None
