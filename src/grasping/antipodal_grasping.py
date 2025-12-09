"""
antipodal grasp generation based on manipulation textbook 5.12
Simplified for tabletop grasping without full collision checking
"""
import numpy as np
from pydrake.all import RigidTransform, RotationMatrix, Sphere, Rgba

# heavier vertical penalty for when objects are close together on tray
VERTICAL_PENALTY_WEIGHT = 12.0  # used to be 5


def score_grasp_simple(X_G, cloud, verbose=False): # verbose for debugging info
    """
    Score a grasp candidate, follows matching notebook scoring, lower score better
    - penalize deviation from vertical 20.0 * R_G[2,1], vertical grasps to avoid collision
        w other items on tray
    - reward alignment with normals -sum(n_x^2), normals opposing each other along gripper x-axis
    - adjust grasp to center on contact points
    
    Returns (cost, adjusted_X_G, contact_info)
    contact_info = dict with contact points for visualization
    """
    R_G = X_G.rotation().matrix()
    
    # Transform cloud into gripper frame
    X_GW = X_G.inverse()
    p_GC = X_GW.multiply(cloud.xyzs())
    
    # Crop to finger region (contact area between fingers)
    # Extended deeper into gripper (before 0.1-0.1125) to capture more contact
    crop_min = np.array([-0.05, 0.085, -0.00625])
    crop_max = np.array([0.05, 0.125, 0.00625])
    
    # find points within finger region
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
    
    # adjust grasp to center on contact points (same as 5.12 notebook)
    X_G_adjusted = X_G
    if num_contact_points > 0:
        p_GC_x = p_GC[0, indices]
        p_Gcenter_x = (p_GC_x.min() + p_GC_x.max()) / 2.0
        # move gripper along its x-axis to center on contacts
        X_G_adjusted = RigidTransform(
            X_G.rotation(),
            X_G.translation() + X_G.rotation().multiply(np.array([p_Gcenter_x, 0, 0]))
        )
        
        # recompute with adjusted pose
        X_GW = X_G_adjusted.inverse()
        p_GC = X_GW.multiply(cloud.xyzs())
        # recompute indices
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
    
    if num_contact_points == 0:
        return np.inf, X_G_adjusted, None
    
    # get normals in gripper frame
    n_GC = X_GW.rotation().multiply(cloud.normals()[:, indices])
    
    # Penalize deviation from vertical
    # the example nb uses 20.0 for dense bins but choose 12.0 to generalize more
    vertical_cost = VERTICAL_PENALTY_WEIGHT * R_G[2, 1]
    
    # reward alignment with gripper x-axis (normals opposing each other)
    normal_alignment = -np.sum(n_GC[0, :] ** 2)
    
    # reward grasp candidates with more contact points (new)
    # bc more points = more stable grasp (works better for round objs like mgu)
    contact_density = num_contact_points / cloud.size()
    density_reward = -15.0 * contact_density  # increased reward weight
    
    # check contact balance left-right (x-axis distribution) (new)
    # alg prefers grasp where both fingers contact object equally
    # grasp more stable and less like to twist or slip on round surfaces
    p_GC_x = p_GC[0, indices]
    left_contacts = np.sum(p_GC_x < -0.01)
    right_contacts = np.sum(p_GC_x > 0.01)
    # absolute diff normalized by total contact pts (+1 for div by zero error)
    balance = 1.0 - abs(left_contacts - right_contacts) / (left_contacts + right_contacts + 1)
    balance_reward = -5.0 * balance  # reward balanced contact
    
    # require more contact points for stability
    # at least 6% of cloud points in contact region (was 8% but relaxed for thin objects like mug handle bc it failed )
    sparsity_penalty = 0.0
    min_required = max(6, cloud.size() * 0.06)
    if num_contact_points < min_required:
        sparsity_penalty = 100.0  # high penalty for sparse contact
    
    cost = vertical_cost + normal_alignment + density_reward + balance_reward + sparsity_penalty
    
    # Get contact points in world frame for visualization
    contact_points_W = cloud.xyzs()[:, indices]
    contact_info = {
        'num_contacts': num_contact_points,
        'left_contacts': left_contacts,
        'right_contacts': right_contacts,
        'balance': balance,
        'density': contact_density,
        'contact_points_W': contact_points_W,
        'vertical_cost': vertical_cost,
        'normal_alignment': normal_alignment,
        'density_reward': density_reward,
        'balance_reward': balance_reward,
        'sparsity_penalty': sparsity_penalty,
    }
    
    if verbose:
        print(f"    Contact analysis:")
        print(f"      Points: {num_contact_points} ({contact_density*100:.1f}% of cloud)")
        print(f"      Left/Right: {left_contacts}/{right_contacts} (balance={balance:.3f})")
        print(f"      Vertical cost: {vertical_cost:.3f}")
        print(f"      Normal align: {normal_alignment:.3f}")
        print(f"      Density reward: {density_reward:.3f}")
        print(f"      Balance reward: {balance_reward:.3f}")
        print(f"      Sparsity penalty: {sparsity_penalty:.3f}")
        print(f"      Total cost: {cost:.3f}")
    
    return cost, X_G_adjusted, contact_info


def generate_antipodal_grasp_candidate(cloud, rng):
    """
    Generate one antipodal grasp candidate similar to 5.12 notebook algorithm:
    1) Pick random point and align gripper x-axis with normal
    2) Try different roll angles around the normal (7 total)
    3) Score each candidate and return first one with finite cost
    
    Returns (cost, X_G) or (np.inf, None) if no valid grasp found
    """
    if cloud.size() == 0:
        return np.inf, None
    
    # pick random point
    index = rng.integers(0, cloud.size())
    
    # S is sample point/frame
    p_WS = cloud.xyz(index)
    n_WS = cloud.normal(index)
    
    # Check normal is valid
    if not np.isclose(np.linalg.norm(n_WS), 1.0):
        return np.inf, None
    
    # Align gripper x-axis with normal
    Gx = n_WS
    
    # Make orthonormal y-axis, aligned with world down
    y = np.array([0.0, 0.0, -1.0]) # y is antiparallel to garvity, cant construct valid orthonormal gripper frame from this
    if np.abs(np.dot(y, Gx)) < 1e-6:
        # Normal pointing straight down so reject sample
        # ensure grasp on side surfaces
        return np.inf, None
    
    Gy = y - np.dot(y, Gx) * Gx
    Gy = Gy / np.linalg.norm(Gy) # tilted toward world down
    Gz = np.cross(Gx, Gy)
    R_WG = RotationMatrix(np.vstack((Gx, Gy, Gz)).T) # rotation from gripper to world frame
    
    # Gripper geometry offset (from notebook)
    p_GS_G = np.array([0.054 - 0.01, 0.10625, 0])
    
    # Try orientations from center out
    min_roll = -np.pi / 3.0
    max_roll = np.pi / 3.0
    alpha = np.array([0.5, 0.65, 0.35, 0.8, 0.2, 1.0, 0.0])
    
    for a in alpha:
        theta = min_roll + (max_roll - min_roll) * a
        
        # Rotate around gripper x-axis (normal)
        R_WG2 = R_WG.multiply(RotationMatrix.MakeXRotation(theta))
        
        # Calculate gripper position
        p_SG_W = -R_WG2.multiply(p_GS_G) # offset from sample point to gripper origin in world frame
        p_WG = p_WS + p_SG_W
        
        X_G = RigidTransform(R_WG2, p_WG)
        
        # Score this candidate with grasp adjusted to center on contact pts
        cost, X_G_adjusted, contact_info = score_grasp_simple(X_G, cloud)
        
        # Return first valid grasp (finite cost = valid)
        if np.isfinite(cost):
            return cost, X_G_adjusted, contact_info
    
    return np.inf, None, None


def sample_antipodal_grasps(cloud, rng, num_samples=100):
    """
    Sample multiple antipodal grasp candidates.
    Returns list of (cost, X_G, contact_info) tuples for valid grasps only.
    """
    grasps = []
    costs = []
    contact_infos = []
    
    for i in range(num_samples):
        cost, X_G, contact_info = generate_antipodal_grasp_candidate(cloud, rng)
        if np.isfinite(cost):
            costs.append(cost)
            grasps.append(X_G)
            contact_infos.append(contact_info)
    
    return costs, grasps, contact_infos
    
def get_best_antipodal_grasp(cloud, rng, num_samples=100, meshcat=None):
    """
    Sample multiple grasps and return the best one.
    Matches the notebook's approach of sampling and scoring.
    """
    costs, grasps, contact_infos = sample_antipodal_grasps(cloud, rng, num_samples)
    
    if len(grasps) == 0:
        print(f"  WARNING: No valid grasp candidates from {num_samples} samples")
        print(f"  Cloud has {cloud.size()} points - may need better normals or more samples")
        return None
    
    # find best (lowest cost)
    best_idx = np.argmin(costs)
    best_cost = costs[best_idx]
    best_contact = contact_infos[best_idx]
    
    print(f"  Antipodal grasp: cost={best_cost:.3f}, candidates={len(grasps)}/{num_samples}")
    print(f"    Contacts: {best_contact['num_contacts']} points ({best_contact['density']*100:.1f}% of cloud)")
    print(f"    Left/Right: {best_contact['left_contacts']}/{best_contact['right_contacts']} (balance={best_contact['balance']:.3f})")
    
    # visualize contact points in meshcat for debugging
    if meshcat is not None and best_contact is not None:
        contact_pts = best_contact['contact_points_W']

        # clear previous contact markers to avoid clutter
        try:
            meshcat.Delete("grasp_contacts")
        except Exception:
            pass

        # show contact points as small red spheres under a common folder
        max_pts = min(contact_pts.shape[1], 50)  # limit for clarity
        for i in range(max_pts):
            pt = contact_pts[:, i]
            meshcat.SetObject(f"grasp_contacts/contact_{i}", Sphere(0.008), Rgba(1, 0, 0, 0.8))
            meshcat.SetTransform(f"grasp_contacts/contact_{i}", RigidTransform(pt))

        print(f"    Visualized {max_pts} contact points in red (folder: grasp_contacts)")
    
    return grasps[best_idx]



# notes on differences from 5.12 notebook
# Notebook is for bin-picking: objects cluttered, collision critical, strict vertical required
# in this scenario objects isolated, less  collision risk bc placing on empty table, flexible orientations OK
# changes:
# reduced vertical penalty (20.0 -> 12): tray allows slightly angled grasps but still constrains to verticla descent
#   to prevent bumping into other objs
# added contact density reward, prefer grasps with more surface contact
# added balance check: ensure both fingers contact object equally
# deeper contact region: 0.085-0.125 vs 0.1-0.1125 for better grip/deeper into the object