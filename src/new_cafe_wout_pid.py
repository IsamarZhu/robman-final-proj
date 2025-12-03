from pathlib import Path
import time
import numpy as np
import trimesh

from pydrake.all import (
    DiagramBuilder,
    InverseKinematics,
    PointCloud,
    Rgba,
    RigidTransform,
    RotationMatrix,
    Simulator,
    Solve,
    StartMeshcat,
    BasicVector,
    LeafSystem,
)

from pydrake.geometry import (
    MakeRenderEngineVtk,
    RenderEngineVtkParams,
    Box,
)

from pydrake.systems.primitives import ConstantVectorSource

from manipulation import running_as_notebook
from manipulation.station import (
    LoadScenario,
    MakeHardwareStation,
    AddPointClouds,
)
from manipulation.icp import IterativeClosestPoint

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

SCENARIO_PATH = Path("/workspaces/robman-final-proj/src/new_scenario.yaml")
MUG_MESH_PATH = Path(
    "/workspaces/robman-final-proj/assets/mug/google_16k/textured.obj"
)

VOXEL_SIZE = 0.005
BAND_THICKNESS = 0.02          # top band for rim
N_SAMPLE_POINTS = 1500
MAX_ICP_ITERS = 25

TABLE_Z_THRESH_MARGIN = 0.02   # how far above "table" z we keep points
SAFE_CLEARANCE = 0.25          # flying height above largest object
GRASP_BELOW_RIM = 0
PLACE_ABOVE_TABLE = 0.02

WSG_OPEN = 0.12
WSG_CLOSED = 0.03
PHASE_DT = 2.5
HOLD_DT  = 1.0


if running_as_notebook:
    import mpld3
    mpld3.enable_notebook()


# --------------------------------------------------------------------------- #
# Small helper: joint position command source for iiwa
#   Outputs full [q_des (Nq), v_des (Nq)] to match iiwa_arm.desired_state
# --------------------------------------------------------------------------- #

class JointPositionCommandSource(LeafSystem):
    """
    Outputs a 2*Nq-dim iiwa desired state:
      [q_desired (Nq), v_desired (Nq)]
    We keep v_desired = 0 and only update q_desired via set_q_desired().
    """

    def __init__(self, q_initial: np.ndarray):
        super().__init__()
        q_initial = np.copy(q_initial).reshape(-1)
        self._nq = q_initial.shape[0]   # should be 10 for mobile iiwa
        self._q_des = q_initial

        self.DeclareVectorOutputPort(
            "iiwa_desired_state",
            BasicVector(2 * self._nq),
            self._DoCalcOutput,
        )

    def _DoCalcOutput(self, context, output: BasicVector):
        v_des = np.zeros(self._nq)
        desired_state = np.concatenate([self._q_des, v_des])
        output.SetFromVector(desired_state)

    def set_q_desired(self, q_des: np.ndarray):
        q_des = np.copy(q_des).reshape(-1)
        assert q_des.shape[0] == self._nq, f"q_des must be {self._nq}-dim"
        self._q_des = q_des

class WsgCommandSource(LeafSystem):
    """
    Outputs a 1-dim desired opening width for the Schunk WSG.
    We can update it via set_width().
    """
    def __init__(self, initial_width: float):
        super().__init__()
        self._width = float(initial_width)
        self.DeclareVectorOutputPort(
            "wsg_position",
            BasicVector(1),
            self._DoCalcOutput,
        )

    def _DoCalcOutput(self, context, output: BasicVector):
        output.SetFromVector([self._width])

    def set_width(self, width: float):
        self._width = float(width)


# --------------------------------------------------------------------------- #
# Point cloud helpers (using HardwareStation camera point cloud ports)
# --------------------------------------------------------------------------- #

def downsample(pc: PointCloud, voxel_size: float) -> PointCloud:
    if pc.xyzs().shape[1] == 0:
        return pc
    return pc.VoxelizedDownSample(voxel_size)


def remove_below_z(pc: PointCloud, z_thresh: float) -> PointCloud:
    """Keep only points with z > z_thresh."""
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


def keep_top_band(pc: PointCloud, band_thickness: float) -> PointCloud:
    """Keep only the top band_thickness meters near the global max z."""
    xyz = pc.xyzs()
    if xyz.shape[1] == 0:
        return pc
    z = xyz[2, :]
    z_max = float(np.max(z))
    mask = (z > (z_max - band_thickness))
    idx = np.where(mask)[0]
    if idx.size == 0:
        return pc
    sel = xyz[:, idx]
    out = PointCloud(sel.shape[1])
    out.mutable_xyzs()[:] = sel
    return out


def build_rim_pointcloud(diagram, context, meshcat) -> PointCloud:
    """
    Uses point cloud outputs (camera0/1/2.point_cloud) that were created
    via AddPointClouds(scenario, station, builder, meshcat).

    Expects diagram to have exported output ports:
      "camera0.point_cloud", "camera1.point_cloud", "camera2.point_cloud".

    NOTE: We no longer render these point clouds in Meshcat.
    """

    pc0 = diagram.GetOutputPort("camera0.point_cloud").Eval(context)
    pc1 = diagram.GetOutputPort("camera1.point_cloud").Eval(context)
    pc2 = diagram.GetOutputPort("camera2.point_cloud").Eval(context)

    xyz = np.concatenate(
        [pc0.xyzs(), pc1.xyzs(), pc2.xyzs()],
        axis=1,
    )
    concat_pc = PointCloud(xyz.shape[1])
    concat_pc.mutable_xyzs()[:] = xyz

    # Downsample a bit
    down_pc = downsample(concat_pc, VOXEL_SIZE)
    xyz_down = down_pc.xyzs()
    print("Downsampled N =", xyz_down.shape[1])
    print(
        "z min/max:",
        float(np.min(xyz_down[2, :])),
        float(np.max(xyz_down[2, :])),
    )

    # Estimate "support surface" height from the 10th percentile of z
    z_vals = xyz_down[2, :]
    if z_vals.size == 0:
        return down_pc

    z_surface = float(np.percentile(z_vals, 10))
    z_thresh = z_surface + TABLE_Z_THRESH_MARGIN
    print(
        "Estimated support surface z:",
        z_surface,
        " -> using thresh:",
        z_thresh,
    )

    obj_pc = remove_below_z(down_pc, z_thresh)
    print("After below-z removal: N =", obj_pc.xyzs().shape[1])

    rim_pc = keep_top_band(obj_pc, BAND_THICKNESS)
    if rim_pc.xyzs().shape[1] == 0:
        rim_pc = obj_pc

    xyz_rim = rim_pc.xyzs()
    print("Rim N =", xyz_rim.shape[1])
    print(
        "Rim z min/max:",
        float(np.min(xyz_rim[2, :])),
        float(np.max(xyz_rim[2, :])),
    )

    # visualize rim points in Meshcat (correct API)
    meshcat.SetObject(
        "debug/rim_points",
        rim_pc,
        point_size=0.01,
        rgba=Rgba(1, 0, 0, 1),
    )

    return rim_pc


# --------------------------------------------------------------------------- #
# ICP on rim cloud  (matches class notebook style: p_Om vs p_Ws, X_Ohat init)
# --------------------------------------------------------------------------- #

def estimate_mug_pose_icp(meshcat, rim_pc: PointCloud):
    p_Ws = rim_pc.xyzs()  # (3, N)
    if p_Ws.shape[1] == 0:
        raise RuntimeError("rim_pc has no points")

    # ----------------- 1) INITIAL GUESS FOR ICP (still rim-based) -----------------
    # Just use simple centroid as before for the ICP seed:
    center_xyz_seed = np.mean(p_Ws, axis=1)

    # ----------------- 2) MODEL POINTS FOR ICP -----------------
    mug_mesh = trimesh.load(str(MUG_MESH_PATH), force="mesh")
    pts = mug_mesh.sample(N_SAMPLE_POINTS)  # (N, 3)
    p_Om = pts.T                            # (3, N)

    z_rim_world = float(np.max(p_Ws[2, :]))
    model_top_z = float(np.max(p_Om[2, :]))

    initial_translation = [
        center_xyz_seed[0],
        center_xyz_seed[1],
        z_rim_world - model_top_z,
    ]
    initial_guess = RigidTransform(
        RotationMatrix(),  # assume mug is upright-ish
        initial_translation,
    )

    X_WM_hat, cost = IterativeClosestPoint(
        p_Om=p_Om,
        p_Ws=p_Ws,
        X_Ohat=initial_guess,
        meshcat=meshcat,
        meshcat_scene_path="icp/mug",
        max_iterations=MAX_ICP_ITERS,
    )
    print("ICP cost:", cost)

    # ----------------- 3) MUG VERTICAL EXTENTS (from model+ICP) -----------------
    R_WM = X_WM_hat.rotation().matrix()
    p_WM = X_WM_hat.translation().reshape(3, 1)
    p_Wm = R_WM @ p_Om + p_WM
    z_vals = p_Wm[2, :]

    mug_bottom_z = float(np.min(z_vals))
    mug_top_z = float(np.max(z_vals))

    # ----------------- 4) RIM CENTER FROM OBSERVED POINTS ONLY -----------------
    xyz = p_Ws
    z_all = xyz[2, :]
    z_max_rim = float(np.max(z_all))

    # Take a thin top slice of the rim points (e.g. top 1 cm)
    TOP_SLICE = 0.01
    mask_top = z_all > (z_max_rim - TOP_SLICE)
    if np.count_nonzero(mask_top) >= 3:
        xy_top = xyz[:2, mask_top].T  # (Nt, 2)
    else:
        # Fallback: use all rim points if something went weird
        xy_top = xyz[:2, :].T

    center_xy = fit_circle_xy(xy_top)      # <- ONLY uses observed rim points
    mug_center_xyz = np.array([center_xy[0], center_xy[1], z_max_rim])

    # Optional debug viz: show the center as a tiny box
    meshcat.SetObject(
        "debug/rim_center",
        Box(0.01, 0.01, 0.01),
        Rgba(0, 1, 1, 1),
    )
    meshcat.SetTransform(
        "debug/rim_center",
        RigidTransform(mug_center_xyz),
    )

    return X_WM_hat, mug_bottom_z, mug_top_z, mug_center_xyz


def fit_circle_xy(xy: np.ndarray) -> np.ndarray:
    """
    Least-squares fit of a circle in the x-y plane.

    Args:
        xy: (N, 2) array of [x, y] points.

    Returns:
        center: np.array([cx, cy])
    """
    assert xy.ndim == 2 and xy.shape[1] == 2
    x = xy[:, 0]
    y = xy[:, 1]

    # (x - a)^2 + (y - b)^2 = r^2
    # => x^2 + y^2 = 2*a*x + 2*b*y + c
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b_vec = x**2 + y**2

    params, *_ = np.linalg.lstsq(A, b_vec, rcond=None)
    a, b, c = params
    return np.array([a, b])



# --------------------------------------------------------------------------- #
# IK on HardwareStation plant
# --------------------------------------------------------------------------- #

def ik_for_ee_position(
    plant,
    plant_context,
    iiwa_model,
    wsg_model,
    p_W_target,
    R_WG_desired=None,
    theta_bound=0.8,
    pos_tol=0.08,
    strict=False,              # <--- new
):
    """
    Solve IK for the WSG 'body' frame to reach p_W_target, optionally with
    desired orientation R_WG_desired.

    - IK works in the full plant generalized position vector.
    - We lock any iiwa base DOFs if present (iiwa_base_x/y/z).
    - We return the full iiwa positions for this model instance (10 DOFs).
    """
    q_seed_full = plant.GetPositions(plant_context)

    ik = InverseKinematics(plant, plant_context)
    q_decision = ik.q()
    world_frame = plant.world_frame()
    ee_frame = plant.GetFrameByName("body", wsg_model)

    # If the iiwa has mobile base DOFs named iiwa_base_x/y/z, lock them
    for joint_name in ["iiwa_base_x", "iiwa_base_y", "iiwa_base_z"]:
        try:
            joint = plant.GetJointByName(joint_name, iiwa_model)
        except RuntimeError:
            continue
        idx = joint.position_start()
        q_val = q_seed_full[idx]
        ik.prog().AddBoundingBoxConstraint(
            q_val, q_val, q_decision[idx:idx + 1]
        )

    # Position box around target
    p = np.copy(p_W_target)

    if strict:
        # Exact position (up to solver tolerance)
        lower = p
        upper = p
    else:
        # Box around target
        lower = np.array([p[0] - pos_tol, p[1] - pos_tol, p[2] - pos_tol])
        upper = np.array([p[0] + pos_tol, p[1] + pos_tol, p[2] + pos_tol])

    ik.AddPositionConstraint(
        ee_frame, [0, 0, 0],
        world_frame,
        lower,
        upper,
    )

    # Orientation constraint (e.g. gripper "pointing down")
    if R_WG_desired is not None:
        ik.AddOrientationConstraint(
            world_frame,
            R_WG_desired,
            ee_frame,
            RotationMatrix(),
            theta_bound,
        )

    prog = ik.prog()
    prog.SetInitialGuess(q_decision, q_seed_full)

    result = Solve(prog)
    if not result.is_success():
        print("IK FAILED for target:", p_W_target)
        print("  lower:", lower, "upper:", upper)
        raise RuntimeError(f"IK failed for target {p_W_target} ")

    q_sol_full = result.GetSolution(q_decision)

    plant.SetPositions(plant_context, q_sol_full)
    q_iiwa = plant.GetPositions(plant_context, iiwa_model)
    return np.copy(q_iiwa)


# --------------------------------------------------------------------------- #
# Pick-and-place using HardwareStation's built-in iiwa controller
# --------------------------------------------------------------------------- #

# def run_pick_place(
#     meshcat,
#     simulator,
#     diagram,
#     station,
#     plant,
#     plant_context,
#     iiwa_model,
#     wsg_model,
#     cmd_source: JointPositionCommandSource,
#     wsg_cmd_source: WsgCommandSource,
#     X_WM_hat,
#     mug_bottom_z,
#     mug_top_z,
#     mug_center_xyz,    # <--- new
# ):


#     context = simulator.get_mutable_context()

#     # Environment frames
#     table_inst = plant.GetModelInstanceByName("table0")
#     table_body = plant.GetBodyByName("table_body", table_inst)
#     X_WT = plant.EvalBodyPoseInWorld(plant_context, table_body)
#     p_WT = X_WT.translation()
#     table_height = p_WT[2]

#     tray_inst = plant.GetModelInstanceByName("tray")
#     tray_body = plant.GetBodyByName("tray_link", tray_inst)
#     X_WTray = plant.EvalBodyPoseInWorld(plant_context, tray_body)
#     p_WTray = X_WTray.translation()
#     tray_height = p_WTray[2]

#     print("table_height:", table_height, "tray_height:", tray_height)

#     # Gripper orientation: "point down"
#     R_WG_desired = RotationMatrix.MakeXRotation(-np.pi / 2.0)

#     # Use rim center for x, y so the grasp is centered on the mug
#     p_mug_xy = mug_center_xyz[:2]
#     x_mug, y_mug = float(p_mug_xy[0]), float(p_mug_xy[1])

#     # ------------------ Pick-side keypoints (tray) ------------------ #

#     z_grasp = mug_top_z - GRASP_BELOW_RIM
#     z_lift = max(
#         mug_top_z + 0.18,
#         tray_height + 0.25,
#     )

#     z_over_mug = max(
#         mug_top_z + 0.25,
#         tray_height + 0.35,
#     )

#     p_over_mug = np.array([x_mug, y_mug, z_over_mug])

#     EXTRA_HEIGHT = 0.20
#     max_high_z = table_height + 0.35   # heuristic upper bound for reachability
#     z_high = min(z_over_mug + EXTRA_HEIGHT, max_high_z)

#     p_high_over_mug = np.array([x_mug, y_mug, z_high])



#     p_grasp    = np.array([x_mug, y_mug, z_grasp])
#     p_lift     = np.array([x_mug, y_mug, z_lift])


#     print("p_over_mug:", p_over_mug)
#     print("p_grasp   :", p_grasp)
#     print("p_lift    :", p_lift)

#     meshcat.SetObject("debug/over_mug",
#                       Box(0.03, 0.03, 0.03), Rgba(0, 1, 0, 0.4))
#     meshcat.SetTransform("debug/over_mug",
#                          RigidTransform(R_WG_desired, p_over_mug))
#     meshcat.SetObject("debug/grasp",
#                       Box(0.03, 0.03, 0.03), Rgba(1, 0, 0, 0.4))
#     meshcat.SetTransform("debug/grasp",
#                          RigidTransform(R_WG_desired, p_grasp))

#     # ------------------ Place-side keypoints (table edge) ------------------ #
#     p_table_center = p_WT.copy()
#     print("Table center (x, y):", p_table_center[:2])

#     robot_xy = np.array([0.0, 0.0])
#     table_xy = p_table_center[:2]
#     direction = table_xy - robot_xy
#     norm = np.linalg.norm(direction)
#     if norm < 1e-6:
#         # Degenerate case; just don't offset.
#         direction = np.array([0.0, -1.0])
#         norm = 1.0
#     direction /= norm

#     EDGE_DISTANCE = 0.35  # how far toward the robot from table center (m)
#     p_place_xy = table_xy - EDGE_DISTANCE * direction   # "edge-ish" toward robot

#     safe_over_table_z = table_height + SAFE_CLEARANCE

#     p_over_table = np.array([
#         p_place_xy[0],
#         p_place_xy[1],
#         safe_over_table_z,
#     ])
#     p_place = np.array([
#         p_place_xy[0],
#         p_place_xy[1],
#         table_height + PLACE_ABOVE_TABLE,
#     ])
#     p_retreat = p_over_table.copy()

#     print("p_place_xy (edge-ish):", p_place_xy)
#     print("p_over_table:", p_over_table)
#     print("p_place     :", p_place)

#     # ------------------ Initial iiwa configuration ------------------ #
#     q_start_full = plant.GetPositions(plant_context, iiwa_model)

#     # Helper: IK with current plant_context as seed each time
#         # Helper: IK with configurable strictness
#     def ik_at(p_W, strict=True, pos_tol=0.01):
#         return ik_for_ee_position(
#             plant, plant_context, iiwa_model, wsg_model,
#             p_W,
#             R_WG_desired=R_WG_desired,
#             theta_bound=1.0,
#             pos_tol=pos_tol,
#             strict=strict,
#         )

        
#     q_high_over_m_full = ik_at(p_high_over_mug, strict=False, pos_tol=0.05)
    


#     # Over-mug pose
#     q_over_m_full = ik_at(p_over_mug)

#     # Build a *staircase* of poses going straight down from over_mug to grasp
#     N_DESCENT = 6   # more steps = straighter line
#     descent_qs = []
#     for alpha in np.linspace(0.0, 1.0, N_DESCENT, endpoint=True):
#         z = (1 - alpha) * p_over_mug[2] + alpha * p_grasp[2]
#         p = np.array([x_mug, y_mug, z])
#         q = ik_at(p)
#         descent_qs.append(q)
#     q_descend_full = descent_qs[-1]   # bottom pose

#     # Build a *staircase* for lifting back up from grasp to lift
#     N_LIFT = 6
#     lift_qs = []
#     for alpha in np.linspace(0.0, 1.0, N_LIFT, endpoint=True):
#         z = (1 - alpha) * p_grasp[2] + alpha * p_lift[2]
#         p = np.array([x_mug, y_mug, z])
#         q = ik_at(p)
#         lift_qs.append(q)
#     q_lift_full = lift_qs[-1]


#     # Table side: more relaxed position tolerance; no orientation constraint.
#     q_over_t_full = ik_for_ee_position(
#         plant, plant_context, iiwa_model, wsg_model,
#         p_over_table,
#         R_WG_desired=None,
#         pos_tol=0.25,
#     )
#     q_place_full = ik_for_ee_position(
#         plant, plant_context, iiwa_model, wsg_model,
#         p_place,
#         R_WG_desired=None,
#         pos_tol=0.25,
#     )

#     # We *reuse* q_over_t_full as the "retreat" pose.
#     q_retreat_full = q_over_t_full


#     def move_to(q_des_full, gripper_mode: str, dt: float = PHASE_DT):
#         nonlocal t
#         cmd_source.set_q_desired(q_des_full)

#         if gripper_mode == "open":
#             wsg_cmd_source.set_width(WSG_OPEN)
#         elif gripper_mode == "closed":
#             wsg_cmd_source.set_width(WSG_CLOSED)

#         simulator.AdvanceTo(t + dt)
#         t += dt
#         diagram.ForcedPublish(context)

        
#     # def set_wsg_width(width: float):
#     #     left_joint.set_translation(plant_context, +0.5 * width)
#     #     right_joint.set_translation(plant_context, -0.5 * width)


#     # ------------------ Keyframe script (full q_des for iiwa) ----------- #
#         # ------------------ Keyframe script (full q_des for iiwa) ----------- #
#     keyframes = [
#         ("start",         q_start_full,   "open"),
#         ("over_mug",      q_over_m_full,  "open"),
#         # move straight down with gripper open
#         ("descend_open",  q_descend_full, "open"),
#         # small pause at bottom, still open (ensures it's fully down)
#         ("hold_bottom",   q_descend_full, "open"),
#         # now close while arm is stationary at the bottom
#         ("close",         q_descend_full, "closed"),
#         ("lift",          q_lift_full,    "closed"),
#         ("over_table",    q_over_t_full,  "closed"),
#         ("place",         q_place_full,   "open"),
#         ("retreat",       q_retreat_full, "open"),
#     ]


#     # ------------------ Initialize ------------------ #
#     plant.SetPositions(plant_context, iiwa_model, q_start_full)
#     plant.SetVelocities(plant_context, iiwa_model, np.zeros_like(q_start_full))
#     cmd_source.set_q_desired(q_start_full)
#     wsg_cmd_source.set_width(WSG_OPEN)
#     diagram.ForcedPublish(context)

#     t = context.get_time()

#     # ------------------ Execute 8-step pattern ------------------ #

#     print("Executing full pick-and-place sequence (HardwareStation)...")

#     # 1) start → HIGH over mug (open)
#     print(" -> start")
#     move_to(q_start_full, "open", dt=PHASE_DT)

#     print(" -> high_over_mug")
#     move_to(q_high_over_m_full, "open", dt=PHASE_DT)

#     # 2) descend from HIGH → over_mug in small steps
#     print(" -> descend (high → over_mug)")
#     N_DESCENT_STAGE1 = 6
#     for alpha in np.linspace(0.0, 1.0, N_DESCENT_STAGE1):
#         z = (1 - alpha) * p_high_over_mug[2] + alpha * p_over_mug[2]
#         p = np.array([x_mug, y_mug, z])
#         # relaxed: allow a small box around the exact point
#         q = ik_at(p, strict=False, pos_tol=0.03)
#         move_to(q, "open", dt=PHASE_DT / N_DESCENT_STAGE1)


#     print(" -> over_mug")
#     move_to(q_over_m_full, "open", dt=PHASE_DT)


#     # 2) descend straight down in several small steps (open)
#     print(" -> descend (straight down, open)")
#     for q in descent_qs:
#         move_to(q, "open", dt=PHASE_DT / len(descent_qs))

#     # 3) hold at bottom, still open
#     print(" -> hold_bottom")
#     move_to(q_descend_full, "open", dt=HOLD_DT)

#     # 4) close in place (no motion)
#     print(" -> close")
#     move_to(q_descend_full, "closed", dt=HOLD_DT)

#     # 5) lift straight up in several small steps (closed)
#     print(" -> lift (straight up, closed)")
#     for q in lift_qs:
#         move_to(q, "closed", dt=PHASE_DT / len(lift_qs))

#     # 6) move over table edge (closed)
#     print(" -> over_table")
#     move_to(q_over_t_full, "closed", dt=PHASE_DT)

#     # 7) place on table and open
#     print(" -> place")
#     move_to(q_place_full, "open", dt=PHASE_DT)

#     # 8) retreat
#     print(" -> retreat")
#     move_to(q_retreat_full, "open", dt=PHASE_DT)

def run_pick_place(
    meshcat,
    simulator,
    diagram,
    station,
    plant,
    plant_context,
    iiwa_model,
    wsg_model,
    cmd_source: JointPositionCommandSource,
    wsg_cmd_source: WsgCommandSource,
    X_WM_hat,
    mug_bottom_z,
    mug_top_z,
    mug_center_xyz,
):
    context = simulator.get_mutable_context()

    # ------------------ Environment frames ------------------ #
    table_inst = plant.GetModelInstanceByName("table0")
    table_body = plant.GetBodyByName("table_body", table_inst)
    X_WT = plant.EvalBodyPoseInWorld(plant_context, table_body)
    p_WT = X_WT.translation()
    table_height = p_WT[2]

    tray_inst = plant.GetModelInstanceByName("tray")
    tray_body = plant.GetBodyByName("tray_link", tray_inst)
    X_WTray = plant.EvalBodyPoseInWorld(plant_context, tray_body)
    p_WTray = X_WTray.translation()
    tray_height = p_WTray[2]

    print("table_height:", table_height, "tray_height:", tray_height)

    # Gripper "pointing down"
    R_WG_desired = RotationMatrix.MakeXRotation(-np.pi / 2.0)

    # Use rim center for x, y so the grasp is centered on the mug
    x_mug, y_mug = mug_center_xyz[0], mug_center_xyz[1]

    # ------------------ Pick-side keypoints ------------------ #
    # Where we actually want the grasp to be (slightly below the rim so
    # the fingers straddle it). You can tune GRASP_BELOW_RIM.
    z_grasp = mug_top_z - GRASP_BELOW_RIM

    # Safe hover just above mug
    z_over_mug = max(
        mug_top_z + 0.25,
        tray_height + 0.35,
    )

    # Even higher “approach from far above”
    EXTRA_HEIGHT = 0.20
    max_high_z = table_height + 0.35   # heuristic reachable upper bound
    z_high = min(z_over_mug + EXTRA_HEIGHT, max_high_z)

    # Lift height after grasp
    z_lift = max(
        mug_top_z + 0.18,
        tray_height + 0.25,
    )

    p_high_over_mug = np.array([x_mug, y_mug, z_high])
    p_over_mug      = np.array([x_mug, y_mug, z_over_mug])
    p_grasp         = np.array([x_mug, y_mug, z_grasp])
    p_lift          = np.array([x_mug, y_mug, z_lift])

    print("p_high_over_mug:", p_high_over_mug)
    print("p_over_mug     :", p_over_mug)
    print("p_grasp        :", p_grasp)
    print("p_lift         :", p_lift)

    meshcat.SetObject("debug/over_mug",
                      Box(0.03, 0.03, 0.03), Rgba(0, 1, 0, 0.4))
    meshcat.SetTransform("debug/over_mug",
                         RigidTransform(R_WG_desired, p_over_mug))
    meshcat.SetObject("debug/grasp",
                      Box(0.03, 0.03, 0.03), Rgba(1, 0, 0, 0.4))
    meshcat.SetTransform("debug/grasp",
                         RigidTransform(R_WG_desired, p_grasp))

    # ------------------ Place-side keypoints (table edge) ------------------ #
    p_table_center = p_WT.copy()
    print("Table center (x, y):", p_table_center[:2])

    robot_xy = np.array([0.0, 0.0])
    table_xy = p_table_center[:2]
    direction = table_xy - robot_xy
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        direction = np.array([0.0, -1.0])
        norm = 1.0
    direction /= norm

    EDGE_DISTANCE = 0.35
    p_place_xy = table_xy - EDGE_DISTANCE * direction   # towards robot

    safe_over_table_z = table_height + SAFE_CLEARANCE

    p_over_table = np.array([
        p_place_xy[0],
        p_place_xy[1],
        safe_over_table_z,
    ])
    p_place = np.array([
        p_place_xy[0],
        p_place_xy[1],
        table_height + PLACE_ABOVE_TABLE,
    ])
    p_retreat = p_over_table.copy()

    print("p_place_xy (edge-ish):", p_place_xy)
    print("p_over_table:", p_over_table)
    print("p_place     :", p_place)

    # ------------------ IK helpers ------------------ #

    def ik_at(p_W, use_orientation=True, strict=True, pos_tol=0.01):
        """IK to put WSG body at p_W (and optionally oriented down)."""
        R_des = R_WG_desired if use_orientation else None
        return ik_for_ee_position(
            plant,
            plant_context,
            iiwa_model,
            wsg_model,
            p_W_target=p_W,
            R_WG_desired=R_des,
            theta_bound=1.0,
            pos_tol=pos_tol,
            strict=strict,
        )

    # ------------------ Solve IK for key poses ------------------ #
    q_start_full = plant.GetPositions(plant_context, iiwa_model)

    # High approach: relaxed position box, but oriented down
    q_high_over_m = ik_at(p_high_over_mug, use_orientation=True,
                          strict=False, pos_tol=0.05)
    # Over mug: stricter
    q_over_m = ik_at(p_over_mug, use_orientation=True,
                     strict=True, pos_tol=0.01)
    # Grasp pose: very strict (this is the one that must be "exact")
    q_grasp = ik_at(p_grasp, use_orientation=True,
                    strict=True, pos_tol=0.005)
    # Lift pose: strict-ish
    q_lift = ik_at(p_lift, use_orientation=True,
                   strict=True, pos_tol=0.01)

    # Table side: no orientation constraint; relaxed position tolerance
    q_over_t = ik_at(p_over_table, use_orientation=False,
                     strict=False, pos_tol=0.25)
    q_place_q = ik_at(p_place, use_orientation=False,
                      strict=False, pos_tol=0.25)
    q_retreat = q_over_t.copy()

    # ------------------ Low-level motion helpers ------------------ #
    def move_to(q_des_full, gripper_mode: str, dt: float):
        nonlocal t
        cmd_source.set_q_desired(q_des_full)

        if gripper_mode == "open":
            wsg_cmd_source.set_width(WSG_OPEN)
        elif gripper_mode == "closed":
            wsg_cmd_source.set_width(WSG_CLOSED)

        simulator.AdvanceTo(t + dt)
        t += dt
        diagram.ForcedPublish(context)

    def vertical_sweep(z_start, z_end, n_steps, gripper_mode):
        """Move straight in z over the mug (fixed x,y), recomputing IK each step."""
        for alpha in np.linspace(0.0, 1.0, n_steps):
            z = (1 - alpha) * z_start + alpha * z_end
            p = np.array([x_mug, y_mug, z])
            q = ik_at(p, use_orientation=True,
                      strict=True, pos_tol=0.01)
            move_to(q, gripper_mode, dt=PHASE_DT / n_steps)

    # ------------------ Initialize ------------------ #
    plant.SetPositions(plant_context, iiwa_model, q_start_full)
    plant.SetVelocities(plant_context, iiwa_model,
                        np.zeros_like(q_start_full))
    cmd_source.set_q_desired(q_start_full)
    wsg_cmd_source.set_width(WSG_OPEN)
    diagram.ForcedPublish(context)

    t = context.get_time()

    # ------------------ 8-step script ------------------ #
    print("Executing full pick-and-place sequence (HardwareStation)...")

    # 1) start → high over mug (open)
    print(" -> start")
    move_to(q_start_full, "open", dt=PHASE_DT)

    print(" -> high_over_mug")
    move_to(q_high_over_m, "open", dt=PHASE_DT)

    # # 2) high → over_mug, vertical sweep (open)
    # print(" -> descend (high → over_mug, open)")
    # vertical_sweep(z_start=p_high_over_mug[2],
    #                z_end=p_over_mug[2],
    #                n_steps=6,
    #                gripper_mode="open")
    
    # 2) high → over_mug (just move in joint space; still above mug)
    print(" -> over_mug")
    move_to(q_over_m, "open", dt=PHASE_DT)

    # 3) over_mug → grasp, vertical sweep (open)
    print(" -> descend (over_mug → grasp, open)")
    vertical_sweep(z_start=p_over_mug[2],
                   z_end=p_grasp[2],
                   n_steps=10,
                   gripper_mode="open")

    # 4) small hold at bottom, still open (ensure fully down)
    print(" -> hold_bottom (open)")
    move_to(q_grasp, "open", dt=HOLD_DT)

    # # 5) now close in place (NO motion)
    # print(" -> close (in place)")
    # move_to(q_grasp, "closed", dt=HOLD_DT + 3)
    
    # After vertical_sweep down:
    print(" -> hold_bottom (open)")
    q_at_bottom = plant.GetPositions(plant_context, iiwa_model)
    move_to(q_at_bottom, "open", dt=HOLD_DT)

    print(" -> close (in place)")
    move_to(q_at_bottom, "closed", dt=HOLD_DT + 3)


    # 6) lift straight up (grasp → lift), vertical sweep (closed)
    print(" -> lift (closed)")
    vertical_sweep(z_start=p_grasp[2],
                   z_end=p_lift[2],
                   n_steps=6,
                   gripper_mode="closed")

    # 7) move over table edge (closed)
    print(" -> over_table")
    move_to(q_over_t, "closed", dt=PHASE_DT)

    # 8) place and open, then retreat
    print(" -> place")
    move_to(q_place_q, "open", dt=PHASE_DT)

    print(" -> retreat")
    move_to(q_retreat, "open", dt=PHASE_DT)



# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    meshcat = StartMeshcat()
    print("Meshcat URL:", meshcat.web_url())

    # Load scenario and create HardwareStation
    with open(SCENARIO_PATH, "r") as f:
        scenario_yaml = f.read()
    scenario = LoadScenario(data=scenario_yaml)

    builder = DiagramBuilder()
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat=meshcat))
    plant = station.GetSubsystemByName("plant")
    scene_graph = station.GetSubsystemByName("scene_graph")

    # Ensure we have a renderer (often already there)
    renderer_name = "renderer"
    if not scene_graph.HasRenderer(renderer_name):
        scene_graph.AddRenderer(
            renderer_name, MakeRenderEngineVtk(RenderEngineVtkParams())
        )

    # Models
    iiwa_model = plant.GetModelInstanceByName("iiwa_arm")
    wsg_model = plant.GetModelInstanceByName("wsg_arm")

    # Initial iiwa positions (full 10-dof vector)
    default_context = plant.CreateDefaultContext()
    q_start_full = plant.GetPositions(default_context, iiwa_model)

    # Command source → station's iiwa_arm.desired_state input port (size 2*Nq)
    cmd_source = builder.AddSystem(JointPositionCommandSource(q_start_full))
    iiwa_desired_state_port = station.GetInputPort("iiwa_arm.desired_state")
    builder.Connect(
        cmd_source.get_output_port(),   # 2*Nq-dim [q_des, v_des]
        iiwa_desired_state_port,
    )

    # --- WSG command wiring ---
    wsg_cmd_source = builder.AddSystem(WsgCommandSource(WSG_OPEN))
    builder.Connect(
        wsg_cmd_source.get_output_port(),
        station.GetInputPort("wsg_arm.position"),
    )


    wsg_force_limit_source = builder.AddSystem(ConstantVectorSource([40.0]))
    builder.Connect(
        wsg_force_limit_source.get_output_port(),
        station.GetInputPort("wsg_arm.force_limit"),
    )

    # --- Add point clouds like in the class ICP notebook ---
    # (We pass meshcat here so AddPointClouds can use the cameras, but we no
    #  longer explicitly render the point clouds ourselves.)
    to_point_cloud = AddPointClouds(
        scenario=scenario,
        station=station,
        builder=builder,
        meshcat=None,
    )

    # Export camera point cloud ports as "camera0.point_cloud", etc.
    for name, system in to_point_cloud.items():
        builder.ExportOutput(
            system.GetOutputPort("point_cloud"),
            f"{name}.point_cloud",
        )

    # --------------------------------------------------------------
    # Build diagram and run
    # --------------------------------------------------------------
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    # Let things settle (mug falls onto tray)
    simulator.Initialize()
    simulator.AdvanceTo(1.0)
    diagram.ForcedPublish(context)

    # Build rim cloud and run ICP (matches class ICP pattern)
    rim_pc = build_rim_pointcloud(diagram, context, meshcat)
    X_WM_hat, mug_bottom_z, mug_top_z, mug_center_xyz = estimate_mug_pose_icp(meshcat, rim_pc)


    print("Estimated mug pose:", X_WM_hat)
    print("mug_bottom_z:", mug_bottom_z, "mug_top_z:", mug_top_z)

    # Optional: ground truth comparison if mug1 exists
    try:
        mug_instance = plant.GetModelInstanceByName("mug1")
        mug_body = plant.GetBodyByName("base_link", mug_instance)
        X_WM_true = plant.EvalBodyPoseInWorld(plant_context, mug_body)
        print("True mug pose:", X_WM_true)
    except RuntimeError:
        pass

    meshcat.StartRecording()
    
    run_pick_place(
        meshcat,
        simulator,
        diagram,
        station,
        plant,
        plant_context,
        iiwa_model,
        wsg_model,
        cmd_source,
        wsg_cmd_source,
        X_WM_hat,
        mug_bottom_z,
        mug_top_z,
        mug_center_xyz,      # <--- new
    )


    meshcat.StopRecording()
    meshcat.PublishRecording()

    print("Done; keeping Meshcat open.")
    input("Simulation done. Press Enter to exit...")
    # while True:
    #     time.sleep(1)


def visualize_scene():
    """Builds the HardwareStation, simulates briefly, and visualizes the scene + rim"""
    meshcat = StartMeshcat()
    print("Meshcat URL:", meshcat.web_url())

    # Load scenario and create HardwareStation
    with open(SCENARIO_PATH, "r") as f:
        scenario_yaml = f.read()
    scenario = LoadScenario(data=scenario_yaml)

    builder = DiagramBuilder()
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat=meshcat))
    plant = station.GetSubsystemByName("plant")
    scene_graph = station.GetSubsystemByName("scene_graph")

    # Ensure renderer
    renderer_name = "renderer"
    if not scene_graph.HasRenderer(renderer_name):
        scene_graph.AddRenderer(
            renderer_name, MakeRenderEngineVtk(RenderEngineVtkParams())
        )

    iiwa_model = plant.GetModelInstanceByName("iiwa_arm")
    wsg_model = plant.GetModelInstanceByName("wsg_arm")

    default_context = plant.CreateDefaultContext()
    q_start_full = plant.GetPositions(default_context, iiwa_model)

    cmd_source = builder.AddSystem(JointPositionCommandSource(q_start_full))
    builder.Connect(
        cmd_source.get_output_port(),   # 2*Nq-dim [q_des, v_des]
        station.GetInputPort("iiwa_arm.desired_state"),
    )

    wsg_position_source = builder.AddSystem(ConstantVectorSource([WSG_OPEN]))
    builder.Connect(
        wsg_position_source.get_output_port(),
        station.GetInputPort("wsg_arm.position"),
    )

    wsg_force_limit_source = builder.AddSystem(ConstantVectorSource([40.0]))
    builder.Connect(
        wsg_force_limit_source.get_output_port(),
        station.GetInputPort("wsg_arm.force_limit"),
    )

    to_point_cloud = AddPointClouds(
        scenario=scenario,
        station=station,
        builder=builder,
        meshcat=None,      # don't draw raw camera clouds
    )
    for name, system in to_point_cloud.items():
        builder.ExportOutput(
            system.GetOutputPort("point_cloud"),
            f"{name}.point_cloud",
        )

    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()

    simulator.Initialize()
    simulator.AdvanceTo(1.0)   # let mug settle
    diagram.ForcedPublish(context)

    # visualize just rim pc
    try:
        rim_pc = build_rim_pointcloud(diagram, context, meshcat)
        print("Rim point cloud visualized at Meshcat path 'debug/rim_points'.")
    except Exception as e:
        print("Skipping rim point cloud viz due to error:", e)

    print("Scene is live in Meshcat.")
    input("Press Enter to exit visualization...\n")




if __name__ == "__main__":
    main()
    # visualize_scene()
