from pathlib import Path
import time
import mpld3
import numpy as np
from pydrake.all import (
    DiagramBuilder,
    InverseKinematics,
    PiecewisePolynomial,
    RotationMatrix,
    Simulator,
    StartMeshcat,
    TrajectorySource,
    Solve,
)
from pydrake.systems.primitives import ConstantVectorSource

from manipulation import running_as_notebook
from manipulation.station import LoadScenario, MakeHardwareStation

WAIT_BEFORE_MOVE = 1.0  # seconds to wait before the free arm starts moving

# Meshcat + scenario
# -----------------------------------------------------------------------------
if running_as_notebook:
    mpld3.enable_notebook()

meshcat = StartMeshcat()

scenario_file = Path("/workspaces/robman-final-proj/src/scenario.yaml")
with open(scenario_file, "r") as f:
    scenario_yaml = f.read()
scenario = LoadScenario(data=scenario_yaml)

builder = DiagramBuilder()
station = builder.AddSystem(MakeHardwareStation(scenario, meshcat=meshcat))
plant = station.GetSubsystemByName("plant")


# planning context (like interactive_ik.ipynb)
# -----------------------------------------------------------------------------
station_plan_context = station.CreateDefaultContext()
plant_plan_context = plant.GetMyContextFromRoot(station_plan_context)

# model instances
iiwa_arm = plant.GetModelInstanceByName("iiwa_arm") # free arm
wsg_arm = plant.GetModelInstanceByName("wsg_arm") # free arm gripper
iiwa_plate = plant.GetModelInstanceByName("iiwa_plate") # tray arm
wsg_plate = plant.GetModelInstanceByName("wsg_plate") # tray gripper

ee_frame = plant.GetFrameByName("body", wsg_arm) # end-effector frame
world_frame = plant.world_frame()

# object poses
# -----------------------------------------------------------------------------
# mug pose (on tray)
mug_instance = plant.GetModelInstanceByName("mug")
mug_body = plant.GetBodyByName("base_link", mug_instance)
X_WM = plant.EvalBodyPoseInWorld(plant_plan_context, mug_body)
p_WM = X_WM.translation()

# free arm base pose (iiwa_arm base link)
iiwa_base_body = plant.GetBodyByName("iiwa_link_0", iiwa_arm)
X_WB = plant.EvalBodyPoseInWorld(plant_plan_context, iiwa_base_body)
p_WB = X_WB.translation()

# table pose (target surface)
table_inst = plant.GetModelInstanceByName("table")
table_body = plant.GetBodyByName("table_body", table_inst)
X_WT = plant.EvalBodyPoseInWorld(plant_plan_context, table_body)
p_WT = X_WT.translation()

# init joint positions
q_start = plant.GetPositions(plant_plan_context, iiwa_arm)
q_plate_default = plant.GetPositions(plant_plan_context, iiwa_plate)


# Desired gripper orientation & side direction
# -----------------------------------------------------------------------------
# side approach towards free arm base in xy-plane
side_xy = p_WB[:2] - p_WM[:2]
side_xy_norm = np.linalg.norm(side_xy) # length of vector in XY plane

# side dir for gripper approach
side_dir = np.array([
    side_xy[0] / side_xy_norm,
    side_xy[1] / side_xy_norm,
    0.0,
])

# horizontal yaw that points the gripper from the arm toward the mug
yaw = np.arctan2(side_dir[1], side_dir[0])

# rotate extra 90 deg so fingers line up correctly with mug
# should approach from the side
R_WG_desired = RotationMatrix.MakeZRotation(yaw + np.pi / 2.0) 


# IK helper to reach a desired EE position in world
# -----------------------------------------------------------------------------
def ik_for_ee_position(p_W_target, use_orientation: bool):
    """
    Compute iiwa_arm joint positions so that wsg_arm::body is at p_W_target

    If use_orientation is True: enforce horizontal side-grasp orientation
    False: only a position constraint
    """
    ik = InverseKinematics(plant, plant_plan_context)
    q = ik.q()

    # position constraint
    p_tol = 0.005  # 5 mm box
    ik.AddPositionConstraint(
        ee_frame, [0, 0, 0],
        world_frame,
        p_W_target - p_tol,
        p_W_target + p_tol,
    )

    if use_orientation:
        ik.AddOrientationConstraint(
            world_frame,       # frameAbar
            R_WG_desired,      # R_AbarA
            ee_frame,          # frameBbar
            RotationMatrix(),  # R_BbarB (identity)
            0.20,              # theta_bound in radians (~11.5 deg)
        )

    prog = ik.prog()
    q_seed_full = plant.GetPositions(plant_plan_context)
    prog.SetInitialGuess(q, q_seed_full)

    result = Solve(prog) # solve using constraints
    if not result.is_success():
        raise RuntimeError(f"IK failed for target {p_W_target}")

    q_sol = result.GetSolution(q)
    plant.SetPositions(plant_plan_context, q_sol) # update context
    return plant.GetPositions(plant_plan_context, iiwa_arm)



# points in world for approach and place traj
# -----------------------------------------------------------------------------
z_body_offset = -0.02  # aim slightly below mug center
p_body = p_WM + np.array([0.0, 0.0, z_body_offset]) # grasp body point

mug_radius = 0.06   # rough radius
far_clearance  = mug_radius + 0.12   # pre-grasp a bit away from mug to avoid collision
near_clearance = mug_radius + 0.025  # closer grasp pose

# small offset in +y
y_offset = 0.1  # 10 cm

extra_in = 0.08 # 8.5 cm deeper toward the mug
z_offset =  0.015 # 0.015 offset up

p_pre_grasp = (
    p_body
    + side_dir * (far_clearance + extra_in)
    + np.array([0.0, y_offset, z_offset])
)

p_grasp = (
    p_body
    + side_dir * (near_clearance + extra_in)
    + np.array([0.0, y_offset, z_offset])
)

p_lift = p_grasp + np.array([0.0, 0.0, 0.18])

# over-table / place / retreat
table_height = p_WT[2] + 0.05  # small bump above the body frame

# choose reachable point near the edge of table so arm can reach
p_place_xy = np.array([1.0, -0.05])  # x,y near front edge of table, towards the arm

p_over_table = np.array([p_place_xy[0], p_place_xy[1], table_height + 0.20])
p_place = np.array([p_place_xy[0], p_place_xy[1], table_height + 0.02])
p_retreat = p_over_table.copy()

# Solve IK for each point along traj
# -----------------------------------------------------------------------------
# use orientations all true so that gripper is horizontal ("liquid" in mug wont spill)
q_pre = ik_for_ee_position(p_pre_grasp,  use_orientation=True)
q_grasp = ik_for_ee_position(p_grasp, use_orientation=True)
q_lift = ik_for_ee_position(p_lift, use_orientation=True)
q_over = ik_for_ee_position(p_over_table, use_orientation=True)
q_place = ik_for_ee_position(p_place, use_orientation=True)
q_retreat = ik_for_ee_position(p_retreat, use_orientation=True)


# rotate before appraoching mug
p_orient = p_body + side_dir * (far_clearance + 0.10)  # 15cm farther bac
q_orient = ik_for_ee_position(p_orient, use_orientation=True)


# piecewise arm traj
# -----------------------------------------------------------------------------

arm_knots = np.column_stack([
    q_start,   # initial
    q_start,   # hold
    q_orient,  # rotate far away
    q_pre,     # pre-grasp
    q_grasp,   # arrive at grasp
    q_grasp,   # hold at grasp  (for closing)
    q_lift,    # lift
    q_over,    # over table
    q_place,   # arrive at place
    q_place,   # hold at place  (for opening)
    q_retreat, # retreat
])

arm_times = np.array([
    0.0,
    WAIT_BEFORE_MOVE,        # hold start
    WAIT_BEFORE_MOVE + 1.0,  # rotate early
    WAIT_BEFORE_MOVE + 3.0,  # pre-grasp
    WAIT_BEFORE_MOVE + 5.0,  # arrive grasp
    WAIT_BEFORE_MOVE + 6.0,  # hold at grasp
    WAIT_BEFORE_MOVE + 8.0,  # lift
    WAIT_BEFORE_MOVE + 10.0, # over table
    WAIT_BEFORE_MOVE + 12.0, # arrive place
    WAIT_BEFORE_MOVE + 13.0, # hold at place
    WAIT_BEFORE_MOVE + 15.0, # retreat
])

assert arm_knots.shape[1] == len(arm_times)
arm_traj = PiecewisePolynomial.FirstOrderHold(arm_times, arm_knots)


# gripper trajectory: open -> close while stopped at grasp -> open while stopped at place
# -------------------------------------------------------------------------
t_close = WAIT_BEFORE_MOVE + 5.5   # between grasp and grasp_hold
t_open  = WAIT_BEFORE_MOVE + 12.5  # between place and place_hold
t_end   = arm_times[-1]

wsg_times = np.array([
    0.0,       # open initially
    t_close,   # close while arm is holding at grasp
    t_open,    # open while arm is holding at place
    t_end,     # stay open during retreat
])

# 0.1=open, 0.03=closed
CLOSE = 0.03   # 3 cm gap instead of 0.0 for collision safety
wsg_knots = np.array([[0.1, CLOSE, 0.1, 0.1]])
wsg_traj = PiecewisePolynomial.ZeroOrderHold(wsg_times, wsg_knots)


# Build runtime diagram
# -----------------------------------------------------------------------------
# tray arm fixed at its default pose (holding tray)
position_plate_source = builder.AddSystem(
    ConstantVectorSource(q_plate_default)
)
builder.Connect(
    position_plate_source.get_output_port(),
    station.GetInputPort("iiwa_plate.position")
)

# tray gripper stays closed
wsg_plate_source = builder.AddSystem(ConstantVectorSource([0.0]))
builder.Connect(
    wsg_plate_source.get_output_port(),
    station.GetInputPort("wsg_plate.position")
)

# free arm and gripper controlled by trajs
arm_traj_source = builder.AddSystem(TrajectorySource(arm_traj))
builder.Connect(
    arm_traj_source.get_output_port(),
    station.GetInputPort("iiwa_arm.position")
)

wsg_traj_source = builder.AddSystem(TrajectorySource(wsg_traj))
builder.Connect(
    wsg_traj_source.get_output_port(),
    station.GetInputPort("wsg_arm.position")
)


# Simulate
# -----------------------------------------------------------------------------
diagram = builder.Build()
simulator = Simulator(diagram)
context = simulator.get_mutable_context()

diagram.ForcedPublish(context)

if running_as_notebook:
    simulator.set_target_realtime_rate(1.0)

print("Meshcat URL:", meshcat.web_url())
meshcat.StartRecording()

T_final = max(arm_traj.end_time(), wsg_traj.end_time())
simulator.AdvanceTo(T_final)

time.sleep(5.0)
meshcat.PublishRecording()

input("Simulation done. Press Enter to exit...")

# TODO force control for better grasping
# TODO sample antipodal grasps so simpler finds better grasps
# TODO force control for placing mug on table, won't always know exact height