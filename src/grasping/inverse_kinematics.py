"""
Inverse kinematics utilities for mobile manipulator with base and joint locking.
"""

import numpy as np
from pydrake.all import (
    InverseKinematics,
    RotationMatrix,
    Solve,
)


def solve_ik(
    plant,
    plant_context,
    iiwa_model,
    wsg_model,
    p_W_target,
    R_WG_desired=None,
    position_tolerance=0.01,
    lock_base=False,
    theta_bound=0.2,
    base_positions_to_lock=None,
):
    """
    Solve IK for gripper to reach target position (and optionally orientation).

    This solver supports locking the mobile base and first arm joint to keep
    the robot stable during manipulation tasks.

    Args:
        plant: MultibodyPlant
        plant_context: plant context
        iiwa_model: iiwa model instance
        wsg_model: wsg gripper model instance
        p_W_target: target position in world frame (3,) array
        R_WG_desired: desired gripper orientation (RotationMatrix) or None
        position_tolerance: tolerance for position constraint (meters)
        lock_base: if True, lock mobile base joints and first arm joint
        theta_bound: orientation tolerance (radians)
        base_positions_to_lock: dict of {joint_name: value} for explicit locking

    Returns:
        q_iiwa: joint positions for iiwa (10-dof for mobile base + arm)

    Raises:
        RuntimeError: if IK solver fails to find a solution
    """

    q_seed = plant.GetPositions(plant_context)

    ik = InverseKinematics(plant, plant_context)
    q_decision = ik.q()
    world_frame = plant.world_frame()
    ee_frame = plant.GetFrameByName("body", wsg_model)

    # lock mobile base joints and first arm joint (to keep tray stable)
    joints_to_lock = ["iiwa_base_x", "iiwa_base_y", "iiwa_base_z", "iiwa_joint_1"]
    locked_values = {}

    if lock_base:
        for joint_name in joints_to_lock:
            try:
                joint = plant.GetJointByName(joint_name, iiwa_model)
                idx = joint.position_start()

                # Use explicitly provided position if available, otherwise use current
                if base_positions_to_lock and joint_name in base_positions_to_lock:
                    q_val = base_positions_to_lock[joint_name]
                else:
                    q_val = q_seed[idx]

                locked_values[joint_name] = q_val

                ik.prog().AddBoundingBoxConstraint(
                    q_val, q_val, q_decision[idx : idx + 1]
                )
            except RuntimeError:
                continue

    # position constraint (small box around target)
    tol = position_tolerance
    lower = p_W_target - tol
    upper = p_W_target + tol

    ik.AddPositionConstraint(
        ee_frame,
        [0, 0, 0],
        world_frame,
        lower,
        upper,
    )

    # orientation constraint if specified
    if R_WG_desired is not None:
        ik.AddOrientationConstraint(
            world_frame,
            R_WG_desired,
            ee_frame,
            RotationMatrix(),
            theta_bound,
        )

    prog = ik.prog()
    prog.SetInitialGuess(q_decision, q_seed)

    result = Solve(prog)
    if not result.is_success():
        print(f"\n!!! IK FAILED !!!")
        print(f"  Target position: {p_W_target}")
        print(f"  Position tolerance: ±{position_tolerance}m")
        print(f"  Orientation constraint: {R_WG_desired is not None}")
        print(f"  Base locked: {lock_base}")
        if R_WG_desired is not None:
            print(f"  Theta bound: {theta_bound} rad")
        raise RuntimeError(f"IK failed for target {p_W_target}")

    q_sol_full = result.GetSolution(q_decision)

    # don't modify plant_context - keep it clean
    temp_context = plant.CreateDefaultContext()
    plant.SetPositions(temp_context, q_sol_full)
    q_iiwa = plant.GetPositions(temp_context, iiwa_model)

    return np.copy(q_iiwa)


def get_locked_joint_positions(plant, plant_context, iiwa_model):
    """
    get current positions of joints to lock (base + first arm joint)
    """

    q_plant_full = plant.GetPositions(plant_context)

    # Get joints to lock from FULL plant state
    base_x_joint = plant.GetJointByName("iiwa_base_x", iiwa_model)
    base_y_joint = plant.GetJointByName("iiwa_base_y", iiwa_model)
    base_z_joint = plant.GetJointByName("iiwa_base_z", iiwa_model)
    
    # we must lock the first joint as the tray is welded to this joint
    joint_1 = plant.GetJointByName("iiwa_joint_1", iiwa_model)

    return {
        "iiwa_base_x": q_plant_full[base_x_joint.position_start()],
        "iiwa_base_y": q_plant_full[base_y_joint.position_start()],
        "iiwa_base_z": q_plant_full[base_z_joint.position_start()],
        "iiwa_joint_1": q_plant_full[joint_1.position_start()],
    }
