"""
Motion primitives for robotic manipulation tasks.
"""

import numpy as np
from pydrake.all import (
    Rgba,
    RigidTransform,
    RotationMatrix,
    PiecewisePolynomial,
)
from pydrake.geometry import Box

from grasping.inverse_kinematics import solve_ik, get_locked_joint_positions


def pick_object(
    meshcat,
    simulator,
    diagram,
    plant,
    plant_context,
    iiwa_model,
    wsg_model,
    cmd_source,
    wsg_cmd_source,
    grasp_center_xyz,
    object_top_z,
    approach_height=0.15,
    lift_height=0.20,
    wsg_open=0.107,
    wsg_closed=0.080,
    move_time=2.5,
    grasp_time=2.0,
    lift_time=4.0,
):
    """
    execute simple 4-step pick sequence:
    1. approach position (above object)
    2. descend to grasp
    3. close gripper
    4. lift up
    """
    context = simulator.get_mutable_context()

    # gripper orientation: pointing down
    R_WG_down = RotationMatrix.MakeXRotation(-np.pi / 2.0)

    # key positions
    x, y, z_grasp = grasp_center_xyz
    z_grasp += 0.05  # small offset to account for gripper finger length

    p_approach = np.array([x, y, z_grasp + approach_height])
    p_grasp = np.array([x, y, z_grasp])
    p_lift = np.array([x, y, z_grasp + lift_height])
    p_lift_higher = np.array([x, y, z_grasp + lift_height + 0.1])  # 10cm higher

    q_current = plant.GetPositions(plant_context, iiwa_model)
    current_base_x = q_current[0]
    current_base_y = q_current[1]

    # offsets relative to current base position
    place_offset_x = -0.40
    place_offset_y = 0.3

    # ignore for now; maybe reconciliation
    # place_offset_x = 0.3  # Offset in x from current base position
    # place_offset_y = 0.4   # Offset in y from current base position

    place_x = current_base_x + place_offset_x
    place_y = current_base_y + place_offset_y
    place_z = p_lift_higher[2]
    p_place_target = np.array([place_x, place_y, place_z])

    meshcat.SetObject("traj/approach", Box(0.015, 0.015, 0.015), Rgba(0, 1, 0, 0.5))
    meshcat.SetTransform("traj/approach", RigidTransform(p_approach))

    meshcat.SetObject("traj/grasp", Box(0.015, 0.015, 0.015), Rgba(1, 0, 0, 0.8))
    meshcat.SetTransform("traj/grasp", RigidTransform(p_grasp))

    meshcat.SetObject("traj/lift", Box(0.015, 0.015, 0.015), Rgba(0, 0, 1, 0.5))
    meshcat.SetTransform("traj/lift", RigidTransform(p_lift))

    meshcat.SetObject("traj/lift_higher", Box(0.015, 0.015, 0.015), Rgba(1, 1, 0, 0.5))
    meshcat.SetTransform("traj/lift_higher", RigidTransform(p_lift_higher))

    meshcat.SetObject("traj/place_target", Box(0.03, 0.03, 0.03), Rgba(1, 0, 1, 0.8))
    meshcat.SetTransform("traj/place_target", RigidTransform(R_WG_down, p_place_target))

    print("\nsolving ik for poses")

    base_positions_lock = get_locked_joint_positions(plant, plant_context, iiwa_model)
    q_start = plant.GetPositions(plant_context, iiwa_model)

    q_approach = solve_ik(
        plant,
        plant_context,
        iiwa_model,
        wsg_model,
        p_approach,
        R_WG_down,
        position_tolerance=0.02,
        lock_base=True,
        theta_bound=0.3,
        base_positions_to_lock=base_positions_lock,
    )

    q_grasp = solve_ik(
        plant,
        plant_context,
        iiwa_model,
        wsg_model,
        p_grasp,
        R_WG_down,
        position_tolerance=0.02,
        lock_base=True,
        theta_bound=0.3,
        base_positions_to_lock=base_positions_lock,
    )

    q_lift = solve_ik(
        plant,
        plant_context,
        iiwa_model,
        wsg_model,
        p_lift,
        R_WG_down,
        position_tolerance=0.08,
        lock_base=True,
        theta_bound=0.8,
        base_positions_to_lock=base_positions_lock,
    )

    q_lift_higher = solve_ik(
        plant,
        plant_context,
        iiwa_model,
        wsg_model,
        p_lift_higher,
        R_WG_down,
        position_tolerance=0.08,
        lock_base=True,
        theta_bound=0.8,
        base_positions_to_lock=base_positions_lock,
    )

    q_place = solve_ik(
        plant,
        plant_context,
        iiwa_model,
        wsg_model,
        p_place_target,
        R_WG_down,
        position_tolerance=0.08,
        lock_base=True,
        theta_bound=0.8,
        base_positions_to_lock=base_positions_lock,
    )

    p_descent_target = np.array([p_place_target[0], p_place_target[1], p_place_target[2] - 0.40])
    q_descent = solve_ik(
        plant,
        plant_context,
        iiwa_model,
        wsg_model,
        p_descent_target,
        R_WG_down,
        position_tolerance=0.08,
        lock_base=True,
        theta_bound=0.8,
        base_positions_to_lock=base_positions_lock,
    )

    # motion helper with smooth trajectory interpolation
    def move_to_smooth(q_des, gripper_width, duration, num_steps=50, monitor_collision=False, baseline_force=0.0):
        """smoothly interpolate to joint configuration with specified gripper width"""
        q_current = plant.GetPositions(plant_context, iiwa_model)

        times = [0.0, duration]
        q_knots = np.column_stack([q_current, q_des])
        trajectory = PiecewisePolynomial.CubicWithContinuousSecondDerivatives(
            times, q_knots, np.zeros(len(q_current)), np.zeros(len(q_des))
        )

        wsg_cmd_source.set_width(gripper_width)
        t_start = simulator.get_context().get_time()
        dt = duration / num_steps
        collision_detected = False

        for i in range(num_steps + 1):
            t_rel = i * dt 
            if t_rel > duration:
                t_rel = duration

            q_interp = trajectory.value(t_rel).flatten()
            cmd_source.set_q_desired(q_interp)
            t_abs = t_start + t_rel
            tau_contact = plant.get_generalized_contact_forces_output_port(wsg_model).Eval(plant_context)
            total_force = np.sum(np.abs(tau_contact))

            # check for collision if monitoring is enabled
            if monitor_collision and total_force > baseline_force + 5.0:
                collision_detected = True
                break

            simulator.AdvanceTo(t_abs)
            diagram.ForcedPublish(context)

        return collision_detected

        
    # initializaiton stuff
    plant.SetPositions(plant_context, iiwa_model, q_start)
    plant.SetVelocities(plant_context, iiwa_model, np.zeros_like(q_start))
    cmd_source.set_q_desired(q_start)
    wsg_cmd_source.set_width(wsg_open)
    diagram.ForcedPublish(context)

    # execute pick sequence

    print("1. move to start position")
    move_to_smooth(q_start, wsg_open, move_time)

    print("2. approach above object")
    move_to_smooth(q_approach, wsg_open, move_time)

    print("3. descend to grasp position")
    move_to_smooth(q_grasp, wsg_open, move_time)

    print("4. close gripper to grasp")
    move_to_smooth(q_grasp, wsg_closed, grasp_time)

    print("5. lift object (slower)")
    move_to_smooth(q_lift, wsg_closed, lift_time)

    # save baseline contact force after lifting object
    tau_contact_baseline = plant.get_generalized_contact_forces_output_port(wsg_model).Eval(plant_context)
    baseline_force = np.sum(np.abs(tau_contact_baseline))

    print("6. lift 10cm higher")
    move_to_smooth(q_lift_higher, wsg_closed, lift_time)

    print("7. move to place target")
    move_to_smooth(q_place, wsg_closed, move_time)

    print("8. hold at place target to stabilize grip")
    move_to_smooth(q_place, wsg_closed, 2.0)  # Hold for 2 seconds to stabilize

    q_actual = plant.GetPositions(plant_context, iiwa_model)
    collision = move_to_smooth(q_descent, wsg_closed, 5.0, monitor_collision=True, baseline_force=baseline_force)

    if collision:
        wsg_cmd_source.set_width(wsg_open)
        t = simulator.get_context().get_time()
        simulator.AdvanceTo(t + 2.0)  
        diagram.ForcedPublish(context)

    if not collision:
        wsg_cmd_source.set_width(wsg_open)
        t = simulator.get_context().get_time()
        simulator.AdvanceTo(t + grasp_time)
        diagram.ForcedPublish(context)
        
    move_to_smooth(q_place, wsg_open, move_time)
    move_to_smooth(q_start, wsg_open, move_time)

