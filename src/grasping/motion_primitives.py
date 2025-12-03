"""
Motion primitives for robotic manipulation tasks.
"""

import numpy as np
from pydrake.all import (
    Rgba,
    RigidTransform,
    RotationMatrix,
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
    Execute simple 4-step pick sequence:
    1. Approach position (above object)
    2. Descend to grasp
    3. Close gripper
    4. Lift up

    Args:
        meshcat: Meshcat visualizer
        simulator: Drake simulator
        diagram: Drake diagram
        plant: MultibodyPlant
        plant_context: plant context
        iiwa_model: iiwa model instance
        wsg_model: wsg gripper model instance
        cmd_source: Joint position command source
        wsg_cmd_source: Gripper command source
        grasp_center_xyz: (x, y, z) grasp position
        object_top_z: z-coordinate of object top (unused but kept for compatibility)
        approach_height: Height above grasp to approach from (meters)
        lift_height: Height to lift after grasp (meters)
        wsg_open: Open gripper width (meters)
        wsg_closed: Closed gripper width (meters)
        move_time: Time for each motion phase (seconds)
        grasp_time: Time to close gripper (seconds)
        lift_time: Time to lift object (seconds)
    """

    print("\n=== PICK SEQUENCE ===")

    context = simulator.get_mutable_context()

    # Gripper orientation: pointing down
    R_WG_down = RotationMatrix.MakeXRotation(-np.pi / 2.0)

    # Key positions
    x, y, z_grasp = grasp_center_xyz
    z_grasp += 0.05  # Small offset to account for gripper finger length

    p_approach = np.array([x, y, z_grasp + approach_height])
    p_grasp = np.array([x, y, z_grasp])
    p_lift = np.array([x, y, z_grasp + lift_height])
    p_lift_higher = np.array([x, y, z_grasp + lift_height + 0.1])  # 10cm higher

    # Target place position - offset from current robot base position
    # Get current robot base position
    q_current = plant.GetPositions(plant_context, iiwa_model)
    current_base_x = q_current[0]  # iiwa_base_x
    current_base_y = q_current[1]  # iiwa_base_y

    # Customize these offsets relative to current base position
    place_offset_x = -0.40  # Offset in x from current base position
    place_offset_y = 0.3   # Offset in y from current base position

    place_x = current_base_x + place_offset_x
    place_y = current_base_y + place_offset_y
    place_z = p_lift_higher[2]  # Same height as lift_higher
    p_place_target = np.array([place_x, place_y, place_z])

    print(f"Approach:      {p_approach}")
    print(f"Grasp:         {p_grasp}")
    print(f"Lift:          {p_lift}")
    print(f"Lift higher:   {p_lift_higher}")
    print(f"Current base:  x={current_base_x:.3f}, y={current_base_y:.3f}")
    print(f"Place offset:  dx={place_offset_x:.3f}, dy={place_offset_y:.3f}")
    print(f"Place target:  {p_place_target}")

    # Visualize trajectory
    meshcat.SetObject("traj/approach", Box(0.015, 0.015, 0.015), Rgba(0, 1, 0, 0.5))
    meshcat.SetTransform("traj/approach", RigidTransform(p_approach))

    meshcat.SetObject("traj/grasp", Box(0.015, 0.015, 0.015), Rgba(1, 0, 0, 0.8))
    meshcat.SetTransform("traj/grasp", RigidTransform(p_grasp))

    meshcat.SetObject("traj/lift", Box(0.015, 0.015, 0.015), Rgba(0, 0, 1, 0.5))
    meshcat.SetTransform("traj/lift", RigidTransform(p_lift))

    meshcat.SetObject("traj/lift_higher", Box(0.015, 0.015, 0.015), Rgba(1, 1, 0, 0.5))
    meshcat.SetTransform("traj/lift_higher", RigidTransform(p_lift_higher))

    # Visualize place target position (magenta box, gripper pointing down)
    meshcat.SetObject("traj/place_target", Box(0.03, 0.03, 0.03), Rgba(1, 0, 1, 0.8))
    meshcat.SetTransform("traj/place_target", RigidTransform(R_WG_down, p_place_target))

    print("\nSolving IK for key poses...")

    base_positions_lock = get_locked_joint_positions(plant, plant_context, iiwa_model)

    print(
        f"Locking base at: x={base_positions_lock['iiwa_base_x']:.3f}, "
        f"y={base_positions_lock['iiwa_base_y']:.3f}, "
        f"z={base_positions_lock['iiwa_base_z']:.3f}"
    )
    print(f"Locking joint_1 at: {base_positions_lock['iiwa_joint_1']:.3f} rad")

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
    print("  ✓ Approach pose")

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
    print("  ✓ Grasp pose")

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
    print("  ✓ Lift pose")

    # Lift higher: Base locked, relaxed tolerance
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
    print("  ✓ Lift higher pose")

    # Place target: Base locked, moderate tolerance
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
    print("  ✓ Place target pose")

    # motion helper
    def move_to(q_des, gripper_width, duration):
        """move robot to joint configuration with specified gripper width"""
        cmd_source.set_q_desired(q_des)
        wsg_cmd_source.set_width(gripper_width)
        t = simulator.get_context().get_time()
        simulator.AdvanceTo(t + duration)
        diagram.ForcedPublish(context)

        # Monitor contact forces
        tau_contact = plant.get_generalized_contact_forces_output_port(wsg_model).Eval(plant_context)
        total_force = np.sum(np.abs(tau_contact))
        print(f"    Contact forces - Total: {total_force:.4f} N, Gripper width: {gripper_width:.4f} m")

    # initializaiton stuff
    plant.SetPositions(plant_context, iiwa_model, q_start)
    plant.SetVelocities(plant_context, iiwa_model, np.zeros_like(q_start))
    cmd_source.set_q_desired(q_start)
    wsg_cmd_source.set_width(wsg_open)
    diagram.ForcedPublish(context)

    # execute pick sequence
    print("\nExecuting motion...")

    print("1. move to start position")
    move_to(q_start, wsg_open, move_time)

    print("2. approach above object")
    move_to(q_approach, wsg_open, move_time)

    print("3. descend to grasp position")
    move_to(q_grasp, wsg_open, move_time)

    print("4. close gripper to grasp")
    move_to(q_grasp, wsg_closed, grasp_time)

    print("5. lift object (slower)")
    move_to(q_lift, wsg_closed, lift_time)

    print("6. lift 10cm higher")
    move_to(q_lift_higher, wsg_closed, lift_time)

    print("7. move to place target")
    move_to(q_place, wsg_closed, move_time)

    print("7.5. hold at place target to stabilize grip")
    move_to(q_place, wsg_closed, 2.0)  # Hold for 2 seconds to stabilize

    # Sync plant_context with actual robot state after motion
    q_actual = plant.GetPositions(plant_context, iiwa_model)
    print(f"  Robot state after place: base=[{q_actual[0]:.3f}, {q_actual[1]:.3f}, {q_actual[2]:.3f}]")

    print("8. incremental descent (2cm steps, monitoring forces)")
    z_current = p_place_target[2]
    descent_step = 0.02  # 2cm per step
    step_time = 0.5      # Slower to maintain grip stability
    step_count = 0

    # Start with the place position as seed
    q_current = q_place

    while True:
        step_count += 1
        z_current -= descent_step
        p_descent = np.array([p_place_target[0], p_place_target[1], z_current])

        print(f"  Step {step_count}: trying z={z_current:.3f}m (target={p_descent}) - ", end="")

        # Update plant_context with current position to use as IK seed
        plant.SetPositions(plant_context, iiwa_model, q_current)

        # Verify base is still locked
        q_check = plant.GetPositions(plant_context, iiwa_model)
        print(f"seed_base=[{q_check[0]:.3f},{q_check[1]:.3f}] - ", end="")

        # Solve IK with strict downward orientation
        try:
            q_descent = solve_ik(
                plant,
                plant_context,
                iiwa_model,
                wsg_model,
                p_descent,
                R_WG_down,
                position_tolerance=0.10,  # Large tolerance for incremental steps
                lock_base=True,
                theta_bound=0.5,  # Relaxed orientation
                base_positions_to_lock=base_positions_lock,
            )

            # Execute motion with tighter gripper control
            # Re-command gripper width to ensure it stays closed
            wsg_cmd_source.set_width(wsg_closed)
            move_to(q_descent, wsg_closed, step_time)

            # Check contact forces after motion
            tau_contact = plant.get_generalized_contact_forces_output_port(wsg_model).Eval(plant_context)
            total_force = np.sum(np.abs(tau_contact))

            # If forces drop too low, we may be losing grip
            if total_force < 0.1:
                print(f"\n  WARNING: Low contact force detected ({total_force:.4f} N) - grip may be weakening!")

            # Update current position for next iteration
            q_current = q_descent

        except RuntimeError as e:
            print(f"IK failed - stopping descent")
            print(f"  Error: {e}")
            break

    print("9. open gripper to release")
    # move_to(q_descent, wsg_open, grasp_time)

    print("\n✓ pick and place sequence complete!")
