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
    object_cloud=None,
):
    """
    execute simple 4-step pick sequence:
    1. approach position (above object)
    2. descend to grasp
    3. close gripper
    4. lift up
    
    if object_cloud is provided, computes antipodal grasp pose
    Otherwise uses default downward grasp
    """
    context = simulator.get_mutable_context()

    R_WG_down = RotationMatrix.MakeXRotation(-np.pi / 2.0)

    # compute antipodal grasp if cloud provided
    X_WG_grasp = None
    if object_cloud is not None:
        from grasping.antipodal_grasping import get_best_antipodal_grasp
        print("Finding antipodal grasp...")
        X_WG_grasp = get_best_antipodal_grasp(
            object_cloud,
            rng=np.random.default_rng(),
            num_samples=500,
        )
        if X_WG_grasp is not None:
            print("Found antipodal grasp, executing with antipodal approach")
        else:
            print("No valid antipodal grasp found, falling back to downward grasp")
    
    # use antipodal grasp if found
    if X_WG_grasp is not None:
        print("Executing with antipodal grasp")
        R_WG = X_WG_grasp.rotation()
        p_grasp = X_WG_grasp.translation()
        
        # debug: visualize the grasp pose
        print(f"    Grasp position: {p_grasp}")
        print(f"    Grasp x-axis (into object): {R_WG.multiply(np.array([1, 0, 0]))}")
        
        # visualize grasp frame
        # meshcat.SetObject("grasp/antipodal_frame", Box(0.03, 0.03, 0.03), Rgba(0, 1, 1, 0.8))
        # meshcat.SetTransform("grasp/antipodal_frame", X_WG_grasp)
        
        # Approach: back up along gripper x-axis (normal direction)
        p_approach = p_grasp - approach_height * R_WG.multiply(np.array([1, 0, 0]))
        
        # Lift: back up along gripper x-axis
        p_lift = p_grasp - lift_height * R_WG.multiply(np.array([1, 0, 0]))
        p_lift_higher = p_grasp - (lift_height + 0.1) * R_WG.multiply(np.array([1, 0, 0]))
    else:
        # default when no antipodal grasp found downward grasp
        R_WG = R_WG_down
        
        x, y, z_grasp = grasp_center_xyz
        z_grasp += 0.05  # small offset to account for gripper finger length
        
        p_approach = np.array([x, y, z_grasp + approach_height])
        p_grasp = np.array([x, y, z_grasp])
        p_lift = np.array([x, y, z_grasp + lift_height])
        p_lift_higher = np.array([x, y, z_grasp + lift_height + 0.1])  # 10cm higher

    q_current = plant.GetPositions(plant_context, iiwa_model)
    current_base_x = q_current[0]
    current_base_y = q_current[1]
    current_theta = q_current[3]

    # place  at table edge offset from current base
    offset_distance = 0.5
    place_direction = current_theta + np.pi/2

    place_x = current_base_x + offset_distance * np.cos(place_direction)
    place_y = current_base_y + offset_distance * np.sin(place_direction)
    place_z = p_lift_higher[2]
    p_place_target = np.array([place_x, place_y, place_z])

    # meshcat.SetObject("traj/approach", Box(0.015, 0.015, 0.015), Rgba(0, 1, 0, 0.5))
    # meshcat.SetTransform("traj/approach", RigidTransform(p_approach))

    # meshcat.SetObject("traj/grasp", Box(0.015, 0.015, 0.015), Rgba(1, 0, 0, 0.8))
    # meshcat.SetTransform("traj/grasp", RigidTransform(p_grasp))

    # meshcat.SetObject("traj/lift", Box(0.015, 0.015, 0.015), Rgba(0, 0, 1, 0.5))
    # meshcat.SetTransform("traj/lift", RigidTransform(p_lift))

    # meshcat.SetObject("traj/lift_higher", Box(0.015, 0.015, 0.015), Rgba(1, 1, 0, 0.5))
    # meshcat.SetTransform("traj/lift_higher", RigidTransform(p_lift_higher))

    # meshcat.SetObject("traj/place_target", Box(0.03, 0.03, 0.03), Rgba(1, 0, 1, 0.8))
    # meshcat.SetTransform("traj/place_target", RigidTransform(R_WG_down, p_place_target))

    base_positions_lock = get_locked_joint_positions(plant, plant_context, iiwa_model)
    q_start = plant.GetPositions(plant_context, iiwa_model)

    # For antipodal grasps, use more relaxed constraints (no strict orientation)
    # For downward grasps, also relax for better reachability but keep stricter 
    if X_WG_grasp is not None:
        # for antipodal only enforce grasp orientation tightly, relax everything else
        approach_theta = 1.5  # very relaxed for approach
        grasp_theta = 0.5     # moderate for actual grasp
        lift_theta = 1.5      # very relaxed for lift/place
        approach_pos_tol = 0.01  # tighter position tolerance for centering
        grasp_pos_tol = 0.008     # very tight for actual grasp
        lift_pos_tol = 0.03      # moderate for lift/place
    else:
        # Downward: relaxed constraints for better reachability
        approach_theta = 0.5
        grasp_theta = 0.5
        lift_theta = 1.5
        approach_pos_tol = 0.02
        grasp_pos_tol = 0.01
        lift_pos_tol = 0.03

    q_approach = solve_ik(
        plant,
        plant_context,
        iiwa_model,
        wsg_model,
        p_approach,
        R_WG, # use same orientation as grasp
        position_tolerance=approach_pos_tol,
        lock_base=True,
        theta_bound=approach_theta,
        base_positions_to_lock=base_positions_lock,
    )

    q_grasp = solve_ik(
        plant,
        plant_context,
        iiwa_model,
        wsg_model,
        p_grasp,
        R_WG,
        position_tolerance=grasp_pos_tol,  # Very tight for actual grasp
        lock_base=True,
        theta_bound=grasp_theta,
        base_positions_to_lock=base_positions_lock,
    )

    q_lift = solve_ik(
        plant,
        plant_context,
        iiwa_model,
        wsg_model,
        p_lift,
        R_WG,
        position_tolerance=lift_pos_tol,
        lock_base=True,
        theta_bound=lift_theta,
        base_positions_to_lock=base_positions_lock,
    )

    q_lift_higher = solve_ik(
        plant,
        plant_context,
        iiwa_model,
        wsg_model,
        p_lift_higher,
        R_WG,
        position_tolerance=lift_pos_tol,
        lock_base=True,
        theta_bound=lift_theta,
        base_positions_to_lock=base_positions_lock,
    )

    # for placing, always use downward orientation and moderate tolerances
    # relax more since gripper might be in awkward orientation after antipodal grasp to prevent ik failure
    place_pos_tol = 0.05
    place_theta = 1.2

    q_place = solve_ik(
        plant,
        plant_context,
        iiwa_model,
        wsg_model,
        p_place_target,
        R_WG_down,
        position_tolerance=place_pos_tol,
        lock_base=True,
        theta_bound=place_theta,
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
        position_tolerance=place_pos_tol,
        lock_base=True,
        theta_bound=place_theta,
        base_positions_to_lock=base_positions_lock,
    )

    # motion helper with smooth trajectory interpolation
    def move_to_smooth(q_des, gripper_width, duration, num_steps=50, monitor_collision=False, baseline_force=0.0):
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

