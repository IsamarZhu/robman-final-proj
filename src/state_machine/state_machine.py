import numpy as np
from pydrake.all import RigidTransform, RotationMatrix, PiecewisePolynomial

from state_machine.states import CafeState
from perception.object_detection import detect_and_locate_object
from grasping.inverse_kinematics import solve_ik, get_locked_joint_positions


class CafeStateMachine:
    def __init__(
        self,
        env,
        approach_height=0.15,
        lift_height=0.20,
        grasp_offset=0.00,
        wsg_open=0.107,
        wsg_closed=0.015,
        move_time=2.5,
        grasp_time=2.0,
        lift_time=4.0,
        dbscan_eps=0.03,
        dbscan_min_samples=50,
    ):
        self.env = env
        self.current_state = CafeState.PERCEPTION
        self.object_queue = ["mug", "gelatin_box", "tomato_soup"]
        self.current_object_index = 0

        self.approach_height = approach_height
        self.lift_height = lift_height
        self.grasp_offset = grasp_offset
        self.wsg_open = wsg_open
        self.wsg_closed = wsg_closed
        self.move_time = move_time
        self.grasp_time = grasp_time
        self.lift_time = lift_time
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples

        self.grasp_center_xyz = None
        self.object_top_z = None
        self.baseline_force = 0.0

        self.q_approach = None
        self.q_grasp = None
        self.q_lift = None
        self.q_lift_higher = None
        self.q_place = None

    def run(self):
        while self.current_object_index < len(self.object_queue):
            if self.current_state == CafeState.PERCEPTION:
                self.perception_state()
            elif self.current_state == CafeState.PICK:
                self.pick_state()
            elif self.current_state == CafeState.PLACE:
                self.place_state()
            elif self.current_state == CafeState.EXECUTE_MOTION:
                self.execute_motion_state()

    def perception_state(self):
        if self.current_object_index > 0:
            self.env.settle_scene(duration=2.0)

        target_object = self.object_queue[self.current_object_index]
        print(f"\n[PERCEPTION]")

        X_WO, self.grasp_center_xyz, self.object_top_z = detect_and_locate_object(
            self.env.diagram,
            self.env.context,
            self.env.meshcat,
            target_object=target_object,
            dbscan_eps=self.dbscan_eps,
            dbscan_min_samples=self.dbscan_min_samples,
            grasp_offset=self.grasp_offset,
        )

        self.current_state = CafeState.PICK

    def compute_trajectories(self):
        plant = self.env.plant
        plant_context = self.env.plant_context
        iiwa_model = self.env.iiwa_model
        wsg_model = self.env.wsg_model

        R_WG_down = RotationMatrix.MakeXRotation(-np.pi / 2.0)

        x, y, z_grasp = self.grasp_center_xyz
        z_grasp += 0.05

        p_approach = np.array([x, y, z_grasp + self.approach_height])
        p_grasp = np.array([x, y, z_grasp])
        p_lift = np.array([x, y, z_grasp + self.lift_height])
        p_lift_higher = np.array([x, y, z_grasp + self.lift_height + 0.1])

        q_current = plant.GetPositions(plant_context, iiwa_model)
        current_base_x = q_current[0]
        current_base_y = q_current[1]

        place_offset_x = -0.40
        place_offset_y = 0.3

        place_x = current_base_x + place_offset_x
        place_y = current_base_y + place_offset_y
        place_z = p_lift_higher[2]
        p_place_target = np.array([place_x, place_y, place_z])

        base_positions_lock = get_locked_joint_positions(plant, plant_context, iiwa_model)

        self.q_approach = solve_ik(
            plant, plant_context, iiwa_model, wsg_model,
            p_approach, R_WG_down,
            position_tolerance=0.02, lock_base=True, theta_bound=0.3,
            base_positions_to_lock=base_positions_lock,
        )

        self.q_grasp = solve_ik(
            plant, plant_context, iiwa_model, wsg_model,
            p_grasp, R_WG_down,
            position_tolerance=0.02, lock_base=True, theta_bound=0.3,
            base_positions_to_lock=base_positions_lock,
        )

        self.q_lift = solve_ik(
            plant, plant_context, iiwa_model, wsg_model,
            p_lift, R_WG_down,
            position_tolerance=0.02, lock_base=True, theta_bound=0.3,
            base_positions_to_lock=base_positions_lock,
        )

        self.q_lift_higher = solve_ik(
            plant, plant_context, iiwa_model, wsg_model,
            p_lift_higher, R_WG_down,
            position_tolerance=0.02, lock_base=True, theta_bound=0.8,
            base_positions_to_lock=base_positions_lock,
        )

        self.q_place = solve_ik(
            plant, plant_context, iiwa_model, wsg_model,
            p_place_target, R_WG_down,
            position_tolerance=0.08, lock_base=True, theta_bound=0.8,
            base_positions_to_lock=base_positions_lock,
        )

    def pick_state(self):
        print("\n[PICK]")

        plant = self.env.plant
        plant_context = self.env.plant_context
        iiwa_model = self.env.iiwa_model
        cmd_source = self.env.cmd_source
        wsg_cmd_source = self.env.wsg_cmd_source
        diagram = self.env.diagram
        context = self.env.context

        self.compute_trajectories()

        q_start = plant.GetPositions(plant_context, iiwa_model)
        plant.SetPositions(plant_context, iiwa_model, q_start)
        plant.SetVelocities(plant_context, iiwa_model, np.zeros_like(q_start))
        cmd_source.set_q_desired(q_start)
        wsg_cmd_source.set_width(self.wsg_open)
        diagram.ForcedPublish(context)

        print("1. move to start position")
        self.move_to_smooth(q_start, self.wsg_open, self.move_time)

        print("2. approach above object")
        self.move_to_smooth(self.q_approach, self.wsg_open, self.move_time)

        print("3. descend to grasp position")
        self.move_to_smooth(self.q_grasp, self.wsg_open, self.move_time)

        print("4. close gripper to grasp")
        self.move_to_smooth(self.q_grasp, self.wsg_closed, self.grasp_time)

        print("5. lift object")
        self.move_to_smooth(self.q_lift, self.wsg_closed, self.lift_time)

        wsg_model = self.env.wsg_model
        tau_contact_baseline = plant.get_generalized_contact_forces_output_port(wsg_model).Eval(plant_context)
        self.baseline_force = np.sum(np.abs(tau_contact_baseline))

        print("6. lift higher")
        self.move_to_smooth(self.q_lift_higher, self.wsg_closed, self.lift_time)

        print("7. move to place target")
        self.move_to_smooth(self.q_place, self.wsg_closed, self.move_time)

        print("8. hold at place target")
        self.move_to_smooth(self.q_place, self.wsg_closed, 2.0)

        self.current_state = CafeState.PLACE

    def place_state(self):
        print("\n[PLACE]")

        plant = self.env.plant
        plant_context = self.env.plant_context
        iiwa_model = self.env.iiwa_model
        wsg_model = self.env.wsg_model

        R_WG_down = RotationMatrix.MakeXRotation(-np.pi / 2.0)

        q_current = plant.GetPositions(plant_context, iiwa_model)
        current_base_x = q_current[0]
        current_base_y = q_current[1]

        place_offset_x = -0.40
        place_offset_y = 0.3

        x, y, z_grasp = self.grasp_center_xyz
        z_grasp += 0.05
        place_z = z_grasp + self.lift_height + 0.1

        place_x = current_base_x + place_offset_x
        place_y = current_base_y + place_offset_y
        p_place_target = np.array([place_x, place_y, place_z])

        p_descent_target = np.array([p_place_target[0], p_place_target[1], p_place_target[2] - 0.40])

        base_positions_lock = get_locked_joint_positions(plant, plant_context, iiwa_model)
        q_descent = solve_ik(
            plant, plant_context, iiwa_model, wsg_model,
            p_descent_target, R_WG_down,
            position_tolerance=0.08, lock_base=True, theta_bound=0.8,
            base_positions_to_lock=base_positions_lock,
        )

        collision = self.move_to_smooth(
            q_descent, self.wsg_closed, 5.0,
            monitor_collision=True, baseline_force=self.baseline_force
        )

        if collision:
            print("9. collision detected, opening gripper")
            wsg_cmd_source = self.env.wsg_cmd_source
            simulator = self.env.simulator
            wsg_cmd_source.set_width(self.wsg_open)
            t = simulator.get_context().get_time()
            simulator.AdvanceTo(t + 2.0)
            self.env.diagram.ForcedPublish(self.env.context)

        self.current_state = CafeState.MOVE

    def move_state(self):
        print("\n[MOVE]")
        self.env.settle_scene(duration=2.0)

        self.current_object_index += 1
        if self.current_object_index < len(self.object_queue):
            self.current_state = CafeState.PERCEPTION
        else:
            print("\n[COMPLETE]")

    def move_to_smooth(self, q_des, gripper_width, duration, num_steps=50, monitor_collision=False, baseline_force=0.0):
        plant = self.env.plant
        plant_context = self.env.plant_context
        iiwa_model = self.env.iiwa_model
        wsg_model = self.env.wsg_model
        cmd_source = self.env.cmd_source
        wsg_cmd_source = self.env.wsg_cmd_source
        simulator = self.env.simulator
        diagram = self.env.diagram
        context = self.env.context

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

            if monitor_collision and total_force > baseline_force + 5.0:
                collision_detected = True
                break

            simulator.AdvanceTo(t_abs)
            diagram.ForcedPublish(context)

        return collision_detected
