import numpy as np
from pydrake.all import (
    BasicVector,
    Context,
    LeafSystem,
)


class PIDController(LeafSystem):
    """PID controller for the IIWA robot"""

    def __init__(self, kp: float, kd: float, ki: float, q_desired: np.ndarray) -> None:
        LeafSystem.__init__(self)

        self.input_port = self.DeclareVectorInputPort("iiwa_state", 14)
        self.output_port = self.DeclareVectorOutputPort("iiwa_torque", 7, self.ComputeTorque)

        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.q_desired = q_desired
        self.qdot_desired = np.zeros(7)
        self.integral_error = np.zeros(7)

        self.prev_time = 0.0

    def ComputeTorque(self, context: Context, output: BasicVector) -> None:
        iiwa_state = self.input_port.Eval(context)
        q = iiwa_state[:7]
        qdot = iiwa_state[7:]

        current_time = context.get_time()
        dt = current_time - self.prev_time

        # Compute position and velocity errors
        position_error = self.q_desired - q
        velocity_error = self.qdot_desired - qdot

        # Update integral error
        if dt > 0:  # Avoid division by zero on first call
            self.integral_error += position_error * dt

        # Compute PID control law
        torque = self.kp * position_error + self.kd * velocity_error + self.ki * self.integral_error

        # Update previous time for next iteration
        self.prev_time = current_time

        output.set_value(torque)

    def set_desired_position(self, q_desired: np.ndarray) -> None:
        """Update the desired position during runtime"""
        self.q_desired = q_desired
        # reset integral error when changing targets to avoid windup
        self.integral_error = np.zeros(7)