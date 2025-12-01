import matplotlib.pyplot as plt
import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    BasicVector,
    Context,
    Diagram,
    DiagramBuilder,
    LeafSystem,
    MeshcatVisualizer,
    ModelInstanceIndex,
    MultibodyPlant,
    Parser,
    Simulator,
    StartMeshcat,
    VectorLogSink,
)

# from manipulation import running_as_notebook

class PIDController(LeafSystem):
    """PID controller for the IIWA robot"""

    def __init__(self, kp: float, kd: float, ki: float, q_desired: np.ndarray) -> None:
        LeafSystem.__init__(self)

        self.input_port = self.DeclareVectorInputPort("iiwa_state", 14)
        self.output_port = self.DeclareVectorOutputPort("iiwa_torque", 7, self.ComputeTorque)

        self.kp = kp
        self.kd = kd
        self.ki = ki  # New: integral gain
        self.q_desired = q_desired
        self.qdot_desired = np.zeros(7)
        self.integral_error = np.zeros(7)

        self.prev_time = 0.0

    def ComputeTorque(self, context: Context, output: BasicVector) -> None:
        # TODO: Extract state information (same as PD controller)
        iiwa_state = self.input_port.Eval(context)
        q = iiwa_state[:7]
        qdot = iiwa_state[7:]

        current_time = context.get_time()
        dt = current_time - self.prev_time

        # TODO: Compute position and velocity errors (same as PD controller)
        position_error = self.q_desired - q
        velocity_error = self.qdot_desired - qdot

        # TODO: Update integral error
        if dt > 0:  # Avoid division by zero on first call
            # YOUR CODE HERE - update self.integral_error
            self.integral_error += position_error * dt

        # TODO: Compute PID control law
        # HINT: Combine all three terms: proportional + derivative + integral
        torque = self.kp * position_error + self.kd * velocity_error + self.ki * self.integral_error

        # Update previous time for next iteration
        self.prev_time = current_time

        output.set_value(torque)