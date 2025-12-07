from pydrake.all import (
    DiagramBuilder,
    InverseKinematics,
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
import numpy as np
class JointPositionCommandSource(LeafSystem):
    def __init__(self, q_initial: np.ndarray):
        super().__init__()
        q_initial = np.copy(q_initial).reshape(-1)
        self._nq = q_initial.shape[0]
        self._q_des = q_initial
        self._v_des = np.zeros(self._nq)

        self.DeclareVectorOutputPort(
            "iiwa_desired_state",
            BasicVector(2 * self._nq),
            self._DoCalcOutput,
        )

    def _DoCalcOutput(self, context, output: BasicVector):
        # v_des = np.zeros(self._nq)
        desired_state = np.concatenate([self._q_des, self._v_des])
        output.SetFromVector(desired_state)

    def set_q_desired(self, q_des: np.ndarray, v_des: np.ndarray = None):
        q_des = np.copy(q_des).reshape(-1)
        assert q_des.shape[0] == self._nq
        self._q_des = q_des

        if v_des is not None:
            v_des = np.copy(v_des).reshape(-1)
            assert v_des.shape[0] == self._nq
            self._v_des = v_des
        else:
            self._v_des = np.zeros(self._nq)
class WsgCommandSource(LeafSystem):
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