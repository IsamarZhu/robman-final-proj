from state_machine.states import CafeState
from perception.object_detection import detect_and_locate_object
from grasping.motion_primitives import pick_object

# --------------------------------------------------------------------------- #
# Configuration (maybe incorporate back later?)
# --------------------------------------------------------------------------- #

# motion parameters
APPROACH_HEIGHT = 0.15
LIFT_HEIGHT = 0.2
GRASP_OFFSET = 0.00

# gripper settings
WSG_OPEN = 0.107
WSG_CLOSED = 0.015

# timing
MOVE_TIME = 2.5
GRASP_TIME = 2.0
LIFT_TIME = 4.0 

# segmentation parameters
DBSCAN_EPS = 0.03
DBSCAN_MIN_SAMPLES = 50


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
        self.scenario_number = env.scenario_number
        if self.scenario_number == "one":
            self.object_queue = ["mug", "gelatin_box", "tomato_soup"]
        elif self.scenario_number == "two":
            self.object_queue = ["potted_meat", "apple", "master_chef"]
        elif self.scenario_number == "three":
            self.object_queue = ["pudding", "tuna"]
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

    def run(self):
        while self.current_object_index < len(self.object_queue):
            if self.current_state == CafeState.PERCEPTION:
                self.perception_state()
            elif self.current_state == CafeState.PICK:
                self.pick_state()
            elif self.current_state == CafeState.MOVE:
                self.move_state()

    def perception_state(self):
        if self.current_object_index > 0:
            self.env.settle_scene(duration=2.0)

        target_object = self.object_queue[self.current_object_index]
        print(f"\n[PERCEPTION]")

        X_WO, self.grasp_center_xyz, self.object_top_z = detect_and_locate_object(
            self.scenario_number,
            self.env.diagram,
            self.env.context,
            self.env.meshcat,
            target_object=target_object,
            dbscan_eps=self.dbscan_eps,
            dbscan_min_samples=self.dbscan_min_samples,
            grasp_offset=self.grasp_offset,
        )

        self.current_state = CafeState.PICK

    def pick_state(self):
        print("\n[PICK]")

        pick_object(
            self.env.meshcat,
            self.env.simulator,
            self.env.diagram,
            self.env.plant,
            self.env.plant_context,
            self.env.iiwa_model,
            self.env.wsg_model,
            self.env.cmd_source,
            self.env.wsg_cmd_source,
            self.grasp_center_xyz,
            self.object_top_z,
            approach_height=self.approach_height,
            lift_height=self.lift_height,
            wsg_open=self.wsg_open,
            wsg_closed=self.wsg_closed,
            move_time=self.move_time,
            grasp_time=self.grasp_time,
            lift_time=self.lift_time,
        )

        self.current_state = CafeState.MOVE

    def move_state(self):
        print("\n[MOVE]")
        self.env.settle_scene(duration=2.0)

        self.current_object_index += 1
        if self.current_object_index < len(self.object_queue):
            self.current_state = CafeState.PERCEPTION
        else:
            print("\n[COMPLETE]")
