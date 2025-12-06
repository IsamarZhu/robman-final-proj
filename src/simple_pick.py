from pathlib import Path
from setup import SimulationEnvironment
from tasks import PickAndPlaceTask


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

SCENARIO_PATH = Path("/workspaces/robman-final-proj/src/environment/scenario_two.yaml")

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

# objects to pick (in order)
# OBJECTS_TO_PICK = ["mug", "gelatin_box", "tomato_soup"]
OBJECTS_TO_PICK = ["potted_meat", "apple"]


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    env = SimulationEnvironment(SCENARIO_PATH, wsg_open=WSG_OPEN)
    env.build()

    # create pick-and-place task
    task = PickAndPlaceTask(
        env,
        approach_height=APPROACH_HEIGHT,
        lift_height=LIFT_HEIGHT,
        grasp_offset=GRASP_OFFSET,
        wsg_open=WSG_OPEN,
        wsg_closed=WSG_CLOSED,
        move_time=MOVE_TIME,
        grasp_time=GRASP_TIME,
        lift_time=LIFT_TIME,
        dbscan_eps=DBSCAN_EPS,
        dbscan_min_samples=DBSCAN_MIN_SAMPLES,
    )

    env.meshcat.StartRecording()
    
    # executing pick-and-place for each object
    for i, obj_name in enumerate(OBJECTS_TO_PICK):
        task.execute(obj_name)
        if i < len(OBJECTS_TO_PICK) - 1:
            env.settle_scene(duration=2.0)

    env.meshcat.StopRecording()
    env.meshcat.PublishRecording()


if __name__ == "__main__":
    main()
