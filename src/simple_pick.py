"""
Simplified object picking demo using general object segmentation + ICP pose estimation.
This approach works for any object (mug, gelatin box, etc.) as long as a mesh template is
available.
"""

from pathlib import Path
from setup import SimulationEnvironment
from tasks import PickAndPlaceTask


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

SCENARIO_PATH = Path("/workspaces/robman-final-proj/src/grasp.yaml")

# motion parameters
APPROACH_HEIGHT = 0.15  # Height above object to approach from (meters)
LIFT_HEIGHT = 0.20      # Height to lift after grasp (meters)
GRASP_OFFSET = 0.00     # Offset from top of object (0 = grasp at top)

# gripper settings
WSG_OPEN = 0.107
WSG_CLOSED = 0.015  # Tighter closure to maintain grip during descent

# timing
MOVE_TIME = 2.5         # Time for each motion phase (seconds)
GRASP_TIME = 2.0        # Time to close gripper (seconds)
LIFT_TIME = 4.0         # Time to lift object (slower to be careful)

# segmentation parameters
DBSCAN_EPS = 0.03       # DBSCAN clustering epsilon
DBSCAN_MIN_SAMPLES = 50 # DBSCAN min samples per cluster

# objects to pick (in order)
OBJECTS_TO_PICK = ["mug", "gelatin_box", "tomato_soup"]


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

    print("\n" + "=" * 70)
    print(f"DONE! All {len(OBJECTS_TO_PICK)} objects picked and placed.")
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
