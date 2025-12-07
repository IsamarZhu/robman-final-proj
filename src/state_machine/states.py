from enum import Enum


class CafeState(Enum):
    IDLE = 0

    PERCEPTION = 1
    PICK = 2
    PLACE = 3
    MOVE = 4

    # navigation to next table
    NAVIGATE_ROTATE_TO_WAYPOINT = 10
    NAVIGATE_MOVE_TO_WAYPOINT = 11
    NAVIGATE_ROTATE_TO_TABLE = 12
    NAVIGATE_SLIDE_LEFT = 13
    NAVIGATE_SLIDE_RIGHT = 14