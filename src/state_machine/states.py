from enum import Enum


class CafeState(Enum):
    PERCEPTION = 1
    PICK = 2
    PLACE = 3
    MOVE = 4
