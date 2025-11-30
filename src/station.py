from pydrake.all import Diagram
class SimpleStation(Diagram):
    """
    A minimal station-like wrapper that provides the interface AddPointClouds expects.
    This is needed because we can't use MakeHardwareStation with a free body.
    """
    def __init__(self, plant, scene_graph, builder):
        Diagram.__init__(self)
        self._plant = plant
        self._scene_graph = scene_graph
        
    def GetSubsystemByName(self, name):
        if name == "plant":
            return self._plant
        elif name == "scene_graph":
            return self._scene_graph
        else:
            raise ValueError(f"Unknown subsystem: {name}")