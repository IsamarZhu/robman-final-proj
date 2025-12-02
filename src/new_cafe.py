from pathlib import Path
from time import sleep

from pydrake.all import (
    StartMeshcat,
    Simulator,
    DiagramBuilder,
)
from manipulation.station import LoadScenario, MakeHardwareStation


def simulate_scenario():
    meshcat = StartMeshcat()
    scenario_file = Path("/workspaces/robman-final-proj/src/new_scenario.yaml")
    scenario = LoadScenario(filename=str(scenario_file))
    builder = DiagramBuilder()    
    station = builder.AddSystem(MakeHardwareStation(
        scenario=scenario,
        meshcat=meshcat,
    ))
    
    diagram = builder.Build()
    
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    context = simulator.get_mutable_context()
    
    diagram.ForcedPublish(context)
    meshcat.StartRecording()
    simulator.AdvanceTo(10.0)
    meshcat.StopRecording()
    meshcat.PublishRecording()
    
    while True:
        sleep(1)


if __name__ == "__main__":
    simulate_scenario()