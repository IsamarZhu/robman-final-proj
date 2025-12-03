from pathlib import Path
from time import sleep

from pydrake.all import (
    StartMeshcat,
    Simulator,
    DiagramBuilder,
    ConstantVectorSource,
)
from manipulation.station import LoadScenario, MakeHardwareStation
from temp_perception import perceive_tables

def simulate_scenario():
    meshcat = StartMeshcat()
    scenario_file = Path("/workspaces/robman-final-proj/src/new_scenario.yaml")
    scenario = LoadScenario(filename=str(scenario_file))
    builder = DiagramBuilder()
    station = builder.AddSystem(MakeHardwareStation(
        scenario=scenario,
        meshcat=meshcat,
    ))
    
    # Mobile base positions (x, y, z) + 7 arm joints = 10 total
    mobile_base_positions = [1.4, 0.0, 0.1]  # iiwa_base_x, y, z
    arm_positions = [1.94, 0.1, 0.0, -0.9, 0.6, 1.7, 0.0]  # 7 arm joints
    
    all_positions = mobile_base_positions + arm_positions
    
    # desired_state needs positions AND velocities (20 values total)
    # Format: [q1, ..., q10, v1, ..., v10]
    desired_state = all_positions + [0.0] * 10  # 10 positions + 10 zero velocities
    
    state_source = builder.AddSystem(ConstantVectorSource(desired_state))
    
    builder.Connect(
        state_source.get_output_port(),
        station.GetInputPort("iiwa_arm.desired_state")
    )
    
    # Also need to control the gripper
    gripper_source = builder.AddSystem(ConstantVectorSource([0.1]))  # Gripper open
    builder.Connect(
        gripper_source.get_output_port(),
        station.GetInputPort("wsg_arm.position")
    )
    
    diagram = builder.Build()
    
    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    context = simulator.get_mutable_context()
    station_context = station.GetMyContextFromRoot(context)
    
    tables = perceive_tables(station, station_context)

    diagram.ForcedPublish(context)
    meshcat.StartRecording()
    simulator.AdvanceTo(10.0)
    meshcat.StopRecording()
    meshcat.PublishRecording()
    
    while True:
        sleep(1)


if __name__ == "__main__":
    simulate_scenario()