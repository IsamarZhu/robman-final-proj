from pathlib import Path
import time
import mpld3
import numpy as np
from pydrake.all import (
    AddFrameTriadIllustration,
    BasicVector,
    Context,
    DiagramBuilder,
    Integrator,
    JacobianWrtVariable,
    LeafSystem,
    MultibodyPlant,
    PiecewisePolynomial,
    PiecewisePose,
    RigidTransform,
    RotationMatrix,
    Simulator,
    StartMeshcat,
    Trajectory,
    TrajectorySource,
)
import tempfile

from manipulation import running_as_notebook
from manipulation.station import LoadScenario, MakeHardwareStation

if running_as_notebook:
    mpld3.enable_notebook()

meshcat = StartMeshcat()
scenario_file = Path("/workspaces/robman-final-proj/src/scenario.yaml")

with open(scenario_file, "r") as f:
    scenario_yaml = f.read()

scenario = LoadScenario(data=scenario_yaml)

builder = DiagramBuilder()
station = MakeHardwareStation(scenario, meshcat=meshcat)
builder.AddSystem(station)
plant = station.GetSubsystemByName("plant")

from pydrake.systems.primitives import ConstantVectorSource
initial_positions_arm = [
    -1.57,  # joint 1
    0.9,    # joint 2
    0,      # joint 3
    -0.9,   # joint 4
    0,      # joint 5
    1.6,    # joint 6
    0       # joint 7
]
initial_positions_plate = [
    -1.57,  # joint 1
    -1.5,    # joint 2
    0,      # joint 3
    0.9,   # joint 4
    0,      # joint 5
    -1.6,    # joint 6
    0.3       # joint 7
]

# currently making both arms be held to the same initial position
position_arm_source = builder.AddSystem(ConstantVectorSource(initial_positions_arm))
builder.Connect(
    position_arm_source.get_output_port(),
    station.GetInputPort("iiwa_arm.position")
)
position_plate_source = builder.AddSystem(ConstantVectorSource(initial_positions_plate))
builder.Connect(
    position_plate_source.get_output_port(),
    station.GetInputPort("iiwa_plate.position")
)

wsg_arm_source = builder.AddSystem(ConstantVectorSource([0.1]))
builder.Connect(
    wsg_arm_source.get_output_port(),
    station.GetInputPort("wsg_arm.position")
)
wsg_plate_source = builder.AddSystem(ConstantVectorSource([0]))
builder.Connect(
    wsg_plate_source.get_output_port(),
    station.GetInputPort("wsg_plate.position")
)

diagram = builder.Build()
simulator = Simulator(diagram)
context = simulator.get_mutable_context()
station_context = station.GetMyContextFromRoot(context)

diagram.ForcedPublish(context)

if running_as_notebook:
    simulator.set_target_realtime_rate(1.0)

meshcat.StartRecording()
simulator.AdvanceTo(500.0)
time.sleep(30.0)
# meshcat.PublishRecording() #turning this on terminates or smthn, idk