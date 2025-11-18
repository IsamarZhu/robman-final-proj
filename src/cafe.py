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
floor_sdf = """<?xml version="1.0"?>
<sdf version="1.7">
  <model name="table">
    <pose>0 0 0 0 0 0</pose>
    <link name="table_link">
      <inertial>
        <mass>20</mass>
        <inertia>
          <ixx>1.0</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>1.0</iyy>
          <iyz>0.0</iyz>
          <izz>1.0</izz>
        </inertia>
      </inertial>
      <collision name="box_collision">
        <geometry>
          <box>
            <size>2 2 0.1</size>
          </box>
        </geometry>
      </collision>
      <visual name="box_visual">
        <geometry>
          <box>
            <size>10 10 0.1</size>
          </box>
        </geometry>
      </visual>
    </link>
  </model>
</sdf>
"""

robot_base_sdf = """<?xml version="1.0"?>
<sdf version="1.7">
  <model name="table">
    <pose>0 0 0 0 0 0</pose>
    <link name="table_link">
      <inertial>
        <mass>20</mass>
        <inertia>
          <ixx>1.0</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>1.0</iyy>
          <iyz>0.0</iyz>
          <izz>1.0</izz>
        </inertia>
      </inertial>
      <collision name="box_collision">
        <geometry>
          <box>
            <size>0.5 1 0.5</size>
          </box>
        </geometry>
      </collision>
      <visual name="box_visual">
        <geometry>
          <box>
            <size>0.5 1 0.5</size>
          </box>
        </geometry>
        <material>
            <ambient>0 0 1 1</ambient>
            <diffuse>0 0 1 1</diffuse>
            <specular>0 0 0.3 1</specular>
            <emissive>0 0 0 1</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""

tray_sdf = """<?xml version="1.0"?>
<sdf version="1.7">
  <model name="table">
    <pose>0 0 0 0 0 0</pose>
    <link name="table_link">
      <inertial>
        <mass>20</mass>
        <inertia>
          <ixx>1.0</ixx>
          <ixy>0.0</ixy>
          <ixz>0.0</ixz>
          <iyy>1.0</iyy>
          <iyz>0.0</iyz>
          <izz>1.0</izz>
        </inertia>
      </inertial>
      <collision name="box_collision">
        <geometry>
          <box>
            <size>0.2 0.2 0.01</size>
          </box>
        </geometry>
      </collision>
      <visual name="box_visual">
        <geometry>
          <box>
            <size>0.2 0.2 0.01</size>
          </box>
        </geometry>
        <material>
            <ambient>0 0 1 1</ambient>
            <diffuse>0 0 1 1</diffuse>
            <specular>0 0 0.3 1</specular>
            <emissive>0 0 0 1</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""

tmp_dir = tempfile.gettempdir()
floor_path = Path(tmp_dir) / "floor.sdf"
floor_path.write_text(floor_sdf)

robot_base_path = Path(tmp_dir) / "robot_base.sdf"
robot_base_path.write_text(robot_base_sdf)

tray_path = Path(tmp_dir) / "tray.sdf"
tray_path.write_text(tray_sdf)

scenario_yaml = f"""directives:
- add_model:
    name: floor
    file: file://{floor_path}
- add_weld:
    parent: world
    child: floor::table_link
    X_PC:
        translation: [0, 0, 0.0]
        rotation: !Rpy {{ deg: [0, 0, 0]}}
        
- add_model:
    name: robot_base
    file: file://{robot_base_path}
- add_weld:
    parent: floor::table_link
    child: robot_base::table_link
    X_PC:
        translation: [1.4, 0, 0.4]
        rotation: !Rpy {{ deg: [0, 0, 0]}}
    
- add_model:
    name: table
    file: package://drake_models/manipulation_station/table_wide.sdf
- add_weld:
    parent: world
    child: table::table_body
    X_PC:
        translation: [0, 0, 1.0]
        rotation: !Rpy {{ deg: [0, 0, 0] }}
        
- add_model:
    name: mug
    file: file:///workspaces/robman-final-proj/assets/mug/mug.sdf
    default_free_body_pose:
      base_link:
        translation: [1.4, 0.35, 1.35]
        rotation: !Rpy {{ deg: [0, 0, 0] }}


- add_model:
    name: iiwa_arm
    file: package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf
    default_joint_positions:
        iiwa_joint_1: [-1.57]
        iiwa_joint_2: [0.1]
        iiwa_joint_3: [0]
        iiwa_joint_4: [-0.9]
        iiwa_joint_5: [0]
        iiwa_joint_6: [ 1.6]
        iiwa_joint_7: [0]
- add_weld:
    parent: robot_base::table_link
    child: iiwa_arm::iiwa_link_0
    X_PC:
        translation: [0, -0.3, 0.25]
        rotation: !Rpy {{ deg: [0, 0, 180]}}
        
- add_model:
    name: wsg_arm
    file: package://manipulation/hydro/schunk_wsg_50_with_tip.sdf
- add_weld:
    parent: iiwa_arm::iiwa_link_7
    child: wsg_arm::body
    X_PC:
        translation: [0, 0, 0.09]
        rotation: !Rpy {{ deg: [90, 0, 0]}}


- add_model:
    name: iiwa_plate
    file: package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf
    default_joint_positions:
        iiwa_joint_1: [-1.57]
        iiwa_joint_2: [-1.87]
        iiwa_joint_3: [0]
        iiwa_joint_4: [-1.8]
        iiwa_joint_5: [0]
        iiwa_joint_6: [1.6]
        iiwa_joint_7: [1.3]
- add_weld:
    parent: robot_base::table_link
    child: iiwa_plate::iiwa_link_0
    X_PC:
        translation: [0, 0.3, 0.25]
        rotation: !Rpy {{ deg: [0, 0, 0]}}
- add_model:
    name: wsg_plate
    file: package://manipulation/hydro/schunk_wsg_50_with_tip.sdf
- add_weld:
    parent: iiwa_plate::iiwa_link_7
    child: wsg_plate::body
    X_PC:
        translation: [0, 0, 0.09]
        rotation: !Rpy {{ deg: [90, 0, 105]}}
- add_model:
    name: tray
    file: file://{tray_path}
- add_weld:
    parent: wsg_plate::body
    child: tray::table_link
    X_PC:
        translation: [0, 0.15, 0.0]
        rotation: !Rpy {{ deg: [0, 90, 0]}}

model_drivers:
    iiwa_arm: !IiwaDriver
      control_mode: position_only
      hand_model_name: wsg_arm
    wsg_arm: !SchunkWsgDriver {{}}
    iiwa_plate: !IiwaDriver
      control_mode: position_only
      hand_model_name: wsg_plate
    wsg_plate: !SchunkWsgDriver {{}}
"""

meshcat = StartMeshcat()

scenario = LoadScenario(data=scenario_yaml)
builder = DiagramBuilder()
station = MakeHardwareStation(scenario, meshcat=meshcat)
builder.AddSystem(station)
plant = station.GetSubsystemByName("plant")

from pydrake.systems.primitives import ConstantVectorSource
initial_positions_arm = [
    -1.57,  # joint 1
    0.1,    # joint 2
    0,      # joint 3
    -0.9,   # joint 4
    0,      # joint 5
    1.6,    # joint 6
    0       # joint 7
]
initial_positions_plate = [
    -1.57,  # joint 1
    -1.87,    # joint 2
    0,      # joint 3
    -1.8,   # joint 4
    0,      # joint 5
    1.6,    # joint 6
    1.3       # joint 7
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