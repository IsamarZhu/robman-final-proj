"""
Simplified object picking demo using general object segmentation + ICP pose estimation.
This approach works for any object (mug, gelatin box, etc.), not just mugs.
"""

from pathlib import Path

from pydrake.all import (
    DiagramBuilder,
    Simulator,
    StartMeshcat,
)

from pydrake.geometry import (
    MakeRenderEngineVtk,
    RenderEngineVtkParams,
)

from pydrake.systems.primitives import ConstantVectorSource

from manipulation.station import (
    LoadScenario,
    MakeHardwareStation,
    AddPointClouds,
)

# Import from refactored modules
from perception.object_detection import detect_and_locate_object
from grasping.command_systems import JointPositionCommandSource, WsgCommandSource
from grasping.motion_primitives import pick_object


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

SCENARIO_PATH = Path("/workspaces/robman-final-proj/src/grasp.yaml")

# Motion parameters
APPROACH_HEIGHT = 0.15  # Height above object to approach from (meters)
LIFT_HEIGHT = 0.20      # Height to lift after grasp (meters)
GRASP_OFFSET = 0.00     # Offset from top of object (0 = grasp at top)

# Gripper settings
WSG_OPEN = 0.107
WSG_CLOSED = 0.015  # Tighter closure to maintain grip during descent

# Timing
MOVE_TIME = 2.5         # Time for each motion phase (seconds)
GRASP_TIME = 2.0        # Time to close gripper (seconds)
LIFT_TIME = 4.0         # Time to lift object (slower to be careful)

# Segmentation parameters
DBSCAN_EPS = 0.03       # DBSCAN clustering epsilon
DBSCAN_MIN_SAMPLES = 50 # DBSCAN min samples per cluster

# Target object
TARGET_OBJECT = "mug"   # Options: "mug", "gelatin_box"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    print("=" * 70)
    print("SIMPLIFIED OBJECT PICKING - Using Segmentation + ICP")
    print("=" * 70)

    meshcat = StartMeshcat()
    print(f"\nMeshcat: {meshcat.web_url()}")

    # Load scenario
    print("\nLoading scenario...")
    with open(SCENARIO_PATH, "r") as f:
        scenario_yaml = f.read()
    scenario = LoadScenario(data=scenario_yaml)

    # Build hardware station
    print("Building hardware station...")
    builder = DiagramBuilder()
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat=meshcat))
    plant = station.GetSubsystemByName("plant")
    scene_graph = station.GetSubsystemByName("scene_graph")

    # Add renderer
    renderer_name = "renderer"
    if not scene_graph.HasRenderer(renderer_name):
        scene_graph.AddRenderer(
            renderer_name, MakeRenderEngineVtk(RenderEngineVtkParams())
        )

    # Get robot models
    iiwa_model = plant.GetModelInstanceByName("iiwa_arm")
    wsg_model = plant.GetModelInstanceByName("wsg_arm")

    # Create command sources
    default_context = plant.CreateDefaultContext()
    q_start = plant.GetPositions(default_context, iiwa_model)

    cmd_source = builder.AddSystem(JointPositionCommandSource(q_start))
    builder.Connect(
        cmd_source.get_output_port(),
        station.GetInputPort("iiwa_arm.desired_state"),
    )

    wsg_cmd_source = builder.AddSystem(WsgCommandSource(WSG_OPEN))
    builder.Connect(
        wsg_cmd_source.get_output_port(),
        station.GetInputPort("wsg_arm.position"),
    )

    wsg_force_limit = builder.AddSystem(ConstantVectorSource([40.0]))
    builder.Connect(
        wsg_force_limit.get_output_port(),
        station.GetInputPort("wsg_arm.force_limit"),
    )

    # Add point cloud cameras
    print("Setting up cameras...")
    to_point_cloud = AddPointClouds(
        scenario=scenario,
        station=station,
        builder=builder,
        meshcat=None,  # Don't visualize raw clouds
    )

    for name, system in to_point_cloud.items():
        builder.ExportOutput(
            system.GetOutputPort("point_cloud"),
            f"{name}.point_cloud",
        )

    # Build and initialize simulator
    print("Building diagram...")
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    # Let scene settle
    print("\nLetting scene settle...")
    simulator.Initialize()
    simulator.AdvanceTo(1.0)
    diagram.ForcedPublish(context)
    print("✓ Scene ready")

    # Detect object and compute grasp
    X_WO, grasp_center_xyz, object_top_z = detect_and_locate_object(
        diagram,
        context,
        meshcat,
        target_object=TARGET_OBJECT,
        dbscan_eps=DBSCAN_EPS,
        dbscan_min_samples=DBSCAN_MIN_SAMPLES,
        grasp_offset=GRASP_OFFSET,
    )

    # Start recording
    meshcat.StartRecording()

    # Execute pick
    pick_object(
        meshcat,
        simulator,
        diagram,
        plant,
        plant_context,
        iiwa_model,
        wsg_model,
        cmd_source,
        wsg_cmd_source,
        grasp_center_xyz,
        object_top_z,
        approach_height=APPROACH_HEIGHT,
        lift_height=LIFT_HEIGHT,
        wsg_open=WSG_OPEN,
        wsg_closed=WSG_CLOSED,
        move_time=MOVE_TIME,
        grasp_time=GRASP_TIME,
        lift_time=LIFT_TIME,
    )

    # Finish recording
    meshcat.StopRecording()
    meshcat.PublishRecording()

    print("\n" + "=" * 70)
    print("DONE! Check Meshcat for visualization.")
    print("=" * 70)
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
