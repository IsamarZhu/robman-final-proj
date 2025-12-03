from pathlib import Path
from time import sleep

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    StartMeshcat,
    Simulator,
    DiagramBuilder,
    Rgba,
    ConstantVectorSource,
    PiecewisePolynomial,
    TrajectorySource,
)
from manipulation.station import LoadScenario, MakeHardwareStation, AddPointClouds
from perception.segmentation import build_pointcloud, segment_objects_clustering, ObjectDetector
import numpy as np

def simulate_scenario():
    print("Maggie Yao")
    meshcat = StartMeshcat()
    scenario_file = Path("/workspaces/robman-final-proj/src/maggie.yaml")
    scenario = LoadScenario(filename=str(scenario_file))
    builder = DiagramBuilder()
    station = builder.AddSystem(MakeHardwareStation(
        scenario=scenario,
        meshcat=meshcat,
    ))

    mobile_base_positions = [1.4, 0.0, 0.1]  # iiwa_base_x, y, z
    arm_positions = [1.94, 0.1, 0.0, -0.9, 0.6, 1.7, 0.0]  # 7 arm joints    
    all_positions = mobile_base_positions + arm_positions
    desired_state = all_positions + [0.0] * 10  # 10 positions + 10 zero velocities
    
    state_source = builder.AddSystem(ConstantVectorSource(desired_state))
    
    builder.Connect(
        state_source.get_output_port(),
        station.GetInputPort("iiwa_arm.desired_state")
    )

    gripper_source = builder.AddSystem(ConstantVectorSource([0.1]))  
    builder.Connect(
        gripper_source.get_output_port(),
        station.GetInputPort("wsg_arm.position")
    )
    
    to_point_cloud = AddPointClouds(
        scenario=scenario,
        station=station,
        builder=builder,
        meshcat=meshcat,
    )
    for name, system in to_point_cloud.items():
        builder.ExportOutput(
            system.GetOutputPort("point_cloud"),
            f"{name}.point_cloud",
        )
    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()  
    plant = station.GetSubsystemByName("plant")  
    plant_context = plant.GetMyMutableContextFromRoot(context)

    diagram.ForcedPublish(context)
    meshcat.StartRecording()
    
    
    ########## computing mug ##########
    
    pointcloud = build_pointcloud(diagram, context)
    object_clouds = segment_objects_clustering(pointcloud)
    colors = [Rgba(1, 0, 0), Rgba(0, 1, 0), Rgba(0, 0, 1)]
    detector = ObjectDetector()
    
    for i, obj_pc in enumerate(object_clouds):
        meshcat.SetObject(
            f"object_{i}_pc",
            obj_pc,
            point_size=0.02,
            rgba=colors[i % len(colors)],
        )
        best_name, best_pose, best_score = detector.match_object(obj_pc)
        print(f"object {i}: best match = {best_name}, score = {best_score}")
    
    ########## computing mug ##########
    
    # meshcat.SetObject("pointcloud", pointcloud, point_size=0.1, rgba=Rgba(1, 0, 0, 0.8),)
    simulator.AdvanceTo(10.0)
    meshcat.StopRecording()
    meshcat.PublishRecording()
    
    while True:
        sleep(1)
        


if __name__ == "__main__":
    simulate_scenario()
    