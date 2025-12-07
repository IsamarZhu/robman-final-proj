"""
Setup utilities for building Drake simulation environment.
"""

from pathlib import Path
from pydrake.all import DiagramBuilder, Simulator, StartMeshcat
from pydrake.geometry import MakeRenderEngineVtk, RenderEngineVtkParams
from pydrake.systems.primitives import ConstantVectorSource
from manipulation.station import LoadScenario, MakeHardwareStation, AddPointClouds

from grasping.command_systems import JointPositionCommandSource, WsgCommandSource


class SimulationEnvironment:
    def __init__(self, scenario_number, wsg_open=0.107):
        self.scenario_path = Path(f"src/environment/scenario_{scenario_number}.yaml")
        self.wsg_open = wsg_open
        self.meshcat = None
        self.diagram = None
        self.simulator = None
        self.context = None
        self.plant = None
        self.plant_context = None
        self.iiwa_model = None
        self.wsg_model = None
        self.cmd_source = None
        self.wsg_cmd_source = None
        self.scenario_number = scenario_number

    def build(self):

        self.meshcat = StartMeshcat()
        self.meshcat.SetProperty("/Grid", "visible", False)
        with open(self.scenario_path, "r") as f:
            scenario_yaml = f.read()
        scenario = LoadScenario(data=scenario_yaml)

        builder = DiagramBuilder()
        station = builder.AddSystem(MakeHardwareStation(scenario, meshcat=self.meshcat))
        self.plant = station.GetSubsystemByName("plant")
        scene_graph = station.GetSubsystemByName("scene_graph")

        renderer_name = "renderer"
        if not scene_graph.HasRenderer(renderer_name):
            scene_graph.AddRenderer(
                renderer_name, MakeRenderEngineVtk(RenderEngineVtkParams())
            )

        # get robot models
        self.iiwa_model = self.plant.GetModelInstanceByName("iiwa_arm")
        self.wsg_model = self.plant.GetModelInstanceByName("wsg_arm")

        # create command sources
        default_context = self.plant.CreateDefaultContext()
        q_start = self.plant.GetPositions(default_context, self.iiwa_model)

        self.cmd_source = builder.AddSystem(JointPositionCommandSource(q_start))
        builder.Connect(
            self.cmd_source.get_output_port(),
            station.GetInputPort("iiwa_arm.desired_state"),
        )

        self.wsg_cmd_source = builder.AddSystem(WsgCommandSource(self.wsg_open))
        builder.Connect(
            self.wsg_cmd_source.get_output_port(),
            station.GetInputPort("wsg_arm.position"),
        )

        wsg_force_limit = builder.AddSystem(ConstantVectorSource([40.0]))
        builder.Connect(
            wsg_force_limit.get_output_port(),
            station.GetInputPort("wsg_arm.force_limit"),
        )

        # add point cloud cameras
        print("Setting up cameras...")
        to_point_cloud = AddPointClouds(
            scenario=scenario,
            station=station,
            builder=builder,
            meshcat=None,
        )

        for name, system in to_point_cloud.items():
            builder.ExportOutput(
                system.GetOutputPort("point_cloud"),
                f"{name}.point_cloud",
            )

        self.diagram = builder.Build()
        self.simulator = Simulator(self.diagram)
        self.context = self.simulator.get_mutable_context()
        self.plant_context = self.plant.GetMyMutableContextFromRoot(self.context)

        self.simulator.Initialize()
        self.simulator.AdvanceTo(1.0)
        self.diagram.ForcedPublish(self.context)

    def settle_scene(self, duration=2.0):
        t = self.simulator.get_context().get_time()
        self.simulator.AdvanceTo(t + duration)
        self.diagram.ForcedPublish(self.context)