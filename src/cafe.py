import time
from pydrake.all import ModelVisualizer, Simulator, StartMeshcat

from manipulation import ConfigureParser, running_as_notebook
from manipulation.station import LoadScenario, MakeHardwareStation
meshcat = StartMeshcat()

visualizer = ModelVisualizer(meshcat=meshcat)
ConfigureParser(visualizer.parser())
visualizer.AddModels(
    url="package://manipulation/mobile_iiwa14_primitive_collision.urdf"
)
visualizer.parser().AddModels(
    url="package://drake_models/manipulation_station/table_wide.sdf"
)
visualizer.Run(loop_once=not running_as_notebook)
meshcat.DeleteAddedControls()

time.sleep(30)