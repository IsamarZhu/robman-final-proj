import numpy as np
from IPython.display import clear_output
from pydrake.all import (
    AbstractValue,
    AddMultibodyPlantSceneGraph,
    Concatenate,
    DiagramBuilder,
    JointSliders,
    LeafSystem,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Parser,
    PointCloud,
    RandomGenerator,
    Rgba,
    RigidTransform,
    RotationMatrix,
    Simulator,
    StartMeshcat,
    UniformlyRandomRotationMatrix,
)

from manipulation import running_as_notebook
from manipulation.scenarios import AddFloatingRpyJoint, AddRgbdSensors, ycb
from manipulation.utils import ConfigureParser
from manipulation.clutter import GraspCandidateCost, GenerateAntipodalGraspCandidate
from pathlib import Path

def draw_grasp_candidate(meshcat, X_G, prefix="gripper", draw_frames=True):
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    ConfigureParser(parser)
    parser.AddModelsFromUrl("package://manipulation/schunk_wsg_50_welded_fingers.sdf")
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("body"), X_G)
    plant.Finalize()

    params = MeshcatVisualizerParams()
    params.prefix = prefix
    params.delete_prefix_on_initialization_event = False
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat, params)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    diagram.ForcedPublish(context)

def make_internal_model():  
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    ConfigureParser(parser)
    parser.AddModelsFromUrl(Path("/workspaces/robman-final-proj/single_robot_arm/grasping/simplified_grasp.yaml"))
    plant.Finalize()
    return builder.Build()

def generate_antipodal_grasp(diagram, context, cloud):
    rng = np.random.default_rng()
    
    # making an internal model that ignores information about the tables and people
    # todo: may need to adjust later to move location of the robot once we move to a different location
    
    internal_model = make_internal_model()
    internal_model_context = internal_model.CreateDefaultContext()
    number_of_attempts = 300
    costs, X_Gs = [], []
    
    for i in range(number_of_attempts):
        cost, X_G = GenerateAntipodalGraspCandidate(diagram, internal_model_context, cloud, rng)
        
        if np.isfinite(cost):
            costs.append(cost)
            X_Gs.append(X_G)
            
        indices = np.asarray(costs).argsort()[:5]
        min_cost_XGs = []
        for idx in indices:
            min_cost_XGs.append(X_Gs[idx])
            
   
    plant = diagram.GetSubsystemByName("station").GetSubsystemByName("plant")
    plant_context = plant.GetMyContextFromRoot(context)
    wsg = plant.GetBodyByName("body")
    wsg_pose = plant.EvalBodyPoseInWorld(plant_context, wsg)
    wsg_position = wsg_pose.translation()
    wsg_rotation = wsg_pose.rotation()
    
    # we are optimizing for the grasp that is the following:
    #   1. closest to the gripper orientation with minimal rotation + translation
    #   2. aligned with the gripper facing down
    
    best_score = float('inf')
    best_X_G = None    
    down_direction = np.array([0, 0, -1])
    
    for X_G in min_cost_XGs:
        # distance penalty
        distance = np.linalg.norm(X_G.translation() - wsg_position)
        
        # rotation penalty
        grasp_z_axis = X_G.rotation().matrix()[:, 2]
        alignment = np.dot(grasp_z_axis, down_direction) 
        rotation_penalty = 1 - alignment  # 0 = perfectly aligned, 2 = pointing upwards
        
        # orientation difference from current gripper
        relative_rotation = wsg_rotation.inverse().multiply(X_G.rotation())
        angle_diff = np.arccos(np.clip((np.trace(relative_rotation.matrix()) - 1) / 2, -1, 1))
        
        # random coefficients for now
        score = distance + 2.0 * rotation_penalty + 0.5 * angle_diff
        
        if score < best_score:
            best_score = score
            best_X_G = X_G
            
    return best_X_G