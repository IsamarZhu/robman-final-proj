import numpy as np
from perception.object_detection import detect_and_locate_object
from grasping.motion_primitives import pick_object
from grasping.antipodal_grasping import get_best_antipodal_grasp


class PickAndPlaceTask:
    """manages detection and pick-and-place execution for objects yay"""

    def __init__(
        self,
        env,
        approach_height=0.15,
        lift_height=0.20,
        grasp_offset=0.00,
        wsg_open=0.107,
        wsg_closed=0.015,
        move_time=2.5,
        grasp_time=2.0,
        lift_time=4.0,
        dbscan_eps=0.03,
        dbscan_min_samples=50,
        use_antipodal_grasping=True,
        num_grasp_samples=2000,
    ):
        self.env = env
        self.approach_height = approach_height
        self.lift_height = lift_height
        self.grasp_offset = grasp_offset
        self.wsg_open = wsg_open
        self.wsg_closed = wsg_closed
        self.move_time = move_time
        self.grasp_time = grasp_time
        self.lift_time = lift_time
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.use_antipodal_grasping = use_antipodal_grasping
        self.num_grasp_samples = num_grasp_samples

    def execute(self, target_object):
        """
        execute complete pick-and-place for a target object using antipodal grasping
        """
        print(f"PICKING OBJECT: {target_object.upper()}")

        # detection of object
        detection_result = detect_and_locate_object(
            self.env.scenario_number,
            self.env.diagram,
            self.env.context,
            self.env.meshcat,
            target_object=target_object,
            dbscan_eps=self.dbscan_eps,
            dbscan_min_samples=self.dbscan_min_samples,
            grasp_offset=self.grasp_offset,
        )
        
        X_WO, grasp_center_xyz, object_top_z, object_cloud = detection_result

        # use antipodal grasping to find best grasp pose
        X_WG_grasp = None
        if self.use_antipodal_grasping:
            print("\nFinding antipodal grasp...")
            X_WG_grasp = get_best_antipodal_grasp(
                object_cloud,
                rng=np.random.default_rng(),
                num_samples=self.num_grasp_samples,
                meshcat=self.env.meshcat,
            )
            
            if X_WG_grasp is not None:
                print(f"  Found valid antipodal grasp!")
            else:
                print("  No valid antipodal grasp found, using default downward grasp")

        # executing the pick and place
        pick_object(
            self.env.meshcat,
            self.env.simulator,
            self.env.diagram,
            self.env.plant,
            self.env.plant_context,
            self.env.iiwa_model,
            self.env.wsg_model,
            self.env.cmd_source,
            self.env.wsg_cmd_source,
            grasp_center_xyz,
            object_top_z,
            approach_height=self.approach_height,
            lift_height=self.lift_height,
            wsg_open=self.wsg_open,
            wsg_closed=self.wsg_closed,
            move_time=self.move_time,
            grasp_time=self.grasp_time,
            lift_time=self.lift_time,
            X_WG_grasp=X_WG_grasp,
        )