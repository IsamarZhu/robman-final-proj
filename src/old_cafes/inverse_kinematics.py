from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    Parser,
    DiagramBuilder,
    Concatenate,
    PointCloud,
    Simulator,
    StartMeshcat,
    SpatialVelocity,
    Rgba,
    RigidTransform,
    RotationMatrix,
    InverseKinematics,
    Solve,
)

def ik_for_ee_position(plant,
                    plant_context,
                    wsg_arm_instance,
                    R_WG_desired,
                    robot_base_instance,
                    iiwa_arm_instance,
                    p_W_target,
                    use_orientation: bool,
                    theta_bound: float = 1.0,
                    clamp_to_tray: bool = True):
    ik = InverseKinematics(plant, plant_context)
    q = ik.q()

    world_frame = plant.world_frame()
    ee_frame = plant.GetFrameByName("body", wsg_arm_instance)

    # lock base
    base_body = plant.GetBodyByName("robot_base_link", robot_base_instance)
    X_WB = plant.EvalBodyPoseInWorld(plant_context, base_body)
    p_WB = X_WB.translation()
    ik.AddPositionConstraint(
        base_body.body_frame(), [0, 0, 0],
        world_frame,
        p_WB, p_WB,
    )
    R_WB = X_WB.rotation()
    ik.AddOrientationConstraint(
        world_frame, R_WB,
        base_body.body_frame(), RotationMatrix(),
        0.0,
    )

    # --- position constraint box ---
    p = p_W_target.copy()
    if clamp_to_tray:
        # never let the lower z bound go below mug_bottom_z + margin
        p[2] = max(p[2], 1.04 + 0.01)

    p_tol = 0.06

    lower = p.copy()
    upper = p.copy()

    # x,y ± p_tol
    lower[0:2] = p[0:2] - p_tol
    upper[0:2] = p[0:2] + p_tol

    if clamp_to_tray:
        # z ∈ [p[2], p[2] + p_tol]  (one-sided; keeps us above tray/mugs)
        lower[2] = p[2]
        upper[2] = p[2] + p_tol
    else:
        # symmetric z near table targets
        lower[2] = p[2] - p_tol
        upper[2] = p[2] + p_tol

    ik.AddPositionConstraint(
        ee_frame, [0, 0, 0],
        world_frame,
        lower,
        upper,
    )

    if use_orientation:
        ik.AddOrientationConstraint(
            world_frame,
            R_WG_desired,
            ee_frame,
            RotationMatrix(),
            theta_bound,
        )

    prog = ik.prog()
    q_seed = plant.GetPositions(plant_context)
    prog.SetInitialGuess(q, q_seed)

    result = Solve(prog)
    # print(f"IK target (raw) {p_W_target}, (used) {p}, clamp_to_tray={clamp_to_tray}")

    if not result.is_success():
        raise RuntimeError(f"IK failed for target {p_W_target} (clamped {p})")

    q_sol = result.GetSolution(q)
    plant.SetPositions(plant_context, q_sol)
    return plant.GetPositions(plant_context, iiwa_arm_instance)