import numpy as np

def place_object(plant, wsg_arm_instance, plant_context):
    # we detect that an object has been successfully placed down by checking 
    tau_contact = plant.get_generalized_contact_forces_output_port(wsg_arm_instance).Eval(plant_context)
    total_force = np.sum(np.abs(tau_contact))