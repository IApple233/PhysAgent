from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("wooden_spinning_top_upright")
class WoodenSpinningTopUpright(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Apply a continuous torque around the Z-axis to make the top rotate steadily.
        """
        # Object index 0 is the spinning top
        obj_idx = 0
        
        # Define torque vector: [0, 0, torque_magnitude] for rotation around Z-axis (vertical)
        # We apply a constant torque to maintain steady rotation against friction
        torque_magnitude = 15.0
        torque_vector = np.array([[0.0, 0.0, torque_magnitude]], dtype=np.float32)
        
        # Apply torque to the object
        self.all_objs[obj_idx].solver.apply_links_external_torque(
            torque=torque_vector, 
            links_idx=[self.all_objs[obj_idx].idx]
        )
