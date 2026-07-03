from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("lamp")
class Lamp(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Apply a gentle constant force to simulate river current drifting the lantern.
        """
        # Object index 0 is the lantern
        obj_idx = 0
        
        # Define a gentle force direction (diagonal movement looks more natural)
        # Normalized direction vector [x, y, z]
        # Z is 0 because it's floating on surface, force is horizontal
        direction = np.array([1.0, 0.5, 0.0])
        direction = direction / np.linalg.norm(direction)
        direction = direction.reshape(1, 3)
        
        # Force magnitude: small to simulate gentle water current
        # Since friction is low (water-like), small force creates movement
        force_magnitude = 0.5
        
        force = direction * force_magnitude
        
        # Apply force to the lantern
        # We apply it continuously to simulate ongoing current
        self.all_objs[obj_idx].solver.apply_links_external_force(
            force=force, 
            links_idx=[self.all_objs[obj_idx].idx]
        )
        