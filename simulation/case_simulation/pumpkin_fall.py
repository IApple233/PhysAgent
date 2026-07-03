from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("pumpkin_fall")
class PumpkinFall(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        """
        Offset reconstructed objects along Genesis Y to compensate for 3D reconstruction misalignment.
        """
        self.all_obj_info[0]['center'][1] += 0.05
        self.all_obj_info[1]['center'][1] -= 0.3

        return super().add_entities_to_scene(scene, obj_materials, obj_vis_modes)

    def fix_particles(self):
        self.stool_init_pos = self.all_objs[1].get_particles()[0]

    def custom_simulation(self, sid):
        if sid < 18:
            self.all_objs[1].set_position(self.stool_init_pos)
    
    # def custom_simulation(self, sid):
    #     """
    #     Apply a strong downward force to the pumpkin to ensure it falls 
    #     forcefully onto the flimsy stool, causing it to break/crush.
    #     """
    #     # Apply force for the first 15 steps to initiate the fall
    #     if sid < 15:
    #         pumpkin = self.all_objs[0] # Index 0 is the pumpkin (rigid)
            
    #         # Force direction: Down (negative Z)
    #         force_dir = np.array([0.0, 0.0, -1.0])
            
    #         # Strength: High enough to crush the "flimsy" stool (simulated via low MPM_E)
    #         strength = 800.0 
            
    #         force = force_dir * strength
    #         force = force.reshape(1, 3)
            
    #         pumpkin.solver.apply_links_external_force(force=force, links_idx=[pumpkin.idx])
