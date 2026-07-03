from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("scooter_collide")
class ScooterCollide(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Implementation for Rigid body forces/torques based on sid.
        Remember to use pos_B - pos_A for ALL collision targeting!
        """
        # Apply a push force to the scooter (object 0) towards the trash can (object 1)
        # only at the beginning of the simulation to initiate movement.
        if 1<= sid <= 12:
            scooter_idx = 0
            trash_can_idx = 1
            
            # Get positions (pull to CPU numpy for calculation)
            pos_scooter = self.all_objs[scooter_idx].get_pos().cpu().numpy()
            pos_trash_can = self.all_objs[trash_can_idx].get_pos().cpu().numpy()
            
            # Calculate direction from scooter to trash can
            force_direction = pos_trash_can - pos_scooter
            force_direction = force_direction / (np.linalg.norm(force_direction) + 1e-8)
            force_direction = force_direction.reshape(1, 3)
            
            # Apply force to the scooter
            # Strength needs to be sufficient to overcome friction and move the scooter
            strength = 2.3
            self.all_objs[scooter_idx].solver.apply_links_external_force(
                force=force_direction * strength, 
                links_idx=[self.all_objs[scooter_idx].idx]
            )
