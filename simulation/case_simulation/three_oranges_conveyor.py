from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import genesis as gs

@register_case("three_oranges_conveyor")
class ThreeOrangesConveyor(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Apply force to the two oranges on the conveyor belt to push them off the edge.
        The belt moves roughly from left to right (positive X) and slightly back (negative Y).
        """
        # Apply force for the first part of the simulation to get them moving
        if sid < 50:
            # Define force direction based on belt orientation (Right and slightly Back)
            # Normalized vector roughly [1, -0.15, 0]
            force_dir = np.array([1.0, -0.15, 0.0])
            force_dir = force_dir / np.linalg.norm(force_dir)
            force_dir = force_dir.reshape(1, 3)
            
            # Apply to orange 0 and orange 1 (indices 0 and 1)
            # Orange 2 is already on the floor, so we don't push it.
            strength = 2500.0 # Moderate force to overcome friction and accelerate
            
            for idx in [0, 1]:
                self.all_objs[idx].solver.apply_links_external_force(
                    force=force_dir * strength, 
                    links_idx=[self.all_objs[idx].idx]
                )
