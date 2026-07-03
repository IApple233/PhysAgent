from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("books_fall_top_three")
class BooksFallTopThree(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Apply a horizontal force to the top three books (indices 0, 1, 2)
        to push them off the table to the left (-X direction).
        The bottom book (index 3, support) remains static.
        """
        # Apply force for the first 10 steps to give an impulse
        if sid < 10:
            # Force direction: Left (-X)
            # Magnitude: Strong enough to overcome friction and push the stack
            force_magnitude = 300.0 
            force_direction = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
            force = force_direction * force_magnitude
            force = force.reshape(1, 3)
            
            # Apply to top 3 books (indices 0, 1, 2)
            for i in range(3):
                self.all_objs[i].solver.apply_links_external_force(
                    force=force, 
                    links_idx=[self.all_objs[i].idx]
                )
