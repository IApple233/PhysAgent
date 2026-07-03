from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("firework_upper_right_event1")
class FireworkUpperRightEvent1(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        The rocket smokes for a moment (sid < 20) without moving.
        Then it ignites suddenly and shoots upward in a steep straight line.
        """
        # Delay phase: rocket sits still (smoking visually in video gen)
        if sid < 20:
            return

        # Launch phase: apply strong upward force
        obj = self.all_objs[0]
        
        # Force direction: Global Z (up). 
        # Magnitude: High enough to overcome gravity and accelerate rapidly.
        force = np.array([[0, 0, 200.0]], dtype=np.float32)
        
        # Apply force to the rigid body
        obj.solver.apply_links_external_force(force=force, links_idx=[obj.idx])
