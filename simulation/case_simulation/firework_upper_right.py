from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("firework_upper_right")
class FireworkUpperRight(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Apply a constant force to the rocket to simulate propulsion.
        The force is directed up and right to match the "upper-right" trajectory.
        Since the force is applied at the center of mass (default for apply_links_external_force on the whole body),
        there will be no torque, resulting in a straight-line translation with no rotation ("no turn, wobble").
        """
        # Force vector: [Right (X), Forward/Back (Y), Up (Z)]
        # We want Up and Right.
        # Magnitude tuned for dt=0.01 and light object (rho=300).
        # Fz needs to overcome gravity (m*g) and provide upward acceleration.
        # Fx provides rightward acceleration.
        # Ratio Fz/Fx ~ 2 or 3 for a steep diagonal.
        thrust_force = np.array([3.0, 0.0, 12.0], dtype=np.float32).reshape(1, 3)
        
        # Apply force to the rocket (object index 0)
        self.all_objs[0].solver.apply_links_external_force(
            force=thrust_force, 
            links_idx=[self.all_objs[0].idx]
        )
