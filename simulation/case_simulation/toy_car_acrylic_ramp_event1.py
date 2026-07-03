from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("toy_car_acrylic_ramp_event1")
class ToyCarAcrylicRampEvent1(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Apply initial velocity to simulate rolling down the ramp.
        The car is object 0.
        We apply a small linear velocity down the ramp (positive X)
        and an angular velocity to simulate rolling (around Y axis).
        High friction in YAML will stop it.
        """
        if sid == 0:
            obj = self.all_objs[0]
            # Linear velocity: moving right (down the ramp in image space, which is +X in Genesis usually)
            # Angular velocity: rolling forward. 
            # If car moves +X, rolling is rotation around -Y (right hand rule: thumb -Y, fingers curl +X to +Z? No.
            # Thumb -Y (into screen), fingers curl from +Z (up) to +X (right). Yes.
            # So angular velocity [0, -w, 0].
            
            # Adjust magnitudes based on "slowly"
            v_x = 0.5 
            w_y = -3.0 # Rolling
            
            init_qvel = np.array([
                v_x, 0.0, 0.0,
                0.0, w_y, 0.0
            ], dtype=np.float32)
            
            obj.set_dofs_velocity(
                velocity=init_qvel,
                dofs_idx_local=np.arange(6),
            )
