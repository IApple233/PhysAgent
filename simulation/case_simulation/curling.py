from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("curling")
class CurlingStones(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Apply initial velocity to the red curling stone (index 1) 
        towards the yellow curling stone (index 0).
        """
        if sid == 0:
            # Object 0: Yellow Stone (Target)
            # Object 1: Red Stone (Striker)
            obj_yellow = self.all_objs[0]
            obj_red = self.all_objs[1]

            # Get current positions
            pos_yellow = obj_yellow.get_pos().cpu().numpy()
            pos_red = obj_red.get_pos().cpu().numpy()

            # Calculate direction from Red to Yellow
            direction = pos_yellow - pos_red
            dist = np.linalg.norm(direction)
            
            if dist > 1e-5:
                direction = direction / dist
                
                # Set initial velocity for the red stone
                # Speed ~2.5 m/s is appropriate for curling
                speed = 2.5
                velocity = direction * speed
                
                # Construct initial qvel: [vx, vy, vz, wx, wy, wz]
                # We only want linear velocity, no angular velocity initially
                init_qvel = np.array([
                    velocity[0], velocity[1], velocity[2],
                    0.0, 0.0, 0.0
                ], dtype=np.float32)
                
                obj_red.set_dofs_velocity(
                    velocity=init_qvel,
                    dofs_idx_local=np.arange(6),
                )
