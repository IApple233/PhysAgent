from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("nailong_ring_half_hangs_on_body")
class NailongRingHalfHangsOnBody(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Launch the blue ring (object 1) towards the yellow monster (object 0).
        The ring flies leftward and slightly down to clip the monster's head/arm.
        """
        if sid == 0:
            ring_obj = self.all_objs[1]
            monster_obj = self.all_objs[0]
            
            # Get positions as numpy arrays
            pos_ring = ring_obj.get_pos().cpu().numpy()
            pos_monster = monster_obj.get_pos().cpu().numpy()
            
            # Calculate direction from ring to monster
            direction = pos_monster - pos_ring
            
            # We want primarily horizontal motion (leftward)
            # Zero out Z to make it fly horizontally initially, letting gravity act later
            # Or keep a small Z component if aiming is needed. 
            # "flies leftward... instead of dropping straight down" implies strong horizontal component.
            # Let's aim directly but ensure horizontal dominance.
            # The monster is lower, so direction[2] is negative.
            # Let's keep the direction but normalize.
            
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction = direction / norm
            
            # Keep the launch tunable from config so slow-motion variants do not
            # need a separate hard-coded handler.
            speed = float(self.config.get("ring_initial_speed", 1.5))
            velocity = direction * speed
            
            # Angular velocity to make it tumble and hang diagonally
            # Rotation around X and Z axes
            angular_velocity = np.array(
                self.config.get("ring_initial_angular_velocity", [0.9, 0.0, 0.6])
            )
            
            # Set velocity for the ring (free rigid body, 6 DoFs)
            init_qvel = np.concatenate([velocity, angular_velocity]).astype(np.float32)
            ring_obj.set_dofs_velocity(
                velocity=init_qvel,
                dofs_idx_local=np.arange(6),
            )
