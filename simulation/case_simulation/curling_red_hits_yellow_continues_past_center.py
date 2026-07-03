from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("curling_red_hits_yellow_continues_past_center")
class CurlingRedHitsYellowContinuesPastCenter(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Implementation for rigid body forces/torques or one-shot initial
        linear/angular velocities based on sid.
        """
        # Apply initial velocity to the red stone (Object 0) at the start of simulation
        if sid == 0:
            red_stone = self.all_objs[0]
            yellow_stone = self.all_objs[1]

            # Get current positions
            pos_red = red_stone.get_pos().cpu().numpy()
            pos_yellow = yellow_stone.get_pos().cpu().numpy()

            # Calculate direction from red to yellow
            # Red is "back" (top of image), Yellow is "center" (bottom of image).
            # We want Red to move towards Yellow.
            direction = pos_yellow - pos_red
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction = direction / norm
            
            # Set a high initial velocity for the slide
            # Curling stones are heavy (~20kg), so we need momentum to push the other stone and continue.
            speed = 4.0  # m/s, fast slide
            velocity_linear = direction * speed
            
            # Add a slight angular velocity (spin) for realism, typical in curling
            # Spin around Z axis (vertical)
            angular_velocity = np.array([0.0, 0.0, 3.0], dtype=np.float32)
            
            # Combine linear and angular velocity [vx, vy, vz, wx, wy, wz]
            init_qvel = np.concatenate([velocity_linear, angular_velocity]).astype(np.float32)
            
            # Set velocity for the red stone (first 6 DoFs)
            red_stone.set_dofs_velocity(
                velocity=init_qvel,
                dofs_idx_local=np.arange(6),
            )
