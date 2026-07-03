from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("pool_cue_ball_corner_8_ball")
class PoolCueBallCorner8Ball(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Launch the cue ball towards the 8-ball at the start of the simulation.
        """
        if sid == 0:
            # Object 0 is the white cue ball
            # Object 1 is the black 8-ball
            cue_ball = self.all_objs[0]
            eight_ball = self.all_objs[1]
            
            # Get positions to calculate direction
            pos_cue = cue_ball.get_pos().cpu().numpy()
            pos_eight = eight_ball.get_pos().cpu().numpy()
            
            # Calculate direction vector from cue ball to 8-ball
            direction = pos_eight - pos_cue
            dist = np.linalg.norm(direction)
            
            if dist > 1e-6:
                direction = direction / dist
            
            # Set a velocity sufficient to hit the 8-ball hard
            # The 8-ball is near the pocket, so a direct hit should send it in
            speed = 4.5 
            velocity_vec = direction * speed
            
            # Apply initial linear velocity to the cue ball
            # dofs layout: [vx, vy, vz, wx, wy, wz]
            init_qvel = np.hstack([velocity_vec, [0.0, 0.0, 0.0]]).astype(np.float32)
            cue_ball.set_dofs_velocity(velocity=init_qvel, dofs_idx_local=np.arange(6))
