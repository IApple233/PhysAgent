from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("bowling_ball_right_pin")
class BowlingBallRightPin(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Apply initial velocity to the bowling ball to simulate a diagonal roll
        that hits only the right-most pin.
        Object 0: Ball
        Object 1: Left Pin
        Object 2: Middle Pin
        Object 3: Right Pin
        """
        if sid == 0:
            ball_obj = self.all_objs[0]
            
            # Estimate radius from visual size relative to pins.
            # Ball is roughly 0.11m radius.
            radius = 0.11
            
            # Target: Hit Right Pin (Obj 3) while missing Left (Obj 1) and Middle (Obj 2).
            # Trajectory: Diagonal from Front-Left to Back-Right.
            # Forward velocity (Y) needs to be significant.
            # Lateral velocity (X) needs to be positive (Right) to cross the lane.
            # Ratio vx/vy should be small but non-zero to create the diagonal path.
            
            vy = 6.0  # Forward speed (m/s)
            vx = 0.8  # Lateral speed (m/s) - enough to drift right over the distance
            
            # Rolling motion: omega_x = -vy / R (for forward roll +Y)
            wx = -vy / radius
            
            # Add slight sidespin (wz) to enhance the "veer" to the right.
            # Negative wz (CW) creates Magnus force to the right (+X).
            wz = -5.0 
            
            init_qvel = np.array([
                vx, vy, 0.0,  # Linear velocity
                wx, 0.0, wz   # Angular velocity
            ], dtype=np.float32)
            
            ball_obj.set_dofs_velocity(
                velocity=init_qvel,
                dofs_idx_local=np.arange(6),
            )
