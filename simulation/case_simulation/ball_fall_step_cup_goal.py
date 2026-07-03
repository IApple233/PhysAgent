from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("ball_fall_step_cup_goal")
class BallFallStepCupGoal(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        The ball (index 0) needs to roll from the upper step (right/background)
        towards the cup (left/foreground).
        We apply an initial velocity to simulate the roll.
        """
        if sid == 0:
            # Object 0 is the red ball.
            # Object 1 is the paper cup.
            # The ball is at x~600 (right), y~380 (back).
            # The cup is at x~350 (left), y~600 (front).
            # We need velocity in -X (left) and -Y (forward/towards camera).
            
            ball = self.all_objs[0]
            
            # Linear velocity: Left (-X), Forward (-Y), 0 Z.
            # Magnitude: Needs to be enough to reach the cup but not overshoot wildly.
            # vx = -1.5, vy = -0.5 seems reasonable for the scale.
            init_qvel = np.array([-1.5, -0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            
            ball.set_dofs_velocity(
                velocity=init_qvel,
                dofs_idx_local=np.arange(6),
            )
