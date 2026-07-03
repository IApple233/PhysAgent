from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("tennis_ball_wooden_board_knockdown")
class TennisBallWoodenBoardKnockdown(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Apply a small initial force to the tennis ball to start it rolling down the ramp.
        The force is directed towards the wooden board to ensure collision.
        """
        # Object 0: Tennis Ball
        # Object 1: Wooden Board
        
        if sid < 30:  # Apply force for the first 30 steps to initiate rolling
            # Get positions (pull to CPU for numpy math)
            pos_ball = self.all_objs[0].get_pos().cpu().numpy()
            pos_board = self.all_objs[1].get_pos().cpu().numpy()
            
            # Calculate direction from ball to board
            force_dir = pos_board - pos_ball
            norm = np.linalg.norm(force_dir)
            if norm > 1e-6:
                force_dir = force_dir / norm
            
            # Apply a small force to push the ball
            # Strength reduced from 50.0 to 5.0 to prevent explosion/tunneling
            strength = 5.0
            force_vec = (force_dir * strength).reshape(1, 3)
            
            self.all_objs[0].solver.apply_links_external_force(
                force=force_vec, 
                links_idx=[self.all_objs[0].idx]
            )
