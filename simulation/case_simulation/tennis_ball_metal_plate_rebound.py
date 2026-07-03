from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import genesis as gs

@register_case("tennis_ball_metal_plate_rebound")
class TennisBallMetalPlateRebound(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Ensure the ball moves if it gets stuck, though with the corrected ramp geometry
        and gravity, it should roll naturally.
        """
        # Only apply checks after a few steps to let physics settle
        if sid > 10:
            # Get ball position and velocity
            ball_obj = self.all_objs[0]
            pos = ball_obj.get_pos().cpu().numpy()
            vel = ball_obj.get_vel().cpu().numpy()
            
            # If velocity is very low (stuck), apply a small nudge down the slope
            # The slope goes from left (high) to right (low).
            # A nudge in +X direction should help.
            if np.linalg.norm(vel) < 0.01:
                nudge = np.array([[0.5, 0.0, 0.0]]) # Small force in X direction
                ball_obj.solver.apply_links_external_force(force=nudge, links_idx=[ball_obj.idx])
