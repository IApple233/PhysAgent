from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("ball_fall_in_cup")
class BallFallInCup(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        # Adjust initial positions to ensure stability and correct trajectory
        for i in range(len(self.all_obj_info)):
            center = self.all_obj_info[i]['center']
            
            # Lift objects slightly to ensure they clear the reconstructed step mesh roughness
            # and sit securely on the support surface.
            # We add to the center Z coordinate.
            lift_offset = torch.tensor([0.0, 0.0, 0.06], device=center.device, dtype=center.dtype)
            self.all_obj_info[i]['center'] = center + lift_offset
            
            if i == 0: # Ball (Index 0)
                # Ball is on the upper step (right side in image, higher X).
                # Nudge it left (negative X) towards the edge and the cup.
                nudge = torch.tensor([-0.12, 0.0, 0.0], device=center.device, dtype=center.dtype)
                self.all_obj_info[i]['center'] += nudge
                
            elif i == 1: # Cup (Index 1)
                # Cup is on the lower step (left side in image, lower X).
                # Nudge it right (positive X) to align it under the ball's expected path.
                nudge = torch.tensor([0.06, 0.0, 0.0], device=center.device, dtype=center.dtype)
                self.all_obj_info[i]['center'] += nudge

        return super().add_entities_to_scene(scene, obj_materials, obj_vis_modes)

    def custom_simulation(self, sid):
        # Apply a continuous force to the ball to roll it off the step towards the cup
        # Apply for the first 15 frames to give it momentum
        if sid <= 15:
            ball_obj = self.all_objs[0]
            cup_obj = self.all_objs[1]
            
            # Get current positions (pull to CPU for numpy vector math)
            pos_ball = ball_obj.get_pos().cpu().numpy()
            pos_cup = cup_obj.get_pos().cpu().numpy()
            
            # Calculate direction from ball to cup
            # Ball is at higher X, Cup at lower X. Direction should be negative X.
            force_dir = pos_cup - pos_ball
            
            # We want a horizontal push to roll it off the edge.
            # Zero out Z to let gravity handle the fall naturally.
            force_dir[2] = 0.0 
            
            norm = np.linalg.norm(force_dir)
            if norm > 1e-8:
                force_dir = force_dir / norm
            
            # Apply force
            # Strength needs to be sufficient to overcome friction and roll the ball.
            strength = 12.0 
            force_vec = (force_dir * strength).reshape(1, 3).astype(np.float32)
            
            ball_obj.solver.apply_links_external_force(force=force_vec, links_idx=[ball_obj.idx])

    def detect_ground_plane(self, ground_plane):
        # Ensure the clearance is set correctly for the static support
        self.config['static_support_clearance'] = 0.06
        return super().detect_ground_plane(ground_plane)
