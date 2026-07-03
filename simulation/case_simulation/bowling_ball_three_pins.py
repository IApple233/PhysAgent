from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("bowling_ball_three_pins")
class BowlingBallThreePins(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Apply a force to the bowling ball (object 0) to roll it towards the pins (objects 1, 2, 3).
        """
        # Apply force for the first 10 frames to simulate the throw/roll
        if sid < 10:
            ball_idx = 0
            # Target the middle pin (index 2) for direction
            target_idx = 2
            
            # Get positions (pull to CPU for numpy math)
            pos_ball = self.all_objs[ball_idx].get_pos().cpu().numpy()
            pos_target = self.all_objs[target_idx].get_pos().cpu().numpy()
            
            # Calculate direction vector
            force_dir = pos_target - pos_ball
            
            # Normalize
            norm = np.linalg.norm(force_dir)
            if norm > 1e-6:
                force_dir = force_dir / norm
            
            # Ensure force is horizontal (along the lane) to avoid lifting the ball
            # Assuming Z is up.
            force_dir[2] = 0.0
            
            # Renormalize after zeroing Z
            norm_xy = np.linalg.norm(force_dir)
            if norm_xy > 1e-6:
                force_dir = force_dir / norm_xy

            # Apply force
            # Strength tuned for realistic bowling speed (~5-8 m/s)
            strength = 50
            force_vec = (force_dir * strength).reshape(1, 3)
            
            self.all_objs[ball_idx].solver.apply_links_external_force(
                force=force_vec, 
                links_idx=[self.all_objs[ball_idx].idx]
            )

    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        """
        No specific initial position tweaks needed; we simulate the object 
        as reconstructed (tilted) to match the image context.
        """
        for i in range(len(self.all_obj_info)):
            # 【关键修改 4】将 0.05 改为 0.002 (2毫米)
            # 5厘米的初始跌落对于物理引擎来说太高了，会产生巨大的反弹力
            # 2毫米刚好足够防止穿模，又能让书本迅速且平稳地“贴”在桌面上
            if i!=0:
                self.all_obj_info[i]['center'][2] = self.all_obj_info[0]['center'][2]+(self.all_obj_info[i]['size'][2]-self.all_obj_info[0]['size'][2])*0.45
        return super().add_entities_to_scene(scene, obj_materials, obj_vis_modes)