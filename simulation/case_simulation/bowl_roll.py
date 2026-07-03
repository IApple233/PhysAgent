from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("bowl_roll")
class BowlRoll(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Apply a small initial torque/force to encourage the bowl to roll/fall 
        if it is in an unstable equilibrium or resting state.
        """
        # Only apply force at the beginning to initiate motion
        if sid < 10:
            # Object 0 is the bowl
            obj_idx = 0
            
            # Get current position to calculate force direction relative to object
            # We want to push it slightly to encourage rolling
            pos = self.all_objs[obj_idx].get_pos().cpu().numpy()
            
            # Define a push direction (e.g., along the table surface)
            # Since we don't know the exact table normal without querying the support mesh,
            # we assume a general horizontal push + slight down to encourage contact
            # Using a generic horizontal vector is safer if gravity is standard Z-down
            push_dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            push_dir = push_dir / np.linalg.norm(push_dir)
            push_dir = push_dir.reshape(1, 3)
            
            # Apply a small force to start the roll
            force_strength = 2.0
            self.all_objs[obj_idx].solver.apply_links_external_force(
                force=push_dir * force_strength, 
                links_idx=[self.all_objs[obj_idx].idx]
            )
            
            # Also apply a small torque to encourage tipping/rolling
            torque_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64) # Rotate around Y
            torque = torque_axis * 0.5
            torque = torque.reshape(1, 3)
            
            self.all_objs[obj_idx].solver.apply_links_external_torque(
                torque=torque,
                links_idx=[self.all_objs[obj_idx].idx]
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
            self.all_obj_info[i]['center'][2] += 0.1
            self.all_obj_info[i]['center'][1] -= 0.2
        return super().add_entities_to_scene(scene, obj_materials, obj_vis_modes)
