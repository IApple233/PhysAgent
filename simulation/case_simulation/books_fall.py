from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("books_fall")
class BooksFall(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Pushes the stack of books to the right (positive X) to simulate them falling off the desk.
        """
        # Apply force for the first 40 steps to push the books off the table
        if sid < 60:
            # Force direction: Push to the right (positive X)
            # We apply force to all books to ensure the stack moves together initially
            force_strength = 20.0
            force_direction = np.array([-1.0, -0.6, 0.0])
            force_direction = force_direction / np.linalg.norm(force_direction)
            force_direction = force_direction.reshape(1, 3)
            
            force_vector = force_direction * force_strength
            
            # Apply to all 4 books (indices 0, 1, 2, 3)
            for i in range(4):
                self.all_objs[i].solver.apply_links_external_force(
                    force=force_vector, 
                    links_idx=[self.all_objs[i].idx]
                )
        
        # Add a slight random torque to the top books to encourage chaotic tumbling once they are in the air
        if 20 < sid < 60:
            for i in range(2): # Top two books (Blue and White)
                # Random torque
                torque_strength = 1.0
                torque = np.array([
                    np.random.uniform(-torque_strength, torque_strength),
                    np.random.uniform(-torque_strength, torque_strength),
                    np.random.uniform(-torque_strength, torque_strength)
                ]).reshape(1, 3)
                
                self.all_objs[i].solver.apply_links_external_torque(
                    torque=torque,
                    links_idx=[self.all_objs[i].idx]
                )

    def detect_ground_plane(self, ground_plane):
        # The table is the support, but we also need the floor.
        # The background collision mesh will handle the floor.
        # We don't need to override this unless we want to disable the infinite plane 
        # or adjust its offset. The default behavior with background_collision_mode="mesh"
        # and support_object_points should work.
        # However, to ensure the books fall onto the floor mesh and not an infinite plane 
        # that might be at the wrong height, we can rely on the reconstructed background.
        # The system handles this via background_collision_mode.
        pass

    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        """
        No specific initial position tweaks needed; we simulate the object 
        as reconstructed (tilted) to match the image context.
        """
        for i in range(len(self.all_obj_info)):
            # 【关键修改 4】将 0.05 改为 0.002 (2毫米)
            # 5厘米的初始跌落对于物理引擎来说太高了，会产生巨大的反弹力
            # 2毫米刚好足够防止穿模，又能让书本迅速且平稳地“贴”在桌面上
            self.all_obj_info[i]['center'][2] += 0.15
        return super().add_entities_to_scene(scene, obj_materials, obj_vis_modes)
