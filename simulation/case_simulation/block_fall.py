from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("block_fall")
class BlockFall(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Apply a gentle wind force to the yellow cube.
        The force starts small (vibration) and gradually increases to push it off the edge.
        """
        # Object 0 is the yellow cube
        cube_idx = 0
        
        # Get current position to calculate direction if needed, 
        # but for a global wind, a fixed direction is often sufficient.
        # Based on the image, the table is to the left, the drop is to the right.
        # So force should be in +X direction.
        
        # Calculate force magnitude based on time step (sid)
        # "tiny vibration" initially, then "slowly slides"
        if sid < 10:
            force_mag = 0.5  # Very gentle vibration
        elif sid < 40:
            force_mag = 1 + (sid - 10) * 0.5  # Gradual increase
        else:
            force_mag = 0.5  # Sustained gentle push
            
        # Force direction: +X (towards the edge/drop)
        # We assume the camera view aligns roughly with standard axes where right is +X.
        # If the table is rotated, this might need adjustment, but +X is the best guess
        # for "inside to outside" based on the image composition.
        force_direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        force_direction = force_direction / np.linalg.norm(force_direction)
        force_direction = force_direction.reshape(1, 3)
        
        force = force_direction * force_mag
        
        # Apply force
        self.all_objs[cube_idx].solver.apply_links_external_force(
            force=force, 
            links_idx=[self.all_objs[cube_idx].idx]
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
            self.all_obj_info[i]['center'][2] -= 0.012
            self.all_obj_info[i]['center'][1] += 0.15
            self.all_obj_info[i]['center'][0] += 0.1
        return super().add_entities_to_scene(scene, obj_materials, obj_vis_modes)
