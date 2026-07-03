from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("persimmon")
class ThreePersimmon(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Apply lateral forces to top and middle persimmons to make them tumble off.
        Bottom persimmon (index 2) remains in place with high friction.
        Use position-based force direction for accurate collision targeting.
        """
        # Only apply force during initial frames to initiate the tumble
        if sid > 15:
            return
        
        # Top persimmon (index 0) - push to the right
        pos_top = self.all_objs[0].get_pos().cpu().numpy()
        pos_bottom = self.all_objs[2].get_pos().cpu().numpy()
        
        # Calculate force direction from bottom to top persimmon, then add lateral component
        force_direction_top = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        force_direction_top = force_direction_top / np.linalg.norm(force_direction_top)
        force_direction_top = force_direction_top.reshape(1, 3)
        
        # Middle persimmon (index 1) - push to the right with slight downward angle
        pos_middle = self.all_objs[1].get_pos().cpu().numpy()
        force_direction_middle = np.array([1.0, 0.0, -0.3], dtype=np.float64)
        force_direction_middle = force_direction_middle / np.linalg.norm(force_direction_middle)
        force_direction_middle = force_direction_middle.reshape(1, 3)
        
        # Apply forces with appropriate strength
        force_strength_top = 15.0
        force_strength_middle = 12.0
        
        self.all_objs[0].solver.apply_links_external_force(
            force=force_direction_top * force_strength_top, 
            links_idx=[self.all_objs[0].idx]
        )
        
        self.all_objs[1].solver.apply_links_external_force(
            force=force_direction_middle * force_strength_middle, 
            links_idx=[self.all_objs[1].idx]
        )
        
        # Bottom persimmon (index 2) - no force applied, stays in place
        # High friction from YAML config keeps it stationary
        
    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        """
        Ensure proper initial positioning. Bottom persimmon should be on table.
        Use gravity_direction if background_collision_sets_gravity is true.
        """
        # Check if we should use reconstructed gravity direction
        if self.config.get('background_collision_sets_gravity', False):
            reference_center = self.all_obj_info[0]['center']
            normal = torch.as_tensor(
                self.config.get('gravity_direction', [0, 0, 1]),
                dtype=reference_center.dtype,
                device=reference_center.device,
            )
            normal = normal / torch.linalg.norm(normal).clamp_min(1e-8)
            # Slightly lift all persimmons to ensure clean collision initialization
            for i in range(len(self.all_obj_info)):
                self.all_obj_info[i]['center'] += normal * 0.02
        
        return super().add_entities_to_scene(scene, obj_materials, obj_vis_modes)
