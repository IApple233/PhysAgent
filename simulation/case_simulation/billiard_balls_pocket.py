from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("billiard_balls_pocket")
class BilliardBallsPocket(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Apply impulse force to pocket the black ball (object index 1).
        Use collision targeting with position-based direction calculation.
        Object indices: 0=white/cue ball, 1=black ball, 2=red ball, 3=blue ball
        """
        if sid == 0:
            # Get positions for collision targeting
            black_ball_pos = self.all_objs[1].get_pos().cpu().numpy()
            pocket_target_pos = np.array([[[0.75, 0.35, 0.0]]])  # Approximate pocket location
            
            # Calculate force direction from black ball to pocket
            force_direction = pocket_target_pos - black_ball_pos
            force_direction = force_direction / np.linalg.norm(force_direction)
            force_direction = force_direction.reshape(1, 3)
            
            # Apply impulse force to black ball (object index 1)
            impulse_strength = 15.0
            self.all_objs[1].solver.apply_links_external_force(
                force=force_direction * impulse_strength,
                links_idx=[self.all_objs[1].idx]
            )
            
        elif sid <= 20:
            # Apply small continuous force to ensure ball reaches pocket
            black_ball_pos = self.all_objs[1].get_pos().cpu().numpy()
            pocket_target_pos = np.array([[[0.75, 0.35, 0.0]]])
            
            force_direction = pocket_target_pos - black_ball_pos
            force_direction = force_direction / np.linalg.norm(force_direction)
            force_direction = force_direction.reshape(1, 3)
            
            # Smaller continuous force
            continuous_strength = 2.0
            self.all_objs[1].solver.apply_links_external_force(
                force=force_direction * continuous_strength,
                links_idx=[self.all_objs[1].idx]
            )

    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        """
        Adjust initial positions to ensure balls are properly on the table surface.
        Use self.all_obj_info only (self.all_objs not available yet).
        """
        # Slightly lift balls to ensure they start above the table surface
        for i in range(len(self.all_obj_info)):
            # Get the support normal from config if background_collision_sets_gravity is true
            normal = np.asarray(self.config.get('gravity_direction', [0, 0, 1]), dtype=np.float64)
            # Small offset to prevent initial penetration
            self.all_obj_info[i]['center'] += normal * 0.02
        
        return super().add_entities_to_scene(scene, obj_materials, obj_vis_modes)
