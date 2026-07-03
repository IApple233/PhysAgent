from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("two_duck")
class TwoDuck(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Apply gentle push force to left duck (index 0) which then collides with right duck (index 1).
        Reduced force strength for realistic water drag simulation.
        """
        # Only apply push force in early simulation steps
        if sid <= 30:
            # Get current positions for collision targeting
            pos_left = self.all_objs[0].get_pos().cpu().numpy()
            pos_right = self.all_objs[1].get_pos().cpu().numpy()
            
            # Calculate direction from left duck to right duck
            force_direction = pos_right - pos_left
            force_norm = np.linalg.norm(force_direction)
            if force_norm > 0.001:
                force_direction = force_direction / force_norm
            else:
                force_direction = np.array([1.0, 0.0, 0.0])
            
            force_direction = force_direction.reshape(1, 3)
            
            # Reduced strength for gentle push with water resistance
            strength = 3.0
            force = force_direction * strength
            
            # Apply force to left duck (index 0)
            self.all_objs[0].solver.apply_links_external_force(
                force=force, 
                links_idx=[self.all_objs[0].idx]
            )
        
        # Apply small continuous drag force to both ducks to simulate water resistance
        if sid > 30:
            for i in range(2):
                vel = self.all_objs[i].get_vel().cpu().numpy()
                drag_force = -vel * 0.5  # Simple linear drag
                drag_force = drag_force.reshape(1, 3)
                self.all_objs[i].solver.apply_links_external_force(
                    force=drag_force,
                    links_idx=[self.all_objs[i].idx]
                )

    def fix_particles(self):
        """
        Not needed for rigid body ducks.
        """
        pass

    def create_force_fields(self):
        """
        Not needed - using direct force application in custom_simulation.
        """
        pass
        
    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        """
        Slightly lift ducks above water surface to ensure proper floating behavior.
        """
        # Small offset to ensure ducks start slightly above the water plane
        for i in range(len(self.all_obj_info)):
            self.all_obj_info[i]['center'][2] += 0.02
        
        return super().add_entities_to_scene(scene, obj_materials, obj_vis_modes)
        
    def detect_ground_plane(self, ground_plane):
        """
        Use reconstructed background for water surface plane.
        """
        self.config['background_plane_position_mode'] = 'background_depth'
        self.config['background_plane_offset'] = 0.0
        return super().detect_ground_plane(ground_plane)
