from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("mugfall")
class Mugfall(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        """
        The mug is already suspended in the image. We do not need to lift it.
        However, we ensure the ground plane logic respects the background depth.
        """
        # Ensure the ground plane is detected from the background (table) 
        # and not snapped to the floating mug.
        self.config['background_plane_position_mode'] = 'background_depth'
        self.config['background_plane_offset'] = 0.0
        
        return super().add_entities_to_scene(scene, obj_materials, obj_vis_modes)

    def detect_ground_plane(self, ground_plane):
        """
        Explicitly manage ground plane to ensure the table (background) 
        is used as the collision floor, not a plane generated at the mug's feet.
        """
        # The YAML config handles most of this, but we enforce the mode here 
        # to guarantee the table surface is reconstructed as the floor.
        self.config['background_plane_position_mode'] = 'background_depth'
        self.config['background_collision_roi'] = [0.0, 0.6, 1.0, 1.0] # Focus on bottom 40% for table
        return super().detect_ground_plane(ground_plane)

    def custom_simulation(self, sid):
        """
        Optional: Add a slight initial perturbation if the mug falls too perfectly vertically.
        For this case, standard gravity and collision physics should handle the tip-over 
        naturally due to mesh asymmetry or slight angle, but we can add a tiny torque 
        after impact if needed. For now, we let physics take over.
        """
        pass
