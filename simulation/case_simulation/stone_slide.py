from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("stone_slide")
class StoneSlide(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        # The stone needs to be placed higher up the dune to slide down.
        # Based on the image, the dune rises towards the right (positive X).
        # We move the stone slightly up the slope (positive X) and lift it slightly (positive Z)
        # to ensure it's "released" and not initially penetrating the sand mesh.
        # Note: Since background_collision_sets_gravity is false, Z is global up.
        # We assume the reconstruction aligns X roughly with image X.
        
        # Move up the slope (Right)
        # self.all_obj_info[0]['center'][0] += 0.15
        # Lift slightly to avoid initial intersection
        self.all_obj_info[0]['center'][2] += 0.09
        
        return super().add_entities_to_scene(scene, obj_materials, obj_vis_modes)

    def custom_simulation(self, sid):
        # Apply a small nudge force to ensure sliding starts if friction is high or slope is shallow.
        # Force direction: Down the slope (Left, -X).
        # We apply this only at the beginning.
        if sid < 10:
            # Force vector: [-1.0, 0.0, 0.0] (Left/Down-slope)
            # We need to normalize and scale.
            force_dir = np.array([-1.0, 0.0, -1.0], dtype=np.float64)
            force_mag = 5.0 # Small nudge
            force = (force_dir / (np.linalg.norm(force_dir) + 1e-8)) * force_mag
            force = force.reshape(1, 3)
            
            # Apply to the stone (index 0)
            self.all_objs[0].solver.apply_links_external_force(force=force, links_idx=[self.all_objs[0].idx])
