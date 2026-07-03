from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("multi_objects")
class MultiObjects(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Apply a forward force to push the blocks into the holes.
        The force is applied for the first 15 steps to initiate the slide.
        Direction is -Y (towards the holes/camera).
        """
        # Force parameters
        force_strength = 4.0
        force_direction = np.array([0, -1, 0], dtype=np.float32)
        force_direction = force_direction / np.linalg.norm(force_direction)
        force_vec = force_direction * force_strength
        force_vec = force_vec.reshape(1, 3)

        # Apply force to both objects for the first 15 steps
        if sid < 15:
            for i in range(len(self.all_objs)):
                obj = self.all_objs[i]
                obj.solver.apply_links_external_force(force=force_vec, links_idx=[obj.idx])

    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        """
        No initial position tweaks needed as objects are already aligned with holes.
        """
        return super().add_entities_to_scene(scene, obj_materials, obj_vis_modes)
