from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("stone_slide_event1")
class StoneSlideEvent1(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        # Lift the stone slightly to ensure it's not intersecting the sand initially
        # and to give it a tiny bit of potential energy to start the slide.
        # Object 0 is the stone.
        if len(self.all_obj_info) > 0:
            # Lift along Z axis (up)
            self.all_obj_info[0]['center'][2] += 0.02
            
        return super().add_entities_to_scene(scene, obj_materials, obj_vis_modes)

    def get_case_static_support_offset_gs(self, spec):
        name = str(spec.get("name", spec.get("label", ""))).lower()
        if "sand" in name or "dune" in name:
            return np.array([0.0, 0.0, -0.035], dtype=np.float64)
        return None

    def custom_simulation(self, sid):
        # Apply a small nudge to the stone to initiate movement if gravity isn't enough
        # or to ensure it moves in the desired direction.
        # Object 0 is the stone (rigid).
        if sid == 0:
            stone_obj = self.all_objs[0]
            # Apply a small force forward and slightly down to simulate a nudge or initial slide
            # Force direction: roughly forward (Y) and down (-Z) relative to the slope
            # Assuming the slope goes somewhat forward.
            force_dir = np.array([-1.0, 0.15, -0.15], dtype=np.float32)
            force_dir = force_dir / np.linalg.norm(force_dir)
            force_magnitude = 3.0  # Small force; sand friction should stop it naturally.
            
            force_vector = (force_dir * force_magnitude).reshape(1, 3)
            stone_obj.solver.apply_links_external_force(force=force_vector, links_idx=[stone_obj.idx])
