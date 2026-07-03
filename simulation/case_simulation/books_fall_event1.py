from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("books_fall_event1")
class BooksFallEvent1(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Simulate the bottom book (index 3, green) sliding out rapidly to the right.
        This removes support for the upper books (indices 0, 1, 2), causing them to drop.
        Extended simulation duration (150 frames) allows time for the drop and settle.
        """
        # Object indices based on all_object_points order:
        # 0: Blue (Top)
        # 1: Beige
        # 2: Red
        # 3: Green (Bottom)
        
        bottom_book_idx = 3
        bottom_book = self.all_objs[bottom_book_idx]
        
        # Apply a strong force to the right (positive X) to slide the bottom book out.
        # We apply this force for the first part of the simulation to initiate the slide.
        # 150 frames * 0.01 dt = 1.5 seconds total.
        # Pulling for ~0.2 seconds (20 frames) should be sufficient to clear the stack.
        if sid < 20:
            # Force magnitude needs to overcome friction and accelerate the book.
            # Mass approx 2kg (estimated). Force ~ 500N for rapid acceleration.
            force_magnitude = 600.0
            force_dir = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            force = force_dir * force_magnitude
            force = force.reshape(1, 3)
            
            # Apply force to the bottom book
            bottom_book.solver.apply_links_external_force(force=force, links_idx=[bottom_book.idx])
            
            # To encourage "fanning outward" and chaotic tumbling as requested,
            # we can apply a tiny perturbation force to the book just above (Red book, idx 2)
            # in the opposite direction or slightly sideways to break symmetry.
            # This helps prevent a perfect vertical drop if friction is too uniform.
            if sid < 10:
                red_book = self.all_objs[2]
                perturbation_force = np.array([-50.0, 20.0, 0.0], dtype=np.float32) # Slight left and forward/back
                perturbation_force = perturbation_force.reshape(1, 3)
                red_book.solver.apply_links_external_force(force=perturbation_force, links_idx=[red_book.idx])

    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        """
        No initial position tweaks needed. Books are stacked on the table.
        """
        return super().add_entities_to_scene(scene, obj_materials, obj_vis_modes)

    def get_case_static_support_offset_gs(self, spec):
        name = str(spec.get("name", "")).lower()
        if "table" in name:
            return np.array([0.0, 0.0, -0.035], dtype=np.float64)
        return None
