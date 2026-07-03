from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("pumpkin_fall_pumpkin_splats_stool_intact")
class PumpkinFallPumpkinSplitsStoolIntact(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def set_simulation_bounds(self, all_obj_occupied_lower_bound, all_obj_occupied_upper_bound):
        super().set_simulation_bounds(all_obj_occupied_lower_bound, all_obj_occupied_upper_bound)
        pad = torch.ones_like(self.all_obj_occupied_size) * 0.45
        self.simulation_lower_bound = self.simulation_lower_bound - pad
        self.simulation_upper_bound = self.simulation_upper_bound + pad

    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        # The source image already places the pumpkin above the stool.  The
        # generated version lifted it again, which pushed MPM particles outside
        # the solver bounds.  Keep the reconstructed layout and let gravity drive
        # the fall.
        return super().add_entities_to_scene(scene, obj_materials, obj_vis_modes)

    def custom_simulation(self, sid):
        # No custom forces needed; gravity handles the fall.
        # The MPM sand material handles the bursting/splatting.
        # The Rigid material handles the stool wobble (via physics engine).
        pass
