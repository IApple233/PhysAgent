from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("snowball_fence")
class SnowballFence(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    # def get_case_static_support_offset_gs(self, spec):
    #     support_name = str(spec.get('name', '')).lower()
    #     support_index = spec.get('object_index', None)
    #     if support_name == 'fence' or support_index == 0:
    #         return np.array([1.5, 0.0, 0.0], dtype=np.float64)
    #     return None

    def custom_simulation(self, sid):
        """
        Give the snowball an initial velocity toward the fence.
        """
        if sid == 0:
            snowball = self.all_objs[0]
            n_particles = snowball._n_particles
            batch_size = snowball.solver._B
            init_vel = np.zeros((batch_size, n_particles, 3), dtype=np.float32)
            init_vel[:, :, 0] = 4.5  # +X: toward the fence
            snowball.set_vel(snowball._sim.cur_substep_local, init_vel)
