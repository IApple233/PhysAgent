from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("snowball_fence_event1")
class SnowballFenceEvent1(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def create_force_fields(self):
        """
        Apply an initial horizontal force to simulate the throw/arc of the snowball.
        The snowball is the only dynamic object (mpm_sand), so the force applies to all particles.
        """
        dt = self.config['dt']
        
        @ti.func
        def force_func(pos, vel, t, i):
            acc = ti.Vector([0.0, 0.0, 0.0], dt=gs.ti_float)
            # Apply a horizontal acceleration for the first 0.15 seconds to
            # simulate the throw toward the fence. Taichi kernels cannot return
            # from inside a dynamic branch, so assign then return once.
            if t < 0.15:
                acc = ti.Vector([35.0, 0.0, 2.0], dt=gs.ti_float)
            return acc

        force_field = gs.force_fields.Custom(force_func)
        force_field.activate()
        self.scene.add_force_field(force_field=force_field)

    def get_case_static_support_offset_gs(self, spec):
        name = str(spec.get("name", spec.get("label", ""))).lower()
        if "fence" in name:
            return np.array([0.0, 0.0, -0.025], dtype=np.float64)
        return None
