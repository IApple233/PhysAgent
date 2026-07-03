from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("nailong_event1")
class NailongEvent1(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        # Object 0 is the yellow toy and remains fixed. Object 1 is the blue
        # ring; shift it slightly toward the camera and up so it misses the toy
        # in front, then lands on the tabletop under gravity.
        if len(self.all_obj_info) > 1:
            lift = torch.tensor(
                [0.0, -0.10, 0.08],
                device=self.device,
                dtype=self.all_obj_info[1]["center"].dtype,
            )
            for key in ("min", "max", "center", "vertices"):
                value = self.all_obj_info[1].get(key)
                if isinstance(value, torch.Tensor):
                    self.all_obj_info[1][key] = value + lift
        return super().add_entities_to_scene(scene, obj_materials, obj_vis_modes)

    def custom_simulation(self, sid):
        if sid == 0:
            ring_obj = self.all_objs[1]
            init_qvel = np.array([
                0.0,      # vx
                -0.8,     # vy: gentle slide toward camera after contact
                -1.2,     # vz: modest downward speed, avoids tunneling
                0.0, 0.0, 0.2
            ], dtype=np.float32)
            ring_obj.set_dofs_velocity(
                velocity=init_qvel,
                dofs_idx_local=np.arange(6),
            )
