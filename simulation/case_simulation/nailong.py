from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import genesis as gs

@register_case("nailong")
class Nailong(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        if self.all_obj_info:
            lift = float(self.config.get("monster_initial_lift", 0.04))
            shift = torch.tensor([0.0, 0.0, lift], device=self.device, dtype=self.all_obj_info[0]['center'].dtype)
            for key in ('min', 'max', 'center', 'vertices'):
                if key in self.all_obj_info[0] and isinstance(self.all_obj_info[0][key], torch.Tensor):
                    self.all_obj_info[0][key] = self.all_obj_info[0][key] + shift
        return super().add_entities_to_scene(scene, obj_materials, obj_vis_modes)

    def custom_simulation(self, sid):
        """
        Give the ring one initial velocity toward the monster, then let physics run.
        """
        if sid == 0:
            monster = self.all_objs[0]
            ring = self.all_objs[1]

            pos_monster = monster.get_pos().cpu().numpy()
            pos_ring = ring.get_pos().cpu().numpy()
            horizontal_delta = pos_monster - pos_ring
            horizontal_delta[2] = 0.0
            self._ring_initial_horizontal_delta = horizontal_delta.astype(np.float32)

            horizontal_gain = float(self.config.get("ring_initial_horizontal_gain", 1.25))
            vel_linear = np.array([
                horizontal_delta[0] * horizontal_gain,
                horizontal_delta[1] * horizontal_gain,
                -float(self.config.get("ring_initial_downward_speed", 0.75)),
            ], dtype=np.float32)
            vel_angular = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            init_qvel = np.concatenate([vel_linear, vel_angular]).astype(np.float32)
            ring.set_dofs_velocity(
                velocity=init_qvel,
                dofs_idx_local=np.arange(6)
            )
