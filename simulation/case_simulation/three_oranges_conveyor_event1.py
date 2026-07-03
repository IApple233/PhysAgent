from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np


@register_case("three_oranges_conveyor_event1")
class ThreeOrangesConveyorEvent1(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        objs = super().add_entities_to_scene(scene, obj_materials, obj_vis_modes)
        # Keep the oranges just above the reconstructed belt/floor to avoid
        # initial interpenetration with thin static support meshes.
        for obj_idx, obj in enumerate(objs[: len(self.all_obj_info)]):
            z_lift = 0.035 if obj_idx < 2 else 0.045
            offset = np.array([0.0, 0.0, z_lift], dtype=np.float64)
            center = self.all_obj_info[obj_idx]["center"].detach().cpu().numpy().astype(np.float64)
            if self._set_dynamic_entity_center(obj, center + offset, obj_idx):
                self._update_dynamic_object_info_after_shift(obj_idx, offset)
        return objs

    def get_case_static_support_offset_gs(self, spec):
        name = str(spec.get("name", spec.get("label", ""))).lower()
        if "conveyor" in name or "belt" in name:
            return np.array([0.0, 0.0, -0.035], dtype=np.float64)
        return None

    def custom_simulation(self, sid):
        if sid == 0 and len(self.all_objs) >= 3:
            left = self.all_objs[0]
            right = self.all_objs[1]
            lower = self.all_objs[2]

            pos_left = left.get_pos().cpu().numpy()
            pos_right = right.get_pos().cpu().numpy()
            pos_lower = lower.get_pos().cpu().numpy()

            # Left orange drops first toward the lower orange.
            dir_left = pos_lower - pos_left
            dir_left[2] = -abs(dir_left[2]) - 0.2
            dir_left = dir_left / (np.linalg.norm(dir_left) + 1e-8)
            left.set_dofs_velocity(
                velocity=np.array([*(dir_left * 1.6), 0.0, 0.0, 0.0], dtype=np.float32),
                dofs_idx_local=np.arange(6),
            )

            # The second upper orange is delayed; a tiny backward velocity keeps
            # it on the belt initially.
            dir_back = pos_right - pos_left
            dir_back[2] = 0.0
            dir_back = dir_back / (np.linalg.norm(dir_back) + 1e-8)
            right.set_dofs_velocity(
                velocity=np.array([*(dir_back * 0.15), 0.0, 0.0, 0.0], dtype=np.float32),
                dofs_idx_local=np.arange(6),
            )

        if 25 <= sid < 60 and len(self.all_objs) >= 2:
            # Then the second orange rolls off after a visible delay.
            force = np.array([[16.0, -2.0, -1.0]], dtype=np.float32)
            self.all_objs[1].solver.apply_links_external_force(force=force, links_idx=[self.all_objs[1].idx])
