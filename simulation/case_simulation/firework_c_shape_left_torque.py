from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np


@register_case("firework_c_shape_left_torque")
class FireworkCShapeLeftTorque(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        obj = self.all_objs[0]

        if sid == 0:
            init_qvel = np.array(
                self.config.get("launch_velocity", [0.34, 0.0, 0.8, 0.0, 0.0, 0.08]),
                dtype=np.float32,
            )
            obj.set_dofs_velocity(
                velocity=init_qvel,
                dofs_idx_local=np.arange(6),
            )

        turn_start = int(self.config.get("turn_start_frame", 34))
        turn_end = int(self.config.get("turn_end_frame", 63))
        if turn_start <= sid <= turn_end:
            left_force = np.array(
                self.config.get("left_turn_force", [-0.002, 0.0, 0.08]),
                dtype=np.float32,
            ).reshape(1, 3)
            left_torque = np.array(
                self.config.get("left_turn_torque", [0.0, -0.01, 0.0]),
                dtype=np.float32,
            ).reshape(1, 3)
            obj.solver.apply_links_external_force(force=left_force, links_idx=[obj.idx])
            obj.solver.apply_links_external_torque(torque=left_torque, links_idx=[obj.idx])

        pull_start = int(self.config.get("sustained_pull_start_frame", turn_start))
        pull_end = int(self.config.get("sustained_pull_end_frame", 80))
        if pull_start <= sid <= pull_end:
            pull_force = np.array(
                self.config.get("sustained_pull_force", [-0.0173, 0.0, 0.01]),
                dtype=np.float32,
            ).reshape(1, 3)
            obj.solver.apply_links_external_force(force=pull_force, links_idx=[obj.idx])
