from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np


@register_case("firework_underpowered_fall")
class FireworkUnderpoweredFall(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        obj = self.all_objs[0]

        if sid == 0:
            init_qvel = np.array(
                self.config.get("launch_velocity", [1.6, 0.0, 4.8, 0.0, 0.0, 0.35]),
                dtype=np.float32,
            )
            obj.set_dofs_velocity(
                velocity=init_qvel,
                dofs_idx_local=np.arange(6),
            )

        thrust_end = int(self.config.get("thrust_end_frame", 18))
        if sid <= thrust_end:
            weak_thrust = np.array(
                self.config.get("weak_thrust_force", [0.9, 0.0, 3.8]),
                dtype=np.float32,
            ).reshape(1, 3)
            obj.solver.apply_links_external_force(force=weak_thrust, links_idx=[obj.idx])
