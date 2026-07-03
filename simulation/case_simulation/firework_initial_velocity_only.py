from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np


@register_case("firework_initial_velocity_only")
class FireworkInitialVelocityOnly(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        obj = self.all_objs[0]

        if sid == 0:
            init_qvel = np.array(
                self.config.get("launch_velocity", [2.75, 0.0, 4.5, 0.0, 0.0, 0.0]),
                dtype=np.float32,
            )
            obj.set_dofs_velocity(
                velocity=init_qvel,
                dofs_idx_local=np.arange(6),
            )
