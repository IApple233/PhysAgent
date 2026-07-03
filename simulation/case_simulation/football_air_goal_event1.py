from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np


@register_case("football_air_goal_event1")
class FootballAirGoalEvent1(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def get_case_static_support_offset_gs(self, spec):
        name = str(spec.get("name", spec.get("label", ""))).lower()
        if "goal" in name:
            return np.array([0.0, 0.0, -0.025], dtype=np.float64)
        return None

    def custom_simulation(self, sid):
        if sid == 0:
            # High forward arc aimed at the front underside of the crossbar.
            init_qvel = np.array([0.2, 11.5, 8.8, -6.0, 0.0, 0.0], dtype=np.float32)
            self.all_objs[0].set_dofs_velocity(
                velocity=init_qvel,
                dofs_idx_local=np.arange(6),
            )
