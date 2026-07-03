from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np


@register_case("soccer_ball_goal_event1")
class SoccerBallGoalEvent1(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def get_case_static_support_offset_gs(self, spec):
        name = str(spec.get("name", spec.get("label", ""))).lower()
        if "goal" in name:
            return np.array([0.0, 0.0, -0.025], dtype=np.float64)
        return None

    def custom_simulation(self, sid):
        if sid == 0:
            # Roll toward the left post with enough speed to rebound across the goal mouth.
            init_qvel = np.array([-0.7, 9.0, 0.0, -82.0, 0.0, 0.0], dtype=np.float32)
            self.all_objs[0].set_dofs_velocity(
                velocity=init_qvel,
                dofs_idx_local=np.arange(6),
            )
