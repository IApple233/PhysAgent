from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("football_air_goal")
class FootballAirGoal(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Apply initial kick velocity to the soccer ball at the start of simulation.
        """
        if sid == 0:
            obj = self.all_objs[0]
            # Kick upward (Z) and forward (Y) towards the goal.
            # Slight rightward (X) component to aim for center of goal from left position.
            # Add some topspin (rotation around X) for a realistic arc.
            # vx, vy, vz, wx, wy, wz
            init_qvel = np.array([2.0, 12.0, 8.0, -5.0, 0.0, 0.0], dtype=np.float32)
            obj.set_dofs_velocity(
                velocity=init_qvel,
                dofs_idx_local=np.arange(6),
            )
