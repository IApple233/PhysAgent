from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("teaser_agent_ag")
class TeaserAgentAg(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Implementation for rigid body forces/torques or one-shot initial
        linear/angular velocities based on sid.
        """
        if sid == 0:
            # Cat (Index 0) starts at bottom left, needs to move Up/Right to trace 'A'
            # Velocity: [vx, vy, vz, wx, wy, wz]
            # vx > 0 (Right), vy < 0 (Forward/Up in image)
            cat = self.all_objs[0]
            cat_vel = np.array([0.2, -0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            cat.set_dofs_velocity(velocity=cat_vel, dofs_idx_local=np.arange(6))

            # Duck (Index 1) starts at top right, needs to move Down/Left to trace 'g'
            # Velocity: [vx, vy, vz, wx, wy, wz]
            # vx < 0 (Left), vy > 0 (Backward/Down in image)
            duck = self.all_objs[1]
            duck_vel = np.array([-0.2, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            duck.set_dofs_velocity(velocity=duck_vel, dofs_idx_local=np.arange(6))
