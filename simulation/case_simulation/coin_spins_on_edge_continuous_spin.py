from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("coin_spins_on_edge_continuous_spin")
class CoinSpinsOnEdgeContinuousSpin(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Keep the coin spinning with only APIs supported by the Genesis rigid body
        wrapper used elsewhere in this repo.
        """
        obj = self.all_objs[0]
        if sid == 0:
            init_qvel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 28.0], dtype=np.float32)
            obj.set_dofs_velocity(velocity=init_qvel, dofs_idx_local=np.arange(6))

        torque = np.array([[0.0, 0.0, 18.0]], dtype=np.float32)
        obj.solver.apply_links_external_torque(torque=torque, links_idx=[obj.idx])
