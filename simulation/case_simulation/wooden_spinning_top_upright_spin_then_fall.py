from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np


@register_case("wooden_spinning_top_upright_spin_then_fall")
class WoodenSpinningTopUprightSpinThenFall(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        top = self.all_objs[0]
        if sid == 0:
            init_qvel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 22.0], dtype=np.float32)
            top.set_dofs_velocity(velocity=init_qvel, dofs_idx_local=np.arange(6))

        if sid < 35:
            spin_torque = np.array([[0.0, 0.0, 12.0]], dtype=np.float32)
            top.solver.apply_links_external_torque(torque=spin_torque, links_idx=[top.idx])
        elif sid < 70:
            fall_torque = np.array([[4.5, 0.0, 1.5]], dtype=np.float32)
            top.solver.apply_links_external_torque(torque=fall_torque, links_idx=[top.idx])
