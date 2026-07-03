from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np


@register_case("knife_above_cutting_board_handle_fixed_clockwise_rotation")
class KnifeAboveCuttingBoardHandleFixedClockwiseRotation(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        knife = self.all_objs[0]
        if sid == 0:
            # Spin in place rather than throwing the knife through the board.
            init_qvel = np.array([0.0, 0.0, 0.0, 0.0, -8.0, 0.0], dtype=np.float32)
            knife.set_dofs_velocity(velocity=init_qvel, dofs_idx_local=np.arange(6))

        # A small lift counteracts gravity so the visual remains a floating
        # pivot-like rotation above the board.
        lift = np.array([[0.0, 0.0, 9.0]], dtype=np.float32)
        knife.solver.apply_links_external_force(force=lift, links_idx=[knife.idx])

        clockwise_torque = np.array([[0.0, -14.0, 0.0]], dtype=np.float32)
        knife.solver.apply_links_external_torque(torque=clockwise_torque, links_idx=[knife.idx])
