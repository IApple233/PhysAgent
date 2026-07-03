from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("coin_spins_on_edge")
class CoinSpinsOnEdge(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Apply initial torque to spin the coin around the vertical axis (Z).
        The coin is standing on its edge. Spinning on edge implies rotation around Z.
        """
        # Apply spin torque for the first 20 steps to get it going
        if sid < 20:
            # Torque around Z-axis (vertical)
            torque = np.array([[0.0, 0.0, 50.0]])
            self.all_objs[0].solver.apply_links_external_torque(torque=torque, links_idx=[self.all_objs[0].idx])
