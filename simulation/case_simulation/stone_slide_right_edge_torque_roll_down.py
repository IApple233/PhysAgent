from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("stone_slide_right_edge_torque_roll_down")
class StoneSlideRightEdgeTorqueRollDown(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Apply an initial torque to lift the stone's upper-right edge,
        causing it to tip over its lower-left edge.
        The torque is applied around the Y-axis (depth), which lifts the +X side (Right/Uphill).
        This makes the stone stand up perpendicular to the slope (leaning downhill),
        causing it to fall/roll down the slope (Left/-X) due to gravity.
        """
        if sid < 30:
            # Torque around Y-axis to lift the right side (uphill side)
            # Magnitude 15.0 to overcome gravity and friction initially
            torque = np.array([0.0, 15.0, 0.0], dtype=np.float32)
            torque = torque.reshape(1, 3)
            
            # Apply to the stone (object index 0)
            self.all_objs[0].solver.apply_links_external_torque(
                torque=torque, 
                links_idx=[self.all_objs[0].idx]
            )
        elif sid < 70:
            force = np.array([[-18.0, 0.0, -4.0]], dtype=np.float32).reshape(1, 3)
            self.all_objs[0].solver.apply_links_external_force(
                force=force,
                links_idx=[self.all_objs[0].idx],
            )
