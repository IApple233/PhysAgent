from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("bird_gap_flight")
class BirdGapFlight(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        obj = self.all_objs[0]
        
        # Initial launch: Forward velocity (X) and slight upward velocity (Z)
        # The bird starts slightly below the gap center, so it needs to rise.
        if sid == 0:
            # vx=2.0 (forward), vz=1.5 (upward start)
            init_qvel = np.array([2.0, 0.0, 1.5, 0.0, 0.0, 0.0], dtype=np.float32)
            obj.set_dofs_velocity(velocity=init_qvel, dofs_idx_local=np.arange(6))
            
        # Flapping logic: Apply periodic upward force to simulate flapping and bobbing
        # Flap every 15 frames (0.15s at dt=0.01) -> ~6.6 Hz, reasonable for a small bird
        if sid % 15 == 0 and sid > 0:
            # Upward force to counteract gravity (-9.8) and provide lift
            # 25.0 is strong enough to create a visible upward impulse
            flap_force = np.array([[0.0, 0.0, 25.0]], dtype=np.float32)
            obj.solver.apply_links_external_force(force=flap_force, links_idx=[obj.idx])
            
        # Continuous small forward force to maintain speed and overcome air resistance/damping
        forward_force = np.array([[1.5, 0.0, 0.0]], dtype=np.float32)
        obj.solver.apply_links_external_force(force=forward_force, links_idx=[obj.idx])
