from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("nailong_ring_toss")
class NailongRingToss(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Apply a horizontal force to the ring (obj 1) to guide it towards the monster (obj 0).
        This simulates the "aligns" part of the prompt.
        Gravity handles the "descends" and "drops" parts.
        """
        # Object indices based on all_object_points order:
        # 0: Monster
        # 1: Ring
        monster_idx = 0
        ring_idx = 1

        # Apply alignment force for the first ~2.5 seconds (50 steps * 0.05 dt)
        if sid < 50:
            # Get positions (pull to CPU numpy for math)
            monster_pos = self.all_objs[monster_idx].get_pos().cpu().numpy()
            ring_pos = self.all_objs[ring_idx].get_pos().cpu().numpy()

            # Calculate horizontal direction vector
            diff = monster_pos - ring_pos
            diff[2] = 0.0  # Zero out Z component to only apply horizontal force
            
            dist = np.linalg.norm(diff)
            
            # Only apply force if not yet aligned (threshold 5cm)
            if dist > 0.05:
                force_dir = diff / dist
                # Apply force. Strength 3.0 is sufficient to move the ring horizontally
                # while gravity pulls it down.
                strength = 3.0
                force = force_dir * strength
                
                # Apply force to the ring
                # Force array shape must be (1, 3)
                self.all_objs[ring_idx].solver.apply_links_external_force(
                    force=force.reshape(1, 3).astype(np.float32), 
                    links_idx=[self.all_objs[ring_idx].idx]
                )
