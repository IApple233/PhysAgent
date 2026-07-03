from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("bedminten")
class Bedminten(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)
        # Store the initial position of the net (Object 1) to pin it
        # all_obj_info[1] corresponds to the net
        self.net_initial_pos = self.all_obj_info[1]['center'].cpu().numpy()

    def custom_simulation(self, sid):
        """
        Simulate the tennis ball hitting the net.
        Object 0: Tennis Ball (Rigid)
        Object 1: Tennis Net (Rigid, but pinned via restoring force)
        """
        # Apply force to the ball (Object 0) to push it towards the net for the first few steps
        if sid < 15:
            pos_ball = self.all_objs[0].get_pos().cpu().numpy()
            # Target is the net's initial position
            target_pos = self.net_initial_pos
            
            direction = target_pos - pos_ball
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction = direction / norm
            
            # Apply a strong force to simulate a hit/serve towards the net
            # Strength needs to be sufficient to cover the distance
            strength = 300.0 
            force_ball = (direction * strength).reshape(1, 3)
            self.all_objs[0].solver.apply_links_external_force(force=force_ball, links_idx=[self.all_objs[0].idx])

        # Pin the net (Object 1) by applying a restoring force
        # This keeps the net static while allowing collision
        pos_net_current = self.all_objs[1].get_pos().cpu().numpy()
        diff = self.net_initial_pos - pos_net_current
        
        # High stiffness to keep the net effectively fixed
        stiffness = 100000.0
        force_net = (diff * stiffness).reshape(1, 3)
        self.all_objs[1].solver.apply_links_external_force(force=force_net, links_idx=[self.all_objs[1].idx])
        
    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        """
        No initial position tweaks needed as the ball is already in the air.
        """
        return super().add_entities_to_scene(scene, obj_materials, obj_vis_modes)
