from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("pinball_collision2")
class PinballCollision2(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Apply a small initial force to the pinball to ensure it starts rolling down the lane.
        The lane is reconstructed with a tilt (or gravity is aligned), so a small nudge helps.
        Object 0 is the pinball.
        """
        if sid < 10:
            # Apply a force in the direction of the lane (roughly +X in local lane coordinates)
            # Since the reconstruction aligns the lane, we push along the lane's primary axis.
            # Assuming the lane runs somewhat diagonally or along X in the reconstructed space.
            # We apply a force that encourages movement towards the bumpers.
            force_magnitude = 5.0
            # Direction: Towards the bumpers (which are generally 'forward' and 'right' in the image)
            # In 3D space, this is likely +X and +Y depending on alignment. 
            # A simple push along X is often sufficient for lane-like structures.
            force_direction = np.array([1.0, 0.5, 0.0], dtype=np.float64)
            force_direction = force_direction / np.linalg.norm(force_direction)
            force = force_direction * force_magnitude
            force = force.reshape(1, 3)
            
            # Apply to the pinball (object 0)
            self.all_objs[0].solver.apply_links_external_force(force=force, links_idx=[self.all_objs[0].idx])

    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        """
        Ensure the pinball is slightly above the lane surface to avoid initial penetration issues.
        """
        # Lift the pinball (object 0) slightly along the Z axis (or normal)
        # Since we don't have the normal here easily without scene built, we lift in Z.
        # The reconstruction might place it exactly on the surface.
        if len(self.all_obj_info) > 0:
            # Lift by 0.01 meters
            self.all_obj_info[0]['center'][2] += 0.02
            
        return super().add_entities_to_scene(scene, obj_materials, obj_vis_modes)
