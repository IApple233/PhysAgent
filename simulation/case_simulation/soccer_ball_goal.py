from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("soccer_ball_goal")
class SoccerBallGoal(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Implementation for rigid body forces/torques or one-shot initial
        linear/angular velocities based on sid.
        Remember to use pos_B - pos_A for ALL collision targeting!
        For thrown/tossed/launched rigid bodies, prefer setting
        set_dofs_velocity(...) once at sid == 0 instead of simulating the
        launch with an unrealistically large force.
        """
        if sid == 0:
            obj = self.all_objs[0]
            # Kick forward (positive Y direction towards the goal) and roll (negative X rotation).
            # Assuming a standard soccer ball radius of ~0.11m.
            # Linear velocity ~12 m/s (fast kick).
            # Angular velocity w = v / r = 12 / 0.11 ≈ 109 rad/s.
            # Rotation around X-axis: negative value makes the top move forward (+Y).
            init_qvel = np.array([0.0, 12.0, 0.0, -110.0, 0.0, 0.0], dtype=np.float32)
            obj.set_dofs_velocity(
                velocity=init_qvel,
                dofs_idx_local=np.arange(6),
            )
        
    def fix_particles(self):
        """
        Implementation for pinning soft body/cloth particles.
        Remember: Use init_particles -> torch.where -> find_closest_particle -> fix_particle.
        DO NOT use get_pos() for particles.
        """
        pass

    def create_force_fields(self):
        """
        Implementation for Taichi-based continuous wind/particle forces.
        Remember: Must call force_field.activate() before adding to scene!
        """
        pass
        
    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        """
        Implementation for tweaking initial positions before building.
        Inside this method, self.all_objs does not exist yet.
        Use self.all_obj_info only, then return super().add_entities_to_scene(...).
        Must return super().add_entities_to_scene(...)
        """
        return super().add_entities_to_scene(scene, obj_materials, obj_vis_modes)
        
    def detect_ground_plane(self, ground_plane):
        """
        Override with 'pass' if the prompt explicitly specifies the object is
        suspended/hanging in mid-air, or set background_plane_offset /
        background_plane_position_mode and then delegate to super().
        Otherwise, omit this method entirely.
        """
        pass
