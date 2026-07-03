from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("scooter_collide_event1")
class ScooterCollideEvent1(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)
        self.has_set_initial_velocity = False

    def custom_simulation(self, sid):
        # Object 0: Scooter
        # Object 1: Trash Can
        
        scooter = self.all_objs[0]
        trash_can = self.all_objs[1]
        
        # Get positions (pull to CPU for numpy math)
        pos_scooter = scooter.get_pos().cpu().numpy()
        pos_can = trash_can.get_pos().cpu().numpy()
        
        # Calculate direction from scooter to can
        direction = pos_can - pos_scooter
        distance = np.linalg.norm(direction)
        if distance > 1e-5:
            direction = direction / distance
        else:
            direction = np.array([1.0, 0.0, 0.0]) # Fallback
            
        # Set initial velocity at the very start
        if sid == 0 and not self.has_set_initial_velocity:
            # Velocity towards the can. 
            # Distance is roughly 3-4 meters. 
            # Speed ~ 2.5 m/s to cover distance in ~1.5s, then brake.
            speed = 3.0 
            velocity = direction * speed
            
            # Set linear velocity (first 3 dofs)
            # Angular velocity can be small or zero to keep it upright-ish, 
            # but rigid body physics will handle the rest.
            init_qvel = np.array([velocity[0], velocity[1], velocity[2], 0.0, 0.0, 0.0], dtype=np.float32)
            scooter.set_dofs_velocity(velocity=init_qvel, dofs_idx_local=np.arange(6))
            self.has_set_initial_velocity = True
            
        # "Slows just before contact" logic
        # Apply braking force after some time/distance to simulate slowing down before impact
        # Assuming total sim is 81 frames * 0.05 dt = 4 seconds.
        # Contact should happen around frame 40-50?
        # Let's apply braking force starting at sid=30 (1.5 seconds in)
        if sid >= 30 and sid < 60: # Brake for a duration
            # Get current velocity to oppose it
            current_vel = scooter.get_dofs_velocity(dofs_idx_local=np.arange(3)).cpu().numpy()
            speed = np.linalg.norm(current_vel)
            
            if speed > 0.1: # Only brake if moving
                # Braking force magnitude
                brake_strength = 15.0 # Strong brake to slow down quickly
                brake_force = -current_vel / speed * brake_strength
                brake_force = brake_force.reshape(1, 3)
                
                scooter.solver.apply_links_external_force(force=brake_force, links_idx=[scooter.idx])
        
        # Ensure trash can is heavy/stable (handled by high rho in YAML, but we can add damping if needed)
        # The prompt says "trash can only rattles". High mass + friction should do it.
        # We can add a small damping force to the trash can to stop it from sliding too much if hit hard
        if sid > 0:
            can_vel = trash_can.get_dofs_velocity(dofs_idx_local=np.arange(3)).cpu().numpy()
            can_speed = np.linalg.norm(can_vel)
            if can_speed > 0.05:
                # Damping to stop rattling/sliding
                damping = -can_vel * 20.0 
                damping = damping.reshape(1, 3)
                trash_can.solver.apply_links_external_force(force=damping, links_idx=[trash_can.idx])

    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        # No specific position tweaks needed, reconstruction should place them on the road.
        return super().add_entities_to_scene(scene, obj_materials, obj_vis_modes)
