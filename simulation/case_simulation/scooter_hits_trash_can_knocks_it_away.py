from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("scooter_hits_trash_can_knocks_it_away")
class ScooterHitsTrashCan(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Implementation for rigid body forces/torques or one-shot initial
        linear/angular velocities based on sid.
        """
        # Apply initial velocity to the scooter at the very first step
        if sid == 0:
            scooter = self.all_objs[0]
            trash_can = self.all_objs[1]

            # Get positions to calculate direction
            pos_scooter = scooter.get_pos().cpu().numpy()
            pos_can = trash_can.get_pos().cpu().numpy()

            # Calculate direction vector from scooter to can
            direction = pos_can - pos_scooter
            
            # We want the scooter to move primarily along the road (X-Y plane)
            # Zero out Z to prevent launching into the air initially, 
            # relying on the collision to lift the can.
            direction[2] = 0.0
            
            dist = np.linalg.norm(direction)
            if dist > 1e-5:
                direction = direction / dist
            
            # Moderate scooter speed; the trash can gets an additional impulse
            # below so the scooter itself does not need to tumble violently.
            speed = 3.2
            velocity_vec = direction * speed
            
            # Set linear velocity [vx, vy, vz] and zero angular velocity [wx, wy, wz]
            # The scooter is a free body, so first 6 dofs are linear + angular
            init_qvel = np.array([
                velocity_vec[0], velocity_vec[1], velocity_vec[2],
                0.0, 0.0, 0.0
            ], dtype=np.float32)
            
            scooter.set_dofs_velocity(
                velocity=init_qvel,
                dofs_idx_local=np.arange(6),
            )

        if 12 <= sid < 36:
            trash_can = self.all_objs[1]
            force = np.array([[260.0, 0.0, 85.0]], dtype=np.float32)
            trash_can.solver.apply_links_external_force(force=force, links_idx=[trash_can.idx])

        if sid == 42:
            scooter = self.all_objs[0]
            stop_qvel = np.zeros(6, dtype=np.float32)
            scooter.set_dofs_velocity(velocity=stop_qvel, dofs_idx_local=np.arange(6))
