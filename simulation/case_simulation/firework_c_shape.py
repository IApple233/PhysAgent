from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("firework_c_shape")
class FireworkCShape(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Implementation for rigid body forces/torques or one-shot initial
        linear/angular velocities based on sid.
        """
        obj = self.all_objs[0]
        
        # Initial launch: Diagonally up and right
        if sid == 0:
            # vx=3.0 (Right), vy=0.0, vz=8.0 (Up)
            # Angular velocity: small spin for realism
            init_qvel = np.array([3.0, 0.0, 8.0, 0.0, 0.0, 2.0], dtype=np.float32)
            obj.set_dofs_velocity(
                velocity=init_qvel,
                dofs_idx_local=np.arange(6),
            )
        
        # Mid-flight air disturbance: Push left
        # Applied from sid=40 to sid=60 (approx 0.8s to 1.2s)
        if 40 <= sid <= 60:
            # Force to the left (-X direction)
            force_mag = 25.0
            force_dir = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
            force = force_dir * force_mag
            force = force.reshape(1, 3)
            obj.solver.apply_links_external_force(force=force, links_idx=[obj.idx])
