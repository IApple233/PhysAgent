from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("foam_block_acrylic_ramp_base_pulled_right_wobble")
class FoamBlockAcrylicRampBasePulledRightWobble(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Pull the acrylic ramp (Object 1) to the right.
        The foam block (Object 0, pbd_elastic) will be dragged by friction.
        """
        if sid == 0:
            foam = self.all_objs[0]
            if hasattr(foam, "n_particles"):
                n_particles = foam.n_particles
                init_vel = np.zeros((n_particles, 3), dtype=np.float32)
                init_vel[:, 0] = 0.28
                init_vel[:, 2] = 0.08
                foam.solver._kernel_set_particles_vel(
                    foam._sim.cur_substep_local,
                    foam._particle_start,
                    n_particles,
                    init_vel,
                )

        ramp_idx = 1
        
        # Apply force to the ramp to pull it to the right (+X direction)
        # We apply this force for the first part of the simulation to initiate the movement.
        if sid < 60:
            # Get ramp position to ensure force is applied correctly (though force is global)
            # We use a constant force to pull the ramp.
            # The ramp is heavy (rho 1200), so we need a significant force.
            force_magnitude = 260.0
            force_direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            force = force_direction * force_magnitude
            force = force.reshape(1, 3)
            
            # Apply force to the ramp
            self.all_objs[ramp_idx].solver.apply_links_external_force(
                force=force, 
                links_idx=[self.all_objs[ramp_idx].idx]
            )

    def create_force_fields(self):
        center = self.all_obj_info[0]['center'].detach().cpu().numpy().astype(np.float32)
        center_x = float(center[0])
        center_y = float(center[1])
        center_z = float(center[2])

        @ti.func
        def force_func(pos, vel, t, i):
            phase = 5.5 * t + 3.0 * pos[0]
            buoyancy = 5.8
            drift = ti.Vector([
                0.25 + 0.18 * ti.sin(phase),
                0.10 * ti.sin(phase + 1.3),
                buoyancy + 0.22 * ti.cos(phase + pos[1]),
            ], dt=gs.ti_float)
            r = ti.Vector([pos[0] - center_x, pos[1] - center_y, pos[2] - center_z], dt=gs.ti_float)
            wobble = ti.Vector([-r[2], 0.15 * r[0], 0.2 * r[0]], dt=gs.ti_float) * (0.7 * ti.sin(4.0 * t))
            return drift + wobble

        force_field = gs.force_fields.Custom(force_func)
        force_field.activate()
        self.scene.add_force_field(force_field=force_field)
            
    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        """
        Adjust initial positions if necessary.
        The foam block is on the ramp. The ramp is on the table.
        We might need to lift the ramp slightly to ensure it sits on the plane correctly.
        """
        # Lift the ramp slightly to ensure it rests on the ground plane
        # Object 1 is the ramp.
        # self.all_obj_info[1]['center'][2] += 0.01
        
        return super().add_entities_to_scene(scene, obj_materials, obj_vis_modes)
