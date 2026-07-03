from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("foam_block_acrylic_ramp")
class FoamBlockAcrylicRamp(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        lift_offset = float(self.config.get('foam_initial_lift_offset', 0.12))
        for obj_info in self.all_obj_info:
            obj_info['center'][2] += lift_offset
        return super().add_entities_to_scene(scene, obj_materials, obj_vis_modes)

    def create_force_fields(self):
        """
        Keep the foam block airborne with a buoyancy-like vertical acceleration.
        The horizontal motion is initialized once in custom_simulation(); this
        field adds vertical lift plus small zero-mean drift disturbances.
        """
        buoyancy_accel = float(self.config.get('foam_buoyancy_accel', 7.0))
        turbulence_accel = float(self.config.get('foam_turbulence_accel', 0.35))
        turbulence_freq = float(self.config.get('foam_turbulence_freq', 7.0))
        turbulence_spatial = float(self.config.get('foam_turbulence_spatial', 4.0))
        torque_accel = float(self.config.get('foam_torque_accel', 1.2))
        torque_freq = float(self.config.get('foam_torque_freq', 4.5))
        center = self.all_obj_info[0]['center'].detach().cpu().numpy().astype(np.float32)
        center_x = float(center[0])
        center_y = float(center[1])
        center_z = float(center[2])

        @ti.func
        def force_func(pos, vel, t, i):
            phase = turbulence_freq * t + turbulence_spatial * pos[0]
            drift_x = 0.4 * turbulence_accel * ti.sin(phase + 1.7 * pos[2])
            drift_y = turbulence_accel * ti.sin(0.7 * phase + 2.3 * pos[1])
            drift_z = 0.5 * turbulence_accel * ti.cos(0.9 * phase)
            r = ti.Vector([pos[0] - center_x, pos[1] - center_y, pos[2] - center_z], dt=gs.ti_float)
            swirl = torque_accel * ti.sin(torque_freq * t)
            torque_like = ti.Vector([-r[2], 0.25 * r[0], r[0]], dt=gs.ti_float) * swirl
            return ti.Vector([drift_x, drift_y, buoyancy_accel + drift_z], dt=gs.ti_float) + torque_like

        force_field = gs.force_fields.Custom(force_func)
        force_field.activate()
        self.scene.add_force_field(force_field=force_field)

    def custom_simulation(self, sid):
        if sid != 0:
            return

        foam = self.all_objs[0]
        n_particles = foam.n_particles
        init_vel = np.zeros((n_particles, 3), dtype=np.float32)
        init_vel[:, 0] = float(self.config.get('foam_initial_velocity_x', 0.7))
        foam.solver._kernel_set_particles_vel(
            foam._sim.cur_substep_local,
            foam._particle_start,
            n_particles,
            init_vel,
        )
