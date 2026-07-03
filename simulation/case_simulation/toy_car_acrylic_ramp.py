from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("toy_car_acrylic_ramp")
class ToyCarAcrylicRamp(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def _car_motion_direction(self):
        return self._configured_tangent_direction(
            self.config.get('motion_image_direction', [1.0, 0.0]),
            fallback_gs=self.config.get('motion_direction_gs', [1.0, 0.0, 0.0]),
        )

    def _turn_force_direction(self):
        return self._configured_tangent_direction(
            self.config.get('turn_force_image_direction', [1.0, 0.0]),
            fallback_gs=self.config.get('turn_force_direction_gs', [1.0, 0.0, 0.0]),
        )

    def _configured_tangent_direction(self, image_direction, fallback_gs):
        direct_gs = bool(self.config.get('toy_car_use_direct_gs_directions', False))
        if direct_gs:
            return self._normalize_vector(fallback_gs, default=[1.0, 0.0, 0.0])

        normal = self._support_reference_normal()
        tangent_u, tangent_v = self._orthonormal_tangent_basis(normal)
        image_direction = np.asarray(image_direction, dtype=np.float64)
        if image_direction.shape != (2,) or not np.isfinite(image_direction).all():
            return self._normalize_vector(fallback_gs, default=[1.0, 0.0, 0.0])

        direction = image_direction[0] * tangent_v - image_direction[1] * tangent_u
        return self._normalize_vector(direction, default=fallback_gs)

    def _car_roll_axis(self, motion_direction):
        normal = self._support_reference_normal()
        roll_axis = np.cross(motion_direction, normal)
        return self._normalize_vector(roll_axis, default=[0.0, -1.0, 0.0])

    def _turn_yaw_axis(self):
        return self._support_reference_normal()

    def get_case_static_support_offset_gs(self, spec):
        support_name = str(spec.get('name', '')).lower()
        support_index = spec.get('object_index', None)
        if support_name in {'conveyor', 'conveyor_belt'} or support_index == 0:
            return np.array([0.0, 0.0, -0.005], dtype=np.float64)

        return None
    
    def custom_simulation(self, sid):
        """
        Apply a gentle push to the right (positive X) to start the car rolling down the ramp.
        Also apply a small torque to simulate wheel rotation.
        """
        if len(self.all_objs) == 0:
            return

        car_obj = self.all_objs[0]

        initial_push_end = int(self.config.get('initial_push_end_frame', 5))
        if sid < initial_push_end:
            # Get car position (index 0)
            # We don't strictly need position for a fixed direction push, 
            # but following the rule to use dynamic targeting if interacting.
            # Here, it's just an initial push.

            force_dir = self._car_motion_direction()
            force_dir = force_dir.reshape(1, 3)
            
            force_strength = float(self.config.get('initial_push_strength', 50.0))
            force = force_dir * force_strength
            
            car_obj.solver.apply_links_external_force(force=force, links_idx=[car_obj.idx])
            
            # Torque to simulate rolling (around Y axis, negative for forward roll)
            # The car is facing +X. Rolling forward means rotating around -Y (right hand rule).
            # torque_dir = np.array([0.0, -1.0, 0.0])
            # torque_dir = torque_dir.reshape(1, 3)
            # torque_strength = 0.5
            # torque = torque_dir * torque_strength
            
            # car_obj.solver.apply_links_external_torque(torque=torque, links_idx=[car_obj.idx])

        if 9 <= sid <= 24:
            pitch_up_torque = self._car_roll_axis(self._car_motion_direction()).reshape(1, 3)
            car_obj.solver.apply_links_external_torque(torque=pitch_up_torque, links_idx=[car_obj.idx])

        rolling_start = int(self.config.get('rolling_force_start_frame', initial_push_end))
        rolling_end = int(self.config.get('rolling_force_end_frame', 35))
        if rolling_start <= sid <= rolling_end:
            force_dir = self._car_motion_direction()
            force_dir = force_dir.reshape(1, 3)
            
            force_strength = float(self.config.get('rolling_force_strength', 0.5))
            force = force_dir * force_strength
            
            car_obj.solver.apply_links_external_force(force=force, links_idx=[car_obj.idx])

        turn_start = int(self.config.get('turn_force_start_frame', 50))
        turn_end = int(self.config.get('turn_force_end_frame', 110))
        turn_dir = self._turn_force_direction().reshape(1, 3)
        yaw_axis = self._turn_yaw_axis().reshape(1, 3)
        pull_frame = int(self.config.get('turn_right_pull_frame', turn_start))
        if sid == pull_frame:
            right_speed = float(self.config.get('turn_right_initial_speed', 2.0))
            yaw_speed = float(self.config.get('turn_right_initial_yaw_speed', -1.0))
            init_qvel = np.array(
                [
                    turn_dir[0, 0] * right_speed,
                    turn_dir[0, 1] * right_speed,
                    turn_dir[0, 2] * right_speed,
                    yaw_axis[0, 0] * yaw_speed,
                    yaw_axis[0, 1] * yaw_speed,
                    yaw_axis[0, 2] * yaw_speed,
                ],
                dtype=np.float32,
            )
            car_obj.set_dofs_velocity(
                velocity=init_qvel,
                dofs_idx_local=np.arange(6),
            )

            pull_strength = float(self.config.get('turn_right_pull_strength', 45.0))
            car_obj.solver.apply_links_external_force(
                force=turn_dir * pull_strength,
                links_idx=[car_obj.idx],
            )

        if turn_start <= sid <= turn_end:
            turn_strength = float(self.config.get('turn_force_strength', 4.0))
            car_obj.solver.apply_links_external_force(
                force=turn_dir * turn_strength,
                links_idx=[car_obj.idx],
            )

            yaw_strength = float(self.config.get('turn_yaw_torque_strength', -0.8))
            car_obj.solver.apply_links_external_torque(
                torque=yaw_axis * yaw_strength,
                links_idx=[car_obj.idx],
            )


    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        """
        No specific position tweaks needed as the car is already on the ramp in the image.
        """
        return super().add_entities_to_scene(scene, obj_materials, obj_vis_modes)
