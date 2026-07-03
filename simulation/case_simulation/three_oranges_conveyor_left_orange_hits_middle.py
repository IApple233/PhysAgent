from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("three_oranges_conveyor_left_orange_hits_middle")
class ThreeOrangesConveyorLeftOrangeHitsMiddle(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Apply initial velocity to the left orange (index 0) towards the middle orange (index 1).
        The physics engine handles the collision, stopping of the first orange, and the falling/bouncing of the second.
        """
        if sid == 0:
            # Get the left orange (index 0) and middle orange (index 1)
            obj_0 = self.all_objs[0]
            obj_1 = self.all_objs[1]

            # Get positions to calculate direction
            pos_0 = obj_0.get_pos().cpu().numpy()
            pos_1 = obj_1.get_pos().cpu().numpy()

            # Calculate direction from obj_0 to obj_1
            direction = pos_1 - pos_0
            norm = np.linalg.norm(direction)
            if norm > 1e-5:
                direction = direction / norm
            else:
                # Fallback if positions are too close (shouldn't happen based on image)
                direction = np.array([1.0, 0.0, 0.0])

            # Set initial velocity for obj_0 towards obj_1
            # Slow enough that it can stop on the belt after contact.
            speed = 1.15
            velocity_vec = direction * speed
            
            # Linear velocity [vx, vy, vz], Angular velocity [wx, wy, wz] = 0 (rolling induced by friction)
            init_qvel = np.array([velocity_vec[0], velocity_vec[1], velocity_vec[2], 0.0, 0.0, 0.0], dtype=np.float32)
            
            obj_0.set_dofs_velocity(
                velocity=init_qvel,
                dofs_idx_local=np.arange(6),
            )

        if 18 <= sid < 46:
            obj_0 = self.all_objs[0]
            obj_1 = self.all_objs[1]
            pos_0 = obj_0.get_pos().cpu().numpy()
            pos_1 = obj_1.get_pos().cpu().numpy()
            direction = pos_1 - pos_0
            direction[2] = 0.0
            norm = np.linalg.norm(direction)
            if norm > 1e-5:
                direction = direction / norm
            else:
                direction = np.array([1.0, 0.0, 0.0])
            force = (direction * 32.0).reshape(1, 3).astype(np.float32)
            obj_1.solver.apply_links_external_force(force=force, links_idx=[obj_1.idx])

        if sid == 30:
            stop_qvel = np.zeros(6, dtype=np.float32)
            self.all_objs[0].set_dofs_velocity(velocity=stop_qvel, dofs_idx_local=np.arange(6))

    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        """
        The reconstructed belt/floor tend to sit slightly too high for this case.
        Lift the fruit a little after entity creation so the initial layout starts
        outside the static support meshes.
        """
        objs = super().add_entities_to_scene(scene, obj_materials, obj_vis_modes)
        offsets = [
            np.array([0.0, 0.0, 0.025], dtype=np.float64),
            np.array([0.0, 0.0, 0.025], dtype=np.float64),
            np.array([0.0, 0.0, 0.045], dtype=np.float64),
        ]
        for obj_idx, offset in enumerate(offsets[:len(objs)]):
            current_center = self.all_obj_info[obj_idx]['center'].detach().cpu().numpy().astype(np.float64)
            new_center = current_center + offset
            if self._set_dynamic_entity_center(objs[obj_idx], new_center, obj_idx):
                self._update_dynamic_object_info_after_shift(obj_idx, offset)
        return objs

    def get_case_static_support_offset_gs(self, spec):
        name = str(spec.get('name', spec.get('label', ''))).lower()
        mesh_path = str(spec.get('mesh_path_gs') or spec.get('mesh_path') or spec.get('file') or '').lower()
        support_id = f"{name} {mesh_path}"
        if 'conveyor' in support_id or 'belt' in support_id:
            return np.array([0.0, 0.0, -0.035], dtype=np.float64)
        if 'floor' in support_id:
            return np.array([0.0, 0.0, -0.045], dtype=np.float64)
        return None
