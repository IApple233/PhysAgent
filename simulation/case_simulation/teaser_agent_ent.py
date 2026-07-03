from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("teaser_agent_ent")
class TeaserAgentEnt(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        # Lift the Rubik's cube (object 2) slightly so it falls/tumbles
        # Object indices: 0=Top, 1=Plane, 2=Cube
        if len(self.all_obj_info) > 2:
            # Lift along Z (up)
            self.all_obj_info[2]['center'][2] += 0.2
            
        return super().add_entities_to_scene(scene, obj_materials, obj_vis_modes)

    def custom_simulation(self, sid):
        if sid == 0:
            # Object 0: Spinning Top
            # Set high angular velocity around Z axis (spin)
            if len(self.all_objs) > 0:
                obj_top = self.all_objs[0]
                # [vx, vy, vz, wx, wy, wz]
                # Spin around Z
                init_qvel_top = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 15.0], dtype=np.float32)
                obj_top.set_dofs_velocity(
                    velocity=init_qvel_top,
                    dofs_idx_local=np.arange(6),
                )

            # Object 1: Paper Airplane
            # Set forward velocity (flying along the 'n')
            # In top-down view, 'up' the image is -Y in Genesis (assuming standard mapping)
            # Or just forward in the scene. Let's try -Y.
            if len(self.all_objs) > 1:
                obj_plane = self.all_objs[1]
                # Fly forward (-Y) and slightly up (Z) to glide
                init_qvel_plane = np.array([0.0, -2.5, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                obj_plane.set_dofs_velocity(
                    velocity=init_qvel_plane,
                    dofs_idx_local=np.arange(6),
                )

            # Object 2: Rubik's Cube
            # Just let it drop (gravity acts on it). 
            # Maybe a slight random tumble?
            if len(self.all_objs) > 2:
                obj_cube = self.all_objs[2]
                # Small initial rotation/tumble
                init_qvel_cube = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32)
                obj_cube.set_dofs_velocity(
                    velocity=init_qvel_cube,
                    dofs_idx_local=np.arange(6),
                )
