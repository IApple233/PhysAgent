from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("toy_car_acrylic_ramp_turn_right")
class ToyCarAcrylicRampTurnRight(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Simulates the car rolling down the ramp and turning right on the table.
        """
        car = self.all_objs[0]
        
        # Initial push at the very beginning
        if sid == 0:
            # Apply initial velocity to push the car gently to the right (down the ramp)
            # and add angular velocity to simulate rolling wheels.
            # X is right, Y is depth, Z is up.
            # Ramp slopes down to the right.
            # vx: forward speed, wy: rolling rotation (around Y axis)
            init_qvel = np.array([0.8, 0.0, 0.0, 0.0, -4.0, 0.0], dtype=np.float32)
            car.set_dofs_velocity(velocity=init_qvel, dofs_idx_local=np.arange(6))
        
        # Turn right after the car has likely reached the flat tabletop
        # Ramp descent takes roughly 60-80 frames. Table travel starts after.
        # Apply torque to turn right (clockwise around Z axis)
        if 90 <= sid <= 130:
            torque = np.array([0.0, 0.0, -0.4], dtype=np.float32).reshape(1, 3)
            car.solver.apply_links_external_torque(torque=torque, links_idx=[car.idx])

    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        """
        Adjust initial position slightly to ensure no intersection with the ramp.
        """
        # Lift the car slightly to ensure it starts above the ramp surface
        self.all_obj_info[0]['center'][2] += 0.02
        return super().add_entities_to_scene(scene, obj_materials, obj_vis_modes)
