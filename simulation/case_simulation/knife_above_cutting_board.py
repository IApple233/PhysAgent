from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("knife_above_cutting_board")
class KnifeAboveCuttingBoard(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Simulates the knife toss.
        1. At sid=0, apply an initial upward velocity and angular velocity to the knife.
        2. The knife rotates end-over-end.
        3. Gravity pulls it down.
        4. It impacts the board. The high rigid_coup_softness allows the tip to penetrate.
        """
        if sid == 0:
            knife = self.all_objs[0]
            
            # Initial state: Knife is floating above the board (from reconstruction).
            # We want to toss it UPWARD so it rotates and comes back down tip-first.
            
            # Linear velocity: Upward (Z) and slight forward/backward if needed, but mostly Up.
            # vz = 4.0 gives a decent air time.
            vz = 4.5
            
            # Angular velocity: Needs to rotate around the axis perpendicular to the blade.
            # Based on the image, the knife is roughly in the X-Z plane (diagonal).
            # Rotation around Y axis (wy) will cause it to flip end-over-end.
            # We need enough rotation so that when it falls, the tip is down.
            # wy = 10.0 rad/s is fast enough for a flip.
            wy = 12.0
            
            # DoFs: [vx, vy, vz, wx, wy, wz]
            init_qvel = np.array([0.0, 0.0, vz, 0.0, wy, 0.0], dtype=np.float32)
            
            knife.set_dofs_velocity(velocity=init_qvel, dofs_idx_local=np.arange(6))
