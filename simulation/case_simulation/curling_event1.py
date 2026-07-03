from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("curling_event1")
class CurlingEvent1(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Implementation for rigid body forces/torques or one-shot initial
        linear/angular velocities based on sid.
        """
        # Object 0: Red stone (top in image, further back in scene)
        # Object 1: Yellow stone (bottom in image, on target)
        
        if sid == 0:
            red_stone = self.all_objs[0]
            
            # The prompt requires the red stone to curl "wide to the left".
            # In curling physics (and this coordinate system):
            # - Motion is primarily forward (+Y direction towards the target).
            # - Counter-clockwise rotation (positive Z angular velocity) causes a leftward curl.
            # - Clockwise rotation (negative Z) causes a rightward curl.
            # Previous failure was spin_z = -4.0 (right curl).
            # Fix: Use positive spin_z (e.g., +5.0) for left curl.
            
            # Initial velocity setup:
            # vx: Slight negative bias to ensure it starts heading left of the yellow stone.
            # vy: Positive forward velocity to slide down the ice.
            # vz: 0 (flat on ice).
            # wx, wy: 0 (no tumbling).
            # wz: Positive value for counter-clockwise spin (Left Curl).
            
            init_qvel = np.array([
                -0.8,   # vx: slight leftward bias
                3.5,    # vy: forward slide speed
                0.0,    # vz
                0.0,    # wx
                0.0,    # wy
                5.0     # wz: counter-clockwise spin for left curl
            ], dtype=np.float32)
            
            red_stone.set_dofs_velocity(
                velocity=init_qvel,
                dofs_idx_local=np.arange(6),
            )
