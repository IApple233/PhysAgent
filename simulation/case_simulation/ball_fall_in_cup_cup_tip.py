from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("ball_fall_in_cup_cup_tip")
class BallFallInCupCupTip(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Apply an initial impulse to the ball (Object 0) to make it roll off the upper step.
        The ball is at [156, 60] (Upper Step) and needs to move towards the cup at [91, 102] (Lower Step).
        This implies movement in -X direction (Left).
        """
        # Object 0 is the Ball
        obj_ball = self.all_objs[0]
        
        # Apply initial velocity only at the start to trigger the roll
        if sid == 0:
            # Velocity: Left (-X), No Y, No Z (gravity handles fall)
            # Angular velocity: Around Y axis to simulate rolling
            # [vx, vy, vz, wx, wy, wz]
            # Moving -X implies rolling around +Y (Right hand rule: Thumb +Y, Fingers curl Z to -X... wait.
            # Top moves -X, Bottom stationary. Rotation is around +Y.
            init_qvel = np.array([
                -2.5, 0.0, 0.0,  # Linear velocity (Left)
                0.0, 5.0, 0.0    # Angular velocity (Rolling)
            ], dtype=np.float32)
            
            obj_ball.set_dofs_velocity(
                velocity=init_qvel,
                dofs_idx_local=np.arange(6),
            )

    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        """
        No specific position tweaks needed as objects are already on the steps.
        """
        return super().add_entities_to_scene(scene, obj_materials, obj_vis_modes)
