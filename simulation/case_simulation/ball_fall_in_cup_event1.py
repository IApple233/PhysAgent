from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("ball_fall_in_cup_event1")
class BallFallInCupEvent1(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Apply initial velocity to the ball to make it drop/roll towards the cup.
        Ball is index 0. Cup is index 1.
        """
        if sid == 0:
            ball_obj = self.all_objs[0]
            cup_obj = self.all_objs[1]
            
            # Get positions
            pos_ball = ball_obj.get_pos().cpu().numpy()
            pos_cup = cup_obj.get_pos().cpu().numpy()
            
            # Calculate direction from ball to cup
            # The ball is on the top right, cup on bottom left.
            # Direction should be roughly (-x, -z)
            direction = pos_cup - pos_ball
            
            # We want a push towards the cup, but mostly horizontal to clear the edge,
            # letting gravity handle the fall.
            # However, since the cup is lower, a direct vector helps aim.
            # Let's normalize the horizontal component mostly.
            
            horizontal_dir = direction.copy()
            horizontal_dir[2] = 0 # Ignore Z for direction calculation to aim horizontally
            
            norm = np.linalg.norm(horizontal_dir)
            if norm > 1e-5:
                horizontal_dir = horizontal_dir / norm
            else:
                horizontal_dir = np.array([1.0, 0.0, 0.0]) # Fallback
                
            # Apply initial linear velocity
            # Speed needs to be enough to clear the step edge and hit the cup
            speed = 2.5 
            initial_vel = horizontal_dir * speed
            
            # Add a slight downward component to help it fall off the edge if friction is high
            initial_vel[2] = -0.5 
            
            # Apply angular velocity for rolling
            # Rolling axis is perpendicular to motion and up vector (0,0,1)
            # Cross product of motion (x, y, 0) and up (0, 0, 1) is (y, -x, 0)
            # Wait, standard rolling: v = omega x r. 
            # If moving -X, rotating around +Y.
            # Axis = cross(horizontal_dir, [0, 0, 1])
            roll_axis = np.cross(horizontal_dir, np.array([0.0, 0.0, 1.0]))
            roll_speed = 10.0
            initial_ang_vel = roll_axis * roll_speed
            
            # Combine
            # dofs: [vx, vy, vz, wx, wy, wz]
            init_qvel = np.array([
                initial_vel[0], initial_vel[1], initial_vel[2],
                initial_ang_vel[0], initial_ang_vel[1], initial_ang_vel[2]
            ], dtype=np.float32)
            
            ball_obj.set_dofs_velocity(
                velocity=init_qvel,
                dofs_idx_local=np.arange(6),
            )

    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        """
        Adjust positions if necessary. 
        The ball is on the edge. We might want to ensure it's slightly off-center 
        to encourage falling, or just rely on the velocity push.
        """
        # No major position tweaks needed, the velocity push in custom_simulation handles the action.
        return super().add_entities_to_scene(scene, obj_materials, obj_vis_modes)
