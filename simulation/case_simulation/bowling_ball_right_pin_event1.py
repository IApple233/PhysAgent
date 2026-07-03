from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("bowling_ball_right_pin_event1")
class BowlingBallRightPinEvent1(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        # Object indices based on all_object_points order:
        # 0: Ball
        # 1: Left Pin
        # 2: Center Pin
        # 3: Right Pin
        
        if sid == 0:
            # Get positions
            pos_ball = self.all_objs[0].get_pos().cpu().numpy()
            pos_pin_center = self.all_objs[2].get_pos().cpu().numpy() # Aim at center pin initially
            
            # Direction to center pin (simulating a throw aimed for a strike/center)
            direction = pos_pin_center - pos_ball
            direction[2] = 0 # Keep movement mostly horizontal
            direction /= np.linalg.norm(direction)
            
            # Velocity magnitude
            speed = 5.0
            velocity = direction * speed
            
            # Set linear velocity [vx, vy, vz] and angular velocity [wx, wy, wz]
            # Forward roll around X axis (assuming X is right, Y is forward)
            # Adding spin makes it roll realistically
            init_qvel = np.array([velocity[0], velocity[1], velocity[2], 15.0, 0.0, 0.0], dtype=np.float32)
            
            self.all_objs[0].set_dofs_velocity(
                velocity=init_qvel,
                dofs_idx_local=np.arange(6)
            )
        
        # Apply "Veer Left" force
        # The prompt says "veers left instead of right".
        # We apply a force in the -X direction (Left) to curve the ball away from the center pin
        # and towards the left pin.
        if sid < 40: # Apply force for the first part of the roll
            force_magnitude = 12.0 
            # Force direction: Left (-X)
            force_dir = np.array([-1.0, 0.0, 0.0], dtype=np.float32).reshape(1, 3)
            
            self.all_objs[0].solver.apply_links_external_force(
                force=force_dir * force_magnitude, 
                links_idx=[self.all_objs[0].idx]
            )
