from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("teaser_phys_ph")
class TeaserPhysPh(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Custom simulation to drive the car and ball along the letter paths.
        Car (Obj 0): Traces 'P'.
        Ball (Obj 1): Traces 'h'.
        """
        # Total frames: 81
        # dt: 0.1
        
        car = self.all_objs[0]
        ball = self.all_objs[1]
        
        # Get current positions (CPU numpy)
        car_pos = car.get_pos().cpu().numpy()
        ball_pos = ball.get_pos().cpu().numpy()
        
        # Define path parameters
        # Speeds
        v_car = 0.08
        v_ball = 0.08
        
        # Car Path (P)
        # Phase 1: Up (0-25)
        # Phase 2: Right (25-45)
        # Phase 3: Down/Left curve (45-65)
        # Phase 4: Stop/Settle (65-80)
        
        car_vx, car_vy, car_vz = 0.0, 0.0, 0.0
        car_wx, car_wy, car_wz = 0.0, 0.0, 0.0
        
        if sid < 25:
            # Move Up (+Y)
            car_vx = 0.0
            car_vy = v_car
            # Face Up (0 degrees yaw? No, Y is forward in some coords, but here Y is image-up)
            # Image Up is +Y. Car should face +Y.
            # Default car orientation might be facing +X or +Y.
            # Let's assume we just set velocity.
            car_wz = 0.0 
        elif sid < 45:
            # Move Right (+X)
            car_vx = v_car
            car_vy = 0.0
            # Face Right (+X) -> Rotate -90 deg (if Y is up) or 90 deg?
            # If facing Y (up), to face X (right), rotate -90 deg around Z.
            car_wz = -0.5 # Simple rotation
        elif sid < 65:
            # Curve Down (-Y) and Left (-X) to form P loop
            # Move in a circle?
            # Approximate: Move Down and Left
            car_vx = -v_car * 0.5
            car_vy = -v_car
            car_wz = 0.5 # Rotate back?
        else:
            # Slow down
            car_vx = 0.0
            car_vy = 0.0
            car_wz = 0.0
            
        # Ball Path (h)
        # Phase 1: Down (0-20)
        # Phase 2: Up (20-35)
        # Phase 3: Right/Down curve (35-55)
        # Phase 4: Down (55-80)
        
        ball_vx, ball_vy, ball_vz = 0.0, 0.0, 0.0
        ball_wx, ball_wy, ball_wz = 0.0, 0.0, 0.0
        
        if sid < 20:
            # Move Down (-Y)
            ball_vx = 0.0
            ball_vy = -v_ball
            # Roll: Rotate around X axis (since moving in Y)
            # Rolling forward (down -Y) means rotation around -X?
            # v = r * w. w = v / r.
            # Let's just add some spin.
            ball_wx = 5.0 
        elif sid < 35:
            # Move Up (+Y)
            ball_vx = 0.0
            ball_vy = v_ball
            ball_wx = -5.0 # Reverse spin
        elif sid < 55:
            # Right (+X) and Down (-Y) curve
            ball_vx = v_ball
            ball_vy = -v_ball * 0.5
            ball_wx = 2.0
            ball_wz = -2.0 # Some curve spin
        else:
            # Move Down (-Y)
            ball_vx = 0.0
            ball_vy = -v_ball
            ball_wx = 5.0
            
        # Apply velocities
        # Car
        car_vel = np.array([car_vx, car_vy, car_vz, car_wx, car_wy, car_wz], dtype=np.float32)
        car.set_dofs_velocity(velocity=car_vel, dofs_idx_local=np.arange(6))
        
        # Ball
        ball_vel = np.array([ball_vx, ball_vy, ball_vz, ball_wx, ball_wy, ball_wz], dtype=np.float32)
        ball.set_dofs_velocity(velocity=ball_vel, dofs_idx_local=np.arange(6))
