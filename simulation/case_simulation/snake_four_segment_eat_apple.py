from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("snake_four_segment_eat_apple")
class SnakeFourSegmentEatApple(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Simulates the snake moving in four segments to reach the apple.
        Segment 1: Right (sid 0-20)
        Segment 2: Down (sid 20-40)
        Segment 3: Right (sid 40-60)
        Segment 4: Down to apple (sid 60-81)
        """
        snake_obj = self.all_objs[0]
        apple_obj = self.all_objs[1]

        # Keep apple static
        apple_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        apple_obj.set_dofs_velocity(velocity=apple_vel, dofs_idx_local=np.arange(6))

        # Snake movement logic
        # Assuming X is Right, Y is Down (forward in world space)
        vx, vy = 0.0, 0.0
        speed = 0.4  # Units per second
        
        if sid < 20:
            # Move Right
            vx = speed
            vy = 0.0
        elif sid < 40:
            # Move Down
            vx = 0.0
            vy = speed
        elif sid < 60:
            # Move Right
            vx = speed
            vy = 0.0
        else:
            # Move Down towards apple
            vx = 0.0
            vy = speed
            
        # Set velocity: [vx, vy, vz, wx, wy, wz]
        # vz=0 (sliding on plane), angular=0 (no rotation)
        snake_vel = np.array([vx, vy, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        snake_obj.set_dofs_velocity(velocity=snake_vel, dofs_idx_local=np.arange(6))
