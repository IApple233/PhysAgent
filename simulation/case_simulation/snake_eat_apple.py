from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("snake_eat_apple")
class SnakeEatApple(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)
        self.state = 0  # 0: moving right, 1: turning, 2: moving down
        self.turn_counter = 0
        
    def custom_simulation(self, sid):
        # Object 0 is the snake, Object 1 is the apple
        snake = self.all_objs[0]
        apple = self.all_objs[1]
        
        # Get positions for logic checks
        pos_snake = snake.get_pos().cpu().numpy()[0]
        pos_apple = apple.get_pos().cpu().numpy()[0]
        
        # State 0: Move Right (+X)
        if self.state == 0:
            # Set velocity to move right (assuming X is right in this view)
            # Velocity [vx, vy, vz, wx, wy, wz]
            velocity = np.array([2.5, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            snake.set_dofs_velocity(velocity=velocity, dofs_idx_local=np.arange(6))
            
            # Check if snake has reached the apple's column (X coordinate)
            # Allow a small margin
            if pos_snake[0] >= pos_apple[0] - 0.3:
                self.state = 1
                # Stop linear motion to prepare for turn
                stop_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                snake.set_dofs_velocity(velocity=stop_vel, dofs_idx_local=np.arange(6))
                
        # State 1: Turn (Rotate around Z axis to face +Y)
        elif self.state == 1:
            # Apply torque around Z axis to rotate clockwise/counter-clockwise to face down
            # Assuming Z is up, rotating from +X to +Y is +90 deg (Counter Clockwise looking from top)
            # Wait, standard math: X(1,0) -> Y(0,1) is +90 deg rotation.
            # Torque should be positive Z.
            torque = np.array([0.0, 0.0, 15.0], dtype=np.float32).reshape(1, 3)
            snake.solver.apply_links_external_torque(torque=torque, links_idx=[snake.idx])
            
            self.turn_counter += 1
            
            # After some steps, assume turn is complete and switch to moving down
            # 20 steps at dt=0.05 is 1 second. Should be enough for a 90 deg turn with that torque.
            if self.turn_counter > 15:
                self.state = 2
                self.turn_counter = 0
                # Stop torque and rotation
                stop_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                snake.set_dofs_velocity(velocity=stop_vel, dofs_idx_local=np.arange(6))
                
        # State 2: Move Down (+Y)
        elif self.state == 2:
            # Set velocity to move down (assuming Y is forward/down in this view)
            velocity = np.array([0.0, 2.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            snake.set_dofs_velocity(velocity=velocity, dofs_idx_local=np.arange(6))
            
            # Optional: Stop if close to apple (eaten)
            dist = np.linalg.norm(pos_snake - pos_apple)
            if dist < 0.2:
                 stop_vel = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                 snake.set_dofs_velocity(velocity=stop_vel, dofs_idx_local=np.arange(6))
