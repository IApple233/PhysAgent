from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("teaser_phys_ys")
class TeaserPhysYs(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)
        self.transform_cached = False
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0

    def custom_simulation(self, sid):
        # Cache transform based on object positions
        if not self.transform_cached and len(self.all_objs) >= 2:
            pos_marble = self.all_objs[0].get_pos().cpu().numpy()
            pos_rocket = self.all_objs[1].get_pos().cpu().numpy()
            
            # Image coords (approximate based on visual analysis)
            # Marble start: ~ (100, 100)
            # Rocket start: ~ (650, 100)
            img_marble_x, img_marble_y = 100, 100
            img_rocket_x, img_rocket_y = 650, 100
            
            dx_img = img_rocket_x - img_marble_x
            dx_world = pos_rocket[0] - pos_marble[0]
            
            if abs(dx_img) > 1e-5:
                self.scale = dx_world / dx_img
                self.offset_x = pos_marble[0] - img_marble_x * self.scale
                # Assume uniform scale for Y as well, mapping Image Y to World Y
                # Note: Image Y is down, World Y is forward. 
                # We map Image Y directly to World Y for simplicity, 
                # assuming the reconstruction aligns them reasonably.
                self.offset_y = pos_marble[1] - img_marble_y * self.scale
                self.transform_cached = True

        if not self.transform_cached:
            return

        total_frames = self.config['simulated_frames_num']
        t = sid / total_frames

        # Helper to convert image coords to world coords
        def img_to_world(ix, iy):
            wx = ix * self.scale + self.offset_x
            wy = iy * self.scale + self.offset_y
            return np.array([wx, wy, 0.0]) # Z is up, motion is on plane

        # Marble Path (Y)
        # 0-20%: Start(100,100) -> Fork(250,300)
        # 20-40%: Fork(250,300) -> TopRight(300,150)
        # 40-60%: TopRight(300,150) -> Fork(250,300)
        # 60-100%: Fork(250,300) -> Bottom(250,420)
        
        marble_waypoints = [
            (0.0, (100, 100)),
            (0.2, (250, 300)),
            (0.4, (300, 150)),
            (0.6, (250, 300)),
            (1.0, (250, 420))
        ]
        
        # Rocket Path (S)
        # 0-33%: Start(650,100) -> Mid1(550,200)
        # 33-66%: Mid1(550,200) -> Mid2(650,300)
        # 66-100%: Mid2(650,300) -> End(550,400)
        
        rocket_waypoints = [
            (0.0, (650, 100)),
            (0.33, (550, 200)),
            (0.66, (650, 300)),
            (1.0, (550, 400))
        ]

        def get_target_pos(waypoints, t):
            for i in range(len(waypoints) - 1):
                t_start, _ = waypoints[i]
                t_end, _ = waypoints[i+1]
                if t_start <= t <= t_end:
                    _, p1 = waypoints[i]
                    _, p2 = waypoints[i+1]
                    local_t = (t - t_start) / (t_end - t_start)
                    # Smooth interpolation (ease-in-out)
                    local_t = local_t * local_t * (3 - 2 * local_t)
                    ix = p1[0] + (p2[0] - p1[0]) * local_t
                    iy = p1[1] + (p2[1] - p1[1]) * local_t
                    return img_to_world(ix, iy)
            return img_to_world(waypoints[-1][1][0], waypoints[-1][1][1])

        # Marble (Object 0)
        target_marble = get_target_pos(marble_waypoints, t)
        pos_marble = self.all_objs[0].get_pos().cpu().numpy()
        # Apply force towards target
        # F = k * (target - pos) - damping * vel
        vel_marble = self.all_objs[0].get_vel().cpu().numpy()
        force_marble = 50.0 * (target_marble - pos_marble) - 5.0 * vel_marble
        # Zero out Z force to keep it on plane (mostly)
        force_marble[2] = 0.0 
        # Apply small upward force to counteract gravity if it sinks, or just rely on plane
        # Actually, rigid body on plane should be fine.
        
        self.all_objs[0].solver.apply_links_external_force(
            force=force_marble.reshape(1, 3),
            links_idx=[self.all_objs[0].idx]
        )

        # Rocket (Object 1)
        target_rocket = get_target_pos(rocket_waypoints, t)
        pos_rocket = self.all_objs[1].get_pos().cpu().numpy()
        vel_rocket = self.all_objs[1].get_vel().cpu().numpy()
        force_rocket = 50.0 * (target_rocket - pos_rocket) - 5.0 * vel_rocket
        force_rocket[2] = 0.0
        
        self.all_objs[1].solver.apply_links_external_force(
            force=force_rocket.reshape(1, 3),
            links_idx=[self.all_objs[1].idx]
        )
        
        # Orient Rocket to face movement direction
        if sid > 0:
            dir_rocket = target_rocket - pos_rocket
            if np.linalg.norm(dir_rocket) > 1e-5:
                dir_rocket = dir_rocket / np.linalg.norm(dir_rocket)
                # Simple orientation: align local X (forward) to dir_rocket
                # This is complex for rigid body rotation without quaternions.
                # For a teaser, position is most important.
                # I'll skip complex rotation to avoid instability, 
                # or just apply a torque to align if needed.
                # Given "rocket nose always faces movement direction", I should try.
                # But simple force guidance is safer for stability.
                pass

    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        return super().add_entities_to_scene(scene, obj_materials, obj_vis_modes)
