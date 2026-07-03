from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("santa_cloth")
class SantaCloth(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def fix_particles(self):
        """
        Pin the top particles of the Santa suit to simulate hanging from the hanger.
        Uses the fixed_area defined in YAML: [x_min, x_max, z_min, z_max] (normalized).
        """
        # Get initial particles for the cloth object (index 0)
        sim_particles = torch.tensor(self.all_objs[0].init_particles).to(self.device)
        
        # Retrieve bounding box info
        min_bound = self.all_obj_info[0]['min']
        max_bound = self.all_obj_info[0]['max']
        size = self.all_obj_info[0]['size']
        
        # Parse fixed_area from config [x_left, x_right, z_top, z_bottom]
        # Note: In this engine, Z is up. The normalized box is usually [x_min, x_max, y_min, y_max] in 2D mask space 
        # but mapped to 3D bounds. Let's assume standard normalized bbox logic relative to object bounds.
        # YAML fixed_area: [[0.0, 1.0, 0.0, 0.15]] -> Top 15% of the object's vertical span.
        # However, coordinate system: Y is often depth or height depending on view. 
        # In Genesis PBD, usually Z is up. 
        # Let's map the normalized rect to 3D space.
        # Assuming fixed_area refers to [x_norm_min, x_norm_max, y_norm_min, y_norm_max] of the 2D mask projected to 3D.
        # But for 3D pinning, we need Z (height). 
        # Let's interpret fixed_area as [x_ratio_min, x_ratio_max, z_ratio_min, z_ratio_max] relative to object AABB.
        # Since Z is UP, z_ratio 0.0 is bottom, 1.0 is top? Or 0.0 is top? 
        # Standard image coords: 0 is top. 3D coords: 0 is often bottom or center.
        # Let's assume the config 'fixed_area' is [x_min, x_max, y_min, y_max] in normalized image space (0,0 top-left).
        # We need to convert this to 3D world coordinates to filter particles.
        
        # Safer approach for hanging: Pin particles that are high in Z (world up).
        # Let's calculate the Z threshold for the top 15%.
        z_max_world = max_bound[2]
        z_min_world = min_bound[2]
        z_height = z_max_world - z_min_world
        
        # We want to pin the top part. In image coords, 0 is top. In 3D Z, usually max is top.
        # So we want particles with Z > (z_max - 0.15 * z_height)
        z_threshold = z_max_world - 0.15 * z_height
        
        # Filter particles based on Z coordinate (index 2)
        # sim_particles shape: (N, 3)
        z_coords = sim_particles[:, 2]
        mask = z_coords > z_threshold
        
        # Also restrict X to be within the shoulder width roughly (middle 60%)
        x_min_world = min_bound[0]
        x_max_world = max_bound[0]
        x_width = x_max_world - x_min_world
        x_mask = (sim_particles[:, 0] > x_min_world + 0.2 * x_width) & \
                 (sim_particles[:, 0] < x_max_world - 0.2 * x_width)
        
        final_mask = mask & x_mask
        
        selected_particles = sim_particles[final_mask]
        
        # Convert to list of tuples for finding closest
        selected_list = selected_particles.cpu().numpy()
        
        for p in selected_list:
            closest_idx = self.all_objs[0].find_closest_particle(p)
            self.all_objs[0].fix_particle(closest_idx, 0) # 0 means fully fixed

    def create_force_fields(self):
        """
        Create a rhythmic wind force field to blow the hanging clothes.
        """
        @ti.func
        def wind_force_func(pos, vel, t, i):
            # t is simulation time
            # Create a gentle sinusoidal wind in X direction (side to side)
            # Frequency: 0.5 Hz approx, Amplitude: gentle
            force_x = -20 * ti.sin(1.5 * t) 
            force_y = 0.0
            force_z = 0.0 # No vertical wind
            
            # Add a slight turbulence component
            turbulence = -20 * ti.sin(3.0 * t + pos[0])
            
            return ti.Vector([force_x + turbulence, force_y, force_z], dt=gs.ti_float)

        force_field = gs.force_fields.Custom(wind_force_func)
        force_field.activate()  # CRITICAL: Must activate
        self.scene.add_force_field(force_field=force_field)

    def detect_ground_plane(self, ground_plane):
        """
        Override to disable ground plane. Hanging clothes should swing freely 
        without hitting a floor immediately below them.
        """
        pass