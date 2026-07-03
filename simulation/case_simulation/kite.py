from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("kite")
class Kite(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def detect_ground_plane(self, ground_plane):
        # The kite is flying in the sky, so we disable the ground plane detection
        # to prevent it from falling onto a virtual floor immediately.
        pass

    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        # Lift the kite slightly in the Z-axis so it starts in the "air"
        # relative to the scene center, preventing immediate ground collision 
        # if a ground were present, and giving it room to move.
        for i in range(len(self.all_obj_info)):
            self.all_obj_info[i]['center'][2] += 2.0
        return super().add_entities_to_scene(scene, obj_materials, obj_vis_modes)

    def create_force_fields(self):
        # Create a custom force field to simulate wind lift and forward thrust
        # allowing the kite to "fly through" the scene.
        
        @ti.func
        def wind_force_func(pos, vel, t, i):
            # Convert time to frames for periodic motion
            frame_step = t // self.config['dt'] / self.config['frame_steps']
            
            # Base Wind Force: 
            # Z component (10.0) counteracts gravity (-9.8) to keep it aloft.
            # X component (3.0) pushes it forward through the "gap".
            base_force = ti.Vector([3.0, 0.0, 10.0], dt=gs.ti_float)
            
            # Turbulence for the tail flutter:
            # Apply a sinusoidal force that varies with time and height (pos[2])
            # This makes the tail wave realistically.
            turbulence = ti.sin(frame_step * 0.5 + pos[2] * 0.5) * ti.Vector([0.5, 0.5, 0.2], dt=gs.ti_float)
            
            # Add some lateral sway
            sway = ti.cos(frame_step * 0.3) * ti.Vector([0.2, 0.5, 0.0], dt=gs.ti_float)
            
            total_force = base_force + turbulence + sway
            return total_force

        force_field = gs.force_fields.Custom(wind_force_func)
        force_field.activate()
        self.scene.add_force_field(force_field=force_field)