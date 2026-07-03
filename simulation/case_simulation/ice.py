from simulation.case_simulation.case_handler import CaseHandler, register_case
import numpy as np
import torch
import gstaichi as ti
import genesis as gs

@register_case("ice")
class Ice(CaseHandler):
    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)

    def custom_simulation(self, sid):
        """
        Implementation for rigid body forces/torques.
        Pushes the iceberg towards the gap in the ice shelf.
        """
        # Object 0 is the iceberg (dynamic)
        iceberg = self.all_objs[0]
        
        # Apply a gentle drift force towards the left (towards the shelf/gap)
        # Based on image analysis: Shelf is Left (-X), Iceberg is Right (+X).
        # We apply force in -X direction.
        if sid < 60: # Push for the first 60 steps
            # Force direction: Left (-X) and slightly Forward/Backward to align if needed
            # Using a small force for "slowly moves forward"
            force_direction = np.array([-0.3, 0.0, 0.0], dtype=np.float32)
            force_direction = force_direction.reshape(1, 3)
            
            # Apply force
            iceberg.solver.apply_links_external_force(
                force=force_direction, 
                links_idx=[iceberg.idx]
            )
            
            # Apply a small torque to help it "align" with the opening
            # Rotating around Z axis (vertical)
            torque = np.array([0.0, 0.0, 0.02], dtype=np.float32)
            torque = torque.reshape(1, 3)
            iceberg.solver.apply_links_external_torque(
                torque=torque,
                links_idx=[iceberg.idx]
            )

    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        """
        Adjust initial positions if necessary.
        The iceberg might need to be slightly lifted to ensure it floats on the water plane
        and doesn't intersect the shelf mesh initially.
        """
        # Lift the iceberg slightly to ensure it's floating on the water plane
        # and not intersecting the shelf mesh walls immediately.
        # The shelf is the support, but the water is the plane.
        # We want the iceberg to be at the water level.
        # SAM3D might place it slightly low or high.
        # Let's nudge it up slightly to ensure it's "floating".
        self.all_obj_info[0]['center'][2] += 0.05
        
        return super().add_entities_to_scene(scene, obj_materials, obj_vis_modes)
