from submodules.sam_3d_objects.notebook.inference import Inference
import numpy as np
import torch
import trimesh
from pytorch3d.transforms import quaternion_to_matrix, Transform3d
from simulation.utils import intrinsics_to_fov_opencv
from typing import Tuple

from scipy.spatial import cKDTree
from trimesh import smoothing

import numpy as np
import torch
from copy import deepcopy
from pytorch3d.transforms import quaternion_to_matrix
from sam3d_objects.data.dataset.tdfy.transforms_3d import compose_transform
from sam3d_objects.pipeline import inference_utils
from sam3d_objects.pipeline import inference_pipeline
from sam3d_objects.pipeline import inference_pipeline_pointmap

_R_ZUP_TO_YUP = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
_R_YUP_TO_ZUP = _R_ZUP_TO_YUP.T

R_yup_to_zup = torch.tensor([[-1,0,0],[0,0,1],[0,1,0]], dtype=torch.float32)
R_flip_z = torch.tensor([[1,0,0],[0,1,0],[0,0,-1]], dtype=torch.float32)
R_pytorch3d_to_cam = torch.tensor([[-1,0,0],[0,1,0],[0,0,1]], dtype=torch.float32)

def configure_sam3d_sparse_downsample(max_coords=None, downsample_factor=None):
    if not hasattr(inference_utils, "_realwonder_original_downsample_sparse_structure"):
        inference_utils._realwonder_original_downsample_sparse_structure = (
            inference_utils.downsample_sparse_structure
        )

        def patched_downsample_sparse_structure(coord_batch, max_coords=42000, downsample_factor=2):
            configured_max_coords = getattr(inference_utils, "_realwonder_sparse_max_coords", None)
            configured_factor = getattr(inference_utils, "_realwonder_sparse_downsample_factor", None)
            if configured_max_coords is not None:
                max_coords = min(int(max_coords), int(configured_max_coords))
            if configured_factor is not None:
                downsample_factor = int(configured_factor)
            return inference_utils._realwonder_original_downsample_sparse_structure(
                coord_batch,
                max_coords=max_coords,
                downsample_factor=downsample_factor,
            )

        inference_utils.downsample_sparse_structure = patched_downsample_sparse_structure
        inference_pipeline.downsample_sparse_structure = patched_downsample_sparse_structure
        inference_pipeline_pointmap.downsample_sparse_structure = patched_downsample_sparse_structure

    inference_utils._realwonder_sparse_max_coords = (
        None if max_coords is None else int(max_coords)
    )
    inference_utils._realwonder_sparse_downsample_factor = (
        None if downsample_factor is None else int(downsample_factor)
    )

def transform_mesh_vertices(vertices, rotation, translation, scale):

    if isinstance(vertices, np.ndarray):
        vertices = torch.tensor(vertices, dtype=torch.float32)

    vertices = vertices.unsqueeze(0)  #  batch dimension [1, N, 3]
    vertices = vertices @ R_flip_z.to(vertices.device) 
    vertices = vertices @ R_yup_to_zup.to(vertices.device)
    R_mat = quaternion_to_matrix(rotation.to(vertices.device))
    tfm = Transform3d(dtype=vertices.dtype, device=vertices.device)
    tfm = (
        tfm.scale(scale)
           .rotate(R_mat)
           .translate(translation[0], translation[1], translation[2])
    )
    vertices_world = tfm.transform_points(vertices)
    vertices_world = vertices_world @ R_pytorch3d_to_cam.to(vertices_world.device)
    
    return vertices_world[0]  # remove batch dimension


class Sam3DMeshGenerator:
    def __init__(self, config, device="cuda"):
        self.config = config
        self.device = device
        config_path = "submodules/sam_3d_objects/checkpoints/hf/pipeline.yaml"
        configure_sam3d_sparse_downsample(
            max_coords=config.get("sam3d_max_sparse_coords", None),
            downsample_factor=config.get("sam3d_sparse_downsample_factor", None),
        )
        self.model = Inference(config_path, compile=False)

    def __call__(
        self,
        image,
        mask,
        mesh_resize_factor=1.0,
        target_faces=500,
        seed=42,
        simplify_proxy_scale=1.0,
        voxel_pitch=None,
        smoothing_iterations=None,
    ) -> Tuple[trimesh.Trimesh, float, float, float]:
        result = self.model(image, mask, seed=seed)

        intrinsics = result.get("intrinsics")  # [3, 3]

        fx_pixels, fy_pixels, fov_x_deg, fov_y_deg, fov_x_rad, fov_y_rad = (
            intrinsics_to_fov_opencv(intrinsics, image.shape[:2])
        )

        mesh = result["glb"]

        vertices = mesh.vertices
        
        # S = result["scale"][0].cpu().float() * mesh_resize_factor
        # T = result["translation"][0].cpu().float()
        # R = result["rotation"].squeeze().cpu().float()
        # vertices_transformed = transform_mesh_vertices(vertices, R, T, S)
        # mesh.vertices = vertices_transformed.cpu().numpy().astype(np.float32)

        vertices = vertices.astype(np.float32) @ _R_YUP_TO_ZUP
        vertices_tensor = torch.from_numpy(vertices).float().to(result["rotation"].device)
        R_l2c = quaternion_to_matrix(result["rotation"])
        l2c_transform = compose_transform(
            scale=result["scale"] * mesh_resize_factor,
            rotation=R_l2c,
            translation=result["translation"],
        )
        vertices = l2c_transform.transform_points(vertices_tensor.unsqueeze(0))
        mesh.vertices = vertices.squeeze(0).cpu().numpy() # @ _R_ZUP_TO_YUP

        # # import pdb; pdb.set_trace()
        # vertices = mesh.vertices
        # vertices = vertices.astype(np.float32) @ _R_YUP_TO_ZUP
        # vertices_tensor = torch.from_numpy(vertices).float().to(result["rotation"].device)
        # R_l2c = quaternion_to_matrix(result["rotation"])

        # x_angle = torch.tensor(-0.0 * torch.pi / 180.0, device=result["rotation"].device)
        # cos_a, sin_a = torch.cos(x_angle), torch.sin(x_angle)
        # # z_angle = torch.tensor(5.0 * torch.pi / 180.0, device=result["rotation"].device)
        # # cos_z, sin_z = torch.cos(z_angle), torch.sin(z_angle)
        # R_x = torch.tensor([[1, 0, 0],
        #                 [0, cos_a, -sin_a],
        #                 [0, sin_a, cos_a]], device=result["rotation"].device, dtype=R_l2c.dtype)
        # # R_z = torch.tensor([[cos_z, -sin_z, 0],
        # #                 [sin_z, cos_z, 0],
        # #                 [0, 0, 1]], device=result["rotation"].device, dtype=R_l2c.dtype)
        # combined_rotation = R_x @ R_l2c

        # l2c_transform = compose_transform(
        #     scale=result["scale"] * mesh_resize_factor,
        #     rotation=combined_rotation,
        #     translation=result["translation"],
        # )
        # vertices = l2c_transform.transform_points(vertices_tensor.unsqueeze(0))
        # mesh.vertices = vertices.squeeze(0).cpu().numpy()


        mesh_trimesh = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces[:, [0, 2, 1]],
            # faces=mesh.faces,
            vertex_colors=mesh.visual.vertex_colors[:, :3],
        )

        def simplify_mesh_with_smooth_colors(mesh, target_faces=20000, k_neighbors=3):
            
            original_vertices = mesh.vertices.copy()
            original_colors = mesh.visual.vertex_colors.copy().astype(np.float32)
            
            simplified_mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
            tree = cKDTree(original_vertices)
            distances, indices = tree.query(simplified_mesh.vertices, k=k_neighbors)
            
            weights = 1.0 / (distances + 1e-8)
            weights = weights / weights.sum(axis=1, keepdims=True)
            
            interpolated_colors = np.zeros((len(simplified_mesh.vertices), 4), dtype=np.float32)
            for i in range(len(simplified_mesh.vertices)):
                interpolated_colors[i] = np.average(original_colors[indices[i]], weights=weights[i], axis=0)
            
            simplified_mesh.visual.vertex_colors = interpolated_colors.astype(np.uint8)
            
            return simplified_mesh

        simplify_proxy_scale = float(simplify_proxy_scale)
        simplify_mesh = mesh_trimesh
        simplify_anchor = None
        if abs(simplify_proxy_scale) > 1e-8 and abs(simplify_proxy_scale - 1.0) > 1e-8:
            simplify_mesh = mesh_trimesh.copy()
            simplify_anchor = simplify_mesh.vertices.mean(axis=0)
            simplify_mesh.vertices = (
                (simplify_mesh.vertices - simplify_anchor[None, :]) * simplify_proxy_scale
                + simplify_anchor[None, :]
            )

        simplified_mesh = simplify_mesh_with_smooth_colors(simplify_mesh, target_faces=target_faces)

        if self.config.get("original_geometry_downsample", True):
            mesh_trimesh = simplify_mesh_with_smooth_colors(mesh_trimesh, target_faces=self.config.get("target_faces", 5000))

        if voxel_pitch is None:
            voxel_pitch = float(self.config.get("mesh_voxel_pitch", 0.01))
        else:
            voxel_pitch = float(voxel_pitch)
        smoothing_iterations = int(
            self.config.get("mesh_smoothing_iterations", 0)
            if smoothing_iterations is None
            else smoothing_iterations
        )

        # if not simplified_mesh.is_watertight:
        print("mesh is not watertight, filling holes...")
        vox = simplified_mesh.voxelized(pitch=voxel_pitch)
        volume = vox.fill()
        new_mesh = volume.marching_cubes
        
        new_mesh.apply_transform(vox.transform)
        simplified_mesh = new_mesh
        if smoothing_iterations > 0:
            smoothing.filter_humphrey(
                simplified_mesh,
                alpha=float(self.config.get("mesh_smoothing_alpha", 0.1)),
                beta=float(self.config.get("mesh_smoothing_beta", 0.5)),
                iterations=smoothing_iterations,
            )
        if len(simplified_mesh.faces) > target_faces:
            simplified_mesh = simplified_mesh.simplify_quadric_decimation(face_count=target_faces)
        if simplify_anchor is not None:
            simplified_mesh.vertices = (
                (simplified_mesh.vertices - simplify_anchor[None, :]) / simplify_proxy_scale
                + simplify_anchor[None, :]
            )
        print(simplified_mesh.vertices.min(), simplified_mesh.vertices.max())

        return mesh_trimesh, simplified_mesh, fx_pixels, fov_x_deg, fov_x_rad
