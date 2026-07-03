import torch
from PIL import Image
import math
import os
import sys
import gc
import json
from pathlib import Path

os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "stdlib")

REPO_ROOT = Path(__file__).resolve().parents[2]
SUBMODULE_PATHS = [
    REPO_ROOT,
    REPO_ROOT / "submodules" / "flux_controlnet_inpainting",
]
for path in SUBMODULE_PATHS:
    if path.exists() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
if "bool" not in np.__dict__:
    np.bool = bool
from einops import rearrange
import trimesh
import cv2
from moge.model.v1 import MoGeModel
from typing import NamedTuple, Sequence, Union

from torchvision import utils as torchvision_utils
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage
from pytorch3d.renderer.blending import BlendParams, softmax_rgb_blend, hard_rgb_blend
import torch.nn as nn
import matplotlib.pyplot as plt

from pytorch3d.structures import Pointclouds, Meshes
from pytorch3d.renderer import (
    PointsRenderer, PointsRasterizer, PointsRasterizationSettings, AlphaCompositor,
    MeshRenderer, MeshRasterizer, RasterizationSettings, SoftPhongShader,
    PerspectiveCameras, BlendParams, PointLights, TexturesVertex, mesh, HardFlatShader, Textures, NormWeightedCompositor
)
from kornia.geometry import PinholeCamera

from simulation.image23D.segmenter import RepViTSegmenter, SegmentAnythingSegmenter
from simulation.image23D.mesh_generator import Sam3DMeshGenerator
from simulation.image23D.inpainter import FluxInpainter

from pytorch3d.renderer.mesh.textures import TexturesVertex

from simulation.utils import (
    soft_stitching,
    dilate_binary_mask,
    extract_foreground_depth_torch,
    save_point_cloud_as_ply,
    save_depth_map,
    save_mask_kps,
    remove_isolated_areas,
    render_mesh_with_occlusion_detection,
    create_occluded_submesh,
    pt3d_to_gs,
    # sample_mesh_surface,
    # match_color_style,
)

class HardShader(nn.Module):
    def __init__(self, device="cpu", cameras=None, blend_params=None):
        super().__init__()
        self.cameras = cameras
        self.blend_params = (
            blend_params if blend_params is not None else MyBlendParams()
        )

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of TexturedSoftPhongShader"
            raise ValueError(msg)
        # get renderer output
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        # images = softmax_rgb_blend(texels, fragments, blend_params)
        images = hard_rgb_blend(texels, fragments, blend_params)

        return images

class MyBlendParams(NamedTuple):
    """
    Data class to store blending params with defaults

    Members:
        sigma (float): For SoftmaxPhong, controls the width of the sigmoid
            function used to calculate the 2D distance based probability. Determines
            the sharpness of the edges of the shape. Higher => faces have less defined
            edges. For SplatterPhong, this is the standard deviation of the Gaussian
            kernel. Higher => splats have a stronger effect and the rendered image is
            more blurry.
        gamma (float): Controls the scaling of the exponential function used
            to set the opacity of the color.
            Higher => faces are more transparent.
        background_color: RGB values for the background color as a tuple or
            as a tensor of three floats.
    """

    sigma: float = 1e-4
    gamma: float = 1e-4
    background_color: Union[torch.Tensor, Sequence[float]] = (0.0, 0.0, 0.0)


# pytorch3d space
class SingleViewReconstructor(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.target_size = (512, 512)
        self.config = config
        self.device = config['device']
        
        self.input_image_pil = Image.open(os.path.join(config['data_path'], 'input.png')).convert('RGB')
        self.input_image = ToTensor()(self.input_image_pil).to(self.device)
        self.output_folder = Path(config['output_folder']) / 'render'
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.output_folder_frames = self.output_folder / 'frames'
        self.output_folder_frames.mkdir(parents=True, exist_ok=True)
        self.output_folder_masks = self.output_folder / 'masks'
        self.output_folder_masks.mkdir(parents=True, exist_ok=True)
        self.output_folder_optical_flow = self.output_folder / 'optical_flow'
        self.output_folder_optical_flow.mkdir(parents=True, exist_ok=True)

        self.target_size = self.input_image_pil.size
        
        self.previous_frame_data = None
        self.optical_flow = np.array([])

        self.franka_mesh = None
        self.merge_mask = True if 'merge_mask' in self.config and self.config['merge_mask'] else False

        self.fg_objects = []
        self.support_object_masks = []
        self.static_collision_meshes = []
        self.cache_bg = None
        self.background_collision_normal = None
        self.background_plane_point = None
        self.bg_depth_map = None
        self.bg_valid_depth_mask = None
        self.background_collision_image = None
        self.background_collision_points = None
        self.background_collision_points_colors = None
        self.background_collision_depth_map = None
        self.background_collision_valid_depth_mask = None
        self.object_kp_alignment_sources = {}
        self.object_reconstruction_cameras = {}

    def _tensor_scalar_to_float(self, value):
        if torch.is_tensor(value):
            return float(value.detach().cpu().item())
        return float(value)

    def _fov_x_from_focal_length(self, focal_length_pixels):
        focal_length_pixels = max(float(focal_length_pixels), 1e-6)
        width = float(self.target_size[0])
        return math.degrees(2.0 * math.atan(width / (2.0 * focal_length_pixels)))

    def _setup_global_mesh_init_camera(self, mesh_init_records):
        focal_records = []
        for record in mesh_init_records:
            try:
                focal = self._tensor_scalar_to_float(record['focal_length_pixels'])
            except Exception:
                continue
            if np.isfinite(focal) and focal > 1e-6:
                focal_records.append((record['key'], focal))

        if not focal_records:
            fallback_fov = float(self.config.get('fov_x_input', 60.0))
            width = float(self.target_size[0])
            global_focal = width / (2.0 * math.tan(math.radians(fallback_fov) / 2.0))
            focal_records = [('fallback_config_fov_x_input', global_focal)]

        # mode = str(self.config.get('mesh_init_camera_mode', 'global_median')).strip().lower()
        mode = str(self.config.get('mesh_init_camera_mode', 'global_mean')).strip().lower()
        focal_values = [focal for _, focal in focal_records]
        if mode in {'fixed', 'fixed_focal', 'locked', 'locked_focal'}:
            configured_focal = self.config.get('global_mesh_init_focal_length_pixels', None)
            try:
                configured_focal = float(configured_focal)
            except (TypeError, ValueError):
                configured_focal = None
            if configured_focal is None or not np.isfinite(configured_focal) or configured_focal <= 1e-6:
                fallback_fov = float(self.config.get('fov_x_input', 60.0))
                width = float(self.target_size[0])
                configured_focal = width / (2.0 * math.tan(math.radians(fallback_fov) / 2.0))
            global_focal = configured_focal
            source = 'fixed_config_focal'
            resolved_mode = 'fixed_focal'
        elif mode in {'first', 'first_object', 'object_0'}:
            global_focal = focal_records[0][1]
            source = focal_records[0][0]
            resolved_mode = 'first_object'
        elif mode in {'mean', 'global_mean', 'average', 'global_average'}:
            global_focal = float(np.mean(focal_values))
            source = 'mean_of_sam3d_reconstruction_focals'
            resolved_mode = 'global_mean'
        else:
            global_focal = float(np.median(focal_values))
            source = 'median_of_sam3d_reconstruction_focals'
            resolved_mode = 'global_median'

        global_fov_x = self._fov_x_from_focal_length(global_focal)
        self.init_focal_length = torch.tensor(global_focal, device=self.device, dtype=torch.float32)
        self.current_camera = self.get_camera_at_origin(focal_length=self.init_focal_length)
        self.config['mesh_init_camera_mode'] = resolved_mode
        self.config['global_mesh_init_focal_length_pixels'] = global_focal
        self.config['global_mesh_init_fov_x_degrees'] = global_fov_x
        self.config['global_mesh_init_focal_source'] = source
        self.config['global_mesh_init_focal_candidates'] = [
            {'key': key, 'focal_length_pixels': focal}
            for key, focal in focal_records
        ]
        self.config['fov_x_input'] = global_fov_x

        for record in mesh_init_records:
            camera_info = self.object_reconstruction_cameras.get(record['key'])
            if camera_info is None:
                continue
            camera_info['used_for_global_camera_estimate'] = record['key'] in {key for key, _ in focal_records}
            camera_info['used_as_global_scene_camera'] = record['key'] == source
            camera_info['used_for_mesh_init_proxy'] = False
            camera_info['mesh_init_render_focal_length_pixels'] = global_focal
            camera_info['mesh_init_render_fov_x_degrees'] = global_fov_x

        print(
            "Using robust global mesh-init camera: "
            f"mode={resolved_mode}, focal={global_focal:.3f}px, "
            f"fov_x={global_fov_x:.3f}deg, candidates={len(focal_records)}"
        )
        return self.init_focal_length, torch.tensor(global_fov_x, device=self.device, dtype=torch.float32)

    def _static_support_points(self):
        for key in ('support_object_points', 'static_object_points', 'fixed_object_points'):
            points = self.config.get(key, None)
            if points:
                return points
        return []

    def _static_support_masks_idx(self, count):
        masks_idx = None
        for key in ('support_object_masks_idx', 'static_object_masks_idx', 'fixed_object_masks_idx'):
            if key in self.config:
                masks_idx = self.config.get(key)
                break
        if masks_idx is None:
            masks_idx = [0] * count
        elif isinstance(masks_idx, (list, tuple)):
            masks_idx = list(masks_idx)
        elif hasattr(masks_idx, '__iter__') and not isinstance(masks_idx, (str, bytes)):
            masks_idx = list(masks_idx)
        else:
            masks_idx = [masks_idx]
        if len(masks_idx) == 1 and count > 1:
            masks_idx = masks_idx * count
        elif len(masks_idx) < count:
            masks_idx = masks_idx + [0] * (count - len(masks_idx))
        elif len(masks_idx) > count:
            masks_idx = masks_idx[:count]
        self.config['support_object_masks_idx'] = masks_idx
        return masks_idx

    def _static_support_names(self, count):
        names = self.config.get('support_object_names', None)
        if names is None:
            names = self.config.get('static_object_names', None)
        if names is None:
            names = [f"support_object_{idx:02d}" for idx in range(count)]
        elif isinstance(names, str):
            names = [names]
        elif hasattr(names, '__iter__'):
            names = list(names)
        else:
            names = [str(names)]
        if len(names) < count:
            names.extend(f"support_object_{idx:02d}" for idx in range(len(names), count))
        return names[:count]

    def _debug_suffix(self, idx):
        if isinstance(idx, (int, np.integer)):
            return f"{int(idx):02d}"
        return str(idx)

    def _safe_np_normalize(self, vector, default=None):
        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm < 1e-8:
            if default is None:
                return None
            return np.asarray(default, dtype=np.float32)
        return vector / norm

    def _fit_plane_normal_np(self, points, up_hint=None):
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[0] < 3:
            return self._safe_np_normalize(up_hint, default=[0.0, 0.0, 1.0])
        centered = points - points.mean(axis=0, keepdims=True)
        try:
            _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return self._safe_np_normalize(up_hint, default=[0.0, 0.0, 1.0])
        if singular_values[0] < 1e-8:
            return self._safe_np_normalize(up_hint, default=[0.0, 0.0, 1.0])
        normal = vh[-1]
        normal = self._safe_np_normalize(normal, default=up_hint if up_hint is not None else [0.0, 0.0, 1.0])
        if up_hint is not None and np.dot(normal, up_hint) < 0:
            normal = -normal
        return normal

    def _estimate_static_support_plane(self, vertices_list, object_centers=None, up_hint=None):
        if not vertices_list:
            return None, None

        support_vertices = np.concatenate(vertices_list, axis=0).astype(np.float32)
        if support_vertices.shape[0] < 3:
            return None, None

        up_hint = np.asarray(up_hint if up_hint is not None else [0.0, 0.0, 1.0], dtype=np.float32)
        normal = self._fit_plane_normal_np(support_vertices, up_hint=up_hint)
        if np.dot(normal, up_hint) < 0:
            normal = -normal
        if object_centers:
            object_centers = np.asarray(object_centers, dtype=np.float32)
            object_side = object_centers.mean(axis=0) - support_vertices.mean(axis=0)
            if np.dot(normal, object_side) < 0:
                normal = -normal

        height_quantile = float(self.config.get('static_support_plane_height_quantile', 0.5))
        height_quantile = float(np.clip(height_quantile, 0.0, 1.0))
        support_heights = support_vertices @ normal
        support_height = float(np.quantile(support_heights, height_quantile))
        point = support_vertices.mean(axis=0)
        point = point + normal * (support_height - float(point @ normal))
        return normal, point

    def _depth_edge_penalty(self):
        if not hasattr(self, 'input_image_points'):
            return None
        try:
            input_points = self.input_image_points.detach().cpu().numpy().reshape(
                self.target_size[1],
                self.target_size[0],
                3,
            )
        except Exception:
            return None

        depth = input_points[..., 2].astype(np.float32)
        finite = np.isfinite(depth)
        if not finite.any():
            return None
        fill_value = float(np.nanmedian(depth[finite]))
        depth = np.where(finite, depth, fill_value)
        grad_x = np.zeros_like(depth)
        grad_y = np.zeros_like(depth)
        grad_x[:, 1:] = np.abs(depth[:, 1:] - depth[:, :-1])
        grad_y[1:, :] = np.abs(depth[1:, :] - depth[:-1, :])
        return np.maximum(grad_x, grad_y)

    def _select_auto_stable_kps(self, mask):
        mask_np = (mask.detach().cpu().numpy() != 0).astype(np.uint8)
        candidate_coords = np.argwhere(mask_np > 0)
        if candidate_coords.shape[0] == 0:
            raise ValueError("Cannot sample keypoints from an empty mask.")

        target_count = 4

        distance_map = cv2.distanceTransform(mask_np, cv2.DIST_L2, 5).astype(np.float32)
        margin_ratio = 0.2
        min_margin_px = 8.0
        max_margin_px = 64.0
        margin_px = np.clip(distance_map.max() * margin_ratio, min_margin_px, max_margin_px)

        stable_mask = distance_map >= margin_px
        if stable_mask.sum() < target_count:
            stable_mask = distance_map >= max(min_margin_px, margin_px * 0.7)
        if stable_mask.sum() < target_count:
            stable_mask = mask_np > 0

        candidate_coords = np.argwhere(stable_mask)
        candidate_scores = distance_map[stable_mask].astype(np.float32)

        edge_penalty = self._depth_edge_penalty()
        if edge_penalty is not None and candidate_coords.shape[0] > 0:
            depth_weight = 0.35
            penalty_values = edge_penalty[stable_mask]
            finite_penalty = np.isfinite(penalty_values)
            if finite_penalty.any():
                penalty_values = np.where(finite_penalty, penalty_values, penalty_values[finite_penalty].max())
                norm = np.quantile(penalty_values, 0.8)
                if norm > 1e-8:
                    candidate_scores = candidate_scores - depth_weight * np.clip(penalty_values / norm, 0.0, 1.0) * max(distance_map.max(), 1.0)

        min_distance_px = max(6.0, margin_px)
        separation_weight = max(1.0, margin_px * 0.35)
        selected = []
        selected_mask = np.zeros(candidate_coords.shape[0], dtype=bool)
        min_distance_sq = np.full(candidate_coords.shape[0], np.inf, dtype=np.float32)

        for _ in range(target_count):
            if selected:
                last = np.asarray(selected[-1], dtype=np.float32)
                delta = candidate_coords.astype(np.float32) - last[None, :]
                min_distance_sq = np.minimum(min_distance_sq, np.sum(delta * delta, axis=1))
            combined_score = candidate_scores.copy()
            if np.isfinite(min_distance_sq).any():
                combined_score = combined_score + separation_weight * np.sqrt(np.maximum(min_distance_sq, 0.0))
                combined_score[min_distance_sq < (min_distance_px ** 2)] -= 1e6
            combined_score[selected_mask] = -np.inf
            best_idx = int(np.argmax(combined_score))
            if not np.isfinite(combined_score[best_idx]):
                remaining = np.where(~selected_mask)[0]
                if remaining.shape[0] == 0:
                    break
                best_idx = int(remaining[np.argmax(candidate_scores[remaining])])
            selected_mask[best_idx] = True
            selected.append(candidate_coords[best_idx])

        selected = np.asarray(selected, dtype=np.int64)
        if selected.shape[0] == 0:
            selected = np.asarray([candidate_coords[np.argmax(candidate_scores)]], dtype=np.int64)
        # Keep correspondences stable between the image mask and rendered mesh mask.
        # The greedy selection order can differ across masks, which may produce a
        # negative alignment scale and mirror the reconstructed object.
        order = np.lexsort((selected[:, 0], selected[:, 1]))
        selected = selected[order]
        return (
            torch.from_numpy(selected[:, 0]).to(self.device, dtype=torch.long),
            torch.from_numpy(selected[:, 1]).to(self.device, dtype=torch.long),
        )

    @torch.no_grad()
    def reconstruct(self):
        target_size = (self.input_image_pil.size[0], self.input_image_pil.size[1])
        sequential_object_inpainting = bool(
            self.config.get(
                'sequential_object_inpainting',
                self.config.get('object_occlusion_aware_inpainting', False),
            )
        )
        sequential_inpainted_image_pil = None
        object_alignment_images_pil = [self.input_image_pil]

        def normalize_indices(indices, count):
            if indices is None:
                normalized = [0] * count
            elif isinstance(indices, (list, tuple)):
                normalized = list(indices)
            elif hasattr(indices, '__iter__') and not isinstance(indices, (str, bytes)):
                normalized = list(indices)
            else:
                normalized = [indices]
            if len(normalized) == 1 and count > 1:
                normalized = normalized * count
            elif len(normalized) < count:
                normalized = normalized + [0] * (count - len(normalized))
            elif len(normalized) > count:
                normalized = normalized[:count]
            return normalized

        if 'segmenter' not in self.config or self.config['segmenter'] == "repvit":
            self.object_id = self.config['object_id']
            self.segmenter = RepViTSegmenter(self.device)
            target_masks = self.segmenter(self.input_image_pil, target_class=self.object_id, merge_mask=self.merge_mask)
        elif self.config['segmenter'] == "sam2":
            self.segmenter = SegmentAnythingSegmenter(self.config, self.device)
            object_points = self.config.get('all_object_points', []) or []
            if sequential_object_inpainting and len(object_points) > 1:
                object_masks_idx = normalize_indices(
                    self.config.get('all_object_masks_idx', [0]),
                    len(object_points),
                )
                self.config['all_object_masks_idx'] = object_masks_idx
                target_masks = []
                current_image_pil = self.input_image_pil
                sequential_inpainter = None
                for object_idx, object_points_i in enumerate(object_points):
                    per_mask = self.segmenter.predict_masks(
                        current_image_pil,
                        [object_points_i],
                        [object_masks_idx[object_idx]],
                        input_prefix="input_points",
                        mask_prefix="object",
                        debug_index_offset=object_idx,
                    )[0]
                    target_masks.append(per_mask)
                    if sequential_inpainter is None:
                        sequential_inpainter = FluxInpainter(device=self.device)
                    current_image_tensor = ToTensor()(current_image_pil).to(self.device)
                    current_mask_tensor = torch.from_numpy(per_mask).to(self.device)
                    current_image_pil = sequential_inpainter(
                        current_image_tensor,
                        current_mask_tensor,
                        size=target_size,
                        prompt=self.config['inpainting_prompt'],
                        negative_prompt=self.config['inpainting_negative_prompt'],
                    )
                    if object_idx < len(object_points) - 1:
                        object_alignment_images_pil.append(current_image_pil)
                        current_image_pil.save(
                            os.path.join(
                                self.config['output_folder'],
                                f"object_alignment_image_{object_idx + 1:02d}.png",
                            )
                        )
                    del current_image_tensor, current_mask_tensor
                    torch.cuda.empty_cache()
                if sequential_inpainter is not None:
                    del sequential_inpainter
                    gc.collect()
                    torch.cuda.empty_cache()
                sequential_inpainted_image_pil = current_image_pil
                self.config['sequential_object_inpainting'] = True
            else:
                target_masks = self.segmenter(self.input_image_pil)
                self.config['sequential_object_inpainting'] = False
        else:
            raise ValueError(f"Invalid segmenter: {self.config['segmenter']}")

        self.object_masks = [torch.from_numpy(mask).to(self.device) for mask in target_masks]
        support_points = self._static_support_points()
        self.support_object_masks = []
        if support_points:
            if not isinstance(self.segmenter, SegmentAnythingSegmenter):
                print("Warning: static support objects require segmenter=sam2; skipping support object masks.")
            else:
                support_masks_idx = self._static_support_masks_idx(len(support_points))
                support_masks = self.segmenter.predict_masks(
                    self.input_image_pil,
                    support_points,
                    support_masks_idx,
                    input_prefix="support_input_points",
                    mask_prefix="support_object",
                )
                self.support_object_masks = [torch.from_numpy(mask).to(self.device) for mask in support_masks]

        if hasattr(self, 'segmenter'):
            del self.segmenter
            gc.collect()
            torch.cuda.empty_cache()

        remove_support_from_background = bool(
            self.config.get(
                'remove_support_from_background_inpainting',
                len(self.support_object_masks) > 0,
            )
        )
        background_collision_mode = str(self.config.get('background_collision_mode', '')).strip().lower()
        skip_background_collision_reconstruction = bool(
            self.support_object_masks
        ) and (
            self.config.get('static_support_replaces_background_collision', False)
            or background_collision_mode in {'static_support', 'static_support_only'}
        )
        self.config['remove_support_from_background_inpainting'] = remove_support_from_background
        visual_inpaint_masks = list(self.object_masks)
        support_inpaint_masks = list(self.support_object_masks) if remove_support_from_background and not skip_background_collision_reconstruction else []
        collision_inpaint_masks = visual_inpaint_masks + support_inpaint_masks
        has_separate_collision_inpainting = bool(support_inpaint_masks)
        self.config['skip_background_collision_reconstruction'] = skip_background_collision_reconstruction
        self.config['background_collision_uses_support_removed_inpainting'] = has_separate_collision_inpainting

        def union_masks(masks):
            union_mask = torch.zeros_like(self.object_masks[0], dtype=torch.bool)
            for mask in masks:
                union_mask = union_mask | mask.bool()
            return union_mask

        inpainted_image_path = self.config.get(
            'precomputed_inpainted_image_path',
            os.path.join(self.config['data_path'], 'inpainted.png'),
        )
        background_collision_inpainted_image_path = self.config.get(
            'precomputed_background_collision_inpainted_image_path',
            None,
        )
        force_regenerate_support_inpainting = bool(
            self.config.get(
                'force_regenerate_inpainted_with_support',
                has_separate_collision_inpainting,
            )
        )
        use_precomputed_inpainted = (
            os.path.exists(inpainted_image_path)
            and not bool(self.config.get('force_regenerate_visual_inpainted', False))
        )
        use_precomputed_background_collision_inpainted = (
            has_separate_collision_inpainting
            and background_collision_inpainted_image_path
            and os.path.exists(background_collision_inpainted_image_path)
            and not bool(self.config.get('force_regenerate_background_collision_inpainted', False))
        )
        self.config['force_regenerate_inpainted_with_support'] = force_regenerate_support_inpainting
        inpainter = None
        if sequential_inpainted_image_pil is not None:
            self.inpainted_image_pil = sequential_inpainted_image_pil
            self.inpainted_image = ToTensor()(self.inpainted_image_pil).to(self.device)
            visual_inpainting_mask = union_masks(visual_inpaint_masks)
            self.inpainted_image_pil.save(os.path.join(self.config['output_folder'], 'inpainted_image.png'))
            if self.config.get('debug', False):
                torchvision_utils.save_image(visual_inpainting_mask.float(), os.path.join(self.config['output_folder'], 'inpainter_masks.png'))
            self.config['visual_inpainting_source'] = 'sequential_object_inpainting'
        elif use_precomputed_inpainted:
            print(f"Using precomputed inpainted image: {inpainted_image_path}")
            self.inpainted_image_pil = Image.open(inpainted_image_path).convert('RGB')
            self.inpainted_image = ToTensor()(self.inpainted_image_pil).to(self.device)
            if self.config.get('debug', False):
                self.inpainted_image_pil.save(os.path.join(self.config['output_folder'], 'inpainted_image.png'))
        else:
            inpainter = FluxInpainter(device=self.device)
            visual_inpainting_mask = union_masks(visual_inpaint_masks)
            if self.config.get('debug', False):
                torchvision_utils.save_image(visual_inpainting_mask.float(), os.path.join(self.config['output_folder'], 'inpainter_masks.png'))

            self.inpainted_image_pil = inpainter(self.input_image, visual_inpainting_mask, size=target_size, prompt=self.config['inpainting_prompt'], negative_prompt=self.config['inpainting_negative_prompt'])
            self.inpainted_image = ToTensor()(self.inpainted_image_pil).to(self.device)
            self.inpainted_image_pil.save(os.path.join(self.config['output_folder'], 'inpainted_image.png'))

        if self.config.get('debug', False) and use_precomputed_inpainted:
            visual_inpainting_mask = union_masks(visual_inpaint_masks)
            torchvision_utils.save_image(visual_inpainting_mask.float(), os.path.join(self.config['output_folder'], 'inpainter_masks.png'))

        if has_separate_collision_inpainting:
            if use_precomputed_background_collision_inpainted:
                print(f"Using precomputed background collision inpainted image: {background_collision_inpainted_image_path}")
                self.background_collision_image_pil = Image.open(background_collision_inpainted_image_path).convert('RGB')
                self.background_collision_image = ToTensor()(self.background_collision_image_pil).to(self.device)
                if self.config.get('debug', False):
                    self.background_collision_image_pil.save(os.path.join(self.config['output_folder'], 'background_collision_inpainted_image.png'))
            else:
                if force_regenerate_support_inpainting:
                    print(
                        "Generating support-removed inpainting for background collision only; "
                        "visual background keeps support objects."
                    )
                if inpainter is None:
                    inpainter = FluxInpainter(device=self.device)
                support_inpainting_mask = union_masks(support_inpaint_masks)
                background_collision_inpainting_mask = union_masks(collision_inpaint_masks)
                if self.config.get('debug', False):
                    torchvision_utils.save_image(
                        support_inpainting_mask.float(),
                        os.path.join(self.config['output_folder'], 'support_inpainter_masks.png'),
                    )
                    torchvision_utils.save_image(
                        background_collision_inpainting_mask.float(),
                        os.path.join(self.config['output_folder'], 'background_collision_inpainter_masks.png'),
                    )
                self.background_collision_image_pil = inpainter(
                    self.input_image,
                    background_collision_inpainting_mask,
                    size=target_size,
                    prompt=self.config['inpainting_prompt'],
                    negative_prompt=self.config['inpainting_negative_prompt'],
                )
                self.background_collision_image = ToTensor()(self.background_collision_image_pil).to(self.device)
                self.background_collision_image_pil.save(os.path.join(self.config['output_folder'], 'background_collision_inpainted_image.png'))
            if self.config.get('debug', False) and use_precomputed_background_collision_inpainted:
                support_inpainting_mask = union_masks(support_inpaint_masks)
                background_collision_inpainting_mask = union_masks(collision_inpaint_masks)
                torchvision_utils.save_image(
                    support_inpainting_mask.float(),
                    os.path.join(self.config['output_folder'], 'support_inpainter_masks.png'),
                )
                torchvision_utils.save_image(
                    background_collision_inpainting_mask.float(),
                    os.path.join(self.config['output_folder'], 'background_collision_inpainter_masks.png'),
                )
        else:
            self.background_collision_image_pil = self.inpainted_image_pil
            self.background_collision_image = self.inpainted_image

        if inpainter is not None:
            del inpainter
            gc.collect()
            torch.cuda.empty_cache()

        if (
            'stitched_inpainting' in self.config
            and self.config['stitched_inpainting']
            and sequential_inpainted_image_pil is None
        ):
            # all_dilated_masks = [torch.from_numpy(dilate_binary_mask(per_mask, size=(512, 512), kernel_size=3, iterations=1)).unsqueeze(0).unsqueeze(0).to(self.device) for per_mask in self.object_masks]
            self.inpainted_image = soft_stitching(self.inpainted_image.unsqueeze(0), self.input_image.unsqueeze(0), [per_mask.unsqueeze(0).unsqueeze(0) for per_mask in visual_inpaint_masks]).squeeze(0)
            if has_separate_collision_inpainting:
                self.background_collision_image = soft_stitching(
                    self.background_collision_image.unsqueeze(0),
                    self.input_image.unsqueeze(0),
                    [per_mask.unsqueeze(0).unsqueeze(0) for per_mask in collision_inpaint_masks],
                ).squeeze(0)
            else:
                self.background_collision_image = self.inpainted_image
            # self.inpainted_image = soft_stitching(self.inpainted_image.unsqueeze(0), self.input_image.unsqueeze(0), all_dilated_masks).squeeze(0)
        
        if self.config.get('debug', False):
            torchvision_utils.save_image(self.inpainted_image, os.path.join(self.config['output_folder'], 'stitched_inpainted_image.png'))
            if has_separate_collision_inpainting:
                torchvision_utils.save_image(
                    self.background_collision_image,
                    os.path.join(self.config['output_folder'], 'stitched_background_collision_inpainted_image.png'),
                )
        
        reinitialize_mesh_generator_per_object = bool(
            self.config.get('reinitialize_mesh_generator_per_object', False)
        )
        self.mesh_generator = None if reinitialize_mesh_generator_per_object else Sam3DMeshGenerator(self.config)

        def run_mesh_generator_once(*args, **kwargs):
            mesh_generator = (
                Sam3DMeshGenerator(self.config)
                if reinitialize_mesh_generator_per_object
                else self.mesh_generator
            )
            try:
                return mesh_generator(*args, **kwargs)
            finally:
                if reinitialize_mesh_generator_per_object:
                    del mesh_generator
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    gc.collect()
                    torch.cuda.empty_cache()

        self.fg_meshes = []
        self.fg_pcs = []
        dynamic_mesh_init_records = []
        for idx, per_mask in enumerate(self.object_masks):
            if 'refine_mask' in self.config and self.config['refine_mask']:
                min_size = self.config['min_size'] if 'min_size' in self.config else 100
                per_mask = torch.from_numpy(remove_isolated_areas(per_mask.cpu().numpy(), min_size=min_size)).to(self.device)
                self.object_masks[idx] = per_mask
                if self.config.get('debug', False):
                    torchvision_utils.save_image(
                        per_mask.float(),
                        os.path.join(self.config['output_folder'], f"refined_mask_{idx:02d}.png")
                    )
            else:
                if self.config.get('debug', False):
                    torchvision_utils.save_image(
                        per_mask.float(),
                        os.path.join(self.config['output_folder'], f"mask_{idx:02d}.png")
                    )
                print(f"Refine mask is disabled, using original mask")

            print("GPU memory before Point Render:", torch.cuda.memory_allocated() / 1024**2, "MB")
            gc.collect()
            torch.cuda.empty_cache()
            original_mesh, simplified_mesh, fx_pixels, fx_deg, _ = run_mesh_generator_once(
                np.array(self.input_image_pil),
                per_mask.cpu().numpy(),
                mesh_resize_factor=self.config['mesh_resize_factor'],
                target_faces=self.config['target_faces'],
            )
            record_key = f"object_{idx:02d}"
            dynamic_mesh_init_records.append({
                'key': record_key,
                'kind': 'object',
                'idx': idx,
                'mask': per_mask,
                'original_mesh': original_mesh,
                'simplified_mesh': simplified_mesh,
                'focal_length_pixels': fx_pixels,
                'fov_x_degrees': fx_deg,
            })
            self.object_reconstruction_cameras[record_key] = {
                'focal_length_pixels': self._tensor_scalar_to_float(fx_pixels),
                'fov_x_degrees': self._tensor_scalar_to_float(fx_deg),
                'used_for_global_camera_estimate': True,
                'used_for_mesh_init_proxy': False,
                'used_as_global_scene_camera': False,
                'image_source': 'input_image',
            }
            gc.collect()
            torch.cuda.empty_cache()
            print("GPU memory before Point Render:", torch.cuda.memory_allocated() / 1024**2, "MB")

        self._static_support_mesh_init_records = []
        if self.support_object_masks:
            support_names = self._static_support_names(len(self.support_object_masks))
            use_inpainted_support_image = bool(
                self.config.get('support_reconstruction_use_inpainted_image', True)
            )
            has_inpainted_support_image = use_inpainted_support_image and hasattr(self, 'inpainted_image_pil')
            support_image_pil = self.inpainted_image_pil if has_inpainted_support_image else self.input_image_pil
            support_image_source = 'inpainted_dynamic_removed' if has_inpainted_support_image else 'input_image'
            self.config['support_reconstruction_image_source'] = support_image_source
            for support_idx, support_mask in enumerate(self.support_object_masks):
                print(f"Preparing static support mesh-init object {support_idx}: {support_names[support_idx]}")
                support_mesh_resize_factor = float(self.config.get('support_mesh_resize_factor', 0.3))
                support_world_resize_factor = float(self.config.get('support_world_resize_factor', 1.0))
                support_target_faces = int(self.config.get('support_target_faces', 5000))
                original_mesh, simplified_mesh, support_fx_pixels, support_fx_deg, _ = run_mesh_generator_once(
                    np.array(support_image_pil),
                    support_mask.cpu().numpy(),
                    mesh_resize_factor=support_world_resize_factor,
                    target_faces=support_target_faces,
                    simplify_proxy_scale=support_mesh_resize_factor,
                    voxel_pitch=self.config.get('support_voxel_pitch', self.config.get('mesh_voxel_pitch', 0.01)),
                    smoothing_iterations=self.config.get(
                        'support_smoothing_iterations',
                        self.config.get('mesh_smoothing_iterations', 0),
                    ),
                )
                record_key = f"support_{support_idx:02d}"
                support_record = {
                    'key': record_key,
                    'kind': 'support',
                    'idx': support_idx,
                    'mask': support_mask,
                    'name': support_names[support_idx],
                    'original_mesh': original_mesh,
                    'simplified_mesh': simplified_mesh,
                    'focal_length_pixels': support_fx_pixels,
                    'fov_x_degrees': support_fx_deg,
                    'image_source': support_image_source,
                    'mesh_resize_factor': support_mesh_resize_factor,
                    'world_resize_factor': support_world_resize_factor,
                    'target_faces': support_target_faces,
                }
                self._static_support_mesh_init_records.append(support_record)
                self.object_reconstruction_cameras[record_key] = {
                    'focal_length_pixels': self._tensor_scalar_to_float(support_fx_pixels),
                    'fov_x_degrees': self._tensor_scalar_to_float(support_fx_deg),
                    'used_for_global_camera_estimate': True,
                    'used_for_mesh_init_proxy': False,
                    'used_as_global_scene_camera': False,
                    'image_source': support_image_source,
                }

        _, scene_fov_x = self._setup_global_mesh_init_camera(
            dynamic_mesh_init_records + self._static_support_mesh_init_records
        )
        moge_model = None
        current_object_kp_points = None
        current_object_kp_source = 'original_input_depth'
        object_count = len(self.object_masks)
        object_kp_points_by_idx = [None] * object_count
        object_kp_source_by_idx = ['original_input_depth'] * object_count
        use_sequential_depth = bool(self.config.get('sequential_object_inpainting', False))

        def normalize_depth(depth_map, valid_mask):
            if valid_mask.any():
                max_val = depth_map[valid_mask].max()
                depth_map = depth_map.clone()
                depth_map[~valid_mask] = max_val
            if 'remap_depth' in self.config:
                depth_map = self.remap_depth(depth_map, self.config['remap_depth'], valid_mask)
            if 'flat_depth' in self.config:
                flat_depth = depth_map[~torch.isinf(depth_map)].mean().item()
                flat_depth = 1.99
                depth_map = torch.full_like(depth_map, fill_value=flat_depth)
            return depth_map

        def infer_depth_points(image_tensor, source_name, save_prefix=None):
            depth_map = moge_model.infer(image_tensor, fov_x=scene_fov_x)['depth']
            valid_mask = ~torch.isinf(depth_map)
            depth_map = normalize_depth(depth_map, valid_mask)
            if self.config.get('debug', False) and save_prefix:
                save_depth_map(
                    depth_map.cpu().numpy(),
                    os.path.join(self.config['output_folder'], f"{save_prefix}.png"),
                )
            points, colors = self.depth2pc(depth_map, image_tensor)
            return points, colors, depth_map, valid_mask, source_name
        for record in dynamic_mesh_init_records:
            idx = record['idx']
            per_mask = record['mask']
            original_mesh = record['original_mesh']
            simplified_mesh = record['simplified_mesh']
            fx_pixels = record['focal_length_pixels']
            fx_deg = record['fov_x_degrees']
            if idx == 0:
                # background point cloud
                moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(self.device)
                moge_model.eval()
                with torch.no_grad():
                    depth_inpainted = moge_model.infer(self.inpainted_image, fov_x=scene_fov_x)['depth']
                    if self.config.get('background_collision_uses_support_removed_inpainting', False):
                        depth_background_collision = moge_model.infer(self.background_collision_image, fov_x=scene_fov_x)['depth']
                    else:
                        depth_background_collision = depth_inpainted
                    depth_input = moge_model.infer(self.input_image, fov_x=scene_fov_x)['depth']
                    mask_noninf_inpainted = ~torch.isinf(depth_inpainted)
                    mask_noninf_background_collision = ~torch.isinf(depth_background_collision)
                    mask_noninf_input = ~torch.isinf(depth_input)

                    if mask_noninf_inpainted.any():
                        max_val_inpainted = depth_inpainted[mask_noninf_inpainted].max()
                        depth_inpainted = depth_inpainted.clone()
                        depth_inpainted[~mask_noninf_inpainted] = max_val_inpainted
                    if mask_noninf_background_collision.any():
                        max_val_background_collision = depth_background_collision[mask_noninf_background_collision].max()
                        depth_background_collision = depth_background_collision.clone()
                        depth_background_collision[~mask_noninf_background_collision] = max_val_background_collision
                    if mask_noninf_input.any():
                        max_val_input = depth_input[mask_noninf_input].max()
                        depth_input = depth_input.clone()
                        depth_input[~mask_noninf_input] = max_val_input
                    if 'remap_depth' in self.config:
                        depth_inpainted = self.remap_depth(depth_inpainted, self.config['remap_depth'], mask_noninf_inpainted)
                        depth_background_collision = self.remap_depth(depth_background_collision, self.config['remap_depth'], mask_noninf_background_collision)
                        depth_input = self.remap_depth(depth_input, self.config['remap_depth'], mask_noninf_input)

                    if 'flat_depth' in self.config:
                        flat_depth_inpainted = depth_inpainted[~torch.isinf(depth_inpainted)].mean().item()
                        flat_depth_inpainted = 1.99
                        # flat_depth_input = depth_input[~torch.isinf(depth_input)].mean().item()
                        depth_inpainted = torch.full_like(depth_inpainted, fill_value=flat_depth_inpainted)
                        depth_background_collision = torch.full_like(depth_background_collision, fill_value=flat_depth_inpainted)
                        # depth_input = torch.full_like(depth_input, fill_value=flat_depth_input)
                
                if self.config.get('debug', False):
                    save_depth_map(depth_inpainted.cpu().numpy(), os.path.join(self.config['output_folder'], f"depth_inpainted_{idx:02d}.png"))
                    if self.config.get('background_collision_uses_support_removed_inpainting', False):
                        save_depth_map(depth_background_collision.cpu().numpy(), os.path.join(self.config['output_folder'], f"depth_background_collision_{idx:02d}.png"))
                    save_depth_map(depth_input.cpu().numpy(), os.path.join(self.config['output_folder'], f"depth_input_{idx:02d}.png"))
                
                self.bg_points, self.bg_points_colors = self.depth2pc(depth_inpainted, self.inpainted_image)
                self.background_collision_points, self.background_collision_points_colors = self.depth2pc(
                    depth_background_collision,
                    self.background_collision_image,
                )
                self.input_image_points, self.input_image_colors = self.depth2pc(depth_input, self.input_image)
                self.bg_depth_map = depth_inpainted.detach().clone()
                self.bg_valid_depth_mask = mask_noninf_inpainted.detach().clone()
                self.background_collision_depth_map = depth_background_collision.detach().clone()
                self.background_collision_valid_depth_mask = mask_noninf_background_collision.detach().clone()
                object_kp_points_by_idx[0] = self.input_image_points
                object_kp_source_by_idx[0] = 'original_input_depth'
                if use_sequential_depth:
                    for align_idx in range(1, object_count):
                        if align_idx >= len(object_alignment_images_pil):
                            continue
                        alignment_image = ToTensor()(object_alignment_images_pil[align_idx]).to(self.device)
                        with torch.no_grad():
                            (
                                object_kp_points_by_idx[align_idx],
                                _,
                                _,
                                _,
                                object_kp_source_by_idx[align_idx],
                            ) = infer_depth_points(
                                alignment_image,
                                f"dynamic_removed_through_object_{align_idx - 1:02d}",
                                save_prefix=f"depth_object_alignment_{align_idx:02d}",
                            )
                        del alignment_image
                del moge_model
                moge_model = None
                gc.collect()
                torch.cuda.empty_cache()
                current_object_kp_points = (
                    object_kp_points_by_idx[idx]
                    if object_kp_points_by_idx[idx] is not None
                    else self.input_image_points
                )
                current_object_kp_source = object_kp_source_by_idx[idx]

            if idx > 0:
                current_object_kp_points = (
                    object_kp_points_by_idx[idx]
                    if object_kp_points_by_idx[idx] is not None
                    else self.input_image_points
                )
                current_object_kp_source = object_kp_source_by_idx[idx]

            if 'obj_kp_matching' in self.config and self.config['obj_kp_matching']:
                # scale = 1.0
                # translation = np.array([-0.01, 0.08, 0.0])
                # simplified_mesh.vertices = simplified_mesh.vertices * scale + translation
                # original_mesh.vertices = original_mesh.vertices * scale + translation

                alignment_points = current_object_kp_points if current_object_kp_points is not None else self.input_image_points
                self.object_kp_alignment_sources[f"object_{idx:02d}"] = current_object_kp_source
                scale, translation = self.obj_kp_matching(
                    per_mask,
                    torch.from_numpy(original_mesh.vertices).to(self.device).float(),
                    torch.from_numpy(original_mesh.faces).to(self.device).long(),
                    idx,
                    unprojected_points=alignment_points,
                    render_camera=self.current_camera,
                    render_focal_length=self.init_focal_length,
                )
                simplified_mesh.vertices = simplified_mesh.vertices * scale.item() + translation.cpu().numpy()
                original_mesh.vertices = original_mesh.vertices * scale.item() + translation.cpu().numpy()
            

            # 修复：获取真实高宽，并作为参数传给它
            target_h, target_w = self.target_size[1], self.target_size[0]
            per_mask_from_mesh, depth_map, occluded_vertices_mask = render_mesh_with_occlusion_detection(
                torch.from_numpy(original_mesh.vertices).to(self.device).float(), 
                torch.from_numpy(original_mesh.faces).to(self.device).long(), 
                torch.from_numpy(original_mesh.visual.vertex_colors).to(self.device).float()[:,:3]/255.0, 
                self.current_camera,
                image_size=(target_h, target_w)
            )
            occluded_submesh_vertices, occluded_submesh_faces, occluded_submesh_colors = create_occluded_submesh(torch.from_numpy(original_mesh.vertices).to(self.device).float(), torch.from_numpy(original_mesh.faces).to(self.device).long(), torch.from_numpy(original_mesh.visual.vertex_colors).to(self.device).float()[:,:3]/255.0, occluded_vertices_mask)
            per_points, per_colors = self.depth2pc(depth_map, self.input_image, per_mask_from_mesh)

            # occluded_submesh_colors[:] = torch.tensor([216/255.0, 190/255.0, 150/255.0], device=occluded_submesh_colors.device).unsqueeze(0).expand_as(occluded_submesh_colors)

            if self.config.get('use_rgb_frontside', True):
                merged_per_points = torch.cat([per_points, occluded_submesh_vertices], dim=0)
                merged_per_colors = torch.cat([per_colors, occluded_submesh_colors], dim=0)
            else:
                merged_per_points = torch.from_numpy(original_mesh.vertices).to(self.device).float()
                merged_per_colors = torch.from_numpy(original_mesh.visual.vertex_colors).to(self.device).float()[:,:3]/255.0

            self.fg_meshes.append(
                {
                    'vertices': torch.from_numpy(simplified_mesh.vertices).to(self.device).float(),
                    'faces': torch.from_numpy(simplified_mesh.faces).to(self.device).long(),
                    'colors': torch.from_numpy(simplified_mesh.visual.vertex_colors).to(self.device).float()[:,:3]/255.0
                }
            )
            # self.fg_meshes.append(
            #     {
            #         'vertices': torch.from_numpy(simplified_mesh.vertices).float().cpu(),
            #         'faces': torch.from_numpy(simplified_mesh.faces).long().cpu(),
            #         'colors': torch.from_numpy(simplified_mesh.visual.vertex_colors).float()[:,:3].cpu()/255.0
            #     }
            # )

            self.fg_pcs.append(
                # {
                #     'points': torch.from_numpy(original_mesh.vertices).to(self.device).float(),
                #     'colors': torch.from_numpy(original_mesh.visual.vertex_colors).to(self.device).float()[:,:3]/255.0
                # }
                {
                    'points': merged_per_points,
                    'colors': merged_per_colors
                }
            )

            if self.config.get('debug', False):
                save_point_cloud_as_ply(
                    merged_per_points.cpu(),
                    merged_per_colors.cpu(),
                    os.path.join(self.config['output_folder'], f"merged_per_points_{idx:02d}.ply")
                )

            if self.config.get('debug', False):
                original_mesh.export(os.path.join(self.config['output_folder'], f"sam3d_mesh_{idx:02d}.obj"))
                simplified_mesh.export(os.path.join(self.config['output_folder'], f"sam3d_mesh_{idx:02d}_simplified.obj"))

            # [新增] 主动删除当前循环产生的大块无用变量
            record['original_mesh'] = None
            record['simplified_mesh'] = None
            del original_mesh, simplified_mesh
            del per_mask_from_mesh, depth_map, occluded_vertices_mask
            del occluded_submesh_vertices, occluded_submesh_faces, occluded_submesh_colors
            del per_points, per_colors, merged_per_points, merged_per_colors
            
            # [新增] 清空 PyTorch 的显存缓存
            torch.cuda.empty_cache()

        self.config['object_kp_alignment_sources'] = self.object_kp_alignment_sources
        self.config['object_reconstruction_cameras'] = self.object_reconstruction_cameras
        if moge_model is not None:
            del moge_model
            gc.collect()
            torch.cuda.empty_cache()

        if self.support_object_masks:
            self.reconstruct_static_support_objects()
            self.config['object_kp_alignment_sources'] = self.object_kp_alignment_sources
            self.config['object_reconstruction_cameras'] = self.object_reconstruction_cameras

        if self.config.get('skip_background_collision_reconstruction', False):
            print("Skipping reconstructed background collision because static support meshes are used instead.")
        elif self.background_collision_depth_map is not None:
            self.estimate_background_collision_plane(
                self.background_collision_depth_map,
                self.background_collision_valid_depth_mask,
            )
            self.create_background_collision_mesh(
                self.background_collision_depth_map,
                self.background_collision_valid_depth_mask,
            )

        if self.config.get('debug', False):
            save_point_cloud_as_ply(self.bg_points, self.bg_points_colors, os.path.join(self.config['output_folder'], 'projected_bg_points.ply'))
            if self.background_collision_points is not None:
                save_point_cloud_as_ply(
                    self.background_collision_points,
                    self.background_collision_points_colors,
                    os.path.join(self.config['output_folder'], 'projected_background_collision_points.ply'),
                )

        # self.render(render_bg=True, render_obj=True, render_mesh=True, frame_id=0, save=True, mask=True)
        # import pdb; pdb.set_trace()

        self.num_fg_objects = len(self.fg_pcs)
        
        self.ground_plane_normal = None

        if 'estimate_plane' in self.config and self.config['estimate_plane']:
            self.ground_plane_normal = self.estimate_plane_normal_simple(self.fg_pcs[-1]['points'].cpu().numpy())
            if self.ground_plane_normal[1] < 0:
                self.ground_plane_normal = -self.ground_plane_normal
            self.fg_pcs = self.fg_pcs[:-1]
            self.fg_meshes = self.fg_meshes[:-1]

        return self.fg_pcs, self.fg_meshes, self.ground_plane_normal, self.config
        
    # def depth2pc(self, depth_map, image, mask=None):
    #     # initialize the point cloud for background
    #     kf_camera = self.convert_pytorch3d_kornia(self.current_camera, self.init_focal_length)
    #     point_depth = rearrange(depth_map.unsqueeze(0), "c h w -> (w h) c")
    #     # Set all inf values in point_depth to 6
    #     # point_depth[point_depth == float('inf')] = 6

    #     x = torch.arange(self.target_size[0]).float() + 0.5
    #     y = torch.arange(self.target_size[1]).float() + 0.5

    #     points_cloud = torch.stack(torch.meshgrid(x, y, indexing="ij"), -1)
    #     points_cloud = rearrange(points_cloud, "h w c -> (h w) c").to(self.device)

    #     unprojected_points = kf_camera.unproject(points_cloud, point_depth)
    #     points_colors = rearrange(image, "c h w -> (w h) c")

    #     if mask is not None:
    #         mask = rearrange(mask, "h w -> (w h)")
    #         unprojected_points = unprojected_points[mask]
    #         points_colors = points_colors[mask]

    #     return unprojected_points, points_colors
    def depth2pc(self, depth_map, image, mask=None):
            # 动态获取深度图的 Height 和 Width
            H, W = depth_map.shape
            kf_camera = self.convert_pytorch3d_kornia(self.current_camera, self.init_focal_length)
            
            # 修复：必须使用 (h w) 展平，以匹配图片实际的行优先排布
            point_depth = rearrange(depth_map.unsqueeze(0), "c h w -> (h w) c")

            # 生成对应实际宽高的坐标
            x = torch.arange(W).float() + 0.5
            y = torch.arange(H).float() + 0.5

            # 修复：使用 indexing="xy" 保证生成的网格 shape 为 (H, W)
            grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
            points_cloud = torch.stack([grid_x, grid_y], dim=-1).to(self.device) # shape: (H, W, 2)
            points_cloud = rearrange(points_cloud, "h w c -> (h w) c")

            unprojected_points = kf_camera.unproject(points_cloud, point_depth)
            
            # 修复：图片颜色展平也要改为 (h w)
            points_colors = rearrange(image, "c h w -> (h w) c")

            if mask is not None:
                mask = rearrange(mask, "h w -> (h w)")
                unprojected_points = unprojected_points[mask]
                points_colors = points_colors[mask]

            return unprojected_points, points_colors

    def reconstruct_static_support_objects(self):
        support_names = self._static_support_names(len(self.support_object_masks))
        support_collision_objects = []
        self.static_collision_meshes = []
        support_vertices_pt3d = []
        support_vertices_gs = []
        output_folder = Path(self.config['output_folder'])
        output_folder.mkdir(parents=True, exist_ok=True)
        use_inpainted_support_image = bool(
            self.config.get('support_reconstruction_use_inpainted_image', True)
        )
        has_inpainted_support_image = use_inpainted_support_image and hasattr(self, 'inpainted_image_pil')
        support_image_pil = self.inpainted_image_pil if has_inpainted_support_image else self.input_image_pil
        support_image_source = 'inpainted_dynamic_removed' if has_inpainted_support_image else 'input_image'
        self.config['support_reconstruction_image_source'] = support_image_source
        cached_support_records = getattr(self, '_static_support_mesh_init_records', [])

        for support_idx, support_mask in enumerate(self.support_object_masks):
            if self.config.get('debug', False):
                torchvision_utils.save_image(
                    support_mask.float(),
                    output_folder / f"support_mask_{support_idx:02d}.png",
                )

            print(f"Reconstructing static support object {support_idx}: {support_names[support_idx]}")
            cached_record = cached_support_records[support_idx] if support_idx < len(cached_support_records) else None
            if cached_record is not None and cached_record.get('original_mesh') is not None:
                original_mesh = cached_record['original_mesh']
                simplified_mesh = cached_record['simplified_mesh']
                support_fx_pixels = cached_record['focal_length_pixels']
                support_fx_deg = cached_record['fov_x_degrees']
                support_image_source = cached_record.get('image_source', support_image_source)
                support_mesh_resize_factor = cached_record.get(
                    'mesh_resize_factor',
                    float(self.config.get('support_mesh_resize_factor', 0.3)),
                )
                support_world_resize_factor = cached_record.get(
                    'world_resize_factor',
                    float(self.config.get('support_world_resize_factor', 1.0)),
                )
                support_target_faces = cached_record.get(
                    'target_faces',
                    int(self.config.get('support_target_faces', 5000)),
                )
            else:
                support_mesh_resize_factor = float(self.config.get('support_mesh_resize_factor', 0.3))
                support_world_resize_factor = float(self.config.get('support_world_resize_factor', 1.0))
                support_target_faces = int(self.config.get('support_target_faces', 5000))
                mesh_generator = self.mesh_generator
                created_mesh_generator = False
                if mesh_generator is None:
                    mesh_generator = Sam3DMeshGenerator(self.config)
                    created_mesh_generator = True
                try:
                    original_mesh, simplified_mesh, support_fx_pixels, support_fx_deg, _ = mesh_generator(
                        np.array(support_image_pil),
                        support_mask.cpu().numpy(),
                        mesh_resize_factor=support_world_resize_factor,
                        target_faces=support_target_faces,
                        simplify_proxy_scale=support_mesh_resize_factor,
                        voxel_pitch=self.config.get('support_voxel_pitch', self.config.get('mesh_voxel_pitch', 0.01)),
                        smoothing_iterations=self.config.get(
                            'support_smoothing_iterations',
                            self.config.get('mesh_smoothing_iterations', 0),
                        ),
                    )
                finally:
                    if created_mesh_generator:
                        del mesh_generator
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                        gc.collect()
                        torch.cuda.empty_cache()
                record_key = f"support_{support_idx:02d}"
                self.object_reconstruction_cameras.setdefault(record_key, {
                    'focal_length_pixels': self._tensor_scalar_to_float(support_fx_pixels),
                    'fov_x_degrees': self._tensor_scalar_to_float(support_fx_deg),
                    'used_for_global_camera_estimate': False,
                    'used_for_mesh_init_proxy': False,
                    'used_as_global_scene_camera': False,
                    'image_source': support_image_source,
                })

            support_kp_matching = self.config.get(
                'support_obj_kp_matching',
                self.config.get('obj_kp_matching', False),
            )
            if support_kp_matching and hasattr(self, 'input_image_points'):
                debug_idx = f"support_{support_idx:02d}"
                support_alignment_points = self.bg_points if self.bg_points is not None else self.input_image_points
                self.object_kp_alignment_sources[f"support_{support_idx:02d}"] = (
                    'dynamic_removed_background_depth'
                    if self.bg_points is not None
                    else 'original_input_depth'
                )
                scale, translation = self.obj_kp_matching(
                    support_mask,
                    torch.from_numpy(original_mesh.vertices).to(self.device).float(),
                    torch.from_numpy(original_mesh.faces).to(self.device).long(),
                    debug_idx,
                    kp_key='support_obj_kp',
                    unprojected_points=support_alignment_points,
                    render_camera=self.current_camera,
                    render_focal_length=self.init_focal_length,
                )
                simplified_mesh.vertices = simplified_mesh.vertices * scale.item() + translation.cpu().numpy()
                original_mesh.vertices = original_mesh.vertices * scale.item() + translation.cpu().numpy()

            if self.config.get('debug', False):
                original_mesh.export(output_folder / f"support_object_mesh_{support_idx:02d}.obj")
                simplified_mesh.export(output_folder / f"support_object_mesh_{support_idx:02d}_simplified.obj")

            pt3d_path = output_folder / f"static_support_mesh_{support_idx:02d}_pt3d.obj"
            gs_path = output_folder / f"static_support_mesh_{support_idx:02d}_gs.obj"
            support_collision_mesh = simplified_mesh.copy()
            support_collision_mesh.export(pt3d_path)
            support_vertices_pt3d.append(np.asarray(support_collision_mesh.vertices, dtype=np.float32))

            vertices_gs = pt3d_to_gs(np.asarray(support_collision_mesh.vertices, dtype=np.float32))
            mesh_gs = trimesh.Trimesh(
                vertices=vertices_gs,
                faces=np.asarray(support_collision_mesh.faces),
                process=False,
            )
            try:
                mesh_gs.visual.vertex_colors = np.asarray(support_collision_mesh.visual.vertex_colors)
            except Exception:
                pass
            mesh_gs.export(gs_path)
            support_vertices_gs.append(vertices_gs)

            bounds_min = vertices_gs.min(axis=0)
            bounds_max = vertices_gs.max(axis=0)
            support_info = {
                'object_index': support_idx,
                'name': support_names[support_idx],
                'mesh_path_pt3d': str(pt3d_path),
                'mesh_path_gs': str(gs_path),
                'bounds_gs': {
                    'min': bounds_min.tolist(),
                    'max': bounds_max.tolist(),
                },
                'vertex_count': int(mesh_gs.vertices.shape[0]),
                'face_count': int(mesh_gs.faces.shape[0]),
                'mesh_resize_factor': support_mesh_resize_factor,
                'world_resize_factor': support_world_resize_factor,
                'target_faces': support_target_faces,
                'fov_x_degrees': self._tensor_scalar_to_float(support_fx_deg),
                'focal_length_pixels': self._tensor_scalar_to_float(support_fx_pixels),
                'image_source': support_image_source,
                'fixed': True,
                'source': 'sam3d_static_support_object',
            }
            support_collision_objects.append(support_info)
            self.static_collision_meshes.append(support_info)

            if cached_record is not None:
                cached_record['original_mesh'] = None
                cached_record['simplified_mesh'] = None
            del original_mesh, simplified_mesh, support_collision_mesh, mesh_gs
            torch.cuda.empty_cache()

        self._static_support_mesh_init_records = []
        self.config['static_collision_objects'] = support_collision_objects
        self.config['static_support_object_count'] = len(support_collision_objects)
        fg_centers_pt3d = [
            pc_info['points'].mean(dim=0).detach().cpu().numpy()
            for pc_info in self.fg_pcs
        ]
        fg_centers_gs = [pt3d_to_gs(center) for center in fg_centers_pt3d]

        support_normal_pt3d, support_point_pt3d = self._estimate_static_support_plane(
            support_vertices_pt3d,
            object_centers=fg_centers_pt3d,
            up_hint=[0.0, 1.0, 0.0],
        )
        support_normal_gs, support_point_gs = self._estimate_static_support_plane(
            support_vertices_gs,
            object_centers=fg_centers_gs,
            up_hint=[0.0, 0.0, 1.0],
        )
        if support_normal_pt3d is not None and support_point_pt3d is not None:
            self.config['static_support_plane_normal_pt3d'] = support_normal_pt3d.tolist()
            self.config['static_support_plane_point_pt3d'] = support_point_pt3d.tolist()
        if support_normal_gs is not None and support_point_gs is not None:
            self.config['static_support_plane_normal_gs'] = support_normal_gs.tolist()
            self.config['static_support_plane_point_gs'] = support_point_gs.tolist()
            self.config['static_support_plane_height_quantile'] = float(
                self.config.get('static_support_plane_height_quantile', 0.5)
            )
            print(
                "Estimated static support plane from reconstructed support meshes: "
                f"normal={np.round(support_normal_gs, 4).tolist()}, "
                f"point={np.round(support_point_gs, 4).tolist()}"
            )

        return support_collision_objects

    def _background_collision_roi_mask(self, height, width):
        configured_roi = self.config.get('background_collision_roi', None)
        if configured_roi is not None:
            if len(configured_roi) > 0 and not isinstance(configured_roi[0], (int, float, str)):
                configured_roi = configured_roi[0]
            if len(configured_roi) != 4:
                raise ValueError("background_collision_roi must be [x_min, y_min, x_max, y_max].")
            x_min, y_min, x_max, y_max = [float(v) for v in configured_roi]
            if max(abs(x_min), abs(y_min), abs(x_max), abs(y_max)) <= 1.0:
                x_min, x_max = x_min * width, x_max * width
                y_min, y_max = y_min * height, y_max * height
            x_min = max(0, min(width, int(round(x_min))))
            x_max = max(0, min(width, int(round(x_max))))
            y_min = max(0, min(height, int(round(y_min))))
            y_max = max(0, min(height, int(round(y_max))))
            roi_mask = torch.zeros((height, width), dtype=torch.bool, device=self.device)
            if x_max > x_min and y_max > y_min:
                roi_mask[y_min:y_max, x_min:x_max] = True
                return roi_mask
            print("Warning: background_collision_roi is empty; falling back to object-based ROI.")

        if not self.config.get('background_collision_crop_to_objects', True):
            return torch.ones((height, width), dtype=torch.bool, device=self.device)

        union_mask = torch.zeros((height, width), dtype=torch.bool, device=self.device)
        for per_mask in self.object_masks:
            if per_mask.shape != union_mask.shape:
                continue
            union_mask = union_mask | per_mask.bool()

        if not union_mask.any():
            return torch.ones((height, width), dtype=torch.bool, device=self.device)

        ys, xs = torch.where(union_mask)
        default_margin = int(max(height, width) * 0.1)
        margin = int(self.config.get('background_collision_margin_px', default_margin))
        y_min = max(0, int(ys.min().item()) - margin)
        y_max = min(height, int(ys.max().item()) + margin + 1)
        x_min = max(0, int(xs.min().item()) - margin)
        x_max = min(width, int(xs.max().item()) + margin + 1)

        roi_mask = torch.zeros_like(union_mask)
        roi_mask[y_min:y_max, x_min:x_max] = True
        return roi_mask

    def _background_collision_mode(self):
        mode = self.config.get('background_collision_mode', None)
        if mode is None:
            if self.config.get('use_reconstructed_background_mesh_collision', False):
                return 'mesh'
            if self.config.get('use_reconstructed_background_collision', False):
                return 'mesh'
            return 'plane'
        return str(mode).strip().lower()

    def _roi_mask_from_config(self, configured_roi, height, width):
        if len(configured_roi) > 0 and not isinstance(configured_roi[0], (int, float, str)):
            configured_roi = configured_roi[0]
        if len(configured_roi) != 4:
            raise ValueError("background collision ROI must be [x_min, y_min, x_max, y_max].")

        x_min, y_min, x_max, y_max = [float(v) for v in configured_roi]
        if max(abs(x_min), abs(y_min), abs(x_max), abs(y_max)) <= 1.0:
            x_min, x_max = x_min * width, x_max * width
            y_min, y_max = y_min * height, y_max * height

        x_min = max(0, min(width, int(round(x_min))))
        x_max = max(0, min(width, int(round(x_max))))
        y_min = max(0, min(height, int(round(y_min))))
        y_max = max(0, min(height, int(round(y_max))))

        roi_mask = torch.zeros((height, width), dtype=torch.bool, device=self.device)
        if x_max > x_min and y_max > y_min:
            roi_mask[y_min:y_max, x_min:x_max] = True
        return roi_mask

    def _background_collision_mesh_mask(self, height, width):
        configured_roi = self.config.get('background_collision_mesh_roi', None)
        if configured_roi is not None:
            roi_mask = self._roi_mask_from_config(configured_roi, height, width)
            if roi_mask.any():
                return roi_mask
            print("Warning: background_collision_mesh_roi is empty; using the full inpainted background.")

        if self.config.get('background_collision_mesh_use_plane_roi', False):
            return self._background_collision_roi_mask(height, width)

        return torch.ones((height, width), dtype=torch.bool, device=self.device)

    def _orient_faces_toward_camera(self, vertices, faces):
        if faces.size == 0:
            return faces
        triangles = vertices[faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        centers = triangles.mean(axis=1)
        # The PyTorch3D camera is at the origin. Visible background surface normals
        # should point toward the camera so the front face bounds free space.
        flip = np.sum(normals * centers, axis=1) > 0
        faces = faces.copy()
        faces[flip] = faces[flip][:, [0, 2, 1]]
        return faces

    def _boundary_edges(self, faces):
        edge_counts = {}
        for tri in faces:
            for edge in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
                key = tuple(sorted((int(edge[0]), int(edge[1]))))
                edge_counts[key] = edge_counts.get(key, 0) + 1
        return [edge for edge, count in edge_counts.items() if count == 1]

    def create_background_collision_mesh(self, depth_map, valid_mask=None):
        mode = self._background_collision_mode()
        if mode not in {'mesh', 'both', 'background_mesh', 'background_mesh_only'}:
            return None

        self.config['background_collision_mode'] = mode
        for stale_key in ('background_collision_mesh_path_pt3d', 'background_collision_mesh_path_gs'):
            if stale_key in self.config:
                del self.config[stale_key]

        height, width = depth_map.shape
        collision_points = getattr(self, 'background_collision_points', None)
        if collision_points is None:
            collision_points = getattr(self, 'bg_points', None)
        if collision_points is None or collision_points.numel() == 0:
            print("Warning: reconstructed background collision mesh skipped because background collision points are empty.")
            return None

        stride = max(
            1,
            int(
                self.config.get(
                    'background_collision_mesh_stride',
                    self.config.get('background_collision_stride', 8),
                )
            ),
        )
        if self.config.get('background_collision_mesh_auto_stride', True):
            max_faces = self.config.get('background_collision_mesh_max_faces', 12000)
            if max_faces is not None:
                max_faces = max(2, int(max_faces))
                approx_rows = max(2, int(math.ceil(height / stride)) + 1)
                approx_cols = max(2, int(math.ceil(width / stride)) + 1)
                approx_top_faces = 2 * (approx_rows - 1) * (approx_cols - 1)
                if approx_top_faces > max_faces:
                    stride = max(stride + 1, int(math.ceil(stride * math.sqrt(approx_top_faces / max_faces))))
                    print(
                        "Adjusted background collision mesh stride for face budget: "
                        f"stride={stride}, max_faces={max_faces}, estimated_top_faces={approx_top_faces}"
                    )
        self.config['background_collision_mesh_effective_stride'] = int(stride)
        rows = np.arange(0, height, stride, dtype=np.int64)
        cols = np.arange(0, width, stride, dtype=np.int64)
        if rows[-1] != height - 1:
            rows = np.append(rows, height - 1)
        if cols[-1] != width - 1:
            cols = np.append(cols, width - 1)

        rows_t = torch.as_tensor(rows, dtype=torch.long, device=self.device)
        cols_t = torch.as_tensor(cols, dtype=torch.long, device=self.device)
        points_grid = collision_points.reshape(height, width, 3)
        points_ds = points_grid.index_select(0, rows_t).index_select(1, cols_t)
        depth_ds = depth_map.index_select(0, rows_t).index_select(1, cols_t)
        mesh_mask = self._background_collision_mesh_mask(height, width)
        mesh_mask_ds = mesh_mask.index_select(0, rows_t).index_select(1, cols_t)

        finite_ds = torch.isfinite(points_ds).all(dim=-1) & torch.isfinite(depth_ds) & mesh_mask_ds
        if valid_mask is not None and self.config.get('background_collision_mesh_require_valid_depth', False):
            valid_ds = valid_mask.index_select(0, rows_t).index_select(1, cols_t).bool()
            finite_ds = finite_ds & valid_ds

        if finite_ds.sum().item() < 4:
            print("Warning: reconstructed background collision mesh skipped because too few valid points were found.")
            return None

        points_np = points_ds.detach().cpu().numpy().astype(np.float32)
        valid_np = finite_ds.detach().cpu().numpy()
        depth_np = depth_ds.detach().cpu().numpy()
        index_map = -np.ones(valid_np.shape, dtype=np.int64)
        vertices_top = points_np[valid_np]
        index_map[valid_np] = np.arange(vertices_top.shape[0], dtype=np.int64)

        collision_color_image = getattr(self, 'background_collision_image', None)
        if collision_color_image is None:
            collision_color_image = self.inpainted_image
        color_grid = rearrange(collision_color_image, "c h w -> h w c")
        color_ds = color_grid.index_select(0, rows_t).index_select(1, cols_t)
        vertex_colors = (color_ds.detach().cpu().numpy()[valid_np].clip(0.0, 1.0) * 255).astype(np.uint8)

        max_depth_delta = self.config.get('background_collision_mesh_max_depth_delta', None)
        max_depth_delta = None if max_depth_delta is None else float(max_depth_delta)

        top_faces = []
        h_ds, w_ds = index_map.shape
        for r in range(h_ds - 1):
            for c in range(w_ds - 1):
                ids = [
                    index_map[r, c],
                    index_map[r, c + 1],
                    index_map[r + 1, c],
                    index_map[r + 1, c + 1],
                ]
                if min(ids) < 0:
                    continue
                if max_depth_delta is not None:
                    cell_depths = [
                        depth_np[r, c],
                        depth_np[r, c + 1],
                        depth_np[r + 1, c],
                        depth_np[r + 1, c + 1],
                    ]
                    if np.nanmax(cell_depths) - np.nanmin(cell_depths) > max_depth_delta:
                        continue

                i00, i01, i10, i11 = [int(i) for i in ids]
                top_faces.append([i00, i10, i11])
                top_faces.append([i00, i11, i01])

        min_faces = int(self.config.get('background_collision_mesh_min_faces', 2))
        if len(top_faces) < min_faces:
            print(
                "Warning: reconstructed background collision mesh skipped because "
                f"only {len(top_faces)} faces were generated."
            )
            return None

        top_faces = np.asarray(top_faces, dtype=np.int64)
        top_faces = self._orient_faces_toward_camera(vertices_top, top_faces)

        thickness = float(self.config.get('background_collision_mesh_thickness', 0.03))
        ray_dirs = vertices_top.copy()
        ray_norm = np.linalg.norm(ray_dirs, axis=1, keepdims=True).clip(min=1e-6)
        vertices_back = vertices_top + ray_dirs / ray_norm * thickness

        vertex_count = vertices_top.shape[0]
        back_faces = top_faces[:, [0, 2, 1]] + vertex_count
        side_faces = []
        for a, b in self._boundary_edges(top_faces):
            side_faces.append([a, b, b + vertex_count])
            side_faces.append([a, b + vertex_count, a + vertex_count])

        vertices = np.concatenate([vertices_top, vertices_back], axis=0)
        faces = np.concatenate(
            [
                top_faces,
                back_faces,
                np.asarray(side_faces, dtype=np.int64),
            ],
            axis=0,
        )
        all_vertex_colors = np.concatenate([vertex_colors, vertex_colors], axis=0)

        mesh_pt3d = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        mesh_pt3d.visual.vertex_colors = all_vertex_colors
        try:
            if hasattr(mesh_pt3d, 'remove_degenerate_faces'):
                mesh_pt3d.remove_degenerate_faces()
            elif hasattr(mesh_pt3d, 'nondegenerate_faces'):
                mesh_pt3d.update_faces(mesh_pt3d.nondegenerate_faces())

            if hasattr(mesh_pt3d, 'remove_duplicate_faces'):
                mesh_pt3d.remove_duplicate_faces()
            elif hasattr(mesh_pt3d, 'unique_faces'):
                mesh_pt3d.update_faces(mesh_pt3d.unique_faces())
            mesh_pt3d.fix_normals()
        except Exception as exc:
            print(f"Warning: background collision mesh cleanup failed: {exc}")

        output_folder = Path(self.config['output_folder'])
        output_folder.mkdir(parents=True, exist_ok=True)
        pt3d_path = output_folder / 'background_collision_mesh_pt3d.obj'
        gs_path = output_folder / 'background_collision_mesh_gs.obj'
        mesh_pt3d.export(pt3d_path)

        vertices_gs = pt3d_to_gs(np.asarray(mesh_pt3d.vertices, dtype=np.float32))
        mesh_gs = trimesh.Trimesh(vertices=vertices_gs, faces=np.asarray(mesh_pt3d.faces), process=False)
        mesh_gs.visual.vertex_colors = np.asarray(mesh_pt3d.visual.vertex_colors)
        mesh_gs.export(gs_path)

        bounds_min = vertices_gs.min(axis=0)
        bounds_max = vertices_gs.max(axis=0)
        self.config['background_collision_mesh_path_pt3d'] = str(pt3d_path)
        self.config['background_collision_mesh_path_gs'] = str(gs_path)
        self.config['background_collision_mesh_stride'] = int(stride)
        self.config['background_collision_mesh_thickness'] = float(thickness)
        self.config['background_collision_mesh_vertex_count'] = int(mesh_gs.vertices.shape[0])
        self.config['background_collision_mesh_face_count'] = int(mesh_gs.faces.shape[0])
        self.config['background_collision_mesh_bounds_gs'] = {
            'min': bounds_min.tolist(),
            'max': bounds_max.tolist(),
        }
        self.config['background_collision_mesh_source'] = (
            'inpainted_background_depth_unprojected_with_same_camera_as_foreground'
        )

        print(
            "Created reconstructed background collision mesh: "
            f"path={gs_path}, vertices={mesh_gs.vertices.shape[0]}, "
            f"faces={mesh_gs.faces.shape[0]}, stride={stride}, thickness={thickness}"
        )
        return {
            'path_pt3d': str(pt3d_path),
            'path_gs': str(gs_path),
            'bounds_gs': self.config['background_collision_mesh_bounds_gs'],
        }

    def _snap_background_normal(self, normal):
        snap_degrees = float(self.config.get('background_plane_snap_degrees', 45.0))
        if snap_degrees <= 0:
            return normal / max(np.linalg.norm(normal), 1e-8), 0.0

        candidates = []
        # PyTorch3D coordinates: +Y maps to Genesis +Z, the original vertical/up normal.
        for polar_deg in np.arange(0.0, 180.0 + 1e-6, snap_degrees):
            polar = np.deg2rad(polar_deg)
            if abs(np.sin(polar)) < 1e-6:
                candidates.append([0.0, np.cos(polar), 0.0])
                continue
            for azimuth_deg in np.arange(0.0, 360.0, snap_degrees):
                azimuth = np.deg2rad(azimuth_deg)
                candidates.append([
                    np.sin(polar) * np.cos(azimuth),
                    np.cos(polar),
                    np.sin(polar) * np.sin(azimuth),
                ])

        candidates = np.asarray(candidates, dtype=np.float32)
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True).clip(min=1e-8)
        normal = normal.astype(np.float32)
        normal /= max(np.linalg.norm(normal), 1e-8)
        best_idx = int(np.argmax(candidates @ normal))
        snapped = candidates[best_idx]
        angle = np.rad2deg(np.arccos(np.clip(float(np.dot(snapped, normal)), -1.0, 1.0)))
        return snapped, angle

    def estimate_background_collision_plane(self, depth_map, valid_mask=None):
        if (
            not self.config.get('use_reconstructed_background_collision', False)
            and self._background_collision_mode() not in {'mesh', 'both', 'background_mesh', 'background_mesh_only'}
        ):
            return None

        self.background_collision_normal = None
        self.background_plane_point = None
        height, width = depth_map.shape
        collision_points = getattr(self, 'background_collision_points', None)
        if collision_points is None:
            collision_points = getattr(self, 'bg_points', None)
        if collision_points is None or collision_points.numel() == 0:
            return None

        stride = max(1, int(self.config.get('background_plane_stride', self.config.get('background_collision_stride', 8))))
        points_grid = collision_points.reshape(height, width, 3)
        roi_mask = self._background_collision_roi_mask(height, width)

        points_ds = points_grid[::stride, ::stride]
        roi_ds = roi_mask[::stride, ::stride]
        depth_ds = depth_map[::stride, ::stride]

        finite_ds = torch.isfinite(points_ds).all(dim=-1) & torch.isfinite(depth_ds)
        if valid_mask is not None and self.config.get('background_collision_require_valid_depth', False):
            finite_ds = finite_ds & valid_mask[::stride, ::stride].bool()
        valid_ds = finite_ds & roi_ds

        if valid_ds.sum().item() < 3:
            print("Warning: reconstructed background plane skipped because too few valid background points were found.")
            return None

        plane_points = points_ds[valid_ds]
        max_normal_points = int(self.config.get('background_collision_normal_points', 20000))
        if plane_points.shape[0] > max_normal_points:
            sample_idx = torch.linspace(
                0,
                plane_points.shape[0] - 1,
                steps=max_normal_points,
                device=plane_points.device,
            ).long()
            plane_points = plane_points[sample_idx]

        raw_normal = self.estimate_plane_normal_simple(plane_points.detach().cpu().numpy())
        plane_point = plane_points.mean(dim=0)
        if self.fg_pcs:
            fg_centers = torch.stack([pc_info['points'].mean(dim=0) for pc_info in self.fg_pcs], dim=0)
            object_side = (fg_centers.mean(dim=0) - plane_point).detach().cpu().numpy()
            if np.dot(raw_normal, object_side) < 0:
                raw_normal = -raw_normal
        elif raw_normal[1] < 0:
            raw_normal = -raw_normal

        snapped_normal, snap_angle = self._snap_background_normal(raw_normal)
        max_snap_angle = float(self.config.get('background_plane_max_snap_angle_degrees', 30.0))
        if snap_angle > max_snap_angle:
            print(
                f"Warning: background plane normal snap angle {snap_angle:.2f} deg exceeds "
                f"{max_snap_angle:.2f} deg; using raw normal instead."
            )
            snapped_normal = raw_normal / max(np.linalg.norm(raw_normal), 1e-8)
            snap_angle = 0.0

        if snapped_normal[1] < 0 and not self.fg_pcs:
            snapped_normal = -snapped_normal

        self.background_collision_normal = snapped_normal
        self.background_plane_point = plane_point.detach().cpu().numpy()
        self.config['background_plane_raw_normal_pt3d'] = raw_normal.tolist()
        self.config['background_plane_normal_pt3d'] = snapped_normal.tolist()
        self.config['background_plane_point_pt3d'] = self.background_plane_point.tolist()
        self.config['background_plane_snap_angle_degrees'] = float(snap_angle)

        print(
            "Estimated reconstructed background plane normal: "
            f"raw={np.round(raw_normal, 4).tolist()}, "
            f"snapped={np.round(snapped_normal, 4).tolist()}, "
            f"snap_angle={snap_angle:.2f} deg, stride={stride}"
        )
        return {
            'normal': snapped_normal,
            'point': self.background_plane_point,
            'raw_normal': raw_normal,
        }

    @torch.no_grad()
    def get_camera_at_origin(self, focal_length=None):
        W, H = self.target_size
        K = torch.zeros((1, 4, 4), device=self.device)
        if focal_length is None:
            focal_length = self.init_focal_length
        focal_length = torch.as_tensor(focal_length, device=self.device, dtype=torch.float32)
        K[0, 0, 0] = focal_length
        K[0, 1, 1] = focal_length
        # 动态计算光心
        K[0, 0, 2] = W / 2.0 
        K[0, 1, 2] = H / 2.0
        K[0, 3, 2] = 1
        K[0, 2, 3] = 1
        R = torch.eye(3, device=self.device).unsqueeze(0)
        T = torch.zeros((1, 3), device=self.device)
        camera = PerspectiveCameras(
            K=K, R=R, T=T, in_ndc=False, image_size=((H, W),), device=self.device
        )
        return camera
    
    def convert_pytorch3d_kornia(self, camera, focal_length, update_intrinsics_parameters=None):
        W, H = self.target_size  # 获取真实宽高
        transform_matrix_pt3d = camera.get_world_to_view_transform().get_matrix()[0]
        transform_matrix_w2c_pt3d = transform_matrix_pt3d.transpose(0, 1)

        pt3d_to_kornia = torch.diag(torch.tensor([-1.0, -1, 1, 1], device=camera.device))
        transform_matrix_w2c_kornia = pt3d_to_kornia @ transform_matrix_w2c_pt3d

        extrinsics = transform_matrix_w2c_kornia.unsqueeze(0)
        
        # 替换为真实的 H 和 W
        h = torch.tensor([H], device="cuda")
        w = torch.tensor([W], device="cuda")
        K = torch.eye(4)[None].to("cuda")
        K[0, 0, 2] = W / 2.0
        K[0, 1, 2] = H / 2.0
        K[0, 0, 0] = focal_length
        K[0, 1, 1] = focal_length
        
        if update_intrinsics_parameters is not None:
            u0, v0, w_crop, h_crop, p_left, p_right, p_up, p_down, scale = (
                update_intrinsics_parameters
            )
            new_cx = (K[0, 0, 2] - u0 + p_left) * scale
            new_cy = (K[0, 1, 2] - v0 + p_up) * scale
            new_fx = K[0, 0, 0] * scale
            new_fy = K[0, 1, 1] * scale
            K[0, 0, 2] = new_cx
            K[0, 1, 2] = new_cy
            K[0, 0, 0] = new_fx
            K[0, 1, 1] = new_fy
            new_h = torch.tensor([H], device="cuda")
            new_w = torch.tensor([H], device="cuda")
            return PinholeCamera(K, extrinsics, new_h, new_w)

        return PinholeCamera(K, extrinsics, h, w)

    def _background_image_tensor(self, image_size):
        background = getattr(self, 'inpainted_image', None)
        if background is None:
            background = self.input_image
        background = background.to(self.device)
        target_h, target_w = image_size
        if background.shape[-2:] != (target_h, target_w):
            background = torch.nn.functional.interpolate(
                background.unsqueeze(0),
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)
        return background.permute(1, 2, 0).clamp(0, 1)

    def render(self, render_bg=True, render_obj=True, render_mesh=True, frame_id=0, save=True, mask=True, 
            compute_optical_flow=True):
        """
        Render function with optical flow support based on the original Gaussian splatting logic.
        
        Args:
            render_bg: Whether to render background
            render_obj: Whether to render foreground objects  
            render_mesh: Whether to render mesh
            frame_id: Current frame ID
            save: Whether to save outputs
            mask: Whether to save masks
            compute_optical_flow: Whether to compute optical flow
            prev_frame_data: Dictionary containing previous frame's point positions and camera
                            Format: {
                                'fg_points': previous frame foreground points,
                                'camera': previous frame camera,
                                'bg_points': previous frame background points (optional)
                            }
        
        Returns:
            image_pil: Rendered image
            fg_points_mask: Foreground points mask
            mesh_mask: Mesh mask  
            optical_flow: Optical flow (H, W, 3) if compute_optical_flow=True, else None (third channel is 0 for foreground)
        """
        cameras = self.current_camera
        # image_size = self.target_size[0]
        image_size = (self.target_size[1], self.target_size[0])
        optical_flow = None

        ### 1. Render background
        background_mode = str(
            self.config.get(
                'render_background_mode',
                self.config.get('simulation_background_mode', 'inpainted_image'),
            )
        ).strip().lower()
        use_image_background = background_mode in {'image', 'inpainted', 'inpainted_image', '2d'}
        self.config['render_background_mode'] = 'inpainted_image' if use_image_background else background_mode
        if render_bg and use_image_background:
            base_rgb = self._background_image_tensor(image_size)
        elif render_bg and self.cache_bg is None:
            bg_pc = Pointclouds(
                points=[self.bg_points],
                features=[self.bg_points_colors]
            )
            bg_raster_settings = PointsRasterizationSettings(
                image_size=image_size,
                radius= 0.0001 if 'bg_points_render_radius' not in self.config else self.config['bg_points_render_radius'],
                points_per_pixel=30
            )
            bg_renderer = PointsRenderer(
                rasterizer=PointsRasterizer(cameras=cameras, raster_settings=bg_raster_settings),
                compositor=AlphaCompositor()
            )
            bg_image = bg_renderer(bg_pc)
            self.cache_bg = bg_image
        elif render_bg and self.cache_bg is not None:
            bg_image = self.cache_bg
        else:
            bg_image = torch.zeros(1, image_size[0], image_size[1], 3, device=self.device)

        if not (render_bg and use_image_background):
            base_rgb = bg_image[0].clone()
        final_rgb = base_rgb.clone()

        ### 2. Render foreground point clouds
        all_fg_points = []
        all_fg_colors = []
        
        for pc_info in self.fg_pcs:
            points = pc_info['points']
            colors = pc_info['colors']
            
            all_fg_points.append(points)
            all_fg_colors.append(colors)
        
        combined_fg_points = torch.cat(all_fg_points, dim=0)
        combined_fg_colors = torch.cat(all_fg_colors, dim=0)

        flow_rendered_points = combined_fg_points.clone()
        
        alpha = 1.0
        combined_rgba = torch.cat([
            combined_fg_colors,
            alpha * torch.ones_like(combined_fg_colors[..., :1])
        ], dim=-1)
        
        fg_pc = Pointclouds(
            points=[combined_fg_points],
            features=[combined_rgba]
        )
        
        fg_raster_settings = PointsRasterizationSettings(
            image_size=image_size,
            radius=0.01 if 'fg_points_render_radius' not in self.config else self.config['fg_points_render_radius'],
            points_per_pixel=30,
            max_points_per_bin = 20000,
            bin_size=0,
        )
        
        fg_rasterizer = PointsRasterizer(cameras=cameras, raster_settings=fg_raster_settings)
        fg_renderer = PointsRenderer(
            rasterizer=fg_rasterizer,
            compositor=AlphaCompositor()
        )
        
        fg_image = fg_renderer(fg_pc)
        fg_rgb = fg_image[0, ..., :3]
        fg_alpha = fg_image[0, ..., 3:4]
        
        fragments = fg_rasterizer(fg_pc)
        fg_depth = fragments.zbuf[0, ..., 0]
        
        fg_points_mask = torch.where(fg_alpha.squeeze(-1) > self.config['alpha_threshold'], 1.0, 0.0).unsqueeze(-1)
        
        fg_mask_2d = fg_points_mask.squeeze(-1)
        final_rgb = fg_rgb * fg_mask_2d.unsqueeze(-1) + final_rgb * (1.0 - fg_mask_2d.unsqueeze(-1))

        ### 4. Render mesh
        mesh_mask = torch.zeros(image_size[0], image_size[1], 1, dtype=torch.float32, device=self.device)
        
        if render_mesh and self.franka_mesh is not None:
            from pytorch3d.renderer import (
                MeshRenderer, MeshRasterizer, SoftPhongShader,
                RasterizationSettings, BlendParams
            )
            from pytorch3d.structures import Meshes
            from pytorch3d.renderer.mesh.textures import TexturesVertex

            vertices = self.franka_mesh['vertices']
            faces = self.franka_mesh['faces']
            colors = self.franka_mesh['colors']

            flow_rendered_points = torch.cat([flow_rendered_points, vertices], dim=0)
            
            if not isinstance(vertices, torch.Tensor):
                vertices = torch.tensor(vertices, dtype=torch.float32, device=self.device)
            if not isinstance(faces, torch.Tensor):
                faces = torch.tensor(faces, dtype=torch.long, device=self.device)
            if not isinstance(colors, torch.Tensor):
                colors = torch.tensor(colors, dtype=torch.float32, device=self.device)
            
            vertices = vertices.to(self.device)
            faces = faces.to(self.device)
            colors = colors.to(self.device)
            
            textures = TexturesVertex(verts_features=[colors])
            combined_mesh = Meshes(verts=[vertices], faces=[faces], textures=textures)
                
            mesh_raster_settings = RasterizationSettings(
                image_size=image_size,
                blur_radius=0.0,
                faces_per_pixel=10,
                bin_size=0
            )
            
            mesh_rasterizer = MeshRasterizer(cameras=cameras, raster_settings=mesh_raster_settings)
            mesh_renderer = MeshRenderer(
                rasterizer=mesh_rasterizer,
                shader=SoftPhongShader(
                    device=self.device,
                    cameras=cameras,
                    blend_params=BlendParams(background_color=(0.0, 0.0, 0.0))
                )
            )
            
            mesh_image = mesh_renderer(combined_mesh)
            mesh_rgb = mesh_image[0, ..., :3]
            mesh_alpha = mesh_image[0, ..., 3:4]
            
            mesh_fragments = mesh_rasterizer(combined_mesh)
            mesh_depth = mesh_fragments.zbuf[0, ..., 0]
            
            mesh_mask_2d = torch.where(mesh_alpha.squeeze(-1) > 0.01, 1.0, 0.0)
            
            fg_depth_valid = torch.where(fg_mask_2d > 0, fg_depth, torch.tensor(float('inf'), device=self.device))
            mesh_depth_valid = torch.where(mesh_mask_2d > 0, mesh_depth, torch.tensor(float('inf'), device=self.device))
            
            mesh_closer_bool = (mesh_depth_valid < fg_depth_valid) & (mesh_mask_2d > 0)
            mesh_closer_float = mesh_closer_bool.float()
            mesh_mask = mesh_closer_float.unsqueeze(-1)
            
            mesh_closer_3d = mesh_closer_float.unsqueeze(-1)
            final_rgb = mesh_rgb * mesh_closer_3d + final_rgb * (1.0 - mesh_closer_3d)
            
            fg_points_mask = torch.where(mesh_closer_bool.unsqueeze(-1), 
                                    torch.zeros_like(fg_points_mask), 
                                    fg_points_mask)


        # 3. Compute optical flow if requested (following original logic)
        if compute_optical_flow and self.previous_frame_data is not None:
            
            optical_flow = self._compute_optical_flow_pytorch3d_style(
                current_fg_points=flow_rendered_points,
                prev_fg_points=self.previous_frame_data['flow_rendered_points'],
                current_camera=cameras,
                prev_camera=self.previous_frame_data['camera'],
                image_size=image_size,
                frame_id=frame_id
            )
            
            if self.optical_flow.size == 0:
                self.optical_flow = np.expand_dims(optical_flow.cpu().numpy(), 0)
            else:
                self.optical_flow = np.concatenate([self.optical_flow, np.expand_dims(optical_flow.cpu().numpy(), 0)])

        ### 5. Save outputs
        if mask and save:
            
            points_mask_path = self.output_folder_masks / f"points_mask_{frame_id:04d}.png"
            points_mask_to_save = fg_points_mask.squeeze(2) if fg_points_mask.dim() == 3 else fg_points_mask
            ToPILImage()(points_mask_to_save.unsqueeze(0).clamp(0, 1).cpu()).save(points_mask_path.as_posix())
            
            mesh_mask_path = self.output_folder_masks / f"mesh_mask_{frame_id:04d}.png"
            mesh_mask_to_save = mesh_mask.squeeze(2) if mesh_mask.dim() == 3 else mesh_mask
            ToPILImage()(mesh_mask_to_save.unsqueeze(0).clamp(0, 1).cpu()).save(mesh_mask_path.as_posix())
        
        # if save and compute_optical_flow and optical_flow is not None:
        #     self._save_optical_flow(optical_flow, frame_id)
        
        image_pil = ToPILImage()(final_rgb.permute(2, 0, 1).clamp(0, 1).cpu())
        if save:
            image_path = self.output_folder_frames / f"frame_{frame_id:04d}.png"
            image_pil.save(image_path.as_posix())
        
        self.previous_frame_data = {
            'camera': cameras,
            'bg_points': self.bg_points,
            'flow_rendered_points': flow_rendered_points
        }

        return image_pil, fg_points_mask, mesh_mask


    def save_optical_flow(self, optical_flow, valid_mask, frame_id):

        # Extract flow components
        flow_x = optical_flow[:, :, 0].cpu().numpy()
        flow_y = optical_flow[:, :, 1].cpu().numpy()
        valid_mask_np = valid_mask.cpu().numpy()
        
        # Convert flow to HSV color representation
        angle = np.arctan2(-flow_y, flow_x)
        
        # Create HSV image
        hsv = np.zeros((optical_flow.shape[0], optical_flow.shape[1], 3), dtype=np.uint8)
        hsv[..., 0] = (angle + np.pi) / (2 * np.pi) * 179
        hsv[..., 1] = 255
        hsv[..., 2] = 255
        # magnitude = np.sqrt(flow_x**2 + flow_y**2)
        # hsv[..., 2] = np.clip(magnitude * 255 / np.max(magnitude), 0, 255).astype(np.uint8)
        
        # Apply valid mask
        hsv[~valid_mask_np] = 0
        
        # Convert HSV to RGB
        flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # Create color wheel
        def create_color_wheel(size=256):
            x = np.linspace(-1, 1, size)
            y = np.linspace(-1, 1, size)
            X, Y = np.meshgrid(x, y)
            
            magnitude = np.sqrt(X**2 + Y**2)
            angle = np.arctan2(-Y, X)
            
            mask = magnitude <= 1.0
            
            hsv_wheel = np.zeros((size, size, 3), dtype=np.uint8)
            hsv_wheel[mask, 0] = ((angle[mask] + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
            hsv_wheel[mask, 1] = 255
            hsv_wheel[mask, 2] = 255
            
            rgb_wheel = cv2.cvtColor(hsv_wheel, cv2.COLOR_HSV2RGB)
            return rgb_wheel
        
        color_wheel = create_color_wheel()
        
        # Save visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Flow visualization
        ax1.imshow(flow_rgb)
        ax1.set_title(f'Optical Flow Direction - Frame {frame_id}')
        ax1.axis('off')
        
        # Color wheel
        ax2.imshow(color_wheel)
        ax2.set_title('Flow Direction Color Wheel')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_folder_optical_flow}/optical_flow_frame_{frame_id:04d}.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _compute_optical_flow_pytorch3d_style(self, current_fg_points, prev_fg_points, 
                                        current_camera, prev_camera, image_size=(512,512), frame_id=0):
        
        if current_fg_points.shape[0] > prev_fg_points.shape[0]:
            current_fg_points = current_fg_points[:prev_fg_points.shape[0]]
        elif prev_fg_points.shape[0] > current_fg_points.shape[0]:
            prev_more = prev_fg_points[-(prev_fg_points.shape[0] - current_fg_points.shape[0]):]
            current_fg_points = torch.cat([current_fg_points, prev_more], dim=0)
        
        current_uv = self._proj_uv(current_fg_points, current_camera, image_size)
        prev_uv = self._proj_uv(prev_fg_points, prev_camera, image_size)
        
        delta_uv = current_uv - prev_uv

        delta_uv_3d = torch.cat([delta_uv, torch.zeros_like(delta_uv[:, :1])], dim=-1)
        

        flow_colors = delta_uv_3d.clone()
        xy_flow = flow_colors[:, :2]

        magnitude = torch.sqrt(xy_flow[:, 0]**2 + xy_flow[:, 1]**2)
        zero_flow_mask = magnitude < 1e-4

        min_val = xy_flow.min()
        max_val = xy_flow.max()

        if max_val - min_val > 1e-4:
            flow_colors[:, :2] = 0.1 + (xy_flow - min_val) / (max_val - min_val) * 0.8
            flow_colors[zero_flow_mask, :2] = 0.0
        else:
            flow_colors[:, :2] = 0.5
        
        flow_colors = torch.clamp(flow_colors, 0, 1)

        alpha = 1.0
        flow_rgba = torch.cat([
            flow_colors,
            alpha * torch.ones_like(flow_colors[..., :1])
        ], dim=-1)

        point_cloud = Pointclouds(
            points=[prev_fg_points],
            features=[flow_rgba]
        )
        
        raster_settings = PointsRasterizationSettings(
            image_size=image_size,
            radius=0.01 if 'fg_points_render_radius' not in self.config else self.config['fg_points_render_radius'],
            points_per_pixel=50,
        )
        
        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=current_camera, raster_settings=raster_settings),
            compositor=AlphaCompositor()
        )
        
        flow_image = renderer(point_cloud)
        
        flow_alpha = flow_image[0, :, :, 3]
        valid_mask = flow_alpha > self.config['alpha_threshold']

        optical_flow = torch.zeros(image_size[0], image_size[1], 3, device=self.device)

        if valid_mask.sum() > 0 and max_val - min_val > 1e-4:
            rendered_flow = flow_image[0, :, :, :2][valid_mask]
            
            zero_pixels = torch.all(rendered_flow < 0.05, dim=-1)
            normal_pixels = ~zero_pixels
            
            full_flow = torch.zeros_like(rendered_flow)
            
            if normal_pixels.sum() > 0:
                full_flow[normal_pixels] = (rendered_flow[normal_pixels] - 0.1) / 0.8 * (max_val - min_val) + min_val
            
            if zero_pixels.sum() > 0:
                full_flow[zero_pixels] = 0.0
            
            optical_flow[:, :, :2][valid_mask] = full_flow

            meaningful_mask = valid_mask.clone()
            valid_coords = torch.where(valid_mask)
            zero_coords_in_valid = zero_pixels
            meaningful_mask[valid_coords[0][zero_coords_in_valid], valid_coords[1][zero_coords_in_valid]] = False
            
            if self.config.get('debug', False):
                self.save_optical_flow(optical_flow, meaningful_mask, frame_id)

        return optical_flow


    def _proj_uv(self, xyz, camera, image_size):
        device = xyz.device
        
        K_4x4 = camera.K[0]
        intr = K_4x4[:3, :3].clone()
        
        w2c = torch.eye(4).float().to(device)
        R_w2c = camera.R[0]
        T_w2c = camera.T[0]
        w2c[:3, :3] = R_w2c
        w2c[:3, 3] = T_w2c

        intr[2, 2] = 1.0
        
        intr = intr.to(device)
        
        c_xyz = xyz @ w2c[:3, :3] + w2c[:3, 3]
        i_xyz = (intr @ c_xyz.T).T
        uv = i_xyz[:, :2] / i_xyz[:, -1:].clip(1e-3)

        if isinstance(image_size, tuple):
            size_tensor = torch.tensor([image_size[1], image_size[0]], device=device)
            uv = size_tensor - uv
        else:
            uv = image_size - uv
        
        return uv

    def obj_kp_matching(
        self,
        mask,
        mesh_vertices,
        mesh_faces,
        idx,
        kp_key='obj_kp',
        unprojected_points=None,
        render_camera=None,
        render_focal_length=None,
    ):

        suffix = self._debug_suffix(idx)
        if render_camera is None:
            render_camera = self.current_camera
        gt_kp_h, gt_kp_w = self.kps_from_quants(mask, idx, kp_key=kp_key)
        if self.config.get('debug', False):
            gt_kp_save_path = (self.output_folder / f"gt_kps_{suffix}.png").as_posix()
            save_mask_kps(mask, gt_kp_h, gt_kp_w, gt_kp_save_path)

        verts_min = mesh_vertices.min(dim=0)[0].unsqueeze(0).unsqueeze(0)
        verts_max = mesh_vertices.max(dim=0)[0].unsqueeze(0).unsqueeze(0)

        proxy_colors = ((mesh_vertices.clone() - verts_min) / (
            verts_max - verts_min
        )).squeeze(0)
        
        z_translation = torch.tensor([0, 0, 0.5], device=self.device)
        mesh_vertices += z_translation

        def render_mesh(mesh_vertices, mesh_faces, mesh_colors):
            textures = Textures(verts_rgb=mesh_colors.unsqueeze(0))
            obj_mesh = Meshes(
                verts=[mesh_vertices],
                faces=[mesh_faces],
                textures=textures
            )
            obj_raster_settings = RasterizationSettings(
                image_size=(self.target_size[1], self.target_size[0]),
                blur_radius=0.0,
                faces_per_pixel=1,
                bin_size=0
            )

            obj_renderer = MeshRenderer(
                rasterizer=MeshRasterizer(cameras=render_camera, raster_settings=obj_raster_settings),
                shader=HardShader(device=self.device, cameras=render_camera),
            )
            torch.cuda.empty_cache()
            rendered_images = obj_renderer(obj_mesh)
            rendered_rgb = rendered_images[0, ..., :3]
            rendered_mask = rendered_images[0, ..., -1]
            rendered_mask = (rendered_mask > 0).float()
            rendered_rgb = rendered_rgb.permute(2, 0, 1).clamp(0, 1)

            return rendered_rgb, rendered_mask


        fg_render, fg_mask = render_mesh(mesh_vertices, mesh_faces, proxy_colors)
        raw_fg_render = fg_render
        raw_fg_mask = fg_mask
        target_mask = mask.float()
        masked_fg_mask = raw_fg_mask * target_mask
        raw_area = raw_fg_mask.sum().item()
        target_area = target_mask.sum().item()
        overlap_area = masked_fg_mask.sum().item()
        retain_ratio = overlap_area / max(raw_area, 1.0)
        target_coverage_ratio = overlap_area / max(target_area, 1.0)
        fg_render = raw_fg_render
        fg_mask = raw_fg_mask
        if self.config.get('debug', False):
            def mask_bbox(mask_tensor):
                coords = torch.nonzero(mask_tensor > 0, as_tuple=False)
                if coords.numel() == 0:
                    return None
                y_min = int(coords[:, 0].min().item())
                x_min = int(coords[:, 1].min().item())
                y_max = int(coords[:, 0].max().item())
                x_max = int(coords[:, 1].max().item())
                return [x_min, y_min, x_max, y_max]

            raw_proxy_path = self.output_folder / f"mesh_init_render_proxy_color_raw_{suffix}.png"
            intersection_proxy_path = self.output_folder / f"mesh_init_render_proxy_color_intersection_{suffix}.png"
            final_proxy_path = self.output_folder / f"mesh_init_render_proxy_color_{suffix}.png"
            torchvision_utils.save_image(raw_fg_render, raw_proxy_path)
            torchvision_utils.save_image(raw_fg_render * target_mask.unsqueeze(0), intersection_proxy_path)
            torchvision_utils.save_image(fg_render, self.output_folder / f"mesh_init_render_proxy_color_{suffix}.png")
            torchvision_utils.save_image(
                raw_fg_mask.unsqueeze(0),
                self.output_folder / f"obj_kp_proxy_raw_mask_{suffix}.png",
            )
            torchvision_utils.save_image(
                target_mask.unsqueeze(0),
                self.output_folder / f"obj_kp_proxy_target_mask_{suffix}.png",
            )
            torchvision_utils.save_image(
                masked_fg_mask.unsqueeze(0),
                self.output_folder / f"obj_kp_proxy_intersection_mask_{suffix}.png",
            )
            debug_info = {
                "suffix": suffix,
                "kp_key": kp_key,
                "same_logic_for_support_and_dynamic_objects": True,
                "decision": "use_raw_proxy",
                "decision_reason": "mask_clipping_removed",
                "render_focal_length_pixels": (
                    float(render_focal_length.detach().cpu().item())
                    if torch.is_tensor(render_focal_length)
                    else float(render_focal_length)
                    if render_focal_length is not None
                    else None
                ),
                "global_scene_focal_length_pixels": (
                    float(self.init_focal_length.detach().cpu().item())
                    if torch.is_tensor(self.init_focal_length)
                    else float(self.init_focal_length)
                    if hasattr(self, 'init_focal_length')
                    else None
                ),
                "raw_area": raw_area,
                "target_area": target_area,
                "overlap_area": overlap_area,
                "retain_ratio": retain_ratio,
                "target_coverage_ratio": target_coverage_ratio,
                "raw_proxy_bbox_xyxy": mask_bbox(raw_fg_mask),
                "target_mask_bbox_xyxy": mask_bbox(target_mask),
                "intersection_bbox_xyxy": mask_bbox(masked_fg_mask),
                "image_size_hw": [int(raw_fg_mask.shape[0]), int(raw_fg_mask.shape[1])],
                "raw_proxy_path": raw_proxy_path.name,
                "intersection_proxy_path": intersection_proxy_path.name,
                "final_proxy_path": final_proxy_path.name,
            }
            with open(self.output_folder / f"obj_kp_proxy_clip_debug_{suffix}.json", "w") as f:
                json.dump(debug_info, f, indent=2)

        mesh_kps_h, mesh_kps_w = self.kps_from_quants(fg_mask, idx, kp_key=kp_key)
        if self.config.get('debug', False):
            mesh_kps_save_path = (self.output_folder / f"mesh_kps_{suffix}.png").as_posix()
            save_mask_kps(fg_mask, mesh_kps_h, mesh_kps_w, mesh_kps_save_path)
        

        # input_unprojected_points = rearrange(
        #     self.input_image_points, "(w h) c -> c h w", h=self.target_size[0], w=self.target_size[1]
        # )

        # 修复：改成 (h w)，并且 h 对应 target_size[1](480), w 对应 target_size[0](832)
        if unprojected_points is None:
            unprojected_points = self.input_image_points
        input_unprojected_points = rearrange(
            unprojected_points, "(h w) c -> c h w", h=self.target_size[1], w=self.target_size[0]
        )

        gt_kps = input_unprojected_points[:, gt_kp_h, gt_kp_w].permute(1, 0)
        mesh_kps = fg_render[:, mesh_kps_h, mesh_kps_w].permute(1, 0)
        mesh_kps = mesh_kps * (verts_max[0] - verts_min[0]) + verts_min[0]

        A = mesh_kps
        B = gt_kps.flatten().unsqueeze(-1)
        A_compact = torch.cat(
            [
                A.unsqueeze(-1),
                torch.eye(3)
                .unsqueeze(0)
                .repeat(mesh_kps.shape[0], 1, 1)
                .to(device=self.device),
            ],
            dim=-1,
        )
        A_compact_final = torch.cat([i for i in A_compact], dim=0)

        solution = torch.linalg.lstsq(A_compact_final, B).solution
        scale = solution[0]
        translation = solution[1:, 0]
        if float(scale.detach().cpu()) <= 1e-6:
            a_mean = A.mean(dim=0)
            b_mean = gt_kps.mean(dim=0)
            a_centered = A - a_mean
            b_centered = gt_kps - b_mean
            scale = torch.sqrt(
                (b_centered.square().sum() / a_centered.square().sum().clamp_min(1e-8)).clamp_min(1e-8)
            )
            translation = b_mean - scale * a_mean
            print(
                "Warning: obj_kp_matching produced a non-positive scale; "
                f"using positive scale fallback for object {suffix}."
            )
        mesh_vertices -= z_translation

        return scale, translation


    def kps_from_quants(self, mask, idx, kp_key='obj_kp'):
        mask_np = (mask.detach().cpu().numpy() != 0).astype(np.uint8)
        candidate_coords = np.argwhere(mask_np > 0)
        if candidate_coords.shape[0] == 0:
            raise ValueError("Cannot sample keypoints from an empty mask.")

        kp_config = self.config.get(kp_key, None)
        uses_direct_quantiles = (
            isinstance(kp_config, (list, tuple))
            and len(kp_config) == 2
            and all(isinstance(item, (list, tuple)) for item in kp_config)
            and all(all(isinstance(q, (int, float)) for q in item) for item in kp_config)
        )
        if kp_config is not None and idx is not None and not uses_direct_quantiles:
            try:
                if isinstance(idx, str) and idx.startswith('support_'):
                    config_idx = int(idx.split('_')[-1])
                else:
                    config_idx = int(idx)
                kp_config = kp_config[config_idx]
            except Exception:
                pass

        if kp_config is None:
            h_quants, w_quants = [0.1, 0.9], [0.1, 0.9]
        else:
            h_quants, w_quants = kp_config

        h_quants = [float(q) for q in h_quants]
        w_quants = [float(q) for q in w_quants]
        target_ws = np.quantile(candidate_coords[:, 1], w_quants)

        selected = []
        used_indices = set()
        width_span = max(float(candidate_coords[:, 1].max() - candidate_coords[:, 1].min()), 1.0)
        min_slice_points = max(8, len(h_quants) * 2)

        for target_w in target_ws:
            half_width = max(1.0, width_span * 0.015)
            slice_mask = np.abs(candidate_coords[:, 1].astype(np.float32) - float(target_w)) <= half_width
            while int(slice_mask.sum()) < min_slice_points and half_width < width_span:
                half_width *= 1.7
                slice_mask = np.abs(candidate_coords[:, 1].astype(np.float32) - float(target_w)) <= half_width
            if not slice_mask.any():
                slice_mask = np.ones(candidate_coords.shape[0], dtype=bool)

            slice_indices = np.where(slice_mask)[0]
            slice_coords = candidate_coords[slice_indices]
            target_hs = np.quantile(slice_coords[:, 0], h_quants)

            for target_h in target_hs:
                target = np.asarray([target_h, target_w], dtype=np.float32)
                distances = np.sum((slice_coords.astype(np.float32) - target[None, :]) ** 2, axis=1)
                order = np.argsort(distances)
                chosen_global_idx = int(slice_indices[order[0]])
                for local_idx in order:
                    candidate_global_idx = int(slice_indices[local_idx])
                    if candidate_global_idx not in used_indices:
                        chosen_global_idx = candidate_global_idx
                        break
                used_indices.add(chosen_global_idx)
                selected.append(candidate_coords[chosen_global_idx])

        selected = np.asarray(selected, dtype=np.int64)
        return (
            torch.from_numpy(selected[:, 0]).to(self.device, dtype=torch.long),
            torch.from_numpy(selected[:, 1]).to(self.device, dtype=torch.long),
        )
            
    def update_fg_obj_info(self, all_obj_points):
        for idx, per_obj_point in enumerate(all_obj_points):
            self.fg_pcs[idx]['points'] = per_obj_point.clone()
    

    def estimate_plane_normal_simple(self, vertices):
        """
        Simple version - estimate plane normal vector
        
        Parameters:
        -----------
        vertices : np.ndarray, shape (N, 3)
            vertex coordinates
        
        Returns:
        --------
        normal : np.ndarray, shape (3,)
            unit normal vector [x, y, z]
        """
        centroid = np.mean(vertices, axis=0)
        centered = vertices - centroid
        
        cov_matrix = np.cov(centered.T)
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        
        normal = eigenvecs[:, 0]
        
        return normal
    

    def remap_depth(self, depth_map, remap_depth, valid_mask=None, percentile_clip=95):
        depth_map = depth_map.clone()
        
        valid_depths = depth_map[valid_mask]
        
        clip_max = torch.quantile(valid_depths, percentile_clip / 100.0)
        
        min_val = valid_depths.min()
        max_val = clip_max
        
        if max_val - min_val < 1e-8:
            return depth_map
        
        normalized = torch.zeros_like(depth_map)
        clipped_depths = torch.clamp(depth_map[valid_mask], max=clip_max)
        normalized[valid_mask] = (clipped_depths - min_val) / (max_val - min_val)
        
        remapped = normalized * (remap_depth[1] - remap_depth[0]) + remap_depth[0]
        
        remapped[~valid_mask] = torch.max(remapped[valid_mask])
        
        return remapped
