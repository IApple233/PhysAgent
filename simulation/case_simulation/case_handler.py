"""
Base Case Handler Template
Abstract base class for all simulation case handlers.
"""

from abc import ABC, abstractmethod
import importlib
import numpy as np
import torch
import gstaichi as ti
import genesis as gs
import sys
import trimesh

CASE_REGISTRY = {}

def register_case(case_name: str):
    """
    A decorator to automatically register the CaseHandler subclass to CASE_REGISTRY.
    """
    def decorator(cls):
        if case_name in CASE_REGISTRY:
            raise ValueError(f"Case name '{case_name}' already registered!")
        
        # Register: map the string case_name to the actual Class Object
        CASE_REGISTRY[case_name] = cls
        print(f"Registered Case: '{case_name}' -> {cls.__name__}")
        return cls # Return the unmodified class
    return decorator

class CaseHandler(ABC):
    """
    Abstract base class for handling case-specific simulation logic.
    Each simulation case should inherit from this class.
    """
    
    def __init__(self, config, all_obj_info: list[dict], device: torch.device):
        self.config = config
        self.all_obj_info = all_obj_info
        self.device = device
        self._static_collision_geometry_cache = {}

    def set_simulation_bounds(self, all_obj_occupied_lower_bound, all_obj_occupied_upper_bound):
        self.all_obj_occupied_lower_bound = all_obj_occupied_lower_bound
        self.all_obj_occupied_upper_bound = all_obj_occupied_upper_bound
        mins = [self.all_obj_occupied_lower_bound]
        maxs = [self.all_obj_occupied_upper_bound]
        for spec in self.config.get('static_collision_objects', []) or []:
            bounds = spec.get('bounds_gs') or {}
            bounds_min = np.asarray(bounds.get('min', []), dtype=np.float64)
            bounds_max = np.asarray(bounds.get('max', []), dtype=np.float64)
            if bounds_min.shape != (3,) or bounds_max.shape != (3,):
                continue

            offset = self.get_case_static_support_offset_gs(spec)
            if offset is None:
                offset = np.zeros(3, dtype=np.float64)
            bounds_min = bounds_min + offset
            bounds_max = bounds_max + offset
            mins.append(torch.as_tensor(bounds_min, device=self.device, dtype=self.all_obj_occupied_lower_bound.dtype))
            maxs.append(torch.as_tensor(bounds_max, device=self.device, dtype=self.all_obj_occupied_upper_bound.dtype))

        self.all_obj_occupied_lower_bound = torch.stack(mins, dim=0).amin(dim=0)
        self.all_obj_occupied_upper_bound = torch.stack(maxs, dim=0).amax(dim=0)
        self.all_obj_occupied_size = self.all_obj_occupied_upper_bound - self.all_obj_occupied_lower_bound
        self.simulation_lower_bound = self.all_obj_occupied_lower_bound - 1 * self.all_obj_occupied_size
        self.simulation_upper_bound = self.all_obj_occupied_upper_bound + 1 * self.all_obj_occupied_size

    def get_simulation_bounds(self):
        return self.simulation_lower_bound.cpu().numpy(), self.simulation_upper_bound.cpu().numpy()
    

    def add_entities_to_scene(self, scene, obj_materials, obj_vis_modes):
        self.obj_materials = obj_materials
        self.obj_vis_modes = obj_vis_modes
        self.scene = scene
        self.objs = []
        if 'is_obj_fixed' not in self.config:
            is_obj_fixed = [False] * len(self.all_obj_info)
        else:
            is_obj_fixed = self.config['is_obj_fixed']
        print(len(self.all_obj_info))
        print(len(self.obj_materials))
        for idx, per_obj_info in enumerate(self.all_obj_info):
            if "use_primitive" in self.config and self.config['use_primitive']:

                primitive_morhph = gs.morphs.Box(
                        pos=self.all_obj_info[idx]['center'].cpu().numpy().astype(np.float64),
                        size=self.all_obj_info[idx]['size'].cpu().numpy().astype(np.float64),
                        visualization=True,
                        collision=True,
                        fixed=False,
                    )
                per_obj = self.scene.add_entity(
                    material = self.obj_materials[idx],
                    morph = primitive_morhph,
                    surface = gs.surfaces.Default(
                        color = tuple(np.random.rand(3).tolist() + [1.0]),
                        vis_mode = self.obj_vis_modes[idx],
                    ),
                )
            else:
                # try:
                morph = gs.morphs.Mesh(
                        file = per_obj_info['mesh_path'],
                        scale = 1.0,
                        pos = tuple(per_obj_info['center'].cpu().numpy().astype(np.float64)),
                        euler = (0.0, 0.0, 0.0),
                        fixed = is_obj_fixed[idx],
                        # decimate = self.config['decimate'],
                        # convexify = self.config['convexify'],
                    )
                per_obj = self.scene.add_entity(
                    material = self.obj_materials[idx],
                    morph = morph,
                    # morph = gs.morphs.Box(
                    #     pos = per_obj_info['center'].cpu().numpy(),
                    #     size = per_obj_info['size'].cpu().numpy(),
                    # ),
                    surface = gs.surfaces.Default(
                        color = tuple(np.random.rand(3).tolist() + [1.0]),
                        vis_mode = self.obj_vis_modes[idx],
                    ),
                )
                # except Exception as e:
                #     print(e)
                #     import pdb; pdb.set_trace()
                #     print("trying to add primitive mesh for object", idx)
                #     primitive_morhph = gs.morphs.Box(
                #         pos=self.all_obj_info[idx]['center'].cpu().numpy().astype(np.float64),
                #         size=self.all_obj_info[idx]['size'].cpu().numpy().astype(np.float64),
                #         visualization=True,
                #         collision=True,
                #         fixed=False,
                #     )
                #     per_obj = self.scene.add_entity(
                #         material = self.obj_materials[idx],
                #         morph = primitive_morhph,
                #         surface = gs.surfaces.Default(
                #             color = tuple(np.random.rand(3).tolist() + [1.0]),
                #             vis_mode = self.obj_vis_modes[idx],
                #         ),
                #     )
            self.objs.append(per_obj)
    
        return self.objs



    
    def before_scene_building(self, scene, all_objs, ground_plane):
        self.scene = scene
        self.all_objs = all_objs
        self._add_static_collision_objects()
        # Initial anti-penetration shifts are intentionally disabled for now.
        # Keep the reconstructed static support in the scene, but preserve the
        # raw reconstructed dynamic-object layout for debugging.
        self.config['static_support_overlap_resolution_enabled'] = False
        self.detect_ground_plane(ground_plane)
        self.create_force_fields()
        self.add_robots()
        self.custom_setup()
        self.add_emitters()
    
    def after_scene_building(self):
        self.init_robots_pose()
        self.fix_particles()

    def custom_simulation(self, sid):
        pass

    def after_simulation_step(self, svr):
        pass

    def add_emitters(self):
        """Add emitters if needed for this case."""
        pass

    ## before scene building
    def _configured_background_plane_point(self):
        point = self.config.get('background_plane_point_gs', None)
        if point is None:
            return None
        point = np.asarray(point, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            return None
        return point

    def _apply_ground_plane_offset(self, anchor, normal):
        offset = float(self.config.get('background_plane_offset', self.config.get('ground_plane_offset', 0.0)))
        return anchor - normal * offset

    def _support_anchor_from_object_vertices(self, normal):
        vertices = []
        for obj_info in self.all_obj_info:
            obj_vertices = obj_info.get('vertices')
            if obj_vertices is None:
                continue
            if isinstance(obj_vertices, torch.Tensor):
                obj_vertices = obj_vertices.detach().cpu().numpy()
            vertices.append(obj_vertices)

        if not vertices:
            return self.all_obj_occupied_lower_bound.cpu().numpy()

        vertices = np.concatenate(vertices, axis=0)
        support_idx = int(np.argmin(vertices @ normal))
        anchor = vertices[support_idx].astype(np.float64)
        return anchor

    def _support_anchor_from_object_and_static_collision_vertices(self, normal):
        vertices = []
        for obj_info in self.all_obj_info:
            obj_vertices = obj_info.get('vertices')
            if obj_vertices is None:
                continue
            if isinstance(obj_vertices, torch.Tensor):
                obj_vertices = obj_vertices.detach().cpu().numpy()
            vertices.append(np.asarray(obj_vertices, dtype=np.float64))

        for spec in self._static_collision_object_specs():
            bounds = spec.get('bounds_gs') or {}
            bounds_min = np.asarray(bounds.get('min', []), dtype=np.float64)
            bounds_max = np.asarray(bounds.get('max', []), dtype=np.float64)
            if bounds_min.shape == (3,) and bounds_max.shape == (3,):
                offset = self._static_collision_position_offset(spec)
                vertices.append(self._bbox_corners(bounds_min + offset, bounds_max + offset))
                continue

            geometry = self._load_static_collision_geometry(spec)
            if geometry is not None and geometry.get('vertices') is not None:
                vertices.append(np.asarray(geometry['vertices'], dtype=np.float64))

        if not vertices:
            return self.all_obj_occupied_lower_bound.cpu().numpy()

        vertices = np.concatenate(vertices, axis=0)
        support_idx = int(np.argmin(vertices @ normal))
        anchor = vertices[support_idx].astype(np.float64)
        return anchor

    def _ground_anchor_from_reconstructed_background(self, normal):
        mode = self.config.get('background_plane_position_mode', 'background_depth')
        if mode == 'object_support':
            return self._support_anchor_from_object_vertices(normal)
        if mode != 'background_depth':
            print(
                f"Warning: unknown background_plane_position_mode '{mode}', "
                "falling back to background_depth."
            )

        anchor = self._configured_background_plane_point()
        if anchor is not None:
            return anchor
        return self._support_anchor_from_object_vertices(normal)

    def _ground_collision_material(self):
        return gs.materials.Rigid(
            rho=1000.0 if 'plane_rho' not in self.config else self.config['plane_rho'],
            friction=5 if 'plane_friction' not in self.config else self.config['plane_friction'],
            coup_friction=5.0 if 'plane_coup_friction' not in self.config else self.config['plane_coup_friction'],
            coup_softness=0.002 if 'plane_coup_softness' not in self.config else self.config['plane_coup_softness'],
        )

    def _background_collision_mode(self):
        mode = self.config.get('background_collision_mode', None)
        if mode is None:
            if self.config.get('use_reconstructed_background_collision', False):
                return 'mesh'
            return 'plane'
        return str(mode).strip().lower()

    def add_infinite_plane_with_static_support(self):
        return bool(self.config.get('add_infinite_plane_with_static_support', False))

    def force_front_view_ground_plane(self):
        if 'force_front_view_ground_plane' in self.config:
            return bool(self.config.get('force_front_view_ground_plane'))
        if 'force_horizontal_ground_plane' in self.config:
            return bool(self.config.get('force_horizontal_ground_plane'))
        return bool(
            not self.config.get('use_reconstructed_background_ground_plane', False)
        )

    def _static_collision_object_specs(self):
        specs = self.config.get('static_collision_objects', [])
        if specs is None:
            return []
        return list(specs)

    def get_case_static_support_offset_gs(self, spec):
        """Return a case-specific Genesis-space offset for a static support mesh."""
        return None

    def _static_collision_position_offset(self, spec):
        offset = self.get_case_static_support_offset_gs(spec)
        if offset is None:
            offset = [0.0, 0.0, 0.0]

        offset = np.asarray(offset, dtype=np.float64)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            print(f"Warning: invalid static support position offset {offset}; using [0, 0, 0].")
            return np.zeros(3, dtype=np.float64)
        return offset

    def _normalize_vector(self, vector, default=None):
        vector = np.asarray(vector, dtype=np.float64)
        norm = np.linalg.norm(vector)
        if norm < 1e-8:
            if default is None:
                return None
            return np.asarray(default, dtype=np.float64)
        return vector / norm

    def _support_reference_normal(self):
        for key in (
            'static_support_plane_normal_gs',
            'background_plane_normal_gs',
            'gravity_direction',
        ):
            if key not in self.config:
                continue
            normal = self._normalize_vector(self.config.get(key), default=None)
            if normal is not None:
                return normal
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)

    def _configured_static_support_plane(self, default_normal):
        point = self.config.get('static_support_plane_point_gs', None)
        normal = self._normalize_vector(
            self.config.get('static_support_plane_normal_gs', default_normal),
            default=default_normal,
        )
        if point is None or normal is None:
            return None, None
        point = np.asarray(point, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            return None, None
        return point, normal

    def _orthonormal_tangent_basis(self, normal):
        normal = self._normalize_vector(normal, default=[0.0, 0.0, 1.0])
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(helper, normal)) > 0.9:
            helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        tangent_u = np.cross(normal, helper)
        tangent_u = self._normalize_vector(tangent_u, default=[1.0, 0.0, 0.0])
        tangent_v = np.cross(normal, tangent_u)
        tangent_v = self._normalize_vector(tangent_v, default=[0.0, 1.0, 0.0])
        return tangent_u, tangent_v

    def _bbox_corners(self, bounds_min, bounds_max):
        bounds_min = np.asarray(bounds_min, dtype=np.float64)
        bounds_max = np.asarray(bounds_max, dtype=np.float64)
        return np.asarray(
            [
                [bounds_min[0], bounds_min[1], bounds_min[2]],
                [bounds_min[0], bounds_min[1], bounds_max[2]],
                [bounds_min[0], bounds_max[1], bounds_min[2]],
                [bounds_min[0], bounds_max[1], bounds_max[2]],
                [bounds_max[0], bounds_min[1], bounds_min[2]],
                [bounds_max[0], bounds_min[1], bounds_max[2]],
                [bounds_max[0], bounds_max[1], bounds_min[2]],
                [bounds_max[0], bounds_max[1], bounds_max[2]],
            ],
            dtype=np.float64,
        )

    def _fit_plane_normal(self, points, up_dir):
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] < 3:
            return self._normalize_vector(up_dir, default=[0.0, 0.0, 1.0])
        centered = points - points.mean(axis=0, keepdims=True)
        try:
            _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return self._normalize_vector(up_dir, default=[0.0, 0.0, 1.0])
        if singular_values[-1] < 1e-8 and singular_values[0] < 1e-8:
            return self._normalize_vector(up_dir, default=[0.0, 0.0, 1.0])
        normal = vh[-1]
        normal = self._normalize_vector(normal, default=up_dir)
        if np.dot(normal, up_dir) < 0:
            normal = -normal
        return normal

    def _load_static_collision_geometry(self, spec):
        mesh_path = spec.get('mesh_path_gs') or spec.get('mesh_path') or spec.get('file')
        if not mesh_path:
            return None
        offset = self._static_collision_position_offset(spec)
        cache_key = (str(mesh_path), tuple(np.round(offset, 8).tolist()))
        if cache_key in self._static_collision_geometry_cache:
            return self._static_collision_geometry_cache[cache_key]
        try:
            mesh = trimesh.load_mesh(mesh_path, process=False)
        except Exception as exc:
            print(f"Warning: failed to load static collision mesh '{mesh_path}' for overlap checks: {exc}")
            self._static_collision_geometry_cache[cache_key] = None
            return None

        vertices = np.asarray(mesh.vertices, dtype=np.float64) + offset[None, :]
        faces = np.asarray(mesh.faces, dtype=np.int64)
        face_centers = None
        face_normals = None
        if faces.ndim == 2 and faces.shape[0] > 0:
            triangles = vertices[faces]
            face_centers = triangles.mean(axis=1)
            raw_normals = np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            )
            normal_norm = np.linalg.norm(raw_normals, axis=1, keepdims=True)
            valid = normal_norm[:, 0] > 1e-8
            if valid.any():
                face_normals = raw_normals[valid] / normal_norm[valid]
                face_centers = face_centers[valid]
            else:
                face_centers = None
                face_normals = None

        geometry = {
            'vertices': vertices,
            'face_centers': face_centers,
            'face_normals': face_normals,
        }
        self._static_collision_geometry_cache[cache_key] = geometry
        return geometry

    def _select_support_patch_vertices(self, support_vertices, object_vertices, normal, tangent_margin):
        tangent_u, tangent_v = self._orthonormal_tangent_basis(normal)
        object_uv = np.stack(
            [
                object_vertices @ tangent_u,
                object_vertices @ tangent_v,
            ],
            axis=1,
        )
        support_uv = np.stack(
            [
                support_vertices @ tangent_u,
                support_vertices @ tangent_v,
            ],
            axis=1,
        )

        obj_min_uv = object_uv.min(axis=0) - tangent_margin
        obj_max_uv = object_uv.max(axis=0) + tangent_margin
        in_patch = np.all((support_uv >= obj_min_uv) & (support_uv <= obj_max_uv), axis=1)
        patch_vertices = support_vertices[in_patch]
        if patch_vertices.shape[0] >= 12:
            return patch_vertices

        object_center_uv = object_uv.mean(axis=0)
        tangent_distance_sq = np.sum((support_uv - object_center_uv[None, :]) ** 2, axis=1)
        nearest_count = min(max(32, patch_vertices.shape[0]), support_vertices.shape[0])
        nearest_idx = np.argsort(tangent_distance_sq)[:nearest_count]
        return support_vertices[nearest_idx]

    def _compute_support_separation(self, support_spec, object_vertices, up_dir, clearance, tangent_margin):
        geometry = self._load_static_collision_geometry(support_spec)
        if geometry is None or geometry['vertices'].shape[0] == 0:
            return None

        support_vertices = geometry['vertices']
        patch_vertices = self._select_support_patch_vertices(
            support_vertices,
            object_vertices,
            up_dir,
            tangent_margin,
        )
        if patch_vertices.shape[0] == 0:
            return None

        local_normal = self._fit_plane_normal(patch_vertices, up_dir)
        tangent_u, tangent_v = self._orthonormal_tangent_basis(local_normal)

        object_uv = np.stack(
            [
                object_vertices @ tangent_u,
                object_vertices @ tangent_v,
            ],
            axis=1,
        )
        patch_uv = np.stack(
            [
                patch_vertices @ tangent_u,
                patch_vertices @ tangent_v,
            ],
            axis=1,
        )
        overlap_u = min(object_uv[:, 0].max(), patch_uv[:, 0].max()) - max(object_uv[:, 0].min(), patch_uv[:, 0].min())
        overlap_v = min(object_uv[:, 1].max(), patch_uv[:, 1].max()) - max(object_uv[:, 1].min(), patch_uv[:, 1].min())
        if overlap_u <= 0.0 or overlap_v <= 0.0:
            return None

        support_along_normal = patch_vertices @ local_normal
        object_along_normal = object_vertices @ local_normal
        support_height = float(np.quantile(support_along_normal, 0.95))
        object_height = float(np.quantile(object_along_normal, 0.01))
        delta = support_height - object_height + clearance
        if delta <= 0.0:
            return None
        return local_normal * delta

    def _coarse_support_separation(self, support_spec, obj_min, obj_max, clearance, lateral_margin):
        bounds = support_spec.get('bounds_gs') or {}
        support_min = np.asarray(bounds.get('min', []), dtype=np.float64)
        support_max = np.asarray(bounds.get('max', []), dtype=np.float64)
        if support_min.shape != (3,) or support_max.shape != (3,):
            return None

        overlap_x = min(obj_max[0], support_max[0]) - max(obj_min[0], support_min[0])
        overlap_y = min(obj_max[1], support_max[1]) - max(obj_min[1], support_min[1])
        overlap_z = min(obj_max[2], support_max[2]) - max(obj_min[2], support_min[2])
        if overlap_x <= lateral_margin or overlap_y <= lateral_margin or overlap_z <= 0.0:
            return None
        return np.array([0.0, 0.0, float(support_max[2] - obj_min[2] + clearance)], dtype=np.float64)

    def _static_support_bbox_separation(
        self,
        support_spec,
        obj_min,
        obj_max,
        normal,
        clearance,
        tangent_margin,
        support_plane_point=None,
    ):
        normal = self._normalize_vector(normal, default=[0.0, 0.0, 1.0])
        geometry = self._load_static_collision_geometry(support_spec)
        bounds = support_spec.get('bounds_gs') or {}
        support_min = np.asarray(bounds.get('min', []), dtype=np.float64)
        support_max = np.asarray(bounds.get('max', []), dtype=np.float64)
        if support_min.shape != (3,) or support_max.shape != (3,):
            return None

        object_corners = self._bbox_corners(obj_min, obj_max)
        support_corners = self._bbox_corners(support_min, support_max)

        object_center = 0.5 * (np.asarray(obj_min, dtype=np.float64) + np.asarray(obj_max, dtype=np.float64))
        support_center = 0.5 * (support_min + support_max)
        if (object_center - support_center) @ normal < 0.0:
            normal = -normal

        tangent_u, tangent_v = self._orthonormal_tangent_basis(normal)
        object_uv = np.stack([object_corners @ tangent_u, object_corners @ tangent_v], axis=1)
        support_uv = np.stack([support_corners @ tangent_u, support_corners @ tangent_v], axis=1)
        overlap_u = min(object_uv[:, 0].max(), support_uv[:, 0].max() + tangent_margin) - max(
            object_uv[:, 0].min(), support_uv[:, 0].min() - tangent_margin
        )
        overlap_v = min(object_uv[:, 1].max(), support_uv[:, 1].max() + tangent_margin) - max(
            object_uv[:, 1].min(), support_uv[:, 1].min() - tangent_margin
        )
        if overlap_u <= 0.0 or overlap_v <= 0.0:
            return None

        support_height = None
        if geometry is not None and geometry['vertices'].shape[0] > 0:
            support_vertices = geometry['vertices']
            support_uv_vertices = np.stack(
                [support_vertices @ tangent_u, support_vertices @ tangent_v],
                axis=1,
            )
            obj_min_uv = object_uv.min(axis=0) - tangent_margin
            obj_max_uv = object_uv.max(axis=0) + tangent_margin
            in_patch = np.all(
                (support_uv_vertices >= obj_min_uv) & (support_uv_vertices <= obj_max_uv),
                axis=1,
            )
            patch_vertices = support_vertices[in_patch]
            if patch_vertices.shape[0] > 0:
                height_quantile = float(self.config.get('static_support_local_height_quantile', 1.0))
                height_quantile = float(np.clip(height_quantile, 0.0, 1.0))
                support_height = float(np.quantile(patch_vertices @ normal, height_quantile))

        if support_height is None and support_plane_point is not None:
            support_plane_point = np.asarray(support_plane_point, dtype=np.float64)
            if support_plane_point.shape == (3,) and np.isfinite(support_plane_point).all():
                support_height = float(support_plane_point @ normal)
        if support_height is None:
            support_height = float((support_corners @ normal).max())
        object_height = float((object_corners @ normal).min())
        delta = support_height - object_height + clearance
        if delta <= 0.0:
            return None
        return normal * delta, normal

    def _update_dynamic_object_info_after_shift(self, obj_idx, delta):
        obj_info = self.all_obj_info[obj_idx]
        shift = torch.as_tensor(delta, device=self.device, dtype=obj_info['center'].dtype)
        for key in ('min', 'max', 'center'):
            if key in obj_info and isinstance(obj_info[key], torch.Tensor):
                obj_info[key] = obj_info[key] + shift
        if 'vertices' in obj_info and isinstance(obj_info['vertices'], torch.Tensor):
            obj_info['vertices'] = obj_info['vertices'] + shift
        mins = [info['min'] for info in self.all_obj_info if isinstance(info.get('min'), torch.Tensor)]
        maxs = [info['max'] for info in self.all_obj_info if isinstance(info.get('max'), torch.Tensor)]
        if mins and maxs:
            self.all_obj_occupied_lower_bound = torch.stack(mins, dim=0).amin(dim=0)
            self.all_obj_occupied_upper_bound = torch.stack(maxs, dim=0).amax(dim=0)
            self.all_obj_occupied_size = self.all_obj_occupied_upper_bound - self.all_obj_occupied_lower_bound
            self.simulation_lower_bound = self.all_obj_occupied_lower_bound - 3 * self.all_obj_occupied_size
            self.simulation_upper_bound = self.all_obj_occupied_upper_bound + 3 * self.all_obj_occupied_size

    def _dynamic_support_footprint_groups(self, normal, tangent_margin):
        object_indices = []
        tangent_bounds = []
        tangent_u, tangent_v = self._orthonormal_tangent_basis(normal)
        group_margin = float(self.config.get('static_support_dynamic_group_margin', 0.0))

        for obj_idx, obj_info in enumerate(self.all_obj_info):
            if 'vertices' not in obj_info or not isinstance(obj_info.get('vertices'), torch.Tensor):
                continue
            vertices = obj_info['vertices'].detach().cpu().numpy().astype(np.float64)
            if vertices.size == 0:
                continue
            uv = np.stack([vertices @ tangent_u, vertices @ tangent_v], axis=1)
            object_indices.append(obj_idx)
            tangent_bounds.append((uv.min(axis=0), uv.max(axis=0)))

        parent = list(range(len(object_indices)))

        def find(idx):
            while parent[idx] != idx:
                parent[idx] = parent[parent[idx]]
                idx = parent[idx]
            return idx

        def union(a, b):
            root_a = find(a)
            root_b = find(b)
            if root_a != root_b:
                parent[root_b] = root_a

        for i in range(len(object_indices)):
            min_i, max_i = tangent_bounds[i]
            for j in range(i + 1, len(object_indices)):
                min_j, max_j = tangent_bounds[j]
                overlap_u = min(max_i[0], max_j[0]) - max(min_i[0], min_j[0])
                overlap_v = min(max_i[1], max_j[1]) - max(min_i[1], min_j[1])
                if overlap_u >= -group_margin and overlap_v >= -group_margin:
                    union(i, j)

        groups_by_root = {}
        for local_idx, obj_idx in enumerate(object_indices):
            groups_by_root.setdefault(find(local_idx), []).append(obj_idx)
        return list(groups_by_root.values())

    def _set_dynamic_entity_center(self, obj_entity, new_center, obj_idx):
        new_center_array = np.asarray(new_center, dtype=np.float64)
        if getattr(obj_entity, 'is_built', False):
            obj_entity.set_pos(
                np.asarray(new_center, dtype=np.float32),
                zero_velocity=True,
            )
            return True
        if hasattr(obj_entity, 'morph') and hasattr(obj_entity.morph, 'pos'):
            obj_entity.morph.pos = tuple(new_center_array.tolist())
            return True
        print(
            "Warning: unable to update entity pose before scene build while "
            "resolving overlap with static support; skipping runtime shift for "
            f"object_index={obj_idx}."
        )
        return False

    def _resolve_dynamic_static_support_overlap_grouped(
        self,
        specs,
        support_normal,
        support_plane_point,
        clearance,
        tangent_margin,
        max_passes,
    ):
        groups = self._dynamic_support_footprint_groups(support_normal, tangent_margin)
        if not groups:
            return False

        shifted_any = False
        for group in groups:
            object_vertices_by_idx = {}
            for obj_idx in group:
                obj_info = self.all_obj_info[obj_idx]
                object_vertices_by_idx[obj_idx] = obj_info['vertices'].detach().cpu().numpy().astype(np.float64)

            total_shift = np.zeros(3, dtype=np.float64)
            shift_normal = support_normal

            for _ in range(max_passes):
                group_vertices = np.concatenate(list(object_vertices_by_idx.values()), axis=0)
                group_min = group_vertices.min(axis=0)
                group_max = group_vertices.max(axis=0)
                best_shift = None
                best_shift_norm = 0.0

                for spec in specs:
                    candidate_result = self._static_support_bbox_separation(
                        spec,
                        group_min,
                        group_max,
                        support_normal,
                        clearance,
                        tangent_margin,
                        support_plane_point=support_plane_point,
                    )
                    if candidate_result is None:
                        continue
                    candidate_shift, candidate_normal = candidate_result
                    candidate_norm = float(np.linalg.norm(candidate_shift))
                    if candidate_norm > best_shift_norm:
                        best_shift = candidate_shift
                        shift_normal = candidate_normal
                        best_shift_norm = candidate_norm

                if best_shift is None or best_shift_norm <= 1e-8:
                    break

                total_shift += best_shift
                for obj_idx in object_vertices_by_idx:
                    object_vertices_by_idx[obj_idx] = object_vertices_by_idx[obj_idx] + best_shift[None, :]

            total_shift_norm = float(np.linalg.norm(total_shift))
            if total_shift_norm <= 1e-8:
                continue

            updated_indices = []
            for obj_idx in group:
                obj_entity = self.all_objs[obj_idx]
                obj_info = self.all_obj_info[obj_idx]
                current_center = obj_info['center'].detach().cpu().numpy().astype(np.float64)
                new_center = current_center + total_shift
                if not self._set_dynamic_entity_center(obj_entity, new_center, obj_idx):
                    continue
                self._update_dynamic_object_info_after_shift(obj_idx, total_shift)
                updated_indices.append(obj_idx)

            if not updated_indices:
                continue

            shifted_any = True
            shift_along_normal = float(total_shift @ shift_normal)
            shift_lateral = total_shift - shift_along_normal * shift_normal
            detail = (
                f"object_indices={updated_indices}, delta={np.round(total_shift, 6).tolist()}, "
                f"along_support_normal={shift_along_normal:.6f}, "
                f"lateral_norm={float(np.linalg.norm(shift_lateral)):.6f}"
            )
            if len(updated_indices) > 1:
                detail += ", preserved_dynamic_stack=True"
            print("Shifted dynamic object group to resolve initial overlap with static support: " + detail)

        return shifted_any

    def _resolve_dynamic_static_support_overlap(self):
        specs = self._static_collision_object_specs()
        if not specs or not getattr(self, 'all_objs', None):
            return

        clearance = float(self.config.get('static_support_clearance', 0.03))
        lateral_margin = float(self.config.get('static_support_overlap_margin', 0.0))
        tangent_margin = float(self.config.get('static_support_patch_margin', max(0.05, clearance * 2.0)))
        max_passes = max(1, int(self.config.get('static_support_resolution_passes', 3)))
        up_dir = self._support_reference_normal()
        collision_mode = self._background_collision_mode()
        use_static_support_bbox = (
            collision_mode in {'static_support', 'static_support_only'}
            or bool(self.config.get('static_support_replaces_background_collision', False))
        )
        support_normal = self._normalize_vector(
            self.config.get('static_support_plane_normal_gs', up_dir),
            default=up_dir,
        )
        support_plane_point, _ = self._configured_static_support_plane(default_normal=support_normal)

        if use_static_support_bbox and bool(self.config.get('static_support_preserve_dynamic_stacks', True)):
            if self._resolve_dynamic_static_support_overlap_grouped(
                specs,
                support_normal,
                support_plane_point,
                clearance,
                tangent_margin,
                max_passes,
            ):
                return

        for obj_idx, (obj_entity, obj_info) in enumerate(zip(self.all_objs, self.all_obj_info)):
            if 'vertices' not in obj_info or not isinstance(obj_info.get('vertices'), torch.Tensor):
                continue

            total_shift = np.zeros(3, dtype=np.float64)
            shift_normal = support_normal
            object_vertices = obj_info['vertices'].detach().cpu().numpy().astype(np.float64)

            for _ in range(max_passes):
                best_shift = None
                best_shift_norm = 0.0
                obj_min = object_vertices.min(axis=0)
                obj_max = object_vertices.max(axis=0)

                for spec in specs:
                    if use_static_support_bbox:
                        candidate_result = self._static_support_bbox_separation(
                            spec,
                            obj_min,
                            obj_max,
                            support_normal,
                            clearance,
                            tangent_margin,
                            support_plane_point=support_plane_point,
                        )
                        if candidate_result is None:
                            candidate_shift = None
                            candidate_normal = support_normal
                        else:
                            candidate_shift, candidate_normal = candidate_result
                    else:
                        candidate_shift = self._compute_support_separation(
                            spec,
                            object_vertices,
                            up_dir,
                            clearance,
                            tangent_margin,
                        )
                        if candidate_shift is None:
                            candidate_shift = self._coarse_support_separation(
                                spec,
                                obj_min,
                                obj_max,
                                clearance,
                                lateral_margin,
                            )
                    if candidate_shift is None:
                        continue
                    candidate_norm = float(np.linalg.norm(candidate_shift))
                    if candidate_norm > best_shift_norm:
                        best_shift = candidate_shift
                        if use_static_support_bbox:
                            shift_normal = candidate_normal
                        best_shift_norm = candidate_norm

                if best_shift is None or best_shift_norm <= 1e-8:
                    break

                total_shift += best_shift
                object_vertices = object_vertices + best_shift[None, :]

            total_shift_norm = float(np.linalg.norm(total_shift))
            if total_shift_norm <= 1e-8:
                continue

            current_center = obj_info['center'].detach().cpu().numpy().astype(np.float64)
            new_center = current_center + total_shift
            shift_along_normal = None
            shift_lateral_norm = None
            if use_static_support_bbox:
                shift_along_normal = float(total_shift @ shift_normal)
                shift_lateral = total_shift - shift_along_normal * shift_normal
                shift_lateral_norm = float(np.linalg.norm(shift_lateral))

            if not self._set_dynamic_entity_center(obj_entity, new_center, obj_idx):
                continue

            self._update_dynamic_object_info_after_shift(obj_idx, total_shift)
            detail = f"object_index={obj_idx}, delta={np.round(total_shift, 6).tolist()}"
            if shift_along_normal is not None and shift_lateral_norm is not None:
                detail += (
                    f", along_support_normal={shift_along_normal:.6f}, "
                    f"lateral_norm={shift_lateral_norm:.6f}"
                )
            print("Shifted dynamic object to resolve initial overlap with static support: " + detail)

    def _add_static_collision_objects(self):
        self.static_collision_entities = []
        for spec in self._static_collision_object_specs():
            mesh_path = spec.get('mesh_path_gs') or spec.get('mesh_path') or spec.get('file')
            if not mesh_path:
                continue
            try:
                offset = self._static_collision_position_offset(spec)
                morph = gs.morphs.Mesh(
                    file=mesh_path,
                    scale=1.0,
                    pos=tuple(offset.tolist()),
                    euler=(0.0, 0.0, 0.0),
                    fixed=True,
                    visualization=bool(spec.get('visualization', self.config.get('static_collision_visualization', False))),
                    collision=True,
                    # New support meshes are capped during reconstruction; this is a fallback for older or unusually dense meshes.
                    decimate=bool(spec.get('decimate', self.config.get('static_collision_decimate', True))),
                    decimate_face_num=int(spec.get('decimate_face_num', self.config.get('static_collision_decimate_face_num', 5000))),
                    convexify=bool(spec.get('convexify', self.config.get('static_collision_convexify', False))),
                )
                entity = self.scene.add_entity(
                    material=self._ground_collision_material(),
                    morph=morph,
                    surface=gs.surfaces.Default(
                        color=tuple(spec.get('color', self.config.get('static_collision_color', [0.45, 0.45, 0.45, 0.25]))),
                        vis_mode=spec.get('vis_mode', self.config.get('static_collision_vis_mode', 'collision')),
                    ),
                )
            except Exception as exc:
                print(f"Warning: failed to add static collision object '{mesh_path}': {exc}")
                continue

            self.static_collision_entities.append(entity)
            print(
                "Using static reconstructed support collision object: "
                f"name={spec.get('name', 'unnamed')}, path={mesh_path}, "
                f"case_support_offset_gs={np.round(offset, 6).tolist()}"
            )
        self.static_collision_objects_added = len(self.static_collision_entities) > 0
        return self.static_collision_entities

    def _add_reconstructed_background_collision_mesh(self):
        mesh_path = self.config.get('background_collision_mesh_path_gs', None)
        if not mesh_path:
            return None

        try:
            morph = gs.morphs.Mesh(
                file=mesh_path,
                scale=1.0,
                pos=(0.0, 0.0, 0.0),
                euler=(0.0, 0.0, 0.0),
                fixed=True,
                visualization=bool(self.config.get('background_collision_mesh_visualization', False)),
                collision=True,
                decimate=bool(self.config.get('background_collision_mesh_decimate', True)),
                decimate_face_num=int(self.config.get('background_collision_mesh_decimate_face_num', 5000)),
                convexify=bool(self.config.get('background_collision_mesh_convexify', False)),
            )
            entity = self.scene.add_entity(
                material=self._ground_collision_material(),
                morph=morph,
                surface=gs.surfaces.Default(
                    color=tuple(self.config.get('background_collision_mesh_color', [0.45, 0.45, 0.45, 0.25])),
                    vis_mode=self.config.get('background_collision_mesh_vis_mode', 'collision'),
                ),
            )
        except Exception as exc:
            print(f"Warning: failed to add reconstructed background collision mesh '{mesh_path}': {exc}")
            return None

        self.background_collision_mesh = entity
        print(
            "Using reconstructed background collision mesh instead of an infinite plane: "
            f"path={mesh_path}"
        )
        return entity

    def detect_ground_plane(self, ground_plane):
        """Detect ground plane specific to this case."""
        if self.force_front_view_ground_plane():
            self.normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if self._static_collision_object_specs():
                self.ground_anchor = self._support_anchor_from_object_and_static_collision_vertices(self.normal)
            else:
                self.ground_anchor = self._support_anchor_from_object_vertices(self.normal)
            self.ground_anchor = self._apply_ground_plane_offset(self.ground_anchor, self.normal)
            self.config['background_plane_normal_gs'] = self.normal.tolist()
            self.config['background_plane_point_gs'] = self.ground_anchor.tolist()
            self.config['gravity_direction'] = self.normal.tolist()
            print(
                "Using front-view horizontal ground plane: "
                f"normal={np.round(self.normal, 4).tolist()}, "
                f"anchor={np.round(self.ground_anchor, 4).tolist()}"
            )
        elif (
            self.add_infinite_plane_with_static_support()
            and self._static_collision_object_specs()
        ):
            self.normal = np.array([0, 0, 1], dtype=np.float64)
            self.ground_anchor = self._support_anchor_from_object_and_static_collision_vertices(self.normal)
            self.ground_anchor = self._apply_ground_plane_offset(self.ground_anchor, self.normal)
            self.config['background_plane_normal_gs'] = self.normal.tolist()
            self.config['background_plane_point_gs'] = self.ground_anchor.tolist()
            self.config['gravity_direction'] = self.normal.tolist()
            print(
                "Using horizontal infinite plane with static support collision: "
                f"normal={np.round(self.normal, 4).tolist()}, "
                f"anchor={np.round(self.ground_anchor, 4).tolist()}"
            )
        elif ground_plane is None:
            self.normal = np.array([0, 0, 1], dtype=np.float64)
            self.ground_anchor = self.all_obj_occupied_lower_bound.cpu().numpy()
            self.ground_anchor[2] = self.ground_anchor[2]
        else:
            self.normal = np.asarray(ground_plane, dtype=np.float64)
            norm = np.linalg.norm(self.normal)
            if norm < 1e-8:
                self.normal = np.array([0, 0, 1], dtype=np.float64)
            else:
                self.normal = self.normal / norm
            self.ground_anchor = self._ground_anchor_from_reconstructed_background(self.normal)
            self.ground_anchor = self._apply_ground_plane_offset(self.ground_anchor, self.normal)
            print(
                "Using reconstructed background normal for support collision: "
                f"normal={np.round(self.normal, 4).tolist()}, "
                f"anchor={np.round(self.ground_anchor, 4).tolist()}"
            )
        if ground_plane is None:
            self.ground_anchor = self._apply_ground_plane_offset(self.ground_anchor, self.normal)

        collision_mode = self._background_collision_mode()
        if (
            collision_mode in {'static_support', 'static_support_only'}
            and getattr(self, 'static_collision_objects_added', False)
            and not self.add_infinite_plane_with_static_support()
        ):
            return
        if (
            self.config.get('static_support_replaces_background_collision', False)
            and getattr(self, 'static_collision_objects_added', False)
            and collision_mode in {'mesh', 'background_mesh', 'background_mesh_only'}
        ):
            return
        mesh_entity = None
        if collision_mode in {'mesh', 'both', 'background_mesh', 'background_mesh_only'}:
            mesh_entity = self._add_reconstructed_background_collision_mesh()
            if mesh_entity is not None and collision_mode in {'mesh', 'background_mesh', 'background_mesh_only'}:
                return
            if mesh_entity is None:
                print("Warning: falling back to the infinite ground plane because no background mesh was added.")

        self.scene.add_entity(
            material = self._ground_collision_material(),
            morph = gs.morphs.Plane(pos=(self.ground_anchor[0], self.ground_anchor[1], self.ground_anchor[2]), normal=self.normal)
        )

    
    def create_force_fields(self):
        """Create case-specific force fields."""
        pass
    
    def custom_setup(self):
        """Custom setup for this case."""
        pass
    
    def add_robots(self):
        """Setup robots if needed for this case."""
        pass
    

    ## after scene building
    def init_robots_pose(self):
        """Initialize robots pose if needed for this case."""
        pass

    def fix_particles(self):
        """Fix particles if needed for this case."""
        pass



    def extract_franka_mesh_data_combined(self, target_franka):
        """
        Extract and combine all mesh data into single arrays with transformations applied.
        
        Returns:
            vertices: torch tensor of all transformed vertices
            faces: torch tensor of all faces (with proper indexing)
            colors: torch tensor of per-vertex colors
        """
        
        all_vertices = []
        all_faces = []
        all_colors = []
        
        vertex_offset = 0
        sim_vgeoms_render_T = target_franka.solver._vgeoms_render_T
        
        for vgeom in target_franka.vgeoms:
            verts = vgeom.vmesh.verts  # shape: (N, 3)
            faces = vgeom.vmesh.faces
            
            # Get transformation matrix for this vgeom
            cur_render_T = sim_vgeoms_render_T[vgeom.idx][0]  # shape: (4, 4), remove batch dim
            
            # Apply transformation to vertices
            # Convert vertices to homogeneous coordinates (N, 4)
            verts_homogeneous = np.concatenate([verts, np.ones((len(verts), 1))], axis=1)
            
            # Apply transformation: (N, 4) @ (4, 4)^T = (N, 4)
            verts_transformed = verts_homogeneous @ cur_render_T.T
            
            # Convert back to 3D coordinates (N, 3)
            verts_transformed = verts_transformed[:, :3]
            
            # Get color from surface
            surface = vgeom.vmesh.surface
            if hasattr(surface, 'diffuse_texture') and surface.diffuse_texture is not None:
                color = surface.diffuse_texture.color
            elif surface.color is not None:
                color = surface.color
            else:
                color = (0.5, 0.5, 0.5)
            
            # Offset faces by current vertex count
            faces_offset = faces + vertex_offset
            
            # Create per-vertex colors
            vertex_colors = np.tile(color, (len(verts), 1))
            
            all_vertices.append(verts_transformed)
            all_faces.append(faces_offset)
            all_colors.append(vertex_colors)
            
            vertex_offset += len(verts)
        
        vertices = torch.from_numpy(np.vstack(all_vertices)).to(self.device, dtype=torch.float32) # + self.franka_pos
        faces = torch.from_numpy(np.vstack(all_faces)).to(self.device, dtype=torch.int32)
        colors = torch.from_numpy(np.vstack(all_colors)).to(self.device, dtype=torch.float32)
        
        return vertices, faces, colors

def get_case_handler(case_name: str, config, all_obj_info, device) -> CaseHandler:
    """
    Factory function to return the corresponding CaseHandler instance based on the case name.
    """
    if case_name not in CASE_REGISTRY:
        try:
            importlib.import_module(f"simulation.case_simulation.{case_name}")
        except ModuleNotFoundError as exc:
            if exc.name == f"simulation.case_simulation.{case_name}":
                available = sorted(CASE_REGISTRY.keys())
                raise ValueError(
                    f"Unknown case name: '{case_name}'. Available registered cases: {available}"
                ) from exc
            raise

    if case_name not in CASE_REGISTRY:
        available = sorted(CASE_REGISTRY.keys())
        raise ValueError(
            f"Case '{case_name}' was imported but did not register itself. Registered cases: {available}"
        )

    CaseClass = CASE_REGISTRY[case_name]
    return CaseClass(config, all_obj_info, device)
