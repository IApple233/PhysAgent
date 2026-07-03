import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import trimesh
import cv2
import os
import gc
import genesis as gs
from omegaconf import ListConfig
from pathlib import Path
from pytorch3d.renderer import PerspectiveCameras
from simulation.image23D.single_view_reconstructor import SingleViewReconstructor
from simulation.utils import (
    pt3d_to_gs,
    gs_to_pt3d,
    PRESET_Z_VALUE,
    save_gif_from_image_folder,
    save_video_from_pil,
    pose_to_transform_matrix,
)
from simulation.case_simulation.case_handler import get_case_handler
import time
from simulation.utils import save_gif_from_image_folder

class DiffSim(nn.Module):
    def __init__(self, config): 
        super().__init__()
        self.config = config
        self.device = self.config['device']
        self.output_folder = Path(self.config['output_folder']) / 'simulation'

        self.genesis_frames = self.output_folder / "gs_frames"
        self.genesis_frames.mkdir(parents=True, exist_ok=True)
        self.output_folder.mkdir(parents=True, exist_ok=True)

        self.dt = self.config["dt"]
        self.substeps = self.config["substeps"]
        self.simulated_frames_num = self.config["simulated_frames_num"]
        self.frame_steps = self.config["frame_steps"]
        self.simulation_steps = self.simulated_frames_num * self.frame_steps
        self.material_type = self.config['material_type']

        self.svr = SingleViewReconstructor(config)
        self.fg_pcs_from_3d, self.fg_meshes, self.ground_plane_normal, self.config = self.svr.reconstruct()
        self._setup_world_alignment()
        self._align_reconstructed_config_geometry()
        self._background_collision_normal_space = 'pt3d'
        self._background_collision_point_space = 'pt3d'
        self.background_collision_normal = self._get_background_collision_normal()
        self.background_collision_point = self._get_background_collision_point()
        if self.background_collision_normal is None:
            self.background_collision_normal = self._get_static_support_normal()
            self._background_collision_normal_space = getattr(self, '_static_support_normal_space', 'pt3d')
        if self.background_collision_point is None:
            self.background_collision_point = self._get_static_support_point()
            self._background_collision_point_space = getattr(self, '_static_support_point_space', 'pt3d')

        if (
            self.config.get('add_infinite_plane_with_static_support', False)
            and self.config.get('static_collision_objects')
        ):
            self.background_collision_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            self._background_collision_normal_space = 'gs'

        if (
            self.background_collision_normal is not None
            and self.config.get('background_collision_sets_gravity', True)
        ):
            self.ground_plane_normal = self.background_collision_normal

        # initialize the proxy primitived for foreground object
        if self.ground_plane_normal is not None:
            if self._background_collision_normal_space == 'gs':
                self.ground_plane_normal = np.asarray(self.ground_plane_normal, dtype=np.float32)
            else:
                self.ground_plane_normal = self._pt3d_vector_to_world_gs(self.ground_plane_normal)
            if getattr(self, 'world_alignment_enabled', False):
                self.ground_plane_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            if self.ground_plane_normal[2] < 0:
                self.ground_plane_normal = -self.ground_plane_normal
            self.config['gravity_direction'] = self.ground_plane_normal.tolist()
            self.config['background_plane_normal_gs'] = self.ground_plane_normal.tolist()
        if self.background_collision_point is not None:
            if self._background_collision_point_space == 'gs':
                self.background_collision_point = np.asarray(self.background_collision_point, dtype=np.float32)
            else:
                self.background_collision_point = self._pt3d_to_world_gs(self.background_collision_point)
            self.config['background_plane_point_gs'] = self.background_collision_point.tolist()
        self.fg_pcs = []
        for idx, per_obj_pc in enumerate(self.fg_pcs_from_3d):
            self.fg_pcs.append({
                'points': self._pt3d_to_world_gs(per_obj_pc['points'].clone()),
                'colors': pt3d_to_gs(per_obj_pc['colors'].clone()),
            })
        
        # pytorch to genesis coordinates
        for idx, per_obj_mesh in enumerate(self.fg_meshes):
            per_obj_mesh['vertices'] = self._pt3d_to_world_gs(per_obj_mesh['vertices'])

        gs.init(
            seed=self.config['seed'],
            precision="32",
            backend=gs.gpu,
            logging_level="warning"
        )

        # get the global bounding box for all foreground objects
        self.all_obj_info = []
        self.all_obj_occupied_lower_bound = torch.tensor([float('inf'), float('inf'), float('inf')]).to(self.device)
        self.all_obj_occupied_upper_bound = torch.tensor([float('-inf'), float('-inf'), float('-inf')]).to(self.device)


        for idx, per_mesh_bounds in enumerate(self.fg_meshes):
            per_mesh_min = self.fg_meshes[idx]['vertices'].min(0).values
            per_mesh_max = self.fg_meshes[idx]['vertices'].max(0).values
            per_mesh_center = self.fg_meshes[idx]['vertices'].mean(0)
            per_mesh_size = per_mesh_max - per_mesh_min

            self.fg_meshes[idx]['vertices'] -= per_mesh_center
            # self.fg_pcs[idx]['points'] -= per_mesh_center
            per_obj_mesh_path = os.path.join(self.config['output_folder'], f'fg_mesh_{idx:02d}.obj')
            
            per_trimesh = trimesh.Trimesh(
                vertices=self.fg_meshes[idx]['vertices'].cpu().numpy(), 
                faces=self.fg_meshes[idx]['faces'].cpu().numpy(), 
                vertex_colors=self.fg_meshes[idx]['colors'].cpu().numpy()
            )
            
            per_trimesh.export(per_obj_mesh_path)

            self.all_obj_info.append({
                'min': per_mesh_min,
                'max': per_mesh_max,
                'center': per_mesh_center,
                'size': per_mesh_size,
                'mesh_path': per_obj_mesh_path,
                'vertices': self.fg_meshes[idx]['vertices'] + per_mesh_center,
            })
            

            self.all_obj_occupied_lower_bound = torch.minimum(self.all_obj_occupied_lower_bound, per_mesh_min)
            self.all_obj_occupied_upper_bound = torch.maximum(self.all_obj_occupied_upper_bound, per_mesh_max)

        self.case_handler = get_case_handler(self.config['example_name'], self.config, self.all_obj_info, self.device)
        self.case_handler.set_simulation_bounds(self.all_obj_occupied_lower_bound, self.all_obj_occupied_upper_bound)
        self.simulation_lower_bound, self.simulation_upper_bound = self.case_handler.get_simulation_bounds()

        if self._force_front_view_ground_plane():
            self.ground_plane_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            self.config['gravity_direction'] = self.ground_plane_normal.tolist()
            gravity_dir = self.ground_plane_normal.copy()
        elif self.ground_plane_normal is not None:
            gravity_dir = self.ground_plane_normal.copy()
        else:
            gravity_dir = np.array([0, 0, 1])

        if 'mpm_gravity' in self.config:
            if isinstance(self.config['mpm_gravity'], (int, float)):
                mpm_gravity = tuple(self.config['mpm_gravity'] * np.array(gravity_dir))
            else:
                mpm_gravity = tuple(self._pt3d_vector_to_world_gs(np.array(self.config['mpm_gravity'])))
        else:
            mpm_gravity = None

        if 'pbd_gravity' in self.config:
            if isinstance(self.config['pbd_gravity'], (int, float)):
                pbd_gravity = tuple(self.config['pbd_gravity'] * np.array(gravity_dir))
            else:
                pbd_gravity = tuple(self._pt3d_vector_to_world_gs(np.array(self.config['pbd_gravity'])))
        else:
            pbd_gravity = None

        if 'gravity' in self.config:
            if isinstance(self.config['gravity'], (int, float)):
                gravity = tuple(self.config['gravity'] * np.array(gravity_dir))
            else:
                gravity = tuple(self._pt3d_vector_to_world_gs(np.array(self.config['gravity'])))
        else:
            gravity = tuple(-9.8 * np.array(gravity_dir))
        
        # initialize the genesis scene
        self.scene = gs.Scene(
            sim_options = gs.options.SimOptions(
                dt=self.dt,
                gravity=gravity,
                substeps=self.substeps,
            ),
            show_viewer=False,
            vis_options = gs.options.VisOptions(
                show_world_frame = False,
                world_frame_size = 1.0,
                show_link_frame  = False,
                show_cameras     = False,
                plane_reflection = False,
                ambient_light    = (0.5, 0.5, 0.5),
                lights = [{
                    'type': 'directional',
                    'dir': (0, 0, 1),
                    'color': (1.0, 1.0, 1.0),
                    'intensity': 2.0
                }]
            ),
            renderer = gs.renderers.Rasterizer(),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                enable_collision=True,
                enable_self_collision=False,
                constraint_timeconst = 0.02,
            ),
            pbd_options = gs.options.PBDOptions(
                lower_bound = tuple(self.simulation_lower_bound),
                upper_bound = tuple(self.simulation_upper_bound),
                particle_size = 0.01 if 'particle_size' not in self.config else self.config['particle_size'],
                gravity = pbd_gravity,
            ),
            mpm_options = gs.options.MPMOptions(
                lower_bound = tuple(self.simulation_lower_bound),
                upper_bound = tuple(self.simulation_upper_bound),
                grid_density = 64 if 'MPM_grid_density' not in self.config else self.config['MPM_grid_density'],
                particle_size = 0.01 if 'particle_size' not in self.config else self.config['particle_size'],
                gravity = mpm_gravity,
            ),
            coupler_options=gs.options.LegacyCouplerOptions(
                rigid_pbd=True,
                rigid_mpm=True,
            )
        )

        # get materials for each object
        self.obj_materials= []
        self.obj_vis_modes = []
        for idx, per_material_type in enumerate(self.material_type):
            obj_material, obj_vis_mode = self.get_material_for_each(per_material_type, idx)
            self.obj_materials.append(obj_material)
            self.obj_vis_modes.append(obj_vis_mode)

        self.objs = self.case_handler.add_entities_to_scene(self.scene, self.obj_materials, self.obj_vis_modes)
        W, H = self.svr.target_size
        camera_params = self._setup_input_matching_camera(W, H)
        self._sync_svr_render_camera(camera_params, W, H)

        # 然后再触发 CaseHandler 的生命周期钩子，此时里面就能正确读到 camera_z_top 了
        self.case_handler.before_scene_building(self.scene, self.objs, self.ground_plane_normal)
        self.export_initial_scene_layout()

        # 接着正常添加相机
        self.cam = self.scene.add_camera(
            res    = (W, H),
            pos    = tuple(camera_params['pos']),
            lookat = tuple(camera_params['lookat']),
            up     = tuple(camera_params['up']),
            fov    = camera_params['fov_y_degrees'],
            GUI    = False,
        )

        self.scene.build()
        self.case_handler.after_scene_building()

        # transform and binding
        # self.original_transform_matrix = {}
        self.closest_indices = {}
        self.initial_transform_matrix = {}

        for obj_idx, per_material_type in enumerate(self.material_type):
            if per_material_type == 'rigid':
                # for debug purpose
                # per_object_pos = self.objs[obj_idx].get_pos().cpu().numpy()
                # per_object_quat = self.objs[obj_idx].get_quat().cpu().numpy()
                # per_object_transform_matrix = torch.from_numpy(pose_to_transform_matrix(per_object_pos, per_object_quat)).to(self.device).float()
                # self.initial_transform_matrix[obj_idx] = per_object_transform_matrix
                self.objs[obj_idx].solver.update_vgeoms_render_T()
                rigid_T = self.objs[obj_idx].solver._vgeoms_render_T
                rigid_idx = self.objs[obj_idx].idx
                transform_matrix = torch.tensor(rigid_T[rigid_idx, 0]).to(self.device).float()
                self.initial_transform_matrix[obj_idx] = transform_matrix

            elif per_material_type in ['pbd_liquid', 'pbd_cloth', 'mpm_sand', 'mpm_liquid', 'mpm_elastic', 'mpm_snow', 'mpm_elastic2plastic', 'pbd_elastic', 'pbd_particle']:
                self.closest_indices[obj_idx] = self.map_pc_to_particles(obj_idx)
            else:
                raise NotImplementedError("The current material is not supported for now")

        # --- [Debug] 排查沙子粒子最低点与地面高度 ---
        for obj_idx, per_material_type in enumerate(self.material_type):
            if per_material_type == 'mpm_sand':
                # 获取 Genesis 初始化的物理粒子坐标
                init_particles = self.objs[obj_idx].init_particles
                
                # 假设 Z 轴是重力方向，找出粒子的最低点 Z 坐标
                particles_z_min = init_particles[:, 2].min()
                particles_z_max = init_particles[:, 2].max()
                
                print(f"================ DEBUG INFO ================")
                print(f"Object {obj_idx} ({per_material_type}):")
                print(f"  -> Particles Z-Min (最低点): {particles_z_min:.6f}")
                print(f"  -> Particles Z-Max (最高点): {particles_z_max:.6f}")
                
                # 打印整个仿真的边界（通常决定了 MPM 求解器的隐式地板）
                print(f"  -> Simulation Lower Bound: {self.simulation_lower_bound}")
                print(f"  -> Ground Plane Normal: {self.ground_plane_normal}")
                print(f"============================================")
        # ----------------------------------------------
        print("genesis scene construction finished")

    def _normalize_np(self, value, default):
        value = np.asarray(value, dtype=np.float64)
        if value.shape != (3,) or not np.isfinite(value).all():
            value = np.asarray(default, dtype=np.float64)
        norm = np.linalg.norm(value)
        if norm < 1e-8:
            return np.asarray(default, dtype=np.float64)
        return value / norm

    def _rotation_from_vectors(self, src, dst):
        src = self._normalize_np(src, [0.0, 0.0, 1.0])
        dst = self._normalize_np(dst, [0.0, 0.0, 1.0])
        dot = float(np.clip(np.dot(src, dst), -1.0, 1.0))
        if dot > 1.0 - 1e-8:
            return np.eye(3, dtype=np.float64)
        if dot < -1.0 + 1e-8:
            helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            if abs(np.dot(src, helper)) > 0.9:
                helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            axis = self._normalize_np(np.cross(src, helper), [0.0, 0.0, 1.0])
            return -np.eye(3, dtype=np.float64) + 2.0 * np.outer(axis, axis)

        axis = np.cross(src, dst)
        skew = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ],
            dtype=np.float64,
        )
        axis_norm_sq = float(np.dot(axis, axis))
        return np.eye(3, dtype=np.float64) + skew + skew @ skew * ((1.0 - dot) / axis_norm_sq)

    def _apply_world_alignment_gs(self, xyz):
        if isinstance(xyz, torch.Tensor):
            rotation = torch.as_tensor(self.world_alignment_R_gs, device=xyz.device, dtype=xyz.dtype)
            return torch.matmul(xyz, rotation.transpose(0, 1))
        xyz = np.asarray(xyz)
        return np.matmul(xyz, self.world_alignment_R_gs.T)

    def _apply_inverse_world_alignment_gs(self, xyz):
        if isinstance(xyz, torch.Tensor):
            rotation = torch.as_tensor(self.world_alignment_R_gs, device=xyz.device, dtype=xyz.dtype)
            return torch.matmul(xyz, rotation)
        xyz = np.asarray(xyz)
        return np.matmul(xyz, self.world_alignment_R_gs)

    def _add_aligned_z_offset(self, xyz):
        if PRESET_Z_VALUE == 0:
            return xyz
        if isinstance(xyz, torch.Tensor):
            out = xyz.clone()
        else:
            out = np.asarray(xyz).copy()
        out[..., 2] += PRESET_Z_VALUE
        return out

    def _remove_aligned_z_offset(self, xyz):
        if PRESET_Z_VALUE == 0:
            return xyz
        if isinstance(xyz, torch.Tensor):
            out = xyz.clone()
        else:
            out = np.asarray(xyz).copy()
        out[..., 2] -= PRESET_Z_VALUE
        return out

    def _pt3d_to_world_gs(self, xyz, no_z_offset=False):
        direct_gs = pt3d_to_gs(xyz, no_z_offset=True)
        world_gs = self._apply_world_alignment_gs(direct_gs)
        if not no_z_offset:
            world_gs = self._add_aligned_z_offset(world_gs)
        return world_gs

    def _pt3d_vector_to_world_gs(self, xyz):
        direct_gs = pt3d_to_gs(xyz, no_z_offset=True)
        return self._apply_world_alignment_gs(direct_gs)

    def _world_gs_to_pt3d(self, xyz, no_z_offset=False):
        world_gs = xyz if no_z_offset else self._remove_aligned_z_offset(xyz)
        direct_gs = self._apply_inverse_world_alignment_gs(world_gs)
        return gs_to_pt3d(direct_gs, no_z_offset=True)

    def _world_gs_vector_to_pt3d(self, xyz):
        direct_gs = self._apply_inverse_world_alignment_gs(xyz)
        return gs_to_pt3d(direct_gs, no_z_offset=True)

    def _setup_world_alignment(self):
        self.world_alignment_R_gs = np.eye(3, dtype=np.float64)
        self.world_alignment_enabled = False
        self.config['world_alignment_enabled'] = False
        self.config['world_alignment_mode'] = 'identity'

        if self._force_front_view_ground_plane():
            self.config['world_alignment_disabled_reason'] = 'force_front_view_ground_plane'
            self.config['world_alignment_source'] = 'front_view_horizontal_ground'
            self.config['background_plane_normal_gs'] = [0.0, 0.0, 1.0]
            self.config['gravity_direction'] = [0.0, 0.0, 1.0]
            return

        if not bool(self.config.get('align_reconstruction_to_ground', True)):
            self.config['world_alignment_disabled_reason'] = 'align_reconstruction_to_ground=false'
            return

        source = None
        normal = None
        point = None
        num_points = None
        support_normal = self.config.get('static_support_plane_normal_gs', None)
        support_point = self.config.get('static_support_plane_point_gs', None)
        prefer_static_support = bool(self.config.get('static_collision_objects')) and (
            bool(self.config.get('static_support_replaces_background_collision', False))
            or self._background_collision_mode() in {'static_support', 'static_support_only'}
        )
        if prefer_static_support and support_normal is not None:
            normal = np.asarray(support_normal, dtype=np.float64)
            point = (
                np.asarray(support_point, dtype=np.float64)
                if support_point is not None
                else np.zeros(3, dtype=np.float64)
            )
            source = 'static_support_plane'

        plane = None if normal is not None else self._fit_camera_ground_plane_from_background(align_world=False)
        if plane is not None:
            source = plane['source']
            normal = plane['raw_normal_camera_gs']
            point = plane['point_camera_gs']
            num_points = plane['num_points']
        else:
            if support_normal is not None:
                normal = np.asarray(support_normal, dtype=np.float64)
                point = (
                    np.asarray(support_point, dtype=np.float64)
                    if support_point is not None
                    else np.zeros(3, dtype=np.float64)
                )
                source = 'static_support_plane'

        if normal is None:
            self.config['world_alignment_disabled_reason'] = 'no_background_or_support_plane'
            return

        normal = self._normalize_np(normal, [0.0, 0.0, 1.0])
        if normal[2] < 0:
            normal = -normal
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        angle_degrees = float(np.rad2deg(np.arccos(np.clip(np.dot(normal, up), -1.0, 1.0))))
        min_angle = float(self.config.get('world_alignment_min_angle_degrees', 3.0))

        self.config['world_alignment_source'] = source
        self.config['world_alignment_input_normal_gs'] = normal.tolist()
        self.config['world_alignment_input_point_gs'] = (
            np.asarray(point, dtype=np.float64).tolist()
            if point is not None
            else [0.0, 0.0, 0.0]
        )
        self.config['world_alignment_angle_degrees'] = angle_degrees
        if num_points is not None:
            self.config['world_alignment_plane_num_points'] = int(num_points)

        if angle_degrees < min_angle:
            self.config['world_alignment_disabled_reason'] = (
                f'plane tilt {angle_degrees:.3f} deg < threshold {min_angle:.3f} deg'
            )
            return

        self.world_alignment_R_gs = self._rotation_from_vectors(normal, up)
        self.world_alignment_enabled = True
        self.config['world_alignment_enabled'] = True
        self.config['world_alignment_mode'] = 'rotate_ground_normal_to_z'
        self.config['world_alignment_rotation_gs'] = self.world_alignment_R_gs.tolist()
        self.config['world_alignment_target_normal_gs'] = up.tolist()
        print(
            "Applying reconstruction world alignment: "
            f"source={source}, angle={angle_degrees:.2f} deg, "
            f"normal={np.round(normal, 4).tolist()}"
        )

    def _transform_bounds_gs(self, bounds):
        if not bounds:
            return None
        bounds_min = np.asarray(bounds.get('min', []), dtype=np.float64)
        bounds_max = np.asarray(bounds.get('max', []), dtype=np.float64)
        if bounds_min.shape != (3,) or bounds_max.shape != (3,):
            return None
        corners = np.asarray(
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
        transformed = self._apply_world_alignment_gs(corners)
        return {
            'min': transformed.min(axis=0).tolist(),
            'max': transformed.max(axis=0).tolist(),
        }

    def _align_mesh_file_gs(self, mesh_path, label):
        if not mesh_path or not Path(mesh_path).exists():
            return None, None
        try:
            mesh = trimesh.load_mesh(mesh_path, process=False)
            mesh.vertices = self._apply_world_alignment_gs(np.asarray(mesh.vertices, dtype=np.float64))
        except Exception as exc:
            print(f"Warning: failed to align {label} mesh '{mesh_path}': {exc}")
            return None, None

        output_dir = Path(self.config['output_folder']) / 'world_alignment'
        output_dir.mkdir(parents=True, exist_ok=True)
        aligned_path = output_dir / f"{Path(mesh_path).stem}_world_aligned_gs.obj"
        try:
            mesh.export(aligned_path)
        except Exception as exc:
            print(f"Warning: failed to export aligned {label} mesh '{aligned_path}': {exc}")
            return None, None

        bounds = {
            'min': np.asarray(mesh.vertices, dtype=np.float64).min(axis=0).tolist(),
            'max': np.asarray(mesh.vertices, dtype=np.float64).max(axis=0).tolist(),
        }
        return str(aligned_path), bounds

    def _align_reconstructed_config_geometry(self):
        if not getattr(self, 'world_alignment_enabled', False):
            return

        for normal_key in ('background_plane_normal_gs', 'static_support_plane_normal_gs'):
            if normal_key in self.config:
                normal = self._normalize_np(self.config.get(normal_key), [0.0, 0.0, 1.0])
                normal = self._apply_world_alignment_gs(normal)
                normal = self._normalize_np(normal, [0.0, 0.0, 1.0])
                if normal[2] < 0:
                    normal = -normal
                self.config[normal_key] = normal.tolist()

        for point_key in ('background_plane_point_gs', 'static_support_plane_point_gs'):
            if point_key in self.config:
                point = np.asarray(self.config.get(point_key), dtype=np.float64)
                if point.shape == (3,) and np.isfinite(point).all():
                    self.config[point_key] = self._apply_world_alignment_gs(point).tolist()

        background_mesh_path = self.config.get('background_collision_mesh_path_gs', None)
        aligned_path, bounds = self._align_mesh_file_gs(background_mesh_path, 'background collision')
        if aligned_path is not None:
            self.config['background_collision_mesh_path_gs'] = aligned_path
            self.config['background_collision_mesh_bounds_gs'] = bounds

        for spec in self.config.get('static_collision_objects', []) or []:
            mesh_path = spec.get('mesh_path_gs') or spec.get('mesh_path') or spec.get('file')
            aligned_path, bounds = self._align_mesh_file_gs(mesh_path, f"static support {spec.get('name', '')}".strip())
            if aligned_path is not None:
                spec['mesh_path_gs'] = aligned_path
                if 'mesh_path' in spec:
                    spec['mesh_path'] = aligned_path
                if 'file' in spec:
                    spec['file'] = aligned_path
                spec['bounds_gs'] = bounds
            elif spec.get('bounds_gs') is not None:
                transformed_bounds = self._transform_bounds_gs(spec.get('bounds_gs'))
                if transformed_bounds is not None:
                    spec['bounds_gs'] = transformed_bounds

        self.config['world_alignment_applied_to_reconstruction_outputs'] = True

    def _fov_x_to_y_degrees(self, fov_x_degrees, width, height):
        fov_x_rad = np.deg2rad(float(fov_x_degrees))
        fov_y_rad = 2.0 * np.arctan(np.tan(fov_x_rad / 2.0) * (float(height) / float(width)))
        return float(np.rad2deg(fov_y_rad))

    def _camera_reference_depth(self, camera_pos, camera_forward):
        depths = []
        for obj_info in self.all_obj_info:
            vertices = obj_info.get('vertices')
            if vertices is None:
                center = obj_info.get('center')
                vertices = center[None, :] if center is not None else None
            if vertices is None:
                continue
            if isinstance(vertices, torch.Tensor):
                vertices = vertices.detach().cpu().numpy()
            vertices = np.asarray(vertices, dtype=np.float64)
            if vertices.ndim == 1:
                vertices = vertices[None, :]
            obj_depths = (vertices - camera_pos[None, :]) @ camera_forward
            obj_depths = obj_depths[np.isfinite(obj_depths) & (obj_depths > 1e-6)]
            if obj_depths.size:
                depths.append(obj_depths)

        for spec in self.config.get('static_collision_objects', []) or []:
            bounds = spec.get('bounds_gs') or {}
            bounds_min = np.asarray(bounds.get('min', []), dtype=np.float64)
            bounds_max = np.asarray(bounds.get('max', []), dtype=np.float64)
            if bounds_min.shape != (3,) or bounds_max.shape != (3,):
                continue
            corners = np.asarray(
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
            obj_depths = (corners - camera_pos[None, :]) @ camera_forward
            obj_depths = obj_depths[np.isfinite(obj_depths) & (obj_depths > 1e-6)]
            if obj_depths.size:
                depths.append(obj_depths)

        if not depths:
            return 1.0
        return float(np.median(np.concatenate(depths)))

    def _background_collision_mode(self):
        mode = self.config.get('background_collision_mode', None)
        if mode is None:
            if self.config.get('use_reconstructed_background_collision', False):
                return 'mesh'
            return 'plane'
        return str(mode).strip().lower()

    def _force_front_view_ground_plane(self):
        if 'force_front_view_ground_plane' in self.config:
            return bool(self.config.get('force_front_view_ground_plane'))
        if 'force_horizontal_ground_plane' in self.config:
            return bool(self.config.get('force_horizontal_ground_plane'))
        return bool(
            not self.config.get('use_reconstructed_background_ground_plane', False)
        )

    def _auto_camera_mode(self):
        mode = str(self.config.get('genesis_camera_mode', 'auto')).strip().lower()
        if mode != 'auto':
            return mode
        has_static_support = bool(self.config.get('static_collision_objects'))
        if self._background_collision_mode() == 'plane' and not has_static_support:
            return 'horizontal_ground_from_background'
        return 'match_input'

    def _fit_camera_ground_plane_from_background(self, align_world=True):
        points = getattr(self.svr, 'background_collision_points', None)
        depth_map = getattr(self.svr, 'background_collision_depth_map', None)
        valid_mask = getattr(self.svr, 'background_collision_valid_depth_mask', None)
        source = 'background_collision_points'
        if points is not None and not self.config.get('background_collision_uses_support_removed_inpainting', False):
            source = 'inpainted_background_points'
        elif points is not None:
            source = 'support_removed_background_collision_points'
        if points is None:
            points = getattr(self.svr, 'bg_points', None)
            depth_map = getattr(self.svr, 'bg_depth_map', None)
            valid_mask = getattr(self.svr, 'bg_valid_depth_mask', None)
            source = 'inpainted_background_points'
        if points is None or depth_map is None:
            return None

        height, width = depth_map.shape
        try:
            points_grid = points.reshape(height, width, 3)
        except Exception:
            return None

        stride = max(1, int(self.config.get('sim_camera_background_plane_stride', self.config.get('background_plane_stride', 8))))
        points_ds = points_grid[::stride, ::stride]
        depth_ds = depth_map[::stride, ::stride]
        valid_ds = torch.isfinite(points_ds).all(dim=-1) & torch.isfinite(depth_ds)
        if valid_mask is not None and bool(self.config.get('sim_camera_require_valid_background_depth', False)):
            valid_ds = valid_ds & valid_mask[::stride, ::stride].bool()

        roi_mask = None
        configured_roi = self.config.get('sim_camera_background_roi', None)
        if configured_roi is not None and hasattr(self.svr, '_roi_mask_from_config'):
            try:
                roi_mask = self.svr._roi_mask_from_config(configured_roi, height, width)
            except Exception as exc:
                print(f"Warning: invalid sim_camera_background_roi; falling back to background ROI: {exc}")
        if roi_mask is None and hasattr(self.svr, '_background_collision_roi_mask'):
            try:
                roi_mask = self.svr._background_collision_roi_mask(height, width)
            except Exception:
                roi_mask = None
        if roi_mask is not None:
            valid_ds = valid_ds & roi_mask[::stride, ::stride].bool()

        if valid_ds.sum().item() < 16:
            return None

        plane_points_pt3d = points_ds[valid_ds]
        max_points = int(self.config.get('sim_camera_background_plane_max_points', 20000))
        if plane_points_pt3d.shape[0] > max_points:
            sample_idx = torch.linspace(
                0,
                plane_points_pt3d.shape[0] - 1,
                steps=max_points,
                device=plane_points_pt3d.device,
            ).long()
            plane_points_pt3d = plane_points_pt3d[sample_idx]

        plane_points_gs = pt3d_to_gs(plane_points_pt3d.detach().cpu(), no_z_offset=True).numpy().astype(np.float64)
        if align_world:
            plane_points_gs = self._apply_world_alignment_gs(plane_points_gs).astype(np.float64)
        centered = plane_points_gs - plane_points_gs.mean(axis=0, keepdims=True)
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return None

        normal = vh[-1]
        normal = self._normalize_np(normal, [0.0, 0.0, 1.0])
        if normal[2] < 0:
            normal = -normal

        no_roll_normal = np.array([0.0, normal[1], normal[2]], dtype=np.float64)
        no_roll_normal = self._normalize_np(no_roll_normal, [0.0, 0.0, 1.0])
        if no_roll_normal[2] < 0:
            no_roll_normal = -no_roll_normal

        plane_point = plane_points_gs.mean(axis=0)
        signed_distance = float(np.dot(normal, plane_point))
        camera_height = abs(signed_distance)
        if not np.isfinite(camera_height) or camera_height < 1e-4:
            camera_height = float(self.config.get('sim_camera_default_height', 0.5))

        return {
            'normal_camera_gs': no_roll_normal,
            'raw_normal_camera_gs': normal,
            'point_camera_gs': plane_point,
            'height': float(camera_height),
            'source': source,
            'stride': int(stride),
            'num_points': int(plane_points_gs.shape[0]),
        }

    def _horizontal_ground_camera_from_background(self, fov_y_degrees, align_world_for_plane=False):
        plane = self._fit_camera_ground_plane_from_background(align_world=align_world_for_plane)
        if plane is None:
            return None

        normal_camera = plane['normal_camera_gs']
        forward_z = float(np.clip(normal_camera[1], -0.95, 0.95))
        forward_y = float(np.sqrt(max(1.0 - forward_z * forward_z, 1e-8)))
        camera_forward = self._normalize_np([0.0, forward_y, forward_z], [0.0, 1.0, 0.0])
        camera_up = self._normalize_np([0.0, -forward_z, forward_y], [0.0, 0.0, 1.0])

        height_scale = float(self.config.get('sim_camera_height_scale', 1.0))
        camera_height = max(float(plane['height']) * height_scale, float(self.config.get('sim_camera_min_height', 0.02)))
        camera_pos = np.array([0.0, 0.0, camera_height], dtype=np.float64)
        reference_depth = self._camera_reference_depth(camera_pos, camera_forward)
        reference_depth = max(reference_depth, float(self.config.get('sim_camera_min_lookat_distance', 0.5)))
        camera_lookat = camera_pos + camera_forward * reference_depth

        pitch_degrees = float(np.rad2deg(np.arcsin(np.clip(camera_forward[2], -1.0, 1.0))))
        self.config['sim_camera_background_plane_source'] = plane['source']
        self.config['sim_camera_background_plane_num_points'] = plane['num_points']
        self.config['sim_camera_background_plane_used_stride'] = plane['stride']
        self.config['sim_camera_ground_normal_camera_gs'] = normal_camera.tolist()
        self.config['sim_camera_ground_raw_normal_camera_gs'] = plane['raw_normal_camera_gs'].tolist()
        self.config['sim_camera_height_from_background_plane'] = float(camera_height)
        self.config['sim_camera_estimated_pose_gs'] = {
            'source': plane['source'],
            'mode': 'horizontal_ground_from_background',
            'pos': camera_pos.tolist(),
            'lookat': camera_lookat.tolist(),
            'up': camera_up.tolist(),
            'forward': camera_forward.tolist(),
            'fov_y_degrees': float(fov_y_degrees),
            'reference_depth': float(reference_depth),
            'pitch_degrees': float(pitch_degrees),
        }

        print(
            "Using horizontal-ground camera estimated from background: "
            f"normal_camera={np.round(normal_camera, 4).tolist()}, "
            f"height={camera_height:.4f}, pitch={pitch_degrees:.2f} deg"
        )
        return camera_pos, camera_lookat, camera_up, camera_forward, reference_depth, pitch_degrees

    def _transform_camera_pose_to_world_alignment(self, camera_pos, camera_lookat, camera_up, source_space, mode):
        source_space = str(source_space or 'direct_gs').strip().lower()
        already_world = source_space in {'world', 'world_gs', 'aligned_gs', 'world_aligned_gs'}
        should_transform = (
            getattr(self, 'world_alignment_enabled', False)
            and bool(self.config.get('preserve_input_camera_after_world_alignment', True))
            and not already_world
        )

        camera_pos = np.asarray(camera_pos, dtype=np.float64)
        camera_lookat = np.asarray(camera_lookat, dtype=np.float64)
        camera_up = np.asarray(camera_up, dtype=np.float64)
        if should_transform:
            camera_pos = self._apply_world_alignment_gs(camera_pos).astype(np.float64)
            camera_lookat = self._apply_world_alignment_gs(camera_lookat).astype(np.float64)
            camera_up = self._apply_world_alignment_gs(camera_up).astype(np.float64)

        self.config['sim_camera_pose_source_space'] = source_space
        self.config['sim_camera_pose_transformed_by_world_alignment'] = bool(should_transform)
        self.config['sim_camera_pose_transform_mode'] = mode
        return camera_pos, camera_lookat, camera_up

    def _setup_input_matching_camera(self, width, height):
        mode = self._auto_camera_mode()
        fov_x_degrees = float(self.config.get('fov_x_input', 90.0))
        fov_y_degrees = self._fov_x_to_y_degrees(fov_x_degrees, width, height)
        focal_length_pixels = float(float(width) / (2.0 * np.tan(np.deg2rad(fov_x_degrees) / 2.0)))
        camera_pose_space = str(self.config.get('camera_pose_space', 'direct_gs')).strip().lower()

        if mode in {'legacy', 'old', 'offset_origin'}:
            camera_pos = np.array([0.0, -1.0, 0.0], dtype=np.float64)
            camera_lookat = np.array([0.0, 0.0, 0.0], dtype=np.float64)
            camera_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            fov_y_degrees = float(self.config.get('genesis_legacy_camera_fov_degrees', fov_x_degrees))
        elif mode in {'manual', 'custom', 'explicit'}:
            camera_pos = np.asarray(self.config.get('sim_camera_pos_gs', [0.0, 0.0, 0.0]), dtype=np.float64)
            camera_lookat = np.asarray(self.config.get('sim_camera_lookat_gs', [0.0, 1.0, 0.0]), dtype=np.float64)
            camera_up = np.asarray(self.config.get('sim_camera_up_gs', [0.0, 0.0, 1.0]), dtype=np.float64)
            fov_y_degrees = float(self.config.get('sim_camera_fov_y_degrees', fov_y_degrees))
            camera_pose_space = str(
                self.config.get('manual_camera_space', self.config.get('camera_pose_space', 'direct_gs'))
            ).strip().lower()
        elif mode in {'horizontal_ground_from_background', 'ground_from_background', 'background_ground'}:
            camera_pose_space = 'direct_gs'
            if (
                getattr(self, 'world_alignment_enabled', False)
                and bool(self.config.get('preserve_input_camera_after_world_alignment', True))
            ):
                # The input reconstruction camera is already the canonical direct-GS pose.
                # Let world alignment rotate/translate that pose once, so Genesis and SVR
                # see the same camera used by reconstruction instead of a second heuristic fit.
                camera_pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                camera_lookat = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                camera_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                self.config['sim_camera_background_pose_rule'] = 'canonical_input_pose_transformed_by_world_alignment'
                print(
                    "Using canonical input camera transformed by world alignment "
                    "for horizontal-ground camera mode."
                )
            else:
                background_camera = self._horizontal_ground_camera_from_background(
                    fov_y_degrees,
                    align_world_for_plane=False,
                )
                if background_camera is None:
                    print("Warning: failed to estimate camera from background; falling back to match_input camera.")
                    camera_pos = np.asarray(self.config.get('sim_camera_pos_gs', [0.0, 0.0, 0.0]), dtype=np.float64)
                    camera_lookat = np.asarray(self.config.get('sim_camera_lookat_gs', [0.0, 1.0, 0.0]), dtype=np.float64)
                    camera_up = np.asarray(self.config.get('sim_camera_up_gs', [0.0, 0.0, 1.0]), dtype=np.float64)
                    mode = 'match_input_fallback'
                else:
                    camera_pos, camera_lookat, camera_up, _, _, _ = background_camera
        else:
            # PyTorch3D reconstruction camera: origin, x-left, y-up, z-forward.
            # Genesis coordinates: x-right, y-forward, z-up. Therefore the same
            # input camera is at the Genesis origin looking along +Y with +Z up.
            camera_pos = np.asarray(self.config.get('sim_camera_pos_gs', [0.0, 0.0, 0.0]), dtype=np.float64)
            camera_lookat = np.asarray(self.config.get('sim_camera_lookat_gs', [0.0, 1.0, 0.0]), dtype=np.float64)
            camera_up = np.asarray(self.config.get('sim_camera_up_gs', [0.0, 0.0, 1.0]), dtype=np.float64)
            fov_y_degrees = float(self.config.get('sim_camera_fov_y_degrees', fov_y_degrees))

        camera_pos, camera_lookat, camera_up = self._transform_camera_pose_to_world_alignment(
            camera_pos,
            camera_lookat,
            camera_up,
            camera_pose_space,
            mode,
        )
        camera_forward = self._normalize_np(camera_lookat - camera_pos, [0.0, 1.0, 0.0])
        camera_depth = self._camera_reference_depth(camera_pos, camera_forward)
        camera_up = self._normalize_np(camera_up, [0.0, 0.0, 1.0])
        camera_top_point = (
            camera_pos
            + camera_forward * camera_depth
            + camera_up * camera_depth * np.tan(np.deg2rad(fov_y_degrees) / 2.0)
        )
        camera_z_top = float(camera_top_point[2])

        gravity_ref = self._normalize_np(self.ground_plane_normal if self.ground_plane_normal is not None else [0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
        horizontal_forward = camera_forward - gravity_ref * np.dot(camera_forward, gravity_ref)
        if 'pitch_degrees' not in locals():
            pitch_degrees = float(np.rad2deg(np.arctan2(np.dot(camera_forward, gravity_ref), np.linalg.norm(horizontal_forward))))

        self.config['sim_camera_mode'] = mode
        self.config['sim_camera_resolution'] = [int(width), int(height)]
        self.config['sim_camera_pos_gs'] = camera_pos.tolist()
        self.config['sim_camera_lookat_gs'] = camera_lookat.tolist()
        self.config['sim_camera_forward_gs'] = camera_forward.tolist()
        self.config['sim_camera_up_gs'] = camera_up.tolist()
        self.config['sim_camera_fov_x_degrees'] = fov_x_degrees
        self.config['sim_camera_fov_y_degrees'] = fov_y_degrees
        self.config['sim_camera_focal_length_pixels'] = focal_length_pixels
        self.config['sim_camera_reference_depth'] = camera_depth
        self.config['sim_camera_pitch_degrees_from_gravity'] = pitch_degrees
        self.config['camera_z_top'] = camera_z_top
        self.config['sim_camera_resolved_pose_gs'] = {
            'mode': mode,
            'pos': camera_pos.tolist(),
            'lookat': camera_lookat.tolist(),
            'up': camera_up.tolist(),
            'forward': camera_forward.tolist(),
            'fov_y_degrees': float(fov_y_degrees),
            'reference_depth': float(camera_depth),
            'pitch_degrees_from_gravity': float(pitch_degrees),
        }

        return {
            'pos': camera_pos,
            'lookat': camera_lookat,
            'up': camera_up,
            'forward': camera_forward,
            'fov_y_degrees': fov_y_degrees,
        }

    def _sync_svr_render_camera(self, camera_params, width, height):
        render_camera_mode = str(
            self.config.get(
                'svr_render_camera_mode',
                self.config.get('render_camera_mode', 'input_image'),
            )
        ).strip().lower()
        sync_default = render_camera_mode in {'genesis', 'simulation', 'sim'}
        if not bool(self.config.get('sync_svr_render_camera_to_genesis', sync_default)):
            self.config['svr_render_camera_synced_to_genesis'] = False
            self.config['svr_render_camera_mode'] = 'input_image'
            return

        self.config['svr_render_camera_mode'] = 'genesis'
        camera_pos_gs = np.asarray(camera_params['pos'], dtype=np.float64)
        camera_forward_gs = self._normalize_np(camera_params['forward'], [0.0, 1.0, 0.0])
        camera_up_gs = self._normalize_np(camera_params['up'], [0.0, 0.0, 1.0])
        camera_right_gs = self._normalize_np(np.cross(camera_forward_gs, camera_up_gs), [1.0, 0.0, 0.0])

        camera_pos_pt3d = self._world_gs_to_pt3d(camera_pos_gs, no_z_offset=True)
        camera_forward_pt3d = self._normalize_np(self._world_gs_vector_to_pt3d(camera_forward_gs), [0.0, 0.0, 1.0])
        camera_up_pt3d = self._normalize_np(self._world_gs_vector_to_pt3d(camera_up_gs), [0.0, 1.0, 0.0])
        camera_right_pt3d = self._normalize_np(self._world_gs_vector_to_pt3d(camera_right_gs), [-1.0, 0.0, 0.0])
        camera_left_pt3d = -camera_right_pt3d

        R_np = np.stack([camera_left_pt3d, camera_up_pt3d, camera_forward_pt3d], axis=1)
        T_np = -(camera_pos_pt3d @ R_np)

        focal_length = float(self.config.get('sim_camera_focal_length_pixels', self.config.get('global_mesh_init_focal_length_pixels', width)))
        K = torch.zeros((1, 4, 4), device=self.device, dtype=torch.float32)
        K[0, 0, 0] = focal_length
        K[0, 1, 1] = focal_length
        K[0, 0, 2] = float(width) / 2.0
        K[0, 1, 2] = float(height) / 2.0
        K[0, 3, 2] = 1.0
        K[0, 2, 3] = 1.0

        R = torch.as_tensor(R_np, device=self.device, dtype=torch.float32).unsqueeze(0)
        T = torch.as_tensor(T_np, device=self.device, dtype=torch.float32).unsqueeze(0)
        self.svr.current_camera = PerspectiveCameras(
            K=K,
            R=R,
            T=T,
            in_ndc=False,
            image_size=((height, width),),
            device=self.device,
        )
        self.config['svr_render_camera_synced_to_genesis'] = True
        self.config['svr_render_camera_R'] = R_np.tolist()
        self.config['svr_render_camera_T'] = T_np.tolist()

    def _to_numpy(self, value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _mesh_vertex_colors(self, mesh, fallback):
        colors = None
        try:
            colors = np.asarray(mesh.visual.vertex_colors)
        except Exception:
            colors = None
        if colors is None or colors.shape[0] != len(mesh.vertices):
            colors = np.tile(np.asarray(fallback, dtype=np.float64), (len(mesh.vertices), 1))
        if colors.shape[1] == 3:
            alpha = np.full((colors.shape[0], 1), 255, dtype=colors.dtype)
            colors = np.concatenate([colors, alpha], axis=1)
        return colors[:, :4]

    def _write_obj_mesh(self, handle, name, vertices, faces, colors=None, vertex_offset=0):
        handle.write(f"o {name}\n")
        vertices = np.asarray(vertices, dtype=np.float64)
        if colors is not None:
            colors = np.asarray(colors)
        for idx, vertex in enumerate(vertices):
            if colors is not None and idx < colors.shape[0]:
                color = colors[idx]
                if color.max() > 1.0:
                    color = color[:3] / 255.0
                else:
                    color = color[:3]
                handle.write(
                    "v "
                    f"{vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f} "
                    f"{color[0]:.6f} {color[1]:.6f} {color[2]:.6f}\n"
                )
            else:
                handle.write(f"v {vertex[0]:.8f} {vertex[1]:.8f} {vertex[2]:.8f}\n")
        faces = np.asarray(faces, dtype=np.int64)
        for face in faces:
            face = face + 1 + vertex_offset
            handle.write(f"f {face[0]} {face[1]} {face[2]}\n")
        handle.write("\n")
        return vertex_offset + len(vertices)

    def export_initial_scene_layout(self):
        """Save one OBJ showing the initial dynamic objects and fixed support meshes."""
        output_path = Path(self.config['output_folder']) / "initial_scene_layout.obj"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        vertex_offset = 0
        exported_parts = []

        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write("# Initial RealWonder scene layout in Genesis coordinates.\n")
            handle.write("# Dynamic meshes are translated to their final initialization centers.\n\n")

            for obj_idx, obj_info in enumerate(self.all_obj_info):
                mesh_path = obj_info.get('mesh_path')
                if not mesh_path or not Path(mesh_path).exists():
                    continue
                try:
                    mesh = trimesh.load_mesh(mesh_path, process=False)
                except Exception as exc:
                    print(f"Warning: failed to load dynamic mesh for initial scene export: {mesh_path}: {exc}")
                    continue

                center = self._to_numpy(obj_info.get('center', [0.0, 0.0, 0.0])).astype(np.float64)
                vertices = np.asarray(mesh.vertices, dtype=np.float64) + center[None, :]
                colors = self._mesh_vertex_colors(mesh, fallback=[255, 120, 80, 255])
                vertex_offset = self._write_obj_mesh(
                    handle,
                    f"dynamic_object_{obj_idx:02d}",
                    vertices,
                    mesh.faces,
                    colors=colors,
                    vertex_offset=vertex_offset,
                )
                exported_parts.append(f"dynamic_object_{obj_idx:02d}")

            for support_idx, spec in enumerate(self.config.get('static_collision_objects', []) or []):
                mesh_path = spec.get('mesh_path_gs') or spec.get('mesh_path') or spec.get('file')
                if not mesh_path or not Path(mesh_path).exists():
                    continue
                try:
                    mesh = trimesh.load_mesh(mesh_path, process=False)
                except Exception as exc:
                    print(f"Warning: failed to load static support mesh for initial scene export: {mesh_path}: {exc}")
                    continue

                offset = self.case_handler._static_collision_position_offset(spec)
                vertices = np.asarray(mesh.vertices, dtype=np.float64) + offset[None, :]
                colors = self._mesh_vertex_colors(mesh, fallback=[115, 180, 255, 255])
                part_name = f"static_support_{support_idx:02d}_{spec.get('name', 'support')}"
                vertex_offset = self._write_obj_mesh(
                    handle,
                    part_name,
                    vertices,
                    mesh.faces,
                    colors=colors,
                    vertex_offset=vertex_offset,
                )
                exported_parts.append(part_name)

            background_mesh_path = self.config.get('background_collision_mesh_path_gs')
            if background_mesh_path and Path(background_mesh_path).exists():
                try:
                    mesh = trimesh.load_mesh(background_mesh_path, process=False)
                    colors = self._mesh_vertex_colors(mesh, fallback=[150, 150, 150, 255])
                    vertex_offset = self._write_obj_mesh(
                        handle,
                        "background_collision_mesh",
                        mesh.vertices,
                        mesh.faces,
                        colors=colors,
                        vertex_offset=vertex_offset,
                    )
                    exported_parts.append("background_collision_mesh")
                except Exception as exc:
                    print(f"Warning: failed to load background collision mesh for initial scene export: {background_mesh_path}: {exc}")

        self.config['initial_scene_layout_obj'] = str(output_path)
        print(
            "Initial scene layout OBJ saved: "
            f"{output_path} ({', '.join(exported_parts) if exported_parts else 'no mesh parts'})"
        )

    def _get_background_collision_normal(self):
        normal = getattr(self.svr, 'background_collision_normal', None)
        if normal is None:
            return None
        normal = np.asarray(normal, dtype=np.float32)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            return None
        return normal / norm

    def _get_background_collision_point(self):
        point = getattr(self.svr, 'background_plane_point', None)
        if point is None:
            point = self.config.get('background_plane_point_pt3d', None)
        if point is None:
            return None
        point = np.asarray(point, dtype=np.float32)
        if point.shape != (3,) or not np.isfinite(point).all():
            return None
        return point

    def _get_static_support_normal(self):
        normal = self.config.get('static_support_plane_normal_gs', None)
        if normal is not None:
            normal = np.asarray(normal, dtype=np.float32)
            norm = np.linalg.norm(normal)
            if norm >= 1e-6:
                self._static_support_normal_space = 'gs'
                return normal / norm

        normal = self.config.get('static_support_plane_normal_pt3d', None)
        if normal is None:
            return None
        normal = np.asarray(normal, dtype=np.float32)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            return None
        self._static_support_normal_space = 'pt3d'
        return normal / norm

    def _get_static_support_point(self):
        point = self.config.get('static_support_plane_point_gs', None)
        if point is not None:
            point = np.asarray(point, dtype=np.float32)
            if point.shape == (3,) and np.isfinite(point).all():
                self._static_support_point_space = 'gs'
                return point

        point = self.config.get('static_support_plane_point_pt3d', None)
        if point is None:
            return None
        point = np.asarray(point, dtype=np.float32)
        if point.shape != (3,) or not np.isfinite(point).all():
            return None
        self._static_support_point_space = 'pt3d'
        return point

    def simulate_step(self, sid, output_folder, final_sid=None):

        if self.config.get('debug', False):
            self.cam.start_recording()
        self.case_handler.custom_simulation(sid)
        self.scene.step()
        if self.config.get('debug', False):
            render_out = self.cam.render()
        updated_all_obj_points = []

        for obj_idx, per_material_type in enumerate(self.material_type):

            if per_material_type == 'rigid':
                obj_inertial_pos = self.objs[obj_idx].get_pos().cpu().numpy()
                obj_inertial_quat = self.objs[obj_idx].get_quat().cpu().numpy()
                transform_matrix = torch.from_numpy(pose_to_transform_matrix(obj_inertial_pos, obj_inertial_quat)).to(self.device).float()
                # Inverse the initial transform matrix
                initial_transform_matrix_inv = torch.linalg.inv(self.initial_transform_matrix[obj_idx])
                real_transform_matrix = transform_matrix @ initial_transform_matrix_inv
                points_homo = torch.cat([self.fg_pcs[obj_idx]['points'], torch.ones(self.fg_pcs[obj_idx]['points'].shape[0], 1).to(self.device)], dim=1)
                updated_points = torch.matmul(real_transform_matrix.unsqueeze(0), points_homo.unsqueeze(-1)).squeeze(-1)[:, :3]
                updated_points = self._world_gs_to_pt3d(updated_points)
                updated_all_obj_points.append(updated_points)

                # self.objs[obj_idx].solver.update_vgeoms_render_T() # trigger update
                # rigid_T = self.objs[obj_idx].solver._vgeoms_render_T
                # rigid_idx = self.objs[obj_idx].idx
                # transform_matrix = torch.tensor(rigid_T[rigid_idx, 0]).to(self.device).float() # 0 for env index
                # real_transform_matrix = transform_matrix @ torch.linalg.inv(self.initial_transform_matrix[obj_idx])
                # points_homo = torch.cat([self.fg_pcs[obj_idx]['points'], torch.ones(self.fg_pcs[obj_idx]['points'].shape[0], 1).to(self.device)], dim=1)
                # updated_points = torch.matmul(real_transform_matrix.unsqueeze(0), points_homo.unsqueeze(-1)).squeeze(-1)[:, :3]
                # updated_points = gs_to_pt3d(updated_points)
                # updated_all_obj_points.append(updated_points)

            elif per_material_type in ['pbd_liquid', 'pbd_cloth', 'mpm_sand', 'mpm_liquid', 'mpm_elastic', 'mpm_snow', 'mpm_elastic2plastic', 'pbd_elastic', 'pbd_particle']:
                particles_now_pos_in_gs = self.objs[obj_idx].solver.particles.pos.to_numpy()
                if len(particles_now_pos_in_gs.shape) == 4:
                    particles_now_pos_in_gs = particles_now_pos_in_gs[0, self.objs[obj_idx].particle_start:self.objs[obj_idx].particle_end, 0]
                else:
                    particles_now_pos_in_gs = particles_now_pos_in_gs[self.objs[obj_idx].particle_start:self.objs[obj_idx].particle_end, 0]
                
                particles_start_pos_in_gs = self.objs[obj_idx].init_particles

                particles_now_pos_in_gs = torch.tensor(particles_now_pos_in_gs).to(self.device)
                particles_start_pos_in_gs = torch.tensor(particles_start_pos_in_gs).to(self.device)

                particles_change_pos_in_gs = particles_now_pos_in_gs - particles_start_pos_in_gs
                points_change_pos_in_gs = particles_change_pos_in_gs[self.closest_indices[obj_idx]]
                points_change_pos_in_gs = points_change_pos_in_gs.mean(dim=1)
                updated_points = self.fg_pcs[obj_idx]['points'] + points_change_pos_in_gs
                updated_points = self._world_gs_to_pt3d(updated_points)
                updated_all_obj_points.append(updated_points)
        
        self.case_handler.after_simulation_step(self.svr)

        # if "robot_arm" in self.config['example_name']:
        #     franka_verts, franka_faces, franka_vertex_colors = self.extract_franka_mesh_data_combined(self.case_handler.current_target_franka)
        #     franka_verts = gs_to_pt3d(franka_verts)
        #     self.svr.franka_mesh = {
        #         'vertices': franka_verts,
        #         'faces': franka_faces,
        #         'colors': franka_vertex_colors
        #     }

        if self.config.get('debug', False):
            cv2.imwrite((output_folder / "gs_frames" / f"{sid:04d}.png").as_posix(), render_out[0])

        final_sid = self.simulation_steps - 1 if final_sid is None else final_sid
        if self.config.get('debug', False) and sid == final_sid:
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                gc.collect()
                time.sleep(float(self.config.get("debug_video_save_pause_seconds", 0.5)))
                self.cam.stop_recording(save_to_filename=(output_folder / "render_gs.mp4").as_posix(), fps=10)
            except Exception as exc:
                print(f"Warning: failed to save Genesis debug video render_gs.mp4: {exc}")
                if hasattr(self.cam, "_recorded_imgs"):
                    self.cam._recorded_imgs.clear()
                if hasattr(self.cam, "_in_recording"):
                    self.cam._in_recording = False
            # self.cam.stop_recording()
        
        return updated_all_obj_points

    def simulation_pc_render(self, max_render_frames=None):
        self.simulated_frames = []
        self.simualted_masks = []
        self.simualted_mesh_masks = []
        import time
        start_time = time.time()
        if max_render_frames is None:
            total_steps = self.simulation_steps
        else:
            total_steps = min(self.simulation_steps, max(1, int(max_render_frames)) * self.frame_steps)
        final_sid = total_steps - 1
        for sid in tqdm(range(total_steps)):
            # self.simulate_step(sid, self.output_folder, self.frame_steps)
            all_obj_points = self.simulate_step(sid, self.output_folder, final_sid=final_sid)
            if sid % self.frame_steps == 0:
                self.svr.update_fg_obj_info(all_obj_points)
                frame_id = sid // self.frame_steps
                current_frame, current_points_mask, current_mesh_mask = self.svr.render(frame_id = frame_id, save = self.config.get('debug', False), mask = True)
                self.simulated_frames.append(current_frame)
                self.simualted_masks.append(current_points_mask)
                self.simualted_mesh_masks.append(current_mesh_mask)
        end_time = time.time()
        print(f"Simulation + rendering time: {end_time - start_time} seconds")
        # save the gif of the simualated frames
        if self.config.get('debug', False):
            save_gif_from_image_folder(self.output_folder / "render" / "frames", self.output_folder / "simulated_frames.gif")
            save_gif_from_image_folder(self.output_folder / "gs_frames", self.output_folder / "simulated_frames_gs.gif")
            save_video_from_pil(self.simulated_frames, self.output_folder / "simulated_frames.mp4", fps=10)
            save_gif_from_image_folder(self.output_folder / "render" / "flow_image", self.output_folder / "flow_image.gif")
        return self.simulated_frames, self.simualted_masks, self.simualted_mesh_masks

    def map_pc_to_particles(self, obj_idx):
        sim_particles = torch.tensor(self.objs[obj_idx].init_particles).to(self.device)
        print(f"number of sim_particles: {sim_particles.shape[0]}")
        K = 256
        num_closest = 5 if 'closest_points_num' not in self.config else self.config['closest_points_num']
        point_chunks = torch.split(self.fg_pcs[obj_idx]['points'], K)
        closest_indices = []

        for chunk in tqdm(point_chunks):
            # Calculate pairwise distances between chunk and all particles
            # Using broadcasting to avoid memory issues
            # Shape: [K, 1, 3] - [1, N, 3] -> [K, N, 3] -> [K, N]
            distances = torch.norm(
                chunk.unsqueeze(1) - sim_particles.unsqueeze(0),
                dim=2
            )
            # Get top num_closest indices of closest particles for this chunk
            chunk_closest = torch.topk(distances, k=num_closest, dim=1, largest=False)[1]
            del distances
            closest_indices.append(chunk_closest)

        closest_indices = torch.cat(closest_indices)
        return closest_indices

    def get_material_for_each(self, per_material_type, obj_idx=None):
        if per_material_type == "rigid":
            rigid_rho = self.config.get('rigid_rho', 1000.0)
            if isinstance(rigid_rho, (list, tuple, ListConfig)):
                if obj_idx is None:
                    raise ValueError("obj_idx is required when rigid_rho is configured per object")
                if obj_idx >= len(rigid_rho):
                    raise ValueError(
                        f"rigid_rho has {len(rigid_rho)} values, but object index {obj_idx} was requested"
                    )
                rigid_rho = rigid_rho[obj_idx]

            obj_material = gs.materials.Rigid(
                rho = rigid_rho,
                friction = 5.0 if 'rigid_friction' not in self.config else self.config['rigid_friction'],
                coup_friction = 5 if 'rigid_coup_friction' not in self.config else self.config['rigid_coup_friction'],
                coup_softness = 0.002 if 'rigid_coup_softness' not in self.config else self.config['rigid_coup_softness'],
            )
            obj_vis_mode = "visual"
        elif per_material_type == 'pbd_liquid':
            obj_material = gs.materials.PBD.Liquid(
                rho = 1000.0 if 'pbd_rho' not in self.config else self.config['pbd_rho'],
                density_relaxation = 0.2 if 'pbd_density_relaxation' not in self.config else self.config['pbd_density_relaxation'],
                viscosity_relaxation = 0.1 if 'pbd_viscosity_relaxation' not in self.config else self.config['pbd_viscosity_relaxation'],
            )
            obj_vis_mode = "particle"

        elif per_material_type == "pbd_cloth":
            obj_material = gs.materials.PBD.Cloth(
                rho=4.0 if 'pbd_rho' not in self.config else self.config['pbd_rho'],
                static_friction=0.6 if 'pbd_static_friction' not in self.config else self.config['pbd_static_friction'],
                kinetic_friction=0.35 if 'pbd_kinetic_friction' not in self.config else self.config['pbd_kinetic_friction'],
                stretch_compliance=1e-7 if 'pbd_stretch_compliance' not in self.config else self.config['pbd_stretch_compliance'],
                bending_compliance=1e-5 if 'pbd_bending_compliance' not in self.config else self.config['pbd_bending_compliance'],
                stretch_relaxation=0.7 if 'pbd_stretch_relaxation' not in self.config else self.config['pbd_stretch_relaxation'],
                bending_relaxation=0.1 if 'pbd_bending_relaxation' not in self.config else self.config['pbd_bending_relaxation'],
                air_resistance=5e-3 if 'pbd_air_resistance' not in self.config else self.config['pbd_air_resistance'],

            )
            obj_vis_mode = "particle"
        elif per_material_type == "pbd_elastic":
            obj_material = gs.materials.PBD.Elastic(
                rho=300.0 if 'pbd_elastic_rho' not in self.config else self.config['pbd_elastic_rho'],
                static_friction=0.15 if 'pbd_elastic_static_friction' not in self.config else self.config['pbd_elastic_static_friction'],
                kinetic_friction=0.0 if 'pbd_elastic_kinetic_friction' not in self.config else self.config['pbd_elastic_kinetic_friction'],
                stretch_compliance=0.0 if 'pbd_elastic_stretch_compliance' not in self.config else self.config['pbd_elastic_stretch_compliance'],
                bending_compliance=0.0 if 'pbd_elastic_bending_compliance' not in self.config else self.config['pbd_elastic_bending_compliance'],
                volume_compliance=0.0 if 'pbd_elastic_volume_compliance' not in self.config else self.config['pbd_elastic_volume_compliance'],
                stretch_relaxation=0.1 if 'pbd_elastic_stretch_relaxation' not in self.config else self.config['pbd_elastic_stretch_relaxation'],
                bending_relaxation=0.1 if 'pbd_elastic_bending_relaxation' not in self.config else self.config['pbd_elastic_bending_relaxation'],
                volume_relaxation=0.1 if 'pbd_elastic_volume_relaxation' not in self.config else self.config['pbd_elastic_volume_relaxation'],
            )
            obj_vis_mode = "particle"
        elif per_material_type == "pbd_particle":
            obj_material = gs.materials.PBD.Particle()
            obj_vis_mode = "particle"
        elif per_material_type == "mpm_sand":
            obj_material = gs.materials.MPM.Sand(
                E = 1e6 if 'MPM_E' not in self.config else self.config['MPM_E'],
                nu = 0.2 if 'MPM_nu' not in self.config else self.config['MPM_nu'],
                rho = 1000.0 if 'MPM_rho' not in self.config else self.config['MPM_rho'],
                friction_angle = 45 if 'MPM_friction_angle' not in self.config else self.config['MPM_friction_angle'],
            )
            obj_vis_mode = "particle"
        elif per_material_type == "mpm_elastic":
            obj_material = gs.materials.MPM.Elastic(
                E = 1e6 if 'MPM_E' not in self.config else self.config['MPM_E'],
                nu = 0.2 if 'MPM_nu' not in self.config else self.config['MPM_nu'],
                rho = 1000.0 if 'MPM_rho' not in self.config else self.config['MPM_rho'],
            )
            obj_vis_mode = "particle"
        elif per_material_type == "mpm_liquid":
            obj_material = gs.materials.MPM.Liquid(
                E = 1e6 if 'MPM_E' not in self.config else self.config['MPM_E'],
                nu = 0.2 if 'MPM_nu' not in self.config else self.config['MPM_nu'],
                rho = 1000.0 if 'MPM_rho' not in self.config else self.config['MPM_rho'],
            )
            obj_vis_mode = "particle"
        elif per_material_type == "mpm_snow":
            obj_material = gs.materials.MPM.Snow(
                E = 1e6 if 'MPM_E' not in self.config else self.config['MPM_E'],
                nu = 0.2 if 'MPM_nu' not in self.config else self.config['MPM_nu'],
                rho = 1000.0 if 'MPM_rho' not in self.config else self.config['MPM_rho'],
            )
            obj_vis_mode = "particle"
        elif per_material_type == "mpm_elastic2plastic":
            obj_material = gs.materials.MPM.ElastoPlastic(
                E = 1e6 if 'MPM_E' not in self.config else self.config['MPM_E'],
                nu = 0.2 if 'MPM_nu' not in self.config else self.config['MPM_nu'],
                rho = 1000.0 if 'MPM_rho' not in self.config else self.config['MPM_rho'],
            )
            obj_vis_mode = "particle"
        else:
            raise NotImplementedError(f"The current material {per_material_type} is not supported for now")
        return obj_material, obj_vis_mode

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
            verts = vgeom.vmesh.verts
            faces = vgeom.vmesh.faces
            
            # Get transformation matrix for this vgeom
            cur_render_T = sim_vgeoms_render_T[vgeom.idx][0]
            
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
        
        vertices = torch.from_numpy(np.vstack(all_vertices)).to(self.device, dtype=torch.float32)
        faces = torch.from_numpy(np.vstack(all_faces)).to(self.device, dtype=torch.int32)
        colors = torch.from_numpy(np.vstack(all_colors)).to(self.device, dtype=torch.float32)
        
        return vertices, faces, colors
