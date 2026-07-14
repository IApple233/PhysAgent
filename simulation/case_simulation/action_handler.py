import re
from typing import Any

import numpy as np

from simulation.case_simulation.case_handler import CaseHandler
from simulation.case_simulation.physics_actions import BaseAction, DirectionTo


class ActionCaseHandler(CaseHandler):
    """CaseHandler subclass that executes a high-level physics action schedule."""

    object_names = None

    def __init__(self, config, all_obj_info, device):
        super().__init__(config, all_obj_info, device)
        self._compiled_actions = None
        self._initial_state_by_index = {}

    def build_actions(self):
        """Return a list of physics action objects."""
        return []

    def after_scene_building(self):
        super().after_scene_building()
        self._cache_initial_states()

    def custom_simulation(self, sid):
        for action in self.actions():
            action.apply(self, sid)

    def actions(self):
        if self._compiled_actions is None:
            actions = self.build_actions()
            if actions is None:
                actions = []
            if not isinstance(actions, (list, tuple)):
                raise TypeError("build_actions() must return a list or tuple of physics actions.")
            for action in actions:
                if not isinstance(action, BaseAction):
                    raise TypeError(f"Invalid physics action: {action!r}")
            self._compiled_actions = list(actions)
        return self._compiled_actions

    def names(self):
        names = self.object_names
        if names is None:
            names = (
                self.config.get("object_names")
                or self.config.get("all_object_names")
                or self.config.get("dynamic_object_names")
                or []
            )
        names = list(names or [])
        if len(names) < len(self.all_obj_info):
            names.extend([f"object_{idx}" for idx in range(len(names), len(self.all_obj_info))])
        return [self._normalize_name(name) for name in names[: len(self.all_obj_info)]]

    def object_index(self, name_or_index: Any) -> int:
        if isinstance(name_or_index, int):
            idx = name_or_index
        else:
            raw = str(name_or_index)
            match = re.fullmatch(r"(?:obj(?:ect)?[_ -]?)?(\d+)", raw.strip().lower())
            if match:
                idx = int(match.group(1))
            else:
                normalized = self._normalize_name(raw)
                names = self.names()
                if normalized not in names:
                    raise KeyError(f"Unknown object '{name_or_index}'. Available objects: {names}")
                idx = names.index(normalized)
        if idx < 0 or idx >= len(self.all_objs):
            raise IndexError(f"Object index out of range: {idx}")
        return idx

    def obj(self, name_or_index: Any):
        return self.all_objs[self.object_index(name_or_index)]

    def position(self, name_or_index: Any) -> np.ndarray:
        return self.obj(name_or_index).get_pos().detach().cpu().numpy().astype(np.float32)

    def linear_velocity(self, name_or_index: Any) -> np.ndarray:
        return self._dof_velocity(name_or_index)[:3]

    def angular_velocity(self, name_or_index: Any) -> np.ndarray:
        return self._dof_velocity(name_or_index)[3:6]

    def direction_between(self, source: Any, target: Any, horizontal: bool = False) -> np.ndarray:
        direction = self.position(target) - self.position(source)
        if horizontal:
            direction = direction.copy()
            direction[2] = 0.0
        norm = float(np.linalg.norm(direction))
        if norm < 1e-8:
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        return (direction / norm).astype(np.float32)

    def direction_to(self, target: Any, magnitude: float = 1.0, source: Any = None, horizontal: bool = False):
        return DirectionTo(target=target, source=source, magnitude=magnitude, horizontal=horizontal)

    def point_on(self, name_or_index: Any, offset=(0.0, 0.0, 0.0)) -> np.ndarray:
        return self.position(name_or_index) + np.asarray(offset, dtype=np.float32)

    def set_linear_velocity(self, name_or_index: Any, velocity):
        qvel = self._dof_velocity(name_or_index)
        qvel[:3] = np.asarray(velocity, dtype=np.float32)
        self._set_dof_velocity(name_or_index, qvel)

    def set_angular_velocity(self, name_or_index: Any, angular_velocity):
        qvel = self._dof_velocity(name_or_index)
        qvel[3:6] = np.asarray(angular_velocity, dtype=np.float32)
        self._set_dof_velocity(name_or_index, qvel)

    def apply_force(self, name_or_index: Any, force, point=None):
        obj = self.obj(name_or_index)
        force = np.asarray(force, dtype=np.float32).reshape(1, 3)
        obj.solver.apply_links_external_force(force=force, links_idx=[obj.idx])
        if point is not None:
            center = self.position(name_or_index)
            torque = np.cross(np.asarray(point, dtype=np.float32) - center, force.reshape(3))
            self.apply_torque(name_or_index, torque)

    def apply_torque(self, name_or_index: Any, torque):
        obj = self.obj(name_or_index)
        torque = np.asarray(torque, dtype=np.float32).reshape(1, 3)
        obj.solver.apply_links_external_torque(torque=torque, links_idx=[obj.idx])

    def set_position(self, name_or_index: Any, position):
        obj = self.obj(name_or_index)
        position = np.asarray(position, dtype=np.float32)
        if hasattr(obj, "set_pos"):
            obj.set_pos(position, zero_velocity=True)
        elif hasattr(obj, "set_position"):
            obj.set_position(position)
            self._set_dof_velocity(name_or_index, np.zeros(6, dtype=np.float32))
        else:
            raise AttributeError(f"Object {name_or_index!r} does not support set_pos/set_position.")

    def set_orientation(self, name_or_index: Any, rotation):
        obj = self.obj(name_or_index)
        quat = self._rotation_to_quat(rotation)
        if hasattr(obj, "set_quat"):
            obj.set_quat(quat)
            return
        if hasattr(obj, "set_qpos"):
            pos = self.position(name_or_index)
            obj.set_qpos(np.concatenate([pos, quat]).astype(np.float32))
            return
        raise AttributeError(f"Object {name_or_index!r} does not support set_quat/set_qpos.")

    def fix_object(self, name_or_index: Any):
        idx = self.object_index(name_or_index)
        state = self._initial_state_by_index.get(idx)
        if state is not None:
            self.set_position(idx, state["pos"])
        self._set_dof_velocity(idx, np.zeros(6, dtype=np.float32))

    def _dof_velocity(self, name_or_index: Any) -> np.ndarray:
        obj = self.obj(name_or_index)
        try:
            vel = obj.get_dofs_velocity(dofs_idx_local=np.arange(6)).detach().cpu().numpy()
            vel = np.asarray(vel, dtype=np.float32).reshape(-1)
            if vel.shape[0] >= 6:
                return vel[:6].copy()
        except Exception:
            pass
        return np.zeros(6, dtype=np.float32)

    def _set_dof_velocity(self, name_or_index: Any, qvel):
        obj = self.obj(name_or_index)
        qvel = np.asarray(qvel, dtype=np.float32).reshape(6)
        obj.set_dofs_velocity(velocity=qvel, dofs_idx_local=np.arange(6))

    def _cache_initial_states(self):
        self._initial_state_by_index = {}
        for idx, obj in enumerate(getattr(self, "all_objs", []) or []):
            state = {}
            try:
                state["pos"] = obj.get_pos().detach().cpu().numpy().astype(np.float32)
            except Exception:
                pass
            try:
                state["quat"] = obj.get_quat().detach().cpu().numpy().astype(np.float32)
            except Exception:
                pass
            self._initial_state_by_index[idx] = state

    @staticmethod
    def _normalize_name(name: Any) -> str:
        name = str(name).strip().lower()
        name = re.sub(r"[^a-z0-9]+", "_", name)
        return name.strip("_") or "object"

    @staticmethod
    def _rotation_to_quat(rotation):
        arr = np.asarray(rotation, dtype=np.float32).reshape(-1)
        if arr.shape[0] == 4:
            return arr
        if arr.shape[0] != 3:
            raise ValueError("rotation must be a quaternion [w, x, y, z] or Euler angles [rx, ry, rz] in degrees.")
        rx, ry, rz = np.deg2rad(arr.astype(np.float64))
        cx, sx = np.cos(rx / 2), np.sin(rx / 2)
        cy, sy = np.cos(ry / 2), np.sin(ry / 2)
        cz, sz = np.cos(rz / 2), np.sin(rz / 2)
        quat = np.asarray(
            [
                cx * cy * cz + sx * sy * sz,
                sx * cy * cz - cx * sy * sz,
                cx * sy * cz + sx * cy * sz,
                cx * cy * sz - sx * sy * cz,
            ],
            dtype=np.float32,
        )
        return quat / max(float(np.linalg.norm(quat)), 1e-8)
