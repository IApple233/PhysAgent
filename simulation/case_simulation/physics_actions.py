from dataclasses import dataclass
import math
from typing import Any, Optional

import numpy as np


def _frame_value(frame: int) -> int:
    return max(0, int(frame))


def _duration_value(duration: int) -> int:
    return max(1, int(duration))


def _as_vec3(value: Any, handler=None, action=None, default=None) -> np.ndarray:
    if hasattr(value, "resolve"):
        value = value.resolve(handler, action)
    elif callable(value):
        value = value(handler)
    if value is None:
        if default is None:
            raise ValueError("Expected a 3D vector, got None.")
        value = default
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (3,):
        raise ValueError(f"Expected a 3D vector with shape (3,), got {arr.shape}: {value}")
    if not np.isfinite(arr).all():
        raise ValueError(f"Vector contains non-finite values: {value}")
    return arr


def _normalize(vec: np.ndarray, fallback=(1.0, 0.0, 0.0)) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return np.asarray(fallback, dtype=np.float32)
    return (vec / norm).astype(np.float32)


class DirectionTo:
    """Dynamic vector helper for force/velocity arguments."""

    def __init__(self, target: Any, magnitude: float = 1.0, source: Optional[Any] = None, horizontal: bool = False):
        self.target = target
        self.magnitude = float(magnitude)
        self.source = source
        self.horizontal = bool(horizontal)

    def resolve(self, handler, action):
        source = action.name if self.source is None else self.source
        direction = handler.direction_between(source, self.target, horizontal=self.horizontal)
        return direction * self.magnitude


class CurrentVelocityDirection:
    """Dynamic vector helper that follows an object's current velocity direction."""

    def __init__(self, magnitude: float = 1.0, angle: float = 0.0, horizontal: bool = True):
        self.magnitude = float(magnitude)
        self.angle = float(angle)
        self.horizontal = bool(horizontal)

    def resolve(self, handler, action):
        velocity = handler.linear_velocity(action.name)
        if self.horizontal:
            velocity = velocity.copy()
            velocity[2] = 0.0
        direction = _normalize(velocity)
        direction = rotate_about_z(direction, self.angle)
        return direction * self.magnitude


def rotate_about_z(vector: np.ndarray, angle_degrees: float) -> np.ndarray:
    radians = math.radians(float(angle_degrees))
    cos_v = math.cos(radians)
    sin_v = math.sin(radians)
    x, y, z = vector
    return np.asarray([cos_v * x - sin_v * y, sin_v * x + cos_v * y, z], dtype=np.float32)


@dataclass
class BaseAction:
    name: Any
    frame: int = 0
    duration: int = 1

    def active(self, sid: int) -> bool:
        start = _frame_value(self.frame)
        return start <= int(sid) < start + _duration_value(self.duration)

    def once(self, sid: int) -> bool:
        return int(sid) == _frame_value(self.frame)

    def apply(self, handler, sid: int):
        raise NotImplementedError


@dataclass
class SetVelocity(BaseAction):
    velocity: Any = None

    def __init__(self, name, velocity, frame=0):
        super().__init__(name=name, frame=frame, duration=1)
        self.velocity = velocity

    def apply(self, handler, sid: int):
        if not self.once(sid):
            return
        handler.set_linear_velocity(self.name, _as_vec3(self.velocity, handler, self))


@dataclass
class SetAngularVelocity(BaseAction):
    angular_velocity: Any = None

    def __init__(self, name, angular_velocity, frame=0):
        super().__init__(name=name, frame=frame, duration=1)
        self.angular_velocity = angular_velocity

    def apply(self, handler, sid: int):
        if not self.once(sid):
            return
        handler.set_angular_velocity(self.name, _as_vec3(self.angular_velocity, handler, self))


@dataclass
class ApplyForce(BaseAction):
    force: Any = None
    point: Any = None

    def __init__(self, name, force, point=None, duration=1, frame=0):
        super().__init__(name=name, frame=frame, duration=duration)
        self.force = force
        self.point = point

    def apply(self, handler, sid: int):
        if not self.active(sid):
            return
        force = _as_vec3(self.force, handler, self)
        point = None if self.point is None else _as_vec3(self.point, handler, self)
        handler.apply_force(self.name, force, point=point)


@dataclass
class ApplyTorque(BaseAction):
    torque: Any = None

    def __init__(self, name, torque, duration=1, frame=0):
        super().__init__(name=name, frame=frame, duration=duration)
        self.torque = torque

    def apply(self, handler, sid: int):
        if not self.active(sid):
            return
        handler.apply_torque(self.name, _as_vec3(self.torque, handler, self))


@dataclass
class ApplyAngledForce(BaseAction):
    magnitude: float = 1.0
    angle: float = 0.0
    horizontal: bool = True

    def __init__(self, name, magnitude, angle, duration=1, frame=0, horizontal=True):
        super().__init__(name=name, frame=frame, duration=duration)
        self.magnitude = float(magnitude)
        self.angle = float(angle)
        self.horizontal = bool(horizontal)

    def apply(self, handler, sid: int):
        if not self.active(sid):
            return
        force = CurrentVelocityDirection(
            magnitude=self.magnitude,
            angle=self.angle,
            horizontal=self.horizontal,
        ).resolve(handler, self)
        handler.apply_force(self.name, force)


@dataclass
class ApplyDisturbance(BaseAction):
    model: str = "jitter"
    amplitude: float = 1.0

    def __init__(self, name, model="jitter", amplitude=1.0, duration=1, frame=0):
        super().__init__(name=name, frame=frame, duration=duration)
        self.model = str(model)
        self.amplitude = float(amplitude)

    def apply(self, handler, sid: int):
        if not self.active(sid):
            return
        model = self.model.lower()
        phase = float(int(sid) - _frame_value(self.frame))
        if model in {"up", "vertical", "bump"}:
            force = np.asarray([0.0, 0.0, self.amplitude], dtype=np.float32)
        elif model in {"side", "lateral"}:
            sign = -1.0 if int(phase) % 2 else 1.0
            force = np.asarray([sign * self.amplitude, 0.0, 0.0], dtype=np.float32)
        elif model in {"shake", "jitter", "random"}:
            force = np.asarray(
                [
                    math.sin(phase * 1.7),
                    math.cos(phase * 1.3),
                    0.25 * math.sin(phase * 0.9),
                ],
                dtype=np.float32,
            )
            force = _normalize(force) * self.amplitude
        else:
            raise ValueError(f"Unknown disturbance model: {self.model}")
        handler.apply_force(self.name, force)


@dataclass
class SetPosition(BaseAction):
    position: Any = None

    def __init__(self, name, position, frame=0):
        super().__init__(name=name, frame=frame, duration=1)
        self.position = position

    def apply(self, handler, sid: int):
        if not self.once(sid):
            return
        handler.set_position(self.name, _as_vec3(self.position, handler, self))


@dataclass
class SetOrientation(BaseAction):
    rotation: Any = None

    def __init__(self, name, rotation, frame=0):
        super().__init__(name=name, frame=frame, duration=1)
        self.rotation = rotation

    def apply(self, handler, sid: int):
        if not self.once(sid):
            return
        handler.set_orientation(self.name, self.rotation)


@dataclass
class FixObject(BaseAction):
    def __init__(self, name, duration, frame=0):
        super().__init__(name=name, frame=frame, duration=duration)

    def apply(self, handler, sid: int):
        if not self.active(sid):
            return
        handler.fix_object(self.name)
