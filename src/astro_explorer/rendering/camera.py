"""Perspective camera with explicit matrices (roadmap section 4.8).

No ``glMatrixMode``, ``glTranslatef`` or ``glRotatef``: the matrices are
built here in NumPy and uploaded as uniforms.  That keeps the camera
testable without a GL context and makes the near/far policy explicit, which
matters when the same scene spans parsecs and kilometres.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["Camera", "look_at", "perspective", "orthographic"]


def look_at(eye, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Right-handed view matrix, row-major, ready for ``bytes()`` upload."""
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up = np.asarray(up, dtype=np.float64).reshape(3)

    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-12:
        forward = np.array([0.0, 0.0, -1.0])
    else:
        forward = forward / norm

    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-12:
        # Looking straight along `up`: pick any perpendicular axis.
        right = np.array([1.0, 0.0, 0.0])
    else:
        right = right / right_norm
    true_up = np.cross(right, forward)

    matrix = np.eye(4, dtype=np.float64)
    matrix[0, :3] = right
    matrix[1, :3] = true_up
    matrix[2, :3] = -forward
    matrix[0, 3] = -np.dot(right, eye)
    matrix[1, 3] = -np.dot(true_up, eye)
    matrix[2, 3] = np.dot(forward, eye)
    return matrix


def perspective(fov_y_rad: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Standard OpenGL perspective projection, mapping z to [-1, 1]."""
    if near <= 0.0 or far <= near:
        raise ValueError("require 0 < near < far")
    if aspect <= 0.0:
        raise ValueError("aspect must be positive")

    focal = 1.0 / np.tan(0.5 * fov_y_rad)
    matrix = np.zeros((4, 4), dtype=np.float64)
    matrix[0, 0] = focal / aspect
    matrix[1, 1] = focal
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2.0 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    return matrix


def orthographic(half_width: float, half_height: float, near: float, far: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 0] = 1.0 / half_width
    matrix[1, 1] = 1.0 / half_height
    matrix[2, 2] = -2.0 / (far - near)
    matrix[2, 3] = -(far + near) / (far - near)
    return matrix


@dataclass
class Camera:
    """An orbit camera expressed in the active frame's display units.

    ``distance``, ``near`` and ``far`` are all in the units of whichever
    :class:`~astro_explorer.coordinates.floating_origin.Scale` is active, so
    the camera never has to know whether it is looking at a galaxy or a
    planet.
    """

    target: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    distance: float = 5.0
    yaw: float = 0.0
    pitch: float = 0.35
    fov_y_rad: float = np.radians(45.0)
    aspect: float = 16.0 / 9.0
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)

    #: Near/far are derived from `distance` so precision stays usable across
    #: fifteen orders of magnitude of scene scale.
    near_ratio: float = 1e-3
    far_ratio: float = 1e4

    min_distance: float = 1e-6
    max_distance: float = 1e9
    max_pitch: float = np.radians(89.0)

    def __post_init__(self) -> None:
        self.target = np.asarray(self.target, dtype=np.float64).reshape(3)

    # -- state -----------------------------------------------------------
    @property
    def position(self) -> np.ndarray:
        """Eye position derived from the orbit parameters."""
        cos_pitch = np.cos(self.pitch)
        offset = np.array(
            [
                self.distance * cos_pitch * np.sin(self.yaw),
                self.distance * np.sin(self.pitch),
                self.distance * cos_pitch * np.cos(self.yaw),
            ]
        )
        return self.target + offset

    @property
    def near(self) -> float:
        return max(self.distance * self.near_ratio, 1e-9)

    @property
    def far(self) -> float:
        return self.distance * self.far_ratio

    # -- matrices --------------------------------------------------------
    def view_matrix(self) -> np.ndarray:
        return look_at(self.position, self.target, self.up)

    def projection_matrix(self) -> np.ndarray:
        return perspective(self.fov_y_rad, self.aspect, self.near, self.far)

    def view_projection(self) -> np.ndarray:
        return self.projection_matrix() @ self.view_matrix()

    def uniform_bytes(self) -> bytes:
        """Column-major float32 view-projection, as GLSL expects."""
        return np.ascontiguousarray(
            self.view_projection().T.astype(np.float32)
        ).tobytes()

    # -- interaction -----------------------------------------------------
    def orbit(self, delta_yaw: float, delta_pitch: float) -> None:
        self.yaw = float(np.mod(self.yaw + delta_yaw, 2.0 * np.pi))
        self.pitch = float(np.clip(self.pitch + delta_pitch, -self.max_pitch, self.max_pitch))

    def dolly(self, factor: float) -> None:
        """Multiplicative zoom, so the step feels equal at every scale."""
        self.distance = float(np.clip(self.distance * factor, self.min_distance, self.max_distance))

    def pan(self, right_amount: float, up_amount: float) -> None:
        view = self.view_matrix()
        right = view[0, :3]
        up = view[1, :3]
        self.target = self.target + right * right_amount + up * up_amount

    def frame_object(self, radius: float, *, margin: float = 2.5) -> None:
        """Set the distance so an object of ``radius`` fills the view."""
        radius = max(float(radius), self.min_distance)
        self.distance = float(
            np.clip(margin * radius / np.tan(0.5 * self.fov_y_rad), self.min_distance, self.max_distance)
        )

    def ray_through_pixel(self, x: float, y: float, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        """World-space ray for a pixel, used by picking.

        Returns ``(origin, direction)`` with a unit direction.
        """
        ndc_x = (2.0 * x / max(width, 1)) - 1.0
        ndc_y = 1.0 - (2.0 * y / max(height, 1))

        tan_half = np.tan(0.5 * self.fov_y_rad)
        view = self.view_matrix()
        right, up, backward = view[0, :3], view[1, :3], view[2, :3]
        forward = -backward

        direction = forward + right * (ndc_x * tan_half * self.aspect) + up * (ndc_y * tan_half)
        direction = direction / np.linalg.norm(direction)
        return self.position, direction
