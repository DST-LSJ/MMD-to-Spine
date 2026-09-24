"""3D MMD to 2D Spine projection helpers used by the root executable."""

from __future__ import annotations

from dataclasses import dataclass
import math


Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class ProjectionConfig:
    camera_yaw_deg: float = 30.0
    camera_pitch_deg: float = 0.0
    scale: float = 20.0
    depth_factor: float = 1.0


@dataclass(frozen=True, slots=True)
class ProjectedPoint:
    world: Vector3
    camera: Vector3
    screen: tuple[float, float]
    camera_depth: float


class Projector:
    def __init__(self, config: ProjectionConfig):
        self.config = config
        self.yaw = math.radians(config.camera_yaw_deg)
        self.pitch = math.radians(config.camera_pitch_deg)

    def camera_space(self, vector: Vector3) -> Vector3:
        x, y, z = vector
        yaw_x = math.cos(self.yaw) * x + math.sin(self.yaw) * z
        yaw_z = -math.sin(self.yaw) * x + math.cos(self.yaw) * z
        pitch_y = math.cos(self.pitch) * y - math.sin(self.pitch) * yaw_z
        pitch_z = math.sin(self.pitch) * y + math.cos(self.pitch) * yaw_z
        return yaw_x, pitch_y, pitch_z

    def position(self, vector: Vector3) -> tuple[float, float]:
        return self.project_point(vector).screen

    def project_point(self, vector: Vector3) -> ProjectedPoint:
        """Project a point while retaining camera-space depth.

        ``camera_depth`` is camera-space Z. Under this project's camera
        convention larger values are farther from the observer, so Spine draw
        order is emitted from larger depth (back) to smaller depth (front).
        ``depth_factor`` only stylizes screen XY; it never changes this depth.
        """

        x, y, z = self.camera_space(vector)
        # The camera rotations already mix depth into x/y. depth_factor lets a
        # user attenuate that effect while keeping a normal orthographic view at 1.
        if self.config.depth_factor != 1.0:
            direct_x = vector[0]
            direct_y = vector[1]
            x = direct_x + (x - direct_x) * self.config.depth_factor
            y = direct_y + (y - direct_y) * self.config.depth_factor
        screen = (x * self.config.scale, y * self.config.scale)
        return ProjectedPoint(vector, (x, y, z), screen, z)

    def direction(self, vector: Vector3) -> tuple[float, float]:
        x, y, _ = self.camera_space(vector)
        if self.config.depth_factor != 1.0:
            x = vector[0] + (x - vector[0]) * self.config.depth_factor
            y = vector[1] + (y - vector[1]) * self.config.depth_factor
        return x, y

    def rotation_delta(self, quaternion: Quaternion, rest_axis: Vector3) -> float:
        rest_2d = self.direction(rest_axis)
        animated_2d = self.direction(rotate_vector(quaternion, rest_axis))
        rest_length = math.hypot(*rest_2d)
        animated_length = math.hypot(*animated_2d)
        if rest_length < 1e-7 or animated_length < 1e-4:
            return quaternion_z_angle(quaternion)
        rest_angle = math.atan2(rest_2d[1], rest_2d[0])
        animated_angle = math.atan2(animated_2d[1], animated_2d[0])
        return normalise_degrees(math.degrees(animated_angle - rest_angle))


def rotate_vector(quaternion: Quaternion, vector: Vector3) -> Vector3:
    """Rotate *vector* by an x/y/z/w quaternion without external libraries."""

    qx, qy, qz, qw = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    )


def quaternion_z_angle(quaternion: Quaternion) -> float:
    x, y, z, w = quaternion
    sin_value = 2.0 * (w * z + x * y)
    cos_value = 1.0 - 2.0 * (y * y + z * z)
    return normalise_degrees(math.degrees(math.atan2(sin_value, cos_value)))


def normalise_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


REST_AXES: dict[str, Vector3] = {
    "up": (0.0, 1.0, 0.0),
    "down": (0.0, -1.0, 0.0),
    "left": (-1.0, 0.0, 0.0),
    "right": (1.0, 0.0, 0.0),
}
