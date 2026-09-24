"""Canonical 3D proxy pose for depth sorting when no PMX/PMD is available."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

from .projection import Projector, Quaternion, Vector3, rotate_vector
from .vmd_reader import VMDMotion, sample_track


IDENTITY: Quaternion = (0.0, 0.0, 0.0, 1.0)
ZERO: Vector3 = (0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class Segment3D:
    name: str
    start_joint: str
    end_joint: str
    start: Vector3
    end: Vector3
    confidence: float
    depth_source: str = "canonical_rig_approximation"


@dataclass(slots=True)
class PoseSample:
    source_frame: int
    output_frame: int
    time_seconds: float
    joints: dict[str, Vector3]
    segments: dict[str, Segment3D]
    warnings: list[str] = field(default_factory=list)


def load_rig_profile(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        profile = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read source rig profile {path}: {error}") from error
    if profile.get("depth_source") != "canonical_rig_approximation":
        raise ValueError("P0 source rig profile must declare canonical_rig_approximation")
    if float(profile.get("body_height", 0)) <= 0:
        raise ValueError("source rig profile body_height must be positive")
    for key in ("aliases", "geometry", "confidence"):
        if not isinstance(profile.get(key), dict):
            raise ValueError(f"source rig profile requires object {key!r}")
    return profile


class CanonicalPoseSolver:
    """Evaluate a simplified MMD humanoid with quaternion FK and 3D leg IK."""

    def __init__(self, motion: VMDMotion, profile: dict[str, Any], projector: Projector):
        self.motion = motion
        self.profile = profile
        self.projector = projector
        self.aliases: dict[str, list[str]] = profile["aliases"]
        self.parent_chains: dict[str, list[str]] = profile.get("parent_chains", {})
        self.geometry = {
            name: _vector(value) for name, value in profile["geometry"].items()
        }
        self.depth_source = str(profile["depth_source"])
        self.body_height = float(profile["body_height"])
        self.target_bindings: dict[str, str] = profile.get("target_bindings", {})

    def sample(self, source_frame: int, output_frame: int, fps: float) -> PoseSample:
        warnings: list[str] = []
        root_position = self._position("all_parent", source_frame)
        center = _add(
            self._position("center", source_frame),
            self._position("groove", source_frame),
        )
        root_rotation = self._rotation("all_parent", source_frame)
        torso_parent_rotation = self._rotation_chain(
            "torso_parent_world",
            source_frame,
            ("all_parent", "center", "groove", "waist"),
        )
        lower_rotation = self._rotation_chain(
            "lower_body_world",
            source_frame,
            ("all_parent", "center", "groove", "waist", "lower_body"),
        )
        upper_rotation = self._rotation_chain(
            "upper_body_world",
            source_frame,
            (
                "all_parent",
                "center",
                "groove",
                "waist",
                "upper_body",
                "upper_body_2",
            ),
        )

        pelvis_reference = self.geometry["pelvis_reference"]
        pelvis = _add(root_position, rotate_vector(root_rotation, _add(pelvis_reference, center)))
        # Upper and lower body are sibling branches in the configured MMD
        # reference hierarchy.  The chest origin follows their common parent
        # chain, not lower_body; otherwise a hip twist is applied to the arms
        # even when upper_body counter-rotates it.
        chest = _add(
            pelvis,
            rotate_vector(torso_parent_rotation, self.geometry["chest_offset"]),
        )
        joints: dict[str, Vector3] = {"pelvis": pelvis, "chest": chest}

        self._solve_arm(
            "F", self.target_bindings.get("arm_F", "right"),
            chest, upper_rotation, source_frame, joints,
        )
        self._solve_arm(
            "B", self.target_bindings.get("arm_B", "left"),
            chest, upper_rotation, source_frame, joints,
        )
        self._solve_leg(
            "F", self.target_bindings.get("leg_F", "right"),
            pelvis, pelvis_reference, root_position, root_rotation,
            lower_rotation, center, source_frame, joints, warnings
        )
        self._solve_leg(
            "B", self.target_bindings.get("leg_B", "left"),
            pelvis, pelvis_reference, root_position, root_rotation,
            lower_rotation, center, source_frame, joints, warnings
        )

        confidence = self.profile["confidence"]
        leg_confidence = float(
            confidence.get(
                "legs_ik" if self._has_track("left_foot_ik") else "legs_fk", 0.5
            )
        )
        segments = {
            "body": Segment3D("body", "pelvis", "chest", pelvis, chest, float(confidence.get("body", 0.7))),
            "whole_arm_F": Segment3D("whole_arm_F", "shoulder_F", "wrist_F", joints["shoulder_F"], joints["wrist_F"], float(confidence.get("arms", 0.65))),
            "upper_arm_F": Segment3D("upper_arm_F", "shoulder_F", "elbow_F", joints["shoulder_F"], joints["elbow_F"], float(confidence.get("arms", 0.65))),
            "forearm_hand_F": Segment3D("forearm_hand_F", "elbow_F", "wrist_F", joints["elbow_F"], joints["wrist_F"], float(confidence.get("arms", 0.65))),
            "whole_arm_B": Segment3D("whole_arm_B", "shoulder_B", "wrist_B", joints["shoulder_B"], joints["wrist_B"], float(confidence.get("arms", 0.65))),
            "upper_arm_B": Segment3D("upper_arm_B", "shoulder_B", "elbow_B", joints["shoulder_B"], joints["elbow_B"], float(confidence.get("arms", 0.65))),
            "forearm_hand_B": Segment3D("forearm_hand_B", "elbow_B", "wrist_B", joints["elbow_B"], joints["wrist_B"], float(confidence.get("arms", 0.65))),
            "whole_leg_F": Segment3D("whole_leg_F", "hip_F", "ankle_F", joints["hip_F"], joints["ankle_F"], leg_confidence),
            "upper_leg_F": Segment3D("upper_leg_F", "hip_F", "knee_F", joints["hip_F"], joints["knee_F"], leg_confidence),
            "lower_leg_F": Segment3D("lower_leg_F", "knee_F", "ankle_F", joints["knee_F"], joints["ankle_F"], leg_confidence),
            "whole_leg_B": Segment3D("whole_leg_B", "hip_B", "ankle_B", joints["hip_B"], joints["ankle_B"], leg_confidence),
            "upper_leg_B": Segment3D("upper_leg_B", "hip_B", "knee_B", joints["hip_B"], joints["knee_B"], leg_confidence),
            "lower_leg_B": Segment3D("lower_leg_B", "knee_B", "ankle_B", joints["knee_B"], joints["ankle_B"], leg_confidence),
        }
        return PoseSample(source_frame, output_frame, output_frame / fps, joints, segments, warnings)

    def _solve_arm(
        self,
        suffix: str,
        side: str,
        chest: Vector3,
        chest_rotation: Quaternion,
        frame: int,
        joints: dict[str, Vector3],
    ) -> None:
        shoulder = _add(
            chest,
            rotate_vector(chest_rotation, self.geometry[f"shoulder_{suffix}_offset"]),
        )
        upper_rotation = _qmul(chest_rotation, self._rotation(f"{side}_shoulder", frame))
        upper_rotation = _qmul(upper_rotation, self._rotation(f"{side}_arm", frame))
        elbow = _add(
            shoulder, rotate_vector(upper_rotation, self.geometry[f"upper_arm_{suffix}"])
        )
        forearm_rotation = _qmul(upper_rotation, self._rotation(f"{side}_elbow", frame))
        wrist = _add(
            elbow, rotate_vector(forearm_rotation, self.geometry[f"forearm_{suffix}"])
        )
        joints[f"shoulder_{suffix}"] = shoulder
        joints[f"elbow_{suffix}"] = elbow
        joints[f"wrist_{suffix}"] = wrist

    def _solve_leg(
        self,
        suffix: str,
        side: str,
        pelvis: Vector3,
        pelvis_reference: Vector3,
        root_position: Vector3,
        root_rotation: Quaternion,
        lower_rotation: Quaternion,
        center: Vector3,
        frame: int,
        joints: dict[str, Vector3],
        warnings: list[str],
    ) -> None:
        hip_offset = self.geometry[f"hip_{suffix}_offset"]
        thigh_vector = self.geometry[f"thigh_{suffix}"]
        shin_vector = self.geometry[f"shin_{suffix}"]
        hip = _add(pelvis, rotate_vector(lower_rotation, hip_offset))
        ik_semantic = f"{side}_foot_ik"
        if bool(self.profile.get("leg_ik", {}).get("enabled", True)) and self._has_track(ik_semantic):
            ik_position = self._position(ik_semantic, frame)
            ik_parent_semantic = f"{side}_foot_ik_parent"
            if self._has_track(ik_parent_semantic):
                parent_position = self._position(ik_parent_semantic, frame)
                parent_rotation = self._rotation(ik_parent_semantic, frame)
                ik_position = _add(
                    parent_position,
                    rotate_vector(parent_rotation, ik_position),
                )
            rest_ankle = _add(_add(_add(pelvis_reference, hip_offset), thigh_vector), shin_vector)
            ankle_target = _add(root_position, rotate_vector(root_rotation, _add(rest_ankle, ik_position)))
            hint = _vector(self.profile["leg_ik"].get(f"{side}_bend_hint", [0, 0, -1]))
            knee, ankle, clamped = solve_two_bone_3d(
                hip,
                ankle_target,
                _length(thigh_vector),
                _length(shin_vector),
                # The knee pole follows the character's lower-body facing,
                # including Center/Groove/Waist turns.  Rotating it by only
                # 全ての親 left the knee plane behind during local turns.
                rotate_vector(lower_rotation, hint),
            )
            if clamped:
                warnings.append(f"{side}_leg_ik_target_clamped")
        else:
            thigh_rotation = _qmul(lower_rotation, self._rotation(f"{side}_leg", frame))
            knee = _add(hip, rotate_vector(thigh_rotation, thigh_vector))
            shin_rotation = _qmul(thigh_rotation, self._rotation(f"{side}_knee", frame))
            ankle = _add(knee, rotate_vector(shin_rotation, shin_vector))
        joints[f"hip_{suffix}"] = hip
        joints[f"knee_{suffix}"] = knee
        joints[f"ankle_{suffix}"] = ankle

    def _track_name(self, semantic: str) -> str | None:
        for name in self.aliases.get(semantic, []):
            if name in self.motion.bone_tracks:
                return name
        return None

    def _has_track(self, semantic: str) -> bool:
        return self._track_name(semantic) is not None

    def _position(self, semantic: str, frame: int) -> Vector3:
        name = self._track_name(semantic)
        return sample_track(self.motion.bone_tracks[name], frame).position if name else ZERO

    def _rotation(self, semantic: str, frame: int) -> Quaternion:
        name = self._track_name(semantic)
        return sample_track(self.motion.bone_tracks[name], frame).rotation if name else IDENTITY

    def _rotation_chain(
        self,
        name: str,
        frame: int,
        fallback: tuple[str, ...],
    ) -> Quaternion:
        semantics = self.parent_chains.get(name, list(fallback))
        rotation = IDENTITY
        for semantic in semantics:
            rotation = _qmul(rotation, self._rotation(str(semantic), frame))
        return rotation


def solve_two_bone_3d(
    start: Vector3,
    target: Vector3,
    first_length: float,
    second_length: float,
    bend_hint: Vector3,
) -> tuple[Vector3, Vector3, bool]:
    delta = _subtract(target, start)
    distance = _length(delta)
    if distance < 1e-8:
        direction = (0.0, -1.0, 0.0)
        distance = 1e-8
    else:
        direction = _scale(delta, 1.0 / distance)
    minimum = abs(first_length - second_length) + 1e-5
    # Exact full extension is well-defined here: the bend height becomes zero
    # and the configured hint is simply unused.  Keeping an epsilon below the
    # maximum forced every source leg to retain a small artificial knee bend.
    maximum = first_length + second_length
    clamped_distance = min(maximum, max(minimum, distance))
    clamped = abs(clamped_distance - distance) > 1e-6
    ankle = _add(start, _scale(direction, clamped_distance))
    along = (
        first_length * first_length
        - second_length * second_length
        + clamped_distance * clamped_distance
    ) / (2.0 * clamped_distance)
    height = math.sqrt(max(0.0, first_length * first_length - along * along))
    perpendicular = _subtract(bend_hint, _scale(direction, _dot(bend_hint, direction)))
    if _length(perpendicular) < 1e-7:
        fallback = (1.0, 0.0, 0.0) if abs(direction[0]) < 0.9 else (0.0, 0.0, 1.0)
        perpendicular = _cross(direction, fallback)
    perpendicular = _scale(perpendicular, 1.0 / max(_length(perpendicular), 1e-8))
    knee = _add(_add(start, _scale(direction, along)), _scale(perpendicular, height))
    return knee, ankle, clamped


def pose_sample_to_dict(sample: PoseSample, projector: Projector) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for name, segment in sample.segments.items():
        first = projector.project_point(segment.start)
        second = projector.project_point(segment.end)
        groups[name] = {
            "joints": [segment.start_joint, segment.end_joint],
            "world_start": list(segment.start),
            "world_end": list(segment.end),
            "camera_start": list(first.camera),
            "camera_end": list(second.camera),
            "source_screen_start": list(first.screen),
            "source_screen_end": list(second.screen),
            "depth_start": first.camera_depth,
            "depth_end": second.camera_depth,
            "representative_depth": (first.camera_depth + second.camera_depth) * 0.5,
            "depth_source": segment.depth_source,
            "confidence": segment.confidence,
        }
    return {
        "source_frame": sample.source_frame,
        "output_frame": sample.output_frame,
        "time_seconds": sample.time_seconds,
        "groups": groups,
        "warnings": sample.warnings,
    }


def _vector(value) -> Vector3:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"Expected 3D vector, received {value!r}")
    return float(value[0]), float(value[1]), float(value[2])


def _qmul(first: Quaternion, second: Quaternion) -> Quaternion:
    ax, ay, az, aw = first
    bx, by, bz, bw = second
    value = (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )
    length = math.sqrt(sum(component * component for component in value))
    return tuple(component / length for component in value) if length > 1e-8 else IDENTITY  # type: ignore[return-value]


def _add(first: Vector3, second: Vector3) -> Vector3:
    return tuple(a + b for a, b in zip(first, second))  # type: ignore[return-value]


def _subtract(first: Vector3, second: Vector3) -> Vector3:
    return tuple(a - b for a, b in zip(first, second))  # type: ignore[return-value]


def _scale(value: Vector3, amount: float) -> Vector3:
    return tuple(component * amount for component in value)  # type: ignore[return-value]


def _dot(first: Vector3, second: Vector3) -> float:
    return sum(a * b for a, b in zip(first, second))


def _cross(first: Vector3, second: Vector3) -> Vector3:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _length(value: Vector3) -> float:
    return math.sqrt(_dot(value, value))
