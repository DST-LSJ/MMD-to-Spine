"""Constraint-first MMD 3D to Spine 2D semantic retargeting.

This module deliberately does not copy source bone rotations.  It reconstructs
a small world-space humanoid from VMD data, projects joint directions through
one fixed camera, and solves the supplied Spine setup pose from parent to
child.  The target limbs keep their authored origins and lengths; only body
translation, rotations, and discrete body/head facing mirrors are generated.

The workspace does not contain the original PMX/PMD model.  Consequently the
3D pose is an explicit canonical-rig approximation described by
``source_rig_profile.json``.  Keeping that limitation visible is preferable to
silently treating VMD local positions as world-space joints.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
from typing import Any

from .projection import Projector, Quaternion, Vector3, rotate_vector
from .retarget import (
    BoneAnimation,
    RetargetedAnimation,
    ScalarKey,
    VectorKey,
)
from .source_pose import PoseSample, Segment3D, solve_two_bone_3d
from .spine_reader import SpineBone, SpineSkeleton
from .vmd_reader import (
    BoneKeyframe,
    IDENTITY_KEY,
    VMDMotion,
    interpolation_amount,
)


Matrix2D = tuple[float, float, float, float, float, float]
IDENTITY_Q: Quaternion = (0.0, 0.0, 0.0, 1.0)
ZERO_3: Vector3 = (0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class SemanticRetargetConfig:
    """The small set of policy values used by the new retargeter."""

    # This is the single authored view/facing value.  It replaces the old
    # hand-tuned 145-degree experiment everywhere in the new pipeline.
    view_offset_deg: float = 30.0
    mirror_span_deg: float = 180.0
    mirror_hysteresis_deg: float = 4.0
    mirror_min_hold_frames: int = 2
    head_right_mirror_deg: float = 30.0
    projected_direction_min_ratio: float = 0.04
    source_direction_max_step_deg: float = 45.0
    thigh_branch_switch_vertical_cone_deg: float = 20.0
    thigh_camera_side_multiplier: float = -1.0
    body_rotation_limit_deg: float = 45.0
    head_rotation_limit_deg: float = 60.0
    thigh_rotation_limit_deg: float = 110.0
    maximum_knee_flex_deg: float = 140.0
    knee_bend_sign: float = -1.0  # clockwise in screen coordinates
    body_translation_gain: float = 1.0

    @classmethod
    def from_mapping(cls, value: dict[str, Any] | None) -> "SemanticRetargetConfig":
        value = value or {}
        return cls(
            view_offset_deg=float(value.get("view_offset_deg", 30.0)),
            mirror_span_deg=float(value.get("mirror_span_deg", 180.0)),
            mirror_hysteresis_deg=float(value.get("mirror_hysteresis_deg", 4.0)),
            mirror_min_hold_frames=max(
                0, int(value.get("mirror_min_hold_frames", 2))
            ),
            head_right_mirror_deg=float(
                value.get("head_right_mirror_deg", 30.0)
            ),
            projected_direction_min_ratio=float(
                value.get("projected_direction_min_ratio", 0.04)
            ),
            source_direction_max_step_deg=float(
                value.get("source_direction_max_step_deg", 45.0)
            ),
            thigh_branch_switch_vertical_cone_deg=min(
                90.0,
                max(
                    0.0,
                    float(
                        value.get(
                            "thigh_branch_switch_vertical_cone_deg", 20.0
                        )
                    ),
                ),
            ),
            thigh_camera_side_multiplier=(
                -1.0
                if float(value.get("thigh_camera_side_multiplier", -1.0)) < 0.0
                else 1.0
            ),
            body_rotation_limit_deg=abs(
                float(value.get("body_rotation_limit_deg", 45.0))
            ),
            head_rotation_limit_deg=abs(
                float(value.get("head_rotation_limit_deg", 60.0))
            ),
            thigh_rotation_limit_deg=abs(
                float(value.get("thigh_rotation_limit_deg", 110.0))
            ),
            maximum_knee_flex_deg=max(
                0.0, float(value.get("maximum_knee_flex_deg", 140.0))
            ),
            knee_bend_sign=(
                -1.0 if float(value.get("knee_bend_sign", -1.0)) < 0.0 else 1.0
            ),
            body_translation_gain=float(
                value.get("body_translation_gain", 1.0)
            ),
        )


@dataclass(slots=True)
class SemanticSourceFrame:
    source_frame: int
    output_frame: int
    pose: PoseSample
    motion_offset: Vector3
    body_up: Vector3
    head_up: Vector3
    head_right: Vector3
    body_heading_deg: float
    head_heading_deg: float


@dataclass(slots=True)
class SemanticRetargetResult:
    animation: RetargetedAnimation
    pose_samples: list[PoseSample]
    diagnostics: dict[str, Any]


class _TrackSampler:
    """Fast integer-frame VMD sampler with cached track frame arrays."""

    def __init__(self, motion: VMDMotion):
        self.motion = motion
        self.frames = {
            name: [key.frame for key in track]
            for name, track in motion.bone_tracks.items()
        }
        self.cache_frame: int | None = None
        self.cache: dict[str, BoneKeyframe] = {}

    def sample(self, name: str | None, frame: int) -> BoneKeyframe:
        if not name or name not in self.motion.bone_tracks:
            return IDENTITY_KEY
        if frame != self.cache_frame:
            self.cache_frame = frame
            self.cache = {}
        cached = self.cache.get(name)
        if cached is not None:
            return cached
        track = self.motion.bone_tracks[name]
        track_frames = self.frames[name]
        position = bisect_right(track_frames, frame)
        if position == 0:
            result = IDENTITY_KEY if frame < track[0].frame else track[0]
        elif position >= len(track):
            result = track[-1]
        else:
            first, second = track[position - 1], track[position]
            if first.frame == frame:
                result = first
            else:
                span = second.frame - first.frame
                if span <= 0:
                    result = second
                else:
                    amount = (frame - first.frame) / span
                    channel_amounts = [
                        interpolation_amount(first.interpolation, channel, amount)
                        for channel in range(4)
                    ]
                    sampled_position = tuple(
                        first.position[index]
                        + (second.position[index] - first.position[index])
                        * channel_amounts[index]
                        for index in range(3)
                    )
                    sampled_rotation = _slerp(
                        first.rotation, second.rotation, channel_amounts[3]
                    )
                    result = BoneKeyframe(
                        name,
                        frame,
                        sampled_position,  # type: ignore[arg-type]
                        sampled_rotation,
                        first.interpolation,
                    )
        self.cache[name] = result
        return result


class SemanticPoseSolver:
    """Reconstruct a canonical 3D humanoid and expose semantic world signals."""

    def __init__(
        self,
        motion: VMDMotion,
        profile: dict[str, Any],
        projector: Projector,
    ) -> None:
        self.motion = motion
        self.profile = profile
        self.projector = projector
        self.sampler = _TrackSampler(motion)
        self.aliases: dict[str, list[str]] = profile["aliases"]
        self.geometry = {
            name: _vector3(value) for name, value in profile["geometry"].items()
        }
        self.target_bindings: dict[str, str] = profile.get("target_bindings", {})
        self.depth_source = str(profile.get("depth_source", "canonical_rig_approximation"))
        self.body_height = float(profile.get("body_height", 16.0))
        self.actual_names = {
            semantic: next(
                (name for name in names if name in motion.bone_tracks),
                None,
            )
            for semantic, names in self.aliases.items()
        }
        self.used_tracks: set[str] = set()

    def sample(self, source_frame: int, output_frame: int, fps: float) -> SemanticSourceFrame:
        warnings: list[str] = []
        root_position = self._position("all_parent", source_frame)
        root_rotation = self._rotation("all_parent", source_frame)
        center = _add3(
            self._position("center", source_frame),
            self._position("groove", source_frame),
            self._position("waist", source_frame),
        )
        motion_offset = _add3(root_position, rotate_vector(root_rotation, center))

        base_rotation = self._rotation_chain(
            ("all_parent", "center", "groove", "waist"), source_frame
        )
        lower_rotation = _qmul(
            base_rotation, self._rotation("lower_body", source_frame)
        )
        upper_rotation = _qmul(
            _qmul(base_rotation, self._rotation("upper_body", source_frame)),
            self._rotation("upper_body_2", source_frame),
        )

        pelvis_reference = self.geometry["pelvis_reference"]
        pelvis = _add3(motion_offset, rotate_vector(base_rotation, pelvis_reference))
        chest = _add3(
            pelvis, rotate_vector(upper_rotation, self.geometry["chest_offset"])
        )
        joints: dict[str, Vector3] = {"pelvis": pelvis, "chest": chest}

        self._solve_arm(
            "F",
            self.target_bindings.get("arm_F", "right"),
            chest,
            upper_rotation,
            source_frame,
            joints,
        )
        self._solve_arm(
            "B",
            self.target_bindings.get("arm_B", "left"),
            chest,
            upper_rotation,
            source_frame,
            joints,
        )
        self._solve_leg(
            "F",
            self.target_bindings.get("leg_F", "right"),
            pelvis,
            pelvis_reference,
            root_position,
            root_rotation,
            lower_rotation,
            source_frame,
            joints,
            warnings,
        )
        self._solve_leg(
            "B",
            self.target_bindings.get("leg_B", "left"),
            pelvis,
            pelvis_reference,
            root_position,
            root_rotation,
            lower_rotation,
            source_frame,
            joints,
            warnings,
        )

        head_rotation = _qmul(
            _qmul(upper_rotation, self._rotation("neck", source_frame)),
            self._rotation("head", source_frame),
        )
        lower_up = rotate_vector(lower_rotation, (0.0, 1.0, 0.0))
        upper_up = rotate_vector(upper_rotation, (0.0, 1.0, 0.0))
        body_up = _normalise3(
            _add3(_scale3(lower_up, 0.45), _scale3(upper_up, 0.55))
        )
        head_up = rotate_vector(head_rotation, (0.0, 1.0, 0.0))
        head_right = rotate_vector(head_rotation, (1.0, 0.0, 0.0))

        confidence = self.profile.get("confidence", {})
        leg_confidence = float(
            confidence.get(
                "legs_ik"
                if self._has("left_foot_ik") or self._has("right_foot_ik")
                else "legs_fk",
                0.5,
            )
        )
        arm_confidence = float(confidence.get("arms", 0.65))
        body_confidence = float(confidence.get("body", 0.7))
        segments = {
            "body": Segment3D("body", "pelvis", "chest", pelvis, chest, body_confidence),
            "whole_arm_F": Segment3D("whole_arm_F", "shoulder_F", "wrist_F", joints["shoulder_F"], joints["wrist_F"], arm_confidence),
            "upper_arm_F": Segment3D("upper_arm_F", "shoulder_F", "elbow_F", joints["shoulder_F"], joints["elbow_F"], arm_confidence),
            "forearm_hand_F": Segment3D("forearm_hand_F", "elbow_F", "wrist_F", joints["elbow_F"], joints["wrist_F"], arm_confidence),
            "whole_arm_B": Segment3D("whole_arm_B", "shoulder_B", "wrist_B", joints["shoulder_B"], joints["wrist_B"], arm_confidence),
            "upper_arm_B": Segment3D("upper_arm_B", "shoulder_B", "elbow_B", joints["shoulder_B"], joints["elbow_B"], arm_confidence),
            "forearm_hand_B": Segment3D("forearm_hand_B", "elbow_B", "wrist_B", joints["elbow_B"], joints["wrist_B"], arm_confidence),
            "whole_leg_F": Segment3D("whole_leg_F", "hip_F", "ankle_F", joints["hip_F"], joints["ankle_F"], leg_confidence),
            "upper_leg_F": Segment3D("upper_leg_F", "hip_F", "knee_F", joints["hip_F"], joints["knee_F"], leg_confidence),
            "lower_leg_F": Segment3D("lower_leg_F", "knee_F", "ankle_F", joints["knee_F"], joints["ankle_F"], leg_confidence),
            "whole_leg_B": Segment3D("whole_leg_B", "hip_B", "ankle_B", joints["hip_B"], joints["ankle_B"], leg_confidence),
            "upper_leg_B": Segment3D("upper_leg_B", "hip_B", "knee_B", joints["hip_B"], joints["knee_B"], leg_confidence),
            "lower_leg_B": Segment3D("lower_leg_B", "knee_B", "ankle_B", joints["knee_B"], joints["ankle_B"], leg_confidence),
        }
        pose = PoseSample(
            source_frame,
            output_frame,
            output_frame / fps,
            joints,
            segments,
            warnings,
        )
        return SemanticSourceFrame(
            source_frame,
            output_frame,
            pose,
            motion_offset,
            body_up,
            head_up,
            head_right,
            _heading_deg(lower_rotation),
            _heading_deg(head_rotation),
        )

    def _solve_arm(
        self,
        suffix: str,
        side: str,
        chest: Vector3,
        chest_rotation: Quaternion,
        frame: int,
        joints: dict[str, Vector3],
    ) -> None:
        shoulder = _add3(
            chest,
            rotate_vector(chest_rotation, self.geometry[f"shoulder_{suffix}_offset"]),
        )
        upper_rotation = chest_rotation
        for semantic in (
            f"{side}_shoulder_parent",
            f"{side}_shoulder",
            f"{side}_arm",
        ):
            upper_rotation = _qmul(upper_rotation, self._rotation(semantic, frame))
        elbow = _add3(
            shoulder,
            rotate_vector(upper_rotation, self.geometry[f"upper_arm_{suffix}"]),
        )
        forearm_rotation = _qmul(
            upper_rotation, self._rotation(f"{side}_elbow", frame)
        )
        wrist = _add3(
            elbow,
            rotate_vector(forearm_rotation, self.geometry[f"forearm_{suffix}"]),
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
        frame: int,
        joints: dict[str, Vector3],
        warnings: list[str],
    ) -> None:
        hip_offset = self.geometry[f"hip_{suffix}_offset"]
        thigh_vector = self.geometry[f"thigh_{suffix}"]
        shin_vector = self.geometry[f"shin_{suffix}"]
        hip = _add3(pelvis, rotate_vector(lower_rotation, hip_offset))
        ik_semantic = f"{side}_foot_ik"
        ik_name = self.actual_names.get(ik_semantic)
        use_ik = (
            bool(self.profile.get("leg_ik", {}).get("enabled", True))
            and self._has(ik_semantic)
            and self._ik_enabled(ik_name, frame)
        )
        if use_ik:
            ik_position = self._position(ik_semantic, frame)
            ik_parent_semantic = f"{side}_foot_ik_parent"
            if self._has(ik_parent_semantic):
                parent_position = self._position(ik_parent_semantic, frame)
                parent_rotation = self._rotation(ik_parent_semantic, frame)
                ik_position = _add3(
                    parent_position,
                    rotate_vector(parent_rotation, ik_position),
                )
            rest_ankle = _add3(
                pelvis_reference, hip_offset, thigh_vector, shin_vector
            )
            ankle_target = _add3(
                root_position,
                rotate_vector(root_rotation, _add3(rest_ankle, ik_position)),
            )
            hint = _vector3(
                self.profile.get("leg_ik", {}).get(
                    f"{side}_bend_hint", [0.0, 0.0, -1.0]
                )
            )
            knee, ankle, clamped = solve_two_bone_3d(
                hip,
                ankle_target,
                _length3(thigh_vector),
                _length3(shin_vector),
                rotate_vector(lower_rotation, hint),
            )
            if clamped:
                warnings.append(f"{side}_leg_ik_target_clamped")
        else:
            thigh_rotation = _qmul(
                lower_rotation, self._rotation(f"{side}_leg", frame)
            )
            knee = _add3(hip, rotate_vector(thigh_rotation, thigh_vector))
            shin_rotation = _qmul(
                thigh_rotation, self._rotation(f"{side}_knee", frame)
            )
            ankle = _add3(knee, rotate_vector(shin_rotation, shin_vector))
        joints[f"hip_{suffix}"] = hip
        joints[f"knee_{suffix}"] = knee
        joints[f"ankle_{suffix}"] = ankle

    def _ik_enabled(self, name: str | None, frame: int) -> bool:
        if not name or not self.motion.ik_display_frames:
            return True
        enabled = True
        for display in self.motion.ik_display_frames:
            if display.frame > frame:
                break
            for state in display.states:
                if state.name == name:
                    enabled = state.enabled
        return enabled

    def _has(self, semantic: str) -> bool:
        return self.actual_names.get(semantic) is not None

    def _position(self, semantic: str, frame: int) -> Vector3:
        name = self.actual_names.get(semantic)
        if name:
            self.used_tracks.add(name)
        return self.sampler.sample(name, frame).position

    def _rotation(self, semantic: str, frame: int) -> Quaternion:
        name = self.actual_names.get(semantic)
        if name:
            self.used_tracks.add(name)
        return self.sampler.sample(name, frame).rotation

    def _rotation_chain(
        self, semantics: tuple[str, ...], frame: int
    ) -> Quaternion:
        result = IDENTITY_Q
        for semantic in semantics:
            result = _qmul(result, self._rotation(semantic, frame))
        return result


def retarget_semantic_motion(
    motion: VMDMotion,
    skeleton: SpineSkeleton,
    solver: SemanticPoseSolver,
    projector: Projector,
    *,
    fps: float,
    start_frame: int,
    end_frame: int,
    config: SemanticRetargetConfig,
    progress=None,
) -> SemanticRetargetResult:
    """Build the complete body/head/limb main pose from 3D joint directions."""

    if fps <= 0.0:
        raise ValueError("fps must be positive")
    if end_frame < start_frame:
        raise ValueError("end_frame must not be before start_frame")
    required = {
        "root",
        "body",
        "head",
        "armF",
        "handF",
        "armB",
        "handB",
        "thighF",
        "legF",
        "thighB",
        "legB",
    }
    missing = sorted(required - set(skeleton.bone_by_name))
    if missing:
        raise ValueError(f"Spine template is missing required bones: {missing}")

    frames = []
    total=end_frame-start_frame+1
    for index, source in enumerate(range(start_frame,end_frame+1)):
        frames.append(solver.sample(source,source-start_frame,fps))
        if progress is not None and (index%30==0 or index+1==total):
            progress(.4*(index+1)/total,'计算三维骨架')
    pose_samples = [value.pose for value in frames]
    output_count = len(frames)
    if output_count == 0:
        raise ValueError("No source frames were selected")

    setup_world = _setup_world_matrices(skeleton)
    bones = skeleton.bone_by_name
    target_setup_angles = _target_setup_angles(skeleton, setup_world)

    body_angles, invalid_body = _stable_direction_angles(
        [value.body_up for value in frames],
        projector,
        fallback_deg=target_setup_angles["body"],
        minimum_ratio=config.projected_direction_min_ratio,
        maximum_step_deg=config.source_direction_max_step_deg,
    )
    head_vectors = [
        _head_projection_vector(value, projector) for value in frames
    ]
    head_angles, invalid_head = _stable_screen_angles(
        head_vectors,
        fallback_deg=target_setup_angles["head"],
        minimum_length=1e-5,
        maximum_step_deg=config.source_direction_max_step_deg,
    )

    segment_vectors: dict[str, list[Vector3]] = {}
    for suffix in ("F", "B"):
        segment_vectors[f"arm{suffix}"] = [
            _subtract3(
                value.pose.joints[f"elbow_{suffix}"],
                value.pose.joints[f"shoulder_{suffix}"],
            )
            for value in frames
        ]
        segment_vectors[f"hand{suffix}"] = [
            _subtract3(
                value.pose.joints[f"wrist_{suffix}"],
                value.pose.joints[f"elbow_{suffix}"],
            )
            for value in frames
        ]
        segment_vectors[f"thigh{suffix}"] = [
            _subtract3(
                value.pose.joints[f"knee_{suffix}"],
                value.pose.joints[f"hip_{suffix}"],
            )
            for value in frames
        ]

    segment_angles: dict[str, list[float]] = {}
    invalid_counts: dict[str, int] = {
        "body": invalid_body,
        "head": invalid_head,
    }
    fallback = {
        "armF": target_setup_angles["armF"],
        "handF": target_setup_angles["handF"],
        "armB": target_setup_angles["armB"],
        "handB": target_setup_angles["handB"],
        "thighF": target_setup_angles["thighF"],
        "thighB": target_setup_angles["thighB"],
    }
    for name, values in segment_vectors.items():
        if name.startswith("thigh"):
            # A 2D leg cannot express perspective shortening.  Preserve the
            # source segment's true 3D elevation instead: collapse its whole
            # camera-horizontal (X/Z) length onto one screen side while
            # retaining Y.  This makes a kick towards the camera as visible as
            # a sideways kick, without scaling or translating the target bone.
            angles, invalid = _stable_depth_collapsed_angles(
                values,
                projector,
                fallback_deg=fallback[name],
                side_deadzone_ratio=config.projected_direction_min_ratio,
                minimum_hold_frames=config.mirror_min_hold_frames,
                switch_vertical_cone_deg=(
                    config.thigh_branch_switch_vertical_cone_deg
                ),
                screen_side_multiplier=config.thigh_camera_side_multiplier,
                maximum_step_deg=config.source_direction_max_step_deg,
            )
        else:
            angles, invalid = _stable_direction_angles(
                values,
                projector,
                fallback_deg=fallback[name],
                minimum_ratio=config.projected_direction_min_ratio,
                maximum_step_deg=config.source_direction_max_step_deg,
            )
        segment_angles[name] = angles
        invalid_counts[name] = invalid

    # Preserve the authored, slightly splayed target stance while applying the
    # source thigh's motion relative to a canonical straight-down leg.
    source_rest_leg_angle = _angle2(*projector.direction((0.0, -1.0, 0.0)))
    thigh_offsets = {
        suffix: _wrap_degrees(
            target_setup_angles[f"thigh{suffix}"] - source_rest_leg_angle
        )
        for suffix in ("F", "B")
    }
    # Opt-in measured-rest calibration for PMX FK fixtures. The canonical
    # profile and the existing dance conversion retain their prior behavior.
    measured_leg_rest = solver.profile.get("calibrate_measured_leg_rest", False)
    if measured_leg_rest:
        for suffix in ("F", "B"):
            rx, ry, rz = projector.camera_space(solver.geometry[f"thigh_{suffix}"])
            horizontal = math.hypot(rx, rz)
            # The semantic leg can change its display branch near vertical.
            # Calibrate rest on that same branch so a real, slightly splayed
            # PMX leg returns to target setup on either side of a test reset.
            segment_angles[f"thigh{suffix}"] = [
                angle - _angle2(math.copysign(horizontal, math.cos(math.radians(angle))), ry)
                + source_rest_leg_angle
                for angle in segment_angles[f"thigh{suffix}"]
            ]

    # Calibrate every source bind direction to the actual Spine setup pose.
    # A neutral MMD file must therefore emit zero rotation values and leave the
    # authored natural-standing template untouched instead of forcing its arms
    # into the MMD T-pose.
    setup_pose_offsets: dict[str, float] = {}
    for suffix in ("F", "B"):
        for target_name, geometry_name in (
            (f"arm{suffix}", f"upper_arm_{suffix}"),
            (f"hand{suffix}", f"forearm_{suffix}"),
        ):
            source_rest_angle = _angle2(
                *projector.direction(solver.geometry[geometry_name])
            )
            setup_pose_offsets[target_name] = _wrap_degrees(
                target_setup_angles[target_name] - source_rest_angle
            )

    setup_knee_bends = {
        suffix: _wrap_degrees(
            target_setup_angles[f"leg{suffix}"]
            - target_setup_angles[f"thigh{suffix}"]
        )
        for suffix in ("F", "B")
    }

    flexion: dict[str, list[float]] = {"F": [], "B": []}
    for suffix in ("F", "B"):
        rest_flexion = 0.0
        if measured_leg_rest:
            rest_dot = _dot3(_normalise3(solver.geometry[f"thigh_{suffix}"]),
                             _normalise3(solver.geometry[f"shin_{suffix}"]))
            rest_flexion = math.degrees(math.acos(min(1.0, max(-1.0, rest_dot))))
        for upper, lower in zip(
            segment_vectors[f"thigh{suffix}"],
            [
                _subtract3(
                    value.pose.joints[f"ankle_{suffix}"],
                    value.pose.joints[f"knee_{suffix}"],
                )
                for value in frames
            ],
        ):
            upper_n = _normalise3(upper)
            lower_n = _normalise3(lower)
            dot = min(1.0, max(-1.0, _dot3(upper_n, lower_n)))
            amount = math.degrees(math.acos(dot)) - rest_flexion
            flexion[suffix].append(
                min(config.maximum_knee_flex_deg, max(0.0, amount))
            )

    body_states = _interval_states(
        [value.body_heading_deg for value in frames],
        start_deg=config.view_offset_deg,
        span_deg=config.mirror_span_deg,
        hysteresis_deg=config.mirror_hysteresis_deg,
        minimum_hold_frames=config.mirror_min_hold_frames,
    )
    relative_head_yaw = [
        _wrap_degrees(value.head_heading_deg - value.body_heading_deg)
        for value in frames
    ]
    head_states = _threshold_states(
        relative_head_yaw,
        threshold_deg=config.head_right_mirror_deg,
        hysteresis_deg=config.mirror_hysteresis_deg,
        minimum_hold_frames=config.mirror_min_hold_frames,
    )
    body_switch_frames = _state_switch_frames(body_states)
    head_switch_frames = _state_switch_frames(head_states)

    generated: dict[str, BoneAnimation] = {
        name: BoneAnimation()
        for name in (
            "body",
            "head",
            "armF",
            "handF",
            "armB",
            "handB",
            "thighF",
            "legF",
            "thighB",
            "legB",
        )
    }

    root_world = setup_world["root"]
    projected_leg_ik = None
    if solver.profile.get("leg_retarget_method") == "projected_two_link_ik":
        from .projected_leg_ik import ProjectedLegIK
        projected_leg_ik = ProjectedLegIK(skeleton, solver, projector, config)
    for index, source in enumerate(frames):
        if progress is not None and (index%30==0 or index+1==output_count):
            progress(.4+.6*index/output_count,'转换二维动作')
        output_frame = source.output_frame
        mirror_x = -1.0 if body_states[index] else 1.0
        head_mirror_x = -1.0 if head_states[index] else 1.0

        body_value = _solve_local_rotation(
            root_world,
            bones["body"],
            (0.0, 1.0),
            body_angles[index],
            scale_x_multiplier=mirror_x,
        )
        body_value = min(
            config.body_rotation_limit_deg,
            max(-config.body_rotation_limit_deg, body_value),
        )
        body_translation = projector.position(source.motion_offset)
        body_translation = (
            body_translation[0] * config.body_translation_gain,
            body_translation[1] * config.body_translation_gain,
        )
        generated["body"].rotate.append(ScalarKey(output_frame, body_value))
        generated["body"].translate.append(
            VectorKey(output_frame, *body_translation)
        )
        body_world = _multiply_matrices(
            root_world,
            _animated_local_matrix(
                bones["body"],
                body_value,
                body_translation,
                (mirror_x, 1.0),
            ),
        )

        head_value = _solve_local_rotation(
            body_world,
            bones["head"],
            (0.0, 1.0),
            head_angles[index],
            scale_x_multiplier=head_mirror_x,
        )
        head_value = min(
            config.head_rotation_limit_deg,
            max(-config.head_rotation_limit_deg, head_value),
        )
        generated["head"].rotate.append(ScalarKey(output_frame, head_value))

        for suffix in ("F", "B"):
            arm_name = f"arm{suffix}"
            hand_name = f"hand{suffix}"
            arm_value = _solve_local_rotation(
                body_world,
                bones[arm_name],
                (bones[hand_name].x, bones[hand_name].y),
                segment_angles[arm_name][index]
                + mirror_x * setup_pose_offsets[arm_name],
            )
            generated[arm_name].rotate.append(ScalarKey(output_frame, arm_value))
            arm_world = _multiply_matrices(
                body_world,
                _animated_local_matrix(bones[arm_name], arm_value),
            )
            hand_value = _solve_local_rotation(
                arm_world,
                bones[hand_name],
                (1.0, 0.0),
                segment_angles[hand_name][index]
                + mirror_x * setup_pose_offsets[hand_name],
            )
            generated[hand_name].rotate.append(ScalarKey(output_frame, hand_value))

            thigh_name = f"thigh{suffix}"
            leg_name = f"leg{suffix}"
            if projected_leg_ik is not None:
                thigh_value, leg_value = projected_leg_ik.solve(source, suffix, body_world, mirror_x)
                generated[thigh_name].rotate.append(ScalarKey(output_frame, thigh_value))
                generated[leg_name].rotate.append(ScalarKey(output_frame, leg_value))
                continue
            desired_thigh = (
                segment_angles[thigh_name][index]
                + mirror_x * thigh_offsets[suffix]
            )
            thigh_value = _solve_local_rotation(
                body_world,
                bones[thigh_name],
                (bones[leg_name].x, bones[leg_name].y),
                desired_thigh,
            )
            thigh_value = min(
                config.thigh_rotation_limit_deg,
                max(-config.thigh_rotation_limit_deg, thigh_value),
            )
            generated[thigh_name].rotate.append(
                ScalarKey(output_frame, thigh_value)
            )
            thigh_world = _multiply_matrices(
                body_world,
                _animated_local_matrix(bones[thigh_name], thigh_value),
            )
            # Both knees share one screen-space branch.  A zero source flexion
            # is an exactly straight target leg; it is never forced to retain
            # an artificial bend and can never cross into a reverse joint.
            actual_thigh_world = _matrix_direction_angle(
                thigh_world, (bones[leg_name].x, bones[leg_name].y)
            )
            desired_leg = (
                actual_thigh_world
                + mirror_x * setup_knee_bends[suffix]
                + _knee_screen_bend_delta(
                    flexion[suffix][index],
                    config.knee_bend_sign,
                    mirror_x,
                )
            )
            leg_value = _solve_local_rotation(
                thigh_world,
                bones[leg_name],
                (1.0, 0.0),
                desired_leg,
            )
            generated[leg_name].rotate.append(ScalarKey(output_frame, leg_value))

        if projected_leg_ik is not None:
            projected_leg_ik.previous_mirror = mirror_x

    generated["body"].scale = _state_scale_keys(body_states)
    generated["head"].scale = _state_scale_keys(head_states)

    for name, timeline in generated.items():
        resets = body_switch_frames if name != "body" else set()
        if name == "head":
            resets = resets | head_switch_frames
        timeline.rotate = _unwrap_segmented(timeline.rotate, resets)
        if resets:
            timeline.rotate = _mark_stepped_before(timeline.rotate, resets)

    used = sorted(solver.used_tracks)
    animation = RetargetedAnimation(
        fps=fps,
        max_frame=end_frame - start_frame,
        bones=generated,
        source_bones_used=used,
        source_bones_unmapped=sorted(set(motion.bone_tracks) - set(used)),
        source_start_frame=start_frame,
    )
    animation.arm_pose_diagnostics = {
        "method": "source_bind_delta_to_spine_setup_pose",
        "fixed_limb_length": True,
        "limb_scale_timelines": False,
        "source_rig": solver.depth_source,
        "view_offset_deg": config.view_offset_deg,
        "fixed_arm_lift_compensation_deg": 0.0,
        "setup_pose_offsets_deg": setup_pose_offsets,
    }
    diagnostics = {
        "algorithm": "semantic_constraint_retarget_v2",
        "source_depth": solver.depth_source,
        "view_and_mirror_offset_deg": config.view_offset_deg,
        "body_mirror_switch_frames": sorted(body_switch_frames),
        "head_mirror_switch_frames": sorted(head_switch_frames),
        "invalid_projected_direction_frames": invalid_counts,
        "maximum_knee_flex_deg": {
            suffix: max(values, default=0.0)
            for suffix, values in flexion.items()
        },
        "thigh_projection": "depth_collapsed_true_elevation",
        "thigh_camera_side_multiplier": config.thigh_camera_side_multiplier,
        "setup_knee_bend_deg": setup_knee_bends,
        "neutral_source_preserves_spine_setup_pose": True,
        "knee_bend_sign": config.knee_bend_sign,
        "knee_bend_follows_body_mirror": True,
        "body_mirror_count": sum(body_states),
        "head_mirror_count": sum(head_states),
    }
    if projected_leg_ik is not None:
        diagnostics["leg_ik_2d"] = projected_leg_ik.diagnostics()
        diagnostics["thigh_projection"] = "projected_positions_two_link_ik"
    if progress is not None:
        progress(1.,'主动作完成')
    return SemanticRetargetResult(animation, pose_samples, diagnostics)


def validate_semantic_animation(
    animation: RetargetedAnimation,
    *,
    expected_max_frame: int,
) -> dict[str, Any]:
    """Validate invariants that define the reduced fire-stick skeleton."""

    if animation.max_frame != expected_max_frame:
        raise ValueError(
            f"Animation length mismatch: {animation.max_frame} != {expected_max_frame}"
        )
    limb_names = (
        "armF",
        "handF",
        "armB",
        "handB",
        "thighF",
        "legF",
        "thighB",
        "legB",
    )
    for name in limb_names:
        timeline = animation.bones.get(name)
        if timeline is None or not timeline.rotate:
            raise ValueError(f"Missing required rotation timeline for {name}")
        if timeline.translate:
            raise ValueError(f"{name} illegally contains translation keys")
        if timeline.scale:
            raise ValueError(f"{name} illegally contains scale keys")
    for name, timeline in animation.bones.items():
        for key in timeline.rotate:
            if not math.isfinite(key.value):
                raise ValueError(f"Non-finite rotation in {name}")
        for key in timeline.translate + timeline.scale:
            if not all(math.isfinite(value) for value in (key.x, key.y)):
                raise ValueError(f"Non-finite vector key in {name}")
    for name in ("body", "head"):
        for key in animation.bones[name].scale:
            if key.x not in {-1.0, 1.0} or key.y != 1.0:
                raise ValueError(f"{name} mirror scale must be exactly +/-1, 1")
            if key.curve != "stepped":
                raise ValueError(f"{name} mirror keys must be stepped")
    return {
        "fixed_limb_lengths": True,
        "limb_translation_keys": 0,
        "limb_scale_keys": 0,
        "body_mirror_keys": len(animation.bones["body"].scale),
        "head_mirror_keys": len(animation.bones["head"].scale),
        "duration_frames": animation.max_frame,
    }


def _head_projection_vector(
    value: SemanticSourceFrame, projector: Projector
) -> tuple[float, float]:
    primary = projector.direction(value.head_up)
    if math.hypot(*primary) >= 1e-5:
        return primary
    fallback = projector.direction(value.head_right)
    # A projected right axis is 90 degrees clockwise from the desired up axis.
    return (-fallback[1], fallback[0])


def _target_setup_angles(
    skeleton: SpineSkeleton, setup_world: dict[str, Matrix2D]
) -> dict[str, float]:
    bones = skeleton.bone_by_name
    result = {
        "body": _matrix_direction_angle(setup_world["body"], (0.0, 1.0)),
        "head": _matrix_direction_angle(setup_world["head"], (0.0, 1.0)),
    }
    for suffix in ("F", "B"):
        arm = f"arm{suffix}"
        hand = f"hand{suffix}"
        thigh = f"thigh{suffix}"
        leg = f"leg{suffix}"
        result[arm] = _matrix_direction_angle(
            setup_world[arm], (bones[hand].x, bones[hand].y)
        )
        result[hand] = _matrix_direction_angle(setup_world[hand], (1.0, 0.0))
        result[thigh] = _matrix_direction_angle(
            setup_world[thigh], (bones[leg].x, bones[leg].y)
        )
        result[leg] = _matrix_direction_angle(setup_world[leg], (1.0, 0.0))
    return result


def _stable_direction_angles(
    vectors: list[Vector3],
    projector: Projector,
    *,
    fallback_deg: float,
    minimum_ratio: float,
    maximum_step_deg: float,
) -> tuple[list[float], int]:
    raw: list[float | None] = []
    invalid = 0
    for vector in vectors:
        length = _length3(vector)
        projected = projector.direction(vector)
        ratio = math.hypot(*projected) / max(length, 1e-8)
        if (
            not all(math.isfinite(value) for value in (*vector, *projected))
            or length < 1e-8
            or ratio < max(0.0, minimum_ratio)
        ):
            raw.append(None)
            invalid += 1
        else:
            raw.append(_angle2(*projected))
    return _fill_and_stabilise_angles(raw, fallback_deg, maximum_step_deg), invalid


def _stable_depth_collapsed_angles(
    vectors: list[Vector3],
    projector: Projector,
    *,
    fallback_deg: float,
    side_deadzone_ratio: float,
    minimum_hold_frames: int,
    switch_vertical_cone_deg: float,
    screen_side_multiplier: float,
    maximum_step_deg: float,
) -> tuple[list[float], int]:
    """Reduce a 3D limb to 2D without losing its elevation to depth.

    The camera-horizontal X/Z magnitude becomes the 2D horizontal magnitude,
    while camera Y is preserved.  Only the screen side is discrete.  A small
    dead zone and hold time stop that side from flickering when the limb points
    almost exactly along the camera axis.
    """

    invalid = 0
    threshold = min(0.95, max(0.0, side_deadzone_ratio))
    screen_side_multiplier = -1.0 if screen_side_multiplier < 0.0 else 1.0
    switch_vertical_cone_deg = min(
        90.0, max(0.0, switch_vertical_cone_deg)
    )
    side = -1.0 if math.cos(math.radians(fallback_deg)) < 0.0 else 1.0

    # If a clip starts during an already raised limb pose, take its first
    # unambiguous side immediately instead of showing the fallback side for the
    # hold period.
    for vector in vectors:
        if not all(math.isfinite(value) for value in vector):
            continue
        camera = projector.camera_space(vector)
        horizontal = math.hypot(camera[0], camera[2])
        if horizontal < 1e-8:
            continue
        ratio = camera[0] / horizontal
        if abs(ratio) > threshold:
            camera_side = -1.0 if ratio < 0.0 else 1.0
            side = camera_side * screen_side_multiplier
            break

    pending_side = side
    pending_count = 0
    raw: list[float | None] = []
    required = max(1, int(minimum_hold_frames))
    for vector in vectors:
        length = _length3(vector)
        if not all(math.isfinite(value) for value in vector) or length < 1e-8:
            raw.append(None)
            invalid += 1
            continue

        camera_x, camera_y, camera_z = projector.camera_space(vector)
        horizontal = math.hypot(camera_x, camera_z)
        candidate = side
        if horizontal >= 1e-8:
            ratio = camera_x / horizontal
            if ratio > threshold:
                candidate = screen_side_multiplier
            elif ratio < -threshold:
                candidate = -screen_side_multiplier

        if candidate == side:
            pending_side = side
            pending_count = 0
        else:
            if candidate == pending_side:
                pending_count += 1
            else:
                pending_side = candidate
                pending_count = 1
            vertical_cone = math.degrees(
                math.atan2(horizontal, abs(camera_y))
            )
            if (
                pending_count >= required
                and vertical_cone <= switch_vertical_cone_deg
            ):
                side = candidate
                pending_side = side
                pending_count = 0

        raw.append(_angle2(side * horizontal, camera_y))

    return _fill_and_stabilise_angles(raw, fallback_deg, maximum_step_deg), invalid


def _knee_screen_bend_delta(
    flexion_deg: float,
    knee_bend_sign: float,
    body_scale_x: float,
) -> float:
    """Return the visible knee branch, reversed with the body mirror."""

    mirror_sign = -1.0 if body_scale_x < 0.0 else 1.0
    return knee_bend_sign * mirror_sign * max(0.0, flexion_deg)


def _stable_screen_angles(
    vectors: list[tuple[float, float]],
    *,
    fallback_deg: float,
    minimum_length: float,
    maximum_step_deg: float,
) -> tuple[list[float], int]:
    raw: list[float | None] = []
    invalid = 0
    for vector in vectors:
        if not all(math.isfinite(value) for value in vector) or math.hypot(*vector) < minimum_length:
            raw.append(None)
            invalid += 1
        else:
            raw.append(_angle2(*vector))
    return _fill_and_stabilise_angles(raw, fallback_deg, maximum_step_deg), invalid


def _fill_and_stabilise_angles(
    values: list[float | None], fallback: float, maximum_step: float
) -> list[float]:
    if not values:
        return []
    valid = [index for index, value in enumerate(values) if value is not None]
    if not valid:
        return [fallback for _ in values]
    result = [0.0 for _ in values]
    first = valid[0]
    first_value = float(values[first])
    for index in range(0, first + 1):
        result[index] = first_value
    previous_index = first
    previous_value = first_value
    for current_index in valid[1:]:
        current_value = _unwrap_near(float(values[current_index]), previous_value)
        span = current_index - previous_index
        for offset in range(1, span + 1):
            amount = offset / span
            result[previous_index + offset] = previous_value + (
                current_value - previous_value
            ) * amount
        previous_index = current_index
        previous_value = current_value
    for index in range(previous_index + 1, len(result)):
        result[index] = previous_value

    if maximum_step > 0.0:
        limited = [result[0]]
        for value in result[1:]:
            value = _unwrap_near(value, limited[-1])
            delta = value - limited[-1]
            delta = min(maximum_step, max(-maximum_step, delta))
            limited.append(limited[-1] + delta)
        result = limited
    return result


def _interval_states(
    headings: list[float],
    *,
    start_deg: float,
    span_deg: float,
    hysteresis_deg: float,
    minimum_hold_frames: int,
) -> list[bool]:
    if not headings:
        return []
    start_deg %= 360.0
    span_deg = min(359.0, max(1.0, span_deg))
    hysteresis_deg = min(
        max(0.0, hysteresis_deg), max(0.0, span_deg * 0.45)
    )

    def inside(angle: float, start: float, span: float) -> bool:
        return (angle - start) % 360.0 < span

    state = inside(headings[0], start_deg, span_deg)
    result = [state]
    last_change = 0
    for index, heading in enumerate(headings[1:], 1):
        if state:
            candidate = inside(
                heading,
                start_deg - hysteresis_deg,
                span_deg + 2.0 * hysteresis_deg,
            )
        else:
            candidate = inside(
                heading,
                start_deg + hysteresis_deg,
                max(1.0, span_deg - 2.0 * hysteresis_deg),
            )
        if candidate != state and index - last_change >= minimum_hold_frames:
            state = candidate
            last_change = index
        result.append(state)
    return result


def _threshold_states(
    values: list[float],
    *,
    threshold_deg: float,
    hysteresis_deg: float,
    minimum_hold_frames: int,
) -> list[bool]:
    if not values:
        return []
    state = values[0] > threshold_deg
    result = [state]
    last_change = 0
    for index, value in enumerate(values[1:], 1):
        candidate = (
            value > threshold_deg - hysteresis_deg
            if state
            else value > threshold_deg + hysteresis_deg
        )
        if candidate != state and index - last_change >= minimum_hold_frames:
            state = candidate
            last_change = index
        result.append(state)
    return result


def _state_switch_frames(states: list[bool]) -> set[int]:
    return {
        index
        for index in range(1, len(states))
        if states[index] != states[index - 1]
    }


def _state_scale_keys(states: list[bool]) -> list[VectorKey]:
    if not states:
        return []
    result = [VectorKey(0, -1.0 if states[0] else 1.0, 1.0, "stepped")]
    for frame in sorted(_state_switch_frames(states)):
        result.append(
            VectorKey(frame, -1.0 if states[frame] else 1.0, 1.0, "stepped")
        )
    return result


def _unwrap_segmented(keys: list[ScalarKey], reset_frames: set[int]) -> list[ScalarKey]:
    result: list[ScalarKey] = []
    previous: float | None = None
    for key in keys:
        if key.frame in reset_frames:
            previous = None
        value = key.value if previous is None else _unwrap_near(key.value, previous)
        result.append(ScalarKey(key.frame, value, key.curve))
        previous = value
    return result


def _mark_stepped_before(
    keys: list[ScalarKey], switch_frames: set[int]
) -> list[ScalarKey]:
    result = keys[:]
    for index in range(len(result) - 1):
        if result[index + 1].frame in switch_frames:
            key = result[index]
            result[index] = ScalarKey(key.frame, key.value, "stepped")
    return result


def _solve_local_rotation(
    parent_world: Matrix2D,
    setup: SpineBone,
    reference_vector: tuple[float, float],
    desired_world_angle_deg: float,
    *,
    scale_x_multiplier: float = 1.0,
    scale_y_multiplier: float = 1.0,
) -> float:
    radians = math.radians(desired_world_angle_deg)
    local_direction = _inverse_direction(
        parent_world, (math.cos(radians), math.sin(radians))
    )
    scaled_reference = (
        reference_vector[0] * setup.scale_x * scale_x_multiplier,
        reference_vector[1] * setup.scale_y * scale_y_multiplier,
    )
    return _wrap_degrees(
        _angle2(*local_direction)
        - _angle2(*scaled_reference)
        - setup.rotation
    )


def _animated_local_matrix(
    setup: SpineBone,
    rotation_value: float,
    translation: tuple[float, float] = (0.0, 0.0),
    scale: tuple[float, float] = (1.0, 1.0),
) -> Matrix2D:
    return _local_matrix(
        setup.x + translation[0],
        setup.y + translation[1],
        setup.rotation + rotation_value,
        setup.scale_x * scale[0],
        setup.scale_y * scale[1],
    )


def _setup_world_matrices(skeleton: SpineSkeleton) -> dict[str, Matrix2D]:
    result: dict[str, Matrix2D] = {}
    for bone in skeleton.bones:
        local = _local_matrix(
            bone.x,
            bone.y,
            bone.rotation,
            bone.scale_x,
            bone.scale_y,
        )
        result[bone.name] = (
            local
            if bone.parent is None
            else _multiply_matrices(result[bone.parent], local)
        )
    return result


def _local_matrix(
    x: float,
    y: float,
    rotation_deg: float,
    scale_x: float,
    scale_y: float,
) -> Matrix2D:
    radians = math.radians(rotation_deg)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    return (
        cosine * scale_x,
        -sine * scale_y,
        sine * scale_x,
        cosine * scale_y,
        x,
        y,
    )


def _multiply_matrices(parent: Matrix2D, local: Matrix2D) -> Matrix2D:
    pa, pb, pc, pd, px, py = parent
    la, lb, lc, ld, lx, ly = local
    return (
        pa * la + pb * lc,
        pa * lb + pb * ld,
        pc * la + pd * lc,
        pc * lb + pd * ld,
        pa * lx + pb * ly + px,
        pc * lx + pd * ly + py,
    )


def _inverse_direction(
    matrix: Matrix2D, vector: tuple[float, float]
) -> tuple[float, float]:
    a, b, c, d, _, _ = matrix
    determinant = a * d - b * c
    if abs(determinant) < 1e-10:
        return vector
    return (
        (d * vector[0] - b * vector[1]) / determinant,
        (-c * vector[0] + a * vector[1]) / determinant,
    )


def _matrix_direction_angle(
    matrix: Matrix2D, vector: tuple[float, float]
) -> float:
    a, b, c, d, _, _ = matrix
    return _angle2(a * vector[0] + b * vector[1], c * vector[0] + d * vector[1])


def _heading_deg(rotation: Quaternion) -> float:
    forward = rotate_vector(rotation, (0.0, 0.0, -1.0))
    if math.hypot(forward[0], forward[2]) < 1e-8:
        return 0.0
    # Top view: clockwise from the MMD default forward direction is positive.
    return math.degrees(math.atan2(forward[0], -forward[2])) % 360.0


def _slerp(first: Quaternion, second: Quaternion, amount: float) -> Quaternion:
    dot = sum(a * b for a, b in zip(first, second))
    if dot < 0.0:
        second = tuple(-value for value in second)  # type: ignore[assignment]
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        return _normalise_quaternion(
            tuple(a + amount * (b - a) for a, b in zip(first, second))  # type: ignore[arg-type]
        )
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    if abs(sin_theta) < 1e-8:
        return first
    first_weight = math.sin((1.0 - amount) * theta) / sin_theta
    second_weight = math.sin(amount * theta) / sin_theta
    return _normalise_quaternion(
        tuple(
            first_weight * a + second_weight * b
            for a, b in zip(first, second)
        )  # type: ignore[arg-type]
    )


def _qmul(first: Quaternion, second: Quaternion) -> Quaternion:
    ax, ay, az, aw = first
    bx, by, bz, bw = second
    return _normalise_quaternion(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )
    )


def _normalise_quaternion(value: Quaternion) -> Quaternion:
    length = math.sqrt(sum(component * component for component in value))
    if not math.isfinite(length) or length < 1e-8:
        return IDENTITY_Q
    return tuple(component / length for component in value)  # type: ignore[return-value]


def _vector3(value: Any) -> Vector3:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"Expected a 3D vector, received {value!r}")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"Non-finite 3D vector: {value!r}")
    return result  # type: ignore[return-value]


def _add3(*values: Vector3) -> Vector3:
    return tuple(sum(value[index] for value in values) for index in range(3))  # type: ignore[return-value]


def _subtract3(first: Vector3, second: Vector3) -> Vector3:
    return tuple(a - b for a, b in zip(first, second))  # type: ignore[return-value]


def _scale3(value: Vector3, amount: float) -> Vector3:
    return tuple(component * amount for component in value)  # type: ignore[return-value]


def _dot3(first: Vector3, second: Vector3) -> float:
    return sum(a * b for a, b in zip(first, second))


def _length3(value: Vector3) -> float:
    return math.sqrt(_dot3(value, value))


def _normalise3(value: Vector3) -> Vector3:
    length = _length3(value)
    if not math.isfinite(length) or length < 1e-8:
        return ZERO_3
    return _scale3(value, 1.0 / length)


def _angle2(x: float, y: float) -> float:
    if math.hypot(x, y) < 1e-12:
        return 0.0
    return math.degrees(math.atan2(y, x))


def _wrap_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def _unwrap_near(value: float, reference: float) -> float:
    while value - reference > 180.0:
        value -= 360.0
    while value - reference < -180.0:
        value += 360.0
    return value
