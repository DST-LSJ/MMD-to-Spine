"""Retarget standard VMD bone tracks to a reduced Spine Chibi skeleton."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, TYPE_CHECKING

from .projection import Projector, REST_AXES, rotate_vector
from .spine_reader import SpineSkeleton
from .vmd_reader import VMDMotion, sample_track

if TYPE_CHECKING:
    from .source_pose import CanonicalPoseSolver


class MappingError(ValueError):
    """Raised when a bone mapping is malformed or incompatible."""


_LegMatrix2D = tuple[float, float, float, float, float, float]


@dataclass(frozen=True, slots=True)
class ScalarKey:
    frame: int
    value: float
    curve: str | None = None


@dataclass(frozen=True, slots=True)
class VectorKey:
    frame: int
    x: float
    y: float
    curve: str | None = None


@dataclass(slots=True)
class BoneAnimation:
    rotate: list[ScalarKey] = field(default_factory=list)
    translate: list[VectorKey] = field(default_factory=list)
    scale: list[VectorKey] = field(default_factory=list)


@dataclass(slots=True)
class RetargetedAnimation:
    fps: float
    max_frame: int
    bones: dict[str, BoneAnimation]
    source_bones_used: list[str]
    source_bones_unmapped: list[str]
    source_start_frame: int = 0
    draw_order_events: list[dict[str, Any]] = field(default_factory=list)
    draw_order_diagnostics: dict[str, Any] = field(default_factory=dict)
    arm_pose_diagnostics: dict[str, Any] = field(default_factory=dict)
    # Transient conversion state used by the optional secondary-motion pass.
    # These fields are never serialized into Spine JSON.  Keeping the untouched
    # target timelines here makes the postprocessor idempotent when it is called
    # repeatedly on the same in-memory animation.
    secondary_motion_base_rotations: dict[str, list[ScalarKey]] = field(
        default_factory=dict
    )
    secondary_motion_diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.max_frame / self.fps


def load_mapping(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MappingError(f"Could not read mapping {path}: {error}") from error
    if not isinstance(value, dict) or not isinstance(value.get("targets"), dict):
        raise MappingError("Bone mapping must contain a targets object")
    return value


def validate_mapping(mapping: dict[str, Any], skeleton: SpineSkeleton) -> None:
    target_names = set(skeleton.bone_by_name)
    for target, rule in mapping["targets"].items():
        if target not in target_names:
            raise MappingError(f"Mapped Spine bone does not exist in template: {target}")
        if not isinstance(rule, dict) or not isinstance(rule.get("sources"), list):
            raise MappingError(f"Mapping for {target!r} must contain a sources array")
        if rule.get("axis", "up") not in REST_AXES:
            raise MappingError(f"Unknown rest axis for {target!r}: {rule.get('axis')}")
        for source in rule["sources"]:
            if not isinstance(source, dict) or not isinstance(source.get("bone"), str):
                raise MappingError(f"Invalid source entry in mapping for {target!r}")


def retarget_motion(
    motion: VMDMotion,
    skeleton: SpineSkeleton,
    mapping: dict[str, Any],
    projector: Projector,
    fps: float = 30.0,
    start_frame: int = 0,
    max_frame: int | None = None,
) -> RetargetedAnimation:
    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    validate_mapping(mapping, skeleton)
    source_start_frame = max(0, start_frame)
    source_end_frame = motion.max_frame if max_frame is None else max(0, max_frame)
    if source_end_frame < source_start_frame:
        raise ValueError("The conversion end frame is before its start frame")
    if motion.max_frame < source_start_frame:
        raise ValueError(
            f"Start frame {source_start_frame} is after the last available bone frame "
            f"({motion.max_frame})"
        )
    output_end_frame = source_end_frame - source_start_frame
    output: dict[str, BoneAnimation] = {}
    used: set[str] = set()

    for target, rule in mapping["targets"].items():
        active_sources = [
            source for source in rule["sources"] if source["bone"] in motion.bone_tracks
        ]
        if not active_sources:
            continue
        used.update(source["bone"] for source in active_sources)
        # Quaternion interpolation followed by 3D-to-2D projection is not
        # linear between VMD keyframes. Sampling every source frame prevents
        # Spine from taking a different (and sometimes opposite) 2D arc.
        ordered_frames = range(source_start_frame, source_end_frame + 1)
        axis = REST_AXES[rule.get("axis", "up")]
        rotation_gain = float(rule.get("rotation_gain", 1.0))
        translation_gain = float(rule.get("translation_gain", 1.0))
        rotation_limit = abs(float(rule.get("rotation_limit", 180.0)))
        max_rotation_step = abs(float(rule.get("max_rotation_step_deg", 0.0)))
        has_rotation = any(float(source.get("rotate", 0.0)) != 0 for source in active_sources)
        has_translation = any(
            float(source.get("translate", 0.0)) != 0 for source in active_sources
        )
        animation = BoneAnimation()

        for frame_number in ordered_frames:
            output_frame = frame_number - source_start_frame
            rotation = 0.0
            position = [0.0, 0.0, 0.0]
            for source in active_sources:
                key = sample_track(motion.bone_tracks[source["bone"]], frame_number)
                rotation += float(source.get("rotate", 0.0)) * projector.rotation_delta(
                    key.rotation, axis
                )
                translation_weight = float(source.get("translate", 0.0))
                for index in range(3):
                    position[index] += key.position[index] * translation_weight

            if has_rotation:
                animation.rotate.append(ScalarKey(output_frame, rotation * rotation_gain))
            if has_translation:
                x, y = projector.position(tuple(position))  # type: ignore[arg-type]
                animation.translate.append(
                    VectorKey(output_frame, x * translation_gain, y * translation_gain)
                )

        if animation.rotate:
            animation.rotate = _unwrap_rotation(animation.rotate, rotation_limit)
            if max_rotation_step > 0:
                animation.rotate = _limit_rotation_step(
                    animation.rotate, max_rotation_step
                )
        output[target] = animation

    all_source_names = set(motion.bone_tracks)
    return RetargetedAnimation(
        fps=fps,
        max_frame=output_end_frame,
        bones=output,
        source_bones_used=sorted(used),
        source_bones_unmapped=sorted(all_source_names - used),
        source_start_frame=source_start_frame,
    )


def _unwrap_rotation(keys: list[ScalarKey], limit: float = 180.0) -> list[ScalarKey]:
    if not keys:
        return []
    first_raw = keys[0].value if math.isfinite(keys[0].value) else 0.0
    first_value = min(limit, max(-limit, first_raw))
    result = [ScalarKey(keys[0].frame, first_value, keys[0].curve)]
    # Continue from the value actually emitted to Spine. Previously this kept
    # an unbounded hidden value while only clamping the stored key. One wrap at
    # +/-180 degrees could therefore leave a hand pinned to its limit forever.
    previous = first_value
    for key in keys[1:]:
        value = key.value if math.isfinite(key.value) else previous
        while value - previous > 180.0:
            value -= 360.0
        while value - previous < -180.0:
            value += 360.0
        emitted = min(limit, max(-limit, value))
        result.append(ScalarKey(key.frame, emitted, key.curve))
        previous = emitted
    return result


def _unwrap_rotation_segmented(
    keys: list[ScalarKey],
    limit: float,
    reset_frames: set[int],
) -> list[ScalarKey]:
    """Unwrap rotations without crossing a reflected parent-scale boundary."""

    if not keys or not reset_frames:
        return _unwrap_rotation(keys, limit)
    result: list[ScalarKey] = []
    segment: list[ScalarKey] = []
    for key in keys:
        if segment and key.frame in reset_frames:
            result.extend(_unwrap_rotation(segment, limit))
            segment = []
        segment.append(key)
    if segment:
        result.extend(_unwrap_rotation(segment, limit))
    return result


def _mirror_scale_switch_frames(
    animation: RetargetedAnimation,
    bone_name: str,
) -> set[int]:
    timeline = animation.bones.get(bone_name)
    if timeline is None or len(timeline.scale) < 2:
        return set()
    result: set[int] = set()
    previous_sign = -1 if timeline.scale[0].x < 0.0 else 1
    for key in timeline.scale[1:]:
        sign = -1 if key.x < 0.0 else 1
        if sign != previous_sign:
            result.add(key.frame)
            previous_sign = sign
    return result


def _step_before_frames(
    keys: list[ScalarKey],
    switch_frames: set[int],
) -> list[ScalarKey]:
    """Hold the old local solution until a stepped mirror switch occurs."""

    if len(keys) < 2 or not switch_frames:
        return keys
    result = keys[:]
    for index in range(len(result) - 1):
        if result[index + 1].frame in switch_frames:
            key = result[index]
            result[index] = ScalarKey(key.frame, key.value, "stepped")
    return result


def _limit_rotation_step(
    keys: list[ScalarKey], max_step_per_frame: float
) -> list[ScalarKey]:
    """Slew-limit projected rotations at camera-facing singularities.

    This operates on the dense, one-key-per-source-frame timeline. Normal VMD
    motion is untouched; only an implausibly large single-frame projection jump
    is spread over subsequent frames.
    """

    if not keys or max_step_per_frame <= 0:
        return keys[:]
    result = [keys[0]]
    previous = keys[0].value
    previous_frame = keys[0].frame
    for key in keys[1:]:
        frame_span = max(1, key.frame - previous_frame)
        maximum_delta = max_step_per_frame * frame_span
        delta = key.value - previous
        if delta > maximum_delta:
            value = previous + maximum_delta
        elif delta < -maximum_delta:
            value = previous - maximum_delta
        else:
            value = key.value
        result.append(ScalarKey(key.frame, value, key.curve))
        previous = value
        previous_frame = key.frame
    return result


def apply_leg_ik(
    animation: RetargetedAnimation,
    motion: VMDMotion,
    skeleton: SpineSkeleton,
    projector: Projector,
    config: dict[str, Any],
    source_rig_profile: dict[str, Any] | None = None,
    pose_solver: CanonicalPoseSolver | None = None,
) -> RetargetedAnimation:
    """Retarget each projected 3D leg segment to its Spine counterpart.

    When a :class:`CanonicalPoseSolver` is supplied, the primary path projects
    ``hip -> knee`` and ``knee -> ankle`` independently.  Their directions are
    converted through the actual animated Spine parent matrices, so a segment
    becoming short in camera space changes its scale instead of being mistaken
    for a bent two-bone chain.

    The former foot-endpoint 2D IK remains as a safe fallback for callers that
    do not yet provide a pose solver, or for a frame whose proxy joints are
    missing or non-finite.
    """

    if not bool(config.get("enabled", False)):
        return animation
    configured_strength = min(
        1.0, max(0.0, float(config.get("strength", 0.85)))
    )
    # Most dance VMDs store the visible leg motion in the foot IK tracks while
    # the direct thigh/knee tracks are static or nearly empty.  Blending those
    # zero-valued FK tracks back into the result attenuates every kick, so an
    # explicitly IK-dominant conversion uses the solved result without that
    # residual blend.
    strength = 1.0 if bool(config.get("ik_dominant", False)) else configured_strength
    position_gain = float(config.get("position_gain", 1.0))
    normalise_leg_length = bool(config.get("normalise_source_leg_length", True))
    thigh_translation_gain = float(config.get("thigh_translation_gain", 0.045))
    thigh_translation_limit = abs(float(config.get("thigh_translation_limit_px", 8.0)))
    bone_scaling_enabled = bool(config.get("bone_scaling_enabled", False))
    thigh_depth_scale_gain = abs(float(config.get("thigh_depth_scale_gain", 0.012)))
    thigh_scale_min = float(config.get("thigh_scale_min", 0.92))
    minimum_knee_bend = abs(float(config.get("minimum_knee_bend_deg", 1.0)))
    projected_scale_min = max(
        1e-3, abs(float(config.get("projected_segment_scale_min", 0.1)))
    )
    projected_scale_max = max(
        projected_scale_min,
        abs(float(config.get("projected_segment_scale_max", 2.0))),
    )
    # The child's local scale must be able to divide out every allowed parent
    # scale.  For segment limits [0.1, 2.0], the mathematically complete local
    # range is [0.05, 20.0]; a narrower cap would corrupt the lower segment's
    # requested world-space length at strong foreshortening.
    required_local_scale_max = projected_scale_max / projected_scale_min
    projected_local_scale_max = max(
        required_local_scale_max,
        abs(float(config.get("projected_local_scale_max", required_local_scale_max))),
    )
    projected_direction_min_ratio = max(
        0.0,
        abs(float(config.get("projected_direction_min_ratio", 0.02))),
    )
    knee_bend_constraint_enabled = bool(
        config.get("knee_bend_constraint_enabled", True)
    )
    screen_bend_direction = str(
        config.get("screen_knee_bend_direction", "clockwise")
    ).strip().lower()
    if screen_bend_direction in {"clockwise", "cw"}:
        screen_knee_bend_sign = -1.0
    elif screen_bend_direction in {"counterclockwise", "ccw"}:
        screen_knee_bend_sign = 1.0
    else:
        raise ValueError(
            "screen_knee_bend_direction must be 'clockwise' or "
            "'counterclockwise'"
        )
    max_screen_knee_bend = min(
        179.0,
        max(0.0, abs(float(config.get("max_screen_knee_bend_deg", 140.0)))),
    )
    body_names = [
        name
        for name in config.get("body_sources", ["センター", "グルーブ"])
        if name in motion.bone_tracks
    ]
    chains = (
        (
            "thighF",
            "legF",
            str(config.get("front_source", config.get("right_source", "右足ＩＫ"))),
            str(
                config.get(
                    "front_parent_source",
                    config.get("right_parent_source", "右足IK親"),
                )
            ),
            "F",
        ),
        (
            "thighB",
            "legB",
            str(config.get("back_source", config.get("left_source", "左足ＩＫ"))),
            str(
                config.get(
                    "back_parent_source",
                    config.get("left_parent_source", "左足IK親"),
                )
            ),
            "B",
        ),
    )
    bend_signs = config.get("bend_signs", {})
    bone_lookup = skeleton.bone_by_name
    pose_profile = source_rig_profile
    if pose_profile is None and pose_solver is not None:
        candidate = getattr(pose_solver, "profile", None)
        if isinstance(candidate, dict):
            pose_profile = candidate
    setup_world = _leg_setup_world_matrices(skeleton) if pose_solver is not None else {}
    pose_cache: dict[int, Any | None] = {}

    def pose_at(source_frame: int, output_frame: int) -> Any | None:
        if pose_solver is None:
            return None
        if source_frame not in pose_cache:
            try:
                pose_cache[source_frame] = pose_solver.sample(
                    source_frame, output_frame, animation.fps
                )
            except (ArithmeticError, KeyError, TypeError, ValueError):
                pose_cache[source_frame] = None
        return pose_cache[source_frame]

    for thigh_name, leg_name, ik_name, ik_parent_name, profile_suffix in chains:
        if (
            thigh_name not in bone_lookup
            or leg_name not in bone_lookup
            or (pose_solver is None and ik_name not in motion.bone_tracks)
        ):
            continue
        thigh_setup = bone_lookup[thigh_name]
        leg_setup = bone_lookup[leg_name]
        upper_length = math.hypot(leg_setup.x, leg_setup.y)
        lower_length = leg_setup.length
        if upper_length < 1e-5 or lower_length < 1e-5:
            continue

        target_chain_length = upper_length + lower_length
        source_chain_length = _source_leg_length(
            config,
            source_rig_profile,
            thigh_name,
            profile_suffix,
        )
        length_normalisation = 1.0
        projected_source_length = source_chain_length * abs(projector.config.scale)
        if (
            normalise_leg_length
            and target_chain_length > 1e-5
            and projected_source_length > 1e-5
        ):
            length_normalisation = target_chain_length / projected_source_length
        chain_position_gain = position_gain * length_normalisation

        thigh_setup_radians = math.radians(thigh_setup.rotation)
        upper_rest_angle = math.atan2(leg_setup.y, leg_setup.x) + thigh_setup_radians
        lower_rest_angle = math.radians(thigh_setup.rotation + leg_setup.rotation)
        rest_joint_angle = _normalise_radians(lower_rest_angle - upper_rest_angle)
        rest_joint_degrees = math.degrees(rest_joint_angle)
        setup_bend_sign = 1.0 if rest_joint_angle >= 0 else -1.0
        configured_bend_sign = float(bend_signs.get(thigh_name, setup_bend_sign))
        bend_sign = 1.0 if configured_bend_sign >= 0 else -1.0
        # The fallback endpoint IK must select the setup chain's own branch so
        # zero input returns to the authored pose. The common screen-space bend
        # direction is imposed later when the lower target is constructed.
        solver_bend_sign = (
            setup_bend_sign if knee_bend_constraint_enabled else bend_sign
        )
        rest_foot_x = (
            math.cos(upper_rest_angle) * upper_length
            + math.cos(lower_rest_angle) * lower_length
        )
        rest_foot_y = (
            math.sin(upper_rest_angle) * upper_length
            + math.sin(lower_rest_angle) * lower_length
        )

        pose_calibration: dict[str, float] | None = None
        geometry = (
            pose_profile.get("geometry", {})
            if isinstance(pose_profile, dict)
            else {}
        )
        source_upper_reference = _leg_vector3(
            geometry.get(f"thigh_{profile_suffix}")
        )
        source_lower_reference = _leg_vector3(
            geometry.get(f"shin_{profile_suffix}")
        )
        source_hip_reference = _leg_vector3(
            geometry.get(f"hip_{profile_suffix}_offset")
        )
        if (
            pose_solver is not None
            and source_upper_reference is not None
            and source_lower_reference is not None
            and source_hip_reference is not None
            and thigh_name in setup_world
            and leg_name in setup_world
        ):
            source_upper_2d = projector.direction(source_upper_reference)
            source_lower_2d = projector.direction(source_lower_reference)
            source_hip_2d = projector.direction(source_hip_reference)
            source_upper_length_2d = math.hypot(*source_upper_2d)
            source_lower_length_2d = math.hypot(*source_lower_2d)
            thigh_setup_world = setup_world[thigh_name]
            leg_setup_world = setup_world[leg_name]
            target_upper_2d = (
                leg_setup_world[4] - thigh_setup_world[4],
                leg_setup_world[5] - thigh_setup_world[5],
            )
            target_upper_length_2d = math.hypot(*target_upper_2d)
            target_lower_2d = (leg_setup_world[0], leg_setup_world[2])
            target_lower_length_2d = math.hypot(*target_lower_2d)
            if min(
                source_upper_length_2d,
                source_lower_length_2d,
                target_upper_length_2d,
                target_lower_length_2d,
            ) > 1e-7:
                source_upper_angle = _leg_angle(*source_upper_2d)
                source_lower_angle = _leg_angle(*source_lower_2d)
                target_upper_angle = _leg_angle(*target_upper_2d)
                target_lower_angle = _leg_angle(*target_lower_2d)
                pose_calibration = {
                    "upper_source_angle": source_upper_angle,
                    "lower_source_angle": source_lower_angle,
                    "source_setup_joint_angle": _normalise_degrees(
                        source_lower_angle - source_upper_angle
                    ),
                    "upper_source_length": source_upper_length_2d,
                    "lower_source_length": source_lower_length_2d,
                    "upper_target_angle": target_upper_angle,
                    # The authored Spine setup pose is the maximum extension
                    # boundary requested by the user. Keep each chain's own
                    # small setup delta; only the additional bend direction is
                    # shared by the two knees.
                    "target_setup_joint_angle": _normalise_degrees(
                        target_lower_angle - target_upper_angle
                    ),
                    "hip_source_x": source_hip_2d[0],
                    "hip_source_y": source_hip_2d[1],
                }

        # A mirrored parent may require substantially more local leg rotation
        # to produce the same constrained screen-space bend. Derive a safe
        # per-chain limit instead of truncating every leg at the old 140°.
        leg_local_rotation_limit = 140.0
        if knee_bend_constraint_enabled:
            setup_delta_for_limit = (
                pose_calibration["target_setup_joint_angle"]
                if pose_calibration is not None
                else rest_joint_degrees
            )
            leg_local_rotation_limit = min(
                179.0,
                max(
                    140.0,
                    max_screen_knee_bend
                    + 2.0 * abs(setup_delta_for_limit)
                    + 1.0,
                ),
            )

        existing_thigh = animation.bones.setdefault(thigh_name, BoneAnimation()).rotate
        existing_leg = animation.bones.setdefault(leg_name, BoneAnimation()).rotate
        source_start_frame = animation.source_start_frame
        source_end_frame = source_start_frame + animation.max_frame
        # The projected IK target follows curved VMD interpolation. Dense
        # sampling preserves the configured knee side between source keys.
        frames = range(source_start_frame, source_end_frame + 1)

        thigh_keys: list[ScalarKey] = []
        leg_keys: list[ScalarKey] = []
        thigh_translate_keys: list[VectorKey] = []
        thigh_scale_keys: list[VectorKey] = []
        leg_scale_keys: list[VectorKey] = []
        last_upper_projected_angle = (
            pose_calibration["upper_source_angle"]
            if pose_calibration is not None
            else 0.0
        )
        last_lower_projected_angle = (
            pose_calibration["lower_source_angle"]
            if pose_calibration is not None
            else 0.0
        )
        last_knee_bend_magnitude = 0.0
        knee_preroll_frames = max(
            0, int(config.get("knee_direction_preroll_frames", 120))
        )
        if pose_calibration is not None and source_start_frame > 0:
            earliest_preroll = max(
                0, source_start_frame - knee_preroll_frames
            )
            for preroll_frame in range(
                source_start_frame - 1, earliest_preroll - 1, -1
            ):
                preroll_pose = pose_at(preroll_frame, 0)
                if preroll_pose is None:
                    continue
                preroll_joints = getattr(preroll_pose, "joints", {})
                preroll_hip = _leg_vector3(
                    preroll_joints.get(f"hip_{profile_suffix}")
                )
                preroll_knee = _leg_vector3(
                    preroll_joints.get(f"knee_{profile_suffix}")
                )
                preroll_ankle = _leg_vector3(
                    preroll_joints.get(f"ankle_{profile_suffix}")
                )
                if (
                    preroll_hip is None
                    or preroll_knee is None
                    or preroll_ankle is None
                ):
                    continue
                preroll_upper = projector.direction(
                    tuple(
                        preroll_knee[index] - preroll_hip[index]
                        for index in range(3)
                    )
                )
                preroll_lower = projector.direction(
                    tuple(
                        preroll_ankle[index] - preroll_knee[index]
                        for index in range(3)
                    )
                )
                upper_threshold = (
                    pose_calibration["upper_source_length"]
                    * projected_direction_min_ratio
                )
                lower_threshold = (
                    pose_calibration["lower_source_length"]
                    * projected_direction_min_ratio
                )
                if (
                    all(math.isfinite(value) for value in preroll_upper)
                    and all(math.isfinite(value) for value in preroll_lower)
                    and math.hypot(*preroll_upper)
                    > max(1e-12, upper_threshold)
                    and math.hypot(*preroll_lower)
                    > max(1e-12, lower_threshold)
                ):
                    last_upper_projected_angle = _unwrap_degrees_near(
                        _leg_angle(*preroll_upper),
                        last_upper_projected_angle,
                    )
                    last_lower_projected_angle = _unwrap_degrees_near(
                        _leg_angle(*preroll_lower),
                        last_lower_projected_angle,
                    )
                    preroll_joint_angle = _normalise_degrees(
                        last_lower_projected_angle
                        - last_upper_projected_angle
                    )
                    last_knee_bend_magnitude = min(
                        max_screen_knee_bend,
                        abs(
                            _normalise_degrees(
                                preroll_joint_angle
                                - pose_calibration[
                                    "source_setup_joint_angle"
                                ]
                            )
                        ),
                    )
                    break
        for frame_number in frames:
            output_frame = frame_number - source_start_frame
            legacy_target_available = ik_name in motion.bone_tracks
            relative = (0.0, 0.0, 0.0)
            if legacy_target_available:
                ik_sample = sample_track(
                    motion.bone_tracks[ik_name], frame_number
                )
                ik_position = ik_sample.position
                # Semi-standard MMD rigs commonly animate 足IK親 rather than
                # (or in addition to) the child 足ＩＫ. Its rotation applies to
                # the child's animated offset; exact bind-pivot rotation still
                # requires PMX, but translation-only parent tracks are exact.
                if ik_parent_name and ik_parent_name in motion.bone_tracks:
                    parent_sample = sample_track(
                        motion.bone_tracks[ik_parent_name], frame_number
                    )
                    rotated_child = rotate_vector(
                        parent_sample.rotation, ik_position
                    )
                    ik_position = tuple(
                        parent_sample.position[index] + rotated_child[index]
                        for index in range(3)
                    )
                body_position = [0.0, 0.0, 0.0]
                for body_name in body_names:
                    sampled = sample_track(
                        motion.bone_tracks[body_name], frame_number
                    ).position
                    for index in range(3):
                        body_position[index] += sampled[index]
                relative = tuple(
                    (ik_position[index] - body_position[index])
                    * chain_position_gain
                    for index in range(3)
                )
            delta_x, delta_y = projector.position(relative)  # type: ignore[arg-type]
            camera_depth = projector.camera_space(relative)[2]
            legacy_translate_x = min(
                thigh_translation_limit,
                max(-thigh_translation_limit, delta_x * thigh_translation_gain),
            )
            legacy_translate_y = min(
                thigh_translation_limit,
                max(-thigh_translation_limit, delta_y * thigh_translation_gain),
            )
            translate_x = legacy_translate_x
            translate_y = legacy_translate_y

            pose = pose_at(frame_number, output_frame)
            projected_pose = False
            if pose is not None and pose_calibration is not None:
                joints = getattr(pose, "joints", {})
                hip = _leg_vector3(joints.get(f"hip_{profile_suffix}"))
                knee = _leg_vector3(joints.get(f"knee_{profile_suffix}"))
                ankle = _leg_vector3(joints.get(f"ankle_{profile_suffix}"))
                pelvis = _leg_vector3(joints.get("pelvis"))
                if (
                    hip is not None
                    and knee is not None
                    and ankle is not None
                    and pelvis is not None
                ):
                    upper_source_3d = tuple(
                        knee[index] - hip[index] for index in range(3)
                    )
                    lower_source_3d = tuple(
                        ankle[index] - knee[index] for index in range(3)
                    )
                    upper_source_2d = projector.direction(upper_source_3d)  # type: ignore[arg-type]
                    lower_source_2d = projector.direction(lower_source_3d)  # type: ignore[arg-type]
                    upper_projection_finite = all(
                        math.isfinite(value) for value in upper_source_2d
                    )
                    lower_projection_finite = all(
                        math.isfinite(value) for value in lower_source_2d
                    )
                    upper_projected_length = (
                        math.hypot(*upper_source_2d)
                        if upper_projection_finite
                        else 0.0
                    )
                    lower_projected_length = (
                        math.hypot(*lower_source_2d)
                        if lower_projection_finite
                        else 0.0
                    )
                    upper_direction_threshold = (
                        pose_calibration["upper_source_length"]
                        * projected_direction_min_ratio
                    )
                    lower_direction_threshold = (
                        pose_calibration["lower_source_length"]
                        * projected_direction_min_ratio
                    )
                    upper_direction_stable = (
                        upper_projection_finite
                        and upper_projected_length
                        > max(1e-12, upper_direction_threshold)
                    )
                    lower_direction_stable = (
                        lower_projection_finite
                        and lower_projected_length
                        > max(1e-12, lower_direction_threshold)
                    )
                    if upper_direction_stable:
                        last_upper_projected_angle = _unwrap_degrees_near(
                            _leg_angle(*upper_source_2d),
                            last_upper_projected_angle,
                        )
                    if lower_direction_stable:
                        last_lower_projected_angle = _unwrap_degrees_near(
                            _leg_angle(*lower_source_2d),
                            last_lower_projected_angle,
                        )
                    if upper_direction_stable and lower_direction_stable:
                        source_joint_angle = _normalise_degrees(
                            last_lower_projected_angle
                            - last_upper_projected_angle
                        )
                        source_bend_delta = _normalise_degrees(
                            source_joint_angle
                            - pose_calibration["source_setup_joint_angle"]
                        )
                        last_knee_bend_magnitude = min(
                            max_screen_knee_bend,
                            abs(source_bend_delta),
                        )

                    # A segment parallel to the camera axis has no stable 2D
                    # direction. Preserve its last meaningful direction and
                    # let only its projected scale approach the configured
                    # floor. Falling back to foot-endpoint 2D IK here would
                    # turn pure depth foreshortening into a false knee bend.
                    # Joints exist, so remain on the segment-projection path
                    # even if both screen vectors are numerically degenerate.
                    # The stable angles and scale floors above define that
                    # singular frame without reintroducing endpoint IK.
                    if pose is not None:
                        desired_upper_world = _normalise_degrees(
                            last_upper_projected_angle
                            + pose_calibration["upper_target_angle"]
                            - pose_calibration["upper_source_angle"]
                        )
                        if knee_bend_constraint_enabled:
                            # Both knees share one screen-space bend side.
                            # Magnitude comes from the projected MMD pose, but
                            # its source sign is discarded so an IK branch flip
                            # can never create a reverse joint. At zero bend the
                            # authored setup delta is restored exactly.
                            desired_lower_world = _normalise_degrees(
                                desired_upper_world
                                + pose_calibration["target_setup_joint_angle"]
                                + screen_knee_bend_sign
                                * last_knee_bend_magnitude
                            )
                        else:
                            desired_lower_world = _normalise_degrees(
                                last_lower_projected_angle
                                + pose_calibration["target_setup_joint_angle"]
                                + pose_calibration["upper_target_angle"]
                                - pose_calibration["lower_source_angle"]
                            )
                        upper_scale = min(
                            projected_scale_max,
                            max(
                                projected_scale_min,
                                upper_projected_length
                                / pose_calibration["upper_source_length"],
                            ),
                        )
                        lower_world_scale = min(
                            projected_scale_max,
                            max(
                                projected_scale_min,
                                lower_projected_length
                                / pose_calibration["lower_source_length"],
                            ),
                        )
                        # leg inherits thigh's uniform scale. Divide it out so
                        # the final world-space lower segment receives only its
                        # own projected-length ratio, not the product twice.
                        lower_local_scale = lower_world_scale / max(
                            upper_scale, 1e-6
                        )
                        lower_local_scale = min(
                            projected_local_scale_max,
                            max(
                                1.0 / projected_local_scale_max,
                                lower_local_scale,
                            ),
                        )
                        if not bone_scaling_enabled:
                            upper_scale = 1.0
                            lower_world_scale = 1.0
                            lower_local_scale = 1.0

                        parent_cache: dict[str, _LegMatrix2D] = {}
                        parent_world = _leg_animated_world_matrix(
                            skeleton,
                            animation,
                            thigh_setup.parent,
                            output_frame,
                            parent_cache,
                        )
                        # The segmented 3D pose already contains the foot-IK
                        # motion. Reusing the legacy foot endpoint here would
                        # translate the whole leg a second time. Drive the
                        # optional thigh offset only from the hip's motion
                        # relative to the pelvis, then express that screen-space
                        # delta in the actual (possibly mirrored) Spine parent.
                        hip_relative_3d = tuple(
                            hip[index] - pelvis[index] for index in range(3)
                        )
                        hip_relative_2d = projector.direction(hip_relative_3d)  # type: ignore[arg-type]
                        hip_delta_world = (
                            (
                                hip_relative_2d[0]
                                - pose_calibration["hip_source_x"]
                            )
                            * chain_position_gain
                            * thigh_translation_gain,
                            (
                                hip_relative_2d[1]
                                - pose_calibration["hip_source_y"]
                            )
                            * chain_position_gain
                            * thigh_translation_gain,
                        )
                        hip_delta_local = _leg_inverse_vector(
                            parent_world, hip_delta_world
                        )
                        translate_x = min(
                            thigh_translation_limit,
                            max(-thigh_translation_limit, hip_delta_local[0]),
                        )
                        translate_y = min(
                            thigh_translation_limit,
                            max(-thigh_translation_limit, hip_delta_local[1]),
                        )
                        desired_upper_vector = (
                            math.cos(math.radians(desired_upper_world)),
                            math.sin(math.radians(desired_upper_world)),
                        )
                        upper_local_direction = _leg_inverse_vector(
                            parent_world, desired_upper_vector
                        )
                        upper_reference = (
                            leg_setup.x * thigh_setup.scale_x * upper_scale,
                            leg_setup.y * thigh_setup.scale_y * upper_scale,
                        )
                        desired_thigh = _normalise_degrees(
                            _leg_angle(*upper_local_direction)
                            - _leg_angle(*upper_reference)
                            - thigh_setup.rotation
                        )
                        thigh_value = _sample_scalar(
                            existing_thigh, output_frame
                        )
                        thigh_value += (desired_thigh - thigh_value) * strength
                        thigh_value = min(110.0, max(-110.0, thigh_value))

                        thigh_local = _leg_local_matrix(
                            thigh_setup.x + translate_x,
                            thigh_setup.y + translate_y,
                            thigh_setup.rotation + thigh_value,
                            thigh_setup.scale_x * upper_scale,
                            thigh_setup.scale_y * upper_scale,
                        )
                        thigh_world = _leg_multiply_matrices(
                            parent_world, thigh_local
                        )
                        if knee_bend_constraint_enabled:
                            # Re-anchor the knee constraint to the upper segment
                            # that will actually be emitted. This accounts for
                            # strength blending and the thigh's local limit;
                            # using the ideal pre-limit angle could otherwise
                            # move the lower segment onto the forbidden side.
                            ta, tb, tc, td, _, _ = thigh_world
                            emitted_upper_vector = (
                                ta * leg_setup.x + tb * leg_setup.y,
                                tc * leg_setup.x + td * leg_setup.y,
                            )
                            emitted_upper_world = _leg_angle(
                                *emitted_upper_vector
                            )
                            desired_lower_world = _normalise_degrees(
                                emitted_upper_world
                                + pose_calibration[
                                    "target_setup_joint_angle"
                                ]
                                + screen_knee_bend_sign
                                * last_knee_bend_magnitude
                            )
                        desired_lower_vector = (
                            math.cos(math.radians(desired_lower_world)),
                            math.sin(math.radians(desired_lower_world)),
                        )
                        lower_local_direction = _leg_inverse_vector(
                            thigh_world, desired_lower_vector
                        )
                        lower_reference = (
                            leg_setup.scale_x * lower_local_scale,
                            0.0,
                        )
                        desired_leg = _normalise_degrees(
                            _leg_angle(*lower_local_direction)
                            - _leg_angle(*lower_reference)
                            - leg_setup.rotation
                        )
                        leg_value = _sample_scalar(existing_leg, output_frame)
                        leg_value += (desired_leg - leg_value) * strength
                        leg_value = min(
                            leg_local_rotation_limit,
                            max(-leg_local_rotation_limit, leg_value),
                        )

                        thigh_keys.append(ScalarKey(output_frame, thigh_value))
                        leg_keys.append(ScalarKey(output_frame, leg_value))
                        thigh_scale_keys.append(
                            VectorKey(output_frame, upper_scale, upper_scale)
                        )
                        leg_scale_keys.append(
                            VectorKey(
                                output_frame,
                                lower_local_scale,
                                lower_local_scale,
                            )
                        )
                        projected_pose = True

            if not projected_pose and legacy_target_available:
                upper_angle, lower_angle = _solve_two_bone_ik(
                    rest_foot_x + delta_x,
                    rest_foot_y + delta_y,
                    upper_length,
                    lower_length,
                    solver_bend_sign,
                )
                ik_thigh = math.degrees(
                    _normalise_radians(upper_angle - upper_rest_angle)
                )
                thigh_value = _sample_scalar(existing_thigh, output_frame)
                thigh_value += (ik_thigh - thigh_value) * strength
                thigh_value = min(110.0, max(-110.0, thigh_value))
                if knee_bend_constraint_enabled:
                    fallback_bend_magnitude = min(
                        max_screen_knee_bend,
                        max(
                            0.0,
                            abs(
                                math.degrees(
                                    _normalise_radians(
                                        lower_angle - upper_angle
                                    )
                                )
                            )
                            - abs(rest_joint_degrees),
                        ),
                    )
                    fallback_parent_cache: dict[str, _LegMatrix2D] = {}
                    fallback_parent_world = _leg_animated_world_matrix(
                        skeleton,
                        animation,
                        thigh_setup.parent,
                        output_frame,
                        fallback_parent_cache,
                    )
                    fallback_thigh_local = _leg_local_matrix(
                        thigh_setup.x + translate_x,
                        thigh_setup.y + translate_y,
                        thigh_setup.rotation + thigh_value,
                        thigh_setup.scale_x,
                        thigh_setup.scale_y,
                    )
                    fallback_thigh_world = _leg_multiply_matrices(
                        fallback_parent_world, fallback_thigh_local
                    )
                    fa, fb, fc, fd, _, _ = fallback_thigh_world
                    fallback_upper_vector = (
                        fa * leg_setup.x + fb * leg_setup.y,
                        fc * leg_setup.x + fd * leg_setup.y,
                    )
                    fallback_upper_world = _leg_angle(
                        *fallback_upper_vector
                    )
                    fallback_lower_world = _normalise_degrees(
                        fallback_upper_world
                        + rest_joint_degrees
                        + screen_knee_bend_sign
                        * fallback_bend_magnitude
                    )
                    fallback_lower_local_direction = _leg_inverse_vector(
                        fallback_thigh_world,
                        (
                            math.cos(math.radians(fallback_lower_world)),
                            math.sin(math.radians(fallback_lower_world)),
                        ),
                    )
                    constrained_leg = _normalise_degrees(
                        _leg_angle(*fallback_lower_local_direction)
                        - _leg_angle(leg_setup.scale_x, 0.0)
                        - leg_setup.rotation
                    )
                    leg_value = _sample_scalar(existing_leg, output_frame)
                    leg_value += (
                        constrained_leg - leg_value
                    ) * strength
                else:
                    ik_leg = math.degrees(
                        _normalise_radians(
                            lower_angle
                            - thigh_setup_radians
                            - math.radians(thigh_value)
                            - math.radians(leg_setup.rotation)
                        )
                    )
                    leg_value = _sample_scalar(existing_leg, output_frame)
                    leg_value += (ik_leg - leg_value) * strength
                    if bend_sign > 0:
                        leg_value = max(
                            leg_value,
                            minimum_knee_bend - rest_joint_degrees,
                        )
                    else:
                        leg_value = min(
                            leg_value,
                            -minimum_knee_bend - rest_joint_degrees,
                        )
                leg_value = min(
                    leg_local_rotation_limit,
                    max(-leg_local_rotation_limit, leg_value),
                )
                thigh_keys.append(
                    ScalarKey(output_frame, thigh_value)
                )
                leg_keys.append(
                    ScalarKey(output_frame, leg_value)
                )
                depth_scale = 1.0
                if bone_scaling_enabled:
                    depth_scale = max(
                        thigh_scale_min,
                        1.0
                        - min(
                            1.0 - thigh_scale_min,
                            abs(camera_depth) * thigh_depth_scale_gain,
                        ),
                    )
                thigh_scale_keys.append(
                    VectorKey(output_frame, depth_scale, depth_scale)
                )
                leg_scale_keys.append(VectorKey(output_frame, 1.0, 1.0))
            elif not projected_pose:
                # No canonical joints and no foot-IK endpoint: preserve the
                # already mapped FK timelines rather than inventing a target.
                thigh_keys.append(
                    ScalarKey(
                        output_frame,
                        _sample_scalar(existing_thigh, output_frame),
                    )
                )
                leg_keys.append(
                    ScalarKey(
                        output_frame,
                        _sample_scalar(existing_leg, output_frame),
                    )
                )
                thigh_scale_keys.append(VectorKey(output_frame, 1.0, 1.0))
                leg_scale_keys.append(VectorKey(output_frame, 1.0, 1.0))

            thigh_translate_keys.append(
                VectorKey(output_frame, translate_x, translate_y)
            )

        mirror_reset_frames = _mirror_scale_switch_frames(animation, "body")
        animation.bones[thigh_name].rotate = _step_before_frames(
            _unwrap_rotation_segmented(
                thigh_keys,
                110.0,
                mirror_reset_frames,
            ),
            mirror_reset_frames,
        )
        animation.bones[thigh_name].translate = (
            thigh_translate_keys if abs(thigh_translation_gain) > 1e-12 else []
        )
        animation.bones[leg_name].rotate = _step_before_frames(
            _unwrap_rotation_segmented(
                leg_keys,
                leg_local_rotation_limit,
                mirror_reset_frames,
            ),
            mirror_reset_frames,
        )
        if bone_scaling_enabled:
            animation.bones[thigh_name].scale = thigh_scale_keys
            animation.bones[leg_name].scale = leg_scale_keys
        else:
            # Preserve the template's setup scale and export no limb-scale
            # animation. Body scaleX remains reserved for facing mirroring.
            animation.bones[thigh_name].scale = []
            animation.bones[leg_name].scale = []
        if ik_name in motion.bone_tracks:
            if ik_name not in animation.source_bones_used:
                animation.source_bones_used.append(ik_name)
            if ik_name in animation.source_bones_unmapped:
                animation.source_bones_unmapped.remove(ik_name)
        if ik_parent_name and ik_parent_name in motion.bone_tracks:
            if ik_parent_name not in animation.source_bones_used:
                animation.source_bones_used.append(ik_parent_name)
            if ik_parent_name in animation.source_bones_unmapped:
                animation.source_bones_unmapped.remove(ik_parent_name)

    animation.source_bones_used.sort()
    return animation


def _source_leg_length(
    config: dict[str, Any],
    source_rig_profile: dict[str, Any] | None,
    thigh_name: str,
    profile_suffix: str,
) -> float:
    """Return a configurable or canonical source-chain length in MMD units."""

    configured = config.get("source_leg_length_mmd_units")
    if isinstance(configured, dict):
        value = configured.get(thigh_name, configured.get(profile_suffix))
        if value is not None:
            result = abs(float(value))
            if math.isfinite(result) and result > 1e-5:
                return result
    elif configured is not None:
        result = abs(float(configured))
        if math.isfinite(result) and result > 1e-5:
            return result

    geometry = (
        source_rig_profile.get("geometry", {})
        if isinstance(source_rig_profile, dict)
        else {}
    )
    thigh_vector = geometry.get(f"thigh_{profile_suffix}")
    shin_vector = geometry.get(f"shin_{profile_suffix}")
    if (
        isinstance(thigh_vector, list)
        and len(thigh_vector) == 3
        and isinstance(shin_vector, list)
        and len(shin_vector) == 3
    ):
        try:
            result = math.sqrt(sum(float(value) ** 2 for value in thigh_vector))
            result += math.sqrt(sum(float(value) ** 2 for value in shin_vector))
        except (TypeError, ValueError):
            result = 0.0
        if math.isfinite(result) and result > 1e-5:
            return result

    # The bundled canonical MMD proxy uses two four-unit leg segments.  Keep
    # this fallback explicit so conversion remains deterministic when a custom
    # profile omits geometry.
    return 8.0


def _leg_vector3(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        result = tuple(float(component) for component in value)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(component) for component in result):
        return None
    return result  # type: ignore[return-value]


def _leg_angle(x: float, y: float) -> float:
    if math.hypot(x, y) < 1e-12:
        return 0.0
    return math.degrees(math.atan2(y, x))


def _leg_local_matrix(
    x: float,
    y: float,
    rotation: float,
    scale_x: float,
    scale_y: float,
) -> _LegMatrix2D:
    radians = math.radians(rotation)
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


def _leg_multiply_matrices(
    parent: _LegMatrix2D,
    local: _LegMatrix2D,
) -> _LegMatrix2D:
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


def _leg_inverse_vector(
    matrix: _LegMatrix2D,
    vector: tuple[float, float],
) -> tuple[float, float]:
    a, b, c, d, _, _ = matrix
    determinant = a * d - b * c
    if abs(determinant) < 1e-10:
        return vector
    return (
        (d * vector[0] - b * vector[1]) / determinant,
        (-c * vector[0] + a * vector[1]) / determinant,
    )


def _leg_setup_world_matrices(
    skeleton: SpineSkeleton,
) -> dict[str, _LegMatrix2D]:
    matrices: dict[str, _LegMatrix2D] = {}
    for bone in skeleton.bones:
        local = _leg_local_matrix(
            bone.x,
            bone.y,
            bone.rotation,
            bone.scale_x,
            bone.scale_y,
        )
        matrices[bone.name] = (
            local
            if bone.parent is None
            else _leg_multiply_matrices(matrices[bone.parent], local)
        )
    return matrices


def _leg_sample_vector(
    keys: list[VectorKey],
    frame: int,
    default: float,
) -> tuple[float, float]:
    if not keys:
        return default, default
    if frame <= keys[0].frame:
        if frame == keys[0].frame:
            return keys[0].x, keys[0].y
        return default, default
    for first, second in zip(keys, keys[1:]):
        if frame <= second.frame:
            if frame < second.frame and first.curve == "stepped":
                return first.x, first.y
            span = max(1, second.frame - first.frame)
            amount = (frame - first.frame) / span
            return (
                first.x + (second.x - first.x) * amount,
                first.y + (second.y - first.y) * amount,
            )
    return keys[-1].x, keys[-1].y


def _leg_animated_world_matrix(
    skeleton: SpineSkeleton,
    animation: RetargetedAnimation,
    bone_name: str | None,
    frame: int,
    cache: dict[str, _LegMatrix2D],
) -> _LegMatrix2D:
    if bone_name is None:
        return 1.0, 0.0, 0.0, 1.0, 0.0, 0.0
    cached = cache.get(bone_name)
    if cached is not None:
        return cached
    setup = skeleton.bone_by_name[bone_name]
    timeline = animation.bones.get(bone_name, BoneAnimation())
    translate = _leg_sample_vector(timeline.translate, frame, 0.0)
    scale = _leg_sample_vector(timeline.scale, frame, 1.0)
    rotation = _sample_scalar(timeline.rotate, frame)
    local = _leg_local_matrix(
        setup.x + translate[0],
        setup.y + translate[1],
        setup.rotation + rotation,
        setup.scale_x * scale[0],
        setup.scale_y * scale[1],
    )
    if setup.parent is None:
        world = local
    else:
        world = _leg_multiply_matrices(
            _leg_animated_world_matrix(
                skeleton,
                animation,
                setup.parent,
                frame,
                cache,
            ),
            local,
        )
    cache[bone_name] = world
    return world


def apply_body_facing_mirror(
    animation: RetargetedAnimation,
    motion: VMDMotion,
    config: dict[str, Any],
) -> RetargetedAnimation:
    """Switch the Spine character between its two horizontal facing states.

    The source heading is reconstructed from the global parent, center and
    lower-body rotations.  MMD's forward direction is the absolute 0-degree
    reference.  Entering the configured counterclockwise angular interval
    flips ``body.scaleX``.  Stepped scale keys prevent an unwanted squash
    through zero at a switch.
    """

    if not bool(config.get("enabled", False)):
        return animation
    target_bone = str(config.get("target_bone", "body"))
    source_names = [
        str(name)
        for name in config.get("sources", ["全ての親", "センター", "グルーブ"])
        if str(name) in motion.bone_tracks
    ]
    if not source_names:
        return animation

    counterclockwise_start = float(
        config.get("counterclockwise_start_deg", 45.0)
    ) % 360.0
    mirror_span = float(config.get("mirror_span_deg", 180.0))
    if not math.isfinite(counterclockwise_start) or not math.isfinite(mirror_span):
        raise ValueError("Facing mirror angles must be finite")
    if mirror_span <= 0.0 or mirror_span >= 360.0:
        raise ValueError("Facing mirror span must be greater than 0 and less than 360")
    default_scale_x = float(config.get("default_scale_x", 1.0))
    mirrored_scale_x = float(config.get("mirrored_scale_x", -1.0))
    source_start = animation.source_start_frame
    source_end = source_start + animation.max_frame

    def heading_at(frame: int) -> float | None:
        rotation = (0.0, 0.0, 0.0, 1.0)
        for name in source_names:
            sampled = sample_track(motion.bone_tracks[name], frame)
            rotation = _multiply_quaternions(rotation, sampled.rotation)
        forward = rotate_vector(_normalise_quaternion(rotation), (0.0, 0.0, -1.0))
        if math.hypot(forward[0], forward[2]) < 1e-7:
            return None
        # Looking down from above, +X from the MMD forward axis (-Z) is
        # clockwise in this project's convention.
        return math.degrees(math.atan2(forward[0], -forward[2])) % 360.0

    initial_heading = heading_at(source_start)
    if initial_heading is None:
        return animation

    def inside_mirror_range(clockwise_heading: float) -> bool:
        # heading_at() is clockwise-positive. Convert to the user's top-view
        # convention, where counterclockwise from MMD's default forward axis
        # is positive, then test the interval in a wrap-safe form.
        counterclockwise_heading = (-clockwise_heading) % 360.0
        offset = (counterclockwise_heading - counterclockwise_start) % 360.0
        epsilon = 1e-6
        return epsilon < offset <= mirror_span + epsilon

    mirrored = inside_mirror_range(initial_heading)
    keys = [
        VectorKey(
            0,
            mirrored_scale_x if mirrored else default_scale_x,
            1.0,
            "stepped",
        )
    ]
    last_heading = initial_heading
    for source_frame in range(source_start + 1, source_end + 1):
        heading = heading_at(source_frame)
        if heading is None:
            heading = last_heading
        else:
            last_heading = heading
        next_mirrored = inside_mirror_range(heading)
        if next_mirrored == mirrored:
            continue
        mirrored = next_mirrored
        keys.append(
            VectorKey(
                source_frame - source_start,
                mirrored_scale_x if mirrored else default_scale_x,
                1.0,
                "stepped",
            )
        )

    animation.bones.setdefault(target_bone, BoneAnimation()).scale = keys
    for name in source_names:
        if name not in animation.source_bones_used:
            animation.source_bones_used.append(name)
        if name in animation.source_bones_unmapped:
            animation.source_bones_unmapped.remove(name)
    animation.source_bones_used.sort()
    return animation


def _solve_two_bone_ik(
    target_x: float,
    target_y: float,
    upper_length: float,
    lower_length: float,
    bend_sign: float,
) -> tuple[float, float]:
    distance = math.hypot(target_x, target_y)
    minimum = abs(upper_length - lower_length) + 1e-5
    maximum = upper_length + lower_length - 1e-5
    clamped_distance = min(maximum, max(minimum, distance))
    if distance > 1e-8 and clamped_distance != distance:
        factor = clamped_distance / distance
        target_x *= factor
        target_y *= factor
    cosine = (
        target_x * target_x
        + target_y * target_y
        - upper_length * upper_length
        - lower_length * lower_length
    ) / (2.0 * upper_length * lower_length)
    joint = math.acos(min(1.0, max(-1.0, cosine))) * bend_sign
    upper = math.atan2(target_y, target_x) - math.atan2(
        lower_length * math.sin(joint),
        upper_length + lower_length * math.cos(joint),
    )
    return upper, upper + joint


def _normalise_radians(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _normalise_degrees(value: float) -> float:
    """Wrap an angle to Spine's continuous local-degree interval."""

    return (value + 180.0) % 360.0 - 180.0


def _unwrap_degrees_near(value: float, reference: float) -> float:
    """Return the equivalent degree angle nearest a continuous reference."""

    while value - reference > 180.0:
        value -= 360.0
    while value - reference < -180.0:
        value += 360.0
    return value


def _multiply_quaternions(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    ax, ay, az, aw = first
    bx, by, bz, bw = second
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _normalise_quaternion(
    value: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    length = math.sqrt(sum(component * component for component in value))
    if length < 1e-8:
        return 0.0, 0.0, 0.0, 1.0
    return tuple(component / length for component in value)  # type: ignore[return-value]


def _sample_scalar(keys: list[ScalarKey], frame: int) -> float:
    if not keys:
        return 0.0
    if frame <= keys[0].frame:
        return keys[0].value if frame == keys[0].frame else 0.0
    for first, second in zip(keys, keys[1:]):
        if frame <= second.frame:
            span = second.frame - first.frame
            if span <= 0:
                return second.value
            amount = (frame - first.frame) / span
            return first.value + (second.value - first.value) * amount
    return keys[-1].value
