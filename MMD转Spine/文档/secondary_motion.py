"""Deterministic inertial hair and skirt rotation postprocessing.

The pass consumes only canonical world motion from ``head`` and ``body``.
Horizontal display mirroring is deliberately removed from the physical FK so
that a facing switch neither kicks nor resets an already moving spring.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .retarget import BoneAnimation, RetargetedAnimation, ScalarKey, VectorKey
from .spine_reader import SpineSkeleton


Matrix2D = tuple[float, float, float, float, float, float]


@dataclass(frozen=True, slots=True)
class DriverSample:
    angle: float
    x: float
    y: float


def apply_secondary_motion(
    animation: RetargetedAnimation,
    skeleton: SpineSkeleton,
    config: dict[str, Any],
) -> RetargetedAnimation:
    """Bake additive hair/skirt sway as ordinary local rotate keys.

    The first invocation snapshots the six untouched rotate timelines.  Every
    later invocation restores and rebuilds from that snapshot, so changing a
    tuning value or simply running the pass twice can never accumulate sway.
    """

    if not bool(config.get("enabled", False)):
        # A caller may retune the same in-memory object by first enabling and
        # then disabling this pass. Disabling must remove the baked result.
        for name, base in animation.secondary_motion_base_rotations.items():
            animation.bones.setdefault(name, BoneAnimation()).rotate = list(base)
        animation.secondary_motion_diagnostics = {}
        return animation
    if animation.fps <= 0 or animation.max_frame < 0:
        raise ValueError("Secondary motion requires a positive fps and duration")

    hair_config = _object(config.get("hair"))
    skirt_config = _object(config.get("skirt"))
    twin_tail_config = _object(hair_config.get("twin_tail_cycle"))
    hair_bones = list(_hair_bone_rules(hair_config))
    skirt_bones = _string_list(
        skirt_config.get("bones", ["skirtF", "skirtB"]),
        ["skirtF", "skirtB"],
    )
    target_bones = hair_bones + [name for name in skirt_bones if name not in hair_bones]
    known_bones = skeleton.bone_by_name
    missing = [name for name in target_bones if name not in known_bones]
    if missing:
        raise ValueError(
            "Secondary-motion target bones are missing from the Spine template: "
            + ", ".join(missing)
        )

    _restore_or_capture_base(animation, target_bones)
    invalid_counter = [0]
    mirror_bone = str(config.get("mirror_bone", "body"))
    head_driver = str(hair_config.get("driver_bone", "head"))
    skirt_driver = str(skirt_config.get("driver_bone", "body"))
    if head_driver not in known_bones or skirt_driver not in known_bones:
        raise ValueError("Secondary-motion head/body driver bone is missing")

    driver_names = {head_driver, skirt_driver}
    leg_motion_gain = float(skirt_config.get("leg_motion_gain", 0.0))
    leg_driver_bones = _string_list(
        skirt_config.get("leg_driver_bones", ["legF", "legB"]),
        ["legF", "legB"],
    )
    if leg_motion_gain != 0.0:
        missing_leg_drivers = [
            name for name in leg_driver_bones if name not in known_bones
        ]
        if missing_leg_drivers:
            raise ValueError(
                "Secondary-motion leg driver bones are missing: "
                + ", ".join(missing_leg_drivers)
            )
        driver_names.update(leg_driver_bones)
    sampled = _sample_canonical_world_drivers(
        animation,
        skeleton,
        driver_names,
        mirror_bone,
        invalid_counter,
    )
    global_limits = _object(config.get("limits"))
    hard_limit = max(0.001, abs(float(global_limits.get("hard_angle_deg", 30.0))))
    hard_limit = min(30.0, hard_limit)
    soft_limit = abs(float(global_limits.get("soft_angle_deg", 24.0)))
    soft_limit = min(hard_limit, soft_limit)
    input_angular_velocity = max(
        1.0, abs(float(global_limits.get("input_angular_velocity_deg_per_second", 720.0)))
    )
    input_linear_velocity = max(
        1.0, abs(float(global_limits.get("input_linear_velocity_px_per_second", 2500.0)))
    )
    substeps = max(1, min(16, int(global_limits.get("physics_substeps", 4))))

    leg_targets = _leg_motion_targets(
        sampled,
        skirt_driver,
        leg_driver_bones,
        animation.fps,
        leg_motion_gain,
        input_linear_velocity,
    )

    hair_shared, hair_stats = _simulate_shared_spring(
        sampled[head_driver],
        animation.fps,
        hair_config,
        soft_limit,
        hard_limit,
        input_angular_velocity,
        input_linear_velocity,
        substeps,
    )
    skirt_shared, skirt_stats = _simulate_shared_spring(
        sampled[skirt_driver],
        animation.fps,
        skirt_config,
        soft_limit,
        hard_limit,
        input_angular_velocity,
        input_linear_velocity,
        substeps,
        extra_targets=leg_targets,
    )

    twin_tail_shared, twin_tail_stats = _derive_twin_tail_shared_curve(
        hair_shared,
        animation.fps,
        hair_config,
        twin_tail_config,
        soft_limit,
        hard_limit,
        substeps,
    )
    twin_tail_bones = set(
        _string_list(
            twin_tail_config.get("bones", ["hairTwinF", "hairTwinB"]),
            ["hairTwinF", "hairTwinB"],
        )
    )

    hair_offsets: dict[str, list[float]] = {}
    hair_rules = _hair_bone_rules(hair_config)
    for bone_name, rule in hair_rules.items():
        hair_offsets[bone_name] = _derive_hair_curve(
            twin_tail_shared if bone_name in twin_tail_bones else hair_shared,
            animation.fps,
            _object(rule),
            soft_limit,
            hard_limit,
            float(hair_config.get("max_angular_velocity_deg_per_second", 220.0)),
        )

    # Both skirt pieces request the same extra world rotation.  Convert that
    # shared world delta into local deltas using the actual parent hierarchy.
    # With skirtB parented to skirtF this yields F=S and B=0, preventing 2*S.
    skirt_factors = _shared_world_to_local_factors(skeleton, skirt_bones)
    skirt_offsets = {
        name: [value * skirt_factors[name] for value in skirt_shared]
        for name in skirt_bones
    }

    all_offsets = {**hair_offsets, **skirt_offsets}
    for bone_name, offsets in all_offsets.items():
        base = animation.secondary_motion_base_rotations[bone_name]
        keys = [
            ScalarKey(frame, _sample_scalar(base, frame, invalid_counter) + offsets[frame])
            for frame in range(animation.max_frame + 1)
        ]
        animation.bones.setdefault(bone_name, BoneAnimation()).rotate = keys

    mirror_frames = _mirror_transition_frames(
        animation,
        mirror_bone,
        invalid_counter,
    )
    max_output_velocity = max(
        abs(float(hair_config.get("max_angular_velocity_deg_per_second", 220.0))),
        abs(float(skirt_config.get("max_angular_velocity_deg_per_second", 180.0))),
    )
    mirror_steps = [
        abs(curve[frame] - curve[frame - 1])
        for curve in (hair_shared, twin_tail_shared, skirt_shared)
        for frame in mirror_frames
        if 0 < frame < len(curve)
    ]
    skirt_sync_error = 0.0
    for frame in range(animation.max_frame + 1):
        for bone_name in skirt_bones:
            inherited = 0.0
            current: str | None = bone_name
            while current is not None:
                if current in skirt_offsets:
                    inherited += skirt_offsets[current][frame]
                current = known_bones[current].parent if current in known_bones else None
            skirt_sync_error = max(
                skirt_sync_error, abs(inherited - skirt_shared[frame])
            )

    max_abs_by_bone = {
        name: max((abs(value) for value in values), default=0.0)
        for name, values in all_offsets.items()
    }
    finite = all(
        math.isfinite(value)
        for values in all_offsets.values()
        for value in values
    )
    animation.secondary_motion_diagnostics = {
        "enabled": True,
        "fixed_time_step_seconds": 1.0 / (animation.fps * substeps),
        "physics_substeps": substeps,
        "hard_limit_deg": hard_limit,
        "soft_limit_deg": soft_limit,
        "max_abs_additive_deg_by_bone": max_abs_by_bone,
        "hair_shared_max_abs_deg": max((abs(value) for value in hair_shared), default=0.0),
        "twin_tail_shared_max_abs_deg": max(
            (abs(value) for value in twin_tail_shared), default=0.0
        ),
        "skirt_shared_max_abs_deg": max((abs(value) for value in skirt_shared), default=0.0),
        "hair": hair_stats,
        "twin_tail": twin_tail_stats,
        "skirt": skirt_stats,
        "skirt_leg_driver_bones": leg_driver_bones,
        "skirt_leg_motion_gain": leg_motion_gain,
        "skirt_leg_target_max_abs_deg": max(
            (abs(value) for value in leg_targets), default=0.0
        ),
        "skirt_local_factors": skirt_factors,
        "skirt_world_sync_max_error_deg": skirt_sync_error,
        "mirror_bone": mirror_bone,
        "mirror_scale_sign_ignored": True,
        "mirror_transition_frames": mirror_frames,
        "mirror_max_shared_step_deg": max(mirror_steps, default=0.0),
        "maximum_per_frame_step_deg": max_output_velocity / animation.fps,
        "invalid_input_samples_replaced": invalid_counter[0],
        "all_outputs_finite": finite,
        "idempotent_base_snapshot": True,
        "target_bones": target_bones,
    }
    validate_secondary_motion(animation)
    return animation


def validate_secondary_motion(animation: RetargetedAnimation) -> None:
    """Raise if baked secondary-motion invariants do not hold."""

    diagnostics = animation.secondary_motion_diagnostics
    if not diagnostics.get("enabled"):
        return
    hard_limit = float(diagnostics["hard_limit_deg"])
    maximum = max(
        (float(value) for value in diagnostics["max_abs_additive_deg_by_bone"].values()),
        default=0.0,
    )
    if maximum > hard_limit + 1e-6:
        raise ValueError(
            f"Secondary-motion additive angle {maximum:.6f} exceeds {hard_limit:.6f} degrees"
        )
    if float(diagnostics["skirt_world_sync_max_error_deg"]) > 1e-6:
        raise ValueError("Front/back skirt secondary motion is not world-synchronous")
    if not bool(diagnostics["all_outputs_finite"]):
        raise ValueError("Secondary-motion output contains NaN or infinity")
    mirror_step = float(diagnostics["mirror_max_shared_step_deg"])
    allowed_step = float(diagnostics["maximum_per_frame_step_deg"])
    if mirror_step > allowed_step + 1e-6:
        raise ValueError("Secondary motion is unstable at a facing-mirror transition")


def _simulate_shared_spring(
    samples: list[DriverSample],
    fps: float,
    config: dict[str, Any],
    soft_limit: float,
    hard_limit: float,
    input_angular_velocity_limit: float,
    input_linear_velocity_limit: float,
    substeps: int,
    extra_targets: list[float] | None = None,
) -> tuple[list[float], dict[str, Any]]:
    if not samples:
        return [], {"max_abs_deg": 0.0, "max_velocity_deg_per_second": 0.0}
    frame_dt = 1.0 / fps
    step_dt = frame_dt / substeps
    angular_gain = float(config.get("angular_velocity_gain", 0.055))
    lateral_gain = float(config.get("lateral_velocity_gain", 0.010))
    frequency = max(0.05, abs(float(config.get("spring_frequency_hz", 2.0))))
    damping_ratio = max(0.0, float(config.get("damping_ratio", 0.72)))
    max_velocity = max(
        1.0, abs(float(config.get("max_angular_velocity_deg_per_second", 200.0)))
    )
    omega = 2.0 * math.pi * frequency
    stiffness = omega * omega
    damping = 2.0 * damping_ratio * omega
    state = 0.0
    velocity = 0.0
    result = [0.0]
    max_seen_velocity = 0.0
    soft_clamp_count = 0
    previous = samples[0]

    for frame, sample in enumerate(samples[1:], start=1):
        angular_velocity = _clamp(
            (sample.angle - previous.angle) / frame_dt,
            -input_angular_velocity_limit,
            input_angular_velocity_limit,
        )
        lateral_velocity = _clamp(
            (sample.x - previous.x) / frame_dt,
            -input_linear_velocity_limit,
            input_linear_velocity_limit,
        )
        raw_target = -(
            angular_velocity * angular_gain + lateral_velocity * lateral_gain
        )
        if extra_targets is not None and frame < len(extra_targets):
            raw_target += extra_targets[frame]
        target = _soft_hard_limit(raw_target, soft_limit, hard_limit)
        if abs(raw_target) > soft_limit:
            soft_clamp_count += 1
        for _ in range(substeps):
            previous_state = state
            acceleration = stiffness * (target - state) - damping * velocity
            if not math.isfinite(acceleration):
                acceleration = 0.0
            velocity = _clamp(
                velocity + acceleration * step_dt,
                -max_velocity,
                max_velocity,
            )
            candidate = state + velocity * step_dt
            limited = _soft_hard_limit(candidate, soft_limit, hard_limit)
            state = _clamp(
                limited,
                previous_state - max_velocity * step_dt,
                previous_state + max_velocity * step_dt,
            )
            state = _clamp(state, -hard_limit, hard_limit)
            velocity = (state - previous_state) / step_dt
            if abs(state) >= hard_limit - 1e-9 and state * velocity > 0:
                velocity = 0.0
            max_seen_velocity = max(max_seen_velocity, abs(velocity))
        result.append(state)
        previous = sample
    return result, {
        "max_abs_deg": max((abs(value) for value in result), default=0.0),
        "max_velocity_deg_per_second": max_seen_velocity,
        "soft_clamp_input_frames": soft_clamp_count,
        "spring_frequency_hz": frequency,
        "damping_ratio": damping_ratio,
    }


def _leg_motion_targets(
    sampled: dict[str, list[DriverSample]],
    body_name: str,
    leg_names: list[str],
    fps: float,
    gain: float,
    input_velocity_limit: float,
) -> list[float]:
    """Convert the dominant leg's body-relative knee motion into skirt drive.

    Positions are transformed back into the body's continuous, unmirrored
    orientation before differentiating.  Selecting the stronger leg each frame
    keeps opposite left/right movements from cancelling each other.
    """

    body_samples = sampled.get(body_name, [])
    frame_count = len(body_samples)
    result = [0.0] * frame_count
    active_legs = [name for name in leg_names if name in sampled]
    if gain == 0.0 or not active_legs or frame_count < 2:
        return result

    frame_dt = 1.0 / fps
    previous_relative = {
        name: _body_relative_position(sampled[name][0], body_samples[0])
        for name in active_legs
    }
    for frame in range(1, frame_count):
        candidates: list[float] = []
        body = body_samples[frame]
        for name in active_legs:
            relative = _body_relative_position(sampled[name][frame], body)
            previous = previous_relative[name]
            velocity_x = (relative[0] - previous[0]) / frame_dt
            velocity_y = (relative[1] - previous[1]) / frame_dt
            speed = min(input_velocity_limit, math.hypot(velocity_x, velocity_y))
            if speed > 1e-8:
                # Raising/lowering is the primary sign source.  For a mostly
                # horizontal leg sweep, retain its horizontal direction.
                direction = velocity_y if abs(velocity_y) >= abs(velocity_x) * 0.35 else velocity_x
                candidates.append(math.copysign(speed, direction))
            previous_relative[name] = relative
        if candidates:
            dominant = max(candidates, key=abs)
            result[frame] = -dominant * gain
    return result


def _body_relative_position(
    child: DriverSample, body: DriverSample
) -> tuple[float, float]:
    delta_x = child.x - body.x
    delta_y = child.y - body.y
    radians = math.radians(body.angle)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    return (
        cosine * delta_x + sine * delta_y,
        -sine * delta_x + cosine * delta_y,
    )


def _derive_twin_tail_shared_curve(
    shared: list[float],
    fps: float,
    hair_config: dict[str, Any],
    config: dict[str, Any],
    soft_limit: float,
    hard_limit: float,
    substeps: int,
) -> tuple[list[float], dict[str, Any]]:
    """Create one subtly different cycle shared by both twin tails."""

    if not shared or not config:
        return shared[:], {"enabled": False, "period_scale": 1.0, "difference_mix": 0.0}
    period_scale = _clamp(abs(float(config.get("period_scale", 1.15))), 0.80, 1.50)
    difference_mix = _clamp(float(config.get("difference_mix", 0.30)), 0.0, 0.50)
    damping_ratio = max(0.0, float(config.get("damping_ratio", 0.68)))
    base_frequency = max(
        0.05, abs(float(hair_config.get("spring_frequency_hz", 2.0)))
    )
    frequency = base_frequency / period_scale
    max_velocity = max(
        1.0,
        abs(float(hair_config.get("max_angular_velocity_deg_per_second", 220.0))),
    )
    frame_dt = 1.0 / fps
    step_dt = frame_dt / substeps
    omega = 2.0 * math.pi * frequency
    stiffness = omega * omega
    damping = 2.0 * damping_ratio * omega
    state = shared[0]
    velocity = 0.0
    output = [shared[0]]
    previous_output = shared[0]
    max_step = max_velocity / fps

    for target in shared[1:]:
        for _ in range(substeps):
            previous_state = state
            acceleration = stiffness * (target - state) - damping * velocity
            if not math.isfinite(acceleration):
                acceleration = 0.0
            velocity = _clamp(
                velocity + acceleration * step_dt,
                -max_velocity,
                max_velocity,
            )
            state = _soft_hard_limit(
                state + velocity * step_dt,
                soft_limit,
                hard_limit,
            )
            state = _clamp(
                state,
                previous_state - max_velocity * step_dt,
                previous_state + max_velocity * step_dt,
            )
            velocity = (state - previous_state) / step_dt
        value = target * (1.0 - difference_mix) + state * difference_mix
        value = _clamp(value, previous_output - max_step, previous_output + max_step)
        value = _soft_hard_limit(value, soft_limit, hard_limit)
        output.append(value)
        previous_output = value

    return output, {
        "enabled": True,
        "period_scale": period_scale,
        "difference_mix": difference_mix,
        "frequency_hz": frequency,
        "damping_ratio": damping_ratio,
        "bones": _string_list(
            config.get("bones", ["hairTwinF", "hairTwinB"]),
            ["hairTwinF", "hairTwinB"],
        ),
    }


def _derive_hair_curve(
    shared: list[float],
    fps: float,
    rule: dict[str, Any],
    soft_limit: float,
    hard_limit: float,
    max_velocity: float,
) -> list[float]:
    amplitude = _clamp(float(rule.get("amplitude", 1.0)), 0.85, 1.15)
    response = _clamp(float(rule.get("response", 1.0)), 0.90, 1.0)
    max_step = max(1.0, abs(max_velocity)) / fps
    result: list[float] = []
    previous = 0.0
    for shared_value in shared:
        filtered = previous + (shared_value * amplitude - previous) * response
        # A small response variation may delay a zero crossing, but it must not
        # create a temporary opposite-direction strand.
        if filtered * shared_value < 0.0:
            filtered = 0.0
        filtered = _clamp(filtered, previous - max_step, previous + max_step)
        filtered = _soft_hard_limit(filtered, soft_limit, hard_limit)
        result.append(filtered)
        previous = filtered
    return result


def _sample_canonical_world_drivers(
    animation: RetargetedAnimation,
    skeleton: SpineSkeleton,
    driver_names: set[str],
    mirror_bone: str,
    invalid_counter: list[int],
) -> dict[str, list[DriverSample]]:
    result = {name: [] for name in driver_names}
    previous_raw_angles: dict[str, float] = {}
    previous_unwrapped: dict[str, float] = {}
    previous_positions: dict[str, tuple[float, float]] = {}
    mirror_timelines = animation.bones.get(mirror_bone, BoneAnimation())
    mirror_reference_scale_x = abs(
        _sample_vector(mirror_timelines.scale, 0, 1.0, invalid_counter)[0]
    )
    if mirror_reference_scale_x < 1e-8:
        mirror_reference_scale_x = 1.0
    for frame in range(animation.max_frame + 1):
        matrices: dict[str, Matrix2D] = {}
        for setup in skeleton.bones:
            timelines = animation.bones.get(setup.name, BoneAnimation())
            translate = _sample_vector(timelines.translate, frame, 0.0, invalid_counter)
            scale = _sample_vector(timelines.scale, frame, 1.0, invalid_counter)
            animated_scale_x = scale[0]
            if setup.name == mirror_bone:
                # Facing mirror is display state, not physical scale. Keep the
                # first-frame magnitude constant too, so a 1/-0.9 switch cannot
                # kick the spring through a child-position discontinuity.
                animated_scale_x = mirror_reference_scale_x
            local = _local_matrix(
                setup.x + translate[0],
                setup.y + translate[1],
                setup.rotation + _sample_scalar(timelines.rotate, frame, invalid_counter),
                setup.scale_x * animated_scale_x,
                setup.scale_y * scale[1],
            )
            matrices[setup.name] = (
                local
                if setup.parent is None
                else _multiply_matrices(matrices[setup.parent], local)
            )
            if setup.name not in driver_names:
                continue
            matrix = matrices[setup.name]
            raw_angle = math.degrees(math.atan2(matrix[2], matrix[0]))
            x, y = matrix[4], matrix[5]
            if not all(math.isfinite(value) for value in (raw_angle, x, y)):
                invalid_counter[0] += 1
                raw_angle = previous_raw_angles.get(setup.name, 0.0)
                x, y = previous_positions.get(setup.name, (0.0, 0.0))
            if setup.name in previous_raw_angles:
                delta = _normalise_degrees(raw_angle - previous_raw_angles[setup.name])
                angle = previous_unwrapped[setup.name] + delta
            else:
                angle = raw_angle
            result[setup.name].append(DriverSample(angle, x, y))
            previous_raw_angles[setup.name] = raw_angle
            previous_unwrapped[setup.name] = angle
            previous_positions[setup.name] = (x, y)
    return result


def _restore_or_capture_base(
    animation: RetargetedAnimation, target_bones: list[str]
) -> None:
    if not animation.secondary_motion_base_rotations:
        animation.secondary_motion_base_rotations = {
            name: list(animation.bones.get(name, BoneAnimation()).rotate)
            for name in target_bones
        }
    else:
        for name in target_bones:
            animation.secondary_motion_base_rotations.setdefault(
                name, list(animation.bones.get(name, BoneAnimation()).rotate)
            )
    for name in target_bones:
        animation.bones.setdefault(name, BoneAnimation()).rotate = list(
            animation.secondary_motion_base_rotations[name]
        )


def _shared_world_to_local_factors(
    skeleton: SpineSkeleton, bone_names: list[str]
) -> dict[str, float]:
    targets = set(bone_names)
    factors: dict[str, float] = {}
    for setup in skeleton.bones:
        if setup.name not in targets:
            continue
        inherited = 0.0
        parent = setup.parent
        while parent is not None:
            inherited += factors.get(parent, 0.0)
            parent = skeleton.bone_by_name[parent].parent
        factors[setup.name] = 1.0 - inherited
    return factors


def _mirror_transition_frames(
    animation: RetargetedAnimation,
    mirror_bone: str,
    invalid_counter: list[int],
) -> list[int]:
    timelines = animation.bones.get(mirror_bone, BoneAnimation())
    result: list[int] = []
    previous_sign = 1
    for frame in range(animation.max_frame + 1):
        scale_x = _sample_vector(timelines.scale, frame, 1.0, invalid_counter)[0]
        sign = -1 if scale_x < 0 else 1
        if frame and sign != previous_sign:
            result.append(frame)
        previous_sign = sign
    return result


def _hair_bone_rules(config: dict[str, Any]) -> dict[str, Any]:
    value = config.get(
        "bones",
        {
            "hairTwinF": {"amplitude": 1.02, "response": 1.0},
            "hairTwinB": {"amplitude": 1.0, "response": 1.0},
            "hairF": {"amplitude": 0.96, "response": 1.0},
            "hairB": {"amplitude": 0.98, "response": 1.0},
        },
    )
    if not isinstance(value, dict):
        raise ValueError("secondary_motion.hair.bones must be an object")
    return {str(name): rule for name, rule in value.items()}


def _string_list(value: Any, default: list[str]) -> list[str]:
    if not isinstance(value, list) or not value:
        return default[:]
    return [str(item) for item in value]


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _sample_scalar(
    keys: list[ScalarKey], frame: int, invalid_counter: list[int]
) -> float:
    if not keys:
        return 0.0
    if frame <= keys[0].frame:
        return _finite(keys[0].value if frame == keys[0].frame else 0.0, 0.0, invalid_counter)
    for first, second in zip(keys, keys[1:]):
        if frame <= second.frame:
            first_value = _finite(first.value, 0.0, invalid_counter)
            second_value = _finite(second.value, first_value, invalid_counter)
            span = second.frame - first.frame
            if span <= 0:
                return second_value
            amount = (frame - first.frame) / span
            return first_value + (second_value - first_value) * amount
    return _finite(keys[-1].value, 0.0, invalid_counter)


def _sample_vector(
    keys: list[VectorKey],
    frame: int,
    default: float,
    invalid_counter: list[int],
) -> tuple[float, float]:
    if not keys:
        return default, default
    if frame <= keys[0].frame:
        if frame != keys[0].frame:
            return default, default
        return (
            _finite(keys[0].x, default, invalid_counter),
            _finite(keys[0].y, default, invalid_counter),
        )
    for first, second in zip(keys, keys[1:]):
        if frame <= second.frame:
            first_x = _finite(first.x, default, invalid_counter)
            first_y = _finite(first.y, default, invalid_counter)
            if frame < second.frame and first.curve == "stepped":
                return first_x, first_y
            second_x = _finite(second.x, first_x, invalid_counter)
            second_y = _finite(second.y, first_y, invalid_counter)
            amount = (frame - first.frame) / max(1, second.frame - first.frame)
            return (
                first_x + (second_x - first_x) * amount,
                first_y + (second_y - first_y) * amount,
            )
    return (
        _finite(keys[-1].x, default, invalid_counter),
        _finite(keys[-1].y, default, invalid_counter),
    )


def _finite(value: float, fallback: float, invalid_counter: list[int]) -> float:
    value = float(value)
    if math.isfinite(value):
        return value
    invalid_counter[0] += 1
    return fallback


def _local_matrix(
    x: float, y: float, rotation: float, scale_x: float, scale_y: float
) -> Matrix2D:
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


def _soft_hard_limit(value: float, soft_limit: float, hard_limit: float) -> float:
    if not math.isfinite(value):
        return 0.0
    magnitude = abs(value)
    if magnitude <= soft_limit or hard_limit <= soft_limit:
        return _clamp(value, -hard_limit, hard_limit)
    span = hard_limit - soft_limit
    buffered = soft_limit + span * math.tanh((magnitude - soft_limit) / span)
    return math.copysign(min(hard_limit, buffered), value)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return min(maximum, max(minimum, value))


def _normalise_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0
