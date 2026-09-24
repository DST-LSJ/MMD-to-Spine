"""Shared conversion and secondary-motion settings; clip times live in main.py."""
from typing import Any

SOURCE_AND_OUTPUT_FPS = 30.0

# This one value describes the supplied Spine character's right-front view and
# is also the facing boundary.  The former manually tuned 145-degree value is
# intentionally absent from every calculation in this file.
CHARACTER_VIEW_OFFSET_DEGREES = 30.0

PROJECTION_SCALE_PIXELS_PER_MMD_UNIT = 20.0
BODY_MIRROR_HYSTERESIS_DEGREES = 4.0
HEAD_MIRROR_HYSTERESIS_DEGREES = 4.0

SEMANTIC_RETARGET_SETTINGS: dict[str, Any] = {
    "view_offset_deg": CHARACTER_VIEW_OFFSET_DEGREES,
    "mirror_span_deg": 180.0,
    "mirror_hysteresis_deg": BODY_MIRROR_HYSTERESIS_DEGREES,
    "mirror_min_hold_frames": 2,
    "head_right_mirror_deg": CHARACTER_VIEW_OFFSET_DEGREES,
    "projected_direction_min_ratio": 0.04,
    "source_direction_max_step_deg": 45.0,
    # A raised 3D leg keeps its chosen 2D side.  It may change sides only
    # while near vertical, where the two 2D solutions visually meet.
    "thigh_branch_switch_vertical_cone_deg": 20.0,
    # The supplied Spine character's usable forward-lift branch is opposite
    # the raw camera-X sign: an MMD forward kick must move toward screen right.
    "thigh_camera_side_multiplier": -1.0,
    "body_rotation_limit_deg": 45.0,
    "head_rotation_limit_deg": 60.0,
    "thigh_rotation_limit_deg": 110.0,
    "maximum_knee_flex_deg": 140.0,
    "knee_bend_sign": -1.0,
    "body_translation_gain": 1.0,
}

SECONDARY_MOTION_CONFIG: dict[str, Any] = {
    "enabled": True,
    "mirror_bone": "body",
    "limits": {
        "soft_angle_deg": 24.0,
        "hard_angle_deg": 30.0,
        "input_angular_velocity_deg_per_second": 720.0,
        "input_linear_velocity_px_per_second": 2500.0,
        "physics_substeps": 4,
    },
    "hair": {
        "driver_bone": "head",
        "angular_velocity_gain": 0.1,
        "lateral_velocity_gain": 0.1,
        "spring_frequency_hz": 2.0,
        "damping_ratio": 0.72,
        "max_angular_velocity_deg_per_second": 220.0,
        "twin_tail_cycle": {
            "bones": ["hairTwinF", "hairTwinB"],
            "period_scale": 1.18,
            "difference_mix": 0.35,
            "damping_ratio": 0.68,
        },
        "bones": {
            "hairTwinF": {"amplitude": 1.02, "response": 1.0},
            "hairTwinB": {"amplitude": 1.00, "response": 1.0},
            "hairF": {"amplitude": 0.96, "response": 1.0},
            "hairB": {"amplitude": 0.98, "response": 1.0},
        },
    },
    "skirt": {
        "driver_bone": "body",
        "leg_driver_bones": ["legF", "legB"],
        "leg_motion_gain": 0.035,
        "angular_velocity_gain": 0.4,
        "lateral_velocity_gain": 0.1,
        "spring_frequency_hz": 1.65,
        "damping_ratio": 0.78,
        "max_angular_velocity_deg_per_second": 180.0,
        "bones": ["skirtF", "skirtB"],
    },
}

BLINK_CONFIG: dict[str, Any] = {
    "enabled": True,
    "texture_path": "Texture2D/faceBlink.png",
    "minimum_texture_bytes": 1024,
    "slot": "face",
    "open_attachment": "faceNormal",
    "blink_attachment": "faceBlink",
    "interval_min_seconds": 2.0,
    "interval_max_seconds": 6.0,
    "duration_min_frames": 3,
    "duration_max_frames": 5,
    "seed": 20260916,
}

