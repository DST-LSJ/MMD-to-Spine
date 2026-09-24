"""Spine 4.3 JSON animation writer."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

from .draw_order import decode_draw_order, validate_permutation
from .retarget import RetargetedAnimation, ScalarKey, VectorKey
from .spine_reader import SpineSkeleton


def build_spine_json(
    skeleton: SpineSkeleton,
    animation: RetargetedAnimation,
    animation_name: str,
    preserve_animations: bool = True,
) -> dict[str, Any]:
    if not animation_name.strip():
        raise ValueError("animation_name may not be empty")
    output = copy.deepcopy(skeleton.raw)
    animations = output.setdefault("animations", {}) if preserve_animations else {}
    bone_timelines: dict[str, Any] = {}
    target_order = [bone.name for bone in skeleton.bones]

    for name in target_order:
        bone = animation.bones.get(name)
        if bone is None:
            continue
        timelines: dict[str, Any] = {}
        if bone.rotate:
            timelines["rotate"] = [_rotate_key(key, animation.fps) for key in bone.rotate]
        if bone.translate:
            timelines["translate"] = [
                _vector_key(key, animation.fps) for key in bone.translate
            ]
        if bone.scale:
            timelines["scale"] = [_vector_key(key, animation.fps) for key in bone.scale]
        if timelines:
            bone_timelines[name] = timelines

    generated_animation: dict[str, Any] = {"bones": bone_timelines}
    if animation.draw_order_events:
        draw_order = []
        for event in animation.draw_order_events:
            key: dict[str, Any] = {"time": _number(float(event["time"]))}
            if event.get("offsets"):
                key["offsets"] = [
                    {"slot": value["slot"], "offset": int(value["offset"])}
                    for value in event["offsets"]
                ]
            draw_order.append(key)
        generated_animation["drawOrder"] = draw_order
    animations[animation_name] = generated_animation
    output["animations"] = animations
    return output


def write_spine_json(path: str | Path, data: dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def validate_export(data: dict[str, Any], animation_name: str) -> None:
    if not str(data.get("skeleton", {}).get("spine", "")).startswith("4.3"):
        raise ValueError("Exported skeleton is not marked as Spine 4.3")
    bones = {bone.get("name") for bone in data.get("bones", [])}
    animation = data.get("animations", {}).get(animation_name)
    if not isinstance(animation, dict):
        raise ValueError(f"Exported animation {animation_name!r} is missing")
    constraints = {c["name"]: c for c in data.get("constraints", [])}
    for constraint in constraints.values():
        if constraint.get("type") != "ik":
            continue
        if constraint.get("target") not in bones:
            raise ValueError("IK target bone is missing")
        if len(constraint.get("bones", [])) not in (1, 2) or any(b not in bones for b in constraint["bones"]):
            raise ValueError("Invalid IK bone chain")
    for name, keys in animation.get("ik", {}).items():
        if name not in constraints or constraints[name].get("type") != "ik":
            raise ValueError(f"Unknown Spine 4.3 IK constraint: {name}")
        previous_time = -math.inf
        for key in keys:
            time = float(key.get("time", 0))
            if not math.isfinite(time) or time < previous_time:
                raise ValueError("Invalid IK timeline time")
            previous_time = time
            for field in ("bendPositive", "stretch", "compress"):
                if field in key and not isinstance(key[field], bool):
                    raise ValueError(f"IK {field} must be boolean")
    for bone_name, timelines in animation.get("bones", {}).items():
        if bone_name not in bones:
            raise ValueError(f"Animation references unknown bone {bone_name!r}")
        for timeline_name, keys in timelines.items():
            if timeline_name not in {"rotate", "translate", "scale"}:
                raise ValueError(f"Unsupported generated timeline {timeline_name!r}")
            previous_time = -math.inf
            for key in keys:
                time = float(key.get("time", 0.0))
                if time < previous_time:
                    raise ValueError(f"Timeline {bone_name}/{timeline_name} is unsorted")
                if not all(
                    math.isfinite(float(value))
                    for field, value in key.items()
                    if field in {"time", "value", "x", "y"}
                ):
                    raise ValueError(f"Timeline {bone_name}/{timeline_name} has NaN/Inf")
                previous_time = time
    setup_slots = [slot.get("name") for slot in data.get("slots", [])]
    if any(not isinstance(name, str) for name in setup_slots):
        raise ValueError("Exported skeleton contains an invalid slot name")
    previous_time = -math.inf
    for event in animation.get("drawOrder", []):
        time = float(event.get("time", 0.0))
        if time < previous_time:
            raise ValueError("drawOrder events are not time-sorted")
        offsets = event.get("offsets")
        if offsets is not None:
            decoded = decode_draw_order(setup_slots, offsets)
            validate_permutation(setup_slots, decoded)
        previous_time = time


def _rotate_key(key: ScalarKey, fps: float) -> dict[str, Any]:
    result: dict[str, Any] = {
        "time": _number(key.frame / fps),
        "value": _number(key.value),
    }
    if key.curve is not None:
        result["curve"] = key.curve
    return result


def _vector_key(key: VectorKey, fps: float) -> dict[str, Any]:
    result: dict[str, Any] = {
        "time": _number(key.frame / fps),
        "x": _number(key.x),
        "y": _number(key.y),
    }
    if key.curve is not None:
        result["curve"] = key.curve
    return result


def _number(value: float) -> float:
    rounded = round(float(value), 6)
    return 0.0 if rounded == -0.0 else rounded
