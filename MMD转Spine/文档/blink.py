"""Independent procedural blink timeline for the exported Spine animation."""

from __future__ import annotations

import math
from pathlib import Path
import random
from typing import Any

from .spine_reader import SpineSkeleton


def add_blink_timeline(
    data: dict[str, Any],
    animation_name: str,
    skeleton: SpineSkeleton,
    texture_path: str | Path,
    fps: float,
    max_frame: int,
    config: dict[str, Any],
) -> int:
    """Add deterministic random blinks, or do nothing for an empty texture."""

    if not bool(config.get("enabled", True)):
        return 0
    texture = Path(texture_path)
    minimum_size = int(config.get("minimum_texture_bytes", 1024))
    if not texture.is_file() or texture.stat().st_size < minimum_size:
        return 0

    slot_name = str(config.get("slot", "face"))
    slot = skeleton.slot_by_name.get(slot_name)
    if slot is None:
        return 0
    open_attachment = str(
        config.get("open_attachment") or slot.attachment or "face"
    )
    blink_attachment = str(config.get("blink_attachment", "faceBlink"))
    minimum_gap = max(0, math.ceil(float(config.get("interval_min_seconds", 2.0)) * fps))
    maximum_gap = max(minimum_gap, math.floor(float(config.get("interval_max_seconds", 6.0)) * fps))
    minimum_frames = max(1, int(config.get("duration_min_frames", 3)))
    maximum_frames = max(minimum_frames, int(config.get("duration_max_frames", 5)))
    randomizer = random.Random(int(config.get("seed", 20260916)))

    timeline: list[dict[str, Any]] = []
    cursor = 0
    while True:
        start = cursor + randomizer.randint(minimum_gap, maximum_gap)
        duration = randomizer.randint(minimum_frames, maximum_frames)
        end = start + duration
        if end > max_frame:
            break
        timeline.append({"time": _time(start, fps), "name": blink_attachment})
        timeline.append({"time": _time(end, fps), "name": open_attachment})
        cursor = end

    if timeline:
        animation = data["animations"][animation_name]
        animation.setdefault("slots", {}).setdefault(slot_name, {})[
            "attachment"
        ] = timeline
    return len(timeline) // 2


def _time(frame: int, fps: float) -> float:
    value = round(frame / fps, 6)
    return 0.0 if value == -0.0 else value
