"""Minimal, dependency-free VMD 0001/0002 motion reader.

The converter currently consumes bone keyframes, but the reader also walks the
remaining standard VMD sections so malformed/truncated files produce useful
errors instead of silently returning partial data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from bisect import bisect_right
import io
import math
from pathlib import Path
import struct
from typing import Any, BinaryIO, Iterable


class VMDFormatError(ValueError):
    """Raised when a file is not a supported or complete VMD file."""


@dataclass(frozen=True, slots=True)
class BoneKeyframe:
    bone_name: str
    frame: int
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]
    interpolation: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class IKState:
    """One named IK enable/disable value from a VMD display frame."""

    name: str
    enabled: bool


@dataclass(frozen=True, slots=True)
class IKDisplayKeyframe:
    """Model visibility and IK states recorded at one source frame."""

    frame: int
    model_visible: bool
    states: tuple[IKState, ...]


@dataclass(slots=True)
class VMDMotion:
    model_name: str
    bone_tracks: dict[str, list[BoneKeyframe]]
    source_bone_keyframes: int
    source_max_frame: int = 0
    morph_keyframes: int = 0
    camera_keyframes: int = 0
    light_keyframes: int = 0
    shadow_keyframes: int = 0
    ik_keyframes: int = 0
    ik_display_frames: list[IKDisplayKeyframe] = field(default_factory=list)

    @property
    def max_frame(self) -> int:
        return max(
            (key.frame for track in self.bone_tracks.values() for key in track),
            default=0,
        )

    @property
    def bone_keyframes(self) -> int:
        return sum(len(track) for track in self.bone_tracks.values())


class _BinaryReader:
    def __init__(self, stream: BinaryIO):
        self.stream = stream

    def read(self, size: int, label: str) -> bytes:
        data = self.stream.read(size)
        if len(data) != size:
            raise VMDFormatError(
                f"VMD ended while reading {label}: wanted {size} bytes, "
                f"received {len(data)}"
            )
        return data

    def unpack(self, fmt: str, label: str):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.read(size, label))

    def uint32(self, label: str) -> int:
        return self.unpack("<I", label)[0]

    def remaining(self) -> int | None:
        try:
            current = self.stream.tell()
            self.stream.seek(0, io.SEEK_END)
            end = self.stream.tell()
            self.stream.seek(current)
            return end - current
        except (AttributeError, OSError):
            return None


def _decode_text(raw: bytes) -> str:
    raw = raw.split(b"\0", 1)[0]
    return raw.decode("cp932", errors="replace").strip()


def _normalise_quaternion(
    value: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    length = math.sqrt(sum(component * component for component in value))
    if not math.isfinite(length) or length < 1e-8:
        return (0.0, 0.0, 0.0, 1.0)
    return tuple(component / length for component in value)  # type: ignore[return-value]


def _validate_count(count: int, remaining: int | None, record_size: int, label: str) -> None:
    if count > 20_000_000:
        raise VMDFormatError(f"Unreasonable {label} count: {count}")
    if remaining is not None and count * record_size > remaining:
        raise VMDFormatError(
            f"{label} count {count} does not fit in the remaining VMD data"
        )


def read_vmd(path: str | Path, max_frame: int | None = None) -> VMDMotion:
    """Read bone animation from *path*.

    ``max_frame`` filters returned bone keys but does not stop parsing the file,
    which keeps validation and section counts correct.
    """

    with Path(path).open("rb") as stream:
        return read_vmd_stream(stream, max_frame=max_frame)


def read_vmd_stream(stream: BinaryIO, max_frame: int | None = None) -> VMDMotion:
    reader = _BinaryReader(stream)
    header_raw = reader.read(30, "header")
    header = header_raw.split(b"\0", 1)[0].decode("ascii", errors="replace")
    if not header.startswith("Vocaloid Motion Data"):
        raise VMDFormatError(f"Not a VMD file (header was {header!r})")

    model_name_length = 10 if "0001" in header else 20
    model_name = _decode_text(reader.read(model_name_length, "model name"))

    source_bone_keyframes = reader.uint32("bone keyframe count")
    _validate_count(source_bone_keyframes, reader.remaining(), 111, "bone keyframe")
    deduplicated: dict[str, dict[int, BoneKeyframe]] = {}
    source_max_frame = 0

    for index in range(source_bone_keyframes):
        name = _decode_text(reader.read(15, f"bone keyframe {index} name"))
        frame = reader.uint32(f"bone keyframe {index} frame")
        source_max_frame = max(source_max_frame, frame)
        position = reader.unpack("<3f", f"bone keyframe {index} position")
        rotation = _normalise_quaternion(
            reader.unpack("<4f", f"bone keyframe {index} rotation")
        )
        interpolation = reader.read(64, f"bone keyframe {index} interpolation")
        if max_frame is None or frame <= max_frame:
            key = BoneKeyframe(name, frame, position, rotation, interpolation)
            deduplicated.setdefault(name, {})[frame] = key

    tracks = {
        name: [frames[frame] for frame in sorted(frames)]
        for name, frames in deduplicated.items()
    }
    motion = VMDMotion(
        model_name,
        tracks,
        source_bone_keyframes,
        source_max_frame=source_max_frame,
    )
    _read_optional_sections(reader, motion)
    return motion


def _read_optional_sections(reader: _BinaryReader, motion: VMDMotion) -> None:
    """Walk standard non-bone sections, tolerating legal early EOF variants."""

    if reader.remaining() == 0:
        return
    motion.morph_keyframes = reader.uint32("morph keyframe count")
    _validate_count(motion.morph_keyframes, reader.remaining(), 23, "morph keyframe")
    reader.read(motion.morph_keyframes * 23, "morph keyframes")

    if reader.remaining() == 0:
        return
    motion.camera_keyframes = reader.uint32("camera keyframe count")
    _validate_count(motion.camera_keyframes, reader.remaining(), 61, "camera keyframe")
    reader.read(motion.camera_keyframes * 61, "camera keyframes")

    if reader.remaining() == 0:
        return
    motion.light_keyframes = reader.uint32("light keyframe count")
    _validate_count(motion.light_keyframes, reader.remaining(), 28, "light keyframe")
    reader.read(motion.light_keyframes * 28, "light keyframes")

    if reader.remaining() == 0:
        return
    motion.shadow_keyframes = reader.uint32("self-shadow keyframe count")
    _validate_count(motion.shadow_keyframes, reader.remaining(), 9, "self-shadow keyframe")
    reader.read(motion.shadow_keyframes * 9, "self-shadow keyframes")

    if reader.remaining() == 0:
        return
    display_count = reader.uint32("IK display keyframe count")
    if display_count > 10_000_000:
        raise VMDFormatError(f"Unreasonable IK display keyframe count: {display_count}")
    total_ik = 0
    display_frames: list[IKDisplayKeyframe] = []
    for index in range(display_count):
        frame = reader.uint32(f"IK display keyframe {index} frame")
        model_visible = bool(
            reader.unpack("<B", f"IK display keyframe {index} visibility")[0]
        )
        ik_count = reader.uint32(f"IK display keyframe {index} IK count")
        _validate_count(ik_count, reader.remaining(), 21, "IK state")
        states: list[IKState] = []
        for state_index in range(ik_count):
            name = _decode_text(
                reader.read(
                    20,
                    f"IK display keyframe {index} state {state_index} name",
                )
            )
            enabled = bool(
                reader.unpack(
                    "<B",
                    f"IK display keyframe {index} state {state_index} enabled",
                )[0]
            )
            states.append(IKState(name, enabled))
        display_frames.append(
            IKDisplayKeyframe(frame, model_visible, tuple(states))
        )
        total_ik += ik_count
    motion.ik_keyframes = total_ik
    motion.ik_display_frames = sorted(display_frames, key=lambda value: value.frame)


def _bezier_coordinate(t: float, p1: float, p2: float) -> float:
    inverse = 1.0 - t
    return 3.0 * inverse * inverse * t * p1 + 3.0 * inverse * t * t * p2 + t**3


def interpolation_amount(interpolation: bytes, channel: int, amount: float) -> float:
    """Evaluate a VMD Bezier channel (0=x, 1=y, 2=z, 3=rotation)."""

    amount = min(1.0, max(0.0, amount))
    if len(interpolation) < 16:
        return amount
    x1 = min(1.0, interpolation[channel] / 127.0)
    y1 = min(1.0, interpolation[channel + 4] / 127.0)
    x2 = min(1.0, interpolation[channel + 8] / 127.0)
    y2 = min(1.0, interpolation[channel + 12] / 127.0)
    low, high = 0.0, 1.0
    for _ in range(14):
        midpoint = (low + high) * 0.5
        if _bezier_coordinate(midpoint, x1, x2) < amount:
            low = midpoint
        else:
            high = midpoint
    return _bezier_coordinate((low + high) * 0.5, y1, y2)


IDENTITY_KEY = BoneKeyframe(
    "",
    0,
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
    bytes([20, 20, 20, 20, 20, 20, 20, 20, 107, 107, 107, 107, 107, 107, 107, 107])
    + bytes(48),
)


def _slerp(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
    amount: float,
) -> tuple[float, float, float, float]:
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
    first_weight = math.sin((1.0 - amount) * theta) / sin_theta
    second_weight = math.sin(amount * theta) / sin_theta
    return tuple(
        first_weight * a + second_weight * b for a, b in zip(first, second)
    )  # type: ignore[return-value]


def sample_track(track: list[BoneKeyframe], frame: int) -> BoneKeyframe:
    """Sample a bone track using VMD position Beziers and quaternion slerp."""

    if not track:
        return IDENTITY_KEY
    frames = [key.frame for key in track]
    position = bisect_right(frames, frame)
    if position == 0:
        return IDENTITY_KEY if frame < track[0].frame else track[0]
    if position >= len(track):
        return track[-1]
    first, second = track[position - 1], track[position]
    if first.frame == frame:
        return first
    span = second.frame - first.frame
    if span <= 0:
        return second
    raw_amount = (frame - first.frame) / span
    amounts = [
        interpolation_amount(first.interpolation, channel, raw_amount)
        for channel in range(4)
    ]
    sampled_position = tuple(
        first.position[index]
        + (second.position[index] - first.position[index]) * amounts[index]
        for index in range(3)
    )
    sampled_rotation = _slerp(first.rotation, second.rotation, amounts[3])
    return BoneKeyframe(
        first.bone_name,
        frame,
        sampled_position,  # type: ignore[arg-type]
        sampled_rotation,
        first.interpolation,
    )


def all_keyframes(motion: VMDMotion) -> Iterable[BoneKeyframe]:
    for track in motion.bone_tracks.values():
        yield from track


def motion_to_dict(
    motion: VMDMotion,
    start_frame: int = 0,
    end_frame: int | None = None,
    rebase: bool = False,
) -> dict[str, Any]:
    """Return a normalized, JSON-serializable motion dump for diagnostics."""

    end_frame = motion.max_frame if end_frame is None else end_frame
    selected_tracks = {
        name: [key for key in track if start_frame <= key.frame <= end_frame]
        for name, track in motion.bone_tracks.items()
    }
    selected_tracks = {name: track for name, track in selected_tracks.items() if track}
    selected_key_count = sum(len(track) for track in selected_tracks.values())

    def output_frame(frame: int) -> int:
        return frame - start_frame if rebase else frame

    return {
        "model_name": motion.model_name,
        "fps": 30,
        "source_start_frame": start_frame,
        "source_end_frame": end_frame,
        "max_frame": end_frame - start_frame if rebase else end_frame,
        "bone_count": len(selected_tracks),
        "bone_keyframes": selected_key_count,
        "source_bone_keyframes": motion.source_bone_keyframes,
        "other_sections": {
            "morph_keyframes": motion.morph_keyframes,
            "camera_keyframes": motion.camera_keyframes,
            "light_keyframes": motion.light_keyframes,
            "shadow_keyframes": motion.shadow_keyframes,
            "ik_states": motion.ik_keyframes,
            "ik_display_frames": [
                {
                    "frame": value.frame,
                    "model_visible": value.model_visible,
                    "states": [
                        {"name": state.name, "enabled": state.enabled}
                        for state in value.states
                    ],
                }
                for value in motion.ik_display_frames
                if start_frame <= value.frame <= end_frame
            ],
        },
        "bones": {
            name: [
                {
                    "frame": output_frame(key.frame),
                    "source_frame": key.frame,
                    "time": round(output_frame(key.frame) / 30.0, 7),
                    "position": [round(value, 7) for value in key.position],
                    "rotation_xyzw": [round(value, 9) for value in key.rotation],
                    "interpolation_hex": key.interpolation.hex(),
                }
                for key in track
            ]
            for name, track in sorted(selected_tracks.items())
        },
    }
