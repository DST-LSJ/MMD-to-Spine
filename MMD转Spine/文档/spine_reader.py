"""Reader and light validation for the bundled Spine 4.3 skeleton JSON."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


class SpineFormatError(ValueError):
    """Raised when the template cannot be used as a Spine skeleton."""


@dataclass(frozen=True, slots=True)
class SpineBone:
    name: str
    parent: str | None
    length: float
    x: float
    y: float
    rotation: float
    scale_x: float
    scale_y: float


@dataclass(frozen=True, slots=True)
class SpineSlot:
    name: str
    bone: str
    attachment: str | None
    index: int


@dataclass(slots=True)
class SpineSkeleton:
    path: Path
    raw: dict[str, Any]
    bones: list[SpineBone]
    slots: list[SpineSlot] = field(default_factory=list)

    @property
    def bone_by_name(self) -> dict[str, SpineBone]:
        return {bone.name: bone for bone in self.bones}

    @property
    def version(self) -> str:
        return str(self.raw.get("skeleton", {}).get("spine", ""))

    @property
    def slot_by_name(self) -> dict[str, SpineSlot]:
        return {slot.name: slot for slot in self.slots}

    @property
    def setup_slot_names(self) -> list[str]:
        return [slot.name for slot in self.slots]

    @property
    def bone_parents(self) -> dict[str, str | None]:
        return {bone.name: bone.parent for bone in self.bones}

    def bone_ancestors(self, name: str):
        parents = self.bone_parents
        current: str | None = name
        seen: set[str] = set()
        while current is not None and current not in seen:
            yield current
            seen.add(current)
            current = parents.get(current)


def read_spine_json(path: str | Path) -> SpineSkeleton:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SpineFormatError(f"Could not read Spine JSON {path}: {error}") from error
    if not isinstance(raw, dict) or not isinstance(raw.get("bones"), list):
        raise SpineFormatError("Spine JSON must contain a top-level bones array")

    bones: list[SpineBone] = []
    known: set[str] = set()
    for index, value in enumerate(raw["bones"]):
        if not isinstance(value, dict) or not isinstance(value.get("name"), str):
            raise SpineFormatError(f"Invalid bone entry at index {index}")
        name = value["name"]
        parent = value.get("parent")
        if name in known:
            raise SpineFormatError(f"Duplicate Spine bone name: {name}")
        if parent is not None and parent not in known:
            raise SpineFormatError(
                f"Bone {name!r} references missing or later parent {parent!r}"
            )
        known.add(name)
        bones.append(
            SpineBone(
                name=name,
                parent=parent,
                length=float(value.get("length", 0.0)),
                x=float(value.get("x", 0.0)),
                y=float(value.get("y", 0.0)),
                rotation=float(value.get("rotation", 0.0)),
                scale_x=float(value.get("scaleX", 1.0)),
                scale_y=float(value.get("scaleY", 1.0)),
            )
        )

    version = str(raw.get("skeleton", {}).get("spine", ""))
    if version and not version.startswith("4.3"):
        raise SpineFormatError(
            f"Template reports Spine {version}; this exporter targets Spine 4.3"
        )
    slots: list[SpineSlot] = []
    for index, value in enumerate(raw.get("slots", [])):
        if not isinstance(value, dict):
            raise SpineFormatError(f"Invalid slot entry at index {index}")
        name = value.get("name")
        bone = value.get("bone")
        if not isinstance(name, str) or not isinstance(bone, str):
            raise SpineFormatError(f"Slot {index} requires string name and bone")
        if bone not in known:
            raise SpineFormatError(f"Slot {name!r} references missing bone {bone!r}")
        slots.append(SpineSlot(name, bone, value.get("attachment"), index))
    if len({slot.name for slot in slots}) != len(slots):
        raise SpineFormatError("Duplicate Spine slot name")
    return SpineSkeleton(path, raw, bones, slots)


def skeleton_summary(skeleton: SpineSkeleton) -> dict[str, Any]:
    return {
        "spine_version": skeleton.version,
        "bone_count": len(skeleton.bones),
        "slot_count": len(skeleton.slots),
        "slots": [
            {
                "name": slot.name,
                "bone": slot.bone,
                "attachment": slot.attachment,
                "index": slot.index,
            }
            for slot in skeleton.slots
        ],
        "bones": [
            {
                "name": bone.name,
                "parent": bone.parent,
                "length": bone.length,
                "x": bone.x,
                "y": bone.y,
                "rotation": bone.rotation,
                "scaleX": bone.scale_x,
                "scaleY": bone.scale_y,
            }
            for bone in skeleton.bones
        ],
    }
