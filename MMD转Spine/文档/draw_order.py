"""Dynamic Spine draw-order generation from 3D proxy depth samples."""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .projection import Projector
from .retarget import BoneAnimation, RetargetedAnimation, ScalarKey, VectorKey
from .source_pose import PoseSample, pose_sample_to_dict
from .spine_reader import SpineSkeleton


@dataclass(frozen=True, slots=True)
class ResolvedSlotGroup:
    name: str
    anchor_bone: str
    segment: str
    slots: tuple[str, ...]


@dataclass(slots=True)
class DrawOrderResult:
    events: list[dict[str, Any]]
    depth_samples: list[dict[str, Any]]
    groups: dict[str, ResolvedSlotGroup]
    warnings: list[str] = field(default_factory=list)
    dropped_relations: list[dict[str, Any]] = field(default_factory=list)
    switch_count: int = 0


@dataclass(frozen=True, slots=True)
class Segment2D:
    start: tuple[float, float]
    end: tuple[float, float]


@dataclass(frozen=True, slots=True)
class ProjectedGroup:
    name: str
    target: Segment2D
    depth_start: float
    depth_end: float
    confidence: float

    @property
    def representative_depth(self) -> float:
        return (self.depth_start + self.depth_end) * 0.5


class RelationTracker:
    def __init__(self, initial_a_before_b: bool, minimum_duration: float):
        self.current = initial_a_before_b
        self.minimum_duration = max(0.0, minimum_duration)
        self.pending: bool | None = None
        self.pending_since = 0.0

    def update(self, desired: bool | None, time_seconds: float) -> bool:
        if desired is None or desired == self.current:
            self.pending = None
            return False
        if self.pending != desired:
            self.pending = desired
            self.pending_since = time_seconds
            if self.minimum_duration > 0:
                return False
        if time_seconds - self.pending_since + 1e-9 >= self.minimum_duration:
            self.current = desired
            self.pending = None
            return True
        return False


def load_draw_order_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        config = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read draw-order config {path}: {error}") from error
    if config.get("version") != 1 or not isinstance(config.get("groups"), dict):
        raise ValueError("draw-order config requires version 1 and a groups object")
    if config.get("depth_source") != "canonical_rig_approximation":
        raise ValueError("Unsupported P0 depth source")
    return config


def load_manual_overrides(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"version": 1, "time_space": "output", "overrides": []}
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if value.get("version") != 1 or not isinstance(value.get("overrides"), list):
        raise ValueError("draw-order overrides require version 1 and an overrides array")
    for override in value["overrides"]:
        if not all(key in override for key in ("start", "end", "front", "behind")):
            raise ValueError("Each draw-order override requires start/end/front/behind")
        if float(override["end"]) <= float(override["start"]):
            raise ValueError("Draw-order override end must be greater than start")
    return value


def resolve_slot_groups(
    skeleton: SpineSkeleton, config: dict[str, Any]
) -> dict[str, ResolvedSlotGroup]:
    rules: dict[str, dict[str, Any]] = config["groups"]
    anchors: dict[str, str] = {}
    for name, rule in rules.items():
        anchor = rule.get("anchor_bone")
        if anchor not in skeleton.bone_by_name:
            raise ValueError(f"Draw-order group {name!r} has missing anchor bone {anchor!r}")
        if anchor in anchors:
            raise ValueError(f"Bone {anchor!r} anchors more than one draw-order group")
        anchors[anchor] = name

    overrides: dict[str, str] = config.get("slot_owner_overrides", {})
    members: dict[str, list[str]] = {name: [] for name in rules}
    claimed: dict[str, str] = {}
    for slot in skeleton.slots:
        owner = overrides.get(slot.name)
        if owner is None:
            for ancestor in skeleton.bone_ancestors(slot.bone):
                if ancestor in anchors:
                    owner = anchors[ancestor]
                    break
        if owner is None:
            continue
        if owner not in rules:
            raise ValueError(f"Slot {slot.name!r} override names unknown group {owner!r}")
        excluded = set(rules[owner].get("exclude_slots", []))
        if slot.name in excluded:
            continue
        if slot.name in claimed:
            raise ValueError(f"Slot {slot.name!r} belongs to multiple draw-order groups")
        claimed[slot.name] = owner
        members[owner].append(slot.name)

    for name, rule in rules.items():
        for slot_name in rule.get("include_slots", []):
            if slot_name not in skeleton.slot_by_name:
                raise ValueError(f"Group {name!r} explicitly includes missing slot {slot_name!r}")
            previous = claimed.get(slot_name)
            if previous is not None and previous != name:
                raise ValueError(f"Slot {slot_name!r} is claimed by {previous!r} and {name!r}")
            if slot_name not in members[name]:
                members[name].append(slot_name)
                claimed[slot_name] = name

    setup_index = {slot.name: slot.index for slot in skeleton.slots}
    result: dict[str, ResolvedSlotGroup] = {}
    for name, rule in rules.items():
        ordered = sorted(members[name], key=setup_index.__getitem__)
        internal = rule.get("internal_order", [])
        if internal:
            missing = [slot for slot in internal if slot not in ordered]
            if missing:
                raise ValueError(f"Group {name!r} internal_order has non-member slots: {missing}")
            ordered = list(internal) + [slot for slot in ordered if slot not in internal]
        result[name] = ResolvedSlotGroup(
            name,
            str(rule["anchor_bone"]),
            str(rule.get("segment", name)),
            tuple(ordered),
        )
    return result


def generate_draw_order(
    skeleton: SpineSkeleton,
    animation: RetargetedAnimation,
    pose_samples: list[PoseSample],
    projector: Projector,
    config: dict[str, Any],
    overrides: dict[str, Any] | None = None,
) -> DrawOrderResult:
    if not bool(config.get("enabled", True)):
        return DrawOrderResult([], [], {})
    groups = resolve_slot_groups(skeleton, config)
    setup_slots = skeleton.setup_slot_names
    slot_to_group = {
        slot: group.name for group in groups.values() for slot in group.slots
    }
    items = _setup_items(setup_slots, slot_to_group)
    setup_item_index = {item: index for index, item in enumerate(items)}
    previous_items = items[:]
    previous_slots: list[str] | None = None
    events: list[dict[str, Any]] = []
    depth_samples: list[dict[str, Any]] = []
    warnings: list[str] = []
    dropped: list[dict[str, Any]] = []
    minimum_duration = float(config.get("min_switch_duration_seconds", 0.0))
    epsilon = abs(float(config.get("depth_epsilon_body_height", 0.015)))
    body_height = float(config.get("body_height", 0.0) or 1.0)
    body_reference = config.get("body_reference", {})
    body_slot = str(body_reference.get("slot", "body"))
    body_item = f"slot:{body_slot}"
    if body_item not in items:
        raise ValueError(f"Draw-order body reference slot {body_slot!r} is missing or managed")
    trackers: dict[tuple[str, str], RelationTracker] = {}
    static_groups = set(str(name) for name in config.get("static_groups", []))
    setup_slot_index = {name: index for index, name in enumerate(setup_slots)}

    def initial_relation(first: str, second: str) -> bool:
        first_slot = groups[first].slots[0] if first in groups else first
        second_slot = groups[second].slots[0] if second in groups else second
        return setup_slot_index[first_slot] < setup_slot_index[second_slot]

    for name in groups:
        if name in static_groups:
            continue
        trackers[(name, body_slot)] = RelationTracker(
            initial_relation(name, body_slot), minimum_duration
        )
    for pair in config.get("compare_pairs", []):
        if len(pair) == 2 and pair[0] in groups and pair[1] in groups:
            trackers[(pair[0], pair[1])] = RelationTracker(
                initial_relation(pair[0], pair[1]), minimum_duration
            )

    overrides = overrides or {"time_space": "output", "overrides": []}
    switch_count = 0
    for sample in pose_samples:
        target_segments, body_rect = _target_geometry(
            skeleton, animation, sample.output_frame, groups, body_reference
        )
        projected = _projected_groups(sample, groups, target_segments, projector, warnings)
        body_segment = sample.segments.get("body")
        body_depth = None
        if body_segment is not None:
            first = projector.project_point(body_segment.start)
            second = projector.project_point(body_segment.end)
            body_depth = (first.camera_depth + second.camera_depth) * 0.5

        relation_details: list[dict[str, Any]] = []
        for name, group in projected.items():
            tracker = trackers.get((name, body_slot))
            if tracker is None:
                continue
            desired: bool | None = None
            delta = None
            overlap = _segment_box_overlap(group.target, body_rect, float(config.get("overlap_margin_px", 18.0)))
            if overlap and body_depth is not None:
                delta = (group.representative_depth - body_depth) / body_height
                desired = _desired_before(delta, epsilon)
            if tracker.update(desired, sample.time_seconds):
                switch_count += 1
            relation_details.append({
                "first": name,
                "second": body_slot,
                "a_before_b": tracker.current,
                "raw_depth_delta_normalized": delta,
                "overlap": overlap,
                "mode": "segment_vs_body_proxy",
            })

        for pair in config.get("compare_pairs", []):
            if len(pair) != 2 or tuple(pair) not in trackers:
                continue
            first_name, second_name = pair
            tracker = trackers[(first_name, second_name)]
            first = projected.get(first_name)
            second = projected.get(second_name)
            desired = None
            delta = None
            mode = "missing_depth"
            active_overlap = False
            if first is not None and second is not None:
                comparison = compare_projected_segments(
                    first,
                    second,
                    body_height,
                    float(config.get("overlap_margin_px", 18.0)),
                )
                if comparison is not None:
                    delta, mode = comparison
                    desired = _desired_before(delta, epsilon)
                    active_overlap = True
                else:
                    mode = "no_target_overlap_keep_previous"
            if tracker.update(desired, sample.time_seconds):
                switch_count += 1
            relation_details.append({
                "first": first_name,
                "second": second_name,
                "a_before_b": tracker.current,
                "raw_depth_delta_normalized": delta,
                "overlap": active_overlap,
                "mode": mode,
            })

        edges: set[tuple[str, str]] = set()
        unmanaged = [item for item in items if item.startswith("slot:")]
        edges.update(zip(unmanaged, unmanaged[1:]))
        for constraint in config.get("hard_constraints", []):
            before = str(constraint["before"])
            after = str(constraint["after"])
            _add_hard_edge(items, edges, (before, after))
        manual_edges = _active_manual_edges(
            overrides, sample, groups, body_slot, animation.source_start_frame, animation.fps
        )
        for edge in manual_edges:
            _add_hard_edge(items, edges, edge)

        automatic_edge_bundles: list[tuple[list[tuple[str, str]], dict[str, Any]]] = []
        body_occlusion = config.get("body_occlusion", {})
        body_occlusion_groups = set(body_occlusion.get("groups", []))
        body_occlusion_slots = [
            str(slot) for slot in body_occlusion.get("slots", [body_slot])
        ]
        detail_lookup = {(d["first"], d["second"]): d for d in relation_details}
        for (first_name, second_name), tracker in trackers.items():
            first_item = f"group:{first_name}"
            second_item = body_item if second_name == body_slot else f"group:{second_name}"
            detail = detail_lookup[(first_name, second_name)]
            if not detail["overlap"]:
                continue
            if second_name == body_slot and first_name in body_occlusion_groups:
                related_items = [f"slot:{slot}" for slot in body_occlusion_slots]
                candidate_edges = [
                    (first_item, item) if tracker.current else (item, first_item)
                    for item in related_items
                ]
            else:
                candidate_edges = [
                    (first_item, second_item)
                    if tracker.current
                    else (second_item, first_item)
                ]
            automatic_edge_bundles.append((candidate_edges, detail))
        for candidate_edges, detail in automatic_edge_bundles:
            trial_edges = set(edges)
            has_cycle = False
            for before, after in candidate_edges:
                if before not in items or after not in items:
                    raise ValueError(
                        f"Body occlusion relation references missing item: {(before, after)}"
                    )
                if _would_cycle(items, trial_edges, before, after):
                    has_cycle = True
                    break
                trial_edges.add((before, after))
            if has_cycle:
                dropped.append({
                    "output_frame": sample.output_frame,
                    "time_seconds": sample.time_seconds,
                    "edges": candidate_edges,
                    "reason": "low_confidence_automatic_cycle",
                    "relation": detail,
                })
            else:
                edges = trial_edges

        priority = {item: index for index, item in enumerate(previous_items)}
        for item, index in setup_item_index.items():
            priority.setdefault(item, len(items) + index)
        ordered_items = stable_topological_order(items, edges, priority)
        permutation = _flatten_items(ordered_items, groups)
        validate_permutation(setup_slots, permutation)
        offsets = encode_draw_order(setup_slots, permutation)
        if decode_draw_order(setup_slots, offsets) != permutation:
            raise ValueError("Internal draw-order encode/decode mismatch")

        sample_record = pose_sample_to_dict(sample, projector)
        for name, projected_group in projected.items():
            source_record = sample_record["groups"].get(groups[name].segment, {})
            group_record = dict(source_record)
            group_record["target_screen_start"] = list(projected_group.target.start)
            group_record["target_screen_end"] = list(projected_group.target.end)
            sample_record["groups"][name] = group_record
        sample_record["relations"] = relation_details
        depth_samples.append(sample_record)

        if previous_slots is None or permutation != previous_slots:
            events.append({
                "frame": sample.output_frame,
                "time": sample.output_frame / animation.fps,
                "source_frame": sample.source_frame,
                "source_time": sample.source_frame / animation.fps,
                "offsets": offsets,
                "permutation": permutation,
                "relations": relation_details,
                "manual_override": bool(manual_edges),
            })
        previous_slots = permutation
        previous_items = ordered_items

    return DrawOrderResult(events, depth_samples, groups, warnings, dropped, switch_count)


def encode_draw_order(setup: list[str], permutation: list[str]) -> list[dict[str, Any]]:
    validate_permutation(setup, permutation)
    target_index = {slot: index for index, slot in enumerate(permutation)}
    return [
        {"slot": slot, "offset": target_index[slot] - index}
        for index, slot in enumerate(setup)
        if target_index[slot] != index
    ]


def decode_draw_order(setup: list[str], offsets: list[dict[str, Any]] | None) -> list[str]:
    if not offsets:
        return setup[:]
    index_by_slot = {slot: index for index, slot in enumerate(setup)}
    ordered_offsets = sorted(offsets, key=lambda value: index_by_slot[value["slot"]])
    draw_order: list[int | None] = [None] * len(setup)
    unchanged: list[int] = []
    original_index = 0
    for value in ordered_offsets:
        index = index_by_slot.get(value["slot"])
        if index is None:
            raise ValueError(f"Unknown draw-order slot {value['slot']!r}")
        while original_index != index:
            unchanged.append(original_index)
            original_index += 1
        destination = original_index + int(value["offset"])
        if destination < 0 or destination >= len(setup) or draw_order[destination] is not None:
            raise ValueError("Invalid or colliding draw-order offset")
        draw_order[destination] = original_index
        original_index += 1
    while original_index < len(setup):
        unchanged.append(original_index)
        original_index += 1
    for index in range(len(setup) - 1, -1, -1):
        if draw_order[index] is None:
            if not unchanged:
                raise ValueError("Draw-order offsets leave no unchanged slot to fill")
            draw_order[index] = unchanged.pop()
    return [setup[index] for index in draw_order if index is not None]


def validate_permutation(setup: list[str], permutation: list[str]) -> None:
    if len(permutation) != len(setup) or set(permutation) != set(setup):
        raise ValueError("Draw-order permutation must contain every setup slot exactly once")


def segment_intersection_parameters(
    first: Segment2D, second: Segment2D, epsilon: float = 1e-8
) -> tuple[float, float] | None:
    px, py = first.start
    rx, ry = first.end[0] - px, first.end[1] - py
    qx, qy = second.start
    sx, sy = second.end[0] - qx, second.end[1] - qy
    denominator = rx * sy - ry * sx
    if abs(denominator) < epsilon:
        return None
    qpx, qpy = qx - px, qy - py
    first_t = (qpx * sy - qpy * sx) / denominator
    second_t = (qpx * ry - qpy * rx) / denominator
    if -epsilon <= first_t <= 1 + epsilon and -epsilon <= second_t <= 1 + epsilon:
        return min(1.0, max(0.0, first_t)), min(1.0, max(0.0, second_t))
    return None


def compare_projected_segments(
    first: ProjectedGroup,
    second: ProjectedGroup,
    body_height: float,
    overlap_margin: float = 0.0,
) -> tuple[float, str] | None:
    intersection = segment_intersection_parameters(first.target, second.target)
    if intersection is not None:
        first_t, second_t = intersection
        first_depth = _lerp(first.depth_start, first.depth_end, first_t)
        second_depth = _lerp(second.depth_start, second.depth_end, second_t)
        return (first_depth - second_depth) / body_height, "target_segment_intersection"
    if _segment_bbox_overlap(first.target, second.target, overlap_margin):
        return (
            (first.representative_depth - second.representative_depth) / body_height,
            "expanded_bbox_midpoint_fallback",
        )
    return None


def stable_topological_order(
    nodes: list[str], edges: Iterable[tuple[str, str]], priority: dict[str, int]
) -> list[str]:
    outgoing = {node: set() for node in nodes}
    indegree = {node: 0 for node in nodes}
    for before, after in edges:
        if before == after or before not in outgoing or after not in outgoing:
            continue
        if after not in outgoing[before]:
            outgoing[before].add(after)
            indegree[after] += 1
    ready = [(priority.get(node, len(nodes)), node) for node in nodes if indegree[node] == 0]
    heapq.heapify(ready)
    result: list[str] = []
    while ready:
        _, node = heapq.heappop(ready)
        result.append(node)
        for child in sorted(outgoing[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                heapq.heappush(ready, (priority.get(child, len(nodes)), child))
    if len(result) != len(nodes):
        raise ValueError("Hard draw-order constraints contain a cycle")
    return result


def _target_geometry(
    skeleton: SpineSkeleton,
    animation: RetargetedAnimation,
    frame: int,
    groups: dict[str, ResolvedSlotGroup],
    body_reference: dict[str, Any],
) -> tuple[dict[str, Segment2D], tuple[float, float, float, float]]:
    transforms = _bone_world_transforms(skeleton, animation, frame)
    segment_bones = {
        "whole_arm_F": ("armF", "handF", True),
        "upper_arm_F": ("armF", "handF", False),
        "forearm_hand_F": ("handF", None, True),
        "whole_arm_B": ("armB", "handB", True),
        "upper_arm_B": ("armB", "handB", False),
        "forearm_hand_B": ("handB", None, True),
        "whole_leg_F": ("thighF", "legF", True),
        "upper_leg_F": ("thighF", "legF", False),
        "lower_leg_F": ("legF", None, True),
        "whole_leg_B": ("thighB", "legB", True),
        "upper_leg_B": ("thighB", "legB", False),
        "lower_leg_B": ("legB", None, True),
    }
    segments: dict[str, Segment2D] = {}
    for name in groups:
        start_bone, end_bone, extend_end = segment_bones.get(
            groups[name].segment, (groups[name].anchor_bone, None, True)
        )
        start = transforms[start_bone]
        if end_bone is not None:
            end = transforms[end_bone]
            if extend_end:
                setup = skeleton.bone_by_name[end_bone]
                length = setup.length if setup.length > 0 else 100.0
                endpoint = (
                    end[0] + math.cos(math.radians(end[2])) * length * end[3],
                    end[1] + math.sin(math.radians(end[2])) * length * end[4],
                )
            else:
                endpoint = (end[0], end[1])
        else:
            setup = skeleton.bone_by_name[start_bone]
            length = setup.length if setup.length > 0 else 100.0
            endpoint = (
                start[0] + math.cos(math.radians(start[2])) * length * start[3],
                start[1] + math.sin(math.radians(start[2])) * length * start[4],
            )
        segments[name] = Segment2D((start[0], start[1]), endpoint)
    body = transforms[str(body_reference.get("bone", "body"))]
    offset = body_reference.get("center_offset", [0.0, 55.0])
    center_x = body[0] + float(offset[0])
    center_y = body[1] + float(offset[1])
    half_width = float(body_reference.get("width", 190.0)) * 0.5
    half_height = float(body_reference.get("height", 250.0)) * 0.5
    return segments, (
        center_x - half_width,
        center_y - half_height,
        center_x + half_width,
        center_y + half_height,
    )


def _bone_world_transforms(
    skeleton: SpineSkeleton, animation: RetargetedAnimation, frame: int
) -> dict[str, tuple[float, float, float, float, float]]:
    output: dict[str, tuple[float, float, float, float, float]] = {}
    for setup in skeleton.bones:
        timelines = animation.bones.get(setup.name, BoneAnimation())
        rotation = setup.rotation + _sample_scalar(timelines.rotate, frame, 0.0)
        translate = _sample_vector(timelines.translate, frame, 0.0)
        scale = _sample_vector(timelines.scale, frame, 1.0)
        local_x = setup.x + translate[0]
        local_y = setup.y + translate[1]
        local_scale_x = setup.scale_x * scale[0]
        local_scale_y = setup.scale_y * scale[1]
        if setup.parent is None:
            output[setup.name] = (local_x, local_y, rotation, local_scale_x, local_scale_y)
            continue
        parent = output[setup.parent]
        radians = math.radians(parent[2])
        scaled_x = local_x * parent[3]
        scaled_y = local_y * parent[4]
        world_x = parent[0] + math.cos(radians) * scaled_x - math.sin(radians) * scaled_y
        world_y = parent[1] + math.sin(radians) * scaled_x + math.cos(radians) * scaled_y
        output[setup.name] = (
            world_x,
            world_y,
            parent[2] + rotation,
            parent[3] * local_scale_x,
            parent[4] * local_scale_y,
        )
    return output


def _projected_groups(
    sample: PoseSample,
    groups: dict[str, ResolvedSlotGroup],
    target_segments: dict[str, Segment2D],
    projector: Projector,
    warnings: list[str],
) -> dict[str, ProjectedGroup]:
    result: dict[str, ProjectedGroup] = {}
    for name, group in groups.items():
        source = sample.segments.get(group.segment)
        if source is None or name not in target_segments:
            warning = f"missing_depth:{name}:frame={sample.output_frame}"
            if warning not in warnings:
                warnings.append(warning)
            continue
        first = projector.project_point(source.start)
        second = projector.project_point(source.end)
        result[name] = ProjectedGroup(
            name,
            target_segments[name],
            first.camera_depth,
            second.camera_depth,
            source.confidence,
        )
    return result


def _setup_items(setup_slots: list[str], slot_to_group: dict[str, str]) -> list[str]:
    items: list[str] = []
    emitted: set[str] = set()
    for slot in setup_slots:
        group = slot_to_group.get(slot)
        if group is None:
            items.append(f"slot:{slot}")
        elif group not in emitted:
            items.append(f"group:{group}")
            emitted.add(group)
    return items


def _flatten_items(
    items: list[str], groups: dict[str, ResolvedSlotGroup]
) -> list[str]:
    result: list[str] = []
    for item in items:
        kind, name = item.split(":", 1)
        result.extend(groups[name].slots if kind == "group" else [name])
    return result


def _desired_before(delta: float, epsilon: float) -> bool | None:
    if delta > epsilon:
        return True
    if delta < -epsilon:
        return False
    return None


def _segment_box_overlap(
    segment: Segment2D, box: tuple[float, float, float, float], margin: float
) -> bool:
    min_x = min(segment.start[0], segment.end[0]) - margin
    max_x = max(segment.start[0], segment.end[0]) + margin
    min_y = min(segment.start[1], segment.end[1]) - margin
    max_y = max(segment.start[1], segment.end[1]) + margin
    return not (max_x < box[0] or min_x > box[2] or max_y < box[1] or min_y > box[3])


def _segment_bbox_overlap(first: Segment2D, second: Segment2D, margin: float) -> bool:
    first_box = (
        min(first.start[0], first.end[0]) - margin,
        min(first.start[1], first.end[1]) - margin,
        max(first.start[0], first.end[0]) + margin,
        max(first.start[1], first.end[1]) + margin,
    )
    return _segment_box_overlap(second, first_box, 0.0)


def _active_manual_edges(
    overrides: dict[str, Any],
    sample: PoseSample,
    groups: dict[str, ResolvedSlotGroup],
    body_slot: str,
    source_start_frame: int,
    fps: float,
) -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    default_space = overrides.get("time_space", "output")
    for value in overrides.get("overrides", []):
        space = value.get("time_space", default_space)
        time = sample.time_seconds if space == "output" else sample.source_frame / fps
        if not (float(value["start"]) <= time < float(value["end"])):
            continue
        front = str(value["front"])
        behind = str(value["behind"])
        front_item = f"slot:{body_slot}" if front == "body" else f"group:{front}"
        behind_item = f"slot:{body_slot}" if behind == "body" else f"group:{behind}"
        if front != "body" and front not in groups:
            raise ValueError(f"Manual override references unknown front group {front!r}")
        if behind != "body" and behind not in groups:
            raise ValueError(f"Manual override references unknown behind group {behind!r}")
        result.append((behind_item, front_item))
    return result


def _add_hard_edge(nodes: list[str], edges: set[tuple[str, str]], edge: tuple[str, str]) -> None:
    if edge[0] not in nodes or edge[1] not in nodes:
        raise ValueError(f"Hard draw-order relation references missing item: {edge}")
    if _would_cycle(nodes, edges, edge[0], edge[1]):
        raise ValueError(f"Hard draw-order constraints contain a cycle at {edge}")
    edges.add(edge)


def _would_cycle(
    nodes: list[str], edges: set[tuple[str, str]], before: str, after: str
) -> bool:
    if before == after:
        return True
    outgoing = {node: [] for node in nodes}
    for first, second in edges:
        if first in outgoing:
            outgoing[first].append(second)
    stack = [after]
    visited: set[str] = set()
    while stack:
        node = stack.pop()
        if node == before:
            return True
        if node in visited:
            continue
        visited.add(node)
        stack.extend(outgoing.get(node, []))
    return False


def _sample_scalar(keys: list[ScalarKey], frame: int, default: float) -> float:
    if not keys:
        return default
    if frame <= keys[0].frame:
        return keys[0].value if frame == keys[0].frame else default
    for first, second in zip(keys, keys[1:]):
        if frame <= second.frame:
            amount = (frame - first.frame) / max(1, second.frame - first.frame)
            return _lerp(first.value, second.value, amount)
    return keys[-1].value


def _sample_vector(
    keys: list[VectorKey], frame: int, default: float
) -> tuple[float, float]:
    if not keys:
        return default, default
    if frame <= keys[0].frame:
        return (keys[0].x, keys[0].y) if frame == keys[0].frame else (default, default)
    for first, second in zip(keys, keys[1:]):
        if frame <= second.frame:
            if frame < second.frame and first.curve == "stepped":
                return first.x, first.y
            amount = (frame - first.frame) / max(1, second.frame - first.frame)
            return _lerp(first.x, second.x, amount), _lerp(first.y, second.y, amount)
    return keys[-1].x, keys[-1].y


def _lerp(first: float, second: float, amount: float) -> float:
    return first + (second - first) * amount
