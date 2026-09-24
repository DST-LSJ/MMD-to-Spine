from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import copy

from .conversion_settings import CHARACTER_VIEW_OFFSET_DEGREES, SEMANTIC_RETARGET_SETTINGS
from .runtime_assets import ROOT, DOC, LIMBS, RIG_PATH, load_profile
from .generic_dance_solver import GenericDanceSolver
from .projection import ProjectionConfig, Projector
from .semantic_retarget import SemanticRetargetConfig, retarget_semantic_motion, validate_semantic_animation
from .spine_reader import read_spine_json
from .spine_writer import build_spine_json, validate_export, write_spine_json
from .vmd_reader import read_vmd
from .spine_leg_constraints import add_leg_constraints
from .conversion_settings import SECONDARY_MOTION_CONFIG, BLINK_CONFIG
from .secondary_motion import apply_secondary_motion
from .blink import add_blink_timeline


def convert(source, output_dir=ROOT, *, start_seconds=0.0, duration_seconds=30.0,
            secondary=True, blink=True, output_path=None, animation_name=None,
            write_report=True, draw_order=False, progress=None):
    def notify(value,stage):
        if progress is not None:
            progress(value,stage)
    notify(0.,'读取动作')
    source = Path(source)
    output_dir = Path(output_dir)
    motion = read_vmd(source)
    profile = load_profile()
    template = read_spine_json(DOC / "MMD.4.3.json")
    projector = Projector(ProjectionConfig(camera_yaw_deg=CHARACTER_VIEW_OFFSET_DEGREES,
                                          camera_pitch_deg=0, scale=20, depth_factor=1))
    solver = GenericDanceSolver(motion, profile, projector)
    if not math.isfinite(start_seconds) or start_seconds < 0:
        raise ValueError("起始秒数必须是有限的非负数")
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("持续秒数必须是有限的正数")
    start = round(start_seconds*30)
    if start >= motion.source_max_frame:
        raise ValueError("起始时间已到达或超过VMD末尾")
    end = min(start+max(1, round(duration_seconds*30)), motion.source_max_frame)
    frames = end-start
    result = retarget_semantic_motion(motion, template, solver, projector, fps=30,
        start_frame=start, end_frame=end, config=SemanticRetargetConfig.from_mapping(SEMANTIC_RETARGET_SETTINGS),
        progress=lambda value,stage:notify(.05+.75*value,stage))
    notify(.81,'校验主动作')
    invariants = validate_semantic_animation(result.animation, expected_max_frame=frames)
    if draw_order:
        from .draw_order import generate_draw_order, load_draw_order_config, load_manual_overrides
        config = copy.deepcopy(load_draw_order_config(DOC / "draw_order_config.json"))
        order = generate_draw_order(template, result.animation, result.pose_samples, projector, config,
                                    load_manual_overrides(DOC / "draw_order_overrides.json"))
        result.animation.draw_order_events = order.events
    if secondary:
        notify(.84,'生成头发和裙摆')
        apply_secondary_motion(result.animation, template, copy.deepcopy(SECONDARY_MOTION_CONFIG))
    clip_label = f"通用骨架_{start/30:g}秒起_{frames/30:g}秒"
    name = animation_name or source.stem+"_"+clip_label
    data = build_spine_json(template, result.animation, name, preserve_animations=False)
    data.setdefault("skeleton", {})["images"] = 'Texture2D/'
    notify(.88,'构建骨架和约束')
    validate_export(data, name)
    # Verify actual evaluated limb lengths, all samples and target setup data.
    max_length_error = 0.0
    for pose in result.pose_samples:
        for suffix in ("F", "B"):
            for a, b, geometry in (("shoulder", "elbow", "upper_arm"), ("elbow", "wrist", "forearm"),
                                   ("hip", "knee", "thigh"), ("knee", "ankle", "shin")):
                length = math.dist(pose.joints[f"{a}_{suffix}"], pose.joints[f"{b}_{suffix}"])
                expected = math.sqrt(sum(x*x for x in solver.geometry[f"{geometry}_{suffix}"]))
                max_length_error = max(max_length_error, abs(length-expected))
                if abs(length-expected) > 1e-5:
                    raise AssertionError(f"Unexpected source limb length at {pose.source_frame}: {geometry}")
    assert data["bones"] == template.raw["bones"]
    native_ik = add_leg_constraints(data, name, template, result.diagnostics["leg_ik_2d"])
    blink_count = 0
    if blink:
        blink_count = add_blink_timeline(data, name, template, ROOT / BLINK_CONFIG["texture_path"],
                                         30, frames, copy.deepcopy(BLINK_CONFIG))
    destination = Path(output_path) if output_path else output_dir / (source.stem+"_"+clip_label+"_spine.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    notify(.94,'写入并校验JSON')
    with tempfile.TemporaryDirectory(prefix="pmx_dance_export_") as directory:
        temp = Path(directory)/"output.json"
        write_spine_json(temp, data)
        verified = json.loads(temp.read_text(encoding="utf-8"))
        validate_export(verified, name)
        assert len(verified["animations"]) == 1
        for bone in LIMBS:
            timeline = verified["animations"][name]["bones"][bone]
            assert "translate" not in timeline and "scale" not in timeline
            assert len(timeline["rotate"]) == frames+1
        destination.write_bytes(temp.read_bytes())
    report = {"input": source.name, "output": destination.name, "animation": name,
              "source_frames": [start, end], "source_seconds": [start/30, end/30],
              "duration_seconds": frames/30, "samples": frames+1,
              "requested_duration_seconds": duration_seconds,
              "leg_target_method": "virtual_3d_leg_plane_v1",
              "secondary_motion": result.animation.secondary_motion_diagnostics,
              "blink": {"requested": blink, "events": blink_count},
              "draw_order_enabled": draw_order,
              "source_rig": profile['name'], "source_rig_file": "资源/通用骨架.json",
              "source_rig_basis": "generic_configurable_not_model_measured",
              "solver": solver.depth_source, "ik": solver.ik_statistics, "spine_ik": native_ik,
              "max_source_length_error": max_length_error, "invariants": invariants,
              "diagnostics": result.diagnostics,
              "limitations": ["Foot IK uses an analytic two-bone solver, not MMD CCD and its per-link limits",
                              "Virtual 3D leg-plane target before projection; setup knee boundary preserved; foot projection is intentionally adapted",
                              "Target template has no separate foot bones: independent foot heading is not preserved",
                              "Generic source proportions; no model-specific append/twist constraints, toe IK, physics, skin deformation, or facial/finger retargeting",
                              "Dynamic draw order disabled by default; no key reduction"],
              "visual_acceptance": "Leg-plane method retained; generic source rig conversion awaits visual review"}
    if write_report:
        notify(.98,'写入转换报告')
        diagnostic = DOC / "诊断_v2" / (source.stem+"_"+clip_label)
        diagnostic.mkdir(parents=True, exist_ok=True)
        (diagnostic / "转换报告.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    notify(1.,'转换完成')
    return destination, report

