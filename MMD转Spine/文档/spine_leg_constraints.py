"""Spine 4.3 native two-bone leg constraints with fixed bend direction.

Schema checked against official 4.3 SkeletonJson.ts and IkConstraint.ts.
Existing uniform-scale template supports nonzero child Y with stretch=false.
"""
from __future__ import annotations

import math

from .semantic_retarget import _setup_world_matrices, _inverse_direction


def minimum_segment_distance(a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    length2 = dx*dx+dy*dy
    t = max(0., min(1., -(a[0]*dx+a[1]*dy)/length2)) if length2 > 1e-12 else 0.
    return math.hypot(a[0]+t*dx, a[1]+t*dy)


def add_leg_constraints(data, animation_name, skeleton, diagnostics, fps=30):
    bones = skeleton.bone_by_name
    world = _setup_world_matrices(skeleton)
    body = world["body"]
    animation = data["animations"][animation_name]
    constraints = data.setdefault("constraints", [])
    existing_names = {b["name"] for b in data["bones"]}
    result = {}
    for suffix in ("F", "B"):
        thigh, leg = f"thigh{suffix}", f"leg{suffix}"
        name, target_name = f"legIK{suffix}", f"ankleIK{suffix}"
        if target_name in existing_names or any(c.get("name") == name for c in constraints):
            raise ValueError("Leg IK target or constraint already exists")
        rows = [r for r in diagnostics["frames"] if r["side"] == suffix]
        local_branch = rows[0]["fixed_local_branch"]
        if any(r["branch"]*r["mirror"] != local_branch for r in rows):
            raise ValueError("Knee branch changes in parent space")
        # Our pole sign is the opposite of Spine's signed elbow/knee angle.
        bend_positive = local_branch < 0
        leg_world = world[leg]
        endpoint = (leg_world[4]+leg_world[0]*bones[leg].length,
                    leg_world[5]+leg_world[2]*bones[leg].length)
        setup = _inverse_direction(body, (endpoint[0]-body[4], endpoint[1]-body[5]))
        data["bones"].append({"name": target_name, "parent": "body", "x": setup[0], "y": setup[1]})
        constraints.append({"name": name, "type": "ik", "bones": [thigh, leg], "target": target_name,
                            "mix": 1.0, "softness": 0.0, "bendPositive": bend_positive,
                            "compress": False, "stretch": False})
        keys = [{"time": round(r["frame"]/fps, 6), "x": r["target_body"][0]-setup[0],
                 "y": r["target_body"][1]-setup[1]} for r in rows]
        hip = (bones[thigh].x, bones[thigh].y)
        l1, l2 = rows[0]["first_length"], rows[0]["second_length"]
        max_reach = math.sqrt(l1*l1+l2*l2+2*l1*l2*math.cos(math.radians(rows[0]["minimum_bend"])))
        # A disk is convex: interpolation between valid targets cannot exceed
        # the extension limit. Near the hip, a chord may cut the inner hole;
        # make only those intervals stepped instead of passing through it.
        min_reach = abs(l1-l2)+1e-4
        stepped = 0
        for i, row in enumerate(rows):
            relative = (row["target_body"][0]-hip[0], row["target_body"][1]-hip[1])
            if math.hypot(*relative) > max_reach+1e-5:
                raise ValueError("Ankle target crosses the safe extension boundary")
            if i+1 < len(rows):
                next_relative = (rows[i+1]["target_body"][0]-hip[0], rows[i+1]["target_body"][1]-hip[1])
                if row["mirror"] != rows[i+1]["mirror"] or minimum_segment_distance(relative, next_relative) < min_reach:
                    keys[i]["curve"] = "stepped"
                    stepped += 1
        animation["bones"][target_name] = {"translate": keys}
        animation.setdefault("ik", {})[name] = [{"time": 0, "mix": 1, "softness": 0,
            "bendPositive": bend_positive, "compress": False, "stretch": False}]
        result[suffix] = {"constraint": name, "target": target_name, "bendPositive": bend_positive,
                         "stretch": False, "compress": False, "safe_max_reach": max_reach,
                         "straight_reach": l1+l2, "stepped_target_intervals": stepped}
    return result
