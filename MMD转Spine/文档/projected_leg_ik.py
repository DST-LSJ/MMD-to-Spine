"""Position-based, fixed-length two-link IK after 3D constraint evaluation."""
from __future__ import annotations

import math


def cross(a, b):
    return a[0]*b[1]-a[1]*b[0]


def rotate(v, angle):
    c, s = math.cos(angle), math.sin(angle)
    return c*v[0]-s*v[1], s*v[0]+c*v[1]


def solve_two_link(target, pole, first, second, previous_side, *, minimum_bend=0.0,
                   maximum_bend=179.9, pole_deadband=0.5, fallback_direction=(0.0, -1.0),
                   fixed_side=None):
    """Solve relative knee/ankle positions, clamping reach without scaling.

    side is the sign of cross(hip->ankle, hip->knee), not an Euler angle.
    A nearly collinear pole retains the previous branch.
    """
    if first <= 0 or second <= 0 or not all(math.isfinite(x) for x in (*target, *pole, first, second)):
        raise ValueError("Invalid two-link IK input")
    distance = math.hypot(*target)
    direction = (target[0]/distance, target[1]/distance) if distance > 1e-9 else fallback_direction
    norm = math.hypot(*direction)
    direction = (direction[0]/norm, direction[1]/norm) if norm > 1e-9 else (0., -1.)
    minimum_bend = max(0., min(179.9, minimum_bend))
    maximum_bend = max(minimum_bend, min(179.9, maximum_bend))
    def reach(bend):
        return math.sqrt(max(0., first*first+second*second+2*first*second*math.cos(math.radians(bend))))
    radius = min(reach(minimum_bend), max(max(1e-8, reach(maximum_bend)), distance))
    pole_height = cross(direction, pole)
    side = previous_side if abs(pole_height) <= pole_deadband else (1 if pole_height > 0 else -1)
    if fixed_side is not None:
        side = 1 if fixed_side > 0 else -1
    along = (first*first-second*second+radius*radius)/(2*radius)
    height = math.sqrt(max(0., first*first-along*along))
    knee = (direction[0]*along-direction[1]*side*height,
            direction[1]*along+direction[0]*side*height)
    ankle = (direction[0]*radius, direction[1]*radius)
    return knee, ankle, side, abs(radius-distance) > 1e-6


class ProjectedLegIK:
    """Calibrate source rest positions once, then solve each projected frame."""
    def __init__(self, skeleton, solver, projector, config):
        from .semantic_retarget import _setup_world_matrices
        self.bones = skeleton.bone_by_name
        self.projector = projector
        self.config = config
        self.solver = solver
        self.use_leg_plane = getattr(solver, "profile", {}).get("leg_target_method") == "virtual_3d_leg_plane_v1"
        world = _setup_world_matrices(skeleton)
        self.references = {}
        self.rows = []
        self.previous_mirror = 1
        for suffix in ("F", "B"):
            thigh, leg = world[f"thigh{suffix}"], world[f"leg{suffix}"]
            hip, knee = thigh[4:], leg[4:]
            length = self.bones[f"leg{suffix}"].length
            target_k = (knee[0]-hip[0], knee[1]-hip[1])
            target_a = (target_k[0]+leg[0]*length, target_k[1]+leg[2]*length)
            first, second = math.hypot(*target_k), math.hypot(leg[0]*length, leg[2]*length)
            rest_upper = solver.geometry[f"thigh_{suffix}"]
            rest_lower = solver.geometry[f"shin_{suffix}"]
            source_k = projector.direction(rest_upper)
            source_a = projector.direction(tuple(a+b for a, b in zip(rest_upper, rest_lower)))
            if math.hypot(*source_a) < 1e-6:
                raise ValueError("Source rest leg projection is too short to calibrate")
            scale = math.hypot(*target_a)/math.hypot(*source_a)
            angle = math.atan2(target_a[1], target_a[0])-math.atan2(source_a[1], source_a[0])
            mapped_k = rotate((source_k[0]*scale, source_k[1]*scale), angle)
            correction = (target_k[0]-mapped_k[0], target_k[1]-mapped_k[1])
            bend = math.degrees(math.acos(max(-1., min(1.,
                (target_k[0]*(target_a[0]-target_k[0])+target_k[1]*(target_a[1]-target_k[1]))/(first*second)))))
            side = 1 if cross(target_a, target_k) >= 0 else -1
            bend_overrides = getattr(solver, "profile", {}).get("spine_leg_bend_positive", {})
            if suffix in bend_overrides:
                # Pole-side sign is opposite to Spine's signed bend direction.
                side = -1 if bend_overrides[suffix] else 1
            self.references[suffix] = dict(first=first, second=second, scale=scale, angle=angle,
                correction=correction, setup_angle=math.atan2(target_a[1], target_a[0]),
                target_a=target_a, minimum_bend=bend, side=side, last_side=side,
                rest_upper=rest_upper,
                thigh_setup_offset=math.atan2(target_k[1],target_k[0])-math.atan2(source_k[1],source_k[0]))

        self.plane_adapter = None
        if self.use_leg_plane:
            from .leg_plane_target import LegPlaneTarget
            self.plane_adapter = LegPlaneTarget(self)

    def solve(self, source, suffix, body_world, mirror):
        from .semantic_retarget import (_solve_local_rotation, _multiply_matrices,
            _animated_local_matrix)
        ref = self.references[suffix]
        points = source.pose.joints
        hip3, knee3, ankle3 = (points[f"{n}_{suffix}"] for n in ("hip", "knee", "ankle"))
        # Project all three constrained positions, then work relative to hip.
        hip2, knee2, ankle2 = (self.projector.direction(p) for p in (hip3, knee3, ankle3))
        ankle = rotate(((ankle2[0]-hip2[0])*ref["scale"], (ankle2[1]-hip2[1])*ref["scale"]), mirror*ref["angle"])
        pole = rotate(((knee2[0]-hip2[0])*ref["scale"], (knee2[1]-hip2[1])*ref["scale"]), mirror*ref["angle"])
        # Preserve target setup bend while using the projected knee as pole.
        # Rotate this rest correction along with the projected hip-ankle ray.
        setup_a = (mirror*ref["target_a"][0], ref["target_a"][1])
        ray = math.atan2(ankle[1], ankle[0]) if math.hypot(*ankle) > 1e-8 else math.atan2(setup_a[1], setup_a[0])
        correction = rotate((mirror*ref["correction"][0], ref["correction"][1]),
                            ray-math.atan2(setup_a[1], setup_a[0]))
        pole = (pole[0]+correction[0], pole[1]+correction[1])
        plane_diagnostic = None
        if self.plane_adapter is not None:
            ankle, pole, plane_diagnostic = self.plane_adapter.target(source, suffix, body_world, mirror)
        previous = ref["last_side"]
        if self.previous_mirror != mirror:
            previous *= -1
        minimum_bend = max(2.0, ref["minimum_bend"])
        knee, reached, side, clamped = solve_two_link(ankle, pole, ref["first"], ref["second"], previous,
            minimum_bend=minimum_bend,
            maximum_bend=min(179.9, ref["minimum_bend"]+self.config.maximum_knee_flex_deg),
            fallback_direction=setup_a, fixed_side=ref["side"]*mirror)
        ref["last_side"] = side
        thigh_name, leg_name = f"thigh{suffix}", f"leg{suffix}"
        thigh_value = _solve_local_rotation(body_world, self.bones[thigh_name],
            (self.bones[leg_name].x, self.bones[leg_name].y), math.degrees(math.atan2(knee[1], knee[0])))
        # The endpoint IK, not an independent thigh-angle clamp, owns the pose.
        thigh_world = _multiply_matrices(body_world, _animated_local_matrix(self.bones[thigh_name], thigh_value))
        shin = (reached[0]-knee[0], reached[1]-knee[1])
        leg_value = _solve_local_rotation(thigh_world, self.bones[leg_name], (1., 0.),
                                         math.degrees(math.atan2(shin[1], shin[0])))
        leg_world = _multiply_matrices(thigh_world, _animated_local_matrix(self.bones[leg_name], leg_value))
        actual_hip = thigh_world[4:]
        actual_knee = (leg_world[4]-actual_hip[0], leg_world[5]-actual_hip[1])
        actual_ankle = (actual_knee[0]+leg_world[0]*self.bones[leg_name].length,
                        actual_knee[1]+leg_world[2]*self.bones[leg_name].length)
        error = math.dist(actual_ankle, reached)
        if error > 1e-5:
            raise ValueError("Target parent scale is incompatible with fixed world-length IK")
        from .semantic_retarget import _inverse_direction
        target_body = _inverse_direction(body_world,
            (actual_hip[0]+reached[0]-body_world[4], actual_hip[1]+reached[1]-body_world[5]))
        self.rows.append(dict(frame=source.output_frame, side=suffix, mirror=mirror,
            target_body=target_body, fixed_local_branch=ref["side"],
            minimum_bend=minimum_bend,
            first_length=ref["first"], second_length=ref["second"], projected_hip=hip2,
            projected_knee=knee2, projected_ankle=ankle2, target_ankle=ankle, reached_ankle=reached,
            knee=knee, pole=pole, branch=side, clamped=clamped, endpoint_error=error))
        if plane_diagnostic is not None:
            self.rows[-1].update(plane_diagnostic)
        return thigh_value, leg_value

    def diagnostics(self):
        return {"method": ("virtual_3d_leg_plane_fixed_length_two_link_ik" if self.use_leg_plane
                           else "projected_positions_fixed_length_two_link_ik"),
                "pole": "projected_knee_diagnostic_only_fixed_setup_branch",
                "bend_direction": "constant_per_leg_in_parent_space_mirror_inherited",
                "foot_orientation": "no_independent_target_foot_bone",
                "clamped_frames": {s: sum(r["clamped"] for r in self.rows if r["side"] == s) for s in ("F", "B")},
                "max_endpoint_error": max((r["endpoint_error"] for r in self.rows), default=0.),
                "frames": self.rows}
