"""Virtual 3D leg in the display plane, before projection.

Source points come from the configured source solver, not a native MMD bake.
No support/lift phase classification and no legacy extension correction.
"""
import math

from .projection import rotate_vector


def dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def unit(v):
    length = math.sqrt(dot(v, v))
    if length < 1e-10:
        raise ValueError("Degenerate leg plane vector")
    return tuple(x/length for x in v)


def perpendicular(v, normal):
    return tuple(x-dot(v, normal)*n for x, n in zip(v, normal))


def plane_coordinates(upper, lower, guide, previous_normal=None, previous_down=None):
    """Unfold around world vertical; a transported hinge resolves straight legs.

    The normal's orientation comes from the rest hinge, never from a screen
    left/right decision. At a straight pose use that hinge projected normal to
    the upper segment. Both bone lengths and their included angle are measured
    before unfolding.
    """
    u, v = unit(upper), unit(lower)
    raw = cross(u, v)
    strength = math.sqrt(dot(raw, raw))
    if strength > 1e-6:
        normal = unit(raw)
        alignment = dot(normal, guide)
        if abs(alignment) < 1e-6 and previous_normal is not None:
            alignment = dot(normal, previous_normal)
        if alignment < 0:
            normal = tuple(-x for x in normal)
        fallback = False
    else:
        candidate = perpendicular(guide, u)
        if dot(candidate, candidate) < 1e-12 and previous_normal is not None:
            candidate = perpendicular(previous_normal, u)
        if dot(candidate, candidate) < 1e-12:
            candidate = perpendicular(min(((1.,0.,0.),(0.,1.,0.),(0.,0.,1.)), key=lambda a: abs(dot(a,u))), u)
        normal = unit(candidate)
        fallback = True
    down = perpendicular((0., -1., 0.), normal)
    if dot(down, down) < 1e-10:
        down = perpendicular(previous_down or (0., 0., -1.), normal)
    down = unit(down)
    right = unit(cross(normal, down))
    angle = math.atan2(-dot(u, down), dot(u, right))
    bend = math.degrees(math.acos(max(-1., min(1., dot(u, v)))))
    return angle, bend, normal, down, fallback


class LegPlaneTarget:
    def __init__(self, owner):
        self.owner = owner
        self.states = {}
        for suffix, ref in owner.references.items():
            upper = ref['rest_upper']
            lower = owner.solver.geometry[f'shin_{suffix}']
            # PMX knee bends toward negative local X in this measured model.
            raw = cross(upper, lower)
            guide = unit(tuple(-x for x in raw)) if dot(raw,raw) > 1e-12 else (1.,0.,0.)
            angle, _, _, _, _ = plane_coordinates(upper, lower, guide)
            self.states[suffix] = dict(guide=guide, rest_angle=angle)

    def target(self, source, suffix, body_world, mirror):
        from .semantic_retarget import _solve_local_rotation
        from .projected_leg_ik import rotate
        owner, state = self.owner, self.states[suffix]
        ref = owner.references[suffix]
        p = source.pose.joints
        h,k,a = (p[f'{name}_{suffix}'] for name in ('hip','knee','ankle'))
        upper, lower = tuple(y-x for x,y in zip(h,k)), tuple(y-x for x,y in zip(k,a))
        q = owner.solver.pelvis_rotation(source.pose.source_frame)
        guide = rotate_vector(q, state['guide'])
        angle, bend, normal, down, fallback = plane_coordinates(
            upper, lower, guide, state.get('normal'), state.get('down'))
        state.update(normal=normal, down=down)
        delta = (angle-state['rest_angle']+math.pi)%(2*math.pi)-math.pi
        leg, thigh = owner.bones[f'leg{suffix}'], owner.bones[f'thigh{suffix}']
        setup_angle = math.atan2(leg.y,leg.x)+math.radians(thigh.rotation)
        # Rest world body has no rotation in this template. Calibrate using the
        # measured target rest direction rather than a fitted motion offset.
        rest_world_angle = ref['thigh_setup_offset'] + math.atan2(*reversed(owner.projector.direction(ref['rest_upper'])))
        world_angle = rest_world_angle+delta
        direction = (mirror*math.cos(world_angle), math.sin(world_angle))
        local_angle = _solve_local_rotation(body_world, thigh, (leg.x,leg.y),
                                            math.degrees(math.atan2(direction[1],direction[0])))
        local_angle = (local_angle+180)%360-180
        previous = state.get('angle',local_angle)
        local_angle = previous+(local_angle-previous+180)%360-180
        limit, step = owner.config.thigh_rotation_limit_deg, owner.config.source_direction_max_step_deg
        requested_angle = local_angle
        local_angle = max(-limit,min(limit,local_angle))
        local_angle = max(previous-step,min(previous+step,local_angle))
        state['angle'] = local_angle
        local = rotate((math.cos(setup_angle),math.sin(setup_angle)), math.radians(local_angle))
        direction = (body_world[0]*local[0]+body_world[1]*local[1],
                     body_world[2]*local[0]+body_world[3]*local[1])
        length = math.hypot(*direction)
        direction = tuple(x/length for x in direction)
        # Preserve the accepted template straight boundary.
        minimum = max(2.,ref['minimum_bend'])
        safe_bend = max(minimum,min(bend,min(179.9,ref['minimum_bend']+owner.config.maximum_knee_flex_deg)))
        lower2 = rotate(direction,-ref['side']*mirror*math.radians(safe_bend))
        # Inverse camera basis: construct actual 3D virtual joint positions,
        # adapting the two lengths independently to the target proportions.
        yaw, pitch = owner.projector.yaw, owner.projector.pitch
        right = (math.cos(yaw),0.,math.sin(yaw))
        up = (-math.sin(yaw)*math.sin(pitch),math.cos(pitch),math.cos(yaw)*math.sin(pitch))
        lift = lambda v: tuple(v[0]*r+v[1]*u for r,u in zip(right,up))
        knee3 = lift(tuple(x*ref['first'] for x in direction))
        shin3 = lift(tuple(x*ref['second'] for x in lower2))
        ankle3 = tuple(x+y for x,y in zip(knee3,shin3))
        target = owner.projector.direction(ankle3)
        pole = owner.projector.direction(knee3)
        diagnostic = dict(plane_method='virtual_3d_leg_plane_v1', source_bend=bend,
            plane_normal=normal, plane_down=down, straight_plane_fallback=fallback,
            virtual_hip=(0.,0.,0.),virtual_knee=knee3,virtual_ankle=ankle3,
            desired_bend=safe_bend, source_plane_angle_deg=math.degrees(angle),
            requested_thigh_angle=requested_angle, applied_thigh_angle=local_angle)
        return target, pole, diagnostic
