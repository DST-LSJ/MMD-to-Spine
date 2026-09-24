"""Configurable canonical humanoid FK and analytic foot IK; no model files."""
import copy
from .semantic_retarget import SemanticPoseSolver, _qmul, _length3
from .projection import rotate_vector
from .source_pose import solve_two_bone_3d
from .runtime_assets import add, subtract


class GenericDanceSolver(SemanticPoseSolver):
    def __init__(self, motion, profile, projector):
        profile=copy.deepcopy(profile)
        profile['depth_source']='generic_humanoid_fk_analytic_foot_ik'
        profile['leg_retarget_method']='projected_two_link_ik'
        profile['leg_target_method']='virtual_3d_leg_plane_v1'
        profile['spine_leg_bend_positive']={'F':False,'B':False}
        super().__init__(motion,profile,projector)
        self.ik_statistics={s:{'enabled_frames':0,'clamped_frames':0} for s in 'FB'}

    def pelvis_rotation(self, frame):
        return self._rotation_chain(('all_parent','center','groove','waist','lower_body'),frame)

    def _solve_leg(self,suffix,side,pelvis,pelvis_reference,root_position,
                   root_rotation,lower_rotation,frame,joints,warnings):
        hip_offset=self.geometry[f'hip_{suffix}_offset']
        upper=self.geometry[f'thigh_{suffix}'];lower=self.geometry[f'shin_{suffix}']
        hip=add(pelvis,rotate_vector(lower_rotation,hip_offset))
        semantic=f'{side}_foot_ik'
        enabled=(self.profile.get('leg_ik',{}).get('enabled',True)
                 and self._has(semantic) and self._ik_enabled(self.actual_names.get(semantic),frame))
        if enabled:
            rest=add(add(add(pelvis_reference,hip_offset),upper),lower)
            target=add(rest,self._position(semantic,frame))
            parent=f'{side}_foot_ik_parent'
            if self._has(parent):
                # The generic controller-parent pivot defaults to the ground
                # below the ankle, and can be configured for another rig.
                pivot=tuple(self.profile.get('leg_ik',{}).get(
                    f'{side}_parent_pivot',[rest[0],0.,rest[2]]))
                target=add(add(pivot,self._position(parent,frame)),
                    rotate_vector(self._rotation(parent,frame),subtract(target,pivot)))
            target=add(root_position,rotate_vector(root_rotation,target))
            hint=tuple(self.profile.get('leg_ik',{}).get(f'{side}_bend_hint',[0.,0.,-1.]))
            knee,ankle,clamped=solve_two_bone_3d(hip,target,_length3(upper),_length3(lower),
                rotate_vector(lower_rotation,hint))
            self.ik_statistics[suffix]['enabled_frames']+=1
            self.ik_statistics[suffix]['clamped_frames']+=int(clamped)
            if clamped:
                warnings.append(f'{side}_generic_ik_target_clamped')
        else:
            q=_qmul(lower_rotation,self._rotation(f'{side}_leg',frame))
            knee=add(hip,rotate_vector(q,upper))
            q=_qmul(q,self._rotation(f'{side}_knee',frame))
            ankle=add(knee,rotate_vector(q,lower))
        for name,point in (('hip',hip),('knee',knee),('ankle',ankle)):
            joints[f'{name}_{suffix}']=point
