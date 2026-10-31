"""Independent simulation truth scoring; never provides motion or inspection inputs."""
from collections import Counter
import numpy as np
from yolo_conveyor.orientation.evaluation import rate,distribution
from .geometry import observable_measures
from .gate import POLICY
from .scene import GUIDE_TOP,GUIDE_XY,INSERT_DEPTH


def truth(scene):
    head,tip=scene.object_anchors();axis=head-tip;length=float(np.linalg.norm(axis))
    estimate=dict(head_world_m=head.tolist(),tip_world_m=tip.tolist(),axis_world=(axis/length).tolist(),length_m=length)
    measures=observable_measures(estimate,scene.tool_transform())
    contacts=set()
    for contact in scene.data.contact:
        a,b=(int(scene.model.geom_bodyid[i]) for i in (contact.geom1,contact.geom2))
        if a==scene.body_id and b in scene.finger_bodies:contacts.add(b)
        if b==scene.body_id and a in scene.finger_bodies:contacts.add(a)
    held=len(contacts)==2 and np.linalg.norm((head+tip)/2-scene.tool_transform()[:3,3])<.04
    conforming=(held and measures['tilt_deg']<=POLICY['gt_max_tilt_deg'] and
                POLICY['gt_protrusion_min_m']<=measures['tip_protrusion_m']<=POLICY['gt_protrusion_max_m'] and
                measures['tip_lateral_offset_m']<=POLICY['max_lateral_m'])
    return dict(gt='CONFORMING' if conforming else 'NONCONFORMING',held=bool(held),estimate=estimate,measures=measures)


class PhysicalScore:
    def __init__(self,scene):
        self.scene=scene;self.illegal=set();self.max_penetration=0.

    def observe(self):
        s=self.scene;self.illegal.update(s.illegal_contacts())
        for contact in s.data.contact:
            if s.body_id in (int(s.model.geom_bodyid[contact.geom1]),int(s.model.geom_bodyid[contact.geom2])):
                self.max_penetration=max(self.max_penetration,float(-contact.dist))

    def result(self,attempted,abort):
        final=truth(self.scene);tip=np.array(final['estimate']['tip_world_m'])
        depth=float(GUIDE_TOP-tip[2]);radial=float(np.linalg.norm(tip[:2]-GUIDE_XY));tilt=final['measures']['tilt_deg']
        successful=(attempted and abort is None and final['held'] and abs(depth-INSERT_DEPTH)<=POLICY['final_depth_tolerance_m']
                    and radial<=POLICY['final_radial_m'] and tilt<=POLICY['final_max_tilt_deg'] and not self.illegal
                    and self.max_penetration<=POLICY['max_penetration_m'])
        return dict(preinsert_success=bool(successful),attempted=attempted,abort=abort,held=final['held'],depth_m=depth,
                    radial_m=radial,tilt_deg=tilt,max_penetration_m=self.max_penetration,illegal_contacts=sorted(self.illegal),
                    final_world_points_m=[final['estimate']['head_world_m'],final['estimate']['tip_world_m']])


def summarize(rows):
    bad=[r for r in rows if r['inspection_gt']['gt']=='NONCONFORMING']
    good=[r for r in rows if r['inspection_gt']['gt']=='CONFORMING']
    passed=[r for r in rows if r['decision']['decision']=='PASS']
    attempted=[r for r in rows if r['result']['attempted']]
    return dict(episodes=len(rows),decisions=dict(Counter(r['decision']['decision'] for r in rows)),
                wrong_accept=rate(sum(r['decision']['decision']=='PASS' for r in bad),len(bad)),
                conforming_not_passed=rate(sum(r['decision']['decision']!='PASS' for r in good),len(good)),
                preinsert_success_given_attempt=rate(sum(r['result']['preinsert_success'] for r in attempted),len(attempted)),
                preinsert_success_given_inspection_pass=rate(sum(r['result']['preinsert_success'] for r in passed),len(passed)),
                preinsert_success_all_episodes=rate(sum(r['result']['preinsert_success'] for r in rows),len(rows)),
                permit_denials=dict(Counter(r['permit_denied'] for r in rows if r['permit_denied'])),
                tip_error_mm=distribution([r['prediction_error']['tip_mm'] for r in rows if r['prediction_error']]),
                axis_error_deg=distribution([r['prediction_error']['axis_deg'] for r in rows if r['prediction_error']]),
                inference_ms_two_views=distribution([r['inference_ms'] for r in rows]),
                end_to_end_wall_s=distribution([r['wall_s'] for r in rows]),
                simulated_cycle_s=distribution([r['simulated_s'] for r in rows]),
                continuous_throughput='not measured: independent episodes without return/refeed',
                illegal_contact_episodes=sum(bool(r['result']['illegal_contacts']) for r in rows))
