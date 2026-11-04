"""Independent physical truth and metrics for annular part pre-positioning."""
from collections import Counter
import numpy as np
from yolo_conveyor.orientation.evaluation import rate,distribution
from .data import physical_label
from .scene import SHAFT_XY,SHAFT_TOP,ENGAGEMENT
from .perception import POLICY


class PhysicalScore:
    def __init__(self,scene):
        self.scene=scene;self.illegal=set();self.penetration=0.

    def observe(self):
        s=self.scene;self.illegal.update(s.illegal_contacts())
        for c in s.data.contact:
            if s.body_id in (int(s.model.geom_bodyid[c.geom1]),int(s.model.geom_bodyid[c.geom2])):
                self.penetration=max(self.penetration,float(-c.dist))

    def result(self,attempted,abort):
        s=self.scene;gt=physical_label(s);rotation=s.data.xmat[s.body_id].reshape(3,3)
        local=rotation.T@(np.array([*SHAFT_XY,SHAFT_TOP])-np.array(gt['center_world_m']))
        radial=float(np.linalg.norm(local[:2]));engagement=float(local[2]+s.spec['height']/2)
        success=(attempted and abort is None and gt['held'] and gt['face']=='front' and gt['tilt_deg']<=POLICY['final_tilt_deg']
                 and abs(engagement-ENGAGEMENT)<=POLICY['engagement_tolerance_m'] and radial<=.001-POLICY['radial_margin_m']
                 and not self.illegal and self.penetration<=POLICY['max_penetration_m'])
        return dict(success=bool(success),attempted=attempted,abort=abort,held=gt['held'],engagement_m=engagement,
                    radial_m=radial,tilt_deg=gt['tilt_deg'],illegal_contacts=sorted(self.illegal),max_penetration_m=self.penetration)


def summarize(rows):
    bad=[r for r in rows if r['initial_gt']['gt']=='NONCONFORMING'];good=[r for r in rows if r['initial_gt']['gt']=='CONFORMING']
    tried=[r for r in rows if r['result']['attempted']]
    return dict(episodes=len(rows),decisions=dict(Counter(r['decision']['decision'] for r in rows)),
                wrong_accept=rate(sum(r['decision']['decision']=='PASS' for r in bad),len(bad)),
                conforming_not_passed=rate(sum(r['decision']['decision']!='PASS' for r in good),len(good)),
                success_given_attempt=rate(sum(r['result']['success'] for r in tried),len(tried)),
                success_all=rate(sum(r['result']['success'] for r in rows),len(rows)),
                inference_ms=distribution([r['inference_ms'] for r in rows]),wall_s=distribution([r['wall_s'] for r in rows]),
                simulated_s=distribution([r['simulated_s'] for r in rows]),
                robot_collision_episodes=sum(bool(r['result']['illegal_contacts']) for r in rows),
                continuous_throughput='not measured; independent job episodes')
