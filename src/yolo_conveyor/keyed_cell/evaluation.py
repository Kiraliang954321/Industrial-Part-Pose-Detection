"""Independent keyed-geometry/contact scoring, never used to permit motion."""
import numpy as np
from collections import Counter
from yolo_conveyor.orientation.evaluation import rate,distribution
from yolo_conveyor.annular.evaluation import PhysicalScore as PreviousScore
from yolo_conveyor.pose_cell.evaluation import truth
from yolo_conveyor.pose_cell.geometry import rz,error
from yolo_conveyor.pose_cell.scene import SHAFT_XY,SHAFT_TOP,ENGAGEMENT
from yolo_conveyor.datum_pose.assets import DATUMS
from .scene import FIXTURES

LIMITS=dict(min_index_overlap_m=.0018,index_fit_tolerance_m=.00005,max_index_penetration_m=.0002,retracted_clearance_m=.002)


def index_geometry(s):
    r=s.data.xmat[s.body_id].reshape(3,3);t=s.data.xpos[s.body_id];spec=FIXTURES[s.kind]
    if s.kind=='gear':
        xy=np.array([[x,y] for x in (spec['root_radius_m'],spec['tip_radius_m']) for y in (-spec['width_m']/2,spec['width_m']/2)])
    else:
        a=np.linspace(0,2*np.pi,32,endpoint=False)
        xy=np.column_stack([spec['center_radius_m']+spec['radius_m']*np.cos(a),spec['radius_m']*np.sin(a)])
    world=np.column_stack([xy,np.full(len(xy),SHAFT_TOP)])@rz(s.fixture_angle).T+np.r_[SHAFT_XY,0.]
    local=(world-t)@r;h=s.spec['height']/2
    overlap=float(np.min(local[:,2]+h))
    if s.kind=='gear':
        outside=local[np.linalg.norm(local[:,:2],axis=1)>s.spec['bore']/2]
        clearance=min(float(np.min(outside[:,0])),DATUMS['gear']['root']-float(np.max(outside[:,0])),DATUMS['gear']['width']/2-float(np.max(np.abs(outside[:,1])))) if len(outside) else -1.
    else:clearance=min(float(np.min(local[:,0]))-DATUMS['bearing']['root'],DATUMS['bearing']['width']/2-float(np.max(np.abs(local[:,1]))))
    return dict(index_overlap_m=overlap,index_clearance_m=clearance,angle_to_actual_fixture=error(r,rz(s.fixture_angle)))


class PhysicalScore(PreviousScore):
    def __init__(self,scene):
        super().__init__(scene);self.index_contact_seen=False;self.index_penetration=0.;self.deepest_engagement=-1.;self.insertion_active=False
        self.wrong_angle_passage=False;self.index_entry_seen=False;self.fixture_rotation=rz(scene.fixture_angle)

    def observe(self):
        super().observe();s=self.scene
        if self.insertion_active:
            r=s.data.xmat[s.body_id].reshape(3,3);tip=r.T@(np.r_[SHAFT_XY,SHAFT_TOP]-s.data.xpos[s.body_id])
            self.deepest_engagement=max(self.deepest_engagement,float(tip[2]+s.spec['height']/2))
            if tip[2]+s.spec['height']/2>=LIMITS['min_index_overlap_m'] and error(r,self.fixture_rotation)['rotation_deg']>15:
                self.wrong_angle_passage=True
            if not self.index_entry_seen and s.feature and tip[2]+s.spec['height']/2>=LIMITS['min_index_overlap_m']:
                fit=index_geometry(s)
                self.index_entry_seen=bool(fit['index_overlap_m']>=LIMITS['min_index_overlap_m'] and fit['index_clearance_m']>=-LIMITS['index_fit_tolerance_m'])
        for c in s.data.contact:
            ids=(int(c.geom1),int(c.geom2))
            if s.index_id in ids and s.body_id in [int(s.model.geom_bodyid[g]) for g in ids]:
                self.index_contact_seen=True;self.index_penetration=max(self.index_penetration,float(-c.dist))

    def result(self,attempted,abort):
        p=super().result(attempted,abort);fit=index_geometry(self.scene)
        p.update(fit,index_contact_seen=self.index_contact_seen,max_index_penetration_m=self.index_penetration,deepest_engagement_m=self.deepest_engagement,
                 wrong_angle_passage=self.wrong_angle_passage,index_entry_seen=self.index_entry_seen)
        p['keyed_success']=bool(p['success'] and self.scene.feature and self.index_entry_seen and fit['index_overlap_m']>=LIMITS['min_index_overlap_m']
            and fit['index_clearance_m']>=-LIMITS['index_fit_tolerance_m'] and self.index_penetration<=LIMITS['max_index_penetration_m'])
        p['retracted_and_held']=bool(p['held'] and p['engagement_m']<=-LIMITS['retracted_clearance_m'] and not p['illegal_contacts'])
        return p


def summarize(rows):
    normal=[r for r in rows if r['feature'] and not r['empty'] and not r['fixture_offset_deg'] and r['case']!='blind']
    faults=[r for r in rows if r['fixture_offset_deg']];fault_attempts=[r for r in faults if r['insert_attempted']]
    attempted=[r for r in rows if r['insert_attempted']];normal_attempts=[r for r in normal if r['insert_attempted']]
    unknown=[r for r in rows if not r['feature']];blind=[r for r in rows if r['case']=='blind'];empty=[r for r in rows if r['empty']]
    clear=[r for r in rows if r['feature'] and not r['empty'] and r['case']!='blind']
    measured=[r for r in clear if r['observations'] and r['observations'][0]['measurement']]
    return dict(episodes=len(rows),outcomes=dict(Counter(r['status'] for r in rows)),
        normal_keyed_success=rate(sum(r['success'] for r in normal),len(normal)),
        normal_success_given_attempt=rate(sum(r['success'] for r in normal_attempts),len(normal_attempts)),
        fault_attempt_coverage=rate(len(fault_attempts),len(faults)),
        fault_contact_stop_retract=rate(sum(r['fault_contained'] for r in fault_attempts),len(fault_attempts)),
        fault_contained_all=rate(sum(r['fault_contained'] for r in faults),len(faults)),
        fault_unexpected_mating=rate(sum(r['physical']['keyed_success'] for r in faults),len(faults)),
        fault_unexpected_entry=rate(sum(r['physical']['index_entry_seen'] for r in faults),len(faults)),
        wrong_angle_physical_passage=rate(sum(r['physical']['wrong_angle_passage'] for r in attempted),len(attempted)),
        visual_wrong_permission=rate(sum(not r['preinsert_gt_recipe_aligned'] for r in attempted),len(attempted)),
        no_datum_with_correction_or_insertion=rate(sum(bool(r['corrections'] or r['insert_attempted']) for r in unknown),len(unknown)),
        blind_with_observation=rate(sum(r['observation_moves']>0 for r in blind),len(blind)),
        empty_with_observation=rate(sum(r['observation_moves']>0 for r in empty),len(empty)),
        initial_coverage=rate(len(measured),len(clear)),initial_rotation_deg=distribution([r['observations'][0]['error']['rotation_deg'] for r in measured]),
        force_stop_peak_n=distribution([r['wrench']['trigger']['force_delta_n'] for r in rows if r['wrench']['trigger']]),
        wall_s=distribution([r['wall_s'] for r in rows]),
        note='Fault fixtures are intentionally misindexed relative to the declared recipe. Visual recipe conformity and actual mechanical compatibility are separate. All refusals retained.')
