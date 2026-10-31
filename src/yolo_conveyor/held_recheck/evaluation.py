"""Separate GT audit for initial compliance, actual relative slip and final insertion."""
from collections import Counter
from yolo_conveyor.held.evaluation import truth
from yolo_conveyor.orientation.evaluation import rate,distribution
from .inspection import tool_relative,relative_change,POLICY


def capture_truth(scene):
    value=truth(scene)
    value['relative']=tool_relative(value['estimate'],scene.tool_transform())
    return value


def audit_recheck(before,after):
    delta=relative_change(before['relative'],after['relative'])
    changed=delta['shift_m']>POLICY['gt_relative_shift_m'] or delta['axis_deg']>POLICY['gt_relative_axis_deg']
    return dict(gt='CONFORMING' if after['gt']=='CONFORMING' and not changed else 'NONCONFORMING',
                actual_relative_change=delta,significant_change=changed,held=after['held'])


def summarize(rows):
    bad=[r for r in rows if r['initial_gt']['gt']=='NONCONFORMING']
    good=[r for r in rows if r['initial_gt']['gt']=='CONFORMING' and r['case']!='initial_blackout']
    reached=[r for r in rows if r['post_gt'] is not None]
    unsafe=[r for r in reached if r['post_gt']['gt']=='NONCONFORMING']
    attempted=[r for r in rows if r['result']['attempted']]
    changed=[r for r in reached if r['post_gt']['significant_change'] and r['post_gt']['held']]
    return dict(episodes=len(rows),initial_decisions=dict(Counter(r['initial']['decision'] for r in rows)),
                recheck_decisions=dict(Counter(r['recheck']['decision'] for r in rows if r['recheck'])),
                initial_wrong_accept=rate(sum(r['initial']['decision']=='PASS' for r in bad),len(bad)),
                matched_clear_conforming_initial=dict(n=len(good),raw_h1_pass=sum(r['raw_h1']['decision']=='PASS' for r in good),
                                                     calibrated_pass=sum(r['initial']['decision']=='PASS' for r in good)),
                unsafe_recheck_release=rate(sum(r['recheck'] is not None and r['recheck']['decision']=='PASS' for r in unsafe),len(unsafe)),
                held_significant_change_blocked=rate(sum(not r['result']['attempted'] for r in changed),len(changed)),
                insertion_success_given_attempt=rate(sum(r['result']['preinsert_success'] for r in attempted),len(attempted)),
                insertion_success_all=rate(sum(r['result']['preinsert_success'] for r in rows),len(rows)),
                permit_denials=dict(Counter(r['permit_denied'] for r in rows if r['permit_denied'])),
                aborts=dict(Counter(r['abort'] for r in rows if r['abort'])),
                initial_inference_ms=distribution([r['initial_inference_ms'] for r in rows]),
                recheck_inference_ms=distribution([r['recheck_inference_ms'] for r in rows if r['recheck_inference_ms'] is not None]),
                wall_s=distribution([r['wall_s'] for r in rows]),simulated_s=distribution([r['simulated_s'] for r in rows]),
                robot_collision_episodes=sum(bool(r['result']['illegal_contacts']) for r in rows),
                continuous_throughput='not measured; independent episodes, with injected faults')
