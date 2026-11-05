"""Independent angular error and target-decision scoring."""
from collections import Counter
from yolo_conveyor.orientation.evaluation import distribution,rate


def summarize(rows):
    eligible=[r for r in rows if r['gt']!='INDETERMINATE'];usable=[r for r in eligible if r['measurement']]
    bad=[r for r in eligible if r['gt']=='NONCONFORMING'];good=[r for r in eligible if r['gt']=='CONFORMING']
    unknown=[r for r in rows if r['gt']=='INDETERMINATE']
    return dict(episodes=len(rows),decisions=dict(Counter(r['decision']['decision'] for r in rows)),coverage=rate(len(usable),len(eligible)),
                wrong_accept=rate(sum(r['decision']['decision']=='PASS' for r in bad),len(bad)),
                conforming_not_passed=rate(sum(r['decision']['decision']!='PASS' for r in good),len(good)),
                indeterminate_with_pose=rate(sum(r['measurement'] is not None for r in unknown),len(unknown)),
                axis_error_deg=distribution([r['measurement_error']['axis_deg'] for r in usable]),
                spin_error_deg=distribution([r['measurement_error']['spin_deg'] for r in usable]),
                rotation_error_deg=distribution([r['measurement_error']['rotation_deg'] for r in usable]),
                inference_ms_three_views=distribution([r['inference_ms'] for r in rows]),
                wall_s=distribution([r['wall_s'] for r in rows]),
                notes='Pose measurement and target comparison only; no automated correction/assembly claim')
