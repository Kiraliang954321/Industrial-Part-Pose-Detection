from collections import Counter
from yolo_conveyor.orientation.evaluation import rate,distribution
from yolo_conveyor.pose_cell.evaluation import PhysicalScore,truth


def summarize(rows):
    eligible=[r for r in rows if r['feature'] and not r['empty']];clear=[r for r in eligible if r['case']!='blind']
    initial=[r for r in eligible if r['observations'] and r['observations'][0]['measurement']]
    tried=[r for r in rows if r['insert_attempted']];corrected=[r for r in eligible if r['corrections']]
    missed=[r for r in clear if not r['observations'] or not r['observations'][0]['measurement']]
    negative=[r for r in rows if not r['feature']];empty=[r for r in rows if r['empty']];blind=[r for r in rows if r['case']=='blind']
    paired=[r for r in clear if r.get('baseline_initial') is not None]
    def recovered_error(row):
        return next((o['error']['rotation_deg'] for o in row['observations'][1:] if o['accepted'] and o['corrections_before']==0 and o['error']),None)
    return dict(episodes=len(rows),outcomes=dict(Counter(r['status'] for r in rows)),initial_coverage=rate(len(initial),len(eligible)),
        initial_rotation_deg=distribution([r['observations'][0]['error']['rotation_deg'] for r in initial]),
        initial_position_mm=distribution([r['observations'][0]['position_error_mm'] for r in initial]),
        active_recovery_initial_misses=rate(sum(r['recovered_initial'] for r in missed),len(missed)),
        active_correct_recovery_initial_misses=rate(sum(recovered_error(r) is not None and recovered_error(r)<=10 for r in missed),len(missed)),
        completed_after_initial_miss=rate(sum(r['success'] for r in missed),len(missed)),
        verification_recoveries=sum(r['recovered_verification'] for r in rows),
        observation_move_counts=dict(Counter(r['observation_moves'] for r in rows)),
        observation_seconds=distribution([r['observation_sim_s'] for r in rows if r['observation_moves']]),
        observation_extra_wall_s=distribution([r['observation_extra_wall_s'] for r in rows if r['observation_moves']]),
        aligned_after_correction=rate(sum(r['aligned_gt'] for r in corrected),len(corrected)),
        preposition_given_attempt=rate(sum(r['success'] for r in tried),len(tried)),
        closed_loop_success_all_locatable=rate(sum(r['success'] for r in eligible),len(eligible)),
        closed_loop_success_clear_featured=rate(sum(r['success'] for r in clear),len(clear)),
        wrong_insertion_permission=rate(sum(not r['preinsert_gt_aligned'] for r in tried),len(tried)),
        no_datum_with_correction_or_insertion=rate(sum(bool(r['corrections'] or r['insert_attempted']) for r in negative),len(negative)),
        no_datum_with_observation=rate(sum(r['observation_moves']>0 for r in negative),len(negative)),
        blind_with_observation=rate(sum(r['observation_moves']>0 for r in blind),len(blind)),
        empty_with_observation=rate(sum(r['observation_moves']>0 for r in empty),len(empty)),
        paired_initial_rgb=dict(episodes=len(paired),a3_coverage=rate(sum(r['baseline_initial']['measurement'] is not None for r in paired),len(paired)),
            a4_coverage=rate(sum(r['observations'][0]['measurement'] is not None for r in paired),len(paired)),
            a3_within_10deg_all=rate(sum(r['baseline_initial']['error'] is not None and r['baseline_initial']['error']['rotation_deg']<=10 for r in paired),len(paired)),
            a4_within_10deg_all=rate(sum(r['observations'][0]['error'] is not None and r['observations'][0]['error']['rotation_deg']<=10 for r in paired),len(paired)),
            a3_rotation_deg=distribution([r['baseline_initial']['error']['rotation_deg'] for r in paired if r['baseline_initial']['error']]),
            a4_rotation_deg=distribution([r['observations'][0]['error']['rotation_deg'] for r in paired if r['observations'][0]['error']]),
            note='Same initial RGB only; not a paired physical A3 execution'),
        wall_s=distribution([r['wall_s'] for r in rows]),note='Observation turns, corrective turns and insertion are separate. Wall time includes baseline inference, physics computation and recording; not continuous production throughput. GT never selects an action.')
