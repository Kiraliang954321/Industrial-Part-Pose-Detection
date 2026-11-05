from collections import Counter
import numpy as np
from yolo_conveyor.orientation.evaluation import rate,distribution
from yolo_conveyor.annular.data import physical_label
from yolo_conveyor.annular.evaluation import PhysicalScore
from .geometry import error,describe


def truth(scene,target):
    label=physical_label(scene);r=scene.data.xmat[scene.body_id].reshape(3,3).copy()
    return dict(held=label['held'],rotation=r.tolist(),center=scene.data.xpos[scene.body_id].tolist(),angles=describe(r),target_error=error(r,target))


def summarize(rows):
    eligible=[r for r in rows if r['feature']];initial=[r for r in eligible if r['observations'] and r['observations'][0]['measurement']]
    tried=[r for r in rows if r['insert_attempted']];alignment=[r for r in eligible if r['corrections']>0]
    invalid=[r for r in rows if not r['feature']]
    unsafe=[r for r in tried if not r['preinsert_gt_aligned']]
    clear=[r for r in eligible if r['case']!='blind']
    return dict(episodes=len(rows),outcomes=dict(Counter(r['status'] for r in rows)),
                initial_coverage=rate(len(initial),len(eligible)),
                initial_rotation_deg=distribution([r['observations'][0]['error']['rotation_deg'] for r in initial]),
                initial_position_mm=distribution([r['observations'][0]['position_error_mm'] for r in initial]),
                aligned_after_correction=rate(sum(r['aligned_gt'] for r in alignment),len(alignment)),
                preposition_given_attempt=rate(sum(r['success'] for r in tried),len(tried)),
                closed_loop_success_all_locatable=rate(sum(r['success'] for r in eligible),len(eligible)),
                closed_loop_success_clear_featured=rate(sum(r['success'] for r in clear),len(clear)),
                wrong_insertion_permission=rate(len(unsafe),len(tried)),
                no_datum_with_action=rate(sum(r['corrections']>0 or r['insert_attempted'] for r in invalid),len(invalid)),
                wall_s=distribution([r['wall_s'] for r in rows]),
                note='All correction targets are visual estimates plus robot FK; independent GT only scores results. No continuous throughput claim.')
