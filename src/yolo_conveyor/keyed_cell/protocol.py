"""A5 freeze includes actual mechanical reference controls and visual calibration."""
import argparse
import json
from pathlib import Path
from yolo_conveyor.orientation.common import sha256,unlinked,write_json,environment,utc_now,dataset_file
from yolo_conveyor.pose_cell.runtime import POLICY as EXECUTION,TARGETS
from yolo_conveyor.active_pose.policy import POLICY as OBSERVATION_POLICY
from .runtime import CASES,OFFSETS,WEIGHTS_SHA
from .scene import SETUP
from .guard import GUARD
from .evaluation import LIMITS
from .provenance import sources


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def assignment(seed):return dict(seed=seed,cases=list(CASES),targets=TARGETS,offsets=OFFSETS,episodes=2*len(CASES)*len(TARGETS))


def freeze(args):
    cal=unlinked(args.calibration);ref=unlinked(args.mechanical_reference);output=unlinked(args.output)
    if output.exists():raise FileExistsError('protocol exists')
    m=read(cal/'manifest.json');summary=read(cal/'summary.json');rm=read(ref/'manifest.json');refs=read(ref/'results.json');rows=read(cal/'results.json')
    if summary['status']!='completed' or summary['mode']!='calibration':raise ValueError('complete calibration required')
    if m['sources']!=sources() or rm['sources']!=sources():raise ValueError('reference/calibration source drift')
    if m['setup']!=SETUP or m['guard']!=GUARD or m['score_limits']!=LIMITS or m['execution']!=EXECUTION or m['observation_policy']!=OBSERVATION_POLICY:raise ValueError('configuration drift')
    if rm['scope']!='A5_ORACLE_MECHANICAL_REFERENCE_NOT_NEURAL':raise ValueError('mechanical reference scope')
    for kind in ('bearing','gear'):
        subset=[r for r in refs if r['kind']==kind]
        if {r['case'] for r in subset}!={'matched','wrong30','wrong90','wrong180','no_datum'}:raise ValueError('mechanical reference matrix incomplete')
        for r in subset:
            if r['failure']:raise ValueError('reference execution failure')
            if r['case']=='matched':
                if not r['physical']['keyed_success']:raise ValueError('matched mechanical reference failed')
            elif not (r['blocked'] and r['physical']['index_contact_seen'] and r['physical']['retracted_and_held']
                      and r['physical']['deepest_engagement_m']<LIMITS['min_index_overlap_m']
                      and r['physical']['max_index_penetration_m']<=LIMITS['max_index_penetration_m']
                      and not r['physical']['wrong_angle_passage'] and not r['physical']['index_entry_seen']):raise ValueError('wrong/no-datum part was not mechanically stopped')
    count=len(CASES)*len(TARGETS)
    if not 281000<=args.seed<281867 or sorted(r['index'] for r in rows)!=list(range(2*count)):raise ValueError('evaluation matrix')
    for r in rows:
        k,i=divmod(r['index'],count)
        if (r['kind']!=('bearing','gear')[k] or r['case']!=CASES[i%len(CASES)] or r['seed']!=m['arguments']['seed']+100*k+i
                or r['recipe_angle_deg']!=TARGETS[i//len(CASES)] or r['fixture_offset_deg']!=OFFSETS.get(r['case'],0.)):raise ValueError('calibration assignment mismatch')
    for metric in ('fault_unexpected_mating','fault_unexpected_entry','wrong_angle_physical_passage','visual_wrong_permission','no_datum_with_correction_or_insertion','blind_with_observation','empty_with_observation'):
        if summary['overall'][metric]['x']:raise ValueError('calibration guard failed: '+metric)
    for kind in ('bearing','gear'):
        if summary['by_part'][kind]['normal_keyed_success']['x']==0 or summary['by_part'][kind]['fault_contact_stop_retract']['x']==0:raise ValueError('no physical demonstration for '+kind)
    weights=unlinked(m['arguments']['weights'])
    if sha256(weights)!=WEIGHTS_SHA:raise ValueError('visual model drift')
    for rel,h in read(cal/'artifact-hashes.json').items():
        if sha256(dataset_file(cal,rel))!=h:raise ValueError('calibration artifact drift')
    paths=[weights,cal/'manifest.json',cal/'summary.json',cal/'results.json',cal/'artifact-hashes.json',ref/'manifest.json',ref/'results.json',Path('docs/data/active_pose_a4_protocol_v1.json').resolve()]
    p=dict(status='FROZEN_A5_INDEXED_PROXY_SIMULATION',real_formal_gate='closed',created_at=utc_now(),sources=sources(),setup=SETUP,guard=GUARD,
        score_limits=LIMITS,execution=EXECUTION,observation_policy=OBSERVATION_POLICY,weights_sha256=WEIGHTS_SHA,asset_sha256=m['asset_sha256'],
        packages=environment()['packages'],artifacts={str(p):sha256(p) for p in paths},evaluation=assignment(args.seed),
        rule='No tuning after freeze. Fixture faults intentionally differ from the known recipe. Wrist sensor controls contact stop; object/key truth is only independent scoring. Mechanical references are not neural results.')
    output.parent.mkdir(parents=True,exist_ok=True);write_json(output,p);return dict(path=str(output),sha256=sha256(output),evaluation=p['evaluation'])


def validate(path,args):
    if not path:raise ValueError('frozen protocol required')
    p=read(unlinked(path))
    if p['status']!='FROZEN_A5_INDEXED_PROXY_SIMULATION' or p['real_formal_gate']!='closed':raise ValueError('scope drift')
    if p['sources']!=sources() or p['setup']!=SETUP or p['guard']!=GUARD or p['score_limits']!=LIMITS or p['execution']!=EXECUTION or p['observation_policy']!=OBSERVATION_POLICY:raise ValueError('frozen source/configuration drift')
    for name,h in p['artifacts'].items():
        if sha256(unlinked(name))!=h:raise ValueError('frozen artifact drift')
    if sha256(Path(args.weights))!=p['weights_sha256'] or sha256(Path(args.assets)/'asset-manifest.json')!=p['asset_sha256']:raise ValueError('runtime input drift')
    if environment()['packages']!=p['packages'] or assignment(args.seed)!=p['evaluation']:raise ValueError('environment or evaluation assignment drift')
    return p


def main():
    p=argparse.ArgumentParser()
    for name in ('calibration','mechanical-reference','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--seed',type=int,default=281000);print(json.dumps(freeze(p.parse_args()),indent=2));return 0
