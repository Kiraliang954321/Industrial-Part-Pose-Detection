"""Freeze the complete A4 development artifacts before new physical episodes."""
import argparse
import json
from pathlib import Path
from yolo_conveyor.orientation.common import sha256,unlinked,write_json,environment,utc_now,dataset_file
from yolo_conveyor.pose_cell.scene import SETUP
from yolo_conveyor.pose_cell.metrology import LIMITS
from yolo_conveyor.pose_cell.runtime import POLICY as EXECUTION,TARGETS
from .runtime import CASES
from .policy import POLICY
from .training import BASELINE_SHA
from .data import audit_dataset,read
from .provenance import sources


def assignment(seed):return dict(seed=seed,cases=list(CASES),targets=TARGETS,episodes=2*len(CASES)*len(TARGETS))


def freeze(args):
    cal=unlinked(args.calibration);val=unlinked(args.validation);training=unlinked(args.training);output=unlinked(args.output)
    if output.exists():raise FileExistsError('protocol already exists')
    cm=read(cal/'manifest.json');cs=read(cal/'summary.json');vm=read(val/'manifest.json');tm=read(training/'manifest.json');ts=read(training/'training-summary.json')
    dataset=unlinked(tm['arguments']['dataset']);weights=unlinked(cm['arguments']['weights'])
    if cs['status']!='completed' or cs['mode']!='calibration' or ts['status']!='completed':raise ValueError('completed development required')
    if cm['sources']!=sources() or vm['sources']!=sources():raise ValueError('development source drift')
    if cm['policy']!=POLICY or cm['execution']!=EXECUTION or cm['limits']!=LIMITS or cm['setup']!=SETUP:raise ValueError('development configuration drift')
    if audit_dataset(dataset)['errors'] or tm['dataset_sha256']!=sha256(dataset/'dataset.json') or vm['dataset_sha256']!=tm['dataset_sha256']:raise ValueError('dataset provenance mismatch')
    digest=sha256(weights)
    if digest!=cm['weights_sha256'] or digest!=vm['weights_sha256'] or digest not in {r['sha256'] for r in ts['checkpoints'].values()}:raise ValueError('selected model provenance mismatch')
    if cm['baseline_sha256']!=BASELINE_SHA or vm['baseline_sha256']!=BASELINE_SHA:raise ValueError('baseline provenance')
    rows=read(cal/'results.json');count=len(CASES)*len(TARGETS)
    if not 251000<=args.seed<251873 or sorted(r['index'] for r in rows)!=list(range(2*count)):raise ValueError('evaluation assignment')
    for r in rows:
        k,i=divmod(r['index'],count)
        if (r['kind']!=('bearing','gear')[k] or r['case']!=CASES[i%len(CASES)] or r['seed']!=cm['arguments']['seed']+100*k+i
                or r['target_spin_deg']!=TARGETS[i//len(CASES)]):raise ValueError('calibration matrix mismatch')
        if len(r['observation_attempts'])>POLICY['max_observation_moves'] or r['observation_moves']>POLICY['max_observation_moves']:raise ValueError('observation budget violated')
    for metric in ('wrong_insertion_permission','no_datum_with_correction_or_insertion','blind_with_observation','empty_with_observation'):
        if cs['overall'][metric]['x']:raise ValueError('calibration guard failed: '+metric)
    if any(e['seed']>=241000 for e in read(dataset/'episodes.json')):raise ValueError('data/calibration seed overlap')
    for relative,h in read(cal/'artifact-hashes.json').items():
        if sha256(dataset_file(cal,relative))!=h:raise ValueError('calibration artifact drift')
    paths=[weights,Path(cm['arguments']['baseline_weights']).resolve(),dataset/'dataset.json',dataset/'records.json',dataset/'episodes.json',
        training/'manifest.json',training/'training-summary.json',val/'manifest.json',val/'summary.json',val/'results.json',
        cal/'manifest.json',cal/'summary.json',cal/'results.json',cal/'artifact-hashes.json']
    for path,h in vm['baseline_cache'].items():
        if sha256(Path(path))!=h:raise ValueError('baseline cache drift')
        paths.append(Path(path))
    p=dict(status='FROZEN_A4_CUSTOM_PROXY_SIMULATION',real_formal_gate='closed',created_at=utc_now(),sources=sources(),
        setup=SETUP,policy=POLICY,execution=EXECUTION,limits=LIMITS,weights_sha256=digest,baseline_sha256=BASELINE_SHA,
        asset_sha256=cm['asset_sha256'],packages=environment()['packages'],artifacts={str(p):sha256(p) for p in paths},evaluation=assignment(args.seed),
        rule='No tuning on evaluation. A3 paired comparison uses the same initial RGB only. Observation, correction and insertion are separate actions; GT only scores.')
    output.parent.mkdir(parents=True,exist_ok=True);write_json(output,p);return dict(path=str(output),sha256=sha256(output),evaluation=p['evaluation'])


def validate(path,args):
    if not path:raise ValueError('evaluation requires frozen A4 protocol')
    p=read(unlinked(path))
    if p['status']!='FROZEN_A4_CUSTOM_PROXY_SIMULATION' or p['real_formal_gate']!='closed':raise ValueError('scope mismatch')
    if p['sources']!=sources() or p['setup']!=SETUP or p['policy']!=POLICY or p['execution']!=EXECUTION or p['limits']!=LIMITS:raise ValueError('frozen source/configuration drift')
    for path,h in p['artifacts'].items():
        if sha256(unlinked(path))!=h:raise ValueError('frozen artifact drift')
    if sha256(Path(args.weights))!=p['weights_sha256'] or sha256(Path(args.baseline_weights))!=p['baseline_sha256']:raise ValueError('weight drift')
    if sha256(Path(args.assets)/'asset-manifest.json')!=p['asset_sha256'] or environment()['packages']!=p['packages']:raise ValueError('asset/environment drift')
    if assignment(args.seed)!=p['evaluation']:raise ValueError('evaluation seed or matrix drift')
    return p


def main():
    p=argparse.ArgumentParser()
    for n in ('calibration','validation','training','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--seed',type=int,default=251000);print(json.dumps(freeze(p.parse_args()),indent=2));return 0
