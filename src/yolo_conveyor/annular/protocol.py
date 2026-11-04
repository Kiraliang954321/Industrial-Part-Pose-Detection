"""Freeze the simple A1 recipe and detector before unseen physical job evaluation."""
import argparse
import json
from pathlib import Path
from yolo_conveyor.orientation.common import sha256,unlinked,write_json,environment,utc_now,dataset_file
from .assets import PARTS,NAMES
from .scene import SETUP
from .perception import POLICY
from .runtime import CASES
from .provenance import sources


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def freeze(args):
    calibration=unlinked(args.calibration);training=unlinked(args.training);output=unlinked(args.output)
    if output.exists():raise FileExistsError('use a new protocol file')
    if not (131000<=args.seed<132000-args.episodes_per_part-100 and 10<=args.episodes_per_part<=30 and args.episodes_per_part%5==0):
        raise ValueError('evaluation assignment')
    manifest=read(calibration/'manifest.json');summary=read(calibration/'summary.json');rows=read(calibration/'results.json')
    trained=read(training/'training-summary.json');train_manifest=read(training/'manifest.json')
    if summary['status']!='completed' or summary['mode']!='calibration' or trained['status']!='completed':raise ValueError('completed calibration/training required')
    if manifest['sources']!=sources() or manifest['parts']!=PARTS or manifest['setup']!=SETUP or manifest['policy']!=POLICY:
        raise ValueError('calibration implementation drift')
    for kind in PARTS:
        if summary['by_part'][kind]['success_given_attempt']['x']==0:raise ValueError('both parts need demonstrated pre-positioning')
    weights=unlinked(trained['checkpoints']['best']['path'])
    if sha256(weights)!=trained['checkpoints']['best']['sha256'] or sha256(weights)!=manifest['weights_sha256']:raise ValueError('selected checkpoint mismatch')
    dataset=unlinked(train_manifest['arguments']['dataset'])
    if sha256(dataset/'dataset.json')!=train_manifest['dataset_sha256']:raise ValueError('training dataset drift')
    used={r['seed'] for r in rows}|{e['seed'] for e in read(dataset/'episodes.json')}
    planned={args.seed+k*100+i for k in range(2) for i in range(args.episodes_per_part)}
    if planned&used:raise ValueError('evaluation seed leakage')
    for relative,expected in read(calibration/'artifact-hashes.json').items():
        if sha256(dataset_file(calibration,relative))!=expected:raise ValueError('calibration artifact drift')
    artifacts={str(path):sha256(path) for path in (weights,training/'training-summary.json',training/'manifest.json',dataset/'dataset.json',
               calibration/'manifest.json',calibration/'summary.json',calibration/'results.json',calibration/'artifact-hashes.json')}
    protocol=dict(status='FROZEN_A1_SIMULATION_ONLY',real_formal_gate='closed',created_at=utc_now(),sources=sources(),
                  setup=SETUP,policy=POLICY,parts=PARTS,classes=NAMES,artifacts=artifacts,weights_sha256=sha256(weights),
                  asset_sha256=manifest['asset_sha256'],packages=environment()['packages'],
                  scene_hashes={kind:sha256(calibration/f'{kind}-scene.xml') for kind in PARTS},
                  evaluation=dict(seed=args.seed,episodes_per_part=args.episodes_per_part,cases=list(CASES)),
                  rules='No tuning on evaluation outcomes; proxies and clearance positioning only, not real acceptance')
    output.parent.mkdir(parents=True,exist_ok=True);write_json(output,protocol)
    return dict(path=str(output),sha256=sha256(output),evaluation=protocol['evaluation'])


def validate(path,args):
    if not path:raise ValueError('evaluation requires protocol')
    p=read(unlinked(path))
    if p['status']!='FROZEN_A1_SIMULATION_ONLY' or p['real_formal_gate']!='closed':raise ValueError('protocol scope')
    if (p['sources']!=sources() or p['setup']!=SETUP or p['policy']!=POLICY or p['parts']!=PARTS or p['classes']!=NAMES):
        raise ValueError('frozen source/configuration drift')
    for item,expected in p['artifacts'].items():
        if sha256(unlinked(item))!=expected:raise ValueError('artifact drift')
    if sha256(Path(args.weights))!=p['weights_sha256'] or sha256(Path(args.assets)/'asset-manifest.json')!=p['asset_sha256']:
        raise ValueError('model/robot mismatch')
    if environment()['packages']!=p['packages']:raise ValueError('environment drift')
    if dict(seed=args.seed,episodes_per_part=args.episodes_per_part,cases=list(CASES))!=p['evaluation']:raise ValueError('evaluation assignment mismatch')
    return p


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('calibration','training','output'):p.add_argument('--'+flag,required=True)
    p.add_argument('--seed',type=int,default=131000);p.add_argument('--episodes-per-part',type=int,default=20)
    print(json.dumps(freeze(p.parse_args()),indent=2));return 0
