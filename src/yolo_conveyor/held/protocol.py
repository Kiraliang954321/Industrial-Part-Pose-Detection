"""Freeze H1 simulation development inputs before independent episode evaluation."""
import argparse
import json
from pathlib import Path
from yolo_conveyor.orientation.common import sha256,write_json,unlinked,utc_now,environment
from .provenance import held_sources
from .scene import SETUP
from .gate import POLICY
from .runtime import CASES


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def freeze(args):
    training=unlinked(args.training);calibration=unlinked(args.calibration)
    output=unlinked(args.output)
    if output.exists():raise FileExistsError('protocol already exists; create a new version')
    if not (args.episodes%len(CASES)==0 and 8<=args.episodes<=64 and 71000<=args.seed<72000-args.episodes):
        raise ValueError('invalid independent evaluation plan')
    trained=read(training/'training-summary.json');training_manifest=read(training/'run-manifest.json')
    calibrated=read(calibration/'summary.json');cal_manifest=read(calibration/'run-manifest.json')
    if trained['status']!='completed' or calibrated['status']!='completed' or calibrated['mode']!='calibration':
        raise ValueError('completed training and calibration required')
    if cal_manifest['sources']!=held_sources() or cal_manifest['policy']!=POLICY or cal_manifest['setup']!=SETUP:
        raise ValueError('calibration does not match current sources/config; rerun calibration before freeze')
    best=unlinked(trained['checkpoints']['best']['path'])
    if sha256(best)!=trained['checkpoints']['best']['sha256'] or sha256(best)!=cal_manifest['weights_sha256']:
        raise ValueError('checkpoint/calibration mismatch')
    dataset=unlinked(training_manifest['arguments']['dataset'])
    if sha256(dataset/'dataset.json')!=training_manifest['dataset_sha256']:raise ValueError('training data drift')
    data_seeds={e['seed'] for e in read(dataset/'episodes.json')}
    cal_rows=read(calibration/'results.json');cal_seeds={r['seed'] for r in cal_rows}
    seeds=set(range(args.seed,args.seed+args.episodes))
    if seeds&(data_seeds|cal_seeds) or cal_seeds&data_seeds:raise ValueError('seed/episode leakage')
    overall=calibrated['overall']
    if len(cal_rows)<24 or overall['preinsert_success_given_attempt']['x']==0:
        raise ValueError('need at least 24 calibration episodes and demonstrated pre-insertion')
    artifacts={}
    for path in (best,training/'training-summary.json',training/'run-manifest.json',dataset/'dataset.json',dataset/'audit.json',
                 calibration/'summary.json',calibration/'results.json',calibration/'run-manifest.json'):
        artifacts[str(path)]=sha256(path)
    termination_note=training/'training-termination-note.json'
    if termination_note.exists():artifacts[str(termination_note)]=sha256(termination_note)
    protocol=dict(version=1,status='FROZEN_SIMULATION_DEVELOPMENT_ONLY',real_formal_gate='closed',created_at=utc_now(),
                  sources=held_sources(),setup=SETUP,policy=POLICY,artifacts=artifacts,weights_sha256=sha256(best),
                  asset_manifest_sha256=cal_manifest['asset_manifest_sha256'],scene_sha256=cal_manifest['scene_sha256'],
                  package_versions=environment()['packages'],calibration=str(calibration),training=str(training),
                  evaluation=dict(seed=args.seed,episodes=args.episodes,cases=list(CASES)),
                  calibration_overall=overall,selection_rule='Development Val best checkpoint; calibration then frozen new physical episodes',
                  limitations=calibrated['limitations'])
    output.parent.mkdir(parents=True,exist_ok=True);write_json(output,protocol)
    return dict(protocol=str(output),sha256=sha256(output),evaluation=protocol['evaluation'])


def validate(path,args):
    if not path:raise ValueError('independent evaluation requires --protocol')
    protocol=read(unlinked(path))
    if protocol.get('status')!='FROZEN_SIMULATION_DEVELOPMENT_ONLY' or protocol.get('real_formal_gate')!='closed':
        raise ValueError('wrong protocol scope')
    if protocol['sources']!=held_sources() or protocol['setup']!=SETUP or protocol['policy']!=POLICY:
        raise ValueError('frozen source/config drift')
    for item,expected in protocol['artifacts'].items():
        if sha256(unlinked(item))!=expected:raise ValueError('frozen artifact drift: '+item)
    if sha256(Path(args.weights))!=protocol['weights_sha256']:raise ValueError('frozen weight drift')
    if sha256(Path(args.assets)/'asset-manifest.json')!=protocol['asset_manifest_sha256']:raise ValueError('frozen assets drift')
    if environment()['packages']!=protocol['package_versions']:raise ValueError('frozen environment drift')
    if dict(seed=args.seed,episodes=args.episodes,cases=list(CASES))!=protocol['evaluation']:
        raise ValueError('evaluation plan does not match freeze')
    return protocol


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('training','calibration','output'):p.add_argument('--'+flag,required=True)
    p.add_argument('--seed',type=int,default=71000);p.add_argument('--episodes',type=int,default=24)
    print(json.dumps(freeze(p.parse_args()),indent=2));return 0
