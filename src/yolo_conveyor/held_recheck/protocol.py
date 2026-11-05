"""Freeze calibrated H2 artifacts and reserve new evaluation episodes."""
import argparse
import json
from pathlib import Path
from yolo_conveyor.orientation.common import sha256,unlinked,write_json,environment,utc_now,dataset_file
from .provenance import sources
from .scene import SETUP
from .inspection import POLICY
from .faults import FAULTS
from .runtime import CASES
from .calibration import WEIGHT_SHA


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def freeze(args):
    calibration=unlinked(args.calibration);output=unlinked(args.output)
    if output.exists():raise FileExistsError('protocol exists; use a new version')
    if not (args.episodes%len(CASES)==0 and 36<=args.episodes<=96 and 91000<=args.seed<92000-args.episodes):
        raise ValueError('invalid independent evaluation plan')
    manifest=read(calibration/'manifest.json');report=read(calibration/'summary.json');rows=read(calibration/'results.json')
    if report['status']!='completed' or report['mode']!='calibration' or len(rows)<36:
        raise ValueError('need 36 completed calibration episodes')
    if manifest['sources']!=sources() or manifest['policy']!=POLICY or manifest['setup']!=SETUP or manifest['faults']!=FAULTS:
        raise ValueError('calibrated source/configuration drift')
    if report['overall']['insertion_success_given_attempt']['x']==0 or report['overall']['held_significant_change_blocked']['x']==0:
        raise ValueError('need successful insertion and physically confirmed held-slip evidence')
    calibrator=unlinked(manifest['arguments']['calibrator']);development=calibrator.parent
    if sha256(calibrator)!=manifest['calibrator_sha256']:raise ValueError('calibrator drift')
    dev_manifest=read(development/'manifest.json')
    for name in ('calibration.py','development.py','scene.py'):
        relative='src/yolo_conveyor/held_recheck/'+name
        if dev_manifest['sources'][relative]!=sources()[relative]:raise ValueError('fitting implementation drift')
    artifacts={}
    for root in (development,calibration):
        inventory=root/'artifact-hashes.json'
        for relative,expected in read(inventory).items():
            if sha256(dataset_file(root,relative))!=expected:raise ValueError('artifact changed: '+relative)
        artifacts[str(inventory)]=sha256(inventory)
    weights=unlinked(manifest['arguments']['weights'])
    if sha256(weights)!=WEIGHT_SHA:raise ValueError('weight drift')
    for path in (weights,calibrator,development/'manifest.json',development/'prealign-records.json',
                 calibration/'manifest.json',calibration/'summary.json',calibration/'results.json',
                 Path('docs/data/screw_held_protocol_v1.json').resolve()):
        artifacts[str(path)]=sha256(path)
    used={row['seed'] for row in rows}
    for split_rows in read(development/'prealign-records.json').values():used.update(row['seed'] for row in split_rows)
    if used&set(range(args.seed,args.seed+args.episodes)):raise ValueError('evaluation seed leakage')
    protocol=dict(status='FROZEN_H2_SIMULATION_ONLY',real_formal_gate='closed',created_at=utc_now(),
                  sources=sources(),setup=SETUP,policy=POLICY,faults=FAULTS,artifacts=artifacts,
                  weights_sha256=WEIGHT_SHA,calibrator_sha256=sha256(calibrator),
                  asset_sha256=manifest['asset_sha256'],scene_sha256=manifest['scene_sha256'],
                  packages=environment()['packages'],calibration=str(calibration),
                  evaluation=dict(seed=args.seed,episodes=args.episodes,cases=list(CASES)),
                  rules='No tuning on evaluation; source/config changes require calibration and new reserved evaluation seeds')
    output.parent.mkdir(parents=True,exist_ok=True);write_json(output,protocol)
    return dict(protocol=str(output),sha256=sha256(output),evaluation=protocol['evaluation'])


def validate(path,args):
    if not path:raise ValueError('evaluation requires a protocol')
    protocol=read(unlinked(path))
    if protocol['status']!='FROZEN_H2_SIMULATION_ONLY' or protocol['real_formal_gate']!='closed':raise ValueError('protocol scope')
    if (protocol['sources']!=sources() or protocol['setup']!=SETUP or protocol['policy']!=POLICY or protocol['faults']!=FAULTS):
        raise ValueError('frozen source/configuration changed')
    for item,expected in protocol['artifacts'].items():
        if sha256(unlinked(item))!=expected:raise ValueError('frozen artifact changed')
    if (sha256(Path(args.weights))!=protocol['weights_sha256'] or sha256(Path(args.calibrator))!=protocol['calibrator_sha256']
            or sha256(Path(args.assets)/'asset-manifest.json')!=protocol['asset_sha256']):raise ValueError('runtime input mismatch')
    if environment()['packages']!=protocol['packages']:raise ValueError('frozen environment changed')
    if dict(seed=args.seed,episodes=args.episodes,cases=list(CASES))!=protocol['evaluation']:raise ValueError('evaluation assignment changed')
    return protocol


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration',required=True);p.add_argument('--output',required=True)
    p.add_argument('--seed',type=int,default=91000);p.add_argument('--episodes',type=int,default=36)
    print(json.dumps(freeze(p.parse_args()),indent=2));return 0
