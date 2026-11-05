"""Freeze A2 continuous-orientation inputs before new target-pose episodes."""
import argparse
import json
from pathlib import Path
from yolo_conveyor.orientation.common import sha256,unlinked,write_json,environment,utc_now,dataset_file
from .provenance import sources
from .scene import SETUP
from .perception import POLICY
from .runtime import CASES
from .targets import load


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def freeze(args):
    cal=unlinked(args.calibration);val=unlinked(args.validation);training=unlinked(args.training);output=unlinked(args.output)
    if output.exists():raise FileExistsError('protocol exists')
    m=read(cal/'manifest.json');s=read(cal/'summary.json');v=read(val/'manifest.json');train=read(training/'training-summary.json')
    if s['status']!='completed' or s['mode']!='calibration' or train['status']!='completed':raise ValueError('completed development required')
    if m['sources']!=sources() or m['policy']!=POLICY or m['setup']!=SETUP:raise ValueError('calibration source drift')
    weights=unlinked(m['arguments']['weights']);digest=sha256(weights)
    if digest!=m['weights_sha256'] or digest!=v['weights_sha256'] or digest not in {r['sha256'] for r in train['checkpoints'].values()}:
        raise ValueError('selected model provenance mismatch')
    config=load(m['arguments']['targets']);count=len(config['targets'])*len(CASES)
    if len(read(cal/'results.json'))!=2*count or not 171000<=args.seed<172000-count-100:raise ValueError('evaluation assignment')
    for kind in ('bearing','gear'):
        if s['by_part'][kind]['coverage']['x']==0:raise ValueError('no measured orientation for '+kind)
    artifacts={}
    for path in (weights,training/'training-summary.json',training/'manifest.json',val/'summary.json',val/'results.json',val/'manifest.json',
                 cal/'manifest.json',cal/'summary.json',cal/'results.json',Path(m['arguments']['targets']).resolve()):
        artifacts[str(path)]=sha256(path)
    for relative,expected in read(cal/'artifact-hashes.json').items():
        if sha256(dataset_file(cal,relative))!=expected:raise ValueError('calibration artifact drift')
    artifacts[str(cal/'artifact-hashes.json')]=sha256(cal/'artifact-hashes.json')
    p=dict(status='FROZEN_A2_CUSTOM_PROXY_SIMULATION',real_formal_gate='closed',created_at=utc_now(),sources=sources(),setup=SETUP,policy=POLICY,
           weights_sha256=digest,targets=config,targets_sha256=m['targets_sha256'],asset_sha256=m['asset_sha256'],
           packages=environment()['packages'],artifacts=artifacts,evaluation=dict(seed=args.seed,cases=list(CASES),episodes=2*count),
           rule='No tuning on these outcomes; numerical rotations from RGB only; physical state is independent scoring')
    output.parent.mkdir(parents=True,exist_ok=True);write_json(output,p);return dict(path=str(output),sha256=sha256(output),evaluation=p['evaluation'])


def validate(path,args):
    if not path:raise ValueError('evaluation requires protocol')
    p=read(unlinked(path));config=load(args.targets)
    if p['status']!='FROZEN_A2_CUSTOM_PROXY_SIMULATION' or p['real_formal_gate']!='closed':raise ValueError('scope mismatch')
    if p['sources']!=sources() or p['setup']!=SETUP or p['policy']!=POLICY:raise ValueError('frozen source/configuration drift')
    for item,digest in p['artifacts'].items():
        if sha256(unlinked(item))!=digest:raise ValueError('frozen artifact drift')
    if (sha256(Path(args.weights))!=p['weights_sha256'] or sha256(Path(args.targets))!=p['targets_sha256']
            or sha256(Path(args.assets)/'asset-manifest.json')!=p['asset_sha256']):raise ValueError('runtime input drift')
    if config!=p['targets'] or environment()['packages']!=p['packages']:raise ValueError('target/environment drift')
    if dict(seed=args.seed,cases=list(CASES),episodes=2*len(config['targets'])*len(CASES))!=p['evaluation']:raise ValueError('evaluation seeds changed')
    return p


def main():
    p=argparse.ArgumentParser()
    for name in ('calibration','validation','training','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--seed',type=int,default=171000);print(json.dumps(freeze(p.parse_args()),indent=2));return 0
