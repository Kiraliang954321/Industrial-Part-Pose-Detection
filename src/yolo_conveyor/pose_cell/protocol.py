"""Freeze image-driven A3 development before a new physical evaluation matrix."""
import argparse
import json
from pathlib import Path
from yolo_conveyor.orientation.common import sha256,unlinked,write_json,environment,utc_now,dataset_file
from .provenance import sources
from .scene import SETUP
from .metrology import LIMITS
from .runtime import CASES,TARGETS,POLICY
from .data import audit_dataset


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def assignment(seed):
    return dict(seed=seed,cases=list(CASES),targets=TARGETS,episodes=2*len(CASES)*len(TARGETS))


def freeze(args):
    cal=unlinked(args.calibration);val=unlinked(args.validation);training=unlinked(args.training);output=unlinked(args.output)
    if output.exists():raise FileExistsError('protocol exists')
    m=read(cal/'manifest.json');s=read(cal/'summary.json');v=read(val/'manifest.json');train=read(training/'training-summary.json')
    tm=read(training/'manifest.json');dataset=unlinked(tm['arguments']['dataset'])
    if s['status']!='completed' or s['mode']!='calibration' or train['status']!='completed':raise ValueError('completed development required')
    if m['sources']!=sources() or m['policy']!=POLICY or m['limits']!=LIMITS or m['setup']!=SETUP:raise ValueError('calibration source/configuration drift')
    if v['sources']!=sources():raise ValueError('validation source drift')
    if audit_dataset(dataset)['errors'] or sha256(dataset/'dataset.json')!=tm['dataset_sha256'] or v['dataset_sha256']!=tm['dataset_sha256']:raise ValueError('dataset provenance mismatch')
    weights=unlinked(m['arguments']['weights']);digest=sha256(weights)
    if digest!=m['weights_sha256'] or digest!=v['weights_sha256'] or digest not in {r['sha256'] for r in train['checkpoints'].values()}:
        raise ValueError('selected model provenance mismatch')
    if len(read(cal/'results.json'))!=assignment(args.seed)['episodes'] or not 211000<=args.seed<211876:raise ValueError('evaluation assignment')
    if any(e['seed']>=201000 for e in read(dataset/'episodes.json')):raise ValueError('development seed separation')
    for kind in ('bearing','gear'):
        if s['by_part'][kind]['initial_coverage']['x']==0:raise ValueError('no measured orientation for '+kind)
    artifacts={}
    for path in (weights,dataset/'dataset.json',training/'training-summary.json',training/'manifest.json',val/'summary.json',val/'results.json',val/'manifest.json',
                 cal/'manifest.json',cal/'summary.json',cal/'results.json'):
        artifacts[str(path)]=sha256(path)
    for relative,expected in read(cal/'artifact-hashes.json').items():
        if sha256(dataset_file(cal,relative))!=expected:raise ValueError('calibration artifact drift')
    artifacts[str(cal/'artifact-hashes.json')]=sha256(cal/'artifact-hashes.json')
    p=dict(status='FROZEN_A3_CUSTOM_PROXY_SIMULATION',real_formal_gate='closed',created_at=utc_now(),sources=sources(),setup=SETUP,policy=POLICY,limits=LIMITS,
           weights_sha256=digest,asset_sha256=m['asset_sha256'],packages=environment()['packages'],artifacts=artifacts,evaluation=assignment(args.seed),
           rule='No tuning on evaluation. Image-derived pose drives corrections; robot FK only transforms tool commands. GT only scores; all refusals retained.')
    output.parent.mkdir(parents=True,exist_ok=True);write_json(output,p)
    return dict(path=str(output),sha256=sha256(output),evaluation=p['evaluation'])


def validate(path,args):
    if not path:raise ValueError('evaluation requires frozen protocol')
    p=read(unlinked(path))
    if p['status']!='FROZEN_A3_CUSTOM_PROXY_SIMULATION' or p['real_formal_gate']!='closed':raise ValueError('scope mismatch')
    if p['sources']!=sources() or p['setup']!=SETUP or p['policy']!=POLICY or p['limits']!=LIMITS:raise ValueError('frozen source/configuration drift')
    for item,digest in p['artifacts'].items():
        if sha256(unlinked(item))!=digest:raise ValueError('frozen artifact drift')
    if sha256(Path(args.weights))!=p['weights_sha256'] or sha256(Path(args.assets)/'asset-manifest.json')!=p['asset_sha256']:raise ValueError('runtime input drift')
    if environment()['packages']!=p['packages'] or assignment(args.seed)!=p['evaluation']:raise ValueError('evaluation environment/assignment drift')
    return p


def main():
    p=argparse.ArgumentParser()
    for name in ('calibration','validation','training','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--seed',type=int,default=211000)
    print(json.dumps(freeze(p.parse_args()),indent=2));return 0
