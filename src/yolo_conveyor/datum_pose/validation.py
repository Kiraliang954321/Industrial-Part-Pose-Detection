"""Development Val angle/coverage evaluation, explicitly separate from training mAP."""
import argparse
import json
from pathlib import Path
import cv2
from yolo_conveyor.orientation.common import new_directory,write_json,sha256
from yolo_conveyor.orientation.evaluation import distribution,rate
from .perception import Predictor,measure
from .geometry import errors
from .data import audit_dataset
from .provenance import snapshot


def validate(args):
    root=Path(args.dataset).resolve()
    if audit_dataset(root)['errors']:raise ValueError('dataset audit failed')
    out=new_directory(args.output,[root,args.weights]);predictor=Predictor(args.weights);rows=json.loads((root/'records.json').read_text())
    episodes=[e for e in json.loads((root/'episodes.json').read_text()) if e['split']=='val'];results=[]
    for e in episodes:
        views=[r for r in rows if r['episode']==e['id']];images={r['view']:cv2.imread(str(root/r['image'])) for r in views}
        cropped=json.loads((root/'dataset.json').read_text()).get('input_space')=='fixed_roi'
        predictions=predictor.predict(images,preprocessed=cropped);item=dict(episode=e['id'],kind=e['kind'],observable=e['feature'] and not e['empty'],predictions=predictions)
        try:
            result=measure(images,predictions,{r['view']:r['camera'] for r in views},e['kind'],pixel_scale=2. if cropped else 1.)
            item.update(measurement=result,error=errors(result['rotation'],e['rotation']))
        except ValueError as exc:item.update(measurement=None,rejection=str(exc))
        results.append(item)
    report={}
    for kind in ('bearing','gear'):
        eligible=[r for r in results if r['kind']==kind and r['observable']];usable=[r for r in eligible if r['measurement']]
        negative=[r for r in results if r['kind']==kind and not r['observable']]
        report[kind]=dict(coverage=rate(len(usable),len(eligible)),false_observable=rate(sum(r['measurement'] is not None for r in negative),len(negative)),
                          axis_deg=distribution([r['error']['axis_deg'] for r in usable]),spin_deg=distribution([r['error']['spin_deg'] for r in usable]),
                          rotation_deg=distribution([r['error']['rotation_deg'] for r in usable]))
    write_json(out/'results.json',results);write_json(out/'summary.json',report)
    write_json(out/'manifest.json',dict(scope='A2_VAL_ONLY',weights_sha256=sha256(Path(args.weights)),dataset_sha256=sha256(root/'dataset.json'),sources=snapshot(out)))
    return report


def main():
    p=argparse.ArgumentParser()
    for name in ('dataset','weights','output'):p.add_argument('--'+name,required=True)
    print(json.dumps(validate(p.parse_args()),indent=2));return 0
