"""Oracle-seeded physical fit controls; explicitly not neural evaluation."""
import argparse
import json
import numpy as np
from yolo_conveyor.orientation.common import new_directory,write_json
from yolo_conveyor.pose_cell.geometry import rz
from .scene import KeyedScene,SETUP
from .control import KeyedController
from .guard import Monitor,ContactStop,GUARD
from .evaluation import PhysicalScore,truth
from .provenance import snapshot


def run(args):
    out=new_directory(args.output,[args.assets]);rows=[]
    write_json(out/'manifest.json',dict(scope='A5_ORACLE_MECHANICAL_REFERENCE_NOT_NEURAL',arguments=vars(args),setup=SETUP,guard=GUARD,sources=snapshot(out)))
    for k,kind in enumerate(('bearing','gear')):
        for i,(case,phase,feature) in enumerate([('matched',0.,True),('wrong30',30.,True),('wrong90',90.,True),('wrong180',180.,True),('no_datum',0.,False)]):
            seed=args.seed+100*k+i;folder=out/f'{kind}_{case}';folder.mkdir();s=KeyedScene(args.assets,kind,feature,phase);c=KeyedController(s)
            score=PhysicalScore(s);monitor=Monitor(s,score);attempted=False;blocked=False;failure=None;stage='initialize';target=rz(0)
            try:
                rng=np.random.default_rng(seed);s.initialize_pose(rng,float(rng.uniform(12,25)))
                stage='pick';c.execute(c.pick(),monitor);monitor.fingers=s.data.qpos[s.finger_qpos].copy()
                stage='present';c.execute(c.present(0),monitor);g=truth(s,target)
                stage='oracle_correct';c.execute(c.correction(dict(rotation=g['rotation'],translation_m=g['center']),target),monitor)
                g=truth(s,target);geometry=c.assembly_geometry(dict(rotation=g['rotation'],translation_m=g['center']),target)
                stage='approach';c.execute(c.approach(geometry),monitor);tare=monitor.tare();write_json(folder/'tare.json',tare)
                stage='plan_insert';stages=c.preinsert(geometry);stage='insert';attempted=True;score.insertion_active=True
                try:c.execute(stages,monitor)
                except ContactStop:
                    blocked=True;monitor.disarm();c.execute(c.stop_hold(),monitor);stage='retreat';c.execute(c.retreat(),monitor)
            except ValueError as exc:failure=str(exc)
            finally:
                monitor.disarm();monitor.fingers=None;c.execute(c.stop_hold(.3),lambda _:score.observe())
                result=dict(kind=kind,case=case,seed=seed,phase=phase,feature=feature,attempted=attempted,blocked=blocked,stage=stage,failure=failure,
                    physical=score.result(attempted,'contact_stop' if blocked else failure),wrench=monitor.report(),final=truth(s,target))
                rows.append(result);write_json(folder/'result.json',result);write_json(folder/'trace.json',monitor.trace);s.close()
            print(json.dumps(result),flush=True)
    write_json(out/'results.json',rows);return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=261000)
    run(p.parse_args());return 0
