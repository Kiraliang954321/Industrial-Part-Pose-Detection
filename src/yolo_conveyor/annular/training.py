"""Bounded YOLO26n development fine-tuning for the two annular part families."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import time
from yolo_conveyor.orientation.common import new_directory,unlinked,write_json,sha256,environment
from .assets import NAMES
from .data import audit_dataset
from .provenance import snapshot

INITIAL_SHA='9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef'


def verify_model(model):
    if model.task!='detect' or {int(k):v for k,v in model.names.items()}!={i:n for i,n in enumerate(NAMES)}:
        raise ValueError('checkpoint must declare the six A1 semantic classes')


def train(args):
    if not 1<=args.epochs<=20 or not 1<=args.max_minutes<=30:raise ValueError('training budget')
    root=unlinked(args.dataset);initial=unlinked(args.initial_weights)
    if sha256(initial)!=INITIAL_SHA:raise ValueError('wrong upstream initialization')
    audit=audit_dataset(root)
    if audit['errors']:raise ValueError(str(audit['errors'][:5]))
    out=new_directory(args.output,[root,initial]);os.environ['YOLO_CONFIG_DIR']=str(out/'yolo-config')
    start=time.perf_counter();batch_counts={};budget=False;last_progress=0.
    try:
        import torch
        from ultralytics import YOLO,settings
        torch.set_num_threads(8)
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 required')
        settings.update({k:False for k in ('sync','wandb','mlflow','comet','clearml','neptune','hub','tensorboard') if k in settings})
        inputs=out/'inputs';inputs.mkdir();shutil.copyfile(initial,inputs/'initial.pt')
        if sha256(inputs/'initial.pt')!=INITIAL_SHA:raise ValueError('initial weight snapshot drift')
        for name in ('dataset.json','records.json','episodes.json','data.yaml','parts.json','setup.json'):
            shutil.copyfile(root/name,inputs/name)
        manifest=dict(scope='A1_SIMULATION_DEVELOPMENT',real_formal_gate='closed',arguments=vars(args),sources=snapshot(out),
                      dataset_sha256=sha256(root/'dataset.json'),initial_sha256=INITIAL_SHA,environment=environment(),
                      selection='Ultralytics best development Val detection fitness; independent physical evaluation follows')
        write_json(out/'manifest.json',manifest);write_json(out/'preflight-audit.json',audit)
        def progress(trainer,status='training'):
            write_json(out/'progress.json',dict(status=status,epoch=min(int(trainer.epoch)+1,args.epochs),requested_epochs=args.epochs,
                       seconds=time.perf_counter()-start,time_budget_reached=budget),replace=True)
        def batch_end(trainer):
            nonlocal budget,last_progress
            epoch=int(trainer.epoch)+1;batch_counts[epoch]=batch_counts.get(epoch,0)+1;now=time.perf_counter()
            if now-start>=args.max_minutes*60:budget=True;trainer.stop=True
            if (out/'STOP_REQUESTED').exists():trainer.stop=True
            if now-last_progress>15:progress(trainer);last_progress=now
        model=YOLO(str(inputs/'initial.pt'))
        model.add_callback('on_train_batch_end',batch_end)
        model.add_callback('on_fit_epoch_end',lambda trainer:progress(trainer,'validated'))
        torch.cuda.reset_peak_memory_stats()
        model.train(data=str(root/'data.yaml'),epochs=args.epochs,batch=16,imgsz=640,device='0',workers=0,seed=42,
                    deterministic=True,project=str(out),name='fit',exist_ok=False,optimizer='AdamW',lr0=.0007,lrf=.1,
                    warmup_bias_lr=0.,warmup_epochs=1.,patience=0,amp='bf16',cache=False,plots=False,save=True,
                    mosaic=0.,mixup=0.,copy_paste=0.,degrees=0.,translate=.025,scale=.08,fliplr=0.,flipud=0.,
                    hsv_h=.01,hsv_s=.1,hsv_v=.15,close_mosaic=0,val=True,verbose=False)
        fit=unlinked(model.trainer.save_dir)
        if not fit.is_relative_to(out):raise ValueError('trainer outputs escaped run')
        checkpoints={}
        for name in ('best','last'):
            path=fit/'weights'/f'{name}.pt';verify_model(YOLO(str(path)));checkpoints[name]=dict(path=str(path),sha256=sha256(path))
        with (fit/'results.csv').open(encoding='utf-8',newline='') as stream:epochs=list(csv.DictReader(stream))
        if audit_dataset(root)['errors'] or sha256(root/'dataset.json')!=manifest['dataset_sha256']:raise ValueError('dataset drift')
        summary=dict(status='completed',scope='A1_SIMULATION_DEVELOPMENT',real_formal_gate='closed',checkpoints=checkpoints,
                     logged_epochs=len(epochs),batches_by_epoch=batch_counts,expected_batches_per_epoch=len(model.trainer.train_loader),
                     seconds=time.perf_counter()-start,time_budget_reached=budget,stop_requested=(out/'STOP_REQUESTED').exists(),
                     peak_cuda_bytes=torch.cuda.max_memory_allocated(),last_csv_metrics=epochs[-1])
        write_json(out/'training-summary.json',summary);write_json(out/'progress.json',dict(status='completed',epochs=len(epochs)),replace=True)
        return summary
    except BaseException as exc:
        write_json(out/'progress.json',dict(status='failed',error=str(exc)),replace=True);raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('dataset','initial-weights','output'):p.add_argument('--'+flag,required=True)
    p.add_argument('--epochs',type=int,default=12);p.add_argument('--max-minutes',type=int,default=20)
    print(json.dumps(train(p.parse_args()),indent=2));return 0
