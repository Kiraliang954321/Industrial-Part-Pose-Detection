"""Bounded fine-tuning of the recorded A3 center model on grouped A4 views."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import time
from yolo_conveyor.orientation.common import new_directory,unlinked,write_json,sha256,environment
from yolo_conveyor.pose_cell.training import verify_model
from yolo_conveyor.pose_cell.faces import NAMES,ROI,CENTER_SCHEMA,keypoint_names
from .data import audit_dataset,read
from .provenance import snapshot

BASELINE_SHA='d12fefc4565a20030cdbf752a4f31411be13eb77191f4dddf75d1385de9b6c3c'


def train(args):
    if not 1<=args.epochs<=20 or not 1<=args.max_minutes<=30:raise ValueError('training budget')
    root=unlinked(args.dataset);initial=unlinked(args.initial_weights)
    if sha256(initial)!=BASELINE_SHA:raise ValueError('recorded A3 initialization required')
    audit=audit_dataset(root)
    if audit['errors']:raise ValueError(str(audit['errors'][:5]))
    out=new_directory(args.output,[root,initial]);os.environ['YOLO_CONFIG_DIR']=str(out/'yolo-config')
    import torch
    from ultralytics import YOLO,settings
    torch.set_num_threads(8)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 required')
    settings.update({k:False for k in ('sync','wandb','mlflow','comet','clearml','neptune','hub','tensorboard') if k in settings})
    inputs=out/'inputs';inputs.mkdir();shutil.copyfile(initial,inputs/'initial.pt')
    if sha256(inputs/'initial.pt')!=BASELINE_SHA:raise ValueError('initial snapshot drift')
    for name in ('dataset.json','records.json','episodes.json','data.yaml','manifest.json','grabs.json'):shutil.copyfile(root/name,inputs/name)
    manifest=dict(scope='A4_SIMULATION_DEVELOPMENT',real_formal_gate='closed',arguments=vars(args),sources=snapshot(out),
                  dataset_sha256=sha256(root/'dataset.json'),initial_sha256=BASELINE_SHA,environment=environment(),
                  point_schema=CENTER_SCHEMA,initialization='A3 center head retained without remapping',
                  selection='Development Val pose fitness; direct pose validation and fresh closed-loop evaluation are separate')
    write_json(out/'manifest.json',manifest);write_json(out/'preflight-audit.json',audit)
    start=time.perf_counter();last_progress=0.;batches={};budget=False
    def progress(trainer,status='training'):
        write_json(out/'progress.json',dict(status=status,epoch=int(trainer.epoch)+1,requested_epochs=args.epochs,seconds=time.perf_counter()-start),replace=True)
    def batch_end(trainer):
        nonlocal last_progress,budget
        epoch=int(trainer.epoch)+1;batches[epoch]=batches.get(epoch,0)+1;now=time.perf_counter()
        if now-start>=args.max_minutes*60:trainer.stop=True;budget=True
        if (out/'STOP_REQUESTED').exists():trainer.stop=True
        if now-last_progress>15:progress(trainer);last_progress=now
    def mark(trainer):
        for net in [trainer.model]+([trainer.ema.ema] if trainer.ema is not None else []):
            net.a3_roi=list(ROI);net.a3_point_schema=CENTER_SCHEMA;net.a4_task='active_inspection'
            net.kpt_names={i:keypoint_names(CENTER_SCHEMA) for i in range(len(NAMES))}
    try:
        model=YOLO(str(inputs/'initial.pt'));verify_model(model)
        model.add_callback('on_train_start',mark);model.add_callback('on_train_batch_end',batch_end)
        model.add_callback('on_fit_epoch_end',lambda trainer:progress(trainer,'validated'))
        torch.cuda.reset_peak_memory_stats()
        model.train(data=str(root/'data.yaml'),epochs=args.epochs,batch=32,imgsz=640,device='0',workers=0,seed=42,
                    deterministic=True,project=str(out),name='fit',exist_ok=False,optimizer='AdamW',lr0=.00015,lrf=.1,
                    warmup_bias_lr=0.,warmup_epochs=1.,patience=0,amp='bf16',cache=False,plots=False,save=True,
                    mosaic=0.,mixup=0.,copy_paste=0.,degrees=0.,translate=.025,scale=.08,fliplr=0.,flipud=0.,
                    hsv_h=.01,hsv_s=.1,hsv_v=.15,close_mosaic=0,val=True,verbose=False)
        fit=unlinked(model.trainer.save_dir)
        if not fit.is_relative_to(out):raise ValueError('training output escaped run')
        checkpoints={}
        for name in ('best','last'):
            p=fit/'weights'/f'{name}.pt';verify_model(YOLO(str(p)));checkpoints[name]=dict(path=str(p),sha256=sha256(p))
        with (fit/'results.csv').open(encoding='utf-8',newline='') as f:epochs=list(csv.DictReader(f))
        if audit_dataset(root)['errors'] or sha256(root/'dataset.json')!=manifest['dataset_sha256']:raise ValueError('training data drift')
        result=dict(status='completed',scope=manifest['scope'],real_formal_gate='closed',checkpoints=checkpoints,logged_epochs=len(epochs),
                    batches_by_epoch=batches,expected_batches_per_epoch=len(model.trainer.train_loader),seconds=time.perf_counter()-start,
                    time_budget_reached=budget,peak_cuda_bytes=torch.cuda.max_memory_allocated(),last_csv_metrics=epochs[-1])
        write_json(out/'training-summary.json',result);write_json(out/'progress.json',dict(status='completed',epochs=len(epochs)),replace=True)
        return result
    except BaseException as exc:
        write_json(out/'progress.json',dict(status='failed',error=str(exc)),replace=True);raise


def main():
    p=argparse.ArgumentParser()
    for n in ('dataset','initial-weights','output'):p.add_argument('--'+n,required=True)
    p.add_argument('--epochs',type=int,default=10);p.add_argument('--max-minutes',type=int,default=25)
    print(json.dumps(train(p.parse_args()),indent=2));return 0
