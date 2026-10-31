"""Bounded held-view fine-tuning from the verified M3 development checkpoint."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import time
from yolo_conveyor.orientation.common import new_directory,unlinked,sha256,write_json,environment,utc_now
from .data import audit_dataset,load_dataset,TASK
from .provenance import snapshot_sources

INITIAL_SHA256="0a29b8c5660d62cb19b84bf2cc0805e1c3b3ed2fb2e7478dc80813ab2f1e155a"


def train(args):
    if not (1<=args.epochs<=30 and 1<=args.batch<=16 and 1<=args.max_minutes<=60):
        raise ValueError("held development budget exceeded")
    root=unlinked(args.dataset);initial=unlinked(args.initial_weights)
    if not initial.is_file() or sha256(initial)!=INITIAL_SHA256:raise ValueError("wrong M3 initial checkpoint")
    audit=audit_dataset(root)
    if audit["errors"]:raise ValueError(str(audit["errors"][:5]))
    meta,_=load_dataset(root);out=new_directory(args.output,[root,initial])
    os.environ['YOLO_CONFIG_DIR']=str(out/'yolo-config')
    start=time.perf_counter();budget=False;last_progress=0.;batch_counts={};expected_batches=0
    try:
        from ultralytics import YOLO,settings
        import torch
        from yolo_conveyor.orientation.pose_adapter import verify_model
        torch.set_num_threads(8)
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError("CUDA BF16 required")
        settings.update({k:False for k in ('sync','wandb','mlflow','comet','clearml','neptune','hub','tensorboard') if k in settings})
        inputs=out/'inputs';inputs.mkdir()
        for name in ('dataset.json','records.jsonl','episodes.json','scene.xml','setup.json','data.yaml'):
            shutil.copyfile(root/name,inputs/name)
        shutil.copyfile(initial,inputs/'initial.pt')
        if sha256(inputs/'initial.pt')!=INITIAL_SHA256:raise ValueError("weight snapshot drift")
        manifest=dict(task=TASK,scope='held_simulation_development',real_formal_gate='closed',arguments=vars(args),
                      dataset_sha256=sha256(root/'dataset.json'),sources=snapshot_sources(out),
                      environment=environment(),gpu=torch.cuda.get_device_name(0),created_at=utc_now(),
                      initial_sha256=INITIAL_SHA256,selection='best development Val pose fitness; downstream calibration separate',
                      dataset_counts=meta['counts'],episode_counts=meta['episode_counts'])
        write_json(out/'run-manifest.json',manifest);write_json(out/'preflight-audit.json',audit)
        def progress(trainer,phase='training'):
            write_json(out/'progress.json',dict(status=phase,epoch=min(int(trainer.epoch)+1,args.epochs),
                       requested_epochs=args.epochs,seconds=time.perf_counter()-start,time_budget_reached=budget),replace=True)
        def batch_end(trainer):
            nonlocal budget,last_progress,expected_batches
            epoch=int(trainer.epoch)+1
            batch_counts[epoch]=batch_counts.get(epoch,0)+1
            expected_batches=len(trainer.train_loader)
            now=time.perf_counter()
            if now-start>=args.max_minutes*60:budget=True;trainer.stop=True
            if (out/'STOP_REQUESTED').exists():trainer.stop=True
            if now-last_progress>15:progress(trainer);last_progress=now
        model=YOLO(str(inputs/'initial.pt'));verify_model(model)
        model.add_callback('on_train_batch_end',batch_end)
        model.add_callback('on_fit_epoch_end',lambda trainer:progress(trainer,'epoch_validated'))
        torch.cuda.reset_peak_memory_stats()
        model.train(data=str(root/'data.yaml'),epochs=args.epochs,batch=args.batch,imgsz=640,device='0',workers=0,
                    seed=args.seed,deterministic=True,project=str(out),name='fit',exist_ok=False,
                    optimizer='AdamW',lr0=.0005,lrf=.05,patience=0,amp='bf16',cache=False,plots=False,save=True,
                    save_period=-1,mosaic=0.,mixup=0.,copy_paste=0.,degrees=0.,translate=.025,scale=.05,
                    fliplr=0.,flipud=0.,hsv_h=.01,hsv_s=.1,hsv_v=.15,close_mosaic=0,val=True,verbose=False)
        fit=unlinked(model.trainer.save_dir)
        if not fit.is_relative_to(out):raise ValueError('trainer output escaped run')
        checkpoints={}
        for name in ('best','last'):
            path=fit/'weights'/f'{name}.pt';verify_model(YOLO(str(path)))
            checkpoints[name]=dict(path=str(path),sha256=sha256(path))
        with (fit/'results.csv').open(encoding='utf-8',newline='') as stream:epochs=list(csv.DictReader(stream))
        if audit_dataset(root)['errors'] or sha256(root/'dataset.json')!=manifest['dataset_sha256']:
            raise ValueError('dataset changed during training')
        summary=dict(task=TASK,status='completed',scope='held_simulation_development',real_formal_gate='closed',
                     actual_logged_epochs=len(epochs),completed_full_epochs=sum(n==expected_batches for n in batch_counts.values()),
                     batches_by_epoch=batch_counts,expected_batches_per_epoch=expected_batches,
                     seconds=time.perf_counter()-start,time_budget_reached=budget,
                     stop_requested=(out/'STOP_REQUESTED').exists(),checkpoints=checkpoints,
                     peak_cuda_bytes=torch.cuda.max_memory_allocated(),last_csv_metrics=epochs[-1] if epochs else {},
                     limitation='Val optimization metrics do not establish pre-insertion or real-world success')
        write_json(out/'training-summary.json',summary)
        write_json(out/'progress.json',dict(status='completed',epochs=len(epochs)),replace=True)
        return summary
    except BaseException as exc:
        write_json(out/'progress.json',dict(status='failed',error=str(exc)),replace=True);raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('dataset','initial-weights','output'):p.add_argument('--'+flag,required=True)
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--batch',type=int,default=16)
    p.add_argument('--max-minutes',type=int,default=45);p.add_argument('--seed',type=int,default=42)
    print(json.dumps(train(p.parse_args()),indent=2));return 0
