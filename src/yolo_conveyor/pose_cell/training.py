"""Bounded, versioned Pose training for continuous mechanical-datum orientation."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import time
from yolo_conveyor.orientation.common import new_directory,unlinked,write_json,sha256,environment
from .faces import NAMES,KEYPOINTS,ROI,CORNER_SCHEMA,CENTER_SCHEMA,keypoint_names
from .data import audit_dataset
from .provenance import snapshot

INITIAL_SHA='7cbe908673310b64e16668d608d48e721b23e6fb30d7d254c75b42d019904ed3'
CORNERS_SHA='d3ddd803245388855f8c3daa3987567d979d25ae96210eb0882859f63c893025'


def verify_model(model):
    shape=getattr(model.model,'kpt_shape',None)
    if shape is None:shape=getattr(model.model.model[-1],'kpt_shape',None)
    names=getattr(model.model,'kpt_names',None)
    schema=getattr(model.model,'a3_point_schema',CORNER_SCHEMA)
    if (model.task!='pose' or {int(k):v for k,v in model.names.items()}!={i:n for i,n in enumerate(NAMES)}
            or list(shape or [])!=[5,3] or not names or {int(k):v for k,v in names.items()}!={i:keypoint_names(schema) for i in range(4)}
            or list(getattr(model.model,'a3_roi',[]))!=list(ROI)):
        raise ValueError('checkpoint must declare the custom parts and versioned datum schema')


def train(args):
    if not 1<=args.epochs<=30 or not 1<=args.max_minutes<=30:raise ValueError('training budget')
    root=unlinked(args.dataset);initial=unlinked(args.initial_weights)
    initial_sha=sha256(initial)
    if initial_sha not in (INITIAL_SHA,CORNERS_SHA):raise ValueError('unrecorded initialization')
    audit=audit_dataset(root)
    if audit['errors']:raise ValueError(str(audit['errors'][:5]))
    if json.loads((root/'dataset.json').read_text(encoding='utf-8')).get('input_space')!='fixed_roi':
        raise ValueError('current A2 checkpoints require the declared fixed ROI dataset')
    schema=json.loads((root/'dataset.json').read_text(encoding='utf-8')).get('point_schema',CORNER_SCHEMA)
    if (schema==CENTER_SCHEMA)!=(initial_sha==CORNERS_SHA):raise ValueError('schema initialization mismatch')
    out=new_directory(args.output,[root,initial]);os.environ['YOLO_CONFIG_DIR']=str(out/'yolo-config')
    start=time.perf_counter();batch_counts={};budget=False;last_progress=0.
    try:
        import torch
        from ultralytics import YOLO,settings
        torch.set_num_threads(8)
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 required')
        settings.update({k:False for k in ('sync','wandb','mlflow','comet','clearml','neptune','hub','tensorboard') if k in settings})
        inputs=out/'inputs';inputs.mkdir();shutil.copyfile(initial,inputs/'initial.pt')
        if sha256(inputs/'initial.pt')!=initial_sha:raise ValueError('initial weight snapshot drift')
        for name in ('dataset.json','records.json','episodes.json','data.yaml','manifest.json'):
            shutil.copyfile(root/name,inputs/name)
        manifest=dict(scope='A3_SIMULATION_DEVELOPMENT',real_formal_gate='closed',arguments=vars(args),sources=snapshot(out),
                      dataset_sha256=sha256(root/'dataset.json'),initial_sha256=initial_sha,environment=environment(),
                      initialization='recorded A3 corners with averaged center output' if schema==CENTER_SCHEMA else 'recorded A2 datum landmark checkpoint',point_schema=schema,
                      selection='Ultralytics best development Val pose fitness; direct geometric Val scoring is separate; independent physical evaluation follows')
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
        def mark_preprocessing(trainer):
            for net in [trainer.model]+([trainer.ema.ema] if trainer.ema is not None else []):
                net.a3_roi=list(ROI);net.a3_point_schema=schema
                net.kpt_names={i:keypoint_names(schema) for i in range(4)}
                if schema==CENTER_SCHEMA:
                    head=net.model[-1]
                    with torch.no_grad():
                        for name,group in (('cv4_kpts',3),('one2one_cv4_kpts',3),('cv4_sigma',2),('one2one_cv4_sigma',2)):
                            for layer in getattr(head,name):
                                layer.weight[group:2*group].copy_(layer.weight[group:].reshape(4,group,*layer.weight.shape[1:]).mean(0))
                                layer.bias[group:2*group].copy_(layer.bias[group:].reshape(4,group).mean(0))
        model.add_callback('on_train_start',mark_preprocessing)
        model.add_callback('on_train_batch_end',batch_end)
        model.add_callback('on_fit_epoch_end',lambda trainer:progress(trainer,'validated'))
        torch.cuda.reset_peak_memory_stats()
        model.train(data=str(root/'data.yaml'),epochs=args.epochs,batch=args.batch,imgsz=640,device='0',workers=0,seed=42,
                    deterministic=True,project=str(out),name='fit',exist_ok=False,optimizer='AdamW',
                    lr0=.0003,lrf=.1,
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
        summary=dict(status='completed',scope='A3_SIMULATION_DEVELOPMENT',real_formal_gate='closed',checkpoints=checkpoints,
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
    p.add_argument('--epochs',type=int,default=16);p.add_argument('--max-minutes',type=int,default=16)
    p.add_argument('--batch',type=int,choices=(16,32),default=16)
    print(json.dumps(train(p.parse_args()),indent=2));return 0
