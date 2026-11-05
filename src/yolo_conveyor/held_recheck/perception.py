"""Acquire RGB evidence; calibration and inference never access object truth."""
import hashlib
import time
import cv2
import numpy as np
from yolo_conveyor.orientation.common import write_json,sha256
from yolo_conveyor.orientation.evaluate import draw_predictions


def capture(scene,predictor,calibrator,names,gain,blackout=None):
    images={};times=[]
    for name in names:
        images[name]=np.clip(scene.capture(name).astype(float)*gain,0,255).astype(np.uint8)
        if name==blackout:images[name][:]=0
        times.append(float(scene.data.time))
    predictor.synchronize();start=time.perf_counter()
    raw={name:predictor.predict(image) for name,image in images.items()};predictor.synchronize()
    corrected={};errors=[]
    for name in names:
        try:corrected[name]=calibrator.apply(name,raw[name])
        except ValueError as exc:corrected[name]=[];errors.append(str(exc))
    digest=hashlib.sha256()
    for name in names:digest.update(images[name].tobytes())
    digest.update(str(times).encode())
    return dict(images=images,raw=raw,corrected=corrected,cameras={name:scene.camera_info(name) for name in names},
                tool=scene.tool_transform().tolist(),fingers=scene.data.qpos[scene.finger_qpos].tolist(),
                capture_times=times,evidence_id=digest.hexdigest(),inference_ms=(time.perf_counter()-start)*1000,
                calibration_errors=errors)


def save(directory,observation,decision):
    directory.mkdir()
    record={k:v for k,v in observation.items() if k!='images'}
    record['decision']=decision;record['input_hashes']={}
    for name,image in observation['images'].items():
        path=directory/f'{name}.png'
        if not cv2.imwrite(str(path),image):raise RuntimeError('image write failed')
        record['input_hashes'][name]=sha256(path)
        preview=draw_predictions(image,observation['corrected'][name],decision['decision']+' | '+decision['reason'])
        cv2.imwrite(str(directory/f'{name}_prediction.jpg'),preview)
    write_json(directory/'observation.json',record)
