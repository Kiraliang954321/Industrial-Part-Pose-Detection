import cv2
import numpy as np
from yolo_conveyor.datum_pose.geometry import rz


def about(axis,degrees):
    axis=np.asarray(axis,float);axis=axis/np.linalg.norm(axis)
    return cv2.Rodrigues(axis*np.radians(degrees))[0]


def error(r,target):
    r=np.asarray(r);target=np.asarray(target)
    return dict(axis_deg=float(np.degrees(np.arccos(np.clip(r[:,2]@target[:,2],-1,1)))),
                rotation_deg=float(np.degrees(np.arccos(np.clip((np.trace(r.T@target)-1)/2,-1,1)))))


def describe(r):
    r=np.asarray(r);axis=r[:,2];key=r[:,0]
    tilt=float(np.degrees(np.arccos(np.clip(axis[2],-1,1))))
    return dict(inclination_deg=tilt,axis_world=axis.tolist(),key_world=key.tolist(),
                axis_azimuth_deg=float(np.degrees(np.arctan2(axis[1],axis[0]))%360) if 1<tilt<179 else None,
                key_azimuth_deg=float(np.degrees(np.arctan2(key[1],key[0]))%360) if np.linalg.norm(key[:2])>.1 else None)
