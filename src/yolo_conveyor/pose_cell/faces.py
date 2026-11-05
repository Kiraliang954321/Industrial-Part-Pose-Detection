import numpy as np
from yolo_conveyor.datum_pose.assets import PARTS,DATUMS,KEYPOINTS

NAMES=['bearing_negative_face','bearing_positive_face','gear_negative_face','gear_positive_face']
ROI=(160,160,480,480)
CORNER_SCHEMA='corners_v1'
CENTER_SCHEMA='centers_v2'
CENTER_KEYPOINTS=['bore_center','key_center','unused_2','unused_3','unused_4']


def keypoint_names(schema):
    if schema==CORNER_SCHEMA:return KEYPOINTS
    if schema==CENTER_SCHEMA:return CENTER_KEYPOINTS
    raise ValueError('unknown point schema')


def class_id(kind,face):return ('bearing','gear').index(kind)*2+int(face>0)


def landmarks(kind,face):
    p=PARTS[kind];d=DATUMS[kind];w=d['width']/2;z=face*p['height']/2
    if kind=='bearing':near=d['root'];far=np.sqrt((.0125 if face>0 else .011)**2-w*w)
    else:near=np.sqrt((p['bore']/2)**2-w*w);far=d['root']
    return np.array([[0,0,z],[near,w,z],[far,w,z],[far,-w,z],[near,-w,z]])


def circle_points(kind,face):
    p=PARTS[kind];a=p['bore']/2/np.sqrt(2);z=face*p['height']/2
    return np.array([[-a,a,z],[a,a,z],[a,-a,z],[-a,-a,z]])
