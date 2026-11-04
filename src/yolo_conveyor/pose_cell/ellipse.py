"""Robust RGB bore-ellipse fit; no rendered segmentation or pose truth is accepted."""
import cv2
import numpy as np


def fit_bore(image,center,expected,key_direction=None,pixel_scale=1.):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);center=np.asarray(center,float)
    radius=int(np.clip(expected*.9,35*pixel_scale,95*pixel_scale));x,y=np.round(center).astype(int)
    x0,y0=max(0,x-radius),max(0,y-radius);x1,y1=min(640,x+radius),min(640,y+radius)
    patch=cv2.createCLAHE(clipLimit=2.,tileGridSize=(4,4)).apply(gray[y0:y1,x0:x1])
    gx=cv2.Sobel(patch,cv2.CV_32F,1,0);gy=cv2.Sobel(patch,cv2.CV_32F,0,1)
    high=float(np.clip(np.percentile(np.hypot(gx,gy),92),15,65));edges=cv2.Canny(patch,.4*high,high)
    py,px=np.nonzero(edges);points=np.column_stack([px+x0,py+y0]).astype(float);delta=points-center
    distances=np.linalg.norm(delta,axis=1);valid=(distances>.08*expected)&(distances<.75*expected)
    if key_direction is not None:
        direction=np.asarray(key_direction,float);norm=np.linalg.norm(direction)
        if norm>1:valid&=(delta@direction/norm/np.maximum(distances,1))<.8
    points=points[valid];delta=points-center
    if len(points)<25:raise ValueError('weak_bore_edges')
    design=np.column_stack([delta[:,0]**2,2*delta[:,0]*delta[:,1],delta[:,1]**2])
    rng=np.random.default_rng(19);best=None
    for _ in range(450):
        chosen=rng.choice(len(points),3,replace=False)
        try:q=np.linalg.solve(design[chosen],np.ones(3))
        except np.linalg.LinAlgError:continue
        eig=np.linalg.eigvalsh([[q[0],q[1]],[q[1],q[2]]])
        if min(eig)<=0:continue
        axes=1/np.sqrt(eig)
        if not (.32*expected<=max(axes)<=.7*expected and min(axes)>.10*expected):continue
        error=np.abs(design@q-1)*min(axes)/2;mask=error<1.25*pixel_scale
        score=int(mask.sum())
        if best is None or score>best[0]:best=(score,mask)
    if best is None or best[0]<24:raise ValueError('no_bore_ellipse')
    inliers=points[best[1]];ellipse=cv2.fitEllipse(inliers.astype(np.float32));c=np.array(ellipse[0]);sizes=np.array(ellipse[1]);a=np.radians(ellipse[2])
    if np.linalg.norm(c-center)>7*pixel_scale or not .6*expected<=max(sizes)<=1.45*expected or min(sizes)<.18*max(sizes):raise ValueError('unstable_bore_ellipse')
    rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]]);norm=((inliers-c)@rot)/(sizes/2)
    angles=np.sort(np.arctan2(norm[:,1],norm[:,0]));coverage=2*np.pi-np.max(np.diff(np.r_[angles,angles[0]+2*np.pi]))
    if coverage<np.radians(180):raise ValueError('insufficient_bore_arc')
    return dict(center=c.tolist(),sizes=sizes.tolist(),angle_deg=float(ellipse[2]),inliers=len(inliers),arc_deg=float(np.degrees(coverage)))
