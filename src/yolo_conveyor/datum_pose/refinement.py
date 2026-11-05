"""Local grayscale edge refinement, seeded only by predicted mechanical datums."""
from copy import deepcopy
import cv2
import numpy as np


def refine(image,predictions,kind,pixel_scale=1.):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);output=deepcopy(predictions)
    for item in output:
        points=np.array(item['keypoints'],float);original=points.copy()
        if points.shape!=(5,3) or not np.isfinite(points).all():continue
        # Semantic corner identities remain from the network; do not snap to arbitrary global features.
        corners=points[1:,:2].astype(np.float32).reshape(-1,1,2)
        window=int(7*pixel_scale);margin=window+3
        if np.all(corners>margin) and np.all(corners<640-margin):
            cv2.cornerSubPix(gray,corners,(window,window),(-1,-1),(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_MAX_ITER,40,.01))
            for j,p in enumerate(corners[:,0],1):
                if np.linalg.norm(p-original[j,:2])<=8*pixel_scale:points[j,:2]=p
        # The near bore mouth is circular. Exclude the keyway direction while fitting its ellipse.
        center=original[0,:2];direction=original[1:,:2].mean(0)-center
        dnorm=np.linalg.norm(direction)
        if dnorm>5:
            direction/=dnorm;x,y=np.round(center).astype(int);radius=int(55*pixel_scale)
            x0,y0=max(0,x-radius),max(0,y-radius);x1,y1=min(640,x+radius),min(640,y+radius)
            patch=gray[y0:y1,x0:x1];edge=cv2.Canny(patch,25,70)
            contours,_=cv2.findContours(edge,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
            box=np.array(item['bbox']);expected=max(box[2]-box[0],box[3]-box[1])*(.008/.025 if kind=='bearing' else .006/.02241)
            options=[]
            for contour in contours:
                pts=contour[:,0,:].astype(float)+[x0,y0]
                if len(pts)<18:continue
                if kind=='gear':
                    delta=pts-center;cos=(delta@direction)/np.maximum(np.linalg.norm(delta,axis=1),1.)
                    pts=pts[cos<.72]
                if len(pts)<12:continue
                ellipse=cv2.fitEllipse(pts.astype(np.float32));c=np.array(ellipse[0]);sizes=np.array(ellipse[1]);major=max(sizes)
                if not (.65*expected<=major<=1.4*expected and min(sizes)>.2*major and np.linalg.norm(c-center)<=12*pixel_scale):continue
                a=np.radians(ellipse[2]);rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
                norm=np.linalg.norm(((pts-c)@rot)/(sizes/2),axis=1)
                residual=float(np.median(np.abs(norm-1)))
                if residual<.08:options.append((residual*100+abs(major-expected)/expected+np.linalg.norm(c-center)/30,c,sizes,float(ellipse[2]),residual))
            if options:
                selected=min(options,key=lambda x:x[0]);points[0,:2]=selected[1]
                item['bore_ellipse']=dict(center=selected[1].tolist(),sizes=selected[2].tolist(),angle_deg=selected[3],residual=selected[4])
        item['keypoints']=points.tolist()
        item['edge_refinement_shift_px']=np.linalg.norm(points[:,:2]-original[:,:2],axis=1).tolist()
    return output


def metrology_points(item):
    """Ellipse axes plus the RGB key center define a directed circle frame.

    This uses the declared upward-axis/visible lower-face envelope. The two circle
    samples are correlated geometric constructions, not independent detections.
    """
    ellipse=item.get('bore_ellipse')
    if ellipse is None:return None
    points=np.array(item['keypoints']);c=np.array(ellipse['center']);angle=np.radians(ellipse['angle_deg'])
    rot=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    mapping=rot@np.diag(np.array(ellipse['sizes'])/2)
    key=points[1:5,:2].mean(0);direction=np.linalg.solve(mapping,key-c)
    length=np.linalg.norm(direction)
    if not np.isfinite(length) or length<.2:return None
    direction/=length;perpendicular=np.array([-direction[1],direction[0]])
    confidence=min(float(points[0,2]),float(np.median(points[1:5,2])))
    return [[*c,confidence],[*(c+mapping@direction),confidence],[*(c+mapping@perpendicular),confidence],[*(c-mapping@perpendicular),confidence]]
