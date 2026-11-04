"""Bounded observation poses computed from robot FK, without an object pose."""
import cv2
import numpy as np
from yolo_conveyor.pose_cell.control import CellController
from yolo_conveyor.pose_cell.geometry import about
from yolo_conveyor.pose_cell.scene import CENTER
from yolo_conveyor.simulation.panda_control import Stage

OBSERVATION_ANGLES=(20.,-20.)


class ActiveController(CellController):
    def observe_pose(self,anchor,index):
        if index not in range(len(OBSERVATION_ANGLES)):raise ValueError('observation budget exceeded')
        anchor=np.asarray(anchor);tool=self.scene.tool_transform()
        if np.linalg.norm(anchor[:3,3]-CENTER)>.025:raise ValueError('observation outside inspection cell')
        goal=about(anchor[:3,2],OBSERVATION_ANGLES[index])@anchor[:3,:3]
        vector=cv2.Rodrigues(goal@tool[:3,:3].T)[0].ravel();n=max(1,int(np.ceil(np.linalg.norm(vector)/np.radians(10))))
        q=self.scene.data.qpos[self.scene.arm_qpos].copy();items=[]
        for f in np.linspace(0,1,n+1)[1:]:
            r=cv2.Rodrigues(vector*f)[0]@tool[:3,:3];p=tool[:3,3]+f*(anchor[:3,3]-tool[:3,3])
            q=self.inverse_kinematics(p,r,q);items.append(Stage('active_observe',q,self.close_width,.55))
        items.append(Stage('hold',q,self.close_width,.5))
        return self.stages(items)
