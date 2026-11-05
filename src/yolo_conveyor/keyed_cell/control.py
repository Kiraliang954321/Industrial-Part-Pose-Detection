"""Visual-grip-transform motion with separate approach, insertion and retreat."""
import numpy as np
from yolo_conveyor.active_pose.control import ActiveController
from yolo_conveyor.pose_cell.scene import CENTER,SHAFT_XY,SHAFT_TOP,ENGAGEMENT
from yolo_conveyor.simulation.panda_control import Stage

APPROACH_GAP=.006
INSERT_STEP=.0005
RETRACT_DISTANCE=.008


class KeyedController(ActiveController):
    def _preflight(self,stages,initial):
        s=self.scene;d=self.ik_data;previous=np.asarray(initial).copy()
        fingers=s.data.qpos[s.finger_qpos].copy()
        holding=bool(np.min(fingers)>self.close_width+.0004 and np.max(fingers)<.025)
        for stage in stages:
            for f in np.linspace(0,1,30):
                d.qpos[:]=self.m.qpos0;d.qpos[s.arm_qpos]=previous*(1-f)+stage.q*f
                # A closed-position command is not the actual opening around a held
                # part. Use encoder feedback, without reading the workpiece pose.
                d.qpos[s.finger_qpos]=fingers if holding and stage.gripper<=self.close_width+.001 else max(.002,stage.gripper)
                self.mj.mj_forward(self.m,d)
                if s.illegal_contacts(d):raise ValueError('unsafe joint path at '+stage.name+': '+s.illegal_contacts(d)[0])
            previous=stage.q

    def assembly_geometry(self,measurement,target_r):
        tool=self.scene.tool_transform();r=tool[:3,:3].T@np.asarray(measurement['rotation'])
        local=tool[:3,:3].T@(np.asarray(measurement['translation_m'])-tool[:3,3])
        return dict(rotation=(np.asarray(target_r)@r.T).tolist(),local_center_m=local.tolist())

    def _insertion_stages(self,q,r,local,start):
        approach=self._approach(q,r,local,start)
        return approach+self._insert_from(approach[-1].q,r,local)

    def _approach(self,q,r,local,start):
        items=[];previous=np.asarray(start).copy();h=self.scene.spec['height']/2
        for name,z in [('transfer',CENTER[2]),('above_index',SHAFT_TOP+h+APPROACH_GAP)]:
            destination=np.array([*SHAFT_XY,z])-r@local;n=max(1,int(np.ceil(np.linalg.norm(destination-previous)/.01)))
            for f in np.linspace(0,1,n+1)[1:]:
                q=self.inverse_kinematics(previous+f*(destination-previous),r,q);items.append(Stage(name,q,self.close_width,.2))
            previous=destination
        items.append(Stage('hold',q,self.close_width,.6));return items

    def _insert_from(self,q,r,local):
        h=self.scene.spec['height']/2;items=[];n=int(np.ceil((APPROACH_GAP+ENGAGEMENT)/INSERT_STEP))
        for z in np.linspace(SHAFT_TOP+h+APPROACH_GAP,SHAFT_TOP+h-ENGAGEMENT,n+1)[1:]:
            q=self.inverse_kinematics(np.array([*SHAFT_XY,z])-r@local,r,q);items.append(Stage('keyed_insert',q,self.close_width,.18))
        items.append(Stage('hold',q,self.close_width,.5));return items

    def approach(self,geometry):
        tool=self.scene.tool_transform()
        return self.stages(self._approach(self.scene.data.qpos[self.scene.arm_qpos].copy(),np.array(geometry['rotation']),np.array(geometry['local_center_m']),tool[:3,3]))

    def preinsert(self,geometry):
        return self.stages(self._insert_from(self.scene.data.qpos[self.scene.arm_qpos].copy(),np.array(geometry['rotation']),np.array(geometry['local_center_m'])))

    def retreat(self):
        tool=self.scene.tool_transform();q=self.scene.data.qpos[self.scene.arm_qpos].copy();items=[]
        for distance in np.linspace(0,RETRACT_DISTANCE,9)[1:]:
            q=self.inverse_kinematics(tool[:3,3]+[0,0,distance],tool[:3,:3],q);items.append(Stage('force_retreat',q,self.close_width,.18))
        items.append(Stage('hold',q,self.close_width,.5));return self.stages(items)

    def stop_hold(self,seconds=.15):
        return [Stage('hold',self.scene.data.qpos[self.scene.arm_qpos].copy(),self.close_width,seconds)]
