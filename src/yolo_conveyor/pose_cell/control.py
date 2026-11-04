"""Robot feedback plus supplied visual estimates; no object truth access."""
import cv2
import numpy as np
from yolo_conveyor.annular.control import AssemblyController
from yolo_conveyor.simulation.panda_control import Stage
from .scene import CENTER,NEST,NEST_TOP,SIDE_R,SHAFT_XY,SHAFT_TOP,ENGAGEMENT,CLOSE_BY_KIND
from .geometry import about,rz

CLOSE=.007  # Empty jaws remain clear of the feed post; larger parts still develop servo grip force.


class CellController(AssemblyController):
    @property
    def close_width(self):return CLOSE_BY_KIND[self.scene.kind]

    def pick(self):
        q=self.scene.data.qpos[self.scene.arm_qpos].copy();z=NEST_TOP+self.scene.spec['height']/2
        p=np.array([*NEST,z]);approach=self.inverse_kinematics(p-SIDE_R[:,2]*.07,SIDE_R,q)
        grip=self.inverse_kinematics(p,SIDE_R,approach)
        lifted=self.inverse_kinematics([*NEST,.31],SIDE_R,grip)
        inspect=self.inverse_kinematics(CENTER,SIDE_R,lifted)
        self.inspect_hint=inspect.copy()
        return self.stages([Stage('approach',approach,.04,1.5),Stage('side_grasp',grip,.04,1.3),Stage('hold',grip,.04,.2),
                            Stage('close',grip,self.close_width,.7),Stage('lift',lifted,self.close_width,1.2),Stage('inspect',inspect,self.close_width,1.3),Stage('hold',inspect,self.close_width,.7)])

    def present(self,angle,yaw=0):
        q=self.scene.data.qpos[self.scene.arm_qpos].copy();items=[];count=max(1,int(np.ceil(abs(angle)/25)))
        for f in np.linspace(0,1,count+1)[1:]:
            r=rz(yaw*f)@about(SIDE_R[:,2],angle*f)@SIDE_R
            q=self.inverse_kinematics(CENTER,r,q);items.append(Stage('present',q,self.close_width,.7))
        items.append(Stage('hold',q,self.close_width,.7));return self.stages(items)

    def correction(self,estimate,target_r,center=CENTER):
        tool=self.scene.tool_transform();object_r=np.asarray(estimate['rotation']);object_t=np.asarray(estimate['translation_m'])
        relative_r=tool[:3,:3].T@object_r;relative_t=tool[:3,:3].T@(object_t-tool[:3,3])
        desired_r=np.asarray(target_r)@relative_r.T;desired_t=np.asarray(center)-desired_r@relative_t
        delta=cv2.Rodrigues(desired_r@tool[:3,:3].T)[0].ravel();length=np.linalg.norm(delta)
        alternatives=[delta]
        if length>1e-5:alternatives.append(delta*(1-2*np.pi/length))
        failures=[]
        for vector in alternatives:
            try:
                q=self.scene.data.qpos[self.scene.arm_qpos].copy();items=[];n=max(1,int(np.ceil(np.linalg.norm(vector)/np.radians(20))))
                for f in np.linspace(0,1,n+1)[1:]:
                    r=cv2.Rodrigues(vector*f)[0]@tool[:3,:3];p=tool[:3,3]+f*(desired_t-tool[:3,3])
                    q=self.inverse_kinematics(p,r,q);items.append(Stage('visual_correct',q,self.close_width,.7))
                # Resolve redundant-joint winding near the inspection configuration before
                # the next fresh visual check; the same end pose can otherwise trap joint 7.
                hint=getattr(self,'inspect_hint',q)
                seeds=[hint,q]
                for offset in (-.7,.7,-1.2,1.2):
                    seed=hint.copy();seed[0]+=offset;seed[2]-=offset
                    seeds.append(np.clip(seed,self.scene.arm_ranges[:,0]+.02,self.scene.arm_ranges[:,1]-.02))
                for seed in seeds:
                    try:
                        compact=self.inverse_kinematics(desired_t,desired_r,seed)
                        # Kinematic lookahead selects a redundant-joint configuration
                        # that can also reach the shaft. It uses the measured grip
                        # transform and fixed fixture only, never object truth.
                        self._insertion_stages(compact,desired_r,relative_t,desired_t)
                        return self.stages(items+[Stage('joint_reconfigure',compact,self.close_width,1.2),Stage('hold',compact,self.close_width,.8)])
                    except ValueError:continue
                return self.stages(items+[Stage('hold',q,self.close_width,.8)])
            except ValueError as exc:failures.append(str(exc))
        raise ValueError('correction_unreachable: '+'; '.join(failures))

    def insert(self,estimate,target_r):
        tool=self.scene.tool_transform();r=np.asarray(target_r)@(tool[:3,:3].T@np.asarray(estimate['rotation'])).T
        local=tool[:3,:3].T@(np.asarray(estimate['translation_m'])-tool[:3,3])
        return self.stages(self._insertion_stages(self.scene.data.qpos[self.scene.arm_qpos].copy(),r,local,tool[:3,3]))

    def _insertion_stages(self,q,r,local,start):
        h=self.scene.spec['height']/2;items=[];previous=np.asarray(start).copy()
        for name,z,duration in [('transfer',CENTER[2],1.5),('above_shaft',SHAFT_TOP+h+.015,1.5)]:
            destination=np.array([*SHAFT_XY,z])-r@local;n=max(1,int(np.ceil(np.linalg.norm(destination-previous)/.01)))
            for f in np.linspace(0,1,n+1)[1:]:
                q=self.inverse_kinematics(previous+f*(destination-previous),r,q);items.append(Stage(name,q,self.close_width,max(.18,duration/n)))
            previous=destination
        for z in np.linspace(SHAFT_TOP+h+.015,SHAFT_TOP+h-ENGAGEMENT,19)[1:]:
            q=self.inverse_kinematics(np.array([*SHAFT_XY,z])-r@local,r,q);items.append(Stage('preposition',q,self.close_width,.3))
        items.append(Stage('hold',q,self.close_width,.7));return items
