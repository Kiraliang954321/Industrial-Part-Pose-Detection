"""Read-only line witness and diagnostic video. Truth never feeds event decisions."""

import json
import math
import cv2
import numpy as np


class LineWitness:
    def __init__(self, scene, sequencer, trace, video_path=None):
        self.scene, self.machine, self.trace = scene, sequencer, trace
        self.start = float(scene.data.time)
        self.last_trace = -1.; self.last_frame = -1.
        self.last_center = scene.object_anchors().mean(axis=0)
        self.max_step_m = 0.; self.max_penetration_m = 0.; self.illegal_steps = 0
        self.inspection_image = None; self.video_frames = 0
        self.writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 12, (1280,704)) if video_path else None
        if self.writer and not self.writer.isOpened():
            raise RuntimeError("line video writer unavailable")

    def truth(self):
        s = self.scene
        points = s.object_anchors(); center = points.mean(axis=0); direction = points[0]-points[1]
        supports = set(); penetration = 0.
        for contact in s.data.contact:
            a,b = int(contact.geom1),int(contact.geom2)
            if s.body_id not in (int(s.model.geom_bodyid[a]),int(s.model.geom_bodyid[b])):
                continue
            penetration = max(penetration,float(-contact.dist))
            if contact.dist <= 0:
                other = b if int(s.model.geom_bodyid[a]) == s.body_id else a
                supports.add(s.model.geom(other).name)
        return {"center_world_m": center.tolist(), "heading_world_deg": math.degrees(math.atan2(direction[1],direction[0]))%360,
                "projected_axis_length_m": float(np.linalg.norm(direction[:2])), "support_contacts": sorted(supports),
                "penetration_m": penetration, "linear_speed_m_s":float(np.linalg.norm(s.data.qvel[:3]))}

    def sample(self, stage):
        s = self.scene; t = float(s.data.time-self.start); truth = self.truth()
        center = np.array(truth["center_world_m"])
        self.max_step_m = max(self.max_step_m,float(np.linalg.norm(center-self.last_center))); self.last_center=center
        self.max_penetration_m = max(self.max_penetration_m,truth["penetration_m"])
        illegal = bool(s.illegal_contacts());self.illegal_steps += int(illegal)
        if t-self.last_trace >= .02:
            self.trace.write(json.dumps({"sim_s":t,"stage":stage,"state":self.machine.state.value,
                                         "index_encoder_m":float(s.data.qpos[s.pallet_q]),"robot_q":s.data.qpos[s.arm_qpos].tolist(),
                                         "object_truth":truth,"illegal_contact":illegal},allow_nan=False)+"\n")
            self.last_trace=t
        if self.writer and t-self.last_frame >= 1/12:
            overview=s.capture("overview")
            if self.machine.state.value == "REJECTING":
                detail=s.capture("closeup"); detail_title="PHYSICS: robot reject placement"
            elif self.inspection_image is not None:
                detail=self.inspection_image; detail_title="LAST INSPECTION INPUT (not visual servo)"
            else:
                detail=s.capture("rgb"); detail_title="FIXED INSPECTION CAMERA"
            frame=np.vstack([np.full((64,1280,3),242,np.uint8),np.hstack([overview,detail])])
            outcome=self.machine.event["outcome"] if self.machine.event else "pending"
            color={"PASS":(30,140,30),"FAIL_ORIENTATION":(20,20,190),"UNCERTAIN":(0,110,190)}.get(outcome,(40,40,40))
            cv2.putText(frame,f"{self.machine.state.value} | {outcome} | {stage} | sim {t:.2f}s",(12,25),cv2.FONT_HERSHEY_SIMPLEX,.6,color,1)
            cv2.putText(frame,"INDEXED PALLET / SINGLE-PIECE SIMULATION",(12,50),cv2.FONT_HERSHEY_SIMPLEX,.48,(50,50,50),1)
            cv2.putText(frame,detail_title,(650,50),cv2.FONT_HERSHEY_SIMPLEX,.44,(50,50,50),1)
            self.writer.write(frame);self.video_frames+=1;self.last_frame=t

    def close(self):
        if self.writer:self.writer.release()
