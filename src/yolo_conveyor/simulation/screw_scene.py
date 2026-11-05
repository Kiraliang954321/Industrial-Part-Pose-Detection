"""MuJoCo screw proxy, rigid-body settling and camera projection in metres."""

import math
import numpy as np

ASSET_ID = "screw_proxy_v0"
ANCHORS = np.array([[0.023, 0, 0], [-0.020, 0, 0]], dtype=float)
CONDITIONS = ("nominal", "low_light", "reflection", "background_shift", "motion_blur", "partial_occlusion")
ASSET = {"id": ASSET_ID, "version": 1, "source": "project-authored procedural geometry",
         "physical_part_match": False, "unit": "m", "head_diameter": 0.012,
         "shaft_diameter": 0.006, "overall_length": 0.043, "head_length": 0.006,
         "keypoints_object_m": ANCHORS.tolist(), "object_axis": "+X points from tip to head",
         "limitations": ["Not a manufacturer CAD or standardized screw model",
                         "Visual concentric rings approximate threads; no helical geometry",
                         "Collision shaft is a smooth cylinder; material/contact parameters unvalidated"]}


def scene_xml(size: int) -> str:
    rings = "\n".join(
        f'<geom name="ring{i}" type="cylinder" pos="{x:.6f} 0 0" '
        'quat="0.7071067812 0 0.7071067812 0" size="0.00335 0.00032" '
        'material="steel" contype="0" conaffinity="0" mass="0" group="1"/>'
        for i, x in enumerate(np.linspace(-0.018, 0.014, 17)))
    return f'''<mujoco model="screw_proxy_v0">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"/>
  <visual><global offwidth="{size}" offheight="{size}"/><quality shadowsize="1024"/>
    <headlight ambient="0.25 0.25 0.25" diffuse="0.3 0.3 0.3" specular="0.1 0.1 0.1"/>
    <map znear="0.001" zfar="10"/></visual>
  <asset><material name="steel" rgba="0.22 0.23 0.24 1" specular="0.5" shininess="0.4"/>
    <material name="surface" rgba="0.7 0.65 0.55 1" specular="0.05" shininess="0.1"/>
  </asset>
  <worldbody>
    <light name="key" pos="0.1 -0.08 0.3" dir="-0.2 0.2 -1" diffuse="0.8 0.8 0.8" castshadow="true"/>
    <camera name="rgb" pos="0 0 0.21" quat="1 0 0 0" fovy="42"/>
    <geom name="table" type="plane" size="0.3 0.3 0.01" material="surface" friction="0.8 0.01 0.001"/>
    <body name="screw" pos="0 0 0.03"><freejoint/>
      <geom name="shaft" type="cylinder" pos="-0.0015 0 0" quat="0.7071067812 0 0.7071067812 0"
        size="0.003 0.0185" material="steel" density="7800" friction="0.6 0.005 0.0001"/>
      <geom name="head" type="cylinder" pos="0.02 0 0" quat="0.7071067812 0 0.7071067812 0"
        size="0.006 0.003" material="steel" density="7800" friction="0.6 0.005 0.0001"/>
      {rings}
      <site name="head_axis_end" pos="0.023 0 0" size="0.0001" rgba="0 0 0 0"/>
      <site name="tip_axis_end" pos="-0.020 0 0" size="0.0001" rgba="0 0 0 0"/>
    </body>
    <geom name="occluder" type="box" pos="1 1 0.015" size="0.003 0.012 0.004"
      rgba="0.5 0.4 0.3 1" contype="0" conaffinity="0"/>
  </worldbody>
</mujoco>'''


class ScrewScene:
    def __init__(self, size: int = 640, *, xml: str | None = None):
        import mujoco
        self.mj = mujoco
        self.size = size
        self.xml = scene_xml(size) if xml is None else xml
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=size, width=size)
        self.camera_id = self.model.camera("rgb").id
        self.body_id = self.model.body("screw").id
        self.screw_geom_ids = np.flatnonzero(self.model.geom_bodyid == self.body_id)

    def close(self):
        self.renderer.close()

    def _initialize_additional_bodies(self):
        """Extension point for a parked robot; the original screw scene is unchanged."""

    def render(self, rng, condition: str, has_object: bool) -> tuple:
        mj, m, d = self.mj, self.model, self.data
        mj.mj_resetData(m, d)
        shade = float(rng.uniform(0.12, 0.36))
        m.mat_rgba[m.material("steel").id, :3] = [shade * 0.97, shade, shade * 1.03]
        background = np.repeat(rng.uniform(0.57, 0.88), 3)
        if condition == "background_shift":
            background *= rng.uniform(0.65, 1.0, 3)
        else:
            background *= [1.0, 0.96, 0.9]
        m.mat_rgba[m.material("surface").id, :3] = background
        m.mat_specular[m.material("steel").id] = 1.0 if condition == "reflection" else rng.uniform(0.1, 0.5)
        m.mat_shininess[m.material("steel").id] = 0.8 if condition == "reflection" else 0.25
        intensity = rng.uniform(0.18, 0.35) if condition == "low_light" else rng.uniform(0.65, 1.05)
        m.light_diffuse[0] = intensity
        m.light_pos[0] = [rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15), 0.25]
        m.light_dir[0] = -m.light_pos[0] / np.linalg.norm(m.light_pos[0])
        yaw = rng.uniform(-math.pi, math.pi)
        d.qpos[:3] = [rng.uniform(-0.043, 0.043), rng.uniform(-0.043, 0.043), 0.025]
        d.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        occluder = m.geom("occluder").id
        m.geom_pos[occluder] = [1, 1, 0.015]
        self._initialize_additional_bodies()
        mj.mj_forward(m, d)
        for _ in range(250):
            mj.mj_step(m, d)
        if not has_object:
            d.qpos[2] = -2.0
            d.qvel[:] = 0
        if condition == "partial_occlusion" and has_object:
            m.geom_pos[occluder] = [d.qpos[0], d.qpos[1], 0.019]
            m.geom_rgba[occluder, :3] = background * 0.75
        mj.mj_forward(m, d)
        option = mj.MjvOption()
        option.sitegroup[:] = 0
        self.renderer.update_scene(d, camera="rgb", scene_option=option)
        rgb = self.renderer.render().copy()
        self.renderer.enable_segmentation_rendering()
        segments = self.renderer.render().copy()
        self.renderer.disable_segmentation_rendering()
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        mask = (segments[:, :, 1] == int(mj.mjtObj.mjOBJ_GEOM)) & np.isin(segments[:, :, 0], self.screw_geom_ids)
        world_from_gl = d.cam_xmat[self.camera_id].reshape(3, 3).copy()
        world_from_cv = world_from_gl @ np.diag([1.0, -1.0, -1.0])
        camera_pos = d.cam_xpos[self.camera_id].copy()
        focal = 0.5 * self.size / math.tan(math.radians(m.cam_fovy[self.camera_id]) * 0.5)
        k = np.array([[focal, 0, self.size / 2], [0, focal, self.size / 2], [0, 0, 1]])
        world_from_object = d.xmat[self.body_id].reshape(3, 3).copy()
        object_pos = d.xpos[self.body_id].copy()
        t_world_cam = np.eye(4)
        t_world_cam[:3, :3], t_world_cam[:3, 3] = world_from_cv, camera_pos
        t_world_obj = np.eye(4)
        t_world_obj[:3, :3], t_world_obj[:3, 3] = world_from_object, object_pos
        cam_points = (ANCHORS @ world_from_object.T + object_pos - camera_pos) @ world_from_cv
        pixels = cam_points @ k.T
        pixels = pixels[:, :2] / pixels[:, 2:3]
        visibility = []
        for (x, y), z in zip(pixels, cam_points[:, 2]):
            inside = 0 <= x < self.size and 0 <= y < self.size and z > 0
            if not inside:
                visibility.append(0)
            else:
                ix, iy = min(self.size - 1, int(x)), min(self.size - 1, int(y))
                visibility.append(2 if mask[iy, ix] and abs(float(depth[iy, ix]) - z) < 0.001 else 1)
        import cv2
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if condition == "motion_blur":
            kernel = np.zeros((5, 5), dtype=np.float32)
            kernel[2, :] = 0.2
            bgr = cv2.filter2D(bgr, -1, kernel)
        camera = {"K": k.tolist(), "T_world_camera_cv": t_world_cam.tolist(), "distortion": [0] * 5}
        return bgr, mask, pixels, visibility, camera, t_world_obj, {
            "initial_yaw_rad": float(yaw), "surface_rgb": background.tolist(), "steel_shade": shade,
            "light_intensity": float(intensity), "light_position_m": m.light_pos[0].tolist(),
            "settling_steps": 250, "settling_seconds": float(250 * m.opt.timestep),
            "residual_speed": float(np.linalg.norm(d.qvel)), "blur_kernel_px": 5 if condition == "motion_blur" else 0}
