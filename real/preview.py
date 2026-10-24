"""Offscreen MuJoCo renders of the SO-101 over the bed: sanity-check a plan before moving the real arm."""
from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from real.calib import BedCalib
from real.so101 import JOINTS, MODEL_XML, SO101

W, H = 640, 480


def _quat_from_R(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


def build_scene(calib: BedCalib | None = None, objects: list[dict] | None = None, bed_thickness_mm: float = 6.0):
    spec = mujoco.MjSpec.from_file(str(MODEL_XML))
    spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
    spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]
    spec.visual.headlight.specular = [0.0, 0.0, 0.0]
    wb = spec.worldbody
    light = wb.add_light()
    light.pos, light.dir = [0.2, 0.0, 1.5], [0, 0, -1]
    light.diffuse, light.specular = [0.45, 0.45, 0.45], [0.05, 0.05, 0.05]
    floor_z = 0.0
    if calib is not None:
        bw, bh = calib.bed_mm
        c = calib.bed_to_robot([bw / 2, bh / 2, -bed_thickness_mm / 2])
        g = wb.add_geom()
        g.name = "bed"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [bw / 2000, bh / 2000, bed_thickness_mm / 2000]
        g.pos, g.quat = c, _quat_from_R(calib.R)
        g.rgba = [0.22, 0.22, 0.25, 1]
        g.contype = g.conaffinity = 0
        # bed origin axes: red = bed +x, green = bed +y
        for k, (d, rgba) in enumerate((((1, 0), [0.9, 0.1, 0.1, 1]), ((0, 1), [0.1, 0.8, 0.1, 1]))):
            a = calib.bed_to_robot([0, 0, 1.0])
            b = calib.bed_to_robot([40 * d[0], 40 * d[1], 1.0])
            ax = wb.add_geom()
            ax.type = mujoco.mjtGeom.mjGEOM_CAPSULE
            ax.fromto = np.r_[a, b]
            ax.size = [0.002, 0, 0]
            ax.rgba = rgba
            ax.contype = ax.conaffinity = 0
        floor_z = min(0.0, float(calib.bed_to_robot([bw / 2, bh / 2, -bed_thickness_mm])[2])) - 0.001
        if floor_z < -0.002:                      # pedestal under the robot
            ped = wb.add_geom()
            ped.type = mujoco.mjtGeom.mjGEOM_BOX
            ped.size = [0.06, 0.06, -floor_z / 2]
            ped.pos = [0.02, 0, floor_z / 2]
            ped.rgba = [0.5, 0.5, 0.5, 1]
            ped.contype = ped.conaffinity = 0
    fl = wb.add_geom()
    fl.type = mujoco.mjtGeom.mjGEOM_PLANE
    fl.size = [1, 1, 0.01]
    fl.pos = [0, 0, floor_z - 0.001]
    fl.rgba = [0.42, 0.42, 0.40, 1]
    fl.contype = fl.conaffinity = 0
    for i, o in enumerate(objects or []):
        poly = np.asarray(o.get("polygon_mm") or [], float)
        h = o.get("height_mm") or 5.0
        if len(poly) < 3:
            c, (w_, d_), yaw = o["center_mm"], o["footprint_mm"], math.radians(o["yaw_deg"])
            loc = np.array([[-w_, -d_], [w_, -d_], [w_, d_], [-w_, d_]]) / 2
            rot = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
            poly = loc @ rot.T + c
        verts = calib.bed_to_robot(np.vstack([np.c_[poly, np.zeros(len(poly))], np.c_[poly, np.full(len(poly), h)]]))
        m = spec.add_mesh()
        m.name = f"part{i}"
        m.uservert = verts.ravel().tolist()
        g = wb.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_MESH
        g.meshname = m.name
        g.rgba = [0.95, 0.55, 0.1, 1] if i % 2 == 0 else [0.2, 0.55, 0.95, 1]
        g.contype = g.conaffinity = 0
    tb = wb.add_body()
    tb.name = "target"
    tb.mocap = True
    tg = tb.add_geom()
    tg.type = mujoco.mjtGeom.mjGEOM_SPHERE
    tg.size = [0.005, 0, 0]
    tg.rgba = [1, 0.1, 0.1, 0.8]
    tg.contype = tg.conaffinity = 0
    return spec.compile()


class Previewer:
    def __init__(self, calib: BedCalib | None = None, objects=None, size=(W, H)):
        self.model = build_scene(calib, objects)
        self.data = mujoco.MjData(self.model)
        self.calib = calib
        self.renderer = mujoco.Renderer(self.model, size[1], size[0])
        self.qadr = [self.model.joint(j).qposadr[0] for j in JOINTS]
        m = self.model
        root = m.body("base").id
        robot_bodies = {b for b in range(m.nbody) if b == root or self._descends(b, root)}
        self.robot_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in robot_bodies and m.geom_group[g] == 2]

    def _descends(self, b, root):
        while b > 0:
            b = self.model.body_parentid[b]
            if b == root:
                return True
        return False

    def _ghost(self, on: bool):
        """Semi-transparent arm for the top view so the part under the gripper stays visible."""
        m = self.model
        if on:
            self._saved = (m.geom_matid[self.robot_geoms].copy(), m.geom_rgba[self.robot_geoms].copy())
            m.geom_matid[self.robot_geoms] = -1
            m.geom_rgba[self.robot_geoms] = [1.0, 0.85, 0.1, 0.3]
        else:
            m.geom_matid[self.robot_geoms], m.geom_rgba[self.robot_geoms] = self._saved

    def _cam(self, lookat, dist, az, el):
        cam = mujoco.MjvCamera()
        cam.lookat[:] = lookat
        cam.distance, cam.azimuth, cam.elevation = dist, az, el
        return cam

    def render(self, q, path, target=None, lines=(), line_to=None):
        import cv2
        self.data.qpos[self.qadr] = q
        if target is not None:
            self.data.mocap_pos[0] = target
        else:
            self.data.mocap_pos[0] = [0, 0, -1]
        mujoco.mj_forward(self.model, self.data)
        if self.calib is not None:
            bc = self.calib.bed_to_robot([self.calib.bed_mm[0] / 2, self.calib.bed_mm[1] / 2, 0])
            look = (bc + np.array([0.02, 0, 0.1])) / 2
            fwd = math.degrees(math.atan2(bc[1], bc[0]))
        else:
            look, fwd = np.array([0.15, 0, 0.12]), 0.0
        if self.calib is not None:
            top_look = self.calib.bed_to_robot([self.calib.bed_mm[0] / 2, self.calib.bed_mm[1] * 0.4, 0])
            top_dist = 0.0028 * max(self.calib.bed_mm) + 0.15
        else:
            top_look, top_dist = look, 0.7
        views = [(self._cam(look, 0.85, fwd + 120, -28), False), (self._cam(top_look, top_dist, fwd, -89.5), True)]
        imgs = []
        for cam, ghost in views:
            if ghost:
                self._ghost(True)
            self.renderer.update_scene(self.data, cam)
            imgs.append(self.renderer.render().copy())
            if ghost:
                self._ghost(False)
        img = np.concatenate(imgs, axis=1)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        for k, s in enumerate(lines):
            cv2.putText(img, s, (10, 22 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, s, (10, 22 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "3/4 view", (10, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
        cv2.putText(img, "top view, arm ghosted (red = bed +x, green = bed +y)", (img.shape[1] // 2 + 10, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), img)
        return path


def render_q(q, path, calib=None, objects=None):
    return Previewer(calib, objects).render(q, path, lines=[f"model deg {np.degrees(q[:5]).round(1).tolist()}"])


if __name__ == "__main__":
    import argparse
    from real.arm import JointMap
    from real.calib import CALIB_JSON, BedCalib, nominal
    ap = argparse.ArgumentParser(description="render the SO-101 model at given joints (compare with the real arm)")
    ap.add_argument("out", nargs="?", default="runs/real_pose.png")
    ap.add_argument("--lerobot", default="0,0,0,0,0,0",
                    help="LeRobot values (deg x5, gripper 0-100) in motor order: pan,lift,elbow,wrist_flex,roll,grip")
    a = ap.parse_args()
    vals = [float(v) for v in a.lerobot.split(",")]
    from real.so101 import JOINTS as J
    q = JointMap.load_default().lerobot_to_model(dict(zip(J, vals)))
    calib = BedCalib.load(CALIB_JSON) if CALIB_JSON.exists() else nominal()
    tip = SO101().fk(q)[0]
    print(Previewer(calib).render(q, a.out, target=tip, lines=[f"lerobot {vals}",
                                                               f"fixed-jaw tip {np.round(tip * 1000).tolist()} mm"]))
