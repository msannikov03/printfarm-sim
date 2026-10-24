"""Offscreen rendering helpers (headless Linux: MUJOCO_GL=egl or osmesa).

Geom groups shown by default: 0-2 (physics + lab dressing) and 4 (beauty-only detail from farm.lab, which the chamber
depth cameras in perceive/ do not render). Group 3 holds the YAM collision capsules and collision proxies; group 5 is
the showcase set dressing (scripts/render_showcase.py turns it on).
"""
import mujoco
import numpy as np

DEFAULT_GROUPS = (0, 1, 2, 4)


class Cam:
    def __init__(self, model, width=1280, height=720, groups=DEFAULT_GROUPS):
        if model.vis.global_.offwidth < width or model.vis.global_.offheight < height:
            model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)      # render-only setting
            model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        self.r = mujoco.Renderer(model, height=height, width=width)
        self.free = mujoco.MjvCamera()
        self.free.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        self.opt.geomgroup[list(groups)] = 1    # hide YAM collision capsules (group 3) and sites

    def look(self, lookat, dist, az, el):
        self.free.lookat[:] = lookat
        self.free.distance, self.free.azimuth, self.free.elevation = dist, az, el
        return self

    def rgb(self, data, camera=None):
        self.r.update_scene(data, camera=self.free if camera is None else camera, scene_option=self.opt)
        return self.r.render().copy()

    def depth(self, data, camera):
        self.r.enable_depth_rendering()
        self.r.update_scene(data, camera=camera, scene_option=self.opt)
        d = self.r.render().copy()
        self.r.disable_depth_rendering()
        return d

    def close(self):
        self.r.close()


def save(img, path):
    import imageio.v3 as iio
    iio.imwrite(path, img)
