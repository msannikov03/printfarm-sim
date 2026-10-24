"""Frame recorder: samples the sim at a fixed video rate and writes mp4.

view: a callable(world) -> (lookat, dist, az, el) or a camera name, or an object with render(cam, world) -> image
(farm.director.Director: picks the shot itself, may add a picture-in-picture). Rendering never writes MjData.
"""
import imageio.v2 as imageio
import numpy as np

from farm.render import Cam


class Recorder:
    def __init__(self, model, path, fps=30, width=1280, height=720, view=None, overlay=None, speed=1.0):
        self.cam = Cam(model, width, height)
        self.fps, self.next_t, self.speed = fps, 0.0, speed     # speed > 1: time-lapse
        self.view = view
        self.overlay = overlay              # callable(img, world) -> img
        self.writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=8)
        self.frames = 0

    def __call__(self, w):
        if w.d.time + 1e-9 < self.next_t:
            return
        if hasattr(self.view, "render"):
            img = self.view.render(self.cam, w)
        else:
            v = self.view(w) if self.view else "overview"
            if isinstance(v, str):
                img = self.cam.rgb(w.d, v)
            else:
                img = self.cam.look(*v).rgb(w.d)
        if self.overlay:
            img = self.overlay(img, w)
        self.writer.append_data(np.ascontiguousarray(img))
        k = self.view.time_scale() if hasattr(self.view, "time_scale") else 1.0   # e.g. the director's slow motion
        self.next_t = max(self.next_t + self.speed * k / self.fps, w.d.time)
        self.frames += 1

    def close(self):
        self.writer.close()
        if hasattr(self.view, "close"):
            self.view.close()
