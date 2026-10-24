"""Dive into the running farm with the interactive MuJoCo viewer: orbit, zoom and pause while the robot works.

macOS needs mjpython for the passive viewer (it ships with the mujoco wheel: .venv/bin/mjpython); Linux: python.
    PYTHONPATH=. .venv/bin/mjpython scripts/view_farm.py                       # the whole farm (as run_farm.py)
    PYTHONPATH=. .venv/bin/mjpython scripts/view_farm.py --unload 5            # one unload: P5 door, picks, door
    PYTHONPATH=. .venv/bin/python scripts/view_farm.py --unload 5 --dry-run 300  # no window: step 300 ticks
Keys as in the MuJoCo viewer: space pause/resume the farm, + / - faster / slower (sim time per wall second),
Esc free camera, ctrl + right double-click on the robot to track it (or start with --follow).
Mouse: left-drag orbit, right-drag pan, scroll zoom, double-click selects a body. Ctrl+drag on a selected body
pushes it: that changes the run (everything else here is view-only).
Static scene, nothing scripted (the plain blocking viewer; plain python is fine on macOS too):
    .venv/bin/python -m mujoco.viewer --mjcf=build/printfarm.mjb
build/printfarm.mjb is written by this script: the compiled model as the farm runs it, with the pooled parts parked
(no contacts, gravity-compensated). build/printfarm.xml is the raw MJCF; opened directly, the parked parts pop out of
the floor once its physics runs.
"""
import argparse
import pathlib
import sys
import time
import warnings

import mujoco
import mujoco.viewer
import numpy as np

from farm import catalog as C
from farm import scene
from farm.render import DEFAULT_GROUPS
from farm.world import World
from plan import door as D
from plan.door import TUCK
from plan.ik import IK
from plan.motion import Executor
from sim.farm import Farm

KEY_SPACE = 32
KEYS_FASTER = (61, 334)        # GLFW '=' (the '+' key) and keypad +
KEYS_SLOWER = (45, 333)        # GLFW '-' and keypad -
STATIC_XML, STATIC_MJB = pathlib.Path("build/printfarm.xml"), pathlib.Path("build/printfarm.mjb")


class Stop(Exception):
    """The viewer window was closed (or the dry run is over)."""


class DryViewer:
    """Stand-in for the passive viewer handle: no window, stops after `ticks` control ticks."""

    def __init__(self, ticks):
        self.left, self.cam, self.opt = ticks, mujoco.MjvCamera(), mujoco.MjvOption()

    def is_running(self):
        return self.left > 0

    def sync(self):
        self.left -= 1

    def lock(self):
        import contextlib
        return contextlib.nullcontext()

    def set_texts(self, texts):
        pass


class Pacer:
    """Executor hook, called every control tick (10 ms of sim): keeps the viewer in sync and the sim at real time
    x rate, holds while paused, ends the run when the window closes."""

    def __init__(self, viewer, world, farm, rate=1.0, realtime=True, max_ticks=0):
        self.v, self.w, self.farm = viewer, world, farm
        self.rate, self.paused, self.realtime, self.max_ticks = rate, False, realtime, max_ticks
        self.anchor = None
        self.last_sync = 0.0
        self.ticks = 0

    def key(self, code):                      # viewer UI thread: only set flags here
        if code == KEY_SPACE:
            self.paused = not self.paused
        elif code in KEYS_FASTER:
            self.rate = min(self.rate * 2, 64.0)
        elif code in KEYS_SLOWER:
            self.rate = max(self.rate / 2, 1 / 16)
        self.anchor = None

    def _text(self):
        f = self.farm
        h, rem = divmod(int(f.t + (self.w.d.time - f.t_phys)), 3600)
        state = "PAUSED" if self.paused else f"x{self.rate:g}"
        return (None, None, f"farm {h:02d}:{rem // 60:02d}  {state}\n{f.action}",
                "space pause   + / - speed")

    def __call__(self, w):
        self.ticks += 1
        if not self.v.is_running() or (self.max_ticks and self.ticks > self.max_ticks):
            raise Stop
        while self.paused and self.v.is_running():
            self.v.set_texts(self._text())
            self.v.sync()
            time.sleep(1 / 30)
        now = time.perf_counter()
        if self.realtime:
            if self.anchor is None or now - (self.anchor[0] + (w.d.time - self.anchor[1]) / self.rate) > 0.25:
                self.anchor = (now, w.d.time)       # (re)start pacing: after a pause, a farm-clock jump or a slow plan
            ahead = self.anchor[0] + (w.d.time - self.anchor[1]) / self.rate - now
            if ahead > 0:
                time.sleep(ahead)
        if not self.realtime or time.perf_counter() - self.last_sync >= 1 / 60:
            self.v.set_texts(self._text())
            self.v.sync()
            self.last_sync = time.perf_counter()


def build(a):
    STATIC_XML.parent.mkdir(parents=True, exist_ok=True)
    w = World(scene.build(STATIC_XML))                 # the model the run uses; its MJCF is written for reference
    mujoco.mj_saveModel(w.m, str(STATIC_MJB), None)    # parts parked as World() leaves them: the static scene
    D.VERBOSE = False
    ik = IK(w)
    w.reset(rail_x=C.STATION_X)
    w.d.qpos[w.arm_q] = TUCK
    w.d.ctrl[w.act_arm] = TUCK
    mujoco.mj_forward(w.m, w.d)
    mask = None
    if pathlib.Path(a.mask).exists():
        from plan.bedmap import load_mask
        mask = load_mask(a.mask, mirror=True)
    import plan.unload as U
    U.CAMERA_NOISE, U.CAMERA_SEED = a.realism, a.seed
    ex = Executor(w, realism=None if a.realism == "ideal" else a.realism, ik=ik,
                  rng=np.random.default_rng((a.seed, 7)))
    farm = Farm(w, ik, ex, seed=a.seed, order_scale=a.scale, mask=mask)
    for _ in range(round(1.4 * C.N_PRINTERS)):        # same backlog as scripts/run_farm.py
        farm._new_order(0.0)
    return w, ex, farm


def run_unload(farm, i):
    """One printer's plate, printed and cooled at once, then the farm's own unload (drive, door, picks, door)."""
    farm._advance_timers()
    p = farm.printers[i]
    if p.state == "printing":
        farm._print_done(p, farm.t)
    if p.state not in ("cooling", "waiting"):
        raise SystemExit(f"P{i + 1} got no plate (state {p.state})")
    p.t_release = farm.t
    p.state = "waiting"
    farm.unload(p)
    return farm.summary()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--plates", type=int, default=None)
    ap.add_argument("--unload", type=int, default=None, metavar="P", help="only printer P's unload (1-based)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--realism", default="ideal", choices=["ideal", "realistic", "harsh"])
    ap.add_argument("--mask", default="runs/bedmap.json")
    ap.add_argument("--rate", type=float, default=1.0, help="sim seconds per wall second (+ and - change it)")
    ap.add_argument("--follow", action="store_true", help="camera tracks the carriage (Esc: free camera)")
    ap.add_argument("--dry-run", type=int, default=0, metavar="TICKS",
                    help="no window: run TICKS control ticks through the same loop, unpaced, then stop")
    ap.add_argument("--max-ticks", type=int, default=0,
                    help="with the window: stop and close it after N control ticks (smoke test)")
    a = ap.parse_args()
    warnings.filterwarnings("ignore", message="All-NaN slice")     # lead-time percentiles before any order ships
    if a.unload is not None and not 1 <= a.unload <= C.N_PRINTERS:
        ap.error(f"--unload takes 1..{C.N_PRINTERS}")
    if not a.dry_run and sys.platform == "darwin" and mujoco.viewer._MJPYTHON is None:
        raise SystemExit("macOS: the passive viewer needs mjpython: "
                         "PYTHONPATH=. .venv/bin/mjpython scripts/view_farm.py (or --dry-run N: no window)")

    w, ex, farm = build(a)
    print(f"static scene: .venv/bin/python -m mujoco.viewer --mjcf={STATIC_MJB}")
    pacer = Pacer(None, w, farm, rate=a.rate, realtime=not a.dry_run, max_ticks=a.max_ticks)
    if a.dry_run:
        viewer = DryViewer(a.dry_run)
    else:
        viewer = mujoco.viewer.launch_passive(w.m, w.d, key_callback=pacer.key)
    pacer.v = viewer
    with viewer.lock():
        viewer.opt.geomgroup[:] = 0
        viewer.opt.geomgroup[list(DEFAULT_GROUPS)] = 1    # as the videos: lab detail on, collision capsules off
        if a.follow:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = mujoco.mj_name2id(w.m, mujoco.mjtObj.mjOBJ_BODY, "carriage")
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.4, 125.0, -30.0
        else:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            viewer.cam.lookat[:] = [C.STATION_X / 2, 0.0, 0.8]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 4.5, 120.0, -30.0
    ex.rec = pacer
    res = None
    t0 = time.time()
    try:
        if a.unload is not None:
            res = run_unload(farm, a.unload - 1)
        else:
            res = farm.run(a.hours, max_plates=a.plates)
    except Stop:
        pass
    print(f"{'dry run' if a.dry_run else 'viewer'}: {pacer.ticks} control ticks, sim {w.d.time:.1f} s, "
          f"farm {farm.t / 3600:.2f} h, wall {time.time() - t0:.1f} s, last action: {farm.action}")
    if res is not None:
        print(f"done: {res['stats']}")
    if not a.dry_run and a.max_ticks:
        viewer.close()
    elif not a.dry_run and viewer.is_running():
        print("run finished: the window stays open until you close it")
        while viewer.is_running():
            viewer.sync()
            time.sleep(1 / 30)


if __name__ == "__main__":
    main()
