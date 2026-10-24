"""Run the whole cell: python scripts/run_farm.py --hours 6 --plates 12 --video --speed 8 --seed 0

Station only (two rows), without printing:
  --station-demo   two orders (one package with a large part; one order shipped as two packages) are put on the
                   caddy as if just unloaded, then the robot ships them (video: --video, real-time-ish --speed 1.5)
  --station-bench N  N rounds of: fill the caddy with random orders (all SKUs), ship every package; writes the
                   measured station step times (runs/station_times.json by default, --times)
"""
import argparse
import json
import math
import pathlib
import time

import mujoco
import numpy as np

from farm import catalog as C
from farm.hud import make_overlay
from farm.recorder import Recorder
from farm.world import World
from plan import door as D
from plan.door import TUCK
from plan.ik import IK
from plan.motion import Executor
from sim.farm import Farm

ap = argparse.ArgumentParser()
ap.add_argument("--hours", type=float, default=6.0)
ap.add_argument("--plates", type=int, default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--scale", type=float, default=2.0, help="order rate multiplier")
ap.add_argument("--video", action="store_true")
ap.add_argument("--camera", default="director", choices=["director", "fixed"],
                help="director: automatic shots by context (farm/director.py); fixed: one chase view along the rail")
ap.add_argument("--no-pip", action="store_true", help="director: no chamber-camera inset during picks")
ap.add_argument("--speed", type=float, default=8.0)
ap.add_argument("--mask", default="runs/bedmap.json")
ap.add_argument("--out", default="runs/farm")
ap.add_argument("--realism", default="ideal", choices=["ideal", "realistic", "harsh"],
                help="sensor + actuation realism: chamber-camera noise (perceive/noise.py) and servo/encoder/calibration "
                     "errors (farm/sensors.py)")
ap.add_argument("--station-demo", action="store_true", help="two rows: station only, two demo orders")
ap.add_argument("--station-bench", type=int, default=0, help="two rows: station only, N rounds of random orders")
ap.add_argument("--times", default="runs/station_times.json", help="--station-bench: where to write the step times")
a = ap.parse_args()

out = pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=True)
D.VERBOSE = False
w = World()
ik = IK(w)
station_only = a.station_demo or a.station_bench
x0 = C.printer_frame(C.N_PER_ROW - 2)[0] + 0.35 if station_only else C.STATION_X
w.reset(rail_x=x0)
w.d.qpos[w.arm_q] = TUCK
w.d.ctrl[w.act_arm] = TUCK
mujoco.mj_forward(w.m, w.d)
mask = None
if pathlib.Path(a.mask).exists():
    from plan.bedmap import load_mask
    mask = load_mask(a.mask, mirror=True)
import plan.unload as U
U.CAMERA_NOISE, U.CAMERA_SEED = a.realism, a.seed
ex = Executor(w, realism=None if a.realism == "ideal" else a.realism, ik=ik, rng=np.random.default_rng((a.seed, 7)))
farm = Farm(w, ik, ex, seed=a.seed, order_scale=a.scale, mask=mask, station_noise=a.realism)


def fill_caddy(orders):
    """Put the orders' parts on the caddy as if the robot had just unloaded them (station-only runs): each part
    goes where the allocator says, spawned in its tote / tray slot. Orders that do not fit wait."""
    rng = farm.rng
    for o in orders:
        for part in o.parts:
            slot = farm.alloc.slot_for(part)
            if slot is None:
                continue
            sku = part.split("_")[1]
            b = w.free_part(sku)
            xr = float(w.d.qpos[w.rail_q])
            if slot in C.LARGE_SLOTS:
                dx, dy = C.LARGE_SLOTS[slot]
                p = (xr + dx + rng.uniform(-0.004, 0.004), dy + rng.uniform(-0.004, 0.004), C.DECK_Z + 0.004)
                yaw = float(rng.uniform(-0.05, 0.05))
            else:
                dx, dy = C.CADDY_SLOTS[slot]
                n = len(farm.alloc.contents[slot])
                p = (xr + dx + rng.uniform(-0.025, 0.025), dy + rng.uniform(-0.025, 0.025), C.DECK_Z + 0.006 + 0.03 * n)
                yaw = float(rng.uniform(0, math.pi))
            w.spawn(b, p, yaw, stuck=False)
            farm._caddy_bodies[part] = b
            farm.alloc.placed(part, slot)
        farm.queue = [q for q in farm.queue if q["order_id"] != o.order_id]
    ex.wait(1.0)


view = None
if a.video:
    if a.camera == "director":     # render-only: the farm runs the same with any camera
        from farm.director import Director
        view = Director(w, farm, fps=30, pip=not a.no_pip, wide_every_s=0.0 if station_only else 75.0)
    elif C.N_ROWS == 2:      # both rows in frame: higher, looking down the aisle at an angle
        view = lambda w: ([float(w.d.qpos[w.rail_q]) + 0.3, 0.0, 0.8], 3.4, 90 + 38, -34)
    else:
        view = lambda w: ([float(w.d.qpos[w.rail_q]) + 0.1, 0.05, 0.85], 2.6, 90 + 50, -24)
    name = "station_demo" if a.station_demo else f"farm_s{a.seed}"
    ex.rec = Recorder(w.m, str(out / f"{name}.mp4"), fps=30, width=1280, height=720, view=view,
                      overlay=make_overlay(farm), speed=a.speed)
t0 = time.time()
if station_only:
    if a.station_demo:
        orders = [farm._new_order(0.0, lines=(("bar", 1), ("cube", 2), ("gear", 1))),
                  farm._new_order(0.0, lines=(("knob", 5), ("spacer", 4)))]
        rounds = [orders]
    else:
        rounds = [None] * a.station_bench
    for orders in rounds:
        if orders is None:                             # bench: random orders (all SKUs) while the caddy has room
            from jobs.orders import sample_lines
            from jobs.packaging import plan_packages
            orders, tries = [], 0
            while len(orders) < 4 and tries < 30:
                tries += 1
                lines = sample_lines(farm.rng, tuple(C.CATALOG))
                parts = [f"probe_{s}_{k}" for s, q in lines for k in range(q)]
                pk = plan_packages("probe", parts)
                if sum(1 for p in pk if p.small) > len(farm.alloc.free_slots()) or \
                        sum(len(p.large) for p in pk) > len(farm.alloc.free_large()):
                    continue
                o = farm._new_order(0.0, lines=lines)
                fill_caddy([o])
                orders.append(o)
        else:
            fill_caddy(orders)
        while farm.alloc.handoff_ready():
            farm.station_visit()
    farm.action = "station: boxes ride to the lanes"
    if a.station_demo:                 # the video ends once the last box is sealed and leaving for its lane
        t_seal = w.d.time
        while farm.station.stats["boxes_sealed"] < farm.stats["packages_shipped"] and w.d.time - t_seal < 30.0:
            ex.wait(0.2)
        ex.wait(3.0)
        farm.sync_clock()
    else:
        farm.finish()
    res = farm.summary()
    if a.station_bench:
        farm.station.save_times(a.times)
else:
    res = farm.run(a.hours, max_plates=a.plates)
    farm.finish()                      # boxes still on the conveyor reach their lanes
    res = farm.summary()
res["wall_s"] = time.time() - t0
if farm.station is not None:
    farm.station.save_times(out / f"station_times_s{a.seed}.json")
if ex.rec:
    ex.rec.close()
(out / f"summary_s{a.seed}.json").write_text(json.dumps(res, indent=1, default=float))
(out / f"parts_s{a.seed}.json").write_text(json.dumps(farm.part_log, indent=1, default=float))
(out / f"events_s{a.seed}.json").write_text(json.dumps(farm.events, indent=1, default=str))
if hasattr(view, "cuts"):          # shot list (video time s, kind, key, label): handy for cutting highlights
    (out / f"camera_s{a.seed}.json").write_text(json.dumps(view.cuts, indent=1, default=str))
print(json.dumps(res, indent=1, default=float))
