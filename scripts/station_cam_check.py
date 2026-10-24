"""Station empty-tote check under a camera / actuation realism preset, scored against ground truth (truth is used
here only to score the camera): false alarms on empty totes (fresh, and after a trap-door cycle) and detection of
leftovers - one part lying anywhere in the tote, against a wall, on the flap's free edge, or wedged (held fixed)
against a wall a little above the floor - plus real dumps of a few parts (trap door cycled, whatever stays is a
leftover).

  PYTHONPATH=. python scripts/station_cam_check.py --seeds 0 20 --realism realistic --out runs/station_cam/s0.json
"""
import argparse
import json
import math
import pathlib

import mujoco
import numpy as np

from farm import catalog as C, scene
from farm.world import World
import plan.door as D
from plan.door import TUCK
from plan.ik import IK
from plan.motion import Executor
from plan.station import SHAKES, Station

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs=2, default=(0, 10))
ap.add_argument("--realism", default="realistic")
ap.add_argument("--out", default="runs/station_cam/check.json")
a = ap.parse_args()
D.VERBOSE = False
w = World()
L = scene.station_layout()
totes = [s for s in L["dock"] if s != "waste"]
rows = []


def in_tote(b, x, dx, dy):
    p = w.part_pose(b)[0]
    return abs(p[0] - x - dx) < scene.TOTE_IN and abs(p[1] - dy) < scene.TOTE_IN and p[2] > C.DECK_Z - 0.005


for seed in range(*a.seeds):
    rng = np.random.default_rng(seed)
    w.reset(rail_x=C.RAIL_X_MAX)
    w.d.qpos[w.arm_q] = TUCK
    w.d.ctrl[w.act_arm] = TUCK
    mujoco.mj_forward(w.m, w.d)
    ik = IK(w)
    ex = Executor(w, realism=None if a.realism == "ideal" else a.realism, ik=ik, rng=np.random.default_rng((seed, 7)))
    st = Station(w, noise=a.realism, rng=np.random.default_rng((seed, 11)))
    tote = totes[seed % len(totes)]
    st.dock(ex, ik, L["dock"][tote])
    ex.wait(0.3)
    dx, dy = C.CADDY_SLOTS[tote]
    x = float(w.d.qpos[w.rail_q])                       # truth: where the tote really is (set-up and scoring only)

    def check(case, bodies, **kw):
        e, n, meta = st.cam.tote_empty(tote, ex.x_meas())
        left = [b for b in bodies if in_tote(b, x, dx, dy)]
        ref = st.cam.ref[tote]
        rows.append({"seed": seed, "tote": tote, "case": case, "said_empty": e, "n": n, "truth_left": len(left),
                     "scans": meta["scans"], "quality": meta.get("quality"), "mask_px": int(ref["mask"].sum()),
                     "sigma_mm": round(1000 * ref["sigma"], 3), "thr_mm": round(1000 * ref["thr"], 2), **kw})

    for _ in range(2):                                  # empty tote: every "not empty" is a false alarm
        check("empty", [])
    st.cycle_flap(ex, tote, SHAKES[0])
    ex.wait(0.2)
    check("empty_cycled", [])
    h0 = scene.TOTE_IN - 0.003
    for case in ("anywhere", "wall", "flap_edge", "wedged"):
        sku = str(rng.choice(list(C.SMALL_SKUS)))
        b = w.free_part(sku)
        h = h0 - 0.5 * max(C.CATALOG[sku].footprint) / 1000
        side = int(rng.integers(4))
        wall = [(h, rng.uniform(-h, h)), (-h, rng.uniform(-h, h)), (rng.uniform(-h, h), h), (rng.uniform(-h, h), -h)]
        px, py = {"anywhere": tuple(rng.uniform(-h, h, 2)), "flap_edge": (rng.uniform(-h, h), -h)}.get(case, wall[side])
        yaw = 0.0 if case == "wedged" else float(rng.uniform(0, math.pi))
        lift = 0.006 if case == "wedged" else 0.002
        w.spawn(b, (x + dx + px, dy + py, C.DECK_Z + C.FLAP_T + lift), yaw, stuck=case == "wedged")
        ex.wait(0.6)
        check(case, [b], sku=sku)
        w.set_stuck(b, False)
        w.park(b)
        ex.wait(0.2)
    bodies = []                                         # a real dump: a few parts, trap door cycled
    for k, sku in enumerate(rng.choice(list(C.SMALL_SKUS), int(rng.integers(2, 6)))):
        b = w.free_part(str(sku))
        w.spawn(b, (x + dx + rng.uniform(-0.03, 0.03), dy + rng.uniform(-0.03, 0.03), C.DECK_Z + C.FLAP_T + 0.035 * k),
                float(rng.uniform(0, 6.28)), stuck=False)
        bodies.append(b)
    ex.wait(1.0)
    st.cycle_flap(ex, tote, SHAKES[0])
    ex.wait(0.2)
    check("dump", bodies, parts=len(bodies))
    for b in bodies:
        w.park(b)
    st.cam.close()

emp = [r for r in rows if r["truth_left"] == 0]
full = [r for r in rows if r["truth_left"] > 0]
out = {"realism": a.realism,
       "empty_checks": len(emp), "false_alarms": sum(r["said_empty"] is False for r in emp),
       "empty_no_answer": sum(r["said_empty"] is None for r in emp),
       "leftover_checks": len(full), "missed": sum(r["said_empty"] is True for r in full),
       "leftover_no_answer": sum(r["said_empty"] is None for r in full),
       "scans_per_check": float(np.mean([len(r["scans"]) for r in rows])), "rows": rows}
pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(a.out).write_text(json.dumps(out, indent=1, default=str))
print(json.dumps({k: v for k, v in out.items() if k != "rows"}))
