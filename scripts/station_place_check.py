"""Large-part places at the station under actuation realism: robot-vs-box contacts (truth, scoring only) per place.

  PYTHONPATH=. python scripts/station_place_check.py --seeds 0 10 --realism realistic --out runs/station_place/s0.json
"""
import argparse
import collections
import json
import math
import pathlib

import mujoco
import numpy as np

from farm import catalog as C
from farm.world import World
from jobs.packaging import fit
import plan.door as D
from plan.door import TUCK
from plan.ik import IK
from plan.motion import Executor
from plan.station import Station

CASES = [("housing", "S"), ("plate", "S"), ("housing", "M"), ("bar", "M"), ("plate", "M"), ("housing", "L"),
         ("bar", "L"), ("plate", "L")]
ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs=2, default=(0, 4))
ap.add_argument("--realism", default="realistic")
ap.add_argument("--out", default="runs/station_place/check.json")
ap.add_argument("--place-clear", type=float, default=None, help="override plan.station.PLACE_CLEAR (m)")
ap.add_argument("--cases", default="", help="only these, e.g. housing:S,plate:S")
a = ap.parse_args()
if a.place_clear is not None:
    import plan.station
    plan.station.PLACE_CLEAR = a.place_clear
D.VERBOSE = False
w = World()
w.hooks = []
box_geoms = {}
rows = []
for seed in range(*a.seeds):
    for k, (sku, size) in enumerate(CASES):
        if a.cases and f"{sku}:{size}" not in a.cases.split(","):
            continue
        rng = np.random.default_rng((seed, k))
        w.reset(rail_x=C.RAIL_X_MAX)
        w.d.qpos[w.arm_q] = TUCK
        w.d.ctrl[w.act_arm] = TUCK
        mujoco.mj_forward(w.m, w.d)
        ik = IK(w)
        ex = Executor(w, realism=None if a.realism == "ideal" else a.realism, ik=ik, rng=np.random.default_rng((seed, k, 7)))
        st = Station(w, rng=np.random.default_rng((seed, k, 11)))
        slot = ("L0", "L1")[int(rng.integers(2))]
        sx, sy = C.LARGE_SLOTS[slot]
        b = w.free_part(sku)
        w.spawn(b, (float(w.d.qpos[w.rail_q]) + sx + rng.uniform(-0.004, 0.004), sy + rng.uniform(-0.004, 0.004),
                    C.DECK_Z + 0.004), float(rng.uniform(-0.05, 0.05)), stuck=False)
        ex.wait(0.5)
        box = st.request_box(ex, size, "place")
        bg = {g for g in range(w.m.ngeom) if w.m.geom_bodyid[g] == w.m.body(box).id}
        hits = collections.Counter()
        worst = [0.0]
        true_clear = [None]                             # (m, pair): truth, along the place line and at release
        tick = [0]
        arm = sorted(w.arm_geoms)
        plans = []
        plan_place = st.plan_place
        st.plan_place = lambda *a_, **k_: plans.append(plan_place(*a_, **k_)) or plans[-1]

        mm = w.m
        pairs = [(g, h) for g in arm for h in bg if (mm.geom_contype[g] & mm.geom_conaffinity[h])
                 or (mm.geom_contype[h] & mm.geom_conaffinity[g])]          # pairs that can touch (not the lid)

        def clearance(m, d, names=False):
            """Smallest arm-to-box distance (m) over the pairs that can touch, capped at 5 cm (+ the pair's bodies)."""
            dist, g, h = min((mujoco.mj_geomDistance(m, d, g, h, 0.05, None), g, h) for g, h in pairs)
            return (dist, f"{m.body(m.geom_bodyid[g]).name}/{m.geom(g).name or g}-{m.geom(h).name or h}") if names \
                else dist

        def hook(world):
            d = world.d
            tick[0] += 1
            if ex.log and ex.log[-1][0] in ("place", "release") and tick[0] % 5 == 0:   # down the line, released
                c = clearance(world.m, d, names=True)
                true_clear[0] = c if true_clear[0] is None or c[0] < true_clear[0][0] else true_clear[0]
            for c in d.contact[:d.ncon]:
                r1, r2 = c.geom1 in world.robot_geoms, c.geom2 in world.robot_geoms
                if r1 != r2 and (c.geom2 if r1 else c.geom1) in bg and c.dist < 0:
                    rg = c.geom1 if r1 else c.geom2
                    hits[(world.m.body(world.m.geom_bodyid[rg]).name, ex.log[-1][0] if ex.log else "")] += 1
                    worst[0] = min(worst[0], float(c.dist))
                    break

        w.hooks.append(hook)
        try:
            ok, outcome, sec = st.place_large(ex, ik, slot, box, b, sku, layout=fit(size, [], [sku]).layout[0])
        finally:
            w.hooks.remove(hook)
        plan_clear = None                    # the plan's arm-to-box clearance down the place line (the robot's model)
        plan = next((q for q in reversed(plans) if q is not None), None)
        if plan is not None:
            dd = mujoco.MjData(ik.m)
            dd.qpos[:] = w.d.qpos                       # (the box is still on the place lift)
            dd.qpos[w.rail_q] = plan["x"]
            dd.qpos[w.finger_q] = min(0.037, plan["width"] / 2 + 0.004)
            dd.qpos[w.finger_rq] = -dd.qpos[w.finger_q]
            for q in plan["lower"]:
                dd.qpos[w.arm_q] = q
                mujoco.mj_kinematics(ik.m, dd)
                c = clearance(ik.m, dd, names=True)
                plan_clear = c if plan_clear is None or c[0] < plan_clear[0] else plan_clear
        rows.append({"seed": seed, "sku": sku, "box": size, "slot": slot, "ok": ok, "outcome": outcome,
                     "s": round(sec, 2), "contact_ticks": sum(hits.values()), "worst_mm": round(-1000 * worst[0], 3),
                     "plan_clear_mm": None if plan_clear is None else round(1000 * plan_clear[0], 2),
                     "plan_pair": None if plan_clear is None else plan_clear[1],
                     "plan_gap_mm": None if plan is None else round(1000 * (plan["part_pose"][0][2]
                                                                            - st.box_floor(box)), 1),
                     "plan_tilt_deg": None if plan is None else round(math.degrees(plan["tilt"])),
                     "true_clear_mm": None if true_clear[0] is None else round(1000 * true_clear[0][0], 2),
                     "true_pair": None if true_clear[0] is None else true_clear[0][1],
                     "where": {f"{p}@{m}": n for (p, m), n in hits.items()}, "why": dict(st.why)})
        print(json.dumps(rows[-1]), flush=True)
        w.park(b)
        w.box_park(box)
        st.cam.close()
out = {"realism": a.realism, "places": len(rows), "placed": sum(r["ok"] for r in rows),
       "with_contact": sum(r["contact_ticks"] > 0 for r in rows), "rows": rows}
pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "rows"}))
