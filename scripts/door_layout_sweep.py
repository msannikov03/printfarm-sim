"""Layout sweep for the single everyday door handle: rail distance x arm base height x handle stand-off x handle
height. Per layout: can the door be opened and closed (one swing or a regrasp of the same handle), and how much
of the bed can be grasped with the door open.

python scripts/door_layout_sweep.py --procs 10   -> runs/door_layout_sweep.jsonl
"""
import argparse
import itertools
import json
import multiprocessing as mp
import pathlib
import time

import numpy as np

OUT = pathlib.Path("runs/door_layout_sweep.jsonl")
P = 4
BED_PTS = [(x, y) for y in (40, 120, 200, 236) for x in (40, 128, 216)]


def evaluate(cfg):
    aisle, base_z, off, zl = cfg
    from farm import catalog as C
    C.AISLE_HALF, C.BASE_Z, C.DECK_Z, C.PIN_OFFSET, C.PIN_ZL = aisle, base_z, base_z, off, zl
    import mujoco
    from farm.world import World
    from plan.door import TUCK, plan_door
    from plan.ik import IK
    from plan.unload import plan_grasp
    t0 = time.time()
    w = World()
    ik = IK(w)
    xc, _ = C.printer_frame(P)
    w.reset(rail_x=xc + 0.4)
    w.d.qpos[w.arm_q] = TUCK
    mujoco.mj_forward(w.m, w.d)
    ik.sync(w.d)
    op, note_o = plan_door(ik, w, P, opening=True)
    w.d.qpos[w.door_q[P]] = C.DOOR_OPEN_RAD
    mujoco.mj_forward(w.m, w.d)
    ik.sync(w.d)
    cl, note_c = plan_door(ik, w, P, opening=False)
    reach = 0
    for xy in BED_PTS:
        ok = True
        for sku in ("cube", "clip"):
            b = w.free_part(sku)
            w.spawn(b, C.bed_mm_to_world(P, *xy), 0.0, stuck=False)
            ik.sync(w.d)
            ok &= plan_grasp(ik, w, P, b) is not None
            w.park(b)
        reach += ok
    return {"aisle": aisle, "base_z": base_z, "handle_offset": off, "handle_z": zl,
            "open": None if op is None else len(op), "close": None if cl is None else len(cl),
            "open_rail": None if op is None else [round(float(op[0]["xs"][0] - xc), 2), round(float(op[-1]["xs"][-1] - xc), 2)],
            "reach": reach / len(BED_PTS), "wall_s": round(time.time() - t0, 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=10)
    a = ap.parse_args()
    cfgs = list(itertools.product((0.24, 0.28, 0.34), (0.50, 0.56), (0.11, 0.14), (0.20, 0.30)))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("")
    with mp.Pool(a.procs, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(evaluate, cfgs):
            with OUT.open("a") as f:
                f.write(json.dumps(r) + "\n")
            print(r, flush=True)
