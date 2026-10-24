"""Targeted test of the clearing fallback (plan/sweep.py): one part or debris lump alone on a bed, nothing picked
first, then the same loop as the end of plan.unload.unload_parts (scan, front-most blob, clear it, re-scan, at most
RAKE_ROUNDS times). Reports where the thing ended (simulator truth, reporting only) and the robot's own final scan.

python scripts/test_rake.py cube 150 80 [--yaw DEG] [--printer 4] [--tip] [--realism realistic] [--video] [--verbose]
python scripts/test_rake.py --suite debris|long [--printer 4,14] [--realism R] [--only SKUS] [--no-blob-pick] [--out F]
  debris: the debris SKUs (cube, block, knob, gear) across the deep and side areas of the bed (the two-row cell's
          hard cases); long: bar / plate / housing lying into the printer (yaw 90) and tipped, never picked first.
Positions are bed mm (x across from the left edge as seen from the door, y deep from the front edge); yaw in the bed
frame (0 = long side along the door, 90 = into the printer).
"""
import argparse
import json
import math
import pathlib
import time

import numpy as np

DEBRIS_CASES = [(sku, x, y, yaw) for sku, yaw in (("gear", 20), ("knob", 0), ("cube", 30), ("block", 60))
                for x, y in ((56, 98), (128, 150), (200, 120), (40, 200), (128, 225), (225, 60))]
LONG_CASES = [(sku, x, y, 90, tip) for sku in ("bar", "plate", "housing")
              for x, y in ((70, 120), (128, 130), (190, 110))
              for tip in (False, True)]


def setup_case(sku, bx, by, yaw_deg=0.0, printer=4, tip=False, realism="realistic", seed=0, video=None, verbose=False):
    """World with the door open, the arm tucked and one part / lump on the bed. Returns (w, ik, ex, cam, body, rec)."""
    import mujoco
    from farm import catalog as C
    from farm.world import World
    from plan.ik import IK
    from plan.motion import Executor
    from plan.door import TUCK
    import plan.door as D
    import plan.unload as U
    D.VERBOSE = verbose
    U.CAMERA_NOISE = realism
    U._CAMS.clear()                                   # the camera cache is keyed by id(world)
    w = World()
    ik = IK(w)
    xc, s = C.printer_frame(printer)
    w.reset(rail_x=xc + 0.4)
    w.d.qpos[w.arm_q] = TUCK
    w.d.ctrl[w.act_arm] = TUCK
    w.d.qpos[w.door_q[printer]] = C.DOOR_OPEN_RAD
    mujoco.mj_forward(w.m, w.d)
    b = w.free_part(sku)
    w.spawn(b, C.bed_mm_to_world(printer, bx, by), math.radians(yaw_deg) + (0 if s > 0 else math.pi), stuck=False)
    if tip:                                           # onto its side, as scripts/eval_unload.py inject_faults
        a = w.part_q[b]
        k = C.CATALOG[sku]
        t = np.zeros(4)
        roll = np.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0])
        mujoco.mju_mulQuat(t, w.d.qpos[a + 3:a + 7].copy(), roll)
        w.d.qpos[a + 3:a + 7] = t
        w.d.qpos[a + 2] += max(max(k.footprint), k.height) / 2000 + 0.003
        mujoco.mj_forward(w.m, w.d)
    w.step(300)
    rec = None
    if video:
        from farm.recorder import Recorder
        rec = Recorder(w.m, video, view=lambda w: ([xc, s * 0.2, 0.85], 0.9, 90 + 50 * s, -30))
    ex = Executor(w, rec, realism=realism, ik=ik, rng=np.random.default_rng(10_000 + seed))
    cam = U.bed_camera(w, printer, rng=np.random.default_rng(20_000 + seed))
    return w, ik, ex, cam, b, rec


def run_case(sku, bx, by, yaw_deg=0.0, printer=4, tip=False, realism="realistic", seed=0, video=None, verbose=False):
    import mujoco
    import plan.unload as U
    from plan.sweep import blobs, in_tray, rake
    from perceive.verify import bed_empty, cannot_verify
    w, ik, ex, cam, b, rec = setup_case(sku, bx, by, yaw_deg, printer, tip, realism, seed, video, verbose)
    t0, t_wall = w.d.time, time.time()
    rounds, log = 0, []
    for _ in range(U.RAKE_ROUNDS):
        mujoco.mj_forward(w.m, w.d)
        pts = U.scan(cam)
        todo = blobs(pts)
        ik.belief = U.scene_belief(w, printer, {}, {}, pts)
        if not todo:
            break
        done = False
        for bl in todo:
            ok, _ = rake(ex, ik, printer, bl, np.random.default_rng(seed))
            if ok:
                done = True
                rounds += 1
                log.append(getattr(ik, "last_clear", None))
                break
        if not done:
            break
    ex.wait(0.5)
    mujoco.mj_forward(w.m, w.d)
    pts = U.scan(cam)
    empty = bed_empty(pts)
    for _ in range(U.VERIFY_RESCANS):                 # as unload_parts: "cannot verify" -> look again
        if empty or not cannot_verify(pts):
            break
        pts = U.scan(cam)
        empty = bed_empty(pts)
    if rec:
        rec.close()
    if b not in w.in_use:
        end = "removed"
    elif in_tray(w, printer, b):
        end = "tray"
    elif U.part_in_bin(w, b, "reject"):
        end = "reject_bin"
    elif U.on_bed(w, printer, b):
        end = "on_bed"
    else:
        end = "off_bed"                               # beside the bed, in the chamber / front gap / on the floor
    p = w.part_pose(b)[0]
    return {"sku": sku, "bed_mm": [bx, by], "yaw": yaw_deg, "tip": tip, "printer": printer, "realism": realism,
            "end": end, "robot_bed_empty": bool(empty), "strokes": rounds, "how": log,
            "sim_s": round(w.d.time - t0, 1), "wall_s": round(time.time() - t_wall, 1),
            "final_world": np.round(p, 3).tolist()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sku", nargs="?")
    ap.add_argument("x", nargs="?", type=float)
    ap.add_argument("y", nargs="?", type=float)
    ap.add_argument("--yaw", type=float, default=17.0)
    ap.add_argument("--printer", default="4")
    ap.add_argument("--tip", action="store_true")
    ap.add_argument("--realism", default="realistic")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--suite", choices=["debris", "long"])
    ap.add_argument("--only", default="", help="suite: comma-separated SKUs to keep")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-blob-pick", action="store_true", help="never grasp the blob (push / rake only)")
    a = ap.parse_args()
    if a.no_blob_pick:
        import plan.sweep as SW
        SW.BLOB_PICK = False
    printers = [int(p) for p in a.printer.split(",")]
    if a.suite:
        cases = DEBRIS_CASES if a.suite == "debris" else LONG_CASES
        cases = [c if len(c) == 5 else (*c, False) for c in cases]
        if a.only:
            cases = [c for c in cases if c[0] in a.only.split(",")]
        res = []
        for P in printers:
            for k, (sku, x, y, yaw, tip) in enumerate(cases):
                r = run_case(sku, x, y, yaw, P, tip, a.realism, seed=k, verbose=a.verbose)
                res.append(r)
                print(f"P{P:2d} {sku:8s} ({x:3.0f},{y:3.0f}) yaw {yaw:3.0f}{' tipped' if tip else '       '} -> "
                      f"{r['end']:10s} robot empty {r['robot_bed_empty']!s:5s} strokes {r['strokes']} "
                      f"{r['how']} sim {r['sim_s']}s wall {r['wall_s']}s", flush=True)
        ends = [r["end"] for r in res]
        print({e: ends.count(e) for e in sorted(set(ends))}, f"left on bed {ends.count('on_bed')}/{len(res)}")
        if a.out:
            pathlib.Path(a.out).write_text(json.dumps(res, indent=1))
    else:
        r = run_case(a.sku, a.x, a.y, a.yaw, printers[0], a.tip, a.realism,
                     video=f"runs/rake_{a.sku}.mp4" if a.video else None, verbose=a.verbose)
        print(json.dumps(r, indent=1))
