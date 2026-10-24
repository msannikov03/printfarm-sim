"""Perception error vs ground truth on random noisy plates; missing-part and debris detection.
  PYTHONPATH=. python scripts/test_perception.py [n_seeds] [noise preset: ideal | realistic | harsh]
Monte Carlo over presets, corrections and camera faults: scripts/eval_perception.py."""
import math, sys, numpy as np, mujoco
from farm import catalog as C
from farm.world import World
from jobs.plates import make_plate
from plan.unload import mirror_plate, spawn_plate
from perceive.verify import BedCamera, estimate, bed_empty
P = 4
w = World(); xc, s = C.printer_frame(P)
NOISE = sys.argv[2] if len(sys.argv) > 2 else "ideal"
cam = BedCamera(w, P, noise=NOISE, rng=np.random.default_rng(0))
errs, yerrs, found, total = [], [], 0, 0
for seed in range(int(sys.argv[1]) if len(sys.argv) > 1 else 8):
    rng = np.random.default_rng(seed)
    w.reset(rail_x=xc + 0.4)
    parts = [{"name": f"o{k}_{sku}_{k}", "order_id": f"o{k}", "sku": str(sku)} for k, sku in enumerate(rng.choice(list(C.SMALL_SKUS), 6))]
    plate = mirror_plate(make_plate(parts, P, f"PL{seed}", rng)[0])
    placed = spawn_plate(w, plate, rng, pos_sigma_mm=3.0, yaw_sigma_deg=5.0, stuck=True)
    mujoco.mj_forward(w.m, w.d)
    pts = cam.points_bed_mm()
    objs = [dict(o, height_mm=C.CATALOG[o["sku"]].height, footprint_mm=C.CATALOG[o["sku"]].footprint) for o in plate["objects"]]
    est, unknown = estimate(pts, objs)
    for o in objs:
        b = placed[o["name"]]; p, q = w.part_pose(b)
        tx = (p[0] - xc) * 1000 + 128; ty = (p[1] - C.AISLE_HALF - C.BED_FRONT_GAP) * 1000
        tyaw = math.degrees(2 * math.atan2(q[3], q[0]))
        e = est[o["name"]]; total += 1
        if e["found"]:
            found += 1
            errs.append(math.hypot(e["center_mm"][0] - tx, e["center_mm"][1] - ty))
            k = C.CATALOG[o["sku"]]
            if k.kind == "box" and abs(k.dims[0] - k.dims[1]) > 1:
                d = (e["yaw_deg"] - tyaw + 90) % 180 - 90
                yerrs.append(abs(d))
        prior_err = math.hypot(o["center_mm"][0] - tx, o["center_mm"][1] - ty)
        print(f"  {o['sku']:9s} prior err {prior_err:4.1f} mm -> est err {errs[-1] if e['found'] else float('nan'):4.1f} mm"
              f"  (1-sigma {e['sigma_mm']:.1f} mm, {e['quality']})  top pts {e['n_top']}")
    print(f"seed {seed}: unknown raised points {unknown}")
print(f"FOUND {found}/{total}; centre error mean {np.mean(errs):.2f} mm, p95 {np.percentile(errs, 95):.2f} mm; yaw error (elongated boxes) mean {np.mean(yerrs) if yerrs else float('nan'):.2f} deg")
# missing part + debris
w.reset(rail_x=xc + 0.4); mujoco.mj_forward(w.m, w.d)
print("empty bed -> bed_empty:", bed_empty(cam.points_bed_mm()))
b = w.free_part("cube"); w.spawn(b, C.bed_mm_to_world(P, 128, 128), 0.3, stuck=True); mujoco.mj_forward(w.m, w.d)
pts = cam.points_bed_mm()
print("one leftover cube -> bed_empty:", bed_empty(pts), "| as unknown vs empty prior:", estimate(pts, [])[1], "raised pts")
cam.close()
