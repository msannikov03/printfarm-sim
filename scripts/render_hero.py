"""Hero still of the default cell (two facing rows): plates on several beds, one door open, the arm reaching in to a
planned grasp, status lights by printer state. 4K render downsampled to 1080p.

python scripts/render_hero.py [--out docs/media/hero.png] [--variants]
"""
import argparse
import pathlib

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

from farm import catalog as C, lab
from farm.render import Cam
from farm.world import World
from jobs.plates import make_plate
from plan.bedmap import load_mask
from plan.ik import IK
from plan.unload import mirror_plate, plan_grasp, spawn_plate

SKUS = ["cube", "gear", "clip", "standoff", "block", "knob", "lid", "spacer", "bracket"]


def scene(P=4, seed=7):
    w = World()
    ik = IK(w)
    xc, s = C.printer_frame(P)
    w.reset(rail_x=xc + 0.3)
    rng = np.random.default_rng(seed)
    mask = load_mask("runs/bedmap.json", mirror=True) if pathlib.Path("runs/bedmap.json").exists() else None
    placed = {}
    for i in range(C.N_PRINTERS):
        if i % 3 == 1 and i != P:
            continue                                       # some printers are still printing (empty-looking bed)
        k = 3 + int(rng.integers(0, 4))
        parts = [{"name": f"h{i}_{str(sku)}_{j}", "order_id": f"h{i}", "sku": str(sku)}
                 for j, sku in enumerate(rng.choice(SKUS, k))]
        plate, _ = make_plate(parts, i, f"H{i:02d}", rng, reach_mask=mask)
        placed[i] = spawn_plate(w, mirror_plate(plate), rng, stuck=False)
    w.d.qpos[w.door_q[P]] = C.DOOR_OPEN_RAD
    mujoco.mj_forward(w.m, w.d)
    w.step(200)
    ik.sync(w.d)
    body = sorted(placed[P].values(), key=lambda b: w.part_pose(b)[0][1])[len(placed[P]) // 2]
    g = plan_grasp(ik, w, P, body, rng=rng)
    if g is not None:
        w.d.qpos[w.arm_q], w.d.qpos[w.rail_q] = g["q"], g["x"]
        w.d.qpos[w.finger_q], w.d.qpos[w.finger_rq] = g["width"] / 2 + 0.006, -(g["width"] / 2 + 0.006)
    mujoco.mj_forward(w.m, w.d)
    for i in range(C.N_PRINTERS):
        state = "unloading" if i == P else ("printing" if i % 3 == 1 else ("waiting" if i % 3 == 0 else "cooling"))
        lab.set_status(w.m, i, state)
    return w, P


def render(w, P, out, look, size=(3840, 2160)):
    w.m.vis.quality.shadowsize = 8192
    cam = Cam(w.m, *size, groups=(0, 1, 2, 4))
    img = cam.look(*look).rgb(w.d)
    img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
    imageio.imwrite(out, img)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/media/hero.png")
    ap.add_argument("--variants", action="store_true", help="also write a few alternative angles next to --out")
    a = ap.parse_args()
    w, P = scene()
    xc, s = C.printer_frame(P)
    views = {"": ([xc + 0.15, 0.10, 0.95], 1.9, 22, -24)}         # from inside the aisle, both rows either side
    if a.variants:
        views.update({"_axis": ([xc + 0.15, 0.08, 0.95], 2.1, 0, -20),
                      "_right": ([xc + 0.15, 0.05, 0.95], 1.9, -22, -24),
                      "_wide": ([xc + 0.6, 0.0, 0.9], 3.2, 8, -16)})
    out = pathlib.Path(a.out)
    for tag, look in views.items():
        render(w, P, out.with_name(out.stem + tag + out.suffix), look)
