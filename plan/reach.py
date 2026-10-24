"""M2: reachability / capability map of one printer's bed, and the layout design sweep.

For every bed cell we search grasp pitch (top-down -> tilted into the printer), jaw orientation and carriage
offset along the rail, run multi-start IK, reject collisions (door open), and keep the best-scoring option:
score = manipulability x joint-limit margin. The same machinery picks the carriage position online.

CLI: python -m plan.reach --sweep        (layout sweep, writes runs/reach/sweep.json)
     python -m plan.reach --map          (map for the current catalog layout, writes runs/reach/map.json + png)
"""
import argparse
import itertools
import json
import multiprocessing as mp
import pathlib

import numpy as np

from farm import catalog as C

P = 2                                   # reference printer (row A); row B is the mirror image
PITCHES = (0, 20, 35, 50, 65)              # approach tilt into the printer, deg
DX = np.round(np.arange(-0.15, 0.36, 0.05), 3)   # carriage offset from printer centre (world x), m
GRASP_Z_MM = 12.0
PRE_MM = 60.0
OUT = pathlib.Path("runs/reach")

_W = _IK = None


def _init(overrides):
    global _W, _IK
    for k, v in overrides.items():
        setattr(C, k, v)
    from farm.world import World
    from plan.ik import IK
    _W = World()
    _IK = IK(_W)
    _IK.sync(_W.d)


def seeds(rng, n=10):
    base = [np.array([1.57, 1.2, 1.2, 0, 0, 0]), np.array([1.57, 0.6, 1.8, -0.8, 0, 0]),
            np.array([1.57, 1.8, 1.0, -1.0, 0, 0])]
    lo, hi = _IK.lo, _IK.hi
    return base + [rng.uniform(lo, hi) for _ in range(n)]


def grasp_frames(side, pitch_deg):
    """Two jaw orientations for a given tilt: closing along bed x, or along the in-plane perpendicular."""
    from plan.ik import frame
    a = np.radians(pitch_deg)
    yl = np.array([0, side, 0.0])
    xl = np.array([side, 0, 0.0])
    approach = np.sin(a) * yl + np.array([0, 0, -np.cos(a)])
    return {"jaw_x": frame(approach, xl), "jaw_y": frame(approach, np.cross(approach, xl))}


def best_option(p, R_opts, xc, door_open=True, dx_set=DX, rng=None, q_hint=None, pre=PRE_MM / 1000):
    """Search (jaw, dx, seed) for a collision-free grasp + pre-grasp. Returns dict or None."""
    rng = rng or np.random.default_rng(0)
    doors = {P: C.DOOR_OPEN_RAD if door_open else 0.0}
    best = None
    for jaw, R in R_opts.items():
        approach = R[:, 2]
        for dx in dx_set:
            x = xc + dx
            for q0 in ([q_hint] if q_hint is not None else []) + seeds(rng, 6):
                q, _, ok, _ = _IK.solve(p, R, x, q0=q0)
                if not ok or _IK.collides(q, x, doors, grip=0.037):
                    continue
                qp, _, okp, _ = _IK.solve(p - pre * approach, R, x, q0=q)
                if not okp or _IK.collides(qp, x, doors, grip=0.037):
                    continue
                sc = _IK.manipulability(q, x) * _IK.limit_margin(q)
                if best is None or sc > best["score"]:
                    best = {"score": sc, "jaw": jaw, "dx": float(dx), "q": q.tolist(), "q_pre": qp.tolist()}
                break        # first feasible seed per (jaw, dx) is enough
    return best


def _cell(args):
    bx, by = args
    xc, side = C.printer_frame(P)
    p = np.array(C.bed_mm_to_world(P, bx, by, GRASP_Z_MM))
    rng = np.random.default_rng(int(bx * 1000 + by))
    res = {"bx": bx, "by": by, "pitch": None}
    for pitch in PITCHES:
        b = best_option(p, grasp_frames(side, pitch), xc, rng=rng)
        if b:
            res.update(b, pitch=pitch)
            break            # prefer the least tilted feasible grasp
    return res


def bed_grid(step=32):
    xs = np.arange(16, C.BED_MM[0], step)
    ys = np.arange(C.PURGE_STRIP_MM + 8, C.BED_MM[1], step)
    return [(float(x), float(y)) for y in ys for x in xs]


def run_map(overrides, step=32, procs=11):
    with mp.Pool(procs, initializer=_init, initargs=(overrides,)) as pool:
        cells = pool.map(_cell, bed_grid(step))
    cov = np.mean([c["pitch"] is not None for c in cells])
    return {"overrides": overrides, "coverage": float(cov), "cells": cells}


def plot_map(res, path):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cells = res["cells"]
    fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=200)
    ax.add_patch(plt.Rectangle((0, 0), *C.BED_MM, fc="#2a2b2e", ec="#555"))
    cmap = {0: "#2EA06B", 20: "#7BC96F", 35: "#F2C14E", 50: "#F28C28", 65: "#B5651D", None: "#E03E5A"}
    for c in cells:
        ax.add_patch(plt.Rectangle((c["bx"] - 14, c["by"] - 14), 28, 28, fc=cmap[c["pitch"]], ec="none", alpha=0.95))
    for k, v in cmap.items():
        ax.scatter([], [], c=v, s=80, marker="s", label="unreachable" if k is None else f"{k}° tilt")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=6, frameon=False, fontsize=9)
    ax.annotate("door / aisle side", xy=(128, -6), ha="center", va="top", fontsize=10, color="#444",
                annotation_clip=False)
    ax.set_xlim(-6, C.BED_MM[0] + 6)
    ax.set_ylim(-20, C.BED_MM[1] + 6)
    ax.set_aspect("equal")
    ax.axis("off")
    o = res["overrides"]
    ax.set_title(f"YAM reach on a 256 mm bed — coverage {100 * res['coverage']:.0f}%\n"
                 f"aisle {o.get('AISLE_HALF', C.AISLE_HALF)} m · base {o.get('BASE_Z', C.BASE_Z)} m · "
                 f"bed unload height {o.get('BED_ZL', C.BED_ZL)} m", fontsize=10)
    fig.savefig(path, bbox_inches="tight")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--map", action="store_true")
    ap.add_argument("--step", type=int, default=32)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.sweep:
        rows = []
        for aisle, base, bed in itertools.product((0.18, 0.22, 0.26), (0.78, 0.88, 0.98), (0.10, 0.16)):
            ov = {"AISLE_HALF": aisle, "BASE_Z": base, "BED_ZL": bed}
            r = run_map(ov, step=48)
            pitches = [c["pitch"] for c in r["cells"] if c["pitch"] is not None]
            rows.append({**ov, "coverage": r["coverage"], "mean_tilt": float(np.mean(pitches)) if pitches else None})
            print(rows[-1], flush=True)
        json.dump(rows, open(OUT / "sweep.json", "w"), indent=1)
    if a.map:
        r = run_map({}, step=a.step)
        json.dump(r, open(OUT / "map.json", "w"), indent=1)
        plot_map(r, OUT / "reach_map.png")
        print("coverage", r["coverage"])
