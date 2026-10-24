"""Grasp-feasibility map of the bed -> slicer keep-in mask.

For every grid point a set of probe parts (cube, narrow clip at 0/90 deg, tall standoff) is placed alone on the
bed with the door open; the point is feasible if every probe has a grasp. The packer then only places parts whose
footprint corners and centre are feasible: the robot's capability is fed back into the slicer.
The probes stand in for every part, which holds near the door, where small-only plates put their parts; deeper in
the bed (plates shared with large parts) a lid or block is harder than any probe. Large SKUs do not fit the probe rule
at all (their ends reach cells the probes cannot, the jaws only go to the centre, and the yaw decides how the jaws
come down). So every SKU also gets its own map per packer yaw (0 / 90 deg; the plate is mirrored afterwards): the
part alone on the bed with its centre at the cell, feasible if it has a grasp ("sku" in the json). The packer needs
the centre in the SKU's own map, and for small parts the probe rule as well.

python -m plan.bedmap --procs 8         -> runs/bedmap.json, runs/bedmap.png (probe map)
python -m plan.bedmap --sku --procs 4   -> adds the per-SKU maps to runs/bedmap.json (probe map kept as it is)
"""
import argparse
import json
import multiprocessing as mp
import pathlib

import numpy as np

from farm import catalog as C

P = 4
PROBES = [("cube", 0.0), ("clip", 0.0), ("clip", 90.0), ("standoff", 0.0)]
STEP = 16.0
OUT = pathlib.Path("runs")
_W = _IK = None


def _init():
    global _W, _IK
    from farm.world import World
    from plan.ik import IK
    _W = World()
    _IK = IK(_W)
    xc, _ = C.printer_frame(P)
    _W.reset(rail_x=xc + 0.4)
    _W.d.qpos[_W.door_q[P]] = C.DOOR_OPEN_RAD


def _point(xy):
    import math
    from plan.unload import plan_grasp
    ok = []
    for sku, yaw in PROBES:
        b = _W.free_part(sku)
        _W.spawn(b, C.bed_mm_to_world(P, *xy), math.radians(yaw), stuck=False)
        _IK.sync(_W.d)
        ok.append(plan_grasp(_IK, _W, P, b) is not None)
        _W.park(b)
    return xy, ok


def sku_fits(sku, yaw, xy):
    """Does a part centred at xy (bed mm) with this packer yaw lie inside the packer's bin (margins, purge strip)?"""
    from jobs.plates import BED_MARGIN_MM
    fx, fy = C.CATALOG[sku].footprint if yaw == 0 else C.CATALOG[sku].footprint[::-1]
    return (BED_MARGIN_MM + fx / 2 <= xy[0] <= C.BED_MM[0] - BED_MARGIN_MM - fx / 2
            and C.PURGE_STRIP_MM + fy / 2 <= xy[1] <= C.BED_MM[1] - BED_MARGIN_MM - fy / 2)


def sku_yaws(sku):
    """Packer yaws with distinct maps: one for round and square parts."""
    k = C.CATALOG[sku]
    return (0,) if k.footprint[0] == k.footprint[1] else (0, 90)


def _sku_point(job):
    import math
    from plan.unload import plan_grasp
    sku, yaw, xy = job
    if not sku_fits(sku, yaw, xy):
        return job, False
    b = _W.free_part(sku)
    _W.spawn(b, C.bed_mm_to_world(P, *xy), math.radians(yaw + 180), stuck=False)    # mirror_plate turns parts 180
    _IK.sync(_W.d)
    ok = plan_grasp(_IK, _W, P, b) is not None
    _W.park(b)
    return job, ok


def grid():
    xs = np.arange(STEP / 2, C.BED_MM[0], STEP)
    ys = np.arange(STEP / 2, C.BED_MM[1], STEP)
    return [(float(x), float(y)) for y in ys for x in xs]


def load_mask(path=OUT / "bedmap.json", mirror=False):
    """reach_mask(x_mm, y_mm) for jobs.plates (nearest grid cell; outside the grid = infeasible).
    mirror=True when the plate will be mirrored afterwards (packer coordinates are then x -> W - x).
    mask.sku[(sku, packer yaw)](x, y): the per-SKU centre maps (empty if the json has none)."""
    d = json.loads(pathlib.Path(path).read_text())
    pts = np.array(d["points"])
    W = C.BED_MM[0]

    def lookup(ok, sel=None):
        p = pts if sel is None else pts[sel]
        ok = ok if sel is None else ok[sel]

        def f(x, y):
            if mirror:
                x = W - x
            k = int(np.argmin((p[:, 0] - x) ** 2 + (p[:, 1] - y) ** 2))
            near = np.hypot(p[k, 0] - x, p[k, 1] - y) <= STEP
            return bool(ok[k] and near)
        return f

    mask = lookup(np.array(d["feasible"], bool))
    mask.sku = {}
    for sku, by_yaw in d.get("sku", {}).items():
        for yaw in (0, 90):                       # round / square parts: one map for both yaws
            v = by_yaw.get(str(yaw), by_yaw["0"])
            fits = np.array([sku_fits(sku, yaw if str(yaw) in by_yaw else 0, (x, y)) for x, y in pts])
            if fits.any():                        # nearest cell where the part fits (a part at the margin is
                mask.sku[(sku, yaw)] = lookup(np.array(v, bool), fits)    # between a cell and one it overhangs)
    return mask


def plot(d, path):
    import matplotlib.pyplot as plt
    pts = np.array(d["points"])
    frac = np.array(d["fraction"])
    fig, ax = plt.subplots(figsize=(6.4, 6.6), dpi=200)
    ax.add_patch(plt.Rectangle((0, 0), *C.BED_MM, fc="#2a2b2e", ec="#555"))
    for (x, y), f in zip(pts, frac):
        col = "#2EA06B" if f == 1 else ("#F2C14E" if f >= 0.5 else "#E03E5A")
        ax.add_patch(plt.Rectangle((x - STEP / 2 + 1, y - STEP / 2 + 1), STEP - 2, STEP - 2, fc=col, ec="none"))
    ax.text(C.BED_MM[0] / 2, -10, "door / robot side", ha="center", va="top", fontsize=11, color="#444")
    ax.text(-6, C.BED_MM[1] / 2, "hinge side", ha="right", va="center", rotation=90, fontsize=11, color="#444")
    ax.set_xlim(-30, C.BED_MM[0] + 6)
    ax.set_ylim(-30, C.BED_MM[1] + 6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Grasp-feasible bed area (all probe parts): {100 * np.mean(frac == 1):.0f}%\n"
                 "green = all probes graspable, amber = some, red = none (YAM, door open)", fontsize=11)
    fig.savefig(path, bbox_inches="tight")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--sku", action="store_true", help="add the per-SKU maps to the existing runs/bedmap.json")
    ap.add_argument("--skus", default=",".join(C.CATALOG), help="with --sku: the SKUs to (re)compute")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.sku:
        d = json.loads((OUT / "bedmap.json").read_text())
        pts = [tuple(p) for p in d["points"]]
        skus = a.skus.split(",")
        jobs = [(sku, yaw, p) for sku in skus for yaw in sku_yaws(sku) for p in pts]
        with mp.Pool(a.procs, initializer=_init) as pool:
            res = dict(pool.imap_unordered(_sku_point, jobs, chunksize=1))
        d.setdefault("sku", {}).update(d.pop("large", {}))          # older json: large-SKU maps only
        d["sku"].update({sku: {str(yaw): [res[(sku, yaw, p)] for p in pts] for yaw in sku_yaws(sku)}
                                        for sku in skus})
        (OUT / "bedmap.json").write_text(json.dumps(d))
        for sku, by_yaw in d["sku"].items():
            print(sku, {y: f"{sum(v)}/{sum(sku_fits(sku, int(y), p) for p in pts)}" for y, v in by_yaw.items()})
        raise SystemExit
    pts = grid()
    with mp.Pool(a.procs, initializer=_init) as pool:
        res = dict(pool.imap_unordered(_point, pts, chunksize=2))
    oks = [res[p] for p in pts]
    d = {"printer": P, "step_mm": STEP, "probes": PROBES, "points": pts,
         "feasible": [all(o) for o in oks], "fraction": [float(np.mean(o)) for o in oks]}
    (OUT / "bedmap.json").write_text(json.dumps(d))
    plot(d, OUT / "bedmap.png")
    print(f"feasible {100 * np.mean(d['feasible']):.0f}% of {len(pts)} cells")
