"""Geometry check for the cell layout: one row of printers vs two facing rows with the rail in the middle.

Everything is checked with the real planners on a MuJoCo model built from patched layout constants (farm/catalog.py
and farm/scene.py are not edited: `apply_layout` overrides the constants and the carriage builder in-process, before
the scene is built). Per layout:
  doors      plan.door.plan_door open + close, for a printer in row A and (two rows) one in row B
  grasp      plan.unload.plan_grasp coverage on a bed grid (cube + clip probes, door open), row A
  drop       plan.unload.drop_pose for every caddy bin with the door open (carriage shift needed: 0 / 0.25 / 0.45 m)
  slide      folded-arm (LOW_TUCK) carriage slide past the open door, +-0.5 m
  clearance  carriage deck vs the benches of both rows (analytic)

    python scripts/layout_geometry.py sweep --procs 2                    -> runs/layout/geometry.jsonl
    python scripts/layout_geometry.py episodes --layout facing_a34_b56 --printers 1 4 --seeds 0 1 2 --procs 2
        -> runs/layout/episodes/<layout>/p<printer>/seed_XXX.json  (full unload episodes: door, perception, picks,
           rake, door close; scripts/eval_unload.py's episode on the patched model)
    python scripts/layout_geometry.py summary --layout facing_a40_b56_px [--baseline runs/eval]
        -> runs/layout/episodes_<layout>.json (+ timings_<layout>.json for the DES), compared with the same seeds
           of the one-row batch

Two-row caddy ("side"): the 8 bins sit in two 2x2 blocks beside the arm along the rail, point-symmetric about the
arm base (one block at +x, one at -x), so row B sees exactly the mirror of what row A sees. The deck is sized to
the aisle (|y| <= aisle - 4 cm). The one-row reference keeps the current caddy behind the arm.
"""
import argparse
import itertools
import json
import math
import multiprocessing as mp
import pathlib
import time
import traceback

import numpy as np

OUT = pathlib.Path("runs/layout")
PER_ROW = 3                       # printers per row in the test model (the middle one has neighbours on both sides)
DECK_Z_MAX = 0.56                 # deck (and bins) stay below the door line; a higher base sits on a riser
BIN_PITCH = 0.12
PLUSX_X0 = 0.26                   # +x caddy: nearest bin column; at 0.20 parts in the bin touch the YAM base (padded)
BED_GRID = [(x, y) for y in (40, 105, 170, 236) for x in (30, 98, 158, 226)]
PROBES = ("cube", "clip")


def layouts():
    """name -> layout dict. 'row' = today's cell; 'facing' = two rows facing each other across the rail."""
    out = {"row_a28_b56": dict(rows=1, aisle=0.28, base_z=0.56, caddy="behind")}
    for a, b in itertools.product((0.28, 0.34, 0.37, 0.40, 0.43), (0.50, 0.56, 0.62)):
        out[f"facing_a{round(100 * a)}_b{round(100 * b)}"] = dict(rows=2, aisle=a, base_z=b, caddy="side")
        if a >= 0.335:
            out[f"facing_a{round(100 * a)}_b{round(100 * b)}_px"] = dict(rows=2, aisle=a, base_z=b, caddy="plusx")
    return out


def side_caddy(aisle):
    """Two 2x2 bin blocks at +-x of the arm base, point-symmetric. The block's y offset grows with the aisle so that
    the bins move out from under the opposite row's open door leaf when there is room."""
    half = 0.11 / 2
    deck_y = aisle - 0.04
    by = max(0.0, min(deck_y - BIN_PITCH / 2 - half - 0.01, 0.12))
    xs, ys = (0.20, 0.32), (by - BIN_PITCH / 2, by + BIN_PITCH / 2)
    plus = [(x, y) for x in xs for y in ys]                          # +x block (near the +y row side)
    minus = [(-x, -y) for x, y in plus]                              # point mirror
    order = ["o2", "o0", "o4", "reject", "o3", "o1", "o5", "waste"]  # o2/o3 (allocator's first choice) nearest
    return dict(zip(order, [plus[0], plus[1], plus[2], plus[3], minus[0], minus[1], minus[2], minus[3]]))


def plusx_caddy(aisle):
    """All 8 bins on the +x side of the arm in a 3x3 grid (one spot left free), y >= 0. The YAM's base joint covers
    -150..+175 deg, so bins straight behind the arm (-x) sit in its dead zone; the +x side is in the middle of its
    range for both rows. Row A's open leaf is on the -x side of the carriage; row B's leaf is on the +x side but
    ends at y = -aisle + 0.335, below the bins (y >= -0.055) once the aisle is >= 0.40 m."""
    deck_y = aisle - 0.04
    ys = [y for y in (0.0, BIN_PITCH, 2 * BIN_PITCH) if y + 0.055 <= deck_y]
    spots = sorted(((x, y) for y in ys for x in (PLUSX_X0, PLUSX_X0 + BIN_PITCH, PLUSX_X0 + 2 * BIN_PITCH)),
                   key=lambda p: (round(math.hypot(*p), 3), p[1]))[:8]          # the far corner stays free
    order = ["o2", "o3", "o0", "o5", "o1", "o4", "reject", "waste"]  # nearest first: o2, o3 (allocator's choice)
    return dict(zip(order, spots))


def apply_layout(cfg):
    """Patch farm.catalog (and the carriage builder of farm.scene) for this layout. Call before building a World."""
    from farm import catalog as C
    C.N_ROWS, C.N_PER_ROW = cfg["rows"], PER_ROW
    C.N_PRINTERS = C.N_ROWS * C.N_PER_ROW
    C.AISLE_HALF = cfg["aisle"]
    C.BASE_Z, C.DECK_Z = cfg["base_z"], min(cfg["base_z"], DECK_Z_MAX)
    C.RAIL_X_MAX = (C.N_PER_ROW - 1) * C.PITCH + 0.95
    C.STATION_X = (C.N_PER_ROW - 1) * C.PITCH + 0.75
    if cfg["caddy"] == "side":
        C.CADDY_SLOTS = side_caddy(cfg["aisle"])
    elif cfg["caddy"] == "plusx":
        C.CADDY_SLOTS = plusx_caddy(cfg["aisle"])
        assert len(C.CADDY_SLOTS) == 8, f"aisle {cfg['aisle']} too narrow for the +x caddy"
    from farm import scene
    scene.N_PURGE = C.N_PRINTERS
    if cfg["rows"] == 2:
        scene._carriage = _carriage_two_rows
    if cfg["rows"] == 2:
        _patch_ik_carry_caddy()
    return C


def _patch_ik_carry_caddy():
    """plan.ik.IK.collides tests carriage positions other than the current one, but parts already in the caddy bins
    stay at their world position in the planning model, so to the planner the arm base slides into them. With the
    bins behind the arm (today) that never happens; with bins beside the arm along the rail it blocks door and
    transfer plans. Here they ride along with the carriage, as they do physically (integration note for plan/ik.py)."""
    from farm import catalog as C
    from plan.ik import IK
    if getattr(IK, "_carry_caddy", False):
        return
    orig_sync, orig_collides = IK.sync, IK.collides

    def sync(self, data):
        orig_sync(self, data)
        self._x_sync = float(data.qpos[self.w.rail_q])
        x0, x1, y0, y1 = deck_extent(C)
        zmax = C.DECK_Z + C.BIN_SIZE[2] + 0.06
        self._riders = []
        for n in self.w.in_use:
            a = self.w.part_q[n]
            px, py, pz = data.qpos[a:a + 3]
            if x0 - 0.02 <= px - self._x_sync <= x1 + 0.02 and y0 - 0.02 <= py <= y1 + 0.02 and pz < zmax:
                self._riders.append((a, float(px)))

    def collides(self, q, rail_x, *args, **kw):
        riders = getattr(self, "_riders", ())
        for a, x in riders:
            self.d.qpos[a] = x + (rail_x - self._x_sync)
        try:
            return orig_collides(self, q, rail_x, *args, **kw)
        finally:
            for a, x in riders:
                self.d.qpos[a] = x

    IK.sync, IK.collides, IK._carry_caddy = sync, collides, True


def deck_extent(C):
    """(x_min, x_max, y_min, y_max) of the deck the caddy needs (two rows: confined to the aisle). The -x edge stays
    at -0.27 m unless bins sit there (as today): the folded slide pose (plan.door.LOW_TUCK) hangs the forearm down
    past it."""
    xs = [dx for dx, _ in C.CADDY_SLOTS.values()]
    ys = [dy for _, dy in C.CADDY_SLOTS.values()]
    y_lim = C.AISLE_HALF - 0.04
    return (min(-0.27, min(xs) - 0.07), max(0.27, max(xs) + 0.07),
            max(-y_lim, min(min(ys) - 0.07, -0.12)), min(y_lim, max(max(ys) + 0.07, 0.12)))


def _carriage_two_rows(arm_xml):
    """Carriage for two facing rows: the deck stays inside the aisle, the caddy rides beside the arm (same geoms and
    names as farm.scene._carriage, so the planners find the bins)."""
    from farm import catalog as C
    from farm.scene import _bin, _box
    deck_t = 0.03
    x0, x1, y0, y1 = deck_extent(C)
    g = [_box("deck", ((x0 + x1) / 2, (y0 + y1) / 2, -deck_t / 2), ((x1 - x0) / 2, (y1 - y0) / 2, deck_t / 2),
              mat="carriage"),
         _box(None, (0, 0, -0.25), (0.12, 0.08, 0.22), mat="carriage", extra='contype="0" conaffinity="0"')]
    if C.BASE_Z - C.DECK_Z > 0.01:
        g.append(f'<geom name="riser" type="cylinder" pos="0 0 {(C.BASE_Z - C.DECK_Z) / 2:.4f}" '
                 f'size="0.055 {(C.BASE_Z - C.DECK_Z) / 2:.4f}" material="trim"/>')
    for slot, (dx, dy) in C.CADDY_SLOTS.items():
        mat = "binwaste" if slot == "waste" else "binreject" if slot == "reject" else "bin"
        g.append(_bin(slot, dx, dy, 0.0, mat=mat))
    return (f'<body name="carriage" pos="0 0 {C.DECK_Z:.4f}">'
            f'<joint name="rail" type="slide" axis="1 0 0" range="{C.RAIL_X_MIN} {C.RAIL_X_MAX}" damping="50" armature="5"/>'
            + "".join(g) + arm_xml + '</body>')


def clearance(C):
    """Deck vs benches: the benches occupy |y| >= aisle (rows present) from the floor to the bench top; the deck
    (z ~ 0.53-0.56) is below the bench top, so it must stay inside the aisle. Returns the worst gap (m)."""
    if C.N_ROWS == 2:
        _, _, y0, y1 = deck_extent(C)
    else:   # farm/scene.py: front = min(0.15, AISLE_HALF - 0.04), deck back at -0.46 (no bench behind one row)
        y0, y1 = -0.46, min(0.15, C.AISLE_HALF - 0.04)
    gaps = [C.AISLE_HALF - y1] + ([C.AISLE_HALF + y0] if C.N_ROWS == 2 else [])
    return round(float(min(gaps)), 3)


def _world(cfg):
    C = apply_layout(cfg)
    import mujoco
    from farm.world import World
    from plan.door import TUCK
    from plan.ik import IK
    w = World()
    ik = IK(w)

    def reset(i):
        xc, s = C.printer_frame(i)
        w.reset(rail_x=xc + s * 0.4)
        w.d.qpos[w.arm_q] = TUCK
        mujoco.mj_forward(w.m, w.d)
        ik.sync(w.d)
    return C, w, ik, reset


def door_check(C, w, ik, reset, i):
    import mujoco
    from plan.door import plan_door
    reset(i)
    t0 = time.time()
    op, _ = plan_door(ik, w, i, opening=True)
    w.d.qpos[w.door_q[i]] = C.DOOR_OPEN_RAD
    mujoco.mj_forward(w.m, w.d)
    ik.sync(w.d)
    cl, _ = plan_door(ik, w, i, opening=False)
    w.d.qpos[w.door_q[i]] = 0.0
    mujoco.mj_forward(w.m, w.d)
    return {"open_stages": None if op is None else len(op), "close_stages": None if cl is None else len(cl),
            "s": round(time.time() - t0, 1)}


def sweep_one(item):
    name, cfg = item
    import plan.door as D
    D.VERBOSE = False
    t0 = time.time()
    res = {"layout": name, **cfg}
    try:
        C, w, ik, reset = _world(cfg)
        import mujoco
        from plan.door import slide_free
        from plan.unload import drop_pose, plan_grasp
        res["deck_bench_gap_m"] = clearance(C)
        pa = 1                                            # middle of row A
        res["door_A"] = door_check(C, w, ik, reset, pa)
        if cfg["rows"] == 2:
            res["door_B"] = door_check(C, w, ik, reset, PER_ROW + 1)
        reset(pa)
        xc, s = C.printer_frame(pa)
        w.d.qpos[w.door_q[pa]] = C.DOOR_OPEN_RAD
        mujoco.mj_forward(w.m, w.d)
        ik.sync(w.d)
        grid = []
        for xy in BED_GRID:
            ok = True
            for sku in PROBES:
                b = w.free_part(sku)
                w.spawn(b, C.bed_mm_to_world(pa, *xy), 0.0, stuck=False)
                ik.sync(w.d)
                ok &= plan_grasp(ik, w, pa, b) is not None
                w.park(b)
                if not ok:
                    break
            grid.append(bool(ok))
        ik.sync(w.d)
        res["grasp_grid"] = grid
        res["grasp_coverage"] = round(sum(grid) / len(grid), 3)
        for tag, i in (("A", pa), ("B", PER_ROW + 1))[:cfg["rows"]]:
            reset(i)
            xc, s = C.printer_frame(i)
            w.d.qpos[w.door_q[i]] = C.DOOR_OPEN_RAD
            mujoco.mj_forward(w.m, w.d)
            ik.sync(w.d)
            drops = {}
            for slot in C.CADDY_SLOTS:       # as unload.py: shift the carriage away from the hinge if the bin is blocked
                drops[slot] = None
                for shift in (0.0, 0.25, 0.45):
                    if drop_pose(ik, w, slot, xc + s * (0.15 + shift)) is not None:
                        drops[slot] = shift
                        break
            res[f"drop_shift_m_{tag}"] = drops
            res[f"drop_ok_{tag}"] = sum(v is not None for v in drops.values())
            # the drop logic slides the folded arm away from the hinge (to the printer's right) with the door open
            res[f"slide_open_door_{tag}"] = bool(slide_free(ik, i, xc + s * 0.15, xc + s * 0.6, C.DOOR_OPEN_RAD))
    except Exception:
        res["error"] = traceback.format_exc()
    res["wall_s"] = round(time.time() - t0, 1)
    return res


def episode_one(item):
    name, cfg, printer, seed, out = item
    apply_layout(cfg)
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import eval_unload
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    return eval_unload.episode((seed, False, printer, str(out), False))


def timings_from(eps, source):
    """Measured durations/rates -> DES timings (same formulas as scripts/make_timings.py)."""
    opens = [e["door_open"]["s"] for e in eps if e.get("door_open", {}).get("ok")]
    closes = [e["door_close"]["s"] for e in eps if e.get("door_close", {}).get("ok")]
    parts = [p for e in eps for p in e["parts"]]
    placed = [p for p in parts if p["outcome"] == "placed"]
    purge = [p["s"] for p in placed if p["name"].endswith("purge_tab")]
    picks = [p["s"] for p in placed if not p["name"].endswith("purge_tab")]
    blocking = [p for p in parts if p.get("blocking")]
    swept = [p for p in parts if p["outcome"] == "swept"]
    n = max(len(parts), 1)
    return {"source": source, "rail_speed_mps": 0.4,
            "door_open_s": float(np.mean(opens)) if opens else None,
            "door_close_s": float(np.mean(closes)) if closes else None, "scan_s": 2.0,
            "purge_tab_s": float(np.mean(purge)) if purge else 20.0,
            "pick_place_s_mean": float(np.mean(picks)) if picks else None,
            "pick_place_s_std": float(np.std(picks)) if picks else None,
            "bed_check_s": 2.0, "station_handoff_s": 6.0, "retry_s": 8.0,
            "p_success_first_try": len(placed) / n, "p_intervention_per_part": len(blocking) / n,
            "p_reprint_per_part": (len(parts) - len(placed) - len(blocking)) / n,
            "rake_s": float(np.mean([p["s"] for p in swept])) if swept else 15.0,
            "door_ok_rate": (len(opens) + len(closes)) / max(2 * len(eps), 1),
            "outcomes": {k: sum(p["outcome"] == k for p in parts) for k in sorted({p["outcome"] for p in parts})}}


def summarize_episodes(layout, baseline):
    """Per printer of the layout: placed rate, doors, timings; plus the same seeds of the one-row batch."""
    root = OUT / "episodes" / layout
    res = {"layout": layout, "cfg": layouts()[layout], "printers": {}}
    seeds = set()
    for d in sorted(root.glob("p*")):
        eps = [json.loads(f.read_text()) for f in sorted(d.glob("seed_*.json"))]
        seeds |= {e["seed"] for e in eps}
        res["printers"][d.name] = {"episodes": len(eps), "errors": sum("error" in e for e in eps),
                                   "bed_empty": sum(bool(e.get("bed_empty_after")) for e in eps),
                                   **timings_from(eps, str(d))}
    all_eps = [json.loads(f.read_text()) for f in sorted(root.glob("p*/seed_*.json"))]
    res["all"] = timings_from(all_eps, f"{root} ({len(all_eps)} plates)")
    base = [json.loads(f.read_text()) for f in sorted(pathlib.Path(baseline).glob("seed_*.json"))]
    base = [e for e in base if e["seed"] in seeds]
    res["baseline_same_seeds"] = timings_from(base, f"{baseline} seeds {sorted(seeds)} (one row, aisle 0.28 m)")
    path = OUT / f"episodes_{layout}.json"
    path.write_text(json.dumps(res, indent=1))
    t = res["all"]
    (OUT / f"timings_{layout}.json").write_text(json.dumps({k: v for k, v in t.items() if v is not None}, indent=1))
    for name, r in [*res["printers"].items(), ("all", res["all"]), ("baseline", res["baseline_same_seeds"])]:
        print(f"{name:>9}: outcomes {r['outcomes']}  door ok {r['door_ok_rate']:.2f}  open {r['door_open_s']}  "
              f"close {r['door_close_s']}  pick {r['pick_place_s_mean']}")
    print("wrote", path)


def render(layout, per_row=5, size=(1600, 900)):
    """Stills of a layout (open door on a row-A printer, arm reaching into it) -> runs/layout/render_<layout>_*.png."""
    global PER_ROW
    PER_ROW = per_row
    C, w, ik, reset = _world(layouts()[layout])
    import imageio
    import mujoco
    i = per_row // 2
    xc, s = C.printer_frame(i)
    reset(i)
    w.d.qpos[w.rail_q] = xc + 0.15
    w.d.qpos[w.door_q[i]] = C.DOOR_OPEN_RAD
    w.d.qpos[w.arm_q] = np.array([1.57, 1.0, 1.4, -0.9, 0.0, 0.0])
    mujoco.mj_forward(w.m, w.d)
    r = mujoco.Renderer(w.m, size[1], size[0])
    cam = mujoco.MjvCamera()
    mid = (per_row - 1) * C.PITCH / 2
    views = {"overview": ((mid, 0.0, 0.9), 4.2, 150, -38), "aisle": ((mid, 0.0, 0.95), 2.6, 180, -18)}
    for name, (lookat, dist, az, el) in views.items():
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = lookat, dist, az, el
        r.update_scene(w.d, cam)
        path = OUT / f"render_{layout}_{name}.png"
        imageio.imwrite(path, r.render())
        print("wrote", path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("sweep", "episodes", "summary", "render"))
    ap.add_argument("--baseline", default="runs/eval", help="summary: one-row batch (seed_*.json) to compare")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--only", nargs="*", help="sweep: layout names to run (default: all)")
    ap.add_argument("--layout", help="episodes: layout name")
    ap.add_argument("--printers", type=int, nargs="*", default=[1])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    L = layouts()
    ctx = mp.get_context("spawn")
    if a.stage == "sweep":
        items = [(n, c) for n, c in L.items() if not a.only or n in a.only]
        path = OUT / "geometry.jsonl"
        with ctx.Pool(a.procs, maxtasksperchild=1) as pool:
            for r in pool.imap_unordered(sweep_one, items):
                with path.open("a") as f:
                    f.write(json.dumps(r) + "\n")
                print({k: v for k, v in r.items() if k not in ("grasp_grid", "error")}, flush=True)
                if "error" in r:
                    print(r["error"], flush=True)
        return
    if a.stage == "summary":
        return summarize_episodes(a.layout, a.baseline)
    if a.stage == "render":
        return render(a.layout)
    cfg = L[a.layout]
    items = [(a.layout, cfg, p, s, str(OUT / "episodes" / a.layout / f"p{p}")) for p in a.printers for s in a.seeds]
    with ctx.Pool(a.procs, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(episode_one, items):
            ok = sum(p["outcome"] == "placed" for p in r["parts"])
            print(f"{a.layout} printer {r['printer']} seed {r['seed']}: placed {ok}/{len(r['parts'])} door "
                  f"{r.get('door_open', {}).get('ok')}/{r.get('door_close', {}).get('ok')} "
                  f"{'ERROR' if 'error' in r else ''} wall {r['wall_s']:.0f}s", flush=True)
            if "error" in r:
                print(r["error"], flush=True)


if __name__ == "__main__":
    main()
