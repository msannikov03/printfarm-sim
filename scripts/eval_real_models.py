"""Real 3D models end to end: STL / 3MF -> BambuStudio (arrange on one P1S plate, slice) -> G-code parser (the plan)
-> MuJoCo (the meshes where the slicer put them, on printer 5 of the two-row cell) -> chamber camera + grasp + unload.

    PYTHONPATH=. python scripts/eval_real_models.py --models real/assets/models/*.stl --name open
    PYTHONPATH=. python scripts/eval_real_models.py --sliced runs/real_models/open/plate.gcode.3mf   # skip slicing
    PYTHONPATH=. python scripts/eval_real_models.py --models a.3mf --each --name private --anonymous

--each slices every model file onto its own plate (one episode per file); otherwise all models share one plate.
--anonymous replaces object and file names by part_<k> / model_<k> in every output (for models you may not name).
Writes <out>/report.json and <out>/table.md: per part the G-code plan vs the true mesh footprint (centre, size,
height, yaw), the chamber camera's pose error after it has refined the plan, whether a grasp was found, and how the
part ended (placed in its tote or tray slot, raked into the reject tray, left on the bed) with the failure reason.
"""
import argparse
import collections
import json
import math
import pathlib
import sys
import time
import traceback

import numpy as np

OUT = pathlib.Path("runs/real_models")


def route(k):
    """Where the planner sends a part: 'reject' if no jaw opening fits across it (wider than the gripper on both
    sides: it is not attempted and goes to the rake), 'tray' for a large part, else its tote."""
    from farm import catalog as C
    if min(k.footprint) > C.GRIPPER_MAX_MM:
        return "reject"
    return "tray" if k.large else "tote"


def truth_pose(w, i, body):
    """SIM truth: the body's footprint centre (bed mm), yaw (deg) and tilt of its z axis (deg)."""
    from farm import catalog as C
    p, q = w.part_pose(body)
    xc, s = C.printer_frame(i)
    bx = (s * (p[0] - xc) + C.BED_MM[0] / 2000) * 1000
    by = (s * p[1] - C.AISLE_HALF - C.BED_FRONT_GAP) * 1000
    R = np.zeros(9)
    import mujoco
    mujoco.mju_quat2Mat(R, q)
    R = R.reshape(3, 3)
    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0])) + (0.0 if s > 0 else 180.0)
    tilt = math.degrees(math.acos(min(1.0, abs(R[2, 2]))))
    return np.array([bx, by]), yaw, tilt


def _yaw_diff(a, b, period=180.0):
    return (a - b + period / 2) % period - period / 2


def episode(rp, out, realism="ideal", printer=4, seed=0, video=False, anon=None, prior_mm=1.5, prior_deg=2.0):
    """One plate in physics. rp: sim.real_parts.RealPlate (its SKUs are registered). Each part lies off its sliced pose
    by N(0, prior_mm) per axis and N(0, prior_deg) in yaw (scripts/eval_unload.py's spawn noise). -> result dict."""
    import mujoco
    from farm import catalog as C
    from farm.world import World
    import plan.door as D
    import plan.mesh_grasp as MG
    import plan.unload as U
    from plan.door import TUCK, operate_door
    from plan.ik import IK
    from plan.motion import Executor
    from plan.sweep import in_tray
    from perceive.verify import estimate, plate_objects
    from sim.real_parts import parse_errors, spawn_truth, without_top_model

    anon = anon or (lambda n: n)
    t_wall = time.time()
    U.CAMERA_NOISE = realism
    U._CAMS.clear()                                     # cameras of an earlier plate's world
    U._SNAPCAM.clear()
    rng = np.random.default_rng(seed)
    w = World()
    ik = IK(w)
    xc, s = C.printer_frame(printer)
    w.reset(rail_x=xc + 0.4)
    w.d.qpos[w.arm_q] = TUCK
    w.d.ctrl[w.act_arm] = TUCK
    mujoco.mj_forward(w.m, w.d)
    placed = spawn_truth(w, rp, stuck=False, rng=np.random.default_rng(40_000 + seed), pos_sigma_mm=prior_mm,
                         yaw_sigma_deg=prior_deg)
    w.step(200)
    rec = None
    if video:
        from farm.recorder import Recorder
        rec = Recorder(w.m, str(out / "unload.mp4"),
                       view=lambda w: ([float(w.d.qpos[w.rail_q]) - 0.05, 0.05, 0.85], 1.5, 90 + 45, -25))
    ex = Executor(w, rec, realism=realism, ik=ik, rng=np.random.default_rng(10_000 + seed))
    cam = U.bed_camera(w, printer, rng=np.random.default_rng(20_000 + seed))
    ik.belief = U.prior_belief(w, printer, rp.plate, placed)
    by_obj = {p.obj["name"]: p for p in rp.parts}
    rows = {}
    for p in rp.parts:                                  # the plan vs the true mesh, and how the part lies after release
        b = placed[p.obj["name"]]
        c, yaw, tilt = truth_pose(w, printer, b)
        k = C.CATALOG[p.sku]
        rows[p.obj["name"]] = {"part": anon(p.name), "sku": p.sku, "route": route(k), "large": bool(k.large),
                               "footprint_mm": [round(v, 1) for v in p.obj["footprint_mm"]],
                               "height_mm": round(p.obj["height_mm"], 1), "pieces": p.n_pieces,
                               "plan": parse_errors(p), "settled_tilt_deg": round(tilt, 1),
                               "shift_from_slice_mm": round(float(np.hypot(*(c - p.geom.center_mm))), 2)}

    def probe():
        """With the door open, before the first pick: one chamber scan, the slicer prior refined (with and without
        the mesh's top-face model), and whether the planner finds a grasp from that estimate."""
        # perception: one chamber scan, the slicer prior refined (with and without the mesh's top-face model)
        mujoco.mj_forward(w.m, w.d)
        pts = U.scan(cam)
        objs = plate_objects(rp.plate)
        est, unknown = estimate(pts, objs)
        est_flat = {}
        for p in rp.parts:
            swap_in, swap_back = without_top_model(p.sku)
            swap_in()
            try:
                est_flat.update(estimate(pts, [o for o in objs if o["name"] == p.obj["name"]])[0])
            finally:
                swap_back()
        for n, r in rows.items():
            c, yaw, _ = truth_pose(w, printer, placed[n])
            e, ef = est[n], est_flat[n]
            sq = abs(r["footprint_mm"][0] - r["footprint_mm"][1]) < 0.05 * max(r["footprint_mm"])
            per = 90.0 if sq else 180.0
            r["perception"] = {"found": bool(e["found"]), "quality": e.get("quality"),
                               "center_err_mm": round(float(np.hypot(*(np.array(e["center_mm"]) - c))), 2),
                               "yaw_err_deg": round(float(_yaw_diff(e["yaw_deg"], yaw, per)), 2),
                               "prior_center_err_mm": round(float(np.hypot(*(np.array(by_obj[n].obj["center_mm"]) - c))), 2),
                               "center_err_no_top_model_mm": round(float(np.hypot(*(np.array(ef["center_mm"]) - c))), 2)}
        # grasp: does the planner find one from the camera estimate, before anything has been picked
        ik.belief = U.scene_belief(w, printer, est, dict(placed), pts)
        ik.sync(w.d)
        for n, r in rows.items():
            U.REJECT, MG.STATS = collections.Counter(), collections.Counter()      # why candidates fail (reporting)
            g = U.plan_grasp(ik, w, printer, placed[n], rng=np.random.default_rng(30_000 + seed), est=est[n]) \
                if est[n]["found"] else None
            r["grasp_found"] = g is not None
            r["grasp_candidates"] = len(C.CATALOG[by_obj[n].sku].grasp_alts)   # from the part's own layers
            if g is not None:
                r["grasp"] = {"tilt_deg": g["tilt"], "height_mm": round(1000 * g["h"], 1), "width_mm": round(1000 * g["width"], 1),
                              "search": "layers" if "mesh" in g else "fixed"}
                if "mesh" in g:          # (cx, cy, jaw yaw, lean, tilt, pinch z, width, tip z, cover, score), part frame
                    r["grasp"].update(jaw_yaw_deg=g["mesh"][2], tip_mm=g["mesh"][7], cover_mm=g["mesh"][8])
            elif est[n]["found"]:
                r["grasp_rejects"] = dict((U.REJECT + MG.STATS).most_common(6))
            U.REJECT = None
        return int(unknown)

    res = {"realism": realism, "n_parts": len(rows)}
    tray = {sl: None for sl in C.LARGE_SLOTS}
    totes = [s_ for s_ in C.CADDY_SLOTS if s_.startswith("o")]
    slot_of = {}

    def slot_for(name):
        if name in slot_of:
            return slot_of[name]
        k = C.CATALOG[by_obj[name].sku]
        if route(k) == "tray" and tray:
            free = [sl for sl, v in tray.items() if v is None]
            if not free:                    # a station visit empties the tray (sim: the parts are parked)
                for sl, v in tray.items():
                    if v is not None and placed[v] in w.in_use and U.part_in_bin(w, placed[v], sl):
                        w.park(placed[v])
                tray.update({sl: None for sl in tray})
                free = list(tray)
            tray[free[0]] = name
            slot_of[name] = free[0]
        else:
            slot_of[name] = totes[len([v for v in slot_of.values() if v in totes]) % len(totes)]
        return slot_of[name]

    def where(body, slot):
        if body not in w.in_use:
            return "removed"
        if slot is not None and U.part_in_bin(w, body, slot):
            return "own_slot"
        if U.part_in_bin(w, body, "reject"):
            return "reject_bin"
        if in_tray(w, printer, body):
            return "reject_tray"
        return "on_bed" if U.on_bed(w, printer, body) else U.where_off(w, body)

    def on_result(name, body, slot, outcome, sec):
        intended = slot or slot_of.get(name)
        r = rows[name]
        r.update(outcome=outcome, slot=intended, s=round(sec, 1),
                 placed=bool(intended is not None and body in w.in_use and U.part_in_bin(w, body, intended)),
                 end=where(body, intended))
        if r["end"] == "on_bed":                       # where a leftover lies (SIM truth, reporting only)
            xc, sd = C.printer_frame(printer)
            pos, quat = w.part_pose(body)
            zax = np.array([2 * (quat[1] * quat[3] + quat[0] * quat[2]), 2 * (quat[2] * quat[3] - quat[0] * quat[1]),
                            1 - 2 * (quat[1] ** 2 + quat[2] ** 2)])
            r["end_bed_mm"] = [round(1000 * sd * (pos[0] - xc), 1),
                               round(1000 * (sd * pos[1] - C.AISLE_HALF - C.BED_FRONT_GAP), 1),
                               round(1000 * (pos[2] - C.BENCH_H - C.BED_ZL), 1)]
            r["end_tilt_deg"] = round(float(np.degrees(np.arccos(np.clip(zax[2], -1, 1)))), 1)
        if outcome == "placed" and not r["placed"] and body in w.in_use and not U.on_bed(w, printer, body):
            w.park(body)

    U.DROP_LOG, U.HOUSEKEEPING_LOG = [], []
    attempts = {}                                       # body -> every pick attempt's outcome (the failure reasons)
    pick0 = U.pick_and_place

    def pick_logged(ex_, ik_, i_, body, slot, rng_=None, est=None):
        r_ = pick0(ex_, ik_, i_, body, slot, rng_, est=est)
        attempts.setdefault(body, []).append(r_[1])
        return r_

    U.pick_and_place = pick_logged
    try:
        ok, dt, note = operate_door(ex, ik, printer, opening=True)
        res["door_open"] = {"ok": bool(ok), "s": round(dt, 1), "note": note}
        if ok:
            res["unknown_pts_first_scan"] = probe()
            empty, _ = U.unload_parts(ex, ik, printer, rp.plate, placed, slot_for, on_result, rng)
            res["bed_empty_after"] = bool(empty)
        okc, dtc, notec = operate_door(ex, ik, printer, opening=False)
        res["door_close"] = {"ok": bool(okc), "s": round(dtc, 1), "note": notec}
    except Exception:
        res["error"] = traceback.format_exc()
    finally:
        U.pick_and_place = pick0
    for n, r in rows.items():
        r["attempts"] = attempts.get(placed[n], [])
        r["failure"] = next((a for a in r["attempts"] if a != "placed"), None) if not r.get("placed") else None
        if r["failure"] is None and not r.get("placed"):
            r["failure"] = "no_grasp" if not r["attempts"] else r.get("outcome")
    if rec:
        rec.close()
    res["parts"] = list(rows.values())
    res["sim_s"], res["wall_s"] = round(float(w.d.time), 1), round(time.time() - t_wall, 1)
    return res


def _outcome(r):
    o, f = r.get("outcome", "-"), r.get("failure")
    return f"{o} ({f})" if f and f != o else o


def _grasp(r):
    if not r.get("grasp_found"):
        return "no"
    return "yes (layers)" if r.get("grasp", {}).get("search") == "layers" else "yes"


def table(results, title):
    """Markdown tables: the plan vs the mesh, and the unload."""
    L = [f"## {title}", "", "| Part | Footprint (mm) | Route | Plan: centre / long / short / height error (mm) | "
         "Camera: centre error (mm), yaw (deg) | Same, flat-top model (mm) | Grasp | Outcome | Ends in |",
         "|---|---|---|---|---|---|---|---|---|"]
    for res in results:
        for r in res["parts"]:
            pl, pc = r["plan"], r.get("perception", {})
            L.append(f"| {r['part']} | {r['footprint_mm'][0]:.0f} x {r['footprint_mm'][1]:.0f} x {r['height_mm']:.0f} "
                     f"| {r['route']} | {pl['center_err_mm']:.2f} / {pl['long_err_mm']:+.2f} / {pl['short_err_mm']:+.2f} / "
                     f"{pl['height_err_mm']:+.2f} | {pc.get('center_err_mm', float('nan')):.2f}, "
                     f"{pc.get('yaw_err_deg', float('nan')):+.1f} | {pc.get('center_err_no_top_model_mm', float('nan')):.2f} "
                     f"| {_grasp(r)} | {_outcome(r)} | {r.get('end', '-')} |")
    return "\n".join(L) + "\n"


def aggregate(results):
    ps = [r for res in results for r in res["parts"]]
    if not ps:
        return {}
    f = lambda key, sub: [r[sub][key] for r in ps if sub in r and key in r[sub]]
    pick = [r for r in ps if r["route"] != "reject"]
    return {"plates": len(results), "parts": len(ps),
            "routes": {k: sum(r["route"] == k for r in ps) for k in ("tote", "tray", "reject")},
            "plan_center_err_mm_mean": float(np.mean(f("center_err_mm", "plan"))),
            "plan_center_err_mm_max": float(np.max(f("center_err_mm", "plan"))),
            "plan_short_err_mm_mean": float(np.mean(f("short_err_mm", "plan"))),
            "plan_long_err_mm_mean": float(np.mean(f("long_err_mm", "plan"))),
            "plan_height_err_mm_mean": float(np.mean(np.abs(f("height_err_mm", "plan")))),
            "camera_center_err_mm_mean": float(np.mean(f("center_err_mm", "perception"))),
            "camera_center_err_mm_max": float(np.max(f("center_err_mm", "perception"))),
            "camera_center_err_flat_top_mm_mean": float(np.mean(f("center_err_no_top_model_mm", "perception"))),
            "grasp_found": sum(bool(r.get("grasp_found")) for r in ps),
            "grasp_found_by_layer_search": sum(r.get("grasp", {}).get("search") == "layers" for r in ps),
            "placed": sum(bool(r.get("placed")) for r in ps),
            "placed_of_pickable": f"{sum(bool(r.get('placed')) for r in pick)}/{len(pick)}",
            "outcomes": {o: sum(r.get("outcome") == o for r in ps) for o in sorted({r.get("outcome", "-") for r in ps})},
            "ends": {o: sum(r.get("end") == o for r in ps) for o in sorted({r.get("end", "-") for r in ps})},
            "doors": f"{sum(bool(res.get('door_open', {}).get('ok')) for res in results)}/"
                     f"{sum(bool(res.get('door_close', {}).get('ok')) for res in results)} of {len(results)}",
            "errors": sum("error" in res for res in results)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="*", default=[], help="STL / 3MF files to slice")
    ap.add_argument("--sliced", nargs="*", default=[], help="already sliced plates (.gcode.3mf from real/bambu.py)")
    ap.add_argument("--each", action="store_true", help="one plate per model file")
    ap.add_argument("--orient", action="store_true", help="let the slicer choose the print orientation")
    ap.add_argument("--pack", action="store_true", help="robot-friendly plate: our packer with jaw clearance "
                    "(sim.real_parts.pack_models), sliced as placed; default: the slicer's own arrange")
    ap.add_argument("--max-plates", type=int, default=2, help="with --pack: plates per model group at most")
    ap.add_argument("--name", default="open")
    ap.add_argument("--out", default=None, help=f"output directory (default {OUT}/<name>)")
    ap.add_argument("--realism", default="ideal", choices=["ideal", "realistic", "harsh"])
    ap.add_argument("--printer", type=int, default=4)
    ap.add_argument("--hull", action="store_true", help="one convex hull per part (default: 3 x 3 convex pieces)")
    ap.add_argument("--no-layer-grasp", action="store_true", help="without the grasp search over the part's own layers "
                    "(plan/mesh_grasp.py): the fixed-height grasp only, as before it")
    ap.add_argument("--anonymous", action="store_true", help="no object or file names in the outputs")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    from real.bambu import slice_models
    from sim.real_parts import plate_from_slice
    import plan.mesh_grasp as MG
    MG.ENABLED = not a.no_layer_grasp
    out = pathlib.Path(a.out or OUT / a.name)
    out.mkdir(parents=True, exist_ok=True)
    plates = [pathlib.Path(p) for p in a.sliced]
    groups = [[m] for m in a.models] if a.each else ([a.models] if a.models else [])
    for k, g in enumerate(groups):
        d = out / (f"plate_{k:02d}" if a.each else "plate")
        print(f"slicing {len(g)} model(s) -> {d}", flush=True)
        try:
            if a.pack:                      # as many plates as it takes (at most --max-plates per group)
                from real.mesh_io import load_mesh
                from sim.real_parts import pack_meshes
                left = [m for p in g for m in load_mesh(p)]
                for j in range(a.max_plates):
                    dj = d if j == 0 else d.parent / f"{d.name}_{j}"
                    p3, packed, left = pack_meshes(left, dj / "packed.3mf")
                    print(f"  plate {j}: {len(packed)} object(s) packed with jaw clearance, {len(left)} left",
                          flush=True)
                    if not packed:
                        break
                    plates.append(slice_models([p3], dj, arrange=False))
                    if not left:
                        break
            else:
                plates.append(slice_models(g, d, orient=a.orient))
        except RuntimeError as e:
            print("  slicing failed:", e, flush=True)
    names = {}

    def anon(n):
        if not a.anonymous:
            return n
        return names.setdefault(n, f"part_{len(names)}")

    results = []
    for k, pl in enumerate(plates):
        from farm import catalog as C
        rp = plate_from_slice(pl, printer=a.printer, cache_dir=out / "meshes", plate_id=f"RM{k}",
                              grid=(1, 1) if a.hull else (3, 3), sku_prefix=f"rm{k}_")
        print(f"plate {k}: {len(rp.parts)} parts, G-code frame offset {rp.frame_offset_mm} mm, "
              f"unmatched {len(rp.unmatched)}", flush=True)
        res = episode(rp, out, a.realism, a.printer, a.seed + k, a.video and k == 0, anon)
        res["plate"] = f"model_{k}" if a.anonymous else pl.parent.name + "/" + pl.name
        res["frame_offset_mm"] = rp.frame_offset_mm
        results.append(res)
        for r in res["parts"]:
            print(f"  {r['part']:<28} {r['route']:<6} plan c {r['plan']['center_err_mm']:.2f} mm, camera c "
                  f"{r.get('perception', {}).get('center_err_mm', float('nan')):.2f} mm, grasp "
                  f"{_grasp(r)} -> {_outcome(r)}, ends {r.get('end', '-')}",
                  flush=True)
        print(f"  doors {res.get('door_open', {}).get('ok')}/{res.get('door_close', {}).get('ok')}, "
              f"sim {res['sim_s']} s, wall {res['wall_s']} s{', ERROR' if 'error' in res else ''}", flush=True)
        for sku in [p.sku for p in rp.parts]:          # this plate's SKUs are not needed by the next world
            C.CATALOG.pop(sku, None)
        (out / "report.json").write_text(json.dumps({"results": results, "aggregate": aggregate(results)}, indent=1))
    (out / "table.md").write_text(table(results, a.name))
    print(json.dumps(aggregate(results), indent=1))
    return results


if __name__ == "__main__":
    sys.exit(0 if main() is not None else 1)
