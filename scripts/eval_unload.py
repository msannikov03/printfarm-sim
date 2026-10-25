"""Batch evaluation of the single-printer unload pipeline over randomized plates.

python scripts/eval_unload.py --n 40 --procs 8 --videos 3 [--start 0 --out runs/eval]   -> seed_XXX.json + summary.json (+ mp4s)
--skus small (default: the legacy plates, same RNG streams) | mixed | large (jobs.plates.sample_parts; two rows only):
large parts go to the large-part tray slots (C.LARGE_SLOTS); when both are taken the tray is emptied, as a station
visit would (counted in station_visits). The summary reports small and large parts separately. Plates are packed
under the farm's 3 h cap (one large part and a few small ones); --max-print-h 8 gives dense stress plates (up to three
large parts, small ones packed deep behind them).
"""
import argparse
import collections
import json
import multiprocessing as mp
import pathlib
import time
import traceback

import numpy as np

OUT = pathlib.Path("runs/eval")


def episode(args):
    seed, video, printer, out, faults_on, opts = (*args, *(OUT, False, {})[len(args) - 3:])
    out = pathlib.Path(out)
    globals()["FAULTS"] = faults_on
    import mujoco
    from farm import catalog as C
    from farm.recorder import Recorder
    from farm.world import World
    from jobs.plates import make_plate, sample_parts
    import plan.door as D
    import plan.sweep as SW
    import plan.unload as U
    from plan.door import TUCK, operate_door
    from plan.ik import IK
    from plan.motion import Executor
    from plan.sweep import in_tray
    from plan.unload import mirror_plate, on_bed, part_in_bin, prior_belief, spawn_plate, unload_parts

    realism = opts.get("realism", "ideal")
    U.CAMERA_NOISE = opts.get("camera", realism)
    U.ORACLE = bool(opts.get("oracle", False))
    U.WRIST_ALIGN = not opts.get("no_wrist", False)
    SW.SIDE_PUSH = not opts.get("no_side_push", False)
    t_wall = time.time()
    rng = np.random.default_rng(seed)
    D.VERBOSE = bool(globals().get("VERBOSE_EPISODES"))
    parts = sample_parts(rng, opts.get("skus", "small"))
    mask = None
    if pathlib.Path("runs/bedmap.json").exists():
        from plan.bedmap import load_mask
        mask = load_mask("runs/bedmap.json", mirror=True)
    plate, _ = make_plate(parts, printer, f"PL{seed:03d}", rng, reach_mask=mask,
                          max_print_s=3600 * opts.get("max_print_h", 3.0))
    plate = mirror_plate(plate)
    w = World()
    ik = IK(w)
    xc, s = C.printer_frame(printer)
    w.reset(rail_x=xc + 0.4)
    w.d.qpos[w.arm_q] = TUCK
    w.d.ctrl[w.act_arm] = TUCK
    mujoco.mj_forward(w.m, w.d)
    placed = spawn_plate(w, plate, rng, pos_sigma_mm=1.5, yaw_sigma_deg=2.0, stuck=False)
    faults = {}
    if globals().get("FAULTS"):
        faults = inject_faults(w, printer, plate, placed, rng)
    w.step(200)
    fill = prefill_totes(w, opts.get("tote_fill", 0), np.random.default_rng(30_000 + seed)) \
        if opts.get("tote_fill") else {}
    rec = Recorder(w.m, str(out / f"seed_{seed:03d}.mp4"),
                   view=lambda w: ([float(w.d.qpos[w.rail_q]) - 0.05, 0.05, 0.85], 1.5, 90 + 45, -25)) if video else None
    # actuation realism + the chamber camera noise use their own streams: the plates stay identical across presets
    ex = Executor(w, rec, realism=realism, ik=ik, rng=np.random.default_rng(10_000 + seed))
    U.bed_camera(w, printer, rng=np.random.default_rng(20_000 + seed))
    if not U.ORACLE:
        ik.belief = prior_belief(w, printer, plate, placed)       # before the first scan: the slicer's plate
    res = {"seed": seed, "printer": printer, "n_objects": len(plate["objects"]), "parts": [], "faults": faults,
           "realism": realism, "camera": U.CAMERA_NOISE, "oracle": U.ORACLE, "wrist_align": U.WRIST_ALIGN,
           "skus": opts.get("skus", "small"), "max_print_h": opts.get("max_print_h", 3.0), "station_visits": 0,
           "tote_fill": fill}
    if getattr(ex, "wrist", None) is not None:
        res["handeye_mm"] = np.round(1000 * ex.wrist.handeye, 2).tolist()      # drawn calibration error (truth)
    U.DROP_LOG = []
    U.HOUSEKEEPING_LOG = []
    U.ATTEMPT_LOG = []
    name_of = {b: n for n, b in placed.items()}
    slots = {}
    tray = {s: None for s in C.LARGE_SLOTS}      # large-tray slot -> the part the robot put / is putting there
    large_slot = {}                              # large part -> its slot (scoring)

    def where_now(body, intended):
        """SIM truth, reporting only: where a part (or the debris lump) is now."""
        if body not in w.in_use:        # left the bed elsewhere (side push, dropped): parked by the sim housekeeping
            where = next((e["where"] for e in U.HOUSEKEEPING_LOG if e["name"] == name_of.get(body)), None)
            return f"removed_{where}" if where else "removed"
        if intended is not None and part_in_bin(w, body, intended):
            return "own_bin"
        if part_in_bin(w, body, "reject"):
            return "reject_bin"
        if any(part_in_bin(w, body, s) for s in U.all_slots()):
            return "other_bin"
        if in_tray(w, printer, body):
            return "tray"
        return "on_bed" if on_bed(w, printer, body) else U.where_off(w, body)    # deck, floor, printer

    try:
        ok, dt, note = operate_door(ex, ik, printer, opening=True)
        res["door_open"] = {"ok": ok, "s": dt, "note": note, "deg": float(np.degrees(w.door_angle(printer)))}
        if ok:
            free = ["o0", "o1", "o2", "o3", "o4", "o5"]

            def slot_for(name):
                if name.endswith("purge_tab"):
                    return "waste"
                if C.CATALOG[name.split("_")[1]].large:
                    if name in large_slot and tray[large_slot[name]] == name:
                        return large_slot[name]          # a retry: the same slot
                    if all(tray.values()):               # tray full: a station visit empties it (sim: parked)
                        for s, n in tray.items():
                            b = placed.get(n)
                            if b in w.in_use and part_in_bin(w, b, s):
                                w.park(b)
                        tray.update({s: None for s in tray})
                        res["station_visits"] += 1
                    s = next(s for s, n in tray.items() if n is None)
                    tray[s], large_slot[name] = name, s
                    return s
                oid = name.split("_")[0]
                if oid not in slots:
                    slots[oid] = free.pop(0)
                return slots[oid]

            def on_result(name, body, slot, outcome, sec):
                # outcome = what the robot believes; the rest is simulator truth, for reporting only. Truth is
                # scored against the part's intended bin even when the robot reports a failure (slot None): a part
                # the robot believes stuck on a finger may in fact have dropped into its bin (a missed success)
                intended = slot or ("waste" if name.endswith("purge_tab") else large_slot.get(name)
                                    or slots.get(name.split("_")[0]))
                truly = bool(intended is not None and body in w.in_use and part_in_bin(w, body, intended))
                if name in large_slot and outcome != "placed" and tray[large_slot[name]] == name:
                    tray[large_slot[name]] = None        # the robot believes the slot is still free
                res["parts"].append({"name": name, "sku": w.part_sku[body], "slot": slot, "outcome": outcome,
                                     "large": bool(getattr(C.CATALOG.get(w.part_sku[body]), "large", False)),
                                     "s": round(sec, 2), "believed_placed": outcome == "placed",
                                     "truly_placed": truly, "in_tray": bool(body in w.in_use and in_tray(w, printer, body)),
                                     "blocking": bool(body in w.in_use and on_bed(w, printer, body)),
                                     "end": where_now(body, intended),
                                     "pick_attempts": sum(a["kind"] == "pick" and a["name"] == name
                                                          for a in U.ATTEMPT_LOG),
                                     "drops": [dr for dr in U.DROP_LOG if dr.get("body") == body]})
                U.DROP_LOG[:] = [dr for dr in U.DROP_LOG if dr.get("body") != body]
                if name in large_slot and truly:         # truth, reporting: how centred the lowered placement is
                    sid = mujoco.mj_name2id(w.m, mujoco.mjtObj.mjOBJ_SITE, f"bin_{intended}")
                    p, q = w.part_pose(body)
                    yaw = np.degrees(2 * np.arctan2(q[3], q[0]))
                    res["parts"][-1].update(slot_offset_mm=np.round(1000 * (p[:2] - w.d.site_xpos[sid][:2]), 2).tolist(),
                                            slot_yaw_deg=round(float((yaw + 90) % 180 - 90), 2))
                if outcome == "placed" and not truly and body in w.in_use and not on_bed(w, printer, body):
                    w.park(body)        # SIM housekeeping: a part that missed its bin is collected by the operator

            empty, unknown = unload_parts(ex, ik, printer, plate, placed, slot_for, on_result, rng)
            res["bed_empty_after"], res["unknown_pts"] = bool(empty), int(unknown)
            if "debris" in faults:
                from plan.sweep import in_tray
                db = faults["debris"]["body"]
                faults["debris"].update(on_bed=bool(on_bed(w, printer, db)), in_tray=bool(in_tray(w, printer, db)),
                                        end=where_now(db, None))
        okc, dtc, notec = operate_door(ex, ik, printer, opening=False)
        res["door_close"] = {"ok": okc, "s": dtc, "note": notec, "deg": float(np.degrees(w.door_angle(printer)))}
    except Exception:
        res["error"] = traceback.format_exc()
    if rec:
        rec.close()
    res["sim_s"] = float(w.d.time)
    res["motion"] = D.motion_summary(getattr(w, "transfer_log", []))      # carriage moves (reporting)
    res["moves"] = list(getattr(w, "transfer_log", []))
    res["blob_drops"] = dict(getattr(ex, "blob_drops", {}) or {})   # blob picks into the reject bin, robot's verdict
    res["parked_off"] = list(U.HOUSEKEEPING_LOG or [])              # truth: failed parts that lay off bed / bins
    res["attempts"] = list(U.ATTEMPT_LOG or [])                     # every chamber scan, pick attempt and rake call
    res["wall_s"] = time.time() - t_wall
    if getattr(ex, "hw", None) is not None:
        res["actuation"] = {"zero_deg": np.round(np.degrees(ex.hw.zero), 3).tolist(),
                            "rail_zero_mm": round(1000 * ex.hw.rail_zero, 2)}
    (out / f"seed_{seed:03d}.json").write_text(json.dumps(res, indent=1))
    return res


def prefill_totes(w, n_max, rng):
    """Stress test for the drop: every order tote starts with 0..n_max small parts already in it (as in the farm,
    where a tote collects a package of up to C.TOTE_MAX_PARTS parts over several plates). Each is dropped from above
    the tote at a random pose and settles; one that does not end inside is taken away again. -> {tote: [skus]}"""
    import mujoco
    from farm import catalog as C
    from plan.unload import part_in_bin
    out = {}
    for slot in C.CADDY_SLOTS:
        if not slot.startswith("o"):
            continue
        sid = mujoco.mj_name2id(w.m, mujoco.mjtObj.mjOBJ_SITE, f"bin_{slot}")
        mujoco.mj_forward(w.m, w.d)
        rim = w.d.site_xpos[sid].copy()
        out[slot] = []
        for k in range(int(rng.integers(0, n_max + 1))):
            sku = str(rng.choice(list(C.SMALL_SKUS)))
            b = w.free_part(sku)
            xy = rim[:2] + rng.uniform(-0.025, 0.025, 2)
            w.spawn(b, [xy[0], xy[1], rim[2] + 0.03], float(rng.uniform(0, 2 * np.pi)), stuck=False)
            a = w.part_q[b]
            ax = rng.normal(size=3)
            ax /= np.linalg.norm(ax)
            ang = float(rng.uniform(0, np.pi))
            w.d.qpos[a + 3:a + 7] = [np.cos(ang / 2), *(np.sin(ang / 2) * ax)]
            w.step(400)
            if part_in_bin(w, b, slot):
                out[slot].append(sku)
            else:
                w.park(b)
    mujoco.mj_forward(w.m, w.d)
    return out


def inject_faults(w, printer, plate, placed, rng):
    """Stress test for the rake fallback: one part tipped onto its side (the camera no longer matches it) and one
    unknown lump of debris (e.g. a broken-off failed print) at a free spot on the bed."""
    import mujoco
    from farm import catalog as C
    out = {}
    cands = sorted(n for n in placed if not n.endswith("purge_tab"))
    if cands:
        name = cands[int(rng.integers(len(cands)))]
        b = placed[name]
        a = w.part_q[b]
        k = C.CATALOG[w.part_sku[b]]
        tip = np.zeros(4)
        mujoco.mju_mulQuat(tip, w.d.qpos[a + 3:a + 7].copy(), np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0, 0]))
        w.d.qpos[a + 3:a + 7] = tip
        w.d.qpos[a + 2] += max(max(k.footprint), k.height) / 2000 + 0.003
        out["tipped"] = name
    taken = [o["center_mm"] for o in plate["objects"]] + [plate["purge_tab"]["center_mm"]]
    for _ in range(200):
        xy = rng.uniform([25, 40], [231, 236])
        if all(np.hypot(*(xy - np.array(c))) > 45 for c in taken):
            sku = str(rng.choice(["cube", "block", "knob", "gear"]))
            b = w.free_part(sku)
            w.spawn(b, C.bed_mm_to_world(printer, *xy), float(rng.uniform(0, np.pi)), stuck=False)
            out["debris"] = {"body": b, "sku": sku, "at_mm": [round(float(v), 1) for v in xy]}
            break
    mujoco.mj_forward(w.m, w.d)
    return out


def summarize(results):
    out = collections.Counter()
    by_sku = collections.defaultdict(collections.Counter)
    for r in results:
        out["episodes"] += 1
        out["door_open_ok"] += bool(r.get("door_open", {}).get("ok"))
        out["door_close_ok"] += bool(r.get("door_close", {}).get("ok"))
        out["errors"] += "error" in r
        for p in r["parts"]:
            out["parts"] += 1
            out[f"outcome:{p['outcome']}"] += 1
            out["blocking"] += bool(p.get("blocking"))
            bel = p.get("believed_placed", p["outcome"] == "placed")
            tru = p.get("truly_placed", p["outcome"] == "placed")
            out["believed_placed"] += bel
            out["truly_placed"] += tru
            out["false_success"] += bel and not tru          # robot says placed, part is not in its bin
            out["missed_success"] += tru and not bel         # part is in its bin, robot thinks it failed
            out["in_tray"] += bool(p.get("in_tray"))
            if not tru and "end" in p:
                out[f"end:{p['end']}"] += 1                  # where the parts that are not in their bin are (truth)
            for dr in p.get("drops", []):                    # every release over a bin (retries included)
                out["drops"] += 1
                out["drops_shaken"] += dr.get("shakes", 0) > 0
                out["drops_stuck_first_check"] += bool(dr.get("n_between")) and dr["n_between"][0] >= 25
                if "truly_in_bin" in dr:
                    out["drops_not_in_bin"] += not dr["truly_in_bin"]
                bc = dr.get("bin_check")
                if bc:
                    out[f"bin_check:{bc.get('verdict')}"] += 1
                    out["bin_check_pushes"] += bc.get("pushes", 0)
            by_sku[p.get("sku", p["name"].split("_")[1] if "_" in p["name"] else "?")][p["outcome"]] += 1
    n = max(out["parts"], 1)
    by_size = {}
    for size in ("small", "large"):              # purge tabs are neither
        ps = [p for r in results for p in r["parts"] if not p["name"].endswith("purge_tab")
              and bool(p.get("large")) == (size == "large")]
        tru = sum(p.get("truly_placed", p["outcome"] == "placed") for p in ps)
        bel = sum(p["outcome"] == "placed" for p in ps)
        fs = sum(p["outcome"] == "placed" and not p.get("truly_placed", True) for p in ps)
        by_size[size] = {"parts": len(ps), "truly_placed": tru, "believed_placed": bel, "false_success": fs,
                         "placed_rate": tru / max(len(ps), 1)}
        off = np.array([p["slot_offset_mm"] for p in ps if "slot_offset_mm" in p]).reshape(-1, 2)
        if len(off):                             # lowered placement: part centre vs slot centre (x along, y across)
            yaw = np.abs([p["slot_yaw_deg"] for p in ps if "slot_offset_mm" in p])
            by_size[size].update(slot_offset_y_mm_mean=float(np.mean(np.abs(off[:, 1]))),
                                 slot_offset_y_mm_max=float(np.max(np.abs(off[:, 1]))),
                                 slot_offset_x_mm_mean=float(np.mean(np.abs(off[:, 0]))),
                                 slot_offset_x_mm_max=float(np.max(np.abs(off[:, 0]))),
                                 slot_yaw_deg_mean=float(np.mean(yaw)), slot_yaw_deg_max=float(np.max(yaw)))
    out["station_visits"] = sum(r.get("station_visits", 0) for r in results)
    for r in results:
        for k, v in (r.get("blob_drops") or {}).items():
            out[f"blob_drop:{k}"] += v              # robot: blob picks confirmed in the reject bin / missed / stuck
        for e in r.get("parked_off") or []:
            out[f"parked_off:{e['where']}"] += 1    # truth: a failed part lay on the deck / floor / in the chamber
    for r in results:                               # attempts (batches from this version on): scans, picks, rakes
        if "attempts" not in r:
            continue
        att = r["attempts"]
        picks = [a for a in att if a["kind"] == "pick"]
        out["pick_attempts"] += len(picks)          # every pick_and_place call, retries included
        out["pick_retries"] += len(picks) - len({a["name"] for a in picks})
        out["parts_never_attempted"] += sum(p.get("pick_attempts", 0) == 0 for p in r["parts"])
        for a in picks:
            out[f"pick_attempt:{a['outcome']}"] += 1
        scans = [a for a in att if a["kind"] == "scan"]
        out["bed_scans"] += len(scans)              # chamber scans (a failed frame retried inside one scan)
        out["bed_scan_frame_retries"] += sum(a.get("tries", 1) - 1 for a in scans)
        rakes = [a for a in att if a["kind"] == "rake"]
        out["rake_calls"] += len(rakes)
        out["rake_calls_ok"] += sum(bool(a.get("ok")) for a in rakes)
        out["rake_strokes"] += sum(len(a.get("strokes") or []) for a in rakes)   # pushes, rakes and blob picks
    for r in results:                               # carriage moves: slide pose, times, contacts (reporting)
        mo = r.get("motion") or {}
        out["carriage_moves"] += mo.get("carriage_moves", 0)
        out["folds_low_tuck"] += mo.get("folds_low_tuck", 0)
        out["move_contact_ticks"] += mo.get("contact_ticks", 0)
        for k, v in (mo.get("by_pose") or {}).items():
            out[f"move_pose:{k}"] += v
    out["transfer_s"] = round(sum((r.get("motion") or {}).get("transfer_s", 0.0) for r in results), 2)
    out["bed_not_empty"] = sum(not r.get("bed_empty_after", True) for r in results)
    out["beds_blocked"] = sum(any(p.get("blocking") for p in r["parts"]) for r in results)
    deb = [r["faults"]["debris"] for r in results if r.get("faults", {}).get("debris")]
    if deb:
        out["debris"] = len(deb)
        out["debris_in_tray"] = sum(d.get("in_tray", False) for d in deb)
        out["debris_left_on_bed"] = sum(d.get("on_bed", True) for d in deb)
        for d in deb:
            if "end" in d:
                out[f"debris_end:{d['end']}"] += 1
    out["beds_truly_clear"] = sum(bool(r.get("door_open", {}).get("ok")) and "error" not in r
                                  and not any(p.get("blocking") for p in r["parts"])
                                  and not r.get("faults", {}).get("debris", {}).get("on_bed", False) for r in results)
    return {"counts": dict(out),
            "placed_rate": out["truly_placed"] / n,                 # truth
            "believed_placed_rate": out["believed_placed"] / n,     # what the robot reports
            "false_success_rate": out["false_success"] / max(out["believed_placed"], 1),
            "blocking_rate": out["blocking"] / n,
            "by_size": by_size,
            "by_sku": {k: dict(v) for k, v in by_sku.items()},
            "mean_sim_s": float(np.mean([r["sim_s"] for r in results])),
            "sim_s_per_part": float(sum(r["sim_s"] for r in results) / n),
            "mean_s_per_placed_part": float(np.mean([p["s"] for r in results for p in r["parts"]
                                                     if p["outcome"] == "placed"] or [0]))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--videos", type=int, default=3)
    ap.add_argument("--printer", type=int, default=4)
    ap.add_argument("--start", type=int, default=0, help="first seed")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--faults", action="store_true", help="tip one part over and drop debris on every bed")
    ap.add_argument("--realism", default="ideal", choices=["ideal", "realistic", "harsh"],
                    help="actuation realism (farm/sensors.py) + chamber-camera noise preset of the same name")
    ap.add_argument("--camera", default=None, help="override the chamber-camera noise preset")
    ap.add_argument("--oracle", action="store_true", help="legacy: plan and check with simulator truth")
    ap.add_argument("--no-wrist", action="store_true", help="disable the wrist-camera alignment")
    ap.add_argument("--no-side-push", action="store_true", help="rake only toward the front edge")
    ap.add_argument("--skus", default="small", choices=["small", "mixed", "large"],
                    help="part mix (jobs.plates.sample_parts): small = the legacy plates; mixed / large need two rows")
    ap.add_argument("--tote-fill", type=int, default=0,
                    help="stress the drop: every order tote starts with 0..N small parts in it (the farm's piles)")
    ap.add_argument("--max-print-h", type=float, default=3.0,
                    help="plate print-time cap (default 3 h, the farm's MAX_PLATE_S: a large part and a few small "
                         "ones; 8 h = dense stress plates, up to three large parts with small ones packed behind)")
    a = ap.parse_args()
    from farm import catalog as C
    if a.skus != "small" and not C.LARGE_SLOTS:
        ap.error(f"--skus {a.skus} needs the large-part tray (PRINTFARM_LAYOUT=two_rows)")
    FAULTS = a.faults
    OUT = pathlib.Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    opts = {"realism": a.realism, "camera": a.camera or a.realism, "oracle": a.oracle, "no_wrist": a.no_wrist,
            "no_side_push": a.no_side_push, "skus": a.skus, "max_print_h": a.max_print_h, "tote_fill": a.tote_fill}
    jobs = [(s, s - a.start < a.videos, a.printer, str(OUT), a.faults, opts) for s in range(a.start, a.start + a.n)]
    results = []
    with mp.Pool(a.procs, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(episode, jobs):
            results.append(r)
            ok = sum(p.get("truly_placed", p["outcome"] == "placed") for p in r["parts"])
            bel = sum(p["outcome"] == "placed" for p in r["parts"])
            lg = [p for p in r["parts"] if p.get("large")]
            lgs = f" large {sum(p['truly_placed'] for p in lg)}/{len(lg)}" if lg else ""
            print(f"seed {r['seed']:3d}: placed {ok}/{len(r['parts'])}{lgs} (robot says {bel}) door "
                  f"{r.get('door_open', {}).get('ok')}/{r.get('door_close', {}).get('ok')} "
                  f"{'ERROR' if 'error' in r else ''} wall {r['wall_s']:.0f}s", flush=True)
            (OUT / "summary.json").write_text(json.dumps(summarize(results), indent=1))
    print(json.dumps(summarize(results), indent=1))
