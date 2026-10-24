"""Monte Carlo evaluation of chamber-camera perception under realistic sensor errors (perceive/noise.py).

For every configuration (noise preset x corrections on/off, plus injected camera faults) the SAME random scenes are
scanned (paired comparison): a random plate (jobs.plates.make_plate + plan.unload.mirror_plate) spawned with slicer
prior noise (1.5 mm / 2 deg, as in eval_unload), sometimes one part missing from the bed (the prior still lists it)
and sometimes one part nudged 5-12 mm (as a neighbouring pick does). Each trial also draws a new camera calibration
error (BedCamera.resample). Per trial: one plate scan, one empty-bed scan, one scan with a single leftover lump
5-15 mm tall. Ground truth (sim poses) is used only for scoring.

  PYTHONPATH=. python scripts/eval_perception.py --n 150 --procs 3      # -> runs/perception/*.json + perception.png
"""
import argparse
import json
import math
import multiprocessing as mp
import pathlib
import time

import numpy as np

OUT = pathlib.Path("runs/perception")
P = 4                                            # printer used (all row-A printers share the camera geometry)
PRIOR_POS_MM, PRIOR_YAW_DEG = 1.5, 2.0           # slicer prior noise of the spawned parts (as eval_unload)
P_MISSING, P_NUDGE = 0.25, 0.25
LEFTOVER_MM = (5.0, 8.0, 15.0)

CONFIGS = [  # name, preset, corrections (None = default for the preset), forced fault
    ("ideal", "ideal", None, None),
    ("ideal+pipeline", "ideal", True, None),
    ("realistic/naive", "realistic", False, None),
    ("realistic", "realistic", True, None),
    ("harsh/naive", "harsh", False, None),
    ("harsh", "harsh", True, None),
    ("realistic+occlusion", "realistic", True, "occlusion"),
    ("realistic+smear", "realistic", True, "smear"),
    ("realistic+dead", "realistic", True, "dead"),
    ("realistic+blackout", "realistic", True, "blackout"),
]


def _scene(rng, w, C, make_plate, mirror_plate, spawn_plate, mask, skus="small"):
    from jobs.plates import sample_parts
    parts = sample_parts(rng, skus)
    cap = {} if skus == "small" else {"max_print_s": 8 * 3600.0}       # as eval_unload: denser large-part plates
    plate = mirror_plate(make_plate(parts, P, "PLMC", rng, reach_mask=mask, **cap)[0])
    missing = None
    objs = plate["objects"]
    if objs and rng.random() < P_MISSING:
        missing = objs[int(rng.integers(len(objs)))]["name"]
    spawn = dict(plate, objects=[o for o in objs if o["name"] != missing])
    placed = spawn_plate(w, spawn, rng, pos_sigma_mm=PRIOR_POS_MM, yaw_sigma_deg=PRIOR_YAW_DEG, stuck=True)
    nudged = None
    cands = [o for o in spawn["objects"]]
    if cands and rng.random() < P_NUDGE:
        o = cands[int(rng.integers(len(cands)))]
        b = placed[o["name"]]
        p, q = w.part_pose(b)
        a = rng.uniform(0, 2 * math.pi)
        dist = rng.uniform(5.0, 12.0)
        xc, s = C.printer_frame(P)
        yaw = 2 * math.atan2(q[3], q[0])
        w.spawn(b, (p[0] + s * dist / 1000 * math.cos(a), p[1] + s * dist / 1000 * math.sin(a), p[2]), yaw, stuck=True)
        nudged = o["name"]
    return plate, placed, missing, nudged


def _truth(w, C, body):
    p, q = w.part_pose(body)
    xc, s = C.printer_frame(P)
    tx = s * (p[0] - xc) * 1000 + C.BED_MM[0] / 2
    ty = (s * p[1] - C.AISLE_HALF - C.BED_FRONT_GAP) * 1000
    tyaw = math.degrees(2 * math.atan2(q[3], q[0])) + (0.0 if s > 0 else 180.0)
    return tx, ty, tyaw


def run_config(args):
    name, preset, corrections, force, n, seed, skus = (*args, "small")[:7]
    import mujoco
    from farm import catalog as C
    from farm.world import World
    from jobs.plates import make_plate
    from perceive.verify import BedCamera, bed_empty, estimate, plate_objects, _yaw_observable
    from plan.unload import mirror_plate, spawn_plate
    mask = None
    if pathlib.Path("runs/bedmap.json").exists():
        from plan.bedmap import load_mask
        mask = load_mask("runs/bedmap.json", mirror=True)
    w = World()
    xc, s = C.printer_frame(P)
    cam = BedCamera(w, P, noise=preset, rng=np.random.default_rng(seed), corrections=corrections)
    cam.force = force
    recs, beds, t_scan, quality, errors = [], [], [], {}, 0
    for t in range(n):
        rng = np.random.default_rng((seed, t))                  # scene: identical across configurations
        cam.rng = np.random.default_rng((seed, t, 7))           # camera: new calibration + noise per trial
        cam.resample(cam.rng)
        w.reset(rail_x=xc + 0.4)
        plate, placed, missing, nudged = _scene(rng, w, C, make_plate, mirror_plate, spawn_plate, mask, skus)
        mujoco.mj_forward(w.m, w.d)
        objs = plate_objects(plate)
        t0 = time.perf_counter()
        try:
            pts = cam.points_bed_mm()
            est, unknown = estimate(pts, objs)
        except Exception as e:                                  # the contract: never crash
            errors += 1
            print(f"[{name}] trial {t}: {type(e).__name__}: {e}", flush=True)
            continue
        t_scan.append(time.perf_counter() - t0)
        q = (cam.last or {}).get("quality", "ok")
        quality[q] = quality.get(q, 0) + 1
        for o in objs:
            e = est[o["name"]]
            present = o["name"] != missing
            r = {"t": t, "sku": o["sku"], "large": o["sku"] in C.LARGE_SKUS, "present": present, "found": bool(e["found"]),
                 "nudged": o["name"] == nudged, "quality": e.get("quality"), "observed": e.get("observed", True),
                 "sigma_mm": e.get("sigma_mm"), "prior_rejected": bool(e.get("prior_rejected", False))}
            if present:
                tx, ty, tyaw = _truth(w, C, placed[o["name"]])
                ex, ey = e["center_mm"][0] - tx, e["center_mm"][1] - ty
                r.update(err=float(math.hypot(ex, ey)), ex=float(ex), ey=float(ey),
                         prior_err=float(math.hypot(o["center_mm"][0] - tx, o["center_mm"][1] - ty)))
                cov = np.asarray(e["cov_mm2"])
                r["nees"] = float(np.array([ex, ey]) @ np.linalg.solve(cov, np.array([ex, ey])))
                if _yaw_observable(o) and o["sku"] != "purge_tab":
                    k = C.CATALOG[o["sku"]]
                    per = 360.0 if k.kind == "L" else 180.0
                    r["yaw_err"] = float(abs((e["yaw_deg"] - tyaw + per / 2) % per - per / 2))
                    r["yaw_sigma"] = e.get("yaw_sigma_deg")
            recs.append(r)
        # empty bed
        for b in list(w.in_use):
            w.park(b)
        mujoco.mj_forward(w.m, w.d)
        pe = cam.points_bed_mm()
        beds.append({"t": t, "kind": "empty", "empty": bool(bed_empty(pe)), "q": (cam.last or {}).get("quality")})
        # one leftover lump, sunk so that hp mm stick out of the bed (a broken-off print / blob)
        hp = float(LEFTOVER_MM[t % len(LEFTOVER_MM)])
        sku = str(rng.choice(["cube", "block", "knob", "standoff", "spacer"]))
        b = w.free_part(sku)
        x, y = rng.uniform(25, 231), rng.uniform(30, 236)
        wx, wy, wz = C.bed_mm_to_world(P, x, y)
        w.spawn(b, (wx, wy, wz - (C.CATALOG[sku].height - hp) / 1000), float(rng.uniform(0, math.pi)), stuck=True)
        mujoco.mj_forward(w.m, w.d)
        pl = cam.points_bed_mm()
        beds.append({"t": t, "kind": "leftover", "h_mm": hp, "sku": sku, "at": [round(x, 1), round(y, 1)],
                     "empty": bool(bed_empty(pl)), "q": (cam.last or {}).get("quality")})
        w.park(b)
    cam.close()
    return {"name": name, "preset": preset, "corrections": corrections, "force": force, "n": n, "records": recs,
            "beds": beds, "t_scan_s": t_scan, "quality": quality, "exceptions": errors}


def edge_bias():
    """Per-edge read-out bias of the bed-outline fit on the perfect render of the empty bed (-> EDGE_BIAS_MM)."""
    import mujoco
    from farm import catalog as C
    from farm.world import World
    from perceive import verify as V
    w = World()
    xc, s = C.printer_frame(P)
    w.reset(rail_x=xc + 0.4)
    mujoco.mj_forward(w.m, w.d)
    saved = dict(V.EDGE_BIAS_MM)
    V.EDGE_BIAS_MM.update({k: 0.0 for k in saved})
    cam = V.BedCamera(w, P, corrections=True)
    pts, meta = cam.scan()
    raw = np.asarray(cam._points_legacy())
    ol = V.fit_bed_outline(raw, 0.0, np.random.default_rng(0))
    V.EDGE_BIAS_MM.update(saved)
    nominal = {"left": 0.0, "right": C.BED_MM[0], "front": 0.0, "back": C.BED_MM[1]}
    out = {k: round(e["a"] - nominal[k], 3) for k, e in ol["edges"].items()}
    print("measured edge read-out (mm, + = toward +x/+y):", out, "| in use:", saved)
    cam.close()
    return out


def markdown_table(summary_path):
    """The results table of docs/perception.md from runs/perception/summary.json."""
    d = json.loads(pathlib.Path(summary_path).read_text())
    rows = [("config", "trials", "found", "centre mean / p95 (mm)", "yaw mean (deg)", "NEES (2 = honest)",
             "in 95 % ellipse", "missing detected", "false empty", "false not empty", "ms / scan")]

    def pct(v):
        return "-" if v is None else f"{100 * v:.0f} %"

    for c in d["configs"]:
        rows.append((c["config"], str(c["trials"]), pct(c["found_rate"]),
                     f"{c['centre_err_mean_mm']:.2f} / {c['centre_err_p95_mm']:.2f}", f"{c['yaw_err_mean_deg']:.2f}",
                     "-" if c["nees_mean"] is None else f"{c['nees_mean']:.2f}", pct(c["within_2d_95"]),
                     pct(c["missing_detected"]), pct(c["false_empty_rate"]), pct(c["false_not_empty_rate"]),
                     "-" if c["scan_runtime_ms_mean"] is None else f"{c['scan_runtime_ms_mean']:.0f}"))
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def summarize(res):
    recs = res["records"]
    pres = [r for r in recs if r["present"]]
    found = [r for r in pres if r["found"]]
    seen = [r for r in found if r.get("observed", True)]
    err = np.array([r["err"] for r in found]) if found else np.array([np.nan])
    prior = np.array([r["prior_err"] for r in found]) if found else np.array([np.nan])
    yaw = np.array([r["yaw_err"] for r in found if "yaw_err" in r] or [np.nan])
    nees = np.array([r["nees"] for r in found if r.get("nees") is not None])
    sig = np.array([r["sigma_mm"] for r in found if r.get("sigma_mm") is not None])
    absent = [r for r in recs if not r["present"]]
    nud = [r for r in found if r["nudged"]]
    beds = res["beds"]
    empt = [b for b in beds if b["kind"] == "empty"]
    left = [b for b in beds if b["kind"] == "leftover"]
    by_h = {}
    for b in left:
        by_h.setdefault(str(b["h_mm"]), []).append(b["empty"])
    out = {
        "config": res["name"], "trials": res["n"], "objects_present": len(pres), "exceptions": res["exceptions"],
        "found_rate": len(found) / max(1, len(pres)),
        "observed_rate": len(seen) / max(1, len(pres)),
        "centre_err_mean_mm": float(np.nanmean(err)), "centre_err_p95_mm": float(np.nanpercentile(err, 95)),
        "centre_err_max_mm": float(np.nanmax(err)),
        "prior_err_mean_mm": float(np.nanmean(prior)),
        "yaw_err_mean_deg": float(np.nanmean(yaw)), "yaw_err_p95_deg": float(np.nanpercentile(yaw, 95)),
        "nudged_err_mean_mm": float(np.mean([r["err"] for r in nud])) if nud else None,
        "sigma_mean_mm": float(np.mean(sig)) if len(sig) else None,
        "nees_mean": float(np.mean(nees)) if len(nees) else None,       # 2.0 for a consistent 2-D covariance
        "within_2d_1sigma": float(np.mean(nees < 2.279)) if len(nees) else None,   # chi2(2) 68.3 %
        "within_2d_95": float(np.mean(nees < 5.991)) if len(nees) else None,       # chi2(2) 95 %
        "missing_detected": sum(not r["found"] for r in absent) / max(1, len(absent)) if absent else None,
        "false_found_rate": sum(r["found"] for r in absent) / max(1, len(absent)) if absent else None,
        "false_missing_rate": sum(not r["found"] for r in pres) / max(1, len(pres)),
        "false_not_empty_rate": sum(not b["empty"] for b in empt) / max(1, len(empt)),
        "false_empty_rate": sum(b["empty"] for b in left) / max(1, len(left)),
        "false_empty_by_height": {h: float(np.mean(v)) for h, v in sorted(by_h.items(), key=lambda kv: float(kv[0]))},
        "scan_quality": res["quality"],
        "scan_runtime_ms_mean": 1000 * float(np.mean(res["t_scan_s"])) if res["t_scan_s"] else None,
        "scan_runtime_ms_p95": 1000 * float(np.percentile(res["t_scan_s"], 95)) if res["t_scan_s"] else None,
    }
    lg = [r for r in found if r.get("large")]
    if lg:                                       # large SKUs (--skus mixed / large)
        le = np.array([r["err"] for r in lg])
        ly = np.array([r["yaw_err"] for r in lg if "yaw_err" in r] or [np.nan])
        ln = np.array([r["nees"] for r in lg if r.get("nees") is not None])
        out.update(large_found_rate=len(lg) / max(1, sum(r.get("large", False) for r in pres)),
                   large_centre_err_mean_mm=float(np.mean(le)), large_centre_err_p95_mm=float(np.percentile(le, 95)),
                   large_centre_err_max_mm=float(np.max(le)), large_yaw_err_mean_deg=float(np.nanmean(ly)),
                   large_yaw_err_p95_deg=float(np.nanpercentile(ly, 95)),
                   large_nees_mean=float(np.mean(ln)) if len(ln) else None, large_n=len(lg))
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


SWEEP_K = (0.5, 1.0, 1.5, 2.0, 3.0)     # realistic preset with every error magnitude / probability scaled by k


def plot(results, path):
    """(a) mean centre error naive vs corrected per preset (log scale; p95 as a tick), (b) covariance consistency,
    (c) sensitivity of the corrected pipeline to the noise level."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import chi2
    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    blue, orange, grey = "#2a78d6", "#eb6834", "#a3a29c"        # reference categorical slots 1, 2 + neutral
    by = {r["config"]: r for r in results}
    sweep = sorted((r for r in results if r["config"].startswith("realistic x")), key=lambda r: r["_k"])
    fig, ax = plt.subplots(1, 3 if sweep else 2, figsize=(15 if sweep else 10.5, 4.2), dpi=150)
    fig.patch.set_facecolor(surface)
    # (a) naive (the original estimator on raw single frames) vs the corrected pipeline
    a = ax[0]
    presets = [p for p in ("ideal", "realistic", "harsh") if p in by]
    naive = {"ideal": "ideal", "realistic": "realistic/naive", "harsh": "harsh/naive"}
    corr = {"ideal": "ideal+pipeline", "realistic": "realistic", "harsh": "harsh"}
    x = np.arange(len(presets))
    for off, names, col, lab in ((-0.2, naive, grey, "no corrections"), (0.2, corr, blue, "corrected pipeline")):
        mean = [by[names[p]]["centre_err_mean_mm"] if names[p] in by else np.nan for p in presets]
        p95 = [by[names[p]]["centre_err_p95_mm"] if names[p] in by else np.nan for p in presets]
        a.bar(x + off, mean, 0.36, color=col, label=lab, edgecolor=surface, linewidth=2)
        a.scatter(x + off, p95, marker="_", s=260, color=ink, linewidths=1.6, zorder=3)
        for xi, m, q in zip(x + off, mean, p95):
            if np.isfinite(m):
                a.text(xi, m * 1.08, f"{m:.2f}", ha="center", va="bottom", fontsize=8, color=ink)
                a.text(xi, q * 1.15, f"p95 {q:.1f}", ha="center", va="bottom", fontsize=7, color=muted)
    a.set_yscale("log")
    a.set_ylim(0.05, 300)
    a.set_xticks(x, presets)
    a.set_ylabel("part centre error, mm (log)", color=ink)
    a.set_title("Centre error by camera preset (bar = mean, tick = p95)", fontsize=10, color=ink, loc="left")
    a.legend(fontsize=8, frameon=False, loc="upper left")
    # (b)
    b = ax[1]
    xs = np.linspace(0, 12, 200)
    b.plot(xs, chi2.cdf(xs, 2), color=ink, lw=1.2, ls="--", label="consistent (chi-square, 2 dof)")
    for nm, c in (("realistic", blue), ("harsh", orange)):
        rr = by.get(nm)
        if rr is None or not rr.get("_nees"):
            continue
        v = np.sort(rr["_nees"])
        b.plot(v, np.arange(1, len(v) + 1) / len(v), color=c, lw=2, label=f"{nm} (mean {np.mean(v):.2f}; ideal 2)")
    b.set_xlim(0, 12)
    b.set_xlabel("normalised squared error  e' Sigma^-1 e", color=ink)
    b.set_ylabel("share of parts", color=ink)
    b.set_title("Is the reported covariance honest?", fontsize=10, color=ink, loc="left")
    b.legend(fontsize=8, frameon=False, loc="lower right")
    # (c)
    if sweep:
        c = ax[2]
        ks = [r["_k"] for r in sweep]
        c.plot(ks, [r["centre_err_mean_mm"] for r in sweep], color=blue, lw=2, marker="o", ms=6,
               label="centre error mean")
        c.plot(ks, [r["centre_err_p95_mm"] for r in sweep], color=orange, lw=2, marker="o", ms=6,
               label="centre error p95")
        for r in sweep:
            c.annotate(f"{r['centre_err_p95_mm']:.1f}", (r["_k"], r["centre_err_p95_mm"]), textcoords="offset points",
                       xytext=(0, 6), ha="center", fontsize=7, color=muted)
        c.set_xlabel("error magnitude, x realistic preset", color=ink)
        c.set_ylabel("mm", color=ink)
        c.set_ylim(bottom=0)
        c.set_title("Sensitivity of the corrected pipeline", fontsize=10, color=ink, loc="left")
        c.legend(fontsize=8, frameon=False, loc="upper left")
    for a_ in ax:
        a_.set_facecolor(surface)
        a_.grid(color=grid, lw=0.6)
        a_.set_axisbelow(True)
        for sp in ("top", "right"):
            a_.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            a_.spines[sp].set_color(grid)
        a_.tick_params(colors=muted, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="trials (scenes) per configuration")
    ap.add_argument("--n-faults", type=int, default=40, help="trials per injected-fault configuration")
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated configuration names")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--edge-bias", action="store_true", help="measure the outline fit's per-edge bias and exit")
    ap.add_argument("--sweep", action="store_true", help="also scale the realistic preset's errors by SWEEP_K")
    ap.add_argument("--n-sweep", type=int, default=60)
    ap.add_argument("--table", action="store_true", help="print the markdown results table from <out>/summary.json")
    ap.add_argument("--skus", default="small", choices=["small", "mixed", "large"],
                    help="part mix of the scenes (jobs.plates.sample_parts); small = the legacy scenes")
    a = ap.parse_args()
    if a.edge_bias:
        edge_bias()
        raise SystemExit
    if a.table:
        print(markdown_table(pathlib.Path(a.out) / "summary.json"))
        raise SystemExit
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    only = set(filter(None, a.only.split(",")))
    jobs = [(nm, pr, co, fo, a.n_faults if fo else a.n, a.seed, a.skus) for nm, pr, co, fo in CONFIGS
            if not only or nm in only]
    if a.sweep:
        from perceive.noise import PRESETS, scaled
        jobs += [(f"realistic x{k:g}", scaled(PRESETS["realistic"], k), True, None, a.n_sweep, a.seed, a.skus)
                 for k in SWEEP_K]
    t0 = time.time()
    with mp.get_context("spawn").Pool(min(a.procs, len(jobs))) as pool:
        raw = pool.map(run_config, jobs, chunksize=1)
    summ = []
    for r in raw:
        s = summarize(r)
        if not isinstance(r["preset"], str):
            r["preset"] = r["preset"].name
        fn = r["name"].replace("/", "_").replace("+", "_").replace(" ", "_")
        (out / f"{fn}.json").write_text(json.dumps({"summary": s, **r}))
        if r["name"].startswith("realistic x"):
            s["_k"] = float(r["name"].split("x")[-1])
        s["_nees"] = [x["nees"] for x in r["records"] if x["present"] and x["found"] and x.get("nees") is not None]
        summ.append(s)
    plot(summ, out / "perception.png")
    for s in summ:
        s.pop("_nees", None)
        s.pop("_k", None)
    (out / "summary.json").write_text(json.dumps({"seed": a.seed, "n": a.n, "wall_s": round(time.time() - t0, 1),
                                                  "configs": summ}, indent=1))
    cols = ["config", "found_rate", "centre_err_mean_mm", "centre_err_p95_mm", "yaw_err_mean_deg", "nees_mean",
            "within_2d_95", "missing_detected", "false_found_rate", "false_not_empty_rate", "false_empty_rate",
            "scan_runtime_ms_mean"]
    if a.skus != "small":
        cols += ["large_n", "large_found_rate", "large_centre_err_mean_mm", "large_centre_err_p95_mm",
                 "large_yaw_err_mean_deg", "large_nees_mean"]
    print(" | ".join(cols))
    for s in summ:
        print(" | ".join(str(s.get(c)) for c in cols))
    print(f"wall {time.time() - t0:.0f} s -> {out}")
