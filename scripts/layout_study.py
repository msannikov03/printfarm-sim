"""Printers per arm: how many printers can one rail-mounted arm serve, and in which layout? (DES sweep)

Layouts (fleet.des.make_layout): one row along a wall or two facing rows with the rail in the middle, the hand-off
station at the end of the rail or mid-rail. Robot timings: runs/timings.json (measured in MuJoCo).
Demand modes, as multiples of the 10-printer base order stream (jobs/orders.py, ASSUMED rates), scaled with N:
  cap   1.5 x N/10: more orders than the printers can print, so the queue never empties (capacity)
  prop  1.0 x N/10: every printer sees today's demand per printer (lead times)
Knee (written to des_study.json):
  cap   the largest N at which the printers lose <= 5 % of their hours waiting for the arm (hours blocked by a
        stuck part outside shifts wait for a person, not the arm, and are not counted)
  prop  the largest N at which the order lead time p90 stays within 1.25 x the 10-printer value

    python scripts/layout_study.py --procs 2                       -> runs/layout/des_runs.jsonl + des_study.json
    python scripts/layout_study.py --sens --layout facing --station center --procs 2   (sensitivities, cap mode)
"""
import argparse
import itertools
import json
import multiprocessing as mp
import pathlib
import time

import numpy as np

OUT = pathlib.Path("runs/layout")
N_LIST = (10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70)
LAYOUTS = (("row", "end"), ("row", "center"), ("facing", "end"), ("facing", "center"))
MODES = {"cap": 1.5, "prop": 1.0}
KEYS = ("printer_util_pct", "printer_wait_pct", "printer_blocked_h", "parts_per_day", "parts_per_printer_day",
        "lead_time_p50_h", "lead_time_p90_h", "on_time_pct", "robot_util_pct", "rail_m", "station_visits",
        "caddy_full_events", "partial_handoffs", "interventions", "robot_reprints", "rail_length_m")
WAIT_MAX_PCT, LEAD_MAX_X = 5.0, 1.25
SENS_N = (20, 30, 40, 50, 60)


def rake_share(src=pathlib.Path("runs/eval_faults")):
    """Share of failed parts the rake takes off the bed (measured fault batch), as scripts/des_sensitivity.py."""
    eps = [json.loads(f.read_text()) for f in sorted(src.glob("seed_*.json"))]
    failed = [p for e in eps for p in e["parts"] if p["outcome"] != "placed"]
    if not failed:
        return 0.789, "fallback 0.789 (fault batch of 2026-10-22; runs/eval_faults missing)"
    return sum(not p.get("blocking") for p in failed) / len(failed), f"{src} ({len(failed)} failed parts)"


def sens_variants():
    """name -> (Params overrides, timings overrides). All ASSUMED what-ifs around the measured baseline."""
    share, _ = rake_share()
    p = 0.02
    return {
        "base": ({}, {}),
        "plate_1.5h": ({"max_plate_s": 1.5 * 3600}, {}),
        "plate_6h": ({"max_plate_s": 6 * 3600}, {}),
        "arm_2x_faster": ({"robot_slowdown": 0.5}, {}),
        "caddy_12_bins": ({"order_slots": 12}, {}),
        "rail_1.0_mps": ({}, {"rail_speed_mps": 1.0}),
        "fail_2pct_rake": ({}, {"p_intervention_per_part": p * (1 - share), "p_reprint_per_part": p * share}),
    }


def run_one(task):
    from fleet.des import Params, Sim, load_timings
    tag, layout, station, n, mode, seed, days, p_over, t_over = task
    T, src = load_timings(t_over.pop("_path", "runs/timings.json"))
    T.update(t_over)
    policy = "human" if tag == "human" else "order_aware"
    kw = dict(days=days, policy=policy, seed=seed, layout=layout, n_printers=n, station=station,
              order_scale=MODES[mode] * n / 10)
    kw.update(p_over)
    t0 = time.time()
    sim = Sim(Params(**kw), timings=T)
    k = sim.run()
    return {"tag": tag, "layout": layout, "station": station, "n": n, "mode": mode, "seed": seed,
            "timings": src, "wall_s": round(time.time() - t0, 1), **{key: k.get(key) for key in KEYS}}


def aggregate(rows, days):
    groups = {}
    for r in rows:
        groups.setdefault((r["tag"], r["layout"], r["station"], r["n"], r["mode"]), []).append(r)
    out = []
    for (tag, layout, station, n, mode), rs in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0][:3])) + kv[0][3:]):
        row = {"tag": tag, "layout": layout, "station": station, "n": n, "mode": mode, "seeds": len(rs)}
        for key in KEYS:
            v = [r[key] for r in rs if r[key] is not None]
            if v:
                row[key] = round(float(np.mean(v)), 2)
                row[key + "_sd"] = round(float(np.std(v, ddof=1)), 2) if len(v) > 1 else 0.0
        out.append(arm_wait(row, days))
    return out


def arm_wait(row, days):
    """Printer time lost waiting for the arm: waiting minus the hours blocked by a stuck part out of shift (those
    wait for a person, not for the arm), % of printer-hours."""
    blocked_pct = 100 * row.get("printer_blocked_h", 0.0) / (row["n"] * days * 24)
    row["arm_wait_pct"] = round(max(0.0, row["printer_wait_pct"] - blocked_pct), 2)
    return row


def knees(agg):
    """Per (tag, layout, station): knee N by the two criteria (see module docstring)."""
    res = {}
    keys = sorted({(a["tag"], a["layout"], a["station"]) for a in agg})
    for tag, layout, station in keys:
        cap = sorted((a for a in agg if (a["tag"], a["layout"], a["station"], a["mode"]) == (tag, layout, station, "cap")),
                     key=lambda a: a["n"])
        prop = sorted((a for a in agg if (a["tag"], a["layout"], a["station"], a["mode"]) == (tag, layout, station, "prop")),
                      key=lambda a: a["n"])
        k = {}
        ok = [a["n"] for a in cap if a.get("arm_wait_pct", 1e9) <= WAIT_MAX_PCT]
        k["cap_knee_n"] = max(ok) if ok else None
        if cap:
            best = max(cap, key=lambda a: a["parts_per_day"])
            k["cap_max_parts_per_day"] = best["parts_per_day"]
            k["cap_max_at_n"] = best["n"]
        if prop:
            ref = prop[0]["lead_time_p90_h"]
            ok = [a["n"] for a in prop if a["lead_time_p90_h"] <= LEAD_MAX_X * ref]
            k["prop_knee_n"] = max(ok) if ok else None
            k["prop_ref_lead_p90_h"] = ref
        res[f"{tag}/{layout}/{station}"] = k
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n", type=int, nargs="*", default=list(N_LIST))
    ap.add_argument("--sens", action="store_true", help="sensitivity variants on one layout (cap mode)")
    ap.add_argument("--layout", default="facing")
    ap.add_argument("--station", default="center")
    ap.add_argument("--timings", help="--sens: also run the base variant with these measured timings (e.g. "
                                      "runs/layout/timings_<layout>.json from the two-row episodes)")
    ap.add_argument("--recompute", action="store_true", help="re-aggregate the raw runs file without simulating")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = range(a.seeds)
    if a.sens:
        variants = sens_variants()
        if a.timings:
            variants["measured_" + pathlib.Path(a.timings).stem] = ({}, {"_path": a.timings})
        tasks = [(tag, a.layout, a.station, n, "cap", s, a.days, po, dict(to))
                 for tag, (po, to) in variants.items() for n in SENS_N for s in seeds]
        raw, out = OUT / "des_sens_runs.jsonl", OUT / "des_sens.json"
    else:
        tasks = [("base", lay, st, n, mode, s, a.days, {}, {})
                 for (lay, st), n, mode, s in itertools.product(LAYOUTS, a.n, MODES, seeds)]
        tasks += [("human", "row", "end", 10, mode, s, a.days, {}, {}) for mode in MODES for s in seeds]
        raw, out = OUT / "des_runs.jsonl", OUT / "des_study.json"
    tasks.sort(key=lambda t: -t[3])           # big cells first (longest), better packing over the pool
    rows, t0 = [], time.time()
    if a.recompute:                           # re-aggregate the raw runs (no new simulations)
        rows = [json.loads(line) for line in raw.read_text().splitlines()]
    else:
        raw.write_text("")
        with mp.get_context("spawn").Pool(a.procs) as pool:
            for r in pool.imap_unordered(run_one, tasks):
                rows.append(r)
                with raw.open("a") as f:
                    f.write(json.dumps(r) + "\n")
                if len(rows) % 20 == 0:
                    print(f"{len(rows)}/{len(tasks)} runs, {time.time() - t0:.0f} s", flush=True)
    agg = aggregate(rows, a.days)
    share, share_src = rake_share()
    res = {"note": "DES (fleet/des.py), order_aware dispatch, robot timings from runs/timings.json; order rates and "
                   "every non-robot duration are ASSUMPTIONS. mean (and sd) over seeds.",
           "days": a.days, "seeds": list(seeds), "modes": MODES,
           "knee_rules": {"cap": f"arm_wait_pct (printer waiting minus blocked-by-stuck-part hours) <= {WAIT_MAX_PCT}",
                          "prop": f"lead_time_p90_h <= {LEAD_MAX_X} x N=10"},
           "rake_share": round(share, 3), "rake_share_source": share_src,
           "variants": {k: {"params": v[0], "timings": v[1]} for k, v in variants.items()} if a.sens else None,
           "rows": agg, "knees": knees(agg)}
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res["knees"], indent=1))
    print(f"wrote {out} ({len(rows)} runs, {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
