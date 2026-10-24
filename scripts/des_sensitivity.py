"""Robot cell vs human baseline as the per-part failure rate grows, with and without the rake fallback.
Without the rake every failure blocks its printer (out of shift: until the next operator check); with it, the measured
share of failures that the rake clears (fault batch) is reprinted instead.

Station work (bed to box) uses the measured step times, runs/station_times.json (fleet.des.load_station_times).

python3 scripts/des_sensitivity.py [runs/eval_faults] [--skus all]
    -> runs/des_box/sensitivity[_all].json + table (today's cell, farm.catalog: N_PRINTERS printers)
python3 scripts/des_sensitivity.py --layout facing --station center --n-printers 30 --order-scale 3
    -> runs/layout/sensitivity_facing_center_30.json (a larger cell)
The human baseline keeps one operator per 10 printers: it is run on a 10-printer row at the same demand per printer,
and its per-printer KPIs are compared.
"""
import argparse
import json
import pathlib

import numpy as np

from farm.catalog import N_PRINTERS
from fleet.des import SKU_MIXES, Params, Sim, load_station_times, load_timings, sku_mix

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("src", nargs="?", default="runs/eval_faults", help="fault batch dir (seed_*.json)")
ap.add_argument("--layout", default="default", help="default | row | facing (fleet.des.make_layout)")
ap.add_argument("--station", default="end")
ap.add_argument("--n-printers", type=int, default=N_PRINTERS, help="printers per arm (default: today's cell)")
ap.add_argument("--order-scale", type=float, default=None, help="order rate x (default: n_printers / 10)")
ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--skus", default="small", choices=SKU_MIXES, help="order SKU mix (fleet.des.sku_mix)")
ap.add_argument("--out", default=None)
a = ap.parse_args()

src = pathlib.Path(a.src)
eps = [json.loads(f.read_text()) for f in sorted(src.glob("seed_*.json"))]
failed = [p for e in eps for p in e["parts"] if p["outcome"] != "placed"]
assert failed, f"no failed parts in {src}/seed_*.json: run the fault batch first (scripts/eval_unload.py --faults)"
cleared = sum(not p.get("blocking") for p in failed) / max(len(failed), 1)   # share of failures no longer on the bed
base, src_t = load_timings()
station, src_s = load_station_times()
skus, excluded = sku_mix(a.skus)
SEEDS, KEYS = range(a.seeds), ("printer_util_pct", "parts_per_day", "lead_time_p90_h", "printer_blocked_h",
                               "interventions")
scale = a.n_printers / 10 if a.order_scale is None else a.order_scale
cell = dict(layout=a.layout, n_printers=a.n_printers, station=a.station, order_scale=scale, order_skus=skus)


def run(policy, p_fail, rake, **kw):
    t = dict(base, p_intervention_per_part=p_fail * (1 - cleared * rake), p_reprint_per_part=p_fail * cleared * rake)
    ks = []
    for s in SEEDS:
        sim = Sim(Params(days=7, policy=policy, seed=s, **kw), timings=t, station=station)
        sim.run()
        ks.append(sim.kpis())
    out = {k: round(float(np.mean([x[k] for x in ks])), 1) for k in KEYS}
    out["parts_per_printer_day"] = round(float(np.mean([x["parts_per_printer_day"] for x in ks])), 2)
    if policy != "human":
        out["station_busy_pct"] = round(float(np.mean([x["station_busy_pct"] for x in ks])), 1)
    return out


human = run("human", 0.0, 0, layout="row", n_printers=10, order_scale=scale * 10 / a.n_printers,
            order_skus=skus)                                               # one operator per 10 printers
rows = []
for p_fail in (0.0, 0.01, 0.02, 0.05, 0.10, 0.20):
    for rake in (0, 1):
        rows.append({"p_fail": p_fail, "rake": bool(rake), **run("order_aware", p_fail, rake, **cell)})
out = {"timings": src_t, "station_times": src_s, "skus": a.skus, "skus_excluded": list(excluded),
       "rake_clears_share": round(cleared, 3), "fault_batch": str(src), "cell": cell, "human": human, "rows": rows}
default = a.layout == "default" and a.n_printers == N_PRINTERS
tag = "" if a.skus == "small" else f"_{a.skus}"
path = pathlib.Path(a.out or (f"runs/des_box/sensitivity{tag}.json" if default else
                              f"runs/layout/sensitivity_{a.layout}_{a.station}_{a.n_printers}{tag}.json"))
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(out, indent=1))
print(f"rake clears {100 * cleared:.0f} % of failed parts off the bed (from {src}, {len(failed)} failures); cell {cell}")
keys = (*KEYS, "parts_per_printer_day", "station_busy_pct")
print(f"{'fail/part':>9} {'rake':>5} " + " ".join(f"{k:>18}" for k in keys))
print(f"{'human/10':>15} " + " ".join(f"{human.get(k, '-'):>18}" for k in keys))
for r in rows:
    print(f"{r['p_fail']:>9.2f} {str(r['rake']):>5} " + " ".join(f"{r[k]:>18}" for k in keys))
print("wrote", path)
