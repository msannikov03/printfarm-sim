"""Markdown tables for docs/layout.md, straight from the run outputs in runs/layout (no hand-copied numbers).

    python scripts/layout_tables.py [--layout facing --station center --n 40] > runs/layout/tables.md
"""
import argparse
import json
import pathlib

R = pathlib.Path("runs/layout")


def md(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---:" if i else "---" for i in range(len(header))) + "|"]
    out += ["| " + " | ".join("-" if v is None else str(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def f(v, nd=1):
    return None if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else v)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--layout", default="facing")
    ap.add_argument("--station", default="center")
    ap.add_argument("--n", type=int, default=40, help="recommended printers per arm")
    ap.add_argument("--geometry", default="facing_a40_b56_px", help="recommended geometry (layout_geometry name)")
    a = ap.parse_args()
    study = json.loads((R / "des_study.json").read_text())
    rows = [r for r in study["rows"] if r["tag"] == "base"]
    lab = {("row", "end"): "one row, station at end", ("row", "center"): "one row, station mid-rail",
           ("facing", "end"): "two facing rows, station at end", ("facing", "center"): "two facing rows, station mid-rail"}

    print("### Knee per layout\n")
    t = []
    for key, k in study["knees"].items():
        tag, lay, st = key.split("/")
        if tag != "base":
            continue
        t.append((lab[(lay, st)], k["cap_knee_n"], k["prop_knee_n"], f(k.get("cap_max_parts_per_day"), 0),
                  k.get("cap_max_at_n")))
    print(md(["layout", "knee, saturated demand (waiting for the arm <= 5 %)", "knee, lead p90 <= 1.25 x", "max parts/day per arm",
              "at N"], t))

    hum = {r["mode"]: r for r in study["rows"] if r["tag"] == "human"}
    print(f"\nHuman baseline (10 printers, one operator 09-18 Mon-Fri): saturated util "
          f"{hum['cap']['printer_util_pct']} %, {hum['cap']['parts_per_day']} parts/day; today's demand: lead p50 "
          f"{hum['prop']['lead_time_p50_h']} h / p90 {hum['prop']['lead_time_p90_h']} h.\n")

    print(f"### {lab[(a.layout, a.station)]}: printers per arm\n")
    sel = {(r["n"], r["mode"]): r for r in rows if (r["layout"], r["station"]) == (a.layout, a.station)}
    t = []
    for n in sorted({n for n, _ in sel}):
        c, p = sel.get((n, "cap")), sel.get((n, "prop"))
        t.append((n, f(c["printer_util_pct"]), f(c["arm_wait_pct"]), f(c["robot_util_pct"]), f(c["parts_per_day"], 0),
                  f(c["parts_per_printer_day"]), f(p["lead_time_p50_h"]), f(p["lead_time_p90_h"]), f(p["robot_util_pct"]),
                  f(c["rail_length_m"]), f(c["rail_m"] / study["days"], 0), f(c["caddy_full_events"], 0)))
    print(md(["N", "util %", "waiting for arm %", "arm busy %", "parts/day", "parts/printer-day", "lead p50 h (today's demand)",
              "lead p90 h", "arm busy % (today's demand)", "rail m", "rail travel m/day", "caddy-full events/wk"], t))

    print(f"\n### All layouts at saturated demand\n")
    t = []
    for n in (20, 30, a.n, 50, 60):
        for lay in lab:
            r = next((r for r in rows if (r["layout"], r["station"]) == lay and r["n"] == n and r["mode"] == "cap"), None)
            p = next((r for r in rows if (r["layout"], r["station"]) == lay and r["n"] == n and r["mode"] == "prop"), None)
            if r:
                t.append((n, lab[lay], f(r["printer_util_pct"]), f(r["arm_wait_pct"]), f(r["robot_util_pct"]),
                          f(r["parts_per_day"], 0), f(p["lead_time_p90_h"]) if p else None, f(r["rail_length_m"]),
                          f(r["rail_m"] / study["days"], 0)))
    t = list(dict.fromkeys(t))
    print(md(["N", "layout", "util %", "waiting for arm %", "arm busy %", "parts/day", "lead p90 h (today's demand)", "rail m",
              "rail travel m/day"], t))

    latest = {}                       # sweeps in run order; a later run of the same layout supersedes an earlier one
    for p in (R / "geometry_v1.jsonl", R / "geometry_v2.jsonl", R / "geometry.jsonl"):
        for line in (p.read_text().splitlines() if p.exists() else []):
            g = json.loads(line)
            if p.name == "geometry_v2.jsonl" and g["caddy"] == "plusx":
                continue              # +x caddy changed after that sweep (bins 6 cm further out, deck -x edge kept)
            latest[g["layout"]] = g
    geo = list(latest.values())
    if geo:
        print("\n### Geometry (MuJoCo planners, scripts/layout_geometry.py sweep)\n")
        t = []
        for g in sorted(geo, key=lambda g: (g["rows"], g["aisle"], g["base_z"], g["caddy"])):
            da, db = g.get("door_A", {}), g.get("door_B", {})
            ok = lambda d: None if not d else f"{'yes' if d.get('open_stages') else 'NO'} / {'yes' if d.get('close_stages') else 'NO'}"
            dA = g.get("drop_ok_A", g.get("drop_ok"))
            dB = g.get("drop_ok_B")
            t.append((g["rows"], f"{g['aisle']:.2f}", f"{g['base_z']:.2f}", g["caddy"], ok(da), ok(db) if db else "-",
                      f"{100 * g['grasp_coverage']:.0f} %" if "grasp_coverage" in g else None,
                      f"{dA}/8" if dA is not None else None, f"{dB}/8" if dB is not None else "-",
                      f"{g.get('deck_bench_gap_m')}" + (" ERROR" if "error" in g else "")))
        print(md(["rows", "aisle m", "arm base z m", "caddy", "door A open/close", "door B open/close",
                  "bed grasp coverage (16 pts x 2 probes)", "bins reachable A", "bins reachable B",
                  "deck-bench gap m"], t))

    ep = R / f"episodes_{a.geometry}.json"
    if ep.exists():
        e = json.loads(ep.read_text())
        print(f"\n### Full unload episodes on the two-row model ({a.geometry})\n")
        t = []
        for name, r in [*e["printers"].items(), ("both rows", e["all"]), ("one row, same seeds", e["baseline_same_seeds"])]:
            o = r["outcomes"]
            n = sum(o.values())
            t.append((name, n, f"{100 * o.get('placed', 0) / max(n, 1):.1f} %", f(r["door_ok_rate"], 2),
                      f(r["door_open_s"]), f(r["door_close_s"]), f(r["pick_place_s_mean"]),
                      ", ".join(f"{k} {v}" for k, v in o.items() if k != "placed") or "-"))
        print(md(["printer", "parts", "placed", "door ok", "open s", "close s", "s per pick", "other outcomes"], t))

    sp = R / "des_sens.json"
    if sp.exists():
        s = json.loads(sp.read_text())
        print(f"\n### What moves the knee ({s['rows'][0]['layout']}, {s['rows'][0]['station']}, saturated demand): "
              "printer time waiting for the arm % / parts per day\n")
        ns = sorted({r["n"] for r in s["rows"]})
        tags = list(dict.fromkeys(r["tag"] for r in s["rows"]))
        by = {(r["tag"], r["n"]): r for r in s["rows"]}
        t = [(tag, *(f"{by[(tag, n)]['arm_wait_pct']:.1f} / {by[(tag, n)]['parts_per_day']:.0f}" for n in ns),
              s["knees"].get(f"{tag}/{s['rows'][0]['layout']}/{s['rows'][0]['station']}", {}).get("cap_knee_n"))
             for tag in tags]
        print(md(["variant", *(f"N={n}" for n in ns), "knee N"], t))

    ec = R / "layout_economics.json"
    if ec.exists():
        e = json.loads(ec.read_text())
        print(f"\n### Economics (ASSUMED prices; human baseline {e['human_util_pct']} % util)\n")
        ends = {(a.layout, a.station), ("row", "end")}        # the recommendation and today's arrangement
        t = [(lab[(r["layout"], r["station"])], r["n"], f(float(r["rail_m"])), f"{r['cell_cost_usd']:,}",
              f"{r['cost_per_printer_usd']:,}", r["extra_printer_h_per_printer_month"], f"{r['benefit_usd_per_cell_month']:,}",
              r["cell_payback_months"], r["farm_payback_months"])
             for r in e["rows"] if r["n"] in (10, 20, 30, a.n, 50, 60) and (r["layout"], r["station"]) in ends]
        print(md(["layout", "N", "rail m", "cell cost USD", "per printer USD", "extra printer-h / printer-month",
                  "B USD / cell-month", "cell payback months (C/B)", f"{e['farm_printers']}-printer farm payback months"], t))

    for p in sorted(R.glob("sensitivity_*.json")):
        s = json.loads(p.read_text())
        c = s["cell"]
        print(f"\n### Failure-rate sensitivity, {c['n_printers']} printers, layout {c['layout']}, station "
              f"{c['station']} (rake clears {100 * s['rake_clears_share']:.0f} % of failed parts; human = one operator "
              f"per 10 printers)\n")
        h = s["human"]
        t = [("human, 10 printers", "-", f(h["printer_util_pct"]), f(h["parts_per_printer_day"], 2), f(h["lead_time_p90_h"]),
              f(h["printer_blocked_h"]))]
        t += [(f"{100 * r['p_fail']:.0f} %", "yes" if r["rake"] else "no", f(r["printer_util_pct"]),
               f(r["parts_per_printer_day"], 2), f(r["lead_time_p90_h"]), f(r["printer_blocked_h"])) for r in s["rows"]]
        print(md(["failed picks / part", "rake", "util %", "parts/printer-day", "lead p90 h", "printer-h blocked/wk"], t))


if __name__ == "__main__":
    main()
