"""Fill the {{PLACEHOLDER}} numbers in README.md and docs/*.md from run outputs, and lint the docs.

    python scripts/fill_results.py --batch runs/eval --batch-real runs/eval_real --batch-faults runs/eval_faults \
        --des runs/des --farm runs/farm3 --set PERC_IDEAL_MM="0.72 / 2.3 mm" ...      # dry run: print the values
    ... --write                       # replace the placeholders in place (under --root, default: the repo)
    python scripts/fill_results.py --root OUT_DIR --lint     # no placeholders left, every relative link resolves

Each placeholder's source is listed in docs/results.md. Sources that are not given are skipped: their placeholders
stay, and --lint reports them. Pure stdlib (runs without the project venv).
"""
import argparse
import json
import pathlib
import re
import sys

PH = re.compile(r"\{\{([A-Z0-9_]+)\}\}")
LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")


def pct(x, nd=1):
    return f"{100 * x:.{nd}f} %"


def from_eval(d, prefix):
    s = json.loads((pathlib.Path(d) / "summary.json").read_text())
    c = s["counts"]
    ep, parts = c["episodes"], c["parts"]
    placed = c.get("truly_placed", c.get("outcome:placed", 0))      # ground truth: the part is in its order bin
    out = {
        f"{prefix}_PLACED": f"{placed}/{parts} ({pct(placed / parts)})",
        f"{prefix}_DOORS": f"{c['door_open_ok']}/{ep} / {c['door_close_ok']}/{ep}",
        # truly clear beds (no part or debris left; ground truth), not the camera's "confirmed empty" verdict
        f"{prefix}_BEDS": f"{ep - c.get('beds_blocked', c.get('bed_not_empty', 0)) - c.get('debris_left_on_bed', 0)}/{ep}",
        f"{prefix}_SPP": f"{s['mean_s_per_placed_part']:.1f} s",
        f"{prefix}_FALSE": str(c.get("false_success", c.get("believed_not_true", 0))),
        f"{prefix}_BLOCK": str(c.get("beds_blocked", c.get("blocking", 0))),
    }
    failed = parts - placed
    if failed:
        out[f"{prefix}_RAKE_SHARE"] = pct((failed - c.get("blocking", 0)) / failed, 0)
    return out


def from_des(d):
    r = json.loads((pathlib.Path(d) / "kpis_compare.json").read_text())["runs"]
    rob, hum = r["order_aware"]["mean"], r["human"]["mean"]
    out = {}
    for tag, m in (("R", rob), ("H", hum)):
        out[f"DES_UTIL_{tag}"] = f"{m['printer_util_pct']:.0f} %"
        out[f"DES_PPD_{tag}"] = f"{m['parts_per_day']:.0f}"
        out[f"DES_LEAD50_{tag}"] = f"{m['lead_time_p50_h']:.1f} h"
        out[f"DES_LEAD_{tag}"] = f"{m['lead_time_p90_h']:.1f} h"
    out["DES_ROBOT_BUSY"] = f"{rob['robot_util_pct']:.0f} %"
    sp = pathlib.Path(d) / "sensitivity.json"
    if sp.exists():
        s = json.loads(sp.read_text())
        # compare good output (parts/day): with the rake, reprints add printing hours, so utilisation would flatter it
        key = "parts_per_printer_day" if "parts_per_printer_day" in s["human"] else "parts_per_day"
        hu = s["human"][key]
        no = sorted((x["p_fail"], x[key]) for x in s["rows"] if not x["rake"])
        cross = None
        for (p0, u0), (p1, u1) in zip(no, no[1:]):
            if u0 >= hu > u1:
                cross = p0 + (u0 - hu) / (u0 - u1) * (p1 - p0)
                break
        if cross is not None:
            out["SENS_BREAKEVEN"] = f"{100 * cross:.0f} %"
        better = [x for x in s["rows"] if x["rake"] and x[key] > hu]
        if better:
            top = max(better, key=lambda x: x["p_fail"])
            out["SENS_PMAX"] = f"{100 * top['p_fail']:.0f} %"
            unit = "parts per printer per day" if key == "parts_per_printer_day" else "parts/day"
            out["SENS_UTIL_PMAX"] = f"{top[key]:.1f} vs {hu:.1f} {unit}"
    return out


def from_farm(d):
    f = sorted(pathlib.Path(d).glob("summary_s*.json"))[0]
    s = json.loads(f.read_text())
    st, picks = s["stats"], s["picks"]
    n = st.get("plates_unloaded", 0)
    return {
        "FARM_PLATES": str(n),
        "FARM_HOURS": f"{s['farm_hours']:.1f}",
        "FARM_PICKS": f"{picks.get('placed', 0)}/{sum(picks.values())}",
        "FARM_DOORS": f"{st.get('door_open_ok', 0)}/{n} opened and {st.get('door_close_ok', 0)}/{n} closed",
        "FARM_LEAD": f"{s['lead_h_p50']:.1f} h / {s['lead_h_p90']:.1f} h",
        "FARM_INTERV": str(st.get("interventions", 0)),
        "FARM_BOXES": f"{s['station']['stats']['boxes_complete']}/{s['station']['stats']['boxes_sealed']}"
                      if "station" in s else "n/a",
    }


def from_box_large(dirs):
    """Large parts bed -> tray from two eval_unload --skus mixed batches (row A, row B)."""
    rows = []
    for d in dirs.split(","):
        L = json.loads((pathlib.Path(d) / "summary.json").read_text())["by_size"]["large"]
        rows.append((L["truly_placed"], L["parts"]))
    ok, n = sum(r[0] for r in rows), sum(r[1] for r in rows)
    return {"BOX_LARGE": f"{ok}/{n} placed (row A {rows[0][0]}/{rows[0][1]}, row B {rows[1][0]}/{rows[1][1]})"}


def from_station_times(f):
    t = json.loads(pathlib.Path(f).read_text())["package"]
    return {"BOX_STATION_S": f"{t['mean']:.0f} s (sd {t['sd']:.0f} s, n = {t['n']})"}


def from_queue_policy(f):
    r = next(x for x in json.loads(pathlib.Path(f).read_text())["before_after"] if x["mix"] == "small")
    return {"BOX_PARKS": f"{r['fifo']['partial_handoffs']:.0f} with first-in-first-out printing, "
                         f"{r['package']['partial_handoffs']:.1f} with package-aware printing"}


def doc_files(root):
    root = pathlib.Path(root)
    return [p for p in [root / "README.md", *sorted((root / "docs").glob("*.md"))] if p.exists()]


def lint(root):
    bad = 0
    for p in doc_files(root):
        text = p.read_text()
        for k in sorted(set(PH.findall(text))):
            print(f"{p.relative_to(root)}: unfilled {{{{{k}}}}}")
            bad += 1
        for target in LINK.findall(text):
            if re.match(r"^[a-z]+:", target) or target.startswith("#"):
                continue
            path = (p.parent / target.split("#")[0]).resolve()
            if not path.exists():
                print(f"{p.relative_to(root)}: broken link {target}")
                bad += 1
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(pathlib.Path(__file__).resolve().parents[1]))
    ap.add_argument("--batch", help="ideal-sensor batch dir (summary.json)")
    ap.add_argument("--batch-real", help="realistic-sensor batch dir")
    ap.add_argument("--batch-harsh", help="harsh-sensor batch dir")
    ap.add_argument("--batch-rowa", help="default cell, row A printer batch dir")
    ap.add_argument("--batch-rowb", help="default cell, row B printer batch dir")
    ap.add_argument("--batch-faults", help="fault batch dir")
    ap.add_argument("--des", help="DES output dir (kpis_compare.json, sensitivity.json)")
    ap.add_argument("--farm", help="whole-cell run dir (summary_s*.json)")
    ap.add_argument("--box-large", help="two eval dirs (row A,row B) of eval_unload --skus mixed")
    ap.add_argument("--station-times", help="runs/station_times.json")
    ap.add_argument("--queue-policy", help="DES queue-policy before/after (queue_policy.json)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="manual value (may repeat)")
    ap.add_argument("--write", action="store_true", help="replace placeholders in place")
    ap.add_argument("--lint", action="store_true", help="only check: no placeholders, no broken relative links")
    a = ap.parse_args(argv)

    if a.lint:
        bad = lint(a.root)
        print(f"lint: {bad} problem(s)")
        return 1 if bad else 0

    vals = {}
    for src, fn in ((a.batch, lambda d: from_eval(d, "IDEAL")), (a.batch_real, lambda d: from_eval(d, "REAL")),
                    (a.batch_harsh, lambda d: from_eval(d, "HARSH")),
                    (a.batch_rowa, lambda d: from_eval(d, "ROWA")), (a.batch_rowb, lambda d: from_eval(d, "ROWB")),
                    (a.batch_faults, lambda d: from_eval(d, "FAULT")), (a.des, from_des), (a.farm, from_farm),
                    (a.box_large, from_box_large), (a.station_times, from_station_times),
                    (a.queue_policy, from_queue_policy)):
        if src:
            vals.update(fn(src))
    for kv in a.set:
        k, _, v = kv.partition("=")
        vals[k.strip()] = v
    for k in sorted(vals):
        print(f"{k:20s} {vals[k]}")

    used = set()
    for p in doc_files(a.root):
        text = p.read_text()
        new = PH.sub(lambda m: (used.add(m.group(1)) or vals[m.group(1)]) if m.group(1) in vals else m.group(0), text)
        if new != text and a.write:
            p.write_text(new)
            print(f"wrote {p}")
    unused = sorted(set(vals) - used)
    if unused:
        print("not used in any doc:", ", ".join(unused))
    if not a.write:
        print("(dry run: add --write to replace)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
