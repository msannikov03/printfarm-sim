"""Measured MuJoCo durations/rates -> runs/timings.json for the fleet DES.

python scripts/make_timings.py runs/eval [out.json]       (reads seed_*.json from the physics batch)
A mixed / large batch (eval_unload --skus) adds pick_place_large_s_* (lowered placement into the tray); the small
pick statistics then exclude the large parts.
"""
import glob
import json
import pathlib
import sys

import numpy as np

src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "runs/eval")
eps = [json.loads(pathlib.Path(f).read_text()) for f in sorted(glob.glob(str(src / "seed_*.json")))]
opens = [e["door_open"]["s"] for e in eps if e.get("door_open", {}).get("ok")]
closes = [e["door_close"]["s"] for e in eps if e.get("door_close", {}).get("ok")]
parts = [p for e in eps for p in e["parts"]]
# truth (the part is in its bin) when the batch recorded it; older batches only have the robot's outcome
placed = [p for p in parts if p.get("truly_placed", p["outcome"] == "placed")]
false_success = [p for p in parts if p["outcome"] == "placed" and not p.get("truly_placed", True)]
purge = [p["s"] for p in placed if p["name"].endswith("purge_tab")]
blocking = [p for p in parts if p.get("blocking")]
swept = [p for p in parts if p["outcome"] == "swept"]
blocked_plates = sum(1 for e in eps if any(p.get("blocking") for p in e["parts"]) or not e.get("bed_empty_after", True))
picks = [p["s"] for p in placed if not p["name"].endswith("purge_tab") and not p.get("large")]
picks_large = [p["s"] for p in placed if p.get("large")]      # eval_unload --skus mixed / large: lowered tray place
n = max(len(parts), 1)
t = {
    "source": f"MuJoCo batch {src} ({len(eps)} plates, {len(parts)} parts)",
    "rail_speed_mps": 0.4,
    "door_open_s": float(np.mean(opens)), "door_close_s": float(np.mean(closes)),
    "scan_s": 2.0,
    "purge_tab_s": float(np.mean(purge)) if purge else 20.0,
    "pick_place_s_mean": float(np.mean(picks)), "pick_place_s_std": float(np.std(picks)),
    "bed_check_s": 2.0, "station_handoff_s": 6.0, "retry_s": 8.0,
    "p_success_first_try": len(placed) / n,
    # a part only blocks its printer if it is still on the bed after the rake fallback; anything else that was not
    # placed (raked into the reject tray, dropped off the bed) is reprinted without stopping the printer
    "p_intervention_per_part": len(blocking) / n,
    "p_reprint_per_part": (len(parts) - len(placed) - len(blocking)) / n,
    "p_blocked_plate": blocked_plates / max(len(eps), 1),
    # the robot reported 'placed' but the part is not in its bin: an order ships short (caught at packing)
    "p_false_success_per_part": len(false_success) / n,
    "realism": sorted({e.get("realism", "ideal") for e in eps}),
    "rake_s": float(np.mean([p["s"] for p in swept])) if swept else 15.0,
    "door_ok_rate": (len(opens) + len(closes)) / max(2 * len(eps), 1),
    "outcomes": {k: sum(p["outcome"] == k for p in parts) for k in sorted({p["outcome"] for p in parts})},
}
if picks_large:
    t.update(pick_place_large_s_mean=float(np.mean(picks_large)), pick_place_large_s_std=float(np.std(picks_large)),
             n_large=len(picks_large))
out = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else "runs/timings.json")
out.write_text(json.dumps(t, indent=1))
print(json.dumps(t, indent=1))
