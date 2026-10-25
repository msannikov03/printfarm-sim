"""scripts/eval_unload.py summarize: attempt counts (scans, pick attempts with retries, rake calls)."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("eval_unload", Path(__file__).parents[1] / "scripts" / "eval_unload.py")
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)


def part(name, outcome, n_picks, truly=None):
    return {"name": name, "outcome": outcome, "truly_placed": outcome == "placed" if truly is None else truly,
            "pick_attempts": n_picks, "s": 10.0, "drops": [], "end": "own_bin" if outcome == "placed" else "tray"}


def episode(attempts=True):
    r = {"seed": 0, "printer": 4, "door_open": {"ok": True}, "door_close": {"ok": True}, "sim_s": 100.0,
         "parts": [part("P_purge_tab", "placed", 1), part("o0_cube_0", "placed", 2), part("o0_lid_1", "swept", 0)]}
    if attempts:
        r["attempts"] = [{"kind": "scan", "tries": 1}, {"kind": "pick", "name": "P_purge_tab", "outcome": "placed"},
                         {"kind": "scan", "tries": 2}, {"kind": "pick", "name": "o0_cube_0", "outcome": "slipped"},
                         {"kind": "scan", "tries": 1}, {"kind": "pick", "name": "o0_cube_0", "outcome": "placed"},
                         {"kind": "rake", "phase": "final", "ok": True, "strokes": ["behind:front"]},
                         {"kind": "rake", "phase": "final", "ok": False, "strokes": []}]
    return r


def test_attempt_counts():
    c = E.summarize([episode(), episode()])["counts"]
    assert c["pick_attempts"] == 6 and c["pick_retries"] == 2 and c["parts_never_attempted"] == 2
    assert c["pick_attempt:slipped"] == 2 and c["pick_attempt:placed"] == 4
    assert c["bed_scans"] == 6 and c["bed_scan_frame_retries"] == 2
    assert c["rake_calls"] == 4 and c["rake_calls_ok"] == 2 and c["rake_strokes"] == 2


def test_old_batches_unchanged():
    c = E.summarize([episode(attempts=False)])["counts"]          # seed files written before the attempt log
    assert not any(k.startswith(("pick_attempt", "bed_scan", "rake_")) for k in c)
    assert c["parts"] == 3 and c["truly_placed"] == 2 and c["outcome:swept"] == 1
