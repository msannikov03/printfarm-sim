"""Replay a planned demo on the Pi itself (no MuJoCo, no Mac connection needed while it runs).

Fallback for a dead tailnet / lab network: plan on the Mac as usual (dry run, previews), copy the log over
(USB stick or Ethernet cable), replay it next to the arm:
    Mac:  python -m real.run_demo --printer ... --gcode plate.gcode --sim-preview      -> runs/real_demo_log.json
    Pi:   python3 -m real.replay real_demo_log.json --port /dev/so101 [--film] [--go]
Same moves as run_demo --live (ready -> per part: transit, hover, gripper open/close, retreat -> start pose),
through the same ArmController limits (min-jerk, joint limits, keep-out incl. the bed from the log, stall stop).
Without --go it only prints the sequence. Also works against a server: --arm remote://<pi>:8765.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path


def sequence(log: dict, dwell_s: float, gripper: bool) -> list:
    """[(label, q_rad list | ('gripper', pct) | ('dwell', s))] from a run_demo log."""
    seq = []
    rad = lambda d: [math.radians(v) for v in d]
    if log.get("ready_q_deg"):
        seq.append(("ready pose", rad(log["ready_q_deg"])))
    for r in log.get("results", []):
        if "pick" in r or r.get("status") not in ("dry-run ok", "moved"):
            continue
        name = r["name"]
        if r.get("q_transit_deg"):
            seq.append((f"transit above {name}", rad(r["q_transit_deg"])))
        seq.append((f"hover {name}", rad(r["q_model_deg"])))
        if gripper:
            seq.append((f"open over {name}", ("gripper", 60.0)))
        if dwell_s:
            seq.append((f"pause over {name}", ("dwell", dwell_s)))
        if gripper:
            seq.append((f"close over {name}", ("gripper", 0.0)))
        if r.get("q_transit_deg"):
            seq.append((f"retreat from {name}", rad(r["q_transit_deg"])))
    if log.get("ready_q_deg"):
        seq.append(("ready pose", rad(log["ready_q_deg"])))
    return seq


def main(argv=None):
    ap = argparse.ArgumentParser(description="replay runs/real_demo_log.json on the arm (Pi-side fallback)")
    ap.add_argument("log", help="real_demo_log.json written by python -m real.run_demo")
    ap.add_argument("--port", help="servo bus (/dev/so101, /dev/ttyACM0, 'mock')")
    ap.add_argument("--arm", help="remote://<pi>:8765 instead of a local bus")
    ap.add_argument("--speed", type=float, default=15.0)
    ap.add_argument("--dwell-s", type=float, default=2.5)
    ap.add_argument("--film", action="store_true", help="narration + gripper open/close at each part")
    ap.add_argument("--go", action="store_true", help="actually move (default: print the sequence)")
    ap.add_argument("--allow-nominal", action="store_true")
    a = ap.parse_args(argv)
    log = json.loads(Path(a.log).read_text())
    seq = sequence(log, a.dwell_s, gripper=a.film)
    print(f"== {len(seq)} steps from {a.log} (bed calibration: {log.get('calib', {}).get('method')})")
    if not a.go:
        for label, s in seq:
            print(f"   {label:<28} {s if isinstance(s, tuple) else [round(math.degrees(v), 1) for v in s]}")
        print("== add --go to move")
        return 0
    if not (a.port or a.arm):
        ap.error("--port or --arm")
    from real.hello_arm import _open
    ns = argparse.Namespace(arm=a.arm, port=a.port, calib=None, speed=a.speed, allow_nominal=a.allow_nominal)
    from real.joint_calib import JOINT_CALIB_JSON
    ns.calib = JOINT_CALIB_JSON
    arm = _open(ns)
    t0 = time.monotonic()
    say = (lambda m: print(f"\n[{time.monotonic() - t0:6.1f} s] >>> {m}", flush=True)) if a.film else print
    try:
        c = log.get("calib")
        if c and c.get("T_robot_bed"):
            T = c["T_robot_bed"]
            arm.set_keepout(bed=dict(R=[row[:3] for row in T[:3]], t=[row[3] for row in T[:3]],
                                     size_mm=c.get("bed_mm", [220, 220]), margin_mm=8.0))
        q_home = arm.read_q()
        n_parts = sum(1 for lbl, _ in seq if lbl.startswith("hover "))
        say(f"print finished - {n_parts} parts on the bed")
        for label, s in seq + [("start pose", q_home)]:
            if isinstance(s, tuple) and s[0] == "dwell":
                time.sleep(s[1])
                continue
            if label.startswith("hover "):
                say(label.replace("hover ", "over "))
            try:
                if isinstance(s, tuple):
                    r = arm.gripper(open_pct=s[1], speed_deg_s=a.speed * 3, wait=True)
                else:
                    r = arm.move(list(s), speed_deg_s=a.speed, wait=True)
                if r.get("state") == "aborted":
                    raise RuntimeError(r.get("reason"))
            except RuntimeError as e:
                print(f"   {label}: STOPPED {e}")
                return 2
            print(f"   {label} done")
        say("done - back at the start pose")
        return 0
    except KeyboardInterrupt:
        arm.stop("Ctrl-C")
        return 130
    finally:
        arm.close()


if __name__ == "__main__":
    sys.exit(main())
