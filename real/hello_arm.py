"""'Arm hello': the first thing to run on the real SO-101. No MuJoCo, no printer, no bed calibration.

    python -m real.hello_arm --arm remote://<pi>:8765          # Mac -> Pi arm server
    python3 -m real.hello_arm --port /dev/ttyACM0               # on the Pi itself (or the Mac with the arm plugged in)
    python -m real.hello_arm --port mock --go                   # rehearsal

1. bus check: every servo answers, supply voltage, temperature, status flags;
2. reads the joints (model degrees) and the fixed-jaw tip position (pure-Python FK);
3. with --go only: a small, slow wave relative to the CURRENT pose (wrist up, pan left/right, gripper
   open/close, back to the start), every move through the same server-side limits as the demo
   (min-jerk, joint limits, keep-out, stall stop). Without --go it prints the moves and stops.
"""
from __future__ import annotations

import argparse
import math
import sys
import time

from real.kin_lite import JOINTS, Chain


def _open(a):
    if a.arm:
        from real.arm_client import RemoteArm
        return RemoteArm(a.arm).connect()
    from real.arm_server import MOCK_REST_Q, ArmController, Limits, mock_bus_at
    from real.feetech import FeetechBus
    from real.joint_calib import JointCalib
    calib = JointCalib.load_or_nominal(a.calib)
    mock = a.port == "mock"
    bus = mock_bus_at(MOCK_REST_Q, calib) if mock else FeetechBus(a.port).open()
    return ArmController(bus, calib, limits=Limits(max_speed_deg_s=max(a.speed, 1.0)), heartbeat_s=None,
                         allow_nominal=mock or a.allow_nominal).connect()


def hello_moves(q0, amp_deg=15.0):
    """(label, target q) relative to the start pose, clipped to stay sensible."""
    d = math.radians(amp_deg)
    up = list(q0)
    up[3] = q0[3] - d                                  # wrist_flex - : gripper tips up
    left, right = list(up), list(up)
    left[0], right[0] = q0[0] - d, q0[0] + d           # shoulder_pan - : arm swings to its left
    return [("wrist up", up), ("wave left", left), ("wave right", right), ("centre", up),
            ("gripper open", ("gripper", 50.0)), ("gripper close", ("gripper", 0.0)), ("back to start", list(q0))]


def main(argv=None):
    ap = argparse.ArgumentParser(description="SO-101 hello: bus check + a small slow wave")
    ap.add_argument("--arm", help="remote://<pi>:8765")
    ap.add_argument("--port", help="local servo bus (/dev/ttyACM0, /dev/tty.usbmodem..., 'mock')")
    ap.add_argument("--calib", default=None, help="joint_calib.json for --port (default real/joint_calib.json)")
    ap.add_argument("--allow-nominal", action="store_true")
    ap.add_argument("--go", action="store_true", help="actually move (default: only print)")
    ap.add_argument("--amp", type=float, default=15.0, help="wave amplitude deg")
    ap.add_argument("--speed", type=float, default=20.0, help="deg/s")
    a = ap.parse_args(argv)
    if not (a.arm or a.port):
        ap.error("--arm remote://<pi>:8765 or --port /dev/ttyACM0")
    if a.calib is None:
        from real.joint_calib import JOINT_CALIB_JSON
        a.calib = JOINT_CALIB_JSON
    arm = _open(a)
    chain = Chain()
    try:
        s = arm.status()
        print(f"== calibration: {s['calib']}   E-stopped: {s['estopped']}")
        bad = []
        for j in JOINTS:
            h = s["health"].get(j)
            if not h:
                bad.append(f"{j}: no answer")
                print(f"   {j:<13} NO ANSWER")
                continue
            print(f"   {j:<13} raw {h['pos']:5d}  {h['volts']:4.1f} V  {h['temp_c']:3d} C  load {h['load_pct']:5.1f} %"
                  f"  torque {s['torque'].get(j)}" + (f"  FLAGS {h['flags']}" if h["flags"] else ""))
            if h["flags"]:
                bad.append(f"{j}: {h['flags']}")
            if h["temp_c"] >= 60:
                bad.append(f"{j}: {h['temp_c']} C")
        volts = [s["health"][j]["volts"] for j in JOINTS if s["health"].get(j)]
        if volts and min(volts) < 4.5:            # 5 V (7.4 V servos) and 12 V kits both exist
            bad.append(f"supply {min(volts):.1f} V: servo power is off or the wrong adapter (USB alone does not power them)")
        q0 = arm.read_q()
        print("== joints (model deg):", [round(math.degrees(v), 1) for v in q0])
        print("== fixed-jaw tip (robot frame, mm):", [round(v * 1000) for v in chain.point(q0)])
        if bad:
            print("PROBLEMS:", *bad, sep="\n  ")
            sys.exit(1)
        moves = hello_moves(q0, a.amp)
        if not a.go:
            for label, q in moves:
                print(f"   would move: {label:<14} -> "
                      + (f"gripper {q[1]:.0f} % open" if isinstance(q, tuple) else str([round(math.degrees(v), 1) for v in q])))
            print("== bus OK. Add --go to wave (small, slow, relative to this pose).")
            return 0
        for label, q in moves:
            t = time.monotonic()
            try:
                if isinstance(q, tuple):
                    r = arm.gripper(open_pct=q[1], speed_deg_s=a.speed * 3, wait=True)
                else:
                    r = arm.move(q, speed_deg_s=a.speed, wait=True)
            except RuntimeError as e:                  # refused (limits / keep-out) or aborted (stall, watchdog)
                print(f"   {label}: STOPPED {e}")
                return 2
            if r.get("state") == "aborted":
                print(f"   {label}: ABORTED {r.get('reason')}")
                return 2
            print(f"   {label:<14} {time.monotonic() - t:4.1f} s  error deg {r.get('err_deg')}")
        print("== hello done")
        return 0
    except KeyboardInterrupt:
        arm.stop("Ctrl-C")
        print("\nstopped (holding)")
        return 130
    finally:
        arm.close()


if __name__ == "__main__":
    sys.exit(main())
