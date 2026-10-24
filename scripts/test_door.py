"""Open + close printer doors with the whole-body handle swing; report success (from sensors), true angles, timing and
the forces the coupling carries.

python scripts/test_door.py 4 --video                 # one printer, video -> runs/door_p4.mp4
python scripts/test_door.py --all                     # all printers, physical grasp -> runs/door/test_physical.json
python scripts/test_door.py --all --coupling weld     # same with the modelled rigid coupler
"""
import argparse
import json
import pathlib
import time

import mujoco
import numpy as np

from farm import catalog as C
from farm.world import World
from plan.ik import IK
from plan.motion import Executor
import plan.door as D


def run(P, video=False, verbose=True, arc_seg_s=None, squeeze=None):
    D.VERBOSE = verbose
    if arc_seg_s:
        D.ARC_SEG_S = arc_seg_s
    if squeeze:
        D.SQUEEZE_N = squeeze
    w = World()
    ik = IK(w)
    xc, s = C.printer_frame(P)
    w.reset(rail_x=float(np.clip(xc + 0.5 * s, C.RAIL_X_MIN, C.RAIL_X_MAX)))   # (row B end printer: in range)
    w.d.qpos[w.arm_q] = D.TUCK
    w.d.ctrl[w.act_arm] = D.TUCK
    mujoco.mj_forward(w.m, w.d)
    rec = None
    if video:
        from farm.recorder import Recorder
        rec = Recorder(w.m, f"runs/door_p{P}.mp4", view=lambda w: ([xc - 0.05, s * 0.15, 0.9], 1.5, 90 * s + 40, -20))
    ex = Executor(w, rec)
    out = {"printer": P, "coupling": D.COUPLING}
    for opening in (True, False):
        t = time.time()
        w.door_log = []
        ok, dt, note = D.operate_door(ex, ik, P, opening=opening)
        ex.wait(0.5 if opening else 0.8)
        true = float(np.degrees(w.door_angle(P)))
        key = "open" if opening else "close"
        out[key] = {"ok": ok, "s": round(dt, 2), "note": note, "true_deg": round(true, 2),
                    "belief_deg": round(float(np.degrees(w.door_belief(P))), 2), "reed_closed": w.reed_closed(P),
                    "arcs": w.door_log, "wall_s": round(time.time() - t, 1)}
        # independent truth check for the report: open = within 6 deg of target, closed = latched (< 1 deg)
        out[key]["true_ok"] = (true > np.degrees(C.DOOR_OPEN_RAD) - 6) if opening else (true < 1.0)
        print(f"P{P} {key.upper():5s} ok={ok} ({note}) true {true:6.2f} deg, belief "
              f"{out[key]['belief_deg']:6.2f}, reed {'closed' if w.reed_closed(P) else 'open'}, sim {dt:.1f}s, "
              f"wall {time.time() - t:.0f}s", flush=True)
    out["robot_env_contacts_end"] = len(w.contacts_robot_env())
    out["motion"] = D.motion_summary(getattr(w, "transfer_log", []))      # carriage moves (reporting)
    out["moves"] = list(getattr(w, "transfer_log", []))
    print(f"P{P} carriage moves {out['motion']['carriage_moves']} {out['motion']['by_pose']}, transfer "
          f"{out['motion']['transfer_s']} s, contact ticks {out['motion']['contact_ticks']}", flush=True)
    if rec:
        rec.close()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("printer", nargs="?", type=int, default=4)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--coupling", default=D.COUPLING, choices=("physical", "weld"))
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--arc-seg", type=float, default=None, help="seconds per arc segment")
    ap.add_argument("--squeeze", type=float, default=None, help="jaw squeeze on the bar (N)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    D.COUPLING = a.coupling
    printers = range(C.N_PRINTERS) if a.all else [a.printer]
    res = [run(P, a.video, not a.quiet, a.arc_seg, a.squeeze) for P in printers]
    n = len(res)
    so = sum(r["open"]["ok"] and r["open"]["true_ok"] for r in res)
    sc = sum(r["close"]["ok"] and r["close"]["true_ok"] for r in res)
    fp = sum(r[k]["ok"] and not r[k]["true_ok"] for r in res for k in ("open", "close"))
    print(f"\n{a.coupling}: open {so}/{n}, close {sc}/{n}, sensor said ok but truth disagrees: {fp}; "
          f"open {np.mean([r['open']['s'] for r in res]):.1f} s, close {np.mean([r['close']['s'] for r in res]):.1f} s")
    arcs = [arc for r in res for k in ("open", "close") for arc in r[k]["arcs"]]
    if arcs:
        for f in ("peak_push_N", "p95_push_N", "peak_side_N", "p95_side_N", "mean_squeeze_N", "peak_other_contact_N",
                  "peak_hinge_Nm", "peak_joint_torque_frac", "side_over_2mu_N"):
            v = [arc[f] for arc in arcs if f in arc]
            print(f"  {f:24s} max {max(v):7.2f}  median {np.median(v):7.2f}")
    if a.out or a.all:
        p = pathlib.Path(a.out or f"runs/door/test_{a.coupling}.json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(res, indent=1, default=float))
        print("->", p)
