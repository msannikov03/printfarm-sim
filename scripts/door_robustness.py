"""Door robustness sweep: open + close every printer under perturbed door physics and placement, with the robot
planning on the NOMINAL model (catalog geometry) and deciding from sensors only (arm FK, gripper encoder, reed switch).

Perturbations (applied to the simulated world after the planner's model copy was made):
  latch_f0      magnetic catch holding force at contact (N)
  damping       hinge viscous damping (N m s/rad)            nominal 1.5
  friction      hinge Coulomb friction (N m)                 nominal 0.05
  place_mm      printer placement error, uniform +-mm in x and y (per printer, seeded)
  yaw_deg       printer yaw error, uniform +-deg
  sag_mm        door sag: the leaf (and handle) sits this much lower than nominal (leaf clearance is 4 mm)
  squeeze       jaw squeeze per pad (N)
  mu            pad / bar friction coefficient
  arc_seg       seconds per arc segment (swing speed)

python scripts/door_robustness.py --procs 2          -> runs/door/robustness.jsonl + table
"""
import argparse
import json
import multiprocessing as mp
import pathlib
import time

import numpy as np

OUT = pathlib.Path("runs/door/robustness.jsonl")
NOMINAL = {"latch_f0": 10.0, "damping": 1.5, "friction": 0.05, "place_mm": 0.0, "yaw_deg": 0.0, "sag_mm": 0.0,
           "squeeze": 5.0, "mu": 1.2, "arc_seg": 0.45}
CONDITIONS = [
    ("nominal", {}),
    ("no magnet", {"latch_f0": 0.0}),
    ("magnet 15 N", {"latch_f0": 15.0}),
    ("magnet 18 N", {"latch_f0": 18.0}),
    ("magnet 22 N (5 lbf catch)", {"latch_f0": 22.0}),
    ("magnet 30 N (3 kgf catch)", {"latch_f0": 30.0}),
    ("magnet 54 N (5.5 kgf catch)", {"latch_f0": 54.0}),
    ("hinge damping 0.1", {"damping": 0.1}),
    ("hinge damping 0.5", {"damping": 0.5}),
    ("hinge damping 4", {"damping": 4.0}),
    ("hinge friction 0.3 N m", {"friction": 0.3}),
    ("hinge friction 1.0 N m", {"friction": 1.0}),
    ("placement +-2 mm, +-0.5 deg", {"place_mm": 2.0, "yaw_deg": 0.5}),
    ("placement +-5 mm, +-1 deg", {"place_mm": 5.0, "yaw_deg": 1.0}),
    ("placement +-10 mm, +-2 deg", {"place_mm": 10.0, "yaw_deg": 2.0}),
    ("placement +-15 mm, +-3 deg", {"place_mm": 15.0, "yaw_deg": 3.0}),
    ("door sag 2 mm", {"sag_mm": 2.0}),
    ("door sag 3.5 mm", {"sag_mm": 3.5}),
    ("squeeze 1 N/pad", {"squeeze": 1.0}),
    ("squeeze 2.5 N/pad", {"squeeze": 2.5}),
    ("squeeze 9 N/pad", {"squeeze": 9.0}),
    ("pad friction 0.5", {"mu": 0.5}),
    ("pad friction 0.8", {"mu": 0.8}),
    ("swing 1.5x faster", {"arc_seg": 0.30}),
    ("swing 2x faster", {"arc_seg": 0.225}),
    ("combined realistic", {"latch_f0": 15.0, "damping": 0.3, "friction": 0.2, "place_mm": 5.0, "yaw_deg": 1.0,
                            "sag_mm": 3.0}),
    ("combined realistic, squeeze 2.5, mu 0.8", {"latch_f0": 15.0, "damping": 0.3, "friction": 0.2, "place_mm": 5.0,
                                                 "yaw_deg": 1.0, "sag_mm": 3.0, "squeeze": 2.5, "mu": 0.8}),
]


def apply(w, i, cfg, rng):
    """Perturb the simulated world (not the planner's copy) for printer i."""
    import math
    import mujoco
    m = w.m
    w.latch_f0 = cfg["latch_f0"]
    v = w.door_v[i]
    m.dof_damping[v] = cfg["damping"]
    m.dof_frictionloss[v] = cfg["friction"]
    pb, db = m.body(f"printer{i}").id, m.body(f"door{i}").id
    e = cfg["place_mm"] / 1000.0
    dx, dy = rng.uniform(-e, e, 2) if e > 0 else (0.0, 0.0)
    yaw = math.radians(rng.uniform(-cfg["yaw_deg"], cfg["yaw_deg"])) if cfg["yaw_deg"] > 0 else 0.0
    m.body_pos[pb, :2] += (dx, dy)
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]), m.body_quat[pb].copy())
    m.body_quat[pb] = q
    m.body_pos[db, 2] -= cfg["sag_mm"] / 1000.0
    pin = m.geom(f"pin{i}").id
    fingers = [g for g in w.robot_geoms if m.geom_bodyid[g] in w.finger_bodies]
    for g in fingers + [pin]:
        m.geom_friction[g, 0] = cfg["mu"]
    mujoco.mj_forward(m, w.d)
    return {"dx_mm": round(1000 * float(dx), 2), "dy_mm": round(1000 * float(dy), 2),
            "yaw_deg": round(math.degrees(yaw), 3)}


def episode(job):
    name, over, P, seed = job
    cfg = dict(NOMINAL, **over)
    import mujoco
    from farm import catalog as C
    from farm.world import World
    from plan.ik import IK
    from plan.motion import Executor
    import plan.door as D
    D.VERBOSE = False
    D.SQUEEZE_N, D.ARC_SEG_S = cfg["squeeze"], cfg["arc_seg"]
    t0 = time.time()
    w = World()
    ik = IK(w)                                  # planner copy = nominal model
    xc, s = C.printer_frame(P)
    w.reset(rail_x=xc + 0.5 * s)
    w.d.qpos[w.arm_q] = D.TUCK
    w.d.ctrl[w.act_arm] = D.TUCK
    pert = apply(w, P, cfg, np.random.default_rng(seed))
    ex = Executor(w)
    ex.wait(0.2)
    res = {"cond": name, "printer": P, "seed": seed, **pert}
    for opening in (True, False):
        w.door_log = []
        try:
            ok, dt, note = D.operate_door(ex, ik, P, opening=opening)
        except Exception as exc:                # a crash counts as a failure, keep the sweep going
            ok, dt, note = False, 0.0, f"exception: {exc!r}"
        ex.wait(0.5)
        true = float(np.degrees(w.door_angle(P)))
        true_ok = true > np.degrees(C.DOOR_OPEN_RAD) - 6 if opening else true < 1.0
        k = "open" if opening else "close"
        arcs = w.door_log
        res[k] = {"ok": bool(ok), "true_ok": bool(true_ok), "s": round(dt, 2), "note": note, "true_deg": round(true, 2),
                  "belief_deg": round(float(np.degrees(w.door_belief(P))), 2),
                  "peak_push_N": max([a.get("peak_push_N", 0) for a in arcs] or [0]),
                  "p95_side_N": max([a.get("p95_side_N", 0) for a in arcs] or [0]),
                  "peak_joint_frac": max([a.get("peak_joint_torque_frac", 0) for a in arcs] or [0])}
        if not ok and opening:                  # cannot close a door that did not open
            break
    res["wall_s"] = round(time.time() - t0, 1)
    return res


def catch_table(forces, P=4):
    """Catch force vs. how the bar is held while breaking the latch: jaw opening, peak push, arm torque."""
    import mujoco
    from farm import catalog as C
    from farm.world import World
    from plan.ik import IK
    from plan.motion import Executor
    import plan.door as D
    D.VERBOSE = False
    print(f"{'catch F0':>8s} {'open':>5s} {'peak push N':>11s} {'jaw gap on bar mm':>17s} {'max jaw gap mm':>14s} "
          f"{'arm torque %':>12s}")
    for f0 in forces:
        w = World()
        ik = IK(w)
        xc, s = C.printer_frame(P)
        w.reset(rail_x=xc + 0.5 * s)
        w.d.qpos[w.arm_q] = D.TUCK
        w.d.ctrl[w.act_arm] = D.TUCK
        w.latch_f0 = f0
        mujoco.mj_forward(w.m, w.d)
        probes = []
        orig = D.DoorProbe
        D.DoorProbe = lambda *a: probes.append(orig(*a)) or probes[-1]
        try:
            ok, _, _ = D.operate_door(Executor(w), ik, P, opening=True)
        finally:
            D.DoorProbe = orig
        a = np.array([r[:10] for r in probes[0].rows])
        gap = 2000 * a[:, 9]
        print(f"{f0:8.1f} {str(ok):>5s} {np.abs(a[:, 2]).max():11.1f} {np.median(gap):17.1f} {gap.max():14.1f} "
              f"{100 * a[:, 8].max():12.0f}", flush=True)


def table(rows):
    by = {}
    for r in rows:
        by.setdefault(r["cond"], []).append(r)
    print(f"{'condition':42s} {'open':>6s} {'close':>6s} {'both':>6s} {'sens!=true':>10s} {'open s':>7s} {'close s':>7s} "
          f"{'push N':>7s} {'side N':>7s} {'jt %':>5s} {'est err':>8s}")
    for name, _ in CONDITIONS:
        rs = by.get(name, [])
        if not rs:
            continue
        n = len(rs)
        o = [r for r in rs if r["open"]["ok"] and r["open"]["true_ok"]]
        c = [r for r in rs if "close" in r and r["close"]["ok"] and r["close"]["true_ok"]]
        mism = sum(r[k]["ok"] != r[k]["true_ok"] for r in rs for k in ("open", "close") if k in r)
        ts_o = np.mean([r["open"]["s"] for r in o]) if o else float("nan")
        ts_c = np.mean([r["close"]["s"] for r in c]) if c else float("nan")
        push = max(r[k]["peak_push_N"] for r in rs for k in ("open", "close") if k in r)
        side = max(r[k]["p95_side_N"] for r in rs for k in ("open", "close") if k in r)
        jt = max(r[k]["peak_joint_frac"] for r in rs for k in ("open", "close") if k in r)
        est = [abs(r["open"]["belief_deg"] - r["open"]["true_deg"]) for r in o]      # FK door estimate after opening
        print(f"{name:42s} {len(o):3d}/{n:<2d} {len(c):3d}/{n:<2d} {len(c):3d}/{n:<2d} {mism:10d} {ts_o:7.1f} {ts_c:7.1f} "
              f"{push:7.1f} {side:7.1f} {100 * jt:5.0f} {max(est) if est else float('nan'):8.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--printers", default="0,1,2,3,4,5,6,7,8,9", help="printers for random-placement conditions")
    ap.add_argument("--seeds", type=int, default=2, help="random placements per printer")
    ap.add_argument("--det-printers", type=lambda v: [int(p) for p in v.split(",")], default=[0, 4, 9],
                    help="printers for deterministic conditions")
    ap.add_argument("--only", default=None, help="comma-separated condition indices")
    ap.add_argument("--table", action="store_true", help="only print the table of an existing run")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--catch", default=None, help="comma-separated catch forces (N): jaw/force table, one process")
    a = ap.parse_args()
    OUT = pathlib.Path(a.out)
    if a.catch:
        catch_table([float(f) for f in a.catch.split(",")])
        raise SystemExit
    if a.table:
        table([json.loads(l) for l in OUT.read_text().splitlines() if l.strip()])
        raise SystemExit
    printers = [int(p) for p in a.printers.split(",")]
    conds = CONDITIONS if a.only is None else [CONDITIONS[int(k)] for k in a.only.split(",")]
    jobs = []
    for ci, (name, over) in enumerate(conds):
        if over.get("place_mm") or over.get("yaw_deg"):          # random placement: 2 seeds per printer
            jobs += [(name, over, P, 1000 * ci + 10 * k + P) for k in range(a.seeds) for P in printers]
        else:     # deterministic (all printers are identical copies): the two rail ends and the middle
            jobs += [(name, over, P, 1000 * ci + P) for P in a.det_printers]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with mp.Pool(a.procs, maxtasksperchild=4) as pool, OUT.open("a") as f:
        for r in pool.imap_unordered(episode, jobs):
            f.write(json.dumps(r) + "\n")
            f.flush()
            rows.append(r)
            o, c = r["open"], r.get("close", {})
            print(f"{r['cond']:42s} P{r['printer']} open {o['ok']}/{o['true_ok']} {o['note'][:40]:40s} "
                  f"close {c.get('ok')}/{c.get('true_ok')} {str(c.get('note', ''))[:40]}", flush=True)
    table(rows)
