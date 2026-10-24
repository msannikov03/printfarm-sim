"""Door primitive: hold the D-handle bar in the jaws and swing the door along its hinge arc (arm + rail).

Coupling (COUPLING):
- "physical" (default): the parallel jaws close around the 15 mm bar, with the closing axis along the bar's direction of
  travel. The bar spins freely on its own axis (joint pinspin{i}), so the door moves only through contact forces:
  the pads push the bar along the arc (form closure along the jaw axis) and friction holds it in the other two
  directions. Nothing is locked.
- "weld": the old modelled coupler, a rigid weld link_6 <-> pin{i} while engaged (kept for comparison).

Sensing (no ground truth in the decisions):
- while the bar is in the jaws, the door angle is estimated from the arm's forward kinematics (joint + rail encoders,
  with noise) through the known hinge geometry -> World.door_est[i];
- "holding the bar" is read from the gripper encoder (jaw gap ~ bar diameter);
- "closed" is confirmed by the reed switch at the magnetic catch (World.reed_closed).
World.door_angle(i) (truth) is used only for printing and reporting.

The carriage position for an arc is chosen by search: every arc sample must have a collision-free IK solution; the
score is the worst-case manipulability x joint-limit margin along the arc.
"""
import collections
import math

import mujoco
import numpy as np

from farm import catalog as C
from farm.world import GRIP_OPEN, HOME, LATCH_R
from plan.ik import frame
from plan.motion import cartesian_line

TUCK = np.array([0.0, 1.4, 2.6, 0.5, 0.0, 0.0])     # compact upright 'candle' (neutral via-pose for planning)
LOW_TUCK = np.array([0.0, 0.0, 3.3, 0.75, 0.0, 0.0])  # folded below the door line: the carriage slides under open doors
SLIDE_CLEAR = 0.004     # in a slide the arm moves rigidly with its deck, so arm-deck gaps don't change: fine clearance
SLIDE_CLEAR_TALL = 0.012    # upright / held-pose slides: the taller arm sways under the carriage's acceleration (servo
                            # compliance; measured in physics by SlideProbe), so these keep a wider gap to the scene
SLIDE_STEP = 0.02       # slide check: one sample per 2 cm of rail (and both ends); with the clearance this leaves no gap
                        # through which a thin obstacle (an open leaf's edge, a 15 mm handle bar) could slip unchecked
COMPACT_R = 0.30        # the current pose slides as it is when the fingertips (and a held part) are within this of the
                        # arm's base axis: over the deck (|y| < 0.31 m), clear of the aisle in front of the doors
COMPACT_DZ = 0.90       # ... and at most this above the base: no taller than the candle (TUCK's pinch is 0.86 m up),
                        # the tallest slide pose (its slide over the whole rail: tests/test_station.py)
SLIDE_ACC_TALL = 0.7    # peak carriage acceleration in an upright / held-pose slide (m/s^2). Measured (SlideProbe, the
                        # pinch vs a rigid arm): on its position servos the candle swayed 16 mm in a 10 cm hop at the
                        # rail's default timing (peak 3.8 m/s^2), more than SLIDE_CLEAR_TALL; 6-9 mm capped at 1 m/s^2
SLIDE_SETTLE = (0.004, 0.6)     # before a slide: wait until the joints track within this (rad, encoders), at most this
                                # long (s): the residual swing of the move into the slide pose adds to the sway


ARC_N = 14
PRE_UP = 0.05
PRE_BACK = 0.08

COUPLING = "physical"               # "physical" (jaws only, contact forces) | "weld" (modelled rigid coupler)
HANDLE_R = 0.0075                   # bar radius (farm/scene.py pin{i})
GRASP_DEPTH = 0.025                 # bar this far behind the pinch point (toward the palm): the YAM pads run from ~10 mm
                                    # ahead of the pinch to 70 mm behind it, so a bar held AT the pinch has ~2 mm before
                                    # it slides out past the fingertips; 25 mm deep leaves ~27 mm to the tips, 37 to the back
SQUEEZE_N = 5.0                     # squeeze per pad on the bar (N). The jaws are coupled, so the gripper actuator
                                    # applies 2x this; it is force-limited at 20 N (max 10 N per pad)
ARC_SEG_S = 0.45                    # seconds per arc segment (ARC_N segments)
HOLD_TOL = (-0.003, 0.004)          # jaw half-gap - HANDLE_R that reads as "bar between the pads" (m)
ENC_SIGMA_Q = math.radians(0.05)    # joint encoder noise, 1 sigma (ASSUMPTION)
ENC_SIGMA_X = 0.0002                # rail encoder noise, 1 sigma (m, ASSUMPTION)
OPEN_OK_RAD = math.radians(95)      # opened if the FK estimate at release is at least this
CLOSE_TRIES = 2                     # closing: re-grasp and push again if the reed switch stays open
OVERPUSH_RAD = math.radians(4)      # the re-push aims this far past nominal closed: if the printer stands a few mm off
                                    # its nominal pose, the nominal 0 deg ends short of the latch; the leaf's own stop
                                    # and the back-drivable jaws limit the push
LATCH_DWELL_S = 0.3                 # wait after release before reading the reed switch


def pin_world(i, theta, which="outer"):
    a, b = (C.PIN_S, -C.PIN_OFFSET) if which == "outer" else (C.IPIN_S, C.IPIN_OFFSET)
    xl = C.HINGE_XL + a * math.cos(theta) + b * math.sin(theta)
    yl = -0.004 - a * math.sin(theta) + b * math.cos(theta)
    return np.array(C.local_to_world(i, xl, yl, C.PIN_ZL))


def handle_angle(i, p):
    """Invert pin_world: door angle (rad) whose nominal bar centre is closest to world point p, plus the radial and
    vertical residuals (m) of p from the nominal bar circle (grasp-quality / model-error indicators)."""
    xc, s = C.printer_frame(i)
    xl, yl, zl = s * (p[0] - xc), s * p[1] - C.AISLE_HALF, p[2] - C.BENCH_H
    vx, vy = xl - C.HINGE_XL, yl + 0.004
    a, b = C.PIN_S, -C.PIN_OFFSET
    th = math.atan2(b, a) - math.atan2(vy, vx)
    th = (th + math.pi) % (2 * math.pi) - math.pi
    return th, math.hypot(vx, vy) - math.hypot(a, b), zl - C.PIN_ZL


GRASPS = [(phi, pitch) for phi in (0.0, 30.0, 60.0, 90.0) for pitch in (0.0, 25.0, 45.0)]


def pin_tangent(i, theta, which="outer"):
    """Horizontal direction the pin moves when the door opens (unit vector, world)."""
    v = pin_world(i, theta + 1e-3, which) - pin_world(i, theta - 1e-3, which)
    return v / np.linalg.norm(v)


def pin_frame(i, theta, grasp=(0.0, 0.0)):
    """Side grasp of the vertical pin. phi: approach angle in the door's frame (0 = along the leaf toward the
    hinge, coming from the free-edge side; 90 = straight at the glass); pitch tilts the approach downward.
    The yaw follows the door, so the arm stays on the free-edge side for the whole arc."""
    _, s = C.printer_frame(i)
    d = np.array([s * math.cos(theta), -s * math.sin(theta), 0.0])      # leaf direction hinge -> free edge
    n = np.array([-s * math.sin(theta), -s * math.cos(theta), 0.0])     # door outer normal
    phi, pitch = (math.radians(v) for v in grasp)
    h = -math.cos(phi) * d - math.sin(phi) * n
    approach = math.cos(pitch) * h + np.array([0, 0, -math.sin(pitch)])
    return frame(approach, np.cross([0.0, 0.0, 1.0], h))


VERTICAL = np.array([0.0, 0.0, 1.0])


VERBOSE = True
RAIL_MIN_L = -0.9      # the carriage may pass under an open door (arm in LOW_TUCK)


# ---------------------------------------------------------------- sensing
def jaw_half_gap(w, ex=None):
    """Gripper encoder: distance of each pad from the jaw centre line (m). Uses the executor's hardware model
    (ex.hw, farm/sensors.py: noisy gap reading) when one is attached."""
    hw = getattr(ex, "hw", None)
    return hw.gap_mm() / 2000.0 if hw is not None else float(w.d.qpos[w.finger_q])


def holding_bar(w, ex=None):
    e = jaw_half_gap(w, ex) - HANDLE_R
    return HOLD_TOL[0] <= e <= HOLD_TOL[1]


def encoders(w, ex=None):
    """Arm joint + rail readings: the executor's hardware model (zero offsets, backlash, noise) when attached,
    else the true joints with ENC_SIGMA_Q / ENC_SIGMA_X noise."""
    hw = getattr(ex, "hw", None)
    if hw is not None:
        return hw.encoders(), hw.rail_encoder()
    return (w.d.qpos[w.arm_q] + w.rng.normal(0.0, ENC_SIGMA_Q, 6),
            float(w.d.qpos[w.rail_q]) + w.rng.normal(0.0, ENC_SIGMA_X))


def sense_door(ik, w, i, which="outer", ex=None):
    """Door angle from the arm's encoders while the bar is in the jaws: FK of the pinch point in the robot's own
    model, minus GRASP_DEPTH along the approach, mapped through the hinge geometry.
    Returns (theta, radial_residual, z_residual)."""
    q, x = encoders(w, ex)
    p, R = ik.fk(q, x)
    return handle_angle(i, p - GRASP_DEPTH * R[:, 2])


def sync_beliefs(ik, w):
    """Planning state = the robot's knowledge: joint/part state from the sim, door angles from door_belief()."""
    ik.sync(w.d)
    for j in range(C.N_PRINTERS):
        ik.d.qpos[w.door_q[j]] = w.door_belief(j)


class DoorProbe:
    """Executor hook: logs what the coupling carries while the robot swings door i (reporting only).
    physical: contact forces fingers -> bar, plus any other robot -> door contact (glass, stand-offs);
    weld: the weld's constraint force. Forces are split into the door's frame at the bar: t (direction of opening
    travel), r (horizontal, hinge -> bar), z (up)."""

    def __init__(self, w, i):
        m = w.m
        self.w, self.i = w, i
        self.pin_geom = m.geom(f"pin{i}").id
        self.door_bodies = {m.body(f"door{i}").id, m.body(f"pin{i}").id}
        self.finger_geoms = {g for g in w.robot_geoms if m.geom_bodyid[g] in w.finger_bodies}
        self.eq = w.coupler[(i, "outer")]
        self.hinge = np.array(C.local_to_world(i, C.HINGE_XL, -0.004, 0.0))
        self.f6 = np.zeros(6)
        self.rows = []

    def __call__(self, w):
        m, d = w.m, w.d
        f_bar, f_other, squeeze = np.zeros(3), np.zeros(3), 0.0
        for k in range(d.ncon):
            c = d.contact[k]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            door1, door2 = b1 in self.door_bodies, b2 in self.door_bodies
            if door1 == door2:
                continue
            rob = c.geom2 if door1 else c.geom1
            if rob not in w.robot_geoms:
                continue
            mujoco.mj_contactForce(m, d, k, self.f6)
            fw = c.frame.reshape(3, 3).T @ self.f6[:3]          # force on geom2 from geom1 (world)
            on_door = fw if door2 else -fw
            if self.pin_geom in (c.geom1, c.geom2) and rob in self.finger_geoms:
                f_bar += on_door
                squeeze += self.f6[0]
            else:
                f_other += on_door
        weld = np.zeros(3)
        if d.eq_active[self.eq]:
            rows = np.flatnonzero((d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id == self.eq))
            weld = -d.efc_force[rows[:3]]        # sign: checked against the opening direction in tests/test_door.py
        th = float(d.qpos[w.door_q[self.i]])
        p = d.xpos[m.body(f"pin{self.i}").id]
        rv = p - self.hinge
        rv[2] = 0.0
        rhat = rv / max(np.linalg.norm(rv), 1e-9)
        that = pin_tangent(self.i, th)
        f = f_bar + f_other + weld
        tau = float(np.cross(rv, f) @ np.array([0.0, 0.0, -1.0]))      # about the hinge axis, + opens
        jt = np.abs(d.actuator_force[w.act_arm]) / m.actuator_forcerange[w.act_arm, 1]
        self.rows.append((d.time, th, f @ that, f @ rhat, f[2], squeeze / 2, np.linalg.norm(f_other), tau,
                          float(np.max(jt)), jaw_half_gap(w)))

    def summary(self, mu=1.2):
        if not self.rows:
            return {}
        a = np.array([r[:10] for r in self.rows], float)
        ft, fr, fz, sq, fo, tau, jt = (np.abs(a[:, k]) for k in (2, 3, 4, 5, 6, 7, 8))
        frz = np.hypot(a[:, 3], a[:, 4])
        p = lambda v, q=95: round(float(np.percentile(v, q)), 2)
        return {"n": len(a), "mean_push_N": round(float(a[:, 2].mean()), 3), "mean_hinge_Nm": round(float(a[:, 7].mean()), 3),
                "peak_push_N": round(float(ft.max()), 2), "p95_push_N": p(ft),
                "peak_side_N": round(float(frz.max()), 2), "p95_side_N": p(frz),
                "peak_radial_N": round(float(fr.max()), 2), "peak_vertical_N": round(float(fz.max()), 2),
                "mean_squeeze_N": round(float(sq.mean()), 2), "peak_other_contact_N": round(float(fo.max()), 2),
                "peak_hinge_Nm": round(float(tau.max()), 3), "peak_joint_torque_frac": round(float(jt.max()), 3),
                "side_over_2mu_N": round(float(np.percentile(frz, 99)) / (2 * mu), 2)}


# ---------------------------------------------------------------- planning
def plan_arc(ik, w, i, theta0, theta1, dx_set=None, rng=None, n_seeds=6, which="outer", return_all=False,
             fix_start=False):
    """Whole-body plan for dragging the handle bar from theta0 to theta1.

    The jaws close along the handle's direction of travel, so they push the bar instead of relying on friction.
    The bar spins freely in its bracket, so rotation about the jaw axis (the approach pitch) stays free, and so
    does the carriage position; both are chosen by IK continuation.
    Searches start positions and seeds; score = worst manipulability x joint-limit margin along the arc."""
    xc, s = C.printer_frame(i)
    rng = rng or np.random.default_rng(i)
    dx_set = np.round(np.arange(0.0, 0.61, 0.1), 3) if dx_set is None else dx_set
    thetas = np.linspace(theta0, theta1, ARC_N + 1)
    pin_name = f"{'i' if which == 'inner' else ''}pin{i}"
    ignore = {w.m.body(pin_name).id}
    lo, hi = sorted((xc + s * RAIL_MIN_L, xc + s * 0.9))
    eye = np.eye(3)
    q_pref = np.array([1.57 * s, 1.1, 1.3, -0.6, 0.0, 0.0])     # arm facing its own printer, elbow up
    best, found = None, []
    for dx in dx_set:
        seeds = [np.array([1.57 * s, 1.2, 1.2, 0, 0, 0])] + [rng.uniform(ik.lo, ik.hi) for _ in range(n_seeds)]
        for q0 in seeds:
            q, x = q0, xc + s * dx
            qs, xs, worst, a_prev = [], [], np.inf, None
            for k, th in enumerate(thetas):
                bar, a = pin_world(i, th, which), a_prev
                for _ in range(3):      # pinch = bar + GRASP_DEPTH along the approach axis the solution ends up with
                    tgt = bar if a is None else bar + GRASP_DEPTH * a
                    q, x, ok, _ = ik.solve(tgt, eye, x, q0=q, iters=150, align_y=pin_tangent(i, th, which),
                                           rail_free=not (fix_start and k == 0), rail_bounds=(lo, hi), q_pref=q_pref,
                                           x_pref=xc + s * dx)
                    a_new = ik.fk(q, x)[1][:, 2]
                    if a is not None and np.linalg.norm(a_new - a) * GRASP_DEPTH < 5e-4:
                        break
                    a = a_new
                if ok:
                    p_now, R_now = ik.fk(q, x)
                    ok = np.linalg.norm(p_now - GRASP_DEPTH * R_now[:, 2] - bar) < 2e-3
                if not ok or ik.collides(q, x, {i: th}, grip=0.006, ignore_bodies=ignore):
                    break
                a_prev = a
                worst = min(worst, ik.manipulability(q, x) * ik.limit_margin(q))
                qs.append(q.copy())
                xs.append(x)
            if len(qs) < len(thetas):
                continue
            # pre-grasp: back off along the solved approach axis
            p0, R0 = ik.fk(qs[0], xs[0])
            pre = p0 - (PRE_BACK + GRASP_DEPTH) * R0[:, 2]
            qp, _, ok, _ = ik.solve(pre, R0, xs[0], q0=qs[0], symmetric=False)
            if not ok or ik.collides(qp, xs[0], {i: theta0}, grip=0.037, ignore_bodies=ignore):
                continue
            cand = {"which": which, "x": xs[0], "xs": xs, "dx": float(dx), "score": worst, "q_pre": qp, "qs": qs,
                    "thetas": thetas, "R0": R0, "p0": p0, "R1": ik.fk(qs[-1], xs[-1])[1], "p1": ik.fk(qs[-1], xs[-1])[0]}
            found.append(cand)
            if best is None or worst > best["score"]:
                best = cand
            break
    if return_all:
        return sorted(found, key=lambda c: -c["score"])
    return best


def _plan_q(ik, q0, q1, ctx, iters=3000):
    """Joint-space path at a fixed carriage position; splits through the tuck pose if the direct search fails."""
    from plan.rrt import JointPlanner
    jp = JointPlanner(ik)
    path = jp.plan(q0, q1, ctx, max_iters=iters)
    if path is None and not np.allclose(q0, TUCK) and not np.allclose(q1, TUCK):
        a = jp.plan(q0, TUCK, ctx, max_iters=iters)
        b = jp.plan(TUCK, q1, ctx, max_iters=iters) if a is not None else None
        path = None if b is None else a + b[1:]
    return path


class SlideProbe:
    """Executor hook for one phase of a carriage move (reporting only, SIM truth): ticks with a contact between the
    arm (or a held part) and anything not on the carriage, the deepest such contact, the pairs, and in a slide the
    sway: how far the pinch strays from where a rigid arm on the moving carriage would put it."""

    def __init__(self, w, held=(), slide=False):
        m = w.m
        self.w, self.slide = w, slide
        self.arm = np.zeros(m.ngeom, bool)
        self.arm[list(w.arm_geoms | set(held))] = True
        self.rig = np.zeros(m.ngeom, bool)
        self.rig[list(w.robot_geoms | set(held))] = True
        self.p0 = w.d.site_xpos[w.grasp_site].copy()
        self.x0 = float(w.d.qpos[w.rail_q])
        self.ticks, self.pen, self.sway, self.pairs = 0, 0.0, 0.0, collections.Counter()

    def __call__(self, w):
        m, d = w.m, w.d
        if self.slide:
            rigid = self.p0 + np.array([float(d.qpos[w.rail_q]) - self.x0, 0.0, 0.0])
            self.sway = max(self.sway, float(np.linalg.norm(d.site_xpos[w.grasp_site] - rigid)))
        hit = False
        for c in d.contact[:d.ncon]:
            if c.dist >= 0.0 or self.arm[c.geom1] == self.arm[c.geom2]:
                continue
            if self.rig[c.geom2 if self.arm[c.geom1] else c.geom1]:
                continue
            hit = True
            self.pen = max(self.pen, -float(c.dist))
            self.pairs[f"{m.body(m.geom_bodyid[c.geom1]).name}-{m.body(m.geom_bodyid[c.geom2]).name}"] += 1
        self.ticks += hit


def _run_phase(ex, rec, phase, fn, slide=False):
    """Run fn() with a SlideProbe hooked in; add its time and contacts to the move record (reporting only)."""
    w = ex.w
    ik_held = getattr(ex, "_ik_held", ())
    probe = SlideProbe(w, ik_held, slide)
    hooks = list(getattr(w, "hooks", []))
    w.hooks = hooks + [probe]
    t0 = w.d.time
    try:
        fn()
    finally:
        w.hooks = hooks
    rec[f"{phase}_s"] = rec.get(f"{phase}_s", 0.0) + w.d.time - t0
    rec["contact_ticks"] = rec.get("contact_ticks", 0) + probe.ticks
    if probe.ticks:
        rec[f"{phase}_contacts"] = rec.get(f"{phase}_contacts", 0) + probe.ticks
    rec["max_pen_mm"] = max(rec.get("max_pen_mm", 0.0), 1000 * probe.pen)
    pairs = rec.setdefault("pairs", {})
    for k, v in probe.pairs.items():
        pairs[k] = pairs.get(k, 0) + v
    if slide:
        rec["sway_mm"] = round(1000 * probe.sway, 2)


def _log_move(w, rec):
    if not hasattr(w, "transfer_log"):
        w.transfer_log = []
    rec["total_s"] = sum(rec.get(f"{k}_s", 0.0) for k in ("tuck", "slide", "unfold", "path"))
    w.transfer_log.append(rec)


def motion_summary(log):
    """Carriage moves made by transfer(): slide pose used, times, contacts, sway (SIM truth, reporting only)."""
    log = list(log or [])
    by = collections.Counter(r["pose"] for r in log)
    tot = lambda k: round(float(sum(r.get(k, 0.0) for r in log)), 2)
    sway = collections.defaultdict(float)
    pairs = collections.Counter()
    for r in log:
        sway[r["pose"]] = max(sway[r["pose"]], r.get("sway_mm", 0.0))
        pairs.update(r.get("pairs", {}))
    n = len(log)
    return {"carriage_moves": n, "by_pose": dict(by), "folds_low_tuck": by["low"],
            "folds_per_move": round(by["low"] / max(n, 1), 3),
            "transfer_s": tot("total_s"), "slide_s": tot("slide_s"), "tuck_s": tot("tuck_s"),
            "unfold_s": tot("unfold_s"), "path7d_s": tot("path_s"),
            "mean_move_s": round(tot("total_s") / max(n, 1), 2),
            "contact_ticks": int(sum(r.get("contact_ticks", 0) for r in log)),
            "moves_with_contact": int(sum(r.get("contact_ticks", 0) > 0 for r in log)),
            "max_pen_mm": round(max([r.get("max_pen_mm", 0.0) for r in log] or [0.0]), 2),
            "max_sway_mm": {k: round(v, 2) for k, v in sway.items()}, "contact_pairs": dict(pairs.most_common(10))}


def transfer(ex, ik, q_goal, x, ctx_extra=None):
    """Collision-free move of arm (and carriage) to (q_goal, x): joint-space planning at a fixed carriage position,
    or, when the carriage moves, into a slide pose -> checked slide -> out to the goal, falling back to planning in
    joint+rail space. Slide poses, cheapest first: the current pose if compact(), the upright candle TUCK, the folded
    LOW_TUCK (under open doors); the first with a free slide over the whole rail interval (believed doors, own rig
    ignored, SLIDE_CLEAR_TALL / SLIDE_CLEAR), a free path into it and a free path out of it is used.
    A held part is planned as part of the arm throughout (IK.attach), parts lying on the carriage ride along."""
    from plan.rrt import JointPlanner, RAIL_W
    w = ex.w
    ex._ik_held = tuple(ik.held[0]) if ik.held is not None else ()
    sync_beliefs(ik, w)          # doors as the robot believes them (reed switch + FK estimates), not truth
    ctx = {"x": x, "grip": max(ex.finger_gap_mm(), 0.0) / 2000.0}    # jaw opening from the gripper encoder
    ctx.update(ctx_extra or {})
    q_goal = np.asarray(q_goal, float)
    if abs(ex.x - x) < 1e-4:
        path = _plan_q(ik, ex.q, q_goal, ctx)
        if path is None:
            if VERBOSE:
                print("    transfer: no joint path at fixed carriage")
            return False
        ex.move_path(path, label="transfer")
        return True
    here = dict(ctx, x=ex.x)
    rig = set(ctx.get("ignore", ())) | own_rig(w)     # the arm slides rigidly with its own deck and caddy
    rec = {"t": round(float(w.d.time), 2), "dx": round(float(x - ex.x), 4), "held": ik.held is not None}
    # slide poses, cheapest first: the current pose if compact, the upright candle, folded low (passes under open
    # doors). The first whose slide (whole rail interval, believed doors), path in and path out are free wins.
    cands = [("hold", ex.q.copy(), SLIDE_CLEAR_TALL)] if compact(ik, ex.q, ex.x) else []
    cands += [("upright", TUCK, SLIDE_CLEAR_TALL), ("low", LOW_TUCK, SLIDE_CLEAR)]
    why = []
    for k, (name, q_s, clr) in enumerate(cands):
        if any(np.allclose(q_s, c[1], atol=1e-3) for c in cands[:k]):
            continue                                       # e.g. the current pose IS the candle
        at = slide_blocked(ik, q_s, ex.x, x, ctx["grip"], rig, clr)
        if at is not None:
            why.append(f"{name} slide blocked at x={at:.3f}")
            if VERBOSE and name == "low":
                hits = sorted({(ik.m.body(ik.m.geom_bodyid[c.geom1]).name, ik.m.body(ik.m.geom_bodyid[c.geom2]).name,
                                round(1000 * c.dist, 1)) for c in ik.d.contact[:ik.d.ncon]
                               if (c.geom1 in w.arm_geoms) != (c.geom2 in w.arm_geoms) and c.dist < ik.MARGIN})
                print(f"    slide blocked at x={at:.3f} (from {ex.x:.3f} to {x:.3f}): {hits[:4]}")
            continue
        same_in = np.allclose(ex.q, q_s, atol=1e-3)
        p_in = None if same_in else _plan_q(ik, ex.q, q_s, here)
        if not same_in and p_in is None:
            why.append(f"{name}: no path into it")
            continue
        same_out = np.allclose(q_s, q_goal, atol=1e-3)
        p_out = None if same_out else _plan_q(ik, q_s, q_goal, ctx)
        if not same_out and p_out is None:
            why.append(f"{name}: no path out of it")
            continue
        rec["pose"] = name
        if p_in is not None:
            _run_phase(ex, rec, "tuck", lambda: ex.move_path(p_in, label="tuck"))
        _run_phase(ex, rec, "tuck", lambda: ex.settle(*SLIDE_SETTLE))   # the slide starts with the arm at rest

        _run_phase(ex, rec, "slide", lambda: slide(ex, x, name), slide=True)
        if p_out is not None:
            _run_phase(ex, rec, "unfold", lambda: ex.move_path(p_out, label="unfold"))
        _log_move(w, rec)
        return True
    lo, hi = min(ex.x, x) - 0.3, max(ex.x, x) + 0.3
    path = JointPlanner(ik, rail_bounds=(lo, hi)).plan(np.r_[ex.q, ex.x * RAIL_W], np.r_[q_goal, x * RAIL_W], ctx,
                                                       max_iters=4000)
    if path is None:
        if VERBOSE:
            print(f"    transfer: no path ({'; '.join(why)}; 7-D FAIL)")
        _log_move(w, dict(rec, pose="fail"))
        return False
    rec["pose"] = "7d"

    def run_path():
        for v in path[1:]:
            ex.move(v[:6], v[6] / RAIL_W, label="transfer")
    _run_phase(ex, rec, "path", run_path)
    _log_move(w, rec)
    return True


def _close_on_bar(ex, label):
    """Close the jaws around the bar with SQUEEZE_N per pad (position servo with the target inside the bar: the
    coupled jaws split the actuator force kp * error between the two pads)."""
    w = ex.w
    kp = float(w.m.actuator_gainprm[w.act_grip, 0])
    ex.jaws(max(HANDLE_R - 2 * SQUEEZE_N / kp, 0.0), 0.4, label=f"{label}:grip")


def execute_arc(ex, ik, i, plan, label, from_tuck=True):
    """Approach the handle bar, close the jaws on it, swing it along the planned arc (arm + rail), release and back
    off. Returns the elapsed sim time, or a failure string. Updates w.door_est from the arm's FK and appends the
    coupling-force summary of this arc to w.door_log."""
    w = ex.w
    which = plan["which"]
    t0 = w.d.time
    if not transfer(ex, ik, plan["q_pre"], plan["x"]):
        return "no collision-free transfer to the pre-pose"
    if float(w.d.ctrl[w.act_grip]) < GRIP_OPEN - 1e-4:     # an earlier step left the jaws shut: open them before
        ex.grip(False, duration=0.3, label=f"{label}:open")   # they go around the bar
    th0 = plan["thetas"][0]
    p0 = plan["p0"]
    down = cartesian_line(ik, p0 - (PRE_BACK + GRASP_DEPTH) * plan["R0"][:, 2], p0, plan["R0"], plan["x"],
                          plan["q_pre"], n=10, symmetric=False)
    ex.follow(down or [plan["qs"][0]], label=f"{label}:insert")
    ex.settle(tol=0.004, timeout=2.0)
    if COUPLING == "weld":
        ex.grip(True, duration=0.4, label=f"{label}:grip", width=0.013)
    else:
        _close_on_bar(ex, label)
    gap0 = jaw_half_gap(w, ex)
    th_meas, dr, dz = sense_door(ik, w, i, which, ex)
    if VERBOSE:
        print(f"    [{label}] plan start {np.degrees(th0):.1f} deg, door est {np.degrees(th_meas):.1f} "
              f"(true {np.degrees(w.door_angle(i)):.1f}) deg, jaw half-gap {1000 * gap0:.1f} mm, "
              f"bar offset r {1000 * dr:+.1f} z {1000 * dz:+.1f} mm")
    if not holding_bar(w, ex):
        ex.grip(False, duration=0.3)
        return f"handle not in the jaws (jaw half-gap {1000 * gap0:.1f} mm)"
    w.door_est[i] = th_meas
    if COUPLING == "weld":
        w.couple(i, True, which)
    probe = DoorProbe(w, i)
    hooks = list(getattr(w, "hooks", []))
    w.hooks = hooks + [probe]
    try:
        ex.follow(plan["qs"], plan["xs"], seg_time=ARC_SEG_S, label=f"{label}:arc")
        ex.settle()
    finally:
        w.hooks = hooks
    held = holding_bar(w, ex)
    th_end, dr, dz = sense_door(ik, w, i, which, ex)
    w.door_est[i] = th_end
    summ = probe.summary()
    summ.update(label=label, theta0_deg=round(float(np.degrees(th0)), 1),
                theta1_deg=round(float(np.degrees(plan["thetas"][-1])), 1),
                est_end_deg=round(float(np.degrees(th_end)), 2), true_end_deg=round(float(np.degrees(w.door_angle(i))), 2),
                held=held, coupling=COUPLING)
    w.door_log = getattr(w, "door_log", []) + [summ]
    if VERBOSE:
        print(f"    [{label}] arc end: plan {np.degrees(plan['thetas'][-1]):.1f} deg, est {np.degrees(th_end):.1f} "
              f"(true {np.degrees(w.door_angle(i)):.1f}) deg, still holding {held}; push p95 {summ.get('p95_push_N')} N, "
              f"side p95 {summ.get('p95_side_N')} N, other contact peak {summ.get('peak_other_contact_N')} N")
    if COUPLING == "weld":
        w.couple(i, False, which)
    ex.grip(False, label=f"{label}:release")
    p1 = plan["p1"]
    back = p1 - (PRE_BACK + GRASP_DEPTH) * plan["R1"][:, 2] + [0, 0, PRE_UP]
    out = cartesian_line(ik, p1, back, plan["R1"], plan["xs"][-1], plan["qs"][-1], n=8, symmetric=False)
    if out:
        ex.follow(out, label=f"{label}:retract")
    if not held:
        return f"lost the handle during the swing (door est {np.degrees(th_end):.0f} deg)"
    if VERBOSE:
        print(f"    [{label}] after retract: door {np.degrees(w.door_angle(i)):.1f} deg (true)")
    return w.d.time - t0


def compact(ik, q, x):
    """Can the arm slide in pose q as it is? Fingertips and a held part (planning model: as the robot believes it is
    held) within COMPACT_R of the base axis and at most COMPACT_DZ above the base."""
    m, d = ik.m, ik.d
    ik.fk(q, x)
    base = d.xpos[m.body("arm").id]
    pts = [d.site_xpos[ik.site]] + [d.xpos[b] for b in ik.w.finger_bodies]
    if ik.held is not None:
        pts.append(d.site_xpos[ik.site] + d.site_xmat[ik.site].reshape(3, 3) @ ik.held[2])
    pts = np.array(pts) - base
    return bool(np.max(np.hypot(pts[:, 0], pts[:, 1])) <= COMPACT_R and np.max(pts[:, 2]) <= COMPACT_DZ)


def slide_blocked(ik, q, x0, x1, grip, ignore, clearance, doors=None):
    """First rail position on x0 -> x1 at which the arm in pose q (with a held part) comes within `clearance` of the
    robot's believed scene (doors from sync_beliefs, `ignore` = the carriage's own rig), or None if the whole slide is
    free. Sampled every SLIDE_STEP and at both ends."""
    n = max(24, int(math.ceil(abs(x1 - x0) / SLIDE_STEP)))
    for k in range(n + 1):
        xk = x0 + (x1 - x0) * k / n
        if ik.collides(q, xk, doors, grip=grip, ignore_bodies=ignore, clearance=clearance):
            return xk
    return None


def slide(ex, x, pose):
    """The carriage move itself (arm joints held). Tall poses may be limited to SLIDE_ACC_TALL: a quintic move over
    distance L in T seconds peaks at 5.77 L / T^2."""
    from plan.motion import PAYLOAD_SCALE, RAIL_VMAX
    dur = None
    if pose != "low" and SLIDE_ACC_TALL:
        L = abs(x - ex.x)
        k = PAYLOAD_SCALE if ex.payload else 1.0
        dur = max(L / (RAIL_VMAX * k) * 1.6, 0.3, math.sqrt(5.7735 * L / SLIDE_ACC_TALL))
    ex.move(None, x, duration=dur, label="slide")


def own_rig(w):
    """Carriage bodies other than the arm (deck, caddy, bins): they move rigidly with the arm in a slide, so
    arm-vs-rig gaps cannot change there (model error must not make a slide look blocked)."""
    return w.robot_bodies - w._subtree("arm")


def slide_free(ik, i, x0, x1, theta):
    """Can the carriage slide x0 -> x1 with the arm folded low while door i stands at theta?"""
    rig = own_rig(ik.w)
    return all(not ik.collides(LOW_TUCK, x0 + (x1 - x0) * k / 24, {i: theta}, ignore_bodies=rig,
                               clearance=SLIDE_CLEAR) for k in range(25))


def plan_door(ik, w, i, opening, overpush=0.0):
    """Door plan on the single everyday handle, from the robot's belief of the door angle (reed switch / last FK
    estimate): one continuous whole-body swing if feasible; otherwise two swings with a regrasp of the same handle
    (typically from the other side of the bar), joined by a tucked carriage slide that must be collision-free with
    the door at the regrasp angle. Returns ([], "already ...") if there is nothing to do."""
    a = w.door_belief(i)
    b = C.DOOR_OPEN_RAD if opening else -overpush
    if opening and not w.reed_closed(i) and not np.isfinite(w.door_est[i]):
        return None, "door ajar (reed switch open) and its angle is unknown"
    if opening and a >= OPEN_OK_RAD:
        return [], "already open"
    if not opening and w.reed_closed(i):
        return [], "already closed"
    one = plan_arc(ik, w, i, a, b, which="outer", n_seeds=6)
    if one is not None:
        return [one], "ok"
    dxs = np.round(np.arange(-0.15, 0.71, 0.05), 2)
    best = None
    for m in (C.DOOR_REGRASP_RAD, math.radians(35), math.radians(65)):
        p1s = plan_arc(ik, w, i, a, m, which="outer", return_all=True, n_seeds=5, dx_set=dxs)
        p2s = plan_arc(ik, w, i, m, b, which="outer", return_all=True, n_seeds=5, dx_set=dxs) if p1s else []
        for p1 in p1s:
            for p2 in p2s:
                if slide_free(ik, i, p1["xs"][-1], p2["xs"][0], m):
                    sc = min(p1["score"], p2["score"]) - 0.002 * abs(p2["xs"][0] - p1["xs"][-1])
                    if best is None or sc > best[0]:
                        best = (sc, [p1, p2])
        if best is not None:
            return best[1], "ok"
    return None, "no single swing and no regrasp pair"


def operate_door(ex, ik, i, opening):
    """Plan and execute opening (or closing) of printer i's door. Returns (ok, seconds, note).
    Success is decided from sensors: opening = FK estimate at release >= OPEN_OK_RAD; closing = reed switch."""
    w = ex.w
    t0 = w.d.time
    tries = 1 if opening else CLOSE_TRIES
    note, repush = "ok", 0
    for attempt in range(tries):
        sync_beliefs(ik, w)
        plans, note = plan_door(ik, w, i, opening, overpush=OVERPUSH_RAD if attempt else 0.0)
        if plans is None:
            return False, w.d.time - t0, note
        for k, p in enumerate(plans):
            if k == 1 and abs(w.door_belief(i) - p["thetas"][0]) > np.radians(3):
                sync_beliefs(ik, w)
                xc, s = C.printer_frame(i)
                p = plan_arc(ik, w, i, w.door_belief(i), p["thetas"][-1], which=p["which"],
                             dx_set=[s * (p["xs"][0] - xc)], n_seeds=12)
                if p is None:
                    return False, w.d.time - t0, "door drifted and the second stage could not be re-planned"
            r = execute_arc(ex, ik, i, p, ("open" if opening else "close") + f":{p['which']}", from_tuck=k == 0)
            if isinstance(r, str):
                return False, w.d.time - t0, r
        if opening:
            break
        ex.wait(LATCH_DWELL_S)
        if w.reed_closed(i):
            break
        repush += 1
        if VERBOSE:
            print(f"    reed switch still open after close attempt {attempt + 1} "
                  f"(true {np.degrees(w.door_angle(i)):.2f} deg)")
    xc, s = C.printer_frame(i)
    if not transfer(ex, ik, TUCK, xc + s * 0.35):     # park upright (transfer folds low only if a slide needs it)
        return True, w.d.time - t0, "door done; could not park"
    if opening and w.door_belief(i) < OPEN_OK_RAD:
        return False, w.d.time - t0, f"door estimate {np.degrees(w.door_belief(i)):.0f} deg after opening"
    if not opening and not w.reed_closed(i):
        return False, w.d.time - t0, f"reed switch open after {tries} close attempts"
    return True, w.d.time - t0, note if not repush else f"{note} (re-pushed {repush}x)"
