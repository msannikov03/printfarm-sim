"""Grasp search for real parts (mesh SKUs): the fallback of plan.unload.plan_grasp when its own grasp finds none.

plan_grasp closes the jaws across the footprint's narrow side, or across the grasp cross-section of a real part, at
fixed heights (60 % of the part's height, 4 mm under its top). Real parts often have no grasp there: the wide outline
is low and only a small feature reaches those heights, the narrow side is near the 70 mm stroke, or the narrow side
points into the printer, where the arm cannot turn the jaws (jaws pointing into the printer are out of its reach).

The candidates come from the part's own layers, made when the plate is planned (sim/real_parts.py ->
real.mesh_io.grasp_candidates -> SKU.grasp_alts): jaw axis, approach tilt, finger position and tip height, scored by
jaw fit (stroke margin), cover and the distance from the mesh centroid, ordered by score x the arm's reach of the frame
(frame_prior) and spread over frames and gripped features. Here each candidate is checked in the planning scene, in
that order: IK (several carriage positions and seeds) with the lowest finger point put at the candidate's
tip height, collision-free grasp and pre-grasp with the jaws open (GRASP_CLEAR to the neighbours, the printer and the
door, as plan_grasp), the fingertips above the bed. The part itself is an obstacle too (its convex collision pieces at
the estimated pose): the open jaws clear it by SELF_CLEAR plus its pose uncertainty, as the wrist camera centres them
on it at the pre-grasp (plan_grasp ignores a small part's own body and keeps GRASP_CLEAR to a large one). The first
feasible candidate wins, at its best carriage position.

Only geometry the robot has: the G-code's footprint and height, the mesh in the job's 3MF (its layers), the camera's
pose estimate. Primitive SKUs have no candidates, so their grasps do not change.
"""
import collections
import math

import numpy as np

from farm import catalog as C

TILTS = (50, 65, 80)     # approach tilts searched (deg from vertical, about the jaw axis, leaning into the printer):
                         # through the door the YAM reaches the bed only this tilted (IK probes: 0-35 deg find none)
JAW_YAW_MAX = 60.0       # jaw axes within this of the bed's x axis (door plane): steeper ones reach the bed only from a
                         # few carriage positions, and there the arm meets the door frame; into the printer: never
TIP_TOL = 0.0008         # the lowest finger point is put within this of the candidate's tip height (m)
SELF_CLEAR = 0.0015      # the open jaws clear the part itself by this, plus its pose pad (plan.unload.PAD_K x sigma)
PLAN_CLEAR_MM = 3.0      # ... which the candidates allow for: SELF_CLEAR + the pad of a typical estimate (2 x 0.7 mm)
MAX_CANDS = 40           # candidates tried per part (in their order)
ENABLED = True           # plan.unload.plan_grasp uses the fallback (False: the grasp planner as before)
STATS = collections.Counter()   # why candidates failed (reporting: scripts/eval_real_models.py)


def candidate_dirs(part_yaw_deg, step=15):
    """Jaw axes to search in the part's own frame for a part lying at part_yaw_deg on the bed, with the lean of the
    approach into the printer (+y of the bed): [(yaw_deg, lean)]. Only jaw axes within JAW_YAW_MAX of the bed's x
    axis (the door plane): the arm cannot reach the bed with the jaws pointing into the printer."""
    out = []
    for th in range(0, 180, step):
        phi = (part_yaw_deg + th + 90.0) % 180.0 - 90.0          # jaw axis on the bed, (-90, 90]
        if abs(phi) > JAW_YAW_MAX:
            continue
        lean = 1 if math.cos(math.radians(part_yaw_deg + th)) > 0 else -1   # the jaw normal with +y on the bed
        out.append((float(th), lean))
    return out


TILT_PRIOR = {65: 1.0, 50: 0.8, 80: 0.6}     # how well the arm reaches the bed (IK probes over the bed, door open:
YAW_PRIOR = ((30.0, 1.0), (45.0, 0.8), (90.0, 0.6))   # tilt 65 the most, jaw axes near the door plane the most)


def frame_prior(part_yaw_deg):
    """Weight of a candidate frame (jaw yaw in the part frame, lean, tilt) for a part lying at part_yaw_deg: orders
    the candidates (real.mesh_io.grasp_candidates prior) by the arm's reach, which IK checks in the end anyway."""
    def w(yaw, lean, tilt):
        phi = abs((part_yaw_deg + yaw + 90.0) % 180.0 - 90.0)
        return TILT_PRIOR.get(int(tilt), 0.5) * next(v for lim, v in YAW_PRIOR if phi <= lim)
    return w


def plan_tilts(large):
    """Tilts to search for a part: at most plan.unload.LARGE_TILT for a large part (it is put down flat in a tray slot
    with the same tilt, and steeper jaws reach past the slot walls)."""
    from plan.unload import LARGE_TILT
    return tuple(t for t in TILTS if not large or t <= LARGE_TILT)


def _tip_z(ik, q, x, grip):
    """Lowest finger point (world z) with the arm at (q, x) and the jaws at grip (planning model)."""
    import mujoco
    from plan.unload import finger_min_z
    d = ik.d
    d.qpos[ik.qadr] = q
    d.qpos[ik.w.rail_q] = x
    d.qpos[ik.w.finger_q] = grip
    d.qpos[ik.w.finger_rq] = -grip
    mujoco.mj_kinematics(ik.m, d)
    return finger_min_z(ik)


def _blocked(ik, w, q, x, doors, grip, ignore, part):
    """Collision check of a grasp-side pose: GRASP_CLEAR to everything but the bed (TIP_CLEAR) and the part itself,
    SELF_CLEAR (+ its pad) to the part. -> None, "col_other" or "col_self"."""
    from plan.unload import GRASP_CLEAR
    if ik.collides(q, x, doors, grip=grip, ignore_bodies=ignore | {part}, clearance=GRASP_CLEAR):
        return "col_other"
    for c in ik.d.contact[:ik.d.ncon]:          # (the same forward pass: every contact within the robot's margin)
        r1, r2 = c.geom1 in w.arm_geoms, c.geom2 in w.arm_geoms
        if r1 != r2 and ik.m.geom_bodyid[c.geom2 if r1 else c.geom1] == part \
                and c.dist - ik.pad.get(part, 0.0) <= SELF_CLEAR:
            return "col_self"
    return None


def mesh_grasp(ik, w, i, body, est=None, rng=None, n_seeds=3):
    """The first feasible grasp of SKU.grasp_alts for pool body `body` on printer i (est: the camera's estimate; None:
    simulator truth, debug only). Returns a plan.unload.grasp_search candidate (+ "mesh": the candidate) or None."""
    import mujoco
    from plan import unload as U
    from plan.door import RAIL_MIN_L
    from plan.ik import frame
    sku = w.part_sku[body]
    k = C.CATALOG.get(sku)
    if k is None or not k.grasp_alts or min(k.footprint) > C.GRIPPER_MAX_MM:
        return None            # a primitive, a part too flat for the jaws, or refused (wider than the stroke both ways)
    if est is not None:
        pos, R = U.pose_from_estimate(i, est)
    else:
        pos, quat = w.part_pose(body)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        R = R.reshape(3, 3)
    xc, s = C.printer_frame(i)
    rng = rng or np.random.default_rng(0)
    bed_z = C.BENCH_H + C.BED_ZL
    into = np.array([0, s, 0.0])
    ignore = {w.m.body(f"bed{i}").id}                    # the bed is checked by TIP_CLEAR
    part = w.part_body[body]
    doors = {i: w.door_belief(i)}
    seeds = [np.array([1.57 * s, 1.0, 1.4, -0.9, 0, 0]), np.array([1.57 * s, 0.6, 1.9, -1.3, 0, 0])]
    max_tilt = U.LARGE_TILT if k.large else 90
    unreachable = []                 # (frame, pinch): candidates whose IK failed at every carriage position and seed
    for cand in k.grasp_alts[:MAX_CANDS]:
        cx, cy, yaw, lean, tilt, pz, wd, z_tip = cand[:8]
        if tilt > max_tilt:
            continue
        if any(f == (yaw, lean, tilt) and math.hypot(cx - px, cy - py) < 20.0 and abs(pz - pzz) < 6.0
               for f, (px, py, pzz) in unreachable):
            STATS["mesh:ik_frame"] += 1                  # the same arm pose nearly: no IK solution there either
            continue
        a, th = math.radians(tilt), math.radians(yaw)
        jaw = R @ np.array([math.cos(th), math.sin(th), 0.0])
        lean_v = lean * (R @ np.array([-math.sin(th), math.cos(th), 0.0]))
        if tilt and float(lean_v @ into) < 0.2:
            STATS["mesh:leans_out"] += 1                 # the estimate turned it: leaning out of the door
            continue
        approach = math.sin(a) * lean_v - math.cos(a) * np.array([0.0, 0.0, 1.0])
        Rf = frame(approach, jaw)
        target0 = pos + R @ np.array([cx / 1000, cy / 1000, 0.0])
        target0[2] = bed_z + pz / 1000
        width = wd / 1000
        grip = min(0.037, width / 2 + 0.008)
        tip_goal = bed_z + z_tip / 1000
        best, n_ik = None, 0
        for dx in U.DXS:
            x = xc + s * dx
            if s * (x - xc) < RAIL_MIN_L:
                continue
            for q0 in seeds + [rng.uniform(ik.lo, ik.hi) for _ in range(n_seeds)]:
                q, _, ok, _ = ik.solve(target0, Rf, x, q0=q0, iters=150)
                if not ok:
                    STATS["mesh:ik"] += 1
                    continue
                n_ik += 1
                _, Rs = ik.fk(q, x)              # (the jaws may come out turned 180 deg about the approach)
                target = target0.copy()
                for _ in range(3):        # lowest finger point to the tip height: slide along the approach (the finger
                    err = _tip_z(ik, q, x, grip) - tip_goal          # line the candidate was checked on)
                    if abs(err) < TIP_TOL:
                        break
                    target = target + approach * err / max(math.cos(a), 0.15)
                    q, _, ok, _ = ik.solve(target, Rs, x, q0=q, iters=80, symmetric=False)
                    if not ok:
                        break
                if not ok or abs(_tip_z(ik, q, x, grip) - tip_goal) >= TIP_TOL:
                    STATS["mesh:tip"] += 1
                    continue
                why = _blocked(ik, w, q, x, doors, grip, ignore, part)
                if why:
                    STATS["mesh:" + why] += 1
                    continue
                mujoco.mj_forward(ik.m, ik.d)
                if U.finger_min_z(ik) < bed_z + U.TIP_CLEAR:
                    STATS["mesh:tips_below_bed"] += 1
                    continue
                _, Rs = ik.fk(q, x)
                pre = target - U.PRE_BACK * Rs[:, 2]
                qp, _, okp, _ = ik.solve(pre, Rs, x, q0=q, symmetric=False)
                if not okp or _blocked(ik, w, qp, x, doors, grip, ignore, part):
                    STATS["mesh:pregrasp"] += 1
                    continue
                sc = ik.manipulability(q, x) * ik.limit_margin(q)
                if best is None or sc > best["score"]:
                    best = {"q": q, "x": x, "R": Rs, "q_pre": qp, "target": target, "pre": pre, "width": width,
                            "score": sc, "h": float(target[2] - bed_z), "tilt": int(tilt), "mesh": tuple(cand)}
                break
        if best is not None:
            STATS["mesh:found"] += 1
            return best
        if not n_ik:
            unreachable.append(((yaw, lean, tilt), (cx, cy, pz)))
    STATS["mesh:none"] += 1
    return None
