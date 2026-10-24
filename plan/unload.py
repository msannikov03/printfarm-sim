"""Unload a printer bed: spawn a plate from the slicer output, plan grasps from the known geometry, pick each
part, drop it into its order's caddy bin.

Grasp synthesis (no learning): the slicer gives each part's SKU, centre, yaw and height. Jaws close across the
narrow side of the footprint (any horizontal direction for cylinders); the approach direction about the jaw axis,
the grasp height and the carriage position are searched; the fingertips must stay above the bed.

Large SKUs (catalog SKU.large: longer than a tote takes) go to a slot of the large-part tray (C.LARGE_SLOTS) instead
of a bin: they are not dropped but placed (place_pose): lowered flat into the slot, long axis along the slot's x,
released PLACE_ABOVE over the floor, then confirmed in the slot by the wrist camera (slot_seen).
"""
import math

import mujoco
import numpy as np

from farm import catalog as C
from plan.door import RAIL_MIN_L, TUCK, VERBOSE, transfer
from plan.motion import cartesian_line

PRE_BACK = 0.07
LIFT = 0.04
TIP_CLEAR = 0.0015       # fingertips must stay this far above the bed (m)
DROP_ABOVE = 0.035       # held part's centre above the bin rim when releasing
SHAKES = 2               # wrist-roll shakes when a released part stays on the fingers

# ---- realism switches (scripts/eval_unload.py --realism sets these) ----
CAMERA_NOISE = "ideal"   # chamber-camera noise preset (perceive/noise.py)
CAMERA_SEED = 0          # per-episode seed of the camera noise stream (set by eval / farm)
RESCANS = 2              # a failed scan (no frame, blackout) is retried this many times before acting
VERIFY_RESCANS = 2       # final bed check: an occluded scan that cannot verify the bed is repeated this often
CAMERA_CORRECTIONS = True  # robust estimator (one-to-one matching, plane/outline fit, Bayesian fusion) even on
                           # perfect depth: the legacy estimator could match a missing part to an identical neighbour
ORACLE = False           # True: legacy planning/checks from simulator truth (A/B comparison only)
WRIST_ALIGN = True       # wrist-camera alignment at the pre-grasp (corrects camera + arm calibration errors)
PAD_K = 2.0              # believed obstacles are inflated by PAD_K x their pose sigma
SIGMA_DEFAULT_MM = 0.7   # pose sigma when the estimate carries none (measured mean error with perfect depth)
N_BETWEEN = 25           # wrist-camera points inside the jaw volume that mean "something is between the jaws"
WRIST_MAX_SHIFT = 0.010  # larger wrist corrections are treated as a misdetection (m)
PLACE_ABOVE = 0.010      # large parts: released with the underside this far above the tray-slot floor (m)
PLACE_LIFT = 0.06        # ... approached and left straight down / up from this far above the release pose (m)
RELEASE_OPEN = 0.006     # ... the jaws open only this far past the part per side (the slot walls are close)
N_SLOT = 40              # wrist-camera points on the part's top face that confirm it lies in the slot
LARGE_TILT = 65          # large parts: steepest approach tilt (deg)
EXIT_HEIGHTS = (0.17, 0.11, 0.08)   # straight exit out of the door at this height above the bed (m); a long part held
                                    # with a steep tilt brings the forearm to the door header when lifted high: lower


def is_large(w, body):
    k = C.CATALOG.get(w.part_sku[body])
    return k is not None and k.large


def part_name_for(obj):
    return obj["name"]


def spawn_plate(w, plate, rng=None, pos_sigma_mm=0.0, yaw_sigma_deg=0.0, stuck=True):
    """Spawn every object (and the purge tab) on printer `plate['printer']`. Returns {object name: pool body}."""
    i = plate["printer"]
    _, s = C.printer_frame(i)
    rng = rng or np.random.default_rng(0)
    placed = {}
    for o in plate["objects"]:
        body = w.free_part(o["sku"])
        dx, dy = rng.normal(0, pos_sigma_mm, 2) if pos_sigma_mm else (0.0, 0.0)
        yaw = math.radians(o["yaw_deg"] + (rng.normal(0, yaw_sigma_deg) if yaw_sigma_deg else 0.0))
        w.spawn(body, C.bed_mm_to_world(i, o["center_mm"][0] + dx, o["center_mm"][1] + dy), yaw + (0 if s > 0 else math.pi),
                stuck=stuck)
        placed[o["name"]] = body
    pt = plate["purge_tab"]
    body = w.free_part("purge_tab")
    w.spawn(body, C.bed_mm_to_world(i, *pt["center_mm"]), 0.0 if s > 0 else math.pi, stuck=stuck)
    placed[f"{plate['plate_id']}_purge_tab"] = body
    return placed


def finger_min_z(ik):
    """Lowest point of the finger collision geometry in the IK scratch state (after mj_forward)."""
    m, d = ik.m, ik.d
    zmin = np.inf
    for g in range(m.ngeom):
        if m.geom_bodyid[g] not in ik.w.finger_bodies or not m.geom_contype[g]:
            continue
        R = d.geom_xmat[g].reshape(3, 3)
        c = d.geom_xpos[g]
        size = m.geom_size[g]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            ext = np.abs(R) @ size
            zmin = min(zmin, c[2] - ext[2])
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            zmin = min(zmin, c[2] - abs(R[2, 2]) * size[1] - size[0])
        else:
            zmin = min(zmin, c[2] - size[0])
    return zmin


def pose_from_estimate(i, e):
    """Perceived (bed mm, yaw deg) -> world position of the footprint centre at bed level and rotation."""
    _, s = C.printer_frame(i)
    pos = np.array(C.bed_mm_to_world(i, *e["center_mm"]))
    yaw = math.radians(e["yaw_deg"]) + (0.0 if s > 0 else math.pi)
    c, sn = math.cos(yaw), math.sin(yaw)
    return pos, np.array([[c, -sn, 0], [sn, c, 0], [0, 0, 1.0]])


def grasp_spec(w, body, est=None, i=None):
    """Grasp point (world), jaw options [(axis or None, width)] (narrow side first) and candidate heights.
    With `est` (perception estimate) the pose comes from the camera; otherwise from the simulator (debug only)."""
    sku = w.part_sku[body]
    if est is not None:
        pos, R = pose_from_estimate(i, est)
    else:
        pos, quat = w.part_pose(body)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        R = R.reshape(3, 3)
    gmax = C.GRIPPER_MAX_MM / 1000
    if sku == "purge_tab":                                 # the grab tab at the end of the prime line
        tx, ty, tz = (v / 1000 for v in C.PURGE_TAB_MM)
        return pos, [(R[:, 0], tx), (R[:, 1], ty)], [tz - 0.004, tz * 0.6]   # high on the tab, away from the line
    k = C.CATALOG[sku]
    d = [v / 1000 for v in k.dims]
    if k.kind == "cyl":
        h = d[1]
        return pos, [(None, d[0])], [h * 0.6, max(h - 0.004, 0.006)]
    if k.kind == "L":
        ln, wd, bt, uh, ut = d
        p = pos + R @ np.array([-ln / 2 + ut / 2, 0, 0])
        return p, [(R[:, 0], ut), (R[:, 1], wd)], [uh * 0.7, uh - 0.004]
    fx, fy, h = d
    opts = sorted([(R[:, 0], fx), (R[:, 1], fy)], key=lambda o: o[1])
    hs = [h * 0.6, max(h - 0.004, 0.005)] + ([h - 0.0025] if k.large else [])   # a low large part: high in its top
    return pos, [o for o in opts if o[1] <= gmax], hs                           # edge, so a moderate tilt reaches


MIN_COVER = 0.005       # the jaws must cover at least this much of the part's height
GRASP_CLEAR = 0.004     # clearance to neighbouring parts / bed for the final grasp (transfers use the 1.5 cm margin)
TILTS = (0, 20, 35, 50, 65, 80, 90)
DXS = (0.0, 0.15, -0.1, 0.3, 0.45)


def grasp_frames(jaw, into):
    """Explicit grasp orientations: jaws along `jaw` (horizontal), approach tilted about the jaw axis by each tilt,
    toward (+) or away from (-) the printer interior. For cylinders (jaw=None) try two jaw directions."""
    from plan.ik import frame
    jaws = [jaw] if jaw is not None else [np.array([1.0, 0, 0]), np.array([0, 1.0, 0]),
                                          np.array([0.7071, 0.7071, 0]), np.array([0.7071, -0.7071, 0])]
    out = []
    for j in jaws:
        u = np.cross([0, 0, 1.0], j)
        u = u if np.dot(u, into) >= 0 else -u                    # horizontal, pointing into the printer
        for t in TILTS:
            for sgn in ((1, -1) if t else (1,)):
                a = math.radians(t)
                approach = sgn * math.sin(a) * u - math.cos(a) * np.array([0, 0, 1.0])
                out.append((t * sgn, frame(approach, j)))
    out.sort(key=lambda tf: (tf[0] < 0, abs(tf[0])))             # prefer tilting into the printer, then small tilt
    return out


REJECT = None            # set to a collections.Counter() to tally grasp-candidate rejections (debugging)


def _rej(why):
    if REJECT is not None:
        REJECT[why] += 1


def plan_grasp(ik, w, i, body, rng=None, n_seeds=3, est=None):
    """Search grasp height x orientation x carriage position x IK seed for a collision-free grasp with the fingertips
    above the bed and a collision-free pre-grasp. Returns the best-scoring candidate or None."""
    p0, jaw_opts, heights = grasp_spec(w, body, est, i)
    sku = w.part_sku[body]
    part_h = (C.PURGE_TAB_MM[2] if sku == "purge_tab" else C.CATALOG[sku].height) / 1000
    # a long part stays an obstacle (its believed pose): a steep approach along it would put the palm on its far end;
    # the open jaws keep PRESHAPE > GRASP_CLEAR from its sides. At most LARGE_TILT: it is put down flat in a tray slot
    # with the same tilt, and steeper jaws reach past the slot walls
    if is_large(w, body):
        return grasp_search(ik, w, i, p0, jaw_opts, heights, part_h, set(), rng, n_seeds, max_tilt=LARGE_TILT)
    return grasp_search(ik, w, i, p0, jaw_opts, heights, part_h, {w.part_body[body]}, rng, n_seeds)


def grasp_search(ik, w, i, p0, jaw_opts, heights, part_h, ignore, rng=None, n_seeds=3, max_tilt=90):
    """The grasp search behind plan_grasp, for any graspable thing described by a grasp point p0 (world, bed level),
    jaw options [(axis or None, width)] and candidate heights above the bed (approach tilt up to max_tilt deg)."""
    xc, s = C.printer_frame(i)
    rng = rng or np.random.default_rng(0)
    bed_z = C.BENCH_H + C.BED_ZL
    into = np.array([0, s, 0.0])
    ignore = set(ignore) | {w.m.body(f"bed{i}").id}
    doors = {i: w.door_belief(i)}
    seeds = [np.array([1.57 * s, 1.0, 1.4, -0.9, 0, 0]), np.array([1.57 * s, 0.6, 1.9, -1.3, 0, 0])]
    best = None
    frames = [(tilt, R, width) for jaw, width in jaw_opts for tilt, R in grasp_frames(jaw, into)
              if abs(tilt) <= max_tilt]
    frames.sort(key=lambda f: (f[0] < 0, abs(f[0])))           # into the printer first, then small tilt
    for tilt, R, width in frames:
        grip = max(width / 2 + 0.004, 0.012)
        approach = R[:, 2]
        low = [0.0115 * abs(approach[2]) + 0.006] if part_h < 0.015 else []   # flat parts only
        for h in list(heights) + low:
            target = p0.copy()
            target[2] = bed_z + h
            for dx in DXS:
                x = xc + s * dx
                if s * (x - xc) < RAIL_MIN_L:
                    continue
                for q0 in seeds + [rng.uniform(ik.lo, ik.hi) for _ in range(n_seeds)]:
                    q, _, ok, _ = ik.solve(target, R, x, q0=q0, iters=150)
                    if not ok:
                        _rej("ik")
                        continue
                    if ik.collides(q, x, doors, grip=min(0.037, width / 2 + 0.008), ignore_bodies=ignore,
                                   clearance=GRASP_CLEAR):
                        if REJECT is not None:
                            for c in ik.d.contact[:ik.d.ncon]:
                                a1, a2 = c.geom1 in w.arm_geoms, c.geom2 in w.arm_geoms
                                if a1 != a2 and c.dist <= GRASP_CLEAR:
                                    o = c.geom2 if a1 else c.geom1
                                    if w.m.geom_bodyid[o] not in ignore:
                                        _rej(f"col:{w.m.body(w.m.geom_bodyid[c.geom1 if a1 else c.geom2]).name}"
                                             f"-{w.m.body(w.m.geom_bodyid[o]).name}")
                                        break
                        continue
                    mujoco.mj_forward(ik.m, ik.d)
                    fz = finger_min_z(ik)
                    if fz < bed_z + TIP_CLEAR:
                        _rej("tips_below_bed")
                        continue
                    if fz > bed_z + part_h - min(MIN_COVER, 0.6 * part_h):
                        _rej("shallow_grasp")                     # pads would only touch the part's top edge
                        continue
                    _, Rs = ik.fk(q, x)
                    pre = target - PRE_BACK * Rs[:, 2]
                    qp, _, okp, _ = ik.solve(pre, Rs, x, q0=q, symmetric=False)
                    pre_grip = min(0.037, width / 2 + 0.008)
                    if not okp or ik.collides(qp, x, doors, grip=pre_grip, ignore_bodies=ignore, clearance=GRASP_CLEAR):
                        _rej("pregrasp")
                        continue
                    sc = ik.manipulability(q, x) * ik.limit_margin(q)
                    cand = {"q": q, "x": x, "R": Rs, "q_pre": qp, "target": target, "pre": pre, "width": width,
                            "score": sc, "h": h, "tilt": tilt}
                    if best is None or sc > best["score"]:
                        best = cand
                    break
            if best is not None:
                return best          # first feasible (orientation, height) in preference order, best carriage x
    return best


def drop_pose(ik, w, slot, x, rel=None, above=DROP_ABOVE):
    """Joint configuration placing the held part (offset `rel` in the pinch frame) `above` the bin rim.
    A large-tray slot is not a drop: it routes to place_pose (the held part as attached) and returns the approach
    pose PLACE_LIFT above the release."""
    if slot in C.LARGE_SLOTS:
        pl = place_pose(ik, w, slot, x)
        return None if pl is None else pl["q_up"]
    sid = mujoco.mj_name2id(w.m, mujoco.mjtObj.mjOBJ_SITE, f"bin_{slot}")
    ik.sync(w.d)
    ik.fk(TUCK, x)                       # the bins ride on the carriage: only the rail position matters here
    mujoco.mj_forward(ik.m, ik.d)
    target = ik.d.site_xpos[sid] + [0, 0, above]
    to_bin = target - [x, 0.0, C.BASE_Z]
    yaw = math.atan2(to_bin[1], to_bin[0])
    seeds = [TUCK] + [np.array([yaw + dy, j2, j3, -1.2, 0, 0]) for dy in (0.0, 0.4, -0.4)
                      for j2, j3 in ((0.3, 2.0), (0.8, 2.6), (0.5, 1.6))]
    r = np.array([to_bin[0], to_bin[1], 0.0]) / max(np.hypot(to_bin[0], to_bin[1]), 1e-6)
    best = None
    for tilt in (0.0, 20.0, 35.0):                      # a drop may lean outward a little
        a = math.radians(tilt)
        down = np.array([0, 0, -math.cos(a)]) + math.sin(a) * r
        for q0 in seeds:
            q0 = np.clip(q0, ik.lo, ik.hi)
            tgt = target.copy()
            for _ in range(3 if rel is not None else 1):     # aim the part, not the fingers, at the bin centre
                q, _, ok, _ = ik.solve(tgt, np.eye(3), x, q0=q0, iters=200, align_z=down)
                if not ok or rel is None:
                    break
                _, R = ik.fk(q, x)
                tgt = target - R @ rel
                q0 = q
            if ok and not ik.collides(q, x, None, grip=0.02):
                sc = ik.limit_margin(q) - 0.01 * tilt / 35
                if best is None or sc > best[0]:
                    best = (sc, q)
        if best is not None:
            break
    return None if best is None else best[1]


def slot_floor(ik, w, slot, x):
    """World centre of the floor of caddy bin / tray slot `slot` with the carriage at rail x (planning model)."""
    sid = mujoco.mj_name2id(w.m, mujoco.mjtObj.mjOBJ_SITE, f"bin_{slot}")
    ik.fk(TUCK, x)                       # the bins ride on the carriage: only the rail position matters here
    mujoco.mj_forward(ik.m, ik.d)
    return ik.d.site_xpos[sid] - [0.0, 0.0, slot_size(slot)[2] - 0.003]       # floor plate 3 mm (farm/scene._bin)


def place_pose(ik, w, slot, x, above=PLACE_ABOVE, width=None):
    """Lowered placement of the held (attached) long part into large-tray slot `slot` with the carriage at x: the part
    lies flat, long axis (its body x) along the slot's x, underside `above` the slot floor. The pinch pose follows
    from the part's pose in the jaws as the robot believes it (IK.attach). Checked: release pose with the jaws at the
    release opening, approach pose PLACE_LIFT higher, and the straight line between them (fine clearance).
    Returns {"q", "q_up", "down", "R", "p", "x", "target"} or None (p: pinch, target: the part's body origin)."""
    if ik.held is None:
        return None
    _, _, rel_p, rel_R = ik.held
    ik.sync(w.d)
    target = slot_floor(ik, w, slot, x) + [0.0, 0.0, above]
    g_rel = min(0.037, (width or 0.05) / 2 + RELEASE_OPEN)
    to = target - [x, 0.0, C.BASE_Z]
    yaw = math.atan2(to[1], to[0])
    seeds = [TUCK] + [np.array([yaw + dy, j2, j3, -1.2, 0, 0]) for dy in (0.0, 0.4, -0.4)
                      for j2, j3 in ((0.3, 2.0), (0.8, 2.6), (0.5, 1.6))]
    best = None
    for a in (0.0, math.pi):                         # boxes: the long axis along +x or -x of the slot
        Rpart = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1.0]])
        R = Rpart @ rel_R.T
        p = target - R @ rel_p
        for q0 in seeds:
            q, _, ok, _ = ik.solve(p, R, x, q0=np.clip(q0, ik.lo, ik.hi), iters=200, symmetric=False)
            if not ok or ik.collides(q, x, None, grip=g_rel, clearance=GRASP_CLEAR):
                continue
            q_up, _, ok, _ = ik.solve(p + [0, 0, PLACE_LIFT], R, x, q0=q, symmetric=False)
            if not ok or ik.collides(q_up, x, None, grip=(width or 0.05) / 2):
                continue
            down = cartesian_via(ik, [p + [0, 0, PLACE_LIFT], p], R, x, q_up, (), None, n=6, grip=(width or 0.05) / 2)
            if down is None:
                continue
            sc = min(ik.limit_margin(q), ik.limit_margin(q_up))
            if best is None or sc > best["score"]:
                best = {"q": down[-1], "q_up": q_up, "down": down, "R": R, "p": p, "x": x, "score": sc,
                        "target": target}
            break
    return best


def slot_seen(ex, ik, slot, part_h):
    """Wrist camera, from above the slot after the release: depth points on a top face at the part's height inside the
    slot (believed frame: encoders + the robot's model) -> count. An empty slot shows only its floor."""
    pts = _wrist(ex).points_pinch()
    p_b, R_b = ik.fk(ex.q_meas(), ex.x_meas())
    pw = pts @ R_b.T + p_b
    c = slot_floor(ik, ex.w, slot, ex.x_meas())
    sx, sy, _ = slot_size(slot)
    sel = (np.abs(pw[:, 0] - c[0]) < sx / 2 - 0.006) & (np.abs(pw[:, 1] - c[1]) < sy / 2 - 0.006) & \
          (np.abs(pw[:, 2] - c[2] - part_h) < max(0.004, 0.3 * part_h))
    return int(sel.sum())


def pick_and_place(ex, ik, i, body, slot, rng=None, est=None):
    """Full pick of one part into a caddy slot. Returns (ok, outcome, seconds, attempts)."""
    w = ex.w
    t0 = w.d.time
    others0 = {n: w.part_pose(n)[0].copy() for n in w.in_use if n != body}
    try:
        return _pick_and_place(ex, ik, i, body, slot, rng, t0, est)
    finally:
        ex.payload = False
        ik.detach()
        from plan import door as _door
        moved = {n: 1000 * np.linalg.norm(w.part_pose(n)[0] - p) for n, p in others0.items() if n in w.in_use}
        bumped = {n: round(v, 1) for n, v in moved.items() if v > 3.0}
        if bumped and _door.VERBOSE:
            print(f"    disturbed during this pick (mm): {bumped}")


def _wrist(ex):
    if getattr(ex, "wrist", None) is None:
        from farm.sensors import WristCam
        ex.wrist = WristCam(ex.w)
    return ex.wrist


def top_center(w, body, pos, R):
    """Believed centre of the part's top face (world, bed level) and its height, from its believed pose."""
    sku = w.part_sku[body]
    if sku == "purge_tab":
        return pos.copy(), C.PURGE_TAB_MM[2] / 1000
    k = C.CATALOG[sku]
    if k.kind == "L":
        ln, wd, bt, uh, ut = (v / 1000 for v in k.dims)
        return pos + R @ np.array([-ln / 2 + ut / 2, 0, 0]), uh
    return pos.copy(), k.height / 1000


def wrist_offset(ex, ik, w, i, body, est, verbose=False):
    """Wrist depth at the pre-grasp -> correction (m, believed world frame) of the part's position relative to the
    gripper: xy from the part's top face, z from the bed surface next to it. The wrist camera is hand-eye
    calibrated, so this cancels chamber-camera error AND arm/base calibration error. None if not measurable."""
    pts = _wrist(ex).points_pinch()
    p_b, R_b = ik.fk(ex.q_meas(), ex.x_meas())             # believed pinch pose (encoders + the robot's model)
    pw = pts @ R_b.T + p_b
    pos, R = pose_from_estimate(i, est)
    top, h = top_center(w, body, pos, R)
    bed_z = C.BENCH_H + C.BED_ZL
    sku = w.part_sku[body]
    fx, fy = C.PURGE_TAB_MM[:2] if sku == "purge_tab" else C.CATALOG[sku].footprint
    uv_all = (pw[:, :2] - pos[:2]) @ R[:2, :2]          # in the part's believed footprint frame
    large = is_large(w, body)
    if large:     # a disc around a long part reaches off the bed (plate edges, front gap): a band around its
        xc, s = C.printer_frame(i)                       # footprint rectangle instead, on the bed plate only
        bx, by = s * (pw[:, 0] - xc) + C.BED_MM[0] / 2000, s * pw[:, 1] - C.AISLE_HALF - C.BED_FRONT_GAP
        out = np.maximum(np.abs(uv_all[:, 0]) - fx / 2000, np.abs(uv_all[:, 1]) - fy / 2000)
        ring = (out > 0.004) & (out < 0.030) & (bx > 0.003) & (bx < C.BED_MM[0] / 1000 - 0.003) & \
               (by > 0.003) & (by < C.BED_MM[1] / 1000 - 0.003)
    else:
        r_part = 0.5 * np.hypot(fx, fy) / 1000
        dxy = np.hypot(pw[:, 0] - pos[0], pw[:, 1] - pos[1])
        ring = (dxy > r_part + 0.004) & (dxy < r_part + 0.030)
    ring &= np.abs(pw[:, 2] - bed_z) < 0.006
    if ring.sum() < 30:
        return None
    dz = float(np.median(pw[ring, 2]) - bed_z)
    near = (np.abs(uv_all[:, 0]) < fx / 2000 + 0.005) & (np.abs(uv_all[:, 1]) < fy / 2000 + 0.005) & \
           (np.abs(pw[:, 2] - dz - (bed_z + h)) < 0.0025)
    if near.sum() < 30:
        return None
    # top-face centre = midrange along the part's own axes: the band also catches the upper edge of the faces
    # that look at the camera, which would bias a centroid toward the camera; the extent is not biased by them
    uv = (pw[near, :2] - top[:2]) @ R[:2, :2]
    lo_, hi_ = np.percentile(uv, 1, axis=0), np.percentile(uv, 99, axis=0)
    k = C.CATALOG.get(sku)
    dims = np.array((k.dims[4], k.dims[1]) if k is not None and k.kind == "L" else (fx, fy)) / 1000
    if large:     # the wrist sees only part of a long top face: correct across it (the jaw axis) only
        lo_[0], hi_[0], dims[0] = -dims[0] / 2, dims[0] / 2, dims[0]
    if np.any(hi_ - lo_ < 0.75 * dims) or np.any(hi_ - lo_ > dims + 0.006):
        if verbose:
            print(f"    wrist: top face seen {np.round(1000 * (hi_ - lo_), 1)} mm vs {np.round(1000 * dims, 1)}: "
                  f"partly hidden or merged, no correction")
        return None
    mid = 0.5 * (lo_ + hi_)
    c = top[:2] + R[:2, :2] @ mid
    shift = np.array([c[0] - top[0], c[1] - top[1], dz])
    if verbose:
        print(f"    wrist: {int(ring.sum())} bed pts, {int(near.sum())} top pts, shift (mm) {np.round(1000 * shift, 1)}")
    return shift if np.linalg.norm(shift) < WRIST_MAX_SHIFT else None


def _pick_and_place(ex, ik, i, body, slot, rng, t0, est=None):
    w = ex.w
    ik.sync(w.d)
    from plan import door as _door
    oracle = ORACLE or est is None            # without a perception estimate (debug scripts) fall back to truth
    if _door.VERBOSE and est is not None:
        p, q = w.part_pose(body)
        xc, sgn = C.printer_frame(i)
        tx, ty = (p[0] - xc) * 1000 + C.BED_MM[0] / 2, (p[1] - C.AISLE_HALF - C.BED_FRONT_GAP) * 1000
        tyaw = math.degrees(2 * math.atan2(q[3], q[0]))
        print(f"    est {w.part_sku[body]}: ({est['center_mm'][0]:.1f},{est['center_mm'][1]:.1f}) yaw {est['yaw_deg']:.1f}"
              f" | truth ({tx:.1f},{ty:.1f}) yaw {tyaw:.1f} | upright z {1000 * (p[2] - C.BENCH_H - C.BED_ZL):.1f}mm "
              f"quat {np.round(q, 2)}")
    g = plan_grasp(ik, w, i, body, rng=rng, est=est)
    if g is None:
        return False, "no_grasp", 0.0, 0
    ignore = {w.part_body[body]}
    bed = w.m.body(f"bed{i}").id
    doors = {i: w.door_belief(i)}
    ex.grip(False, duration=0.3, label="preshape", width=g["width"])     # jaws just wider than the part
    # approach: staging point outside the door -> straight in to the pre-grasp (falls back to RRT)
    stage = outside_point(i, g["pre"])
    entry = cartesian_via(ik, [stage, np.array([g["pre"][0], g["pre"][1], stage[2]]), g["pre"]], g["R"], g["x"],
                          g["q_pre"], ignore, doors)
    q_stage = None
    if entry is not None:
        q_stage, _, ok, _ = ik.solve(stage, g["R"], g["x"], q0=entry[0], symmetric=False)
        q_stage = q_stage if ok else None
    if q_stage is not None and transfer(ex, ik, q_stage, g["x"], {"ignore": ignore}):
        ex.follow(entry, label="enter")
    elif not transfer(ex, ik, g["q_pre"], g["x"], {"ignore": ignore}):
        return False, "no_path_to_part", w.d.time - t0, 0
    shift = np.zeros(3)
    line = None
    if WRIST_ALIGN and not oracle:
        # wrist-camera look at the pre-grasp: where is the part relative to the gripper, really?
        ex.settle(tol=0.004, timeout=1.0)
        sft = wrist_offset(ex, ik, w, i, body, est, verbose=_door.VERBOSE)
        if sft is not None and np.linalg.norm(sft) > 0.0005:
            line = cartesian_via(ik, [g["pre"], g["pre"] + sft, g["target"] + sft], g["R"], g["x"], g["q_pre"],
                                 ignore | {bed}, doors, n=8)
            if line is not None:
                shift = sft
    if line is None:
        line = cartesian_via(ik, [g["pre"], g["target"]], g["R"], g["x"], g["q_pre"], ignore | {bed}, doors, n=8)
    if line is None:
        return False, "insert_blocked", w.d.time - t0, 0
    target, pre = g["target"] + shift, g["pre"] + shift
    ex.follow(line, label="insert")
    ex.settle(tol=0.004, timeout=1.5)
    for _ in range(2):          # fine alignment: re-aim by the pinch error the encoders show (servo sag/tracking)
        here = w.d.site_xpos[w.grasp_site] if oracle else ik.fk(ex.q_meas(), ex.x_meas())[0]
        err = target - here
        if np.linalg.norm(err) < 0.0015:
            break
        q_c, _, ok_c, _ = ik.solve(target + err, g["R"], g["x"], q0=ex.q, iters=60, symmetric=False)
        if not ok_c:
            break
        ex.move(q_c, duration=0.4, label="align")
        ex.settle(tol=0.003, timeout=1.0)
    track_err = float(np.degrees(np.max(np.abs(ex.q_meas() - ex.q))))
    site_err = 1000 * np.linalg.norm(w.d.site_xpos[w.grasp_site] - target)      # truth: reporting only
    ex.grip(True, duration=0.5, label="grip", width=g["width"])
    ex.payload = True
    ex.wait(0.15)
    z0 = w.part_pose(body)[0][2]
    if _door.VERBOSE:
        _snap(w, w.d.site_xpos[w.grasp_site], f"grip_{w.part_sku[body]}")
        hits = sorted({(w.m.body(w.m.geom_bodyid[c.geom1]).name, w.m.body(w.m.geom_bodyid[c.geom2]).name)
                       for c in w.d.contact[:w.d.ncon]
                       if (c.geom1 in w.arm_geoms) != (c.geom2 in w.arm_geoms)})
        print(f"    contacts at grip: {hits}")
        Rp = w.d.site_xmat[w.grasp_site].reshape(3, 3)
        rel = Rp.T @ (w.part_pose(body)[0] - w.d.site_xpos[w.grasp_site])
        print(f"    grip: tilt {g['tilt']:+d} h {1000 * g['h']:.0f}mm  track err {track_err:.2f} deg  pinch-to-target "
              f"{site_err:.1f}mm (truth)  part rel pinch (x,y,z mm) {np.round(1000 * rel, 1)}  gap "
              f"{ex.finger_gap_mm():.1f}mm (width {1000 * g['width']:.0f})")
    # lift = back out along the approach line (the verified insertion path, reversed), then up
    back = line[::-1] + [g["q_pre"]]
    ex.follow(back, label="lift")
    up = cartesian_via(ik, [g["pre"], g["pre"] + [0, 0, LIFT]], g["R"], g["x"], g["q_pre"], ignore, doors, n=4)
    if up:
        ex.follow(up, label="lift_up")
    ex.wait(0.1)
    gap = ex.finger_gap_mm()
    held = gap > 1000 * (g["width"] - 0.010) + 2.0          # jaws stopped on the part, not at their target
    if oracle:
        lifted = w.part_pose(body)[0][2] - z0 >= 0.02
    else:                                                  # wrist camera: is the part between the jaws?
        n_in = _wrist(ex).between_jaws(gap)          # gap-dominant; the wrist view decides marginal gap readings
        lifted = n_in >= N_BETWEEN or gap > 1000 * (g["width"] - 0.010) + 4.0
        if _door.VERBOSE:
            print(f"    lift check: gap {gap:.1f} mm (held {held}), wrist points between the jaws {n_in}")
    if not lifted or not held:
        ex.grip(False, duration=0.3)
        if oracle and w.d.eq_active[w.weld[body]]:
            return False, "stuck", w.d.time - t0, 1
        return False, ("slipped" if not held else "not_lifted"), w.d.time - t0, 1
    ik.sync(w.d)
    if oracle:
        ik.attach(body, w.d)        # from here on the held part is planned as part of the robot
        Rp = w.d.site_xmat[w.grasp_site].reshape(3, 3)
        rel = Rp.T @ (part_center(w, body) - w.d.site_xpos[w.grasp_site])
    else:                           # the robot believes the part sits in the jaws as planned (+ the wrist shift)
        pos, R = pose_from_estimate(i, est)
        pos = pos + shift
        Rg = g["R"]
        rel_p, rel_R = Rg.T @ (pos - target), Rg.T @ R
        ik.attach(body, w.d, rel=(rel_p, rel_R))
        _, h = top_center(w, body, pos, R)
        rel = rel_p + rel_R @ np.array([0.0, 0.0, h / 2])           # the part's centre in the pinch frame
    # the caddy rides on the carriage: if the bin is blocked here (e.g. under the open door), slide away first
    xc, s = C.printer_frame(i)
    q_drop = x_drop = pl = None
    large = slot in C.LARGE_SLOTS
    for sh in (0.0, 0.25, 0.45):
        x_try = ex.x + s * sh
        if large:                                        # lowered placement, not a drop
            pl = place_pose(ik, w, slot, x_try, width=g["width"])
            q_drop = None if pl is None else pl["q_up"]
        else:
            q_drop = drop_pose(ik, w, slot, x_try, rel, above=drop_height(w, body))
        if q_drop is not None:
            x_drop = x_try
            break
    if q_drop is None:
        ex.grip(False, duration=0.3)
        return False, "no_drop_pose", w.d.time - t0, 1
    # retreat: straight up and out through the door, then a planned move to the bin
    p_now = g["pre"] + [0, 0, LIFT if up else 0.0]
    back = max(C.CATALOG[w.part_sku[body]].footprint) / 2000 if is_large(w, body) else 0.0
    for hgt in EXIT_HEIGHTS if is_large(w, body) else EXIT_HEIGHTS[:1]:
        o = outside_point(i, p_now, height=hgt, back=back)
        out = cartesian_via(ik, [p_now, np.array([p_now[0], p_now[1], o[2]]), o], g["R"], g["x"], ex.q, ignore, doors)
        if out is not None:
            break
    if out is not None:
        ex.follow(out, label="exit")
    if large and out is not None and abs(x_drop - ex.x) > 1e-4:
        # a long part is carried along the aisle as it hangs after the exit: the folded slide pose (transfer) would
        # turn it over, and it creeps in the jaws
        ik.sync(w.d)
        hit = next((k for k in range(13) if ik.collides(ex.q, ex.x + (x_drop - ex.x) * k / 12, doors,
                                                         grip=g["width"] / 2)), None)
        if hit is None:
            ex.move(None, x_drop, label="slide")
        elif _door.VERBOSE:
            rob = w.arm_geoms | ik.held[0]
            print(f"    aisle slide blocked at {hit}/12: "
                  f"{sorted({(ik.m.body(ik.m.geom_bodyid[c.geom1]).name, ik.m.body(ik.m.geom_bodyid[c.geom2]).name, round(1000 * c.dist, 1)) for c in ik.d.contact[:ik.d.ncon] if (c.geom1 in rob) != (c.geom2 in rob)})[:4]}")
    if not transfer(ex, ik, q_drop, x_drop, {"ignore": ignore, "grip": g["width"] / 2 if large else 0.012}):
        ex.grip(False, duration=0.3)
        return False, "no_path_to_bin", w.d.time - t0, 1
    if large:
        believed, outcome = _place_in_slot(ex, ik, body, slot, g, pl, oracle)
        return believed, outcome, w.d.time - t0, 1
    if _door.VERBOSE:
        sid = mujoco.mj_name2id(w.m, mujoco.mjtObj.mjOBJ_SITE, f"bin_{slot}")
        print(f"    pre-release: part {np.round(part_center(w, body), 3)} vs bin rim {np.round(w.d.site_xpos[sid], 3)}"
              f", pinch {np.round(w.d.site_xpos[w.grasp_site], 3)}")
        _snap(w, w.d.site_xpos[w.grasp_site], f"release_{w.part_sku[body]}")
    if not oracle and ex.finger_gap_mm() <= 1000 * (g["width"] - 0.010) + 2.0:
        ex.grip(False, duration=0.3)                     # the jaws closed on nothing: lost on the way
        return False, "dropped_in_transit", w.d.time - t0, 1
    z_rel = part_center(w, body)[2]
    ik.detach()
    ex.grip(False, duration=0.4, label="release")
    ex.wait(0.3)

    def still_held():                                    # release check
        if oracle:
            return part_center(w, body)[2] > z_rel - 0.02
        n = _wrist(ex).between_jaws(ex.finger_gap_mm())   # wrist camera: anything left between the jaws?
        if _door.VERBOSE:
            print(f"    release check: wrist points between the jaws {n}")
        return n >= N_BETWEEN

    shakes = 0
    while shakes < SHAKES and still_held():
        q_rel = ex.q                    # a part caught on a finger is shaken off
        for roll in (0.9, -0.9, 0.0):   # wrist roll swings the fingertips on a ~37 mm circle above the bin
            ex.move(np.r_[q_rel[:5], q_rel[5] + roll], duration=0.2, label="shake")
        shakes += 1
        ex.wait(0.2)
    ex.wait(0.3)
    if oracle:
        believed = part_in_bin(w, body, slot)
        outcome = "placed" if believed else "missed_bin"
    else:
        believed = not (shakes == SHAKES and still_held())
        outcome = "placed" if believed else "stuck_on_finger"
    if _door.VERBOSE:
        sid = mujoco.mj_name2id(w.m, mujoco.mjtObj.mjOBJ_SITE, f"bin_{slot}")
        print(f"    drop: rel(pinch frame) {np.round(1000 * rel, 1)} mm; part now {np.round(w.part_pose(body)[0], 3)}; "
              f"bin rim {np.round(w.d.site_xpos[sid], 3)}; carriage {ex.x:.3f} (drop x {x_drop:.3f}); "
              f"robot: {outcome}, truth in bin: {part_in_bin(w, body, slot)}")
    return believed, outcome, w.d.time - t0, 1


def _place_in_slot(ex, ik, body, slot, g, pl, oracle):
    """Lower the held long part into its tray slot (pl from place_pose, the arm at pl["q_up"]), open the jaws a few mm,
    check that they are empty, back out straight up and look into the slot. -> (believed, outcome)."""
    w = ex.w
    from plan import door as _door
    width = g["width"]
    if not oracle and ex.finger_gap_mm() <= 1000 * (width - 0.010) + 2.0:
        ex.grip(False, duration=0.3)                     # the jaws closed on nothing: lost on the way
        return False, "dropped_in_transit"
    ex.follow(pl["down"], label="place")
    ex.settle(tol=0.004, timeout=1.0)
    ik.detach()
    g_rel = min(0.037, width / 2 + RELEASE_OPEN)
    ex.jaws(g_rel, duration=0.4, label="release")
    ex.payload = False
    ex.wait(0.3)

    def still_held():                                    # release check, from above: anything left between the jaws?
        if oracle:
            return w.part_pose(body)[0][2] > pl["target"][2] + 0.02
        return _wrist(ex).between_jaws(ex.finger_gap_mm()) >= N_BETWEEN

    p, ign = pl["p"], {w.part_body[body]}
    up = cartesian_via(ik, [p, p + [0, 0, PLACE_LIFT]], pl["R"], pl["x"], ex.q, ign, None, n=4, grip=g_rel)
    ex.follow(up if up is not None else pl["down"][::-1], label="place_up")
    ex.settle(tol=0.004, timeout=1.0)
    held = still_held()
    if held:                                             # caught on a finger: open fully over the slot, look again
        ex.jaws(0.037, duration=0.3, label="release")
        ex.wait(0.3)
        held = still_held()
    h = C.CATALOG[w.part_sku[body]].height / 1000
    if oracle:
        believed = part_in_bin(w, body, slot)
        outcome = "placed" if believed else "missed_slot"
    else:
        n = slot_seen(ex, ik, slot, h)
        believed = not held and n >= N_SLOT
        outcome = "placed" if believed else "stuck_on_finger" if held else "not_in_slot"
    if _door.VERBOSE:
        print(f"    place {slot}: robot {outcome} (held {held}{'' if oracle else f', slot points {n}'}); "
              f"truth in slot: {part_in_bin(w, body, slot)}, part {np.round(w.part_pose(body)[0], 3)}")
    return believed, outcome


def outside_point(i, p, height=0.17, back=0.0):
    """A staging point straight out of printer i's door from world point p: just outside the door plane, high.
    back: this much further into the aisle (a held long part reaches back toward the printer)."""
    xc, s = C.printer_frame(i)
    q = np.array(p, float).copy()
    q[1] = s * (C.AISLE_HALF - 0.07 - back)
    q[2] = C.BENCH_H + C.BED_ZL + height
    return q


def cartesian_via(ik, pts, R, x, q_seed, ignore=(), doors=None, n=8, grip=0.012):
    """IK along a polyline at fixed orientation; every sample must be collision-free (fine clearance)."""
    qs, q = [], np.asarray(q_seed, float)
    for a, b in zip(pts[:-1], pts[1:]):
        for s in np.linspace(0, 1, n + 1)[1:]:
            q, _, ok, _ = ik.solve(a + s * (b - a), R, x, q0=q, iters=80, symmetric=False)
            if not ok or ik.collides(q, x, doors, grip=grip, ignore_bodies=ignore, clearance=GRASP_CLEAR):
                return None
            qs.append(q.copy())
    return qs


def drop_height(w, body):
    """Release height of the part centre above the rim: a tilted, wide part must clear the rim entirely."""
    sku = w.part_sku[body]
    if sku == "purge_tab":
        return DROP_ABOVE
    k = C.CATALOG[sku]
    return 0.015 + 0.5 * max(max(k.footprint), k.height) / 1000


_SNAPCAM = {}


def _snap(w, at, name):
    from farm.render import Cam, save
    if id(w) not in _SNAPCAM:
        _SNAPCAM[id(w)] = Cam(w.m, 640, 480)
    cam = _SNAPCAM[id(w)]
    imgs = [cam.look(at, 0.28, az, el).rgb(w.d) for az, el in ((90 + 70, -15), (90 - 70, -15), (90, -80))]
    save(np.concatenate(imgs, axis=1), f"runs/snap_{name}.png")


def on_bed(w, i, body):
    """Is the part lying on printer i's bed (within the bed area, near bed height)?"""
    xc, s = C.printer_frame(i)
    p = w.part_pose(body)[0]
    xl, yl = s * (p[0] - xc), s * p[1] - C.AISLE_HALF - C.BED_FRONT_GAP
    return (abs(xl) < C.BED_MM[0] / 2000 and 0 < yl < C.BED_MM[1] / 1000
            and abs(p[2] - (C.BENCH_H + C.BED_ZL)) < 0.03)


def part_center(w, body):
    """Approximate centre of mass of a pooled part (world)."""
    return w.d.xipos[w.part_body[body]].copy()


def mirror_plate(plate):
    """Gather parts toward the robot side: mirror the packer's bottom-left layout to bottom-right (yaw +180 deg
    keeps asymmetric parts consistent). The slicer puts the purge tab at front-centre, so it stays there."""
    W = plate["bed_mm"][0]
    pt = dict(plate["purge_tab"])
    pt["center_mm"] = [W - pt["center_mm"][0], pt["center_mm"][1]]
    out = dict(plate, objects=[], purge_tab=pt)
    for o in plate["objects"]:
        o = dict(o, center_mm=[W - o["center_mm"][0], o["center_mm"][1]], yaw_deg=(o["yaw_deg"] + 180.0) % 360,
                 polygon_mm=[[W - x, y] for x, y in o["polygon_mm"]])
        out["objects"].append(o)
    return out


def slot_size(slot):
    """(x, y, wall height) of a caddy bin or a large-tray slot (m, outer)."""
    return C.LARGE_SLOT_SIZE if slot in C.LARGE_SLOTS else C.BIN_SIZE


def all_slots():
    """Every caddy bin and large-tray slot."""
    return (*C.CADDY_SLOTS, *C.LARGE_SLOTS)


def part_in_bin(w, body, slot):
    """SIM truth (reporting): is the part inside caddy bin / tray slot `slot`? In a large-tray slot it must also lie
    flat with its long axis along the slot (what the station's place into the box expects)."""
    sid = mujoco.mj_name2id(w.m, mujoco.mjtObj.mjOBJ_SITE, f"bin_{slot}")
    c = w.d.site_xpos[sid]
    p, q = w.part_pose(body)
    sx, sy, h = slot_size(slot)
    inside = abs(p[0] - c[0]) < sx / 2 and abs(p[1] - c[1]) < sy / 2 and c[2] - h - 0.01 < p[2] < c[2] + 0.02
    if not inside or slot not in C.LARGE_SLOTS:
        return inside
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, q)
    return R[8] > 0.95 and abs(R[0]) > math.cos(math.radians(20))


part_in_slot = part_in_bin


def pick_priority(w, name, body):
    """Purge tab first, then the tippiest parts (tall relative to their footprint) before anything can knock
    them over, then by name for determinism. Large parts last: lying flat they do not tip, and a long part is
    carried out more easily over an emptier bed (the next feasible part is picked if one is blocked)."""
    if name.endswith("purge_tab"):
        return (0, 0.0, name)
    k = C.CATALOG[w.part_sku[body]]
    if k.large:
        return (2, -k.height / min(k.footprint), name)
    return (1, -k.height / min(k.footprint), name)


_CAMS = {}


def bed_camera(w, i, rng=None):
    """Chamber depth camera of printer i (cached per world), with the CAMERA_NOISE preset when perception has it."""
    from perceive.verify import BedCamera
    if (id(w), i) not in _CAMS:
        try:
            _CAMS[(id(w), i)] = BedCamera(w, i, noise=CAMERA_NOISE, corrections=CAMERA_CORRECTIONS,
                                          rng=rng if rng is not None else np.random.default_rng((CAMERA_SEED, i)))
        except TypeError:                                  # perception without noise presets
            _CAMS[(id(w), i)] = BedCamera(w, i)
    return _CAMS[(id(w), i)]


def scan(cam):
    """One chamber scan; a failed scan (quality 'failed': no frame, blackout) is retried up to RESCANS times."""
    pts = cam.points_bed_mm()
    for _ in range(RESCANS):
        meta = getattr(pts, "meta", None)
        if not meta or meta.get("quality") != "failed":
            break
        pts = cam.points_bed_mm()
    return pts


EXPLAIN_MM = 6.0         # raised points within a found part's footprint + this margin belong to that part


def _yaw_quat(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def scene_belief(w, i, est, remaining, points, objs=None):
    """The planning scene the robot believes in, from one chamber scan: every part the camera found at its
    estimated pose (inflated by PAD_K x its sigma), and every raised blob not explained by a found part as a box
    obstacle (debris, knocked-over parts). Parts the robot has not seen are not in it."""
    from plan.ik import Belief
    from plan.sweep import blobs
    from perceive.verify import RAISED_MM
    _, s = C.printer_frame(i)
    parts = {}
    raised = points[points[:, 2] > RAISED_MM]
    keep = np.ones(len(raised), bool)
    for n, body in remaining.items():
        e = est.get(n)
        if e is None or not e.get("found"):
            continue
        sig = float(e.get("sigma_mm", SIGMA_DEFAULT_MM))
        pos, R = pose_from_estimate(i, e)
        parts[body] = (pos, _yaw_quat(math.atan2(R[1, 0], R[0, 0])), PAD_K * sig / 1000)
        sku = w.part_sku[body]
        fx, fy = C.PURGE_TAB_MM[:2] if sku == "purge_tab" else C.CATALOG[sku].footprint
        yr = math.radians(e["yaw_deg"])
        dx, dy = raised[:, 0] - e["center_mm"][0], raised[:, 1] - e["center_mm"][1]
        u, v = dx * math.cos(yr) + dy * math.sin(yr), -dx * math.sin(yr) + dy * math.cos(yr)
        keep &= ~((np.abs(u) < fx / 2 + EXPLAIN_MM) & (np.abs(v) < fy / 2 + EXPLAIN_MM))
    bl, raw = [], []
    for b in blobs(raised[keep]):
        (x0, y0), (x1, y1) = b["lo"], b["hi"]
        c = np.array(C.bed_mm_to_world(i, (x0 + x1) / 2, (y0 + y1) / 2))
        half = np.array([(x1 - x0) / 2 + 1.0, (y1 - y0) / 2 + 1.0, max(b["h"], 2.0) / 2]) / 1000
        bl.append((c, half, 0.0 if s > 0 else math.pi, PAD_K * SIGMA_DEFAULT_MM / 1000))
        raw.append(b)
    bel = Belief(parts, bl)
    bel.raw = raw                        # the blob records (bed mm), front-most first, for the rake / blob pick
    return bel


def prior_belief(w, i, plate, bodies, pad_mm=3.0):
    """Planning scene before the first chamber scan: the slicer's plate (every object at its sliced pose)."""
    from plan.ik import Belief
    from perceive.verify import plate_objects
    parts = {}
    for o in plate_objects(plate):
        if o["name"] in bodies:
            pos, R = pose_from_estimate(i, o)
            parts[bodies[o["name"]]] = (pos, _yaw_quat(math.atan2(R[1, 0], R[0, 0])), pad_mm / 1000)
    return Belief(parts)


RAKE = True              # fallback: rake what the robot could not pick into the printer's reject tray (plan/sweep.py)
RAKE_ROUNDS = 4
CLEAR_FIRST = 3          # unknown blobs that box in every remaining part are cleared mid-unload (at most this often)
RETRY = ("slipped", "not_lifted", "stuck")


def unload_parts(ex, ik, i, plate, bodies, slot_for, on_result, rng=None):
    """Perception-in-the-loop unload of printer i's bed. Before every pick the chamber camera re-scans the bed, the
    slicer prior is refined and the planning scene is rebuilt from what the camera sees (scene_belief); grasps are
    planned from the estimate. Parts the robot cannot pick are raked off the bed into the reject tray at the end
    (RAKE), so the printer is free again.
    on_result(name, body, slot, outcome, sec) is called once per part with what the ROBOT believes: 'placed'
    (released over its bin, jaws empty after the release); 'swept' (raked, and the final scan shows a clear bed);
    or the failure ('no_grasp', 'insert_blocked', 'slipped', 'stuck_on_finger', 'missing', ...). Ground truth
    (part_in_bin / in_tray / on_bed) is for the caller's reporting only.
    Returns (bed_empty_after, unknown_points_first_scan)."""
    from perceive.verify import bed_empty, cannot_verify, estimate, plate_objects
    w = ex.w
    cam = bed_camera(w, i)
    objs = {o["name"]: o for o in plate_objects(plate)}
    remaining = dict(bodies)
    attempts, first_fail, leftovers = {}, {}, {}     # leftovers: name -> (body, why) not placed after picking
    unknown0 = None
    clears, t_clear = 0, 0.0
    while remaining:
        mujoco.mj_forward(w.m, w.d)
        pts = scan(cam)
        est, unknown = estimate(pts, [objs[n] for n in remaining])
        unknown0 = unknown if unknown0 is None else unknown0
        for n in [n for n in remaining if not est[n]["found"]]:
            leftovers[n] = (remaining.pop(n), first_fail.get(n, "missing"))   # moved / knocked over: rake pass
        names = sorted(remaining, key=lambda n: pick_priority(w, n, remaining[n]))
        for n in names:                                  # the printer's own start code places the tab exactly
            if n.endswith("purge_tab"):
                est[n] = dict(est[n], center_mm=list(objs[n]["center_mm"]), yaw_deg=objs[n]["yaw_deg"])
        if not ORACLE:
            ik.belief = scene_belief(w, i, est, remaining, pts)
        if not remaining:
            break
        ik.sync(w.d)
        name = next((n for n in names if plan_grasp(ik, w, i, remaining[n], rng=rng, est=est[n]) is not None), None)
        if name is None and RAKE and not ORACLE and ik.belief.blobs and clears < CLEAR_FIRST:
            # every remaining part is boxed in by something the camera cannot explain (debris, a knocked-over part):
            # clear the front-most such blob first, then re-scan and pick again
            from plan.sweep import rake as _rake
            clears += 1
            for b in ik.belief.raw:
                ok_r, sec_r = _rake(ex, ik, i, b, rng)
                t_clear += sec_r
                if ok_r:
                    break
            else:
                clears = CLEAR_FIRST
            continue
        if name is None:
            for n in names:
                leftovers[n] = (remaining.pop(n), first_fail.get(n, "no_grasp"))
            break
        body = remaining.pop(name)
        slot = slot_for(name)
        ok, outcome, sec, _ = pick_and_place(ex, ik, i, body, slot, rng, est=est[name])
        if ok:
            on_result(name, body, slot, outcome, sec)
            continue
        first_fail.setdefault(name, outcome)
        if ORACLE:
            retry = outcome in RETRY and on_bed(w, i, body)
            if not retry and not on_bed(w, i, body):
                if body in w.in_use and not part_in_bin(w, body, slot):
                    w.park(body)
                on_result(name, body, slot, outcome, sec)
                continue
        else:                                            # the next scan decides: found -> retry, else rake pass
            retry = outcome in RETRY or outcome in ("no_drop_pose", "no_path_to_bin")
        if retry and attempts.get(name, 0) < 1:
            attempts[name] = attempts.get(name, 0) + 1
            remaining[name] = body
            continue
        leftovers[name] = (body, outcome)
    _housekeeping(w, i, leftovers)
    t_rake = 0.0
    if RAKE and leftovers or RAKE and not bed_empty(scan(cam)):
        from plan.sweep import blobs, rake
        for _ in range(RAKE_ROUNDS):                     # front-most blob first; re-scan after every rake
            mujoco.mj_forward(w.m, w.d)
            todo = blobs(scan(cam))
            if not ORACLE:
                ik.belief = scene_belief(w, i, {}, {}, scan(cam))
            done = False
            for b in todo:
                ok, sec = rake(ex, ik, i, b, rng)
                t_rake += sec
                if ok:
                    done = True
                    break
            if not done:
                break
        _housekeeping(w, i, leftovers)
    mujoco.mj_forward(w.m, w.d)
    pts = scan(cam)
    empty = bed_empty(pts)
    for _ in range(VERIFY_RESCANS):              # "cannot verify" (occluded / failed scan, no leftover seen): look again
        if empty or not cannot_verify(pts):
            break
        pts = scan(cam)
        empty = bed_empty(pts)
    if not ORACLE:                  # what is left on this bed stays in the planning scene (door close, next moves)
        ik.belief = scene_belief(w, i, {}, {}, pts)
    from plan.sweep import in_tray
    for n, (body, why) in leftovers.items():
        if ORACLE:
            out = "swept" if in_tray(w, i, body) else why
        else:                                            # robot: raked and the bed looks clear -> it is gone
            out = "swept" if (RAKE and t_rake > 0 and empty) else why
        on_result(n, body, None, out, t_rake / max(len(leftovers), 1))
    return empty, (unknown0 or 0)


def _housekeeping(w, i, leftovers):
    """SIMULATION housekeeping, not a robot decision: a failed part that fell off the bed somewhere other than a
    bin or the reject tray (floor, deck) is taken out of the world, as an operator would collect it later."""
    from plan.sweep import in_tray
    for n, (body, why) in leftovers.items():
        if body in w.in_use and not on_bed(w, i, body) and not in_tray(w, i, body) and \
                not any(part_in_bin(w, body, s) for s in all_slots()):
            w.park(body)
