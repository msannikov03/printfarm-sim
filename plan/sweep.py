"""Fallback when a part cannot be picked (no grasp, slipped, knocked over) or the camera sees debris: rake it off the
bed's front edge into the printer's reject tray (farm/scene.py), so the printer can start its next plate. The part is
reprinted; the operator empties the tray once per shift.

No grasp is needed: jaws opened to the blob's width, fingertips a few mm above the bed behind the blob, a straight
pull toward the door. Blobs come from the chamber camera (raised points), not from the slicer, so knocked-over parts
and unknown debris are handled the same way.
Long blobs (large SKUs, 130-150 mm): never picked into the caddy's 110 mm reject bin (they would land across it);
lying along the door they are raked into the 300 mm wide tray like any part, lying into the printer they are pushed
off the nearer side edge first (lengthwise beside the bed: the tray is only 40 mm deep).
"""
import math

import mujoco
import numpy as np
from scipy import ndimage

from farm import catalog as C
from plan.door import RAIL_MIN_L, transfer
from plan.ik import frame
from plan.unload import DXS, LIFT as PICK_LIFT, TIP_CLEAR, cartesian_via, drop_pose, finger_min_z, grasp_search, \
    outside_point

GRID_MM = 2.0
MIN_BLOB_PTS = 30
MAX_BLOB_H_MM = 60.0                 # taller "blobs" are the robot or the door leaf in view, not a print (parts <= 50 mm
                                     # even tipped); a real cell masks the robot by its known pose
TILTS = (0, 20, -20, 35, -35, 50, -50, 65, -65)     # tool lean about the jaw axis (deg)
BACK_GAPS = (0.016, 0.026, 0.04, 0.055)   # pinch site behind the blob's back edge (m); larger gaps let a leaning
                                         # finger clear a tall blob's top edge (it then pushes on that edge)
EDGE_PAST = 0.02                     # pull until the pinch is this far past the bed's front edge (m)
TIP_Z = 0.003                        # fingertips above the bed while raking (m)
LIFT = 0.06
REJECT = None                        # set to a collections.Counter() to tally rejections (debugging)


def _rej(why):
    if REJECT is not None:
        REJECT[why] += 1


def blobs(points):
    """Raised points (bed mm) -> blobs {"lo": (x, y), "hi": (x, y), "h": mm, "n": points}, front-most first."""
    from perceive.verify import raised_points
    p = raised_points(points)                  # noise-aware: no false blobs from depth noise
    if len(p) < MIN_BLOB_PTS:
        return []
    ij = np.floor(p[:, :2] / GRID_MM).astype(int)
    ij -= ij.min(0)
    grid = np.zeros(ij.max(0) + 1, bool)
    grid[ij[:, 0], ij[:, 1]] = True
    lab, n = ndimage.label(ndimage.binary_dilation(grid, iterations=2), structure=np.ones((3, 3)))
    k = lab[ij[:, 0], ij[:, 1]]
    out = []
    for b in range(1, n + 1):
        q = p[k == b]
        if len(q) >= MIN_BLOB_PTS and q[:, 2].max() <= MAX_BLOB_H_MM:
            import cv2
            top = q[q[:, 2] > q[:, 2].max() - 3.0]           # top face: the outline without the visible side faces
            rect = cv2.minAreaRect((top if len(top) >= 10 else q)[:, :2].astype(np.float32))
            out.append({"lo": q[:, :2].min(0), "hi": q[:, :2].max(0), "h": float(q[:, 2].max()), "n": int(len(q)),
                        "rect": (tuple(map(float, rect[0])), tuple(map(float, rect[1])), float(rect[2]))})
    return sorted(out, key=lambda b: b["lo"][1])


def blob_bodies(w, i, blob, margin_mm=10.0, ik=None):
    """Bodies the rake is meant to touch (planning ignores them). With a belief scene: the blob's proxy obstacles
    and believed parts inside the blob; without one (oracle): pooled bodies whose true centre lies in the blob."""
    xc, s = C.printer_frame(i)

    def inside(p):
        bx = (s * (p[0] - xc) + C.BED_MM[0] / 2000) * 1000
        by = (s * p[1] - C.AISLE_HALF - C.BED_FRONT_GAP) * 1000
        return (blob["lo"][0] - margin_mm <= bx <= blob["hi"][0] + margin_mm
                and blob["lo"][1] - margin_mm <= by <= blob["hi"][1] + margin_mm)

    if ik is not None and ik.belief is not None:
        out = {bid for b, bid in ik.proxies if inside(b[0])}
        out |= {w.part_body[n] for n, (p, _, _) in ik.belief.parts.items() if inside(p)}
        return out
    return {w.part_body[n] for n in w.in_use if inside(w.part_pose(n)[0])}


def rake_frames(into):
    """(mode, tilt, R): 'behind' = jaws side by side across the pull (both fingertips push the blob's back face),
    'cage' = jaws along the pull straddling the blob (the rear finger pushes). Tilt leans the tool about the jaw axis."""
    z = np.array([0, 0, 1.0])
    out = []
    for mode, jaw in (("behind", np.cross(into, z)), ("cage", into)):
        u = np.cross(z, jaw)
        for t in TILTS:
            a = math.radians(t)
            out.append((mode, t, frame(math.sin(a) * u - math.cos(a) * z, jaw)))
    return sorted(out, key=lambda f: abs(f[1]))


LEANS = (35, 50, 65, 20)


def side_frames(push, door_in):
    """Side push deep in the bed: jaws closed (one blunt pusher), the tool leaning with its tip into the printer
    (like a deep grasp), so the arm reaches far in from the door while the fingertips skim the bed."""
    z = np.array([0, 0, 1.0])
    return [("behind", t, frame(math.sin(math.radians(t)) * door_in - math.cos(math.radians(t)) * z, push))
            for t in LEANS]


DIRECTIONS = {"front": (0.0, -1.0), "left": (-1.0, 0.0), "right": (1.0, 0.0)}   # push direction in the bed frame


LONG_Y_MM = 60.0         # a blob longer than this into the printer would go end-first over the front edge: it does not
                         # land in the 40 mm deep reject tray but leans out of it toward the door


def directions_for(blob):
    """Front pull into the reject tray first; for blobs it cannot reach behind (deep in the bed), push sideways off
    the nearer side edge, then the far one. A long blob lying into the printer (a large part) goes sideways first:
    it then lies lengthwise beside the bed instead of standing in the front gap."""
    cx = (blob["lo"][0] + blob["hi"][0]) / 2
    side = ("left", "right") if cx < C.BED_MM[0] / 2 else ("right", "left")
    if blob["hi"][1] - blob["lo"][1] > LONG_Y_MM:
        return side + ("front",)
    return ("front",) + side


def fits_reject_bin(blob):
    """Would a blob dropped into the caddy's reject bin land inside it (footprint well under the bin opening)?"""
    w_, h_ = blob["rect"][1] if "rect" in blob else np.subtract(blob["hi"], blob["lo"])
    return max(w_, h_) <= 1000 * C.BIN_SIZE[0] - 20.0


def plan_rake(ik, w, i, blob, rng=None, direction="front"):
    """Search rake style x tilt x back gap x carriage position for a collision-free push of `blob` off the bed
    edge in `direction` ('front' = toward the door, into the reject tray; 'left'/'right' = off the side edge).
    Returns a plan dict or None."""
    xc, s = C.printer_frame(i)
    rng = rng or np.random.default_rng(0)
    bed_z = C.BENCH_H + C.BED_ZL
    db = np.array(DIRECTIONS[direction])                  # push direction, bed frame
    dw = s * np.array([db[0], db[1], 0.0])                # ... world frame
    into = -dw                                            # from the target edge toward the blob
    lo, hi = np.asarray(blob["lo"], float), np.asarray(blob["hi"], float)
    ctr, ext = (lo + hi) / 2, (hi - lo) / 2
    along = float(np.abs(db) @ ext)                       # blob half extent along the push (mm)
    edge = {"front": ctr[1], "left": ctr[0], "right": C.BED_MM[0] - ctr[0]}[direction]   # centre -> edge (mm)
    h = blob["h"] / 1000
    bed = w.m.body(f"bed{i}").id
    touch = blob_bodies(w, i, blob, ik=ik) | {bed}
    doors = {i: w.door_belief(i)}
    seeds = [np.array([1.57 * s, 1.0, 1.4, -0.9, 0, 0]), np.array([1.57 * s, 0.6, 1.9, -1.3, 0, 0])]
    frames = rake_frames(into) if direction == "front" else \
        side_frames(dw, np.array([0.0, s, 0.0])) + [f for f in rake_frames(into) if f[0] == "behind" and abs(f[1]) <= 35]
    for mode, tilt, R in frames:
        if mode == "behind":
            g_open = 0.0                    # jaws closed: one blunt pusher (open jaws let round parts wedge between)
            starts = [((along + 1000 * back), back) for back in BACK_GAPS]
        else:
            g_open = along / 1000 + 0.008      # jaws (along the push) straddle the blob; the rear finger pushes
            if g_open > 0.037:
                continue
            starts = [(0.0, 0.0)]
        for off_mm, back in starts:
            for dx in DXS:
                x = xc + s * dx
                if s * (x - xc) < RAIL_MIN_L:
                    continue
                sb = ctr - db * off_mm                     # start (bed mm): behind the blob's back face
                if not (-5 <= sb[0] <= C.BED_MM[0] + 5 and -5 <= sb[1] <= C.BED_MM[1] + 5):
                    continue
                start = np.array(C.bed_mm_to_world(i, *sb))
                start[2] = bed_z + 0.012
                q = None
                for q0 in seeds + [rng.uniform(ik.lo, ik.hi)]:
                    for _ in range(2):                         # second pass: correct the height by the fingertip z
                        qs, _, ok, _ = ik.solve(start, R, x, q0=q0, iters=150)
                        if not ok:
                            _rej("ik")
                            break
                        ik.collides(qs, x, doors, grip=g_open, ignore_bodies={bed})
                        mujoco.mj_forward(ik.m, ik.d)
                        start[2] += bed_z + TIP_Z - finger_min_z(ik)
                        q0 = qs
                    if not ok:
                        continue
                    if ik.collides(qs, x, doors, grip=g_open, ignore_bodies={bed}, clearance=0.002):
                        _rej(f"start_collides:{mode}")
                        continue
                    mujoco.mj_forward(ik.m, ik.d)
                    fz = finger_min_z(ik) - bed_z
                    if TIP_CLEAR <= fz <= max(TIP_Z + 0.003, min(0.5 * h, 0.008)):
                        q = qs
                        break
                    _rej(f"tip_z {1000 * fz:.0f}mm")
                if q is None:
                    continue
                _, Rs = ik.fk(q, x)                     # the solved frame (the jaw may be flipped: symmetric)
                pre = start + [0, 0, h + 0.03]
                q_pre, _, okp, _ = ik.solve(pre, Rs, x, q0=q, symmetric=False)
                if not okp:
                    _rej("pre_ik")
                    continue
                down = cartesian_via(ik, [pre, start], Rs, x, q_pre, {bed}, doors, n=6, grip=g_open)
                length = (edge + off_mm) / 1000 + EDGE_PAST + along / 1000 * (mode == "cage")
                end = start + length * dw
                pull = down and cartesian_via(ik, [start, end], Rs, x, down[-1], touch, doors,
                                              n=max(8, int(length / 0.01)), grip=g_open)
                lift = pull and cartesian_via(ik, [end, end + [0, 0, LIFT]], Rs, x, pull[-1], touch, doors, n=4,
                                              grip=g_open)
                _rej("down" if not down else "pull" if not pull else "lift" if not lift else "ok")
                if lift:
                    open_ok = not ik.collides(pull[-1], x, doors, grip=0.037, ignore_bodies=touch, clearance=0.002)
                    return {"x": x, "open_ok": open_ok, "R": Rs, "pre": pre, "q_pre": q_pre, "down": down, "pull": pull, "lift": lift,
                            "end": end + [0, 0, LIFT], "g_open": g_open, "tilt": tilt, "mode": mode, "touch": touch,
                            "direction": direction}
    return None


SIDE_PUSH = True         # last resort: push a blob off the nearer side edge (it lands in the chamber beside the bed)
BLOB_PICK = True         # deep blobs the front pull cannot get behind are picked into the caddy's reject bin


def plan_blob_grasp(ik, w, i, blob, rng=None):
    """Grasp an unknown blob as a box: the camera outline (min-area rectangle of its top face) gives the centre,
    the jaw axes and widths; the blob height gives the grasp heights. No model of the part is needed."""
    _, s = C.printer_frame(i)
    (cx, cy), (wu, wv), ang = blob["rect"]
    a = math.radians(ang)
    u = s * np.array([math.cos(a), math.sin(a), 0.0])
    v = s * np.array([-math.sin(a), math.cos(a), 0.0])
    p0 = np.array(C.bed_mm_to_world(i, cx, cy))
    gmax = C.GRIPPER_MAX_MM / 1000
    opts = sorted([(u, wu / 1000 + 0.002), (v, wv / 1000 + 0.002)], key=lambda o: o[1])
    opts = [o for o in opts if o[1] <= gmax]
    h = blob["h"] / 1000
    if not opts or h < 0.006:
        return None
    return grasp_search(ik, w, i, p0, opts, [0.6 * h, max(h - 0.004, 0.005)], h, blob_bodies(w, i, blob, ik=ik), rng)


def pick_blob(ex, ik, i, blob, g, rng=None):
    """Execute a blob grasp and drop it into the reject bin. Returns True if the jaws held something to the bin."""
    w = ex.w
    touch = blob_bodies(w, i, blob, ik=ik)
    bed = w.m.body(f"bed{i}").id
    doors = {i: w.door_belief(i)}
    ex.grip(False, duration=0.3, label="preshape", width=g["width"])
    stage = outside_point(i, g["pre"])
    entry = cartesian_via(ik, [stage, np.array([g["pre"][0], g["pre"][1], stage[2]]), g["pre"]], g["R"], g["x"],
                          g["q_pre"], touch, doors)
    q_stage = None
    if entry is not None:
        q_stage, _, ok, _ = ik.solve(stage, g["R"], g["x"], q0=entry[0], symmetric=False)
        q_stage = q_stage if ok else None
    if q_stage is not None and transfer(ex, ik, q_stage, g["x"], {"ignore": touch}):
        ex.follow(entry, label="enter")
    elif not transfer(ex, ik, g["q_pre"], g["x"], {"ignore": touch}):
        return False
    line = cartesian_via(ik, [g["pre"], g["target"]], g["R"], g["x"], g["q_pre"], touch | {bed}, doors, n=8)
    if line is None:
        return False
    ex.follow(line, label="insert")
    ex.settle(tol=0.004, timeout=1.5)
    ex.grip(True, duration=0.5, label="grip", width=g["width"])
    ex.payload = True
    ex.wait(0.15)
    ex.follow(line[::-1] + [g["q_pre"]], label="lift")
    up = cartesian_via(ik, [g["pre"], g["pre"] + [0, 0, PICK_LIFT]], g["R"], g["x"], g["q_pre"], touch, doors, n=4)
    if up:
        ex.follow(up, label="lift_up")
    held = ex.finger_gap_mm() > 1000 * (g["width"] - 0.010) + 2.0
    if not held:
        ex.grip(False, duration=0.3)
        ex.payload = False
        return False
    p_now = g["pre"] + [0, 0, PICK_LIFT if up else 0.0]
    out = cartesian_via(ik, [p_now, outside_point(i, p_now)], g["R"], g["x"], ex.q, touch, doors)
    if out is not None:
        ex.follow(out, label="exit")
    _, s = C.printer_frame(i)
    for sh in (0.0, 0.25, 0.45):
        q_drop = drop_pose(ik, w, "reject", ex.x + s * sh, None, above=0.06)
        if q_drop is not None and transfer(ex, ik, q_drop, ex.x + s * sh, {"ignore": touch, "grip": 0.012}):
            break
    ex.grip(False, duration=0.4, label="release")
    ex.payload = False
    ex.wait(0.3)
    return True


def rake(ex, ik, i, blob, rng=None):
    """Clear one blob off the bed: rake it over the front edge into the reject tray; if the arm cannot get behind it
    (deep in the bed), pick it into the reject bin (BLOB_PICK); last, push it off a side edge (SIDE_PUSH).
    Returns (executed, seconds)."""
    w = ex.w
    t0 = w.d.time
    ik.sync(w.d)
    from plan import door as _door
    if BLOB_PICK and "rect" in blob and fits_reject_bin(blob):   # graspable: picking is more reliable than pushing
        g = plan_blob_grasp(ik, w, i, blob, rng)
        if g is not None:
            if _door.VERBOSE:
                print(f"    blob pick into the reject bin: blob {np.round(blob['lo'])}-{np.round(blob['hi'])} "
                      f"h {blob['h']:.0f}mm width {1000 * g['width']:.0f}mm")
            if pick_blob(ex, ik, i, blob, g, rng):
                return True, w.d.time - t0
            ik.sync(w.d)
    plan = None
    for direction in directions_for(blob):
        if direction != "front" and not SIDE_PUSH:
            continue
        plan = plan_rake(ik, w, i, blob, rng, direction)
        if plan is not None:
            break
    if plan is None:
        return False, 0.0
    from plan import door as _door
    if _door.VERBOSE:
        print(f"    rake {plan['direction']} ({plan['mode']}, tilt {plan['tilt']}) blob {np.round(blob['lo'])}-"
              f"{np.round(blob['hi'])} h {blob['h']:.0f}mm")
    doors = {i: w.door_belief(i)}
    ex.jaws(plan["g_open"], label="rake_shape")
    stage = outside_point(i, plan["pre"])
    entry = cartesian_via(ik, [stage, np.array([plan["pre"][0], plan["pre"][1], stage[2]]), plan["pre"]], plan["R"],
                          plan["x"], plan["q_pre"], (), doors, grip=plan["g_open"])
    q_stage = None
    if entry is not None:
        q_stage, _, ok, _ = ik.solve(stage, plan["R"], plan["x"], q0=entry[0], symmetric=False)
        q_stage = q_stage if ok else None
    if q_stage is not None and transfer(ex, ik, q_stage, plan["x"]):
        ex.follow(entry, label="rake_enter")
    elif not transfer(ex, ik, plan["q_pre"], plan["x"]):
        return False, w.d.time - t0
    ex.follow(plan["down"], label="rake_down")
    ex.payload = True                       # slow pull
    ex.follow(plan["pull"], label="rake")
    ex.payload = False
    if plan["open_ok"]:
        ex.jaws(0.037, duration=0.3, label="rake_release")   # free anything caught between the fingers
    ex.wait(0.3)
    ex.follow(plan["lift"], label="rake_up")
    p = plan["end"]
    out = cartesian_via(ik, [p, outside_point(i, p)], plan["R"], plan["x"], ex.q, plan["touch"], doors,
                        grip=plan["g_open"])
    if out is not None:
        ex.follow(out, label="rake_exit")
    return True, w.d.time - t0


def in_tray(w, i, body):
    """Is the part inside printer i's reject tray?"""
    sid = mujoco.mj_name2id(w.m, mujoco.mjtObj.mjOBJ_SITE, f"bin_tray{i}")
    R = w.d.site_xmat[sid].reshape(3, 3)
    p = R.T @ (w.part_pose(body)[0] - w.d.site_xpos[sid])
    hx, hy, th = C.REJECT_TRAY
    return abs(p[0]) < hx and abs(p[1]) < hy + 0.005 and -th - 0.01 < p[2] < 0.03
