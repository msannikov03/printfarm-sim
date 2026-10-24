"""Fallback when a part cannot be picked (no grasp, slipped, knocked over) or the camera sees debris: rake it off the
bed's front edge into the printer's reject tray (farm/scene.py), so the printer can start its next plate. The part is
reprinted; the operator empties the tray once per shift.

No grasp is needed: jaws opened to the blob's width, fingertips a few mm above the bed behind the blob, a straight
pull toward the door. Blobs come from the chamber camera (raised points), not from the slicer, so knocked-over parts
and unknown debris are handled the same way.
Long blobs (large SKUs, 130-150 mm): never picked into the caddy's 110 mm reject bin (they would land across it);
lying along the door they are raked into the 300 mm wide tray like any part, lying into the printer they are pushed
off the nearer side edge first (lengthwise beside the bed: the tray is only 40 mm deep).
Deep in the two-row bed none of that reaches; the deep pushes below (rail push, arm side push, back push) clear what
is left onto the chamber floor beside or behind the bed, and every stroke is checked by a re-scan.
"""
import math

import mujoco
import numpy as np
from scipy import ndimage

from farm import catalog as C
from plan.door import RAIL_MIN_L, transfer
from plan.ik import frame
from plan.unload import DXS, GRASP_CLEAR, LIFT as PICK_LIFT, TIP_CLEAR, cartesian_via, drop_pose, finger_min_z, \
    grasp_search, outside_point

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


def plan_blob_grasp(ik, w, i, blob, rng=None, n_seeds=3):
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
    return grasp_search(ik, w, i, p0, opts, [0.6 * h, max(h - 0.004, 0.005)], h, blob_bodies(w, i, blob, ik=ik), rng,
                        n_seeds)


def _blob_box(i, blob, g):
    """The blob as a box in the pinch frame of grasp g (camera outline + height, as plan_blob_grasp saw it): what the
    release into the reject bin lowers and fits between the walls (plan.unload.box_pose)."""
    _, s = C.printer_frame(i)
    (cx, cy), (wu, wv), ang = blob["rect"]
    a = math.radians(ang)
    u = s * np.array([math.cos(a), math.sin(a), 0.0])
    v = s * np.array([-math.sin(a), math.cos(a), 0.0])
    h = blob["h"] / 1000
    c = np.array(C.bed_mm_to_world(i, cx, cy)) + [0.0, 0.0, h / 2]
    R = g["R"]
    return R.T @ (c - g["target"]), R.T @ np.column_stack([u, v, [0.0, 0.0, 1.0]]), np.array([wu, wv, 1000 * h]) / 2000


def pick_blob(ex, ik, i, blob, g, rng=None):
    """Execute a blob grasp and put it into the reject bin like a small part (plan.unload._release_into_bin: level
    jaws, lowered below the rim when the blob fits between the walls, the wrist camera looks into the bin after).
    Returns "in" (the camera confirms it in the bin), "missed" (not seen in the bin: on the deck or the floor, or on
    a rim after the pushes), "unchecked" (released over the bin, the camera could not see in), "lost" (the gripper
    encoder shows it fell on the way to the bin) or "stuck" (still on a finger), all of them truthy: the blob left
    the bed; False if the jaws did not lift it. Every result is counted in ex.blob_drops."""
    from plan import unload as U
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
    box = _blob_box(i, blob, g)
    size = max(blob["rect"][1][0], blob["rect"][1][1], blob["h"])
    rp = None
    for sh in (0.0, 0.25, 0.45):
        x_try = ex.x + s * sh
        rp = U.release_pose(ik, w, "reject", x_try, box[0], 0.015 + size / 2000, q_ref=g["q_pre"], box=box)
        q_go = drop_pose(ik, w, "reject", x_try, None, above=0.06) if rp is None else \
            rp["q"] if rp["q_look"] is None else rp["q_look"]
        if q_go is not None and transfer(ex, ik, q_go, x_try, {"ignore": touch, "grip": 0.012}):
            break
        rp = None
    if ex.finger_gap_mm() <= 1000 * (g["width"] - 0.010) + 2.0:
        ex.grip(False, duration=0.3)                 # the jaws closed on nothing: it fell on the way (deck, floor)
        ex.payload = False
        res = "lost"
    elif rp is None:                                 # legacy drop from above the bin, nothing to check it with
        ex.grip(False, duration=0.4, label="release")
        ex.payload = False
        ex.wait(0.3)
        res = "unchecked"
    else:
        ok, outcome, _ = U._release_into_bin(ex, ik, None, "reject", g, rp, box[0], size=size, box=box)
        ex.payload = False
        res = ("in" if outcome == "placed" else "unchecked") if ok else "stuck" if outcome == "stuck_on_finger" \
            else "missed"
    drops = getattr(ex, "blob_drops", None)
    if drops is None:
        import collections
        drops = ex.blob_drops = collections.Counter()
    drops[res] += 1
    return res


# ---------------------------------------------------------------- deep clearing (two-row cell)
# The two-row bed is 12 cm farther from the rail. Measured on the nominal arm model (bed centre line, door open): with
# the fingertips a few mm above the bed the arm reaches only with the tool leaning >= 50 deg into the printer (65 deg
# to ~180 mm deep, 80 deg to the back edge); vertical or door-side leans never reach. A steep lean makes a front pull a
# wedge: the fingers slope over the blob, and the only face toward the door presses it into the bed (part-bed friction
# 1.2 needs a lean under ~40 deg to slide it). What does work deep in the bed is a horizontal push sideways with a
# vertical face of the closed fingers (pad backs or finger edges), or the fingertips pushing straight into the printer.
# So, after the original planners: side pushes off the nearer, then the far side edge, either with the CARRIAGE doing
# the stroke (rail push: the arm holds one low pose and the rail carries it along the bed's x, so only the start pose
# has to be reachable) or with the arm; then a back push off the bed's back edge. The blob lands on the chamber floor
# beside / behind the bed (59 / 76 mm gaps), not in the tray: a person collects it at the next service (on hardware,
# check it cannot reach the bed's Z drive). Every stroke is checked by a re-scan: a stroke that left the blob where it
# was is not tried again on it (rake() tries the next kind; this also catches an original front rake that jams).
DEEP_TILTS = (65, 80, 50)                 # approach lean into the printer (deg), the ones that reach first
LEAN_YAWS = (0, 20, -20, 35, -35)         # lean direction swung about z from straight into the printer (deg)
PUSH_GAPS = (0.016, 0.026, 0.04)          # pinch site behind the blob's back face (m)
FREE_DXS = (0.0, 0.15, -0.1, 0.3, -0.2, 0.45)   # carriage position relative to the printer centre (m, row frame)
PUSH_PAST = 0.015                         # side / back push: the pinch ends this far short of the edge plus the
                                          # blob's half extent, i.e. the blob centre ends >= this past the edge
BACK_TILT_MIN = 65                        # back push: steeper leans only (the tip face must push mostly horizontally)
MAX_STROKES = 3                           # strokes on one blob in one rake() call (final pass: re-scan in between)
DEEP_SEARCH = True                        # False: the original planners only (A/B)
ORIGINAL = True                           # False: skip the original rakes (tests of the deep pushes alone)
RAIL_PUSH = True
BACK_PUSH = True
SAME_BLOB_MM = 15.0                       # a re-scanned blob this close to where it was did not move
HIST_S = 600.0                            # strokes remembered this long (sim s): one unload's clearing
MAX_JUMP = 0.5                            # rad: larger joint steps between line samples are a branch flip
DEEP_KINDS = ("rail", "side", "back")
SETTLE_S = 0.5                            # wait after a deep push before the next scan (s)
RETRY_SEEDS = 12                          # last resort for a blob nothing else clears: the blob grasp with this many
                                          # random IK seeds (instead of 3); deep at the back the grasp is at the edge
                                          # of reach, where the seeds decide
SIDE_GAP = C.PRINTER_W / 2 - 0.008 - C.BED_MM[0] / 2000                      # bed side edge -> chamber wall (m)
BACK_GAP = C.PRINTER_D - 0.008 - C.BED_FRONT_GAP - C.BED_MM[1] / 1000        # bed back edge -> back wall (m)
FIT_MARGIN_MM = 6.0      # a blob is pushed off an edge only if its extent along the push fits that gap with this to
                         # spare: it drops to the chamber floor instead of wedging between the bed edge and the wall
                         # (lengthwise beside the bed: bar 18 and plate 50 mm fit; the 55 mm housing lying into the
                         # printer wedged in the 59 mm gap, once above the bed surface, so it is not pushed there)


def deep_frames(push, door_in, tilts=DEEP_TILTS):
    """(tilt, lean yaw, face, R) for a closed-jaw horizontal push along `push` (world, horizontal): the tool leans
    with its tip into the printer, the lean swung by LEAN_YAWS, and the jaws rolled so that a vertical face of the
    closed fingers faces the push: 'pads' (jaw axis horizontal, the pad backs push) or 'edges' (the finger edges)."""
    z = np.array([0, 0, 1.0])
    out = []
    for t in tilts:
        a = math.radians(t)
        for yw in LEAN_YAWS:
            c, sn = math.cos(math.radians(yw)), math.sin(math.radians(yw))
            hz = np.array([c * door_in[0] - sn * door_in[1], sn * door_in[0] + c * door_in[1], 0.0])
            appr = math.sin(a) * hz - math.cos(a) * z
            side = np.cross(z, hz)                       # horizontal, across the lean
            if abs(side @ push) < 0.5:                   # this lean does not face the push with a vertical face
                continue
            up = np.cross(appr, side)                    # across the approach, in the lean plane
            out.append((t, yw, "pads", frame(appr, side)))
            out.append((t, yw, "edges", frame(appr, up)))
    return out


def _approaches(door_in, tilts):
    """(tilt, lean yaw, approach) leaning into the printer (back push: the jaw roll is free)."""
    z = np.array([0, 0, 1.0])
    out = []
    for t in tilts:
        a = math.radians(t)
        for yw in LEAN_YAWS:
            c, sn = math.cos(math.radians(yw)), math.sin(math.radians(yw))
            hz = np.array([c * door_in[0] - sn * door_in[1], sn * door_in[0] + c * door_in[1], 0.0])
            out.append((t, yw, math.sin(a) * hz - math.cos(a) * z))
    return out


def _solve(ik, p, R, appr, x, q0, iters=150):
    """Full frame R (symmetric jaws), or only the approach axis when R is None. Returns (q, ok)."""
    if R is not None:
        q, _, ok, _ = ik.solve(p, R, x, q0=q0, iters=iters)
        return q, ok
    q, _, ok, _ = ik.solve(p, np.eye(3), x, q0=q0, iters=iters, align_z=appr)
    return q, ok and ik.fk(q, x)[1][:, 2] @ appr > 0.995       # align_z also accepts the tool pointing up


_GRID = np.array(np.meshgrid(*[np.linspace(-1, 1, 5)] * 3)).reshape(3, -1).T     # box sample points (unit)


def finger_low(ik, axis, lo, hi):
    """(lowest point of the finger collision geometry, lowest point of the part of it whose world `axis` (0 = x,
    1 = y) coordinate lies in [lo, hi]) in the IK scratch state (after a forward pass); the second is inf if no part
    of the fingers is there. A steep lean can put the finger base lowest, far from the blob: what pushes the blob is
    the finger where the blob is, and that is what has to skim the bed."""
    m, d = ik.m, ik.d
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] not in ik.w.finger_bodies or not m.geom_contype[g]:
            continue
        R, c, size = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g], m.geom_size[g]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            pts.append(c + (_GRID * size) @ R.T)
        elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            t = np.linspace(-size[1], size[1], 7)[:, None]
            pts.append(c + t * R[:, 2] - [0.0, 0.0, size[0]])
        else:
            pts.append((c - [0.0, 0.0, size[0]])[None])
    pts = np.concatenate(pts)
    inside = (pts[:, axis] >= lo) & (pts[:, axis] <= hi)
    return float(pts[:, 2].min()), float(pts[inside, 2].min()) if inside.any() else np.inf


def _low_pose(ik, p, R, appr, x, seeds, doors, bed, band, slab, warm=None):
    """Closed jaws, pinch site at world point p, the height corrected so that the fingers are TIP_Z above the bed
    where the blob is (slab = (world axis, lo, hi)); nothing of them lower than TIP_CLEAR anywhere.
    warm: converged solutions of earlier candidates, tried first and extended (warm starts). Returns (q, p) or None."""
    bed_z = C.BENCH_H + C.BED_ZL
    for q0 in (warm[-2:][::-1] if warm else []) + list(seeds):
        tgt = np.array(p, float)
        q, ok = _solve(ik, tgt, R, appr, x, q0)
        if ok:
            ik.collides(q, x, None, grip=0.0, ignore_bodies={bed})            # forward pass: finger heights
            z_slab = finger_low(ik, *slab)[1]
            if not np.isfinite(z_slab):
                _rej("deep_not_beside")
                continue
            tgt[2] += bed_z + TIP_Z - z_slab
            q, ok = _solve(ik, tgt, R, appr, x, q)
        if ok and warm is not None:
            warm.append(q.copy())
        if not ok:
            _rej("deep_ik")
            continue
        if ik.collides(q, x, doors, grip=0.0, ignore_bodies={bed}, clearance=0.002):
            _rej("deep_start_collides")
            continue
        z_all, z_slab = (z - bed_z for z in finger_low(ik, *slab))
        if z_all < band[0] or not band[0] <= z_slab <= band[1]:
            _rej("deep_tip_z")
            continue
        return q, tgt
    return None


def _line(ik, pts, R, appr, x, q_seed, ignore, doors, n, band=None, slab=None):
    """Closed jaws along a polyline at a fixed frame (or fixed approach, roll free, when R is None); every sample
    collision-free (fine clearance) and, with band=(lo, hi) and slab, the fingers that high above the bed where the
    blob is and nothing of them below band[0]."""
    bed_z = C.BENCH_H + C.BED_ZL
    qs, q = [], np.asarray(q_seed, float)
    for a, b in zip(pts[:-1], pts[1:]):
        for s in np.linspace(0, 1, n + 1)[1:]:
            if R is not None:
                q1, _, ok, _ = ik.solve(a + s * (b - a), R, x, q0=q, iters=80, symmetric=False)
            else:
                q1, ok = _solve(ik, a + s * (b - a), None, appr, x, q, iters=80)
            if not ok or np.max(np.abs(q1 - q)) > MAX_JUMP or \
                    ik.collides(q1, x, doors, grip=0.0, ignore_bodies=ignore, clearance=GRASP_CLEAR):
                return None
            if band is not None:
                z_all, z_slab = (z - bed_z for z in finger_low(ik, *slab))
                if z_all < band[0] or not band[0] <= z_slab <= band[1]:
                    return None
            q = q1
            qs.append(q.copy())
    return qs


def plan_push(ik, w, i, blob, rng=None, kind="rail", direction="left"):
    """Deep-bed push of `blob` (see above). kind 'rail': the carriage carries the arm, held in one low pose, along the
    bed's x off the `direction` ('left' / 'right') edge; 'side': the arm strokes there at a fixed carriage position;
    'back': the fingertips push it into the printer off the back edge (direction 'back'). Returns a plan dict (the
    keys of plan_rake, plus 'xs': rail positions per stroke sample, 'x_end') or None."""
    from plan.door import own_rig
    xc, s = C.printer_frame(i)
    rng = rng or np.random.default_rng(0)
    bed_z = C.BENCH_H + C.BED_ZL
    db = np.array(PUSH_DIRECTIONS[direction])
    dw = s * np.array([db[0], db[1], 0.0])
    lo, hi = np.asarray(blob["lo"], float), np.asarray(blob["hi"], float)
    ctr, ext = (lo + hi) / 2, (hi - lo) / 2
    along = float(np.abs(db) @ ext)
    edge = {"left": ctr[0], "right": C.BED_MM[0] - ctr[0], "back": C.BED_MM[1] - ctr[1]}[direction]
    h = blob["h"] / 1000
    bed = w.m.body(f"bed{i}").id
    touch = blob_bodies(w, i, blob, ik=ik) | {bed}
    if not _corridor_clear(ik, w, i, blob, db, touch):
        _rej("deep_corridor")
        return None
    doors = {i: w.door_belief(i)}
    door_in = np.array([0.0, s, 0.0])
    band = (TIP_CLEAR, max(TIP_Z + 0.003, min(0.5 * h, 0.008)))
    if direction == "back":                              # where the blob is, across the push (world axis, lo, hi)
        ends = [xc + s * (v / 1000 - C.BED_MM[0] / 2000) for v in (lo[0], hi[0])]
        slab = (0, min(ends), max(ends))
    else:
        ends = [s * (C.AISLE_HALF + C.BED_FRONT_GAP + v / 1000) for v in (lo[1], hi[1])]
        slab = (1, min(ends), max(ends))
    seeds = [np.array([1.57 * s, 1.0, 1.4, -0.9, 0, 0]), np.array([1.57 * s, 0.6, 1.9, -1.3, 0, 0])]
    rig = own_rig(w)
    warm = []                                             # converged low poses: seeds for the next candidates
    if kind == "back":
        frames = [(t, yw, "tips", None, a) for t, yw, a in _approaches(door_in, [t for t in DEEP_TILTS
                                                                                 if t >= BACK_TILT_MIN])]
    else:
        frames = [(t, yw, face, R, R[:, 2]) for t, yw, face, R in deep_frames(dw, door_in)]
    for tilt, yw, face, R, appr in frames:
        for back in PUSH_GAPS:
            off_mm = along + 1000 * back
            sb = ctr - db * off_mm                           # start (bed mm): behind the blob's back face
            if not (-5 <= sb[0] <= C.BED_MM[0] + 5 and -5 <= sb[1] <= C.BED_MM[1] + 5):
                continue
            start0 = np.array(C.bed_mm_to_world(i, *sb))
            start0[2] = bed_z + 0.012
            length = (edge + off_mm - along) / 1000 + PUSH_PAST    # blob centre >= PUSH_PAST past the edge
            n = max(8, int(length / 0.01))
            for dx in FREE_DXS:
                x = xc + s * dx
                if s * (x - xc) < RAIL_MIN_L:
                    continue
                low = _low_pose(ik, start0, R, appr, x, seeds + [rng.uniform(ik.lo, ik.hi)], doors, bed, band, slab,
                                warm)
                if low is None:
                    continue
                q, start = low
                Rs = ik.fk(q, x)[1] if R is not None else None       # the solved frame (jaws may be flipped)
                pre = start + [0, 0, h + 0.03]
                q_pre, ok = _solve(ik, pre, Rs, appr, x, q)
                if not ok:
                    _rej("deep_pre_ik")
                    continue
                down = _line(ik, [pre, start], Rs, appr, x, q_pre, {bed}, doors, 6)
                if not down:
                    _rej("deep_down")
                    continue
                end = start + length * dw
                if kind == "rail":
                    xs = [x + dw[0] * length * k / n for k in range(1, n + 1)]
                    if any(ik.collides(down[-1], xk, doors, grip=0.0, ignore_bodies=touch | rig,
                                       clearance=GRASP_CLEAR) for xk in xs):
                        _rej("rail_slide")
                        continue
                    pull, x_end = [down[-1].copy() for _ in xs], xs[-1]
                else:
                    xs, x_end = None, x
                    pull = _line(ik, [start, end], Rs, appr, x, down[-1], touch, doors, n, band=band, slab=slab)
                    if pull is None:
                        _rej("deep_stroke")
                        continue
                lift = _line(ik, [end, end + [0, 0, LIFT]], Rs, appr, x_end, pull[-1], touch, doors, 4)
                if not lift:
                    _rej("deep_lift")
                    continue
                _rej("deep_ok")
                return {"x": x, "x_end": x_end, "open_ok": False, "R": ik.fk(q_pre, x)[1], "pre": pre,
                        "q_pre": q_pre, "down": down, "pull": pull, "xs": xs, "lift": lift, "end": end + [0, 0, LIFT],
                        "g_open": 0.0, "tilt": tilt, "lean_yaw": yw, "face": face, "mode": kind, "touch": touch,
                        "direction": direction, "length": length}
    return None


PUSH_DIRECTIONS = dict(DIRECTIONS, back=(0.0, 1.0))
CORRIDOR_MM = 10.0       # the lane a pushed blob travels along must be this far clear of everything else on the bed


def _corridor_clear(ik, w, i, blob, db, touch):
    """Is the lane the blob travels along (its footprint swept along db to the edge, CORRIDOR_MM wider) free of the
    other things the camera sees on the bed (believed parts, other blobs)? Mid-unload a push must not shove the blob
    into the parts still to be picked. Belief only; no belief (oracle) -> True."""
    bel = getattr(ik, "belief", None)
    if bel is None:
        return True
    xc, s = C.printer_frame(i)
    lo, hi = np.asarray(blob["lo"], float), np.asarray(blob["hi"], float)
    db = np.asarray(db, float)
    lane_lo, lane_hi = lo - CORRIDOR_MM, hi + CORRIDOR_MM
    k = int(np.flatnonzero(db)[0])                     # the push axis (bed frame)
    if db[k] > 0:
        lane_hi[k] = C.BED_MM[k] + CORRIDOR_MM
    else:
        lane_lo[k] = -CORRIDOR_MM
    boxes = [(np.asarray(b["lo"], float), np.asarray(b["hi"], float)) for b in getattr(bel, "raw", [])
             if not np.all((lo - 5.0 <= _centre(b)) & (_centre(b) <= hi + 5.0))]     # (the blob itself, re-scanned)
    for n, (p, _, _) in bel.parts.items():
        if w.part_body[n] in touch:
            continue
        sku = w.part_sku[n]
        r = 0.5 * float(np.hypot(*(C.PURGE_TAB_MM[:2] if sku == "purge_tab" else C.CATALOG[sku].footprint)))
        c = np.array([(s * (p[0] - xc) + C.BED_MM[0] / 2000) * 1000,
                      (s * p[1] - C.AISLE_HALF - C.BED_FRONT_GAP) * 1000])
        boxes.append((c - r, c + r))
    return not any(np.all(b_lo < lane_hi) and np.all(b_hi > lane_lo) for b_lo, b_hi in boxes)


def _extent(blob):
    """Extent of the blob along the bed's x and y (mm): the smaller of two over-estimates, the raised-point box (it
    also holds the visible side faces and the camera blur) and the box around the top-face outline rectangle (too
    large for round things when the rectangle comes out rotated)."""
    ex, ey = np.subtract(blob["hi"], blob["lo"])
    if "rect" not in blob:
        return float(ex), float(ey)
    _, (rw, rh), ang = blob["rect"]
    a = math.radians(ang)
    return (min(float(ex), abs(rw * math.cos(a)) + abs(rh * math.sin(a))),
            min(float(ey), abs(rw * math.sin(a)) + abs(rh * math.cos(a))))


def _tried(ik, i, blob, t):
    """Stroke kinds already executed on this blob (where it still is) in this clearing: (mode, direction)."""
    c = _centre(blob)
    return {k for (j, cc, k, tt) in getattr(ik, "clear_hist", []) if j == i and t - tt < HIST_S
            and np.hypot(*(cc - c)) < SAME_BLOB_MM}


def _plan_any(ik, w, i, blob, rng, skip=()):
    """The clearing plan for one blob, in order of preference: the original rakes (front into the tray, then the side
    pushes, as directions_for orders them), then the deep pushes: rail push off the nearer side edge, then the far
    one, the same with arm strokes, and a back push. Kinds in `skip` ((mode, direction)) are not tried."""
    dirs = [d for d in directions_for(blob) if d == "front" or SIDE_PUSH]
    for direction in dirs if ORIGINAL else ():
        if any(k[1] == direction and k[0] in ("behind", "cage") for k in skip):
            continue                                     # an original rake that left this blob where it was
        plan = plan_rake(ik, w, i, blob, rng, direction)
        if plan is not None:
            return plan
    if not DEEP_SEARCH:
        return None
    wx, wy = _extent(blob)
    sides =[d for d in dirs if d != "front"] if wx <= 1000 * SIDE_GAP - FIT_MARGIN_MM else []
    tries = [("rail", d) for d in sides if RAIL_PUSH] + [("side", d) for d in sides] + \
        ([("back", "back")] if BACK_PUSH and wy <= 1000 * BACK_GAP - FIT_MARGIN_MM else [])
    for kind, direction in tries:
        if (kind, direction) in skip:
            continue
        plan = plan_push(ik, w, i, blob, rng, kind, direction)
        if plan is not None:
            return plan
    return None


def _execute(ex, ik, i, plan):
    """Run a rake / push plan: in through the door, down behind the blob, the stroke (arm or carriage), release,
    up and out. Returns False if the arm could not get to the start."""
    w = ex.w
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
        return False
    ex.follow(plan["down"], label="rake_down")
    ex.payload = True                       # slow stroke
    ex.follow(plan["pull"], xs=plan.get("xs"), label="rail_push" if plan.get("xs") else "rake")
    ex.payload = False
    if plan["open_ok"]:
        ex.jaws(0.037, duration=0.3, label="rake_release")   # free anything caught between the fingers
    ex.wait(0.3)
    ex.follow(plan["lift"], label="rake_up")
    p, x_end = plan["end"], plan.get("x_end", plan["x"])
    R_end = ik.fk(plan["lift"][-1], x_end)[1]
    out = cartesian_via(ik, [p, outside_point(i, p)], R_end, x_end, ex.q, plan["touch"], doors, grip=plan["g_open"])
    out_ok = out is not None
    if out_ok:
        ex.follow(out, label="rake_exit")
    else:                                   # no straight way out (beside a post after a rail push): a planned move to
        from plan.door import TUCK          # the tuck pose, so the arm is not left in the chamber camera's view
        ik.sync(w.d)
        out_ok = transfer(ex, ik, TUCK, ex.x, {"ignore": plan["touch"]})
        if not out_ok and _door_verbose():
            print("    rake exit: no path out of the printer")
    if plan["mode"] in DEEP_KINDS:
        if out_ok:                          # jaws open again, out of the printer: the door handle is approached with
            ex.grip(False, duration=0.3, label="rake_open")     # open jaws (closed ones hit the bar and the leaf)
        ex.wait(SETTLE_S)                   # what fell off an edge comes to rest before the camera looks again
    return True


def _door_verbose():
    from plan import door as _door
    return _door.VERBOSE


def _centre(b):
    return (np.asarray(b["lo"], float) + np.asarray(b["hi"], float)) / 2


def _rescan(ex, ik, i):
    """Chamber scan -> blobs; the planning scene is rebuilt from it (final clearing pass: no believed parts)."""
    from plan.unload import bed_camera, scan, scene_belief
    w = ex.w
    mujoco.mj_forward(w.m, w.d)
    pts = scan(bed_camera(w, i))
    ik.belief = scene_belief(w, i, {}, {}, pts)
    ik.sync(w.d)
    return blobs(pts)


def rake(ex, ik, i, blob, rng=None):
    """Clear one blob off the bed: pick it into the reject bin when it fits there and has a grasp (BLOB_PICK);
    else the first feasible plan of _plan_any (front rake into the reject tray, side pushes, deep pushes).
    Every executed stroke is remembered (ik.clear_hist); a kind that left the blob where it was is not planned on it
    again. In the final clearing pass (no believed parts) the camera re-scans after a stroke and, while the blob is
    still there, the next kind is tried, at most MAX_STROKES strokes; mid-unload the caller re-scans and re-estimates.
    When no plan is left, the blob grasp is searched once more with RETRY_SEEDS IK seeds. Returns (executed, seconds)."""
    w = ex.w
    t0 = w.d.time
    ik.sync(w.d)
    from plan import door as _door
    ik.clear_hist = [e for e in getattr(ik, "clear_hist", []) if t0 - e[3] < HIST_S]
    ik.last_clear = []
    for _ in range(MAX_STROKES):
        pickable = BLOB_PICK and "rect" in blob and fits_reject_bin(blob)
        g = None
        if pickable:                                                 # graspable: picking beats pushing
            g = plan_blob_grasp(ik, w, i, blob, rng)
            if g is not None:
                if _door.VERBOSE:
                    print(f"    blob pick into the reject bin: blob {np.round(blob['lo'])}-{np.round(blob['hi'])} "
                          f"h {blob['h']:.0f}mm width {1000 * g['width']:.0f}mm")
                res = pick_blob(ex, ik, i, blob, g, rng)
                if res:
                    ik.last_clear.append("pick" if res in ("in", "unchecked") else f"pick_{res}")
                    return True, w.d.time - t0
                ik.last_clear.append("pick_failed")
                ik.sync(w.d)
        plan = _plan_any(ik, w, i, blob, rng, skip=_tried(ik, i, blob, w.d.time))
        if plan is None:
            if DEEP_SEARCH and pickable and g is None:   # nothing else left: the blob grasp again, with a wider IK
                g = plan_blob_grasp(ik, w, i, blob, rng, n_seeds=RETRY_SEEDS)     # search (more random seeds)
                res = pick_blob(ex, ik, i, blob, g, rng) if g is not None else False
                if res:
                    ik.last_clear.append("pick_retry" if res in ("in", "unchecked") else f"pick_retry_{res}")
                    return True, w.d.time - t0
                if g is not None:
                    ik.last_clear.append("pick_failed")
            break
        if _door.VERBOSE:
            print(f"    rake {plan['direction']} ({plan['mode']}, tilt {plan['tilt']}) blob {np.round(blob['lo'])}-"
                  f"{np.round(blob['hi'])} h {blob['h']:.0f}mm")
        if not _execute(ex, ik, i, plan):                 # could not get to the start (as before: time counted)
            return any(c != "pick_failed" for c in ik.last_clear), w.d.time - t0
        ik.last_clear.append(f"{plan['mode']}:{plan['direction']}")
        ik.clear_hist.append((i, _centre(blob), (plan["mode"], plan["direction"]), w.d.time))
        if ik.belief is not None and ik.belief.parts:
            return True, w.d.time - t0                     # mid-unload: the caller re-scans and re-estimates
        still = [b for b in _rescan(ex, ik, i) if np.hypot(*(_centre(b) - _centre(blob))) < SAME_BLOB_MM]
        if not still:
            return True, w.d.time - t0                     # moved (or gone): the caller's next scan decides
        blob = still[0]
    executed = any(c != "pick_failed" for c in ik.last_clear)
    return executed, (w.d.time - t0) if executed else 0.0


def in_tray(w, i, body):
    """Is the part inside printer i's reject tray?"""
    sid = mujoco.mj_name2id(w.m, mujoco.mjtObj.mjOBJ_SITE, f"bin_tray{i}")
    R = w.d.site_xmat[sid].reshape(3, 3)
    p = R.T @ (w.part_pose(body)[0] - w.d.site_xpos[sid])
    hx, hy, th = C.REJECT_TRAY
    return abs(p[0]) < hx and abs(p[1]) < hy + 0.005 and -th - 0.01 < p[2] < 0.03
