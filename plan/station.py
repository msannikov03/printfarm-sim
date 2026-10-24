"""Station skills (two-row cell): bed to box at the +x end of the rail.

One package at a time: one box, one tote (packages never mix).
  request_box   the station takes an erected box of the package's size from the magazine and conveys it to the
                big-part place lift beside the deck (package with large parts) or straight under the package's tote
  place_large   the arm takes a large part from its tray slot (top-down grasp across the narrow side, at a point along
                the part chosen so the place is in reach; the wrist depth camera first measures where the part lies in
                the slot) and sets it on the box floor at its layout pose
  dump_tote     the rail brings the tote over the conveyor, the dump lift raises the box under the drop chute, the trap
                door opens and shakes, shuts, and the station depth camera checks the tote is empty; a leftover part:
                open + shake harder, then an operator (the caller counts it)
  seal_and_ship the box rides into the sealer tunnel and on to its carrier lane. KINEMATIC FROM HERE: once the box is
                inside the tunnel its contents are taken out of the simulation (parked) and a lid is shown; the sealed
                box is then moved as a rigid body (docs/station.md)
  dump_waste    the waste bin's trap door over the scrap bin (purge tabs; the operator empties the bin)
Decisions use sensors and the robot's own bookkeeping: flap encoders, rail and joint encoders, the gripper encoder,
the wrist and station depth cameras; box positions are what the station controller commanded. Where parts really
ended up (part_in_box) is ground truth, for scoring only.
"""
import collections
import itertools
import math

import cv2
import mujoco
import numpy as np

from farm import catalog as C
from farm import scene
from farm.world import FLAP_SHUT
from plan.door import TUCK, transfer
from plan.ik import Belief, frame
from plan.motion import cartesian_line, smooth
from plan.unload import finger_min_z

FLAP_RAMP_S, FLAP_HOLD_S, SHAKE_S = 0.5, 0.6, 0.18     # trap door: open / close ramp, dwell open, one shake swing
SHAKES = (2, 5)                                          # shakes on the first dump, on the retry
CAM_SETTLE_S = 0.2                                       # trap door shut -> camera check (parts come to rest)
SEAL_S = 6.0                                             # dwell in the sealer tunnel (tape + label), kinematic
PLACE_GAPS = (0.006, 0.015, 0.03)                        # part bottom above the box floor when released: low first;
                                                         # higher keeps the fingers' thick roots above a deep box's rim
FINGER_BELOW = 0.0099                                    # YAM fingertips below the pinch point (top-down)
PAD_HALF = 0.007                                         # finger pad half width along the part (tilted grasps)
RETREAT_H = 0.14                                         # straight up after a place: the jaws clear the tallest rim
GRASP_SEEDS = [np.array(q) for q in ((-0.3, 2.0, 1.5, -1.0, 0.0, -0.3), (-0.3, 0.9, 1.6, -1.0, 0.0, 0.0))]
PLACE_SEEDS = [np.array(q) for q in ((1.1, 1.4, 1.2, -1.0, 0.0, 0.0), (1.2, 2.6, 2.2, -1.1, 0.0, -0.4),
                                     (1.2, 2.9, 2.6, -1.3, 0.0, -0.4))]
LOOK_H = 0.11                                            # wrist look pose: pinch this high above the tray floor
LOOK_SEEDS = [np.array(q) for q in ((-0.32, 2.0, 1.78, -1.35, 0.0, -0.32), (-0.55, 2.21, 2.11, -1.46, 0.0, -0.55),
                                    (-0.5, 1.94, 1.7, -1.33, 0.0, -2.0))]
TIP_CLEAR = 0.0015                                       # fingertips above the tray floor
PRESHAPES = (0.008, 0.004)                               # jaw clearance per side when going down beside the part
IK_TOL = (2e-4, math.radians(0.3))                       # station IK: position (m), rotation (rad)
PLACE_CLEAR = 0.006                                      # arm vs the box down the place line (planned; the real arm
                                                         # is a few mm off its model under realistic actuation)
LANE_KEEP = 2                                            # sealed boxes shown per lane (older ones: collected)


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).ravel())
    return q


# ---------------------------------------------------------------- station depth camera
class StationCamera:
    """Depth camera on a boom straight above the drop line (scene: above the robot's reach), looking down into the
    docked tote. Same error model as the chamber cameras (perceive/noise.py preset: range-squared noise, flying pixels,
    dropouts, holes, sheen, calibration error; failed, occluded and smeared scans); a scan is the median of FRAMES
    frames at FPS (the caller waits scan_s). The camera beside the deck it replaces looked in at 37 degrees: the near
    wall of the 70 mm deep tote hid half of the floor.
    Empty check = background subtraction against the empty tote recorded at commissioning (the camera's calibration
    error is the same in both and cancels):
      reference   per-pixel median of REF_SCANS scans; the spread of those scans is the measured scan noise
      floor mask  the flat floor plateau around the tote centre in the reference itself (no calibration needed to find
                  it), eroded ERODE_PX off the walls (dock repeatability, flying pixels at the wall tops)
      raised      reference - scan > max(RAISED_MM, K_SIGMA x the measured noise of the difference); opened (3x3);
                  blobs of MIN_BLOB_PX or more (the smallest part's top is ~4x that)
      decision    a blob seen again (overlapping) in a second scan = a leftover (one scan's blob alone can be a smeared
                  lens); a scan without blobs = empty once the floor has been seen with no hole a part could hide in
                  (HIDE_PX; holes and specular sheen leave NaN patches); an occluded scan (something near the lens)
                  or one with no depth is repeated; MAX_SCANS in all, else no answer (the operator looks)."""
    W, H, FPS = 640, 480, 30.0
    FRAMES, REF_SCANS, MAX_SCANS = 24, 5, 4
    K_SIGMA, RAISED_MM, MAX_RAISE, ERODE_PX = 4.5, 3.0, 0.12, 4
    MIN_BLOB_PX, HIDE_PX, ROI_PAD_PX = 60, 120, 28

    def __init__(self, world, noise="ideal", rng=None):
        from perceive.noise import Calibration, get_noise
        self.w = world
        self.cam = world.m.camera("station").id
        self.noise = get_noise(noise)
        self.rng = rng if rng is not None else np.random.default_rng(77)
        self.calib = Calibration.sample(self.noise, self.rng)
        self.r = None
        self.opt = mujoco.MjvOption()
        self.opt.geomgroup[:] = 0
        self.opt.geomgroup[[0, 1, 2]] = 1
        self.ref = {}                 # slot -> dict(x, roi, z, mask, thr)
        self.last = None

    @property
    def scan_s(self):
        return self.FRAMES / self.FPS

    def _render(self, data):
        if self.r is None:
            self.r = mujoco.Renderer(self.w.m, height=self.H, width=self.W)
            self.r.enable_depth_rendering()
        self.r.update_scene(data, camera=self.cam, scene_option=self.opt)
        return self.r.render().astype(np.float64)

    def depth(self, data=None, frames=None, roi=None, retries=2):
        """Noisy depth (m, NaN = no depth; per-pixel median of `frames` frames) of the region roi = (v0, v1, u0, u1)
        of the image (whole image if None), or None if the camera gave no frame / no depth. The scan's conditions
        (occluder, smear, sheen, holes) are drawn for the whole image; only the region is corrupted (cheaper)."""
        import warnings
        from perceive.noise import corrupt, sample_scan
        nz, data = self.noise, data if data is not None else self.w.d
        v0, v1, u0, u1 = roi or (0, self.H, 0, self.W)
        for _ in range(retries + 1):
            sc = sample_scan(nz, self.rng, (self.H, self.W), 2)
            if sc["mode"] not in ("dead", "blackout"):
                break
        else:
            return None
        mujoco.mj_forward(self.w.m, data)
        z = self._render(data)[v0:v1, u0:u1]
        if nz.is_ideal:
            return z
        sc = _crop_scan(sc, v0, v1, u0, u1)
        stack = np.stack([corrupt(z, nz, self.rng, sc, self.calib) for _ in range(frames or self.FRAMES)])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = np.nanmedian(stack, 0)
        out[np.isfinite(stack).sum(0) < len(stack) // 2] = np.nan
        return out

    def pixel(self, p_world, data=None):
        """Believed pixel (u, v) of a world point (the camera's believed intrinsics and extrinsics)."""
        d, cal = data if data is not None else self.w.d, self.calib
        R = d.cam_xmat[self.cam].reshape(3, 3) @ cal.R_err
        pc = R.T @ (np.asarray(p_world, float) - d.cam_xpos[self.cam] - cal.t_err)
        f = 0.5 * self.H / math.tan(math.radians(self.w.m.cam_fovy[self.cam]) / 2) * cal.f_scale
        return (pc[0] / -pc[2] * f + self.W / 2 - 0.5 + cal.c_off[0],
                -pc[1] / -pc[2] * f + self.H / 2 - 0.5 + cal.c_off[1])

    def commission(self, docks):
        """Reference scans of every (empty) tote at its dock position, taken on a scratch copy of the cell (parts
        parked, trap doors shut on their stops, arm tucked). docks: slot -> carriage x."""
        import warnings
        from plan.door import TUCK
        m = self.w.m
        d = mujoco.MjData(m)
        for slot, x in docks.items():
            mujoco.mj_resetData(m, d)
            d.qpos[self.w.rail_q] = x
            d.qpos[self.w.arm_q] = TUCK
            mujoco.mj_kinematics(m, d)
            mujoco.mj_camlight(m, d)
            dx, dy = C.CADDY_SLOTS[slot]
            o = scene.TOTE_IN + 0.003
            uv = np.array([self.pixel((x + dx + sx * o, dy + sy * o, C.DECK_Z + C.BIN_SIZE[2]), d)
                           for sx in (-1, 1) for sy in (-1, 1)])
            p = self.ROI_PAD_PX
            roi = (max(int(uv[:, 1].min()) - p, 0), min(int(uv[:, 1].max()) + p, self.H),
                   max(int(uv[:, 0].min()) - p, 0), min(int(uv[:, 0].max()) + p, self.W))
            scans = []
            for _ in range(4 * self.REF_SCANS):              # occluded / failed scans are not used
                z = self.depth(d, roi=roi, retries=10)
                if z is not None and np.mean(z[np.isfinite(z)] < 0.5) < 0.01:
                    scans.append(z)
                if len(scans) == self.REF_SCANS:
                    break
            if len(scans) < 2:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                z_ref = np.nanmedian(np.stack(scans), 0)
            u, v = self.pixel((x + dx, dy, C.DECK_Z + C.FLAP_T), d)
            u, v = int(round(u)) - roi[2], int(round(v)) - roi[0]
            if not (0 <= v < z_ref.shape[0] and 0 <= u < z_ref.shape[1]):
                continue
            floor = float(np.nanmedian(z_ref[max(v - 5, 0):v + 6, max(u - 5, 0):u + 6]))
            flat = np.isfinite(z_ref) & (np.abs(np.nan_to_num(z_ref) - floor) < 0.006)
            _, lab = cv2.connectedComponents(flat.astype(np.uint8), connectivity=4)
            mask = (lab == lab[v, u]) & flat
            k = 2 * self.ERODE_PX + 1
            mask = cv2.erode(mask.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
            res = np.concatenate([(s - z_ref)[mask & np.isfinite(s)] for s in scans])
            n = len(scans)
            sig = 1.4826 * float(np.median(np.abs(res))) * math.sqrt(n / (n - 1))      # one scan's noise
            thr = max(self.RAISED_MM / 1000, self.K_SIGMA * sig * math.sqrt(1 + 1.57 / n))
            self.ref[slot] = {"x": x, "roi": roi, "z": z_ref, "mask": mask, "sigma": sig, "thr": thr}

    def _scan(self, ref):
        """One scan of a docked tote -> (raised mask, blob sizes, valid mask) or a reason it is not usable."""
        z = self.depth(roi=ref["roi"])
        if z is None:
            return "no depth"
        mask = ref["mask"]
        ok = mask & np.isfinite(z)
        diff = np.where(ok, ref["z"] - np.nan_to_num(z), 0.0)
        if np.count_nonzero(diff > self.MAX_RAISE) > 0.01 * mask.sum():
            return "occluded"                           # something close to the lens
        raised = ((diff > ref["thr"]) & (diff < self.MAX_RAISE)).astype(np.uint8)
        raised = cv2.morphologyEx(raised, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(raised, connectivity=8)
        keep = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] >= self.MIN_BLOB_PX]
        return np.isin(lab, keep), [int(stats[k, cv2.CC_STAT_AREA]) for k in keep], ok

    def tote_empty(self, slot, carriage_x):
        """Is the tote (trap door shut) empty? -> (True / False / None = no answer, raised pixels, meta);
        meta["scan_s"]: the camera time used (the caller waits it)."""
        meta = {"slot": slot, "scans": [], "scan_s": 0.0}
        self.last = meta
        ref = self.ref.get(slot)
        if ref is None or abs(carriage_x - ref["x"]) > 0.003:
            meta["quality"] = "not docked" if ref is not None else "no reference"
            return None, 0, meta
        mask = ref["mask"]
        seen_clear = np.zeros_like(mask)
        before = np.zeros_like(mask)                    # blobs of the earlier scans
        for _ in range(self.MAX_SCANS):
            meta["scan_s"] += self.scan_s
            got = self._scan(ref)
            if isinstance(got, str):
                meta["scans"].append(got)
                continue
            raised, blobs, ok = got
            meta["scans"].append(blobs)
            if (raised & before).any():
                meta.update(quality="ok", blobs=blobs)
                return False, int(raised.sum()), meta
            before |= raised
            if not blobs:
                seen_clear |= ok
                hidden = (mask & ~seen_clear).astype(np.uint8)
                n, _, stats, _ = cv2.connectedComponentsWithStats(hidden, connectivity=8)
                if n == 1 or stats[1:, cv2.CC_STAT_AREA].max() < self.HIDE_PX:
                    meta.update(quality="ok", blobs=[])
                    return True, 0, meta
        meta["quality"] = "unconfirmed" if before.any() else "failed"
        return None, int(before.sum()), meta

    def close(self):
        if self.r is not None:
            self.r.close()
            self.r = None


def _crop_scan(sc, v0, v1, u0, u1):
    """A sample_scan() description (whole image) moved into the image region (v0:v1, u0:u1)."""
    out = dict(sc)
    h, w = v1 - v0, u1 - u0
    if sc.get("occl") is not None:
        a, b, c, d, zo = sc["occl"]
        a, b, c, d = max(a - v0, 0), min(b - v0, h), max(c - u0, 0), min(d - u0, w)
        out["occl"] = (a, b, c, d, zo) if a < b and c < d else None
    if sc.get("smear") is not None:
        cx, cy, r = sc["smear"]
        out["smear"] = (cx - u0, cy - v0, r)
    out["sheen"] = [(cx - u0, cy - v0, r) for cx, cy, r in sc.get("sheen", [])]
    if sc.get("holes") is not None:
        out["holes"] = sc["holes"][v0:v1, u0:u1]
    return out


# ---------------------------------------------------------------- truth (scoring only)
def part_in_box(w, body, box):
    """Ground truth: is the part inside the box? (scoring / housekeeping only, never a robot decision)"""
    ox, oy, oz = scene.box_outer(w.box_size[box])
    rel = w.part_pose(body)[0] - w.box_pos(box)
    return abs(rel[0]) < ox / 2 and abs(rel[1]) < oy / 2 and -0.002 < rel[2] < oz + 0.03


# ---------------------------------------------------------------- the station
class Station:
    """Station hardware state + skills. `times` collects measured step durations (sim s) for runs/station_times.json:
      to_station   carriage drive from the printers to the station (arm tucked; rail travel, farm runs only)
      dock         carriage move between station positions (arm tucked: fold, slide, unfold)
      box_request  box request -> box in position (place lift or under the tote); runs while the carriage docks
      large_place  one large part, tray -> box (wrist look, grasp, place, back to tuck), docked
      to_tote      box (place lift -> under the tote, lifted) and carriage (-> tote dock) in parallel
      tote_dump    trap door open, shake, shut, camera check (scan time included) (+ retry), box lowered
      seal         box -> sealer, sealed (asynchronous: the robot does not wait for it)
      waste_dump   waste bin over the scrap bin, docking included
      package      robot time per package at the station (sim/farm.py), all of the above that blocks it (a drive
                   from the printers, to_station, is not included)"""

    def __init__(self, world, noise="ideal", rng=None, log=None):
        self.w = world
        self.L = scene.station_layout()
        self.cam = StationCamera(world, noise, rng)
        self.cam.commission({s: x for s, x in self.L["dock"].items() if s != "waste"})
        self.times = collections.defaultdict(list)
        self.lanes = {c: collections.deque() for c in C.LANE_ORDER}
        self.contents = {}                     # box -> [part bodies] the robot put in it (bookkeeping)
        self.placed = {}                       # body -> (pos, quat) where the robot released a large part
        self.scrap = []                        # bodies dumped into the scrap bin
        self.shipped = []                      # (box, lane, [bodies found in it at sealing]) - truth, scoring
        self._req = {}
        self._tunnel_free = 0.0               # sim time the sealer tunnel is clear for the next box
        self.stats = collections.Counter()    # truth scoring at sealing (boxes_sealed, parts_missing, ...)
        self.why = collections.Counter()      # plan_place rejections (debugging)
        self.deck_dz = 0.0                    # the robot's model of its own deck height minus the design (_belief)
        self.log = log or (lambda s: None)
        self.on_action = None                 # callback(str): what the station is doing (HUD / camera director)
        self.tag = ""
        self.action = ""

    def say(self, text):
        """Set the current action; `tag` (e.g. the package id) is appended."""
        self.action = f"{text} ({self.tag})" if self.tag else text

    @property
    def action(self):
        return self._action

    @action.setter
    def action(self, text):
        self._action = text
        if self.on_action is not None and text:
            self.on_action(text)

    # ------------------------------------------------------------ boxes
    def free_box(self, size, ex=None):
        """A box of `size` from the pool; if every one is out, the oldest one waiting in a lane is collected (and if
        none has reached a lane yet, wait for one)."""
        w = self.w
        for _ in range(600):
            for n, s in w.box_size.items():
                if s == size and n not in w.box_live:
                    return n
            old = [(t, c, n) for c, q in self.lanes.items() for t, n in q if w.box_size[n] == size]
            if old or ex is None:
                break
            ex.wait(0.1)
        if not old:
            raise RuntimeError(f"box pool exhausted for size {size}")
        _, c, n = min(old)
        self.lanes[c] = collections.deque((t, b) for t, b in self.lanes[c] if b != n)
        w.box_park(n)                          # collected by the carrier
        return n

    def dump_point(self, size, tote):
        L = self.L
        return np.array([L["x_drop"], L["dump_y"][tote] + L["dump_dy"], L["z_conv"]])

    def request_box(self, ex, size, dest, wait=True):
        """Request an erected box of `size`; it leaves the magazine and rides the conveyor to dest: "place" (the
        big-part place lift, raised) or a tote name (on the conveyor under that tote). Returns the box name."""
        w, L = self.w, self.L
        t0 = w.d.time
        while not self.line_clear() and w.d.time - t0 < 60.0:   # one box on the line at a time: the previous
            ex.wait(0.1)                                         # one must be in the sealer (or past it)
        box = self.free_box(size, ex)
        zc, xd = L["z_conv"], L["x_drop"]
        w.box_spawn(box, (L["mag_x"][size], L["y_mag"], zc))
        self.contents[box] = []
        path = [(L["mag_x"][size], L["y_disp"], zc), (xd, L["y_disp"], zc)]
        if dest == "place":
            path += [(xd, L["y_place"][size], zc), (xd, L["y_place"][size], L["z_place"])]
        else:
            path.append(self.dump_point(size, dest))
        self._req[box] = (t0, w.box_move(box, path))      # requested, in position
        if wait:
            self.wait_box(ex, box)
        return box

    def line_clear(self):
        """No box between the magazine and the sealer: the previous box has entered the tunnel (it may still be
        sealing there). Two boxes on the line could meet (the station controller knows where its boxes are)."""
        L = self.L
        for b in self.w.box_live:
            p = self.w.box_pos(b)
            if abs(p[0] - L["x_drop"]) < 0.01 and p[1] > L["y_seal"] + 0.005:
                return False
        return True

    def wait_box(self, ex, box):
        while self.w.box_busy(box):
            ex.wait(0.05)
        if box in self._req:                   # request -> box in position (it may have waited for the robot)
            t0, t1 = self._req.pop(box)
            self.times["box_request"].append(t1 - t0)

    def box_floor(self, box):
        return self.w.box_pos(box)[2] + C.BOX_WALL_MM / 1000

    # ------------------------------------------------------------ carriage
    def flaps_shut(self):
        return all(self.w.flap_shut(s) for s in self.w.flaps)

    def dock(self, ex, ik, x):
        """Carriage to x with the arm tucked (all trap doors must read shut: an open flap hangs below the deck)."""
        t0 = self.w.d.time
        while not self.flaps_shut():
            if self.w.d.time - t0 > 2.0:
                raise RuntimeError("a trap door does not read shut: the carriage must not move")
            ex.wait(0.1)
        if abs(ex.x - x) < 1e-4 and np.allclose(ex.q, TUCK, atol=1e-3):
            return True
        far = ex.x < self.x_station - 0.05             # coming from the printers: rail travel, timed on its own
        ok = transfer(ex, ik, TUCK, x)
        self.times["to_station" if far else "dock"].append(self.w.d.time - t0)
        return ok

    @property
    def x_station(self):
        """Lowest carriage position the station uses (a dock or the big-part place)."""
        return min(min(self.L["dock"].values()), self.L["x_place"])

    # ------------------------------------------------------------ trap doors
    def _flap_to(self, ex, slot, target, seconds):
        a = self.w.flap_act[slot]
        g0 = float(self.w.d.ctrl[a])
        n = max(int(seconds / 0.01), 1)
        for k in range(1, n + 1):
            self.w.d.ctrl[a] = g0 + (target - g0) * smooth(k / n)
            ex._tick()

    def cycle_flap(self, ex, slot, shakes):
        """Open, dwell, shake, shut. Returns (opened, shut) from the flap encoder."""
        self._flap_to(ex, slot, C.FLAP_OPEN, FLAP_RAMP_S)
        ex.wait(FLAP_HOLD_S)
        opened = self.w.flap_sensor(slot) > C.FLAP_OPEN - math.radians(8)
        for _ in range(shakes):
            self._flap_to(ex, slot, C.FLAP_SHAKE, SHAKE_S)
            self._flap_to(ex, slot, C.FLAP_OPEN, SHAKE_S)
        ex.wait(0.3)
        shut = False
        for _ in range(3):                     # a part caught at the hinge: open a little and shut again
            self._flap_to(ex, slot, FLAP_SHUT, FLAP_RAMP_S)
            ex.wait(0.25)
            if self.w.flap_shut(slot):
                shut = True
                break
            self._flap_to(ex, slot, C.FLAP_SHAKE, FLAP_RAMP_S)
        return opened, shut

    def dump_tote(self, ex, ik, tote, box):
        """Drop the tote's parts into the box (module docstring). Returns a dict: outcome 'dumped' | 'operator'
        (still not empty after the retry, or the camera could not tell, or the trap door would not shut), empty
        (station camera), tries, seconds."""
        w, L = self.w, self.L
        t0 = w.d.time
        size = w.box_size[box]
        pt = self.dump_point(size, tote)
        g = w.box_goal(box)
        w.box_move(box, [(g[0], g[1], pt[2]), pt])          # down off the place lift, along under the tote
        self.dock(ex, ik, L["dock"][tote])
        w.box_move(box, [(pt[0], pt[1], L["z_dump"][size])])
        self.wait_box(ex, box)
        self.times["to_tote"].append(w.d.time - t0)    # box + carriage to the tote (in parallel), box lifted
        t0 = w.d.time
        self.say(f"station: dump tote {tote} -> box")
        out = {"outcome": "dumped", "empty": None, "tries": 0}
        for k, shakes in enumerate(SHAKES):
            out["tries"] = k + 1
            opened, shut = self.cycle_flap(ex, tote, shakes)
            if not shut:
                out.update(outcome="operator", why="trap door does not shut")
                break
            self.say(f"station: camera checks tote {tote}")
            ex.wait(CAM_SETTLE_S)
            empty, n, meta = self.cam.tote_empty(tote, ex.x_meas())
            ex.wait(meta["scan_s"])                    # the frames take their time
            out.update(empty=empty, raised_px=n, scans=meta["scans"])
            if empty:
                break
            if k == 0:
                self.say(f"station: tote {tote} not empty - shake again")
        if not out["empty"]:
            out["outcome"] = "operator"
            out.setdefault("why", "camera: tote not empty" if out["empty"] is False else "camera: no usable scan")
        w.box_move(box, [pt])                  # lower back onto the conveyor
        out["s"] = w.d.time - t0
        self.times["tote_dump"].append(out["s"])
        return out

    def dump_waste(self, ex, ik):
        """Waste bin (purge tabs) over the scrap bin: open, shake, shut. The operator empties the scrap bin: what fell
        in at the last dump is taken out of the simulation now."""
        w = self.w
        t0 = w.d.time
        for b in self.scrap:
            if b in w.in_use:
                w.park(b)
        self.scrap = []
        self.dock(ex, ik, self.L["dock"]["waste"])
        self.say("station: waste -> scrap bin")
        _, shut = self.cycle_flap(ex, "waste", SHAKES[0])
        self.times["waste_dump"].append(w.d.time - t0)
        return shut

    # ------------------------------------------------------------ sealing / shipping (kinematic)
    def seal_and_ship(self, box, lane, on_lane=None, expect=()):
        """Send the box through the sealer to the carrier lane (asynchronous: it runs while the robot works on).
        KINEMATIC FROM HERE: when the box is inside the tunnel, every part inside it is taken out of the simulation
        (parked) and the lid is shown. expect: the bodies that should be in it - scoring only (truth: what is
        really inside at sealing; a missing part, lying wherever it fell, is collected). Returns (t_sealed,
        t_in_lane) in sim time."""
        w, L = self.w, self.L
        xd, zc = L["x_drop"], L["z_conv"]
        p = w.box_goal(box)
        t_start = max(w.d.time, w.box_segs[box][-1][1] if w.box_segs.get(box) else 0.0)
        w.box_move(box, [(p[0], p[1], zc)])
        sealed = {}

        def seal(world):
            inside = [b for b in list(world.in_use) if part_in_box(world, b, box)]
            for b in inside:
                world.park(b)
            world.box_seal(box)
            lost = [b for b in expect if b not in inside]
            for b in lost:
                if b in world.in_use:
                    world.park(b)
            self.stats["boxes_sealed"] += 1
            self.stats["parts_shipped"] += len(inside)
            self.stats["parts_missing"] += len(lost)            # truth: expected but not in the box
            self.stats["boxes_complete"] += not lost
            sealed["parts"] = inside
            sealed["t"] = world.d.time
            self.shipped.append((box, lane, inside))

        # one box in the tunnel at a time: wait on the conveyor until the previous one has left it
        t_go = w.box_segs[box][-1][1] if w.box_segs.get(box) else w.d.time
        hold = self._tunnel_free - (t_go + 1.875 * abs(p[1] - L["y_seal"]) / C.BOX_V)
        if hold > 0:
            w.box_dwell(box, hold)
        t_seal = w.box_move(box, [(xd, L["y_seal"], zc)], on_done=seal)
        lx = L["lane_x"][lane]

        def make_room(world):                   # the lane shows LANE_KEEP boxes; the oldest is collected
            while len(self.lanes[lane]) >= LANE_KEEP:
                _, old = self.lanes[lane].popleft()
                world.box_park(old)

        w.box_dwell(box, SEAL_S, on_done=make_room)
        used = {round(float(world_x), 3) for world_x in (w.box_goal(b)[0] for _, b in self.lanes[lane])}
        slot_x = next((x for x in (lx - 0.1, lx + 0.1) if round(x, 3) not in used), lx - 0.1)
        if len(self.lanes[lane]) >= LANE_KEEP:             # the slot frees when the oldest box is collected
            slot_x = w.box_goal(self.lanes[lane][0][1])[0]

        def arrive(world):
            self.lanes[lane].append((world.d.time, box))
            if on_lane:
                on_lane(box, lane)

        t_lane = w.box_move(box, [(xd, L["y_sort"], zc), (slot_x, L["y_sort"], zc), (slot_x, L["y_sort"] + 0.2, zc),
                                  (slot_x, L["y_lane"], L["z_lane"])], on_done=arrive)
        self._tunnel_free = t_seal + SEAL_S + 1.5
        self.times["seal"].append(t_seal + SEAL_S - t_start)
        self.contents.pop(box, None)
        return t_seal, t_lane

    # ------------------------------------------------------------ large parts: tray -> box
    def slot_floor(self, slot, x):
        dx, dy = C.LARGE_SLOTS[slot]
        return np.array([x + dx, dy, C.DECK_Z + 0.003 + self.deck_dz])

    def measure_slot(self, ex, ik, slot, sku):
        """Wrist depth camera over the tray slot: (centre xy world, yaw, top height) of the part lying in it, or None.
        The slot pose comes from the rail encoder, the camera pose from the joint encoders (believed FK)."""
        from plan.unload import _wrist
        w = self.w
        x = ex.x
        c = self.slot_floor(slot, x)
        q = None
        for jaw in ([0, 1.0, 0], [1.0, 0, 0]):
            q = self._ik(ik, c + [0, 0, LOOK_H], frame([0, 0, -1.0], jaw), x, LOOK_SEEDS)
            if q is not None and not ik.collides(q, x, grip=0.037):
                break
            q = None
        if q is None or not transfer(ex, ik, q, x):
            return None
        ex.settle(tol=0.004, timeout=1.0)
        pts = _wrist(ex).points_pinch(max_range=0.5)
        pts = pts[pts[:, 2] > 0.02]                           # past the fingertips (the jaws are in view)
        p_b, R_b = ik.fk(ex.q_meas(), ex.x_meas())
        pw = pts @ R_b.T + p_b
        hx, hy = C.LARGE_SLOT_SIZE[0] / 2 - 0.007, C.LARGE_SLOT_SIZE[1] / 2 - 0.007
        sel = (np.abs(pw[:, 0] - c[0]) < hx) & (np.abs(pw[:, 1] - c[1]) < hy) & (pw[:, 2] > c[2] + 0.003)
        if sel.sum() < 40:
            return None
        top_z = np.percentile(pw[sel, 2], 98)
        top = sel & (pw[:, 2] > top_z - 0.004)
        xy = pw[top, :2]
        m = xy.mean(0)
        ev, evec = np.linalg.eigh(np.cov((xy - m).T))
        a = evec[:, 1]                                       # long axis of the top face
        yaw = math.atan2(a[1], a[0])
        uv = (xy - m) @ np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        lo, hi = np.percentile(uv, 1, axis=0), np.percentile(uv, 99, axis=0)
        ctr = m + np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]) @ ((lo + hi) / 2)
        fx, fy = (v / 1000 for v in C.CATALOG[sku].footprint)
        ext = hi - lo
        if abs(ext[0] - fx) > 0.02 or abs(ext[1] - fy) > 0.015:
            return None                                       # not the expected part, or partly hidden
        return ctr, yaw, float(top_z - c[2])

    def place_targets(self, box, sku, layout):
        """World (centre at floor height, yaw) of the part on the box floor: the layout pose (box frame: x along the
        long side = world +y, y = world -x; yaw 0 = the part's long side along box x), then the same pose shifted
        along the jaw axis toward the box centre (finger room), as long as the part stays on the floor and clear of
        the parts already in the box (where the robot released them)."""
        w = self.w
        bp, fl = w.box_pos(box), self.box_floor(box)
        xm, ym, yd = layout
        c0 = np.array([bp[0] - ym / 1000, bp[1] + xm / 1000])
        yaw = math.radians(yd + 90.0)
        jaw = np.array([-math.sin(yaw), math.cos(yaw)])            # the part's narrow axis (jaws close along it)
        fx, fy = (v / 1000 for v in C.CATALOG[sku].footprint)
        ox, oy, _ = scene.box_outer(w.box_size[box])
        t = C.BOX_WALL_MM / 1000
        lim = np.array([ox / 2 - t - 0.003, oy / 2 - t - 0.003])
        others = []
        for b in self.contents.get(box, []):
            if b in self.placed:
                p, q = self.placed[b]
                ys = 2 * math.atan2(q[3], q[0])
                sx, sy = (v / 1000 for v in C.CATALOG[w.part_sku[b]].footprint)
                others.append((p[:2], ys, sx, sy))

        def ok(c):
            ext = np.abs([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]]) @ [fx / 2, fy / 2]
            if np.any(np.abs(c - bp[:2]) + ext > lim):
                return False
            for p, ys, sx, sy in others:                           # footprint AABBs must not overlap (+3 mm)
                e2 = np.abs([[math.cos(ys), -math.sin(ys)], [math.sin(ys), math.cos(ys)]]) @ [sx / 2, sy / 2]
                if np.all(np.abs(c - p) < ext + e2 + 0.003):
                    return False
            return True

        toward = 1.0 if jaw @ (bp[:2] - c0) >= 0 else -1.0
        out = []
        for d in (0.0, 0.01, 0.02, 0.03, -0.01):
            c = c0 + toward * d * jaw
            if ok(c):
                out.append((np.array([c[0], c[1], fl]), yaw))
        return out

    def _grasp_options(self, sku):
        """(offset along the part's long axis, tilt about the jaw axis) to try, smallest first."""
        fx = C.CATALOG[sku].footprint[0] / 1000
        smax = max(fx / 2 - 0.022, 0.0)
        offs = [0.0] + [s for s in (0.015, 0.03, 0.045) if s <= smax]
        return [(s * sg, t) for t in (0.0, math.radians(15), math.radians(30)) for s in offs for sg in (1, -1)
                if s or sg > 0]

    def plan_place(self, ik, body, sku, est, target, x):
        """Grasp in the tray + place in the box, planned together (one held pose for both): grasp point along the
        part (offset), pinch height, tilt about the jaw axis (leaning out extends the reach) and jaw opening are
        searched; the place - the usual limit - is checked first. est = (centre xy, yaw, top height) in the slot;
        target = (centre xy, yaw) of the part on the box floor (z: box floor). Returns a plan dict or None."""
        w = self.w
        k = C.CATALOG[sku]
        fx, fy = (v / 1000 for v in k.footprint)
        h = k.height / 1000
        (cxy, yaw, top), (txyz, tyaw) = est, target
        floor_t = C.DECK_Z + 0.003 + self.deck_dz       # the tray floor where the robot's model has it
        p_pick = np.array([cxy[0], cxy[1], floor_t])
        ignore = {w.part_body[body]}
        placed = {w.part_body[b] for b in self.placed if b in w.in_use} | ignore
        # pinch height on the part: high first (the fingers' thick roots stay above the box rim at the place);
        # flat parts: the fingertips (FINGER_BELOW under the pinch) just above the floor
        def zgs(tilt):                         # a tilted pad reaches lower by its half width x sin(tilt)
            low = FINGER_BELOW * math.cos(tilt) + PAD_HALF * math.sin(tilt) + TIP_CLEAR
            return [z for z in ([h - 0.004, 0.6 * h] if h > 0.012 else [low]) if z >= low - 1e-9] or [low]

        yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2      # the part looks the same turned by a half turn
        combos = [(s, tilt, jaw, zg) for (s, tilt), jaw in itertools.product(self._grasp_options(sku), (1.0, -1.0))
                  for zg in zgs(tilt)]
        for s, tilt, jaw, zg in combos:
            g = np.array([s, 0.0, zg])
            R_rel = _ry(tilt) @ frame([0, 0, -1.0], [0, jaw, 0])     # pinch frame in the part frame (the gripper
            # is symmetric: jaw -1 is the same grasp turned by a half turn about the approach)
            rel = (-R_rel.T @ g, R_rel.T)                             # part frame in the pinch frame
            for flip, gap in itertools.product((0.0, math.pi), PLACE_GAPS):
                Rt = _rz(tyaw + flip)
                pt = np.array([txyz[0], txyz[1], txyz[2] + gap])
                pout, Rout = pt + Rt @ g, Rt @ R_rel
                ik.attach(body, None, rel=rel)
                try:
                    q_p = self._ik(ik, pout, Rout, x, PLACE_SEEDS, n_rand=4)
                    if q_p is None or ik.collides(q_p, x, grip=fy / 2 + PRESHAPES[-1], clearance=0.003):
                        self.why["place_ik" if q_p is None else "place_collides"] += 1
                        continue
                    q_pp = self._ik(ik, pout + [0, 0, 0.09], Rout, x, [q_p])
                    if q_pp is None or ik.collides(q_pp, x, grip=fy / 2 + PRESHAPES[-1]):
                        self.why["preplace"] += 1
                        continue
                    lower = cartesian_line(ik, pout + [0, 0, 0.09], pout, Rout, x, q_pp, n=10, symmetric=False)
                    if not lower or any(ik.collides(q, x, grip=fy / 2 + PRESHAPES[-1], clearance=0.003)
                                        for q in lower):
                        self.why["place_line"] += 1
                        continue
                finally:
                    ik.detach()
                # the arm alone, down the whole place line with the jaws at the release opening: PLACE_CLEAR from the
                # box (the real arm is off its model by a few mm; parts already placed were checked above)
                if any(ik.collides(q, x, grip=fy / 2 + PRESHAPES[-1], ignore_bodies=placed, clearance=PLACE_CLEAR)
                       for q in lower):
                    self.why["place_clear"] += 1
                    continue
                for pre, yp in itertools.product(PRESHAPES, (yaw, yaw + math.pi)):   # both labellings of the pick
                    grip = min(0.037, fy / 2 + pre)      # jaw opening around the part (the slot walls are close)
                    Rp = _rz(yp)
                    pin, Rin = p_pick + Rp @ g, Rp @ R_rel
                    q_g = self._ik(ik, pin, Rin, x, GRASP_SEEDS, n_rand=2)
                    if q_g is None or ik.collides(q_g, x, grip=grip, ignore_bodies=ignore, clearance=0.001):
                        self.why["grasp_ik" if q_g is None else "grasp_collides"] += 1
                        continue
                    mujoco.mj_forward(ik.m, ik.d)
                    if finger_min_z(ik) < floor_t + TIP_CLEAR:
                        self.why["tips_low"] += 1
                        continue
                    q_gp = self._ik(ik, pin - 0.08 * Rin[:, 2], Rin, x, [q_g])
                    if q_gp is None or ik.collides(q_gp, x, grip=grip, ignore_bodies=ignore):
                        self.why["pregrasp"] += 1
                        continue
                    return {"q_g": q_g, "q_gp": q_gp, "q_p": q_p, "q_pp": q_pp, "lower": lower, "rel": rel,
                            "grip": grip, "width": fy, "pin": pin, "Rin": Rin, "pout": pout, "Rout": Rout, "s": s,
                            "tilt": tilt, "zg": zg, "h": h, "x": x, "part_pose": (pt, _quat(Rt))}
        return None

    @staticmethod
    def _ik(ik, p, R, x, seeds, n_rand=0):
        """IK to IK_TOL (tighter than the solver's default 1 mm / 1.5 deg: at a 33 mm jaw opening, 1.5 deg of roll
        puts one fingertip 0.9 mm lower, which is most of the tips' clearance over the tray floor)."""
        rng = np.random.default_rng(0)
        for q0 in list(seeds) + [rng.uniform(ik.lo, ik.hi) for _ in range(n_rand)]:
            q, _, ok, _ = ik.solve(p, R, x, q0=q0, iters=300, symmetric=False, tol_p=IK_TOL[0], tol_r=IK_TOL[1])
            if ok:
                return q
        return None

    def place_large(self, ex, ik, slot, box, body, sku, layout=(0.0, 0.0, 0.0)):
        """Tray slot -> box (module docstring). layout = (x_mm, y_mm, yaw_deg) of the part's centre on the box's inner
        floor in the box frame (x along the box's long side = world +y here). Returns (ok, outcome, seconds)."""
        from plan.unload import N_BETWEEN, _wrist
        w, L = self.w, self.L
        x = L["x_place"]
        self.dock(ex, ik, x)
        t0 = w.d.time                          # (docking is timed on its own)
        self.say(f"station: place {sku} from {slot}")
        self._belief(ik)
        est = self.measure_slot(ex, ik, slot, sku)
        if est is None:                                       # fall back to the slot centre (bookkeeping)
            c = self.slot_floor(slot, x)
            est = (c[:2], 0.0, C.CATALOG[sku].height / 1000)
        self._belief(ik, est=(body, sku, est))
        ik.sync(w.d)
        plan = None
        for target in self.place_targets(box, sku, layout):
            plan = self.plan_place(ik, body, sku, est, target, x)
            if plan is not None:
                break
        if plan is None:
            transfer(ex, ik, TUCK, x)
            return False, "no_plan", w.d.time - t0
        ignore = {w.part_body[body]}
        if not transfer(ex, ik, plan["q_gp"], x, {"ignore": ignore}):
            transfer(ex, ik, TUCK, x)
            return False, "no_path_to_part", w.d.time - t0
        ex.jaws(plan["grip"], 0.3, label="preshape")
        down = cartesian_line(ik, plan["pin"] - 0.08 * plan["Rin"][:, 2], plan["pin"], plan["Rin"], x, plan["q_gp"],
                              n=8, symmetric=False)
        ex.follow(down or [plan["q_g"]], label="insert")
        ex.settle(tol=0.004, timeout=1.0)
        ex.grip(True, duration=0.5, label="grip", width=plan["width"])
        ex.payload = True
        ex.wait(0.15)
        up = cartesian_line(ik, plan["pin"], plan["pin"] - 0.08 * plan["Rin"][:, 2], plan["Rin"], x, plan["q_g"], n=8,
                            symmetric=False)
        ex.follow(up or [plan["q_gp"]], label="lift")
        gap = ex.finger_gap_mm()
        held = gap > 1000 * (plan["width"] - 0.010) + 2.0 and self._between_jaws(ex, plan, gap) >= N_BETWEEN
        if not held:
            ex.payload = False
            ex.grip(False, duration=0.3)
            transfer(ex, ik, TUCK, x)
            return False, "not_held", w.d.time - t0
        ik.sync(w.d)
        ik.attach(body, w.d, rel=plan["rel"])
        try:
            ok = transfer(ex, ik, plan["q_pp"], x, {"grip": plan["grip"]})
            if not ok:
                ex.grip(False, duration=0.3)
                return False, "no_path_to_box", w.d.time - t0
            ex.follow(plan["lower"], label="place")     # the checked place line
            ex.settle(tol=0.004, timeout=1.0)
        finally:
            ik.detach()
            ex.payload = False
        ex.jaws(min(0.037, plan["width"] / 2 + PRESHAPES[-1]), 0.4, label="release")   # just clear of the part:
                                                     # wide open, the fingers would hit the box walls
        ex.wait(0.3)
        back = cartesian_line(ik, plan["pout"], plan["pout"] + [0, 0, RETREAT_H], plan["Rout"], x, plan["q_p"], n=10,
                              symmetric=False)
        ex.follow(back or [plan["q_pp"]], label="retreat")
        left = self.in_jaws(ex, ik, box) >= N_BETWEEN          # above the rim: still in the jaws?
        if left:                               # caught on a pad: open wide (the fingertips are above the rim now)
            ex.grip(False, duration=0.3, label="release wide")
            ex.wait(0.6)
            left = self.in_jaws(ex, ik, box) >= N_BETWEEN
            self.stats["place_release_wide"] += 1
        transfer(ex, ik, TUCK, x)
        if left:
            self.stats["place_stuck_truth_in_box"] += part_in_box(w, body, box)   # scoring only
            return False, "stuck_on_finger", w.d.time - t0
        self.placed[body] = plan["part_pose"]
        self.contents.setdefault(box, []).append(body)
        sec = w.d.time - t0
        self.times["large_place"].append(sec)
        return True, "placed", sec

    @staticmethod
    def _between_jaws(ex, plan, gap):
        """Wrist depth points in the jaw volume, reaching up to the held part's top face: a tall part pinched low
        (the housing at 0.6 of its height) shows the camera only its top, 14 mm above the pinch point, outside the
        default +-12 mm volume of WristCam.between_jaws."""
        from plan.unload import _wrist
        p = _wrist(ex).points_pinch()
        up = (plan["h"] - plan["zg"] + 0.006) / math.cos(plan["tilt"]) + 0.010 * math.tan(plan["tilt"])
        hy = max(gap / 2000 - 0.003, 0.002)
        sel = (np.abs(p[:, 0]) < 0.010) & (np.abs(p[:, 1]) < hy) & (p[:, 2] > -max(up, 0.012)) & (p[:, 2] < 0.012)
        return int(sel.sum())

    def in_jaws(self, ex, ik, box):
        """Wrist depth points between the jaws, not counting the box's own walls (the station knows where the box
        is: a wall top seen between tilted jaws is not a part stuck on a pad)."""
        from plan.unload import _wrist
        cam = _wrist(ex)
        pts = cam.points_pinch()
        p_b, R_b = ik.fk(ex.q_meas(), ex.x_meas())
        pw = pts @ R_b.T + p_b
        ox, oy, oz = scene.box_outer(self.w.box_size[box])
        t = C.BOX_WALL_MM / 1000 + 0.006
        rel = pw - self.w.box_pos(box)
        inside = (np.abs(rel[:, 0]) < ox / 2 + 0.006) & (np.abs(rel[:, 1]) < oy / 2 + 0.006) & (rel[:, 2] < oz + 0.006)
        wall = inside & ((np.abs(rel[:, 0]) > ox / 2 - t) | (np.abs(rel[:, 1]) > oy / 2 - t))
        return cam.between_jaws(ex.finger_gap_mm(), pts[~wall])

    def _belief(self, ik, est=None):
        """Planning scene at the station: the parts the robot has put in a box (where it released them) and the part
        it is about to pick (measured in its slot). Nothing else movable is in it.
        deck_dz: the tray rides on the carriage with the arm, so tray-relative targets are taken in the robot's model
        of its carriage. That model carries the registration error (sensors.perturb_model moves the whole carriage):
        a pinch planned 1.5 mm over the design floor height could be 0.1 mm over the floor the collision check sees.
        (The design value is the scene's carriage body, which the sim's truth keeps.)"""
        cid = self.w.m.body("carriage").id
        self.deck_dz = float(ik.m.body_pos[cid][2] - self.w.m.body_pos[cid][2])
        parts = {b: (p, q, 0.006) for b, (p, q) in self.placed.items() if b in self.w.in_use}
        if est is not None:
            body, sku, (cxy, yaw, _) = est
            parts[body] = (np.array([cxy[0], cxy[1], C.DECK_Z + 0.003 + self.deck_dz]), _quat(_rz(yaw)), 0.003)
        ik.belief = Belief(parts)
        ik.sync(self.w.d)

    def forget(self, bodies):
        for b in bodies:
            self.placed.pop(b, None)

    def times_summary(self):
        return {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n": len(v)}
                for k, v in sorted(self.times.items()) if v}

    def summary(self):
        return {"stats": dict(self.stats), "times_s": self.times_summary()}

    def save_times(self, path):
        import json
        import pathlib
        out = self.times_summary()
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(path).write_text(json.dumps(out, indent=1))
        return out
