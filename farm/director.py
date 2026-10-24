"""Automatic camera director for farm videos (render-only).

Called once per video frame (farm/recorder.py hands it the frame's camera), it picks a shot from what the farm is
doing (farm.action) and where the robot is:
  * unload  - camera in the aisle on the far side, looking diagonally into printer i's open door: arm, bed and the
              caddy bins the parts go to;
  * door    - the same idea a little wider, so the leaf's swing is in frame (also used when the carriage arrives
              after a short drive: the approach and the door opening stay one shot);
  * purge   - a close-up of the purge tab pick (the grab tab with its prime line at the front of the bed, picked
              first, into the waste tote): narrow lens from across the aisle, played near real time, for the first
              PURGE_CLOSEUPS plates of a video; it lasts while the gripper is at the door (then the unload shot);
  * drive   - a high chase/tracking shot along the aisle, behind the carriage;
  * station - the bin hand-off and the packaging line (two rows: docking, box requests, the waste dump);
  * place   - two rows: the arm takes a large part from the tray and sets it into the box on the place lift;
  * dump    - two rows: close and low, from just past the deck's +x end and a little to the flap's free side, under
              the deck's overhang: the trap door swinging open with the parts on it, the parts dropping through the
              clear chute into the lifted box (a short push-in);
  * seal    - two rows: the box in the sealer tunnel and on its way to the carrier lane;
  * wide    - establishing shots of the whole cell: at the start, then at the first cut (other than to an unload)
              once `wide_every_s` of video has passed since the last one.
Fixed-position shots are chosen among candidate camera positions: mujoco.mj_ray from the camera to each target point
(gripper, bed, caddy, door handle, deck ...) rejects views blocked by printers, AMS units, benches or the other row,
see-through geoms (door glass, lids) are looked through, a frustum test keeps the targets inside the frame (clear of
the HUD), and candidates too close to geometry or to the robot are dropped. The best visible one wins. While a shot
runs, its main target is re-checked; if it stays hidden or out of frame, the director cuts to the next best angle.
Lookat / distance / azimuth / elevation follow critically damped springs; a change of context is a clean cut, and
every shot is held for at least `min_shot_s` of video.

Render-only: the director reads MjData (body, site and geom poses) and casts rays; it never writes MjData, never
steps, and owns no random state, so the simulated farm is identical with any camera. Its only model writes are
render settings set and restored around its own render call (free-camera field of view; for the overhead shot of the
two-row cell, the alpha of row B's wall fittings): nothing else renders or perceives in between.
"""
import math
import re
from dataclasses import dataclass, field

import mujoco
import numpy as np

from farm import catalog as C
from farm import lab
from farm.render import DEFAULT_GROUPS

SEE_THROUGH_ALPHA = 0.45      # geoms more transparent than this (door glass, roof glass, lids) do not occlude
CAM_CLEARANCE = 0.22          # m of free space around a camera (near plane; no camera inside a cable or a lamp)
ROBOT_CLEARANCE = 0.35        # m from any robot body when the shot starts
SHORT_DRIVE = 0.9             # m: a shorter drive is filmed as the approach to the printer, not as a chase
ROOM_MARGIN = (0.35, 0.15, 0.25)   # x, y, ceiling margin of a camera inside the room
FRAME_X, FRAME_Y = (-0.85, 0.85), (-0.55, 0.72)   # usable part of the frame (tan units / half-size): clear of the HUD
FOVY = {"unload": 58.0, "door": 60.0, "station": 52.0, "drive": 50.0, "follow": 45.0, "wide": 48.0, "place": 50.0,
        "dump": 44.0, "seal": 52.0, "purge": 30.0}
CLEARANCE = {"dump": 0.12, "purge": 0.15}   # per-shot camera clearance (default CAM_CLEARANCE): close-ups sit close
DUMP_PUSH, DUMP_PUSH_S = 0.3, 2.5   # dump shot: starts this much farther out, pushes in over this many seconds
DUMP_SLOWMO, DUMP_SLOWMO_S = 0.35, 2.0   # ... and the first seconds (the drop) play at this fraction of the speed
DIST_OK = {"unload": (0.9, 1.8), "door": (1.1, 2.2), "station": (1.7, 3.2), "place": (1.0, 2.0), "dump": (0.3, 0.7),
           "seal": (1.3, 2.6), "purge": (0.45, 1.0)}
PURGE_CLOSEUPS = 3            # purge close-ups per video (the first plates' purge picks)
PURGE_SLOWMO = 0.15           # ... played at this fraction of the recorder's speed (time-lapse 8x -> ~1.2x real time)
PURGE_AT_DOOR = 0.14          # ... while the pinch is at most this far in front of the door plane (m)
_TOTE = re.compile(r"tote (o\d|waste)")
_ACT = re.compile(r"P(\d+): (.+)")
_DIRS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
                 + [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
_DIRS /= np.linalg.norm(_DIRS, axis=1, keepdims=True)


def look_from(pos, lookat):
    """Free-camera parameters (dist, azimuth, elevation in degrees) that put the camera at pos looking at lookat."""
    v = np.asarray(lookat, float) - np.asarray(pos, float)
    dist = float(np.linalg.norm(v))
    return dist, math.degrees(math.atan2(v[1], v[0])), math.degrees(math.asin(np.clip(v[2] / max(dist, 1e-9), -1, 1)))


def cam_pos(lookat, dist, az, el):
    """Camera position of MuJoCo's free camera (lookat, distance, azimuth, elevation)."""
    a, e = math.radians(az), math.radians(el)
    fwd = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    return np.asarray(lookat, float) - dist * fwd


def _wrap(a):
    return (a + 180.0) % 360.0 - 180.0


class Spring:
    """Critically damped spring towards a goal (exact step, stable for any dt)."""

    def __init__(self, omega):
        self.w, self.x, self.v = omega, None, None

    def reset(self, x):
        self.x, self.v = np.array(x, float), np.zeros(np.shape(x))

    def step(self, goal, dt):
        if self.x is None:
            self.reset(goal)
            return self.x
        x0, v0 = self.x - goal, self.v
        e, c = math.exp(-self.w * dt), v0 + self.w * x0
        self.x = goal + (x0 + c * dt) * e
        self.v = (v0 - self.w * c * dt) * e
        return self.x


@dataclass
class Shot:
    kind: str                     # wide | drive | door | unload | station | follow
    key: tuple                    # what it films: a new key is a cut
    t0: float                     # video time of the cut (s)
    pos: np.ndarray = None        # fixed camera position (door / unload / station / wide)
    offset: tuple = None          # (dist, az, el) behind the lookat for tracking shots (drive / follow)
    lookat: np.ndarray = None     # fixed lookat (wide)
    fovy: float = 45.0
    cutaway: bool = False
    score: float = 0.0
    hidden: int = 0               # consecutive checks with the main target hidden or out of frame
    label: str = ""
    tried: list = field(default_factory=list)


class Director:
    """Per-frame shot choice for farm/recorder.py. `farm` needs only `.action` (sim/farm.py Farm)."""

    def __init__(self, world, farm, fps=30, min_shot_s=3.0, wide_every_s=75.0, wide_s=4.5, pip=True,
                 pip_size=(352, 198), log=None):
        self.w, self.farm = world, farm
        self.fps, self.dt = fps, 1.0 / fps
        self.min_shot_s, self.wide_every_s, self.wide_s = min_shot_s, wide_every_s, wide_s
        self.pip_on, self.pip_size, self.log = pip, pip_size, log
        self.aspect = 16 / 9
        self._pip_cam = None
        m = world.m
        self.frame, self.shot, self.last_wide, self.n_wide = 0, None, -1e9, 0
        self.cuts = []                # (video t, kind, key, label): for logs / cutting highlights
        self.groups = np.zeros(6, np.uint8)
        self.groups[list(DEFAULT_GROUPS)] = 1
        alpha = m.geom_rgba[:, 3].copy()
        has_mat = m.geom_matid >= 0
        alpha[has_mat] = np.minimum(alpha[has_mat], m.mat_rgba[m.geom_matid[has_mat], 3])
        self.see_through = alpha < SEE_THROUGH_ALPHA
        self.robot_geom = np.zeros(m.ngeom, bool)
        self.robot_geom[list(world.robot_geoms)] = True
        self.part_geom = np.zeros(m.ngeom, bool)
        for gs in world.part_geoms.values():
            self.part_geom[gs] = True
        sid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n)
        self.bed_site = [sid(f"bedc{i}") for i in range(C.N_PRINTERS)]
        self.pin_site = [sid(f"pinsite{i}") for i in range(C.N_PRINTERS)]
        self.glass = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"doorglass{i}") for i in range(C.N_PRINTERS)]
        self.carriage = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "carriage")
        flaps = {m.body(f"flap_{s}").id for s in getattr(world, "flaps", ())}   # trap doors barely move: a
        self.robot_bodies = np.array(sorted(world.robot_bodies - flaps))         # camera may come close to them
        self.chamber_cam = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, f"chamber{i}")
                            for i in range(C.N_PRINTERS)]
        self.caddy = np.r_[np.mean(list(C.CADDY_SLOTS.values()), axis=0), C.BIN_SIZE[2]]   # bins' centre on the deck
        self.wb_mats = [k for k in range(m.nmat)
                        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, k) or "").endswith(lab.WALL_B)]
        self.room = (lab.WALL_LEFT_X + ROOM_MARGIN[0], lab.WALL_RIGHT_X - ROOM_MARGIN[0],
                     lab.WALL_FRONT_Y + ROOM_MARGIN[1], lab.WALL_BACK_Y - ROOM_MARGIN[1], lab.CEIL_Z - ROOM_MARGIN[2])
        self.x_mid = (C.N_PER_ROW - 1) * C.PITCH / 2
        self.spring_look, self.spring_cam = Spring(3.2), Spring(3.2)     # lookat; (dist, az, el)
        self.stn = None
        if C.TWO_ROWS and hasattr(world, "box_live"):
            from farm import scene
            self.stn = scene.station_layout()
        self._gid = np.zeros(1, np.int32)
        self._last_action, self.n_purge = None, 0     # purge picks seen so far (the first PURGE_CLOSEUPS get close-ups)
        self._purge_phase = None                      # current purge pick: "approach" -> "at" the door -> "left"

    # ------------------------------------------------------------------ geometry helpers
    def _site(self, k):
        return self.w.d.site_xpos[k].copy()

    def _x(self):
        return float(self.w.d.qpos[self.w.rail_q])

    def pinch(self):
        return self._site(self.w.grasp_site)

    def deck(self, x=None):
        p = self.w.d.xpos[self.carriage] + np.array([0.0, 0.0, 0.08])
        if x is not None:
            p[0] = x
        return p

    def bins(self):
        return self.w.d.xpos[self.carriage] + self.caddy

    def _ray(self, p, t):
        """First opaque geom between p and t: (geom, distance) or None when t is in view."""
        m, d = self.w.m, self.w.d
        v = np.asarray(t, float) - p
        L = float(np.linalg.norm(v))
        if L < 1e-6:
            return None
        u = v / L
        start, run = np.array(p, float), 0.0
        for _ in range(6):
            h = mujoco.mj_ray(m, d, start, u, self.groups, 1, -1, self._gid)
            if h < 0 or run + h >= L - 0.02:
                return None
            g = int(self._gid[0])
            if not self.see_through[g]:
                return g, run + h
            start = start + u * (h + 1e-3)
            run += h + 1e-3
        return None

    def visible(self, p, t, robot_ok=0.5, near=0.12):
        """1 if t is in view from p; robot_ok if the robot (or a part) is in front of it, which is still the
        subject; a hit within `near` of t (the gripper's own fingers, a part on the bed) counts as visible."""
        hit = self._ray(p, t)
        if hit is None:
            return 1.0
        g, dist = hit
        ours = self.robot_geom[g] or self.part_geom[g]
        if ours and np.linalg.norm(np.asarray(t) - p) - dist < near:
            return 1.0
        return robot_ok if ours else 0.0

    def in_frame(self, p, look, fovy, t):
        """Is t inside the usable part of the frame of a camera at p looking at look?"""
        f = np.asarray(look, float) - p
        f /= np.linalg.norm(f)
        r = np.cross(f, [0.0, 0.0, 1.0])
        nr = np.linalg.norm(r)
        if nr < 1e-6:
            return False
        r /= nr
        u = np.cross(r, f)
        v = np.asarray(t, float) - p
        z = v @ f
        if z < 0.15:
            return False
        tn = math.tan(math.radians(fovy) / 2)
        x, y = (v @ r) / (z * tn * self.aspect), (v @ u) / (z * tn)
        return FRAME_X[0] < x < FRAME_X[1] and FRAME_Y[0] < y < FRAME_Y[1]

    def _in_room(self, p):
        x0, x1, y0, y1, z1 = self.room
        return x0 < p[0] < x1 and y0 < p[1] < y1 and 0.4 < p[2] < z1

    def _clear(self, p, r=CAM_CLEARANCE):
        m, d = self.w.m, self.w.d
        for u in _DIRS:
            h = mujoco.mj_ray(m, d, p, u, self.groups, 1, -1, self._gid)
            if 0 <= h < r:
                return False
        return True

    def _far_from_robot(self, p):
        return np.min(np.linalg.norm(self.w.d.xpos[self.robot_bodies] - p, axis=1)) > ROBOT_CLEARANCE

    # ------------------------------------------------------------------ context
    def context(self):
        """(kind, printer, destination x) from the farm's current action and the carriage."""
        a = getattr(self.farm, "action", "") or ""
        if a.startswith("station"):
            if self.stn is not None:                  # two rows: the bed-to-box steps get their own shots
                if "place" in a:
                    return "place", None, None
                mt = _TOTE.search(a)
                if mt and mt.group(1) in self.stn["dump_y"]:
                    return "dump", mt.group(1), None
                if "seal" in a or "lane" in a:
                    return "seal", None, None
            return "station", None, None
        mt = _ACT.match(a)
        if not mt:
            return "idle", None, None
        i, rest = int(mt.group(1)) - 1, mt.group(2)
        if not 0 <= i < C.N_PRINTERS:
            return "idle", None, None
        if rest.startswith("drive") or rest.startswith("waiting"):
            xc, s = C.printer_frame(i)
            dest = xc + s * 0.35
            if abs(dest - self._x()) > SHORT_DRIVE:
                return "drive", i, dest
            return "door", i, None
        if "door" in rest:
            return "door", i, None
        if rest.startswith("purge") and self.n_purge <= PURGE_CLOSEUPS and self._purge_phase != "left":
            return "purge", i, None
        return "unload", i, None

    def _track_purge(self):
        """Count purge picks (an action that turns to "P<n>: purge -> ..." is a new one) and follow the current one:
        the close-up runs from the start of the pick until the gripper has been in the doorway and left it."""
        a = getattr(self.farm, "action", "") or ""
        mt = _ACT.match(a)
        purge = bool(mt) and mt.group(2).startswith("purge")
        if a != self._last_action and purge:
            self.n_purge += 1
            self._purge_phase = "approach"
        self._last_action = a
        if not purge:
            self._purge_phase = None
            return
        i = int(mt.group(1)) - 1
        if not 0 <= i < C.N_PRINTERS:
            return
        at = self._at_door(i)
        if self._purge_phase == "approach" and at:
            self._purge_phase = "at"
        elif self._purge_phase == "at" and not at:
            self._purge_phase = "left"

    def _at_door(self, i):
        """Is the gripper in the doorway of printer i (inside, or at most PURGE_AT_DOOR in front of the door)?"""
        _, s = C.printer_frame(i)
        return s * self.pinch()[1] - C.AISLE_HALF > -PURGE_AT_DOOR

    def _purge_pts(self, i):
        """(tab, middle of the prime line) of printer i's plate (world): the slicer's tab position (plate record),
        front-centre by default; the line runs along the bed's +x from the tab."""
        pt = None
        try:
            pt = self.farm.printers[i].plate["purge_tab"]["center_mm"]
        except (AttributeError, IndexError, KeyError, TypeError):
            pass
        cx, cy = pt if pt is not None else (C.BED_MM[0] / 2, C.PURGE_STRIP_MM / 2)
        tab = np.array(C.bed_mm_to_world(i, cx, cy)) + [0, 0, C.PURGE_TAB_MM[2] / 2000]
        mid = np.array(C.bed_mm_to_world(i, cx + (C.PURGE_TAB_MM[0] + C.PURGE_LINE_MM[0]) / 2, cy))
        return tab, mid

    # ------------------------------------------------------------------ shot targets / candidates
    def _bed(self, i, dz=0.025):
        return self._site(self.bed_site[i]) + [0, 0, dz]

    def _targets(self, kind, i):
        """[(point, weight, robot_ok)] the shot should show; the first is the main one (re-checked while it runs)."""
        if kind == "purge":
            tab, mid = self._purge_pts(i)
            opening = np.array(C.local_to_world(i, 0.0, 0.0, 0.18))
            return [(tab, 2.0, 0.6), (mid, 1.0, 0.7), (self.pinch(), 1.0, 0.7), (opening, 0.3, 1.0)]
        if kind == "unload":
            _, s = C.printer_frame(i)
            bed = self._bed(i)
            corners = [bed + [s * dx, s * dy, 0] for dx in (-0.09, 0.09) for dy in (-0.08, 0.08)]
            opening = np.array(C.local_to_world(i, 0.0, 0.0, 0.24))
            return ([(self.pinch(), 2.0, 0.6), (bed, 1.0, 0.7)] + [(c, 0.3, 0.7) for c in corners]
                    + [(self.bins(), 1.2, 0.8), (opening, 0.6, 1.0)])
        if kind == "door":
            handle = self._site(self.pin_site[i])
            leaf = self.w.d.geom_xpos[self.glass[i]].copy()
            swing = np.array(C.local_to_world(i, C.HINGE_XL + 0.12, -0.20, 0.25))    # the leaf half way open
            opening = np.array(C.local_to_world(i, 0.0, 0.0, 0.24))
            return [(handle, 2.0, 0.8), (leaf, 1.0, 0.8), (swing, 1.0, 0.8), (opening, 0.6, 1.0),
                    (self.pinch(), 1.0, 0.8), (self._bed(i), 0.4, 0.8)]
        L = self.stn
        if kind == "place":                            # box on the place lift, the arm, the tray
            box = np.array([L["x_drop"], max(L["y_place"].values()) - 0.03, L["z_place"] + 0.06])
            tray = np.r_[np.mean(list(C.LARGE_SLOTS.values()), axis=0) + [L["x_place"], 0.0], C.DECK_Z + 0.03]
            return [(box, 2.0, 0.7), (self.pinch(), 1.0, 0.8), (tray, 1.0, 0.7)]
        if kind == "dump":                             # the box under the tote, the drop, the open trap door
            y = L["dump_y"][i] + L["dump_dy"]
            return [(np.array([L["x_drop"], y, L["rim_dump"] + 0.02]), 2.0, 0.5),
                    (np.array([L["x_drop"], y, C.DECK_Z - 0.07]), 1.5, 0.5),
                    (np.array([L["x_drop"], L["dump_y"][i] + 0.03, C.DECK_Z - 0.05]), 1.0, 0.8)]
        if kind == "seal":
            return [(np.array([L["x_drop"], L["y_seal"], L["z_conv"] + 0.12]), 2.0, 0.6),
                    (np.array([L["x_drop"] + 0.6, L["y_sort"], L["z_conv"]]), 1.0, 0.6),
                    (np.array([L["lane_x"]["courier_B"], L["y_lane"], 0.2]), 0.7, 0.6)]
        if kind == "station" and L is not None:        # docking / box request / waste: deck end + conveyor
            sx = L["x_place"]
            dx = np.array([sx - self._x(), 0.0, 0.0])
            return [(self.deck(sx), 2.0, 1.0), (self.bins() + dx, 1.0, 1.0),
                    (np.array([L["x_drop"], L["y_disp"], L["z_conv"] + 0.05]), 1.0, 0.5),
                    (np.array([L["x_drop"], L["y_seal"], L["z_conv"] + 0.1]), 0.5, 0.5)]
        if kind == "station":                          # where the robot will be, not where it is now
            dx = np.array([C.STATION_X - self._x(), 0.0, 0.0])
            sx = C.STATION_X
            return [(self.deck(sx), 2.0, 1.0), (self.bins() + dx, 1.0, 1.0), (self.pinch() + dx, 1.0, 1.0),
                    (np.array([sx + 0.6, 0.0, 0.63]), 1.0, 0.5), (np.array([sx + 1.3, 0.0, 0.75]), 0.5, 0.5)]
        return [(self.deck(), 2.0, 1.0), (self.pinch(), 1.0, 1.0)]

    def _lookat(self, shot, nominal=False):
        kind, i = shot.kind, shot.key[1]
        if kind == "purge":                            # the tab and its line, drifting a little with the gripper
            tab, mid = self._purge_pts(i)
            base = 0.65 * tab + 0.35 * mid
            off = self.pinch() - base
            return base if nominal else base + 0.2 * off * min(1.0, 0.25 / max(np.linalg.norm(off), 1e-6))
        if kind == "unload":                           # bed, caddy, and a little of the gripper
            bed, bins = self._bed(i, 0.04), self.bins()
            base = 0.62 * bed + 0.38 * bins
            if nominal:
                return base
            g = self.pinch()
            off = g - base
            n = np.linalg.norm(off)
            return base + 0.2 * off * min(1.0, 0.4 / max(n, 1e-6))
        if kind == "door":
            leaf = self.w.d.geom_xpos[self.glass[i]]
            opening = np.array(C.local_to_world(i, 0.0, 0.0, 0.22))
            if nominal:
                return 0.5 * opening + 0.5 * np.array(C.local_to_world(i, C.HINGE_XL + 0.1, -0.12, 0.22))
            return 0.4 * leaf + 0.4 * opening + 0.2 * self.pinch()
        if kind in ("place", "dump", "seal"):
            pts = self._targets(kind, i)
            look = sum(wt * q for q, wt, _ in pts) / sum(wt for _, wt, _ in pts)
            if kind == "place" and not nominal:          # drift a little with the gripper
                look = look + 0.15 * (self.pinch() - look)
            return look
        if kind == "station" and self.stn is not None:
            deck = self.deck(self.stn["x_place"] if nominal else None)
            return 0.5 * deck + 0.5 * np.array([self.stn["x_drop"], 0.0, 0.45])
        if kind == "station":
            deck = self.deck(C.STATION_X if nominal else None)
            return 0.5 * deck + 0.5 * np.array([C.STATION_X + 0.75, 0.0, 0.62])
        x = self._x()
        if kind == "drive":
            return np.array([x + shot.key[2] * 0.45, 0.0, 0.75])
        if kind == "follow":
            return np.array([x + 0.3, 0.0, 0.8] if C.N_ROWS == 2 else [x + 0.1, 0.05, 0.85])
        return shot.lookat

    def _candidates(self, kind, i):
        """[(camera position, preference)] for the fixed-position shots."""
        out = []
        if kind == "purge":                            # across the aisle, low and close, looking in through the door
            xc, s = C.printer_frame(i)
            for dx in (0.12, 0.25, 0.38, 0.5, -0.3, -0.45):            # dx > 0: away from the hinge
                for depth in (0.3, 0.45, 0.55, 0.65):
                    for z in (0.98, 1.1, 1.25, 1.4):
                        out.append((np.array([xc + s * dx, s * (C.AISLE_HALF - depth), z]),
                                    (0.4 if dx > 0 else 0.0) - 0.15 * abs(z - 1.1)))
        elif kind in ("unload", "door"):
            xc, s = C.printer_frame(i)
            # depth: from the door plane across the aisle; two rows: stay inside the aisle (not over the other row)
            depths = (0.55, 0.67, 0.78) if C.N_ROWS == 2 else (0.60, 0.80, 1.00)
            if kind == "unload":
                dxs, zs = (0.15, 0.40, 0.65, 0.90, 1.20, -0.35, -0.60, -0.90), (1.15, 1.35, 1.55, 1.80)
            else:
                dxs, zs = (0.45, 0.75, 1.05, 1.35, -0.55, -0.85, -1.15), (1.30, 1.50, 1.70, 1.90)
            for dx in dxs:                              # dx > 0: the side away from the hinge (leaf out of the way)
                for depth in depths:
                    for z in zs:
                        out.append((np.array([xc + s * dx, s * (C.AISLE_HALF - depth), z]), 0.4 if dx > 0 else 0.0))
        elif kind == "place":                          # from above the +y side / the -y side, looking at the box
            xd = self.stn["x_drop"]
            for x in (xd - 0.4, xd - 0.1, xd + 0.25, xd + 0.55):
                for y in (0.75, 0.9, -0.55, -0.75):
                    for z in (1.5, 1.8):
                        out.append((np.array([x, y, z]), 0.4 if y > 0 else 0.1))
        elif kind == "dump":                           # close and low under the overhang; the flap's free side (-y)
            xd, y = self.stn["x_drop"], self.stn["dump_y"][i]     # shows its face with the parts sliding off it
            for x in (xd + 0.36, xd + 0.40, xd + 0.45):
                for dy, pref in ((-0.16, 0.5), (-0.22, 0.45), (-0.10, 0.4), (0.05, 0.25), (0.16, 0.15)):
                    for z in (0.46, 0.49):
                        out.append((np.array([x, y + dy, z]), pref))
        elif kind == "seal":
            xd = self.stn["x_drop"]
            for x in (xd + 0.4, xd + 0.8, xd + 1.2):
                for y in (-0.1, 0.1, 0.3):
                    for z in (1.0, 1.4):
                        out.append((np.array([x, y, z]), 0.3))
        elif kind == "station" and self.stn is not None:
            sx = self.stn["x_place"]
            for y in (-0.8, -0.55, 0.55, 0.8):
                for x in (sx - 0.6, sx, sx + 0.6, sx + 1.1):
                    out += [(np.array([x, y, 1.6]), 0.4), (np.array([x, y, 2.0]), 0.4)]
        elif kind == "station":
            sx = C.STATION_X
            for y in (-0.85, -0.6, -0.3, 0.3, 0.6, 0.85):
                out += [(np.array([sx + 0.9, y, 2.0]), 0.5), (np.array([sx + 0.9, y, 2.4]), 0.5),
                        (np.array([sx + 0.3, y, 2.2]), 0.4), (np.array([sx + 1.5, y, 2.3]), 0.4),
                        (np.array([sx - 0.8, y * 0.4, 2.3]), 0.3), (np.array([sx - 1.3, y * 0.4, 2.1]), 0.2),
                        (np.array([sx + 2.4, y * 0.8, 2.1]), 0.1)]
        return out

    def _pick_fixed(self, kind, key, t, exclude=()):
        i = key[1]
        fovy = FOVY[kind]
        targets = self._targets(kind, i)
        look = self._lookat(Shot(kind, key, t), nominal=True)
        dist_ok = DIST_OK[kind]
        scored = []
        for p, pref in self._candidates(kind, i):
            if not self._in_room(p) or any(np.linalg.norm(p - q) < 0.05 for q in exclude):
                continue
            dist, _, el = look_from(p, look)
            pref -= 0.6 * max(0.0, dist_ok[0] - dist, dist - dist_ok[1]) + 0.03 * max(0.0, -55 - el, el + 18)
            vis = 0.0
            for k, (q, wt, ok) in enumerate(targets):
                v = self.visible(p, q, ok) if self.in_frame(p, look, fovy, q) else 0.0
                vis += wt * v
                if k == 0 and v <= 0.0:
                    vis -= 1.0                          # main target hidden: only as a last resort
            scored.append((vis + pref, p))
        scored.sort(key=lambda sp: -sp[0])
        for score, p in scored[:40]:
            if self._clear(p, CLEARANCE.get(kind, CAM_CLEARANCE)) and self._far_from_robot(p):
                label = kind if i is None else f"{kind} {i}" if isinstance(i, str) else f"{kind} P{i + 1}"
                return Shot(kind, key, t, pos=p, fovy=fovy, score=score, label=label)
        return None

    def _pick_drive(self, key, t):
        dest, sign = key[1], key[2]
        x = self._x()
        best = None
        for dist in (2.3, 2.7):
            for daz in (0, 14, -14, 28, -28):
                for el in (-32, -40, -26):
                    az = (0.0 if sign > 0 else 180.0) + daz
                    score = -0.02 * abs(daz) - 0.02 * abs(el + 32) - 0.3 * abs(dist - 2.3)
                    ok = True
                    for xr in (x, dest):                 # the camera rides along: check start and end of the move
                        look = np.array([xr + sign * 0.45, 0.0, 0.75])
                        p = cam_pos(look, dist, az, el)
                        if not self._in_room(p) or not self._clear(p):
                            ok = False
                            break
                        deck = np.array([xr, 0.0, C.DECK_Z + 0.08])
                        score += 2.0 * self.visible(p, deck, 1.0) + 1.0 * self.visible(p, deck + [0, 0, 0.3], 1.0)
                    if ok and (best is None or score > best[0]):
                        best = (score, (dist, az, el))
        if best is None:
            return None
        return Shot("drive", key, t, offset=best[1], fovy=FOVY["drive"], score=best[0], label="drive")

    def _pick_wide(self, t):
        self.n_wide += 1
        if self.n_wide % 2 == 1:                       # overhead "dollhouse": walls and ceiling are one-sided
            look = np.array([self.x_mid + 1.0, 0.0 if C.N_ROWS == 2 else 0.3, 0.7])
            pos = np.array([self.x_mid - 0.6, -3.2, 5.6])
            return Shot("wide", ("wide", self.n_wide), t, pos=pos, lookat=look, fovy=FOVY["wide"], cutaway=True,
                        label="wide overhead")
        look = np.array([self.x_mid - 0.2, 0.0 if C.N_ROWS == 2 else 0.25, 0.8])   # down the aisle from the bay
        pos = np.array([max(lab.WALL_LEFT_X + 1.4, -2.7), 0.0 if C.N_ROWS == 2 else -0.6, 2.5])
        return Shot("wide", ("wide", self.n_wide), t, pos=pos, lookat=look, fovy=FOVY["wide"], label="wide aisle")

    def _fallback(self, t, key):
        """No candidate passed: the old fixed chase view, following the carriage."""
        return Shot("follow", key, t, offset=(3.4, 128.0, -34.0) if C.N_ROWS == 2 else (2.6, 140.0, -24.0),
                    fovy=FOVY["follow"], label="follow")

    # ------------------------------------------------------------------ per frame
    def _want(self):
        kind, i, dest = self.context()
        if kind == "drive":
            return kind, ("drive", dest, 1.0 if dest >= self._x() else -1.0)
        return kind, (kind, i)

    def _cut(self, shot):
        self.shot = shot
        self.cuts.append((round(self.frame * self.dt, 2), shot.kind, shot.key, shot.label))
        if self.log:
            self.log(f"[camera {self.frame * self.dt:6.1f}s] {shot.label}")
        look = self._lookat(shot)
        if shot.offset is not None:
            self.spring_cam.reset(np.array(shot.offset, float))
        else:
            self.spring_cam.reset(self._goal(shot, look, shot.t0))
        self.spring_look.reset(look)

    def _goal(self, shot, look, t):
        """(dist, az, el) of a fixed-position shot; the dump shot pushes in from DUMP_PUSH farther out."""
        g = np.array(look_from(shot.pos, look), float)
        if shot.kind == "dump":
            g[0] *= 1.0 + DUMP_PUSH * max(0.0, 1.0 - (t - shot.t0) / DUMP_PUSH_S)
        return g

    def _new_shot(self, kind, key, t, exclude=()):
        if kind == "wide":
            self.last_wide = t
            return self._pick_wide(t)
        if kind == "drive":
            return self._pick_drive(key, t) or self._fallback(t, key)
        if kind in ("unload", "door", "station", "place", "dump", "seal", "purge"):
            return self._pick_fixed(kind, key, t, exclude) or self._fallback(t, key)
        return self._fallback(t, key)

    def _main_hidden(self, shot):
        q, _, ok = self._targets(shot.kind, shot.key[1])[0]
        if shot.kind == "station":
            q = self.deck()
        if shot.kind == "dump":                         # the drop under the deck is the subject
            ok = 1.0
        look = self.spring_look.x
        return self.visible(shot.pos, q, ok) < 0.5 or not self.in_frame(shot.pos, look, shot.fovy, q)

    def update(self):
        """Advance one video frame; returns (lookat, dist, az, el, fovy, cutaway)."""
        t = self.frame * self.dt
        self._track_purge()
        kind, key = self._want()
        shot = self.shot
        if shot is None:
            self._cut(self._new_shot("wide", None, t) if self.wide_every_s else self._new_shot(kind, key, t))
        else:
            age = t - shot.t0
            if shot.kind == "wide":
                if age >= self.wide_s:
                    self._cut(self._new_shot(kind, key, t))
            elif key != shot.key and kind == "purge":
                # the purge close-up does not wait for min_shot_s (at 8x the whole pick would be over before the cut),
                # and it is cut to only once a fixed angle is found (no fallback flash; retried every 3rd frame)
                alt = self._pick_fixed(kind, key, t) if self.frame % 3 == 0 else None
                if alt is not None:
                    self._cut(alt)
            elif key != shot.key and kind != "idle" and age >= self.min_shot_s:
                due = self.wide_every_s and t - self.last_wide >= self.wide_every_s and kind != "unload"
                self._cut(self._new_shot("wide" if due else kind, key, t))
            elif shot.kind == "follow" and key == shot.key and kind in DIST_OK and self.frame % 15 == 0:
                alt = self._pick_fixed(kind, key, t)     # a fallback: retry the real shot (the scene may have
                if alt is not None:                      # cleared, e.g. the carriage has docked)
                    self._cut(alt)
            elif shot.pos is not None and self.frame % 6 == 0 and shot.kind != "follow":
                shot.hidden = shot.hidden + 1 if self._main_hidden(shot) else 0
                if shot.hidden * 6 * self.dt > 1.2 and age >= self.min_shot_s:
                    alt = self._new_shot(shot.kind, shot.key, t, exclude=[shot.pos] + shot.tried)
                    if alt.pos is not None:              # a different angle on the same action
                        alt.tried = shot.tried + [shot.pos]
                        self._cut(alt)
                    else:
                        shot.hidden = 0
        shot = self.shot
        self.frame += 1
        if shot.kind == "wide":                          # slow push-in
            f = min((t - shot.t0) / max(self.wide_s, 1e-6), 1.0)
            dist, az, el = look_from(shot.pos, shot.lookat)
            return shot.lookat, dist * (1.06 - 0.08 * f), az, el, shot.fovy, shot.cutaway
        look = self.spring_look.step(self._lookat(shot), self.dt)
        goal = np.array(shot.offset, float) if shot.offset is not None else self._goal(shot, look, t)
        cur = self.spring_cam.x
        goal[1] = cur[1] + _wrap(goal[1] - cur[1])
        dist, az, el = self.spring_cam.step(goal, self.dt)
        return look, float(dist), float(_wrap(az)), float(el), shot.fovy, shot.cutaway

    def time_scale(self):
        """Playback speed factor for farm/recorder.py: slow motion while the parts drop into the box."""
        s = self.shot
        if s is not None and s.kind == "dump" and (self.frame - 1) * self.dt - s.t0 < DUMP_SLOWMO_S:
            return DUMP_SLOWMO
        if s is not None and s.kind == "purge":
            return PURGE_SLOWMO
        return 1.0

    def __call__(self, w):
        """Plain view callable (lookat, dist, az, el) for code that renders by itself (default field of view)."""
        return self.update()[:4]

    # ------------------------------------------------------------------ rendering
    def render(self, cam, w):
        self.aspect = cam.r.width / cam.r.height
        lookat, dist, az, el, fovy, cut = self.update()
        m = self.w.m
        fovy0 = float(m.vis.global_.fovy)
        saved = m.mat_rgba[self.wb_mats, 3].copy() if cut and self.wb_mats else None
        try:
            m.vis.global_.fovy = fovy
            if saved is not None:
                m.mat_rgba[self.wb_mats, 3] = 0.0
            img = cam.look(lookat, dist, az, el).rgb(w.d)
        finally:
            m.vis.global_.fovy = fovy0
            if saved is not None:
                m.mat_rgba[self.wb_mats, 3] = saved
        if self.pip_on and self.shot.kind == "unload":
            img = self._pip(img, self.shot.key[1])
        return img

    def _pip(self, img, i):
        """Chamber camera of printer i, inset at the top left under the HUD bar."""
        import cv2
        from farm.render import Cam
        if self._pip_cam is None:
            self._pip_cam = Cam(self.w.m, *self.pip_size)
        small = self._pip_cam.rgb(self.w.d, self.chamber_cam[i])
        pw, ph = self.pip_size
        x0, y0 = 16, 76
        img = np.ascontiguousarray(img)
        if img.shape[0] < y0 + ph + 4 or img.shape[1] < x0 + pw + 4:
            return img
        img[y0 - 2:y0 + ph + 2, x0 - 2:x0 + pw + 2] = (20, 22, 26)
        img[y0:y0 + ph, x0:x0 + pw] = small
        cv2.putText(img, f"P{i + 1} chamber camera", (x0 + 6, y0 + ph - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def close(self):
        if self._pip_cam is not None:
            self._pip_cam.close()
            self._pip_cam = None
