"""What the robot can actually measure, and how its actuation differs from its own model.

The planner and the checks in plan/ must only use what a real robot has:
  * joint encoders (noisy, read in the robot's own calibration frame) -> forward kinematics in the robot's MODEL;
  * the gripper motor encoder (jaw gap);
  * the chamber depth camera (perceive/verify.py);
  * a wrist depth camera on link_6, hand-eye calibrated to the pinch point (WristCam below).
Simulator state (part poses, the true pinch position) is used only to report results.

Actuation realism (`PRESETS`, applied by `setup`):
  * joint zero offsets: the true joint goes to commanded + zero; the encoder reads true - zero, so the robot does not
    see it (it is a calibration error, not a tracking error);
  * link / base calibration error: the planning model (`IK.m`) gets perturbed link offsets and a perturbed arm base,
    so the robot's kinematic model differs from the true one;
  * rail encoder zero offset;
  * backlash between motor and joint (motor-side encoder: invisible to the robot);
  * control latency (commands reach the servos `latency_s` late);
  * encoder noise and gripper-gap noise.
All magnitudes are ASSUMPTIONS for a hobby-grade 6-DoF arm with planetary-geared QDD motors after a touch-off
calibration (see `PRESETS` comments); "ideal" reproduces the old behaviour exactly.
"""
import collections
import math

import mujoco
import numpy as np

from farm import catalog as C

PRESETS = {
    # zero_sigma_deg: per-joint homing/zero error; link_sigma_mm: per-link offset error in the robot's model;
    # base_sigma_mm: arm base / printer registration error after touch-off; rail_zero_sigma_mm: rail encoder zero;
    # backlash_deg: gearbox backlash (full play); latency_s: command latency; enc_noise_deg: encoder read noise;
    # grip_noise_mm: jaw-gap read noise; wrist_sigma_mm: wrist depth noise (per pixel); handeye_sigma_mm: wrist
    # camera hand-eye calibration error.
    "ideal": dict(zero_sigma_deg=0.0, link_sigma_mm=0.0, base_sigma_mm=0.0, rail_zero_sigma_mm=0.0,
                  backlash_deg=0.0, latency_s=0.0, enc_noise_deg=0.0, grip_noise_mm=0.0,
                  wrist_sigma_mm=0.0, handeye_sigma_mm=0.0),
    # ASSUMPTIONS: homing repeatability of geared QDD joints ~0.1-0.3 deg; machined/printed link tolerance ~0.3 mm;
    # touch-off registration of the printer to the robot ~1 mm (real/calib.py); planetary gearbox play ~0.1 deg;
    # USB-CAN bus + 100 Hz loop ~20 ms; short-range stereo depth (D405 class) ~0.5 mm at 10-20 cm.
    "realistic": dict(zero_sigma_deg=0.15, link_sigma_mm=0.3, base_sigma_mm=1.0, rail_zero_sigma_mm=0.5,
                      backlash_deg=0.1, latency_s=0.02, enc_noise_deg=0.01, grip_noise_mm=0.3,
                      wrist_sigma_mm=0.5, handeye_sigma_mm=0.5),
    # stress test: roughly 3x the realistic numbers
    "harsh": dict(zero_sigma_deg=0.4, link_sigma_mm=1.0, base_sigma_mm=3.0, rail_zero_sigma_mm=1.5,
                  backlash_deg=0.3, latency_s=0.05, enc_noise_deg=0.03, grip_noise_mm=0.8,
                  wrist_sigma_mm=1.5, handeye_sigma_mm=1.5),
}

CTRL_DT = 0.01          # must match plan.motion.CTRL_DT


def preset(p):
    return dict(PRESETS[p]) if isinstance(p, str) else dict(p)


class Actuation:
    """Hardware layer between the robot's commands (in its own calibration frame) and the simulated servos.
    `cmd_q`, `cmd_x` are what the robot commands; `tick()` writes the true servo targets into `w.d.ctrl`."""

    def __init__(self, world, p="ideal", rng=None):
        self.w = world
        self.p = preset(p)
        rng = np.random.default_rng(0) if rng is None else rng
        self.rng = rng
        self.zero = np.radians(rng.normal(0.0, self.p["zero_sigma_deg"], 6)) if self.p["zero_sigma_deg"] else np.zeros(6)
        self.rail_zero = rng.normal(0.0, self.p["rail_zero_sigma_mm"]) / 1000 if self.p["rail_zero_sigma_mm"] else 0.0
        self.play = math.radians(self.p["backlash_deg"]) / 2
        self.lag = int(round(self.p["latency_s"] / CTRL_DT))
        self.enc_sigma = math.radians(self.p["enc_noise_deg"])
        d = world.d
        # the robot believes it is where it commands: initialise from the current servo targets
        self.cmd_q = d.ctrl[world.act_arm].copy() - self.zero
        self.cmd_x = float(d.ctrl[world.act_rail]) - self.rail_zero
        self.motor = self.cmd_q + self.zero          # motor-side angle (what the servo loop drives)
        self.out = self.motor.copy()                 # joint-side angle after the backlash
        self.fifo = collections.deque()

    def tick(self):
        """Advance one control period: latency FIFO -> zero offset -> backlash -> servo targets."""
        self.fifo.append((self.cmd_q.copy(), self.cmd_x))
        if len(self.fifo) <= self.lag:               # latency: nothing new has arrived yet, servos hold
            return
        q, x = self.fifo.popleft()
        self.motor = q + self.zero
        if self.play > 0:
            e = self.motor - self.out
            self.out = np.where(e > self.play, self.motor - self.play, np.where(e < -self.play, self.motor + self.play,
                                                                                 self.out))
        else:
            self.out = self.motor.copy()
        self.w.d.ctrl[self.w.act_arm] = self.out
        self.w.d.ctrl[self.w.act_rail] = x + self.rail_zero

    def encoders(self):
        """Joint encoders in the robot's frame: true joint + backlash gap state - zero offset + noise."""
        q = self.w.d.qpos[self.w.arm_q] + (self.motor - self.out) - self.zero
        if self.enc_sigma:
            q = q + self.rng.normal(0.0, self.enc_sigma, 6)
        return q

    def rail_encoder(self):
        return float(self.w.d.qpos[self.w.rail_q]) - self.rail_zero

    def gap_mm(self):
        g = 2000.0 * float(self.w.d.qpos[self.w.finger_q])
        return g + (self.rng.normal(0.0, self.p["grip_noise_mm"]) if self.p["grip_noise_mm"] else 0.0)


def perturb_model(ik, p="ideal", rng=None):
    """Give the planning model (the robot's model of itself) calibration errors: link offsets and the arm base."""
    p = preset(p)
    rng = np.random.default_rng(1) if rng is None else rng
    m = ik.m
    if p["base_sigma_mm"]:      # registration of the whole robot (carriage, arm, caddy) to the printers
        m.body_pos[m.body("carriage").id] += rng.normal(0.0, p["base_sigma_mm"] / 1000, 3)
    if p["link_sigma_mm"]:
        for k in range(1, 7):
            b = m.body(f"link_{k}").id
            m.body_pos[b] += rng.normal(0.0, p["link_sigma_mm"] / 1000, 3)
    ik.model_error = p


def setup(ex, ik, p="ideal", rng=None):
    """Switch an Executor + IK pair to an actuation-realism preset. 'ideal' leaves both untouched (legacy path)."""
    p = preset(p)
    rng = np.random.default_rng(0) if rng is None else rng
    ex.realism = p
    if any(p[k] for k in p):
        ex.hw = Actuation(ex.w, p, rng)
        perturb_model(ik, p, rng)
    ex.wrist = WristCam(ex.w, sigma_mm=p["wrist_sigma_mm"], handeye_mm=p["handeye_sigma_mm"], rng=rng)
    return ex


def fk_pinch(ik, q, x):
    """Pinch pose (position, rotation) the robot believes from joint readings (its own kinematic model)."""
    return ik.fk(q, x)


# ---------------- wrist depth camera ----------------
MOUNT_POS = np.array([0.0, -0.07, 0.02])       # in link_6: the scene's 'wrist' camera mount point
PINCH_IN_L6 = np.array([0.0, -0.044, 0.1367])  # the pinch site in link_6 (farm/scene.py)
NEAR_M = 0.02                                   # wrist camera near plane (m)
COMMISSION_SCANS = 5                            # empty-jaw reference: scans averaged
HANDEYE_BAND = 0.006                            # ... a pad face is looked for within this of its nominal place (m)


class WristCam:
    """Depth camera rigidly mounted on link_6 at MOUNT_POS, aimed at the pinch point (hand-eye calibrated).
    Rendered as a MuJoCo free camera placed from link_6's pose every frame (no scene change needed).
    Points are returned in the PINCH frame (x, y = jaw closing axis, z = approach), which needs only the
    hand-eye calibration, not the arm's kinematics. Uses the default free-camera field of view."""

    def __init__(self, world, width=320, height=240, sigma_mm=0.0, handeye_mm=0.0, rng=None, self_reference=True):
        self.w = world
        self.W, self.H = width, height
        self.r = None
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 0
        opt.geomgroup[[0, 1, 2]] = 1
        self.opt = opt
        rng = np.random.default_rng(2) if rng is None else rng
        self.rng = rng
        self.sigma = sigma_mm / 1000
        self.handeye = rng.normal(0.0, handeye_mm / 1000, 3) if handeye_mm else np.zeros(3)
        self.renders = 0
        self.ref = None                  # empty-jaw reference (commission), taken at the first between-jaws check
        self.self_reference = self_reference

    def _renderer(self):
        if self.r is None:
            self.r = mujoco.Renderer(self.w.m, height=self.H, width=self.W)
            self.r.enable_depth_rendering()
        return self.r

    def points_pinch(self, max_range=0.45):
        """Depth image -> points (m) in the pinch frame."""
        mujoco.mj_kinematics(self.w.m, self.w.d)
        return self._points(self.w.d, max_range)

    def _points(self, d, max_range=0.45, only=None, rng=None):
        """Render from state d (kinematics computed). only: body ids that are drawn (None: the whole scene); rng: the
        depth-noise stream (default: the camera's own)."""
        m = self.w.m
        b = self.w.link6
        R6, p6 = d.xmat[b].reshape(3, 3), d.xpos[b]
        cpos = p6 + R6 @ MOUNT_POS
        look = p6 + R6 @ PINCH_IN_L6
        f = look - cpos
        dist = float(np.linalg.norm(f))
        f /= dist
        self.cam.lookat[:] = look
        self.cam.distance = dist
        self.cam.azimuth = math.degrees(math.atan2(f[1], f[0]))
        self.cam.elevation = math.degrees(math.asin(np.clip(f[2], -1, 1)))
        r = self._renderer()
        znear = m.vis.map.znear                      # the scene's near plane (~13 cm) would clip the jaws away:
        m.vis.map.znear = NEAR_M / max(m.stat.extent, 1e-6)    # a short-range depth camera sees from ~2 cm
        try:
            r.update_scene(d, camera=self.cam, scene_option=self.opt)
            if only is not None:                     # everything else is moved out of the camera's range
                scn = r.scene
                for k in range(scn.ngeom):
                    gk = scn.geoms[k]
                    if gk.objtype != mujoco.mjtObj.mjOBJ_GEOM or m.geom_bodyid[gk.objid] not in only:
                        gk.pos[:] = (0.0, 0.0, -100.0)
            depth = r.render().astype(np.float64)
        finally:
            m.vis.map.znear = znear
        self.renders += 1
        gl = r.scene.camera
        fwd, up = np.array(gl[0].forward), np.array(gl[0].up)
        right = np.cross(fwd, up)
        pos = 0.5 * (np.array(gl[0].pos) + np.array(gl[1].pos))     # the two eyes sit +-ipd/2 off the head
        fpx = 0.5 * self.H / math.tan(math.radians(m.vis.global_.fovy) / 2)
        v, u = np.mgrid[0:self.H, 0:self.W]
        z = depth
        if self.sigma:
            z = z + (self.rng if rng is None else rng).normal(0.0, self.sigma, z.shape)
        ok = (depth > 0.01) & (depth < max_range)
        xc = (u - self.W / 2 + 0.5) * z / fpx
        yc = -(v - self.H / 2 + 0.5) * z / fpx
        pw = pos + z[..., None] * fwd + xc[..., None] * right + yc[..., None] * up
        pw = pw[ok]
        sid = self.w.grasp_site                     # camera -> pinch is a fixed, calibrated transform
        Rp, pp = d.site_xmat[sid].reshape(3, 3), d.site_xpos[sid]
        return (pw - pp) @ Rp + self.handeye

    def commission(self, gap_mm=None, scans=COMMISSION_SCANS, half_len=0.012, half_w=0.010):
        """Reference scan of the empty open jaws (once, at installation: the jaws fully open, `gap_mm` from the
        averaged gripper encoder, nothing but the gripper in view; the sim renders a copy of the state with only the
        gripper drawn, so it does not matter what the live jaws hold). Measures where the inner face of each pad
        appears in this camera's own pinch frame: the between-the-jaws test then runs against the measured jaws, not
        the nominal ones, which cancels the hand-eye error along the jaw axis (and pad wear, a bent finger).
        Returns the reference (gap_mm, +y face, -y face) or None if the pads were not seen."""
        from farm.world import GRIP_OPEN
        gap_mm = 2000 * GRIP_OPEN if gap_mm is None else gap_mm
        m = self.w.m
        d = mujoco.MjData(m)
        d.qpos[:] = self.w.d.qpos
        g = gap_mm / 2000
        d.qpos[self.w.finger_q], d.qpos[self.w.finger_rq] = g, -g
        mujoco.mj_kinematics(m, d)
        grip = self.w.finger_bodies | {self.w.link6}
        rng = np.random.default_rng(7)               # own noise stream: the episode's stream stays as it was
        pts = np.vstack([self._points(d, only=grip, rng=rng) for _ in range(scans)])
        win = (np.abs(pts[:, 0]) < half_w) & (np.abs(pts[:, 2]) < half_len)
        faces = []
        for side in (1.0, -1.0):
            y = side * pts[win, 1]
            y = y[np.abs(y - g) < HANDEYE_BAND]     # the pad's inner face, within the hand-eye tolerance of nominal
            if len(y) < 50 * scans:
                self.ref = None
                return None
            faces.append(side * float(np.percentile(y, 10)))   # the innermost surface, robust to noise
        self.ref = (float(gap_mm), faces[0], faces[1])
        return self.ref

    def between_jaws(self, gap_mm, pts=None, half_len=0.012, half_w=0.010, inset=0.003):
        """Number of depth points inside the jaw volume (between the pads, `inset` in from each): something is held /
        stuck. The pads are where the empty-jaw reference saw them (commission), moved by the measured gap change;
        without a reference (no rendering), at their nominal place."""
        p = self.points_pinch() if pts is None else pts
        if self.ref is None and self.self_reference:
            self.self_reference = False             # (taken once; if the pads were not seen: nominal jaws)
            try:
                self.commission()
            except Exception:                       # no rendering for the reference: nominal jaws
                pass
        if self.ref is not None:
            g0, fp, fn = self.ref
            dg = (gap_mm - g0) / 2000
            hi, lo = fp + dg - inset, fn - dg + inset
            if hi - lo < 0.004:
                hi, lo = (hi + lo) / 2 + 0.002, (hi + lo) / 2 - 0.002
        else:
            hi = max(gap_mm / 2000 - inset, 0.002)
            lo = -hi
        sel = (np.abs(p[:, 0]) < half_w) & (p[:, 1] < hi) & (p[:, 1] > lo) & (np.abs(p[:, 2]) < half_len)
        return int(sel.sum())

    def close(self):
        if self.r is not None:
            self.r.close()
            self.r = None
