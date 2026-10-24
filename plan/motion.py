"""Execute joint/Cartesian motions on the simulated robot (position actuators), with frame recording."""
import numpy as np

from farm import catalog as C
from farm.world import GRIP_CLOSED, GRIP_OPEN

ARM_VMAX = 1.1        # rad/s per joint (cruise)
RAIL_VMAX = 0.40      # m/s
PAYLOAD_SCALE = 0.5
PRESHAPE = 0.008
CTRL_DT = 0.01        # control period (5 physics steps)


def smooth(s):
    return s * s * s * (10 - 15 * s + 6 * s * s)     # quintic, zero vel/acc at the ends


class Executor:
    def __init__(self, world, recorder=None, log=None, realism=None, ik=None, rng=None):
        self.w = world
        self.rec = recorder
        self.log = log if log is not None else []
        self.steps_per_ctrl = int(round(CTRL_DT / world.m.opt.timestep))
        self.payload = False         # holding a part: move at PAYLOAD_SCALE of the cruise speeds
        self.hw = None               # farm.sensors.Actuation (realism preset); None = ideal servos (legacy path)
        self.wrist = None            # farm.sensors.WristCam
        self.realism = None
        if realism is not None:
            from farm import sensors
            sensors.setup(self, ik, realism, rng)

    @property
    def q(self):
        """Commanded arm joints, in the robot's own calibration frame."""
        return self.hw.cmd_q.copy() if self.hw else self.w.d.ctrl[self.w.act_arm].copy()

    @property
    def x(self):
        return float(self.hw.cmd_x) if self.hw else float(self.w.d.ctrl[self.w.act_rail])

    def _cmd(self, q, x):
        if self.hw:
            self.hw.cmd_q[:] = q
            self.hw.cmd_x = float(x)
        else:
            self.w.d.ctrl[self.w.act_arm] = q
            self.w.d.ctrl[self.w.act_rail] = x

    def q_meas(self):
        """Joint encoder reading (robot frame)."""
        return self.hw.encoders() if self.hw else self.w.d.qpos[self.w.arm_q].copy()

    def x_meas(self):
        return self.hw.rail_encoder() if self.hw else float(self.w.d.qpos[self.w.rail_q])

    def _tick(self):
        if self.hw:
            self.hw.tick()
        self.w.step(self.steps_per_ctrl)
        for hook in getattr(self.w, "hooks", ()):
            hook(self.w)
        if self.rec:
            self.rec(self.w)

    def wait(self, seconds):
        for _ in range(int(seconds / CTRL_DT)):
            self._tick()

    def move(self, q=None, x=None, duration=None, label=""):
        q0, x0 = self.q, self.x
        q1 = q0 if q is None else np.asarray(q, float)
        x1 = x0 if x is None else float(x)
        if duration is None:
            k = PAYLOAD_SCALE if self.payload else 1.0
            duration = max(np.max(np.abs(q1 - q0)) / (ARM_VMAX * k) * 1.6, abs(x1 - x0) / (RAIL_VMAX * k) * 1.6, 0.3)
        n = max(int(duration / CTRL_DT), 1)
        for k in range(1, n + 1):
            s = smooth(k / n)
            self._cmd(q0 + s * (q1 - q0), x0 + s * (x1 - x0))
            self._tick()
        self.log.append((label or "move", duration))
        return duration

    def follow(self, qs, xs=None, seg_time=None, label=""):
        """Follow a dense list of joint configurations (e.g. from Cartesian IK), constant-speed segments."""
        total = 0.0
        x = self.x
        for k, q in enumerate(qs):
            xk = x if xs is None else xs[k]
            dq = np.max(np.abs(np.asarray(q) - self.q))
            k = PAYLOAD_SCALE if self.payload else 1.0
            t = seg_time or max(dq / (ARM_VMAX * 0.5 * k), abs(xk - self.x) / (RAIL_VMAX * 0.5 * k), CTRL_DT)
            n = max(int(t / CTRL_DT), 1)
            q0, x0 = self.q, self.x
            for j in range(1, n + 1):
                s = j / n
                self._cmd(q0 + s * (np.asarray(q) - q0), x0 + s * (xk - x0))
                self._tick()
            total += n * CTRL_DT
        self.log.append((label or "follow", total))
        return total

    def move_path(self, qs, x=None, label=""):
        """Follow planner waypoints with smooth per-segment timing (rail held or moved on the first segment)."""
        total = 0.0
        for k, q in enumerate(qs[1:]):
            total += self.move(q, x if k == 0 else None, label=label)
        return total

    def settle(self, tol=0.01, timeout=1.0):
        """Wait until joints track their targets."""
        t = 0.0
        while t < timeout:
            err = np.max(np.abs(self.q_meas() - self.q))           # encoders vs command (what the robot sees)
            if err < tol:
                break
            self._tick()
            t += CTRL_DT
        return t

    def grip(self, close, duration=0.6, label="", width=None):
        """Close fully, or to 5 mm per jaw under `width` (m): ~10 N squeeze, force-limited at 20 N.
        Open fully, or pre-shape to width/2 + PRESHAPE per jaw so open jaws don't sweep into neighbours."""
        if close:
            target = GRIP_CLOSED if width is None else max(width / 2 - 0.005, 0.0)
        else:                                   # open fully, or pre-shape to the part width + 8 mm per jaw
            target = GRIP_OPEN if width is None else min(GRIP_OPEN, width / 2 + PRESHAPE)
        return self.jaws(target, duration, label or ("close" if close else "open"))

    def jaws(self, target, duration=0.4, label="jaws"):
        """Drive each jaw to `target` (m from the centre line)."""
        g0 = float(self.w.d.ctrl[self.w.act_grip])
        n = int(duration / CTRL_DT)
        for k in range(1, n + 1):
            self.w.d.ctrl[self.w.act_grip] = g0 + smooth(k / n) * (target - g0)
            self._tick()
        self.log.append((label, duration))
        return duration

    def finger_gap_mm(self):
        """Jaw gap from the gripper motor encoder (mm)."""
        return self.hw.gap_mm() if self.hw else 2000 * float(self.w.d.qpos[self.w.finger_q])


def cartesian_line(ik, p0, p1, R, x, q_seed, n=12, symmetric=True):
    """IK along a straight line p0->p1 at fixed orientation and rail position. Returns list of q or None."""
    qs, q = [], np.asarray(q_seed, float)
    for s in np.linspace(0, 1, n + 1)[1:]:
        q, _, ok, _ = ik.solve(p0 + s * (p1 - p0), R, x, q0=q, iters=80, symmetric=symmetric)
        if not ok:
            return None
        qs.append(q.copy())
    return qs
