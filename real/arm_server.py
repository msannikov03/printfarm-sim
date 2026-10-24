"""SO-101 arm server for the Raspberry Pi: Feetech bus + smooth motion + safety, JSON lines over TCP.

On the Pi (the arm's USB adapter plugged into it):
    python3 -m real.arm_server --port /dev/ttyACM0                 # listens on the Tailscale IP, TCP 8765
    python3 -m real.arm_server --mock --listen 127.0.0.1:8765       # rehearsal without hardware (any machine)
On the Mac:
    python -m real.arm_client remote://<pi>:8765 status
    python -m real.run_demo ... --live --arm remote://<pi>:8765

Only the Python standard library + pyserial (no numpy, no MuJoCo, no LeRobot): the joint calibration is
real/joint_calib.json (real/joint_calib.py), the forward kinematics for the keep-out checks is
real/kin_lite.py with the chain exported from the MuJoCo model (real/so101_chain.json).

Motion: every move is a minimum-jerk joint-space trajectory (s = 10t^3 - 15t^4 + 6t^5) whose duration
respects the speed AND acceleration limits of every joint; it is streamed to the servos at 50 Hz with one
SYNC WRITE per tick, while the present positions are read back (SYNC READ) for stall detection.
The servos' own goal-speed / acceleration registers are set as a hardware cap underneath.
Safety:
  - targets outside the model joint range or the calibrated servo range are refused (not clamped);
  - every trajectory is sampled and refused if a check point (both jaw tips, gripper, wrist, elbow) leaves
    the workspace box, goes below the floor, or comes within the bed margin over the bed footprint
    (the bed pose comes from the client: set_keepout);
  - stall / collision: tracking error > stall_deg for stall_s -> stop and hold;
  - watchdog: the controlling client must send something every heartbeat_s (the client library pings
    4x per second); if it vanishes (crash, Wi-Fi, tailnet) the move stops and the arm HOLDS its pose;
  - estop (from ANY connected client): torque off immediately (the arm drops!), moves refused until
    torque on;
  - one controlling client at a time; others are read-only (status, read, stop, estop) unless they take over.
Protocol: one JSON object per line. Request {"id": 7, "cmd": "move", ...}; reply {"id": 7, "ok": true, ...}
or {"id": 7, "ok": false, "error": "..."}.  Commands: hello, ping, status, read, read_raw, move, wait,
gripper, stop, estop, torque, set_keepout, get_keepout, calib, set_calib, calib_from_servos.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import socketserver
import subprocess
import threading
import time
from pathlib import Path

from real.feetech import CommError, FeetechBus, MockServo, MockSerial
from real.joint_calib import JOINT_CALIB_JSON, JOINTS, CalibError, JointCalib, from_servos, nominal
from real.kin_lite import Chain

log = logging.getLogger("real.arm_server")
VERSION = "1"
DEFAULT_TCP_PORT = 8765
TICK_DEG = 360.0 / 4096


class MotionError(RuntimeError):
    pass


def minjerk(s: float) -> float:
    s = min(1.0, max(0.0, s))
    return s * s * s * (10 - 15 * s + 6 * s * s)


class Limits:
    def __init__(self, max_speed_deg_s=30.0, max_accel_deg_s2=60.0, gripper_speed_deg_s=90.0, rate_hz=50.0,
                 min_duration_s=0.4, stall_deg=15.0, stall_s=0.35, settle_tol_deg=1.5, settle_s=1.5,
                 servo_speed_cap_deg_s=120.0, servo_acc=40, gripper_torque_pct=50.0, max_comm_fail=10):
        self.__dict__.update({k: v for k, v in locals().items() if k != "self"})

    def to_json(self):
        return dict(self.__dict__)


class KeepOut:
    """Forbidden space for the check points (robot base frame, metres)."""

    def __init__(self, box_min=(-0.15, -0.40, -0.02), box_max=(0.50, 0.40, 0.55), bed=None):
        self.box_min, self.box_max, self.bed = list(box_min), list(box_max), bed

    def set_bed(self, R, t, size_mm, margin_mm=8.0, apron_mm=20.0):
        """Bed pose from the bed->robot calibration: p_robot = R @ p_bed_mm / 1000 + t."""
        self.bed = dict(R=[list(map(float, r)) for r in R], t=list(map(float, t)), size_mm=list(map(float, size_mm)),
                        margin_mm=float(margin_mm), apron_mm=float(apron_mm))

    def depth(self, pts: dict) -> tuple[float, str]:
        """Largest penetration (m) into forbidden space over the check points, and what it was."""
        worst, what = 0.0, ""
        for name, p in pts.items():
            for k in range(3):
                d = max(self.box_min[k] - p[k], p[k] - self.box_max[k])
                if d > worst:
                    worst, what = d, f"{name} {'xyz'[k]}={p[k] * 1000:.0f} mm outside the workspace box"
            if self.bed is not None and name in ("fixed_tip", "moving_tip", "gripper", "wrist"):
                R, t, b = self.bed["R"], self.bed["t"], self.bed
                v = [p[i] - t[i] for i in range(3)]
                pb = [1000.0 * sum(R[i][k] * v[i] for i in range(3)) for k in range(3)]     # R^T (p - t)
                a = b["apron_mm"]
                if -a <= pb[0] <= b["size_mm"][0] + a and -a <= pb[1] <= b["size_mm"][1] + a:
                    d = (b["margin_mm"] - pb[2]) / 1000.0
                    if d > worst:
                        worst, what = d, (f"{name} {pb[2]:.0f} mm above the bed at ({pb[0]:.0f}, {pb[1]:.0f}) "
                                          f"(margin {b['margin_mm']:.0f} mm)")
        return worst, what

    def to_json(self):
        return dict(box_min=self.box_min, box_max=self.box_max, bed=self.bed)


class ArmController:
    """Owns the bus. move() starts a background trajectory; everything else is quick."""

    def __init__(self, bus: FeetechBus, calib: JointCalib, chain: Chain | None = None, limits: Limits | None = None,
                 keepout: KeepOut | None = None, heartbeat_s: float | None = 2.0, allow_nominal: bool = False,
                 clock=time.monotonic, sleep=time.sleep, calib_path=JOINT_CALIB_JSON):
        self.bus, self.calib = bus, calib
        self.chain = chain or Chain()
        self.lim = limits or Limits()
        self.keepout = keepout or KeepOut()
        self.heartbeat_s, self.allow_nominal = heartbeat_s, allow_nominal
        self.clock, self.sleep, self.calib_path = clock, sleep, calib_path
        self.ids = calib.ids()
        self.found: dict = {}
        self.estopped = False
        self.last_hb = clock()
        self._thread: threading.Thread | None = None
        self._abort = threading.Event()
        self._abort_reason = ""
        self.result: dict = {"state": "idle"}
        self.q_cmd: list | None = None
        self.lock = threading.RLock()
        self.events: list[str] = []

    # ------------------------------------------------------------------ setup
    def _event(self, msg):
        log.info(msg)
        self.events = (self.events + [time.strftime("%H:%M:%S ") + msg])[-30:]

    def connect(self):
        self.found = self.bus.scan(self.ids)
        missing = [i for i in self.ids if i not in self.found]
        if missing:
            raise MotionError(f"servo ids {missing} do not answer (found {sorted(self.found)}): check the 3-pin "
                              f"daisy chain, the servo power supply (the USB cable does not power them), and --baud")
        self.configure()
        h = self.bus.read_health(self.ids)
        v = [h[i]["volts"] for i in h]
        self._event(f"connected: ids {sorted(self.found)}, supply {min(v):.1f}-{max(v):.1f} V, calib {self.calib.method}")
        return self

    def configure(self):
        """SRAM only: hardware speed/acceleration caps and the gripper torque limit (LeRobot uses 50 %)."""
        self.bus.set_profile(self.ids, speed=int(self.lim.servo_speed_cap_deg_s / TICK_DEG), acc=self.lim.servo_acc)
        g = self.calib.joints["gripper"]["id"]
        self.bus.sync_write_reg("Torque_Limit", {g: int(self.lim.gripper_torque_pct * 10)})

    def close(self):
        self.stop("server shutting down")
        self.bus.close()

    # ------------------------------------------------------------------ state
    def heartbeat(self):
        self.last_hb = self.clock()

    def read_raw(self) -> dict:
        r = self.bus.read_positions(self.ids)
        if len(r) < len(self.ids):
            raise CommError(f"no position from ids {[i for i in self.ids if i not in r]}")
        return {j: r[self.calib.joints[j]["id"]] for j in JOINTS}

    def read_q(self) -> list:
        return self.calib.raw_to_model(self.read_raw())

    def torque_state(self) -> dict:
        return self.bus.sync_read_reg("Torque_Enable", self.ids)

    def status(self) -> dict:
        health = self.bus.read_health(self.ids)
        q = None
        if all(self.calib.joints[j]["id"] in health for j in JOINTS):
            q = self.calib.raw_to_model({j: health[self.calib.joints[j]["id"]]["pos"] for j in JOINTS})
        tq = self.torque_state()
        return dict(q=q, q_deg=None if q is None else [round(math.degrees(v), 2) for v in q],
                    tip_mm=None if q is None else [round(v * 1000, 1) for v in self.chain.point(q)],
                    torque={j: tq.get(self.calib.joints[j]["id"]) for j in JOINTS}, estopped=self.estopped,
                    moving=self.moving(), result=self.result, calib=self.calib.method,
                    health={j: health.get(self.calib.joints[j]["id"]) for j in JOINTS},
                    bus=dict(self.bus.stats), events=self.events[-8:])

    def moving(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------ planning
    def joint_limits(self):
        lo_m, hi_m = self.chain.lo_hi()
        lo_c, hi_c = self.calib.model_limits()
        return [max(a, b) for a, b in zip(lo_m, lo_c)], [min(a, b) for a, b in zip(hi_m, hi_c)]

    def plan(self, q0, q1, speed_deg_s=None, accel_deg_s2=None, samples_per_rad=60) -> tuple[float, list]:
        """(duration, target) of the min-jerk move q0 -> q1 within the limits; raises MotionError if unsafe.
        A target within 3 ticks (0.26 deg) of a limit is clamped onto it (tick rounding of the calibration)."""
        if len(q1) != 6 or any(not math.isfinite(v) for v in q1):
            raise MotionError(f"target must be 6 finite joint angles (rad), got {q1}")
        lo, hi = self.joint_limits()
        tol = math.radians(3 * TICK_DEG)
        bad = [f"{j} {math.degrees(q1[k]):.1f} deg not in [{math.degrees(lo[k]):.1f}, {math.degrees(hi[k]):.1f}]"
               for k, j in enumerate(JOINTS) if not lo[k] - tol <= q1[k] <= hi[k] + tol]
        if bad:
            raise MotionError("target outside joint limits: " + "; ".join(bad))
        q1 = [min(hi[k], max(lo[k], q1[k])) for k in range(6)]
        v = min(float(speed_deg_s or self.lim.max_speed_deg_s), self.lim.max_speed_deg_s)
        a = min(float(accel_deg_s2 or self.lim.max_accel_deg_s2), self.lim.max_accel_deg_s2)
        T = self.lim.min_duration_s
        for k in range(6):
            d = abs(math.degrees(q1[k] - q0[k]))
            vk = self.lim.gripper_speed_deg_s if k == 5 else v
            T = max(T, 1.875 * d / vk, math.sqrt(5.7735 * d / a) if k < 5 else 0.0)
        # keep-out along the path (the start may already be inside: then it must not get any deeper)
        d0, _ = self.keepout.depth(self.chain.check_points(q0))
        span = max(abs(q1[k] - q0[k]) for k in range(5))
        n = max(4, int(span * samples_per_rad))
        for i in range(1, n + 1):
            s = minjerk(i / n)
            q = [q0[k] + (q1[k] - q0[k]) * s for k in range(6)]
            d, what = self.keepout.depth(self.chain.check_points(q))
            if (i == n and d > 0) or d > d0 + 1e-4:
                raise MotionError(f"keep-out at {100 * i / n:.0f} % of the move: {what}")
        return T, q1

    # ------------------------------------------------------------------ motion
    def _check_ready(self):
        if self.estopped:
            raise MotionError("E-STOPPED: send torque on (the arm holds where it is) before moving again")
        if self.calib.method == "nominal" and not self.allow_nominal:
            raise MotionError("joint calibration is NOMINAL (not measured): run python -m real.joint_calib "
                              "from-servos | from-lerobot | guided, or start the server with --allow-nominal")

    def move(self, q1, speed_deg_s=None, accel_deg_s2=None, wait=False, timeout=None) -> dict:
        with self.lock:
            self._check_ready()
            if self.moving():
                raise MotionError("already moving (send stop first)")
            q0 = self.read_q()
            q1 = [float(v) for v in q1]
            if len(q1) == 5:
                q1.append(q0[5])
            T, q1 = self.plan(q0, q1, speed_deg_s, accel_deg_s2)
            if any(v != 1 for v in self.torque_state().values()):
                self.torque(True)
            self._abort.clear()
            self.result = dict(state="moving", duration_s=round(T, 3), target_deg=[round(math.degrees(v), 2) for v in q1])
            self.heartbeat()
            self._thread = threading.Thread(target=self._run, args=(q0, q1, T), daemon=True)
            self._thread.start()
        if wait:
            return self.wait(timeout)
        return dict(self.result)

    def gripper(self, open_pct=None, rad=None, speed_deg_s=None, wait=False) -> dict:
        """Gripper only: rad (model angle) or open_pct of the calibrated range (0 = closed)."""
        q = self.read_q()
        if rad is not None:
            q[5] = float(rad)
        else:
            lo, hi = self.joint_limits()
            q[5] = lo[5] + (hi[5] - lo[5]) * min(100.0, max(0.0, float(open_pct))) / 100.0
        return self.move(q, speed_deg_s, wait=wait)

    def wait(self, timeout=None) -> dict:
        t = self._thread
        if t is not None:
            t.join(timeout)
        return dict(self.result)

    def _ids_goal(self, q) -> dict:
        raw = self.calib.model_to_raw(q, clamp=True)
        return {self.calib.joints[j]["id"]: raw[j] for j in raw}

    def _run(self, q0, q1, T):
        lim, dt = self.lim, 1.0 / self.lim.rate_hz
        t0 = self.clock()
        stall_since, fails, reason, err_max = None, 0, "", 0.0
        tick = 0
        try:
            while True:
                now = self.clock()
                if self._abort.is_set():
                    reason = self._abort_reason or "stopped"
                    break
                if self.heartbeat_s and now - self.last_hb > self.heartbeat_s:
                    reason = f"watchdog: no message from the client for {now - self.last_hb:.1f} s"
                    break
                t = now - t0
                s = minjerk(t / T)
                q = [q0[k] + (q1[k] - q0[k]) * s for k in range(6)]
                self.q_cmd = q
                self.bus.write_goals(self._ids_goal(q))
                try:
                    meas = self.calib.raw_to_model(self.read_raw())
                    fails = 0
                    err = max(abs(q[k] - meas[k]) for k in range(5))
                    err_max = max(err_max, err)
                    if err > math.radians(lim.stall_deg):
                        stall_since = stall_since if stall_since is not None else now
                        if now - stall_since > lim.stall_s:
                            reason = (f"stall/collision: tracking error {math.degrees(err):.1f} deg for "
                                      f"{now - stall_since:.2f} s")
                            break
                    else:
                        stall_since = None
                except CommError as e:
                    fails += 1
                    if fails > lim.max_comm_fail:
                        reason = f"bus error: {e}"
                        break
                if t >= T:
                    break
                tick += 1
                self.sleep(max(0.0, t0 + tick * dt - self.clock()))
            if reason:
                self.hold()
                self.result = dict(state="aborted", reason=reason, at_deg=[round(math.degrees(v), 1) for v in self.read_q()])
                self._event("move aborted: " + reason)
                return
            # settle on the final goal (gripper excluded: it may be squeezing a part)
            t_end = self.clock() + lim.settle_s
            err = [0.0] * 6
            while True:
                meas = self.calib.raw_to_model(self.read_raw())
                err = [abs(math.degrees(q1[k] - meas[k])) for k in range(6)]
                if max(err[:5]) <= lim.settle_tol_deg or self.clock() > t_end:
                    break
                if self._abort.is_set() or (self.heartbeat_s and self.clock() - self.last_hb > self.heartbeat_s):
                    self.hold()
                    break
                self.bus.write_goals(self._ids_goal(q1))
                self.sleep(dt)
            self.result = dict(state="done", duration_s=round(self.clock() - t0, 3), reached=max(err[:5]) <= lim.settle_tol_deg,
                               err_deg=[round(e, 2) for e in err], max_track_err_deg=round(math.degrees(err_max), 2),
                               q=meas)
        except Exception as e:                       # never leave the arm chasing a stale goal
            log.exception("motion thread")
            try:
                self.hold()
            except Exception:
                pass
            self.result = dict(state="aborted", reason=f"error: {e}")

    def hold(self):
        """Goal := present position on every joint (the arm stays where it is, torque unchanged)."""
        r = self.bus.read_positions(self.ids)
        if r:
            self.bus.write_goals(r)

    def stop(self, reason="stop requested") -> dict:
        self._abort_reason = reason
        self._abort.set()
        if self._thread is not None:
            self._thread.join(2.0)
        else:
            try:
                self.hold()
            except CommError:
                pass
        return dict(self.result)

    def estop(self) -> dict:
        self._abort_reason = "E-STOP"
        self._abort.set()
        st = self.bus.torque(self.ids, False)
        self.estopped = True
        self._event(f"E-STOP: torque off {st}")
        if self._thread is not None:
            self._thread.join(2.0)
        return dict(torque=st)

    def torque(self, on: bool) -> dict:
        if on:
            self.hold()                     # goal := present, or the arm jumps to a stale goal
            self.configure()
        st = self.bus.torque(self.ids, on)
        if on:
            self.estopped = False
        self._event(f"torque {'on' if on else 'off'}: {st}")
        return st

    def set_keepout(self, bed=None, box_min=None, box_max=None) -> dict:
        if box_min is not None:
            self.keepout.box_min, self.keepout.box_max = list(box_min), list(box_max)
        if bed is None:
            self.keepout.bed = None
        else:
            self.keepout.set_bed(bed["R"], bed["t"], bed["size_mm"], bed.get("margin_mm", 8.0), bed.get("apron_mm", 20.0))
        return self.keepout.to_json()

    def set_calib(self, d: dict, save=True):
        if self.moving():
            raise MotionError("cannot change the calibration while moving")
        c = JointCalib.from_dict(d)
        self.calib, self.ids = c, c.ids()
        if save:
            c.save(self.calib_path)
        self._event(f"joint calibration replaced: {c.method}")
        return c


# ============================================================================ TCP server

class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        srv: ArmServer = self.server.app
        cid = srv.new_client(self.client_address)
        authed = srv.token is None
        try:
            for line in self.rfile:
                try:
                    req = json.loads(line)
                except json.JSONDecodeError:
                    self._send({"ok": False, "error": "bad json"})
                    continue
                rid, cmd = req.get("id"), req.get("cmd")
                if not authed:
                    if cmd == "hello" and req.get("token") == srv.token:
                        authed = True
                    else:
                        self._send({"id": rid, "ok": False, "error": "unauthorised: hello with the right token first"})
                        continue
                try:
                    reply = {"id": rid, "ok": True, **srv.dispatch(cid, cmd, req)}
                except Exception as e:          # report every failure to the client, keep serving
                    if not isinstance(e, (MotionError, CalibError, CommError, ValueError, KeyError, TypeError)):
                        log.exception("command %s", cmd)
                    reply = {"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}"}
                self._send(reply)
        except (ConnectionError, OSError):
            pass
        finally:
            srv.client_gone(cid)

    def _send(self, d):
        self.wfile.write((json.dumps(d, default=float) + "\n").encode())
        self.wfile.flush()


class _TCP(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


class ArmServer:
    READ_ONLY = {"hello", "ping", "status", "read", "read_raw", "stop", "estop", "calib", "get_keepout", "wait"}

    def __init__(self, ctrl: ArmController, host="127.0.0.1", port=DEFAULT_TCP_PORT, token=None, mock=False):
        self.ctrl, self.host, self.port, self.token, self.mock = ctrl, host, port, token, mock
        self.owner = None
        self._n = 0
        self.lock = threading.Lock()
        self.tcp = None

    def new_client(self, addr):
        with self.lock:
            self._n += 1
            cid = self._n
        self.ctrl._event(f"client {cid} connected from {addr[0]}")
        return cid

    def client_gone(self, cid):
        with self.lock:
            was_owner = self.owner == cid
            if was_owner:
                self.owner = None
        if was_owner and self.ctrl.moving():
            self.ctrl.stop("controlling client disconnected")
        self.ctrl._event(f"client {cid} gone" + (" (was in control: arm holds)" if was_owner else ""))

    def dispatch(self, cid, cmd, req) -> dict:
        c = self.ctrl
        with self.lock:
            if self.owner is None and cmd not in ("ping", "status", "read", "read_raw", "estop"):
                self.owner = cid
            if cmd == "hello" and req.get("takeover"):
                self.owner = cid
            is_owner = self.owner == cid
        if is_owner:
            c.heartbeat()
        elif cmd not in self.READ_ONLY:
            raise MotionError(f"client {self.owner} is in control; this connection is read-only "
                              f"(hello with takeover=true to take control)")
        if cmd == "ping":
            return {"t": time.time()}
        if cmd == "hello":
            lo, hi = c.joint_limits()
            return dict(server="printpick-arm", version=VERSION, joints=list(JOINTS), ids=c.ids, found=c.found,
                        calib=c.calib.method, mock=self.mock, in_control=is_owner, estopped=c.estopped,
                        limits=c.lim.to_json(), joint_lo=lo, joint_hi=hi, keepout=c.keepout.to_json(),
                        heartbeat_s=c.heartbeat_s)
        if cmd == "status":
            return c.status()
        if cmd == "read":
            return {"q": c.read_q()}
        if cmd == "read_raw":
            return {"raw": c.read_raw()}
        if cmd == "move":
            return c.move(req["q"], req.get("speed_deg_s"), req.get("accel_deg_s2"))
        if cmd == "wait":                      # served in slices: the owner waiting on us is alive
            t_end = time.monotonic() + min(float(req.get("timeout", 0.5)), 0.5)
            r = c.wait(0.0)
            while r.get("state") == "moving" and time.monotonic() < t_end:
                r = c.wait(0.05)
                if is_owner:
                    c.heartbeat()
            return r
        if cmd == "gripper":
            return c.gripper(req.get("open_pct"), req.get("rad"), req.get("speed_deg_s"))
        if cmd == "stop":
            return c.stop(req.get("reason", "stop requested"))
        if cmd == "estop":
            return c.estop()
        if cmd == "torque":
            return {"torque": c.torque(bool(req["on"]))}
        if cmd == "set_keepout":
            if "bed" in req:
                c.set_keepout(req["bed"], req.get("box_min"), req.get("box_max"))
            elif "box_min" in req:
                c.keepout.box_min, c.keepout.box_max = list(req["box_min"]), list(req["box_max"])
            return {"keepout": c.keepout.to_json()}
        if cmd == "get_keepout":
            return {"keepout": c.keepout.to_json()}
        if cmd == "calib":
            return {"calib": c.calib.to_json()}
        if cmd == "set_calib":
            return {"calib": c.set_calib(req["calib"]).to_json()}
        if cmd == "calib_from_servos":
            return {"calib": from_servos(c.bus).to_json()}
        raise ValueError(f"unknown command {cmd!r}")

    def serve_forever(self):
        self.tcp = _TCP((self.host, self.port), _Handler)
        self.tcp.app = self
        self.port = self.tcp.server_address[1]
        self.tcp.serve_forever(poll_interval=0.2)

    def start_background(self):
        self.tcp = _TCP((self.host, self.port), _Handler)
        self.tcp.app = self
        self.port = self.tcp.server_address[1]
        threading.Thread(target=self.tcp.serve_forever, kwargs=dict(poll_interval=0.05), daemon=True).start()
        return self

    def shutdown(self):
        if self.tcp is not None:
            self.tcp.shutdown()
            self.tcp.server_close()


def tailscale_ip() -> str | None:
    try:
        out = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=3)
        ip = out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
        return ip
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def mock_bus_at(q_model, calib: JointCalib, **kw) -> FeetechBus:
    raw = calib.model_to_raw(q_model)
    ser = MockSerial([MockServo(calib.joints[j]["id"], actual=raw[j]) for j in JOINTS], **kw)
    return FeetechBus("mock", ser=ser, timeout_s=0.02)


MOCK_REST_Q = [0.0, -1.2, 1.2, 1.0, 0.0, 0.0]


def build_controller(a) -> tuple[ArmController, bool]:
    calib_path = Path(a.calib)
    if a.mock:
        calib = JointCalib.load(calib_path) if calib_path.exists() and not a.mock_nominal else nominal()
        bus = mock_bus_at(MOCK_REST_Q, calib, time_scale=1.0)
    else:
        bus = FeetechBus(a.port, a.baud).open()
        if calib_path.exists():
            calib = JointCalib.load(calib_path)
        else:
            try:
                calib = from_servos(bus)
                calib.save(calib_path)
                print(f"no {calib_path}: built it from the LeRobot calibration stored in the servos ({calib.notes})")
            except (CalibError, CommError) as e:
                calib = nominal()
                print(f"WARNING: no joint calibration ({e}).\n  Moves are refused until you run "
                      f"python -m real.joint_calib guided --arm remote://<this pi>:{a.listen.split(':')[-1]}")
    lim = Limits(max_speed_deg_s=a.max_speed, max_accel_deg_s2=a.max_accel)
    ko = KeepOut(box_min=(-0.15, -0.40, a.floor_mm / 1000.0))
    ctrl = ArmController(bus, calib, limits=lim, keepout=ko, heartbeat_s=a.heartbeat_s,
                         allow_nominal=a.allow_nominal or a.mock, calib_path=calib_path)
    return ctrl.connect(), a.mock


def main(argv=None):
    ap = argparse.ArgumentParser(description="SO-101 arm server (Pi side)")
    ap.add_argument("--port", default="/dev/ttyACM0", help="servo bus serial port")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--listen", default=f"auto:{DEFAULT_TCP_PORT}",
                    help="HOST:PORT. 'auto' = this machine's Tailscale IPv4 (else 127.0.0.1); 0.0.0.0 = every "
                         "interface (then set --token)")
    ap.add_argument("--token", help="shared secret; clients use remote://host:port?token=...")
    ap.add_argument("--calib", default=str(JOINT_CALIB_JSON))
    ap.add_argument("--mock", action="store_true", help="simulated servos (rehearsal, no hardware)")
    ap.add_argument("--mock-nominal", action="store_true", help="mock: ignore an existing joint_calib.json")
    ap.add_argument("--allow-nominal", action="store_true", help="allow moves with the unmeasured calibration")
    ap.add_argument("--max-speed", type=float, default=30.0, help="joint speed cap deg/s (clients can only go slower)")
    ap.add_argument("--max-accel", type=float, default=60.0, help="joint acceleration cap deg/s^2")
    ap.add_argument("--heartbeat-s", type=float, default=2.0, help="watchdog: stop + hold after this much silence")
    ap.add_argument("--floor-mm", type=float, default=-20.0, help="no check point below this z (robot base frame)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    host, _, port = a.listen.rpartition(":")
    if host == "auto":
        ip = tailscale_ip()
        t_end = time.monotonic() + (30.0 if shutil.which("tailscale") else 0.0)
        while ip is None and time.monotonic() < t_end:          # at boot tailscaled may still be connecting
            time.sleep(2.0)
            ip = tailscale_ip()
        host = ip or "127.0.0.1"
        if host == "127.0.0.1":
            print("tailscale not found / not up: listening on 127.0.0.1 only (use --listen 0.0.0.0:8765 --token X "
                  "for a LAN, or run the demo on this machine)")
    if host not in ("127.0.0.1", "localhost") and not host.startswith("100.") and not a.token:
        print(f"WARNING: listening on {host} without --token: anyone on this network can move the arm")
    ctrl, mock = build_controller(a)
    print(ctrl.calib.summary())
    srv = ArmServer(ctrl, host, int(port), a.token, mock=mock)
    print(f"arm server on {host}:{port}  ->  on the Mac: --arm remote://{host}:{port}"
          + (f"?token={a.token}" if a.token else "") + ("   [MOCK servos]" if mock else ""), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        ctrl.close()


if __name__ == "__main__":
    main()
