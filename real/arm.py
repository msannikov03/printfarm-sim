"""SO-101 follower adapter: MuJoCo-model joint radians <-> LeRobot motor values, with safety rails.

    arm = Arm(port, robot_id="follower1", dry_run=True)    # dry-run: no hardware, logs + optional render
    arm.connect(); arm.move_to(q_model_rad); arm.gripper(open_pct); arm.disconnect()

Units (see JointMap): body joints are driven in LeRobot DEGREES mode (use_degrees=True), where 0 deg is
the middle of the range recorded by `lerobot-calibrate`; the menagerie model is derived from the
SO-ARM100 'so101_new_calib' description whose zero pose is that same middle pose, so by default
model_rad = radians(lerobot_deg) with sign +1, offset 0 for every body joint. The gripper is 0..100
(LeRobot RANGE_0_100) mapped linearly onto the model's gripper joint range (0 = closed).
VERIFY THIS ON THE REAL ARM: `python -m real.arm read --port ... --render runs/real_mirror.png` renders
the model at the live joint readings; it must look like the real arm. Fix signs/offsets in
real/joint_map.json if not.

Safety (live mode): every waypoint is checked with FK against a workspace box and a floor; motion is
interpolated in joint space at <= max_speed_deg_s with <= max_step_deg per command; LeRobot's own
max_relative_target is set as a second limiter; nothing moves without a y/N confirmation unless
confirm=False is passed explicitly.

Backends (the `port` argument / --arm option):
  /dev/tty.usbmodemXXXX or lerobot:///dev/...   LeRobot 0.6.1 SO101Follower (needs real/.venv-arm, Python >= 3.12)
  feetech:///dev/ttyACM0                         real/feetech.py driver in this process (pyserial only), joint
                                                 calibration real/joint_calib.json, motion by real/arm_server.ArmController
  remote://host:8765[?token=...]                 the Pi arm server (real/arm_server.py) over the tailnet
  feetech://mock, remote:// to a --mock server   rehearsal without hardware
With feetech/remote the server streams a smooth minimum-jerk trajectory to the target itself (speed AND
acceleration limited, keep-out and stall checks, watchdog); this class still plans and checks the path
with the MuJoCo model first and asks y/N.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from real import ROOT
from real.so101 import FIXED_TIP, GRIPPER_CLOSED_RAD, GRIPPER_OPEN_RAD, JOINTS, SO101

log = logging.getLogger("real.arm")
JOINT_MAP_JSON = ROOT / "joint_map.json"


class SafetyError(RuntimeError):
    pass


def arm_backend(port: str | None) -> tuple[str, str | None]:
    """'remote://h:p' -> ('remote', url); 'feetech:///dev/x' -> ('feetech', '/dev/x'); else ('lerobot', port)."""
    if not port:
        return "lerobot", port
    if port.startswith(("remote://", "tcp://")):
        return "remote", port
    for kind in ("feetech", "lerobot"):
        if port.startswith(kind + ":"):
            rest = port.split(":", 1)[1]
            return kind, rest[2:] if rest.startswith("//") else rest      # feetech:///dev/x -> /dev/x, //mock -> mock
    return "lerobot", port


@dataclass
class JointMap:
    """lerobot_deg = sign * degrees(model_rad) + offset_deg   (body joints, LeRobot use_degrees=True)
    gripper:  lerobot 0..100  <->  model rad  closed_rad + (open_rad - closed_rad) * v / 100
    """
    names: tuple = JOINTS
    sign: dict = field(default_factory=lambda: {j: 1.0 for j in JOINTS[:5]})
    offset_deg: dict = field(default_factory=lambda: {j: 0.0 for j in JOINTS[:5]})
    gripper_closed_rad: float = GRIPPER_CLOSED_RAD
    gripper_open_rad: float = GRIPPER_OPEN_RAD

    def model_to_lerobot(self, q) -> dict:
        q = np.asarray(q, float)
        out = {j: float(self.sign[j] * math.degrees(q[i]) + self.offset_deg[j]) for i, j in enumerate(self.names[:5])}
        if len(q) > 5:
            g = (q[5] - self.gripper_closed_rad) / (self.gripper_open_rad - self.gripper_closed_rad) * 100.0
            out["gripper"] = float(np.clip(g, 0.0, 100.0))
        return out

    def lerobot_to_model(self, v: dict) -> np.ndarray:
        q = np.zeros(6)
        for i, j in enumerate(self.names[:5]):
            q[i] = math.radians((float(v[j]) - self.offset_deg[j]) / self.sign[j])
        g = float(v.get("gripper", 0.0))
        q[5] = self.gripper_closed_rad + (self.gripper_open_rad - self.gripper_closed_rad) * g / 100.0
        return q

    def to_json(self):
        return dict(sign=self.sign, offset_deg=self.offset_deg, gripper_closed_rad=self.gripper_closed_rad,
                    gripper_open_rad=self.gripper_open_rad,
                    formula="lerobot_deg = sign * degrees(model_rad) + offset_deg; gripper 0..100 linear "
                            "from closed_rad to open_rad")

    def save(self, path=JOINT_MAP_JSON):
        Path(path).write_text(json.dumps(self.to_json(), indent=2))

    @classmethod
    def load(cls, path) -> "JointMap":
        d = json.loads(Path(path).read_text())
        jm = cls()
        jm.sign.update({k: float(v) for k, v in d.get("sign", {}).items()})
        jm.offset_deg.update({k: float(v) for k, v in d.get("offset_deg", {}).items()})
        jm.gripper_closed_rad = float(d.get("gripper_closed_rad", jm.gripper_closed_rad))
        jm.gripper_open_rad = float(d.get("gripper_open_rad", jm.gripper_open_rad))
        return jm

    @classmethod
    def load_default(cls) -> "JointMap":
        return cls.load(JOINT_MAP_JSON) if JOINT_MAP_JSON.exists() else cls()


@dataclass
class Safety:
    max_speed_deg_s: float = 20.0         # joint-space speed of the interpolation
    max_step_deg: float = 2.0             # per command (rate 25 Hz -> 50 deg/s cap; speed limit binds first)
    rate_hz: float = 25.0
    lerobot_max_relative_target: float = 5.0   # LeRobot clips each command to this many deg (gripper: units) from present
    max_gripper_step: float = 2.0         # gripper units (0..100) per command
    max_accel_deg_s2: float = 60.0        # feetech / remote backends: the server's min-jerk profile
    settle_s: float = 1.5                 # keep sending the final target until within settle_tol_deg (live)
    settle_tol_deg: float = 1.0
    # workspace box for the fixed-jaw tip AND the hover tool point, robot frame (m)
    box_min: tuple = (-0.05, -0.30, -0.02)
    box_max: tuple = (0.45, 0.30, 0.40)
    floor_z: float | None = None          # if set: tool tip must stay above this robot-frame z (bed + margin)
    check_points: int = 10                # FK checks per interpolated segment
    extra_checks: list = field(default_factory=list)   # fn(q, what) raising SafetyError (e.g. part clearance)

    def check_q(self, kin: SO101, q, what="target"):
        for tcp in (FIXED_TIP,):
            p, _ = kin.fk(q, tcp)
            if np.any(p < np.array(self.box_min)) or np.any(p > np.array(self.box_max)):
                raise SafetyError(f"{what}: tool tip {np.round(p * 1000).tolist()} mm leaves the workspace box "
                                  f"{np.round(np.array(self.box_min) * 1000).tolist()}..{np.round(np.array(self.box_max) * 1000).tolist()}")
            if self.floor_z is not None and p[2] < self.floor_z:
                raise SafetyError(f"{what}: tool tip z {p[2] * 1000:.0f} mm is below the floor limit "
                                  f"{self.floor_z * 1000:.0f} mm (bed + margin)")
        for fn in self.extra_checks:
            fn(q, what)


def _import_lerobot():
    """LeRobot moved the SO-101 classes between releases; try the known locations."""
    errors = []
    for mod, cls, cfg in (("lerobot.robots.so_follower", "SO101Follower", "SO101FollowerConfig"),      # 0.4.3+ (0.6.1)
                          ("lerobot.robots.so101_follower", "SO101Follower", "SO101FollowerConfig"),  # <= 0.4.2
                          ("lerobot.common.robots.so101_follower", "SO101Follower", "SO101FollowerConfig")):
        try:
            m = __import__(mod, fromlist=[cls, cfg])
            return getattr(m, cls), getattr(m, cfg)
        except (ImportError, AttributeError) as e:
            errors.append(f"{mod}: {e}")
    raise ImportError("lerobot with SO-101 support not found in this Python. Do NOT add it to the shared sim venv\n"
                      "(it downgrades numpy/opencv and pulls torch). Make the separate arm venv once:\n"
                      "  ./real/setup_arm_env.sh      # -> real/.venv-arm (Python 3.12, lerobot[feetech]==0.6.1)\n"
                      "and run live commands with real/.venv-arm/bin/python\n" + "\n".join(errors))


class Arm:
    def __init__(self, port: str | None, robot_id: str = "follower", dry_run: bool = True,
                 joint_map: JointMap | None = None, safety: Safety | None = None, confirm: bool = True,
                 kin: SO101 | None = None, sim_hand=None, calibration_dir: str | None = None, render_dir=None,
                 input_fn=input, sleep=time.sleep, release_on_exit: bool = False):
        self.port, self.robot_id, self.dry_run = port, robot_id, dry_run
        self.jm = joint_map or JointMap.load_default()
        self.safety = safety or Safety()
        self.confirm = confirm
        self.kin = kin or SO101()
        self.sim_hand = sim_hand            # BedCalib: dry-run touch-off simulates a hand at bed points
        self.calibration_dir = calibration_dir
        self.render_dir = render_dir
        self.input = input_fn
        self.release_on_exit = release_on_exit   # False: motors keep holding after disconnect (arm does not drop)
        self.sleep = sleep
        self.robot = None
        self.q = np.array([0.0, -1.2, 1.2, 1.0, 0.0, 0.0])   # dry-run state (a folded, safe-looking pose)
        self.sent: list[dict] = []          # every command (lerobot units), for logs/tests
        self._rng = np.random.default_rng(0)
        self.backend, self.target = arm_backend(port)
        self.drv = None                     # feetech / remote driver (read_q, move, torque, stop, set_keepout, close)

    # ------------------------------------------------------------------ connection
    def _connect_driver(self):
        if self.backend == "remote":
            from real.arm_client import RemoteArm
            self.drv = RemoteArm(self.target).connect()
            info = self.drv.info
            print(f"  arm server {self.target.split('?')[0]}: calib {info['calib']}, servos {sorted(info['found'])}"
                  + (" [MOCK servos]" if info.get("mock") else "") + (", E-STOPPED" if info.get("estopped") else ""))
        else:
            from real.arm_server import ArmController, mock_bus_at, MOCK_REST_Q
            from real.feetech import FeetechBus
            from real.joint_calib import JointCalib
            calib = JointCalib.load_or_nominal()
            mock = self.target == "mock"
            bus = mock_bus_at(MOCK_REST_Q, calib) if mock else FeetechBus(self.target).open()
            self.drv = ArmController(bus, calib, heartbeat_s=None, allow_nominal=mock).connect()
            print(f"  feetech bus {self.target}: calib {calib.method}" + (" [MOCK servos]" if mock else ""))
        self.q = np.array(self.drv.read_q(), float)

    def set_bed_keepout(self, calib, margin_mm: float = 8.0):
        """Second line of defence on the server: no jaw tip within margin_mm of the bed (feetech / remote)."""
        if self.drv is not None:
            from real.arm_client import bed_keepout
            self.drv.set_keepout(bed=bed_keepout(calib, margin_mm))

    def connect(self):
        if self.dry_run:
            log.info("dry-run arm: no hardware")
            return
        if self.backend in ("remote", "feetech"):
            return self._connect_driver()
        SO101Follower, SO101FollowerConfig = _import_lerobot()
        kw = dict(port=self.port, id=self.robot_id, use_degrees=True,
                  max_relative_target=self.safety.lerobot_max_relative_target,
                  disable_torque_on_disconnect=self.release_on_exit)
        if self.calibration_dir:
            kw["calibration_dir"] = Path(self.calibration_dir)
        try:
            cfg = SO101FollowerConfig(**kw)
        except TypeError:                   # older/newer config without some field
            kw.pop("disable_torque_on_disconnect", None)
            cfg = SO101FollowerConfig(**kw)
        self.robot = SO101Follower(cfg)
        self.robot.connect(calibrate=False)
        if hasattr(self.robot, "is_calibrated") and not self.robot.is_calibrated:
            self.robot.disconnect()
            raise RuntimeError(f"SO-101 '{self.robot_id}': calibration file missing or different from the values "
                               f"stored in the motors. Either\n"
                               f"  lerobot-calibrate --robot.type=so101_follower --robot.port={self.port} "
                               f"--robot.id={self.robot_id}   (if a file exists: press ENTER to write it to the motors)\n"
                               f"or, if it was calibrated with LeRobot <= 0.4.2, point --calibration-dir at "
                               f"~/.cache/huggingface/lerobot/calibration/robots/so101_follower "
                               f"(newer versions look in .../robots/so_follower)")
        self.q = self.read_joints_rad()
        log.info("connected; joints (model deg) %s", np.degrees(self.q).round(1).tolist())

    def disconnect(self):
        if self.drv is not None:
            if self.release_on_exit:
                self.drv.torque(False)
            self.drv.close()          # remote: the server keeps holding the pose; local: stop + close the port
            self.drv = None
        if self.robot is not None:
            self.robot.disconnect()
            self.robot = None

    # ------------------------------------------------------------------ reading
    def read_lerobot(self, target_hint_bed_mm=None) -> dict:
        if not self.dry_run and self.drv is not None:
            return self.jm.model_to_lerobot(self.drv.read_q())
        if not self.dry_run:
            obs = self.robot.get_observation()
            return {j: float(obs[f"{j}.pos"]) for j in JOINTS}
        if self.sim_hand is not None and target_hint_bed_mm is not None:
            # simulated hand-guiding for dry-run touch-off: IK of the fixed tip onto the bed point + noise
            p = self.sim_hand.bed_to_robot([*target_hint_bed_mm, 0.0])
            r = self.kin.solve(p, approach=np.array([0, 0, -1.0]), tcp=FIXED_TIP, ang_tol_deg=30.0)
            if not r.ok:
                r = self.kin.solve(p, tcp=FIXED_TIP)
            q = r.q + np.r_[self._rng.normal(0, math.radians(0.3), 5), 0.0]
            return self.jm.model_to_lerobot(q)
        return self.jm.model_to_lerobot(self.q)

    def read_joints_rad(self, target_hint_bed_mm=None) -> np.ndarray:
        return self.jm.lerobot_to_model(self.read_lerobot(target_hint_bed_mm))

    def torque(self, on: bool):
        if self.dry_run:
            log.info("dry-run: torque %s", "on" if on else "off")
            return
        if self.drv is not None:
            if not on:
                print("  torque OFF: support the arm, it will sag under gravity")
            self.drv.torque(on)               # the server sets goal := present before torque on
            return
        bus = getattr(self.robot, "bus", None)
        if bus is None:
            raise RuntimeError("cannot reach the motor bus to toggle torque")
        if on:
            # goal := present first, or the arm jumps back to the last commanded pose when torque returns
            obs = self.robot.get_observation()
            self.robot.send_action({f"{j}.pos": float(obs[f"{j}.pos"]) for j in JOINTS})
            bus.enable_torque()
        else:
            print("  torque OFF: support the arm, it will sag under gravity")
            bus.disable_torque()

    # ------------------------------------------------------------------ motion
    def plan(self, q_target, q_start=None) -> list[np.ndarray]:
        """Joint-space interpolation at the speed limit, every waypoint safety-checked. Raises SafetyError."""
        q0 = np.array(self.q if q_start is None else q_start, float)
        q1 = np.array(q_target, float)
        if len(q1) == 5:
            q1 = np.r_[q1, q0[5]]
        lo, hi = self.kin.lo, self.kin.hi
        if np.any(q1[:5] < lo[:5] - 1e-6) or np.any(q1[:5] > hi[:5] + 1e-6):
            raise SafetyError(f"target outside joint limits: {np.degrees(q1).round(1).tolist()}")
        self.safety.check_q(self.kin, q1, "target")
        d = np.degrees(np.abs(q1[:5] - q0[:5])).max()
        step = min(self.safety.max_step_deg, self.safety.max_speed_deg_s / self.safety.rate_hz)
        dg = abs(self.jm.model_to_lerobot(q1)["gripper"] - self.jm.model_to_lerobot(q0)["gripper"])
        n = max(1, int(math.ceil(d / step)), int(math.ceil(dg / self.safety.max_gripper_step)))
        path = [q0 + (q1 - q0) * (k / n) for k in range(1, n + 1)]
        for k in np.linspace(0, n - 1, min(n, self.safety.check_points)).astype(int):
            self.safety.check_q(self.kin, path[k], f"waypoint {k + 1}/{n}")
        return path

    def move_to(self, q_target, label: str = "", confirm: bool | None = None) -> list[np.ndarray]:
        path = self.plan(q_target)
        q1 = path[-1]
        tip, _ = self.kin.fk(q1, FIXED_TIP)
        cmd = self.jm.model_to_lerobot(q1)
        msg = (f"move {label}: {len(path)} steps, ~{len(path) / self.safety.rate_hz:.1f} s, fixed-jaw tip -> "
               f"{np.round(tip * 1000).astype(int).tolist()} mm; lerobot "
               + ", ".join(f"{k}={v:.1f}" for k, v in cmd.items()))
        print(("[dry-run] " if self.dry_run else "") + msg)
        if not self.dry_run and (self.confirm if confirm is None else confirm):
            if self.input("  execute on the REAL arm? [y/N] ").strip().lower() != "y":
                raise SafetyError("motion declined by the operator")
        if not self.dry_run and self.drv is not None:
            return self._move_driver(path, q1)
        for q in path:
            a = self.jm.model_to_lerobot(q)
            self.sent.append(a)
            if not self.dry_run:
                self.robot.send_action({f"{j}.pos": v for j, v in a.items()})
                self.sleep(1.0 / self.safety.rate_hz)
        self.q = q1.copy()
        if not self.dry_run:
            # LeRobot clips each goal to max_relative_target from the PRESENT position, so a lagging joint
            # would stop short: keep re-sending the final goal until it is reached (or settle_s runs out)
            final = {f"{j}.pos": v for j, v in cmd.items()}
            for _ in range(max(1, int(self.safety.settle_s * self.safety.rate_hz))):
                meas = self.read_joints_rad()
                if np.degrees(np.abs(meas[:5] - q1[:5])).max() <= self.safety.settle_tol_deg:
                    break
                self.robot.send_action(final)
                self.sleep(1.0 / self.safety.rate_hz)
            meas = self.read_joints_rad()
            err = np.degrees(np.abs(meas[:5] - q1[:5]))
            print(f"  reached; tracking error per joint (deg) {err.round(1).tolist()}")
            if err.max() > 5:
                log.warning("tracking error > 5 deg: blocked, overloaded or a joint map problem")
            self.q = meas
        return path

    def _move_driver(self, path, q1):
        """feetech / remote: one target, the server interpolates (min-jerk) and checks keep-out + stalls."""
        from real.arm_client import RemoteError
        self.sent.extend(self.jm.model_to_lerobot(q) for q in (path[0], path[-1]))
        try:
            r = self.drv.move([float(v) for v in q1], speed_deg_s=self.safety.max_speed_deg_s,
                              accel_deg_s2=self.safety.max_accel_deg_s2, wait=True)
        except RemoteError as e:
            raise SafetyError(f"arm server: {e}") from e
        except RuntimeError as e:                 # local controller: MotionError / CommError
            raise SafetyError(str(e)) from e
        if r.get("state") == "aborted":
            raise SafetyError(f"move aborted: {r.get('reason')}")
        meas = np.array(self.drv.read_q(), float)
        err = np.degrees(np.abs(meas[:5] - q1[:5]))
        print(f"  reached in {r.get('duration_s', 0):.1f} s; error per joint (deg) {err.round(1).tolist()}"
              + (f", peak tracking error {r['max_track_err_deg']:.1f}" if "max_track_err_deg" in r else ""))
        if err.max() > 5:
            log.warning("tracking error > 5 deg: blocked, overloaded or a calibration problem")
        self.q = meas
        return path

    def gripper(self, open_pct: float, confirm: bool | None = None):
        g = GRIPPER_CLOSED_RAD + (GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD) * float(np.clip(open_pct, 0, 100)) / 100
        q = self.q.copy()
        q[5] = g
        return self.move_to(q, label=f"gripper {open_pct:.0f}%", confirm=confirm)


# ----------------------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description="SO-101 adapter utilities")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("read", help="print live joints (lerobot + model) and the fixed-tip position")
    r.add_argument("--port", "--arm", dest="port", required=True, help="LeRobot port, remote://<pi>:8765 or feetech:///dev/...")
    r.add_argument("--robot-id", default="follower")
    r.add_argument("--calibration-dir", help="LeRobot calibration dir if not the default")
    r.add_argument("--render", help="also render the MuJoCo model at the live joints to this PNG")
    r.add_argument("--loop", action="store_true", help="keep printing at 2 Hz (torque off: move it by hand)")
    sub.add_parser("map", help="write the default joint map to real/joint_map.json for editing")
    a = ap.parse_args(argv)
    if a.cmd == "map":
        JointMap.load_default().save()
        print("wrote", JOINT_MAP_JSON)
        return
    arm = Arm(a.port, robot_id=a.robot_id, dry_run=False, calibration_dir=a.calibration_dir)
    arm.connect()
    try:
        if a.loop:
            arm.torque(False)
        while True:
            v = arm.read_lerobot()
            q = arm.jm.lerobot_to_model(v)
            tip, _ = arm.kin.fk(q, FIXED_TIP)
            print("lerobot " + " ".join(f"{k}={x:7.2f}" for k, x in v.items())
                  + f" | model deg {np.degrees(q[:5]).round(1).tolist()} | tip mm {np.round(tip * 1000).astype(int).tolist()}")
            if a.render:
                from real.preview import render_q
                render_q(q, a.render)
                print("rendered", a.render)
            if not a.loop:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        if a.loop:
            arm.torque(True)
        arm.disconnect()


if __name__ == "__main__":
    main()
