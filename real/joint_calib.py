"""Joint calibration: Feetech raw ticks <-> MuJoCo SO-101 model joint angles (real/joint_calib.json).

Per joint:  model_rad = sign * scale * (raw - zero_raw),   raw limited to [min_raw, max_raw]
(scale = 2*pi/4096 rad per tick for a body joint; the gripper may use a fitted scale).

Four ways to get it, best first:
  from-servos   the arm was calibrated with LeRobot (lerobot-calibrate): LeRobot stores Homing_Offset and
                Min/Max_Position_Limit IN THE SERVOS' EEPROM, so reading them back reproduces LeRobot's
                own conversion exactly, no JSON file needed:
                    python -m real.joint_calib from-servos --port /dev/ttyACM0
  from-lerobot  the LeRobot calibration JSON (~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json,
                {motor: {id, drive_mode, homing_offset, range_min, range_max}}):
                    python -m real.joint_calib from-lerobot follower1.json [--port /dev/ttyACM0 to cross-check]
  guided        no LeRobot calibration: torque off, put the arm in the ZERO pose, then move each joint a bit in
                its + direction (gives zero and sign; also catches swapped motor ids), then close/open the gripper:
                    python -m real.joint_calib guided --port /dev/ttyACM0      (or --arm remote://pi:8765)
  nominal       zero = 2048, sign +1 (factory centre): only for rehearsals / the mock bus.
Then check:     python -m real.joint_calib live --port /dev/ttyACM0   (torque off, prints model angles + tip)

LeRobot conventions mirrored here (v0.6.1, motors/motors_bus.py _normalize, so_follower.py):
  body joints DEGREES:  deg = (raw - (range_min + range_max) / 2) * 360 / 4095  (drive_mode not applied)
  gripper RANGE_0_100:  v = (raw - range_min) / (range_max - range_min) * 100   (drive_mode 1 -> 100 - v)
and real/arm.py's JointMap (lerobot_deg = sign * model_deg + offset_deg; gripper 0..100 -> closed..open rad),
so `from-lerobot` / `from-servos` give exactly the angles the LeRobot path gives. real/joint_map.json, if
present, is folded in.

The ZERO pose of the model (all joints 0), as seen from the side:
  upper arm vertical, forearm horizontal pointing forward, wrist straight (gripper in line with the forearm,
  pointing forward), the FIXED jaw at the bottom and the moving jaw on top, gripper closed.
The + direction of each joint (computed from the model, real/tests check it):
  shoulder_pan  +  the arm swings to ITS right (clockwise seen from above)
  shoulder_lift +  the upper arm tips forward / down
  elbow_flex    +  the forearm tips down
  wrist_flex    +  the gripper tips down
  wrist_roll    +  seen from behind the arm, the top of the gripper rolls to the left (counter-clockwise)
  gripper       +  opens
Pure Python + json: runs on the Pi (no numpy / mujoco).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from real.kin_lite import JOINTS

ROOT = Path(__file__).resolve().parent
JOINT_CALIB_JSON = ROOT / "joint_calib.json"
JOINT_MAP_JSON = ROOT / "joint_map.json"
IDS = {j: i + 1 for i, j in enumerate(JOINTS)}            # SO-101: motor ids 1..6 in this order (LeRobot)
TICK_RAD = 2 * math.pi / 4096                             # STS3215: 4096 ticks per turn
LEROBOT_TICK_RAD = 2 * math.pi / 4095                     # LeRobot DEGREES mode divides by (resolution - 1)
GRIPPER_CLOSED_RAD, GRIPPER_OPEN_RAD = -0.174533, 1.7453292   # model gripper range (= real/so101.py)
PLUS_DIRECTION = {
    "shoulder_pan": "swing the arm to ITS right (clockwise seen from above)",
    "shoulder_lift": "tip the upper arm forward / down",
    "elbow_flex": "tip the forearm down",
    "wrist_flex": "tip the gripper down",
    "wrist_roll": "roll the gripper: seen from behind the arm, its top goes to the left (counter-clockwise)",
    "gripper": "open the gripper",
}
ZERO_POSE_TEXT = ("ZERO pose: upper arm vertical, forearm horizontal pointing straight forward, wrist straight "
                  "(gripper in line with the forearm), fixed jaw at the bottom / moving jaw on top, gripper closed")


class CalibError(RuntimeError):
    pass


class JointCalib:
    def __init__(self, joints: dict, method: str = "nominal", notes: str = "", created: str | None = None):
        self.joints = {j: dict(joints[j]) for j in JOINTS}
        self.method, self.notes = method, notes
        self.created = created or time.strftime("%Y-%m-%d %H:%M:%S")

    # ------------------------------------------------------------------ conversion
    def ids(self) -> list[int]:
        return [self.joints[j]["id"] for j in JOINTS]

    def raw_to_model(self, raw: dict) -> list[float]:
        """raw: {joint name or motor id: ticks} -> 6 model angles (rad); missing joints -> nan."""
        out = []
        for j in JOINTS:
            c = self.joints[j]
            r = raw.get(j, raw.get(c["id"]))
            out.append(float("nan") if r is None else c["sign"] * c["scale"] * (float(r) - c["zero_raw"]))
        return out

    def model_to_raw(self, q, clamp: bool = True) -> dict:
        """6 (or 5) model angles -> {joint: ticks}. Raises CalibError outside [min_raw, max_raw] if not clamp."""
        out = {}
        for i, j in enumerate(JOINTS[:len(q)]):
            c = self.joints[j]
            r = c["zero_raw"] + q[i] / (c["sign"] * c["scale"])
            if r < c["min_raw"] - 0.5 or r > c["max_raw"] + 0.5:
                if not clamp:
                    raise CalibError(f"{j}: {math.degrees(q[i]):.1f} deg = raw {r:.0f} is outside the servo range "
                                     f"{c['min_raw']}..{c['max_raw']}")
                r = min(c["max_raw"], max(c["min_raw"], r))
            out[j] = int(round(r))
        return out

    def model_limits(self) -> tuple[list, list]:
        """Model-angle interval covered by [min_raw, max_raw] per joint."""
        lo, hi = [], []
        for j in JOINTS:
            c = self.joints[j]
            a = c["sign"] * c["scale"] * (c["min_raw"] - c["zero_raw"])
            b = c["sign"] * c["scale"] * (c["max_raw"] - c["zero_raw"])
            lo.append(min(a, b))
            hi.append(max(a, b))
        return lo, hi

    # ------------------------------------------------------------------ io
    def to_json(self) -> dict:
        return dict(method=self.method, created=self.created, notes=self.notes,
                    formula="model_rad = sign * scale * (raw - zero_raw); raw clamped to [min_raw, max_raw]",
                    joints=self.joints)

    def save(self, path=JOINT_CALIB_JSON) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_json(), indent=2))
        return path

    @classmethod
    def from_dict(cls, d: dict) -> "JointCalib":
        missing = [j for j in JOINTS if j not in d.get("joints", {})]
        if missing:
            raise CalibError(f"joint calibration lacks {missing}")
        return cls(d["joints"], d.get("method", "?"), d.get("notes", ""), d.get("created"))

    @classmethod
    def load(cls, path=JOINT_CALIB_JSON) -> "JointCalib":
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def load_or_nominal(cls, path=JOINT_CALIB_JSON) -> "JointCalib":
        return cls.load(path) if Path(path).exists() else nominal()

    def summary(self) -> str:
        lo, hi = self.model_limits()
        rows = [f"joint calibration: method={self.method} created={self.created} {self.notes}"]
        for k, j in enumerate(JOINTS):
            c = self.joints[j]
            rows.append(f"  {j:<13} id {c['id']}  zero {c['zero_raw']:7.1f}  sign {c['sign']:+.0f}  "
                        f"{math.degrees(c['scale']) * 1000:6.2f} mdeg/tick  raw {c['min_raw']}..{c['max_raw']}  "
                        f"= model {math.degrees(lo[k]):6.1f}..{math.degrees(hi[k]):6.1f} deg")
        return "\n".join(rows)


# ---------------------------------------------------------------------------- constructors

def _limits_from_model(zero, sign, scale, j, ranges=None):
    lo, hi = (ranges or _model_ranges())[j]
    a, b = zero + lo / (sign * scale), zero + hi / (sign * scale)
    return int(max(0, math.ceil(min(a, b)))), int(min(4095, math.floor(max(a, b))))


def _model_ranges():
    try:
        from real.kin_lite import Chain
        return Chain().ranges
    except Exception:          # chain file missing: the MJCF ranges, copied
        return {"shoulder_pan": (-1.91986, 1.91986), "shoulder_lift": (-1.7453293, 1.7453293),
                "elbow_flex": (-1.69, 1.69), "wrist_flex": (-1.658063, 1.658063),
                "wrist_roll": (-2.7438473, 2.8412063), "gripper": (GRIPPER_CLOSED_RAD, GRIPPER_OPEN_RAD)}


def nominal() -> JointCalib:
    """Factory centre 2048 = model zero, sign +1. NOT measured: rehearsals and the mock bus only."""
    joints = {}
    for j in JOINTS:
        mn, mx = _limits_from_model(2048.0, 1.0, TICK_RAD, j)
        joints[j] = dict(id=IDS[j], zero_raw=2048.0, sign=1.0, scale=TICK_RAD, min_raw=mn, max_raw=mx)
    return JointCalib(joints, "nominal", "NOT MEASURED: factory centre = model zero")


def _joint_map() -> tuple[dict, dict]:
    if JOINT_MAP_JSON.exists():
        d = json.loads(JOINT_MAP_JSON.read_text())
        return d.get("sign", {}), d.get("offset_deg", {})
    return {}, {}


def from_lerobot(cal: dict, method: str = "lerobot-json", joint_map: tuple | None = None) -> JointCalib:
    """cal: LeRobot calibration {motor: {id, drive_mode, homing_offset, range_min, range_max}}.
    Composes LeRobot's DEGREES / RANGE_0_100 normalisation with real/arm.py's JointMap (sign, offset_deg)."""
    signs, offs = joint_map if joint_map is not None else _joint_map()
    joints, notes = {}, []
    for j in JOINTS:
        if j not in cal:
            raise CalibError(f"LeRobot calibration has no '{j}' (keys: {sorted(cal)})")
        c = cal[j]
        mn, mx = int(c["range_min"]), int(c["range_max"])
        if mx <= mn:
            raise CalibError(f"{j}: range_min {mn} >= range_max {mx}")
        if j != "gripper":
            s = float(signs.get(j, 1.0))
            mid = (mn + mx) / 2.0
            zero = mid + float(offs.get(j, 0.0)) * 4095.0 / 360.0
            joints[j] = dict(id=int(c["id"]), zero_raw=zero, sign=s, scale=LEROBOT_TICK_RAD, min_raw=mn, max_raw=mx)
        else:
            scale = (GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD) / (mx - mn)
            if int(c.get("drive_mode", 0)):
                joints[j] = dict(id=int(c["id"]), zero_raw=mx + GRIPPER_CLOSED_RAD / scale, sign=-1.0, scale=scale,
                                 min_raw=mn, max_raw=mx)
            else:
                joints[j] = dict(id=int(c["id"]), zero_raw=mn - GRIPPER_CLOSED_RAD / scale, sign=1.0, scale=scale,
                                 min_raw=mn, max_raw=mx)
        if j not in ("wrist_roll", "gripper") and (mn, mx) == (0, 4095):
            notes.append(f"{j} has the factory range 0..4095 (not calibrated by LeRobot?)")
    return JointCalib(joints, method, "; ".join(notes))


def from_servos(bus, ids=None) -> JointCalib:
    """Read what LeRobot wrote into the servo EEPROM (Homing_Offset, Min/Max_Position_Limit) [LeRobot
    feetech.py read_calibration: drive_mode is always 0] and build the same conversion as from_lerobot."""
    ids = ids or IDS
    eep = bus.read_eeprom_calibration([ids[j] for j in JOINTS])
    cal = {j: dict(id=ids[j], drive_mode=0, **eep[ids[j]]) for j in JOINTS}
    uncal = [j for j in JOINTS if j not in ("wrist_roll",) and (cal[j]["range_min"], cal[j]["range_max"]) == (0, 4095)]
    if len(uncal) >= 3:
        raise CalibError(f"servos {uncal} have the factory limits 0..4095 and homing offsets "
                         f"{[cal[j]['homing_offset'] for j in uncal]}: this arm was not calibrated with LeRobot. "
                         f"Use: python -m real.joint_calib guided ...")
    c = from_lerobot(cal, method="lerobot-eeprom")
    c.notes = (c.notes + "; " if c.notes else "") + "homing offsets " + str({j: cal[j]["homing_offset"] for j in JOINTS})
    return c


def cross_check(c: JointCalib, bus) -> list[str]:
    """Compare a calibration with the servo EEPROM limits (what LeRobot's is_calibrated compares)."""
    eep = bus.read_eeprom_calibration(c.ids())
    out = []
    for j in JOINTS:
        e, k = eep[c.joints[j]["id"]], c.joints[j]
        if (e["range_min"], e["range_max"]) != (k["min_raw"], k["max_raw"]):
            out.append(f"{j}: file range {k['min_raw']}..{k['max_raw']} but the servo has "
                       f"{e['range_min']}..{e['range_max']} (calibrated again since? use from-servos)")
    return out


# ---------------------------------------------------------------------------- guided procedure

def _avg(reader, n=8):
    acc = {j: 0.0 for j in JOINTS}
    for _ in range(n):
        r = reader.read_raw()
        for j in JOINTS:
            if r.get(j) is None:
                raise CalibError(f"no reading from {j}")
            acc[j] += r[j]
    return {j: v / n for j, v in acc.items()}


def guided(reader, input_fn=input, print_fn=print, min_move_deg: float = 12.0) -> JointCalib:
    """reader: .read_raw() -> {joint: ticks}, .torque(on). Torque goes OFF for the whole procedure."""
    print_fn("Guided joint calibration. Torque goes OFF now: SUPPORT THE ARM, it will sag.")
    input_fn("  hold the arm, then press Enter ")
    reader.torque(False)
    try:
        input_fn(f"[1/8] Put the arm in the {ZERO_POSE_TEXT}. Hold it still, press Enter ")
        zero = _avg(reader)
        print_fn("      zero pose raw: " + ", ".join(f"{j}={zero[j]:.0f}" for j in JOINTS))
        signs, notes = {}, []
        for k, j in enumerate(JOINTS[:5], 2):
            before = _avg(reader, 4)
            input_fn(f"[{k}/8] Move ONLY {j} by 20-45 deg: {PLUS_DIRECTION[j]}. Hold it, press Enter ")
            after = _avg(reader, 4)
            d = {m: (after[m] - before[m]) * TICK_RAD for m in JOINTS}
            moved = max(d, key=lambda m: abs(d[m]))
            if abs(d[j]) < math.radians(min_move_deg):
                raise CalibError(f"{j} moved only {math.degrees(d[j]):.1f} deg (need > {min_move_deg:.0f}); "
                                 + (f"but {moved} moved {math.degrees(d[moved]):.1f} deg: motor ids swapped / "
                                    f"wrong cable?" if moved != j else "move it further"))
            if moved != j:
                raise CalibError(f"you moved {j} but servo id {IDS[moved]} ({moved}) changed most "
                                 f"({math.degrees(d[moved]):.1f} deg): motor ids or cabling do not match the SO-101 order")
            others = [m for m in JOINTS[:5] if m != j and abs(d[m]) > math.radians(8)]
            if others:
                notes.append(f"while moving {j}, {others} also moved > 8 deg")
            signs[j] = 1.0 if d[j] > 0 else -1.0
            print_fn(f"      {j}: raw {before[j]:.0f} -> {after[j]:.0f}  => sign {signs[j]:+.0f}")
        input_fn("[7/8] Close the gripper fully (jaws touching), press Enter ")
        g_closed = _avg(reader)["gripper"]
        input_fn("[8/8] Open the gripper fully, press Enter ")
        g_open = _avg(reader)["gripper"]
        if abs(g_open - g_closed) * TICK_RAD < math.radians(30):
            raise CalibError(f"gripper moved only {math.degrees(abs(g_open - g_closed) * TICK_RAD):.0f} deg")
    finally:
        input_fn("  Hold the arm (torque comes back on at the CURRENT pose), press Enter ")
        reader.torque(True)
    joints = {}
    ranges = _model_ranges()
    for j in JOINTS[:5]:
        mn, mx = _limits_from_model(zero[j], signs[j], TICK_RAD, j, ranges)
        joints[j] = dict(id=IDS[j], zero_raw=round(zero[j], 1), sign=signs[j], scale=TICK_RAD, min_raw=mn, max_raw=mx)
        lo, hi = ranges[j]
        need = (hi - lo) / TICK_RAD
        if mx - mn < need - 2:
            notes.append(f"{j}: the encoder wraps inside the joint range; usable raw {mn}..{mx} "
                         f"({math.degrees((mx - mn) * TICK_RAD):.0f} of {math.degrees(hi - lo):.0f} deg). "
                         f"lerobot-calibrate (homing offsets) avoids this")
    gs = 1.0 if g_open > g_closed else -1.0
    gz = g_closed - GRIPPER_CLOSED_RAD / (gs * TICK_RAD)
    joints["gripper"] = dict(id=IDS["gripper"], zero_raw=round(gz, 1), sign=gs, scale=TICK_RAD,
                             min_raw=int(round(min(g_closed, g_open))), max_raw=int(round(max(g_closed, g_open))))
    c = JointCalib(joints, "guided", "; ".join(notes))
    print_fn(c.summary())
    return c


class BusReader:
    """Adapter: FeetechBus -> read_raw()/torque() for guided() and live()."""

    def __init__(self, bus, ids=None):
        self.bus, self.ids = bus, ids or IDS

    def read_raw(self) -> dict:
        r = self.bus.read_positions(list(self.ids.values()))
        return {j: r.get(self.ids[j]) for j in JOINTS}

    def torque(self, on: bool):
        ids = list(self.ids.values())
        if on:                       # goal := present first, or the arm jumps to a stale goal
            self.bus.write_goals(self.bus.read_positions(ids))
        self.bus.torque(ids, on)


def _open_reader(a):
    if getattr(a, "arm", None):
        from real.arm_client import RemoteArm
        r = RemoteArm(a.arm)
        r.connect()
        return r, r.close
    from real.feetech import FeetechBus, mock_so101
    bus = FeetechBus(a.port, ser=mock_so101() if a.port == "mock" else None).open()
    return BusReader(bus), bus.close


def main(argv=None):
    ap = argparse.ArgumentParser(description="SO-101 joint calibration (raw ticks <-> model angles)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, hlp in (("from-servos", "read LeRobot's calibration back from the servo EEPROM"),
                      ("guided", "zero pose + one move per joint + gripper closed/open (torque off)"),
                      ("live", "torque off; print model angles + fixed-jaw tip at 5 Hz (mirror check)")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("--port", help="/dev/ttyACM0 ('mock' = fake bus)")
        p.add_argument("--arm", help="remote://host:port (the Pi arm server) instead of a local port")
        p.add_argument("-o", "--out", default=str(JOINT_CALIB_JSON))
        if name == "live":
            p.add_argument("--calib", default=str(JOINT_CALIB_JSON))
            p.add_argument("--keep-torque", action="store_true")
    f = sub.add_parser("from-lerobot", help="convert a LeRobot calibration JSON")
    f.add_argument("json")
    f.add_argument("--port", help="also compare with the servo EEPROM")
    f.add_argument("-o", "--out", default=str(JOINT_CALIB_JSON))
    s = sub.add_parser("show")
    s.add_argument("path", nargs="?", default=str(JOINT_CALIB_JSON))
    n = sub.add_parser("nominal", help="write the nominal (unmeasured) calibration")
    n.add_argument("-o", "--out", default=str(JOINT_CALIB_JSON))
    a = ap.parse_args(argv)

    if a.cmd == "show":
        print(JointCalib.load(a.path).summary())
        return
    if a.cmd == "nominal":
        c = nominal()
        print(c.summary())
        print("saved", c.save(a.out))
        return c
    if a.cmd == "from-lerobot":
        c = from_lerobot(json.loads(Path(a.json).read_text()))
        if a.port:
            from real.feetech import FeetechBus
            with FeetechBus(a.port) as bus:
                for w in cross_check(c, bus):
                    print("WARNING:", w)
        print(c.summary())
        print("saved", c.save(a.out))
        return c
    if not (a.port or a.arm):
        ap.error("--port or --arm is required")
    reader, close = _open_reader(a)
    try:
        if a.cmd == "from-servos":
            if a.arm:
                c = JointCalib.from_dict(reader.call("calib_from_servos")["calib"])
            else:
                c = from_servos(reader.bus)
        elif a.cmd == "guided":
            c = guided(reader)
        else:
            calib = JointCalib.load_or_nominal(a.calib)
            from real.kin_lite import Chain
            chain = Chain()
            if not a.keep_torque:
                input("torque goes OFF (support the arm), Enter ")
                reader.torque(False)
            try:
                while True:
                    q = calib.raw_to_model(reader.read_raw())
                    tip = chain.point(q)
                    print("model deg " + " ".join(f"{math.degrees(v):7.1f}" for v in q)
                          + "  | fixed-jaw tip mm " + " ".join(f"{v * 1000:6.0f}" for v in tip), flush=True)
                    time.sleep(0.2)
            except KeyboardInterrupt:
                pass
            finally:
                if not a.keep_torque:
                    input("\nhold the arm: torque comes back on at the current pose, Enter ")
                    reader.torque(True)
            return
        print(c.summary())
        if a.arm:
            reader.call("set_calib", calib=c.to_json())
            print("sent to the arm server (it saved it on the Pi and reloaded)")
        print("saved", c.save(a.out))
        return c
    finally:
        close()


if __name__ == "__main__":
    main()
