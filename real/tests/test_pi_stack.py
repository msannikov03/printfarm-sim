"""Feetech driver, joint calibration, pure-Python FK, Pi arm server / client, arm backends (no hardware)."""
import json
import math
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from real import ROOT
from real.arm_client import MoveAborted, RemoteArm, RemoteError, bed_keepout, parse_url
from real.arm_server import (MOCK_REST_Q, ArmController, ArmServer, KeepOut, Limits, MotionError, minjerk,
                             mock_bus_at)
from real.feetech import (INST_SYNC_WRITE, REGISTERS, CommError, FeetechBus, FeetechError, MockSerial, MockServo,
                          checksum, decode, encode, make_packet, mock_so101)
from real.joint_calib import (GRIPPER_CLOSED_RAD, GRIPPER_OPEN_RAD, IDS, JOINTS, TICK_RAD, BusReader, CalibError,
                              JointCalib, from_lerobot, from_servos, guided, nominal)
from real.kin_lite import Chain

FAST = dict(max_speed_deg_s=400.0, max_accel_deg_s2=4000.0, min_duration_s=0.05, settle_s=0.3,
            servo_speed_cap_deg_s=2000.0, servo_acc=254)


# ----------------------------------------------------------------------------- protocol

def test_packets_match_the_manual_examples():
    # Feetech manual examples: PING id 1 = FF FF 01 02 01 FB; READ 2 bytes @0x38 from id 1 = FF FF 01 04 02 38 02 BE
    assert make_packet(1, 0x01).hex(" ") == "ff ff 01 02 01 fb"
    assert make_packet(1, 0x02, (0x38, 2)).hex(" ") == "ff ff 01 04 02 38 02 be"
    assert checksum([1, 2, 1]) == 0xFB
    assert REGISTERS["Present_Position"] == (56, 2) and REGISTERS["Goal_Position"] == (42, 2)
    assert REGISTERS["Torque_Enable"] == (40, 1) and REGISTERS["Homing_Offset"] == (31, 2)


def test_sign_magnitude():
    assert encode("Goal_Position", -100) == 0x8064 and decode("Goal_Position", 0x8064) == -100
    assert encode("Homing_Offset", -1470) == (1470 | 1 << 11) and decode("Homing_Offset", 1470 | 1 << 11) == -1470
    assert decode("Present_Load", 1 << 10 | 250) == -250 and decode("Present_Voltage", 121) == 121
    assert encode("Goal_Position", 3000) == 3000


def test_bus_scan_read_write_sync_and_motion():
    ser = mock_so101({1: 1000, 2: 3000})
    bus = FeetechBus("mock", ser=ser).open()
    assert bus.scan(range(1, 8)) == {i: 777 for i in range(1, 7)}
    assert bus.read_positions(range(1, 7)) == {1: 1000, 2: 3000, 3: 2048, 4: 2048, 5: 2048, 6: 2048}
    assert bus.torque(range(1, 7), True) == {i: 1 for i in range(1, 7)}
    bus.set_profile([1], speed=2000, acc=100)
    bus.write_goals({1: 1200})
    assert any(p[4] == INST_SYNC_WRITE for p in ser.log)
    t_end = time.monotonic() + 2.0
    while bus.read_positions([1])[1] != 1200 and time.monotonic() < t_end:
        time.sleep(0.01)
    assert bus.read_positions([1])[1] == 1200
    h = bus.read_health([1, 2])
    assert h[2]["volts"] == pytest.approx(12.1) and h[2]["temp_c"] == 31 and h[2]["flags"] == []
    assert bus.torque([1, 2], False) == {1: 0, 2: 0}
    with pytest.raises(FeetechError, match="EEPROM"):
        bus.write_reg(1, "Min_Position_Limit", 10)         # the driver never rewrites the servo config
    with pytest.raises(FeetechError, match="EEPROM"):
        bus.sync_write_reg("ID", {1: 9})


def test_bus_retries_echo_and_missing_servo():
    ser = mock_so101(echo=True, drop_every=4, corrupt_every=7)
    bus = FeetechBus("mock", ser=ser, timeout_s=0.01, retries=3).open()
    for _ in range(20):
        assert bus.read_positions(range(1, 7)) == {i: 2048 for i in range(1, 7)}
        assert bus.read_reg(3, "Model_Number") == 777
    assert bus.stats["echo"] > 0 and bus.stats["retries"] > 0 and bus.stats["corrupt"] > 0
    del ser.servos[4]                                        # unplugged servo
    assert 4 not in bus.read_positions(range(1, 7))
    assert not bus.ping(4)
    with pytest.raises(CommError, match="id 4"):
        bus.read_reg(4, "Model_Number")


def test_bus_through_pyserial_on_a_pty():
    """The real pyserial path (serial.Serial, in_waiting, read timeouts) against simulated servos behind a pty."""
    import os
    import pty
    import select
    pytest.importorskip("serial")
    master, slave = pty.openpty()
    name = os.ttyname(slave)
    mock = mock_so101({3: 1234})
    stop = threading.Event()

    def pump():
        buf = b""
        while not stop.is_set():
            r, _, _ = select.select([master], [], [], 0.01)
            if not r:
                continue
            buf += os.read(master, 4096)
            while True:                                   # hand whole instruction packets to the mock
                i = buf.find(b"\xff\xff")
                if i < 0 or len(buf) < i + 4:
                    break
                n = buf[i + 3] + 4
                if len(buf) < i + n:
                    break
                mock.write(buf[i:i + n])
                buf = buf[i + n:]
                out = mock.read(4096)
                if out:
                    os.write(master, out)
    th = threading.Thread(target=pump, daemon=True)
    th.start()
    try:
        bus = FeetechBus(name, baudrate=115200, timeout_s=0.1).open()     # a pty ignores the (non-standard) 1 Mbps
        assert bus.scan() == {i: 777 for i in range(1, 7)}
        assert bus.read_positions(range(1, 7))[3] == 1234
        bus.torque([3], True)
        bus.write_goals({3: 1300})
        time.sleep(0.3)
        assert abs(bus.read_positions([3])[3] - 1300) <= 1
        assert bus.read_health([1])[1]["volts"] == pytest.approx(12.1)
        bus.close()
    finally:
        stop.set()
        th.join(1)
        os.close(master)
        os.close(slave)


def test_homing_offset_semantics():
    """Present_Position = Actual_Position - Homing_Offset (LeRobot feetech.py), both in the servo and the mock."""
    s = MockServo(1, actual=2500, homing_offset=-300, lo=900, hi=3100)
    bus = FeetechBus("mock", ser=MockSerial([s])).open()
    assert bus.read_reg(1, "Present_Position") == 2800
    assert bus.read_eeprom_calibration([1]) == {1: dict(homing_offset=-300, range_min=900, range_max=3100)}


# ----------------------------------------------------------------------------- kinematics

def test_kin_lite_matches_mujoco():
    from real.so101 import FIXED_TIP, SO101
    c, k = Chain(), SO101()
    lo, hi = c.lo_hi()
    rng = np.random.default_rng(1)
    for _ in range(100):
        q = rng.uniform(lo, hi)
        assert np.allclose(c.point(q), k.fk(q, FIXED_TIP)[0], atol=1e-9)
        k._set(q)
        assert np.allclose(c.point(q, "moving_tip"), k.data.geom_xpos[k.model.geom("moving_jaw_sph_tip1").id], atol=1e-9)
    assert c.ranges == {j: tuple(k.model.joint(j).range) for j in JOINTS}


def test_plus_direction_texts_match_the_model():
    """The words used in the guided calibration / runbook are what the model does for +q."""
    from real.so101 import SO101
    c, k = Chain(), SO101()
    q0 = np.zeros(6)
    tip0 = np.array(c.point(q0))

    def tip(j, d=0.3):
        q = q0.copy()
        q[JOINTS.index(j)] += d
        return np.array(c.point(q))
    assert tip("shoulder_pan")[1] < tip0[1] - 0.05            # + swings to the arm's right (-y)
    assert tip("shoulder_lift")[2] < tip0[2] - 0.05           # + tips the upper arm forward / down
    assert tip("elbow_flex")[2] < tip0[2] - 0.05              # + forearm down
    assert tip("wrist_flex")[2] < tip0[2] - 0.02              # + gripper down
    _, jaw = k.tool_axes(np.r_[0, 0, 0, 0, 0.4, 0])
    assert jaw[1] > 0.2                                       # + roll: top of the gripper to the left (seen from behind)
    assert k.jaw_opening_mm(1.0) > k.jaw_opening_mm(0.0)      # + gripper opens
    f = c.frames(q0)                                          # ZERO pose: upper arm ~vertical, forearm ~horizontal
    up = np.array(f["lower_arm"][0]) - np.array(f["upper_arm"][0])
    fore = np.array(f["wrist"][0]) - np.array(f["lower_arm"][0])
    assert abs(up[2]) > 3 * abs(up[0]) and abs(fore[0]) > 10 * abs(fore[2])
    assert c.point(q0, "moving_tip")[2] > c.point(q0, "fixed_tip")[2]     # fixed jaw at the bottom


# ----------------------------------------------------------------------------- joint calibration

LR_CAL = {"shoulder_pan": dict(id=1, drive_mode=0, homing_offset=-1470, range_min=758, range_max=3292),
          "shoulder_lift": dict(id=2, drive_mode=0, homing_offset=128, range_min=820, range_max=3240),
          "elbow_flex": dict(id=3, drive_mode=0, homing_offset=-911, range_min=870, range_max=3090),
          "wrist_flex": dict(id=4, drive_mode=0, homing_offset=55, range_min=880, range_max=3170),
          "wrist_roll": dict(id=5, drive_mode=0, homing_offset=-200, range_min=0, range_max=4095),
          "gripper": dict(id=6, drive_mode=0, homing_offset=1407, range_min=2020, range_max=3400)}


def _lerobot_reference(raw: dict) -> list:
    """LeRobot v0.6.1 _normalize (DEGREES body, RANGE_0_100 gripper) + real/arm.py JointMap defaults."""
    out = []
    for j in JOINTS[:5]:
        c = LR_CAL[j]
        out.append(math.radians((raw[j] - (c["range_min"] + c["range_max"]) / 2) * 360 / 4095))
    c = LR_CAL["gripper"]
    v = (min(c["range_max"], max(c["range_min"], raw["gripper"])) - c["range_min"]) / (c["range_max"] - c["range_min"]) * 100
    out.append(GRIPPER_CLOSED_RAD + (GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD) * v / 100)
    return out


def test_from_lerobot_reproduces_lerobot_units():
    c = from_lerobot(LR_CAL, joint_map=({}, {}))
    rng = np.random.default_rng(2)
    for _ in range(50):
        raw = {j: int(rng.integers(LR_CAL[j]["range_min"], LR_CAL[j]["range_max"] + 1)) for j in JOINTS}
        assert np.allclose(c.raw_to_model(raw), _lerobot_reference(raw), atol=1e-9)
        q = c.raw_to_model(raw)
        back = c.model_to_raw(q)
        assert all(abs(back[j] - raw[j]) <= 1 for j in JOINTS)
    # drive_mode 1 on the gripper flips it; a JointMap sign/offset is folded in
    flipped = dict(LR_CAL, gripper=dict(LR_CAL["gripper"], drive_mode=1))
    g = from_lerobot(flipped, joint_map=({}, {})).raw_to_model({**{j: 2048 for j in JOINTS}, "gripper": 3400})[5]
    assert g == pytest.approx(GRIPPER_CLOSED_RAD)
    jm = from_lerobot(LR_CAL, joint_map=({"elbow_flex": -1.0}, {"wrist_flex": 7.5}))
    mid = {j: (LR_CAL[j]["range_min"] + LR_CAL[j]["range_max"]) / 2 for j in JOINTS}
    q = jm.raw_to_model(mid)
    assert q[2] == pytest.approx(0) and math.degrees(q[3]) == pytest.approx(-7.5, abs=0.01)
    raw = dict(mid, elbow_flex=mid["elbow_flex"] + 100)
    assert jm.raw_to_model(raw)[2] == pytest.approx(-100 * 2 * math.pi / 4095)
    with pytest.raises(CalibError, match="outside"):
        c.model_to_raw([3.0, 0, 0, 0, 0, 0], clamp=False)


def test_from_servos_reads_the_lerobot_calibration_back(tmp_path):
    servos = [MockServo(LR_CAL[j]["id"], actual=2048 + LR_CAL[j]["homing_offset"], homing_offset=LR_CAL[j]["homing_offset"],
                        lo=LR_CAL[j]["range_min"], hi=LR_CAL[j]["range_max"]) for j in JOINTS]
    bus = FeetechBus("mock", ser=MockSerial(servos)).open()
    c = from_servos(bus)
    ref = from_lerobot(LR_CAL)
    assert c.method == "lerobot-eeprom"
    for j in JOINTS:
        for k in ("zero_raw", "sign", "scale", "min_raw", "max_raw"):
            assert c.joints[j][k] == pytest.approx(ref.joints[j][k])
    c.save(tmp_path / "jc.json")
    assert JointCalib.load(tmp_path / "jc.json").joints == c.joints
    with pytest.raises(CalibError, match="not calibrated"):
        from_servos(FeetechBus("mock", ser=mock_so101()).open())          # factory servos


class _Hand:
    """Simulated person for the guided procedure: moves mock servos when prompted."""

    def __init__(self, ser, signs, zero, swap=None):
        self.ser, self.signs, self.zero, self.swap = ser, signs, zero, swap or {}
        self.prompts = []

    def __call__(self, msg):
        self.prompts.append(msg)
        sv = self.ser.servos
        if "ZERO pose" in msg:
            for j in JOINTS[:5]:
                sv[IDS[j]].actual = self.zero[j]
        for j in JOINTS[:5]:
            if f"Move ONLY {j} " in msg:
                sid = IDS[self.swap.get(j, j)]
                sv[sid].actual += self.signs[j] * 400        # ~35 deg in the + direction
        if "Close the gripper" in msg:
            sv[6].actual = 1500
        if "Open the gripper" in msg:
            sv[6].actual = 1500 + 1200 * self.signs["gripper"]
        return ""


def test_guided_calibration_recovers_zero_and_signs():
    ser = mock_so101()
    bus = FeetechBus("mock", ser=ser).open()
    signs = {"shoulder_pan": 1, "shoulder_lift": -1, "elbow_flex": 1, "wrist_flex": -1, "wrist_roll": 1, "gripper": 1}
    zero = {"shoulder_pan": 2100, "shoulder_lift": 1900, "elbow_flex": 2300, "wrist_flex": 1800, "wrist_roll": 2048}
    hand = _Hand(ser, signs, zero)
    c = guided(BusReader(bus), input_fn=hand, print_fn=lambda *a: None)
    for j in JOINTS[:5]:
        assert c.joints[j]["zero_raw"] == pytest.approx(zero[j]) and c.joints[j]["sign"] == signs[j]
    assert c.raw_to_model({**zero, "gripper": 1500}) == pytest.approx([0, 0, 0, 0, 0, GRIPPER_CLOSED_RAD], abs=1e-3)
    assert c.joints["gripper"]["min_raw"] == 1500 and c.joints["gripper"]["max_raw"] == 2700
    lo, hi = c.model_limits()
    assert lo[1] == pytest.approx(-1.7453293, abs=2 * TICK_RAD) and hi[1] == pytest.approx(1.7453293, abs=2 * TICK_RAD)
    assert bus.torque(range(1, 7), True)        # torque was restored at the end
    # swapped motor ids are caught
    ser2 = mock_so101()
    hand2 = _Hand(ser2, signs, zero, swap={"elbow_flex": "wrist_flex"})
    with pytest.raises(CalibError, match="ids"):
        guided(BusReader(FeetechBus("mock", ser=ser2).open()), input_fn=hand2, print_fn=lambda *a: None)


# ----------------------------------------------------------------------------- controller

@pytest.fixture
def ctrl():
    c = nominal()
    bus = mock_bus_at(MOCK_REST_Q, c)
    ct = ArmController(bus, c, limits=Limits(**FAST), allow_nominal=True, heartbeat_s=None).connect()
    yield ct
    ct.close()


def test_minjerk_profile():
    assert minjerk(0) == 0 and minjerk(1) == 1 and minjerk(0.5) == pytest.approx(0.5)
    s = np.array([minjerk(t) for t in np.linspace(0, 1, 2001)])
    v = np.diff(s) * 2000
    assert v.max() == pytest.approx(1.875, rel=1e-3) and v[0] < 1e-4 and v[-1] < 1e-4


def test_controller_move_respects_speed_and_reaches(ctrl):
    ctrl.lim.max_speed_deg_s, ctrl.lim.max_accel_deg_s2 = 60.0, 400.0
    q0 = ctrl.read_q()
    q1 = [0.3, -0.9, 1.0, 0.8, 0.2, 0.5]
    T, _ = ctrl.plan(q0, q1)
    d = max(abs(math.degrees(a - b)) for a, b in zip(q0[:5], q1[:5]))
    dg = abs(math.degrees(q1[5] - q0[5]))
    assert T == pytest.approx(max(1.875 * d / 60.0, math.sqrt(5.7735 * d / 400.0), 1.875 * dg / 90.0), rel=1e-6)
    ser = ctrl.bus.ser
    n0 = len(ser.log)
    t = time.monotonic()
    r = ctrl.move(q1, wait=True)
    assert r["state"] == "done" and r["reached"] and time.monotonic() - t >= T * 0.95
    assert np.allclose(ctrl.read_q()[:5], q1[:5], atol=math.radians(1.5))
    # the streamed goals never step more than the speed limit allows per tick (+ timing jitter)
    goals = [p for p in ser.log[n0:] if p[4] == INST_SYNC_WRITE and p[5] == REGISTERS["Goal_Position"][0]]
    pan = [p[8] | p[9] << 8 for p in goals]           # first id block: id 1 = shoulder_pan
    assert len(goals) > 10 and max(abs(np.diff(pan))) * 360 / 4096 <= 60.0 / 50 * 3


def test_controller_refuses_limits_keepout_and_nominal(ctrl):
    q0 = ctrl.read_q()
    with pytest.raises(MotionError, match="joint limits"):
        ctrl.move([2.5, 0, 0, 0, 0, 0])
    with pytest.raises(MotionError, match="6 finite"):
        ctrl.move([0, 0, float("nan"), 0, 0, 0])
    # a bed 100 mm above the base plate, 150..370 mm in front of the base: a target 40 mm into it is refused
    R = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]                      # bed x = robot -y, bed y = robot x
    ctrl.set_keepout(bed=dict(R=R, t=[0.15, 0.11, 0.10], size_mm=[220, 220], margin_mm=8))
    from real.so101 import FIXED_TIP, SO101
    k = SO101()
    r = k.solve_hover(np.array([0.26, 0.0, 0.06]), tcp=FIXED_TIP)
    assert r.ok
    with pytest.raises(MotionError, match="keep-out.*above the bed"):
        ctrl.move(list(r.q))
    r2 = k.solve_hover(np.array([0.26, 0.0, 0.16]), tcp=FIXED_TIP)
    assert ctrl.move(list(r2.q), wait=True)["state"] == "done"     # 60 mm above the bed: fine
    assert ctrl.read_q() != q0
    ctrl.allow_nominal = False
    with pytest.raises(MotionError, match="NOMINAL"):
        ctrl.move(list(r2.q))


class VirtualClock:
    """Shared simulated time for the controller and the mock servos: sleep() advances it. The motion thread then
    runs its 50 Hz loop as fast as the CPU allows, so a test cannot depend on how busy the machine is."""

    def __init__(self):
        self.t = 0.0
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            return self.t

    def sleep(self, dt):
        with self.lock:
            self.t += max(float(dt), 0.0)


def test_watchdog_stops_and_holds():
    vc = VirtualClock()
    c = nominal()
    ct = ArmController(mock_bus_at(MOCK_REST_Q, c, clock=vc), c, limits=Limits(**dict(FAST, max_speed_deg_s=20.0)),
                       allow_nominal=True, heartbeat_s=1.0, clock=vc, sleep=vc.sleep).connect()
    try:
        t0 = vc()
        ct.move([0.6, -1.2, 1.2, 1.0, 0.0, 0.0])                   # 34 deg at 20 deg/s: 3.2 s > watchdog 1 s
        r = ct.wait(30.0)                                           # wall-clock bound only (virtual time inside)
        assert r["state"] == "aborted" and "watchdog" in r["reason"]
        assert 1.0 < vc() - t0 < 1.1                                # stopped one control tick after the 1 s silence
        vc.sleep(0.05)                                              # the servos brake (a few ms at 20 deg/s)
        a = ct.read_raw()
        vc.sleep(0.3)
        assert ct.read_raw() == a                                   # holds where it stopped
        assert 0.03 < ct.read_q()[0] < 0.5                          # moved, and stopped well short of the 0.6 target
    finally:
        ct.close()


def test_stall_detection(ctrl):
    ctrl.lim.max_speed_deg_s, ctrl.lim.stall_deg, ctrl.lim.stall_s = 90.0, 8.0, 0.1
    ctrl.bus.ser.servos[1].stall_at = 2048 + 120                      # an obstacle 10.5 deg into the pan move
    r = ctrl.move([0.6, -1.2, 1.2, 1.0, 0.0, 0.0], wait=True)
    assert r["state"] == "aborted" and "stall" in r["reason"]


def test_estop_and_resume(ctrl):
    ctrl.move([0.2, -1.2, 1.2, 1.0, 0.0, 0.0], wait=True)
    st = ctrl.estop()
    assert set(st["torque"].values()) == {0} and ctrl.estopped
    with pytest.raises(MotionError, match="E-STOPPED"):
        ctrl.move([0.0, -1.2, 1.2, 1.0, 0.0, 0.0])
    ctrl.bus.ser.servos[2].actual -= 200                              # the arm sagged while limp
    sagged = ctrl.read_raw()
    ctrl.torque(True)                                                 # goal := present: no jump back
    time.sleep(0.2)
    assert abs(ctrl.read_raw()["shoulder_lift"] - sagged["shoulder_lift"]) <= 1
    assert ctrl.move([0.0, -1.2, 1.2, 1.0, 0.0, 0.0], wait=True)["state"] == "done"


# ----------------------------------------------------------------------------- server + client

@pytest.fixture
def server():
    c = nominal()
    ct = ArmController(mock_bus_at(MOCK_REST_Q, c), c, limits=Limits(**FAST), allow_nominal=True,
                       heartbeat_s=0.5).connect()
    srv = ArmServer(ct, "127.0.0.1", 0, token="t0k", mock=True).start_background()
    yield srv
    srv.shutdown()
    ct.close()


def test_parse_url():
    assert parse_url("remote://10.0.0.2:9000?token=abc") == ("10.0.0.2", 9000, "abc")
    assert parse_url("pi.local") == ("pi.local", 8765, None)


def test_server_client_roundtrip(server):
    url = f"remote://127.0.0.1:{server.port}?token=t0k"
    with pytest.raises(RemoteError, match="unauthorised"):
        RemoteArm(f"remote://127.0.0.1:{server.port}?token=wrong").connect()
    a = RemoteArm(url).connect()
    try:
        assert a.info["in_control"] and a.info["mock"] and a.info["found"] == {str(i): 777 for i in range(1, 7)}
        r = a.move([0.3, -1.0, 1.1, 1.0, 0.1, 0.4], speed_deg_s=120)
        assert r["state"] == "done" and r["reached"]
        assert np.allclose(a.read_q()[:5], [0.3, -1.0, 1.1, 1.0, 0.1], atol=math.radians(1.5))
        with pytest.raises(RemoteError, match="joint limits"):
            a.move([0, 0, 0, 0, 3.5, 0])
        assert a.gripper(open_pct=50)["state"] == "done"
        # a second client is read-only, but may read and e-stop
        b = RemoteArm(url, takeover=False).connect()
        assert not b.info["in_control"] and len(b.read_q()) == 6
        with pytest.raises(RemoteError, match="read-only"):
            b.move(a.read_q())
        b.estop()
        with pytest.raises(RemoteError, match="E-STOPPED"):
            a.move(a.read_q())
        assert set(a.torque(True).values()) == {1}
        b.close()
        s = a.status()
        assert s["torque"]["gripper"] == 1 and s["health"]["elbow_flex"]["volts"] > 11
    finally:
        a.close()


def test_owner_disconnect_mid_move_holds(server):
    url = f"remote://127.0.0.1:{server.port}?token=t0k"
    server.ctrl.lim.max_speed_deg_s = 10.0
    a = RemoteArm(url, heartbeat_s=0).connect()                     # no heartbeat thread
    a.move([0.8, -1.2, 1.2, 1.0, 0.0, 0.0], wait=False)
    time.sleep(0.2)
    a.close()                                                        # client vanishes
    r = server.ctrl.wait(3.0)
    assert r["state"] == "aborted" and ("disconnected" in r["reason"] or "watchdog" in r["reason"])
    q = server.ctrl.read_q()[0]
    time.sleep(0.3)
    assert abs(server.ctrl.read_q()[0] - q) < math.radians(0.2) and q < 0.8 - 0.1


def test_client_heartbeat_keeps_long_move_alive(server):
    url = f"remote://127.0.0.1:{server.port}?token=t0k"
    server.ctrl.lim.max_speed_deg_s = 25.0
    a = RemoteArm(url, heartbeat_s=0.1).connect()
    try:
        r = a.move([0.5, -1.2, 1.2, 1.0, 0.0, 0.0])                  # ~1.5 s > the 0.5 s watchdog
        assert r["state"] == "done"
    finally:
        a.close()


# ----------------------------------------------------------------------------- arm backends + demo

def test_arm_backend_parsing():
    from real.arm import arm_backend
    assert arm_backend("remote://1.2.3.4:8765") == ("remote", "remote://1.2.3.4:8765")
    assert arm_backend("feetech:///dev/ttyACM0") == ("feetech", "/dev/ttyACM0")
    assert arm_backend("feetech://mock") == ("feetech", "mock")
    assert arm_backend("/dev/tty.usbmodem1") == ("lerobot", "/dev/tty.usbmodem1")
    assert arm_backend("lerobot:///dev/tty.usbmodem1") == ("lerobot", "/dev/tty.usbmodem1")


def test_arm_feetech_mock_backend():
    from real.arm import Arm, Safety
    arm = Arm("feetech://mock", dry_run=False, confirm=False, safety=Safety(max_speed_deg_s=120, max_accel_deg_s2=2000))
    arm.connect()
    try:
        arm.drv.lim.__dict__.update(FAST)
        q = np.array([0.2, -1.0, 1.0, 1.1, 0.0, 0.0])
        arm.move_to(q, "t")
        assert np.allclose(arm.q[:5], q[:5], atol=math.radians(1.5))
        assert np.allclose(arm.read_joints_rad()[:5], q[:5], atol=math.radians(1.5))
        arm.gripper(50)
        assert arm.q[5] > 0.5
    finally:
        arm.disconnect()


def test_run_demo_live_through_remote_server(server, tmp_path, monkeypatch):
    from real.run_demo import main
    monkeypatch.setattr("builtins.input", lambda msg="": "go")
    rep = main(["--mock", "--mock-print-s", "0.3", "--mock-tau-s", "0.3", "--live", "--allow-nominal",
                "--arm", f"remote://127.0.0.1:{server.port}?token=t0k", "--film", "--speed", "200", "--accel", "3000",
                "--dwell-s", "0", "--runs-dir", str(tmp_path), "--calib", str(tmp_path / "none.json")])
    assert [r["status"] for r in rep["results"]] == ["moved"] * 3
    ko = server.ctrl.keepout.bed
    assert ko is not None and ko["size_mm"] == [220.0, 220.0]          # the bed keep-out was pushed to the server
    assert np.allclose(server.ctrl.read_q()[:5], MOCK_REST_Q[:5], atol=math.radians(1.5))   # back at the start pose


def test_run_demo_dry_run_pick(tmp_path):
    from real.run_demo import main
    from real.synth_gcode import make_gcode
    g = tmp_path / "p.gcode"
    g.write_text(make_gcode())
    rep = main(["--gcode", str(g), "--dry-run", "--runs-dir", str(tmp_path), "--calib", str(tmp_path / "none.json"),
                "--pick", "first", "--box-mm=-40,40,100"])
    pick = [r for r in rep["results"] if "pick" in r]
    assert len(pick) == 1 and pick[0]["status"] == "dry-run ok", pick


def test_hello_arm_mock():
    from real.hello_arm import main
    assert main(["--port", "mock"]) == 0
    assert main(["--port", "mock", "--go", "--speed", "90", "--amp", "8"]) == 0


def test_pi_scripts_parse_and_pi_modules_need_no_numpy():
    for sh in sorted((ROOT / "pi").glob("*.sh")):
        subprocess.run(["bash", "-n", str(sh)], check=True)
    # the Pi side must import with the standard library only (+ pyserial, lazily)
    code = ("import sys; sys.modules['numpy'] = None; sys.modules['mujoco'] = None; sys.modules['serial'] = None\n"
            "import real.feetech, real.joint_calib, real.kin_lite, real.arm_server, real.arm_client, real.hello_arm\n"
            "import real.replay, real.printer, real.printer.creality_ws, real.printer.moonraker, real.discover\n"
            "from real.arm_server import main\n"
            "print('ok')")
    out = subprocess.run([__import__("sys").executable, "-c", code], cwd=ROOT.parent, capture_output=True, text=True)
    assert out.stdout.strip() == "ok", out.stderr


def test_server_cli_mock_end_to_end(tmp_path):
    """python -m real.arm_server --mock in a subprocess, the CLI client talks to it."""
    import socket
    import sys
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    env = dict(__import__("os").environ, PYTHONPATH=str(ROOT.parent))
    p = subprocess.Popen([sys.executable, "-m", "real.arm_server", "--mock", "--mock-nominal", "--listen",
                          f"127.0.0.1:{port}", "--calib", str(tmp_path / "jc.json")], env=env, cwd=ROOT.parent,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        t_end = time.monotonic() + 15
        while time.monotonic() < t_end:
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        out = subprocess.run([sys.executable, "-m", "real.arm_client", f"remote://127.0.0.1:{port}", "jog",
                              "shoulder_pan", "5", "--speed", "30"], env=env, cwd=ROOT.parent, capture_output=True,
                             text=True, timeout=30)
        assert "'state': 'done'" in out.stdout, out.stdout + out.stderr
        out = subprocess.run([sys.executable, "-m", "real.arm_client", f"remote://127.0.0.1:{port}", "status"],
                             env=env, cwd=ROOT.parent, capture_output=True, text=True, timeout=30)
        assert "shoulder_pan" in out.stdout and "12.1 V" in out.stdout, out.stdout + out.stderr
    finally:
        p.terminate()
        p.wait(5)


def test_replay_on_the_pi_from_a_dry_run_log(tmp_path, monkeypatch):
    """Plan on the Mac (dry run) -> replay next to the arm without MuJoCo (tailnet-down fallback)."""
    import real.arm_server as srv
    from real.replay import main as replay
    from real.replay import sequence
    from real.run_demo import main as demo
    from real.synth_gcode import make_gcode
    g = tmp_path / "p.gcode"
    g.write_text(make_gcode())
    demo(["--gcode", str(g), "--dry-run", "--runs-dir", str(tmp_path), "--calib", str(tmp_path / "none.json")])
    log_path = str(tmp_path / "real_demo_log.json")
    log = json.loads(Path(log_path).read_text())
    seq = sequence(log, dwell_s=0.0, gripper=True)
    assert [s[0] for s in seq].count("ready pose") == 2 and sum(s[0].startswith("hover ") for s in seq) == 3
    assert replay([log_path]) == 0                                     # print only
    fast = srv.Limits
    monkeypatch.setattr(srv, "Limits", lambda **kw: fast(**FAST))
    assert replay([log_path, "--port", "mock", "--go", "--film", "--dwell-s", "0", "--speed", "200"]) == 0
