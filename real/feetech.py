"""Minimal Feetech STS3215 bus driver for the SO-101 (pyserial only, no LeRobot, no torch).

    bus = FeetechBus("/dev/ttyACM0")          # 1 Mbps, the SO-101 default
    bus.open()
    bus.scan()                                # {1: 777, 2: 777, ...} id -> model number (777 = STS3215)
    pos = bus.read_positions([1, 2, 3, 4, 5, 6])          # raw ticks, 0..4095
    bus.set_profile([1, 2, 3], speed=600, acc=20)          # goal speed (steps/s) + acceleration (100 steps/s^2)
    bus.write_goals({1: 2048, 2: 2100})                    # one SYNC WRITE packet
    bus.torque([1, 2, 3, 4, 5, 6], False)                  # limp (the arm sags!)

Protocol (Feetech "FT-SCS", protocol 0 = STS/SMS series, little-endian):
    instruction  FF FF  ID  LEN  INSTR  P1..Pn  CHK      LEN = n + 2
    status       FF FF  ID  LEN  ERR    P1..Pn  CHK      CHK = ~(ID + LEN + INSTR/ERR + sum(P)) & 0xFF
    PING 0x01, READ 0x02 (addr, n), WRITE 0x03 (addr, data..), SYNC READ 0x82 (addr, n, id1, id2..; every
    servo answers with its own status packet, in ID order), SYNC WRITE 0x83 (addr, n, [id, data..]..;
    broadcast id 0xFE, no answer). Status error bits: 1 voltage, 2 angle, 4 overheat, 8 overcurrent,
    32 overload.
Sources (checked 2026-10-22):
  [L1] LeRobot v0.6.1 src/lerobot/motors/feetech/tables.py (STS_SMS_SERIES_CONTROL_TABLE, baud table,
       sign-magnitude bits, MODEL_NUMBER_TABLE sts3215 = 777, resolution 4096)
       https://github.com/huggingface/lerobot/blob/v0.6.1/src/lerobot/motors/feetech/tables.py
  [L2] LeRobot v0.6.1 motors/feetech/feetech.py (Present_Position = Actual_Position - Homing_Offset;
       configure_motors: Return_Delay_Time 0, Acceleration 254; torque: Torque_Enable then Lock) and
       motors/motors_bus.py (DEGREES = (raw - (min+max)/2) * 360 / 4095; RANGE_0_100 for the gripper)
  [L3] LeRobot v0.6.1 tests/mocks/mock_feetech.py (instruction / status packet layouts, citing the Waveshare
       "Communication_Protocol_User_Manual-EN(191218-0923).pdf")
  [S1] feetech-servo-sdk 1.0.0 (PyPI, module scservo_sdk): scservo_def.py (INST_* codes, BROADCAST_ID 0xFE,
       SCS_TOHOST/SCS_TOSCS sign-magnitude) and protocol_packet_handler.py (checksum, ERRBIT_* values)
  [F1] Feetech SMS/STS memory table (doc.feetech.cn FT-SMS-STS-emanual, as summarised at
       https://aifitlab-wiki.super.site/feetech-servo-motor-docs/feetech-sts-servo-memory-table-analysis):
       units below (0.087 deg/tick, Acceleration 100 steps/s^2 = 8.7 deg/s^2, voltage 0.1 V, load 0.1 %,
       current 6.5 mA, Torque_Enable 128 = "set the current position as 2048" (EEPROM! never used here)).
UNITS NOT VERIFIED ON HARDWARE: goal speed is 1 step/s per count (0.0146 rpm) in the default Phase setting
per [F1]; the alternative Phase bit2 unit is 50 steps/s (0.732 rpm). `python -m real.feetech watch` prints
Present_Velocity while you move a joint so you can check.

This module never writes EEPROM registers (ID, baud, limits, offsets): it only touches SRAM
(Torque_Enable, Acceleration, Goal_Position, Goal_Velocity, Torque_Limit), so it cannot undo a LeRobot
calibration. It has no third-party dependency besides pyserial (imported lazily: tests use MockSerial).
"""
from __future__ import annotations

import argparse
import logging
import math
import threading
import time

log = logging.getLogger("real.feetech")

# --------------------------------------------------------------------------- protocol constants [S1]
INST_PING, INST_READ, INST_WRITE, INST_REG_WRITE, INST_ACTION = 0x01, 0x02, 0x03, 0x04, 0x05
INST_SYNC_READ, INST_SYNC_WRITE = 0x82, 0x83
BROADCAST_ID = 0xFE
MAX_ID = 0xFC
ERR_BITS = {1: "voltage", 2: "angle", 4: "overheat", 8: "overcurrent", 32: "overload"}   # [S1] ERRBIT_*
STATUS_BITS = {1: "voltage", 2: "magnetic encoder", 4: "temperature", 8: "current", 32: "load"}  # reg 65 [F1]

STS3215_MODEL = 777                       # [L1] MODEL_NUMBER_TABLE
RESOLUTION = 4096                         # ticks per turn [L1]; 0.087890625 deg per tick
BAUDRATES = {1_000_000: 0, 500_000: 1, 250_000: 2, 128_000: 3, 115_200: 4, 57_600: 5, 38_400: 6, 19_200: 7}  # [L1]

# name: (address, bytes). [L1] STS_SMS_SERIES_CONTROL_TABLE, cross-checked with [F1].
REGISTERS = {
    # EEPROM (read only here)
    "Firmware_Major_Version": (0, 1), "Firmware_Minor_Version": (1, 1), "Model_Number": (3, 2),
    "ID": (5, 1), "Baud_Rate": (6, 1), "Return_Delay_Time": (7, 1), "Response_Status_Level": (8, 1),
    "Min_Position_Limit": (9, 2), "Max_Position_Limit": (11, 2), "Max_Temperature_Limit": (13, 1),
    "Max_Voltage_Limit": (14, 1), "Min_Voltage_Limit": (15, 1), "Max_Torque_Limit": (16, 2), "Phase": (18, 1),
    "Unloading_Condition": (19, 1), "LED_Alarm_Condition": (20, 1), "P_Coefficient": (21, 1),
    "D_Coefficient": (22, 1), "I_Coefficient": (23, 1), "Minimum_Startup_Force": (24, 2),
    "CW_Dead_Zone": (26, 1), "CCW_Dead_Zone": (27, 1), "Protection_Current": (28, 2),
    "Angular_Resolution": (30, 1), "Homing_Offset": (31, 2), "Operating_Mode": (33, 1),
    "Protective_Torque": (34, 1), "Protection_Time": (35, 1), "Overload_Torque": (36, 1),
    # SRAM
    "Torque_Enable": (40, 1),     # 0 off, 1 on, 2 damping; 128 = recentre (EEPROM write: never sent here)
    "Acceleration": (41, 1),      # 100 steps/s^2 per count (8.7 deg/s^2), 0 = maximum   [F1]
    "Goal_Position": (42, 2),     # ticks, sign-magnitude bit 15
    "Goal_Time": (44, 2),
    "Goal_Velocity": (46, 2),     # steps/s (0 = maximum speed), sign-magnitude bit 15   [F1]
    "Torque_Limit": (48, 2),      # 0.1 % (1000 = 100 %), copied from Max_Torque_Limit at power-on [F1]
    "Lock": (55, 1),
    "Present_Position": (56, 2), "Present_Velocity": (58, 2), "Present_Load": (60, 2),
    "Present_Voltage": (62, 1),   # 0.1 V
    "Present_Temperature": (63, 1),   # deg C
    "Status": (65, 1), "Moving": (66, 1),
    "Present_Current": (69, 2),   # 6.5 mA per count
}
SIGN_BITS = {"Homing_Offset": 11, "Goal_Position": 15, "Goal_Velocity": 15, "Present_Position": 15,
             "Present_Velocity": 15, "Present_Load": 10}           # [L1] STS_SMS_SERIES_ENCODINGS_TABLE
EEPROM_END = 40                    # addresses < 40 are EEPROM: refused by write_reg unless allow_eeprom=True
HEALTH_ADDR, HEALTH_LEN = 56, 15   # one block read: position .. current (56..70)


class FeetechError(RuntimeError):
    pass


class CommError(FeetechError):
    """No (valid) answer: timeout, checksum, wrong id. Retried."""


class ServoError(FeetechError):
    """The servo answered with a hardware error flag (overload, overheat, ...)."""


def checksum(body) -> int:
    """~(sum of ID, LEN, INSTR/ERR, params) & 0xFF   [S1] protocol_packet_handler.txPacket"""
    return (~sum(body)) & 0xFF


def make_packet(sid: int, instr: int, params=()) -> bytes:
    body = [sid & 0xFF, len(params) + 2, instr & 0xFF, *[p & 0xFF for p in params]]
    return bytes([0xFF, 0xFF, *body, checksum(body)])


def encode(name: str, value: int) -> int:
    """Signed int -> register value (sign-magnitude for the registers in SIGN_BITS) [S1] SCS_TOSCS."""
    value = int(round(value))
    bit = SIGN_BITS.get(name)
    if bit is not None and value < 0:
        return (-value) | (1 << bit)
    return value


def decode(name: str, raw: int) -> int:
    """Register value -> signed int [S1] SCS_TOHOST."""
    bit = SIGN_BITS.get(name)
    if bit is not None and raw & (1 << bit):
        return -(raw & ((1 << bit) - 1))
    return raw


def to_bytes(value: int, n: int) -> list[int]:
    return [(value >> (8 * i)) & 0xFF for i in range(n)]          # little-endian (STS: END = 0)


def from_bytes(data) -> int:
    return sum(b << (8 * i) for i, b in enumerate(data))


def err_text(err: int) -> str:
    return ",".join(v for k, v in ERR_BITS.items() if err & k) or f"0x{err:02x}"


class FeetechBus:
    """Half-duplex TTL bus through the SO-101 USB adapter (CH343, /dev/ttyACM*; or CH340/FTDI /dev/ttyUSB*).

    Thread-safe: every transaction holds a lock. Reads retry `retries` times on timeouts / bad checksums.
    If the adapter echoes our own bytes back (some half-duplex boards do), the echo is recognised and
    skipped. `ser` may be any object with write/read/in_waiting/reset_input_buffer (see MockSerial)."""

    def __init__(self, port: str | None = None, baudrate: int = 1_000_000, timeout_s: float = 0.03,
                 retries: int = 2, ser=None):
        self.port, self.baudrate, self.timeout_s, self.retries = port, baudrate, timeout_s, retries
        self.ser = ser
        self.lock = threading.RLock()
        self.status_level: dict[int, int] = {}       # Response_Status_Level per id (1 = answers writes)
        self.stats = {"tx": 0, "rx": 0, "timeouts": 0, "corrupt": 0, "echo": 0, "retries": 0}
        self._buf = bytearray()

    # ------------------------------------------------------------------ port
    def open(self):
        if self.ser is None:
            try:
                import serial
            except ImportError as e:            # pragma: no cover - depends on the machine
                raise ImportError("pyserial missing: pip install pyserial (or run real/pi/setup_pi.sh)") from e
            self.ser = serial.Serial(self.port, self.baudrate, timeout=0.002, write_timeout=0.1)
        if hasattr(self.ser, "reset_input_buffer"):
            self.ser.reset_input_buffer()
        return self

    def close(self):
        if self.ser is not None and hasattr(self.ser, "close"):
            self.ser.close()
        self.ser = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ framing
    def _tx(self, pkt: bytes):
        self._buf.clear()
        if hasattr(self.ser, "reset_input_buffer"):
            self.ser.reset_input_buffer()              # drop stale bytes from an earlier timed-out answer
        self.ser.write(pkt)
        if hasattr(self.ser, "flush"):
            self.ser.flush()
        self.stats["tx"] += 1
        self._last_tx = pkt

    def _fill(self, t_end: float) -> bool:
        n = getattr(self.ser, "in_waiting", 0) or 1
        chunk = self.ser.read(n)
        if chunk:
            self._buf += chunk
            return True
        return time.monotonic() < t_end

    def _rx(self, t_end: float):
        """Next valid status packet -> (id, err, params bytes). Raises CommError on timeout."""
        buf = self._buf
        while True:
            i = buf.find(b"\xff\xff")
            if i < 0:
                del buf[:max(0, len(buf) - 1)]
            else:
                del buf[:i]
                tx = getattr(self, "_last_tx", None)
                if tx and len(buf) >= len(tx) and buf[:len(tx)] == tx:      # half-duplex echo of our request
                    del buf[:len(tx)]
                    self.stats["echo"] += 1
                    continue
                if len(buf) >= 4 and buf[2] == 0xFF:          # FF FF FF: shift by one
                    del buf[:1]
                    continue
                if len(buf) >= 4:
                    ln = buf[3]
                    if ln < 2 or buf[2] > 0xFD:
                        del buf[:2]
                        self.stats["corrupt"] += 1
                        continue
                    total = ln + 4
                    if len(buf) >= total:
                        pkt = bytes(buf[:total])
                        del buf[:total]
                        if checksum(pkt[2:-1]) != pkt[-1]:
                            self.stats["corrupt"] += 1
                            continue
                        self.stats["rx"] += 1
                        return pkt[2], pkt[4], pkt[5:-1]
            if not self._fill(t_end):
                self.stats["timeouts"] += 1
                raise CommError("timeout")

    def _txrx(self, sid: int, instr: int, params=(), expect_len: int | None = None):
        """One instruction with an answer from `sid`, retried. Returns (err, params)."""
        pkt = make_packet(sid, instr, params)
        last = None
        for attempt in range(self.retries + 1):
            if attempt:
                self.stats["retries"] += 1
            with self.lock:
                self._tx(pkt)
                t_end = time.monotonic() + self.timeout_s
                try:
                    while True:
                        rid, err, data = self._rx(t_end)
                        if rid == sid:
                            break
                    if expect_len is not None and len(data) != expect_len:
                        raise CommError(f"id {sid}: {len(data)} data bytes, expected {expect_len}")
                    return err, data
                except CommError as e:
                    last = e
        raise CommError(f"servo id {sid}: no valid answer to instruction 0x{instr:02x} after "
                        f"{self.retries + 1} tries ({last})")

    @staticmethod
    def _check(sid, err, strict):
        if err and strict:
            raise ServoError(f"servo id {sid} reports {err_text(err)}")
        if err:
            log.warning("servo id %s reports %s", sid, err_text(err))

    # ------------------------------------------------------------------ primitives
    def ping(self, sid: int) -> bool:
        try:
            err, _ = self._txrx(sid, INST_PING)
            return True
        except CommError:
            return False

    def read(self, sid: int, addr: int, n: int, strict: bool = False) -> bytes:
        err, data = self._txrx(sid, INST_READ, (addr, n), expect_len=n)
        self._check(sid, err, strict)
        return data

    def write(self, sid: int, addr: int, data, strict: bool = False):
        params = (addr, *data)
        if sid == BROADCAST_ID or self.status_level.get(sid, 1) == 0:
            with self.lock:
                self._tx(make_packet(sid, INST_WRITE, params))
            return
        err, _ = self._txrx(sid, INST_WRITE, params)
        self._check(sid, err, strict)

    def sync_write(self, addr: int, n: int, values: dict):
        """values: {id: register value (already encoded)}. One broadcast packet, no answer."""
        params = [addr, n]
        for sid, v in values.items():
            params += [sid, *to_bytes(int(v), n)]
        with self.lock:
            self._tx(make_packet(BROADCAST_ID, INST_SYNC_WRITE, params))

    def sync_read(self, addr: int, n: int, ids) -> dict:
        """{id: data bytes}. Servos that do not answer are retried with a plain READ; still missing -> absent."""
        ids = list(ids)
        out: dict[int, bytes] = {}
        for attempt in range(self.retries + 1):
            todo = [i for i in ids if i not in out]
            if not todo:
                break
            if attempt:
                self.stats["retries"] += 1
            with self.lock:
                self._tx(make_packet(BROADCAST_ID, INST_SYNC_READ, [addr, n, *todo]))
                t_end = time.monotonic() + self.timeout_s + 0.0002 * len(todo) * (6 + n)
                try:
                    while len([i for i in todo if i in out]) < len(todo):
                        rid, err, data = self._rx(t_end)
                        if rid in todo and len(data) == n:
                            out[rid] = data
                            self._check(rid, err, False)
                except CommError:
                    pass
        for sid in [i for i in ids if i not in out]:          # last resort: one by one
            try:
                out[sid] = self.read(sid, addr, n)
            except CommError:
                pass
        return out

    # ------------------------------------------------------------------ named registers
    def read_reg(self, sid: int, name: str) -> int:
        addr, n = REGISTERS[name]
        return decode(name, from_bytes(self.read(sid, addr, n)))

    def write_reg(self, sid: int, name: str, value: int, allow_eeprom: bool = False):
        addr, n = REGISTERS[name]
        if addr < EEPROM_END and not allow_eeprom:
            raise FeetechError(f"{name} (address {addr}) is an EEPROM register: refusing to write it "
                               f"(this driver never changes the servo configuration)")
        self.write(sid, addr, to_bytes(encode(name, value), n))

    def sync_read_reg(self, name: str, ids) -> dict:
        addr, n = REGISTERS[name]
        return {sid: decode(name, from_bytes(d)) for sid, d in self.sync_read(addr, n, ids).items()}

    def sync_write_reg(self, name: str, values: dict):
        addr, n = REGISTERS[name]
        if addr < EEPROM_END:
            raise FeetechError(f"{name} is an EEPROM register: refusing to write it")
        self.sync_write(addr, n, {sid: encode(name, v) for sid, v in values.items()})

    # ------------------------------------------------------------------ convenience
    def scan(self, ids=range(1, 7)) -> dict:
        """{id: model number} for every id that answers a PING; also caches Response_Status_Level."""
        found = {}
        for sid in ids:
            if self.ping(sid):
                try:
                    found[sid] = self.read_reg(sid, "Model_Number")
                    self.status_level[sid] = self.read_reg(sid, "Response_Status_Level")
                except CommError:
                    found[sid] = -1
        return found

    def read_positions(self, ids) -> dict:
        return self.sync_read_reg("Present_Position", ids)

    def write_goals(self, goals: dict):
        """{id: raw ticks} in one SYNC WRITE (the servo clamps to its own Min/Max_Position_Limit)."""
        self.sync_write_reg("Goal_Position", {i: int(round(v)) for i, v in goals.items()})

    def set_profile(self, ids, speed: int | None = None, acc: int | None = None):
        """Hardware motion limits used by the servo's own profile generator: goal speed in steps/s
        (0 = unlimited; 1 step = 0.088 deg) and acceleration in units of 100 steps/s^2 (0 = unlimited, 254 max).
        LeRobot sets acc 254 and leaves speed at 0 [L2], so an un-interpolated goal snaps; we cap both."""
        ids = list(ids)
        if acc is not None:
            self.sync_write_reg("Acceleration", {i: int(max(0, min(254, acc))) for i in ids})
        if speed is not None:
            self.sync_write_reg("Goal_Velocity", {i: int(max(0, min(32766, speed))) for i in ids})

    def torque(self, ids, on: bool, verify: bool = True) -> dict:
        """Torque on/off. Broadcast first (one packet reaches every servo even if one is flaky), then each
        servo individually (acknowledged), then read back. Returns {id: state read back (or None)}."""
        ids = list(ids)
        v = 1 if on else 0
        self.sync_write(REGISTERS["Torque_Enable"][0], 1, {i: v for i in ids})
        for sid in ids:
            try:
                self.write_reg(sid, "Torque_Enable", v)
            except CommError as e:
                log.warning("torque %s: %s", "on" if on else "off", e)
        if not verify:
            return {}
        got = self.sync_read_reg("Torque_Enable", ids)
        return {i: got.get(i) for i in ids}

    def read_health(self, ids) -> dict:
        """One SYNC READ of addresses 56..70 -> per id: pos, vel (steps/s), load (%), volts, temp_c, status
        flags, moving, current_ma."""
        out = {}
        for sid, d in self.sync_read(HEALTH_ADDR, HEALTH_LEN, ids).items():
            g = lambda a, n: from_bytes(d[a - HEALTH_ADDR:a - HEALTH_ADDR + n])
            st = g(65, 1)
            out[sid] = dict(pos=decode("Present_Position", g(56, 2)), vel=decode("Present_Velocity", g(58, 2)),
                            load_pct=decode("Present_Load", g(60, 2)) / 10.0, volts=g(62, 1) / 10.0,
                            temp_c=g(63, 1), status=st, flags=[v for k, v in STATUS_BITS.items() if st & k],
                            moving=g(66, 1), current_ma=round(g(69, 2) * 6.5, 1))
        return out

    def read_eeprom_calibration(self, ids) -> dict:
        """What LeRobot's read_calibration() reads [L2]: Homing_Offset, Min/Max_Position_Limit per id."""
        out = {}
        for sid in ids:
            out[sid] = dict(homing_offset=self.read_reg(sid, "Homing_Offset"),
                            range_min=self.read_reg(sid, "Min_Position_Limit"),
                            range_max=self.read_reg(sid, "Max_Position_Limit"))
        return out


# --------------------------------------------------------------------------- mock bus (tests, rehearsals)

class MockServo:
    """Register memory + a simple motion model of one STS3215 (position-mode only)."""

    def __init__(self, sid: int, actual: int = 2048, homing_offset: int = 0, lo: int = 0, hi: int = 4095,
                 max_speed: float = 3000.0):
        self.sid = sid
        self.mem = bytearray(90)
        self.actual = float(actual)          # encoder position before the homing offset
        self.max_speed = max_speed           # steps/s at "speed 0" (no limit); STS3215 ~ 45 rpm unloaded at 12 V
        self.vel = 0.0
        self.hand = None                     # torque off: tests set .actual directly (a hand moving it)
        self.fault = 0                       # error bits returned in status packets
        self.stall_at = None                 # simulated obstacle: actual cannot pass this value
        self._set("Model_Number", STS3215_MODEL)
        self._set("ID", sid)
        self._set("Response_Status_Level", 1)
        self._set("Min_Position_Limit", lo)
        self._set("Max_Position_Limit", hi)
        self._set("Homing_Offset", encode("Homing_Offset", homing_offset))
        self._set("Max_Torque_Limit", 1000)
        self._set("Torque_Limit", 1000)
        self._set("Present_Voltage", 121)
        self._set("Present_Temperature", 31)
        self._set("Firmware_Major_Version", 3)
        self._set("Firmware_Minor_Version", 10)
        self._sync_present()
        self._set("Goal_Position", encode("Goal_Position", self.present()))

    def _set(self, name, v):
        a, n = REGISTERS[name]
        self.mem[a:a + n] = bytes(to_bytes(v, n))

    def get(self, name) -> int:
        a, n = REGISTERS[name]
        return decode(name, from_bytes(self.mem[a:a + n]))

    def present(self) -> int:
        return int(round(self.actual)) - self.get("Homing_Offset")      # [L2] Present = Actual - Homing_Offset

    def _sync_present(self):
        self._set("Present_Position", encode("Present_Position", self.present()))
        self._set("Present_Velocity", encode("Present_Velocity", int(self.vel)))
        self._set("Present_Load", encode("Present_Load", int(min(1000, abs(self.vel) / 3))))
        self._set("Moving", 1 if abs(self.vel) > 1 else 0)
        self._set("Present_Current", int(abs(self.vel) / 30))

    def step(self, dt: float):
        if self.get("Torque_Enable") != 1:
            self.vel = 0.0
            self._sync_present()
            return
        lo, hi = self.get("Min_Position_Limit"), self.get("Max_Position_Limit")
        goal = min(hi, max(lo, self.get("Goal_Position"))) + self.get("Homing_Offset")
        vmax = self.get("Goal_Velocity") or self.max_speed
        acc = self.get("Acceleration") * 100.0 or 1e9
        err = goal - self.actual
        v_stop = math.sqrt(2.0 * min(acc, 1e6) * abs(err))          # brake in time (trapezoidal profile)
        v_des = math.copysign(min(vmax, v_stop, abs(err) / max(dt, 1e-3)), err)
        dv = max(-acc * dt, min(acc * dt, v_des - self.vel))
        self.vel += dv
        new = self.actual + self.vel * dt
        if (err > 0 and new > goal) or (err < 0 and new < goal):
            new, self.vel = goal, 0.0
        if self.stall_at is not None and (self.actual - self.stall_at) * (new - self.stall_at) <= 0 \
                and new != self.stall_at:
            new, self.vel = self.stall_at, 0.0
        self.actual = new
        self._sync_present()


class MockSerial:
    """A pyserial stand-in with STS3215 servos behind it. Answers PING / READ / WRITE / SYNC READ / SYNC WRITE
    exactly like the protocol above; can echo TX bytes, drop or corrupt answers (fault injection).
    Servo motion is integrated in (scaled) wall-clock time on every access."""

    def __init__(self, servos=None, echo: bool = False, drop_every: int = 0, corrupt_every: int = 0,
                 time_scale: float = 1.0, clock=time.monotonic):
        self.servos = {s.sid: s for s in (servos or [MockServo(i) for i in range(1, 7)])}
        self.echo, self.drop_every, self.corrupt_every = echo, drop_every, corrupt_every
        self.time_scale, self.clock = time_scale, clock
        self.rx = bytearray()
        self.log: list[bytes] = []
        self.n_answers = 0
        self.is_open = True
        self.timeout = 0.0
        self._t = clock()
        self.lock = threading.RLock()

    def _advance(self):
        now = self.clock()
        dt = (now - self._t) * self.time_scale
        self._t = now
        while dt > 0:
            h = min(dt, 0.005)
            for s in self.servos.values():
                s.step(h)
            dt -= h

    @property
    def in_waiting(self):
        with self.lock:
            return len(self.rx)

    def reset_input_buffer(self):
        with self.lock:
            self.rx.clear()

    def flush(self):
        pass

    def close(self):
        self.is_open = False

    def read(self, n=1):
        with self.lock:
            out = bytes(self.rx[:n])
            del self.rx[:n]
            return out

    def _answer(self, sid, params=b""):
        self.n_answers += 1
        if self.drop_every and self.n_answers % self.drop_every == 0:
            return
        s = self.servos[sid]
        body = [sid, len(params) + 2, s.fault, *params]
        pkt = bytearray([0xFF, 0xFF, *body, checksum(body)])
        if self.corrupt_every and self.n_answers % self.corrupt_every == 0:
            pkt[-1] ^= 0x5A
        self.rx += pkt

    def write(self, data: bytes):
        with self.lock:
            self._advance()
            data = bytes(data)
            self.log.append(data)
            if self.echo:
                self.rx += data
            if len(data) < 6 or data[:2] != b"\xff\xff" or checksum(data[2:-1]) != data[-1]:
                return len(data)
            sid, ln, ins = data[2], data[3], data[4]
            p = data[5:5 + ln - 2]
            if ins == INST_PING and sid in self.servos:
                self._answer(sid)
            elif ins == INST_READ and sid in self.servos:
                a, n = p[0], p[1]
                self._answer(sid, bytes(self.servos[sid].mem[a:a + n]))
            elif ins == INST_WRITE:
                targets = self.servos.values() if sid == BROADCAST_ID else (
                    [self.servos[sid]] if sid in self.servos else [])
                for s in targets:
                    s.mem[p[0]:p[0] + len(p) - 1] = p[1:]
                    if sid != BROADCAST_ID and s.get("Response_Status_Level"):
                        self._answer(s.sid)
            elif ins == INST_SYNC_WRITE and sid == BROADCAST_ID:
                a, n = p[0], p[1]
                for k in range(2, len(p), n + 1):
                    s = self.servos.get(p[k])
                    if s is not None:
                        s.mem[a:a + n] = p[k + 1:k + 1 + n]
            elif ins == INST_SYNC_READ and sid == BROADCAST_ID:
                a, n = p[0], p[1]
                for i in p[2:]:
                    if i in self.servos:
                        self._answer(i, bytes(self.servos[i].mem[a:a + n]))
            return len(data)


def mock_so101(raw=None, **kw) -> MockSerial:
    """Six mock servos (ids 1..6) at the given raw positions (default: all 2048)."""
    raw = raw or {}
    return MockSerial([MockServo(i, actual=raw.get(i, 2048)) for i in range(1, 7)], **kw)


# --------------------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description="Feetech STS3215 bus tools (SO-101)")
    ap.add_argument("--port", required=True, help="/dev/ttyACM0 (Pi), /dev/tty.usbmodem* (Mac); 'mock' for a fake bus")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--ids", default="1,2,3,4,5,6")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("scan", help="ping ids (and try other baud rates if nothing answers)")
    sub.add_parser("health", help="position, speed, load, voltage, temperature, status flags per servo")
    w = sub.add_parser("watch", help="print positions / speeds at 5 Hz (move the arm by hand with torque off)")
    w.add_argument("--torque-off", action="store_true")
    sub.add_parser("eeprom", help="print the calibration-relevant EEPROM registers (homing offset, limits)")
    t = sub.add_parser("torque", help="torque on|off for all ids")
    t.add_argument("state", choices=["on", "off"])
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ids = [int(x) for x in a.ids.split(",")]
    bus = FeetechBus(a.port, a.baud, ser=mock_so101() if a.port == "mock" else None).open()
    try:
        if a.cmd == "scan":
            found = bus.scan(ids)
            print(f"{a.baud} baud: " + (", ".join(f"id {i} model {m}" + (" (STS3215)" if m == STS3215_MODEL else "")
                                                for i, m in found.items()) or "nothing answers"))
            if not found and a.port != "mock":
                for b in BAUDRATES:
                    if b == a.baud:
                        continue
                    bus.close()
                    bus = FeetechBus(a.port, b, timeout_s=0.02, retries=0).open()
                    f = bus.scan(ids)
                    print(f"{b} baud: {f or 'nothing'}")
                    if f:
                        print(f"-> the servos run at {b} baud (SO-101 default is 1000000): pass --baud {b}")
                        break
            missing = [i for i in ids if i not in found]
            if found and missing:
                print(f"MISSING ids {missing}: loose 3-pin cable in the daisy chain after the last id found, "
                      f"or a servo that still has the factory id 1 (two servos with the same id corrupt each other)")
        elif a.cmd == "health":
            for sid, h in bus.read_health(ids).items():
                print(f"id {sid}: pos {h['pos']:5d}  vel {h['vel']:5d}  load {h['load_pct']:5.1f}%  "
                      f"{h['volts']:4.1f} V  {h['temp_c']} C  current {h['current_ma']:6.1f} mA"
                      + (f"  FLAGS {h['flags']}" if h["flags"] else ""))
        elif a.cmd == "eeprom":
            for sid, c in bus.read_eeprom_calibration(ids).items():
                print(f"id {sid}: homing_offset {c['homing_offset']:6d}  range {c['range_min']}..{c['range_max']}")
        elif a.cmd == "torque":
            print(bus.torque(ids, a.state == "on"))
        elif a.cmd == "watch":
            if a.torque_off:
                bus.torque(ids, False)
            try:
                while True:
                    h = bus.read_health(ids)
                    print("  ".join(f"{i}:{h[i]['pos']:5d}/{h[i]['vel']:+5d}" if i in h else f"{i}:  ---"
                                    for i in ids), flush=True)
                    time.sleep(0.2)
            except KeyboardInterrupt:
                pass
    finally:
        bus.close()


if __name__ == "__main__":
    main()
