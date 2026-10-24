import json
import math
import sys
import types

import numpy as np
import pytest

from real.arm import Arm, JointMap, Safety, SafetyError, _import_lerobot
from real.so101 import GRIPPER_CLOSED_RAD, GRIPPER_OPEN_RAD, JOINTS, SO101


@pytest.fixture(scope="module")
def kin():
    return SO101()


def test_joint_map_roundtrip_and_units():
    jm = JointMap()
    q = np.array([0.3, -1.0, 1.2, 0.5, -2.0, 0.4])
    v = jm.model_to_lerobot(q)
    assert list(v) == list(JOINTS)                              # LeRobot motor names == MJCF joint names
    assert v["shoulder_pan"] == pytest.approx(math.degrees(0.3))  # degrees, sign +1, offset 0 by default
    assert np.allclose(jm.lerobot_to_model(v), q)
    assert jm.model_to_lerobot([0, 0, 0, 0, 0, GRIPPER_CLOSED_RAD])["gripper"] == pytest.approx(0)
    assert jm.model_to_lerobot([0, 0, 0, 0, 0, GRIPPER_OPEN_RAD])["gripper"] == pytest.approx(100)
    jm2 = JointMap()
    jm2.sign["elbow_flex"], jm2.offset_deg["wrist_flex"] = -1.0, 7.5
    assert jm2.model_to_lerobot(q)["elbow_flex"] == pytest.approx(-math.degrees(1.2))
    assert np.allclose(jm2.lerobot_to_model(jm2.model_to_lerobot(q)), q)


def test_joint_map_json(tmp_path):
    jm = JointMap()
    jm.sign["shoulder_lift"] = -1.0
    jm.save(tmp_path / "jm.json")
    assert JointMap.load(tmp_path / "jm.json").sign["shoulder_lift"] == -1.0


def test_dry_run_motion_is_interpolated_and_checked(kin):
    arm = Arm(None, dry_run=True, kin=kin, safety=Safety(max_speed_deg_s=20, rate_hz=25))
    q1 = np.array([0.2, -0.6, 0.8, 1.2, 0.3, 0.0])
    path = arm.move_to(q1, "t")
    steps = np.degrees(np.abs(np.diff(np.array([arm.q * 0 + p for p in path]), axis=0)))
    assert np.allclose(path[-1], q1) and steps.max() <= 0.8 + 1e-6      # 20 deg/s at 25 Hz
    with pytest.raises(SafetyError, match="joint limits"):
        arm.plan(np.array([3.0, 0, 0, 0, 0, 0]))
    arm.safety.box_max = (0.1, 0.3, 0.4)                                 # stretched arm tip x ~0.39 m
    with pytest.raises(SafetyError, match="workspace box"):
        arm.plan(np.zeros(6))


class _FakeBus:
    def __init__(self):
        self.torque = True
        self.log = []

    def enable_torque(self):
        self.torque = True
        self.log.append("on")

    def disable_torque(self):
        self.torque = False
        self.log.append("off")


class _FakeRobot:
    instances = []

    def __init__(self, cfg):
        self.cfg, self.bus, self.actions = cfg, _FakeBus(), []
        self.pos = {j: 0.0 for j in JOINTS}
        self.pos.update(shoulder_lift=-60.0, elbow_flex=60.0, wrist_flex=60.0)
        self.is_calibrated = True
        _FakeRobot.instances.append(self)

    def connect(self, calibrate=True):
        assert calibrate is False

    def get_observation(self):
        return {f"{j}.pos": v for j, v in self.pos.items()}

    def send_action(self, a):
        self.actions.append(dict(a))
        self.pos.update({k.removesuffix(".pos"): v for k, v in a.items()})
        return a

    def disconnect(self):
        pass


class _FakeCfg:
    def __init__(self, port, id, use_degrees=False, max_relative_target=None, disable_torque_on_disconnect=True,
                 calibration_dir=None):
        self.__dict__.update(locals())


@pytest.fixture
def fake_lerobot(monkeypatch):
    mod = types.ModuleType("lerobot.robots.so_follower")
    mod.SO101Follower, mod.SO101FollowerConfig = _FakeRobot, _FakeCfg
    for name in ("lerobot", "lerobot.robots"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "lerobot.robots.so_follower", mod)
    _FakeRobot.instances.clear()
    return _FakeRobot


def test_live_adapter_against_fake_lerobot(kin, fake_lerobot):
    answers = iter(["n", "y"])
    arm = Arm("/dev/tty.fake", robot_id="f1", dry_run=False, kin=kin, input_fn=lambda m: next(answers),
              sleep=lambda s: None)
    arm.connect()
    robot = fake_lerobot.instances[0]
    assert robot.cfg.use_degrees is True and robot.cfg.max_relative_target == 5.0
    assert robot.cfg.disable_torque_on_disconnect is False and robot.cfg.id == "f1"
    assert np.allclose(np.degrees(arm.q[1:4]), [-60, 60, 60])
    target = np.array([0.1, -0.8, 0.9, 1.1, 0.2, 0.3])
    with pytest.raises(SafetyError, match="declined"):
        arm.move_to(target, "t")
    assert robot.actions == []
    arm.move_to(target, "t")
    last = robot.actions[-1]
    assert set(last) == {f"{j}.pos" for j in JOINTS}
    assert last["elbow_flex.pos"] == pytest.approx(math.degrees(0.9))
    steps = np.array([[a[f"{j}.pos"] for j in JOINTS[:5]] for a in robot.actions])
    assert np.abs(np.diff(steps, axis=0)).max() <= 0.8 + 1e-6
    n = len(robot.actions)
    arm.gripper(60, confirm=False)                    # gripper moves in <= 2-unit steps, not one jump
    g = [x["gripper.pos"] for x in robot.actions[n:]]
    assert g[-1] == pytest.approx(60) and np.abs(np.diff(g)).max() <= 2.0 + 1e-6
    arm.torque(False)
    robot.pos["wrist_roll"] = 33.0          # hand moved it
    arm.torque(True)
    assert robot.actions[-1]["wrist_roll.pos"] == 33.0 and robot.bus.log == ["off", "on"]   # goal := present
    arm.disconnect()


def test_import_lerobot_message_when_missing():
    import importlib.util
    if importlib.util.find_spec("lerobot") is not None:
        pytest.skip("lerobot is installed")
    with pytest.raises(ImportError, match="lerobot\\[feetech\\]"):
        _import_lerobot()


# ----------------------------------------------------------------------------- discover

def test_discover_fingerprints_mocks():
    from real.discover import fingerprint
    from real.mock_printer import MockPrinter
    for kind, want in (("moonraker", "moonraker://"), ("creality", "creality://"), ("octoprint", "octoprint://")):
        with MockPrinter(kind, files_port=None, print_s=30) as m:
            port = int(m.url.split("://")[1].split("?")[0].split(":")[1])
            r = fingerprint("127.0.0.1", ports=(port,))
            assert r["open"] == [port] and any(s.startswith(want) for s in r["suggest"]), r


def test_discover_usb_hints(monkeypatch):
    from serial.tools import list_ports

    from real.discover import serial_ports

    class P:
        def __init__(self, dev, vid, pid):
            self.device, self.vid, self.pid = dev, vid, pid
            self.description, self.manufacturer, self.serial_number = "USB Serial", "QinHeng", "x"
    monkeypatch.setattr(list_ports, "comports", lambda: [P("/dev/tty.usbserial-1", 0x1A86, 0x7523),
                                                          P("/dev/tty.usbmodem5A7A", 0x1A86, 0x55D3),
                                                          P("/dev/tty.Bluetooth", None, None)])
    out = serial_ports()
    assert len(out) == 2 and "Creality" in out[0]["hint"] and "SO-101" in out[1]["hint"]


# ----------------------------------------------------------------------------- end to end

@pytest.mark.parametrize("kind,style", [("creality", "orca"), ("moonraker", "none")])
def test_run_demo_mock_dry_run(tmp_path, kind, style):
    from real.run_demo import main
    rep = main(["--mock", "--mock-kind", kind, "--mock-style", style, "--mock-print-s", "0.5",
                "--mock-tau-s", "0.3", "--dry-run", "--runs-dir", str(tmp_path), "--calib", str(tmp_path / "none.json")])
    assert rep["printer_status"]["state"] == "complete" and rep["printer_status"]["bed_c"] <= 35
    assert [r["status"] for r in rep["results"]] == ["dry-run ok"] * 3
    for r in rep["results"]:
        # the arm's hover point lands 40 mm above the part top, over its centre (bed frame)
        assert np.allclose(r["reached_bed_mm"][:2], r["center_mm"], atol=1.5)
        assert r["reached_bed_mm"][2] == pytest.approx(r["height_mm"] + 40, abs=1.5)
    log = json.loads((tmp_path / "real_demo_log.json").read_text())
    assert log["plan"]["pick_order"] == [r["name"] for r in log["results"]]


def test_run_demo_preview_renders(tmp_path):
    from real.run_demo import main
    from real.synth_gcode import SynthOpts, make_gcode
    g = tmp_path / "p.gcode"
    g.write_text(make_gcode(opts=SynthOpts(style="cura")))
    main(["--gcode", str(g), "--dry-run", "--sim-preview", "--runs-dir", str(tmp_path),
          "--calib", str(tmp_path / "none.json")])
    pngs = sorted(p.name for p in tmp_path.glob("real_preview_*.png"))
    assert pngs[0] == "real_preview_00_overview.png" and len(pngs) == 4


def test_live_refuses_nominal_calibration(tmp_path):
    from real.run_demo import main
    from real.synth_gcode import make_gcode
    g = tmp_path / "p.gcode"
    g.write_text(make_gcode())
    with pytest.raises(SystemExit, match="nominal"):
        main(["--gcode", str(g), "--live", "--port", "/dev/null", "--runs-dir", str(tmp_path),
              "--calib", str(tmp_path / "none.json")])


def test_run_demo_falls_back_to_firmware_objects(tmp_path, monkeypatch):
    """Stock Creality without a file API: parts come from the websocket's object list."""
    import real.mock_printer as mp
    from real.run_demo import main
    real_init = mp.MockPrinter.__init__

    def no_files(self, kind, *a, **k):
        k["files_port"] = None
        real_init(self, kind, *a, **k)
    monkeypatch.setattr(mp.MockPrinter, "__init__", no_files)
    monkeypatch.setattr("real.printer.creality_ws.CrealityWSPrinter.download_gcode",
                        lambda self, dest, filename=None: (_ for _ in ()).throw(
                            __import__("real.printer", fromlist=["DownloadUnsupported"]).DownloadUnsupported("no api")))
    rep = main(["--mock", "--mock-print-s", "0.3", "--mock-tau-s", "0.3", "--dry-run", "--runs-dir", str(tmp_path),
                "--calib", str(tmp_path / "none.json"), "--part-height", "20"])
    assert rep["plan"]["method"] == "firmware_objects"
    assert sorted(o["name"] for o in rep["plan"]["objects"]) == ["bracket", "cube", "knob"]
    assert all(r["status"] == "dry-run ok" for r in rep["results"])


def test_discover_creality_http_info():
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from real.discover import fingerprint

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = b'{"model": "K1C", "mac": "00:11:22:33:44:55"}' if self.path == "/info" else b"nope"
            self.send_response(200 if self.path == "/info" else 404)
            self.end_headers()
            self.wfile.write(body)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        r = fingerprint("127.0.0.1", ports=(srv.server_address[1],))
        assert r["suggest"] == ["creality://127.0.0.1:9999"] and "K1C" in r["found"][0]
    finally:
        srv.shutdown()
