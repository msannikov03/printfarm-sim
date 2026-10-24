import json

import pytest

from real.mock_printer import GCODE_NAME, MockPrinter, PrintSim, creality_frame
from real.printer import DownloadUnsupported, PrintFailed, PrinterError, PrinterStatus, make_printer
from real.printer.base import Printer, wait_until_done_and_cool
from real.printer.creality_ws import coerce, derive_state
from real.printer.marlin_serial import MarlinSerialPrinter
from real.printer.octoprint import map_state

FAST = dict(print_s=0.6, tau_s=0.3)
FAST_WS = dict(print_s=0.6, tau_s=1.0, push_s=0.1)


@pytest.mark.parametrize("kind", ["moonraker", "creality", "octoprint"])
def test_backend_against_mock_full_cycle(kind, tmp_path):
    with MockPrinter(kind, **(FAST_WS if kind == "creality" else FAST)) as m:
        p = make_printer(m.url)
        try:
            st = p.status()
            assert st.state == "printing" and 0 <= st.progress < 1 and st.bed_c == pytest.approx(60, abs=0.5)
            assert st.filename.endswith(GCODE_NAME) and st.backend == kind
            phases = []
            st = p.wait_until_done_and_cool(release_c=35, poll_s=0.05, timeout_s=15,
                                            on_update=lambda ph, s: phases.append(ph))
            assert st.state == "complete" and st.bed_c <= 35 and st.progress == pytest.approx(1.0)
            assert phases[0] == "waiting_for_print" and "cooling" in phases and phases[-1] == "ready"
            f = p.download_gcode(tmp_path)
            assert f.read_text() == m.sim.gcode
        finally:
            p.close()


@pytest.mark.parametrize("kind", ["moonraker", "creality"])
def test_firmware_objects(kind):
    with MockPrinter(kind, **FAST) as m:
        p = make_printer(m.url)
        try:
            objs = p.objects()
            assert [o["name"] for o in objs] == ["bracket", "knob", "cube"]
            assert len(objs[0]["polygon"]) == 4 and objs[0]["center"] == [70.0, 60.0]
        finally:
            p.close()


@pytest.mark.parametrize("kind,fail", [("moonraker", "cancelled"), ("moonraker", "error"),
                                       ("creality", "cancelled"), ("octoprint", "error")])
def test_failed_print_raises(kind, fail):
    with MockPrinter(kind, fail=fail, **FAST) as m:
        p = make_printer(m.url)
        try:
            with pytest.raises(PrintFailed):
                p.wait_until_done_and_cool(poll_s=0.05, timeout_s=10)
        finally:
            p.close()


def test_creality_unreachable_raises_quickly():
    from real.printer.creality_ws import CrealityWSPrinter
    p = CrealityWSPrinter("127.0.0.1", 9, first_timeout=0.5)
    try:
        with pytest.raises(PrinterError):
            p.status()
    finally:
        p.close()


def test_creality_download_without_file_api_gives_instructions(tmp_path):
    with MockPrinter("creality", files_port=None, **FAST) as m:
        from real.printer.creality_ws import CrealityWSPrinter
        host, port = m.url.split("://")[1].split("?")[0].split(":")
        p = CrealityWSPrinter(host, int(port), moonraker_ports=(9,))
        try:
            with pytest.raises(DownloadUnsupported, match="--gcode"):
                p.download_gcode(tmp_path)
        finally:
            p.close()


def test_creality_state_table_and_coercion():
    f = "/usr/data/printer_data/gcodes/a.gcode"
    assert derive_state({"printFileName": f, "printProgress": 100, "state": 0}) == "complete"
    assert derive_state({"printFileName": f, "printProgress": 40, "state": 1}) == "printing"
    assert derive_state({"printFileName": f, "printProgress": 40, "state": 5}) == "paused"
    assert derive_state({"printFileName": f, "printProgress": 40, "state": 4}) == "cancelled"
    assert derive_state({"printFileName": f, "dProgress": 100}) == "complete"
    assert derive_state({"printFileName": "", "state": 0}) == "idle"
    assert derive_state({"printFileName": f, "printProgress": 99, "state": 2}) == "complete"
    assert derive_state({"printFileName": f, "printProgress": 40, "state": 3}) == "error"
    assert derive_state({"printFileName": f, "printProgress": 0, "state": 9}) == "printing"
    c = coerce({"nozzleTemp": "25.140000", "state": "1", "printFileName": f, "hostname": "K1C-AB12"})
    assert c["nozzleTemp"] == pytest.approx(25.14) and c["state"] == 1 and c["printFileName"] == f
    assert c["hostname"] == "K1C-AB12"
    fr = creality_frame(PrintSim(print_s=100), full=True)
    assert isinstance(fr["bedTemp0"], str) and fr["model"] == "K1C"     # mock mimics string temps


def test_octoprint_state_map():
    assert map_state("Printing from SD", 50) == "printing"
    assert map_state("Operational", 100.0) == "complete"
    assert map_state("Operational", 37.0) == "idle"
    assert map_state("Paused", 37.0) == "paused"
    assert map_state("Offline after error", None) == "error"


# ----------------------------------------------------------------------------- state machine

class Scripted(Printer):
    name = "scripted"

    def __init__(self, seq):
        self.seq = list(seq)

    def status(self):
        item = self.seq.pop(0) if len(self.seq) > 1 else self.seq[0]
        if isinstance(item, Exception):
            raise item
        state, bed = item
        return PrinterStatus(state=state, bed_c=bed, backend=self.name)


def _run(seq, **kw):
    phases = []
    st = wait_until_done_and_cool(Scripted(seq), poll_s=0, sleep=lambda s: None,
                                  on_update=lambda ph, s: phases.append(ph), **kw)
    return st, phases


def test_state_machine_tolerates_disconnects_and_cools():
    seq = [("printing", 60), OSError("wifi"), PrinterError("timeout"), ("printing", 60), ("complete", 58),
           ConnectionError("reset"), ("complete", 40), ("complete", 34.9)]
    st, ph = _run(seq)
    assert st.bed_c == 34.9
    assert ph == ["waiting_for_print", "offline", "offline", "waiting_for_print", "cooling", "offline", "cooling",
                  "ready"]


def test_state_machine_idle_after_printing_counts_as_done():
    st, ph = _run([("printing", 60), ("idle", 50), ("idle", 30)])
    assert ph[-1] == "ready"


def test_state_machine_idle_without_print_needs_accept_idle():
    with pytest.raises(TimeoutError):
        clock = iter(range(1000))
        wait_until_done_and_cool(Scripted([("idle", 25)]), poll_s=0, sleep=lambda s: None, accept_idle=False,
                                 timeout_s=50, clock=lambda: next(clock))
    st, ph = _run([("idle", 25)])
    assert ph == ["ready"]


def test_state_machine_unknown_bed_never_ready_and_new_print_resets():
    st, ph = _run([("complete", None), ("complete", None), ("printing", 60), ("complete", 50), ("complete", 30)])
    assert ph == ["cooling", "cooling", "waiting_for_print", "cooling", "ready"]


def test_state_machine_offline_too_long_raises():
    clock = iter(range(0, 10000, 100))
    with pytest.raises(OSError):
        wait_until_done_and_cool(Scripted([OSError("gone")]), poll_s=0, sleep=lambda s: None, max_offline_s=250,
                                 clock=lambda: next(clock))


def test_state_machine_error_raises():
    with pytest.raises(PrintFailed):
        _run([("printing", 60), ("error", 60)])


# ----------------------------------------------------------------------------- Marlin serial (fake port)

class FakeMarlin:
    """Answers like Marlin 2.x on a Creality 4.2.x board, printing from SD."""

    def __init__(self, total=1000, step=400):
        self.pos, self.total, self.step = 0, total, step
        self.out = [b"start\n", b"echo:Marlin 2.1.2\n"]
        self.closed = False

    def write(self, b):
        cmd = b.decode().strip()
        if cmd == "M115":
            self.out += [b"FIRMWARE_NAME:Marlin 2.1.2 (Creality) SOURCE_CODE_URL:github.com/MarlinFirmware/Marlin "
                         b"PROTOCOL_VERSION:1.0 MACHINE_TYPE:Ender-3 V2 EXTRUDER_COUNT:1\n", b"ok\n"]
        elif cmd == "M105":
            done = self.pos >= self.total
            bed = 60.0 if not done else 30.0
            self.out.append(f"ok T:{215.2 if not done else 40.1} /{220 if not done else 0}.00 B:{bed} /"
                            f"{60 if not done else 0}.00 @:0 B@:0\n".encode())
        elif cmd == "M27":
            if self.pos < self.total:
                self.pos = min(self.total, self.pos + self.step)
                if self.pos >= self.total:
                    self.out += [b"Done printing file\n", b"Not SD printing\n", b"ok\n"]
                else:
                    self.out += [f"SD printing byte {self.pos}/{self.total}\n".encode(), b"ok\n"]
            else:
                self.out += [b"Not SD printing\n", b"ok\n"]
        elif cmd == "M27 C":
            self.out += [b"Current file: PLATE~1.GCO plate_k1.gcode\n", b"ok\n"]
        elif cmd == "M31":
            self.out += [b"echo:Print time: 1h 2m 3s\n", b"ok\n"]
        else:
            self.out.append(b"ok\n")

    def readline(self):
        return self.out.pop(0) if self.out else b""

    def close(self):
        self.closed = True


def test_marlin_serial_status_cycle(tmp_path):
    fake = FakeMarlin()
    p = MarlinSerialPrinter(ser=fake)
    assert "Ender-3 V2" in p.firmware
    st = p.status()
    assert st.state == "printing" and st.progress == pytest.approx(0.4) and st.bed_c == 60.0
    assert st.nozzle_c == pytest.approx(215.2) and st.filename == "plate_k1.gcode" and st.elapsed_s == 3723
    st = p.wait_until_done_and_cool(poll_s=0, sleep=lambda s: None)
    assert st.state == "complete" and st.bed_c == 30.0
    with pytest.raises(DownloadUnsupported):
        p.download_gcode(tmp_path)
    p.close()
    assert fake.closed


def test_make_printer_urls():
    with pytest.raises(ValueError):
        make_printer("ftp://x")
    from real.printer.moonraker import MoonrakerPrinter
    p = make_printer("192.168.1.5:4408")
    assert isinstance(p, MoonrakerPrinter) and p.base == "http://192.168.1.5:4408"
    assert json.dumps(PrinterStatus(state="idle").to_dict())
