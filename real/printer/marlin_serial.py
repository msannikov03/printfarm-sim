"""Marlin over USB serial (Ender-3 / Ender-3 V2 / V3 SE / S1 and other non-network Creality printers).

Only useful when the printer prints FROM ITS SD CARD (M27 reports SD progress). If a PC/OctoPrint is
streaming the print over the same USB port, use that host instead: one serial port, one owner.

WARNING: on old 8-bit boards (Melzi / Creality 1.1.x) opening the port can pulse DTR and RESET the
board, killing a running print. We open with DTR/RTS low, but test on a non-critical print first.
G-code reference: https://marlinfw.org/docs/gcode/M027.html, M105, M115, M31.
"""
from __future__ import annotations

import re
import time

from real.printer.base import DownloadUnsupported, MANUAL_GCODE_HELP, Printer, PrinterError, PrinterStatus

TEMP_RE = re.compile(r"\b([TB])(\d?):\s*([-\d.]+)\s*/\s*([-\d.]+)")
SD_RE = re.compile(r"SD printing byte (\d+)\s*/\s*(\d+)", re.I)
FILE_RE = re.compile(r"Current file:\s*(.+)", re.I)
PRINT_TIME_RE = re.compile(r"Print time:\s*(?:(\d+)d\s*)?(?:(\d+)h\s*)?(?:(\d+)m\s*)?(?:(\d+)s)?", re.I)


def open_serial(port: str, baud: int = 115200, timeout: float = 0.5):
    import serial          # pyserial
    s = serial.Serial()
    s.port, s.baudrate, s.timeout = port, baud, timeout
    s.dtr = False          # do not auto-reset Arduino-style boards
    s.rts = False
    s.open()
    return s


class MarlinSerialPrinter(Printer):
    name = "marlin"

    def __init__(self, port: str | None = None, baud: int = 115200, ser=None, boot_wait_s: float = 2.0):
        self.ser = ser if ser is not None else open_serial(port, baud)
        self.done_seen = False
        self.paused_seen = False
        self.firmware = ""
        if ser is None:
            time.sleep(boot_wait_s)
        self._drain()
        try:
            self.firmware = " ".join(self.cmd("M115", timeout=3.0))
        except PrinterError:
            pass

    def _readline(self) -> str:
        b = self.ser.readline()
        line = b.decode("ascii", "replace").strip() if isinstance(b, (bytes, bytearray)) else str(b).strip()
        if line:
            self._watch(line)
        return line

    def _watch(self, line: str):
        low = line.lower()
        if "done printing file" in low or "//action:print_end" in low or "print finished" in low:
            self.done_seen = True
        if "//action:paused" in low or "//action:pause" in low:
            self.paused_seen = True
        if "//action:resumed" in low or "//action:resume" in low:
            self.paused_seen = False
        if SD_RE.search(line):
            self.done_seen = False

    def _drain(self):
        for _ in range(200):
            if not self._readline():
                break

    def cmd(self, gcode: str, timeout: float = 2.0) -> list[str]:
        """Send one command, return the lines before 'ok' (unsolicited lines are watched too)."""
        self.ser.write((gcode.strip() + "\n").encode())
        out, t0 = [], time.monotonic()
        while time.monotonic() - t0 < timeout:
            line = self._readline()
            if not line:
                continue
            if line.startswith("ok"):
                rest = line[2:].strip()
                if rest:
                    out.append(rest)
                return out
            if line.lower().startswith("echo:busy"):
                t0 = time.monotonic()
                continue
            out.append(line)
        raise PrinterError(f"no 'ok' for {gcode} within {timeout} s (got {out[-3:]})")

    def status(self) -> PrinterStatus:
        temps = " ".join(self.cmd("M105"))
        sd = " ".join(self.cmd("M27"))
        fname = None
        try:
            m = FILE_RE.search(" ".join(self.cmd("M27 C")))
            if m and "(no file)" not in m.group(1):
                parts = m.group(1).split()
                fname = parts[-1] if len(parts) > 1 else parts[0]   # "SHORT.GCO Long Name.gcode" -> long name
        except PrinterError:
            pass
        elapsed = None
        try:
            m = PRINT_TIME_RE.search(" ".join(self.cmd("M31")))
            if m and any(m.groups()):
                d, h, mi, s = (int(v or 0) for v in m.groups())
                elapsed = float(((d * 24 + h) * 60 + mi) * 60 + s)
        except PrinterError:
            pass
        t = {k + n: (float(a), float(b)) for k, n, a, b in TEMP_RE.findall(temps)}
        noz = t.get("T") or t.get("T0")
        bed = t.get("B")
        m = SD_RE.search(sd)
        prog = None
        if m:
            done, total = int(m.group(1)), int(m.group(2))
            prog = done / total if total else None
            state = "paused" if self.paused_seen else "printing"
            if total and done >= total:
                state = "complete"
        elif self.done_seen:
            state, prog = "complete", 1.0
        elif self.paused_seen:
            state = "paused"
        else:
            state = "idle"
        left = elapsed / prog - elapsed if (elapsed and prog and 0.01 < prog < 1) else None
        return PrinterStatus(state=state, progress=prog, filename=fname,
                             bed_c=bed[0] if bed else None, bed_target_c=bed[1] if bed else None,
                             nozzle_c=noz[0] if noz else None, nozzle_target_c=noz[1] if noz else None,
                             time_left_s=left, elapsed_s=elapsed, backend=self.name,
                             raw={"M105": temps, "M27": sd, "firmware": self.firmware})

    def download_gcode(self, dest, filename=None):
        raise DownloadUnsupported("Marlin cannot send SD-card files back over USB.\n" + MANUAL_GCODE_HELP)

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass
