"""Stock Creality OS websocket backend (K1 / K1C / K1 SE / K1 Max / K2 family / Ender-3 V3 / V3 KE / V3 Plus / Hi).

Protocol, as implemented by the Home Assistant integration ha_creality_ws (ws_client.py / utils.py,
https://github.com/3dg1luk43/ha_creality_ws, commit a8e5491, 2026-10-08):
  - ws://HOST:9999 (path "/", optional subprotocol "wsslicer" = what the printer's web UI offers)
  - the printer pushes JSON objects with a SUBSET of keys each time; merge them into one state dict;
    numbers often arrive as strings ("25.140000")
  - heartbeat: printer sends {"ModeCode": "heart_beat", ...}; reply with the literal text "ok"
  - keep it talking: {"method": "get", "params": {"ReqPrinterPara": 1}} (every ~10 s when quiet)
  - objects of the running job: {"method": "get", "params": {"reqPrintObjects": 1}}
  - file listing (big): {"method": "get", "params": {"reqGcodeFile": 1}} -> "retGcodeFileInfo2": [...]
  - keys: state, printProgress (0-100, or dProgress), printFileName (full path), nozzleTemp,
    targetNozzleTemp, bedTemp0, targetBedTemp0, boxTemp, printJobTime (s), printLeftTime (s),
    layer, TotalLayer, err {errcode, key}, model, modelVersion, hostname
  - state table (derive_print_state): with a file name: printProgress >= 100 -> completed,
    state 5 -> paused, 4 -> stopped, 1 -> printing, 0/7 -> processing; no file name -> idle.
    Creality's own enum (CrealityPrint v5 RemotePrinter.h): 0 idle, 1 printing, 2 complete, 3 failed,
    4 stopped, 5 paused; a real K1C also shows 9 for ~0.5 s at job start and 7 for ~45 s while a cancel ends.
    The printer keeps the last job's file name and 100 % after it finishes (so 'complete' persists).
    A non-zero err.errcode can be stale (never cleared), so it is reported but does not fail the job.
G-code download: the stock websocket has no file transfer. Stock K1C (2025, fw 1.0.0.22) and K2 firmware
also run a Moonraker-compatible API on :7125 (BELIEVED, community reports) -> /server/files/gcodes/...
is tried there; otherwise copy the file by hand (--gcode).
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.parse
from pathlib import Path

from real.printer import _http
from real.printer.base import (DownloadUnsupported, MANUAL_GCODE_HELP, Printer, PrinterError, PrinterStatus,
                               fnum)

log = logging.getLogger("real.printer.creality")
SUBPROTOCOL = "wsslicer"
HEARTBEAT_S = 10.0


def coerce(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, str):
            try:
                out[k] = float(v) if any(c in v for c in ".eE") else int(v)
                continue
            except ValueError:
                pass
        out[k] = v
    return out


def derive_state(d: dict) -> str:
    filename = d.get("printFileName") or ""
    prog = fnum(d.get("printProgress") if d.get("printProgress") is not None else d.get("dProgress"))
    st = d.get("state")
    st = int(st) if isinstance(st, (int, float)) else None
    if not filename:
        return "idle"
    if (prog is not None and prog >= 100) or st == 2:
        return "complete"
    if st == 3:
        return "error"            # failed
    if st == 5:
        return "paused"
    if st == 4:
        return "cancelled"
    if st in (1, 0, 7, 9):
        return "printing"         # 0/7/9 with a file = processing (starting, homing, finishing a cancel)
    return "unknown"


class CrealityWSPrinter(Printer):
    name = "creality"

    def __init__(self, host: str, port: int = 9999, first_timeout: float = 8.0, stale_s: float = 35.0,
                 moonraker_ports=(7125, 4408, 4409)):
        self.host, self.port = host, port
        self.url = f"ws://{host}:{port}/"
        self.first_timeout, self.stale_s = first_timeout, stale_s
        self.moonraker_ports = moonraker_ports
        self.state: dict = {}
        self.last_rx = 0.0
        self.connected = False
        self.errors = 0
        self._ws = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._first = threading.Event()
        self._thread = threading.Thread(target=self._run, name="creality-ws", daemon=True)
        self._thread.start()

    # ---------------------------------------------------------------- background connection
    def _run(self):
        from websockets.sync.client import connect
        backoff = 1.0
        while not self._stop.is_set():
            try:
                with connect(self.url, subprotocols=[SUBPROTOCOL], open_timeout=5, close_timeout=1,
                             max_size=8 * 1024 * 1024) as ws:
                    self._ws, self.connected, backoff = ws, True, 1.0
                    self._send({"method": "get", "params": {"ReqPrinterPara": 1}})
                    last_probe = time.monotonic()
                    while not self._stop.is_set():
                        try:
                            raw = ws.recv(timeout=1.0)
                        except TimeoutError:
                            raw = None
                        now = time.monotonic()
                        if raw is not None:
                            self._handle(raw, ws)
                        if now - max(self.last_rx, last_probe) > HEARTBEAT_S:
                            self._send({"method": "get", "params": {"ReqPrinterPara": 1}})
                            last_probe = now
            except Exception as e:                       # noqa: BLE001 - reconnect on anything
                self.errors += 1
                log.debug("creality ws error: %s", e)
            self._ws, self.connected = None, False
            if self._stop.wait(backoff):
                break
            backoff = min(backoff * 2, 15.0)

    def _handle(self, raw, ws):
        text = raw.decode("utf-8", "ignore") if isinstance(raw, (bytes, bytearray)) else raw
        if text == "ok":
            return
        try:
            d = json.loads(text)
        except json.JSONDecodeError:
            return
        if not isinstance(d, dict):
            return
        self.last_rx = time.monotonic()
        if d.get("ModeCode") == "heart_beat":
            try:
                ws.send("ok")
            except Exception:                            # noqa: BLE001
                pass
            self._first.set()
            return
        with self._lock:
            self.state.update(coerce(d))
        self._first.set()

    def _send(self, obj):
        ws = self._ws
        if ws is not None:
            try:
                ws.send(json.dumps(obj))
            except Exception:                            # noqa: BLE001
                pass

    # ---------------------------------------------------------------- Printer API
    def raw(self) -> dict:
        if not self._first.wait(self.first_timeout):
            raise PrinterError(f"no data from {self.url} within {self.first_timeout:.0f} s "
                               f"(wrong IP/port, printer asleep, or not Creality OS)")
        if time.monotonic() - self.last_rx > self.stale_s:
            raise PrinterError(f"{self.url}: no data for {time.monotonic() - self.last_rx:.0f} s (disconnected)")
        with self._lock:
            return dict(self.state)

    def status(self) -> PrinterStatus:
        d = self.raw()
        state = derive_state(d)
        prog = fnum(d.get("printProgress") if d.get("printProgress") is not None else d.get("dProgress"))
        err = d.get("err")
        code = err.get("errcode") if isinstance(err, dict) else err
        msg = f"errcode={code}" if fnum(code) not in (None, 0.0) else ""
        pos = None
        if isinstance(d.get("curPosition"), str):          # "X:110.00 Y:110.00 Z:25.00"
            m = dict(re.findall(r"([XYZ]):\s*([-\d.]+)", d["curPosition"]))
            pos = [float(m[k]) for k in "XYZ"] if len(m) == 3 else None
        return PrinterStatus(state=state, progress=None if prog is None else prog / 100.0,
                             filename=d.get("printFileName") or None,
                             bed_c=fnum(d.get("bedTemp0")), bed_target_c=fnum(d.get("targetBedTemp0")),
                             nozzle_c=fnum(d.get("nozzleTemp")), nozzle_target_c=fnum(d.get("targetNozzleTemp")),
                             time_left_s=fnum(d.get("printLeftTime")), elapsed_s=fnum(d.get("printJobTime")),
                             message=msg, position=pos, backend=self.name, raw=d)

    def info(self) -> dict:
        d = self.raw()
        return {k: d.get(k) for k in ("model", "modelVersion", "hostname", "maxBedTemp", "maxNozzleTemp")}

    def objects(self, timeout: float = 3.0) -> list | None:
        """Objects of the running job as the firmware reports them (format varies; raw list)."""
        self.raw()                                       # connected and talking
        self._send({"method": "get", "params": {"reqPrintObjects": 1}})
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            with self._lock:
                objs = self.state.get("objects_list") or self.state.get("objectsList") or self.state.get("objects")
            if objs is not None:
                if isinstance(objs, str):
                    try:
                        objs = json.loads(objs)
                    except json.JSONDecodeError:
                        return None
                if isinstance(objs, dict):
                    objs = objs.get("list")
                return objs if isinstance(objs, list) else None
            time.sleep(0.1)
        return None

    def download_gcode(self, dest, filename=None) -> Path:
        fname = filename or self.status().filename
        if not fname:
            raise PrinterError("no current/last print file name reported by the printer")
        rel = fname.split("/gcodes/", 1)[-1].lstrip("/")
        dest = Path(dest)
        if dest.is_dir() or str(dest).endswith("/"):
            dest = dest / Path(rel).name
        tried = []
        for port in self.moonraker_ports:
            url = f"http://{self.host}:{port}/server/files/gcodes/{urllib.parse.quote(rel)}"
            try:
                return _http.download(url, dest, timeout=30)
            except PrinterError as e:
                tried.append(f"{url}: {e}")
        raise DownloadUnsupported(f"printer reports file '{fname}' but no Moonraker-compatible file API "
                                  f"answered:\n  " + "\n  ".join(tried) + "\n" + MANUAL_GCODE_HELP)

    def close(self):
        self._stop.set()
        ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception:                            # noqa: BLE001
                pass
        self._thread.join(timeout=3)
