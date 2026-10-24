"""Moonraker (Klipper) backend: rooted Creality OS (K1 / K1 Max / Ender-3 V3 KE), K2 with Moonraker,
Creality Sonic Pad, any Klipper + Fluidd/Mainsail machine.

Port 7125 is Moonraker itself; Fluidd (4408) / Mainsail (4409) / port 80 nginx proxy the same API.
API: https://moonraker.readthedocs.io/en/latest/external_api/printer/ and .../external_api/file_manager/
"""
from __future__ import annotations

import urllib.parse
from pathlib import Path

from real.printer import _http
from real.printer.base import Printer, PrinterError, PrinterStatus, fnum

STATE_MAP = {"standby": "idle", "printing": "printing", "paused": "paused", "complete": "complete",
             "cancelled": "cancelled", "error": "error"}
QUERY = "print_stats&heater_bed&extruder&virtual_sdcard&display_status&toolhead"


class MoonrakerPrinter(Printer):
    name = "moonraker"

    def __init__(self, host: str, port: int = 7125, api_key: str | None = None, timeout: float = 5.0):
        self.base = f"http://{host}:{port}"
        self.headers = {"X-Api-Key": api_key} if api_key else {}
        self.timeout = timeout

    def _get(self, path: str):
        return _http.get_json(self.base + path, self.headers, self.timeout)

    def info(self) -> dict:
        return self._get("/server/info").get("result", {})

    def query(self, objects: str = QUERY) -> dict:
        return self._get(f"/printer/objects/query?{objects}")["result"]["status"]

    def status(self) -> PrinterStatus:
        s = self.query()
        ps, vs, ds = s.get("print_stats", {}), s.get("virtual_sdcard", {}), s.get("display_status", {})
        bed, ext = s.get("heater_bed", {}), s.get("extruder", {})
        pos = (s.get("toolhead") or {}).get("position")
        state = STATE_MAP.get(ps.get("state"), "unknown")
        prog = fnum(vs.get("progress"))
        if prog is None:
            prog = fnum(ds.get("progress"))
        dur = fnum(ps.get("print_duration"))
        left = None
        if state in ("printing", "paused") and prog and prog > 0.01 and dur:
            left = dur / prog - dur
        elif state == "complete":
            left, prog = 0.0, 1.0 if prog is None or prog > 0.99 else prog
        return PrinterStatus(state=state, progress=prog, filename=ps.get("filename") or None,
                             bed_c=fnum(bed.get("temperature")), bed_target_c=fnum(bed.get("target")),
                             nozzle_c=fnum(ext.get("temperature")), nozzle_target_c=fnum(ext.get("target")),
                             time_left_s=left, elapsed_s=dur, message=ps.get("message") or ds.get("message") or "",
                             position=[float(v) for v in pos[:3]] if pos else None, backend=self.name, raw=s)

    def objects(self) -> list[dict] | None:
        try:
            eo = self.query("exclude_object").get("exclude_object", {})
        except PrinterError:
            return None
        return eo.get("objects") or None

    def list_files(self) -> list[dict]:
        return self._get("/server/files/list?root=gcodes").get("result", [])

    def metadata(self, filename: str) -> dict:
        return self._get("/server/files/metadata?filename=" + urllib.parse.quote(filename)).get("result", {})

    def download_gcode(self, dest, filename=None) -> Path:
        filename = filename or self.status().filename
        if not filename:
            files = sorted(self.list_files(), key=lambda f: f.get("modified", 0))
            if not files:
                raise PrinterError("no current print and no files on the printer")
            filename = files[-1]["path"]
        dest = Path(dest)
        if dest.is_dir() or str(dest).endswith("/"):
            dest = dest / Path(filename).name
        url = f"{self.base}/server/files/gcodes/{urllib.parse.quote(filename)}"
        return _http.download(url, dest, self.headers)
