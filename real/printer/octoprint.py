"""OctoPrint backend (Marlin printer behind an OctoPrint Pi, e.g. an older Ender).

API: https://docs.octoprint.org/en/master/api/job.html, .../api/printer.html, .../api/files.html
Needs an API key (OctoPrint > Settings > Application Keys), sent as X-Api-Key.
"""
from __future__ import annotations

import urllib.parse
from pathlib import Path

from real.printer import _http
from real.printer.base import Printer, PrinterError, PrinterStatus, fnum


def map_state(text: str, completion: float | None) -> str:
    t = (text or "").lower()
    if t.startswith("printing") or t in ("starting", "starting print", "finishing", "resuming"):
        return "printing"
    if t.startswith("paus"):
        return "paused"
    if t.startswith("cancel"):
        return "cancelled"
    if "error" in t or t.startswith("offline") or "closed" in t:
        return "error"
    if t.startswith("operational") or t == "ready":
        # OctoPrint keeps the last job's completion after it ends: 100 -> finished
        return "complete" if completion is not None and completion >= 99.9 else "idle"
    return "unknown"


class OctoPrintPrinter(Printer):
    name = "octoprint"

    def __init__(self, host: str, port: int = 80, api_key: str | None = None, timeout: float = 5.0):
        self.base = f"http://{host}:{port}"
        self.headers = {"X-Api-Key": api_key} if api_key else {}
        self.timeout = timeout

    def _get(self, path):
        return _http.get_json(self.base + path, self.headers, self.timeout)

    def status(self) -> PrinterStatus:
        job = self._get("/api/job")
        prog = job.get("progress") or {}
        completion = fnum(prog.get("completion"))
        state = map_state(job.get("state", ""), completion)
        temps = {}
        try:
            temps = self._get("/api/printer?exclude=sd").get("temperature", {})
        except PrinterError as e:               # 409 = printer not operational
            if "409" not in str(e):
                raise
        bed, tool = temps.get("bed") or {}, temps.get("tool0") or {}
        f = (job.get("job") or {}).get("file") or {}
        return PrinterStatus(state=state, progress=None if completion is None else completion / 100.0,
                             filename=f.get("path") or f.get("name"), bed_c=fnum(bed.get("actual")),
                             bed_target_c=fnum(bed.get("target")), nozzle_c=fnum(tool.get("actual")),
                             nozzle_target_c=fnum(tool.get("target")), time_left_s=fnum(prog.get("printTimeLeft")),
                             elapsed_s=fnum(prog.get("printTime")), message=job.get("error") or "",
                             backend=self.name, raw={"job": job, "temperature": temps})

    def download_gcode(self, dest, filename=None) -> Path:
        st = self.status()
        f = (st.raw["job"].get("job") or {}).get("file") or {}
        origin = f.get("origin") or "local"
        path = filename or f.get("path")
        if not path:
            raise PrinterError("OctoPrint has no selected job file")
        if origin == "sdcard":
            raise PrinterError("the job runs from the printer's SD card; OctoPrint cannot download it. "
                               "Copy the file from the SD card or re-export it from the slicer (--gcode).")
        meta = self._get(f"/api/files/local/{urllib.parse.quote(path)}")
        url = (meta.get("refs") or {}).get("download") or f"{self.base}/downloads/files/local/{urllib.parse.quote(path)}"
        dest = Path(dest)
        if dest.is_dir() or str(dest).endswith("/"):
            dest = dest / Path(path).name
        return _http.download(url, dest, self.headers)
