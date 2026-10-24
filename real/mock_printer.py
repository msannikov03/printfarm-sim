"""Mock printers for testing without hardware: print -> complete -> bed cooling, over real sockets.

    python -m real.mock_printer --kind moonraker            # http://127.0.0.1:7125
    python -m real.mock_printer --kind creality             # ws://127.0.0.1:9999 (+ Moonraker-style files on 7125)
    python -m real.mock_printer --kind octoprint            # http://127.0.0.1:5000, API key "mockkey"
    then:  python -m real.run_demo --printer moonraker://127.0.0.1:7125 --dry-run

Timeline (scaled by --print-s / --tau-s): printing for print_s (progress 0 -> 100 %, bed 60 C, nozzle 220 C),
then complete; heaters off; temperatures decay exponentially to 25 C with time constant tau_s.
The served G-code is a synthetic plate (real/synth_gcode.py) so the downstream parser has real input.
"""
from __future__ import annotations

import argparse
import json
import math
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from real.synth_gcode import SynthOpts, make_gcode

GCODE_NAME = "printpick_plate.gcode"
GCODE_DIR = "/usr/data/printer_data/gcodes"


class PrintSim:
    def __init__(self, print_s=20.0, tau_s=10.0, bed_c=60.0, nozzle_c=220.0, ambient_c=25.0, style="orca",
                 fail=None, clock=time.monotonic, bed=(220.0, 220.0)):
        self.print_s, self.tau_s = print_s, tau_s
        self.bed_set, self.noz_set, self.amb = bed_c, nozzle_c, ambient_c
        self.fail = fail                       # None | "cancelled" | "error": end state instead of complete
        self.clock = clock
        self.t0 = clock()
        self.gcode = make_gcode(opts=SynthOpts(style=style, bed=bed))
        self.started_wall = time.time()

    def snapshot(self) -> dict:
        t = self.clock() - self.t0
        if t < self.print_s:
            return dict(state="printing", progress=t / self.print_s, bed=self.bed_set, bed_target=self.bed_set,
                        nozzle=self.noz_set, nozzle_target=self.noz_set, elapsed=t, left=self.print_s - t)
        dt = t - self.print_s
        bed = self.amb + (self.bed_set - self.amb) * math.exp(-dt / self.tau_s)
        noz = self.amb + (self.noz_set - self.amb) * math.exp(-dt / (self.tau_s / 3))
        end = self.fail or "complete"
        return dict(state=end, progress=1.0 if end == "complete" else 0.6, bed=bed, bed_target=0.0, nozzle=noz,
                    nozzle_target=0.0, elapsed=self.print_s, left=0.0)

    def bed_ready_s(self, release_c=35.0) -> float:
        return self.print_s + self.tau_s * math.log((self.bed_set - self.amb) / (release_c - self.amb))


# ----------------------------------------------------------------------------- Moonraker

def _moonraker_handler(sim: PrintSim):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            s = sim.snapshot()
            ms = {"printing": "printing", "complete": "complete", "cancelled": "cancelled", "error": "error"}[s["state"]]
            if u.path == "/server/info":
                return self._json({"result": {"klippy_connected": True, "klippy_state": "ready",
                                              "moonraker_version": "v0.9.3-mock"}})
            if u.path == "/printer/info":
                return self._json({"result": {"state": "ready", "hostname": "mock-k1", "software_version": "mock"}})
            if u.path == "/printer/objects/query":
                objs = set(urllib.parse.parse_qs(u.query, keep_blank_values=True))
                full = {
                    "print_stats": {"state": ms, "filename": GCODE_NAME, "print_duration": s["elapsed"],
                                    "total_duration": s["elapsed"] + 5, "message": "" if ms != "error" else "MCU shutdown"},
                    "virtual_sdcard": {"progress": s["progress"], "is_active": ms == "printing",
                                       "file_path": f"{GCODE_DIR}/{GCODE_NAME}"},
                    "display_status": {"progress": s["progress"], "message": None},
                    "heater_bed": {"temperature": round(s["bed"], 2), "target": s["bed_target"], "power": 0.0},
                    "extruder": {"temperature": round(s["nozzle"], 2), "target": s["nozzle_target"], "power": 0.0},
                    "exclude_object": {"objects": [], "excluded_objects": [], "current_object": None},
                    "toolhead": {"position": [110.0, 200.0, 30.0, 0.0], "homed_axes": "xyz"},
                }
                from real.gcode_objects import parse_segments
                seg = parse_segments(sim.gcode)
                full["exclude_object"]["objects"] = [{"name": n, **d} for n, d in seg.defines.items()]
                return self._json({"result": {"eventtime": time.time(),
                                              "status": {k: v for k, v in full.items() if k in objs}}})
            if u.path == "/server/files/list":
                return self._json({"result": [{"path": GCODE_NAME, "modified": sim.started_wall,
                                               "size": len(sim.gcode), "permissions": "rw"}]})
            if u.path == "/server/files/metadata":
                return self._json({"result": {"filename": GCODE_NAME, "slicer": "mock", "object_height": 20.0,
                                              "estimated_time": sim.print_s}})
            if u.path.startswith("/server/files/gcodes/"):
                name = urllib.parse.unquote(u.path[len("/server/files/gcodes/"):])
                if name != GCODE_NAME:
                    return self._json({"error": {"code": 404, "message": f"File {name} does not exist"}}, 404)
                body = sim.gcode.encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self._json({"error": {"code": 404, "message": "Not Found"}}, 404)
    return H


# ----------------------------------------------------------------------------- OctoPrint

def _octoprint_handler(sim: PrintSim, key="mockkey"):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urllib.parse.urlparse(self.path)
            if u.path == "/":
                body = b"<html><head><title>OctoPrint</title></head></html>"
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)
                return
            if self.headers.get("X-Api-Key") != key:
                return self._json({"error": "Forbidden"}, 403)
            s = sim.snapshot()
            text = {"printing": "Printing", "complete": "Operational", "cancelled": "Operational",
                    "error": "Error: mock"}[s["state"]]
            if u.path == "/api/version":
                return self._json({"api": "0.1", "server": "1.10.0-mock", "text": "OctoPrint 1.10.0"})
            if u.path == "/api/job":
                comp = 100.0 if s["state"] == "complete" else s["progress"] * 100
                return self._json({"job": {"file": {"name": GCODE_NAME, "path": GCODE_NAME, "origin": "local"}},
                                   "progress": {"completion": comp, "printTime": int(s["elapsed"]),
                                                "printTimeLeft": int(s["left"])}, "state": text})
            if u.path == "/api/printer":
                if s["state"] == "error":
                    return self._json({"error": "Printer is not operational"}, 409)
                return self._json({"temperature": {"bed": {"actual": s["bed"], "target": s["bed_target"]},
                                                   "tool0": {"actual": s["nozzle"], "target": s["nozzle_target"]}},
                                   "state": {"text": text}})
            if u.path == f"/api/files/local/{GCODE_NAME}":
                host = self.headers.get("Host")
                return self._json({"name": GCODE_NAME, "origin": "local",
                                   "refs": {"download": f"http://{host}/downloads/files/local/{GCODE_NAME}"}})
            if u.path == f"/downloads/files/local/{GCODE_NAME}":
                body = sim.gcode.encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self._json({"error": "Not found"}, 404)
    return H


# ----------------------------------------------------------------------------- Creality OS websocket

def creality_frame(sim: PrintSim, full: bool) -> dict:
    s = sim.snapshot()
    code = {"printing": 1, "complete": 2, "cancelled": 4, "error": 4}[s["state"]]
    d = {"nozzleTemp": f"{s['nozzle']:.6f}", "bedTemp0": f"{s['bed']:.6f}",
         "targetNozzleTemp": int(s["nozzle_target"]), "targetBedTemp0": int(s["bed_target"]),
         "printProgress": int(s["progress"] * 100), "printJobTime": int(s["elapsed"]),
         "printLeftTime": int(s["left"]), "state": code, "err": {"errcode": 2003 if s["state"] == "error" else 0,
                                                                  "key": 0}}
    if full:
        d.update({"model": "K1C", "modelVersion": "printer hw ver:;printer sw ver:;DWIN hw ver:CR4CU220812S11;",
                  "hostname": "K1C-MOCK", "printFileName": f"{GCODE_DIR}/{GCODE_NAME}", "boxTemp": 30,
                  "TotalLayer": 100, "layer": int(s["progress"] * 100), "maxBedTemp": 100, "maxNozzleTemp": 300,
                  "curPosition": "X:110.00 Y:110.00 Z:25.00"})
    return d


def serve_creality(sim: PrintSim, host="127.0.0.1", port=9999, push_s=0.5, heartbeat_s=10.0):
    from websockets.sync.server import serve

    def handler(ws):
        def pusher():
            last_hb = time.monotonic()
            try:
                ws.send(json.dumps(creality_frame(sim, True)))
                while True:
                    time.sleep(push_s)
                    ws.send(json.dumps(creality_frame(sim, False)))
                    if time.monotonic() - last_hb > heartbeat_s:
                        ws.send(json.dumps({"ModeCode": "heart_beat", "msg": time.strftime("%Y-%m-%d %H:%M:%S")}))
                        last_hb = time.monotonic()
            except Exception:                            # noqa: BLE001 - client went away
                pass
        threading.Thread(target=pusher, daemon=True).start()
        try:
            for msg in ws:
                if msg == "ok":
                    continue
                try:
                    req = json.loads(msg)
                except json.JSONDecodeError:
                    continue
                params = req.get("params", {}) if isinstance(req, dict) else {}
                if "ReqPrinterPara" in params:
                    ws.send(json.dumps(creality_frame(sim, True)))
                if "reqGcodeFile" in params:
                    ws.send(json.dumps({"retGcodeFileInfo2": [{"name": GCODE_NAME, "path": f"{GCODE_DIR}/{GCODE_NAME}",
                                                               "file_size": len(sim.gcode)}]}))
                if "reqPrintObjects" in params:
                    from real.gcode_objects import parse_segments
                    seg = parse_segments(sim.gcode)
                    # the real K1C answers with JSON *strings* (HA capture): {"objects": "[...]", ...}
                    objs = [{"name": n, **d} for n, d in seg.defines.items()]
                    ws.send(json.dumps({"current_object": "", "excluded_objects": "[ ]", "objects": json.dumps(objs)}))
        except Exception:                                # noqa: BLE001
            pass

    def pick(conn, offered):                     # accept with or without the web UI's subprotocol
        return "wsslicer" if "wsslicer" in offered else None

    return serve(handler, host, port, select_subprotocol=pick)


# ----------------------------------------------------------------------------- launcher

class MockPrinter:
    """Start a mock in background threads. .url is the printer URL for make_printer(); .stop() to end."""

    def __init__(self, kind="moonraker", host="127.0.0.1", port=0, files_port=0, push_s=0.5, **sim_kw):
        self.kind, self.sim = kind, PrintSim(**sim_kw)
        self.servers = []
        if kind == "moonraker":
            srv = ThreadingHTTPServer((host, port), _moonraker_handler(self.sim))
            self._start_http(srv)
            self.url = f"moonraker://{host}:{srv.server_address[1]}"
        elif kind == "octoprint":
            srv = ThreadingHTTPServer((host, port), _octoprint_handler(self.sim))
            self._start_http(srv)
            self.url = f"octoprint://{host}:{srv.server_address[1]}?key=mockkey"
        elif kind == "creality":
            ws = serve_creality(self.sim, host, port, push_s=push_s)
            threading.Thread(target=ws.serve_forever, daemon=True).start()
            self.servers.append(ws)
            wsport = ws.socket.getsockname()[1]
            self.url = f"creality://{host}:{wsport}"
            self.files_port = None
            if files_port is not None:                   # K1C-2025 / K2 style Moonraker-compatible file API
                fsrv = ThreadingHTTPServer((host, files_port), _moonraker_handler(self.sim))
                self._start_http(fsrv)
                self.files_port = fsrv.server_address[1]
                self.url += f"?files_port={self.files_port}"
        else:
            raise ValueError(kind)

    def _start_http(self, srv):
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        self.servers.append(srv)

    def stop(self):
        for s in self.servers:
            s.shutdown()
            if hasattr(s, "server_close"):
                s.server_close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.stop()


def main(argv=None):
    ap = argparse.ArgumentParser(description="mock Creality/Klipper/OctoPrint printer")
    ap.add_argument("--kind", choices=["moonraker", "creality", "octoprint"], default="moonraker")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, help="default 7125 / 9999 / 5000")
    ap.add_argument("--print-s", type=float, default=60.0)
    ap.add_argument("--tau-s", type=float, default=20.0, help="bed cooling time constant")
    ap.add_argument("--style", default="orca", help="G-code labels: orca | orca_comments | m486 | cura | none")
    ap.add_argument("--fail", choices=["cancelled", "error"])
    a = ap.parse_args(argv)
    port = a.port or {"moonraker": 7125, "creality": 9999, "octoprint": 5000}[a.kind]
    m = MockPrinter(a.kind, a.host, port, files_port=7125 if a.kind == "creality" else 0, print_s=a.print_s,
                    tau_s=a.tau_s, style=a.style, fail=a.fail)
    print(f"mock {a.kind} up: --printer {m.url}" + (f"  (file API on :{m.files_port})" if a.kind == "creality" else ""))
    print(f"print ends in {a.print_s:.0f} s, bed <= 35 C at ~{m.sim.bed_ready_s():.0f} s. Ctrl-C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        m.stop()


if __name__ == "__main__":
    main()
