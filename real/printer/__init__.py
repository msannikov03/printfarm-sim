"""Printer backends behind one interface.

    from real.printer import make_printer
    p = make_printer("creality://192.168.1.57")          # stock Creality OS websocket, port 9999
                                                         # (?files_port=7125: where to try G-code download)
    p = make_printer("moonraker://192.168.1.57:7125")    # rooted Creality OS / Klipper / Sonic Pad (or :4408/:4409)
    p = make_printer("octoprint://192.168.1.60:80?key=ABC")
    p = make_printer("serial:///dev/tty.usbserial-1420?baud=115200")   # Marlin over USB
    st = p.status(); p.wait_until_done_and_cool(release_c=35)
"""
from __future__ import annotations

import urllib.parse

from real.printer.base import (DownloadUnsupported, PrintFailed, Printer, PrinterError, PrinterStatus,
                               wait_until_done_and_cool)

__all__ = ["make_printer", "Printer", "PrinterStatus", "PrinterError", "PrintFailed", "DownloadUnsupported",
           "wait_until_done_and_cool"]

DEFAULT_PORTS = {"moonraker": 7125, "creality": 9999, "octoprint": 80}


def make_printer(url: str) -> Printer:
    u = urllib.parse.urlparse(url if "://" in url else "moonraker://" + url)
    q = dict(urllib.parse.parse_qsl(u.query))
    kind = u.scheme.lower()
    if kind in ("moonraker", "klipper", "fluidd", "mainsail"):
        from real.printer.moonraker import MoonrakerPrinter
        return MoonrakerPrinter(u.hostname, u.port or DEFAULT_PORTS["moonraker"], api_key=q.get("key"))
    if kind in ("creality", "crealityos", "ws"):
        from real.printer.creality_ws import CrealityWSPrinter
        files = (int(q["files_port"]),) if "files_port" in q else (7125, 4408, 4409)
        return CrealityWSPrinter(u.hostname, u.port or DEFAULT_PORTS["creality"], moonraker_ports=files)
    if kind == "octoprint":
        from real.printer.octoprint import OctoPrintPrinter
        return OctoPrintPrinter(u.hostname, u.port or DEFAULT_PORTS["octoprint"], api_key=q.get("key"))
    if kind in ("serial", "marlin"):
        from real.printer.marlin_serial import MarlinSerialPrinter
        return MarlinSerialPrinter(u.path or u.netloc, baud=int(q.get("baud", 115200)))
    raise ValueError(f"unknown printer URL scheme '{kind}' (use creality:// moonraker:// octoprint:// serial://)")
