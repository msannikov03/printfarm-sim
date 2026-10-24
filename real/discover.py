"""Find the printer: LAN scan + fingerprint, and USB serial ports (printer vs SO-101 adapter).

    python -m real.discover                    # scan this Mac's /24 + list USB serial ports
    python -m real.discover --subnet 192.168.1.0/24
    python -m real.discover --host 192.168.1.57  # fingerprint one host
    python -m real.discover --serial-only --probe-serial   # also ask unknown ports for M115 (Marlin)

Ports: 80 (stock Creality web UI / nginx / OctoPrint), 4408 Fluidd, 4409 Mainsail (rooted Creality OS),
7125 Moonraker, 9999 Creality OS websocket, 5000 OctoPrint dev, 8080 misc web UIs.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import socket
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PORTS = (80, 4408, 4409, 7125, 9999, 5000, 8080)

# USB VID:PID hints. BELIEVED, from vendor docs / LeRobot + Creality community reports (see README).
USB_HINTS = {
    (0x1A86, 0x7523): "WCH CH340: typical Creality Marlin board (Ender-3/V2/S1, 4.2.x)",
    (0x1A86, 0x55D3): "WCH CH343: typical Waveshare/Feetech bus-servo adapter = SO-101 ARM (do not send G-code)",
    (0x1A86, 0x55D4): "WCH CH9102: some bus-servo adapters / ESP32 boards",
    (0x0483, 0x5740): "STM32 virtual COM: some Creality/BTT 32-bit boards (Marlin)",
    (0x1D50, 0x614E): "Klipper MCU (USB): printer MCU driven by a host, not directly",
    (0x2341, 0x0042): "Arduino Mega 2560 (RAMPS Marlin printers)",
    (0x10C4, 0xEA60): "Silicon Labs CP210x: some printers / ESP32",
    (0x0403, 0x6001): "FTDI FT232: some printers / servo adapters",
}


def local_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))        # no packet is sent
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def port_open(host: str, port: int, timeout=0.4) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _http_get(url, timeout=2.0, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read(200_000)


def _title(html: bytes) -> str:
    low = html.decode("utf-8", "replace")
    i, j = low.lower().find("<title>"), low.lower().find("</title>")
    return low[i + 7:j].strip()[:60] if 0 <= i < j else ""


def ws_hello(host: str, port: int = 9999, timeout=3.0) -> dict | None:
    """Connect to the Creality OS websocket and return the first JSON object it pushes (or None)."""
    try:
        from websockets.sync.client import connect
    except ImportError:
        return {"error": "websockets not installed"}
    try:
        with connect(f"ws://{host}:{port}/", subprotocols=["wsslicer"], open_timeout=timeout, close_timeout=1) as ws:
            merged = {}
            for _ in range(8):
                try:
                    msg = ws.recv(timeout=timeout)
                except TimeoutError:
                    break
                try:
                    d = json.loads(msg)
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(d, dict):
                    merged.update(d)
                    if any(k in merged for k in ("nozzleTemp", "bedTemp0", "model", "hostname", "state")):
                        break
            return merged
    except Exception as e:                        # noqa: BLE001 - any failure = not a Creality WS
        return {"error": f"{type(e).__name__}: {e}"}


def _creality_ws(host, p, out) -> bool:
    d = ws_hello(host, p)
    if d and "error" not in d:
        keys = sorted(d)[:12]
        model = d.get("model") or d.get("modelVersion") or d.get("hostname") or "?"
        out["found"].append(f"{p}: Creality OS websocket (model={model}; keys {keys})")
        out["suggest"].append(f"creality://{host}:{p}")
        return True
    if p == 9999:
        out["found"].append(f"{p}: open, but no Creality JSON ({(d or {}).get('error', 'silent')})")
    return False


def _http_probe(host, p, out) -> bool:
    """True if identified. Moonraker (/server/info), OctoPrint (/api/version), else page title."""
    try:
        code, body = _http_get(f"http://{host}:{p}/server/info")
        j = json.loads(body)
        if "result" in j and ("klippy_state" in j["result"] or "moonraker_version" in j["result"]):
            r = j["result"]
            out["found"].append(f"{p}: Moonraker {r.get('moonraker_version', '')} klippy={r.get('klippy_state')}")
            out["suggest"].append(f"moonraker://{host}:{p}")
            return True
    except Exception:                              # noqa: BLE001
        pass
    try:
        code, body = _http_get(f"http://{host}:{p}/api/version")
        j = json.loads(body)
        if "server" in j or "api" in j:
            out["found"].append(f"{p}: OctoPrint {j.get('server', '')} (API key needed for status)")
            out["suggest"].append(f"octoprint://{host}:{p}?key=YOUR_KEY")
            return True
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            out["found"].append(f"{p}: /api/version -> {e.code} (OctoPrint? needs X-Api-Key)")
            out["suggest"].append(f"octoprint://{host}:{p}?key=YOUR_KEY")
            return True
    except Exception:                              # noqa: BLE001
        pass
    try:                                           # stock Creality OS: GET /info -> {"model": ..., "mac": ...}
        code, body = _http_get(f"http://{host}:{p}/info")
        j = json.loads(body)
        if isinstance(j, dict) and "model" in j:
            out["found"].append(f"{p}: Creality OS HTTP /info model={j.get('model')} (status is on ws port 9999)")
            out["suggest"].append(f"creality://{host}:9999")
            return True
    except Exception:                              # noqa: BLE001
        pass
    try:
        code, body = _http_get(f"http://{host}:{p}/")
        low = body.lower()
        kind = ("Fluidd" if b"fluidd" in low else "Mainsail" if b"mainsail" in low else
                "OctoPrint" if b"octoprint" in low else "Creality web UI" if b"creality" in low else "web")
        out["found"].append(f"{p}: HTTP {code} {kind} title='{_title(body)}'")
        return kind != "web"
    except Exception:                              # noqa: BLE001 - e.g. a websocket-only port answers 426/400
        return False


def fingerprint(host: str, ports=PORTS) -> dict:
    open_ports = [p for p in ports if port_open(host, p)]
    out = {"host": host, "open": open_ports, "found": [], "suggest": []}
    for p in open_ports:
        if p == 9999:
            _creality_ws(host, p, out)
        elif not _http_probe(host, p, out) and not _creality_ws(host, p, out):
            if not any(f.startswith(f"{p}:") for f in out["found"]):
                out["found"].append(f"{p}: open, not identified")
    return out


def scan(subnet: str, ports=PORTS, workers=128):
    net = ipaddress.ip_network(subnet, strict=False)
    hosts = [str(h) for h in net.hosts()]
    with ThreadPoolExecutor(workers) as ex:
        alive = [h for h, ok in zip(hosts, ex.map(lambda h: any(port_open(h, p, 0.3) for p in ports), hosts)) if ok]
    with ThreadPoolExecutor(16) as ex:
        return list(ex.map(lambda h: fingerprint(h, ports), alive))


def serial_ports(probe=False, baud=115200):
    try:
        from serial.tools import list_ports
    except ImportError:
        return [{"device": "?", "note": "pyserial not installed"}]
    out = []
    for p in list_ports.comports():
        if p.vid is None:
            continue                               # Bluetooth / debug consoles
        hint = USB_HINTS.get((p.vid, p.pid), "unknown USB serial device")
        d = dict(device=p.device, vid_pid=f"{p.vid:04x}:{p.pid:04x}", desc=p.description,
                 manufacturer=p.manufacturer, serial=p.serial_number, hint=hint)
        if probe and "SO-101" not in hint:
            d["m115"] = probe_marlin(p.device, baud)
        out.append(d)
    return out


def probe_marlin(device, baud=115200, timeout=3.0) -> str:
    """Send M115 and return the firmware line. May reset an 8-bit board: never probe a printing machine."""
    import time
    try:
        from real.printer.marlin_serial import open_serial
        s = open_serial(device, baud, timeout=0.3)
    except Exception as e:                         # noqa: BLE001
        return f"cannot open: {e}"
    try:
        time.sleep(2.0)
        s.reset_input_buffer()
        s.write(b"M115\n")
        t0, lines = time.monotonic(), []
        while time.monotonic() - t0 < timeout:
            ln = s.readline().decode("ascii", "replace").strip()
            if ln:
                lines.append(ln)
                if "FIRMWARE_NAME" in ln or ln.startswith("ok"):
                    break
        fw = [ln for ln in lines if "FIRMWARE_NAME" in ln]
        return fw[0][:120] if fw else ("no reply" if not lines else lines[-1][:120])
    finally:
        s.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description="find the Creality printer on the LAN / USB")
    ap.add_argument("--subnet", help="CIDR to scan (default: this machine's /24)")
    ap.add_argument("--host", help="fingerprint one host only")
    ap.add_argument("--serial-only", action="store_true")
    ap.add_argument("--probe-serial", action="store_true", help="send M115 to non-arm USB serial ports")
    a = ap.parse_args(argv)
    print("== USB serial ports")
    ports = serial_ports(a.probe_serial)
    for d in ports:
        print(f"  {d['device']:<34} {d.get('vid_pid', ''):<10} {d.get('desc') or ''} | {d['hint']}"
              + (f" | M115: {d['m115']}" if "m115" in d else ""))
    if not ports:
        print("  none (printer USB not plugged in, or no driver)")
    if a.serial_only:
        return
    if a.host:
        results = [fingerprint(a.host)]
    else:
        ip = local_ip()
        subnet = a.subnet or str(ipaddress.ip_network(f"{ip}/24", strict=False))
        print(f"== scanning {subnet} (this machine {ip}), ports {PORTS} ...")
        results = scan(subnet)
    for r in results:
        print(f"\n{r['host']}: open {r['open']}")
        for f in r["found"]:
            print("   ", f)
        for s in r["suggest"]:
            print("    -> use: --printer", s)
    if not any(r["suggest"] for r in results):
        print("\nno printer API found. Check: printer on the same Wi-Fi/LAN (not a guest network with client "
              "isolation), its IP on the touchscreen (Settings > Network), or use --host IP.")
    return results


if __name__ == "__main__":
    main()
