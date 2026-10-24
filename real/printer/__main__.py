"""Printer CLI: status / watch / wait / download / firmware objects.

    python -m real.printer creality://192.168.1.57             # one status line (+ model info)
    python -m real.printer moonraker://192.168.1.57 --watch    # status every 5 s
    python -m real.printer creality://IP --wait --release-c 35 # block until done + cool
    python -m real.printer creality://IP --download runs/real_gcode
    python -m real.printer creality://IP --objects             # objects the firmware knows (exclude_object)
"""
import argparse
import json
import time
from pathlib import Path

from real.printer import make_printer


def main(argv=None):
    ap = argparse.ArgumentParser(description="talk to the printer")
    ap.add_argument("url", help="creality://IP | moonraker://IP:7125 | octoprint://IP?key=K | serial:///dev/...")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--wait", action="store_true", help="wait until the print is done and the bed is cool")
    ap.add_argument("--release-c", type=float, default=35.0)
    ap.add_argument("--poll-s", type=float, default=5.0)
    ap.add_argument("--download", metavar="DIR", help="download the current/last G-code into DIR")
    ap.add_argument("--objects", action="store_true")
    ap.add_argument("--raw", action="store_true", help="dump the raw status dict")
    a = ap.parse_args(argv)
    p = make_printer(a.url)
    try:
        st = p.status()
        print(st.short())
        if hasattr(p, "info"):
            try:
                print("info:", p.info())
            except Exception as e:                       # noqa: BLE001
                print("info: n/a", e)
        if a.raw:
            print(json.dumps(st.raw, indent=1, default=str)[:4000])
        while a.watch:
            time.sleep(a.poll_s)
            print(p.status().short())
        if a.wait:
            st = p.wait_until_done_and_cool(a.release_c, a.poll_s, on_update=lambda ph, s: print(f"[{ph}]", s.short() if s else ""))
            print("READY:", st.short())
        if a.objects:
            objs = p.objects()
            print(f"firmware objects: {'none reported' if not objs else len(objs)}")
            for o in objs or []:
                print(f"  {o.get('name')}: center {o.get('center')} polygon {len(o.get('polygon') or [])} pts")
        if a.download:
            Path(a.download).mkdir(parents=True, exist_ok=True)
            print("downloaded:", p.download_gcode(Path(a.download)))
    except KeyboardInterrupt:
        pass
    finally:
        p.close()


if __name__ == "__main__":
    main()
