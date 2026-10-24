"""Mac-side client of the Pi arm server (real/arm_server.py). Standard library only.

    arm = RemoteArm("remote://192.168.1.50:8765")      # ?token=... if the server has one
    arm.connect()                                          # hello + takes control + heartbeat thread (4 Hz)
    arm.read_q()                                           # 6 model angles (rad)
    arm.move(q, speed_deg_s=20)                            # blocks until done; raises on abort / refusal
    arm.estop(); arm.close()

CLI (quick checks from the Mac):
    python -m real.arm_client remote://pi:8765 hello|status|read|stop|estop|torque-on|torque-off
    python -m real.arm_client remote://pi:8765 jog shoulder_pan +10      # one joint, degrees, slow
    python -m real.arm_client remote://pi:8765 gripper 60                # open 60 %
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import socket
import sys
import threading
import time
from urllib.parse import parse_qs, urlparse

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


class RemoteError(RuntimeError):
    pass


class MoveAborted(RemoteError):
    pass


def parse_url(url: str) -> tuple[str, int, str | None]:
    u = urlparse(url if "://" in url else "remote://" + url)
    if u.scheme not in ("remote", "tcp"):
        raise ValueError(f"expected remote://host:port, got {url}")
    tok = parse_qs(u.query).get("token", [None])[0]
    return u.hostname, u.port or 8765, tok


class RemoteArm:
    def __init__(self, url: str, timeout_s: float = 5.0, heartbeat_s: float = 0.25, takeover: bool = True):
        self.url = url
        self.host, self.port, self.token = parse_url(url)
        self.timeout_s, self.heartbeat_s, self.takeover = timeout_s, heartbeat_s, takeover
        self.sock = None
        self.f = None
        self.lock = threading.RLock()
        self._ids = itertools.count(1)
        self._stop = threading.Event()
        self.info: dict = {}

    # ------------------------------------------------------------------ connection
    def connect(self):
        try:
            self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        except OSError as e:
            raise RemoteError(f"cannot reach the arm server at {self.host}:{self.port} ({e}). Is it running on the Pi "
                              f"(python3 -m real.arm_server)? Tailnet up (tailscale ping {self.host})?") from e
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.f = self.sock.makefile("rwb")
        self.info = self.call("hello", token=self.token, takeover=self.takeover)
        if self.heartbeat_s:
            threading.Thread(target=self._beat, daemon=True).start()
        return self

    def close(self):
        self._stop.set()
        with self.lock:
            if self.sock is not None:
                try:
                    self.sock.close()
                finally:
                    self.sock = self.f = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()

    def _beat(self):
        while not self._stop.wait(self.heartbeat_s):
            if self.lock.acquire(timeout=self.heartbeat_s):
                try:
                    if self.sock is None:
                        return
                    self._call_locked("ping")
                except (RemoteError, OSError):
                    pass
                finally:
                    self.lock.release()

    def _call_locked(self, cmd, **kw):
        rid = next(self._ids)
        self.f.write((json.dumps({"id": rid, "cmd": cmd, **kw}) + "\n").encode())
        self.f.flush()
        while True:
            line = self.f.readline()
            if not line:
                raise RemoteError("arm server closed the connection")
            r = json.loads(line)
            if r.get("id") == rid:
                break
        if not r.get("ok"):
            raise RemoteError(r.get("error", "unknown error"))
        return r

    def call(self, cmd, **kw) -> dict:
        with self.lock:
            if self.sock is None:
                raise RemoteError("not connected")
            try:
                return self._call_locked(cmd, **kw)
            except socket.timeout as e:
                raise RemoteError(f"arm server did not answer '{cmd}' within {self.timeout_s} s") from e

    # ------------------------------------------------------------------ arm API (same as ArmController)
    def read_q(self) -> list:
        return self.call("read")["q"]

    def read_raw(self) -> dict:
        return self.call("read_raw")["raw"]

    def status(self) -> dict:
        return self.call("status")

    def move(self, q, speed_deg_s=None, accel_deg_s2=None, wait=True, timeout=None) -> dict:
        r = self.call("move", q=[float(v) for v in q], speed_deg_s=speed_deg_s, accel_deg_s2=accel_deg_s2)
        if not wait:
            return r
        t_end = time.monotonic() + (timeout or r.get("duration_s", 10) + 10)
        while True:
            r = self.call("wait", timeout=0.5)
            if r.get("state") != "moving":
                break
            if time.monotonic() > t_end:
                self.stop("client timeout")
                raise MoveAborted("move did not finish in time (stopped)")
        if r.get("state") == "aborted":
            raise MoveAborted(r.get("reason", "aborted"))
        return r

    def gripper(self, open_pct=None, rad=None, speed_deg_s=None, wait=True):
        kw = {"rad": rad} if rad is not None else {"open_pct": open_pct}
        r = self.call("gripper", speed_deg_s=speed_deg_s, **kw)
        if wait:
            while r.get("state") == "moving":
                r = self.call("wait", timeout=0.5)
        return r

    def stop(self, reason="stop requested"):
        return self.call("stop", reason=reason)

    def estop(self):
        return self.call("estop")

    def torque(self, on: bool):
        return self.call("torque", on=bool(on))["torque"]

    def set_keepout(self, bed=None, box_min=None, box_max=None):
        kw = {"bed": bed}
        if box_min is not None:
            kw.update(box_min=list(box_min), box_max=list(box_max))
        return self.call("set_keepout", **kw)["keepout"]


def bed_keepout(calib, margin_mm: float = 8.0) -> dict:
    """Bed keep-out for the server from a real.calib.BedCalib (p_robot = R p_bed/1000 + t)."""
    return dict(R=[[float(v) for v in row] for row in calib.R], t=[float(v) for v in calib.t],
                size_mm=[float(v) for v in calib.bed_mm], margin_mm=float(margin_mm))


def main(argv=None):
    ap = argparse.ArgumentParser(description="talk to the Pi arm server")
    ap.add_argument("url", help="remote://host:port[?token=...]")
    ap.add_argument("cmd", choices=["hello", "status", "read", "stop", "estop", "torque-on", "torque-off", "jog",
                                    "gripper", "watch"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--speed", type=float, default=15.0)
    a = ap.parse_args(argv)
    ro = a.cmd in ("hello", "status", "read", "watch", "estop", "stop")
    arm = RemoteArm(a.url, takeover=not ro).connect()
    try:
        if a.cmd == "hello":
            print(json.dumps(arm.info, indent=1))
        elif a.cmd == "status":
            s = arm.status()
            print(f"calib {s['calib']}  estopped {s['estopped']}  moving {s['moving']}  last {s['result']}")
            print("model deg", s["q_deg"], " tip mm", s["tip_mm"])
            for j, h in s["health"].items():
                if h:
                    print(f"  {j:<13} pos {h['pos']:5d} load {h['load_pct']:5.1f}% {h['volts']:4.1f} V {h['temp_c']} C"
                          f" torque {s['torque'].get(j)}" + (f" FLAGS {h['flags']}" if h["flags"] else ""))
                else:
                    print(f"  {j:<13} NO ANSWER")
            print("events:", *s["events"], sep="\n  ")
        elif a.cmd == "read":
            print([round(math.degrees(v), 2) for v in arm.read_q()])
        elif a.cmd == "watch":
            while True:
                print(" ".join(f"{math.degrees(v):7.1f}" for v in arm.read_q()), flush=True)
                time.sleep(0.2)
        elif a.cmd == "stop":
            print(arm.stop())
        elif a.cmd == "estop":
            print(arm.estop())
        elif a.cmd == "torque-on":
            print(arm.torque(True))
        elif a.cmd == "torque-off":
            input("torque OFF: support the arm, Enter ")
            print(arm.torque(False))
        elif a.cmd == "jog":
            j, d = a.args[0], float(a.args[1])
            q = arm.read_q()
            q[JOINTS.index(j)] += math.radians(d)
            print(arm.move(q, speed_deg_s=a.speed))
        elif a.cmd == "gripper":
            print(arm.gripper(float(a.args[0]), speed_deg_s=a.speed * 3))
    except KeyboardInterrupt:
        pass
    except RemoteError as e:
        sys.exit(f"error: {e}")
    finally:
        arm.close()


if __name__ == "__main__":
    main()
