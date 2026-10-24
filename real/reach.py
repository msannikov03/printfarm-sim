"""Hover poses over bed points and an ASCII reach map of the bed for a given calibration."""
from __future__ import annotations

import math

import numpy as np

from real.calib import BedCalib
from real.so101 import SO101, IKResult, hover_tcp


def jaw_dir_bed(obj: dict) -> np.ndarray:
    """Jaw closing direction in the bed frame: across the part's short side (footprint = [long, short])."""
    th = math.radians(obj.get("yaw_deg", 0.0) + 90.0)
    return np.array([math.cos(th), math.sin(th)])


def hover_target(calib: BedCalib, obj: dict, hover_mm: float = 40.0):
    """Robot-frame hover point (m) above the part top, and the jaw axis in the robot frame."""
    h = obj.get("height_mm") or 0.0
    c = obj["center_mm"]
    p = calib.bed_to_robot([c[0], c[1], h + hover_mm])
    jaw = calib.bed_dir_to_robot(jaw_dir_bed(obj))
    return p, jaw


def solve_object(kin: SO101, calib: BedCalib, obj: dict, hover_mm=40.0, q0=None, align_jaws=True) -> IKResult:
    p, jaw = hover_target(calib, obj, hover_mm)
    width = min(obj.get("footprint_mm", [20, 20])[1], 60.0)
    return kin.solve_hover(p, jaw=jaw if align_jaws else None, tcp=hover_tcp(width), q0=q0)


def reach_grid(calib: BedCalib, hover_mm=40.0, step_mm=20.0, kin=None):
    kin = kin or SO101()
    w, h = calib.bed_mm
    xs = np.arange(step_mm / 2, w, step_mm)
    ys = np.arange(step_mm / 2, h, step_mm)
    grid = np.empty((len(ys), len(xs)), dtype=object)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            r = kin.solve_hover(calib.bed_to_robot([x, y, hover_mm]), tcp=hover_tcp(20))
            grid[i, j] = r
    return xs, ys, grid


def print_reach_map(calib: BedCalib, hover_mm=40.0, step_mm=20.0, kin=None):
    """Rows = bed y (back at the top), columns = bed x. D = straight down, digits = tilt (pitch // 10),
    p = position-only (no orientation control), . = unreachable."""
    xs, ys, g = reach_grid(calib, hover_mm, step_mm, kin)
    print(f"reach map: tool tips {hover_mm:.0f} mm above the bed, {step_mm:.0f} mm grid "
          f"(D=down, 3-8=tilted 30-80 deg, p=position only, .=no)")
    for i in reversed(range(len(ys))):
        row = ""
        for j in range(len(xs)):
            r = g[i, j]
            if not r.ok:
                row += " ."
            elif r.pitch_deg is None:
                row += " p"
            elif r.pitch_deg >= 90:
                row += " D"
            else:
                row += f" {int(r.pitch_deg) // 10}"
        print(f"  y={ys[i]:5.0f} |{row}")
    print("          +" + "--" * len(xs) + "  (front edge, x = 0 .. %.0f mm)" % calib.bed_mm[0])
    ok = sum(r.ok for r in g.ravel())
    print(f"  reachable: {ok}/{g.size} cells")
    return xs, ys, g
