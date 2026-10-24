"""Synthetic sliced G-code for tests and the mock printers.

Writes a plausible single-extruder file in one of four labelling styles:
  orca     - Orca / Creality Print with "Label objects" + "Exclude objects" (Klipper flavour):
             EXCLUDE_OBJECT_DEFINE header + "; printing object" / EXCLUDE_OBJECT_START blocks
  orca_comments - Orca / Creality Print with only "; printing object" comments (Marlin flavour)
  m486     - Marlin firmware labels "M486 S<id> A<name>" ... "M486 S-1" (PrusaSlicer/Orca for Marlin)
  cura     - Cura ";MESH:name" / ";MESH:NONMESH" blocks, ";LAYER:n" markers
  none     - no object labels at all (forces the clustering fallback)

Every style can carry an Ender-style prime/purge line at the left bed edge, a skirt, and arcs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Part:
    name: str
    center: tuple[float, float]
    size: tuple[float, float]          # (along yaw, across yaw) for a box; (d, d) for a cylinder
    height: float
    yaw_deg: float = 0.0
    shape: str = "box"                 # box | cyl

    def outline(self, n_cyl: int = 32) -> np.ndarray:
        if self.shape == "cyl":
            r = self.size[0] / 2
            a = np.linspace(0, 2 * math.pi, n_cyl, endpoint=False)
            loc = np.c_[r * np.cos(a), r * np.sin(a)]
        else:
            w, h = self.size[0] / 2, self.size[1] / 2
            loc = np.array([[-w, -h], [w, -h], [w, h], [-w, h]])
        c, s = math.cos(math.radians(self.yaw_deg)), math.sin(math.radians(self.yaw_deg))
        return loc @ np.array([[c, s], [-s, c]]) + np.asarray(self.center)


DEFAULT_PARTS = [
    Part("bracket", (70.0, 60.0), (40.0, 18.0), 12.0, yaw_deg=30.0),
    Part("knob", (140.0, 55.0), (24.0, 24.0), 18.0, shape="cyl"),
    Part("cube", (105.0, 110.0), (20.0, 20.0), 20.0, yaw_deg=-15.0),
]


@dataclass
class SynthOpts:
    style: str = "orca"                # orca | orca_comments | m486 | cura | none
    bed: tuple[float, float] = (220.0, 220.0)
    layer_h: float = 0.2
    purge: bool = True
    purge_comment: bool = True         # False: purge line has no comment (geometry heuristic must catch it)
    layer_markers: bool = True         # False: no ;LAYER markers (start-gcode region cannot be detected)
    skirt: bool = False
    relative_e: bool = False
    extra_header: list[str] = field(default_factory=list)


def _infill(poly: np.ndarray, spacing: float, angle_deg: float) -> list[tuple[np.ndarray, np.ndarray]]:
    """Zig-zag scanlines clipped to a convex polygon (enough to make the first layers solid)."""
    c, s = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    rot = np.array([[c, -s], [s, c]])
    p = poly @ rot                                   # rotate so scanlines are horizontal
    lines = []
    ys = np.arange(p[:, 1].min() + spacing / 2, p[:, 1].max(), spacing)
    for i, y in enumerate(ys):
        xs = []
        for a, b in zip(p, np.roll(p, -1, axis=0)):
            if (a[1] - y) * (b[1] - y) < 0:
                t = (y - a[1]) / (b[1] - a[1])
                xs.append(a[0] + t * (b[0] - a[0]))
        if len(xs) >= 2:
            x0, x1 = min(xs) + 0.2, max(xs) - 0.2
            seg = [np.array([x0, y]), np.array([x1, y])]
            if i % 2:
                seg.reverse()
            lines.append((seg[0] @ rot.T, seg[1] @ rot.T))
    return lines


class _Writer:
    def __init__(self, relative_e: bool):
        self.out: list[str] = []
        self.e = 0.0
        self.rel = relative_e
        self.x = self.y = 0.0

    def line(self, s: str):
        self.out.append(s)

    def travel(self, x, y, z=None):
        zs = f" Z{z:.3f}" if z is not None else ""
        self.out.append(f"G0 X{x:.3f} Y{y:.3f}{zs} F9000")
        self.x, self.y = x, y

    def extrude(self, x, y, comment: str = "", w: float = 0.45, h: float = 0.2):
        de = math.hypot(x - self.x, y - self.y) * w * h / 2.405   # 1.75 mm filament
        self.e += de
        e = de if self.rel else self.e
        c = f" ; {comment}" if comment else ""
        self.out.append(f"G1 X{x:.3f} Y{y:.3f} E{e:.5f} F1800{c}")
        self.x, self.y = x, y

    def arc_circle(self, cx, cy, r, h: float):
        """Full circle as two G2 half arcs starting at (cx + r, cy)."""
        self.travel(cx + r, cy)
        for (ex, ey), (i, j) in (((cx - r, cy), (-r, 0.0)), ((cx + r, cy), (r, 0.0))):
            de = math.pi * r * 0.45 * h / 2.405
            self.e += de
            e = de if self.rel else self.e
            self.out.append(f"G2 X{ex:.3f} Y{ey:.3f} I{i:.3f} J{j:.3f} E{e:.5f} F1800")
            self.x, self.y = ex, ey


def make_gcode(parts: list[Part] | None = None, opts: SynthOpts | None = None) -> str:
    parts = DEFAULT_PARTS if parts is None else parts
    o = opts or SynthOpts()
    w = _Writer(o.relative_e)
    bx, by = o.bed
    st = o.style
    # ---- header (Orca/Creality Print put the config block at the END; Cura puts metadata at the top)
    if st == "cura":
        w.line(";FLAVOR:Marlin")
        w.line(";TIME:3600")
        w.line(f";LAYER_COUNT:{int(max(p.height for p in parts) / o.layer_h)}")
        w.line(";Generated with Cura_SteamEngine 5.7.0")
    else:
        w.line("; HEADER_BLOCK_START")
        w.line("; generated by Creality_Print V6.2.1 (synthetic test file)")
        w.line(f"; max_z_height: {max(p.height for p in parts):.2f}")
        w.line("; HEADER_BLOCK_END")
    for h in o.extra_header:
        w.line(h)
    if st == "orca":
        for p in parts:
            poly = p.outline(16)
            pts = ",".join(f"[{x:.3f},{y:.3f}]" for x, y in poly)
            w.line(f"EXCLUDE_OBJECT_DEFINE NAME={p.name} CENTER={p.center[0]:.4f},{p.center[1]:.4f} POLYGON=[{pts}]")
    # ---- start gcode (machine start + prime line)
    w.line("G90")
    w.line("M83" if o.relative_e else "M82")
    w.line("M140 S60")
    w.line("M104 S220")
    w.line("G28")
    w.line("G92 E0")
    if o.purge:
        cm = "prime line" if o.purge_comment else ""
        w.travel(3.0, 20.0, 0.28)
        w.extrude(3.0, 160.0, cm)
        w.travel(3.6, 160.0)
        w.extrude(3.6, 20.0, cm)
        w.line("G92 E0")
        w.e = 0.0
    # ---- layers
    z_top = max(p.height for p in parts)
    n_layers = int(round(z_top / o.layer_h))
    for li in range(n_layers):
        z = round((li + 1) * o.layer_h, 3)
        if o.layer_markers:
            if st == "cura":
                w.line(f";LAYER:{li}")
            else:
                w.line(";LAYER_CHANGE")
                w.line(f";Z:{z:.3f}")
                w.line(f";HEIGHT:{o.layer_h:.3f}")
        w.line(f"G1 Z{z:.3f} F600")
        if li == 0 and o.skirt:
            allp = np.vstack([p.outline() for p in parts])
            lo, hi = allp.min(0) - 6, allp.max(0) + 6
            if st != "none" and o.layer_markers:
                w.line(";TYPE:Skirt" if st != "cura" else ";TYPE:SKIRT")
            if st == "cura":
                w.line(";MESH:NONMESH")
            w.travel(lo[0], lo[1])
            for x, y in [(hi[0], lo[1]), (hi[0], hi[1]), (lo[0], hi[1]), (lo[0], lo[1])]:
                w.extrude(x, y)
        for idx, p in enumerate(parts):
            if z > p.height + 1e-6:
                continue
            if st == "m486":
                w.line(f'M486 S{idx} A"{p.name}"')
            if st in ("orca", "orca_comments"):
                w.line(f"; printing object {p.name}")
            if st == "orca":
                w.line(f"EXCLUDE_OBJECT_START NAME={p.name}")
            if st == "cura":
                w.line(f";MESH:{p.name}")
                w.line(";TYPE:WALL-OUTER")
            elif st != "none":
                w.line(";TYPE:Outer wall")
            if p.shape == "cyl":
                w.arc_circle(p.center[0], p.center[1], p.size[0] / 2 - 0.2, o.layer_h)
            else:
                poly = p.outline()
                w.travel(*poly[0])
                for q in list(poly[1:]) + [poly[0]]:
                    w.extrude(*q)
            if li < 3 or li >= n_layers - 3:          # solid bottom/top
                if st == "cura":
                    w.line(";TYPE:SKIN")
                elif st != "none":
                    w.line(";TYPE:Bottom surface" if li < 3 else ";TYPE:Top surface")
                for a, b in _infill(p.outline(), 0.45, 45 + 90 * (li % 2) + p.yaw_deg):
                    w.travel(*a)
                    w.extrude(*b)
            if st == "orca":
                w.line(f"EXCLUDE_OBJECT_END NAME={p.name}")
            if st in ("orca", "orca_comments"):
                w.line(f"; stop printing object {p.name}")
            if st == "m486":
                w.line("M486 S-1")
            if st == "cura":
                w.line(";MESH:NONMESH")
    # ---- end gcode
    w.line("G1 E-2 F2400")
    w.line(f"G1 Z{z_top + 10:.2f} F600")
    w.line(f"G0 X{bx / 2:.1f} Y{by - 5:.1f} F6000")
    w.line("M140 S0")
    w.line("M104 S0")
    w.line("M84")
    if st != "cura":
        w.line("; CONFIG_BLOCK_START")
        w.line(f"; printable_area = 0x0,{bx:g}x0,{bx:g}x{by:g},0x{by:g}")
        w.line("; printable_height = 250")
        w.line("; exclude_object = " + ("1" if st == "orca" else "0"))
        w.line("; gcode_label_objects = " + ("1" if st in ("orca", "orca_comments") else "0"))
        w.line("; CONFIG_BLOCK_END")
    return "\n".join(w.out) + "\n"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="write a synthetic sliced G-code file")
    ap.add_argument("out")
    ap.add_argument("--style", default="orca", choices=["orca", "orca_comments", "m486", "cura", "none"])
    ap.add_argument("--no-purge", action="store_true")
    ap.add_argument("--skirt", action="store_true")
    a = ap.parse_args()
    open(a.out, "w").write(make_gcode(opts=SynthOpts(style=a.style, purge=not a.no_purge, skirt=a.skirt)))
    print("wrote", a.out)
