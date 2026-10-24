"""Where is each printed part on the bed? Parts from a sliced G-code file -> pick plan JSON.

Two paths:
  1. Labelled objects (preferred). Recognised labels, in priority order:
       - Klipper exclude-object: EXCLUDE_OBJECT_DEFINE NAME=.. CENTER=x,y POLYGON=[[x,y],..]
         + EXCLUDE_OBJECT_START/END NAME=..   (Orca / Creality Print "Exclude objects", Klipper flavour)
       - BambuStudio: "; start printing object, unique label id: N" ... "; stop printing object, unique label id: N"
         (always on); the id -> object name map is in the plate archive's Metadata/slice_info.config
       - "; printing object NAME" ... "; stop printing object NAME"
         (Orca / Creality Print / PrusaSlicer "Label objects")
       - Marlin "M486 S<id> [A<name>]" ... "M486 S-1"   (slicer "firmware-specific" labels for Marlin)
       - Cura ";MESH:name" ... ";MESH:NONMESH"
     Geometry always comes from the object's own extrusion moves (silhouette hull, min-area rect, max Z),
     with the DEFINE polygon as a fallback if an object has no moves.
  2. Unlabelled fallback: rasterise the extrusions of the first 3 layers into an occupancy grid,
     connected components = parts; each part's height = max Z of extrusions over its footprint.

Purge / prime lines and skirts are excluded: moves before the first layer marker (start G-code),
moves commented purge/prime/intro/wipe, ;TYPE:Skirt / prime tower (Bambu: "; FEATURE: Skirt"), and geometrically
thin lines. A brim stays with its object: it is printed fused to the part and comes off the bed with it.

Purge tab (docs/purge.md): a small printed object the start G-code primes into, picked first into the waste bin
with the prime line fused to it. It is the object named like one (PURGE_OBJECT, e.g. EXCLUDE_OBJECT_DEFINE
NAME=purge_tab), else the object whose footprint holds the end of the prime line (the unlabelled fallback):
role "purge" (other objects: "part"), plan["purge_object"], first in the pick order.

Bed frame = slicer frame: mm, origin at the bed's front-left corner (x right, y towards the back)
for Creality K1/K2/Ender machines. Pick order is nearest-to-robot first (default robot in front).

CLI:  python -m real.gcode_objects plate.gcode [--bed 220x220] [--robot-side front] [-o plan.json]
"""
from __future__ import annotations

import argparse
import html
import io
import json
import math
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError

# Bed sizes (printable area, mm) of common Creality machines, for when the G-code header has none.
# BELIEVED from Creality spec pages; the printable_area line in the G-code always wins.
KNOWN_BEDS = {
    "k1": (220, 220), "k1c": (220, 220), "k1se": (220, 220), "k1max": (300, 300),
    "k2plus": (350, 350), "k2pro": (300, 300), "k2": (260, 260), "hi": (260, 260),
    "ender3v3ke": (220, 220), "ender3v3se": (220, 220), "ender3v3": (220, 220),
    "ender3": (220, 220), "ender3v2": (220, 220), "ender3s1": (220, 220), "ender5s1": (220, 220),
}

PURGE_WORDS = re.compile(r"\b(purge|prime|priming|intro|wipe|nozzle clean|draw the (first|second) line|flush)", re.I)
SKIP_TYPES = re.compile(r"skirt|prime.?tower|wipe.?tower|custom", re.I)
PURGE_OBJECT = re.compile(r"purge|prime.?tab|calibration.?tab", re.I)     # object names of a purge tab
LAYER_MARK = re.compile(r"^;\s*(LAYER_CHANGE|LAYER:\s*-?\d+|CHANGE_LAYER|layer num/total_layer_count)", re.I)
NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)"
WORD = re.compile(rf"([A-Za-z])\s*({NUM})")


@dataclass
class Segments:
    """All extrusion segments of a file (numpy arrays of equal length)."""
    p0: np.ndarray            # (n, 2) xy start
    p1: np.ndarray            # (n, 2) xy end
    z: np.ndarray             # (n,)
    region: np.ndarray        # (n,) 0 = body, 1 = start gcode (before first layer marker)
    purge: np.ndarray         # (n,) bool, purge/prime comment on the move line
    skip_type: np.ndarray     # (n,) bool, ;TYPE: skirt / prime tower / custom
    labels: dict              # scheme -> (n,) object array of label or None
    defines: dict             # name -> {"center": [x, y], "polygon": [[x, y], ...]}
    header: dict              # parsed header/config values (printable_area, ...)
    has_layer_markers: bool


def read_gcode_text(path: str | Path) -> str:
    """Plain .gcode, or a .gcode.3mf / .3mf zip (Bambu/Orca/Creality Print plate archive)."""
    path = Path(path)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            names = sorted(n for n in z.namelist() if n.lower().endswith(".gcode"))
            if not names:
                raise ValueError(f"{path}: zip/3mf archive without a .gcode inside (export 'G-code', not the project)")
            return z.read(names[0]).decode("utf-8", "replace")
    return path.read_text(errors="replace")


def read_label_names(path: str | Path) -> dict:
    """BambuStudio plate archive (.gcode.3mf): G-code label id -> object name, from Metadata/slice_info.config
    ("<object identify_id=.. name=..>"). Names that occur twice get "#<id>" appended. {} for anything else."""
    path = Path(path)
    if not zipfile.is_zipfile(path):
        return {}
    with zipfile.ZipFile(path) as z:
        cfg = [n for n in z.namelist() if n.lower().endswith("slice_info.config")]
        if not cfg:
            return {}
        text = z.read(cfg[0]).decode("utf-8", "replace")
    pairs = []
    for tag in re.findall(r"<object\b[^>]*>", text):
        i, n = re.search(r'identify_id="(\d+)"', tag), re.search(r'name="([^"]*)"', tag)
        if i:
            pairs.append((int(i.group(1)), html.unescape(n.group(1)) if n else f"object_{i.group(1)}"))
    count = {}
    for _, n in pairs:
        count[n] = count.get(n, 0) + 1
    return {i: (n if count[n] == 1 else f"{n}#{i}") for i, n in pairs}


def _arc_points(x0, y0, x1, y1, i, j, cw: bool, step_mm: float = 2.0):
    cx, cy = x0 + i, y0 + j
    r = math.hypot(i, j)
    a0, a1 = math.atan2(y0 - cy, x0 - cx), math.atan2(y1 - cy, x1 - cx)
    if cw:
        while a1 >= a0 - 1e-9:
            a1 -= 2 * math.pi
    else:
        while a1 <= a0 + 1e-9:
            a1 += 2 * math.pi
    n = max(2, int(abs(a1 - a0) * r / step_mm) + 1)
    a = np.linspace(a0, a1, n + 1)
    return np.c_[cx + r * np.cos(a), cy + r * np.sin(a)]


def parse_segments(text: str) -> Segments:
    x = y = z = e = 0.0
    abs_xyz, abs_e = True, True
    scale = 1.0
    in_start = True
    has_layers = False
    cur = {"exclude": None, "bambu": None, "comment": None, "m486": None, "mesh": None}
    m486_names, m486_id = {}, -1
    in_flush = False
    cur_type = ""
    defines, header = {}, {}
    P0, P1, Z, R, PU, SK = [], [], [], [], [], []
    L = {k: [] for k in cur}

    def add(a, b, zz, purge):
        P0.append(a)
        P1.append(b)
        Z.append(zz)
        R.append(1 if in_start else 0)
        PU.append(purge)
        SK.append(bool(SKIP_TYPES.search(cur_type)))
        for k in cur:
            L[k].append(cur[k])

    for raw in io.StringIO(text):
        line = raw.strip()
        if not line:
            continue
        code, _, comment = line.partition(";")
        code = code.strip()
        cl = comment.strip()
        if not code:                                        # pure comment line
            if LAYER_MARK.match(line):
                in_start, has_layers = False, True
            elif m := re.match(r";\s*start printing object, unique label id:\s*(\d+)", line, re.I):
                cur["bambu"] = f"id_{m.group(1)}"                    # BambuStudio / newer Orca style
            elif re.match(r";\s*stop printing object, unique label id:", line, re.I):
                cur["bambu"] = None
            elif m := re.match(r";\s*printing object\s+(.+?)\s*$", line, re.I):
                cur["comment"] = re.sub(r"\s+id:\d+\s+copy\s+\d+$", "", m.group(1))
            elif re.match(r";\s*stop printing object", line, re.I):
                cur["comment"] = None
            elif m := re.match(r";MESH:(.+)$", line):
                name = m.group(1).strip()
                cur["mesh"] = None if name.upper() == "NONMESH" else name
            elif re.match(r";\s*FLUSH_START", line, re.I):           # K2 / CFS flush block
                in_flush = True
            elif re.match(r";\s*FLUSH_END", line, re.I):
                in_flush = False
            elif m := re.match(r";\s*(?:TYPE|FEATURE):\s*(.*)$", line, re.I):   # Bambu: "; FEATURE: Skirt"
                cur_type = m.group(1)
            elif m := re.match(r";\s*([A-Za-z_][\w ]*?)\s*[=:]\s*(.+)$", line):
                key = m.group(1).strip().lower().replace(" ", "_")
                if key in ("printable_area", "bed_shape", "printable_height", "max_z_height", "machine_name",
                           "printer_model", "printer_settings_id", "flavor", "exclude_object",
                           "gcode_label_objects", "label_objects", "machine_width", "machine_depth",
                           "bed_size", "extruder_offset"):
                    header.setdefault(key, m.group(2).strip())
            continue
        up = code.upper()
        if up.startswith("EXCLUDE_OBJECT_DEFINE"):
            nm = re.search(r'NAME=("[^"]*"|\S+)', code)
            cm = re.search(rf"CENTER=({NUM}),({NUM})", code)
            pm = re.search(r"POLYGON=(\[.*\])", code)
            if nm:
                name = nm.group(1).strip('"')
                d = defines.setdefault(name, {})
                if cm:
                    d["center"] = [float(cm.group(1)), float(cm.group(2))]
                if pm:
                    try:
                        d["polygon"] = json.loads(pm.group(1))
                    except json.JSONDecodeError:
                        pass
            continue
        if up.startswith("EXCLUDE_OBJECT_START"):
            nm = re.search(r'NAME=("[^"]*"|\S+)', code)
            cur["exclude"] = nm.group(1).strip('"') if nm else None
            continue
        if up.startswith("EXCLUDE_OBJECT_END"):
            cur["exclude"] = None
            continue
        if up.startswith("SET_PRINT_STATS_INFO") and "CURRENT_LAYER" in up:
            in_start, has_layers = False, True
            continue
        if not re.match(r"[GgMm]\d", code):              # Klipper macros, T0, ...
            continue
        if up.startswith("M486"):                           # Marlin/Prusa firmware object labels
            sm = re.search(r"\bS\s*(-?\d+)", code)
            am = re.search(r'\bA\s*"?([^"]+)"?', code)
            if sm:
                m486_id = int(sm.group(1))
                if am and m486_id >= 0:
                    m486_names[m486_id] = am.group(1).strip()
                cur["m486"] = None if m486_id < 0 else m486_names.get(m486_id, f"object_{m486_id}")
            elif am and cur["m486"] is not None:               # "M486 A<name>" names the current id
                m486_names[m486_id] = am.group(1).strip()
                cur["m486"] = m486_names[m486_id]
            continue
        words = WORD.findall(code)
        g = words[0][0].upper() + str(int(float(words[0][1])))
        args = {k.upper(): float(v) for k, v in words[1:]}
        if g == "G90":
            abs_xyz = True
        elif g == "G91":
            abs_xyz = False
        elif g == "M82":
            abs_e = True
        elif g == "M83":
            abs_e = False
        elif g == "G20":
            scale = 25.4
        elif g == "G21":
            scale = 1.0
        elif g == "G92":
            x = args["X"] * scale if "X" in args else x
            y = args["Y"] * scale if "Y" in args else y
            z = args["Z"] * scale if "Z" in args else z
            e = args["E"] if "E" in args else (0.0 if not args else e)
        elif g == "G28":
            x, y = 0.0, 0.0
        elif g in ("G0", "G1", "G2", "G3"):
            nx = (args["X"] * scale if abs_xyz else x + args["X"] * scale) if "X" in args else x
            ny = (args["Y"] * scale if abs_xyz else y + args["Y"] * scale) if "Y" in args else y
            nz = (args["Z"] * scale if abs_xyz else z + args["Z"] * scale) if "Z" in args else z
            de = 0.0
            if "E" in args:
                de = args["E"] - e if abs_e else args["E"]
                e = args["E"] if abs_e else e + args["E"]
            moved = abs(nx - x) > 1e-6 or abs(ny - y) > 1e-6
            if de > 1e-6 and moved:                         # (G0 with E extrudes too: Bambu primes with "G0 X240 E15")
                purge = in_flush or bool(PURGE_WORDS.search(cl))
                if g in ("G2", "G3") and ("I" in args or "J" in args):
                    pts = _arc_points(x, y, nx, ny, args.get("I", 0.0) * scale, args.get("J", 0.0) * scale, g == "G2")
                    for a, b in zip(pts[:-1], pts[1:]):
                        add(tuple(a), tuple(b), nz, purge)
                else:
                    add((x, y), (nx, ny), nz, purge)
            x, y, z = nx, ny, nz

    n = len(P0)
    return Segments(
        p0=np.asarray(P0, float).reshape(n, 2), p1=np.asarray(P1, float).reshape(n, 2),
        z=np.asarray(Z, float), region=np.asarray(R, int), purge=np.asarray(PU, bool),
        skip_type=np.asarray(SK, bool), labels={k: np.asarray(v, dtype=object) for k, v in L.items()},
        defines=defines, header=header, has_layer_markers=has_layers)


# ----------------------------------------------------------------------------- geometry helpers

def convex_hull(pts: np.ndarray) -> np.ndarray:
    pts = np.unique(np.round(np.asarray(pts, float), 4), axis=0)
    if len(pts) < 3:
        return pts
    try:
        return pts[ConvexHull(pts).vertices]
    except QhullError:                          # collinear
        d = pts - pts.mean(0)
        u = np.linalg.svd(d, full_matrices=False)[2][0]
        t = d @ u
        return pts[[np.argmin(t), np.argmax(t)]]


def min_area_rect(hull: np.ndarray):
    """Rotating calipers. Returns (center[2], (long, short), yaw_deg of the long side in (-90, 90])."""
    hull = np.asarray(hull, float)
    if len(hull) == 1:
        return hull[0], (0.0, 0.0), 0.0
    if len(hull) == 2:
        d = hull[1] - hull[0]
        return hull.mean(0), (float(np.hypot(*d)), 0.0), _norm_yaw(math.degrees(math.atan2(d[1], d[0])))
    best = None
    for a, b in zip(hull, np.roll(hull, -1, axis=0)):
        th = math.atan2(b[1] - a[1], b[0] - a[0])
        c, s = math.cos(th), math.sin(th)
        R = np.array([[c, s], [-s, c]])            # world -> rect frame
        q = hull @ R.T
        lo, hi = q.min(0), q.max(0)
        area = float(np.prod(hi - lo))
        if best is None or area < best[0] - 1e-9:
            best = (area, th, lo, hi, R)
    _, th, lo, hi, R = best
    center = ((lo + hi) / 2) @ R
    w, h = hi - lo
    yaw = math.degrees(th)
    if h > w:
        w, h, yaw = h, w, yaw + 90.0
    return center, (float(w), float(h)), _norm_yaw(yaw)


def _norm_yaw(deg: float) -> float:
    deg = (deg + 90.0) % 180.0 - 90.0              # (-90, 90]
    return 90.0 if abs(deg + 90.0) < 1e-9 else deg


def _sample_segments(p0, p1, step):
    out = []
    for a, b in zip(p0, p1):
        n = max(1, int(math.ceil(np.hypot(*(b - a)) / step)))
        t = np.linspace(0, 1, n + 1)[:, None]
        out.append(a + t * (b - a))
    return np.vstack(out) if out else np.zeros((0, 2))


def _geom(pts: np.ndarray) -> dict:
    hull = convex_hull(pts)
    c, (w, h), yaw = min_area_rect(hull)
    return dict(center_mm=[round(float(c[0]), 2), round(float(c[1]), 2)],
                footprint_mm=[round(w, 2), round(h, 2)], yaw_deg=round(yaw, 2),
                polygon_mm=np.round(hull, 2).tolist())


# ----------------------------------------------------------------------------- extraction

def _usable(seg: Segments, use_start: bool = False) -> np.ndarray:
    ok = ~seg.purge & ~seg.skip_type
    if seg.has_layer_markers and not use_start:
        ok &= seg.region == 0
    return ok


def labelled_objects(seg: Segments) -> tuple[str | None, list[dict]]:
    ok = _usable(seg)
    for scheme in ("exclude", "bambu", "comment", "m486", "mesh"):
        lab = seg.labels[scheme]
        names = [n for n in dict.fromkeys(lab[ok]) if n is not None]
        if scheme == "exclude" and not names and seg.defines:
            names = list(seg.defines)                     # defined but no START blocks with moves
        if not names:
            continue
        objs = []
        for name in names:
            sel = ok & (lab == name)
            d = seg.defines.get(name, {})
            if sel.any():
                pts = np.vstack([seg.p0[sel], seg.p1[sel]])
                o = dict(name=name, **_geom(pts), height_mm=round(float(seg.z[sel].max()), 3),
                         source="extrusions")
            elif "polygon" in d:
                o = dict(name=name, **_geom(np.asarray(d["polygon"], float)), height_mm=None,
                         source="define_polygon")
            else:
                continue
            if "center" in d:
                o["define_center_mm"] = d["center"]
            objs.append(o)
        return {"exclude": "exclude_object", "bambu": "bambu_label", "comment": "printing_object", "m486": "m486",
                "mesh": "cura_mesh"}[scheme], objs
    return None, []


def cluster_objects(seg: Segments, bed=None, cell_mm: float = 0.5, n_layers: int = 3, join_mm: float = 1.0):
    """Unlabelled fallback. Returns (objects, excluded) where excluded carries the reason."""
    ok = _usable(seg)
    if not ok.any():
        return [], []
    zs = np.unique(np.round(seg.z[ok], 3))
    zs = zs[zs > 0]
    if not len(zs):
        return [], []
    z_cut = zs[min(n_layers, len(zs)) - 1] + 1e-6
    first = ok & (seg.z <= z_cut)
    pts = _sample_segments(seg.p0[first], seg.p1[first], cell_mm / 2)
    lo = pts.min(0) - 5.0
    shape = tuple((np.ceil((pts.max(0) + 5.0 - lo) / cell_mm)).astype(int) + 1)
    occ = np.zeros(shape, bool)
    ij = np.floor((pts - lo) / cell_mm).astype(int)
    occ[ij[:, 0], ij[:, 1]] = True
    r = max(1, int(round(join_mm / cell_mm)))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    grown = ndimage.binary_dilation(occ, structure=(xx ** 2 + yy ** 2) <= r * r)
    lab, n = ndimage.label(grown, structure=np.ones((3, 3)))
    pt_lab = lab[ij[:, 0], ij[:, 1]]

    # heights: every usable extrusion endpoint over the whole file, looked up in the label image
    allp = np.vstack([seg.p0[ok], seg.p1[ok]])
    allz = np.concatenate([seg.z[ok], seg.z[ok]])
    aij = np.floor((allp - lo) / cell_mm).astype(int)
    inside = (aij >= 0).all(1) & (aij[:, 0] < shape[0]) & (aij[:, 1] < shape[1])
    alab = np.zeros(len(allp), int)
    alab[inside] = lab[aij[inside, 0], aij[inside, 1]]
    zmax = ndimage.maximum(allz, alab, index=np.arange(1, n + 1)) if n else []

    objs, excluded = [], []
    for k in range(1, n + 1):
        cpts = pts[pt_lab == k]
        if len(cpts) < 3:
            continue
        g = _geom(cpts)
        w, h = g["footprint_mm"]
        hull_area = _poly_area(np.asarray(g["polygon_mm"]))
        fill = (lab == k).sum() * cell_mm ** 2 / max(hull_area, 1e-6)
        reason = None
        near_edge = bed is not None and _near_edge(g["center_mm"], (w, h), g["yaw_deg"], bed, 10.0)
        if h <= 2.5 and w >= 6 * max(h, 0.4):
            reason = "purge_line (thin and long)"
        elif near_edge and h <= 4.0 and w >= 20:
            reason = "purge_line (thin, at bed edge)"
        elif fill < 0.25 and hull_area > 400:
            reason = "skirt (ring, mostly empty)"
        elif w * h < 4.0:
            reason = "speck (< 4 mm^2)"
        g["height_mm"] = round(float(zmax[k - 1]), 3)
        if reason:
            excluded.append(dict(reason=reason, **g))
        else:
            objs.append(g)
    objs.sort(key=lambda o: (o["center_mm"][1], o["center_mm"][0]))
    for i, o in enumerate(objs, 1):
        o["name"] = f"part_{i}"
        o["source"] = "cluster"
    objs = [{"name": o.pop("name"), **o} for o in objs]
    return objs, excluded


def _poly_area(p: np.ndarray) -> float:
    if len(p) < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _near_edge(c, wh, yaw, bed, margin):
    th = math.radians(yaw)
    ex = abs(wh[0] / 2 * math.cos(th)) + abs(wh[1] / 2 * math.sin(th))
    ey = abs(wh[0] / 2 * math.sin(th)) + abs(wh[1] / 2 * math.cos(th))
    return (c[0] - ex < margin or c[0] + ex > bed[0] - margin or
            c[1] - ey < margin or c[1] + ey > bed[1] - margin)


def bed_size(seg: Segments, override=None, printer_model: str | None = None):
    """(w, h, source). Override > G-code header > known model > extents of the print."""
    if override:
        return float(override[0]), float(override[1]), "cli"
    for key in ("printable_area", "bed_shape"):
        if key in seg.header:
            try:
                xy = np.array([[float(v) for v in p.split("x")] for p in seg.header[key].split(",")])
                return float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1])), f"gcode:{key}"
            except ValueError:
                pass
    if m := re.match(rf"\s*({NUM})\s*[xX*]\s*({NUM})", seg.header.get("bed_size", "")):
        return float(m.group(1)), float(m.group(2)), "gcode:bed_size"
    if "machine_width" in seg.header and "machine_depth" in seg.header:
        return float(seg.header["machine_width"]), float(seg.header["machine_depth"]), "gcode:machine_width/depth"
    if printer_model:
        key = re.sub(r"[^a-z0-9]", "", printer_model.lower())
        if key in KNOWN_BEDS:
            return *map(float, KNOWN_BEDS[key]), f"known_model:{key}"
    if len(seg.p1):
        hi = np.vstack([seg.p0, seg.p1]).max(0)
        return float(max(220.0, np.ceil(hi[0] / 10) * 10)), float(max(220.0, np.ceil(hi[1] / 10) * 10)), "guess:extents"
    return 220.0, 220.0, "guess:default"


ROBOT_SIDES = ("front", "back", "left", "right")


def pick_order(objs: list[dict], bed, robot_side: str = "front", row_mm: float = 15.0) -> list[str]:
    """Nearest-to-robot row first (rows = parts within row_mm of each other), tallest first inside a row."""
    w, h = bed
    dist = {"front": lambda c: c[1], "back": lambda c: h - c[1],
            "left": lambda c: c[0], "right": lambda c: w - c[0]}[robot_side]
    along = {"front": lambda c: c[0], "back": lambda c: -c[0], "left": lambda c: -c[1], "right": lambda c: c[1]}[robot_side]
    srt = sorted(objs, key=lambda o: dist(o["center_mm"]))
    rows, start = [], None
    for o in srt:
        d = dist(o["center_mm"])
        if start is None or d - start > row_mm:
            rows.append([])
            start = d
        rows[-1].append(o)
    order = []
    for row in rows:
        row.sort(key=lambda o: (-(o["height_mm"] or 0.0), along(o["center_mm"])))
        order += [o["name"] for o in row]
    return order


def prime_line_end(seg: Segments):
    """End point (x, y) of the last prime / purge extrusion (commented, or in the start G-code), or None."""
    pr = seg.purge | (seg.region == 1) if seg.has_layer_markers else seg.purge
    k = np.flatnonzero(pr)
    return None if not len(k) else seg.p1[k[-1]].astype(float)


def _inside(pt, poly, tol_mm: float) -> bool:
    """Point in (or within tol_mm of) a convex polygon given in either winding."""
    p = np.asarray(poly, float)
    if len(p) < 3:
        return False
    e = np.roll(p, -1, axis=0) - p
    v = np.asarray(pt, float) - p
    d = (e[:, 0] * v[:, 1] - e[:, 1] * v[:, 0]) / np.maximum(np.hypot(e[:, 0], e[:, 1]), 1e-9)   # + = left of edge
    return bool(np.all(d >= -tol_mm) or np.all(d <= tol_mm))


def mark_purge(objs: list[dict], line_end=None, tol_mm: float = 2.0):
    """Tag each object's role ("purge" / "part"): the purge tab is the object named like one, else the one whose
    footprint holds the prime line's end. Returns its name or None."""
    for o in objs:
        o["role"] = "part"
    cand = [o for o in objs if PURGE_OBJECT.search(str(o.get("name", "")))]
    if not cand and line_end is not None:
        cand = [o for o in objs if _inside(line_end, o.get("polygon_mm") or [], tol_mm)]
    if not cand:
        return None
    cand[0]["role"] = "purge"
    return cand[0]["name"]


def _order(objs, bed, robot_side, purge):
    """Pick order: the purge tab first (into the waste bin, its prime line with it), then the parts."""
    rest = pick_order([o for o in objs if o["name"] != purge], bed, robot_side)
    return ([purge] if purge else []) + rest


def extract(path=None, *, text=None, bed=None, printer_model=None, robot_side="front",
            force_cluster=False) -> dict:
    """Pick plan dict from a G-code file (path) or its contents (text)."""
    seg = parse_segments(text if text is not None else read_gcode_text(path))
    bw, bh, bsrc = bed_size(seg, bed, printer_model)
    method, objs = (None, []) if force_cluster else labelled_objects(seg)
    if method == "bambu_label":                  # "id_N" -> the object's name from the plate archive, when it has one
        names = read_label_names(path) if path is not None else {}
        for o in objs:
            o["label_id"] = int(o["name"][3:])
            o["name"] = names.get(o["label_id"], o["name"])
    excluded = []
    if not objs:
        objs, excluded = cluster_objects(seg, bed=(bw, bh))
        method = "cluster"
    if not seg.has_layer_markers:
        note = "no layer markers: start-gcode region unknown, purge detected by comments/geometry only"
    else:
        note = ""
    n_purge = int((seg.purge | (seg.region == 1)).sum()) if seg.has_layer_markers else int(seg.purge.sum())
    end = prime_line_end(seg)
    purge = mark_purge(objs, end)
    return dict(
        source=None if path is None else str(path),
        bed_mm=[bw, bh], bed_source=bsrc, method=method, labelled=method != "cluster",
        robot_side=robot_side, objects=objs, pick_order=_order(objs, (bw, bh), robot_side, purge),
        purge_object=purge, prime_line_end_mm=None if end is None else np.round(end, 2).tolist(),
        excluded=excluded, n_extrusion_segments=int(len(seg.z)), n_purge_or_start_segments=n_purge,
        note=note)


def plan_from_firmware_objects(objs: list, bed=(220.0, 220.0), height_mm: float | None = None,
                               robot_side: str = "front") -> dict:
    """Fallback when the G-code cannot be fetched: objects the firmware reports (Klipper exclude_object /
    Creality reqPrintObjects: name, center, polygon). Heights are unknown -> height_mm (assumed) for all."""
    out = []
    for i, o in enumerate(objs or []):
        poly = np.asarray(o.get("polygon") or [], float)
        if len(poly) >= 3:
            g = _geom(poly)
        elif o.get("center"):
            g = dict(center_mm=[float(v) for v in o["center"]], footprint_mm=[20.0, 20.0], yaw_deg=0.0, polygon_mm=[])
        else:
            continue
        out.append(dict(name=str(o.get("name") or f"object_{i}"), **g, height_mm=height_mm, source="firmware"))
    purge = mark_purge(out)                      # (by name only: the firmware does not report the prime line)
    return dict(source="firmware objects", bed_mm=[float(bed[0]), float(bed[1])], bed_source="cli/default",
                method="firmware_objects", labelled=True, robot_side=robot_side, objects=out,
                pick_order=_order(out, bed, robot_side, purge), purge_object=purge, excluded=[], n_extrusion_segments=0,
                n_purge_or_start_segments=0, note=f"heights unknown: assumed {height_mm} mm for every part")


def main(argv=None):
    ap = argparse.ArgumentParser(description="sliced G-code -> part list + pick order (JSON)")
    ap.add_argument("gcode")
    ap.add_argument("--bed", help="bed size WxH in mm, e.g. 220x220 (default: from G-code header)")
    ap.add_argument("--printer-model", help=f"fallback bed size by model: {', '.join(KNOWN_BEDS)}")
    ap.add_argument("--robot-side", default="front", choices=ROBOT_SIDES, help="bed edge the robot reaches from")
    ap.add_argument("--cluster", action="store_true", help="ignore labels, use the clustering fallback")
    ap.add_argument("-o", "--out", help="write the pick plan JSON here")
    a = ap.parse_args(argv)
    bed = tuple(float(v) for v in a.bed.lower().split("x")) if a.bed else None
    plan = extract(a.gcode, bed=bed, printer_model=a.printer_model, robot_side=a.robot_side, force_cluster=a.cluster)
    print_plan(plan)
    if a.out:
        Path(a.out).write_text(json.dumps(plan, indent=2))
        print("wrote", a.out)
    return plan


def print_plan(plan: dict, file=sys.stdout):
    print(f"bed {plan['bed_mm'][0]:.0f} x {plan['bed_mm'][1]:.0f} mm ({plan['bed_source']}), "
          f"method={plan['method']}, {len(plan['objects'])} parts", file=file)
    by = {o["name"]: o for o in plan["objects"]}
    for i, n in enumerate(plan["pick_order"], 1):
        o = by[n]
        h = o["height_mm"]
        tag = "  <- purge tab: waste bin (prime line attached)" if o.get("role") == "purge" else ""
        print(f"  {i}. {n:<24} c=({o['center_mm'][0]:6.1f},{o['center_mm'][1]:6.1f}) "
              f"fp={o['footprint_mm'][0]:5.1f}x{o['footprint_mm'][1]:<5.1f} yaw={o['yaw_deg']:6.1f} "
              f"h={'?' if h is None else f'{h:.1f}'}{tag}", file=file)
    for e in plan["excluded"]:
        print(f"  excluded: {e['reason']} at ({e['center_mm'][0]:.1f},{e['center_mm'][1]:.1f}) "
              f"{e['footprint_mm'][0]:.1f}x{e['footprint_mm'][1]:.1f} mm", file=file)
    if plan.get("note"):
        print("  note:", plan["note"], file=file)


if __name__ == "__main__":
    main()
