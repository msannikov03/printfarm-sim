"""Slicer stand-in: pack queued parts onto a P1S bed and emit plate.json + labelled-object G-code.

MaxRects with the bottom-left rule (parts gather toward the door), yaw 0/90, PART_SPACING_MM between parts,
the front PURGE_STRIP_MM kept free for the purge tab (front-centre, with a gripper keep-out around it). Parts are taken FIFO by order; an order that would fit
on a fresh plate is never split (it waits for the next plate while later orders fill the gaps).
With a reach mask every part's centre must lie in the robot's grasp map for that SKU and yaw (plan/bedmap.py --sku);
a small part's footprint must also lie in the probe-part map. Large SKUs (SKU.large) get JAW_KEEPOUT_MM more room
along their narrow side (where the open jaws come down). A part whose
own print time exceeds the cap still gets a plate (alone), so nothing in the queue can starve.
The G-code carries footprint-only toolpaths: enough for gcode_prior.parse, not a printable file.
"""
import argparse, json
from pathlib import Path
import numpy as np

from farm.catalog import BED_MM, CATALOG, LARGE_SKUS, PART_SPACING_MM, PURGE_STRIP_MM, SMALL_SKUS, print_time_s

MAX_PLATE_S = 3 * 3600.0            # cap on one plate's print time (ASSUMPTION)
BED_MARGIN_MM = 5.0                 # keep-out along the bed edges (ASSUMPTION)
from farm.catalog import PURGE_LINE_MM, PURGE_TAB_CLEAR_MM, PURGE_TAB_MM   # tab x, y, z; prime line (mm)
PURGE_TAB_CM3 = PURGE_TAB_MM[0] * PURGE_TAB_MM[1] * PURGE_TAB_MM[2] / 1000.0
LOOKAHEAD_ORDERS = 30               # orders scanned past the first one that was held back
JAW_KEEPOUT_MM = 10.0               # large parts: extra room on both long sides (jaws open near their full stroke)
E_PER_MM = 0.0333                   # 0.4 x 0.2 mm line on 1.75 mm filament


class MaxRects:
    """Free-rectangle packer over [x0, x1] x [y0, y1]; rects are (x, y, w, h)."""

    def __init__(self, x0, y0, x1, y1):
        self.free = [(x0, y0, x1 - x0, y1 - y0)]

    SCAN_MM = 8.0

    def find(self, w, h, ok=None):
        """Bottom-left position for a w x h rect, or None. With a mask (`ok`), positions inside each free rect
        are scanned on a SCAN_MM grid when its bottom-left corner is rejected."""
        best = None
        for fx, fy, fw, fh in self.free:
            if w > fw + 1e-9 or h > fh + 1e-9:
                continue
            if ok is None:
                cands = [(fx, fy)]
            else:
                xs = np.arange(fx, fx + fw - w + 1e-9, self.SCAN_MM)
                ys = np.arange(fy, fy + fh - h + 1e-9, self.SCAN_MM)
                cands = sorted(((float(x), float(y)) for y in ys for x in xs), key=lambda p: (p[1], p[0]))
            for x, y in cands:
                if best is not None and (y + h, x) >= best[0]:
                    break
                if ok is None or ok(x, y):
                    best = ((y + h, x), x, y)
                    break
        return None if best is None else best[1:]

    def place(self, x, y, w, h):
        out = []
        for f in self.free:
            fx, fy, fw, fh = f
            if x >= fx + fw or x + w <= fx or y >= fy + fh or y + h <= fy:
                out.append(f)
                continue
            if x > fx:
                out.append((fx, fy, x - fx, fh))
            if x + w < fx + fw:
                out.append((x + w, fy, fx + fw - x - w, fh))
            if y > fy:
                out.append((fx, fy, fw, y - fy))
            if y + h < fy + fh:
                out.append((fx, y + h, fw, fy + fh - y - h))
        inside = lambda a, b: (a[0] >= b[0] and a[1] >= b[1] and a[0] + a[2] <= b[0] + b[2]
                               and a[1] + a[3] <= b[1] + b[3])
        self.free = [a for i, a in enumerate(out)
                     if not any(j != i and inside(a, b) and (a != b or j < i) for j, b in enumerate(out))]


def _rect_ok(reach_mask, x, y, w, h):
    pts = [(x, y), (x + w, y), (x, y + h), (x + w, y + h), (x + w / 2, y + h / 2)]
    return all(reach_mask(px, py) for px, py in pts)


def _keepout(sku, w, h):
    """Extra keep-out (x, y) in mm on both sides of a large part's narrow side (the jaw axis)."""
    if not sku.large:
        return 0.0, 0.0
    return (JAW_KEEPOUT_MM, 0.0) if w < h else (0.0, JAW_KEEPOUT_MM)


def _object(part, x, y, w, h, yaw):
    sku = CATALOG[part["sku"]]
    cx, cy = round(x + w / 2, 3), round(y + h / 2, 3)
    poly = [[round(x, 3), round(y, 3)], [round(x + w, 3), round(y, 3)],
            [round(x + w, 3), round(y + h, 3)], [round(x, 3), round(y + h, 3)]]
    return {"name": part["name"], "order_id": part["order_id"], "sku": part["sku"],
            "center_mm": [cx, cy], "yaw_deg": float(yaw), "height_mm": float(sku.height),
            "footprint_mm": [float(v) for v in sku.footprint], "polygon_mm": poly}


class _Plate:
    def __init__(self, reach_mask, max_print_s, purge=None):
        s = PART_SPACING_MM   # rects are inflated by s; the bin is too, so parts may touch the margin line
        self.mr = MaxRects(BED_MARGIN_MM, PURGE_STRIP_MM, BED_MM[0] - BED_MARGIN_MM + s, BED_MM[1] - BED_MARGIN_MM + s)
        if purge is not None:      # keep-out around the tab: parts end >= PURGE_TAB_CLEAR_MM from it on every side
            (cx, cy), (w, d, _) = purge["center_mm"], purge["dims_mm"]
            g = PURGE_TAB_CLEAR_MM
            x0, x1, y1 = cx - w / 2 - g + s, cx + w / 2 + g, cy + d / 2 + g
            if y1 > PURGE_STRIP_MM:
                self.mr.place(x0, PURGE_STRIP_MM, x1 - x0, y1 - PURGE_STRIP_MM)
        self.mask, self.max_s = reach_mask, max_print_s
        self.vol, self.objects = PURGE_TAB_CM3, []

    def add(self, part):
        sku = CATALOG[part["sku"]]
        if print_time_s(self.vol + sku.volume_cm3) > self.max_s and self.objects:
            return False                 # (a part over the cap on its own is printed alone)
        s, best = PART_SPACING_MM, None
        for yaw in (0, 90):
            w, h = sku.footprint if yaw == 0 else sku.footprint[::-1]
            kx, ky = _keepout(sku, w, h)
            pos = self.mr.find(w + s + 2 * kx, h + s + 2 * ky, self._ok(sku, yaw, w, h, kx, ky))
            if pos and (best is None or (pos[1] + h + 2 * ky, pos[0]) < (best[1] + best[3] + 2 * best[6], best[0])):
                best = (*pos, w, h, yaw, kx, ky)
        if best is None:
            return False
        x, y, w, h, yaw, kx, ky = best
        self.mr.place(x, y, w + s + 2 * kx, h + s + 2 * ky)
        self.vol += sku.volume_cm3
        self.objects.append(_object(part, x + kx, y + ky, w, h, yaw))
        return True

    def _ok(self, sku, yaw, w, h, kx, ky):
        """Position test for a packed rect's corner (x, y): the part's centre in the grasp map of its own SKU at this
        yaw (when the mask has one), and for a small part its footprint corners and centre in the probe-part mask."""
        if self.mask is None:
            return None
        lm = getattr(self.mask, "sku", {}).get((sku.name, yaw))
        if lm is not None and sku.large:
            return lambda x, y: lm(x + kx + w / 2, y + ky + h / 2)
        if lm is not None:
            return lambda x, y: _rect_ok(self.mask, x, y, w, h) and lm(x + w / 2, y + h / 2)
        return lambda x, y: _rect_ok(self.mask, x + kx, y + ky, w, h)

    def add_all(self, parts):
        """Place all parts or none."""
        free, vol, n = list(self.mr.free), self.vol, len(self.objects)
        if all(self.add(p) for p in parts):
            return True
        self.mr.free, self.vol, self.objects = free, vol, self.objects[:n]
        return False


def _by_order(parts):
    groups = {}
    for p in parts:
        groups.setdefault(p["order_id"], []).append(p)
    return [sorted(g, key=lambda p: -np.prod(CATALOG[p["sku"]].footprint)) for g in groups.values()]


def pack(parts, reach_mask=None, max_print_s=MAX_PLATE_S, purge=None):
    """Choose and place parts (FIFO list of {"name", "order_id", "sku"}) -> plate.json object dicts.
    With `purge` (the tab dict) no part is placed within PURGE_TAB_CLEAR_MM of the tab."""
    plate, held = _Plate(reach_mask, max_print_s, purge), None
    for k, grp in enumerate(_by_order(parts)):
        if held is not None and k > held + LOOKAHEAD_ORDERS:
            break
        if plate.add_all(grp):
            continue
        if _Plate(reach_mask, max_print_s, purge).add_all(grp):
            held = k if held is None else held      # fits a fresh plate: keep it whole for the next one
        else:
            for p in grp:                           # too big for any one plate: split is unavoidable
                plate.add(p)
    return plate.objects


def _purge_tab(reach_mask):
    """Front-centre of the strip (nearest the robot, fixed by the start G-code); off-centre only if unreachable.
    The prime line runs PURGE_LINE_MM[0] along +x from the tab's right face: it must stay on the bed, also on the
    mirrored layout (plan/unload.py mirror_plate)."""
    w, d, _ = PURGE_TAB_MM
    cy = PURGE_STRIP_MM / 2
    reach = w / 2 + PURGE_LINE_MM[0] + BED_MARGIN_MM
    for cx in [BED_MM[0] / 2, *sorted(np.arange(10.0, BED_MM[0] - 10.0 + 1e-9, 5.0), key=lambda x: abs(x - BED_MM[0] / 2))]:
        if max(cx, BED_MM[0] - cx) + reach > BED_MM[0]:
            continue
        if reach_mask is None or _rect_ok(reach_mask, cx - w / 2, cy - d / 2, w, d):
            return {"center_mm": [float(cx), cy], "dims_mm": list(PURGE_TAB_MM)}
    raise ValueError("no reachable purge-tab position in the purge strip")


def sample_parts(rng, skus="small"):
    """Random part queue for the eval scripts (orders = consecutive pairs). 'small': 3-7 small parts (the legacy draw,
    same RNG stream); 'mixed': 1-2 large parts queued first, then 2-5 small; 'large': 1-3 large parts."""
    if skus == "small":
        names = rng.choice(list(SMALL_SKUS), int(rng.integers(3, 8)))
    elif skus == "mixed":
        names = [*rng.choice(list(LARGE_SKUS), int(rng.integers(1, 3))),
                 *rng.choice(list(SMALL_SKUS), int(rng.integers(2, 6)))]
    elif skus == "large":
        names = rng.choice(list(LARGE_SKUS), int(rng.integers(1, 4)))
    else:
        raise ValueError(f"skus must be small, mixed or large, not {skus!r}")
    return [{"name": f"o{k // 2}_{sku}_{k}", "order_id": f"o{k // 2}", "sku": str(sku)} for k, sku in enumerate(names)]


def make_plate(parts, printer, plate_id, rng, reach_mask=None, max_print_s=MAX_PLATE_S):
    """Pack `parts` (FIFO queue) into one plate -> (plate_dict, gcode_str). Unplaced parts stay queued."""
    purge = _purge_tab(reach_mask)
    objects = pack(parts, reach_mask, max_print_s, purge)
    vol = PURGE_TAB_CM3 + sum(CATALOG[o["sku"]].volume_cm3 for o in objects)
    plate = {"plate_id": plate_id, "printer": int(printer), "print_time_s": round(print_time_s(vol), 1),
             "bed_mm": [int(BED_MM[0]), int(BED_MM[1])], "purge_tab": purge, "objects": objects}
    return plate, to_gcode(plate)


def _loop(x0, y0, x1, y1, tag=""):
    pts = [(x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    seg = [abs(x1 - x0), abs(y1 - y0)] * 2
    return [f"G1 X{x:.3f} Y{y:.3f} E{d * E_PER_MM:.4f} F1800{tag}" for (x, y), d in zip(pts, seg)]


def to_gcode(plate):
    """OrcaSlicer/Klipper-style G-code with EXCLUDE_OBJECT labels (footprint-only toolpaths)."""
    t = int(plate["print_time_s"])
    objs = plate["objects"]
    g = ["; HEADER_BLOCK_START",
         "; generated by printfarm-sim jobs/plates.py (slicer stand-in, footprint-only toolpaths)",
         f"; plate_id = {plate['plate_id']}", f"; printer = {plate['printer']}",
         f"; bed_size = {plate['bed_mm'][0]}x{plate['bed_mm'][1]}", f"; total objects = {len(objs)}",
         f"; estimated printing time (normal mode) = {t // 3600}h {t % 3600 // 60}m {t % 60}s",
         "; HEADER_BLOCK_END", "", "; EXECUTABLE_BLOCK_START"]
    for o in objs:
        poly = json.dumps(o["polygon_mm"], separators=(",", ":"))
        g.append(f"EXCLUDE_OBJECT_DEFINE NAME={o['name']} CENTER={o['center_mm'][0]:.3f},{o['center_mm'][1]:.3f} "
                 f"POLYGON={poly}")
    g += ["M140 S60", "M104 S220", "G28", "M190 S60", "M109 S220", "G90", "M83", ""]
    (cx, cy), (w, d, h) = plate["purge_tab"]["center_mm"], plate["purge_tab"]["dims_mm"]
    x0, y0, x1, y1 = cx - w / 2, cy - d / 2, cx + w / 2, cy + d / 2
    ln = PURGE_LINE_MM[0]
    g += ["; prime line along the front strip, ending under the purge tab (fused to its first layer; docs/purge.md)",
          "G1 Z2.000 F1200", f"G1 X{x1 + ln:.3f} Y{cy:.3f} F12000", "G1 Z0.300 F600",
          f"G1 X{cx:.3f} Y{cy:.3f} E{(x1 + ln - cx) * E_PER_MM * 3:.4f} F900 ; prime line", ""]
    g += ["; purge tab (front strip), picked first into the waste bin", "G1 Z2.000 F1200",
          f"G1 X{x0:.3f} Y{y0:.3f} F12000", "G1 Z0.200 F600 ; purge", *_loop(x0, y0, x1, y1, " ; purge"),
          f"G1 Z{h:.3f} F600 ; purge", *_loop(x0, y0, x1, y1, " ; purge"), ""]
    for o in objs:
        (px0, py0), _, (px1, py1), _ = o["polygon_mm"]
        g += [f"G1 Z{max(o['height_mm'], h) + 2:.3f} F1200", f"G1 X{px0:.3f} Y{py0:.3f} F12000",
              f"EXCLUDE_OBJECT_START NAME={o['name']}", "G1 Z0.200 F600", *_loop(px0, py0, px1, py1),
              "; intermediate layers elided", f"G1 Z{o['height_mm']:.3f} F600", *_loop(px0, py0, px1, py1),
              f"EXCLUDE_OBJECT_END NAME={o['name']}", ""]
    top = max([h] + [o["height_mm"] for o in objs])
    g += ["; end: heaters off, bed parked at the unload height", f"G1 Z{top + 10:.3f} F1200",
          "M104 S0", "M140 S0", "M106 S0", "M84", "; EXECUTABLE_BLOCK_END", ""]
    return "\n".join(g)


def write_plate(plate, gcode, out_dir):
    d = Path(out_dir) / plate["plate_id"]
    d.mkdir(parents=True, exist_ok=True)
    (d / "plate.json").write_text(json.dumps(plate, indent=1))
    (d / "plate.gcode").write_text(gcode)
    return d


if __name__ == "__main__":
    from jobs.orders import OrderStream, part_sku
    ap = argparse.ArgumentParser(description="Write example plates (plate.json + plate.gcode) from the order stream.")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/plates")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    queue = [{"name": n, "order_id": o.order_id, "sku": part_sku(n)}
             for o in OrderStream(a.seed).generate(a.hours * 3600) for n in o.parts]
    for i in range(a.n):
        plate, gcode = make_plate(queue, i % 10, f"P{i:04d}", rng)
        done = {o["name"] for o in plate["objects"]}
        queue = [p for p in queue if p["name"] not in done]
        print(write_plate(plate, gcode, a.out), f"{len(done)} parts, {plate['print_time_s'] / 3600:.2f} h")
