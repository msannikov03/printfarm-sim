"""Build the PrintPick farm MJCF: rail carriage + YAM + 10 enclosed printers + caddy + packaging station.

The YAM comes from mujoco_menagerie (assets/yam). Everything else is generated here.
"""
import math
import pathlib
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from farm import catalog as C
from farm import lab

ROOT = pathlib.Path(__file__).resolve().parents[1]
YAM_DIR = ROOT / "assets" / "yam"
POOL_PER_SKU = max(12, round(1.25 * C.N_PRINTERS))   # free bodies per SKU kept parked until a print finishes
                                                     # (every printer can hold a plate at once)
N_PURGE = C.N_PRINTERS    # one purge tab per printer
DOOR_CLEAR = 0.004        # door leaf clearance to the plinth below and the header above (m)


def _f(*v):
    return " ".join(f"{x:.5g}" for x in v)


def _box(name, center, half, rgba=None, mat=None, cls=None, extra=""):
    a = f'name="{name}" ' if name else ""
    a += f'type="box" pos="{_f(*center)}" size="{_f(*half)}"'
    if mat:
        a += f' material="{mat}"'
    if rgba:
        a += f' rgba="{_f(*rgba)}"'
    if cls:
        a += f' class="{cls}"'
    return f"<geom {a} {extra}/>"


def _yam_parts():
    tree = ET.parse(YAM_DIR / "yam.xml")
    root = tree.getroot()
    default = root.find("default")
    # stiffer servo loops than the menagerie defaults so trajectories track well in sim
    for d in default.iter("default"):
        pos = d.find("position")
        if pos is None:
            continue
        if d.get("class") == "dm4340":
            pos.set("kp", "180"), pos.set("kv", "14")
        elif d.get("class") == "dm4310":
            pos.set("kp", "60"), pos.set("kv", "3")
        elif d.get("class") == "finger":
            pos.set("kp", "2000"), pos.set("kv", "60")
            pos.set("forcerange", "-20 20")
    asset = root.find("asset")
    arm = root.find("worldbody").find("body")
    arm.set("pos", f"0 0 {C.BASE_Z - C.DECK_Z:.4f}")
    # wrist camera on link_6, looking along the gripper (+z of link_6)
    for b in arm.iter("body"):
        if b.get("name") == "link_6":
            ET.SubElement(b, "camera", name="wrist", pos="0 -0.07 0.02", quat="0 0.966 0 0.259", fovy="75")
            # calibrated pinch point: centre of the finger pads (menagerie grasp_site is 44 mm off the pads)
            ET.SubElement(b, "site", name="pinch", pos="0 -0.044 0.1367", quat="1 0 0 -1", size="0.004",
                          rgba="1 0 1 1", group="5")
    for act in root.find("actuator"):
        if act.get("name") == "joint4":   # menagerie overrides kp on joint4; keep our class value
            act.attrib.pop("kp", None), act.attrib.pop("kv", None)
    env = ET.SubElement(default, "default", {"class": "env"})
    ET.SubElement(env, "geom", contype="1", conaffinity="1", group="1")
    part = ET.SubElement(default, "default", {"class": "part"})
    ET.SubElement(part, "geom", contype="1", conaffinity="1", condim="4", friction="1.2 0.02 0.001",
                  solref="0.004 1", solimp="0.95 0.99 0.001", group="1")
    s = lambda e: ET.tostring(e, encoding="unicode")
    return s(default), "".join(s(e) for e in asset), s(arm), "".join(s(e) for e in root.find("equality")), \
        "".join(s(e) for e in root.find("actuator"))


def _printer(i):
    xc, side = C.printer_frame(i)
    W, D, H = C.PRINTER_W, C.PRINTER_D, C.PRINTER_H
    quat = "1 0 0 0" if side > 0 else "0 0 0 1"
    g = []
    t = 0.008
    g.append(_box(None, (0, D / 2, 0.015), (W / 2, D / 2, 0.015), mat="shell"))                 # plinth
    g.append(_box(None, (-W / 2 + t / 2, D / 2, H / 2), (t / 2, D / 2, H / 2), mat="shell"))    # left
    g.append(_box(None, (W / 2 - t / 2, D / 2, H / 2), (t / 2, D / 2, H / 2), mat="shell"))     # right
    g.append(_box(None, (0, D - t / 2, H / 2), (W / 2, t / 2, H / 2), mat="shell"))             # back
    g.append(_box(None, (0, D / 2, H - t / 2), (W / 2, D / 2, t / 2), mat="roofglass"))         # glass lid
    g.append(_box(None, (-0.186, 0.008, H / 2), (0.009, 0.008, H / 2), mat="trim"))             # front posts
    g.append(_box(None, (0.186, 0.008, H / 2), (0.009, 0.008, H / 2), mat="trim"))
    g.append(_box(None, (0, 0.008, H - 0.015), (W / 2, 0.008, 0.015), mat="trim"))              # header
    # CoreXY gantry + parked toolhead (collision obstacles for the arm)
    g.append(_box(None, (-0.165, D / 2, 0.40), (0.012, D / 2 - 0.01, 0.012), mat="trim"))
    g.append(_box(None, (0.165, D / 2, 0.40), (0.012, D / 2 - 0.01, 0.012), mat="trim"))
    g.append(_box(None, (0, D - 0.05, 0.40), (0.165, 0.012, 0.012), mat="trim"))
    g.append(_box(f"toolhead{i}", (0.12, D - 0.05, 0.37), (0.035, 0.03, 0.045), mat="toolhead"))
    # bed carrier (visual) + bed plate (its own body so adhesion can be expressed relative to it)
    by = C.BED_FRONT_GAP + C.BED_MM[1] / 2000
    g.append(_box(None, (0, by, C.BED_ZL - 0.03), (0.11, 0.11, 0.02), mat="trim", extra='contype="0" conaffinity="0"'))
    bed = (f'<body name="bed{i}" pos="0 {by:.4f} {C.BED_ZL:.4f}">'
           + _box(f"bedplate{i}", (0, 0, -0.003), (C.BED_MM[0] / 2000, C.BED_MM[1] / 2000, 0.003), mat="pei",
                  extra='friction="0.6 0.01 0.001"')
           + f'<site name="bedc{i}" pos="0 0 0" size="0.004" group="5"/></body>')
    hx, hy, th = C.REJECT_TRAY          # reject tray on the plinth, in the front gap below the bed edge
    g.append(_bin(f"tray{i}", 0.0, C.BED_FRONT_GAP - hy - 0.002, 0.03, half=(hx, hy), h=th, mat="binreject"))
    cam = (f'<camera name="chamber{i}" pos="-0.16 0.03 {H - 0.03:.3f}" '
           f'xyaxes="0.62 -0.78 0 0.55 0.44 0.71" fovy="70"/>')
    # door: hinge on the left at the front plane; axis -z so +q opens outward
    # the leaf keeps DOOR_CLEAR to the plinth and the header: the printer body is welded to the world, so MuJoCo does
    # not filter door-printer contacts, and a zero gap made the leaf rub on the plinth through the whole swing
    dz0, dh = C.DOOR_Z0 + DOOR_CLEAR, C.DOOR_H - 2 * DOOR_CLEAR
    door = (
        f'<body name="door{i}" pos="{C.HINGE_XL:.4f} -0.004 0">'
        f'<joint name="door{i}" type="hinge" axis="0 0 -1" range="0 1.78" damping="1.5" frictionloss="0.05" armature="0.01"/>'
        + _box(f"doorglass{i}", (C.DOOR_W / 2, 0, dz0 + dh / 2), (C.DOOR_W / 2, 0.003, dh / 2), mat="glass",
               extra='mass="1.6"')
        + _box(None, (C.DOOR_W - 0.006, 0, dz0 + dh / 2), (0.006, 0.005, dh / 2), mat="trim", extra='mass="0.1"')
        + _box(None, (0.006, 0, dz0 + dh / 2), (0.006, 0.005, dh / 2), mat="trim", extra='mass="0.1"')
        # D-handle at the everyday handle spot: a vertical bar on two stand-offs, graspable from outside or
        # from between bar and glass. The bar spins freely (so a gripper can hold it while the door turns).
        + "".join(_box(f"handlearm{i}_{k}", (C.PIN_S, -(C.PIN_OFFSET + 0.004) / 2, C.PIN_ZL + sz * C.HANDLE_LEN / 2),
                       (0.009, (C.PIN_OFFSET - 0.004) / 2, 0.008), mat="robotorange", extra='mass="0.02"')
                  for k, sz in enumerate((-1, 1)))
        + f'<body name="pin{i}" pos="{C.PIN_S:.4f} {-C.PIN_OFFSET:.4f} {C.PIN_ZL:.4f}">'
          f'<joint name="pinspin{i}" type="hinge" axis="0 0 1" damping="0.001"/>'
          f'<geom name="pin{i}" type="cylinder" size="0.0075 {C.HANDLE_LEN / 2 + 0.008:.4f}" material="robotorange" '
          f'mass="0.02" friction="1.2 0.02 0.001"/>'
          f'<site name="pinsite{i}" pos="0 0 0" size="0.003" group="5"/></body>'
        + '</body>')
    return (f'<body name="printer{i}" pos="{xc:.4f} {side * C.AISLE_HALF:.4f} {C.BENCH_H:.4f}" quat="{quat}" '
            f'childclass="env">' + "".join(g) + bed + cam + door + '</body>')


def _bin(name, cx, cy, z0, half=(C.BIN_SIZE[0] / 2, C.BIN_SIZE[1] / 2), h=C.BIN_SIZE[2], mat="bin", floor=True):
    t = 0.003
    hx, hy = half
    g = ([_box(None, (cx, cy, z0 + t / 2), (hx, hy, t / 2), mat=mat)] if floor else []) + [
         _box(None, (cx - hx + t / 2, cy, z0 + h / 2), (t / 2, hy, h / 2), mat=mat),
         _box(None, (cx + hx - t / 2, cy, z0 + h / 2), (t / 2, hy, h / 2), mat=mat),
         _box(None, (cx, cy - hy + t / 2, z0 + h / 2), (hx, t / 2, h / 2), mat=mat),
         _box(None, (cx, cy + hy - t / 2, z0 + h / 2), (hx, t / 2, h / 2), mat=mat)]
    return "".join(g) + f'<site name="bin_{name}" pos="{_f(cx, cy, z0 + h)}" size="0.004" group="5"/>'


def deck_extent():
    """(x0, x1, y0, y1) of the carriage deck relative to the arm base. One row: the caddy sits behind the arm.
    Two rows: the deck must stay inside the aisle (benches are solid up to the deck height); its -x edge stays at
    -0.27 m because the folded slide pose (plan.door.LOW_TUCK) hangs the forearm past it."""
    if C.N_ROWS == 1:
        return -0.27, 0.27, -0.46, min(0.15, C.AISLE_HALF - 0.04)
    hx, hy = C.LARGE_SLOT_SIZE[0] / 2 + 0.015, C.LARGE_SLOT_SIZE[1] / 2 + 0.015
    boxes = [(dx, dy, 0.07, 0.07) for dx, dy in C.CADDY_SLOTS.values()] + \
            [(dx, dy, hx, hy) for dx, dy in C.LARGE_SLOTS.values()]
    y_lim = C.AISLE_HALF - 0.04
    return (min(-0.27, min(x - ex for x, _, ex, _ in boxes)), max(0.27, max(x + ex for x, _, ex, _ in boxes)),
            max(-y_lim, min(min(y - ey for _, y, _, ey in boxes), -0.12)),
            min(y_lim, max(max(y + ey for _, y, _, ey in boxes), 0.12)))


DECK_T = 0.03
TOTE_IN = C.BIN_SIZE[0] / 2 - 0.003          # tote inner half size (walls 3 mm)
FLAP_LEN = 2 * TOTE_IN - 0.002               # flap plate along y (1 mm gap to the hinge line and the far wall)
FLAP_HX = TOTE_IN - 0.001                    # flap half-width in x


def flap_slots():
    """Bins with a trap-door bottom: the six package totes and the waste bin (two rows; one row keeps plain bins)."""
    return tuple(s for s in C.CADDY_SLOTS if s != "reject") if C.TWO_ROWS else ()


def _plate_with_holes(x0, x1, y0, y1, holes):
    """Rectangle minus rectangular holes -> [(x0, x1, y0, y1)] boxes: grid cells cut at the hole edges, merged along
    y within each column."""
    xs = sorted({x0, x1, *(min(max(v, x0), x1) for h in holes for v in h[:2])})
    ys = sorted({y0, y1, *(min(max(v, y0), y1) for h in holes for v in h[2:])})
    out = []
    for xa, xb in zip(xs, xs[1:]):
        run = None
        for ya, yb in zip(ys, ys[1:]):
            cx, cy = (xa + xb) / 2, (ya + yb) / 2
            if not any(h[0] < cx < h[1] and h[2] < cy < h[3] for h in holes):
                run = (run[0], yb) if run else (ya, yb)
            elif run:
                out.append((xa, xb, *run))
                run = None
        if run:
            out.append((xa, xb, *run))
    return out


def _flap(slot, dx, dy):
    """Trap-door bottom of a tote: a plate hinged at the tote's +y inner edge, axis +x, so a positive angle swings its
    free (-y) edge down through the opening in the deck. Its top is the tote floor. A position actuator holds it
    against the joint stop at 0 (latched shut); the joint angle is read by an encoder (World.flap_sensor)."""
    t = C.FLAP_T
    return (f'<body name="flap_{slot}" pos="{_f(dx, dy + TOTE_IN, t / 2)}">'
            f'<joint name="flap_{slot}" type="hinge" axis="1 0 0" range="0 1.5" damping="0.02" armature="0.001"/>'
            + _box(f"flap_{slot}", (0, -(FLAP_LEN / 2 + 0.001), 0), (FLAP_HX, FLAP_LEN / 2, t / 2),
                   mat="binwaste" if slot == "waste" else "bin",
                   extra='mass="0.05" solref="0.004 1" solimp="0.95 0.99 0.001"')   # stiff: parts slide, not sink
            + f'<geom type="cylinder" fromto="{_f(-FLAP_HX + 0.004, -0.002, 0, FLAP_HX - 0.004, -0.002, 0)}" '
              f'size="0.003" material="trim" contype="0" conaffinity="0" mass="0"/>'
            + '</body>')


def _carriage(arm_xml):
    x0, x1, y0, y1 = deck_extent()
    holes = [(C.CADDY_SLOTS[s][0] - TOTE_IN, C.CADDY_SLOTS[s][0] + TOTE_IN,
              C.CADDY_SLOTS[s][1] - TOTE_IN, C.CADDY_SLOTS[s][1] + TOTE_IN) for s in flap_slots()]
    pieces = sorted(_plate_with_holes(x0, x1, y0, y1, holes), key=lambda b: -(b[1] - b[0]) * (b[3] - b[2]))
    g = [_box("deck" if k == 0 else f"deck{k}", ((a + b) / 2, (c + d) / 2, -DECK_T / 2),
              ((b - a) / 2, (d - c) / 2, DECK_T / 2), mat="carriage") for k, (a, b, c, d) in enumerate(pieces)]
    if C.TWO_ROWS:          # column under the arm, slim enough to stay clear of the station conveyor at full travel;
        (xa, xb), hy = C.COLUMN_X, C.COLUMN_HY   # its foot clears the rail's guide strips (the bogie runs on them)
        z0, z1 = 0.14 - C.DECK_Z, -DECK_T
        g.append(_box("column", ((xa + xb) / 2, 0, (z0 + z1) / 2), ((xb - xa) / 2, hy, (z1 - z0) / 2),
                      mat="carriage", extra='contype="0" conaffinity="0"'))
    else:
        g.append(_box("column", (0, 0, -0.25), (0.12, 0.08, 0.22), mat="carriage", extra='contype="0" conaffinity="0"'))
    if C.BASE_Z - C.DECK_Z > 0.01:
        g.append(f'<geom name="riser" type="cylinder" pos="0 0 {(C.BASE_Z - C.DECK_Z) / 2:.4f}" '
                 f'size="0.055 {(C.BASE_Z - C.DECK_Z) / 2:.4f}" material="trim"/>')
    for slot, (dx, dy) in C.CADDY_SLOTS.items():
        mat = "binwaste" if slot == "waste" else "binreject" if slot == "reject" else "bin"
        g.append(_bin(slot, dx, dy, 0.0, mat=mat, floor=slot not in flap_slots()))
        if slot in flap_slots():
            g.append(_flap(slot, dx, dy))

    for slot, (dx, dy) in C.LARGE_SLOTS.items():          # large-part tray: one long part per slot, laid along x
        g.append(_bin(slot, dx, dy, 0.0, half=(C.LARGE_SLOT_SIZE[0] / 2, C.LARGE_SLOT_SIZE[1] / 2),
                      h=C.LARGE_SLOT_SIZE[2], mat="trim"))
    return (f'<body name="carriage" pos="0 0 {C.DECK_Z:.4f}">'
            f'<joint name="rail" type="slide" axis="1 0 0" range="{C.RAIL_X_MIN} {C.RAIL_X_MAX}" damping="50" armature="5"/>'
            + "".join(g) + arm_xml + '</body>')


# ---------------------------------------------------------------- station (two rows)
BOX_BACKING = 0.006                          # invisible collision backing outside a box's walls (_box_body)


def box_outer(size):
    """Shipping box outer dims (world x, world y, height) in m: the box rides with its long side along the conveyor
    (world y); C.BOX_TYPES gives the inner (long, short, height)."""
    lx, ly, lz = (v / 1000 for v in C.BOX_TYPES[size])
    t = C.BOX_WALL_MM / 1000
    return ly + 2 * t, lx + 2 * t, lz + t


CHUTE_H, CHUTE_IN, CHUTE_OUT = 0.05, 0.136, 0.098   # drop chute height, top and bottom opening across x (m)


def _open_flap_low_z():
    """Lowest point of a flap at C.FLAP_OPEN (world z)."""
    a = C.FLAP_OPEN
    ys, zs = (-0.001, -(FLAP_LEN + 0.001)), (-C.FLAP_T / 2, C.FLAP_T / 2)
    return C.DECK_Z + C.FLAP_T / 2 + min(y * math.sin(a) + z * math.cos(a) for y in ys for z in zs)


def station_layout():
    """Two-row station geometry (world frame, m), derived from the deck and the catalog's station constants.
    Boxes are addressed by their bottom centre."""
    x0, x1, y0, y1 = deck_extent()
    xd, clr = C.X_DROP, C.STATION_CLEAR
    out = {s: box_outer(s) for s in C.BOX_TYPES}
    ly = max(o[1] for o in out.values())
    wall_y = C.AISLE_HALF + 0.64                       # farm/lab.py WALL_BACK_Y
    # drop chute: two steep plates along y under the tote rows (below the open flap's swing) funnel the falling
    # parts in x from the tote's 104 mm to CHUTE_OUT, well inside the narrowest box (S: 110 mm inside); the box rim
    # stays just under the chute while dumping
    chute_top = _open_flap_low_z() - 0.02             # a part sliding off the flap tip never bridges the gap
    chute_bot = chute_top - CHUTE_H
    rim_dump = chute_bot - 0.005
    totes = {s: xy for s, xy in C.CADDY_SLOTS.items() if s in flap_slots() and s != "waste"}
    tote_dy = [dy for _, dy in totes.values()]
    wdx, wdy = C.CADDY_SLOTS["waste"]
    y_mag = wall_y - 0.05 - ly / 2
    L = {
        "x_drop": xd, "z_conv": C.Z_CONV, "rim_dump": rim_dump, "chute_z": (chute_bot, chute_top),
        "chute_hx": (CHUTE_OUT / 2, CHUTE_IN / 2),
        "deck": (x0, x1, y0, y1), "deck_x1_max": C.RAIL_X_MAX + x1,
        "y_mag": y_mag, "mag_x": {"S": xd - 0.32, "M": xd, "L": xd + 0.30},
        "y_disp": y_mag - ly - 0.03,                   # cross-transfer line in front of the magazine
        "y_place": {s: y1 + clr + BOX_BACKING + o[1] / 2 for s, o in out.items()},   # clear of the deck's +y edge
        "z_place": C.PLACE_FLOOR_Z - C.BOX_WALL_MM / 1000,             # box bottom on the raised place lift
        "x_place": C.RAIL_X_MAX,                       # carriage position for the big-part place
        "dump_dy": -0.01,                              # box centre this far toward the flap's free edge
        "z_dump": {s: rim_dump - o[2] for s, o in out.items()},
        "dock": {s: xd - dx for s, (dx, _) in totes.items()},
        "dump_y": dict(totes),
        "y_seal": y0 - clr - 0.17, "seal_half": (0.16, 0.17),
        "y_sort": -wall_y + 0.05 + ly / 2,
        "lane_x": {c: C.RAIL_X_MAX + x1 + 0.35 + 0.45 * k for k, c in enumerate(C.LANE_ORDER)}, "y_lane": -0.50,
        "z_lane": 0.06,
        "x_scrap": xd + C.CONV_HALF_W + 0.03 + 0.07, "y_scrap": wdy, "scrap_half": (0.07, 0.10), "z_scrap_top": 0.30,
        "cam_pos": (xd, float(np.mean(tote_dy)), C.ROBOT_TOP_Z + clr),       # lens: straight down at the drop line
        "cam_post_x": C.RAIL_X_MAX + x1 + 0.20,                              # its boom's post, past the deck's end
        "lift_dump_y": (min(tote_dy) - 0.10, min(max(tote_dy) + 0.08, y1 + clr - 0.01)),
    }
    L["lift_place_y"] = (y1 + clr, y1 + clr + ly + 0.01)
    L["dock"]["waste"] = L["x_scrap"] - wdx
    L["dump_y"] = {s: dy for s, (_, dy) in totes.items()}
    return L


def _box_body(name, size, park):
    """A shipping box: floor + four walls that collide with parts and the arm (contype 2: not with other boxes or the
    conveyor), on three prescribed slide joints (farm.world moves it kinematically; parts inside are physical).
    The lid (hidden until sealing) is render-only."""
    ox, oy, h = box_outer(size)
    hx, hy = ox / 2, oy / 2
    t, bk = C.BOX_WALL_MM / 1000, BOX_BACKING
    col = ('contype="2" conaffinity="1" condim="4" friction="0.8 0.01 0.001" solref="0.004 1" '
           'solimp="0.95 0.99 0.001" material="cardboard"')   # as stiff as the parts: no sinking in
    g = [_box(None, (0, 0, t / 2), (hx, hy, t / 2), extra=col),
         _box(None, (hx - t / 2, 0, h / 2), (t / 2, hy, h / 2), extra=col),
         _box(None, (-hx + t / 2, 0, h / 2), (t / 2, hy, h / 2), extra=col),
         _box(None, (0, hy - t / 2, h / 2), (hx - t, t / 2, h / 2), extra=col),
         _box(None, (0, -hy + t / 2, h / 2), (hx - t, t / 2, h / 2), extra=col),
         # invisible backing (group 3) under the floor and outside the walls: a part falling ~0.2 m moves ~4 mm per
         # physics step, so a lone 4 mm sheet lets it tunnel through
         _box(None, (0, 0, -0.005), (hx + bk, hy + bk, 0.005), extra=col + ' group="3"'),
         _box(None, (hx + bk / 2, 0, h / 2), (bk / 2, hy + bk, h / 2), extra=col + ' group="3"'),
         _box(None, (-hx - bk / 2, 0, h / 2), (bk / 2, hy + bk, h / 2), extra=col + ' group="3"'),
         _box(None, (0, hy + bk / 2, h / 2), (hx, bk / 2, h / 2), extra=col + ' group="3"'),
         _box(None, (0, -hy - bk / 2, h / 2), (hx, bk / 2, h / 2), extra=col + ' group="3"'),
         _box(f"{name}_lid", (0, 0, h + 0.002), (hx + 0.001, hy + 0.001, 0.002), rgba=(0.74, 0.58, 0.4, 0),
              extra='contype="0" conaffinity="0" group="2"'),
         _box(f"{name}_tape", (0, 0, h + 0.0045), (0.024, hy + 0.002, 0.0006), rgba=(0.86, 0.75, 0.56, 0),
              extra='contype="0" conaffinity="0" group="2"')]
    j = "".join(f'<joint name="{name}_{a}" type="slide" axis="{ax}" armature="10" damping="0"/>'
                for a, ax in (("x", "1 0 0"), ("y", "0 1 0"), ("z", "0 0 1")))
    return (f'<body name="{name}" pos="{_f(*park)}">{j}<inertial pos="0 0 {h / 2:.4f}" mass="1" '
            f'diaginertia="0.01 0.01 0.01"/>' + "".join(g) + '</body>')


def box_pool():
    """[(body name, size)] of the shipping box pool."""
    return [(f"box{s}{k}", s) for s in C.BOX_TYPES for k in range(C.BOX_POOL)]


def box_park(k):
    return (-6.0 - 0.35 * k, 2.5, -1.0)


def _look_xyaxes(pos, target):
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    return (*r, *np.cross(r, f))


def _station_two():
    """Functional station hardware. Static parts are visual (contype 0) except surfaces a dropped part lands on
    (belt, scrap bin: contype 4, so parts hit them and boxes pass through); boxes and lift tables move kinematically.
    Dressing (rollers, legs, tunnel details, signs) is added by farm/lab.py."""
    L = station_layout()
    xd, zc, cw = L["x_drop"], L["z_conv"], C.CONV_HALF_W - 0.015
    vis = 'contype="0" conaffinity="0"'
    land = 'contype="4" conaffinity="1" solref="0.004 1" solimp="0.95 0.99 0.001"'
    ly = max(box_outer(s)[1] for s in C.BOX_TYPES)
    (ya, yb), (yp0, yp1) = L["lift_dump_y"], L["lift_place_y"]
    # belt sections along y (gaps where the two lift tables sit), surface 2 mm under the roller tops
    segs = [(L["y_sort"] - 0.15, ya), (yb, yp0), (yp1, L["y_mag"] + ly / 2)]
    g = [_box("stn_belt" if k == 0 else f"stn_belt{k}", (xd, (a + b) / 2, zc - 0.012), (cw, (b - a) / 2, 0.01),
              mat="belt", extra=land) for k, (a, b) in enumerate(segs)]
    xs1 = max(L["lane_x"].values()) + 0.3                # sorter along +x at y_sort
    g.append(_box("stn_sorter", ((xd + cw + xs1) / 2, L["y_sort"], zc - 0.012), ((xs1 - xd - cw) / 2, 0.15, 0.01),
                  mat="belt", extra=land))
    # magazine shelf (the erected-box stacks stand on it) and the cross-transfer in front of it
    xm0, xm1 = min(L["mag_x"].values()) - 0.12, max(L["mag_x"].values()) + 0.12
    g.append(_box("stn_magazine", ((xm0 + xm1) / 2, (L["y_disp"] + L["y_mag"]) / 2, zc - 0.012),
                  ((xm1 - xm0) / 2, (L["y_mag"] - L["y_disp"] + ly) / 2, 0.01), mat="belt", extra=vis))
    # drop chute under the tote rows (physical for parts: contype 4), clear polycarbonate
    (zb, zt), (hb, ht) = L["chute_z"], L["chute_hx"]
    ang = math.atan2(ht - hb, zt - zb)
    for e in (-1, 1):
        g.append(_box(f"stn_chute{'ab'[e > 0]}", (xd + e * ((hb + ht) / 2 + 0.0015), (ya + yb) / 2, (zb + zt) / 2),
                      (0.0015, (yb - ya) / 2, math.hypot(ht - hb, zt - zb) / 2), mat="glass",
                      extra=land + f' euler="0 {e * ang:.4f} 0"'))
    # sealer tunnel (visual): roof + side walls around the belt
    sx, sy = L["seal_half"]
    ys = L["y_seal"]
    g += [_box("stn_tunnel", (xd, ys, zc + 0.31), (sx, sy, 0.012), mat="robotorange", extra=vis),
          _box(None, (xd + sx - 0.01, ys, zc + 0.15), (0.01, sy, 0.15), mat="robotorange", extra=vis),
          _box(None, (xd - sx + 0.01, ys, zc + 0.15), (0.01, sy, 0.15), mat="robotorange", extra=vis)]
    # carrier lanes (open totes on the floor beside the sorter)
    for k, c in enumerate(C.LANE_ORDER):
        g.append(_box(f"stn_lane{k}", (L["lane_x"][c], L["y_lane"], 0.025), (0.22, 0.18, 0.025), mat=f"lane{k}",
                      extra=vis))
    # scrap bin under the waste flap's dock position (physical: purge tabs and failed prints land in it)
    hx, hy = L["scrap_half"]
    xsb, ysb, zt = L["x_scrap"], L["y_scrap"], L["z_scrap_top"]
    z0, t = 0.03, 0.004
    g += [_box("stn_scrap", (xsb, ysb, z0 + t / 2), (hx, hy, t / 2), mat="binwaste", extra=land),
          _box(None, (xsb + hx - t / 2, ysb, (z0 + zt) / 2), (t / 2, hy, (zt - z0) / 2), mat="binwaste", extra=land),
          _box(None, (xsb - hx + t / 2, ysb, (z0 + zt) / 2), (t / 2, hy, (zt - z0) / 2), mat="binwaste", extra=land),
          _box(None, (xsb, ysb + hy - t / 2, (z0 + zt) / 2), (hx, t / 2, (zt - z0) / 2), mat="binwaste", extra=land),
          _box(None, (xsb, ysb - hy + t / 2, (z0 + zt) / 2), (hx, t / 2, (zt - z0) / 2), mat="binwaste", extra=land)]
    # station depth camera straight above the drop line, on a boom from a post past the deck's +x end at full travel;
    # everything over the deck is above the robot's reach (C.ROBOT_TOP_Z) + the clearance. Image x along world +y.
    cp, px = np.array(L["cam_pos"]), L["cam_post_x"]
    zb = cp[2] + 0.05                                   # boom axis
    g.append(_box("stn_campost", (px, cp[1], (zb + 0.02) / 2), (0.02, 0.02, (zb + 0.02) / 2), mat="trim", extra=vis))
    g.append(_box("stn_camboom", ((px + cp[0] - 0.03) / 2, cp[1], zb), ((px - cp[0] + 0.03) / 2, 0.015, 0.015),
                  mat="trim", extra=vis))
    g.append(_box("stn_camera", cp + [0, 0, 0.0175], (0.015, 0.05, 0.0175), mat="toolhead", extra=vis))
    cam = (f'<camera name="station" pos="{_f(*cp)}" xyaxes="0 1 0 -1 0 0" fovy="{C.STATION_CAM_FOVY:g}"/>')
    # lift tables (mocap, visual): top = the box bottom while a box rides on them
    lifts = "".join(
        f'<body name="{n}" mocap="true" pos="{_f(xd, (a + b) / 2, zc - 0.002)}">'
        + _box(None, (0, 0, -0.008), (cw - 0.01, (b - a) / 2 - 0.005, 0.008), mat="rail", extra=vis)
        + _box(None, (0, 0, -0.15), (0.03, 0.03, 0.14), mat="rail", extra=vis) + '</body>'
        for n, (a, b) in (("lift_dump", (ya, yb)), ("lift_place", (yp0, yp1))))
    return "".join(g) + cam, lifts


def _part_body(name, sku, park_xyz):
    s = C.CATALOG[sku] if sku != "purge_tab" else None
    geoms = []
    cls = 'class="part"'
    if sku == "purge_tab":
        # body origin = the grab tab (C.PURGE_TAB_MM); a short 25 mm prime line runs to its right
        tx, ty, tz = (v / 2000 for v in C.PURGE_TAB_MM)
        geoms.append(_box(None, (0.0, 0.0, tz), (tx, ty, tz), mat="purge", extra=cls))
        geoms.append(_box(None, (tx + 0.0125, 0.0, 0.0006), (0.0125, 0.0018, 0.0006), mat="purge", extra=cls))
    elif s.kind == "box":
        x, y, z = (d / 1000 for d in s.dims)
        geoms.append(_box(None, (0, 0, z / 2), (x / 2, y / 2, z / 2), mat="pla", extra=cls))
    elif s.kind == "cyl":
        d, h = (v / 1000 for v in s.dims)
        geoms.append(f'<geom type="cylinder" pos="0 0 {h / 2:.4f}" size="{d / 2:.4f} {h / 2:.4f}" material="pla" {cls}/>')
    else:  # L bracket: base plate + upright at the -x end
        ln, w, bt, uh, ut = (v / 1000 for v in s.dims)
        geoms.append(_box(None, (0, 0, bt / 2), (ln / 2, w / 2, bt / 2), mat="pla", extra=cls))
        geoms.append(_box(None, (-ln / 2 + ut / 2, 0, uh / 2), (ut / 2, w / 2, uh / 2), mat="pla", extra=cls))
    mass = 0.002 if sku == "purge_tab" else max(s.mass_g, 1.0) / 1000
    return (f'<body name="{name}" pos="{_f(*park_xyz)}"><freejoint name="{name}"/>'
            f'<inertial pos="0 0 0.005" mass="{mass:.4f}" diaginertia="2e-6 2e-6 2e-6"/>'
            + "".join(geoms) + '</body>')


def part_pool():
    """Names of pooled part bodies: [(body_name, sku)]."""
    pool = [(f"{sku}_{k}", sku) for sku in C.CATALOG for k in range(POOL_PER_SKU)]
    pool += [(f"purge_tab_{k}", "purge_tab") for k in range(N_PURGE)]
    return pool


def _station():
    x0 = C.STATION_X + 0.45
    g = [_box(None, (x0 + 0.75, 0, 0.30), (0.85, 0.16, 0.30), mat="trim", extra='contype="0" conaffinity="0"'),
         _box("belt", (x0 + 0.75, 0, 0.605), (0.85, 0.14, 0.005), mat="belt", extra='contype="0" conaffinity="0"'),
         # packaging tunnel
         _box(None, (x0 + 0.85, 0, 0.88), (0.18, 0.2, 0.02), mat="robotorange", extra='contype="0" conaffinity="0"'),
         _box(None, (x0 + 0.85, 0.19, 0.74), (0.18, 0.012, 0.14), mat="robotorange", extra='contype="0" conaffinity="0"'),
         _box(None, (x0 + 0.85, -0.19, 0.74), (0.18, 0.012, 0.14), mat="robotorange", extra='contype="0" conaffinity="0"'),
         ]
    # three shipping lanes (carriers) after the tunnel
    for k, y in enumerate((-0.45, 0.0, 0.45)):
        g.append(_box(None, (x0 + 1.95, y, 0.25), (0.25, 0.16, 0.25), mat=f"lane{k}",
                      extra='contype="0" conaffinity="0"'))
    return "".join(g)


def build_xml(rail_x0=1.2):
    default, yam_assets, arm, equality, actuators = _yam_parts()
    pool = part_pool()
    park = [(-3.0 - 0.12 * (k // 12), -1.0 + 0.12 * (k % 12), -0.5) for k in range(len(pool))]   # under the floor, out of view
    parts = "".join(_part_body(n, s, park[k]) for k, (n, s) in enumerate(pool))
    welds = "".join(f'<weld name="stick_{n}" body1="{n}" active="false" solref="0.005 1"/>' for n, _ in pool)
    welds += "".join(f'<weld name="coupler{i}" body1="link_6" body2="pin{i}" active="false" solref="0.01 1"/>'
                     for i in range(C.N_PRINTERS))
    benches = "".join(
        _box(None, (C.N_PER_ROW * C.PITCH / 2 - C.PITCH / 2, s * (C.AISLE_HALF + 0.25), C.BENCH_H / 2),
             (C.N_PER_ROW * C.PITCH / 2 + 0.05, 0.25, C.BENCH_H / 2), mat="bench", cls="env")
        for s in ((+1, -1) if C.N_ROWS == 2 else (+1,)))
    # two rows: the rail ends RAIL_END past the column at full travel (the station conveyor follows after a gap)
    rx0 = C.RAIL_X_MIN - 0.4
    rx1 = C.RAIL_X_MAX + C.COLUMN_X[1] + C.RAIL_END if C.TWO_ROWS else C.RAIL_X_MAX + 0.4
    rail = _box(None, ((rx0 + rx1) / 2, 0, 0.06), ((rx1 - rx0) / 2, 0.06, 0.06), mat="rail",
                extra='contype="0" conaffinity="0"')
    printers = "".join(_printer(i) for i in range(C.N_PRINTERS))
    packages = "".join(
        f'<body name="pkg{k}" mocap="true" pos="{6 + k * 0.3} 3 -1"><geom type="box" size="0.07 0.09 0.05" '
        f'material="cardboard" contype="0" conaffinity="0"/></body>' for k in range(10))   # legacy (one row)
    station, lifts, boxes, flap_act = _station(), "", "", ""
    if C.TWO_ROWS:
        station, lifts = _station_two()
        boxes = "".join(_box_body(n, s, box_park(k)) for k, (n, s) in enumerate(box_pool()))
        flap_act = "".join(f'<position name="flap_{s}" joint="flap_{s}" kp="4" kv="0.15" ctrlrange="-0.05 1.5" '
                           f'forcerange="-1.5 1.5"/>' for s in flap_slots())
    return f"""<mujoco model="printpick_farm">
  <compiler angle="radian" meshdir="{(YAM_DIR / 'assets').as_posix()}" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="10" noslip_iterations="2"/>
  <size memory="200M"/>
  <visual>
    <global offwidth="1920" offheight="1080" azimuth="135" elevation="-25"/>
    <quality shadowsize="4096" offsamples="8"/>
    <headlight diffuse="0.45 0.45 0.45" ambient="0.35 0.35 0.37" specular="0.05 0.05 0.05"/>
    <map znear="0.01" zfar="50"/>
  </visual>
  {default}
  <asset>
    {yam_assets}
    <texture name="floor" type="2d" builtin="checker" rgb1="0.78 0.79 0.8" rgb2="0.74 0.75 0.76" width="512" height="512"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.95 0.96 0.97" rgb2="0.82 0.84 0.87" width="512" height="512"/>
    <material name="floor" texture="floor" texrepeat="8 8" reflectance="0.08"/>
    <material name="shell" rgba="0.16 0.17 0.19 1"/>
    <material name="trim" rgba="0.3 0.31 0.33 1"/>
    <material name="toolhead" rgba="0.22 0.22 0.24 1"/>
    <material name="glass" rgba="0.75 0.85 0.95 0.22" specular="0.6" shininess="0.9"/>
    <material name="roofglass" rgba="0.7 0.8 0.9 0.15"/>
    <material name="pei" rgba="0.55 0.45 0.25 1" specular="0.3"/>
    <material name="pla" rgba="0.95 0.55 0.15 1"/>
    <material name="purge" rgba="0.85 0.2 0.35 1"/>
    <material name="robotorange" rgba="0.95 0.45 0.1 1"/>
    <material name="carriage" rgba="0.2 0.42 0.72 1"/>
    <material name="rail" rgba="0.62 0.64 0.67 1" specular="0.5"/>
    <material name="bench" rgba="0.86 0.87 0.88 1"/>
    <material name="bin" rgba="0.18 0.62 0.42 1"/>
    <material name="binreject" rgba="0.85 0.2 0.25 1"/>
    <material name="binwaste" rgba="0.35 0.35 0.38 1"/>
    <material name="belt" rgba="0.12 0.12 0.13 1"/>
    <material name="cardboard" rgba="0.76 0.6 0.42 1"/>
    <material name="lane0" rgba="0.95 0.75 0.2 1"/>
    <material name="lane1" rgba="0.3 0.6 0.9 1"/>
    <material name="lane2" rgba="0.85 0.3 0.3 1"/>
  </asset>
  <worldbody>
    <light pos="1 0 4" dir="0 0 -1" directional="true" diffuse="0.55 0.55 0.55" castshadow="true"/>
    <light pos="-1 -2 3" dir="0.3 0.5 -1" diffuse="0.25 0.25 0.25" castshadow="false"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="floor" class="env"/>
    {rail}{benches}
    {printers}
    {_carriage(arm)}
    {station}
    {parts}
    {packages}{lifts}{boxes}
    <camera name="overview" pos="-1.6 -2.6 2.6" xyaxes="0.85 -0.53 0 0.33 0.53 0.78" fovy="55"/>
    <camera name="aisle" pos="-1.1 0 1.55" xyaxes="0 -1 0 0.45 0 0.89" fovy="60"/>
  </worldbody>
  <equality>{equality}{welds}</equality>
  <actuator>
    <position name="rail" joint="rail" kp="6000" kv="800" ctrlrange="{C.RAIL_X_MIN} {C.RAIL_X_MAX}" forcerange="-400 400"/>
    {actuators}{flap_act}
  </actuator>
</mujoco>"""


def build(path=None, decorate=True):
    xml = lab.decorate(build_xml()) if decorate else build_xml()
    if path:
        pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(path).write_text(xml)
    return mujoco.MjModel.from_xml_string(xml)


if __name__ == "__main__":
    m = build(ROOT / "build" / "printfarm.xml")
    print(f"nbody={m.nbody} njnt={m.njnt} nq={m.nq} nu={m.nu} ngeom={m.ngeom} neq={m.neq}")
