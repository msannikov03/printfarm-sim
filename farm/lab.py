"""Lab dressing: make the farm MJCF read as a real print lab (room, lights, printer details, furniture).

decorate(xml) takes the string from farm.scene.build_xml() and returns it with additions only:
  - textures/materials (generated PNGs under assets/lab/, rebuilt on demand; assets/lab is not in git);
  - re-tuned existing materials, headlight and lights (looks only);
  - static geoms on the worldbody. No bodies, joints, actuators or equalities are added, so physics/planning
    indexing is unchanged (worldbody geom ids do shift; nothing in the code relies on raw geom ids).
Everything added is visual-only (contype=conaffinity=0, group 2), except the AMS units on top of the printers:
each has one collision box (class "env", hidden group 3), set back from the door plane.
Render cost: opaque untextured pieces are merged into one mesh per material (~600 primitives -> ~70 draw calls).
Walls, ceiling and lamp panels are one-sided mesh quads: they show from inside and vanish in outside views.
Renderer quirks handled in _lighting(): at most 8 lights, spot lights blacken surfaces behind them, and a
directional light's shadow map only covers geometry below its pos.
Two-row cell (PRINTFARM_LAYOUT=two_rows, row B on -y facing row A across the rail): a wall behind each row with the
same cable gap, an operator bay at the -x end and the packing line at +x; every service (tray, drops, power strip,
board) runs on its own row's wall, nothing crosses the aisle. Row B's wall items use "_wb" material variants so a
renderer can cut them away with cutaway() for views from outside that wall. The one-row output is unchanged.
"""
import math
import os
import pathlib
import xml.etree.ElementTree as ET

import numpy as np

from farm import catalog as C

ROOT = pathlib.Path(__file__).resolve().parents[1]
TEX_DIR = ROOT / "assets" / "lab"
TEX_VERSION = 5                     # bump to regenerate every texture
STATS = {}                          # last decorate(): primitives requested, geoms emitted
PRIMS = []                          # last decorate(): (lo, hi, section) world AABB of every dressing primitive, before
                                    # merging (tests/test_station.py sweeps them against the carriage)

# room (world frame, m)
TWO_ROWS = C.N_ROWS == 2            # PRINTFARM_LAYOUT=two_rows: row B on -y, mirrored about the rail
WALL_BACK_Y = C.AISLE_HALF + 0.64   # 14 cm behind the bench for cables
if TWO_ROWS:                        # a wall behind each row (same cable gap); operator bay at -x, packing at +x
    WALL_LEFT_X, WALL_RIGHT_X, WALL_FRONT_Y = -4.25, round(C.STATION_X + 4.25, 4), -WALL_BACK_Y
else:
    WALL_LEFT_X, WALL_RIGHT_X, WALL_FRONT_Y = -3.45, 8.95, -4.3
CEIL_Z = 3.0
MAX_LIGHTS = 8                      # the GL renderer drops every light past 8 (headlight included)
STATUS_RGBA = {"printing": (0.15, 0.85, 0.35, 1), "cooling": (1.0, 0.65, 0.1, 1), "ready": (0.2, 0.6, 1.0, 1),
               "waiting": (0.2, 0.6, 1.0, 1), "unloading": (1.0, 0.45, 0.08, 1), "idle": (0.75, 0.77, 0.8, 1)}


WALL_B = "_wb"                      # two-row cell: material suffix of everything mounted on row B's wall
WALL_B_MATS = ("lab_galv", "lab_cable", "lab_cable_blue", "lab_elec", "lab_frame", "lab_warning", "lab_plastic_white",
               "lab_skirt", "lab_sky", "lab_bezel", "lab_steel", "lab_alu", "lab_red", "lab_plastic_dark", "lab_fire")


def cutaway(m, hide=True):
    """Two-row cell, render-only: hide (alpha 0) or restore everything mounted on row B's wall, so a camera outside
    that wall sees a clean cutaway (the wall itself is one-sided and already vanishes from outside)."""
    import mujoco
    for k in range(m.nmat):
        if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, k) or "").endswith(WALL_B):
            m.mat_rgba[k][3] = 0.0 if hide else 1.0


def set_status(m, i, state):
    """Recolour printer i's status light bar (render-only: material colour) to one of STATUS_RGBA's states."""
    import mujoco
    mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, f"lab_status{i}")
    if mid >= 0:
        m.mat_rgba[mid] = STATUS_RGBA.get(state, STATUS_RGBA["idle"])
# geom groups: 0-2 physics + dressing (seen by the chamber depth cameras); 4 = beauty-only detail inside the printers
# (farm.render.Cam shows it, perception does not, so depth input is unchanged); 5 = showcase set dressing (prints in
# progress on the beds), shown only by scripts/render_showcase.py
GROUP_DETAIL, GROUP_SHOWCASE = 4, 5
SPOOL_COLORS = {
    "white": (0.93, 0.93, 0.91), "black": (0.07, 0.07, 0.08), "orange": (0.96, 0.42, 0.06),
    "blue": (0.08, 0.32, 0.82), "green": (0.12, 0.62, 0.28), "red": (0.82, 0.1, 0.1),
    "grey": (0.52, 0.53, 0.55), "yellow": (0.98, 0.8, 0.08), "purple": (0.45, 0.22, 0.68),
    "teal": (0.05, 0.58, 0.58),
}


def _f(*v):
    return " ".join(f"{x:.5g}" for x in v)


# ---------------------------------------------------------------- textures (numpy + Pillow, deterministic)
def _noise(shape, sigma, rng):
    """Tileable Gaussian-filtered white noise, unit std."""
    w = rng.standard_normal(shape)
    ky = np.fft.fftfreq(shape[0])[:, None]
    kx = np.fft.fftfreq(shape[1])[None, :]
    sy, sx = (sigma, sigma) if np.isscalar(sigma) else sigma
    g = np.exp(-2 * np.pi ** 2 * ((sx * kx) ** 2 + (sy * ky) ** 2))
    out = np.real(np.fft.ifft2(np.fft.fft2(w) * g))
    return out / (out.std() + 1e-9)


def _u8(a):
    return (np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)


def _rgb(base, field):
    return np.stack([base[k] + field for k in range(3)], -1)


def _font(size):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except Exception:                   # Pillow < 10.1 or no FreeType: small bitmap font
        return ImageFont.load_default()


def _tex_floor(rng):
    n = 1024                                            # one tile = 4 m x 4 m, saw cuts every 2 m
    f = (0.022 * _noise((n, n), 70, rng) + 0.012 * _noise((n, n), 14, rng) + 0.007 * _noise((n, n), 1.5, rng))
    speck = rng.random((n, n))
    f -= 0.07 * (speck < 0.004) - 0.05 * (speck > 0.997)
    f += 0.012 * _noise((n, n), (1.0, 60.0), rng) * (_noise((n, n), 90, rng) > 1.0)     # faint wheel scuffs
    img = _rgb((0.70, 0.705, 0.70), f)
    for c in (0, n // 2):
        for d, a in ((-1, 0.5), (0, 1.0), (1, 1.0), (2, 0.5)):
            img[(c + d) % n, :, :] *= 1 - 0.18 * a
            img[:, (c + d) % n, :] *= 1 - 0.18 * a
    return img


def _tex_ceiling(rng):
    n = 512                                             # 2 x 2 acoustic tiles of 0.6 m
    img = _rgb((0.9, 0.9, 0.89), 0.025 * _noise((n, n), 1.2, rng) + 0.01 * _noise((n, n), 20, rng))
    for c in (0, n // 2):
        for d in range(-2, 3):
            img[(c + d) % n, :, :] = (0.78, 0.78, 0.78)
            img[:, (c + d) % n, :] = (0.78, 0.78, 0.78)
    return img


def _tex_sky(rng):
    h, w = 512, 1024
    t = np.linspace(0, 1, h)[:, None, None]
    top, hor = np.array([0.42, 0.62, 0.88]), np.array([0.84, 0.9, 0.97])
    img = top * (1 - t) + hor * t + np.zeros((h, w, 3))
    cl = _noise((h, w), (18, 60), rng) + 0.5 * _noise((h, w), 6, rng)
    cloud = np.clip((cl - 0.6) / 1.2, 0, 1)[..., None] * (1 - t) ** 0.5
    img = img * (1 - 0.85 * cloud) + 0.98 * 0.85 * cloud
    # distant skyline in the lower third
    x = 0
    while x < w:
        bw = int(rng.integers(30, 110))
        bh = int(rng.integers(40, 170))
        shade = 0.6 + 0.12 * rng.random()
        col = np.array([shade * 0.92, shade * 0.97, shade * 1.04])
        img[h - bh:, x:x + bw] = col
        for wy in range(h - bh + 8, h - 6, 14):           # window rows
            img[wy:wy + 5, x + 4:x + bw - 4:9][:, :, :] = np.minimum(col + 0.12, 1)
        x += bw + int(rng.integers(0, 12))
    img[h - 30:] = img[h - 30:] * 0.85 + 0.15 * np.array([0.55, 0.62, 0.5])   # trees / haze band
    return img


def _tex_hazard(rng):
    n = 128
    y, x = np.mgrid[0:n, 0:n]
    stripe = ((x + y) % 64) < 32
    img = np.where(stripe[..., None], np.array([0.98, 0.78, 0.04]), np.array([0.07, 0.07, 0.07]))
    return img + 0.02 * _noise((n, n), 1.0, rng)[..., None]


def _tex_brushed(rng):
    n = 256
    return _rgb((0.74, 0.76, 0.79), 0.035 * _noise((n, n), (0.6, 90), rng) + 0.01 * _noise((n, n), 0.7, rng))


def _tex_pegboard(rng):
    n = 128                                             # 4 x 4 holes, 25 mm pitch (0.1 m tile)
    img = _rgb((0.86, 0.84, 0.8), 0.01 * _noise((n, n), 2, rng))
    y, x = np.mgrid[0:n, 0:n]
    hole = ((x % 32 - 16) ** 2 + (y % 32 - 16) ** 2) < 16
    img[hole] = (0.25, 0.24, 0.22)
    return img


def _cabinet_front(w_m, h_m, n_mod, px_per_m=440):
    """Steel cabinet row with a laminate worktop edge and a toe kick, drawn at true aspect."""
    W, H = int(w_m * px_per_m), int(h_m * px_per_m)
    rng = np.random.default_rng(7)
    img = _rgb((0.8, 0.81, 0.82), 0.008 * _noise((H, W), 2, rng))
    top = int(0.035 * px_per_m)
    kick = int(0.07 * px_per_m)
    img[:top] = (0.3, 0.31, 0.33)
    img[top:top + 2] = (0.55, 0.56, 0.57)
    img[H - kick:] = (0.16, 0.16, 0.17)
    mw = W / n_mod
    drawer = int(0.16 * px_per_m)
    for k in range(n_mod):
        x0, x1 = int(k * mw), int((k + 1) * mw)
        img[top:H - kick, max(x0 - 1, 0):x0 + 2] = (0.32, 0.33, 0.35)                 # module gap
        img[top + drawer:top + drawer + 3, x0:x1] = (0.36, 0.37, 0.39)               # drawer / door split
        xm = (x0 + x1) // 2
        img[top + drawer:H - kick, xm - 1:xm + 1] = (0.42, 0.43, 0.45)               # double door split
        hy = top + drawer // 2
        img[hy - 3:hy + 3, xm - int(0.06 * px_per_m):xm + int(0.06 * px_per_m)] = (0.2, 0.2, 0.21)   # drawer pull
        for hx in (xm - int(0.03 * px_per_m), xm + int(0.03 * px_per_m)):                           # door pulls
            img[top + drawer + int(0.05 * px_per_m):top + drawer + int(0.17 * px_per_m), hx - 3:hx + 3] = (0.2, 0.2, 0.21)
        # small white label holder on the drawer
        img[hy - 9:hy - 5, x0 + 12:x0 + 12 + int(0.05 * px_per_m)] = (0.93, 0.93, 0.9)
    return img


def _text_img(size, bg, lines, fg=(255, 255, 255), accent=None):
    """lines: [(text, font_px, (x, y))]. Returns float RGB array."""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", size, tuple(int(255 * c) for c in bg))
    dr = ImageDraw.Draw(im)
    if accent:
        dr.rectangle(accent[0], fill=tuple(int(255 * c) for c in accent[1]))
    for txt, px, xy, *col in lines:
        dr.text(xy, txt, font=_font(px), fill=col[0] if col else fg)
    return np.asarray(im, np.float64) / 255


def _tex_sign(rng):
    return _text_img((1024, 192), (0.13, 0.15, 0.18),
                     [("PRINTPICK CELL", 92, (70, 22)), ("AUTOMATED PRINT FARM  ·  10 PRINTERS  ·  1 ROBOT", 30, (74, 128),
                                                         (240, 150, 60))],
                     accent=((0, 0, 30, 192), (0.95, 0.45, 0.1)))


def _tex_printer_screen(rng):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (256, 80), (8, 12, 18))
    dr = ImageDraw.Draw(im)
    dr.text((10, 6), "PLA  220°/65°", font=_font(22), fill=(225, 232, 240))
    dr.rectangle((10, 42, 246, 58), outline=(70, 80, 90))
    dr.rectangle((12, 44, 170, 56), fill=(40, 200, 110))
    dr.text((180, 60), "72%", font=_font(16), fill=(180, 190, 200))
    return np.asarray(im, np.float64) / 255


def _tex_monitor(rng):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (512, 288), (14, 20, 32))
    dr = ImageDraw.Draw(im)
    dr.rectangle((0, 0, 512, 30), fill=(24, 34, 52))
    dr.text((12, 5), "PRINTPICK  FLEET", font=_font(18), fill=(235, 240, 245))
    dr.text((380, 7), "ROBOT: UNLOADING", font=_font(13), fill=(240, 150, 60))
    cols = [(46, 190, 110), (46, 190, 110), (240, 150, 60), (46, 190, 110), (80, 140, 230),
            (46, 190, 110), (240, 150, 60), (46, 190, 110), (46, 190, 110), (80, 140, 230)]
    for k in range(10):
        x0, y0 = 12 + (k % 5) * 99, 42 + (k // 5) * 112
        dr.rectangle((x0, y0, x0 + 90, y0 + 102), fill=(26, 36, 54))
        dr.text((x0 + 8, y0 + 6), f"P{k + 1:02d}", font=_font(16), fill=(220, 226, 235))
        dr.rectangle((x0 + 8, y0 + 34, x0 + 82, y0 + 40), fill=(50, 60, 80))
        p = int(rng.integers(10, 74))
        dr.rectangle((x0 + 8, y0 + 34, x0 + 8 + p, y0 + 40), fill=cols[k])
        for j in range(3):
            dr.rectangle((x0 + 8, y0 + 54 + 14 * j, x0 + 8 + int(rng.integers(20, 74)), y0 + 60 + 14 * j),
                         fill=(70, 82, 104))
    return np.asarray(im, np.float64) / 255


def _tex_laptop(rng):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (256, 160), (22, 24, 30))
    dr = ImageDraw.Draw(im)
    palette = [(120, 200, 140), (210, 210, 220), (130, 170, 240), (240, 180, 90)]
    for j in range(13):
        x = 8 + 10 * int(rng.integers(0, 4))
        dr.rectangle((x, 8 + 11 * j, x + int(rng.integers(30, 200)), 13 + 11 * j), fill=palette[int(rng.integers(0, 4))])
    return np.asarray(im, np.float64) / 255


def _tex_whiteboard(rng):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (512, 256), (242, 244, 244))
    dr = ImageDraw.Draw(im)
    for k, t in enumerate(("QUEUE", "PRINTING", "PACKED")):
        dr.text((20 + 166 * k, 10), t, font=_font(20), fill=(30, 60, 140))
        if k:
            dr.line((166 * k + 8, 8, 166 * k + 8, 246), fill=(60, 60, 70), width=2)
    notes = [(255, 236, 110), (255, 170, 190), (160, 220, 255), (190, 240, 160)]
    for k in range(11):
        c = k % 3
        x0 = 18 + 166 * c + int(rng.integers(0, 70))
        y0 = 44 + 50 * int(rng.integers(0, 4))
        dr.rectangle((x0, y0, x0 + 42, y0 + 38), fill=notes[int(rng.integers(0, 4))])
        dr.line((x0 + 5, y0 + 12, x0 + 34, y0 + 12), fill=(90, 90, 90), width=2)
    dr.line((350, 210, 470, 180), fill=(200, 40, 40), width=3)
    return np.asarray(im, np.float64) / 255


def _tex_lane(label, bg):
    return lambda rng: _text_img((256, 128), bg, [(label, 34, (16, 14), (20, 20, 20)), ("LANE", 22, (16, 76), (40, 40, 40))])


def _tex_warning(rng):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (256, 320), (250, 205, 20))
    dr = ImageDraw.Draw(im)
    dr.polygon([(128, 18), (238, 200), (18, 200)], outline=(15, 15, 15), width=12)
    dr.rectangle((120, 80, 136, 150), fill=(15, 15, 15))
    dr.rectangle((120, 162, 136, 178), fill=(15, 15, 15))
    dr.text((22, 214), "ROBOT", font=_font(42), fill=(15, 15, 15))
    dr.text((22, 262), "WORK AREA", font=_font(34), fill=(15, 15, 15))
    return np.asarray(im, np.float64) / 255


def _tex_fire(rng):
    return _text_img((256, 128), (0.8, 0.08, 0.08), [("FIRE", 44, (20, 10)), ("EXTINGUISHER", 28, (20, 70))])


def _tex_kraft(rng, tape=None, label=False):
    n = 256
    img = _rgb((0.72, 0.56, 0.38), 0.02 * _noise((n, n), 3, rng) + 0.012 * _noise((n, n), (0.6, 30), rng))
    if tape == "h":
        img[n // 2 - 26:n // 2 + 26] = img[n // 2 - 26:n // 2 + 26] * 0.6 + 0.4 * np.array([0.86, 0.74, 0.55])
    if label:
        img[70:190, 40:200] = (0.96, 0.96, 0.95)
        for k in range(26):                         # barcode
            if rng.random() < 0.6:
                img[150:182, 52 + 5 * k:55 + 5 * k] = 0.05
        for j in range(3):
            img[82 + 18 * j:90 + 18 * j, 52:52 + int(rng.integers(60, 140))] = 0.25
    return img


def _tex_pei(rng):
    """Textured PEI spring-steel sheet: bronze powder coat with fine glitter and a slightly darker border."""
    n = 512
    f = 0.035 * _noise((n, n), 0.5, rng) + 0.006 * _noise((n, n), 4, rng) + 0.004 * _noise((n, n), 60, rng)
    sp = rng.random((n, n))
    f += 0.12 * (sp > 0.997) - 0.04 * (sp < 0.002)
    img = _rgb((0.74, 0.57, 0.3), f)
    b = 14
    img[:b], img[-b:], img[:, :b], img[:, -b:] = (img[:b] * 0.82, img[-b:] * 0.82, img[:, :b] * 0.82,
                                                  img[:, -b:] * 0.82)
    return img


def _tex_reject(rng):
    return _text_img((512, 96), (0.78, 0.09, 0.1), [("REJECT  >  REPRINT", 46, (22, 18), (255, 255, 255))])


BIN_LABELS = (("o0", "ORDER 1"), ("o1", "ORDER 2"), ("o2", "ORDER 3"), ("o3", "ORDER 4"), ("o4", "ORDER 5"),
              ("o5", "ORDER 6"), ("reject", "REJECT"), ("waste", "WASTE"))


def _tex_binlabels(rng):
    """Atlas: one 256 x 64 tag per caddy slot, stacked vertically (row k = BIN_LABELS[k])."""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (256, 64 * len(BIN_LABELS)), (245, 245, 240))
    dr = ImageDraw.Draw(im)
    for k, (slot, txt) in enumerate(BIN_LABELS):
        y = 64 * k
        col = (200, 30, 40) if slot == "reject" else (70, 72, 78) if slot == "waste" else (24, 120, 80)
        dr.rectangle((0, y, 255, y + 63), fill=(245, 245, 240), outline=(30, 30, 30), width=3)
        dr.rectangle((0, y, 40, y + 63), fill=col)
        dr.text((54, y + 12), txt, font=_font(34), fill=(20, 20, 20))
    return np.asarray(im, np.float64) / 255


def _tex_carriage_plate(rng):
    return _text_img((512, 128), (0.12, 0.13, 0.15), [("PRINTPICK", 56, (24, 14), (240, 240, 240)),
                                                      ("RAIL CELL  ·  YAM 6-DOF", 26, (26, 84), (240, 150, 60))],
                     accent=((480, 0, 512, 128), (0.95, 0.45, 0.1)))


def _tex_sign_row(row, first, last):
    return lambda rng: _text_img(
        (1024, 192), (0.13, 0.15, 0.18),
        [("PRINTPICK CELL", 92, (70, 22)),
         (f"ROW {row}  ·  PRINTERS {first:02d}-{last:02d}  ·  {C.N_PRINTERS} PRINTERS  ·  1 ROBOT", 30, (74, 128),
          (240, 150, 60))],
        accent=((0, 0, 30, 192), (0.95, 0.45, 0.1)))


def _tex_monitor_fleet(rng):
    """Fleet dashboard for the two-row cell: one tile per printer, row A over row B."""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (512, 288), (14, 20, 32))
    dr = ImageDraw.Draw(im)
    dr.rectangle((0, 0, 512, 30), fill=(24, 34, 52))
    dr.text((12, 5), "PRINTPICK  FLEET", font=_font(18), fill=(235, 240, 245))
    dr.text((380, 7), "ROBOT: UNLOADING", font=_font(13), fill=(240, 150, 60))
    cols = [(46, 190, 110)] * 6 + [(240, 150, 60), (80, 140, 230)]
    n_col = C.N_PER_ROW // 2
    tw = (512 - 24) // n_col
    for k in range(C.N_PRINTERS):
        row, j = divmod(k, C.N_PER_ROW)
        x0 = 12 + (j % n_col) * tw
        y0 = 38 + (2 * row + j // n_col) * 56 + 8 * row
        dr.rectangle((x0, y0, x0 + tw - 5, y0 + 50), fill=(26, 36, 54))
        dr.text((x0 + 4, y0 + 3), f"P{k + 1:02d}", font=_font(12), fill=(220, 226, 235))
        dr.rectangle((x0 + 4, y0 + 24, x0 + tw - 9, y0 + 28), fill=(50, 60, 80))
        p = int(rng.integers(6, tw - 13))
        dr.rectangle((x0 + 4, y0 + 24, x0 + 4 + p, y0 + 28), fill=cols[int(rng.integers(0, len(cols)))])
        dr.rectangle((x0 + 4, y0 + 36, x0 + 4 + int(rng.integers(10, tw - 13)), y0 + 40), fill=(70, 82, 104))
    n = C.N_PER_ROW
    dr.text((14, 272), f"ROW A  P01-P{n:02d}   ·   ROW B  P{n + 1:02d}-P{2 * n:02d}", font=_font(11),
            fill=(150, 160, 175))
    return np.asarray(im, np.float64) / 255


def _tex_exit(rng):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (384, 128), (20, 140, 70))
    dr = ImageDraw.Draw(im)
    dr.text((150, 24), "EXIT", font=_font(64), fill=(245, 250, 245))
    dr.ellipse((40, 16, 64, 40), fill=(245, 250, 245))                          # running figure
    dr.line((52, 40, 46, 76), fill=(245, 250, 245), width=10)
    dr.line((46, 76, 30, 108), fill=(245, 250, 245), width=9)
    dr.line((46, 76, 70, 104), fill=(245, 250, 245), width=9)
    dr.line((50, 50, 76, 62), fill=(245, 250, 245), width=8)
    dr.line((50, 50, 28, 64), fill=(245, 250, 245), width=8)
    dr.polygon([(92, 50), (122, 64), (92, 78)], fill=(245, 250, 245))           # arrow
    dr.rectangle((80, 59, 96, 69), fill=(245, 250, 245))
    return np.asarray(im, np.float64) / 255


# two-row cell only, generated with their own seeds (TEXTURES' seeds depend on its sorted names: keep them stable)
TEXTURES_TWO_ROWS = {
    "sign_a2": _tex_sign_row("A", 1, C.N_PER_ROW), "sign_b2": _tex_sign_row("B", C.N_PER_ROW + 1, C.N_PRINTERS),
    "monitor_fleet": _tex_monitor_fleet, "exit": _tex_exit,
}


TEXTURES = {
    "floor": _tex_floor, "ceiling": _tex_ceiling, "sky": _tex_sky, "hazard": _tex_hazard, "brushed": _tex_brushed,
    "pegboard": _tex_pegboard, "sign": _tex_sign, "pscreen": _tex_printer_screen, "monitor": _tex_monitor,
    "laptop": _tex_laptop, "whiteboard": _tex_whiteboard, "warning": _tex_warning, "fire": _tex_fire,
    "lane0": _tex_lane("COURIER A", (0.95, 0.78, 0.2)), "lane1": _tex_lane("COURIER B", (0.35, 0.62, 0.92)),
    "lane2": _tex_lane("POST", (0.88, 0.32, 0.32)),
    "kraft": lambda rng: _tex_kraft(rng), "kraft_top": lambda rng: _tex_kraft(rng, tape="h"),
    "kraft_label": lambda rng: _tex_kraft(rng, label=True),
    "pei": _tex_pei, "reject": _tex_reject, "binlabels": _tex_binlabels, "carrplate": _tex_carriage_plate,
}


def _square(img, n=1024):
    from PIL import Image
    return np.asarray(Image.fromarray(_u8(img)).resize((n, n), Image.LANCZOS), np.float64) / 255


def _bench_faces(w_m, d_m, h_m):
    """Cube-map faces for the bench box (MuJoCo: down = -y face, front = +z, right/left = +-x)."""
    from PIL import Image
    n_mod = max(1, round(w_m / 0.47))
    front = _cabinet_front(w_m, h_m, n_mod)
    side = _cabinet_front(d_m, h_m, 1)
    rng = np.random.default_rng(3)
    top = _rgb((0.72, 0.73, 0.73), 0.01 * _noise((256, 256), 2, rng))
    rot = lambda a, k: np.asarray(Image.fromarray(_u8(_square(a))).rotate(k), np.float64) / 255
    return {"down": _square(front), "up": _square(front), "front": _square(top), "back": _square(top),
            "right": rot(side, 90), "left": rot(side, -90)}


def _write_png(path, img):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp.png")
    Image.fromarray(_u8(img)).save(tmp)
    os.replace(tmp, path)                   # atomic: several sim processes may build at once


def _tex_path(name):
    return TEX_DIR / f"{name}_v{TEX_VERSION}.png"


def ensure_textures(bench_dims=None):
    """Generate missing texture PNGs; returns {name: path}."""
    paths = {}
    for k, (name, fn) in enumerate(sorted(TEXTURES.items())):
        p = _tex_path(name)
        if not p.exists():
            _write_png(p, fn(np.random.default_rng(100 + k)))
        paths[name] = p
    if bench_dims:
        tag = "bench_" + "x".join(f"{v * 1000:.0f}" for v in bench_dims)
        faces = None
        for face in ("down", "up", "front", "back", "right", "left"):
            p = _tex_path(f"{tag}_{face}")
            if not p.exists():
                faces = faces or _bench_faces(*bench_dims)
                _write_png(p, faces[face])
            paths[f"bench_{face}"] = p
    if TWO_ROWS:
        for k, (name, fn) in enumerate(sorted(TEXTURES_TWO_ROWS.items())):
            p = _tex_path(name)
            if not p.exists():
                _write_png(p, fn(np.random.default_rng(500 + k)))
            paths[name] = p
        paths["sign"], paths["monitor"] = paths["sign_a2"], paths["monitor_fleet"]   # same texture names, new pictures
    return paths


# ---------------------------------------------------------------- MJCF helpers
def _quat_mat(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _frame(u):
    u = u / np.linalg.norm(u)
    a = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
    v = np.cross(u, a)
    v /= np.linalg.norm(v)
    return u, v, np.cross(u, v)


class _Mesh:
    """Triangle soup in chunks (vertices, normals, optional texcoords, faces), wound counter-clockwise outward."""

    def __init__(self):
        self.chunks = []

    def _add(self, v, n, f, t=None):
        self.chunks.append((np.asarray(v, float), np.asarray(n, float), np.asarray(f, int), t))

    def quad(self, ps, n, uvs=None):
        ps, n = np.asarray(ps, float), np.asarray(n, float)
        f = [(0, 1, 2), (0, 2, 3)] if np.dot(np.cross(ps[1] - ps[0], ps[2] - ps[0]), n) >= 0 else [(0, 2, 1), (0, 3, 2)]
        self._add(ps, np.tile(n, (4, 1)), f, None if uvs is None else np.asarray(uvs, float))

    def box(self, c, h, R):
        c, h = np.asarray(c, float), np.asarray(h, float)
        v, n, f = [], [], []
        for ax, (b, d), hand in ((0, (1, 2), 1), (1, (0, 2), -1), (2, (0, 1), 1)):
            for s in (-1, 1):
                fc, eb, ed = c + R[:, ax] * s * h[ax], R[:, b] * h[b], R[:, d] * h[d]
                k = len(v)
                v += [fc - eb - ed, fc + eb - ed, fc + eb + ed, fc - eb + ed]
                n += [R[:, ax] * s] * 4
                f += [(k, k + 1, k + 2), (k, k + 2, k + 3)] if s * hand > 0 else [(k, k + 2, k + 1), (k, k + 3, k + 2)]
        self._add(v, n, f)

    def tube(self, p0, p1, r, nseg, caps=("flat", "flat")):
        """Cylinder side + flat or hemispherical caps. (u, v, w) right-handed: side quads (k, k+1) are CCW outward."""
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        u, v, w = _frame(p1 - p0)
        th = np.linspace(0, 2 * np.pi, nseg, endpoint=False)
        d = np.cos(th)[:, None] * v + np.sin(th)[:, None] * w            # (nseg, 3) radial directions
        k = np.arange(nseg)
        k2 = (k + 1) % nseg
        verts, norms, faces = [], [], []
        base = 0

        def ring(center, rad, along, nrm):
            nonlocal base
            verts.append(center + along + rad * d)
            norms.append(nrm)
            base += nseg
            return np.arange(base - nseg, base)

        def strip(a, b, flip):           # a -> b along +u (flip=False) or -u (flip=True)
            t1 = np.stack([a[k], a[k2], b[k2]], 1)
            t2 = np.stack([a[k], b[k2], b[k]], 1)
            faces.extend([t1[:, [0, 2, 1]], t2[:, [0, 2, 1]]] if flip else [t1, t2])

        r0 = ring(p0, r, 0, d)
        r1 = ring(p1, r, 0, d)
        strip(r0, r1, False)
        for p, sgn, cap, edge in ((p0, -1, caps[0], r0), (p1, 1, caps[1], r1)):
            if cap == "flat":
                rim = ring(p, r, 0, np.tile(sgn * u, (nseg, 1)))
                verts.append(p[None]), norms.append(sgn * u[None])
                base += 1
                fan = np.stack([np.full(nseg, base - 1), rim[k], rim[k2]], 1)
                faces.append(fan if sgn > 0 else fan[:, [0, 2, 1]])
            else:
                prev = edge
                for j in (1, 2):
                    phi = j * math.pi / 6
                    nrm = sgn * u * math.sin(phi) + math.cos(phi) * d
                    cur = ring(p, r * math.cos(phi), sgn * u * r * math.sin(phi), nrm)
                    strip(prev, cur, sgn < 0)
                    prev = cur
                verts.append((p + sgn * u * r)[None]), norms.append(sgn * u[None])
                base += 1
                fan = np.stack([prev[k], prev[k2], np.full(nseg, base - 1)], 1)
                faces.append(fan if sgn > 0 else fan[:, [0, 2, 1]])
        self._add(np.concatenate(verts), np.concatenate(norms), np.concatenate(faces))

    def arrays(self):
        vs, ns, fs, ts, off = [], [], [], [], 0
        for v, n, f, t in self.chunks:
            vs.append(v), ns.append(n), fs.append(f + off)
            ts.append(t)
            off += len(v)
        tex = None if any(t is None for t in ts) else np.concatenate(ts)
        return np.concatenate(vs), np.concatenate(ns), np.concatenate(fs), tex


class _Sink:
    """Collects the dressing. Opaque untextured visual geoms are merged into one mesh per material (one draw call
    each in the main, reflection and shadow passes); planes, textured, transparent and collidable geoms stay
    primitives on the worldbody."""

    def __init__(self, world, asset, collide_cls):
        self.world, self.asset, self.collide_cls = world, asset, collide_cls
        self.mats = {m.get("name"): m for m in asset.findall("material")}
        self.meshes = {}
        self.n_prim = self.n_geom = self.n_mesh = 0
        self.section = ""               # label of the part of the room being built (PRIMS)
        PRIMS.clear()

    def _log(self, typ, pos, size, quat, fromto, xyaxes):
        if fromto is not None:
            a, b, r = np.array(fromto[:3]), np.array(fromto[3:]), size[0]
            lo, hi = np.minimum(a, b) - r, np.maximum(a, b) + r
        elif typ == "plane":
            u, v = np.array(xyaxes[:3]), np.array(xyaxes[3:])
            e = np.abs(u) * size[0] + np.abs(v) * size[1]
            lo, hi = np.array(pos) - e, np.array(pos) + e
        else:
            R = _quat_mat(quat) if quat is not None else np.eye(3)
            e = np.abs(R) @ np.array(size[:3] if typ == "box" else (size[0], size[0], size[0]))
            lo, hi = np.array(pos) - e, np.array(pos) + e
        PRIMS.append((lo, hi, self.section))

    def _batchable(self, mat, collide, group):
        m = self.mats.get(mat)
        if m is None or collide or group not in (2, 4, 5) or m.get("texture"):
            return False
        return float((m.get("rgba") or "1 1 1 1").split()[3]) >= 1.0

    def add(self, typ, pos=None, size=None, mat=None, rgba=None, quat=None, fromto=None, xyaxes=None, collide=False,
            group=2, tag=""):
        """tag: pieces with a tag go into their own named mesh geom (lab_batch_<mat>_g<group>_<tag>) so a renderer can
        address them, e.g. hide one printer's set dressing."""
        self.n_prim += 1
        self._log(typ, pos, size, quat, fromto, xyaxes)
        if typ != "plane" and rgba is None and self._batchable(mat, collide, group):
            ms = self.meshes.setdefault((mat, group, tag), _Mesh())
            if typ == "box":
                ms.box(pos, size, _quat_mat(quat) if quat is not None else np.eye(3))
            else:
                r = size[0]
                nseg = 8 if r < 0.006 else 12 if r < 0.03 else 20 if r < 0.08 else 24
                ms.tube(fromto[:3], fromto[3:], r, nseg, ("round", "round") if typ == "capsule" else ("flat", "flat"))
            return
        a = {"type": typ}
        if fromto is not None:
            a["fromto"] = _f(*fromto)
        else:
            a["pos"] = _f(*pos)
        if size is not None:
            a["size"] = _f(*size)
        if quat is not None:
            a["quat"] = _f(*quat)
        if xyaxes is not None:
            a["xyaxes"] = _f(*xyaxes)
        if mat:
            a["material"] = mat
        if rgba is not None:
            a["rgba"] = _f(*rgba)
        if collide:
            if self.collide_cls:
                a["class"] = self.collide_cls
            else:
                a["contype"], a["conaffinity"] = "1", "1"
        else:
            a["contype"], a["conaffinity"] = "0", "0"
        a["group"] = str(group)
        ET.SubElement(self.world, "geom", a)
        self.n_geom += 1

    def flush(self):
        for (mat, group, tag), ms in sorted(self.meshes.items()):
            v, n, f, t = ms.arrays()
            n = n / np.linalg.norm(n, axis=1, keepdims=True)
            name = f"lab_batch_{mat}" + (f"_g{group}" if group != 2 else "") + (f"_{tag}" if tag else "")
            a = {"name": name, "inertia": "shell", "vertex": " ".join(f"{x:.6g}" for x in v.ravel()),
                 "normal": " ".join(f"{x:.4g}" for x in n.ravel()), "face": " ".join(map(str, f.ravel()))}
            if t is not None:
                a["texcoord"] = " ".join(f"{x:.5g}" for x in t.ravel())
            ET.SubElement(self.asset, "mesh", a)
            ET.SubElement(self.world, "geom", {"name": name, "type": "mesh", "mesh": name, "material": mat,
                                               "contype": "0", "conaffinity": "0", "group": str(group)})
            self.n_geom += 1
            self.n_mesh += 1
        self.meshes = {}

    def box(self, lo, hi, mat=None, **kw):
        """Axis-aligned box from two corners."""
        c = [(a + b) / 2 for a, b in zip(lo, hi)]
        h = [abs(b - a) / 2 for a, b in zip(lo, hi)]
        self.add("box", c, h, mat, **kw)

    def cyl(self, p0, p1, r, mat=None, **kw):
        self.add("cylinder", fromto=(*p0, *p1), size=(r,), mat=mat, **kw)

    def cap(self, p0, p1, r, mat=None, **kw):
        self.add("capsule", fromto=(*p0, *p1), size=(r,), mat=mat, **kw)

    def poly(self, pts, r, mat=None):
        for a, b in zip(pts[:-1], pts[1:]):
            self.cap(a, b, r, mat)

    def panel(self, corners, normal, mat, tile=None, uvs=None, group=2):
        """One-sided quad (meshes are back-face culled: invisible from behind, unlike plane geoms, whose back side
        renders translucent). tile: texture tile size in m for textured materials; uvs: explicit corner texcoords."""
        self.n_prim += 1
        c = np.asarray(corners, float)
        PRIMS.append((c.min(0), c.max(0), self.section))
        if tile:
            c = np.asarray(corners, float)
            e1, e2 = c[1] - c[0], c[3] - c[0]
            l1, l2 = np.linalg.norm(e1), np.linalg.norm(e2)
            uvs = [(0, 0), (l1 / tile, 0), (l1 / tile, l2 / tile), (0, l2 / tile)]
        self.meshes.setdefault((mat, group, ""), _Mesh()).quad(corners, normal, uvs)

    def plane(self, center, half_wh, xaxis, yaxis, mat):
        self.add("plane", center, (*half_wh, max(half_wh)), mat, xyaxes=(*xaxis, *yaxis))


def _mat_quat(R):
    """Rotation matrix -> unit quaternion (w, x, y, z)."""
    R = np.asarray(R, float)
    t = np.trace(R)
    if t > 0:
        k = 2 * math.sqrt(1 + t)
        q = (k / 4, (R[2, 1] - R[1, 2]) / k, (R[0, 2] - R[2, 0]) / k, (R[1, 0] - R[0, 1]) / k)
    else:
        i = int(np.argmax(np.diag(R)))
        j, m = (i + 1) % 3, (i + 2) % 3
        k = 2 * math.sqrt(1 + R[i, i] - R[j, j] - R[m, m])
        v = [0.0, 0.0, 0.0]
        v[i], v[j], v[m] = k / 4, (R[j, i] + R[i, j]) / k, (R[m, i] + R[i, m]) / k
        q = ((R[m, j] - R[j, m]) / k, *v)
    return tuple(float(x) for x in q)


def _rot_z(quarter_turns):
    c, s = ((1, 0), (0, 1), (-1, 0), (0, -1))[quarter_turns % 4]
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


MIRROR_Y = np.diag([1.0, -1.0, 1.0])


class _Xf:
    """A _Sink seen through the map p -> R p + t, R a signed permutation (quarter turns about z, or the y mirror that
    takes row A's wall to row B's). Lets the furniture/services code written for one wall place its pieces against
    another. Under a mirror, boxes stay boxes (they are symmetric) and textured planes/panels are flipped back so
    text still reads correctly."""

    def __init__(self, s, R=None, t=(0.0, 0.0, 0.0), suffix=""):
        self.s, self.R, self.t = s, np.eye(3) if R is None else np.asarray(R, float), np.asarray(t, float)
        self.mirror = np.linalg.det(self.R) < 0
        self.suffix = suffix            # material variant, e.g. WALL_B ("_wb": row B's wall items, see cutaway())

    def M(self, mat):
        return mat if not (mat and self.suffix) or mat.endswith(self.suffix) else mat + self.suffix

    def P(self, p):
        return tuple(float(v) for v in self.R @ np.asarray(p, float) + self.t)

    def V(self, v):
        return tuple(float(x) for x in self.R @ np.asarray(v, float))

    def _xaxis(self, a):
        a = self.V(a)
        return tuple(-v for v in a) if self.mirror else a

    def add(self, typ, pos=None, size=None, mat=None, rgba=None, quat=None, fromto=None, xyaxes=None, **kw):
        if fromto is not None:
            fromto = (*self.P(fromto[:3]), *self.P(fromto[3:]))
        if pos is not None:
            pos = self.P(pos)
        if xyaxes is not None:
            xyaxes = (*self._xaxis(xyaxes[:3]), *self.V(xyaxes[3:]))
        elif typ == "box":
            Rb = self.R @ (_quat_mat(quat) if quat is not None else np.eye(3))
            if self.mirror:
                Rb = Rb @ MIRROR_Y                  # a box is symmetric: flip one local axis back to a rotation
            quat = None if np.allclose(Rb, np.eye(3)) else _mat_quat(Rb)
        self.s.add(typ, pos, size, self.M(mat), rgba=rgba, quat=quat, fromto=fromto, xyaxes=xyaxes, **kw)

    def box(self, lo, hi, mat=None, **kw):
        a, b = np.array(self.P(lo)), np.array(self.P(hi))
        self.s.box(tuple(np.minimum(a, b)), tuple(np.maximum(a, b)), self.M(mat), **kw)

    def cyl(self, p0, p1, r, mat=None, **kw):
        self.s.cyl(self.P(p0), self.P(p1), r, self.M(mat), **kw)

    def cap(self, p0, p1, r, mat=None, **kw):
        self.s.cap(self.P(p0), self.P(p1), r, self.M(mat), **kw)

    def poly(self, pts, r, mat=None):
        self.s.poly([self.P(p) for p in pts], r, self.M(mat))

    def panel(self, corners, normal, mat, tile=None, uvs=None, group=2):
        if self.mirror and uvs is not None:
            us = [u for u, _ in uvs]
            uvs = [(max(us) + min(us) - u, v) for u, v in uvs]
        self.s.panel([self.P(c) for c in corners], self.V(normal), self.M(mat), tile=tile, uvs=uvs, group=group)

    def plane(self, center, half_wh, xaxis, yaxis, mat):
        self.s.plane(self.P(center), half_wh, self._xaxis(xaxis), self.V(yaxis), self.M(mat))


def _vec(s):
    return [float(v) for v in s.split()]


def _find_geoms(world, pred):
    return [g for g in world.findall("geom") if pred(g)]


# ---------------------------------------------------------------- assets
def _assets(asset, paths):
    def tex(name, file, **kw):
        ET.SubElement(asset, "texture", {"name": name, "type": "2d", "file": str(file), **kw})

    def mat(name, **kw):
        ET.SubElement(asset, "material", {"name": name, **{k: (_f(*v) if isinstance(v, tuple) else str(v))
                                                           for k, v in kw.items()}})

    for n in ("floor", "ceiling", "sky", "hazard", "brushed", "pegboard", "sign", "pscreen", "monitor", "laptop",
              "whiteboard", "warning", "fire", "lane0", "lane1", "lane2", "kraft", "kraft_label", "pei", "reject",
              "binlabels", "carrplate"):
        tex(f"lab_{n}", paths[n])
    if TWO_ROWS:
        tex("lab_sign_b", paths["sign_b2"])
        tex("lab_exit", paths["exit"])
    if "bench_down" in paths:
        ET.SubElement(asset, "texture", {"name": "lab_bench", "type": "cube",
                                         **{f"file{f}": str(paths[f"bench_{f}"]) for f in
                                            ("down", "up", "front", "back", "right", "left")}})
    ET.SubElement(asset, "texture", {"name": "lab_box", "type": "cube", "fileup": str(paths["kraft"]),
                                     "filedown": str(paths["kraft_label"]), "filefront": str(paths["kraft_top"]),
                                     "fileback": str(paths["kraft"]), "fileleft": str(paths["kraft"]),
                                     "fileright": str(paths["kraft"])})
    # room
    # walls/ceiling carry a little emission: stand-in for bounce light (the renderer has no GI)
    mat("lab_wall", rgba=(0.86, 0.855, 0.84, 1), emission=0.22, specular=0.08, shininess=0.1)
    mat("lab_wall_front", rgba=(0.86, 0.855, 0.84, 1), emission=0.5, specular=0.08, shininess=0.1)   # no light hits it
    mat("lab_ceiling", texture="lab_ceiling", emission=0.45, specular=0.02)
    mat("lab_outside", rgba=(0.58, 0.59, 0.61, 1), specular=0.0)
    mat("lab_skirt", rgba=(0.22, 0.23, 0.24, 1), specular=0.2)
    mat("lab_lamp", rgba=(1, 1, 0.97, 1), emission=1.0, specular=0)
    mat("lab_sky", texture="lab_sky", emission=0.95, specular=0)
    mat("lab_frame", rgba=(0.86, 0.87, 0.88, 1), specular=0.4, shininess=0.4)
    mat("lab_hazard", texture="lab_hazard", texuniform="true", texrepeat=(8, 8), specular=0.2)
    mat("lab_tape_green", rgba=(0.2, 0.62, 0.32, 1), specular=0.2)
    mat("lab_sign", texture="lab_sign", emission=0.15, specular=0.1)
    mat("lab_galv", rgba=(0.66, 0.68, 0.7, 1), specular=0.5, shininess=0.5)
    mat("lab_cable", rgba=(0.06, 0.06, 0.07, 1), specular=0.3, shininess=0.6)
    mat("lab_cable_blue", rgba=(0.1, 0.3, 0.75, 1), specular=0.3, shininess=0.6)
    mat("lab_elec", rgba=(0.78, 0.79, 0.8, 1), specular=0.3)
    mat("lab_clockface", rgba=(0.97, 0.97, 0.96, 1), specular=0.1)
    # printers
    mat("lab_ams", rgba=(0.82, 0.83, 0.84, 1), specular=0.25, shininess=0.4)
    mat("lab_amslid", rgba=(0.2, 0.21, 0.24, 0.38), specular=0.9, shininess=0.95)
    mat("lab_rubber", rgba=(0.1, 0.1, 0.11, 1), specular=0.1)
    mat("lab_flange", rgba=(0.12, 0.12, 0.13, 1), specular=0.6, shininess=0.8)
    mat("lab_ptfe", rgba=(0.93, 0.94, 0.94, 1), specular=0.6, shininess=0.8)
    mat("lab_bezel", rgba=(0.03, 0.03, 0.04, 1), specular=0.9, shininess=0.95)
    mat("lab_pscreen", texture="lab_pscreen", emission=0.8, specular=0.6, shininess=0.9)
    mat("lab_logo", rgba=(0.78, 0.79, 0.8, 1), specular=0.6)
    mat("lab_led", rgba=(1, 0.98, 0.92, 1), emission=1.0)
    mat("lab_led_green", rgba=(0.2, 1, 0.45, 1), emission=1.0)
    mat("lab_led_white", rgba=(0.9, 0.95, 1, 1), emission=0.9)
    for name, rgb in SPOOL_COLORS.items():
        mat(f"lab_fil_{name}", rgba=(*rgb, 1), specular=0.55, shininess=0.7)
    # rail, station, furniture
    mat("lab_brushed", texture="lab_brushed", texuniform="true", texrepeat=(1.5, 1.5), specular=0.7, shininess=0.55)
    mat("lab_steel", rgba=(0.2, 0.21, 0.23, 1), specular=0.6, shininess=0.6)
    mat("lab_endstop", rgba=(0.98, 0.78, 0.05, 1), specular=0.3)
    mat("lab_roller", rgba=(0.72, 0.74, 0.77, 1), specular=0.8, shininess=0.7)
    mat("lab_alu", rgba=(0.75, 0.77, 0.8, 1), specular=0.7, shininess=0.6)
    mat("lab_cardboard", texture="lab_box", rgba=(1, 1, 1, 1), specular=0.05)
    mat("lab_kraft", rgba=(0.72, 0.56, 0.38, 1), specular=0.05)
    mat("lab_kraft_tape", rgba=(0.84, 0.72, 0.54, 1), specular=0.5, shininess=0.6)
    mat("lab_pallet", rgba=(0.66, 0.53, 0.36, 1), specular=0.05)
    mat("lab_worktop", rgba=(0.82, 0.8, 0.76, 1), specular=0.25, shininess=0.3)
    mat("lab_frame_dark", rgba=(0.26, 0.27, 0.29, 1), specular=0.4)
    mat("lab_shelfsteel", rgba=(0.55, 0.58, 0.62, 1), specular=0.4)
    mat("lab_plastic_white", rgba=(0.92, 0.92, 0.9, 1), specular=0.3)
    mat("lab_plastic_dark", rgba=(0.15, 0.16, 0.17, 1), specular=0.4)
    mat("lab_dryclear", rgba=(0.85, 0.9, 0.95, 0.35), specular=0.9, shininess=0.9)
    mat("lab_red", rgba=(0.8, 0.07, 0.06, 1), specular=0.7, shininess=0.7)
    mat("lab_yellow", rgba=(0.98, 0.78, 0.05, 1), specular=0.3)
    mat("lab_blue_bin", rgba=(0.1, 0.36, 0.72, 1), specular=0.3)
    mat("lab_grey_bin", rgba=(0.32, 0.33, 0.35, 1), specular=0.3)
    mat("lab_mat_green", rgba=(0.12, 0.38, 0.26, 1), specular=0.05)
    mat("lab_monitor", texture="lab_monitor", emission=0.85, specular=0.5)
    mat("lab_laptop", texture="lab_laptop", emission=0.8, specular=0.5)
    mat("lab_whiteboard", texture="lab_whiteboard", specular=0.5, shininess=0.6)
    mat("lab_pegboard", texture="lab_pegboard", texuniform="true", texrepeat=(10, 10), specular=0.05)
    mat("lab_warning", texture="lab_warning", emission=0.1, specular=0.2)
    mat("lab_fire", texture="lab_fire", emission=0.1, specular=0.2)
    for k in range(3):
        mat(f"lab_lanesign{k}", texture=f"lab_lane{k}", emission=0.1, specular=0.2)
    mat("lab_tower_g", rgba=(0.15, 0.95, 0.3, 1), emission=0.9)
    mat("lab_tower_a", rgba=(0.8, 0.5, 0.1, 1), emission=0.05, specular=0.6)
    mat("lab_tower_r", rgba=(0.6, 0.08, 0.08, 1), emission=0.05, specular=0.6)
    mat("lab_bubble", rgba=(0.85, 0.92, 0.95, 0.6), specular=0.8, shininess=0.9)
    # printer interior / status / labels / carriage
    mat("lab_reject", texture="lab_reject", emission=0.15, specular=0.3)
    mat("lab_binlabel", texture="lab_binlabels", emission=0.1, specular=0.2)
    mat("lab_carrplate", texture="lab_carrplate", emission=0.1, specular=0.4)
    mat("lab_toolcover", rgba=(0.86, 0.87, 0.88, 1), specular=0.5, shininess=0.6)
    mat("lab_brass", rgba=(0.85, 0.65, 0.3, 1), specular=0.9, shininess=0.9)
    mat("lab_chrome", rgba=(0.8, 0.82, 0.85, 1), specular=1, shininess=1, reflectance=0.2)
    mat("lab_carbon", rgba=(0.05, 0.05, 0.06, 1), specular=0.7, shininess=0.8)
    mat("lab_cablechain", rgba=(0.09, 0.09, 0.1, 1), specular=0.25)
    mat("lab_anodized", rgba=(0.2, 0.21, 0.23, 1), specular=0.55, shininess=0.6)
    for i in range(C.N_PRINTERS):        # per-printer status bar: a renderer may recolour it (m.mat_rgba) per frame
        mat(f"lab_status{i}", rgba=STATUS_RGBA["printing"], emission=1.0, specular=0)
    if TWO_ROWS:                         # row B's wall: own material variants (cutaway() hides them), sign, door, exit
        mats = {m.get("name"): m for m in asset.findall("material")}
        for base in WALL_B_MATS:
            ET.SubElement(asset, "material", {**mats[base].attrib, "name": base + WALL_B})
        mat("lab_sign" + WALL_B, texture="lab_sign_b", emission=0.15, specular=0.1)
        mat("lab_exit" + WALL_B, texture="lab_exit", emission=0.9, specular=0.1)
        mat("lab_door" + WALL_B, rgba=(0.55, 0.6, 0.66, 1), emission=0.18, specular=0.35, shininess=0.4)


def _retune(asset):
    """Existing materials: same names, better-reading surfaces."""
    tune = {
        "floor": dict(texture="lab_floor", texuniform="true", texrepeat="0.25 0.25", reflectance="0.045",
                      specular="0.3", shininess="0.45", rgba="0.8 0.81 0.82 1"),     # sealed concrete, not a mirror
        "shell": dict(rgba="0.14 0.145 0.155 1", specular="0.3", shininess="0.35"),
        "trim": dict(rgba="0.26 0.27 0.29 1", specular="0.45", shininess="0.5"),
        "toolhead": dict(rgba="0.2 0.2 0.22 1", specular="0.5", shininess="0.6"),
        "roofglass": dict(rgba="0.4 0.45 0.5 0.3", specular="1", shininess="1"),
        "pei": dict(texture="lab_pei", texuniform="true", texrepeat="4 4", rgba="1 1 1 1", specular="0.45",
                    shininess="0.5", emission="0.18"),          # emission: stand-in for the chamber LED bar
        "robotorange": dict(rgba="0.98 0.45 0.08 1", specular="0.5", shininess="0.6"),
        "carriage": dict(texture="lab_brushed", texuniform="true", texrepeat="2 2", rgba="0.86 0.87 0.9 1",
                         specular="0.6", shininess="0.5"),       # aluminium tooling plate
        "rail": dict(texture="lab_brushed", texuniform="true", texrepeat="1.5 1.5", rgba="1 1 1 1", specular="0.75",
                     shininess="0.55"),
        "bench": dict(texture="lab_bench", rgba="1 1 1 1", specular="0.3", shininess="0.35"),
        "bin": dict(rgba="0.13 0.5 0.36 1", specular="0.35", shininess="0.4"),
        "binreject": dict(rgba="0.82 0.12 0.14 1", specular="0.45", shininess="0.5"),
        "binwaste": dict(rgba="0.3 0.31 0.33 1", specular="0.35", shininess="0.4"),
        "glass": dict(rgba="0.62 0.7 0.76 0.16", specular="1", shininess="1"),
        "belt": dict(rgba="0.09 0.09 0.1 1", specular="0.15", shininess="0.2"),
        "cardboard": dict(texture="lab_box", rgba="1 1 1 1", specular="0.05"),
        "pla": dict(specular="0.5", shininess="0.6"),
    }
    have_bench = any(t.get("name") == "lab_bench" for t in asset.findall("texture"))
    for m in asset.findall("material"):
        kw = tune.get(m.get("name"))
        if kw and not (m.get("name") == "bench" and not have_bench):
            m.attrib.update(kw)


# ---------------------------------------------------------------- lighting
def _lighting(root, world):
    vis = root.find("visual")
    if vis is None:
        vis = ET.Element("visual")
        root.insert(0, vis)
    hl = vis.find("headlight")
    if hl is None:
        hl = ET.SubElement(vis, "headlight")
    hl.attrib.update(diffuse="0.18 0.18 0.18", ambient="0.22 0.22 0.23", specular="0.05 0.05 0.05")
    rgba = vis.find("rgba")
    if rgba is None:
        rgba = ET.SubElement(vis, "rgba")
    rgba.set("haze", "0.86 0.88 0.9 1")
    mp = vis.find("map")
    if mp is not None:
        mp.set("shadowclip", "0.6")
    # Renderer quirk (MuJoCo 3.16): every surface behind a spot light's plane renders black. So the existing fill
    # spot becomes directional, and the added spots hang above the ceiling pointing straight down.
    # The shadow light sits just below the lamps/ceiling: geometry above a directional light's pos casts no shadow
    # (receivers above it get garbage shadow lookups, so keep it above every wall item).
    xc = (WALL_LEFT_X + WALL_RIGHT_X) / 2
    for li in world.findall("light"):
        if li.get("castshadow", "true") == "true":
            # key light, nearly vertical. Tilting it more than ~8 deg makes the walls' shadow sweep a hard wedge across the
            # floor in wide views (tested 10-22: dir 0.1 0.18 and 0.3 0.4 both show it at shadowsize 4096)
            if TWO_ROWS:                # centred on the long room, no tilt across the aisle: both rows read the same
                li.attrib.update(pos=_f(xc, 0.0, CEIL_Z - 0.06), dir="0.06 0.0 -1", diffuse="0.55 0.545 0.53",
                                 specular="0.15 0.15 0.15", directional="true")
            else:
                li.attrib.update(pos=_f(2.3, -0.4, CEIL_Z - 0.06), dir="0.05 0.14 -1", diffuse="0.55 0.545 0.53",
                                 specular="0.15 0.15 0.15", directional="true")
        else:
            li.attrib.update(directional="true", dir="-0.3 0.55 -1", diffuse="0.16 0.17 0.19", specular="0 0 0")
    n_all = len(list(root.iter("light"))) + 1          # + headlight
    spots = [  # x, y, diffuse: pools of light under the lamp rows
        (0.9, -0.5, (0.22, 0.218, 0.21)), (3.3, -0.5, (0.22, 0.218, 0.21)),
        (6.3, -0.3, (0.22, 0.22, 0.21)), (-2.3, -0.4, (0.21, 0.21, 0.2)),
    ]
    if TWO_ROWS:                        # pools evenly down the aisle (5 spots + 2 lights + headlight = 8)
        dx = (WALL_RIGHT_X - WALL_LEFT_X) / 5
        spots = [(WALL_LEFT_X + dx * (k + 0.5), 0.0, (0.2, 0.198, 0.19)) for k in range(5)]
    for x, y, dif in spots[:max(0, MAX_LIGHTS - n_all)]:
        ET.SubElement(world, "light", {"pos": _f(x, y, CEIL_Z + 0.4), "dir": "0 0 -1", "diffuse": _f(*dif),
                                       "cutoff": "80", "exponent": "5", "specular": "0.05 0.05 0.05",
                                       "castshadow": "false"})


# ---------------------------------------------------------------- room
def _room(s, xmin, xmax):
    xl, xr, yf, yb, zc = WALL_LEFT_X, WALL_RIGHT_X, WALL_FRONT_Y, WALL_BACK_Y, CEIL_Z
    s.panel([(xl, yb, 0), (xr, yb, 0), (xr, yb, zc), (xl, yb, zc)], (0, -1, 0), "lab_wall")
    s.panel([(xl, yf, 0), (xl, yb, 0), (xl, yb, zc), (xl, yf, zc)], (1, 0, 0), "lab_wall")
    s.panel([(xr, yb, 0), (xr, yf, 0), (xr, yf, zc), (xr, yb, zc)], (-1, 0, 0), "lab_wall")
    s.panel([(xr, yf, 0), (xl, yf, 0), (xl, yf, zc), (xr, yf, zc)], (0, 1, 0), "lab_wall_front")
    s.panel([(xl, yb, zc), (xr, yb, zc), (xr, yf, zc), (xl, yf, zc)], (0, 0, -1), "lab_ceiling", tile=1.2)
    # skirting
    t, h = 0.015, 0.1
    s.box((WALL_LEFT_X, WALL_BACK_Y - t, 0), (WALL_RIGHT_X, WALL_BACK_Y, h), "lab_skirt")
    s.box((WALL_LEFT_X, WALL_FRONT_Y, 0), (WALL_LEFT_X + t, WALL_BACK_Y, h), "lab_skirt")
    s.box((WALL_RIGHT_X - t, WALL_FRONT_Y, 0), (WALL_RIGHT_X, WALL_BACK_Y, h), "lab_skirt")
    # recessed LED panels (1.2 x 0.6) in the tile grid, one-sided like the ceiling
    for x in (-2.1, 0.3, 2.7, 5.1, 7.5):
        for y in (0.15, -1.65, -3.3):
            z = CEIL_Z - 0.003
            s.panel([(x - 0.6, y - 0.3, z), (x + 0.6, y - 0.3, z), (x + 0.6, y + 0.3, z), (x - 0.6, y + 0.3, z)],
                    (0, 0, -1), "lab_lamp")
    # plain ground outside the room (only seen from outside cutaway views)
    big = 40.0
    for (x0, x1, y0, y1) in ((WALL_LEFT_X - big, WALL_LEFT_X, WALL_FRONT_Y - big, WALL_BACK_Y + big),
                             (WALL_RIGHT_X, WALL_RIGHT_X + big, WALL_FRONT_Y - big, WALL_BACK_Y + big),
                             (WALL_LEFT_X, WALL_RIGHT_X, WALL_BACK_Y, WALL_BACK_Y + big),
                             (WALL_LEFT_X, WALL_RIGHT_X, WALL_FRONT_Y - big, WALL_FRONT_Y)):
        s.plane(((x0 + x1) / 2, (y0 + y1) / 2, 0.003), ((x1 - x0) / 2, (y1 - y0) / 2), (1, 0, 0), (0, 1, 0),
                "lab_outside")
    # windows: back wall left of the row and behind the station, one in the right wall at the end of the aisle
    _window_back(s, xmin - 1.65, xmin - 0.75, 1.1, 2.25)
    _window_back(s, 5.75, 7.15, 1.2, 2.3)
    _window_right(s, -2.3, -0.75, 1.1, 2.3)
    # sign above the row
    xm = (xmin + xmax) / 2
    yb = WALL_BACK_Y
    s.box((xm - 0.98, yb - 0.025, 2.33), (xm + 0.98, yb, 2.71), "lab_skirt")
    s.plane((xm, yb - 0.026, 2.52), (0.95, 0.178), (1, 0, 0), (0, 0, 1), "lab_sign")
    # wall clock
    cxk = xmax + 0.75
    s.cyl((cxk, yb, 2.45), (cxk, yb - 0.035, 2.45), 0.165, "lab_skirt")
    s.cyl((cxk, yb - 0.035, 2.45), (cxk, yb - 0.04, 2.45), 0.15, "lab_clockface")
    s.box((cxk - 0.004, yb - 0.046, 2.45), (cxk + 0.004, yb - 0.042, 2.57), "lab_steel")
    s.add("box", (cxk + 0.04, yb - 0.048, 2.47), (0.06, 0.002, 0.005), "lab_steel", quat=(0.97, 0, 0.26, 0))


def _room_two(s, xmin, xmax):
    """Two facing rows: a wall behind each row (the same 14 cm cable gap), end walls closing the operator bay at -x and
    the packing end at +x; windows, signs and lamps arranged symmetrically about the rail."""
    xl, xr, yf, yb, zc = WALL_LEFT_X, WALL_RIGHT_X, WALL_FRONT_Y, WALL_BACK_Y, CEIL_Z
    s.panel([(xl, yb, 0), (xr, yb, 0), (xr, yb, zc), (xl, yb, zc)], (0, -1, 0), "lab_wall")
    s.panel([(xr, yf, 0), (xl, yf, 0), (xl, yf, zc), (xr, yf, zc)], (0, 1, 0), "lab_wall")
    sb = _Xf(s, suffix=WALL_B)          # items mounted on row B's wall
    s.panel([(xl, yf, 0), (xl, yb, 0), (xl, yb, zc), (xl, yf, zc)], (1, 0, 0), "lab_wall")
    s.panel([(xr, yb, 0), (xr, yf, 0), (xr, yf, zc), (xr, yb, zc)], (-1, 0, 0), "lab_wall")
    # ceiling: tile grid aligned with the lamp rows (lines at y = +-0.3, +-0.9 and every 0.6 m in x from the lamps)
    n_lamp = int((xr - xl - 1.8) // 2.4) + 1                        # 2.4 m apart, >= 0.3 m from the end walls
    lamp_x = [(xl + xr) / 2 + 2.4 * (k - (n_lamp - 1) / 2) for k in range(n_lamp)]
    ux, vy = lamp_x[0] - 0.6, 0.3
    s.panel([(xl, yb, zc), (xr, yb, zc), (xr, yf, zc), (xl, yf, zc)], (0, 0, -1), "lab_ceiling",
            uvs=[((x - ux) / 1.2, (vy - y) / 1.2) for x, y in ((xl, yb), (xr, yb), (xr, yf), (xl, yf))])
    t, h = 0.015, 0.1                   # skirting
    s.box((xl, yb - t, 0), (xr, yb, h), "lab_skirt")
    sb.box((xl, yf, 0), (xr, yf + t, h), "lab_skirt")
    s.box((xl, yf, 0), (xl + t, yb, h), "lab_skirt")
    s.box((xr - t, yf, 0), (xr, yb, h), "lab_skirt")
    # recessed LED panels (1.2 x 0.6): a line above each row's door plane, the aisle between them
    for x in lamp_x:
        for y in (0.6, -0.6):
            z = CEIL_Z - 0.003
            s.panel([(x - 0.6, y - 0.3, z), (x + 0.6, y - 0.3, z), (x + 0.6, y + 0.3, z), (x - 0.6, y + 0.3, z)],
                    (0, 0, -1), "lab_lamp")
    big = 40.0                          # plain ground outside the room (cutaway views)
    for (x0, x1, y0, y1) in ((xl - big, xl, yf - big, yb + big), (xr, xr + big, yf - big, yb + big),
                             (xl, xr, yb, yb + big), (xl, xr, yf - big, yf)):
        s.plane(((x0 + x1) / 2, (y0 + y1) / 2, 0.003), ((x1 - x0) / 2, (y1 - y0) / 2), (1, 0, 0), (0, 1, 0),
                "lab_outside")
    # windows: row A's wall above the operator bay's bins, both long walls behind the station, the far end wall
    _window_back(s, xmin - 1.65, xmin - 0.75, 1.1, 2.25)
    ws = C.STATION_X + 0.86
    _window_back(s, ws, ws + 1.4, 1.2, 2.3)
    _window_front(sb, ws, ws + 1.4, 1.2, 2.3)
    _window_right(s, -0.7, 0.7, 1.15, 2.3)
    # a sign above each row
    xm = (xmin + xmax) / 2
    s.box((xm - 0.98, yb - 0.025, 2.33), (xm + 0.98, yb, 2.71), "lab_skirt")
    s.plane((xm, yb - 0.026, 2.52), (0.95, 0.178), (1, 0, 0), (0, 0, 1), "lab_sign")
    sb.box((xm - 0.98, yf, 2.33), (xm + 0.98, yf + 0.025, 2.71), "lab_skirt")
    sb.plane((xm, yf + 0.026, 2.52), (0.95, 0.178), (-1, 0, 0), (0, 0, 1), "lab_sign")
    # wall clock on row A's wall past the end of the cable tray
    cxk = xmax + 0.75
    s.cyl((cxk, yb, 2.45), (cxk, yb - 0.035, 2.45), 0.165, "lab_skirt")
    s.cyl((cxk, yb - 0.035, 2.45), (cxk, yb - 0.04, 2.45), 0.15, "lab_clockface")
    s.box((cxk - 0.004, yb - 0.046, 2.45), (cxk + 0.004, yb - 0.042, 2.57), "lab_steel")
    s.add("box", (cxk + 0.04, yb - 0.048, 2.47), (0.06, 0.002, 0.005), "lab_steel", quat=(0.97, 0, 0.26, 0))
    _door(sb, DOOR_X)


DOOR_X = (-2.5, -1.6)                 # two-row cell: personnel door in row B's wall, opening into the operator bay


def _door(s, xs, h=2.1):
    """Closed steel door in row B's wall (normal +y): frame, leaf with vision panel, push bar, kick plate, exit sign."""
    (x0, x1), yf = xs, WALL_FRONT_Y
    xm = (x0 + x1) / 2
    s.box((x0 - 0.06, yf, 0), (x0, yf + 0.035, h + 0.06), "lab_frame")
    s.box((x1, yf, 0), (x1 + 0.06, yf + 0.035, h + 0.06), "lab_frame")
    s.box((x0, yf, h), (x1, yf + 0.035, h + 0.06), "lab_frame")
    s.box((x0 + 0.004, yf, 0.008), (x1 - 0.004, yf + 0.022, h - 0.004), "lab_door")
    s.box((xm - 0.12, yf + 0.022, 1.32), (xm + 0.12, yf + 0.026, 1.86), "lab_frame")
    s.box((xm - 0.1, yf + 0.026, 1.34), (xm + 0.1, yf + 0.028, 1.84), "lab_bezel")      # wired-glass vision panel
    s.box((x0 + 0.02, yf + 0.022, 0.02), (x1 - 0.02, yf + 0.025, 0.25), "lab_steel")  # kick plate
    for xb in (x0 + 0.1, x1 - 0.1):
        s.box((xb - 0.02, yf + 0.022, 0.97), (xb + 0.02, yf + 0.07, 1.03), "lab_steel")
    s.cyl((x0 + 0.08, yf + 0.065, 1.0), (x1 - 0.08, yf + 0.065, 1.0), 0.016, "lab_alu")  # push bar
    s.box((xm - 0.17, yf, h + 0.12), (xm + 0.17, yf + 0.05, h + 0.25), "lab_plastic_white")
    s.plane((xm, yf + 0.051, h + 0.185), (0.15, 0.05), (-1, 0, 0), (0, 0, 1), "lab_exit")


def _window_frame(s, lo, hi, axis):
    """Frame + mullion + sill around an opening in a wall; axis 'y' = back wall (normal -y), 'x' = right wall,
    '-y' = row B's wall (two-row cell, normal +y)."""
    fw, fd = 0.05, 0.06
    (a0, a1), (z0, z1), w = lo, hi, axis
    am = (a0 + a1) / 2

    def b(u0, u1, v0, v1, d0, d1, mat="lab_frame"):
        if w == "y":
            s.box((u0, WALL_BACK_Y - d1, v0), (u1, WALL_BACK_Y - d0, v1), mat)
        elif w == "-y":
            s.box((u0, WALL_FRONT_Y + d0, v0), (u1, WALL_FRONT_Y + d1, v1), mat)
        else:
            s.box((WALL_RIGHT_X - d1, u0, v0), (WALL_RIGHT_X - d0, u1, v1), mat)

    b(a0 - fw, a1 + fw, z1, z1 + fw, 0, fd)
    b(a0 - fw, a1 + fw, z0 - fw, z0, 0, fd)
    b(a0 - fw, a0, z0, z1, 0, fd)
    b(a1, a1 + fw, z0, z1, 0, fd)
    b(am - 0.02, am + 0.02, z0, z1, 0, fd * 0.7)
    b(a0 - 0.02, a1 + 0.02, z0 + 0.62 * (z1 - z0) - 0.015, z0 + 0.62 * (z1 - z0) + 0.015, 0, fd * 0.7)
    b(a0 - fw - 0.04, a1 + fw + 0.04, z0 - fw - 0.03, z0 - fw, 0, 0.16, "lab_frame")       # sill


def _window_back(s, x0, x1, z0, z1):
    s.plane(((x0 + x1) / 2, WALL_BACK_Y - 0.004, (z0 + z1) / 2), ((x1 - x0) / 2, (z1 - z0) / 2), (1, 0, 0), (0, 0, 1),
            "lab_sky")
    _window_frame(s, (x0, x1), (z0, z1), "y")


def _window_front(s, x0, x1, z0, z1):
    """Window in row B's wall (two-row cell): the back-wall window seen from the other side of the aisle."""
    s.plane(((x0 + x1) / 2, WALL_FRONT_Y + 0.004, (z0 + z1) / 2), ((x1 - x0) / 2, (z1 - z0) / 2), (-1, 0, 0),
            (0, 0, 1), "lab_sky")
    _window_frame(s, (x0, x1), (z0, z1), "-y")


def _window_right(s, y0, y1, z0, z1):
    s.plane((WALL_RIGHT_X - 0.004, (y0 + y1) / 2, (z0 + z1) / 2), ((y1 - y0) / 2, (z1 - z0) / 2), (0, -1, 0),
            (0, 0, 1), "lab_sky")
    _window_frame(s, (y0, y1), (z0, z1), "x")


def _services(s, xmin, xmax, bench_top, bench_back):
    """Cable tray above the row, drops to every printer, power strip, distribution board, conduit."""
    yb = WALL_BACK_Y
    z0, x0, x1 = 2.12, xmin - 0.6, xmax + 0.3
    s.box((x0, yb - 0.27, z0), (x1, yb - 0.04, z0 + 0.006), "lab_galv")
    s.box((x0, yb - 0.27, z0), (x1, yb - 0.264, z0 + 0.07), "lab_galv")
    s.box((x0, yb - 0.046, z0), (x1, yb - 0.04, z0 + 0.07), "lab_galv")
    s.cap((x0 + 0.05, yb - 0.2, z0 + 0.025), (x1 - 0.05, yb - 0.2, z0 + 0.025), 0.018, "lab_cable")
    s.cap((x0 + 0.05, yb - 0.14, z0 + 0.022), (x1 - 0.05, yb - 0.14, z0 + 0.022), 0.014, "lab_cable_blue")
    x = x0 + 0.1
    while x < x1:
        s.box((x - 0.015, yb - 0.27, z0 - 0.03), (x + 0.015, yb, z0), "lab_galv")
        x += 1.2
    # distribution board + conduit down/up at the left end of the row
    xe = x0 + 0.2
    s.box((xe - 0.22, yb - 0.12, 1.25), (xe + 0.22, yb, 1.85), "lab_elec")
    s.box((xe - 0.2, yb - 0.123, 1.27), (xe + 0.2, yb - 0.12, 1.83), "lab_frame")
    s.plane((xe, yb - 0.1235, 1.7), (0.045, 0.056), (1, 0, 0), (0, 0, 1), "lab_warning")
    s.cyl((xe, yb - 0.03, 1.85), (xe, yb - 0.03, z0), 0.02, "lab_galv")
    s.cyl((xe + 0.12, yb - 0.03, 0.0), (xe + 0.12, yb - 0.03, 1.25), 0.02, "lab_galv")
    # power strip along the bench back + vertical feed
    s.box((xmin + 0.05, bench_back - 0.08, bench_top), (xmax - 0.05, bench_back - 0.03, bench_top + 0.035),
          "lab_plastic_white")
    s.cyl((xmax - 0.04, yb - 0.05, bench_top + 0.02), (xmax - 0.04, yb - 0.05, z0), 0.012, "lab_cable")


# ---------------------------------------------------------------- printers
def _printer_dress(s, pos, side, i, collide_cls, bench_back):
    """AMS on top, PTFE, display, logo, chamber LED strip, power cable. Local: x right (aisle view), y in, z up."""
    W, D, H = C.PRINTER_W, C.PRINTER_D, C.PRINTER_H
    px, py, pz = pos

    def P(xl, yl, zl):
        return (px + side * xl, py + side * yl, pz + zl)

    def B(lo, hi, mat, **kw):
        a, b = P(*lo), P(*hi)
        s.box(a, b, mat, **kw)

    # AMS: front face 0.08 behind the door plane, collision proxy = one box (hidden group 3)
    ax, ay0, ay1, az0, az1 = 0.18, 0.08, 0.37, H, H + 0.23
    B((-ax, ay0, az0), (ax, ay1, az1), None, collide=True, group=3)
    zt = az0 + 0.125                                                    # tub top / lid split
    B((-ax, ay0, az0 + 0.012), (ax, ay1, zt), "lab_ams")
    B((-ax + 0.002, ay0 + 0.002, zt), (ax - 0.002, ay1 - 0.002, az1), "lab_amslid")
    B((-0.06, ay0 - 0.004, zt + 0.004), (0.06, ay0, zt + 0.016), "lab_plastic_dark")     # lid grip
    rng = np.random.default_rng(1000 + i)
    names = list(SPOOL_COLORS)
    pick = rng.choice(len(names), 4, replace=False)
    zc, yc = zt - 0.01, (ay0 + ay1) / 2
    for k, xk in enumerate((-0.132, -0.044, 0.044, 0.132)):
        s.cyl(P(xk - 0.03, yc, zc), P(xk + 0.03, yc, zc), 0.088, f"lab_fil_{names[pick[k]]}")
        for e in (-1, 1):
            s.cyl(P(xk + e * 0.03, yc, zc), P(xk + e * 0.034, yc, zc), 0.097, "lab_flange")
        led = "lab_led_green" if k == i % 4 else "lab_led_white"
        B((xk - 0.012, ay0 - 0.002, az0 + 0.07), (xk + 0.012, ay0, az0 + 0.078), led)
    # PTFE from the AMS top-back over to the printer's back panel
    pts = [(0.1, ay1 - 0.03, az1), (0.1, ay1 + 0.01, az1 + 0.03), (0.1, D + 0.06, az1 - 0.01),
           (0.1, D + 0.075, H - 0.05), (0.1, D + 0.045, H - 0.12), (0.1, D + 0.004, H - 0.15)]
    s.poly([P(*p) for p in pts], 0.0032, "lab_ptfe")
    # front display on the fixed header (above the door leaf: z 0.43..0.46), logo on the plinth, status LED
    B((-0.168, -0.004, H - 0.027), (-0.092, 0.0, H - 0.003), "lab_bezel")
    s.plane(P(-0.13, -0.0042, H - 0.015), (0.033, 0.0095), (side, 0, 0), (0, 0, 1), "lab_pscreen")
    B((-0.03, -0.002, 0.011), (0.03, 0.0, 0.019), "lab_logo")
    B((0.15, -0.002, H - 0.017), (0.158, 0.0, H - 0.013), "lab_led_green")
    # chamber LED bar behind the header, inside the top front
    B((-0.15, 0.017, H - 0.027), (0.15, 0.027, H - 0.017), "lab_led")
    # power cable: back panel -> plug on the power strip (strip runs along the bench back; row B: its own bench)
    ys0, ys1 = abs(bench_back) - 0.08 - side * py, abs(bench_back) - 0.03 - side * py     # strip, local y
    pc = [(-0.12, D + 0.002, 0.07), (-0.12, D + 0.02, 0.05), (-0.12, ys0 - 0.005, 0.045)]
    s.poly([P(*p) for p in pc], 0.0035, "lab_cable")
    B((-0.14, ys0 + 0.005, 0.035), (-0.1, ys1 - 0.005, 0.06), "lab_plastic_dark")
    # data drop: down the wall from its own row's cable tray, then over to the printer's back panel
    xd = px + side * 0.15
    yw = (WALL_BACK_Y if side > 0 else WALL_FRONT_Y) - side * 0.05
    s.poly([(xd, yw, 2.12), (xd, yw, 1.0), (xd, py + side * (D + 0.004), pz + 0.24)], 0.0035,
           "lab_cable" + (WALL_B if side < 0 else ""))


def _printer_detail(s, pos, side, i, rng):
    """Beauty-only detail inside/on each printer (GROUP_DETAIL, invisible to the chamber depth camera): toolhead cover,
    fan, hotend and nozzle around the parked toolhead box, guide rods, exhaust grille, the chamber camera itself, a
    label on the reject tray; plus the status light bar on the front header (lab_status<i>, recolourable)."""
    W, D, H = C.PRINTER_W, C.PRINTER_D, C.PRINTER_H
    px, py, pz = pos
    G = GROUP_DETAIL

    def P(xl, yl, zl):
        return (px + side * xl, py + side * yl, pz + zl)

    def B(lo, hi, mat, group=G):
        s.box(P(*lo), P(*hi), mat, group=group)

    def Cy(a, b, r, mat, group=G):
        s.cyl(P(*a), P(*b), r, mat, group=group)

    # toolhead (collision box: x 0.085..0.155, y 0.31..0.37, z 0.325..0.415): cover, fan, hotend, nozzle, cable
    B((0.087, 0.305, 0.33), (0.153, 0.31, 0.41), "lab_toolcover")
    Cy((0.12, 0.305, 0.388), (0.12, 0.302, 0.388), 0.019, "lab_plastic_dark")
    Cy((0.12, 0.302, 0.388), (0.12, 0.3005, 0.388), 0.007, "lab_steel")
    B((0.108, 0.328, 0.311), (0.132, 0.352, 0.325), "lab_steel")
    Cy((0.12, 0.34, 0.311), (0.12, 0.34, 0.304), 0.0045, "lab_brass")
    s.poly([P(0.12, 0.345, 0.415), P(0.12, 0.36, 0.437), P(0.02, 0.372, 0.443), P(-0.15, 0.372, 0.443)], 0.006,
           "lab_cablechain")
    # guide rods in the back corners, exhaust grille on the back wall
    for e in (-1, 1):
        Cy((e * 0.172, 0.366, 0.03), (e * 0.172, 0.366, 0.44), 0.0045, "lab_chrome")
    Cy((-0.06, D - 0.0085, 0.27), (-0.06, D - 0.0105, 0.27), 0.045, "lab_plastic_dark")
    for k in range(3):
        B((-0.1, D - 0.012, 0.247 + 0.02 * k), (-0.02, D - 0.0105, 0.252 + 0.02 * k), "lab_steel")
    # the chamber camera (perception) at the top-front-left corner
    B((-0.178, 0.018, H - 0.045), (-0.142, 0.042, H - 0.02), "lab_plastic_dark")
    Cy((-0.16, 0.03, H - 0.046), (-0.16, 0.03, H - 0.05), 0.007, "lab_bezel")
    # reject tray label (tray front wall outer face at y_l = 0.008, z 0.03..0.06)
    a, b = P(-0.075, 0.0075, 0.034), P(0.075, 0.0075, 0.056)
    # (P() turns row B by 180 deg, so the first corner is on the viewer's left in both rows: same texcoords)
    s.panel([(a[0], a[1], a[2]), (b[0], a[1], a[2]), (b[0], a[1], b[2]), (a[0], a[1], b[2])], (0, -side, 0),
            "lab_reject", uvs=[(0, 1), (1, 1), (1, 0), (0, 0)], group=G)
    # status light bar on the header (above the door leaf), recoloured by renderers through m.mat_rgba
    B((-0.07, -0.003, H - 0.012), (0.13, 0.0, H - 0.007), f"lab_status{i}", group=2)


def _wip_prints(s, i, color, rng):
    """Showcase set dressing (GROUP_SHOWCASE): parts half-way through printing on printer i's bed, in its filament
    colour, as one mesh geom per printer (lab_batch_lab_fil_<c>_g5_wip<i>) so a renderer can hide the printer it unloads.
    Never rendered by the sim's own cameras: real parts are spawned on these beds in farm runs."""
    G, mat, tag = GROUP_SHOWCASE, f"lab_fil_{color}", f"wip{i}"
    slots = [(55, 70), (128, 72), (200, 70), (70, 160), (140, 165), (205, 170), (100, 225), (180, 228)]
    pick = rng.choice(len(slots), int(rng.integers(3, 6)), replace=False)
    frac = float(rng.uniform(0.25, 0.85))                 # print progress: every part is at the same layer
    for k in pick:
        x, y = slots[k]
        x, y = x + rng.uniform(-8, 8), y + rng.uniform(-8, 8)
        h = max(frac * float(rng.uniform(10, 26)), 1.5) / 1000
        z0 = C.bed_mm_to_world(i, x, y)
        if rng.random() < 0.5:
            r = float(rng.uniform(9, 20)) / 1000
            s.cyl(z0, (z0[0], z0[1], z0[2] + h), r, mat, group=G, tag=tag)
        else:
            hx, hy = (float(v) / 2000 for v in rng.uniform(16, 40, 2))
            s.box((z0[0] - hx, z0[1] - hy, z0[2]), (z0[0] + hx, z0[1] + hy, z0[2] + h), mat, group=G, tag=tag)
    a, b = C.bed_mm_to_world(i, 20, 6), C.bed_mm_to_world(i, 95, 6)  # prime line
    s.box((a[0], a[1] - 0.0012, a[2]), (b[0], b[1] + 0.0012, a[2] + 0.0006), mat, group=G, tag=tag)


def _carriage_dress(root, asset):
    """Visual detail on the moving carriage body: bin labels, a name plate, bumper strips, control cabinet with an
    e-stop. Every geom is density 0 and contype/conaffinity 0: body mass, inertia and contacts are unchanged."""
    car = root.find(".//body[@name='carriage']")
    if car is None:
        return 0
    decks = [g for g in car.findall("geom") if (g.get("name") or "").startswith("deck")]   # two rows: the deck is
    if not decks:                                                                           # cut open under the totes
        return 0
    lo = np.min([np.subtract(_vec(g.get("pos")), _vec(g.get("size"))) for g in decks], axis=0)
    hi = np.max([np.add(_vec(g.get("pos")), _vec(g.get("size"))) for g in decks], axis=0)
    x0, x1, y0, y1, ztop = lo[0], hi[0], lo[1], hi[1], hi[2]
    n = 0

    def geom(**a):
        nonlocal n
        a.update(contype="0", conaffinity="0", density="0", group=a.get("group", "2"))
        ET.SubElement(car, "geom", {k: (_f(*v) if isinstance(v, (tuple, list)) else str(v)) for k, v in a.items()})
        n += 1

    def box(lo, hi, mat):
        geom(type="box", pos=tuple((a + b) / 2 for a, b in zip(lo, hi)), size=tuple(abs(b - a) / 2 for a, b in zip(lo, hi)),
             material=mat)

    # bumper strips at both ends of the deck (direction of travel)
    for e in (-1, 1):
        xe = x1 if e > 0 else x0
        box((min(xe, xe + e * 0.016), y0, ztop - 0.032), (max(xe, xe + e * 0.016), y1, ztop - 0.002), "lab_hazard")
    # control cabinet under the deck, behind the column, with an e-stop and a status lamp facing the aisle
    if TWO_ROWS:                        # two rows: the totes' trap doors open under the +x part of the deck and the
        # station conveyor passes under it, so the cabinet hangs under the -x end, behind the column
        cx0, cx1, cy0, cy1, cz0, cz1 = x0 + 0.02, C.COLUMN_X[0] - 0.012, y0 + 0.04, y1 - 0.04, -0.27, ztop - 0.03
    else:
        cx0, cx1, cy0, cy1, cz0, cz1 = 0.13, 0.25, y0 + 0.05, -0.1, -0.27, ztop - 0.03
    box((cx0, cy0, cz0), (cx1, cy1, cz1), "lab_anodized")
    box((cx0 + 0.02, cy0 - 0.012, cz1 - 0.07), (cx0 + 0.07, cy0, cz1 - 0.02), "lab_yellow")
    geom(type="cylinder", fromto=(cx0 + 0.045, cy0 - 0.012, cz1 - 0.045, cx0 + 0.045, cy0 - 0.03, cz1 - 0.045),
         size=(0.016,), material="lab_red")
    geom(type="cylinder", fromto=(cx1 - 0.03, cy0 - 0.001, cz1 - 0.045, cx1 - 0.03, cy0 - 0.008, cz1 - 0.045),
         size=(0.008,), material="lab_led_green")
    # textured quads (name plate on the column, a tag on every bin): one mesh, atlas UVs
    ms = _Mesh()
    col = next((g for g in car.findall("geom") if g.get("name") == "column"), None)
    if col is not None:
        (ccx, ccy, ccz), (chx, chy, chz) = _vec(col.get("pos")), _vec(col.get("size"))
        yf = ccy - chy - 0.0006
        pw = min(0.1, chx - 0.006)      # plate half-width (4:1 texture)
        ph = pw / 4
        ms.quad([(ccx - pw, yf, ccz + 0.05), (ccx + pw, yf, ccz + 0.05), (ccx + pw, yf, ccz + 0.05 + 2 * ph),
                 (ccx - pw, yf, ccz + 0.05 + 2 * ph)], (0, -1, 0), [(0, 1), (1, 1), (1, 0), (0, 0)])
        if TWO_ROWS:                    # the column faces a row on either side
            yb = ccy + chy + 0.0006
            ms.quad([(ccx + pw, yb, ccz + 0.05), (ccx - pw, yb, ccz + 0.05), (ccx - pw, yb, ccz + 0.05 + 2 * ph),
                     (ccx + pw, yb, ccz + 0.05 + 2 * ph)], (0, 1, 0), [(0, 1), (1, 1), (1, 0), (0, 0)])
    if ms.chunks:
        _mesh_geom(car, asset, ms, "lab_carr_plate", "lab_carrplate")
        n += 1
    ms = _Mesh()
    nl = len(BIN_LABELS)
    hx_b, hy_b = C.BIN_SIZE[0] / 2, C.BIN_SIZE[1] / 2
    for k, (slot, _) in enumerate(BIN_LABELS):
        if slot not in C.CADDY_SLOTS:
            continue
        bx, by = C.CADDY_SLOTS[slot]
        v0, v1 = k / nl, (k + 1) / nl
        if TWO_ROWS:                    # bins packed in a block beside the arm: tag an outer face of each
            _bin_tag(ms, slot, bx, by, hx_b, hy_b, v0, v1)
            continue
        yf = by - hy_b - 0.0006
        ms.quad([(bx - 0.042, yf, 0.022), (bx + 0.042, yf, 0.022), (bx + 0.042, yf, 0.043), (bx - 0.042, yf, 0.043)],
                (0, -1, 0), [(0, v1), (1, v1), (1, v0), (0, v0)])
    if ms.chunks:
        _mesh_geom(car, asset, ms, "lab_carr_binlabels", "lab_binlabel")
        n += 1
    return n


def _bin_tag(ms, slot, bx, by, hx, hy, v0, v1):
    """Label quad on the first free outer face of a caddy bin (-y, +y, +x, -x); none if the bin is boxed in."""
    others = [xy for sl, xy in C.CADDY_SLOTS.items() if sl != slot]
    pitch_x, pitch_y = 2 * hx + 0.01, 2 * hy + 0.01
    uv = [(0, v1), (1, v1), (1, v0), (0, v0)]
    z0, z1, a = 0.022, 0.043, 0.042
    for ux, uy in ((0, -1), (0, 1), (1, 0), (-1, 0)):
        if any(abs(ox - bx - ux * pitch_x) < 0.03 and abs(oy - by - uy * pitch_y) < 0.03 for ox, oy in others):
            continue
        if uy:                          # corners left -> right as seen from outside the face
            yf = by + uy * (hy + 0.0006)
            xa, xb = (bx - a, bx + a) if uy < 0 else (bx + a, bx - a)
            ms.quad([(xa, yf, z0), (xb, yf, z0), (xb, yf, z1), (xa, yf, z1)], (0, uy, 0), uv)
        else:
            xf = bx + ux * (hx + 0.0006)
            ya, yb = (by - a, by + a) if ux > 0 else (by + a, by - a)
            ms.quad([(xf, ya, z0), (xf, yb, z0), (xf, yb, z1), (xf, ya, z1)], (ux, 0, 0), uv)
        return


def _mesh_geom(body, asset, ms, name, mat):
    v, nn, f, t = ms.arrays()
    a = {"name": name, "inertia": "shell", "vertex": " ".join(f"{x:.6g}" for x in v.ravel()),
         "normal": " ".join(f"{x:.4g}" for x in nn.ravel()), "face": " ".join(map(str, f.ravel()))}
    if t is not None:
        a["texcoord"] = " ".join(f"{x:.5g}" for x in t.ravel())
    ET.SubElement(asset, "mesh", a)
    ET.SubElement(body, "geom", {"name": name, "type": "mesh", "mesh": name, "material": mat, "contype": "0",
                                 "conaffinity": "0", "density": "0", "group": "2"})


# ---------------------------------------------------------------- rail
def _rail(s, rail):
    (cx, cy, cz), (hx, hy, hz) = _vec(rail.get("pos")), _vec(rail.get("size"))
    x0, x1, top = cx - hx, cx + hx, cz + hz
    for e in (-1, 1):
        s.box((x0 + 0.03, cy + e * 0.035 - 0.009, top), (x1 - 0.03, cy + e * 0.035 + 0.009, top + 0.014), "lab_steel")
        s.box((x0, cy + e * (hy + 0.0005) - 0.001, cz - 0.006), (x1, cy + e * (hy + 0.0005) + 0.001, cz + 0.006),
              "lab_steel")
    for x in (x0, x1):                  # end caps (two rows: inside the rail's +x end, the conveyor follows)
        xa, xb = (x - C.RAIL_END + 0.004, x) if TWO_ROWS and x == x1 else (x - 0.02, x + 0.02)
        s.box((xa, cy - hy - 0.005, 0), (xb, cy + hy + 0.005, top + 0.03), "lab_plastic_dark")
    # end stops; two rows: the +x one in the last C.RAIL_END of the rail, past the column at full travel
    for xa, xb in ((x0 + 0.02, x0 + 0.06), (x1 - C.RAIL_END + 0.004, x1 - 0.002) if TWO_ROWS else (x1 - 0.06, x1 - 0.02)):
        s.box((xa, cy - 0.05, top + 0.014), (xb, cy + 0.05, top + 0.06), "lab_endstop")
    x = x0 + 0.3
    while x < x1 - 0.1:
        s.box((x - 0.04, cy - hy - 0.035, 0), (x + 0.04, cy + hy + 0.035, 0.01), "lab_steel")
        x += 0.8


def _floor_marks(s, rail, station_x0, station_hy):
    """Hazard tape around the carriage + arm sweep, walkway lines in front."""
    (cx, _, _), (hx, _, _) = _vec(rail.get("pos")), _vec(rail.get("size"))
    xl = min(C.RAIL_X_MIN - 0.27 - 0.42, cx - hx - 0.25)
    xr = station_x0 + 0.36
    yf = -1.02
    w, t = 0.06, 0.0016
    s.box((xl, yf - w, 0), (xr + w, yf, t), "lab_hazard")
    s.box((xl - w, yf - w, 0), (xl, WALL_BACK_Y - 0.02, t), "lab_hazard")
    s.box((xr, yf, 0), (xr + w, -station_hy - 0.04, t), "lab_hazard")
    s.box((xr, station_hy + 0.04, 0), (xr + w, WALL_BACK_Y - 0.02, t), "lab_hazard")
    # walkway
    for y in (-1.85, -3.0):
        s.box((WALL_LEFT_X + 0.6, y - 0.025, 0), (WALL_RIGHT_X - 0.6, y + 0.025, t), "lab_tape_green")
    return xl, xr, yf


def _floor_marks_two(s, rail, station_x0, station_hy, bench_x0, bench_x1, gap=None):
    """Two rows: the robot zone is the aisle plus the open floor past both ends of the benches. Hazard tape crosses the
    room wall to wall at the operator bay and at the packing line (a gap for the conveyor); along the aisle a strip runs
    in front of each row's bench (never under it)."""
    (cx, _, _), (hx, _, _) = _vec(rail.get("pos")), _vec(rail.get("size"))
    xl = min(C.RAIL_X_MIN - 0.27 - 0.42, cx - hx - 0.25)
    xr = station_x0 + 0.36
    w, t = 0.06, 0.0016
    yw = WALL_BACK_Y - 0.02
    s.box((xl - w, -yw, 0), (xl, yw, t), "lab_hazard")
    g0, g1 = gap or (-station_hy - 0.04, station_hy + 0.04)     # opening where the parcels leave the zone
    s.box((xr, -yw, 0), (xr + w, g0, t), "lab_hazard")
    s.box((xr, g1, 0), (xr + w, yw, t), "lab_hazard")
    ya = C.AISLE_HALF - 0.025
    for e in (1, -1):
        s.box((bench_x0, min(e * ya, e * (ya - w)), 0), (bench_x1, max(e * ya, e * (ya - w)), t), "lab_hazard")
    # operator bay: green walkway from the door to the zone line, stand-off box in front of the warning post
    xd0, xd1 = DOOR_X
    yf = WALL_FRONT_Y
    g = 0.05
    s.box((xd0 - 0.1, yf + 0.03, 0), (xd0 - 0.1 + g, -0.25, t), "lab_tape_green")
    s.box((xd0 - 0.1, -0.25 - g, 0), (xl - w - 0.12, -0.25, t), "lab_tape_green")
    return xl, xr


def _cable_channel(s, rail):
    """Two rows: the carriage's energy chain runs in a covered floor channel beside the rail; nothing crosses the aisle.
    Chequer-plate cover with a slot for the chain's moving end."""
    (cx, cy, _), (hx, hy, _) = _vec(rail.get("pos")), _vec(rail.get("size"))
    x0, x1 = cx - hx + 0.05, cx + hx - 0.05
    y0, y1 = cy + hy + 0.05, cy + hy + 0.16
    s.box((x0, y0, 0), (x1, y1, 0.016), "lab_brushed")
    s.box((x0, (y0 + y1) / 2 - 0.007, 0.016), (x1, (y0 + y1) / 2 + 0.007, 0.0168), "lab_cablechain")
    for x in (x0, x1):
        s.box((x - 0.01, y0 - 0.005, 0), (x + 0.01, y1 + 0.005, 0.03), "lab_plastic_dark")


# ---------------------------------------------------------------- station
def _station(s, belt, tunnel, lanes):
    (bx, by, bz), (bhx, bhy, bhz) = _vec(belt.get("pos")), _vec(belt.get("size"))
    btop = bz + bhz
    tx0 = tx1 = None
    if tunnel is not None:
        (tx, ty, tz), (thx, thy, thz) = _vec(tunnel.get("pos")), _vec(tunnel.get("size"))
        tx0, tx1 = tx - thx, tx + thx
    # roller bed up to the tunnel exit (packages ride 1 cm above the belt), drums at both ends
    x_end = tx1 if tx1 else bx + bhx
    x = bx - bhx + 0.03
    r = 0.018
    while x < x_end - 0.01:
        s.cyl((x, by - bhy + 0.006, btop + 0.008 - r), (x, by + bhy - 0.006, btop + 0.008 - r), r, "lab_roller")
        x += 0.055
    for xe in (bx - bhx, bx + bhx):
        s.cyl((xe, by - bhy - 0.01, btop - 0.025), (xe, by + bhy + 0.01, btop - 0.025), 0.026, "lab_roller")
    # side guides before the tunnel only (packages fan out to the lanes after it)
    xg1 = tx0 if tx0 else bx + bhx
    for e in (-1, 1):
        s.box((bx - bhx, by + e * (bhy + 0.004) - 0.008, btop - 0.03), (xg1, by + e * (bhy + 0.004) + 0.008, btop + 0.03),
              "lab_alu")
    # e-stop on the base, facing the aisle
    s.box((bx - bhx + 0.46, by - bhy - 0.05, btop - 0.16), (bx - bhx + 0.54, by - bhy - 0.02, btop - 0.08), "lab_yellow")
    s.cyl((bx - bhx + 0.5, by - bhy - 0.05, btop - 0.12), (bx - bhx + 0.5, by - bhy - 0.07, btop - 0.12), 0.022,
          "lab_red")
    if tunnel is not None:
        # print-and-apply labeller on the -y wall of the tunnel, tower light on top, entry/exit curtains
        yw = ty - thy - 0.01
        s.box((tx - 0.13, yw - 0.16, tz - 0.2), (tx + 0.13, yw, tz - 0.03), "lab_plastic_white")
        s.box((tx - 0.13, yw - 0.161, tz - 0.12), (tx - 0.02, yw - 0.16, tz - 0.06), "lab_bezel")
        s.plane((tx - 0.075, yw - 0.1615, tz - 0.09), (0.045, 0.024), (1, 0, 0), (0, 0, 1), "lab_pscreen")
        s.cyl((tx + 0.06, yw - 0.12, tz - 0.01), (tx + 0.06, yw - 0.04, tz - 0.01), 0.055, "lab_plastic_white")
        s.cyl((tx + 0.06, yw - 0.121, tz - 0.01), (tx + 0.06, yw - 0.039, tz - 0.01), 0.022, "lab_plastic_dark")
        s.box((tx - 0.15, yw - 0.12, tz - 0.21), (tx + 0.15, yw, tz - 0.2), "lab_steel")
        xt, yt, zt = tx + thx - 0.05, ty + thy - 0.05, tz + thz
        s.cyl((xt, yt, zt), (xt, yt, zt + 0.12), 0.01, "lab_steel")
        for k, m in enumerate(("lab_tower_r", "lab_tower_a", "lab_tower_g")):
            s.cyl((xt, yt, zt + 0.12 + 0.045 * (2 - k)), (xt, yt, zt + 0.12 + 0.045 * (3 - k) - 0.003), 0.024, m)
        s.cyl((xt, yt, zt + 0.255), (xt, yt, zt + 0.265), 0.024, "lab_plastic_dark")
        for xc in (tx0 + 0.005, tx1 - 0.005):
            s.box((xc - 0.003, ty - thy + 0.02, btop + 0.12), (xc + 0.003, ty + thy - 0.02, tz - thz), "lab_amslid")
    # lanes = carrier totes: dark base + rim (open on the conveyor side, where parcels drop in), carrier label,
    # floor outline in the lane colour, coloured sign post with a sign facing the aisle (heights staggered)
    c45 = math.sqrt(0.5)
    for k, g in enumerate(lanes):
        (lx, ly, lz), (lhx, lhy, lhz) = _vec(g.get("pos")), _vec(g.get("size"))
        mat = g.get("material")
        x0, x1, y0, y1, ztop = lx - lhx, lx + lhx, ly - lhy, ly + lhy, lz + lhz
        s.box((x0 - 0.05, y0 - 0.05, 0), (x1 + 0.05, y1 + 0.05, 0.0016), mat)
        s.box((x0 - 0.006, y0 - 0.006, 0), (x1 + 0.006, y1 + 0.006, 0.05), "lab_plastic_dark")
        s.box((x1 - 0.02, y0 - 0.008, ztop - 0.03), (x1 + 0.008, y1 + 0.008, ztop + 0.012), "lab_plastic_dark")
        for yy in (y0, y1):
            s.box((x0 + 0.1, yy - 0.008, ztop - 0.03), (x1, yy + 0.008, ztop + 0.012), "lab_plastic_dark")
        s.box((x0 + 0.14, y0 - 0.002, ztop - 0.2), (x0 + 0.36, y0, ztop - 0.08), "lab_plastic_white")
        xp, yp, zs = x1 + 0.12, ly + 0.08, 1.18 + 0.2 * k
        s.cyl((xp, yp, 0), (xp, yp, zs + 0.15), 0.022, mat)
        s.cyl((xp, yp, 0), (xp, yp, 0.02), 0.08, "lab_steel")
        q = (math.cos(-math.pi / 8), 0, 0, math.sin(-math.pi / 8))       # yaw -45 deg: sign faces (-1,-1)/sqrt2
        s.add("box", (xp + 0.004 * c45, yp + 0.004 * c45, zs), (0.15, 0.006, 0.075), "lab_steel", quat=q)
        s.plane((xp - 0.003 * c45, yp - 0.003 * c45, zs), (0.145, 0.072), (c45, -c45, 0), (0, 0, 1), f"lab_lanesign{k}")
    # packing supplies on the +y side: table with flat boxes, tape gun, bubble wrap
    xa, xb, ya, yb2 = bx - 0.3, bx + 0.75, by + bhy + 0.3, WALL_BACK_Y - 0.05
    _table(s, xa, xb, ya, yb2, 0.86, "lab_worktop")
    for k in range(7):
        s.box((xa + 0.08, ya + 0.06, 0.86 + 0.012 * k), (xa + 0.58, ya + 0.42, 0.86 + 0.012 * (k + 1) - 0.002),
              "lab_kraft")
    s.cyl((xb - 0.13, ya + 0.06, 0.86 + 0.1), (xb - 0.13, yb2 - 0.18, 0.86 + 0.1), 0.1, "lab_bubble")
    s.box((xa + 0.7, ya + 0.1, 0.86), (xa + 0.78, ya + 0.24, 0.92), "lab_red")
    s.cyl((xa + 0.74, ya + 0.1, 0.94), (xa + 0.74, ya + 0.24, 0.94), 0.04, "lab_kraft")


def _station_two(s, L):
    """Two-row station dressing around farm.scene._station_two: roller conveyor along y under the deck's +x overhang,
    lift housings, box magazine, sealer tunnel details, sorter with diverters, carrier lane totes, scrap bin, camera
    post, packing table. Everything here is below the deck or outside the carriage's swept volume (the test in
    tests/test_station.py sweeps these primitives)."""
    from farm import scene
    xd, zc, cw = L["x_drop"], L["z_conv"], C.CONV_HALF_W
    ly = max(scene.box_outer(k)[1] for k in C.BOX_TYPES)
    y_lo, y_hi = L["y_sort"] - 0.15, L["y_mag"] + ly / 2
    lifts = (L["lift_dump_y"], L["lift_place_y"])
    r = 0.018
    # rollers (skipped where a lift table sits), side frame with guides, legs
    y = y_lo + 0.03
    while y < y_hi - 0.02:
        if not any(a - 0.02 < y < b + 0.02 for a, b in lifts):
            s.cyl((xd - cw + 0.016, y, zc - r), (xd + cw - 0.016, y, zc - r), r, "lab_roller")
        y += 0.06
    for e in (-1, 1):
        s.box((xd + e * cw - (0.012 if e > 0 else 0), y_lo, zc - 0.07), (xd + e * cw + (0 if e > 0 else 0.012), y_hi,
              zc - 0.01), "lab_alu")
        s.box((xd + e * (cw - 0.004) - 0.003, y_lo, zc - 0.01), (xd + e * (cw - 0.004) + 0.003, y_hi, zc + 0.022),
              "lab_alu")
        yy = y_lo + 0.05
        while yy < y_hi:
            s.box((xd + e * (cw - 0.02) - 0.015, yy - 0.015, 0), (xd + e * (cw - 0.02) + 0.015, yy + 0.015,
                  zc - 0.07), "lab_frame_dark")
            yy += 0.55
    (zb, zt), (hb, ht) = L["chute_z"], L["chute_hx"]   # clear drop chute: a thin top edge, hung from the frame
    ya, yb = L["lift_dump_y"]                           # at its ends (nothing across the view into the box)
    for e in (-1, 1):
        s.box((xd + e * (ht + 0.003) - 0.0025, ya, zt - 0.0025), (xd + e * (ht + 0.003) + 0.0025, yb, zt + 0.0025),
              "lab_alu")
        for yy in (ya, yb):
            s.box((xd + e * (cw - 0.006) - 0.004, yy - 0.004, zc + 0.022), (xd + e * (cw - 0.006) + 0.004, yy + 0.004,
                  zt + 0.004), "lab_alu")
            s.box((min(xd + e * (cw - 0.006), xd + e * ht), yy - 0.004, zt - 0.004),
                  (max(xd + e * (cw - 0.006), xd + e * ht), yy + 0.004, zt + 0.004), "lab_alu")
    for a, b in lifts:                  # lift housings (the tables' rams run in them)
        c = (a + b) / 2
        s.box((xd - 0.06, c - 0.06, 0), (xd + 0.06, c + 0.06, 0.13), "lab_anodized")
        s.box((xd - 0.065, c - 0.065, 0.13), (xd + 0.065, c + 0.065, 0.14), "lab_yellow")
    # magazine: a frame around each stack of erected boxes; the stack rests on ledges above the release slot
    for k, (sz, mx) in enumerate(sorted(L["mag_x"].items(), key=lambda kv: kv[1])):
        ox, oy, oz = scene.box_outer(sz)
        y0, y1 = L["y_mag"] - oy / 2 - 0.012, L["y_mag"] + oy / 2 + 0.012
        x0, x1 = mx - ox / 2 - 0.012, mx + ox / 2 + 0.012
        top = zc + 4 * oz + 0.08
        for px in (x0, x1):
            for py in (y0, y1):
                s.box((px - 0.008, py - 0.008, zc - 0.012), (px + 0.008, py + 0.008, top), "lab_alu")
        s.box((x0 - 0.008, y1 - 0.008, top - 0.016), (x1 + 0.008, y1 + 0.008, top), "lab_alu")
        s.box((x0 - 0.008, y0 - 0.008, top - 0.016), (x1 + 0.008, y0 + 0.008, top), "lab_alu")
        zl = zc + oz + 0.012
        for px, e in ((x0, 1), (x1, -1)):
            s.box((min(px, px + e * 0.014), y0, zl - 0.01), (max(px, px + e * 0.014), y1, zl), "lab_steel")
        for j in range(3):
            z0 = zl + j * (oz + 0.002)
            s.box((mx - ox / 2, L["y_mag"] - oy / 2, z0), (mx + ox / 2, L["y_mag"] + oy / 2, z0 + oz), "lab_kraft")
            s.box((mx - ox / 2 - 0.001, L["y_mag"] - 0.024, z0 + oz - 0.03), (mx + ox / 2 + 0.001, L["y_mag"] + 0.024,
                  z0 + oz + 0.001), "lab_kraft_tape")
        s.plane((mx, y0 - 0.009, top + 0.05), (0.05, 0.03), (1, 0, 0), (0, 0, 1), f"lab_lanesign{k}")
    # cross transfer in front of the magazine: rollers along y
    xm0, xm1 = min(L["mag_x"].values()) - 0.12, max(L["mag_x"].values()) + 0.12
    x = xm0 + 0.03
    while x < xm1 - 0.02:
        if abs(x - xd) > cw:
            s.cyl((x, L["y_disp"] - ly / 2 + 0.01, zc - r), (x, L["y_disp"] + ly / 2 - 0.01, zc - r), r, "lab_roller")
        x += 0.06
    # sealer tunnel: print-and-apply labeller on its +x wall, tower light, entry/exit curtains
    sx, sy = L["seal_half"]
    ys, zt = L["y_seal"], zc + 0.31
    xw = xd + sx + 0.01                 # (a dark housing: from above, a white one read as a stray slab)
    s.box((xw, ys - 0.13, zt - 0.2), (xw + 0.12, ys + 0.13, zt - 0.05), "lab_plastic_dark")
    s.box((xw + 0.12, ys - 0.12, zt - 0.14), (xw + 0.121, ys - 0.02, zt - 0.08), "lab_bezel")
    s.plane((xw + 0.1215, ys - 0.07, zt - 0.11), (0.045, 0.024), (0, 1, 0), (0, 0, 1), "lab_pscreen")
    s.cyl((xw + 0.03, ys + 0.06, zt - 0.02), (xw + 0.09, ys + 0.06, zt - 0.02), 0.04, "lab_plastic_white")
    s.cyl((xw + 0.029, ys + 0.06, zt - 0.02), (xw + 0.091, ys + 0.06, zt - 0.02), 0.016, "lab_plastic_dark")
    s.box((xw, ys - 0.15, zt - 0.21), (xw + 0.12, ys + 0.15, zt - 0.2), "lab_steel")
    xt, ytw = xd + sx - 0.05, ys - sy + 0.05
    s.cyl((xt, ytw, zt + 0.012), (xt, ytw, zt + 0.13), 0.01, "lab_steel")
    for k, m in enumerate(("lab_tower_r", "lab_tower_a", "lab_tower_g")):
        s.cyl((xt, ytw, zt + 0.13 + 0.045 * (2 - k)), (xt, ytw, zt + 0.13 + 0.045 * (3 - k) - 0.003), 0.024, m)
    s.cyl((xt, ytw, zt + 0.265), (xt, ytw, zt + 0.275), 0.024, "lab_plastic_dark")
    for yc in (ys - sy + 0.005, ys + sy - 0.005):
        s.box((xd - sx + 0.02, yc - 0.003, zc + 0.13), (xd + sx - 0.02, yc + 0.003, zt - 0.012), "lab_amslid")
    # sorter along +x: rollers along y, frame, a diverter paddle at each lane
    ysr = L["y_sort"]
    xs1 = max(L["lane_x"].values()) + 0.3
    x = xd + cw + 0.03
    while x < xs1 - 0.02:
        s.cyl((x, ysr - 0.14, zc - r), (x, ysr + 0.14, zc - r), r, "lab_roller")
        x += 0.06
    for e in (-1, 1):
        s.box((xd + cw, ysr + e * 0.15 - 0.006, zc - 0.07), (xs1, ysr + e * 0.15 + 0.006, zc + 0.022), "lab_alu")
    x = xd + 0.4
    while x < xs1:
        for e in (-1, 1):
            s.box((x - 0.015, ysr + e * 0.14 - 0.015, 0), (x + 0.015, ysr + e * 0.14 + 0.015, zc - 0.07),
                  "lab_frame_dark")
        x += 0.6
    c45 = math.sqrt(0.5)
    for k, c in enumerate(C.LANE_ORDER):
        lx = L["lane_x"][c]
        s.box((lx - 0.1, ysr - 0.182, zc - 0.02), (lx + 0.1, ysr - 0.158, zc + 0.08), "lab_plastic_dark")  # pusher
        s.box((lx - 0.09, ysr - 0.16, zc + 0.0), (lx + 0.09, ysr - 0.152, zc + 0.07), "lab_yellow")
        # lane tote: dark rim, carrier colour band, label, sign post
        x0, x1, y0, y1 = lx - 0.22, lx + 0.22, L["y_lane"] - 0.18, L["y_lane"] + 0.18
        zt_ = zc - 0.02
        for a, b in (((x0, y0), (x1, y0 + 0.012)), ((x0, y1 - 0.012), (x1, y1)), ((x0, y0), (x0 + 0.012, y1)),
                     ((x1 - 0.012, y0), (x1, y1))):
            s.box((*a, 0.05), (*b, zt_), "lab_plastic_dark")
        s.box((x0 - 0.001, y1 - 0.004, zt_ - 0.06), (x1 + 0.001, y1 + 0.001, zt_ - 0.03), f"lane{k}")
        s.box((x0 - 0.05, y0 - 0.05, 0), (x1 + 0.05, y1 + 0.05, 0.0016), f"lane{k}")
        xp, yp, zs = x1 + 0.1, y1 - 0.05, 1.18 + 0.2 * k
        s.cyl((xp, yp, 0), (xp, yp, zs + 0.15), 0.022, f"lane{k}")
        s.cyl((xp, yp, 0), (xp, yp, 0.02), 0.08, "lab_steel")
        q = (math.cos(-math.pi / 8), 0, 0, math.sin(-math.pi / 8))
        s.add("box", (xp + 0.004 * c45, yp + 0.004 * c45, zs), (0.15, 0.006, 0.075), "lab_steel", quat=q)
        s.plane((xp - 0.003 * c45, yp - 0.003 * c45, zs), (0.145, 0.072), (c45, -c45, 0), (0, 0, 1), f"lab_lanesign{k}")
    # scrap bin: a short chute lip toward the conveyor side, warning label
    hx, hy = L["scrap_half"]
    xsb, ysb, zt2 = L["x_scrap"], L["y_scrap"], L["z_scrap_top"]
    s.box((xsb - hx, ysb - hy, 0), (xsb + hx, ysb + hy, 0.03), "lab_grey_bin")
    s.plane((xsb + hx + 0.001, ysb, zt2 - 0.06), (0.05, 0.03), (0, 1, 0), (0, 0, 1), "lab_warning")
    # e-stop pillar beside the conveyor's -y end, facing the aisle end
    ex_, ey = xd - 0.3, L["y_seal"] - 0.1
    s.box((ex_ - 0.03, ey - 0.03, 0), (ex_ + 0.03, ey + 0.03, 1.0), "lab_yellow")
    s.cyl((ex_, ey + 0.03, 0.95), (ex_, ey + 0.05, 0.95), 0.025, "lab_red")
    # packing table with flat-packs and tape on the +y side, past the magazine
    xa, xb, ya, yb = xd + 0.55, xd + 1.6, L["y_mag"] - 0.3, WALL_BACK_Y - 0.05
    _table(s, xa, xb, ya, yb, 0.86, "lab_worktop")
    for k in range(7):
        s.box((xa + 0.08, ya + 0.06, 0.86 + 0.012 * k), (xa + 0.58, ya + 0.36, 0.86 + 0.012 * (k + 1) - 0.002),
              "lab_kraft")
    s.box((xa + 0.7, ya + 0.1, 0.86), (xa + 0.78, ya + 0.24, 0.92), "lab_red")
    s.cyl((xa + 0.74, ya + 0.1, 0.94), (xa + 0.74, ya + 0.24, 0.94), 0.04, "lab_kraft")
    return xs1 + 0.1


# ---------------------------------------------------------------- furniture
def _table(s, x0, x1, y0, y1, h, top_mat, leg_mat="lab_frame_dark", shelf=True):
    s.box((x0, y0, h - 0.03), (x1, y1, h), top_mat)
    for x in (x0 + 0.03, x1 - 0.03):
        for y in (y0 + 0.03, y1 - 0.03):
            s.box((x - 0.02, y - 0.02, 0), (x + 0.02, y + 0.02, h - 0.03), leg_mat)
    if shelf:
        s.box((x0 + 0.03, y0 + 0.03, 0.14), (x1 - 0.03, y1 - 0.03, 0.16), leg_mat)


def _spool(s, c, axis, color, r=0.1, w=0.065, flange="lab_flange"):
    a = np.asarray(axis, float) * w / 2
    c = np.asarray(c, float)
    s.cyl(c - a * 0.9, c + a * 0.9, r * 0.9, f"lab_fil_{color}")
    s.cyl(c - a, c - a * 0.9, r, flange)
    s.cyl(c + a * 0.9, c + a, r, flange)


def _furniture_left(s, xl):
    """Workbench with monitor/laptop/tools/dry box, pegboard, whiteboard, shelving with spools, bins, cart, extinguisher."""
    yb = WALL_BACK_Y
    x0, x1 = WALL_LEFT_X + 0.15, xl - 0.85
    y0 = yb - 0.72
    _workbench(s, x0, x1, yb)
    _whiteboard(s, WALL_LEFT_X, yb)
    _shelving(s, WALL_LEFT_X, -2.45, -1.15)
    _waste_bins(s, xl, yb)
    _cart(s, x0 + 0.85, x0 + 1.55, -3.85, -3.4)
    _stool(s, ((x0 + x1) / 2 - 0.3, y0 - 0.45))
    _extinguisher(s, WALL_LEFT_X, -3.2)
    _warning_post(s, xl - 0.3, -1.3)


def _workbench(s, x0, x1, yb):
    """Workbench against a wall at y = yb (normal -y), x0..x1, with its pegboard on the wall."""
    y0, y1, h = yb - 0.72, yb - 0.04, 0.9
    _table(s, x0, x1, y0, y1, h, "lab_worktop")
    s.box((x0 + 0.06, y0 + 0.05, 0.16), (x0 + 0.5, y1 - 0.08, h - 0.04), "lab_shelfsteel")          # drawer unit
    for k in range(4):
        z = 0.2 + k * 0.17
        s.box((x0 + 0.08, y0 + 0.048, z), (x0 + 0.48, y0 + 0.05, z + 0.15), "lab_elec")
        s.box((x0 + 0.22, y0 + 0.04, z + 0.11), (x0 + 0.34, y0 + 0.048, z + 0.125), "lab_steel")
    s.box((x0 + 0.1, y0 + 0.08, h), (x0 + 0.62, y0 + 0.42, h + 0.003), "lab_mat_green")       # cutting mat
    # monitor on a stand
    mx = (x0 + x1) / 2 + 0.1
    s.box((mx - 0.12, yb - 0.3, h), (mx + 0.12, yb - 0.14, h + 0.012), "lab_plastic_dark")
    s.box((mx - 0.025, yb - 0.21, h), (mx + 0.025, yb - 0.17, h + 0.22), "lab_plastic_dark")
    s.box((mx - 0.31, yb - 0.24, h + 0.12), (mx + 0.31, yb - 0.21, h + 0.48), "lab_plastic_dark")
    s.plane((mx, yb - 0.2405, h + 0.305), (0.295, 0.168), (1, 0, 0), (0, 0, 1), "lab_monitor")
    # laptop (open ~105 deg)
    lx, ly = x0 + 0.45, y0 + 0.28
    s.box((lx - 0.17, ly - 0.12, h), (lx + 0.17, ly + 0.12, h + 0.016), "lab_alu")
    ang = math.radians(15)
    q = (math.cos(-ang / 2), math.sin(-ang / 2), 0, 0)
    sc = (lx, ly + 0.12 + 0.11 * math.sin(ang), h + 0.016 + 0.11 * math.cos(ang))
    s.add("box", sc, (0.17, 0.006, 0.11), "lab_alu", quat=q)
    s.add("plane", (sc[0], sc[1] - 0.0065 * math.cos(ang), sc[2] + 0.0065 * math.sin(ang)), (0.155, 0.098, 0.1),
          "lab_laptop", xyaxes=(1, 0, 0, 0, math.sin(ang), math.cos(ang)))
    # filament dry box with two spools + hygrometer
    dx0 = x1 - 0.55
    s.box((dx0, yb - 0.4, h), (dx0 + 0.42, yb - 0.12, h + 0.24), "lab_dryclear")
    for k, col in enumerate(("teal", "white")):
        _spool(s, (dx0 + 0.12 + 0.18 * k, yb - 0.26, h + 0.115), (1, 0, 0), col, r=0.1)
    s.plane((dx0 + 0.21, yb - 0.402, h + 0.198), (0.03, 0.017), (1, 0, 0), (0, 0, 1), "lab_pscreen")
    # loose spools + printed parts tray + tools
    _spool(s, (x1 - 0.22, y0 + 0.15, h + 0.1), (1, 0, 0), "orange", flange="lab_kraft")
    _spool(s, (x1 - 0.32, y0 + 0.15, h + 0.1), (1, 0, 0), "white")
    s.box((x0 + 0.7, y0 + 0.08, h), (x0 + 0.98, y0 + 0.3, h + 0.04), "lab_grey_bin")
    for k, col in enumerate(("red", "blue", "yellow", "green")):
        s.box((x0 + 0.73 + 0.06 * k, y0 + 0.12, h + 0.04), (x0 + 0.77 + 0.06 * k, y0 + 0.16 + 0.02 * k, h + 0.06),
              f"lab_fil_{col}")
    s.box((x0 + 0.66, y0 + 0.36, h), (x0 + 0.84, y0 + 0.38, h + 0.004), "lab_steel")              # scraper
    s.cap((x0 + 0.88, y0 + 0.4, h + 0.012), (x0 + 1.0, y0 + 0.44, h + 0.012), 0.01, "lab_red")      # screwdriver
    s.cap((x0 + 1.0, y0 + 0.44, h + 0.012), (x0 + 1.07, y0 + 0.465, h + 0.012), 0.003, "lab_steel")
    # pegboard with tools
    px0, px1 = x0 + 0.05, mx - 0.4
    s.box((px0, yb - 0.02, h + 0.1), (px1, yb, h + 0.85), "lab_pegboard")
    for k in range(5):
        tx = px0 + 0.1 + k * (px1 - px0 - 0.2) / 4
        s.box((tx - 0.012, yb - 0.035, h + 0.42 - 0.04 * (k % 2)), (tx + 0.012, yb - 0.02, h + 0.7), "lab_steel")
    s.cyl((px0 + 0.2, yb - 0.04, h + 0.25), (px0 + 0.2, yb - 0.025, h + 0.25), 0.1, "lab_fil_yellow")


def _whiteboard(s, wx, yb):
    """Whiteboard on a wall at x = wx (normal +x), y from yb - 1.95 to yb - 0.75."""
    s.box((wx, yb - 1.95, 1.0), (wx + 0.02, yb - 0.75, 1.95), "lab_frame")
    s.plane((wx + 0.0205, yb - 1.35, 1.475), (0.58, 0.455), (0, 1, 0), (0, 0, 1), "lab_whiteboard")
    s.box((wx + 0.02, yb - 1.9, 0.98), (wx + 0.07, yb - 0.8, 1.0), "lab_frame")


def _shelving(s, wx, sy0, sy1):
    """Steel shelving with spool boxes and spools against a wall at x = wx (normal +x), y from sy0 to sy1."""
    sx0, sx1 = wx + 0.05, wx + 0.5
    for x in (sx0, sx1 - 0.03):
        for y in (sy0, sy1 - 0.03):
            s.box((x, y, 0), (x + 0.03, y + 0.03, 1.95), "lab_shelfsteel")
    rng = np.random.default_rng(5)
    names = list(SPOOL_COLORS)
    for k, z in enumerate((0.12, 0.55, 0.98, 1.41, 1.85)):
        s.box((sx0, sy0, z), (sx1, sy1, z + 0.02), "lab_shelfsteel")
        if z > 1.8:
            continue
        y = sy0 + 0.04
        while y < sy1 - 0.25:
            if k % 2 == 0:                      # boxed spools (kraft + coloured label)
                s.box((sx0 + 0.04, y, z + 0.02), (sx0 + 0.25, y + 0.21, z + 0.1), "lab_kraft")
                s.box((sx0 + 0.25, y + 0.05, z + 0.04), (sx0 + 0.252, y + 0.16, z + 0.08),
                      f"lab_fil_{names[int(rng.integers(0, len(names)))]}")
                s.box((sx0 + 0.04, y, z + 0.1), (sx0 + 0.25, y + 0.21, z + 0.18), "lab_kraft")
                y += 0.24
            else:                               # loose spools standing edge-on to the room
                fl = ("lab_flange", "lab_kraft", "lab_plastic_white")[int(rng.integers(0, 3))]
                _spool(s, (sx0 + 0.21, y + 0.04, z + 0.12), (0, 1, 0), names[int(rng.integers(0, len(names)))],
                       flange=fl)
                y += 0.085


def _waste_bins(s, xl, yb):
    """General waste + failed-print recycling bins against the wall at y = yb, left of x = xl."""
    for k, (bxc, mat) in enumerate(((xl - 0.3, "lab_grey_bin"), (xl - 0.65, "lab_blue_bin"))):
        s.cyl((bxc, yb - 0.25, 0), (bxc, yb - 0.25, 0.62), 0.17, mat)
        s.cyl((bxc, yb - 0.25, 0.62), (bxc, yb - 0.25, 0.65), 0.18, mat)


def _cart(s, cx0, cx1, cy0, cy1):
    """Shop cart with spools and a bin, handle at the +x end."""
    for z in (0.18, 0.75):
        s.box((cx0, cy0, z), (cx1, cy1, z + 0.025), "lab_shelfsteel")
    for x in (cx0 + 0.02, cx1 - 0.02):
        for y in (cy0 + 0.02, cy1 - 0.02):
            s.cyl((x, y, 0.06), (x, y, 0.775), 0.012, "lab_steel")
            s.cyl((x, y - 0.03, 0.03), (x, y + 0.03, 0.03), 0.03, "lab_rubber")
    s.cyl((cx1 + 0.02, cy0 + 0.02, 0.775), (cx1 + 0.02, cy0 + 0.02, 1.0), 0.012, "lab_steel")
    s.cyl((cx1 + 0.02, cy1 - 0.02, 0.775), (cx1 + 0.02, cy1 - 0.02, 1.0), 0.012, "lab_steel")
    s.cyl((cx1 + 0.02, cy0 + 0.02, 1.0), (cx1 + 0.02, cy1 - 0.02, 1.0), 0.014, "lab_plastic_dark")
    for k, col in enumerate(("purple", "grey", "green")):
        _spool(s, (cx0 + 0.13 + 0.07 * k, (cy0 + cy1) / 2, 0.775 + 0.1), (1, 0, 0), col)
    s.box((cx0 + 0.4, cy0 + 0.05, 0.775), (cx1 - 0.05, cy1 - 0.05, 0.92), "lab_blue_bin")


def _stool(s, st):
    s.cyl((*st, 0.0), (*st, 0.02), 0.25, "lab_steel")
    s.cyl((*st, 0.02), (*st, 0.6), 0.025, "lab_alu")
    s.cyl((*st, 0.6), (*st, 0.66), 0.19, "lab_plastic_dark")


def _extinguisher(s, wx, fy):
    """Fire extinguisher + sign on a wall at x = wx (normal +x)."""
    s.box((wx, fy - 0.06, 0.55), (wx + 0.03, fy + 0.06, 0.62), "lab_steel")
    s.cyl((wx + 0.11, fy, 0.18), (wx + 0.11, fy, 0.66), 0.075, "lab_red")
    s.cyl((wx + 0.11, fy, 0.66), (wx + 0.11, fy, 0.72), 0.022, "lab_steel")
    s.box((wx + 0.07, fy - 0.07, 0.71), (wx + 0.15, fy + 0.02, 0.725), "lab_plastic_dark")
    s.poly([(wx + 0.11, fy + 0.03, 0.7), (wx + 0.2, fy + 0.07, 0.6), (wx + 0.19, fy + 0.07, 0.3)], 0.008, "lab_cable")
    s.plane((wx + 0.002, fy, 1.05), (0.15, 0.075), (0, 1, 0), (0, 0, 1), "lab_fire")


def _warning_post(s, ex, ey):
    """Robot-area warning sign + e-stop pedestal; sign and e-stop face -y."""
    s.cyl((ex, ey, 0), (ex, ey, 0.02), 0.12, "lab_steel")
    s.box((ex - 0.03, ey - 0.03, 0.02), (ex + 0.03, ey + 0.03, 1.05), "lab_yellow")
    s.box((ex - 0.06, ey - 0.05, 1.05), (ex + 0.06, ey + 0.05, 1.15), "lab_yellow")
    s.cyl((ex, ey - 0.05, 1.1), (ex, ey - 0.075, 1.1), 0.03, "lab_red")
    s.box((ex - 0.13, ey - 0.012, 1.2), (ex + 0.13, ey - 0.002, 1.53), "lab_steel")
    s.plane((ex, ey - 0.0125, 1.365), (0.125, 0.157), (1, 0, 0), (0, 0, 1), "lab_warning")
    s.box((ex - 0.015, ey - 0.002, 1.15), (ex + 0.015, ey + 0.01, 1.53), "lab_steel")


def _furniture_two(s, xl, lanes_x1):
    """Two rows: free floor only at the ends of the room. Operator bay (-x): workbench, cart and bins along row A's
    wall; whiteboard and extinguisher on the end wall; spool shelving and the door in row B's wall; the robot-zone
    warning post at the zone line, facing the bay. Packing end (+x): pallet, flat-packs, roll cage."""
    yb, yf, wl = WALL_BACK_Y, WALL_FRONT_Y, WALL_LEFT_X
    x0 = wl + 0.15
    x1 = x0 + 1.31
    _workbench(s, x0, x1, yb)
    _whiteboard(s, wl, 1.45)
    _shelving(_Xf(s, _rot_z(1), (wl + 1.35, yf, 0.0)), 0.0, 0.0, 1.05)        # a quarter turn: along row B's wall
    _waste_bins(s, xl, yb)
    _cart(_Xf(s, _rot_z(-1), (x1 + 0.17, yb - 0.08, 0.0)), 0.0, 0.7, 0.0, 0.45)   # parked nose-in, handle to the room
    _stool(s, (x0 + 0.55, 0.0))
    _extinguisher(_Xf(s, _rot_z(1), (DOOR_X[0] - 0.24, yf, 0.0), suffix=WALL_B), 0.0, 0.0)   # row B's wall, by the door
    _warning_post(_Xf(s, _rot_z(-1), (xl - 0.25, -0.62, 0.0)), 0.0, 0.0)       # sign + e-stop face the bay (-x)
    _furniture_right(s, lanes_x1, pallet_y=(yf + 0.05, yf + 0.85), cage_y=(0.3, 0.9))


def _furniture_right(s, lanes_x1, pallet_y=(-1.75, -0.95), cage_y=(-3.85, -3.25)):
    """Pallet with shipping boxes, flat-pack stack against the right wall, a roll cage."""
    xr = WALL_RIGHT_X
    px0, px1, (py0, py1) = lanes_x1 + 0.45, xr - 0.15, pallet_y
    for x in np.linspace(px0, px1 - 0.1, 3):
        s.box((x, py0, 0), (x + 0.1, py1, 0.1), "lab_pallet")
    s.box((px0, py0, 0.1), (px1, py1, 0.125), "lab_pallet")
    sizes = [(0.4, 0.3, 0.3), (0.3, 0.3, 0.25), (0.35, 0.25, 0.2)]
    rng = np.random.default_rng(11)
    for layer in range(3):
        x = px0 + 0.02
        z = 0.125 + sum(sizes[k][2] for k in range(layer))
        sx, sy, sz = sizes[layer]
        while x + sx <= px1 - 0.01:
            y = py0 + 0.02
            while y + sy <= py1 - 0.01:
                if layer < 2 or rng.random() < 0.6:
                    s.box((x, y, z), (x + sx - 0.006, y + sy - 0.006, z + sz), "lab_kraft")
                    s.box((x - 0.001, y + sy / 2 - 0.027, z + sz - 0.04), (x + sx - 0.005, y + sy / 2 + 0.021,
                                                                            z + sz + 0.001), "lab_kraft_tape")
                y += sy
            x += sx
    # flat-packed boxes leaning on the right wall
    for k in range(6):
        y0 = 0.05 + 0.03 * k
        s.add("box", (xr - 0.08 - 0.012 * k, y0 + 0.3, 0.5), (0.006, 0.38, 0.5), "lab_kraft",
              quat=(math.cos(0.06), 0, -math.sin(0.06), 0))
    # roll cage (wire cart) with a few parcels
    cx0, cx1, (cy0, cy1) = px0, px0 + 0.7, cage_y
    for x in (cx0, cx1):
        for y in (cy0, cy1):
            s.cyl((x, y, 0.05), (x, y, 1.6), 0.012, "lab_galv")
    for z in (0.12, 1.6):
        s.box((cx0, cy0, z - 0.01), (cx1, cy1, z + 0.01), "lab_galv")
    for x in np.linspace(cx0, cx1, 6):
        s.cyl((x, cy0, 0.12), (x, cy0, 1.6), 0.004, "lab_galv")
        s.cyl((x, cy1, 0.12), (x, cy1, 1.6), 0.004, "lab_galv")
    for k, (dx, dy, sz) in enumerate(((0.05, 0.05, 0.3), (0.38, 0.06, 0.25), (0.1, 0.32, 0.22), (0.2, 0.1, 0.2))):
        z0 = 0.13 if k < 3 else 0.43
        s.box((cx0 + dx, cy0 + dy, z0), (cx0 + dx + 0.28, cy0 + dy + 0.24, z0 + sz), "lab_cardboard")


# ---------------------------------------------------------------- entry point
def _pin_statistic(root, xml):
    """The room would grow the auto extent (12.9 -> 17.6 m), which rescales znear/zfar (camera clipping, depth
    precision of the chamber cameras) and the shadow range. Pin the undecorated values."""
    import mujoco
    st = root.find("statistic")
    if st is not None and st.get("extent") and st.get("center"):
        return
    m = mujoco.MjModel.from_xml_string(xml)
    if st is None:
        st = ET.Element("statistic")
        root.insert(list(root).index(root.find("worldbody")), st)
    st.set("extent", repr(float(m.stat.extent)))
    st.set("center", " ".join(repr(float(v)) for v in m.stat.center))


def decorate(xml: str) -> str:
    root = ET.fromstring(xml)
    _pin_statistic(root, xml)
    asset, world = root.find("asset"), root.find("worldbody")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    collide_cls = "env" if any(d.get("class") == "env" for d in root.iter("default")) else None

    bench = (_find_geoms(world, lambda g: g.get("material") == "bench") or [None])[0]
    bench_dims = None
    if bench is not None:
        bench_dims = tuple(2 * v for v in (_vec(bench.get("size"))[0], _vec(bench.get("size"))[1],
                                           _vec(bench.get("size"))[2]))
    paths = ensure_textures(bench_dims)
    _assets(asset, paths)
    _retune(asset)
    _lighting(root, world)

    s = _Sink(world, asset, collide_cls)
    printers = []
    for b in world.findall("body"):
        nm = b.get("name", "")
        if nm.startswith("printer") and nm[7:].isdigit():
            q = _vec(b.get("quat", "1 0 0 0"))
            printers.append((int(nm[7:]), _vec(b.get("pos")), -1 if abs(q[0]) < 0.5 else 1))
    xs = [p[1][0] for p in printers] or [0.0, (C.N_PER_ROW - 1) * C.PITCH]
    xmin, xmax = min(xs) - C.PRINTER_W / 2, max(xs) + C.PRINTER_W / 2
    if bench is not None:
        (bcx, bcy, bcz), (bhx, bhy, bhz) = _vec(bench.get("pos")), _vec(bench.get("size"))
        bench_top, bench_back = bcz + bhz, bcy + bhy
    else:
        bench_top, bench_back = C.BENCH_H, C.AISLE_HALF + 0.5

    s.section = "room"
    if TWO_ROWS:
        _room_two(s, xmin, xmax)
        _services(s, xmin, xmax, bench_top, bench_back)
        _services(_Xf(s, MIRROR_Y, suffix=WALL_B), xmin, xmax, bench_top, bench_back)   # row B: own tray, strip, board
    else:
        _room(s, xmin, xmax)
        _services(s, xmin, xmax, bench_top, bench_back)
    s.section = "printers"
    for i, pos, side in printers:
        _printer_dress(s, pos, side, i, collide_cls, bench_back)
        rng = np.random.default_rng(3000 + i)
        _printer_detail(s, pos, side, i, rng)
        names = list(SPOOL_COLORS)
        _wip_prints(s, i, names[np.random.default_rng(1000 + i).choice(len(names), 4, replace=False)[0]], rng)
    n_car = _carriage_dress(root, asset)
    rail = (_find_geoms(world, lambda g: g.get("material") == "rail") or [None])[0]
    belt = (_find_geoms(world, lambda g: g.get("name") == "belt") or [None])[0]
    lanes = sorted(_find_geoms(world, lambda g: (g.get("material") or "").startswith("lane")),
                   key=lambda g: g.get("material"))
    orange = _find_geoms(world, lambda g: g.get("material") == "robotorange")
    tunnel = max(orange, key=lambda g: _vec(g.get("pos"))[2]) if orange else None
    xl, xr = -1.15, 5.6
    two_stn = TWO_ROWS and any(g.get("name") == "stn_belt" for g in world.findall("geom"))
    if two_stn:
        from farm import scene
        stn = scene.station_layout()
    if rail is not None:
        s.section = "rail"
        _rail(s, rail)
        st_x0, st_hy = (C.STATION_X + 0.45, 0.16)
        if belt is not None:
            st_x0 = _vec(belt.get("pos"))[0] - _vec(belt.get("size"))[0]
            st_hy = _vec(belt.get("size"))[1] + 0.02
        s.section = "floor"
        if TWO_ROWS:
            _cable_channel(s, rail)
            bx0, bx1 = (bcx - bhx, bcx + bhx) if bench is not None else (xmin, xmax)
            if two_stn:                 # zone line just past the deck's +x edge at full travel; boxes leave on the sorter
                xl, xr = _floor_marks_two(s, rail, stn["deck_x1_max"] + C.STATION_CLEAR + 0.06 - 0.36, 0.0, bx0, bx1,
                                          gap=(stn["y_sort"] - 0.19, stn["y_sort"] + 0.19))
            else:
                xl, xr = _floor_marks_two(s, rail, st_x0, st_hy, bx0, bx1)
        else:
            xl, xr, _ = _floor_marks(s, rail, st_x0, st_hy)
    s.section = "station"
    lanes_x1 = max((_vec(g.get("pos"))[0] + _vec(g.get("size"))[0] for g in lanes), default=xr + 2.0)
    if two_stn:
        lanes_x1 = _station_two(s, stn)
    elif belt is not None:
        _station(s, belt, tunnel, lanes)
    s.section = "furniture"
    if TWO_ROWS:
        _furniture_two(s, xl, lanes_x1)
    else:
        _furniture_left(s, xl)
        lanes_x1 = max((_vec(g.get("pos"))[0] + _vec(g.get("size"))[0] for g in lanes), default=xr + 2.0)
        _furniture_right(s, lanes_x1)
    s.flush()
    STATS.update(primitives=s.n_prim, geoms=s.n_geom, meshes=s.n_mesh, carriage_geoms=n_car)
    return ET.tostring(root, encoding="unicode")
