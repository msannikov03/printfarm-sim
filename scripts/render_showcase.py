"""Cinematic showcase of ONE real physics run of the cell -> runs/showcase/ (mp4) and docs/media/ (README GIFs + stills).

  PYTHONPATH=. .venv/bin/python scripts/render_showcase.py                  # physics run + every output (~10 min)
  PYTHONPATH=. .venv/bin/python scripts/render_showcase.py --render-only    # re-render from runs/showcase/record.npz
  PYTHONPATH=. .venv/bin/python scripts/render_showcase.py --render-only --contact   # one still per shot (framing)
  options: --seed 3 --faults --printer 4 --fps 30 --size 1920x1080 --quick (960x540, 15 fps, no GIFs)

The run is the same pipeline as scripts/eval_unload.py (same plate generator, same fault injection with --faults): the
carriage drives in from the packaging station, opens the door, unloads the bed with the chamber camera in the loop,
rakes what it cannot pick into the reject tray, closes the door and drives back, where a kinematic bin exchanger and
the (kinematic) packaging line take the finished orders. Joint states are recorded at the 100 Hz control rate; every
camera is rendered afterwards from those states, so nothing is staged. Caption numbers come from this run; the end
card quotes the batch summaries in runs/ and names them.

Outputs: runs/showcase/showcase.mp4 (1080p), record.npz + record.json (states + events), shots.png (--contact);
docs/media/hero.png, perception.png, showcase.gif, pick.gif, rake.gif (<= 8 MB each).
"""
import argparse
import importlib.util
import json
import math
import pathlib
import subprocess
import sys
import tempfile
import time

import mujoco
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from farm import catalog as C  # noqa: E402
from farm import lab  # noqa: E402

OUT = ROOT / "runs" / "showcase"
MEDIA = ROOT / "docs" / "media"
ORDER_RGB = [(0.96, 0.42, 0.06), (0.1, 0.42, 0.88), (0.13, 0.66, 0.32), (0.98, 0.76, 0.08), (0.55, 0.27, 0.78),
             (0.05, 0.62, 0.62)]
DEBRIS_RGB = (0.2, 0.2, 0.22)
GIF_MAX_BYTES = 8 * 1000 * 1000
WRONG_MATCH_MM = 5.0             # an estimate this far from ground truth is a wrong association (scored, never fed back)
GIF_MONTAGE = ("establish", "door_open", "pick_hero", "rake", "station")
ACCENT = (242, 140, 40)


# ====================================================================================== phase 1: physics + recording
class StateLog:
    """Executor recorder: snapshot of qpos and mocap positions at every control tick (10 ms)."""

    def __init__(self):
        self.t, self.q, self.mocap = [], [], []

    def __call__(self, w):
        self.t.append(float(w.d.time))
        self.q.append(w.d.qpos.astype(np.float32))
        self.mocap.append(w.d.mocap_pos.astype(np.float32).ravel())


def _load_eval_module():
    spec = importlib.util.spec_from_file_location("eval_unload", ROOT / "scripts" / "eval_unload.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bed_mm(i, p):
    """World point -> bed mm (x, y) of printer i."""
    xc, s = C.printer_frame(i)
    return [(s * (p[0] - xc) + C.BED_MM[0] / 2000) * 1000, (s * p[1] - C.AISLE_HALF - C.BED_FRONT_GAP) * 1000]


def record(a):
    from farm.world import World
    from jobs.plates import make_plate
    import perceive.verify as V
    import plan.door as D
    import plan.sweep as S
    import plan.unload as U
    from plan.door import TUCK, operate_door, transfer
    from plan.ik import IK
    from plan.motion import Executor
    from sim.farm import LANES

    i, seed = a.printer, a.seed
    rng = np.random.default_rng(seed)
    D.VERBOSE = False
    # -- the plate: identical draws to scripts/eval_unload.py episode()
    n_parts = int(rng.integers(3, 8))
    skus = rng.choice(list(C.SMALL_SKUS), n_parts)
    parts = [{"name": f"o{k // 2}_{sku}_{k}", "order_id": f"o{k // 2}", "sku": str(sku)} for k, sku in enumerate(skus)]
    mask = None
    if (ROOT / "runs" / "bedmap.json").exists():
        from plan.bedmap import load_mask
        mask = load_mask(str(ROOT / "runs" / "bedmap.json"), mirror=True)
    plate, _ = make_plate(parts, i, f"PL{seed:03d}", rng, reach_mask=mask)
    plate = U.mirror_plate(plate)
    w = World()
    ik = IK(w)
    xc, _ = C.printer_frame(i)
    w.reset(rail_x=C.STATION_X)                        # start at the packaging station: the drive in is shot 1
    w.d.qpos[w.arm_q] = TUCK
    w.d.ctrl[w.act_arm] = TUCK
    mujoco.mj_forward(w.m, w.d)
    placed = U.spawn_plate(w, plate, rng, pos_sigma_mm=1.5, yaw_sigma_deg=2.0, stuck=False)
    faults = _load_eval_module().inject_faults(w, i, plate, placed, rng) if a.faults else {}
    w.step(200)
    log = StateLog()
    U.CAMERA_NOISE, U.CAMERA_SEED = a.realism, seed
    ex = Executor(w, log, realism=None if a.realism == "ideal" else a.realism, ik=ik,
                  rng=np.random.default_rng((seed, 7)))
    log(w)
    ev = {"labels": [], "picks": [], "scans": [], "rakes": [], "results": [], "spans": {}}

    def span(name, t0):
        ev["spans"][name] = [t0, float(w.d.time)]

    for meth in ("move", "follow", "move_path", "grip", "jaws", "wait"):     # current action label, for captions
        f = getattr(ex, meth, None)
        if f is not None:
            def wrapped(*args, _f=f, _m=meth, **kw):
                ev["labels"].append([float(w.d.time), kw.get("label") or _m])
                return _f(*args, **kw)
            setattr(ex, meth, wrapped)

    pp0 = U.pick_and_place

    def pick_and_place(*args, **kw):
        body, slot = args[3], args[4]
        t0 = float(w.d.time)
        r = pp0(*args, **kw)
        ev["picks"].append({"t0": t0, "t1": float(w.d.time), "body": body, "sku": w.part_sku[body], "slot": slot,
                            "ok": bool(r[0]), "outcome": str(r[1]), "s": float(r[2])})
        return r
    U.pick_and_place = pick_and_place

    est0 = V.estimate
    name_of = {b: n for n, b in placed.items()}

    def estimate(points, objects, *args, **kw):
        est, unknown = est0(points, objects, *args, **kw)
        rec = {"t": float(w.d.time), "unknown": int(unknown), "objects": []}
        for o in objects:
            e = est.get(o["name"], {})
            b = placed.get(o["name"])
            gt = bed_mm(i, w.part_pose(b)[0]) if b is not None and b in w.in_use else None
            rec["objects"].append({"name": o["name"], "sku": o["sku"], "prior": list(o["center_mm"]),
                                   "prior_yaw": float(o["yaw_deg"]), "height": float(o["height_mm"]),
                                   "footprint": list(o.get("footprint_mm", (30, 30))), "found": bool(e.get("found")),
                                   "est": list(e.get("center_mm", o["center_mm"])),
                                   "est_yaw": float(e.get("yaw_deg", o["yaw_deg"])), "gt": gt})
        ev["scans"].append(rec)
        return est, unknown
    V.estimate = estimate

    rake0 = S.rake

    def rake(*args, **kw):
        t0 = float(w.d.time)
        blob = args[3]
        r = rake0(*args, **kw)
        ev["rakes"].append({"t0": t0, "t1": float(w.d.time), "ok": bool(r[0]),
                            "lo": [float(v) for v in blob["lo"]], "hi": [float(v) for v in blob["hi"]]})
        return r
    S.rake = rake

    slots, free = {}, ["o0", "o1", "o2", "o3", "o4", "o5"]

    def slot_for(name):
        if name.endswith("purge_tab"):
            return "waste"
        oid = name.split("_")[0]
        if oid not in slots:
            slots[oid] = free.pop(0)
        return slots[oid]

    def on_result(name, body, slot, outcome, sec):
        ev["results"].append({"name": name, "body": body, "sku": w.part_sku[body], "slot": slot, "outcome": outcome,
                              "s": round(float(sec), 2), "t": float(w.d.time)})

    t_wall = time.time()
    try:
        t0 = float(w.d.time)
        transfer(ex, ik, TUCK, xc + 0.4)
        span("drive_in", t0)
        t0 = float(w.d.time)
        ok, dt, note = operate_door(ex, ik, i, opening=True)
        span("door_open", t0)
        ev["door_open"] = {"ok": bool(ok), "s": float(dt), "deg": float(np.degrees(w.door_angle(i)))}
        if ok:
            t0 = float(w.d.time)
            empty, unknown = U.unload_parts(ex, ik, i, plate, placed, slot_for, on_result, rng)
            span("unload", t0)
            ev["bed_empty"] = bool(empty)
        t0 = float(w.d.time)
        okc, dtc, _ = operate_door(ex, ik, i, opening=False)
        span("door_close", t0)
        ev["door_close"] = {"ok": bool(okc), "s": float(dtc), "deg": float(np.degrees(w.door_angle(i)))}
        t0 = float(w.d.time)
        transfer(ex, ik, TUCK, C.STATION_X)
        span("drive_out", t0)
    finally:
        U.pick_and_place, V.estimate, S.rake = pp0, est0, rake0
    bins = {}
    for r in ev["results"]:
        if r["outcome"] == "placed" and r["slot"] and r["slot"].startswith("o"):
            bins.setdefault(r["slot"], []).append(r["body"])
    if C.TWO_ROWS:                                     # -- station: bed to box (plan/station.py), physical up to sealing
        from plan.station import Station
        st = Station(w, noise=a.realism, rng=np.random.default_rng((seed, 11)))
        t_arr, seals, t_end = float(w.d.time), [], float(w.d.time)
        for k, (slot, bodies) in enumerate(sorted(bins.items())):
            box = st.request_box(ex, "S", slot, wait=False)
            st.dock(ex, ik, st.L["dock"][slot])
            st.wait_box(ex, box)
            out = st.dump_tote(ex, ik, slot, box)
            t_seal, t_lane = st.seal_and_ship(box, list(LANES)[k % 3], expect=bodies)
            seals.append([box, float(t_seal), out["outcome"]])
            t_end = max(t_end, t_lane)
        if any(r["slot"] == "waste" for r in ev["results"]):
            st.dump_waste(ex, ik)
        transfer(ex, ik, TUCK, st.L["x_place"])
        ex.wait(max(t_end - float(w.d.time) + 1.0, 0.5))
        ev["station"] = {"t_arrive": t_arr, "seals": seals, "stats": dict(st.stats)}
        span("station", t_arr)
        bins = {}
    # -- one row: kinematic bin exchanger (parts lifted from each order bin onto the belt) + packaging line
    lane_y = list(LANES.values())
    jobs, t_arr = [], float(w.d.time)
    x0 = C.STATION_X + 0.45
    for k, (slot, bodies) in enumerate(sorted(bins.items())):
        jobs.append({"slot": slot, "bodies": bodies, "t0": t_arr + 0.6 + 2.2 * k, "pkg": k, "lane": lane_y[k % 3],
                     "starts": {}})
    T_LIFT, T_BELT, T_PACK, T_LANE = 1.3, 2.4, 1.6, 2.2

    def station_hook(w_):
        t = w_.d.time
        for j in jobs:
            s = t - j["t0"]
            if s < 0:
                continue
            if s < T_LIFT:
                f = 0.5 - 0.5 * math.cos(math.pi * s / T_LIFT)
                for k, b in enumerate(j["bodies"]):
                    if b not in j["starts"]:
                        j["starts"][b] = w_.part_pose(b)[0].copy()
                        w_.m.body_gravcomp[w_.part_body[b]] = 1.0      # carried kinematically from here on
                        for g in w_.part_geoms[b]:
                            w_.m.geom_contype[g] = w_.m.geom_conaffinity[g] = 0
                    p0 = j["starts"][b]
                    p1 = np.array([x0 + 0.02 * k, 0.0, 0.625])
                    p = (1 - f) * p0 + f * p1
                    p[2] += 0.18 * math.sin(math.pi * f)
                    qa = w_.part_q[b]
                    w_.d.qpos[qa:qa + 3] = p
                    w_.d.qvel[w_.part_v[b]:w_.part_v[b] + 6] = 0
                continue
            for b in j["bodies"]:
                if b in w_.in_use:
                    w_.park(b)
            mid = w_.m.body_mocapid[w_.m.body(f"pkg{j['pkg']}").id]
            s -= T_LIFT
            if s < T_BELT:
                pos = [x0 + 0.85 * s / T_BELT, 0.0, 0.67]
            elif s < T_BELT + T_PACK:
                pos = [x0 + 0.85, 0.0, 0.67]
            elif s < T_BELT + T_PACK + T_LANE:
                f = (s - T_BELT - T_PACK) / T_LANE
                pos = [x0 + 0.85 + 0.9 * f, j["lane"] * f, 0.67 - 0.1 * f]
            else:
                pos = [x0 + 1.75, j["lane"], 0.57]
            w_.d.mocap_pos[mid] = pos

    if not C.TWO_ROWS:
        w.hooks = [station_hook]
        t_end = (jobs[-1]["t0"] if jobs else t_arr) + T_LIFT + T_BELT + T_PACK + T_LANE + 1.5
        ex.wait(t_end - float(w.d.time))
        w.hooks = []
        ev["station"] = {"t_arrive": t_arr, "jobs": [{k: v for k, v in j.items() if k != "starts"} for j in jobs],
                         "T": [T_LIFT, T_BELT, T_PACK, T_LANE]}
        span("station", t_arr)
    meta = {"seed": seed, "printer": i, "faults": faults, "plate": plate, "placed": placed, "orders": slots,
            "events": ev, "wall_s": time.time() - t_wall, "sim_s": float(w.d.time),
            "pool_skus": {n: s for n, s in w.part_sku.items()}}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "record.npz", t=np.array(log.t), q=np.array(log.q), mocap=np.array(log.mocap))
    (OUT / "record.json").write_text(json.dumps(meta, indent=1, default=float))
    placed_n = sum(r["outcome"] == "placed" for r in ev["results"])
    print(f"recorded {len(log.t)} states, {meta['sim_s']:.1f} s sim in {meta['wall_s']:.0f} s wall; "
          f"placed {placed_n}/{len(ev['results'])}, outcomes {[r['outcome'] for r in ev['results']]}, "
          f"door {ev.get('door_open', {}).get('ok')}/{ev.get('door_close', {}).get('ok')}, "
          f"bed empty {ev.get('bed_empty')}, rakes {[r['ok'] for r in ev['rakes']]}")
    return meta


# ====================================================================================== phase 2: rendering helpers
def _font_path(weight):
    cands = {"bold": [("/System/Library/Fonts/Avenir Next.ttc", 0), ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0)],
             "demi": [("/System/Library/Fonts/Avenir Next.ttc", 2), ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 0)],
             "regular": [("/System/Library/Fonts/Avenir Next.ttc", 5), ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0)]}
    for p, idx in cands[weight]:
        if pathlib.Path(p).exists():
            return p, idx
    return None, 0


_FONTS = {}


def font(size, weight="regular"):
    from PIL import ImageFont
    key = (size, weight)
    if key not in _FONTS:
        p, idx = _font_path(weight)
        try:
            _FONTS[key] = ImageFont.truetype(p, size, index=idx) if p else ImageFont.load_default(size=size)
        except Exception:
            _FONTS[key] = ImageFont.load_default()
    return _FONTS[key]


class Text:
    """Draw captions on an RGB frame (PIL, alpha-composited panels)."""

    def __init__(self, img):
        from PIL import Image, ImageDraw
        self.im = Image.fromarray(img).convert("RGBA")
        self.layer = Image.new("RGBA", self.im.size, (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.layer)
        self.s = self.im.size[1] / 1080          # layout is designed at 1080p

    def px(self, v):
        return int(round(v * self.s))

    def panel(self, box, fill=(14, 16, 20, 190), r=10):
        self.d.rounded_rectangle(tuple(self.px(v) for v in box), radius=self.px(r), fill=fill)

    def text(self, xy, s, size, weight="regular", fill=(255, 255, 255, 255), anchor="la"):
        self.d.text((self.px(xy[0]), self.px(xy[1])), s, font=font(self.px(size), weight), fill=fill, anchor=anchor)

    def width(self, s, size, weight="regular"):
        return self.d.textlength(s, font=font(self.px(size), weight)) / self.s

    def done(self):
        from PIL import Image
        return np.asarray(Image.alpha_composite(self.im, self.layer).convert("RGB"))


def lower_third(img, title, detail="", alpha=1.0):
    if alpha <= 0:
        return img
    t = Text(img)
    a = int(255 * alpha)
    w_ = max(t.width(title, 38, "demi"), t.width(detail, 26)) + 70
    t.panel((48, 880, 48 + w_, 880 + (128 if detail else 84)), fill=(14, 16, 20, int(200 * alpha)))
    t.panel((48, 880, 56, 880 + (128 if detail else 84)), fill=(*ACCENT, a), r=3)
    t.text((80, 896), title, 38, "demi", (255, 255, 255, a))
    if detail:
        t.text((80, 952), detail, 26, "regular", (205, 210, 218, a))
    return t.done()


def badge(img, speed, t_sim):
    t = Text(img)
    s = "real time" if abs(speed - 1) < 0.05 else f"{speed:.1f}x"
    t.panel((1690, 36, 1880, 104), fill=(14, 16, 20, 170))
    t.text((1785, 44), s, 28, "demi", (255, 255, 255, 255), anchor="ma")
    t.text((1785, 78), f"sim t = {t_sim:5.1f} s", 18, "regular", (190, 195, 205, 255), anchor="ma")
    t.panel((36, 36, 60 + t.width("PrintPick ", 30, "bold") + t.width("MuJoCo physics, one recorded run", 20), 92),
            fill=(14, 16, 20, 170))
    t.text((48, 44), "PrintPick", 30, "bold", (255, 255, 255, 235))
    t.text((48 + t.width("PrintPick ", 30, "bold"), 52), "MuJoCo physics, one recorded run", 20, "regular",
           (225, 228, 235, 220))
    return t.done()


def depth_to_height(depth, m, d, cam_id, i):
    """Per-pixel height above printer i's bed (mm) and an inside-bed mask, from a chamber depth image (as perceive/)."""
    H, W = depth.shape
    f = 0.5 * H / math.tan(math.radians(m.cam_fovy[cam_id]) / 2)
    v, u = np.mgrid[0:H, 0:W]
    z = depth.astype(np.float64)
    pc = np.stack([(u - W / 2 + 0.5) * z / f, -(v - H / 2 + 0.5) * z / f, -z], -1)
    pw = pc @ d.cam_xmat[cam_id].reshape(3, 3).T + d.cam_xpos[cam_id]
    xc, s = C.printer_frame(i)
    bx = (s * (pw[..., 0] - xc) + C.BED_MM[0] / 2000) * 1000
    by = (s * pw[..., 1] - C.AISLE_HALF - C.BED_FRONT_GAP) * 1000
    bz = (pw[..., 2] - C.BENCH_H - C.BED_ZL) * 1000
    inside = (bx > -5) & (bx < C.BED_MM[0] + 5) & (by > -5) & (by < C.BED_MM[1] + 5) & (bz < 80) & (bz > -20)
    return bz, inside


def project(m, d, cam_id, pts_w, W, H):
    f = 0.5 * H / math.tan(math.radians(m.cam_fovy[cam_id]) / 2)
    R = d.cam_xmat[cam_id].reshape(3, 3)
    pc = (np.asarray(pts_w) - d.cam_xpos[cam_id]) @ R
    return np.stack([W / 2 - 0.5 + f * pc[:, 0] / -pc[:, 2], H / 2 - 0.5 - f * pc[:, 1] / -pc[:, 2]], 1)


def footprint(o, center, yaw, z_mm, i, n=40):
    """World points of a part's footprint outline at height z (bed mm) for drawing."""
    k = C.CATALOG.get(o["sku"])
    if k is not None and k.kind == "cyl":
        th = np.linspace(0, 2 * np.pi, n)
        loc = np.stack([np.cos(th), np.sin(th)], 1) * k.dims[0] / 2
    else:
        fx, fy = o["footprint"]
        loc = np.array([[-fx, -fy], [fx, -fy], [fx, fy], [-fx, fy], [-fx, -fy]]) / 2
    c, s_ = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    xy = loc @ np.array([[c, s_], [-s_, c]]) + np.asarray(center)
    return np.array([C.bed_mm_to_world(i, x, y, z_mm) for x, y in xy])


def perception_panel(m, d, i, scan, W=880, H=660, prog=1.0):
    """Two views of one scan: chamber RGB with slicer prior (dashed white) vs camera estimate (green) footprints, and
    the depth heat-map (height above the bed) with unexplained raised blobs boxed in red. Returns (left, right, stats)."""
    import cv2
    from plan.sweep import blobs
    cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, f"chamber{i}")
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[[0, 1, 2]] = 1                       # exactly what the perception camera sees
    r = mujoco.Renderer(m, height=480, width=640)      # the sensor: 640 x 480 depth
    r.enable_depth_rendering()
    r.update_scene(d, camera=cam_id, scene_option=opt)
    depth = r.render().copy()
    r.close()
    r = mujoco.Renderer(m, height=960, width=1280)     # RGB at 2x for display only
    r.update_scene(d, camera=cam_id, scene_option=opt)
    rgb = r.render().copy()
    r.close()
    # zoom both panels onto the bed (the camera also sees the chamber walls)
    corners = [C.bed_mm_to_world(i, x, y, z) for x in (0, C.BED_MM[0]) for y in (0, C.BED_MM[1]) for z in (0, 30)]
    cp = project(m, d, cam_id, corners, 640, 480)
    lo, hi = cp.min(0), cp.max(0)
    c, half = (lo + hi) / 2, (hi - lo) / 2 * 1.08
    half = np.maximum(half, [half[1] * W / H, half[0] * H / W])
    half = np.minimum(half, [320, 240])
    c = np.clip(c, half, [640, 480] - half)
    x0, y0 = c - half
    x1, y1 = c + half
    sx, sy = W / (x1 - x0), H / (y1 - y0)

    def tf(P):
        return (np.asarray(P) - [x0, y0]) * [sx, sy]
    hz, inside = depth_to_height(depth, m, d, cam_id, i)
    # heat-map: turbo over 0..40 mm above the bed, dimmed outside the bed
    norm = np.clip(hz / 40.0, 0, 1)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1].copy()
    heat[~inside] = (heat[~inside] * 0.18 + 20).astype(np.uint8)
    heat[inside & (hz < 1.5)] = (heat[inside & (hz < 1.5)] * 0.55).astype(np.uint8)
    left = cv2.resize(rgb[int(2 * y0):int(2 * y1), int(2 * x0):int(2 * x1)], (W, H), interpolation=cv2.INTER_AREA)
    right = cv2.resize(heat[int(y0):int(y1), int(x0):int(x1)], (W, H), interpolation=cv2.INTER_NEAREST)
    errs, wrong, missing = [], 0, 0
    for o in scan["objects"]:
        if o["sku"] == "purge_tab":
            o = dict(o, footprint=list(C.PURGE_TAB_MM[:2]))
        z = o["height"]
        P = tf(project(m, d, cam_id, footprint(o, o["prior"], o["prior_yaw"], z, i), 640, 480))
        _dashed(left, P, (255, 255, 255), 2)
        err = math.hypot(o["est"][0] - o["gt"][0], o["est"][1] - o["gt"][1]) if o["gt"] is not None else None
        bad = o["found"] and err is not None and err > WRONG_MATCH_MM      # ground truth: scoring only
        if o["found"] and not bad and err is not None:
            errs.append(err)
        wrong += bad
        missing += not o["found"]
        if prog > 0.35 and o["found"]:
            col = (255, 150, 30) if bad else (60, 235, 120)
            E = tf(project(m, d, cam_id, footprint(o, o["est"], o["est_yaw"], z, i), 640, 480))
            cv2.polylines(left, [E.astype(np.int32)], True, col, 2, cv2.LINE_AA)
            cv2.polylines(right, [E.astype(np.int32)], True, (255, 255, 255), 2, cv2.LINE_AA)
            if bad:
                _label(left, "wrong match", (E[:, 0].max() + 8, E[:, 1].mean() - 10), col)
        elif not o["found"] and prog > 0.35:
            c = tf(project(m, d, cam_id, [C.bed_mm_to_world(i, *o["prior"], z)], 640, 480))[0]
            _label(left, "missing", (c[0] - 40, c[1] - 30), (255, 90, 90))
    # unexplained raised points -> blobs (what the rake fallback acts on); same back-projection as perceive/
    bz = hz[inside]
    v, u = np.nonzero(inside)
    xc, s = C.printer_frame(i)
    f = 0.5 * depth.shape[0] / math.tan(math.radians(m.cam_fovy[cam_id]) / 2)
    zc = depth[v, u].astype(np.float64)
    pc = np.stack([(u - 320 + 0.5) * zc / f, -(v - 240 + 0.5) * zc / f, -zc], -1)
    pw = pc @ d.cam_xmat[cam_id].reshape(3, 3).T + d.cam_xpos[cam_id]
    bx = (s * (pw[:, 0] - xc) + C.BED_MM[0] / 2000) * 1000
    by = (s * pw[:, 1] - C.AISLE_HALF - C.BED_FRONT_GAP) * 1000
    pts = np.stack([bx, by, bz], -1)
    found = [o for o in scan["objects"] if o["found"]]
    n_unexpl = 0
    for b in blobs(pts):
        cx, cy = (b["lo"] + b["hi"]) / 2
        if any(math.hypot(cx - o["est"][0], cy - o["est"][1]) < 0.5 * math.hypot(*o["footprint"]) + 6 for o in found):
            continue
        n_unexpl += 1
        if prog > 0.6:
            lo, hi = b["lo"] - 4, b["hi"] + 4
            box = np.array([C.bed_mm_to_world(i, x, y, 0) for x, y in ((lo[0], lo[1]), (hi[0], lo[1]), (hi[0], hi[1]),
                                                                     (lo[0], hi[1]))])
            Q = (tf(project(m, d, cam_id, box, 640, 480))).astype(np.int32)
            for img in (left, right):
                cv2.polylines(img, [Q], True, (255, 60, 60), 3, cv2.LINE_AA)
            _label(left, "unknown: rake", (Q[:, 0].min(), Q[:, 1].min() - 34), (255, 90, 90))
    stats = {"mean_err_mm": float(np.mean(errs)) if errs else float("nan"), "max_err_mm": float(np.max(errs)) if errs else
             float("nan"), "n_ok": len(errs), "n_wrong_match": int(wrong), "n_missing": int(missing),
             "n_objects": len(scan["objects"]), "n_unknown_blobs": n_unexpl, "unknown_pts": scan["unknown"]}
    return left, right, stats


def _label(img, text, xy, col):
    import cv2
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    x, y = int(np.clip(xy[0], 2, img.shape[1] - tw - 12)), int(np.clip(xy[1], 2, img.shape[0] - th - 12))
    cv2.rectangle(img, (x, y), (x + tw + 10, y + th + 12), (16, 18, 22), -1)
    cv2.putText(img, text, (x + 5, y + th + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2, cv2.LINE_AA)


def _dashed(img, P, col, th, dash=10):
    import cv2
    P = np.asarray(P, float)
    for a, b in zip(P[:-1], P[1:]):
        L = np.linalg.norm(b - a)
        n = max(int(L / dash), 1)
        for k in range(0, n, 2):
            p, q = a + (b - a) * k / n, a + (b - a) * min(k + 1, n) / n
            cv2.line(img, tuple(np.round(p).astype(int)), tuple(np.round(q).astype(int)), col, th, cv2.LINE_AA)
    if len(P) > 2 and np.linalg.norm(P[0] - P[-1]) > 1:
        _dashed(img, [P[-1], P[0]], col, th, dash)


def perception_card(left, right, stats, W=1920, H=1080, title=True):
    t = Text(np.full((H, W, 3), (16, 18, 22), np.uint8))
    if title:
        t.text((80, 60), "Perception: verify the slicer's plan with the chamber depth camera", 40, "demi")
        t.text((80, 116), "Every part's pose is known from the slicer; the camera only refines it and finds what does "
                          "not belong on the bed.", 24, "regular", (190, 195, 205, 255))
    img = t.done().copy()
    y0 = 190
    img[y0:y0 + left.shape[0], 70:70 + left.shape[1]] = left
    img[y0:y0 + right.shape[0], W - 70 - right.shape[1]:W - 70] = right
    t = Text(img)
    yl = y0 + left.shape[0] + 24
    t.text((70, yl), "chamber camera", 24, "demi")
    t.d.line([(t.px(330), t.px(yl + 16)), (t.px(370), t.px(yl + 16))], fill=(255, 255, 255, 255), width=3)
    t.text((380, yl), "slicer prior", 22, "regular", (225, 228, 235, 255))
    t.d.line([(t.px(540), t.px(yl + 16)), (t.px(580), t.px(yl + 16))], fill=(60, 235, 120, 255), width=4)
    t.text((590, yl), "camera estimate", 22, "regular", (225, 228, 235, 255))
    xr = W - 70 - right.shape[1]
    t.text((xr, yl), "depth: height above the bed, 0 to 40 mm", 24, "demi")
    t.d.line([(t.px(xr + 520), t.px(yl + 16)), (t.px(xr + 560), t.px(yl + 16))], fill=(255, 60, 60, 255), width=4)
    t.text((xr + 570, yl), "unexplained blob", 22, "regular", (225, 228, 235, 255))
    msg = (f"{stats['n_ok']}/{stats['n_objects']} slicer objects confirmed, centre error {stats['mean_err_mm']:.2f} mm mean, "
           f"{stats['max_err_mm']:.2f} mm max (vs ground truth);  {stats['n_unknown_blobs']} unexplained blob(s) -> rake")
    t.text((70, yl + 56), msg, 26, "regular", (240, 240, 240, 255))
    if stats["n_wrong_match"] or stats["n_missing"]:
        t.text((70, yl + 96), f"{stats['n_wrong_match']} estimate(s) matched to the wrong part (orange, > {WRONG_MATCH_MM:.0f} mm "
                              f"from ground truth), {stats['n_missing']} missing; parts that stay on the bed "
                              "are left to the rake fallback", 22, "regular", (255, 190, 120, 255))
    return t.done()


END_SOURCES = {"eval": "runs/eval", "faults": "runs/eval_faults", "des": "runs/des"}   # set from the CLI


def end_card(W=1920, H=1080):
    """Measured numbers from the batch summaries in runs/ (each tile names its source)."""
    def load(p):
        try:
            return json.loads((ROOT / p).read_text())
        except Exception:
            return None
    src = END_SOURCES
    ev, fa = load(f"{src['eval']}/summary.json"), load(f"{src['faults']}/summary.json")
    des = load(f"{src['des']}/kpis_compare.json")
    tiles = []
    if ev:
        c = ev["counts"]
        tiles.append((f"{c.get('outcome:placed', 0)}/{c.get('parts', 0)}", "parts placed in order bins",
                      f"{c.get('episodes', 0)} random plates, sim  ({src['eval']})"))
        tiles.append((f"{c.get('door_open_ok', 0)}/{c.get('episodes', 0)}", "doors opened and closed",
                      "one handle, whole-body swing"))
        tiles.append((f"{ev['mean_s_per_placed_part']:.1f} s", "per part, door to bin", f"simulated time ({src['eval']})"))
    if fa:
        c = fa["counts"]
        failed = c.get("parts", 0) - c.get("outcome:placed", 0)
        if failed:
            tiles.append((f"{100 * (failed - c.get('blocking', 0)) / failed:.0f} %", "of unpicked parts cleared off the bed",
                          f"1 tipped part + 1 debris lump per bed  ({src['faults']})"))
    if des and "human" in des.get("runs", {}):
        rob = des["runs"].get("order_aware", next(iter(des["runs"].values())))["mean"]
        hum = des["runs"]["human"]["mean"]
        tiles.append((f"{rob['printer_util_pct']:.0f} % vs {hum['printer_util_pct']:.0f} %",
                      "printer utilisation, robot vs manual", f"{des.get('days', 7):.0f}-day fleet model, assumed demand"))
        tiles.append((f"{rob['lead_time_p90_h']:.1f} h vs {hum['lead_time_p90_h']:.1f} h", "order lead time p90",
                      "same fleet model  (runs/des)"))
    t = Text(np.full((H, W, 3), (16, 18, 22), np.uint8))
    t.text((W / 2, 92), "PrintPick", 64, "bold", anchor="ma")
    t.text((W / 2, 178), "a simulated robot cell that unloads a 3D-print farm", 30, "regular", (200, 205, 212, 255),
           anchor="ma")
    cols, tw, th = 3, 520, 230
    x0 = (W - cols * tw - (cols - 1) * 40) / 2
    for k, (big, lab_, src) in enumerate(tiles[:6]):
        x, y = x0 + (k % cols) * (tw + 40), 290 + (k // cols) * (th + 40)
        t.panel((x, y, x + tw, y + th), fill=(30, 33, 40, 255), r=14)
        t.panel((x, y, x + 8, y + th), fill=(*ACCENT, 255), r=3)
        t.text((x + 40, y + 34), big, 56 if len(big) < 12 else 46, "bold")
        t.text((x + 40, y + 120), lab_, 26, "demi", (235, 238, 242, 255))
        t.text((x + 40, y + 168), src, 19, "regular", (160, 166, 176, 255))
    t.text((W / 2, H - 92), "MuJoCo 3 physics  ·  YAM arm on a linear rail  ·  grasps from slicer geometry, no learned "
                            "policy  ·  packaging line kinematic", 22, "regular", (160, 166, 176, 255), anchor="ma")
    return t.done()


def title_card(img, alpha):
    t = Text(img)
    H = img.shape[0] / t.s
    t.panel((0, H - 400, 1920, H), fill=(10, 12, 15, int(170 * alpha)), r=0)
    a = int(255 * alpha)
    t.text((96, H - 330), "PrintPick", 96, "bold", (255, 255, 255, a))
    t.text((100, H - 200), "One arm on a rail unloads a farm of enclosed 3D printers:", 34, "demi", (235, 238, 242, a))
    t.text((100, H - 150), "opens the door, checks the bed with the chamber camera, picks every part into its order bin,",
           28, "regular", (205, 210, 218, a))
    t.text((100, H - 112), "rakes what it cannot pick into a reject tray, closes the door.  Simulated end to end in MuJoCo.",
           28, "regular", (205, 210, 218, a))
    return t.done()


# ====================================================================================== phase 2: the cut
class Player:
    """Renders frames from the recorded states."""

    def __init__(self, a, meta, rec):
        from farm.render import Cam
        from farm.world import World
        self.meta, self.i = meta, meta["printer"]
        self.T, self.Q, self.M = rec["t"], rec["q"], rec["mocap"]
        self.w = World()
        m = self.m = self.w.m
        m.vis.quality.shadowsize = a.shadow
        self.W, self.H = a.size
        self.cam = Cam(m, self.W, self.H, groups=(0, 1, 2, 4, 5))
        self.wrist = Cam(m, 640, 360, groups=(0, 1, 2, 4, 5))
        self.d = self.w.d
        # set dressing: hide the unloaded printer's prints-in-progress (its bed holds the real parts)
        for g in range(m.ngeom):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if nm.startswith("lab_batch_") and nm.endswith(f"_wip{self.i}"):
                m.geom_group[g] = 3
        # part colours: one filament colour per order; the fault debris lump looks like a failed print
        orders = sorted({n.split("_")[0] for n in meta["placed"] if not n.endswith("purge_tab")})
        for n, b in meta["placed"].items():
            if n.endswith("purge_tab"):
                continue
            rgb = ORDER_RGB[orders.index(n.split("_")[0]) % len(ORDER_RGB)]
            for g in self.w.part_geoms[b]:
                m.geom_rgba[g] = (*rgb, 1)
        deb = (meta.get("faults") or {}).get("debris")
        if deb:
            for g in self.w.part_geoms[deb["body"]]:
                m.geom_rgba[g] = (*DEBRIS_RGB, 1)
        self.status_mat = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, f"lab_status{j}") for j in range(C.N_PRINTERS)]
        sp = meta["events"]["spans"]
        self.t_unload0 = sp.get("door_open", [0, 0])[0]
        self.t_unload1 = sp.get("door_close", [0, 0])[1]

    def at(self, t):
        k = int(np.clip(np.searchsorted(self.T, t), 0, len(self.T) - 1))
        self.d.qpos[:] = self.Q[k]
        self.d.mocap_pos[:] = self.M[k].reshape(-1, 3)
        for box, ts, _ in self.meta["events"].get("station", {}).get("seals", []):   # lids (render-only state)
            self.w.box_seal(box, t >= ts)
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_camlight(self.m, self.d)
        for j, mid in enumerate(self.status_mat):
            if mid < 0:
                continue
            st = "printing"
            if j == self.i:
                st = "ready" if t < self.t_unload0 else "unloading" if t < self.t_unload1 else "printing"
            self.m.mat_rgba[mid] = lab.STATUS_RGBA[st]
        return self

    def rail_x(self):
        return float(self.d.qpos[self.w.rail_q])

    def view(self, v):
        return self.cam.look(*v).rgb(self.d)

    def wrist_img(self):
        return self.wrist.rgb(self.d, "wrist")


def inset(img, small, label, s=1.0):
    import cv2
    H, W = img.shape[:2]
    w_, h_ = int(480 * s), int(270 * s)
    sm = cv2.resize(small, (w_, h_), interpolation=cv2.INTER_AREA)
    x1, y1 = W - int(48 * s), H - int(60 * s)
    x0, y0 = x1 - w_, y1 - h_
    out = img.copy()
    out[y0 - 3:y1 + 3, x0 - 3:x1 + 3] = (240, 240, 240)
    out[y0:y1, x0:x1] = sm
    t = Text(out)
    t.panel((x0 / t.s, (y0 / t.s) - 40, x0 / t.s + t.width(label, 20, "demi") + 24, y0 / t.s - 4),
            fill=(14, 16, 20, 200), r=6)
    t.text((x0 / t.s + 12, y0 / t.s - 36), label, 20, "demi")
    return t.done()


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def lerp(a, b, u):
    return [x + (y - x) * u for x, y in zip(a, b)] if isinstance(a, (list, tuple)) else a + (b - a) * u


def build_shots(meta, P):
    """The cut: [(name, t0, t1, seconds_on_screen, camera(u, t) -> view, caption(u, t) -> (title, detail), opts)]."""
    ev = meta["events"]
    sp = ev["spans"]
    i = meta["printer"]
    xc, _ = C.printer_frame(i)
    picks = ev["picks"]
    shots = []
    zbed = C.BENCH_H + C.BED_ZL
    ybed = C.AISLE_HALF + C.BED_FRONT_GAP + C.BED_MM[1] / 2000
    rail_len = C.RAIL_X_MAX - C.RAIL_X_MIN

    def follow_cam(u, t):
        P.at(t)
        x = P.rail_x()
        return ([x - 0.15, 0.12, 0.85], lerp(2.9, 2.35, ease(u)), lerp(150, 126, ease(u)), lerp(-15, -13, u))

    t0 = sp["drive_in"][0]
    shots.append(dict(name="title", t0=t0, t1=t0, sec=4.0, cam=lambda u, t: ([2.4, 0.3, 0.95], lerp(5.6, 5.0, u),
                                                                              lerp(116, 110, u), -15), title=True))
    d0, d1 = sp["drive_in"]
    shots.append(dict(name="establish", t0=d0, t1=d1, sec=6.5, cam=follow_cam,
                      cap=("10 enclosed printers, one arm on a rail",
                           f"YAM 6-DOF arm on a {rail_len:.1f} m linear axis; bins for 6 orders ride on the carriage")))
    o0, o1 = sp["door_open"]
    do = ev.get("door_open", {})
    shots.append(dict(name="door_open", t0=o0, t1=o1, sec=7.0,
                      cam=lambda u, t: ([xc - 0.1, 0.08, 0.9], lerp(1.95, 1.55, ease(u)), lerp(130, 118, ease(u)), -12),
                      cap=("Opening the door: one everyday handle, one whole-body swing",
                           f"arm + rail move together; the arm folds under the door.  This run: "
                           f"{do.get('deg', 0):.1f} deg in {do.get('s', 0):.1f} s")))
    if ev["scans"]:
        sc = ev["scans"][0]
        shots.append(dict(name="perception", t0=sc["t"], t1=sc["t"], sec=7.0, perception=0))
    into = lambda u, t: ([xc + 0.02, ybed - 0.03, zbed + 0.06], lerp(0.92, 0.82, ease(u)), lerp(116, 108, ease(u)), -28)
    into2 = lambda u, t: ([xc + 0.02, ybed - 0.03, zbed + 0.06], 0.8, lerp(132, 124, ease(u)), -22)
    bins = lambda u, t: (P.at(t) and [P.rail_x() + 0.05, -0.12, 0.72], 1.45, lerp(62, 52, u), -38)
    for k, p in enumerate(picks):
        cap = (f"Pick {k + 1} of {len(picks)}:  {p['sku'].replace('_', ' ')}  ->  "
               f"{'waste bin' if p['slot'] == 'waste' else 'ORDER ' + str(int(p['slot'][1]) + 1) if p['slot'] and p['slot'].startswith('o') else str(p['slot'])}",
               f"{'placed' if p['ok'] else p['outcome']} in {p['s']:.1f} s;  grasp planned from the slicer's geometry "
               f"and the camera's pose estimate, no learned policy")
        if k == 1:
            shots.append(dict(name="pick_hero", t0=p["t0"], t1=p["t1"], sec=12.0, cam=into, cap=cap, wrist=True))
        else:
            shots.append(dict(name=f"pick{k}", t0=p["t0"], t1=p["t1"], sec=max(3.2, (p["t1"] - p["t0"]) / 5.0),
                              cam=bins if k % 3 == 2 else into2, cap=cap, wrist=k % 3 != 2))
    for k, r in enumerate(ev["rakes"]):
        if not r["ok"]:
            continue
        tray_cam = lambda u, t: ([xc + 0.02, C.AISLE_HALF + 0.08, C.BENCH_H + 0.12], lerp(0.85, 0.75, ease(u)),
                                 lerp(122, 114, u), -20)
        f = meta.get("faults") or {}
        what = "a tipped part and a lump of debris" if f.get("tipped") and f.get("debris") else "what it cannot pick"
        first = not any(s_["name"].startswith("rake") for s_ in shots)
        shots.append(dict(name="rake" if first else f"rake{k}", t0=r["t0"] - 0.5, t1=r["t1"],
                          sec=max(5.0, (r["t1"] - r["t0"]) / (1.6 if first else 3.5)), cam=tray_cam, wrist=False,
                          cap=("No grasp?  Rake it into the reject tray",
                               f"camera blobs for {what} (injected faults): pulled off the bed into the tray, reprinted; "
                               "the printer is free again")))
    c0, c1 = sp["door_close"]
    dc = ev.get("door_close", {})
    res = ev["results"]
    n_ok = sum(r["outcome"] == "placed" for r in res)
    n_sw = sum(r["outcome"] == "swept" for r in res)
    shots.append(dict(name="door_close", t0=c0, t1=c1, sec=5.0,
                      cam=lambda u, t: ([xc - 0.1, 0.08, 0.9], 1.75, lerp(112, 124, u), -13),
                      cap=("Bed clear, door closed: the next print starts",
                           f"this plate: {n_ok} placed, {n_sw} raked to the tray;  bed empty: "
                           f"{'yes' if ev.get('bed_empty') else 'no'};  door closed to {dc.get('deg', 0):.2f} deg")))
    s0 = sp["drive_out"][0]
    s1 = sp["station"][1]
    sx = C.STATION_X
    if C.TWO_ROWS:                                    # bed to box: the box under the tote, the drop, the sealer
        xd = C.X_DROP
        shots.append(dict(name="station", t0=max(s0, sp["drive_out"][1] - 2.5), t1=s1, sec=10.0,
                          cam=lambda u, t: ([xd - 0.02, 0.1, 0.42], lerp(1.15, 1.0, ease(u)), lerp(200, 212, u),
                                            -14),                    # low, past the deck's end: under the overhang
                          cap=("Finished packages go straight into their box",
                               "each tote's trap door drops its parts into the requested box; a depth camera checks "
                               "the tote is empty; sealer and carrier lanes are kinematic")))
    else:
        shots.append(dict(name="station", t0=max(s0, sp["drive_out"][1] - 2.5), t1=s1, sec=8.0,
                          cam=lambda u, t: ([sx + 0.7, 0.0, 0.6], lerp(2.6, 2.3, ease(u)), lerp(100, 92, u), -36),
                          cap=("Finished orders go to packing",
                               "bins handed off at the station; the packaging line and carrier lanes are kinematic")))
    shots.append(dict(name="end", t0=s1, t1=s1, sec=6.0, end=True))
    return shots


def render(a, meta, rec):
    import cv2
    import imageio.v2 as imageio
    P = Player(a, meta, rec)
    shots = build_shots(meta, P)
    fps = a.fps
    OUT.mkdir(parents=True, exist_ok=True)
    MEDIA.mkdir(parents=True, exist_ok=True)
    if a.contact:
        return contact_sheet(P, shots)
    stats = {}
    perc_cards = {}
    for sh in shots:                               # pre-render the perception panels (frozen state)
        if "perception" in sh:
            P.at(sh["t0"])
            left, right, st = perception_panel(P.m, P.d, P.i, meta["events"]["scans"][sh["perception"]])
            perc_cards[sh["name"]] = (left, right, st)
            stats["perception"] = st
            if not a.quick:
                im = perception_card(left, right, st)
                imageio.imwrite(MEDIA / "perception.png", im)
    writer = imageio.get_writer(str(OUT / "showcase.mp4"), fps=fps, codec="libx264", quality=None, pixelformat="yuv420p",
                                ffmpeg_params=["-crf", "20", "-preset", "slow", "-movflags", "+faststart"],
                                macro_block_size=8)
    gif_frames = {"pick": [], "rake": [], "showcase": []}
    prev = None
    n_frames, t_render = 0, time.time()
    fade = int(0.35 * fps)
    for sh in shots:
        n = max(int(round(sh["sec"] * fps)), 1)
        speed = (sh["t1"] - sh["t0"]) / sh["sec"]
        for k in range(n):
            u = k / max(n - 1, 1)
            t = sh["t0"] + (sh["t1"] - sh["t0"]) * u
            if sh.get("end"):
                if "_cache" not in sh:
                    sh["_cache"] = cv2.resize(end_card(), (P.W, P.H), interpolation=cv2.INTER_AREA)
                img = sh["_cache"]
            elif "perception" in sh:
                left, right, st = perc_cards[sh["name"]]
                if k in (0, int(0.36 * n), int(0.61 * n)):
                    P.at(sh["t0"])
                    left, right, st = perception_panel(P.m, P.d, P.i, meta["events"]["scans"][sh["perception"]],
                                                       prog=u + 0.01)
                    sh["_cache"] = perception_card(left, right, st)
                    if P.W != 1920:
                        sh["_cache"] = cv2.resize(sh["_cache"], (P.W, P.H), interpolation=cv2.INTER_AREA)
                img = sh["_cache"]
            else:
                P.at(t)
                img = P.view(sh["cam"](u, t))
                if sh.get("title"):
                    img = title_card(img, min(1.0, 4 * u) if u < 0.85 else max(0.0, (1 - u) / 0.15))
                else:
                    if sh.get("wrist"):
                        img = inset(img, P.wrist_img(), "wrist camera", P.H / 1080)
                    if "cap" in sh:
                        al = min(1.0, u * n / (0.4 * fps)) if k < 0.4 * fps else 1.0
                        img = lower_third(img, *sh["cap"], alpha=al)
                    img = badge(img, speed, t)
            if prev is not None and k < fade:
                al = (k + 1) / (fade + 1)
                img = (img.astype(np.float32) * al + prev.astype(np.float32) * (1 - al)).astype(np.uint8)
            writer.append_data(img)
            n_frames += 1
            name = sh["name"]
            g12, g10 = max(1, round(fps / 12)), max(1, round(fps / 10))
            sm = lambda: cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
            if name == "pick_hero" and k % g12 == 0:
                gif_frames["pick"].append(sm())
            if name == "rake" and k % g12 == 0:
                gif_frames["rake"].append(sm())
            if name in GIF_MONTAGE and k % g10 == 0 and abs(k - n // 2) <= 1.3 * fps:
                gif_frames["showcase"].append(sm())
        prev = img
    writer.close()
    print(f"showcase.mp4: {n_frames} frames ({n_frames / fps:.1f} s at {fps} fps, {P.W}x{P.H}) rendered in "
          f"{time.time() - t_render:.0f} s; {(OUT / 'showcase.mp4').stat().st_size / 1e6:.1f} MB")
    if not a.quick:
        hero(P, meta, a)
        gifs(gif_frames, fps)
    (OUT / "render_stats.json").write_text(json.dumps({"frames": n_frames, "fps": fps, "size": [P.W, P.H],
                                                       "shots": [{k: v for k, v in s.items() if k in ("name", "t0", "t1", "sec")}
                                                                 for s in shots], **stats}, indent=1))
    return stats


def hero(P, meta, a):
    """4K render downsampled to 1080p (supersampling), mid-pick, door open, the whole row behind."""
    import cv2
    from farm.render import Cam
    import imageio.v2 as imageio
    ev = meta["events"]
    picks = ev["picks"]
    p = picks[min(1, len(picks) - 1)] if picks else None
    t = p["t0"] + 0.35 * (p["t1"] - p["t0"]) if p else ev["spans"]["door_open"][1]
    P.at(t)
    xc, _ = C.printer_frame(P.i)
    big = Cam(P.m, 3840, 2160, groups=(0, 1, 2, 4, 5))
    img = big.look([xc + 0.35, 0.22, 0.98], 2.35, 132, -16).rgb(P.d)
    big.close()
    img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
    imageio.imwrite(MEDIA / "hero.png", img)
    print(f"hero.png at sim t = {t:.1f} s")


def _write_gif(frames, path, fps, width):
    import imageio.v2 as imageio
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory() as td:
        src = pathlib.Path(td) / "src.mp4"
        wr = imageio.get_writer(str(src), fps=fps, codec="libx264", quality=None, pixelformat="yuv444p",
                                ffmpeg_params=["-crf", "12"], macro_block_size=8)
        for f in frames:
            wr.append_data(f)
        wr.close()
        vf = (f"scale={width}:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors=160:stats_mode=diff[p];"
              f"[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(src), "-vf", vf, "-loop", "0", str(path)], check=True)
    return path.stat().st_size


def gifs(gf, fps):
    """README GIFs <= 8 MB (frames already subsampled while rendering); shrink the width until under the limit."""
    out = {}
    specs = {"pick": (gf["pick"], fps / max(1, round(fps / 12)), 8.0),
             "rake": (gf["rake"], fps / max(1, round(fps / 12)), 8.0),
             "showcase": (gf["showcase"], fps / max(1, round(fps / 10)), 14.0)}   # middle 2.6 s of each highlight
    for name, (frames, gfps, max_s) in specs.items():
        if not frames:
            continue
        sel = frames[:int(max_s * gfps)]
        for width in (640, 560, 480, 400):
            size = _write_gif(sel, MEDIA / f"{name}.gif", gfps, width)
            if size <= GIF_MAX_BYTES:
                break
        out[name] = {"frames": len(sel), "fps": round(gfps, 2), "width": width, "MB": round(size / 1e6, 2)}
        print(f"{name}.gif: {len(sel)} frames @ {gfps:.1f} fps, {width} px wide, {size / 1e6:.2f} MB")
    (OUT / "gifs.json").write_text(json.dumps(out, indent=1))


def contact_sheet(P, shots):
    import cv2
    import imageio.v2 as imageio
    tiles = []
    for sh in shots:
        if sh.get("end") or "perception" in sh:
            continue
        for u in (0.15, 0.5, 0.85):
            t = sh["t0"] + (sh["t1"] - sh["t0"]) * u
            P.at(t)
            img = P.view(sh["cam"](u, t))
            if sh.get("wrist"):
                img = inset(img, P.wrist_img(), "wrist camera", P.H / 1080)
            img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)
            cv2.putText(img, f"{sh['name']} u={u:.2f} t={t:.1f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            tiles.append(img)
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[k:k + 3], 1) for k in range(0, len(tiles), 3)]
    imageio.imwrite(OUT / "shots.png", np.concatenate(rows, 0))
    print("wrote", OUT / "shots.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--realism", default="realistic", choices=["ideal", "realistic", "harsh"],
                    help="sensor + actuation realism of the recorded run (same presets as eval_unload)")
    ap.add_argument("--eval", default="runs/eval", help="batch shown on the end card (summary.json)")
    ap.add_argument("--eval-faults", default="runs/eval_faults", help="fault batch shown on the end card")
    ap.add_argument("--des", default="runs/des", help="fleet-model output shown on the end card")
    ap.add_argument("--printer", type=int, default=4)
    ap.add_argument("--faults", action=argparse.BooleanOptionalAction, default=True,
                    help="inject one tipped part + one debris lump (scripts/eval_unload.py inject_faults)")
    ap.add_argument("--render-only", action="store_true")
    ap.add_argument("--record-only", action="store_true")
    ap.add_argument("--contact", action="store_true", help="one still per shot -> runs/showcase/shots.png")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--shadow", type=int, default=8192, help="shadow map size (render only)")
    ap.add_argument("--out", default=str(OUT), help="video, recording and stats directory")
    ap.add_argument("--media", default=str(MEDIA), help="README stills and GIFs directory")
    a = ap.parse_args()
    globals().update(OUT=pathlib.Path(a.out), MEDIA=pathlib.Path(a.media))
    END_SOURCES.update(eval=a.eval, faults=a.eval_faults, des=a.des)
    a.size = tuple(int(v) for v in a.size.split("x"))
    if a.quick:
        a.size, a.fps, a.shadow = (960, 540), 15, 4096
    if a.render_only:
        meta = json.loads((OUT / "record.json").read_text())
    else:
        meta = record(a)
    if a.record_only:
        return
    z = np.load(OUT / "record.npz")
    rec = {"t": z["t"], "q": z["q"], "mocap": z["mocap"]}
    render(a, meta, rec)


if __name__ == "__main__":
    main()
