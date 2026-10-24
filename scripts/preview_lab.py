"""Preview the lab dressing (farm.lab.decorate) and check it leaves physics/planning untouched.

  PYTHONPATH=. .venv/bin/python scripts/preview_lab.py [--raw] [--views overview,station] [--no-timing]
  ... --fingerprint fp.json                 # save the physics fingerprint of the model World() builds
  ... --compare fp.json --checks-only       # fail unless bodies/joints/actuators/collidable geoms are identical
  PRINTFARM_LAYOUT=two_rows ... --out runs/lab_two_rows --prefix two_rows_   # the two-row cell (its own views)

- model counts (nq, nv, nu, neq, njnt, nbody, nmocap) must be identical with and without the dressing, and so must body
  masses/inertias, joints, actuators and equalities (the carriage dressing is density 0);
- --compare: the collidable-geom set (by body, shape, pose and contact parameters) must match a saved fingerprint;
- chamber-camera depth (perception input) must be identical;
- renders runs/lab_preview_<view>.png (and runs/lab_preview_raw_<view>.png with --raw);
- reports render time per 1280x720 frame and mj_step time, before/after.
"""
import argparse
import json
import pathlib
import time

import mujoco
import numpy as np

from farm import catalog as C
from farm import lab, scene
from farm.render import Cam, save
from farm.world import World

try:
    from plan.door import TUCK
except Exception:                       # planning stack not importable: any compact pose will do
    from farm.world import HOME as TUCK

OUT = pathlib.Path("runs")
RAIL_X = 2.3
VIEWS = {   # name: (lookat, dist, azimuth, elevation); az 90 looks along +y (aisle -> printers)
    "overview": ([2.75, -0.9, 0.45], 10.0, 78, -30),
    "aisle": ([3.2, 0.12, 0.85], 4.6, 1, -10),
    "printer": ([1.36, 0.45, 1.1], 1.3, 68, -14),
    "station": ([6.5, -0.05, 0.75], 3.3, 112, -22),
    "behind_robot": ([RAIL_X, 0.45, 1.0], 2.4, 90, -16),
    "video": ([RAIL_X + 0.1, 0.05, 0.85], 2.6, 140, -24),     # scripts/run_farm.py time-lapse camera
    "lab_left": ([-2.0, -0.6, 0.95], 4.2, 118, -18),
}
if C.N_ROWS == 2:   # PRINTFARM_LAYOUT=two_rows: facing rows, rail in the middle; farm.lab's two-row room
    _XC = (C.N_PER_ROW - 1) * C.PITCH / 2
    VIEWS = {
        "aisle_end": ([_XC + 0.53, 0.0, 1.0], _XC + 0.53 - lab.WALL_LEFT_X - 1.3, 0, -8),  # from the operator bay
        "overview": ([_XC + 0.43, 0.0, 0.3], 9.6, 58, -40),       # 3/4 overview, row B's wall cut away
        "overview_end": ([_XC - 0.4, 0.45, 0.4], 9.0, 11, -36),   # high view down the room from behind the bay
        "carriage": ([RAIL_X + 0.15, 0.0, 0.75], 2.2, 5, -34),    # the carriage between the rows
        "station_end": ([C.STATION_X + 0.3, 0.0, 0.9], 3.6, 180, -12),   # packing end, looking back up the aisle
        "station": ([C.X_DROP + 0.1, -0.1, 0.45], 2.6, 135, -28),       # bed-to-box station from the -y / +x corner
        "operator_bay": ([-2.7, 0.0, 0.9], 2.9, 180, -16),        # the bay at the -x end, from the aisle mouth
        "row_b": ([RAIL_X - 0.3, -0.5, 0.95], 0.85, 290, -10),    # row B fronts from across the aisle
    }
CUTAWAY = {"overview"} if C.N_ROWS == 2 else set()   # rendered with lab.cutaway(): row B's wall items hidden
COUNTS = ("nq", "nv", "nu", "na", "neq", "njnt", "nbody", "nmocap", "nsite", "ncam", "nsensor")


def physics_fingerprint(m):
    """Everything the dressing must not change: counts, body mass/inertia, joints, actuators, equalities and the set of
    collidable geoms (identified by body + shape + pose + contact parameters, not by geom id, which the dressing
    shifts). Returns a JSON-able dict."""
    r = lambda a: np.round(np.asarray(a, float), 9).tolist()
    name = lambda t, i: mujoco.mj_id2name(m, t, i) or f"#{i}"
    B = mujoco.mjtObj.mjOBJ_BODY
    coll = []
    for g in range(m.ngeom):
        if m.geom_contype[g] or m.geom_conaffinity[g]:
            coll.append([name(B, m.geom_bodyid[g]), int(m.geom_type[g]), r(m.geom_size[g]), r(m.geom_pos[g]),
                         r(m.geom_quat[g]), int(m.geom_contype[g]), int(m.geom_conaffinity[g]), int(m.geom_condim[g]),
                         r(m.geom_friction[g]), r(m.geom_solref[g]), r(m.geom_solimp[g]), r(m.geom_margin[g]),
                         int(m.geom_priority[g]), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""])
    coll.sort(key=lambda c: json.dumps(c))
    return {
        "counts": {k: int(getattr(m, k)) for k in COUNTS},
        "opt": r([m.opt.timestep, m.opt.impratio, m.opt.noslip_iterations, m.opt.integrator, m.opt.cone]),
        "stat": r([m.stat.extent, *m.stat.center]),
        "bodies": [[name(B, b), r(m.body_pos[b]), r(m.body_quat[b]), r(m.body_mass[b]), r(m.body_inertia[b]),
                    r(m.body_ipos[b]), r(m.body_iquat[b]), r(m.body_gravcomp[b])] for b in range(m.nbody)],
        "joints": [[name(mujoco.mjtObj.mjOBJ_JOINT, j), int(m.jnt_type[j]), r(m.jnt_axis[j]), r(m.jnt_pos[j]),
                    r(m.jnt_range[j])] for j in range(m.njnt)],
        "dofs": r(np.stack([m.dof_damping, m.dof_armature, m.dof_frictionloss], 1)),
        "actuators": r(np.concatenate([m.actuator_gainprm[:, :3], m.actuator_biasprm[:, :3], m.actuator_ctrlrange,
                                       m.actuator_forcerange], 1)),
        "eq": r(m.eq_data), "collidable_geoms": coll,
    }


def compare_fingerprints(a, b):
    """-> list of differing keys (collidable geoms: added/removed entries)."""
    bad = [k for k in a if k != "collidable_geoms" and a[k] != b.get(k)]
    ca, cb = {json.dumps(c) for c in a["collidable_geoms"]}, {json.dumps(c) for c in b["collidable_geoms"]}
    if ca != cb:
        bad.append(f"collidable_geoms: -{len(ca - cb)} +{len(cb - ca)}")
    return bad


def make_world(xml):
    w = World(model=mujoco.MjModel.from_xml_string(xml))
    w.reset(rail_x=RAIL_X)
    w.d.qpos[w.arm_q] = TUCK
    w.d.ctrl[w.act_arm] = TUCK
    w.d.qpos[w.door_q[5]] = C.DOOR_OPEN_RAD          # one open door for the close-ups
    if C.N_ROWS == 2:
        w.d.qpos[w.door_q[C.N_PER_ROW + 5]] = C.DOOR_OPEN_RAD   # and the facing one in row B
    for k, (sku, (bx, by)) in enumerate((("gear", (90, 120)), ("knob", (150, 90)), ("block", (120, 180)))):
        name = f"{sku}_0"
        w.spawn(name, C.bed_mm_to_world(5, bx, by), 0.3 * k)
    if C.N_ROWS == 2:                                # boxes at the station: place lift, sealer, a lane
        from farm import scene
        L = scene.station_layout()
        w.box_spawn("boxM0", (L["x_drop"], L["y_place"]["M"], L["z_place"]))
        w.box_spawn("boxS0", (L["x_drop"], L["y_seal"], L["z_conv"]))
        w.box_seal("boxS0")
        w.box_spawn("boxL0", (L["lane_x"]["courier_A"] - 0.1, L["y_lane"], L["z_lane"]))
        w.box_seal("boxL0")
        w._lift_tops()
    else:
        for k in range(3):                           # parcels on the line
            mid = w.m.body_mocapid[w.m.body(f"pkg{k}").id]
            w.d.mocap_pos[mid] = [C.STATION_X + 0.45 + 0.3 * k, 0.0, 0.67]
    mujoco.mj_forward(w.m, w.d)
    return w


def chamber_points(w, i):
    """Perception input: chamber-camera depth back-projected and cropped to the bed (perceive/verify.py)."""
    from perceive.verify import BedCamera
    cam = BedCamera(w, i)
    pts = cam.points_bed_mm()
    cam.close()
    return pts


def render_all(w, prefix, views, timing):
    cam = Cam(w.m)
    OUT.mkdir(parents=True, exist_ok=True)
    times = []
    for name in views:
        lab.cutaway(w.m, name in CUTAWAY)            # two rows: hide row B's wall items for the outside view
        cam.look(*VIEWS[name]).rgb(w.d)              # warm-up (texture upload, shadow maps)
        t0 = time.perf_counter()
        n = 4 if timing else 1
        for _ in range(n):
            img = cam.look(*VIEWS[name]).rgb(w.d)
        times.append((time.perf_counter() - t0) / n)
        save(img, OUT / f"{prefix}{name}.png")
    cam.r.close()
    return float(np.mean(times))


def step_time(w, n=400):
    d0 = mujoco.MjData(w.m)
    d0.qpos[:], d0.ctrl[:], d0.mocap_pos[:] = w.d.qpos, w.d.ctrl, w.d.mocap_pos
    t0 = time.perf_counter()
    for _ in range(n):
        mujoco.mj_step(w.m, d0)
    return (time.perf_counter() - t0) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="store_true", help="also render the undecorated scene")
    ap.add_argument("--views", default=",".join(VIEWS))
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--fingerprint", help="write the decorated model's physics fingerprint (JSON) here")
    ap.add_argument("--compare", help="fail if the physics fingerprint differs from this JSON (e.g. one from 268bf2e)")
    ap.add_argument("--checks-only", action="store_true", help="no renders")
    ap.add_argument("--out", default="runs", help="directory for the renders")
    ap.add_argument("--prefix", default="lab_preview_", help="render file name prefix")
    a = ap.parse_args()
    global OUT
    OUT = pathlib.Path(a.out)
    views = a.views.split(",")
    OUT.mkdir(parents=True, exist_ok=True)

    raw = scene.build_xml()
    t0 = time.perf_counter()
    dec = lab.decorate(raw)
    t_dec = time.perf_counter() - t0
    w0, w1 = make_world(raw), make_world(dec)
    m0, m1 = w0.m, w1.m
    fp = physics_fingerprint(scene.build())          # exactly what World() builds
    if a.fingerprint:
        pathlib.Path(a.fingerprint).write_text(json.dumps(fp))
        print(f"fingerprint -> {a.fingerprint}: {len(fp['collidable_geoms'])} collidable geoms, {len(fp['bodies'])} bodies")
    if a.compare:
        bad = compare_fingerprints(json.loads(pathlib.Path(a.compare).read_text()), fp)
        print(f"physics fingerprint vs {a.compare}: {'IDENTICAL' if not bad else 'DIFFERS: ' + ', '.join(bad)} "
              f"({len(fp['collidable_geoms'])} collidable geoms)")
        assert not bad, bad
    # the dressing itself adds no physics: undecorated vs decorated agree except the dressing's own collision proxies
    f0 = physics_fingerprint(m0)
    bad = [k for k in compare_fingerprints(f0, physics_fingerprint(m1)) if not k.startswith("collidable_geoms")]
    assert not bad, f"decorate() changed {bad}"

    print(f"decorate(): {t_dec * 1000:.0f} ms, {lab.STATS['primitives']} primitives -> {lab.STATS['geoms']} geoms "
          f"({lab.STATS['meshes']} batched meshes)")
    print(f"  ngeom {m0.ngeom} -> {m1.ngeom} (+{m1.ngeom - m0.ngeom}), "
          f"lights {m0.nlight} -> {m1.nlight}, textures {m0.ntex} -> {m1.ntex}, materials {m0.nmat} -> {m1.nmat}")
    for k in COUNTS:
        v0, v1 = getattr(m0, k), getattr(m1, k)
        print(f"  {k:8s} {v0:6d} -> {v1:6d}")
        assert v0 == v1, f"{k} changed: {v0} -> {v1}"
    print(f"  stat.extent {m0.stat.extent:.3f} -> {m1.stat.extent:.3f}, center {np.round(m0.stat.center, 3)} -> "
          f"{np.round(m1.stat.center, 3)}")
    assert m1.nlight + 1 <= lab.MAX_LIGHTS, "GL renderer drops lights past 8 (headlight included)"
    # every added geom except the AMS proxies is visual-only; the proxies must sit behind the door plane
    n_coll = 0
    for g in range(m1.ngeom):
        if m1.geom_bodyid[g] == 0 and (m1.geom_contype[g] or m1.geom_conaffinity[g]):
            n_coll += 1
    print(f"  collidable worldbody geoms {sum(1 for g in range(m0.ngeom) if m0.geom_bodyid[g] == 0 and (m0.geom_contype[g] or m0.geom_conaffinity[g]))} -> {n_coll}")
    mujoco.mj_forward(m1, w1.d)
    for g in range(m1.ngeom):
        if m1.geom_bodyid[g] == 0 and m1.geom_contype[g] and m1.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX \
                and m1.geom_group[g] == 3:
            front = abs(w1.d.geom_xpos[g][1]) - m1.geom_size[g][1]     # row B (y < 0) mirrors row A
            assert front >= C.AISLE_HALF + 0.06 - 1e-6, f"AMS proxy {g} front at y={front:.3f}"
    # perception input unchanged (the cameras also see out through the door glass; those points are cropped)
    for i in (0, 4, 5, 9) + ((C.N_PER_ROW, C.N_PER_ROW + 5, C.N_PRINTERS - 1) if C.N_ROWS == 2 else ()):
        p0, p1 = chamber_points(w0, i), chamber_points(w1, i)
        diff = float(np.abs(p0 - p1).max()) if p0.shape == p1.shape else float("inf")
        print(f"  chamber{i}: {len(p0)} -> {len(p1)} bed points, max |diff| = {diff:.2e} mm")
        assert diff < 1e-3, "dressing changes what the chamber camera sees on the bed"
    # contacts at rest unchanged
    print(f"  ncon at rest {w0.d.ncon} -> {w1.d.ncon}")

    if a.checks_only:
        return
    if not a.no_timing:
        s0, s1 = step_time(w0), step_time(w1)
        print(f"mj_step: {s0 * 1e3:.3f} -> {s1 * 1e3:.3f} ms ({s1 / s0:.2f}x)")
    if a.raw:
        t_raw = render_all(w0, a.prefix + "raw_", views, not a.no_timing)
    t_lab = render_all(w1, a.prefix, views, not a.no_timing)
    if a.raw:
        print(f"render 1280x720: {t_raw * 1e3:.1f} -> {t_lab * 1e3:.1f} ms/frame ({t_lab / t_raw:.2f}x)")
    else:
        print(f"render 1280x720: {t_lab * 1e3:.1f} ms/frame")
    print("wrote", ", ".join(str(OUT / f"{a.prefix}{v}.png") for v in views))


if __name__ == "__main__":
    main()
