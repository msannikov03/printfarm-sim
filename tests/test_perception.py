"""Fast tests for the chamber-camera error model and the perception correction pipeline (perceive/)."""
import math
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from farm import catalog as C
from perceive import verify as V
from perceive.noise import PRESETS, CameraNoise, Calibration, corrupt, get_noise, sample_scan

P = 4


@pytest.fixture(scope="module")
def world():
    from farm.world import World
    return World()


def _plate(w, seed, n=5):
    from jobs.plates import make_plate
    from plan.unload import mirror_plate, spawn_plate
    rng = np.random.default_rng(seed)
    xc, _ = C.printer_frame(P)
    w.reset(rail_x=xc + 0.4)
    skus = rng.choice(list(C.SMALL_SKUS), n)
    parts = [{"name": f"o{k}_{s}_{k}", "order_id": f"o{k}", "sku": str(s)} for k, s in enumerate(skus)]
    plate = mirror_plate(make_plate(parts, P, f"T{seed}", rng)[0])
    placed = spawn_plate(w, plate, rng, pos_sigma_mm=1.5, yaw_sigma_deg=2.0, stuck=True)
    mujoco.mj_forward(w.m, w.d)
    return plate, placed


def _truth(w, body):
    p, _ = w.part_pose(body)
    xc, s = C.printer_frame(P)
    return s * (p[0] - xc) * 1000 + C.BED_MM[0] / 2, (s * p[1] - C.AISLE_HALF - C.BED_FRONT_GAP) * 1000


def _errors(w, plate, placed, est):
    out = []
    for o in V.plate_objects(plate):
        e = est[o["name"]]
        tx, ty = _truth(w, placed[o["name"]])
        out.append(math.hypot(e["center_mm"][0] - tx, e["center_mm"][1] - ty))
    return np.array(out)


def _legacy_points(cam):
    """The original BedCamera.points_bed_mm, verbatim (reference for 'ideal = unchanged')."""
    m, d = cam.w.m, cam.w.d
    cam.r.update_scene(d, camera=cam.cam, scene_option=cam.opt)
    depth = cam.r.render()
    f = 0.5 * V.H / math.tan(math.radians(m.cam_fovy[cam.cam]) / 2)
    v, u = np.mgrid[0:V.H, 0:V.W]
    z = depth.astype(np.float64)
    pc = np.stack([(u - V.W / 2 + 0.5) * z / f, -(v - V.H / 2 + 0.5) * z / f, -z], -1).reshape(-1, 3)
    R = d.cam_xmat[cam.cam].reshape(3, 3)
    pw = pc @ R.T + d.cam_xpos[cam.cam]
    xc, s = C.printer_frame(cam.i)
    bx = (s * (pw[:, 0] - xc) + C.BED_MM[0] / 2000) * 1000
    by = (s * pw[:, 1] - C.AISLE_HALF - C.BED_FRONT_GAP) * 1000
    bz = (pw[:, 2] - C.BENCH_H - C.BED_ZL) * 1000
    keep = (bx > -5) & (bx < C.BED_MM[0] + 5) & (by > -5) & (by < C.BED_MM[1] + 5) & (bz < 80)
    return np.stack([bx, by, bz], -1)[keep]


def test_presets_contract():
    assert {"ideal", "realistic", "harsh"} <= set(PRESETS)
    assert PRESETS["ideal"].is_ideal and get_noise(None).is_ideal
    assert not PRESETS["realistic"].is_ideal and not PRESETS["harsh"].is_ideal
    assert PRESETS["harsh"].axial_mm > PRESETS["realistic"].axial_mm > 0
    with pytest.raises(ValueError):
        get_noise("nope")
    # axial noise grows with z^2
    n = PRESETS["realistic"]
    assert n.sigma_z_mm(0.8) == pytest.approx(4 * n.sigma_z_mm(0.4))


def test_corrupt_statistics():
    """Per-pixel axial noise has the modelled sigma; dropouts and failures behave."""
    rng = np.random.default_rng(0)
    nz = CameraNoise(axial_mm=1.0)
    depth = np.full((120, 160), 0.4, np.float32)
    sc = sample_scan(nz, rng, depth.shape)
    z = corrupt(depth, nz, rng, sc)
    assert np.nanstd(z - 0.4) * 1000 == pytest.approx(1.0, rel=0.05)
    z = corrupt(depth, CameraNoise(dropout=0.2), rng, sample_scan(CameraNoise(dropout=0.2), rng, depth.shape))
    assert np.isnan(z).mean() == pytest.approx(0.2, abs=0.02)
    for mode in ("dead", "blackout"):
        sc = sample_scan(PRESETS["realistic"], rng, depth.shape, force=mode)
        assert np.isnan(corrupt(depth, PRESETS["realistic"], rng, sc)).all()
    cal = Calibration.sample(PRESETS["ideal"], rng)
    assert np.allclose(cal.R_err, np.eye(3)) and not cal.t_err.any() and cal.f_scale == 1.0


def test_fuse_math():
    mu, cov, d2 = V._fuse(np.zeros(2), np.eye(2) * 4.0, np.array([1.0, 0.0]), np.eye(2) * 1.0)
    assert mu == pytest.approx([0.8, 0.0])                   # weight 4 / (4 + 1) on the camera
    assert np.allclose(cov, np.eye(2) * 0.8)                 # smaller than both
    assert d2 == pytest.approx(0.2)


def test_ideal_default_unchanged(world):
    w = world
    plate, _ = _plate(w, 1)
    cam = V.BedCamera(w, P)
    pts = cam.points_bed_mm()
    assert type(pts) is np.ndarray and np.array_equal(pts, _legacy_points(cam))
    objs = V.plate_objects(plate)
    est, unknown = V.estimate(pts, objs)
    ref, ref_unknown = V._estimate_legacy(pts, objs)
    assert unknown == ref_unknown
    for n, e in est.items():
        assert e["center_mm"] == ref[n]["center_mm"] and e["yaw_deg"] == ref[n]["yaw_deg"]
        assert e["found"] == ref[n]["found"] and e["n_top"] == ref[n]["n_top"]
        assert isinstance(e["sigma_mm"], float) and np.shape(e["cov_mm2"]) == (2, 2) and e["quality"] == "ok"
    cam.close()


def test_realistic_accuracy_and_contract(world):
    w = world
    cam = V.BedCamera(w, P, noise="realistic", rng=np.random.default_rng(5))
    errs = []
    for seed in (2, 3):
        plate, placed = _plate(w, seed)
        cam.resample(np.random.default_rng(seed))
        pts = cam.points_bed_mm()
        assert isinstance(pts, V.BedPoints) and pts.meta["quality"] in ("ok", "degraded")
        est, _ = V.estimate(pts, V.plate_objects(plate))
        for e in est.values():
            assert e["found"]
            assert e["quality"] in ("ok", "degraded", "failed")
            assert isinstance(e["sigma_mm"], float) and e["sigma_mm"] > 0
            cov = np.asarray(e["cov_mm2"])
            assert cov.shape == (2, 2) and np.all(np.linalg.eigvalsh(cov) > 0)
        errs.extend(_errors(w, plate, placed, est))
    assert np.mean(errs) < 1.0 and np.max(errs) < 3.0
    cam.close()


def test_noise_is_seeded(world):
    w = world
    _plate(w, 4)
    a = V.BedCamera(w, P, noise="realistic", rng=np.random.default_rng(9)).points_bed_mm()
    b = V.BedCamera(w, P, noise="realistic", rng=np.random.default_rng(9)).points_bed_mm()
    c = V.BedCamera(w, P, noise="realistic", rng=np.random.default_rng(10)).points_bed_mm()
    assert np.array_equal(np.asarray(a), np.asarray(b))
    assert len(a) != len(c) or not np.array_equal(np.asarray(a), np.asarray(c))


def test_calibration_error_is_corrected(world):
    """Only calibration error (1 deg, 3 mm, 1 % focal, 0.8 % depth scale): the plane + outline corrections remove it."""
    w = world
    plate, placed = _plate(w, 6)
    nz = CameraNoise(name="calib", ext_t_mm=3.0, ext_r_deg=1.0, f_rel=0.01, c_px=3.0, depth_scale=0.008)
    on = V.BedCamera(w, P, noise=nz, rng=np.random.default_rng(1))
    off = V.BedCamera(w, P, noise=nz, rng=np.random.default_rng(1), corrections=False)
    objs = V.plate_objects(plate)
    e_on = _errors(w, plate, placed, V.estimate(on.points_bed_mm(), objs)[0])
    e_off = _errors(w, plate, placed, V.estimate(off.points_bed_mm(), objs)[0])
    assert e_on.mean() < 1.5 and e_on.mean() < 0.5 * e_off.mean()
    on.close()
    off.close()


@pytest.mark.parametrize("mode", ["dead", "blackout"])
def test_failed_camera_gives_prior(world, mode):
    w = world
    plate, _ = _plate(w, 7)
    cam = V.BedCamera(w, P, noise="realistic", rng=np.random.default_rng(0))
    cam.force = mode
    pts = cam.points_bed_mm()
    assert len(pts) == 0 and pts.meta["quality"] == "failed"
    objs = V.plate_objects(plate)
    est, unknown = V.estimate(pts, objs)
    for o in objs:
        e = est[o["name"]]
        assert e["found"] and not e["observed"] and e["quality"] == "failed"
        assert e["center_mm"] == [float(v) for v in o["center_mm"]]
        assert e["sigma_mm"] == pytest.approx(V.PRIOR_SIGMA_MM * V.FAIL_INFLATE * math.sqrt(2))
    assert unknown == 0 and V.bed_empty(pts) is False        # cannot verify: never "empty"
    cam.close()


def test_occluded_final_scan_is_rechecked(world):
    """An empty bed seen through an occluder at the lens cannot be verified; that is not a leftover, so the unload
    scans again instead of calling a person."""
    w = world
    xc, _ = C.printer_frame(P)
    w.reset(rail_x=xc + 0.4)
    mujoco.mj_forward(w.m, w.d)
    cam = V.BedCamera(w, P, noise="realistic", rng=np.random.default_rng(5))
    cam.force = "occlusion"
    pts = cam.points_bed_mm()
    if pts.meta["quality"] != "failed" and pts.meta["near_frac"] < V.NEAR_FRAC_MAX:
        pytest.skip("this occluder draw left the bed verifiable")
    assert not V.bed_empty(pts) and V.cannot_verify(pts)
    cam.force = None
    pts = cam.points_bed_mm()
    if pts.meta["quality"] != "failed":
        assert V.bed_empty(pts) and not V.cannot_verify(pts)
    cam.close()


def test_bed_empty_robust(world):
    w = world
    xc, _ = C.printer_frame(P)
    cam = V.BedCamera(w, P, noise="realistic", rng=np.random.default_rng(3))
    for k in range(3):
        w.reset(rail_x=xc + 0.4)
        mujoco.mj_forward(w.m, w.d)
        cam.resample(np.random.default_rng(100 + k))
        pts = cam.points_bed_mm()
        if pts.meta["quality"] != "failed":
            assert V.bed_empty(pts), "noise alone must not look like a leftover"
    # a lump 5 mm proud of the bed (a cube sunk into the plate)
    b = w.free_part("cube")
    x, y, z = C.bed_mm_to_world(P, 140, 120)
    w.spawn(b, (x, y, z - (C.CATALOG["cube"].height - 5.0) / 1000), 0.3, stuck=True)
    mujoco.mj_forward(w.m, w.d)
    pts = cam.points_bed_mm()
    assert not V.bed_empty(pts)
    assert any(bl["h"] > 3.0 for bl in V.leftover_blobs(pts))
    cam.close()


def _large_plate(w, seed, skus):
    """A plate as the slicer packs it with the robot's grasp maps (large parts lie into the printer)."""
    import pathlib
    from jobs.plates import make_plate
    from plan.bedmap import load_mask
    from plan.unload import mirror_plate, spawn_plate
    rng = np.random.default_rng(seed)
    xc, _ = C.printer_frame(P)
    w.reset(rail_x=xc + 0.4)
    mask = load_mask("runs/bedmap.json", mirror=True) if pathlib.Path("runs/bedmap.json").exists() else None
    parts = [{"name": f"o{k}_{s}_{k}", "order_id": f"o{k}", "sku": s} for k, s in enumerate(skus)]
    plate = mirror_plate(make_plate(parts, P, f"L{seed}", rng, reach_mask=mask, max_print_s=8 * 3600.0)[0])
    placed = spawn_plate(w, plate, rng, pos_sigma_mm=1.5, yaw_sigma_deg=2.0, stuck=True)
    mujoco.mj_forward(w.m, w.d)
    return plate, placed


def test_large_parts_realistic_accuracy(world):
    """Long parts on a realistic camera: centre within 3 mm, yaw within 1.5 deg. (A part lying into the printer has
    its near end sampled far more densely: a median-centred gate along it used to cut the far end, ~0.5-1 mm bias.)"""
    w = world
    cam = V.BedCamera(w, P, noise="realistic", rng=np.random.default_rng(5))
    errs, yerrs = [], []
    for seed, skus in ((0, ("bar", "housing", "cube")), (1, ("plate", "bar", "clip")), (2, ("housing", "plate"))):
        plate, placed = _large_plate(w, seed, skus)
        cam.resample(np.random.default_rng(seed))
        est, _ = V.estimate(cam.points_bed_mm(), V.plate_objects(plate))
        for o in plate["objects"]:
            if not C.CATALOG[o["sku"]].large:
                continue
            e = est[o["name"]]
            assert e["found"] and e["quality"] in ("ok", "degraded")
            tx, ty = _truth(w, placed[o["name"]])
            errs.append(math.hypot(e["center_mm"][0] - tx, e["center_mm"][1] - ty))
            q = w.part_pose(placed[o["name"]])[1]
            tyaw = math.degrees(2 * math.atan2(q[3], q[0]))
            yerrs.append(abs((e["yaw_deg"] - tyaw + 90) % 180 - 90))
    assert len(errs) >= 4 and max(errs) < 3.0 and np.mean(errs) < 1.2 and max(yerrs) < 1.5
    cam.close()


@pytest.mark.parametrize("noise", ["ideal", "realistic"])
def test_missing_large_part_not_taken_from_a_neighbour(world, noise):
    """Two bars side by side, one missing: the wide (nudged-part) search must not claim the other bar's top face."""
    w = world
    plate, placed = _large_plate(w, 3, ("bar", "bar"))
    bars = [o["name"] for o in plate["objects"] if o["sku"] == "bar"]
    assert len(bars) == 2
    w.park(placed[bars[0]])
    mujoco.mj_forward(w.m, w.d)
    cam = V.BedCamera(w, P, noise=noise, rng=np.random.default_rng(4))
    est, _ = V.estimate(cam.points_bed_mm(), V.plate_objects(plate))
    assert not est[bars[0]]["found"] and est[bars[1]]["found"]
    cam.close()


def test_missing_part_detected(world):
    """A part listed by the slicer but absent from the bed is reported missing (bare bed seen where it should be)."""
    w = world
    plate, placed = _plate(w, 8)
    victim = plate["objects"][0]["name"]
    w.park(placed[victim])
    mujoco.mj_forward(w.m, w.d)
    cam = V.BedCamera(w, P, noise="realistic", rng=np.random.default_rng(2))
    est, _ = V.estimate(cam.points_bed_mm(), V.plate_objects(plate))
    assert not est[victim]["found"]
    assert all(e["found"] for n, e in est.items() if n != victim)
    cam.close()


@pytest.mark.parametrize("noise", ["ideal", "realistic"])
def test_thin_leftovers_footprint_check(world, noise):
    """A part flatter than LEFTOVER_MIN_H_MM passes leftover_blobs. The footprint check (self-referenced against the
    bed ring around it) finds a 0.8 mm part at its planned footprint, and raises no false alarm on an empty bed."""
    w = world
    xc, _ = C.printer_frame(P)
    w.reset(rail_x=xc + 0.4)
    mujoco.mj_forward(w.m, w.d)
    objs = [{"name": f"t{i}{j}", "center_mm": [x, y], "yaw_deg": 30.0 * i, "height_mm": 0.8, "footprint_mm": (40.0, 25.0)}
            for i, x in enumerate((60.0, 128.0, 196.0)) for j, y in enumerate((70.0, 128.0, 196.0))]
    checked = 0
    for seed in range(3):
        cam = V.BedCamera(w, P, noise=noise, rng=np.random.default_rng(seed))
        pts = cam.points_bed_mm()
        cam.close()
        meta = getattr(pts, "meta", None)
        if meta is not None and meta["quality"] == "failed":
            continue
        checked += 1
        assert V.bed_empty(pts) and V.thin_leftovers(pts, objs) == []
        for o in objs:                       # the part: its footprint 0.8 mm higher than the bed
            loc = (np.asarray(pts)[:, :2] - o["center_mm"]) @ V._rot2(o["yaw_deg"])
            on = (np.abs(loc[:, 0]) <= 20.0) & (np.abs(loc[:, 1]) <= 12.5)
            q = np.asarray(pts).copy() if meta is None else V.BedPoints(np.asarray(pts).copy(), meta)
            q[on, 2] += 0.8
            assert V.bed_empty(q), "0.8 mm is below leftover_blobs: the case this check exists for"
            assert V.thin_leftovers(q, objs) == [o["name"]]
    assert checked >= 2


def test_large_low_part_does_not_read_as_bed_noise(world):
    """A part lower than 4 mm can fill whole noise-grid cells; its height must not be taken for local bed noise
    (which would raise the 'raised' threshold above the part and hide it)."""
    w = world
    xc, _ = C.printer_frame(P)
    w.reset(rail_x=xc + 0.4)
    mujoco.mj_forward(w.m, w.d)
    cam = V.BedCamera(w, P, noise="realistic", rng=np.random.default_rng(3))
    pts = cam.points_bed_mm()
    cam.close()
    assert pts.meta["quality"] != "failed" and V.bed_empty(pts)
    q = np.asarray(pts).copy()
    on = (q[:, 0] > 2) & (q[:, 0] < 100) & (q[:, 1] > 100) & (q[:, 1] < 200)      # a 98 x 100 mm part ...
    q[on, 2] += 2.4                                                               # ... 2.4 mm floor ...
    rim = on & ((q[:, 0] < 6) | (q[:, 0] > 96) | (q[:, 1] < 104) | (q[:, 1] > 196))
    q[rim, 2] += 3.5                                                              # ... with 5.9 mm walls
    meta = dict(pts.meta, noise_grid=V.noise_grid(q))
    sg = np.asarray(meta["noise_grid"]["sigma"], float)
    cells = (np.arange(sg.shape[0])[:, None] * 48.0, np.arange(sg.shape[1])[None, :] * 48.0)
    inside = (cells[0] >= 2) & (cells[0] + 48 <= 100) & (cells[1] >= 100) & (cells[1] + 48 <= 200)
    assert inside.any() and np.nanmax(np.where(inside, sg, np.nan)) < 1.0, "the part's floor was taken for bed noise"
    assert not V.bed_empty(V.BedPoints(q, meta))
