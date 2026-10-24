"""Planning realism: the robot plans and checks only with what it can measure.

Fast checks (no full unload episodes): actuation presets, encoders, the belief-based planning scene, the wrist
camera's between-the-jaws test and the scan -> belief conversion.
"""
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from farm import catalog as C  # noqa: E402

P = 4


@pytest.fixture(scope="module")
def world():
    from farm.world import World
    from plan.door import TUCK
    w = World()
    xc, _ = C.printer_frame(P)
    w.reset(rail_x=xc + 0.35)
    w.d.qpos[w.arm_q] = TUCK
    w.d.ctrl[w.act_arm] = TUCK
    w.d.qpos[w.door_q[P]] = C.DOOR_OPEN_RAD
    mujoco.mj_forward(w.m, w.d)
    return w


def _fresh(world):
    from plan.ik import IK
    for n in list(world.in_use):
        world.park(n)
    mujoco.mj_forward(world.m, world.d)
    return IK(world)


def test_ideal_preset_is_the_legacy_path(world):
    from plan.motion import Executor
    ik = _fresh(world)
    ex = Executor(world, realism="ideal", ik=ik)
    assert ex.hw is None                                  # no hardware layer: commands go straight to the servos
    np.testing.assert_allclose(ex.q, world.d.ctrl[world.act_arm])
    np.testing.assert_allclose(ex.q_meas(), world.d.qpos[world.arm_q])


def test_zero_offsets_are_invisible_to_the_encoders(world):
    from farm.sensors import PRESETS
    from plan.motion import Executor
    ik = _fresh(world)
    m0 = ik.m.body_pos.copy()
    p = dict(PRESETS["realistic"], enc_noise_deg=0.0, backlash_deg=0.0)
    ex = Executor(world, realism=p, ik=ik, rng=np.random.default_rng(3))
    assert ex.hw is not None and np.any(ex.hw.zero != 0)
    assert not np.allclose(ik.m.body_pos, m0)              # the robot's kinematic model differs from the truth
    q1 = ex.q + np.array([0.05, -0.05, 0.05, 0.0, 0.05, 0.0])
    ex.move(q1, duration=0.5)
    ex.settle(tol=0.002, timeout=2.0)
    ex.wait(0.5)
    enc_err = np.abs(ex.q_meas() - q1)                     # what the robot sees: only servo sag
    true_err = np.abs(world.d.qpos[world.arm_q] - q1)      # what is true: sag + the zero offsets
    assert np.max(np.abs((world.d.qpos[world.arm_q] - ex.q_meas()) - ex.hw.zero)) < 1e-9
    assert np.max(true_err) > np.max(np.abs(ex.hw.zero)) * 0.5
    assert np.max(enc_err) < np.radians(0.5)


def test_latency_delays_commands(world):
    from farm.sensors import Actuation
    _fresh(world)
    hw = Actuation(world, dict(zero_sigma_deg=0, link_sigma_mm=0, base_sigma_mm=0, rail_zero_sigma_mm=0,
                               backlash_deg=0, latency_s=0.03, enc_noise_deg=0, grip_noise_mm=0,
                               wrist_sigma_mm=0, handeye_sigma_mm=0))
    c0 = world.d.ctrl[world.act_arm].copy()
    hw.cmd_q = hw.cmd_q + 0.1
    hw.tick(), hw.tick(), hw.tick()
    np.testing.assert_allclose(world.d.ctrl[world.act_arm], c0)      # 3 ticks = 30 ms: not there yet
    hw.tick()
    np.testing.assert_allclose(world.d.ctrl[world.act_arm], c0 + 0.1)


def test_backlash_deadband(world):
    from farm.sensors import Actuation
    _fresh(world)
    hw = Actuation(world, dict(zero_sigma_deg=0, link_sigma_mm=0, base_sigma_mm=0, rail_zero_sigma_mm=0,
                               backlash_deg=1.0, latency_s=0.0, enc_noise_deg=0, grip_noise_mm=0,
                               wrist_sigma_mm=0, handeye_sigma_mm=0))
    q0 = hw.cmd_q.copy()
    hw.cmd_q = q0 + np.radians(0.3)                        # inside the +-0.5 deg play: the joint does not move
    hw.tick()
    np.testing.assert_allclose(world.d.ctrl[world.act_arm], q0, atol=1e-12)
    hw.cmd_q = q0 + np.radians(2.0)                        # beyond it: the joint lags by half the play
    hw.tick()
    np.testing.assert_allclose(world.d.ctrl[world.act_arm], q0 + np.radians(1.5), atol=1e-12)


def _pose_at(i, x_mm, y_mm):
    pos = np.array(C.bed_mm_to_world(i, x_mm, y_mm))
    return pos, np.array([1.0, 0, 0, 0])


def test_belief_scene_hides_unseen_parts_and_pads_seen_ones(world):
    from plan.ik import Belief
    from plan.unload import plan_grasp
    ik = _fresh(world)
    b = world.free_part("cube")
    world.spawn(b, C.bed_mm_to_world(P, 150, 80), 0.0, stuck=False)
    other = world.free_part("block")
    world.spawn(other, C.bed_mm_to_world(P, 150, 115), 0.0, stuck=False)   # neighbour right behind the cube
    mujoco.mj_forward(world.m, world.d)
    bid = world.part_body[other]
    a = ik.m.jnt_qposadr[ik.m.body_jntadr[bid]]
    ik.belief = Belief()                                   # the robot has seen nothing yet
    ik.sync(world.d)
    assert all(ik.m.geom_contype[g] == 0 for g in world.part_geoms[other])
    ik.belief = Belief({other: (*_pose_at(P, 150, 115), 0.004)})
    ik.sync(world.d)
    assert all(ik.m.geom_contype[g] == 1 for g in world.part_geoms[other])
    np.testing.assert_allclose(ik.d.qpos[a:a + 3], C.bed_mm_to_world(P, 150, 115))
    assert ik.pad[bid] == pytest.approx(0.004)
    est = {"center_mm": [150.0, 80.0], "yaw_deg": 0.0}
    assert plan_grasp(ik, world, P, b, est=est) is not None
    world.park(b), world.park(other)
    ik.belief = None


def test_unexplained_blob_becomes_an_obstacle(world):
    from plan.unload import scene_belief
    ik = _fresh(world)
    pts = np.array([[x, y, 0.0] for x in np.arange(0, 256, 2.0) for y in np.arange(0, 256, 2.0)])
    box = (np.abs(pts[:, 0] - 100) < 12) & (np.abs(pts[:, 1] - 150) < 12)
    pts[box, 2] = 20.0                                     # a 24 x 24 x 20 mm lump nobody sliced
    bel = scene_belief(world, P, {}, {}, pts)
    assert len(bel.blobs) == 1 and not bel.parts
    ik.belief = bel
    ik.sync(world.d)
    assert len(ik.proxies) == 1
    (blob, bid), = ik.proxies
    g = [g for g in range(ik.m.ngeom) if ik.m.geom_bodyid[g] == bid][0]
    assert ik.m.geom_size[g][2] == pytest.approx(0.010, abs=1e-3)
    assert ik.m.geom_contype[g] == 1
    ik.belief = None
    ik.sync(world.d)                                       # proxies are restored when the belief goes away
    assert ik.m.geom_size[g][0] == pytest.approx(world.m.geom_size[g][0])


def test_wrist_camera_sees_a_part_between_the_jaws(world):
    from farm.sensors import WristCam
    from plan.unload import N_BETWEEN, plan_grasp
    ik = _fresh(world)
    b = world.free_part("cube")
    world.spawn(b, C.bed_mm_to_world(P, 150, 80), 0.3, stuck=False)
    ik.sync(world.d)
    g = plan_grasp(ik, world, P, b)
    assert g is not None
    try:
        cam = WristCam(world)
        world.d.qpos[world.arm_q], world.d.qpos[world.rail_q] = g["q"], g["x"]
        gw = g["width"] / 2 + 0.008
        world.d.qpos[world.finger_q], world.d.qpos[world.finger_rq] = gw, -gw
        mujoco.mj_forward(world.m, world.d)
        held = cam.between_jaws(2000 * gw)
        world.park(b)
        mujoco.mj_forward(world.m, world.d)
        world.d.qpos[world.arm_q] = g["q_pre"]               # empty jaws away from the bed
        mujoco.mj_forward(world.m, world.d)
        empty = cam.between_jaws(2000 * gw)
    except Exception as e:                                 # no OpenGL context on this machine
        pytest.skip(f"rendering unavailable: {e}")
    assert held >= 4 * N_BETWEEN
    assert empty < N_BETWEEN


# ---- large SKUs (bed to box): packer, grasp across the narrow side, lowered placement into the tray ----

def _parts(*skus):
    return [{"name": f"o{k}_{s}_{k}", "order_id": f"o{k}", "sku": s} for k, s in enumerate(skus)]


def test_sample_parts_small_is_the_legacy_draw():
    from jobs.plates import sample_parts
    for seed in range(5):
        a, b = np.random.default_rng(seed), np.random.default_rng(seed)
        skus = b.choice(list(C.SMALL_SKUS), int(b.integers(3, 8)))
        assert [p["sku"] for p in sample_parts(a, "small")] == [str(s) for s in skus]
        assert a.random() == b.random()                    # the stream continues identically
    assert any(C.CATALOG[p["sku"]].large for p in sample_parts(np.random.default_rng(0), "mixed"))


def test_every_large_sku_fits_a_plate_alone():
    from jobs.plates import MAX_PLATE_S, make_plate
    for s in C.LARGE_SKUS:
        assert max(C.CATALOG[s].footprint) > C.TOTE_PART_MAX_MM and min(C.CATALOG[s].footprint) <= 55
        plate, _ = make_plate(_parts(s), 4, "T", np.random.default_rng(0))
        assert [o["sku"] for o in plate["objects"]] == [s] and plate["print_time_s"] <= MAX_PLATE_S
    # a part over the cap on its own is printed alone (nothing starves), never next to another part
    plate, _ = make_plate(_parts("housing", "cube"), 4, "T", np.random.default_rng(0), max_print_s=3600.0)
    assert [o["sku"] for o in plate["objects"]] == ["housing"]


def test_packer_large_part_grasp_map_and_jaw_keepout():
    from jobs.plates import JAW_KEEPOUT_MM, make_plate

    def mask(x, y):
        return True
    mask.sku = {(s, yaw): (lambda x, y, ok=(yaw == 90): ok) for s in C.LARGE_SKUS for yaw in (0, 90)}
    plate, _ = make_plate(_parts("bar", "gear", "cube", "lid", "clip"), 4, "T", np.random.default_rng(0),
                          reach_mask=mask)
    objs = {o["sku"]: o for o in plate["objects"]}
    assert "bar" in objs and objs["bar"]["yaw_deg"] == 90.0     # only yaw 90 is in this grasp map
    rect = lambda o: (min(p[0] for p in o["polygon_mm"]), min(p[1] for p in o["polygon_mm"]),   # noqa: E731
                      max(p[0] for p in o["polygon_mm"]), max(p[1] for p in o["polygon_mm"]))
    bx0, by0, bx1, by1 = rect(objs["bar"])
    assert bx1 - bx0 == pytest.approx(C.CATALOG["bar"].dims[1])
    for s, o in objs.items():
        if s == "bar":
            continue
        x0, y0, x1, y1 = rect(o)
        if y0 < by1 and by0 < y1:                              # beside the bar: room for the open jaws
            assert max(x0 - bx1, bx0 - x1) >= C.PART_SPACING_MM + JAW_KEEPOUT_MM - 1e-9


def _maps():
    import json
    import pathlib
    path = pathlib.Path("runs/bedmap.json")
    d = json.loads(path.read_text()) if path.exists() else {}
    if "sku" not in d:
        pytest.skip("runs/bedmap.json has no per-SKU maps (python -m plan.bedmap --sku)")
    return path, d


def test_sku_maps_leave_small_only_plates_unchanged(tmp_path):
    """The per-SKU maps only bite deep in the bed: the small-only eval plates (all near the door) are exactly the
    plates the probe map alone gives (the legacy regression baseline stays comparable)."""
    import json
    from jobs.plates import make_plate, sample_parts
    from plan.bedmap import load_mask
    path, d = _maps()
    d.pop("sku")
    (tmp_path / "probe.json").write_text(json.dumps(d))
    masks = load_mask(tmp_path / "probe.json", mirror=True), load_mask(path, mirror=True)
    for seed in range(12):
        plates = []
        for m in masks:
            rng = np.random.default_rng(seed)
            plates.append(make_plate(sample_parts(rng, "small"), 4, f"PL{seed:03d}", rng, reach_mask=m)[0])
        assert plates[0] == plates[1]


def test_dense_plates_put_every_part_in_its_own_grasp_map():
    from jobs.plates import make_plate, sample_parts
    from plan.bedmap import load_mask
    from plan.unload import mirror_plate
    path, _ = _maps()
    m = load_mask(path, mirror=False)                  # checked on the mirrored plate: bed coordinates
    deep = 0
    for seed in range(10):
        rng = np.random.default_rng(seed)
        plate = mirror_plate(make_plate(sample_parts(rng, "mixed"), 4, "T", rng,
                                        reach_mask=load_mask(path, mirror=True), max_print_s=8 * 3600.0)[0])
        for o in plate["objects"]:
            assert m.sku[(o["sku"], int(o["yaw_deg"]) % 180)](*o["center_mm"]), o
            deep += o["center_mm"][1] > 100
    assert deep > 0                                    # the dense plates do reach deep into the bed


def test_long_blobs_are_not_dropped_into_the_reject_bin():
    from plan.sweep import directions_for, fits_reject_bin
    bar_into = {"lo": (100.0, 40.0), "hi": (118.0, 190.0), "rect": ((109.0, 115.0), (18.0, 150.0), 90.0)}
    bar_across = {"lo": (40.0, 60.0), "hi": (190.0, 78.0), "rect": ((115.0, 69.0), (150.0, 18.0), 0.0)}
    cube = {"lo": (40.0, 60.0), "hi": (65.0, 85.0), "rect": ((52.5, 72.5), (25.0, 25.0), 0.0)}
    assert not fits_reject_bin(bar_into) and not fits_reject_bin(bar_across) and fits_reject_bin(cube)
    assert directions_for(bar_into)[0] == "left" and directions_for(bar_into)[-1] == "front"
    assert directions_for(bar_across)[0] == "front" and directions_for(cube)[0] == "front"


def _spawn_large(world, sku, x_mm, y_mm, yaw_deg=90.0):
    b = world.free_part(sku)
    world.spawn(b, C.bed_mm_to_world(P, x_mm, y_mm), np.radians(yaw_deg), stuck=False)
    mujoco.mj_forward(world.m, world.d)
    return b, {"center_mm": [float(x_mm), float(y_mm)], "yaw_deg": float(yaw_deg)}


def test_large_part_grasped_across_its_narrow_side(world):
    from plan.unload import plan_grasp
    ik = _fresh(world)
    b, est = _spawn_large(world, "bar", 180, 130)
    try:
        g = plan_grasp(ik, world, P, b, est=est)
        assert g is not None and g["width"] == pytest.approx(C.CATALOG["bar"].dims[1] / 1000)
        assert abs(g["R"][0, 1]) > 0.95                    # jaws along world x: across the bar lying into the printer
    finally:
        world.park(b)


@pytest.mark.skipif(not C.LARGE_SLOTS, reason="one-row cell: no large-part tray")
def test_large_part_placed_flat_along_the_slot(world):
    from plan.unload import PLACE_ABOVE, part_in_slot, place_pose, plan_grasp, pose_from_estimate, slot_floor
    ik = _fresh(world)
    b, est = _spawn_large(world, "housing", 60, 110)
    g = plan_grasp(ik, world, P, b, est=est)
    assert g is not None
    pos, R = pose_from_estimate(P, est)
    ik.attach(b, world.d, rel=(g["R"].T @ (pos - g["target"]), g["R"].T @ R))     # as the robot believes it
    try:
        pl = place_pose(ik, world, "L0", g["x"], width=g["width"])
        assert pl is not None
        p, Rp = ik.fk(pl["q"], g["x"])
        _, _, rel_p, rel_R = ik.held
        part_p, part_R = p + Rp @ rel_p, Rp @ rel_R
        floor = slot_floor(ik, world, "L0", g["x"])
    finally:
        ik.detach()
    assert part_R[2, 2] > 0.999 and abs(part_R[0, 0]) > 0.999            # flat, long axis along the slot's x
    assert part_p[2] - floor[2] == pytest.approx(PLACE_ABOVE, abs=0.002)  # lowered: released 1 cm above the floor
    assert np.hypot(*(part_p[:2] - floor[:2])) < 0.002
    # released there (carriage at the planned x), the housing settles in the slot; turned across, it does not count
    x0 = float(world.d.qpos[world.rail_q])
    try:
        world.d.qpos[world.rail_q] = world.d.ctrl[world.act_rail] = g["x"]
        world.spawn(b, part_p, float(np.arctan2(part_R[1, 0], part_R[0, 0])), stuck=False)
        world.step(400)
        assert part_in_slot(world, b, "L0") and not part_in_slot(world, b, "L1")
        world.spawn(b, floor + [0, 0, 0.001], np.pi / 2, stuck=False)
        assert not part_in_slot(world, b, "L0")
    finally:
        world.park(b)
        world.d.qpos[world.rail_q] = world.d.ctrl[world.act_rail] = x0
        mujoco.mj_forward(world.m, world.d)
