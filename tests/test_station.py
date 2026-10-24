"""Bed to box station (two-row cell): clearance by construction, trap-door totes, kinematic boxes, station skills."""
import math

import mujoco
import numpy as np
import pytest

from farm import catalog as C

pytestmark = pytest.mark.skipif(not C.TWO_ROWS, reason="the trap-door station is the two-row cell's")
EPS = 1e-4


@pytest.fixture(scope="module")
def world():
    from farm.world import World
    return World()


def _aabb(m, d, g):
    """World AABB (lo, hi) of geom g from its local AABB (visual geoms included)."""
    R = d.geom_xmat[g].reshape(3, 3)
    c = d.geom_xpos[g] + R @ m.geom_aabb[g, :3]
    e = np.abs(R) @ m.geom_aabb[g, 3:]
    return c - e, c + e


def _carriage(w, x, flaps=None, arm=None):
    """AABBs of every carriage geom (deck, totes, flaps, tray, column, dressing, arm in TUCK) at rail position x."""
    from plan.door import TUCK
    m, d = w.m, mujoco.MjData(w.m)
    d.qpos[:] = w.d.qpos
    d.qpos[w.rail_q] = x
    d.qpos[w.arm_q] = TUCK if arm is None else arm
    for s, a in (flaps or {}).items():
        d.qpos[w.flap_q[s]] = a
    mujoco.mj_kinematics(m, d)
    return [(g, *_aabb(m, d, g)) for g in sorted(w.robot_geoms) if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE]


def _static(w):
    """(label, lo, hi) of everything that is not the carriage, a part or a box: scene geoms (printers, benches, rail,
    station) from the model, and the lab dressing primitive by primitive (it is merged into meshes in the model)."""
    from farm import lab
    m, d = w.m, w.d
    mujoco.mj_kinematics(m, d)
    skip = set(w.robot_geoms) | {g for gs in w.part_geoms.values() for g in gs}
    skip |= {g for b in w.box_body.values() for g in range(m.ngeom) if m.geom_bodyid[g] == b}
    out = []
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if g in skip or m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE or name.startswith("lab_"):
            continue
        b = m.geom_bodyid[g]
        if m.geom_group[g] == 2 and b == 0:
            continue                                      # lab primitive: counted from lab.PRIMS
        lo, hi = _aabb(m, d, g)
        out.append((f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)}/{name or g}", lo, hi))
    out += [(f"lab:{sec}", lo, hi) for lo, hi, sec in lab.PRIMS]
    return out


def _hits(car, things, x0, x1):
    """Pairs (thing, carriage geom) whose AABBs overlap anywhere while the carriage slides x0 -> x1 (car at x0)."""
    out = []
    for label, lo, hi in things:
        for g, clo, chi in car:
            slo, shi = clo.copy(), chi.copy()
            shi[0] += x1 - x0
            if np.all(lo < shi - EPS) and np.all(slo < hi - EPS):
                out.append((label, g))
    return out


@pytest.mark.parametrize("pose", ["TUCK", "LOW_TUCK"])
def test_no_geom_in_the_carriage_swept_volume(world, pose):
    """Over the whole rail range, arm in TUCK (and in the folded slide pose), flaps shut: no station, lab, printer or
    bench geom (visual ones included) intersects the carriage (deck, totes, flaps, tray, column, dressing, arm)."""
    import plan.door as D
    w = world
    w.reset(rail_x=C.RAIL_X_MIN)
    car = _carriage(w, C.RAIL_X_MIN, arm=getattr(D, pose))
    hits = _hits(car, _static(w), C.RAIL_X_MIN, C.RAIL_X_MAX)
    names = sorted({(lab, mujoco.mj_id2name(w.m, mujoco.mjtObj.mjOBJ_BODY, w.m.geom_bodyid[g])) for lab, g in hits})
    assert not names, names[:20]


def test_robot_cannot_reach_above_its_top(world):
    """C.ROBOT_TOP_Z bounds every carriage + arm geom (bounding spheres) over the joint ranges: random configurations
    and every combination of joint limits and mid-range."""
    import itertools
    w = world
    m, d = w.m, mujoco.MjData(w.m)
    adr = [list(m.jnt_qposadr).index(int(a)) for a in w.arm_q]
    lo, hi = m.jnt_range[adr, 0], m.jnt_range[adr, 1]
    geoms = np.array(sorted(w.robot_geoms))
    rng = np.random.default_rng(0)
    qs = [lo + (hi - lo) * rng.random(len(adr)) for _ in range(3000)]
    qs += [np.array(c) for c in itertools.product(*[(a, (a + b) / 2, b) for a, b in zip(lo, hi)])]
    top = 0.0
    for q in qs:
        d.qpos[w.arm_q] = q
        mujoco.mj_kinematics(m, d)
        top = max(top, float(np.max(d.geom_xpos[geoms, 2] + m.geom_rbound[geoms])))
    assert top <= C.ROBOT_TOP_Z, top


def test_station_hardware_outside_the_deck_footprint(world):
    """The station rule: hardware that reaches above the deck's underside lies beside the deck (|y| beyond its y
    extent), past its +x edge at full travel, or above the robot's reach (the camera boom); everything else stays
    below the deck."""
    from farm import lab, scene
    w = world
    L = scene.station_layout()
    x0, x1, y0, y1 = L["deck"]
    clr = C.STATION_CLEAR
    under = C.DECK_Z - scene.DECK_T - clr
    items = [(f"lab:{sec}", lo, hi) for lo, hi, sec in lab.PRIMS if sec == "station"]
    items += [(lb, lo, hi) for lb, lo, hi in _static(w) if "stn_" in lb or lb.startswith("lift_")]
    bad = [lb for lb, lo, hi in items
           if hi[2] > under and not (lo[1] >= y1 + clr - EPS or hi[1] <= y0 - clr + EPS or lo[0] >= L["deck_x1_max"] + clr - EPS
                                     or lo[2] >= C.ROBOT_TOP_Z + clr - EPS)]
    assert not bad, bad[:10]
    assert C.X_DROP - max(dx for s, (dx, _) in C.CADDY_SLOTS.items() if s in scene.flap_slots()) >= C.RAIL_X_MIN
    assert all(C.RAIL_X_MIN <= x <= C.RAIL_X_MAX for x in L["dock"].values())
    assert C.RAIL_X_MIN <= L["x_place"] <= C.RAIL_X_MAX


def _box_aabbs(w, name, pos):
    m, d = w.m, mujoco.MjData(w.m)
    d.qpos[:] = w.d.qpos
    q = w.box_q[name]
    d.qpos[q:q + 3] = np.asarray(pos) - w.box_park_pos[name]
    mujoco.mj_kinematics(m, d)
    b = w.box_body[name]
    return [(f"{name}@{np.round(pos, 3)}", *_aabb(m, d, g)) for g in range(m.ngeom) if m.geom_bodyid[g] == b
            and m.geom_rgba[g, 3] > 0 or (m.geom_bodyid[g] == b and m.geom_group[g] == 3)]


def test_boxes_on_their_route_clear_of_the_carriage(world):
    """Every box size, lowered anywhere on the conveyor / sorter / lanes or raised on the place lift, is outside the
    carriage's swept volume (the carriage may move while a box travels)."""
    from farm import scene
    w = world
    L = scene.station_layout()
    w.reset(rail_x=C.RAIL_X_MIN)
    car = _carriage(w, C.RAIL_X_MIN)
    xd, zc = L["x_drop"], L["z_conv"]
    things = []
    for s in C.BOX_TYPES:
        n = f"box{s}0"
        for y in np.linspace(L["y_sort"], L["y_disp"], 25):
            things += _box_aabbs(w, n, (xd, y, zc))
        things += _box_aabbs(w, n, (xd, L["y_place"][s], L["z_place"]))
        things += _box_aabbs(w, n, (L["mag_x"][s], L["y_mag"], zc))
        for c, lx in L["lane_x"].items():
            things += _box_aabbs(w, n, (lx, L["y_sort"], zc))
            things += _box_aabbs(w, n, (lx, L["y_lane"], L["z_lane"]))
    assert not _hits(car, things, C.RAIL_X_MIN, C.RAIL_X_MAX)


def test_raised_box_clears_the_open_flap(world):
    """Docked at each tote, a box of each size raised to the dump height stays clear of the carriage with the tote's
    flap anywhere between shut and fully open (the rim stays below the swing)."""
    from farm import scene
    w = world
    L = scene.station_layout()
    for slot, x in L["dock"].items():
        if slot == "waste":
            continue
        for s in C.BOX_TYPES:
            pos = (L["x_drop"], L["dump_y"][slot] + L["dump_dy"], L["z_dump"][s])
            box = _box_aabbs(w, f"box{s}0", pos)
            for a in np.linspace(0.0, C.FLAP_OPEN + math.radians(5), 8):
                car = _carriage(w, x, {slot: a})
                assert not _hits(car, box, x, x), (slot, s, a)


def _spawn_tote(w, slot, rng, n=6):
    from farm import scene
    dx, dy = C.CADDY_SLOTS[slot]
    x = float(w.d.qpos[w.rail_q])
    bodies = []
    for k, sku in enumerate(rng.choice(list(C.SMALL_SKUS), n)):
        b = w.free_part(str(sku))
        w.spawn(b, (x + dx + rng.uniform(-0.03, 0.03), dy + rng.uniform(-0.03, 0.03), C.DECK_Z + C.FLAP_T + 0.035 * k),
                rng.uniform(0, 6.28), stuck=False)
        bodies.append(b)
    return bodies


def test_box_carries_loose_contents(world):
    """A box moved by its prescribed joints carries loose parts along (a mocap box would not)."""
    from farm import scene
    from plan.motion import Executor
    w = world
    L = scene.station_layout()
    w.reset(rail_x=C.RAIL_X_MIN)
    ex = Executor(w)
    p0 = np.array([L["x_drop"], L["y_disp"], L["z_conv"]])
    w.box_spawn("boxM0", p0)
    b = w.free_part("cube")
    w.spawn(b, p0 + [0.0, 0.0, 0.005], 0.3, stuck=False)
    ex.wait(0.5)
    rel0 = w.part_pose(b)[0] - w.box_pos("boxM0")
    t_end = w.box_move("boxM0", [(L["x_drop"], L["y_seal"], L["z_conv"])])
    ex.wait(t_end - w.d.time + 0.3)
    rel1 = w.part_pose(b)[0] - w.box_pos("boxM0")
    assert np.linalg.norm(rel1 - rel0) < 0.005
    w.park(b)
    w.box_park("boxM0")


@pytest.fixture
def cell(world):
    """Docked at the station, arm tucked, quiet planners; returns (world, ik, executor, station)."""
    import plan.door as D
    from plan.ik import IK
    from plan.motion import Executor
    from plan.station import Station
    D.VERBOSE = False
    w = world
    w.reset(rail_x=C.RAIL_X_MAX)
    w.d.qpos[w.arm_q] = D.TUCK
    w.d.ctrl[w.act_arm] = D.TUCK
    mujoco.mj_forward(w.m, w.d)
    return w, IK(w), Executor(w), Station(w)


def test_dump_tote_into_box_and_camera_check(cell):
    """Trap door: the tote's parts end up in the box (truth), the station camera reports the tote empty from depth, and
    reports a tote with a part in it as not empty."""
    from plan.station import part_in_box
    w, ik, ex, st = cell
    st.dock(ex, ik, st.L["dock"]["o3"])
    bodies = _spawn_tote(w, "o3", np.random.default_rng(3), n=5)
    ex.wait(1.0)
    box = st.request_box(ex, "S", "o3")
    out = st.dump_tote(ex, ik, "o3", box)
    assert out["outcome"] == "dumped" and out["empty"] is True, out
    assert all(part_in_box(w, b, box) for b in bodies)
    assert w.flap_shut("o3") and abs(w.flap_angle("o3")) < math.radians(2)
    from plan.station import StationCamera
    noisy = StationCamera(w, "realistic", np.random.default_rng(5))
    noisy.commission({s: x for s, x in st.L["dock"].items() if s != "waste"})
    assert noisy.tote_empty("o3", ex.x_meas())[0] is True
    other = _spawn_tote(w, "o1", np.random.default_rng(4), n=1)
    st.dock(ex, ik, st.L["dock"]["o1"])
    ex.wait(0.5)
    assert st.cam.tote_empty("o1", ex.x_meas())[0] is False
    assert noisy.tote_empty("o1", ex.x_meas())[0] is False
    assert st.cam.tote_empty("o3", ex.x_meas())[0] is None                # not docked under the camera
    noisy.close()
    t_seal, t_lane = st.seal_and_ship(box, "post", expect=bodies)
    ex.wait(t_lane - w.d.time + 0.2)
    assert st.stats["boxes_complete"] == 1 and st.stats["parts_missing"] == 0
    assert not any(b in w.in_use for b in bodies)                     # kinematic from the sealer on: parked
    assert w.m.geom_rgba[w.box_lid[box][0], 3] == 1.0
    assert np.linalg.norm(w.box_pos(box)[:2] - [st.L["lane_x"]["post"] - 0.1, st.L["y_lane"]]) < 1e-3
    for b in other:
        w.park(b)


def _leftover(w, slot, sku, where, wedged=False):
    """One part in the docked tote: 'free_edge' (on the flap at its free edge), 'corner', '+x wall' / '-x wall'
    (against it); wedged = held fixed (welded) 6 mm above the floor, as if jammed between the walls."""
    from farm import scene
    dx, dy = C.CADDY_SLOTS[slot]
    x = float(w.d.qpos[w.rail_q])
    h = scene.TOTE_IN - 0.003 - max(C.CATALOG[sku].footprint) / 2000
    px, py = {"free_edge": (0.0, -h), "corner": (h, h), "+x wall": (h, 0.01), "-x wall": (-h, -0.01)}[where]
    b = w.free_part(sku)
    w.spawn(b, (x + dx + px, dy + py, C.DECK_Z + C.FLAP_T + (0.006 if wedged else 0.002)), 0.0, stuck=wedged)
    return b


def test_station_camera_under_realistic_noise(cell):
    """Realistic depth noise (range-squared noise, flying pixels, holes, sheen, calibration error, failed / occluded /
    smeared scans): repeated checks of an empty tote never raise an alarm, and a single leftover - the lowest-profile
    part on the flap's free edge, a small one standing in a corner or against a wall, a wedged one - is always seen."""
    from plan.station import StationCamera
    w, ik, ex, st = cell
    cam = StationCamera(w, "realistic", np.random.default_rng(11))
    cam.commission({s: x for s, x in st.L["dock"].items() if s != "waste"})
    for slot in ("o2", "o0"):                           # the centre row and an outer row (seen off-axis)
        st.dock(ex, ik, st.L["dock"][slot])
        ex.wait(0.3)
        for _ in range(6):
            assert cam.tote_empty(slot, ex.x_meas())[0] is True, cam.last
        st.cycle_flap(ex, slot, 2)                      # the trap door back on its stop after a dump
        assert cam.tote_empty(slot, ex.x_meas())[0] is True, cam.last
        for sku, where, wedged in (("lid", "free_edge", False), ("standoff", "corner", False),
                                   ("spacer", "+x wall", False), ("clip", "-x wall", True)):
            b = _leftover(w, slot, sku, where, wedged)
            ex.wait(0.5)
            assert cam.tote_empty(slot, ex.x_meas())[0] is False, (slot, sku, where, cam.last)
            w.set_stuck(b, False)
            w.park(b)
            ex.wait(0.2)
    cam.close()


def test_dump_with_a_wedged_part_goes_to_the_operator(cell):
    """A part jammed in the tote stays through the dump and the harder retry: the camera sees it both times and the
    station hands the tote to the operator; the loose parts are in the box."""
    from plan.station import part_in_box
    w, ik, ex, st = cell
    st.dock(ex, ik, st.L["dock"]["o5"])
    bodies = _spawn_tote(w, "o5", np.random.default_rng(8), n=3)
    jam = _leftover(w, "o5", "block", "+x wall", wedged=True)
    ex.wait(1.0)
    box = st.request_box(ex, "M", "o5")
    out = st.dump_tote(ex, ik, "o5", box)
    assert out["outcome"] == "operator" and out["empty"] is False and out["tries"] == 2, out
    assert out["why"] == "camera: tote not empty"
    assert all(part_in_box(w, b, box) for b in bodies)
    w.set_stuck(jam, False)
    for b in bodies + [jam]:
        w.park(b)
    w.box_park(box)


def test_place_large_part_from_the_tray(cell):
    """The arm takes a bar from its tray slot (wrist camera measures it first) and sets it on the box floor."""
    from plan.station import part_in_box
    w, ik, ex, st = cell
    sx, sy = C.LARGE_SLOTS["L0"]
    b = w.free_part("bar")
    w.spawn(b, (C.RAIL_X_MAX + sx + 0.006, sy - 0.01, C.DECK_Z + 0.004), 0.05, stuck=False)
    ex.wait(0.5)
    box = st.request_box(ex, "M", "place")
    ok, outcome, sec = st.place_large(ex, ik, "L0", box, b, "bar", (0.0, 0.0, 0.0))
    assert ok and outcome == "placed", outcome
    assert part_in_box(w, b, box)
    p, q = w.part_pose(b)
    assert abs(p[2] - st.box_floor(box)) < 0.004                      # lying flat on the floor
    assert abs(abs(math.cos(2 * math.atan2(q[3], q[0]))) - 0.0) < 0.2   # long side along world y (box x)
    w.park(b)
    w.box_park(box)


def test_place_plate_with_the_carriage_registered_high(cell):
    """The robot's model of its carriage 1.2 mm higher than the design (a registration error, as
    sensors.perturb_model draws it): the flat plate is still grasped 1.5 mm over the tray floor of that model, not
    over the design floor (where every grasp touched the modelled floor: no plan), and placed in the S box."""
    from plan.station import part_in_box
    w, ik, ex, st = cell
    cid = w.m.body("carriage").id
    ik.m.body_pos[cid][2] += 0.0012
    try:
        sx, sy = C.LARGE_SLOTS["L0"]
        b = w.free_part("plate")
        w.spawn(b, (C.RAIL_X_MAX + sx + 0.002, sy - 0.003, C.DECK_Z + 0.004), -0.04, stuck=False)
        ex.wait(0.5)
        box = st.request_box(ex, "S", "place")
        ok, outcome, sec = st.place_large(ex, ik, "L0", box, b, "plate", (0.0, 0.0, 0.0))
        assert ok and outcome == "placed", (outcome, dict(st.why))
        assert part_in_box(w, b, box)
        w.park(b)
        w.box_park(box)
    finally:
        ik.m.body_pos[cid][2] -= 0.0012
