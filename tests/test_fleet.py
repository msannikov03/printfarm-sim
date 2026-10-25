"""Fleet / logistics invariants: plate packing + G-code round trip, caddy allocator, DES conservation."""
import itertools
import numpy as np
import pytest

from farm.catalog import (BED_MM, BOX_TYPES, CATALOG, LARGE_SKUS, LARGE_SLOTS, PART_SPACING_MM, PURGE_STRIP_MM,
                          SMALL_SKUS, print_time_s)
from fleet.allocator import SLOT_COST_ORDER, CaddyAllocator
from fleet.des import Params, Sim, simulate
from fleet.scheduler import Candidate, Snapshot, choose
from jobs.gcode_prior import parse, plan, purge_tab
from jobs.orders import OrderStream, part_sku
from jobs import packaging as K
from jobs.packaging import box_for, plan_packages
from jobs.plates import BED_MARGIN_MM, MAX_PLATE_S, PURGE_TAB_CM3, make_plate, pack


def queue_of(orders):
    return [{"name": n, "order_id": o.order_id, "sku": part_sku(n)} for o in orders for n in o.parts]


def plates_from(queue, n, reach_mask=None, seed=0):
    rng, out = np.random.default_rng(seed), []
    for i in range(n):
        plate, gcode = make_plate(queue, i % 10, f"P{i:04d}", rng, reach_mask)
        done = {o["name"] for o in plate["objects"]}
        queue = [p for p in queue if p["name"] not in done]
        out.append((plate, gcode))
    return out, queue


@pytest.fixture(scope="module")
def orders():
    return OrderStream(seed=3).generate(36 * 3600)


@pytest.fixture(scope="module")
def plates(orders):
    return plates_from(queue_of(orders), 12)[0]


def test_order_stream(orders):
    assert orders == OrderStream(seed=3).generate(36 * 3600)
    assert all(a.t_s <= b.t_s for a, b in zip(orders, orders[1:]))
    for o in orders:
        assert 1 <= len(o.lines) <= 4 and len({s for s, _ in o.lines}) == len(o.lines)
        assert all(s in CATALOG and 1 <= q <= 8 for s, q in o.lines) and sum(q > 3 for _, q in o.lines) <= 1
        assert o.carrier in ("courier_A", "courier_B", "post") and o.ship_by_s > o.t_s
        assert sorted(p for k in o.packages for p in k.parts) == sorted(o.parts) and set(o.boxes) <= set(BOX_TYPES)
        assert o.book_entry()["packages"] is o.packages
    assert any(len(o.packages) > 1 for o in orders) and any(q > 3 for o in orders for _, q in o.lines)
    day = [o for o in orders if 8 <= o.t_s / 3600 % 24 < 20]
    assert len(day) > len(orders) - len(day)


def test_gcode_round_trip(plates, tmp_path):
    for plate, gcode in plates:
        path = tmp_path / f"{plate['plate_id']}.gcode"
        path.write_text(gcode)
        objs, purge = parse(path)
        assert set(objs) == {o["name"] for o in plate["objects"]}
        for o in plate["objects"]:
            g = objs[o["name"]]
            assert np.allclose(g["center"], o["center_mm"], atol=1e-3)
            assert g["height"] == pytest.approx(o["height_mm"])
            assert np.allclose(g["poly"], o["polygon_mm"], atol=1e-3)
        tab = purge_tab(path)
        assert np.allclose(tab["center_mm"], plate["purge_tab"]["center_mm"], atol=1e-3)
        assert np.allclose(tab["dims_mm"], plate["purge_tab"]["dims_mm"], atol=1e-3)
        assert purge and len(plan(objs)) == len(objs)


def rects(plate):
    return [(min(p[0] for p in o["polygon_mm"]), min(p[1] for p in o["polygon_mm"]),
             max(p[0] for p in o["polygon_mm"]), max(p[1] for p in o["polygon_mm"])) for o in plate["objects"]]


def test_plate_invariants(plates):
    for plate, _ in plates:
        assert plate["objects"] and plate["print_time_s"] <= MAX_PLATE_S
        assert plate["bed_mm"] == [256, 256] and set(plate) == {"plate_id", "printer", "print_time_s", "bed_mm",
                                                                 "purge_tab", "objects"}
        rs = rects(plate)
        for o, (x0, y0, x1, y1) in zip(plate["objects"], rs):
            fw, fd = CATALOG[o["sku"]].footprint
            assert (x1 - x0, y1 - y0) == pytest.approx((fw, fd) if o["yaw_deg"] == 0 else (fd, fw))
            assert np.allclose(o["center_mm"], [(x0 + x1) / 2, (y0 + y1) / 2])
            assert x0 >= BED_MARGIN_MM - 1e-9 and x1 <= BED_MM[0] - BED_MARGIN_MM + 1e-9
            assert y0 >= PURGE_STRIP_MM - 1e-9 and y1 <= BED_MM[1] - BED_MARGIN_MM + 1e-9
        for a, b in itertools.combinations(rs, 2):
            gap = max(b[0] - a[2], a[0] - b[2], b[1] - a[3], a[1] - b[3])
            assert gap >= PART_SPACING_MM - 1e-9, (a, b)
        (cx, cy), (w, d, _) = plate["purge_tab"]["center_mm"], plate["purge_tab"]["dims_mm"]
        assert 0 <= cy - d / 2 and cy + d / 2 <= PURGE_STRIP_MM and 0 <= cx - w / 2 and cx + w / 2 <= BED_MM[0]


def test_orders_kept_whole_and_fifo(orders):
    queue, rng, on = queue_of(orders), np.random.default_rng(0), {}
    for i in range(15):
        plate, _ = make_plate(queue, 0, f"P{i:04d}", rng)
        ids = {o["order_id"] for o in plate["objects"]}
        assert queue[0]["order_id"] in ids                  # the oldest queued order is always served
        for oid in ids:
            on.setdefault(oid, set()).add(i)
        queue = [p for p in queue if p["name"] not in {o["name"] for o in plate["objects"]}]
    fits_one = {o.order_id for o in orders if len(pack(queue_of([o]))) == len(o.parts)}
    split = [oid for oid, ps in on.items() if len(ps) > 1 and oid in fits_one]
    assert not split, split
    assert any(len(ps) > 1 for ps in on.values())          # the stream does contain too-big orders


def test_reach_mask(orders):
    mask = lambda x, y: x <= 128.0 and y <= 200.0
    out, _ = plates_from(queue_of(orders), 4, reach_mask=mask)
    for plate, _ in out:
        assert plate["objects"]
        assert all(mask(*p) for o in plate["objects"] for p in o["polygon_mm"])
        (cx, cy), (w, d, _) = plate["purge_tab"]["center_mm"], plate["purge_tab"]["dims_mm"]
        assert mask(cx + w / 2, cy + d / 2)


def book(n_orders=12, seed=0):
    rng = np.random.default_rng(seed)
    return {f"o{i}": {"parts": [f"o{i}_cube_{k}" for k in range(1, rng.integers(1, 4) + 1)], "carrier": "post"}
            for i in range(n_orders)}


def entry(lines, oid="o0"):
    parts = [f"{oid}_{sku}_{k}" for sku, q in lines for k in range(1, q + 1)]
    return {"parts": parts, "packages": tuple(plan_packages(oid, parts)), "carrier": "post"}


def check_caddy(a):
    """Caddy invariants: one package per tote, one reserved large part per tray slot, nothing shipped on board."""
    for t in a.totes:
        if t in a.tote_pkg:
            assert {a.part_pkg[q] for q in a.contents[t]} <= {a.tote_pkg[t]}
            assert not set(a.contents[t]) & a.large_parts
        else:
            assert not a.contents[t]
    for s in a.large:
        q = a.large_part.get(s)
        assert a.contents[s] == ([q] if q is not None and q in a.placed_parts[a.part_pkg[q]] else [])
    on_board = {q for s in (*a.totes, *a.large) for q in a.contents[s]}
    shelf = [q for ps in a.shelf.values() for q in ps]
    assert len(shelf) == len(set(shelf)) and not on_board & set(shelf)
    assert all(a.part_pkg[q] not in a.shipped for q in on_board | set(shelf))
    assert len(set(a.tote_pkg.values())) == len(a.tote_pkg)          # a package holds at most one tote
    assert all(not a.remaining_package(r.package.package_id) for r in a.handoff_ready())


def test_allocator_rules():
    a = CaddyAllocator(book(), n_large=0)
    assert a.slot_for("P0001_purge_tab") == "waste" and a.slot_for("o0_cube_1", defective=True) == "reject"
    got = [a.slot_for(f"o{i}_cube_1") for i in range(6)]
    assert got == list(SLOT_COST_ORDER)
    assert a.slot_for("o0_cube_1") == "o2"           # same package -> same tote
    assert a.slot_for("o6_cube_1") is None and a.metrics["none_events"] == 1
    with pytest.raises(ValueError):
        a.placed("o1_cube_1", "o0")                   # o0 holds o2's package
    for p in book()["o0"]["parts"]:
        done = a.placed(p, "o2")
    assert done == "o0-p0"
    (r,) = a.handoff_ready()
    assert (r.package.package_id, r.tote, r.large_slots, r.shelf, r.package.box) == ("o0-p0", "o2", (), (), "S")
    assert a.handed_off("o0-p0") == "o0" and "o2" in a.free_slots() and a.order_shipped("o0")
    with pytest.raises(ValueError):
        a.slot_for("o0_cube_1")                       # shipped packages never come back
    with pytest.raises(ValueError):
        a.handed_off("o1-p0")                         # incomplete
    assert a.slot_for("o6_cube_1") == "o2"
    check_caddy(a)


def test_allocator_multi_package_order():
    """15 small parts = 3 packages = 3 totes; the order ships with its last package; oldest order first."""
    a = CaddyAllocator({"o0": entry([("gear", 8), ("cube", 4), ("clip", 3)]), "o1": entry([("knob", 2)], "o1")},
                       n_large=0)
    pk = a.book["o0"]["packages"]
    assert len(pk) == 3 and [k.package_id for k in pk] == ["o0-p0", "o0-p1", "o0-p2"]
    for p in a.book["o1"]["parts"]:
        a.placed(p, a.slot_for(p))
    totes, done = {}, []
    for k in pk:
        for p in k.small:
            t = a.slot_for(p)
            assert totes.setdefault(k.package_id, t) == t
            done.append(a.placed(p, t))
        check_caddy(a)
    assert len(set(totes.values())) == 3 and set(totes.values()).isdisjoint(a.free_slots())
    assert [d for d in done if d] == ["o0-p0", "o0-p1", "o0-p2"]
    assert [r.package.package_id for r in a.handoff_ready()] == ["o0-p0", "o0-p1", "o0-p2", "o1-p0"]
    assert [a.handed_off(r.package.package_id) for r in a.handoff_ready()] == [None, None, "o0", "o1"]
    assert a.done == {"o0", "o1"} and len(a.free_slots()) == 6
    assert a.metrics["packages_shipped"] == 4 and a.metrics["orders_shipped"] == 2 and a.metrics["max_totes_used"] == 4


def test_allocator_large_tray(monkeypatch):
    monkeypatch.setattr(K, "LARGE_COPRINT", False)                   # allow 2-large packages (box rule only)
    pair = next(s for s in LARGE_SKUS if box_for((), [s, s]) is not None)    # two of them fit one box end to end
    a = CaddyAllocator({"o0": entry([(pair, 3), ("cube", 2)]), "o1": entry([("plate", 1)], "o1")}, n_large=2)
    l1, l2, l3 = (f"o0_{pair}_{k}" for k in (1, 2, 3))
    p0, p1 = a.book["o0"]["packages"]
    assert p0.large == (l1, l2) and p0.small == ("o0_cube_1", "o0_cube_2") and p1.large == (l3,)
    assert a.plate_effect([l1, "o0_cube_1", "x_purge_tab"]) == (1, 1, 0, 0, 1)
    assert a.slot_for(l1) == "L0" and a.slot_for(l1) == "L0"   # reserved for that part
    assert a.slot_for(l2) == "L1"
    assert a.slot_for(l3) is None and a.slot_for("o1_plate_1") is None and a.metrics["none_large"] == 2
    with pytest.raises(ValueError):
        a.placed(l2, "L0")                            # L0 is reserved for l1
    a.placed(l1, "L0")
    a.placed(l2, "L1")
    tote = a.slot_for("o0_cube_1")
    with pytest.raises(ValueError):
        a.placed(l3, tote)                            # a large part never goes into a tote
    a.placed("o0_cube_1", tote)
    assert a.plate_effect(["o0_cube_2"]) == (0, 0, 1, 1, 0)
    assert a.placed("o0_cube_2", tote) == "o0-p0"
    assert a.handoff_ready() == [(p0, tote, ("L0", "L1"), ())]
    assert a.handed_off("o0-p0") is None and a.free_large() == ["L0", "L1"] and tote in a.free_slots()
    assert a.slot_for(l3) == "L0" and a.placed(l3, "L0") == "o0-p1"
    assert a.handoff_ready() == [(p1, None, ("L0",), ())]
    assert a.handed_off("o0-p1") == "o0" and a.order_shipped("o0") and not a.order_shipped("o1")
    with pytest.raises(ValueError):
        CaddyAllocator({"o0": entry([("bar", 1)])}, n_large=0).slot_for("o0_bar_1")   # no tray in this cell


def test_allocator_park_and_shelf():
    a = CaddyAllocator({f"o{i}": entry([("cube", 2), ("plate", 1)], f"o{i}") for i in range(3)}, n_slots=2,
                       n_large=1)
    assert all(len(e["packages"]) == 1 for e in a.book.values())
    a.placed("o0_cube_1", a.slot_for("o0_cube_1"))
    assert a.slot_for("o0_plate_1") == "L0"
    a.placed("o1_cube_1", a.slot_for("o1_cube_1"))
    assert a.slot_for("o2_cube_1") is None
    assert a.park("o3") == "o1-p0" and a.shelf == {"o1-p0": ["o1_cube_1"]} and a.metrics["parks"] == 1
    assert a.slot_for("o2_cube_1") == "o3"
    a.placed("o0_plate_1", "L0")
    a.placed("o0_cube_2", "o2")
    with pytest.raises(ValueError):
        a.park("o2")                                  # complete: hand it off instead
    with pytest.raises(ValueError):
        a.park("L0")
    assert a.slot_for("o1_plate_1") is None
    assert a.handed_off("o0-p0") == "o0" and a.free_large() == ["L0"]
    a.placed("o1_plate_1", a.slot_for("o1_plate_1"))
    tote = a.slot_for("o1_cube_2")                   # o1's later parts get a new tote
    assert tote == "o2" and a.placed("o1_cube_2", tote) == "o1-p0"
    (r,) = a.handoff_ready()
    assert (r.tote, r.large_slots, r.shelf) == ("o2", ("L0",), ("o1_cube_1",))
    assert a.handed_off("o1-p0") == "o1" and ("handoff", "shelf", "o1-p0", ["o1_cube_1"]) in a.history
    a.placed("o2_plate_1", a.slot_for("o2_plate_1"))
    assert a.park("L0") == "o2-p0" and a.metrics["parks_large"] == 1 and a.shelf == {"o2-p0": ["o2_plate_1"]}
    check_caddy(a)


def test_allocator_unreserve():
    a = CaddyAllocator({"o0": entry([("bar", 1), ("cube", 2)])}, n_large=2)
    assert a.slot_for("o0_bar_1") == "L0" and a.unreserve("o0_bar_1") == "L0" and a.free_large() == ["L0", "L1"]
    t = a.slot_for("o0_cube_1")
    assert a.unreserve("o0_cube_1") == t and t in a.free_slots()   # empty tote: freed
    a.placed("o0_cube_1", a.slot_for("o0_cube_1"))
    assert a.unreserve("o0_cube_2") is None                        # tote already holds a part: kept


def test_allocator_one_row_path():
    """No tray, small SKUs only: one tote per package, as the legacy order bins for orders of <= 6 parts."""
    b = book()
    a = CaddyAllocator(b, n_large=0)
    assert a.large == () and a.free_large() == [] and all(len(a.order_pkgs[o]) == 1 for o in b)
    for oid in list(b)[:6]:
        for p in b[oid]["parts"]:
            a.placed(p, a.slot_for(p))
    assert [r.tote for r in a.handoff_ready()] == [a.totes[i] for i in range(6)] and not a.free_slots()
    assert CaddyAllocator().large == tuple(LARGE_SLOTS)


def test_allocator_invariants_fuzz():
    rng = np.random.default_rng(1)
    b = {}
    for i in range(40):
        skus = rng.choice(list(CATALOG), int(rng.integers(1, 4)), replace=False)
        lines = [(str(s), int(rng.integers(1, 4))) for s in skus]
        if i % 7 == 0:
            lines[0] = (lines[0][0], 8)               # bulk line: a multi-package order
        b[f"o{i}"] = entry(lines, f"o{i}")
    a = CaddyAllocator(b, n_large=2)
    todo = [p for e in b.values() for p in e["parts"]]
    rng.shuffle(todo)
    while todo:
        p = todo.pop()
        if rng.random() < 0.05:                       # defective copy, reprinted later
            a.placed(p, a.slot_for(p, defective=True))
            todo.insert(0, p)
            continue
        slot = a.slot_for(p)
        while slot is None:                           # station: hand off, else park (deadlock breaker)
            ready = a.handoff_ready()
            for r in ready:
                a.handed_off(r.package.package_id)
            if not ready:
                held = {s: a.part_pkg[q] for s, q in a.large_part.items()} if p in a.large_parts else dict(a.tote_pkg)
                a.park(max(sorted(held), key=lambda s: len(a.remaining_package(held[s]))))
            slot = a.slot_for(p)
        a.placed(p, slot)
        check_caddy(a)
    for r in a.handoff_ready():
        a.handed_off(r.package.package_id)
    shipped = [q for ev, _, _, ps in a.history if ev == "handoff" for q in ps]
    assert sorted(shipped) == sorted(p for e in b.values() for p in e["parts"])   # every part shipped exactly once
    assert a.done == set(b) and not a.tote_pkg and not a.large_part and not a.shelf
    assert a.metrics["packages_shipped"] == sum(len(e["packages"]) for e in b.values()) > len(b)
    assert a.metrics["max_totes_used"] <= 6 and a.metrics["max_large_used"] == 2
    assert a.metrics["parks"] > a.metrics["parks_large"] > 0 and a.metrics["reject"] > 0


def test_scheduler_policies():
    c = [Candidate(0, 0.15, t_done=100.0, t_release=550.0, n_parts=4),
         Candidate(4, 1.99, t_done=0.0, t_release=450.0, n_parts=4, new_totes=3),
         Candidate(7, 1.69, t_done=400.0, t_release=850.0, n_parts=2, completes=1, frees=1)]
    snap = Snapshot(t=600.0, robot_x=0.2, rail_speed=0.4, candidates=tuple(c), next_job_s=10800, max_plate_s=10800,
                    unload_fixed_s=45.0, pick_s=11.0, free_slots=1, ready_packages=0, station_x=2.59, station_s=6.0)
    assert choose("fifo", snap).printer == 4
    assert choose("nearest", snap).printer == 0
    assert choose("max_util", snap).printer in (0, 4)
    assert choose("order_aware", snap).printer == 0            # printer 4 needs 3 totes, only 1 free
    assert choose("fifo", Snapshot(**{**snap.__dict__, "candidates": ()})) is None
    c[0] = Candidate(0, 0.15, t_done=100.0, t_release=550.0, n_parts=4, new_large=3)
    tray = Snapshot(**{**snap.__dict__, "candidates": tuple(c), "free_large": 2})
    assert choose("order_aware", tray).printer == 7            # 3 large parts, 2 tray slots; 4 still short of totes
    assert choose("order_aware", Snapshot(**{**tray.__dict__, "free_large": None})).printer == 0   # no tray modelled


def check_conservation(sim):
    queued = set(sim.queue)
    on_bed = {n for pr in sim.printers for n in pr.todo}
    reprint = {arg for _, _, kind, arg in sim.heap if kind == "reprint"}
    for n, part in sim.parts.items():
        assert part["good"] <= 1, n
        assert part["prints"] == part["good"] + part["rejects"] + (n in on_bed), n
        where = [n in queued, n in on_bed, part["good"] == 1, n in reprint]
        assert sum(where) == 1, (n, where)
    for oid, o in sim.orders.items():
        st = sim.ost[oid]
        if st["t_shipped"] is not None:
            assert all(sim.parts[n]["good"] == 1 and sim.parts[n]["t_good"] <= st["t_shipped"] for n in o.parts)
            assert st["t_complete"] <= st["t_shipped"]
            assert st["t_shipped"] == max(sim.pst[k.package_id]["t_shipped"] for k in o.packages)   # last package


@pytest.mark.parametrize("policy", ["fifo", "order_aware", "human"])
def test_des_conservation_after_drain(policy):
    sim = simulate(days=3, policy=policy, seed=1)
    check_conservation(sim)
    assert all(st["t_shipped"] is not None for st in sim.ost.values())   # drained: nothing lost
    assert all(p["good"] == 1 for p in sim.parts.values())                # every ordered part printed once (good)


@pytest.mark.parametrize("policy", ["nearest", "max_util", "human"])
def test_des_no_order_lost_at_horizon(policy):
    sim = Sim(Params(days=2, policy=policy, seed=2, drain_days=0.0))
    sim.run()
    check_conservation(sim)
    queued, on_bed = set(sim.queue), {n for pr in sim.printers for n in pr.todo}
    for oid, o in sim.orders.items():
        shipped = sim.ost[oid]["t_shipped"] is not None
        started = any(sim.parts[n]["prints"] for n in o.parts)
        waiting = all(n in queued for n in o.parts)
        assert shipped or started or waiting, oid
        assert not (shipped and any(n in queued or n in on_bed for n in o.parts)), oid
    if policy != "human":
        caddy = {q for s in SLOT_COST_ORDER for q in sim.alloc.contents[s]}
        assert all(sim.parts[n]["good"] == 1 for n in caddy)


def test_human_baseline():
    sim = simulate(days=7, policy="human", seed=0)
    k = sim.kpis()
    assert k["robot_util_pct"] is None and k["interventions"] == 0 and k["rail_m"] == 0
    assert k["orders_shipped"] > 0 and k["human_h"] > 0
    for t0, _, _, what in sim.busy:                   # the operator only starts unloading inside shifts
        assert what == "pack" or sim.in_shift(t0), t0 / 3600
    starts = [r[0] for r in sim.events if r[1] == "human_check"]
    assert all(sim.in_shift(t) for t in starts)
    robot = simulate(days=7, policy="order_aware", seed=0).kpis()
    assert robot["printer_util_pct"] > k["printer_util_pct"]
    assert robot["lead_time_p90_h"] < k["lead_time_p90_h"]


def test_shift_past_midnight():
    H = 3600.0
    day = Sim(Params(days=1, policy="human", seed=0))
    grid = np.arange(0.0, 7 * 24 * H, 0.25 * H)
    old = [int(t // 86400) % 7 in (0, 1, 2, 3, 4) and 9.0 <= (t / H) % 24 < 18.0 for t in grid]
    assert [day.in_shift(t) for t in grid] == old                       # the default shift is unchanged
    two = Sim(Params(days=1, policy="human", seed=0, shift_h=(9.0, 26.0)))   # 9-18 plus 18-02
    mon, tue, sat = 0.0, 24 * H, 5 * 24 * H
    assert two.in_shift(mon + 20 * H) and two.in_shift(tue + 1.5 * H) and not two.in_shift(tue + 3 * H)
    assert two.in_shift(sat + 1 * H) and not two.in_shift(sat + 10 * H)      # Friday's evening shift ends Sat 02:00
    assert not two.in_shift(mon + 1 * H)                                     # Sunday is no workday
    assert two.next_check(tue + 0.5 * H) == tue + 1 * H                      # the last check of Monday's shift
    assert two.next_check(tue + 2.5 * H) == tue + 9 * H
    assert two.next_check(sat + 3 * H) == 7 * 24 * H + 9 * H                 # weekend: next Monday 09:00
    seven = Sim(Params(days=1, policy="human", seed=0, shift_h=(9.0, 26.0), workdays=tuple(range(7))))
    assert seven.in_shift(mon + 1 * H) and seven.next_check(sat + 3 * H) == sat + 9 * H
    with pytest.raises(AssertionError):
        Sim(Params(days=1, policy="human", seed=0, shift_h=(9.0, 40.0)))   # longer than a day


def test_human_second_shift():
    base = simulate(days=7, policy="human", seed=0).kpis()
    sim = simulate(days=7, policy="human", seed=0, shift_h=(9.0, 26.0))
    k = sim.kpis()
    check_conservation(sim)
    for t0, _, _, what in sim.busy:                   # the operator still starts unloading only inside a shift
        assert what == "pack" or sim.in_shift(t0), t0 / 3600
    assert k["parts_per_day"] > base["parts_per_day"] and k["printer_util_pct"] > base["printer_util_pct"]
    assert k["lead_time_p50_h"] < base["lead_time_p50_h"]


def test_determinism():
    a, b = simulate(days=2, policy="order_aware", seed=5), simulate(days=2, policy="order_aware", seed=5)
    assert a.kpis() == b.kpis() and a.events == b.events
    assert simulate(days=2, policy="order_aware", seed=6).kpis() != a.kpis()


# ---- layouts: printers per arm (fleet.des.make_layout, scripts/layout_study.py) ----
from fleet.allocator import slot_cost_order
from fleet.des import default_layout, make_layout
from farm.catalog import N_PRINTERS, N_ROWS

CATALOG_KIND = "row" if N_ROWS == 1 else "facing"     # the cell farm.catalog builds (PRINTFARM_LAYOUT)


def test_make_layout_geometry():
    d, r = default_layout(), make_layout(CATALOG_KIND, N_PRINTERS, "end")
    assert r.frames == d.frames and r.station_x == pytest.approx(d.station_x)
    assert (r.rail_min, r.rail_max) == pytest.approx((d.rail_min, d.rail_max)) and r.x_open == pytest.approx(d.x_open)
    f = make_layout("facing", 11, "end")
    assert f.n == 11 and sum(s > 0 for _, s in f.frames) == 6 and sum(s < 0 for _, s in f.frames) == 5
    for (xc, s), xo in zip(f.frames, f.x_open):          # door pull from the printer's right, mirrored for row B
        assert xo == pytest.approx(min(f.rail_max, max(f.rail_min, xc + s * 0.15)))
    assert f.rail_length_m < make_layout("row", 11, "end").rail_length_m
    c = make_layout("facing", 20, "center")
    xs = sorted({round(x, 3) for x, _ in c.frames})
    assert len(xs) == 10 and all(abs(x - c.station_x) > 0.2 for x in xs)   # the station takes one column
    assert min(xs) < c.station_x < max(xs)
    rc = make_layout("row", 20, "center")
    assert rc.frames == make_layout("row", 20, "end").frames and rc.rail_length_m < make_layout("row", 20).rail_length_m
    for lay in (f, c, rc):
        assert all(lay.rail_min <= x <= lay.rail_max for x in lay.x_open + [lay.station_x])


def test_default_layout_unchanged():
    a = simulate(days=2, policy="order_aware", seed=3)
    b = simulate(days=2, policy="order_aware", seed=3, layout=CATALOG_KIND, n_printers=N_PRINTERS, station="end")
    assert a.kpis() == b.kpis() and a.events == b.events


@pytest.mark.parametrize("layout,station,n", [("facing", "center", 16), ("row", "center", 13), ("facing", "end", 7)])
def test_layout_des_conservation(layout, station, n):
    sim = simulate(days=2, policy="order_aware", seed=4, layout=layout, n_printers=n, station=station,
                   order_scale=n / 10)
    check_conservation(sim)
    assert all(st["t_shipped"] is not None for st in sim.ost.values())
    k = sim.kpis()
    assert k["n_printers"] == n == len(sim.printers) and k["layout"] == layout
    sh = sim.state_hours()
    assert k["printer_util_pct"] == pytest.approx(100 * sh["printing"] / (n * 2 * 24), abs=0.01)
    assert len({round(pr.x_open, 3) for pr in sim.printers}) == n     # every printer has its own door-pull spot


def test_more_printers_per_arm():
    """Saturated demand: twice the printers on one arm give more output and a busier arm (well below the knee)."""
    k10, k20 = (simulate(days=3, policy="order_aware", seed=0, layout="facing", n_printers=n,
                         order_scale=0.15 * n).kpis() for n in (10, 20))
    assert k20["parts_per_day"] > 1.7 * k10["parts_per_day"]
    assert k20["robot_util_pct"] > 1.6 * k10["robot_util_pct"]
    assert k20["printer_wait_pct"] < 5.0


def test_caddy_slots_param():
    assert slot_cost_order(6) == SLOT_COST_ORDER and slot_cost_order(4) == SLOT_COST_ORDER[:4]
    big = slot_cost_order(9)
    assert big[:6] == SLOT_COST_ORDER and set(big) == {f"o{k}" for k in range(9)}
    a = CaddyAllocator(book(), n_slots=9)
    assert [a.slot_for(f"o{i}_cube_1") for i in range(9)] == list(big) and a.slot_for("o9_cube_1") is None
    sim = simulate(days=2, policy="order_aware", seed=1, order_slots=12, order_scale=2.0)
    check_conservation(sim)
    assert 6 < sim.alloc.metrics["max_totes_used"] <= 12


def test_des_packages_and_tray():
    """Orders with large SKUs (two-row default once integrated): totes + tray slots, multi-package orders, drained.
    A SKU that alone prints longer than MAX_PLATE_S is left out (make_plate would never take it)."""
    skus = tuple(s for s in CATALOG if print_time_s(CATALOG[s].volume_cm3 + PURGE_TAB_CM3) <= MAX_PLATE_S)
    sim = simulate(days=3, policy="order_aware", seed=2, order_skus=skus, large_slots=2, order_scale=1.5)
    check_conservation(sim)
    assert all(st["t_shipped"] is not None for st in sim.ost.values())
    k = sim.kpis()
    assert k["packages_per_order"] > 1.05 and k["max_tray_slots_used"] == 2 and k["max_caddy_slots_used"] <= 6
    assert any(sim.parts[n]["sku"] in LARGE_SKUS for n in sim.parts)
    a = sim.alloc
    assert a.done == set(sim.orders) and a.metrics["packages_shipped"] == sum(len(o.packages) for o in sim.orders.values())


# ---- bed to box at the station (measured step times, runs/station_times.json) ----
from fleet.des import DEFAULT_STATION, load_station_times, sku_mix
from jobs.orders import Order


def test_station_times(tmp_path):
    s, src = load_station_times()
    assert src == "runs/station_times.json" and s["large_place"] > s["tote_dump"] > 0
    assert s["waste_dump"] == pytest.approx(s["dock"] + s["tote_dump"])          # not measured yet: composed
    d, src = load_station_times(tmp_path / "missing.json")
    assert d == {**DEFAULT_STATION, "waste_dump": DEFAULT_STATION["dock"] + DEFAULT_STATION["tote_dump"]}
    skus, out = sku_mix("all")
    assert set(skus) | set(out) == set(CATALOG) and sku_mix("small") == (SMALL_SKUS, ())
    assert all((print_time_s(CATALOG[k].volume_cm3 + PURGE_TAB_CM3) > MAX_PLATE_S) == (k in out) for k in CATALOG)


def test_station_visit_bed_to_box():
    """Dock + per package (box request, one place per tray part, tote dump) + waste dump; seals in the background; the
    order ships with its last package."""
    sim = Sim(Params(days=1, seed=0, large_slots=2))
    S, a = sim.S, sim.alloc
    o = Order("x0", 0.0, "c0", "post", 1e9, (("cube", 7), ("bar", 1)))
    sim.orders["x0"] = o
    sim.ost["x0"] = {"t_complete": 50.0, "t_shipped": None, "good": 8, "packed": False}
    a.add_order("x0", o.book_entry())
    for p in o.parts:
        a.placed(p, a.slot_for(p))
    p0, p1 = o.packages
    assert len(p0.large) == 1 and len(p0.small) == 6 and p1.parts == ("x0_cube_7",)
    sim.robot_x, sim.waste = sim.station_x, 3
    t1 = sim.station_visit(100.0)
    pkg = S["box_request"] + S["to_tote"] + S["tote_dump"]
    t_p0 = 100.0 + S["dock"] + pkg + S["large_place"]
    assert sim.pst["x0-p0"]["t_shipped"] == pytest.approx(t_p0 + S["seal"] + sim.p.lane_s)
    assert sim.pst["x0-p1"]["t_shipped"] == pytest.approx(t_p0 + pkg + S["seal"] + sim.p.lane_s)
    assert sim.ost["x0"]["t_shipped"] == sim.pst["x0-p1"]["t_shipped"] and a.order_shipped("x0")
    assert t1 == pytest.approx(t_p0 + pkg + S["waste_dump"]) and sim.waste == 0
    assert sim.k["station_s"] == pytest.approx(t1 - 100.0) and sim.k["waste_dumps"] == 1


# ---- package-aware plate composition (fleet/queue_policy.py) ----
from fleet.queue_policy import MAX_SPLIT, order_queue, plate_volume_cm3, split_packages


def test_order_queue_rules():
    a = CaddyAllocator({"o0": entry([("cube", 6)], "o0"), "o1": entry([("cube", 1)], "o1"),
                        "o2": entry([("gear", 6)], "o2"), "o3": entry([("knob", 6)], "o3")}, n_large=0)
    cap = plate_volume_cm3(MAX_PLATE_S)
    assert print_time_s(PURGE_TAB_CM3 + cap) == pytest.approx(MAX_PLATE_S, abs=1.0)
    assert all(6 * CATALOG[s].volume_cm3 > cap for s in ("cube", "gear", "knob"))      # each needs two plates
    fifo = [p for o in ("o0", "o1", "o2", "o3") for p in a.book[o]["parts"]]
    queue = [p for p in fifo if p not in ("o0_cube_1", "o0_cube_2")]                 # o0 is half printed
    queue = [p for p in queue if p != "o1_cube_1"] + ["o1_cube_1"]                    # (o1 behind the others)
    assert split_packages(queue, a) == {"o0-p0"}
    got = order_queue(queue, a)
    assert [d["name"] for d in got][:4] == ["o0_cube_3", "o0_cube_4", "o0_cube_5", "o0_cube_6"]   # split first
    assert sorted(d["name"] for d in got) == sorted(queue)                           # one split: nothing held
    assert {d["order_id"] for d in got} == {"o0-p0", "o1-p0", "o2-p0", "o3-p0"}      # packer groups by package
    q2 = [p for p in queue if p != "o2_gear_1"]                                       # a second package split
    got = [d["name"] for d in order_queue(q2, a)]
    assert len(split_packages(q2, a)) == MAX_SPLIT and not any(n.startswith("o3_") for n in got)   # big one waits
    assert got[:9] == [p for p in q2 if p.startswith(("o0_", "o2_"))] and got[-1] == "o1_cube_1"  # small one runs
    dicts = [{"name": n, "order_id": n.split("_")[0], "sku": part_sku(n)} for n in queue]
    out = order_queue(dicts, a)
    assert [d["name"] for d in out] == [d["name"] for d in order_queue(queue, a)]
    assert all(d["order_id"] == d["name"].split("_")[0] for d in dicts)               # caller's queue untouched
    assert order_queue([{"name": "P1_x_1", "order_id": "P1", "sku": "cube"}], a)[0]["order_id"] == "P1"   # unknown


def test_des_queue_policy():
    """Package-aware plates keep totes from being held across plates: far fewer caddy-full events and parks."""
    fifo, pkg = (simulate(days=4, policy="order_aware", seed=0, queue_policy=q) for q in ("fifo", "package"))
    for sim in (fifo, pkg):
        check_conservation(sim)
        assert all(st["t_shipped"] is not None for st in sim.ost.values())
    kf, kp = fifo.kpis(), pkg.kpis()
    assert kp["partial_handoffs"] * 4 <= kf["partial_handoffs"] and kp["caddy_full_events"] * 4 <= kf["caddy_full_events"]
    assert kp["parts_per_day"] >= 0.98 * kf["parts_per_day"]
    assert simulate(days=2, policy="human", seed=0, queue_policy="fifo").kpis() == \
        simulate(days=2, policy="human", seed=0).kpis()                                # the human ignores it


# ---- tray-aware flow: large parts last, the last tray slot kept for packages that complete ----
from fleet.scheduler import TRAY_HOLD_S, order_aware_choice, tray_short


def test_order_queue_large_last():
    a = CaddyAllocator({"o0": entry([("housing", 1), ("cube", 2)], "o0")}, n_large=2)
    q = list(a.book["o0"]["parts"])
    assert [d["name"] for d in order_queue(q, a)] == ["o0_cube_1", "o0_cube_2"]       # housing waits for its tote
    assert [d["name"] for d in order_queue(["o0_housing_1"], a)] == ["o0_housing_1"]   # small parts are printing
    assert [d["name"] for d in order_queue(q, a, large_last=False)][0] == "o0_housing_1"


def test_tray_aware_choice():
    a = CaddyAllocator({"o0": entry([("bar", 1), ("cube", 1)], "o0"), "o1": entry([("plate", 1)], "o1")}, n_large=2)
    assert a.plate_effect(["o0_bar_1"]).open_large == 1 and a.plate_effect(["o0_bar_1", "o0_cube_1"]).open_large == 0
    assert a.plate_effect(["o1_plate_1"]).open_large == 0
    opn = Candidate(0, 0.15, t_done=0.0, t_release=500.0, n_parts=1, new_large=1, open_large=1)
    done = Candidate(1, 0.61, t_done=0.0, t_release=500.0, n_parts=1, new_large=1)
    snap = Snapshot(t=600.0, robot_x=0.2, rail_speed=0.4, candidates=(opn, done), next_job_s=10800, max_plate_s=10800,
                    unload_fixed_s=45.0, pick_s=11.0, free_slots=6, ready_packages=0, station_x=2.59, station_s=6.0,
                    free_large=1)
    assert tray_short(snap, opn) == 1 and tray_short(snap, done) == 0      # the last slot: only a completing package
    assert tray_short(Snapshot(**{**snap.__dict__, "free_large": 2}), opn) == 0
    assert tray_short(Snapshot(**{**snap.__dict__, "t": 500.0 + TRAY_HOLD_S}), opn) == 0   # waited long enough
    assert order_aware_choice(snap).printer == 1
    only = Snapshot(**{**snap.__dict__, "candidates": (opn,)})
    assert order_aware_choice(only) is None                                 # wait for a tray slot
    assert order_aware_choice(Snapshot(**{**only.__dict__, "tray_wait": False})).printer == 0
    assert order_aware_choice(Snapshot(**{**only.__dict__, "ready_packages": 1})).printer == 0   # station first
    full = Snapshot(**{**only.__dict__, "free_large": 0, "t": 500.0 + TRAY_HOLD_S})
    assert order_aware_choice(full).printer == 0                            # overdue: unload (a park may follow)


def test_des_tray_policy():
    """All SKUs at matched demand: the tray-aware flow keeps parks down without costing output."""
    kw = dict(days=4, policy="order_aware", seed=0, order_scale=1.48, order_skus=sku_mix("all")[0], large_slots=2)
    off, on = (simulate(**kw, tray_aware=f, tray_wait=f, large_last=f) for f in (False, True))
    for sim in (off, on):
        check_conservation(sim)
        assert all(st["t_shipped"] is not None for st in sim.ost.values())
    ko, kn = off.kpis(), on.kpis()
    assert kn["partial_handoffs"] <= ko["partial_handoffs"] and kn["parts_per_day"] >= 0.98 * ko["parts_per_day"]
