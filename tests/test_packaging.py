"""Package planning (jobs/packaging.py) and the shared order-line sampler (jobs/orders.py).
Dimensions are read from farm.catalog, so the checks hold when the large SKUs are resized."""
from functools import lru_cache
import itertools

import numpy as np
import pytest

from farm import catalog as C
from jobs import packaging as K
from jobs.orders import BULK_QTY, LINES_P, Order, OrderStream, sample_lines
from jobs.packaging import box_for, boxes_by_size, fit, jaw_room_mm, package_fit, plan_packages


def parts_of(lines, oid="o0"):
    return [f"{oid}_{sku}_{k}" for sku, q in lines for k in range(1, q + 1)]


def skus_in(pkg):
    return {K.part_sku(p) for p in pkg.parts}


@lru_cache(maxsize=None)
def min_sets(skus):
    """Fewest sets of <= LARGE_PER_PACKAGE large SKUs (sorted tuple) that may share a package, by brute force."""
    if not skus:
        return 0
    first, rest = skus[0], skus[1:]
    best = 1 + min_sets(rest)
    for k in range(1, C.LARGE_PER_PACKAGE):
        for idx in itertools.combinations(range(len(rest)), k):
            if K.large_pair_ok([first] + [rest[i] for i in idx]):
                best = min(best, 1 + min_sets(tuple(r for i, r in enumerate(rest) if i not in idx)))
    return best


def check_plan(oid, parts, pkgs):
    assert sorted(p for k in pkgs for p in k.parts) == sorted(parts)              # each part exactly once
    assert [k.package_id for k in pkgs] == [f"{oid}-p{i}" for i in range(len(pkgs))]
    for k in pkgs:
        assert k.order_id == oid and all(p.startswith(oid + "_") for p in k.parts)
        assert k.parts and len(k.small) <= C.TOTE_MAX_PARTS and len(k.large) <= C.LARGE_PER_PACKAGE
        assert all(not C.CATALOG[K.part_sku(p)].large for p in k.small)
        assert all(C.CATALOG[K.part_sku(p)].large for p in k.large)
        assert package_fit(k) is not None
        smaller = boxes_by_size()[:boxes_by_size().index(k.box)]
        assert all(fit(b, [K.part_sku(p) for p in k.small], [K.part_sku(p) for p in k.large]) is None for b in smaller)


def test_fifteen_small_parts_three_packages():
    parts = parts_of([("gear", 8), ("cube", 4), ("clip", 3)])
    pkgs = plan_packages("o0", parts)
    check_plan("o0", parts, pkgs)
    assert len(pkgs) == 3 and {k.box for k in pkgs} == {"S"}
    assert sum("gear" in skus_in(k) for k in pkgs) == 2                 # 8 gears > one tote: split once
    assert sum("cube" in skus_in(k) for k in pkgs) == sum("clip" in skus_in(k) for k in pkgs) == 1
    one = plan_packages("o1", parts_of([("cube", 15)], "o1"))
    assert [len(k.small) for k in one] == [6, 6, 3]


def test_bar_forces_box_m():
    assert fit("S", (), ["bar"]) is None and fit("M", (), ["bar"]) is not None
    for lines in ([("bar", 1)], [("bar", 1), ("cube", 2)], [("bar", 1), ("knob", 6)]):
        pkgs = plan_packages("o0", parts_of(lines))
        check_plan("o0", parts_of(lines), pkgs)
        assert len(pkgs) == 1 and pkgs[0].box in ("M", "L")
    lines = [("bar", 2), ("knob", 6)]
    pkgs = plan_packages("o0", parts_of(lines))
    check_plan("o0", parts_of(lines), pkgs)
    assert len(pkgs) == min_sets(("bar", "bar")) and all(k.box in ("M", "L") for k in pkgs)


def test_large_and_small_mix():
    lines = [("housing", 2), ("plate", 1), ("cube", 7)]
    pkgs = plan_packages("o0", parts_of(lines))
    check_plan("o0", parts_of(lines), pkgs)
    assert len(pkgs) == max(min_sets(("housing", "housing", "plate")), 2)
    if K.large_pair_ok(["housing", "housing"]):                      # same SKU kept together on a tie
        assert any(set(k.large) == {"o0_housing_1", "o0_housing_2"} for k in pkgs)
    f = package_fit(next(k for k in pkgs if "o0_plate_1" in k.large))
    assert f.end_gap_mm >= K.BOX_CLEAR_MM and f.jaw_room_mm >= f.jaw_need_mm
    # fewest packages beats same-SKU sets: 2 housings + 2 plates
    lines = [("housing", 2), ("plate", 2)]
    pkgs = plan_packages("o0", parts_of(lines))
    check_plan("o0", parts_of(lines), pkgs)
    assert len(pkgs) == min_sets(("housing", "housing", "plate", "plate"))
    # more large parts than small: large-only packages need no tote
    lines = [("bar", 5), ("cube", 1)]
    pkgs = plan_packages("o0", parts_of(lines))
    check_plan("o0", parts_of(lines), pkgs)
    assert len(pkgs) == min_sets(("bar",) * 5) and sum(bool(k.small) for k in pkgs) == 1


def test_large_parts_share_a_package_only_if_coprinted(monkeypatch):
    """Two large parts in one package must print on one plate (else the first blocks a tray slot meanwhile)."""
    parts = parts_of([("housing", 2)])
    two = K.coprint(["housing", "housing"])
    assert len(plan_packages("o0", parts)) == (1 if two else 2)
    assert K.coprint(["housing"]) and not K.coprint(["housing"] * 2, max_print_s=600.0)
    monkeypatch.setattr(K, "LARGE_COPRINT", False)
    (k,) = plan_packages("o0", parts)                                  # the box alone would take both
    assert k.large == ("o0_housing_1", "o0_housing_2") and k.box == "M"


def test_plan_deterministic_and_order_free():
    rng = np.random.default_rng(0)
    for _ in range(50):
        parts = parts_of(sample_lines(rng, tuple(C.CATALOG)))
        a = plan_packages("o0", parts)
        shuffled = list(parts)
        rng.shuffle(shuffled)
        assert a == plan_packages("o0", parts) == plan_packages("o0", shuffled)


def test_plan_conservation_and_minimal_count():
    rng = np.random.default_rng(1)
    n_multi = 0
    for i in range(400):
        lines = sample_lines(rng, tuple(C.CATALOG))
        parts = parts_of(lines, f"o{i}")
        pkgs = plan_packages(f"o{i}", parts)
        check_plan(f"o{i}", parts, pkgs)
        n_small = sum(q for s, q in lines if not C.CATALOG[s].large)
        large = tuple(sorted(s for s, q in lines if C.CATALOG[s].large for _ in range(q)))
        assert len(pkgs) == max(-(-n_small // C.TOTE_MAX_PARTS), min_sets(large))
        assert sum(bool(k.small) for k in pkgs) == -(-n_small // C.TOTE_MAX_PARTS)   # fewest totes
        n_multi += len(pkgs) > 1
    assert n_multi > 10


def test_plan_rejects_mixed_orders():
    with pytest.raises(ValueError):
        plan_packages("o1", ["o1_cube_1", "o2_cube_1"])
    with pytest.raises(ValueError):
        plan_packages("o1", ["o1_cube_1", "o1_cube_1"])
    assert plan_packages("o1", []) == []


def test_split_further_when_no_box_fits(monkeypatch):
    monkeypatch.setattr(C, "BOX_TYPES", {"S": C.BOX_TYPES["S"]})
    monkeypatch.setattr(C, "BOX_FILL", 0.23)
    assert box_for(["bracket"] * 6, ["housing"]) is None                # too much volume for one S box
    parts = parts_of([("housing", 1), ("bracket", 6)])
    pkgs = plan_packages("o0", parts)
    check_plan("o0", parts, pkgs)
    assert len(pkgs) == 2 and {k.box for k in pkgs} == {"S"} and sum(bool(k.large) for k in pkgs) == 1
    with pytest.raises(ValueError):
        plan_packages("o0", parts_of([("bar", 1)]))                        # fits no box at all


def test_jaw_rule_keeps_measured_places():
    """Station place trials (scripts/station_place_check.py): fingers beside the long sides need room, more in deeper boxes."""
    assert 32.5 < jaw_room_mm(C.BOX_TYPES["M"][2]) <= 37.5                 # M: failed at 32.5 mm, placed at 37.5
    for box, sku in (("S", "housing"), ("M", "housing"), ("S", "plate"), ("M", "bar")):
        assert fit(box, (), [sku]) is not None, (box, sku)                 # placed centred in the sim


@pytest.mark.parametrize("box", list(C.BOX_TYPES))
def test_fit_geometry(box):
    """Every layout is placeable: long sides along box x, centred across, end to end, clearance and jaw room."""
    w, d, h = C.BOX_TYPES[box]
    c, jaw = K.BOX_CLEAR_MM, jaw_room_mm(h)
    combos = [(s,) for s in C.LARGE_SKUS] + list(itertools.combinations_with_replacement(C.LARGE_SKUS, 2))
    n_fit = 0
    for large in combos:
        for small in ((), ("bracket",) * C.TOTE_MAX_PARTS):
            f = fit(box, small, large)
            if f is None:
                continue
            n_fit += 1
            spans = []
            for sku, (x, y, yaw) in zip(large, f.layout):
                fp = C.CATALOG[sku].footprint
                along, across = fp if yaw == 0 else fp[::-1]
                assert y == 0.0 and along == max(fp)                       # long side along x, centred across
                assert (d - across) / 2 >= jaw - 1e-6                      # room for the fingers
                spans.append((x - along / 2, x + along / 2))
            spans.sort()
            assert spans[0][0] >= -w / 2 + c - 1e-6 and spans[-1][1] <= w / 2 - c + 1e-6
            assert all(b[0] - a[1] >= c - 1e-6 for a, b in zip(spans, spans[1:]))   # end to end, never side by side
            assert f.fill <= C.BOX_FILL and f.height_mm <= h
            assert max(C.CATALOG[s].height for s in large) + f.pile_mm == pytest.approx(f.height_mm, abs=0.11)
            if small:
                assert K.TOTE_OPENING_MM <= min(w, d) and f.opening_gap_mm >= 0
    assert n_fit > 0


def test_sample_lines_mix():
    rng = np.random.default_rng(3)
    small = [sample_lines(rng) for _ in range(3000)]
    assert all(s in C.ORDER_SKUS for ls in small for s, _ in ls)
    n_lines = np.bincount([len(ls) for ls in small], minlength=5)[1:] / len(small)
    assert np.allclose(n_lines, LINES_P, atol=0.03)
    bulk = [ls for ls in small if any(q > 3 for _, q in ls)]
    assert 0.06 < len(bulk) / len(small) < 0.10
    assert all(sum(q > 3 for _, q in ls) == 1 and all(q <= BULK_QTY[1] for _, q in ls) for ls in bulk)
    every = [sample_lines(rng, tuple(C.CATALOG)) for _ in range(5000)]
    lines = [s for ls in every for s, _ in ls]
    share = sum(C.CATALOG[s].large for s in lines) / len(lines)
    assert 0.12 < share < 0.20 and {s for s in lines} == set(C.CATALOG)
    a, b = np.random.default_rng(9), np.random.default_rng(9)
    assert [sample_lines(a, C.CATALOG) for _ in range(20)] == [sample_lines(b, tuple(reversed(C.CATALOG)))
                                                                for _ in range(20)]


def test_order_packages():
    o = Order("o5", 0.0, "c1", "post", 3600.0, (("gear", 7), ("bar", 1)))
    assert o.packages is o.packages and len(o.packages) == 2
    assert o.boxes == tuple(k.box for k in o.packages) and o.book_entry()["packages"] == o.packages
    orders = OrderStream(seed=4, skus=tuple(C.CATALOG)).generate(48 * 3600)
    assert any(k.large for o in orders for k in o.packages)
    assert sum(len(o.parts) for o in orders) == sum(len(k.parts) for o in orders for k in o.packages)
