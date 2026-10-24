"""Package planning: order -> packages -> box size (pure Python).

A package is what leaves the cell in one shipping box: the small parts of ONE tote (TOTE_MAX_PARTS at most) plus at
most LARGE_PER_PACKAGE large parts from the large-part tray. Orders larger than that ship as several packages;
a package never holds parts of two orders, and packages of one order never share a tote or a box.

plan_packages(order_id, parts) is deterministic and does not depend on the order the parts are listed in:
1. large parts go into the fewest sets of at most LARGE_PER_PACKAGE that fit a box (exact search over SKU counts;
   ties: fewest mixed-SKU sets, then the least box volume). Parts that fit no box together ship separately, and with
   LARGE_COPRINT so do parts that one plate cannot print together: the first would wait in the 2-slot large-part tray
   while the other prints, and two such packages block the tray (fleet DES, all SKUs: most of the parks). With
   today's SKUs no two large parts fit one 3 h plate, so every large part ships in its own package;
2. small parts, grouped by SKU, go first-fit-decreasing into the fewest totes, ceil(n / TOTE_MAX_PARTS); a group
   bigger than a tote fills whole totes first, and a group is split only where it fits no tote whole (same-SKU
   parts stay together where possible);
3. n = max(large sets, totes) packages: totes and large sets are paired to minimise the total box volume;
4. box = the smallest BOX_TYPES entry that fits (fit()). A package that fits no box sheds small parts into extra
   small-only packages (split further); with today's catalogue this never happens.

Box fit, fit(): rules for a loose-fill shipping box. The placing rule is from the station's place trials in the sim
(scripts/station_place_check.py); the rest are ASSUMPTIONS:
- the arm places a large part top-down with the jaws closed across its narrow side, so the fingers stand beside the
  part's long sides and their roots reach the rim of a deep box. Large parts therefore lie flat as printed with their
  long side along box x, centred across the box (y = 0); two parts go end to end along x (never side by side
  across), BOX_CLEAR_MM apart and from the end walls;
- the room beside the long sides, (inner y - part width) / 2, must be at least jaw_room_mm(inner z): LARGE_JAW_ROOM_MM
  in a box LARGE_JAW_DEPTH_MM deep, linear in depth. Measured: box M (90 deep) placed at 37.5 mm and failed at
  32.5 mm; box S (70 deep) placed a housing at 27.5 mm and a plate at 30 mm. The rule gives S 26.4, M 34.0 mm; L (110
  deep, 41.6 mm) is extrapolated, not measured;
- with small parts, the tote's drop opening (TOTE_OPENING_MM square) lies inside the inner floor;
- total part envelope (bounding-box) volume <= C.BOX_FILL x inner volume;
- tallest large part + the small-part pile (pile_mm) <= inner height (the pile is assumed to land on top).
Box frame at the station: origin at the centre of the inner floor, x along the box's long side (BOX_TYPES[box][0]) =
world +y, y along the short side = world -x; smaller x is nearer the arm. Fit.layout says where each large part goes:
its centre (mm) and yaw (0: the part's footprint[0] along box x; the long side is always along x); in a two-part row
the first part of the package's `large` is nearer the arm.

    python -m jobs.packaging        # fit table of the large SKUs (docs)
"""
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import itertools
import math

from farm import catalog as C
from jobs.plates import MAX_PLATE_S, PURGE_TAB_CM3

BOX_CLEAR_MM = 10.0          # large part <-> part / end-wall gap along box x (arm place tolerance)
LARGE_JAW_ROOM_MM = 34.0     # room beside a large part's long sides for the gripper fingers, in a box ...
LARGE_JAW_DEPTH_MM = 90.0    # ... this deep (box M: placed at 37.5 mm, failed at 32.5); scaled linearly with depth
TOTE_WALL_MM = 3.0           # farm.scene._bin wall
TOTE_OPENING_MM = 1000.0 * min(C.BIN_SIZE[:2]) - 2 * TOTE_WALL_MM   # trap-door drop opening, ~104 mm square
PAIRING_MAX = 6              # totes <-> large sets paired exhaustively up to this many packages (in order beyond)
LARGE_COPRINT = True         # large parts share a package only if one plate prints them together (module docstring)


@dataclass(frozen=True)
class Package:
    package_id: str     # "<order_id>-p<k>"
    order_id: str
    small: tuple        # part names that go into the package's tote
    large: tuple        # part names carried in the large-part tray
    box: str            # key of C.BOX_TYPES

    @property
    def parts(self):
        return self.small + self.large


@dataclass(frozen=True)
class Fit:
    """How a package's parts sit in one box (module docstring)."""
    box: str
    layout: tuple           # per large part, in the order given: (x_mm, y_mm, yaw_deg) of its centre, box frame
    end_gap_mm: float       # large parts <-> end walls along box x (>= BOX_CLEAR_MM); None without large parts
    jaw_room_mm: float      # smallest room beside a large part's long sides (>= jaw_need_mm); None without large
    jaw_need_mm: float      # jaw_room_mm(inner z) of this box
    opening_gap_mm: float   # (inner short side - TOTE_OPENING_MM) / 2, rim margin around the drop; None without small
    fill: float             # part envelope volume / inner volume (<= C.BOX_FILL)
    height_mm: float        # tallest large part + small-part pile (<= inner height)
    pile_mm: float


def part_sku(name):
    return name.rsplit("_", 2)[1]


def _skus(parts):
    return [part_sku(p) for p in parts]


def envelope(sku):
    """(x, y, z) mm bounding box of a SKU as printed."""
    s = C.CATALOG[sku]
    return (*s.footprint, s.height)


def envelope_cm3(sku):
    return math.prod(envelope(sku)) / 1000.0


def jaw_room_mm(depth_mm):
    """Room the fingers need beside a large part's long sides in a box this deep (module docstring)."""
    return LARGE_JAW_ROOM_MM * depth_mm / LARGE_JAW_DEPTH_MM


def pile_mm(small, floor_mm2, fill=None):
    """Height of the loose pile a tote drops (ASSUMPTION): its bulk (envelope volume / BOX_FILL) spread over the
    floor, but at least the thickest part lying on its largest face."""
    if not small:
        return 0.0
    bulk = sum(envelope_cm3(s) for s in small) * 1000.0 / (C.BOX_FILL if fill is None else fill)
    return max(max(min(envelope(s)) for s in small), bulk / floor_mm2)


def _row(large, w, d, c, jaw):
    """Large SKUs end to end along box x (long sides along x), centred -> (end_gap, jaw_room, layout) or None."""
    fp = [C.CATALOG[s].footprint for s in large]
    dims = [(max(f), min(f)) for f in fp]                   # (along x, across)
    length = sum(a for a, _ in dims) + c * (len(dims) - 1)
    end, room = (w - length) / 2, (d - max(b for _, b in dims)) / 2
    if end < c - 1e-9 or room < jaw - 1e-9:
        return None
    layout, x = [], -length / 2
    for (a, _), f in zip(dims, fp):
        layout.append((round(x + a / 2, 1), 0.0, 0 if f[0] >= f[1] else 90))
        x += a + c
    return round(end, 1), round(room, 1), tuple(layout)


@lru_cache(maxsize=None)
def _fit(dims, fill, clear, jaw, opening, small, large):
    w, d, h = dims
    if small and opening > min(w, d):
        return None
    vol = sum(envelope_cm3(s) for s in small + large) * 1000.0
    if vol > fill * w * d * h:
        return None
    row = (None, None, ())
    if large:
        row = _row(large, w, d, clear, jaw)
        if row is None:
            return None
    pile = pile_mm(small, w * d, fill)
    height = max((C.CATALOG[s].height for s in large), default=0.0) + pile
    if height > h:
        return None
    end, room, layout = row
    return (layout, end, room, round(jaw, 1), round((min(w, d) - opening) / 2, 1) if small else None,
            round(vol / (w * d * h), 3), round(float(height), 1), round(float(pile), 1))


def fit(box, small=(), large=()):
    """Fit of parts (SKU names) in BOX_TYPES[box] -> Fit, or None if they do not fit."""
    dims = tuple(C.BOX_TYPES[box])
    got = _fit(dims, C.BOX_FILL, BOX_CLEAR_MM, jaw_room_mm(dims[2]), TOTE_OPENING_MM, tuple(sorted(small)),
               tuple(large))
    return None if got is None else Fit(box, *got)


def boxes_by_size():
    return sorted(C.BOX_TYPES, key=lambda b: math.prod(C.BOX_TYPES[b]))


def box_for(small=(), large=()):
    """Fit in the smallest box that takes the parts (SKU names), or None."""
    for b in boxes_by_size():
        f = fit(b, small, large)
        if f is not None:
            return f
    return None


def package_fit(pkg):
    """Fit of a planned package in its box; layout entries follow pkg.large (where the arm places each large part)."""
    return fit(pkg.box, _skus(pkg.small), _skus(pkg.large))


def coprint(skus, max_print_s=MAX_PLATE_S):
    """One plate prints these SKUs together within max_print_s (purge tab included; print volume only)."""
    return C.print_time_s(PURGE_TAB_CM3 + sum(C.CATALOG[s].volume_cm3 for s in skus)) <= max_print_s


def large_pair_ok(skus):
    """These large SKUs may share a package: a box takes them and (LARGE_COPRINT) one plate prints them."""
    return box_for((), skus) is not None and (len(skus) < 2 or not LARGE_COPRINT or coprint(skus))


def _key(name):
    k = name.rsplit("_", 2)[2]
    return part_sku(name), int(k) if k.isdigit() else -1, k


def _groups(parts):
    """Parts grouped by SKU, biggest group first (then by SKU name)."""
    g = {}
    for p in parts:
        g.setdefault(part_sku(p), []).append(p)
    return sorted(g.values(), key=lambda ps: (-len(ps), part_sku(ps[0])))


def _large_sets(large):
    """Large parts -> the fewest sets of <= LARGE_PER_PACKAGE that fit a box (and with LARGE_COPRINT print on one
    plate). Exact search over SKU counts; ties:
    fewest mixed-SKU sets, then the least box volume, then the first found."""
    groups = _groups(large)
    skus = [part_sku(g[0]) for g in groups]
    for s, g in zip(skus, groups):
        if box_for((), [s]) is None:
            raise ValueError(f"{g[0]}: fits no box in BOX_TYPES")
    kinds = {}                                               # SKU-index tuple -> box volume of the set alone
    for n in range(1, C.LARGE_PER_PACKAGE + 1):
        for t in itertools.combinations_with_replacement(range(len(skus)), n):
            f = box_for((), [skus[i] for i in t])
            if f is not None and (n == 1 or not LARGE_COPRINT or coprint([skus[i] for i in t])):
                kinds[t] = math.prod(C.BOX_TYPES[f.box])

    @lru_cache(maxsize=None)
    def best(counts):
        if not any(counts):
            return (0, 0, 0.0), ()
        i0 = next(i for i, n in enumerate(counts) if n)      # some set holds a part of the first SKU left
        out = None
        for t, vol in kinds.items():
            need = Counter(t)
            if i0 not in need or any(counts[i] < k for i, k in need.items()):
                continue
            (n, mixed, v), sets = best(tuple(c - need[i] for i, c in enumerate(counts)))
            cost = (n + 1, mixed + (len(need) > 1), v + vol)
            if out is None or cost < out[0]:
                out = (cost, (t,) + sets)
        return out

    left = [list(g) for g in groups]
    sets = sorted(best(tuple(map(len, groups)))[1], key=lambda t: -len(t))   # fullest sets first (stable)
    return [[left[i].pop(0) for i in t] for t in sets]


def _totes(small):
    """Small parts -> the fewest totes of TOTE_MAX_PARTS, same-SKU parts together where possible."""
    cap = C.TOTE_MAX_PARTS
    bins, rest, k = [[] for _ in range(-(-len(small) // cap))], [], 0
    for g in _groups(small):
        while len(g) >= cap:                         # a whole tote of one SKU
            bins[k], g, k = g[:cap], g[cap:], k + 1
        if g:
            rest.append(g)
    for g in sorted(rest, key=lambda ps: (-len(ps), part_sku(ps[0]))):
        b = next((b for b in bins if len(b) + len(g) <= cap), None)
        if b is not None:
            b.extend(g)
            continue
        for b in sorted(bins, key=len):              # fits no tote whole: split it, emptiest totes first
            take = cap - len(b)
            b.extend(g[:take])
            g = g[take:]
            if not g:
                break
    return bins


def _pair(totes, sets):
    """Totes x large sets -> n = max(len) packages, paired for the least total box volume (first best = stable)."""
    n = max(len(totes), len(sets))
    totes = totes + [[] for _ in range(n - len(totes))]
    sets = sets + [[] for _ in range(n - len(sets))]

    def cost(perm):
        fits = [box_for(_skus(totes[i]), _skus(sets[j])) for i, j in enumerate(perm)]
        return sum(f is None for f in fits), sum(math.prod(C.BOX_TYPES[f.box]) for f in fits if f is not None)

    perms = itertools.permutations(range(n)) if n <= PAIRING_MAX else [tuple(range(n))]
    best = min(perms, key=cost)
    return [(list(totes[i]), list(sets[j])) for i, j in enumerate(best)]


def _small_only(ps):
    if box_for(_skus(ps), ()) is not None:
        return [ps]
    if len(ps) == 1:
        raise ValueError(f"{ps[0]}: fits no box in BOX_TYPES")
    h = (len(ps) + 1) // 2
    return _small_only(ps[:h]) + _small_only(ps[h:])


def plan_packages(order_id, parts):
    """parts: part names '<order>_<sku>_<k>' of ONE order -> [Package] (deterministic; module docstring)."""
    parts = sorted(parts, key=_key)
    if len(set(parts)) != len(parts):
        raise ValueError(f"{order_id}: duplicate parts")
    other = [p for p in parts if p.rsplit("_", 2)[0] != order_id]
    if other:
        raise ValueError(f"{order_id}: parts of another order {other}")
    small = [p for p in parts if not C.CATALOG[part_sku(p)].large]
    large = [p for p in parts if C.CATALOG[part_sku(p)].large]
    planned, spill = [], []
    for s, lg in _pair(_totes(small), _large_sets(large)):
        while s and box_for(_skus(s), _skus(lg)) is None:   # split further: shed small parts
            spill.insert(0, s.pop())
        if s or lg:
            planned.append((s, lg))
    for t in _totes(spill):
        planned += [(ps, []) for ps in _small_only(t)]
    return [Package(f"{order_id}-p{k}", order_id, tuple(s), tuple(lg), box_for(_skus(s), _skus(lg)).box)
            for k, (s, lg) in enumerate(planned)]


def main():
    print(f"boxes (inner mm): {C.BOX_TYPES}, fill <= {C.BOX_FILL}, end clearance {BOX_CLEAR_MM} mm, jaw room "
          + ", ".join(f"{b} {jaw_room_mm(C.BOX_TYPES[b][2]):.1f}" for b in boxes_by_size())
          + f" mm, tote opening {TOTE_OPENING_MM:.0f} mm")
    for s in C.LARGE_SKUS:
        print(f"  {s:8} {envelope(s)} mm")
    print(f"{'large parts':28} {'box':>5} {'end gap':>7} {'jaw room':>8} {'fill':>5} {'height':>6}   layout (x, y, yaw)")
    combos = [(s,) for s in C.LARGE_SKUS] + list(itertools.combinations_with_replacement(C.LARGE_SKUS, 2))
    for large in combos:
        for small in ((), ("cube",) * C.TOTE_MAX_PARTS):
            label = " + ".join(large) + (f" + {len(small)} small" if small else "")
            f = box_for(small, large)
            if f is None:
                print(f"{label:28} split: separate packages")
                continue
            print(f"{label:28} {f.box:>5} {f.end_gap_mm:>7} {f.jaw_room_mm:>8} {f.fill:>5} {f.height_mm:>6}   {f.layout}")


if __name__ == "__main__":
    main()
