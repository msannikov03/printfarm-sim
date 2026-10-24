"""Package-aware plate composition (pure Python): the order in which queued parts are offered to the plate packer.

Why: in fleet/des.py (2x10 cell, small SKU mix, order_scale 2, FIFO queue) at a caddy-full event 81 % of the held
totes belonged to packages whose remaining parts were printing on ANOTHER plate (12 % were complete, 4 % had the rest
on the plate being unloaded, 3 % still queued). 30 % of packages were printed across 2+ plates, started 1.2 h apart on
average. The packer (jobs/plates.py pack) fills a plate's spare print time with a fragment of a package too big for one
plate, and the rest of that package keeps its FIFO place behind other orders, so its tote stays open for hours. Bulk
orders held only 22 % of those totes, so more totes would not remove the cause.

order_queue(queue, alloc, max_print_s) -> the queue as the packer should see it:
1. split packages first: parts of packages that are partly printed (some parts on a plate or in the caddy, some still
   queued) go to the front, in queue order, so the next plate finishes them;
2. admission: while MAX_SPLIT or more packages are split, a package not started yet that needs more than one plate
   (more print volume than one plate takes in max_print_s) is held back; it starts once the split ones are done;
3. the packer groups by package instead of order (each entry's "order_id" becomes its package_id, the packer's group
   key), so a package that fits one plate is never split and an order's packages print independently;
4. large parts last (large_last): a package's large parts wait while any of its small parts is still queued, so a
   large part reaches the 2-slot tray when its package is (nearly) complete instead of waiting there for a tote.
   With all SKUs at matched demand this, the scheduler's tray rule (fleet.scheduler) and one large part per package
   (jobs.packaging LARGE_COPRINT) took parks from 15.6 to 1.0 per week (fleet/des.py, 5 seeds).
Everything else keeps its FIFO order. Entries may be part names (fleet/des.py) or {"name", "order_id", "sku"} dicts
(sim/farm.py); the result is a new list of dicts, the caller's queue is not changed.

    plate, _ = make_plate(order_queue(self.queue, self.alloc, MAX_PLATE_S), ...)      # sim/farm.py _start_print
"""
from functools import lru_cache

from farm.catalog import CATALOG, print_time_s
from jobs.plates import MAX_PLATE_S, PURGE_TAB_CM3

MAX_SPLIT = 2           # packages allowed to be split across plates at once before big new ones wait (DES-tuned)


def _name(q):
    return q["name"] if isinstance(q, dict) else q


def _sku(name):
    return name.rsplit("_", 2)[1]


@lru_cache(maxsize=None)
def plate_volume_cm3(max_print_s=MAX_PLATE_S):
    """Part volume one plate prints within max_print_s (print_time_s inverted, purge tab included)."""
    lo, hi = 0.0, 1.0
    while print_time_s(PURGE_TAB_CM3 + hi) <= max_print_s:
        lo, hi = hi, 2 * hi
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if print_time_s(PURGE_TAB_CM3 + mid) <= max_print_s else (lo, mid)
    return lo


def package_volume_cm3(pkg):
    return sum(CATALOG[_sku(p)].volume_cm3 for p in pkg.parts)


def split_packages(names, alloc):
    """Package ids with parts still queued (`names`) and parts already out of the queue (on a plate, in the caddy)."""
    queued = set(names)
    pids = {alloc.part_pkg[n] for n in queued if n in alloc.part_pkg}
    return {k for k in pids if any(p not in queued for p in alloc.packages[k].parts)}


def order_queue(queue, alloc, max_print_s=MAX_PLATE_S, max_split=MAX_SPLIT, large_last=True):
    """Queue (names or dicts, FIFO) -> [{"name", "order_id": package_id, "sku"}] in the order the plate packer should
    take them (module docstring). Parts the allocator does not know keep their own group and FIFO place."""
    names = [_name(q) for q in queue]
    queued = set(names)
    split = split_packages(names, alloc)
    cap = plate_volume_cm3(max_print_s)
    pid = {n: alloc.part_pkg.get(n) for n in names}

    def held(n):
        k = pid[n]
        if k is None:
            return False
        if large_last and n in alloc.large_parts and any(p in queued for p in alloc.packages[k].small):
            return True
        return len(split) >= max_split and k not in split and package_volume_cm3(alloc.packages[k]) > cap

    first = [q for q, n in zip(queue, names) if pid[n] in split and not held(n)]
    rest = [q for q, n in zip(queue, names) if pid[n] not in split and not held(n)]
    out = []
    for q in first + rest:
        n = _name(q)
        d = dict(q) if isinstance(q, dict) else {"name": n, "order_id": n.rsplit("_", 2)[0], "sku": _sku(n)}
        if pid[n] is not None:
            d["order_id"] = pid[n]
        out.append(d)
    return out
