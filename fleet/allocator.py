"""Package-aware allocation of the on-carriage caddy (pure Python).

Capacity:
- totes "o0".."o5" (`n_slots`, more in fleet studies): a tote holds the small parts of ONE package (trap-door bottom:
  the station drops them into the package's box);
- large-part tray slots "L0", "L1" (`n_large`; farm.catalog.LARGE_SLOTS, two-row cell only): a slot holds ONE large
  part, reserved for that part when slot_for hands it out;
- "reject" (defective parts, emptied by an operator) and "waste" (purge tabs, dumped at the scrap chute).
Packages come from the order book (info["packages"], jobs.packaging.plan_packages; planned here if missing).
Packages never mix: a tote holds parts of one package, and a package's parts only go into its own tote / slots.
The one-row cell has no tray (n_large = 0) and orders only small SKUs: one tote per package, as the order bins were.

    alloc = CaddyAllocator(order_book=None, n_slots=6, n_large=len(C.LARGE_SLOTS))
    alloc.add_order(order_id, info)            # info: {"parts": [...], "packages": (Package, ...), "carrier": ...}
    slot = alloc.slot_for(part, defective=False)
        # "waste" purge tab | "reject" defective | "o<k>" the tote of the part's package (the first small part of a
        # package takes the cheapest free tote) | "L<k>" a free tray slot, reserved for this large part (asking again
        # returns the same slot) | None: no free tote / tray slot -> hand off ready packages or park, then ask again
    pid = alloc.placed(part, slot)             # record the drop -> package_id if that package is now complete
    for r in alloc.handoff_ready():            # [ReadyPackage], oldest order first, then package index
        r.package                              # jobs.packaging.Package (.package_id, .order_id, .small, .large, .box)
        r.tote                                 # "o<k>" holding its small parts, or None (no small parts on the caddy)
        r.large_slots                          # ("L0", ...) holding its large parts (tray order)
        r.shelf                                # parts parked earlier on the station's consolidation shelf
        oid = alloc.handed_off(r.package.package_id)   # frees tote + tray slots -> order_id if that order has now
                                                       # shipped completely (all its packages), else None
    pid = alloc.park(slot)                     # deadlock breaker: an INCOMPLETE package's tote or tray slot goes to
                                               # the consolidation shelf; its later parts get a new tote / slot
    alloc.unreserve(part)                      # the part asked for a slot but will not be dropped (failed pick ->
                                               # reprint): frees its tray slot, or its package's tote while empty

Queries: free_slots() (free totes, cheapest first), free_large(), plate_effect(parts), remaining(order_id),
remaining_package(pid), package_of(part), order_shipped(order_id), `done` (shipped orders), `shipped` (packages),
`contents` (slot -> parts), `metrics`.
"""
from typing import NamedTuple

from farm import catalog as C
from jobs.packaging import plan_packages

SLOT_COST_ORDER = ("o2", "o3", "o0", "o5", "o1", "o4")   # closest to the arm base first
assert len(SLOT_COST_ORDER) == C.CADDY_ORDER_SLOTS


def is_purge(part_name):
    return part_name.endswith("purge_tab")


def slot_cost_order(n_slots=C.CADDY_ORDER_SLOTS):
    """Totes, cheapest first: today's 6-tote caddy uses SLOT_COST_ORDER; a larger caddy (fleet studies)
    appends o6, o7, ... after it."""
    return SLOT_COST_ORDER[:n_slots] if n_slots <= len(SLOT_COST_ORDER) else \
        SLOT_COST_ORDER + tuple(f"o{k}" for k in range(len(SLOT_COST_ORDER), n_slots))


def large_slot_order(n_large=None):
    """Tray slots, in farm.catalog.LARGE_SLOTS order (nearest the arm first), then L<k> for a larger tray."""
    base = tuple(C.LARGE_SLOTS)
    n = len(base) if n_large is None else n_large
    return base[:n] + tuple(f"L{k}" for k in range(len(base), n))


class ReadyPackage(NamedTuple):
    package: object          # jobs.packaging.Package
    tote: str                # "o<k>" or None
    large_slots: tuple       # ("L0", ...)
    shelf: tuple             # parts on the consolidation shelf (parked earlier)


class PlateEffect(NamedTuple):
    """What unloading a plate would do to the caddy (for the scheduler)."""
    new_totes: int           # packages on the plate whose small parts would need a fresh tote
    new_large: int           # large parts on the plate without a reserved tray slot (each needs a free one)
    completes: int           # packages whose remaining parts are all on the plate
    frees: int               # ... of which already hold a tote / tray slot (a hand-off frees them)
    open_large: int = 0      # new_large parts whose package is still incomplete after the plate (they stay in the tray)


class CaddyAllocator:
    def __init__(self, order_book=None, n_slots=C.CADDY_ORDER_SLOTS, n_large=None):
        self.totes, self.large = slot_cost_order(n_slots), large_slot_order(n_large)
        self.book, self.seq = {}, {}                     # order_id -> info / arrival index
        self.packages, self.order_pkgs, self.pkg_rank = {}, {}, {}
        self.part_pkg, self.part_order, self.large_parts = {}, {}, set()
        self.placed_parts = {}                           # package_id -> parts dropped into its tote / slots
        self.tote_pkg, self.pkg_tote = {}, {}            # tote <-> package
        self.large_part, self.part_large = {}, {}        # tray slot <-> large part (reserved or placed)
        self.shelf = {}                                  # package_id -> parts parked on the consolidation shelf
        self.complete, self.shipped, self.done = set(), set(), set()   # packages ready / shipped; orders shipped
        self.contents = {s: [] for s in (*self.totes, *self.large, "reject", "waste")}
        self.history = []                                # (event, slot | "shelf", package_id, [parts])
        self.metrics = {"none_events": 0, "none_large": 0, "max_totes_used": 0, "max_large_used": 0, "placed": 0,
                        "waste": 0, "reject": 0, "packages_shipped": 0, "orders_shipped": 0, "parks": 0,
                        "parks_large": 0}
        for oid, info in (order_book or {}).items():
            self.add_order(oid, info)

    def add_order(self, order_id, info):
        assert order_id not in self.book, order_id
        pkgs = tuple(info.get("packages") or plan_packages(order_id, info["parts"]))
        assert sorted(p for k in pkgs for p in k.parts) == sorted(info["parts"]), order_id
        self.book[order_id], self.seq[order_id] = info, len(self.seq)
        self.order_pkgs[order_id] = [k.package_id for k in pkgs]
        for i, k in enumerate(pkgs):
            assert k.order_id == order_id and k.package_id not in self.packages, k.package_id
            self.packages[k.package_id], self.pkg_rank[k.package_id] = k, (self.seq[order_id], i)
            self.placed_parts[k.package_id] = set()
            for p in k.parts:
                assert p not in self.part_pkg, p
                self.part_pkg[p], self.part_order[p] = k.package_id, order_id
            self.large_parts.update(k.large)

    # ---- queries ----
    def free_slots(self):
        """Free totes, cheapest first."""
        return [s for s in self.totes if s not in self.tote_pkg]

    def free_large(self):
        return [s for s in self.large if s not in self.large_part]

    def package_of(self, part_name):
        return self.packages[self.part_pkg[part_name]]

    def remaining_package(self, package_id):
        return set(self.packages[package_id].parts) - self.placed_parts[package_id]

    def remaining(self, order_id):
        """Parts of the order not yet dropped into the caddy."""
        return set().union(*(self.remaining_package(k) for k in self.order_pkgs[order_id]))

    def order_shipped(self, order_id):
        return order_id in self.done

    def large_slots_of(self, package_id):
        return tuple(s for s in self.large if s in self.large_part and self.part_pkg[self.large_part[s]] == package_id)

    def holds_slot(self, package_id):
        return package_id in self.pkg_tote or bool(self.large_slots_of(package_id))

    @property
    def slot_order(self):
        """tote -> order_id (LEGACY view for farm/hud.py)."""
        return {t: self.packages[k].order_id for t, k in self.tote_pkg.items()}

    def plate_effect(self, parts):
        """PlateEffect of dropping these parts (a plate's names; purge tabs ignored) into the caddy now."""
        parts = [p for p in parts if not is_purge(p)]
        pids = {self.part_pkg[p] for p in parts}
        small_pids = {self.part_pkg[p] for p in parts if p not in self.large_parts}
        on = set(parts)
        comp = [k for k in pids if self.remaining_package(k) <= on]
        new_large = [p for p in parts if p in self.large_parts and p not in self.part_large]
        return PlateEffect(sum(k not in self.pkg_tote for k in small_pids), len(new_large), len(comp),
                           sum(self.holds_slot(k) for k in comp),
                           sum(self.part_pkg[p] not in comp for p in new_large))

    # ---- actions ----
    def slot_for(self, part_name, defective=False):
        """Slot for the part in hand (module docstring); None = no free tote / tray slot."""
        if is_purge(part_name):
            return "waste"
        if defective:
            return "reject"
        pid = self.part_pkg[part_name]
        if pid in self.shipped or part_name in self.placed_parts[pid]:
            raise ValueError(f"{part_name}: package {pid} already handed off or part already placed")
        if part_name in self.large_parts:
            if not self.large:
                raise ValueError(f"{part_name}: large part, but this caddy has no large-part tray")
            if part_name in self.part_large:
                return self.part_large[part_name]
            free = self.free_large()
            if not free:
                self.metrics["none_events"] += 1
                self.metrics["none_large"] += 1
                return None
            self.large_part[free[0]], self.part_large[part_name] = part_name, free[0]
            self.metrics["max_large_used"] = max(self.metrics["max_large_used"], len(self.large_part))
            return free[0]
        if pid in self.pkg_tote:
            return self.pkg_tote[pid]
        free = self.free_slots()
        if not free:
            self.metrics["none_events"] += 1
            return None
        self.tote_pkg[free[0]], self.pkg_tote[pid] = pid, free[0]
        self.metrics["max_totes_used"] = max(self.metrics["max_totes_used"], len(self.tote_pkg))
        return free[0]

    def placed(self, part_name, slot):
        """Record a drop; returns the package_id if that package is now complete, else None."""
        if slot in ("waste", "reject"):
            self.contents[slot].append(part_name)
            self.metrics[slot] += 1
            return None
        pid = self.part_pkg[part_name]
        ok = self.large_part.get(slot) == part_name if slot in self.large else \
            self.tote_pkg.get(slot) == pid and part_name not in self.large_parts
        if not ok or part_name in self.placed_parts[pid]:
            raise ValueError(f"{part_name} ({pid}) dropped into {slot}, which holds "
                             f"{self.tote_pkg.get(slot) or self.large_part.get(slot)}")
        self.contents[slot].append(part_name)
        self.placed_parts[pid].add(part_name)
        self.metrics["placed"] += 1
        if self.remaining_package(pid):
            return None
        self.complete.add(pid)
        return pid

    def handoff_ready(self):
        """[ReadyPackage] for complete packages still on the caddy, oldest order first, then package index."""
        return [ReadyPackage(self.packages[k], self.pkg_tote.get(k), self.large_slots_of(k),
                             tuple(self.shelf.get(k, ())))
                for k in sorted(self.complete, key=self.pkg_rank.get)]

    def _release(self, slot, event):
        parts, self.contents[slot] = self.contents[slot], []
        if slot in self.tote_pkg:
            pid = self.tote_pkg.pop(slot)
            del self.pkg_tote[pid]
        else:
            part = self.large_part.pop(slot)
            del self.part_large[part]
            pid = self.part_pkg[part]
        self.history.append((event, slot, pid, parts))
        return pid, parts

    def handed_off(self, package_id):
        """The complete package leaves at the station (its tote, tray parts and shelf parts go into its box): frees
        its tote and tray slots. Returns the order_id if all of that order's packages have now shipped, else None."""
        if package_id not in self.complete:
            raise ValueError(f"{package_id}: not a complete package on the caddy")
        for slot in ((self.pkg_tote[package_id],) if package_id in self.pkg_tote else ()) + \
                self.large_slots_of(package_id):
            self._release(slot, "handoff")
        if package_id in self.shelf:
            self.history.append(("handoff", "shelf", package_id, self.shelf.pop(package_id)))
        self.complete.discard(package_id)
        self.shipped.add(package_id)
        self.metrics["packages_shipped"] += 1
        oid = self.packages[package_id].order_id
        if not all(k in self.shipped for k in self.order_pkgs[oid]):
            return None
        self.done.add(oid)
        self.metrics["orders_shipped"] += 1
        return oid

    def park(self, slot):
        """An incomplete package's tote or tray slot to the consolidation shelf (its parts wait there until the
        package ships; its later parts get a new tote / slot). Returns the package_id."""
        pid = self.tote_pkg.get(slot) or (self.part_pkg[self.large_part[slot]] if slot in self.large_part else None)
        if pid is None:
            raise ValueError(f"slot {slot} holds no package")
        if pid in self.complete:
            raise ValueError(f"slot {slot} holds complete package {pid}: hand it off instead")
        _, parts = self._release(slot, "park")
        if parts:
            self.shelf.setdefault(pid, []).extend(parts)
        self.metrics["parks"] += 1
        self.metrics["parks_large"] += slot in self.large
        return pid

    def unreserve(self, part_name):
        """The part in hand will not be dropped now (failed pick -> reprint): free its tray-slot reservation, or its
        package's tote while that tote is still empty. Returns the freed slot or None."""
        if part_name in self.part_large and part_name not in self.placed_parts[self.part_pkg[part_name]]:
            slot = self.part_large[part_name]
            self._release(slot, "unreserve")
            return slot
        tote = self.pkg_tote.get(self.part_pkg.get(part_name))
        if tote is not None and part_name not in self.large_parts and not self.contents[tote]:
            self._release(tote, "unreserve")
            return tote
        return None
