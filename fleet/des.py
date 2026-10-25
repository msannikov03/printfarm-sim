"""Discrete-event simulation of a PrintPick cell (one arm on a rail): robot policies vs a human baseline.

Time is seconds from Monday 00:00. Every rate and duration here is an ASSUMPTION (Params), except the robot
timings, which come from runs/timings.json (MuJoCo side) when it exists and from DESIGN.md defaults otherwise, and the
packing-station step times, from runs/station_times.json (MuJoCo station, plan/station.py; DEFAULT_STATION otherwise).
Orders arrive during [0, days); the run then drains without new orders so every order gets a lead time.
Robot plates are composed package-aware (fleet.queue_policy: split packages first, big new packages wait while two
are split, a package's large parts after its small parts, grouped by package), which keeps totes and tray slots from
being held open across plates; order_aware keeps the last tray slot for packages that complete and waits (TRAY_HOLD_S
at most) rather than unload into a full tray (fleet.scheduler). The human packs FIFO by order.
Bed to box: an order ships as one or more packages (jobs.packaging); the caddy holds one package per tote (order_slots)
plus one large part per tray slot (large_slots) (fleet.allocator). At the station the robot docks, then per ready
package: box request + one place per large part + tote dump; the box seals and rides to its carrier lane in the
background. An order has shipped when its last package is in its lane (lead time). Purge tabs are dumped from the
waste bin at the station. Station step times are means (their spread is not modelled).
Printer, robot and counter KPIs are measured over [0, days).
The cell geometry is a Layout (printer positions along the rail + the hand-off station). By default it is today's
cell from farm.catalog (one row of N_PRINTERS along a wall); `make_layout` builds one row of any length or two
facing rows with the rail in the middle (scripts/layout_study.py sweeps printers per arm).

    python -m fleet.des --days 7 --policy order_aware --seed 0 --out runs/des/
    python -m fleet.des --days 7 --layout facing --n-printers 30 --order-scale 3
    python -m fleet.des --compare --days 7 --out runs/des_box/small
    python -m fleet.des --compare --days 7 --skus all --out runs/des_box/all
"""
import argparse, csv, heapq, json, math, time, zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
import numpy as np

from farm.catalog import (CADDY_ORDER_SLOTS, CATALOG, LARGE_SLOTS, N_PRINTERS, N_ROWS, PITCH, RAIL_X_MAX, RAIL_X_MIN,
                          SMALL_SKUS, STATION_X, print_time_s, printer_frame)
from fleet.allocator import CaddyAllocator
from fleet.queue_policy import order_queue
from fleet.scheduler import (POLICIES, TRAY_HOLD_S, WAIT_FOR_RELEASE, Candidate, Snapshot, choose, cool_wait_s,
                             shortfall, travel_s)
from jobs.orders import OrderStream, part_sku
from jobs.plates import MAX_PLATE_S, PURGE_TAB_CM3, make_plate

DAY = 86400.0
STATES = ("printing", "cooling", "waiting", "unloading", "idle", "blocked")
DEFAULT_TIMINGS = {"rail_speed_mps": 0.4, "door_open_s": 14.0, "door_close_s": 12.0, "scan_s": 4.0,
                   "purge_tab_s": 12.0, "pick_place_s_mean": 11.0, "pick_place_s_std": 2.0,
                   "bed_check_s": 3.0, "station_handoff_s": 6.0, "retry_s": 8.0,
                   "p_success_first_try": 0.95, "p_intervention_per_part": 0.01, "p_reprint_per_part": 0.0,
                   "rake_s": 15.0}
# packing-station step means (sim s), runs/station_times.json at e169b23 (n = 17..56): the fallback if the file is
# missing. Steps as plan/station.py Station: dock = carriage between station positions; box_request = request -> box
# in position, while the carriage docks (includes waiting for the previous box to enter the sealer, ~7.7 s when the
# line is free); large_place per large part; to_tote + tote_dump (trap door + camera check) per tote; seal = box ->
# sealer -> sealed, in the background; package = measured robot-blocking time per package (scheduler estimate).
DEFAULT_STATION = {"dock": 4.75, "box_request": 11.58, "large_place": 16.81, "to_tote": 1.73, "tote_dump": 2.87,
                   "seal": 10.76, "package": 21.01}
LAYOUTS = ("row", "facing")
STATIONS = ("end", "center")
SKU_MIXES = ("small", "all")


@dataclass(frozen=True)
class Layout:
    """Where the printers and the hand-off station sit along the rail (rail coordinate x, m).
    frames: (x_centre, side) per printer; side +1 = row A (+y), -1 = row B (-y), as farm.catalog.printer_frame."""
    kind: str
    frames: tuple
    station_x: float
    rail_min: float
    rail_max: float
    x_open_dx: float = 0.15            # carriage offset to the printer's right for the door pull

    @property
    def n(self):
        return len(self.frames)

    @property
    def rail_length_m(self):
        return self.rail_max - self.rail_min

    @property
    def x_open(self):
        return [min(self.rail_max, max(self.rail_min, xc + s * self.x_open_dx)) for xc, s in self.frames]


def make_layout(kind="row", n=N_PRINTERS, station="end", pitch=PITCH, x_open_dx=0.15):
    """row:    one row of n printers along a wall (today's cell: n = N_PRINTERS, station "end").
    facing: two rows facing each other across the rail: ceil(n/2) columns, row A first (as printer_frame).
    station "end":    the hand-off station 0.75 m past the last column (as farm.catalog.STATION_X).
            "center": mid-rail. One row leaves the far side of the aisle free, so the station stands there;
                      two facing rows give up one column of bench to it (later columns move one pitch along)."""
    assert kind in LAYOUTS and station in STATIONS and n >= 1, (kind, station, n)
    cols = n if kind == "row" else math.ceil(n / 2)
    frames = [((i % cols) * pitch, +1 if i < cols else -1) for i in range(n)]
    if station == "end":
        station_x = (cols - 1) * pitch + 0.75
    elif kind == "row":
        station_x = (cols - 1) * pitch / 2
    else:
        gap = math.ceil(cols / 2)
        frames = [(x + pitch * ((i % cols) >= gap), s) for i, (x, s) in enumerate(frames)]
        station_x = gap * pitch
    xs = [x for x, _ in frames]
    rail_min = min(xs) - 0.45
    rail_max = max(max(xs) + 0.45, station_x + 0.2)
    return Layout(kind, tuple(frames), float(station_x), float(rail_min), float(rail_max), x_open_dx)


def default_layout(x_open_dx=0.15):
    """Today's cell, as farm.catalog defines it (PRINTFARM_LAYOUT: one row, or two facing rows)."""
    return Layout("row" if N_ROWS == 1 else "facing", tuple(map(printer_frame, range(N_PRINTERS))), STATION_X,
                  RAIL_X_MIN, RAIL_X_MAX, x_open_dx)


def load_timings(path="runs/timings.json"):
    t = dict(DEFAULT_TIMINGS)
    if Path(path).exists():
        t.update(json.loads(Path(path).read_text()))
        return t, str(path)
    return t, "DESIGN.md defaults"


def load_station_times(path="runs/station_times.json"):
    """Station step means (sim s) -> (dict, source). Steps missing from the file come from DEFAULT_STATION;
    waste_dump (dock to the scrap bin + one flap cycle, not in the file yet) defaults to dock + tote_dump."""
    s, src = dict(DEFAULT_STATION), "DEFAULT_STATION (station_times.json of e169b23)"
    if Path(path).exists():
        s.update({k: v["mean"] if isinstance(v, dict) else float(v) for k, v in json.loads(Path(path).read_text()).items()})
        src = str(path)
    s.setdefault("waste_dump", s["dock"] + s["tote_dump"])
    return s, src


def sku_mix(mix, max_plate_s=MAX_PLATE_S):
    """Order SKU mix -> (Params.order_skus, excluded). "small": farm.catalog.SMALL_SKUS. "all": every catalogue
    SKU one plate can print; a SKU that alone (+ purge tab) prints longer than max_plate_s is never plated, so its
    orders would never ship: it is excluded and reported."""
    if mix == "small":
        return SMALL_SKUS, ()
    assert mix == "all", mix
    ok = tuple(s for s in CATALOG if print_time_s(CATALOG[s].volume_cm3 + PURGE_TAB_CM3) <= max_plate_s)
    return ok, tuple(s for s in CATALOG if s not in ok)


@dataclass
class Params:
    days: float = 7.0
    policy: str = "order_aware"        # a scheduler policy, or "human" for the manual baseline
    seed: int = 0
    order_scale: float = N_PRINTERS / 10.0   # order rate multiplier; the default keeps demand per printer constant
                                       # (the order stream is sized for 10 printers); >1 per printer: capacity run
    drain_days: float = 5.0
    max_plate_s: float = MAX_PLATE_S
    bed_c: float = 60.0                # bed cools as Newton's law, door closed (ASSUMPTIONS)
    ambient_c: float = 25.0
    cool_tau_s: float = 360.0
    release_c: float = 35.0
    p_defect: float = 0.01             # part fails the camera check -> reject bin + reprint
    x_open_dx: float = 0.15            # carriage offset to the printer's right for the door pull
    pick_clip: tuple = (0.5, 2.0)      # pick time clipped to [lo, hi] x mean
    handoff_age_s: float = 1800.0      # complete bins wait at most this long before a station run
    workdays: tuple = (0, 1, 2, 3, 4)  # Mon..Fri: days a shift starts on
    shift_h: tuple = (9.0, 18.0)       # operator hours from the start of a workday; an end past 24 runs past
                                       # midnight (9, 26 = a day shift 9-18 and an evening shift 18-02)
    human_response_s: float = 600.0    # robot mode: a person reaches a stuck pick (in shift)
    human_fix_s: float = 120.0
    check_every_s: float = 3600.0      # human baseline
    human_unload_s: float = 180.0
    human_part_s: float = 20.0
    human_pack_s: float = 60.0         # per package
    human_long_plates: bool = True     # baseline sizes a plate to end at the operator's next check
    consolidate_s: float = 30.0        # operator adds parked parts (consolidation shelf) to the box: delays its seal
    lane_s: float = 15.0               # sealer / packing bench -> carrier lane
    waste_max_tabs: int = 60           # waste bin full: purge tabs (~4 cm^3) at loose fill in a 104 x 104 x 67 mm bin
    queue_window: int = 400            # queued parts offered to the plate packer
    queue_policy: str = "package"      # robot: "package" (fleet.queue_policy.order_queue) | "fifo" (by order); the
                                       # human baseline always packs FIFO by order
    tray_aware: bool = True            # order_aware: an incomplete package's large part may not take the last tray slot
    tray_wait: bool = True             # order_aware: wait (TRAY_HOLD_S at most) instead of unloading into a full tray
    large_last: bool = True            # package queue: a package's large parts wait while its small parts are queued
    robot_slowdown: float = 1.0        # stress test: robot durations x this, rail speed / this
    layout: str = "default"            # "default" (farm.catalog's cell) | "row" | "facing" (make_layout)
    n_printers: int = N_PRINTERS       # printers served by the one arm (layouts "row" / "facing")
    station: str = "end"               # hand-off station position (make_layout)
    order_slots: int = CADDY_ORDER_SLOTS   # totes on the carriage caddy (one package each)
    large_slots: int = len(LARGE_SLOTS)    # large-part tray slots (0 in the one-row cell)
    order_skus: tuple = None           # SKUs orders draw from (None: farm.catalog.ORDER_SKUS)


@dataclass
class Printer:
    idx: int
    x_open: float
    state: str = "idle"
    plate: dict = None
    todo: list = field(default_factory=list)
    purge: bool = False
    freed: set = field(default_factory=set)    # parts a person already loosened
    t_done: float = 0.0
    t_release: float = 0.0
    claimed: bool = False                      # robot/person owns it until released
    log: list = field(default_factory=lambda: [(0.0, "idle")])


class Sim:
    def __init__(self, p: Params, timings=None, station=None):
        self.p = p
        self.T, self.timings_src = (dict(timings), "given") if timings else load_timings()
        self.S, self.station_src = (dict(station), "given") if station else load_station_times()
        for key in [k for k in self.T if k.endswith(("_s", "_s_mean", "_s_std"))]:
            self.T[key] *= p.robot_slowdown
        self.S = {k: v * p.robot_slowdown for k, v in self.S.items()}
        self.T["rail_speed_mps"] /= p.robot_slowdown
        self.human = p.policy == "human"
        assert self.human or p.policy in POLICIES, p.policy
        assert 0.0 <= p.shift_h[0] < p.shift_h[1] <= p.shift_h[0] + 24.0, p.shift_h
        self.H = p.days * DAY
        self.heap, self.seq, self.t = [], 0, 0.0
        self.cool_s = p.cool_tau_s * math.log((p.bed_c - p.ambient_c) / (p.release_c - p.ambient_c))
        self.orders = {o.order_id: o for o in OrderStream(p.seed, scale=p.order_scale, skus=p.order_skus)
                       .generate(self.H)}
        self.ost = {oid: {"t_complete": None, "t_shipped": None, "good": 0, "packed": False} for oid in self.orders}
        self.pst = {}                        # package_id -> {"t_complete", "t_shipped"}
        self.queue, self.parts, self.waste = [], {}, 0
        self.alloc = CaddyAllocator(n_slots=p.order_slots, n_large=p.large_slots)
        if p.layout == "default":
            self.layout = default_layout(p.x_open_dx)
            x_open = self.T.get("x_open_m") or self.layout.x_open
        else:
            self.layout = make_layout(p.layout, p.n_printers, p.station, x_open_dx=p.x_open_dx)
            x_open = self.layout.x_open
        self.station_x = self.layout.station_x
        self.printers = [Printer(i, x_open[i]) for i in range(self.layout.n)]
        self.robot_x, self.robot_busy, self.line_free, self.plate_n = self.station_x, False, 0.0, 0
        self.rng = np.random.default_rng([p.seed, 7])
        self.k = {k: 0.0 for k in ("parts_good", "parts_rejected", "interventions", "caddy_full_events",
                                   "station_visits", "partial_handoffs", "early_pick_fails", "retries",
                                   "rail_m", "human_s", "plates", "robot_reprints", "station_s", "waste_dumps",
                                   "packages_handed")}
        self.lanes = {}
        self.events, self.busy = [], []      # busy: (t0, t1, wait_s inside, what)
        for o in self.orders.values():
            self.push(o.t_s, "order", o.order_id)
        if self.human:
            self.push(self.next_check(0.0), "check", None)

    # ---- plumbing ----
    def push(self, t, kind, arg):
        heapq.heappush(self.heap, (t, self.seq, kind, arg))
        self.seq += 1

    def log(self, t, kind, printer="", ref="", detail=""):
        self.events.append((round(t, 3), kind, printer, ref, detail))

    def bump(self, key, t, n=1.0):
        if t < self.H:
            self.k[key] += n

    def set_state(self, pr, t, state):
        assert t >= pr.log[-1][0] - 1e-6, (pr.idx, t, pr.log[-1])
        pr.state = state
        pr.log.append((t, state))

    def in_shift(self, t):
        """Inside an operator shift. A shift belongs to the workday it starts on (it may end after midnight)."""
        day, h = divmod(t / 3600.0, 24.0)
        lo, hi = self.p.shift_h
        return any((int(day) - k) % 7 in self.p.workdays and lo <= h + 24.0 * k < hi for k in (0, 1))

    def next_check(self, t):
        """First operator check (hourly marks inside the shift, workdays) at or after t."""
        lo, hi = (h * 3600.0 for h in self.p.shift_h)
        for day in range(int(t // DAY) - 1, int(t // DAY) + 8):          # from yesterday: a shift past midnight
            if day % 7 in self.p.workdays:
                for m in np.arange(day * DAY + lo, day * DAY + hi, self.p.check_every_s):
                    if m >= t:
                        return float(m)
        raise RuntimeError("no workday in the week")

    def draws(self, name):
        """Per-print random numbers, keyed by part name: the same part sees the same luck under every policy."""
        g = np.random.default_rng([self.p.seed, zlib.crc32(name.encode()), self.parts[name]["prints"]])
        return g.random(3), g.standard_normal(2)

    # ---- run ----
    def run(self):
        t_end = self.H + self.p.drain_days * DAY
        while self.heap:
            t, _, kind, arg = heapq.heappop(self.heap)
            if t > t_end:
                break
            self.t = t
            getattr(self, "_on_" + kind)(arg)
            if not self.human and not self.robot_busy:
                self.robot_decide()
            if t >= self.H and kind in ("robot_free", "human_done", "check"):
                if all(s["t_shipped"] is not None for s in self.ost.values()):
                    break
        self.t_end = self.t
        return self.kpis()

    # ---- orders and printers ----
    def _on_order(self, oid):
        o = self.orders[oid]
        self.alloc.add_order(oid, o.book_entry())
        for n in o.parts:
            self.parts[n] = {"order_id": oid, "sku": part_sku(n), "prints": 0, "good": 0, "rejects": 0}
            self.queue.append(n)
        self.log(self.t, "order", "", oid, f"{len(o.parts)} parts, {o.carrier}, {'+'.join(o.boxes)}")
        self.start_idle()

    def _on_reprint(self, name):
        self.queue.insert(0, name)
        self.start_idle()

    def start_idle(self):
        for pr in self.printers:
            if self.queue and pr.state == "idle" and not pr.claimed:
                self.start_plate(pr)

    def next_job_s(self):
        vol = 0.0
        for n in self.queue:
            vol += CATALOG[self.parts[n]["sku"]].volume_cm3
            if print_time_s(vol) >= self.p.max_plate_s:
                return self.p.max_plate_s
        return print_time_s(vol) if self.queue else 0.0

    def start_plate(self, pr):
        max_s = self.p.max_plate_s
        if self.human and self.p.human_long_plates:
            max_s = max(max_s, self.next_check(self.t + max_s) - self.t - self.cool_s)
        if self.human or self.p.queue_policy == "fifo":
            window = [{"name": n, "order_id": self.parts[n]["order_id"], "sku": self.parts[n]["sku"]}
                      for n in self.queue[:self.p.queue_window]]
        else:
            assert self.p.queue_policy == "package", self.p.queue_policy
            window = order_queue(self.queue, self.alloc, max_s, large_last=self.p.large_last)[:self.p.queue_window]
        plate, _ = make_plate(window, pr.idx, f"P{self.plate_n:04d}", self.rng, max_print_s=max_s)
        if not plate["objects"]:
            return
        self.plate_n += 1
        self.bump("plates", self.t)
        names = {o["name"] for o in plate["objects"]}
        self.queue = [n for n in self.queue if n not in names]
        for n in names:
            self.parts[n]["prints"] += 1
        pr.plate, pr.purge, pr.freed = plate, True, set()
        pr.todo = [o["name"] for o in sorted(plate["objects"], key=lambda o: (o["center_mm"][1], o["center_mm"][0]))]
        self.set_state(pr, self.t, "printing")
        self.log(self.t, "plate_start", pr.idx, plate["plate_id"], f"{len(names)} parts, {plate['print_time_s']:.0f} s")
        self.push(self.t + plate["print_time_s"], "print_done", pr.idx)

    def _on_print_done(self, i):
        pr = self.printers[i]
        pr.t_done, pr.t_release = self.t, self.t + self.cool_s
        self.set_state(pr, self.t, "cooling")
        self.push(pr.t_release, "released", i)

    def _on_released(self, i):
        pr = self.printers[i]
        if pr.state == "cooling" and not pr.claimed:
            self.set_state(pr, self.t, "waiting")

    def part_unloaded(self, name, t, defective):
        part = self.parts[name]
        if defective:
            part["rejects"] += 1
            self.bump("parts_rejected", t)
            self.push(t, "reprint", name)
            return
        part["good"] += 1
        part["t_good"] = t
        self.bump("parts_good", t)
        oid = part["order_id"]
        self.ost[oid]["good"] += 1
        if self.ost[oid]["good"] == len(self.orders[oid].parts):
            self.ost[oid]["t_complete"] = t

    def ship(self, t, pkg, pack_s):
        """One package from t: the sealer (robot mode; in the background, one box at a time) or hand packing (baseline,
        one operator) takes pack_s, then lane_s to its carrier lane. The order has shipped when its last package has."""
        o = self.orders[pkg.order_id]
        start = max(t, self.line_free)
        self.line_free = start + pack_s
        t_ship = self.line_free + self.p.lane_s
        self.pst.setdefault(pkg.package_id, {"t_complete": t})["t_shipped"] = t_ship
        self.lanes[o.carrier] = self.lanes.get(o.carrier, 0) + 1
        self.log(t_ship, "shipped", "", pkg.package_id, f"{pkg.box} -> {o.carrier}")
        if all("t_shipped" in self.pst.get(k.package_id, {}) for k in o.packages):
            self.ost[o.order_id]["t_shipped"] = max(self.pst[k.package_id]["t_shipped"] for k in o.packages)

    # ---- robot ----
    def drive(self, t, x):
        d = abs(x - self.robot_x)
        self.bump("rail_m", t, d)
        self.robot_x = x
        return t + d / self.T["rail_speed_mps"]

    def pick_s(self, z):
        m, lo, hi = self.T["pick_place_s_mean"], *self.p.pick_clip
        return float(np.clip(m + self.T["pick_place_s_std"] * z, lo * m, hi * m))

    def robot_until(self, t, wait=0.0, what=""):
        self.robot_busy = True
        self.busy.append((self.t, t, wait, what))
        self.push(t, "robot_free", None)

    def _on_robot_free(self, _):
        self.robot_busy = False

    def snapshot(self, cands, ready):
        T, a = self.T, self.alloc
        cs = []
        for pr in cands:
            e = a.plate_effect(pr.todo)
            cs.append(Candidate(pr.idx, pr.x_open, pr.t_done, pr.t_release, len(pr.todo), e.new_totes, e.completes,
                                e.frees, e.new_large, e.open_large if self.p.tray_aware else 0))
        fixed = T["door_open_s"] + T["scan_s"] + T["purge_tab_s"] + T["bed_check_s"] + T["door_close_s"]
        return Snapshot(self.t, self.robot_x, T["rail_speed_mps"], tuple(cs), self.next_job_s(), self.p.max_plate_s,
                        fixed, T["pick_place_s_mean"], len(a.free_slots()), len(ready), self.station_x,
                        self.S["package"], len(a.free_large()) if a.large else None, self.p.tray_wait)

    def robot_decide(self):
        a = self.alloc
        ready = a.handoff_ready()
        cands = [pr for pr in self.printers if pr.state in ("cooling", "waiting") and not pr.claimed]
        if self.waste >= self.p.waste_max_tabs:
            return self.robot_until(self.station_visit(self.t), what="station")
        if ready:
            oldest = min(self.pst[r.package.package_id]["t_complete"] for r in ready)
            full = not a.free_slots() or (a.large and not a.free_large())
            if full or not cands or self.t - oldest >= self.p.handoff_age_s:
                return self.robot_until(self.station_visit(self.t), what="station")
        if not cands:
            return
        snap = self.snapshot(cands, ready)
        c = choose(self.p.policy, snap)
        if c is None:                              # order_aware waits for a tray slot (fleet.scheduler)
            return self.push(max(self.t + 1.0, min(pr.t_release for pr in cands) + TRAY_HOLD_S + 1.0), "wake", None)
        if self.p.policy == "order_aware" and shortfall(snap, c) > 0 and ready:
            return self.robot_until(self.station_visit(self.t), what="station")
        lead = self.T["door_open_s"] + self.T["scan_s"]
        if self.p.policy in WAIT_FOR_RELEASE and cool_wait_s(snap, c) > lead + 1.0:
            return self.push(c.t_release - lead - travel_s(snap, c), "wake", None)   # stay free until then
        self.unload(self.printers[c.printer])

    def _on_wake(self, _):
        pass

    def station_visit(self, t, need=None):
        """Bed to box at the station (step times: self.S, measured in plan/station.py). Dock, then per ready package
        (oldest order first): box request (overlaps the docking moves) + large_place per part in the tray + to_tote +
        tote_dump if it has a tote; the box then seals in the background (ship). Parts parked earlier are added by the
        operator (consolidate_s, delays the seal, not the robot). Then the waste bin is dumped if it holds purge tabs,
        and while `need` ("tote" | "large") has no free slot, the incomplete package furthest from complete (newest
        order on ties) is parked from a tote / tray slot (station_handoff_s each, ASSUMPTION)."""
        T, S, a = self.T, self.S, self.alloc
        t = self.drive(t, self.station_x)
        t0 = t
        self.bump("station_visits", t)
        t += S["dock"]
        for r in a.handoff_ready():
            k = r.package
            t += S["box_request"] + len(r.large_slots) * S["large_place"]
            if r.tote is not None:
                t += S["to_tote"] + S["tote_dump"]
            a.handed_off(k.package_id)
            self.bump("packages_handed", t)
            self.log(t, "handoff", "", k.package_id, f"box {k.box}, tote {r.tote or '-'}, tray {len(r.large_slots)}"
                                                      f", shelf {len(r.shelf)}")
            self.ship(t, k, S["seal"] + (self.p.consolidate_s if r.shelf else 0.0))
        if self.waste:
            t += S["waste_dump"]
            self.bump("waste_dumps", t)
            self.waste = 0
        while need is not None and not (a.free_slots() if need == "tote" else a.free_large()):
            held = a.tote_pkg if need == "tote" else {s: a.part_pkg[q] for s, q in a.large_part.items()}
            slot, pid = max(held.items(), key=lambda sk: (len(a.remaining_package(sk[1])),
                                                          self.orders[a.packages[sk[1]].order_id].t_s))
            t += T["station_handoff_s"]
            a.park(slot)
            self.bump("partial_handoffs", t)
            self.log(t, "park", "", pid, slot)
        self.bump("station_s", t0, t - t0)
        return t

    def unload(self, pr):
        """The DESIGN.md per-printer sequence, timed; caddy updates are applied as the robot reaches them."""
        T, p, a = self.T, self.p, self.alloc
        pr.claimed = True
        t = self.drive(self.t, pr.x_open)
        wait = 0.0
        if self.p.policy in WAIT_FOR_RELEASE:      # open the door so the first pick meets a released bed
            t_open = pr.t_release - T["door_open_s"] - T["scan_s"]
            wait, t = max(0.0, t_open - t), max(t, t_open)
        if pr.state == "cooling" and pr.t_release <= t:
            self.set_state(pr, pr.t_release, "waiting")
        self.set_state(pr, t, "unloading")
        self.log(t, "unload_start", pr.idx, pr.plate["plate_id"], f"{len(pr.todo)} parts")
        t += T["door_open_s"] + T["scan_s"]
        if t < pr.t_release - 1e-6:                # bed still warm: the first pick fails, then wait
            self.bump("early_pick_fails", t)
            t += T["retry_s"]
            wait += max(0.0, pr.t_release - t)
            t = max(t, pr.t_release)
        if pr.purge:
            name = f"{pr.plate['plate_id']}_purge_tab"
            t += T["purge_tab_s"]
            a.placed(name, a.slot_for(name))
            self.waste += 1
            pr.purge = False
        blocked = False
        for name in list(pr.todo):
            u, z = self.draws(name)
            if name not in pr.freed and u[1] < T["p_intervention_per_part"]:
                self.bump("interventions", t)
                pr.freed.add(name)
                self.log(t, "intervention", pr.idx, name, "in shift" if self.in_shift(t) else "blocked")
                if not self.in_shift(t):
                    blocked = True
                    break
                wait += p.human_response_s + p.human_fix_s
                t += p.human_response_s + p.human_fix_s
                self.bump("human_s", t, p.human_fix_s)
            if name not in pr.freed and u[1] < T["p_intervention_per_part"] + T["p_reprint_per_part"]:
                # not picked, but raked into the reject tray / dropped off the bed: reprinted, printer not blocked
                self.bump("robot_reprints", t)
                t += T["rake_s"]
                pr.todo.remove(name)
                self.part_unloaded(name, t, True)
                continue
            defective = bool(u[0] < p.p_defect)
            slot = a.slot_for(name, defective)
            if slot is None:                       # caddy full mid-plate: close, go to the station, come back
                self.bump("caddy_full_events", t)
                need = "large" if name in a.large_parts else "tote"
                t = self.station_visit(t + T["door_close_s"], need=need)
                t = self.drive(t, pr.x_open) + T["door_open_s"]
                slot = a.slot_for(name)
            t += self.pick_s(z[0])
            if u[2] > T["p_success_first_try"]:
                self.bump("retries", t)
                t += T["retry_s"] + self.pick_s(z[1])
            if a.placed(name, slot):
                self.pst[a.part_pkg[name]] = {"t_complete": t}
            pr.todo.remove(name)
            self.part_unloaded(name, t, defective)
        if blocked:                                # out of shift: leave it for the next shift
            t += T["door_close_s"]
            self.set_state(pr, t, "blocked")
            fix = self.next_check(t) + p.human_fix_s
            self.bump("human_s", fix, p.human_fix_s)
            self.push(fix, "unblock", pr.idx)
        else:
            t += T["bed_check_s"] + T["door_close_s"]
            self.push(t, "unload_done", pr.idx)
        self.robot_until(t, wait, f"unload P{pr.idx}")

    def _on_unblock(self, i):
        pr = self.printers[i]
        self.set_state(pr, self.t, "waiting")
        pr.claimed = False

    def _on_unload_done(self, i):
        pr = self.printers[i]
        self.log(self.t, "unload_done", i, pr.plate["plate_id"])
        pr.claimed, pr.plate = False, None
        self.set_state(pr, self.t, "idle")
        self.start_idle()

    # ---- human baseline ----
    def _on_check(self, _):
        self.log(self.t, "human_check")
        self.human_next()

    def human_next(self):
        waiting = [pr for pr in self.printers if pr.state == "waiting"]
        if waiting and self.in_shift(self.t):
            pr = waiting[0]
            pr.claimed = True
            self.set_state(pr, self.t, "unloading")
            t = self.t + self.p.human_unload_s + self.p.human_part_s * len(pr.todo)
            for name in pr.todo:
                u, _ = self.draws(name)
                self.part_unloaded(name, t, bool(u[0] < self.p.p_defect))
            pr.todo, pr.purge = [], False
            self.busy.append((self.t, t, 0.0, f"unload P{pr.idx}"))
            self.bump("human_s", self.t, t - self.t)
            return self.push(t, "human_done", pr.idx)
        t = self.t
        for oid, s in self.ost.items():            # pack every complete order by hand, package by package
            if s["t_complete"] is not None and not s["packed"]:
                s["packed"] = True
                for k in self.orders[oid].packages:
                    self.ship(t, k, self.p.human_pack_s)
                    t = self.line_free
        if t > self.t:
            self.busy.append((self.t, t, 0.0, "pack"))
            self.bump("human_s", self.t, t - self.t)
        self.push(self.next_check(max(t, self.t + 1.0)), "check", None)

    def _on_human_done(self, i):
        pr = self.printers[i]
        pr.claimed, pr.plate = False, None
        self.set_state(pr, self.t, "idle")
        self.start_idle()
        self.human_next()

    # ---- outputs ----
    def state_hours(self, t0=0.0, t1=None):
        """Hours per state summed over printers, clipped to [t0, t1]."""
        t1 = self.H if t1 is None else t1
        out = dict.fromkeys(STATES, 0.0)
        for pr in self.printers:
            for (a, s), (b, _) in zip(pr.log, pr.log[1:] + [(self.t_end, None)]):
                out[s] += max(0.0, min(b, t1) - max(a, t0)) / 3600.0
        return out

    def kpis(self):
        H, days = self.H, self.p.days
        sh = self.state_hours()
        lead, late, shipped_in_window = [], 0, 0
        for oid, o in self.orders.items():
            ts = self.ost[oid]["t_shipped"]
            lead.append(((ts if ts is not None else self.t_end) - o.t_s) / 3600.0)
            late += ts is None or ts > o.ship_by_s
            shipped_in_window += ts is not None and ts < H
        busy = sum(max(0.0, min(b, H) - a) * (1 - w / (b - a)) for a, b, w, _ in self.busy if b > a)
        lead = np.array(lead) if lead else np.zeros(1)
        k = self.k
        return {
            "policy": self.p.policy, "days": days, "seed": self.p.seed, "order_scale": self.p.order_scale,
            "timings_source": self.timings_src,
            "layout": self.layout.kind, "n_printers": len(self.printers), "station": self.p.station,
            "rail_length_m": round(self.layout.rail_length_m, 2),
            "printer_util_pct": round(100 * sh["printing"] / (len(self.printers) * H / 3600), 2),
            "printer_wait_pct": round(100 * (sh["waiting"] + sh["blocked"]) / (len(self.printers) * H / 3600), 2),
            "parts_per_printer_day": round(k["parts_good"] / days / len(self.printers), 2),
            "printer_waiting_h": round(sh["waiting"] + sh["blocked"], 2), "printer_blocked_h": round(sh["blocked"], 2),
            "printer_cooling_h": round(sh["cooling"], 2), "printer_unloading_h": round(sh["unloading"], 2),
            "printer_idle_h": round(sh["idle"], 2), "printing_h": round(sh["printing"], 2),
            "parts_per_day": round(k["parts_good"] / days, 2), "parts_rejected": int(k["parts_rejected"]),
            "plates": int(k["plates"]),
            "orders_arrived": len(self.orders), "orders_shipped": shipped_in_window,
            "orders_unshipped_at_end": int(sum(s["t_shipped"] is None for s in self.ost.values())),
            "lead_time_p50_h": round(float(np.percentile(lead, 50)), 2),
            "lead_time_p90_h": round(float(np.percentile(lead, 90)), 2),
            "on_time_pct": round(100 * (1 - late / max(1, len(self.orders))), 2),
            "robot_util_pct": None if self.human else round(100 * busy / H, 2),
            "rail_m": round(k["rail_m"], 1), "interventions": int(k["interventions"]),
            "robot_reprints": int(k["robot_reprints"]),
            "caddy_full_events": int(k["caddy_full_events"]), "station_visits": int(k["station_visits"]),
            "partial_handoffs": int(k["partial_handoffs"]), "early_pick_fails": int(k["early_pick_fails"]),
            "max_caddy_slots_used": None if self.human else self.alloc.metrics["max_totes_used"],
            "max_tray_slots_used": None if self.human else self.alloc.metrics["max_large_used"],
            "packages_shipped": sum(v.get("t_shipped", H) < H for v in self.pst.values()),
            "packages_per_order": round(float(np.mean([len(o.packages) for o in self.orders.values()] or [0])), 3),
            "large_parts_pct": round(100 * float(np.mean([CATALOG[p["sku"]].large for p in self.parts.values()]
                                                          or [0])), 2),
            "station_busy_pct": None if self.human else round(100 * k["station_s"] / H, 2),
            "station_share_of_robot_pct": None if self.human else round(100 * k["station_s"] / max(busy, 1e-9), 1),
            "station_s_per_package": None if self.human else round(k["station_s"] / max(k["packages_handed"], 1), 1),
            "waste_dumps": int(k["waste_dumps"]), "station_source": self.station_src,
            "order_skus": "ORDER_SKUS" if self.p.order_skus is None else "+".join(self.p.order_skus),
            "human_h": round(k["human_s"] / 3600, 2), "lanes": dict(sorted(self.lanes.items())),
        }

    def write(self, out):
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        rows = list(self.events)
        for pr in self.printers:
            rows += [(round(t, 3), "printer_state", pr.idx, "", s) for t, s in pr.log]
        actor = "human" if self.human else "robot"
        rows += [(round(a, 3), f"{actor}_busy", "", what, round(b, 3)) for a, b, _, what in self.busy]
        rows.sort(key=lambda r: r[0])
        with open(out / "events.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "kind", "printer", "ref", "detail"])
            w.writerows(rows)
        kp = self.kpis()
        (out / "kpis.json").write_text(json.dumps({"kpis": kp, "params": asdict(self.p), "timings": self.T}, indent=1))
        return kp


def simulate(**kw):
    sim = Sim(Params(**kw))
    sim.run()
    return sim


def summarize(ks):
    """Mean / sample std over seeds for numeric KPIs."""
    num = [k for k, v in ks[0].items() if isinstance(v, (int, float)) and not isinstance(v, bool)
           and all(isinstance(x[k], (int, float)) for x in ks)]
    xs = {k: np.array([x[k] for x in ks], float) for k in num}
    return {"mean": {**ks[0], **{k: round(float(v.mean()), 2) for k, v in xs.items()}},
            "std": {k: round(float(v.std(ddof=1)), 2) if len(v) > 1 else 0.0 for k, v in xs.items()},
            "per_seed": ks}


COMPARE = ("fifo", "nearest", "max_util", "order_aware", "human")
TABLE = ("printer_util_pct", "printer_waiting_h", "printer_blocked_h", "parts_per_day", "orders_shipped",
         "lead_time_p50_h", "lead_time_p90_h", "robot_util_pct", "rail_m", "interventions", "robot_reprints",
         "caddy_full_events", "station_visits", "partial_handoffs", "early_pick_fails", "human_h",
         "packages_per_order", "packages_shipped", "station_busy_pct", "station_share_of_robot_pct",
         "station_s_per_package", "max_caddy_slots_used", "max_tray_slots_used", "waste_dumps")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--days", type=float, default=7.0)
    ap.add_argument("--policy", default="order_aware", choices=[*POLICIES, "human"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/des/")
    ap.add_argument("--compare", action="store_true", help="all policies + human baseline, plus capacity runs")
    ap.add_argument("--seeds", type=int, default=5, help="--compare: seeds seed..seed+N-1 (events.csv for the first)")
    ap.add_argument("--capacity-scale", type=float, default=3.0, help="order-rate multiplier for capacity runs")
    ap.add_argument("--layout", default="default", choices=("default", *LAYOUTS), help="single run: cell layout")
    ap.add_argument("--n-printers", type=int, default=N_PRINTERS, help="single run: printers per arm (row/facing)")
    ap.add_argument("--station", default="end", choices=STATIONS, help="single run: hand-off station position")
    ap.add_argument("--order-scale", type=float, default=1.0, help="single run: order-rate multiplier")
    ap.add_argument("--skus", default="small", choices=SKU_MIXES,
                    help="order SKU mix: small = farm.catalog.SMALL_SKUS, all = every SKU a plate can print")
    ap.add_argument("--queue-policy", default="package", choices=("package", "fifo"),
                    help="robot plate composition (fleet.queue_policy); fifo = by order, as before bed to box")
    ap.add_argument("--shift-h", help="operator shift start,end (h) on each workday; an end past 24 runs past "
                                      "midnight, e.g. 9,26 = 9-18 plus 18-02 (default 9,18)")
    ap.add_argument("--workdays", help="days a shift starts on, 0 = Monday: a range 0-6 or a list 0,1,2 (default 0-4)")
    ap.add_argument("--no-long-plates", action="store_true",
                    help="human baseline: cap plates at the print-time cap instead of sizing them to the next check")
    ap.add_argument("--policies", help="--compare: comma-separated subset of " + ",".join(COMPARE))
    a = ap.parse_args()
    t0, out = time.time(), Path(a.out)
    skus, excluded = sku_mix(a.skus)
    if excluded:
        print(f"--skus all: left out {', '.join(excluded)} (prints longer than one plate, MAX_PLATE_S)")
    extra = {}
    if a.shift_h:
        extra["shift_h"] = tuple(float(v) for v in a.shift_h.split(","))
    if a.workdays:
        lo, _, hi = a.workdays.partition("-")
        extra["workdays"] = tuple(range(int(lo), int(hi) + 1)) if hi else tuple(int(v) for v in a.workdays.split(","))
    if a.no_long_plates:
        extra["human_long_plates"] = False
    if not a.compare:
        kp = simulate(days=a.days, policy=a.policy, seed=a.seed, layout=a.layout, n_printers=a.n_printers,
                      station=a.station, order_scale=a.order_scale, order_skus=skus,
                      queue_policy=a.queue_policy, **extra).write(out)
        print(json.dumps(kp, indent=1))
        return
    seeds = list(range(a.seed, a.seed + a.seeds))
    pols = tuple(a.policies.split(",")) if a.policies else COMPARE
    assert all(p in COMPARE for p in pols), pols

    def batch(pol, scale, tag):
        sims = [simulate(days=a.days, policy=pol, seed=s, order_scale=scale * N_PRINTERS / 10.0, order_skus=skus,
                         queue_policy=a.queue_policy, **extra) for s in seeds]
        return summarize([sims[0].write(out / tag)] + [s.kpis() for s in sims[1:]])

    runs = {pol: batch(pol, 1.0, pol) for pol in pols}
    cap = {pol: batch(pol, a.capacity_scale, f"capacity_{pol}") for pol in ("order_aware", "human") if pol in pols}
    m = runs[pols[0]]["mean"]
    res = {"note": "simulation, assumed parameters (fleet/des.py Params; robot timings: "
                   f"{m['timings_source']}; station times: {m['station_source']}); mean/std over seeds",
           "days": a.days, "seeds": seeds, "capacity_scale": a.capacity_scale, "skus": a.skus,
           "queue_policy": a.queue_policy, "params_override": {k: list(v) if isinstance(v, tuple) else v
                                                                for k, v in extra.items()},
           "skus_excluded": list(excluded), "runs": runs, "capacity": cap}
    (out / "kpis_compare.json").write_text(json.dumps(res, indent=1))
    w = max(map(len, TABLE))
    print(f"mean over seeds {seeds}")
    print(f"{'':{w}} " + " ".join(f"{p:>11}" for p in pols))
    for key in TABLE:
        vals = [runs[p]["mean"][key] for p in pols]
        print(f"{key:{w}} " + " ".join(f"{'-' if v is None else v:>11}" for v in vals))
    c = {p: cap[p]["mean"]["printing_h"] for p in cap}
    if len(c) == 2:
        print(f"capacity (orders x{a.capacity_scale}): printing_h order_aware {c['order_aware']} vs human "
              f"{c['human']} -> +{c['order_aware'] - c['human']:.0f} printer-h per {a.days:g} days")
    print(f"done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
