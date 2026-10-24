"""The whole cell: orders -> packages -> slicing -> printing -> cooling -> robot unload (physics) -> package totes and
the large-part tray -> station: bed to box -> carrier lanes.

Farm clock vs physics: printing and cooling are timers on the farm clock. Every robot action is simulated in
MuJoCo and advances the farm clock by the simulated time it took. When the robot has nothing to do, the farm
clock jumps to the next event without stepping physics.

Station (two rows, plan/station.py): one package at a time - request its box, place its large parts from the tray
(arm, physical grasp), drop its tote through the trap door into the box (physics; station camera checks the tote is
empty), then seal and convey it to its carrier lane. From the sealer on the box is KINEMATIC (contents parked, lid
shown). The one-row cell keeps the legacy kinematic hand-off (animated boxes).
A part the robot could not pick is raked into the printer's reject tray (or was dropped off the bed) and is reprinted:
the printer is not blocked. Only a part still on the bed after the rake fallback is a human intervention: the
operator puts it into its tote and the robot waits OPERATOR_S.
"""
import collections
import math

import mujoco
import numpy as np

from farm import catalog as C
from fleet.allocator import CaddyAllocator
from fleet.queue_policy import order_queue
from fleet.scheduler import TRAY_HOLD_S, Candidate, Snapshot, choose
from jobs.orders import Order, OrderStream, sample_lines
from jobs.packaging import package_fit, part_sku
from jobs.plates import make_plate
from plan.door import TUCK, motion_summary, operate_door, transfer
from farm.lab import set_status
from plan.sweep import in_tray
from plan.unload import mirror_plate, on_bed, spawn_plate, unload_parts

AMBIENT_C, BED_HOT_C, RELEASE_C, COOL_TAU_S = 25.0, 60.0, 35.0, 360.0      # ASSUMPTIONS (same as the DES)
COOL_S = COOL_TAU_S * math.log((BED_HOT_C - AMBIENT_C) / (RELEASE_C - AMBIENT_C))
OPERATOR_S = 120.0          # human fixes a failed pick / clears a tote / places a part (ASSUMPTION)
STATION_S = 30.0            # robot time per package at the station, for the scheduler (runs/station_times.json)
LEGACY_HANDOFF_S = 6.0      # one row: kinematic bin exchanger cycle per package
PACK_S = {"S": 35.0, "M": 45.0, "L": 55.0}                  # one row: kinematic packing time per box size
LANES = {"courier_A": -0.45, "courier_B": 0.0, "post": 0.45}   # one row: lane y past the legacy packing tunnel
MAX_PLATE_S = 3 * 3600.0


class Printer:
    def __init__(self, i):
        self.i = i
        self.state = "idle"        # idle | printing | cooling | waiting | unloading
        self.plate = None
        self.t_end = self.t_done = self.t_release = 0.0
        self.bodies = {}


class Farm:
    def __init__(self, world, ik, ex, seed=0, order_scale=1.0, mask=None, policy="order_aware", log=print,
                 station_noise="ideal"):
        self.w, self.ik, self.ex = world, ik, ex
        self.rng = np.random.default_rng(seed)
        self.stream = OrderStream(seed=seed, scale=order_scale)
        self.alloc = CaddyAllocator()
        self.printers = [Printer(i) for i in range(C.N_PRINTERS)]
        self.queue = []                  # unprinted parts, FIFO: {"name", "order_id", "sku"}
        self.orders = {}
        self.t = 0.0                     # farm clock (s)
        self.t_phys = world.d.time       # physics time already accounted for
        self.mask, self.policy, self.log = mask, policy, log
        self.plate_n = 0
        self.stats = collections.Counter()
        self.drive_s = 0.0          # robot time in the farm's drives to a printer (reporting)
        self.events = []
        self.part_log = []
        self.lead = []
        self.station = None
        if C.TWO_ROWS:
            from plan.station import Station
            self.station = Station(world, noise=station_noise, rng=np.random.default_rng((seed, 11)))
            self.station.on_action = lambda text: setattr(self, "action", text)
        self.packages = []               # one row: [body id, t0, lane y, pack s] of animated boxes
        self.pkg_free = [world.m.body(f"pkg{k}").id for k in range(10)]
        self.action = "starting"
        self._caddy_bodies, self._waste_bodies, self._tray_bodies = {}, [], {}
        self._shelf_bodies = set()       # parts on the consolidation shelf (out of the simulation)
        world.hooks = [self._animate]
        if self.station is not None:
            self._init_monitor()
            world.hooks.append(self._monitor)
        self._next_arrival = self._sample_arrival(0.0)

    # ------------------------------------------------------------------ clock
    def sync_clock(self):
        dt = self.w.d.time - self.t_phys
        self.t_phys = self.w.d.time
        self.t += dt
        self._advance_timers()

    def jump(self, t_new):
        """Advance the farm clock without physics (robot idle)."""
        self.t = max(self.t, t_new)
        self._advance_timers()

    def _sample_arrival(self, t):
        rate = self.stream.rate(t) / 3600.0
        return t + self.rng.exponential(1.0 / max(rate, 1e-9))

    def _advance_timers(self):
        while self._next_arrival <= self.t:
            self._new_order(self._next_arrival)
            self._next_arrival = self._sample_arrival(self._next_arrival)
        for p in self.printers:
            if p.state == "printing" and self.t >= p.t_end:
                self._print_done(p, p.t_end)
            if p.state == "cooling" and self.t >= p.t_release:
                p.state = "waiting"
        for p in self.printers:
            if p.state == "idle":
                self._start_print(p)

    def next_event_t(self):
        ts = [self._next_arrival] + [p.t_end for p in self.printers if p.state == "printing"] + \
             [p.t_release for p in self.printers if p.state == "cooling"] + \
             [p.t_release + TRAY_HOLD_S for p in self.printers      # a plate held back for a tray slot is due then
              if p.state == "waiting" and p.t_release + TRAY_HOLD_S > self.t]
        return min(ts)

    # ------------------------------------------------------------------ orders / printing
    def _new_order(self, t, lines=None):
        k = len(self.orders)
        cust = int(self.rng.integers(self.stream.n_customers))
        carrier = str(self.rng.choice(list(LANES), p=[0.45, 0.35, 0.20]))
        lines = lines or sample_lines(self.rng)
        o = Order(f"ord{k}", t, f"c{cust}", carrier, t + 24 * 3600, lines)
        self.orders[o.order_id] = o
        self.alloc.add_order(o.order_id, o.book_entry())
        for name in o.parts:
            self.queue.append({"name": name, "order_id": o.order_id, "sku": part_sku(name)})
        self.stats["orders_in"] += 1
        self.stats["packages_in"] += len(o.packages)
        return o

    def _start_print(self, p):
        if not self.queue:
            return
        self.plate_n += 1
        queue = order_queue(self.queue, self.alloc, MAX_PLATE_S)     # package-aware: finish split packages first
        plate, _ = make_plate(queue, p.i, f"PL{self.plate_n:04d}", self.rng, reach_mask=self.mask, max_print_s=MAX_PLATE_S)
        if not plate["objects"]:
            return
        plate = mirror_plate(plate)
        names = {o["name"] for o in plate["objects"]}
        self.queue = [q for q in self.queue if q["name"] not in names]
        p.plate, p.state = plate, "printing"
        p.t_end = self.t + plate["print_time_s"]
        self.events.append((self.t, p.i, "print_start", plate["plate_id"], len(names)))

    def _print_done(self, p, t):
        p.bodies = spawn_plate(self.w, p.plate, self.rng, pos_sigma_mm=1.5, yaw_sigma_deg=2.0, stuck=True)
        p.state, p.t_done, p.t_release = "cooling", t, t + COOL_S
        self.events.append((t, p.i, "print_done", p.plate["plate_id"], len(p.bodies)))

    def bed_temp(self, p):
        if p.state == "printing":
            return BED_HOT_C
        if p.state in ("cooling", "waiting", "unloading"):
            return AMBIENT_C + (BED_HOT_C - AMBIENT_C) * math.exp(-(self.t - p.t_done) / COOL_TAU_S)
        return AMBIENT_C

    # ------------------------------------------------------------------ robot decisions
    def choose_printer(self):
        cands = []
        for p in self.printers:
            if p.state in ("cooling", "waiting"):
                e = self.alloc.plate_effect(list(p.bodies))
                xc, s = C.printer_frame(p.i)
                cands.append(Candidate(p.i, xc + s * 0.15, p.t_done, p.t_release, len(p.bodies), e.new_totes,
                                       e.completes, e.frees, e.new_large, e.open_large))
        if not cands:
            return None
        snap = Snapshot(self.t, self.ex.x, 0.4, tuple(cands), 3600.0 if self.queue else 0.0, MAX_PLATE_S, 75.0,
                        18.0, len(self.alloc.free_slots()), len(self.alloc.handoff_ready()), C.STATION_X, STATION_S,
                        len(self.alloc.free_large()) if self.alloc.large else None)
        c = choose(self.policy, snap)
        return None if c is None else self.printers[c.printer]

    # ------------------------------------------------------------------ robot actions (physics)
    def drive(self, x):
        t0 = self.w.d.time
        transfer(self.ex, self.ik, TUCK, x)
        self.drive_s += self.w.d.time - t0
        self.sync_clock()

    def unload(self, p):
        i = p.i
        xc, _ = C.printer_frame(i)
        self.action = f"P{i + 1}: drive"
        self.drive(xc + C.printer_frame(i)[1] * 0.35)
        if self.t < p.t_release:                       # hold off until the bed has released the parts
            self.action = f"P{i + 1}: waiting for the bed to cool"
            self.jump(p.t_release)
        for b in p.bodies.values():
            self.w.set_stuck(b, False)
        for b in self._tray_bodies.pop(i, []):          # the operator emptied the reject tray during the print
            self.w.park(b)
        p.state = "unloading"
        t0 = self.t
        self.action = f"P{i + 1}: open door"
        ok, _, note = operate_door(self.ex, self.ik, i, opening=True)
        self.sync_clock()
        self.stats["door_open_ok" if ok else "door_open_fail"] += 1
        if not ok:
            self.log(f"[{self._clock()}] P{i + 1}: door open FAILED ({note}) -> operator")
            self.stats["interventions"] += 1
            self.jump(self.t + OPERATOR_S)

        def slot_for(name):
            slot = self.alloc.slot_for(name)
            if slot is None:                          # no free tote / tray slot: hand off, else park one at the shelf
                self.station_visit(need="large" if name in self.alloc.large_parts else "tote")
                slot = self.alloc.slot_for(name)
            self.action = f"P{i + 1}: {name.split('_')[1] if '_' in name else name} -> {slot}"
            return slot

        def on_result(name, body, slot, outcome, sec):
            self.sync_clock()
            blocking = outcome != "placed" and on_bed(self.w, i, body)
            self.part_log.append({"t": self.t, "printer": i, "part": name, "sku": self.w.part_sku[body],
                                  "slot": slot, "outcome": outcome, "s": round(sec, 2), "blocking": bool(blocking)})
            self.stats[f"pick:{outcome}"] += 1
            if outcome != "placed" and not blocking:   # raked into the tray / dropped off the bed: reprint it
                self.stats["reprints"] += 1
                if in_tray(self.w, i, body):
                    self._tray_bodies.setdefault(i, []).append(body)
                elif body in self.w.in_use:
                    self.w.park(body)                  # lost on the floor / deck: collected later
                if not name.endswith("purge_tab"):
                    if slot is not None:
                        self.alloc.unreserve(name)
                    self.queue.insert(0, {"name": name, "order_id": name.rsplit("_", 2)[0], "sku": self.w.part_sku[body]})
                return
            if slot is None:                           # still on the bed: the operator takes it to its tote
                slot = slot_for(name)
            if outcome != "placed":
                self.stats["interventions"] += 1
                self.jump(self.t + OPERATOR_S)
                self._operator_places(body, slot)
            if slot == "waste":
                self._waste_bodies.append(body)
            else:
                self._caddy_bodies[name] = body
            done = self.alloc.placed(name, slot)
            if done:
                self.events.append((self.t, i, "package_complete", done, slot))

        drops0 = dict(getattr(self.ex, "blob_drops", None) or {})
        empty, unknown = unload_parts(self.ex, self.ik, i, p.plate, p.bodies, slot_for, on_result, self.rng)
        self.sync_clock()
        drops1 = getattr(self.ex, "blob_drops", None) or {}
        stray = sum(drops1.get(k, 0) - drops0.get(k, 0) for k in ("missed", "lost"))
        if stray:                                       # a cleared blob not confirmed in the reject bin (on the deck,
            self.stats["blob_not_in_reject"] += stray       # the floor, a bin rim): an operator collects it
            self.stats["interventions"] += 1
            self.jump(self.t + OPERATOR_S)
        if unknown > 200:
            self.stats["debris_flags"] += 1
        if not empty:                                   # camera cannot confirm a clear bed: an operator checks it
            self.stats["bed_not_empty"] += 1                # before the printer starts again (counted intervention)
            self.stats["interventions"] += 1
            self.jump(self.t + OPERATOR_S)
            for b in list(p.bodies.values()):
                if b in self.w.in_use and on_bed(self.w, i, b):
                    self.w.park(b)
        self.action = f"P{i + 1}: close door"
        ok, _, note = operate_door(self.ex, self.ik, i, opening=False)
        self.sync_clock()
        self.stats["door_close_ok" if ok else "door_close_fail"] += 1
        if not ok:
            self.stats["interventions"] += 1
            self.jump(self.t + OPERATOR_S)
            self.w.d.qpos[self.w.door_q[i]] = 0.0
        self.events.append((self.t, i, "unloaded", p.plate["plate_id"], round(self.t - t0, 1)))
        self.stats["plates_unloaded"] += 1
        p.state, p.plate, p.bodies = "idle", None, {}
        self._advance_timers()

    def _operator_places(self, body, slot):
        sid = mujoco.mj_name2id(self.w.m, mujoco.mjtObj.mjOBJ_SITE, f"bin_{slot}")
        mujoco.mj_forward(self.w.m, self.w.d)
        c = self.w.d.site_xpos[sid].copy()
        depth = C.LARGE_SLOT_SIZE[2] if slot in C.LARGE_SLOTS else C.BIN_SIZE[2]
        self.w.spawn(body, c + [0, 0, -depth + 0.004], 0.0, stuck=False)

    # ------------------------------------------------------------------ station
    def station_visit(self, need=None):
        """Ship every ready package, one at a time; dump the waste bin. need = "tote" | "large": the unload in progress
        needs a free one of that kind - if shipping freed none, park an incomplete package's tote / tray slot on the
        station's consolidation shelf (kinematic: its parts leave the simulation until the package ships)."""
        ready = self.alloc.handoff_ready()
        if not ready and need is None:
            return
        self.action = "station: dock"
        if self.station is None:
            self.drive(C.STATION_X)
        for r in ready:
            if self.station is None:
                self._ship_legacy(r)
            else:
                self._ship(r)
        if self._waste_bodies:
            if self.station is not None:
                self.station.dump_waste(self.ex, self.ik)
                self.sync_clock()
                self.station.scrap += self._waste_bodies
            else:
                for b in self._waste_bodies:
                    self.w.park(b)
            self._waste_bodies = []
        if need == "tote" and not self.alloc.free_slots() or need == "large" and not self.alloc.free_large():
            slots = list(self.alloc.tote_pkg) if need == "tote" else list(self.alloc.large_part)
            slot = slots[0]
            for part in list(self.alloc.contents[slot]):
                b = self._caddy_bodies.pop(part, None)
                if b is not None:
                    self.w.park(b)
                    self._shelf_bodies.add(b)
            self.alloc.park(slot)
            self.stats["parks_large" if need == "large" else "parks"] += 1
        if self.station is None:
            self.ex.wait(LEGACY_HANDOFF_S * max(len(ready), 1))
        self.sync_clock()

    def _ship(self, r):
        """One package through the station (plan/station.py): box -> large parts -> tote dump -> seal -> lane."""
        st, w, ex, ik = self.station, self.w, self.ex, self.ik
        pkg = r.package
        o = self.orders[pkg.order_id]
        L = st.L
        t_pkg = w.d.time
        st.tag = pkg.package_id
        self.action = f"station: request box {pkg.box} for {pkg.package_id}"
        box = st.request_box(ex, pkg.box, "place" if pkg.large else (r.tote or "place"), wait=False)
        n_far = len(st.times["to_station"])
        st.dock(ex, ik, L["x_place"] if pkg.large else L["dock"][r.tote] if r.tote else L["x_place"])
        self.sync_clock()
        if len(st.times["to_station"]) > n_far:   # came from the printers: the drive is timed as to_station, and the
            t_pkg = w.d.time                      # package clock starts on arrival
        st.wait_box(ex, box)
        self.sync_clock()
        expect = []
        fit = package_fit(pkg) if pkg.large else None
        for k, part in enumerate(pkg.large):
            slot = self.alloc.part_large.get(part)
            body = self._caddy_bodies.get(part)
            if slot is None or body is None:            # on the consolidation shelf: added by the operator below
                continue
            sku = part_sku(part)
            self.action = f"station: place {sku} from {slot} ({pkg.package_id})"
            ok, outcome, sec = st.place_large(ex, ik, slot, box, body, sku, layout=fit.layout[k] if fit else (0, 0, 0))
            self.sync_clock()
            self.stats[f"place:{outcome}"] += 1
            if not ok:                                  # an operator checks; puts it into the box if it is not there
                self.stats["interventions"] += 1
                self.jump(self.t + OPERATOR_S)
                from plan.station import part_in_box
                if not part_in_box(w, body, box):
                    self._into_box(body, box)
            expect.append(body)
        if r.tote is not None:
            self.action = f"station: carriage and box to the drop ({pkg.package_id})"   # dump_tote names the tote
            out = st.dump_tote(ex, ik, r.tote, box)
            self.sync_clock()
            self.stats["tote_dumps"] += 1
            self.stats["tote_retries"] += out["tries"] > 1
            tote_bodies = [self._caddy_bodies[p] for p in pkg.small if p in self._caddy_bodies]
            if out["outcome"] != "dumped":              # leftover / unclear scan / jammed door: the operator clears it
                self.stats["interventions"] += 1
                self.stats["tote_operator"] += 1
                self.jump(self.t + OPERATOR_S)
                from plan.station import part_in_box
                for b in tote_bodies:
                    if b in w.in_use and not part_in_box(w, b, box):
                        self._into_box(b, box)
            expect += tote_bodies
        for part in r.shelf:                            # parked parts: the operator adds them from the shelf
            sku = part_sku(part)
            b = w.free_part(sku)
            self._into_box(b, box)
            self._shelf_bodies.discard(b)
            expect.append(b)
            self.stats["shelf_adds"] += 1
        self.action = f"station: seal {pkg.package_id} -> lane {o.carrier}"
        st.seal_and_ship(box, o.carrier, expect=expect)
        st.times["package"].append(w.d.time - t_pkg)
        st.tag = ""
        for part in pkg.parts:
            self._caddy_bodies.pop(part, None)
        st.forget(expect)
        oid = self.alloc.handed_off(pkg.package_id)
        self.stats["packages_shipped"] += 1
        self.events.append((self.t, -1, "shipped", pkg.package_id, pkg.box))
        if oid:
            self.lead.append(self.t - o.t_s)
            self.stats["orders_shipped"] += 1
            self.events.append((self.t, -1, "order_shipped", oid, len(o.packages)))

    def _into_box(self, body, box):
        """Operator: put a part into the box by hand (spawned on its floor, a little off centre)."""
        from farm.scene import box_outer
        p = self.w.box_pos(box) + [self.rng.uniform(-0.02, 0.02), self.rng.uniform(-0.03, 0.03),
                                    box_outer(self.w.box_size[box])[2] - 0.01]     # dropped in from the rim
        self.w.spawn(body, p, float(self.rng.uniform(0, math.pi)), stuck=False)

    def _ship_legacy(self, r):
        """One row: kinematic hand-off (the tote's contents leave the simulation; an animated box rides to its lane)."""
        pkg = r.package
        for part in pkg.parts:
            b = self._caddy_bodies.pop(part, None)
            if b is not None:
                self.w.park(b)
        oid = self.alloc.handed_off(pkg.package_id)
        o = self.orders[pkg.order_id]
        self.stats["packages_shipped"] += 1
        self.events.append((self.t, -1, "shipped", pkg.package_id, pkg.box))
        if oid:
            self.lead.append(self.t - o.t_s)
            self.stats["orders_shipped"] += 1
        if self.pkg_free:
            bid = self.pkg_free.pop(0)
            self.packages.append([bid, self.w.d.time, LANES[o.carrier], PACK_S[pkg.box]])

    def _init_monitor(self):
        """Station overlap monitor (reporting only): robot contacts with station hardware that has collision (belt,
        chute, scrap bin, boxes), and the carriage (deck, totes, flaps, tray, column, dressing; not the arm, which may
        reach into a box) overlapping any static station / lab geom, checked every 0.5 s of sim time."""
        from farm import lab
        w, m = self.w, self.w.m
        st = {g for g in range(m.ngeom) if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("stn_")
              or m.geom_bodyid[g] in w.box_body.values()}
        self._stn_geoms = st
        self._car_geoms = np.array(sorted(w.robot_geoms - w.arm_geoms))
        boxes = [(lo, hi) for lo, hi, _ in lab.PRIMS]                    # lab dressing, primitive by primitive
        for g in range(m.ngeom):                                          # scene geoms on the world body
            if m.geom_bodyid[g] or m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE or \
                    (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("lab_") or m.geom_group[g] == 2:
                continue
            R = w.d.geom_xmat[g].reshape(3, 3)
            c, e = w.d.geom_xpos[g] + R @ m.geom_aabb[g, :3], np.abs(R) @ m.geom_aabb[g, 3:]
            boxes.append((c - e, c + e))
        self._static = (np.array([b[0] for b in boxes]), np.array([b[1] for b in boxes]))
        self._mon_k = 0
        self.contact_pairs = collections.Counter()
        self.contact_log = []

    def _monitor(self, w):
        m, d = w.m, w.d
        for c in d.contact[:d.ncon]:
            r1, r2 = c.geom1 in w.robot_geoms, c.geom2 in w.robot_geoms
            if r1 != r2 and (c.geom2 if r1 else c.geom1) in self._stn_geoms and c.dist < 0:
                self.stats["station_contacts"] += 1         # (ticks with a contact; pairs in contact_pairs)
                rb, ob = (m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2])[::1 if r1 else -1]
                self.contact_pairs[f"{m.body(rb).name}-{m.body(ob).name}"] += 1
                if len(self.contact_log) < 40:              # when and where (first ticks), for debugging
                    self.contact_log.append({"t": round(float(d.time), 3), "action": self.action,
                                             "pair": f"{m.body(rb).name}-{m.body(ob).name}",
                                             "depth_mm": round(-1000 * float(c.dist), 2),
                                             "at_rel_body_mm": [round(1000 * v, 1) for v in c.pos - d.xpos[ob]],
                                             "last_motion": self.ex.log[-1][0] if self.ex.log else ""})
                break
        self._mon_k += 1
        if self._mon_k % 50:
            return
        g = self._car_geoms
        R = d.geom_xmat[g].reshape(-1, 3, 3)
        c = d.geom_xpos[g] + np.einsum("nij,nj->ni", R, m.geom_aabb[g, :3])
        e = np.einsum("nij,nj->ni", np.abs(R), m.geom_aabb[g, 3:])
        lo, hi = self._static
        hit = np.all((c - e)[:, None] < hi[None] - 1e-4, axis=2) & np.all(lo[None] < (c + e)[:, None] - 1e-4, axis=2)
        if hit.any():
            self.stats["station_overlaps"] += 1

    def _animate(self, w):
        """Printer status light bars; one row: the kinematic packaging line (belt -> tunnel -> carrier lane)."""
        for p in self.printers:
            set_status(w.m, p.i, p.state)
        if self.station is not None:
            return
        x0 = C.STATION_X + 0.45
        keep = []
        for rec in self.packages:
            bid, t0, lane_y, pack_s = rec
            mid = w.m.body_mocapid[bid]
            s = w.d.time - t0
            if s < 3.0:
                pos = [x0 + 0.85 * s / 3.0, 0.0, 0.67]
            elif s < 3.0 + pack_s / 10:
                pos = [x0 + 0.85, 0.0, 0.67]
            elif s < 6.0 + pack_s / 10:
                f = (s - 3.0 - pack_s / 10) / 3.0
                pos = [x0 + 0.85 + 0.9 * f, lane_y * f, 0.67 - 0.1 * f]
            else:
                pos = [6 + bid * 0.01, 3.0, -1.0]
                self.pkg_free.append(bid)
                w.d.mocap_pos[mid] = pos
                continue
            w.d.mocap_pos[mid] = pos
            keep.append(rec)
        self.packages = keep

    # ------------------------------------------------------------------ main loop
    def _clock(self):
        h, rem = divmod(int(self.t), 3600)
        return f"{h:02d}:{rem // 60:02d}"

    def run(self, hours, max_plates=None):
        self._advance_timers()
        t_end = self.t + hours * 3600
        while self.t < t_end and (max_plates is None or self.stats["plates_unloaded"] < max_plates):
            ready = self.alloc.handoff_ready()
            p = self.choose_printer()
            tray_full = bool(self.alloc.large) and not self.alloc.free_large()
            if ready and (p is None or len(ready) >= 2 or len(self.alloc.free_slots()) < 2 or tray_full):
                self.station_visit()
                continue
            if p is None:
                self.action = "idle: all printers busy"
                self.jump(self.next_event_t())
                continue
            self.log(f"[{self._clock()}] unload P{p.i + 1} ({len(p.bodies)} items, bed {self.bed_temp(p):.0f} C)")
            self.unload(p)
        return self.summary()

    def finish(self, max_s=60.0):
        """Let boxes still on the conveyor reach their lanes (physics only, robot idle in TUCK)."""
        if self.station is None:
            return
        w = self.w
        t0 = w.d.time
        while any(w.box_busy(b) for b in w.box_live) and w.d.time - t0 < max_s:
            self.ex.wait(0.5)
        self.sync_clock()

    def summary(self):
        lead = np.array(self.lead) / 3600 if self.lead else np.array([np.nan])
        picks = {k[5:]: v for k, v in self.stats.items() if k.startswith("pick:")}
        n_picks = sum(picks.values())
        out = {"farm_hours": self.t / 3600, "stats": dict(self.stats), "picks": picks,
               "pick_success": picks.get("placed", 0) / max(n_picks, 1),
               "interventions_per_part": self.stats["interventions"] / max(n_picks, 1),
               "reprints_per_part": self.stats["reprints"] / max(n_picks, 1),
               "lead_h_p50": float(np.nanpercentile(lead, 50)), "lead_h_p90": float(np.nanpercentile(lead, 90)),
               "caddy": self.alloc.metrics,
               "motion": dict(motion_summary(getattr(self.w, "transfer_log", [])), drive_s=round(self.drive_s, 2))}
        if self.station is not None:
            out["station"] = self.station.summary()
            out["station"]["robot_contacts"] = dict(self.contact_pairs)
            out["station"]["robot_contact_log"] = self.contact_log
        return out
