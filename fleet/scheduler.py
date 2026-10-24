"""Robot task selection over printers whose print is done (cooled or still cooling).

Pure functions over a Snapshot. Weights are hand-set heuristics (ASSUMPTIONS), not fitted.
- fifo:        oldest finish first.
- nearest:     least rail travel from the carriage.
- max_util:    weighted-shortest-robot-time rule (Smith's rule): value of restarting the printer
               (queued work for its next plate x waiting urgency) per robot-second spent on it,
               where robot time = travel + remaining cooling wait + unload. This is the
               "job length x urgency - travel - cooling" trade-off in unit-consistent form.
- order_aware: max_util x bonus for plates that complete packages / free caddy capacity;
               plates needing more new totes / tray slots than the caddy can offer are deferred. The 2-slot
               large-part tray is kept for packages that complete (tray_short); with nothing to hand off and only
               such plates waiting, the robot waits up to TRAY_HOLD_S for a slot instead of parking a package.
Caddy terms (fleet.allocator): a tote holds the small parts of one package, a tray slot one large part.
"""
from dataclasses import dataclass

URGENCY_S = 1800.0      # waiting age at which urgency doubles
MIN_WORK = 0.05         # value of an unload when nothing is queued (lead time only)
W_COMPLETE, W_FREE = 0.5, 0.5
INFEASIBLE = 0.01
WAIT_FOR_RELEASE = {"max_util", "order_aware"}   # hold off instead of picking from a warm bed
TRAY_HOLD_S = 1800.0    # order_aware: longest a plate waits so an incomplete package does not take the last tray slot


@dataclass(frozen=True)
class Candidate:
    printer: int
    x: float            # rail position for opening the door (m)
    t_done: float       # print finished (s)
    t_release: float    # bed below the release temperature (s)
    n_parts: int
    new_totes: int = 0      # packages on the plate that would need a fresh tote
    completes: int = 0      # packages whose remaining parts are all on this plate
    frees: int = 0          # ... of which already hold a tote / tray slot (a hand-off frees them)
    new_large: int = 0      # large parts on the plate (one free tray slot each)
    open_large: int = 0     # ... whose package stays incomplete after this plate (they wait in the tray)


@dataclass(frozen=True)
class Snapshot:
    t: float
    robot_x: float
    rail_speed: float
    candidates: tuple
    next_job_s: float       # print time the next plate would get from the queue (0 = nothing queued)
    max_plate_s: float
    unload_fixed_s: float   # door open/close, scan, purge tab, bed check
    pick_s: float           # mean per part
    free_slots: int         # free totes
    ready_packages: int     # complete packages on the caddy (a station visit frees their totes / slots)
    station_x: float
    station_s: float        # per package
    free_large: int = None  # free tray slots (None: no tray modelled)
    tray_wait: bool = True  # order_aware: wait rather than unload a plate whose large part finds no tray slot


def travel_s(s, c):
    return abs(c.x - s.robot_x) / s.rail_speed


def cool_wait_s(s, c):
    return max(0.0, c.t_release - s.t - travel_s(s, c))


def robot_s(s, c):
    return travel_s(s, c) + cool_wait_s(s, c) + s.unload_fixed_s + c.n_parts * s.pick_s


def max_util_score(s, c):
    work = max(MIN_WORK, min(1.0, s.next_job_s / s.max_plate_s))
    urgency = 1.0 + max(0.0, s.t - c.t_release) / URGENCY_S
    return work * urgency / robot_s(s, c)


def tray_short(s, c):
    """Tray slots the plate needs beyond what is free now. The tray (2 slots) is kept for packages that complete: a
    large part whose package stays incomplete (open_large) may not take the last free slot, unless the plate has waited
    TRAY_HOLD_S since its release."""
    if s.free_large is None:
        return 0
    short = max(0, c.new_large - s.free_large)
    if c.open_large and s.t - c.t_release < TRAY_HOLD_S:
        short += max(0, c.open_large - max(0, s.free_large - 1))
    return short


def shortfall(s, c):
    """Totes + tray slots the plate needs beyond what is free now."""
    return max(0, c.new_totes - s.free_slots) + tray_short(s, c)


def order_aware_score(s, c):
    score = max_util_score(s, c) * (1.0 + W_COMPLETE * c.completes + W_FREE * c.frees)
    short = shortfall(s, c)
    if short > 0 and short <= s.ready_packages:   # hand off first: add the station detour
        detour = 2 * abs(s.station_x - c.x) / s.rail_speed + s.ready_packages * s.station_s
        score *= robot_s(s, c) / (robot_s(s, c) + detour)
    elif short > 0:
        score *= INFEASIBLE
    return score


def order_aware_choice(s):
    """Best order_aware candidate. With tray_wait, a plate whose large part would find no tray slot (tray_short) is
    not unloaded while nothing is ready to hand off: another candidate without that shortage goes first, and if there
    is none the robot waits (None) - a tray slot frees when a held package's last part arrives - for at most
    TRAY_HOLD_S after the plate's release."""
    best = lambda cs: max(cs, key=lambda c: (order_aware_score(s, c), -c.printer))
    c = best(s.candidates)
    if s.tray_wait and not s.ready_packages and tray_short(s, c):
        ok = [x for x in s.candidates if not tray_short(s, x)]
        if ok:
            return best(ok)
        overdue = [x for x in s.candidates if s.t - x.t_release >= TRAY_HOLD_S]
        return best(overdue) if overdue else None
    return c


POLICIES = {
    "fifo": lambda s: min(s.candidates, key=lambda c: (c.t_done, c.printer)),
    "nearest": lambda s: min(s.candidates, key=lambda c: (abs(c.x - s.robot_x), c.t_done, c.printer)),
    "max_util": lambda s: max(s.candidates, key=lambda c: (max_util_score(s, c), -c.printer)),
    "order_aware": order_aware_choice,
}


def choose(policy, snap):
    """Candidate to unload next, or None if there is none (order_aware: or if it waits for a tray slot)."""
    return POLICIES[policy](snap) if snap.candidates else None
