"""Seeded synthetic order stream (all rates and mixes are ASSUMPTIONS).

Arrivals are a piecewise-constant Poisson process: `rate_day` orders/h between `day_hours`,
`rate_night` outside, the same every day of the week (online shop). Time is seconds from Monday 00:00.
Line items come from sample_lines (shared with sim/farm.py); each order ships as jobs.packaging.plan_packages says.
"""
from dataclasses import dataclass
from functools import cached_property
import numpy as np

from farm.catalog import CATALOG, ORDER_SKUS
from jobs.packaging import plan_packages

CARRIERS = {"courier_A": 0.45, "courier_B": 0.35, "post": 0.20}   # share of orders (ASSUMPTION)
SLA_H = {"courier_A": 24.0, "courier_B": 24.0, "post": 48.0}       # ship-by after arrival (ASSUMPTION)
LINES_P = (0.45, 0.30, 0.15, 0.10)    # P(1..4 line items)
QTY_P = (0.60, 0.30, 0.10)            # P(qty 1..3)
P_BULK = 0.08                         # P(one line of the order is a bulk line) (ASSUMPTION)
BULK_QTY = (4, 8)                     # bulk line quantity, uniform (ASSUMPTION)
LARGE_LINE_SHARE = 0.15               # share of the SKU draw weight on large SKUs, when offered (ASSUMPTION)


def sample_lines(rng, skus=None):
    """Line items ((sku, qty), ...) of one order, distinct SKUs, from `skus` (default ORDER_SKUS; taken in sorted
    order, so the caller's ordering does not matter). 1..4 lines (LINES_P), qty 1..3 (QTY_P); small SKUs equally
    likely, large SKUs (if offered) share LARGE_LINE_SHARE of the weight; with P_BULK one line, drawn uniformly, gets a
    bulk quantity (BULK_QTY), so multi-package orders occur."""
    skus = sorted(ORDER_SKUS if skus is None else skus)
    n_lines = min(int(rng.choice(len(LINES_P), p=LINES_P)) + 1, len(skus))
    n_large = sum(CATALOG[s].large for s in skus)
    if 0 < n_large < len(skus):
        w = [LARGE_LINE_SHARE / n_large if CATALOG[s].large else (1 - LARGE_LINE_SHARE) / (len(skus) - n_large)
             for s in skus]
        picked = rng.choice(skus, n_lines, replace=False, p=w)
    else:
        picked = rng.choice(skus, n_lines, replace=False)
    lines = [(str(s), int(rng.choice(len(QTY_P), p=QTY_P)) + 1) for s in picked]
    if rng.random() < P_BULK:
        k = int(rng.integers(n_lines))
        lines[k] = (lines[k][0], int(rng.integers(BULK_QTY[0], BULK_QTY[1] + 1)))
    return tuple(lines)


@dataclass(frozen=True)
class Order:
    order_id: str
    t_s: float
    customer: str
    carrier: str
    ship_by_s: float
    lines: tuple      # ((sku, qty), ...), distinct SKUs

    @property
    def parts(self):
        return [f"{self.order_id}_{sku}_{k}" for sku, q in self.lines for k in range(1, q + 1)]

    @property
    def volume_cm3(self):
        return sum(CATALOG[sku].volume_cm3 * q for sku, q in self.lines)

    @cached_property
    def packages(self):
        """(Package, ...) the order ships as (jobs.packaging.plan_packages)."""
        return tuple(plan_packages(self.order_id, self.parts))

    @property
    def boxes(self):
        return tuple(p.box for p in self.packages)

    def book_entry(self):
        """Entry for the CaddyAllocator order book."""
        return {"parts": self.parts, "packages": self.packages, "volume_cm3": self.volume_cm3,
                "carrier": self.carrier}


def part_sku(name):
    """'o12_knob_1' -> 'knob'; '<plate>_purge_tab' -> 'purge_tab'."""
    return "purge_tab" if name.endswith("purge_tab") else name.split("_")[1]


@dataclass
class OrderStream:
    seed: int = 0
    rate_day: float = 6.0       # orders/h during day_hours (ASSUMPTION)
    rate_night: float = 1.5     # orders/h otherwise (ASSUMPTION)
    day_hours: tuple = (8, 20)
    scale: float = 1.0          # multiplies both rates (e.g. >1 for a saturated capacity run)
    n_customers: int = 500
    skus: tuple = None          # SKUs on offer (None: ORDER_SKUS)

    def rate(self, t_s):
        h = (t_s % 86400.0) / 3600.0
        lo, hi = self.day_hours
        return self.scale * (self.rate_day if lo <= h < hi else self.rate_night)

    def generate(self, t_end_s):
        """All orders arriving in [0, t_end_s), sorted by time; ids o0, o1, ..."""
        rng = np.random.default_rng(self.seed)
        times = []
        for h0 in np.arange(0.0, t_end_s, 3600.0):
            span = min(3600.0, t_end_s - h0)
            n = rng.poisson(self.rate(h0) * span / 3600.0)
            times.extend(h0 + np.sort(rng.uniform(0.0, span, n)))
        orders = []
        for i, t in enumerate(times):
            lines = sample_lines(rng, self.skus)
            carrier = str(rng.choice(list(CARRIERS), p=list(CARRIERS.values())))
            orders.append(Order(f"o{i}", float(t), f"c{rng.integers(self.n_customers):04d}", carrier,
                                float(t) + SLA_H[carrier] * 3600.0, lines))
        return orders
