"""Plots from the DES outputs (python -m fleet.des --compare) -> PNGs at 200 dpi.

    python -m report.plots --des runs/des --out runs/plots
    python -m report.plots --layout runs/layout --out runs/layout      (printers-per-arm study, scripts/layout_study.py)

(a) gantt_48h.png      printer states over 48 h, robot (order_aware) vs human baseline
(b) kpi_compare.png    utilisation / waiting hours / lead time p90 / robot time at the packing station, 4 policies +
                       human (+ packages per order)
(c) economics.png      payback = d + C/B for 1 / 10 / 100 cells; every money figure is an ASSUMPTION (ECON)
(d) layout_knee.png, layout_sensitivity.png, layout_economics.png   printers per arm, one row vs two facing rows
"""
import argparse, csv, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from farm.catalog import N_PRINTERS

ECON = {                                  # USD; ALL ASSUMPTIONS, none measured
    "yam_arm": 3000.0,                    # list price per arm, verify with a quote
    "rail_carriage": 1500.0,
    "door_pins_caddy": 100.0,
    "camera_compute": 400.0,              # wrist camera + edge PC
    "station_bed_to_box": 4500.0,         # packing station at STATION_X: box magazine 600, conveyor + 2 lifts 1500,
                                          # tape sealer 1500, carrier lanes 300, station camera 300, trap-door totes
                                          # + large-part tray 300 (was a 500 bin exchanger before bed to box)
    "integration_one_off": 40000.0,       # engineering to productise the cell (once per fleet)
    "setup_per_cell": 1500.0,             # install + calibration
    "maintenance_per_cell_month": 50.0,
    "value_per_printer_h": 1.50,          # contribution margin of one extra printer-hour
    "deploy_months": {1: 2.0, 10: 4.0, 100: 9.0},   # d: time until the fleet is live
}
HW_KEYS = ("yam_arm", "rail_carriage", "door_pins_caddy", "camera_compute", "station_bed_to_box")

STATE_COLORS = {"printing": "#2B8FD6", "cooling": "#F2C14E", "waiting": "#E03E5A", "blocked": "#9E1F3A",
                "unloading": "#F28C28", "idle": "#D9DCE1"}
STATE_LABELS = {"printing": "printing", "cooling": "cooling", "waiting": "waiting for unload",
                "blocked": "blocked (stuck part, waits for a person)", "unloading": "unloading", "idle": "idle (no work)"}
ROBOT, ROBOT_LIGHT, HUMAN, STATION = "#2a78d6", "#9ec5f4", "#eb6834", "#1baf7a"
FLEET_COLORS = {1: "#2a78d6", 10: "#eb6834", 100: "#1baf7a"}
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
POLICY_LABELS = {"fifo": "FIFO", "nearest": "Nearest", "max_util": "Max\nutil", "order_aware": "Order\naware",
                 "human": "Human\nbaseline"}
DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"], "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13, "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "savefig.dpi": 200,
})


def footnote(fig, text):
    fig.text(0.01, 0.012, text, fontsize=11, color=MUTED, ha="left", va="bottom")


def read_events(path):
    states, busy = {i: [] for i in range(N_PRINTERS)}, []
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["kind"] == "printer_state":
                states.setdefault(int(r["printer"]), []).append((float(r["t_s"]), r["detail"]))
            elif r["kind"].endswith("_busy"):
                busy.append((float(r["t_s"]), float(r["detail"])))
    return states, busy


def clock(t_s):
    d, h = divmod(t_s / 3600.0, 24.0)
    return f"{DAYS[int(d) % 7]} {int(h):02d}:00"


def gantt(des, out, t0_h=60.0, hours=48.0, shift=(9.0, 18.0), workdays=range(5)):
    t0, t1 = t0_h * 3600, (t0_h + hours) * 3600
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    for ax, run, title, actor in zip(axes, ("order_aware", "human"),
                                     ("Robot cell (order-aware dispatch)", "Human baseline (hourly checks, Mon-Fri 09-18)"),
                                     ("robot", "operator")):
        states, busy = read_events(Path(des) / run / "events.csv")
        for i, seq in states.items():
            ends = [t for t, _ in seq[1:]] + [t1]
            for (a, s), b in zip(seq, ends):
                a, b = max(a, t0), min(b, t1)
                if b > a:
                    ax.broken_barh([((a - t0) / 3600, (b - a) / 3600)], (i - 0.4, 0.8), color=STATE_COLORS[s], lw=0)
        y = N_PRINTERS + 0.3
        if actor == "operator":
            for day in range(int(t0 // 86400), int(t1 // 86400) + 1):
                if day % 7 in workdays:
                    a, b = max(day * 86400 + shift[0] * 3600, t0), min(day * 86400 + shift[1] * 3600, t1)
                    if b > a:
                        ax.broken_barh([((a - t0) / 3600, (b - a) / 3600)], (y - 0.35, 0.7), color="#efeee9", lw=0)
        segs = [((max(a, t0) - t0) / 3600, (min(b, t1) - max(a, t0)) / 3600) for a, b in busy if b > t0 and a < t1]
        ax.broken_barh([s for s in segs if s[1] > 0], (y - 0.35, 0.7), color=INK2, lw=0)
        ax.set_yticks([*range(N_PRINTERS), y], [f"P{i}" for i in range(N_PRINTERS)] + [f"{actor} busy"])
        ax.set_ylim(y + 0.7, -0.7)
        ax.set_title(title, pad=8)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    ticks = np.arange(0, hours + 1, 6)
    axes[1].set_xticks(ticks, [clock(t0 + h * 3600) for h in ticks])
    axes[1].set_xlim(0, hours)
    used = ["printing", "cooling", "waiting", "blocked", "unloading", "idle"]
    fig.legend([Patch(color=STATE_COLORS[s]) for s in used], [STATE_LABELS[s] for s in used], ncol=3,
               loc="upper right", bbox_to_anchor=(0.99, 0.995), frameon=False, handlelength=1.2, columnspacing=1.2)
    fig.suptitle(f"{N_PRINTERS} printers over 48 hours", x=0.01, ha="left", fontsize=20, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.84, bottom=0.1, hspace=0.28)
    footnote(fig, "Simulation, assumed parameters (fleet/des.py). Seed 0, same order stream in both panels. "
                  "Gray band in the operator row = shift hours.")
    fig.savefig(out / "gantt_48h.png")
    plt.close(fig)


def kpi_compare(cmp, out):
    runs = cmp["runs"]
    pols = list(POLICY_LABELS)
    x = np.arange(len(pols))
    colors = [HUMAN if p == "human" else ROBOT for p in pols]
    mean = lambda k: np.array([runs[p]["mean"][k] for p in pols], float)
    std = lambda k: np.array([runs[p]["std"].get(k, 0.0) for p in pols], float)
    n_pr = int(runs["order_aware"]["mean"]["n_printers"])
    fig, axes = plt.subplots(1, 4, figsize=(20, 6.6))
    panels = [("printer_util_pct", "Printer utilisation", "% of printer-hours printing", "{:.0f}%"),
              ("printer_waiting_h", "Printer-hours lost waiting", f"hours per week ({n_pr} printers)", "{:.0f} h"),
              ("lead_time_p90_h", "Order lead time, p90", "hours, arrival to last package shipped", "{:.0f} h"),
              ("robot_util_pct", "Robot busy", "% of the week", "{:.0f}%")]
    for ax, (key, title, ylabel, fmt) in zip(axes, panels):
        if key == "robot_util_pct":                     # robot only: unloading vs the packing station
            rp = [p for p in pols if p != "human"]
            xr = np.arange(len(rp))
            m = np.array([runs[p]["mean"][key] for p in rp], float)
            st = np.array([runs[p]["mean"].get("station_busy_pct") or 0.0 for p in rp], float)
            ax.bar(xr, m - st, 0.68, color=ROBOT)
            ax.bar(xr, st, 0.68, bottom=m - st, color=STATION)
            for xi, mi, si in zip(xr, m, st):
                ax.text(xi, mi + 0.02 * m.max(), fmt.format(mi), ha="center", va="bottom", fontsize=13, color=INK)
            ax.set_xticks(xr, [POLICY_LABELS[p] for p in rp])
            ax.set_title(title, pad=10)
            ax.set_ylabel(ylabel)
            ax.set_ylim(0, 1.25 * m.max())
            ax.yaxis.grid(True, color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            ax.tick_params(axis="x", length=0)
            continue
        m, s = mean(key), std(key)
        if key == "printer_waiting_h":
            blocked = mean("printer_blocked_h")
            ax.bar(x, m - blocked, 0.68, color=colors)
            ax.bar(x, blocked, 0.68, bottom=m - blocked + 0.004 * m.max(),
                   color=[ROBOT_LIGHT if p != "human" else HUMAN for p in pols])
        else:
            ax.bar(x, m, 0.68, color=colors)
        ax.errorbar(x, m, yerr=s, fmt="none", ecolor=INK2, elinewidth=1.2, capsize=4)
        for xi, mi, si in zip(x, m, s):
            ax.text(xi, mi + si + 0.02 * m.max(), fmt.format(mi), ha="center", va="bottom", fontsize=13, color=INK)
        ax.set_xticks(x, [POLICY_LABELS[p] for p in pols])
        ax.set_title(title, pad=10)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.18 * (m + s).max())
        ax.yaxis.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", length=0)
    handles = [Patch(color=ROBOT), Patch(color=ROBOT_LIGHT), Patch(color=STATION), Patch(color=HUMAN)]
    fig.legend(handles, ["Robot, by dispatch policy", "Robot: blocked by a stuck part outside shifts",
                         "Robot at the packing station (bed to box)", "Human baseline"], ncol=4, loc="upper left",
               bbox_to_anchor=(0.005, 0.875), frameon=False)
    fig.suptitle(f"Robot policies vs human baseline, one {n_pr}-printer cell, one week", x=0.01, ha="left",
                 fontsize=20, fontweight="bold", y=0.985)
    oa = runs["order_aware"]["mean"]
    st = oa.get("station_busy_pct")
    fig.text(0.01, 0.905, f"The robot is busy {oa['robot_util_pct']:.0f}% of the time"
                          + (f" ({st:.0f}% at the packing station)" if st is not None else "")
                          + ", so the dispatch policy barely moves printer KPIs"
                          + (f"; orders ship as {oa['packages_per_order']:.2f} packages on average."
                             if "packages_per_order" in oa else "."),
             fontsize=14, color=INK2, ha="left")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.71, bottom=0.17, wspace=0.3)
    n = len(cmp["seeds"])
    footnote(fig, f"Simulation, assumed parameters (fleet/des.py; robot timings: {oa['timings_source']}"
                  + (f"; station: {oa['station_source']}" if "station_source" in oa else "")
                  + f"). Mean of {n} seeds, error bars = 1 sd. {cmp['days']:g} days of orders"
                  + (f", SKU mix: {cmp['skus']}" if "skus" in cmp else "") + ".")
    fig.savefig(out / "kpi_compare.png")
    plt.close(fig)


def economics(cmp, out, econ=ECON):
    cap = cmp["capacity"]
    days = cmp["days"]
    extra_h_month = (cap["order_aware"]["mean"]["printing_h"] - cap["human"]["mean"]["printing_h"]) / days * 30.4
    hw = sum(econ[k] for k in HW_KEYS)
    benefit = lambda v: extra_h_month * v - econ["maintenance_per_cell_month"]       # per cell per month
    cost = lambda n: econ["integration_one_off"] + n * (hw + econ["setup_per_cell"])
    payback = lambda n, v: econ["deploy_months"][n] + cost(n) / (n * benefit(v))
    fleets = list(econ["deploy_months"])
    fig = plt.figure(figsize=(16, 7.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 0.85], wspace=0.32)
    ax = fig.add_subplot(gs[0])
    v0 = econ["value_per_printer_h"]
    y = np.arange(len(fleets))
    d = np.array([econ["deploy_months"][n] for n in fleets])
    cb = np.array([cost(n) / (n * benefit(v0)) for n in fleets])
    ax.barh(y, d, 0.6, color="#c3c2b7")
    ax.barh(y, cb, 0.6, left=d + 0.15, color=[FLEET_COLORS[n] for n in fleets])
    for yi, di, ci in zip(y, d, cb):
        ax.text(di + ci + 0.6, yi, f"{di + ci:.0f} months", va="center", fontsize=14, color=INK)
    i = int(np.argmax(np.minimum(d, cb)))                       # label the segments where both fit
    ax.text(d[i] / 2, y[i], "d", ha="center", va="center", fontsize=14, color=INK)
    ax.text(d[i] + 0.15 + cb[i] / 2, y[i], "C / B", ha="center", va="center", fontsize=14, color="white")
    ax.set_yticks(y, [f"{n} cell" + ("s" if n > 1 else "") for n in fleets])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.3 * (d + cb).max())
    ax.set_xlabel("months to payback")
    ax.set_title("Payback = d (deployment) + C / B", pad=10)
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax2 = fig.add_subplot(gs[1])
    vs = np.linspace(0.75, 4.0, 60)
    vs = vs[np.array([benefit(v) for v in vs]) > 0]
    for n in fleets:
        p = [payback(n, v) for v in vs]
        ax2.plot(vs, p, color=FLEET_COLORS[n], lw=2)
        ax2.text(vs[-1] + 0.05, p[-1], f"{n} cell" + ("s" if n > 1 else ""), va="center", fontsize=13, color=INK2)
    ax2.axvline(v0, color=MUTED, lw=1)
    ax2.text(v0 + 0.05, 0.97, "base case", transform=ax2.get_xaxis_transform(), fontsize=12, color=MUTED, va="top")
    ax2.set_xlim(vs[0], vs[-1] + 0.6)
    ax2.set_ylim(0, min(72, 1.1 * max(payback(n, vs[0]) for n in fleets)))
    ax2.set_xlabel("value of one extra printer-hour (USD)")
    ax2.set_ylabel("months to payback")
    ax2.set_title("Sensitivity to printer-hour value", pad=10)
    ax2.yaxis.grid(True, color=GRID, lw=0.8)
    ax2.set_axisbelow(True)
    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")
    lines = ["ASSUMPTIONS, USD (not quotes)",
             f"YAM arm (list price)  {econ['yam_arm']:,.0f}", f"Rail + carriage  {econ['rail_carriage']:,.0f}",
             f"Door handles + caddy + trays  {econ['door_pins_caddy']:,.0f}", f"Camera + compute  {econ['camera_compute']:,.0f}",
             f"Packing station (bed to box)  {econ['station_bed_to_box']:,.0f}",
             f"Setup per cell  {econ['setup_per_cell']:,.0f}",
             f"Integration, one-off  {econ['integration_one_off']:,.0f}",
             f"Maintenance  {econ['maintenance_per_cell_month']:,.0f} / cell / month",
             f"Printer-hour value  {v0:.2f}", "d = " + ", ".join(f"{m:g} mo ({n})" for n, m in econ['deploy_months'].items()),
             "", "FROM THE DES, capacity runs",
             f"+{extra_h_month:,.0f} printer-h / cell / month",
             f"B = {benefit(v0):,.0f} / cell / month", f"C = {econ['integration_one_off']:,.0f} + n x {hw + econ['setup_per_cell']:,.0f}"]
    for i, s in enumerate(lines):
        bold = s.startswith(("ASSUMPTIONS", "FROM THE DES"))
        ax3.text(0.0, 1.0 - i * 0.064, s, transform=ax3.transAxes, fontsize=12.5 if not bold else 13,
                 fontweight="bold" if bold else "normal", color=INK if bold else INK2, va="top")
    fig.suptitle("Economics: robot cells vs the human baseline", x=0.01, ha="left", fontsize=20, fontweight="bold",
                 y=0.975)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.17)
    footnote(fig, f"Simulation, assumed parameters. B counts only printer-hours recovered under saturated demand "
                  f"(orders x{cmp['capacity_scale']:g}, order-aware robot vs human); labour saved is left out of B; "
                  "the packing station is in C.")
    fig.savefig(out / "economics.png")
    plt.close(fig)
    return {"extra_printer_h_per_cell_month": round(extra_h_month, 1), "benefit_usd_per_cell_month": round(benefit(v0)),
            "payback_months": {n: round(payback(n, v0), 1) for n in fleets}}


# ---------------- printers per arm (scripts/layout_study.py) ----------------
AQUA = "#1baf7a"
LAYOUT_STYLE = {("row", "end"): (ROBOT, "-"), ("row", "center"): (ROBOT, "--"),
                ("facing", "end"): (AQUA, "-"), ("facing", "center"): (AQUA, "--")}
LAYOUT_LABELS = {("row", "end"): "one row, station at the end", ("row", "center"): "one row, station mid-rail",
                 ("facing", "end"): "two facing rows, station at the end",
                 ("facing", "center"): "two facing rows, station mid-rail"}
SENS_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
ECON_LAYOUT = {                           # USD; ASSUMPTIONS: ECON's per-cell items split so they scale with the layout
    "carriage": 700.0,                    # carriage, drive, cable chain (ECON rail_carriage 1500 = 700 + 5.5 m x 145)
    "rail_per_m": 145.0,                  # profile rail + belt per metre of travel
    "caddy": 50.0,                        # ECON door_pins_caddy 100 = caddy 50 + 10 printers x 5
    "per_printer_kit": 5.0,               # D-handle + reject tray, per printer
    "chamber_camera_per_printer": 0.0,    # base case: the arm's camera (ECON camera_compute); the sim's per-printer
                                          # chamber depth camera would add its unit price here
    "hours_per_month": 730.0,
    "farm_printers": 120,                 # farm size for the fleet payback panel
}


def cell_cost(n, rail_m, econ=ECON, el=ECON_LAYOUT):
    """Hardware + setup of one arm cell serving n printers on a rail of rail_m metres (USD, ASSUMPTIONS)."""
    return (econ["yam_arm"] + el["carriage"] + el["rail_per_m"] * rail_m + el["caddy"] + econ["camera_compute"]
            + econ["station_bed_to_box"] + econ["setup_per_cell"]
            + n * (el["per_printer_kit"] + el["chamber_camera_per_printer"]))


def layout_rows(study, mode="cap", tag="base"):
    """{(layout, station): [row by n]} from des_study.json / des_sens.json."""
    out = {}
    for r in study["rows"]:
        if r["mode"] == mode and r["tag"] == tag:
            out.setdefault((r["layout"], r["station"]), []).append(r)
    return {k: sorted(v, key=lambda r: r["n"]) for k, v in out.items()}


def human_util(study, mode="cap"):
    return next(r["printer_util_pct"] for r in study["rows"] if r["tag"] == "human" and r["mode"] == mode)


def layout_economics_table(study, econ=ECON, el=ECON_LAYOUT):
    """Per layout and printers per arm: cost per printer, extra printer-hours vs the human baseline, payback.
    B = n x (robot util - human util) x hours x value - maintenance (saturated demand; labour left out of B and C,
    as in economics()); fleet payback for el['farm_printers'] printers = d(10 cells) + (integration + cells x C) /
    (cells x B)."""
    u_h = human_util(study, "cap")
    v, hpm, F = econ["value_per_printer_h"], el["hours_per_month"], el["farm_printers"]
    d = econ["deploy_months"][10]
    table = []
    for key, rows in layout_rows(study, "cap").items():
        for r in rows:
            n, c = r["n"], cell_cost(r["n"], r["rail_length_m"], econ, el)
            extra_h = (r["printer_util_pct"] - u_h) / 100 * hpm          # per printer per month
            b = n * extra_h * v - econ["maintenance_per_cell_month"]
            cells = F / n
            table.append({"layout": key[0], "station": key[1], "n": n, "rail_m": r["rail_length_m"],
                          "cell_cost_usd": round(c), "cost_per_printer_usd": round(c / n),
                          "extra_printer_h_per_printer_month": round(extra_h, 1),
                          "benefit_usd_per_cell_month": round(b), "cell_payback_months": round(c / b, 1) if b > 0 else None,
                          "farm_payback_months": round(d + (econ["integration_one_off"] + cells * c) / (cells * b), 1)
                          if b > 0 else None})
    return {"human_util_pct": u_h, "farm_printers": F, "rows": table}


def _layout_axes(ax, title, ylabel):
    ax.set_title(title, pad=8)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("printers per arm")
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def layout_knee(study, out, mark=None):
    """Capacity (saturated demand) and lead time (demand per printer as today) vs printers per arm."""
    cap, prop = layout_rows(study, "cap"), layout_rows(study, "prop")
    panels = [(cap, "printer_util_pct", "Printer utilisation (saturated demand)", "% of printer-hours printing"),
              (cap, "arm_wait_pct", "Printer time lost waiting for the arm", "% of printer-hours"),
              (cap, "robot_util_pct", "Arm busy", "% of the week"),
              (cap, "parts_per_day", "Output per arm (saturated demand)", "good parts per day"),
              (prop, "lead_time_p90_h", "Order lead time p90 (today's demand per printer)", "hours"),
              (cap, "caddy_full_events", "Caddy full mid-plate (saturated demand)", "events per week")]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5))
    for ax, (data, key, title, ylabel) in zip(axes.flat, panels):
        for lay, rows in data.items():
            color, ls = LAYOUT_STYLE[lay]
            n = [r["n"] for r in rows]
            m = np.array([r[key] for r in rows], float)
            sd = np.array([r.get(key + "_sd", 0.0) for r in rows], float)
            ax.plot(n, m, color=color, ls=ls, lw=2, marker="o", ms=4)
            ax.fill_between(n, m - sd, m + sd, color=color, alpha=0.08, lw=0)
        if key == "printer_util_pct":
            u_h = human_util(study, "cap")
            ax.axhline(u_h, color=HUMAN, lw=1.5)
            ax.text(ax.get_xlim()[0] + 0.5, u_h + 1, f"human baseline, 10 printers: {u_h:.0f}%", color=INK2, fontsize=12)
        if key == "arm_wait_pct":
            ax.axhline(5.0, color=MUTED, lw=1, ls=":")
            ax.text(ax.get_xlim()[0] + 0.5, 5.6, "knee rule: 5 %", color=MUTED, fontsize=12)
        if mark:
            ax.axvline(mark, color=MUTED, lw=1)
        _layout_axes(ax, title, ylabel)
    handles = [plt.Line2D([], [], color=c, ls=ls, lw=2) for c, ls in LAYOUT_STYLE.values()]
    fig.legend(handles, list(LAYOUT_LABELS.values()), ncol=4, loc="upper left", bbox_to_anchor=(0.005, 0.93),
               frameon=False, handlelength=2.4)
    fig.suptitle("How many printers can one arm serve?", x=0.01, ha="left", fontsize=20, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.84, bottom=0.08, hspace=0.42, wspace=0.25)
    footnote(fig, f"Simulation (fleet/des.py), robot timings measured in MuJoCo (runs/timings.json); order rates ASSUMED. "
                  f"{study['days']:g} days, mean of {len(study['seeds'])} seeds, band = 1 sd. Saturated demand = "
                  f"{study['modes']['cap']:g} x today's orders per printer."
                  + (f" Vertical line: recommended {mark} printers." if mark else ""))
    fig.savefig(out / "layout_knee.png")
    plt.close(fig)


def layout_sensitivity(sens, out):
    """Printer waiting share vs printers per arm under what-if variants (one layout, saturated demand)."""
    rows = {}
    for r in sens["rows"]:
        rows.setdefault(r["tag"], []).append(r)
    fig, axes = plt.subplots(1, 2, figsize=(17, 6.8))
    for ax, key, title, ylabel in ((axes[0], "arm_wait_pct", "Printer time lost waiting for the arm", "% of printer-hours"),
                                   (axes[1], "parts_per_printer_day", "Output per printer", "good parts per printer-day")):
        for (tag, rs), color in zip(rows.items(), SENS_COLORS):
            rs = sorted(rs, key=lambda r: r["n"])
            ax.plot([r["n"] for r in rs], [r[key] for r in rs], color=color, lw=2.6 if tag == "base" else 1.8,
                    marker="o", ms=4, label=tag.replace("_", " "))
        if key == "arm_wait_pct":
            ax.axhline(5.0, color=MUTED, lw=1, ls=":")
        _layout_axes(ax, title, ylabel)
    axes[0].legend(frameon=False, fontsize=12, loc="upper left")
    lay = sens["rows"][0]
    fig.suptitle(f"What moves the knee ({LAYOUT_LABELS[(lay['layout'], lay['station'])]})", x=0.01, ha="left",
                 fontsize=20, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.14, wspace=0.22)
    footnote(fig, "Simulation, saturated demand. Variants are ASSUMED what-ifs around the measured timings: plate length cap, "
                  "arm 2x faster, 12 order bins, 1 m/s rail, 2 % failed picks/part with the measured rake share.")
    fig.savefig(out / "layout_sensitivity.png")
    plt.close(fig)


def layout_economics(study, out, econ=ECON, el=ECON_LAYOUT, mark=None):
    tab = layout_economics_table(study, econ, el)
    by = {}
    for r in tab["rows"]:
        by.setdefault((r["layout"], r["station"]), []).append(r)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.8))
    panels = [("cost_per_printer_usd", "Cell cost per printer served", "USD per printer"),
              ("extra_printer_h_per_printer_month", "Printer-hours gained vs human", "hours per printer per month"),
              ("farm_payback_months", f"Payback, {tab['farm_printers']}-printer farm", "months (d + C / B)")]
    for ax, (key, title, ylabel) in zip(axes, panels):
        for lay, rows in by.items():
            color, ls = LAYOUT_STYLE[lay]
            rows = [r for r in sorted(rows, key=lambda r: r["n"]) if r[key] is not None]
            ax.plot([r["n"] for r in rows], [r[key] for r in rows], color=color, ls=ls, lw=2, marker="o", ms=4)
        if mark:
            ax.axvline(mark, color=MUTED, lw=1)
        _layout_axes(ax, title, ylabel)
        ax.set_ylim(bottom=0)
    handles = [plt.Line2D([], [], color=c, ls=ls, lw=2) for c, ls in LAYOUT_STYLE.values()]
    fig.legend(handles, list(LAYOUT_LABELS.values()), ncol=4, loc="upper left", bbox_to_anchor=(0.005, 0.9),
               frameon=False, handlelength=2.4)
    fig.suptitle("Economics vs printers per arm", x=0.01, ha="left", fontsize=20, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.76, bottom=0.15, wspace=0.25)
    footnote(fig, f"ASSUMED prices (report/plots.py ECON, ECON_LAYOUT): arm {econ['yam_arm']:,.0f}, carriage "
                  f"{el['carriage']:,.0f} + rail {el['rail_per_m']:,.0f}/m, {el['per_printer_kit']:,.0f}/printer kit, "
                  f"printer-hour {econ['value_per_printer_h']:.2f}, integration {econ['integration_one_off']:,.0f}, "
                  f"d = {econ['deploy_months'][10]:g} months. Human baseline {tab['human_util_pct']:.0f}% util; "
                  "labour left out of B and C.")
    fig.savefig(out / "layout_economics.png")
    plt.close(fig)
    return tab


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--des", default="runs/des")
    ap.add_argument("--out", default="runs/plots")
    ap.add_argument("--gantt-start-h", type=float, default=60.0, help="hours after Monday 00:00")
    ap.add_argument("--layout", help="printers-per-arm study dir (des_study.json [+ des_sens.json]) -> layout_*.png")
    ap.add_argument("--mark", type=int, help="--layout: printers per arm to mark (the recommendation)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.layout:
        study = json.loads((Path(a.layout) / "des_study.json").read_text())
        layout_knee(study, out, a.mark)
        tab = layout_economics(study, out, mark=a.mark)
        (out / "layout_economics.json").write_text(json.dumps(tab, indent=1))
        if (Path(a.layout) / "des_sens.json").exists():
            layout_sensitivity(json.loads((Path(a.layout) / "des_sens.json").read_text()), out)
        print("wrote", *sorted(str(p) for p in out.glob("layout_*")))
        return
    cmp = json.loads((Path(a.des) / "kpis_compare.json").read_text())
    gantt(a.des, out, a.gantt_start_h)
    kpi_compare(cmp, out)
    print(json.dumps(economics(cmp, out), indent=1))
    print("wrote", *sorted(str(p) for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
