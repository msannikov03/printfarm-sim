"""Top-down drawing of the two cell layouts at true scale: today's single row and the recommended two facing rows
(aisle, carriage deck, caddy bins, arm base, the open door leaf and the handle stand-off that sets the aisle width).

    python scripts/layout_figure.py [--aisle 0.40 --caddy plusx] -> runs/layout/layout_plan.png
"""
import argparse
import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Rectangle

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from farm import catalog as C                     # noqa: E402
from report.plots import INK, INK2, MUTED, GRID, ROBOT, AQUA, HUMAN, footnote   # noqa: E402,F401

PRINTER, BENCH, DECK, BIN, LEAF = "#3a3b3f", "#e9e8e3", "#9ec5f4", "#1baf7a", "#eb6834"


def draw_cell(ax, rows, aisle, caddy, cols=4, carriage_col=1, title=""):
    pitch, W, D = C.PITCH, C.PRINTER_W, C.PRINTER_D
    sides = (+1, -1) if rows == 2 else (+1,)
    x0, x1 = -0.45, (cols - 1) * pitch + 0.45
    for s in sides:                                   # bench + printers + closed doors with their D-handles
        y_b = s * aisle
        ax.add_patch(Rectangle((x0, min(y_b, y_b + s * 0.5)), x1 - x0, 0.5, color=BENCH, lw=0))
        for c in range(cols):
            xc = c * pitch
            ax.add_patch(Rectangle((xc - W / 2, min(y_b, y_b + s * D)), W, D, color=PRINTER, lw=0))
            if not (s == +1 and c == carriage_col):
                hx = xc + s * (C.HINGE_XL + C.PIN_S)      # handle on a closed door: stand-off into the aisle
                ax.plot([hx, hx], [y_b, y_b - s * C.PIN_OFFSET], color=LEAF, lw=2.2, solid_capstyle="butt")
                ax.add_patch(Circle((hx, y_b - s * C.PIN_OFFSET), 0.009, color=LEAF, lw=0))
    ax.plot([x0 - 0.2, x1 + 0.2], [0, 0], color=MUTED, lw=1, ls=(0, (6, 4)))          # rail axis
    ax.text(x1 + 0.22, 0, "rail", va="center", fontsize=11, color=MUTED)
    # the working printer (row A, column carriage_col): door open at 100 deg, carriage at its door-pull position
    xc = carriage_col * pitch
    hinge = (xc + C.HINGE_XL, aisle)
    th = C.DOOR_OPEN_RAD
    tip = (hinge[0] + C.DOOR_W * math.cos(th), hinge[1] - C.DOOR_W * math.sin(th))
    ax.plot([hinge[0], tip[0]], [hinge[1], tip[1]], color=LEAF, lw=3)
    ax.text(tip[0] - 0.03, tip[1], "open door\n(100 deg)", ha="right", va="center", fontsize=10.5, color=INK2)
    cx = xc + 0.15
    if caddy is None:   # today's carriage (farm/scene.py): deck y -0.46 .. 0.15, bins behind the arm
        deck = (cx - 0.27, -0.46, 0.54, min(0.15, aisle - 0.04) + 0.46)
        bins = C.CADDY_SLOTS
    else:
        from layout_geometry import deck_extent, plusx_caddy, side_caddy
        bins = {"plusx": plusx_caddy, "side": side_caddy}[caddy](aisle)
        saved = C.CADDY_SLOTS, C.AISLE_HALF
        C.CADDY_SLOTS, C.AISLE_HALF = bins, aisle
        x0_, x1_, y0, y1 = deck_extent(C)
        C.CADDY_SLOTS, C.AISLE_HALF = saved
        deck = (cx + x0_, y0, x1_ - x0_, y1 - y0)
    ax.add_patch(Rectangle(deck[:2], deck[2], deck[3], color=DECK, lw=0, alpha=0.9))
    for slot, (dx, dy) in bins.items():
        ax.add_patch(Rectangle((cx + dx - 0.055, dy - 0.055), 0.11, 0.11, fc="none", ec=BIN, lw=1.6))
    ax.add_patch(Circle((cx, 0), 0.055, color=ROBOT, lw=0))
    ax.text(cx, -0.02 if caddy else 0.07, "arm", ha="center", va="top" if caddy else "bottom", fontsize=10.5,
            color="white" if caddy else INK)
    # dimensions: aisle, pitch
    xd = x0 - 0.12
    for s in sides:
        ax.annotate("", (xd, 0), (xd, s * aisle), arrowprops=dict(arrowstyle="<->", color=INK2, lw=1))
        ax.text(xd - 0.03, s * aisle / 2, f"{aisle:.2f} m", ha="right", va="center", fontsize=11, color=INK2)
    yd = aisle + D + 0.08
    ax.annotate("", (0, yd), (pitch, yd), arrowprops=dict(arrowstyle="<->", color=INK2, lw=1))
    ax.text(pitch / 2, yd + 0.03, f"pitch {pitch:.2f} m", ha="center", va="bottom", fontsize=11, color=INK2)
    ax.set_aspect("equal")
    ax.set_xlim(x0 - 0.5, x1 + 0.45)
    lo = -aisle - D - 0.15 if rows == 2 else -0.6
    ax.set_ylim(lo, aisle + D + 0.22)
    ax.axis("off")
    ax.set_title(title, pad=6)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--aisle", type=float, default=0.40)
    ap.add_argument("--caddy", default="plusx")
    ap.add_argument("--out", default="runs/layout/layout_plan.png")
    a = ap.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.6), gridspec_kw={"width_ratios": [1, 1]})
    draw_cell(axes[0], 1, C.AISLE_HALF, None, title=f"Today: one row, rail {C.AISLE_HALF:.2f} m from the doors")
    draw_cell(axes[1], 2, a.aisle, a.caddy, title=f"Two facing rows, rail {a.aisle:.2f} m from each row")
    fig.suptitle("Cell layouts, top view to scale", x=0.01, ha="left", fontsize=20, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.08, wspace=0.05)
    footnote(fig, "Dark: printers (0.39 m) on benches; orange: door leaf / D-handles (0.11 m stand-off); light blue: "
                  "carriage deck; green: caddy bins (0.11 m); blue: arm base. Only the working printer's door is open.")
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
