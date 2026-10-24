"""Shared constants: farm layout, printer geometry, SKU catalogue.

Pure Python (no MuJoCo) so the fleet/DES side can import it.
Units: metres for the scene, millimetres for bed/slicer coordinates.

World: rail along +X at y=0, printers in one row on +y (doors face -y). (A mirrored row B on -y is supported.)
Printer local frame: origin on the door plane at the printer's centre line, z=0 at the enclosure floor;
x_l to the right as seen from the aisle, y_l into the printer, z up. The door hinge is on the LEFT.
Bed frame (mm): origin at the bed's front-left corner seen from the aisle, x right, y into the printer.
"""
import math
import os
from dataclasses import dataclass

# ---- printer: P1S-class enclosed CoreXY ----
BED_MM = (256.0, 256.0)
PRINTER_W, PRINTER_D, PRINTER_H = 0.39, 0.39, 0.46
DOOR_W, DOOR_H = 0.34, 0.40           # glass leaf; hinge at x_l = -0.175
DOOR_Z0 = 0.03                        # door bottom above the enclosure floor
HINGE_XL = -0.175
DOOR_OPEN_RAD = math.radians(100)     # leaf clear of the opening
DOOR_REGRASP_RAD = math.radians(50)   # if one continuous swing is infeasible: regrasp the same handle here
PIN_S = 0.30                          # handle bar position along the leaf from the hinge: the everyday spot
PIN_OFFSET = 0.11                     # handle bar stand-off from the glass: room for a jaw on either side
HANDLE_LEN = 0.16                     # D-handle bar length (stand-offs at both ends)
PIN_ZL = 0.26                         # handle bar centre (grasp point) above the enclosure floor
BED_FRONT_GAP = 0.05                  # door plane -> bed front edge (m)
BED_ZL = 0.10                         # bed surface at the unload height (end G-code); M2 may change
PURGE_STRIP_MM = 26.0                 # front strip of the bed reserved for the purge tab
PURGE_TAB_MM = (10.0, 20.0, 20.0)     # grab tab at the end of the prime line, sized for the YAM jaws
PURGE_LINE_MM = (55.0, 4.0, 0.6)      # prime (calibration) line: length along the front strip, width, height. It runs
                                      # along the bed's +x from the tab's +x face and is fused to the tab's first layer
                                      # (docs/purge.md). 0.6 mm (a real prime line: 0.2-0.6) stays well under
                                      # perception's RAISED_MM (1.5 mm): the chamber camera sees bed there (at 1.2 mm,
                                      # noise lifted part of a 55 mm line into a "debris" blob beside the tab)
PURGE_TAB_CLEAR_MM = 20.0             # no part within this of the tab: room for the open jaws around it
REJECT_TRAY = (0.15, 0.020, 0.030)    # per-printer reject tray in the front gap under the bed edge: half-width,
                                      # half-depth, wall height (m). Parts the robot cannot pick are raked into
                                      # it and reprinted; the operator empties it once per shift
PART_SPACING_MM = 25.0              # gripper-aware: room for a jaw pad + clearance between parts

# ---- farm layout ----
# PRINTFARM_LAYOUT selects the cell (see docs/layout.md):
#   "two_rows": two facing rows of 10 printers on one rail (symmetric: half the rail per printer,
#               the arm turns 180 deg between rows; aisle 0.40 m so the elbow clears the opposite row's handles);
#   "one_row":  one row of 10 along a wall (the original cell; the sensor-realism batches in docs/sim2real.md use it).
LAYOUT = os.environ.get("PRINTFARM_LAYOUT", "two_rows")
if LAYOUT not in ("one_row", "two_rows"):
    raise ValueError(f"PRINTFARM_LAYOUT must be one_row or two_rows, not {LAYOUT!r}")
TWO_ROWS = LAYOUT == "two_rows"
N_ROWS, N_PER_ROW = (2, 10) if TWO_ROWS else (1, 10)
N_PRINTERS = N_ROWS * N_PER_ROW
PITCH = 0.46                          # printer centre-to-centre along the rail
BENCH_H = 0.72                        # enclosure floor height above the floor
AISLE_HALF = 0.40 if TWO_ROWS else 0.28   # rail axis -> door plane (0.37 m fails to close, 0.43 m loses the bed back)
DECK_Z = 0.56                         # carriage deck top: low enough that the folded arm passes under open doors
BASE_Z = 0.56                         # YAM bolted on the deck; folded, it clears the door bottom edge (0.75 m)
RAIL_X_MIN, RAIL_X_MAX = -0.45, (N_PER_ROW - 1) * PITCH + 0.95
STATION_X = (N_PER_ROW - 1) * PITCH + 0.75   # bin hand-off / packaging station at the +x end

# ---- on-carriage sorting caddy ----
CADDY_ORDER_SLOTS = 6                 # one open order per slot
CADDY_SLOTS = {                       # slot -> (dx, dy) on the deck relative to the arm base (m); behind the arm,
    "o0": (-0.18, -0.27), "o1": (-0.06, -0.27), "o2": (0.06, -0.27), "o3": (0.18, -0.27),   # away from the doors
    "o4": (-0.18, -0.39), "o5": (-0.06, -0.39), "reject": (0.06, -0.39), "waste": (0.18, -0.39),
}
if TWO_ROWS:                          # behind the arm is the other row: all bins beside the arm on the +x side (the
    CADDY_SLOTS = {                   # base joint covers -150..+175 deg, so straight behind is its dead zone)
        "o2": (0.26, 0.0), "o3": (0.26, 0.12), "o0": (0.26, 0.24), "o5": (0.38, 0.0), "o1": (0.38, 0.12),
        "o4": (0.38, 0.24), "reject": (0.50, 0.0), "waste": (0.50, 0.12),
    }
BIN_SIZE = (0.11, 0.11, 0.07)         # x, y, height (open-top); the six order bins are the package totes

# ---- bed to box: packages, totes, large-part tray, shipping boxes ----
# An order ships as one or more packages. A package = the small parts in ONE tote (an order bin with a trap-door
# bottom) + at most LARGE_PER_PACKAGE large parts carried in the large-part tray; the station drops the tote into
# the requested box and the arm places the large parts. Packages never mix.
TOTE_MAX_PARTS = 6                    # small parts per tote (loose pile, no part dropped onto a full tote)
TOTE_PART_MAX_MM = 80.0               # a part whose footprint is longer than this is "large" (does not fit a tote)
LARGE_PER_PACKAGE = 2
LARGE_SLOT_SIZE = (0.19, 0.115, 0.03)   # large-part tray slot: x, y (outer), wall height (m); 115 mm: the station's
                                        # jaws still fit beside a 55 mm housing lying 15 mm off centre
LARGE_SLOTS = {                       # slot -> (dx, dy) on the deck relative to the arm base (two rows only; the
    "L0": (0.32, -0.135), "L1": (0.32, -0.25),   # one-row cell keeps the small-SKU legacy flow). L0 at -0.135: the
} if TWO_ROWS else {}                 # knuckles of a held housing (61 mm off its centre line) clear bin o2's wall
BOX_TYPES = {                         # shipping boxes, inner (x, y, z) mm, smallest first
    "S": (150.0, 110.0, 70.0), "M": (200.0, 150.0, 90.0), "L": (260.0, 190.0, 110.0)}
BOX_WALL_MM = 4.0
BOX_FILL = 0.35                       # loose fill: part envelope (bounding-box) volume / inner volume (ASSUMPTION)
PLA_G_PER_CM3 = 1.24 * 0.45           # density x effective infill (ASSUMPTION)
GRIPPER_MAX_MM = 70.0                 # YAM gripper stroke 2 x 37.5 mm, minus margin


@dataclass(frozen=True)
class SKU:
    name: str
    kind: str      # "box": (x, y, z) | "cyl": (d, h) | "L": (len, width, base_t, upright_h, upright_t), mm
    dims: tuple
    solid_frac: float = 1.0   # printed material / bounding volume (hollow housings, tubes); sim geometry stays solid
    # real parts from a mesh (sim/real_parts.py): kind "box", dims = the footprint rectangle x height the planner
    # works with; the body's collision geometry is convex mesh pieces (col_meshes: vertex lists, mm, part frame) and
    # the chamber cameras see the true mesh (vis_mesh: an STL in the part frame, mm). top_mm = (tx, ty, ox, oy, area):
    # the extent, centre offset and area (mm^2) of the top face, where the depth camera finds the part's highest surface
    vis_mesh: str = ""
    col_meshes: tuple = ()
    top_mm: tuple = ()
    grasp_mm: tuple = ()      # real parts: (cx, cy, wx, wy) where the jaws close (centre, opening along x / y), from
                              # the part's cross-sections at the grasp heights (the slicer's layers there)
    pool: int = 0             # pooled bodies of this SKU in the scene (0: farm.scene.POOL_PER_SKU)

    @property
    def large(self):
        return max(self.footprint) > TOTE_PART_MAX_MM

    @property
    def footprint(self):
        return (self.dims[0], self.dims[0]) if self.kind == "cyl" else (self.dims[0], self.dims[1])

    @property
    def height(self):
        return self.dims[{"cyl": 1, "L": 3, "box": 2}[self.kind]]

    @property
    def volume_cm3(self):
        d = self.dims
        if self.kind == "cyl":
            v = math.pi * (d[0] / 2) ** 2 * d[1]
        elif self.kind == "L":
            v = d[0] * d[1] * d[2] + d[4] * d[1] * (d[3] - d[2])
        else:
            v = d[0] * d[1] * d[2]
        return v * self.solid_frac / 1000.0

    @property
    def mass_g(self):
        return self.volume_cm3 * PLA_G_PER_CM3


CATALOG = {s.name: s for s in [
    SKU("block",    "box", (30, 20, 15)),
    SKU("cube",     "box", (25, 25, 25)),
    SKU("clip",     "box", (28, 12, 14)),
    SKU("lid",      "box", (50, 32, 8)),
    SKU("spacer",   "cyl", (16, 22)),
    SKU("knob",     "cyl", (30, 16)),
    SKU("gear",     "cyl", (42, 10)),
    SKU("standoff", "cyl", (12, 32)),
    SKU("bracket",  "L",   (44, 20, 6, 26, 6)),
    # large SKUs (bed to box): longer than a tote takes, picked across the narrow side (<= 55 mm), carried in the
    # large-part tray and placed into the box by the arm. Each prints alone in < 3 h (sim/farm.py MAX_PLATE_S);
    # grasp maps (plan/bedmap.py --large): all three are graspable anywhere lying into the printer (yaw 90), never
    # lying along the door (jaws pointing into the printer)
    SKU("housing",  "box", (85, 55, 35), solid_frac=0.2),    # open-top enclosure shell, 2.5 mm walls + floor
    SKU("plate",    "box", (120, 50, 15), solid_frac=0.3),   # cover plate: 2.5 mm sheet + 15 mm rim. A bare sheet
    # does not work with the YAM pads, which reach 10 mm past the pinch point: 130 x 50 x 5 was graspable at 3 of 72 bed
    # cells, and at 10 mm only the pad tips touch it (the long part creeps out of the jaws in transit)
    SKU("bar",      "box", (150, 18, 18), solid_frac=0.6),   # square tube
]}


def register_sku(sku: SKU) -> SKU:
    """Add a SKU (e.g. a real part from a mesh) to CATALOG, before the scene is built (farm.scene reads CATALOG).
    The order mix (SMALL_SKUS, LARGE_SKUS, ORDER_SKUS) is fixed at import and does not change."""
    CATALOG[sku.name] = sku
    return sku


SMALL_SKUS = tuple(n for n, s in CATALOG.items() if not s.large)   # insertion order = the legacy catalogue order
LARGE_SKUS = tuple(n for n, s in CATALOG.items() if s.large)
ORDER_SKUS = SMALL_SKUS + LARGE_SKUS if TWO_ROWS else SMALL_SKUS   # what orders draw from (one row: no large-part tray)


def print_time_s(total_volume_cm3: float) -> float:
    """Synthetic print-time model (ASSUMPTION): 10 min heat-up/levelling + 15 cm^3/h."""
    return 600.0 + total_volume_cm3 / 15.0 * 3600.0


def printer_frame(i: int):
    """Printer i (0..9) -> (x_centre, side). side=+1: row A on +y (x_l=+X, y_l=+Y); side=-1: row B (x_l=-X, y_l=-Y)."""
    row, col = divmod(i, N_PER_ROW)
    return col * PITCH, (+1 if row == 0 else -1)


def local_to_world(i: int, xl: float, yl: float, zl: float):
    xc, s = printer_frame(i)
    return (xc + s * xl, s * (AISLE_HALF + yl), BENCH_H + zl)


def bed_mm_to_world(i: int, x_mm: float, y_mm: float, z_mm: float = 0.0):
    xl = x_mm / 1000.0 - BED_MM[0] / 2000.0
    yl = BED_FRONT_GAP + y_mm / 1000.0
    return local_to_world(i, xl, yl, BED_ZL + z_mm / 1000.0)


# ---- station (bed to box) ----
# Two-row cell. The six order bins o0..o5 are package totes with a trap-door bottom: a flap hinged at the tote's +y
# inner edge, held shut against its stop by a position actuator, with an angle encoder; the deck is open under each
# tote (and under "waste"). A cross conveyor along world y at x = X_DROP runs under the deck's +x overhang: box
# magazine (+y) -> big-part place lift beside the deck -> dump lift under the package's tote -> sealer tunnel (-y)
# -> sorter -> 3 carrier lanes. Positions that depend on the deck come from farm.scene.station_layout().
COLUMN_X = (-0.12, -0.02)             # carriage column under the deck (visual), x range relative to the arm base: set
COLUMN_HY = 0.08                      # back behind the base so the deck's +x end overhangs the station conveyor
RAIL_END = 0.02                       # rail (end stop) beyond the column at the end of travel
STATION_CLEAR = 0.02                  # minimum gap between the carriage's swept volume and any station hardware
CONV_HALF_W = 0.11                    # conveyor frame half-width across x (the widest box, L, is 0.198 m)
X_DROP = RAIL_X_MAX + COLUMN_X[1] + RAIL_END + STATION_CLEAR + CONV_HALF_W   # conveyor axis (world x): the column
                                      # and the rail's end stop at full travel stay STATION_CLEAR off the conveyor;
                                      # this also keeps the big-part place within the arm's reach (0.13 m past the base)
FLAP_T = 0.004                        # flap plate thickness (its top is the tote floor, as the old bin floor)
FLAP_OPEN = math.radians(80)          # dump angle: parts slide off past ~50 deg (part friction 1.2)
FLAP_SHAKE = math.radians(55)         # shake: swing between this and FLAP_OPEN
FLAP_CLOSED_RAD = math.radians(3)     # encoder below this = latched shut
Z_CONV = 0.26                         # conveyor roller tops: an L box (0.114 m) passes under the drop chute
PLACE_FLOOR_Z = 0.40                  # place lift: box floor height for the big-part place (arm reach)
BOX_V, BOX_A = 0.25, 0.6              # box speed (m/s) and peak acceleration (m/s^2) on the conveyor and lifts
BOX_POOL = 5                          # boxes per size in the pool (in the magazine, on the line and in the lanes)
ROBOT_TOP_Z = 1.46                    # highest point the carriage + arm can reach over the joint ranges (geom bounding
                                      # spheres; tests/test_station.py checks it): station hardware above this + the
                                      # clearance is out of the robot's reach anywhere on the rail
STATION_CAM_FOVY = 21.0               # station depth camera: straight down over the drop line from a boom above
                                      # ROBOT_TOP_Z (a camera beside the deck saw half of a 70 mm deep tote's floor)
LANE_ORDER = ("courier_A", "courier_B", "post")   # carrier lanes along the sorter, nearest first
