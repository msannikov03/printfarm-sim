# PrintPick: design decisions

This page explains the cell layout and the reasoning behind it. `farm/catalog.py` holds the authoritative values;
the numbers below are copied from it. Module structure and data contracts are in
[docs/architecture.md](docs/architecture.md).

## Cell

- **Twenty P1S-class enclosed printers** (256 x 256 mm bed) in **two facing rows of ten** on 0.72 m benches, 0.46 m
  apart (`PRINTFARM_LAYOUT=two_rows`, the default; `one_row` builds the original single row of ten).
- **One I2RT YAM arm** (6 DoF + parallel gripper, 70 mm usable stroke) is bolted to a carriage on a linear rail down
  the middle of the aisle. The rail axis runs 0.40 m from each row's door plane (0.28 m in the one-row cell). The rail
  is the arm's 7th axis in IK, planning and door motion.
- **Low deck.** The arm base sits at 0.56 m, below the doors' bottom edge (0.75 m). A folded pose (`LOW_TUCK`)
  keeps the whole arm under 0.70 m, so the carriage can drive under doors that are open. Without this, an open door
  blocks the rail, and the door can only be swung from one side of the hinge. A layout sweep found no workable
  geometry at the old base height.
- **Caddy beside the arm** (two rows; behind it in the one-row cell): eight open bins on the deck on the +x side,
  where the base joint (-150 to +175 deg) reaches them from both rows. Six hold one open order each; the other two
  are `reject` and `waste` (purge tabs). Every drop pose is reachable, and a drop may lean up to 35°.
- **Reject tray per printer**: a shallow tray in the gap under the front edge of the bed. Parts that cannot be
  picked are raked into it and reprinted; an operator empties the trays once per shift.
- **Station** at the +x end of the rail: complete order bins are handed off there and packed by order volume
  (`envelope` / `box_S` / `box_M`) into carrier lanes. The packaging line is kinematic.
- **Why 0.40 m between rail and doors**: at 0.28-0.34 m the forearm hits the opposite row's D-handles during deep
  picks, at 0.37 m the doors open but do not close, at 0.43 m the back of the bed is out of reach. Printers per arm,
  station position and economics are compared in [docs/layout.md](docs/layout.md).

## Door

- The glass door is hinged on the left (seen from the aisle) and opens to 100°, with a magnet latch at 0°.
- The only addition to the printer is a **D-handle**: a 16 cm bar on two stand-offs, 0.30 m from the hinge (where a
  person would pull) and 0.11 m off the glass. That is enough room for a jaw on either side of the bar.
- **Opening is one continuous whole-body swing.** The planner samples the hinge arc and needs a collision-free IK
  solution for every sample, with the arm and the rail moving together. The carriage passes under the opening door.
  If no single swing works, the robot regrasps the same handle at 50°, then 35° or 65°.
- Closing is the same motion in reverse. The latch re-engages at 0°.
- Simulation simplification: while the gripper holds the handle, the grip is a rigid lock. The hinge, the end stop
  and the latch are full dynamics. See [docs/door.md](docs/door.md).

## Per-printer sequence

1. The scheduler (`order_aware`) chooses a finished printer and waits until its bed has cooled below the release
   temperature (35 °C assumed). Adhesion is a weld that releases at that point.
2. Open the door.
3. The chamber camera refines the slicer's part poses and flags missing parts and debris.
4. Pick the purge tab into the waste bin, then each part into its order's bin. Tall parts go first, and the bed is
   re-scanned before every pick:
   - grasp from the slicer geometry (refined by the camera): jaws across the narrow side of the footprint, approach
     tilt 0 to 90°, carriage position searched, fingertips kept above the bed;
   - staging pose, checked insert, closed-loop fine alignment, close at about 10 N, lift and back out along the
     approach, exit through the door (at half speed while holding a part), part-centred drop;
   - release check: if the part did not fall, roll the wrist and release again.
5. Anything still on the bed (a failed pick, a tipped part, debris the camera sees) is raked into the reject tray,
   in up to three rounds.
6. Bed-empty check, then close the door; the printer starts its next plate. A part still on the bed after the
   rake counts as a human intervention and blocks that printer.

## Slicer stand-in: design for automation

- `PART_SPACING_MM = 25` between parts, so a jaw pad fits.
- A 26 mm purge strip at the front holds a short prime line ending in a 10 x 20 x 20 mm grab tab. The tab sits at the
  front centre, and parts keep 20 mm away from it.
- `plan/bedmap.py` computes where on the bed every probe part can be grasped: 88 % of the bed in the current
  `runs/bedmap.json` (16 mm grid). The packer only places parts inside that area (`reach_mask`).
- `mirror_plate()` packs parts toward the robot's side.

## Timings contract (`runs/timings.json`)

The DES falls back to these defaults when no measured file exists. The measured file (`scripts/make_timings.py`)
uses the same keys; see [docs/architecture.md](docs/architecture.md#runstimingsjson-measured-in-physics-consumed-by-the-des).

```json
{"rail_speed_mps": 0.4, "door_open_s": 14.0, "door_close_s": 12.0, "scan_s": 4.0,
 "purge_tab_s": 12.0, "pick_place_s_mean": 11.0, "pick_place_s_std": 2.0,
 "bed_check_s": 3.0, "station_handoff_s": 6.0, "retry_s": 8.0,
 "p_success_first_try": 0.95, "p_intervention_per_part": 0.01, "p_reprint_per_part": 0.0, "rake_s": 15.0}
```
