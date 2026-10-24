# Calibration line and purge tab

Every print starts with a prime line (also called purge or calibration line): before the first layer, the hotend
extrudes a line to bring the flow up to pressure. On a manual farm someone scrapes it off the bed. A robot cell has
to remove it as well. If it does not, the next plate's first layer lands on top of it.

The cell's answer is to end the line in a small printed grab tab. The robot picks the tab first, and the line comes
off with it.

## Layout: slicer and start G-code

| item | value | where |
|---|---|---|
| front strip, kept free of parts | 26 mm deep, full bed width | `PURGE_STRIP_MM`, packer in `jobs/plates.py` |
| purge tab object | 10 x 20 mm footprint, 20 mm tall, at the front centre of the strip (x = bed width / 2, y = 13 mm) | `PURGE_TAB_MM` |
| keep-out around the tab | parts at least 20 mm away (room for the open jaws) | `PURGE_TAB_CLEAR_MM` |
| prime line | 55 mm along the strip, to the right of the tab (+x), ending at the tab's centre | `PURGE_LINE_MM` |

1. **The slicer adds the tab to every plate as an ordinary, labelled object.** In Orca or Creality Print, add a
   10 x 20 x 20 mm box named `purge_tab` at the front centre, and turn on "Label objects" and "Exclude objects". The
   G-code then carries `EXCLUDE_OBJECT_DEFINE NAME=purge_tab CENTER=128,13 POLYGON=[...]` (with Marlin flavour, an
   `M486` label). The tab is 10 mm across the jaws' closing axis and gives 20 mm of grip height.
2. **The printer's start G-code primes along the front strip and ends inside the tab's footprint.** The tab's first
   layer is printed over the end of the line, so the line fuses to the tab.
   - Klipper: a macro, called from `PRINT_START` once the nozzle is hot.
     ```ini
     [gcode_macro PRIME_TO_TAB]
     description: Prime along the front strip, ending under the purge tab
     gcode:
       {% set cx = params.CX|default(128)|float %}    # tab centre = the slicer's purge_tab object
       {% set cy = params.CY|default(13)|float %}
       {% set len = params.LEN|default(55)|float %}
       G90
       G1 Z2 F1200
       G1 X{cx + 5 + len} Y{cy} F12000               # start right of the tab
       G1 Z0.3 F600
       M83
       G1 X{cx} Y{cy} E{(len + 5) * 0.1} F900        # 0.8 x 0.3 mm line into the tab's footprint
       G1 E-0.8 F2400
       G1 Z2 F1200
     ```
   - Marlin: the slicer's "Machine start G-code", with the same moves. For a 256 mm bed:
     ```gcode
     G92 E0
     G1 Z2 F1200
     G1 X188 Y13 F6000        ; the tab is at X128 Y13: the line starts 60 mm to its right
     G1 Z0.3 F600
     G1 X128 Y13 E6.0 F900    ; prime line, ending at the tab's centre
     G92 E0
     G1 Z2 F1200
     ```
   - Creality K1 and Ender machines draw their stock prime line along the left bed edge (`;Draw the first line`).
     Replace it with the moves above.
3. **The parts print as usual**, outside the strip.

Bambu-class printers (X1, P1, A1) purge into a chute at the back instead, so no line is left on the bed. On those
the tab is optional. Check the stock start G-code for any short line it still draws on the plate.

## What the robot does

- Planning starts from the slicer's plate. The tab's pose is taken from the slicer, because the start G-code places
  it exactly. The chamber camera does not move it.
- The tab is picked before any part (`plan/unload.py` `pick_priority`):
  - the jaws close across its 10 mm side (or its 20 mm side), high on the tab, 12 to 16 mm above the bed, so the pad clears the line;
  - lifting peels the line off the cooled PEI sheet along with the tab.
- **The drop centres tab and line over the waste tote together** (`held_offset`):
  - they are 65 mm end to end, and the tote is 104 mm inside, so they fit in any orientation;
  - the wrist camera then checks the tote, the same way as for any part;
  - in that look, the held part is masked out as a sphere around the tab plus a thin capsule along the line
    (`held_mask`). A sphere big enough to cover the whole line also hid the tote's walls, so the rim height came
    out wrong and the check reported the tab "on the rim".
- At the station, the waste tote's trap door dumps into the scrap bin, and the operator empties that bin.
- **Real cell** (`real/gcode_objects.py`):
  - the purge object is the object named `purge_tab` (or anything matching `purge|prime.?tab`);
  - in an unlabelled file it is the object whose footprint holds the end of the prime line;
  - it gets `role: "purge"`, is reported as `purge_object`, and comes first in `pick_order`;
  - `python -m real.synth_gcode out.gcode --purge-tab` writes this layout for the mock printers and tests.

## In the simulation

- Tab and line are one rigid body (`farm/scene.py`), in vivid cyan. That colour is distinct from the orange parts,
  the red reject tote, the grey waste tote and the bronze PEI bed.
- The line is 55 x 4 x 0.6 mm, as thick as a real prime line (0.2 to 0.6 mm).
  - That is well below perception's 1.5 mm "raised" threshold, so the chamber camera sees the line as bed. At
    1.2 mm, depth noise on row B lifted part of the line into a "debris" blob beside the tab. That blocked the
    tab's grasp, and the tab was then raked away.
  - Simplification: a real line is flexible and dangles from the lifted tab. In the sim it stays rigid and sticks
    out sideways.
- The farm video shows the first few purge picks as close-ups (`farm/director.py`, the "purge" shot).
