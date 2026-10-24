# Demo plate

Three simple parts with known sizes (see real/DEMO_DAY.md, "Demo plate"). Import all three into Creality Print or
OrcaSlicer and place them at these bed positions (mm, part centre):

| file | size (mm) | centre (x, y) | rotation |
|---|---|---|---|
| cube_20x20x15.stl | 20 x 20 x 15 | (110, 100) | 0 |
| cylinder_d20x15.stl | d 20 x 15 | (60, 60) | 0 |
| bar_40x18x10.stl | 40 x 18 x 10 | (160, 60) | 30 deg |

Turn on **Label objects** and **Exclude objects** (G-code output), no brim, so `real.gcode_objects` can read every
part's centre and footprint from the exported G-code.
