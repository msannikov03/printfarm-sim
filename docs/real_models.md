# Real 3D models: slicer, plan, sim, unload

Every part in the catalogue (`farm/catalog.py`) is a primitive: a box, a cylinder or an L. This page tests the
pipeline on real 3D models instead. STL and 3MF files go through a real slicer and the repository's G-code parser
into the MuJoCo cell, and the robot unloads them with the same perception, grasp and unload code as the primitive
plates. One question: can the robot unload them, and if not, why not?

## What runs

```
models (STL / 3MF) -> BambuStudio CLI: arrange on one P1S plate, slice  (real/bambu.py)
                   -> plate.gcode.3mf: the arranged models + the labelled G-code
                   -> G-code parser: footprint, yaw, height, pick order per object  (real/gcode_objects.py)
                   -> mesh parts in the two-row cell, printer 5, where the slicer put them  (sim/real_parts.py)
                   -> door, chamber camera, grasp search, pick, place, rake fallback  (plan/unload.py)
```

1. **Slice.** `python -m real.bambu OUT model.stl ...` runs BambuStudio's command line (`--arrange 1 --slice 0
   --export-3mf`) with its bundled Bambu Lab P1S 0.4 mm / 0.20 mm Standard / Bambu PLA Basic profiles. Farm settings
   on top: no brim (a brim widens the printed footprint past the model) and object labels on. The bundled profiles
   are layered (`inherits` a parent, `include` start / end G-code templates), and the command line does not resolve
   the layers: raw profiles slice silently with a 200 mm bed and no start G-code. `real.bambu.resolve_profile` flattens
   them first.
2. **Plan.** The parser reads BambuStudio's labels (`; start printing object, unique label id: N` ... `; stop printing
   object ...`, method `bambu_label`) and maps the ids to object names from the archive's `Metadata/slice_info.config`.
   Each object's footprint rectangle, yaw and height come from its own extrusion moves, as for the other slicers.
   There is no purge object: a Bambu printer purges into its chute, and the machine's own prime lines are in the start
   G-code, before the first layer.
3. **Sim.** `sim/real_parts.py` matches every G-code object to its mesh in the same 3MF (label id, else name, else
   position) and registers one SKU per part. The planner plans with the G-code's view: a box of the parsed footprint
   and height. Two things come from the mesh that ships in the same 3MF:
   - the **top band**: where a depth camera sees the part's highest surface (a rim, an upright, a boss), as an extent,
     an offset from the footprint centre and an area;
   - the **grasp cross-section**: where the jaws close at the grasp heights (60 % of the height and 4 mm below the
     top), from the mesh's sections at those heights. These are exactly the slicer's layers there. From the footprint
     centre, the outline in a 12 mm band along each axis gives the span the jaws must open to and its midpoint, the
     grasp centre: an enclosure gives its outer walls, an L bracket its upright;
   - the **layer grasps**: grasps searched over the part's own layers (jaw axis, approach tilt, finger position, tip
     height), tried only when the grasp above finds none. See [Grasp search over the part's own
     layers](#grasp-search-over-the-parts-own-layers).

   The body in physics is the mesh: convex pieces for contacts and the true mesh for the cameras. The pieces come
   from a grid split of the surface (up to 3 x 3 cells, about one per 25 mm), with one convex hull per cell. The mesh
   is a visual-only child body: the chamber and wrist cameras see it, and contacts use the pieces. Each part is
   spawned at the mesh's pose plus N(0, 1.5 mm) per axis and N(0, 2 deg) in yaw, the same spawn noise as
   `scripts/eval_unload.py`, so the slicer prior is not exact and the camera has something to correct.
4. **Unload.** `scripts/eval_real_models.py` runs the existing `plan.unload.unload_parts` on printer 5 (row A of the
   default two-row cell), with ideal sensors. The grasp planner tries its own grasp first and the layer grasps when it
   finds none (`--no-layer-grasp`: its own grasp only, as before the search). Routing follows the catalogue rules: a
   part longer than 80 mm goes to the large-part tray, a smaller one to a tote, and a part wider than the 70 mm jaw
   stroke on both sides is never attempted (route `reject`). Anything left goes to the rake fallback.

Two plate layouts, from the same oriented models:

- **Slicer's arrange**: BambuStudio's own `--arrange` (and `--orient` for the downloaded STLs). This is what a
  person gets by pressing "arrange" and "slice".
- **Robot-friendly**: `sim.real_parts.pack_meshes` packs the same models with the cell's packer (`jobs/plates.py`
  MaxRects): 25 mm between parts for the jaws (`PART_SPACING_MM`), the front 30 mm of the bed left free. The plate is
  then sliced as placed (`--arrange 0`). This is the farm's design-for-automation rule applied to real models.

## Grasp search over the part's own layers

`plan.unload.plan_grasp` closes the jaws across the footprint's narrow side, or across the grasp cross-section of a
real part, at two fixed heights, and searches the approach tilt, the carriage position and the IK seed. On real
parts it found a grasp for 1 of 22 private parts. Probing the planning model (IK over the bed, door open) shows why
fixed heights and the footprint's axes miss so much:

- **The arm reaches the bed only tilted.** Through the door, no approach 0 to 35 deg from vertical has an IK solution
  anywhere on the bed. 50 and 65 deg, leaning into the printer, work almost everywhere, and 80 deg in places.
- **Jaws pointing into the printer are out of reach.** A jaw axis more than about 60 deg from the door plane reaches
  the bed from one or two carriage positions at most, and there the arm meets the door frame. A part whose narrow
  side points into the printer has to be gripped across its long side.
- **A tilted finger is wide in every layer.** At 65 deg the finger, 14 mm across, cuts each horizontal layer over
  14 / cos 65 deg = 33 mm. The jaws must clear everything in that strip, so a jaw axis turned away from a part's
  edge sees the edge run along the jaw over the whole strip.
- **The fingertips are about 10 mm below the pinch point** at 50 to 65 deg. A pinch height taken from the part's
  height puts the tips into the bed, or far above a low part.
- **Small features at the fixed heights.** The wide outline is often lower than both heights.

The search uses only what the robot has: the G-code's footprint, yaw and height, and the mesh that comes with the job
in the 3MF, in the plan's frame. It never uses the simulator's truth.

1. **Layers.** The mesh's sections every 0.6 mm (three 0.2 mm slicer layers). These are the outlines the slicer
   prints (`real.mesh_io.grasp_candidates`, run when the plate is planned; the SKU keeps the result as `grasp_alts`).
2. **Frames.** Jaw axes every 15 deg in the part's frame, those within 60 deg of the door plane at the part's G-code
   yaw. The approach leans into the printer, tilted 50, 65 or 80 deg from vertical about the jaw axis. A large part
   gets 65 deg at most, as for its tray placement (`plan/mesh_grasp.py`: `candidate_dirs`, `TILTS`, `plan_tilts`).
3. **Finger column.** The finger is a strip in the plane normal to the jaw axis. It ends 10 mm past the pinch point,
   is 14 mm wide and is checked 60 mm back along the finger (YAM, measured in the planning model). For every finger
   position across the jaw axis (1 mm steps) and every height of its lowest point (from 1.8 mm, 0.3 mm above
   `TIP_CLEAR`): the material in that strip, grown by 3 mm of clearance, layer by layer. Its extent along the jaw
   axis is the width the jaws must open past and close on.
4. **Kept** if that width fits the 70 mm stroke, the pads (the strip itself, its first 40 mm) touch the widest
   material on both sides over at least 2 mm of height, and the fingertips reach min(5 mm, 60 % of the column's top)
   below its top (`MIN_COVER`, with the column's own top).
5. **Score.** Cover (up to 8 mm) x stroke margin (the 74 mm opening minus the width, up to 16 mm) / (1 + the distance
   of the contact from the mesh centroid / 25 mm). The part hangs from the jaws. Candidates are ordered by the score
   times a reach prior (tilt 65 before 50 before 80 deg; jaw axes near the door plane first), spread over frames and
   over what they grip; 40 per part.
6. **Planner check** (`plan/mesh_grasp.py`). This runs only when `plan_grasp`'s own grasp fails, and only for SKUs
   with layer grasps (real meshes). Candidates go in order:
   - IK at five carriage positions with 2 + 3 seeds;
   - the pinch slides along the approach (the finger line the candidate was checked on) until the lowest finger point
     is at the candidate's tip height, within 0.8 mm;
   - with the jaws open, grasp and pre-grasp must clear the neighbours, the printer and the door by `GRASP_CLEAR`
     (4 mm, plus pose uncertainty);
   - the part itself is an obstacle too, with its convex collision pieces at the estimated pose. The jaws clear it by
     1.5 mm plus its pose pad: the wrist camera centres them on it at the pre-grasp. `plan_grasp` ignores a small
     part's own body and keeps `GRASP_CLEAR` to a large one;
   - the fingertips stay `TIP_CLEAR` above the bed.

   The first feasible candidate wins, at its best carriage position. A frame with no IK solution at all is skipped
   for the candidates near it.

The catalogue's primitive SKUs carry no layer grasps, so their planning does not change (regression check below).

## Models

Open models, downloaded at test time by `python -m real.fetch_models` (sources, licenses and sha1 in
[`real/open_models.json`](../real/open_models.json); nothing is redistributed here):

| Model | What it is | License | Sliced footprint (mm) |
|---|---|---|---|
| `benchy` | 3DBenchy, the calibration benchmark | CC0 | 60 x 31 x 48 |
| `calib_10cc` | 10 cm³ calibration block | public domain | 50 x 20 x 10 |
| `lock_bracket` | lock bracket (Apollo 11 hatch, Smithsonian scan) | CC0 | 73 x 12 x 27 |
| `door_bracket` | door-side bracket (same source) | CC0 | 100 x 66 x 20 |
| `clamp` | backplate clamp of a small PC case | CC BY 4.0 | 77 x 14 x 10 |
| `cargo_box` | 3DBenchy's cargo box, a small open box | CC0 | 12 x 11 x 9 |
| `switch_case` | switch enclosure shell | CC BY 4.0 | 175 x 16 x 107 (the slicer stood it on edge) |

Generated models (ours, `real.mesh_io.cup_mesh` and `l_bracket_mesh`): a 60 x 40 x 25 mm open enclosure with 2 mm
walls and a 50 x 20 x 30 mm L bracket. The unit tests use them, and so does the checked-in G-code snippet.

Also tested, locally only: 7 private real-world models, reported below in aggregate only.

## Results

### The plan vs the mesh

The parser's footprint rectangle compared with the true footprint of each model (min-area rectangle of the mesh's
outline, in the printer's frame), for the 7 open models on the slicer's plate:

| Model | Centre error | Long side | Short side | Height | Gap to the nearest part |
|---|---|---|---|---|---|
| `benchy` | 0.00 | -0.42 | -0.41 | +0.00 | 2.4 |
| `calib_10cc` | 0.00 | -0.42 | -0.42 | +0.00 | 2.4 |
| `lock_bracket` | 0.00 | -0.42 | -0.42 | +0.05 | 2.4 |
| `door_bracket` | 0.00 | -0.43 | -0.42 | +0.02 | 2.4 |
| `clamp` | 0.00 | -0.42 | -0.42 | +0.00 | 2.4 |
| `cargo_box` | 0.01 | -0.42 | -0.42 | -0.40 | 2.4 |
| `switch_case` | 0.00 | -0.42 | -0.42 | +0.00 | 2.4 |

All values in mm; size and height errors are plan minus mesh.

- **Centre**: within 0.01 mm for every part. This holds only in the printer's frame. The 3MF's build transforms
  are in the slicer's plate frame, and a P1S has `extruder_offset = 0x2`: the G-code is 2 mm lower in y than the
  model in the 3MF (G-code y = plate y - 2 mm). The plan must therefore come from the G-code, not from the 3MF
  positions. `sim.real_parts.gcode_frame_offset` reads the offset from the G-code.
- **Size**: every footprint is 0.42 mm short on both sides. That is one line width: the parser fits the extrusion
  centre lines, and the outer wall runs half a line width inside the model's outline. The parser does not add the
  line width back; the grasp opening margins absorb it.
- **Height**: within 0.05 mm, except the cargo box (-0.40 mm, two 0.2 mm layers; not traced further).
- **Spacing**: BambuStudio's arrange leaves 2.4 mm between neighbouring parts, on every plate tried here. The cell
  needs about 25 mm for a jaw beside a part.
- **Pick order**: the parser's front-to-back, tallest-first order. No purge object on a Bambu plate.

### Unloading: open models

Printer 5, ideal sensors, one plate per layout. Columns: the plan error (centre / long side / short side / height,
mm); the slicer prior's centre error after the spawn noise; the chamber camera's centre error after it refines the
prior, with the top-band model and without it (every part treated as a prism with a flat top over its whole
footprint); whether the grasp planner finds a grasp before the first pick; and how the part ends.

**The slicer's arrange** (7 parts on one plate, 2.4 mm apart). Door open and closed again.

| Part | Route | Prior error | Camera, top band | Camera, flat top | Grasp | Outcome | Ends |
|---|---|---|---|---|---|---|---|
| `door_bracket` | tray | 1.7 | 34.5 | 1.5 | no | no_grasp | on the bed |
| `benchy` | tote | 1.4 | 0.6 | 1.4 | no | no_grasp | on the bed |
| `switch_case` | tray | 0.3 | 0.3 | 0.3 | no | no_grasp | on the bed |
| `clamp` | tote | 0.5 | 0.9 | 0.6 | no | no_grasp | on the bed |
| `lock_bracket` | tote | 1.3 | 2.2 | 1.0 | no | no_grasp | on the bed |
| `cargo_box` | tote | 3.3 | 0.6 | 8.2 | no | no_grasp | on the bed |
| `calib_10cc` | tote | 1.4 | 0.5 | 3.2 | no | no_grasp | on the bed |

No part has room for a jaw, so the planner finds no grasp for any of them and makes no pick attempt. The rake
fallback clears none of them either, so the plate needs a person. This is the main result for a slicer's own
arrangement: **unpickable as packed**, whatever the parts are.

**Robot-friendly plate** (the same 7 oriented models, 25 mm apart). Door open and closed again.

| Part | Route | Prior error | Camera, top band | Camera, flat top | Grasp | Outcome | Ends |
|---|---|---|---|---|---|---|---|
| `clamp` | tote | 1.4 | 0.2 | 1.4 | yes | placed | its tote |
| `benchy` | tote | 0.3 | 0.4 | 0.4 | yes | dropped_in_transit (robot's verdict) | its tote |
| `calib_10cc` | tote | 0.5 | 1.2 | 1.2 | no | no_grasp | removed (raked off) |
| `lock_bracket` | tote | 1.1 | 1.1 | 1.5 | no | missing, then no_grasp | on the bed |
| `cargo_box` | tote | 1.7 | 2.7 | 38.3 | no | no_grasp | removed (raked off) |
| `switch_case` | tray | 3.3 | 3.3 | 3.3 | no | missing, then no_grasp | removed (raked off) |
| `door_bracket` | tray | 1.4 | 26.3 | 1.4 | no | no_grasp | removed (raked off) |

Errors in mm (centre). "Removed" means the rake pushed the part off the bed (into the reject tray or past it); the
sim then takes it away, as an operator would.

- **Placed: 2 of 7.** The clamp is placed cleanly. 3DBenchy also ends in its tote, but the robot reports
  `dropped_in_transit`: it lost the part on the way out, and the part happened to fall into the tote.
- **The slicer's orientation matters too.** BambuStudio's `--orient` stood the switch case on its
  175 mm edge: a 107 mm wall across the middle of the bed. The arm reaches in from the door, and the two parts that
  lie behind that wall (the calibration block, the lock bracket) get no grasp; the wall is the likely reason, not
  checked further. On a farm, a part like that should print lying flat.
- **No grasp, cause not traced**: the cargo box (12 x 11 x 9 mm, the smallest part) and the door bracket (100 x 66 mm,
  in front of the wall).
- **Camera.** The top-band model helps where the top is a small or off-centre feature (the cargo box: 2.7 mm instead
  of 38 mm; the clamp: 0.2 instead of 1.4). It hurts on the door bracket (26 mm), where the camera's top band and the
  mesh's top band disagree. There the fit jumps to the wrong surface and the gate takes it over the prior. A flat-top
  model falls back to the prior (1.4 mm) on that part.

The two generated models (an open enclosure and an L bracket, 35 mm apart, the end-to-end test in
`tests/test_real_models.py`) are both placed in their totes, with camera errors under 1 mm.

**With the grasp search over the part's own layers** (same plates, seeds and settings; runs
`runs/real_models/open_*_grasp2`). Before: the tables above. "Layers" is a grasp from the search; its tilt and width
follow.

| Part (robot-friendly plate) | Grasp before | Grasp after | Outcome before | Outcome after | Ends before / after |
|---|---|---|---|---|---|
| `cargo_box` | no | layers, 65 deg, 14 mm | no_grasp | slipped, twice | raked off / raked off |
| `clamp` | yes (50 deg, 2 mm) | the same | placed | placed | its tote / its tote |
| `benchy` | yes (50 deg, 21 mm) | the same | dropped_in_transit | dropped_in_transit | its tote / its tote |
| `calib_10cc` | no | layers, 65 deg, 50 mm | no_grasp | insert_blocked | raked off / reject bin |
| `lock_bracket` | no | no (camera: missing) | missing | missing | on the bed / on the bed |
| `switch_case` | no | no (camera: missing) | missing | missing | raked off / on the bed |
| `door_bracket` | no | no | no_grasp | no_grasp | raked off / reject bin |

- **Grasps 2 -> 4 of 7, placed 2 -> 2.** The two new grasps fail in physics. On the cargo box (a 12 x 11 x 9 mm open
  box, one convex piece) the jaws close to 6 mm on a planned 14 mm, so the pads meet the box somewhere the finger
  model did not expect. It slips, and on the retry it has been knocked over. The calibration block's grasp, across
  its 50 mm length, is found, but the straight insert from the pre-grasp is blocked: next to it stands the switch
  case on edge, which the camera has lost ("missing"), so it is a blob in the planning scene. In an earlier try with
  a different candidate order, the cargo box was placed. Single episodes, so a grasp that works once is not yet a
  rate.
- **The door bracket** (100 x 66 mm, tray) has candidates, but every one collides with a neighbour or has no IK
  solution: its narrow side points into the printer, and grasps across the corners need the arm right over the
  standing switch case.
- **The slicer's arrange: still 0 of 7.** With 2.4 mm between parts the search finds nothing either. The rejections
  are IK failures and collisions with the neighbours, for every candidate.
- **Planning time.** A part without a grasp is searched again before every pick (up to 40 candidates, about 10 s
  each time). The robot-friendly plate took 791 s of wall time instead of 564 s; the private plates took about as
  long as before.

### Unloading: private models (aggregate)

7 private real-world models: 7 project files, 82 objects in all.

**Plan, on the slicer's own plates** (BambuStudio arrange, plate 1 of each file; 50 parts):

- centre error mean 0.09 mm, max 2.4 mm; height error mean 0.07 mm, max 0.10 mm;
- footprint long side -0.41 mm and short side -0.58 mm on average (one line width, as above);
- the two outliers (centre 1.9 and 2.4 mm, short side -2.8 and -3.4 mm) have features thinner than a line width:
  those do not print, so the printed footprint is smaller than the model's;
- every part with a neighbour on its plate (49 of 50) is 2.4 mm from the nearest one: no grasps.

**Unload, robot-friendly plates** (25 mm apart, one plate per file; the largest objects that fit on one plate first:
22 parts on 7 plates):

| | Result |
|---|---|
| Routes | 11 tote, 8 large-part tray, 3 refused: wider than the 70 mm stroke on both sides |
| Plan centre error, mean / max | 0.21 / 2.4 mm (the same two outliers) |
| Plan height error, max | 0.10 mm |
| Slicer prior's centre error after the spawn noise, mean | 2.1 mm |
| Camera centre error with the top-band model, mean / median / max | 2.2 / 1.9 / 6.0 mm (11 of 22 within 2 mm) |
| Same with a flat-top model, mean / median | 4.2 / 2.2 mm |
| Grasp found before the first pick | 1 of 22 |
| Placed in its tote or tray slot | 1 of 22 (1 of the 19 not refused) |
| Raked off the bed (reject tray, or off the bed) | 4 |
| Still on the bed after the rake rounds | 17 |
| Doors opened and closed | 7 of 7 |
| Plates the robot reported clear while parts were still on them | 3 of 7 before the fix (8 parts, 7 of them 2 mm or thinner); 0 of 7 after it |

Why so few grasps (the first run, before the grasp search):

- **Flat parts.** 12 of the 22 parts are 6 mm tall or less, 9 of them under 4 mm (down to 0.4 mm). The fingertips
  must stay 1.5 mm above the bed (`TIP_CLEAR`) and still reach down 60 % of a thin part's height (`MIN_COVER`), which
  is impossible below about 3.75 mm. Such parts need another tool: a scraper, a suction cup, or the bed-flex release
  on the roadmap.
- **Large parts near the stroke.** Most large parts are 58 to 64 mm across their narrow side, against a 70 mm
  stroke (the jaws open to 74 mm). That is the likely reason they get no grasp; it was not traced further.
- **Small features at the grasp heights.** For some parts the cross-section at the fixed grasp heights (60 % of the
  height, 4 mm under the top) is only a small feature, about 3 x 12 mm, while the wide outline is lower than both
  heights. A search over the part's own layers for the height with the widest section would find it.
- **Tiny parts**: 3 x 3 x 6 mm.

**With the grasp search over the part's own layers** (same 7 sliced plates, seeds and settings; run
`runs/real_models/private_packed_grasp2`):

| | Before | After |
|---|---|---|
| Grasp found before the first pick | 1 of 22 | 3 of 22 (2 from the layer search) |
| Placed in its tote or tray slot (truth) | 1 of 22 (1 of the 19 not refused) | 2 of 22 (2 of 19) |
| Flat parts, under 3.75 mm (9) | no grasp, none placed | the same: no candidates at all |
| Refused, wider than the stroke on both sides (3, one of them also flat) | not attempted | the same |
| Large parts, 58 to 64 mm across the narrow side (7, tray) | no grasp | no grasp |
| The rest, small parts 5.8 to 13 mm tall (4) | 1 grasp, 1 placed | 3 grasps, 2 placed |
| Pick attempts | 2 (1 placed, 1 found no grasp at pick time) | 3 (1 placed, 1 slipped, 1 placed but reported `not_in_bin`) |
| Doors opened and closed | 7 of 7 | 7 of 7 |
| Wall time per plate | 36 to 289 s | 36 to 290 s |

The four small parts, after:

- **A 3.4 x 3.4 x 5.8 mm part:** a layer grasp, 65 deg, 4 mm wide, tips 1.8 mm above the bed. It ends in its tote,
  but the robot reports `not_in_bin`: the bin check needs a few 3 mm height-map cells to change, and a part this small
  barely fills one. A missed success, not a false one.
- **Its twin on the next plate** is not found by the chamber camera ("missing"), so it is never planned.
- **A 52 x 11 x 13 mm bar** behind two large parts: a layer grasp, 65 deg, 50 mm across its length, the jaw axis
  15 deg off the bar's axis (across its 11 mm width the jaws would point into the printer). The jaws close on its two
  11 mm end faces and lose it on the lift: it slipped.
- **A 40 x 8 x 10 mm bar:** placed with the fixed grasp, as before.

What still fails, and why:

- **Flat parts (9 of 22)**: below about 3.75 mm. Honest misses, for another tool (above).
- **Large parts lying along the door (7)**: 58 to 64 mm across, with that narrow side pointing into the printer. To
  close across it, the jaws would point into the printer, and the arm cannot reach the bed that way. Within the jaw
  axes it does reach (60 deg of the door plane), the 65 deg finger's strip (33 mm per layer, 47 mm with the
  clearance) makes every chord across the part wider than the stroke, except across corners, where the pads touch
  only edges. One of them is really a 1.5 mm plate with a 2.5 mm rib under a 14 mm hooked tab. The jaws touch the
  hook's lip over less than 2 mm, so the cover rule refuses it. Turned 90 deg on the plate (long side into the
  printer, as the catalogue's large SKUs are packed), they would likely be gripped across their narrow side. That is
  untested: the private plates were used as sliced.
- **Refused (2)**: 113 x 89 and 244 x 100 mm.
- **Weak grasps the physics rejects**: the bar's grasp on its end faces (above). The search scores cover and
  stroke margin, not how the pads meet a face at an angle.

The bed verdicts of this run predate the local-noise fix below (`noise_grid`): one plate was still reported clear
with a 6 mm large part on it. The grasp and placement numbers do not depend on the bed check.

The camera is no better than the prior on these plates (2.2 mm against 2.1 mm). The top-band model helps on average
(2.2 mm against 4.2 mm with a flat top), but the error on these real tops (thin rims, small features) is larger
than the 0.5 mm on the primitives.

**Safety finding, now fixed: the bed-empty check missed flat parts.** In the first run, on 3 of the 7 plates the robot
reported the bed clear with parts still on it: seven thin parts (four of 0.8 mm, three of 2 mm) and one 6 mm part.
The next plate would have printed on top of them. There were two causes, both fixed in `perceive/verify.py`:

- **Below the leftover height.** A leftover counts only when it is at least 3 mm high (`LEFTOVER_MIN_H_MM`), so a
  thinner part was invisible. Now every planned part lower than 5 mm (`THIN_PART_MM`) is also checked at its own
  slicer footprint (`thin_leftovers`). The 75th-percentile height inside the footprint is compared with the bare bed
  in a ring around it, so local bed-plane and calibration bias cancel. The part is still there if the footprint
  stands at least max(0.4 mm, half the part's height) above the ring. Measured on empty beds, the rise is at most
  0.27 mm with realistic sensors (0.56 mm harsh: under the harsh preset this can call a person in vain).
- **A large flat part read as bed noise.** The camera's local noise estimate per 48 mm cell was 1.48 x median |z| of
  the bed-level points. A part lower than 4 mm that fills a whole cell (here a 91 x 62 mm case part, 6 mm walls on a
  2.4 mm floor) set that estimate to its own floor height, 3.6 mm. That lifted the "raised" threshold to 12.5 mm and
  hid the part. The estimate is now the smaller of that and the spread around the cell's own median, which stays at
  the true noise on a flat part.

After both fixes, with the same plates and seeds, the robot reports **0 of 7 plates clear with a part left on them**:
every plate with a leftover is flagged for a person. Re-runs on the published two-row seeds (5 plates per row,
realistic sensors) give identical outcomes for every part, the same bed verdicts and the same doors. A part
thinner than about 0.5 mm, a few layers, cannot be told from the bed by a D435-class camera at this range, so it
stays a limit.

### The grasp search and the primitives

The search is a fallback for real parts only: `plan_grasp` runs it only when its own grasp fails, and only for SKUs
with layer grasps, which only real meshes have. To check that the primitives' results do not move,
`scripts/eval_unload.py --n 5 --realism realistic` ran on printer 4 (row A) and printer 14 (row B), seeds 0 to 4, on
master and on the branch with the search (one process each, `PYTHONHASHSEED=0`, outputs in `runs/grasp2_regress`).
All 10 episode records are identical except for wall-clock time: every part's outcome, truth and end (52 of 52
placed), the bed verdicts, the doors, the simulated time to the millisecond, every carriage move and drop record.

## What real models broke, and the fixes

| Problem on real models | Fix |
|---|---|
| BambuStudio's command line ignores its own profile layers (200 mm bed, no start G-code) | `real.bambu.resolve_profile` flattens `inherits` + `include` |
| Bambu labels were parsed as anonymous `id_N` objects | own label scheme `bambu_label`, names from `slice_info.config` |
| Bambu's skirt is marked `; FEATURE: Skirt`, not `;TYPE:Skirt` | both markers count as the feature type |
| Bambu primes with `G0 X240 E15` (a G0 that extrudes) | a G0 with extrusion counts as an extrusion |
| 3MF positions differ from the G-code by the extruder offset (2 mm on a P1S) | meshes moved into the G-code frame |
| Project 3MFs carry modifier volumes (a settings region 4 mm taller than the part, in one private file) | parts that `model_settings.config` marks as not printed are left out |
| A model sunk into the plate prints only above it | the mesh is cut at the bed |
| BambuStudio writes unescaped quotes into `model_settings.config` (not valid XML) | read with regular expressions |
| A grasp at the footprint centre closes on air when the material is elsewhere (an L bracket's upright) | the grasp centre and the jaw openings come from the part's cross-sections at the grasp heights |
| Perception assumed a flat top over the whole footprint: a rim looked like a badly covered lid, and the camera was down-weighted to the prior | top-band model from the mesh: its extent, offset and area |
| The slicer's arrange packs parts a few mm apart: a neighbour's points fall inside a part's search window | points inside another object's footprint are no longer taken as this part's top |
| MuJoCo reads at most 200,000 STL faces (3DBenchy has 225,706) | visual meshes decimated by vertex clustering |
| Many convex pieces on one bed overflowed MuJoCo's constraint arena | about one piece per 25 mm (at most 3 x 3) and a larger arena when mesh parts are present |
| Fixed grasp heights and the footprint's axes miss real parts: low wide outlines, jaws that would point into the printer, fingertips 10 mm below the pinch at the only tilts the arm reaches | a grasp search over the part's own layers (jaw axis, tilt, finger position, tip height), a fallback for mesh SKUs only (`plan/mesh_grasp.py`) |

None of the sim-side changes affect the primitive SKUs. A primitive has no top band or grasp cross-section, so its
paths are unchanged, and on the packer's plates no two parts come within the 9 mm that the neighbour rule needs. The
existing test suite passes unchanged.

## Limits

- **Collision geometry is approximate.** A grid split with one convex hull per cell is not a convex decomposition:
  each cell fills its own concavities, so a slot, a hole or the inside of a hook is solid for contacts. Grasps on
  such features are optimistic. The cameras see the true mesh. A proper decomposition (CoACD, V-HACD) would add a
  dependency.
- **Ideal sensors only.** The realistic and harsh camera / actuation presets of `eval_unload.py` were not run on
  real models.
- **Rigid PLA, a cooled bed.** No flex, no adhesion left after cooling, no stringing, no brim, no supports. A model
  that needs supports would have to be oriented so that it does not.
- **One slicer.** Only BambuStudio was available. The Orca / Klipper, M486 and Cura label paths are covered by the
  synthetic G-code tests (`real/tests/test_gcode_objects.py`), not by real slices.
- **The top band and the grasp section need the mesh.** The real cell gets the mesh with the job, in the same 3MF as
  the G-code. A plain `.gcode` file has neither. The same information is in the G-code's layers (the top layers,
  the layer at the grasp height); that parser is not written.
- **The grasp search's finger is a model.** A slanted strip, 14 mm wide, 10 mm past the pinch. It
  is measured in the planning model but not fitted to every finger geom, so where the pads touch can differ: on the
  cargo box the jaws closed to 6 mm on a planned 14 mm. The score knows cover and stroke margin, not the angle at which
  a pad meets a face (the bar's end faces) or friction.
- **No finger inside a part.** Grasps with one finger inside an enclosure (a wall between the jaws) are not searched.
  The convex collision pieces fill the inside anyway, so in this sim they would fail.
- **Fixed reach facts.** The tilts (50 / 65 / 80 deg), the jaw-axis limit (60 deg of the door plane) and the reach
  prior come from IK probes of this cell. Another arm or another door would need them measured again.
- **Thin parts.** Nothing under about 4 mm can be grasped by the jaws. The bed check now finds flat leftovers down
  to about 0.8 mm (see the safety finding above), but not much below 0.5 mm.
- **One printer, one seed per plate.** The numbers are single episodes, not batch statistics.

## Reproduce

```bash
python -m real.fetch_models                                        # open models -> real/assets/models/
PYTHONPATH=. python scripts/eval_real_models.py --models real/assets/models/*.stl --orient \
    --name open_arranged                                            # the slicer's arrange (runs/real_models/...)
PYTHONPATH=. python scripts/eval_real_models.py --models runs/real_models/open_arranged/plate/plate.gcode.3mf \
    --pack --max-plates 3 --name open_packed                        # the same oriented models, robot-friendly plates
PYTHONPATH=. python scripts/eval_real_models.py --models my/*.3mf --each --pack --max-plates 1 --anonymous \
    --name mine                                                     # your own files, robot-friendly plates
PYTHONPATH=. python scripts/eval_real_models.py --sliced runs/real_models/open_packed/plate/plate.gcode.3mf \
    --name open_packed_grasp2                                       # an already sliced plate (the *_grasp2 runs);
                                                                    # --no-layer-grasp: the fixed grasp only
python -m pytest tests/test_real_models.py real/tests/test_bambu_gcode.py real/tests/test_mesh_io.py
```

Each run writes `report.json` (every number above, per part) and `table.md`. Tests that need BambuStudio or the
fetched models skip without them. The one full-unload test (`test_unload_real_mesh_parts_end_to_end`, generated
models and a synthetic G-code, about 45 s) is marked `slow`. The G-code snippet in `real/tests/data/` is
BambuStudio's own output for the two generated models, trimmed to the header, the start G-code, two layers and each
object's last layer.
