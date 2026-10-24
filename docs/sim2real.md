# Sim to real: what goes wrong on real hardware, and how the system copes

A simulator that reads its own state is not a robot. This page lists every place where the real world differs from a
clean physics simulation, how each difference is modelled here, what the pipeline does about it, and what it costs in
measured success. The rule in the code: **the robot decides only from its sensors; ground truth is used only to
score the result.**

## The gaps

| Gap | Real-world cause | How it is modelled | What the pipeline does |
|---|---|---|---|
| Depth noise | stereo matching error grows with range (sigma_z ~ z^2), speckle | per-pixel axial noise sized from the stereo error formula for a D435-class camera 0.33 m above the bed (1 sigma 0.8 mm realistic, 1.6 mm harsh at 0.40 m, the bed-centre range; the bed is seen at 0.33-0.55 m), partly spatially correlated; lateral jitter; quantization; per-frame temporal noise | median of 4 frames; top-face fit over many points with an adaptive height band and outlier gate |
| Missing and false depth | dark or shiny PLA, the PEI sheet's sheen, edges, occlusion | random dropouts and structured holes, a specular sheen patch on the bed, IR-dark parts (segmentation render), flying pixels at depth edges | neighbour-support filter for flying pixels; a region the camera did not see keeps the slicer prior with quality "degraded" instead of being called missing |
| Camera calibration | bumped mount, thermal drift, imperfect intrinsics | persistent extrinsic error (1 mm / 0.25 deg realistic, 3 mm / 1 deg harsh), focal and principal-point error, depth-scale error | RANSAC bed-plane fit removes height and tilt; the four build-plate edges (depth steps) fix x, y and yaw; measured per-edge read-out bias subtracted |
| Camera failure | no frame, USB reset, dead camera, blackout, something in front of the lens, smeared lens | per-scan failure draws (realistic: 1 % no frame, 0.2 % dead, 0.5 % blackout, 2 % occluder, 2 % smear) | a failed scan is retried twice; if the camera stays down, the slicer prior is used with 4x covariance and the bed-empty check answers "cannot verify" instead of "empty" |
| Pose uncertainty | all of the above | each estimate is a Gaussian posterior: slicer prior N(mu, (2 mm)^2) fused with a camera covariance built from the noise model, the number of supporting points and the calibration residual; a chi-square gate drops the prior when a part was nudged | obstacle padding in the planning scene = 2 x the estimate's sigma; parts the camera did not see are not obstacles the robot "knows" about |
| Robot's own state | homing error, link tolerances, base registration, gearbox backlash, bus latency, encoder noise | `farm/sensors.py` presets (realistic: 0.15 deg joint zeros, 0.3 mm links, 1 mm base registration, 0.1 deg backlash, 20 ms latency); the robot plans with a perturbed copy of its own model, so belief != truth | wrist-camera alignment at the pre-grasp measures where the part really is relative to the jaws (cancels arm and chamber-camera calibration error); settling uses encoder readings, not the true pose |
| Grasp outcome | parts slip, stick to a finger, fall in transit | real contact physics | lift check from the gripper encoder (jaw gap), release check from the wrist camera (points still between the jaws -> wrist shake), drop-in-transit check before release |
| Door | catch force, handle compliance, hinge friction, printer placement error | jaws cage the handle bar; the door moves only through contact forces; magnetic catch F0/(1+g/2 mm)^2; placement error, sag and friction swept | door angle from arm kinematics while holding the bar; a reed switch confirms "closed" (re-push if not); an ajar door with unknown angle is refused |
| Unexpected objects | tipped parts, failed-print debris | fault batches inject one tipped part and one debris lump per bed | unexplained raised blobs become obstacles; after the picks they are cleared: picked as a box into the reject bin, raked into the reject tray, or pushed off the side edge |

## Perception, Monte Carlo (`scripts/eval_perception.py --n 150 --n-faults 40 --sweep`)

Random plates with 1.5 mm / 2 deg print-placement noise; a quarter have a missing part and a quarter a part nudged
5-12 mm; every trial draws a new camera calibration error.

| Configuration | Found | Centre error mean / p95 | Yaw mean | Missing part detected | Calibrated covariance (NEES, 2 = honest) |
|---|---|---|---|---|---|
| Perfect depth, original estimator | 100 % | 0.33 / 1.10 mm | 0.42 deg | 59 % | 0.97 |
| Perfect depth, new estimator (default) | 100 % | 0.20 / 0.54 mm | 0.09 deg | 100 % | 0.79 |
| Realistic, no corrections | 89 % | 5.07 / 11.33 mm | 1.54 deg | 56 % | 19.4 |
| **Realistic, with corrections** | **100 %** | **0.51 / 1.04 mm** | **0.47 deg** | **100 %** | **1.03** |
| Harsh, no corrections | 60 % | 17.7 / 56.1 mm | 2.01 deg | 56 % | 8.2 |
| Harsh, with corrections | 99 % | 1.80 / 4.33 mm | 1.43 deg | 76 % | 2.38 |

The corrections are worth an order of magnitude: realistic calibration error alone moves every estimate by ~5 mm,
more than the 8 mm jaw pre-shape margin can absorb. Details and sources: [perception.md](perception.md).

## End to end, single-row cell: 40 random plates per batch, 195 parts (`scripts/eval_unload.py --realism ...`)

| Sensors and actuation | Placed (robot's report) | Robot believed placed but wrong | Doors open / closed | Beds left clear | Sim s per pick |
|---|---|---|---|---|---|
| Ideal | 195/195 (100.0 %) | 0 | 40/40 / 40/40 | 40/40 | 22.2 s |
| Realistic | 194/195 (99.5 %) | 0 | 40/40 / 40/40 | 40/40 | 22.1 s |
| Harsh | 137/195 (70.3 %) | 1 | 40/40 / 38/40 | 37/40 | 23.2 s |
| Realistic + faults | 162/195 (83.1 %) | 0 | 40/40 / 40/40 | 39/40 | 22.1 s |

"Placed" counts the parts the robot reports placed (`outcome:placed`). Ground truth is that minus the false successes
in the next column (harsh: 137/195), plus any part that did reach its bin while the robot reported a failure: these
batches did not score that case, and on the two-row cell below it is large under harsh sensing.

## End to end, two-row cell: fresh seeds

The default cell, seeds 100+ (not used during development), row A = printer 4, row B = printer 14 (commands in
[results.md](results.md)). "Reported" is the robot's own verdict; "true" is scored against each part's intended bin.

| Batch | Reported placed | Truly placed | False / missed successes | Debris left on the bed | Beds truly clear | Doors | Sim s per pick |
|---|---|---|---|---|---|---|---|
| Realistic + faults, row A, 20 plates | 67/82 | 68/82 (82.9 %) | 0 / 1 | 6/20 | 14/20 | 20/20 / 20/20 | 15.3 s |
| Realistic + faults, row B, 20 plates | 67/82 | 67/82 (81.7 %) | 0 / 0 | 8/20 | 12/20 | 20/20 / 20/20 | 20.1 s |
| Harsh, row A, 10 plates | 31/43 | 41/43 (95.3 %) | 0 / 10 | - | 10/10 | 10/10 / 10/10 | 15.9 s |
| Harsh, row B, 10 plates | 30/43 | 39/43 (90.7 %) | 0 / 9 | - | 10/10 | 10/10 / 10/10 | 20.0 s |

Re-running the published realistic seeds 0-19 on the same code reproduces them seed for seed (row A 99/100, row B
100/100). What the failures are:
- **Faults, parts.** Every plate has one part tipped onto its side. The camera no longer matches it ("missing") or the
  pick fails; it is then picked into the reject bin as an unknown blob (21 of 29 failed parts), raked into the tray
  (2) or pushed off a side edge (6, removed by the housekeeping). No part was left on a bed, so picking matches the
  single-row fault batch (83 %).
- **Faults, debris.** The lump stays on 14 of 40 beds, against 1 of 40 on the single-row cell. The two-row bed is
  12 cm farther from the rail (aisle half-width 0.40 m against 0.28 m), and the low poses the rake and the blob grasp
  need are at the edge of reach: for a 42 x 10 mm gear lump at (56, 98) mm, the blob grasp finds no IK solution in
  1829 tries and the front rake has exactly one feasible candidate (nominal arm model, clean bed). The camera flags every such bed "not empty", so none is silent, but each needs a person.
- **Harsh.** 19 of the 25 parts the robot reported as failed are in their bins: on two plates per row the drawn
  wrist-camera hand-eye error (1.5 mm per axis) puts the finger pads inside the "between the jaws" volume, so every
  release on that plate reads "stuck on a finger". These are false alarms (the part would be reprinted, and an extra
  part ships). The real failures are parts the camera cannot match (raked or picked away) and one drop in transit.

## Door (`scripts/test_door.py --all`, `scripts/door_robustness.py`)

Physical jaw grasp, no rigid lock. Single-row cell: 10/10 printers open and 10/10 close (20.9 s / 21.0 s). Two-row
cell: 40/40 opened and 40/40 closed in the realistic batches of both rows (23.2 s / 21.1 s, `runs/timings.json`),
and every door of the fresh two-row batches above opened and closed. Single-row sweeps: robust to catch forces of
0-54 N, hinge friction and damping, door sag, 2x swing speed and printer placement error up to +-10 mm / +-2 deg
(20/20). Measured coupling loads give the grip spec for a real gripper: closing force >= catch force + 1.5 N (25 N with
margin), >= 25 mm of pad slide room either side of the bar, catch <= 18 N for a squeeze hold. See [door.md](door.md).

## Bed to box: large parts and the packing station

Large parts (housing 85 x 55 x 35 mm shell, rimmed plate 120 x 50 x 15 mm, bar 150 x 18 x 18 mm tube) are picked
across their narrow side, carried out of the printer hanging as gripped, and set down flat in a tray slot on the
carriage. Realistic preset, plates packed with the farm's 3 h cap (commands in [results.md](results.md)):

| Batch | Large parts truly in their slot | Small parts truly placed | False successes |
|---|---|---|---|
| Row A (printer 4), 20 plates | 20/20 | 11/11 | 0 |
| Row B (printer 14), 20 plates | 20/20 | 11/11 | 0 |
| Dense-plate stress, row A, 20 plates packed up to 8 h (never printed by the farm) | 34/35 | 60/73 | 0 |
| Dense-plate stress, row B, same | 33/35 | 61/73 | 1 |

- Perception of large parts (40 scenes, 56 large parts): centre error mean / p95 0.63 / 1.31 mm on the realistic
  preset, 2.23 / 4.16 mm on the harsh one. Long parts are matched on their footprint rectangle, not a disc.
- Slot placement: across the slot mean about 1 mm (max 3.8 mm), along it max 6.3 mm, yaw max 1.7 deg. The station
  measures the part in its slot with the wrist camera before picking it again.
- Dense plates are the weak case: large parts and their jaw keep-outs take the front of the bed, so small parts land
  deep, where a lid or a gear near full jaw opening has a marginal grasp. The slicer stand-in only places a part where
  that SKU's own grasp map says it can be picked, which removes the ungraspable cells but not the marginal ones.
- A long part lying into the printer that cannot be picked has no rake plan (the reject tray is in front of the bed):
  it needs a person. Lying along the door, the bar and the plate are raked into the tray.

**Packing station sensing.** The empty-tote check is a depth camera on a boom straight above the drop line, out of
the arm's reach. Each check compares a scan with the per-pixel median of five reference scans taken at commissioning,
thresholds at 4.5 times the noise measured from those scans, and reports a leftover only when a second scan sees the
same blob; a patch without depth large enough to hide a part counts as "cannot verify". Scored against ground truth on
the realistic preset (`scripts/station_cam_check.py`, 200 seeds): 0 false alarms in 800 empty checks (including 200
real dumps) and 0 missed leftovers in 800 checks with a part left in the tote (against a wall, at the flap edge, wedged
above the floor). The first version (one reference scan, a camera on a post looking in at 37 degrees, so the near wall
hid half the floor) raised 20 false alarms and missed 61 leftovers in 240 checks each.

**A part on a tote wall.** In the final 16-plate farm run one lid (50 x 32 mm), dropped into its tote, came to rest
on the tote's rim instead of its floor. The trap door cannot move it; the station camera saw it on both checks and
the tote went to an operator, who put the lid into the box (nothing was lost). The unload's own release check
only asks whether the part left the fingers, so it counted the drop as placed.

**Large-part placement at the station.** The arm measures each large part in its tray slot with the wrist camera,
then plans grasp and place together, and collision-checks the whole place line (held part 3 mm, arm 6 mm from the
box). Realistic actuation, 18 seeds for each of 8 part/box pairs (`scripts/station_place_check.py`): 144/144
placed, 0 contacts with the box. The tightest pair is the plate in an S box: its smallest real arm-to-box clearance
was 1.5 mm (median 5.1 mm), because the arm model can be about 5 mm off the real arm. Two causes of earlier failures
were sim-to-real effects worth keeping in mind on hardware: grasp heights must come from the robot's own registered
model of its carriage, not from the design drawing, and a "still held" check must look where the held part actually
is (a tall part pinched low shows its top face outside the usual window).

## Not modelled (yet)

- Chamber temperature and a glass lid in front of the depth camera; the depth noise is applied to a rendered image,
  so holes do not depend on the scene content.
- Bed adhesion is an on/off weld released at 35 C: no warped parts, no stringing, no brims.
- Part geometry is primitive shapes, not meshes; friction values are chosen, not measured.
- MuJoCo soft contacts with noslip make friction stickier than real pads (door side loads are an upper bound).
- All sensor and actuator magnitudes are labelled assumptions in `perceive/noise.py` and `farm/sensors.py`, not
  measurements of a specific unit.
- The unload batches (`scripts/eval_unload.py`) spawn every part free on the bed (`spawn_plate(..., stuck=False)`):
  adhesion is not exercised there at all. Only the whole-cell run welds parts while the bed is hot.
- MuJoCo combines the friction of two geoms by taking the larger coefficient, so every contact of a part (pads, PEI
  sheet, other parts) uses the parts' 1.2: grasps are stickier and raking is harder than PLA on PEI.
- The wrist camera has Gaussian depth noise and a hand-eye offset only: no dropouts, flying pixels or reflections.
- Simulation housekeeping removes a part that left the bed anywhere other than a bin or the reject tray (floor, deck,
  the chamber beside the bed). A real cell has to find and collect those parts.
- Debris is one upright catalogue primitive (gear, cube, block or knob). Spaghetti, half-printed parts still stuck to
  the sheet and loose filament are not modelled.
- Packing station: the boxes ride on prescribed slide joints (their contents are physics), and from sealing on
  the line is kinematic: the contents are parked when the box enters the sealer and a lid is shown. The
  consolidation shelf for partial packages is bookkeeping, not physics.
- The simulated arm is the I2RT YAM on a rail. The real-hardware kit in `real/` uses an SO-101 with hobby servos, so
  none of the arm numbers here (accuracy, speed, reach, grip force) carry over to that arm without new measurements.

## Where the code still touches simulator truth (audit)

Sensor models read the simulator by design: the cameras render the true scene from their true mounts (calibration
error is added to what the software believes), encoders and the gripper gap are the true joints plus offsets, play
and noise, and the reed switch reads the true door gap. The `ideal` preset falls back to exact readings. Everything
else in the decision code was checked; fixed in the audit: the jaw opening used to plan transfers (now the gripper
encoder), the arm state used to locate the bins for a drop (now independent of it), and the bed region used for the
scan quality checks (now from the believed camera calibration). What remains, all outside the robot's own decisions:
- `IK.sync` copies the true arm, rail and jaw joints into the planning state; every IK and collision query sets them
  explicitly, so the copy only seeds the riders of the oracle mode.
- `sim/farm.py` decides "blocking -> operator" or "off the bed -> reprint" from the true part position, and keeps
  printing when the camera says "not empty" or "cannot verify"; the DES does the same (`p_blocked_plate` from
  `scripts/make_timings.py` is not used by `fleet/des.py`). A real cell must stop on the camera's verdict.
- `sim/farm.py` sets no planning belief before its first bed scan, so the drive and door moves until then plan with
  the true part poses (the batches set the slicer prior first; the fix is `ik.belief = prior_belief(...)` in
  `Farm.unload` before the door opens).
- Housekeeping (see above) parks parts that left the bed, and the operator model places parts and closes doors.
- `--oracle` (A/B comparison only) plans and checks from truth on purpose.

## Bridging to hardware

What has to be measured on a real cell before the numbers on this page mean anything for it, ranked by how far the
gap can move the headline result and how little the model is backed by data today. For each: a cheap measurement,
the acceptance threshold the simulation implies, and where the measurement goes in the code. Thresholds marked
"sim" come from the batches and sweeps above; the rest are engineering margins, labelled as such.

**1. Bed adhesion and release** (highest risk: not in the unload batches at all, see above).
- Measure: print each catalogue SKU on the production sheet, in each filament in use. Log the bed thermistor while it
  cools (fits `COOL_TAU_S`). At the release temperature, pull each part straight up and push it sideways at its grasp
  height with a push-pull force gauge (a luggage scale and a string loop works); 10 parts per SKU.
- Pass: the vertical release force stays below half of what the jaws can hold by friction (2 x mu x squeeze, with
  mu from item 3; margin, not sim); the sideways release force stays below the push the arm delivers at the
  fingertips (measure that with the same gauge, arm commanded 5 mm into it). A part that fails either test is a
  `not_lifted` pick and a rake that cannot move it.
- Update: `RELEASE_C`, `COOL_TAU_S` in `sim/farm.py` and `fleet/des.py`; a release-force model in place of the on/off
  weld in `farm/world.py` (`set_stuck`); fault batches with stuck parts (`spawn_plate(..., stuck=True)` plus a
  release law) in `scripts/eval_unload.py`.

**2. Arm calibration, backlash and compliance.** Realistic preset: 0.15 deg joint zeros, 0.3 mm links, 1 mm base
registration, 0.1 deg backlash, 20 ms latency. That gives 99-100 % placed; the harsh preset (about 3x) gives about
70 % reported placed, and the wrist alignment (it measures the part relative to the jaws at the pre-grasp) absorbs
static error only up to `WRIST_MAX_SHIFT` = 10 mm.
- Measure: (a) homing repeatability: home 20 times against a dial indicator at the gripper; (b) static accuracy:
  touch the pinch point onto 9-12 points of a printed grid plate on the bed and fit them with `real/calib.py`
  (Kabsch): the residual RMS is link plus registration error; (c) backlash per joint: dial indicator at the fingertip,
  push both ways by hand with the servos holding; (d) compliance: hang the heaviest part plus 0.5 kg and read the sag;
  (e) latency: timestamp a step command and the first encoder change; (f) encoder noise: log the encoders at rest.
- Pass: touch-off residual <= 1 mm RMS (sim: the realistic preset's 1 mm registration is fully absorbed); backlash
  <= 0.2 deg per joint (0.2 deg at 0.5 m reach is 1.7 mm; margin); latency <= 30 ms; the pinch error the wrist camera
  measures at the pre-grasp <= 3 mm on average, well inside the 10 mm gate. Anything near the harsh preset (0.4 deg
  zeros, 3 mm registration, 0.3 deg backlash, 50 ms) predicts about 70 % placed.
- Update: a measured preset next to `"realistic"` in `farm/sensors.py` `PRESETS` (`zero_sigma_deg`,
  `link_sigma_mm`, `base_sigma_mm`, `rail_zero_sigma_mm`, `backlash_deg`, `latency_s`, `enc_noise_deg`,
  `grip_noise_mm`). Compliance under load has no field yet: it would need one in `Actuation`.

**3. Gripper squeeze force and pad friction.** Sim: gripper force-limited at 20 N (10 N per pad), pick squeeze set by
a position target 5 mm inside the part (`Executor.grip`), part friction 1.2 with noslip (stickier than real).
- Measure: close the jaws on a load cell (or a kitchen scale and a printed spacer) at several targets: force vs
  position error, and the force limit; pull a gripped PLA part out of the jaws with a force gauge:
  mu = F_pull / (2 x F_squeeze); drag a part across the PEI sheet with the gauge: the part-bed mu that the rake works
  against.
- Pass: pull-out force >= 5x the heaviest part's weight at its grasp height (margin for 1.3 g transport loads and a
  tilted grasp; not sim); door hold as specified in [door.md](door.md): closing force >= catch force + 1.5 N (sim),
  >= 25 mm pad slide room either side of the bar (sim); the door sweep passed with pad mu down to 0.5 (sim).
- Update: the part `friction` in `farm/scene.py` (split pads, PEI and part-part with geom priorities, since MuJoCo
  otherwise takes the larger value), the gripper `forcerange` in `farm/scene.py`, the squeeze in `plan/motion.py`
  `Executor.grip` and `plan/door.py` `SQUEEZE_N`, and `grip_noise_mm` in `farm/sensors.py`.

**4. Chamber and wrist depth cameras: noise and calibration.** Sim: the realistic chamber camera gives 0.51 / 1.04 mm
centre error (mean / p95) after corrections; at 2x the noise still 100 % found (1.30 / 3.49 mm); at 3x, 42 % of
beds with a 5-15 mm leftover are reported empty (harsh: 2 %; [perception.md](perception.md)). Wrist camera: 0.5 mm noise, 0.5 mm hand-eye error.
- Measure: mount the camera at the chamber pose. (a) 100 frames of the bare PEI sheet: per-pixel temporal sigma
  across the 0.33-0.55 m range (`axial_mm`), the autocorrelation of the residual after a plane fit (`axial_corr_*`),
  the fill and the sheen hole (`dropout`, `sheen_*`); (b) a printed step target (5 / 10 / 20 mm) and parts in white,
  black, silver and translucent filament: flying pixels at the steps, dropout per colour (`flying_frac`, `dark_*`);
  (c) a gauge block or a 1-2-3 block for the depth scale (`depth_scale`); (d) the plane and outline corrections the
  pipeline already logs (`meta["plane"]`, `meta["outline"]`) over a week of scans: their spread is the calibration
  error and its drift (`ext_t_mm`, `ext_r_deg`), and the failure counts give `p_no_frame`, `p_dead`, `p_blackout`;
  (e) wrist: the pinch-frame cloud of the empty open jaws (should hold ~0 points between them) and of a known block
  between them (its offset is the hand-eye error).
- Pass: axial sigma <= 1.6 mm at 0.40 m (2x realistic, sim); post-correction centre error p95 <= 4 mm, half the 8 mm
  jaw pre-shape margin (sim: harsh is 4.3 mm); **no false "empty" in >= 100 scans of a bed with a 5-15 mm leftover**
  (printing over a part is the one failure that must not happen; sim realistic: 0 %); false "not empty" <= 3 % (each costs an operator check; sim realistic:
  3 %); wrist hand-eye error <= 0.5 mm and the empty-jaw count well below `N_BETWEEN` (sim: at the harsh 1.5 mm
  whole plates are reported "stuck on a finger" while every part is in its bin, see the two-row batches).
- Update: a measured `CameraNoise` preset in `perceive/noise.py` `PRESETS`; `wrist_sigma_mm`, `handeye_sigma_mm` in
  `farm/sensors.py`; `N_BETWEEN` in `plan/unload.py` if the empty-jaw count is not near zero.

**5. Door catch force and handle grip.** Sim: magnetic catch F0 = 10 N falling as 1/(1 + g / 2 mm)^2, hinge damping
1.5 N m s/rad and friction 0.05 N m, a 15 mm bar, 5 N squeeze per pad; robust from 0 to 54 N of catch, hinge
friction up to 1 N m and printer placement error up to +-10 mm / +-2 deg.
- Measure: force gauge or luggage scale at the handle, pulled normal to the glass: the pull-off force, and with
  0.5 / 1 / 2 / 5 mm shims at the magnet the force-gap curve (scale to the magnet by the lever ratio from the hinge);
  the force at the handle that keeps the door moving slowly (hinge friction); the reed switch's make and break gaps
  with feeler gauges; the open stop angle; the bar diameter and its play in the bracket.
- Pass: catch force at the magnet <= 18 N for a squeeze hold with the 20 N gripper (sim), otherwise non-back-drivable
  jaws or a hook grasp; the reed switch makes below 3 mm and breaks above 4 mm (sim thresholds `REED_GAP`,
  `REED_HYST`); the door opens to >= 95 deg (`OPEN_OK_RAD`).
- Update: `LATCH_F0`, `LATCH_G0`, `LATCH_GMAX`, `REED_GAP`, `REED_HYST` in `farm/world.py`; the hinge `damping` and
  `frictionloss` in `farm/scene.py`; `HANDLE_R`, `GRASP_DEPTH`, `SQUEEZE_N`, `OPEN_OK_RAD` in `plan/door.py`.

**6. Part geometry.** Sim: boxes, cylinders and an L at the catalogue dimensions; no elephant foot, warp, brim or
stringing.
- Measure: calipers on the footprint and height of 10 prints per SKU; feeler gauge under the corners (warp);
  photographs of stringing between neighbouring parts; the slicer's labelled footprints against the prints.
- Pass: footprint and height within +-0.5 mm of `farm/catalog.py` (the top-face band is +-1.5 mm, `TOP_TOL_MM`);
  warp < 1 mm; no strings that join two parts into one camera blob.
- Update: the `SKU` dimensions in `farm/catalog.py`; `TOP_TOL_MM`, `FOOT_MARGIN_MM` in `perceive/verify.py`; meshes
  instead of primitives if a shape differs.

**What should transfer, and what is most at risk.** Expected to transfer (belief, not measured): the structure of the
pipeline (slicer prior plus depth refinement, sensor-only checks, the rake-or-reprint fallback), the perception
corrections as long as the real camera is within about 2x of the realistic model, the door strategy if the catch is
<= 18 N, and the fleet-level conclusions that depend on timings rather than contact physics (the arm is idle most of
the time, so reliability matters more than speed). Most at risk: the absolute pick rates (99-100 % realistic), which
rest on free parts, sticky simulated friction and primitive geometry; the rake and debris clearing, which already
degrades on the two-row cell where the bed is 12 cm farther from the rail (14 of 40 debris lumps left, above); the
zero-intervention fleet numbers, because the DES restarts a printer even when the robot's camera says "not empty" or
"cannot verify" (3-7.5 % of realistic unloads, 35 % of the two-row fault plates), where a real cell must call a
person; and every time figure, because the YAM arm, its speed limits and the 0.4 m/s rail are assumptions.
