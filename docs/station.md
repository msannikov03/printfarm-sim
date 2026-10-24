# Bed to box: the packing station (two-row cell)

The station is at the +x end of the rail. It takes one package at a time, using one box and one tote, and turns the
package into a sealed box on its carrier lane. The code is in `plan/station.py` (the skills), `farm/scene.py` (the
geometry and `station_layout()`), `farm/world.py` (trap doors and boxes) and `sim/farm.py` (the farm flow).

## Hardware

- **Totes.** The six order bins `o0..o5` are package totes with a trap-door bottom. The flap is hinged at the tote's
  +y inner edge. A position actuator holds it shut against its joint stop, and a hinge encoder reads its angle
  (`World.flap_sensor`). The deck is open under each tote. The waste bin has the same kind of flap. The reject bin is
  a plain bin that an operator empties.
- **Large-part tray.** Large parts lie in the tray slots `L0`/`L1` on the deck.
- **Cross conveyor.** It runs along world y at `x = X_DROP`, under the deck's +x overhang. `X_DROP` follows from the
  clearance stack at full travel: column, rail end stop, `STATION_CLEAR` and the conveyor frame (`farm/catalog.py`).
  The carriage column sits behind the arm base, so the deck's +x end can overhang the conveyor. This also keeps the
  big-part place within the arm's reach.
- **Box magazine.** It holds S/M/L stacks on the +y side, with a cross transfer in front.
- **Place lift.** It sits beside the deck on the +y side and raises the box to the arm for the big-part place.
- **Dump lift.** It sits under the tote rows and raises the box until its rim is just below a steep drop chute. The
  chute funnels the falling parts well inside the narrowest box. It hangs below the swing of the open flap.
- **Station depth camera.** It hangs from a boom straight above the drop line and looks down into the docked tote
  to verify that it is empty. The boom comes from a post past the deck's +x edge, above the robot's reach
  (`C.ROBOT_TOP_Z`). An earlier camera on the post looked in at 37 degrees, and the near wall of the 70 mm deep tote
  hid half of the floor from it.
- **Downstream.** The sealer tunnel is on the -y side, followed by a sorter and the three carrier lanes.
- **Scrap bin.** It sits under the waste bin's dock position.

**Clearance by construction.** Nothing can collide with the carriage. Over the whole rail range, with the arm tucked,
no geom of the station, the lab dressing, a printer or a bench intersects the carriage's swept volume: deck, totes,
flaps, tray, column, dressing or arm. Visual geoms are included. Any station hardware that reaches above the deck's
underside lies beside the deck (|y| beyond its extent), past its +x edge at full travel, or above `C.ROBOT_TOP_Z`
plus the clearance (the camera boom). `C.ROBOT_TOP_Z` bounds the carriage and the arm over all joint ranges.
`tests/test_station.py` checks this with an AABB sweep, lab dressing included primitive by primitive. It also checks
every box on its route, and the raised box against the flap at every angle.

## Sequence per package

1. **Request box.** The box of the package's size leaves the magazine. If the package has large parts, it goes to
   the place lift. Otherwise it goes straight under the package's tote. The carriage docks while the box travels.
2. **Place the large parts.** The wrist depth camera measures each large part where it lies in its slot. The arm then
   grasps it top-down across its narrow side and sets it on the box floor at the package layout pose. The robot
   shifts the pose toward the box centre if the fingers need room. The jaws must read empty after release.
3. **Dump the tote.** The rail brings the tote over the conveyor and the dump lift raises the box. The trap door
   opens, shakes and shuts, and its encoder must read it shut. The station camera then compares a depth scan of the
   tote with the empty-tote reference recorded at commissioning, so its calibration error cancels (details below).
   If a part is left, the door opens again with harder shaking. If the tote is still not empty, an operator clears it
   (a counted intervention).
4. **Seal and ship.** The box rides into the sealer and then along the sorter to its carrier lane. Only one box is in
   the tunnel at a time.

Parts parked earlier on the consolidation shelf (a full caddy) are added to the box by the operator.

## Empty-tote check

The camera has the same error model as the chamber cameras (`perceive/noise.py`). A scan is the per-pixel median of
24 frames (0.8 s at 30 fps, which the station waits).

- **Reference.** At commissioning, each empty tote is scanned five times at its dock position. The per-pixel median
  is the reference. The spread of the five scans is the measured noise of one scan.
- **Floor mask.** The flat floor plateau around the tote centre is taken from the reference itself, eroded 4 px
  (about 3 mm) off the walls. No calibration is needed to find it.
- **Raised pixels.** A pixel is raised when the reference minus the scan exceeds the larger of 3 mm and 4.5 times
  the measured noise of the difference (4.3 to 5.6 mm under realistic noise). Raised pixels are opened with a 3x3
  kernel. Blobs of 60 px or more count; the smallest part's top is about 250 px.
- **Decision.** A blob that is seen again, overlapping, in a second scan is a leftover. One scan's blob alone can
  come from a smeared lens. A scan without blobs means the tote is empty, but only once the floor has been seen with
  no hole a part could hide in (holes and specular sheen leave patches with no depth). An occluded scan (something
  near the lens) or one with no depth is repeated. After four scans without an answer, the operator looks.

`scripts/station_cam_check.py` scores the check against ground truth: empty totes (fresh and after a trap-door
cycle), one leftover part (anywhere, against a wall, on the flap's free edge, or wedged above the floor), and real
dumps. Realistic preset, 200 seeds: 800 checks of empty totes gave 0 false alarms and 0 non-answers, and 800 checks
with a leftover gave 0 misses, at 1.6 scans per check on average. The camera beside the deck, scored the same way on
80 seeds, raised 20 false alarms in 240 empty checks and missed 61 of 240 leftovers.

## Big-part place clearance

The place is planned with the part held. The held part and the arm must stay 3 mm from everything along the place
line. The arm alone must also stay `PLACE_CLEAR` (6 mm) from the box down the whole line, with the jaws at their
release opening, because under realistic actuation the real arm is a few millimetres off its model. A place that
needs more room is released higher (`PLACE_GAPS`). `scripts/station_place_check.py` counts robot-to-box contacts
over many places. Realistic preset, 18 seeds x 8 part/box pairs (144 places): with a 3 mm clearance, 2 housing
places in the S box touched the rim with a finger link. With `PLACE_CLEAR` there are 0 contacts. The real arm was up
to 5 mm off its model on the place line, so the 6 mm clearance is a margin, not a guarantee. The tightest pair is
plate -> S: it is released 30 mm up, and its smallest real clearance was 1.5 mm (plate -> M: 5.3 mm).

Two grasp details make all 144 places succeed (134 before):
- **Tray-relative targets in the robot's own carriage model.** The tray rides on the carriage with the arm. The
  robot's model of its carriage carries the registration error, so the pinch over the tray floor is planned from
  that model's floor (`Station.deck_dz`), not from the design height. With the design height, a flat plate's
  fingertips, planned 1.5 mm over the floor, sat 0.1 mm over the modelled floor when the model was 1.4 mm high, and
  no grasp passed the check (`no_plan`). Station IK also solves to 0.2 mm and 0.3 deg (`IK_TOL`).
- **Held check up to the part's top.** A housing pinched at 0.6 of its height shows the wrist camera only its top
  face, 14 mm above the pinch point. That is outside `WristCam.between_jaws`'s +-12 mm volume, so a held housing read
  as `not_held`. The station's check reaches up to the held part's top.

## What is kinematic

- **Boxes.** A box moves along prescribed, accel-limited paths. Its three slide joints get their position and
  velocity set before every physics step. Parts inside a box are physical: they are carried by friction and bounce off
  the walls. A mocap box could not carry them, because MuJoCo gives mocap bodies no velocity and the walls sweep
  through loose contents.
- **Sealing.** The station is kinematic from sealing on. Once the box is inside the tunnel, every part inside it is
  taken out of the simulation (parked) and a lid is shown. The sealed box then moves as a rigid body to its lane.
- **Shelf and lift tables.** The consolidation shelf is kinematic, and so are the lift tables, which are visual and
  follow the box.

Ground truth (`plan.station.part_in_box`) is used only for scoring. When a box is sealed, the station counts the
parts it finds inside against the package (`parts_missing`).

## Measured times

`scripts/run_farm.py --station-bench N` writes `runs/station_times.json` with the mean, sd and n of each step (sim
seconds). The steps are defined in `plan.station.Station`. In farm runs, the drive from the printers to the station
is timed as `to_station`, not as `dock`, and a package's time does not include it. `dock` is a move between station
positions.
