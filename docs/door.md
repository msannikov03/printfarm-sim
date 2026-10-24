# Door handling

Every unload starts and ends with the printer door: the robot opens it, unloads the bed, and closes it again.
This page covers how the door is built in the simulation, how the robot plans the swing, what holds the handle,
how the robot knows where the door is, and what the robustness sweep measured.

Code: `plan/door.py` (planning, execution, sensing, force probe), `farm/world.py` (magnetic catch, reed switch,
door belief), door bodies in `farm/scene.py`, `scripts/test_door.py`, `scripts/door_robustness.py`, `tests/test_door.py`.

## Design

**Door.** A 340 × 392 mm glass leaf, hinged on the left (seen from the aisle) with a hinge axis at the enclosure's
left edge. It has an end stop at 102° and opens to 100°. The leaf keeps a 4 mm clearance to the plinth below and the
header above.

**Handle.** One everyday D-handle near the free edge: a vertical bar 15 mm in diameter and 160 mm long, on two
stand-offs that hold it 110 mm in front of the glass, 300 mm along the leaf from the hinge and 260 mm above the
enclosure floor. The bar spins freely on its own axis (joint `pinspin{i}`), like a sleeve on a pin. The 110 mm
stand-off leaves room for a jaw between the bar and the glass. There is no inner pin, which would get in the way of
the print head.

**Swing.** `plan_door()` searches for one continuous whole-body arc (6 arm joints + rail) that drags the bar from the
current angle to the target in 14 steps. Every sample must have a collision-free IK solution, and the score is the
worst manipulability × joint-limit margin along the arc. The arm folds below the door line (`LOW_TUCK`) when a
carriage move passes under the open leaf; other moves slide with the arm upright (`transfer`, DESIGN.md). If no single arc exists, it falls back to two arcs with a regrasp at
50/35/65°, joined by a tucked carriage slide. In the current layout the single arc is always found.

**Grasp geometry.** The jaws close along the bar's direction of travel, so the pads push the bar along the arc.
The approach is horizontal and points along the leaf toward the hinge, so the 80 mm pad length lies along the
radius and the 12 mm pad width is vertical (checked on the planned arcs: approach · radial = −1.00, pad width ·
vertical = −1.00). The pinch target is placed so that the bar sits **25 mm behind the pinch point**
(`GRASP_DEPTH`). The YAM pads run from about 10 mm ahead of the pinch point to 70 mm behind it, so a bar held *at*
the pinch point has only about 2 mm before it slides out past the fingertips. That was the cause of every
placement-error failure in the first sweep (below). At 25 mm deep, the bar has about 27 mm of slide room to the tips
and 37 mm to the back, and the two fingers cage it.

## Coupling: physical by default, weld optional

| | `COUPLING = "physical"` (default) | `COUPLING = "weld"` (option) |
|---|---|---|
| what holds the bar | jaw contact only: pads squeeze the bar with 5 N per pad; the bar spins freely | a weld equality `coupler{i}` (link_6 ↔ pin{i}) switched on after the jaws close |
| along the jaw axis | pads push the bar (normal force), up to the jaw actuator's closing force | rigid |
| in the pad plane | friction (μ = 1.2); the bar may slide along the pads | rigid |
| what can go wrong | bar slides out of the jaws, jaws are back-driven open | nothing: the weld hides overloads |

In the physical mode nothing is locked. The door moves only through contact forces, and success is read from
sensors. Switch with `plan.door.COUPLING = "weld"` or `scripts/test_door.py --coupling weld`.

## Sensing: no ground truth in the decisions

| Question | Sensor | Implementation |
|---|---|---|
| Is the bar in the jaws? | gripper encoder: jaw half-gap within −3/+4 mm of the bar radius | `holding_bar()`, after closing and after the swing |
| Door angle while holding | arm joint + rail encoders → FK of the pinch point in the robot's own model → minus `GRASP_DEPTH` along the approach → inverse hinge geometry (`handle_angle()`) | `sense_door()` → `World.door_est[i]` |
| Is the door shut? | reed switch at the magnetic catch: closes below a 3 mm gap, reopens above 4 mm (1 mm hysteresis) | `World.reed_closed(i)` |
| Door angle at any time | `World.door_belief(i)`: 0 if the reed switch is closed; otherwise the last FK estimate; with no estimate, nominal open | used by `plan_door()` and by `transfer()` for the planning scene |

- Encoder noise defaults to σ = 0.05° per joint and σ = 0.2 mm on the rail (**assumption**). When an executor
  carries the hardware model `ex.hw` from `farm/sensors.py` (zero offsets, backlash, noise), the door code reads
  its encoders and gripper gap instead.
- An opening counts as successful if the FK estimate at release is ≥ 95°. A closing counts as successful if the
  reed switch is closed after a 0.3 s dwell. Otherwise the robot re-grasps and pushes once more (2 tries).
- If a door is ajar (reed switch open) and the robot has no estimate of its angle, `plan_door(opening=True)` refuses
  with "angle unknown" rather than assuming it is open.
- The re-push aims 4° past nominal closed (`OVERPUSH_RAD`). If a printer stands a few mm off its nominal pose, the
  nominal 0° arc ends short of the latch, so the leaf's own stop and the back-drivable jaws limit the extra push.
  The first attempt always aims at 0°.
- `World.door_angle(i)` (ground truth) is used only for printing, reporting and test setup.

## Magnetic catch and hinge (modelled)

- **Catch:** a magnet at the free edge, 0.33 m from the hinge axis. The pull is F(g) = F0 / (1 + g/2 mm)² for gap
  g ≤ 20 mm, so it is strongest at contact, half at 0.8 mm and about 8 % at 5 mm. It is applied as a hinge torque in
  `World.step()`. The default is **F0 = 10 N (assumption)**: the real holding force of a P1S-class door is not
  known. Catalogue ratings of small cabinet catches, used as the sweep range:
  - a magnetic touch latch, about 1 kg ([Bunting](https://e-magnetsuk.com/product/recessed-magnetic-touch-latch-tl4n-black));
  - a plastic cabinet catch, 5 lb ≈ 22 N ([Amerock](https://absupply.net/amerock-ambp9765-pt-cabinet-catch-plastic-casing-magnet-and-strike-plate-pull-force-5-lb.aspx));
  - a single-magnet catch, about 3 kg ≈ 30 N ([Zalonis](https://zalonis.eu/en-gb/furniture-fittings/furniture-accessories/furniture-magnets/magnetic-catch-20x21x44-counterpart-black));
  - a plastic catch, 5.5 kg ≈ 54 N ([RS PRO](https://uk.rs-online.com/web/p/door-catches/6659954)).

  These are nominal pulls against a steel plate at zero gap.
- **Hinge:** damping 1.5 N·m·s/rad, Coulomb friction 0.05 N·m, leaf mass about 1.8 kg (**assumptions**; swept below).
- The old latch model (a linear spring within 3.4° of closed) had zero holding force at contact, so opening needed
  no break-away pull. The magnet model replaces it.

## Results (all measured, sim time)

### All 10 printers, open + close (`scripts/test_door.py --all [--coupling weld]`)

| | physical (default) | weld |
|---|---|---|
| opened (sensor and truth agree) | **10/10** | 10/10 |
| closed (reed switch and truth agree) | **10/10** | 10/10 |
| sensor said OK but truth disagrees | 0 | 0 |
| open / close time | 20.9 s / 21.0 s | 20.8 s / 21.0 s |
| true angle after open / after close | 99.90° / −0.05° | same |
| FK estimate after open | 99.77° (error 0.13°) | — |

The 10 printers are geometric copies, so with nominal geometry they behave identically. Variation comes from the
sweep below.

### Forces the coupling carries (max over 20 arcs, opening and closing)

| | physical | weld |
|---|---|---|
| peak push along the travel direction (magnet break-away) | 11.2 N | 11.5 N |
| push during the swing, p95 | 1.8 N | 1.8 N |
| side load in the pad plane, p95 / peak | 10.4 / 11.1 N | **24.5 / 27.0 N** |
| peak hinge torque | 3.6 N·m | 3.7 N·m |
| peak arm joint torque (fraction of limit) | 17 % | 23 % |
| force on the door other than the bar (glass, stand-offs) | 0 N | 0 N |

The side load is not needed to move the door. It is the arm's planned path fighting the hinge circle. The weld
carries up to 27 N of it rigidly. A real gripper cannot, and does not need to: with friction the bar slides along the
pads, and with a squeeze of 1 N per pad the side load drops to 1.5 N while open and close still succeed (sweep).

### Catch force vs. what holds the bar (`scripts/door_robustness.py --catch 0,10,15,18,22,30,54`, printer 5)

| catch F0 | peak push | jaw gap during the swing (bar = 15 mm) | arm joint torque | mode |
|---|---|---|---|---|
| 0 N | 2.0 N | 13 mm | 11 % | squeeze |
| 10 N (default) | 11.2 N | ≤ 15.5 mm | 17 % | squeeze |
| 15 N | 16.5 N | ≤ 20.5 mm | 24 % | squeeze, jaws back-driven 7 mm |
| 18 N | 19.5 N | ≤ 23.5 mm | 29 % | squeeze, near the 20 N jaw limit |
| 22 N | 24.7 N | **69 mm** | 40 % | hook: jaws thrown open, bar caged |
| 30 N | 31.9 N | 70 mm | 53 % | hook |
| 54 N | 56.2 N | 75 mm | **100 %** | hook, arm at its torque limit |

The jaws are coupled and back-drivable. A push along the jaw axis opens them once it exceeds the actuator's closing
force (20 N in the model). From 22 N the jaws are thrown fully open, and the finger at its travel end stop pulls the
bar like a hook. This works only because the bar sits 25 mm deep and stays caged between the 80 mm fingers. With the
earlier shallow grasp, every catch of 22 N or more failed (0/3 each) because the bar slipped out past the tips.

### Grip-force spec for a real gripper (derived from the runs above)

- **Closing force along the jaw axis** must be at least the break-away push, about F0 + 1.5 N. With a 2× margin,
  that is 25 N for the 10 N default catch. Otherwise the jaws must be self-locking. If they are back-drivable, the
  fingers must cage the bar, and the jaw end stop must carry the push.
- **Squeeze per pad:** 1 N is enough; 5 N is the default. Side loads are internal, so the bar is allowed to slide.
- **Pad length** along the approach: at least 25 mm of slide room on each side of the bar.
- **Opening:** at least 15 mm plus twice the capture tolerance. By geometry (not measured), the YAM's 74 mm opening
  catches the bar with the door up to about ±5° off its believed angle: (37 − 7.5) mm / 320 mm.
- **Catch:** keep F0 at or below about 18 N for a squeeze hold with this gripper. Above 30 N, the arm needs more
  than 50 % of its joint torque just to break the catch.

### Robustness sweep (`scripts/door_robustness.py`)

Setup:
- The robot plans on the **nominal** model; perturbations are applied only to the simulated world.
- Decisions come from sensors only. "OK" means sensor and truth agree.
- Deterministic conditions run on printers 1, 5 and 10. Random placement conditions run with 2 seeds × 10 printers.
- "est err" is the largest |FK estimate − truth| after opening.

| condition | open | close | sensor ≠ truth | open s | close s | peak push N | side p95 N | arm torque % | est err ° |
|---|---|---|---|---|---|---|---|---|---|
| nominal | 3/3 | 3/3 | 0 | 20.9 | 21.0 | 11.2 | 10.4 | 17 | 0.13 |
| no magnet | 3/3 | 3/3 | 0 | 20.9 | 21.0 | 4.2 | 10.3 | 12 | 0.18 |
| magnet 15 N | 3/3 | 3/3 | 0 | 20.9 | 21.0 | 16.5 | 10.4 | 24 | 0.13 |
| magnet 18 N | 3/3 | 3/3 | 0 | 20.9 | 21.0 | 19.5 | 10.4 | 29 | 0.13 |
| magnet 22 N (5 lb catch) | 3/3 | 3/3 | 0 | 20.9 | 21.0 | 24.7 | 10.4 | 40 | 0.15 |
| magnet 30 N (3 kg catch) | 3/3 | 3/3 | 0 | 20.9 | 21.0 | 31.9 | 10.3 | 53 | 0.15 |
| magnet 54 N (5.5 kg catch) | 3/3 | 3/3 | 0 | 20.9 | 21.0 | 56.2 | 11.4 | 100 | 0.13 |
| hinge damping 0.1 | 3/3 | 3/3 | 0 | 20.9 | 21.0 | 11.2 | 10.4 | 17 | 0.13 |
| hinge damping 0.5 | 3/3 | 3/3 | 0 | 20.9 | 21.0 | 11.2 | 10.3 | 17 | 0.12 |
| hinge damping 4 | 3/3 | 3/3 | 0 | 20.8 | 21.0 | 11.4 | 10.0 | 17 | 0.67 |
| hinge friction 0.3 N·m | 3/3 | 3/3 | 0 | 20.9 | 21.0 | 12.1 | 10.1 | 18 | 0.27 |
| hinge friction 1.0 N·m | 3/3 | 3/3 | 0 | 20.8 | 21.0 | 14.4 | 9.6 | 21 | 0.63 |
| placement ±2 mm, ±0.5° | 20/20 | 20/20 | 0 | 20.9 | 21.0 | 11.4 | 11.1 | 17 | 1.04 |
| placement ±5 mm, ±1° | 20/20 | 20/20 | 0 | 21.1 | 21.0 | 11.3 | 14.1 | 17 | 2.03 |
| placement ±10 mm, ±2° | 20/20 | 20/20 | 0 | 21.1 | 21.8 | 11.3 | 14.3 | 18 | 3.94 |
| placement ±15 mm, ±3° | 19/20 | 19/20 | 1 | 21.4 | 22.8 | 19.4 | 13.9 | 27 | 4.39 |
| door sag 2 mm | 3/3 | 3/3 | 0 | 20.9 | 21.0 | 11.2 | 10.4 | 17 | 0.13 |
| door sag 3.5 mm | 3/3 | 3/3 | 0 | 20.9 | 21.0 | 11.2 | 10.4 | 17 | 0.13 |
| squeeze 1 N/pad | 3/3 | 3/3 | 0 | 20.7 | 20.4 | 11.3 | 1.5 | 17 | 1.38 |
| squeeze 2.5 N/pad | 3/3 | 3/3 | 0 | 20.8 | 21.0 | 11.3 | 4.9 | 17 | 0.95 |
| squeeze 9 N/pad | 3/3 | 3/3 | 0 | 21.0 | 21.0 | 11.5 | 15.3 | 17 | 0.03 |
| pad friction 0.5 | 3/3 | 3/3 | 0 | 20.8 | 21.0 | 11.3 | 4.4 | 17 | 1.10 |
| pad friction 0.8 | 3/3 | 3/3 | 0 | 20.8 | 21.0 | 11.3 | 6.8 | 17 | 0.72 |
| swing 1.5× faster | 3/3 | 3/3 | 0 | 18.6 | 18.8 | 11.6 | 11.0 | 17 | 0.45 |
| swing 2× faster | 3/3 | 3/3 | 0 | 17.5 | 17.6 | 11.8 | 13.4 | 18 | 0.15 |
| combined: catch 15 N, damping 0.3, friction 0.2 N·m, ±5 mm/±1°, sag 3 mm | **20/20** | **20/20** | 0 | 21.2 | 21.0 | 16.9 | 15.3 | 27 | 2.21 |
| same, squeeze 2.5 N/pad, pad μ 0.8 | **20/20** | **20/20** | 0 | 20.8 | 20.5 | 16.9 | 5.0 | 27 | 2.64 |

**Before the fixes** (bar held at the pinch point; leaf touching the plinth; no over-push; file
`runs/door/robustness_v1_shallow_grasp.jsonl`), the same sweep measured:
- catch 22/30/54 N: 0/3 each;
- placement ±5 mm: 19/20; ±10 mm: 13/20;
- swing 2× faster: 0/3;
- door sag 2 and 5 mm: 0/3. The lowered leaf intersected the plinth and was pushed open before the robot touched
  it. MuJoCo does not filter door–printer contacts, because the printer body is welded to the world.
- Every failure was the bar slipping out past the fingertips, or the leaf rubbing the frame.

**Reading the sweep:**
- Within ±10 mm / ±2° of printer placement, the door works at 40/40 operations.
- At ±10 mm and ±15 mm, three closes needed the over-push re-push to latch. Without it (previous code), two of those
  stopped about 3° short.
- At ±15 mm / ±3°, 19/20 cycles succeed. The one miss is the only sensor/truth disagreement in the whole sweep, and
  it is a false alarm on the safe side: the jaws lost the bar at the very end of the swing, so the robot reported a
  failed open, but the door was in fact open (102°).
- No condition produced a false success, where the sensors said OK and the door was not.
- The FK door estimate inherits the placement error, because it uses the nominal hinge: about 2° at ±5 mm and 4° at
  ±10 mm. That stays inside the jaws' capture range and the planners' 15 mm clearance at the free edge.
- **Calibrate each printer's pose at installation to a few mm**, for example with the touch-off in `real/calib.py`.

### Pipeline check

`scripts/eval_unload.py --n 4` with the physical door ran 4 plates: 21/21 parts placed, doors 4/4 opened and 4/4
closed, 21.6 s per placed part. The batch baseline was also 21.6 s per part.

## Physical vs modelled

**Physical (MuJoCo contact and joint dynamics):**
- the bar in the jaws (squeeze, friction, back-drive of the coupled jaws, jaw end stops);
- the free-spinning bar;
- hinge damping and friction, the end stops, and the leaf mass and inertia;
- the arm and rail servos pulling the door.

**Modelled:**
- the magnetic catch, applied as a gap-dependent hinge torque, not a magnetic field;
- the reed switch, a threshold on the true gap with hysteresis;
- encoder noise;
- door sag, modelled as a vertical offset of the whole leaf rather than a hinge-axis tilt;
- printer placement error, a rigid offset and yaw of the whole printer.

MuJoCo's soft contacts and the `noslip` solver make friction "stickier" than real pads. The side-load numbers are
therefore an upper bound on what friction can carry, not a model of pad wear or compliance.

**Not modelled:**
- arm compliance beyond the position-servo gains;
- pad deformation;
- a scraping leaf when the sag exceeds the 4 mm clearance;
- vision of the door: the chamber camera is not used for the door.

## Run

```bash
PYTHONPATH=. .venv/bin/python scripts/test_door.py 4 --video                # one printer, video -> runs/door_p4.mp4
PYTHONPATH=. .venv/bin/python scripts/test_door.py --all                    # 10 printers -> runs/door/test_physical.json
PYTHONPATH=. .venv/bin/python scripts/test_door.py --all --coupling weld    # modelled coupler for comparison
PYTHONPATH=. .venv/bin/python scripts/door_robustness.py --procs 2          # sweep -> runs/door/robustness.jsonl
PYTHONPATH=. .venv/bin/python scripts/door_robustness.py --table            # table of an existing sweep
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_door.py
```
