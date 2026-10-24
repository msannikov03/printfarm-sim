# PrintPick `real/`: Creality printer + SO-101 hardware toolkit

Goal of the first hardware run: the print finishes, the bed cools, and then the SO-101 moves above each printed part. Grasping is a stretch goal.

The pipeline:
1. Read the printer state: printing, complete, temperatures, file name.
2. Wait until the print is done **and** the bed is cool.
3. Get the sliced G-code.
4. Extract every part from it: centre, footprint, yaw, height, and a pick order.
5. Transform bed mm into the SO-101 frame (calibration).
6. Run IK on the official MuJoCo model and hover the arm 40 mm above each part. Optionally open and close the gripper.

The arm runs in **dry-run by default**. Live mode asks `y/N` before every move.

Everything here is self-contained. It does not import `farm/ plan/ sim/ perceive/`.

```
real/
  printer/          Printer interface + backends: creality_ws (port 9999), moonraker, octoprint, marlin_serial
    __main__.py     python -m real.printer URL [--watch|--wait|--download DIR|--objects]
  discover.py       LAN scan + fingerprint; USB serial list (printer vs SO-101 adapter)
  gcode_objects.py  G-code -> parts (labels or clustering fallback) -> pick plan JSON
  so101.py          MuJoCo SO-101 model: FK, damped-least-squares IK, tool points
  calib.py          bed -> robot: touch-off (hand-guide or jog), tape-measure, ArUco stub; real/calib.json
  arm.py            SO-101 adapter, backends: LeRobot | feetech:///dev/.. (direct) | remote://pi:8765; safety, dry-run
  feetech.py        dependency-light STS3215 bus driver (pyserial): ping/read/write/sync read/sync write, retries,
                    echo-safe, health; MockSerial (simulated servos) for tests and rehearsals
  joint_calib.py    raw ticks <-> model joint angles: from-servos (LeRobot EEPROM) | from-lerobot JSON | guided
  kin_lite.py       pure-Python FK from so101_chain.json (exported from the MuJoCo model, verified to 1e-9 m)
  arm_server.py     Pi side: JSON-over-TCP server, min-jerk streaming at 50 Hz, keep-out, stall stop, watchdog, E-stop
  arm_client.py     Mac side: RemoteArm + CLI (status / jog / gripper / estop)
  hello_arm.py      first sanity check on the real arm (bus report + a small slow wave)
  replay.py         replay a planned demo log on the Pi without MuJoCo (network-down fallback)
  pi/               setup_pi.sh, make_wheelhouse.sh (offline bundle), printpick-arm.service
  DEMO_DAY.md       runbook: arrival -> printer -> arm/Pi/tailnet -> calibration -> film, troubleshooting
  reach.py          hover targets, ASCII reach map of the bed
  preview.py        offscreen renders (3/4 view + top view)
  run_demo.py       end-to-end script
  mock_printer.py   mock Creality WS / Moonraker / OctoPrint servers (print -> complete -> cooling)
  synth_gcode.py    synthetic labelled/unlabelled G-code (tests, mocks)
  assets/so101/     mujoco_menagerie robotstudio_so101 (gitignored; ./real/fetch_assets.sh re-fetches)
  tests/            pytest, no hardware needed
```

## Install

Run this on the laptop that will drive everything. It uses the project venv, Python 3.11.

```bash
cd printfarm-sim
uv pip install --python .venv/bin/python websockets pyserial      # already in the venv here
./real/fetch_assets.sh                                              # only on a fresh clone
.venv/bin/python -m pytest -q real/tests                            # 93 tests (1 skipped), ~75 s
PYTHONPATH=. .venv/bin/python -m real.run_demo --mock --dry-run --sim-preview   # full rehearsal, no hardware
./real/setup_arm_env.sh     # LIVE arm only: separate real/.venv-arm (Python 3.12 + lerobot[feetech]==0.6.1)
```
**Do not `pip install lerobot` into the shared `.venv`.**
- lerobot 0.6.1 needs Python ≥ 3.12. On the 3.11 venv, pip resolves to 0.4.4 instead.
- That install would downgrade numpy (2.4 → 2.2) and opencv (5.0 → 4.12) and pull in torch, which could break the simulation.
- Run live commands with `real/.venv-arm/bin/python`. 0.4.4 has the same `so_follower` API, so it would also work.

All commands below assume:
- you are in the repo root;
- they are prefixed with `PYTHONPATH=. .venv/bin/python` (written as `py` below).

Anything that talks to the real arm (`real.arm`, `calib touch` without `--dry-run`, `run_demo --live`) needs `PYTHONPATH=. real/.venv-arm/bin/python` instead.

### Raspberry Pi arm server (recommended: no LeRobot, no torch)
The arm plugs into a Pi on the tailnet; the Pi runs `real/arm_server.py` (Python 3.11-3.14 + pyserial only); the Mac
plans with MuJoCo and sends joint targets with `--arm remote://<pi>:8765`. Step by step: **[DEMO_DAY.md](DEMO_DAY.md)**.
```bash
bash real/pi/make_wheelhouse.sh 3.14 aarch64          # Mac: offline bundle runs/printpick_pi_bundle.tar.gz
# Pi: unpack to ~/printpick, then: bash real/pi/setup_pi.sh [--service]
real/.venv-pi/bin/python -m real.feetech --port /dev/so101 scan          # Pi: 6 x model 777
real/.venv-pi/bin/python -m real.joint_calib from-servos --port /dev/so101
real/.venv-pi/bin/python -m real.arm_server --port /dev/so101             # prints remote://100.x.y.z:8765
py -m real.hello_arm --arm remote://<pi>:8765 --go                        # Mac
py -m real.run_demo --printer creality://IP --gcode plate.gcode --live --arm remote://<pi>:8765 --film
```
The same driver also runs in-process on the Mac with the arm plugged in: `--arm feetech:///dev/tty.usbmodemXXXX`.
Rehearsal without hardware: `py -m real.arm_server --mock --listen 127.0.0.1:8765` + `--arm remote://127.0.0.1:8765 --allow-nominal`.

---

## Bring-up checklist

### 0. Before touching hardware (5 min)
- [ ] `py -m pytest -q real/tests` → `92 passed, 1 skipped`.
- [ ] `py -m real.run_demo --mock --dry-run --sim-preview` → ends with `== done: 3/3 parts hovered (dry-run)`. Then open `runs/real_preview_01_*.png`.

### 1. Identify the printer: model, firmware, root
On the touchscreen, go to **Settings → About / System info**. Write down the model, the firmware version and the IP (**Settings → Network**).

| Printer | Firmware | Backend to use | URL |
|---|---|---|---|
| K1 / K1C / K1 SE / K1 Max, Ender-3 V3 / V3 KE / V3 Plus, Hi, K2 / K2 Pro / K2 Plus | Creality OS (stock) | `creality` (websocket, port 9999) | `creality://IP` |
| K2 family (stock); possibly K1C 2025 (fw 1.0.0.22; one forum report yes, another analysis says root was removed) | stock, but also runs Moonraker on 7125 | `moonraker` works too, and can download G-code | `moonraker://IP:7125` |
| Any K1/KE **rooted** (Creality Helper Script: Moonraker + Fluidd 4408 / Mainsail 4409), Sonic Pad, any Klipper | Klipper | `moonraker` | `moonraker://IP:7125` (or `:4408`/`:4409`) |
| Old Ender-3 / V2 / S1 / V3 SE behind OctoPrint | Marlin | `octoprint` | `octoprint://IP:80?key=APIKEY` |
| Old Ender (Marlin), no network, printing from SD | Marlin | `serial` (USB) | `serial:///dev/tty.usbserial-XXXX` |

If you are unsure, step 2 tells you.

### 2. Find it on the network
```bash
py -m real.discover                       # scans this Mac's /24 on ports 80 4408 4409 7125 9999 5000 8080 + lists USB serial
py -m real.discover --host 192.168.1.57   # just the printer's IP from its screen
```
Expected output: `-> use: --printer creality://192.168.1.57` (or `moonraker://...`).
- The Mac and the printer must be on the same LAN. Guest Wi-Fi with client isolation will not work; use a phone hotspot or a travel router as the fallback.
- **USB:**
  - The SO-101 adapter usually shows as `/dev/tty.usbmodem...` (VID:PID `1a86:55d3`, CH343). BELIEVED; the source is a third-party blog.
  - A Creality Marlin board (4.2.x, STM32 behind a CH340) usually shows as `/dev/tty.usbserial-...` or `wchusbserial...` (`1a86:7523`).
  - Klipper-based Creality printers (K1/KE/K2/Hi) have no USB G-code port.
  - Never send G-code to the arm's port.
  - `--probe-serial` sends `M115` to non-arm ports only. Do not probe a printer that is mid-print: old 8-bit boards reset when the port opens.

### 3. Talk to it
```bash
py -m real.printer creality://IP            # one status line + model info
py -m real.printer creality://IP --watch    # every 5 s
py -m real.printer creality://IP --objects  # objects the firmware knows (needs "Exclude objects" in the slicer)
py -m real.printer creality://IP --download runs/real_gcode   # tries Moonraker-style file API on 7125/4408/4409
```
Expected: `[creality] printing  42.0%  bed 60.0/60C  nozzle 220.1/220C  left 12.3 min  file=/usr/data/printer_data/gcodes/x.gcode  pos=[...]`.

### 4. Slicer settings (Creality Print 5/6/7 and Orca are the same family)
- Process settings → **Others** → group **G-code output** → tick **Label objects** and **Exclude objects**.
  - Both are Advanced-mode options, so switch the advanced view on. The stock "0.20mm Standard @Creality K1C" profile already has both on.
  - Source: the CrealityPrint source, `Tab.cpp` and `PrintConfig.cpp`.
- What they emit:
  - `EXCLUDE_OBJECT_DEFINE NAME=<obj>_id_<i>_copy_<j> CENTER=.. POLYGON=..` and `; printing object <obj> id:i copy j`.
  - With Marlin flavour: `M486`.
  - The parser reads all of them, and the K1 firmware lists them.
- Leave about 20 mm between parts so the gripper fits. Keep parts in the front ~2/3 of the bed (see the reach map in step 8).
- **Always also export the sliced file** (File → Export G-code / "Export plate sliced file") to the laptop. If the printer will not serve the file, run with `--gcode that_file.gcode`. `.gcode.3mf` also works.
- Without labels it still works: extrusions of the first 3 layers are clustered into parts. Purge lines and skirts are dropped. Parts must not touch each other or share a brim.

Check what the parser sees:
```bash
py -m real.gcode_objects plate.gcode [--bed 220x220] [-o runs/plan.json]
```

### 5. SO-101: LeRobot setup (skip the steps already done)
```bash
lerobot-find-port                                             # unplug/replug to see which /dev/tty.usbmodem* is the arm
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/tty.usbmodemXXXX --robot.id=follower1
```
- Calibration files live in `~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json` in LeRobot ≥ 0.4.3.
- LeRobot ≤ 0.4.2 used `.../robots/so101_follower/`. If the arm was calibrated with an old version, pass `--calibration-dir` with that folder.

**Mirror check.** This catches unit and sign errors before anything moves:
```bash
py -m real.arm read --port /dev/tty.usbmodemXXXX --robot-id follower1 --render runs/real_mirror.png
py -m real.preview runs/real_pose_zero.png --lerobot 0,0,0,0,0,50   # what "all joints at 0 deg" looks like
```
- Put the real arm in a distinctive pose (elbow bent, wrist turned). The render at the live readings must look the same.
- LeRobot 0 deg is the middle-of-range calibration pose: upper arm vertical, forearm horizontal forward, gripper forward.
- If one joint is mirrored, run `py -m real.arm map`, then edit `real/joint_map.json` (`sign: -1` or `offset_deg`).

### 6. Place the arm
- Clamp the arm base centred on the bed, about **110 mm in front of the bed's front edge**, facing into the printer. The base plate should be at about bed height.
- With this layout (simulated), a 220 mm bed is reachable everywhere 40 mm above the bed: pointing straight down up to y ≈ 170 mm, tilted beyond that.
- Enclosed K1/K2: the door must stay open. The bed sits behind the frame, so the base will likely be lower than the bed. Measure it.
- **Bed position changes between prints:**
  - **K1/K1C:** the end macro parks the head at the right (X max−10, Y mid; on a K1C that is 219, 113.5).
    - If the bed is in the upper half of its travel, it is lowered to half travel (Z ≈ 125 mm on a K1C).
    - Prints under ~125 mm tall all end at the same bed height. Taller ones end lower.
    - Source: CrealityOfficial/K1_Series_Klipper `gcode_macro.cfg`. A real capture shows `X:219.00 Y:113.50 Z:125.06`.
  - **Ender-3 V3 KE:** the end G-code brings the bed fully forward (`G1 X2 Y218 ;Present print`).
  - **K2 Plus / Hi:** end position unknown. Look at it after a print.
  - Calibrate with the bed where it is after a print. If it moved, pass `--bed-offset-mm dx,dy,dz`. `run_demo` prints the toolhead position at READY so you can compare.

### 7. Calibrate bed → robot (`real/calib.json`)
**Method A, touch-off (recommended).** Use 4 points spread over the bed and touch them with the **fixed jaw tip**. That is the jaw that does not move when the gripper opens.
```bash
# A1 jog (torque ON): arm goes 15 mm above each point using a rough guess, you type nudges: z-5, x+1, y-0.5, then 'ok'
py -m real.calib manual --base-in-bed 110,-110,0 --yaw 90 -o runs/guess.json   # tape-measured guess (base origin in bed mm)
py -m real.calib touch --jog --guess runs/guess.json --port /dev/tty.usbmodemXXXX --robot-id follower1 \
    --points 20,20 200,20 200,200 20,200
# A2 hand-guide (torque OFF: support the arm): push the tip onto each point, press Enter
py -m real.calib touch --port /dev/tty.usbmodemXXXX --robot-id follower1 --points 20,20 200,20 200,200 20,200
```
- Mark the points with a pen dot or tape cross at known bed mm. Use the slicer's bed origin (front-left on Creality), measured from the bed edge.
- Points must be inside the reach. Use `--points` to move them inward.
- Output: per-point residuals and the RMS. **RMS < 3 mm good; > 5 mm means redo** (a bad touch, a wrong joint map, or a wrong point coordinate).
- The fit only uses the robot's own FK. The joint map (step 5) must be right first.
- To refit after fixing the joint map: `py -m real.calib fit real/calib.json`. It reuses the recorded LeRobot readings.
- To check: `py -m real.calib show` prints the bed corners in the robot frame and the **reach map**.

**Method C, tape measure (fallback, ±1–2 cm):** `py -m real.calib manual --base-in-bed X,Y,Z --yaw 90`.
- `X,Y,Z` is the robot base origin in bed mm. In the MJCF the origin is at the bottom of the base, 39 mm behind the pan (shoulder) axis, on the arm's centre line. Z is negative if the base is below the bed surface.
- `yaw` is the angle of the robot's forward axis from bed +x (90 = facing into the printer).

**Method B, ArUco (stub):** `calib.aruco_T_cam_bed()` gives the bed pose in a camera frame from markers at known bed positions. Tested only on synthetic data. You must also supply the camera pose in the robot frame.

### 8. Dry run with previews (the arm does not move)
```bash
py -m real.run_demo --printer creality://IP --sim-preview            # waits for done + cool, downloads or --gcode
py -m real.run_demo --printer creality://IP --no-wait --gcode plate.gcode --sim-preview   # skip the wait
```
Look at `runs/real_preview_*.png`:
- 3/4 view;
- top view with the arm ghosted;
- the red dot is the hover target.

`runs/real_pick_plan.json` and `runs/real_demo_log.json` hold the parts, the IK solutions, and the LeRobot joint targets.

### 9. Live
```bash
PYTHONPATH=. real/.venv-arm/bin/python -m real.run_demo --printer creality://IP --gcode plate.gcode --live \
    --port /dev/tty.usbmodemXXXX --robot-id follower1 [--gripper-demo] [--park] [--speed 15]
```
Each move is printed (steps, duration, target tip mm, LeRobot values) and asks `execute on the REAL arm? [y/N]`. Answering `n` stops the run.

## Safety
- One person holds the SO-101 power switch or USB cable during every live move. Keep hands out of the printer.
- Built into `arm.py`:
  - joint-limit check;
  - workspace box for the fixed-jaw tip (x -50…450, y ±300, z -20…400 mm robot frame);
  - every waypoint checked against **bed + 10 mm** and **part tops + 8 mm**;
  - joint-space interpolation at ≤ 20 deg/s (≤ 0.8 deg per 25 Hz command);
  - LeRobot `max_relative_target = 5 deg` as a second limiter;
  - y/N before every move.
- Not built in: collisions of the arm links with the printer frame or door. Watch the elbow.
- Motors **stay powered at exit** (`disable_torque_on_disconnect=False`), so the arm does not drop onto the bed. Power it off from a safe pose.
- Torque-off hand-guiding: the arm sags. When torque returns, goal is set to the present position first, so it does not jump.
- Live motion refuses the nominal (unmeasured) calibration unless `--allow-nominal` is passed.
- The bed must be ≤ 35 °C before unloading (`--release-c`). A cancelled or errored print raises `PrintFailed`, and nothing moves.

## Frames and units
- **Bed:** the slicer frame, in mm. Origin at the front-left corner, x right, y towards the back, z up from the bed surface.
- **Robot:** the SO-101 MJCF base frame, in m. x forward (where the arm points at q = 0), y left, z up.
  - `calib.json` stores `T_robot_bed`: `p_robot = R @ p_bed/1000 + t`.
- **Joints:** the MJCF order is the LeRobot motor order, `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`.
  - Body joints: `lerobot_deg = sign·degrees(model_rad) + offset`. The default is sign +1, offset 0 (LeRobot `use_degrees=True`).
  - Gripper: LeRobot 0..100 (0 = closed) maps linearly onto the model's −0.1745…1.745 rad.
- **Tool points** (gripper body frame):
  - `FIXED_TIP` is the fixed-jaw tip, used for calibration;
  - `hover_tcp(w)` is between the jaw pads for a part w mm wide, at the tip plane. "40 mm above the part" means the jaw tips are 40 mm above its top.
- **Hover orientation:** jaws pointing straight down if reachable, else tilted 80…30° toward the target. Jaws close across the part's short side (`yaw + 90°`).

## Expected demo output (`--mock --dry-run --sim-preview`)
```
== mock creality printer at creality://127.0.0.1:PORT?files_port=PORT (print 4 s, then cooling)
   [waiting_for_print] [creality] printing    0.0%  bed 60.0/60C ...
   [cooling] [creality] complete  100.0%  bed 58.9/0C ...
   [ready] [creality] complete  100.0%  bed 34.6/0C ...
== READY: print complete, bed 34.6 C <= 35 C; toolhead at [110.0, 110.0, 25.0] ...
== downloaded runs/real_gcode/printpick_plate.gcode
bed 220 x 220 mm (gcode:printable_area), method=exclude_object, 3 parts
  1. knob     c=( 140.0,  55.0) fp= 23.6x23.6  yaw=  44.7 h=18.0
  2. bracket  c=(  70.0,  60.0) fp= 40.0x18.0  yaw=  30.0 h=12.0
  3. cube     c=( 105.0, 110.0) fp= 20.0x20.0  yaw= -15.0 h=20.0
== no calibration at real/calib.json: using the NOMINAL placement ...
   1. knob     bed (140,55) h=18 -> robot [165, -30, 58] mm: IK ok (down, err 0.0 mm)
   ... 3/3 IK ok
== done: 3/3 parts hovered (dry-run); log runs/real_demo_log.json
```

## Troubleshooting
| Symptom | Likely cause | Fix |
|---|---|---|
| `discover` finds nothing | different subnet, client isolation, printer asleep | `--host IP` from the screen; same Wi-Fi; phone hotspot |
| `creality://` → `no data within 8 s` | not Creality OS, or fw blocks port 9999 | check port with `discover --host IP` (`/info` on :80 names the model); try `moonraker://IP:7125` |
| `complete` immediately at start | the **last** job's filename/100 % persist | expected: treated as done (the plate is still on the bed) |
| Download fails (`DownloadUnsupported`) | stock fw without a file API | `--gcode` exported file; or run without: demo falls back to firmware object list (heights from `--part-height`) |
| 0 parts / one giant part | no labels and parts touch / shared brim | enable Label + Exclude objects; or space parts ≥ 5 mm, no shared brim |
| Purge line counted as a part | unusual purge shape | label objects; `--cluster` off; check `excluded:` lines |
| Bed size wrong | no `printable_area` header (Cura) | `--bed 220x220` or `--printer-model k1max` |
| `not calibrated` from LeRobot | calibration file id/dir mismatch | same `--robot-id` as `lerobot-calibrate`; `--calibration-dir` for old LeRobot dirs |
| Mirror render ≠ real arm | joint sign/offset | `py -m real.arm map`, edit `real/joint_map.json`, recheck |
| Calibration RMS > 5 mm | bad touch / point coords / joint map | redo points; check mirror; `calib fit` after fixing map |
| Hover lands consistently offset | bed moved since calibration (Z bed, Y bed-slinger) | `--bed-offset-mm`; recalibrate in the post-print bed position |
| `UNREACHABLE` | part outside reach | move the arm closer/centred; `calib show` reach map; slice parts toward the robot |
| `SAFETY: ... hits a part` | joint-space path dips over a tall part | raise `TRANSIT_MM` in run_demo.py, reorder, or hover higher (`--hover-mm 60`) |
| Arm jerks / stalls, tracking error > 5 deg | blocked, overload, low voltage | power supply; `--speed 10`; check cables |
| Serial print died when connecting | 8-bit board auto-reset on DTR | don't connect over USB during SD prints; use OctoPrint/screen |

## VERIFIED vs UNTESTED
**VERIFIED here (pytest, no hardware; the Pi stack is in `tests/test_pi_stack.py`):**
- Feetech driver through real pyserial on a pty; packets against the manual's examples (PING `FF FF 01 02 01 FB`, READ `FF FF 01 04 02 38 02 BE`), sign-magnitude
  registers, sync read / sync write, retries under dropped and corrupted answers, adapter echo, a missing servo, EEPROM writes refused.
- Joint calibration: `from-lerobot` reproduces LeRobot 0.6.1's DEGREES / RANGE_0_100 numbers exactly (50 random raw vectors);
  `from-servos` reads the same back from the servo EEPROM; the guided procedure recovers zero + signs and rejects swapped ids.
- Pure-Python FK = MuJoCo FK (max 4e-16 m over 200 random poses); the "+ direction" words in the runbook match the model.
- Arm server: min-jerk timing within the speed/accel limits, joint-limit and bed keep-out refusals, stall stop, watchdog
  hold, E-stop + resume without a jump, token auth, one controlling client, owner disconnect mid-move -> hold.
- `run_demo --live --film --arm remote://...` end to end against the server with simulated servos; `--pick` dry run;
  `hello_arm`; `replay` of a dry-run log; `arm_server` + `arm_client` CLIs as subprocesses; the Pi modules import with
  numpy / MuJoCo / pyserial blocked; `setup_pi.sh` + `make_wheelhouse.sh` (bundle unpacked and set up on macOS; the
  venv-without-pip + pip-from-wheel fallback). NOT run on Linux/Pi yet: udev rule, dialout, systemd unit.

**Earlier toolkit tests:**
- Backends against the mocks, over real sockets: Moonraker HTTP, Creality websocket (string-valued temps, heart_beat→"ok", partial frames, subprotocol), OctoPrint REST, Marlin over a fake serial.
- The done → cooling → ready state machine with disconnects, a new print starting, an unknown bed temperature, and failures.
- G-code download, plus the firmware-objects fallback.
- G-code extraction on:
  - synthetic Orca/Creality (EXCLUDE_OBJECT + `; printing object`), comment-only, M486 and Cura `;MESH:` files;
  - an unlabelled file with a comment-less purge line and a skirt, and no layer markers;
  - G2 arcs, relative E, and `.gcode.3mf`;
  - the simulator's own plates (`runs/plates`), with parity against `jobs/gcode_prior.py`.
- Kinematics:
  - MJCF joint order equals the LeRobot motor names;
  - FK tip matches the model geometry;
  - IK round-trips;
  - hover reachability and jaw alignment on bed points.
- Calibration: Kabsch with 0.5 mm noise (RMS < 1.5 mm); touch-off and jog sessions with a simulated hand.
- The arm adapter against a fake LeRobot: config fields, action keys, degrees, step limit, decline-stops, and goal := present when torque comes back on.
- `discover` fingerprints all three mocks.
- End-to-end `run_demo` (dry run) with renders.

**UNTESTED (needs the hardware):**
- The real Creality websocket field set and state codes on *your* firmware. They follow ha_creality_ws, which is verified by its users on K1C, KE, K2 Pro/Plus, K1 and Hi.
- Whether your unit serves G-code on 7125.
- Real Creality Print output, and whether "Label/Exclude objects" sit in the same menu in your version.
- LeRobot live calls (v0.6.1 API read from source, never run here).
- The joint sign/zero correspondence on the physical arm. It is designed to match `so101_new_calib`, and LeRobot's own kinematics uses `deg2rad(lerobot_deg)` with that URDF. The **mirror check** in step 5 is mandatory.
- Hand-guiding stiffness, and IK accuracy versus the real arm (link tolerances, backlash). Expect a few mm.
- The Marlin serial path on a real board, and the OctoPrint API key flow.
- The ArUco method (synthetic only, stub).
- USB VID:PID hints (BELIEVED).

## Sources (checked October 2026)
- **Creality OS websocket.** Port 9999, `heart_beat`→`"ok"`, `ReqPrinterPara`, `reqPrintObjects`, `reqGcodeFile`→`retGcodeFileInfo2`, the field names, the state table (progress ≥ 100 = complete; 5 paused; 4 stopped; 1/0/7 busy), the stale `err.errcode`, the persisting last filename, and the `wsslicer` subprotocol. All from ha_creality_ws `ws_client.py`, `utils.py` and `const.py` @ a8e5491: https://github.com/3dg1luk43/ha_creality_ws
- Also on port 9999: https://pypi.org/project/creality-k1-api/
- Creality's own code, in two places:
  - CrealityPrint v5 (`release-v5.1.7`, `kernel/BasicKernel/external/localnetworkinterface/`): the state enum 0–5, the request keys, the `set` commands, and both `reqGcodeFile` reply shapes. https://github.com/CrealityOfficial/CrealityPrint
  - The OrcaSlicer Creality uploader (ws path `/`, text frames, `/upload/<name>`): https://github.com/OrcaSlicer/OrcaSlicer/blob/main/src/slic3r/Utils/CrealityPrint.cpp
- K1C end-of-print macro: https://github.com/CrealityOfficial/K1_Series_Klipper (`config/K1C_CR4CU220812S12/gcode_macro.cfg`)
- K1C 2025 root removed (conflicts with the forum report below): https://github.com/ahanel13/creality-k1c-2025-analysis
- Stock K1C 2025 (fw 1.0.0.22): port 80 returns 404, the Moonraker-compatible API on 7125 works, and 9999 is the status stream. https://forum.creality.com/t/k1c-2025-orcaslicer-workarounds-for-missing-web-interface/47164
- K2 Plus ports: 7125 Moonraker, 9999 websocket, 80 upload, 8000 WebRTC. https://github.com/pdromnt/k2dash
- Label/Exclude objects in Creality Print, and K1 exclude-object firmware support (1.3.3.8):
  - https://forum.creality.com/t/object-deletion/39726
  - https://forum.creality.com/t/help-new-creality-print-version-missing-exclude-object/47040
  - https://forum.creality.com/t/creality-print-7-exclude-objects/48557
  - https://docs.mainsail.xyz/overview/features/exclude-objects
- Moonraker API: https://moonraker.readthedocs.io/en/latest/external_api/printer/ and `.../external_api/file_manager/`
  - Klipper upper-cases `exclude_object` names.
- OctoPrint REST: https://docs.octoprint.org/en/master/api/job.html, `api/printer.html`, `api/files.html`
- Marlin: https://marlinfw.org/docs/gcode/M027.html, M105, M115, M031, https://marlinfw.org/docs/gcode/M486.html
- **LeRobot v0.6.1** (2026-08-03): https://github.com/huggingface/lerobot/tree/v0.6.1/src/lerobot
  - `robots/so_follower/config_so_follower.py`: `SO101FollowerConfig = SOFollowerRobotConfig`, `use_degrees=True` default, `max_relative_target`, `disable_torque_on_disconnect`.
  - `robots/so_follower/so_follower.py`: `SO101Follower = SOFollower`, `name="so_follower"`, motors 1–6, gripper `RANGE_0_100`, `send_action({"<motor>.pos": v})`.
  - `motors/motors_bus.py`: DEGREES = `(raw - (min+max)/2)·360/4095`.
  - `model/kinematics.py`: `deg2rad(joint_pos_deg)` into the URDF.
  - `robots/so_follower/so101.md`: CLI, `/dev/tty.usbmodem…`.
  - LeRobot ≤ 0.4.2 used `robots/so101_follower`.
  - `requires-python >= 3.12` from 0.5.x on, which is why `setup_arm_env.sh` exists.
  - `configure()` sets the servo acceleration to the maximum (254) and no goal speed, so un-interpolated goals snap. Our adapter interpolates.
- SO-101 sim calibration convention ("new calib" = zero at the middle of the range; gripper 0 closed / 100 open): https://github.com/TheRobotStudio/SO-ARM100/tree/main/Simulation/SO101
- MuJoCo model: https://github.com/google-deepmind/mujoco_menagerie/tree/main/robotstudio_so101 @ cda5236 (2026-10-19)
