# Demo day: Creality printer + SO-101 on a Raspberry Pi, filmed in 30-45 minutes

The shot: the printer reports **done**, the bed cools, the arm moves to each printed part (jaws turned to the part,
40 mm above it), opens and closes the gripper, optionally drops one part into a box, and goes back to rest.

```
   Mac (planning, MuJoCo IK, previews)  --tailnet-->  Raspberry Pi (arm server)  --USB-->  SO-101 bus adapter
        |                                                 python3 -m real.arm_server        6 x STS3215, 1 Mbps
        +--LAN / hotspot-->  Creality printer (status websocket :9999 / Moonraker :7125)
```
The Pi needs only Python 3.11-3.14 + pyserial (no LeRobot, no torch, no MuJoCo). The Mac uses the project venv.
Commands below: `py` = `PYTHONPATH=. .venv/bin/python` on the Mac (repo root); `vpy` = `real/.venv-pi/bin/python`
on the Pi (in `~/printpick`). `<pi>` = the Pi's Tailscale IP (`tailscale ip -4` on the Pi) or MagicDNS name.

---

## The evening before (Mac, 20 min)
```bash
py -m pytest -q real/tests                                      # all pass, no hardware
py -m real.arm_server --mock --listen 127.0.0.1:8765 &          # simulated servos behind the real server
py -m real.run_demo --mock --live --arm remote://127.0.0.1:8765 --film --allow-nominal --sim-preview
kill %1
bash real/pi/make_wheelhouse.sh 3.14 aarch64     # Pi Python version! -> runs/printpick_pi_bundle.tar.gz (17 MB;
                                                 # NO_NUMPY=1 in front: 1.8 MB, numpy is optional)
```
- Copy the bundle to the Pi now if it is reachable (`scp runs/printpick_pi_bundle.tar.gz <pi>:`), **and** onto a USB stick.
  On the Pi: `mkdir -p ~/printpick && tar xzf printpick_pi_bundle.tar.gz -C ~/printpick && cd ~/printpick && bash real/pi/setup_pi.sh`
  (it ends with a mock self-test; `--service` also installs a start-at-boot service).
- Slice the demo plate (below) and export the `.gcode` to the Mac. Put it on a USB stick for the printer too.

**Demo plate (Creality Print 5-7 or OrcaSlicer).** Three slicer primitives, so their sizes are known
(checked: with the nominal arm placement all three hover poses and a pick of the first part into `--box-mm=-60,40,90`
solve in a dry run):

| part | size (mm) | centre on the bed (x, y) |
|---|---|---|
| cube | 20 x 20 x 15 | (110, 100) |
| cylinder | d 20 x 15 | (60, 60) |
| bar | 40 x 18 x 10, turned 30 deg | (160, 60) |
- Front two thirds of the bed (the arm stands in front of it), at least 25 mm between parts (jaws + fingers).
- Process -> Others -> G-code output: **Label objects** and **Exclude objects** on (advanced view). No brim, skirt
  distance >= 5 mm (or no skirt), so no extrusion connects two parts.
- Fast and stiff: 0.28 mm layers, 2 walls, 15 % infill. Bed 55 C for PLA (it cools to 35 C sooner).
- Check on the Mac before printing: `py -m real.gcode_objects plate.gcode` -> 3 parts with these centres and heights;
  `py -m real.run_demo --gcode plate.gcode --no-wait --sim-preview` -> 3/3 IK ok.
- Print time and cool-down time depend on the printer; time both on the day (status line shows `left N min`).
- Pack: Pi + its PSU (Pi 5: 27 W USB-C), the arm's own servo PSU, USB cable adapter-to-Pi, Ethernet cable,
  2 clamps, tape, a fine marker, a small box for the picked part, phone tripod. Phone hotspot ready.

## T+0 Arrive: printer first (the print is the longest step)
1. On the touchscreen: **Settings -> About / System**: model + firmware; **Network**: IP. Join the Mac to the same network.
2. `py -m real.discover --host <printer-ip>` -> prints the URL to use (`creality://IP`, `moonraker://IP:7125`, ...).
3. `py -m real.printer <url>` -> one status line (`[creality] standby ... bed 24/0C`). If nothing answers: table below.
4. Start the demo plate (from the slicer "send to printer", the printer's USB stick, or its file list).

| Printer (firmware) | Status API | G-code download | Use |
|---|---|---|---|
| K1 / K1C / K1 SE / K1 Max, Ender-3 V3 / V3 KE / V3 Plus, Hi, CR-10 SE (Creality OS, stock) | websocket :9999 | usually no (some K1C 2025 fw expose Moonraker :7125) | `creality://IP` + `--gcode exported.gcode` |
| K2 / K2 Pro / K2 Plus (stock) | websocket :9999 + Moonraker :7125 | yes (:7125 file API) | `moonraker://IP:7125` |
| Any K1/KE **rooted** (helper script: Moonraker + Fluidd :4408 / Mainsail :4409), Sonic Pad, any Klipper | Moonraker :7125 | yes | `moonraker://IP:7125` |
| Ender-3 / V2 / S1 / V3 SE / Ender-5 (Marlin) behind OctoPrint | OctoPrint REST :80 | yes | `octoprint://IP?key=APIKEY` |
| Same, no network (SD card printing) | USB serial (M27/M105) | no | `serial:///dev/tty.usbserial-*` (never during an SD print on 8-bit boards: DTR reset) |

Sources and caveats are in [README.md](README.md) (Sources); stock behaviour differs between firmware versions,
so trust `real.discover` over this table.

## T+5 Arm + Pi + tailnet (while it prints)
1. **Power and cables.** Servo PSU into the bus adapter board (USB alone does NOT power the servos), adapter USB into
   the Pi, Pi PSU. Clamp the arm base ~110 mm in front of the bed edge, centred, facing the bed, base about at bed
   height (README step 6). Keep the arm in its folded rest pose.
2. **Tailnet.** Mac: Tailscale app connected; `tailscale status`; `tailscale ping <pi>` (a `via DERP` path is fine: the
   Pi does the 50 Hz control loop itself, the Mac only sends targets). `ssh <user>@<pi>`.
3. **Bus check (Pi).**
   ```bash
   cd ~/printpick
   vpy -m real.feetech --port /dev/so101 scan      # -> id 1..6 model 777 (STS3215)   (/dev/ttyACM0 if no udev rule)
   vpy -m real.feetech --port /dev/so101 health    # volts (~5-7.4 V or ~12 V, depends on the kit), temps, flags
   ```
4. **Joint calibration (Pi), once.** If the arm was ever calibrated with LeRobot, its calibration is in the servos:
   ```bash
   vpy -m real.joint_calib from-servos --port /dev/so101        # -> real/joint_calib.json
   ```
   Otherwise (it says "not calibrated with LeRobot"): `vpy -m real.joint_calib guided --port /dev/so101`, 8 prompts,
   torque off, hold the arm: ZERO pose = upper arm vertical, forearm horizontal forward, wrist straight, fixed jaw at
   the bottom, gripper closed; then each joint a little in its + direction (pan: to the arm's right; lift: forward;
   elbow: down; wrist: down; roll: top of the gripper to the left seen from behind); then gripper closed / open.
   It refuses swapped motor ids. **Check:** `vpy -m real.joint_calib live --port /dev/so101`, put the arm in the ZERO pose:
   every angle within ~5 deg of 0; lift the forearm: elbow_flex goes negative.
5. **Hello (Pi):** `vpy -m real.hello_arm --port /dev/so101` (bus report), then `... --go` (small slow wave).
6. **Server (Pi):** `vpy -m real.arm_server --port /dev/so101` (or `systemctl --user start printpick-arm`).
   It prints `--arm remote://100.x.y.z:8765`. Leave it running.
7. **Mac:** `py -m real.arm_client remote://<pi>:8765 status` and `py -m real.hello_arm --arm remote://<pi>:8765 --go`.
   **Mirror check:** `py -m real.arm read --arm remote://<pi>:8765 --render runs/real_mirror.png`: the render must look like the arm.

## T+20 Print done -> bed cooling -> bed calibration (uses the cooling time)
```bash
py -m real.printer <url> --wait            # returns when complete AND the bed <= 35 C (Ctrl-C when you are ready)
```
Calibrate with the bed **where it is after the print** (K1-family beds stay low; bed-slingers park Y). Touch 4 marked
points away from the parts with the fixed jaw tip (pen dots at known slicer coordinates):
```bash
py -m real.calib manual --base-in-bed 110,-110,0 --yaw 90 -o runs/guess.json      # tape-measured guess
py -m real.calib touch --jog --guess runs/guess.json --arm remote://<pi>:8765 --points 20,20 200,20 200,200 20,200
py -m real.calib show                       # RMS < 3 mm good, > 5 mm redo; reach map
```
Jog with `z-5`, `x+1`, `y-0.5`, then `ok`. (If the bed is below the arm base the server's floor stops the tip: restart
it with `--floor-mm -150`.)

## T+30 Dry run with previews, then live
```bash
py -m real.run_demo --printer <url> --gcode plate.gcode --no-wait --sim-preview          # nothing moves
open runs/real_preview_*.png                                                              # red dot = hover target
py -m real.run_demo --printer <url> --gcode plate.gcode --no-wait --live --arm remote://<pi>:8765   # y/N per move
```
Optional pick (one part into the box; drop point in bed mm, use `=` because of the minus sign):
`... --live --arm remote://<pi>:8765 --pick first --box-mm=-60,40,90`

## T+35 Film
```bash
py -m real.run_demo --printer <url> --gcode plate.gcode --live --arm remote://<pi>:8765 --film [--pick first --box-mm=...]
```
One `go` prompt, then: narration lines with timestamps, 15 deg/s minimum-jerk moves, ready pose, per part
transit -> hover (jaws turned to the part) -> open -> 2.5 s -> close -> retreat, back to the start pose.
Without `--no-wait` it first waits for **done + cool**, so you can start it while the print finishes.
**Ctrl-C** = stop and hold. **E-stop** from a second terminal: `py -m real.arm_client remote://<pi>:8765 estop`
(torque off: the arm drops; resume with `... torque-on`). Pulling the Mac's network also stops it: the server
watchdog holds the pose after 2 s without a message.

**Shot list (phone at 60 fps, exposure locked, tripod at bed height, 45 deg)**
1. Wide: printer + arm + laptop; the print's last layer; screen shows 100 %.
2. Terminal close-up: `READY: print complete, bed 34.6 C` and the parts list.
3. Arm rises to the ready pose; first hover: hold on the jaws turning to the part's angle.
4. Gripper open/close over each part (the console narration in the corner of the frame).
5. Side by side: `runs/real_preview_0N_*.png` (simulation) next to the real arm at the same part.
6. Optional pick: close-up of the jaws closing, the part dropping into the box.
7. End: the arm folds back to rest; printer screen idle.

## Troubleshooting
| Symptom | Cause | Fix |
|---|---|---|
| `scan`: nothing answers | servo PSU off; wrong port; wrong baud | power the adapter board; `ls /dev/ttyACM* /dev/ttyUSB*`; `scan` tries all bauds and prints the right `--baud` |
| `scan`: some ids missing | loose 3-pin cable after the last id found; a servo still at factory id 1 | reseat the daisy chain; set ids once with `lerobot-setup-motors` on a laptop |
| `Permission denied: /dev/ttyACM0` | not in `dialout` | `sudo usermod -aG dialout $USER` + log in again, or rerun `setup_pi.sh` (udev rule) |
| garbage / timeouts right after plugging in | ModemManager probing the port | udev rule from `setup_pi.sh` (ID_MM_DEVICE_IGNORE) or `sudo systemctl stop ModemManager` |
| `servo id N reports overload / overheat` | blocked joint, gripper squeezing too long | `estop`, free it, wait; `health` shows temperature |
| move `aborted: stall/collision` | arm hit something / too heavy | free it; it holds where it stopped; replan |
| move refused `keep-out ... above the bed` | target too close to the bed (8 mm margin) | check the bed calibration (`calib show`), hover higher |
| move refused `target outside joint limits` | IK at a joint limit or a bad joint calibration | `joint_calib live` check; re-place the arm |
| `cannot reach the arm server` | server not running / tailnet down | `tailscale ping <pi>`; on the Pi `systemctl --user status printpick-arm` |
| tailnet down, same LAN | Tailscale blocked by the lab network | Pi: `vpy -m real.arm_server --port /dev/so101 --listen 0.0.0.0:8765 --token SECRET`; Mac: `--arm remote://<pi-lan-ip>:8765?token=SECRET` (or both on the phone hotspot) |
| no common network at all | | Ethernet cable Mac <-> Pi (link-local: `ssh <user>@<pi-hostname>.local`), same as above with the cable IP; or **replay on the Pi**: plan on the Mac (dry run), copy `runs/real_demo_log.json` by USB stick, `vpy -m real.replay real_demo_log.json --port /dev/so101 --film --go` |
| Pi unusable | | plug the arm's USB into the Mac: `--arm feetech:///dev/tty.usbmodemXXXX` (same driver, in-process) |
| printer API unreachable | other subnet / client isolation / firmware | same Wi-Fi or hotspot; `discover --host IP`; skip it: `--gcode exported.gcode --no-wait` and watch the screen |
| `DownloadUnsupported` | stock firmware without a file API | `--gcode exported.gcode` (always export it from the slicer) |
| hover consistently offset | the bed moved since calibration | `--bed-offset-mm dx,dy,dz` or re-touch in the post-print bed position |
| arm sags when the server stops | normal: torque stays on (holds) until `torque-off`/power off | support it before `torque-off` |

## Safety summary
Server side (Pi): minimum-jerk profiles capped at 30 deg/s and 60 deg/s^2 (film: 15 / 30), servo speed and
acceleration registers as a hardware cap, gripper torque 50 %, joint limits from the MuJoCo model and the
calibration (refused, not clamped), keep-out of both jaw tips + wrist + elbow (workspace box, floor, bed with
8 mm margin), stall stop at 15 deg tracking error, 2 s heartbeat watchdog, E-stop from any client, one
controlling client. Mac side: the same path is checked against the bed and every part before it is sent, and
nothing moves without `y` (or the single `go` of `--film`). One person keeps a hand on the servo PSU switch.
