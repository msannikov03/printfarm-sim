"""PrintPick hardware demo: printer done + cool -> G-code -> parts -> SO-101 hovers over each part.

Dry run with everything mocked (no hardware):
    PYTHONPATH=. .venv/bin/python -m real.run_demo --mock --dry-run --sim-preview
Real printer, arm still in dry-run (check the plan and the previews first):
    PYTHONPATH=. .venv/bin/python -m real.run_demo --printer creality://192.168.1.57 --sim-preview
Real arm (asks y/N before every motion):
    PYTHONPATH=. .venv/bin/python -m real.run_demo --printer creality://192.168.1.57 --gcode plate.gcode \
        --live --port /dev/tty.usbmodemXXXX --robot-id follower1 [--gripper-demo]
Real arm on the Raspberry Pi arm server (real/arm_server.py), film mode (one confirmation, slow smooth
moves, narration, gripper open/close at each part, back to the start pose):
    PYTHONPATH=. .venv/bin/python -m real.run_demo --printer creality://192.168.1.57 --gcode plate.gcode \
        --live --arm remote://<pi-tailscale-ip>:8765 --film
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np

from real import ROOT, RUNS
from real.arm import Arm, JointMap, Safety, SafetyError
from real.calib import CALIB_JSON, BedCalib, nominal
from real.gcode_objects import KNOWN_BEDS, ROBOT_SIDES, extract, plan_from_firmware_objects, print_plan
from real.printer import DownloadUnsupported, PrinterError
from real.so101 import FIXED_TIP, SO101, hover_tcp, pinch

log = logging.getLogger("real.demo")
TRANSIT_MM = 60.0          # clearance above the tallest part for moves between parts
HOVER_MIN_MM = 20.0        # a hover out of reach is retried this low, in HOVER_STEP_MM steps: a tall part near the
HOVER_STEP_MM = 5.0        # base (the purge tab on the front strip, docs/purge.md) is reachable only a little lower
READY_TIP_M = (0.15, 0.0, 0.20)   # 'ready' pose: jaws pointing down, 15 cm in front of the base, 20 cm up


def ready_q(kin: SO101) -> np.ndarray:
    r = kin.solve_hover(np.array(READY_TIP_M), tcp=FIXED_TIP)
    return r.q


def parts_clearance_check(kin: SO101, calib: BedCalib, objects, margin_mm=8.0, radius_mm=25.0, bed_margin_mm=10.0):
    """Waypoint check: over the bed (its footprint + 20 mm) the fixed-jaw tip must stay bed_margin above the bed
    surface, and margin above any part whose centre is within (part half-diagonal + radius) in XY."""
    bw, bh = calib.bed_mm
    parts = [(np.array(o["center_mm"]), 0.5 * math.hypot(*o["footprint_mm"]), o.get("height_mm") or 0.0)
             for o in objects]

    def check(q, what):
        p, _ = kin.fk(q, FIXED_TIP)
        b = calib.robot_to_bed(p)
        over_bed = -20 <= b[0] <= bw + 20 and -20 <= b[1] <= bh + 20
        if over_bed and b[2] < bed_margin_mm:
            raise SafetyError(f"{what}: tip {b[2]:.0f} mm above the bed (< {bed_margin_mm:.0f} mm)")
        for c, r, h in parts:
            if np.hypot(*(b[:2] - c)) < r + radius_mm and b[2] < h + margin_mm:
                raise SafetyError(f"{what}: tip at bed ({b[0]:.0f},{b[1]:.0f},{b[2]:.0f}) mm hits a {h:.0f} mm part")
    return check


def hover_plan(kin: SO101, calib: BedCalib, plan: dict, hover_mm: float, q_start, align_jaws=True):
    """For each object in pick order: transit pose (above the tallest part) + hover pose. IK only. A hover out of
    reach is retried lower (HOVER_STEP_MM steps, down to HOVER_MIN_MM); the step's "hover_mm" is the height used."""
    by = {o["name"]: o for o in plan["objects"]}
    hmax = max([o.get("height_mm") or 0.0 for o in plan["objects"]] + [0.0])
    steps, q = [], np.array(q_start, float)
    for name in plan["pick_order"]:
        o = by[name]
        width = min(o["footprint_mm"][1], 60.0)
        tcp = hover_tcp(width)
        th = math.radians(o["yaw_deg"] + 90.0)
        jaw = calib.bed_dir_to_robot([math.cos(th), math.sin(th)]) if align_jaws else None
        h = o.get("height_mm") or 0.0
        grip = kin.gripper_for_opening(min(width + 20.0, 90.0))      # jaws open: part width + 20 mm
        tries = [hover_mm] + list(np.arange(hover_mm - HOVER_STEP_MM, HOVER_MIN_MM - 1e-6, -HOVER_STEP_MM))
        first = None
        for hv in tries:
            hover = calib.bed_to_robot([*o["center_mm"], h + hv])
            transit = calib.bed_to_robot([*o["center_mm"], max(hmax + TRANSIT_MM, h + hv)])
            rt = kin.solve_hover(transit, jaw=jaw, tcp=tcp, q0=q)
            rh = kin.solve_hover(hover, jaw=jaw, tcp=tcp, q0=rt.q if rt.ok else q)
            first = first or (hv, hover, transit, rt, rh)
            if rh.ok:
                break
        else:                          # out of reach at every height: report it at the requested one
            hv, hover, transit, rt, rh = first
        if rh.ok and rh.pitch_deg is not None and abs((rt.pitch_deg or 0) - rh.pitch_deg) > 1e-6:
            # keep the transit pose at the same tilt as the hover so the final descent is straight
            rt2 = kin.solve_hover(transit, jaw=jaw, tcp=tcp, q0=rh.q, pitches=(rh.pitch_deg,))
            rt = rt2 if rt2.ok else rt
        rt.q[5] = rh.q[5] = grip
        steps.append(dict(name=name, obj=o, hover_robot=hover, transit_robot=transit, tcp=tcp, ik_transit=rt,
                          ik_hover=rh, hover_mm=float(hv)))
        if rh.ok:
            q = rh.q
    return steps


def move_seconds(q0, q1, speed, accel, gripper_speed=90.0, settle=0.3):
    """Duration of the arm server's min-jerk move (real/arm_server.py ArmController.plan) + settling."""
    d = np.degrees(np.abs(np.asarray(q1, float) - np.asarray(q0, float)))
    return max(0.4, 1.875 * d[:5].max() / speed, math.sqrt(5.7735 * d[:5].max() / accel),
               1.875 * d[5] / gripper_speed) + settle


def film_seconds(q_start, q_ready, steps, speed, accel, dwell_s, gripper) -> float:
    """Expected length of a --film run (ready -> per part transit/hover/open/dwell/close/retreat -> start)."""
    t, q = 0.0, np.asarray(q_start, float)
    for s in steps:
        if not s["ik_hover"].ok:
            continue
        for target in ([s["ik_transit"].q] if s["ik_transit"].ok else []) + [s["ik_hover"].q]:
            t += move_seconds(q, target, speed, accel)
            q = np.asarray(target, float)
        if gripper:
            t += 2 * move_seconds(q, np.r_[q[:5], q[5] + 0.9], speed, accel)
        t += dwell_s
        if s["ik_transit"].ok:
            t += move_seconds(q, s["ik_transit"].q, speed, accel)
            q = np.asarray(s["ik_transit"].q, float)
    t += move_seconds(q_start, q_ready, speed, accel) * 2 + move_seconds(q, q_ready, speed, accel)
    return t


class Narrator:
    """Console narration for filming: big step lines with the time since the start."""

    def __init__(self, on=True):
        self.on, self.t0 = on, time.monotonic()

    def __call__(self, msg):
        if self.on:
            print(f"\n[{time.monotonic() - self.t0:6.1f} s] >>> {msg}", flush=True)


def grasp_plan(kin: SO101, calib: BedCalib, o: dict, q_ref, approach_mm=40.0):
    """Pinch the part's upper half from above, jaws across its short side. Returns (pre, grasp) IK results
    and the grip width (mm). The jaw tips end max(2, h - 14) mm above the bed."""
    width = min(o["footprint_mm"][1], 60.0)
    tcp = pinch(width, depth_mm=8.0)
    th = math.radians(o["yaw_deg"] + 90.0)
    jaw = calib.bed_dir_to_robot([math.cos(th), math.sin(th)])
    h = o.get("height_mm") or 10.0
    z_pinch = max(2.0, h - 14.0) + 8.0
    tgt = calib.bed_to_robot([*o["center_mm"], z_pinch])
    pre = kin.solve_hover(calib.bed_to_robot([*o["center_mm"], z_pinch + approach_mm]), jaw=jaw, tcp=tcp, q0=q_ref)
    g = kin.solve_hover(tgt, jaw=jaw, tcp=tcp, q0=pre.q if pre.ok else q_ref,
                        pitches=(pre.pitch_deg,) if pre.ok and pre.pitch_deg else (90, 80, 70, 60, 50))
    return pre, g, width


def main(argv=None):
    ap = argparse.ArgumentParser(description="PrintPick real demo: wait for print, parse G-code, hover the SO-101")
    src = ap.add_argument_group("printer")
    src.add_argument("--printer", help="creality://IP | moonraker://IP:7125 | octoprint://IP?key=K | serial:///dev/..")
    src.add_argument("--mock", action="store_true", help="start a local mock printer (no hardware)")
    src.add_argument("--mock-kind", default="creality", choices=["creality", "moonraker", "octoprint"])
    src.add_argument("--mock-style", default="orca", help="mock G-code labels: orca|orca_comments|m486|cura|none")
    src.add_argument("--mock-print-s", type=float, default=4.0)
    src.add_argument("--mock-tau-s", type=float, default=2.0)
    src.add_argument("--no-wait", action="store_true", help="do not wait for done + cool (e.g. bed already empty test)")
    src.add_argument("--release-c", type=float, default=35.0, help="bed temperature at which parts may be removed")
    src.add_argument("--poll-s", type=float, default=5.0)
    g = ap.add_argument_group("plate")
    g.add_argument("--gcode", help="sliced file (.gcode or .gcode.3mf); default: download from the printer")
    g.add_argument("--bed", help="bed WxH mm if the G-code has no printable_area header, e.g. 220x220")
    g.add_argument("--printer-model", help="bed size fallback by model name (k1, k1max, ender3v3ke, ...)")
    g.add_argument("--robot-side", default="front", choices=ROBOT_SIDES)
    g.add_argument("--cluster", action="store_true", help="ignore object labels; cluster extrusions instead")
    g.add_argument("--part-height", type=float, default=25.0,
                   help="assumed part height (mm) when only firmware objects are available (no G-code)")
    r = ap.add_argument_group("robot")
    r.add_argument("--calib", default=str(CALIB_JSON), help="bed->robot calibration (python -m real.calib ...)")
    r.add_argument("--allow-nominal", action="store_true", help="live motion with the nominal (unmeasured) placement")
    r.add_argument("--bed-offset-mm", help="dx,dy,dz: bed moved since calibration (Z-bed printers end at another "
                   "height; bed-slingers park the bed in Y), e.g. 0,0,-12")
    r.add_argument("--hover-mm", type=float, default=40.0, help="jaw tips this far above each part's top")
    r.add_argument("--no-align-jaws", action="store_true", help="do not rotate the wrist to the part yaw")
    r.add_argument("--live", action="store_true", help="drive the real SO-101 (default: dry run)")
    r.add_argument("--dry-run", action="store_true", help="explicit dry run (default unless --live)")
    r.add_argument("--port", help="SO-101 serial port (lerobot-find-port), LeRobot backend")
    r.add_argument("--arm", help="arm backend URL instead of --port: remote://<pi>:8765 (Pi arm server), "
                   "feetech:///dev/ttyACM0 (direct driver here), feetech://mock (rehearsal)")
    r.add_argument("--robot-id", default="follower", help="the --robot.id used with lerobot-calibrate")
    r.add_argument("--calibration-dir", help="LeRobot calibration dir if not the default")
    r.add_argument("--yes", action="store_true", help="do not ask y/N before each live motion (careful)")
    r.add_argument("--speed", type=float, help="max joint speed deg/s (default 20; 15 with --film)")
    r.add_argument("--accel", type=float, help="max joint acceleration deg/s^2, feetech/remote (default 40; 30 with --film)")
    r.add_argument("--film", action="store_true", help="filming: one confirmation for the whole run, slow "
                   "(15 deg/s), narration, gripper open/close at each part, dwell 2.5 s, back to the start pose")
    r.add_argument("--pick", help="also pick ONE part (its name, or 'first') and drop it at --box-mm")
    r.add_argument("--box-mm", help="drop point x,y,z in BED mm (e.g. 110,-60,80: in front of the bed, 8 cm up)")
    r.add_argument("--gripper-demo", action="store_true", help="open and close the gripper at each hover")
    r.add_argument("--dwell-s", type=float, help="pause over each part, s (default 1.5; 2.5 with --film)")
    r.add_argument("--park", action="store_true", help="end in the 'ready' pose (tips 20 cm up, 15 cm ahead)")
    o = ap.add_argument_group("output")
    o.add_argument("--sim-preview", action="store_true", help="render runs/real_preview_*.png")
    o.add_argument("--runs-dir", default=str(RUNS), help="downloads, pick plan, previews and the log go here")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if a.live and a.dry_run:
        ap.error("--live and --dry-run are exclusive")
    if a.live and not (a.port or a.arm):
        ap.error("--live needs --arm remote://<pi>:8765 (Pi server) or --port (LeRobot; lerobot-find-port)")
    if a.pick and not a.box_mm:
        ap.error("--pick needs --box-mm x,y,z (drop point in bed mm)")
    a.speed = a.speed or (15.0 if a.film else 20.0)
    a.accel = a.accel or (30.0 if a.film else 40.0)
    a.dwell_s = a.dwell_s if a.dwell_s is not None else (2.5 if a.film else 1.5)
    if a.film:
        a.gripper_demo, a.park = True, True
    say = Narrator(on=a.film or a.live)
    runs = Path(a.runs_dir)
    runs.mkdir(parents=True, exist_ok=True)
    out = runs / "real_demo_log.json"
    report = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(a)}

    # ------------------------------------------------------------------ 1. printer
    mock = printer = None
    if a.mock:
        from real.mock_printer import MockPrinter
        mock = MockPrinter(a.mock_kind, print_s=a.mock_print_s, tau_s=a.mock_tau_s, style=a.mock_style)
        a.printer = mock.url
        a.poll_s = min(a.poll_s, 0.5)
        print(f"== mock {a.mock_kind} printer at {mock.url} (print {a.mock_print_s:.0f} s, then cooling)")
    try:
        if a.printer:
            from real.printer import make_printer
            printer = make_printer(a.printer)
            st = printer.status()
            print("== printer:", st.short())
            if not a.no_wait:
                last = [None, 0.0]

                def on_update(phase, s):
                    now = time.monotonic()
                    if phase != last[0] or now - last[1] > 30:
                        print(f"   [{phase}] " + (s.short() if s else "(no data)"))
                        last[:] = [phase, now]
                st = printer.wait_until_done_and_cool(release_c=a.release_c, poll_s=a.poll_s, on_update=on_update)
                print(f"== READY: print complete, bed {st.bed_c:.1f} C <= {a.release_c:.0f} C"
                      + (f"; toolhead at {st.position} (compare with calibration time)" if st.position else ""))
            report["printer_status"] = st.to_dict()

        # -------------------------------------------------------------- 2. G-code
        gpath, plan = a.gcode, None
        bed = tuple(float(v) for v in a.bed.lower().split("x")) if a.bed else None
        if not gpath:
            if printer is None:
                ap.error("need --gcode PATH or a printer (--printer / --mock) to download from")
            (runs / "real_gcode").mkdir(exist_ok=True)
            try:
                gpath = printer.download_gcode(runs / "real_gcode")
                print("== downloaded", gpath)
            except (DownloadUnsupported, PrinterError) as e:
                print(f"== G-code download failed:\n{e}")
                objs = printer.objects()
                if not objs:
                    sys.exit("no G-code and no firmware object list: rerun with --gcode PATH")
                fb = bed or KNOWN_BEDS.get((a.printer_model or "").lower().replace("-", "").replace(" ", ""),
                                           (220.0, 220.0))
                plan = plan_from_firmware_objects(objs, fb, a.part_height, a.robot_side)
                print(f"== using {len(objs)} objects reported by the firmware (heights ASSUMED {a.part_height} mm)")
        if plan is None:
            plan = extract(gpath, bed=bed, printer_model=a.printer_model, robot_side=a.robot_side,
                           force_cluster=a.cluster)
            print("== parts from", gpath)
        print_plan(plan)
        (runs / "real_pick_plan.json").write_text(json.dumps(plan, indent=2))
        report["plan"] = plan
        if not plan["objects"]:
            print("no parts found in the G-code; nothing to do")
            return report
    finally:
        if printer is not None:
            printer.close()
        if mock is not None:
            mock.stop()

    # ------------------------------------------------------------------ 3. calibration
    if Path(a.calib).exists():
        calib = BedCalib.load(a.calib)
        print(f"== calibration {a.calib}: method={calib.method} rms={calib.rms_mm} mm")
    else:
        calib = nominal(plan["bed_mm"])
        print(f"== no calibration at {a.calib}: using the NOMINAL placement (arm base centred, 110 mm in front "
              f"of the bed's front edge, at bed height). Run `python -m real.calib touch ...` before live motion.")
        if a.live and not a.allow_nominal:
            sys.exit("refusing live motion with a nominal calibration (pass --allow-nominal to override)")
    if list(calib.bed_mm) != list(plan["bed_mm"]):
        print(f"   note: calibration bed {calib.bed_mm} vs G-code bed {plan['bed_mm']} (only the transform is used)")
    calib.bed_mm = plan["bed_mm"]
    if a.bed_offset_mm:
        calib = calib.shifted([float(v) for v in a.bed_offset_mm.split(",")])
        print(f"   bed offset {a.bed_offset_mm} mm applied")

    # ------------------------------------------------------------------ 4. arm + plan
    kin = SO101()
    safety = Safety(max_speed_deg_s=a.speed, max_accel_deg_s2=a.accel)
    arm = Arm(a.arm or a.port, robot_id=a.robot_id, dry_run=not a.live, joint_map=JointMap.load_default(),
              safety=safety, confirm=not (a.yes or a.film), kin=kin, calibration_dir=a.calibration_dir)
    arm.connect()
    arm.set_bed_keepout(calib, margin_mm=1.5 if a.pick else 8.0)
    q_home = arm.q.copy()
    check = parts_clearance_check(kin, calib, plan["objects"])
    steps = hover_plan(kin, calib, plan, a.hover_mm, arm.q, align_jaws=not a.no_align_jaws)
    n_ok = sum(bool(s["ik_hover"].ok) for s in steps)
    if a.live and a.film:
        est = film_seconds(arm.q, ready_q(kin), steps, a.speed, a.accel, a.dwell_s, a.gripper_demo)
        print(f"\n== FILM RUN: {n_ok}/{len(steps)} parts reachable, ~{est / 60:.1f} min, joints <= {a.speed:.0f} deg/s."
              f"\n   Ctrl-C = stop and hold.  Second terminal: python -m real.arm_client {a.arm or 'remote://PI:8765'} "
              f"estop  (torque off)")
        if input("   start the camera, then type 'go' + Enter: ").strip().lower() != "go":
            arm.disconnect()
            sys.exit("not started")
    prev = None
    if a.sim_preview:
        from real.preview import Previewer
        prev = Previewer(calib, plan["objects"])
    print(f"== hover plan ({'LIVE' if a.live else 'dry-run'}): tips {a.hover_mm:.0f} mm above each part")
    results = []
    say(f"print finished - {len(steps)} parts on the bed: " + ", ".join(plan["pick_order"]))
    try:
        if a.film or a.pick:
            say("moving to the ready pose above the bed")
            try:
                arm.move_to(ready_q(kin), label="ready pose")
            except SafetyError as e:
                print(f"      SAFETY: {e}")
        for i, s in enumerate(steps, 1):
            rh, rt, o = s["ik_hover"], s["ik_transit"], s["obj"]
            hb = calib.robot_to_bed(kin.fk(rh.q, s["tcp"])[0])
            row = dict(name=s["name"], center_mm=o["center_mm"], height_mm=o.get("height_mm"),
                       hover_robot_mm=np.round(s["hover_robot"] * 1000, 1).tolist(), ik_ok=bool(rh.ok),
                       pitch_deg=rh.pitch_deg, pos_err_mm=round(rh.pos_err_mm, 2), ang_err_deg=round(rh.ang_err_deg, 2),
                       q_model_deg=np.degrees(rh.q).round(2).tolist(), lerobot=arm.jm.model_to_lerobot(rh.q),
                       q_transit_deg=np.degrees(rt.q).round(2).tolist() if rt.ok else None,
                       reached_bed_mm=np.round(hb, 1).tolist(), hover_mm=s["hover_mm"])
            tag = (f"{i}. {s['name']:<16} bed ({o['center_mm'][0]:.0f},{o['center_mm'][1]:.0f}) h={o.get('height_mm') or 0:.0f}"
                   f" -> robot {np.round(s['hover_robot'] * 1000).astype(int).tolist()} mm")
            if not rh.ok:
                row["status"] = "unreachable"
                print(f"   {tag}: UNREACHABLE (best pos err {rh.pos_err_mm:.0f} mm) - move the arm closer / re-plan")
            else:
                ori = "down" if rh.pitch_deg == 90 else (f"tilt {rh.pitch_deg:.0f} deg" if rh.pitch_deg else "free")
                low = f", hover lowered to {s['hover_mm']:.0f} mm" if s["hover_mm"] < a.hover_mm else ""
                print(f"   {tag}: IK ok ({ori}, err {rh.pos_err_mm:.1f} mm{low})")
                try:
                    safety.extra_checks = [check]
                    fp = o["footprint_mm"]
                    say(f"part {i}/{len(steps)}: {s['name']} - {fp[0]:.0f} x {fp[1]:.0f} mm, "
                        f"{o.get('height_mm') or 0:.0f} mm tall, at bed ({o['center_mm'][0]:.0f}, {o['center_mm'][1]:.0f}) mm")
                    if rt.ok:
                        arm.move_to(rt.q, label=f"transit above {s['name']}")
                    arm.move_to(rh.q, label=f"hover {s['name']}")
                    if a.gripper_demo:
                        say(f"over {s['name']}: jaws aligned with the part, {s['hover_mm']:.0f} mm above it")
                        arm.gripper(60)
                    if a.live:
                        time.sleep(a.dwell_s)
                    if a.gripper_demo:
                        arm.gripper(0)
                    if rt.ok:
                        arm.move_to(rt.q, label=f"retreat from {s['name']}")
                    row["status"] = "moved" if a.live else "dry-run ok"
                except SafetyError as e:
                    row["status"] = f"blocked: {e}"
                    print(f"      SAFETY: {e}")
                    if "declined" in str(e):
                        break
            if prev is not None:
                lines = [f"{i}/{len(steps)} {s['name']}: hover {s['hover_mm']:.0f} mm above part (h={o.get('height_mm') or 0:.0f} mm)",
                         f"bed ({o['center_mm'][0]:.0f}, {o['center_mm'][1]:.0f}) mm  status: {row['status']}",
                         f"pitch {rh.pitch_deg}  err {rh.pos_err_mm:.1f} mm  calib: {calib.method}"]
                f = runs / f"real_preview_{i:02d}_{s['name']}.png"
                prev.render(rh.q, f, target=s["hover_robot"], lines=lines)
                row["preview"] = str(f)
            results.append(row)
        if a.pick:
            results.append(pick_one(a, arm, kin, calib, plan, say))
        if a.park:
            say("all parts visited - back to the start pose")
            safety.extra_checks = []
            arm.move_to(ready_q(kin), label="ready pose")
            if a.live:
                arm.move_to(q_home, label="start pose")
    except KeyboardInterrupt:
        print("\n== Ctrl-C: stopping, the arm holds its pose")
        if arm.drv is not None:
            arm.drv.stop("operator Ctrl-C")
        raise
    finally:
        arm.disconnect()
    if prev is not None and steps:
        f = runs / "real_preview_00_overview.png"
        prev.render(ready_q(kin), f, lines=[f"{len(steps)} parts, pick order: " + " > ".join(plan["pick_order"])[:80],
                                      f"bed {plan['bed_mm'][0]:.0f}x{plan['bed_mm'][1]:.0f} mm, calib {calib.method}"])
        print("== previews:", f.parent / "real_preview_*.png")
    report["results"] = results
    report["ready_q_deg"] = np.degrees(ready_q(kin)).round(2).tolist()
    report["calib"] = calib.to_json()
    out.write_text(json.dumps(report, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    ok = sum(r["status"] in ("moved", "dry-run ok") for r in results if "pick" not in r)
    picks = [r for r in results if "pick" in r]
    print(f"== done: {ok}/{len(results) - len(picks)} parts hovered ({'live' if a.live else 'dry-run'})"
          + "".join(f"; pick {p['pick']}: {p['status']}" for p in picks) + f"; log {out}")
    return report


def pick_one(a, arm, kin, calib, plan, say) -> dict:
    """Optional: grasp ONE part and drop it at --box-mm (bed mm): above -> descend -> close -> lift -> over the
    box -> open. During this sequence only the bed (1.5 mm) and the OTHER parts are keep-out."""
    by = {o["name"]: o for o in plan["objects"]}
    name = plan["pick_order"][0] if a.pick == "first" else a.pick
    if name not in by:
        return dict(pick=name, status=f"no part named {name} (have {list(by)})")
    o = by[name]
    others = [x for x in plan["objects"] if x["name"] != name]
    row = dict(pick=name, status="?")
    pre, g, width = grasp_plan(kin, calib, o, arm.q)
    if not (pre.ok and g.ok):
        row["status"] = f"grasp unreachable (pre {pre.pos_err_mm:.0f} mm, grasp {g.pos_err_mm:.0f} mm)"
        print("   pick:", row["status"])
        return row
    box = [float(v) for v in a.box_mm.split(",")]
    r_box = kin.solve_hover(calib.bed_to_robot(box), tcp=pinch(width, 8.0), q0=pre.q)
    if not r_box.ok:
        row["status"] = f"box point {box} unreachable"
        print("   pick:", row["status"])
        return row
    open_rad = kin.gripper_for_opening(min(width + 25.0, 90.0))
    closed_rad = kin.gripper_for_opening(max(width - 6.0, 0.0))     # squeeze: the 50 % torque limit holds it
    try:
        arm.safety.extra_checks = [parts_clearance_check(kin, calib, others, bed_margin_mm=1.5)]
        say(f"picking {name}: {width:.0f} mm grip across its short side")
        for q, label, grip in ((pre.q, "above the grasp", open_rad), (g.q, "grasp height", open_rad),
                               (g.q, "close jaws", closed_rad), (pre.q, "lift", closed_rad),
                               (r_box.q, "over the box", closed_rad), (r_box.q, "release", open_rad)):
            q = np.array(q, float).copy()
            q[5] = grip
            arm.move_to(q, label=f"pick {name}: {label}")
        say(f"{name} dropped in the box")
        row["status"] = "picked" if a.live else "dry-run ok"
    except SafetyError as e:
        row["status"] = f"blocked: {e}"
        print(f"      SAFETY: {e}")
    return row


if __name__ == "__main__":
    main()
