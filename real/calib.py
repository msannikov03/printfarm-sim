"""Bed -> robot calibration for the SO-101.

Frames: bed = slicer frame (mm, origin front-left corner, z up from the bed surface);
robot = SO-101 MJCF base frame (m, x forward, y left, z up). Stored as a 4x4 T_robot_bed in metres:
    p_robot = T[:3, :3] @ (p_bed_mm / 1000) + T[:3, 3]

Method A, touch-off (recommended):
    python -m real.calib touch --port /dev/tty.usbmodemXXXX --robot-id follower1 \
        --points 20,20 200,20 200,200 20,200
  Torque is switched off; for each bed point you hand-guide the FIXED jaw tip onto the point and press
  Enter; joints are read, FK gives the tip in the robot frame, and a Kabsch fit gives T (RMS residual
  printed). 4 points spread over the bed; RMS < 3 mm is good, > 5 mm means a bad touch or a unit/
  offset problem (see README troubleshooting).
Method B, ArUco (optional): markers at known bed positions seen by a camera with known intrinsics and
  a known camera pose in the robot frame (e.g. the SO-101 wrist camera at a known joint pose).
Method C, tape measure (fallback): python -m real.calib manual --base-in-bed 110,-60,0 --yaw 90
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from real import ROOT

CALIB_JSON = ROOT / "calib.json"


def kabsch(P: np.ndarray, Q: np.ndarray, scale: bool = False):
    """Least-squares R, t (, s) with Q ~ s R P + t (Umeyama 1991). P, Q: (n, 3), n >= 3 non-collinear."""
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    if len(P) < 3:
        raise ValueError("need at least 3 point pairs")
    mp, mq = P.mean(0), Q.mean(0)
    X, Y = P - mp, Q - mq
    if np.linalg.matrix_rank(X, tol=1e-6) < 2:
        raise ValueError("calibration points are collinear; use points spread over the bed")
    U, S, Vt = np.linalg.svd(Y.T @ X / len(P))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt)) or 1.0
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / (X ** 2).sum(1).mean()) if scale else 1.0
    t = mq - s * R @ mp
    return (R, t, s) if scale else (R, t)


def rz(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


@dataclass
class BedCalib:
    T_robot_bed: np.ndarray
    method: str = "nominal"
    rms_mm: float | None = None
    bed_mm: list = field(default_factory=lambda: [220.0, 220.0])
    points: list = field(default_factory=list)
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))
    notes: str = ""
    checks: list = field(default_factory=list)    # held-out control points (touched, not in the fit): errors in mm

    @property
    def R(self):
        return self.T_robot_bed[:3, :3]

    @property
    def t(self):
        return self.T_robot_bed[:3, 3]

    def bed_to_robot(self, p_mm) -> np.ndarray:
        """bed mm (x, y[, z]) -> robot metres. Accepts (3,), (2,), (n, 2) or (n, 3)."""
        p = np.atleast_2d(np.asarray(p_mm, float))
        if p.shape[1] == 2:
            p = np.c_[p, np.zeros(len(p))]
        q = p / 1000.0 @ self.R.T + self.t
        return q[0] if np.ndim(p_mm) == 1 else q

    def robot_to_bed(self, p_m) -> np.ndarray:
        p = np.atleast_2d(np.asarray(p_m, float))
        q = (p - self.t) @ self.R * 1000.0
        return q[0] if np.ndim(p_m) == 1 else q

    def shifted(self, d_mm) -> "BedCalib":
        """The bed has moved by d_mm (bed axes) since calibration, e.g. (0, 0, -15) if a Z-bed printer
        (K1/K2/Hi) ends 15 mm lower, or (0, 30, 0) if a bed-slinger parks its bed 30 mm further back."""
        T = self.T_robot_bed.copy()
        T[:3, 3] = self.t + self.R @ (np.asarray(d_mm, float) / 1000.0)
        return BedCalib(T, method=self.method, rms_mm=self.rms_mm, bed_mm=list(self.bed_mm), points=self.points,
                        created=self.created, notes=self.notes + f"; shifted by {list(d_mm)} mm")

    def bed_dir_to_robot(self, v) -> np.ndarray:
        v = np.asarray(v, float)
        return self.R @ (np.r_[v, 0.0] if len(v) == 2 else v)

    def to_json(self) -> dict:
        return dict(T_robot_bed=np.round(self.T_robot_bed, 6).tolist(), method=self.method,
                    rms_mm=self.rms_mm, bed_mm=list(self.bed_mm), points=self.points, created=self.created,
                    notes=self.notes, frames="p_robot[m] = R @ p_bed[mm]/1000 + t; bed = slicer frame, "
                                         "robot = SO-101 MJCF base (x fwd, y left, z up)",
                    **({"checks": self.checks} if self.checks else {}))

    def save(self, path=CALIB_JSON) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_json(), indent=2))
        return path

    @classmethod
    def load(cls, path=CALIB_JSON) -> "BedCalib":
        d = json.loads(Path(path).read_text())
        return cls(T_robot_bed=np.array(d["T_robot_bed"], float), method=d.get("method", "?"),
                   rms_mm=d.get("rms_mm"), bed_mm=d.get("bed_mm", [220, 220]), points=d.get("points", []),
                   created=d.get("created", ""), notes=d.get("notes", ""), checks=d.get("checks", []))


def from_base_pose(base_in_bed_mm, yaw_deg: float, bed_mm=(220, 220), method="manual", notes="") -> BedCalib:
    """Robot base origin at base_in_bed_mm (x, y, z in the bed frame; z = base plate height relative to the
    bed surface, negative if the bed is higher), robot x-axis rotated yaw_deg from bed +x (90 = facing +y)."""
    R_bed_robot = rz(yaw_deg)
    R = R_bed_robot.T
    b = np.asarray(base_in_bed_mm, float) / 1000.0
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -R @ b
    return BedCalib(T, method=method, bed_mm=list(bed_mm), notes=notes or
                    f"base at bed ({base_in_bed_mm[0]}, {base_in_bed_mm[1]}, {base_in_bed_mm[2]}) mm, yaw {yaw_deg} deg")


def nominal(bed_mm=(220, 220), gap_mm: float = 110.0, base_below_bed_mm: float = 0.0) -> BedCalib:
    """Default for --mock / previews: arm centred in front of the bed edge (y = -gap), facing into the printer."""
    return from_base_pose((bed_mm[0] / 2, -gap_mm, -base_below_bed_mm), 90.0, bed_mm, method="nominal",
                          notes="NOMINAL placement (not measured) - calibrate before moving the real arm")


def fit_touch(bed_pts_mm, robot_pts_m, bed_mm=(220, 220), tip_above_bed_mm: float = 0.0, joints=None) -> BedCalib:
    """Rigid fit from touched points. tip_above_bed_mm: e.g. 0.1 if you touch a sheet of paper."""
    P = np.atleast_2d(np.asarray(bed_pts_mm, float))
    if P.shape[1] == 2:
        P = np.c_[P, np.full(len(P), tip_above_bed_mm)]
    Q = np.asarray(robot_pts_m, float)
    R, t = kabsch(P / 1000.0, Q)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    res = (P / 1000.0 @ R.T + t - Q) * 1000.0
    err = np.linalg.norm(res, axis=1)
    pts = [dict(bed_mm=P[i].round(3).tolist(), robot_m=Q[i].round(6).tolist(), residual_mm=round(float(err[i]), 3),
                **({"lerobot": joints[i]} if joints is not None else {})) for i in range(len(P))]
    rms = float(np.sqrt((err ** 2).mean()))
    c = BedCalib(T, method="touch", rms_mm=round(rms, 3), bed_mm=list(bed_mm), points=pts)
    tilt = math.degrees(math.acos(np.clip(R[2, 2], -1, 1)))
    c.notes = f"bed plane tilt vs robot base {tilt:.1f} deg"
    return c


def held_out_errors(c: BedCalib, bed_pts_mm, robot_pts_m, tip_above_bed_mm: float = 0.0, joints=None) -> list:
    """Held-out control points: touched like the fit points but not used in the fit. For each, the fitted
    calibration maps the touched tip into the bed frame; the error is that minus the point touched (mm, bed axes:
    x, y in the bed plane, z up). A low fit residual does not prove the calibration; this error is what to compare
    with the task tolerance."""
    out = []
    for k, (b, q) in enumerate(zip(bed_pts_mm, robot_pts_m)):
        b = np.asarray(b, float)
        b = np.r_[b[:2], tip_above_bed_mm] if len(b) == 2 else b
        e = c.robot_to_bed(np.asarray(q, float)) - b
        out.append(dict(bed_mm=b.round(3).tolist(), robot_m=np.asarray(q, float).round(6).tolist(),
                        error_mm=e.round(3).tolist(), error_xy_mm=round(float(np.hypot(e[0], e[1])), 3),
                        error_3d_mm=round(float(np.linalg.norm(e)), 3),
                        **({"lerobot": joints[k]} if joints is not None else {})))
    return out


# ----------------------------------------------------------------------------- Method B (ArUco)

def aruco_T_cam_bed(image, K, dist, marker_len_mm: float, markers: dict, dictionary: str = "DICT_4X4_50"):
    """Pose of the bed in the camera frame from ArUco markers at known bed positions.

    markers: {id: (x_mm, y_mm)} = marker CENTRE on the bed, marker axes aligned with the bed axes.
    Returns (T_cam_bed 4x4 in metres, ids_used, reprojection_rms_px). Compose with a known camera pose:
    T_robot_bed = T_robot_cam @ T_cam_bed.   STUB-LEVEL: tested on synthetic images only.
    """
    import cv2
    aruco = cv2.aruco
    params = aruco.DetectorParameters()
    params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    det = aruco.ArucoDetector(aruco.getPredefinedDictionary(getattr(aruco, dictionary)), params)
    corners, ids, _ = det.detectMarkers(image)
    if ids is None:
        raise RuntimeError("no ArUco markers detected")
    h = marker_len_mm / 2
    obj, img, used = [], [], []
    for c, i in zip(corners, ids.ravel()):
        if int(i) not in markers:
            continue
        x, y = markers[int(i)]
        # ArUco corner order: top-left, top-right, bottom-right, bottom-left (marker y up = bed +y)
        obj += [[x - h, y + h, 0], [x + h, y + h, 0], [x + h, y - h, 0], [x - h, y - h, 0]]
        img += c.reshape(4, 2).tolist()
        used.append(int(i))
    if not used:
        raise RuntimeError(f"detected ids {ids.ravel().tolist()} but none are in the marker table")
    obj = np.array(obj, float) / 1000.0
    img = np.array(img, float)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE if len(used) == 1 else cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed")
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    rms = float(np.sqrt(((proj.reshape(-1, 2) - img) ** 2).sum(1).mean()))
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return T, used, rms


# ----------------------------------------------------------------------------- CLI

def _parse_pts(items):
    return [tuple(float(v) for v in s.split(",")) for s in items]


def _record(arm, kin, samples, hint=None):
    from real.so101 import FIXED_TIP
    reads = [arm.read_lerobot(target_hint_bed_mm=hint) for _ in range(samples)]
    lr = {j: float(np.mean([r[j] for r in reads])) for j in reads[0]}
    q = arm.jm.lerobot_to_model(lr)
    p, _ = kin.fk(q, FIXED_TIP)
    print(f"    joints(model deg) {np.degrees(q[:5]).round(1).tolist()} -> fixed-jaw tip in robot frame "
          f"{(p * 1000).round(1).tolist()} mm")
    return p, lr


JOG_HELP = ("jog: x+5 / y-2 / z-1 (mm, robot frame: x forward, y left, z up; several per line ok), "
            "'ok' = record this point, 'q' = abort")


def _jog(arm, kin, start_tip, input_fn, label):
    """Move the fixed-jaw tip with typed mm steps (torque ON, jaws pointing down). Returns when 'ok'."""
    import re
    from real.so101 import FIXED_TIP
    tip = np.array(start_tip, float)
    first = True
    while True:
        r = kin.solve(tip, approach=np.array([0, 0, -1.0]), tcp=FIXED_TIP, q0=arm.q, ang_tol_deg=25.0)
        if not r.ok:
            r = kin.solve(tip, tcp=FIXED_TIP, q0=arm.q)
        if r.ok:
            q = r.q.copy()
            q[5] = arm.q[5]
            arm.move_to(q, label=f"jog {label}", confirm=None if first else False)   # ask before the big move
            first = False
        else:
            print(f"    cannot reach {np.round(tip * 1000, 1).tolist()} mm; step back")
        cmd = input_fn(f"  {label} tip at {np.round(kin.fk(arm.q, FIXED_TIP)[0] * 1000, 1).tolist()} mm  {JOG_HELP}\n  > ")
        cmd = cmd.strip().lower()
        if cmd == "ok":
            return
        if cmd == "q":
            raise KeyboardInterrupt("touch-off aborted")
        for ax, v in re.findall(r"([xyz])\s*([-+]?\d+(?:\.\d+)?)", cmd):
            tip["xyz".index(ax)] += float(v) / 1000.0


def touch_session(arm, bed_pts, kin=None, samples: int = 10, input_fn=input, tip_above_bed_mm=0.0, bed_mm=(220, 220),
                  jog_guess: "BedCalib | None" = None, check_pts=()):
    """Interactive touch-off. Hand-guide mode (default): torque off, push the fixed-jaw tip onto each point.
    Jog mode (jog_guess = a rough calibration, e.g. nominal or tape-measured): torque on, the arm goes 15 mm
    above each point via the guess and you nudge it down with typed mm steps. check_pts: held-out control points,
    touched after the fit points the same way and left out of the fit (BedCalib.checks). Returns BedCalib."""
    from real.so101 import FIXED_TIP, SO101
    kin = kin or SO101()
    robot_pts, joints = [], []
    n_fit, todo = len(bed_pts), list(bed_pts) + list(check_pts)
    if jog_guess is None:
        arm.torque(False)
    try:
        for i, (x, y) in enumerate(todo, 1):
            kind = "held-out check point" if i > n_fit else "bed point"
            label = f"[{i}/{len(todo)}] {kind} ({x:.0f}, {y:.0f}) mm"
            if jog_guess is None:
                input_fn(f"{label}: hand-guide the FIXED jaw tip onto it, hold still, press Enter ")
                p, lr = _record(arm, kin, samples, hint=(x, y))
            else:
                _jog(arm, kin, jog_guess.bed_to_robot([x, y, 15.0]), input_fn, label)
                p, lr = _record(arm, kin, samples)
                up = arm.q.copy()
                r = kin.solve(p + [0, 0, 0.03], tcp=FIXED_TIP, q0=up)
                if r.ok:
                    r.q[5] = up[5]
                    arm.move_to(r.q, label="lift 30 mm", confirm=False)
            robot_pts.append(p)
            joints.append(lr)
    finally:
        if jog_guess is None:
            arm.torque(True)
    c = fit_touch(bed_pts, robot_pts[:n_fit], bed_mm=bed_mm, tip_above_bed_mm=tip_above_bed_mm, joints=joints[:n_fit])
    if check_pts:
        c.checks = held_out_errors(c, check_pts, robot_pts[n_fit:], tip_above_bed_mm, joints[n_fit:])
    return c


def print_calib(c: BedCalib):
    print(f"method={c.method}  rms={c.rms_mm} mm  bed={c.bed_mm}  {c.notes}")
    for p in c.points:
        print(f"  bed {p['bed_mm']} -> robot {np.round(np.array(p['robot_m']) * 1000, 1).tolist()} mm, "
              f"residual {p['residual_mm']} mm")
    corners = [(0, 0), (c.bed_mm[0], 0), (c.bed_mm[0], c.bed_mm[1]), (0, c.bed_mm[1])]
    for b, r in zip(corners, c.bed_to_robot(corners)):
        print(f"  bed corner {b} -> robot ({r[0] * 1000:.0f}, {r[1] * 1000:.0f}, {r[2] * 1000:.0f}) mm")
    for p in c.checks:
        print(f"  held-out {p['bed_mm']}: error {p['error_mm']} mm (bed x, y, z), in-plane {p['error_xy_mm']} mm, "
              f"3D {p['error_3d_mm']} mm")
    if c.rms_mm is not None and c.rms_mm > 5:
        print("  WARNING: RMS > 5 mm - re-touch the points or check joint offsets/signs (README troubleshooting)")


def _scripted_jog():
    """Dry-run jog: go down 15 mm and accept (the simulated guess is exact)."""
    seq = iter(["z-10", "z-5", "ok"] * 100)

    def f(msg):
        cmd = next(seq)
        print(msg.splitlines()[0] + f"  (dry-run types: {cmd})")
        return cmd
    return f


def main(argv=None):
    ap = argparse.ArgumentParser(description="bed -> SO-101 calibration")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("touch", help="touch-off with the real arm (torque off, hand-guide)")
    t.add_argument("--port", "--arm", dest="port", help="SO-101 serial port (LeRobot), or remote://<pi>:8765, "
                   "feetech:///dev/ttyACM0 (see real/arm.py backends)")
    t.add_argument("--robot-id", default="follower", help="LeRobot calibration id used with lerobot-calibrate")
    t.add_argument("--calibration-dir", help="LeRobot calibration dir if not the default")
    t.add_argument("--points", nargs="+", default=["20,20", "200,20", "200,200", "20,200"], help="bed points x,y mm")
    t.add_argument("--check", nargs="+", default=[], metavar="X,Y",
                   help="held-out control points x,y mm: touched after the fit points, not used in the fit; their "
                        "error is the one to compare with the task tolerance")
    t.add_argument("--bed", default="220x220")
    t.add_argument("--tip-above-bed", type=float, default=0.0, help="mm, e.g. 0.1 if touching a paper sheet")
    t.add_argument("--dry-run", action="store_true", help="simulated hand on a nominal placement (no hardware)")
    t.add_argument("--jog", action="store_true", help="torque on, nudge the tip with typed mm steps instead of "
                   "hand-guiding (starts 15 mm above each point using --guess)")
    t.add_argument("--guess", help="rough calibration JSON for --jog (default: nominal placement, or the existing "
                   "real/calib.json if present)")
    t.add_argument("--joint-map", help="joint_map.json (signs/offsets), default real/joint_map.json if present")
    t.add_argument("-o", "--out", default=str(CALIB_JSON))
    m = sub.add_parser("manual", help="tape-measure placement")
    m.add_argument("--base-in-bed", required=True, help="x,y,z mm of the robot base origin in the bed frame")
    m.add_argument("--yaw", type=float, default=90.0, help="robot x-axis angle from bed +x (deg); 90 = facing +y")
    m.add_argument("--bed", default="220x220")
    m.add_argument("-o", "--out", default=str(CALIB_JSON))
    f = sub.add_parser("fit", help="refit a previous touch calib (or [{bed_mm:[x,y], lerobot:{joint: value}}]) "
                                   "with the current joint map")
    f.add_argument("points_json")
    f.add_argument("--bed", default="220x220")
    f.add_argument("-o", "--out", default=str(CALIB_JSON))
    s = sub.add_parser("show", help="print a calibration file and its reach map")
    s.add_argument("path", nargs="?", default=str(CALIB_JSON))
    s.add_argument("--hover-mm", type=float, default=40.0)
    a = ap.parse_args(argv)
    bed = tuple(float(v) for v in getattr(a, "bed", "220x220").lower().split("x")) if hasattr(a, "bed") else None

    if a.cmd == "manual":
        c = from_base_pose(tuple(float(v) for v in a.base_in_bed.split(",")), a.yaw, bed)
    elif a.cmd == "touch":
        from real.arm import Arm, JointMap
        jm = JointMap.load(a.joint_map) if a.joint_map else JointMap.load_default()
        if a.dry_run:
            arm = Arm(None, dry_run=True, joint_map=jm, sim_hand=nominal(bed))
            inp = lambda msg: print(msg + "(dry-run: simulated)")
        else:
            arm = Arm(a.port, robot_id=a.robot_id, dry_run=False, joint_map=jm, calibration_dir=a.calibration_dir)
            arm.connect()
            inp = input
        try:
            guess = None
            if a.jog:
                guess = BedCalib.load(a.guess) if a.guess else (BedCalib.load(CALIB_JSON) if CALIB_JSON.exists()
                                                                else nominal(bed))
                if a.dry_run:
                    inp = _scripted_jog()
            c = touch_session(arm, _parse_pts(a.points), input_fn=inp, tip_above_bed_mm=a.tip_above_bed, bed_mm=bed,
                              jog_guess=guess, check_pts=_parse_pts(a.check))
        finally:
            arm.disconnect()
    elif a.cmd == "fit":
        from real.so101 import FIXED_TIP, SO101
        d = json.loads(Path(a.points_json).read_text())
        pts = d["points"] if isinstance(d, dict) else d
        kin = SO101()
        from real.arm import JointMap
        jm = JointMap.load_default()
        robot = [kin.fk(jm.lerobot_to_model(p["lerobot"]), FIXED_TIP)[0] for p in pts]
        c = fit_touch([p["bed_mm"][:2] for p in pts], robot, bed_mm=bed, joints=[p["lerobot"] for p in pts])
    else:
        c = BedCalib.load(a.path)
        print_calib(c)
        from real.reach import print_reach_map
        print_reach_map(c, hover_mm=a.hover_mm)
        return c
    print_calib(c)
    print("saved", c.save(a.out))
    return c


if __name__ == "__main__":
    main()
