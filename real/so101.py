"""SO-101 kinematics on the official MuJoCo model (mujoco_menagerie robotstudio_so101).

Robot frame = the MJCF world frame = the arm's base plate: x forward (the direction the arm points at
q = 0), y to the left, z up, metres. Joint order = MJCF order = LeRobot motor order:
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.

Tool points are defined in the `gripper` body frame (the wrist_roll link that carries the FIXED jaw),
so they do not move when the gripper opens:
  FIXED_TIP  tip of the fixed jaw (what you touch the bed with during calibration)
  pinch(w)   point between the jaw pads for a part w mm wide (the moving jaw closes onto the fixed one)
Axes in the same frame: approach = -z (out of the jaws), jaw closing axis = +x (fixed -> moving jaw).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from real import ASSETS

MODEL_DIR = ASSETS / "so101"
MODEL_XML = MODEL_DIR / "so101.xml"
SCENE_XML = MODEL_DIR / "scene.xml"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ARM = JOINTS[:5]
TCP_BODY = "gripper"

FIXED_TIP = np.array([-0.0081, 0.0, -0.1018])     # fixed_jaw_sph_tip1 (+ its 0.75 mm radius)
APPROACH_LOCAL = np.array([0.0, 0.0, -1.0])
JAW_LOCAL = np.array([1.0, 0.0, 0.0])
PAN_AXIS_XY = np.array([0.0388353, 0.0])           # shoulder body position in the base frame
GRIPPER_CLOSED_RAD, GRIPPER_OPEN_RAD = -0.174533, 1.7453292


def pinch(width_mm: float = 20.0, depth_mm: float = 8.0) -> np.ndarray:
    """Grasp centre for a part `width_mm` wide, `depth_mm` back from the jaw tips."""
    w = float(np.clip(width_mm, 4.0, 90.0)) / 1000.0
    return FIXED_TIP + np.array([w / 2 + 0.0008, 0.0, depth_mm / 1000.0])


def hover_tcp(width_mm: float = 20.0) -> np.ndarray:
    """Hover reference: between the jaws, at the tip plane (so 'N mm above' = jaw tips N mm above)."""
    return pinch(width_mm, depth_mm=0.0)


@dataclass
class IKResult:
    q: np.ndarray                 # 6 joints (rad), gripper untouched
    ok: bool
    pos_err_mm: float
    ang_err_deg: float
    pitch_deg: float | None = None        # approach pitch below horizontal (90 = straight down)
    note: str = ""
    tried: list = field(default_factory=list)


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class SO101:
    def __init__(self, xml=MODEL_XML):
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.qadr = np.array([m.joint(j).qposadr[0] for j in JOINTS])
        self.dadr = np.array([m.joint(j).dofadr[0] for j in JOINTS])
        self.lo = np.array([m.joint(j).range[0] for j in JOINTS])
        self.hi = np.array([m.joint(j).range[1] for j in JOINTS])
        self.body = m.body(TCP_BODY).id

    # -------------------------------------------------------------- forward kinematics
    def _set(self, q):
        q = np.asarray(q, float)
        self.data.qpos[self.qadr[:len(q)]] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def fk(self, q, tcp=FIXED_TIP):
        """(position m, body rotation 3x3) of a tool point given in the gripper-body frame."""
        self._set(q)
        R = self.data.xmat[self.body].reshape(3, 3).copy()
        return self.data.xpos[self.body] + R @ np.asarray(tcp, float), R

    def tool_axes(self, q):
        _, R = self.fk(q)
        return R @ APPROACH_LOCAL, R @ JAW_LOCAL

    def body_pose(self, q, body: str):
        self._set(q)
        b = self.model.body(body).id
        return self.data.xpos[b].copy(), self.data.xmat[b].reshape(3, 3).copy()

    def jaw_opening_mm(self, gripper_rad: float) -> float:
        """Distance between the jaw-tip spheres for a gripper joint angle (wrist pose irrelevant)."""
        q = np.zeros(6)
        q[5] = gripper_rad
        self._set(q)
        m, d = self.model, self.data
        a = d.geom_xpos[m.geom("fixed_jaw_sph_tip1").id]
        b = d.geom_xpos[m.geom("moving_jaw_sph_tip1").id]
        return float(np.linalg.norm(a - b) * 1000)

    def gripper_for_opening(self, mm: float) -> float:
        """Gripper joint angle (rad) that opens the jaw tips to about `mm`."""
        if not hasattr(self, "_open_tab"):
            g = np.linspace(GRIPPER_CLOSED_RAD, GRIPPER_OPEN_RAD, 60)
            self._open_tab = (np.array([self.jaw_opening_mm(x) for x in g]), g)
        o, g = self._open_tab
        return float(np.interp(mm, o, g))

    def _jac(self, point):
        jp = np.zeros((3, self.model.nv))
        jr = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jp, jr, point, self.body)
        cols = self.dadr[:5]
        return jp[:, cols], jr[:, cols]

    # -------------------------------------------------------------- inverse kinematics
    def ik(self, target, approach=None, jaw=None, tcp=FIXED_TIP, q0=None, iters=300,
           w_rot=0.08, lam=0.02, pos_tol_mm=1.0, ang_tol_deg=2.5) -> IKResult:
        """Damped-least-squares IK on the 5 arm joints, clamped to joint limits.

        target: tool point position (m, robot frame). approach: desired approach axis (unit, world) or None.
        jaw: desired jaw closing axis (only its component perpendicular to approach is used; sign-free).
        """
        target = np.asarray(target, float)
        q = np.zeros(6) if q0 is None else np.array(q0, float).copy()
        q[:5] = np.clip(q[:5], self.lo[:5], self.hi[:5])
        a_t = None if approach is None else np.asarray(approach, float) / np.linalg.norm(approach)
        j_t = None
        if jaw is not None and a_t is not None:
            j_t = np.asarray(jaw, float) - np.dot(jaw, a_t) * a_t
            j_t = None if np.linalg.norm(j_t) < 1e-6 else j_t / np.linalg.norm(j_t)
        for _ in range(iters):
            p, R = self.fk(q, tcp)
            Jp, Jr = self._jac(p)
            rows_e, rows_J = [target - p], [Jp]
            if a_t is not None:
                a = R @ APPROACH_LOCAL
                rows_e.append(w_rot * np.cross(a, a_t))
                rows_J.append(w_rot * (np.eye(3) - np.outer(a, a)) @ Jr)
                if j_t is not None:
                    jv = R @ JAW_LOCAL
                    jt = j_t if np.dot(jv, j_t) >= 0 else -j_t
                    rows_e.append(w_rot * np.cross(jv, jt))
                    rows_J.append(w_rot * np.outer(a, a) @ Jr)      # roll about the approach axis only
            e = np.concatenate(rows_e)
            J = np.vstack(rows_J)
            if np.linalg.norm(e[:3]) < 2e-4 and (len(e) == 3 or np.linalg.norm(e[3:]) < w_rot * 1e-3):
                break
            dq = J.T @ np.linalg.solve(J @ J.T + lam ** 2 * np.eye(len(e)), e)
            n = np.linalg.norm(dq)
            if n > 0.2:
                dq *= 0.2 / n
            q[:5] = np.clip(q[:5] + dq, self.lo[:5], self.hi[:5])
        return self._grade(q, target, a_t, j_t, tcp, pos_tol_mm, ang_tol_deg)

    def _grade(self, q, target, a_t, j_t, tcp, pos_tol_mm, ang_tol_deg) -> IKResult:
        p, R = self.fk(q, tcp)
        pe = float(np.linalg.norm(target - p) * 1000)
        ae = 0.0
        if a_t is not None:
            ae = math.degrees(math.acos(np.clip(np.dot(R @ APPROACH_LOCAL, a_t), -1, 1)))
            if j_t is not None:
                ae = max(ae, math.degrees(math.acos(np.clip(abs(np.dot(R @ JAW_LOCAL, j_t)), -1, 1))))
        return IKResult(q=q.copy(), ok=pe <= pos_tol_mm and ae <= ang_tol_deg, pos_err_mm=pe, ang_err_deg=ae)

    def seeds(self, target, q0=None):
        pan = -math.atan2(target[1] - PAN_AXIS_XY[1], target[0] - PAN_AXIS_XY[0])   # +pan turns clockwise (seen from above)
        pan = float(np.clip(pan, self.lo[0], self.hi[0]))
        out = [] if q0 is None else [np.asarray(q0, float)]
        for lift, elbow, wflex in ((-0.3, 0.6, 1.2), (0.3, -0.2, 1.4), (-0.8, 1.2, 1.0), (0.6, -0.8, 1.5)):
            out.append(np.array([pan, lift, elbow, wflex, 0.0, 0.0]))
        return out

    def solve(self, target, approach=None, jaw=None, tcp=FIXED_TIP, q0=None, **kw) -> IKResult:
        """IK from several seeds; best = ok and closest to q0 (or first seed), else the lowest error."""
        res = [self.ik(target, approach, jaw, tcp, s, **kw) for s in self.seeds(target, q0)]
        ref = np.asarray(q0 if q0 is not None else res[0].q)
        good = [r for r in res if r.ok]
        if good:
            return min(good, key=lambda r: np.linalg.norm(r.q[:5] - ref[:5]))
        return min(res, key=lambda r: r.pos_err_mm + r.ang_err_deg)

    def solve_hover(self, target, jaw=None, tcp=FIXED_TIP, q0=None,
                    pitches=(90, 80, 70, 60, 50, 40, 30)) -> IKResult:
        """Gripper pointing down if reachable, else tilted towards the target (pitch below horizontal)."""
        target = np.asarray(target, float)
        radial = target[:2] - PAN_AXIS_XY
        radial = radial / (np.linalg.norm(radial) + 1e-12)
        tried = []
        for p in pitches:
            pr = math.radians(p)
            approach = np.array([math.cos(pr) * radial[0], math.cos(pr) * radial[1], -math.sin(pr)])
            r = self.solve(target, approach, jaw, tcp, q0, ang_tol_deg=2.5 if p == 90 else 5.0)
            tried.append((p, round(r.pos_err_mm, 2), round(r.ang_err_deg, 2)))
            if r.ok:
                r.pitch_deg, r.tried = float(p), tried
                return r
        r = self.solve(target, None, None, tcp, q0)
        r.pitch_deg, r.tried = None, tried
        r.note = "position-only fallback" if r.ok else "unreachable"
        return r
