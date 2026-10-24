"""Runtime wrapper around the farm model: index lookups, part spawning/parking, adhesion, door latch, stepping."""
import math

import mujoco
import numpy as np

from farm import catalog as C
from farm import scene

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
HOME = np.array([0.0, 1.05, 1.05, 0.0, 0.0, 0.0])
GRIP_OPEN, GRIP_CLOSED = 0.037, 0.0
# Magnetic door catch at the free edge (ASSUMPTION, see docs/door.md): holding force F0 at zero gap, falling off as
# F0 / (1 + g/G0)^2 with the gap g at the magnet (nominal catalogue ratings of small cabinet catches: 1-5.5 kgf).
LATCH_F0 = 10.0                         # N at contact
LATCH_G0 = 0.002                        # m: force halves at ~0.8 mm, ~8 % left at 5 mm
LATCH_GMAX = 0.02                       # m: no pull beyond 2 cm
LATCH_R = C.DOOR_W - 0.01               # magnet / strike plate distance from the hinge axis (m)
# Reed switch next to the magnet (ASSUMPTION): closes below REED_GAP, opens again above REED_GAP + REED_HYST
REED_GAP, REED_HYST = 0.003, 0.001
FLAP_SHUT = -0.05                       # flap actuator target when shut: pressed against the joint stop at 0
FLAP_ENC_SIGMA = math.radians(0.2)      # flap hinge encoder noise, 1 sigma (ASSUMPTION)


def _yaw_quat(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


class World:
    def __init__(self, model=None):
        self.m = model or scene.build()
        self.d = mujoco.MjData(self.m)
        m = self.m
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        self.rail_q = m.jnt_qposadr[jid("rail")]
        self.arm_q = np.array([m.jnt_qposadr[jid(n)] for n in ARM_JOINTS])
        self.arm_v = np.array([m.jnt_dofadr[jid(n)] for n in ARM_JOINTS])
        self.rail_v = m.jnt_dofadr[jid("rail")]
        self.finger_q = m.jnt_qposadr[jid("left_finger")]
        self.finger_rq = m.jnt_qposadr[jid("right_finger")]
        self.door_q = np.array([m.jnt_qposadr[jid(f"door{i}")] for i in range(C.N_PRINTERS)])
        self.door_v = np.array([m.jnt_dofadr[jid(f"door{i}")] for i in range(C.N_PRINTERS)])
        aid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
        self.act_rail = aid("rail")
        self.act_arm = np.array([aid(n) for n in ARM_JOINTS])
        self.act_grip = aid("gripper")
        self.grasp_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "pinch")
        self.pool = scene.part_pool()
        self.part_body = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n, _ in self.pool}
        self.part_sku = dict(self.pool)
        self.part_q = {n: m.jnt_qposadr[m.body_jntadr[b]] for n, b in self.part_body.items()}
        self.part_v = {n: m.jnt_dofadr[m.body_jntadr[b]] for n, b in self.part_body.items()}
        self.part_geoms = {n: [g for g in range(m.ngeom) if m.geom_bodyid[g] == b] for n, b in self.part_body.items()}
        self.weld = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"stick_{n}") for n, _ in self.pool}
        self.park_pose = {n: self.d.qpos[self.part_q[n]:self.part_q[n] + 7].copy() for n, _ in self.pool}
        self.coupler = {(i, k): mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"{'i' if k == 'inner' else ''}coupler{i}")
                        for i in range(C.N_PRINTERS) for k in ("outer",)}
        self.link6 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "link_6")
        self.in_use = set()
        self.robot_bodies = self._subtree("carriage")
        self.robot_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in self.robot_bodies}
        self.finger_bodies = self._subtree("link_left_finger") | self._subtree("link_right_finger")
        self.arm_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] in self._subtree("arm")}
        self.t_sim = 0.0
        self.latch_f0, self.latch_g0 = LATCH_F0, LATCH_G0      # per-instance so robustness sweeps can vary them
        self.rng = np.random.default_rng(0)                     # sensor noise
        # trap-door totes (two rows): flap joint, actuator
        self.flaps = scene.flap_slots()
        self.flap_q = {s: m.jnt_qposadr[jid(f"flap_{s}")] for s in self.flaps}
        self.flap_act = {s: aid(f"flap_{s}") for s in self.flaps}
        # shipping boxes: three prescribed slide joints each (World.step sets their qpos/qvel every physics step)
        self.box_size = dict(scene.box_pool()) if C.TWO_ROWS else {}
        self.box_park_pos = {n: np.array(scene.box_park(k)) for k, n in enumerate(self.box_size)}
        self.box_q = {n: m.jnt_qposadr[jid(f"{n}_x")] for n in self.box_size}
        self.box_v = {n: m.jnt_dofadr[jid(f"{n}_x")] for n in self.box_size}
        self.box_body = {n: m.body(n).id for n in self.box_size}
        self.box_geoms = {n: [g for g in range(m.ngeom) if m.geom_bodyid[g] == b and m.geom_contype[g]]
                          for n, b in self.box_body.items()}
        self.box_lid = {n: [m.geom(f"{n}_lid").id, m.geom(f"{n}_tape").id] for n in self.box_size}
        self.lifts = {n: m.body_mocapid[m.body(n).id] for n in ("lift_dump", "lift_place")} if C.TWO_ROWS else {}
        self.reset()

    def _subtree(self, name):
        root = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)
        out = {root}
        for b in range(self.m.nbody):
            p = b
            while p > 0:
                if p == root:
                    out.add(b)
                    break
                p = self.m.body_parentid[p]
        return out

    # ---------------- state ----------------
    def reset(self, rail_x=1.0):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.rail_q] = rail_x
        self.d.qpos[self.arm_q] = HOME
        self.d.ctrl[self.act_rail] = rail_x
        self.d.ctrl[self.act_arm] = HOME
        self.d.ctrl[self.act_grip] = GRIP_OPEN
        for n, _ in self.pool:
            self.park(n)
        for s in self.flaps:
            self.d.ctrl[self.flap_act[s]] = FLAP_SHUT
        self.box_segs, self.box_hold, self.box_live = {}, {}, set()
        for n in self.box_size:
            self.box_park(n)
        self._lift_tops()
        mujoco.mj_forward(self.m, self.d)
        # robot-side door knowledge: angle estimate from arm FK while holding the handle (NaN = none since the
        # door was last seen latched) and the reed switches. Ground truth (door_angle) is for reporting only.
        self.door_est = np.full(C.N_PRINTERS, np.nan)
        self.reed = [self.door_angle(i) * LATCH_R < REED_GAP for i in range(C.N_PRINTERS)]

    def park(self, name):
        q, v = self.part_q[name], self.part_v[name]
        self.d.qpos[q:q + 7] = self.park_pose[name]
        self.d.qvel[v:v + 6] = 0
        b = self.part_body[name]
        self.m.body_gravcomp[b] = 1.0
        for g in self.part_geoms[name]:
            self.m.geom_contype[g] = self.m.geom_conaffinity[g] = 0
        self.d.eq_active[self.weld[name]] = 0
        self.in_use.discard(name)

    def spawn(self, name, pos, yaw, stuck=True):
        """Place a part on a bed (world pos of its footprint centre at bed level)."""
        q, v = self.part_q[name], self.part_v[name]
        self.d.qpos[q:q + 3] = pos
        self.d.qpos[q + 3:q + 7] = _yaw_quat(yaw)
        self.d.qvel[v:v + 6] = 0
        b = self.part_body[name]
        self.m.body_gravcomp[b] = 0.0
        for g in self.part_geoms[name]:
            self.m.geom_contype[g] = self.m.geom_conaffinity[g] = 1
        self.in_use.add(name)
        mujoco.mj_forward(self.m, self.d)
        self.set_stuck(name, stuck)

    def set_stuck(self, name, stuck):
        """Bed adhesion: weld the part to the world at its current pose while the bed is hot."""
        e = self.weld[name]
        if stuck:
            mujoco.mj_forward(self.m, self.d)
            b = self.part_body[name]
            # relpose = pose of body2 (world) expressed in body1 (part): inverse of the part's world pose
            qinv = np.zeros(4)
            mujoco.mju_negQuat(qinv, self.d.xquat[b])
            pinv = np.zeros(3)
            mujoco.mju_rotVecQuat(pinv, -self.d.xpos[b], qinv)
            self.m.eq_data[e, :3] = 0
            self.m.eq_data[e, 3:6] = pinv
            self.m.eq_data[e, 6:10] = qinv
            self.m.eq_data[e, 10] = 1.0
        self.d.eq_active[e] = 1 if stuck else 0

    def couple(self, i, on, which="outer"):
        """Optional modelled door coupler: a rigid weld between link_6 and printer i's handle bar while engaged.
        Not used by default: plan/door.py holds the bar with the jaws alone (COUPLING = "physical")."""
        e = self.coupler[(i, which)]
        if on:
            mujoco.mj_forward(self.m, self.d)
            b1, b2 = self.link6, self.m.body(f"{'i' if which == 'inner' else ''}pin{i}").id
            q1inv = np.zeros(4)
            mujoco.mju_negQuat(q1inv, self.d.xquat[b1])
            rel_p = np.zeros(3)
            mujoco.mju_rotVecQuat(rel_p, self.d.xpos[b2] - self.d.xpos[b1], q1inv)
            rel_q = np.zeros(4)
            mujoco.mju_mulQuat(rel_q, q1inv, self.d.xquat[b2])
            self.m.eq_data[e, :3] = 0
            self.m.eq_data[e, 3:6] = rel_p
            self.m.eq_data[e, 6:10] = rel_q
            self.m.eq_data[e, 10] = 1.0
        self.d.eq_active[e] = 1 if on else 0

    def free_part(self, pool_sku):
        for n, s in self.pool:
            if s == pool_sku and n not in self.in_use:
                return n
        raise RuntimeError(f"part pool exhausted for {pool_sku}")

    def part_pose(self, name):
        b = self.part_body[name]
        return self.d.xpos[b].copy(), self.d.xquat[b].copy()

    def door_angle(self, i):
        """Ground-truth door angle (rad). For reporting and test setup only; the robot uses door_belief()."""
        return float(self.d.qpos[self.door_q[i]])

    def _read_reed(self, i):
        g = self.door_angle(i) * LATCH_R
        self.reed[i] = g < REED_GAP if not self.reed[i] else g < REED_GAP + REED_HYST
        if self.reed[i]:
            self.door_est[i] = np.nan          # latched: any older FK estimate is obsolete
        return self.reed[i]

    def reed_closed(self, i):
        """Reed switch at the magnet: True when the door is latched (gap below REED_GAP, with hysteresis)."""
        return self._read_reed(i)

    def door_belief(self, i):
        """The robot's estimate of door i's angle (rad), from sensors only: 0 while the reed switch reports it
        latched; else the last arm-FK estimate taken while holding the handle; if there is none (the door was
        opened by someone else), the nominal open angle."""
        if self._read_reed(i):
            return 0.0
        e = self.door_est[i]
        return C.DOOR_OPEN_RAD if not np.isfinite(e) else float(e)

    def latch_torque(self, a):
        """Magnet torque on a door at angle a (rad): pulls toward closed, strongest at contact."""
        g = max(a, 0.0) * LATCH_R
        if g > LATCH_GMAX or self.latch_f0 <= 0:
            return 0.0
        return -self.latch_f0 / (1.0 + g / self.latch_g0) ** 2 * LATCH_R

    # ---------------- trap-door totes ----------------
    def flap_angle(self, slot):
        """Ground-truth flap angle (rad, 0 = shut). Reporting and tests only; the robot reads flap_sensor()."""
        return float(self.d.qpos[self.flap_q[slot]])

    def flap_sensor(self, slot):
        """Flap hinge encoder (rad), with noise."""
        return self.flap_angle(slot) + float(self.rng.normal(0.0, FLAP_ENC_SIGMA))

    def flap_shut(self, slot):
        return self.flap_sensor(slot) < C.FLAP_CLOSED_RAD

    # ---------------- shipping boxes (kinematic) ----------------
    # A box is moved by prescribing its three slide joints (position AND velocity) before every physics step, along
    # smooth accel-limited segments. A mocap body would not do: MuJoCo gives mocap bodies no velocity, so friction
    # does not carry loose contents and the walls sweep through them (tested: a part slid 0.6 m out of a mocap box
    # moved 0.6 m in 2.5 s; with prescribed joints it stayed within 2 mm). Contact forces never move a box.
    def box_spawn(self, name, pos):
        b = self.box_body[name]
        self.m.body_gravcomp[b] = 0.0
        for g in self.box_geoms[name]:
            self.m.geom_contype[g], self.m.geom_conaffinity[g] = 2, 1
        self.box_segs[name] = []
        self.box_hold[name] = np.array(pos, float)
        self.box_live.add(name)
        self._prescribe(name, self.box_hold[name], np.zeros(3))
        mujoco.mj_forward(self.m, self.d)

    def box_park(self, name):
        b = self.box_body[name]
        self.m.body_gravcomp[b] = 1.0
        for g in self.box_geoms[name]:
            self.m.geom_contype[g] = self.m.geom_conaffinity[g] = 0
        self.box_seal(name, False)
        self.box_live.discard(name)
        self.box_segs.pop(name, None)
        self.box_hold.pop(name, None)
        self._prescribe(name, self.box_park_pos[name], np.zeros(3))

    def box_seal(self, name, on=True):
        """Render-only: show (or hide) the lid and tape."""
        for g in self.box_lid[name]:
            self.m.geom_rgba[g, 3] = 1.0 if on else 0.0

    def box_pos(self, name):
        """Commanded position of the box (bottom centre, world): the station controller's own knowledge."""
        return self.box_hold[name].copy()

    def box_move(self, name, waypoints, on_done=None, t0=None):
        """Queue straight moves through waypoints (each: smooth, peak speed BOX_V, peak acceleration BOX_A) after the
        moves already queued; on_done(world) runs when the last one ends. Returns the end time (sim s)."""
        segs = self.box_segs[name]
        t = max(self.d.time if t0 is None else t0, segs[-1][1] if segs else 0.0)
        p = self.box_goal(name)
        for q in waypoints:
            q = np.array(q, float)
            dist = float(np.linalg.norm(q - p))
            if dist < 1e-6:
                continue
            dur = max(1.875 * dist / C.BOX_V, math.sqrt(5.774 * dist / C.BOX_A), 0.05)
            segs.append((t, t + dur, p, q, None))
            t, p = t + dur, q
        if on_done is not None and segs:
            a, b, p0, p1, _ = segs[-1]
            segs[-1] = (a, b, p0, p1, on_done)
        elif on_done is not None:
            on_done(self)
        return t

    def box_dwell(self, name, seconds, on_done=None):
        """Hold still for `seconds` after the queued moves (e.g. in the sealer). Returns the end time."""
        segs = self.box_segs[name]
        t = max(self.d.time, segs[-1][1] if segs else 0.0)
        p = self.box_goal(name)
        segs.append((t, t + seconds, p, p, on_done))
        return t + seconds

    def box_goal(self, name):
        """Where the box will be once its queued moves are done."""
        segs = self.box_segs[name]
        return (segs[-1][3] if segs else self.box_hold[name]).copy()

    def box_busy(self, name):
        return bool(self.box_segs.get(name))

    def _prescribe(self, name, p, v):
        q, dv = self.box_q[name], self.box_v[name]
        self.d.qpos[q:q + 3] = p - self.box_park_pos[name]
        self.d.qvel[dv:dv + 3] = v

    def _kinematics(self):
        t = self.d.time
        done = []
        for n in self.box_live:
            segs = self.box_segs[n]
            while segs and t >= segs[0][1]:
                a, b, p0, p1, cb = segs.pop(0)
                self.box_hold[n] = p1
                if cb is not None:
                    done.append(cb)
            if segs and t >= segs[0][0]:
                a, b, p0, p1, _ = segs[0]
                s = (t - a) / (b - a)
                p = p0 + (s * s * s * (10 - 15 * s + 6 * s * s)) * (p1 - p0)
                v = (30 * s * s * (1 - s) ** 2) / (b - a) * (p1 - p0)
                self.box_hold[n] = p
            else:
                p, v = self.box_hold[n], np.zeros(3)
            self._prescribe(n, p, v)
        for cb in done:
            cb(self)
        self._lift_tops()

    def _lift_tops(self):
        """Lift tables (visual) follow the bottom of a box standing on them."""
        if not self.lifts:
            return
        L = scene.station_layout() if not hasattr(self, "_stn") else self._stn
        self._stn = L
        for lift, mid in self.lifts.items():
            a, b = L[lift + "_y"]
            top = L["z_conv"]
            for n in self.box_live:
                p = self.box_hold[n]
                if abs(p[0] - L["x_drop"]) < 0.02 and a < p[1] < b and top - 0.001 < p[2] < top + 0.3:
                    top = max(top, p[2])
            self.d.mocap_pos[mid] = [L["x_drop"], (a + b) / 2, top - 0.002]

    # ---------------- stepping ----------------
    def step(self, n=1):
        for _ in range(n):
            for i in range(C.N_PRINTERS):          # magnetic catch near closed
                self.d.qfrc_applied[self.door_v[i]] = self.latch_torque(self.d.qpos[self.door_q[i]])
            if self.box_live:
                self._kinematics()
            mujoco.mj_step(self.m, self.d)
        for i in range(C.N_PRINTERS):              # reed switches (sampled at the control rate)
            self._read_reed(i)
        self.t_sim = self.d.time

    def contacts_robot_env(self, ignore_bodies=()):
        """Contacts between robot geoms and anything else (excluding fingers vs ignore_bodies)."""
        out = []
        for c in self.d.contact[:self.d.ncon]:
            g1, g2 = c.geom1, c.geom2
            r1, r2 = g1 in self.robot_geoms, g2 in self.robot_geoms
            if r1 == r2:
                continue
            other = g2 if r1 else g1
            ob = self.m.geom_bodyid[other]
            if ob in ignore_bodies:
                continue
            out.append((g1, g2, c.dist))
        return out
