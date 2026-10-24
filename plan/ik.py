"""Damped-least-squares IK for the YAM grasp_site, optionally with the rail as a 7th axis.

Works on a scratch MjData so planning never disturbs the simulation.
Grasp frame convention (grasp_site): z = approach direction (out of the gripper), y = finger closing direction.
"""
import mujoco
import numpy as np

from farm.world import HOME


def frame(approach, closing):
    """Rotation matrix with z=approach, y=closing (orthogonalised)."""
    z = np.asarray(approach, float)
    z /= np.linalg.norm(z)
    y = np.asarray(closing, float) - np.dot(closing, z) * z
    y /= np.linalg.norm(y)
    return np.column_stack([np.cross(y, z), y, z])


class Belief:
    """The robot's planning scene for movable things (everything static comes from the model).

    parts: pool body name -> (world position of the footprint centre at bed level, yaw quaternion, pad in m):
           parts the camera found on the bed (estimated pose, pad from the estimate's uncertainty) and parts the
           robot itself put into a caddy bin (bookkeeping: bin floor centre).
    blobs: unexplained raised regions from the chamber camera (debris, knocked-over parts):
           (centre world xyz at bed level, half sizes (hx, hy, hz) in the bed frame, yaw of the bed frame, pad).
    Everything else (true poses of parts the robot has not seen) is NOT in the planning scene."""

    def __init__(self, parts=None, blobs=None):
        self.parts = dict(parts or {})
        self.blobs = list(blobs or [])


PROXY_SKUS = ("cube", "block", "clip", "lid")      # single-box pool bodies, resized in the planning model as blobs


class IK:
    MARGIN = 0.015          # planning clearance: robot geoms are padded by this much in the planning model

    def __init__(self, world):
        import copy
        self.w = world
        self.m = copy.deepcopy(world.m)          # planning copy: padded robot geometry, never stepped
        for g in world.arm_geoms:
            self.m.geom_margin[g] = self.MARGIN
        self.d = mujoco.MjData(self.m)
        self.cols = np.array(list(world.arm_v))
        self.qadr = np.array(list(world.arm_q))
        self.lo = self.m.jnt_range[[self.m.dof_jntid[v] for v in self.cols], 0]
        self.hi = self.m.jnt_range[[self.m.dof_jntid[v] for v in self.cols], 1]
        self.site = world.grasp_site
        self.held = None                         # (geoms, qpos adr, pos, rot in the pinch frame) of a held part
        self.jp = np.zeros((3, self.m.nv))
        self.jr = np.zeros((3, self.m.nv))
        self.belief = None                       # Belief: plan with what the robot knows; None = sim truth (oracle)
        self.pad = {}                            # body id -> extra clearance (uncertainty) in collision checks
        self.proxies = []                        # [(blob, body id)] currently standing in for unknown blobs
        self._part_geoms = {n: [g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == b]
                            for n, b in world.part_body.items()}
        from farm.scene import deck_extent                                     # carriage deck (caddy area)
        x0, x1, y0, y1 = deck_extent()
        self._deck = (np.array([(x0 + x1) / 2, (y0 + y1) / 2, 0.0]), np.array([(x1 - x0) / 2, (y1 - y0) / 2, 0.0]))
        self._riders, self._x_sync = [], 0.0     # parts lying on the carriage: they ride along with the rail

    def sync(self, data):
        """Planning state <- robot state. Arm, rail, jaws and doors are copied; movable parts come from the
        belief (when set), so the planner never sees the true pose of a part the robot has not measured."""
        self.d.qpos[:] = data.qpos
        if hasattr(self.w, "door_belief"):                    # doors as the robot senses them, not the true angle
            for j, adr in enumerate(self.w.door_q):
                self.d.qpos[adr] = self.w.door_belief(j)
        for slot, adr in getattr(self.w, "flap_q", {}).items():   # tote trap doors from their encoders
            self.d.qpos[adr] = self.w.flap_sensor(slot)
        self.m.geom_contype[:] = self.w.m.geom_contype       # parked/spawned parts toggle these at runtime
        self.m.geom_conaffinity[:] = self.w.m.geom_conaffinity
        self._restore_proxies()
        if self.belief is not None:
            self._apply_belief()
        self._find_riders()

    def _find_riders(self):
        """Parts lying in the caddy bins / totes and the large-part tray (on the carriage deck) move with the carriage.
        collides() tests other rail positions than the synced one, so it shifts them along; otherwise the planner sees
        the arm base drive into parts it has just dropped (with bins beside the arm this blocks transfers). Parts below
        the deck (in a shipping box under its overhang) do not ride."""
        self._riders, self._x_sync = [], float(self.d.qpos[self.w.rail_q])
        if self._deck is None:
            return
        from farm import catalog as C
        (cx, cy, _), (hx, hy, _) = self._deck
        top = C.DECK_Z + C.BIN_SIZE[2] + 0.06
        for n, geoms in self._part_geoms.items():
            if n not in self.w.in_use or not geoms or not self.m.geom_contype[geoms[0]]:
                continue
            a = self.w.part_q[n]
            px, py, pz = self.d.qpos[a:a + 3]
            if abs(px - self._x_sync - cx) <= hx + 0.02 and abs(py - cy) <= hy + 0.02 and C.DECK_Z - 0.01 < pz < top:
                self._riders.append((a, float(px)))

    def _restore_proxies(self):
        """Give last round's blob proxies their pool geometry back."""
        m, w = self.m, self.w
        for _, bid in self.proxies:
            for g in range(m.ngeom):
                if m.geom_bodyid[g] == bid:
                    m.geom_size[g], m.geom_pos[g] = w.m.geom_size[g], w.m.geom_pos[g]
                    m.geom_rbound[g], m.geom_aabb[g] = w.m.geom_rbound[g], w.m.geom_aabb[g]
        self.proxies, self.pad = [], {}

    def _apply_belief(self):
        m, w, bel = self.m, self.w, self.belief
        held = self.held[0] if self.held is not None else set()
        for n, geoms in self._part_geoms.items():
            if geoms and geoms[0] in held:
                continue                                     # the held part is carried by attach()
            a = w.part_q[n]
            if n in bel.parts:
                pos, quat, pad = bel.parts[n]
                self.d.qpos[a:a + 3], self.d.qpos[a + 3:a + 7] = pos, quat
                on, marg = 1, self.MARGIN + pad
                self.pad[w.part_body[n]] = pad
            else:
                self.d.qpos[a:a + 7] = w.park_pose[n]
                on, marg = 0, w.m.geom_margin[geoms[0]] if geoms else 0.0
            for g in geoms:
                m.geom_contype[g] = m.geom_conaffinity[g] = on
                m.geom_margin[g] = marg
        free = [n for n, s in w.pool if s in PROXY_SKUS and n not in w.in_use and n not in bel.parts
                and not (held and self._part_geoms[n][0] in held)]
        for blob, n in zip(bel.blobs, free):
            c, half, yaw, pad = blob
            bid, (g,) = w.part_body[n], self._part_geoms[n]
            a = w.part_q[n]
            self.d.qpos[a:a + 3] = c
            self.d.qpos[a + 3:a + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
            m.geom_size[g] = half
            m.geom_pos[g] = [0.0, 0.0, half[2]]
            m.geom_rbound[g] = float(np.linalg.norm(half))
            m.geom_aabb[g] = [0.0, 0.0, 0.0, *half]
            m.geom_contype[g] = m.geom_conaffinity[g] = 1
            m.geom_margin[g] = self.MARGIN + pad
            self.pad[bid] = pad
            self.proxies.append((blob, bid))

    def attach(self, body, data, rel=None):
        """Plan with the part `body` (pool name) rigidly held relative to the pinch site: its geoms count as robot
        (padded like the arm), so a held part cannot be swept through the environment.
        rel = (position, rotation) of the part's body frame in the pinch frame, as the robot believes it (from the
        planned grasp); None = read the true relative pose from `data` (oracle)."""
        bid = self.w.part_body[body]
        geoms = {g for g in range(self.m.ngeom) if self.m.geom_bodyid[g] == bid}
        for g in geoms:
            self.m.geom_margin[g] = self.MARGIN
            self.m.geom_contype[g] = self.m.geom_conaffinity[g] = 1
        if rel is None:
            p, R = data.site_xpos[self.site], data.site_xmat[self.site].reshape(3, 3)
            Rb = data.xmat[bid].reshape(3, 3)
            rel = (R.T @ (data.xpos[bid] - p), R.T @ Rb)
        self.held = (geoms, self.m.jnt_qposadr[self.m.body_jntadr[bid]], np.asarray(rel[0], float),
                     np.asarray(rel[1], float))

    def detach(self):
        if self.held is not None:
            for g in self.held[0]:
                self.m.geom_margin[g] = self.w.m.geom_margin[g]
        self.held = None

    def fk(self, q_arm, rail_x):
        self.d.qpos[self.qadr] = q_arm
        self.d.qpos[self.w.rail_q] = rail_x
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def solve(self, p_target, R_target, rail_x, q0=None, rail_free=False, rail_bounds=None,
              iters=150, tol_p=1e-3, tol_r=np.radians(1.5), w_rot=0.5, symmetric=True, align_x=None, align_y=None, align_z=None, jaw_horizontal=False, q_pref=None, x_pref=None, k_null=0.15):
        """Returns (q_arm, rail_x, ok, info). symmetric: a parallel gripper may also take R rotated 180 deg about z.
        align_x: if given (unit vector), only the grasp frame's x axis is constrained to it (or its negative).
        jaw_horizontal: only require the jaw-closing axis (y) to be horizontal, e.g. around a vertical bar;
        the other two rotations stay free."""
        targets = [R_target, R_target @ np.diag([-1.0, -1.0, 1.0])] if symmetric else [R_target]
        q = np.array(HOME if q0 is None else q0, float)
        x = float(rail_x)
        cols = np.r_[self.cols, self.w.rail_v] if rail_free else self.cols
        lam = 0.03
        for it in range(iters):
            p, R = self.fk(q, x)
            ep = p_target - p
            if align_z is not None:
                er = np.cross(R[:, 2], align_z)
            elif align_y is not None:
                er = min((np.cross(R[:, 1], v) for v in (align_y, -np.asarray(align_y))), key=np.linalg.norm)
            elif jaw_horizontal:
                y = R[:, 1]
                yh = np.array([y[0], y[1], 0.0])
                er = np.cross(y, yh / max(np.linalg.norm(yh), 1e-9))
            elif align_x is not None:
                er = min((np.cross(R[:, 0], v) for v in (align_x, -np.asarray(align_x))), key=np.linalg.norm)
            else:
                er = min((self._rot_err(Rt, R) for Rt in targets), key=np.linalg.norm)
            if np.linalg.norm(ep) < tol_p and np.linalg.norm(er) < tol_r:
                return q, x, True, {"iters": it, "err_p": np.linalg.norm(ep), "err_r": np.linalg.norm(er)}
            mujoco.mj_comPos(self.m, self.d)
            mujoco.mj_jacSite(self.m, self.d, self.jp, self.jr, self.site)
            J = np.vstack([self.jp[:, cols], w_rot * self.jr[:, cols]])
            e = np.r_[ep, w_rot * er]
            JJt = J @ J.T + lam ** 2 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            if q_pref is not None or x_pref is not None:
                # secondary task in the null space: drift toward a preferred posture / carriage position
                g = np.zeros(J.shape[1])
                if q_pref is not None:
                    g[:6] = np.asarray(q_pref) - q
                if rail_free and x_pref is not None:
                    g[6] = 3.0 * (x_pref - x)
                N = np.eye(J.shape[1]) - J.T @ np.linalg.solve(JJt, J)
                dq = dq + k_null * N @ g
            step = np.clip(dq, -0.25, 0.25)
            q = np.clip(q + step[:6], self.lo, self.hi)
            if rail_free:
                x = float(np.clip(x + step[6], *(rail_bounds or (self.m.jnt_range[self.m.dof_jntid[self.w.rail_v]]))))
        p, R = self.fk(q, x)
        return q, x, False, {"iters": iters, "err_p": float(np.linalg.norm(p_target - p))}

    @staticmethod
    def _rot_err(R_target, R):
        er, quat = np.zeros(3), np.zeros(4)
        mujoco.mju_mat2Quat(quat, (R_target @ R.T).flatten())
        mujoco.mju_quat2Vel(er, quat, 1.0)
        return er

    def manipulability(self, q, rail_x):
        self.fk(q, rail_x)
        mujoco.mj_comPos(self.m, self.d)
        mujoco.mj_jacSite(self.m, self.d, self.jp, self.jr, self.site)
        J = np.vstack([self.jp[:, self.cols], self.jr[:, self.cols]])
        return float(np.sqrt(max(np.linalg.det(J @ J.T), 0.0)))

    def limit_margin(self, q):
        span = self.hi - self.lo
        return float(np.min(np.minimum(q - self.lo, self.hi - q) / span))

    def collides(self, q, rail_x, door_angles=None, grip=0.037, ignore_bodies=(), extra_qpos=None, clearance=None):
        """Robot-vs-environment collision at a configuration (scratch data, full forward pass)."""
        for a, x in self._riders:                 # caddy contents ride with the carriage
            self.d.qpos[a] = x + (rail_x - self._x_sync)
        try:
            return self._collides(q, rail_x, door_angles, grip, ignore_bodies, extra_qpos, clearance)
        finally:
            for a, x in self._riders:
                self.d.qpos[a] = x

    def _collides(self, q, rail_x, door_angles, grip, ignore_bodies, extra_qpos, clearance):
        d = self.d
        d.qpos[self.qadr] = q
        d.qpos[self.w.rail_q] = rail_x
        d.qpos[self.w.finger_q] = grip
        d.qpos[self.w.finger_rq] = -grip          # jaws are coupled: keep the planning state symmetric
        if door_angles is not None:
            for i, a in door_angles.items():
                d.qpos[self.w.door_q[i]] = a
        robot = self.w.arm_geoms
        if self.held is not None:          # carry the held part along with the pinch site
            geoms, adr, pr, Rr = self.held
            mujoco.mj_kinematics(self.m, d)
            Rs = d.site_xmat[self.site].reshape(3, 3)
            d.qpos[adr:adr + 3] = d.site_xpos[self.site] + Rs @ pr
            mujoco.mju_mat2Quat(d.qpos[adr + 3:adr + 7], (Rs @ Rr).ravel())
            robot = robot | geoms
        mujoco.mj_forward(self.m, d)
        for c in d.contact[:d.ncon]:      # only the arm (+ a held part) counts as "robot" (parts in the caddy are fine)
            r1, r2 = c.geom1 in robot, c.geom2 in robot
            if r1 == r2:
                continue
            other = c.geom2 if r1 else c.geom1
            ob = self.m.geom_bodyid[other]
            if c.dist - self.pad.get(ob, 0.0) > (self.MARGIN - 0.003 if clearance is None else clearance):
                continue                   # believed obstacles are inflated by their pose uncertainty (pad)
            if ob in ignore_bodies:
                continue
            return True
        return False
