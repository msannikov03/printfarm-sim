"""RRT-Connect in the arm's joint space (rail fixed) with shortcut smoothing.

Collision checks run on the IK helper's scratch data, synced to the live sim, so open doors, parts on the bed
and the caddy are all obstacles.
"""
import numpy as np

STEP = 0.12          # rad, tree extension step
EDGE_RES = 0.04      # rad, collision-check resolution along an edge


RAIL_W = 2.0         # 1 m of rail counts like 2 rad of joint motion in the metric


class JointPlanner:
    """Plans in arm joint space; with rail_bounds the carriage position is a 7th planning dimension
    (state = [q1..q6, rail_x * RAIL_W])."""

    def __init__(self, ik, rng=None, rail_bounds=None):
        self.ik = ik
        self.rng = rng or np.random.default_rng(0)
        self.rail = rail_bounds is not None
        self._escape = []
        self.lo, self.hi = ik.lo, ik.hi
        if self.rail:
            self.lo = np.r_[self.lo, rail_bounds[0] * RAIL_W]
            self.hi = np.r_[self.hi, rail_bounds[1] * RAIL_W]

    ESCAPE = 0.2         # if the start sits inside the clearance margin, allow this much motion to escape it
    START_TOL = -0.003   # the robot IS at its start state: model error may show it up to 3 mm "inside" something

    def free(self, v, ctx, clearance=None):
        q, x = (v[:6], v[6] / RAIL_W) if self.rail else (v, ctx["x"])
        if clearance is None:
            for e, c in self._escape:    # near a tight start/goal (e.g. inside a printer): fine clearance only
                if np.max(np.abs(v - e)) < self.ESCAPE:
                    clearance = c if clearance is None else min(clearance, c)
        return not self.ik.collides(q, x, ctx.get("doors"), grip=ctx.get("grip", 0.037),
                                    ignore_bodies=ctx.get("ignore", ()), clearance=clearance)

    def edge_free(self, a, b, ctx):
        n = max(int(np.max(np.abs(b - a)) / EDGE_RES), 1)
        return all(self.free(a + (b - a) * k / n, ctx) for k in range(1, n + 1))

    def plan(self, q0, q1, ctx, max_iters=1500):
        """Returns a list of joint waypoints from q0 to q1 (inclusive), or None."""
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        self._escape = []
        for e in (q0, q1):           # endpoints inside the margin but with fine clearance get an escape zone
            if not self.free(e, ctx) and self.free(e, ctx, clearance=0.003):
                self._escape.append((e, 0.003))
        if not self._escape or self._escape[0][0] is not q0:
            if not self.free(q0, ctx) and self.free(q0, ctx, clearance=self.START_TOL):
                self._escape.insert(0, (q0, self.START_TOL))     # calibration error, not a real collision
        if not self.free(q1, ctx):
            return None
        if self.edge_free(q0, q1, ctx):
            return [q0, q1]
        trees = [([q0], [-1]), ([q1], [-1])]          # (nodes, parents); tree 0 grows from the start
        a = 0
        for it in range(max_iters):
            (ta, pa), (tb, pb) = trees[a], trees[1 - a]
            goal = tb[0]
            target = goal if self.rng.random() < 0.1 else self.rng.uniform(self.lo, self.hi)
            new = self._extend(ta, pa, target, ctx)
            if new is not None and self._connect(tb, pb, ta[new], ctx):
                half_a = self._trace(ta, pa, new)[::-1]          # root_a ... meeting point
                half_b = self._trace(tb, pb, len(tb) - 1)        # meeting point ... root_b
                path = half_a + half_b[1:]
                if a == 1:
                    path = path[::-1]
                return self.shortcut(path, ctx)
            a = 1 - a
        return None

    def _nearest(self, tree, q):
        return int(np.argmin([np.sum((t - q) ** 2) for t in tree]))

    def _extend(self, tree, parent, target, ctx):
        k = self._nearest(tree, target)
        d = target - tree[k]
        dist = np.linalg.norm(d)
        new = target if dist <= STEP else tree[k] + d / dist * STEP
        if self.edge_free(tree[k], new, ctx):
            tree.append(new)
            parent.append(k)
            return len(tree) - 1
        return None

    def _connect(self, tree, parent, target, ctx):
        while True:
            n = len(tree)
            k = self._extend(tree, parent, target, ctx)
            if k is None:
                return False
            if np.allclose(tree[k], target):
                return True
            if len(tree) == n:
                return False

    @staticmethod
    def _trace(tree, parent, k):
        out = []
        while k != -1:
            out.append(tree[k])
            k = parent[k]
        return out

    def shortcut(self, path, ctx, tries=60):
        path = [np.asarray(p) for p in path]
        for _ in range(tries):
            if len(path) <= 2:
                break
            i, j = sorted(self.rng.choice(len(path), 2, replace=False))
            if j - i < 2:
                continue
            if self.edge_free(path[i], path[j], ctx):
                path = path[:i + 1] + path[j:]
        return path
