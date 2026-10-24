"""Find a folded arm pose whose every collision geom stays below the door's bottom edge (so the carriage can
drive under an open door), for a given base height."""
import itertools, sys
import numpy as np, mujoco
from farm import catalog as C
base_z = float(sys.argv[1]) if len(sys.argv) > 1 else C.BASE_Z
C.BASE_Z = C.DECK_Z = base_z
from farm.world import World
from plan.ik import IK
w = World(); ik = IK(w); ik.sync(w.d)
from farm.scene import DOOR_CLEAR
door_bottom = C.BENCH_H + C.DOOR_Z0 + DOOR_CLEAR          # leaf bottom edge (with its clearance to the plinth)
geoms = sorted(w.arm_geoms)
def top_and_radius(q):
    ik.fk(q, 0.0); mujoco.mj_forward(ik.m, ik.d)
    zt, r, rx = -1, 0, 0
    for g in geoms:
        if not ik.m.geom_contype[g]:
            continue
        R = ik.d.geom_xmat[g].reshape(3, 3); c = ik.d.geom_xpos[g]; sz = ik.m.geom_size[g]
        if ik.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            ext = np.abs(R[:, 2]) * sz[1] + sz[0]
        elif ik.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            ext = np.abs(R) @ sz
        else:
            ext = np.full(3, sz[0])
        zt = max(zt, c[2] + ext[2]); r = max(r, abs(c[1]) + ext[1]); rx = max(rx, abs(c[0]) + ext[0])
    return zt, max(r, rx)
best = []
for j1, j2, j3, j4, j5 in itertools.product((0.0, 1.57, -1.57), np.linspace(0, 3.6, 13), np.linspace(0, 3.6, 13), np.linspace(-1.5, 1.5, 5), (0.0, 1.5)):
    q = np.array([j1, j2, j3, j4, j5, 0.0])
    zt, ry = top_and_radius(q)
    if zt < door_bottom - 0.01 and not ik.collides(q, 0.0, None, clearance=0.0):
        best.append((zt, ry, q))
best.sort(key=lambda t: (t[1], t[0]))
print(f"base {base_z}: door bottom {door_bottom:.3f}; low poses found: {len(best)} (sorted by max horizontal extent)")
for zt, ry, q in best[:4]:
    print(f"  top {zt:.3f} m, max horizontal extent {ry:.3f} m, q={np.round(q, 2)}")
