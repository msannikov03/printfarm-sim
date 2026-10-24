import numpy as np, mujoco, itertools
from farm import catalog as C
from farm.world import World
from plan.ik import IK
w = World(); ik = IK(w); ik.sync(w.d)
bodies = [b for b in w.robot_bodies if w.m.body(b).name.startswith("link")]
best = []
for j2, j3, j4 in itertools.product(np.linspace(0, 3.6, 19), np.linspace(0, 3.6, 19), np.linspace(-1.5, 1.5, 7)):
    q = np.array([0.0, j2, j3, j4, 0.0, 0.0])
    ik.fk(q, 0.0); mujoco.mj_forward(w.m, ik.d)
    pts = np.array([ik.d.xpos[b] for b in bodies] + [ik.d.site_xpos[w.grasp_site]])
    r = np.max(np.linalg.norm(pts[:, :2] - [0, 0], axis=1))
    zmax = np.max(pts[:, 2])
    if not ik.collides(q, 0.0):
        best.append((r, zmax, q))
best.sort(key=lambda t: (round(t[0], 2), t[1]))
for r, z, q in best[:6]:
    print(f"radius {r:.3f} m  top {z:.2f} m  q={np.round(q, 2)}")
