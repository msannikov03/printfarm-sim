import numpy as np
from farm import catalog as C
from farm.world import World
from plan.ik import IK
from plan.door import plan_arc
w = World(); ik = IK(w); P = 4; xc, s = C.printer_frame(P); ik.sync(w.d)
for deg in (75, 85, 95):
    op = plan_arc(ik, w, P, 0.0, np.radians(deg), n_seeds=8)
    cl = plan_arc(ik, w, P, np.radians(deg), 0.0, n_seeds=8) if op else None
    print(deg, "open", None if not op else (np.round(np.array(op["xs"])[[0, -1]] - xc, 2), round(op["score"], 4)),
          "close", None if not cl else (np.round(np.array(cl["xs"])[[0, -1]] - xc, 2), round(cl["score"], 4)), flush=True)
