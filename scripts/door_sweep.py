"""Quick design sweep of the handle position for the whole-body door arc."""
import sys, itertools, numpy as np
from farm import catalog as C
results = []
for pin_s, pin_z in itertools.product((0.12, 0.17, 0.22), (0.22, 0.32)):
    C.PIN_S, C.PIN_ZL = pin_s, pin_z
    import importlib, farm.scene, farm.world, plan.door
    importlib.reload(farm.scene); importlib.reload(farm.world); importlib.reload(plan.door)
    from farm.world import World
    from plan.ik import IK
    from plan.door import plan_arc
    w = World(); ik = IK(w); ik.sync(w.d)
    op = plan_arc(ik, w, 4, 0.0, C.DOOR_OPEN_RAD, n_seeds=4)
    xc, _ = C.printer_frame(4)
    cl = None
    if op:
        cl = plan_arc(ik, w, 4, C.DOOR_OPEN_RAD, 0.0, n_seeds=4)
    print(pin_s, pin_z, "OPEN", None if not op else (round(op["xs"][0] - xc, 2), round(op["xs"][-1] - xc, 2), round(op["score"], 4)),
          "CLOSE", None if not cl else (round(cl["xs"][0] - xc, 2), round(cl["xs"][-1] - xc, 2), round(cl["score"], 4)), flush=True)
