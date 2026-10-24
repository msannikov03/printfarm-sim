"""Pick one SKU from the bed centre and drop it into bin o1; report where it lands."""
import sys, numpy as np, mujoco
from farm import catalog as C
from farm.world import World
from plan.ik import IK
from plan.motion import Executor
import plan.door as D
from plan.door import TUCK
from plan.unload import pick_and_place, part_center
from farm.recorder import Recorder
sku = sys.argv[1]; P = 4
D.VERBOSE = True
w = World(); ik = IK(w); xc, s = C.printer_frame(P)
w.reset(rail_x=xc + 0.35); w.d.qpos[w.arm_q] = TUCK; w.d.ctrl[w.act_arm] = TUCK
w.d.qpos[w.door_q[P]] = C.DOOR_OPEN_RAD; mujoco.mj_forward(w.m, w.d)
b = w.free_part(sku); w.spawn(b, C.bed_mm_to_world(P, 150, 80), 0.0, stuck=False); w.step(100)
rec = Recorder(w.m, f"runs/drop_{sku}.mp4", view=lambda w: ([xc + 0.2, -0.25, 0.75], 0.9, 90 + 120, -35)) if "--video" in sys.argv else None
ex = Executor(w, rec)
ok, outcome, sec, _ = pick_and_place(ex, ik, P, b, "o1")
print(sku, outcome, round(sec, 1), "final part pos", np.round(w.part_pose(b)[0], 3))
if rec: rec.close()
