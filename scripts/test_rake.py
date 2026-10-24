"""Rake one part off the bed into the reject tray; report where it lands.
python scripts/test_rake.py cube 150 80 [--video] [--yaw DEG]   (bed frame; default 17 deg)"""
import sys, numpy as np, mujoco
from farm import catalog as C
from farm.world import World
from plan.ik import IK
from plan.motion import Executor
from plan.door import TUCK
from plan.sweep import blobs, in_tray, rake
from plan.unload import bed_camera, on_bed, scene_belief
import plan.door as D
D.VERBOSE = "--verbose" in sys.argv
from farm.recorder import Recorder
sku, bx, by = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]); P = 4
w = World(); ik = IK(w); xc, s = C.printer_frame(P)
w.reset(rail_x=xc + 0.35); w.d.qpos[w.arm_q] = TUCK; w.d.ctrl[w.act_arm] = TUCK
w.d.qpos[w.door_q[P]] = C.DOOR_OPEN_RAD; mujoco.mj_forward(w.m, w.d)
yaw = np.radians(float(sys.argv[sys.argv.index("--yaw") + 1])) if "--yaw" in sys.argv else 0.3
b = w.free_part(sku); w.spawn(b, C.bed_mm_to_world(P, bx, by), yaw, stuck=False); w.step(100)
rec = Recorder(w.m, f"runs/rake_{sku}.mp4", view=lambda w: ([xc, 0.2, 0.85], 0.9, 90 + 50, -30)) if "--video" in sys.argv else None
ex = Executor(w, rec)
mujoco.mj_forward(w.m, w.d)
pts = bed_camera(w, P).points_bed_mm(); bl = blobs(pts); ik.belief = scene_belief(w, P, {}, {}, pts)   # plan with what the camera sees
print("blobs", [(np.round(x["lo"]), np.round(x["hi"]), round(x["h"], 1), x["n"]) for x in bl])
for rnd in range(3):                     # as unload_parts: re-scan and rake again until the bed is clear
    ok, sec = rake(ex, ik, P, bl[0])
    ex.wait(0.5)
    from plan.unload import part_in_bin
    print(f"round {rnd}:", sku, "raked" if ok else "NO PLAN", round(sec, 1), "s; on bed", on_bed(w, P, b),
          "in tray", in_tray(w, P, b), "in reject bin", part_in_bin(w, b, "reject"), "pos",
          np.round(w.part_pose(b)[0], 3), "door plane y", round(C.AISLE_HALF, 3))
    mujoco.mj_forward(w.m, w.d)
    pts = bed_camera(w, P).points_bed_mm(); bl = blobs(pts); ik.belief = scene_belief(w, P, {}, {}, pts)
    if not ok or not bl:
        break
if rec: rec.close()
