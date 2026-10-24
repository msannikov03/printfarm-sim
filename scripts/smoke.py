"""M1 smoke: build, spawn a few parts on printer 2, step 1000x, check stability, render views."""
import numpy as np
from farm import catalog as C
from farm.world import World
from farm.render import Cam, save

w = World()
spawned = []
for k, (sku, xy) in enumerate([("knob", (60, 60)), ("block", (150, 120)), ("bracket", (200, 200)), ("cube", (100, 180))]):
    n = w.free_part(sku)
    w.spawn(n, C.bed_mm_to_world(2, *xy), yaw=0.3 * k, stuck=(k % 2 == 0))
    spawned.append(n)
p0 = {n: w.part_pose(n)[0] for n in spawned}
w.d.qpos[w.door_q[7]] = 1.2
w.step(1000)
for n in spawned:
    print(n, "drift mm", np.round(1000 * np.linalg.norm(w.part_pose(n)[0] - p0[n]), 2))
print("rail", w.d.qpos[w.rail_q], "arm", np.round(w.d.qpos[w.arm_q], 3), "door7", round(w.door_angle(7), 3))
print("ncon", w.d.ncon, "robot-env contacts", len(w.contacts_robot_env()))
cam = Cam(w.m, 1600, 900)
save(cam.rgb(w.d, "overview"), "runs/m1_overview.png")
save(cam.rgb(w.d, "aisle"), "runs/m1_aisle.png")
xc, s = C.printer_frame(2)
save(cam.look([xc, 0.35, 0.85], 1.1, -60, -30).rgb(w.d), "runs/m1_printer2.png")
save(cam.rgb(w.d, "chamber2"), "runs/m1_chamber2.png")
save(cam.rgb(w.d, "wrist"), "runs/m1_wrist.png")
