"""M4: full unload of one printer: open door -> purge tab to waste -> each part to its order bin -> close door."""
import sys, time
import numpy as np, mujoco
from farm import catalog as C
from farm.world import World
from farm.recorder import Recorder
from jobs.plates import make_plate
from plan.ik import IK
from plan.motion import Executor
from plan.door import operate_door, TUCK
import plan.door as D
from plan.unload import spawn_plate, pick_and_place

P = 4
seed = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0
video = "--video" in sys.argv
D.VERBOSE = "--verbose" in sys.argv
rng = np.random.default_rng(seed)
skus = list(C.CATALOG)
parts = [{"name": f"o{k // 2}_{sku}_{k}", "order_id": f"o{k // 2}", "sku": sku}
         for k, sku in enumerate(rng.choice(skus, 6, replace=False))]
plate, gcode = make_plate(parts, P, f"PL{seed:03d}", rng)
from plan.unload import mirror_plate
plate = mirror_plate(plate)
print("plate:", [(o["sku"], o["center_mm"], o["yaw_deg"]) for o in plate["objects"]], "purge", plate["purge_tab"]["center_mm"])

w = World(); ik = IK(w); xc, s = C.printer_frame(P)
w.reset(rail_x=xc + 0.4)
w.d.qpos[w.arm_q] = TUCK; w.d.ctrl[w.act_arm] = TUCK; mujoco.mj_forward(w.m, w.d)
placed = spawn_plate(w, plate, rng, stuck=False)
w.step(200)
rec = Recorder(w.m, f"runs/unload_p{P}_s{seed}.mp4", view=lambda w: ([xc + 0.05, 0.2, 0.9], 1.6, 90 + 40, -22)) if video else None
ex = Executor(w, rec)
t0 = time.time()
ok, dt, note = operate_door(ex, ik, P, opening=True)
print(f"door open: {ok} {note} {np.degrees(w.door_angle(P)):.1f} deg, {dt:.1f}s")
slots = {}
free_slots = ["o0", "o1", "o2", "o3", "o4", "o5"]
results = []
remaining = dict(placed)
from plan.unload import plan_grasp
while remaining:
    ik.sync(w.d)
    # dynamic order: purge tab first, then any part that currently has a feasible grasp
    names = sorted(remaining, key=lambda n: (0 if n.endswith("purge_tab") else 1, n))
    name = next((n for n in names if plan_grasp(ik, w, P, remaining[n]) is not None), None)
    if name is None:
        for n in names:
            results.append((n, "-", "no_grasp", 0.0)); print(f"  {n:22s} no feasible grasp")
        break
    body = remaining.pop(name)
    if name.endswith("purge_tab"):
        slot = "waste"
    else:
        oid = name.split("_")[0]
        if oid not in slots:
            slots[oid] = free_slots.pop(0)
        slot = slots[oid]
    ok, outcome, sec, att = pick_and_place(ex, ik, P, body, slot, rng)
    results.append((name, slot, outcome, round(sec, 1)))
    print(f"  {name:22s} -> {slot:6s} {outcome:16s} {sec:5.1f}s", flush=True)
okc, dtc, notec = operate_door(ex, ik, P, opening=False)
print(f"door close: {okc} {notec} {np.degrees(w.door_angle(P)):.2f} deg")
n_ok = sum(r[2] == "placed" for r in results)
print(f"RESULT placed {n_ok}/{len(results)}, sim time {w.d.time:.0f}s, wall {time.time() - t0:.0f}s")
if rec:
    rec.close()
