"""Combined layout check: door arc (open+close) feasibility + bed reach at 6 key points, per layout."""
import itertools, importlib, sys, numpy as np, multiprocessing as mp
def run(cfg):
    base, aisle, pin_z = cfg
    from farm import catalog as C
    C.BASE_Z, C.AISLE_HALF, C.PIN_ZL = base, aisle, pin_z
    import farm.scene, farm.world, plan.door, plan.reach
    for m in (farm.scene, farm.world, plan.door, plan.reach): importlib.reload(m)
    from farm.world import World
    from plan.ik import IK
    from plan.door import plan_arc
    import plan.reach as R
    w = World(); ik = IK(w); ik.sync(w.d)
    R._W, R._IK = w, ik; R.P = 4
    xc, s = C.printer_frame(4)
    op = plan_arc(ik, w, 4, 0.0, C.DOOR_OPEN_RAD, n_seeds=5)
    cl = plan_arc(ik, w, 4, C.DOOR_OPEN_RAD, 0.0, n_seeds=5) if op else None
    pts = [(20, 30), (236, 30), (128, 128), (20, 236), (236, 236), (128, 236)]
    reach = []
    for bx, by in pts:
        p = np.array(C.bed_mm_to_world(4, bx, by, 12)); got = None
        for pitch in R.PITCHES:
            b = R.best_option(p, R.grasp_frames(s, pitch), xc, dx_set=np.round(np.arange(-0.2, 0.41, 0.1), 2))
            if b: got = pitch; break
        reach.append(got)
    return cfg, op is not None, cl is not None, reach
if __name__ == "__main__":
    cfgs = list(itertools.product((0.62, 0.70), (0.22, 0.28), (0.22, 0.32)))
    with mp.Pool(8) as pool:
        for cfg, o, c, r in pool.imap_unordered(run, cfgs):
            print(f"base={cfg[0]} aisle={cfg[1]} pin_z={cfg[2]}  door_open={o} door_close={c}  reach(tilt per pt)={r}", flush=True)
