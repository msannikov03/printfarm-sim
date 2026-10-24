"""PrintPick: build a pick plan from sliced G-code (slicer = prior for perception)."""
import json, re, sys
import numpy as np

GRIPPER_MAX_MM = 70.0   # YAM linear gripper (see farm/catalog.py)
DOOR_Y = 0.0            # door is at the front edge (Y=0), the arm reaches in from there

def parse(path):
    objs, cur, purge = {}, None, []
    x = y = 0.0
    for line in open(path):
        m = re.match(r"EXCLUDE_OBJECT_DEFINE NAME=(\S+) CENTER=([\d.]+),([\d.]+) POLYGON=(\[.*\])", line)
        if m:
            poly = np.array(json.loads(m.group(4)), float)
            objs[m.group(1)] = dict(center=[float(m.group(2)), float(m.group(3))], poly=poly, height=0.0)
            continue
        if (m := re.match(r"EXCLUDE_OBJECT_START NAME=(\S+)", line)):
            cur = m.group(1)
        elif line.startswith("EXCLUDE_OBJECT_END"):
            cur = None
        if line.startswith("G1"):
            xm, ym, zm = (re.search(rf"{a}([-\d.]+)", line) for a in "XYZ")
            nx, ny = (float(xm.group(1)) if xm else x), (float(ym.group(1)) if ym else y)
            if cur and zm:
                objs[cur]["height"] = max(objs[cur]["height"], float(zm.group(1)))
            if "purge" in line:
                purge.append([[x, y], [nx, ny]])
            x, y = nx, ny
    return objs, purge

def purge_tab(path):
    """Bounding box of the '; purge' moves -> {"center_mm": [x, y], "dims_mm": [dx, dy, z_top]}, or None."""
    pts, z_top, x, y = [], 0.0, 0.0, 0.0
    for line in open(path):
        if not line.startswith("G1"):
            continue
        xm, ym, zm = (re.search(rf"{a}([-\d.]+)", line) for a in "XYZ")
        x, y = (float(xm.group(1)) if xm else x), (float(ym.group(1)) if ym else y)
        if "purge" in line:
            pts.append((x, y))
            z_top = max(z_top, float(zm.group(1))) if zm else z_top
    if not pts:
        return None
    lo, hi = np.min(pts, axis=0), np.max(pts, axis=0)
    return dict(center_mm=((lo + hi) / 2).round(3).tolist(), dims_mm=[*(hi - lo).round(3).tolist(), z_top])

def plan(objs):
    steps = []
    for name, o in objs.items():
        w, h = np.ptp(o["poly"], axis=0)
        grasp_w, yaw = (min(w, h), 0 if w <= h else 90)   # close jaws across the narrow side
        ok = grasp_w <= GRIPPER_MAX_MM
        steps.append(dict(name=name, center_mm=o["center"], footprint_mm=[float(w), float(h)],
                          height_mm=o["height"], grasp_width_mm=float(grasp_w), jaw_yaw_deg=yaw,
                          action="pick" if ok else "push_to_edge_then_pick"))
    # nearest-to-door first, so the arm never reaches over a part it has not removed
    return sorted(steps, key=lambda s: s["center_mm"][1] - DOOR_Y)

if __name__ == "__main__":
    objs, purge = parse(sys.argv[1])
    steps = plan(objs)
    out = dict(objects=steps, purge_line=purge, purge_tab=purge_tab(sys.argv[1]),
               purge_action="sweep with printed scraper tip (front edge, y≈1.5 mm)")
    json.dump(out, open("pick_plan.json", "w"), indent=2)
    for i, s in enumerate(steps, 1):
        print(f"{i}. {s['name']:<14} c=({s['center_mm'][0]:.0f},{s['center_mm'][1]:.0f}) "
              f"h={s['height_mm']:.0f} grasp={s['grasp_width_mm']:.0f}mm yaw={s['jaw_yaw_deg']} -> {s['action']}")
    print("purge segments:", len(purge))
