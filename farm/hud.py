"""Video overlay: farm clock, current robot action, printer states, caddy bins, shipping counters."""
import cv2
import numpy as np

from farm import catalog as C

STATE_COL = {"printing": (43, 143, 214), "cooling": (242, 193, 78), "waiting": (224, 62, 90),
             "unloading": (242, 140, 40), "idle": (190, 193, 198)}


def _txt(img, s, xy, scale=0.55, col=(255, 255, 255), th=1):
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, col, th, cv2.LINE_AA)


def make_overlay(farm):
    def overlay(img, w):
        img = np.ascontiguousarray(img)
        h, wd, _ = img.shape
        bar = img[:64].astype(np.float32)
        img[:64] = (bar * 0.25 + np.array([20, 22, 26]) * 0.75).astype(np.uint8)
        farm_t = farm.t + (w.d.time - farm.t_phys)
        hh, rem = divmod(int(farm_t), 3600)
        _txt(img, f"farm time {hh:02d}:{rem // 60:02d}", (16, 26), 0.7, (242, 140, 40), 2)
        _txt(img, farm.action, (16, 52), 0.6)
        st = farm.stats
        picks = sum(v for k, v in st.items() if k.startswith("pick:"))
        ok = st.get("pick:placed", 0)
        _txt(img, f"orders in {st.get('orders_in', 0)}  shipped {st.get('orders_shipped', 0)}   "
                  f"picks {ok}/{picks}   interventions {st.get('interventions', 0)}", (wd - 620, 26), 0.6)
        _txt(img, f"packages shipped {st.get('packages_shipped', 0)}   simulation (MuJoCo) - "
                  f"{'sealer + lanes kinematic' if C.TWO_ROWS else 'packaging line kinematic'}", (wd - 620, 52), 0.45,
             (170, 175, 182))
        # printers: one line of 10 per row (row A, row B in the two-row cell)
        per_row = C.N_PER_ROW
        n_rows = (len(farm.printers) + per_row - 1) // per_row
        row_h = 48 if n_rows > 1 else 54
        x0 = 66 if n_rows > 1 else 16
        y_top = h - 8 - n_rows * row_h
        cv2.rectangle(img, (8, y_top - 22), (x0 + per_row * 92, h - 8), (20, 22, 26), -1)
        for p in farm.printers:
            r, c = divmod(p.i, per_row)
            x, y0 = x0 + c * 92, y_top + r * row_h + 4
            cv2.rectangle(img, (x, y0), (x + 84, y0 + 20), STATE_COL[p.state], -1)
            _txt(img, f"P{p.i + 1} {p.state[:5]}", (x + 4, y0 + 15), 0.45, (20, 20, 20), 1)
            _txt(img, f"{farm.bed_temp(p):.0f}C  {len(p.bodies)} pcs", (x + 2, y0 + 37), 0.42)
        if n_rows > 1:
            for r in range(n_rows):
                _txt(img, f"row {chr(65 + r)}", (14, y_top + r * row_h + 19), 0.45, (170, 175, 182))
        _txt(img, "printers", (x0, y_top - 6), 0.45, (170, 175, 182))
        # caddy: package totes, large-part tray, reject / waste
        al = farm.alloc
        rows = [(t, al.tote_pkg.get(t)) for t in ("o0", "o1", "o2", "o3", "o4", "o5")] + \
               [(sl, al.part_pkg[al.large_part[sl]] if sl in al.large_part else None) for sl in al.large] + \
               [("reject", None), ("waste", None)]
        cx, cy = wd - 260, 90
        cv2.rectangle(img, (cx - 10, cy - 22), (wd - 10, cy + len(rows) * 24 + 4), (20, 22, 26), -1)
        _txt(img, "caddy (package : parts in / needed)", (cx, cy - 6), 0.45, (170, 175, 182))
        for k, (slot, pid) in enumerate(rows):
            n = len(al.contents.get(slot, []))
            if pid and slot.startswith("o"):
                label = f"{slot:6s} {pid:11s} {n}/{len(al.packages[pid].small)}"
            elif pid:
                label = f"{slot:6s} {pid:11s} {al.large_part[slot].split('_')[1] if n else 'reserved'}"
            else:
                label = f"{slot:6s} {n if slot in ('reject', 'waste') else ''}"
            _txt(img, label, (cx, cy + 20 + k * 24), 0.5, (255, 255, 255) if pid else (150, 150, 150))
        return img
    return overlay
