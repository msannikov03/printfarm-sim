"""Real 3D models as sim parts: a sliced plate (a BambuStudio .gcode.3mf, real/bambu.py) -> the robot's plate, planned
from the G-code (real/gcode_objects.py), and mesh bodies placed where the slicer put them (the 3MF's own meshes).

    parts = plate_from_slice("plate.gcode.3mf", printer=4, cache_dir)     # registers one SKU per part (farm.catalog)
    w = World()                                                            # ... so build the world after this
    placed = spawn_truth(w, parts)                                         # bodies at the true (mesh) poses

What the robot gets (parts.plate) is the G-code's view: per object the footprint rectangle (centre, yaw, long x short
side) and the height, from its extrusion moves. Its SKU (kind "box", those dims) also carries the top face of the
model (SKU.top_mm, from the mesh in the same 3MF): where the chamber camera should find the part's highest surface.
The body itself is the model: convex pieces of the mesh for contacts (convex_pieces: a grid split of the surface,
one hull per cell, or one convex hull for the whole part) and the true mesh for the cameras.

Frames. The G-code is in the printer's frame, the 3MF's build transforms in the slicer's plate frame; they differ by
the machine's extruder offset (a P1S: "; extruder_offset = 0x2", G-code y = plate y - 2 mm). The printed part lies
where the nozzle went, so the meshes are moved into the G-code frame before anything is compared or spawned.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from farm import catalog as C
from real.gcode_objects import extract, read_gcode_text
from plan.mesh_grasp import PLAN_CLEAR_MM, candidate_dirs, frame_prior, plan_tilts
from real.mesh_io import (PartGeometry, grasp_candidates, grasp_section, hull_points, load_3mf, simplify,
                          surface_samples, write_stl)

MAX_HULL_PTS = 48        # vertices per convex collision piece (MuJoCo collides with their hull)


@dataclass
class RealPart:
    name: str                # object name in the slicer
    sku: str                 # the registered SKU (farm.catalog.CATALOG)
    obj: dict                # the planner's plate object (from the G-code)
    geom: PartGeometry       # the model in its own frame, with its true pose on the bed
    parsed: dict             # the parser's object (real.gcode_objects)
    n_pieces: int = 1


@dataclass
class RealPlate:
    plate: dict
    parts: list
    plan: dict               # real.gcode_objects.extract output
    frame_offset_mm: tuple   # G-code frame - 3MF plate frame (minus the extruder offset)
    unmatched: list = field(default_factory=list)   # G-code objects without a mesh, or meshes without G-code


def gcode_frame_offset(text: str) -> np.ndarray:
    """G-code frame minus slicer plate frame (mm): minus the first extruder's offset in the config block."""
    m = re.search(r"^;\s*extruder_offset\s*=\s*([-\d.]+)x([-\d.]+)", text, re.M)
    return -np.array([float(m.group(1)), float(m.group(2))]) if m else np.zeros(2)


PIECE_MM = 25.0          # convex_pieces: about one cell per this much footprint length (at most 3 per side)


def convex_pieces(v: np.ndarray, f: np.ndarray, grid=(3, 3), sample_mm=1.5, overlap_mm=0.8, min_pts=12):
    """Simple convex decomposition: sample the surface every ~sample_mm, split the samples on a grid of cells in the
    part's footprint (at most grid[0] x grid[1], about one per PIECE_MM; cells overlap by overlap_mm), one convex hull
    per cell -> [vertex arrays (k, 3)]. grid=(1, 1) is the plain convex hull of the whole part."""
    p = surface_samples(v, f, sample_mm)
    ext = np.ptp(p[:, :2], axis=0)
    grid = tuple(int(np.clip(round(e / PIECE_MM), 1, g)) for e, g in zip(ext, grid))
    if grid == (1, 1):
        return [hull_points(p, MAX_HULL_PTS)]
    lo, hi = p[:, :2].min(0), p[:, :2].max(0)
    out = []
    xs = np.linspace(lo[0], hi[0], grid[0] + 1)
    ys = np.linspace(lo[1], hi[1], grid[1] + 1)
    for i in range(grid[0]):
        for j in range(grid[1]):
            m = ((p[:, 0] >= xs[i] - overlap_mm) & (p[:, 0] <= xs[i + 1] + overlap_mm) &
                 (p[:, 1] >= ys[j] - overlap_mm) & (p[:, 1] <= ys[j + 1] + overlap_mm))
            q = p[m]
            if len(q) < min_pts:
                continue
            try:
                ConvexHull(q)
            except QhullError:          # flat (one face of a thin sheet): no volume in this cell
                continue
            out.append(hull_points(q, MAX_HULL_PTS))
    return out or [hull_points(p, MAX_HULL_PTS)]


def match_mesh(o, meshes, tol_mm=10.0):
    """The mesh of a G-code object: same Bambu label id, else same name, else the nearest footprint centre within
    tol_mm (a plate sliced as placed carries no label ids or names). None if nothing matches."""
    from real.mesh_io import footprint
    for m in meshes:
        if o.get("label_id") is not None and m.meta.get("identify_id") == o["label_id"]:
            return m
    for m in meshes:
        if m.name == o["name"]:
            return m
    if not meshes:
        return None
    d = [float(np.hypot(*(footprint(m)[0] - o["center_mm"]))) for m in meshes]
    k = int(np.argmin(d))
    return meshes[k] if d[k] < tol_mm else None


def plate_from_slice(gcode_3mf, printer=4, cache_dir="runs/real_models/meshes", plate_id="RM", grid=(3, 3),
                     sku_prefix="rm") -> RealPlate:
    """Parse a sliced plate, match each G-code object to its mesh (by the Bambu label id), register one SKU per part
    and build the planner's plate (no purge tab: a Bambu printer purges into its chute)."""
    gcode_3mf = Path(gcode_3mf)
    cache = Path(cache_dir)
    plan = extract(gcode_3mf)
    off = gcode_frame_offset(read_gcode_text(gcode_3mf))
    meshes = load_3mf(gcode_3mf)
    for m in meshes:
        m.vertices[:, :2] += off                   # into the G-code (printer) frame
    parts, unmatched, objs = [], [], []
    for k, o in enumerate(plan["objects"]):
        mesh = match_mesh(o, meshes)
        if mesh is None:
            unmatched.append({"gcode_object": o["name"]})
            continue
        meshes = [m for m in meshes if m is not mesh]          # (by identity: Mesh holds arrays)
        if o["name"].startswith("id_"):            # a plate the slicer did not name (sliced as placed): the mesh's name
            o["name"] = mesh.name
        g = PartGeometry.from_mesh(mesh, yaw_hint_deg=o["yaw_deg"], bed_z=0.0)
        fx, fy = o["footprint_mm"]                 # the parser: long side along its yaw
        h = float(o["height_mm"] or g.dims_mm[2])
        sku = f"{sku_prefix}{k}"
        vis = write_stl(cache / f"{plate_id}_{sku}_vis.stl", *simplify(g.vertices, g.faces))   # (MuJoCo: <= 200k faces)
        pieces = convex_pieces(g.vertices, g.faces, grid)
        gs = grasp_section(g.vertices, g.faces, [0.6 * h, max(h - 4.0, 0.5 * h)])
        alts = grasp_candidates(g.vertices, g.faces, h, dirs=candidate_dirs(o["yaw_deg"]),
                                tilts=plan_tilts(max(fx, fy) > C.TOTE_PART_MAX_MM), stroke_mm=C.GRIPPER_MAX_MM,
                                clear_mm=PLAN_CLEAR_MM, prior=frame_prior(o["yaw_deg"]))
        bbox = max(fx, 1.0) * max(fy, 1.0) * max(h, 1.0) / 1000.0
        C.register_sku(C.SKU(sku, "box", (float(fx), float(fy), h), solid_frac=min(1.0, g.volume_cm3 / bbox),
                             vis_mesh=str(vis), col_meshes=tuple(tuple(map(tuple, pc.round(4))) for pc in pieces),
                             top_mm=tuple(round(t, 3) for t in g.top_mm),
                             grasp_mm=tuple(round(t, 3) for t in gs) if gs else (),
                             grasp_alts=tuple(tuple(round(t, 3) for t in a) for a in alts), pool=1))
        obj = {"name": f"{plate_id}_{sku}", "order_id": f"{plate_id}_{sku}", "sku": sku,
               "center_mm": [float(v) for v in o["center_mm"]], "yaw_deg": float(o["yaw_deg"]), "height_mm": h,
               "footprint_mm": [float(fx), float(fy)], "polygon_mm": o["polygon_mm"], "source_name": o["name"]}
        objs.append(obj)
        parts.append(RealPart(o["name"], sku, obj, g, o, len(pieces)))
    on_bed = lambda m: (0 <= m.vertices[:, 0].mean() <= C.BED_MM[0]) and (0 <= m.vertices[:, 1].mean() <= C.BED_MM[1])
    unmatched += [{"mesh": m.name} for m in meshes if on_bed(m)]     # (meshes off this bed are on other plates)
    plate ={"plate_id": plate_id, "printer": int(printer), "bed_mm": [int(C.BED_MM[0]), int(C.BED_MM[1])],
             "purge_tab": None, "objects": objs, "source": gcode_3mf.name}
    return RealPlate(plate, parts, plan, tuple(float(x) for x in off), unmatched)


def spawn_truth(w, rp: RealPlate, stuck=False, rng=None, pos_sigma_mm=0.0, yaw_sigma_deg=0.0) -> dict:
    """Spawn every part's body at its true pose (the mesh's footprint frame), optionally off it by a random shift
    (as scripts/eval_unload.py does: parts move a little while printing and cooling, so the slicer prior is not exact
    and the camera has something to correct). -> {plate object name: pool body}."""
    i = rp.plate["printer"]
    _, s = C.printer_frame(i)
    rng = rng or np.random.default_rng(0)
    placed = {}
    for p in rp.parts:
        body = w.free_part(p.sku)
        dx, dy = rng.normal(0, pos_sigma_mm, 2) if pos_sigma_mm else (0.0, 0.0)
        dyaw = rng.normal(0, yaw_sigma_deg) if yaw_sigma_deg else 0.0
        w.spawn(body, C.bed_mm_to_world(i, p.geom.center_mm[0] + dx, p.geom.center_mm[1] + dy),
                math.radians(p.geom.yaw_deg + dyaw) + (0 if s > 0 else math.pi), stuck=stuck)
        placed[p.obj["name"]] = body
    return placed


def parse_errors(p: RealPart) -> dict:
    """The G-code plan vs the true mesh footprint for one part: centre error, footprint size error (long, short side),
    yaw error (mod 180, or mod 90 for a near-square footprint) and height error."""
    c_err = float(np.hypot(*(np.array(p.parsed["center_mm"]) - p.geom.center_mm)))
    plong, pshort = sorted(p.parsed["footprint_mm"], reverse=True)
    tlong, tshort = sorted(p.geom.dims_mm[:2], reverse=True)
    period = 90.0 if tlong - tshort < 0.05 * tlong else 180.0
    dyaw = (p.parsed["yaw_deg"] - p.geom.yaw_deg + period / 2) % period - period / 2
    return {"center_err_mm": round(c_err, 3), "long_err_mm": round(plong - tlong, 3),
            "short_err_mm": round(pshort - tshort, 3), "yaw_err_deg": round(float(dyaw), 3),
            "height_err_mm": round(float((p.parsed["height_mm"] or 0.0) - p.geom.dims_mm[2]), 3),
            "true_footprint_mm": [round(tlong, 2), round(tshort, 2)], "true_height_mm": round(p.geom.dims_mm[2], 2)}


def without_top_model(sku: str):
    """Context helper for an ablation: the same SKU without its top-face model (the planner treats the part as a
    prism with a flat top over its whole footprint). Returns (swap_in, swap_back) callables."""
    orig = C.CATALOG[sku]
    return (lambda: C.CATALOG.__setitem__(sku, replace(orig, top_mm=()))), (lambda: C.CATALOG.__setitem__(sku, orig))


def pack_models(paths, out_3mf, spacing_mm=C.PART_SPACING_MM, y0_mm=30.0):
    """pack_meshes for model files (STL / 3MF). Returns (path, n_packed, n_left_out)."""
    from real.mesh_io import load_mesh
    meshes = [m for p in paths for m in load_mesh(p)]
    path, packed, left = pack_meshes(meshes, out_3mf, spacing_mm, y0_mm)
    return path, len(packed), len(left)


def pack_meshes(meshes, out_3mf, spacing_mm=C.PART_SPACING_MM, y0_mm=30.0):
    """Robot-friendly plate: every model's footprint rectangle packed on the P1S bed with the cell's jaw clearance
    (jobs/plates.py MaxRects, bottom-left, yaw 0 / 90, spacing_mm between parts, the front y0_mm kept free: the
    printer's own exclusion zone at the front-left corner) -> a 3MF with those build transforms, to slice as placed
    (real.bambu.slice_models(..., arrange=False)). Largest first; what does not fit is left for the next plate.
    Returns (path, packed meshes, meshes left out)."""
    from jobs.plates import BED_MARGIN_MM, MaxRects
    from real.mesh_io import footprint, rot_z, write_3mf
    s = spacing_mm
    mr = MaxRects(BED_MARGIN_MM, y0_mm, C.BED_MM[0] - BED_MARGIN_MM + s, C.BED_MM[1] - BED_MARGIN_MM + s)
    out, T, left = [], [], []
    for m in sorted(meshes, key=lambda m: -np.prod(footprint(m)[1])):
        c, (L, S), yaw, _ = footprint(m)
        best = None
        for turn, (w, h) in ((0.0, (L, S)), (90.0, (S, L))):
            pos = mr.find(w + s, h + s)
            if pos and (best is None or (pos[1] + h, pos[0]) < (best[1] + best[3], best[0])):
                best = (*pos, w, h, turn)
        if best is None:
            left.append(m)
            continue
        x, y, w, h, turn = best
        mr.place(x, y, w + s, h + s)
        R = np.eye(4)
        R[:3, :3] = rot_z(turn - yaw)
        A = np.eye(4)
        A[:3, 3] = [-c[0], -c[1], -m.vertices[:, 2].min()]
        B = np.eye(4)
        B[:3, 3] = [x + w / 2, y + h / 2, 0.0]
        T.append(B @ R @ A)
        out.append(m)
    names = {}
    for m in out:                                   # unique object names (the parser keys on them)
        k = names.get(m.name, 0)
        names[m.name] = k + 1
        if k:
            m.name = f"{m.name}_{k + 1}"
    if out:
        write_3mf(out_3mf, out, T)
    return Path(out_3mf), out, left
