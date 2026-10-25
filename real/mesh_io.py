"""Triangle meshes of real printed parts: STL / 3MF loading, footprints, top faces and collision hulls (numpy + scipy).

    load_mesh("part.stl")                 -> Mesh (one per file; mm)
    load_3mf("plate.gcode.3mf")           -> [Mesh] in plate coordinates (build transforms applied), with the
                                             BambuStudio object name and G-code label id where the file has them
    footprint(mesh)                       -> centre, (long, short), yaw of the long side, height: the same
                                             min-area rectangle real.gcode_objects fits to a part's extrusions
    PartGeometry.from_mesh(mesh, ...)     -> the mesh in the part's own frame + its top face (for the sim)

3MF: the core spec (vertices / triangles, components, build items with 4x3 transforms, model units) plus the
production extension's external object files (p:path, as BambuStudio writes them). Units are converted to mm.
"""
from __future__ import annotations

import html
import math
import re
import struct
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from real.gcode_objects import convex_hull, min_area_rect

UNIT_MM = {"micron": 0.001, "millimeter": 1.0, "centimeter": 10.0, "inch": 25.4, "foot": 304.8, "meter": 1000.0}


@dataclass
class Mesh:
    name: str
    vertices: np.ndarray                  # (n, 3) float, mm
    faces: np.ndarray                     # (m, 3) int
    meta: dict = field(default_factory=dict)   # 3MF: object_id, identify_id (Bambu G-code label id), source

    def transformed(self, T: np.ndarray) -> "Mesh":
        """Apply a 4x4 transform (column-vector convention)."""
        v = self.vertices @ T[:3, :3].T + T[:3, 3]
        return Mesh(self.name, v, self.faces.copy(), dict(self.meta))

    @property
    def bounds(self):
        return self.vertices.min(0), self.vertices.max(0)


def _dedupe(tri: np.ndarray, name: str) -> Mesh:
    """(m, 3, 3) triangle soup -> indexed mesh (vertices merged at 1e-5 mm)."""
    v = tri.reshape(-1, 3)
    uniq, inv = np.unique(np.round(v, 5), axis=0, return_inverse=True)
    return Mesh(name, uniq.astype(float), inv.reshape(-1, 3).astype(np.int64))


def load_stl(path) -> Mesh:
    path = Path(path)
    data = path.read_bytes()
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * n == len(data):                                   # binary
            rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
                                count=n, offset=84)
            return _dedupe(rec["v"].astype(float), path.stem)
    nums = re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", data)       # ASCII
    if not nums or len(nums) % 3:
        raise ValueError(f"{path}: not a readable STL")
    return _dedupe(np.array(nums, dtype=float).reshape(-1, 3, 3), path.stem)


def _ns(tag):
    return tag.split("}", 1)[1] if "}" in tag else tag


def _transform(s: str | None) -> np.ndarray:
    """3MF transform "m00 m01 m02 m10 m11 m12 m20 m21 m22 m30 m31 m32" (row vectors) -> 4x4 column convention."""
    T = np.eye(4)
    if s:
        m = np.array([float(x) for x in s.split()], float).reshape(4, 3)
        T[:3, :3], T[:3, 3] = m[:3].T, m[3]
    return T


def load_3mf(path, build_only=True) -> list[Mesh]:
    """Every build item of a 3MF as one mesh in plate coordinates (mm). Object names come from the object's
    name attribute or BambuStudio's Metadata/model_settings.config; meta["identify_id"] is the label id BambuStudio
    writes into its G-code ("; start printing object, unique label id: N") when the file has a sliced plate. Parts
    that model_settings.config marks as not printed (modifier / negative volumes, support blockers) are left out."""
    path = Path(path)
    with zipfile.ZipFile(path) as z:
        files = {n: z.read(n) for n in z.namelist() if n.lower().endswith(".model")}
        settings = z.read("Metadata/model_settings.config").decode("utf-8", "replace") \
            if "Metadata/model_settings.config" in z.namelist() else None
    root_name = next((n for n in files if n.lower() == "3d/3dmodel.model"), None) or next(iter(files), None)
    if root_name is None:
        raise ValueError(f"{path}: no 3D model in the archive")
    docs = {}

    def doc(name):
        name = name.lstrip("/")
        if name not in docs:
            r = ET.fromstring(files[name])
            scale = UNIT_MM.get(r.get("unit", "millimeter"), 1.0)
            objs = {}
            for res in r:
                if _ns(res.tag) != "resources":
                    continue
                for o in res:
                    if _ns(o.tag) == "object":
                        objs[o.get("id")] = o
            docs[name] = (r, objs, scale)
        return docs[name]

    def collect(name, oid, T, out, skip=()):
        r, objs, _ = doc(name)
        o = objs[oid]
        for child in o:
            t = _ns(child.tag)
            if t == "mesh":
                vs = [c for c in child if _ns(c.tag) == "vertices"][0]
                ts = [c for c in child if _ns(c.tag) == "triangles"][0]
                v = np.array([[float(e.get("x")), float(e.get("y")), float(e.get("z"))] for e in vs], float)
                f = np.array([[int(e.get("v1")), int(e.get("v2")), int(e.get("v3"))] for e in ts], np.int64)
                out.append((v @ T[:3, :3].T + T[:3, 3], f))
            elif t == "components":
                for c in child:
                    if c.get("objectid") in skip:          # a modifier / negative volume: not printed
                        continue
                    p = next((v for k, v in c.attrib.items() if _ns(k) == "path"), None)
                    collect(p or name, c.get("objectid"), T @ _transform(c.get("transform")), out)

    names, ident, not_printed = {}, {}, {}
    if settings:                  # BambuStudio: object names and plate label ids. Read with regular expressions:
        # BambuStudio does not escape quotes inside attribute values there (per-object settings), so it is not XML
        for oid, body in re.findall(r'<object id="(\d+)">(.*?)</object>', settings, re.S):
            m = re.search(r'<metadata key="name" value="([^"]*)"', body)
            if m:
                names[oid] = html.unescape(m.group(1))
            not_printed[oid] = {pid for pid, sub in re.findall(r'<part id="(\d+)" subtype="(\w+)"', body)
                                if sub != "normal_part"}     # modifier_part, negative_part, support_blocker, ...
        for body in re.findall(r"<model_instance>(.*?)</model_instance>", settings, re.S):
            md = dict(re.findall(r'<metadata key="(\w+)" value="([^"]*)"', body))
            if md.get("object_id") and md.get("identify_id"):
                ident.setdefault(md["object_id"], []).append(int(md["identify_id"]))
    r, objs, scale = doc(root_name)          # (one unit for the whole package: the root model's)
    meshes = []
    items = [it for b in r if _ns(b.tag) == "build" for it in b if _ns(it.tag) == "item"]
    if not build_only and not items:
        items = [ET.Element("item", objectid=k) for k in objs]
    seen = {}
    for it in items:
        oid = it.get("objectid")
        parts = []
        collect(root_name, oid, _transform(it.get("transform")), parts, not_printed.get(oid, ()))
        if not parts:
            continue
        off = np.cumsum([0] + [len(v) for v, _ in parts[:-1]])
        v = np.vstack([p[0] for p in parts]) * scale
        f = np.vstack([p[1] + o for p, o in zip(parts, off)])
        k = seen.get(oid, 0)
        seen[oid] = k + 1
        nm = objs[oid].get("name") or names.get(oid) or f"object_{oid}"
        meta = {"object_id": oid, "source": path.name}
        if oid in ident and k < len(ident[oid]):
            meta["identify_id"] = ident[oid][k]
        meshes.append(Mesh(nm, v, f, meta))
    return meshes


def load_mesh(path) -> list[Mesh]:
    """STL -> [one mesh]; 3MF -> one mesh per build item."""
    path = Path(path)
    if zipfile.is_zipfile(path):
        return load_3mf(path)
    return [load_stl(path)]


def footprint(mesh_or_vertices):
    """(centre (2,), (long, short), yaw_deg of the long side in (-90, 90], height): min-area rectangle of the
    XY convex hull, height above the lowest vertex. The same rectangle real.gcode_objects fits to extrusions."""
    v = mesh_or_vertices.vertices if isinstance(mesh_or_vertices, Mesh) else np.asarray(mesh_or_vertices)
    c, (w, h), yaw = min_area_rect(convex_hull(v[:, :2]))
    return np.asarray(c, float), (float(w), float(h)), float(yaw), float(np.ptp(v[:, 2]))


def volume_mm3(mesh: Mesh) -> float:
    a, b, c = (mesh.vertices[mesh.faces[:, k]] for k in range(3))
    return float(abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)


def hull_points(v: np.ndarray, max_points: int | None = None) -> np.ndarray:
    """Vertices of the convex hull (optionally thinned to at most max_points, farthest-point order)."""
    try:
        h = v[ConvexHull(v).vertices]
    except QhullError:
        h = v
    if max_points and len(h) > max_points:
        keep = [int(np.argmax(np.linalg.norm(h - h.mean(0), axis=1)))]
        d = np.linalg.norm(h - h[keep[0]], axis=1)
        for _ in range(max_points - 1):
            keep.append(int(np.argmax(d)))
            d = np.minimum(d, np.linalg.norm(h - h[keep[-1]], axis=1))
        h = h[keep]
    return h


def rot_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass
class PartGeometry:
    """A real part in its own frame: origin at the footprint rectangle's centre on the bed, x along `yaw_deg`.
    top_mm: (tx, ty, ox, oy, area) = extent, centre offset and area (mm^2, 1 mm grid) of the top band (the surface
    within top_tol_mm of the highest point), in the same frame: where a depth camera finds the part's highest surface
    (a rim, an upright, a boss)."""
    name: str
    vertices: np.ndarray        # (n, 3) mm, local frame
    faces: np.ndarray
    center_mm: tuple            # footprint centre on the plate (truth)
    yaw_deg: float              # local x axis on the plate (truth)
    dims_mm: tuple              # (along x, along y, height) of the footprint rectangle
    top_mm: tuple
    volume_cm3: float

    @classmethod
    def from_mesh(cls, mesh: Mesh, yaw_hint_deg: float | None = None, top_tol_mm: float = 1.5,
                  bed_z: float | None = None):
        """The mesh's true footprint frame. With yaw_hint_deg (e.g. the slicer's yaw for this part) the frame is
        the footprint-rectangle orientation closest to it (mod 90 deg), so its axes match the plan's. With bed_z (a
        mesh in plate coordinates) a model sunk into the bed is cut there, as the slicer cuts it (approximately: the
        vertices below the bed are pressed onto it)."""
        if bed_z is not None and mesh.vertices[:, 2].min() < bed_z - 0.01:
            mesh = Mesh(mesh.name, mesh.vertices.copy(), mesh.faces, dict(mesh.meta, sunk_mm=float(
                bed_z - mesh.vertices[:, 2].min())))
            mesh.vertices[:, 2] = np.maximum(mesh.vertices[:, 2], bed_z)
        c, (L, S), yaw, h = footprint(mesh)
        dims = (L, S)
        if yaw_hint_deg is not None:
            k = round(((yaw_hint_deg - yaw + 45.0) // 90.0))
            yaw += 90.0 * k
            if k % 2:
                dims = (S, L)
        v = (mesh.vertices - [c[0], c[1], mesh.vertices[:, 2].min()]) @ rot_z(yaw)       # rows: R^T p
        zt = v[:, 2].max()
        # what a depth camera above the part sees in its top band: every surface point within top_tol_mm of the
        # highest point (flat tops, the upper edges of walls and ribs, slopes), on a 1 mm grid
        sp = surface_samples(v, mesh.faces, 0.5)
        pts = sp[sp[:, 2] > zt - top_tol_mm, :2]
        cells = np.unique(np.floor(pts).astype(np.int64), axis=0)
        lo, hi = pts.min(0), pts.max(0)
        top = (float(hi[0] - lo[0]), float(hi[1] - lo[1]), float((hi[0] + lo[0]) / 2), float((hi[1] + lo[1]) / 2),
               float(len(cells)))
        return cls(mesh.name, v, mesh.faces.copy(), (float(c[0]), float(c[1])), float(yaw),
                   (float(dims[0]), float(dims[1]), float(h)), top, volume_mm3(mesh) / 1000.0)


def surface_samples(v: np.ndarray, f: np.ndarray, step_mm: float = 1.5) -> np.ndarray:
    """Points on the mesh surface about every step_mm (the vertices plus a barycentric grid on larger triangles)."""
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    edge = np.maximum(np.linalg.norm(b - a, axis=1), np.linalg.norm(c - a, axis=1))
    n = np.clip(np.ceil(edge / step_mm).astype(int), 1, 60)
    pts = [v]
    for k in np.unique(n):
        if k == 1:
            continue
        sel = n == k
        u, w = np.meshgrid(np.arange(k + 1), np.arange(k + 1))
        keep = (u + w) <= k
        uu, ww = u[keep] / k, w[keep] / k
        pts.append((a[sel, None] + uu[None, :, None] * (b - a)[sel, None] + ww[None, :, None] * (c - a)[sel, None])
                   .reshape(-1, 3))
    return np.vstack(pts)


def write_stl(path, vertices: np.ndarray, faces: np.ndarray, scale: float = 1.0) -> Path:
    """Binary STL (MuJoCo reads it natively)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tri = np.asarray(vertices, float)[np.asarray(faces)] * scale
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(tri), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"], rec["v"] = n, tri
    with open(path, "wb") as f:
        f.write(b"printfarm-sim mesh".ljust(80, b" "))
        f.write(struct.pack("<I", len(tri)))
        f.write(rec.tobytes())
    return path


def hull_faces(points: np.ndarray):
    """(vertices, faces) of the convex hull of `points` (consistently wound, outward normals)."""
    h = ConvexHull(points)
    v = points[h.vertices]
    remap = {int(k): i for i, k in enumerate(h.vertices)}
    f = np.array([[remap[int(k)] for k in s] for s in h.simplices], np.int64)
    c = v.mean(0)
    a, b, cc = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    flip = np.einsum("ij,ij->i", np.cross(b - a, cc - a), a - c) < 0
    f[flip] = f[flip][:, ::-1]
    return v, f


# ----------------------------------------------------------------------------- generated test geometry (our own)

def _quad(f, a, b, c, d):
    f += [(a, b, c), (a, c, d)]


def cup_mesh(x=60.0, y=40.0, h=25.0, wall=2.0, floor=2.0, name="enclosure") -> Mesh:
    """Open-top box (an enclosure shell): watertight, outward normals, origin at the bottom face's centre."""
    ox, oy, ix, iy = x / 2, y / 2, x / 2 - wall, y / 2 - wall
    rect = lambda hx, hy, z: [(-hx, -hy, z), (hx, -hy, z), (hx, hy, z), (-hx, hy, z)]
    v = np.array(rect(ox, oy, 0) + rect(ox, oy, h) + rect(ix, iy, h) + rect(ix, iy, floor), float)
    f = []
    _quad(f, 0, 3, 2, 1)                                       # bottom (down)
    for k in range(4):
        a, b = k, (k + 1) % 4
        _quad(f, a, b, 4 + b, 4 + a)                           # outer walls
        _quad(f, 4 + a, 4 + b, 8 + b, 8 + a)                   # rim
        _quad(f, 8 + a, 8 + b, 12 + b, 12 + a)                 # inner walls (facing in)
    _quad(f, 12, 13, 14, 15)                                   # inner floor (up)
    return Mesh(name, v, np.array(f, np.int64))


def l_bracket_mesh(length=50.0, width=20.0, base_t=4.0, upright_h=30.0, upright_t=4.0, name="l_bracket") -> Mesh:
    """L bracket (base plate + upright at the -x end) extruded along y: watertight, origin at the base's centre."""
    x0, x1 = -length / 2, length / 2
    prof = [(x0, 0), (x1, 0), (x1, base_t), (x0 + upright_t, base_t), (x0 + upright_t, upright_h), (x0, upright_h)]
    n = len(prof)
    v = np.array([(px, -width / 2, pz) for px, pz in prof] + [(px, width / 2, pz) for px, pz in prof], float)
    f = []
    for k in range(n):
        a, b = k, (k + 1) % n
        _quad(f, a, n + a, n + b, b)                           # sides of the extrusion
    caps = [(0, 1, 2), (0, 2, 3), (0, 3, 5), (3, 4, 5)]        # the L profile, triangulated
    f += [c for c in caps] + [(n + c, n + b, n + a) for a, b, c in caps]
    m = Mesh(name, v, np.array(f, np.int64))
    a, b, c = (m.vertices[m.faces[:, k]] for k in range(3))
    if np.einsum("ij,ij->i", a, np.cross(b, c)).sum() < 0:     # outward normals
        m.faces = m.faces[:, ::-1].copy()
    return m


def write_3mf(path, meshes, transforms=None, extra=None) -> Path:
    """Minimal core-spec 3MF: one object per mesh (named), one build item each (4x4 transform, default identity).
    extra: {archive name: text} added as is (e.g. a G-code for tests)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    res, build = [], []
    for k, m in enumerate(meshes, 1):
        vs = "".join(f'<vertex x="{x:.9g}" y="{y:.9g}" z="{z:.9g}"/>' for x, y, z in m.vertices)
        ts = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in m.faces)
        res.append(f'<object id="{k}" name="{html.escape(m.name)}" type="model"><mesh><vertices>{vs}</vertices>'
                   f'<triangles>{ts}</triangles></mesh></object>')
        T = np.eye(4) if transforms is None else np.asarray(transforms[k - 1], float)
        t = " ".join(f"{x:.9g}" for x in [*T[:3, :3].T.ravel(), *T[:3, 3]])
        build.append(f'<item objectid="{k}" transform="{t}"/>')
    model = ('<?xml version="1.0" encoding="UTF-8"?><model unit="millimeter" xml:lang="en-US" '
             'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"><resources>' + "".join(res)
             + "</resources><build>" + "".join(build) + "</build></model>")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.'
                   'openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/'
                   'vnd.openxmlformats-package.relationships+xml"/><Default Extension="model" ContentType="application/'
                   'vnd.ms-package.3dmanufacturing-3dmodel+xml"/></Types>')
        z.writestr("_rels/.rels", '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.'
                   'openxmlformats.org/package/2006/relationships"><Relationship Target="/3D/3dmodel.model" Id="rel0" '
                   'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
        z.writestr("3D/3dmodel.model", model)
        for n, text in (extra or {}).items():
            z.writestr(n, text)
    return path


def section_segments(v: np.ndarray, f: np.ndarray, z: float) -> np.ndarray:
    """Segments (k, 2, 2) where the plane at height z cuts the mesh's triangles: the part's outline at that layer."""
    t = v[f]                                             # (m, 3, 3)
    t = t[(t[:, :, 2].min(1) < z) & (t[:, :, 2].max(1) > z)]
    pts = np.full((len(t), 3, 2), np.nan)
    for k, (a, b) in enumerate(((0, 1), (1, 2), (2, 0))):
        za, zb = t[:, a, 2], t[:, b, 2]
        e = (za - z) * (zb - z) < 0
        s = (z - za[e]) / (zb[e] - za[e])
        pts[e, k] = t[e, a, :2] + s[:, None] * (t[e, b, :2] - t[e, a, :2])
    ok = ~np.isnan(pts[:, :, 0])
    keep = ok.sum(1) == 2
    return np.stack([pts[keep][ok[keep]][0::2], pts[keep][ok[keep]][1::2]], axis=1) if keep.any() else np.zeros((0, 2, 2))


def _clip_extent(seg, axis, centre, band):
    """Extent along `axis` of the segment parts within `band` of `centre` across it: (lo, hi) or None."""
    o = 1 - axis
    p, q = seg[:, 0], seg[:, 1]
    lo_b, hi_b = centre[o] - band, centre[o] + band
    d = q[:, o] - p[:, o]
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = np.where(np.abs(d) > 1e-12, (lo_b - p[:, o]) / d, -np.inf)
        t1 = np.where(np.abs(d) > 1e-12, (hi_b - p[:, o]) / d, np.inf)
    ta, tb = np.clip(np.minimum(t0, t1), 0, 1), np.clip(np.maximum(t0, t1), 0, 1)
    inside = np.where(np.abs(d) > 1e-12, ta < tb, (p[:, o] >= lo_b) & (p[:, o] <= hi_b))
    if not inside.any():
        return None
    a = p[inside, axis] + ta[inside] * (q[inside, axis] - p[inside, axis])
    b = p[inside, axis] + tb[inside] * (q[inside, axis] - p[inside, axis])
    return float(min(a.min(), b.min())), float(max(a.max(), b.max()))


def grasp_section(v, f, heights, centre=(0.0, 0.0), band_mm=6.0):
    """Where the jaws close on a part, from its cross-sections at the grasp heights (the slicer's layers there).
    Starting from `centre` (the footprint centre): the outline within band_mm across a line along x gives the extent
    the jaws must span along x, and its midpoint; around that midpoint the same along y; then x again around the new
    centre. An enclosure gives its outer walls, an L bracket its upright. -> (cx, cy, wx, wy) mm (grasp centre, opening
    needed along x and along y), or None if the planes miss the part there."""
    seg = np.concatenate([section_segments(v, f, z) for z in heights])
    if not len(seg):
        return None
    c = np.asarray(centre, float)
    for first in (0, 1):                          # x first, else (the line along x misses the part) y first
        second = 1 - first
        e1 = _clip_extent(seg, first, c, band_mm)
        if e1 is None:
            continue
        c1 = c.copy()
        c1[first] = (e1[0] + e1[1]) / 2
        e2 = _clip_extent(seg, second, c1, band_mm)
        if e2 is None:
            continue
        c1[second] = (e2[0] + e2[1]) / 2
        e1 = _clip_extent(seg, first, c1, band_mm) or e1
        c1[first] = (e1[0] + e1[1]) / 2
        ex, ey = (e1, e2) if first == 0 else (e2, e1)
        return float(c1[0]), float(c1[1]), ex[1] - ex[0], ey[1] - ey[0]
    return None


# ---- grasp search over the part's own layers (the fallback of plan.unload's fixed-height grasp, plan/mesh_grasp.py)
GRASP_STEP_MM = 0.6      # a section every 3 slicer layers (0.2 mm)
ROW_MM = 1.0             # section outlines are binned in rows this wide across the jaw axis
# a finger of the parallel gripper, in the plane normal to the jaw axis (YAM, measured in the planning model): it ends
# FINGER_END_MM past the pinch point (the pad centre), is 2 x FINGER_HALF_MM wide there (pad and capsules) and is
# checked FINGER_LEN_MM back from its end (higher up, the wrist and the palm are left to the planner's collision check)
FINGER_END_MM = 10.0
FINGER_HALF_MM = 7.0
FINGER_LEN_MM = 60.0
PAD_LEN_MM = 40.0        # ... and grips with its first PAD_LEN_MM (the pad)


def mesh_centroid(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Volume centroid of a closed mesh (mm); the vertex mean for a mesh without volume."""
    a, b, c = (v[f[:, k]] for k in range(3))
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    if abs(vol.sum()) < 1e-9:
        return v.mean(0)
    return (vol[:, None] * (a + b + c) / 4.0).sum(0) / vol.sum()


def _sections(v, f, zs):
    """section_segments at every height in zs (the triangles are gathered once)."""
    t = v[f]
    zlo, zhi = t[:, :, 2].min(1), t[:, :, 2].max(1)
    out = []
    for z in zs:
        s = (zlo < z) & (zhi > z)
        out.append(section_segments(t[s].reshape(-1, 3), np.arange(3 * int(s.sum())).reshape(-1, 3), z)
                   if s.any() else np.zeros((0, 2, 2)))
    return out


def _row_extents(seg, w0, n_rows, row=ROW_MM):
    """Extent along u (column 0) of a section outline (segments in (u, w) coordinates) in rows of width `row` across
    it, starting at w0: (lo, hi) per row, +inf / -inf where a row holds no outline."""
    lo, hi = np.full(n_rows, np.inf), np.full(n_rows, -np.inf)
    if not len(seg):
        return lo, hi
    p, q = seg[:, 0], seg[:, 1]
    wa, wb = np.minimum(p[:, 1], q[:, 1]), np.maximum(p[:, 1], q[:, 1])
    r0 = np.clip(np.floor((wa - w0) / row).astype(int), 0, n_rows - 1)
    r1 = np.clip(np.floor((wb - w0) / row).astype(int), 0, n_rows - 1)
    cnt = r1 - r0 + 1
    k = np.repeat(np.arange(len(seg)), cnt)
    rows = r0[k] + np.arange(int(cnt.sum())) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    P, Q = p[k], q[k]
    d = Q[:, 1] - P[:, 1]
    flat = np.abs(d) < 1e-12
    dd = np.where(flat, 1.0, d)
    b0 = w0 + rows * row
    t0, t1 = (b0 - P[:, 1]) / dd, (b0 + row - P[:, 1]) / dd
    ta = np.where(flat, 0.0, np.clip(np.minimum(t0, t1), 0, 1))
    tb = np.where(flat, 1.0, np.clip(np.maximum(t0, t1), 0, 1))
    ua, ub = P[:, 0] + ta * (Q[:, 0] - P[:, 0]), P[:, 0] + tb * (Q[:, 0] - P[:, 0])
    np.minimum.at(lo, rows, np.minimum(ua, ub))
    np.maximum.at(hi, rows, np.maximum(ua, ub))
    return lo, hi


def _window(x, a, b, fill, op):
    """out[j] = op(x[j + a .. j + b]) (rows outside x count as `fill`); a <= b, integers."""
    n = len(x)
    pad = np.concatenate([np.full(max(-a, 0), fill), x, np.full(max(b, 0), fill)])
    start = a + max(-a, 0)
    view = np.lib.stride_tricks.sliding_window_view(pad, b - a + 1)
    return op(view[start:start + n], axis=1)


def _strip_extents(lo, hi, dz, sa, ca, b, end, length=FINGER_LEN_MM):
    """One layer of the finger column: the extent along the jaw axis of the material (row extents lo / hi of the
    layer) inside the finger's strip, for every finger position across the jaw axis (the rows). The strip: within b
    of the finger axis, from `end` past the finger's end to `length` back from it, in the plane normal to the jaw
    axis; the finger tilted by asin(sa) from vertical and leaning toward +w; dz: the layer height above the finger's
    end. -> (lo, hi) per finger end row (+inf / -inf: no material), or None (the strip misses the layer)."""
    A, B = ((-b - dz * sa) / ca, (b - dz * sa) / ca) if ca > 1e-9 else (-np.inf, np.inf)    # across the finger axis
    if sa > 1e-9:                                                                            # along it
        A, B = max(A, (dz * ca - length) / sa), min(B, (dz * ca + end) / sa)
    elif not -end <= dz <= length:
        return None
    if A > B:
        return None
    ra, rb = int(math.floor(A / ROW_MM)), int(math.ceil(B / ROW_MM))
    return _window(lo, ra, rb, np.inf, np.min), _window(hi, ra, rb, -np.inf, np.max)


def grasp_candidates(v, f, height=None, dirs=None, tilts=(50, 65, 80), stroke_mm=70.0, open_mm=74.0, tip_mm=1.8,
                     min_cover_mm=5.0, cover_frac=0.6, touch_mm=0.5, clear_mm=5.5, step_mm=GRASP_STEP_MM,
                     min_touch_mm=2.0, n_keep=40, prior=None):
    """Grasp search over a part's own layers: where a parallel-jaw gripper can close on it, from its sections every
    step_mm (the slicer's layers there). v, f: the mesh in the part's own frame (bed at z = 0, mm).

    Searched: the jaw axis (dirs: [(yaw_deg, lean)] in the part frame, default every 15 deg, both leans; lean = +1 / -1:
    the approach leans toward +/- the horizontal normal of the jaw axis, i.e. into the printer), the approach tilt
    from vertical about the jaw axis (tilts, deg), where the finger lies across the jaw axis (every ROW_MM) and the
    height of its lowest point (tip_mm up, every step_mm). The finger is the slanted strip of the FINGER_* model in
    the plane normal to the jaw axis. For each:
    - the column: every section within that strip grown by clear_mm (the planner's clearance to the part, plus its
      pose uncertainty); the open jaws must clear its extent along the jaw axis, which is the grasp width (at most
      stroke_mm): the jaws open past it and close on it;
    - the cover: the height over which the pads themselves (the strip, not grown, its first PAD_LEN_MM) reach that
      extent on both sides (sections within touch_mm of it), at least min_touch_mm (not an edge or a lip only);
      the lowest finger point must be min(min_cover_mm,
      cover_frac x the top of what the pads reach) below that top (plan.unload's MIN_COVER rule, with the column's
      own top);
    - scored by the cover (up to 8 mm), the stroke margin (open_mm minus the width, up to 16 mm) and the distance of
      the contact (where the pads touch, on average) from the mesh centroid: the part hangs from the jaws.
    The picks are spread over frames (jaw axis, lean, tilt) and over what they grip (width, contact): the planner tries
    other features and other arm poses early. prior(yaw_deg, lean, tilt_deg) -> weight orders the frames by how well
    the arm reaches them (default: all alike); the scores returned are the mesh's own.
    -> [(cx, cy, yaw_deg, lean, tilt_deg, pz, width, z_tip, cover, score)] in that order, at most n_keep: (cx, cy, pz)
    the pinch point (pad centre) and z_tip the lowest finger point, mm."""
    v = np.asarray(v, float)
    h = float(height if height is not None else v[:, 2].max())
    zs = np.arange(step_mm / 2, h, step_mm)
    if not len(zs):
        return []
    secs = _sections(v, f, zs)
    cen = mesh_centroid(v, f)[:2]
    dirs = dirs if dirs is not None else [(float(y), s) for y in range(0, 180, 15) for s in (1, -1)]
    found = []
    for yaw, lean in dirs:
        c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
        rot = np.array([[c, s], [-lean * s, lean * c]])           # part xy -> (u along the jaw, w toward the lean)
        rs = [seg @ rot.T if len(seg) else seg for seg in secs]
        allw = np.concatenate([r[:, :, 1].ravel() for r in rs if len(r)] or [np.zeros(0)])
        if not len(allw):
            continue
        w0 = float(allw.min()) - ROW_MM
        n_rows = int(math.ceil((float(allw.max()) - w0) / ROW_MM)) + 2
        ext = [_row_extents(r, w0, n_rows) for r in rs]
        LO, HI = np.array([e[0] for e in ext]), np.array([e[1] for e in ext])      # (layer, row)
        cu, cw = rot @ cen
        z_hi = h - min(min_cover_mm, cover_frac * h)            # deepest useful tip: the MIN_COVER rule, whole part
        tips = np.arange(tip_mm, z_hi + 1e-9, max(step_mm, (z_hi - tip_mm) / 24))
        for tilt in tilts:
            a = math.radians(tilt)
            sa, ca = math.sin(a), math.cos(a)
            for z_tip in tips:
                z_e = z_tip + FINGER_HALF_MM * sa                  # the finger's end (on its axis): the corner is lowest
                k0 = int(np.searchsorted(zs, z_tip - clear_mm))
                k1 = int(np.searchsorted(zs, z_e + FINGER_LEN_MM * ca + (FINGER_HALF_MM + clear_mm) * sa + clear_mm,
                                         side="right"))
                col = [(zs[kz], _strip_extents(LO[kz], HI[kz], zs[kz] - z_e, sa, ca, FINGER_HALF_MM + clear_mm,
                                               clear_mm), _strip_extents(LO[kz], HI[kz], zs[kz] - z_e, sa, ca,
                                                                         FINGER_HALF_MM, 0.0, PAD_LEN_MM))
                       for kz in range(k0, k1)]
                col = [cz for cz in col if cz[1] is not None]
                if not col:
                    continue
                lo_o = np.min([cz[1][0] for cz in col], axis=0)        # opening: the finger grown by the clearance
                hi_o = np.max([cz[1][1] for cz in col], axis=0)
                width = hi_o - lo_o
                ok = np.isfinite(width) & (width <= stroke_mm) & (width > 0.5)
                pads = [cz for cz in col if cz[2] is not None]
                if not ok.any() or not pads:
                    continue
                zk = np.array([cz[0] for cz in pads])
                L = np.array([cz[2][0] for cz in pads])               # what the pads themselves reach, per layer
                H = np.array([cz[2][1] for cz in pads])
                has = np.isfinite(L)
                top = np.where(has, zk[:, None], 0.0).max(0)
                ok &= z_tip <= top - np.minimum(min_cover_mm, cover_frac * top)
                t_lo, t_hi = has & (L <= lo_o + touch_mm), has & (H >= hi_o - touch_mm)
                cover = np.minimum(t_lo.sum(0), t_hi.sum(0)) * step_mm   # the pads touch the widest material on both
                ok &= cover >= min_touch_mm                               # sides, over this much height
                touch = t_lo | t_hi
                zc = (touch * zk[:, None]).sum(0) / np.maximum(touch.sum(0), 1)     # mean height of the contact
                for j in np.flatnonzero(ok):
                    we = w0 + (j + 0.5) * ROW_MM                   # finger end, across the jaw axis
                    pw, pz = we - FINGER_END_MM * sa, z_e + FINGER_END_MM * ca       # the pinch
                    pu = 0.5 * (lo_o[j] + hi_o[j])
                    wc = we - (zc[j] - z_e) * (sa / ca if ca > 1e-9 else 0.0)        # the contact: on the finger axis
                    d = math.hypot(pu - cu, wc - cw)
                    sc = min(cover[j], 8.0) / 8.0 * min(open_mm - width[j], 16.0) / 16.0 / (1.0 + d / 25.0) \
                        + 1e-3 * min(cover[j], 30.0) / 30.0          # (ties: the deeper cover)
                    x, y = np.linalg.solve(rot, [pu, pw])
                    gx, gy = np.linalg.solve(rot, [pu, wc])
                    found.append(((float(x), float(y), float(yaw), int(lean), int(tilt), float(pz), float(width[j]),
                                   float(z_tip), float(cover[j]), float(sc)), (float(gx), float(gy))))
    if not found:
        return []
    # spread: each pick is the best score x 0.7 per pick with the same frame x 0.6 per pick gripping the same thing
    # (width within 4 mm, contact within 10 mm), so that the planner tries other features and other arm poses early
    sc = np.array([cand[-1] * (prior(*cand[2:5]) if prior else 1.0) for cand, _ in found])
    frames, groups = {}, {}
    fid = np.array([frames.setdefault(cand[2:5], len(frames)) for cand, _ in found])
    gid = np.array([groups.setdefault((round(cand[6] / 4.0), round(gx / 10.0), round(gy / 10.0)), len(groups))
                    for cand, (gx, gy) in found])
    nf, ng = np.zeros(len(frames)), np.zeros(len(groups))
    keep, used = [], np.zeros(len(found), bool)
    while len(keep) < min(n_keep, len(found)):
        eff = np.where(used, -1.0, sc * 0.7 ** nf[fid] * 0.6 ** ng[gid])
        k = int(np.argmax(eff))
        used[k] = True
        nf[fid[k]] += 1
        ng[gid[k]] += 1
        keep.append(found[k][0])
    return keep


def simplify(vertices: np.ndarray, faces: np.ndarray, max_faces: int = 150_000, cell_mm: float = 0.1):
    """Vertex-clustering decimation (for a visual mesh: MuJoCo reads at most 200k STL faces): merge vertices on a
    grid, doubling the cell until at most max_faces non-degenerate triangles are left. -> (vertices, faces)."""
    v, f = np.asarray(vertices, float), np.asarray(faces)
    while len(f) > max_faces:
        q, inv = np.unique(np.round(v / cell_mm), axis=0, return_inverse=True)
        inv = inv.ravel()
        nf = inv[f]
        nf = nf[(nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])]
        nf = np.unique(nf, axis=0)
        sums = np.zeros((len(q), 3))
        np.add.at(sums, inv, v)
        v, f = sums / np.bincount(inv, minlength=len(q))[:, None], nf
        cell_mm *= 2.0
    return v, f
