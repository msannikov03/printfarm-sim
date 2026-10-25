"""Mesh loading (STL / 3MF), footprints, top faces and grasp cross-sections of real parts."""
import math
import zipfile

import numpy as np
import pytest

from real.mesh_io import (PartGeometry, cup_mesh, footprint, grasp_section, l_bracket_mesh, load_3mf, load_mesh,
                          load_stl, rot_z, volume_mm3, write_3mf, write_stl)


def test_stl_binary_and_ascii(tmp_path):
    m = l_bracket_mesh()
    p = write_stl(tmp_path / "b.stl", m.vertices, m.faces)
    b = load_stl(p)
    assert len(b.vertices) == len(m.vertices) and len(b.faces) == len(m.faces)
    assert volume_mm3(b) == pytest.approx(50 * 20 * 4 + 4 * 20 * 26)
    tri = m.vertices[m.faces]
    ascii_ = "solid t\n" + "".join("facet normal 0 0 0\nouter loop\n" + "".join(f"vertex {x} {y} {z}\n" for x, y, z in t)
                                   + "endloop\nendfacet\n" for t in tri) + "endsolid t\n"
    (tmp_path / "a.stl").write_text(ascii_)
    a = load_mesh(tmp_path / "a.stl")[0]
    assert volume_mm3(a) == pytest.approx(volume_mm3(m))


def test_3mf_build_transforms_and_names(tmp_path):
    T = np.eye(4)
    T[:3, :3] = rot_z(30.0)
    T[:3, 3] = [120.0, 80.0, 0.0]
    p = write_3mf(tmp_path / "x.3mf", [cup_mesh(name="box & lid"), l_bracket_mesh()], [T, np.eye(4)])
    ms = load_3mf(p)
    assert [m.name for m in ms] == ["box & lid", "l_bracket"]
    c, (L, S), yaw, h = footprint(ms[0])
    assert np.allclose(c, [120, 80], atol=1e-3) and (L, S, h) == pytest.approx((60, 40, 25), abs=1e-3)
    assert yaw == pytest.approx(30.0, abs=1e-3)


def test_3mf_components_external_objects_and_bambu_ids(tmp_path):
    """BambuStudio layout: the build item points at an object made of components in 3D/Objects/*.model (p:path), and
    Metadata/model_settings.config (not valid XML: unescaped quotes) maps the object to its G-code label id."""
    m = cup_mesh()
    vs = "".join(f'<vertex x="{x}" y="{y}" z="{z}"/>' for x, y, z in m.vertices)
    ts = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in m.faces)
    ns = 'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" ' \
         'xmlns:p="http://schemas.microsoft.com/3dmanufacturing/production/2015/06"'
    obj = f'<model unit="millimeter" {ns}><resources><object id="1" type="model"><mesh><vertices>{vs}</vertices>' \
          f'<triangles>{ts}</triangles></mesh></object></resources><build/></model>'
    root = f'<model unit="millimeter" {ns}><resources><object id="2" type="model"><components>' \
           f'<component p:path="/3D/Objects/object_1.model" objectid="1" transform="1 0 0 0 1 0 0 0 1 0 0 12.5"/>' \
           f'</components></object></resources><build><item objectid="2" transform="1 0 0 0 1 0 0 0 1 100 90 -12.5"/>' \
           f'</build></model>'
    cfg = '<config><object id="2"><metadata key="name" value="case.stl"/>' \
          '<metadata key="compatible_printers" value=""P1S";"X1""/></object><plate><model_instance>' \
          '<metadata key="object_id" value="2"/><metadata key="identify_id" value="77"/></model_instance></plate></config>'
    p = tmp_path / "bambu.3mf"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("3D/3dmodel.model", root)
        z.writestr("3D/Objects/object_1.model", obj)
        z.writestr("Metadata/model_settings.config", cfg)
    (mesh,) = load_3mf(p)
    assert mesh.name == "case.stl" and mesh.meta["identify_id"] == 77
    assert np.allclose(mesh.vertices.min(0), [70, 70, 0]) and np.allclose(mesh.vertices.max(0), [130, 110, 25])


def test_3mf_units(tmp_path):
    p = write_3mf(tmp_path / "u.3mf", [cup_mesh(2.0, 1.0, 0.5, 0.1, 0.1)])
    with zipfile.ZipFile(p) as z:
        files = {n: z.read(n) for n in z.namelist()}
    files["3D/3dmodel.model"] = files["3D/3dmodel.model"].replace(b'unit="millimeter"', b'unit="inch"')
    with zipfile.ZipFile(p, "w") as z:
        for n, b in files.items():
            z.writestr(n, b)
    _, (L, S), _, h = footprint(load_3mf(p)[0])
    assert (L, S, h) == pytest.approx((50.8, 25.4, 12.7))


def test_part_geometry_top_face_and_grasp_section():
    g = PartGeometry.from_mesh(l_bracket_mesh())
    tx, ty, ox, oy, area = g.top_mm
    assert (tx, ty, ox, oy) == pytest.approx((4.0, 20.0, -23.0, 0.0)) and area == pytest.approx(80.0, rel=0.4)
    h = g.dims_mm[2]
    assert grasp_section(g.vertices, g.faces, [0.6 * h, h - 4]) == pytest.approx((-23.0, 0.0, 4.0, 20.0))   # upright
    cup = PartGeometry.from_mesh(cup_mesh())
    assert 60 * 40 - 56 * 36 <= cup.top_mm[4] < 0.3 * 60 * 40       # the rim (1 mm grid), not the whole footprint
    assert grasp_section(cup.vertices, cup.faces, [15, 21]) == pytest.approx((0, 0, 60, 40))           # outer walls
    assert cup.volume_cm3 == pytest.approx((60 * 40 * 25 - 56 * 36 * 23) / 1000)


def test_part_geometry_frame_follows_the_yaw_hint():
    m = l_bracket_mesh()
    m.vertices = m.vertices @ rot_z(90.0).T + [100.0, 50.0, 0.0]
    g = PartGeometry.from_mesh(m, yaw_hint_deg=0.0)          # the footprint rectangle, turned to the hint's frame
    assert g.yaw_deg % 180 == pytest.approx(0.0, abs=1e-6) and g.dims_mm[:2] == pytest.approx((20.0, 50.0))
    assert g.center_mm == pytest.approx((100.0, 50.0))
    g2 = PartGeometry.from_mesh(m, yaw_hint_deg=90.0)
    assert g2.yaw_deg == pytest.approx(90.0) and g2.dims_mm[:2] == pytest.approx((50.0, 20.0))
    assert g2.top_mm[2:4] == pytest.approx((-23.0, 0.0))     # the upright, in the part's own frame
    assert math.isclose(g2.volume_cm3, 6.08)


def test_3mf_leaves_out_modifier_volumes(tmp_path):
    """A BambuStudio object made of a printed part and a modifier volume: only the printed part is the model."""
    cup, mod = cup_mesh(), cup_mesh(10, 10, 40, 1, 1)
    ns = 'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02"'
    mesh_xml = lambda m: ("<mesh><vertices>" + "".join(f'<vertex x="{x}" y="{y}" z="{z}"/>' for x, y, z in m.vertices)
                          + "</vertices><triangles>" + "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>'
                                                               for a, b, c in m.faces) + "</triangles></mesh>")
    model = (f'<model unit="millimeter" {ns}><resources><object id="1" type="model">{mesh_xml(cup)}</object>'
             f'<object id="2" type="model">{mesh_xml(mod)}</object><object id="3" type="model"><components>'
             f'<component objectid="1"/><component objectid="2"/></components></object></resources>'
             f'<build><item objectid="3" transform="1 0 0 0 1 0 0 0 1 100 100 0"/></build></model>')
    cfg = ('<config><object id="3"><metadata key="name" value="lid"/><part id="1" subtype="normal_part"></part>'
           '<part id="2" subtype="modifier_part"></part></object></config>')
    p = tmp_path / "m.3mf"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("3D/3dmodel.model", model)
        z.writestr("Metadata/model_settings.config", cfg)
    (m,) = load_3mf(p)
    assert len(m.vertices) == len(cup.vertices) and footprint(m)[3] == pytest.approx(25.0)


def test_part_sunk_into_the_bed_is_cut_there():
    m = cup_mesh()
    m.vertices[:, 2] -= 5.0                       # 5 mm below the plate: the slicer prints the 20 mm above it
    g = PartGeometry.from_mesh(m, bed_z=0.0)
    assert g.dims_mm[2] == pytest.approx(20.0) and g.vertices[:, 2].min() == pytest.approx(0.0)
    assert g.volume_cm3 == pytest.approx((60 * 40 - 56 * 36) * 20 / 1000)    # the floor was below the bed


# ---- grasp search over the part's own layers (real.mesh_io.grasp_candidates)

def _block(x0, x1, y0, y1, z0, z1):
    """Closed axis-aligned box, outward normals."""
    v = np.array([(x, y, z) for z in (z0, z1) for y in (y0, y1) for x in (x0, x1)], float)
    f = [(0, 2, 3), (0, 3, 1), (4, 5, 7), (4, 7, 6), (0, 1, 5), (0, 5, 4), (2, 6, 7), (2, 7, 3), (0, 4, 6), (0, 6, 2),
         (1, 3, 7), (1, 7, 5)]
    return v, np.array(f, np.int64)


def _union(*blocks):
    vs, fs, n = [], [], 0
    for v, f in blocks:
        vs.append(v)
        fs.append(f + n)
        n += len(v)
    return np.vstack(vs), np.vstack(fs)


def test_grasp_search_finds_the_low_wide_outline():
    """A 60 x 40 x 6 base with a small 8 x 8 boss to 26 mm at one end: the fixed grasp heights (60 % of the height,
    4 mm under the top) see only the boss; the search over the layers also finds the base, low, across its width."""
    from real.mesh_io import grasp_candidates
    v, f = _union(_block(-30, 30, -20, 20, 0, 6), _block(18, 26, -4, 4, 6, 26))
    h = 26.0
    assert grasp_section(v, f, [0.6 * h, h - 4])[2:] == pytest.approx((8.0, 8.0))     # the boss only
    cands = grasp_candidates(v, f, h, clear_mm=3.0)
    assert cands and all(c[6] <= 70.0 for c in cands)                                 # within the stroke
    base = [c for c in cands if c[7] < 6.0 - 0.6 * 6.0 + 1e-6 and abs(c[6] - 40.0) < 1.0]
    assert base, "no grasp on the base"
    c = base[0]
    assert abs(math.sin(math.radians(c[2]))) > 0.99                                  # jaws across the 40 mm side
    assert c[7] >= 1.8 and c[8] >= 1.0                                                # tips above the bed, cover
    boss = [c for c in cands if abs(c[6] - 8.0) < 1.0]
    assert boss and min(c[7] for c in boss) >= 6.0 + 3.0 - 1e-6     # the boss: tips clear the base below by clear_mm


def test_grasp_search_turns_the_jaws_across_the_side_that_fits():
    """A 90 x 50 x 20 block: wider than the 70 mm stroke along x, not along y. Every candidate fits the stroke, and
    the best closes across the 50 mm side."""
    from real.mesh_io import grasp_candidates
    v, f = _block(-45, 45, -25, 25, 0, 20)
    cands = grasp_candidates(v, f, 20.0, clear_mm=3.0)
    assert cands and all(c[6] <= 70.0 for c in cands)
    best = cands[0]
    assert best[6] == pytest.approx(50.0, abs=0.5) and abs(math.sin(math.radians(best[2]))) > 0.99
    # jaw axes at 45 deg or less from x see more than the stroke (the finger's strip is slanted and wide)
    assert all(abs(math.sin(math.radians(c[2]))) > 0.7 for c in cands)


def test_grasp_search_leaves_flat_parts_alone():
    """Under about 4.5 mm (tips 1.8 mm above the bed, and 60 % of the height covered) no candidate: a 30 x 30 x 3 sheet
    stays for another tool. A 6 mm part has some."""
    from real.mesh_io import grasp_candidates
    assert grasp_candidates(*_block(-15, 15, -15, 15, 0, 3), 3.0) == []
    assert grasp_candidates(*_block(-15, 15, -15, 15, 0, 6), 6.0, clear_mm=3.0)


def test_grasp_search_closes_on_the_outer_walls_of_an_enclosure():
    from real.mesh_io import grasp_candidates
    cup = PartGeometry.from_mesh(cup_mesh())
    cands = grasp_candidates(cup.vertices, cup.faces, 25.0, clear_mm=3.0)
    assert cands and {round(c[6]) for c in cands} <= {40, 60} | set(range(41, 71))
    assert min(c[6] for c in cands) == pytest.approx(40.0, abs=0.5)                   # across the outer walls


def test_candidate_dirs_keep_the_jaws_out_of_the_printer_direction():
    """Jaw axes the arm can reach the bed with: within JAW_YAW_MAX of the door plane, leaning into the printer."""
    from plan.mesh_grasp import JAW_YAW_MAX, candidate_dirs
    for part_yaw in (0.0, 30.0, 90.0, -62.3):
        dirs = candidate_dirs(part_yaw)
        assert dirs
        for th, lean in dirs:
            phi = (part_yaw + th + 90.0) % 180.0 - 90.0
            assert abs(phi) <= JAW_YAW_MAX
            normal_y = lean * math.cos(math.radians(part_yaw + th))       # the lean (jaw normal) on the bed, y part
            assert normal_y > 0
