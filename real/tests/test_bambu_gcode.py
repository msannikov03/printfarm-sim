"""BambuStudio G-code: object labels, names from the plate archive, skirt / start-G-code exclusion, the extruder offset.

The snippet in data/ is BambuStudio's own output for two generated models (real.mesh_io.cup_mesh and
l_bracket_mesh), trimmed to the header, the start G-code, layers 1-2 and the last layer."""
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from real.bambu import find_bambu, resolve_profile, profiles_dir
from real.gcode_objects import extract, parse_segments, read_label_names

DATA = Path(__file__).parent / "data"
TRUTH = json.loads((DATA / "bambu_p1s_plate.json").read_text())["meshes"]
OFFSET = np.array([0.0, -2.0])          # P1S "; extruder_offset = 0x2": G-code y = slicer plate y - 2 mm


def _archive(tmp_path, with_info=True):
    z = tmp_path / "plate.gcode.3mf"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("Metadata/plate_1.gcode", (DATA / "bambu_p1s_plate.gcode").read_text())
        if with_info:
            f.writestr("Metadata/slice_info.config", (DATA / "bambu_p1s_slice_info.config").read_text())
    return z


def test_bambu_labels_names_and_geometry(tmp_path):
    plan = extract(_archive(tmp_path))
    assert plan["method"] == "bambu_label" and plan["labelled"]
    assert plan["bed_mm"] == [256.0, 256.0] and plan["bed_source"] == "gcode:printable_area"
    by = {o["name"]: o for o in plan["objects"]}
    assert set(by) == set(TRUTH)
    for name, t in TRUTH.items():
        o = by[name]
        assert o["label_id"] == t["identify_id"]
        assert np.allclose(o["center_mm"], np.array(t["plate_center_mm"]) + OFFSET, atol=0.2), (name, o["center_mm"])
        assert o["height_mm"] == pytest.approx(t["height_mm"], abs=0.05)       # the last layer is in the snippet
        # extrusion centre lines: the outer wall sits half a line width (0.42 mm) inside the model's outline
        assert np.allclose(o["footprint_mm"], np.array(t["footprint_mm"]) - 0.42, atol=0.15), (name, o["footprint_mm"])
        assert abs(o["yaw_deg"]) < 0.5 or abs(abs(o["yaw_deg"]) - 180) < 0.5
    assert plan["purge_object"] is None                        # a Bambu printer purges into its chute
    assert plan["pick_order"][0] == "gen_l_bracket.stl"        # nearer the front, and taller
    assert plan["n_purge_or_start_segments"] > 0               # the machine's own prime lines: not an object


def test_bambu_plain_gcode_keeps_label_ids():
    plan = extract(text=(DATA / "bambu_p1s_plate.gcode").read_text())
    assert plan["method"] == "bambu_label"
    assert sorted(o["name"] for o in plan["objects"]) == ["id_47", "id_58"]
    assert parse_segments((DATA / "bambu_p1s_plate.gcode").read_text()).header["extruder_offset"] == "0x2"


def test_bambu_feature_skirt_is_not_part_of_the_object():
    sq = [(50, 50), (70, 50), (70, 70), (50, 70), (50, 50)]
    ring = [(40, 40), (80, 40), (80, 80), (40, 80), (40, 40)]
    g = ["G90", "M83", "; CHANGE_LAYER", "; Z_HEIGHT: 0.2", "G1 Z0.2", "; FEATURE: Skirt", f"G1 X{ring[0][0]} Y{ring[0][1]}"]
    g += [f"G1 X{x} Y{y} E0.5" for x, y in ring[1:]]
    g += ["; start printing object, unique label id: 5", "M624 AQAAAAAAAAA=", "; FEATURE: Outer wall",
          f"G1 X{sq[0][0]} Y{sq[0][1]}"] + [f"G1 X{x} Y{y} E0.5" for x, y in sq[1:]]
    g += ["; stop printing object, unique label id: 5", "M625"]
    plan = extract(text="\n".join(g))
    assert len(plan["objects"]) == 1 and plan["objects"][0]["name"] == "id_5"
    assert plan["objects"][0]["footprint_mm"] == pytest.approx([20.0, 20.0], abs=0.01)


def test_label_names_duplicates_get_the_id(tmp_path):
    z = tmp_path / "p.gcode.3mf"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("Metadata/slice_info.config", '<config><plate><object identify_id="7" name="clip.stl" skipped="false" />'
                   '<object identify_id="9" name="clip.stl" skipped="false" /><object identify_id="11" name="a &amp; b" '
                   'skipped="false" /></plate></config>')
    assert read_label_names(z) == {7: "clip.stl#7", 9: "clip.stl#9", 11: "a & b"}
    assert read_label_names(DATA / "bambu_p1s_plate.gcode") == {}


def test_profile_flattening_when_installed():
    exe = find_bambu()
    root = profiles_dir(exe) if exe else None
    if root is None:
        pytest.skip("BambuStudio not installed")
    m = resolve_profile("machine", "Bambu Lab P1S 0.4 nozzle", root)
    assert m["printable_area"] == ["0x0", "256x0", "256x256", "0x256"]        # from the parent profile
    assert "machine_start_gcode" in m and len(m["machine_start_gcode"]) > 100  # from the included template
    assert "inherits" not in m and "include" not in m
