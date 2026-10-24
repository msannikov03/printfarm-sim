import json
import math
import zipfile

import numpy as np
import pytest

from real.gcode_objects import extract, main, min_area_rect, parse_segments, pick_order
from real.synth_gcode import DEFAULT_PARTS, Part, SynthOpts, make_gcode

TRUTH = {p.name: p for p in DEFAULT_PARTS}


def _match(plan, labelled=True):
    objs = plan["objects"]
    assert len(objs) == len(DEFAULT_PARTS)
    for o in objs:
        if labelled:
            p = TRUTH[o["name"]]
        else:   # cluster names are generic: match by nearest centre
            p = min(DEFAULT_PARTS, key=lambda p: np.hypot(*(np.array(p.center) - o["center_mm"])))
        assert np.allclose(o["center_mm"], p.center, atol=0.6), (o, p)
        assert o["height_mm"] == pytest.approx(p.height, abs=0.05)
        long_, short = sorted(p.size, reverse=True)
        assert o["footprint_mm"][0] == pytest.approx(long_, abs=1.0)
        assert o["footprint_mm"][1] == pytest.approx(short, abs=1.0)
        if p.shape == "box" and abs(p.size[0] - p.size[1]) > 2:      # yaw only defined for non-square boxes
            assert (o["yaw_deg"] - p.yaw_deg + 90) % 180 - 90 == pytest.approx(0, abs=1.0)


@pytest.mark.parametrize("style,method", [("orca", "exclude_object"), ("orca_comments", "printing_object"),
                                          ("m486", "m486"), ("cura", "cura_mesh")])
def test_labelled_styles(style, method):
    plan = extract(text=make_gcode(opts=SynthOpts(style=style)))
    assert plan["method"] == method and plan["labelled"]
    _match(plan)


def test_orca_bed_from_printable_area_and_define_center():
    plan = extract(text=make_gcode(opts=SynthOpts(style="orca", bed=(300, 300))))
    assert plan["bed_mm"] == [300, 300] and plan["bed_source"] == "gcode:printable_area"
    by = {o["name"]: o for o in plan["objects"]}
    assert by["cube"]["define_center_mm"] == [105.0, 110.0]


def test_unlabelled_cluster_excludes_commentless_purge_and_skirt():
    # no labels, no layer markers, purge line without comment, skirt: only geometry can sort it out
    opts = SynthOpts(style="none", purge_comment=False, layer_markers=False, skirt=True)
    plan = extract(text=make_gcode(opts=opts))
    assert plan["method"] == "cluster" and not plan["labelled"]
    _match(plan, labelled=False)
    reasons = " ".join(e["reason"] for e in plan["excluded"])
    assert "purge" in reasons and "skirt" in reasons
    purge = [e for e in plan["excluded"] if "purge" in e["reason"]][0]
    assert purge["center_mm"][0] < 10                      # the left-edge prime line


def test_unlabelled_with_layer_markers_drops_start_gcode():
    plan = extract(text=make_gcode(opts=SynthOpts(style="none", purge_comment=False)))
    assert plan["method"] == "cluster"
    _match(plan, labelled=False)
    assert plan["n_purge_or_start_segments"] >= 2
    assert not plan["excluded"]                            # removed before clustering (start region)


def test_force_cluster_on_labelled_file_matches_labels():
    plan = extract(text=make_gcode(opts=SynthOpts(style="orca", skirt=True)), force_cluster=True)
    _match(plan, labelled=False)


def test_relative_extrusion_and_arcs():
    plan = extract(text=make_gcode(opts=SynthOpts(style="orca_comments", relative_e=True)))
    _match(plan)
    seg = parse_segments(make_gcode(opts=SynthOpts(style="none")))
    assert len(seg.z) > 100


def test_purge_comment_flags_segments():
    seg = parse_segments(make_gcode(opts=SynthOpts(style="orca")))
    assert seg.purge.sum() == 2 and seg.has_layer_markers


def test_min_area_rect_rotated_box():
    p = Part("x", (50, 50), (40, 10), 5, yaw_deg=25)
    c, (w, h), yaw = min_area_rect(p.outline())
    assert np.allclose(c, [50, 50], atol=1e-6) and w == pytest.approx(40) and h == pytest.approx(10)
    assert yaw == pytest.approx(25)


def test_pick_order_front_to_back_tallest_first():
    objs = [dict(name="a", center_mm=[50, 100], height_mm=5), dict(name="b", center_mm=[150, 30], height_mm=5),
            dict(name="c", center_mm=[100, 35], height_mm=30), dict(name="d", center_mm=[60, 180], height_mm=50)]
    assert pick_order(objs, (220, 220), "front") == ["c", "b", "a", "d"]
    assert pick_order(objs, (220, 220), "back")[0] == "d"


def test_3mf_archive_and_cli(tmp_path):
    g = make_gcode(opts=SynthOpts(style="orca"))
    z = tmp_path / "plate.gcode.3mf"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("Metadata/plate_1.gcode", g)
        f.writestr("3D/3dmodel.model", "<model/>")
    out = tmp_path / "plan.json"
    main([str(z), "-o", str(out)])
    plan = json.loads(out.read_text())
    assert plan["method"] == "exclude_object" and len(plan["objects"]) == 3


def test_sim_plates_parity():
    """On the simulator's own plates (runs/plates, written by jobs/plates.py) we agree with plate.json and
    with the sim's parser jobs/gcode_prior.py."""
    from real import REPO
    gp = pytest.importorskip("jobs.gcode_prior")
    files = sorted((REPO / "runs" / "plates").glob("*/plate.gcode"))
    if not files:
        pytest.skip("no sim plates in runs/plates (python -m jobs.plates writes them)")
    for f in files[:3]:
        truth = json.loads((f.parent / "plate.json").read_text())
        ours = {o["name"]: o for o in extract(f)["objects"]}
        theirs, _ = gp.parse(str(f))
        assert set(ours) == {o["name"] for o in truth["objects"]} == set(theirs)
        for o in truth["objects"]:
            assert np.allclose(ours[o["name"]]["center_mm"], o["center_mm"], atol=0.5)
            assert ours[o["name"]]["height_mm"] == pytest.approx(o["height_mm"], abs=0.01)
            assert ours[o["name"]]["height_mm"] == pytest.approx(theirs[o["name"]]["height"], abs=0.01)


def test_creality_purge_conventions():
    """KE start G-code ';Draw the first line' and the K2 ';FLUSH_START' block count as purge."""
    from real.gcode_objects import parse_segments
    g = "\n".join(["G90", "M82", "G92 E0", "G1 X0.1 Y20 Z0.3 F5000", "G1 X0.1 Y200 E15 F1500 ;Draw the first line",
                    "; FLUSH_START", "G1 X5 Y20 E25", "; FLUSH_END", "G1 X10 Y20 E26 ; ordinary"])
    seg = parse_segments(g)
    assert seg.purge.tolist() == [True, True, False]
