"""Real 3D models in the cell: mesh parts in the MuJoCo scene, the plan from a sliced plate, and one full unload.

The plates here are built from generated geometry (real.mesh_io.cup_mesh / l_bracket_mesh) and a synthetic labelled
G-code (real.synth_gcode), so they run without BambuStudio or downloaded models; the BambuStudio test slices the same
models for real and is skipped when the program is not installed. docs/real_models.md has the full evaluation."""
import hashlib
import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pytest

from farm import catalog as C
from real.mesh_io import cup_mesh, footprint, l_bracket_mesh, rot_z, write_3mf
from real.synth_gcode import Part, SynthOpts, make_gcode

ROOT = Path(__file__).resolve().parents[1]
# (mesh, centre on the bed (mm), yaw (deg)): 35 mm apart, the cell's jaw clearance (jobs/plates.py packs with 25 mm)
LAYOUT = [(cup_mesh(name="enclosure"), (90.0, 120.0), 0.0), (l_bracket_mesh(name="bracket"), (180.0, 125.0), 90.0)]


def _plate(tmp_path):
    """A .gcode.3mf: the models at LAYOUT in 3D/3dmodel.model and a labelled G-code of their footprints."""
    meshes, T, parts = [], [], []
    for m, c, yaw in LAYOUT:
        M = np.eye(4)
        M[:3, :3] = rot_z(yaw)
        M[:3, 3] = [c[0], c[1], 0.0]
        meshes.append(m)
        T.append(M)
        (fc, (L, S), fyaw, h) = footprint(m.vertices @ M[:3, :3].T + M[:3, 3])
        parts.append(Part(m.name, tuple(fc), (L, S), h, yaw_deg=fyaw))
    g = make_gcode(parts, SynthOpts(style="orca", bed=(256.0, 256.0), purge=False))
    return write_3mf(tmp_path / "plate.gcode.3mf", meshes, T, extra={"Metadata/plate_1.gcode": g})


@pytest.fixture
def plate(tmp_path):
    from sim.real_parts import plate_from_slice
    rp = plate_from_slice(_plate(tmp_path), printer=4, cache_dir=tmp_path / "meshes", plate_id="T", sku_prefix="t")
    yield rp
    for p in rp.parts:                     # the scene of later tests is built without them
        C.CATALOG.pop(p.sku, None)


def test_open_model_manifest():
    man = json.loads((ROOT / "real" / "open_models.json").read_text())["models"]
    assert len({m["id"] for m in man}) == len(man) >= 3
    for m in man:
        assert m["license"] in ("CC0-1.0", "CC-BY-4.0", "public domain", "MIT"), m["id"]
        assert m["url"].startswith("https://") and m["page"].startswith("https://")
        assert re.fullmatch(r"[0-9a-f]{40}", m["sha1"]) and m["author"]


def test_fetched_models_load_when_present():
    from real.fetch_models import available
    from real.mesh_io import load_mesh
    have = available()
    if not have:
        pytest.skip("open models not fetched (python -m real.fetch_models)")
    man = {m["id"]: m for m in json.loads((ROOT / "real" / "open_models.json").read_text())["models"]}
    for mid, p in have.items():
        assert hashlib.sha1(p.read_bytes()).hexdigest() == man[mid]["sha1"]
        (m,) = load_mesh(p)
        assert len(m.faces) >= 12 and np.ptp(m.vertices, axis=0).min() > 0.5


def test_plan_from_a_sliced_plate(plate):
    """Each G-code object is matched to its mesh; the plan (G-code footprint) agrees with the mesh's own footprint,
    and the SKU carries the mesh's top face and grasp cross-section."""
    from sim.real_parts import parse_errors
    assert len(plate.parts) == 2 and not plate.unmatched and plate.plate["purge_tab"] is None
    by = {p.name: p for p in plate.parts}
    for p in plate.parts:
        e = parse_errors(p)
        assert e["center_err_mm"] < 0.3 and abs(e["long_err_mm"]) < 0.6 and abs(e["short_err_mm"]) < 0.6
        assert abs(e["height_err_mm"]) < 0.25 and abs(e["yaw_err_deg"]) < 1.0
    k = C.CATALOG[by["bracket"].sku]
    assert k.kind == "box" and k.pool == 1 and len(k.col_meshes) >= 2
    assert abs(k.grasp_mm[2] - 4.0) < 0.6 or abs(k.grasp_mm[3] - 4.0) < 0.6      # the jaws close on the upright
    assert np.hypot(*k.top_mm[2:4]) == pytest.approx(23.0, abs=0.5)               # its top face is off-centre
    enc = C.CATALOG[by["enclosure"].sku]
    assert 60 * 40 - 56 * 36 <= enc.top_mm[4] < 0.3 * 60 * 40                     # a rim, not a lid


def test_mesh_parts_settle_and_the_camera_finds_them(plate):
    import mujoco
    from farm.world import World
    from perceive.verify import BedCamera, estimate, plate_objects
    from sim.real_parts import spawn_truth
    w = World()
    placed = spawn_truth(w, plate)
    w.step(300)
    cam = BedCamera(w, 4, corrections=True)
    mujoco.mj_forward(w.m, w.d)
    est, _ = estimate(cam.points_bed_mm(), plate_objects(plate.plate))
    for p in plate.parts:
        b = placed[p.obj["name"]]
        pos, q = w.part_pose(b)
        assert abs(pos[2] - (C.BENCH_H + C.BED_ZL)) < 0.001                       # resting flat on the bed
        assert abs(q[1]) < 0.01 and abs(q[2]) < 0.01
        e = est[p.obj["name"]]
        assert e["found"] and np.hypot(*(np.array(e["center_mm"]) - p.geom.center_mm)) < 1.0, (p.name, e)
    cam.close()


def test_wider_than_the_gripper_is_routed_to_reject():
    spec = importlib.util.spec_from_file_location("eval_real_models", ROOT / "scripts" / "eval_real_models.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    assert ev.route(C.SKU("x", "box", (175.0, 107.0, 16.0))) == "reject"
    assert ev.route(C.SKU("x", "box", (100.0, 66.0, 20.0))) == "tray"
    assert ev.route(C.SKU("x", "box", (60.0, 31.0, 48.0))) == "tote"


@pytest.mark.slow
def test_unload_real_mesh_parts_end_to_end(plate, tmp_path):
    """Door, chamber camera, grasp from the plan, pick into the totes, door: both mesh parts end in their tote."""
    spec = importlib.util.spec_from_file_location("eval_real_models", ROOT / "scripts" / "eval_real_models.py")
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    res = ev.episode(plate, tmp_path, realism="ideal", printer=4, seed=0)
    assert "error" not in res, res.get("error")
    assert res["door_open"]["ok"] and res["door_close"]["ok"]
    for r in res["parts"]:
        assert r["grasp_found"] and r["placed"] and r["end"] == "own_slot", r
        assert r["perception"]["center_err_mm"] < 1.0


@pytest.mark.skipif(not __import__("real.bambu", fromlist=["find_bambu"]).find_bambu(),
                    reason="BambuStudio not installed")
def test_bambustudio_slice_and_plan(tmp_path):
    from real.bambu import slice_models
    from real.mesh_io import write_stl
    from sim.real_parts import pack_models, parse_errors, plate_from_slice
    files = [write_stl(tmp_path / f"{m.name}.stl", m.vertices, m.faces) for m, _, _ in LAYOUT]
    packed, n_in, n_out = pack_models(files, tmp_path / "packed.3mf")
    assert (n_in, n_out) == (2, 0)
    out = slice_models([packed], tmp_path / "slice", arrange=False)
    rp = plate_from_slice(out, cache_dir=tmp_path / "m", plate_id="B", sku_prefix="b")
    try:
        assert rp.frame_offset_mm == pytest.approx((0.0, -2.0))       # the P1S extruder offset
        assert len(rp.parts) == 2 and not rp.unmatched
        for p in rp.parts:
            e = parse_errors(p)
            assert e["center_err_mm"] < 0.2 and -0.6 < e["short_err_mm"] < 0.0 and abs(e["height_err_mm"]) < 0.1
    finally:
        for p in rp.parts:
            C.CATALOG.pop(p.sku, None)


def test_layer_grasp_search_is_a_fallback_for_mesh_skus_only(monkeypatch):
    """plan_grasp keeps its own grasp first; only a real part (a SKU with grasp_alts) whose fixed-height grasp fails
    goes on to the search over its layers. Primitives never do: their planning is unchanged."""
    import plan.mesh_grasp as MG
    import plan.unload as U
    assert not any(k.grasp_alts for k in C.CATALOG.values())         # the catalogue: primitives only
    calls, found = [], {"fixed": None}
    monkeypatch.setattr(U, "grasp_search", lambda *a, **k: found["fixed"])
    monkeypatch.setattr(MG, "mesh_grasp", lambda ik, w, i, body, est, rng, n: calls.append(body) or {"mesh": ()})

    class W:                                                            # just what plan_grasp reads
        part_sku = {"prim": "block", "real": "t_mesh"}
        part_body = {"prim": 1, "real": 2}

    C.register_sku(C.SKU("t_mesh", "box", (30.0, 20.0, 10.0), grasp_alts=((0.0,) * 10,), pool=1))
    est = {"center_mm": [128.0, 100.0], "yaw_deg": 0.0}
    try:
        assert U.plan_grasp(None, W(), 4, "prim", est=est) is None and calls == []
        assert U.plan_grasp(None, W(), 4, "real", est=est) == {"mesh": ()} and calls == ["real"]
        found["fixed"] = {"fixed": True}                                # its own grasp found: no fallback
        assert U.plan_grasp(None, W(), 4, "real", est=est) == {"fixed": True} and calls == ["real"]
        found["fixed"] = None
        monkeypatch.setattr(MG, "ENABLED", False)
        assert U.plan_grasp(None, W(), 4, "real", est=est) is None and calls == ["real"]
    finally:
        C.CATALOG.pop("t_mesh", None)


def test_plate_parts_carry_layer_grasps(plate):
    """The SKUs of a sliced plate carry the grasps searched over their layers: within the stroke, tips above the bed,
    jaw axes within JAW_YAW_MAX of the door plane on the bed."""
    from plan.mesh_grasp import JAW_YAW_MAX
    for p in plate.parts:
        k = C.CATALOG[p.sku]
        assert k.grasp_alts, p.name
        for cx, cy, yaw, lean, tilt, pz, wd, z_tip, cover, sc in k.grasp_alts:
            assert 0 < wd <= C.GRIPPER_MAX_MM and z_tip >= 1.5 and cover >= 2.0 and tilt in (50, 65, 80)
            assert abs((p.obj["yaw_deg"] + yaw + 90.0) % 180.0 - 90.0) <= JAW_YAW_MAX + 1e-6
