"""Clearing fallback (plan/sweep.py): the deep pushes for what cannot be picked or raked into the tray.

Fast checks of the push frames, the planner order and the stroke memory, plus two physical checks on the default cell:
a flat gear lump deep in the bed (row A) and a bar lying into the printer (both rows) leave the bed with a rail push.
"""
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from farm import catalog as C  # noqa: E402
import plan.sweep as SW  # noqa: E402


@pytest.mark.parametrize("s", [1, -1])
@pytest.mark.parametrize("push_x", [1.0, -1.0])
def test_deep_frames_push_with_a_vertical_face(s, push_x):
    door_in = np.array([0.0, s, 0.0])
    push = s * np.array([push_x, 0.0, 0.0])
    frames = SW.deep_frames(push, door_in)
    assert frames
    for tilt, yw, face, R in frames:
        assert R[:, 2] @ door_in > np.sin(np.radians(tilt)) * np.cos(np.radians(yw)) - 1e-9   # tip into the printer
        assert R[:, 2][2] < 0                                                                       # pointing down
        n = R[:, 1] if face == "pads" else R[:, 0]          # normal of the pushing face of the closed fingers
        assert abs(n[2]) < 1e-9                             # vertical face: the push is horizontal, no wedge
        assert abs(n @ push) >= 0.5


def test_back_push_leans_steeply():
    for t, yw, a in SW._approaches(np.array([0.0, 1.0, 0.0]), [t for t in SW.DEEP_TILTS if t >= SW.BACK_TILT_MIN]):
        assert t >= SW.BACK_TILT_MIN                         # the tip face pushes mostly horizontally
        assert a[1] == pytest.approx(np.sin(np.radians(t)) * np.cos(np.radians(yw)))
        assert -a[2] == pytest.approx(np.cos(np.radians(t)))


def _blob(cx, cy, wx=30.0, wy=30.0, h=10.0):
    return {"lo": np.array([cx - wx / 2, cy - wy / 2]), "hi": np.array([cx + wx / 2, cy + wy / 2]), "h": h, "n": 500,
            "rect": ((cx, cy), (wx, wy), 0.0)}


def test_plan_order_and_skips(monkeypatch):
    calls = []
    monkeypatch.setattr(SW, "plan_rake", lambda ik, w, i, b, rng, d: calls.append(("rake", d)))
    monkeypatch.setattr(SW, "plan_push", lambda ik, w, i, b, rng, kind, d: calls.append((kind, d)))
    assert SW._plan_any(None, None, 4, _blob(60, 150), None) is None
    assert calls == [("rake", "front"), ("rake", "left"), ("rake", "right"), ("rail", "left"), ("rail", "right"),
                     ("side", "left"), ("side", "right"), ("back", "back")]
    calls.clear()                  # a long blob lying into the printer: sideways first, never off the back (too long)
    SW._plan_any(None, None, 4, _blob(200, 120, wx=18, wy=150), None, skip={("behind", "right"), ("rail", "right")})
    assert calls == [("rake", "left"), ("rake", "front"), ("rail", "left"), ("side", "right"), ("side", "left")]
    calls.clear()                  # too wide for the gap beside the bed: no side push
    SW._plan_any(None, None, 4, _blob(60, 150, wx=80, wy=40), None)
    assert [c for c in calls if c[0] != "rake"] == [("back", "back")]


def test_extent_for_the_gap_fit():
    gear = {"lo": np.array([35.7, 77.0]), "hi": np.array([78.5, 121.9]),           # a 42 mm gear as the camera saw it:
            "rect": ((56.5, 97.7), (41.0, 41.2), -48.4)}                          # its outline rectangle came out rotated
    assert max(SW._extent(gear)) < 45.0
    plate = {"lo": np.array([44.0, 59.0]), "hi": np.array([102.0, 182.0]),        # 50 x 120 plate lying into the printer:
             "rect": ((73.0, 120.0), (50.5, 120.4), 0.0)}                         # the point box includes side faces
    assert SW._extent(plate)[0] == pytest.approx(50.5)
    assert SW._extent(plate)[0] <= 1000 * SW.SIDE_GAP - SW.FIT_MARGIN_MM          # fits beside the bed


def test_stroke_memory():
    class Ik:
        clear_hist = [(4, np.array([100.0, 150.0]), ("rail", "left"), 10.0),
                      (5, np.array([100.0, 150.0]), ("side", "left"), 10.0),          # another printer
                      (4, np.array([160.0, 150.0]), ("back", "back"), 10.0),          # another blob
                      (4, np.array([102.0, 151.0]), ("behind", "front"), -1000.0)]    # an earlier unload
    assert SW._tried(Ik, 4, _blob(103, 148), 20.0) == {("rail", "left")}


def test_push_lane_must_be_clear():
    from plan.ik import Belief

    class W:
        part_body = {"p": 7}
        part_sku = {"p": "cube"}

    class Ik:
        pass

    target = _blob(100, 150)
    other = _blob(200, 152)                                    # to the right of the target, same depth
    Ik.belief = Belief()
    Ik.belief.raw = [dict(target, lo=target["lo"] + 0.7, hi=target["hi"] + 0.7), other]   # re-scanned target
    assert SW._corridor_clear(Ik, W, 4, target, (-1.0, 0.0), set())          # left: clear
    assert not SW._corridor_clear(Ik, W, 4, target, (1.0, 0.0), set())       # right: the other blob is in the lane
    assert SW._corridor_clear(Ik, W, 4, target, (0.0, 1.0), set())           # back: clear
    xc, s = C.printer_frame(4)
    Ik.belief.raw = [target]
    Ik.belief.parts = {"p": (np.array(C.bed_mm_to_world(4, 100, 220)), None, 0.0)}   # a part behind it
    assert not SW._corridor_clear(Ik, W, 4, target, (0.0, 1.0), set())
    assert SW._corridor_clear(Ik, W, 4, target, (0.0, 1.0), {7})              # ... unless it is the blob itself


@pytest.fixture(scope="module")
def quiet():
    import plan.door as D
    old = D.VERBOSE
    D.VERBOSE = False
    yield
    D.VERBOSE = old


def _case(sku, x, y, yaw, printer, tip=False):
    import sys
    sys.path.insert(0, "scripts")
    from test_rake import run_case
    return run_case(sku, x, y, yaw, printer, tip, realism="realistic")


@pytest.mark.skipif(not C.TWO_ROWS, reason="the deep-bed cases are the two-row cell's")
def test_deep_gear_lump_leaves_the_bed(quiet):
    """A 42 x 10 mm gear lump 150 mm deep: no grasp, no feasible front rake (and a steep one would wedge it into the
    bed); the carriage pushes it off a side edge. Truth is used only to score."""
    r = _case("gear", 128, 150, 20, 4)
    assert r["end"] != "on_bed" and r["robot_bed_empty"], r
    assert any(k.startswith(("rail", "side", "back")) for k in r["how"][0]), r


@pytest.mark.skipif(not C.TWO_ROWS, reason="large parts: two-row cell")
@pytest.mark.parametrize("printer", [4, 14])
def test_unpickable_bar_lying_into_the_printer_leaves_the_bed(quiet, printer):
    """A 150 x 18 mm bar lying into the printer, never picked: pushed off the nearer side edge, lengthwise."""
    r = _case("bar", 70, 120, 90, printer)
    assert r["end"] != "on_bed" and r["robot_bed_empty"], r
