"""Door handling: hinge geometry inversion, magnetic catch, reed switch + belief, and physical open/close."""
import math

import mujoco
import numpy as np
import pytest

from farm import catalog as C
from farm.world import LATCH_GMAX, LATCH_R, REED_GAP, REED_HYST, World
import plan.door as D
from plan.ik import IK
from plan.motion import Executor


@pytest.fixture
def world():
    return World()


@pytest.mark.parametrize("i", [0, 9])
@pytest.mark.parametrize("deg", [0.0, 12.0, 50.0, 100.0])
def test_handle_angle_inverts_pin_world(i, deg):
    th, dr, dz = D.handle_angle(i, D.pin_world(i, math.radians(deg)))
    assert abs(math.degrees(th) - deg) < 1e-6
    assert abs(dr) < 1e-9 and abs(dz) < 1e-9


def test_magnetic_catch_profile(world):
    w = world
    t0 = w.latch_torque(0.0)
    assert t0 == pytest.approx(-w.latch_f0 * LATCH_R)            # strongest at contact, pulls closed
    gaps = [0.0, 0.001, 0.003, 0.01]
    mags = [abs(w.latch_torque(g / LATCH_R)) for g in gaps]
    assert all(a > b for a, b in zip(mags, mags[1:]))
    assert w.latch_torque((LATCH_GMAX + 0.001) / LATCH_R) == 0.0
    w.latch_f0 = 0.0
    assert w.latch_torque(0.0) == 0.0


def test_reed_switch_and_belief(world):
    w = world
    i = 3
    assert w.reed_closed(i) and w.door_belief(i) == 0.0
    w.d.qpos[w.door_q[i]] = 0.5                                   # opened by someone else: no FK estimate
    assert not w.reed_closed(i)
    assert w.door_belief(i) == pytest.approx(C.DOOR_OPEN_RAD)
    w.door_est[i] = 0.6                                           # robot measured it while holding the handle
    assert w.door_belief(i) == pytest.approx(0.6)
    mid = (REED_GAP + 0.5 * REED_HYST) / LATCH_R                  # inside the hysteresis band: keeps its state
    w.d.qpos[w.door_q[i]] = mid
    assert not w.reed_closed(i)
    w.d.qpos[w.door_q[i]] = 0.0
    assert w.reed_closed(i) and np.isnan(w.door_est[i])           # latched: the old estimate is dropped
    w.d.qpos[w.door_q[i]] = mid
    assert w.reed_closed(i)


def _setup(P):
    w = World()
    ik = IK(w)
    xc, s = C.printer_frame(P)
    w.reset(rail_x=float(np.clip(xc + 0.5 * s, C.RAIL_X_MIN, C.RAIL_X_MAX)))   # (row B end printer: in range)
    w.d.qpos[w.arm_q] = D.TUCK
    w.d.ctrl[w.act_arm] = D.TUCK
    mujoco.mj_forward(w.m, w.d)
    return w, ik, Executor(w)


@pytest.fixture
def quiet(monkeypatch):
    monkeypatch.setattr(D, "VERBOSE", False)


@pytest.mark.parametrize("P", [6, 13, 18])
def test_physical_open_close_no_weld(quiet, monkeypatch, P):
    """Default coupling: the door moves only through jaw contact; success is decided from sensors and agrees with
    the truth; the weld never engages. Row A and (two rows) row B printers."""
    if P >= C.N_PRINTERS:
        pytest.skip("one-row cell")
    monkeypatch.setattr(D, "COUPLING", "physical")
    w, ik, ex = _setup(P)
    eq = w.coupler[(P, "outer")]
    weld_seen = []
    w.hooks = [lambda w: weld_seen.append(int(w.d.eq_active[eq]))]
    ok, dt, note = D.operate_door(ex, ik, P, opening=True)
    assert ok, note
    assert math.degrees(w.door_angle(P)) > math.degrees(C.DOOR_OPEN_RAD) - 6
    assert abs(w.door_belief(P) - w.door_angle(P)) < math.radians(1.5)       # FK estimate vs truth
    arc = w.door_log[-1]
    assert arc["held"] and arc["mean_push_N"] > 0 and arc["mean_hinge_Nm"] > 0   # pads push the bar open
    assert arc["peak_other_contact_N"] < 1.0                                  # no pushing on the glass
    ok, dt, note = D.operate_door(ex, ik, P, opening=False)
    assert ok, note
    assert w.reed_closed(P) and math.degrees(w.door_angle(P)) < 1.0
    assert w.door_log[-1]["mean_push_N"] < 0                                  # and push it shut
    assert max(weld_seen) == 0


def test_weld_option_still_works(quiet, monkeypatch):
    monkeypatch.setattr(D, "COUPLING", "weld")
    P = 2
    w, ik, ex = _setup(P)
    ok, _, note = D.operate_door(ex, ik, P, opening=True)
    assert ok, note
    arc = w.door_log[-1]
    assert arc["coupling"] == "weld" and arc["mean_push_N"] > 0 and arc["mean_hinge_Nm"] > 0   # weld force sign
    ok, _, note = D.operate_door(ex, ik, P, opening=False)
    assert ok and w.reed_closed(P), note


def test_missed_handle_is_detected(quiet, monkeypatch):
    """If the handle is not where the robot believes, the gripper encoder reports an empty grasp."""
    monkeypatch.setattr(D, "COUPLING", "physical")
    P = 4
    w, ik, ex = _setup(P)
    w.m.body_pos[w.m.body(f"pin{P}").id, 1] -= 0.05           # bar 5 cm further out than the planner believes:
    mujoco.mj_forward(w.m, w.d)                               # outside the open jaws (37 mm per side)
    ok, _, note = D.operate_door(ex, ik, P, opening=True)
    assert not ok and "not in the jaws" in note, note
    assert math.degrees(w.door_angle(P)) < 10.0          # (the retreating jaws may nudge the misplaced bar)
    assert w.reed_closed(P) == (w.door_angle(P) * 0.33 < 0.003)   # the reed switch tells the truth either way


def test_ajar_door_with_unknown_angle_is_not_assumed_open(quiet, world):
    w = world
    ik = IK(w)
    w.d.qpos[w.door_q[1]] = 0.4                               # someone left it ajar; no FK estimate exists
    plans, note = D.plan_door(ik, w, 1, opening=True)
    assert plans is None and "unknown" in note
