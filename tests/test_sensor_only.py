"""Decisions use sensors, not simulator truth: regression tests for truth reads found in the sim-cheat audit."""
import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from farm import catalog as C  # noqa: E402

P = 4


@pytest.fixture(scope="module")
def world():
    from farm.world import World
    from plan.door import TUCK
    w = World()
    xc, _ = C.printer_frame(P)
    w.reset(rail_x=xc + 0.35)
    w.d.qpos[w.arm_q] = TUCK
    w.d.ctrl[w.act_arm] = TUCK
    mujoco.mj_forward(w.m, w.d)
    return w


def test_transfer_plans_the_jaws_from_the_gripper_encoder(world, monkeypatch):
    """The finger opening in transfer planning comes from the gripper encoder, not the true finger joint."""
    import plan.door as D
    from plan.ik import IK
    from plan.motion import Executor
    ik = IK(world)
    ex = Executor(world, realism="realistic", ik=ik, rng=np.random.default_rng(0))
    seen = []
    monkeypatch.setattr(D, "_plan_q", lambda ik_, q0, q1, ctx, iters=3000: seen.append(dict(ctx)))
    monkeypatch.setattr(ex, "finger_gap_mm", lambda: 24.0)        # what the encoder reads
    world.d.qpos[world.finger_q] = 0.030                           # the truth differs (30 mm per jaw)
    assert D.transfer(ex, ik, ex.q, ex.x) is False                  # fixed-carriage branch, planner stubbed out
    assert seen and seen[0]["grip"] == pytest.approx(0.012)        # 24 mm gap -> 12 mm per jaw


def test_drop_pose_does_not_need_the_true_arm_state(world):
    """The bin target depends only on the carriage: poisoning the true arm joints must not change the drop pose."""
    from plan.ik import IK
    from plan.unload import drop_pose
    ik = IK(world)
    xc, s = C.printer_frame(P)
    q_ref = drop_pose(ik, world, "o0", xc + s * 0.35)
    keep = world.d.qpos[world.arm_q].copy()
    try:
        world.d.qpos[world.arm_q] = np.nan
        q = drop_pose(ik, world, "o0", xc + s * 0.35)
    finally:
        world.d.qpos[world.arm_q] = keep
        mujoco.mj_forward(world.m, world.d)
    assert q_ref is not None and q is not None and np.allclose(q, q_ref)


def test_scan_quality_uses_the_believed_calibration(world):
    """Fill and near-lens fraction are measured inside the bed region the robot BELIEVES it sees (its own camera
    calibration), not the region the true camera pose would give."""
    from perceive import verify as V
    from perceive.noise import Calibration
    cam = V.BedCamera(world, P, noise="realistic", rng=np.random.default_rng(1))
    try:
        cam.calib = Calibration(np.eye(3), np.zeros(3), 1.0, np.zeros(2), 0.0)
        cam._roi_b = None
        assert np.array_equal(cam.believed_roi(), cam.bed_roi())    # no calibration error: the same region
        cam.calib = Calibration(np.eye(3), np.array([0.02, 0.0, 0.0]), 1.0, np.zeros(2), 0.0)
        cam._roi_b = None
        moved = cam.believed_roi() != cam.bed_roi()
        assert moved.sum() > 1000                                     # a 20 mm belief error shifts the region
        _, meta = cam.scan()
        assert meta["fill"] > 0.5
        cam.believed_roi = lambda: np.zeros_like(moved)              # robot believes it sees no bed at all
        _, meta = cam.scan()
        assert meta["fill"] == 0.0 and meta["quality"] == "failed"
    finally:
        cam.close()


# ---- small parts into a bin: level jaws, lowered release, the wrist camera looks into the bin afterwards ----
SLOT = "o1"                                    # an order bin in both layouts


def _robot(world, preset, seed):
    from plan.ik import IK
    from plan.motion import Executor
    from plan.door import TUCK
    world.d.qpos[world.arm_q] = TUCK
    world.d.ctrl[world.act_arm] = TUCK
    world.hooks = []
    mujoco.mj_forward(world.m, world.d)
    ik = IK(world)
    ex = Executor(world, realism=preset, ik=ik, rng=np.random.default_rng(seed))
    ik.sync(world.d)
    return ik, ex


def _put_arm(world, ex, q, gw):
    """Test setup: the arm at q (robot frame) and the jaws at gw per side, at rest (kinematic placement)."""
    zero = ex.hw.zero if ex.hw is not None else 0.0
    world.d.qpos[world.arm_q] = q + zero
    world.d.qvel[world.arm_v] = 0.0
    world.d.ctrl[world.act_arm] = q + zero
    if ex.hw is not None:
        ex.hw.cmd_q[:] = q
        ex.hw.motor[:] = ex.hw.out[:] = q + zero
        ex.hw.fifo.clear()
    world.d.qpos[world.finger_q], world.d.qpos[world.finger_rq] = gw, -gw
    world.d.ctrl[world.act_grip] = gw
    mujoco.mj_forward(world.m, world.d)


def _restore(world):
    from plan.door import TUCK
    for n in list(world.in_use):
        world.park(n)
    world.hooks = []
    world.d.qpos[world.arm_q] = TUCK
    world.d.ctrl[world.act_arm] = TUCK
    world.d.qvel[:] = 0.0
    mujoco.mj_forward(world.m, world.d)


@pytest.mark.parametrize("preset", ["realistic", "harsh"])
def test_bin_check_finds_a_part_on_the_rim(world, preset):
    """The farm's false success: a lid dropped into its tote came to rest on the tote wall (centre 57 mm out, at rim
    height). The wrist camera's look into the bin after the release must call that "on the rim", a part inside "in",
    and an unchanged bin "not seen"; the wall tops are measured in the scan, so the calibration errors cancel."""
    import plan.unload as U
    from plan.unload import part_in_bin
    ik, ex = _robot(world, preset, 7)
    try:
        rp = U.release_pose(ik, world, SLOT, ex.x, np.zeros(3), 0.04)
        assert rp is not None and abs(rp["R"][2, 1]) < 0.02          # the jaw axis is level over the bin
        _put_arm(world, ex, rp["q_look"] if rp["q_look"] is not None else rp["q"], 0.037)
        pre = U.bin_look(ex, ik, SLOT)
        sid = mujoco.mj_name2id(world.m, mujoco.mjtObj.mjOBJ_SITE, f"bin_{SLOT}")
        rim = world.d.site_xpos[sid].copy()
        land = rp["target"][:2]
        verdict = {}
        lid = world.free_part("lid")
        world.spawn(lid, rim + [0.0, 0.057, 0.0], 0.2, stuck=False)          # on the wall, as in the farm run
        assert not part_in_bin(world, lid, SLOT)                            # (a false success if reported placed)
        verdict["rim"] = U.bin_verdict(U.bin_look(ex, ik, SLOT, ref=pre), pre["h"], land)[0]
        world.park(lid)
        world.spawn(lid, rim + [0.012, -0.008, -0.066], 0.4, stuck=False)   # on the bin floor
        verdict["in"] = U.bin_verdict(U.bin_look(ex, ik, SLOT, ref=pre), pre["h"], land)[0]
        world.park(lid)
        mujoco.mj_forward(world.m, world.d)
        verdict["none"] = U.bin_verdict(U.bin_look(ex, ik, SLOT, ref=pre), pre["h"], land)[0]
    except Exception as e:
        if "OpenGL" in str(e) or "GLFW" in str(e) or "EGL" in str(e):
            pytest.skip(f"rendering unavailable: {e}")
        raise
    finally:
        _restore(world)
    assert verdict == {"rim": "on_rim", "in": "in", "none": "not_seen"}, verdict


def test_part_on_the_rim_is_pushed_in(world):
    """A lid bridging the gap between two totes (resting on both walls, as in the farm run) is seen on the rim and
    pushed into its own tote with the closed jaws; the look afterwards finds it inside."""
    import plan.unload as U
    from plan.unload import part_in_bin
    ik, ex = _robot(world, "realistic", 5)
    try:
        rp = U.release_pose(ik, world, SLOT, ex.x, np.zeros(3), 0.04)
        q_look = rp["q_look"] if rp["q_look"] is not None else rp["q"]
        _put_arm(world, ex, q_look, 0.037)
        pre = U.bin_look(ex, ik, SLOT)
        sid = mujoco.mj_name2id(world.m, mujoco.mjtObj.mjOBJ_SITE, f"bin_{SLOT}")
        rim = world.d.site_xpos[sid].copy()
        gap = 0.5 * (0.12 - C.BIN_SIZE[0])                             # half the gap to the -x neighbour (pitch 0.12)
        lid = world.free_part("lid")
        world.spawn(lid, rim + [-C.BIN_SIZE[0] / 2 - gap, 0.0, 0.0], 0.1, stuck=False)
        world.step(300)                                                # it settles on the two walls
        assert not part_in_bin(world, lid, SLOT)
        look = U.bin_look(ex, ik, SLOT, ref=pre)
        assert U.bin_verdict(look, pre["h"], rp["target"][:2])[0] == "on_rim"
        assert U.push_into_bin(ex, ik, look, SLOT)
        ex.wait(0.5)
        inside = part_in_bin(world, lid, SLOT)
        U.transfer(ex, ik, q_look, ex.x, {"grip": 0.0})
        ex.jaws(0.037, duration=0.3)
        ex.settle(tol=0.004, timeout=0.5)
        verdict = U.bin_verdict(U.bin_look(ex, ik, SLOT, ref=pre), pre["h"], rp["target"][:2])[0]
    except Exception as e:
        if "OpenGL" in str(e) or "GLFW" in str(e) or "EGL" in str(e):
            pytest.skip(f"rendering unavailable: {e}")
        raise
    finally:
        _restore(world)
    assert inside and verdict == "in"


@pytest.mark.parametrize("stuck", [False, True])
def test_release_into_bin_end_to_end(world, stuck):
    """The whole release with physics: the held part is lowered below the rim, released, the jaws are checked and the
    wrist camera looks into the bin. A part that falls is reported placed and is in its bin; a part that really
    sticks to a pad (test setup: glued to the finger every control tick) is reported stuck on the finger. The wrist
    camera carries a 4 mm hand-eye error along the jaw axis: with the nominal jaw volume the empty jaws would read
    "stuck" (test_planning.py); the empty-jaw reference must keep both answers right."""
    import plan.unload as U
    from plan.unload import part_in_bin
    ik, ex = _robot(world, "realistic", 11)
    ex.wrist.handeye = np.array([0.0005, 0.004, -0.0005])
    try:
        rp = U.release_pose(ik, world, SLOT, ex.x, np.zeros(3), 0.04)
        assert rp is not None
        rp = dict(rp, q_look=None)                       # start over the bin (no transfer from the look pose)
        width = 0.025
        _put_arm(world, ex, rp["q"], width / 2)
        b = world.free_part("cube")
        Rp = world.d.site_xmat[world.grasp_site].reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, Rp.flatten())
        world.spawn(b, world.d.site_xpos[world.grasp_site], 0.0, stuck=False)
        a, v = world.part_q[b], world.part_v[b]
        world.d.qpos[a:a + 3] = world.d.site_xpos[world.grasp_site] - Rp @ world.m.body_ipos[world.part_body[b]]
        world.d.qpos[a + 3:a + 7] = quat
        mujoco.mj_forward(world.m, world.d)
        if stuck:                                        # the cube follows the left finger wherever it goes
            fb = world.m.body("link_left_finger").id
            Rf, pf = world.d.xmat[fb].reshape(3, 3), world.d.xpos[fb].copy()
            off, Rrel = Rf.T @ (world.d.qpos[a:a + 3] - pf), Rf.T @ Rp

            def glue(w):
                R = w.d.xmat[fb].reshape(3, 3)
                w.d.qpos[a:a + 3] = w.d.xpos[fb] + R @ off
                mujoco.mju_mat2Quat(w.d.qpos[a + 3:a + 7], (R @ Rrel).flatten())
                w.d.qvel[v:v + 6] = 0.0

            world.hooks = [glue]
        ex.grip(True, duration=0.3, width=width)        # squeeze it
        ex.payload = True
        ik.sync(world.d)
        ik.attach(b, world.d)
        g = {"width": width}
        rel = Rp.T @ (world.d.xipos[world.part_body[b]] - world.d.site_xpos[world.grasp_site])
        ok, outcome, _ = U._release_into_bin(ex, ik, b, SLOT, g, rp, rel)
        truly = part_in_bin(world, b, SLOT)
    except Exception as e:
        if "OpenGL" in str(e) or "GLFW" in str(e) or "EGL" in str(e):
            pytest.skip(f"rendering unavailable: {e}")
        raise
    finally:
        ik.detach()
        ex.payload = False
        _restore(world)
    if stuck:
        assert (ok, outcome) == (False, "stuck_on_finger")
    else:
        assert (ok, outcome, truly) == (True, "placed", True)
