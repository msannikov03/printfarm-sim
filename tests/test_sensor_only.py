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
