import math

import numpy as np
import pytest

from real.calib import BedCalib, fit_touch, from_base_pose, kabsch, main as calib_main, nominal
from real.reach import hover_target, solve_object
from real.so101 import APPROACH_LOCAL, FIXED_TIP, JOINTS, SO101, hover_tcp


@pytest.fixture(scope="module")
def kin():
    return SO101()


def test_model_joint_order_matches_lerobot_motor_names(kin):
    names = [kin.model.joint(i).name for i in range(kin.model.njnt)]
    assert names == list(JOINTS) == ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll",
                                     "gripper"]


def test_fk_fixed_tip_matches_model_geom(kin):
    q = np.array([0.3, -0.5, 0.7, 1.0, 0.4, 0.2])
    p, _ = kin.fk(q, FIXED_TIP)
    d = kin.data
    g = d.geom_xpos[kin.model.geom("fixed_jaw_sph_tip1").id]
    assert np.linalg.norm(p - g) * 1000 < 1.5          # tip point = the jaw-tip sphere (+ its radius)


def test_ik_roundtrip_random_poses(kin):
    rng = np.random.default_rng(1)
    for _ in range(10):
        q = rng.uniform(kin.lo * 0.6, kin.hi * 0.6)
        p, R = kin.fk(q, FIXED_TIP)
        r = kin.solve(p, approach=R @ APPROACH_LOCAL, tcp=FIXED_TIP, q0=q + rng.normal(0, 0.2, 6))
        assert r.ok and r.pos_err_mm < 1.0, r


@pytest.mark.parametrize("xy", [(110, 30), (30, 30), (190, 40), (110, 110), (40, 150), (180, 200)])
def test_hover_reachable_over_bed_points(kin, xy):
    calib = nominal((220, 220))
    obj = dict(center_mm=list(xy), height_mm=15.0, footprint_mm=[30, 15], yaw_deg=30.0)
    r = solve_object(kin, calib, obj, hover_mm=40)
    assert r.ok and r.pitch_deg is not None, r
    p, _ = kin.fk(r.q, hover_tcp(15))
    target, jaw = hover_target(calib, obj, 40)
    assert np.linalg.norm(p - target) * 1000 < 1.0
    # tool tips 40 mm above the 15 mm tall part
    assert calib.robot_to_bed(p)[2] == pytest.approx(55.0, abs=1.0)
    if r.pitch_deg == 90:
        a, j = kin.tool_axes(r.q)
        assert a[2] < -0.99                            # pointing straight down
        assert abs(np.dot(j, jaw)) > 0.99              # jaws close across the short side


def test_kabsch_recovers_transform_with_noise():
    rng = np.random.default_rng(0)
    true = from_base_pose((105, -70, -12), 87.0, (220, 220))
    bed = np.array([[20, 20], [200, 20], [200, 200], [20, 200], [110, 110]], float)
    robot = true.bed_to_robot(bed) + rng.normal(0, 0.0005, (len(bed), 3))     # 0.5 mm noise
    c = fit_touch(bed, robot)
    assert c.rms_mm < 1.5
    test = np.array([[0, 0, 0], [220, 220, 50], [50, 170, 20]], float)
    err = np.linalg.norm(c.bed_to_robot(test) - true.bed_to_robot(test), axis=1) * 1000
    assert err.max() < 2.0


def test_kabsch_exact_and_errors():
    R0 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    P = np.random.default_rng(3).normal(size=(6, 3))
    R, t = kabsch(P, P @ R0.T + [1, 2, 3])
    assert np.allclose(R, R0) and np.allclose(t, [1, 2, 3])
    with pytest.raises(ValueError):
        kabsch(P[:2], P[:2])
    with pytest.raises(ValueError):
        kabsch(np.c_[np.arange(4), np.arange(4), np.zeros(4)], np.zeros((4, 3)))


def test_calib_json_roundtrip_and_inverse(tmp_path):
    c = from_base_pose((100, -50, 5), 80.0, (300, 300))
    f = c.save(tmp_path / "c.json")
    c2 = BedCalib.load(f)
    p = np.array([[10, 20, 30], [250, 280, 0]], float)
    assert np.allclose(c2.robot_to_bed(c2.bed_to_robot(p)), p, atol=1e-3)
    assert np.allclose(c.bed_to_robot([100, -50, 5]), 0, atol=1e-9)        # base origin
    # robot +x points along bed direction yaw=80 deg
    d = c.bed_to_robot([100 + 10 * math.cos(math.radians(80)), -50 + 10 * math.sin(math.radians(80)), 5])
    assert d[0] == pytest.approx(0.01) and abs(d[1]) < 1e-9


def test_touch_off_dry_run_cli(tmp_path):
    """Simulated hand-guiding against a nominal placement: the fit must recover it (0.3 deg joint noise)."""
    out = tmp_path / "calib.json"
    c = calib_main(["touch", "--dry-run", "-o", str(out)])
    assert out.exists() and c.method == "touch" and c.rms_mm < 3.0
    truth = nominal((220, 220))
    pts = np.array([[0, 0, 0], [220, 0, 0], [110, 110, 40]], float)
    assert (np.linalg.norm(c.bed_to_robot(pts) - truth.bed_to_robot(pts), axis=1) * 1000).max() < 4.0


def test_touch_off_jog_dry_run_cli(tmp_path):
    out = tmp_path / "calib.json"
    c = calib_main(["touch", "--dry-run", "--jog", "--guess", str(nominal((220, 220)).save(tmp_path / "g.json")),
                    "-o", str(out)])
    assert c.rms_mm < 1.0
    assert np.allclose(c.bed_to_robot([110, 110, 0]), nominal((220, 220)).bed_to_robot([110, 110, 0]), atol=0.002)


def test_aruco_stub_recovers_bed_pose_on_synthetic_image():
    cv2 = pytest.importorskip("cv2")
    from real.calib import aruco_T_cam_bed
    K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
    # camera 40 cm above the bed centre, looking down (bed z towards the camera), slightly rotated
    R = cv2.Rodrigues(np.array([math.pi, 0.0, 0.15]))[0]
    t = R @ -np.array([0.11, 0.11, 0.40])
    img = np.full((480, 640), 255, np.uint8)
    markers = {3: (60.0, 60.0), 7: (160.0, 150.0)}
    L = 40.0
    for mid, (x, y) in markers.items():
        tag = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), mid, 200)
        tag = cv2.copyMakeBorder(tag, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)   # quiet zone
        h = L / 2 * 280 / 200                                                        # bed mm incl. border
        obj = np.array([[x - h, y + h, 0], [x + h, y + h, 0], [x + h, y - h, 0], [x - h, y - h, 0]]) / 1000
        px = (K @ (obj @ R.T + t).T).T
        px = (px[:, :2] / px[:, 2:]).astype(np.float32)
        H = cv2.getPerspectiveTransform(np.float32([[0, 0], [279, 0], [279, 279], [0, 279]]), px)
        warped = cv2.warpPerspective(tag, H, (640, 480), borderValue=255)
        mask = cv2.warpPerspective(np.full_like(tag, 255), H, (640, 480)) > 0
        img[mask] = warped[mask]
    T, used, rms = aruco_T_cam_bed(img, K, np.zeros(5), L, markers)
    assert sorted(used) == [3, 7] and rms < 1.5
    bed = np.array([[0, 0, 0], [220, 0, 0], [220, 220, 0], [0, 220, 0], [110, 110, 30]]) / 1000
    err = np.linalg.norm(bed @ T[:3, :3].T + T[:3, 3] - (bed @ R.T + t), axis=1) * 1000
    assert err.max() < 6.0          # mm; two near-fronto-parallel markers leave ~2 deg of tilt ambiguity
