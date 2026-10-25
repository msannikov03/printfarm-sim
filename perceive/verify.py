"""Perception: verify the bed against the slicer prior with the printer's chamber camera (depth).

The slicer says what should be on the bed, where, and how tall. The camera only refines and verifies:
  1. render depth from the chamber camera and back-project to bed coordinates;
  2. for each slicer object: take points near its prior footprint whose height is close to the object's known top
     (its top face), estimate the centre (centroid) and yaw (principal axis, folded by the shape's symmetry);
  3. flag objects with too few top-face points as MISSING; any remaining raised points not explained by the
     slicer objects are an UNKNOWN blob (failed print / debris);
  4. after unloading, the same check on an empty prior is the bed-empty test.
Ground truth (sim poses) is used only to report the estimation error.

Sensor realism (perceive/noise.py, docs/perception.md). BedCamera(world, printer, noise=...) with noise "ideal" (the
default: the perfect render, exactly the original behaviour), "realistic", "harsh" or a CameraNoise. With a noisy
camera, points_bed_mm() runs the correction pipeline and returns a BedPoints array (a plain (N, 3) ndarray that also
carries `.meta` about the scan), which estimate() / bed_empty() / raised_points() read:
  - multi-frame temporal median (frames), flying-pixel / speckle filter against the local median;
  - RANSAC bed-plane fit: removes the height offset and tilt that extrinsic errors put on the bed (rotation about the
    believed camera centre, so the lateral shift of that rotation goes too);
  - bed-plate outline fit (4 straight edges, robust line fits): x, y and yaw of the camera relative to the plate;
  - robust top-face estimation (adaptive height band from the noise model, outlier gate, trimmed rectangle fit);
  - Bayesian fusion of the slicer prior N(mu_slicer, Sigma_prior) with the camera N(mu_cam, Sigma_cam), with a
    chi-square gate (a part nudged by a neighbouring pick rejects the prior); Sigma_cam comes from the noise model and
    the number of supporting points;
  - failure handling: a failed scan gives the prior with inflated covariance and quality "failed" (never raises);
    a region the camera did not see (holes, occluder) gives the prior with quality "degraded" instead of "missing".
estimate() adds 'sigma_mm' (RMS radial 1-sigma = sqrt(trace(cov))), 'cov_mm2' (2x2), 'quality' ("ok" | "degraded" |
"failed"), 'yaw_sigma_deg' and 'observed' to every object.
"""
import math
import time
import warnings

import mujoco
import numpy as np
from scipy import ndimage

from farm import catalog as C
from perceive.noise import Z_REF, Calibration, corrupt, get_noise, sample_scan

W, H = 640, 480
RAISED_MM = 1.5          # a point counts as "on the bed" above this height
SEARCH_MM = 18.0         # search radius around the prior footprint
TOP_TOL_MM = 1.5         # top-face band below the known height
MIN_TOP_PTS = 25
FOOT_MARGIN_MM = 8.0     # prior pose noise budget (about 3 sigma)
RELOCATE_MM = 30.0       # second pass for parts nudged by a neighbouring pick
MAX_YAW_CORR_DEG = 20.0  # the camera may correct the slicer's yaw by at most this much

# ---- uncertainty model (the numbers marked "measured" come from scripts/eval_perception.py) ----
PRIOR_SIGMA_MM = 2.0          # slicer prior, 1 sigma per axis (parts shift a little while printing / cooling)
PRIOR_YAW_SIGMA_DEG = 3.0
SIGMA_FLOOR_MM = 0.25         # robust estimator floor per axis with perfect depth (measured: ideal + pipeline)
SIGMA_FLOOR_LEGACY_MM = 0.45  # original estimator floor per axis with perfect depth (measured: ideal)
SIGMA_CAL_FIT_MM = 0.2        # calibration residual per axis after the plane + outline corrections (measured)
YAW_FLOOR_DEG = 0.15          # camera yaw floor, elongated box top faces (measured: ideal + pipeline)
YAW_FLOOR_L_DEG = 0.3         # ... L bracket (base vs upright)
YAW_K, YAW_K_L = 0.5, 0.25    # camera yaw sigma = hypot(floor, deg(k * sigma_xy / lever)) (fit to measured errors)
FAIL_INFLATE = 2.0            # failed / unseen: prior covariance x FAIL_INFLATE^2
GATE_CHI2 = 13.8              # chi-square (2 dof, 99.9 %): camera far from the prior -> the part moved, drop the prior
YAW_GATE_SIG = 3.3
# ---- robust raised-point / bed-empty test ----
RAISED_SIGMAS = 3.5           # raised threshold = max(RAISED_MM, 3.5 sigma of the point's fused depth noise)
MIN_RAISED_PTS = 8            # isolated raised specks smaller than this (2 mm grid components) are noise
LEFTOVER_MIN_PTS = 12         # a leftover = a raised component with at least this many points
LEFTOVER_MIN_H_MM = 3.0       # ... whose top (95th percentile height) is at least this high (target: >= 5 mm lumps)
OBSERVED_FRAC = 0.3           # a footprint with < 30 % of the expected points was not seen (holes / occluder)
FAR_RANGE_M = 0.55            # far bed corner seen from the chamber camera
NEAR_M = 0.20                 # depth closer than this is not the bed or a part (bed >= 0.33 m): an occluder
NEAR_FRAC_MAX = 0.01
BLUR_BINS_MAX = 10            # outline bins with a blurred step -> smeared lens (noise alone: 0-6)
UNSEEN_MM2 = 600.0            # bed_empty: an unobserved bed patch larger than this cannot be verified
THIN_PART_MM = 5.0            # planned parts lower than this are also checked by footprint (thin_leftovers): a
                              # leftover_blobs component must reach LEFTOVER_MIN_H_MM, so a 2 mm part passes it
THIN_RISE_FRAC = 0.5          # ... still on the bed: the footprint stands above the bed ring around it by this share
THIN_RISE_MIN_MM = 0.4        # ... of the planned height, and by at least this much (empty bed, measured: rise p99 /
                              # max 0.24 / 0.27 mm realistic, 0.53 / 0.56 mm harsh: harsh may call a person in vain)
THIN_RING_MM = (3.0, 10.0)    # ... bed reference: this far outside the footprint (cancels local plane / calibration bias)
THIN_PCT = 75                 # ... statistic on both sides (a flat part may fill only part of its footprint rectangle)
THIN_MIN_PTS = 20


class BedPoints(np.ndarray):
    """(N, 3) bed-frame points in mm plus `meta` (dict) about the scan that produced them."""

    def __new__(cls, a, meta=None):
        obj = np.asarray(a, dtype=np.float64).view(cls)
        obj.meta = meta
        return obj

    def __array_finalize__(self, obj):
        self.meta = getattr(obj, "meta", None)


def _meta(points):
    m = getattr(points, "meta", None)
    return m if isinstance(m, dict) and not m.get("ideal") else None


def _rot_to_z(n):
    """Rotation matrix taking unit vector n onto +z (minimal rotation)."""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, z)
    s, c = np.linalg.norm(v), float(np.dot(n, z))
    if s < 1e-12:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def fit_bed_plane(p, sigma_mm, rng, iters=80):
    """RANSAC plane through bed-level points near the nominal bed (bed mm). -> dict(n, c, inlier_frac, rms) | None."""
    m = (p[:, 0] > 8) & (p[:, 0] < C.BED_MM[0] - 8) & (p[:, 1] > 8) & (p[:, 1] < C.BED_MM[1] - 8) & \
        (np.abs(p[:, 2]) < 15)
    q = p[m]
    if len(q) < 500:
        return None
    if len(q) > 8000:
        q = q[rng.choice(len(q), 8000, replace=False)]
    thr = max(1.0, 3.0 * sigma_mm)
    best, best_cnt = None, -1
    for _ in range(iters):
        a, b, c = q[rng.choice(len(q), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-6:
            continue
        n = n / nn * (1 if n[2] >= 0 else -1)
        if n[2] < 0.97:                                   # the bed is within ~14 deg of level
            continue
        cnt = int((np.abs((q - a) @ n) < thr).sum())
        if cnt > best_cnt:
            best, best_cnt = (n, a), cnt
    if best is None:
        return None
    n, a = best
    inl = q[np.abs((q - a) @ n) < thr]
    cen = inl.mean(0)
    n = np.linalg.svd(inl - cen, full_matrices=False)[2][2]
    n = n if n[2] >= 0 else -n
    r = (inl - cen) @ n
    return {"n": n, "c": cen, "inlier_frac": len(inl) / len(q), "rms": float(r.std()), "n_inl": int(len(inl))}


def _fit_line(t, e, thr, rng, iters=60):
    """Robust line e = a + b (t - 128) -> (a, b, inlier mask) or None."""
    if len(t) < 6:
        return None
    best, best_cnt = None, -1
    for _ in range(iters):
        i, j = rng.choice(len(t), 2, replace=False)
        if abs(t[i] - t[j]) < 20:
            continue
        b = (e[j] - e[i]) / (t[j] - t[i])
        a = e[i] + b * (128.0 - t[i])
        cnt = int((np.abs(a + b * (t - 128.0) - e) < thr).sum())
        if cnt > best_cnt:
            best, best_cnt = (a, b), cnt
    if best is None:
        return None
    a, b = best
    inl = np.abs(a + b * (t - 128.0) - e) < thr
    if inl.sum() < 8 or np.ptp(t[inl]) < 60:
        return None
    b, a = np.polyfit(t[inl] - 128.0, e[inl], 1)
    return float(a), float(b), inl


SIZE_TOL_MM = 3.0       # measured plate width / depth must match the nominal within this (else an edge is corrupt)
RAMP_PTS = 2            # a bin with this many points on a ramp just outside its edge is blurred (smeared lens)
# The outermost bed-level sample of a bin lies up to one pixel footprint inside the true edge, so each edge reads a
# little inward (more at the far edges, where a pixel covers more bed). Measured once on the perfect render of the
# empty bed (scripts/eval_perception.py --edge-bias), like an install-time calibration of this camera geometry.
EDGE_BIAS_MM = {"left": 0.08, "right": -0.38, "front": 0.10, "back": -0.31}


def fit_bed_outline(p, sigma_mm, rng, bin_mm=4.0, budget_mm=None):
    """Bed-plate outline from bed-level points (after levelling): the four plate edges are depth steps, so the outermost
    bed-level point per bin along each edge traces it. Bins where the step is blurred (points on a ramp just outside
    the edge: smeared lens) are skipped; opposite edges must agree with the plate size; a shift beyond 4x the
    calibration budget is not trusted. -> dict(dx, dy, dyaw_deg, nx, ny, edges) | None.
    dx, dy: measured plate centre minus nominal (mm, None if that axis is not measured); dyaw: plate rotation (deg)."""
    band = max(1.0, 3.0 * sigma_mm)
    q = p[np.abs(p[:, 2]) < band]
    ramp = p[(p[:, 2] < -band) & (p[:, 2] > -30.0)]
    Wb, Hb = C.BED_MM
    spec = {"left": (0, -1, 0.0), "right": (0, 1, Wb), "front": (1, -1, 0.0), "back": (1, 1, Hb)}
    edges, blurred = {}, 0
    for name, (ax, side, nom) in spec.items():
        o = 1 - ax
        lim = Wb if o == 0 else Hb
        s = q[(np.abs(q[:, ax] - nom) < 20) & (q[:, o] > 10) & (q[:, o] < lim - 10)]
        if len(s) < 50:
            continue
        idx = np.floor(s[:, o] / bin_mm).astype(int)
        order = np.argsort(idx, kind="stable")
        idx, s = idx[order], s[order]
        starts = np.flatnonzero(np.r_[True, idx[1:] != idx[:-1]])
        counts = np.diff(np.r_[starts, len(idx)])
        ext = (np.maximum if side > 0 else np.minimum).reduceat(s[:, ax], starts)
        bins = idx[starts]
        # blur test: ramp points (below the bed, above the floor) 2-15 mm outside this bin's edge
        r = ramp[(np.abs(ramp[:, ax] - nom) < 35) & (ramp[:, o] > 10) & (ramp[:, o] < lim - 10)]
        rb = np.floor(r[:, o] / bin_mm).astype(int)
        pos = np.searchsorted(bins, rb)
        ok = (pos < len(bins)) & (bins[np.minimum(pos, len(bins) - 1)] == rb)
        out_d = side * (r[ok, ax] - ext[pos[ok]])
        nramp = np.bincount(pos[ok][(out_d > 2.0) & (out_d < 15.0)], minlength=len(bins))
        keep = (counts >= 3) & (nramp < RAMP_PTS)
        t = (bins + 0.5) * bin_mm
        fit = _fit_line(t[keep], ext[keep], max(1.2, 2.5 * sigma_mm), rng)
        if fit is not None:
            a, b, inl = fit
            a -= EDGE_BIAS_MM[name]
            edges[name] = {"a": a, "b": b, "n": int(inl.sum()), "span": float(np.ptp(t[keep][inl]))}
        blurred += int((nramp >= RAMP_PTS).sum())
    # opposite edges must agree with the plate size; otherwise keep the one with more support
    for lo, hi, size in (("left", "right", Wb), ("front", "back", Hb)):
        if lo in edges and hi in edges and abs(edges[hi]["a"] - edges[lo]["a"] - size) > SIZE_TOL_MM:
            worse = lo if edges[lo]["n"] * edges[lo]["span"] < edges[hi]["n"] * edges[hi]["span"] else hi
            edges.pop(worse)
    if not edges:
        return None
    yaws, wts = [], []
    for name, e in edges.items():
        yaws.append(-math.atan(e["b"]) if name in ("left", "right") else math.atan(e["b"]))
        wts.append(e["n"] * e["span"] ** 2)
    dyaw = float(np.average(yaws, weights=wts))

    def centre(lo, hi, size):
        if lo in edges and hi in edges:
            return 0.5 * (edges[lo]["a"] + edges[hi]["a"]) - size / 2, 2
        if lo in edges:
            return edges[lo]["a"], 1
        if hi in edges:
            return edges[hi]["a"] - size, 1
        return None, 0

    dx, nx = centre("left", "right", Wb)
    dy, ny = centre("front", "back", Hb)
    if budget_mm:                       # a shift far beyond what the calibration can be off by: not trusted
        if dx is not None and abs(dx) > 4 * budget_mm:
            dx, nx = None, 0
        if dy is not None and abs(dy) > 4 * budget_mm:
            dy, ny = None, 0
    return {"dx": dx, "dy": dy, "dyaw_deg": math.degrees(dyaw), "nx": nx, "ny": ny, "blurred_bins": blurred,
            "edges": {k: {kk: round(float(vv), 4) for kk, vv in v.items()} for k, v in edges.items()}}


class BedCamera:
    """Chamber depth camera of printer `printer`.
    noise: "ideal" (default; perfect render, original behaviour) | "realistic" | "harsh" | CameraNoise.
    rng: numpy Generator for every noise draw (seeded; default: seeded by the printer index).
    frames: frames averaged per scan (default: the preset's). corrections: None = on for a noisy camera;
    False = naive pipeline (one raw frame, no plane / outline correction, original estimator); True = the full
    pipeline even on the ideal camera. force: fault injection for one scan mode ('dead' | 'blackout' | 'occlusion' |
    'smear' | 'no_frame'), applied to every scan while set."""

    def __init__(self, world, printer, noise="ideal", rng=None, frames=None, corrections=None, max_retries=2):
        self.w, self.i = world, printer
        self.cam = mujoco.mj_name2id(world.m, mujoco.mjtObj.mjOBJ_CAMERA, f"chamber{printer}")
        self.r = mujoco.Renderer(world.m, height=H, width=W)
        self.r.enable_depth_rendering()
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 0
        opt.geomgroup[[0, 1, 2]] = 1
        self.opt = opt
        self.noise = get_noise(noise)
        self.rng = rng if rng is not None else np.random.default_rng(1000 + int(printer))
        self.frames = int(frames) if frames else int(self.noise.frames)
        self.legacy = self.noise.is_ideal and corrections is None
        self.corrections = (not self.noise.is_ideal) if corrections is None else bool(corrections)
        self.max_retries = max_retries
        self.force = None
        self.rs = None                 # segmentation renderer (IR-dark parts), created on demand
        self.last = None               # meta of the last scan
        self._roi = None
        self.resample(self.rng)

    # ---------------------------------------------------------------- calibration / materials
    def resample(self, rng=None):
        """New calibration error and new set of IR-dark parts (Monte Carlo over camera installations)."""
        rng = self.rng if rng is None else rng
        self.calib = Calibration.sample(self.noise, rng)
        self._roi_b = None             # the believed bed region depends on the calibration
        self._dark_seed = int(rng.integers(2 ** 31)) if self.noise.dark_frac else 0

    def _f_px(self):
        return 0.5 * H / math.tan(math.radians(self.w.m.cam_fovy[self.cam]) / 2)

    def _to_bed(self, pw):
        xc, s = C.printer_frame(self.i)
        xl = s * (pw[..., 0] - xc)
        yl = s * pw[..., 1] - C.AISLE_HALF
        bx = (xl + C.BED_MM[0] / 2000) * 1000
        by = (yl - C.BED_FRONT_GAP) * 1000
        bz = (pw[..., 2] - C.BENCH_H - C.BED_ZL) * 1000
        return np.stack([bx, by, bz], -1)

    def _render(self):
        self.r.update_scene(self.w.d, camera=self.cam, scene_option=self.opt)
        return self.r.render()

    # ---------------------------------------------------------------- legacy (ideal) path: unchanged
    def _points_legacy(self):
        m, d = self.w.m, self.w.d
        self.r.update_scene(d, camera=self.cam, scene_option=self.opt)
        depth = self.r.render()
        f = 0.5 * H / math.tan(math.radians(m.cam_fovy[self.cam]) / 2)
        v, u = np.mgrid[0:H, 0:W]
        z = depth.astype(np.float64)
        pc = np.stack([(u - W / 2 + 0.5) * z / f, -(v - H / 2 + 0.5) * z / f, -z], -1).reshape(-1, 3)
        R = d.cam_xmat[self.cam].reshape(3, 3)
        pw = pc @ R.T + d.cam_xpos[self.cam]
        xc, s = C.printer_frame(self.i)
        xl = s * (pw[:, 0] - xc)
        yl = s * pw[:, 1] - C.AISLE_HALF
        bx = (xl + C.BED_MM[0] / 2000) * 1000
        by = (yl - C.BED_FRONT_GAP) * 1000
        bz = (pw[:, 2] - C.BENCH_H - C.BED_ZL) * 1000
        keep = (bx > -5) & (bx < C.BED_MM[0] + 5) & (by > -5) & (by < C.BED_MM[1] + 5) & (bz < 80)
        return np.stack([bx, by, bz], -1)[keep]

    def points_bed_mm(self):
        """Depth -> 3D points in bed coordinates (mm): x right, y into the printer, z up from the bed.
        Ideal camera: the original perfect-render path (plain ndarray). Otherwise a BedPoints with `.meta`."""
        if self.legacy:
            self.last = {"ideal": True, "quality": "ok"}
            return self._points_legacy()
        pts, meta = self.scan()
        return BedPoints(pts, meta)

    # ---------------------------------------------------------------- noisy path
    def bed_roi(self):
        """Pixels whose ray (true calibration) hits the bed plate: where the bed should be seen."""
        if self._roi is None:
            d = self.w.d
            f = self._f_px()
            v, u = np.mgrid[0:H, 0:W]
            ray = np.stack([(u - W / 2 + 0.5) / f, -(v - H / 2 + 0.5) / f, -np.ones((H, W))], -1)
            R = d.cam_xmat[self.cam].reshape(3, 3)
            rw = ray @ R.T
            c = d.cam_xpos[self.cam]
            zb = C.BENCH_H + C.BED_ZL
            with np.errstate(divide="ignore", invalid="ignore"):
                lam = (zb - c[2]) / rw[..., 2]
            pb = self._to_bed(c + lam[..., None] * rw)
            self._bed_depth = np.where(lam > 0, lam, np.nan)       # z-depth of the bed plane along each pixel
            self._roi = (lam > 0) & (pb[..., 0] > 0) & (pb[..., 0] < C.BED_MM[0]) & (pb[..., 1] > 0) & \
                        (pb[..., 1] < C.BED_MM[1])
        return self._roi

    def believed_roi(self):
        """Pixels whose ray, back-projected with the BELIEVED calibration (the robot's own intrinsics and extrinsics),
        hits the nominal bed plate. The scan's quality metrics (fill, something at the lens) use this region: the
        robot does not know the true camera pose. bed_roi() (true pose) only places the sheen in the noise model."""
        if self._roi_b is None:
            d, cal = self.w.d, self.calib
            fb = self._f_px() * cal.f_scale
            v, u = np.mgrid[0:H, 0:W]
            ray = np.stack([(u - W / 2 + 0.5 - cal.c_off[0]) / fb, -(v - H / 2 + 0.5 - cal.c_off[1]) / fb,
                            -np.ones((H, W))], -1)
            rw = ray @ (d.cam_xmat[self.cam].reshape(3, 3) @ cal.R_err).T
            c = d.cam_xpos[self.cam] + cal.t_err
            with np.errstate(divide="ignore", invalid="ignore"):
                lam = (C.BENCH_H + C.BED_ZL - c[2]) / rw[..., 2]
            pb = self._to_bed(c + lam[..., None] * rw)
            self._roi_b = (lam > 0) & (pb[..., 0] > 0) & (pb[..., 0] < C.BED_MM[0]) & (pb[..., 1] > 0) & \
                          (pb[..., 1] < C.BED_MM[1])
        return self._roi_b

    def _dark_mask(self):
        """IR-dark (black filament) part pixels, from a segmentation render."""
        if self.rs is None:
            self.rs = mujoco.Renderer(self.w.m, height=H, width=W)
            self.rs.enable_segmentation_rendering()
        self.rs.update_scene(self.w.d, camera=self.cam, scene_option=self.opt)
        seg = self.rs.render()
        gid = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
        dark_bodies = []
        for n in self.w.in_use:
            b = self.w.part_body[n]
            u = np.random.default_rng((self._dark_seed, int(b))).random()
            if u < self.noise.dark_frac:
                dark_bodies.append(b)
        if not dark_bodies:
            return np.zeros((H, W), bool)
        body = np.where(gid >= 0, self.w.m.body_rootid[self.w.m.geom_bodyid[np.clip(gid, 0, None)]], -1)
        return np.isin(body, dark_bodies)

    def sigma_z_eff_mm(self, z_m):
        """1-sigma depth noise after the temporal fusion of this camera's frames (mm)."""
        k = 1.0 / math.sqrt(self.frames) if self.corrections else 1.0
        return self.noise.sigma_z_mm(z_m) * k

    def scan(self):
        """One noisy scan -> (points (N, 3) bed mm, meta). Never raises on a camera failure: quality 'failed'."""
        t0 = time.perf_counter()
        nz, rng = self.noise, self.rng
        frames = self.frames if self.corrections else 1
        f = self._f_px()
        sc = sample_scan(nz, rng, (H, W), self.max_retries, self.force)
        sig_bed = float(self.sigma_z_eff_mm(FAR_RANGE_M))
        meta = {"ideal": False, "noise": nz.name, "mode": sc["mode"], "retries": sc["retries"], "frames": frames,
                "corrections": self.corrections, "axial_mm": nz.axial_mm, "lateral_px": nz.lateral_px,
                "corr_frac": nz.axial_corr_frac, "corr_px": nz.axial_corr_px, "f_px": f,
                "scale_sigma": math.hypot(nz.f_rel, nz.depth_scale),
                "sigma_z_far_mm": sig_bed, "calib": self.calib.summary(), "fill": 0.0, "plane": None,
                "outline": None, "quality": "failed", "cam_bed_mm": None, "sigma_cal_mm": None, "near_frac": 0.0}
        empty = np.zeros((0, 3))
        if sc["mode"] in ("dead", "blackout"):
            meta["t_scan_s"] = time.perf_counter() - t0
            self.last = meta
            return empty, meta
        depth = self._render()
        dark = self._dark_mask() if nz.dark_frac else None
        roi = self.bed_roi()
        with np.errstate(invalid="ignore"):
            bare = roi & (np.abs(depth - self._bed_depth) < 0.001)   # the PEI sheet itself (where a sheen can be)
        stack = np.stack([corrupt(depth, nz, rng, sc, self.calib, dark, bare) for _ in range(frames)])
        if frames == 1:
            z = stack[0]
        else:
            ok = np.isfinite(stack)
            cnt = ok.sum(0)
            with warnings.catch_warnings():                  # all-NaN pixels (holes) stay NaN
                warnings.simplefilter("ignore", RuntimeWarning)
                z = np.nanmedian(stack, 0) if frames >= 3 else np.nanmean(stack, 0)
            z[cnt < math.ceil(frames / 2)] = np.nan
        if self.corrections:
            z = self._despeckle(z)
        valid = np.isfinite(z)
        roi_b = self.believed_roi()          # where the robot expects the bed (its own calibration, not the truth)
        meta["fill"] = float(valid[roi_b].mean()) if roi_b.any() else 0.0
        vr = valid & roi_b                   # where the bed should be seen: nothing is closer than ~0.29 m there
        meta["near_frac"] = float((z[vr] < NEAR_M).mean()) if vr.any() else 0.0           # something at the lens
        # back-project with the BELIEVED intrinsics and extrinsics (the calibration error lives here)
        d = self.w.d
        cal = self.calib
        fb = f * cal.f_scale
        v, u = np.mgrid[0:H, 0:W]
        zz = np.where(valid, z, 0.0)
        pc = np.stack([(u - W / 2 + 0.5 - cal.c_off[0]) * zz / fb, -(v - H / 2 + 0.5 - cal.c_off[1]) * zz / fb, -zz],
                      -1)[valid]
        Rb = d.cam_xmat[self.cam].reshape(3, 3) @ cal.R_err
        tb = d.cam_xpos[self.cam] + cal.t_err
        p = self._to_bed(pc @ Rb.T + tb)
        cam_b = self._to_bed(tb[None])[0]
        sig_cal = self._calib_budget_mm()
        if self.corrections and len(p):
            pl = fit_bed_plane(p, sig_bed, rng)
            if pl is not None:
                Rc = _rot_to_z(pl["n"])
                p = (p - cam_b) @ Rc.T + cam_b
                off = float(((pl["c"] - cam_b) @ Rc.T + cam_b)[2])
                p[:, 2] -= off
                meta["plane"] = {"tilt_deg": round(math.degrees(math.acos(min(1.0, pl["n"][2]))), 4),
                                 "offset_mm": round(off, 3), "inlier_frac": round(pl["inlier_frac"], 3),
                                 "rms_mm": round(pl["rms"], 3)}
                ol = fit_bed_outline(p, sig_bed, rng, budget_mm=sig_cal)
                if ol is not None:
                    a = math.radians(ol["dyaw_deg"])
                    ca, sa = math.cos(a), math.sin(a)
                    cx = C.BED_MM[0] / 2 + (ol["dx"] or 0.0)
                    cy = C.BED_MM[1] / 2 + (ol["dy"] or 0.0)
                    x, y = p[:, 0] - cx, p[:, 1] - cy
                    p[:, 0] = ca * x + sa * y + C.BED_MM[0] / 2
                    p[:, 1] = -sa * x + ca * y + C.BED_MM[1] / 2
                    cx_, cy_ = cam_b[0] - cx, cam_b[1] - cy
                    cam_b = np.array([ca * cx_ + sa * cy_ + C.BED_MM[0] / 2, -sa * cx_ + ca * cy_ + C.BED_MM[1] / 2,
                                      cam_b[2]])
                    meta["outline"] = {"dx": ol["dx"], "dy": ol["dy"], "dyaw_deg": round(ol["dyaw_deg"], 4),
                                       "nx": ol["nx"], "ny": ol["ny"], "edges": sorted(ol["edges"]),
                                       "blurred_bins": int(ol["blurred_bins"])}
                    if ol["nx"] and ol["ny"]:
                        sig_cal = SIGMA_CAL_FIT_MM * (1.0 if ol["nx"] == 2 and ol["ny"] == 2 else 1.6)
            cam_b[2] = cam_b[2] - (meta["plane"]["offset_mm"] if meta["plane"] else 0.0)
        keep = (p[:, 0] > -5) & (p[:, 0] < C.BED_MM[0] + 5) & (p[:, 1] > -5) & (p[:, 1] < C.BED_MM[1] + 5) & \
               (p[:, 2] < 80)
        p = p[keep]
        if self.corrections and meta["plane"] is not None:
            meta["noise_grid"] = noise_grid(p)
        meta["cam_bed_mm"] = [float(v) for v in cam_b]
        meta["sigma_cal_mm"] = float(sig_cal)
        meta["quality"] = self._quality(meta, len(p))
        meta["t_scan_s"] = time.perf_counter() - t0
        self.last = meta
        return p, meta

    def _calib_budget_mm(self):
        """Per-axis lateral error that the calibration error budget implies without correction (mm, 1 sigma)."""
        nz = self.noise
        lever = Z_REF * 1000
        return math.sqrt(nz.ext_t_mm ** 2 + (lever * math.radians(nz.ext_r_deg)) ** 2 + (150.0 * nz.f_rel) ** 2
                         + (lever * nz.c_px / self._f_px()) ** 2 + (lever * nz.depth_scale) ** 2 + SIGMA_CAL_FIT_MM ** 2)

    def _despeckle(self, z):
        """Flying-pixel / speckle filter: keep a depth pixel only if at least 2 of its 8 neighbours agree with it
        (within 4 sigma of the fused depth, >= 2 mm). Flying pixels sit between two surfaces and agree with neither;
        thin ridges (a 6 mm upright is ~5 px wide) keep their edge pixels, unlike a median filter."""
        valid = np.isfinite(z)
        zp = np.pad(np.where(valid, z, np.nan), 1, constant_values=np.nan)
        thr = np.maximum(4.0 * self.sigma_z_eff_mm(np.where(valid, z, Z_REF)) / 1000.0, 0.002)
        support = np.zeros(z.shape, np.int8)
        with np.errstate(invalid="ignore"):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy or dx:
                        nb = zp[1 + dy:1 + dy + z.shape[0], 1 + dx:1 + dx + z.shape[1]]
                        support += (np.abs(nb - z) < thr)
        out = z.copy()
        out[valid & (support < 2)] = np.nan
        return out

    def _quality(self, meta, n_pts):
        if n_pts == 0 or meta["fill"] < 0.15:
            return "failed"
        if not self.corrections:
            return "ok" if meta["fill"] >= 0.75 and meta["near_frac"] < NEAR_FRAC_MAX else "degraded"
        if meta["plane"] is None:
            return "failed"
        # from the data only (the simulated fault mode is never looked at): low fill, something at the lens, an edge
        # blurred (smeared lens), the plate outline not measured, or a noisy bed plane
        exp_rms = max(0.5, self.noise.sigma_z_mm(Z_REF) / math.sqrt(meta["frames"]))
        ol = meta["outline"]
        if meta["fill"] < 0.75 or meta["near_frac"] >= NEAR_FRAC_MAX or ol is None or not (ol["nx"] and ol["ny"]) \
                or ol["blurred_bins"] >= BLUR_BINS_MAX or meta["plane"]["rms_mm"] > 3 * exp_rms:
            return "degraded"
        return "ok"

    def close(self):
        self.r.close()
        if self.rs is not None:
            self.rs.close()


def _sym(sku):
    if sku == "purge_tab":
        return 180.0
    k = C.CATALOG[sku]
    if k.kind == "cyl":
        return None
    if k.kind == "box":
        return 90.0 if abs(k.dims[0] - k.dims[1]) < 1e-6 else 180.0
    return 360.0


def _large(o):
    k = C.CATALOG.get(o["sku"])
    return k is not None and k.large


def _in_footprint(o, pts, margin):
    """Points within the prior footprint (rotated by the prior yaw) + margin (mm)."""
    cx, cy = o["center_mm"]
    fx, fy = o.get("footprint_mm", (30, 30))
    yr = math.radians(o["yaw_deg"])
    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    u, v = dx * math.cos(yr) + dy * math.sin(yr), -dx * math.sin(yr) + dy * math.cos(yr)
    return (np.abs(u) < fx / 2 + margin) & (np.abs(v) < fy / 2 + margin)


def _explains(o, raised):
    """Raised points a slicer object accounts for: a disc around it, or its footprint + SEARCH_MM for a long part
    (a disc around a 150 mm bar would also swallow debris beside it)."""
    if _large(o):
        return _in_footprint(o, raised, SEARCH_MM)
    cx, cy = o["center_mm"]
    fx, fy = o.get("footprint_mm", (30, 30))
    return np.hypot(raised[:, 0] - cx, raised[:, 1] - cy) < 0.5 * math.hypot(fx, fy) + SEARCH_MM


def _estimate_legacy(points, objects):
    """The original estimator (perfect depth)."""
    raised = points[points[:, 2] > RAISED_MM]
    explained = np.zeros(len(raised), bool)
    est = {}
    for o in objects:
        cx, cy = o["center_mm"]
        fx, fy = o.get("footprint_mm", (30, 30))
        r = 0.5 * math.hypot(fx, fy) + SEARCH_MM
        explained |= _explains(o, raised)
        yr = math.radians(o["yaw_deg"])            # search inside the prior footprint (rotated) + margin
        dx, dy = raised[:, 0] - cx, raised[:, 1] - cy
        u, v = dx * math.cos(yr) + dy * math.sin(yr), -dx * math.sin(yr) + dy * math.cos(yr)
        near = (np.abs(u) < fx / 2 + FOOT_MARGIN_MM) & (np.abs(v) < fy / 2 + FOOT_MARGIN_MM)
        top = raised[near & (np.abs(raised[:, 2] - o["height_mm"]) < TOP_TOL_MM)]     # its own top face only
        if len(top) < MIN_TOP_PTS:
            # second pass: the part may have been nudged; look wider for a blob at the right height
            wide = raised[(np.hypot(raised[:, 0] - cx, raised[:, 1] - cy) < r + RELOCATE_MM)
                          & (np.abs(raised[:, 2] - o["height_mm"]) < TOP_TOL_MM)]
            if _large(o):    # a long part's wide search reaches its neighbours: never take their top faces
                for o2 in objects:
                    if o2 is not o:
                        wide = wide[~_in_footprint(o2, wide, 3.0)]
            if len(wide) >= MIN_TOP_PTS:
                c0 = np.median(wide[:, :2], axis=0)
                if _large(o):
                    top = wide[_in_footprint(dict(o, center_mm=c0), wide, 2.0)]
                else:
                    top = wide[np.hypot(wide[:, 0] - c0[0], wide[:, 1] - c0[1]) < 0.5 * math.hypot(fx, fy) + 2]
            if len(top) < MIN_TOP_PTS:
                est[o["name"]] = {"found": False, "n_top": int(len(top)), "center_mm": [cx, cy],
                                  "yaw_deg": o["yaw_deg"]}
                continue
        c = top[:, :2].mean(0)
        yaw = o["yaw_deg"]
        period = _sym(o["sku"])
        if period is not None and period < 360 and len(top) > 40:
            import cv2
            (rx, ry), (w_, h_), ang = cv2.minAreaRect(top[:, :2].astype(np.float32))
            c = np.array([rx, ry])
            if abs(w_ - h_) > 0.12 * max(w_, h_):          # elongated top face: its long side gives the yaw
                long_ang = ang if w_ >= h_ else ang + 90.0
                k = C.CATALOG.get(o["sku"])
                long_is_x = k is None or k.footprint[0] >= k.footprint[1]
                cand = long_ang - (0.0 if long_is_x else 90.0)
                dyaw = (cand - o["yaw_deg"] + period / 2) % period - period / 2
                if abs(dyaw) <= MAX_YAW_CORR_DEG:          # prints only rotate a little: larger = misfit
                    yaw = o["yaw_deg"] + dyaw
        off = _top_offset(o)       # top face off the footprint centre (L bracket's upright, a real part's top)
        if off is not None:
            c = c - _rot2(yaw) @ off
        est[o["name"]] = {"found": True, "n_top": int(len(top)), "center_mm": [float(c[0]), float(c[1])],
                          "yaw_deg": float(yaw)}
    unknown = int((~explained).sum())
    return est, unknown


def _yaw_observable(o):
    """True if the camera measures this object's yaw (elongated box top face or the L bracket)."""
    k = C.CATALOG.get(o["sku"])
    if k is None:
        return o["sku"] == "purge_tab"
    if k.kind == "L":
        return True
    return k.kind == "box" and abs(k.dims[0] - k.dims[1]) > 0.12 * max(k.dims[0], k.dims[1])


def _top_offset(o):
    """Centre of the top face relative to the footprint centre, in the part frame (mm), or None if they coincide:
    the L bracket's upright, or a real part's top face (SKU.top_mm, from its mesh)."""
    k = C.CATALOG.get(o["sku"])
    if k is not None and k.kind == "L":
        ln, wd, bt, uh, ut = k.dims
        return np.array([-ln / 2 + ut / 2, 0.0])
    if k is not None and k.top_mm and math.hypot(k.top_mm[2], k.top_mm[3]) > 0.05:
        return np.array(k.top_mm[2:4], float)
    return None


def _rot2(yaw_deg):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return np.array([[c, -s], [s, c]])


def _lever_mm(o):
    """Distance from the measured top face to the footprint centre (the L bracket's upright, a real part's top)."""
    off = _top_offset(o)
    return 0.0 if off is None else float(np.hypot(*off))


def _decorate(e, o, cov, yaw_sig, quality, observed=True, prior_rejected=False):
    cov = np.asarray(cov, float)
    e.update(sigma_mm=float(math.sqrt(max(np.trace(cov), 0.0))), cov_mm2=[[float(v) for v in r] for r in cov],
             yaw_sigma_deg=float(yaw_sig), quality=quality, observed=bool(observed))
    if prior_rejected:
        e["prior_rejected"] = True
    return e


def _top_dims(o):
    """Top-face dims (along local x, y) in mm."""
    k = C.CATALOG.get(o["sku"])
    if k is None:
        fx, fy = o.get("footprint_mm", (30, 30))
        return float(fx), float(fy)
    if k.kind == "L":
        return float(k.dims[4]), float(k.dims[1])
    if k.top_mm:                                   # a real part: its top face, from the mesh
        return max(float(k.top_mm[0]), 1.0), max(float(k.top_mm[1]), 1.0)
    return float(k.footprint[0]), float(k.footprint[1])


def _top_area(o):
    k = C.CATALOG.get(o["sku"])
    tx, ty = _top_dims(o)
    if k is not None and len(k.top_mm) > 4:          # a real part's top face (a rim is far less than its extent)
        return max(float(k.top_mm[4]), 1.0)
    return math.pi * tx * tx / 4 if (k is not None and k.kind == "cyl") else tx * ty


def _pix_area_mm2(meta, x, y, z):
    """Bed area one depth pixel covers at bed point (x, y, z) mm (range and incidence from the camera)."""
    c = np.asarray(meta["cam_bed_mm"])
    r = math.dist(c, (x, y, z))
    cos_t = max(0.2, (c[2] - z) / max(r, 1e-6))
    return (r / meta["f_px"]) ** 2 / cos_t


def _range_m(meta, x, y, z):
    return math.dist(meta["cam_bed_mm"], (x, y, z)) / 1000.0


def sigma_z_fused_mm(meta, z_range_m):
    k = 1.0 / math.sqrt(meta["frames"])
    return meta["axial_mm"] * (z_range_m / Z_REF) ** 2 * k


NOISE_CELL_MM = 48.0          # local depth-noise grid (robust MAD of bed-level residuals per cell)


def noise_grid(p, cell_mm=NOISE_CELL_MM):
    """Per-cell robust sigma (mm) of bed-level points (|z| < 4 mm: below any part top) after levelling: 1.4826 x the
    smaller of median |z| and the MAD around the cell's own median. A cell is large enough that a part or leftover lump
    covers well under half of it, but a large flat part lower than 4 mm can fill a whole cell: then median |z| is its
    height, not noise (it would hide the part), while the MAD around the median stays small. NaN = too few points."""
    n = int(math.ceil(C.BED_MM[0] / cell_mm))
    m = int(math.ceil(C.BED_MM[1] / cell_mm))
    q = p[np.abs(p[:, 2]) < 4.0]
    ij = np.clip(np.floor(q[:, :2] / cell_mm).astype(int), 0, [n - 1, m - 1])
    out = np.full((n, m), np.nan)
    key = ij[:, 0] * m + ij[:, 1]
    order = np.argsort(key, kind="stable")
    key, z = key[order], q[order, 2]
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]]) if len(key) else np.array([], int)
    for a, b in zip(starts, np.r_[starts[1:], len(key)]):
        if b - a >= 30:
            zc = z[a:b]
            out.flat[key[a]] = 1.4826 * min(float(np.median(np.abs(zc))), float(np.median(np.abs(zc - np.median(zc)))))
    return {"cell_mm": cell_mm, "sigma": out.tolist()}


def point_sigma_mm(P, meta):
    """1-sigma depth noise per point (mm): max(noise model at the point's range, measured local bed noise)."""
    c = np.asarray(meta["cam_bed_mm"])
    rng_m = np.linalg.norm(P - c, axis=1) / 1000.0
    s = np.asarray(sigma_z_fused_mm(meta, rng_m), float)
    g = meta.get("noise_grid")
    if g:
        sg = np.asarray(g["sigma"], float)
        ij = np.clip(np.floor(P[:, :2] / g["cell_mm"]).astype(int), 0, np.array(sg.shape) - 1)
        loc = sg[ij[:, 0], ij[:, 1]]
        s = np.where(np.isfinite(loc), np.maximum(s, loc), s)
    return s


def raised_mask(points):
    """Mask of points that are really above the bed. Ideal camera: z > RAISED_MM (original rule). Noisy camera:
    z > max(RAISED_MM, RAISED_SIGMAS x the point's depth sigma: the noise model at its range or the measured local bed
    noise, whichever is larger), and isolated specks (2 mm grid components with < MIN_RAISED_PTS points) dropped."""
    P = np.asarray(points)
    meta = _meta(points)
    if meta is None:
        return P[:, 2] > RAISED_MM
    if not meta.get("corrections", True):
        return P[:, 2] > RAISED_MM
    thr = np.maximum(RAISED_MM, RAISED_SIGMAS * point_sigma_mm(P, meta))
    m = P[:, 2] > thr
    if m.sum() == 0:
        return m
    lab, sizes = _components(P[m])
    good = sizes[lab] >= MIN_RAISED_PTS
    out = np.zeros(len(P), bool)
    out[np.flatnonzero(m)[good]] = True
    return out


def raised_points(points):
    """Raised points (robust to camera noise): use instead of points[points[:, 2] > RAISED_MM]."""
    return np.asarray(points)[raised_mask(points)]


def _components(p, grid_mm=2.0):
    """Connected components of points on a grid_mm grid (8-connected, one cell of dilation).
    -> (label per point (0..n-1), points per label)."""
    ij = np.floor(p[:, :2] / grid_mm).astype(int)
    ij -= ij.min(0)
    g = np.zeros(ij.max(0) + 1, bool)
    g[ij[:, 0], ij[:, 1]] = True
    lab, n = ndimage.label(ndimage.binary_dilation(g, iterations=1), structure=np.ones((3, 3)))
    k = lab[ij[:, 0], ij[:, 1]] - 1
    return k, np.bincount(k, minlength=n)


def _extent_centre(p, q=1.0, n_dir=8):
    """Centre of a round top face from its trimmed extents in n_dir directions (least squares). The mean is pulled
    toward the camera by side-wall points inside the height band; the extents are not (the wall is at the rim)."""
    th = np.arange(n_dir) * math.pi / n_dir
    D = np.stack([np.cos(th), np.sin(th)], 1)
    proj = p @ D.T
    mid = 0.5 * (np.percentile(proj, q, axis=0) + np.percentile(proj, 100 - q, axis=0))
    return np.linalg.lstsq(D, mid, rcond=None)[0]


def _fuse(mu_p, cov_p, mu_c, cov_c):
    """Gaussian fusion of prior and measurement -> (mu, cov, d2)."""
    S = cov_p + cov_c
    dlt = mu_c - mu_p
    d2 = float(dlt @ np.linalg.solve(S, dlt))
    K = cov_p @ np.linalg.inv(S)
    return mu_p + K @ dlt, (np.eye(2) - K) @ cov_p, d2


def _prior_only(o, inflate, quality, n_top=0):
    s = PRIOR_SIGMA_MM * inflate
    e = {"found": True, "n_top": int(n_top), "center_mm": [float(v) for v in o["center_mm"]],
         "yaw_deg": float(o["yaw_deg"])}
    return _decorate(e, o, np.eye(2) * s * s, PRIOR_YAW_SIGMA_DEG * inflate, quality, observed=False)


def _estimate_robust(points, objects, meta):
    import cv2
    P = np.asarray(points)
    q_scan = meta["quality"]
    if q_scan == "failed" or len(P) == 0:
        return {o["name"]: _prior_only(o, FAIL_INFLATE, "failed") for o in objects}, 0
    rm = raised_mask(points)
    raised = P[rm]
    explained = np.zeros(len(raised), bool)
    est = {}
    sig_cal = meta["sigma_cal_mm"]

    def footprint(o, pts, margin):
        cx, cy = o["center_mm"]
        fx, fy = o.get("footprint_mm", (30, 30))
        yr = math.radians(o["yaw_deg"])
        dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
        u, v = dx * math.cos(yr) + dy * math.sin(yr), -dx * math.sin(yr) + dy * math.cos(yr)
        return (np.abs(u) < fx / 2 + margin) & (np.abs(v) < fy / 2 + margin)

    for o in objects:
        cx, cy = o["center_mm"]
        fx, fy = o.get("footprint_mm", (30, 30))
        h = float(o["height_mm"])
        r = 0.5 * math.hypot(fx, fy) + SEARCH_MM
        explained |= _explains(o, raised)
        rng_m = _range_m(meta, cx, cy, h)
        sz = sigma_z_fused_mm(meta, rng_m)
        tol = max(TOP_TOL_MM, 3.0 * sz)
        pix = _pix_area_mm2(meta, cx, cy, h)
        n_exp = _top_area(o) / pix
        min_pts = int(max(8, min(MIN_TOP_PTS, 0.3 * n_exp)))
        yr = math.radians(o["yaw_deg"])
        dx, dy = raised[:, 0] - cx, raised[:, 1] - cy
        near = footprint(o, raised, FOOT_MARGIN_MM)
        if len(objects) > 1:  # a neighbour within the search margin (a slicer's own arrange packs parts a few mm apart):
            near &= ~np.any([footprint(o2, raised, 1.0) for o2 in objects if o2 is not o], axis=0)   # not ours
        if _large(o):        # the far end of a long part is noisier than its centre: a band from each point's own sigma
            band = np.abs(raised[:, 2] - h) < np.maximum(TOP_TOL_MM, 3.0 * point_sigma_mm(raised, meta))
        else:
            band = np.abs(raised[:, 2] - h) < tol
        top = raised[near & band]
        weak = False
        if len(top) < min_pts:
            # nudged? look wider, but not inside another object's footprint (that would be the neighbour)
            others = np.zeros(len(raised), bool)
            for o2 in objects:
                if o2 is not o:
                    others |= footprint(o2, raised, 3.0)
            wide = raised[(np.hypot(dx, dy) < r + RELOCATE_MM) & band & ~others]
            if len(wide) >= min_pts:
                c0 = np.median(wide[:, :2], axis=0)
                if _large(o):
                    top = wide[_in_footprint(dict(o, center_mm=c0), wide, 2.0)]
                else:
                    top = wide[np.hypot(wide[:, 0] - c0[0], wide[:, 1] - c0[1]) < 0.5 * math.hypot(fx, fy) + 2]
        if len(top) < min_pts:
            # sparse / noisy top face (IR-dark filament, holes): a wider band, if its points sit at the right height
            t2 = raised[near & (np.abs(raised[:, 2] - h) < 2.5 * tol)]
            if len(t2) >= max(6, min_pts // 2) and abs(float(np.median(t2[:, 2])) - h) < tol:
                top, weak = t2, True
        if len(top) < min_pts and not weak:
            # missing, or just not seen? Look at every point (any height) in the core of the prior footprint:
            # few points = not seen (holes, occluder, shadow); mostly bed level = missing (bare bed where it should
            # be); something raised but no top face = an anomaly (tipped / wrong part) on a good scan, unknown on a
            # degraded one (smeared lens)
            dxa, dya = P[:, 0] - cx, P[:, 1] - cy
            ua, va = dxa * math.cos(yr) + dya * math.sin(yr), -dxa * math.sin(yr) + dya * math.cos(yr)
            core = (np.abs(ua) < 0.35 * fx) & (np.abs(va) < 0.35 * fy)
            n_in = int(core.sum())
            n_in_exp = 0.49 * fx * fy / _pix_area_mm2(meta, cx, cy, 0.0)
            n_bed = int((core & (np.abs(P[:, 2]) < max(1.0, 3.0 * meta["sigma_z_far_mm"]))).sum())
            if n_in < OBSERVED_FRAC * n_in_exp or (n_bed < 0.5 * n_in and q_scan != "ok"):
                est[o["name"]] = _prior_only(o, FAIL_INFLATE, "degraded", n_top=len(top))
            else:
                e = {"found": False, "n_top": int(len(top)), "center_mm": [cx, cy], "yaw_deg": o["yaw_deg"]}
                est[o["name"]] = _decorate(e, o, np.eye(2) * PRIOR_SIGMA_MM ** 2, PRIOR_YAW_SIGMA_DEG, q_scan)
            continue
        # robust top face: gate around the median, then the original centre / yaw logic on the inliers
        t2 = top[:, :2]
        med = np.median(t2, 0)
        tx, ty = _top_dims(o)
        if _large(o):        # a long top face: gate across it only (a disc would admit points beside it); along it the
            # prior footprint gate (near) holds, the median is no centre there (the near end is sampled densely)
            t2 = t2[np.abs((t2 - med) @ [-math.sin(yr), math.cos(yr)]) < ty / 2 + 2.0]
        else:
            t2 = t2[np.hypot(t2[:, 0] - med[0], t2[:, 1] - med[1]) < 0.5 * math.hypot(tx, ty) + 2.0]
        n = len(t2)
        c = t2.mean(0)
        kk = C.CATALOG.get(o["sku"])
        if kk is not None and kk.kind == "cyl" and n >= 20:
            c = _extent_centre(t2)
        yaw = float(o["yaw_deg"])
        yaw_meas = None
        period = _sym(o["sku"])
        if period is not None and period < 360 and n > 40:
            (rx, ry), (w_, h_), ang = cv2.minAreaRect(t2.astype(np.float32))
            c = np.array([rx, ry])
            if abs(w_ - h_) > 0.12 * max(w_, h_):
                long_ang = ang if w_ >= h_ else ang + 90.0
                tdx, tdy = _top_dims(o)
                long_is_x = tdx >= tdy
                cand = long_ang - (0.0 if long_is_x else 90.0)
                dyaw = (cand - o["yaw_deg"] + period / 2) % period - period / 2
                if abs(dyaw) <= MAX_YAW_CORR_DEG:
                    yaw_meas = o["yaw_deg"] + dyaw
        k = C.CATALOG.get(o["sku"])
        if k is not None and k.kind == "L":
            # yaw from the base: its top face (base_t high) lies on the +x side of the upright
            ln, wd, bt, uh, ut = k.dims
            bb = raised[near & (np.abs(raised[:, 2] - bt) < max(TOP_TOL_MM, 3.0 * sz))
                        & (np.hypot(raised[:, 0] - c[0], raised[:, 1] - c[1]) < ln)]
            if len(bb) >= min_pts:
                vb = np.median(bb[:, :2], 0) - c
                dist = float(np.hypot(*vb))
                if 0.6 * (ln - ut) / 2 < dist < 1.4 * (ln - ut) / 2 + 4:
                    cand = math.degrees(math.atan2(vb[1], vb[0]))
                    dyaw = (cand - o["yaw_deg"] + 180) % 360 - 180
                    if abs(dyaw) <= MAX_YAW_CORR_DEG:
                        yaw_meas = o["yaw_deg"] + dyaw
        # camera measurement covariance (per axis): estimator floor + per-point noise / effective count
        # + calibration residual + incomplete top face
        # scale-type calibration errors (focal length, depth scale) are not removed by the rigid outline
        # correction: they grow with the distance from the bed centre
        r_c = math.hypot(cx - C.BED_MM[0] / 2, cy - C.BED_MM[1] / 2)
        sig_cal = math.hypot(meta["sigma_cal_mm"], r_c * meta.get("scale_sigma", 0.0))
        cos_t = max(0.2, (meta["cam_bed_mm"][2] - h) / max(rng_m * 1000, 1e-6))
        sin_t = math.sqrt(max(0.0, 1 - cos_t ** 2))
        s_pt2 = (meta["lateral_px"] * rng_m * 1000 / meta["f_px"]) ** 2 + (sz * sin_t) ** 2
        n_eff = n / (1.0 + meta["corr_frac"] * math.pi * meta["corr_px"] ** 2)
        cover = min(1.0, n / max(n_exp, 1.0))
        s_cov = 0.29 * (1.0 - cover) * max(tx, ty)
        var_c = SIGMA_FLOOR_MM ** 2 + s_pt2 / max(n_eff, 1.0) + sig_cal ** 2 + s_cov ** 2
        # yaw: fuse camera yaw (if measured) with the prior
        if yaw_meas is not None:
            if k is not None and k.kind == "L":
                lever = (k.dims[0] - k.dims[4]) / 2
                ys_c = math.hypot(YAW_FLOOR_L_DEG, math.degrees(YAW_K_L * math.sqrt(var_c) / lever))
            else:
                ys_c = math.hypot(YAW_FLOOR_DEG, math.degrees(YAW_K * math.sqrt(var_c) / max(tx, ty)))
            dy_ = (yaw_meas - yaw + 180) % 360 - 180
            if abs(dy_) > YAW_GATE_SIG * math.hypot(ys_c, PRIOR_YAW_SIGMA_DEG):
                yaw, ys = yaw_meas, ys_c
            else:
                wgt = PRIOR_YAW_SIGMA_DEG ** 2 / (PRIOR_YAW_SIGMA_DEG ** 2 + ys_c ** 2)
                yaw = yaw + wgt * dy_
                ys = math.sqrt(wgt) * ys_c
        else:
            ys = PRIOR_YAW_SIGMA_DEG
        lev = _lever_mm(o)
        if lev:                    # the top face is off the footprint centre: shift by it at the estimated yaw
            c = c - _rot2(yaw) @ _top_offset(o)
            var_c += (lev * math.radians(ys)) ** 2
        cov_c = np.eye(2) * var_c
        cov_p = np.eye(2) * PRIOR_SIGMA_MM ** 2
        mu, cov, d2 = _fuse(np.array([cx, cy], float), cov_p, np.asarray(c, float), cov_c)
        rejected = d2 > GATE_CHI2
        if rejected:
            mu, cov = np.asarray(c, float), cov_c
        quality = q_scan if (cover >= 0.5 and not weak) else "degraded"
        e = {"found": True, "n_top": int(n), "center_mm": [float(mu[0]), float(mu[1])], "yaw_deg": float(yaw)}
        est[o["name"]] = _decorate(e, o, cov, ys, quality, prior_rejected=rejected)
    unknown = int((~explained).sum())
    return est, unknown


def _naive_sigma(meta):
    return math.hypot(SIGMA_FLOOR_LEGACY_MM, meta["sigma_cal_mm"] or 0.0)


def estimate(points, objects):
    """objects: [{"name", "sku", "center_mm", "yaw_deg", "height_mm"}] (slicer prior).
    Returns ({name: {"center_mm", "yaw_deg", "n_top", "found", "sigma_mm", "cov_mm2", "quality", "yaw_sigma_deg",
    "observed"}}, unknown_point_count).
    Ideal camera (plain array): the original estimator, unchanged; sigma from the measured estimator floor.
    Noisy camera (BedPoints): the robust estimator + Bayesian fusion with the slicer prior (see module doc)."""
    meta = _meta(points)
    if meta is not None and meta.get("corrections", True):
        return _estimate_robust(points, objects, meta)
    est, unknown = _estimate_legacy(np.asarray(points), objects)
    s = SIGMA_FLOOR_LEGACY_MM if meta is None else _naive_sigma(meta)
    q = "ok" if meta is None else meta["quality"]
    by_name = {o["name"]: o for o in objects}
    for n, e in est.items():
        o = by_name[n]
        if e["found"]:      # the original estimator keeps the prior yaw for the L bracket: lever x yaw error
            var = s * s + (_lever_mm(o) * math.radians(PRIOR_YAW_SIGMA_DEG)) ** 2
            ys = YAW_FLOOR_DEG if _yaw_observable(o) and not _lever_mm(o) else PRIOR_YAW_SIGMA_DEG
            _decorate(e, o, np.eye(2) * var, ys, q)
        else:
            _decorate(e, o, np.eye(2) * PRIOR_SIGMA_MM ** 2, PRIOR_YAW_SIGMA_DEG, q)
    return est, unknown


def leftover_blobs(points):
    """Raised components that count as leftovers: [{"lo", "hi", "h", "n"}] (bed mm). Noise-robust (raised_mask)."""
    P = np.asarray(points)
    m = raised_mask(points)
    q = P[m]
    if len(q) < LEFTOVER_MIN_PTS:
        return []
    lab, sizes = _components(q)
    out = []
    for b in np.flatnonzero(sizes >= LEFTOVER_MIN_PTS):
        r = q[lab == b]
        if np.percentile(r[:, 2], 95) < LEFTOVER_MIN_H_MM:
            continue
        out.append({"lo": r[:, :2].min(0).tolist(), "hi": r[:, :2].max(0).tolist(),
                    "h": float(np.percentile(r[:, 2], 95)), "n": int(len(r))})
    return out


def bed_empty(points, tol_pts=30):
    """True if fewer than tol_pts points are raised above the bed (leftovers / debris otherwise).
    Noisy camera: no leftover component (leftover_blobs) on the bed; a failed scan cannot verify -> False."""
    meta = _meta(points)
    if meta is None or not meta.get("corrections", True):
        if meta is not None and meta["quality"] == "failed":
            return False
        return int((np.asarray(points)[:, 2] > RAISED_MM).sum()) < tol_pts
    if meta["quality"] == "failed":
        return False
    if meta["near_frac"] >= NEAR_FRAC_MAX and unseen_area_mm2(points) > UNSEEN_MM2:
        return False                               # something at the lens hides part of the bed: not verified
    return len(leftover_blobs(points)) == 0


def thin_leftovers(points, objects):
    """Planned parts too flat for leftover_blobs (0 < height < THIN_PART_MM) that are still on the bed, by name.
    Self-referenced per part: the THIN_PCT percentile of the heights inside the slicer footprint (shrunk 1 mm) against
    the same percentile of the bed ring around it (THIN_RING_MM outside, raised points left out); a rise of at least
    max(THIN_RISE_MIN_MM, THIN_RISE_FRAC x planned height) means the part is still there. A footprint or ring the
    camera did not see (< THIN_MIN_PTS points) is reported too: it cannot be verified."""
    P = np.asarray(points)
    if not len(P):
        return [o["name"] for o in objects if 0.0 < float(o.get("height_mm") or 0.0) < THIN_PART_MM]
    raised = raised_mask(points)
    out = []
    for o in objects:
        h = float(o.get("height_mm") or 0.0)
        if not 0.0 < h < THIN_PART_MM:
            continue
        fx, fy = (float(v) for v in o["footprint_mm"])
        loc = (P[:, :2] - np.asarray(o["center_mm"], float)) @ _rot2(o["yaw_deg"])     # bed -> part frame
        ax, ay = np.abs(loc[:, 0]), np.abs(loc[:, 1])
        inside = (ax <= max(fx / 2 - 1.0, fx / 4)) & (ay <= max(fy / 2 - 1.0, fy / 4))
        d = np.maximum(ax - fx / 2, ay - fy / 2)
        ring = (d >= THIN_RING_MM[0]) & (d <= THIN_RING_MM[1]) & ~raised
        if inside.sum() < THIN_MIN_PTS or ring.sum() < THIN_MIN_PTS:
            out.append(o["name"])
            continue
        rise = np.percentile(P[inside, 2], THIN_PCT) - np.percentile(P[ring, 2], THIN_PCT)
        if rise >= max(THIN_RISE_MIN_MM, THIN_RISE_FRAC * h):
            out.append(o["name"])
    return out


def cannot_verify(points):
    """bed_empty is False only because this scan could not see the whole bed (failed, or something at the lens hides
    part of it), not because of a leftover it saw: worth another scan before calling a person."""
    meta = _meta(points)
    if meta is None or not meta.get("corrections", True):
        return meta is not None and meta["quality"] == "failed"
    if meta["quality"] == "failed":
        return True
    return meta["near_frac"] >= NEAR_FRAC_MAX and unseen_area_mm2(points) > UNSEEN_MM2 and not leftover_blobs(points)


def unseen_area_mm2(points, cell_mm=6.0):
    """Largest connected bed patch (cell_mm grid) without a single depth point: what the camera did not see."""
    P = np.asarray(points)
    n = int(C.BED_MM[0] // cell_mm)
    m = int(C.BED_MM[1] // cell_mm)
    ij = np.floor(P[:, :2] / cell_mm).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < n) & (ij[:, 1] >= 0) & (ij[:, 1] < m)
    seen = np.zeros((n, m), bool)
    seen[ij[ok, 0], ij[ok, 1]] = True
    lab, k = ndimage.label(~seen)
    if k == 0:
        return 0.0
    return float(np.bincount(lab.ravel())[1:].max() * cell_mm * cell_mm)


def plate_objects(plate):
    """Slicer prior for perception: plate objects + the purge tab, with known heights and local footprints."""
    objs = [dict(o, height_mm=C.CATALOG[o["sku"]].height, footprint_mm=C.CATALOG[o["sku"]].footprint)
            for o in plate["objects"]]
    pt = plate.get("purge_tab")
    if pt is None:                                 # no tab on this plate (e.g. a Bambu printer: it purges into a chute)
        return objs
    objs.append({"name": f"{plate['plate_id']}_purge_tab", "sku": "purge_tab", "center_mm": pt["center_mm"],
                 "yaw_deg": 0.0, "height_mm": float(pt["dims_mm"][2]), "footprint_mm": tuple(pt["dims_mm"][:2])})
    return objs
