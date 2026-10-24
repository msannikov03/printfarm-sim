"""Depth-camera error model for the chamber camera (bed verification).

Stock enclosed printers ship an RGB chamber camera only. ASSUMPTION: a short-range active-stereo depth camera
(Intel RealSense D405/D435-type or an Orbbec equivalent) is added on the chamber ceiling, ~0.33 m above the bed at
the unload height, looking down at the bed (sim camera `chamber{i}`: 640x480, 86 x 70 deg FOV, f = 343 px; the bed
is seen at 0.33-0.55 m range).

What is modelled (sources in docs/perception.md):
  axial noise     stereo triangulation: sigma_z = z^2 * sigma_d / (f * b)  ->  sigma_z(z) = axial_mm * (z / Z_REF)^2;
                  part of it spatially correlated over the block-matching window (axial_corr_px)
  lateral noise   per-pixel jitter of where a depth sample lands (edge position noise), in pixels
  quantization    depth unit of the output stream (0.1 mm or 1 mm)
  depth scale     relative scale error of the depth stream (persistent; what an on-site "tare" calibration bounds)
  flying pixels   at depth discontinuities a pixel takes a depth between foreground and background
  dropouts        iid invalid pixels, structured holes (texture-poor patches), a specular sheen spot on the PEI
                  sheet (projector reflection), IR-dark parts (black filament) with low fill and higher noise
  temporal noise  every frame has fresh noise; the scan averages `frames` frames
  calibration     extrinsic pose error (translation + rotation, Gaussian, persistent per camera) and intrinsic error
                  (focal length, principal point): the software back-projects with the believed (wrong) parameters
  failures        frame grab fails (retried), camera dead for the scan, blackout frame (no valid depth),
                  partial occlusion near the lens (cable / the arm / a hand), smeared or fogged lens region

Presets: "ideal" (no error at all: exactly the old perfect-render behaviour, the default), "realistic", "harsh".
All randomness comes from the caller's numpy Generator (seeded).
"""
import math
from dataclasses import dataclass, fields, replace

import cv2
import numpy as np

Z_REF = 0.40            # m: axial_mm is the 1-sigma axial noise at this range (bed centre is ~0.40 m away)


@dataclass(frozen=True)
class CameraNoise:
    name: str = "custom"
    # ---- per-pixel depth errors (fresh every frame) ----
    axial_mm: float = 0.0          # 1-sigma axial (depth) noise at Z_REF; scales with z^2
    axial_corr_frac: float = 0.0   # share of the axial variance that is spatially correlated
    axial_corr_px: float = 3.0     # correlation length of that share (stereo matching window), px
    lateral_px: float = 0.0        # 1-sigma lateral jitter of each depth sample, px
    quant_mm: float = 0.0          # depth unit (output rounded to it)
    flying_frac: float = 0.0       # share of depth-edge pixels that become flying pixels
    edge_mm: float = 4.0           # a depth step larger than this (3x3 window) is a depth edge
    dropout: float = 0.0           # iid invalid pixels
    holes: float = 0.0             # image share covered by structured holes (static within a scan)
    hole_px: float = 5.0           # hole blob scale, px
    sheen_p: float = 0.0           # probability of a specular sheen spot on the bed in a scan
    sheen_px: float = 30.0         # sheen spot radius, px
    sheen_drop: float = 0.0        # dropout inside the sheen spot
    dark_frac: float = 0.0         # share of parts printed in IR-dark (black) filament
    dark_drop: float = 0.0         # dropout on dark parts
    dark_noise: float = 1.0        # axial noise multiplier on dark parts
    # ---- calibration errors (persistent per camera, resampled by BedCamera.resample) ----
    depth_scale: float = 0.0       # 1-sigma relative depth scale error
    ext_t_mm: float = 0.0          # extrinsic translation error, 1 sigma per axis
    ext_r_deg: float = 0.0         # extrinsic rotation error, 1 sigma per axis
    f_rel: float = 0.0             # focal length relative error, 1 sigma
    c_px: float = 0.0              # principal point error, 1 sigma per axis, px
    # ---- per-scan failure modes ----
    p_no_frame: float = 0.0        # a frame grab fails (the camera retries it)
    p_dead: float = 0.0            # the camera stays unavailable for the whole scan
    p_blackout: float = 0.0        # the scan's frames carry no valid depth (projector / exposure failure)
    p_occlusion: float = 0.0       # something near the lens covers part of the view
    occl_frac: tuple = (0.10, 0.30)
    p_smear: float = 0.0           # smeared / fogged lens region: blurred, noisier, more dropouts
    smear_frac: tuple = (0.10, 0.30)
    frames: int = 1                # frames averaged per scan by default

    @property
    def is_ideal(self):
        skip = {"name", "axial_corr_px", "edge_mm", "hole_px", "sheen_px", "dark_noise", "occl_frac", "smear_frac",
                "frames"}
        return all(getattr(self, f.name) == 0 for f in fields(self) if f.name not in skip)

    def sigma_z_mm(self, z_m):
        """1-sigma axial noise of one frame at range z (m), mm."""
        return self.axial_mm * (np.asarray(z_m) / Z_REF) ** 2


# Numbers and their reasoning: docs/perception.md. Stereo: sigma_z = z^2 * sigma_d / (f * b).
# realistic ~ D435-type (b = 50 mm) at 848x480 (f ~ 424 px), subpixel sigma_d ~ 0.1 px (Intel: < 0.1 for a
#   well-calibrated camera on a textured target): 0.40^2 * 0.1 / (424 * 0.05) = 0.75 mm at 0.4 m.
# harsh ~ subpixel 0.2 px (Intel's re-calibration threshold), poor texture, a bumped camera, more failures.
PRESETS = {
    "ideal": CameraNoise(name="ideal"),
    "realistic": CameraNoise(
        name="realistic", axial_mm=0.8, axial_corr_frac=0.5, axial_corr_px=3.0, lateral_px=0.3, quant_mm=0.1,
        flying_frac=0.3, dropout=0.01, holes=0.005, sheen_p=0.5, sheen_px=28.0, sheen_drop=0.6,
        dark_frac=0.2, dark_drop=0.3, dark_noise=1.5,
        depth_scale=0.002, ext_t_mm=1.0, ext_r_deg=0.25, f_rel=0.003, c_px=1.0,
        p_no_frame=0.01, p_dead=0.002, p_blackout=0.005, p_occlusion=0.02, p_smear=0.02, frames=4),
    "harsh": CameraNoise(
        name="harsh", axial_mm=1.6, axial_corr_frac=0.6, axial_corr_px=4.0, lateral_px=0.6, quant_mm=1.0,
        flying_frac=0.6, dropout=0.04, holes=0.03, sheen_p=1.0, sheen_px=45.0, sheen_drop=0.8,
        dark_frac=0.4, dark_drop=0.6, dark_noise=2.5,
        depth_scale=0.008, ext_t_mm=3.0, ext_r_deg=1.0, f_rel=0.01, c_px=3.0,
        p_no_frame=0.05, p_dead=0.02, p_blackout=0.02, p_occlusion=0.10, p_smear=0.10, frames=4),
}


def get_noise(noise):
    """None | preset name | CameraNoise -> CameraNoise."""
    if noise is None:
        return PRESETS["ideal"]
    if isinstance(noise, CameraNoise):
        return noise
    if isinstance(noise, str):
        if noise not in PRESETS:
            raise ValueError(f"unknown camera noise preset {noise!r}; one of {sorted(PRESETS)}")
        return PRESETS[noise]
    raise TypeError(f"noise must be None, a preset name or a CameraNoise, not {type(noise).__name__}")


def scaled(noise, k):
    """The same error model with every error magnitude and probability multiplied by k (sensitivity sweeps)."""
    keep = {"name", "axial_corr_frac", "axial_corr_px", "edge_mm", "hole_px", "sheen_px", "dark_noise", "occl_frac",
            "smear_frac", "frames"}
    ch = {f.name: getattr(noise, f.name) * k for f in fields(noise) if f.name not in keep}
    for p in ("flying_frac", "dropout", "holes", "sheen_p", "sheen_drop", "dark_frac", "dark_drop", "p_no_frame",
              "p_dead", "p_blackout", "p_occlusion", "p_smear"):
        ch[p] = min(1.0, ch[p])
    return replace(noise, name=f"{noise.name}x{k:g}", **ch)


# ------------------------------------------------------------------ calibration (persistent per camera)
def rotvec_to_mat(r):
    return cv2.Rodrigues(np.asarray(r, np.float64).reshape(3, 1))[0]


@dataclass
class Calibration:
    """Believed-minus-true camera parameters. R_err rotates in the camera frame, t_err is in world metres."""
    R_err: np.ndarray
    t_err: np.ndarray
    f_scale: float
    c_off: np.ndarray       # principal point offset (px): believed = true + c_off
    depth_scale: float      # measured depth = true depth * (1 + depth_scale)

    @staticmethod
    def sample(noise, rng):
        r = rng.normal(0.0, math.radians(noise.ext_r_deg), 3) if noise.ext_r_deg else np.zeros(3)
        t = rng.normal(0.0, noise.ext_t_mm / 1000, 3) if noise.ext_t_mm else np.zeros(3)
        fs = 1.0 + (rng.normal(0.0, noise.f_rel) if noise.f_rel else 0.0)
        c = rng.normal(0.0, noise.c_px, 2) if noise.c_px else np.zeros(2)
        ds = rng.normal(0.0, noise.depth_scale) if noise.depth_scale else 0.0
        return Calibration(rotvec_to_mat(r), t, float(fs), c, float(ds))

    def summary(self):
        ang = float(np.degrees(np.linalg.norm(cv2.Rodrigues(self.R_err)[0])))
        return {"rot_deg": round(ang, 3), "t_mm": [round(float(v) * 1000, 3) for v in self.t_err],
                "f_scale": round(self.f_scale, 5), "c_off_px": [round(float(v), 3) for v in self.c_off],
                "depth_scale": round(self.depth_scale, 5)}


# ------------------------------------------------------------------ per-scan conditions
def sample_scan(noise, rng, shape, max_retries=2, force=None):
    """Draw the conditions of one scan: failure mode, occluder, smear, sheen spots and holes (static in a scan).
    force: None | 'dead' | 'blackout' | 'occlusion' | 'smear' | 'no_frame' (fault injection for tests/evaluation)."""
    H, W = shape
    sc = {"mode": "ok", "retries": 0, "occl": None, "smear": None, "sheen": [], "holes": None}
    # frame grabs: each attempt fails with p_no_frame; p_dead = the camera never answers in this scan
    if force == "dead" or (noise.p_dead and rng.random() < noise.p_dead):
        sc["mode"], sc["retries"] = "dead", max_retries
        return sc
    tries = 0
    while noise.p_no_frame and rng.random() < noise.p_no_frame and tries <= max_retries:
        tries += 1
    if force == "no_frame":
        tries = max(tries, 1)
    if tries > max_retries:
        sc["mode"], sc["retries"] = "dead", max_retries
        return sc
    sc["retries"] = tries
    if force == "blackout" or (noise.p_blackout and rng.random() < noise.p_blackout):
        sc["mode"] = "blackout"
        return sc
    if force == "occlusion" or (noise.p_occlusion and rng.random() < noise.p_occlusion):
        a = rng.uniform(*noise.occl_frac) if noise.p_occlusion else rng.uniform(0.1, 0.3)
        ar = rng.uniform(0.3, 1.0)                      # an elongated thing (cable, arm link) or a blob (hand)
        h = int(min(H, math.sqrt(a * H * W * ar)))
        w = int(min(W, a * H * W / max(h, 1)))
        y0 = int(rng.integers(0, max(1, H - h + 1)))
        side = int(rng.integers(0, 4))                  # it enters from an image border
        x0 = [0, W - w, int(rng.integers(0, max(1, W - w + 1))), int(rng.integers(0, max(1, W - w + 1)))][side]
        if side >= 2:
            y0 = 0 if side == 2 else H - h
        sc["occl"] = (y0, y0 + h, x0, x0 + w, float(rng.uniform(0.06, 0.15)))
        sc["mode"] = "occlusion"
    if force == "smear" or (noise.p_smear and rng.random() < noise.p_smear):
        a = rng.uniform(*noise.smear_frac) if noise.p_smear else rng.uniform(0.1, 0.3)
        r = math.sqrt(a * H * W / math.pi)
        sc["smear"] = (float(rng.uniform(0.2, 0.8) * W), float(rng.uniform(0.3, 0.9) * H), float(r))
        sc["mode"] = "smear" if sc["mode"] == "ok" else sc["mode"] + "+smear"
    if noise.sheen_p and rng.random() < noise.sheen_p:
        for _ in range(1 + int(noise.sheen_p >= 1.0 and rng.random() < 0.5)):
            sc["sheen"].append((float(rng.uniform(0.3, 0.7) * W), float(rng.uniform(0.45, 0.85) * H),
                                float(noise.sheen_px * rng.uniform(0.7, 1.3))))
    if noise.holes:
        f = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), noise.hole_px)
        f /= max(float(f.std()), 1e-9)
        sc["holes"] = f > np.quantile(f, 1.0 - noise.holes)
    return sc


def _disk(shape, cx, cy, r):
    v, u = np.ogrid[:shape[0], :shape[1]]
    return (u - cx) ** 2 + (v - cy) ** 2 < r * r


def corrupt(depth, noise, rng, scan, calib=None, dark=None, bed_roi=None):
    """One noisy frame from a true z-depth image (m). Returns float64 depth (m) with NaN = no depth.
    scan: sample_scan() output (static in a scan); dark: bool mask of IR-dark part pixels; bed_roi: bool mask of
    pixels that look at the bed (where a specular sheen can appear)."""
    H, W = depth.shape
    if scan["mode"] in ("dead", "blackout"):
        return np.full((H, W), np.nan)
    z = depth.astype(np.float64)
    if calib is not None and calib.depth_scale:
        z = z * (1.0 + calib.depth_scale)
    sig = noise.sigma_z_mm(z) / 1000.0
    if dark is not None and noise.dark_noise != 1.0:
        sig = np.where(dark, sig * noise.dark_noise, sig)
    smear = None
    if scan["smear"] is not None:
        smear = _disk((H, W), *scan["smear"])
        sig = np.where(smear, sig * 3.0, sig)
    zn = z.copy()
    if noise.axial_mm:
        n = rng.standard_normal((H, W))
        a = noise.axial_corr_frac
        if a > 0:
            c = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), noise.axial_corr_px)
            c = c.astype(np.float64) / max(float(c.std()), 1e-9)
            n = math.sqrt(1.0 - a) * n + math.sqrt(a) * c
        zn = zn + sig * n
    if smear is not None:                                   # a smeared lens blurs depth across edges
        zb = cv2.GaussianBlur(z.astype(np.float32), (0, 0), 6.0).astype(np.float64)
        zn = np.where(smear, zb + (zn - z), zn)
    if noise.flying_frac:                                   # flying pixels between foreground and background
        k = np.ones((3, 3), np.uint8)
        z32 = z.astype(np.float32)
        lo, hi = cv2.erode(z32, k).astype(np.float64), cv2.dilate(z32, k).astype(np.float64)
        fly = ((hi - lo) > noise.edge_mm / 1000.0) & (rng.random((H, W)) < noise.flying_frac)
        zn[fly] = (lo + rng.random((H, W)) * (hi - lo))[fly]
    if noise.lateral_px:                                    # where each sample lands: nearest-neighbour jitter
        v, u = np.mgrid[0:H, 0:W].astype(np.float32)
        mx = u + rng.normal(0, noise.lateral_px, (H, W)).astype(np.float32)
        my = v + rng.normal(0, noise.lateral_px, (H, W)).astype(np.float32)
        zn = cv2.remap(zn.astype(np.float32), mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
        zn = zn.astype(np.float64)
    if noise.quant_mm:
        q = noise.quant_mm / 1000.0
        zn = np.round(zn / q) * q
    bad = np.zeros((H, W), bool)
    if noise.dropout:
        bad |= rng.random((H, W)) < noise.dropout
    if scan["holes"] is not None:
        bad |= scan["holes"]
    for (cx, cy, r) in scan["sheen"]:
        m = _disk((H, W), cx, cy, r)
        if bed_roi is not None:
            m &= bed_roi
        bad |= m & (rng.random((H, W)) < noise.sheen_drop)
    if dark is not None and noise.dark_drop:
        bad |= dark & (rng.random((H, W)) < noise.dark_drop)
    if smear is not None:
        bad |= smear & (rng.random((H, W)) < 0.3)
    zn[bad] = np.nan
    if scan["occl"] is not None:                            # an object a few cm from the lens
        y0, y1, x0, x1, zo = scan["occl"]
        blk = zn[y0:y1, x0:x1]
        blk[:] = zo + rng.normal(0, 0.002, blk.shape)
        blk[rng.random(blk.shape) < 0.5] = np.nan           # too close for stereo (below min-Z): mostly holes
    return zn
