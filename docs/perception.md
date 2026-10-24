# Perception under real sensor errors

The robot never trusts the simulator's ground truth. It knows what *should* be on the bed from the slicer
(part, position, yaw, height) and uses a chamber depth camera only to **refine and verify** that prior:
where exactly each part ended up, whether one is missing, and whether the bed is empty after unloading.

With a perfect rendered depth image this is easy (0.3 mm mean centre error). This page is about what happens
with a real camera: noise, holes, flying pixels, a camera that is a few millimetres and a fraction of a degree
off its calibration, a smeared lens, a cable in the view, or no frame at all.

Code: `perceive/noise.py` (error model), `perceive/verify.py` (camera, corrections, estimator),
`scripts/eval_perception.py` (Monte Carlo), `tests/test_perception.py`.

## The sensor (assumption)

Stock enclosed printers ship an **RGB** chamber camera only. **Assumption:** a short-range active-stereo depth camera
(Intel RealSense D405 / D435 type, or an Orbbec equivalent) is added on the chamber ceiling, about 0.33 m above the
bed at the unload height, looking down at the bed from the front-left corner. The simulated camera renders
640 x 480 depth with an 86 x 70 deg field of view (focal length 343 px); the bed is seen at 0.33-0.55 m range.
That range is inside the D405's stated ideal range (7-50 cm, with < 2 % depth accuracy at 50 cm) and above the
D435's minimum range at 848 x 480 (about 17 cm) [1, 2].

Not modelled: chamber temperature (a hot chamber is outside the operating range of some depth cameras; mount it
outside a glass lid or use one rated for it) and the glass lid itself (reflections) if the camera looks through it.

## Error model (`perceive/noise.py`)

Every scan renders the true depth once and then corrupts `frames` independent copies of it.

| Effect | Model | realistic | harsh | Basis |
|---|---|---|---|---|
| Axial (depth) noise | sigma_z(z) = a * (z / 0.4 m)^2, half of the variance spatially correlated (stereo matching window) | a = 0.8 mm | 1.6 mm | stereo triangulation: sigma_z = z^2 * sigma_d / (f * b); D435-type, b = 50 mm at 848 px (f ~ 424 px), subpixel sigma_d = 0.1 px gives 0.75 mm at 0.4 m [1, 3]; harsh = 0.2 px, Intel's re-calibration threshold [1]; quadratic growth with range [4, 6] |
| Lateral noise | each sample lands at a jittered pixel | 0.3 px | 0.6 px | edge-position noise [4] (assumption on size) |
| Quantization | depth unit | 0.1 mm | 1 mm | D4xx depth unit is configurable (1 mm default) |
| Depth scale | persistent relative scale error | 0.2 % | 0.8 % | D405 accuracy < 2 % at 50 cm [2]; realistic assumes an on-site tare (self-calibration [3]) |
| Flying pixels | share of depth-edge pixels given a depth between foreground and background | 30 % | 60 % | well-known stereo/ToF artefact (assumption on share) |
| Random dropouts | iid invalid pixels | 1 % | 4 % | assumption |
| Structured holes | static blobs (texture-poor patches) | 0.5 % of image | 3 % | assumption |
| Specular sheen on the PEI sheet | spot (r ~ 28 / 45 px) of dropouts where the bare sheet is seen | p = 0.5, 60 % | p = 1, 80 % | glossy surfaces degrade active stereo [1, 5] |
| IR-dark parts (black filament) | per-part, lower fill and more noise | 20 % of parts, 30 % dropout, x1.5 noise | 40 %, 60 %, x2.5 | material dependence [5] (assumption on size) |
| Temporal noise | fresh noise every frame; a scan fuses 4 frames | 4 frames | 4 frames | time averaging that ignores holes is Intel's recommendation [1] |
| Extrinsic calibration | Gaussian camera pose error, persistent per camera | 1 mm, 0.25 deg per axis | 3 mm, 1 deg | assumption: checkerboard calibration vs a bumped camera |
| Intrinsic calibration | focal length, principal point | 0.3 %, 1 px | 1 %, 3 px | assumption |
| Frame grab fails (retried twice) | per attempt | 1 % | 5 % | assumption |
| Camera dead for the scan | | 0.2 % | 2 % | assumption |
| Blackout (frame without depth) | | 0.5 % | 2 % | assumption |
| Occluder near the lens (cable, arm, hand) | 10-30 % of the image at 6-15 cm | 2 % | 10 % | assumption |
| Smeared / fogged lens | 10-30 % of the image blurred, 3x noise, 30 % dropouts | 2 % | 10 % | assumption |

`ideal` is no error at all: exactly the original perfect-render behaviour, and the default everywhere.
`perceive.noise.scaled(noise, k)` multiplies every magnitude and probability by k (sensitivity sweep below).

## Correction pipeline (`perceive/verify.py`)

`BedCamera(world, printer, noise="realistic", rng=np.random.default_rng(seed))` returns a point cloud that has
already been corrected; `estimate()` / `bed_empty()` read the scan's metadata from it.

1. **Temporal fusion.** Median of the 4 frames per pixel (a pixel needs 2 valid frames).
2. **Flying-pixel / speckle filter.** A depth pixel survives only if at least 2 of its 8 neighbours agree with it
   (within 4 sigma, >= 2 mm). Flying pixels sit between two surfaces and agree with neither. (A 3x3 median filter
   was tried first: it eats the edges of thin features such as the 6 mm upright of the bracket.)
3. **Bed plane (RANSAC).** The bed is the dominant plane. Fitting it and rotating it level about the *believed
   camera centre*, then removing its height offset, cancels the tilt, the height error and the lateral shift that
   a rotation error of the camera puts on the bed (3 of the 6 pose degrees of freedom).
4. **Plate outline (the fiducial).** The remaining error is a 2-D rigid motion (x, y, yaw) in the bed plane. It is
   measured from the build plate's own edges: each of the 4 edges is a depth step, the outermost bed-level point in
   each 4 mm bin traces it, and a robust line fit gives its position and angle. Opposite edges give the centre with
   their read-out biases cancelling; four 256 mm edges give the yaw to a few hundredths of a degree.
   Safeguards: bins where the step is blurred (points on a ramp just outside the edge: a smeared lens) are skipped;
   opposite edges must agree with the plate size within 3 mm; a shift beyond 4x the calibration budget is not
   trusted. The per-edge read-out bias of this camera geometry (a pixel's footprint, up to 0.4 mm on the far edges)
   is measured once on the perfect render (`--edge-bias`), like an install-time calibration.

   *Why not the purge tab?* It is printed by the start G-code at a known spot, so it is in printer coordinates,
   which is good, but it is 10 x 20 mm (its own yaw is only good to about a degree), it is the first object the
   robot removes, and it is itself one of the things being measured. *Printed corner marks* would cost filament and
   time on every plate and must themselves be removed. The plate edges are always there, 256 mm long, and located
   on the bed by its alignment pins (assumption: their offset to printer coordinates is calibrated once). The tab is
   still a good cross-check for translation.
5. **Robust top-face estimation.** Height band from the noise model (3 sigma at the part's range, >= 1.5 mm);
   a radial gate around the median; boxes: minimum-area rectangle (centre + yaw); cylinders: centre from trimmed
   extents in 8 directions (the plain mean is pulled toward the camera by side-wall points inside the band);
   L bracket: yaw from the base face vs the upright (the original estimator kept the slicer yaw there).
   A nudged part is searched for wider, but never inside another object's footprint (the original estimator could
   "find" a missing part on its neighbour).
6. **Bayesian fusion with the slicer prior.** Prior N(mu_slicer, Sigma_p), Sigma_p = (2 mm)^2 I (parts shift a
   little while printing and cooling; the simulation spawns them with 1.5 mm noise). Camera N(mu_cam, Sigma_c) with

   sigma_c^2 = floor^2 + sigma_pt^2 / n_eff + sigma_cal^2 + (scale_err * r)^2 + sigma_cov^2

   floor = 0.25 mm (estimator floor on perfect depth, measured); sigma_pt = per-point lateral + projected depth noise
   at the part's range and viewing angle; n_eff = top-face points / (1 + correlated share * pi * corr_px^2);
   sigma_cal = 0.2 mm residual after the plane + outline fit (measured), or the full calibration budget when the
   outline was not measured; scale_err * r = focal / depth-scale error times the distance from the bed centre (not
   removable by a rigid correction); sigma_cov grows when the top face is only partly seen.
   Posterior: K = Sigma_p (Sigma_p + Sigma_c)^-1, mu = mu_p + K (mu_c - mu_p), Sigma = (I - K) Sigma_p.
   **Gate:** if the Mahalanobis distance^2 exceeds 13.8 (chi-square, 2 dof, 99.9 %) the part has moved (a
   neighbouring pick nudged it): the prior is dropped and the camera alone is used. Yaw is fused the same way
   (camera yaw sigma fitted to measured errors).
   `estimate()` returns `sigma_mm` = sqrt(trace(cov)) (RMS radial error), `cov_mm2`, `yaw_sigma_deg`.

## Failure handling

| Situation | What `estimate()` says | `bed_empty()` |
|---|---|---|
| Scan OK | camera estimate fused with the prior, `quality` "ok" | leftover test |
| Scan degraded (fill < 75 %, something at the lens, blurred edges, outline not measured, noisy plane) | same, `quality` "degraded" | leftover test; `False` if an occluder hides > 600 mm^2 of bed |
| Part region not seen (holes, occluder, shadow) | slicer prior, covariance x4, `observed` False, "degraded", `found` True | |
| Bare bed seen where the part should be | `found` False (missing: the rake pass takes over) | |
| Something raised but no top face, good scan | `found` False (tipped / wrong part) | |
| Same on a degraded scan (e.g. smeared lens) | prior, "degraded" | |
| No frame / blackout / dead camera | slicer prior, covariance x4, `quality` "failed", never raises | `False` (cannot verify) |

Scan quality is decided from the data only; the simulated fault mode is never looked at.
`bed_empty` on a noisy camera: a leftover is a connected raised component of >= 12 points whose top is >= 3 mm,
where "raised" = above 3.5 sigma of the point's depth noise (the noise model at its range or the bed noise measured
locally in 48 mm cells, whichever is larger) and isolated specks are dropped.

## Results

Monte Carlo, `scripts/eval_perception.py --n 150 --n-faults 40 --sweep --procs 3` (seed 0, 205 s on 3 cores).
Every configuration scans the **same** scenes: a random plate of 3-7 parts (packer + mirror, as in the unload
evaluation) spawned with 1.5 mm / 2 deg prior noise; in 25 % of the scenes one listed part is missing from the bed
and in 25 % one part is nudged 5-12 mm (as a neighbouring pick does). Every trial draws a new camera calibration
error. Each trial also scans the empty bed and the bed with one leftover lump 5, 8 or 15 mm proud of the plate.
"no corrections" = one raw frame, no plane / outline correction, the original estimator. Raw records:
`runs/perception/*.json`.

![perception results](perception.png)

| config | trials | found | centre mean / p95 (mm) | yaw mean (deg) | NEES (2 = honest) | in 95 % ellipse | missing detected | false empty | false not empty | ms / scan |
|---|---|---|---|---|---|---|---|---|---|---|
| ideal | 150 | 100 % | 0.33 / 1.10 | 0.42 | 0.97 | 97 % | 59 % | 0 % | 0 % | 19 |
| ideal+pipeline | 150 | 100 % | 0.20 / 0.54 | 0.09 | 0.79 | 100 % | 100 % | 0 % | 0 % | 63 |
| realistic/naive | 150 | 89 % | 5.07 / 11.33 | 1.54 | 19.36 | 91 % | 56 % | 1 % | 69 % | 59 |
| realistic | 150 | 100 % | 0.51 / 1.04 | 0.47 | 1.03 | 99 % | 100 % | 0 % | 3 % | 216 |
| harsh/naive | 150 | 60 % | 17.73 / 56.11 | 2.01 | 8.24 | 75 % | 56 % | 9 % | 78 % | 65 |
| harsh | 150 | 99 % | 1.80 / 4.33 | 1.43 | 2.38 | 91 % | 76 % | 2 % | 24 % | 234 |
| realistic+occlusion | 40 | 100 % | 0.92 / 2.52 | 0.90 | 1.67 | 95 % | 57 % | 0 % | 78 % | 249 |
| realistic+smear | 40 | 100 % | 1.24 / 3.49 | 1.10 | 1.56 | 95 % | 100 % | 8 % | 52 % | 251 |
| realistic+dead | 40 | 100 % | 1.75 / 3.96 | 1.34 | 0.41 | 99 % | 0 % | 0 % | 100 % | 0 |
| realistic+blackout | 40 | 100 % | 1.75 / 3.96 | 1.34 | 0.41 | 99 % | 0 % | 0 % | 100 % | 0 |
| realistic x0.5 | 60 | 100 % | 0.30 / 0.65 | 0.25 | 0.89 | 99 % | 100 % | 0 % | 2 % | 211 |
| realistic x1 | 60 | 100 % | 0.48 / 0.90 | 0.47 | 1.01 | 99 % | 100 % | 0 % | 3 % | 216 |
| realistic x1.5 | 60 | 100 % | 0.78 / 2.28 | 0.78 | 1.54 | 95 % | 100 % | 0 % | 8 % | 230 |
| realistic x2 | 60 | 100 % | 1.30 / 3.49 | 1.24 | 2.77 | 89 % | 100 % | 0 % | 8 % | 248 |
| realistic x3 | 60 | 97 % | 1.74 / 3.78 | 1.53 | 4.14 | 93 % | 82 % | 42 % | 10 % | 241 |

Columns: *found* = parts reported present among those on the bed (for a failed camera every part is assumed present
at its prior: that is the fallback, not a detection). *NEES* = mean of e' Sigma^-1 e over found parts (2 for an
honest 2-D covariance; below 2 = conservative). *in 95 % ellipse* = share of errors inside the reported 95 %
ellipse. *missing detected* = parts listed by the slicer but absent from the bed that were reported missing.
*false empty* = a bed with a 5-15 mm leftover reported empty; *false not empty* = an empty bed not confirmed empty
(including "cannot verify": failed scans and occluded views). *ms / scan* = wall time of scan + estimate on one
Apple-silicon core (4 frames of 640 x 480).

What the numbers say:
- **Corrections are not optional.** With the realistic error model and no corrections the centre error is
  5.1 mm mean / 11.3 mm p95, 11 % of parts are not found and 69 % of empty beds look occupied (the bed is tilted
  by the calibration error and the raw single-frame noise crosses the 1.5 mm raised threshold). With the pipeline:
  **0.51 mm mean / 1.04 mm p95, 100 % found, 100 % of missing parts detected, no leftover missed, 3 % of empty beds
  not confirmed** (3 of those 4 scans failed and 1 was degraded: the camera says "cannot verify", not "occupied").
- **The covariance is honest.** Realistic: NEES 1.03 and 99 % of errors inside the 95 % ellipse (slightly
  conservative); harsh: NEES 2.38, 91 % inside (slightly optimistic in the tail).
- **The robust estimator also helps on perfect depth.** On the perfect render, 0.20 / 0.54 mm instead of
  0.33 / 1.10 mm, yaw 0.09 deg instead of 0.42 deg (the bracket's yaw is now measured), and missing parts are
  detected 100 % instead of 59 % (the original wide search "found" a missing part on its neighbour). The default
  ideal camera still runs the original code unchanged.
- **Harsh** (2x noise, 3-4x calibration error, 5-10x failure rates) still gives 1.8 mm mean / 4.3 mm p95 with 99 % found,
  but 24 % of empty beds are not confirmed and 24 % of missing parts are assumed present (7 of those 8 because their
  footprint was not seen, so the prior is kept with an inflated covariance and "degraded" / "failed" quality).
- **Faults never crash and never pass silently.** An occluder or a smeared lens is detected from the data (near
  depth in the bed region, blurred plate edges) and the scan is "degraded"; a dead camera or a blackout gives the
  slicer prior with a 4x covariance and "failed". The weakest case is the smeared lens: 8 % of leftovers missed and
  52 % of empty beds not confirmed - it needs cleaning, which is what the quality flag is for.
- **Sensitivity.** Scaling every realistic error by k: the p95 centre error stays under 1 mm up to k = 1 and under
  2.3 mm at k = 1.5; at k = 3 leftover detection breaks down (42 % of leftovers missed) - the noise is then larger
  than a 5 mm lump at the far corner.

## Using it in the cell

- `plan/unload.py` builds one camera per printer in `bed_camera()`; switching the whole cell to a noisy camera is
  `BedCamera(w, i, noise=CAMERA_NOISE, rng=np.random.default_rng((seed, i)))` there.
- Anything that thresholds raised points itself (the rake's blob finder) should use
  `perceive.verify.raised_points(points)`: identical to `points[points[:, 2] > RAISED_MM]` for the ideal camera,
  noise-robust otherwise. Measured on empty beds: the rake's blob finder fed the corrected cloud as-is reports a
  blob in 25 of 25 harsh scans; fed `raised_points()` in 0 of 25 (realistic: 1 of 30 either way, a smeared lens).
- A "failed" scan is usually transient (a dropped frame, a cable in the view): scan again before acting on it.
  `sigma_mm` can widen the gripper pre-shape for parts the camera saw poorly.

## Limitations

- The noise is applied to a rendered depth image; real stereo artefacts depend on texture and lighting in ways a
  depth-image model only approximates (holes here are content-independent, which is pessimistic for leftovers).
- Calibration-error sizes, failure probabilities and the dark-part share are assumptions, not measurements. The
  sensitivity sweep shows how the result degrades if they are worse.
- The camera geometry is one fixed mount. A camera closer to the bed or a higher resolution would shrink sigma_z
  (z^2) and the pixel footprint.
- The smeared-lens case is detected (blurred plate edges) but a smeared lens still hides small leftovers: it needs a
  cleaning alert, not more software.
- The slicer prior sigma (2 mm) matches the simulation's spawn noise (1.5 mm); a printer that shifts parts more
  needs a larger prior or the fusion trusts it too much.

## Reproduce

```bash
PYTHONPATH=. .venv/bin/python scripts/eval_perception.py --n 150 --n-faults 40 --sweep --procs 3   # -> runs/perception/
PYTHONPATH=. .venv/bin/python scripts/eval_perception.py --edge-bias          # outline read-out bias (EDGE_BIAS_MM)
PYTHONPATH=. .venv/bin/python scripts/test_perception.py 10 realistic         # per-part print-out
PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_perception.py
```

## Sources

1. Intel RealSense, "Tuning depth cameras for best performance" (subpixel RMS < 0.1 px for a well-calibrated camera,
   approaching 0.05 on textured targets, re-calibrate above ~0.2; min-Z scales with resolution, D435 at 848 x 480
   ~16.8 cm; time averaging that ignores holes; glossy surfaces). https://dev.realsenseai.com/?p=41439
2. Intel RealSense D405 product page (ideal range 7-50 cm, depth accuracy < 2 % at 50 cm, 87 x 58 deg).
   https://store.intelrealsense.com/buy-intel-realsense-depth-camera-d405.html
3. Intel RealSense, "Self-Calibration for D400 Series Depth Cameras" (RMS depth-error metric from focal length,
   baseline, distance; D435 baseline ~50 mm, HFOV ~90 deg; on-chip tare). https://dev.realsenseai.com/?p=41425
4. C. V. Nguyen, S. Izadi, D. Lovell, "Modeling Kinect Sensor Noise for Improved 3D Reconstruction and Tracking",
   3DIMPVT 2012 (axial noise growing quadratically with range, lateral noise linearly).
5. G. Halmetschlager-Funek, M. Suchi, M. Kampel, M. Vincze, "An Empirical Evaluation of Ten Depth Cameras",
   IEEE Robotics & Automation Magazine, doi:10.1109/MRA.2018.2852795 (bias, precision, lateral noise, materials,
   lighting).
6. "Noise Analysis and Modeling of the PMD Flexx2 Depth Camera for Robotic Applications", arXiv:2412.15040
   (axial noise vs distance and incidence angle, compared with the D435).
7. "Empirical Comparison of Four Stereoscopic Depth Sensing Cameras for Robotics Applications", arXiv:2501.07421
   (D435 best on table-top scenes up to 1 m, error under 1 cm).
