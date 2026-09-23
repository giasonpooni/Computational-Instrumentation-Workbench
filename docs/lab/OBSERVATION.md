# Instrument observation experiments (T045–T059)

Section 4 of the computational-experimentalist queue asks what an instrument
observes, how that observation relates to surface geometry, and how
calibration, distortion, quantization, timing, loss and staleness corrupt it.
Every instrument, surface, marker and noise value here is **declared and
synthetic**. The tasks establish properties of the models and of this code.
They do not establish the accuracy, calibration validity or safety of any
physical instrument. Each task records that boundary as a `not_established`
finding in its physical, calibration, sensor-performance or authority domain.

Code: `src/ciw/lab/observation.py` (task registrations) with helpers
`observation_modes.py` (typed records, refusals, mappings, retention),
`observation_chord.py` (chord expansion, sympy check, integrated geodesics),
`observation_camera.py` (pinhole stereo rig, distortion, triangulation) and
`observation_signals.py` (encoder, IMU, timing, Kalman/RTS, statistics).
Tests: `tests/test_lab_observation.py`.

```
python -m ciw lab run T045 T046 T047 T048 T049 T050 T051 T052 T053 T054 T055 T056 T057 T058 T059 --output-dir out/lab
python -m ciw lab report T046 --retained out/lab
python -m pytest -q tests/test_lab_observation.py
```

The section runs in about 6 s and its tests in about 7 s on one core. sympy
(T046, T047) and scipy (T057) are optional. Without them the derivation
findings are labelled `analytic` instead of `independently_verified`, and the
Riccati fixed-point and chi-square-quantile findings of T057 are
`numerically_verified` instead of `independently_verified`.

## Observation modes

Every observation is an `Observation` record: mode, value, unit, frame id,
clock id, clock basis, epoch, time, calibration reference, sequence number,
raw-data reference, optional latency and surface model, and the digests of the
records it was derived from, plus the identities of the mappings applied. The
mode declares what the record may contain:

| Mode | Quantity | Unit | Components | Frame kind | Clock basis | Geometry | Cannot observe |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `intrinsic_geodesic_distance` | arc length of a declared geodesic along the surface | m | 1 | `surface_chart` | acquisition | intrinsic | chord or any embedding quantity; the path unless declared; the part's pose |
| `camera_chord_distance` | straight-line distance between two triangulated markers | m | 1 | `camera_rig` | acquisition | extrinsic | intrinsic distance without a surface model; the surface between markers; occluded markers |
| `reconstructed_surface_distance` | geodesic length on a declared or reconstructed surface model | m | 1 | `reconstruction` | acquisition | intrinsic (model) | the true surface where it departs from the model; sub-resolution curvature; the path actually followed |
| `encoder_displacement` | actuator axis displacement | m | 1 | `axis` | acquisition | none | load position without kinematics; surface geometry; backlash state without direction history |
| `tracker_measurement` | marker position | m | 3 | `tracker` | arrival | extrinsic | distance along the surface; acquisition time without declared latency; body orientation |
| `image_residual` | reprojection residual | px | 2 | `image` | acquisition | none | depth along the ray; metric scale without calibration; surface geometry |
| `imu_orientation` | body orientation as a rotation vector | rad | 3 | `body` | acquisition | none | absolute heading without a reference; position; surface geometry |

Noise-model parameters in the registry (for example 0.25 px pixel noise,
20 µm backlash, 1 mrad/s gyro bias) are declared placeholders
(`"status": "declared_placeholder"`), not instrument characteristics.

Refusal codes (`ObservationRefusal.code`):

| Code | Raised when |
| --- | --- |
| `missing_frame`, `missing_clock`, `missing_epoch`, `missing_calibration`, `missing_clock_basis`, `missing_time` | a record lacks that reference (`calibration_ref` may be `not_applied`, but must be stated) |
| `frame_kind_mismatch`, `clock_basis_mismatch`, `unit_mismatch`, `invalid_value` | an unmapped record disagrees with its mode |
| `surface_model_required` | a surface distance, or a chord-to-surface conversion, has no declared surface model |
| `mode_substitution` | one mode is used where another is required (all 42 ordered pairs are refused) |
| `frame_mismatch`, `clock_mismatch`, `epoch_mismatch` | two records are combined across frames, clocks or epochs without a declared mapping |
| `mapping_not_applicable` | a declared mapping does not start from the record's frame or clock |
| `missing_latency`, `future_observation`, `stale_observation` | an age cannot be computed, is negative, or exceeds the validity limit |
| `zero_filled_missing`, `unbacked_value`, `sequence_gap` | a stream fills gaps with fabricated values or loses sequence positions |
| `not_retained`, `not_admitted`, `admission_digest_mismatch` | state is updated from evidence that was not retained, not admitted, or altered after admission |

A camera chord becomes a distance along the surface only through
`chord_to_surface_distance(record, surface_model)`. The result has mode
`reconstructed_surface_distance`. It records the chord's digest in
`derived_from` and the model's identity in `mappings`. It is still refused
where an `intrinsic_geodesic_distance` is required, because it was derived
through a model and not observed intrinsically.

## Chord versus geodesic correction

Let γ(s) be a unit-speed curve with Frenet frame (T, N, B), curvature κ(s) and
torsion τ(s): T' = κN, N' = −κT + τB, B' = −τN. Differentiating γ' = T repeatedly:

    γ''  = κ N
    γ''' = −κ² T + κ' N + κτ B
    γ⁽⁴⁾·T = −3κκ',   γ''·γ''' = κκ'

so, with Δ(s) = γ(s) − γ(0) = Σ γ⁽ⁿ⁾(0) sⁿ/n!,

    |Δ|² = s² − κ² s⁴/12 − κκ' s⁵/12 + O(s⁶)
    c = |Δ| = s − κ² s³/24 − κκ' s⁴/24 + c₅ s⁵ + O(s⁶)

where everything is evaluated at s = 0. For constant κ and τ,
c₅ = κ⁴/1920 + κ²τ²/720. When τ = 0 as well (a circle) the chord is exactly
c = 2 sin(κs/2)/κ = s − κ²s³/24 + κ⁴s⁵/1920 − … Along a geodesic of a
surface, the geodesic curvature vanishes, so the space curvature is the
absolute normal curvature κ = |II(T, T)|.

sympy re-derives the series from the Frenet–Serret recursion with polynomial
κ(s), τ(s). It gives c₅ = (3κ⁴ + 8κ²τ² − 72κκ'' − 64κ'²)/5760 in general.
Its coefficients agree exactly with the hand derivation at rational sample
points. That agreement is the `independent_check` of T046.

Counterexamples found by the task:

* **The remainder is not O(s⁵) in general.** With κ taken at the start point,
  the s⁴ term −κκ's⁴/24 survives whenever κ' ≠ 0. On a torus geodesic
  (R = 2, r = 1, start (0, π/4), heading 0.6 rad), the fitted s⁴ coefficient
  of s − c − κ₀²s³/24 matches κ₀κ₀'/24 to 5e-7 relative. Taking κ at the arc
  midpoint cancels the term by symmetry.
* **Constant curvature is not enough for the circle formula.** A helix on a
  cylinder has constant κ and τ ≠ 0. Its chord differs from 2 sin(κs/2)/κ by
  κ²τ²s⁵/720 + O(s⁷).

## Cylinder chord coefficient

On a cylinder of radius R, the geodesic at angle α from the circumferential
direction is the helix X(s) = (R cos(s cos α/R), R sin(s cos α/R), s sin α),
with κ = cos²α/R and τ = sin α cos α/R. Its exact chord is

    c² = (2R sin(s cos α/(2R)))² + (s sin α)²
       = s² − cos⁴α s⁴/(12R²) + cos⁶α s⁶/(360R⁴) + O(s⁸)

so s − c = cos⁴α s³/(24R²) + O(s⁵), which is κ²/24 with κ = cos²α/R. The
correction is largest for circumferential paths (α = 0, coefficient
1/(24R²)) and exactly zero along the axial rulings (α = 90°). This holds
although the cylinder's Gaussian curvature is zero everywhere. The chord
correction depends on the extrinsic normal curvature in the path direction,
not on intrinsic curvature. T047 records both facts as counterexamples: to
"flat surfaces need no correction" and to "the correction depends on the
surface alone".

## Experiment designs

Every task follows hypothesis → prediction → synthetic protocol →
implementation → execution → independent comparison → uncertainty and
provenance → regression test. Monte Carlo comparisons use seeded PCG64
generators and a two-sided 99.9 % bound (|z| ≤ 3.29).

### T045 Typed observation modes
Validate the registry declarations. Strip each reference from a valid record
(9 refusals) and attempt all 42 ordered mode substitutions. Convert chords to
arc lengths with and without a declared cylinder model.

### T046 Chord-versus-geodesic correction
Hand derivation and sympy (exact rational comparison). Sphere geodesics
integrated by `ciw.lab.jacobi` (RK4, R = 0.5, 1, 2) are checked against
2R sin(s/2R) and a fitted s³ coefficient. Then the torus and helix
counterexamples above.

### T047 Cylinder coefficient
sympy series of the exact chord. Least-squares fits of (s − c)/s³ = a + bs² + ds⁴
at seven angles and two radii. Geodesics integrated on
`ciw.lab.surfaces.Cylinder`. Rulings against circumferential paths.

### T048 Synthetic camera measurements
Declared pinhole stereo pair: f = 2400 px, 2048 × 1536 px, baseline 0.2 m,
converging at 0.6 m. Markers every 15 mm along three helices (α = 0°, 45°,
90°) on a cylinder with R = 0.1 m, all visible in both images. Noise-free DLT
and ray-midpoint triangulation are compared with truth. The chord-for-geodesic
substitution bias (up to 7.07 mm at s = 0.12 m, circumferential) is compared
with the model conversion and with 0.25 px pixel noise (chord RMS about
0.19 mm, `synthetic`).

### T049 Calibration perturbations
Pixels from the true rig are triangulated with a believed rig. The believed rig
has a focal error on both cameras, a right-camera principal-point error and a
right-camera rotation about its centre. The central-difference Jacobian is
compared with closed forms on a rectified rig (∂c/∂f = ΔZ²/(cf) and
∂Z/∂c_x = −Z²/(bf)) and checked by step halving. First-order predictions are
compared with direct recomputation over five scales; the residual slope is 2.
Errors that change horizontal disparity dominate: for the 0.12 m chord, 1 mrad
of right-camera yaw moves it by 0.36 mm and 1 px of horizontal principal point
by 0.15 mm, against about 2 µm for 1 px of vertical principal point.
Counterexample: a common focal error leaves same-depth chords unchanged (the
rectified-rig map is X_b = X, Z_b = Z f_b/f), so it is not a uniform scale.

### T050 Perspective and lens distortion
Brown–Conrady radial and tangential distortion: the displacement is exactly
f(k₁r³ + k₂r⁵) (slope 3 in r, 1 in k₁). The chord bias from uncorrected
distortion follows J_pix·δpix to first order (0.3 % residual at |k₁| ≤ 0.1),
is odd in k₁ and grows with image radius. Undistorting with the true model
removes it. Counterexample: for k₁ = −1.2 the radial map folds at
r = 1/√(−3k₁) = 0.527, inside the image corner (0.533). Two radii then share
one distorted radius, and fixed-point undistortion misses by about 22 px.

### T051 Quantization and pixel noise
e = round(x + n) − x has variance σ² + 1/12 when the sub-pixel phase is
uniform, because the rounding error is then uniform and independent of n. This
is checked at five values of σ. Counterexample: an integer-aligned coordinate
with σ = 0.1 px has essentially no rounding error. Chord standard deviations
follow √((σ² + 1/12) Σ J²). This is checked against 4000 seeded stereo trials
with a random grid phase, using 99.9 % intervals.

### T052 Encoder bias, scale and backlash
Reading = (1 + s)·play_b(x) + β + noise. The play error lies in [0, b] and
changes only during take-up after a reversal. A fit on engaged samples with a
direction term recovers (1 + s, β, (1 + s)b). Counterexample: fitting without
the direction term biases β by about b times the falling fraction (z ≈ 40).

### T053 IMU drift and orientation noise
Single-axis heading error has mean bt, variance N²t and MSE N²t + (bt)²,
checked with 2000 runs. Strapdown SO(3): the bias error follows
e_{k+1} = exp(−[ω]dt)e_k + J_r(ωdt)b dt. Counterexample: on a body rotating
at 1 rev/s, a transverse bias gives an error bounded by 2|b⊥|/|ω| instead of
|b⊥|t. Angle random walk stays isotropic (3N²t) under rotation.

### T054 Asynchronous timestamps
Sensor B (30 Hz) is interpolated to sensor A's times (100 Hz). A clock offset
δ gives e = −vδ + O(δ²), slope 2, below the curvature bound. After linear
interpolation the error is exactly −Sδ, where S is the interpolant slope; its
regression on −vδ is 1 − O((ωh)²). Jitter adds S²σ_j²((1 − w)² + w²). A
declared `ClockMapping` removes the offset exactly. Combining the clocks
without a mapping is refused.

### T055 Dropped observations
Drops stay `None` at their sequence positions. Zero-filled streams and values
without raw references are refused. Zero filling biases a mean by −pμ;
explicit gaps are unbiased with variance σ²E[1/N]. Counterexample: value-only
detection finds the fills when the signal is far from zero and none near
zero, so provenance is required. Sample-and-hold tracking of a random walk has
MSE qE[age] + r. At the same 20 % drop rate, Gilbert–Elliott bursts
(E[age] = 1) give about 3.6 times the Bernoulli error (E[age] = 0.25).

### T056 Stale-state observations
Age = t_use − (t_arrival − latency). Flags equal age > limit exactly. The
error is v·age for constant velocity, with an O(a²) residual below
A ω² a²/2 otherwise. Missing latency and future records are refused.
Counterexample: an observation that arrived 5 ms ago with 35 ms latency is
stale under a 30 ms limit.

### T057 Raw, filtered and smoothed estimates
Constant-velocity Kalman filter (Joseph form) and Rauch–Tung–Striebel smoother
on a seeded 300-run ensemble. P_filt − P_smooth is positive semidefinite at
every step. It is only rank one at step N − 2, so strict reduction is tested
on its trace. Position RMSE orders smoothed ≤ filtered ≤ raw, and the
ensemble-average NEES lies inside the 95 % χ²(2N)/N bounds at about 97 % of
steps. The bounds use Wilson–Hilferty quantiles. They are always checked
against a series evaluation of the chi-square CDF (regularized incomplete
gamma function), and against `scipy.stats.chi2` when scipy is available. The Riccati fixed point is compared with
`scipy.linalg.solve_discrete_are`. Counterexample: the smoothed error is larger
than the filtered error at about 29 % of individual samples. The ordering is
an ensemble property.

### T058 Frame and clock basis tracking
Combining records across frame, clock, epoch or time basis is refused.
Declared rigid frame mappings (quarter turn with dyadic translation) match
hand-written formulas exactly, and general rotations round-trip to 4e-16.
Chords are invariant under them. Clock mappings (arrival → acquisition by the
declared latency, then clock → clock) are exact on dyadic times. Mappings for
another frame, frame kind or epoch are refused.

### T059 Retained without admission
Records of all seven modes are retained with `retention: retained` and
`state_admission: not_performed`, and the state digest does not change.
Updates from unadmitted records are refused. Admission validates the record
and binds its digest. An admitted scalar update is exact (prior 1.0/0.5,
observation 2.0/0.5 → 1.5/0.25). Tampered, unretained and mode-substituted
admissions are refused.

## What remains physically unverified

None of the following is established by this section:

* Real sensor performance: pixel noise, marker localization, encoder
  resolution and backlash, gyro bias stability and random walk, tracker noise,
  link loss and latency distributions.
* Calibration validity: whether any real rig's intrinsics, extrinsics,
  distortion, clock offsets and frame mappings match the declared ones.
* Physical geometry: whether a real part is a cylinder of known radius and
  axis, whether markers lie on one geodesic, and whether the chord correction
  computed from nominal curvature holds for measured chords.
* Authority: admission in `StateStore` is workbench bookkeeping and confers no
  actuator, safety or production authority.

The next step toward any of these is acquisition. Retain raw bytes with device
identity, acquisition time and calibration reference, as required for a
`hardware_measured` finding, then repeat T048–T056 against that data.
Recommended follow-ups in the queue: T060 (multi-sensor bench) and T066
(filtered residuals must use the filter covariance, not the raw sensor
covariance).
