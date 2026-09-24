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
`observation_chord.py` (chord expansion, sympy references, integrated geodesics),
`observation_camera.py` (pinhole stereo rig, distortion, triangulation,
circular markers under perspective, shared-phase rounding) and
`observation_signals.py` (encoder, IMU, timing, Kalman/RTS, statistics).
Tests: `tests/test_lab_observation.py`.

```
python -m ciw lab run T045 T046 T047 T048 T049 T050 T051 T052 T053 T054 T055 T056 T057 T058 T059 --output-dir out/lab
python -m ciw lab report T046 --retained out/lab
python -m pytest -q tests/test_lab_observation.py
```

The section runs in about 9 s and its tests in about 13 s on one core. sympy
(T046, T047) and scipy (T057) are optional. Without them the derivation
findings are labelled `analytic` instead of `independently_verified`, and the
Riccati fixed-point and chi-square-quantile findings of T057 are
`numerically_verified` instead of `independently_verified`. These four labels
are environment dependent; the report prose is worded identically in both
environments so that only the labels differ.

Every computational finding carries a per-finding `uncertainty` (AUTHORING
rule 5): `exact` for counts, refusal codes and dyadic arithmetic; `roundoff`
for closed-form comparisons; `truncation_bound` for fitted coefficients
(estimated by fitting one more polynomial term) and linearizations;
`reference_error` where the recorded value is itself an approximation error
(a finite-difference Jacobian, the Wilson–Hilferty quantiles); and
`monte_carlo_95ci` (1.96 standard errors from independent runs) for seeded
ensembles.

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

In general, carrying the expansion one order further (γ⁽⁵⁾ enters |Δ|² at
s⁶) gives

    c₅ = (3κ⁴ + 8κ²τ² − 72κκ'' − 64κ'²)/5760,

which reduces to the constant-curvature value above when κ' = κ'' = 0.

T046 compares the closed form with two sympy references. Both curve models
are written in `observation_chord`; sympy performs their series expansion,
simplification and exact arithmetic, so sympy's independence covers that
algebra, not the model definitions.

* **Frenet–Serret Taylor recursion** (`sympy_general_series`). sympy expands
  the series with polynomial κ(s), τ(s). Each residual sympy(cₙ) − closed(cₙ)
  for n = 1…5, and the constant-curvature c₅, is simplified as a polynomial
  in (κ₀, κ₀', κ₀'', τ₀); the check passes only if all six residuals are
  identically zero. This route uses the same Frenet equations as the hand
  derivation, so a shared conceptual error in them would pass both sides. It
  is recorded as an ordinary `exact_arithmetic` check, not as the independent
  comparison.
* **Explicit polynomial space curves** (`sympy_explicit_curve_check`). For
  three curves g(t) = (t, a₂t² + a₃t³ + a₄t⁴, b₃t³ + b₄t⁴) with rational
  coefficients, sympy expands the arc length s(t) = ∫|g'| dt, reverts it to
  t(s), and expands the chord |g(t(s)) − g(0)| in s. κ and τ come from the
  cross-product formulas κ = |g' × g''|/|g'|³ and
  τ = (g' × g'')·g'''/|g' × g''|², re-expanded in s, which gives κ₀, κ₀',
  κ₀'' and τ₀. The closed-form c₁…c₅ at those values must equal the chord
  coefficients exactly: all 15 rational residuals are 0. This route uses
  neither the Frenet recursion nor the hand derivation, and it is the
  `independent_check` of T046. It is evaluated at three curves, not
  symbolically, because the fully symbolic version is too slow for the run
  budget. A closed form that is wrong as a polynomial would have to vanish at
  all three points by coincidence.

Counterexamples found by the task:

* **The remainder is not O(s⁵) in general.** With κ taken at the start point,
  the s⁴ term −κκ's⁴/24 survives whenever κ' ≠ 0. On a torus geodesic
  (R = 2, r = 1, start (0, π/4), heading 0.6 rad), the fitted s⁴ coefficient
  of s − c − κ₀²s³/24 matches κ₀κ₀'/24 to 5e-7 relative. Taking κ at the arc
  midpoint cancels the term by symmetry: the fitted midpoint coefficient is
  3e-7 of κ₀κ₀'/24, the same size as its fit truncation (4e-7), so only the
  bound (below 1e-4) is regression-tested.
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
arc lengths with and without a declared cylinder model. The chords come from
embedded points of `Cylinder.exact_geodesic` (straight lines in the (φ, z)
chart), a forward model independent of the closed-form helix chord that the
conversion inverts; the recovered arcs agree to 3e-17 m.

### T046 Chord-versus-geodesic correction
Hand derivation and the two sympy references above: a symbolic identity for
every coefficient through the Frenet recursion, and exact rational agreement
on three explicit curves without it. Sphere
geodesics integrated by `ciw.lab.jacobi` (RK4, R = 0.5, 1, 2) are checked
against 2R sin(s/2R) and a fitted s³ coefficient. Then the torus and helix
counterexamples above.

### T047 Cylinder coefficient
sympy series of the exact helix chord. The expression
√((2R sin(s cos α/2R))² + (s sin α)²) is written in
`observation_chord.sympy_cylinder_series`. It does not use the Frenet
expansion. Its c₃ and c₅ minus the closed form at κ = cos²α/R,
τ = sin α cos α/R simplify to 0 symbolically in R and α. The check counts the
residuals that do not (`exact_arithmetic`, observed 0). Least-squares fits of
(s − c)/s³ = a + bs² + ds⁴ at seven angles and two radii. Geodesics integrated
on `ciw.lab.surfaces.Cylinder`. Rulings against circumferential paths. On the
rulings the check bounds max |s − c|/R over s > 0 (two-sided, about 3e-16).
The s = 0 sample, where s − c = 0 exactly, is excluded so that it cannot
anchor the bound.

### T048 Synthetic camera measurements
Declared pinhole stereo pair: f = 2400 px, 2048 × 1536 px, baseline 0.2 m,
converging at 0.6 m. Markers every 15 mm along three helices (α = 0°, 45°,
90°) on a cylinder with R = 0.1 m. All 27 are checked to be front-facing and
inside both images. Noise-free DLT
and ray-midpoint triangulation are compared with truth (the two triangulations
are a same-origin `cross_implementation` check). The chord-for-geodesic
substitution bias equals s − c(s) exactly; it is largest on the
circumferential helix (7.07 mm at s = 0.12 m), 1.77 mm at 45° and zero on the
ruling, and these orderings are checked. With 0.25 px pixel noise and integer
rounding (grid phase drawn independently per marker, camera and axis), the
per-helix mean squared chord and converted-arc errors match first-order
propagation, σ_c² = (σ² + 1/12)ΣJ² and σ_s = σ_c/c'(s), within the 99.9 %
Monte Carlo bound (chord RMS about 0.19 mm overall; 500 trials give 1.7–2.1 %
relative standard error per helix RMS). For the longest circumferential chord
the bias is 64 times the noise RMS. Independence of the rounding errors
between markers is a property of this generator, not a checked failure mode;
T051 studies a shared phase.

### T049 Calibration perturbations
Pixels from the true rig are triangulated with a believed rig. Its 12 declared
parameters are the focal length (common to both cameras, plus an extra
right-camera error), the left and right principal points, and the right
camera's rotation about its centre and the shift of that centre (the x shift
changes the baseline). Chords are invariant under a rigid motion of the whole
believed rig. A left-camera pose error is therefore equivalent to a
right-camera pose error and is not listed separately. Aspect ratio and skew
are not perturbed; the report names them as omitted. The central-difference
Jacobian is compared with three closed forms on a rectified rig:
∂c/∂f = ΔZ²/(cf), ∂Z/∂c_x = −Z²/(bf) for the right camera, and ∂c/∂b = c/b.
It is checked by step halving. First-order predictions are compared with
direct recomputation over five scales; the residual slope is 2.
Errors that change horizontal disparity dominate: for the 0.12 m chord, 1 mrad
of right-camera yaw moves it by 0.36 mm and 1 px of horizontal principal point
by 0.15 mm, against about 6 µm for 1 mrad of pitch and 2 µm for 1 px of
vertical principal point. The dominance is checked only between like units
(|∂c/∂c_x| / |∂c/∂c_y| = 68 and |∂c/∂yaw| / |∂c/∂pitch| = 60, both ≥ 10); a
pixel-against-milliradian ratio would depend on an arbitrary unit choice.
Counterexample: a common focal error leaves same-depth chords unchanged (the
rectified-rig map is X_b = X, Z_b = Z f_b/f), so it is not a uniform scale.
A baseline error is one. A believed baseline b' scales X, Y and Z by b'/b, so
every chord scales by b'/b; for 1 mm this holds to 2e-16 m over all 24 pairs.
On the converged rig, 1 mm of baseline moves the 0.12 m chord by 0.59 mm.

### T050 Perspective and lens distortion
Brown–Conrady radial and tangential distortion,
x_d = x(1 + k₁r² + k₂r⁴) + tangential terms. On the image axis the
implemented displacement is f(k₁r³ + k₂r⁵); this is recorded only as an
implementation-consistency check, because it is the model's definition.

Chord-bias law. Uncorrected distortion displaces each endpoint's pixels by
δ = f k₁ r³ (radially), so the chord bias is b ≈ J_pix·δ_pix with J_pix the
chord's pixel Jacobian. A displacement shared by all image points hardly
changes a chord; what matters are displacement differences, between the two
endpoints and between the two cameras (which changes disparity and so the
depth scale). For image radii differing by dr both are
f k₁((r + dr)³ − r³) ≈ 3 f k₁ r² dr, so for chords of fixed image extent the
bias scales as k₁ r²: slope 2 in the mean image radius and 1 in k₁. The markers (five points at s = 0–0.06 m on the 45° helix,
pairs (0, 2), (0, 4), (1, 3)) are shifted across the image in five steps. The
first-order prediction J_pix·δ_pix holds to 0.3 % at |k₁| ≤ 0.1, the bias is
odd in k₁, the fitted log–log slope in radius is 2.04 (checked within 0.1 of
2), and the slope in |k₁| is within 1e-4 of 1. Pairwise slopes rise from 2.00
to 2.18 at the largest radius. Two causes contribute, and the run does not
separate them: the higher-order terms of f k₁((r + dr)³ − r³), and the pixel
Jacobian and displacement directions changing as the markers move (k₂ = 0 in
these runs).
Undistorting with the true (k₁, k₂, p₁, p₂) model removes the bias to
6e-16 m.

Fold counterexample. For k₁ < 0, r(1 + k₁r²) peaks at r_f = 1/√(−3k₁), where
the distorted radius is r_d,f = (2/3)/√(−3k₁). Pixel coordinates are distorted
coordinates, so the fold lies inside an image whose corner radius is ρ when
r_d,f < ρ, i.e. k₁ < −4/(27ρ²) = −0.521 for this rig (ρ = 0.533). At
k₁ = −0.6 the fold circle r_d,f = 0.497 lies inside the image: the corner
pixels beyond it (1.05 % of the image) have no preimage, and every pixel
inside it has two. The point at undistorted radius 0.85 on the image diagonal
and the point at 0.636 map to the same pixel (515 px apart before
distortion); fixed-point undistortion run to convergence (400 iterations)
returns the monotone-branch preimage 0.636, not 0.85.

Perspective: circular markers. The markers in T048 are points. A real marker
is often a flat disc, and a detector reports the centre of its image ellipse.
Under full perspective that centre is not the image of the disc's centre.
Take a disc with centre C = (X, Y, Z), unit normal n and radius ρ, in camera
coordinates. Its image conic has the dual H diag(ρ², ρ², −1) Hᵀ with
H = [e₁ e₂ C]. The ellipse centre is the pole of the line at infinity,
Z C − ρ² t, with t = e_z − n_z n the in-plane component of the optical axis:

    x_e = (Z X − ρ² t_x)/(Z² − ρ² t_z),   offset from X/Z = ρ²(X t_z − Z t_x)/(Z (Z² − ρ² t_z))

and likewise for y. The offset is O(f ρ²/Z²). It vanishes for a
fronto-parallel disc (t = 0) and under any affine (weak-perspective)
projection, because affine maps preserve ellipse centres. On the optical axis
it reduces to the familiar eccentricity f ρ² sin β cos β/(Z² − ρ² sin² β)
for a tilt β.

Each of the 27 T048 markers becomes a disc of radius 2, 4 or 8 mm in the
cylinder's tangent plane. Its image centre is located as the centre of the
conic through 64 projected rim points, a forward computation that does not use
the closed form. The closed form matches it to 2.3e-13 px. Two controls give
0 offset to rounding: fronto-parallel discs, and the weak-perspective
projection. The largest offset is 0.011, 0.045 and 0.178 px (slope 2.0000 in
ρ). Triangulating the ellipse centres biases the chords by J_pix·δ_pix to
first order (residual 4e-4 relative). At 8 mm the bias reaches 80 µm on the
circumferential helix, 40 µm at 45° and 12 µm on the ruling, with slope 2 in
ρ. For comparison, the 0.25 px noise gives a chord RMS of about 0.19 mm.
Counterexample: the ellipse centre is not the projected marker centre
(marker 10, left camera, 0.178 px at 8 mm). Distortion and perspective are
studied separately; a real detector (edge or intensity centroid) is not
modelled.

### T051 Quantization and pixel noise
e = round(x + n) − x has variance σ² + 1/12 when the sub-pixel phase is
uniform, because the rounding error is then uniform and independent of n. This
is checked at five values of σ. Counterexample: an integer-aligned coordinate
with σ = 0.1 px has almost no total error. At x = 512 the rounding error is
q = −n (variance σ²), which cancels the Gaussian noise, so e = n + q is 0
unless |n| > 1/2. The error variance, exactly 2Φ(−5) = 5.7e-7 px², is far
below both σ² + 1/12 and σ² (checked below 1 % of σ²). Chord standard
deviations follow √((σ² + 1/12) Σ J²), which assumes independent errors in the
eight pixel coordinates of a pair. The generator therefore draws the grid
phase independently per marker, camera and axis. The law is checked against
4000 seeded stereo trials with 99.9 % bounds (max |z| = 1.8).

Shared grid phase. If one phase per camera and axis is shared by all markers,
the rounding errors of a pair's two markers on the same camera and axis are
correlated. With d = frac(x₁ − x₂), the sawtooth Fourier series gives

    Cov(e₁, e₂) = Σ_k cos(2πkd) exp(−4π²k²σ²)/(2π²k²)   (= 1/12 − d(1 − d)/2 for σ = 0)

so the chord variance is J Σ Jᵀ. The same 4000-trial stereo Monte Carlo with a
shared phase matches J Σ Jᵀ for all six pairs at σ = 0 and σ = 0.25 px (max
|z| = 2.4). At σ = 0 it refutes the independent law: the predicted variance
departs from it by −44 % to +0.3 %, and the independent law gives |z| = 25.6.
At σ = 0.25 px the Gaussian noise damps the correlation. The predicted
departure is −2.2 % to −0.5 %, too small for 4000 trials to resolve, and it is
recorded from the exact series. A real rig's grid phase is neither
independent per marker nor exactly shared.

### T052 Encoder bias, scale and backlash
Reading = (1 + s)·play_b(x) + β + noise. The play error lies in [0, b] and
changes only during take-up after a reversal. A fit on engaged samples with a
direction term recovers (1 + s, β, (1 + s)b). Counterexample: fitting on
[x, 1] without the direction term absorbs the OLS projection of
(1 + s)(play_b(x) − x) onto [x, 1]; the predicted offset error is 0.4976 b and
the observed one 0.4979 b (z ≈ 40 against the naive standard error). The
simpler estimate "b times the falling fraction" (0.458 b) is not accurate
enough: the mean play error is 0.479 b because take-up windows add partial
errors, and the fitted slope changes by −6.9e-5, which with the mean position
of 5.3 mm moves the intercept by a further 0.018 b.

### T053 IMU drift and orientation noise
Single-axis heading error has mean bt, variance N²t and MSE N²t + (bt)²,
checked with 2000 runs. Strapdown SO(3): the bias error follows
e_{k+1} = exp(−[ω]dt)e_k + J_r(ωdt)b dt. Counterexample: on a body rotating
at 1 rev/s, a transverse bias gives an error bounded by 2|b⊥|/|ω| instead of
|b⊥|t. Angle random walk stays isotropic under rotation: each body axis has
spread N²t (checked per axis, since a trace test alone, 3N²t, cannot see
variance moving between axes).

### T054 Asynchronous timestamps
Sensor B (30 Hz) is interpolated to sensor A's times (100 Hz). A clock offset
δ gives e = −vδ + O(δ²), slope 2, below the curvature bound, over seven
offsets δ = 1e-4 to 1e-2 s (geometric). At δ = 2e-3 s, after linear
interpolation the error is exactly −Sδ, where S is the interpolant slope. For
a tone A sin(ωt + φ) sampled every h, S = Aω cos(ωt_mid + φ)·sinc(ωh/2), and
averaging cos(ω(t − t_mid)) over the position within the interval gives a
second sinc(ωh/2). The regression of −Sδ on −vδ is therefore the
velocity-power-weighted sinc²(ωh/2) ≈ 1 − (ωh)²/12, predicted 0.99208 against
0.99190 observed (checked within 5e-4; the secant factor alone,
1 − (ωh)²/24 = 0.99602, would fail). Jitter adds S²σ_j²((1 − w)² + w²). A
declared `ClockMapping` removes the declared 1/256 s offset of clock B
exactly. Combining the clocks without a mapping is refused.

### T055 Dropped observations
Drops stay `None` at their sequence positions. Zero-filled streams and values
without raw references are refused. Zero filling biases a mean by −pμ;
explicit gaps are unbiased with variance σ²E[1/N].

Counterexample to "zero fills can be recognized from the values alone". A
neighbour-median detector finds every fill when the signal is 50σ from zero,
so value-based detection works in favourable signals. The witness is a
stationary encoder axis whose 0.5 µm vibration is quantized to the declared
1 µm resolution, so genuine readings of exactly 0 counts are common. Of the 46
dropped samples, 26 had a genuine reading of 0 counts. Both histories are
built as typed records: the complete acquired stream, and the same stream
with those 26 samples dropped (`with_drops`) and then zero-filled
(`zero_fill`). Their values agree bit for bit, and exactly 26 positions were
filled. No rule that sees values alone can therefore recover the fill
positions of both. Provenance can: `admit_stream` accepts the complete stream
and refuses the zero-filled one (`zero_filled_missing`). On the same stream
the "flag every exact zero" rule flags 107 genuine readings and the
neighbour-median detector misses all 46 fills. Provenance (the raw
reference) is required.

Sample-and-hold tracking of a random walk has MSE qE[age] + r. At the same
20 % drop rate, Gilbert–Elliott bursts (E[age] = 1) give about 3.6 times the
Bernoulli error (E[age] = 0.25).

### T056 Stale-state observations
Age = t_use − (t_arrival − latency). Flags equal age > limit exactly. For
constant velocity the error of using each of the 400 tracker records at its
use time, v·t_use minus the recorded position, equals v times the age that
`acquisition_age` computes from the record (to 2e-16 m); otherwise the
residual is O(a²), below A ω² a²/2. Missing latency and future records are
refused. The acquisition-age definition means that a record that arrived
5 ms before use with 35 ms latency (acquisition age 40 ms) is refused under a
30 ms limit, although its arrival age is within the limit.

### T057 Raw, filtered and smoothed estimates
Constant-velocity Kalman filter (Joseph form) and Rauch–Tung–Striebel smoother
on a seeded 300-run ensemble. P_filt − P_smooth is positive semidefinite at
every step. It is only rank one at step N − 2, so strict reduction is tested
on its trace. Position RMSE orders smoothed ≤ filtered ≤ raw, and the
ensemble-average NEES lies inside the 95 % χ²(2N)/N bounds at about 97 % of
steps. The bounds use Wilson–Hilferty quantiles. They are always checked
against a series evaluation of the chi-square CDF (regularized incomplete
gamma function), and against `scipy.stats.chi2` when scipy is available. The
Riccati fixed point is compared with the filter's final predicted covariance
and, when scipy is available, with `scipy.linalg.solve_discrete_are`; only
the scipy comparisons make these two findings `independently_verified`.
Counterexample: the smoothed error is larger
than the filtered error at about 29 % of individual samples. The ordering is
an ensemble property.

### T058 Frame and clock basis tracking
Combining records across frame, clock, epoch or time basis is refused.
Declared rigid frame mappings (quarter turn with dyadic translation) match
hand-written formulas exactly. A general rotation round-trips to 4e-16 m and
preserves the pairwise distances of the mapped tracker positions to 9e-16 m;
both are checked against 16ε·max|p| ≈ 1e-14 m, since each mapped coordinate is
rounded. Distance-mode records (chords, geodesic distances) are carried
through a frame mapping unchanged by design, so their invariance is a
property of `apply_frame`, not a measurement. Clock mappings (arrival →
acquisition by the declared latency, then clock → clock) are exact on dyadic
times. Mappings for another frame, frame kind or epoch are refused.

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
