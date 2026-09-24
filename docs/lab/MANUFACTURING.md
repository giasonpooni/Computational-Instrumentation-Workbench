# Manufacturing and robotic use cases (T126–T141)

Section 9 of the computational-experimentalist queue designs measurement
protocols and models the geometric sensitivity of manufacturing and robotic
paths. **Nothing in this section is measured.** Every specimen, instrument
uncertainty, tolerance, start-pose error, friction coefficient and steering
limit is a declared planning input. Model predictions carry the evidence label
that `ciw.lab.evidence` computes from their basis; every claim about a physical
part, a real instrument, a calibration, machine safety or production acceptance
is recorded as a `not_established` finding in its physical or authority domain.

Code: `src/ciw/lab/manufacturing.py` (task registrations),
`manufacturing_geometry.py` (routes, curvature, standoff, coverage),
`manufacturing_metrology.py` (sampling, registration, artifacts, frames, Gage
R&R), `manufacturing_records.py` (protocols, retention, comparison, acceptance
policy and acceptance-language screen). Tests: `tests/test_lab_manufacturing.py`.

```sh
ciw lab run T126 T127 T128 T129 T130 T131 T132 T133 T134 T135 T136 T137 T138 T139 T140 T141 --output-dir results/lab
ciw lab report T137 --retained results/lab
```

## Declared specimens (millimetres)

| Specimen | Model | Parameters | Role |
| --- | --- | --- | --- |
| Flat plate | `Plane` | 300 × 300 × 6, flatness 0.05 (declared) | zero-curvature control |
| Rolled cylinder | `Cylinder` | R = 100, length 300, radius tolerance ±0.1 | extrinsic curvature, K = 0 |
| Domed coupon | `GaussianBump` | h = 10, σ = 20; chart x ∈ [−60, 140], y ∈ [−100, 100]; h ± 0.2, σ ± 0.5 | intrinsic curvature (focusing) |
| Winding mandrel | `Torus` | R = 150, r = 50 | K changes sign |

The coupon crest has principal radius σ²/h = 40 mm and lies 60 mm from the
station edge, so the straight route from the station (−60, 0) to the far edge
x = 140 crosses the dome and passes through a lateral focal point.

## Measurement protocol record

`ciw.lab-measurement-protocol.v1` (validated by
`manufacturing_records.validate_protocol`) holds: specimen and surface model;
fixture (3-2-1 kinematic nest); datum frames A (primary, 3 points), B
(secondary, 2 points), C (tertiary, 1 point) and the PART frame built by
`datum_frame_321`; the frame chain INSTRUMENT → WORLD → FIXTURE → PART → CAD;
marker layout; paths; instruments with **declared** standard uncertainties
(`status: declared_not_verified`); required raw data; calibration artifacts
(scale bar SB-1000, gauge sphere GS-25.4, step gauge SG-200); environment;
procedure; predicted quantities (each copying the evidence label of the finding
that produced it); acceptance criteria stated as **hypotheses**; an empty
hardware slot `{"status": "not_acquired", "records": []}`; and
`production_acceptance: "outside_system"`.

**Path realization.** A path that a robot traces from the model would only
test the robot's reproduction of its own program. The protocols therefore
realize each path as the centreline of a 3 mm unsteered adhesive tape laid from
a start jig that sets its start point and heading; without in-plane steering
the tape follows a geodesic (a physical assumption, recorded `not_established`).
The CMM probes each tape centreline at s = 0 and 20 mm, so the realized start
pose of the offset tape can be budgeted (open loop) or conditioned on (T138,
T140).

The validator refuses nine mutations, each with its own code: missing markers;
a filled hardware slot without acquisition fields; records in an empty slot; a
slot whose raw digest is not 64 hex characters or whose time is not ISO 8601
with an offset (`acquisition_malformed`); a slot that does not cite a T139
retention identity (`retention_record_missing`); a criterion marked
`accepted`; a prediction labelled `hardware_measured`; an instrument claimed
verified; and acceptance declared inside the system. A well-formed slot passes
this schema check only: whether its bytes came from an instrument is decided by
the runner's hardware gate, not by the protocol validator.

Declared instruments (1σ per coordinate; planning values only): camera
(photogrammetry) 0.02 mm, laser tracker 0.015 mm + 6 µm/m, CMM 0.002 mm, laser
line scanner 0.01 mm at 0.05 mm native spacing. A marker-pair distance has
u_pair = √2 × 0.02 = 0.028 mm with the camera. Declared relative start-pose
error of an offset tape (jig and laying): 0.05 mm lateral and 0.5 mrad heading
(1σ); its CMM estimate has √2 × 0.002 = 0.0028 mm and 2 × 0.002/20 = 0.2 mrad.

<a id="t126"></a>
## T126 — Flat-plate control (MFG-FLAT-PLATE-01)

5 × 5 coded markers at 60 mm pitch; nominal path N0 from (−120, 0) along +x for
240 mm, a 2 mm laterally offset path and a 5 mrad heading-offset path.

Predictions: chord = geodesic for every marker pair. The geodesic distance is
found by shooting (an RK4 geodesic from one marker; the arclength where its
along-track coordinate passes the other marker), not from the chord formula:
max |geodesic − chord| = 2.8e−14 mm over 47 pairs, and the geodesic of that
length closes onto the target within 2e−14 mm. The Jacobi transfer is
[[1, s], [0, 1]]; offset-path separation δ + s sin δθ (2 mm constant; 0.3, 0.6,
0.9, 1.2 mm at 60…240 mm for 5 mrad). Hypotheses H1 (E_n ≤ 1 per marker pair)
and H2 (the heading-offset slope equals the start heading difference measured
by the CMM). The plate isolates instrument, frame and procedure error from
curvature.

<a id="t127"></a>
## T127 — Rolled-cylinder control (MFG-CYLINDER-01)

Three rings (z = 50, 150, 250) of 12 markers. The cylinder is intrinsically
flat: its Jacobi transfer along a 45° helix equals the plate's exactly. Its
extrinsic curvature makes chords shorter than geodesics:
gap(d) = d − 2R sin(d/2R) = d³/24R² − d⁵/1920R⁴ + …, zero on axial rulings.

| Pair | Geodesic (mm) | Gap (mm) |
| --- | --- | --- |
| circumferential 30° | 52.36 | 0.598 |
| circumferential 90° | 157.08 | 15.658 |
| axial 100 / 200 mm | 100 / 200 | 0 |

Counterexample: "on a developable part the chord between markers equals their
surface distance" fails at 90° (chord 141.42 vs geodesic 157.08 mm). Minimum
circumferential separation whose gap reaches k·u_pair (k = 2): camera 23.9 mm,
tracker 21.7 mm, CMM 11.1 mm.

<a id="t128"></a>
## T128 — Domed coupon (MFG-COUPON-01)

Stations at k·L/8 along the nominal route (L = 202.180 mm) and along the
2 mm lateral and 5 mrad heading offset tapes.

* First focal point (zero of j_lat): s_f = 153.407 mm (s_f/L = 0.759); the
  exactly offset 2 mm route crosses the nominal route at 154.361 mm (the shift is
  second order in the offset; Richardson in offset² recovers s_f to 1.1e−3 mm).
* 2 mm offset separation at the end: −1.037 mm (plate: +2 mm), 9.3 × the
  combined k = 2 uncertainty of the open-loop prediction and the camera pair.
  5 mrad heading offset at the end: 0.540 mm (plate at equal length: 1.011 mm).
* Linearization remainder: order 3.00 on the symmetry axis (the separation is
  odd in the offset), 2.00 off axis.
* Solver: Richardson estimate 1.05e−6.

Second derivations are separate findings with stable claims:

| Finding | Always present | Label |
| --- | --- | --- |
| RK4 vs ciw's adaptive Dormand–Prince 5(4) (9.8e−7) | yes | `numerically_verified` (same origin) |
| Christoffel symbols vs central differences of the metric (4.2e−13); K vs the Monge formula on central differences of the height values (7.8e−11) | yes | `numerically_verified` (same origin) |
| scipy DOP853 integrating the ciw right-hand side (9.8e−7) | scipy installed | `independently_verified` (independent time stepping only; the geometry code is shared) |
| sympy derivation of Γ and K from the height formula at six points (rounding level) | sympy installed | `independently_verified` |

Without scipy or sympy the last two findings keep their claims but have no
value and are `not_established` with `expected_not_established`; the report
prose does not depend on the environment. Regenerating T128 in a numpy-only
environment therefore differs from a full-environment run on exactly these two
findings (label, value and `expected_not_established`). `ciw lab verify`
reports these as problems, and marks the label change "optional modules
differ"; compare T128 only between runs made with the same optional modules.

Hypotheses (conditioned on the CMM start pose of the offset tape): H1 the offset
tape crosses the nominal tape where the geodesic re-integrated from its measured
start crosses, within the T140 expanded uncertainty of the focal distance; H2
the measured separations follow that conditioned Jacobi prediction (E_n ≤ 1 at
stations 1–8, with U_p from dome tolerances, start-pose estimate and solver)
and reject the flat-plate prediction at the last two stations.

<a id="t129"></a>
## T129 — Surface metrology: sampling density versus curvature

A least-squares fit z = a + b x + c x² over a centred window W with spacing d
estimates curvature 2c. For a profile with quartic term q x⁴ and white noise σ:

* bias(2c) = 2q · (fitted x² coefficient of x⁴) → (3/7) q W² for dense uniform sampling;
* std(2c) = σ √(720 d) / W^{5/2} (large n; the exact value comes from (XᵀX)⁻¹).

Allotting half the tolerance τ to the bias and half to k = 2 standard
deviations gives W = √(7τ/(6|q|)) and d = (τ W^{5/2}/(4σ))²/720. Averaging m
aligned repeat scans divides σ by √m, so m = ⌈d_native/d⌉.

| Profile | κ (1/mm) | τ | W (mm) | spacing (mm) | exact bias + 2σ (1/mm) |
| --- | --- | --- | --- | --- | --- |
| coupon crest | 0.025 | 5% | 13.66 | 0.646 | 1.22e−3 ≤ 1.25e−3 |
| cylinder circumference | 0.01 | 5% | 68.3 | 3.42 (≥ 21 samples) | 3.1e−4 ≤ 5e−4 |
| flat plate | 0 | 1e−4 /mm | 100 | 5.0 | 1.1e−5 ≤ 1e−4 |

Repeat scans at 0.05 mm native spacing on the coupon crest: 5% → 1, 2% → 5,
1% → 109 (impractical; a higher-order local fit is a deferred question).
Counterexamples: a window chosen from the osculating circle (quartic κ³/8)
gives 1.96 × the tolerance in bias on the Gaussian crest, whose quartic is
σ²/h² = 4 times larger; and a window of W/3 at the designed spacing has 4.8 × the
RMS error. Registration of repeat scans (Kabsch, 6 targets): the residual has
3N − 6 degrees of freedom and the rotation covariance σ²(Σ|p|²I − ppᵀ)⁻¹, both
confirmed by Monte Carlo.

<a id="t130"></a>
## T130 — Calibration artifacts and datum frames

* Gauge sphere Ø25.4 probed at 25 points on a 75° cap (CMM): geometric fit;
  Monte Carlo spread matches σ√diag((JᵀJ)⁻¹) within 2.1%.
* Step gauge 10…200 mm: m = (1 + e)L + b recovers a declared 50 ppm scale error
  and 1 µm offset exactly without noise; with noise the spread matches the
  linearized one within 3.2%.
* 3-2-1 datum frame: orthonormal, rigid-motion equivariant; tracker probing
  (0.015 mm) gives rotation std ≈ 100–130 µrad.
* Frame chain with left perturbations T = exp(ξ)T̄:
  C = Σ Ad(T₁…T_{k−1}) C_k Ad(…)ᵀ, point covariance [I, −(Tp)^] C [I, −(Tp)^]ᵀ.
  Coupon far-corner std ≈ (0.045, 0.040, 0.045) mm; the WORLD → FIXTURE link
  dominates (0.062 mm RSS) through its rotation lever arm. First order agrees
  with exact SE(3) sampling within 3–4% (relative Frobenius).

<a id="t131"></a>
## T131 — Repeatability and Gage R&R (ANOVA method)

Balanced crossed design, 10 parts × 3 operators × 3 replicates. Expected mean
squares give σ²_e = MS_E, σ²_po = (MS_PO − MS_E)/r, σ²_o = (MS_O − MS_PO)/(pr),
σ²_p = (MS_P − MS_PO)/(or). On 2000 synthetic studies with declared components
(part 0.050, operator 0.006, interaction 0.004, repeatability 0.010 mm) the raw
estimators are unbiased. True %GRR is 23.9%, but a single study's 90% interval
is [16.5%, 38.9%]: the raw GRR estimate is a linear combination of independent
scaled χ² mean squares, and its Monte Carlo mean and variance match the exact
values Σc²·2E[MS]²/df within their standard errors. The operator component is
negative (truncated) in 10.8% of studies (F(2, 18) theory: 11.3%). Unbalanced or
incomplete data are refused. No real gage capability is established, and the
AIAG interaction-pooling rule is not applied.

<a id="t132"></a>
## T132 — Fibre/tape placement on a cylindrical mandrel

On the development (Rφ, z) placement paths are straight (helices). With N the
tangent rotated +90° in (φ, z) and θ(z) measured from the axis, a variable-angle
course has signed κ_g = −θ′(z) cos θ (the heading from the circumferential
direction is 90° − θ), so |κ_g| = θ′ cos θ. The 30°→60° course over 300 mm has
max |κ_g| = 1.51e−3 /mm (steering radius 661.6 mm vs the declared 635 mm limit —
a physical claim, not established). Lateral deviation:
e(s) = δ + R cos θ δφ₀ + s(δψ + sin θ cos θ δR/R) (flat Jacobi transfer). With
declared 1σ sources (0.10 mm, 0.5 mrad, 0.05 mm radius, 50 µrad encoder) the
2σ stack reaches the 0.5 mm spec at L_max = 409.8 mm (0.594 mm at 500 mm);
20000 draws through the exact development model give a spread within 4
standard errors of the RSS. Counterexample: a helix programmed in machine
angles is not insensitive to radius error — 0.2 mm on R = 100 mm drifts
0.999 mm over 1 m at 45°.

<a id="t133"></a>
## T133 — Winding path sensitivity (torus mandrel)

Geodesic windings conserve Clairaut's c = ρ² dφ/ds (drift < 4e−9 relative).
Launched on the outer equator at ψ = 50° from the parallel, the path librates:
it turns at θ = ±115.39° before reaching the inner equator (period 782.687 mm).
At ψ = 70° it circulates through the negatively curved inner region (period
366.234 mm).

In **both** regimes the heading-error Jacobi field grows secularly (linearly),
not boundedly and not exponentially. On a surface of revolution the rotational
Killing field is a periodic normal Jacobi field, so the one-period monodromy M
is parabolic (trace M = 2, checked to 1e−7) and j_head(nP) = n·M01 (checked
over three periods to 1e−7). The mechanism: a heading error δψ changes the
Clairaut constant by −ρ₀ sin ψ δψ and with it the azimuth advance A(c) per
period, so M01 = ρ₀² sin²ψ dA/dc, which the code evaluates by Gauss–Legendre
quadrature and matches to 1.3e−7 relative. Growth rate |M01|/P relative to the
cylinder (rate 1): **1.042** at 50° (librating) and **2.954** at 70°
(circulating). A 1 mrad heading error moves the turnaround by
ρ₀ sin ψ δψ/(r sin θ_turn) = 3.39 mrad (integrated: within 0.12%).

Constant-angle (loxodrome) winding is not geodesic on the torus
(counterexample); its slippage tendency abs(κ_g/κ_n) reaches 0.44 at 50° (above
the declared μ = 0.2 on 65% of the path) and 0.14 at 70°. The rate is per unit
heading error; how a physical winding error grows also depends on the machine's
corrections between layers.

<a id="t134"></a>
## T134 — Coating or welding trajectory sensitivity

Along the coupon route with a registration box |δ| ≤ 0.3 mm, |δθ| ≤ 2 mrad the
lateral envelope δ|j_lat| + δθ|j_head| peaks at 0.420 mm (s = 60 mm), confirmed
by exact perturbation at the box vertices. A lateral tool offset e produces a
standoff error −κ⊥e²/2 (≤ 1.8 µm here) and a tilt κ⊥e (≤ 8.7 mrad), confirmed by
ray casting (relative difference ≤ 9.4e−5, checked against 1e−3 of the series
plus 1e−9 mm); the ray caster itself is checked on a Monge-form cylinder
z = √(R² − y²) − R against R − √(R² − e²) (difference 9e−15 mm). The
tool-centre-point path X + Hn has speed factor √((1 − Hκ_n)² + (Hτ_g)²): a
15 mm welding torch sees 0.84–1.37; a 120 mm spray gun exceeds the 94.5 mm
concave radius (at the route nodes) on the dome rim and its TCP path folds back
(33 reversed segments) — a counterexample to "offset tool paths of smooth
surface paths are smooth".

<a id="t135"></a>
## T135 — Robotic inspection scan paths

Rows are launched from the edge x = −60 mm (geodesic) or follow y = const in the
chart (chart-parallel); the footprint is a 20 mm swath (3D distance ≤ 10 mm).
Coverage is an area fraction over 6400 area-weighted Halton points (re-checked
on 25600 further points for plans that look complete).

| Plan | Rows | Length (mm) | Coverage |
| --- | --- | --- | --- |
| geodesic, 20 mm spacing | 10 | 2177 | 0.960 |
| geodesic, 14 mm | 15 | 3199 | 0.9994 |
| geodesic, 12 mm | 17 | 3600 | 1 |
| geodesic, Jacobi-tightened 12.86 mm | 16 | 3398 | 0.99984 (0.99977 on the larger sample) |
| chart-parallel, 20 mm | 10 | 2184 | 0.998 |
| chart-parallel, 18 mm | 12 | 2602 | 1 (shortest complete) |

The flat plate is covered exactly at 20 mm spacing; on the dome, geodesic rows
focus and cross (max j_lat 1.56), leaving 4% uncovered (counterexample).
First-order Jacobi tightening nearly but not completely restores coverage.
Checks: rows are mirror images across y = 0 (a negative-offset row integrated
directly equals the mirrored positive row exactly), and no sample lies within
3.6e−4 mm of a footprint boundary, so rounding cannot flip a coverage count.

<a id="t136"></a>
## T136 — Ranking routes by calibration tolerance

Candidate routes: geodesics from the station (−60, 0) at 0…25° to the far edge.
First order, half the 0.5 mm spec goes to each error source:
δ_req = spec/(2 max|j_lat|), δθ_req = spec/(2 max|j_head|). This allocation is
not enough: at the (+, +) or (−, −) corner of the tolerance box the exactly
perturbed 15°, 20° and 25° routes reach 1.0091, 1.0023 and 1.0003 × the spec
(counterexample). All four corners of every box are evaluated exactly, and a
route whose worst corner exceeds the spec has both tolerances derated by the
worst ratio until every corner is within 0.9999 of the spec. The published
tolerances are the derated ones; at their corners the realized error is
0.73–0.9999 × the spec.

| Route | L (mm) | max abs(j_head) (mm) | δθ first order (mrad) | δθ derated (mrad) | δ derated (mm) | flat plate δθ (mrad) |
| --- | --- | --- | --- | --- | --- | --- |
| 0° | 202.18 | 107.9 | 2.316 | 2.316 | 0.244 | 1.237 |
| 5° | 202.34 | 126.4 | 1.977 | 1.977 | 0.243 | 1.236 |
| 10° | 203.11 | 169.4 | 1.476 | 1.476 | 0.243 | 1.231 |
| 15° | 205.28 | 213.1 | 1.173 | 1.162 | 0.210 | 1.218 |
| 20° | 209.76 | 242.8 | 1.030 | 1.027 | 0.153 | 1.192 |
| 25° | 217.19 | 257.3 | 0.972 | 0.971 | 0.141 | 1.151 |

The dome acts as a lens: the straight route tolerates 1.87 × the heading error
of a flat route of equal length. Tape or robot path-following error after the
start is not included.

<a id="t137"></a>
## T137 — Ranking routes by focus margin

Focus margin = (nearest focal or conjugate point along the route) / L, or
horizon/L as a lower bound when none lies within the horizon.

| Route | L (mm) | Nearest focus (mm) | Margin |
| --- | --- | --- | --- |
| 15°, 20°, 25° | 205–217 | none within horizon | ≥ 2.52 |
| 10° | 203.11 | focal 329.9 | 1.62 |
| 5° | 202.34 | focal 176.0 | 0.87 |
| 0° | 202.18 | focal 153.4 | 0.76 |

Counterexample: the shortest candidate (0°) has the worst focus margin, and the
calibration ranking (T136) puts the same route first. The straight route is
nonetheless a local length minimum to the edge: the second variation of length
equals the Jacobi index form j_head(L) j_head′(L) = 36.964 mm (Richardson finite
differences: 36.963 mm, relative difference 1.6e−5 against a 2e−4 tolerance).

<a id="t138"></a>
## T138 — Predicted versus measured path separation (partial)

Predicted separation of the 2 mm offset tape at the MFG-COUPON-01 stations:
2.000, 2.005, 2.051, 1.733, 1.157, 0.607, 0.059, −0.489, −1.037 mm.

The prediction uncertainty has three components: the dome tolerances (central
differences, dominant beyond mid-route, up to 0.148 mm standard), the solver
(Richardson, ≤ 3e−8 mm) and the realized start pose of the offset tape. Start
sensitivities are central differences of the exactly re-integrated offset
route, and forward and central differences of every sensitivity agree within
3% (linear regime).

* Open loop (declared jig error 0.05 mm, 0.5 mrad): k = 2 uncertainty 0.10 mm
  at the start, up to 0.320 mm at the end.
* Conditioned on the CMM start pose (the prediction re-integrated from it):
  0.006 mm at the start, up to 0.300 mm at the end.

A correct model compared with a tape realized 0.1 mm off its nominal offset (2σ
of the jig; the "measurement" is the exactly re-integrated route, model output
standing in for a perfect instrument) fails E_n ≤ 1 (max 1.81) when U_p covers
only the dome tolerances and the solver, and passes (max 0.85) once the
start-pose term is included — a counterexample to "a correct model passes
against its open-loop prediction when U covers the instrument and the dome".
The measured branch passes too: re-integrating the prediction from a CMM
estimate of the realized start pose that is 2σ off in offset (±0.0057 mm) and
heading (±0.4 mrad), at all four sign corners, gives max E_n 0.46 against the
conditioned U_p.

Comparison rule E_n = |m − p|/√(U_m² + U_p²) ≤ 1. **No measured separation
exists**; `compare_separation` refuses absent measurements, a schema fixture (as
is and relabelled as a measurement) and digest mismatches. The task stays
`partial` until the protocol is executed, retained (T139) and parsed by a
registered reader.

<a id="t139"></a>
## T139 — Retention of raw measurements, calibration and frame metadata

`ciw.lab-measurement-retention.v1`: `record_kind` (measurement | schema_fixture),
protocol id, raw files (name, SHA-256, bytes, media type), instrument (id, kind,
serial), calibration (applied with reference, 64-hex digest and an ISO 8601
validity window, or not_applied), frame chain links (parent, child, proper
rotation, translation, 6 × 6 PSD covariance over (ρ, φ), source) with
continuity, and clock (source, ISO 8601 time with explicit offset,
synchronization, uncertainty). Eleven mutations are refused by code, including
a calibration digest that is not SHA-256 hex and an acquisition time outside the
calibration validity window. Covariance symmetry and positive semidefiniteness
are tested relative to the matrix scale, so a valid rank-3 covariance with
entries of 7e2 mm² (rounding-level eigenvalue −2e−14 mm²) is kept.

`to_acquisition` produces the acquisition fields of a `hardware_measured`
finding only for a `measurement` record whose raw bytes are presented and
match. Its `raw_sha256` is the SHA-256 of a canonical manifest of every raw file
(name, digest, size, media type), so it binds all files, and the citing task
must retain the manifest bytes. A record of kind `schema_fixture` is refused, and
so is any record carrying the fixture's markers (raw bytes starting with
`SCHEMA FIXTURE`, a `FIXTURE-` serial, an all-zero calibration digest), whatever
its `record_kind` says. T139 retains the fixture record, not the fixture bytes,
so no retained artifact carries a digest a hardware claim could cite. These
validators cannot tell whether bytes came from an instrument; the runner also
requires a hardware probe that succeeded in the task citing them. No real
measurement is retained.

<a id="t140"></a>
## T140 — Instrument-, geometry-, execution- or solver-limited

u_c² = u_instrument² + u_geometry² + u_execution² + u_solver²; geometry from
central differences over rectangular tolerances (u = a/√3); execution = start
pose uncertainty × the Jacobi fields (declared jig error open loop, CMM estimate
when conditioned); solver = |Q(h) − Q(2h)|/15 for a reported h-solution and
(16/15)|Q(h) − Q(2h)| only when the reported value is the coarse one (the 6-step
control); the limiting term has a variance share > 50%, otherwise "mixed".

| Quantity | Value | Instrument | Geometry | Execution | Solver | Limiting |
| --- | --- | --- | --- | --- | --- | --- |
| cylinder gap, 90° pair | 15.66 mm | 0.028 | 0.009 | 0 | 7e−14 | instrument |
| coupon separation at L, 5 mrad, open loop | 0.540 mm | 0.028 | 0.023 | 0.060 | 5e−9 | execution |
| coupon separation at L, 5 mrad, conditioned | 0.540 mm | 0.028 | 0.023 | 0.022 | 5e−9 | mixed |
| coupon separation at L, 2 mm lateral, conditioned | −1.037 mm | 0.028 | 0.148 | 0.022 | 3e−8 | geometry |
| coupon focal distance, conditioned | 153.4 mm | 1.3 | 4.4 | 0.83 | 9e−7 | geometry |
| plate separation at 240 mm, 5 mrad, open loop | 1.2 mm | 0.028 | 1.4e−7 | 0.130 | 0 | execution |
| plate separation at 240 mm, 5 mrad, conditioned | 1.2 mm | 0.028 | 1.4e−7 | 0.048 | 0 | execution |
| coarse-solver control (6 RK4 steps) | 0.625 mm | 0.028 | 0.023 | 0.022 | 0.094 | solver |

The plate geometry term models the declared 0.05 mm flatness as a Gaussian bump
(u = 0.05/√3 mm, width 75 mm); its effect is second order. Over the 240 mm plate
route even a CMM start-pose estimate leaves the 20 mm heading baseline dominant.
The coarse control validates the solver estimate: the actual error against a
256-step reference is 0.90 × the Richardson estimate. Counterexample: the
focal-distance prediction is geometry-limited — a ±0.2 mm dome-height tolerance
moves the focus by millimetres — so scanning the as-built coupon (T129) matters
more than a better camera.

<a id="t141"></a>
## T141 — Production acceptance stays outside the system

`AcceptancePolicy.decide` always refuses (`production_acceptance_outside_system`)
and `record` returns `decision: not_performed`. Over 64 bases × 5 authority
domains (including hardware acquisition, passing checks and independent checks)
no claim filed in an authority domain is ever established; a forged
`hardware_measured` acceptance finding is refused; protocols refuse acceptance
inside the system and criteria marked `accepted`.

The label function cannot see which domain a free-text claim belongs to: "Coupon
lot accepted for production" filed in `computational_pipeline` with one passing
check is labelled `numerically_verified` by `evidence.finding` (recorded as a
counterexample to "the lab API cannot mark production acceptance"). Every task
of this section therefore passes its findings through
`screen_acceptance_language`, which refuses acceptance and rejection decision
phrases (accepted, approved, rejected, scrapped, quarantined, signed off,
dispositioned, released for or to production, passed or passes inspection or
acceptance, certified for production) in claims and string values outside the
authority domains; a refusal blocks the task. The task checks the screen with
"Coupon lot accepted for production" and "Coupon lot rejected for production". The screen is a vocabulary check on this section only,
so "domain assignment of free-text claims is machine-checked across the lab" is
recorded as `not_established`. Production acceptance and industrial readiness
are recorded as `not_established`.

## What must be measured

None of the following exists in the repository. Each is required before any
physical claim of this section can change label.

1. MFG-FLAT-PLATE-01: marker coordinates (camera, CMM), tape start poses (CMM
   at s = 0 and 20 mm), tape centreline points, plate flatness and temperature;
   instrument checks on SB-1000 and GS-25.4.
2. MFG-CYLINDER-01: marker rings (camera, tracker), tube radius and roundness
   at three rings (CMM), helix and offset-helix tape start poses and centrelines.
3. MFG-COUPON-01: as-built coupon scan (T129 design: 13.7 mm windows at
   ≤ 0.65 mm spacing for 5% curvature), tape start poses (CMM), station targets
   on the nominal, lateral and heading offset tapes, the crossing arclength of
   the lateral offset tape.
4. Calibration records for the camera, tracker, CMM and scanner; gauge sphere,
   step gauge and scale bar certificates; the frame chain with its measured
   covariances; the realized start-pose error of the jig and tape laying.
5. A real Gage R&R study (10 × 3 × 3) on the coupon features.
6. Process data for placement (tow positions, gaps/overlaps, wrinkling at the
   steering radius), winding (slip versus abs(κ_g/κ_n), friction coefficient)
   and coating/welding (standoff, lateral error, deposited thickness).
7. Scanner footprint on the coupon at the planned standoff.

Every measurement is retained through a T139 record and compared through T138;
acceptance of parts or processes remains an external decision (T141).

## Counterexamples recorded

| Task | Refuted statement |
| --- | --- |
| T127 | On a developable part the chord between markers equals their surface distance |
| T129 | A window from the osculating circle bounds the quadratic-fit bias on any convex profile |
| T129 | A smaller fitting window always improves the curvature estimate |
| T132 | A helix programmed in machine angles is insensitive to mandrel radius error |
| T133 | A constant winding angle is geodesic and slip-free on every mandrel of revolution |
| T134 | The standoff tool path of a smooth surface path is itself smooth |
| T135 | Geodesic rows at the swath spacing cover a curved coupon as they cover a plate |
| T136 | Allocating spec/(2 max abs(j)) to each error source keeps the exactly perturbed route within the spec |
| T137 | The shortest route is the one with the largest focus margin |
| T138 | A correct model passes E_n ≤ 1 against its open-loop prediction when U covers only the instrument and the dome tolerances |
| T140 | Curved-surface predictions compared with photogrammetry are instrument-limited |
| T141 | The lab API cannot mark production acceptance |
