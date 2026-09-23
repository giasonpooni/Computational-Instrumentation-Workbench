# Manufacturing and robotic use cases (T126–T141)

Section 9 of the computational-experimentalist queue designs measurement
protocols and models the geometric sensitivity of manufacturing and robotic
paths. **Nothing in this section is measured.** Every specimen, instrument
uncertainty, tolerance, friction coefficient and steering limit is a declared
planning input. Model predictions carry the evidence label that
`ciw.lab.evidence` computes from their basis; every claim about a physical part,
a real instrument, a calibration, machine safety or production acceptance is
recorded as a `not_established` finding in its physical or authority domain.

Code: `src/ciw/lab/manufacturing.py` (task registrations),
`manufacturing_geometry.py` (routes, curvature, standoff, coverage),
`manufacturing_metrology.py` (sampling, registration, artifacts, frames, Gage
R&R), `manufacturing_records.py` (protocols, retention, comparison, acceptance
policy). Tests: `tests/test_lab_manufacturing.py`.

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
marker layout; executed paths; instruments with **declared** standard
uncertainties (`status: declared_not_verified`); required raw data; calibration
artifacts (scale bar SB-1000, gauge sphere GS-25.4, step gauge SG-200);
environment; procedure; predicted quantities (each copying the evidence label of
the finding that produced it); acceptance criteria stated as **hypotheses**; an
empty hardware slot `{"status": "not_acquired", "records": []}`; and
`production_acceptance: "outside_system"`.

The validator refuses seven mutations, each with its own code: missing
markers, a filled hardware slot without acquisition fields, records in an empty
slot, a criterion marked `accepted`, a prediction labelled `hardware_measured`,
an instrument claimed verified, and acceptance declared inside the system.

Declared instruments (1σ per coordinate; planning values only): camera
(photogrammetry) 0.02 mm, laser tracker 0.015 mm + 6 µm/m, CMM 0.002 mm, laser
line scanner 0.01 mm at 0.05 mm native spacing. A marker-pair distance has
u_pair = √2 × 0.02 = 0.028 mm with the camera.

<a id="t126"></a>
## T126 — Flat-plate control (MFG-FLAT-PLATE-01)

5 × 5 coded markers at 60 mm pitch; nominal path N0 from (−120, 0) along +x for
240 mm, a 2 mm laterally offset path and a 5 mrad heading-offset path.

Predictions: chord = geodesic for every marker pair (max difference 0 on 47
pairs, RK4 closure ≤ 6e−14 mm); Jacobi transfer [[1, s], [0, 1]]; offset-path
separation δ + s sin δθ (2 mm constant; 0.3, 0.6, 0.9, 1.2 mm at 60…240 mm for
5 mrad). Hypotheses H1 (E_n ≤ 1 per marker pair) and H2 (heading slope 5 mrad).
The plate isolates instrument, frame and procedure error from curvature.

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
2 mm lateral and 5 mrad heading offset routes.

* First focal point (zero of j_lat): s_f = 153.407 mm (s_f/L = 0.759); the
  exactly offset 2 mm route crosses the nominal route at 154.361 mm (the shift is
  second order in the offset; Richardson in offset² recovers s_f to 1e−3 mm).
* 2 mm offset separation at the end: −1.037 mm (plate: +2 mm). 5 mrad heading
  offset at the end: 0.540 mm (plate at equal length: 1.011 mm).
* Linearization remainder: order 3.0 on the symmetry axis (the separation is odd
  in the offset), 2.0 off axis.
* Solver: Richardson estimate 1e−6; RK4 vs scipy DOP853 9.8e−7; Christoffel
  symbols and K vs a sympy derivation at rounding level. Without scipy/sympy the
  same findings fall back to ciw's Dormand–Prince integrator and central
  differences and are labelled `numerically_verified`, not independent.

Hypotheses: H1 the offset route crosses near s_f (within the T140 uncertainty);
H2 the measured separations follow the Jacobi prediction and reject the
flat-plate prediction at the last two stations.

<a id="t129"></a>
## T129 — Surface metrology: sampling density versus curvature

A least-squares fit z = a + b x + c x² over a centred window W with spacing d
estimates curvature 2c. For a profile with quartic term q x⁴ and white noise σ:

* bias(2c) = 2q · (fitted x² coefficient of x⁴) → (3/7) q W² for dense uniform sampling;
* std(2c) = σ √(720 d) / W^{5/2} (large n; the exact value comes from (XᵀX)⁻¹).

Allotting half the tolerance τ to the bias and half to k = 2 standard
deviations gives W = √(7τ/(6|q|)) and d = (τ W^{5/2}/(4σ))²/720. Averaging m
aligned repeat scans divides σ by √m, so m = ⌈d_native/d⌉.

| Profile | κ (1/mm) | τ | W (mm) | spacing (mm) | exact |bias| + 2σ |
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
  Monte Carlo spread matches σ√diag((JᵀJ)⁻¹) within 2%.
* Step gauge 10…200 mm: m = (1 + e)L + b recovers a declared 50 ppm scale error.
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
estimators are unbiased; true %GRR 23.9% but a single study's 90% interval is
[16.5%, 38.9%]; the operator component is negative (truncated) in 10.8% of
studies (F(2, 18) theory: 11.3%). Unbalanced or incomplete data are refused. No
real gage capability is established.

<a id="t132"></a>
## T132 — Fibre/tape placement on a cylindrical mandrel

On the development (Rφ, z) placement paths are straight (helices). A
variable-angle course θ(z) (from the axis) has κ_g = θ′(z) cos θ; the 30°→60°
course over 300 mm has max κ_g = 1.51e−3 /mm (steering radius 661.6 mm vs the
declared 635 mm limit — a physical claim, not established). Lateral deviation:
e(s) = δ + R cos θ δφ₀ + s(δψ + sin θ cos θ δR/R) (flat Jacobi transfer). With
declared 1σ sources (0.10 mm, 0.5 mrad, 0.05 mm radius, 50 µrad encoder) the
2σ stack reaches the 0.5 mm spec at L_max = 409.8 mm (0.594 mm at 500 mm).
Counterexample: a helix programmed in machine angles is not insensitive to
radius error — 0.2 mm on R = 100 mm drifts 0.999 mm over 1 m at 45°.

<a id="t133"></a>
## T133 — Winding path sensitivity (torus mandrel)

Geodesic windings conserve Clairaut's c = ρ² dφ/ds (drift < 1e−8 relative).
Launched on the outer equator at ψ = 50° from the parallel, the path turns at
θ = 115.39° and heading errors stay bounded (conjugate points at 538, 950 and
1349 mm; max |j_head|/length = 0.92). At ψ = 70° the path crosses the negatively
curved inner equator and the heading field grows to 2.89 × the cylinder value.
A 1 mrad heading error moves the turnaround by ρ₀ sin ψ δψ/(r sin θ_turn) =
3.39 mrad (integrated: within 0.2%). Constant-angle (loxodrome) winding is not
geodesic on the torus (counterexample); its slippage tendency |κ_g/κ_n| reaches
0.44 at 50° (above the declared μ = 0.2 on 65% of the path) and 0.14 at 70°.

<a id="t134"></a>
## T134 — Coating or welding trajectory sensitivity

Along the coupon route with a registration box |δ| ≤ 0.3 mm, |δθ| ≤ 2 mrad the
lateral envelope δ|j_lat| + δθ|j_head| peaks at 0.420 mm (s = 60 mm), confirmed
by exact perturbation at the box vertices. A lateral tool offset e produces a
standoff error −κ⊥e²/2 (≤ 1.8 µm here) and a tilt κ⊥e (≤ 8.7 mrad), confirmed by
ray casting. The tool-centre-point path X + Hn has speed factor
√((1 − Hκ_n)² + (Hτ_g)²): a 15 mm welding torch sees 0.84–1.37; a 120 mm spray
gun exceeds the 94.5 mm concave radius on the dome rim and its TCP path folds
back (33 reversed segments) — a counterexample to "offset tool paths of smooth
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
| geodesic, 12 mm | 17 | 3599 | 1 |
| geodesic, Jacobi-tightened 12.86 mm | 16 | 3398 | 0.9998 |
| chart-parallel, 20 mm | 10 | 2184 | 0.998 |
| chart-parallel, 18 mm | 12 | 2602 | 1 (shortest complete) |

The flat plate is covered exactly at 20 mm spacing; on the dome, geodesic rows
focus and cross (max j_lat 1.56), leaving 4% uncovered (counterexample).
First-order Jacobi tightening nearly but not completely restores coverage.

<a id="t136"></a>
## T136 — Ranking routes by calibration tolerance

Candidate routes: geodesics from the station (−60, 0) at 0…25° to the far edge.
Worst-case allocation keeps |e| ≤ 0.5 mm: δ_req = spec/(2 max|j_lat|),
δθ_req = spec/(2 max|j_head|); exact perturbation at the box vertices realizes
0.73–1.01 × the spec (second-order excess ≤ 1%).

| Route | L (mm) | max |j_head| (mm) | δθ_req (mrad) | flat plate (mrad) |
| --- | --- | --- | --- | --- |
| 0° | 202.18 | 107.9 | 2.316 | 1.237 |
| 5° | 202.34 | 126.4 | 1.977 | 1.236 |
| 10° | 203.11 | 169.4 | 1.476 | 1.231 |
| 15° | 205.28 | 213.1 | 1.173 | 1.218 |
| 20° | 209.76 | 242.8 | 1.030 | 1.192 |
| 25° | 217.19 | 257.3 | 0.972 | 1.151 |

The dome acts as a lens: the straight route tolerates 1.9 × the heading error
of a flat route of equal length.

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
equals the Jacobi index form j_head(L) j_head′(L) = 36.964 mm (finite
differences: 36.963 mm).

<a id="t138"></a>
## T138 — Predicted versus measured path separation (partial)

Predicted separation of the 2 mm offset route at the MFG-COUPON-01 stations:
2.000, 2.005, 2.051, 1.733, 1.157, 0.607, 0.059, −0.489, −1.037 mm, with k = 2
prediction uncertainty up to 0.30 mm (dome tolerances; solver negligible).
Comparison rule E_n = |m − p|/√(U_m² + U_p²) ≤ 1. **No measured separation
exists**; `compare_separation` refuses absent measurements, schema fixtures and
digest mismatches. The task stays `partial` until the protocol is executed,
retained (T139) and parsed by a registered reader.

<a id="t139"></a>
## T139 — Retention of raw measurements, calibration and frame metadata

`ciw.lab-measurement-retention.v1`: `record_kind` (measurement | schema_fixture),
protocol id, raw files (name, SHA-256, bytes, media type), instrument (id, kind,
serial), calibration (applied with reference, digest and validity window, or
not_applied), frame chain links (parent, child, proper rotation, translation,
6 × 6 PSD covariance over (ρ, φ), source) with continuity, and clock (source,
ISO 8601 time with explicit offset, synchronization, uncertainty). Nine
mutations are refused by code. `to_acquisition` produces the acquisition fields
of a `hardware_measured` finding only for a `measurement` record whose raw
bytes are presented and match; a schema fixture never does. No real
measurement is retained.

<a id="t140"></a>
## T140 — Instrument-, geometry- or solver-limited

u_c² = u_instrument² + u_geometry² + u_solver²; geometry from central
differences over rectangular tolerances (u = a/√3); solver from RK4 Richardson;
the limiting term has a variance share > 50%.

| Quantity | Value | Instrument | Geometry | Solver | Limiting |
| --- | --- | --- | --- | --- | --- |
| cylinder gap, 90° pair | 15.66 mm | 0.028 | 0.009 | 7e−14 | instrument |
| coupon separation at L, 5 mrad | 0.540 mm | 0.028 | 0.023 | 8e−8 | instrument |
| coupon focal distance | 153.4 mm | 1.3 | 4.4 | 1.5e−5 | geometry |
| plate separation at 240 mm | 1.2 mm | 0.028 | 0 | 0 | instrument |
| coarse-solver control (8 steps) | 0.500 mm | 0.028 | 0.023 | 0.042 | solver |

Counterexample: the focal-distance prediction is geometry-limited — a ±0.2 mm
dome-height tolerance moves the focus by millimetres — so scanning the as-built
coupon (T129) matters more than a better camera.

<a id="t141"></a>
## T141 — Production acceptance stays outside the system

`AcceptancePolicy.decide` always refuses (`production_acceptance_outside_system`)
and `record` returns `decision: not_performed`. Over 64 bases × 5 authority
domains (including hardware acquisition, passing checks and independent checks)
no authority claim is ever established; a forged `hardware_measured` acceptance
finding is refused; protocols refuse acceptance inside the system and criteria
marked `accepted`. Production acceptance and industrial readiness are recorded
as `not_established`.

## What must be measured

None of the following exists in the repository. Each is required before any
physical claim of this section can change label.

1. MFG-FLAT-PLATE-01: marker coordinates (camera, CMM), offset-path points,
   plate flatness and temperature; instrument checks on SB-1000 and GS-25.4.
2. MFG-CYLINDER-01: marker rings (camera, tracker), tube radius and roundness
   at three rings (CMM), helix and offset-helix points.
3. MFG-COUPON-01: as-built coupon scan (T129 design: 13.7 mm windows at
   ≤ 0.65 mm spacing for 5% curvature), station targets on the nominal, lateral
   and heading offset routes, the crossing arclength of the lateral offset route.
4. Calibration records for the camera, tracker, CMM and scanner; gauge sphere,
   step gauge and scale bar certificates; the frame chain with its measured
   covariances.
5. A real Gage R&R study (10 × 3 × 3) on the coupon features.
6. Process data for placement (tow positions, gaps/overlaps, wrinkling at the
   steering radius), winding (slip versus |κ_g/κ_n|, friction coefficient) and
   coating/welding (standoff, lateral error, deposited thickness).
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
| T137 | The shortest route is the one with the largest focus margin |
| T140 | Curved-surface predictions compared with photogrammetry are instrument-limited |
