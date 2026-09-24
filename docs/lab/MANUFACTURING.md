# Manufacturing and robotic use cases (T126–T141)

Section 9 of the computational-experimentalist queue designs measurement
protocols and models the geometric sensitivity of manufacturing and robotic
paths. **Nothing in this section is measured.** Every specimen, instrument
uncertainty, tolerance, start-pose error, friction coefficient and steering
limit is a declared planning input. Model predictions carry the evidence label
that `ciw.lab.evidence` computes from their basis; every claim about a physical
part, a real instrument, a calibration, machine safety, customer demand or
production acceptance is recorded as a `not_established` finding in its
physical or authority domain. Instrument exports an operator binds to T138 are
read, retained and compared computationally, and a retention record bound to
T139 with its raw files is validated against them and retained; they are not
authenticated and no metrology instrument probe exists, so they never change a
physical label.

Code: `src/ciw/lab/manufacturing.py` (task registrations),
`manufacturing_geometry.py` (routes, curvature, standoff, coverage),
`manufacturing_metrology.py` (sampling, as-built dome fit, registration,
artifacts, cylinder fit, 3-2-1 and axis-primary frames, Gage R&R),
`manufacturing_records.py` (protocols and their executability checks,
retention, the capture reader, separation and marker-pair comparators,
acceptance policy and acceptance-language screen). Tests:
`tests/test_lab_manufacturing.py`.

```sh
ciw lab run T126 T127 T128 T129 T130 T131 T132 T133 T134 T135 T136 T137 T138 T139 T140 T141 --output-dir results/lab
ciw lab report T137 --retained results/lab
ciw lab run T138 --capture photogrammetry=targets.csv --capture cmm=start-pose.csv --output-dir results/coupon
ciw lab run T139 --capture retention=record.json --capture photogrammetry=targets.csv --output-dir results/coupon
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
`manufacturing_records.validate_protocol`) is written so that a technician can
execute it as it stands. It holds: specimen and surface model; **fixtures**,
each with what it locates, its contacts, geometry and clamping; **datum
frames**; the frame chain INSTRUMENT → WORLD → FIXTURE → PART → CAD; marker
layout; paths; instruments with **declared** standard uncertainties
(`status: declared_not_verified`) and their **settings** (camera stations,
targets and exposure; tracker SMR and averaging; CMM stylus, approach speed and
qualification; scanner standoff, incidence limit and spacing; film graduation
and reading); required raw data; the **raw formats** of the instrument exports
(role, schema, instrument, frame, columns, row identifiers and the reader that
parses them); calibration artifacts (scale bar SB-1000, gauge sphere GS-25.4,
step gauge SG-200); environment; the **procedure** as numbered steps, each
naming the devices it uses, the datums it touches, its outputs and, where one
applies, a step-level check (for example: each instrument check reads its
artifact within 2 U of the certificate, or the run stops); predicted
quantities (each copying the evidence label of the finding that produced it);
acceptance criteria stated as **hypotheses**; an empty hardware slot
`{"status": "not_acquired", "records": []}`; and
`production_acceptance: "outside_system"`. Every protocol starts with a soak,
instrument checks and locating the specimen in its holder with the PART frame
built from its datums, and ends with instrument checks and retention. In
between come its own steps. The three controls MFG-FLAT-PLATE-01,
MFG-CYLINDER-01 and MFG-COUPON-01 register the holder with the tracker (its
four declared datum targets), measure their markers, film or scan, and then
lay their tapes one at a time (see *Path realization*). The surface scan
MFG-SCAN-01 (T129) has no tracker, no registration step and no tapes; it also
carries a `scan_plan`. Replicates are explicit repeat steps that list, by
number, the measurement steps they repeat after re-seating the specimen:
locating and the marker, film or scan steps, and, for each tape while it is on
the specimen, its start-pose probing and target imaging. A repeat never lists
a step that lays, places or removes anything.

**Fixtures and datums.** A plate or a formed coupon sits in **FX-321**, a
3-2-1 kinematic nest (three spherical rests on the back face A, two side stops
on the long edge B, one end stop on C); the PART frame is built by
`datum_frame_321`. A tube has no face to rest on, so MFG-CYLINDER-01 has its
own scheme: it lies in **VB-01**, a pair of 90° vees with an end stop, their
centres 100 and 200 mm from the stop, between the marker rings at 50, 150 and
250 mm, so no ring rests in a vee. The tube is turned until the axial scribe B
faces straight up, which puts the seam weld (φ = 180°) at the bottom between
the vee contact lines; only the upper half, within 90° of B, is open to the
camera, tracker and CMM. Datum A is the axis of a least-squares cylinder
(`manufacturing_metrology.fit_cylinder`) through CMM points at φ = −75°, −15°,
45° and 75° on each marker ring (between markers, on the upper half, clear of
the tapes; 12 points), B the scribe probed at z = 40 and 260 mm, C the end
face, and the PART frame is built axis first (`datum_frame_axis`: z along A
from C into the tube, origin where A meets C, x towards B). The locate step
records the fitted radius and each ring's roundness over the probed arc; a
roundness above the declared 0.1 mm form tolerance plus 2 U of the CMM
(0.108 mm) stops the run and is recorded as a tube outside the cylinder model,
since re-seating cannot change the form. Each holder declares four datum
targets (1.5 in SMR nests that also take a coded photogrammetry adapter): the
tracker measures them to register FIXTURE in WORLD, and every camera image set
includes them, so image sets taken while different tapes are on the specimen
share one frame; the declared camera uncertainty covers a target coordinate in
that frame, registration included. A step that measures datum targets must use
a fixture that declares them (`fixture_targets_undeclared`).

**Path realization.** A path that a robot traces from the model would only
test the robot's reproduction of its own program. The protocols therefore
realize each path as the centreline of a 3 mm unsteered adhesive tape laid from
the start jig **JIG-START-01**, a declared fixture with one single-slot insert
per path: each 3.05 mm × 20 mm slot is offset laterally and rotated in heading
from the nominal slot as its path declares (0 and 2 mm lateral, 5 mrad
heading). The jig is located by two dowel pins against datum B and a stop
against C (on the tube: a vee foot, a pointer on scribe B and a stop against
C). Without in-plane steering the tape follows a geodesic (a physical
assumption, recorded `not_established`).

The tapes cannot lie on a specimen together. An offset tape starts 2 mm from
the nominal one, less than a tape width, and slots 2 mm apart would overlap in
one jig; on the coupon every lateral offset crosses the nominal tape at the
focus, and the heading tape stays within 0.54 mm of it. So each tape is laid
alone: the jig is seated with that tape's insert, the tape laid and the jig
removed; the CMM probes its start pose, 6 mm coded targets are placed on it and
imaged, the specimen is re-seated twice for replicates of those measurements,
and the targets and tape are removed before the next tape is laid. Every tape
is measured in the PART frame of the specimen's own datums. Each tape path
carries its predicted centreline (`centreline_mm`, 3D, about every 5 mm), and
the protocol a `tape_layout` (tape 3 mm, targets and markers 6 mm). The
validator walks the procedure's `lays`, `places` and `removes` and refuses two
slots of one laying less than one slot width apart (`jig_slots_overlap`), two
tapes on the specimen together whose centrelines come closer than a tape width
plus a target diameter (`tapes_overlap`), a tape centreline that passes a
specimen marker or a declared datum probe point closer than 6 mm
(`path_over_marker`), a tape never laid (`tape_not_laid`), and a repeat step
that lists a laying, placing or removing step (`repeat_includes_laying`).
Chords between sampled points never exceed surface distances, so these
clearances err on the side of refusal. The plate tapes run along y = 30 mm,
midway between two marker rows (at least 28 mm from any marker); the cylinder
helices start 72° from B on end face C and cross the rings between markers and
datum points (at least 15.2 mm clear).

The declared start-pose error of an offset tape relative to the nominal one
has two terms (1σ): its insert and the laying, 0.05 mm and 0.5 mrad, and the
re-seating of the jig, 0.01 mm and 0.1 mrad per seating, of which the relative
pose carries two, √2 × (0.01 mm, 0.1 mrad); together 0.052 mm and 0.52 mrad.
The jig is not checked before laying: the CMM probes each laid tape centreline
at s = 0 and 20 mm, so the realized start pose of the offset tape is budgeted
(open loop, both terms) or conditioned on (T138, T140).

Every protocol's refusal matrix covers seventeen mutations, and a tape
protocol's four more (two offset tapes in one laying, a tape left on the
specimen when the next is laid, a marker under a tape, a repeat that lists a
laying step), each with its own code: missing
markers; a filled hardware slot without acquisition fields; records in an empty
slot; a slot whose raw digest is not 64 hex characters or whose time is not ISO
8601 with an offset (`acquisition_malformed`); a slot that does not cite a T139
retention identity (`retention_record_missing`); a criterion marked
`accepted`; a prediction labelled `hardware_measured`; an instrument claimed
verified; acceptance declared inside the system; a step that names the start
jig without using a declared jig and a step that uses a device the protocol
does not declare (`undeclared_device`); an unnumbered free-text step
(`procedure_step_malformed`); an instrument without settings; a raw format
of an undeclared instrument; an acceptance criterion that names an undeclared
instrument (a surface distance "by tape-measure and laser interferometer");
a step that measures the datum targets of a fixture without any
(`fixture_targets_undeclared`); and a repeat step that repeats itself. A step,
a path realization or an acceptance criterion's statement or test that names a
device by word (jig, nest, V-block, CMM, camera or photogrammetry, tracker,
scanner, film, tape measure, interferometer) or by identifier (FX-, JIG-, VB-,
SB-, GS-, SG-) must name a declared device of that kind (for a step or path,
among the devices it uses), and a step may touch only declared datums
(`undeclared_datum`). A well-formed slot passes this schema
check only: whether its bytes came from an instrument is decided by the
runner's hardware gate, not by the protocol validator.

Declared instruments (1σ per coordinate; planning values only): camera
(photogrammetry) 0.02 mm, laser tracker 0.015 mm + 6 µm/m, CMM 0.002 mm, laser
line scanner 0.01 mm at 0.05 mm native spacing, unrolled-film gauge (flexible
film with a printed 0.5 mm scale, read under a 10× loupe) 0.05 mm. A
marker-pair distance has u_pair = √2 × 0.02 = 0.028 mm with the camera; a
measured cylinder gap (film minus chord) has √(0.05² + 0.028²) = 0.057 mm.
Declared relative start-pose error of an offset tape: insert and laying
0.05 mm and 0.5 mrad, jig re-seating √2 × (0.01 mm, 0.1 mrad), together
0.052 mm and 0.52 mrad (1σ); its CMM estimate has √2 × 0.002 = 0.0028 mm and
2 × 0.002/20 = 0.2 mrad.

**Raw formats of the captures.** Each control protocol defines the CSV its
instrument exports must follow, read by `manufacturing_records.read_capture`:
a header of `# key: value` lines (schema, protocol, instrument, frame, unit
`mm`, origin `measurement` or `synthetic`), then a table.
`ciw.lab-mfg-target-capture.v1` has columns `target,x_mm,y_mm,z_mm,u_mm`
(coordinates in the CAD frame through the declared PART → CAD placement, u the
declared 1σ per coordinate, which must be positive: a zero u would make E_n
infinite or undefined, so `read_capture` refuses it with `capture_value`);
`ciw.lab-mfg-distance-capture.v1` has `pair,distance_mm,u_mm`. Roles:
`photogrammetry` (the camera's marker or station-target coordinates; on the
cylinder only the 21 markers on the upper half), `cmm` (tape centreline
points `<path>-S0` and `<path>-S20`) and, on the cylinder, `film` (surface
distance per marker pair).
A synthetic capture starts with the line
`# SYNTHETIC CAPTURE - NOT A MEASUREMENT` and declares origin synthetic, and
only such a capture may; `to_acquisition` refuses its bytes as hardware
evidence. The origin of any other capture is the operator's declaration:
nothing authenticates it.

<a id="t126"></a>
## T126 — Flat-plate control (MFG-FLAT-PLATE-01)

5 × 5 coded markers at 60 mm pitch; nominal path N0 from (−120, 30) along +x
for 240 mm (midway between the marker rows y = 0 and 60), a 2 mm laterally
offset path and a 5 mrad heading-offset path, each laid, measured and removed
in turn.

Predictions: chord = geodesic for every marker pair. The geodesic distance is
found by shooting: the launch heading starts 0.1 rad off the chord direction
and Newton's method on the miss distance (with the Jacobi heading field as its
derivative) solves for it; the distance is the arclength where the converged
geodesic passes the other marker. Over 47 pairs the solved heading matches the
chord direction to rounding, the arclength matches the chord to rounding, and
the geodesic of that length closes onto the target. On the plane this is a
zero-curvature sanity check of the shooting machinery (RK4 and the Jacobi field
are exact there), not a test that could detect a curvature error. The Jacobi transfer is
[[1, s], [0, 1]]; offset-path separation δ + s sin δθ (2 mm constant; 0.3, 0.6,
0.9, 1.2 mm at 60…240 mm for 5 mrad). Hypotheses H1 (E_n ≤ 1 per marker pair)
and H2 (the heading-offset slope equals the start heading difference measured
by the CMM). The plate isolates instrument, frame and procedure error from
curvature. A camera export of the 25 markers bound to T138
(`--capture photogrammetry=`) is compared pair by pair (47 chords, E_n per
pair); the start-pose export (`--capture cmm=`) is read and retained but does
not yet condition the plate prediction.

<a id="t127"></a>
## T127 — Rolled-cylinder control (MFG-CYLINDER-01)

Three rings (z = 50, 150, 250) of 12 markers; the 21 on the upper half (within
90° of scribe B) are measured, and every marker pair uses them (circumferential
pairs are placed about B, for example C1-09 to C1-03 for 180°). The helix tape
HX45 and its 2 mm offset start 72° from B on end face C. The cylinder is intrinsically
flat: its Jacobi transfer along a 45° helix equals the plate's exactly. Its
extrinsic curvature makes chords shorter than geodesics:
gap(d) = d − 2R sin(d/2R) = d³/24R² − d⁵/1920R⁴ + …, zero on axial rulings.

| Pair | Geodesic (mm) | Gap (mm) |
| --- | --- | --- |
| circumferential 30° | 52.36 | 0.596 |
| circumferential 90° | 157.08 | 15.658 |
| axial 100 / 200 mm | 100 / 200 | 0 |

Counterexample: "on a developable part the chord between markers equals their
surface distance" fails at 90° (chord 141.42 vs geodesic 157.08 mm). Minimum
circumferential separation whose gap reaches k·u_pair (k = 2): camera 23.9 mm,
tracker 21.7 mm, CMM 11.1 mm. The two-term series is checked against the exact
gap with its alternating-series remainder bound as a signed margin
(|gap − series| − bound ≤ 0, retained with its negative value).

No step of a chord-only protocol measures a surface distance, so the gap is
measured with a second instrument: the unrolled-film gauge is laid from marker
centre to marker centre of each pair without tension or steering. Hypotheses:
H1 the film distance minus the camera chord equals the predicted gap
(E_n ≤ 1 with U_m = 2√(u_film² + u_pair²) = 0.1149 mm and
U_p = 2 |∂gap/∂R| u_R, the T140 radius term); H2 the offset helices separate as
on the plate; H3 the chord alone differs from the predicted geodesic distance
for circumferential pairs of 30° and more (the extrinsic signature; pairs below
the 23.9 mm resolvable arc are not tested). The tube is held in VB-01 and its
datum frame is built axis first; on noise-free synthetic CMM points of a tube
in a known pose (the protocol's 12 datum-A points, two scribe points, three
end-face points) the cylinder fit and `datum_frame_axis` recover the pose and
radius to rounding, and
a scribe on the axis, an end face parallel to it and a five-point fit are
refused (finding "The axis-primary datum frame ..."). Bound to T138
(`--capture photogrammetry= --capture film=`), the camera and film exports are
compared for H1 and H3.

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
start crosses (linear interpolation between the two stations whose separations
change sign, applied alike to measured and predicted values), within the T140
expanded uncertainty of the focal distance; H2 the measured separations follow
that conditioned Jacobi prediction (E_n ≤ 1 at stations 1–8, with U_p from dome
tolerances, start-pose estimate and solver) and reject the flat-plate
prediction at the last two stations. The tapes N, L and H are laid, measured
and removed one at a time (they cross or come within 0.54 mm of each other);
their station targets (N0–N8, L0–L8, H0–H8), imaged with the FX-321 datum
targets in every image set, and their start-pose points (N, L and H at s = 0
and 20 mm) are exported as the `photogrammetry` and `cmm` captures that T138
reads.

<a id="t129"></a>
## T129 — Surface metrology protocol: as-built dome and sampling density

**Protocol MFG-SCAN-01** (`protocol-surface-scan.json`, validated and mutated
like the other protocols) identifies the as-built coupon before MFG-COUPON-01.
Setup: laser line scanner (declared 0.01 mm point noise, 0.05 mm native
spacing, 100 mm standoff, incidence limit 30°; the coupon's steepest slope is
16.9°), two orthogonal raster passes over the chart extent. Targets: six
sphere-mounted registration targets off the dome (none occludes the crest
windows), measured by the CMM in the PART frame; each pass is registered to
them by Kabsch. Filtering: incidence and scanner flags only, no smoothing,
fit residuals beyond 5σ rejected and counted. Retained raw data: native point
clouds with scanner settings and digests, target coordinates, registration
transforms and residuals, fit parameters, covariance and residual map.
Hypotheses: H1 the coupon is a Gaussian dome of revolution (χ²/dof of the global
fit ≤ 1 + 3.09 √(2/dof) = 1.086 on the 2601-point grid); H2 the fitted height and
width lie within the forming tolerances; H3 the crest curvature of the local fit
equals h/σ² of the global fit.

**As-built identification.** The global fit
z = z₀ + a_x x + a_y y + h exp(−((x − x₀)² + (y − y₀)²) / 2σ²) (Gauss–Newton from
the nominal dome on the scan decimated to a 4 mm grid) recovers a perturbed dome
(h = 10.15, σ = 19.7, centre (0.4, −0.3) mm, base offset and tilt) exactly
without noise; on 200 seeded synthetic scans its spread matches σ²(JᵀJ)⁻¹ within
four standard errors (largest difference 8%), its mean is unbiased and χ²/dof
averages 1. The scan-derived covariance of (h, σ, x₀, y₀) adds a declared 20 ppm
scanner scale term and the target registration term: standard uncertainties
0.0016 mm (height), 0.0026 mm (width, correlation −0.59) and 0.005 mm (centre),
against 0.115 and 0.289 mm from the declared tolerances; T138 and T140 use it
for their scan-conditioned geometry terms. It holds only under the Gaussian
model: an elliptical dome with σ_x = 20.5 and σ_y = 19.5 mm, inside the declared
width tolerance, gives χ²/dof = 20.0 against the 1.086 threshold (counterexample
to "a dome inside the forming tolerances is described by the nominal Gaussian
model"), while the threshold flags 1 of 200 Gaussian scans (checked ≤ 2%). A
rejected model needs a prediction on a surface fitted to the scan, which is not
implemented. The protocol defines the raw formats of its exports (the CMM
target file, `ciw.lab-mfg-target-capture.v1` with rows RT1–RT6, and the
scanner's native point cloud per pass), but no task reads a bound scan yet: a
reader that runs `fit_dome` and the model test on it is open work.

**Local curvature sampling.** A least-squares fit z = a + b x + c x² over a centred window W with spacing d
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
RMS error. Registration of repeat scans on the six MFG-SCAN-01 targets (Kabsch):
the residual has 3N − 6 degrees of freedom and the rotation covariance
σ²(Σ|p|²I − ppᵀ)⁻¹, both confirmed by Monte Carlo.

<a id="t130"></a>
## T130 — Calibration artifacts and datum frames

* Gauge sphere Ø25.4 probed at 25 points on a 75° cap (CMM): geometric fit;
  Monte Carlo spread matches σ√diag((JᵀJ)⁻¹) within 2.1%.
* Step gauge 10…200 mm: m = (1 + e)L + b recovers a declared 50 ppm scale error
  and 1 µm offset exactly without noise; with noise the spread matches the
  linearized one within 3.2%.
* 3-2-1 datum frame: orthonormal, rigid-motion equivariant; tracker probing
  (0.015 mm) gives rotation std ≈ 100–130 µrad. Collinear primary points and a
  secondary direction normal to the primary plane are refused (refusal checks
  of the task).
* Frame chain with left perturbations T = exp(ξ)T̄:
  C = Σ Ad(T₁…T_{k−1}) C_k Ad(…)ᵀ, point covariance [I, −(Tp)^] C [I, −(Tp)^]ᵀ.
  Coupon far-corner std ≈ (0.045, 0.040, 0.045) mm; the WORLD → FIXTURE link
  dominates (0.062 mm RSS) through its rotation lever arm. First order agrees
  with exact SE(3) sampling within 3–4% (relative Frobenius, the Monte Carlo
  check); the stated uncertainty of the first-order std is its linearization
  error in mm, estimated by symmetric sigma points pushed through the exact
  SE(3) chain (4e−10 mm).

<a id="t131"></a>
## T131 — Repeatability and Gage R&R (ANOVA method)

Balanced crossed design, 10 parts × 3 operators × 3 replicates. Expected mean
squares give σ²_e = MS_E, σ²_po = (MS_PO − MS_E)/r, σ²_o = (MS_O − MS_PO)/(pr),
σ²_p = (MS_P − MS_PO)/(or). On 2000 synthetic studies with declared components
(part 0.050, operator 0.006, interaction 0.004, repeatability 0.010 mm) the raw
estimators are unbiased. True %GRR is 23.9%, but a single study's 90% interval
is [16.5%, 38.9%]: the raw GRR estimate is a linear combination of independent
scaled χ² mean squares, and its Monte Carlo mean and variance match the exact
values Σc²·2E[MS]²/df within their standard errors; the quantiles carry
distribution-free 95% order-statistic intervals in %. The operator component is
negative (truncated) in 10.8% of studies (F(2, 18) theory: 11.3%). Unbalanced or
incomplete data are refused. No real gage capability is established, and the
AIAG interaction-pooling rule is not applied.

**Procedure** (`gage-rr-procedure.json`, a plan): each protocol has one
specimen, so its ten "parts" are ten features of it whose true values span the
measured range (plate: ten marker pairs; cylinder: the ten T127 pairs; coupon:
ten gage targets at s = kL/9 on the lateral tape, laid once and left alone on
the coupon for the study, whose offsets from the nominal route span 2 to
−1 mm; separations between tapes are not stable parts, since the tapes are
never on the coupon together). Three operators, three replicate rounds; before every
round the specimen is removed, re-seated in its holder (FX-321, or VB-01 for the
tube), the datums re-probed and the
PART frame rebuilt; within a round each operator measures the parts in a
SHA-256-keyed pseudo-random order (the operator order is keyed the same way),
features are coded and operators do not see earlier readings. Because the
between-part spread is a spread of measurands, not of a process, %GRR is
reported against the declared tolerance (P/T = 6 s_GRR / T). Raw components are
reported, negative ones truncated and flagged, without pooling. A separate
type-1 study takes 25 readings of the SG-200 100 mm step by one operator in one
setup (bias and repeatability). Thresholds are hypotheses: %GRR (P/T) ≤ 10%,
ndc ≥ 5, Cg and Cgk ≥ 1.33.

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
tool-centre-point path X + Hn has speed factor √((1 − Hκ_n)² + (Hτ_g)²)
(Weingarten: dn/ds = −κ_n t − τ_g N). It is checked at every node against
|t + H dn/ds| with dn/ds from central differences of the unit normal one 1 µm
RK4 geodesic step either side (largest difference 1.8e−8 against 1e−7, both
tools), and segment by segment against the speed |ΔP|/Δs of the welding-torch
polyline (2.4e−4 against 2e−3; wrong factors such as |1 + Hκ_n| or 1 miss by
0.1 to 1). On the nominal route τ_g = 0 by symmetry, so the checks also run on a
route launched 10° off the axis, where Hτ_g reaches 0.098 (torch) and 0.78
(gun) and the τ_g term adds up to 4.7e−3 to the torch factor, far above the
pointwise tolerance (checked); there the torch sees 0.86–1.32. On the nominal
route a 15 mm welding torch sees 0.84–1.37;
a 120 mm spray gun exceeds the 94.5 mm concave radius (at the route nodes) on
the dome rim and its TCP path folds back (33 reversed segments) — a
counterexample to "offset tool paths of smooth surface paths are smooth".

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
worst ratio until every corner is within 0.9999 of the spec. That stop
condition is a convergence diagnostic, not evidence: every corner of every
derated box is re-evaluated by a second computation (base and perturbed
geodesics by ciw's adaptive Dormand–Prince integrator at the route nodes, from
an exp-map start coded separately from `jacobi.perturbed_start`, separation
evaluated inline), which must agree with the RK4 ratios within 1e−4 (they agree
to 2e−8) and stay within the spec. A wrong exact perturbation (for example scaled by 1.08) still
meets the loop's stop condition but fails the agreement (tested). The published
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

The **focus margin** is s_c − L, with s_c the first conjugate point (zero of
j_head) along the route, as T024, T025 and T032 define it. A route whose j_head
has no zero within its horizon (2.5 × its reach) has a censored margin: like
T024, T137 records it with no value (`focus_margin_mm: null`) and keeps the
horizon-dependent lower bound horizon − L under its own key
(`focus_margin_lower_bound_mm`, beside `horizon_mm`), so an aggregate cannot
show a bound as a margin. No candidate route has a conjugate point within its
horizon, and j_head(s)/s stays above 0.53 on (0, L] for every route, so every
focus margin is positive (lower bounds from 297.8 mm for 0° to 334.5 mm for
25°): each route is locally length minimizing to the edge, and under the focus
margin the six routes form one tied tier of censored margins.

What separates them is the **focal clearance ratio**: (nearest focal or
conjugate point, a zero of j_lat or j_head) / L, or horizon/L as a lower bound
when none lies within the horizon. Routes with only a lower bound are tied: the
ranking is a list of tiers, with the unresolved 15°, 20° and 25° routes sharing
the first tier (checked to lead: their smallest lower bound exceeds every
resolved ratio).

| Route | L (mm) | Nearest focus (mm) | Focal clearance ratio | Focus margin s_c − L (mm) |
| --- | --- | --- | --- | --- |
| 15°, 20°, 25° | 205–217 | none within horizon | ≥ 2.52 | ≥ 312 |
| 10° | 203.11 | focal 329.9 | 1.62 | ≥ 304.6 |
| 5° | 202.34 | focal 176.0 | 0.87 | ≥ 299.6 |
| 0° | 202.18 | focal 153.4 | 0.76 | ≥ 297.8 |

Counterexample (ranked by the focal clearance ratio): the shortest candidate
(0°) has the lowest focal clearance ratio, and the calibration ranking (T136)
puts the same route first. Its focus margin is positive: the straight route is
a local length minimum to the edge, and the second variation of length equals
the Jacobi index form j_head(L) j_head′(L) = 36.964 mm (Richardson finite
differences: 36.963 mm, relative difference 1.6e−5 against a 2e−4 tolerance);
the stated uncertainty is that residual in mm. Its focal point belongs to the
lateral start offsets (a zero of j_lat), which do not contradict minimality to
the edge line.

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

* Open loop (declared start-pose error: insert and laying 0.05 mm, 0.5 mrad,
  and two jig seatings of 0.01 mm, 0.1 mrad): k = 2 uncertainty 0.104 mm at the
  start, up to 0.322 mm at the end; T138 keeps the insert-and-laying and
  re-seating terms separately (`execution_terms_mm`).
* Conditioned on the CMM start pose (the prediction re-integrated from it):
  0.006 mm at the start, up to 0.300 mm at the end.
* Conditioned also on the as-built scan (MFG-SCAN-01, T129): the geometry term
  becomes gᵀCg with the dome sensitivities g (height, width, centre; the centre
  by moving the start) and the scan-derived covariance C, which removes most of
  the dome-tolerance term: 0.006 mm at the start, up to 0.045 mm at the end;
  valid only if the scan passes the Gaussian model test.

A correct model compared with a tape realized 0.104 mm off its nominal offset
(2σ of the declared open-loop start error; the "measurement" is the exactly
re-integrated route, model output standing in for a perfect instrument) fails
E_n ≤ 1 (max 1.88) when U_p covers only the dome tolerances and the solver, and
passes (max 0.86) once the start-pose term is included — a counterexample to "a correct model passes
against its open-loop prediction when U covers the instrument and the dome".
The measured branch passes too: re-integrating the prediction from a CMM
estimate of the realized start pose that is 2σ off in offset (±0.0057 mm) and
heading (±0.4 mrad), at all four sign corners, gives max E_n 0.46 against the
conditioned U_p.

Comparison rule E_n = |m − p|/√(U_m² + U_p²) ≤ 1. **No measured separation
exists**; `compare_separation` refuses absent measurements, a schema fixture (as
is and relabelled as a measurement) and digest mismatches. The same rule serves
the plate and cylinder controls through the pair comparator
`compare_pair_distances` (per marker pair: plate chord = geodesic with U_p = 0,
the flatness term being second order; cylinder chord-geodesic gap with
U_p = 2 |∂gap/∂R| u_R); T138 retains both pair tables and records their
refusals without a measurement. The success path of both comparators (E_n,
agreement, station and pair mismatches) is exercised by the tests on a
synthetic measurement-kind record that is never retained.

**Operator captures.** T138 reads the instrument exports an operator binds
with `ciw lab run T138 --capture photogrammetry=<targets.csv> [--capture
cmm=<start-pose.csv>] [--capture film=<film.csv>]` through `ctx.capture`, which
retains their bytes as `artifacts/T138/capture-<role>.csv`. `compare_captures`
parses them in the formats their protocol defines and compares:

* MFG-COUPON-01: each separation is the difference of the L and N targets at a
  station projected on the model's in-surface normal of the nominal route
  there (equal to the normal separation of the exactly offset route to
  second order); stations 1–8 are compared with the open-loop prediction and,
  with a `cmm` capture, with the prediction re-integrated from the estimated
  start pose (e(0) = δ, e(20) = δ j_lat(20) + δθ j_head(20), which avoids the
  δ (j_lat(20) − 1)/20 ≈ 0.1 mrad bias of a plain angle between the two 20 mm
  chords), plus the crossing arclength (H1): the measured crossing carries the
  capture's own u, propagated through the interpolation between the two
  bracketing stations (u(s*) = Δs √(v_{k+1}² u_k² + v_k² u_{k+1}²)/(v_k −
  v_{k+1})², about u_sep/|slope|), and the predicted one only the prediction
  terms of the T140 focal row (geometry, start pose, solver), not its declared
  instrument term, which the capture replaces;
* MFG-FLAT-PLATE-01: the 47 marker-pair chords;
* MFG-CYLINDER-01: the ten pair chords against the predicted chords and
  geodesics (H3) and, with a `film` capture, film minus chord against the
  predicted gaps (H1).

The result is the computational finding "Normalized errors of the bound
metrology captures against the prediction of their protocol, computed from
their unauthenticated bytes", whose basis lists the capture digests as inputs
(`authenticated: false`) and declares synthetic inputs when the captures say
they are synthetic; a capture the reader refuses (a malformed file, a zero u)
refutes it. Roles a protocol defines for retention only, such as a plate or
cylinder start-pose file, or a coupon start-pose file without the targets it
would condition, are parsed and retained, and the claim is recorded as
expected-unestablished with a note naming them. Without a capture (the
retained run) the claim is recorded as expected-unestablished too. The reader
and reductions are checked on noise-free synthetic captures from
`synthetic_capture` (a lateral tape started 2.1 mm and 0.4 mrad off): start
pose recovered to 1.2e−7 mm and 1.2e−6 rad, separations to 1.3e−4 mm, plate
and cylinder chords and gaps to rounding; eight malformed or relabelled
variants and a synthetic capture listed in a measurement record are refused.

The physical claims stay `not_established` whatever a capture shows: nothing
authenticates it and no metrology instrument probe exists (a capture role
without an instrument in `runner.CAPTURE_INSTRUMENTS` has no route to
`hardware_measured`). The deferred research question is a signed-capture trust
anchor (instrument-held keys that sign each export, verified by the workbench)
or a `hardware:metrology` probe of an instrument attached to the analysing
host. The task stays `partial` until then.

<a id="t139"></a>
## T139 — Retention of raw measurements, calibration and frame metadata (partial without a bound record)

The retention mechanism is exercised on a synthetic schema fixture. A real
acquisition reaches T139 as operator captures: the retention record under the
role `retention` (JSON) and its raw files under `photogrammetry`, `cmm`, `film`
or `scanner`:

```sh
ciw lab run T139 --capture retention=record.json --capture photogrammetry=targets.csv --capture cmm=start-pose.csv
```

T139 validates the record (`validate_retention`), requires a protocol of this
section, matches each raw entry to the bound bytes by SHA-256 (an entry with no
matching bytes is refused, `raw_digest_mismatch`), builds its acquisition fields
with `to_acquisition` (refusing a schema fixture, the fixture's markers and
synthetic-capture bytes) and retains the record, the raw files, the raw
manifest and the acquisition fields. The finding "A bound retention record
validates against the raw bytes bound with it and yields acquisition fields" is
computational: established when the record passes, refuted when it is refused,
expected-unestablished when none is bound. The task completes only for a
measurement record that matches bound bytes; with a refused record, or none
(the retained clean-room run), it stays `partial` and names the missing
acquisition, since a retention task whose retained material is synthetic or
absent is `partial`. The bytes are the operator's and unauthenticated: the
physical claim "A real measurement ... has been retained" stays
`not_established`, and a `hardware_measured` label also needs the metrology
probe or signed-capture trust anchor T138 defers.


`ciw.lab-measurement-retention.v1`: `record_kind` (measurement | schema_fixture),
protocol id, raw files (name, SHA-256, bytes, media type), instrument (id, kind,
serial), calibration (applied with reference, 64-hex digest and an ISO 8601
validity window, or not_applied), frame chain links (parent, child, proper
rotation, translation, 6 × 6 PSD covariance over (ρ, φ), source) with
continuity, and clock (source, ISO 8601 time with explicit offset,
synchronization, uncertainty). Eleven mutations are refused by code, including
a calibration digest that is not SHA-256 hex and an acquisition time outside the
calibration validity window. Covariance symmetry and positive semidefiniteness
are tested relative to the matrix scale: a rank-3 covariance with eigenvalues
up to 1e6 mm² and an explicit −1e−8 mm² eigenvalue (1e−14 of the largest, a
few times the eigensolver's rounding level of 1.3e−9 mm²) is kept although an
absolute 1e−12 mm² threshold would refuse it (by a factor of 1e4, checked), and
the same matrix with an
eigenvalue of −1e−3 of the largest is refused.

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
central differences over rectangular tolerances (u = a/√3), or gᵀCg with the
MFG-SCAN-01 dome covariance when conditioned on the as-built scan; execution =
start pose uncertainty × the Jacobi fields (open loop: the declared insert and
laying error and the re-seating of the jig for each tape, in quadrature, both
recorded as `execution_terms`; conditioned: the CMM estimate); solver = |Q(h) − Q(2h)|/15 for a reported h-solution and
(16/15)|Q(h) − Q(2h)| only when the reported value is the coarse one (the 6-step
control); the limiting term has a variance share > 50%, otherwise "mixed". Every
statement below holds under the declared instrument, start-pose and dome
uncertainties, which are themselves `not_established` calibration claims.

| Quantity | Value | Instrument | Geometry | Execution | Solver | Limiting |
| --- | --- | --- | --- | --- | --- | --- |
| cylinder gap, 90° pair (film − chord) | 15.66 mm | 0.057 | 0.009 | 0 | 0 | instrument |
| coupon separation at L, 5 mrad, open loop | 0.540 mm | 0.028 | 0.023 | 0.063 | 5e−9 | execution |
| coupon separation at L, 5 mrad, conditioned | 0.540 mm | 0.028 | 0.023 | 0.022 | 5e−9 | mixed |
| coupon separation at L, 2 mm lateral, open loop | −1.037 mm | 0.028 | 0.148 | 0.063 | 3e−8 | geometry |
| coupon separation at L, 2 mm lateral, conditioned | −1.037 mm | 0.028 | 0.148 | 0.022 | 3e−8 | geometry |
| coupon separation at L, 2 mm lateral, conditioned and scanned | −1.037 mm | 0.028 | 0.0019 | 0.022 | 3e−8 | instrument |
| coupon focal distance, conditioned | 153.4 mm | 1.3 | 4.4 | 0.83 | 9e−7 | geometry |
| coupon focal distance, conditioned and scanned | 153.4 mm | 1.3 | 0.056 | 0.83 | 9e−7 | instrument |
| plate separation at 240 mm, 5 mrad, open loop | 1.2 mm | 0.028 | 1.4e−7 | 0.135 | 0 | execution |
| plate separation at 240 mm, 5 mrad, conditioned | 1.2 mm | 0.028 | 1.4e−7 | 0.048 | 0 | execution |
| coarse-solver control (6 RK4 steps) | 0.625 mm | 0.028 | 0.023 | 0.022 | 0.094 | solver |

The open-loop execution terms split as insert and laying / jig re-seating:
0.060 / 0.017 mm (coupon, 5 mrad), 0.060 / 0.017 mm (coupon, 2 mm lateral) and
0.130 / 0.037 mm (plate), checked to add in quadrature to the execution
component. The plate geometry term models the declared 0.05 mm flatness as a
Gaussian bump centred under the route (u = 0.05/√3 mm, width 75 mm); its effect
is second order. Over the 240 mm plate
route even a CMM start-pose estimate leaves the 20 mm heading baseline dominant.
The coarse control validates the solver estimate: the actual error against a
256-step reference is 0.90 × the Richardson estimate. The start pose limits the
heading-offset and plate separations open loop; the 2 mm lateral separation at
the coupon route end is geometry-limited already open loop (from mid-route on,
geometry outweighs the open-loop start-pose term). Counterexample: under the
declared tolerances and instrument uncertainties the focal-distance prediction
is geometry-limited — a ±0.2 mm dome-height tolerance moves the focus by
millimetres — so scanning the as-built coupon (T129) matters more than a better
camera. Conditioning on the MFG-SCAN-01 covariance shrinks the geometry term of
both coupon quantities to 1.3% of the tolerance-based one (checked ≤ 10%), after
which both are instrument-limited; this holds only while the scan passes its
Gaussian model test and the declared scanner terms hold.

<a id="t141"></a>
## T141 — Production acceptance stays outside the system

`AcceptancePolicy.decide` always refuses (`production_acceptance_outside_system`)
and `record` returns `decision: not_performed`. Over 64 bases × 5 authority
domains (including hardware acquisition, passing checks and independent checks)
no claim filed in an authority domain is ever established; a forged
`hardware_measured` acceptance finding is refused; protocols refuse acceptance
inside the system and criteria marked `accepted`.

The domain of a claim is its author's declaration, so the label function alone
cannot see an acceptance statement filed elsewhere. T141 recorded that
"Coupon lot accepted for production" filed in `computational_pipeline` with one
passing check was labelled `numerically_verified` by `evidence.finding`. That
loophole is now closed for such wording: `evidence.screen_authority_claim`
refuses authority-outcome phrases in computational and physical claims in
every section, and T141 records the refusals of `evidence.finding` (the
acceptance and rejection statements in `computational_pipeline`, the
acceptance statement filed as `physical` with an acquisition record) and of
`validate_finding` (a hand-built record carrying the statement) — a
counterexample to "evidence.finding labels a claim from its basis and domain
alone". Every task of this section also passes its findings through
`screen_acceptance_language`, which refuses decision words (accepted, approved,
rejected, scrapped, quarantined, signed off, dispositioned, released for or to
production, passed or passes inspection or acceptance, certified for
production) in claims and string values outside the authority domains; a
refusal blocks the task. The task checks it on hand-built records and on
"Coupon lot scrapped", which the evidence screen passes. Both screens match
phrases, not meaning: the paraphrase "Coupon lot fit for shipment to the
customer" filed in `computational_pipeline` with a passing check is still
labelled `numerically_verified` (recorded as a counterexample to "the lab API
cannot mark production acceptance"), so "domain assignment of free-text claims
is machine-checked across the lab" stays `not_established`, and making the
domain derivable from the claim is the deferred research question. Production
acceptance, industrial readiness and customer demand ("manufacturers need
curvature-aware placement, winding, coating or inspection path checking") are
recorded as `not_established`.

## What must be measured

None of the following exists in the repository. Each is required before any
physical claim of this section can change label.

1. MFG-FLAT-PLATE-01: marker coordinates (camera, CMM), tape start poses (CMM
   at s = 0 and 20 mm), tape centreline points, plate flatness and temperature;
   instrument checks on SB-1000 and GS-25.4.
2. MFG-CYLINDER-01: the 21 markers on the upper half of the rings (camera,
   tracker), tube axis, radius and roundness at three rings (CMM, in VB-01),
   film surface distances of the ten marker pairs, helix and offset-helix tape
   start poses and centrelines.
3. MFG-SCAN-01 and MFG-COUPON-01: the as-built coupon scan (registration
   targets, both raster passes, the global dome fit with its model test and the
   crest windows: 13.7 mm at ≤ 0.65 mm spacing for 5% curvature), tape start
   poses (CMM), station targets on the nominal, lateral and heading offset
   tapes, the crossing arclength of the lateral offset tape.
4. Calibration records for the camera, tracker, CMM and scanner; gauge sphere,
   step gauge and scale bar certificates; the frame chain with its measured
   covariances; the realized start-pose error of the jig inserts, the tape
   laying and the re-seating of the jig.
5. A real Gage R&R study (10 × 3 × 3) and type-1 study following
   `gage-rr-procedure.json` (T131).
6. Process data for placement (tow positions, gaps/overlaps, wrinkling at the
   steering radius), winding (slip versus abs(κ_g/κ_n), friction coefficient)
   and coating/welding (standoff, lateral error, deposited thickness).
7. Scanner footprint on the coupon at the planned standoff.

Every measurement is exported in its protocol's raw format, bound to T138 as an
operator capture (station separations on the coupon, marker pairs on the plate
and cylinder), described by a retention record bound with its raw files to T139
(`--capture retention=`), which validates the record against those bytes, and
retained with `ciw lab hardware retain`. Until a metrology instrument probe or a signed-capture trust
anchor exists, those comparisons stay computational and the physical claims
`not_established`. Acceptance of parts or processes remains an external
decision (T141), and customer demand is not surveyed here.

## Counterexamples recorded

| Task | Refuted statement |
| --- | --- |
| T127 | On a developable part the chord between markers equals their surface distance |
| T129 | A window from the osculating circle bounds the quadratic-fit bias on any convex profile |
| T129 | A smaller fitting window always improves the curvature estimate |
| T129 | A dome inside the declared forming tolerances is described by the nominal Gaussian model |
| T132 | A helix programmed in machine angles is insensitive to mandrel radius error |
| T133 | A constant winding angle is geodesic and slip-free on every mandrel of revolution |
| T134 | The standoff tool path of a smooth surface path is itself smooth |
| T135 | Geodesic rows at the swath spacing cover a curved coupon as they cover a plate |
| T136 | Allocating spec/(2 max abs(j)) to each error source keeps the exactly perturbed route within the spec |
| T137 | The shortest route is the one with the largest focal clearance ratio (farthest, relative to its length, from a focal or conjugate point) |
| T138 | A correct model passes E_n ≤ 1 against its open-loop prediction when U covers only the instrument and the dome tolerances |
| T140 | Curved-surface predictions compared with photogrammetry are instrument-limited |
| T141 | evidence.finding labels a claim from its basis and domain alone, so an acceptance statement filed in a computational domain is established (closed by `evidence.screen_authority_claim`) |
| T141 | The lab API cannot mark production acceptance (a paraphrase outside both screened vocabularies is still labelled by its checks) |
