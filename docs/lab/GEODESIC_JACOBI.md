# Geodesic and Jacobi-field experiments (T001–T009)

Section 1, part 1 of the computational-experimentalist queue. Implementation:
`src/ciw/lab/geodesic_jacobi.py` (tasks) and `src/ciw/lab/geodesic_jacobi_common.py`
(declared paths, closed forms, sympy derivations, mpmath references, the pinned
provider bridge). Tests: `tests/test_lab_geodesic_jacobi.py`.

Run and read:

```
python -m ciw lab run T001 T002 T003 T004 T005 T006 T007 T008 T009 --output-dir out \
    [--provider csg=<Curved-Surface-Geodesic-Sensitivity-Runtime checkout>]
python -m ciw lab report T007 --retained out
```

**Scope and non-claims.** Every surface, path, length and curvature here is a
declared mathematical object in normalized units. The tasks establish agreement
between implementations, convergence rates and invariants of *computations*.
They do not measure, calibrate or certify any physical surface, tool path,
vehicle or sensor. Where a task invites a physical conclusion (T005, T009) that
conclusion is retained as a `physical`-domain finding with no admissible basis,
so its label is `not_established`.

## Surfaces, charts and declared paths

The catalogue (`ciw.lab.surfaces.catalogue()`): plane, unit sphere (polar chart
θ, φ), unit cylinder (φ, z), saddle z = (x² − y²)/2, torus R = 2, r = 1
(φ, θ), gaussian bump z = ½ exp(−(x² + y²)/2), and the upper half-plane with
g = I/y² (K = −1). Two extra charts, `plane-polar` and `cylinder-polar`, view
the flat surfaces through the polar map (r, t) ↦ (r cos t, r sin t), which
gives flat geometry nonzero Christoffel symbols. For R = 1 the cylinder's
development is isometric to the plane, so the two polar charts carry the same
metric and their integrations coincide; they differ only in the embedding used
to measure distances.

Standard paths (`STANDARD`) and constant-curvature paths (`SPECIAL`) are listed
in `geodesic_jacobi_common.py` with start point, heading (from the first
coordinate direction, in the metric's orthonormal frame) and length. Every
integration starts from the same binary64 state `start_state(key)`.

## T001 — the geodesic equation, derived

For a chart u = (u¹, u²) with metric g_ij, the first variation of length (or
energy) gives

    u''^k = −Γ^k_ij u'^i u'^j,   Γ^k_ij = ½ g^kl (∂_i g_jl + ∂_j g_il − ∂_l g_ij).

For an embedding X(u), g_ij = X_i·X_j and ∂_k g_ij = X_ki·X_j + X_i·X_kj.
Gaussian curvature follows either from the second fundamental form,
K = (LN − M²)/(EG − F²), or intrinsically (Brioschi, or R_1212/det g from the
Riemann tensor); their agreement is the Theorema Egregium. By hand:

| Chart | Metric | Nonzero Γ | K |
| --- | --- | --- | --- |
| plane (x, y) | I | none | 0 |
| sphere (θ, φ), radius R | diag(R², R² sin²θ) | Γ^θ_φφ = −sin θ cos θ, Γ^φ_θφ = cot θ | 1/R² |
| cylinder (φ, z), radius R | diag(R², 1) | none | 0 |
| Monge z = f(x, y) | δ_ij + f_i f_j | Γ^k_ij = f_k f_ij / W², W² = 1 + f_x² + f_y² | (f_xx f_yy − f_xy²)/W⁴ |
| saddle, f = c(x² − y²)/2 | as Monge | Γ^x_xx = c²x/W², Γ^y_yy = c²y/W², Γ^x_yy = −c²x/W², Γ^y_xx = −c²y/W² | −c²/(1 + c²(x² + y²))² |
| gaussian bump, f = h e^(−ρ²/2σ²) | as Monge | as Monge | h² e^(−ρ²/σ²)(1 − ρ²/σ²)/(σ⁴ W⁴) |
| torus (φ, θ) | diag((R + r cos θ)², r²) | Γ^φ_φθ = −r sin θ/(R + r cos θ), Γ^θ_φφ = (R + r cos θ) sin θ / r | cos θ / (r (R + r cos θ)) |
| upper half-plane (x, y) | I/(k² y²) | Γ^x_xy = −1/y, Γ^y_xx = 1/y, Γ^y_yy = −1/y | −k² |
| polar (r, t) of the plane | diag(1, r²) | Γ^r_tt = −r, Γ^t_rt = 1/r | 0 |

The geodesic equations are the rows read through u''^k = −Γ^k_ij u'^i u'^j,
for example on the torus φ'' = 2 r sin θ φ'θ'/(R + r cos θ) and
θ'' = −(R + r cos θ) sin θ φ'²/r; on the half-plane x'' = 2x'y'/y and
y'' = (y'² − x'²)/y. The bump curvature is positive for ρ < σ and negative
outside, with maximum h²/σ⁴ = 0.25 at the summit.

The table is not only prose: `hand_geometry(key, u, v)` in
`geodesic_jacobi.py` codes every row (metric, Γ, geodesic acceleration, K; the
torus and half-plane accelerations as written above) from the surface
parameters alone, and T001 compares it with `ciw.lab.surfaces` at the seeded
points, so a transcription error in the table or in the implementation fails a
check.

**Experiment.** Sampling boxes are `ciw.lab.surfaces.SAMPLING_DOMAINS` plus
r ∈ [0.5, 2.5] on the polar charts, 8 seeded points per chart. The hand table
is compared with `ciw.lab.surfaces` (analytic checks). With sympy installed,
`derive(key)` rebuilds metric, Christoffel symbols, geodesic accelerations and
Brioschi curvature symbolically from each embedding (or intrinsic metric); they
are lambdified and compared with `ciw.lab.surfaces` (independent check, checker
`sympy`). Independently of sympy, ciw's own Christoffel symbols are checked for
symmetry, metric compatibility ∇g = 0 and exact metric derivatives against
central differences, and the intrinsic curvature from finite-differenced
Christoffel symbols is compared with the closed-form K. The catalogue surfaces
return closed-form curvatures, so the Theorema Egregium is tested through the
generic second-fundamental-form route (`EmbeddedSurface.gaussian_curvature`,
called unbound) against the finite-difference Riemann curvature and, with sympy,
the Brioschi curvature, on the six embedded charts. `derivations.txt` and
`derivations.tex` (a standalone LaTeX document) retain the symbolic output.

**Result.** Largest sympy/ciw and hand-table/ciw discrepancy 3.2e-16
(relative); largest intrinsic-curvature and extrinsic-intrinsic discrepancy
5.5e-8 (finite-difference truncation). Counterexamples: the polar charts have
|Γ| up to 2.4 with K ≡ 0 (Christoffel symbols do not imply curvature), and the
cylinder has Γ ≡ 0 in (φ, z) with normal curvature 1 (bending in space does not
imply nonzero Γ). The headline label is the weakest established one:
`numerically_verified` (three findings are `independently_verified` against
sympy). Without sympy the task is `partial`, every finding is
`numerically_verified`, and the derivation is checked only through the hand
table and the ciw-only checks.

## T002 — reference solutions

* **Closed forms** (all constant curvature, so the transfer matrix is also
  closed form): great circle X(s) = cos(s/R) X₀ + R sin(s/R) T₀; straight lines
  in the flat (φ, z) and polar developments; half-plane semicircles
  x = c + ρ tanh t, y = ρ / cosh t with t advancing at rate k; transfer matrix
  [[cn_K, sn_K], [−K sn_K, cn_K]].
* **Arbitrary precision** (saddle, torus, gaussian bump): the sympy-derived
  geodesic + Jacobi system (8 states, Brioschi K) is lambdified with
  `modules="mpmath"` and integrated at 34 digits by Gragg–Bulirsch–Stoer
  (modified midpoint, step sequence 2, 4, …, 16, full polynomial
  extrapolation) with 10 and 20 macro-steps; the difference is the retained
  error estimate. The start is the binary64 ciw start converted exactly, so the
  reference differs from the ciw integration only in the integrator and in the
  independently derived equations.
* Compared against it: ciw `richardson_rk4` (200/400 steps) and scipy
  `solve_ivp(DOP853, rtol=1e-13, atol=1e-15)` on the ciw right-hand side.

**Checks.** Gaps to the reference must be at most 1e-12 (the expected
invariant), and at most a tenth of the Richardson error estimate wherever that
estimate exceeds 1e-13: without extrapolation (or with a wrong factor) the RK4
end state misses the reference by about the estimate itself, so a gap far below
it shows that the extrapolation produced the agreement.

**Result.** mpmath self-estimates 1.4e-24 (saddle), 4.4e-23 (torus), 6.4e-22
(bump); largest ciw-vs-reference gap 6.4e-14; gap over the Richardson estimate
0.001–0.017; scipy DOP853 within 2e-14 of every reference; torus Clairaut drift
|ρ²φ' − C₀| is 6.8e-14 for the ciw end state and below 1e-20 for the reference.
Fallbacks: without sympy/mpmath the variable references come from scipy DOP853
applied to the ciw right-hand side — an independent integrator, but the
equations are not independently checked (recorded as an unresolved
assumption); without scipy as well, from a finer ciw Richardson run
(self-convergence only). Either fallback makes the task `partial`.

## T003 — integrator orders

Endpoint error against the T002 reference (embedded distance; chart distance
on the half-plane). Steps: Euler 64–512, midpoint 32–256, RK4 24–192, adaptive
Dormand–Prince rtol = atol ∈ {1e-8, …, 1e-12}.

| Method | Fitted orders on the seven curved charts |
| --- | --- |
| Euler | 1.001–1.008 |
| midpoint | 1.982–2.039 |
| RK4 | 3.888–4.023 (bump slightly pre-asymptotic; pairwise slopes retained) |
| DP5(4) vs evaluations | per chart 4.61–5.86 (artifact only), median 5.39 (claimed) |
| DP5(4) vs tolerance | per chart rtol^0.87…0.99 (artifact only), median 0.971 (claimed) |

The adaptive checks must separate two hypotheses: DP5(4) advancing with the
fifth-order solution (local extrapolation: error ∝ evaluations⁻⁵ ∝ tol) and a
pair that advances with y4 (error ∝ evaluations⁻⁴ ∝ tol^0.8). The thresholds
sit half-way: median effective order ≥ 4.5 (and ≤ 6) and median tolerance
exponent ≥ 0.9 (and ≤ 1.1). Only medians over the seven charts are claimed,
because one accept/reject decision moves a single chart's slope by up to about
0.3 (the per-chart values stay in `orders.json`). As a falsification, T003 runs
`dormand_prince_y4` — the same tableau and controller advancing with y4 — and
records that it fails both thresholds (medians 4.20 and 0.77). Loose tolerances
(1e-6, 1e-7) are excluded from the adaptive fit: the controller's start-step
ramp dominates the evaluation count there. Counterexample: on the flat
Cartesian charts (plane, cylinder) Γ ≡ 0, every method reproduces u₀ + s v₀,
and no order is observable (errors ≤ 2.1e-14).

## T004 — unit-speed drift, no renormalization

g(v, v) is a first integral; its drift is measured, never corrected. Fitted
drift orders: Euler 0.98–1.02, midpoint 1.98–2.03, RK4 3.91–4.02. Evidence that
nothing renormalizes: a speed-1.3 start keeps g = 1.69 to 2.1e-8 (RK4, 128
steps) and 6e-11 (adaptive, rtol 1e-10), and stays at least 0.6 away from 1
under Euler and midpoint too; an AST scan of every state-update function
(`integrators.step_*`, `integrate_fixed`, `richardson_rk4`,
`integrate_adaptive`, `jacobi.rhs`, `jacobi.transfer`, `Surface.geodesic_rhs`,
`Surface.christoffel`) finds no norm-like or square-root call (except the
allow-listed RMS error norm of the adaptive controller, which scales the step,
not the state), no ±½ power, no division by a sqrt/norm/hypot/abs expression
and no division by a name bound to one; y' = y grows to e²|y₀|. The scanner is
itself tested on normalizations written through an intermediate name, a −0.5
power and a reciprocal. Counterexample: Euler renormalized to unit speed after
every step keeps |g − 1| ≤ 4.4e-16 but still misses the great-circle endpoint
by 1.4e-2 at N = 128, with a measured first-order error (order 1.005 over
N = 64…512) — unit speed does not certify accuracy. The initial state is
normalized by design (`unit_tangent`); that is initial data, not a hidden
correction.

## T005 — the Jacobi separation law

Along a unit-speed geodesic a normal Jacobi field J = jN obeys j'' + K j = 0.
For constant K the heading column (j(0) = 0, j'(0) = 1) is
sn_K(s) = sin(√K s)/√K, s, sinh(√−K s)/√−K and the lateral column
(j(0) = 1, j'(0) = 0) is cn_K. On the torus both equators are geodesics
(θ'' ∝ sin θ = 0) with K = 1/(r(R + r)) = 1/3 (outer) and
K = −1/(r(R − r)) = −1 (inner).

Paths: sphere great circle (L = 7), plane, cylinder, half-plane (L = 3), torus
outer equator (L = 11.5), inner equator (L = 6.5); fixed RK4 with step ≤ 0.015
and at twice the step. The model-space K comes from the surface parameters
(1/R², −k², 0, and the equator formulas), never from `gaussian_curvature`,
which the integrated Jacobi equation itself uses. Largest relative error of Φ
against the model law 2.7e-9; step-halving orders 3.98–4.00; K stays constant
to rounding along both equators.

The law is also tested on geodesics rather than on the scalar equation: on the
sphere and the half-plane, closed-form geodesics started from the exact lateral
offset or heading rotation of `jacobi.perturbed_start` are compared with the
unperturbed closed form by exact intrinsic distance
(2R asin(chord/2R); (2/k) asinh(|Δ|/(2√(y₁y₂)))). d(γ_ε(s), γ₀(s))/ε
approaches |sn_K(s)| and |cn_K(s)| with an O(ε²) remainder (sin(d/2) =
sin(ε/2)|sin s| on the unit sphere): order 2.00 on the sphere and 1.99 on the
half-plane, remainder 1.6e-6 and 4.2e-4 at ε = 0.01.

**Provider comparison (optional).** With `--provider csg=<checkout>`, the
checkout is verified with `ciw.lab.runner.git_identity` against the pin shared
with `ciw.geodesic_reference.PINS['curved-path-transfer']` (revision
`bbc535af29c30997e56fd120320c570830676462`, tree
`181b6eb73288d001f45c39bb149b1a80a431f34b`, clean) before and after execution;
a mismatch is refused (`CSG_REVISION_MISMATCH`, `CSG_TREE_MISMATCH`,
`CSG_CHECKOUT_DIRTY`, `CSG_CHECKOUT_UNREADABLE`) and the task is `partial`.
The provider runs in a subprocess (`sys.executable -c <bootstrap> <checkout>/src`)
and returns `integrate_jacobi` traces plus `TransferMap` matrices, determinants
and focus events for the same arclength grids, both from its RK4 trace and from
its closed-form `constant_curvature_transfer`. The independent check (checker
`Curved-Surface-Geodesic-Sensitivity-Runtime@bbc535a…`) compares the ciw
integrated Φ with the provider's closed form: 2.7e-9, the RK4 error. The two
RK4 traces agree to 4.1e-15. That comparison is between two origins running the
same method on the same grid; the contract's `cross_implementation` kind is
defined for same-origin pairs, and it is used here only for want of a kind for
this case, with the caveat stated in the check's reference text (the label
comes from the independent check either way). The provider output is refused
(`CSG_EXECUTION_FAILED`) unless it answers every requested case in the expected
shape. The checkout is also refused as dirty when any untracked file other than
bytecode caches sits under its `src/` (ignored files included), since the
bootstrap puts that directory first on `sys.path`.

**Refusals.** Before verification, the pin-stage code a checkout calls for is
predicted from direct git queries (`rev-parse`, `status`, `ls-files`); the
refusal finding compares that prediction with the code `verify_csg_checkout`
raises. Execution-stage refusals (`CSG_EXECUTION_FAILED`,
`CSG_CHANGED_DURING_EXECUTION`) cannot be predicted and are checked by
membership, stated as such. Reports carry only the code and a fixed sentence;
the message, with the checkout path replaced by `<checkout>`, is retained in
`provider-refusal.json`. T008 records the same refusal finding.

## T006 — Jacobi columns against finite differences

Perturbed starts are exact geometric operations (`jacobi.perturbed_start`):
lateral offset along the normal geodesic with the tangent parallel transported,
or heading rotation. Separation is g(δu, N) at matched nodes. On the sphere,
torus and bump (RK4, 160 steps, ε = 0.08 … 0.01): central differences converge
at order 1.99–2.00, one-sided at 0.95–1.03. Counterexample: for the one-sided
heading difference the error is smallest near ε = 1e-7 (1.3e-8) and grows to
1.1e-4 at ε = 1e-11 (cancellation) — a smaller step is not always better.

## T007 — Wronskian and transfer-matrix determinant

Φ' = AΦ with A = [[0, 1], [−K, 0]], tr A = 0, so det Φ ≡ 1 (Liouville). One
step of each method on the linear part:

* Euler: [[1, h], [−hK_n, 1]], det = 1 + h²K_n **exactly**. Euler is not
  area-preserving where K ≠ 0 (growth for K > 0, shrinkage for K < 0); verified
  per step to 4.8e-16 on every path. For constant K, det Φ_n = (1 + h²K)ⁿ.
* Midpoint: det = 1 + h²(K_mid − K_n)/2 + h⁴K_nK_mid/4. Summing, the O(h²)
  part telescopes: det Φ(L) − 1 = (h²/4)(K(L) − K(0)) + O(h³). Verified: ratio
  to the prediction 0.957–1.004 at N = 200 (checked within 0.05 of 1), the gap
  halving with h, and the extrapolated ratio 2r(h) − r(2h) within 1.1e-4 of 1
  (checked at 1e-3); order 2 on variable curvature and order 3 when
  K(L) = K(0) (constant K: (1 + h⁴K²/4)ⁿ).
* RK4: det − 1 = −h⁶K³/72 + h⁸K⁴/576 for constant K. For smooth K(s) (stage
  offsets e₂h², e₃h² included) sympy gives no h¹…h⁵ terms and the h⁶
  coefficient −(4k₀³ − k₀k₂ + 2k₁² + 4k₀(e₂ + e₃))/288. Hence the determinant
  drifts at **O(h⁵)** globally, one order above RK4's O(h⁴) error — measured
  4.99–5.02 on all nine curved paths. This refutes the naive expectation that
  the determinant drifts at the method's global order (recorded as a
  counterexample).
* Adaptive DP5(4): drift decreases like rtol^0.98…1.33 per path (artifact
  only; single accept/reject decisions move a path's slope). The claim is the
  median, 1.03, checked at ≥ 0.9 — half-way between the 0.8 of a pair
  advancing with y4 and the 1 of local extrapolation, the same threshold as in
  T003.
* K = 0 (plane, cylinder): every method keeps det Φ = 1 exactly.

## T008 — conjugate and focal points

Conjugate points are zeros of j_head after s = 0; focal points of the initial
normal geodesic are zeros of j_lat; both located by cubic-Hermite root finding
on (j, j'). Sphere: π R, 2π R and π R/2, 3π R/2 for R = 1 and R = 2 (error
1.2e-8 at 320 steps, order 4.00). Torus outer equator: π√(r(R + r)) = π√3,
2π√3 and half-way points (error 1.7e-8, order 4.00). None on the inner equator
or the half-plane, where Sturm comparison gives j_head ≥ s and j_lat ≥ 1
(verified for s > 0, where both differences are strictly negative, so the
signed checks are not satisfied trivially by the node s = 0). Sturm
upper-curvature bound: with K ≤ K_max = 1/3 on the torus, no conjugate point
occurs before π√3; on six seeded torus geodesics (L = 12) two reach a conjugate
point, the first at 7.79 (margin 2.34). Seeded bump geodesics leave the
positive-curvature cap before focusing, so four declared chords through the
summit region (start (−8, 0), (−8, 0.2), (−10, 0), (−10, 0.2), heading 0,
L = 30) exercise the bump bound 2π = π/√(h²/σ⁴): all four reach a conjugate
point, the first at 18.5 (margin 12.2 — the bound holds but is far from tight
there). Each surface must have at least one geodesic that reaches a conjugate
point, otherwise the check fails rather than passing vacuously. Zero locations
move by at most 3e-8 between rtol 1e-9 and 1e-10, with no count change.
Counterexample: on variable curvature the first focal point is not half the
first conjugate distance (3.47 against 3.89); if no seeded geodesic provided a
witness, the finding would be recorded as unestablished, not raised. With the
provider bound, ciw zeros match the focus events of the provider's closed-form
transfer (independent check) and of its RK4 trace (same method, different
origin), with equal zero counts.

## T009 — lateral and heading columns separately

Endpoint normal displacement = j_lat(L) δ⊥ + j_head(L) δα. Where along the
path curvature matters differs between the columns. Varying j'' + K j = 0 with
the Green function G(L, s) = j_lat(s) j_head(L) − j_head(s) j_lat(L),

    δj(L) = −∫₀ᴸ G(L, s) j(s) δK(s) ds,

and on constant curvature G(L, s) = sn(L − s). The lateral weight
sn(L − s) cn(s) is largest for early curvature; the heading weight
sn(L − s) sn(s) (s(L − s) on a flat background) is symmetric about mid-path and
vanishes at both ends — the heading column is *not* late-weighted. Checked with
a curvature bump (amplitude 1e-3, width 0.3) at s = 0.5, 1.5, 2.5 on a flat path
of length 3: direct RK4 agrees with the kernel integrals to 5.7e-5 (relative);
the lateral response at 0.5 is 4.94 times that at 2.5; the heading responses
at 0.5 and 2.5 agree to 1.7e-11 of the mid-path response.

The heading symmetry is in fact exact, not first order: reversing a path
(s → L − s) maps Φ(L) to D Φ(L)⁻¹ D with D = diag(1, −1), so j_head(L) is
unchanged and j_lat(L) becomes j_head'(L). Verified by integrating four
reversed geodesics (torus witness pair, gaussian bump, saddle) from their end
points: 1.2e-11. Heading sensitivity therefore cannot distinguish a path from
its reverse; lateral sensitivity can.

Model spaces: j_head/j_lat = tan(√K L)/√K, L, tanh(√−K L)/√−K (verified; on the
half-plane both columns approach e^L/2). Over 16 declared paths the two
rankings have Kendall τ = 0.23 (43 discordant pairs of 120). Witness at equal
length (L = 3, same torus): `torus-outer-to-inner` (K > 0 first) has |j_lat|
0.857 and |j_head| 3.907, `torus-inner-to-outer` (K < 0 first) has 1.803 and
2.846 — lateral error ranks the second path worse, heading error ranks the
first worse. The lateral reversal follows the early weighting; the heading
difference comes from the two curvature profiles not being mirror images (by
reciprocity, exact mirrors would tie), which the first-order kernels explain
only qualitatively (`kernels.svg` plots them on the witness paths). Central
differences (ε = 1e-3) confirm the endpoint sensitivities to 1.9e-6.

## Limits and open questions

* Agreement with sympy, mpmath, scipy or the CSG provider is independent
  *implementation* agreement on declared equations; it is not physical
  validation and not review by another party.
* The mpmath reference is an extrapolated integration whose error estimate is
  empirical (macro-step halving), not a proof.
* Orders are least-squares fits over four halvings; fixed-step tolerances
  were set to cover the observed pre-asymptotic spread and are declared in each
  check. The adaptive thresholds (4.5, 0.9) come from the two competing
  theories, not from the data; per-chart adaptive slopes are not claimed.
* Every numerical finding carries an uncertainty object (`kind`, `value`,
  `basis`): reference error of the mpmath or closed-form reference, RK4
  truncation estimated by step halving, the spread of pairwise log-log slopes
  around a fitted order, or binary64 rounding.
* The RK4 O(h⁶) per-step determinant defect assumes smooth K along the stage
  points; curvature discontinuities (meshes, CAD patches) were not tested.
* Deferred research question: characterize the leading RK4 determinant
  coefficient −(4k₀³ − k₀k₂ + 2k₁²)/288 along whole paths, and whether a
  Wronskian-preserving (symplectic) integrator for the Jacobi block changes
  conjugate-point accuracy near foci (T010–T011).
* Physical claims (T005 separation of real trajectories, T009 which error
  dominates a real tool or vehicle path) are recorded as `not_established`.
