# Geodesic/Jacobi experiments, part 2: limits, invariance and counterexamples (T010–T018)

Implementation: `src/ciw/lab/geodesic_jacobi_limits.py` (tasks) and
`src/ciw/lab/geodesic_jacobi_limits_core.py` (chart maps, perturbation
families, closed forms, constant-curvature transfer matrices). Tests:
`tests/test_lab_geodesic_jacobi_limits.py`. Run with

```
python -m ciw lab run T010 T011 T012 T013 T014 T015 T016 T017 T018 --output-dir <dir>
python -m ciw lab report T010 --retained <dir>
```

The retained reports are the source of truth for every number; the values
quoted below are from one run on Linux/Python 3.11/NumPy 2.4 and are given
only to make the argument readable.

**Scope and non-claims.** All surfaces, lengths, curvatures and perturbations
are declared, normalized test objects from `ciw.lab.surfaces`. Nothing here
measures a physical surface, calibrates a sensor or certifies an operating
envelope. Where a task invites such a conclusion, the claim is recorded in its
physical or authority domain and is `not_established`. Agreement between two
`ciw` computations is reported as numerical verification against a closed
form, an invariant or self-convergence — never as independent verification.
An optional SciPy DOP853 cross-check (T018) is retained as an artifact only, so
no label depends on an optional module.

Notation: along a unit-speed geodesic, `j'' + K j = 0`; `j_head` has
`j(0) = 0, j'(0) = 1`, `j_lat` has `j(0) = 1, j'(0) = 0`. A heading
perturbation `ε` gives separation `d(s) = ε j_head(s) + r(ε, s)`.

## T010 Near-focus and post-focus counterexamples

*Prediction.* Where `j` vanishes (a conjugate point `s*` for `j_head`, a focal
point for a mixed field) the first-order separation is zero and the true
separation is the remainder, generically `O(ε²)`; the relative first-order
error therefore diverges like `1/|s − s*|`, and past `s*` the separation
changes sign with `j` (image inversion). Two symmetric exceptions follow from
the geometry:

* Unit sphere, pure heading: all great circles through a point meet again at
  `s = π`, and the chord is exactly `2 sin(ε/2) |sin s|`. The relative
  first-order error is `2 sin(ε/2)/ε − 1 ≈ −ε²/24` at *every* `s` — it does not
  diverge at the conjugate point.
* Torus `(R, r) = (2, 1)` outer equator (`K = 1/3`, `s* = π√3`): the reflection
  `θ → −θ` makes the normal offset odd in `ε`, and the `O(ε²)` along-track lag
  `−ε² sin(2ωs)/(4ω)` (from Clairaut's relation) vanishes at `s*`. Hence
  `d(s*) = O(ε³)`.
* The sphere with a combined perturbation (lateral `ε`, heading `ε`) has
  `j = ε(cos s + sin s)`, zero at `3π/4`, where the exact chord is `ε²/2 + …`.

*Protocol.* RK4 on grids whose node 400 is `s*` (580 steps); `ε ∈ {0.005,
0.01, 0.02, 0.04}`; embedded chord and its component along the base geodesic's
in-surface normal; sphere runs compared node by node with great circles.

*Result.* Equator: chord exponent 3.000 at `s*`; generic torus path (start
`(0, 0.3)`, heading 0.5, `s* = 6.0514`): exponent 2.02. Divergence slope
−0.99 (generic) and −1.01 (equator); relative error 3.8 one step before `s*`
at `ε = 0.04`. After `s*` the signed separation equals `ε j` to 0.4 % with
`j < 0`. `chord(s*)/chord(s*/2) = 1.6e−4` on the equator. Sphere relative
error uniform at −1.667e−5 (`ε = 0.02`), including one step from `π`.

*Counterexamples.* "The separation of neighbouring geodesics grows
monotonically with length"; "the first-order separation is accurate along the
whole path once ε is small"; "the separation at a conjugate point is of exact
order ε²" (equator); "the relative first-order error diverges at every
conjugate point" (sphere).

*Does not prove.* Genericity of the `O(ε²)` case is argued from one
non-symmetric geodesic, not sampled; chords are extrinsic (intrinsic distance
differs at `O(d³)`).

## T011 Coordinate-change invariance

*Prediction.* With the pullback metric `g' = Jᵀ g(φ(a)) J` and the initial
tangent `t_a = J⁻¹ t_u`, the geodesic, its length and the transfer matrix are
chart independent; the integration error is not. A near fold
`u = c + μa + a³/3` keeps `det J ≥ μ > 0` but makes the chart velocity
`~ 1/μ` over an arclength window `~ μ^{3/2}`, so its fixed-step error grows as
`μ` decreases.

*Protocol.* Plane, unit sphere and `Torus(2, 1)` paths; charts centred at the
rounded chart midpoint of each path: polynomial warp, quadratic shear,
exponential stretch, near-fold (`μ = 0.2, 0.1`), identity. RK4 with
`N = 64, 128, 256`; DP45 at `rtol = 1e−10`; references exact (plane, sphere) or
DP45 at `rtol = 1e−13` (torus).

*Result.* Converged (adaptive) agreement in every chart within 1.3e−10 for the
endpoint, length and transfer matrix; RK4 order ≥ 3.95 in every chart; the
identity chart reproduces the base chart bit for bit. The `μ = 0.1` fold
multiplies the `N = 256` error by 1.1e6 (sphere) and 9.8e4 (torus); smooth
charts change it by 1.4–17×. The plane is exact to roundoff in its base chart
and 1.2e−5 in the fold chart.

*Counterexample.* "A change of chart that preserves the geometry leaves the
fixed-step integration error unchanged."

*Does not prove.* The fold amplification is measured, not derived; its
scaling law in `μ` is open.

## T012 Frame-change invariance

*Prediction.* `Rotated(base, R)` changes only the embedding (`X' = R X`); the
metric, second fundamental form and hence the chart ODE are unchanged in exact
arithmetic. Rotating the reference basis by `β` with heading `h − β` gives the
same tangent. A left-handed basis `(e1, −e2)` maps "+ε" to "−ε" in the
right-handed convention, so signed separations flip sign.

*Result.* Rotations (sphere, torus, saddle, bump; three rotations): max chart
and Jacobi state difference 1.8e−15, embedded endpoints rotate exactly to
1.4e−15, curvature differences 2e−15. Basis rotations: 1.8e−15. Orientation
reversal: separation ratio −1.000000000002. A reflection matrix is refused by
the core ("Frame change requires a proper rotation matrix").

*Counterexample.* "Signed Jacobi separations are invariant under every change
of orthonormal tangent basis" (only up to orientation).

## T013 Flat and developable limit

*Prediction.* `K = 0` on the cylinder, so `j_head(s) = s` exactly, as on the
plane, while a helix of angle `α` has the shorter chord
`√((2R sin(L cos α/(2R)))² + (L sin α)²)`. On the outer equator of `Torus(R, 1)`,
`K = 1/(R + 1)` and `L − j_head(L) = K L³/6 − K² L⁵/120 + …`, exponent −1 in
`R`; the chord deficit of that circle is `L³/(24 (R + 1)²)`, exponent −2.

*Result.* Plane and cylinder Jacobi columns identical bit for bit; chords 3.000
(plane) versus 2.538 (cylinder). Torus deviation matches the closed form to
1.8e−9 relative; fitted exponent (R = 16…1024) −0.9847, equal to the
closed-form value over the same radii; chord-deficit exponent −1.974.

*Counterexamples.* "Surfaces with identical Jacobi fields have identical chords";
"intrinsic and extrinsic signatures of curvature vanish at the same rate in the
flat limit". The physical claim about real workpieces is `not_established`.

## T014 Geodesic reversal and path truncation

*Prediction.* Flipping `(v, j')` inverts the flow. For a one-step method with
local error `C h^{p+1}`, the step with `−h` has error `C(−h)^{p+1}`; the
forward-then-reverse composition cancels at order `h^{p+1}` when `p` is even.
So the return error is `O(h)` for Euler, `O(h³)` for explicit midpoint and
`O(h⁵)` for RK4 (linear check: `R(z)R(−z) = 1 − z²`, `1 + z⁴/4`,
`1 + z⁶/72 + …`). Truncating and continuing on an identical grid is the same
arithmetic as direct integration.

*Result.* Reversal orders 1.05, 3.00, 5.00 (means over sphere, torus,
hyperbolic plane); adaptive return error ≤ 0.95 × rtol. Dyadic continuation
(`h = 2⁻⁷`) is bitwise identical for all three methods; decimal truncation
lengths (first witness `L1 = 1.13`, `L2 = 3`, `N2 = 300`) change the step by
one ulp and break bitwise identity at the 4e−16 level. Adaptive restart agrees
to 0.011 × rtol but not bitwise.

*Counterexamples.* "Forward-then-reversed integration with a method of order p
returns with error ∝ h^p"; "truncate-and-continue is bitwise reproducible for
any truncation length".

## T015 Long-horizon drift

*Prediction.* Speed, torus Clairaut constant and sphere angular momentum are
first integrals. Adaptive local-error control accumulates a secular drift
(exponent 1); a constant speed error produces position error `~ L` on the
sphere, a linearly growing one `~ L²`.

*Protocol.* `Torus(2, 1)` (bounded oscillation about the outer equator) and a
unit-sphere great circle; `h = 1/8` for Euler, midpoint, RK4; DP45 at
`rtol = 1e−6`; envelopes at `L = 10 … 320`; fixed-step runs stop at a
nonfinite state or at the sphere chart's poles.

*Result.* Adaptive energy exponents 0.93 (torus) and 1.00 (sphere); sphere
position exponents 1.02 (RK4) and 2.05 (adaptive). The RK4 energy envelope is
nearly flat over this horizon (exponents 0.15 and 0.007; a secular part appears
on the torus after `L ≈ 100`). Euler on the torus keeps the speed error below
3.5 % yet its Clairaut constant drifts across the separatrix and the orbit
winds around the tube (`max |θ| = 285` against the turning latitude 1.37);
Euler on the sphere reaches the pole of the chart at `s = 14`.

*Counterexamples.* "The energy error of a non-symplectic fixed-step integrator
grows linearly at every horizon"; "a bounded speed error implies a
qualitatively correct long-horizon geodesic".

*Does not prove.* Asymptotic drift laws beyond `L = 320`; symplectic or
symmetric integrators are not compared.

## T016 Strongly negative curvature

*Prediction.* On `K = −k²`, `j_head = sinh(kL)/k`. RK4 applied to the growing
mode has `R(z) = e^z (1 − z⁵/120 + …)`, so the relative error is
`≈ L k⁵ h⁴/120` and the steps for relative accuracy `τ` are
`N = L (L k⁵/(120 τ))^{1/4} ∝ k^{5/4}`. The Jacobi linearization has
eigenvalues `±k` (ratio 1): the step is limited by accuracy on the growing
mode, not by stability of a fast decaying mode, so this is intrinsic
exponential instability, not classical stiffness. Implicit midpoint has
`R(z) = e^z (1 + z³/12 + …)` (steps `∝ k^{3/2}`) and a pole at `kh = 2`.
Along the saddle ridge `y = 0`, `K = −c²/(1 + c²x²)² ≈ −1/(4s²)` away from the
saddle point, so growth is a power law, heuristically `j_head(L) ∝ c^{√2}`.

*Result.* Adaptive `j_head(L)` within 2.6e−10 of `sinh(kL)/k` for
`k = 1, 2, 4, 8`. RK4 error exponent 4.94 (measured/predicted 0.90–1.02).
Required RK4 steps 23, 54, 127, 303 (exponent 1.24); implicit midpoint 832 …
18476 (exponent 1.49); DP45 81 … 505 (exponent 0.88). Accuracy needs 23–50×
the RK4 stability limit. The RK4 geodesic endpoint error grows from 2.2e−9
(`k = 1`) to 0.46 (`k = 8`). Implicit midpoint at `kh = 2.29` changes sign six
times in seven steps. Saddle: `j_head(L)` = 3.2, 14.3, 105.8, 794, 5775 for
`c = 1 … 256` (fitted exponent 1.44 for `c ≥ 16`), DP45 steps grow by about
48 per factor 4 in `c` (logarithmically).

*Counterexamples.* "An implicit (A-stable) integrator removes the step
restriction on strongly negatively curved surfaces"; "Jacobi growth is
exponential in √(peak |K|) × length".

*Does not prove.* The `√2` saddle exponent is a heuristic far-field argument.

## T017 Validity domains of the first-order approximation

*Prediction.* `r(ε, s) = d(s) − ε|j(s)| = C₂(s) ε² + C₃(s) ε³ + …` with `d` the
unsigned separation. The first-order prediction is within relative tolerance
`τ` while `|C₂ ε + C₃ ε²| ≤ τ |j(s)|`. Closed forms: unit sphere, pure heading,
`C₂ = 0`, `C₃ = −|sin s| cos² s/24`, `ε_max = √(24τ)/|cos s|` — no collapse at
`s = π`; hyperbolic plane, `C₂ = 0`, `ε_max = √(24τ)/cosh s`. Generic paths have
`C₂(s*) ≠ 0`, so `ε_max ∝ |s − s*| → 0`; symmetric ones (outer equator) have
`C₂ ≡ 0` and `ε_max ∝ √|s − s*|`.

*Result* (`τ = 0.01`). Generic torus: `C₂(s*) = 1.52`, `ε_max(s*) ≈ 1e−12`,
`ε_max(s*/2) = 0.041`; a new integration at `0.9 s*` gives relative error
0.0050 at `ε_max/2` and 0.020 at `2 ε_max`, confirming the predicted boundary.
Sphere, pure heading: `ε_max = 0.490` at `0.999π` and `1.001π`; C₃ matches its
closed form to 8e−5. Hyperbolic `ε_max` 0.475 … 0.080 for `s = 0.25 … 2.5`
(within 0.35 % of the closed form). Sphere lateral+heading: `C₂(3π/4) = 0.500`.
Equator: `max |C₂| = 5e−6`, cubic remainder at `s*`.

*Counterexample.* "The validity domain of the first-order approximation
shrinks to zero at every conjugate point." Using these domains to certify
machine motion is a `machine_safety` claim and is `not_established`.

## T018 Curvature signal versus integrator error

*Prediction and refutation.* The working expectation was that weak curvature
is the hard case. For `j'' + K j = 0`, however, the `K`-free part `j = s` is
integrated exactly, so the Euler and midpoint errors are themselves
proportional to `K`: the resolvability ratio `S/E` (signal
`S = |j_head(L) − L| ≈ |K| L³/6`) is independent of `K` as `K → 0`; the RK4
error enters at `O(K² h⁴)`, so its ratio grows like `1/K`. The actual limit is
floating point: once `S` approaches `ulp(L)` the computed deviation is rounded
away.

*Protocol.* Thirteen length-2 paths (plane, cylinder, spheres of radius 1 to
1e8, two tori, saddle, two bumps, hyperbolic plane); Euler, midpoint, RK4 with
`N = 4 … 128`; true deviations from cancellation-free closed forms or DP45 at
`rtol = 1e−12`; resolved when `S/E ≥ 10` for every finer step.

*Result.* RK4 resolves every truncation-limited signal from `N = 16`
(minimum ratio 1.2e5). Euler needs `N = 32` (64 on the hyperbolic plane),
midpoint `N = 4` (8), on every such surface regardless of `K`: at `N = 16` the
Euler and midpoint ratios for `K = 1e−8` and `K = 1e−2` agree within 2 %,
and the RK4 ratio grows 98× from `K = 1e−2` to `1e−4`. At `K = 1e−16` the
computed deviation is exactly zero for every method and step (ratio 1), and at
`K = 1e−14` refining RK4 lowers the ratio from 63 to 9.9. Flat surfaces show
no spurious signal.

*Counterexamples.* "Weaker intrinsic curvature is harder to resolve at a fixed
step size"; "refining the step size always makes a nonzero curvature effect
resolvable". Transfer to measured sensor data is a `sensor_performance` claim
and is `not_established`.

## Requested follow-ups

* A symmetric (reversible) integrator in `ciw.lab.integrators` would make the
  reversal test exact and give a long-horizon comparison (T014, T015).
* A `Surface.check` scale that is invariant under anisotropic charts would
  allow tori with `R ≥ 10⁶`; today `det g / tr(g)²` refuses them as degenerate.
* Hyperbolic-plane isometries (Möbius maps) as a `ChartMap` would extend T012 to
  intrinsic surfaces.
