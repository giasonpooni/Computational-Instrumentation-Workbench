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
A reference computed by the same `ciw` integrator at a tighter tolerance is
labelled `self_convergence`, never `high_precision`. An optional SciPy DOP853
cross-check (T018) is retained as an artifact only, so no label depends on an
optional module; it integrates `ciw`'s own right-hand side
(`ciw.lab.jacobi.rhs`), so it checks the time stepper, not the geometry, and
could not support an independent geometric claim even where SciPy is present.

Notation: along a unit-speed geodesic, `j'' + K j = 0`; `j_head` has
`j(0) = 0, j'(0) = 1`, `j_lat` has `j(0) = 1, j'(0) = 0`. A heading
perturbation `ε` gives separation `d(s) = ε j_head(s) + r(ε, s)`.

## T010 Near-focus and post-focus counterexamples

*Prediction.* Where `j` vanishes (a conjugate point `s*` for `j_head`, a focal
point for a mixed field) the first-order separation is zero and the true
separation is the remainder, generically `O(ε²)`; the relative first-order
error therefore diverges like `1/|s − s*|` on both sides of `s*`, so past `s*`
the first-order prediction recovers as `|j|` grows again, and the separation
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
in-surface normal; sphere runs compared node by node with great circles. On
the equator `s* = π√3` is analytic; on the generic path `s*` is located on a
separate 1024-step probe, so its vanishing first-order term on the 580-step
grid is a self-convergence check, not an analytic one. Every torus family is
repeated on a grid with half the steps, and each finding's grid uncertainty is
measured at the configuration and nodes it compares.

*Result.* Equator: chord exponent 3.000 at `s*`; generic torus path (start
`(0, 0.3)`, heading 0.5, `s* = 6.0514`): exponent 2.02. Divergence slope
−0.99 (generic) and −1.01 (equator); relative error 3.8 one step before `s*`
at `ε = 0.04`. Past `s*` the relative error falls again with slope −1.01
(generic) and −0.99 (equator): 3.8 one step after `s*`, 0.018 at `1.25 s*`
(`ε = 0.04`). After `s*` the signed separation equals `ε j` to 0.4 % with
`j < 0`. `chord(s*)/chord(s*/2) = 1.6e−4` on the equator. Sphere relative
error uniform at −1.667e−5 (`ε = 0.02`, largest deviation from uniform
7.7e−8), including one step from `π`. Halving the torus grid changes the
smallest separation at `s*` (equator, `ε = 0.005`) by 1.5e−4 relative, the
signed separations compared for the inversion (`ε = 0.01`, `s*/2` and
`1.25 s*`, both paths) by at most 5.5e−11, and `chord(s*)/chord(s*/2)` by
9.5e−6 relative.

*Counterexamples.* Near focus: "the separation of neighbouring geodesics grows
monotonically with length"; "the first-order separation is accurate along the
whole path once ε is small"; "the separation at a conjugate point is of exact
order ε²" (equator); "the relative first-order error diverges at every
conjugate point" (sphere). Post focus, each witnessed after `s*`:
"neighbouring geodesics stay on the side of the base geodesic they start on"
(both torus paths at `1.25 s*`); "the image of a family of geodesics keeps its
orientation beyond a conjugate point" (sphere, signed ratio −1 between `3π/4`
and `5π/4`); "once the first-order prediction has failed at a conjugate point
it stays invalid beyond it" (generic torus path, relative error 3.8 at
`s* + h` and 0.018 at `1.25 s*`).

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

The torus reference is the same DP45 at a tighter tolerance, so its check is a
self-convergence check.

*Result.* Converged (adaptive) agreement in every chart within 1.3e−10 for the
endpoint and transfer matrix and 1.9e−10 for the length; the identity chart
reproduces the base chart bit for bit. Between `N = 128` and 256 the RK4 error
falls at least like `h^{3.7}` in every chart (smallest order 3.95): the smooth
charts show order 4 within 0.05, while the near-fold charts are still
pre-asymptotic, with orders 4.4–5.5 above 4, so fourth order is not
demonstrated for them at these steps. The bound is not claimed for the coarser
pair: between `N = 64` and 128 the `μ = 0.1` fold on the plane has order 3.68.
The `μ = 0.1` fold multiplies the `N = 256` error by 1.1e6 (sphere) and 9.8e4
(torus); smooth charts change it by 1.4–17×. The plane is exact to roundoff in
its base chart and 1.2e−5 in the fold chart.

*Counterexample.* "A change of chart that preserves the geometry leaves the
fixed-step integration error unchanged."

*Does not prove.* The fold amplification is measured, not derived; its
scaling law in `μ` is open.

## T012 Frame-change invariance

*Prediction.* `Rotated(base, R)` changes only the embedding (`X' = R X`); the
metric, second fundamental form and hence the chart ODE are unchanged in exact
arithmetic. A reference basis rotated by `β` is built through `ciw`'s metric:
`f1 = cos β e1 + sin β e2` and `f2 = N(f1)`, the metric +90° rotation that
`ciw` uses for Jacobi normals. It must be orthonormal, heading `h − β` in it
must be the unit tangent of heading `h`, and heading perturbations `±ε` stated
in it must be the geometric heading perturbation, so that
`(d(+ε) − d(−ε))/(2ε) = j_head + O(ε²)` with `d` the normal separation
`g(δu, N)` of the integrated geodesics. Merely relabelling the start tangent
(`h − β` in a locally rotated basis) would integrate the same ODE twice and
could not detect a wrong metric, normal or geodesic equation; comparing the
basis-stated perturbations with `ciw`'s Jacobi field can. A left-handed basis
`(e1, −e2)` changes two conventions at once:
its heading `−h + ε` is the right-handed heading `h − ε`, and its +90° normal is
`−N`. A "+ε" stated in the left-handed basis is therefore the geometric
perturbation `−ε`; measured along the right-handed normal `N` its separation has
the opposite sign. Measured along its own normal `−N` it agrees with the
right-handed separation to first order only: with
`d(ε) = ε j + C₂ ε² + …` along `N`, the left-handed run along `−N` is
`−d(−ε) = ε j − C₂ ε² + …`, so the Jacobi (first-order) separation is unchanged,
the `ε²` parts have opposite signs, and `own/right − 1 = −2 C₂ ε / j + O(ε²)`.
The finite separations agree exactly only where the separation is odd in `ε`:
on the unit sphere it is exactly `sin(s) sin(ε)`, so no `ε²` term enters either
ratio there, while on a surface without that symmetry the ratio along `−N` is
`1 + O(ε)`.

*Result.* Rotations (sphere, torus, saddle, bump; three rotations): max chart
and Jacobi state difference 1.8e−15, embedded endpoints rotate exactly to
1.4e−15, curvature differences 2e−15. Basis rotations (sphere, torus, saddle,
hyperbolic plane; four angles): the rotated basis is orthonormal to 4.4e−16,
gives the start tangent to 2.2e−16, and heading perturbations `±10⁻⁴` stated in
it reproduce `ciw`'s `j_head` to 2.1e−8 relative (the central-difference and
RK4 mismatch); a deliberately wrong Christoffel model raises that gap to order
one. Orientation on the unit sphere (`ε = 10⁻³`, `L = 2`, RK4 `N = 64`): the
right-handed separation matches `sin(L) sin(ε)` to 8.4e−12; a left-handed "+ε"
measured along `N` gives the ratio −1.000000000002 (along `−N` the same number
negated, +1.000000000002, recorded in the same finding rather than as a second
one). The 2.5e−12 residual is RK4 truncation of the `+ε` and `−ε` runs. Orientation
on the generic `Torus(2, 1)` path (start `(0, 0.3)`, heading 0.5, `L = 3`, RK4
`N = 64`): `own/right − 1` = −4.763e−4, −4.753e−3, −1.885e−2 for
`ε = 10⁻³, 10⁻², 0.04`, log-log slope 0.997 in `ε`; doubling `N` changes the
gap by 1e−7 relative, so it is the `ε²` term of the separation, not truncation.
A reflection matrix is refused by the core ("Frame change requires a proper
rotation matrix").

*Counterexample.* "Finite-`ε` signed separations are invariant under an
orientation-reversing basis change expressed consistently": on the torus path
the separation along the left-handed normal differs from the right-handed one
by `4.8e−4` relative at `ε = 10⁻³`, proportional to `ε`. What is invariant is
the first-order (Jacobi) separation; the exact agreement on the unit sphere is
a consequence of its odd separation, and the sign flip under mixed conventions
is recorded as its own convention finding.

## T013 Flat and developable limit

*Prediction.* The cylinder's second fundamental form has only the `φφ` entry,
so `K = 0` and `j_head(s) = s` exactly, as on the plane, while a helix of angle
`α` has the shorter chord `√((2R sin(L cos α/(2R)))² + (L sin α)²)`. On the
outer equator of `Torus(R, 1)`, `K = 1/(R + 1)` and
`L − j_head(L) = K L³/6 − K² L⁵/120 + …`, so the deviation has exponent −1 in
`R + 1` up to the relative correction `K L²/20`; the equator is a circle of
radius `R + 1` with chord deficit `2(R + 1)(x − sin x)`, `x = L/(2(R + 1))`,
`= L³/(24 (R + 1)²) + …`, exponent −2 in `R + 1`. Both closed forms are
evaluated without cancellation (a series for `x − sin x` below 0.1).

*Protocol.* `Plane` and `Cylinder` return a literal `K = 0`, so their bitwise
agreement only shows that the same Jacobi arithmetic runs on both; it cannot
fail for any geometric or integration error and is kept as context in the
flatness finding, not as a verified finding of its own. Flatness itself is tested by
integrating the cylinder again with `K` recomputed from its second fundamental
form (`SecondFormCurvature`). Tori with `R = 2 … 1024`, RK4 `N = 64`,
`L = 2`; exponents fitted for `R ≥ 16`, in `R` and in `R + 1`.

*Result.* Plane and cylinder Jacobi columns identical bit for bit (context
only); the second-form `K` along the helix is exactly 0 and `max |j_head(s) − s|` is
2.7e−15; chords 3.000 (plane) versus 2.538 (cylinder). The torus deviation
matches its closed form to 1.8e−9 relative and the chord deficit to 5.1e−9.
Fitted exponents: deviation −0.98472 in `R` (equal to the closed-form value over
the same radii) and −0.99749 in `R + 1`; chord deficit −1.9744 in `R` and
−1.99997 in `R + 1`. Of the 0.015 offset of the deviation exponent from −1,
0.013 comes from `K = 1/(R + 1)` rather than `1/R` (the fit of `K L³/6` alone)
and 0.0025 from the `K L²/20` correction; the chord-deficit offset in `R` is the
same `R/(R + 1)` effect.

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
one ulp and break bitwise identity at the 4e−16 level. The witness records that
the step sizes differ and that the result differs; how many final components
differ depends on last-bit rounding of `sin`/`cos` and is kept in the artifact
only. Adaptive restart agrees
to 0.011 × rtol but not bitwise.

*Counterexamples.* "Forward-then-reversed integration with a method of order p
returns with error ∝ h^p"; "truncate-and-continue is bitwise reproducible for
any truncation length".

## T015 Long-horizon drift

*Prediction.* Speed, torus Clairaut constant and sphere angular momentum
`M = X × X'` are first integrals. Adaptive local-error control accumulates a
secular drift (exponent 1). On the unit sphere `|M|` is the speed and `M/|M|`
the normal of the orbit plane, so a position error splits into a cross-track
part, bounded by the tilt of `M/|M|`, and an along-track (phase) part. A speed
error advances the point along its circle by the integrated excess arclength
`∫(|X'| − 1) ds`; a speed error growing `~ L` therefore gives an along-track
error `~ L²`. A tilt of the orbit plane growing `~ L` gives a cross-track error
`~ L`.

*Protocol.* `Torus(2, 1)` (bounded oscillation about the outer equator) and a
unit-sphere great circle; `h = 1/8` for Euler, midpoint, RK4; DP45 at
`rtol = 1e−6`; envelopes at `L = 10 … 320`; fixed-step runs stop at a
nonfinite state or at the sphere chart's poles.

*Result.* Adaptive energy exponents 0.93 (torus) and 1.00 (sphere); sphere
position exponents 1.02 (RK4) and 2.05 (adaptive). The two sphere laws have
different mechanisms. Fixed-step RK4: at `L = 320` the position error 5.41e−4
is cross-track (5.41e−4), matching the orbit-plane tilt 5.46e−4, which grows
with exponent 1.01 while the size of `M` drifts only 2.4e−6; the along-track
part −2.6e−5 has the opposite sign to the integrated speed error +1.6e−4, so a
speed-error (phase) explanation is refuted there: the RK4 error is a
precession of the great-circle plane. Adaptive DP45: the position error
1.02e−2 is along-track (−1.021e−2) and equals the integrated speed error
(−1.023e−2, exponent 2.05), confirming the speed-error mechanism for the `L²`
law. The adaptive torus Clairaut drift has local slopes 1.33, 1.16, 0.93,
1.05, 1.00, so its exponent is fitted where they agree, from `L = 40`: 0.999
(spread 0.07). The RK4 energy envelope is flat up to `L = 160` on the torus
(exponent 0.03 over `L = 10 … 160`) and up to `L = 320` on the sphere
(exponent 0.007). On the torus its local slopes are 0, 0, 0, 0.14 and 0.86, so
a secular term emerges between `L = 160` and 320; a single exponent over all
six checkpoints (0.15) would hide it. Fixed-step Clairaut envelopes are not
described by one exponent (Euler saturates, local slopes 1.3 down to 0.005;
RK4 local slopes rise from 0.18 to 0.74), so only the adaptive Clairaut
exponent is claimed. Euler on the torus keeps the speed error below 3.5 % yet
its Clairaut constant drifts across the separatrix and the orbit winds around
the tube (`max |θ| = 285` against the turning latitude 1.37); Euler on the
sphere reaches the pole of the chart at `s = 14`.

*Counterexamples.* "The energy error of a non-symplectic fixed-step integrator
grows linearly at every horizon" (witness: torus RK4 over `L = 10 … 160`);
"a bounded speed error implies a qualitatively correct long-horizon
geodesic"; "the fixed-step RK4 position error on the sphere is the phase error
of its speed error".

*Does not prove.* Asymptotic drift laws beyond `L = 320`; symplectic or
symmetric integrators are not compared.

## T016 Strongly negative curvature

*Prediction.* On `K = −k²`, `j_head = sinh(kL)/k`. A one-step method of order
`p` whose stability function on the growing mode is
`R(z) = e^z (1 + c z^{p+1} + …)` has relative error `N |c| (kh)^{p+1}` after `N`
steps, so the steps for relative accuracy `τ` are
`N(τ) = L (L |c| k^{p+1}/τ)^{1/p} ∝ k^{(p+1)/p}`. RK4 has `c = −1/120`
(`N ∝ k^{5/4}`); implicit midpoint `c = 1/12` (`N ∝ k^{3/2}`, a pole at
`kh = 2` and a negative step factor beyond); the 2-stage Gauss–Legendre method
(the (2, 2) Padé approximant, implicit, A-stable, order 4) `c = −1/720`
(`N ∝ k^{5/4}`, no real pole), so at equal order it needs
`(120/720)^{1/4} = 0.639` times the RK4 steps at every `k`. The Jacobi generator `[[0, 1], [−K, 0]]` has
eigenvalues `±k` (a derivation: evaluated at the literal `K = −k²` that
`HyperbolicPlane` returns, it could not fail, so the rates are measured from
the integrated flow instead): the step is limited by accuracy on the growing
mode, not by stability of a fast decaying mode, so this is intrinsic
exponential instability, not classical stiffness, and A-stability cannot remove
the `k^{(p+1)/p}` growth. Comparing implicit midpoint (order 2) with RK4 (order
4) alone would confuse order with implicitness; the Gauss–Legendre method
separates them.

Saddle ridge (matched asymptotics, not a proof). Along the ridge `y = 0` of
`Saddle(c)`, `K = −c²/(1 + c²x²)²`. For `c|x| ≫ 1` the arclength from the saddle
point is `s ≈ c x²/2`, so `K ≈ −1/(4s²)`, independent of `c`. Then
`j'' = j/(4s²)` has the power solutions `|s|^a` with `a(a − 1) = 1/4`,
`a± = (1 ± √2)/2`. The field starts at `s = −1` with `j = 0, j' = 1`; inbound
to the core `|s| ~ 1/c` the `a−` mode grows by `c^{−a−}`, the core of width
`~1/c` passes a generic mixture, and outbound to `s = 1` the `a+` mode grows by
`c^{a+}`. Hence `j_head(L) ∝ c^{a+ − a−} = c^{√2}`: polynomial in `c`, although
`√(peak |K|) L = cL` grows without bound.

*Protocol.* `HyperbolicPlane(k)`, `k = 1, 2, 4, 8`, `L = 2`: adaptive DP45 at
`rtol = 1e−10`, RK4 with `N = 128`, and the minimal `N` for relative accuracy
`1e−6` by doubling and bisection on the exact step matrices of RK4, implicit
midpoint and Gauss–Legendre. The matrix-power step counts are checked against
the scalar stability functions on the eigenmodes (`1e−10`) and, for RK4, against
the full geodesic/Jacobi integration (`1e−12`). The growth and decay rates
are measured from the eigenvalues `e^{±kL}` of the integrated transfer matrix
`Φ(L)` (both Jacobi columns, DP45 `rtol = 1e−10`); the decaying eigenvalue is
used only where it exceeds 100 × rtol × the largest entry of `Φ(L)`, since
below that the integration error of the growing mode swamps it. The
exponential growth rate is also fitted across `k` from the adaptive `j_head`,
not from the closed form. `Saddle(c)` ridge geodesic from arclength 1 before the saddle
point, `c = 1 … 16384`, DP45 at `rtol = 1e−9` checked at `1e−11`; the mirror
symmetry `x(L) = −x0` is checked (`y = 0` holds by construction of the start).

*Result.* Adaptive `j_head(L)` within 2.6e−10 of `sinh(kL)/k`; growth rate
fitted from the adaptive runs 1.0009 in `kL`; the integrated `Φ(L)` grows at
rate `k` to 1.6e−11 relative for every `k` and decays at rate `k` to 6.3e−11
for `k ≤ 4` (at `k = 8` the decaying eigenvalue `e^{−16}` is below the
integration error of the growing mode and is not claimed). RK4 error exponent 4.94 (measured/predicted 0.90–1.02). Required
steps: RK4 23, 54, 127, 303 (exponent 1.24); implicit midpoint 832, 2311, 6532,
18476 (exponent 1.49); Gauss–Legendre 15, 35, 83, 196 (exponent 1.24, 0.647–0.654
times the RK4 steps, within 2.3 % of the predicted `(120/720)^{1/4} = 0.639`); DP45 81 … 505 (exponent 0.88). Accuracy needs 23–50× the RK4
stability limit. The RK4 geodesic endpoint error grows from 2.2e−9 (`k = 1`) to
0.46 (`k = 8`). Implicit midpoint at `kh = 2.29` changes sign six times in seven
steps. Saddle: `j_head(L)` = 3.2, 14.3, 105.8, 794, 5775, 4.1e4, 2.9e5, 2.1e6
for `c = 1, 4, … , 16384`; local exponents from `c = 16` decrease
monotonically, 1.454, 1.431, 1.420, 1.416, 1.415, the last within 0.001 of `√2`;
DP45 steps grow by about 47 per factor 4 in `c` (logarithmically).

*Counterexamples.* "An implicit (A-stable) integrator removes the growth of
the step count with `k` on strongly negatively curved surfaces" (an implicit
method of the same order as RK4 needs fewer steps, but the same `k^{5/4}`
growth); "Jacobi growth is exponential in √(peak |K|) × length".

*Does not prove.* The `√2` saddle limit is a matched-asymptotics argument
supported by the local exponents, not a proof; only the ridge geodesic is
studied; the implicit methods are evaluated on the constant-curvature Jacobi
system through their exact step matrices.

## T017 Validity domains of the first-order approximation

*Prediction.* `r(ε, s) = d(s) − ε|j(s)| = C₂(s) ε² + C₃(s) ε³ + …` with `d` the
unsigned separation. The first-order prediction is within relative tolerance
`τ` while `|C₂ ε + C₃ ε²| ≤ τ |j(s)|`. Closed forms: unit sphere, pure heading,
`C₂ = 0`, `C₃ = −|sin s| cos² s/24`, `ε_max = √(24τ)/|cos s|` — no collapse at
`s = π`; hyperbolic plane, `C₂ = 0`, `ε_max = √(24τ)/cosh s`. Generic paths have
`C₂(s*) ≠ 0`, so `ε_max ∝ |s − s*| → 0`; symmetric ones (outer equator) have
`C₂ ≡ 0` and `ε_max ∝ √|s − s*|`.

The sphere with lateral `ε` and heading `ε` behaves differently at its
first-order zero `s0 = 3π/4`: the `ε²/2` offset there is along-track,
orthogonal to `ε j N`, so near `s0` the distance is
`√((ε j)² + ε⁴/4)` and `ε_max = 2|j| √(2τ + τ²) ≈ 2√(2τ) |j'(s0)| |s − s0|`,
still linear in `|s − s0|` but with a coefficient that does not involve `C₂`.

At a first-order zero `ε_max` itself is set by the numerical `|j|` there
(about 1e−10 on the torus, where the grid node is placed on the located zero,
and 1e−16 on the sphere), so it is zero by construction and is not a check;
the collapse is checked through its linear law, using the nodes at `0.99` and
`1.01` of the zero (their average cancels the first-order variation of `j'`
and `C₂` across it).

*Result* (`τ = 0.01`). Generic torus: `C₂(s*) = 1.52`, `ε_max(s*/2) = 0.041`,
and `ε_max/|s − s*|` next to `s*` is 0.007384 against
`τ|j'(s*)|/|C₂(s*)| = 0.007386`. The linear predictions at `ε_max/2` and
`2 ε_max` are `τ/2` and `2τ`; the fitted remainder model
`|C₂ ε + C₃ ε²|/|j|` predicts slightly different values. New integrations at
`0.9 s*` give 0.00499 and 0.0201: on either side of `τ`, within 3.1e−4 and
3e−5 (relative) of the full model (checked to 1e−3, ten times the two-term fit
residual 1e−4), and 0.2 % and 0.5 % from the linear values. Halving the grid
changes these probe errors by 4e−9 and the tabulated `C₂` by at most 6e−7.
Sphere, pure heading: `ε_max = 0.490` at `0.999π` and `1.001π`; C₃ matches its
closed form to 8e−5 wherever that closed form is nonzero (at `s = π/2` it is
zero and the fitted remainder is roundoff). Hyperbolic `ε_max` 0.475 … 0.080 for
`s = 0.25 … 2.5` (within 0.35 % of the closed form; the fitted C₃ absorbs
`ε⁵` terms, up to 0.8 % at `s = 2.5`). Sphere lateral+heading:
`C₂(3π/4) = 0.500`, and `ε_max/|s − s0|` next to `s0` is 0.4013 against
`2√(2τ + τ²)|j'(s0)| = 0.4010`. Equator: `max |C₂| = 5e−6` (grid halving
changes it by 2e−12), cubic remainder at `s*`.

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

*Protocol.* Fourteen length-2 paths (plane, cylinder, six spheres of radius 1
to 1e8, two tori, saddle, two bumps, hyperbolic plane); the plane and cylinder
use `K` recomputed from their second fundamental form, so a zero signal there
tests the embedding and not the literal `K = 0` of the core classes. Euler,
midpoint, RK4 with `N = 4 … 128`; true deviations from cancellation-free closed
forms or DP45 at `rtol = 1e−12` (checked at `1e−11`, a self-convergence
reference); resolved when `S/E ≥ 10` for every finer step. Each error is
split into a truncation part and a rounding part. For constant `K` the Jacobi
part of every explicit Runge–Kutta stage is linear, so the method's step
matrix evaluated in exact rational arithmetic (`fractions.Fraction`, inputs at
their exact binary values) is the run's truncation-only result, and the
computed deviation minus it is the rounding part, both exact. For the
variable-curvature paths the truncation part is the observed error and the
rounding part is estimated by a rerun with the heading column scaled by 3
(identical in exact arithmetic, rounded differently). An entry is
rounding-affected when its rounding part exceeds 0.1 of its truncation part (a
declared convention); rounding-affected ratios stay in the artifacts and out of
the ratio claims and witnesses. An earlier version classed every error below
64 ulp(L) as roundoff; at `K = 1e−14`, where the signal is only 30 ulp(L), that
cut labelled genuine truncation errors as roundoff, so it could not fail.

*Result.* RK4 resolves every truncation-dominated signal from `N = 16` (minimum
ratio 1.2e5, hyperbolic plane); its errors are rounding-affected for `K = 1e−4`
from `N = 32` and at every step for `K ≤ 1e−8`. Euler needs `N = 32` (64 on the
hyperbolic plane), midpoint `N = 4` (8). Between `K = 1` and `1e−2` weaker
curvature *is* harder for Euler and midpoint (ratio factors 0.64 and 0.41 at
`N = 16`); below `K = 1e−2` their ratios no longer depend on `K` (`K = 1e−8`
against `1e−2`: 0.996 and 0.984), and the RK4 ratio grows 98× from `K = 1e−2`
to `1e−4`. At `K = 1e−16` the computed deviation is exactly zero for every
method and step (ratio 1), although the truncation errors alone would give
ratios of at least 42.9 at `N = 128`: rounding, not truncation, is the limit.
At `K = 1e−14` (signal 30 ulp(L)) the Euler and midpoint errors are still
truncation errors (at `N = 4`, 0.987 and 1.08 times their exact-arithmetic
truncation errors; the midpoint truncation part alone is resolved from
`N = 4`), while every RK4 error
there is rounding (exact-arithmetic truncation below 1e−30); no claim is made
about `K = 1e−14` beyond this report. Flat
surfaces show no spurious signal and a second-form `K` of exactly 0. The
optional SciPy DOP853 runs agree with the DP45 references to 3.4e−13; they
share `ciw`'s right-hand side and check the stepper only.

*Counterexamples.* "Weaker intrinsic curvature is harder to resolve at a fixed
step size" (it holds for Euler and midpoint only down to `K ~ 1e−2`, and not
for RK4); "refining the step size always makes a nonzero curvature effect
resolvable". Transfer to measured sensor data is a `sensor_performance` claim
and is `not_established`.

## Requested follow-ups

* A symmetric (reversible) integrator in `ciw.lab.integrators` would make the
  reversal test exact and give a long-horizon comparison (T014, T015).
* A `Surface.check` scale that is invariant under anisotropic charts would
  allow tori with `R ≥ 10⁶`; today `det g / tr(g)²` refuses them as degenerate.
* Hyperbolic-plane isometries (Möbius maps) as a `ChartMap` would extend T012 to
  intrinsic surfaces.
* A second-form (embedding-derived) curvature option in the core surfaces would
  let flatness tests run without the local `SecondFormCurvature` wrapper.
* An implicit geodesic integrator (for example Gauss–Legendre collocation) in
  `ciw.lab.integrators` would let T016 compare implicit methods on the full
  nonlinear system instead of the constant-curvature step matrices.
