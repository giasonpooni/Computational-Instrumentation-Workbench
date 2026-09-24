# Surface interface, derivative checks, charts and singularities

This note is the contract of the reusable surface interface in
`src/ciw/lab/surfaces.py` and the record of the experiments that validate and
extend it (queue tasks T033–T037). The experiments live in
`src/ciw/lab/surfaces_discrete.py` with helpers
`surfaces_discrete_geometry.py` (conformance suite, Brioschi curvature,
stencils, chart maps, singular surfaces, defect mutants),
`surfaces_discrete_ad.py` (dual numbers, sympy and sympy.diffgeom references) and
`surfaces_discrete_charts.py` (chart atlases, singularity scans, pointwise
guard). Tests: `tests/test_lab_surfaces_discrete.py`.

Numbers below are from a retained run (`ciw lab run T033 T034 T035 T036
T037`); the reports and artifacts are the authority, this page summarizes
them. Each report records the source digests of the section modules and of
the core modules its numbers depend on (`src/ciw/lab/surfaces.py`, and
`src/ciw/lab/integrators.py` for T036) under
`provider_runtime_identity.sources`, so the conformance evidence names the
revision of the interface it certifies. Every numerical finding carries its
own uncertainty (`{kind, value, basis}`). Every surface is a normalized mathematical object: nothing here
describes, measures or calibrates a physical surface.

## Interface contract

A `Surface` is one chart `u = (u¹, u²)` of a two-dimensional Riemannian
surface.

| Member | Contract | Checked by |
| --- | --- | --- |
| `metric(u)` | 2×2 array `g_ij`, symmetric and positive definite on the declared domain | T033 symmetry, eigenvalue ratio |
| `metric_derivatives(u)` | `dg[k, i, j] = ∂_k g_ij`, **exact** (never finite differences); symmetric in `i, j`; a gradient field (`∂_l dg[k] = ∂_k dg[l]`) | T033 consistency and mixed partials, T034, T035 |
| `gaussian_curvature(u)` | exact `K` (closed form or second fundamental form); must satisfy the Gauss equation, i.e. equal the curvature computed from `g` alone | T033 Brioschi, T034 Riemann tensor |
| `christoffel(u)` | generic `Γ[k, i, j] = ½ gᵏˡ (∂_i g_jl + ∂_j g_il − ∂_l g_ij)`: symmetric in `i, j`, metric compatible | T033 |
| `geodesic_rhs(y)` | `y = (u, v)`, `u' = v`, `v'ᵏ = −Γᵏ_ij vⁱ vʲ`; no renormalization | T036 |
| `check(u)` | raises `SurfaceRefusal(message, code)` with code `nonfinite_point` for nonfinite coordinates or `degenerate_metric` when `det(g / tr g) = det g / (tr g)² ≤ 1e-12`, a scale-free test roughly equivalent to `cond g > 1e12` (charts may override, e.g. `outside_chart` for the hyperbolic plane at y ≤ 0) | T037 |
| `embedding(u)`, `embedding_jacobian(u)` | optional point in R³ and 3×2 Jacobian; `None` for intrinsic charts | T034 |

`EmbeddedSurface` derives `g` and `dg` from exact `first(u) = (X_u, X_v)` and
`second(u) = (X_uu, X_uv, X_vv)`; `MongeSurface` from
`height_derivatives(u) = (f_x, f_y, f_xx, f_xy, f_yy)`. A `ChartMap` supplies
`forward`, `jacobian`, `hessian[p, i, k] = ∂²uᵖ/∂aⁱ∂aᵏ` and `inverse`;
`Reparametrized(base, chart)` pulls back `g' = Jᵀ g J` with exact
derivatives, and `Rotated(base, R)` rigidly moves an embedded surface.

Extensions added without modifying the core: `PolarChart`, `ShearChart`
(a ↦ (a₁ + c a₂², a₂)), `CubeRootChart` (a ↦ (∛a₁, a₂), whose Jacobian blows
up on a₁ = 0), `Cone` (apex refused with code `conical_singularity`),
`PowerGraph` (z = c ρᵖ, apex refused with `curvature_singularity`),
`ConformalHalfPlane` (g = y^(−2a) I, K = −a y^(2a−2)), `require_regular`,
`Atlas` (with `Atlas.from_chart_maps`, `graph_atlas` and `SphereAtlas`). Every
refusal is the core `SurfaceRefusal(message, code)`; the section defines no
refusal class of its own.

### Declared domains

Conformance is claimed only at seeded points (PCG64 seed 3301, 32 per
surface) of these boxes, which avoid coordinate singularities. The catalogue
boxes are read from the core `SAMPLING_DOMAINS` (`sampling_domain(key)`); only
the three derived surfaces and the hyperbolic length scale are declared by the
section. `l` is the local length scale used to normalize residuals.

| Surface | Domain | l |
| --- | --- | --- |
| plane, gaussian-bump-shear | [−2, 2]² | 1 |
| sphere (R = 1, polar) | θ ∈ [0.3, π − 0.3], φ ∈ [−π, π] | 1 |
| cylinder (R = 1) | φ ∈ [−π, π], z ∈ [−2, 2] | 1 |
| saddle (c = 1) | [−1.5, 1.5]² | 1 |
| torus, rotated-torus (R = 2, r = 1) | [−π, π]² | 1 |
| gaussian-bump (h = 0.5, σ = 1) | [−2.5, 2.5]² | 1 |
| hyperbolic-plane (k = 1) | x ∈ [−2, 2], y ∈ [0.3, 3] | y |
| plane-polar | r ∈ [0.3, 2], t ∈ [−π, π] | 1 |

## Conformance suite (T033)

At each point the suite evaluates, with normalized residuals:

1. symmetry of `g` and the eigenvalue ratio λ_min/λ_max (> 0 ⇔ positive definite);
2. symmetry of `dg` and of `Γ` in the lower indices;
3. metric compatibility `∂_k g_ij = Γˡ_ki g_lj + Γˡ_kj g_il`;
4. derivative consistency: `dg` against fourth-order central differences of `g` (step 1e-3 l);
5. mixed partials of `dg` (by fourth-order differences of the exact `dg`);
6. the Gauss equation: Brioschi's formula, using `E, F, G`, the exact first
   derivatives and second derivatives from differences of the exact first
   derivatives, against the supplied `K`.

A residual that is NaN or infinite at any point fails its identity and is
reported as `nonfinite`; an exception while evaluating a point fails the
surface. (A running `max` would silently drop a NaN, because every comparison
with NaN is false.)

Worst residuals over 32 points (thresholds 1e-13 for metric and `dg`
symmetry and 1e-12 for `Γ` symmetry and compatibility, which are algebraic
identities; 1e-8 for derivative consistency; 1e-7 for mixed partials and the
Gauss equation; eigenvalue ratio at least 1e-3):

| Surface | min λ ratio | Γ asymmetry | compatibility | dg vs differences | mixed partials | Gauss equation |
| --- | --- | --- | --- | --- | --- | --- |
| plane | 1.000 | 0 | 0 | 4.2e-14 | 0 | 0 |
| sphere | 0.124 | 0 | 6.1e-17 | 3.8e-13 | 9.7e-14 | 1.9e-12 |
| cylinder | 1.000 | 0 | 6.2e-33 | 1.5e-13 | 2.3e-30 | 0 |
| saddle | 0.211 | 0 | 1.6e-16 | 1.6e-13 | 3.4e-14 | 4.3e-14 |
| torus | 0.111 | 0 | 8.3e-17 | 2.9e-13 | 1.1e-13 | 2.5e-13 |
| gaussian-bump | 0.916 | 0 | 2.4e-17 | 6.0e-13 | 3.5e-13 | 3.3e-13 |
| hyperbolic-plane | 1.000 | 0 | 1.4e-16 | 8.1e-12 | 1.4e-14 | 7.6e-11 |
| plane-polar | 0.231 | 0 | 1.5e-16 | 1.8e-13 | 9.7e-14 | 2.7e-13 |
| gaussian-bump-shear | 0.108 | 0 | 1.5e-16 | 5.4e-13 | 5.9e-13 | 1.5e-13 |
| rotated-torus | 0.111 | 0 | 1.5e-16 | 3.6e-13 | 1.6e-13 | 2.4e-13 |

The suite rejects all seven seeded defects:

| Mutant | Identities that fail |
| --- | --- |
| sphere R = 2 returning K = 1/R | Gauss equation only |
| torus with `dg[i, k, j]` for `dg[k, i, j]` | dg and Γ symmetry, compatibility, consistency, mixed partials, Gauss |
| saddle with `dg` negated | consistency, Gauss (compatibility **passes**) |
| gaussian bump with `f_xy` dropped | consistency, mixed partials, Gauss |
| indefinite `diag(1, −(1 + y²))` | eigenvalue ratio; the connection identities are *not evaluated* |
| asymmetric `g` | metric symmetry |
| torus with `dg` NaN wherever u₁ > 0 | every identity involving `dg`, each flagged `nonfinite` |

Two lessons are recorded as counterexamples, each on its own finding with
checks that pass when the violation is observed. First, only the Gauss
equation sees a wrong curvature: the misscaled sphere fails the Gauss equation
(residual 0.167) and nothing else, because the other identities never involve
`K`. Second, metric compatibility holds for *any* symmetric `dg` once `Γ` is
built from it. It tests the index conventions of `christoffel`, not whether
`dg` is the derivative of `g`. Only derivative consistency tests that.

Christoffel symmetry is exact (0) for every surface here, because the core
einsum is symmetric term by term whenever `dg` is symmetric in its last two
indices. The check adds evidence only for a surface that overrides
`christoffel()`.

Per-finding uncertainties: the algebraic identities (symmetry,
compatibility, rigid rotation) are rounding level (`roundoff`, 1e-15); the
stencil-based identities carry the fourth-order stencil error at h = 1e-3 l
(`truncation_bound`: 1e-11 for derivative consistency, 1e-12 for mixed
partials, 1e-10 for the Gauss equation), so their residuals measure stencil
error, not a defect; mutant and refusal counts are exact.

## Derivative checks (T034)

Each conformance surface is re-expressed once as a closed-form embedding
(or, for the hyperbolic plane, a closed-form metric) against a small math
namespace. Parameters enter through `ns.const`, which the symbolic namespace
maps to `sympy.Rational` (the exact rational value of the binary float), so
symbolic identities are exact for the declared parameters and nothing is
rounded by `nsimplify`. The same formula feeds:

* **sympy differentiation** (optional, distinct origin): symbolic `g` and
  `dg` for all ten surfaces, evaluated with `lambdify`; worst normalized
  residual against ciw 4.2e-16 (`independently_verified`).
* **sympy.diffgeom** (distinct origin for both differentiation and
  assembly): `Γ` from `metric_to_Christoffel_2nd` and
  `K = g₀ₘ Rᵐ₁₀₁ / det g` from `metric_to_Riemann_components`, for the seven
  surfaces whose symbolic Riemann tensor is cheap (plane, sphere, cylinder,
  saddle, torus, hyperbolic plane, plane-polar). Residuals 1.8e-16 (`Γ`) and
  7.0e-16 (`K`), both `independently_verified`. `sympy.simplify` reduces the
  diffgeom curvature exactly to closed forms restated from
  `ciw.lab.surfaces` (the producer of that finding is the restatement,
  `_declared_curvature`): plane 0, sphere 1, cylinder 0, saddle
  −1/(u² + v² + 1)², torus cos v/(cos v + 2), hyperbolic −1, plane-polar 0.
* **ciw assembly of sympy derivatives**: `Γ` and `R₁₂₁₂ / det g` written in
  ciw code from sympy's derivatives, for all ten surfaces including the three
  heavy ones (gaussian-bump, gaussian-bump-shear, rotated-torus). Residual
  5.6e-16, recorded as a same-origin `cross_implementation` check
  (`numerically_verified`), not as independent evidence.
* **Nested forward-mode dual numbers** (implemented in ciw, so same-origin:
  their comparisons with the ciw interface, the embedding and the seeded
  defect are recorded as `cross_implementation` checks and are
  `numerically_verified` only; the self-test against hand-derived closed
  forms is the one `analytic` check). Tags keep nested perturbations apart,
  and the Siskind–Pearlmutter test d/dx[x · d/dy(x + y)] = 1 passes. Against
  ciw the residuals are 4.4e-16 (`g, dg, Γ`) and 1.0e-15 (`K`, both via
  Brioschi with exact second derivatives and via LN − M²). The dual numbers
  expose the dropped-`f_xy` defect at a normalized error of 0.061.

When sympy is absent the task reports `partial` and records the symbolic
findings as `not_established`. The claim that this agreement certifies
derivatives of surfaces reconstructed from physical measurements is recorded
as a `physical` finding and is `not_established`.

## Analytic versus finite-difference derivatives (T035)

`D_h g = (g(u + h e_k) − g(u − h e_k))/2h = ∂_k g + h²/6 ∂_k³ g + O(h⁴)`,
plus rounding error of about ε|g|/h. The predicted optimum is
`h* = (3ε|g|/|∂³g|)^(1/3)`, which is of order ε^(1/3) ≈ 6.1e-6, with error
of order ε^(2/3). Third derivatives for the prediction come from the dual
numbers. The scan covers 49 relative steps from 1e-1 to 1e-13 at 12 points
per surface, and the table reports median errors.

| Surface | truncation slope | rounding slope | observed h* | predicted h* | min error |
| --- | --- | --- | --- | --- | --- |
| sphere | 2.0000 | −1.01 | 5.6e-6 | 5.6e-6 | 2.0e-11 |
| torus | 2.0000 | −0.98 | 1.0e-5 | 1.0e-5 | 1.1e-11 |
| gaussian-bump | 2.0000 | −1.03 | 1.8e-5 | 1.8e-5 | 5.2e-12 |
| hyperbolic-plane | 2.0000 | −1.03 | 1.8e-6 | 3.2e-6 | 4.0e-11 |

At h = 1e-3, over the 47 of 48 points where the leading term `h²/6 ∂³g`
exceeds the rounding floor `ε|g|/h` by at least 1e4, the difference `FD − dg`
equals that term to within 2.2e-5 relative (tolerance 1e-3). The masked
point is a flat gaussian-bump flank point, u ≈ (2.40, −2.48), where the
rounding floor is 1.5e-3 of the term. Without the mask the comparison would
measure rounding there, not truncation.

The table above reports medians. Per point, the best step over the scan
agrees with the analytic `dg` to within 7.4e-11 normalized, at most 0.71
times that point's own predicted minimum. The finding bounds this ratio by
2, because the `ε|g|/h` rounding model is only a scale. So any error in the
analytic `dg` is below about 7e-11 normalized at every sampled point. The
sharper pointwise evidence for `dg` is T033's derivative consistency
(fourth-order stencil, worst over 32 points per surface: 8.1e-12).

Two counterexamples come out of this scan. First, a smaller step is not
always better: on the sphere the error at h = 1e-12 is 5e6 times the minimum.
Second, not every metric has a truncation branch: the saddle and polar-plane
metrics are quadratic, so their central differences are exact up to rounding
(6.5e-15 at h = 1e-2), and the plane's constant metric differences to exactly
0. Complex-step differentiation was not run, because the core formulas use
`math.*` and reject complex input.

The counterexample witnesses are recorded as log10 values (h_opt, E(h_opt),
E(1e-12)), so the finding's regression tolerance (0.5 decade) bounds them.
The optimal step is resolved on a grid of 4 steps per decade, and the
findings state that as a 0.25-decade uncertainty; the rounding slopes carry
the standard error of their least-squares fit (at most 0.036).

## Chart atlas (T036)

`Atlas` holds charts of one embedded surface, each with a closed-form inverse
of its embedding. Points move between charts through the embedding and the
target chart's inverse; velocities map by `v_B = g_B⁻¹ J_Bᵀ J_A v_A`, which is
exact because `J_A v_A` lies in the shared tangent plane. A chart's
regularity is the scale-free inverse condition number `λ_min/λ_max` of its
metric. `Atlas.from_chart_maps(base, base_inverse, maps)` builds an atlas from
`ChartMap` reparametrizations of a base chart, each inverted by
`map.inverse(base_inverse(X))`. Two atlases are exercised:

* the **sphere atlas** (`SphereAtlas`): chart A is the core polar chart, with
  poles at (0, 0, ±R); chart B is `Rotated(Sphere, R_y(π/2))`, with poles at
  (±R, 0, 0), on the equator of chart A;
* the **graph atlas** (`graph_atlas`) of the Gaussian bump (h = 0.5, σ = 1):
  its global Monge chart (x, y) and the polar chart (r, t) pulled back through
  `PolarChart`, singular at the apex r = 0.

On the sphere the regularity is `sin²θ = det g/R⁴` in each chart, and
`sin²θ_A + sin²θ_B = 1 + y²/R² ≥ 1`. So the better chart always has
regularity at least ½. T036 checks this at 4352 sampled points (minimum
0.502), evaluating every point through each chart's inverse and metric, so a
wrong inverse or metric fails the check. On the graph the Monge regularity is
`1/(1 + |∇f|²) ≥ 1/(1 + h²/(e σ²)) = 0.9158`; the sampled minimum over 256
points of the disc of radius 2 is 0.9158, above that bound.

Transitions are exact to rounding. On the sphere the A→B→A round trip is
exact to 6.5e-16, speed is preserved to 5.2e-16, and the pushforward matches
a central difference of the point map to 3.7e-10 (the central-difference
error at h = 1e-6). On the graph atlas the same checks give 6.0e-16, 8.2e-16
and 3.0e-10.

Integration uses the core RK4 step. After each step, if the active chart's
regularity is below ¼, the state moves to the better chart. On the sphere the
½ ≥ ¼ margin rules out chattering; the lowest post-switch value was 0.754.

The sphere test geodesics are great circles of length 2π at azimuth 1.3, with
closest approach δ to the north pole, integrated in 400 steps and compared
with the exact great circle:

| δ | atlas error (4 switches) | chart A alone |
| --- | --- | --- |
| 0 | 4.9e-8 | 7.4e-14 at 400 steps (depends on the step grid, see below) |
| 1e-1 | 5.7e-8 | 7.2e-6 |
| 1e-2 … 1e-6 | 4.7e-8 … 4.9e-8 | fails (nonfinite state or math domain error) |
| 1e-8 | 4.9e-8 | 1.0e-1 |
| 1e-10 | 4.9e-8 | 1.1e-5 |
| 1e-12 | 4.9e-8 | 1.0e-7 |

The atlas converges at orders 4.16 (200/400 steps) and 3.96 (400/800 steps)
through the switches. The 400-step error differs from the fourth-order
predictions of the 200- and 800-step runs by 5.7e-9 (12%), which the finding
records as its uncertainty.

Away from the exact meridian the single-chart error grows continuously with
δ. It is roughly 1e5 δ for δ ≤ 1e-10 (1.0e-7 at 1e-12, which is only 2.1
times the atlas error, and 1.1e-5 at 1e-10), reaches 0.10 at 1e-8, and the
integration fails for 1e-6 ≤ δ ≤ 1e-2 at 400 steps. The mechanism is the
azimuthal rate `φ' = sin δ/sin²θ`, which peaks at 1/sin δ at closest
approach and is not resolved by the fixed step.

On the exact meridian (δ = 0) chart A alone is accurate at 400 steps (error
7.4e-14), which is a counterexample to "single-chart integration through a
pole always fails". That success belongs to the step grid, not to the chart.
`v_φ` starts at exactly 0 but does not stay there: rounding in `g_12` (1.5e-17
at the start point) makes the off-diagonal Christoffel symbols nonzero and
seeds an angular momentum `L = sin²θ v_φ`, and each pole crossing amplifies
it, the more the closer a step point lands to the pole, where
`v_φ = L/sin²θ` and the `cot θ` terms act. At 400 steps no step point comes
closer than 5.3e-3 to either pole; even so `|v_φ|` reaches 2.5e-10 and `|L|`
7.4e-14. A scan of 101 step counts (350 to 450) separates the runs cleanly by
that grid distance:

| Steps | closest step point to a pole | chart A alone |
| --- | --- | --- |
| 355 | 8.5e-8 | fails (nonfinite state at step 236) |
| 377 | 2.4e-5 | 1.5e-6 |
| 399 | 4.4e-5 | 3.1e-7 |
| 400 | 5.3e-3 | 7.4e-14 |
| 421 | 6.3e-5 | 2.1e-8 |
| 443 | 8.0e-5 | 3.5e-8 |
| the other 96 | ≥ 1e-4 | ≤ 1.2e-10 |

The five runs whose grid comes within 1e-4 of a pole fail or err at least
2.1e-8; all others err at most 1.2e-10. This is a second counterexample: an
accurate single-chart pole crossing at one step count does not carry over to
the neighbouring step count.

On the graph atlas, geodesics start in the polar chart at x = −1.5, offset δ
from the apex, heading in +x, over length 3 with 200 RK4 steps; the reference
is the Monge chart alone at 800 steps. Every run switches once (polar →
Monge, near r ≈ 0.5):

| δ | atlas error | Monge chart alone | polar chart alone |
| --- | --- | --- | --- |
| 0 | 6.1e-11 | 6.1e-11 | 6.1e-11 (radial line) |
| 1e-1 | 4.1e-9 | 6.0e-11 | 4.8e-6 |
| 1e-2 | 4.3e-10 | 6.1e-11 | 0.55 |
| 1e-3, 1e-4 | 7.0e-11, 6.1e-11 | 6.1e-11 | fails (math domain error, nonfinite state) |
| 1e-6 | 6.1e-11 | 6.1e-11 | 1.4e-2 |
| 1e-8 | 6.1e-11 | 6.1e-11 | 4.4e-6 |

Only a nonfinite RK4 state or a math domain error counts as a single-chart
failure, and it is recorded with its message. Any other exception (a bad step
count, an unknown method) propagates instead of strengthening the
counterexample. In the figure `atlas-vs-single-chart.svg`, failed runs sit at
a fixed ceiling of 1, and the single-chart line never bridges them;
`meridian-steps.svg` plots the step-count scan (failures at 2).

Counterexample witnesses are recorded as counts and log10 values, which the
findings' regression tolerances (±1 count or decade) bound.

## Singularity classification (T037)

A scan follows a path into a candidate point over distances r from 1e-1 to
1e-8. It fits power laws r^a for `det g`, `cond g`, `max|Γ|`, `|K|` and the
radial speed `|dX/dr|` on the window r ≤ 1e-5. A fit counts as clean when its
largest log residual is at most 0.05. On that window a smooth quantity
q₀(1 + c r) has a log-slope of at most |c|·1e-5. The scan also computes the
circumference ratio `C(r)/(2π ρ(r))`: loop length over 2π times the radial
geodesic distance, at the smallest r.

The rules are applied in this order:

1. If a quantity used by the rules is not a clean power law, the point is
   `unclassified`.
2. If `|K| ~ r^a` with a ≤ −0.05, it is a `curvature_singularity`.
3. If the radial speed goes as r^b with b ≤ −1 + 1e-3, the radial length
   ∫ r^b dr diverges, and the point is an `infinite_distance_boundary`.
4. If `det g` blows up at finite distance with bounded `K`, it is
   `unclassified`: a removable blow-up chart and a genuine singularity look
   alike here. This rule misses removable coordinate singularities of that
   kind (the cube-root chart below), and T037 counts the miss.
5. If `det g → 0` or `cond g → ∞`, it is a `conical_singularity` when the
   circumference ratio differs from 1 by more than 1e-6, and a
   `coordinate_singularity` otherwise.
6. Anything else is `regular`.

The circumference test needs approach loops that are preimages of geodesic
circles about the candidate point. The caller chooses them (`Approach.polar`)
from knowledge of the chart. The rules also assume rotationally symmetric
approaches whose radial chart lines are geodesics.

| Approach | det | cond | max\|Γ\| | \|K\| | radial speed | circumference ratio | class |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sphere north pole (chart A, polar loops) | r² | r⁻² | r⁻¹ | r⁰ (K = 1) | r⁰ | 1 | coordinate singularity (regular in chart B) |
| plane, polar chart origin | r² | r⁻² | r⁻¹ | K ≡ 0 | r⁰ | 1 | coordinate singularity |
| cone apex, α = π/6 | r² | r⁻² | r⁻¹ | K ≡ 0 | r⁰ | 0.5 = sin α | conical singularity, angle deficit π |
| z = r^(3/2) apex (Monge chart) | r⁰ | r⁰ | r⁰ | r⁻¹ (K r → 9/8) | r⁰ | 1 | curvature singularity |
| z = r^1.8 apex | r⁰ | r⁰ | r^0.6 | r^−0.4 | r⁰ | 1 | curvature singularity |
| hyperbolic plane, y → 0 | y⁻⁴ | 1 | y⁻¹ | K = −1 | y⁻¹ | n/a | infinite-distance boundary |
| g = y^−1.8 I, y → 0 | y^−3.6 | 1 | y⁻¹ | y^−0.2 | y^−0.9 (finite distance) | n/a | curvature singularity |
| plane in the cube-root chart, a₁ → 0 | r^−4/3 | r^−4/3 | r⁻¹ | K ≡ 0 | r^−2/3 (finite distance) | n/a | unclassified (**missed**: truly a removable coordinate singularity) |
| saddle origin, sphere equator | regular | regular | r¹ | bounded | r⁰ | 1 | regular |

The z = r^(3/2) exponents carry the leading smooth correction:
K = (9/8) r⁻¹ (1 + 9r/4)⁻², so the fitted K exponent is −1.000005 and the det
exponent 2.3e-6. The exact power laws (sphere pole, cone, hyperbolic boundary)
fit to rounding.

The scans produce these counterexamples:

* A degenerate metric does not imply a curvature singularity (the sphere
  pole), and a regular metric does not imply bounded curvature
  (z = r^(3/2), where det g → 1 while K ~ 1/r).
* The blow-up of Christoffel symbols is independent of curvature blow-up.
  At the sphere pole Γ ~ r⁻¹ while K = 1; at the z = r^(3/2) apex Γ stays
  bounded while K ~ r⁻¹.
* The polar-plane origin and the cone apex have identical pointwise
  exponents. Only the nonlocal circumference test (1 against sin α)
  separates a removable coordinate singularity from a conical point.
* Detection limits. Cases just beyond each threshold are misclassified, as
  predicted:

  | Case | True type | Scan reads | Why |
  | --- | --- | --- | --- |
  | z = r^1.99 apex | curvature singularity | regular | K exponent −0.02 is above −0.05 |
  | g = y^−1.999 I, y → 0 | curvature singularity at finite distance (about 2e3 from y = 0.2) | infinite-distance boundary | speed exponent −0.9995 is within 1e-3 of −1; K exponent −0.001 |
  | cone with 1 − sin α = 5e-7 | conical singularity | coordinate singularity | circumference defect 5e-7 is below 1e-6 |

* A missed coordinate singularity. The plane in the cube-root chart has
  `cond g → ∞` (exponent −4/3) at finite distance (radial-speed exponent
  −2/3) with K ≡ 0, which is a coordinate singularity by the definition in
  T037's hypothesis, and it is removable. Rule 4 leaves it `unclassified`, so
  the scan reads 9 of the 10 declared approaches correctly. Separating it
  from a genuine finite-distance singularity would need a removability test,
  for example regularity of the metric in radial arclength coordinates.
* Approach-loop dependence. The same sphere pole, scanned with Cartesian
  loops about (0, 0.3) in chart A, gives circumference ratio 0.832 and reads
  as a conical singularity. Those loops are not geodesic circles, so the
  `polar` choice encodes knowledge of the chart and is part of the input.

`require_regular(surface, u)` is the pointwise guard. It raises the core
`SurfaceRefusal` with one of these codes:

| Code | Source | Meaning |
| --- | --- | --- |
| `nonfinite_point` | guard (also the core check) | coordinates are not finite |
| `nonfinite_metric` | guard | the metric is not finite |
| `degenerate_metric` | guard (also the core check) | the metric is indefinite, or its condition number exceeds 1e8 (guard) or det g / (tr g)² ≤ 1e-12, roughly cond g > 1e12 (core). This is a coordinate *or* conical singularity |
| `curvature_blowup` | guard | \|K\| l² exceeds the declared bound (1e6). A slower blow-up passes: z = r^1.8 at ρ = 1e-8 (\|K\| = 4.1e3) is accepted |
| `outside_chart` | core check of the chart, propagated unchanged | for example the hyperbolic plane at y ≤ 0 |
| `curvature_singularity`, `conical_singularity` | declared by the surface at its own apex | author labels (the power-graph metric at ρ = 0, the cone's K at r = 0), not detections; recorded in `refusals.json` without a check |

T037 checks 8 guard and core-check outcomes, including two acceptances, and,
as a separate finding, that the guard propagates the power graph's own apex
refusal unchanged (it neither swallows nor re-codes it). The surface-declared
codes themselves are not counted as detections.

The core `Surface.check` is more lenient than the guard: on the sphere pole
approach it accepts points with `cond g` up to 3.2e11 (θ ≈ 1.8e-6). The
guard refuses from θ ≈ 1e-4.

## What these results do not establish

* Conformance, derivative agreement and classification are shown at sampled
  points of declared domains and for the declared examples. They are not
  proofs for whole domains or for general surfaces. The classifier has the
  detection limits stated above, and it misclassifies beyond them.
* sympy agreement is independent in how derivatives are computed, and for
  seven surfaces also in how the connection and curvature are assembled
  (sympy.diffgeom). The closed-form re-expressions and the restated closed
  forms are hand-written from the same definitions as the core, so a shared
  misreading of a surface definition would pass both.
* Atlases need an embedding and a closed-form inverse for each chart; they
  are exercised on the sphere and on one graph surface. Intrinsic charts
  (the hyperbolic plane) are not covered. Chart switching happens between
  fixed steps; adaptive integration would need event location.
* Nothing here applies to measured surfaces. Noise σ ≫ ε changes the
  optimal difference step to about (σ/|∂³g|)^(1/3), and scanned singular
  points are not rotationally symmetric. The corresponding findings are
  recorded as `not_established` in the physical and industrial-readiness
  domains.

## Requested core change (not made here)

* `Surface.check` already refuses by a scale-free conditioning test,
  `det g / (tr g)² ≤ 1e-12` (roughly `cond g > 1e12`). It could use a
  tighter condition bound (for example `cond g > 1e8`, code
  `degenerate_metric`), as `require_regular` does; on the sphere pole
  approach it now accepts `cond g` up to 3.2e11.

The coded `SurfaceRefusal(message, code)` and `SAMPLING_DOMAINS` /
`sampling_domain()` are already in the core, and this section uses them.
