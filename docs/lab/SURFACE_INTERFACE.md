# Surface interface, derivative checks, charts and singularities

This note is the contract of the reusable surface interface in
`src/ciw/lab/surfaces.py` and the record of the experiments that validate and
extend it (queue tasks T033–T037). The experiments live in
`src/ciw/lab/surfaces_discrete.py` with helpers
`surfaces_discrete_geometry.py` (conformance suite, Brioschi curvature,
stencils, chart maps, singular surfaces, defect mutants),
`surfaces_discrete_ad.py` (dual numbers, sympy references) and
`surfaces_discrete_charts.py` (sphere atlas, singularity scans, pointwise
guard). Tests: `tests/test_lab_surfaces_discrete.py`.

Numbers below are from a retained run (`ciw lab run T033 T034 T035 T036
T037`); the reports and artifacts are the authority, this page summarizes
them. Every surface is a normalized mathematical object: nothing here
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
| `check(u)` | raises `SurfaceRefusal` for nonfinite points or `det g ≤ 1e-12 (tr g)²` | T037 |
| `embedding(u)`, `embedding_jacobian(u)` | optional point in R³ and 3×2 Jacobian; `None` for intrinsic charts | T034 |

`EmbeddedSurface` derives `g` and `dg` from exact `first(u) = (X_u, X_v)` and
`second(u) = (X_uu, X_uv, X_vv)`; `MongeSurface` from
`height_derivatives(u) = (f_x, f_y, f_xx, f_xy, f_yy)`. A `ChartMap` supplies
`forward`, `jacobian`, `hessian[p, i, k] = ∂²uᵖ/∂aⁱ∂aᵏ` and `inverse`;
`Reparametrized(base, chart)` pulls back `g' = Jᵀ g J` with exact
derivatives, and `Rotated(base, R)` rigidly moves an embedded surface.

Extensions added without modifying the core: `PolarChart`, `ShearChart`
(a ↦ (a₁ + c a₂², a₂)), `Cone` (apex refused with code `conical_singularity`),
`PowerGraph` (z = c ρᵖ, apex refused with `curvature_singularity`),
`SingularityRefusal(SurfaceRefusal)` with a machine-readable `code`,
`require_regular` and `SphereAtlas`.

### Declared domains

Conformance is claimed only at seeded points (PCG64 seed 3301, 32 per
surface) of these boxes, which avoid coordinate singularities. `l` is the
local length scale used to normalize residuals.

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

Worst residuals over 32 points (thresholds 1e-12 for the algebraic
identities, 1e-8 for derivative consistency, 1e-7 for mixed partials and the
Gauss equation):

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

The suite rejects all six seeded defects:

| Mutant | Identities that fail |
| --- | --- |
| sphere R = 2 returning K = 1/R | Gauss equation only |
| torus with `dg[i, k, j]` for `dg[k, i, j]` | dg and Γ symmetry, compatibility, consistency, mixed partials, Gauss |
| saddle with `dg` negated | consistency, Gauss (compatibility **passes**) |
| gaussian bump with `f_xy` dropped | consistency, mixed partials, Gauss |
| indefinite `diag(1, −(1 + y²))` | eigenvalue ratio; the connection identities are *not evaluated* |
| asymmetric `g` | metric symmetry |

Two lessons are recorded as counterexamples. First, only the Gauss
equation sees a wrong curvature: the other identities never involve `K`.
Second, metric compatibility holds for *any* symmetric `dg` once `Γ` is built
from it. It tests the index conventions of `christoffel`, not whether `dg` is
the derivative of `g`. Only derivative consistency tests that.

## Derivative checks (T034)

Each conformance surface is re-expressed once as a closed-form embedding
(or, for the hyperbolic plane, a closed-form metric) against a small math
namespace. The same formula feeds:

* **sympy** (optional, distinct implementation origin): symbolic `g`, `dg`,
  `Γ`, and `K = R₁₂₁₂ / det g` from the Riemann tensor of its own Christoffel
  symbols, evaluated with `lambdify`. The worst normalized residual against
  ciw is 3.0e-15 for `g, dg, Γ` and 8.2e-16 for `K`. These findings are
  `independently_verified`. `sympy.simplify` also reduces the symbolic
  Brioschi curvature exactly to the declared closed forms of seven surfaces:
  plane 0, sphere 1, cylinder 0, saddle −1/(u² + v² + 1)², torus
  cos v/(cos v + 2), hyperbolic −1, plane-polar 0.
* **Nested forward-mode dual numbers** (implemented in ciw, so same-origin
  and `numerically_verified` only). Tags keep nested perturbations apart,
  and the Siskind–Pearlmutter test d/dx[x · d/dy(x + y)] = 1 passes. Against
  ciw the residuals are 4.4e-16 (`g, dg, Γ`) and 1.0e-15 (`K`, both via
  Brioschi with exact second derivatives and via LN − M²). The dual numbers
  expose the dropped-`f_xy` defect at a normalized error of 0.061.

When sympy is absent the task reports `partial` and records the symbolic
finding as `not_established`.

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

At h = 1e-3 the difference `FD − dg` equals the predicted leading term
`h²/6 ∂³g` to within 6.6e-4 relative. Any error in the analytic `dg` is
therefore below the V-bottom, about 4e-11 normalized.

Two counterexamples come out of this scan. First, a smaller step is not
always better: on the sphere the error at h = 1e-12 is 5e6 times the minimum.
Second, not every metric has a truncation branch: the saddle and polar-plane
metrics are quadratic, so their central differences are exact up to rounding
(6.5e-15 at h = 1e-2), and the plane's constant metric differences to exactly
0. Complex-step differentiation was not run, because the core formulas use
`math.*` and reject complex input.

## Chart atlas (T036)

The atlas has two charts:

* chart A is the core polar chart, with poles at (0, 0, ±R);
* chart B is `Rotated(Sphere, R_y(π/2))`, with poles at (±R, 0, 0), on the
  equator of chart A.

`det g/R⁴ = sin²θ` in each chart, and `sin²θ_A + sin²θ_B = 1 + y²/R² ≥ 1`.
So the better chart always has `det g/R⁴ ≥ ½`, which T036 confirms at 4352
sampled points (minimum 0.502).

Transitions are closed form. Points map through the embedding and the
target chart's inverse (`atan2`/`hypot`). Velocities map by
`v_B = g_B⁻¹ J_Bᵀ J_A v_A`, which is exact because `J_A v_A` lies in the
shared tangent plane. The A→B→A round trip is exact to 6.5e-16, speed is
preserved to 5.2e-16, and the pushforward matches a central difference of
the point map to 3.7e-10.

Integration uses the core RK4 step. After each step, if the active chart's
`det g/R⁴ < ¼`, the state moves to the better chart. The ½ ≥ ¼ margin rules
out chattering; the lowest post-switch value was 0.754.

The test geodesics are great circles of length 2π at azimuth 1.3, with
closest approach δ to the north pole, integrated in 400 steps and compared
with the exact great circle:

| δ | atlas error (4 switches) | chart A alone |
| --- | --- | --- |
| 0 | 4.9e-8 | 7.4e-14 (exact meridian, v_φ ≡ 0) |
| 1e-1 | 5.7e-8 | 7.2e-6 |
| 1e-2 … 1e-6 | 4.7e-8 … 4.9e-8 | fails (nonfinite state or math domain error) |
| 1e-8 | 4.9e-8 | 1.0e-1 |
| 1e-10 | 4.9e-8 | 1.1e-5 |
| 1e-12 | 4.9e-8 | 1.0e-7 |

The atlas converges at orders 4.16 (200/400 steps) and 3.96 (400/800 steps)
through the switches.

A single chart passes the pole cleanly only on the exact meridian, where
`v_φ` is exactly zero and the `cot θ` terms never act. That case is itself a
counterexample to "single-chart integration through a pole always fails". It
is also a knife edge: for 0 < δ ≲ 1e-2 the Clairaut rate `φ' = sin δ/sin²θ`
makes the equations stiff near the pole.

## Singularity classification (T037)

A scan follows a path into a candidate point over distances r from 1e-1 to
1e-8. It fits power laws r^a for `det g`, `cond g`, `max|Γ|` and `|K|` on
r ≤ 1e-3, and computes two further quantities:

* the circumference ratio `C(r)/(2π ρ(r))`, loop length over 2π times radial
  geodesic distance, taken at the smallest r;
* the share of the radial distance that lies in the last two decades of the
  scan.

The rules are applied in this order:

1. If `K` blows up (exponent ≤ −½), the point is a `curvature_singularity`.
2. If `det g` blows up and the radial distance diverges, it is an
   `infinite_distance_boundary`.
3. If `det g → 0` or `cond g → ∞`, it is a `conical_singularity` when the
   circumference ratio is not 1, and a `coordinate_singularity` otherwise.
4. Anything else is `regular`.

| Approach | det | cond | max\|Γ\| | \|K\| | circumference ratio | class |
| --- | --- | --- | --- | --- | --- | --- |
| sphere north pole (chart A) | r² | r⁻² | r⁻¹ | r⁰ (K = 1) | 1 | coordinate singularity (regular in chart B) |
| plane, polar chart origin | r² | r⁻² | r⁻¹ | K ≡ 0 | 1 | coordinate singularity |
| cone apex, α = π/6 | r² | r⁻² | r⁻¹ | K ≡ 0 | 0.5 = sin α | conical singularity, angle deficit π |
| z = r^(3/2) apex (Monge chart) | r⁰ | r⁰ | r⁰ | r⁻¹ (K r → 9/8) | 1 | curvature singularity |
| hyperbolic plane, y → 0 | y⁻⁴ | 1 | y⁻¹ | K = −1 | n/a | infinite-distance boundary |
| saddle origin, sphere equator | regular | regular | r¹ | bounded | 1 | regular |

The scans produce three counterexamples:

* A degenerate metric does not imply a curvature singularity (the sphere
  pole), and a regular metric does not imply bounded curvature
  (z = r^(3/2), where det g → 1 while K ~ 1/r).
* The blow-up of Christoffel symbols is independent of curvature blow-up.
  At the sphere pole Γ ~ r⁻¹ while K = 1; at the z = r^(3/2) apex Γ stays
  bounded while K ~ r⁻¹.
* The polar-plane origin and the cone apex have identical pointwise
  exponents. Only the nonlocal circumference test (1 against sin α)
  separates a removable coordinate singularity from a conical point.

`require_regular(surface, u)` is the pointwise guard. It raises
`SingularityRefusal` with one of these codes:

| Code | Meaning |
| --- | --- |
| `nonfinite_point` | coordinates are not finite |
| `nonfinite_metric` | the metric is not finite |
| `degenerate_metric` | the metric is indefinite or its condition number exceeds 1e8; this is a coordinate *or* conical singularity |
| `curvature_blowup` | \|K\| l² exceeds the declared bound (1e6) |
| surface's own code | for example `curvature_singularity` at the power-graph apex, or `conical_singularity` for the cone's K at its apex |
| `chart_refused` | the core `Surface.check` refused the point, for example the hyperbolic plane at y ≤ 0 |

The core `Surface.check` is more lenient: on the sphere pole approach it
accepts points with `cond g` up to 3.2e11 (θ ≈ 1.8e-6). The guard refuses
from θ ≈ 1e-4.

## What these results do not establish

* Conformance, derivative agreement and classification are shown at sampled
  points of declared domains and for the declared examples. They are not
  proofs for whole domains or for general surfaces.
* sympy agreement is independent in how derivatives are computed. The
  closed-form re-expressions are hand-written from the same definitions as
  the core, so a shared misreading of a surface definition would pass both.
* The atlas exists for the sphere only. Chart switching happens between
  fixed steps; adaptive integration would need event location.
* Nothing here applies to measured surfaces. Noise σ ≫ ε changes the
  optimal difference step to about (σ/|∂³g|)^(1/3), and scanned singular
  points are not rotationally symmetric. The corresponding findings are
  recorded as `not_established` in the physical and industrial-readiness
  domains.

## Requested core changes (not made here)

* `Surface.check` could refuse by metric condition number (for example
  `cond g > 1e8`) in addition to `det g ≤ 1e-12 (tr g)²`, and raise a coded
  refusal (`degenerate_metric`) as `require_regular` does.
* `SurfaceRefusal` could carry an optional `code` attribute, so that
  `SingularityRefusal` becomes unnecessary.
* The catalogue could export its declared sampling domains (the table
  above), so that conformance domains are not restated per section.
