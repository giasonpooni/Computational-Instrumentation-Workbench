# Formal specifications for the lab queue

These specifications state what each experiment family computes and what its
checks mean. Task identities point to the executable experiments; their
retained reports hold the numbers. Textbook results are cited as such; the
workbench contribution is the executable, evidence-labelled test of them.

## Evidence labels

Let a finding be a tuple (claim, domain, value, basis). The label function
`L(basis, domain)` (`ciw.lab.evidence.supported_label`) is total on
well-formed bases and refuses malformed ones. With `C` the set of passing
reference checks, `F` the failing ones, `I` an independent check between
implementations of different origin, `P` a pinned provider execution, `G` a
declared generator, `D` a derivation and `A` a hardware acquisition record:

1. Authority domains (`machine_safety`, `industrial_readiness`,
   `customer_demand`, `actuator_authority`, `production_acceptance`):
   `L = not_established` for every basis.
2. Any failed check (`F ≠ ∅` or `I` failing): `L = not_established`.
3. Physical domains (`physical`, `calibration`, `sensor_performance`):
   `L = not_established` unless `A` is present; then
   `independently_verified` if `I` passes, else `hardware_measured`.
4. Computational domains: `A` is refused; otherwise the first applicable of
   `I → independently_verified`, `C ≠ ∅ → numerically_verified`,
   `P → provider_backed`, `G → synthetic`, `D → analytic`, else
   `not_established`.
5. Origin rule: `I` requires `origin(producer) ≠ origin(checker)`, where
   `origin` is the leading ASCII name token of the NFKC-normalized,
   casefolded implementation identifier, and both origins belong to a closed
   allowlist (`ciw` plus the recognised external families). The rule is
   symmetric: a pinned provider's output checked by a `ciw` reference is as
   independent as `ciw` output checked by the provider. Code in one family
   never verifies itself independently, and a `cross_implementation` check is
   never `I`.
6. Non-upgrade: a stated label must equal `L(basis, domain)`
   (`validate_finding`); a derived physical status is `hardware_measured` only
   if every input is (`physical_status`).
7. Comparisons: a check passes by `holds(observed, tolerance, comparison)`
   with `abs_le: |x| ≤ t`, `le: x ≤ t` (x ≥ 0 required),
   `ge: x ≥ t`, `signed_le: x ≤ t`, `signed_ge: x ≥ t`, finite `x`,
   `|t| ≤ 10¹⁰⁰`. The stated `passed` must equal the computed one.
8. Primary label of a report: with `E` the established computational labels
   and `R` the refuted computational findings (not `not_established` by
   declaration), `primary = not_established` if `R ≠ ∅` or `E = ∅`, else
   `min(E)` in the order `synthetic < analytic < provider_backed <
   numerically_verified < independently_verified`. The function is
   order-independent and monotone: adding a weaker finding never raises it.
9. Physical gate: an acquisition record enters a report only when a hardware
   probe succeeded in the run and its `raw_sha256` equals the digest of a
   retained artifact of the same task.

T155 checks rules 1–4 exhaustively over a finite basis grammar. T100 checks
that synthetic, provider-backed and physical results remain visibly distinct.

## Geodesic equation and references

In a chart `u = (u¹, u²)` with metric `g_ij(u)`,

```
Γᵏᵢⱼ = ½ gᵏˡ (∂ᵢ gⱼₗ + ∂ⱼ gᵢₗ − ∂ₗ gᵢⱼ),      ü ᵏ + Γᵏᵢⱼ u̇ⁱ u̇ʲ = 0.
```

For an embedding `X(u)` the metric is `gᵢⱼ = Xᵢ·Xⱼ` and
`∂ₖ gᵢⱼ = Xₖᵢ·Xⱼ + Xᵢ·Xₖⱼ`; Gaussian curvature is
`K = (LN − M²)/(EG − F²)` from the second fundamental form. The speed
`g(u̇, u̇)` is a first integral; the integrators never renormalize it, so its
drift is a measured quantity (T004). References: exact geodesics on the plane,
sphere (great circles), cylinder (helices) and hyperbolic plane (semicircles);
high-precision independent integration elsewhere (T002). Integrator orders
1, 2, 4 and the adaptive Dormand–Prince 5(4) behavior are tested in T003.

The hyperbolic closed form for `g = I/(k² y²)` with chart heading `α` is
`x = x₀ + y₀ cos α sinh(ks)/D`, `y = y₀/D`, `D = cosh(ks) − sin α sinh(ks)`,
evaluated without cancellation for every heading, near-vertical ones included.
A chart point is regular when `g` is positive definite with
`det g / (tr g)² > 10⁻¹²` (about a reciprocal condition number, so the test
does not depend on the surface scale); otherwise it is refused as
`degenerate_metric`, and points outside a chart's domain as `outside_chart`.
A reparametrized chart `a ↦ φ(a)` keeps the base chart's domain (`φ(a)`
outside it is refused as `outside_chart`, a nonfinite `φ(a)` as
`degenerate_metric`); regularity is judged on its pullback metric `Jᵀ g J` by
the same test, so a chart map that removes a base coordinate singularity (normal
coordinates about a pole) is regular there. A frame change
`X ↦ R X` requires `max |RᵀR − I| ≤ 10⁻¹²` and `det R = +1`. A negative length
integrates backward from `s = 0` in both the fixed-step and adaptive
integrators. Observed orders are least-squares log–log slopes; zero, negative
or nonfinite errors or step sizes are refused rather than dropped, and so are
tables with fewer than two distinct step sizes (the slope is then undetermined).

## Jacobi equation, transfer matrix and Wronskian

Along a unit-speed geodesic with parallel unit normal `N`, a normal Jacobi
field `J = j N` satisfies `j'' + K(γ(s)) j = 0`. The transfer matrix

```
Φ(s) = [[j_lat,  j_head ],
        [j_lat', j_head']],   Φ(0) = I,
```

has lateral-displacement and heading columns. `det Φ ≡ 1` (Wronskian; T007).
For constant `K`: `j_head = sin(√K s)/√K`, `s`, or `sinh(√−K s)/√−K` (T005).
Conjugate points are the zeros `s ≠ 0` of `j_head` (`s > 0` on a forward
geodesic); focal points of the initial normal geodesic are the zeros of
`j_lat` (T008); both are listed in the order the geodesic meets them. On the
unit sphere they occur at `π` and `π/2`; on the outer torus equator at
`π√(r(R+r))`.

## First-order validity and focal counterexamples

A heading perturbation `ε` produces normal separation
`d(s) = ε j_head(s) + r(ε, s)` with `|r| ≤ C(s) ε²`. The first-order
prediction is valid to relative tolerance `τ` while `ε ≤ τ |j_head(s)| / C(s)`,
a domain that shrinks to zero at conjugate points (T017). Near and after a
focus the relative first-order error diverges and the image inverts (T010).

## Chord versus geodesic distance

A unit-speed curve with curvature `κ` has chord `c = s − κ² s³/24 + O(s⁵)`;
for constant `κ`, `c = 2 sin(κ s/2)/κ`. A geodesic's space curvature equals
its normal curvature. On a cylinder of radius `R`, a helix making angle `α`
with the circumferential direction has `κ = cos²α / R`, so
`s − c ≈ cos⁴α s³ / (24 R²)` (T046, T047). A camera measures `c`, not `s`.

## Filter consistency (NEES/NIS)

For a linear-Gaussian filter with innovation `ν = z − H x̂⁻` and innovation
covariance `S = H P⁻ Hᵀ + R`, the normalized innovation squared `νᵀ S⁻¹ ν`
is χ² with `dim z` degrees of freedom when the model is correct. Using the raw
sensor covariance `R` in place of `S` breaks this (T066). Ignoring declared
cross-correlation breaks NEES consistency (T062). For the linear-Gaussian
bench, the filter estimate equals the batch information-form posterior (T074).

## Flat torus lattices and winding

A flat torus is `ℂ/Λ`, `Λ = ℤω₁ + ℤω₂`. Bases related by `SL(2, ℤ)` generate
the same lattice (T019); reduction preserves area, systole and length
spectrum (T026). Closed geodesics correspond to primitive vectors
`mω₁ + nω₂`, `gcd(m, n) = 1`, of length `|mω₁ + nω₂|` (T020). Because
`K = 0`, heading sensitivity at the target equals path length (T021).

## Deterministic serialization and reduction

Canonical JSON: sorted keys, no insignificant whitespace, UTF-8, finite
binary64 values in shortest round-trip form, NaN and infinities refused
(T146). Floating-point summation is not associative; a declared reduction
order (fixed pairwise tree or compensated summation) makes results
reproducible across implementations (T148), and reduction order can change a
threshold decision near its boundary (T121).
