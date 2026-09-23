# Flat torus and topology (T019–T032)

Section 2 of the computational-experimentalist queue. Implementation:
`src/ciw/lab/flat_torus_topology.py` with helpers
`flat_torus_topology_lattice.py` (exact lattice algebra),
`flat_torus_topology_surfaces.py` (polygonal translation surfaces),
`flat_torus_topology_routes.py` (route search on curved surfaces),
`flat_torus_topology_discrete.py` (grid approximations) and
`flat_torus_topology_provider.py` (optional pinned FTR comparison). Tests:
`tests/test_lab_flat_torus_topology.py`.

```sh
ciw lab run T019 T020 T021 T022 T023 T024 T025 T026 T027 T028 T029 T030 T031 T032 --output-dir results/lab
ciw lab run T019 T020 T026 --output-dir results/lab \
    --provider ftr=<Flat-Torus-Geodesic-Reference checkout> --provider ftr-python=<Python 3.12>
ciw lab report T032 --retained results/lab
```

The whole section runs in about 10–15 s on one core. Every surface is a
declared mathematical object in normalized units. No finding concerns a
physical part, vehicle, sensor or workspace: route safety, sensor closure
performance, grid-planner accuracy and calibration adequacy are recorded as
`machine_safety`, `sensor_performance`, `physical` or `calibration` findings,
and those are always `not_established`.

## Conventions

A flat torus is C / Λ with Λ = Z w1 + Z w2. The lattice is carried by the
Gram matrix G = BᵀB of the oriented basis, written (a, b, c) = (|w1|²,
⟨w1, w2⟩, |w2|²). The hexagonal lattice therefore has the integer form
(2, 1, 2) although its coordinates are irrational, so every basis change,
reduction and length comparison is exact in integers or `Fraction`s.

- Basis change B′ = B M with M ∈ SL(2, Z): G′ = MᵀGM. Lattice (winding)
  coordinates transport as c′ = M⁻¹c. With FTR's action
  τ′ = (aτ + b)/(cτ + d), the same change is the column matrix [[d, b], [c, a]],
  and the label rule m′ = am − bn, n′ = −cm + dn agrees with c′ = M⁻¹c.
- Shape τ = w2/w1 = (b + i√det G)/a. Area-one float lattices (FTR's
  normalization) use G = (1/y, x/y, |τ|²/y).
- Canonical (Gauss) form: |2b| ≤ a ≤ c, and b ≥ 0 when |2b| = a or a = c.
  FTR puts the vertical boundary at Re τ = −1/2 instead, so float comparisons
  with FTR use interior shapes.
- Heading amplification of a route of length L is |j_head(L)|, where
  j'' + K j = 0 with j(0) = 0, j'(0) = 1 (`ciw.lab.jacobi`). The focus margin is
  s_c − L, where s_c is the first conjugate point of the start along the
  extended geodesic. A negative margin means the route has passed a conjugate
  point and is not locally minimizing. The margin is censored (reported as
  `null` with `margin_lower_bound`) when no conjugate point occurs within the
  extension horizon.

## Tasks

| Task | Question | Method | Key result (retained run) |
| --- | --- | --- | --- |
| T019 | Equivalent lattice representatives | Exact Gauss/Lagrange reduction of MᵀGM for all 308 SL(2,Z) matrices with entries ≤ 5 and 200 seeded words, on six lattices; count closed-domain reduced bases; sympy recomputation | 0 failures in 3048 reductions; reduced-basis counts 2 (generic), 4 (vertical boundary, arc, square), 12 (hexagonal), 2 (rectangular), as the stabilizers predict; det 2, det −1 and non-integer changes refused |
| T020 | Classify geodesics by winding | Exact event-driven flow in lattice coordinates for all windings with \|m\|, \|n\| ≤ 6; exact intersection counts; golden-slope returns in binary64 and with mpmath | First return at t = 1/gcd after (\|m\|+\|n\|)/gcd crossings; intersections = \|det\| for 120 pairs; golden gaps φ⁻ᵏ with q·gap → 1/√5; binary64 φ = p/2⁴⁹ is rational |
| T021 | Shortest vs least sensitive | Exact translates q − p + λ; each route integrated with `ciw.lab.jacobi` on the plane chart | j_head(L) = L (error 5e-16), so the length and amplification orders are identical: theorem-backed and numerically checked; special to K = 0 |
| T022 | Degenerate shortest representatives | Exact nearest translates; 12 × 12 census; exact Voronoi-vertex (cut-locus) enumeration | Multiplicities 2 / 4 at half periods; census {1: 121, 2: 22, 4: 1}; Σ(k_v − 2) = 2 on six lattices; binary64 reports a tie at ε = 2⁻⁶⁰ where the exact answer is unique |
| T023 | Sensitivity across headings | Return distance r(θ; L) over 7200 headings (closed-form segment-to-lattice distance) | 18 zeros = the 18 primitive directions with \|v\| ≤ 3; V-slope \|v\| (error 6e-12); closure basins 2 arcsin(ε/\|v\|) widest for the shortest classes |
| T024 | Focus-margin-aware ranking | Torus(2, 1), p = (0, 0.4), q = (2, −0.3): 360-heading fan, batch Newton, `ciw.lab.jacobi` verification, scipy DOP853 check | 8 routes; the orders by length, amplification and margin all differ (table below); a flat torus has no conjugate points |
| T025 | Pareto fronts | Exhaustive dominance on (L, \|j_head\|, −margin) | Front {0, 1, 2, 3, 4} of 8; length/amplification front {0, 1, 2}; length/margin front {0, 3} |
| T026 | Modular-reduction invariance | Exact spectra (Q ≤ 60), det and systole under 368 matrices; the naive label rule; mirror and index-2 examples; FTR fold length pairs | Exactly invariant; the naive rule c′ = Mc fails 6428 times; the mirror image is isospectral but not SL(2,Z)-equivalent; det 2 multiplies the squared area by 4 |
| T027 | Translation-surface examples | Exact polygons (Q, Q(√2), Q(√3)), edge involutions, topology, exact horizontal flows | L-shape and octagon: genus 2, one vertex; octagon cylinders 2 + √2 and 1 + √2 with areas summing to 2 + 2√2; non-translation and unequal gluings refused |
| T028 | Geodesics across glued edges | Exact rational flow on the L-shape; exact Q(√2) against float flow on the octagon | All 96 rational trajectories close (multiplier 1–3) or hit the cone point (6 saddle connections); float matches exact to 2e-15; near-vertex passes refused within tolerance 1e-9 |
| T029 | Cone singularities | Union-find on corners; exact corner angles; commutator cycles for origamis | Octagon and L-shape 6π; H(1,1) origami 4π + 4π; pillowcase 4 × π; tori regular; Gauss–Bonnet defect 0 everywhere |
| T030 | Smooth vs discrete geodesics | Dijkstra (4/8 stencils) and fast marching on N = 30, 60, 120 periodic grids; scipy csgraph check | Graph ratios fixed under refinement: √2 at 45° (4-nbr), 1.08239 at 22.6° (8-nbr), worst sqrt(4 − 2√2) at 22.5°; fast-marching error 5.1% → 3.2% → 1.9% |
| T031 | Route changes under metric perturbation | g = I + εh, h = [[0, 1], [1, 0]]; exact minimization over translates; sympy solve | Switch at ε* = 4δ exactly (1/250 for δ = 1/1000); heading jumps 126.9° while the minimal length stays continuous |
| T032 | Counterexample library | Five witness configurations plus two searches that cannot produce a witness | See below |

### T024 routes (Torus(2, 1), p = (0, 0.4), q = (2, −0.3))

| Route | Length | \|j_head(L)\| | Focus margin |
| --- | --- | --- | --- |
| 0 | 5.8465 | 1.834 | 1.245 |
| 1 | 5.9786 | 0.788 | 0.860 |
| 2 | 5.9986 | 0.437 | −0.457 (past a conjugate point) |
| 3 | 6.3545 | 35.85 | > 8 (censored) |
| 4 | 7.7167 | 35.21 | > 8 (censored) |
| 5 | 8.8016 | 134.5 | 3.749 |
| 6 | 8.9883 | 130.2 | 4.374 |
| 7 | 9.6549 | 172.8 | > 8 (censored) |

The shortest route (0) is neither the least amplifying (2), which passes a
conjugate point, nor the one with the largest margin (3), which amplifies
heading errors about 20 times more. The route set is what the fan search
found up to length 11; completeness is not proven.

## Counterexample library for "shortest means safest" (T032)

Each witness is retained in `artifacts/T032/counterexample-library.json` and
recorded as a finding whose check passes when the violation is observed.

| Refuted statement | Witness | Numbers |
| --- | --- | --- |
| The shortest geodesic has the least heading amplification | Torus(2, 1), inner equator (0, π) → (2.5, π) | shortest L = 2.5, amplification sinh 2.5 = 6.050 (exact, K = −1); a route of length 7.405 has amplification 2.242 |
| The shortest geodesic has the largest focus margin | Torus(2, 1), outer equator (0, 0) → (1.5, 0) | shortest margin π√3 − 4.5 = 0.941 (exact, K = 1/3); a route of length 6.723 has no conjugate point within 8 (margin ≥ 8.04) but amplification 29.5 |
| The shortest route is unique | Torus(2, 1), (0, 0) → (2.2, 0) | two mirror-image shortest routes of length 6.30866770804 (headings ±0.933); the equator route (6.6) lies past its conjugate point |
| Low amplification certifies a robust (locally minimizing) route | GaussianBump(1.5, 1), (−2.5, 0) → (2.5, 0) | side routes (shortest, 5.718) have amplification 6.88; the straight route over the top (5.878) has 6.49 but margin −2.00 |
| A shortest geodesic stays well away from conjugate points | Unit sphere, separation π − 0.05 | minor-arc margin 0.05, j_head = sin 0.05, targeting condition number 1/sin 0.05 ≈ 20 |

Searches that cannot produce a witness, recorded as such: on a flat torus
j_head(s) = s, so the shortest route is exactly the least sensitive (T021). On
the saddle z = (x² − y²)/2 (K < 0, simply connected) the search finds exactly
one route, consistent with Cartan–Hadamard uniqueness. The witnesses show that
the general statements are false; they do not rank routes for any
application.

## Evidence and provenance

- Exact integer, `Fraction` and Q(√d) computations back `numerically_verified`
  findings through `exact_arithmetic` checks. Same-origin comparisons (batch
  RK4 against `ciw.lab.jacobi`, float octagon against exact octagon) use
  `cross_implementation` checks and are never labelled independent.
- Independent checks, when the optional modules are installed: sympy (T019
  matrix algebra, T031 threshold), mpmath (T020 golden gaps), scipy (T024 and
  T032 re-integration with DOP853, T030 graph distances). Without them the same
  findings are `numerically_verified`.
- The pinned Flat-Torus-Geodesic-Reference (revision `dc918562…`, tree
  `1f082c6b…`, the pin in `ciw.geodesic_reference.PINS`) runs in a separate
  Python 3.12 interpreter and is compared on folds (T019), loop lengths,
  crossing counts and area (T020), and fold length pairs (T026). A checkout
  that is unreadable, dirty or at another revision is refused, and the task is
  reported `partial`. In the retained run the provider agreed to 1.1e-12
  (reduced τ), 2e-16 (lengths) and 5e-13 (length pairs). Tests gate on
  `CIW_LAB_FTR_REPO` and `CIW_LAB_FTR_PYTHON`.

## What these results do not establish

- Finite enumerations (matrices with bounded entries, windings with
  |m|, |n| ≤ 6, 120 intersection pairs, six lattices) check theorems on
  declared cases. The general statements rest on the cited derivations.
- Non-closure of irrational directions cannot be observed in floating point:
  every binary64 slope is rational (T020).
- Route sets on curved surfaces come from a fan search and may be
  incomplete. Focus margins use conjugate points of the start only.
- Grid results cover periodic unit grids and displacements within half a
  period; the fast-marching order (0.71) is an empirical fit.
- T031 perturbs the metric by a constant. Spatially varying perturbations also
  bend routes continuously and are not studied.
- No computation here says whether any route is safe to follow, whether a
  sensor achieves a closure tolerance, or whether a calibrated metric
  separates near-tied routes. These are recorded as `not_established`.

## Deferred research questions

- A certified completeness bound for the route search (for example, interval
  shooting over all headings up to a length).
- Cylinder decompositions of the octagon in every periodic direction
  (Veech group action), not only the horizontal one.
- Conjugate points of the target along the reversed route, and a symmetric
  focus margin.
- Spatially varying metric perturbations, and route switching on curved
  surfaces (surfaces-discrete section).
