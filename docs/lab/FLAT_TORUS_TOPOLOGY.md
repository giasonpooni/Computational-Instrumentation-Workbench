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
physical part, vehicle, sensor or workspace. Route safety, sensor closure
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
  extended geodesic: the first zero s_c > 0 of j_head. Only conjugate points
  count; zeros of the lateral column j_lat (focal points of lateral start
  offsets) do not. This is the queue's one definition of "focus margin", used
  in the claims of T024, T025 and T032; a ratio such as (nearest focal or
  conjugate point)/L, which also counts zeros of j_lat, is a different quantity
  and is not a focus margin. A negative margin means the route has passed a
  conjugate point and is not locally minimizing. The margin is censored (reported as
  `null` with `margin_lower_bound`) when no conjugate point occurs within the
  extension horizon; censored margins are only known to exceed the horizon, so
  those routes are tied.
- Targeting condition number 1/|j_head(L)|: the heading change needed per unit
  normal displacement of the target. A small amplification is good for
  open-loop execution but means ill-conditioned targeting, and it becomes
  small exactly as q approaches a conjugate point. Low amplification alone is
  therefore not robustness, and reports give both numbers.

## Tasks

| Task | Question | Method | Key result (retained run) |
| --- | --- | --- | --- |
| T019 | Equivalent lattice representatives | Exact Gauss/Lagrange reduction of MᵀGM for all 308 SL(2,Z) matrices with entries ≤ 5 and 200 seeded words, on six lattices; a second ciw reduction (vector-form Lagrange on the basis vectors) recomputes every canonical form as a cross-implementation check; count closed-domain reduced bases | 0 failures in 3048 reductions, and the second reduction agrees on all 3048 (including the boundary rule b ≥ 0); reduced-basis counts 2 (generic), 4 (vertical boundary, arc, square), 12 (hexagonal), 2 (rectangular), as the stabilizers predict; det 2, det −1 and non-integer changes refused, and the det-2 image has a different canonical form |
| T020 | Classify geodesics by winding | Exact segment-by-segment flow in lattice coordinates for all windings with \|m\|, \|n\| ≤ 6, returns detected by an exact solve on each segment; exact intersection counts; lattice-point count against a brute-force box; golden-slope returns in binary64 and with mpmath | Measured first return at t = 1/gcd with the primitive displacement, gcd returns by t = 1, (\|m\|+\|n\|)/gcd crossings; intersections = \|det\| for 120 pairs; 229 lattice points within R = 20, equal to the box count; golden gaps equal φ⁻ᵏ (ratio ≥ 0.99999993); binary64 φ = p/2⁴⁹ is rational (an `analytic` finding from the IEEE 754 format, so T020's headline label is `analytic`) |
| T021 | Shortest vs least sensitive | Exact translates q − p + λ; each route integrated with `ciw.lab.jacobi` on the plane chart | j_head(L) = L (error 5e-16), so the length and amplification orders are identical: theorem-backed and numerically checked. The equivalence holds whenever j_head(L) is one increasing function of L for all routes (constant K ≤ 0) and can fail with K > 0 somewhere or curvature that differs between routes |
| T022 | Degenerate shortest representatives | Exact nearest translates; 12 × 12 census; exact Voronoi-vertex (cut-locus) enumeration; binary64 with and without a relative tolerance | Multiplicities 2 / 4 at half periods; census {1: 121, 2: 22, 4: 1}; Σ(k_v − 2) = 2 on six lattices; rounding the target (1/2 + 2⁻⁶⁰, 1/4) to binary64 turns multiplicity 1 into 2; at 6 exact ties that binary64 cannot represent, raw comparison undercounts 4 and the tolerance 1e-9 undercounts none |
| T023 | Sensitivity across headings | Return distance r(θ; L) over 7200 headings (closed-form segment-to-lattice distance) | 18 zeros = the 18 primitive directions with \|v\| ≤ 3; V-slope \|v\| (error 6e-12); measured closure basins widest for the shortest classes, with 0 discordant pairs |
| T024 | Focus-margin-aware ranking | Torus(2, 1), p = (0, 0.4), q = (2, −0.3): 1440-heading fan (rerun at 2880), one Newton seed per miss-distance minimum, `ciw.lab.jacobi` verification, scipy DOP853 on a sympy-derived field | 9 routes, unchanged at double density; the shortest route is neither the least amplifying nor in the best focus-margin group (table below); a flat torus has no conjugate points |
| T025 | Pareto fronts | Exhaustive dominance on (L, \|j_head\|, −margin); every front recomputed from the route fields by a numpy dominance matrix (three objectives) and a sort-and-sweep (two objectives) | Front {0, 1, 2, 3, 4} of 9 (route 2 lies past a conjugate point); length/amplification front {0, 1, 2}; length/margin front {0, 3}; the second computation agrees on every front |
| T026 | Modular-reduction invariance | Exact spectra (Q ≤ 60), det and systole under 368 matrices; area-one float spectra under the same 368 matrices; the naive label rule; mirror and index-2 examples; FTR fold length pairs | Exactly invariant; float spectra agree within 1e-8 (1.1e-10, limited by the conditioning of bases with entries up to 85, not by unit roundoff); the naive rule c′ = Mc fails 6428 times (witness: generic lattice, M = [[−5, −4], [−1, −1]], c = (1, 0), Q(c) = 5 but Q′(Mc) = 5153); the mirror image is isospectral but not SL(2,Z)-equivalent; a det-2 matrix changes the spectrum (and, by det(M)² det G, multiplies the squared area by 4) |
| T027 | Translation-surface examples | Exact polygons (Q, Q(√2), Q(√3)), edge involutions, topology, exact horizontal flows | L-shape and octagon: genus 2, one vertex class (checked); octagon cylinders 2 + √2 and 1 + √2 with areas summing to 2 + 2√2; non-translation and unequal gluings refused |
| T028 | Geodesics across glued edges | Exact rational flow on the L-shape; exact Q(√2) against float flow on the octagon; a generic float trajectory (θ = 0.3) against the exact Q(√2) trace of its binary64 start and direction | All 96 rational trajectories (32 primitive directions × 3 starts) close (multiplier 1–3) or hit the cone point (6 saddle connections); float matches exact to 2e-15; the generic trajectory keeps the exact crossing sequence for 400 crossings (position deviation 1.2e-14) and comes no closer than 2.43e-4 (Euclidean) to a vertex, below the along-edge clearance 2.74e-4 at the crossings; near-vertex passes and starts within tolerance 1e-9 of an edge refused |
| T029 | Cone singularities | Union-find on corners; exact corner angles; commutator cycles for origamis; Euler characteristic known independently (declared topology, Riemann–Hurwitz for origamis) | Octagon and L-shape 6π; H(1,1) origami 4π + 4π; pillowcase 4 × π; tori regular; Gauss–Bonnet defect 0 against the independent χ everywhere |
| T030 | Smooth vs discrete geodesics | Dijkstra (4/8 stencils) and fast marching on N = 30, 60, 120 periodic grids; scipy csgraph check | Graph ratios fixed under refinement: √2 at 45° (4-nbr), 1.08239 at 22.6° (8-nbr), worst sqrt(4 − 2√2) at 22.5°; fast-marching error 5.1% → 3.2% → 1.9% |
| T031 | Route changes under metric perturbation | g = I + εh, h = [[0, 1], [1, 0]]; exact minimization over translates; sympy solve | Switch at ε* = 4δ exactly (1/250 for δ = 1/1000); between sweep points ε = 0.0039 and 0.0041 the heading jumps 126.9° (exact up to rounding: each route's heading does not depend on ε) while the minimal length changes by 9e-8, within the continuity bound 4.5e-5 (bracket width × max \|dL/dε\|) |
| T032 | Counterexample library | Four shortest-versus-alternative witnesses, one near-conjugate conditioning witness, and two searches that cannot produce a witness | See below |

### T020: what the golden-slope numbers mean

By Binet's formula, ‖F_k φ‖ = φ⁻ᵏ and |F_k φ⁻ᵏ − 1/√5| = φ⁻²ᵏ/√5 exactly, so
q·gap → 1/√5 is a theorem. At k = 22 the exact residual is 2.85e-10. The
binary64 value of q·gap deviates from 1/√5 by 3.1e-8, which is float
cancellation in q·φ − round(q·φ), not a measured convergence rate. The check
confirms that the float deviation stays within φ⁻²ᵏ/√5 plus a proven rounding
bound. A closure would make a gap vanish, so the check requires gap/φ⁻ᵏ ≥ 0.5
at every return.

### T024 routes (Torus(2, 1), p = (0, 0.4), q = (2, −0.3))

| Route | Length | \|j_head(L)\| | 1/\|j_head(L)\| | Focus margin |
| --- | --- | --- | --- | --- |
| 0 | 5.8465 | 1.834 | 0.545 | 1.245 |
| 1 | 5.9786 | 0.787 | 1.27 | 0.860 |
| 2 | 5.9986 | 0.437 | 2.29 | −0.457 (past a conjugate point) |
| 3 | 6.3545 | 35.85 | 0.028 | > 8 (censored) |
| 4 | 7.7167 | 35.21 | 0.028 | > 8 (censored) |
| 5 | 8.3289 | 197.2 | 0.005 | > 8 (censored) |
| 6 | 8.8016 | 134.5 | 0.007 | 3.749 |
| 7 | 8.9883 | 130.2 | 0.008 | 4.374 |
| 8 | 9.6549 | 172.8 | 0.006 | > 8 (censored) |

The shortest route (0) is not the least amplifying: route 2 amplifies less,
but it lies past a conjugate point and its targeting condition number is 2.3.
Route 0 is also not in the best focus-margin group. Routes 3, 4, 5 and 8 have
no conjugate point within the horizon, so their margins are tied at "more than
about 8". Those routes amplify heading errors 20 to 100 times more than
route 0. Focus-margin groups, best first: [[3, 4, 5, 8], [7], [6], [0], [1], [2]].

A 360-heading fan misses route 5. At 1440 headings the search finds 9 routes,
and 2880 headings give the same 9. Each run of adjacent rays that pass the same
lift of q seeds Newton once per miss-distance minimum. No seed was dropped as
singular, divergent, over length or failing the residual test, and none of
the retained runs merged duplicates. The retained `torus-routes.json` records
these counts.

## Counterexample library for "shortest means safest" (T032)

Each witness is retained in `artifacts/T032/counterexample-library.json` and
recorded as a finding whose check passes when the violation is observed.
Every configuration is searched with a 720-heading fan and rerun at 1440; all
route sets agree.

| Refuted statement | Witness | Numbers |
| --- | --- | --- |
| The shortest geodesic has the least heading amplification | Torus(2, 1), inner equator (0, π) → (2.5, π) | shortest L = 2.5, amplification sinh 2.5 = 6.050 (exact, K = −1); a route of length 7.405 has amplification 2.242 |
| The shortest geodesic has the largest focus margin s_c − L | Torus(2, 1), outer equator (0, 0) → (1.5, 0) | shortest margin π√3 − 4.5 = 0.941 (exact, K = 1/3); a route of length 6.723 has no conjugate point within 8 (margin ≥ 8.04) but amplification 29.5 |
| The shortest route is unique | Torus(2, 1), (0, 0) → (2.2, 0) | two mirror-image shortest routes of length 6.30866770804 (headings ±0.933); the equator route (6.6) lies past its conjugate point |
| Low amplification certifies a robust (locally minimizing) route | GaussianBump(1.5, 1), (−2.5, 0) → (2.5, 0) | side routes (shortest, 5.718) have amplification 6.88; the straight route over the top (5.878) has 6.49 but margin −2.00 |
| Minimizing geodesics on the unit sphere have focus margin s_c − L bounded below by a positive constant | Unit sphere, separations π − δ for δ = 0.1, 0.01, 0.001 (a near-conjugate conditioning witness, not a shortest-versus-alternative one) | `ciw.lab.jacobi` margins 0.1, 0.01, 0.001 (to 1e-7 relative), j_head = sin δ, targeting condition up to 1/sin 0.001 ≈ 1000 |

Searches that cannot produce a witness are recorded as such. On a flat torus
j_head(s) = s, so the shortest route is exactly the least sensitive (T021),
and the same holds for constant K < 0 (j_head = sinh s). On the saddle
z = (x² − y²)/2 (K < 0, simply connected) the search finds exactly one route,
consistent with Cartan–Hadamard uniqueness. The witnesses show that the
general statements are false; they do not rank routes for any application.

## Evidence and provenance

- Exact integer, `Fraction` and Q(√d) computations back `numerically_verified`
  findings through `exact_arithmetic` checks. Float results compared with an
  exact or closed-form value (fast-marching errors, the ε* bisection) use
  `analytic` checks. Same-origin comparisons are `cross_implementation`
  checks and are never labelled independent: batch RK4 against
  `ciw.lab.jacobi`, float octagon against exact octagon, the second
  (vector-form) lattice reduction of T019, and the second Pareto computation
  of T025 (numpy dominance matrix, sort-and-sweep). The T019 second reduction
  recomputes the canonical form of all 3048 bases and words and compares it
  with canonical(G), with `lat.gauss_reduce` and with the boundary rule; a
  test confirms that a wrong boundary representative is caught. A test also
  confirms that a wrong dominance rule, which builds a wrong T025 front, is
  refuted by the second computation.
- The generic T028 octagon trajectory is traced again exactly in Q(√2) from
  the same binary64 start and direction (dyadic rationals). The checks compare
  the crossing sequences, bound the float position deviation by the declared
  tolerance, and require the Euclidean closest approach of the exact
  trajectory to a vertex to exceed the tolerance plus that deviation. The
  along-edge clearance at the crossings cannot fail that test: the float flow
  refuses any crossing within the tolerance.
- Findings that hold by definition or format are `analytic`, not checked:
  the rationality of binary64 slopes (T020, which makes T020's headline
  `analytic`), and det(MᵀGM) = det(M)² det G for the det-2 examples (T019,
  T026).
- The route-set stability findings (T024, T032) declare no completeness
  bound (uncertainty value `null`), because the fan search is not proven
  complete.
- Independent checks, when the optional modules are installed:
  - sympy (T031): solves for the threshold ε*.
  - mpmath (T020): 50-digit golden-slope gaps.
  - scipy with sympy (T024 route table; T032 inner-equator, outer-equator and
    bump witnesses): sympy derives the metric, Christoffel symbols, Gauss
    curvature and heading frame from the embedding X(φ, θ) or the Monge
    graph, and scipy integrates that field with DOP853. It shares no
    hand-written geometry with the batch search or `ciw.lab.surfaces`. Routes
    where only one code finds a conjugate point, and solver runs that stop
    early, are counted as failing checks rather than as infinite errors.
  - scipy (T030): `scipy.sparse.csgraph.dijkstra` on the same stencil graphs.

  Without these modules the same findings are `numerically_verified`.
- The Euler characteristic in the T029 Gauss–Bonnet check comes from the
  declared topology (and from Riemann–Hurwitz for origamis), not from the
  vertex classes. With χ = V − E + F from the same classes the identity holds
  for any partition of the corners. A test confirms that deliberately wrong
  vertex classes are caught.
- The pinned Flat-Torus-Geodesic-Reference (revision `dc918562…`, tree
  `1f082c6b…`, the pin in `ciw.geodesic_reference.PINS`) runs in a separate
  Python 3.12 interpreter. It is compared on folds (T019), loop lengths,
  crossing counts and area (T020), and fold length pairs (T026, retained as
  `ftr-fold-lengths.json`). A checkout that is unreadable, dirty or at another
  revision is refused, as is provider output that is not the expected JSON or
  has missing rows. In those cases the task is reported `partial`. The three
  provider claims are always reported: when the provider is unbound or
  refused they are `not_established` (flagged as expected) with the reason as
  their value. A bound and an unbound run therefore have the same claims and
  differ only in those findings' labels and values. In the retained
  provider-bound run the provider agreed to 1.1e-12 (reduced τ), 2e-16
  (lengths) and 5e-13 (length pairs). Tests gate on `CIW_LAB_FTR_REPO` and
  `CIW_LAB_FTR_PYTHON`.

## What these results do not establish

- Finite enumerations (matrices with bounded entries, windings with
  |m|, |n| ≤ 6, 120 intersection pairs, six lattices) check theorems on
  declared cases. The general statements rest on the cited derivations.
- Non-closure of irrational directions cannot be observed in floating point:
  every binary64 slope is rational (T020).
- Route sets on curved surfaces come from a fan search. They are stable under
  doubling the fan density but may still be incomplete. Focus margins use
  conjugate points of the start only.
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
- The counterexample library on variable-curvature triangle meshes (T032's
  next step). Manufacturing paths are not open here: T137 retains the
  counterexample "The shortest route between a station and an edge is also the
  safest route".

Each task's `recommended_next_task` is its own deferred research question from
the list above or from its unresolved assumptions, never a pointer to a queue
task that has already run.
