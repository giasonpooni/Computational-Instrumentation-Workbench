# Triangle-mesh geodesics and geometry uncertainty (T038–T044)

This note documents the triangle-mesh experiments of section 3 of the
computational-experimentalist queue: the algorithms, the convergence results,
the named refusal states and, above all, what the results do not establish.

**Scope.** Generated meshes only (icosphere, latitude–longitude sphere, prism
cylinder, Schwarz lantern, planar grids, torus grids) in normalized units, with
smooth references from `ciw.lab.surfaces` and `ciw.lab.jacobi`.

**Non-claims.** No scanned surface, scanner, tracker or tape was measured. Every
statement about real scanned surfaces, real sensors or production thresholds is
recorded as a `physical`, `calibration`, `sensor_performance` or
`production_acceptance` finding with evidence status `not_established`. Numbers
quoted below are from a retained run (`ciw lab run T038 … T044`); the reports
and artifacts are the authoritative record, and every label is assigned by
`ciw.lab.evidence.finding`. Every computational finding carries a per-finding
uncertainty (roundoff, truncation bound, Monte Carlo 95 % half-width or the
spread of pairwise local orders) and a regression tolerance.

| File | Contents |
| --- | --- |
| `src/ciw/lab/surfaces_discrete_mesh_geometry.py` | mesh type, generators, validator, tracer, distances, curvature, strip unfolding |
| `src/ciw/lab/surfaces_discrete_mesh_studies.py` | deterministic studies (plain numbers) |
| `src/ciw/lab/surfaces_discrete_mesh.py` | task registrations T038–T044, findings and report fields |
| `tests/test_lab_surfaces_discrete_mesh.py` | regression tests (about 20 s) |

## Algorithms

### Straightest geodesics (T038)

A geodesic is traced as an initial-value problem in the Polthier–Schmies
sense: straight inside a face, and at an edge with unit direction `ê` the
outgoing direction keeps the edge component and the magnitude of the
perpendicular component,

    d' = (d·ê) ê + |d − (d·ê) ê| ŵ₂,

where `ŵ₂` is the in-plane unit vector of the next face perpendicular to the
edge and pointing into that face. This is rotation of the direction about the
edge onto the next face, so the angles on both sides of the edge are equal and
the unfolded path is a straight line. Exit edges are found in a local 2D frame
of each (counterclockwise) face; only edges the ray leaves through are
candidates.

Declared rules:

- **Vertex hits are refused** (`vertex_hit`) when the edge parameter of a
  crossing is within `1e-9` of an endpoint. The Polthier–Schmies continuation
  (bisect the total vertex angle) is not implemented; it is a declared
  extension, not a silent default.
- **Boundaries stop the trace** (`boundary_reached`) and the partial path and
  its length are retained.

### Distances

- **Edge Dijkstra** (heap based): shortest edge path, an upper bound on the
  polyhedral distance. Compared with a dense Floyd–Warshall (ciw code, a
  same-origin `cross_implementation` check) and, when scipy is installed, with
  `scipy.sparse.csgraph.dijkstra` (independent). Without scipy the finding is
  `numerically_verified` instead of `independently_verified`; its claim and the
  report prose are the same in both environments.
- **Steiner graph**: `k` equally spaced points per edge, all node pairs inside
  each face joined. Every graph edge is a straight segment in a face, so graph
  distances are upper bounds; with `k = 2^j − 1` the node sets are nested and
  distances cannot increase with `k`.
- **Heat method** (Crane, Weischedel and Wardetzky 2013) with `t = h²`, lumped
  mass and cotangent Laplacian. The heat solution decays like `exp(−d²/4t)`, and
  its far-field gradient direction needs relative accuracy, so both systems are
  solved directly by a banded Cholesky factorization in reverse Cuthill–McKee
  order written with elementwise NumPy only. Meshes above 3000 vertices (the
  default limit) are refused with `mesh_too_large`; the studies use at most
  642 vertices. (A Krylov solver was tried and rejected: its iterates vanish
  beyond the rings it has reached, leaving zero gradients far from the source.
  Threaded LAPACK was also avoided because its run time varied by two orders of
  magnitude on a contended machine.)
- **Strip unfolding**: a face sequence is laid out in the plane from its edge
  lengths; the planar distance between barycentric markers is the length of the
  straight path through the strip when the segment stays inside it (reported
  as a margin). It is a second ciw implementation of the traced length (it does
  not use the tracer's 3D edge rotation), so agreement is a same-origin
  consistency check, not independent verification; it is also the smooth
  observable used for uncertainty. Faces that do not share an edge are refused
  with `invalid_strip`.

### Curvature and normals

- Angle defect `K_v = (2π − Σθ)/(A_v/3)` with the barycentric area `A_v/3`;
  also with the mixed Voronoi area of Meyer, Desbrun, Schröder and Barr.
  Boundary vertices are refused (`boundary_vertex_curvature`).
- Discrete Gauss–Bonnet `Σ(2π − Σθ) = 2πχ` is exact for closed meshes and is
  checked (residual about `4e-11`).
- Vertex normals: area-weighted sums of the one-ring face normals.

## Refusal states (T042)

Validation reports every defect, in this order, and `TriMesh.build` refuses
with the first one (all are attached to the exception):

| Code | Stage | Meaning |
| --- | --- | --- |
| `invalid_shape` | validation | vertices not `(n, 3)` or faces not `(m, 3)` integers |
| `empty_mesh` | validation | fewer than three vertices or no face |
| `nonfinite_vertex` | validation | NaN or infinite coordinate |
| `invalid_face_index` | validation | face refers to a missing vertex |
| `degenerate_face` | validation | repeated index or `2A/l_max² ≤ 1e-12` |
| `non_manifold_edge` | validation | edge shared by more than two faces |
| `inconsistent_orientation` | validation | both faces traverse an interior edge in the same direction |
| `non_manifold_vertex` | validation | vertex joins several edge-connected face fans (bowtie) |
| `folded_face` | validation | neighbouring unit normals with dot `≤ −0.9` (bend over 154°) |
| `unreferenced_vertex` | validation | vertex in no face |
| `disconnected_components` | validation | more than one face component (when connectivity is required) |
| `open_boundary` | validation | boundary edges on a mesh declared closed |
| `point_outside_face` | tracing | start point not on the start face |
| `invalid_direction` | tracing | zero, nonfinite or non-tangent start direction |
| `boundary_reached` | tracing | a hole or the outer boundary reached before the length |
| `vertex_hit` | tracing | the straightest continuation is not unique |
| `step_budget_exceeded` | tracing | face-crossing budget exhausted |
| `unreachable_target` | query | target in another component |
| `boundary_vertex_curvature` | query | angle defect requested at a boundary vertex |
| `mesh_too_large` | query | the direct heat method's vertex limit is exceeded |
| `invalid_strip` | query | consecutive strip faces do not share an edge |

T042 constructs 23 defect cases (21 distinct codes); all are refused with the
expected code, four valid controls report no issue, and a mesh with three
defects reports all three in the declared order. The counterexample recorded
alongside: evaluating curvature and normals on an unvalidated zero-area face
produces nonfinite values, so the refusals are necessary, not cosmetic. The
fold check was added after T041 showed that tangential jitter of `0.2 h` folds
faces while every structural check still passes; T041 now checks the fold
refusal of every jittered mesh against an independent indicator (a face normal
pointing into the sphere).

The hypothesis is scoped to the declared catalogue. **Not detected:**
self-intersections between non-adjacent faces, unwelded seams (coincident
duplicate vertices pass validation and surface only as `boundary_reached`
during tracing), duplicate faces, and near-degenerate slivers just above the
`1e-12` threshold. **False positives:** a legitimate sharp crease with a bend
over about 154° is refused as `folded_face` (the embedded `m = n²` lantern of
T041 is refused although it does not overlap itself).

## Results

### Solver checks (T038)

| Check | Result |
| --- | --- |
| Sheared planar meshes (min angle down to 11.9°): trace vs straight line | `8.9e-16` |
| Prism cylinder, n = 8…128: trace vs exact development of the mesh | `8.0e-15` |
| 42 sphere traces (levels 1–7): strip layout length vs traced length | `3.1e-15` |
| Edge Dijkstra vs Floyd–Warshall and scipy | `8.9e-16` (`independently_verified` with scipy) |
| Nested Steiner distances, k = 0, 1, 3, 7 | never increase, never below the chord |
| Steiner distance − traced length (level 2, length 1) | mean 3.0e-2 (k=1), 1.3e-2 (k=3), 5.3e-3 (k=7), all positive |

The Steiner sandwich is consistent with the traced geodesics being shortest
paths; it does not prove it.

### Convergence under refinement (T039)

Six declared geodesics of length 2 on icospheres (unit sphere), endpoint
compared with the great circle after radial projection:

| Level | h | mean endpoint error | mean cross-track | mean length defect |
| --- | --- | --- | --- | --- |
| 1 | 0.582 | 9.8e-2 | 5.0e-2 | 7.4e-2 |
| 2 | 0.299 | 3.7e-2 | 2.8e-2 | 1.9e-2 |
| 3 | 0.151 | 1.2e-2 | 1.1e-2 | 4.6e-3 |
| 4 | 0.075 | 3.1e-3 | 2.6e-3 | 1.2e-3 |
| 5 | 0.038 | 1.1e-3 | 1.0e-3 | 2.9e-4 |
| 6 | 0.019 | 5.0e-4 | 4.9e-4 | 7.3e-5 |
| 7 | 0.0094 | 1.2e-4 | 1.1e-4 | 1.9e-5 |

- **Length defect: order 2.00** (pairwise local orders 1.98–2.06). The
  inscribed polyhedron's intrinsic metric differs from the sphere's by `O(h²)`.
- **Endpoint (lateral) error: at least first order, not a clean power law.**
  The least-squares order over levels 2–7 is 1.63, but the pairwise local
  orders are 1.59, 2.02, 1.52, 1.08, 2.11 (cross-track 1.37, 2.06, 1.40, 1.02,
  2.12), and one of 36 per-trace level steps increases the error (trace 6,
  level 5 → 6). Curvature on a mesh is a set of point masses at vertices; a
  straightest geodesic is deflected, relative to the smooth one, by the
  imbalance of vertex curvature on its two sides (a discrepancy sum along the
  path). On the icosphere the lattice rows project to great circles, so paths
  nearly parallel to a row accumulate this imbalance coherently (`O(h)`),
  generic paths less so. The order is an empirical slope for six directions.
- **Prism cylinder: exact closed form.** The mesh develops to a flat strip of
  circumference `2nR sin(π/n)`, and an endpoint at fraction `t` of chord `i` has
  azimuth `2ai + a + atan((2t − 1) tan a)` with `a = π/n`. The helix error is
  therefore

      R |(L cos α / R)(a / sin a − 1) + atan(s tan a) − s a|,   s = 2t − 1,

  a development deficit `≈ L cos α π²/(6n²)` plus a chord-position term bounded
  by `2a³/(9√3) R`. The measured error equals this to 4e-15. Consequently
  `n² × error / (L cos α π²/6)` tends to 1 with an `O(1/n)` oscillation that
  depends on where the endpoint falls on its chord: 1.077, 0.948, 1.018, 0.989,
  0.994 for n = 8…128, each within the derived bound. The fitted order is 2.02.
- **Distances from vertex 0** (levels 1–4): the edge-graph maximum signed
  relative error is 0.145, 0.211, 0.230, 0.234 — a floor, not convergence
  (counterexample to "edge Dijkstra converges"). The floor has a closed form:
  in the limit the star of the valence-5 source is a regular 5-star with rows
  72° apart, and the vertex two hops away between two rows sits at `e₁ + e₂`,
  so edge path / geodesic → `2/(2 cos(π/5)) = √5 − 1`. Aitken extrapolation of
  levels 2–4 gives 0.23613 against `√5 − 2 = 0.23607`. Steiner graphs with
  fixed `k` also keep a floor (k = 3: max |relative error| 0.031, 0.008,
  0.011, 0.014; the level-1 value is large because the inscribed mesh is
  shorter than the sphere). The heat method converges at order 1.01 (levels
  1–3).

### Jacobi fields and curvature (T040)

The finite-difference Jacobi estimate uses paired traces at headings `±δ`:
`j = |X₊(L) − X₋(L)| / (2 sin δ)`, which equals `sin L` exactly on the smooth
unit sphere; the smooth `ciw.lab.jacobi` transfer reproduces `sin L` to 7e-11.

- **δ = 0.1, h → 0: converges.** Mean error 2.2e-1, 8.1e-2, 2.4e-2, 6.2e-3,
  3.2e-3, 1.2e-3 at levels 2–7; least-squares order 1.53 with pairwise local
  orders 1.43, 1.75, 1.95, 0.96, 1.45, so the rate is not a clean power law.
- **h fixed, δ → 0: the flat answer.** At δ = 1e-5 all 30 pairs on levels 2–6
  cross the same face sequence; their common development is a plane, so
  `|X₊ − X₋| = 2L sin δ` and `j = L = 2` exactly (deviation 5e-10), an error of
  `L − sin L = 1.0907` at every one of those levels. Curvature lives at
  vertices, and two geodesics feel it only when a vertex lies between them.
  At level 7 the wedge between a pair is expected to contain about 0.37
  vertices (`2δ(1 − cos L)·V/4π`), and 4 of 6 pairs straddle a vertex. The two
  limits do not commute: mesh Jacobi fields require the swept wedge to contain
  many vertices (`δ L ≫ h` in the linear regime).
- **Valence-5 vertices do not converge to K.** On a polyhedron inscribed in the
  unit sphere the angle defect at a vertex equals the area of the Gauss-image
  polygon of its face normals, and each face normal points at the face
  circumcenter, so in the limit the defect is the Voronoi-cell area. For a
  regular valence-n star of isosceles triangles with apex angle `2π/n` and legs
  `ℓ`, the Voronoi cell is a regular n-gon of circumradius `ℓ/(2 cos(π/n))`:

      Voronoi / (A/3) = [(n/2) ℓ² sin(2π/n) / (4 cos²(π/n))] / [(n/6) ℓ² sin(2π/n)] = 3 / (4 cos²(π/n)).

  For n = 6 this is 1; for n = 5 it is `4.5 − 1.5√5 = 1.1458980`. The 12
  icosahedral vertices are exactly 5-fold symmetric, and the measured values
  1.1795, 1.1538, 1.1479, 1.1464, 1.14602, 1.14593, 1.145906 approach the
  prediction at order 2.00. With the mixed Voronoi area the valence-5 error
  does converge (1.6e-5 at level 7).
- **Valence-6 vertices on the icosahedral mirror planes do not converge
  either.** The maximum error over valence-6 vertices on the projected base
  edges is 1.9e-3, 2.4e-3, 2.6e-3, 2.6e-3 at levels 4–7, and on the base-face
  medians 2.7e-3, 2.1e-3, 2.0e-3, 2.0e-3, while off the 15 mirror planes it is
  1.1e-3, 5.4e-4, 4.3e-4, 4.7e-4. (An earlier version of this note attributed
  the plateau to valence-6 vertices near valence-5 ones; the worst vertices are
  6–15 edge lengths from any valence-5 vertex and lie on the mirror planes.)
  The mixed Voronoi estimator converges on the same vertices (2.2e-5 at level
  7), so their stars stay irregular under refinement; a kink of the recursive
  midpoint map across base edges is a plausible cause, the medians are not
  explained, and no closed form is derived. The RMS error over all vertices
  still falls because these vertices become a vanishing fraction.
- **Torus grids** (sign-changing K): the angle-defect curvature converges at
  order 1.93; Gauss–Bonnet sums are 0 on tori and 4π on spheres to 4e-11.

### Mesh quality at fixed vertex count (T041)

642-vertex icosphere with tangential Gaussian jitter (standard deviation
`a·h`, three seeds) and 642-vertex latitude–longitude spheres:

| Jitter a | mean min angle | curvature RMS (barycentric) | curvature RMS (Voronoi) | geodesic mean error |
| --- | --- | --- | --- | --- |
| 0 | 54.1° | 0.0204 | 0.0048 | 0.0123 |
| 0.05 | 40.1° | 0.0225 | 0.0048 | 0.0117 |
| 0.10 | 25.3° | 0.0333 | 0.0108 | 0.0132 |
| 0.15 | 9.7° | 0.0607 | 0.0408 | 0.0173 |
| 0.2, 0.3 | folded, refused (`folded_face`) | 1.5–28 (unguarded) | — | — |

- **Within the isotropic tangential-jitter family**, curvature error grows
  monotonically as quality falls (seed-averaged; the three-seed t-interval is
  reported with the finding).
- **Folds.** All six meshes jittered by 0.2 h or more are refused as
  `folded_face`, each has inward-pointing faces (2–50), and none of the ten
  meshes below 0.2 h is refused or has an inverted face; the refusal is checked
  mesh by mesh against that independent indicator.
- **Counterexamples.** A jittered mesh with min angle 40.9° has a smaller
  geodesic error (0.0099) than the regular icosphere (54.1°, 0.0123). Across
  families, the 20×32 latitude–longitude sphere (min angle 10.8°) has a
  smaller **barycentric-area** curvature RMS error (0.0161) than the icosphere
  (0.0204), but this is an artifact of the estimator: the icosphere's
  barycentric RMS is dominated by its 12 valence-5 vertices (T040), and with the
  mixed Voronoi area the icosphere (0.0048) beats every latitude–longitude
  sphere (0.0063–0.0260). Within the latitude–longitude family, 40×16 (min
  angle 11.1°, radius ratio 2.87) has larger errors under both area choices
  (0.0272 and 0.0150) than 10×64 (5.5°, 5.48; 0.0198 and 0.0111), so neither
  quality metric orders errors pairwise inside a family either.
- **Rank correlations** over the 15 valid meshes (average ranks, checked
  against `scipy.stats.spearmanr` when installed): the maximum radius ratio
  ranks the Voronoi curvature RMS (ρ = 0.825) and the geodesic error
  (ρ = 0.789; one-sided permutation p ≤ 0.01 for both), but not the barycentric
  RMS (0.275). Minimum angle: −0.021 with the barycentric RMS, −0.661 with the
  Voronoi RMS, −0.799 with the maximum curvature error (values equal to 12
  significant digits share their average rank). With 15 meshes from two
  families the standard error of ρ is about 0.3; the ranking is an observation,
  not a law.
- On planar meshes, straightest geodesics are exact at every tested quality,
  whereas the edge-graph error follows edge directions (0.082 at shear 0 down to
  0.018 at shear 1.5, although the minimum angle falls from 45° to 11.9°).
- **Schwarz lantern** (m = q n² bands, q = 0.25, R = H = 1). Each face has
  base `2R sin(π/n)` and height `sqrt((H/m)² + R²(1 − cos(π/n))²)`, so

      area = 2nR sin(π/n) sqrt(H² + m²R²(1 − cos(π/n))²),   d_H = R(1 − cos(π/n)).

  The Hausdorff distance is two-sided: every face lies between radius
  `R cos(π/n)` and `R` (the sampled mesh-to-cylinder distance attains the bound
  at horizontal edge midpoints), and every horizontal ray from the axis crosses
  the closed mesh tube at such a radius, so each cylinder point is within the
  same distance of the mesh. The lantern develops to a flat strip of the face
  height sum. As n = 4…64, `d_H` falls from 0.29 to 1.2e-3 while the area ratio
  tends to `sqrt(1 + (π²q/2)²) = 1.5881` (1.5873 at n = 64) and the traced
  bottom-to-top geodesic length tends to the same factor times H (1.5879).
  Interior angle defects are exactly zero (isosceles faces give angle sums
  `2α + 4β = 2π`), so the angle-defect Gaussian curvature matches the
  cylinder's `K = 0`; the failure is extrinsic: face normals converge to a tilt
  of `atan(π²qR/2H) = 50.97°` instead of 0 (50.967° at n = 64), and the total
  absolute mean curvature grows like n² (exponent 2.01; 38.6 to 11441 against
  the smooth π). Hausdorff convergence therefore implies convergence of none of
  area, geodesic distance, normals or mean curvature. The strongly pleated
  lantern with m = n² is refused as `folded_face`.

### Uncertainty on vertices, normals and curvature (T043)

Isotropic Gaussian vertex noise `σ` on icosphere-3 (h = 0.151), 4000 seeded
samples per `σ`, compared with the linearization `Var f = σ² |∇f|²` (central
finite differences, step 1e-6). The markers sit on declared geodesic 5 of six,
**chosen because it has the largest vertex margin on icosphere-3** — the best
case for a fixed face corridor.

| Observable | gain `|∇f|` | MC/linear at σ = 1e-4, 1e-3, 3e-3, 1e-2 |
| --- | --- | --- |
| marker distance along a fixed face corridor | 1.05 | 1.005, 1.002, 0.975, 1.022 |
| vertex normal, valence-5 vertex | 6.48 | 0.992, 0.987, 1.023, 0.977 |
| angle-defect curvature, valence-5 vertex | 261 | 1.021, 1.021, 1.073, **1.633** |
| vertex normal, valence-6 vertex | 5.06 | 0.991, 1.001, 0.989, 1.016 |
| angle-defect curvature, valence-6 vertex | 164 | 0.990, 1.020, 1.041, **1.386** |

No sampled vertex normal flipped sign (smallest `n·n₀` 0.98).

- **Linearization criterion for curvature.** For a cone of ring radius `ℓ`
  and apex height `e`, the defect is `2π(1 − ℓ/sqrt(ℓ² + e²)) ≈ π e²/ℓ²`. On a
  sphere `e₀ ≈ ℓ²/(2R)`, so a normal displacement δ contributes
  `(π/R) δ + (π/ℓ²) δ²`: the quadratic term is negligible only when
  `σ R / h² ≪ 1`. At σ = 1e-2, `σ/h² = 0.44` and the linearization
  underestimates the variance by 39–63 % (counterexample recorded).
- **Scaling** (one declared geodesic at every level 2–5). Fitted sensitivity
  exponents: curvature −2.01 and −1.93, normals −1.00 and −0.96, marker
  distance −0.05. The marker-distance gain splits into the vertices of the two
  marker faces (0.89, 1.01, 1.06, 1.04; slope −0.07) and the interior strip
  vertices (0.35, 0.28, 0.22, 0.15; slope 0.39, pairwise 0.29, 0.36, 0.52). The
  `h⁰` term is the barycentric attachment of the markers; the interior shape
  term decreases, heuristically like `sqrt(L h)/R` (an `O(δh/R)` metric change
  from each of about `2L/h` nearby vertices).
- **Refinement makes curvature worse under fixed noise.** At σ = 1e-3 the total
  RMS curvature error at a valence-6 vertex is 0.048, 0.164, 0.681, 3.64 at
  levels 2–5, although the noise-free discretization error falls from 8.9e-3 to
  2e-4 (counterexample to "refining reduces curvature error").
- **Corridor switching is strip dependent.** Fraction of samples whose unfolded
  segment leaves its unperturbed face corridor, for all six declared strips of
  length 1 on icosphere-3:

  | Strip | vertex margin | σ = 1e-4 | 1e-3 | 3e-3 | 1e-2 |
  | --- | --- | --- | --- | --- | --- |
  | 0 | 0.0027 | 0.001 | 0.37 | 0.47 | 0.60 |
  | 1 | 0.0507 | 0 | 0 | 0 | 0.26 |
  | 2 | 0.0369 | 0 | 0 | 0.08 | 0.48 |
  | 3 | 0.0102 | 0 | 0 | 0.01 | 0.29 |
  | 4 | 0.0104 | 0 | 0.03 | 0.26 | 0.49 |
  | 5 (declared) | 0.0528 | 0 | 0 | 0.03 | 0.41 |

  The threshold scales roughly with the smallest segment-to-vertex distance
  (edge margin × h), but strips with equal margins (3 and 4) differ because
  other near-vertex crossings matter. Past it the perturbed geodesic can switch
  corridors, its distance becomes a minimum over corridors, and the
  fixed-corridor variance no longer describes the geodesic distance. Re-tracing
  per sample is not done.

### Geometry versus sensor uncertainty (T044)

Residual `r = y − d(V_nominal)` with `y = d(V_nominal + η) + ε`,
`η ~ N(0, σ_g² I)` per vertex coordinate and independent `ε ~ N(0, σ_s²)`:
`Var r = σ_s² + Var_η d ≈ σ_s² + σ_g² |∇d|²` (law of total variance), with
`|∇d| = 1.05` for the declared marker pair (nominal distance 1.0).

- The nested design (1000 geometry samples × 16 sensor readings) decomposes the
  sums of squares exactly (ANOVA residual 3e-16). Within-group variance matches
  `σ_s²` and the corrected between-group variance matches the geometry part to
  within 4 standard errors in all six scenarios; fresh Monte Carlo totals
  (20000 samples) match `σ_s² + σ_g²|∇d|²` with ratios 0.986–1.007.
- **Where the geometry variance comes from.** `|∇d|² = 1.106` splits exactly
  into a tangential part `1.004²` (91 %) and a normal part `0.312²` (9 %), and
  by vertex group into the two marker faces `1.012²` and the interior strip
  `0.284²`. On a smooth surface tangential vertex noise only re-parameterises
  the mesh to first order; here it matters because it drags the barycentric
  markers.
- **Which dominates depends on the noise model as well as the sigmas.** Under
  isotropic noise the crossover is `σ_s* = |∇d| σ_g`, and at the baseline
  σ_g = 1e-3, σ_s = 5e-4 geometry carries 82 % of the variance. Under
  normal-only (shape) noise of the same σ_g the gain is `|∇_n d| = 0.312`, the
  crossover is `σ_s* = 3.1e-4`, and the sensor dominates (geometry share 0.28;
  Monte Carlo geometry variance / linear 0.993).
- **Averaging does not remove geometry.** With σ_g = 1e-3 and σ_s = 2e-3,
  averaging K = 1, 4, 16, 64 sensor readings on the same surface moves the
  geometry share from 22 % to 95 %; the variance approaches the floor
  `σ_g² |∇d|² = 1.1e-6`, not zero.
- When `σ_s²/N_s ≫ Var_η d` the between-group estimate is dominated by sensor
  noise (it is −0.33 of the geometry variance at σ_g = 1e-4, σ_s = 2e-3): the
  design cannot resolve the geometry part there.
- The linearization holds for the declared strip at these σ_g (no sample left
  its corridor); T043 shows it would not for the smallest-margin strip at
  σ_g = 1e-3.

## What these results do not establish

- Accuracy of any real scanned surface, scanner, tracker, tape or calibration
  (recorded as `not_established`); the Gaussian noise models are declarations.
- Convergence orders for meshes reconstructed from data (irregular valences,
  noise, holes); the fitted lateral order is an empirical slope with irregular
  local orders.
- Pointwise convergence of the barycentric angle-defect curvature anywhere on
  the icosphere family (valence-5 vertices and mirror-plane valence-6 vertices
  are counterexamples; the off-mirror maximum stalls at level 7).
- That traced geodesics are globally shortest (the Steiner sandwich is
  consistent with it only).
- Detection of self-intersections, unwelded seams, duplicate faces or other
  defects outside the catalogue; freedom from false `folded_face` refusals on
  sharp creases.
- That minimum angle or radius ratio predicts error on meshes outside the 15
  tested ones, or any production acceptance threshold on them.
- Which of geometry or sensor noise dominates for a real part: that depends on
  the noise model (isotropic versus normal-only) as well as on the sigmas.

## Open questions and next tasks

1. Implement an exact polyhedral distance (MMP or ICH) as an independent
   reference for the Steiner and heat-method distances.
2. Implement the Polthier–Schmies vertex rule and measure how often generic
   traces need it on irregular meshes.
3. Re-trace per Monte Carlo sample to quantify geodesic-distance uncertainty
   past the strip-dependent corridor-switching threshold.
4. Derive the valence-6 plateau on the icosahedral mirror planes (star shape of
   the recursive midpoint subdivision in the limit).
5. Replace isotropic independent vertex noise by correlated, anisotropic scanner
   models once acquired scan data exist (physical, currently blocked).
6. Carry the geometry/sensor split into the typed observation modes of T045.

## Reproduce

```sh
python -m ciw lab run T038 T039 T040 T041 T042 T043 T044 --output-dir results/lab-mesh
python -m ciw lab report T040 --retained results/lab-mesh
python -m pytest -q tests/test_lab_surfaces_discrete_mesh.py
```
