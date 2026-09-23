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
`ciw.lab.evidence.finding`.

| File | Contents |
| --- | --- |
| `src/ciw/lab/surfaces_discrete_mesh_geometry.py` | mesh type, generators, validator, tracer, distances, curvature, strip unfolding |
| `src/ciw/lab/surfaces_discrete_mesh_studies.py` | deterministic studies (plain numbers) |
| `src/ciw/lab/surfaces_discrete_mesh.py` | task registrations T038–T044, findings and report fields |
| `tests/test_lab_surfaces_discrete_mesh.py` | regression tests (under 20 s) |

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
  polyhedral distance. Checked against `scipy.sparse.csgraph.dijkstra`
  (independent implementation, when installed) and a dense Floyd–Warshall.
- **Steiner graph**: `k` equally spaced points per edge, all node pairs inside
  each face joined. Every graph edge is a straight segment in a face, so graph
  distances are upper bounds; with `k = 2^j − 1` the node sets are nested and
  distances cannot increase with `k`.
- **Heat method** (Crane, Weischedel and Wardetzky 2013) with `t = h²`, lumped
  mass and cotangent Laplacian. The heat solution decays like `exp(−d²/4t)`, and
  its far-field gradient direction needs relative accuracy, so both systems are
  solved directly by a banded Cholesky factorization in reverse Cuthill–McKee
  order written with elementwise NumPy only. (A Krylov solver was tried and
  rejected: its iterates vanish beyond the rings it has reached, leaving zero
  gradients far from the source. Threaded LAPACK was also avoided because its
  run time varied by two orders of magnitude on a contended machine.)
- **Strip unfolding**: a face sequence is laid out in the plane from its edge
  lengths; the planar distance between barycentric markers is the length of the
  straight path through the strip when the segment stays inside it (reported
  as a margin). It is independent of the tracer's 3D rotation, which makes it a
  consistency check, and it is the smooth observable used for uncertainty.

### Curvature and normals

- Angle defect `K_v = (2π − Σθ)/(A_v/3)` with the barycentric area `A_v/3`;
  also with the mixed Voronoi area of Meyer, Desbrun, Schröder and Barr.
  Boundary vertices are refused (`boundary_vertex_curvature`).
- Discrete Gauss–Bonnet `Σ(2π − Σθ) = 2πχ` is exact for closed meshes and is
  checked (residual about `1e-11`).
- Vertex normals: area-weighted sums of face normals.

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

T042 constructs 21 defect cases (19 distinct codes); all are refused with the
expected code, four valid controls report no issue, and a mesh with three
defects reports all three in the declared order. The counterexample recorded
alongside: evaluating curvature and normals on an unvalidated zero-area face
produces nonfinite values, so the refusals are necessary, not cosmetic. The
fold check was added after T041 showed that tangential jitter of `0.2 h` folds
faces while every structural check still passes. Self-intersections between
non-adjacent faces are **not** detected.

## Results

### Solver checks (T038)

| Check | Result |
| --- | --- |
| Sheared planar meshes (min angle down to 11.9°): trace vs straight line | `8.9e-16` |
| Prism cylinder, n = 8…128: trace vs exact development of the mesh | `8.0e-15` |
| 36 sphere traces: unfolded strip length vs traced length | `3.1e-15` |
| Edge Dijkstra vs scipy and Floyd–Warshall | `8.9e-16` (`independently_verified` with scipy) |
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

- **Length defect: order 2.01.** The inscribed polyhedron's intrinsic metric
  differs from the sphere's by `O(h²)`.
- **Endpoint (lateral) error: fitted order 1.60** (cross-track 1.52), not a
  clean power law. Curvature on a mesh is a set of point masses at vertices; a
  straightest geodesic is deflected, relative to the smooth one, by the
  imbalance of vertex curvature on its two sides (a discrepancy sum along the
  path). On the icosphere the lattice rows project to great circles, so paths
  nearly parallel to a row accumulate this imbalance coherently (`O(h)`),
  generic paths less so. The order is an empirical slope for six directions.
- **Prism cylinder: order 2.02.** The mesh develops to a flat strip of
  circumference `2nR sin(π/n)`, so a helix of heading α and length L has
  endpoint error `L cos α (π/(n sin(π/n)) − 1) ≈ L cos α π²/(6n²)`; the measured
  `n² × error` at n = 128 is 0.994 of this constant.
- **Distances from vertex 0** (levels 1–4): the edge-graph maximum relative
  error is 0.145, 0.211, 0.230, 0.234 — a floor, not convergence
  (counterexample to "edge Dijkstra converges"). Steiner graphs with fixed `k`
  also keep a floor (k = 3: 0.011, 0.014 at levels 3, 4). The heat method
  converges at order 1.01 (levels 1–3).

### Jacobi fields and curvature (T040)

The finite-difference Jacobi estimate uses paired traces at headings `±δ`:
`j = |X₊(L) − X₋(L)| / (2 sin δ)`, which equals `sin L` exactly on the smooth
unit sphere; the smooth `ciw.lab.jacobi` transfer reproduces `sin L` to 7e-11.

- **δ = 0.1, h → 0: converges** (mean error 2.2e-1 → 3.2e-3, levels 2–6,
  order 1.59).
- **h fixed, δ → 0: the flat answer.** At δ = 1e-5 all 30 pairs cross the same
  face sequence; their common development is a plane, so
  `|X₊ − X₋| = 2L sin δ` and `j = L = 2` exactly (deviation 5e-10), an error of
  `L − sin L = 1.0907` at every level. Curvature lives at vertices, and two
  geodesics feel it only when a vertex lies between them. The two limits do not
  commute: mesh Jacobi fields require `δ L ≫ h`.
- **Valence-5 vertices do not converge to K.** On a polyhedron inscribed in the
  unit sphere the angle defect at a vertex equals the area of the Gauss-image
  polygon of its face normals, and each face normal points at the face
  circumcenter, so in the limit the defect is the Voronoi-cell area. For a
  regular valence-n star of isosceles triangles with apex angle `2π/n` and legs
  `ℓ`, the Voronoi cell is a regular n-gon of circumradius `ℓ/(2 cos(π/n))`:

      Voronoi / (A/3) = [(n/2) ℓ² sin(2π/n) / (4 cos²(π/n))] / [(n/6) ℓ² sin(2π/n)] = 3 / (4 cos²(π/n)).

  For n = 6 this is 1; for n = 5 it is `4.5 − 1.5√5 = 1.1458980`. The 12
  icosahedral vertices are exactly 5-fold symmetric, and the measured values
  1.1795, 1.1538, 1.1479, 1.1464, 1.14602, 1.14593 approach the prediction at
  order 2.00. The maximum error over valence-6 vertices also plateaus near
  2.5e-3 (vertices near the valence-5 ones stay irregular), while the RMS error
  over all vertices falls like `h` because the bad vertices become a vanishing
  fraction. With the mixed Voronoi area the valence-5 error does converge
  (6.6e-5 at level 6).
- **Torus grids** (sign-changing K): the angle-defect curvature converges at
  order 1.93; Gauss–Bonnet sums are 0 on tori and 4π on spheres to 1e-11.

### Mesh quality at fixed vertex count (T041)

642-vertex icosphere with tangential Gaussian jitter (standard deviation
`a·h`, three seeds) and 642-vertex latitude–longitude spheres:

| Jitter a | mean min angle | curvature RMS error | geodesic mean error |
| --- | --- | --- | --- |
| 0 | 54.1° | 0.0204 | 0.0123 |
| 0.05 | 40.1° | 0.0225 | 0.0117 |
| 0.10 | 25.3° | 0.0333 | 0.0132 |
| 0.15 | 9.7° | 0.0607 | 0.0173 |
| 0.2, 0.3 | folded, refused (`folded_face`) | 1.5–28 (unguarded) | — |

- Within the jitter family, curvature error grows monotonically as quality
  falls (seed-averaged).
- **Counterexamples.** A jittered mesh with min angle 40.9° has a smaller
  geodesic error (0.0099) than the regular icosphere (54.1°, 0.0123). The
  20×32 latitude–longitude sphere (min angle 10.8°) has a smaller curvature RMS
  error (0.0161) than the icosphere (0.0204), though a larger maximum error.
  Across all valid meshes, the Spearman correlation of minimum angle with
  curvature RMS error is −0.02 (none), with maximum curvature error −0.79 and
  with geodesic error −0.70. Minimum angle is not a sufficient predictor.
- On planar meshes, straightest geodesics are exact at every tested quality,
  whereas the edge-graph error follows edge directions (0.082 at shear 0 down to
  0.018 at shear 1.5, although the minimum angle falls from 45° to 11.9°).
- **Schwarz lantern** (m = q n² bands, q = 0.25, R = H = 1). Each face has
  base `2R sin(π/n)` and height `sqrt((H/m)² + R²(1 − cos(π/n))²)`, so

      area = 2nR sin(π/n) sqrt(H² + m²R²(1 − cos(π/n))²),   d_H = R(1 − cos(π/n)),

  and the lantern develops to a flat strip of that height. As n = 4…64,
  `d_H` falls from 0.29 to 1.2e-3 while the area ratio tends to
  `sqrt(1 + (π²q/2)²) = 1.5881` (1.5873 at n = 64) and the traced
  bottom-to-top geodesic length tends to the same factor times H (1.5879).
  Interior angle defects are exactly zero (isosceles faces give angle sums
  `2α + 4β = 2π`), so the angle-defect Gaussian curvature matches the
  cylinder's `K = 0`; the failure is extrinsic: face normals tilt by up to
  50.97° (the limit `atan(π²q/2)`) and the total absolute mean curvature grows
  from 38.6 to 11441 (smooth value π). Hausdorff convergence therefore implies
  convergence of none of area, geodesic distance, normals or mean curvature.
  The strongly pleated lantern with m = n² is refused as `folded_face`.

### Uncertainty on vertices, normals and curvature (T043)

Isotropic Gaussian vertex noise `σ` on icosphere-3 (h = 0.151), 4000 seeded
samples per `σ`, compared with the linearization `Var f = σ² |∇f|²` (central
finite differences, step 1e-6):

| Observable | gain `|∇f|` | MC/linear at σ = 1e-4, 1e-3, 3e-3, 1e-2 |
| --- | --- | --- |
| marker distance along a fixed face corridor | 1.05 | 1.005, 1.002, 0.975, 1.022 |
| vertex normal, valence-5 vertex | 6.48 | 0.992, 0.987, 1.023, 0.977 |
| angle-defect curvature, valence-5 vertex | 261 | 1.021, 1.021, 1.073, **1.633** |
| vertex normal, valence-6 vertex | 5.06 | 0.991, 1.001, 0.989, 1.016 |
| angle-defect curvature, valence-6 vertex | 164 | 0.990, 1.020, 1.041, **1.386** |

- **Linearization criterion for curvature.** For a cone of ring radius `ℓ`
  and apex height `e`, the defect is `2π(1 − ℓ/sqrt(ℓ² + e²)) ≈ π e²/ℓ²`. On a
  sphere `e₀ ≈ ℓ²/(2R)`, so a normal displacement δ contributes
  `(π/R) δ + (π/ℓ²) δ²`: the quadratic term is negligible only when
  `σ R / h² ≪ 1`. At σ = 1e-2, `σ/h² = 0.44` and the linearization
  underestimates the variance by 39–63 % (counterexample recorded).
- **Scaling.** Fitted sensitivity exponents: curvature −2.01 and −1.93, normals
  −1.00 and −0.96, marker distance 0.01. Curvature noise grows like
  `σ/h²`, normal noise like `σ/h`, while the marker distance is dominated by the
  barycentric attachment of its two markers and does not grow.
- **Refinement makes curvature worse under fixed noise.** At σ = 1e-3 the total
  RMS curvature error at a valence-6 vertex is 0.048, 0.164, 0.681, 3.64 at
  levels 2–5, although the noise-free discretization error falls from 8.9e-3 to
  2e-4 (counterexample to "refining reduces curvature error").
- **Corridor switching.** The unfolded marker segment leaves its unperturbed
  face corridor in 0 %, 0 %, 3.1 % and 39.5 % of samples at the four σ. Beyond
  σ ≈ 1e-3 the perturbed geodesic can switch corridors, its distance becomes a
  minimum over corridors, and the fixed-corridor variance no longer describes
  the geodesic distance. Re-tracing per sample is not done.

### Geometry versus sensor uncertainty (T044)

Residual `r = y − d(V_nominal)` with `y = d(V_nominal + η) + ε`,
`η ~ N(0, σ_g² I)` per vertex coordinate and independent `ε ~ N(0, σ_s²)`:
`Var r = σ_s² + Var_η d ≈ σ_s² + σ_g² |∇d|²` (law of total variance), with
`|∇d| = 1.05` for the marker pair (nominal distance 1.0).

- The nested design (1000 geometry samples × 16 sensor readings) decomposes the
  sums of squares exactly (ANOVA residual 3e-16). Within-group variance matches
  `σ_s²` and the corrected between-group variance matches the geometry part to
  within 4 standard errors in all six scenarios; fresh Monte Carlo totals
  (20000 samples) match `σ_s² + σ_g²|∇d|²` with ratios 0.986–1.007.
- **Which dominates** depends only on the declared sigmas: the crossover is
  `σ_s* = |∇d| σ_g`. At the baseline σ_g = 1e-3, σ_s = 5e-4, geometry carries
  82 % of the variance.
- **Averaging does not remove geometry.** With σ_g = 1e-3 and σ_s = 2e-3,
  averaging K = 1, 4, 16, 64 sensor readings on the same surface moves the
  geometry share from 22 % to 95 %; the variance approaches the floor
  `σ_g² |∇d|² = 1.1e-6`, not zero.
- When `σ_s²/N_s ≫ Var_η d` the between-group estimate is dominated by sensor
  noise (it is −0.33 of the geometry variance at σ_g = 1e-4, σ_s = 2e-3): the
  design cannot resolve the geometry part there.

## What these results do not establish

- Accuracy of any real scanned surface, scanner, tracker, tape or calibration
  (recorded as `not_established`); the Gaussian noise models are declarations.
- Convergence orders for meshes reconstructed from data (irregular valences,
  noise, holes); the fitted lateral order is an empirical slope.
- That traced geodesics are globally shortest (the Steiner sandwich is
  consistent with it only).
- Detection of self-intersections, or of defects outside the catalogue.
- Any production acceptance threshold on minimum angle or radius ratio.

## Open questions and next tasks

1. Implement an exact polyhedral distance (MMP or ICH) as an independent
   reference for the Steiner and heat-method distances.
2. Implement the Polthier–Schmies vertex rule and measure how often generic
   traces need it on irregular meshes.
3. Re-trace per Monte Carlo sample to quantify geodesic-distance uncertainty
   past the corridor-switching threshold (σ ≳ 1e-3 on icosphere-3).
4. Replace isotropic independent vertex noise by correlated, anisotropic scanner
   models once acquired scan data exist (physical, currently blocked).
5. Carry the geometry/sensor split into the typed observation modes of T045.

## Reproduce

```sh
python -m ciw lab run T038 T039 T040 T041 T042 T043 T044 --output-dir results/lab-mesh
python -m ciw lab report T040 --retained results/lab-mesh
python -m pytest -q tests/test_lab_surfaces_discrete_mesh.py
```
