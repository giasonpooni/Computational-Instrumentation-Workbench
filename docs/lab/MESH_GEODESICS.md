# Triangle-mesh geodesics and geometry uncertainty (T038–T044)

This note documents the triangle-mesh experiments of section 3 of the
computational-experimentalist queue: the algorithms, the convergence results,
the named refusal states and, above all, what the results do not establish.

**Scope.** Generated meshes only (icosphere, latitude–longitude sphere, prism
cylinder, Schwarz lantern, planar grids, an L-shaped grid, torus grids, a
refined cube) in normalized units, with smooth references from
`ciw.lab.surfaces` and `ciw.lab.jacobi`.

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
| `src/ciw/lab/surfaces_discrete_mesh_exact.py` | exact polyhedral distances (window propagation), surface points inserted as vertices |
| `src/ciw/lab/surfaces_discrete_mesh_studies.py` | deterministic studies (plain numbers) |
| `src/ciw/lab/surfaces_discrete_mesh.py` | task registrations T038–T044, findings and report fields |
| `tests/test_lab_surfaces_discrete_mesh.py` | regression tests (about 40 s) |

## Algorithms

### Straightest geodesics (T038)

T038 delivers an initial-value tracer (straightest geodesics from a point and
a heading), approximate distances (edge graph, Steiner graph, heat method) and
the exact polyhedral distance between two surface points (below), against
which the others are compared. The exact solver returns distances, not the
shortest path polyline.

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

### Exact polyhedral distance (T038)

`ExactGeodesic(mesh).distances(source)` returns the length of the globally
shortest surface path from a vertex to every vertex, exact up to rounding. It
propagates *windows* in the manner of Chen and Han (1990) as improved by Xin
and Wang (2009, "ICH"). A window is an interval `[b0, b1]` of an edge
together with the position of its source unfolded into the plane of the next
face (below the edge) and the source's own distance `σ`; a point `x` of the
interval is reached at distance `σ + |s − x|`.

- **Order.** Windows are processed in increasing order of the smallest
  distance they carry, so vertex distances settle as in Dijkstra's algorithm,
  and a two-point query (`targets=`) stops once no queued window can carry
  less than the target's distance.
- **Propagation.** A window crossing a face sends its rays to the two far
  edges, split at the apex when the apex is in view (the ray through it
  crosses the interval, with a `1e-12` relative slack); a ray through the apex
  gives the apex its distance.
- **Pseudo-sources.** Shortest paths bend only at saddle vertices (angle sum
  above `2π`) and at reflex boundary vertices (angle sum above `π`). Such a
  vertex, once reached, starts windows of its own on the far edges of its
  faces, with `σ` its distance. Every vertex whose angle sum is not below
  `2π` (`π` on the boundary) by more than `1e-9` is made one, flat vertices
  included: a slight saddle taken for flat would leave the wedge behind it,
  as wide as its angle excess, to no window (on a 4 × 4 grid with `1e-6`
  height noise, 5 of its 25 vertices went unreached that way), while a flat
  vertex taken for a pseudo-source only adds windows. Convex vertices are
  passed on both sides.
- **Pruning.** Along an edge `PQ`, `σ + |s − x| − |P x|` never increases away
  from `P` and `σ + |s − x| − |Q x|` never decreases towards `Q`, so each
  endpoint cuts one end of the interval at a single root of a linear
  equation. The apex of either face drops a window only when it beats both
  ends by a margin `c ≥ 0`, because the points it beats by more than `c ≥ 0`
  form a convex set. A point is cut only when a path through a vertex beats
  it by more than `1e-10` of the mean edge. That is sound: a subpath of a
  shortest path is shortest, so a point on a shortest path is never beaten by
  a vertex path, and the window carrying it survives.
- **Points on the surface.** `insert_points(mesh, [(face, xyz)])` makes each
  point a vertex: a point inside a face splits it 1-to-3, a point within
  `1e-9` (barycentric) of an edge splits both faces at the edge 1-to-2, and a
  point at a vertex is that vertex. The point is first projected onto the
  face plane and, near an edge, moved onto the edge (by about `1e-9` of an
  edge length at most), so the new faces lie in the old face planes and the
  polyhedral metric, and every distance, is unchanged; the refined mesh stays
  closed and consistently oriented.

The loop is plain Python. Icosphere-4 (2562 vertices) propagates about 141000
windows per source; the studies stop there. The solver returns distances only:
back-tracing the path polyline through the windows is not implemented.

**Independent implementations.** When installed (the `lab` extra pins both),
two C++ libraries check the solver: `pygeodesic` 0.1.11, a wrapper of
Kirsanov's implementation of the exact MMP algorithm (Surazhsky et al. 2005),
compared at every vertex from the same sources; and `potpourri3d` 1.4.0, whose
`EdgeFlipGeodesicSolver` (geometry-central's FlipOut, Sharp and Crane 2020)
shortens the edge-graph path between two vertices to a *locally* shortest
geodesic, so its length is at least the exact distance and equals it when that
geodesic is also globally shortest. Their origins (`pygeodesic`,
`potpourri3d`) are recognised independent families, and the ciw side of each
check names `ciw <version>` and the solver module's source digest. Without a
package its finding keeps the same claim, value shape and prose and rests on
same-origin checks (source symmetry and the edge Lipschitz bound; for FlipOut,
the edge-graph paths it starts from), so it is `numerically_verified` instead
of `independently_verified`. The run identity records both packages (with
scipy, sympy and mpmath), so `ciw lab verify` names them ("optional modules
differ") when such a label changes between environments.

### Distances

- **Edge Dijkstra** (heap based): shortest edge path, an upper bound on the
  polyhedral distance. Compared with a dense Floyd–Warshall (ciw code, a
  same-origin `cross_implementation` check) and, when scipy is installed, with
  `scipy.sparse.csgraph.dijkstra` (independent). Without scipy the finding is
  `numerically_verified` instead of `independently_verified`; its claim and the
  report prose are the same in both environments.
- **Steiner graph**: `k` equally spaced points per edge, all node pairs inside
  each face joined. Every graph edge is a straight segment in a face, so graph
  distances are upper bounds on the polyhedral distance; with
  `k = 2^j − 1` the node sets are nested and distances cannot increase with
  `k`. That every edge lies in a face is checked by a point-in-face test
  (plane offset and barycentric coordinates) that does not use the graph
  construction; the test fails on an edge joining two faces that share no
  edge. (An earlier check that graph distances never fall below the chord was
  dropped: it holds for any graph with Euclidean edge weights.)
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
- Discrete Gauss–Bonnet `Σ(2π − Σθ) = 2πχ` is an identity for every closed
  mesh whose triangle angles sum to π, whatever vertex the angles are assigned
  to. Its residual (about `4e-11`) is reported only as an implementation
  sanity check, not as a finding.
- Vertex normals: area-weighted sums of the one-ring face normals.
- Per-vertex uncertainty: `vertex_uncertainty(mesh, covariance)` returns the
  linearized standard deviations of every vertex normal (RMS tilt,
  `sqrt(trace Cov n)`) and every angle-defect curvature under a declared
  covariance of independent vertex errors: one variance, one variance per
  vertex, or a 3×3 matrix per vertex. The Jacobians are one-ring central
  differences. Boundary vertices get no curvature, and invalid meshes or
  covariances are refused. `batch_vertex_fields` evaluates normals and
  curvature of whole noisy meshes for the Monte Carlo comparison.

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
| `inverted_face` | validation | face oriented towards a declared centre, `det(p₀ − c, p₁ − c, p₂ − c) ≤ 0` (only when `center=c` is passed) |
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

T042 constructs 25 defect cases (22 distinct codes). All are refused with the
expected code, five valid controls report no issue (two of them validated
with a declared centre), and a mesh with three defects reports all three in
the declared order. Two counterexamples are recorded alongside:

- Evaluating curvature and normals on an unvalidated zero-area face produces
  nonfinite values, so the refusals are necessary, not cosmetic.
- Without a declared centre, the validator accepts a jittered icosphere-3
  (seed 20261940, 0.2 h) with an inverted face.

**Folds and inversions.** The dihedral fold check (`folded_face`) was added
after T041's declared seeds showed tangential jitter of `0.2 h` folding faces.
It bounds the bend between neighbours, so it is not an inversion test:

- An inverted sliver whose bend to every neighbour stays below about 154°
  passes it. Over 60 further seeds per amplitude, 7 meshes at 0.2 h and 1 at
  0.15 h have such a face (T041).
- A crease with no inverted face can fail it. The `q = 1` lantern is refused
  although its smallest face-normal radial component is +0.20.

For a mesh that is star-shaped about a known point, `inspect(..., center=c)`
and `TriMesh.build(..., center=c)` also refuse faces oriented towards `c` as
`inverted_face`. T041 validates its spheres this way. The test assumes
star-shapedness: it would refuse a valid torus about its own centre, so it
runs only on request.

The hypothesis is scoped to the declared catalogue. **Not detected:**

- self-intersections between non-adjacent faces;
- unwelded seams (coincident duplicate vertices pass validation and surface
  only as `boundary_reached` during tracing);
- duplicate faces;
- near-degenerate slivers just above the `1e-12` threshold;
- inverted faces with bends below the fold threshold when no centre is
  declared.

**False positives:** a legitimate sharp crease with a bend over about 154° is
refused as `folded_face`. The embedded `m = n²` lantern of T041 is refused
although it does not overlap itself and no face points inward.

## Results

### Solver checks (T038)

| Check | Result |
| --- | --- |
| Sheared planar meshes (min angle down to 11.9°): trace vs straight line | `8.9e-16` |
| Prism cylinder, n = 8…128: trace vs exact development of the mesh | `8.0e-15` |
| 42 sphere traces (levels 1–7): strip layout length vs traced length | `3.1e-15` |
| Edge Dijkstra vs Floyd–Warshall and scipy | `8.9e-16` (`independently_verified` with scipy) |
| Nested Steiner distances, k = 0, 1, 3, 7 | never increase; 0 of 115740 graph edges leave a face |
| Steiner distance − traced length (level 2, length 1) | mean 3.0e-2 (k=1), 1.3e-2 (k=3), 5.3e-3 (k=7), all positive |

Exact polyhedral distances:

| Check | Result |
| --- | --- |
| Sheared planar meshes, from a vertex and an inserted interior point: exact vs Euclidean | `2.7e-15` |
| L-shaped grid (reflex corner at (0.5, 0.5)), two sources: exact vs segment or path bent at the corner | `1.1e-15`; 20 targets reached around the corner |
| Prism cylinder, n = 8, 16, 32, from a boundary vertex and an inserted point: exact vs development over periodic images | `7.1e-15` |
| Refined cube (4 × 4 squares per face), corner to corners: exact vs 1, √2, √5 | `8.9e-16`; the edge graph gives 1 + √2 = 2.414214 for √5 = 2.236068 |
| Face point and edge point inserted as vertices (icosphere-2, torus 12 × 6): change of distances between original vertices | `1.3e-15` |
| Three sources on icosphere levels 1–4 and a 24 × 12 torus (120 saddles and 48 flat vertices as pseudo-sources): source symmetry and pygeodesic (11088 distances) | `8.0e-15` (`independently_verified` with pygeodesic) |
| FlipOut geodesic − exact over 3267 vertex pairs (icosphere-2, icosphere-3, torus) | smallest −4.4e-15; shortest in 404 of 483, 1475 of 1923, 795 of 861 pairs; largest excess 7.5e-3, 4.9e-3, 0.14 |

Compared with the exact distance from vertex 0 (valence 5):

| Method | Level 1 | Level 2 | Level 3 | Level 4 |
| --- | --- | --- | --- | --- |
| Edge graph, largest relative excess | 0.181 | 0.220 | 0.232 | 0.235 |
| Steiner k = 1, 3, 7, mean excess | — | 1.9e-2, 7.1e-3, 2.4e-3 | 2.1e-2, 7.8e-3, 2.3e-3 | — |
| Heat method (t = h²), largest error | 0.096 | 0.066 | 0.043 | — |
| Exact − great-circle distance, largest | 0.115 | 0.031 | 0.0079 | — |

- No graph path, FlipOut geodesic or traced geodesic is shorter than the
  exact distance (the smallest excess is rounding, down to −4.4e-15). The edge
  graph's relative excess approaches the valence-5 floor `√5 − 2 = 0.236` of
  T039, now measured against the polyhedral distance instead of the sphere.
- The heat method's error against the exact polyhedral distance falls at
  every level, but at a fitted order of 0.60, below its order 1.01 against the
  great circle (T039): on coarse meshes part of its error against the sphere
  cancels the polyhedral metric's own `O(h²)` shortfall (last row).
- **Straightest geodesics are often not shortest paths.** Of the six declared
  length-2 traces (below π), 2, 3, 3 and 2 are shortest paths on levels 1–4
  (within `1e-10` of the exact distance); the others are longer by up to
  1.3e-2, 1.6e-3, 1.4e-4 and 3.7e-5, and never by less than 4.6e-6. On level 2
  one of the six length-1 traces of the Steiner comparison is longer than the
  exact distance by 1.1e-4, so the Steiner-over-traced ordering above is not a
  sandwich of the distance: for large enough k the Steiner distance would fall
  below that traced length. This is a counterexample to "a straightest
  geodesic shorter than π on a mesh inscribed in the sphere is a shortest
  path". It is consistent with the structure of the cut locus of a point on a
  convex polyhedron, a tree with a leaf at every vertex: a straightest
  geodesic passing a vertex closely crosses the branch that ends there, after
  which the path around the vertex's other side is shorter. The excess falls
  with refinement because the curvature concentrated at each vertex does.
  Where a trace crosses the cut locus is observed per trace, not predicted.

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
  order 1.93. (Gauss–Bonnet sums are 0 on tori and 4π on spheres to 4e-11, an
  identity that is reported as a sanity check only.)

### Mesh quality at fixed vertex count (T041)

642-vertex icosphere with tangential Gaussian jitter (standard deviation
`a·h`, three seeds, which are the same three noise fields scaled to each
amplitude) and 642-vertex latitude–longitude spheres. Meshes are validated
with the sphere centre declared, so the valid set excludes inverted faces:

| Jitter a | mean min angle | curvature RMS (barycentric) | curvature RMS (Voronoi) | geodesic mean error |
| --- | --- | --- | --- | --- |
| 0 | 54.1° | 0.0204 | 0.0048 | 0.0123 |
| 0.05 | 40.1° | 0.0225 | 0.0048 | 0.0117 |
| 0.10 | 25.3° | 0.0333 | 0.0108 | 0.0132 |
| 0.15 | 9.7° | 0.0607 | 0.0408 | 0.0173 |
| 0.2, 0.3 (declared seeds) | refused (`folded_face`) | 1.5–28 (unguarded) | — | — |

- **Within the isotropic tangential-jitter family**, curvature error grows
  monotonically as quality falls (seed-averaged; the three-seed t-interval is
  reported with the finding).
- **Folds, declared seeds only.** For the three declared seeds the dihedral
  fold refusals coincide mesh by mesh with inverted faces (face normal
  pointing towards the sphere centre). All six meshes at 0.2 h and 0.3 h are
  inverted and refused; none of the ten below is. This holds for these seeds,
  not in general.
- **Folds over 60 further seeds per amplitude** (seeds 20261939–20261998):

  | Jitter a | inverted, accepted by the dihedral check | inverted, refused | no inverted face |
  | --- | --- | --- | --- |
  | 0.1 | 0 | 0 | 60 |
  | 0.15 | 1 | 1 | 58 |
  | 0.2 | 7 | 32 | 21 |
  | 0.3 | 0 | 60 | 0 |

  So 0.2 h does not always invert a face (21 of 60 meshes have none), and the
  dihedral check misses inverted slivers. The recorded witness is the clearest
  miss, seed 20261944 at 0.2 h: face 854 has normal·radial −0.34 and corner
  angles 3.7°, 4.3° and 172.0°, while the smallest adjacent-normal dot in the
  mesh is −0.44 (a bend of about 116°, far below the 154° threshold).
  Unguarded, its curvature RMS is 0.43. No mesh
  without an inverted face was refused (0 of 240). With the centre declared,
  every inverted mesh is refused as `inverted_face`. Conversely, the `q = 1`
  lantern is refused as folded with no inward face. The dihedral check is
  therefore neither necessary nor sufficient for an inversion.
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
- **Rank correlations** over the 15 pooled valid meshes (average ranks,
  checked against `scipy.stats.spearmanr` when installed). The maximum radius
  ratio is positively rank-correlated with the Voronoi curvature RMS
  (ρ = 0.825; 18 of 105 pairs discordant) and with the geodesic error
  (ρ = 0.789; 20 of 105 discordant). It does not rank either error: the
  association is carried by the jitter family (ρ = 0.915 and 0.855 within it).
  Within the five latitude–longitude spheres it vanishes or reverses
  (ρ = 0.0 and −0.3). For example, 40×16 has ratio 2.87 and Voronoi RMS 0.0150,
  while 10×64 has ratio 5.48 and Voronoi RMS 0.0111. There is no association
  with the barycentric RMS (0.275). Minimum angle gives −0.021 with the
  barycentric RMS, −0.661 with the Voronoi RMS and −0.799 with the maximum
  curvature error. Values equal to 12 significant digits share their average
  rank. The one-sided permutation p ≤ 0.01 treats the pooled meshes as
  exchangeable, which the replicated noise fields and the two families are
  not. With 15 meshes the standard error of ρ is about 0.3, so this is an
  observation, not a law.
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
  lantern with m = n² is refused as `folded_face`, although no face normal
  points towards the axis (smallest radial component +0.20).

### Uncertainty on vertices, normals and curvature (T043)

Isotropic Gaussian vertex noise `σ` on icosphere-3 (h = 0.151), 4000 seeded
samples per `σ`, compared with the linearization `Var f = σ² |∇f|²` (central
finite differences, step 1e-6). The markers sit on declared geodesic 5 of six,
chosen because it has the largest vertex margin on icosphere-3. It stays in
its corridor for `σ ≤ 1e-3`, but a larger margin does not make it the most
robust strip at larger noise (see the corridor table).

| Observable | gain `|∇f|` | MC/linear at σ = 1e-4, 1e-3, 3e-3, 1e-2 |
| --- | --- | --- |
| marker distance along a fixed face corridor | 1.05 | 1.005, 1.002, 0.975, 1.022 |
| vertex normal, valence-5 vertex | 6.48 | 0.992, 0.987, 1.023, 0.977 |
| angle-defect curvature, valence-5 vertex | 261 | 1.021, 1.021, 1.073, **1.633** |
| vertex normal, valence-6 vertex | 5.06 | 0.991, 1.001, 0.989, 1.016 |
| angle-defect curvature, valence-6 vertex | 164 | 0.990, 1.020, 1.041, **1.386** |

No sampled vertex normal flipped sign (smallest `n·n₀` 0.98).

- **Per-vertex field.** The table above covers two sample vertices. The
  per-vertex function `vertex_uncertainty` gives the linearized normal and
  curvature standard deviation at every vertex under a declared vertex
  covariance. On icosphere-3 it is compared with whole-mesh Monte Carlo (2000
  samples) at all 642 vertices, for two covariances:
  - isotropic, σ = 1e-4: curvature SD 0.0164–0.0261, normal SD
    5.1e-4–6.5e-4 rad; max |z| over vertices 2.99 (curvature) and 2.35
    (normal);
  - normal-dominant, σ_n = 3e-4 and σ_t = 1e-4: max |z| 3.11 and 2.37.

  At vertex 0 and at the far valence-6 vertex it agrees with the two-vertex
  propagation to 1e-12. Normals are always derived from the noisy vertices;
  an input uncertainty model for independently measured normals is deferred.

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
  (edge margin × h), but the margin does not order the strips. Strips with
  equal margins (3 and 4) differ, and strip 1 (margin 0.0507) leaves its
  corridor less often than the declared strip 5 at σ = 3e-3 (0 against 0.028)
  and at 1e-2 (0.26 against 0.41). Other near-vertex crossings matter. This is
  recorded as a counterexample to "the largest-margin strip is the most
  robust".

  The declared strip's leave fraction is estimated twice, by two independent
  4000-sample studies: the propagation study (seed 20260938: 0.031 at
  σ = 3e-3, 0.395 at 1e-2) and this six-strip corridor study (seed 20260968:
  0.028 and 0.411). At σ = 1e-2 they differ by 0.0155. That is more than one
  estimate's 95 % half-width (0.015) but within the 95 % interval of the
  difference of two independent fractions (1.96 √(p₁q₁/n + p₂q₂/n) ≈ 0.0215).
  A check on the corridor finding requires every difference to stay within
  that interval, and the report prints both estimates with their seeds. The
  report's sentence is decided by the same test as that check: it says the
  estimates agree only when the largest z is at most 1.96, and otherwise that
  they differ beyond the interval, with the largest z.
  Comparisons between strips use the corridor study only. Past the threshold the perturbed geodesic can switch
  corridors, its distance becomes a minimum over corridors, and the
  fixed-corridor variance no longer describes the geodesic distance. Re-tracing
  per sample is not done.

### Geometry versus sensor uncertainty (T044)

Residual `r = y − d(V_nominal)` with `y = d(V_nominal + η) + ε`,
`η ~ N(0, σ_g² I)` per vertex coordinate and independent `ε ~ N(0, σ_s²)`:
`Var r = σ_s² + Var_η d ≈ σ_s² + σ_g² |∇d|²` (law of total variance), with
`|∇d| = 1.05` for the declared marker pair (nominal distance 1.0).

- Nested design: 1000 geometry samples × 16 sensor readings. Within-group
  variance matches `σ_s²`, and the corrected between-group variance matches
  the linearized geometry part, to within 4 standard errors in all six
  scenarios. Fresh Monte Carlo totals (20000 samples) match
  `σ_s² + σ_g²|∇d|²` with ratios 0.986–1.007. The ANOVA sum-of-squares
  identity (residual 3e-16) holds for any balanced data, so it is reported as a
  bookkeeping sanity value, not as evidence.
- **Where the geometry variance comes from.** `|∇d|² = 1.106` splits (by
  orthogonal projection, an identity not counted as evidence) into a
  tangential part `1.004²` (91 %) and a normal part `0.312²` (9 %), and
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
  its corridor). T043 shows it would not for the smallest-margin strip 0,
  which leaves its corridor in 37.1 % of samples at σ_g = 1e-3. That fraction
  is read from T043's corridor study, not hard-coded.

## What these results do not establish

- Accuracy of any real scanned surface, scanner, tracker, tape or calibration
  (recorded as `not_established`); the Gaussian noise models are declarations.
- Convergence orders for meshes reconstructed from data (irregular valences,
  noise, holes); the fitted lateral order is an empirical slope with irregular
  local orders.
- Pointwise convergence of the barycentric angle-defect curvature anywhere on
  the icosphere family (valence-5 vertices and mirror-plane valence-6 vertices
  are counterexamples; the off-mirror maximum stalls at level 7).
- That traced geodesics are globally shortest: many are not (T038), and where
  one stops being shortest is observed, not predicted. The exact solver gives
  distances only, not the shortest path itself, and is exact only up to
  rounding and the `1e-10` pruning margin; its speed limits it to meshes of a
  few thousand vertices.
- Detection of self-intersections, unwelded seams, duplicate faces or other
  defects outside the catalogue. Detection of inverted faces with small bends
  when no centre is declared. Freedom from false `folded_face` refusals on
  sharp creases.
- That jitter of a given amplitude always, or never, inverts a face. The fold
  rates are counts over 60 seeds of one generator.
- That minimum angle or radius ratio predicts error on meshes outside the 15
  tested ones (the radius-ratio association vanishes within the
  latitude–longitude family), or any production acceptance threshold on them.
- Uncertainty of independently measured normals (normals here are always
  derived from noisy vertices).
- Which of geometry or sensor noise dominates for a real part: that depends on
  the noise model (isotropic versus normal-only) as well as on the sigmas.

## Open questions and next tasks

1. Back-trace the shortest path polyline from the exact solver's windows, and
   locate where straightest geodesics stop being shortest (their crossing of
   the start point's cut locus) as a function of refinement and of the
   distance to the nearest vertex.
2. Implement the Polthier–Schmies vertex rule and measure how often generic
   traces need it on irregular meshes.
3. Re-trace per Monte Carlo sample to quantify geodesic-distance uncertainty
   past the strip-dependent corridor-switching threshold.
4. Derive the valence-6 plateau on the icosahedral mirror planes (star shape of
   the recursive midpoint subdivision in the limit).
5. Replace isotropic independent vertex noise by correlated, anisotropic scanner
   models once acquired scan data exist (physical, currently blocked). The
   scan export would enter as an operator capture read with
   `ctx.capture("scan-export")` (bound by `ciw lab run T043 --capture
   scan-export=PATH`), from which T043 could fit the covariance as a
   computational finding. A scanner claim also needs an acquisition record
   (device, `raw_sha256` of the captured bytes, time, calibration) and either
   a probe of the scanner on the analysing host that succeeds in T043 or a
   signed-capture trust anchor, because the physical gate never accepts an
   unauthenticated capture by itself (`runner.CAPTURE_INSTRUMENTS` has no
   entry for `scan-export`). The run would be retained with
   `ciw lab hardware retain` under `lab/hardware/<run-id>`. Neither the
   capture reader nor a scanner probe exists, so the scanner claims stay
   `not_established` even when such data exist.
6. An inversion test for closed meshes that are not star-shaped (for example,
   a winding-number or orientation test against a declared outward field).
7. Fill `reconstructed_surface_distance`'s `geometry_m2` from a mesh, as
   σ_vertex² |∇d|² with T043's linearized vertex-noise gain (valid only while
   the marker segment stays in its face corridor), instead of declared
   analytic surface parameters, and check it against T044's nested Monte
   Carlo. T045 delivers the split in part: it carries separate geometry and
   sensor variance components for model-derived distances on parametric
   surfaces (plane, sphere, cylinder geodesic) and names the same mesh
   conversion as its next step. T140's instrument and geometry budget
   delivers it for manufacturing predictions only. An
   `intrinsic_geodesic_distance` reading keeps one sensor sigma: as in T044's
   residual, its geometry part belongs to the model prediction it is compared
   with, not to the reading.
8. Heat-method rates for time steps `t = m h²` other than `m = 1`, and
   endpoint-error orders over the direction's angle to the lattice rows.
9. Per-region error attribution against local triangle quality, and the
   radius-ratio association within each mesh family separately.
10. Named refusal states for the defects T042 does not detect: self-intersections
    between non-adjacent faces, unwelded seams and duplicate faces.

Each task's `recommended_next_task` is one of these questions
(`surfaces_discrete_mesh.NEXT_STEPS`), never a queue task that has already
run.

## Reproduce

```sh
python -m ciw lab run T038 T039 T040 T041 T042 T043 T044 --output-dir results/lab-mesh
python -m ciw lab report T040 --retained results/lab-mesh
python -m pytest -q tests/test_lab_surfaces_discrete_mesh.py
```
