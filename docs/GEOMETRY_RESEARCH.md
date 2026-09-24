# Executable geometry providers in the workbench

The three former metadata scaffolds now own bounded numerical implementations.
CIW executes their native APIs in pinned subprocesses, retains their complete
artifacts, and presents the same results through terminal inspection and the
Godot Workbench tab. They share a catalogue and replay history; they do not
become sensor observations or state estimates through integration.

| Source kind / operation | Native repository and role | Implemented claim |
| --- | --- | --- |
| `covariance-geometry` / `ciw.covariance-geometry.v1` | [Covariance Geometry and Geodesic Testbed](https://github.com/giasonpooni/Covariance-Geometry-and-Geodesic-Testbed), `cggt` | Affine-invariant distance and sampled geodesics between declared SPD matrices |
| `mesh-path` / `ciw.mesh-path.v1` | [Intrinsic Surface Geodesics Testbed](https://github.com/giasonpooni/Intrinsic-Surface-Geodesics-Testbed), `isgt` | Shortest paths along triangle-mesh edges, with Euclidean lower bounds and an explicit discretization gap |
| `translation-flow` / `ciw.translation-flow.v1` | [Translation Surface Dynamics Explorer](https://github.com/giasonpooni/Translation-Surface-Dynamics-Explorer), `tsde` | Exact rational trajectory prefixes on connected square-tiled translation surfaces |

The exact commit and source-tree pins are in
[`geometry_research.py`](../src/ciw/geometry_research.py). All three repositories
retain their MIT licences. Their installed-package tests run on Linux and
Windows with Python 3.11 and 3.12. CIW's shared gate uses Python 3.12 and the
workbench's NumPy 2.4.3 pin; the mesh and translation algorithms themselves use
the Python standard library.

## Start and exercise the shared session

Bind clean checkouts at the approved revisions. Build output, extra source files
and changed tracked bytes do not qualify as the pinned runtime. Binding paths
come from trusted startup arguments, never from saved JSON.

```sh
python -m pip install -e '.[dev]'
ciw serve \
  --covariance-geometry-repo /trusted/geometry/cggt \
  --intrinsic-surface-repo /trusted/geometry/isgt \
  --translation-surface-repo /trusted/geometry/tsde \
  --output-dir results/geometry-session

# In another terminal:
python examples/geometry-research/run.py
ciw send source.list
ciw send bundle.list
ciw send execution.list
ciw send workspace.save
```

The example executes and replays all three native providers in one live
session. The shared requests are `source.add`, `operation.execute`, `bundle.get`,
`bundle.replay`, `result.get`, `instrument.inspect` and `experiment.inspect`.
Selecting a retained result does not recompute it. All three have
`fusion_context: null` and are excluded from `fusion.list`.

Source envelopes have exactly `schema`, `experiment_id`, `configuration` and
`request`. Their schemas are `ciw.<source-kind>-source.v1`; the corresponding
retained bundles are `ciw.<source-kind>-session.v1`. `request` is the native
provider's complete versioned request. See the shipped
[covariance](../examples/geometry-research/covariance-geometry.json),
[mesh](../examples/geometry-research/mesh-path.json) and
[translation](../examples/geometry-research/translation-flow.json) examples.
The envelope's configuration fixes the workbench claim and admission policy.
Sources are bounded to 128 KiB; native profiles impose their own tighter limits.

## Covariance geometry

Both matrices use one declared ordered coordinate list, coordinate units and
frame. The profile supports 1–8 coordinates and 2–33 increasing geodesic
parameters including zero and one. Inputs must already be exactly symmetric
and strictly positive definite within declared eigenvalue and condition limits.
No jitter, eigenvalue clipping or input projection is performed. Binary64
intermediate symmetrization is explicitly bounded and recorded.

The provider owns the affine-invariant calculation. The result retains matrix
spectra, eigenvalue margins, conditioning, decomposition residuals, endpoint
checks, distance symmetry, constant-speed checks and log-determinant affinity.
The workbench preserves full matrices and displays their entries, with the
metric distance in unit `1`. Geodesic parameter is not measurement time, and
uncertainty in the declared covariance entries has not been estimated.

The analytical example uses diagonal matrices `(1,4)` and `(4,16)`; its midpoint
is `(2,8)` and distance is `sqrt(2)*log(4)`. Tests also exercise nonorthogonal
congruence and the midpoint matrix equation. These checks establish the bounded
mathematical profile, not that a supplied matrix models physical uncertainty.

## Mesh edge-path baseline

The request retains vertices, triangle indices, units, coordinate frame,
provenance, preprocessing declarations and their content digest. The profile
accepts 3–256 vertices and 1–512 triangles. It rejects degeneracy, duplicate
faces, coincident or isolated vertices and nonmanifold edges/vertex links.
Boundary edges, disconnected components and inconsistent face orientation are
reported. Self-intersections are not checked; the mesh is not silently repaired.

The native algorithm is Dijkstra on the mesh-edge graph. An edge path is an
admissible path on the declared piecewise-flat mesh, so its length supplies a
numerical upper bound on the continuous surface distance. It is **not a
continuous surface-geodesic solver**. The straight Euclidean chord supplies a
lower bound; these binary64 bounds are not certified floating-point intervals.
For disconnected endpoints, distance and upper bound remain `null` and the
path is empty. No zero or infinity is substituted for an unreachable result.

The planar unit-square example has edge-path length `2` and continuous planar
reference `sqrt(2)`. Refinement tests retain the error and show that a fixed set
of edge directions can preserve directional bias even as triangles get smaller.
Discretization uncertainty, scan noise and calibration remain unestablished.
Mesh source units are `m`, `mm` or `normalized_length`; no conversion is inferred.

## Translation-surface flow

The declared surface is assembled from 1–32 unit squares using `right` and `up`
permutations and inverse maps for left/down crossings. The native provider
checks connected gluing, corner identifications and the resulting genus. The
three-square example is a genus-two surface, distinct from a torus cover.

Position, direction, duration and event times are canonical reduced rational
strings, such as `"1/3"`. Inputs use bounded rational arithmetic; there is no
epsilon-based event ordering. A trajectory begins inside a square. The bench
limits the native event budget to 256 crossings; the standalone provider also
has its own larger bounded profile. Every segment and directed gluing event is
retained, including coordinates before and after the translation.

Native status is one of `completed`, `stopped_at_vertex` or
`event_budget_exhausted`. Vertex hits stop explicitly, even when the requested
duration ends at the vertex; no continuation is inferred. Partial records are
useful diagnostics and remain visibly partial. `elapsed + remaining` accounts
for the requested duration. The view's numeric panels are binary64 projections
of retained exact rationals, and the parameter is not a physical clock.

This profile does not claim arbitrary polygon support, physical embedding,
ergodicity, long-time asymptotics or an SP1 proof.

## Evidence, replay and validation

Each native artifact retains the exact request, request digest, numerical
evidence and artifact digest. CIW separately retains submitted source bytes,
execution and result occurrences, provider source tree, interpreter and
dependency identities. Creation performs a fresh same-runtime reproduction;
replay creates new occurrences. Replay is labelled `independent: false`, and
matching content hashes do not authenticate a fabricated numerical report.

Reopening a workspace requires no provider installation for inspection. A new
execution or replay still requires an explicit matching host binding. Retained
JSON cannot choose a module or an executable. The offline validators check
supported contracts, status, identities and evidence consistency; they do not
execute the numerical algorithms or establish physical truth.

```sh
# Fetch the exact public pins into temporary checkouts, install a CIW wheel in
# isolation, and exercise all three providers with no skipped native tests:
python scripts/check_geometry_research.py --output-dir results/geometry-gate

# Or validate already provisioned clean cggt/isgt/tsde directories:
python scripts/check_geometry_research.py \
  --stack-root /trusted/geometry --output-dir results/geometry-gate-local
```

The Linux/Windows `geometry-research` [provider gate](PIPELINES.md#provider-gates) retains all
three original/replay pairs and the shared workspace. It checks analytic
anchors, refusal cases, provider drift, retained bindings, live session changes,
offline restore and read-only inspection. Native repositories separately test
their installed APIs and CLIs. Dedicated ICRH profiles for these three operations
have not been added; no independent harness certification is claimed.

Next extensions should be separate versioned profiles: continuous paths through
mesh-face interiors; additional covariance metrics; and larger or more general
translation surfaces. Physical use requires explicit registration, calibration
and consumer contracts beyond these mathematical references.
