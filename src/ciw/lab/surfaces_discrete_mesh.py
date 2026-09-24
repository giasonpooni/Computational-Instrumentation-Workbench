"""Triangle-mesh geodesics and geometry uncertainty, tasks T038-T044.

Scope: straightest geodesics traced by edge unfolding on generated triangle
meshes (icosphere, prism cylinder, Schwarz lantern, planar grids, torus),
edge-graph, Steiner-graph and heat-method distances, their convergence to the
smooth ``ciw.lab.surfaces`` references, finite-difference mesh Jacobi fields
against the smooth ``ciw.lab.jacobi`` transfer, angle-defect curvature,
mesh-quality effects, named refusal states for invalid surface data, and the
propagation of declared Gaussian vertex noise and sensor noise.

Non-claims: every mesh is generated from a declared formula in normalized
units and every noise model is a declared synthetic model. No scanned surface,
scanner, tracker or tape was measured, so the accuracy of real scanned
surfaces, real sensors and any production threshold is recorded as
``not_established``. Numerical agreement shown here is agreement between
computations; the observed convergence orders are properties of these
generated mesh families, not of meshes reconstructed from data.
"""
from __future__ import annotations

import math

import numpy as np

from . import svg
from .evidence import finding, holds
from .registry import task
from . import surfaces_discrete_mesh_studies as S

MODULE = "src/ciw/lab/surfaces_discrete_mesh.py"
GEOMETRY = "src/ciw/lab/surfaces_discrete_mesh_geometry.py"
STUDIES = "src/ciw/lab/surfaces_discrete_mesh_studies.py"
DOC = "docs/lab/MESH_GEODESICS.md"
TESTS = "tests/test_lab_surfaces_discrete_mesh.py"
FILES = (MODULE, GEOMETRY, STUDIES, DOC)
# Section-wide contract test: each next step names forward work.
NEXT_STEP_TEST = f"{TESTS}::test_next_steps_name_forward_work"
PRODUCER = "ciw.lab.surfaces_discrete_mesh"
GEOM = "ciw.lab.surfaces_discrete_mesh_geometry"
STUD = "ciw.lab.surfaces_discrete_mesh_studies"
TIGHT = {"abs": 1e-9, "rel": 1e-6}
RATE = {"abs": 1e-6, "rel": 1e-6}
EXACT_TOLERANCE = {"abs": 0, "rel": 0}


def _memo(ctx, name, compute):
    return ctx.memo(f"surfaces_discrete_mesh:{name}", compute)


def check(reference, observed, tolerance, comparison="abs_le", *, kind) -> dict:
    """One check object; ``passed`` comes from ciw.lab.evidence.holds, as the validator recomputes it."""
    observed, tolerance = float(observed), float(tolerance)
    return {"reference_kind": kind, "reference": reference, "observed": observed, "tolerance": tolerance,
            "comparison": comparison, "passed": holds(observed, tolerance, comparison)}


def refusal(reference, expected, observed) -> dict:
    observed = observed if observed is not None else "none"
    return {"reference_kind": "refusal", "reference": reference, "expected_refusal": expected,
            "observed_refusal": observed, "passed": observed == expected}


def _u(kind, value, basis) -> dict:
    """Per-finding uncertainty: truncation_bound, monte_carlo_95ci, roundoff or reference_error."""
    return {"kind": kind, "value": float(value), "basis": basis}


EXACT = _u("roundoff", 0.0, "exact comparison of refusal codes, integer counts or ordered code lists")


def _mc(samples) -> dict:
    return _u("monte_carlo_95ci", 1.96 * math.sqrt(2.0 / (samples - 1)),
              f"relative 95% half-width of a Gaussian sample variance from {samples} seeded samples")


def _binomial(fraction, samples) -> dict:
    return _u("monte_carlo_95ci", 1.96 * math.sqrt(max(fraction * (1 - fraction), 0.25 / samples) / samples),
              f"95% half-width of a sample fraction from {samples} seeded samples (floored at p = 1/(4N))")


def jsonable(value):
    """Plain JSON types; nonfinite floats become None so retained artifacts stay strict JSON."""
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def generator(qualified, **params) -> dict:
    """Generator provenance: the fully qualified implementation name plus its declared parameters."""
    return {"name": qualified, **jsonable(params)}


def fields(hypothesis, model, inputs, observation, invariant, experiment, result, uncertainty, failure_modes,
           assumptions, next_task) -> dict:
    return {"hypothesis": hypothesis, "mathematical_model": model, "input_data": inputs,
            "observation_model": observation, "expected_invariant": invariant, "experiment": experiment,
            "numerical_result": result, "uncertainty": uncertainty, "failure_modes_checked": failure_modes,
            "unresolved_assumptions": assumptions, "recommended_next_task": next_task}


# Each task's next step names its own open question, never a queue task that has already run.
NEXT_STEPS = {
    "T038": ("Complete T038: implement an exact two-point polyhedral geodesic (MMP or ICH, or iterative edge flipping) "
             "and compare it with the Steiner, heat-method and traced lengths."),
    "T039": ("Deferred research question: measure the heat-method convergence rate for time steps t = m h^2 over a "
             "range of m (only m = 1, as Crane et al. recommend, is used here) and the endpoint-error order for "
             "geodesic directions sampled over their angle to the lattice rows, on which the retained empirical order "
             "depends."),
    "T040": ("Deferred research question: derive the mechanism of the valence-6 curvature plateau (about 2.6e-3) on "
             "the icosahedral mirror planes, for example by testing whether a smooth refinement (Loop subdivision "
             "projected to the sphere) instead of the recursive midpoint map removes it, and establish or refute "
             "pointwise convergence off the planes beyond level 7 and on irregular torus meshes."),
    "T041": ("Deferred research question: attribute geodesic and curvature errors per region (per-vertex error "
             "against local triangle quality) instead of summarizing each mesh by its worst triangle, and test "
             "whether the radius-ratio association holds within each mesh family separately (it is not positive "
             "within the latitude-longitude family)."),
    "T042": ("Deferred research question: detect the defects recorded here as undetected (self-intersections between "
             "non-adjacent faces, unwelded seams of coincident duplicate vertices, duplicate faces) as named refusal "
             "states, and measure the false-positive rate of every refusal on valid meshes, including legitimate "
             "sharp creases that folded_face now refuses."),
    "T043": ("Deferred research question: re-trace the geodesic per Monte Carlo sample past the strip-dependent "
             "corridor threshold, where the fixed-strip distance stops being the geodesic distance, and propagate a "
             "correlated, anisotropic vertex covariance and independently measured normals. A realistic scanner "
             "covariance needs acquired scan data (hardware-gated). Route: T043 would read the scan export as an "
             "operator capture (ctx.capture('scan-export'), bound with ciw lab run T043 --capture "
             "scan-export=PATH) and fit the covariance from it as a computational finding; a scanner claim needs, "
             "besides an acquisition record (device, raw_sha256 of the captured bytes, acquired_at, calibration), a "
             "probe of the scanner on the analysing host that succeeds in T043 (runner.CAPTURE_INSTRUMENTS has no "
             "entry for scan-export) or a signed-capture trust anchor, and the run is retained with ciw lab "
             "hardware retain under lab/hardware/<run-id>. Neither the capture reader nor a scanner probe exists, so "
             "T043's scanner claims stay not_established even when such data exist."),
    # T045 delivered the split for model-derived distances on parametric surfaces; the mesh form is still open
    # there (T045's own next step names the same conversion).
    "T044": ("Deferred research question: fill reconstructed_surface_distance's geometry_m2 from a mesh, as "
             "sigma_vertex^2 |grad d|^2 with T043's linearized vertex-noise gain (valid only while the marker segment "
             "stays in its face corridor) instead of declared analytic surface parameters, and check it against this "
             "task's nested Monte Carlo. Partly delivered by T045, which carries the geometry/sensor split for "
             "model-derived distances on parametric surfaces (plane, sphere, cylinder geodesic) and names the same "
             "mesh conversion as its next step, and by T140's instrument and geometry budget for manufacturing "
             "predictions. An intrinsic_geodesic_distance reading keeps one sensor sigma: as in this task's residual, "
             "its geometry part belongs to the model prediction it is compared with, not to the reading."),
}


def declared_strip_agreement(first: dict, first_samples: int, second, sigmas, second_samples: int) -> dict:
    """Agreement of two independent Monte Carlo estimates of the declared strip's corridor-leave fraction.

    ``first`` maps str(sigma) to a fraction of ``first_samples`` samples; ``second`` lists fractions of
    ``second_samples`` samples in the order of ``sigmas``. Each difference is compared with the standard error of
    a difference of two independent binomial fractions, whose per-estimate variance is floored at 0.5 / n^2 so that
    two exact zeros agree; 1.96 standard errors bound the 95% interval.
    """
    rows = []
    for sigma, q in zip(sigmas, second):
        p = first[str(sigma)]
        error = math.sqrt(max(p * (1 - p), 0.5 / first_samples) / first_samples
                          + max(q * (1 - q), 0.5 / second_samples) / second_samples)
        rows.append({"sigma": sigma, "difference": abs(p - q), "z": abs(p - q) / error, "interval": 1.96 * error})
    top = max(rows, key=lambda r: r["difference"])
    return {"other": [float(q) for q in second], "max_abs": top["difference"], "interval_at_max": top["interval"],
            "sigma_at_max": top["sigma"], "max_z": max(r["z"] for r in rows)}


# The agreement check's bound: 1.96 standard errors of the difference bound its 95% interval.
AGREEMENT_Z = 1.96


def agreement_text(agreement: dict) -> str:
    """The report sentence on the two declared-strip estimates, decided by the same test as the finding's check."""
    where = (f"largest difference {agreement['max_abs']:.4f} against {agreement['interval_at_max']:.4f} at "
             f"sigma={agreement['sigma_at_max']:g}")
    if holds(agreement["max_z"], AGREEMENT_Z, "le"):
        return ("the two independent estimates agree within the 95% interval of a difference of two independent "
                f"fractions ({where})")
    return ("the two independent estimates differ beyond the 95% interval of a difference of two independent "
            f"fractions ({where}; largest z {agreement['max_z']:.2f} against {AGREEMENT_Z}), so the corridor "
            "finding's agreement check fails")


def orders(hs, errors) -> dict:
    """Least-squares order, pairwise local orders and the largest deviation between them."""
    fit = S.fitted_order(hs, errors)
    local = [math.log(errors[i] / errors[i + 1]) / math.log(hs[i] / hs[i + 1]) for i in range(len(hs) - 1)]
    return {"fit": fit, "local": local, "spread": max(abs(p - fit) for p in local)}


def _order_uncertainty(study) -> dict:
    return _u("reference_error", study["spread"],
              "largest deviation of a pairwise local order from the least-squares order")


def _physical(claim, domain="physical"):
    return finding(claim, domain, None, {"notes": "No scanned surface, device or calibration was acquired; "
                                                  "the computation is synthetic."})


def _missing_witness(claim, reference):
    """A counterexample whose witness no longer exists is recorded as a failed check, not a crash."""
    return finding(claim, "numerical", {"witnesses": 0},
                   {"checks": [check(reference, 0.0, 1.0, "ge", kind="self_convergence")]},
                   uncertainty=EXACT, tolerance=EXACT_TOLERANCE)


# ---------------------------------------------------------------- T038
@task("T038", changed_files=FILES, regression_tests=(
    NEXT_STEP_TEST,
    f"{TESTS}::test_tracer_is_exact_on_developable_meshes",
    f"{TESTS}::test_graph_distances_and_steiner_sandwich",
    f"{TESTS}::test_dijkstra_check_falls_back_without_scipy",
    f"{TESTS}::test_solver_task_report"))
def mesh_geodesic_solver(ctx):
    traces = _memo(ctx, "sphere-traces", S.sphere_trace_study)
    cylinder = _memo(ctx, "cylinder", S.cylinder_study)
    plane = _memo(ctx, "plane", S.plane_study)
    nested = _memo(ctx, "steiner-nested", S.steiner_nested_study)
    independent = _memo(ctx, "dijkstra-independent", S.dijkstra_independent)
    statuses = sorted({t["status"] for row in traces["rows"] for t in row["traces"]})
    completed = sum(r["completed"] for r in traces["rows"])
    plane_error = max(r["max_trace_error"] for r in plane["rows"])
    development = max(r["development_error"] for r in cylinder["rows"])
    unfold = max(r["max_unfold_minus_trace"] for r in traces["rows"])
    gaps = nested["gap_to_traced"]
    ks = sorted(int(k) for k in gaps)
    mean_gap = {k: float(np.mean(gaps[str(k)])) for k in ks}
    gap_steps = [mean_gap[a] - mean_gap[b] for a, b in zip(ks, ks[1:])]
    ctx.artifact_json("sphere-traces.json", jsonable(traces))
    ctx.artifact_json("cylinder-and-plane.json", jsonable({"cylinder": cylinder, "plane": plane}))
    ctx.artifact_json("graph-distances.json", jsonable({"steiner_nested": nested, "dijkstra_independent": independent}))

    # The claim and prose are the same with and without scipy; only the basis (and label) records scipy.
    dijkstra_basis = {"generator": generator(f"{GEOM}.icosphere", level=independent["level"]),
                      "checks": [check(f"dense Floyd-Warshall (ciw) on icosphere-{independent['floyd_level']}",
                                       independent["floyd_max_abs"], 1e-12, kind="cross_implementation")]}
    dijkstra_value = independent["floyd_max_abs"]
    if independent.get("scipy_max_abs") is not None:
        dijkstra_value = max(dijkstra_value, independent["scipy_max_abs"])
        dijkstra_basis["independent_check"] = dict(
            check(f"scipy.sparse.csgraph.dijkstra on the same icosphere-{independent['level']} edge weights "
                  "(equal up to summation order)", independent["scipy_max_abs"], 1e-12, kind="exact_arithmetic"),
            producer={"implementation": PRODUCER, "revision": "working tree"},
            checker={"implementation": "scipy.sparse.csgraph.dijkstra", "revision": independent["scipy"]})
    roundoff = _u("roundoff", 1e-15, "double-precision rounding of deterministic geometry (observed 1e-15 relative)")
    findings = [
        finding("Straightest geodesics on sheared planar meshes coincide with straight lines", "numerical",
                plane_error, {"generator": generator(f"{GEOM}.plane_mesh", shears=[r["shear"] for r in plane["rows"]]),
                              "checks": [check("exact line start + s * direction", plane_error, 1e-12,
                                               kind="analytic")]},
                unit="normalized length", uncertainty=_u("roundoff", plane_error, "largest deviation observed"),
                tolerance={"abs": 1e-12, "rel": 0.0}),
        finding("Traced prism-cylinder geodesics match the exact planar development of the mesh", "numerical",
                development, {"generator": generator(f"{GEOM}.cylinder_mesh", n=[r["n"] for r in cylinder["rows"]]),
                              "checks": [check("development of planar rectangles, circumference 2 n R sin(pi/n)",
                                               development, 1e-11, kind="analytic")]},
                unit="normalized length", uncertainty=_u("roundoff", development, "largest deviation observed"),
                tolerance={"abs": 1e-11, "rel": 0.0}),
        finding("The unfolded face-strip distance equals the traced length on every completed sphere trace",
                "numerical", {"max_abs_difference": unfold, "trace_statuses": statuses},
                {"generator": generator(f"{GEOM}.icosphere", levels=[r["level"] for r in traces["rows"]]),
                 "checks": [check("ciw strip layout from edge lengths against the ciw tracer's 3D edge rotation",
                                  unfold, 1e-11, kind="cross_implementation")]},
                uncertainty=_u("roundoff", unfold, "largest difference observed"),
                tolerance={"abs": 1e-11, "rel": 0.0}),
        finding("Heap Dijkstra edge-graph distances agree with a dense Floyd-Warshall recomputation (and "
                "scipy.sparse.csgraph when installed)", "numerical", dijkstra_value, dijkstra_basis,
                unit="normalized length", uncertainty=_u("roundoff", dijkstra_value, "summation-order rounding"),
                tolerance={"abs": 1e-12, "rel": 0.0}),
        finding("Nested Steiner-graph distances never increase with k, and every Steiner-graph edge lies inside one "
                "mesh face", "numerical",
                {"max_increase_with_k": nested["max_increase_with_k"],
                 "edges_outside_faces": nested["edges_outside_faces"], "edges_checked": nested["edges_checked"],
                 "vertex0_to_last": nested["vertex0_to_last"]},
                {"derivation": "With k = 2^j - 1 the node sets are nested and every face keeps all node pairs, so each "
                               "coarser graph edge is an edge of the finer graph; a segment between two points of one "
                               "planar convex face lies in that face, so every graph path is a surface path and graph "
                               "distances bound the (uncomputed) polyhedral distance from above",
                 "checks": [check("max over vertices of distance(k') - distance(k) for nested k' > k",
                                  nested["max_increase_with_k"], 1e-12, "signed_le", kind="invariant"),
                            check("graph edges (k = 0, 1, 3, 7, with traced endpoints) whose end nodes share no face, "
                                  "by a point-in-face test (plane offset and barycentric coordinates within 1e-12) "
                                  "independent of the graph construction", nested["edges_outside_faces"], 0.0, "le",
                                  kind="exact_arithmetic")]},
                uncertainty=roundoff, tolerance=TIGHT),
        finding("Steiner distances between traced endpoints stay above the traced length and approach it as k grows",
                "numerical", {"min_gap": nested["min_gap"], "mean_gap_by_k": {str(k): v for k, v in mean_gap.items()}},
                {"generator": generator(f"{GEOM}.steiner_graph", level=nested["level"],
                                        traced=len(nested["traced_lengths"])),
                 "checks": [check("smallest gap (Steiner distance - traced length) plus a 1e-12 rounding allowance",
                                  nested["min_gap"] + 1e-12, 0.0, "signed_ge", kind="self_convergence"),
                            check("smallest decrease of the mean gap between consecutive nested k",
                                  min(gap_steps), 0.0, "signed_ge", kind="self_convergence")]},
                unit="normalized length", uncertainty=roundoff, tolerance=TIGHT),
        _physical("Straightest geodesics on a mesh reconstructed from a real scan reproduce the geodesics of the "
                  "scanned physical surface"),
    ]
    result = (f"Plane traces exact to {plane_error:.1e}; prism-cylinder traces match the development to "
              f"{development:.1e}; unfolded strip length equals traced length to {unfold:.1e} over {completed} "
              f"sphere traces (statuses {statuses}); edge Dijkstra agrees with its reference shortest-path "
              f"implementations to {dijkstra_value:.1e}; Steiner distances to traced endpoints exceed the traced "
              f"length by mean " + ", ".join(f"{mean_gap[k]:.2e} (k={k})" for k in ks) + f"; "
              f"{nested['edges_outside_faces']} of {nested['edges_checked']} Steiner-graph edges leave a face. "
              "Delivered: an initial-value straightest-geodesic tracer and approximate edge-graph, Steiner and "
              "heat-method distances. Not delivered: an exact two-point polyhedral distance (MMP, ICH or edge "
              "flipping), so no polyhedral distance is computed to compare these with.")
    # Partial: the task asks for a geodesic solver; the exact two-point (shortest-path) solver was not implemented.
    return {"state": "partial", "findings": findings, "fields": fields(
        "Unfolding across edges (straight in faces, equal angles at edges) yields exact straightest geodesics on "
        "developable meshes, and Steiner-graph paths are surface paths (every graph edge lies in one face), so their "
        "lengths are upper bounds on the polyhedral distance; the polyhedral distance itself is not computed here.",
        "Straightest geodesic: in each face a straight segment; at an edge the direction keeps its edge component and "
        "the magnitude of its perpendicular component (rotation about the edge). Distances: Dijkstra on the edge "
        "graph and on the graph of k Steiner points per edge (all pairs inside each face); vertex hits are refused.",
        ["icosphere levels 1-7 (unit sphere)", "prism cylinder R=1, n=8..128 sectors",
         "sheared planar 8x8 grids (shear 0-1.5)", "six declared sphere geodesics (chart point, heading)"],
        "No physical observation; traced endpoints, lengths and graph distances in normalized units.",
        "Planar and prism meshes are intrinsically flat, so traces must equal straight lines of the development; "
        "Steiner distances with nested point sets are nonincreasing in k, and every graph edge joins two points of "
        "one face.",
        "Trace declared geodesics; compare with exact developments; re-derive lengths by a second ciw implementation "
        "(strip layout from edge lengths); compare Dijkstra with a dense Floyd-Warshall and, when installed, "
        "scipy.sparse.csgraph; test every Steiner-graph edge for a common face; sandwich traced lengths by Steiner "
        "distances.",
        result,
        "Deterministic computation; floating-point rounding only (differences at 1e-15 relative). The strip layout "
        "and Floyd-Warshall are ciw code (same origin); only the scipy comparison is independent. The Steiner "
        "sandwich is consistent with, but does not prove, that the traced geodesics are shortest paths.",
        ["vertex hits (refused, none occurred in the declared set)", "boundary reached", "tracing through a face "
         "without an exit edge", "strip unfolding sign conventions", "graph duplicates from shared face edges",
         "scipy absent (Floyd-Warshall only; the Dijkstra finding is then numerically_verified)"],
        ["Partial delivery: the solver is an initial-value tracer (straightest geodesics from a point and heading) "
         "plus approximate distances. No exact two-point polyhedral geodesic (MMP, ICH or iterative edge flipping) "
         "is implemented, so no shortest path between two given points is solved exactly and the graph and heat "
         "distances are not compared with an exact polyhedral distance.",
         "Vertex hits are refused rather than continued by the Polthier-Schmies angle-bisection rule.",
         "Traced geodesics are shortest paths only when no shorter corridor exists; not proved here."],
        NEXT_STEPS["T038"])}


# ---------------------------------------------------------------- T039
@task("T039", changed_files=FILES, regression_tests=(
    NEXT_STEP_TEST,
    f"{TESTS}::test_convergence_orders_on_small_levels",
    f"{TESTS}::test_convergence_task_report"))
def mesh_refinement_convergence(ctx):
    traces = _memo(ctx, "sphere-traces", S.sphere_trace_study)
    cylinder = _memo(ctx, "cylinder", S.cylinder_study)
    distances = _memo(ctx, "distances", S.distance_study)
    rows = traces["rows"]
    fitted = rows[1:]  # the coarsest level is preasymptotic
    hs = [r["h"] for r in fitted]
    o_length = orders(hs, [r["mean_length_defect"] for r in fitted])
    o_endpoint = orders(hs, [r["mean_endpoint"] for r in fitted])
    o_cross = orders(hs, [r["mean_cross_track"] for r in fitted])
    per_trace = [[t["endpoint"] for t in r["traces"]] for r in rows]
    increases = [{"trace": j, "from_level": rows[i]["level"], "to_level": rows[i + 1]["level"]}
                 for j in range(len(per_trace[0])) for i in range(len(rows) - 1)
                 if per_trace[i + 1][j] > per_trace[i][j]]
    crow = cylinder["rows"]
    o_helix = orders([r["h"] for r in crow], [r["helix_error"] for r in crow])
    leading = cylinder["leading_constant"]
    scaled = [r["scaled_error"] / leading for r in crow]
    exact_gap = max(abs(r["helix_error"] - r["exact_prediction"]) for r in crow)
    # |n^2 e / C - 1| is bounded by the chord-position term plus the development term's own deviation from C.
    bound_excess = max(abs(s - 1) - (r["chord_term_bound"] * r["n"] ** 2 / leading
                                     + abs(r["circumference_prediction"] * r["n"] ** 2 / leading - 1))
                       for s, r in zip(scaled, crow))
    drows = distances["rows"]
    edge = [r["edge_max_signed_rel"] for r in drows]
    aitken = edge[-1] - (edge[-1] - edge[-2]) ** 2 / ((edge[-1] - edge[-2]) - (edge[-2] - edge[-3]))
    steiner = {k: [r[f"steiner{k}_max_abs_rel"] for r in drows] for k in (1, 3)}
    heat_rows = [r for r in drows if "heat_max_abs" in r]
    o_heat = orders([r["h"] for r in heat_rows], [r["heat_max_abs"] for r in heat_rows])
    table = {"sphere": [{k: r[k] for k in ("level", "h", "mean_endpoint", "max_endpoint", "mean_cross_track",
                                           "mean_along_track", "mean_length_defect", "completed")} for r in rows],
             "orders": {"length_defect": o_length, "endpoint": o_endpoint, "cross_track": o_cross,
                        "helix": o_helix, "heat": o_heat},
             "per_trace_endpoint": per_trace, "per_trace_increases": increases,
             "edge_graph_floor": {"prediction": S.EDGE_GRAPH_FLOOR, "aitken": aitken},
             "cylinder": crow, "distances": drows}
    ctx.artifact_json("convergence.json", jsonable(table))
    ctx.artifact_text("convergence.svg", svg.line_plot([
        ("sphere endpoint", [r["h"] for r in rows], [r["mean_endpoint"] for r in rows]),
        ("sphere cross-track", [r["h"] for r in rows], [r["mean_cross_track"] for r in rows]),
        ("sphere length", [r["h"] for r in rows], [r["mean_length_defect"] for r in rows]),
        ("cylinder helix", [r["h"] for r in crow], [r["helix_error"] for r in crow]),
        ("heat (max abs)", [r["h"] for r in heat_rows], [r["heat_max_abs"] for r in heat_rows]),
        ("edge graph |rel|", [r["h"] for r in drows], [r["edge_max_abs_rel"] for r in drows]),
        ("Steiner k=3 |rel|", [r["h"] for r in drows], steiner[3])],
        title="Mesh geodesics under refinement", xlabel="mean edge length h", ylabel="error", logx=True, logy=True))
    levels = [r["level"] for r in fitted]
    findings = [
        finding("Traced-geodesic length defect on icospheres converges at second order", "numerical",
                {"order": o_length["fit"], "local_orders": o_length["local"]},
                {"derivation": "Inscribed icosphere: chords are shorter than arcs by O(h^2) relative, so the "
                               "intrinsic metric of the mesh differs from the sphere's by O(h^2)",
                 "checks": [check("fitted order minus the inscribed-metric prediction 2", o_length["fit"] - 2.0, 0.2,
                                  kind="self_convergence"),
                            check("largest |pairwise local order - 2|", max(abs(p - 2) for p in o_length["local"]),
                                  0.2, "le", kind="self_convergence")]},
                uncertainty=_order_uncertainty(o_length), tolerance=RATE),
        finding("Traced-geodesic endpoint error on icospheres decreases at least at first order, with irregular "
                "pairwise local orders", "numerical",
                {"endpoint_order": o_endpoint["fit"], "endpoint_local_orders": o_endpoint["local"],
                 "cross_track_order": o_cross["fit"], "cross_track_local_orders": o_cross["local"],
                 "finest_mean_endpoint": rows[-1]["mean_endpoint"], "per_trace_increases": increases},
                {"generator": generator(f"{GEOM}.icosphere", levels=levels),
                 "checks": [check("fitted endpoint order", o_endpoint["fit"], 1.0, "ge", kind="self_convergence"),
                            check("finest-pair local endpoint order", o_endpoint["local"][-1], 1.0, "ge",
                                  kind="self_convergence"),
                            check("smallest pairwise local endpoint order", min(o_endpoint["local"]), 1.0, "ge",
                                  kind="self_convergence")]},
                uncertainty=_order_uncertainty(o_endpoint), tolerance=RATE),
        finding("Prism-cylinder helix endpoint error equals the closed-form development and chord-position "
                "prediction, and n^2 error tends to L cos(alpha) pi^2 / 6", "numerical",
                {"order": o_helix["fit"], "local_orders": o_helix["local"], "scaled_to_leading_constant": scaled,
                 "max_abs_minus_exact": exact_gap},
                {"derivation": "Development circumference 2 n R sin(pi/n); an endpoint at fraction t of chord i has "
                               "azimuth 2 a i + a + atan((2t - 1) tan a), a = pi/n, so the error is R |(L cos(alpha)/R)"
                               "(a / sin a - 1) + atan(s tan a) - s a| with s = 2t - 1; the chord-position term is at "
                               "most 2 a^3 / (9 sqrt 3) R, an O(1/n) relative oscillation of n^2 error / C",
                 "checks": [check("max over n of |helix error - exact closed form|", exact_gap, 1e-9, kind="analytic"),
                            check("max over n of |n^2 error / C - 1| minus its derived bound", bound_excess, 0.0,
                                  "signed_le", kind="analytic"),
                            check("fitted order - 2", o_helix["fit"] - 2.0, 0.05, kind="self_convergence")]},
                uncertainty=_u("roundoff", exact_gap, "difference from the closed form"), tolerance=RATE),
        finding("Edge-graph Dijkstra distance from a valence-5 vertex keeps a relative-error floor that tends to "
                "sqrt(5) - 2", "numerical",
                {"levels": [r["level"] for r in drows], "max_signed_relative_error": edge,
                 "aitken_extrapolation": aitken, "prediction": S.EDGE_GRAPH_FLOOR},
                {"derivation": "In the limit the star of a valence-5 vertex is a regular 5-star with rows 72 degrees "
                               "apart; the vertex two hops away between two rows sits at e1 + e2, so edge path / "
                               "geodesic -> 2 / (2 cos(pi/5)) = sqrt(5) - 1",
                 "checks": [check("finest max relative error - (sqrt(5) - 2)", edge[-1] - S.EDGE_GRAPH_FLOOR, 5e-3,
                                  kind="analytic"),
                            check("Aitken extrapolation of the last three levels - (sqrt(5) - 2)",
                                  aitken - S.EDGE_GRAPH_FLOOR, 1e-3, kind="analytic"),
                            check("finest minus second-level error (no decrease)", edge[-1] - edge[1], 0.0,
                                  "signed_ge", kind="self_convergence")]},
                uncertainty=_u("truncation_bound", abs(aitken - edge[-1]), "distance of the finest level from the "
                               "Aitken extrapolation"),
                tolerance=TIGHT,
                counterexample={"statement": "Edge-graph shortest paths converge to the geodesic distance under "
                                             "mesh refinement",
                                "witness": {"mesh": "icosphere levels 1-4, source vertex 0 (valence 5)",
                                            "max_signed_relative_error": edge}}),
        finding("Steiner graphs with a fixed number of points per edge keep a relative-error floor", "numerical",
                {"k1_max_abs_relative_error": steiner[1], "k3_max_abs_relative_error": steiner[3]},
                {"generator": generator(f"{GEOM}.steiner_graph", ks=[1, 3], levels=[r["level"] for r in drows]),
                 "checks": [check("k=3 max |relative error| at the finest level", steiner[3][-1], 0.005, "ge",
                                  kind="self_convergence"),
                            check("k=3 finest minus second-level |relative error|", steiner[3][-1] - steiner[3][1],
                                  0.0, "signed_ge", kind="self_convergence")]},
                uncertainty=_u("roundoff", 1e-15, "deterministic graph distances"), tolerance=TIGHT,
                counterexample={"statement": "Refining the mesh alone makes a Steiner-graph distance with fixed k "
                                             "converge to the geodesic distance",
                                "witness": {"k": 3, "max_abs_relative_error": steiner[3]}}),
        finding("Heat-method distance error decreases under refinement at first order", "numerical",
                {"order": o_heat["fit"], "local_orders": o_heat["local"]},
                {"generator": generator(f"{GEOM}.heat_distance", levels=[r["level"] for r in heat_rows], t="h^2"),
                 "checks": [check("fitted order of the max abs error - 1", o_heat["fit"] - 1.0, 0.25,
                                  kind="self_convergence")]},
                uncertainty=_order_uncertainty(o_heat), tolerance=RATE),
        _physical("The observed convergence orders transfer to meshes reconstructed from real scans"),
    ]
    result = (f"Sphere (levels 2-7): length-defect order {o_length['fit']:.2f}, endpoint order "
              f"{o_endpoint['fit']:.2f} (pairwise " + ", ".join(f"{p:.2f}" for p in o_endpoint["local"])
              + f"), cross-track {o_cross['fit']:.2f} (pairwise " + ", ".join(f"{p:.2f}" for p in o_cross["local"])
              + f"), finest mean endpoint error {rows[-1]['mean_endpoint']:.2e} rad; {len(increases)} of "
              f"{len(per_trace[0]) * (len(rows) - 1)} per-trace level steps increase the endpoint error. Cylinder: "
              f"order {o_helix['fit']:.3f}, helix error equals the closed form to {exact_gap:.1e}, n^2 error / C = "
              + ", ".join(f"{s:.4f}" for s in scaled) + ". Edge Dijkstra max signed relative error "
              + ", ".join(f"{e:.4f}" for e in edge) + f" (levels {drows[0]['level']}-{drows[-1]['level']}), Aitken "
              f"{aitken:.5f} vs sqrt(5) - 2 = {S.EDGE_GRAPH_FLOOR:.5f}; Steiner k=3 max |relative error| "
              + ", ".join(f"{e:.4f}" for e in steiner[3]) + f"; heat-method order {o_heat['fit']:.2f}.")
    return {"state": "completed", "findings": findings, "fields": fields(
        "Straightest mesh geodesics converge to smooth geodesics; the length (metric) error is O(h^2) on inscribed "
        "meshes, the lateral error decreases at least like O(h) but not as a clean power law, and edge-graph "
        "distances do not converge.",
        "Inscribed icosphere: chord/arc metric error O(h^2). Lateral error of a straightest geodesic is driven by the "
        "imbalance of vertex curvature point masses on either side of the path (a discrepancy sum), so its order is "
        "not a single power law. Prism cylinder: development circumference 2nR sin(pi/n) gives L cos(alpha) (a / sin a "
        "- 1) ~ L cos(alpha) pi^2 / (6 n^2), plus a chord-position term atan(s tan a) - s a of relative size O(1/n). "
        "Edge graph at a valence-5 source: floor sqrt(5) - 2.",
        ["icosphere levels 1-7", "prism cylinder n=8..128, alpha=0.5, L=3", "six declared sphere geodesics, L=2",
         "graph distances from vertex 0 on icosphere levels 1-4; heat-method distances on levels 1-3"],
        "Endpoint angle on the unit sphere after radial projection; helix error on the cylinder surface; relative "
        "distance error against the great-circle distance.",
        "Length defect order 2; cylinder error equal to its closed form; edge-graph floor sqrt(5) - 2.",
        "Refine, trace, fit log-log slopes over levels 2-7 (the coarsest level is preasymptotic) and report pairwise "
        "local orders; compare graph and heat-method distances against great-circle distances.",
        result,
        "Deterministic; fitted orders depend on the chosen levels and geodesics (six directions), and the pairwise "
        "local orders of the endpoint error range over about 1 to 2. The lateral-error order is an empirical slope, "
        "not a proved rate.",
        ["preasymptotic coarse levels (level 1 excluded from fits)", "vertex hits during refinement (none)",
         "sign of the length defect (inscribed polyhedron is shorter)", "graph metrication plateau",
         "per-trace non-monotone endpoint errors (reported; consistent with the vertex-discrepancy mechanism)",
         "signed versus absolute relative errors (inscribed meshes can be shorter than the sphere)"],
        ["Endpoint-error order is empirical and depends on how geodesic directions align with lattice rows.",
         "Heat-method time step t = h^2 as recommended by Crane et al.; other choices change the rate.",
         "The edge-graph floor is derived for the valence-5 source used here; other sources have other floors "
         "(1/cos(pi/6) - 1 at regular valence 6)."],
        NEXT_STEPS["T039"])}


# ---------------------------------------------------------------- T040
@task("T040", changed_files=FILES, regression_tests=(
    NEXT_STEP_TEST,
    f"{TESTS}::test_flat_jacobi_counterexample_and_valence_limit",
    f"{TESTS}::test_jacobi_task_report"))
def mesh_jacobi_comparison(ctx):
    study = _memo(ctx, "jacobi", S.jacobi_study)
    curvature = _memo(ctx, "curvature", S.curvature_study)
    rows = study["rows"]
    smooth = study["smooth"]
    by_delta = {}
    for r in rows:
        by_delta.setdefault(r["delta"], []).append(r)
    wide = by_delta[max(by_delta)]
    tiny_delta = min(by_delta)
    # The flat regime needs no vertex in the wedge between a pair; the finest level is the declared crossover.
    tiny = [r for r in by_delta[tiny_delta] if r["level"] < rows[-1]["level"]]
    crossover = by_delta[tiny_delta][-1]
    o_wide = orders([r["h"] for r in wide], [r["mean_abs_error"] for r in wide])
    ratios = [a["mean_abs_error"] / b["mean_abs_error"] for a, b in zip(wide, wide[1:])]
    # A missing flat pair leaves a failing check (1.0 > 1e-6) instead of crashing the task.
    flat_dev = max((r["max_flat_deviation"] for r in tiny if r["max_flat_deviation"] is not None), default=1.0)
    identical = sum(r["identical_face_sequences"] for r in tiny)
    pairs = sum(r["pairs"] for r in tiny)
    flat_error = study["flat_value"] - smooth["closed_form"]
    sphere = curvature["sphere"]
    limit = curvature["valence5_limit"]
    k5 = [r["valence5_value"] for r in sphere]
    dev = [abs(v - limit) for v in k5]
    tail = [r for r in sphere if r["level"] >= 4]
    o_k5 = orders([r["h"] for r in tail], [abs(r["valence5_value"] - limit) for r in tail])
    voronoi = [r["voronoi_valence5_error"] for r in sphere]
    gb = max(max(abs(r["gauss_bonnet_residual"]) for r in sphere),
             max(abs(r["gauss_bonnet_residual"]) for r in curvature["torus"]))
    trows = curvature["torus"]
    o_torus = orders([r["h"] for r in trows], [r["max_error"] for r in trows])
    seam = {"levels": [r["level"] for r in tail],
            "base_edge_max_error": [r["valence6_base_edge_max_error"] for r in tail],
            "median_max_error": [r["valence6_median_max_error"] for r in tail],
            "off_mirror_max_error": [r["valence6_off_mirror_max_error"] for r in tail],
            "voronoi_mirror_max_error": [r["voronoi_valence6_mirror_max_error"] for r in tail]}
    mirror_max = [max(a, b) for a, b in zip(seam["base_edge_max_error"], seam["median_max_error"])]
    ctx.artifact_json("jacobi-and-curvature.json", jsonable({"jacobi": study, "curvature": curvature,
                                                             "valence6_mirror": seam}))
    series = [(f"FD Jacobi d={d:g}", [r["h"] for r in rs], [r["mean_abs_error"] for r in rs])
              for d, rs in sorted(by_delta.items(), reverse=True)]
    series += [("|K5 - 1|", [r["h"] for r in sphere], [abs(v - 1) for v in k5]),
               ("|K5 - limit|", [r["h"] for r in sphere], dev),
               ("Voronoi K5 error", [r["h"] for r in sphere], voronoi),
               ("K6 mirror max", [r["h"] for r in tail], mirror_max),
               ("torus max error", [r["h"] for r in trows], [r["max_error"] for r in trows])]
    ctx.artifact_text("jacobi-curvature.svg", svg.line_plot(
        series, title="Mesh Jacobi fields and angle-defect curvature", xlabel="mean edge length h",
        ylabel="error", logx=True, logy=True))
    findings = [
        finding("The smooth ciw.lab.jacobi heading column equals sin(L) on the unit sphere", "numerical",
                smooth["max_error"], {"generator": generator("ciw.lab.jacobi.transfer", steps=200,
                                                             length=study["length"]),
                                      "checks": [check("closed form sin(L)", smooth["max_error"], 1e-8,
                                                       kind="analytic")]},
                uncertainty=_u("truncation_bound", smooth["max_error"], "200-step integration error observed"),
                tolerance={"abs": 1e-9, "rel": 0.0}),
        finding("Finite-difference mesh Jacobi fields with a 0.1 rad heading offset converge to the smooth field",
                "numerical", {"order": o_wide["fit"], "local_orders": o_wide["local"],
                              "mean_abs_error": [r["mean_abs_error"] for r in wide],
                              "levels": [r["level"] for r in wide]},
                {"generator": generator(f"{GEOM}.icosphere", levels=[r["level"] for r in wide], delta=0.1),
                 "checks": [check("fitted order", o_wide["fit"], 1.0, "ge", kind="self_convergence"),
                            check("finest-pair local order", o_wide["local"][-1], 1.0, "ge", kind="self_convergence"),
                            check("smallest coarse / fine error ratio between levels", min(ratios), 1.0, "ge",
                                  kind="self_convergence"),
                            check("mean abs error at the finest level", wide[-1]["mean_abs_error"], 0.01, "le",
                                  kind="self_convergence")]},
                uncertainty=_order_uncertainty(o_wide), tolerance=RATE),
        finding("At fixed mesh a 1e-5 heading offset gives the flat Jacobi value L instead of sin(L) while no vertex "
                "lies between the paired geodesics", "numerical",
                {"max_flat_deviation": flat_dev, "identical_face_sequences": identical, "pairs": pairs,
                 "levels": [r["level"] for r in tiny], "error_vs_smooth": flat_error,
                 "crossover": {"level": crossover["level"], "pairs": crossover["pairs"],
                               "identical_face_sequences": crossover["identical_face_sequences"],
                               "swept_vertices_per_pair": crossover["swept_vertices"]},
                 "swept_vertices_per_pair": [r["swept_vertices"] for r in by_delta[tiny_delta]]},
                {"derivation": "Paired traces through the same face sequence share one planar development, so "
                               "|X+ - X-| = 2 L sin(delta) and j = L exactly",
                 "checks": [check("max |j_FD - L| on levels with identical face sequences", flat_dev, 1e-6,
                                  kind="analytic"),
                            check("pairs with differing face sequences below the crossover level", pairs - identical,
                                  0.0, kind="exact_arithmetic"),
                            check("pairs straddling a vertex at the crossover level",
                                  crossover["pairs"] - crossover["identical_face_sequences"], 1.0, "ge",
                                  kind="self_convergence")]},
                uncertainty=_u("roundoff", flat_dev, "endpoint rounding amplified by 1 / (2 sin delta) ~ 5e4"),
                tolerance={"abs": 1e-8, "rel": 1e-6},
                counterexample={"statement": "On a fixed mesh the finite-difference Jacobi field converges to the "
                                             "smooth Jacobi field as the perturbation tends to zero",
                                "witness": {"delta": tiny_delta, "levels": [r["level"] for r in tiny],
                                            "j_fd": study["flat_value"], "j_smooth": smooth["closed_form"]}}),
        finding("Angle-defect curvature at valence-5 icosphere vertices converges to (4.5 - 1.5 sqrt 5) K, not K",
                "numerical", {"valence5_values": k5, "predicted_limit": limit, "order_to_limit": o_k5["fit"],
                              "local_orders_to_limit": o_k5["local"]},
                {"derivation": "Regular valence-n star on an umbilic surface: defect ~ Voronoi cell area, ratio "
                               "3 / (4 cos^2(pi/n)); docs/lab/MESH_GEODESICS.md",
                 "checks": [check("finest valence-5 value - limit", k5[-1] - limit, 1e-4, kind="analytic"),
                            check("order of |K5 - limit| - 2 (levels 4-7)", o_k5["fit"] - 2.0, 0.2,
                                  kind="self_convergence")]},
                uncertainty=_u("truncation_bound", dev[-1] / 3, "remaining approach to the limit if the deviation "
                               "keeps falling by the observed factor 4 per level"),
                tolerance=TIGHT,
                counterexample={"statement": "The angle defect over one third of the incident area converges "
                                             "pointwise to the Gaussian curvature under refinement",
                                "witness": {"mesh": "icosphere, 12 valence-5 vertices", "limit": limit,
                                            "finest_value": k5[-1]}}),
        finding("With the mixed Voronoi area the valence-5 curvature estimate converges", "numerical", voronoi,
                {"generator": generator(f"{GEOM}.mixed_voronoi_area", levels=[r["level"] for r in sphere]),
                 "checks": [check("finest valence-5 Voronoi error", voronoi[-1], 1e-3, "le", kind="self_convergence"),
                            check("coarsest minus finest Voronoi valence-5 error", voronoi[0] - voronoi[-1], 0.0,
                                  "signed_ge", kind="self_convergence")]},
                uncertainty=_u("roundoff", 1e-13, "angle sums and areas in double precision"), tolerance=TIGHT),
        finding("Barycentric angle-defect curvature does not converge pointwise at valence-6 icosphere vertices on "
                "the icosahedral mirror planes", "numerical", seam,
                {"generator": generator(f"{STUD}.icosahedral_mirror_classes", levels=seam["levels"]),
                 "checks": [check("smallest base-edge valence-6 max error over levels 4-7",
                                  min(seam["base_edge_max_error"]), 1.5e-3, "ge", kind="self_convergence"),
                            check("finest minus level-4 base-edge max error", seam["base_edge_max_error"][-1]
                                  - seam["base_edge_max_error"][0], 0.0, "signed_ge", kind="self_convergence"),
                            check("finest mixed-Voronoi max error on the same mirror-plane vertices",
                                  seam["voronoi_mirror_max_error"][-1], 1e-4, "le", kind="self_convergence")]},
                uncertainty=_u("roundoff", 1e-13, "angle sums and areas in double precision"), tolerance=TIGHT,
                counterexample={"statement": "Angle-defect curvature with the barycentric area converges pointwise "
                                             "at the valence-6 vertices of icosphere refinements",
                                "witness": {"levels": seam["levels"],
                                            "base_edge_max_error": seam["base_edge_max_error"],
                                            "median_max_error": seam["median_max_error"]}}),
        finding("Angle-defect curvature on regular torus grids converges at second order to the sign-changing K",
                "numerical", {"order": o_torus["fit"], "local_orders": o_torus["local"],
                              "max_error": [r["max_error"] for r in trows]},
                {"generator": generator(f"{GEOM}.torus_mesh", n_theta=[r["n_theta"] for r in trows]),
                 "checks": [check("fitted order - 2", o_torus["fit"] - 2.0, 0.2, kind="self_convergence")]},
                uncertainty=_order_uncertainty(o_torus), tolerance=RATE),
        _physical("Angle-defect curvature of a scanned mesh estimates the Gaussian curvature of the physical part"),
    ]
    wide_errors = ", ".join(format(r["mean_abs_error"], ".2e") for r in wide)
    result = (f"FD Jacobi (delta=0.1): mean error {wide_errors} (levels {wide[0]['level']}-{wide[-1]['level']}, "
              f"order {o_wide['fit']:.2f}, pairwise " + ", ".join(f"{p:.2f}" for p in o_wide["local"])
              + f"); delta={tiny_delta:g}: {identical}/{pairs} pairs on levels {tiny[0]['level']}-{tiny[-1]['level']} "
              f"share face sequences and give j = L = {study['flat_value']} (deviation {flat_dev:.1e}), error "
              f"{flat_error:.4f}; at level {crossover['level']} (about {crossover['swept_vertices']:.2f} vertices "
              f"expected in each pair's wedge) {crossover['pairs'] - crossover['identical_face_sequences']}/"
              f"{crossover['pairs']} pairs straddle a vertex. Valence-5 angle-defect curvature {k5[-1]:.7f} vs "
              f"predicted limit {limit:.7f} (order {o_k5['fit']:.2f}); Voronoi valence-5 error {voronoi[-1]:.1e}. "
              f"Valence-6 max error on base-edge arcs " + ", ".join(f"{v:.2e}" for v in seam["base_edge_max_error"])
              + ", on face medians " + ", ".join(f"{v:.2e}" for v in seam["median_max_error"])
              + ", off the mirror planes " + ", ".join(f"{v:.2e}" for v in seam["off_mirror_max_error"])
              + f" (levels 4-7); torus order {o_torus['fit']:.2f}. Implementation sanity check (not a finding: "
              f"discrete Gauss-Bonnet holds for any closed mesh whose triangle angles sum to pi): residual {gb:.1e}.")
    return {"state": "completed", "findings": findings, "fields": fields(
        "Mesh Jacobi fields and angle-defect curvature approximate the smooth ones only in the right order of limits "
        "and at vertices whose stars become regular.",
        "Smooth: j'' + K j = 0, j_head(s) = sin(s) for K = 1. Mesh: curvature is concentrated at vertices (cone "
        "points); two geodesics are rotated relative to each other only when a vertex lies between them, so the "
        "finite-difference field depends on delta/h. For vertices on a sphere the angle defect tends to K times the "
        "circumcentric (Voronoi) cell area, so defect/(A/3) tends to K Voronoi/(A/3): 3K / (4 cos^2(pi/n)) at "
        "regular valence-n stars, and a value other than K wherever the star stays irregular.",
        ["icosphere levels 1-7", "torus grids n_theta=8..64 (R=2, r=1)", "six declared geodesics, L=2",
         "heading offsets 0.1, 0.03, 0.01, 1e-5 rad", "icosahedral mirror planes (15 great circles)"],
        "Central difference |X+(L) - X-(L)| / (2 sin delta), exact for the smooth unit sphere; vertex curvature "
        "estimates against K = 1 or the torus closed form.",
        "Delta fixed, h -> 0: convergence; h fixed, delta -> 0: the flat value L. Valence-5 limit 4.5 - 1.5 sqrt 5.",
        "Paired traces at +-delta on each level; angle defect with barycentric and mixed Voronoi areas; valence-6 "
        "errors split by the icosahedral mirror planes; torus grids.",
        result,
        "Deterministic. At intermediate delta the FD error depends on how many vertices lie between the paired "
        "geodesics (a pair that straddles a vertex jumps), so intermediate values are reported without a claim; only "
        "the two limits are asserted. The expected wedge vertex count is a heuristic for evenly spread vertices.",
        ["pairs straddling a vertex (jumps)", "identical face sequences (flat regime)", "valence-5 versus valence-6 "
         "vertices", "Voronoi versus barycentric area", "mirror-plane versus generic valence-6 vertices",
         "Gauss-Bonnet sum (implementation sanity check only; an identity for every closed mesh)"],
        ["The valence-6 plateau (about 2.6e-3) lives on the icosahedral mirror planes (base-edge arcs and base-face "
         "medians). The mixed Voronoi area converges on the same vertices, so their stars stay irregular (Voronoi "
         "cell different from A/3) under refinement; why is not derived (a kink of the recursive midpoint map across "
         "base edges is a plausible cause there, and no mechanism is given for the medians), and no closed form for "
         "the plateau is given. The off-mirror maximum stops decreasing near 4.5e-4 at level 7, so pointwise "
         "convergence off the planes is not established either.",
         "The torus grid is regular; irregular torus meshes are not tested here."],
        NEXT_STEPS["T040"])}


# ---------------------------------------------------------------- T041
def _average_ranks(values) -> np.ndarray:
    """Ranks 1..n with ties (equal to 12 significant digits) sharing their average rank, as in scipy.stats.rankdata."""
    values = np.array([float(f"{v:.12g}") for v in values])
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x, y) -> float:
    return float(np.corrcoef(_average_ranks(x), _average_ranks(y))[0, 1])


def _permutation_p(x, y, permutations=2000, seed=S.SEED + 70) -> float:
    """One-sided permutation p-value for the sign of the observed Spearman coefficient."""
    rx, ry = _average_ranks(x), _average_ranks(y)
    observed = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.Generator(np.random.PCG64(seed))
    shuffled = np.array([np.corrcoef(rx, ry[rng.permutation(len(ry))])[0, 1] for _ in range(permutations)])
    extreme = np.sum(shuffled >= observed - 1e-12) if observed >= 0 else np.sum(shuffled <= observed + 1e-12)
    return float((extreme + 1) / (permutations + 1))


def discordant_pairs(x, y) -> int:
    """Pairs ordered oppositely by x and y (ties count as neither)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    i, j = np.triu_indices(len(x), 1)
    return int(np.sum(np.sign(x[i] - x[j]) * np.sign(y[i] - y[j]) < 0))


def _scipy_spearman(comparisons):
    """Largest |ciw - scipy.stats.spearmanr| over (samples, metric, error, value), or None without scipy."""
    if S.optional_version("scipy") is None:
        return None
    from scipy.stats import spearmanr
    worst = 0.0
    for samples, metric, error, value in comparisons:
        x = [float(f"{r[metric]:.12g}") for r in samples]
        y = [float(f"{r[error]:.12g}") for r in samples]
        worst = max(worst, abs(float(spearmanr(x, y).statistic) - value))
    return worst


def _fold_code(issues):
    """The dihedral fold code when present, else the first structural code (None when valid)."""
    return "folded_face" if "folded_face" in issues else (issues[0] if issues else None)


@task("T041", changed_files=FILES, regression_tests=(
    NEXT_STEP_TEST,
    f"{TESTS}::test_schwarz_lantern_counterexample",
    f"{TESTS}::test_rank_correlation_matches_average_ranks",
    f"{TESTS}::test_fold_check_misses_small_bend_inversions",
    f"{TESTS}::test_quality_task_report"))
def mesh_quality_effects(ctx):
    quality = _memo(ctx, "quality", S.quality_study)
    lantern = _memo(ctx, "lantern", S.lantern_study)
    plane = _memo(ctx, "plane", S.plane_study)
    folds = _memo(ctx, "fold-rates", S.fold_rate_study)
    jitter = quality["jitter"]
    valid = [r for r in jitter if not r["issues"]]
    amplitudes = sorted({r["amplitude"] for r in valid})
    averaged, spread = [], 0.0
    for a in amplitudes:
        group = [r for r in valid if r["amplitude"] == a]
        averaged.append({"amplitude": a, "meshes": len(group),
                         "curvature_rms": float(np.mean([r["curvature_rms"] for r in group])),
                         "voronoi_curvature_rms": float(np.mean([r["voronoi_curvature_rms"] for r in group])),
                         "min_angle_deg": float(np.mean([r["min_angle_deg"] for r in group])),
                         "max_radius_ratio": float(np.mean([r["max_radius_ratio"] for r in group])),
                         "geodesic_mean": float(np.mean([r["geodesic_mean"] for r in group]))})
        if len(group) > 1:  # Student t (2 dof) interval of a three-seed mean
            spread = max(spread, 4.303 * float(np.std([r["curvature_rms"] for r in group], ddof=1)) / math.sqrt(3))
    rms_steps = [b["curvature_rms"] - a["curvature_rms"] for a, b in zip(averaged, averaged[1:])]
    angle_steps = [a["min_angle_deg"] - b["min_angle_deg"] for a, b in zip(averaged, averaged[1:])]
    # Declared seeds: the dihedral fold check (no centre declared) against inverted faces, mesh by mesh.
    fold_checks = [refusal(f"structural and dihedral checks on {r['mesh']} ({r['inverted_faces']} inverted faces)",
                           "folded_face" if r["inverted_faces"] > 0 else "none", _fold_code(r["structural_issues"]))
                   for r in jitter]
    frows = {r["amplitude"]: r for r in folds["rows"]}
    further = len(folds["seeds"])
    inverted_accepted = sum(r["inverted_accepted"] for r in folds["rows"])
    clean_refused = sum(r["clean_refused"] for r in folds["rows"])
    fold_meshes = sum(r["meshes"] for r in folds["rows"])
    at_02 = frows[0.2]
    inverted_02 = at_02["inverted_refused"] + at_02["inverted_accepted"]
    fold_witness = folds["witness"]
    fold_generator = generator(f"{STUD}.fold_rate_study", level=folds["level"], amplitudes=list(frows),
                               seeds=[folds["seeds"][0], folds["seeds"][-1]], per_amplitude=further)
    fold_table = [{k: r[k] for k in ("amplitude", "meshes", "inverted_refused", "inverted_accepted", "clean_refused",
                                     "clean_accepted")} for r in folds["rows"]]
    regular = next(r for r in jitter if r["amplitude"] == 0)
    better = [r for r in valid
              if r["geodesic_mean"] < regular["geodesic_mean"] and r["min_angle_deg"] < regular["min_angle_deg"]]
    witness = min(better, key=lambda r: r["geodesic_mean"]) if better else None
    uv = [r for r in quality["latitude_longitude"] if not r["issues"]]
    uv_better = [r for r in uv if r["curvature_rms"] < regular["curvature_rms"]
                 and r["min_angle_deg"] < regular["min_angle_deg"]]
    uv_witness = min(uv_better, key=lambda r: r["curvature_rms"]) if uv_better else None
    # Within-family anisotropy: a pair with better quality on both metrics but larger error on both estimators.
    uv_pairs = [(a, b) for a in uv for b in uv
                if a["min_angle_deg"] > b["min_angle_deg"] and a["max_radius_ratio"] < b["max_radius_ratio"]
                and a["curvature_rms"] > b["curvature_rms"] and a["voronoi_curvature_rms"] > b["voronoi_curvature_rms"]]
    uv_pair = max(uv_pairs, key=lambda p: p[0]["min_angle_deg"] - p[1]["min_angle_deg"]) if uv_pairs else None
    compared = ("mesh", "min_angle_deg", "max_radius_ratio", "curvature_rms", "voronoi_curvature_rms", "curvature_max")
    samples = valid + uv
    metrics, errors = ("min_angle_deg", "max_radius_ratio"), ("curvature_rms", "voronoi_curvature_rms",
                                                              "curvature_max", "geodesic_mean", "area_error")
    table = {m: {e: spearman([r[m] for r in samples], [r[e] for r in samples]) for e in errors} for m in metrics}
    ranked = ("voronoi_curvature_rms", "geodesic_mean")
    p_values = {e: _permutation_p([r["max_radius_ratio"] for r in samples], [r[e] for r in samples]) for e in ranked}
    families = {"latitude_longitude": uv, "jitter": valid}
    within = {f: {e: spearman([r["max_radius_ratio"] for r in rs], [r[e] for r in rs]) for e in ranked}
              for f, rs in families.items()}
    discordant = {e: discordant_pairs([r["max_radius_ratio"] for r in samples], [r[e] for r in samples])
                  for e in ranked}
    scipy_gap = _scipy_spearman([(samples, m, e, v) for m, row in table.items() for e, v in row.items()]
                                + [(families[f], "max_radius_ratio", e, v) for f, row in within.items()
                                   for e, v in row.items()])
    lrows = lantern["rows"]
    last = lrows[-1]
    hausdorff_gap = max(abs(r["hausdorff_sampled"] - r["hausdorff_closed_form"]) for r in lrows)
    area_gap = max(abs(r["area_closed_form_error"]) for r in lrows)
    height_gap = max(abs(r["traced_height"] - r["height_closed_form"]) for r in lrows)
    flat = max(r["max_interior_curvature"] for r in lrows)
    growth = S.fitted_order([r["n"] for r in lrows[-3:]], [r["total_abs_mean_curvature"] for r in lrows[-3:]])
    tilt_gap = last["max_normal_tilt_deg"] - lantern["limit_tilt_deg"]
    plane_error = max(r["max_trace_error"] for r in plane["rows"])
    graph_excess = [r["max_graph_excess"] for r in plane["rows"]]
    ctx.artifact_json("quality.json", jsonable({"quality": quality, "seed_averaged": averaged,
                                                "spearman": table, "spearman_within_family": within,
                                                "discordant_pairs": discordant,
                                                "permutation_p_radius_ratio": p_values,
                                                "scipy_max_abs_difference": scipy_gap, "plane": plane,
                                                "fold_rates": folds}))
    ctx.artifact_json("schwarz-lantern.json", jsonable(lantern))
    ctx.artifact_text("schwarz-lantern.svg", svg.line_plot([
        ("area / (2 pi R H)", [r["n"] for r in lrows], [r["area_ratio"] for r in lrows]),
        ("traced height / H", [r["n"] for r in lrows], [r["traced_height"] for r in lrows]),
        ("Hausdorff distance", [r["n"] for r in lrows], [r["hausdorff_sampled"] for r in lrows]),
        ("|H| total / pi", [r["n"] for r in lrows], [r["total_abs_mean_curvature"] / math.pi for r in lrows])],
        title=f"Schwarz lantern, m = {lantern['q']:g} n^2", xlabel="vertices per ring n", ylabel="value",
        logx=True, logy=True))
    ctx.artifact_text("jitter.svg", svg.line_plot([
        ("curvature RMS", [a["amplitude"] for a in averaged], [a["curvature_rms"] for a in averaged]),
        ("Voronoi RMS", [a["amplitude"] for a in averaged], [a["voronoi_curvature_rms"] for a in averaged]),
        ("geodesic error", [a["amplitude"] for a in averaged], [a["geodesic_mean"] for a in averaged]),
        ("min angle / 1000", [a["amplitude"] for a in averaged], [a["min_angle_deg"] / 1000 for a in averaged])],
        title="Tangential jitter at fixed vertex count (642)", xlabel="jitter amplitude / h", ylabel="value"))
    deterministic = _u("roundoff", 1e-15, "deterministic mesh metrics and traces")
    findings = [
        finding("Seed-averaged curvature RMS error grows and minimum angle falls with tangential jitter", "numerical",
                averaged, {"generator": generator(f"{STUD}.quality_study", level=3, jitter=amplitudes, seeds=[1, 2, 3],
                                                  seed=S.SEED),
                           "checks": [check("smallest increase of curvature RMS between amplitudes", min(rms_steps),
                                            0.0, "signed_ge", kind="self_convergence"),
                                      check("smallest decrease of min angle between amplitudes", min(angle_steps), 0.0,
                                            "signed_ge", kind="self_convergence")]},
                uncertainty=_u("monte_carlo_95ci", spread, "largest t-interval half-width (2 dof) of a three-seed "
                               "mean curvature RMS"),
                tolerance=TIGHT),
        finding("For the three declared seeds per amplitude, the dihedral fold check refuses exactly the jittered "
                "meshes that have an inverted face", "numerical",
                {"meshes": [r["mesh"] for r in jitter], "inverted_faces": [r["inverted_faces"] for r in jitter],
                 "structural_issues": [r["structural_issues"][:1] for r in jitter]},
                {"generator": generator(f"{STUD}.quality_study", level=3,
                                        jitter=sorted({r["amplitude"] for r in jitter}), seeds=[1, 2, 3], seed=S.SEED),
                 "checks": fold_checks},
                uncertainty=EXACT, tolerance=EXACT_TOLERANCE),
    ]
    if fold_witness is None:
        findings.append(_missing_witness(
            f"Over {further} further seeds per amplitude the dihedral fold check accepts some jittered meshes with an "
            "inverted face", "jittered meshes with an inverted face accepted by the dihedral fold check"))
    else:
        findings.append(finding(
            f"Over {further} further seeds per amplitude the dihedral fold check accepts some jittered meshes with an "
            "inverted face", "numerical", {"rates": fold_table, "witness": fold_witness},
            {"generator": fold_generator,
             "checks": [check("jittered meshes with an inverted face that pass the structural and dihedral checks",
                              inverted_accepted, 1.0, "ge", kind="exact_arithmetic"),
                        check("witness inverted face: unit normal . radial direction at its centroid",
                              fold_witness["normal_radial"], 0.0, "signed_le", kind="exact_arithmetic"),
                        check(f"witness mesh: smallest adjacent unit-normal dot minus the fold threshold "
                              f"{S.G.FOLD_COSINE}", fold_witness["min_adjacent_normal_dot"] - S.G.FOLD_COSINE, 0.0,
                              "signed_ge", kind="exact_arithmetic")]},
            uncertainty=_binomial(at_02["inverted_accepted"] / at_02["meshes"], at_02["meshes"]), tolerance=TIGHT,
            counterexample={"statement": "The dihedral fold check (folded_face) refuses every jittered icosphere "
                                         "that has an inverted face",
                            "witness": {k: fold_witness[k] for k in ("amplitude", "seed", "face", "normal_radial",
                                                                     "corner_angles_deg", "neighbour_normal_dots")}}))
    findings += [
        finding(f"A tangential jitter of 0.2 h inverts a face for some but not all of {further} further seeds",
                "numerical", {"inverted_fraction": {str(a): (r["inverted_refused"] + r["inverted_accepted"])
                                                    / r["meshes"] for a, r in frows.items()}},
                {"generator": fold_generator,
                 "checks": [check("meshes jittered by 0.2 h without an inverted face", at_02["meshes"] - inverted_02,
                                  1.0, "ge", kind="exact_arithmetic"),
                            check("meshes jittered by 0.2 h with an inverted face", inverted_02, 1.0, "ge",
                                  kind="exact_arithmetic")]},
                uncertainty=_binomial(inverted_02 / at_02["meshes"], at_02["meshes"]), tolerance=TIGHT,
                counterexample={"statement": "Tangential jitter of 0.2 h or more inverts a face of the level-3 "
                                             "icosphere for every seed",
                                "witness": {"amplitude": 0.2, "meshes": at_02["meshes"],
                                            "without_inverted_face": at_02["meshes"] - inverted_02}}),
        finding(f"Over {further} further seeds per amplitude the dihedral fold check refuses no jittered mesh "
                "without an inverted face", "numerical", {"rates": fold_table},
                {"generator": fold_generator,
                 "checks": [check("jittered meshes without an inverted face refused by the structural and dihedral "
                                  "checks", clean_refused, 0.0, "le", kind="exact_arithmetic")]},
                uncertainty=_binomial(clean_refused / fold_meshes, fold_meshes), tolerance=EXACT_TOLERANCE),
    ]
    if witness is None:
        findings.append(_missing_witness("A mesh with a smaller minimum angle can have a smaller geodesic error",
                                         "valid jittered meshes with smaller min angle and smaller geodesic error"))
    else:
        findings.append(finding(
            "A mesh with a smaller minimum angle can have a smaller geodesic error", "numerical",
            {"regular": {k: regular[k] for k in ("mesh", "min_angle_deg", "geodesic_mean")},
             "witness": {k: witness[k] for k in ("mesh", "min_angle_deg", "geodesic_mean")}},
            {"generator": generator(f"{STUD}._jitter_unvalidated", level=3, jitter=witness["amplitude"],
                                    seed=S.SEED + witness["seed"]),
             "checks": [check("regular minus witness geodesic error",
                              regular["geodesic_mean"] - witness["geodesic_mean"], 0.0, "signed_ge",
                              kind="self_convergence"),
                        check("regular minus witness min angle", regular["min_angle_deg"] - witness["min_angle_deg"],
                              0.0, "signed_ge", kind="self_convergence")]},
            uncertainty=deterministic, tolerance=TIGHT,
            counterexample={"statement": "A larger minimum angle implies a smaller geodesic error",
                            "witness": {"better_quality": regular["mesh"], "worse_quality": witness["mesh"]}}))
    cross_claim = ("Across mesh families a much smaller minimum angle can come with a smaller barycentric-area "
                   "angle-defect RMS error")
    if uv_witness is None:
        findings.append(_missing_witness(cross_claim, "latitude-longitude spheres with smaller min angle and RMS"))
    else:
        findings.append(finding(
            cross_claim, "numerical", {"regular": {k: regular[k] for k in compared},
                                       "witness": {k: uv_witness[k] for k in compared}},
            {"generator": generator(f"{GEOM}.uv_sphere", n_lat=uv_witness["n_lat"], n_lon=uv_witness["n_lon"],
                                    twist=uv_witness["twist"]),
             "checks": [check("regular minus witness barycentric curvature RMS",
                              regular["curvature_rms"] - uv_witness["curvature_rms"], 0.0, "signed_ge",
                              kind="self_convergence"),
                        check("regular minus witness min angle (deg)", regular["min_angle_deg"]
                              - uv_witness["min_angle_deg"], 20.0, "signed_ge", kind="self_convergence")]},
            uncertainty=deterministic, tolerance=TIGHT,
            counterexample={"statement": "Minimum angle orders the barycentric angle-defect curvature error across "
                                         "mesh families",
                            "witness": {"better_quality": regular["mesh"], "worse_quality": uv_witness["mesh"],
                                        "note": "the icosphere's RMS is dominated by its 12 valence-5 vertices "
                                                "(T040); with the mixed Voronoi area the icosphere is better"}}))
    uv_best_voronoi = min(r["voronoi_curvature_rms"] for r in uv)
    findings.append(finding(
        "With the mixed Voronoi area the regular icosphere has a smaller curvature RMS error than every "
        "latitude-longitude sphere of the same vertex count", "numerical",
        {"icosphere": regular["voronoi_curvature_rms"],
         "latitude_longitude": {r["mesh"]: r["voronoi_curvature_rms"] for r in uv}},
        {"generator": generator(f"{GEOM}.mixed_voronoi_area", meshes=[regular["mesh"]] + [r["mesh"] for r in uv]),
         "checks": [check("smallest latitude-longitude Voronoi RMS minus the icosphere's",
                          uv_best_voronoi - regular["voronoi_curvature_rms"], 0.0, "signed_ge",
                          kind="self_convergence")]},
        uncertainty=deterministic, tolerance=TIGHT))
    within_claim = ("Within the latitude-longitude family a mesh with smaller minimum angle and larger radius ratio "
                    "can have smaller curvature errors under both area choices")
    if uv_pair is None:
        findings.append(_missing_witness(within_claim, "latitude-longitude pairs ordered against both quality metrics"))
    else:
        good, poor = uv_pair
        findings.append(finding(
            within_claim, "numerical", {"better_quality": {k: good[k] for k in compared},
                                        "worse_quality": {k: poor[k] for k in compared}},
            {"generator": generator(f"{GEOM}.uv_sphere", meshes=[good["mesh"], poor["mesh"]]),
             "checks": [check("min angle difference (better - worse quality)", good["min_angle_deg"]
                              - poor["min_angle_deg"], 0.0, "signed_ge", kind="self_convergence"),
                        check("radius ratio difference (worse - better quality)", poor["max_radius_ratio"]
                              - good["max_radius_ratio"], 0.0, "signed_ge", kind="self_convergence"),
                        check("barycentric RMS difference (better - worse quality)", good["curvature_rms"]
                              - poor["curvature_rms"], 0.0, "signed_ge", kind="self_convergence"),
                        check("Voronoi RMS difference (better - worse quality)", good["voronoi_curvature_rms"]
                              - poor["voronoi_curvature_rms"], 0.0, "signed_ge", kind="self_convergence")]},
            uncertainty=deterministic, tolerance=TIGHT,
            counterexample={"statement": "Within one mesh family better triangle quality (larger minimum angle, "
                                         "smaller radius ratio) implies smaller curvature error",
                            "witness": {"better_quality": good["mesh"], "worse_quality": poor["mesh"]}}))
    uv_within = within["latitude_longitude"]
    rank_basis = {"generator": generator(f"{STUD}.quality_study", meshes=len(samples), seed=S.SEED),
                  "checks": [check("Spearman rho of max radius ratio with Voronoi curvature RMS (pooled meshes)",
                                   table["max_radius_ratio"]["voronoi_curvature_rms"], 0.6, "signed_ge",
                                   kind="self_convergence"),
                             check("Spearman rho of max radius ratio with geodesic error (pooled meshes)",
                                   table["max_radius_ratio"]["geodesic_mean"], 0.6, "signed_ge",
                                   kind="self_convergence"),
                             check("largest one-sided permutation p-value (2000 seeded permutations; assumes the "
                                   "pooled meshes are exchangeable, although the jitter meshes are three noise fields "
                                   "scaled to each amplitude)", max(p_values.values()), 0.01, "le",
                                   kind="self_convergence"),
                             check("largest Spearman rho of max radius ratio with either error within the "
                                   "latitude-longitude family (1e-12 rounding allowance)",
                                   max(uv_within.values()), 1e-12, "signed_le", kind="self_convergence")]}
    if scipy_gap is not None:
        rank_basis["independent_check"] = dict(
            check("scipy.stats.spearmanr on the same rounded inputs, largest difference over the pooled table and "
                  "the within-family coefficients", scipy_gap, 1e-12, kind="exact_arithmetic"),
            producer={"implementation": PRODUCER, "revision": "working tree"},
            checker={"implementation": "scipy.stats.spearmanr", "revision": S.optional_version("scipy")})
    findings.append(finding(
        "Maximum radius ratio is positively rank-correlated with the mixed-Voronoi curvature error and the geodesic "
        "error across the pooled valid 642-vertex meshes, but not within the latitude-longitude family", "numerical",
        {"spearman": table, "spearman_within_family": within, "discordant_pairs": discordant,
         "pairs": len(samples) * (len(samples) - 1) // 2, "permutation_p": p_values, "meshes": len(samples)},
        rank_basis, uncertainty=_u("reference_error", 1.06 / math.sqrt(len(samples) - 3),
                                   "approximate standard error of a Spearman coefficient from n exchangeable meshes"),
        tolerance=TIGHT))
    findings += [
        finding("Schwarz lantern meshes converge in Hausdorff distance but not in area", "numerical",
                {"n": [r["n"] for r in lrows], "hausdorff": [r["hausdorff_sampled"] for r in lrows],
                 "area_ratio": [r["area_ratio"] for r in lrows], "limit_area_ratio": lantern["limit_area_ratio"]},
                {"derivation": "A = 2 n R sin(pi/n) sqrt(H^2 + m^2 R^2 (1 - cos(pi/n))^2). Hausdorff distance "
                               "d_H = R (1 - cos(pi/n)) in both directions: every face lies between radius "
                               "R cos(pi/n) and R (sampled, attained at horizontal edge midpoints), and every "
                               "horizontal ray from the axis crosses the closed mesh tube at such a radius",
                 "checks": [check("max |sampled mesh-to-cylinder distance - closed form|", hausdorff_gap, 1e-12,
                                  kind="analytic"),
                            check("max |mesh area - closed form|", area_gap, 1e-9, kind="analytic"),
                            check("area ratio at the finest n - 1", last["area_ratio"] - 1.0, 0.5, "signed_ge",
                                  kind="analytic")]},
                uncertainty=_u("roundoff", max(hausdorff_gap, area_gap), "difference from the closed forms"),
                tolerance=TIGHT,
                counterexample={"statement": "Hausdorff convergence of a mesh to a surface implies convergence of area",
                                "witness": {"q": lantern["q"], "n": last["n"], "bands": last["bands"],
                                            "hausdorff": last["hausdorff_sampled"], "area_ratio": last["area_ratio"]}}),
        finding("Schwarz lantern intrinsic height does not converge to the cylinder height", "numerical",
                {"traced_height": [r["traced_height"] for r in lrows], "height": 1.0},
                {"derivation": "The lantern develops to a flat strip of height sqrt(H^2 + m^2 R^2 (1 - cos(pi/n))^2)",
                 "checks": [check("max |traced height - development height|", height_gap, 1e-10, kind="analytic"),
                            check("traced height at the finest n - 1.5 H", last["traced_height"] - 1.5, 0.0,
                                  "signed_ge", kind="analytic")]},
                uncertainty=_u("roundoff", height_gap, "difference from the development"), tolerance=TIGHT,
                counterexample={"statement": "Hausdorff convergence of a mesh implies convergence of its geodesic "
                                             "distances",
                                "witness": {"n": last["n"], "traced_height": last["traced_height"], "H": 1.0}}),
        finding("Lantern angle defects vanish while total absolute mean curvature grows like n^2 and normal tilt "
                "converges to atan(pi^2 q R / 2H) instead of 0", "numerical",
                {"max_interior_angle_defect_curvature": flat,
                 "total_abs_mean_curvature": [r["total_abs_mean_curvature"] for r in lrows],
                 "growth_exponent_in_n": growth,
                 "smooth_total_abs_mean_curvature": lantern["smooth_total_abs_mean_curvature"],
                 "max_normal_tilt_deg": [r["max_normal_tilt_deg"] for r in lrows],
                 "limit_tilt_deg": lantern["limit_tilt_deg"]},
                {"derivation": "Isosceles lantern faces give angle sums 2 alpha + 4 beta = 2 pi at interior vertices; "
                               "a face rises H/m while its apex sags R (1 - cos(pi/n)), so its normal tilts by "
                               "atan(m R (1 - cos(pi/n)) / H) -> atan(pi^2 q R / 2H); about q n^3 edges of length "
                               "~2 pi R / n bend by a fixed angle, so total |H| ~ n^2",
                 "checks": [check("max interior |K| (angle defect)", flat, 1e-9, kind="analytic"),
                            check("max normal tilt at the finest n - limit (deg)", tilt_gap, 0.1, kind="analytic"),
                            check("growth exponent of total |H| in n (last three n) - 2", growth - 2.0, 0.15,
                                  kind="analytic"),
                            check("total |H| at finest n / smooth value", last["total_abs_mean_curvature"]
                                  / lantern["smooth_total_abs_mean_curvature"], 5.0, "ge", kind="analytic")]},
                uncertainty=_u("truncation_bound", abs(tilt_gap), "remaining approach of the tilt to its limit (deg)"),
                tolerance=TIGHT,
                counterexample={"statement": "Hausdorff convergence of a mesh implies convergence of its normals and "
                                             "curvature",
                                "witness": {"n": last["n"], "tilt_deg": last["max_normal_tilt_deg"],
                                            "total_abs_mean_curvature": last["total_abs_mean_curvature"]}}),
        finding("A strongly pleated lantern (m = n^2) is refused as folded although no face normal points towards "
                "the axis", "numerical", lantern["folded"],
                {"generator": generator(f"{GEOM}.cylinder_mesh", lantern=True, n=lantern["folded"]["n"],
                                        q=lantern["folded"]["q"]),
                 "checks": [refusal("validator on the m = n^2 lantern", "folded_face",
                                    lantern["folded"]["issues"][0] if lantern["folded"]["issues"] else None),
                            check("smallest unit face normal . outward radial direction", lantern["folded"]
                                  ["min_normal_radial"], 0.0, "signed_ge", kind="exact_arithmetic")]},
                uncertainty=EXACT, tolerance=TIGHT,
                counterexample={"statement": "The dihedral fold check refuses a mesh only when a face is inverted",
                                "witness": {"q": lantern["folded"]["q"], "n": lantern["folded"]["n"],
                                            "min_normal_radial": lantern["folded"]["min_normal_radial"]}}),
        finding("Straightest geodesics on planar meshes are exact at any tested triangle quality, while edge-graph "
                "distance error changes with the edge directions",
                "numerical", {"max_trace_error": plane_error, "max_graph_excess": graph_excess,
                              "min_angle_deg": [r["min_angle_deg"] for r in plane["rows"]]},
                {"generator": generator(f"{GEOM}.plane_mesh", shears=[r["shear"] for r in plane["rows"]]),
                 "checks": [check("max trace error over all shears", plane_error, 1e-12, kind="analytic"),
                            check("spread of max relative graph excess across shears",
                                  max(graph_excess) - min(graph_excess), 0.01, "ge", kind="self_convergence")]},
                uncertainty=_u("roundoff", plane_error, "largest trace deviation observed"),
                tolerance={"abs": 1e-12, "rel": 1e-6}),
        _physical("A minimum-angle or radius-ratio threshold certifies a scanned mesh for production metrology",
                  "production_acceptance"),
    ]
    refused = [r for r in jitter if r["issues"]]
    rr = table["max_radius_ratio"]
    parts = ["Jitter (seed-averaged, valid meshes): curvature RMS "
             + ", ".join(f"{a['curvature_rms']:.4f}@{a['amplitude']:g}" for a in averaged)
             + f"; for the declared seeds {len(refused)} of {len(jitter)} jittered meshes are refused, and the "
             "dihedral fold refusals coincide with inverted faces mesh by mesh. Over "
             f"{further} further seeds per amplitude (inverted and accepted by the dihedral check / inverted and "
             "refused / no inverted face): "
             + ", ".join(f"{r['inverted_accepted']}/{r['inverted_refused']}/{r['clean_refused'] + r['clean_accepted']}"
                         f"@{r['amplitude']:g}" for r in folds["rows"])
             + f"; {clean_refused} meshes without an inverted face are refused."]
    if fold_witness is not None:
        parts.append(f"Inversion missed by the dihedral check: seed {fold_witness['seed']} at "
                     f"{fold_witness['amplitude']:g} h, face {fold_witness['face']} (normal . radial "
                     f"{fold_witness['normal_radial']:.2f}, corner angles "
                     + ", ".join(f"{a:.1f}" for a in fold_witness["corner_angles_deg"])
                     + f" deg, smallest adjacent normal dot {fold_witness['min_adjacent_normal_dot']:.2f} > "
                     f"{S.G.FOLD_COSINE:g}), unguarded curvature RMS "
                     f"{fold_witness['unguarded_curvature_rms']:.2f}; with the sphere centre declared it is refused "
                     "as inverted_face.")
    if witness is not None:
        parts.append(f"Min angle vs geodesic error counterexample: {regular['mesh']} ({regular['min_angle_deg']:.1f} "
                     f"deg, {regular['geodesic_mean']:.4f}) vs {witness['mesh']} ({witness['min_angle_deg']:.1f} deg, "
                     f"{witness['geodesic_mean']:.4f}).")
    if uv_witness is not None:
        parts.append(f"{uv_witness['mesh']} has min angle {uv_witness['min_angle_deg']:.1f} deg and barycentric "
                     f"curvature RMS {uv_witness['curvature_rms']:.4f} < {regular['curvature_rms']:.4f}, but mixed "
                     f"Voronoi RMS {uv_witness['voronoi_curvature_rms']:.4f} > {regular['voronoi_curvature_rms']:.4f}.")
    if uv_pair is not None:
        parts.append(f"Within the latitude-longitude family {uv_pair[0]['mesh']} (min angle "
                     f"{uv_pair[0]['min_angle_deg']:.1f} deg, radius ratio {uv_pair[0]['max_radius_ratio']:.2f}) has "
                     f"larger errors than {uv_pair[1]['mesh']} ({uv_pair[1]['min_angle_deg']:.1f} deg, "
                     f"{uv_pair[1]['max_radius_ratio']:.2f}).")
    parts.append(f"Spearman over {len(samples)} pooled valid meshes: radius ratio vs Voronoi RMS "
                 f"{rr['voronoi_curvature_rms']:.3f}, vs geodesic error {rr['geodesic_mean']:.3f} (discordant pairs "
                 f"{discordant['voronoi_curvature_rms']} and {discordant['geodesic_mean']} of "
                 f"{len(samples) * (len(samples) - 1) // 2}), vs barycentric RMS {rr['curvature_rms']:.3f}; within the "
                 f"latitude-longitude family {uv_within['voronoi_curvature_rms']:.3f} and "
                 f"{uv_within['geodesic_mean']:.3f}, within the jitter family "
                 f"{within['jitter']['voronoi_curvature_rms']:.3f} and {within['jitter']['geodesic_mean']:.3f}; min "
                 f"angle vs barycentric RMS {table['min_angle_deg']['curvature_rms']:.3f}, vs Voronoi RMS "
                 f"{table['min_angle_deg']['voronoi_curvature_rms']:.3f}.")
    parts.append(f"Lantern (m={lantern['q']:g} n^2, n={last['n']}): Hausdorff {last['hausdorff_sampled']:.2e}, area "
                 f"ratio {last['area_ratio']:.4f} (limit {lantern['limit_area_ratio']:.4f}), traced height "
                 f"{last['traced_height']:.4f}, interior K {flat:.1e}, normal tilt "
                 f"{last['max_normal_tilt_deg']:.3f} deg "
                 f"(limit {lantern['limit_tilt_deg']:.3f}), total |H| exponent {growth:.2f}.")
    parts.append(f"Planar traces exact to {plane_error:.1e} for min angles down to "
                 f"{min(r['min_angle_deg'] for r in plane['rows']):.1f} deg, while the max edge-graph excess moves "
                 "from "
                 f"{graph_excess[0]:.3f} (shear 0) to {graph_excess[-1]:.3f} (shear {plane['rows'][-1]['shear']:g}).")
    return {"state": "completed", "findings": findings, "fields": fields(
        "Under isotropic tangential jitter of the icosphere at fixed vertex count, worse triangle quality comes with "
        "larger curvature error; within the latitude-longitude family and across families quality metrics do not "
        "order the errors pairwise (and the barycentric estimator's cross-family ranking reverses with the mixed "
        "Voronoi area); the maximum radius ratio is positively rank-correlated with the Voronoi-area curvature and "
        "geodesic errors over the pooled meshes, an association carried by the jitter family that vanishes within "
        "the latitude-longitude family; the dihedral fold check is neither necessary nor sufficient for an inverted "
        "face; Hausdorff convergence does not imply convergence of area, distances, normals or mean curvature.",
        "Quality: minimum angle and radius ratio R_circ / (2 r_in). Curvature: angle defect over the barycentric area "
        "A/3 or the mixed Voronoi area. Schwarz lantern with m = q n^2 bands: face height sqrt((H/m)^2 + R^2 (1 - "
        "cos(pi/n))^2), so area and developed height tend to sqrt(1 + (pi^2 q R / 2H)^2) times their cylinder values "
        "while d_H = R (1 - cos(pi/n)) -> 0.",
        ["icosphere level 3 (642 vertices), tangential jitter 0-0.3 h, 3 seeds (the same three noise fields scaled "
         "to each amplitude)", f"fold rates: jitter 0.1, 0.15, 0.2, 0.3 h with {further} further seeds each "
         f"({folds['seeds'][0]}-{folds['seeds'][-1]})", "latitude-longitude spheres with 642 "
         "vertices (20x32, 10x64, 40x16, 20x32 twisted by 0.08 and 0.16 rad per ring)",
         "Schwarz lanterns q=0.25, n=4..64; q=1, n=8", "sheared planar grids"],
        "Curvature error against K = 1, geodesic endpoint error against great circles, area error; lantern area, "
        "developed height, Hausdorff distance, normal tilt and total absolute mean curvature.",
        "Monotone trend within the jitter family; lantern closed forms for area, height, Hausdorff distance and "
        "tilt; angle defects of the lantern are exactly zero. No invariant links the dihedral fold check to inverted "
        "faces: it bounds the bend between neighbours (over about 154 degrees), so an inverted sliver with smaller "
        "bends passes and a crease without an inverted face (the q = 1 lantern) is refused; the rates are measured.",
        "Measure quality metrics and errors for each mesh (both area choices); compare the dihedral fold check with "
        "inverted faces (face normal towards the sphere centre) mesh by mesh for the declared seeds and as rates over "
        "further seeds; rank correlations, pooled and within each family, with a permutation test; trace lantern "
        "geodesics from the bottom boundary to the top boundary.",
        " ".join(parts),
        "Deterministic given the seeds; with three seeds per amplitude the monotone trend is a seed-average "
        "(t-interval reported per finding) and a different seed set could reorder neighbouring amplitudes; the three "
        "seeds are the same noise fields scaled, so amplitudes are not independent samples. Fold rates are counts "
        f"over {further} seeds per amplitude (binomial 95% half-widths per finding). Rank correlations over 15 "
        "meshes have a standard error near 0.3, and the permutation p-value treats the pooled meshes as "
        "exchangeable, which the replicated noise fields and the two families are not.",
        ["inverted faces from large jitter against the dihedral fold check, per mesh and as rates",
         "inverted faces missed by the dihedral check (recorded; refused as inverted_face when the centre is "
         "declared)", "degenerate triangles",
         "vertex hits on skewed meshes (none)", "pole valence on latitude-longitude spheres",
         "lantern pleat folding (refused at q = 1)", "tied ranks in the Spearman statistic (average ranks)",
         "estimator dependence of quality rankings (barycentric versus Voronoi area)"],
        ["Quality metrics are summarised by the worst triangle; per-region error attribution is not attempted.",
         "Only tangential jitter is studied here; normal noise is the subject of T043.",
         "The radius-ratio association is observed on 15 pooled meshes from two families and is carried by the "
         "jitter family (within the latitude-longitude family it is not positive); it is not a general law.",
         "Meshes that pass only the structural and dihedral checks may still contain inverted faces; the valid set "
         "here is validated with the sphere centre declared."],
        NEXT_STEPS["T041"])}


# ---------------------------------------------------------------- T042
@task("T042", changed_files=FILES, regression_tests=(
    NEXT_STEP_TEST,
    f"{TESTS}::test_every_defect_has_a_named_refusal",
    f"{TESTS}::test_fold_check_misses_small_bend_inversions",
    f"{TESTS}::test_refusal_task_report"))
def mesh_refusal_states(ctx):
    study = _memo(ctx, "refusals", S.refusal_study)
    cases = study["cases"]
    undetected = study["undetected_inversion"]
    control_issues = sum(len(c["issues"]) for c in study["controls"])
    expected_multiple = ["nonfinite_vertex", "inconsistent_orientation", "unreferenced_vertex"]
    partial_gap = abs(study["boundary_partial_length"] - study["boundary_analytic_length"])
    ctx.artifact_json("refusal-catalogue.json", jsonable(study))
    findings = [
        finding("Every declared surface-data defect is refused with its named code", "computational_pipeline",
                {c["case"]: c["observed"] for c in cases},
                {"generator": generator(f"{STUD}.refusal_study", cases=len(cases)),
                 "checks": [refusal(f"{c['stage']}: {c['case']}", c["expected"], c["observed"]) for c in cases]},
                uncertainty=EXACT, tolerance=EXACT_TOLERANCE),
        finding("Valid control meshes pass validation without issues", "computational_pipeline", control_issues,
                {"generator": generator(f"{STUD}.refusal_study", part="controls",
                                        meshes=[c["mesh"] for c in study["controls"]]),
                 "checks": [check("issues reported on valid meshes", control_issues, 0.0, "le",
                                  kind="exact_arithmetic")]},
                uncertainty=EXACT, tolerance=EXACT_TOLERANCE),
        finding("A mesh with several defects reports all of them in declared order", "computational_pipeline",
                study["multiple_defects"],
                {"generator": generator(f"{STUD}._octahedron", defects=expected_multiple),
                 "checks": [check("mismatches with the expected ordered codes",
                                  0.0 if study["multiple_defects"] == expected_multiple else 1.0, 0.0, "le",
                                  kind="exact_arithmetic")]},
                uncertainty=EXACT, tolerance=EXACT_TOLERANCE),
        finding("A geodesic stopped at a boundary retains its partial length", "numerical",
                study["boundary_partial_length"],
                {"generator": generator(f"{GEOM}.plane_mesh", nx=4, ny=4),
                 "checks": [check("|partial length - distance to the boundary|", partial_gap, 1e-12, kind="analytic")]},
                unit="normalized length", uncertainty=_u("roundoff", partial_gap, "difference from the closed form"),
                tolerance={"abs": 1e-12, "rel": 0.0}),
        finding("Curvature and normals evaluated on an unvalidated zero-area face are nonfinite", "numerical",
                {"nonfinite_curvatures": study["unguarded_nonfinite_curvatures"],
                 "nonfinite_normals": study["unguarded_nonfinite_normals"]},
                {"generator": generator(f"{GEOM}.TriMesh", validation="bypassed"),
                 "checks": [check("nonfinite curvature values", study["unguarded_nonfinite_curvatures"], 1.0, "ge",
                                  kind="exact_arithmetic"),
                            check("nonfinite normal components", study["unguarded_nonfinite_normals"], 1.0, "ge",
                                  kind="exact_arithmetic")]},
                uncertainty=EXACT, tolerance=EXACT_TOLERANCE,
                counterexample={"statement": "Curvature and normal formulas can be evaluated safely on unvalidated "
                                             "meshes",
                                "witness": {"mesh": "square plus a collinear face", "defect": "zero-area face"}}),
        finding("Without a declared centre the validator accepts a jittered icosphere with an inverted face",
                "computational_pipeline", undetected,
                {"generator": generator(f"{STUD}._jitter_unvalidated", level=3, jitter=undetected["amplitude"],
                                        seed=undetected["seed"]),
                 "checks": [check("issues reported by the structural and dihedral checks",
                                  len(undetected["structural_issues"]), 0.0, "le", kind="exact_arithmetic"),
                            check("faces whose normal points towards the sphere centre", undetected["inverted_faces"],
                                  1.0, "ge", kind="exact_arithmetic")]},
                uncertainty=EXACT, tolerance=EXACT_TOLERANCE,
                counterexample={"statement": "Structural validation, including the dihedral fold check, refuses "
                                            "every mesh with an inverted face",
                                "witness": {"amplitude": undetected["amplitude"], "seed": undetected["seed"]}}),
        _physical("The refusal catalogue covers every defect present in real scanned surface data"),
    ]
    codes = sorted({c["expected"] for c in cases})
    result = (f"{sum(c['observed'] == c['expected'] for c in cases)}/{len(cases)} defect cases refused with the "
              f"expected code ({len(codes)} distinct codes); {control_issues} issues on {len(study['controls'])} valid "
              f"controls; multiple-defect report {study['multiple_defects']}; boundary partial length "
              f"{study['boundary_partial_length']:.15g} (analytic {study['boundary_analytic_length']:.15g}); unguarded "
              f"zero-area evaluation gave {study['unguarded_nonfinite_curvatures']} nonfinite curvature values; the "
              f"jittered icosphere (seed {undetected['seed']}, {undetected['amplitude']:g} h) with "
              f"{undetected['inverted_faces']} inverted face passes the structural and dihedral checks and is refused "
              "as inverted_face only when the sphere centre is declared.")
    return {"state": "completed", "findings": findings, "fields": fields(
        "Each declared defect class in the mesh, trace and query catalogues (MESH_CODES, TRACE_CODES, QUERY_CODES) is "
        "detected before geometry is computed and reported under a stable name (inverted_face only for a mesh "
        "declared star-shaped about a centre), and valid meshes pass; defects outside the catalogue, including "
        "inverted faces of meshes without a declared centre whose bends stay below the fold threshold, are not "
        "claimed.",
        "Validation order: " + ", ".join(S.G.MESH_CODES) + ". Tracing refusals: " + ", ".join(S.G.TRACE_CODES)
        + ". Query refusals: " + ", ".join(S.G.QUERY_CODES) + ".",
        [f"{c['case']} ({c['stage']})" for c in cases],
        "No physical observation; the refusal code (or its absence) for each constructed input.",
        "Exactly the expected code for every defect; no code for valid controls; partial traces keep their length.",
        "Construct each defect explicitly, validate, trace or query, and compare the observed code with the declared "
        "code.",
        result,
        "Exact (categorical); thresholds are declared: zero area at 2A/l_max^2 <= 1e-12, vertex hit at edge "
        "parameter <= 1e-9, fold at adjacent normal dot <= -0.9, inverted face at det(p0 - c, p1 - c, p2 - c) <= 0 "
        "for a declared centre c.",
        codes + ["unguarded evaluation on zero-area faces"],
        ["Thresholds are relative and unitless; scans with very different scales may need other thresholds.",
         "Not detected: self-intersections between non-adjacent faces; unwelded seams (coincident duplicate vertices "
         "pass validation and surface only as boundary_reached during tracing); duplicate faces; near-degenerate "
         "slivers just above the 2A/l_max^2 = 1e-12 threshold; inverted faces whose bend to every neighbour stays "
         "below about 154 degrees when no centre is declared (a jittered icosphere is recorded; T041 measures how "
         "often this happens).",
         "The declared-centre test assumes the surface is star-shaped about that centre; it would refuse valid "
         "surfaces that are not (a torus about its own centre), so it is applied only on request.",
         "False-positive risk: a legitimate sharp crease with a dihedral bend over about 154 degrees is refused as "
         "folded_face (the embedded m = n^2 lantern is refused although it does not overlap itself and no face "
         "points inward)."],
        NEXT_STEPS["T042"])}


# ---------------------------------------------------------------- T043
@task("T043", changed_files=FILES, regression_tests=(
    NEXT_STEP_TEST,
    f"{TESTS}::test_linearization_matches_monte_carlo_and_breaks",
    f"{TESTS}::test_corridor_threshold_depends_on_the_strip",
    f"{TESTS}::test_per_vertex_uncertainty_matches_monte_carlo",
    f"{TESTS}::test_uncertainty_task_report",
    f"{TESTS}::test_t043_reconciles_the_two_declared_strip_estimates",
    f"{TESTS}::test_t043_says_when_the_two_declared_strip_estimates_differ"))
def mesh_vertex_uncertainty(ctx):
    study = _memo(ctx, "uncertainty", S.uncertainty_study)
    corridors = _memo(ctx, "corridors", S.corridor_study)
    scaling = _memo(ctx, "scaling", S.scaling_study)
    refinement = _memo(ctx, "refinement-noise", S.refinement_noise_study)
    field = _memo(ctx, "vertex-field", S.vertex_field_study)
    samples = study["samples"]
    items = {o["observable"]: o for o in study["observables"]}
    distance = items["marker geodesic distance"]
    normals = [o for name, o in items.items() if name.startswith("vertex normal")]
    curvature = [o for name, o in items.items() if name.startswith("angle-defect")]
    distance_dev = max(abs(r["ratio"] - 1) for r in distance["rows"] if r["sigma"] <= 1e-2)
    normal_dev = max(abs(r["ratio"] - 1) for o in normals for r in o["rows"])
    flips = sum(r["flipped_normals"] for o in normals for r in o["rows"])
    min_dot = min(r["min_normal_dot"] for o in normals for r in o["rows"])
    small = [r for o in curvature for r in o["rows"] if r["sigma"] <= 1e-3]
    large = [r for o in curvature for r in o["rows"] if r["sigma"] >= 1e-2]
    curvature_dev = max(abs(r["ratio"] - 1) for r in small)
    breakdown = min(r["ratio"] for r in large)
    invalid = {str(r["sigma"]): r["invalid_fraction"] for r in distance["rows"]}
    corridor_small = max(r["invalid_fraction"] for r in distance["rows"] if r["sigma"] <= 1e-3)
    corridor_large = min(r["invalid_fraction"] for r in distance["rows"] if r["sigma"] >= 1e-2)
    crow = corridors["rows"]
    at = corridors["sigmas"].index(1e-3)
    declared = next(r for r in crow if r["start_index"] == study["strip"]["start_index"])
    others = [r for r in crow if r is not declared]
    worst = max(crow, key=lambda r: r["left_fraction"][at])
    declared_small = max(f for sg, f in zip(corridors["sigmas"], declared["left_fraction"]) if sg <= 1e-3)
    # Robustness at larger noise: z of (declared - other) left fraction, smallest over sigma >= 3e-3.
    wide = [i for i, sg in enumerate(corridors["sigmas"]) if sg >= 3e-3]
    n_corridor = corridors["samples"]
    agreement = declared_strip_agreement(invalid, samples, declared["left_fraction"], corridors["sigmas"], n_corridor)

    def _z_less(row):
        z = []
        for i in wide:
            a, b = declared["left_fraction"][i], row["left_fraction"][i]
            z.append((a - b) / math.sqrt(max(a * (1 - a) + b * (1 - b), 0.5 / n_corridor) / n_corridor))
        return min(z)
    narrower = [r for r in others if r["vertex_margin"] < declared["vertex_margin"]]
    robust = max(narrower, key=_z_less) if narrower else None
    slopes = scaling["slopes"]
    local = scaling["local_slopes"]
    slope_k = [v for k, v in slopes.items() if k.startswith("angle-defect")]
    slope_n = [v for k, v in slopes.items() if k.startswith("vertex normal")]
    slope_d = slopes["marker geodesic distance"]
    srows = scaling["rows"]
    marker_share = min(r["marker-face part"] / r["interior part"] for r in srows)
    rrows = refinement["rows"]
    fmodels = {m["covariance"]: m for m in field["models"]}
    far = field["far_vertex"]
    # The per-vertex field at the two studied vertices against the two-vertex propagation (separate one-ring code).
    sampled = [(fmodels["isotropic"]["curvature_sd"][0], items["angle-defect curvature at valence-5 vertex 0"]),
               (fmodels["isotropic"]["normal_sd"][0], items["vertex normal at valence-5 vertex 0"]),
               (fmodels["isotropic"]["curvature_sd"][far],
                items["angle-defect curvature at valence-6 vertex far from valence 5"]),
               (fmodels["isotropic"]["normal_sd"][far], items["vertex normal at valence-6 vertex far from valence 5"])]
    field_gap = max(abs(sd / (field["sigma"] * o["gradient_norm"]) - 1) for sd, o in sampled)
    field_summary = {name: {k: v for k, v in m.items() if not isinstance(v, np.ndarray)}
                     for name, m in fmodels.items()}
    ctx.artifact_json("vertex-uncertainty.json", jsonable({"propagation": study, "corridors": corridors,
                                                           "scaling": scaling, "refinement": refinement,
                                                           "per_vertex_field": field}))
    short = {"marker geodesic distance": "marker distance", "vertex normal at valence-5 vertex 0": "normal v5",
             "angle-defect curvature at valence-5 vertex 0": "K v5",
             "vertex normal at valence-6 vertex far from valence 5": "normal v6",
             "angle-defect curvature at valence-6 vertex far from valence 5": "K v6"}
    ctx.artifact_text("linearization-ratio.svg", svg.line_plot(
        [(short[o["observable"]], [r["sigma"] for r in o["rows"]], [r["ratio"] for r in o["rows"]])
         for o in study["observables"]],
        title="Monte Carlo / linearized variance", xlabel="vertex noise sigma", ylabel="ratio", logx=True))
    h2 = study["h_squared"]
    local_spread = max(abs(p - slopes[name]) for name, ps in local.items() for p in ps)
    findings = [
        finding("Linearized vertex-noise propagation matches Monte Carlo for the marker distance along a fixed "
                "face corridor", "numerical",
                {"ratios": [r["ratio"] for r in distance["rows"]], "sigmas": [r["sigma"] for r in distance["rows"]],
                 "gradient_norm": distance["gradient_norm"]},
                {"generator": generator(f"{GEOM}.icosphere", level=study["level"], samples=samples, seed=S.SEED,
                                        strip=study["strip"]["start_index"]),
                 "checks": [check("max |MC / linear variance - 1| for sigma <= 1e-2", distance_dev, 0.1, "le",
                                  kind="self_convergence")]},
                uncertainty=_mc(samples), tolerance=TIGHT),
        finding("The declared marker segment stays in its face corridor for sigma <= 1e-3 but leaves it in over 10% "
                "of samples at sigma = 1e-2", "numerical",
                {"left_fraction_by_sigma": invalid, "vertex_margin": study["strip"]["vertex_margin"],
                 "start_index": study["strip"]["start_index"], "seed": S.SEED,
                 "corridor_study": {"seed": S.SEED + 30, "samples": n_corridor,
                                    "left_fraction": agreement["other"], "max_abs_difference": agreement["max_abs"],
                                    "interval_95_at_max": agreement["interval_at_max"]}},
                {"generator": generator(f"{GEOM}.icosphere", level=study["level"], samples=samples, seed=S.SEED),
                 "checks": [check("largest left fraction for sigma <= 1e-3", corridor_small, 0.0, "le",
                                  kind="self_convergence"),
                            check("left fraction at sigma = 1e-2", corridor_large, 0.1, "ge",
                                  kind="self_convergence"),
                            check("largest |difference| between this estimate and the independent six-strip corridor "
                                  f"study (seed {S.SEED + 30}) over sigma, in standard errors of a difference of two "
                                  "independent binomial fractions (1.96 bounds the 95% interval)", agreement["max_z"],
                                  AGREEMENT_Z, "le", kind="self_convergence")]},
                uncertainty=_binomial(corridor_large, samples), tolerance=TIGHT,
                counterexample={"statement": "A fixed face corridor (fixed mesh combinatorics) represents the "
                                             "perturbed marker geodesic at every tested vertex-noise level",
                                "witness": {"sigma": 1e-2, "left_fraction": corridor_large}}),
        finding("The noise level at which a fixed face corridor fails depends on the strip, and the declared strip, "
                "which has the largest vertex margin of the six, stays in its corridor for sigma <= 1e-3", "numerical",
                {"sigmas": corridors["sigmas"], "strips": crow},
                {"generator": generator(f"{STUD}.corridor_study", level=corridors["level"],
                                        samples=corridors["samples"], seed=S.SEED + 30),
                 "checks": [check("largest left fraction over the six strips at sigma = 1e-3",
                                  worst["left_fraction"][at], 0.1, "ge", kind="self_convergence"),
                            check("declared strip's largest left fraction for sigma <= 1e-3", declared_small, 0.0,
                                  "le", kind="self_convergence"),
                            check("declared strip margin minus the largest other margin",
                                  declared["vertex_margin"] - max(r["vertex_margin"] for r in others), 0.0,
                                  "signed_ge", kind="exact_arithmetic")]},
                uncertainty=_binomial(worst["left_fraction"][at], corridors["samples"]), tolerance=TIGHT,
                counterexample={"statement": "The fixed-corridor marker distance is valid for sigma <= 1e-3 on "
                                             "icosphere-3 whichever declared geodesic carries the markers",
                                "witness": {"start_index": worst["start_index"], "sigma": 1e-3,
                                            "vertex_margin": worst["vertex_margin"],
                                            "left_fraction": worst["left_fraction"][at]}}),
    ]
    robust_claim = ("A strip with a smaller vertex margin than the declared strip leaves its face corridor less often "
                    "at every tested sigma >= 3e-3")
    if robust is None:
        findings.append(_missing_witness(robust_claim, "strips with a smaller margin that leave their corridor less "
                                                       "often than the declared strip"))
    else:
        findings.append(finding(
            robust_claim, "numerical",
            {"sigmas": corridors["sigmas"],
             "declared": {k: declared[k] for k in ("start_index", "vertex_margin", "left_fraction")},
             "more_robust": {k: robust[k] for k in ("start_index", "vertex_margin", "left_fraction")}},
            {"generator": generator(f"{STUD}.corridor_study", level=corridors["level"], samples=n_corridor,
                                    seed=S.SEED + 30),
             "checks": [check("smallest z over sigma >= 3e-3 of (declared - witness strip) left fraction, binomial "
                              "standard error of the difference", _z_less(robust), 3.0, "signed_ge",
                              kind="self_convergence"),
                        check("declared strip margin minus the witness strip margin",
                              declared["vertex_margin"] - robust["vertex_margin"], 0.0, "signed_ge",
                              kind="exact_arithmetic")]},
            uncertainty=_binomial(declared["left_fraction"][wide[0]], n_corridor), tolerance=TIGHT,
            counterexample={"statement": "The strip with the largest vertex margin leaves its fixed face corridor "
                                         "least often at every tested noise level",
                            "witness": {"declared_start_index": declared["start_index"],
                                        "more_robust_start_index": robust["start_index"],
                                        "sigmas": [corridors["sigmas"][i] for i in wide],
                                        "declared_left_fraction": [declared["left_fraction"][i] for i in wide],
                                        "witness_left_fraction": [robust["left_fraction"][i] for i in wide]}}))
    findings += [
        finding("Linearized vertex-noise propagation matches Monte Carlo for vertex normals at every tested sigma, "
                "with no normal sign flips", "numerical",
                {"ratios": {o["observable"]: [r["ratio"] for r in o["rows"]] for o in normals},
                 "flipped_normals": flips, "min_normal_dot": min_dot},
                {"generator": generator(f"{GEOM}.icosphere", level=study["level"], samples=samples, seed=S.SEED),
                 "checks": [check("max |MC / linear mean-square angle - 1|", normal_dev, 0.1, "le",
                                  kind="self_convergence"),
                            check("samples with n . n0 < 0 over all sigma", flips, 0.0, "le",
                                  kind="exact_arithmetic")]},
                uncertainty=_mc(samples), tolerance=TIGHT),
        finding("Linearized propagation matches Monte Carlo for angle-defect curvature when sigma <= 1e-3", "numerical",
                {o["observable"]: [r["ratio"] for r in o["rows"]] for o in curvature},
                {"generator": generator(f"{GEOM}.icosphere", level=study["level"], samples=samples, seed=S.SEED),
                 "checks": [check("max |MC / linear variance - 1| for sigma <= 1e-3", curvature_dev, 0.1, "le",
                                  kind="self_convergence")]},
                uncertainty=_mc(samples), tolerance=TIGHT),
        finding("First-order propagation underestimates angle-defect curvature variance at sigma = 1e-2", "numerical",
                {"min_ratio": breakdown, "sigma_over_h_squared": 1e-2 / h2,
                 "bias": [r["bias"] for r in large]},
                {"derivation": "The defect is quadratic in normal displacement (cone defect ~ pi e^2 / l^2); the "
                               "quadratic term matters once sigma R / h^2 is not small",
                 "checks": [check("smallest MC / linear variance ratio at sigma = 1e-2", breakdown, 1.3, "ge",
                                  kind="self_convergence")]},
                uncertainty=_mc(samples), tolerance=TIGHT,
                counterexample={"statement": "First-order (linearized) propagation of vertex noise is adequate for "
                                             "angle-defect curvature at sigma = 1e-2 on icosphere-3",
                                "witness": {"sigma": 1e-2, "h": study["h"], "min_ratio": breakdown}}),
        finding("Sensitivity to vertex noise scales as h^-2 for curvature, h^-1 for normals and h^0 for the marker "
                "distance of one declared geodesic", "numerical", {"slopes": slopes, "local_slopes": local},
                {"derivation": "Curvature = defect / (A/3) with A ~ h^2 and d(defect)/d(normal offset) ~ pi / R; "
                               "normal tilt ~ offset / h; marker distance moves with the barycentric marker vertices",
                 "checks": [check("worst |curvature slope + 2|", max(abs(s + 2) for s in slope_k), 0.25, "le",
                                  kind="self_convergence"),
                            check("worst |normal slope + 1|", max(abs(s + 1) for s in slope_n), 0.2, "le",
                                  kind="self_convergence"),
                            check("marker distance slope (one geodesic at every level)", slope_d, 0.2,
                                  kind="self_convergence")]},
                uncertainty=_u("reference_error", local_spread, "largest deviation of a pairwise local slope from "
                               "the fitted slope"),
                tolerance=RATE),
        finding("The marker-distance sensitivity is carried by the marker-face vertices at order h^0 while the "
                "interior-strip part decreases under refinement", "numerical",
                {"levels": [r["level"] for r in srows], "marker_face_part": [r["marker-face part"] for r in srows],
                 "interior_part": [r["interior part"] for r in srows],
                 "marker_face_slope": slopes["marker-face part"], "interior_slope": slopes["interior part"],
                 "interior_local_slopes": local["interior part"]},
                {"derivation": "Marker-face vertices carry the barycentric markers (an O(1) gain). An interior vertex "
                               "changes the unfolded length only through the metric, O(delta h / R) for each of about "
                               "2 L / h nearby vertices, so |grad_interior d| ~ sqrt(L h) / R (slope 0.5)",
                 "checks": [check("marker-face part slope", slopes["marker-face part"], 0.2, kind="self_convergence"),
                            check("interior part slope - 0.5", slopes["interior part"] - 0.5, 0.25,
                                  kind="self_convergence"),
                            check("smallest marker-face / interior ratio over levels", marker_share, 2.0, "ge",
                                  kind="self_convergence")]},
                uncertainty=_u("reference_error", max(abs(p - slopes["interior part"]) for p in local["interior part"]),
                               "largest deviation of a pairwise local interior slope from the fitted slope"),
                tolerance=RATE),
        finding("Under fixed vertex noise the curvature error grows as the mesh is refined", "numerical",
                {"levels": [r["level"] for r in rrows], "total_rms_error": [r["total_rms_error"] for r in rrows],
                 "discretization_error": [r["discretization_error"] for r in rrows]},
                {"generator": generator(f"{STUD}.refinement_noise_study", levels=[r["level"] for r in rrows],
                                        sigma=refinement["sigma"], samples=refinement["samples"], seed=S.SEED + 50),
                 "checks": [check("finest minus coarsest total RMS error", rrows[-1]["total_rms_error"]
                                  - rrows[0]["total_rms_error"], 0.0, "signed_ge", kind="self_convergence")]},
                uncertainty=_mc(refinement["samples"]), tolerance=TIGHT,
                counterexample={"statement": "Refining the mesh reduces the error of angle-defect curvature",
                                "witness": {"sigma": refinement["sigma"], "coarse": rrows[0], "fine": rrows[-1]}}),
        finding("Linearized per-vertex standard deviations of vertex normals and angle-defect curvature match Monte "
                "Carlo at every vertex of icosphere-3 under isotropic and normal-dominant vertex covariances",
                "numerical", {"models": field_summary, "vertices": field["vertices"],
                              "two_vertex_cross_check_max_rel": field_gap},
                {"generator": generator(f"{STUD}.vertex_field_study", level=field["level"], samples=field["samples"],
                                        seed=S.SEED + 60, sigma=field["sigma"], normal_sigma=field["normal_sigma"],
                                        tangential_sigma=field["tangential_sigma"]),
                 "checks": [check(f"max over vertices of |z| of MC / linear {quantity} ({name} covariance; "
                                  "z = (ratio - 1) / sqrt(2 / (N - 1)), 5 bounds all vertices jointly)",
                                  fmodels[name][f"max_abs_z_{quantity}"], 5.0, "le", kind="self_convergence")
                            for name in fmodels for quantity in ("curvature", "normal")]
                 + [check("per-vertex field (G.vertex_uncertainty) against the two-vertex propagation Jacobians at "
                          "vertex 0 and the far valence-6 vertex, max relative difference", field_gap, 1e-6, "le",
                          kind="cross_implementation")]},
                uncertainty=_mc(field["samples"]), tolerance=TIGHT),
        _physical("Isotropic Gaussian vertex noise of the tested sigma describes the error of a real scanner",
                  "calibration"),
    ]
    robust_text = (f"; strip {robust['start_index']} (margin {robust['vertex_margin']:.4f}) leaves less often than the "
                   "declared strip at sigma >= 3e-3 ("
                   + ", ".join(f"{robust['left_fraction'][i]:.3f} vs {declared['left_fraction'][i]:.3f}" for i in wide)
                   + ")") if robust is not None else ""
    iso, aniso = fmodels["isotropic"], fmodels["normal-dominant"]
    result = (f"Icosphere-{study['level']} (h={study['h']:.4f}), {samples} samples per sigma, declared strip "
              f"{study['strip']['start_index']} (largest margin of six): fixed-corridor distance variance ratio within "
              f"{distance_dev:.3f} of 1 (sigma<=1e-2), but the segment leaves the corridor in "
              + ", ".join(f"{float(v):.3f}@{k}" for k, v in invalid.items())
              + f" of samples (seed {S.SEED}); the six-strip corridor study (seed {S.SEED + 30}, {n_corridor} samples) "
              "gives the declared strip "
              + ", ".join(f"{v:.3f}@{sg:g}" for sg, v in zip(corridors["sigmas"], declared["left_fraction"]))
              + f", and {agreement_text(agreement)}; corridor comparisons "
              "between strips below use the six-strip study only. Over all six strips the left fraction at "
              "sigma=1e-3 ranges "
              f"{min(r['left_fraction'][at] for r in crow):.3f}-{worst['left_fraction'][at]:.3f} (smallest margin "
              f"{worst['vertex_margin']:.4f}){robust_text}. Per-vertex field over all {field['vertices']} vertices "
              f"({field['samples']} samples): isotropic sigma={field['sigma']:g} gives curvature SD "
              f"{iso['curvature_sd_range'][0]:.4f}-{iso['curvature_sd_range'][1]:.4f} and normal SD "
              f"{iso['normal_sd_range'][0]:.2e}-{iso['normal_sd_range'][1]:.2e} rad, max |z| "
              f"{iso['max_abs_z_curvature']:.2f} and {iso['max_abs_z_normal']:.2f}; normal-dominant (sigma_n="
              f"{field['normal_sigma']:g}, sigma_t={field['tangential_sigma']:g}) max |z| "
              f"{aniso['max_abs_z_curvature']:.2f} and {aniso['max_abs_z_normal']:.2f}; two-vertex cross-check "
              f"{field_gap:.1e}. Normal ratio within {normal_dev:.3f}, {flips} sign flips (min n.n0 "
              f"{min_dot:.4f}); curvature ratio within {curvature_dev:.3f} for sigma<=1e-3 but >= {breakdown:.2f} at "
              f"sigma=1e-2 (sigma/h^2={1e-2 / h2:.2f}). Sensitivity slopes (declared geodesic {scaling['start_index']} "
              "at every level): " + ", ".join(f"{short.get(k, k)}: {v:.2f}" for k, v in slopes.items())
              + ". Total curvature RMS error at sigma="
              f"{refinement['sigma']:g}: "
              + ", ".join(f"{r['total_rms_error']:.3f}@L{r['level']}" for r in rrows) + ".")
    return {"state": "completed", "findings": findings, "fields": fields(
        "Gaussian vertex noise propagates to geodesic length and normals nearly linearly, while angle-defect "
        "curvature needs sigma << h^2 / R for linearization and becomes noisier as the mesh is refined; linearized "
        "per-vertex normal and curvature standard deviations under a declared vertex covariance match Monte Carlo at "
        "every vertex in the linear regime; the fixed-corridor distance stays meaningful only below a "
        "strip-dependent noise level that the vertex margin alone does not order.",
        "Linearization Var[f] = sigma^2 |grad f|^2 (or sum_j J_j C_j J_j^T for a declared per-vertex covariance C_j) "
        "with a central finite-difference Jacobian (step 1e-6) over the vertices that affect f; seeded Monte Carlo "
        "with the same noise. Marker distance = planar distance after unfolding a fixed face strip with barycentric "
        "markers.",
        ["icosphere levels 2-5 (level 3 for Monte Carlo)", "sigma = 1e-4, 1e-3, 3e-3, 1e-2 (normalized units, R = 1)",
         f"markers on declared geodesic {S.MARKER_START} of six (length 1), chosen because it has the largest vertex "
         "margin on icosphere-3 (a larger margin does not make it the most robust strip at sigma >= 3e-3); all six "
         "strips are reported for corridor validity",
         "two-vertex study: valence-5 vertex 0 and the valence-6 vertex farthest from valence-5 vertices",
         f"per-vertex field: all {field['vertices']} vertices, isotropic sigma = {field['sigma']:g} and "
         f"normal-dominant covariance (sigma_n = {field['normal_sigma']:g}, sigma_t = {field['tangential_sigma']:g})"],
        "Synthetic Gaussian displacement of every vertex (isotropic, or the declared normal-dominant covariance); no "
        "scanner model beyond that.",
        "MC/linear variance ratio near 1 in the linear regime; ratio > 1 once second-order terms matter; exponents "
        "-2, -1, 0.",
        "For each observable and sigma: FD Jacobian, predicted variance, 4000 seeded samples, ratio, bias and normal "
        "sign flips; corridor-leaving fractions for all six strips; per-vertex linearized normal and curvature "
        "standard deviations (G.vertex_uncertainty) against whole-mesh Monte Carlo at every vertex; gradient norms of "
        "one declared geodesic across levels split into marker-face and interior parts; total error versus level at "
        "fixed sigma.",
        result,
        f"Monte Carlo relative 95% half-width of a variance with {samples} samples is about "
        f"{1.96 * math.sqrt(2 / (samples - 1)):.3f}; checks use 0.1. The fixed-strip distance is the geodesic "
        "distance only while the unfolded segment stays inside the strip (fraction reported per strip).",
        ["strip validity under noise, for all six declared strips", "finite-difference step size",
         "valence-5 versus valence-6 vertices", "nonlinear bias of curvature", "normal sign flips (counted: none)",
         "marker attachment versus interior shape sensitivity"],
        ["Noise is isotropic and independent per vertex; real scanners have correlated, anisotropic errors.",
         "Markers are attached barycentrically to faces, so tangential vertex noise drags them; this dominates the "
         "marker-distance gain (see T044).",
         "The corridor threshold depends on the strip: roughly where sigma reaches the smallest distance between the "
         "unfolded segment and a strip vertex (edge margin times h), but the margin does not order the strips: "
         "strips with equal margins differ, and a smaller-margin strip leaves less often than the declared strip at "
         "sigma >= 3e-3, because other near-vertex crossings matter. Past it the perturbed geodesic can switch "
         "corridors, its distance is a minimum over corridors, and re-tracing per sample is not done.",
         "Normals are always derived from the noisy vertices; an input uncertainty model for independently measured "
         "normals (for example scanner-reported normals) is not implemented and is deferred.",
         "The per-vertex field is checked at sigma where linearization holds (sigma R / h^2 about 0.01); at larger "
         "sigma the curvature SD is underestimated as in the two-vertex study."],
        NEXT_STEPS["T043"])}


# ---------------------------------------------------------------- T044
@task("T044", changed_files=FILES, regression_tests=(
    NEXT_STEP_TEST,
    f"{TESTS}::test_law_of_total_variance_split",
    f"{TESTS}::test_variance_split_task_report",
    f"{TESTS}::test_t044_next_step_names_what_t045_leaves_open"))
def geometry_versus_sensor_uncertainty(ctx):
    study = _memo(ctx, "variance-split", S.variance_split_study)
    corridors = _memo(ctx, "corridors", S.corridor_study)
    at = corridors["sigmas"].index(1e-3)
    narrowest = min(corridors["rows"], key=lambda r: r["vertex_margin"])
    scenarios = study["scenarios"]
    outer, inner, fresh = study["outer"], study["inner"], study["fresh"]
    split = study["gain_split"]
    normal_only = study["normal_only"]
    anova = max(abs(s["anova_residual"]) for s in scenarios)
    corridor_left = max(s["corridor_left_fraction"] for s in scenarios)
    z_fresh, z_within, z_between = [], [], []
    for s in scenarios:
        total = s["predicted_total"]
        z_fresh.append((s["fresh_total"] - total) / (total * math.sqrt(2 / (fresh - 1))))
        z_within.append((s["within_variance"] - s["sensor_variance"])
                        / (s["sensor_variance"] * math.sqrt(2 / (outer * (inner - 1)))))
        between_expected = s["geometry_variance_linear"] + s["sensor_variance"] / inner
        z_between.append((s["between_corrected"] - s["geometry_variance_linear"])
                         / (between_expected * math.sqrt(2 / (outer - 1))))
    baseline = next(s for s in scenarios if s["sigma_geometry"] == 1e-3 and s["sigma_sensor"] == 5e-4)
    averaging = study["averaging"]["rows"]
    z_avg = [(r["variance"] - r["predicted"]) / (r["predicted"] * math.sqrt(2 / (fresh - 1))) for r in averaging]
    geometry_floor = (study["gain"] * study["averaging"]["sigma_geometry"]) ** 2
    se = math.sqrt(2 / (fresh - 1))
    z_normal_geometry = (normal_only["geometry_variance_mc"] / normal_only["geometry_variance_linear"] - 1) / se
    z_normal_total = (normal_only["total_mc"] / normal_only["predicted_total"] - 1) / se
    tangential_fraction = split["tangential"] ** 2 / split["total"] ** 2
    marker_fraction = split["marker"] ** 2 / split["total"] ** 2
    # Bookkeeping identities (hold for any data or gradient): kept as sanity values, not as evidence.
    pythagoras = abs(split["tangential"] ** 2 + split["normal"] ** 2 - split["total"] ** 2) / split["total"] ** 2
    ctx.artifact_json("variance-split.json", jsonable({**study, "z_fresh": z_fresh, "z_within": z_within,
                                                        "z_between": z_between, "z_averaging": z_avg,
                                                        "z_normal_only": [z_normal_geometry, z_normal_total]}))
    ctx.artifact_text("averaging.svg", svg.line_plot([
        ("total variance (MC)", [r["repeats"] for r in averaging], [r["variance"] for r in averaging]),
        ("geometry + sensor/K", [r["repeats"] for r in averaging], [r["predicted"] for r in averaging]),
        ("geometry floor", [r["repeats"] for r in averaging], [geometry_floor] * len(averaging))],
        title="Averaging K sensor readings on one uncertain surface", xlabel="repeated readings K",
        ylabel="variance", logx=True, logy=True))
    shares = [{"sigma_geometry": s["sigma_geometry"], "sigma_sensor": s["sigma_sensor"],
               "geometry_share": s["geometry_share"], "dominant": "geometry" if s["geometry_share"] > 0.5 else "sensor"}
              for s in scenarios]
    nested_share_gap = baseline["between_corrected"] / baseline["nested_total"] - baseline["geometry_share"]
    strip_note = f"declared strip {study['strip_start_index']} (largest vertex margin, T043)"
    findings = [
        finding("Residual variance equals geometry variance plus sensor variance in every scenario", "numerical",
                {"fresh_ratio": [s["fresh_ratio"] for s in scenarios], "z": z_fresh,
                 "corridor_left_fraction": corridor_left},
                {"derivation": "Law of total variance: Var(r) = E[Var(r | eta)] + Var(E[r | eta]) = sigma_s^2 + "
                               "Var_eta(d)",
                 "checks": [check("max |z| of fresh total variance against sigma_s^2 + sigma_g^2 |grad d|^2",
                                  max(abs(z) for z in z_fresh), 4.0, "le", kind="self_convergence"),
                            check(f"largest fraction of samples leaving the fixed corridor ({strip_note})",
                                  corridor_left, 0.001, "le", kind="self_convergence")]},
                uncertainty=_mc(fresh), tolerance=TIGHT),
        finding("Nested variance components are consistent with the declared sensor variance and the linearized "
                "geometry variance within sampling error",
                "numerical", {"within_ratio": [s["within_ratio"] for s in scenarios],
                              "between_ratio": [s["between_ratio"] for s in scenarios],
                              "z_within": z_within, "z_between": z_between},
                {"generator": generator(f"{STUD}.variance_split_study", outer=outer, inner=inner, seed=S.SEED + 100),
                 "checks": [check("max |z| within-group variance vs sigma_s^2", max(abs(z) for z in z_within), 4.0,
                                  "le", kind="self_convergence"),
                            check("max |z| corrected between-group variance vs geometry variance",
                                  max(abs(z) for z in z_between), 4.0, "le", kind="self_convergence")]},
                uncertainty=_mc(outer), tolerance=TIGHT),
        finding("Under isotropic vertex noise with barycentric markers, geometry uncertainty dominates the baseline "
                "marker-distance residual", "numerical",
                {"baseline": {"sigma_geometry": 1e-3, "sigma_sensor": 5e-4,
                              "geometry_share": baseline["geometry_share"],
                              "crossover_sensor_sigma": baseline["crossover_sensor_sigma"]}, "scenarios": shares,
                 "gain": study["gain"]},
                {"generator": generator(f"{STUD}.variance_split_study", level=study["level"], seed=S.SEED + 100,
                                        noise="isotropic", strip=study["strip_start_index"]),
                 "checks": [check("baseline geometry share of the total variance", baseline["geometry_share"], 0.5,
                                  "ge", kind="self_convergence"),
                            check("nested geometry share - linearized share at baseline", nested_share_gap, 0.05,
                                  kind="self_convergence")]},
                uncertainty=_u("monte_carlo_95ci", 1.96 * math.sqrt(2 / (outer - 1)) * baseline["geometry_share"],
                               f"95% half-width of the nested geometry share from {outer} geometry samples"),
                tolerance=TIGHT),
        finding("Tangential vertex displacement, mostly of the marker-face vertices, carries most of the "
                "marker-distance gain |grad d|^2", "numerical",
                {"gain_split": split, "tangential_fraction": tangential_fraction, "marker_fraction": marker_fraction},
                {"derivation": "Isotropic noise splits exactly into normal and tangential parts: |grad d|^2 = "
                               "|grad_n d|^2 + |grad_t d|^2. On a smooth surface tangential noise only "
                               "re-parameterises the mesh to first order, except that it drags barycentric markers",
                 "checks": [check("tangential fraction of |grad d|^2", tangential_fraction, 0.5, "ge",
                                  kind="self_convergence"),
                            check("marker-face fraction of |grad d|^2", marker_fraction, 0.5, "ge",
                                  kind="self_convergence")]},
                uncertainty=_u("truncation_bound", 1e-9, "central differences with step 1e-6: O(tau^2) truncation "
                               "plus about 1e-10 rounding per entry"),
                tolerance=TIGHT),
        finding("Under normal-only (shape) vertex noise of the same sigma the sensor dominates the baseline residual",
                "numerical", normal_only,
                {"generator": generator(f"{STUD}.variance_split_study", level=study["level"], seed=S.SEED + 100,
                                        noise="normal-only", samples=fresh),
                 "checks": [check("normal-only geometry share at sigma_g = 1e-3, sigma_s = 5e-4",
                                  normal_only["geometry_share"], 0.5, "le", kind="self_convergence"),
                            check("z of MC normal-only geometry variance against sigma_g^2 |grad_n d|^2",
                                  z_normal_geometry, 4.0, kind="self_convergence"),
                            check("z of MC total against sigma_g^2 |grad_n d|^2 + sigma_s^2", z_normal_total, 4.0,
                                  kind="self_convergence")]},
                uncertainty=_mc(fresh), tolerance=TIGHT,
                counterexample={"statement": "Whether geometry or sensor noise dominates the marker-distance residual "
                                             "is fixed by sigma_g and sigma_s alone",
                                "witness": {"sigma_geometry": 1e-3, "sigma_sensor": 5e-4,
                                            "isotropic_geometry_share": baseline["geometry_share"],
                                            "normal_only_geometry_share": normal_only["geometry_share"]}}),
        finding("Averaging repeated sensor readings leaves the geometry variance as a floor", "numerical",
                {"repeats": [r["repeats"] for r in averaging], "variance": [r["variance"] for r in averaging],
                 "geometry_floor": geometry_floor, "geometry_share": [r["geometry_share"] for r in averaging],
                 "z": z_avg},
                {"generator": generator(f"{STUD}.variance_split_study",
                                        sigma_geometry=study["averaging"]["sigma_geometry"],
                                        sigma_sensor=study["averaging"]["sigma_sensor"], seed=S.SEED + 100),
                 "checks": [check("variance at the largest K / geometry floor",
                                  averaging[-1]["variance"] / geometry_floor, 0.9, "ge", kind="self_convergence"),
                            check("max |z| against geometry + sensor / K", max(abs(z) for z in z_avg), 4.0, "le",
                                  kind="self_convergence")]},
                uncertainty=_mc(fresh), tolerance=TIGHT,
                counterexample={"statement": "Averaging repeated measurements drives the observation error to zero",
                                "witness": {"repeats": averaging[-1]["repeats"], "variance": averaging[-1]["variance"],
                                            "geometry_floor": geometry_floor}}),
        _physical("A real marker-distance sensor on a real scanned part has this geometry and sensor variance split",
                  "sensor_performance"),
        _physical("The geometry sigma of 1e-3 is the accuracy of a real scanned surface"),
    ]
    result = (f"Distance gain |grad d| = {study['gain']:.4f} (nominal distance {study['nominal_distance']:.6f}; "
              f"tangential {split['tangential']:.4f}, normal {split['normal']:.4f}; marker-face vertices "
              f"{split['marker']:.4f}, interior {split['interior']:.4f}); bookkeeping sanity values (identities, not "
              f"evidence): ANOVA residual {anova:.1e}, normal/tangential split residual {pythagoras:.1e}; fresh "
              "total/predicted ratios " + ", ".join(f"{s['fresh_ratio']:.3f}" for s in scenarios)
              + f" (max |z| {max(abs(z) for z in z_fresh):.2f}); isotropic baseline (sigma_g=1e-3, sigma_s=5e-4) "
              f"geometry share {baseline['geometry_share']:.3f}, crossover sensor sigma "
              f"{baseline['crossover_sensor_sigma']:.2e}; normal-only noise: share "
              f"{normal_only['geometry_share']:.3f}, "
              f"crossover {normal_only['crossover_sensor_sigma']:.2e}, MC/linear geometry variance "
              f"{normal_only['geometry_variance_mc'] / normal_only['geometry_variance_linear']:.3f}; averaging K="
              + ", ".join(f"{r['repeats']}: share {r['geometry_share']:.2f}" for r in averaging)
              + f"; floor {geometry_floor:.3e}.")
    return {"state": "completed", "findings": findings, "fields": fields(
        "For a marker-distance observation on an uncertain surface, geometry and sensor variances add (law of total "
        "variance), can be separated by a nested design, and averaging sensor readings cannot remove the geometry "
        "part; which part dominates depends on the geometry noise model as well as on the sigmas.",
        "Residual r = y - d(V_nominal), y = d(V_nominal + eta) + eps, eta ~ N(0, sigma_g^2 I) per vertex coordinate "
        "(or sigma_g along each vertex normal), eps ~ N(0, sigma_s^2) independent. Var(r) = sigma_s^2 + Var_eta(d) "
        "~ sigma_s^2 + sigma_g^2 |grad d|^2, with |grad_n d|^2 in place of |grad d|^2 for normal-only noise. Nested "
        "design: within-group variance estimates sigma_s^2, corrected between-group variance estimates the geometry "
        "part.",
        [f"icosphere level 3, fixed marker strip {study['strip_start_index']} from T043 (the declared geodesic with "
         "the largest vertex margin of six)",
         "sigma_g in {1e-4, 1e-3}, sigma_s in {1e-4, 5e-4, 2e-3}",
         f"nested {outer} x {inner} and fresh {fresh} samples",
         "averaging K = 1, 4, 16, 64 at sigma_g=1e-3, sigma_s=2e-3", "normal-only noise at sigma_g=1e-3, sigma_s=5e-4"],
        "Synthetic: the as-built surface differs from the nominal mesh by Gaussian vertex noise (isotropic, or along "
        "the normals) and a synthetic sensor adds Gaussian noise to the distance; markers are barycentric in their "
        "faces, so tangential vertex noise moves them.",
        "Within-group variance near sigma_s^2 and corrected between-group variance near the linearized geometry "
        "variance; totals within sampling error of sigma_s^2 + sigma_g^2 |grad d|^2; floor at the geometry variance "
        "under averaging. (The ANOVA sum-of-squares identity and the normal/tangential split of |grad d|^2 hold for "
        "any data and are recorded only as bookkeeping sanity values.)",
        "Nested and fresh seeded Monte Carlo per scenario; normal-only fresh Monte Carlo; z-scores use the Gaussian "
        "variance standard error.",
        result,
        "z-scores use sqrt(2/(N-1)) relative standard errors of Gaussian variances; the geometry variance uses the "
        "T043 linearization, which holds for this declared strip at these sigma (no sample left its corridor), not "
        f"for every strip (T043 reports {narrowest['left_fraction'][at]:.1%} leaving at sigma_g = 1e-3 for the "
        f"smallest-margin strip {narrowest['start_index']}). When "
        "sigma_s^2 / inner greatly exceeds the geometry variance the corrected between-group estimate is dominated by "
        "sensor noise (it can be negative, as at sigma_g = 1e-4, sigma_s = 2e-3); consistency there is not a useful "
        "estimate of the geometry part.",
        ["unidentifiable geometry share when sigma_s^2 / inner >> geometry variance (handled by z-scores)",
         "nonlinearity of the distance (checked in T043)", "strip validity (declared strip only; T043)",
         "noise-model dependence of the dominant component (isotropic versus normal-only)"],
        ["Geometry and sensor noise are independent; a sensor that touches the same surface patch could correlate "
         "them.",
         "Most of the isotropic geometry variance is tangential marker displacement; if physical markers were fixed "
         "on the surface independently of the mesh vertices, the normal-only share (about 0.28 at baseline) is the "
         "relevant one.",
         "Which part dominates is a property of the declared sigmas and noise model, not of any real scanner or "
         "sensor."],
        NEXT_STEPS["T044"])}
