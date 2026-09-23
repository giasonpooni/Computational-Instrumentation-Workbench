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
from .evidence import finding
from .registry import task
from . import surfaces_discrete_mesh_studies as S

MODULE = "src/ciw/lab/surfaces_discrete_mesh.py"
GEOMETRY = "src/ciw/lab/surfaces_discrete_mesh_geometry.py"
STUDIES = "src/ciw/lab/surfaces_discrete_mesh_studies.py"
DOC = "docs/lab/MESH_GEODESICS.md"
TESTS = "tests/test_lab_surfaces_discrete_mesh.py"
FILES = (MODULE, GEOMETRY, STUDIES, DOC)
PRODUCER = "ciw.lab.surfaces_discrete_mesh"
TIGHT = {"abs": 1e-9, "rel": 1e-6}
RATE = {"abs": 1e-6, "rel": 1e-6}


def _memo(ctx, name, compute):
    return ctx.memo(f"surfaces_discrete_mesh:{name}", compute)


def check(reference, observed, tolerance, comparison="abs_le", kind="analytic") -> dict:
    observed, tolerance = float(observed), float(tolerance)
    holds = {"abs_le": abs(observed) <= tolerance, "le": observed <= tolerance, "ge": observed >= tolerance}[comparison]
    return {"reference_kind": kind, "reference": reference, "observed": observed, "tolerance": tolerance,
            "comparison": comparison, "passed": bool(holds)}


def refusal(reference, expected, observed) -> dict:
    return {"reference_kind": "refusal", "reference": reference, "expected_refusal": expected,
            "observed_refusal": observed if observed is not None else "none", "passed": observed == expected}


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


def generator(name, **params) -> dict:
    return {"name": f"ciw.lab.surfaces_discrete_mesh_geometry.{name}", **jsonable(params)}


def fields(hypothesis, model, inputs, observation, invariant, experiment, result, uncertainty, failure_modes,
           assumptions, next_task) -> dict:
    return {"hypothesis": hypothesis, "mathematical_model": model, "input_data": inputs,
            "observation_model": observation, "expected_invariant": invariant, "experiment": experiment,
            "numerical_result": result, "uncertainty": uncertainty, "failure_modes_checked": failure_modes,
            "unresolved_assumptions": assumptions, "recommended_next_task": next_task}


def _order(rows, key, first=None, last=None):
    rows = rows[first:last]
    return S.fitted_order([r["h"] for r in rows], [r[key] for r in rows])


def _physical(claim, domain="physical"):
    return finding(claim, domain, None, {"notes": "No scanned surface, device or calibration was acquired; "
                                                  "the computation is synthetic."})


# ---------------------------------------------------------------- T038
@task("T038", changed_files=FILES, regression_tests=(
    f"{TESTS}::test_tracer_is_exact_on_developable_meshes",
    f"{TESTS}::test_graph_distances_and_steiner_sandwich",
    f"{TESTS}::test_solver_task_report"))
def mesh_geodesic_solver(ctx):
    traces = _memo(ctx, "sphere-traces", S.sphere_trace_study)
    cylinder = _memo(ctx, "cylinder", S.cylinder_study)
    plane = _memo(ctx, "plane", S.plane_study)
    nested = _memo(ctx, "steiner-nested", S.steiner_nested_study)
    independent = _memo(ctx, "dijkstra-independent", S.dijkstra_independent)
    statuses = sorted({t["status"] for row in traces["rows"] for t in row["traces"]})
    plane_error = max(r["max_trace_error"] for r in plane["rows"])
    development = max(r["development_error"] for r in cylinder["rows"])
    unfold = max(r["max_unfold_minus_trace"] for r in traces["rows"])
    gaps = nested["gap_to_traced"]
    ks = sorted(int(k) for k in gaps)
    mean_gap = {k: float(np.mean(gaps[str(k)])) for k in ks}
    ctx.artifact_json("sphere-traces.json", jsonable(traces))
    ctx.artifact_json("cylinder-and-plane.json", jsonable({"cylinder": cylinder, "plane": plane}))
    ctx.artifact_json("graph-distances.json", jsonable({"steiner_nested": nested, "dijkstra_independent": independent}))

    dijkstra_basis = {"generator": generator("icosphere", level=independent["level"]),
                      "checks": [check(f"dense Floyd-Warshall on icosphere-{independent['floyd_level']}",
                                       independent["floyd_max_abs"], 1e-12, kind="invariant")]}
    dijkstra_value = independent["floyd_max_abs"]
    if independent.get("scipy_max_abs") is not None:
        dijkstra_value = max(dijkstra_value, independent["scipy_max_abs"])
        dijkstra_basis["independent_check"] = dict(
            check(f"scipy.sparse.csgraph.dijkstra on icosphere-{independent['level']} edge graph",
                  independent["scipy_max_abs"], 1e-12),
            producer={"implementation": PRODUCER, "revision": "working tree"},
            checker={"implementation": "scipy.sparse.csgraph.dijkstra", "revision": independent["scipy"]})
    findings = [
        finding("Straightest geodesics on sheared planar meshes coincide with straight lines", "numerical",
                plane_error, {"generator": generator("plane_mesh", shears=[r["shear"] for r in plane["rows"]]),
                              "checks": [check("exact line start + s * direction", plane_error, 1e-12)]},
                unit="normalized length", tolerance={"abs": 1e-12, "rel": 0.0}),
        finding("Traced prism-cylinder geodesics match the exact planar development of the mesh", "numerical",
                development, {"generator": generator("cylinder_mesh", n=[r["n"] for r in cylinder["rows"]]),
                              "checks": [check("development of planar rectangles, circumference 2 n R sin(pi/n)",
                                               development, 1e-11)]},
                unit="normalized length", tolerance={"abs": 1e-11, "rel": 0.0}),
        finding("The unfolded face-strip distance equals the traced length on every completed sphere trace",
                "numerical", {"max_abs_difference": unfold, "trace_statuses": statuses},
                {"generator": generator("icosphere", levels=[r["level"] for r in traces["rows"]]),
                 "checks": [check("strip unfolding from edge lengths (independent of the 3D edge rotation)",
                                  unfold, 1e-11, kind="invariant")]},
                tolerance={"abs": 1e-11, "rel": 0.0}),
        finding("Heap Dijkstra edge-graph distances agree with an independent shortest-path implementation",
                "numerical", dijkstra_value, dijkstra_basis, unit="normalized length",
                tolerance={"abs": 1e-12, "rel": 0.0}),
        finding("Nested Steiner-graph distances never increase with k and never fall below the chord", "numerical",
                {"max_increase_with_k": nested["max_increase_with_k"], "max_chord_excess": nested["max_chord_excess"],
                 "vertex0_to_last": nested["vertex0_to_last"]},
                {"generator": generator("steiner_graph", level=nested["level"], ks=nested["ks"]),
                 "checks": [check("distance(k') - distance(k) for nested k' > k", nested["max_increase_with_k"], 1e-12,
                                  "le", "invariant"),
                            check("chord - graph distance (chord is a lower bound)", nested["max_chord_excess"], 1e-12,
                                  "le", "invariant")]},
                tolerance=TIGHT),
        finding("Steiner distances between traced endpoints stay above the traced length and approach it as k grows",
                "numerical", {"min_gap": nested["min_gap"], "mean_gap_by_k": {str(k): v for k, v in mean_gap.items()}},
                {"generator": generator("steiner_graph", level=nested["level"], traced=len(nested["traced_lengths"])),
                 "checks": [check("smallest gap (Steiner distance - traced length) plus a 1e-12 rounding allowance",
                                  nested["min_gap"] + 1e-12, 0.0, "ge", "invariant"),
                            check("mean gap at largest k - mean gap at smallest k", mean_gap[ks[-1]] - mean_gap[ks[0]],
                                  0.0, "le", "self_convergence")]},
                unit="normalized length", tolerance=TIGHT),
        _physical("Straightest geodesics on a mesh reconstructed from a real scan reproduce the geodesics of the "
                  "scanned physical surface"),
    ]
    result = (f"Plane traces exact to {plane_error:.1e}; prism-cylinder traces match the development to "
              f"{development:.1e}; unfolded strip length equals traced length to {unfold:.1e} over "
              f"{sum(r['completed'] for r in traces['rows'])} sphere traces (statuses {statuses}); edge Dijkstra agrees "
              f"with {'scipy and ' if independent.get('scipy') else ''}Floyd-Warshall to {dijkstra_value:.1e}; Steiner "
              f"distances to traced endpoints exceed the traced length by mean "
              + ", ".join(f"{mean_gap[k]:.2e} (k={k})" for k in ks) + ".")
    return {"state": "completed", "findings": findings, "fields": fields(
        "Unfolding across edges (straight in faces, equal angles at edges) yields exact straightest geodesics on "
        "developable meshes, and graph distances bound polyhedral distances from above.",
        "Straightest geodesic: in each face a straight segment; at an edge the direction keeps its edge component and "
        "the magnitude of its perpendicular component (rotation about the edge). Distances: Dijkstra on the edge "
        "graph and on the graph of k Steiner points per edge (all pairs inside each face); vertex hits are refused.",
        ["icosphere levels 1-6 (unit sphere)", "prism cylinder R=1, n=8..128 sectors",
         "sheared planar 8x8 grids (shear 0-1.5)", "six declared sphere geodesics (chart point, heading)"],
        "No physical observation; traced endpoints, lengths and graph distances in normalized units.",
        "Planar and prism meshes are intrinsically flat, so traces must equal straight lines of the development; "
        "Steiner distances with nested point sets are nonincreasing in k and bounded below by chords.",
        "Trace declared geodesics; compare with exact developments; re-derive lengths by strip unfolding; compare "
        "Dijkstra with scipy (when installed) and Floyd-Warshall; sandwich traced lengths by Steiner distances.",
        result,
        "Deterministic computation; floating-point rounding only (differences at 1e-15 relative). The Steiner "
        "sandwich is consistent with, but does not prove, that the traced geodesics are shortest paths.",
        ["vertex hits (refused, none occurred in the declared set)", "boundary reached", "tracing through a face "
         "without an exit edge", "strip unfolding sign conventions", "graph duplicates from shared face edges",
         "scipy absent (falls back to Floyd-Warshall only)"],
        ["Vertex hits are refused rather than continued by the Polthier-Schmies angle-bisection rule.",
         "The Steiner graph is an approximation scheme; no exact polyhedral (MMP/ICH) distance is implemented.",
         "Traced geodesics are shortest paths only when no shorter corridor exists; not proved here."],
        "T039: measure convergence of these traces and distances under refinement; then implement an exact "
        "polyhedral distance (MMP or ICH) as an independent reference.")}


# ---------------------------------------------------------------- T039
@task("T039", changed_files=FILES, regression_tests=(
    f"{TESTS}::test_convergence_orders_on_small_levels",
    f"{TESTS}::test_convergence_task_report"))
def mesh_refinement_convergence(ctx):
    traces = _memo(ctx, "sphere-traces", S.sphere_trace_study)
    cylinder = _memo(ctx, "cylinder", S.cylinder_study)
    distances = _memo(ctx, "distances", S.distance_study)
    rows = traces["rows"]
    p_length = _order(rows, "mean_length_defect", 1)
    p_endpoint = _order(rows, "mean_endpoint", 1)
    p_cross = _order(rows, "mean_cross_track", 1)
    crow = cylinder["rows"]
    p_helix = S.fitted_order([r["h"] for r in crow], [r["helix_error"] for r in crow])
    scaled = crow[-1]["scaled_error"] / cylinder["leading_constant"]
    drows = distances["rows"]
    edge = [r["edge_max_rel"] for r in drows]
    steiner = {k: [r[f"steiner{k}_max_rel"] for r in drows] for k in (1, 3)}
    heat_rows = [r for r in drows if "heat_max_abs" in r]
    p_heat = S.fitted_order([r["h"] for r in heat_rows], [r["heat_max_abs"] for r in heat_rows])
    table = {"sphere": [{k: r[k] for k in ("level", "h", "mean_endpoint", "max_endpoint", "mean_cross_track",
                                           "mean_along_track", "mean_length_defect", "completed")} for r in rows],
             "orders": {"length_defect": p_length, "endpoint": p_endpoint, "cross_track": p_cross,
                        "helix": p_helix, "heat": p_heat},
             "cylinder": crow, "distances": drows}
    ctx.artifact_json("convergence.json", jsonable(table))
    ctx.artifact_text("convergence.svg", svg.line_plot([
        ("sphere endpoint (mean)", [r["h"] for r in rows], [r["mean_endpoint"] for r in rows]),
        ("sphere cross-track", [r["h"] for r in rows], [r["mean_cross_track"] for r in rows]),
        ("sphere length defect", [r["h"] for r in rows], [r["mean_length_defect"] for r in rows]),
        ("cylinder helix", [r["h"] for r in crow], [r["helix_error"] for r in crow]),
        ("heat distance (max)", [r["h"] for r in heat_rows], [r["heat_max_abs"] for r in heat_rows]),
        ("edge Dijkstra (max rel)", [r["h"] for r in drows], edge),
        ("Steiner k=3 (max rel)", [r["h"] for r in drows], steiner[3])],
        title="Mesh geodesics under refinement", xlabel="mean edge length h", ylabel="error", logx=True, logy=True))
    levels = [r["level"] for r in rows[1:]]
    findings = [
        finding("Traced-geodesic length defect on icospheres converges at second order", "numerical", p_length,
                {"generator": generator("icosphere", levels=levels),
                 "checks": [check("fitted order minus the inscribed-metric prediction 2", p_length - 2.0, 0.2,
                                  kind="self_convergence")]},
                unit="order", tolerance=RATE),
        finding("Traced-geodesic endpoint error on icospheres converges at least at first order", "numerical",
                {"endpoint_order": p_endpoint, "cross_track_order": p_cross,
                 "finest_mean_endpoint": rows[-1]["mean_endpoint"]},
                {"generator": generator("icosphere", levels=levels),
                 "checks": [check("fitted endpoint order", p_endpoint, 0.9, "ge", "self_convergence")]},
                tolerance=RATE),
        finding("Prism-cylinder helix endpoint error follows L cos(alpha) pi^2 / (6 n^2)", "numerical",
                {"order": p_helix, "scaled_to_leading_constant": scaled},
                {"generator": generator("cylinder_mesh", n=[r["n"] for r in crow]),
                 "checks": [check("n^2 error / (L cos(alpha) pi^2 / 6) - 1 at the finest n", scaled - 1.0, 0.02),
                            check("fitted order - 2", p_helix - 2.0, 0.05, kind="self_convergence")]},
                tolerance=RATE),
        finding("Edge-graph Dijkstra distance keeps a relative-error floor under refinement", "numerical",
                {"levels": [r["level"] for r in drows], "max_relative_error": edge},
                {"generator": generator("icosphere", levels=[r["level"] for r in drows]),
                 "checks": [check("max relative error at the finest level", edge[-1], 0.2, "ge"),
                            check("finest minus second level error (no decrease)", edge[-1] - edge[1], 0.0, "ge")]},
                tolerance=TIGHT,
                counterexample={"statement": "Edge-graph shortest paths converge to the geodesic distance under "
                                             "mesh refinement",
                                "witness": {"mesh": "icosphere levels 1-4, source vertex 0",
                                            "max_relative_error": edge}}),
        finding("Steiner graphs with a fixed number of points per edge keep a relative-error floor", "numerical",
                {"k1": steiner[1], "k3": steiner[3]},
                {"generator": generator("steiner_graph", ks=[1, 3], levels=[r["level"] for r in drows]),
                 "checks": [check("k=3 max relative error at the finest level", steiner[3][-1], 0.005, "ge"),
                            check("k=3 finest minus coarsest-but-one error", steiner[3][-1] - steiner[3][1], 0.0, "ge")]},
                tolerance=TIGHT,
                counterexample={"statement": "Refining the mesh alone makes a Steiner-graph distance with fixed k "
                                             "converge to the geodesic distance",
                                "witness": {"k": 3, "max_relative_error": steiner[3]}}),
        finding("Heat-method distance error decreases under refinement at about first order", "numerical", p_heat,
                {"generator": generator("heat_distance", levels=[r["level"] for r in heat_rows], t="h^2"),
                 "checks": [check("fitted order of the max abs error", p_heat, 0.8, "ge", "self_convergence")]},
                unit="order", tolerance=RATE),
        _physical("The observed convergence orders transfer to meshes reconstructed from real scans"),
    ]
    result = (f"Sphere (levels 2-6): length-defect order {p_length:.2f}, endpoint order {p_endpoint:.2f} "
              f"(cross-track {p_cross:.2f}), finest mean endpoint error {rows[-1]['mean_endpoint']:.2e} rad. "
              f"Cylinder: order {p_helix:.3f}, n^2 error / predicted constant = {scaled:.4f}. Edge Dijkstra max "
              f"relative error {', '.join(f'{e:.3f}' for e in edge)} (levels {drows[0]['level']}-{drows[-1]['level']}); "
              f"Steiner k=3 {', '.join(f'{e:.4f}' for e in steiner[3])}; heat-method order {p_heat:.2f}.")
    return {"state": "completed", "findings": findings, "fields": fields(
        "Straightest mesh geodesics converge to smooth geodesics; the length (metric) error is O(h^2) on inscribed "
        "meshes, the lateral error is between O(h) and O(h^2), and edge-graph distances do not converge.",
        "Inscribed icosphere: chord/arc metric error O(h^2). Lateral error of a straightest geodesic is driven by the "
        "imbalance of vertex curvature point masses on either side of the path (a discrepancy sum), so its order is "
        "not a single power law. Prism cylinder: development circumference 2nR sin(pi/n), giving error "
        "L cos(alpha) (pi/(n sin(pi/n)) - 1) ~ L cos(alpha) pi^2 / (6 n^2).",
        ["icosphere levels 1-6", "prism cylinder n=8..128, alpha=0.5, L=3", "six declared sphere geodesics, L=2",
         "distances from vertex 0 on icosphere levels 1-4"],
        "Endpoint angle on the unit sphere after radial projection; helix error on the cylinder surface; relative "
        "distance error against the great-circle distance.",
        "Length defect order 2; cylinder constant L cos(alpha) pi^2/6; graph errors bounded below by a nonzero floor.",
        "Refine, trace, fit log-log slopes over levels 2-6 (the coarsest level is preasymptotic); compare graph and "
        "heat-method distances against great-circle distances.",
        result,
        "Deterministic; fitted orders depend on the chosen levels and geodesics (six directions). The lateral-error "
        "order is an empirical slope, not a proved rate.",
        ["preasymptotic coarse levels (level 1 excluded from fits)", "vertex hits during refinement (none)",
         "sign of the length defect (inscribed polyhedron is shorter)", "graph metrication plateau"],
        ["Endpoint-error order is empirical and depends on how geodesic directions align with lattice rows.",
         "Heat-method time step t = h^2 as recommended by Crane et al.; other choices change the rate."],
        "T040: compare finite-difference mesh Jacobi fields with the smooth transfer and locate where curvature "
        "estimates fail.")}


# ---------------------------------------------------------------- T040
@task("T040", changed_files=FILES, regression_tests=(
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
    tiny = by_delta[min(by_delta)]
    p_wide = S.fitted_order([r["h"] for r in wide], [r["mean_abs_error"] for r in wide])
    flat_dev = max(r["max_flat_deviation"] for r in tiny)
    identical = sum(r["identical_face_sequences"] for r in tiny)
    pairs = sum(r["pairs"] for r in tiny)
    flat_error = study["flat_value"] - smooth["closed_form"]
    sphere = curvature["sphere"]
    limit = curvature["valence5_limit"]
    k5 = [r["valence5_value"] for r in sphere]
    dev = [abs(v - limit) for v in k5]
    p_k5 = S.fitted_order([r["h"] for r in sphere[3:]], dev[3:])
    voronoi = [r["voronoi_valence5_error"] for r in sphere]
    gb = max(max(abs(r["gauss_bonnet_residual"]) for r in sphere),
             max(abs(r["gauss_bonnet_residual"]) for r in curvature["torus"]))
    trows = curvature["torus"]
    p_torus = S.fitted_order([r["h"] for r in trows], [r["max_error"] for r in trows])
    ctx.artifact_json("jacobi-and-curvature.json", jsonable({"jacobi": study, "curvature": curvature}))
    series = [(f"FD Jacobi delta={d:g}", [r["h"] for r in rs], [r["mean_abs_error"] for r in rs])
              for d, rs in sorted(by_delta.items(), reverse=True)]
    series += [("|K - 1| valence 5", [r["h"] for r in sphere], [abs(v - 1) for v in k5]),
               ("|K - limit| valence 5", [r["h"] for r in sphere], dev),
               ("Voronoi valence 5", [r["h"] for r in sphere], voronoi),
               ("torus max |K error|", [r["h"] for r in trows], [r["max_error"] for r in trows])]
    ctx.artifact_text("jacobi-curvature.svg", svg.line_plot(
        series, title="Mesh Jacobi fields and angle-defect curvature", xlabel="mean edge length h",
        ylabel="error", logx=True, logy=True))
    findings = [
        finding("The smooth ciw.lab.jacobi heading column equals sin(L) on the unit sphere", "numerical",
                smooth["max_error"], {"generator": generator("jacobi.transfer", steps=200, length=study["length"]),
                                      "checks": [check("closed form sin(L)", smooth["max_error"], 1e-8)]},
                tolerance={"abs": 1e-9, "rel": 0.0}),
        finding("Finite-difference mesh Jacobi fields with a 0.1 rad heading offset converge to the smooth field",
                "numerical", {"order": p_wide, "mean_abs_error": [r["mean_abs_error"] for r in wide],
                              "levels": [r["level"] for r in wide]},
                {"generator": generator("icosphere", levels=[r["level"] for r in wide], delta=0.1),
                 "checks": [check("fitted order", p_wide, 1.0, "ge", "self_convergence"),
                            check("mean abs error at the finest level", wide[-1]["mean_abs_error"], 0.01, "le")]},
                tolerance=RATE),
        finding("At fixed mesh a 1e-5 heading offset gives the flat Jacobi value L instead of sin(L)", "numerical",
                {"max_flat_deviation": flat_dev, "identical_face_sequences": identical, "pairs": pairs,
                 "error_vs_smooth": flat_error},
                {"generator": generator("icosphere", levels=[r["level"] for r in tiny], delta=min(by_delta)),
                 "checks": [check("|j_FD - L| (flat development prediction)", flat_dev, 1e-6, kind="analytic"),
                            check("pairs with differing face sequences", pairs - identical, 0.0, kind="invariant")]},
                tolerance=TIGHT,
                counterexample={"statement": "On a fixed mesh the finite-difference Jacobi field converges to the "
                                             "smooth Jacobi field as the perturbation tends to zero",
                                "witness": {"delta": min(by_delta), "levels": [r["level"] for r in tiny],
                                            "j_fd": study["flat_value"], "j_smooth": smooth["closed_form"]}}),
        finding("Angle-defect curvature at valence-5 icosphere vertices converges to (4.5 - 1.5 sqrt 5) K, not K",
                "numerical", {"valence5_values": k5, "predicted_limit": limit, "order_to_limit": p_k5},
                {"derivation": "Regular valence-n star on an umbilic surface: defect ~ Voronoi cell area, ratio "
                               "3 / (4 cos^2(pi/n)); docs/lab/MESH_GEODESICS.md",
                 "checks": [check("finest valence-5 value - limit", k5[-1] - limit, 1e-4),
                            check("order of |K5 - limit| - 2 (levels 4-6)", p_k5 - 2.0, 0.2, kind="self_convergence")]},
                tolerance=TIGHT,
                counterexample={"statement": "The angle defect over one third of the incident area converges "
                                             "pointwise to the Gaussian curvature under refinement",
                                "witness": {"mesh": "icosphere, 12 valence-5 vertices", "limit": limit,
                                            "finest_value": k5[-1]}}),
        finding("With the mixed Voronoi area the valence-5 curvature estimate converges", "numerical", voronoi,
                {"generator": generator("mixed_voronoi_area", levels=[r["level"] for r in sphere]),
                 "checks": [check("finest valence-5 Voronoi error", voronoi[-1], 1e-3, "le"),
                            check("finest minus coarsest error", voronoi[-1] - voronoi[0], 0.0, "le")]},
                tolerance=TIGHT),
        finding("Discrete Gauss-Bonnet holds: angle defects sum to 4 pi on icospheres and 0 on tori", "numerical", gb,
                {"derivation": "Discrete Gauss-Bonnet: sum of angle defects = 2 pi chi for a closed mesh",
                 "checks": [check("max |sum of defects - 2 pi chi|", gb, 1e-9, kind="invariant")]},
                tolerance={"abs": 1e-9, "rel": 0.0}),
        finding("Angle-defect curvature on regular torus grids converges at second order to the sign-changing K",
                "numerical", {"order": p_torus, "max_error": [r["max_error"] for r in trows]},
                {"generator": generator("torus_mesh", n_theta=[r["n_theta"] for r in trows]),
                 "checks": [check("fitted order - 2", p_torus - 2.0, 0.2, kind="self_convergence")]},
                tolerance=RATE),
        _physical("Angle-defect curvature of a scanned mesh estimates the Gaussian curvature of the physical part"),
    ]
    wide_errors = ", ".join(format(r["mean_abs_error"], ".2e") for r in wide)
    result = (f"FD Jacobi (delta=0.1): mean error {wide_errors} "
              f"(levels {wide[0]['level']}-{wide[-1]['level']}, order {p_wide:.2f}); delta={min(by_delta):g}: "
              f"{identical}/{pairs} pairs share face sequences and give j = L = {study['flat_value']} (deviation "
              f"{flat_dev:.1e}), error {flat_error:.4f} at every level. Valence-5 angle-defect curvature "
              f"{k5[-1]:.7f} vs predicted limit {limit:.7f} (order {p_k5:.2f}); Voronoi valence-5 error "
              f"{voronoi[-1]:.1e}; torus order {p_torus:.2f}; Gauss-Bonnet residual {gb:.1e}.")
    return {"state": "completed", "findings": findings, "fields": fields(
        "Mesh Jacobi fields and angle-defect curvature approximate the smooth ones only in the right order of limits "
        "and away from irregular vertices.",
        "Smooth: j'' + K j = 0, j_head(s) = sin(s) for K = 1. Mesh: curvature is concentrated at vertices (cone "
        "points); two geodesics are rotated relative to each other only when a vertex lies between them, so the "
        "finite-difference field depends on delta/h. Angle defect over A/3 equals Voronoi/(A/3) times K on an "
        "umbilic surface, giving the limit 3 / (4 cos^2(pi/n)) at regular valence-n stars.",
        ["icosphere levels 1-6", "torus grids n_theta=8..64 (R=2, r=1)", "six declared geodesics, L=2",
         "heading offsets 0.1, 0.03, 0.01, 1e-5 rad"],
        "Central difference |X+(L) - X-(L)| / (2 sin delta), exact for the smooth unit sphere; vertex curvature "
        "estimates against K = 1 or the torus closed form.",
        "Delta fixed, h -> 0: convergence; h fixed, delta -> 0: the flat value L. Valence-5 limit 4.5 - 1.5 sqrt 5.",
        "Paired traces at +-delta on each level; angle defect with barycentric and mixed Voronoi areas; torus grids.",
        result,
        "Deterministic. The FD error at intermediate delta mixes the delta/h regime and vertex straddling, so it is "
        "not monotone in delta; only the two limits are asserted.",
        ["pairs straddling a vertex (jumps)", "identical face sequences (flat regime)", "valence-5 versus valence-6 "
         "vertices", "Voronoi versus barycentric area", "Gauss-Bonnet exactness"],
        ["Pointwise convergence at valence-6 vertices near the valence-5 ones is not established (the max error "
         "over valence-6 vertices also plateaus near 2.5e-3).",
         "The torus grid is regular; irregular torus meshes are not tested here."],
        "T041: quantify how mesh quality (jitter, slivers, anisotropy) changes geodesic and curvature errors.")}


# ---------------------------------------------------------------- T041
@task("T041", changed_files=FILES, regression_tests=(
    f"{TESTS}::test_schwarz_lantern_counterexample",
    f"{TESTS}::test_quality_task_report"))
def mesh_quality_effects(ctx):
    quality = _memo(ctx, "quality", S.quality_study)
    lantern = _memo(ctx, "lantern", S.lantern_study)
    plane = _memo(ctx, "plane", S.plane_study)
    jitter = quality["jitter"]
    valid = [r for r in jitter if not r["issues"]]
    folded = [r for r in jitter if "folded_face" in r["issues"]]
    amplitudes = sorted({r["amplitude"] for r in valid})
    averaged = []
    for a in amplitudes:
        group = [r for r in valid if r["amplitude"] == a]
        averaged.append({"amplitude": a, "meshes": len(group),
                         "curvature_rms": float(np.mean([r["curvature_rms"] for r in group])),
                         "min_angle_deg": float(np.mean([r["min_angle_deg"] for r in group])),
                         "max_radius_ratio": float(np.mean([r["max_radius_ratio"] for r in group])),
                         "geodesic_mean": float(np.mean([r["geodesic_mean"] for r in group]))})
    rms_steps = [b["curvature_rms"] - a["curvature_rms"] for a, b in zip(averaged, averaged[1:])]
    angle_steps = [a["min_angle_deg"] - b["min_angle_deg"] for a, b in zip(averaged, averaged[1:])]
    regular = next(r for r in jitter if r["amplitude"] == 0)
    better = [r for r in valid if r["geodesic_mean"] < regular["geodesic_mean"] and r["min_angle_deg"] < regular["min_angle_deg"]]
    witness = min(better, key=lambda r: r["geodesic_mean"]) if better else None
    uv = quality["latitude_longitude"]
    uv_better = [r for r in uv if not r["issues"] and r["curvature_rms"] < regular["curvature_rms"]
                 and r["min_angle_deg"] < regular["min_angle_deg"]]
    uv_witness = min(uv_better, key=lambda r: r["curvature_rms"]) if uv_better else None
    everything = [r for r in valid + [r for r in uv if not r["issues"]]]
    correlations = {name: _spearman([r["min_angle_deg"] for r in everything], [r[name] for r in everything])
                    for name in ("curvature_rms", "curvature_max", "geodesic_mean", "area_error")}
    lrows = lantern["rows"]
    last = lrows[-1]
    hausdorff_gap = max(abs(r["hausdorff_sampled"] - r["hausdorff_closed_form"]) for r in lrows)
    area_gap = max(abs(r["area_closed_form_error"]) for r in lrows)
    height_gap = max(abs(r["traced_height"] - r["height_closed_form"]) for r in lrows)
    flat = max(r["max_interior_curvature"] for r in lrows)
    plane_error = max(r["max_trace_error"] for r in plane["rows"])
    graph_excess = [r["diagonal_graph_excess"] for r in plane["rows"]]
    ctx.artifact_json("quality.json", jsonable({"quality": quality, "seed_averaged": averaged,
                                                "spearman_min_angle": correlations, "plane": plane}))
    ctx.artifact_json("schwarz-lantern.json", jsonable(lantern))
    ctx.artifact_text("schwarz-lantern.svg", svg.line_plot([
        ("area / (2 pi R H)", [r["n"] for r in lrows], [r["area_ratio"] for r in lrows]),
        ("traced height / H", [r["n"] for r in lrows], [r["traced_height"] for r in lrows]),
        ("Hausdorff distance", [r["n"] for r in lrows], [r["hausdorff_sampled"] for r in lrows]),
        ("total |mean curvature| / pi", [r["n"] for r in lrows], [r["total_abs_mean_curvature"] / math.pi for r in lrows])],
        title=f"Schwarz lantern, m = {lantern['q']:g} n^2", xlabel="vertices per ring n", ylabel="value",
        logx=True, logy=True))
    ctx.artifact_text("jitter.svg", svg.line_plot([
        ("curvature RMS error", [a["amplitude"] for a in averaged], [a["curvature_rms"] for a in averaged]),
        ("geodesic mean error", [a["amplitude"] for a in averaged], [a["geodesic_mean"] for a in averaged]),
        ("min angle / 1000 deg", [a["amplitude"] for a in averaged], [a["min_angle_deg"] / 1000 for a in averaged])],
        title="Tangential jitter at fixed vertex count (642)", xlabel="jitter amplitude / h", ylabel="value"))
    folded_checks = [refusal(f"validator on {r['mesh']}", "folded_face", "folded_face" if "folded_face" in r["issues"]
                             else (r["issues"][0] if r["issues"] else None)) for r in folded]
    findings = [
        finding("Seed-averaged curvature RMS error grows and minimum angle falls with tangential jitter", "numerical",
                averaged, {"generator": generator("icosphere", level=3, jitter=amplitudes, seeds=[1, 2, 3], seed=S.SEED),
                           "checks": [check("smallest increase of curvature RMS between amplitudes", min(rms_steps), 0.0,
                                            "ge", "invariant"),
                                      check("smallest decrease of min angle between amplitudes", min(angle_steps), 0.0,
                                            "ge", "invariant")]},
                tolerance=TIGHT),
        finding("Jittered meshes with folded faces are refused and their unguarded curvature error exceeds 1",
                "numerical", {"meshes": [r["mesh"] for r in folded], "curvature_rms": [r["curvature_rms"] for r in folded],
                              "inverted_faces": [r["inverted_faces"] for r in folded]},
                {"generator": generator("icosphere", level=3, jitter="folded", seed=S.SEED),
                 "checks": folded_checks + [check("smallest unguarded curvature RMS among folded meshes",
                                                  min(r["curvature_rms"] for r in folded), 1.0, "ge")]},
                tolerance=TIGHT),
        finding("A mesh with a smaller minimum angle can have a smaller geodesic error", "numerical",
                {"regular": {k: regular[k] for k in ("mesh", "min_angle_deg", "geodesic_mean")},
                 "witness": {k: witness[k] for k in ("mesh", "min_angle_deg", "geodesic_mean")}},
                {"generator": generator("icosphere", level=3, jitter=witness["amplitude"], seed=S.SEED + witness["seed"]),
                 "checks": [check("regular minus witness geodesic error", regular["geodesic_mean"] - witness["geodesic_mean"],
                                  0.0, "ge"),
                            check("regular minus witness min angle", regular["min_angle_deg"] - witness["min_angle_deg"],
                                  0.0, "ge")]},
                tolerance=TIGHT,
                counterexample={"statement": "A larger minimum angle implies a smaller geodesic error",
                                "witness": {"better_quality": regular["mesh"], "worse_quality": witness["mesh"]}}),
        finding("Across mesh families a much smaller minimum angle can come with a smaller curvature RMS error",
                "numerical", {"regular": {k: regular[k] for k in ("mesh", "min_angle_deg", "curvature_rms", "curvature_max")},
                              "witness": {k: uv_witness[k] for k in ("mesh", "min_angle_deg", "curvature_rms", "curvature_max")}},
                {"generator": generator("uv_sphere", n_lat=uv_witness["n_lat"], n_lon=uv_witness["n_lon"]),
                 "checks": [check("regular minus witness curvature RMS", regular["curvature_rms"] - uv_witness["curvature_rms"],
                                  0.0, "ge"),
                            check("regular minus witness min angle (deg)", regular["min_angle_deg"] - uv_witness["min_angle_deg"],
                                  20.0, "ge")]},
                tolerance=TIGHT,
                counterexample={"statement": "Minimum angle orders curvature error across mesh families",
                                "witness": {"better_quality": regular["mesh"], "worse_quality": uv_witness["mesh"],
                                            "note": "the max error still favours the icosphere"}}),
        finding("Spearman rank correlation of minimum angle with each error over valid 642-vertex meshes", "numerical",
                correlations, {"generator": generator("icosphere+uv_sphere", meshes=len(everything), seed=S.SEED)},
                tolerance=TIGHT),
        finding("Schwarz lantern meshes converge in Hausdorff distance but not in area", "numerical",
                {"n": [r["n"] for r in lrows], "hausdorff": [r["hausdorff_sampled"] for r in lrows],
                 "area_ratio": [r["area_ratio"] for r in lrows], "limit_area_ratio": lantern["limit_area_ratio"]},
                {"derivation": "A = 2 n R sin(pi/n) sqrt(H^2 + m^2 R^2 (1 - cos(pi/n))^2), d_H = R (1 - cos(pi/n))",
                 "checks": [check("max |sampled Hausdorff - closed form|", hausdorff_gap, 1e-12),
                            check("max |mesh area - closed form|", area_gap, 1e-9),
                            check("area ratio at the finest n - 1", last["area_ratio"] - 1.0, 0.5, "ge")]},
                tolerance=TIGHT,
                counterexample={"statement": "Hausdorff convergence of a mesh to a surface implies convergence of area",
                                "witness": {"q": lantern["q"], "n": last["n"], "bands": last["bands"],
                                            "hausdorff": last["hausdorff_sampled"], "area_ratio": last["area_ratio"]}}),
        finding("Schwarz lantern intrinsic height does not converge to the cylinder height", "numerical",
                {"traced_height": [r["traced_height"] for r in lrows], "height": 1.0},
                {"derivation": "The lantern develops to a flat strip of height sqrt(H^2 + m^2 R^2 (1 - cos(pi/n))^2)",
                 "checks": [check("max |traced height - development height|", height_gap, 1e-10),
                            check("traced height at the finest n - 1.5 H", last["traced_height"] - 1.5, 0.0, "ge")]},
                tolerance=TIGHT,
                counterexample={"statement": "Hausdorff convergence of a mesh implies convergence of its geodesic "
                                             "distances",
                                "witness": {"n": last["n"], "traced_height": last["traced_height"], "H": 1.0}}),
        finding("Lantern angle defects vanish while total absolute mean curvature and normal tilt do not converge",
                "numerical", {"max_interior_angle_defect_curvature": flat,
                              "total_abs_mean_curvature": [r["total_abs_mean_curvature"] for r in lrows],
                              "smooth_total_abs_mean_curvature": lantern["smooth_total_abs_mean_curvature"],
                              "max_normal_tilt_deg": [r["max_normal_tilt_deg"] for r in lrows],
                              "limit_tilt_deg": lantern["limit_tilt_deg"]},
                {"derivation": "Isosceles lantern faces give angle sums 2 alpha + 4 beta = 2 pi at interior vertices",
                 "checks": [check("max interior |K| (angle defect)", flat, 1e-9, kind="analytic"),
                            check("total |H| at finest n / smooth value", last["total_abs_mean_curvature"]
                                  / lantern["smooth_total_abs_mean_curvature"], 5.0, "ge")]},
                tolerance=TIGHT,
                counterexample={"statement": "Hausdorff convergence of a mesh implies convergence of its normals and "
                                             "curvature",
                                "witness": {"n": last["n"], "tilt_deg": last["max_normal_tilt_deg"],
                                            "total_abs_mean_curvature": last["total_abs_mean_curvature"]}}),
        finding("A strongly pleated lantern (m = n^2) is refused as folded", "numerical", lantern["folded"],
                {"generator": generator("cylinder_mesh", lantern=True, n=lantern["folded"]["n"], q=lantern["folded"]["q"]),
                 "checks": [refusal("validator on the m = n^2 lantern", "folded_face",
                                    lantern["folded"]["issues"][0] if lantern["folded"]["issues"] else None)]}),
        finding("Straightest geodesics on planar meshes are exact at any triangle quality, graph distances are not",
                "numerical", {"max_trace_error": plane_error, "diagonal_graph_excess": graph_excess,
                              "min_angle_deg": [r["min_angle_deg"] for r in plane["rows"]]},
                {"generator": generator("plane_mesh", shears=[r["shear"] for r in plane["rows"]]),
                 "checks": [check("max trace error over all shears", plane_error, 1e-12),
                            check("spread of diagonal graph excess across shears", max(graph_excess) - min(graph_excess),
                                  0.01, "ge")]},
                tolerance={"abs": 1e-12, "rel": 1e-6}),
        _physical("A minimum-angle or radius-ratio threshold certifies a scanned mesh for production metrology",
                  "production_acceptance"),
    ]
    result = (f"Jitter (seed-averaged, valid meshes): curvature RMS "
              + ", ".join(f"{a['curvature_rms']:.4f}@{a['amplitude']:g}" for a in averaged)
              + f"; {len(folded)} jittered meshes folded and refused. Min angle vs geodesic error counterexample: "
              f"{regular['mesh']} ({regular['min_angle_deg']:.1f} deg, {regular['geodesic_mean']:.4f}) vs "
              f"{witness['mesh']} ({witness['min_angle_deg']:.1f} deg, {witness['geodesic_mean']:.4f}). "
              f"{uv_witness['mesh']} has min angle {uv_witness['min_angle_deg']:.1f} deg and curvature RMS "
              f"{uv_witness['curvature_rms']:.4f} < {regular['curvature_rms']:.4f}. Lantern (m={lantern['q']:g} n^2, "
              f"n={last['n']}): Hausdorff {last['hausdorff_sampled']:.2e}, area ratio {last['area_ratio']:.4f} "
              f"(limit {lantern['limit_area_ratio']:.4f}), traced height {last['traced_height']:.4f}, interior K "
              f"{flat:.1e}. Planar traces exact to {plane_error:.1e} for min angles down to "
              f"{min(r['min_angle_deg'] for r in plane['rows']):.1f} deg.")
    return {"state": "completed", "findings": findings, "fields": fields(
        "At fixed vertex count, worse triangle quality increases curvature error within a mesh family, but quality "
        "metrics do not order errors across families; Hausdorff convergence does not imply convergence of area, "
        "distances, normals or mean curvature.",
        "Quality: minimum angle and radius ratio R_circ / (2 r_in). Schwarz lantern with m = q n^2 bands: face "
        "height sqrt((H/m)^2 + R^2 (1 - cos(pi/n))^2), so area and developed height tend to sqrt(1 + (pi^2 q R / 2H)^2) "
        "times their cylinder values while d_H = R (1 - cos(pi/n)) -> 0.",
        ["icosphere level 3 (642 vertices), tangential jitter 0-0.3 h, 3 seeds", "latitude-longitude spheres with 642 "
         "vertices (20x32, 10x64, 40x16, twisted)", "Schwarz lanterns q=0.25, n=4..64; q=1, n=8",
         "sheared planar grids"],
        "Curvature error against K = 1, geodesic endpoint error against great circles, area error; lantern area, "
        "developed height, sampled Hausdorff distance, normal tilt and total absolute mean curvature.",
        "Monotone trend within the jitter family; lantern closed forms for area, height and Hausdorff distance; "
        "angle defects of the lantern are exactly zero.",
        "Measure quality metrics and errors for each mesh; fold detection by the validator; trace lantern geodesics "
        "from the bottom boundary to the top boundary.",
        result,
        "Deterministic given the seeds; with three seeds per amplitude the monotone trend is a seed-average and a "
        "different seed set could reorder neighbouring amplitudes.",
        ["folded (inverted) faces from large jitter", "degenerate triangles", "vertex hits on skewed meshes (none)",
         "pole valence on latitude-longitude spheres", "lantern pleat folding (refused at q = 1)"],
        ["Quality metrics are summarised by the worst triangle; per-region error attribution is not attempted.",
         "Only tangential jitter is studied here; normal noise is the subject of T043."],
        "T042: name the refusal states for invalid or incomplete surface data (including folded faces).")}


def _spearman(x, y) -> float:
    rx = np.argsort(np.argsort(np.asarray(x, dtype=float))).astype(float)
    ry = np.argsort(np.argsort(np.asarray(y, dtype=float))).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


# ---------------------------------------------------------------- T042
@task("T042", changed_files=FILES, regression_tests=(
    f"{TESTS}::test_every_defect_has_a_named_refusal",
    f"{TESTS}::test_refusal_task_report"))
def mesh_refusal_states(ctx):
    study = _memo(ctx, "refusals", S.refusal_study)
    cases = study["cases"]
    control_issues = sum(len(c["issues"]) for c in study["controls"])
    expected_multiple = ["nonfinite_vertex", "inconsistent_orientation", "unreferenced_vertex"]
    partial_gap = abs(study["boundary_partial_length"] - study["boundary_analytic_length"])
    ctx.artifact_json("refusal-catalogue.json", jsonable(study))
    findings = [
        finding("Every declared surface-data defect is refused with its named code", "computational_pipeline",
                {c["case"]: c["observed"] for c in cases},
                {"generator": generator("refusal_study", cases=len(cases)),
                 "checks": [refusal(f"{c['stage']}: {c['case']}", c["expected"], c["observed"]) for c in cases]}),
        finding("Valid control meshes pass validation without issues", "computational_pipeline", control_issues,
                {"generator": generator("controls", meshes=[c["mesh"] for c in study["controls"]]),
                 "checks": [check("issues reported on valid meshes", control_issues, 0.0, kind="invariant")]},
                tolerance={"abs": 0, "rel": 0}),
        finding("A mesh with several defects reports all of them in declared order", "computational_pipeline",
                study["multiple_defects"],
                {"generator": generator("octahedron", defects=expected_multiple),
                 "checks": [check("mismatches with the expected ordered codes",
                                  0.0 if study["multiple_defects"] == expected_multiple else 1.0, 0.0,
                                  kind="exact_arithmetic")]}),
        finding("A geodesic stopped at a boundary retains its partial length", "numerical",
                study["boundary_partial_length"],
                {"generator": generator("plane_mesh", nx=4, ny=4),
                 "checks": [check("|partial length - distance to the boundary|", partial_gap, 1e-12)]},
                unit="normalized length", tolerance={"abs": 1e-12, "rel": 0.0}),
        finding("Curvature and normals evaluated on an unvalidated zero-area face are nonfinite", "numerical",
                {"nonfinite_curvatures": study["unguarded_nonfinite_curvatures"],
                 "nonfinite_normals": study["unguarded_nonfinite_normals"]},
                {"generator": generator("TriMesh", validation="bypassed"),
                 "checks": [check("nonfinite curvature values", study["unguarded_nonfinite_curvatures"], 1.0, "ge"),
                            check("nonfinite normal components", study["unguarded_nonfinite_normals"], 1.0, "ge")]},
                tolerance={"abs": 0, "rel": 0},
                counterexample={"statement": "Curvature and normal formulas can be evaluated safely on unvalidated "
                                             "meshes",
                                "witness": {"mesh": "square plus a collinear face", "defect": "zero-area face"}}),
        _physical("The refusal catalogue covers every defect present in real scanned surface data"),
    ]
    codes = sorted({c["expected"] for c in cases})
    result = (f"{sum(c['observed'] == c['expected'] for c in cases)}/{len(cases)} defect cases refused with the "
              f"expected code ({len(codes)} distinct codes); {control_issues} issues on {len(study['controls'])} valid "
              f"controls; multiple-defect report {study['multiple_defects']}; boundary partial length "
              f"{study['boundary_partial_length']:.15g} (analytic {study['boundary_analytic_length']:.15g}); unguarded "
              f"zero-area evaluation gave {study['unguarded_nonfinite_curvatures']} nonfinite curvature values.")
    return {"state": "completed", "findings": findings, "fields": fields(
        "Each class of invalid or incomplete surface data can be detected before geometry is computed and reported "
        "under a stable name, and valid meshes pass.",
        "Validation order: " + ", ".join(S.G.MESH_CODES) + ". Tracing refusals: " + ", ".join(S.G.TRACE_CODES)
        + ". Query refusals: " + ", ".join(S.G.QUERY_CODES) + ".",
        [f"{c['case']} ({c['stage']})" for c in cases],
        "No physical observation; the refusal code (or its absence) for each constructed input.",
        "Exactly the expected code for every defect; no code for valid controls; partial traces keep their length.",
        "Construct each defect explicitly, validate or trace, and compare the observed code with the declared code.",
        result,
        "Exact (categorical); thresholds are declared: zero area at 2A/l_max^2 <= 1e-12, vertex hit at edge "
        "parameter <= 1e-9, fold at adjacent normal dot <= -0.9.",
        codes + ["unguarded evaluation on zero-area faces"],
        ["Thresholds are relative and unitless; scans with very different scales may need other thresholds.",
         "Self-intersections between non-adjacent faces are not detected."],
        "T043: propagate vertex noise to geodesic length, normals and curvature on validated meshes.")}


# ---------------------------------------------------------------- T043
@task("T043", changed_files=FILES, regression_tests=(
    f"{TESTS}::test_linearization_matches_monte_carlo_and_breaks",
    f"{TESTS}::test_uncertainty_task_report"))
def mesh_vertex_uncertainty(ctx):
    study = _memo(ctx, "uncertainty", S.uncertainty_study)
    scaling = _memo(ctx, "scaling", S.scaling_study)
    refinement = _memo(ctx, "refinement-noise", S.refinement_noise_study)
    items = {o["observable"]: o for o in study["observables"]}
    distance = items["marker geodesic distance"]
    normals = [o for name, o in items.items() if name.startswith("vertex normal")]
    curvature = [o for name, o in items.items() if name.startswith("angle-defect")]
    distance_dev = max(abs(r["ratio"] - 1) for r in distance["rows"] if r["sigma"] <= 1e-2)
    normal_dev = max(abs(r["ratio"] - 1) for o in normals for r in o["rows"])
    small = [r for o in curvature for r in o["rows"] if r["sigma"] <= 1e-3]
    large = [r for o in curvature for r in o["rows"] if r["sigma"] >= 1e-2]
    curvature_dev = max(abs(r["ratio"] - 1) for r in small)
    breakdown = min(r["ratio"] for r in large)
    invalid = {str(r["sigma"]): r["invalid_fraction"] for r in distance["rows"]}
    slopes = scaling["slopes"]
    slope_k = [v for k, v in slopes.items() if k.startswith("angle-defect")]
    slope_n = [v for k, v in slopes.items() if k.startswith("vertex normal")]
    slope_d = slopes["marker geodesic distance"]
    rrows = refinement["rows"]
    ctx.artifact_json("vertex-uncertainty.json", jsonable({"propagation": study, "scaling": scaling,
                                                           "refinement": refinement}))
    ctx.artifact_text("linearization-ratio.svg", svg.line_plot(
        [(o["observable"], [r["sigma"] for r in o["rows"]], [r["ratio"] for r in o["rows"]])
         for o in study["observables"]],
        title="Monte Carlo / linearized variance", xlabel="vertex noise sigma", ylabel="ratio", logx=True))
    h2 = study["h_squared"]
    findings = [
        finding("Linearized vertex-noise propagation matches Monte Carlo for the marker geodesic distance",
                "numerical", {"ratios": [r["ratio"] for r in distance["rows"]], "sigmas": [r["sigma"] for r in distance["rows"]],
                              "gradient_norm": distance["gradient_norm"], "strip_invalid_fraction": invalid},
                {"generator": generator("icosphere", level=study["level"], samples=study["samples"], seed=S.SEED),
                 "checks": [check("max |MC / linear variance - 1| for sigma <= 1e-2", distance_dev, 0.1,
                                  kind="self_convergence")]},
                tolerance=TIGHT),
        finding("Linearized vertex-noise propagation matches Monte Carlo for vertex normals at every tested sigma",
                "numerical", {o["observable"]: [r["ratio"] for r in o["rows"]] for o in normals},
                {"generator": generator("icosphere", level=study["level"], samples=study["samples"], seed=S.SEED),
                 "checks": [check("max |MC / linear mean-square angle - 1|", normal_dev, 0.1, kind="self_convergence")]},
                tolerance=TIGHT),
        finding("Linearized propagation matches Monte Carlo for angle-defect curvature when sigma <= 1e-3", "numerical",
                {o["observable"]: [r["ratio"] for r in o["rows"]] for o in curvature},
                {"generator": generator("icosphere", level=study["level"], samples=study["samples"], seed=S.SEED),
                 "checks": [check("max |MC / linear variance - 1| for sigma <= 1e-3", curvature_dev, 0.1,
                                  kind="self_convergence")]},
                tolerance=TIGHT),
        finding("First-order propagation underestimates angle-defect curvature variance at sigma = 1e-2", "numerical",
                {"min_ratio": breakdown, "sigma_over_h_squared": 1e-2 / h2,
                 "bias": [r["bias"] for r in large]},
                {"derivation": "The defect is quadratic in normal displacement (cone defect ~ pi e^2 / l^2); the "
                               "quadratic term matters once sigma R / h^2 is not small",
                 "checks": [check("smallest MC / linear variance ratio at sigma = 1e-2", breakdown, 1.3, "ge")]},
                tolerance=TIGHT,
                counterexample={"statement": "First-order (linearized) propagation of vertex noise is adequate for "
                                             "angle-defect curvature at sigma = 1e-2 on icosphere-3",
                                "witness": {"sigma": 1e-2, "h": study["h"], "min_ratio": breakdown}}),
        finding("Sensitivity to vertex noise scales as h^-2 for curvature, h^-1 for normals and h^0 for marker distance",
                "numerical", slopes,
                {"derivation": "Curvature = defect / (A/3) with A ~ h^2 and d(defect)/d(normal offset) ~ pi / R; "
                               "normal tilt ~ offset / h; marker distance moves with the barycentric marker vertices",
                 "checks": [check("worst |curvature slope + 2|", max(abs(s + 2) for s in slope_k), 0.25,
                                  kind="self_convergence"),
                            check("worst |normal slope + 1|", max(abs(s + 1) for s in slope_n), 0.2,
                                  kind="self_convergence"),
                            check("|marker distance slope|", slope_d, 0.35, kind="self_convergence")]},
                tolerance=RATE),
        finding("Under fixed vertex noise the curvature error grows as the mesh is refined", "numerical",
                {"levels": [r["level"] for r in rrows], "total_rms_error": [r["total_rms_error"] for r in rrows],
                 "discretization_error": [r["discretization_error"] for r in rrows]},
                {"generator": generator("icosphere", levels=[r["level"] for r in rrows], sigma=refinement["sigma"],
                                        samples=refinement["samples"], seed=S.SEED + 50),
                 "checks": [check("finest minus coarsest total RMS error", rrows[-1]["total_rms_error"]
                                  - rrows[0]["total_rms_error"], 0.0, "ge")]},
                tolerance=TIGHT,
                counterexample={"statement": "Refining the mesh reduces the error of angle-defect curvature",
                                "witness": {"sigma": refinement["sigma"], "coarse": rrows[0], "fine": rrows[-1]}}),
        _physical("Isotropic Gaussian vertex noise of the tested sigma describes the error of a real scanner",
                  "calibration"),
    ]
    result = (f"Icosphere-{study['level']} (h={study['h']:.4f}), {study['samples']} samples per sigma: distance "
              f"variance ratio within {distance_dev:.3f} of 1 (sigma<=1e-2; strip left in "
              + ", ".join(f"{float(v):.3f}@{k}" for k, v in invalid.items()) + " of samples); normal ratio within "
              f"{normal_dev:.3f}; curvature ratio within {curvature_dev:.3f} for sigma<=1e-3 but >= {breakdown:.2f} at "
              f"sigma=1e-2 (sigma/h^2={1e-2 / h2:.2f}). Sensitivity slopes: " + ", ".join(
                  f"{k}: {v:.2f}" for k, v in slopes.items()) + ". Total curvature RMS error at sigma="
              f"{refinement['sigma']:g}: " + ", ".join(f"{r['total_rms_error']:.3f}@L{r['level']}" for r in rrows) + ".")
    return {"state": "completed", "findings": findings, "fields": fields(
        "Gaussian vertex noise propagates to geodesic length and normals nearly linearly, while angle-defect "
        "curvature needs sigma << h^2 / R for linearization and becomes noisier as the mesh is refined.",
        "Linearization Var[f] = sigma^2 |grad f|^2 with a central finite-difference Jacobian (step 1e-6) over the "
        "vertices that affect f; seeded Monte Carlo with the same isotropic noise. Marker distance = planar distance "
        "after unfolding a fixed face strip with barycentric markers.",
        ["icosphere levels 2-5 (level 3 for Monte Carlo)", "sigma = 1e-4, 1e-3, 3e-3, 1e-2 (normalized units, R = 1)",
         "markers from the declared geodesic with the largest vertex margin, length 1", "valence-5 vertex 0 and the "
         "valence-6 vertex farthest from valence-5 vertices"],
        "Synthetic isotropic Gaussian displacement of every vertex; no scanner model beyond that.",
        "MC/linear variance ratio near 1 in the linear regime; ratio > 1 once second-order terms matter; exponents "
        "-2, -1, 0.",
        "For each observable and sigma: FD Jacobian, predicted variance, 4000 seeded samples, ratio and bias; "
        "gradient norms across levels; total error versus level at fixed sigma.",
        result,
        f"Monte Carlo relative standard error of a variance with {study['samples']} samples is about "
        f"{math.sqrt(2 / (study['samples'] - 1)):.3f}; checks use 0.1. The fixed-strip distance is the geodesic "
        "distance only while the unfolded segment stays inside the strip (fraction reported).",
        ["strip validity under noise", "finite-difference step size", "valence-5 versus valence-6 vertices",
         "nonlinear bias of curvature", "normal sign flips (none at tested sigma)"],
        ["Noise is isotropic and independent per vertex; real scanners have correlated, anisotropic errors.",
         "Markers are attached barycentrically to faces, so marker placement error is part of the geometry noise."],
        "T044: separate geometry uncertainty from sensor noise for a marker-distance observation.")}


# ---------------------------------------------------------------- T044
@task("T044", changed_files=FILES, regression_tests=(
    f"{TESTS}::test_law_of_total_variance_split",
    f"{TESTS}::test_variance_split_task_report"))
def geometry_versus_sensor_uncertainty(ctx):
    study = _memo(ctx, "variance-split", S.variance_split_study)
    scenarios = study["scenarios"]
    outer, inner, fresh = study["outer"], study["inner"], study["fresh"]
    anova = max(abs(s["anova_residual"]) for s in scenarios)
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
    ctx.artifact_json("variance-split.json", jsonable({**study, "z_fresh": z_fresh, "z_within": z_within,
                                                        "z_between": z_between, "z_averaging": z_avg}))
    ctx.artifact_text("averaging.svg", svg.line_plot([
        ("total variance (MC)", [r["repeats"] for r in averaging], [r["variance"] for r in averaging]),
        ("geometry + sensor / K", [r["repeats"] for r in averaging], [r["predicted"] for r in averaging]),
        ("geometry floor", [r["repeats"] for r in averaging], [geometry_floor] * len(averaging))],
        title="Averaging K sensor readings on one uncertain surface", xlabel="repeated readings K",
        ylabel="variance", logx=True, logy=True))
    shares = [{"sigma_geometry": s["sigma_geometry"], "sigma_sensor": s["sigma_sensor"],
               "geometry_share": s["geometry_share"], "dominant": "geometry" if s["geometry_share"] > 0.5 else "sensor"}
              for s in scenarios]
    findings = [
        finding("Nested Monte Carlo sums of squares decompose exactly into between and within parts", "numerical",
                anova, {"generator": generator("variance_split_study", outer=outer, inner=inner, seed=S.SEED + 100),
                        "checks": [check("max relative ANOVA residual (SST - SSB - SSW) / SST", anova, 1e-10,
                                         kind="invariant")]},
                tolerance={"abs": 1e-10, "rel": 0.0}),
        finding("Residual variance equals geometry variance plus sensor variance in every scenario", "numerical",
                {"fresh_ratio": [s["fresh_ratio"] for s in scenarios], "z": z_fresh},
                {"derivation": "Law of total variance: Var(r) = E[Var(r | eta)] + Var(E[r | eta]) = sigma_s^2 + "
                               "Var_eta(d)",
                 "checks": [check("max |z| of fresh total variance against sigma_s^2 + sigma_g^2 |grad d|^2",
                                  max(abs(z) for z in z_fresh), 4.0, kind="self_convergence")]},
                tolerance=TIGHT),
        finding("Nested components recover the declared sensor variance and the linearized geometry variance",
                "numerical", {"within_ratio": [s["within_ratio"] for s in scenarios],
                              "between_ratio": [s["between_ratio"] for s in scenarios],
                              "z_within": z_within, "z_between": z_between},
                {"generator": generator("variance_split_study", outer=outer, inner=inner, seed=S.SEED + 100),
                 "checks": [check("max |z| within-group variance vs sigma_s^2", max(abs(z) for z in z_within), 4.0,
                                  kind="self_convergence"),
                            check("max |z| corrected between-group variance vs geometry variance",
                                  max(abs(z) for z in z_between), 4.0, kind="self_convergence")]},
                tolerance=TIGHT),
        finding("Geometry uncertainty dominates the baseline marker-distance residual", "numerical",
                {"baseline": {"sigma_geometry": 1e-3, "sigma_sensor": 5e-4, "geometry_share": baseline["geometry_share"],
                              "crossover_sensor_sigma": baseline["crossover_sensor_sigma"]}, "scenarios": shares,
                 "gain": study["gain"]},
                {"generator": generator("variance_split_study", level=study["level"], seed=S.SEED + 100),
                 "checks": [check("baseline geometry share of the total variance", baseline["geometry_share"], 0.5, "ge"),
                            check("|nested geometry share - linearized share| at baseline",
                                  baseline["between_corrected"] / baseline["nested_total"] - baseline["geometry_share"],
                                  0.05, kind="self_convergence")]},
                tolerance=TIGHT),
        finding("Averaging repeated sensor readings leaves the geometry variance as a floor", "numerical",
                {"repeats": [r["repeats"] for r in averaging], "variance": [r["variance"] for r in averaging],
                 "geometry_floor": geometry_floor, "geometry_share": [r["geometry_share"] for r in averaging],
                 "z": z_avg},
                {"generator": generator("variance_split_study", sigma_geometry=study["averaging"]["sigma_geometry"],
                                        sigma_sensor=study["averaging"]["sigma_sensor"], seed=S.SEED + 100),
                 "checks": [check("variance at the largest K / geometry floor", averaging[-1]["variance"] / geometry_floor,
                                  0.9, "ge"),
                            check("max |z| against geometry + sensor / K", max(abs(z) for z in z_avg), 4.0,
                                  kind="self_convergence")]},
                tolerance=TIGHT,
                counterexample={"statement": "Averaging repeated measurements drives the observation error to zero",
                                "witness": {"repeats": averaging[-1]["repeats"], "variance": averaging[-1]["variance"],
                                            "geometry_floor": geometry_floor}}),
        _physical("A real marker-distance sensor on a real scanned part has this geometry and sensor variance split",
                  "sensor_performance"),
        _physical("The geometry sigma of 1e-3 is the accuracy of a real scanned surface"),
    ]
    result = (f"Distance gain |grad d| = {study['gain']:.4f} (nominal distance {study['nominal_distance']:.6f}); ANOVA "
              f"residual {anova:.1e}; fresh total/predicted ratios "
              + ", ".join(f"{s['fresh_ratio']:.3f}" for s in scenarios)
              + f" (max |z| {max(abs(z) for z in z_fresh):.2f}); baseline (sigma_g=1e-3, sigma_s=5e-4) geometry share "
              f"{baseline['geometry_share']:.3f}, crossover sensor sigma {baseline['crossover_sensor_sigma']:.2e}; "
              f"averaging K=" + ", ".join(f"{r['repeats']}: share {r['geometry_share']:.2f}" for r in averaging)
              + f"; floor {geometry_floor:.3e}.")
    return {"state": "completed", "findings": findings, "fields": fields(
        "For a marker-distance observation on an uncertain surface, geometry and sensor variances add (law of total "
        "variance), can be separated by a nested design, and averaging sensor readings cannot remove the geometry "
        "part.",
        "Residual r = y - d(V_nominal), y = d(V_nominal + eta) + eps, eta ~ N(0, sigma_g^2 I) per vertex coordinate, "
        "eps ~ N(0, sigma_s^2) independent. Var(r) = sigma_s^2 + Var_eta(d) ~ sigma_s^2 + sigma_g^2 |grad d|^2. "
        "Nested design: within-group variance estimates sigma_s^2, corrected between-group variance estimates the "
        "geometry part.",
        ["icosphere level 3, fixed marker strip (T043)", "sigma_g in {1e-4, 1e-3}, sigma_s in {1e-4, 5e-4, 2e-3}",
         f"nested {outer} x {inner} and fresh {fresh} samples", "averaging K = 1, 4, 16, 64 at sigma_g=1e-3, sigma_s=2e-3"],
        "Synthetic: the as-built surface differs from the nominal mesh by Gaussian vertex noise and a synthetic "
        "sensor adds Gaussian noise to the distance.",
        "Exact ANOVA identity; totals within sampling error of sigma_s^2 + sigma_g^2 |grad d|^2; floor at the "
        "geometry variance.",
        "Nested and fresh seeded Monte Carlo per scenario; z-scores use the Gaussian variance standard error.",
        result,
        "z-scores use sqrt(2/(N-1)) relative standard errors of Gaussian variances; the geometry variance uses the "
        "T043 linearization, valid at these sigma (checked there).",
        ["unidentifiable geometry share when sigma_s^2 / inner >> geometry variance (handled by z-scores)",
         "nonlinearity of the distance (checked in T043)", "strip validity (T043)"],
        ["Geometry and sensor noise are independent; a sensor that touches the same surface patch could correlate them.",
         "Which part dominates is a property of the declared sigmas, not of any real scanner or sensor."],
        "T045: carry this split into typed observation modes (intrinsic geodesic distance, camera chord distance).")}
