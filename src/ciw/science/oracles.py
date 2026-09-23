"""Reference oracles: independent routes that must agree with a numerical result.

Each oracle recomputes a quantity by a different route and returns a verdict
with the computed value, the reference value, the error and the tolerance it
was judged against. Oracle kinds:

* ``closed_form``          analytic solutions (great circles, lines, arccosh distance, sin/sinh Jacobi fields)
* ``alternate_implementation`` extrinsic R^3 integration versus intrinsic chart integration
* ``high_order_reference`` implicit Gauss-Legendre (order 6) integration, a different method family
* ``convergence``          step halving with an observed order of accuracy
* ``finite_difference``    perturbed-heading geodesics versus the Jacobi field
* ``conservation``         speed and Clairaut's angular momentum
* ``invariant``            unit rescaling and chart changes must not change geometry
* ``symbolic_check``       hand-derived Christoffel symbols and curvature against the metric

A verdict states applicability explicitly. An oracle that cannot apply (no
closed form on a torus) is ``not_applicable``, which leaves a verification that
required it ``incomplete`` rather than passed.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np

from ._common import Refusal, plain
from .geometry import (Surface, extrinsic_geodesic, geodesic, great_circle, jacobi_closed_form, surface_from_json,
                       unit_direction, closed_form_distance, curvature_from_christoffel, change_chart)

KINDS = frozenset({"closed_form", "alternate_implementation", "high_order_reference", "convergence",
                   "finite_difference", "conservation", "invariant", "symbolic_check"})


@dataclass
class Verdict:
    oracle_id: str
    kind: str
    quantity: str
    applicable: bool
    passed: bool | None = None
    computed: Any = None
    reference: Any = None
    abs_error: float | None = None
    tolerance: float | None = None
    detail: dict | None = None

    def to_json(self) -> dict:
        return plain({key: value for key, value in self.__dict__.items()})


def _tolerance(tolerances: dict, oracle_id: str, reference: Any) -> float:
    override = tolerances.get("oracles", {}).get(oracle_id, {})
    absolute = float(override.get("abs", tolerances.get("abs", 1e-9)))
    relative = float(override.get("rel", tolerances.get("rel", 1e-9)))
    scale = float(np.max(np.abs(np.asarray(reference, dtype=float)))) if reference is not None else 0.0
    return max(absolute, relative * scale)


def _judge(oracle_id: str, kind: str, quantity: str, computed: Any, reference: Any, tolerances: dict,
           detail: dict | None = None) -> Verdict:
    computed_array, reference_array = np.asarray(computed, dtype=float), np.asarray(reference, dtype=float)
    error = float(np.max(np.abs(computed_array - reference_array)))
    tolerance = _tolerance(tolerances, oracle_id, reference)
    return Verdict(oracle_id, kind, quantity, True, error <= tolerance, computed, reference, error, tolerance, detail)


def _na(oracle_id: str, kind: str, reason: str) -> Verdict:
    return Verdict(oracle_id, kind, "", False, None, detail={"reason": reason})


def _velocity(surface: Surface, job: dict) -> np.ndarray:
    if job.get("heading_rad") is not None:
        return unit_direction(surface, job["point"], float(job["heading_rad"]))
    return np.asarray(job["velocity"], dtype=float)


# ---------------------------------------------------------------- oracles
def closed_form_endpoint(job, result, surface, run, tolerances):
    oracle, kind = "closed-form-endpoint", "closed_form"
    if "endpoint" not in result:
        return _na(oracle, kind, "result carries no endpoint")
    velocity = _velocity(surface, job)
    length = float(job["arclength"])
    point = np.asarray(job["point"], dtype=float)
    if surface.kind == "sphere":
        speed = math.sqrt(velocity @ surface.charts[0].metric(point) @ velocity)
        reference = great_circle(surface, point, velocity, length * speed)
        return _judge(oracle, kind, "embedded endpoint (great circle)", result["endpoint"]["x"], reference, tolerances)
    if surface.kind in {"plane", "cylinder"}:
        return _judge(oracle, kind, "chart endpoint (straight line in developed coordinates)",
                      result["endpoint"]["u"], point + length * velocity, tolerances)
    if surface.kind == "hyperbolic-half-plane":
        speed = math.sqrt(velocity @ surface.charts[0].metric(point) @ velocity)
        return _judge(oracle, kind, "distance start-to-end equals arclength (unique geodesics, K=-1)",
                      closed_form_distance(surface, point, result["endpoint"]["u"]), length * speed, tolerances)
    return _na(oracle, kind, f"no closed-form geodesic is implemented for {surface.kind}")


def closed_form_distance_oracle(job, result, surface, run, tolerances):
    oracle, kind = "closed-form-distance", "closed_form"
    if "length" not in result or job.get("target") is None:
        return _na(oracle, kind, "not a boundary-value result")
    point, target = np.asarray(job["point"], dtype=float), np.asarray(job["target"], dtype=float)
    if surface.kind == "cylinder" and result.get("winding") is not None:
        radius = surface.parameters["radius"]
        reference = math.hypot(radius * (target[0] - point[0] + 2 * math.pi * result["winding"]), target[1] - point[1])
        return _judge(oracle, kind, f"geodesic length in winding class {result['winding']}", result["length"],
                      reference, tolerances)
    reference = closed_form_distance(surface, point, target)
    if reference is None:
        return _na(oracle, kind, f"no closed-form distance for {surface.kind}")
    return _judge(oracle, kind, "minimal geodesic distance", result["length"], reference, tolerances)


def extrinsic_alternate(job, result, surface, run, tolerances):
    oracle, kind = "extrinsic-alternate", "alternate_implementation"
    if surface.implicit is None or "x" not in result.get("endpoint", {}):
        return _na(oracle, kind, "surface has no implicit embedding")
    path = extrinsic_geodesic(surface, job["point"], _velocity(surface, job), job["arclength"], job["steps"])
    return _judge(oracle, kind, "embedded endpoint from R^3 integration", result["endpoint"]["x"], path[-1], tolerances,
                  {"surface_residual_max": max(abs(surface.implicit(x)[0]) for x in path)})


def intrinsic_alternate(job, result, surface, run, tolerances):
    oracle, kind = "intrinsic-alternate", "alternate_implementation"
    trajectory = geodesic(surface, job["point"], _velocity(surface, job), job["arclength"], job["steps"])
    return _judge(oracle, kind, "embedded endpoint from chart integration", result["endpoint"]["x"],
                  trajectory.embedded[-1], tolerances)


def gauss_legendre_reference(job, result, surface, run, tolerances):
    """Order-6 implicit Gauss-Legendre integration at twice the step count."""
    oracle, kind = "high-order-reference", "high_order_reference"
    if result.get("transitions"):
        return _na(oracle, kind, "reference integrates in a single chart; the trajectory changed charts")
    chart = surface.charts[0]
    root = math.sqrt(15.0)
    a = np.array([[5 / 36, 2 / 9 - root / 15, 5 / 36 - root / 30],
                  [5 / 36 + root / 24, 2 / 9, 5 / 36 - root / 24],
                  [5 / 36 + root / 30, 2 / 9 + root / 15, 5 / 36]])
    b = np.array([5 / 18, 4 / 9, 5 / 18])

    def rate(y):
        return np.concatenate([y[2:], -np.einsum("kij,i,j->k", chart.christoffel(y[:2]), y[2:], y[2:])])

    steps = 2 * int(job["steps"])
    h = float(job["arclength"]) / steps
    y = np.concatenate([np.asarray(job["point"], dtype=float), _velocity(surface, job)])
    for _ in range(steps):
        k = np.tile(rate(y), (3, 1))
        for _ in range(50):
            updated = np.array([rate(y + h * (a[i] @ k)) for i in range(3)])
            if np.max(np.abs(updated - k)) < 1e-15 * max(1.0, np.max(np.abs(k))):
                k = updated
                break
            k = updated
        y = y + h * (b @ k)
    if chart.quality(y[:2]) < 1e-6:
        return _na(oracle, kind, "reference reached a chart singularity")
    return _judge(oracle, kind, "chart endpoint (Gauss-Legendre, 2x steps)", result["endpoint"]["u"], y[:2], tolerances)


def step_halving(job, result, surface, run, tolerances):
    oracle, kind = "step-halving", "convergence"
    if "endpoint" not in result:
        return _na(oracle, kind, "result carries no endpoint")
    key = "x" if "x" in result["endpoint"] else "u"
    coarse_job = dict(job, steps=max(1, int(job["steps"]) // 2))
    fine_job = dict(job, steps=2 * int(job["steps"]))
    coarse = np.asarray(run(coarse_job)["endpoint"][key], dtype=float)
    fine = np.asarray(run(fine_job)["endpoint"][key], dtype=float)
    base = np.asarray(result["endpoint"][key], dtype=float)
    first, second = float(np.max(np.abs(coarse - base))), float(np.max(np.abs(base - fine)))
    floor = 1e3 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(base))))
    resolved = first > floor and second > floor
    order = math.log2(first / second) if resolved else None
    estimate = second * 16.0 / 15.0
    tolerance = _tolerance(tolerances, oracle, base)
    return Verdict(oracle, kind, "Richardson error estimate of the endpoint", True, estimate <= tolerance,
                   base, fine, estimate, tolerance, {
                       "observed_order": order, "order_status": "resolved" if resolved else "roundoff_limited",
                       "coarse_difference": first, "fine_difference": second, "expected_order": 4})


def speed_conservation(job, result, surface, run, tolerances):
    oracle, kind = "speed-conservation", "conservation"
    if "speed" not in result:
        return _na(oracle, kind, "result carries no speed record")
    initial = result["speed"]["initial"]
    return _judge(oracle, kind, "metric speed |u'|_g along the geodesic", initial + result["speed"]["max_abs_drift"],
                  initial, tolerances)


def clairaut(job, result, surface, run, tolerances):
    oracle, kind = "clairaut", "conservation"
    if "clairaut" not in result:
        return _na(oracle, kind, "not a surface of revolution in a single chart")
    initial = result["clairaut"]["initial"]
    return _judge(oracle, kind, "rho^2 dphi/ds (Clairaut)", initial + result["clairaut"]["max_abs_drift"], initial,
                  tolerances)


_LENGTH_COORDINATES = {"plane": (0, 1), "cylinder": (1,), "sphere": (), "torus": (), "graph": (0, 1),
                       "conformal": (0, 1)}


def _rescaled_surface(record: dict, factor: float) -> dict:
    scaled = dict(record)
    for key in ("radius", "major_radius", "minor_radius"):
        if key in scaled:
            scaled[key] = {"value": scaled[key]["value"], "unit": scaled[key]["unit"]}
    if record["type"] == "graph":
        scaled["height"] = [{"c": t["c"] * factor ** (1 - t["px"] - t["py"]), "px": t["px"], "py": t["py"]}
                            for t in record["height"]]
    if record["type"] == "conformal":
        scaled["sigma"] = [{"c": t["c"] * factor ** (-(t["px"] + t["py"])), "px": t["px"], "py": t["py"]}
                           for t in record["sigma"]]
    return scaled


def unit_invariance(job, result, surface, run, tolerances):
    """Re-express lengths in a unit 1000x smaller; geometry must be unchanged."""
    oracle, kind = "unit-invariance", "invariant"
    if surface.kind not in _LENGTH_COORDINATES or "endpoint" not in result:
        return _na(oracle, kind, f"no length rescaling is defined for {surface.kind}")
    unit = job["length_unit"]
    factor = 1000.0
    target_unit = {"m": "mm", "mm": "um", "cm": "0.01*mm"}.get(unit)
    if target_unit is None or target_unit.startswith("0.01"):
        return _na(oracle, kind, f"no rescaling unit is declared for {unit}")
    indices = _LENGTH_COORDINATES[surface.kind]
    scale = np.array([factor if i in indices else 1.0 for i in range(2)])
    scaled = dict(job, length_unit=target_unit, surface=_rescaled_surface(job["surface"], factor),
                  point=list(np.asarray(job["point"], dtype=float) * scale), arclength=float(job["arclength"]) * factor)
    if job.get("velocity") is not None:
        scaled["velocity"] = list(np.asarray(job["velocity"], dtype=float) * scale / factor)
    rescaled = run(scaled)
    computed = np.asarray(rescaled["endpoint"]["u"], dtype=float) / scale
    return _judge(oracle, kind, f"chart endpoint recomputed in {target_unit}", computed, result["endpoint"]["u"],
                  tolerances, {"rescaled_unit": target_unit, "factor": factor})


def chart_invariance(job, result, surface, run, tolerances):
    oracle, kind = "chart-invariance", "invariant"
    if len(surface.charts) < 2 or "x" not in result.get("endpoint", {}):
        return _na(oracle, kind, "surface declares a single chart")
    point = np.asarray(job["point"], dtype=float)
    velocity = _velocity(surface, job)
    if surface.charts[1].quality(surface.charts[1].invert(surface.charts[0].embed(point))) < 0.3:
        return _na(oracle, kind, f"start point is near a singularity of chart {surface.charts[1].name}")
    mapped, mapped_velocity = change_chart(surface, 0, 1, point, velocity)
    trajectory = geodesic(surface, mapped, mapped_velocity, job["arclength"], job["steps"], chart=1)
    return _judge(oracle, kind, f"embedded endpoint starting in chart {surface.charts[1].name}",
                  result["endpoint"]["x"], trajectory.embedded[-1], tolerances)


def closed_form_jacobi(job, result, surface, run, tolerances):
    oracle, kind = "closed-form-jacobi", "closed_form"
    if "jacobi" not in result or surface.constant_curvature is None:
        return _na(oracle, kind, "no constant-curvature closed form")
    reference = float(jacobi_closed_form(surface.constant_curvature, np.array(result["length"])))
    return _judge(oracle, kind, "j(L) against sin/sinh closed form", result["jacobi"]["value"], reference, tolerances)


def finite_difference_jacobi(job, result, surface, run, tolerances):
    oracle, kind = "finite-difference-jacobi", "finite_difference"
    if "jacobi" not in result or job.get("heading_rad") is None:
        return _na(oracle, kind, "requires a Jacobi result started from a heading")
    epsilon = 1e-5
    ends = []
    for sign in (-1, 1):
        velocity = unit_direction(surface, job["point"], float(job["heading_rad"]) + sign * epsilon)
        ends.append(geodesic(surface, job["point"], velocity, job["arclength"], job["steps"]))
    if ends[0].embedded is not None:
        separation = float(np.linalg.norm(ends[1].embedded[-1] - ends[0].embedded[-1]))
    else:
        delta = ends[1].u[-1] - ends[0].u[-1]
        middle = 0.5 * (ends[0].u[-1] + ends[1].u[-1])
        separation = float(math.sqrt(delta @ surface.charts[0].metric(middle) @ delta))
    local = dict(tolerances, oracles={oracle: {"abs": max(1e-7, float(tolerances.get("abs", 0))),
                                               "rel": max(1e-6, float(tolerances.get("rel", 0)))}})
    return _judge(oracle, kind, "|j(L)| against perturbed-heading endpoint separation / (2 epsilon)",
                  abs(result["jacobi"]["value"]), separation / (2 * epsilon), local, {"epsilon_rad": epsilon})


def conjugate_point(job, result, surface, run, tolerances):
    oracle, kind = "conjugate-point", "closed_form"
    if "jacobi" not in result or surface.constant_curvature is None:
        return _na(oracle, kind, "no closed-form conjugate distance")
    found = result["jacobi"]["conjugate_point"]
    curvature = surface.constant_curvature
    if curvature <= 0:
        return Verdict(oracle, kind, "no conjugate point for K <= 0", True, found is None, found, None,
                       None, None)
    expected = math.pi / math.sqrt(curvature)
    if result["length"] < expected:
        return Verdict(oracle, kind, "no conjugate point before pi/sqrt(K)", True, found is None, found, expected)
    local = dict(tolerances, oracles={oracle: {"abs": max(1e-6, 2 * result["length"] / job["steps"]) ** 2, "rel": 1e-6}})
    if found is None:
        return Verdict(oracle, kind, "first conjugate point pi/sqrt(K)", True, False, None, expected)
    return _judge(oracle, kind, "first conjugate point pi/sqrt(K)", found, expected, local)


def _sample_points(result: dict, surface: Surface) -> list[np.ndarray]:
    samples = result.get("samples", {})
    points = [np.asarray(u, dtype=float) for u, chart in zip(samples.get("u", []), samples.get("chart", [])) if chart == 0]
    return [point for point in points if surface.charts[0].quality(point) > 0.05][:9]


def christoffel_from_metric(job, result, surface, run, tolerances):
    """Hand-derived Christoffel symbols against central differences of the metric."""
    oracle, kind = "christoffel-from-metric", "symbolic_check"
    chart = surface.charts[0]
    worst, reference_at = 0.0, None
    points = _sample_points(result, surface)
    if not points:
        return _na(oracle, kind, "no chart-0 samples")
    for u in points:
        h = 1e-6
        derivative = np.zeros((2, 2, 2))  # [l, i, j] = d_l g_ij
        for l in range(2):
            step = np.zeros(2)
            step[l] = h * max(1.0, abs(u[l]))
            derivative[l] = (chart.metric(u + step) - chart.metric(u - step)) / (2 * step[l])
        inverse = np.linalg.inv(chart.metric(u))
        lowered = (np.einsum("ijl->ijl", derivative) + derivative.transpose(1, 0, 2)
                   - derivative.transpose(1, 2, 0))  # [i, j, l] = d_i g_jl + d_j g_il - d_l g_ij
        expected = 0.5 * np.einsum("kl,ijl->kij", inverse, lowered)
        error = float(np.max(np.abs(chart.christoffel(u) - expected)))
        if error >= worst:
            worst, reference_at = error, u
    tolerance = max(1e-6, float(tolerances.get("oracles", {}).get(oracle, {}).get("abs", 1e-6)))
    return Verdict(oracle, kind, "max |Gamma_declared - Gamma(metric)| over trajectory samples", True,
                   worst <= tolerance, None, None, worst, tolerance, {"worst_point": plain(reference_at),
                                                                        "points": len(points)})


def curvature_consistency(job, result, surface, run, tolerances):
    oracle, kind = "curvature-from-christoffel", "symbolic_check"
    points = _sample_points(result, surface)
    if not points:
        return _na(oracle, kind, "no chart-0 samples")
    chart = surface.charts[0]
    errors = [abs(chart.curvature(u) - curvature_from_christoffel(chart, u)) for u in points]
    scale = max(1.0, max(abs(chart.curvature(u)) for u in points))
    tolerance = 1e-6 * scale
    return Verdict(oracle, kind, "declared K against Riemann tensor from Christoffel symbols", True,
                   max(errors) <= tolerance, None, None, max(errors), tolerance, {"points": len(points)})


def _mesh_reference(job: dict, result: dict, model: str):
    from .solvers import generate_mesh
    sampled = generate_mesh(job)
    if sampled.mesh.identity() != result["mesh_identity"]:
        raise Refusal("mesh_identity_mismatch", "Regenerated mesh differs from the retained mesh identity")
    return sampled, sampled.reference_distance(result["source_vertex"], model)


def _normalized_rms(values: np.ndarray, reference: np.ndarray) -> float:
    return float(np.sqrt(np.mean((values - reference) ** 2)) / max(float(np.max(reference)), 1e-300))


def mesh_smooth_reference(job, result, surface, run, tolerances):
    """Heat-method distances against the smooth-surface closed form (normalized RMS error)."""
    oracle, kind = "mesh-smooth-reference", "closed_form"
    if "heat_distance" not in result:
        return _na(oracle, kind, "not a mesh result")
    try:
        _, reference = _mesh_reference(job, result, "smooth")
    except Refusal as exc:
        return _na(oracle, kind, f"{exc.code}: {exc}")
    error = _normalized_rms(np.asarray(result["heat_distance"]), reference)
    tolerance = float(tolerances.get("oracles", {}).get(oracle, {}).get("rel", tolerances.get("rel", 0.0)))
    return Verdict(oracle, kind, "RMS(heat - smooth) / max(smooth)", True, error <= tolerance, error, 0.0, error,
                   tolerance, {"mean_edge_length": result["mean_edge_length"], "vertices": result["vertices"]})


def mesh_polyhedral_bound(job, result, surface, run, tolerances):
    """Edge-graph (Dijkstra) distances can never undercut the exact polyhedral distance."""
    oracle, kind = "mesh-polyhedral-bound", "invariant"
    if "edge_graph_distance" not in result:
        return _na(oracle, kind, "not a mesh result")
    try:
        _, reference = _mesh_reference(job, result, "polyhedral")
    except Refusal as exc:
        return _na(oracle, kind, f"{exc.code}: {exc}")
    graph = np.asarray(result["edge_graph_distance"])
    undercut = float(np.max(reference - graph))
    tolerance = 1e-12 * max(1.0, float(np.max(reference)))
    return Verdict(oracle, kind, "max(polyhedral - edge graph) must be <= 0", True, undercut <= tolerance, undercut,
                   0.0, max(undercut, 0.0), tolerance, {"max_overestimate": float(np.max(graph - reference))})


def mesh_refinement(job, result, surface, run, tolerances):
    """Doubling the resolution must reduce the smooth-reference error; reports the observed order."""
    oracle, kind = "mesh-refinement", "convergence"
    if "heat_distance" not in result:
        return _na(oracle, kind, "not a mesh result")
    try:
        _, reference = _mesh_reference(job, result, "smooth")
    except Refusal as exc:
        return _na(oracle, kind, f"{exc.code}: {exc}")
    finer = dict(job, mesh=dict(job["mesh"]))
    if "cells" in finer["mesh"]:
        finer["mesh"]["cells"] = [2 * value for value in job["mesh"]["cells"]]
    else:
        finer["mesh"]["subdivisions"] = job["mesh"]["subdivisions"] + 1
    fine_result = run(finer)
    _, fine_reference = _mesh_reference(finer, fine_result, "smooth")
    coarse_error = _normalized_rms(np.asarray(result["heat_distance"]), reference)
    fine_error = _normalized_rms(np.asarray(fine_result["heat_distance"]), fine_reference)
    ratio = result["mean_edge_length"] / fine_result["mean_edge_length"]
    order = math.log(coarse_error / fine_error) / math.log(ratio) if fine_error > 0 and ratio > 1 else None
    return Verdict(oracle, kind, "normalized RMS error after doubling the resolution", True, fine_error < coarse_error,
                   fine_error, coarse_error, abs(coarse_error - fine_error), None,
                   {"observed_order": order, "order_status": "resolved" if order is not None else "unresolved",
                    "fine_vertices": fine_result["vertices"]})


ORACLES: dict[str, tuple[str, Callable]] = {
    "mesh-smooth-reference": ("closed_form", mesh_smooth_reference),
    "mesh-polyhedral-bound": ("invariant", mesh_polyhedral_bound),
    "mesh-refinement": ("convergence", mesh_refinement),
    "closed-form-endpoint": ("closed_form", closed_form_endpoint),
    "closed-form-distance": ("closed_form", closed_form_distance_oracle),
    "extrinsic-alternate": ("alternate_implementation", extrinsic_alternate),
    "intrinsic-alternate": ("alternate_implementation", intrinsic_alternate),
    "high-order-reference": ("high_order_reference", gauss_legendre_reference),
    "step-halving": ("convergence", step_halving),
    "speed-conservation": ("conservation", speed_conservation),
    "clairaut": ("conservation", clairaut),
    "unit-invariance": ("invariant", unit_invariance),
    "chart-invariance": ("invariant", chart_invariance),
    "closed-form-jacobi": ("closed_form", closed_form_jacobi),
    "finite-difference-jacobi": ("finite_difference", finite_difference_jacobi),
    "conjugate-point": ("closed_form", conjugate_point),
    "christoffel-from-metric": ("symbolic_check", christoffel_from_metric),
    "curvature-from-christoffel": ("symbolic_check", curvature_consistency),
}


def evaluate(oracle_ids: list[str], job: dict, result: dict, run: Callable[[dict], dict], tolerances: dict) -> dict:
    """Run the named oracles and summarize: passed only if every oracle applied and passed."""
    surface = surface_from_json(job["surface"], job["length_unit"])
    verdicts = []
    for oracle_id in oracle_ids:
        if oracle_id not in ORACLES:
            raise Refusal("unknown_oracle", f"Oracle {oracle_id!r} is not declared", allowed=sorted(ORACLES))
        try:
            verdict = ORACLES[oracle_id][1](job, result, surface, run, tolerances)
        except Refusal as exc:
            verdict = _na(oracle_id, ORACLES[oracle_id][0], f"oracle refused: {exc.code}: {exc}")
        except (np.linalg.LinAlgError, ArithmeticError, ValueError) as exc:
            verdict = _na(oracle_id, ORACLES[oracle_id][0], f"oracle could not evaluate: {type(exc).__name__}: {exc}")
        verdicts.append(verdict.to_json())
    applicable = [item for item in verdicts if item["applicable"]]
    if any(item["passed"] is False for item in applicable):
        status = "failed"
    elif len(applicable) < len(verdicts):
        status = "incomplete"
    else:
        status = "passed"
    return {"status": status, "verdicts": verdicts,
            "not_applicable": [item["oracle_id"] for item in verdicts if not item["applicable"]]}
