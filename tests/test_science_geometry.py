"""Geometry kernel, solver registry and reference oracles against independent routes."""
import math

import numpy as np
import pytest

from ciw.science._common import Refusal
from ciw.science.geometry import (chord_distance, closed_form_distance, curvature_from_christoffel, extrinsic_geodesic,
                                  first_conjugate_point, geodesic, geodesic_curvature, great_circle, jacobi_closed_form,
                                  log_map, surface_from_json, unit_direction)
from ciw.science.oracles import ORACLES, evaluate
from ciw.science.solvers import default_registry

SPHERE = {"type": "sphere", "radius": {"value": 100, "unit": "mm"}}
CYLINDER = {"type": "cylinder", "radius": {"value": 50, "unit": "mm"}}
TORUS = {"type": "torus", "major_radius": {"value": 0.3, "unit": "m"}, "minor_radius": {"value": 0.1, "unit": "m"}}
GRAPH = {"type": "graph", "height": [{"c": 0.3, "px": 2, "py": 0}, {"c": -0.2, "px": 0, "py": 2}, {"c": 0.1, "px": 1, "py": 1}]}
CONFORMAL = {"type": "conformal", "sigma": [{"c": 0.2, "px": 2, "py": 0}, {"c": 0.1, "px": 0, "py": 2}]}
HYPERBOLIC = {"type": "hyperbolic-half-plane"}


@pytest.mark.parametrize("record, point", [(SPHERE, [0.7, 0.3]), (TORUS, [0.2, 0.9]), (GRAPH, [0.3, -0.4]),
                                           (CONFORMAL, [0.3, -0.4]), (HYPERBOLIC, [0.3, 1.5]), (CYLINDER, [0.1, 0.02])])
def test_declared_curvature_matches_riemann_from_christoffels(record, point):
    surface = surface_from_json(record)
    for chart in surface.charts:
        analytic = chart.curvature(np.array(point))
        assert curvature_from_christoffel(chart, np.array(point)) == pytest.approx(analytic, rel=1e-6, abs=1e-8)


def test_great_circle_through_the_pole_changes_charts():
    surface = surface_from_json(SPHERE)
    start = np.array([math.pi / 2, 0.0])
    velocity = unit_direction(surface, start, math.pi)  # towards the north pole
    length = 1.5 * math.pi * 0.1
    trajectory = geodesic(surface, start, velocity, length, 2000, jacobi=True)
    assert [item["to"] for item in trajectory.transitions][:2] == ["polar-x", "polar-z"]
    expected = np.array([great_circle(surface, start, velocity, s) for s in trajectory.s])
    assert np.max(np.abs(trajectory.embedded - expected)) < 1e-12
    assert first_conjugate_point(trajectory) == pytest.approx(math.pi * 0.1, rel=1e-9)
    assert np.max(np.abs(trajectory.jacobi - jacobi_closed_form(100.0, trajectory.s))) < 1e-12
    with pytest.raises(Refusal) as caught:
        geodesic(surface, start, velocity, length, 2000, transitions=False)
    assert caught.value.code == "chart_singularity"


@pytest.mark.parametrize("record", [TORUS, GRAPH, SPHERE])
def test_extrinsic_route_agrees_with_intrinsic_route(record):
    surface = surface_from_json(record)
    start = np.array([0.4, 0.3]) if record is not SPHERE else np.array([1.0, 0.2])
    velocity = unit_direction(surface, start, 0.8)
    intrinsic = geodesic(surface, start, velocity, 0.5, 1000, transitions=False)
    extrinsic = extrinsic_geodesic(surface, start, velocity, 0.5, 1000)
    assert np.max(np.abs(intrinsic.embedded - extrinsic)) < 1e-9


def test_hyperbolic_jacobi_growth_and_distance():
    surface = surface_from_json(HYPERBOLIC)
    start = np.array([0.0, 1.0])
    trajectory = geodesic(surface, start, unit_direction(surface, start, 0.4), 2.5, 2000, jacobi=True)
    assert closed_form_distance(surface, start, trajectory.u[-1]) == pytest.approx(2.5, rel=1e-10)
    assert trajectory.jacobi[-1] == pytest.approx(math.sinh(2.5), rel=1e-9)
    assert first_conjugate_point(trajectory) is None


def test_log_map_winding_and_conjugate_refusal():
    registry = default_registry()
    log = registry.get("geometry.log-map-shooting.v1")
    base = {"solver_id": log.solver_id, "surface": CYLINDER, "length_unit": "m", "point": [0.0, 0.0],
            "target": [1.0, 0.02], "arclength": 1.0, "steps": 200}
    lengths = {k: log.implementation(dict(base, settings={"winding": k}))["length"] for k in (-1, 0, 1)}
    for k, value in lengths.items():
        assert value == pytest.approx(math.hypot(0.05 * (1.0 + 2 * math.pi * k), 0.02), rel=1e-10)
    sphere = surface_from_json({"type": "sphere", "radius": {"value": 2.0, "unit": "m"}})
    with pytest.raises(Refusal) as caught:
        log_map(sphere, [math.pi / 2, 0], [math.pi / 2, math.pi - 1e-7], guess=[0, math.pi - 1e-7])
    assert caught.value.code == "near_conjugate"


def test_chord_is_shorter_than_intrinsic_distance_on_curved_surfaces():
    cylinder = surface_from_json(CYLINDER)
    assert chord_distance(cylinder, [0, 0], [1.0, 0]) < closed_form_distance(cylinder, [0, 0], [1.0, 0])
    assert chord_distance(cylinder, [0, 0], [0, 0.03]) == pytest.approx(closed_form_distance(cylinder, [0, 0], [0, 0.03]))
    with pytest.raises(Refusal) as caught:
        chord_distance(surface_from_json(HYPERBOLIC), [0, 1], [0, 2])
    assert caught.value.code == "no_embedding"


def test_geodesic_curvature_of_latitudes_and_helices():
    sphere = surface_from_json({"type": "sphere", "radius": {"value": 2, "unit": "m"}})
    path = np.array([[0.8, phi] for phi in np.linspace(0, 1, 60)])
    assert np.allclose(geodesic_curvature(sphere, path), math.cos(0.8) / math.sin(0.8) / 2, rtol=1e-6)
    helix = np.array([[phi, 0.3 * phi] for phi in np.linspace(0, 2, 60)])
    assert np.max(np.abs(geodesic_curvature(surface_from_json(CYLINDER), helix))) < 1e-9


@pytest.mark.parametrize("record, code", [({"type": "torus", "major_radius": {"value": 1, "unit": "m"},
                                             "minor_radius": {"value": 2, "unit": "m"}}, "out_of_domain"),
                                           ({"type": "sphere", "radius": {"value": 1, "unit": "s"}}, "dimension_mismatch"),
                                           ({"type": "klein-bottle"}, "unsupported_surface"),
                                           ({"type": "sphere"}, "malformed_record")])
def test_surface_declarations_are_validated(record, code):
    with pytest.raises(Refusal) as caught:
        surface_from_json(record)
    assert caught.value.code == code


def test_registry_declares_capabilities_and_refuses_missing_ones():
    registry = default_registry()
    assert {"geometry.geodesic-rk4.v1", "geometry.jacobi-rk4.v1", "geometry.mesh-heat.v1", "fusion.kalman-cv.v1"} <= set(registry.ids())
    with pytest.raises(Refusal) as caught:
        registry.require("geometry.geodesic-rk4.v1", ["mesh_surfaces"])
    assert caught.value.code == "solver_capability_missing" and caught.value.detail["missing"] == ["mesh_surfaces"]
    with pytest.raises(Refusal) as caught:
        registry.require("geometry.extrinsic-rk4.v1", [], "hyperbolic-half-plane")
    assert caught.value.code == "solver_domain_unsupported"
    with pytest.raises(Refusal) as caught:
        registry.get("geometry.warp-drive.v1")
    assert caught.value.code == "solver_unavailable"
    for item in registry.describe():
        assert item["equations"] and item["failure_modes"] and item["reference_oracles"]


def _job(solver, surface, **extra):
    base = {"solver_id": solver, "surface": surface, "length_unit": "m", "point": [0.4, 0.3], "heading_rad": 0.7,
            "arclength": 0.4, "steps": 600, "settings": {}}
    return base | extra


@pytest.mark.parametrize("job, oracles", [
    (_job("geometry.jacobi-rk4.v1", SPHERE, point=[1.2, 0.3]),
     ["closed-form-endpoint", "closed-form-jacobi", "finite-difference-jacobi", "conjugate-point", "speed-conservation",
      "clairaut", "extrinsic-alternate", "step-halving", "unit-invariance", "chart-invariance", "high-order-reference",
      "christoffel-from-metric", "curvature-from-christoffel"]),
    (_job("geometry.geodesic-rk4.v1", TORUS, arclength=1.0, steps=1000),
     ["extrinsic-alternate", "high-order-reference", "speed-conservation", "clairaut", "step-halving", "unit-invariance",
      "christoffel-from-metric", "curvature-from-christoffel"]),
    (_job("geometry.geodesic-rk4.v1", CONFORMAL, arclength=1.0, steps=800),
     ["speed-conservation", "step-halving", "unit-invariance", "high-order-reference", "christoffel-from-metric",
      "curvature-from-christoffel"]),
    (_job("geometry.jacobi-rk4.v1", HYPERBOLIC, point=[0.0, 1.0], arclength=2.0, steps=1000),
     ["closed-form-endpoint", "closed-form-jacobi", "conjugate-point", "finite-difference-jacobi", "speed-conservation",
      "high-order-reference"]),
])
def test_oracles_pass_on_well_resolved_problems(job, oracles):
    declaration = default_registry().get(job["solver_id"])
    result = declaration.implementation(job)
    summary = evaluate(oracles, job, result, declaration.implementation, {"abs": 1e-9, "rel": 1e-8})
    failures = [(v["oracle_id"], v["abs_error"], v["tolerance"], (v.get("detail") or {}).get("reason"))
                for v in summary["verdicts"] if v["passed"] is not True]
    assert summary["status"] == "passed", failures


def test_step_halving_reports_fourth_order_on_the_torus():
    declaration = default_registry().get("geometry.geodesic-rk4.v1")
    job = _job(declaration.solver_id, TORUS, arclength=1.0, steps=200, point=[0.1, 0.3], heading_rad=0.8)
    verdict = evaluate(["step-halving"], job, declaration.implementation(job), declaration.implementation,
                       {"abs": 1e-6, "rel": 0})["verdicts"][0]
    assert verdict["detail"]["order_status"] == "resolved"
    assert verdict["detail"]["observed_order"] == pytest.approx(4.0, abs=0.2)


def test_coarse_integration_fails_honestly_and_missing_oracles_leave_verification_incomplete():
    declaration = default_registry().get("geometry.geodesic-rk4.v1")
    job = _job(declaration.solver_id, TORUS, arclength=3.0, steps=12)
    summary = evaluate(["extrinsic-alternate", "closed-form-endpoint"], job, declaration.implementation(job),
                       declaration.implementation, {"abs": 1e-9, "rel": 1e-9})
    assert summary["status"] == "failed"
    job = _job(declaration.solver_id, TORUS, arclength=0.3, steps=400)
    summary = evaluate(["closed-form-endpoint"], job, declaration.implementation(job), declaration.implementation,
                       {"abs": 1e-9, "rel": 1e-9})
    assert summary["status"] == "incomplete" and summary["not_applicable"] == ["closed-form-endpoint"]
    with pytest.raises(Refusal) as caught:
        evaluate(["astrology"], job, {}, declaration.implementation, {})
    assert caught.value.code == "unknown_oracle"
    assert len(ORACLES) >= 18
