"""Model and solver registry: every solver declares what it computes and where it fails.

A declaration names its operation, equations (LaTeX), assumptions, supported
surface kinds, numerical method, precision, units policy, error behaviour,
independent reference oracles and known failure modes, plus a capability map
over the closed vocabulary in ``vocabulary.CAPABILITIES``. The compiler refuses
an experiment that requires a capability the chosen solver does not declare;
a capability absent from the map counts as *not supported*.

Registration is process-local trusted setup: saved experiments name solver
identities but can never register or import code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Mapping

import numpy as np

from ._common import Refusal, integer, plain
from .backends import runtime_identity
from .geometry import (Surface, extrinsic_geodesic, first_conjugate_point, geodesic, jacobi_closed_form, log_map,
                       surface_from_json, unit_direction)
from ..operations.registry import valid_operation_id
from .vocabulary import CAPABILITIES

MAX_SAMPLES = 33


@dataclass(frozen=True)
class SolverDeclaration:
    solver_id: str
    operation: str
    title: str
    equations: tuple[str, ...]
    assumptions: tuple[str, ...]
    domains: tuple[str, ...]
    capabilities: Mapping[str, bool]
    method: str
    precision: str
    units: str
    error_behavior: str
    reference_oracles: tuple[str, ...]
    failure_modes: tuple[str, ...]
    implementation: Callable[[dict], dict] = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        if not valid_operation_id(self.solver_id):
            raise Refusal("malformed_solver", f"Solver identity {self.solver_id!r} must be <name>.v<n>")
        unknown = set(self.capabilities) - CAPABILITIES
        if unknown:
            raise Refusal("unknown_capability", f"Undeclared capabilities {sorted(unknown)}")

    def supports(self, capability: str) -> bool:
        return bool(self.capabilities.get(capability, False))

    def describe(self) -> dict:
        return plain({"solver_id": self.solver_id, "operation": self.operation, "title": self.title,
                      "equations": list(self.equations), "assumptions": list(self.assumptions),
                      "domains": list(self.domains),
                      "capabilities": {name: self.supports(name) for name in sorted(CAPABILITIES)},
                      "method": self.method, "precision": self.precision, "units": self.units,
                      "error_behavior": self.error_behavior, "reference_oracles": list(self.reference_oracles),
                      "failure_modes": list(self.failure_modes)})

    def runtime(self, seed: int | None = None) -> dict:
        return runtime_identity(self.solver_id, seed=seed)


class SolverRegistry:
    def __init__(self) -> None:
        self._solvers: dict[str, SolverDeclaration] = {}

    def register(self, declaration: SolverDeclaration) -> None:
        if declaration.solver_id in self._solvers:
            raise Refusal("duplicate_solver", f"{declaration.solver_id} is already registered")
        self._solvers[declaration.solver_id] = declaration

    def get(self, solver_id: str) -> SolverDeclaration:
        if solver_id not in self._solvers:
            raise Refusal("solver_unavailable", f"No trusted solver is registered as {solver_id!r}",
                          registered=sorted(self._solvers))
        return self._solvers[solver_id]

    def ids(self) -> list[str]:
        return sorted(self._solvers)

    def require(self, solver_id: str, capabilities: list[str] | tuple[str, ...] = (), domain: str | None = None) -> SolverDeclaration:
        declaration = self.get(solver_id)
        unknown = set(capabilities) - CAPABILITIES
        if unknown:
            raise Refusal("unknown_capability", f"Required capabilities are not in the vocabulary: {sorted(unknown)}")
        missing = sorted(name for name in capabilities if not declaration.supports(name))
        if missing:
            raise Refusal("solver_capability_missing", f"{solver_id} does not declare {missing}", missing=missing,
                          solver_id=solver_id)
        if domain is not None and domain not in declaration.domains:
            raise Refusal("solver_domain_unsupported", f"{solver_id} does not support surface {domain!r}",
                          supported=list(declaration.domains))
        return declaration

    def describe(self) -> list[dict]:
        return [self._solvers[key].describe() for key in self.ids()]


# ---------------------------------------------------------------- native implementations
def _surface(job: dict) -> Surface:
    return surface_from_json(job["surface"], job["length_unit"])


def _initial_velocity(surface: Surface, job: dict) -> np.ndarray:
    if job.get("heading_rad") is not None:
        return unit_direction(surface, job["point"], float(job["heading_rad"]))
    if job.get("velocity") is not None:
        return np.asarray(job["velocity"], dtype=float)
    raise Refusal("malformed_job", "A geodesic job needs heading_rad or velocity")


def _samples(trajectory, count: int = MAX_SAMPLES) -> dict:
    index = np.unique(np.linspace(0, len(trajectory.s) - 1, min(count, len(trajectory.s))).round().astype(int))
    record = {"s": trajectory.s[index], "u": trajectory.u[index], "chart": trajectory.chart[index]}
    if trajectory.embedded is not None:
        record["x"] = trajectory.embedded[index]
    return record


def _geodesic_result(surface: Surface, trajectory, velocity: np.ndarray) -> dict:
    result = {
        "endpoint": trajectory.endpoint() | {"chart_name": surface.charts[int(trajectory.chart[-1])].name},
        "length": float(trajectory.s[-1]), "initial_velocity": velocity,
        "transitions": trajectory.transitions,
        "speed": {"initial": float(trajectory.speed[0]),
                  "max_abs_drift": float(np.max(np.abs(trajectory.speed - trajectory.speed[0])))},
        "curvature_range": [float(np.min(trajectory.curvature)), float(np.max(trajectory.curvature))],
        "samples": _samples(trajectory),
    }
    if surface.axis_of_revolution is not None and not trajectory.transitions:
        index, rho = surface.axis_of_revolution
        values = np.array([rho(u) ** 2 * du[index] for u, du in zip(trajectory.u, trajectory.du)])
        result["clairaut"] = {"initial": float(values[0]), "max_abs_drift": float(np.max(np.abs(values - values[0])))}
    return result


def run_geodesic(job: dict) -> dict:
    surface = _surface(job)
    velocity = _initial_velocity(surface, job)
    trajectory = geodesic(surface, job["point"], velocity, job["arclength"], job["steps"],
                          transitions=bool(job.get("settings", {}).get("chart_transitions", True)))
    return plain(_geodesic_result(surface, trajectory, velocity))


def run_jacobi(job: dict) -> dict:
    surface = _surface(job)
    velocity = _initial_velocity(surface, job)
    settings = job.get("settings", {})
    trajectory = geodesic(surface, job["point"], velocity, job["arclength"], job["steps"], jacobi=True,
                          transitions=bool(settings.get("chart_transitions", True)),
                          stop_at_conjugate=bool(settings.get("stop_at_conjugate_point", False)))
    result = _geodesic_result(surface, trajectory, velocity)
    jacobi = {"value": float(trajectory.jacobi[-1]), "derivative": float(trajectory.djacobi[-1]),
              "conjugate_point": first_conjugate_point(trajectory),
              "samples": trajectory.jacobi[np.unique(np.linspace(0, len(trajectory.s) - 1,
                                                                 min(MAX_SAMPLES, len(trajectory.s))).round().astype(int))]}
    if surface.constant_curvature is not None:
        jacobi["closed_form_value"] = float(jacobi_closed_form(surface.constant_curvature, np.array(trajectory.s[-1])))
    heading_std = job.get("settings", {}).get("heading_std_rad")
    if heading_std is not None:
        jacobi["lateral_std"] = abs(jacobi["value"]) * float(heading_std)
        jacobi["lateral_std_basis"] = "first order: |j(L)| * sigma_heading (unit-speed normal Jacobi field)"
    result["jacobi"] = jacobi
    return plain(result)


def run_log(job: dict) -> dict:
    surface = _surface(job)
    settings = job.get("settings", {})
    guess, target = None, np.asarray(job["target"], dtype=float)
    winding = settings.get("winding")
    if winding is not None:
        if surface.kind != "cylinder":
            raise Refusal("winding_unsupported", "Winding classes are declared only for cylinder charts")
        winding = integer(winding, "winding", minimum=-16, maximum=16)
        # Homotopy class k is the lift of the target to phi + 2 pi k in the universal cover (the unrolled plane).
        target = target + np.array([2 * math.pi * winding, 0.0])
        guess = target - np.asarray(job["point"], dtype=float)
    result = log_map(surface, job["point"], target, guess=guess, steps=job["steps"])
    result["winding"] = winding
    result["lifted_target"] = target
    return plain(result)


def run_extrinsic(job: dict) -> dict:
    surface = _surface(job)
    velocity = _initial_velocity(surface, job)
    path = extrinsic_geodesic(surface, job["point"], velocity, job["arclength"], job["steps"])
    residual = max(abs(surface.implicit(x)[0]) for x in path)
    index = np.unique(np.linspace(0, len(path) - 1, min(MAX_SAMPLES, len(path))).round().astype(int))
    return plain({"endpoint": {"x": path[-1]}, "length": float(job["arclength"]), "surface_residual_max": residual,
                  "samples": {"x": path[index]}})


def generate_mesh(job: dict):
    """Regenerate the declared sampled surface of a mesh job (deterministic)."""
    from .mesh import cylinder_patch, icosphere, plane_patch, torus_mesh
    surface, spec, unit = _surface(job), job["mesh"], job["length_unit"]
    kind = spec["kind"]
    if kind == "plane":
        return plane_patch(spec["width"], spec["height"], *spec["cells"], length_unit=unit)
    if kind == "cylinder":
        return cylinder_patch(surface.parameters["radius"], spec["height"], spec["angle_span"], *spec["cells"],
                              length_unit=unit)
    if kind == "sphere":
        return icosphere(surface.parameters["radius"], spec["subdivisions"], length_unit=unit)
    if kind == "torus":
        return torus_mesh(surface.parameters["major_radius"], surface.parameters["minor_radius"], *spec["cells"],
                          length_unit=unit)
    raise Refusal("unsupported_surface", f"No mesh generator for {kind}")


def mesh_source(sampled) -> int:
    """The vertex nearest the middle of the parameter domain (away from patch boundaries)."""
    parameters = sampled.parameters
    if sampled.kind in {"plane", "cylinder_patch", "cylinder_tube"}:
        middle = 0.5 * (parameters.min(axis=0) + parameters.max(axis=0))
        return int(np.argmin(np.linalg.norm(parameters - middle, axis=1)))
    return 0


def run_mesh(job: dict) -> dict:
    from .mesh import edge_graph_distance, heat_method_distance
    sampled = generate_mesh(job)
    source = mesh_source(sampled)
    settings = job.get("settings", {})
    heat = heat_method_distance(sampled.mesh, source, float(settings.get("t_factor", 1.0)),
                                boundary=settings.get("boundary", "average"))
    graph = edge_graph_distance(sampled.mesh, source)
    return plain({"mesh_identity": sampled.mesh.identity(), "mesh_kind": sampled.kind,
                  "vertices": sampled.mesh.vertex_count, "faces": sampled.mesh.face_count,
                  "mean_edge_length": sampled.mesh.mean_edge_length, "source_vertex": source,
                  "heat_distance": heat, "edge_graph_distance": graph,
                  "summary": {"max_heat": float(np.max(heat)), "max_edge_graph": float(np.max(graph))}})


def run_fusion(job: dict) -> dict:
    """Run a declared fusion scenario to a candidate state and an admission decision."""
    from .fusion import AdmissionPolicy, replay, scenario_engine
    engine = scenario_engine(job["scenario"])
    records = replay(engine, job["scenario"])
    candidate = engine.candidate()
    admission = engine.admit(candidate, AdmissionPolicy.from_json(job["policy"]), evaluated_at=candidate["time"]["value"])
    stages: dict[str, int] = {}
    for record in records:
        stages[record["stage"]] = stages.get(record["stage"], 0) + 1
    return plain({"stages": stages, "candidate": candidate, "admission": admission})


_COMMON_ASSUMPTIONS = ("surface declared exactly; no manufacturing deviation is modelled",
                       "coordinates in chart units: angles in rad, lengths in the job length unit")
_KINDS = ("plane", "sphere", "cylinder", "torus", "graph", "conformal", "hyperbolic-half-plane")
_EMBEDDED = ("plane", "sphere", "cylinder", "torus", "graph")


def default_registry() -> SolverRegistry:
    registry = SolverRegistry()
    registry.register(SolverDeclaration(
        "geometry.geodesic-rk4.v1", "geodesic.initial-value", "Intrinsic geodesic initial-value problem",
        (r"\ddot u^k + \Gamma^k_{ij}(u)\,\dot u^i \dot u^j = 0",
         r"\Gamma^k_{ij} = \tfrac12 g^{kl}\left(\partial_i g_{jl} + \partial_j g_{il} - \partial_l g_{ij}\right)"),
        _COMMON_ASSUMPTIONS + ("smooth metric along the path",),
        _KINDS, {"constant_curvature": True, "arbitrary_metric": True, "embedded_surfaces": True,
                 "chart_transitions": True, "closed_form_reference": True, "near_conjugate": True},
        "classical RK4, fixed step, chart monitoring with embedding-mediated transitions (sphere atlas)",
        "binary64", "lengths in the declared working unit; angles in rad",
        "O(h^4) global truncation; estimated by step halving in the oracle layer",
        ("closed-form-endpoint", "extrinsic-alternate", "speed-conservation", "clairaut", "step-halving",
         "unit-invariance"),
        ("coordinate singularity on single-chart surfaces is refused", "no adaptive step control",
         "long integrations accumulate phase error"),
        run_geodesic))
    registry.register(SolverDeclaration(
        "geometry.jacobi-rk4.v1", "jacobi.normal-scalar", "Scalar normal Jacobi field along a unit-speed geodesic",
        (r"j''(s) + K(\gamma(s))\,j(s) = 0,\quad j(0)=0,\ j'(0)=1",
         r"\sigma_\perp(L) \approx |j(L)|\,\sigma_\theta"),
        _COMMON_ASSUMPTIONS + ("unit-speed parametrization", "two-dimensional surface: normal field is scalar"),
        _KINDS, {"constant_curvature": True, "arbitrary_metric": True, "embedded_surfaces": True,
                 "chart_transitions": True, "closed_form_reference": True, "conjugate_detection": True,
                 "near_conjugate": True, "uncertainty_propagation": True},
        "RK4 on the augmented geodesic and Jacobi system", "binary64",
        "j in the working length unit per rad of initial heading",
        "O(h^4) global; conjugate point located by linear interpolation of the sign change",
        ("closed-form-jacobi", "finite-difference-jacobi", "conjugate-point"),
        ("only heading uncertainty is propagated", "first-order propagation fails beyond a conjugate point"),
        run_jacobi))
    registry.register(SolverDeclaration(
        "geometry.log-map-shooting.v1", "geodesic.boundary-value", "Logarithm map by Newton shooting",
        (r"\exp_p(v) = q", r"v_{n+1} = v_n - \left(D\exp_p(v_n)\right)^{-1}\left(\exp_p(v_n) - q\right)"),
        _COMMON_ASSUMPTIONS + ("the requested homotopy class is named by the initial guess (winding on a cylinder)",),
        _KINDS, {"constant_curvature": True, "arbitrary_metric": True, "embedded_surfaces": True,
                 "closed_form_reference": True, "winding_classes": True},
        "Newton iteration with central-difference exponential-map Jacobian", "binary64",
        "lengths in the declared working unit",
        "converges quadratically away from conjugate points; refused when cond(D exp) > 1e5",
        ("closed-form-distance",),
        ("near conjugate points the log map is refused", "no chart transitions inside the shooting loop",
         "may return a non-minimizing geodesic when the guess selects another class"),
        run_log))
    registry.register(SolverDeclaration(
        "geometry.extrinsic-rk4.v1", "geodesic.initial-value", "Extrinsic geodesic in R^3 from the implicit surface",
        (r"\ddot x = -\frac{\dot x^\top \nabla^2 F(x)\,\dot x}{\lVert\nabla F(x)\rVert^2}\,\nabla F(x),\quad F(x)=0",),
        _COMMON_ASSUMPTIONS + ("surface has a smooth implicit equation with nonvanishing gradient",),
        _EMBEDDED, {"constant_curvature": True, "arbitrary_metric": False, "embedded_surfaces": True,
                    "chart_transitions": True, "closed_form_reference": True},
        "classical RK4 in R^6 without Christoffel symbols or charts", "binary64",
        "embedding coordinates in the working length unit",
        "O(h^4) global; drift from F(x)=0 is reported, never projected away",
        ("closed-form-endpoint", "intrinsic-alternate"),
        ("abstract metrics without an embedding are unsupported",), run_extrinsic))
    registry.register(SolverDeclaration(
        "geometry.mesh-heat.v1", "distance.mesh-single-source", "Geodesic distance on a triangle mesh (heat method)",
        (r"(M - tL)\,u = \delta_s,\quad X = -\nabla u / \lVert \nabla u \rVert,\quad L\,\phi = \nabla\cdot X",
         r"t = c\,\bar h^2"),
        ("mesh inscribed in the declared smooth surface", "single source vertex nearest the domain centre",
         "averaged Neumann/Dirichlet boundary conditions unless declared otherwise"),
        ("plane", "cylinder", "sphere", "torus"), {"mesh_surfaces": True, "embedded_surfaces": True},
        "Crane-Weischedel-Wardetzky heat method with dense cotangent Laplacian; Dijkstra edge-graph companion",
        "binary64", "distances in the declared working length unit",
        "roughly first order in mean edge length; largest near the source and the cut locus",
        ("mesh-smooth-reference", "mesh-polyhedral-bound", "mesh-refinement"),
        ("dense solves refused above 3000 vertices", "not an upper or lower bound", "no geodesic paths",
         "obtuse triangles degrade accuracy"), run_mesh))
    registry.register(SolverDeclaration(
        "fusion.kalman-cv.v1", "fusion.linear-kalman", "Asynchronous constant-velocity Kalman fusion with admission",
        (r"x_k = F(\Delta t)\,x_{k-1} + w,\ w \sim \mathcal N(0, Q(\Delta t))",
         r"\nu^\top S^{-1} \nu \le \chi^2_{d}(p)", r"P^+ = (I-KH)P(I-KH)^\top + KRK^\top"),
        ("linear constant-velocity motion", "white, per-observation independent noise",
         "no retrodiction: late observations are refused"),
        ("tracker_position",), {"asynchronous_streams": True, "correlated_noise": True, "clock_alignment": True,
                                "outlier_gating": True, "uncertainty_propagation": True},
        "Joseph-form Kalman update, exact discretisation, NIS gating, windowed NIS admission", "binary64",
        "declared filter unit; covariances converted with units", "exact for the linear-Gaussian model it assumes",
        ("nees-consistency", "nis-window"),
        ("shared calibration biases are treated as white", "timing uncertainty enters to first order"), run_fusion))
    return registry
