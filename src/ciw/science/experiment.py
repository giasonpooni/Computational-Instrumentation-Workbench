"""Declarative experiment specifications and their compilation into jobs.

``ciw.experiment-spec.v1`` is finite JSON (see ``schemas/experiment-spec.v1.schema.json``)
so any language can read it. It states the hypothesis, model, parameters with
units, initial or boundary conditions, sweeps, perturbations, the observation
mode, solver and required capabilities, tolerances, invariants that must hold,
stopping conditions and the evidence status sought.

Compilation is deterministic: the same specification yields the same plan
identity and job identities. It resolves ``"$name"`` parameter references,
converts every quantity into the working units (angles in rad, lengths in the
model length unit), checks solver capabilities and domains, bounds total work,
selects a backend and lists expected artifacts and validation checks. It never
executes a solver; physical measurements it cannot perform become acquisition
requests.
"""
from __future__ import annotations

from copy import deepcopy
import itertools
import math
import re
from typing import Any

import numpy as np

from ._common import (Refusal, content_identity, finite, integer, mapping, plain, require_keys, text)
from .backends import select
from .geometry import surface_from_json
from .oracles import ORACLES
from .solvers import SolverRegistry, default_registry
from .units import Quantity, convert, parse_unit, require_dimension
from .vocabulary import ACQUISITION_KINDS, CAPABILITIES, CLAIM_CLASSES, OBSERVABLES

SCHEMA = "ciw.experiment-spec.v1"
PLAN_SCHEMA = "ciw.execution-plan.v1"
MAX_JOBS = 512
MAX_TOTAL_STEPS = 5_000_000
MAX_SWEEP_VALUES = 256
NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")
IDENT = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")

MODEL_OPERATIONS = {"surface-geodesic": "geodesic.initial-value", "surface-jacobi": "jacobi.normal-scalar",
                    "surface-boundary-value": "geodesic.boundary-value", "mesh-distance": "distance.mesh-single-source"}
MESH_KINDS = {"plane": ({"width", "height", "cells"}, set()), "cylinder": ({"height", "angle_span", "cells"}, set()),
              "sphere": ({"subdivisions"}, set()), "torus": ({"cells"}, set())}
COORDINATE_DIMENSIONS = {"plane": ("m", "m"), "sphere": ("rad", "rad"), "cylinder": ("rad", "m"),
                         "torus": ("rad", "rad"), "graph": ("m", "m"), "conformal": ("m", "m"),
                         "hyperbolic-half-plane": ("1", "1")}
CONSTANT_CURVATURE = {"plane", "sphere", "cylinder", "hyperbolic-half-plane"}
TOP_REQUIRED = {"schema", "experiment_id", "title", "hypothesis", "model", "solver", "stopping", "tolerances",
                "invariants", "evidence"}
TOP_OPTIONAL = {"description", "parameters", "initial_conditions", "boundary", "sweep", "perturbations",
                "observation", "backend", "inputs"}


def validate_spec(spec: Any) -> dict:
    """Structural and vocabulary validation; returns the specification unchanged."""
    spec = require_keys(mapping(spec, "experiment specification"), "experiment specification", TOP_REQUIRED, TOP_OPTIONAL)
    if spec["schema"] != SCHEMA:
        raise Refusal("unsupported_schema", f"Expected {SCHEMA}, found {spec['schema']!r}")
    if not isinstance(spec["experiment_id"], str) or IDENT.fullmatch(spec["experiment_id"]) is None:
        raise Refusal("malformed_record", "experiment_id must be lowercase [a-z0-9._-], at most 128 characters")
    text(spec["title"], "title", 256)
    hypothesis = require_keys(spec["hypothesis"], "hypothesis", {"statement"}, {"competing", "falsified_if"})
    text(hypothesis["statement"], "hypothesis.statement")
    for item in hypothesis.get("competing", []):
        require_keys(item, "hypothesis.competing", {"id", "statement"})
    model = require_keys(spec["model"], "model", {"kind", "surface", "length_unit"}, {"frame", "mesh"})
    if model["kind"] not in MODEL_OPERATIONS:
        raise Refusal("unsupported_model", f"Model kind {model['kind']!r} is not declared", allowed=sorted(MODEL_OPERATIONS))
    require_dimension(model["length_unit"], "m", "model.length_unit")
    surface_from_json(model["surface"], model["length_unit"])
    mesh_model = model["kind"] == "mesh-distance"
    if mesh_model != ("mesh" in model):
        raise Refusal("malformed_record", "model.mesh is required for mesh-distance models and only for them")
    if mesh_model:
        kind = model["surface"]["type"]
        if kind not in MESH_KINDS:
            raise Refusal("unsupported_surface", f"No mesh generator is declared for {kind}", allowed=sorted(MESH_KINDS))
        require_keys(model["mesh"], "model.mesh", MESH_KINDS[kind][0], {"t_factor", "boundary"})
    for name, quantity in spec.get("parameters", {}).items():
        _name(name)
        Quantity.from_json(quantity, f"parameters.{name}")
    if "sweep" in spec:
        sweep = require_keys(spec["sweep"], "sweep", {"mode", "parameters"})
        if sweep["mode"] not in {"product", "zip"}:
            raise Refusal("malformed_record", "sweep.mode must be product or zip")
        for name, axis in mapping(sweep["parameters"], "sweep.parameters").items():
            _name(name)
            _axis(axis, f"sweep.parameters.{name}")
    solver = require_keys(spec["solver"], "solver", {"solver_id"} | (set() if mesh_model else {"steps"}),
                          {"requires", "settings", "steps"})
    text(solver["solver_id"], "solver.solver_id", 128)
    unknown = set(solver.get("requires", [])) - CAPABILITIES
    if unknown:
        raise Refusal("unknown_capability", f"solver.requires names undeclared capabilities {sorted(unknown)}")
    tolerances = require_keys(spec["tolerances"], "tolerances", {"abs", "rel"}, {"oracles"})
    finite(tolerances["abs"], "tolerances.abs", minimum=0.0)
    finite(tolerances["rel"], "tolerances.rel", minimum=0.0)
    for oracle_id, override in tolerances.get("oracles", {}).items():
        if oracle_id not in ORACLES:
            raise Refusal("unknown_oracle", f"tolerances.oracles names undeclared oracle {oracle_id!r}")
        require_keys(override, f"tolerances.oracles.{oracle_id}", set(), {"abs", "rel"})
    if not isinstance(spec["invariants"], list) or not spec["invariants"]:
        raise Refusal("malformed_record", "invariants must name at least one oracle that must pass")
    for oracle_id in spec["invariants"]:
        if oracle_id not in ORACLES:
            raise Refusal("unknown_oracle", f"Invariant {oracle_id!r} is not a declared oracle", allowed=sorted(ORACLES))
    if mesh_model:
        stopping = require_keys(spec["stopping"], "stopping", {"max_vertices"})
        integer(stopping["max_vertices"], "stopping.max_vertices", minimum=3, maximum=3000)
    else:
        stopping = require_keys(spec["stopping"], "stopping", {"arclength", "max_steps"}, {"stop_at_conjugate_point"})
        integer(stopping["max_steps"], "stopping.max_steps", minimum=1, maximum=200_000)
    evidence = require_keys(spec["evidence"], "evidence", {"physical_measurements", "claims_sought"})
    if evidence["physical_measurements"] not in {"not_acquired", "planned", "acquired"}:
        raise Refusal("malformed_record", "evidence.physical_measurements must be not_acquired, planned or acquired")
    if set(evidence["claims_sought"]) - CLAIM_CLASSES:
        raise Refusal("unknown_claim_class", "evidence.claims_sought uses undeclared claim classes")
    if "authorized" in evidence["claims_sought"]:
        raise Refusal("claim_unsupported", "An experiment cannot seek authorization; the authority gate decides that")
    if "observation" in spec:
        observation = require_keys(spec["observation"], "observation", {"observable", "noise_std", "acquisition"},
                                   {"instrument", "frame", "clock", "synthetic_truth", "between"})
        for key in ("observable", "synthetic_truth"):
            if key in observation and observation[key] not in OBSERVABLES:
                raise Refusal("unknown_observable", f"observation.{key} {observation[key]!r} is not declared")
        if observation["acquisition"] not in ACQUISITION_KINDS:
            raise Refusal("unknown_acquisition", "observation.acquisition is not declared")
        if observation["acquisition"] == "physical":
            raise Refusal("acquisition_not_bound", "Physical acquisition is performed by the physical protocol "
                          "manager, not by the compiler; declare planned instead")
        if observation.get("between", "start-end") != "start-end":
            raise Refusal("malformed_record", "observation.between supports start-end in this version")
    for index, item in enumerate(spec.get("perturbations", [])):
        require_keys(item, f"perturbations[{index}]", {"parameter", "distribution", "std"}, {"samples"})
        if item["distribution"] != "normal":
            raise Refusal("malformed_record", "Only normal perturbations are declared")
        if item["parameter"] != "heading":
            raise Refusal("unsupported_perturbation", "This version perturbs the initial heading only")
    if "backend" in spec:
        require_keys(spec["backend"], "backend", set(), {"preferred", "precision", "seed"})
    return spec


def _name(name: Any) -> str:
    if not isinstance(name, str) or NAME.fullmatch(name) is None:
        raise Refusal("malformed_record", f"Parameter name {name!r} must match [a-z][a-z0-9_]*")
    return name


def _axis(axis: Any, name: str) -> list[Quantity]:
    axis = mapping(axis, name)
    unit = parse_unit(axis.get("unit")).expression
    if set(axis) == {"values", "unit"}:
        values = axis["values"]
        if not isinstance(values, list) or not values:
            raise Refusal("malformed_record", f"{name}.values must be a nonempty array")
        numbers = [finite(value, f"{name}.values") for value in values]
    elif set(axis) == {"linspace", "unit"}:
        if not isinstance(axis["linspace"], list) or len(axis["linspace"]) != 3:
            raise Refusal("malformed_record", f"{name}.linspace must be [start, stop, count]")
        start, stop = finite(axis["linspace"][0], name), finite(axis["linspace"][1], name)
        count = integer(axis["linspace"][2], f"{name} count", minimum=1, maximum=MAX_SWEEP_VALUES)
        numbers = list(np.linspace(start, stop, count))
    else:
        raise Refusal("malformed_record", f"{name} must declare values or linspace, plus unit")
    if len(numbers) > MAX_SWEEP_VALUES:
        raise Refusal("oversized_input", f"{name} exceeds {MAX_SWEEP_VALUES} values")
    return [Quantity(float(number), unit) for number in numbers]


def spec_identity(spec: dict) -> str:
    return content_identity(spec)


def _points(spec: dict) -> list[dict[str, Quantity]]:
    base = {name: Quantity.from_json(value, name) for name, value in spec.get("parameters", {}).items()}
    if "sweep" not in spec:
        return [base]
    axes = {name: _axis(axis, name) for name, axis in spec["sweep"]["parameters"].items()}
    names = sorted(axes)
    if spec["sweep"]["mode"] == "zip":
        lengths = {len(axes[name]) for name in names}
        if len(lengths) != 1:
            raise Refusal("malformed_record", "zip sweeps need axes of equal length")
        combos = list(zip(*(axes[name] for name in names)))
    else:
        total = math.prod(len(axes[name]) for name in names)
        if total > MAX_JOBS:
            raise Refusal("oversized_input", f"Sweep expands to {total} points, above {MAX_JOBS}")
        combos = list(itertools.product(*(axes[name] for name in names)))
    return [{**base, **dict(zip(names, combo))} for combo in combos]


def _resolve(value: Any, point: dict[str, Quantity], name: str) -> Any:
    if isinstance(value, str) and value.startswith("$"):
        key = value[1:]
        if key not in point:
            raise Refusal("unresolved_parameter", f"{name} references undeclared parameter {value!r}")
        return point[key].to_json()
    return value


def _scalar(value: Any, point: dict, name: str, unit: str) -> float:
    value = _resolve(value, point, name)
    if isinstance(value, dict):
        quantity = Quantity.from_json(value, name)
        require_dimension(quantity.unit, unit, name)
        return quantity.magnitude(unit)
    return finite(value, name)


def _coordinates(value: Any, point: dict, name: str, surface_kind: str, length_unit: str) -> list[float]:
    value = _resolve(value, point, name)
    value = require_keys(value, name, {"value", "unit"})
    units = value["unit"] if isinstance(value["unit"], list) else [value["unit"]] * 2
    if not isinstance(value["value"], list) or len(value["value"]) != 2 or len(units) != 2:
        raise Refusal("malformed_record", f"{name} must hold two chart coordinates with units")
    result = []
    for index, (number, unit) in enumerate(zip(value["value"], units)):
        if isinstance(number, str):
            quantity = Quantity.from_json(_resolve(number, point, f"{name}[{index}]"), f"{name}[{index}]")
            number, unit = quantity.value, quantity.unit
        expected = COORDINATE_DIMENSIONS[surface_kind][index]
        require_dimension(unit, expected, f"{name}[{index}]")
        target = length_unit if expected == "m" else expected
        result.append(float(convert(finite(number, f"{name}[{index}]"), unit, target)))
    return result


def implied_capabilities(spec: dict) -> list[str]:
    """Capabilities the specification needs even when ``solver.requires`` is silent."""
    kind = spec["model"]["surface"]["type"]
    needed = set(spec["solver"].get("requires", []))
    if spec["model"]["kind"] == "mesh-distance":
        return sorted(needed | {"mesh_surfaces"})
    needed.add("constant_curvature" if kind in CONSTANT_CURVATURE else "arbitrary_metric")
    if spec["model"]["kind"] == "surface-jacobi" and spec["stopping"].get("stop_at_conjugate_point"):
        needed.add("conjugate_detection")
    if spec.get("boundary", {}).get("winding") is not None:
        needed.add("winding_classes")
    if any("samples" not in item for item in spec.get("perturbations", [])):
        needed.add("uncertainty_propagation")
    observation = spec.get("observation")
    if observation and "camera_chord" in {observation["observable"], observation.get("synthetic_truth")}:
        needed.add("embedded_surfaces")
    return sorted(needed)


def compile_spec(spec: dict, registry: SolverRegistry | None = None) -> dict:
    """Expand a validated specification into an execution plan without executing it."""
    spec = validate_spec(deepcopy(spec))
    registry = registry or default_registry()
    model, solver_spec = spec["model"], spec["solver"]
    surface_kind, length_unit = model["surface"]["type"], model["length_unit"]
    declaration = registry.get(solver_spec["solver_id"])
    if declaration.operation != MODEL_OPERATIONS[model["kind"]]:
        raise Refusal("solver_operation_mismatch", f"{declaration.solver_id} computes {declaration.operation}, but the "
                      f"model {model['kind']} needs {MODEL_OPERATIONS[model['kind']]}")
    capabilities = implied_capabilities(spec)
    registry.require(declaration.solver_id, capabilities, surface_kind)
    backend_spec = spec.get("backend", {})
    backend = select(declaration.solver_id, backend_spec.get("preferred"), backend_spec.get("precision", "binary64"))
    seed = backend_spec.get("seed")
    max_steps = spec["stopping"].get("max_steps", 0)
    identity = spec_identity(spec)
    jobs, total_steps = [], 0
    if model["kind"] == "mesh-distance":
        jobs = [_mesh_job(spec, declaration.solver_id, point, identity) for point in _points(spec)]
        total_steps = sum(job["steps"] for job in jobs)
        if len(jobs) > MAX_JOBS:
            raise Refusal("oversized_input", f"Plan exceeds {MAX_JOBS} jobs")
    for point in ([] if model["kind"] == "mesh-distance" else _points(spec)):
        job = {"solver_id": declaration.solver_id, "surface": model["surface"], "length_unit": length_unit,
               "settings": deepcopy(solver_spec.get("settings", {})),
               "sweep_point": {name: quantity.to_json() for name, quantity in sorted(point.items())}}
        steps = _resolve(solver_spec["steps"], point, "solver.steps")
        if isinstance(steps, dict):
            steps = Quantity.from_json(steps, "solver.steps").magnitude("1")
        if isinstance(steps, float) and steps.is_integer():
            steps = int(steps)
        job["steps"] = integer(steps, "solver.steps", minimum=1, maximum=max_steps)
        job["arclength"] = _scalar(spec["stopping"]["arclength"], point, "stopping.arclength",
                                   length_unit if surface_kind != "hyperbolic-half-plane" else "1")
        if job["arclength"] <= 0:
            raise Refusal("out_of_domain", "stopping.arclength must be positive")
        if model["kind"] == "surface-boundary-value":
            boundary = require_keys(spec.get("boundary"), "boundary", {"start", "target"}, {"winding"})
            job["point"] = _coordinates(boundary["start"], point, "boundary.start", surface_kind, length_unit)
            job["target"] = _coordinates(boundary["target"], point, "boundary.target", surface_kind, length_unit)
            if boundary.get("winding") is not None:
                winding = _resolve(boundary["winding"], point, "boundary.winding")
                if isinstance(winding, dict):
                    winding = Quantity.from_json(winding, "boundary.winding").magnitude("1")
                if isinstance(winding, float) and winding.is_integer():
                    winding = int(winding)
                job["settings"]["winding"] = integer(winding, "boundary.winding", minimum=-16, maximum=16)
        else:
            initial = require_keys(spec.get("initial_conditions"), "initial_conditions", {"point"}, {"heading", "velocity"})
            job["point"] = _coordinates(initial["point"], point, "initial_conditions.point", surface_kind, length_unit)
            if ("heading" in initial) == ("velocity" in initial):
                raise Refusal("malformed_record", "initial_conditions needs exactly one of heading or velocity")
            if "heading" in initial:
                job["heading_rad"] = _scalar(initial["heading"], point, "initial_conditions.heading", "rad")
            else:
                job["velocity"] = [float(item) for item in mapping(initial["velocity"], "velocity")["value"]]
        if spec["stopping"].get("stop_at_conjugate_point"):
            job["settings"]["stop_at_conjugate_point"] = True
        members = [job]
        for perturbation in spec.get("perturbations", []):
            std = Quantity.from_json(perturbation["std"], "perturbation.std").magnitude("rad")
            if "samples" not in perturbation:
                job["settings"]["heading_std_rad"] = std
                continue
            samples = integer(perturbation["samples"], "perturbation.samples", minimum=2, maximum=256)
            if "heading_rad" not in job:
                raise Refusal("unsupported_perturbation", "Heading ensembles need a heading initial condition")
            group = content_identity({"spec": identity, "point": job["sweep_point"], "perturbation": perturbation})
            rng = np.random.default_rng(seed if seed is not None else 0)
            job["settings"]["heading_std_rad"] = std
            job["ensemble"] = {"group": group, "role": "nominal"}
            for index, offset in enumerate(rng.normal(0.0, std, samples)):
                member = deepcopy(job)
                member["heading_rad"] = job["heading_rad"] + float(offset)
                member["ensemble"] = {"group": group, "role": "member", "index": index, "offset_rad": float(offset)}
                member["settings"].pop("heading_std_rad", None)
                members.append(member)
        for item in members:
            item["job_id"] = "job-" + content_identity({"spec": identity, "job": item})[7:31]
            total_steps += item["steps"]
            jobs.append(item)
        if len(jobs) > MAX_JOBS or total_steps > MAX_TOTAL_STEPS:
            raise Refusal("oversized_input", f"Plan exceeds {MAX_JOBS} jobs or {MAX_TOTAL_STEPS} integration steps",
                          jobs=len(jobs), steps=total_steps)
    observation = spec.get("observation")
    acquisition_requests = []
    if spec["evidence"]["physical_measurements"] == "planned":
        acquisition_requests.append({
            "instrument": (observation or {}).get("instrument", "undeclared"),
            "observable": (observation or {}).get("observable", "undeclared"),
            "status": "requested", "reason": "physical measurements are planned; the compiler cannot acquire them",
            "jobs": [job["job_id"] for job in jobs if job.get("ensemble", {}).get("role") != "member"]})
    oracles = list(dict.fromkeys(spec["invariants"]))
    plan = {
        "schema": PLAN_SCHEMA, "experiment_id": spec["experiment_id"], "spec_identity": identity,
        "operation": declaration.operation, "solver_id": declaration.solver_id,
        "required_capabilities": capabilities, "backend": backend, "seed": seed,
        "jobs": jobs, "total_steps": total_steps,
        "validation": {"oracles": oracles, "tolerances": spec["tolerances"],
                       "ensemble_checks": sorted({job["ensemble"]["group"] for job in jobs if "ensemble" in job})},
        "observation": observation, "acquisition_requests": acquisition_requests,
        "claims_sought": spec["evidence"]["claims_sought"],
        "expected_artifacts": _expected(jobs, observation, oracles),
    }
    plan["plan_identity"] = content_identity(plan)
    return plain(plan)


def _mesh_value(value: Any, point: dict, name: str, unit: str | None) -> Any:
    value = _resolve(value, point, name)
    if isinstance(value, dict):
        quantity = Quantity.from_json(value, name)
        return quantity.magnitude(unit or "1")
    return value


def _mesh_job(spec: dict, solver_id: str, point: dict, identity: str) -> dict:
    """A mesh job: generator arguments in working units, refined by an optional ``refinement`` parameter."""
    model = spec["model"]
    kind, unit, declared = model["surface"]["type"], model["length_unit"], model["mesh"]
    refinement = point["refinement"].magnitude("1") if "refinement" in point else 1.0
    if refinement <= 0 or not float(refinement).is_integer():
        raise Refusal("malformed_record", "refinement must be a positive integer factor")
    factor = int(refinement)
    args: dict[str, Any] = {}
    if "cells" in declared:
        cells = _resolve(declared["cells"], point, "model.mesh.cells")
        if not isinstance(cells, list) or len(cells) != 2:
            raise Refusal("malformed_record", "model.mesh.cells must be [cells_u, cells_v]")
        args["cells"] = [integer(int(c) if isinstance(c, float) and c.is_integer() else c, "cells", minimum=1,
                                 maximum=512) * factor for c in cells]
    for key, dimension in (("width", "m"), ("height", "m"), ("angle_span", "rad")):
        if key in declared:
            args[key] = float(_mesh_value(declared[key], point, f"model.mesh.{key}", unit if dimension == "m" else "rad"))
    if "subdivisions" in declared:
        base = _mesh_value(declared["subdivisions"], point, "model.mesh.subdivisions", None)
        args["subdivisions"] = integer(int(base), "subdivisions", minimum=0, maximum=6) + int(round(math.log2(factor)))
    settings = {"t_factor": float(declared.get("t_factor", 1.0)), "boundary": declared.get("boundary", "average"),
                **deepcopy(spec["solver"].get("settings", {}))}
    vertices = _mesh_vertices(kind, args)
    if vertices > spec["stopping"]["max_vertices"]:
        raise Refusal("oversized_input", f"Mesh needs {vertices} vertices, above stopping.max_vertices",
                      vertices=vertices, limit=spec["stopping"]["max_vertices"])
    job = {"solver_id": solver_id, "surface": model["surface"], "length_unit": unit, "mesh": {"kind": kind, **args},
           "source": "center", "settings": settings, "steps": vertices,
           "sweep_point": {name: quantity.to_json() for name, quantity in sorted(point.items())}}
    job["job_id"] = "job-" + content_identity({"spec": identity, "job": job})[7:31]
    return job


def _mesh_vertices(kind: str, args: dict) -> int:
    if kind == "sphere":
        return 10 * 4 ** args["subdivisions"] + 2
    u, v = args["cells"]
    if kind == "torus":
        return u * v
    if kind == "cylinder" and math.isclose(args["angle_span"], 2 * math.pi, rel_tol=1e-12):
        return u * (v + 1)
    return (u + 1) * (v + 1)


def _expected(jobs: list[dict], observation: dict | None, oracles: list[str]) -> list[dict]:
    nominal = [job for job in jobs if job.get("ensemble", {}).get("role") != "member"]
    artifacts = [{"kind": "execution", "count": len(jobs)}, {"kind": "numerical_result", "count": len(jobs)},
                 {"kind": "verification", "count": len(nominal), "oracles": oracles}]
    if observation and observation["acquisition"] == "synthetic":
        artifacts.append({"kind": "observation", "acquisition": "synthetic", "count": len(nominal)})
    return artifacts


def latex(spec: dict, registry: SolverRegistry | None = None) -> str:
    """A readable LaTeX view of the specification; it is not an executable model."""
    spec = validate_spec(spec)
    declaration = (registry or default_registry()).get(spec["solver"]["solver_id"])

    def escape(value: str) -> str:
        for old, new in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#"),
                         ("$", r"\$"), ("{", r"\{"), ("}", r"\}")):
            value = value.replace(old, new)
        return value

    lines = [r"\section*{" + escape(spec["title"]) + "}",
             r"\paragraph{Experiment} \texttt{" + escape(spec["experiment_id"]) + "}",
             r"\paragraph{Hypothesis} " + escape(spec["hypothesis"]["statement"]),
             r"\paragraph{Model} " + escape(spec["model"]["kind"]) + " on a " + escape(spec["model"]["surface"]["type"])
             + r" surface; solver \texttt{" + escape(declaration.solver_id) + "}.",
             r"\begin{align}"]
    lines.append(r" \\ ".join(declaration.equations))
    lines.append(r"\end{align}")
    if spec.get("parameters"):
        lines.append(r"\paragraph{Parameters}")
        lines.append(r"\begin{tabular}{ll}")
        for name, quantity in sorted(spec["parameters"].items()):
            lines.append(r"\texttt{" + escape(name) + r"} & $" + repr(quantity["value"]) + r"\,\mathrm{"
                         + escape(str(quantity["unit"])) + r"}$ \\")
        lines.append(r"\end{tabular}")
    lines.append(r"\paragraph{Invariants} " + ", ".join(r"\texttt{" + escape(item) + "}" for item in spec["invariants"]))
    lines.append(r"\paragraph{Tolerances} $\epsilon_{\mathrm{abs}} = " + repr(spec["tolerances"]["abs"])
                 + r",\ \epsilon_{\mathrm{rel}} = " + repr(spec["tolerances"]["rel"]) + "$")
    lines.append(r"\paragraph{Evidence} physical measurements: " + escape(spec["evidence"]["physical_measurements"]))
    return "\n".join(lines) + "\n"
