"""Offline covariance payload schemas and retained-source relationships.

These checks validate declarations against retained observations and outputs.
They never rerun a filter, reconcile a state, differentiate a model, or multiply
a Jacobian through a covariance matrix. Content integrity is not verification
of a provider's numerical computation or of physical truth.
"""

from __future__ import annotations

from copy import deepcopy

from ..core.covariance import validate_covariance_artifact
from ..core.identities import canonical_json, content_identity
from ..core.records import number


FSRT_OPERATION = "fsrt.tank-reconstruct.v2"
JSPT_OPERATION = "jspt.covariance-propagate.v1"
_STATE_ORDER = ["tank-1.mass", "tank-2.mass"]
_FRAME = "reservoir2.mass"
_INDEPENDENCE = (
    "prior_independent_of_observations", "declared_total_independent_of_observations",
    "prior_independent_of_declared_total",
)
_FSRT_ASSUMPTIONS = [
    "Prior, observation errors and declared total are mutually independent except correlations inside the observation matrix.",
    "The prior is isotropic Gaussian with the declared prior_std; the total has its declared independent variance.",
    "One simultaneous snapshot, no temporal covariance model or physical verification.",
]


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _same(actual, expected):
    """Use retained JSON identity semantics, including Boolean/number distinctions."""
    return canonical_json(actual) == canonical_json(expected)


def validate_fsrt_payload(operation_id, data, run, parameters, selection):
    """Bind the additive covariance artifacts to the unchanged FSRT v1 record."""
    from .fsrt_records import validate_payload as validate_legacy
    from ..investigation import _fsrt_inputs_v2

    _require(operation_id == FSRT_OPERATION, "Unsupported FSRT covariance operation")
    _require(isinstance(data, dict) and {"covariance_artifacts", "state_order"} <= data.keys(),
             "FSRT v2 requires ordered covariance artifacts")
    legacy = {key: value for key, value in data.items() if key not in {"covariance_artifacts", "state_order"}}
    validate_legacy("fsrt.tank-reconstruct.v1", legacy, run, parameters, selection)
    expected = _fsrt_inputs_v2(run, parameters)
    observation = expected["observations"][0]
    observed = [index for index, present in enumerate(observation["mask"]) if present]
    _require(observed, "A successful FSRT covariance result requires an observed channel")
    _require(data["state_order"] == expected["state_order"] == _STATE_ORDER,
             "FSRT covariance state order differs from its declared model")
    artifacts = data["covariance_artifacts"]
    names = {"observation", "prior", "declared_total", "innovation", "posterior", "reconciled"}
    _require(isinstance(artifacts, dict) and set(artifacts) == names,
             "FSRT covariance artifact stages must match the v2 contract")
    for artifact in artifacts.values():
        validate_covariance_artifact(artifact)
    source = expected["observation_covariance"]
    _require(_same(artifacts["observation"], source),
             "FSRT observation covariance differs from the retained calibrated source")
    metadata = source["provenance"].get("metadata", {})
    _require(all(metadata.get(key) is True for key in _INDEPENDENCE),
             "FSRT covariance dependence assumptions must remain explicit")
    common_metadata = {
        **{key: True for key in _INDEPENDENCE},
        "shared_dependencies": deepcopy(metadata["shared_dependencies"]),
        "state_order": list(_STATE_ORDER), "source_order": list(observation["source_ids"]),
        "observed_mask": list(observation["mask"]),
        "reference_role": "artifact_quantity_values; absent input coordinates are declared references, not observations",
        "upstream_covariance_metadata": deepcopy(metadata),
    }
    assumptions = source["assumptions"] + _FSRT_ASSUMPTIONS
    model = expected["model"]
    prior = artifacts["prior"]
    total = artifacts["declared_total"]
    posterior = artifacts["posterior"]
    sources = [source["covariance_id"], prior["covariance_id"]]
    evidence = observation["evidence_ids"]
    # This is the declared isotropic prior's scalar parameterization, not a
    # posterior/filter computation. All propagated matrices are compared to
    # numerical fields already retained by the provider.
    try:
        variance = number(model["prior_std"], "prior_std") ** 2
    except OverflowError as exc:
        raise ValueError("Declared prior variance exceeds the finite record domain") from exc
    prior_matrix = [[variance, 0.0], [0.0, variance]]
    innovation = [
        [data["residuals"]["innovation_variance"][i] if i == j else observation["covariance"][i][j]
         for j in observed] for i in observed
    ]
    specs = {
        "prior": (_STATE_ORDER, "parameter", model["prior_mean"], prior_matrix,
                  [], [], "declared_isotropic_gaussian_prior"),
        "declared_total": (["total_mass"], "parameter", [model["total_mass_kg"]],
                           [[model["total_mass_variance_kg2"]]], [], [], "declared_independent_total_mass"),
        "innovation": ([observation["source_ids"][i] for i in observed], "residual",
                       [data["residuals"]["innovation"][i] for i in observed], innovation,
                       sources, evidence, "kalman_innovation_prior_plus_observation_covariance"),
        "posterior": (_STATE_ORDER, "estimated_state", data["unprojected_estimate"]["values"],
                      data["unprojected_estimate"]["covariance"], sources, evidence,
                      "existing_kalman_joseph_update_before_balance"),
        "reconciled": (_STATE_ORDER, "estimated_state", data["estimate"]["values"],
                       data["estimate"]["covariance"], [posterior["covariance_id"], total["covariance_id"]],
                       evidence, "existing_balance_gate_and_reconciliation"),
    }
    for name, (quantities, kind, reference, matrix, source_ids, evidence_ids, method) in specs.items():
        artifact = artifacts[name]
        validate_covariance_artifact(artifact, expected_quantity_ids=quantities,
                                     expected_units=["kg"] * len(quantities), expected_frame=_FRAME)
        _require(_same(artifact["reference_values"], reference) and _same(artifact["matrix"], matrix),
                 f"FSRT {name} covariance/reference differs from its retained scientific data")
        _require(artifact["basis"] == {"kind": kind, "id": FSRT_OPERATION + ":" + name}
                 and artifact["method"] == method, f"FSRT {name} covariance basis or method mismatch")
        stage_metadata = {**common_metadata, "stage": name}
        if name == "reconciled":
            stage_metadata["reconciliation_status"] = data["diagnostics"]["reconciliation_status"]
        _require(_same(artifact["provenance"], {
            "provider": FSRT_OPERATION, "source_evidence_ids": evidence_ids,
            "source_covariance_ids": source_ids, "metadata": stage_metadata,
        }), f"FSRT {name} covariance source provenance mismatch")
        _require(artifact["assumptions"] == assumptions, f"FSRT {name} covariance assumptions mismatch")


def validate_jspt_payload(operation_id, data, run, parameters, selection):
    """Validate covariance push declarations and ancestry without computing J C Jᵀ."""
    from ..covariance_workflow import operation_inputs

    _require(operation_id == JSPT_OPERATION, "Unsupported JSPT covariance operation")
    inputs = operation_inputs(parameters)
    source = inputs["covariance"]
    validate_covariance_artifact(source)
    fields = {"schema", "operation_id", "map_kind", "jacobian_source", "jacobian",
              "input_covariance", "output_covariance", "linearization_point",
              "output_reference_values", "checks", "limits"}
    _require(isinstance(data, dict) and set(data) == fields
             and data["schema"] == "jspt.covariance-result.v1" and data["operation_id"] == operation_id,
             "Invalid JSPT covariance result fields or identity")
    kind = inputs["map_kind"]
    _require(isinstance(kind, str) and kind in {"linear", "local_linearization", "weighted_aggregation", "coordinate_change"},
             "Invalid JSPT covariance mapping kind")
    _require(data["map_kind"] == kind and data["jacobian_source"] == "caller_declared",
             "JSPT cannot promote the declared Jacobian into verified derivatives")
    _require(_same(data["input_covariance"], source), "JSPT input covariance differs from its retained source")
    _require(_same(data["jacobian"], inputs["jacobian"])
             and _same(data["linearization_point"], source["reference_values"])
             and _same(data["output_reference_values"], inputs["output_reference_values"]),
             "JSPT map or reference point differs from its invocation")
    output = data["output_covariance"]
    validate_covariance_artifact(output, expected_quantity_ids=inputs["output_quantity_ids"],
                                 expected_units=inputs["output_units"], expected_frame=inputs["output_frame"])
    _require(_same(output["reference_values"], inputs["output_reference_values"]),
             "JSPT output covariance uses a different reference point")
    rows, columns = len(output["quantity_ids"]), len(source["quantity_ids"])
    jacobian = inputs["jacobian"]
    _require(isinstance(jacobian, list) and len(jacobian) == rows,
             "JSPT Jacobian row order must match output quantities")
    for row in jacobian:
        _require(isinstance(row, list) and len(row) == columns,
                 "JSPT Jacobian column order must match input quantities")
        for value in row:
            number(value, "Jacobian entry")
    _require(kind != "coordinate_change" or rows == columns,
             "Coordinate change must declare equal input/output dimensions")
    mapping = {"operation_id": operation_id, "source_covariance_id": source["covariance_id"],
               **{key: value for key, value in inputs.items() if key != "covariance"}}
    local = kind == "local_linearization"
    _require(output["basis"] == {"kind": "coordinate", "id": content_identity(mapping)},
             "JSPT output basis does not bind the declared ordered map")
    _require(output["method"] == ("first_order_covariance_pushforward" if local else "linear_covariance_pushforward"),
             "JSPT covariance method contradicts the declared mapping")
    kernel = ("sensitivity.coordinates.push_covariance" if kind == "coordinate_change"
              else "sensitivity.covariance.first_order_covariance")
    _require(_same(output["provenance"], {
        "provider": "JSPT:" + operation_id,
        "source_evidence_ids": source["provenance"]["source_evidence_ids"],
        "source_covariance_ids": [source["covariance_id"]],
        "metadata": {
            "kernel": kernel, "map_kind": kind, "jacobian_source": "caller_declared",
            "jacobian": jacobian, "input_quantity_ids": source["quantity_ids"],
            "input_units": source["units"], "input_frame": source["frame"], "input_basis": source["basis"],
            "linearization_point": source["reference_values"], "reference_values_source": "caller_declared",
            "source_uncertainty_context": deepcopy(source["provenance"].get("metadata", {})),
        },
    }), "JSPT covariance source provenance or uncertainty context mismatch")
    assumptions = source["assumptions"] + [
        "Jacobian columns follow input quantity_ids; rows follow output_quantity_ids.",
        "Jacobian coefficients and input/output reference values are caller declarations.",
        "Coefficient units are output_units[row]/input units[column]; units are declared, not inferred.",
        ("The supplied Jacobian is a local first-order approximation at input reference_values."
         if local else "The covariance push is exact for the declared fixed linear map in exact arithmetic."),
    ]
    _require(output["assumptions"] == assumptions, "JSPT covariance assumptions mismatch")
    asymmetric = any(source["matrix"][i][j] != source["matrix"][j][i]
                     for i in range(columns) for j in range(columns))
    checks = {
        "input_positive_semidefinite": True, "output_positive_semidefinite": True,
        "input_covariance_symmetrized": asymmetric, "coordinate_chart_guarded": kind == "coordinate_change",
        "jacobian_verified": False, "reference_values_verified": False,
        "physical_units_verified": False, "monte_carlo_performed": False,
    }
    _require(isinstance(data["checks"], dict) and set(data["checks"]) == set(checks)
             and all(data["checks"][key] is value for key, value in checks.items()),
             "JSPT checks cannot confer physical, derivative, or Monte Carlo verification")
    _require(data["limits"] == [
        "This operation propagates an explicit caller-declared Jacobian; it does not compute or verify derivatives.",
        "Reference values and physical unit/frame compatibility are caller declarations, not physical validation.",
        "No nonlinear Monte Carlo comparison, model adequacy check, certificate or execution authorization is produced.",
    ], "JSPT result must retain its declared operation limits")


def validate_result_dependencies(results):
    """Check retained result/artifact links and acyclic JSPT result ancestry."""
    from ..covariance_workflow import validate_source_link

    _require(isinstance(results, dict), "Retained results must form an identity map")
    edges = {}
    for identity, result in results.items():
        _require(isinstance(result, dict) and result.get("result_id") == identity,
                 "Retained result identity map is inconsistent")
        if result.get("operation_id") != JSPT_OPERATION:
            continue
        parameters = result.get("parameters")
        _require(isinstance(parameters, dict) and isinstance(parameters.get("source_result_id"), str),
                 "JSPT result needs a retained source result identity")
        source_id = parameters["source_result_id"]
        _require(source_id != identity, "A covariance result cannot depend on itself")
        validate_source_link(parameters, results)
        edges[identity] = source_id
    completed = set()
    for start in edges:
        current, visiting = start, set()
        while current in edges and current not in completed:
            _require(current not in visiting, "Covariance result dependencies contain a cycle")
            visiting.add(current)
            current = edges[current]
        completed.update(visiting)
