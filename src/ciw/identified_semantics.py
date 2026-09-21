"""Read-only semantic bindings for the identified-design operation graph.

No provider is imported or executed here. These checks bind retained claims,
native content identities and handoffs; exact pinned replay still establishes
whether the numerical calculations reproduce.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math

from .telemetry import canonical


SCOPE = "conditional_on_identified_point_model"


def _require(condition, message):
    if not condition:
        raise ValueError("Identified-design semantic binding: " + message)


def _same(actual, expected, label):
    _require(canonical(actual) == canonical(expected), label)


def _number(value):
    _require(type(value) in (int, float) and math.isfinite(value), "finite real number required")
    return float(value)


def _floats(value):
    return [_floats(item) for item in value] if isinstance(value, list) else _number(value)


def _matrix(value, rows, columns, label):
    _require(isinstance(value, list) and len(value) == rows, label + " row count")
    _require(all(isinstance(row, list) and len(row) == columns for row in value), label + " column count")
    _floats(value)


def _sidt_hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _native_id(namespace, value):
    return namespace + ":" + sha256(namespace.encode() + b"\0" + canonical(value)).hexdigest()


def _sidt(step, declaration):
    data = step["result"]["data"]
    _same(data["schema"], "sidt.declared-identification-result.v1", "SIDT schema")
    _same(data["operation_id"], "sidt.declared-lti-identification.v1", "SIDT operation")
    _same(data["execution_ref"], step["execution_id"], "SIDT execution")
    _same(data["inputs"], declaration, "SIDT retained source")
    _same(data["input_digest"], "sha256:" + _sidt_hash(declaration), "SIDT input digest")
    _same(data["verification_refs"], [], "SIDT verification scope")
    numerical = data["numerical_result"]
    _same(data["numerical_id"], "sidt:numerical:sha256:" + _sidt_hash(numerical), "SIDT numerical identity")
    _same(data["result_id"], "sidt:result:sha256:" + _sidt_hash({k: v for k, v in data.items() if k != "result_id"}), "SIDT result identity")
    for key in ("model_id", "state_frame", "clock_frame"):
        _same(numerical[key], declaration[key], "SIDT " + key)
    _same(numerical["status"], "identified", "only identified candidates may continue")
    _same(numerical["parameter_covariance"], {"status": "unknown", "matrix": None,
        "reason": "observed_regressor_errors_and_parameter_uncertainty_are_not_modelled"}, "SIDT unknown parameter covariance")
    _same(numerical["applicability"], "conditional_fully_observed_discrete_lti_candidate", "SIDT applicability")
    _same(numerical["evidence_status"], "caller_declared_unattested", "SIDT evidence authority")
    diagnostics, candidate = numerical["diagnostics"], numerical["candidate"]
    n, m = len(declaration["state_names"]), len(declaration["input_names"])
    _matrix(candidate["A"], n, n, "SIDT A")
    _matrix(candidate["B"], n, m, "SIDT B")
    _same(diagnostics["rank"], n + m, "SIDT identified rank")
    _same(diagnostics["regressor_count"], n + m, "SIDT regressor count")
    count = len(declaration["training"]["states"]) - 1
    _same(diagnostics["sample_count"], count, "SIDT transition count")
    _same(diagnostics["degrees_of_freedom"], count - n - m, "SIDT degrees of freedom")
    limit = _number(declaration["condition_limit"])
    _same(diagnostics["condition_limit"], limit, "SIDT condition limit")
    _same(diagnostics["condition_measure"], "spectral_2_norm_in_declared_coordinates", "SIDT conditioning coordinates")
    condition = _number(diagnostics["condition_number"])
    _require(1 <= condition <= limit, "SIDT conditioning blocks an identified claim")
    metadata = {key: declaration[key] for key in ("state_names", "state_units", "input_names", "input_units")}
    metadata.update(sample_interval=_number(declaration["sample_interval"]), output_names=[], output_units=[],
                    conditioning_reference=declaration.get("conditioning_reference"), operation="sidt.discrete-lti-lstsq.v1")
    _same(candidate["metadata"], metadata, "SIDT candidate coordinate and timing metadata")
    candidate_payload = {"schema": "sidt.dynamics-candidate.v1", "A": candidate["A"], "B": candidate["B"],
        "C": None, "D": None, "metadata": metadata, "relative_rank_cutoff": diagnostics["relative_rank_cutoff"]}
    _same(candidate["candidate_digest"], "sha256:" + _sidt_hash(candidate_payload), "SIDT candidate identity")
    if "holdout" not in declaration:
        _same(numerical["holdout_evaluation"], None, "SIDT undeclared holdout")
    else:
        holdout = numerical["holdout_evaluation"]
        _same(holdout["sample_count"], len(declaration["holdout"]["states"]) - 1, "SIDT holdout count")
        _same(holdout["independence"], "declared_disjoint", "SIDT holdout independence")
        _same(holdout["evidence_status"], "caller_declared_unattested", "SIDT holdout authority")
    return candidate


def _oit(data, declaration, design, model, model_id):
    _same(data["model_result_id"], model_id, "OIT model result")
    _same(data["transition"], model["A"], "OIT transition")
    _same(data["horizon"], design["horizon"], "OIT horizon")
    _require(len(data["candidates"]) == len(design["candidates"]), "OIT complete candidate set")
    n = len(declaration["state_names"])
    eligible = []
    limit = None if design["condition_limit"] is None else _number(design["condition_limit"])
    for row, candidate in zip(data["candidates"], design["candidates"]):
        _same(row["candidate_id"], candidate["candidate_id"], "OIT candidate order")
        _same(row["observation_matrix"], candidate["observation_matrix"], "OIT observation matrix")
        assessment, rank = row["assessment"], row["rank"]
        m = len(candidate["observation_matrix"])
        _require(type(rank) is int and 0 <= rank <= min(n, m * design["horizon"]), "OIT rank bounds")
        _same(assessment["state_names"], declaration["state_names"], "OIT state names")
        _same(assessment["state_scales"], _floats(design["state_scales"]), "OIT coordinate scales")
        _same(assessment["horizon"], design["horizon"], "OIT assessment horizon")
        _same(assessment["coordinate_mode"], "scaled coordinates: x = diag(state_scales) z", "OIT coordinate mode")
        for field in ("status", "condition_number", "condition_limit"):
            actual = assessment["diagnostics"][field] if field == "condition_number" else assessment[field]
            _same(row[field], actual, "OIT summarized " + field)
        _same(row["condition_limit"], limit, "OIT declared conditioning policy")
        diagnostics = assessment["diagnostics"]
        _same(diagnostics["rank"], rank, "OIT summarized rank")
        _same(diagnostics["input_dimension"], n, "OIT input dimension")
        _same(diagnostics["full_column_rank"], rank == n, "OIT full rank flag")
        condition = row["condition_number"]
        if condition is not None:
            _require(_number(condition) >= 1, "OIT condition number")
        status = ("unobservable" if rank < n else "unresolved" if limit is None else
                  "ill_conditioned" if condition is None or condition > limit else "observable")
        _same(row["status"], status, "OIT rank/conditioning classification")
        if rank < n:
            _same(condition, None, "OIT deficient full-space condition")
        for key in ("observability_matrix", "analyzed_matrix"):
            _matrix(assessment[key], m * design["horizon"], n, "OIT " + key)
        _same(assessment["observability_matrix"][:m], _floats(candidate["observation_matrix"]), "OIT first observation block")
        scaled = [[value * float(scale) for value, scale in zip(block, design["state_scales"])]
                  for block in assessment["observability_matrix"]]
        _same(assessment["analyzed_matrix"], scaled, "OIT scaled observability matrix")
        if status == "observable":
            eligible.append(row["candidate_id"])
    _same(data["eligible_candidate_ids"], eligible, "OIT eligible set")
    return eligible


def _gsie(data, request, declaration, model_id):
    _same(data["model_result_id"], model_id, "GSIE model result")
    _same(data["dynamics_model_id"], model_id, "GSIE dynamics identity")
    _same(data["parameter_covariance_status"], "unknown", "GSIE unknown parameter uncertainty")
    _same(data["uncertainty_scope"], SCOPE, "GSIE conditional uncertainty scope")
    prior = request["prior"]
    for field in ("frame_id", "units"):
        _same(data[field], prior[field], "GSIE " + field)
    _same(data["predecessor_state_id"], prior["state_id"], "GSIE predecessor")
    _same(data["time"], _number(request["target_time"]), "GSIE target time")
    _same(data["operation_ref"], "geometric-state-inference.predict.v1", "GSIE operation")
    n = len(declaration["state_names"])
    _require(isinstance(data["mean"], list) and len(data["mean"]) == n, "GSIE mean dimension")
    _floats(data["mean"])
    _matrix(data["covariance"], n, n, "GSIE covariance")
    snapshot_prior = {**prior, "time": _number(prior["time"]), "mean": _floats(prior["mean"]),
        "covariance": _floats(prior["covariance"]), "dynamics_model_id": None,
        "predecessor_state_id": None, "operation_ref": None, "replay": None}
    snapshot = {"schema": "geometric-state-inference.replay.v1", "operation_ref": data["operation_ref"],
        "state_geometry": "euclidean.v1", "prior": snapshot_prior, "time": data["time"],
        "dynamics": {"matrix": _floats(request["dynamics"]["matrix"]),
                     "process_covariance": _floats(request["dynamics"]["process_covariance"]), "model_id": model_id}}
    _same(data["replay_snapshot"], snapshot, "GSIE exact prediction snapshot")
    identity = "sha256:" + sha256(b"geometric-state-inference.state-transition.v1\0" + canonical(
        {"configuration": snapshot, "mean": data["mean"], "covariance": data["covariance"]})).hexdigest()
    _same(data["state_id"], identity, "GSIE state identity")


def _gain(earlier, later, dimension):
    earlier, later = _number(earlier), _number(later)
    gain = earlier - later
    _require(math.isfinite(gain), "EDSPT finite reduction")
    tolerance = 64.0 * 2.220446049250313e-16 * dimension * max(abs(earlier), abs(later))
    _require(gain >= -tolerance, "EDSPT nonnegative reduction")
    return max(0.0, gain)


def _edspt(data, request, eligible, model_id):
    for field, expected in (("schema", "edspt.budgeted-ranking.v1"), ("operation", "budgeted-next-observation.v1"),
        ("advisory_only", True), ("selection_scope", "one_candidate_block"), ("model_result_id", model_id),
        ("criterion", request["criterion"]), ("prior_representation", "covariance"),
        ("available_budget", _number(request["budget"])), ("budget_unit", request["cost_unit"])):
        _same(data[field], expected, "EDSPT " + field)
    coordinates = [{**c, "scale": _number(c["scale"])} for c in request["coordinates"]]
    _same(data["coordinates"], coordinates, "EDSPT scaled coordinate declaration")
    _same(data["assumptions"], [
        "Local linear Gaussian observation conditional on the declared model result.",
        "Each candidate's noise has declared zero cross-covariance with the prior error.",
        "Within-block noise correlation is supplied in the full covariance.",
        "Reduction is measured in the shared declared ordered scaled coordinates.",
        "Each alternative is assessed separately; no joint acquisition is proposed.",
        "Model binding and independence declarations require external verification.",
    ], "EDSPT conditional assumptions")
    n, scales = len(coordinates), [c["scale"] for c in coordinates]
    normalized = [[_number(value) / scales[i] / scales[j] for j, value in enumerate(row)]
                  for i, row in enumerate(request["prior_covariance"])]
    _same(data["prior_covariance"], normalized, "EDSPT exact normalized GSIE covariance")
    _matrix(data["prior_precision"], n, n, "EDSPT prior precision")
    candidates = {c["candidate_id"]: c for c in request["candidates"]}
    _same(sorted(candidates), sorted(eligible), "EDSPT OIT eligibility binding")
    _same([row["candidate_id"] for row in data["scores"]], sorted(candidates), "EDSPT complete deterministic score order")
    ranked = []
    for row in data["scores"]:
        candidate = candidates[row["candidate_id"]]
        for key, expected in (("model_result_id", model_id), ("cost", _number(candidate["cost"])),
            ("cost_unit", request["cost_unit"]), ("prior_cross_covariance_policy", "declared_zero")):
            _same(row[key], expected, "EDSPT candidate " + key)
        affordable = row["cost"] <= data["available_budget"]
        _same(row["affordable"], affordable, "EDSPT affordability")
        numerical = row["numerical_score"]
        _same(numerical["candidate_id"], row["candidate_id"], "EDSPT numerical candidate")
        _same(numerical["dimension"], n, "EDSPT dimension")
        rank = numerical["numerical_rank"]
        _require(type(rank) is int and 0 <= rank <= n, "EDSPT numerical rank")
        full = rank == n
        _same(numerical["status"], "full_rank" if full else "singular", "EDSPT rank status")
        for key in ("information", "posterior_precision"):
            _matrix(numerical[key], n, n, "EDSPT " + key)
        combined = [[data["prior_precision"][i][j] + numerical["information"][i][j]
                     for j in range(n)] for i in range(n)]
        _same(numerical["posterior_precision"], combined, "EDSPT prior/information sum")
        d_gain = _gain(numerical["d_opt_logdet"], data["prior_d_opt_logdet"], n) if full else None
        a_gain = _gain(data["prior_a_opt_trace_covariance"], numerical["a_opt_trace_covariance"], n) if full else None
        _same(row["d_opt_logdet_gain"], d_gain, "EDSPT D-opt gain")
        _same(row["a_opt_trace_reduction"], a_gain, "EDSPT A-opt reduction")
        reduction = a_gain if data["criterion"] == "a_opt" else d_gain
        _same(row["expected_uncertainty_reduction"], reduction, "EDSPT selected criterion reduction")
        eligibility = "over_budget" if not affordable else "eligible" if full else "unresolved"
        _same(row["eligibility"], eligibility, "EDSPT budget/rank eligibility")
        if eligibility == "eligible":
            ranked.append((-reduction, row["candidate_id"]))
    selected_order = [identifier for _, identifier in sorted(ranked)]
    _same(data["ranked_candidate_ids"], selected_order, "EDSPT gain ordering and tie break")
    _same(data["selected_candidate_id"], selected_order[0] if selected_order else None, "EDSPT selected candidate")


def _ywir(data, request, selection, selection_id):
    for key, expected in (("schema", "ywir.observation-design-token-result.v1"),
        ("operation_id", "ywir.observation-design-token-admission.v1"), ("authority_scope", "advisory_only"),
        ("budget_unit", "inference_token"), ("selection_content_id", selection_id),
        ("selected_candidate_id", selection)):
        _same(data[key], expected, "YWIR " + key)
    for key in ("department", "token_budget", "requested_tokens"):
        _same(data[key], request[key], "YWIR request " + key)
    request_id = _native_id("ywir:token-request", request)
    _same(data["request_content_id"], request_id, "YWIR request identity")
    _same(data["result_content_id"], _native_id("ywir:token-admission", {k: v for k, v in data.items() if k != "result_content_id"}), "YWIR result identity")
    proposal = {"loop": request_id, "department": request["department"], "tokens": request["requested_tokens"],
        "expected_rank_delta": request["yield_claim"]["expected_rank_delta"],
        "expected_new_morphism": request["yield_claim"]["expected_new_morphism"], "compose_candidates": [],
        "similarity_to_store": request["similarity_to_store"], "plant_refuse": None, "confidence_rising": False,
        "eta_falling": False, "request_evidence": False, "close": False, "reindex_base": None}
    proposal_id = "ywir:proposal:" + sha256(b"ywir.proposal.v1\0" + json.dumps(
        proposal, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    _same(data["proposal_content_id"], proposal_id, "YWIR proposal identity")
    _require(data["status"] in ("admitted", "refused"), "YWIR fixed advisory operation status")
    admitted = data["status"] == "admitted"
    _same(data["admitted"], admitted, "YWIR admission status")
    _same(data["letter"], "ADMIT" if admitted else "REFUSE_SPILL", "YWIR verdict letter")
    _same(data["advisory_token_cap"], request["requested_tokens"] if admitted else 0, "YWIR token cap")
    if admitted:
        _require(0 < request["requested_tokens"] <= request["token_budget"], "YWIR admitted budget")
        _same(data["support_code"], "YIELD_ADMIT", "YWIR admission support")
    _same(data["does_not_claim"], ["building_safe", "lyapunov_certified", "jacobian_accuracy", "answer_true",
        "publisher_identity", "observations_truthful", "model_quality"], "YWIR evidence and authority limitations")


def validate_outputs(bundle):
    """Refuse contradictory retained provider outputs without numerical replay."""
    try:
        source, steps = bundle["configuration"], bundle["steps"]
        declaration, design = source["identification"], source["design"]
        data = [step["result"]["data"] for step in steps]
        model = _sidt(steps[0], declaration)
        model_id = steps[0]["numerical_result_id"]
        eligible = _oit(data[1], declaration, design, model, model_id)
        _gsie(data[2], steps[2]["request"]["inputs"], declaration, model_id)
        _edspt(data[3], steps[3]["request"]["inputs"], eligible, model_id)
        _ywir(data[4], steps[4]["request"]["inputs"], data[3]["selected_candidate_id"], steps[3]["numerical_result_id"])
    except (KeyError, TypeError, IndexError, OverflowError, RecursionError) as exc:
        raise ValueError("Malformed identified-design provider semantics") from exc
