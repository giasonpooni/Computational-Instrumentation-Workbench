"""Experiment specifications, execution into the ledger, replay, claims, authority, design, agents and reports."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from ciw.science import agents, authority, design, ed25519
from ciw.science._common import Refusal
from ciw.science.claims import make_claim, status_statements
from ciw.science.experiment import compile_spec, latex, validate_spec
from ciw.science.geometry import surface_from_json
from ciw.science.ledger import Ledger
from ciw.science.observation import compare, explain, observation, predict
from ciw.science.report import build, markdown
from ciw.science.runner import execute, replay
from ciw.science.solvers import SolverDeclaration, default_registry

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "science"


def spec(name):
    return json.loads((EXAMPLES / "experiments" / name).read_text(encoding="utf-8"))


@pytest.fixture
def chord():
    return spec("cylinder-chord.json")


# ------------------------------------------------------------------ specification and compilation
@pytest.mark.parametrize("name", sorted(path.name for path in (EXAMPLES / "experiments").glob("*.json")))
def test_examples_validate_and_compile_deterministically(name):
    first, second = compile_spec(spec(name)), compile_spec(spec(name))
    assert first["plan_identity"] == second["plan_identity"]
    assert [job["job_id"] for job in first["jobs"]] == [job["job_id"] for job in second["jobs"]]
    assert first["backend"]["backend_id"] == "python-numpy-cpu"


@pytest.mark.parametrize("name", sorted(path.name for path in (EXAMPLES / "experiments").glob("*.json")))
def test_examples_conform_to_the_language_neutral_schema(name):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((ROOT / "src/ciw/science/schemas/experiment-spec.v1.schema.json").read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(spec(name))


def test_units_are_converted_at_compile_time(chord):
    plan = compile_spec(chord)
    job = plan["jobs"][1]
    assert job["arclength"] == pytest.approx(0.06) and job["heading_rad"] == pytest.approx(np.radians(15))
    winding = compile_spec(spec("cylinder-winding.json"))
    assert winding["jobs"][0]["target"] == pytest.approx([np.radians(60), 0.04])
    assert sorted(job["settings"]["winding"] for job in winding["jobs"]) == [-2, -1, 0, 1, 2]


def test_perturbation_ensembles_and_planned_acquisition(chord):
    plan = compile_spec(spec("sphere-jacobi.json"))
    roles = [job.get("ensemble", {}).get("role") for job in plan["jobs"]]
    assert roles.count("nominal") == 5 and roles.count("member") == 5 * 48
    assert "conjugate_detection" in plan["required_capabilities"]
    requests = compile_spec(chord)["acquisition_requests"]
    assert requests and requests[0]["status"] == "requested" and requests[0]["observable"] == "camera_chord"


@pytest.mark.parametrize("mutate, code", [
    (lambda s: s["solver"].update(requires=["mesh_surfaces"]), "solver_capability_missing"),
    (lambda s: s["solver"].update(solver_id="geometry.jacobi-rk4.v1"), "solver_operation_mismatch"),
    (lambda s: s["solver"].update(solver_id="geometry.unknown.v1"), "solver_unavailable"),
    (lambda s: s["stopping"].update(arclength={"value": 1, "unit": "s"}), "dimension_mismatch"),
    (lambda s: s["initial_conditions"].update(heading="$missing"), "unresolved_parameter"),
    (lambda s: s["initial_conditions"].update(point={"value": [0, 0], "unit": ["m", "m"]}), "dimension_mismatch"),
    (lambda s: s["sweep"]["parameters"].update(extra={"linspace": [0, 1, 256], "unit": "1"},
                                               more={"linspace": [0, 1, 8], "unit": "1"}), "oversized_input"),
    (lambda s: s.update(invariants=["astrology"]), "unknown_oracle"),
    (lambda s: s["evidence"].update(claims_sought=["authorized"]), "claim_unsupported"),
    (lambda s: s["observation"].update(acquisition="physical"), "acquisition_not_bound"),
    (lambda s: s["observation"].update(observable="telepathy"), "unknown_observable"),
    (lambda s: s.update(schema="ciw.experiment-spec.v0"), "unsupported_schema"),
    (lambda s: s.update(unexpected=True), "malformed_record"),
    (lambda s: s["model"]["surface"].update(type="klein-bottle"), "unsupported_surface"),
])
def test_compilation_refusals(chord, mutate, code):
    mutate(chord)
    with pytest.raises(Refusal) as caught:
        compile_spec(chord)
    assert caught.value.code == code


def test_latex_view_renders_equations(chord):
    text = latex(chord)
    assert r"\Gamma^k_{ij}" in text and "cylinder" in text and r"\section*" in text


# ------------------------------------------------------------------ execution, verification, replay
def test_execution_retains_every_artifact_and_replays_bitwise(tmp_path, chord):
    ledger = Ledger.create(tmp_path / "l")
    summary = execute(chord, ledger)
    assert summary["completed"] == 7 and summary["verifications"] == {"passed": 7, "failed": 0, "incomplete": 0}
    kinds = [entry["kind"] for entry in ledger.entries()]
    for kind in ("experiment_spec", "execution_plan", "acquisition_request", "runtime_identity", "parameter_identity",
                 "execution", "numerical_result", "verification", "observation", "claim"):
        assert kind in kinds
    assert {c["claim_class"]: c["admitted"] for c in summary["claims"]} == {"computed": True, "predicted": True,
                                                                            "verified": True}
    receipt = replay(ledger, chord["experiment_id"])
    assert receipt["counts"] == {"reproduced": 7, "within_tolerance": 0, "diverged": 0, "refused": 0}
    assert receipt["runtime_drift"] == []
    statements = status_statements(Ledger.open(tmp_path / "l"), chord["experiment_id"])
    assert statements["measurement"].startswith("no physical measurement exists")
    assert statements["calculation"] == "the calculation completed"
    assert statements["numerical_stability"].startswith("the result is numerically stable")
    assert statements["physical_interpretation"] == "the physical interpretation is unresolved"
    assert statements["decision"].startswith("the decision is not authorized")
    assert agents.provenance_audit(ledger) == []


def test_replay_detects_provider_drift_and_divergence(tmp_path, chord):
    ledger = Ledger.create(tmp_path / "l")
    execute(chord, ledger)
    registry = default_registry()
    original = registry.get("geometry.geodesic-rk4.v1")

    def drifted(job):
        result = original.implementation(job)
        result["endpoint"]["u"][0] += 1e-6
        return result

    patched = type(registry)()
    for item in registry.describe():
        declaration = registry.get(item["solver_id"])
        if declaration.solver_id == original.solver_id:
            declaration = SolverDeclaration(**{**declaration.__dict__, "implementation": drifted})
        patched.register(declaration)
    receipt = replay(ledger, chord["experiment_id"], registry=patched)
    assert receipt["counts"]["diverged"] == 7
    assert "diverged" in status_statements(ledger, chord["experiment_id"])["numerical_stability"]


def test_ensemble_check_confirms_first_order_uncertainty(tmp_path):
    reduced = spec("sphere-jacobi.json")
    reduced["sweep"]["parameters"]["heading"]["values"] = [45]
    ledger = Ledger.create(tmp_path / "l")
    execute(reduced, ledger)
    ensemble = [entry for entry in ledger.entries("verification") if "ensemble_group" in entry["body"]]
    assert len(ensemble) == 1 and ensemble[0]["body"]["status"] == "passed"
    verdict = ensemble[0]["body"]["verdicts"][0]
    assert abs(verdict["detail"]["z"]) <= 4 and verdict["detail"]["members"] == 48


def test_refused_jobs_are_retained_without_results(tmp_path, chord):
    chord["solver"]["steps"] = 400
    chord["model"]["surface"] = {"type": "sphere", "radius": {"value": 50, "unit": "mm"}}
    chord["initial_conditions"]["point"] = {"value": [0.0, 0.0], "unit": ["rad", "rad"]}
    chord.pop("observation")
    ledger = Ledger.create(tmp_path / "l")
    summary = execute(chord, ledger)
    assert summary["refused"] == summary["jobs"]
    refusals = [entry["body"]["refusal"]["code"] for entry in ledger.entries("execution")]
    assert set(refusals) == {"chart_singularity"}
    assert not list(ledger.entries("numerical_result"))


# ------------------------------------------------------------------ observation models
def test_observable_mismatch_is_refused_and_diagnosed():
    surface = surface_from_json({"type": "cylinder", "radius": {"value": 50, "unit": "mm"}})
    points = ([0.0, 0.0], [1.0, 0.0])
    chord_value = predict("camera_chord", surface, points=points)["value"]
    measured = observation("camera_chord", chord_value + 1e-6, "m", 2e-6, acquisition="synthetic")
    with pytest.raises(Refusal) as caught:
        compare(measured, predict("intrinsic_distance", surface, points=points))
    assert caught.value.code == "observable_mismatch"
    assert compare(measured, predict("camera_chord", surface, points=points))["consistent"]
    reading = explain(surface, points, chord_value, "m", 2e-6)
    assert reading["consistent_with"] == ["camera_chord"]
    with pytest.raises(Refusal) as caught:
        predict("imu_orientation", surface)
    assert caught.value.code == "observable_model_unbound"
    with pytest.raises(Refusal) as caught:
        observation("camera_chord", 1.0, "m", 1e-6, acquisition="synthetic", time_s=1.0)
    assert caught.value.code == "clock_unspecified"


def test_pinhole_projection_and_depth_refusal():
    camera = {"fx": 800.0, "fy": 800.0, "cx": 320.0, "cy": 240.0, "unit": "m"}
    result = predict("image_residual", camera=camera, point_camera=[0.1, -0.05, 2.0])
    assert result["value"] == pytest.approx([360.0, 220.0]) and result["unit"] == "px"
    with pytest.raises(Refusal) as caught:
        predict("image_residual", camera=camera, point_camera=[0.0, 0.0, -1.0])
    assert caught.value.code == "behind_camera"


def test_misdeclared_instrument_is_diagnosed_in_the_ledger(tmp_path):
    ledger = Ledger.create(tmp_path / "l")
    execute(spec("cylinder-misdeclared.json"), ledger)
    readings = [entry["body"]["detail"]["diagnosis"]["reading"] for entry in ledger.entries("claim")
                if entry["body"]["claim_class"] == "interpretation"]
    assert readings[:3] == ["consistent with camera_chord"] * 3
    assert readings[3] == "indistinguishable at this separation"  # along the axis chord = intrinsic


# ------------------------------------------------------------------ claims and authority
def test_claim_rules(tmp_path, chord):
    ledger = Ledger.create(tmp_path / "l")
    execute(chord, ledger)
    synthetic = next(ledger.entries("observation"))["entry_id"]
    execution = next(ledger.entries("execution"))["entry_id"]
    for claim_class, refs, status in (("measured", [synthetic], None), ("verified", [execution], None),
                                      ("authorized", [execution], None), ("interpretation", [synthetic], "consistent"),
                                      ("computed", [], None), ("predicted", [execution], None)):
        with pytest.raises(Refusal) as caught:
            make_claim(ledger, "x", claim_class, refs, status=status)
        assert caught.value.code == "claim_unsupported"
    assert make_claim(ledger, "ran", "computed", [execution])["kind"] == "claim"


def test_authority_gate_is_read_only(tmp_path, chord):
    ledger = Ledger.create(tmp_path / "l")
    execute(chord, ledger)
    secret = bytes(range(32))
    policy = {"schema": authority.POLICY_SCHEMA, "policy_id": "p", "mode": "operate",
              "approvers": [{"approver_id": "op", "public_key": ed25519.public_key(secret).hex()}],
              "requirements": {"verified_experiments": [chord["experiment_id"]], "operator_approval": True,
                               "watchdog": True, "max_evidence_age_s": 3600}}
    head = ledger.head()
    actuate = authority.propose(ledger, {"kind": "actuate", "target": "axis 1", "watchdog": True}, "tester", [head])
    decision = authority.evaluate(ledger, actuate["entry_id"], policy,
                                  approvals=[authority.approve(secret, "op", actuate["entry_id"])])
    unmet = {item["requirement"] for item in decision["body"]["reasons"] if not item["satisfied"]}
    assert decision["body"]["authorized"] is False and unmet == {"watchdog", "control_path"}
    assert decision["body"]["dispatch"] == "none"
    report = authority.propose(ledger, {"kind": "report", "target": "summary"}, "tester", [head])
    forged = {"approver_id": "op", "signature": "00" * 64}
    decision = authority.evaluate(ledger, report["entry_id"], dict(policy, requirements={"operator_approval": True}),
                                  approvals=[forged])
    assert decision["body"]["authorized"] is False
    decision = authority.evaluate(ledger, report["entry_id"], dict(policy, requirements={"operator_approval": True}),
                                  approvals=[authority.approve(secret, "op", report["entry_id"])])
    assert decision["body"]["authorized"] is True
    explore = dict(policy, mode="explore", requirements={})
    acquire = authority.propose(ledger, {"kind": "acquisition_request", "target": "camera"}, "tester", [head])
    assert authority.evaluate(ledger, acquire["entry_id"], explore)["body"]["authorized"] is False


# ------------------------------------------------------------------ design, agents, report
def test_design_recommends_the_discriminating_coupon():
    problem = json.loads((EXAMPLES / "design" / "chord-or-curvature.json").read_text(encoding="utf-8"))
    ranking = design.rank(problem)
    rows = {row["candidate_id"]: row for row in ranking["ranking"]}
    assert ranking["recommendation"] == "cylinder-r50"
    assert rows["flat-plate"]["eig_bits"] == pytest.approx(0.0, abs=1e-6)
    assert rows["sphere-r50"]["eig_bits"] == pytest.approx(rows["sphere-r50"]["max_eig_bits"], rel=1e-3)
    assert rows["cylinder-r50"]["confounded_pairs"] == [["H-curvature", "H-null"]]
    validate_spec(design.next_experiment(problem, ranking))
    problem["candidates"][1]["available"] = False
    assert design.rank(problem)["recommendation"] == "sphere-r50"


def test_agent_proposals_are_typed_and_never_applied(tmp_path, chord):
    ledger = Ledger.create(tmp_path / "l")
    proposal = agents.submit(ledger, "experiment-design", "experiment_spec", chord, agent="a", rationale="r")
    disposition = agents.dispose(ledger, proposal["entry_id"])
    assert disposition["body"]["disposition"] == "accepted" and disposition["body"]["applied"] is False
    assert not list(ledger.entries("numerical_result"))
    with pytest.raises(Refusal) as caught:
        agents.dispose(ledger, proposal["entry_id"])
    assert caught.value.code == "already_disposed"
    for role, kind, code in (("hardware-interface", "control_command", "proposal_forbidden"),
                             ("documentation", "experiment_spec", "role_not_permitted"),
                             ("oracle-whisperer", "experiment_spec", "unknown_role")):
        with pytest.raises(Refusal) as caught:
            agents.submit(ledger, role, kind, {}, agent="a", rationale="r")
        assert caught.value.code == code
    bad = agents.submit(ledger, "hardware-interface", "acquisition_request",
                        {"observable": "camera_chord", "instrument": "cam", "command": "move"}, agent="a", rationale="r")
    assert agents.dispose(ledger, bad["entry_id"])["body"]["disposition"] == "rejected"


def test_adversarial_agents_find_and_confirm_counterexamples(tmp_path):
    ledger = Ledger.create(tmp_path / "l")
    execute(spec("cylinder-chord.json"), ledger)
    assert all(item["detected"] for item in agents.adversarial_ledger_probe(tmp_path / "l"))
    torus = spec("torus-invariants.json")
    found = [item for item in agents.adversarial_parameter_search(torus, samples=6, seed=1) if item["outcome"] == "failed"]
    assert found, "coarse random step counts should violate the declared tolerances"
    proposal = agents.submit(ledger, "adversarial-test", "counterexample",
                             {key: found[0][key] for key in ("job", "oracles", "tolerances")}, agent="adv", rationale="r")
    assert agents.dispose(ledger, proposal["entry_id"])["body"]["disposition"] == "accepted"


def test_reports_are_reproducible_from_the_ledger(tmp_path, chord):
    ledger = Ledger.create(tmp_path / "l")
    execute(chord, ledger)
    first = markdown(build(ledger))
    assert first == markdown(build(Ledger.open(tmp_path / "l")))
    for fragment in ("integrity **verified**", "the physical interpretation is unresolved", "`closed-form-endpoint`",
                     r"\Gamma^k_{ij}", "synthetic", "Acquisition request"):
        assert fragment in first
    with pytest.raises(Refusal):
        build(ledger, ["no-such-experiment"])
