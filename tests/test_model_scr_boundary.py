"""The Julia worker below SCR's existing dispatch and admission seam.

Requires ``CIW_SCR_REPO`` (the pinned Scientific-Computation-Runtime checkout)
and ``CIW_JULIA``. SCR builds the ``ExecutionSpecification``; the worker runs
it; SCR's ``SpecificationDispatcher`` admits the result as ``simulation:``
evidence through ``run_experiment_step``. No second admission route exists.
"""
import json
import os
from pathlib import Path
import sys

import pytest

from ciw.model import codec
from ciw.model.providers import committed_bytes
from ciw.model.scr_bridge import make_runner
from ciw.model.spec import validate_spec
from ciw.model.worker import JuliaBinding, JuliaWorker, program_bytes

SCR = os.environ.get("CIW_SCR_REPO")
JULIA = os.environ.get("CIW_JULIA")
pytestmark = pytest.mark.skipif(not (SCR and JULIA), reason="CIW_SCR_REPO and CIW_JULIA bind SCR and Julia")
EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "model-core"
OPERATION = "ciw.model.simulate.v1"


@pytest.fixture(scope="module")
def scr():
    sys.path.insert(0, str(Path(SCR).resolve()))
    import execution.engine
    yield execution
    sys.path.remove(str(Path(SCR).resolve()))


def _campaign(scr_modules, runner, spec_for, interpret):
    from evidence.admission import admit_document, admit_referent
    from evidence.pool import EvidencePool
    from evidence.types import make_document, make_referent, make_source
    from execution.dispatcher import SpecificationDispatcher
    from experiment.policy import ExperimentPolicy
    from experiment.session import make_experiment_session
    from experiment.step import run_experiment_step
    from materials.candidates import generate_candidates
    from materials.decision import make_criterion
    from materials.iteration import reevaluate_program
    from materials.optimization import OptimizationPolicy
    from materials.program import make_material_program_query
    from materials.selection import SelectionPolicy
    from materials.utility import ExperimentUtilityInput
    from retrieval.engine import DeterministicRetrievalEngine

    engine = DeterministicRetrievalEngine()
    criterion = make_criterion("mechanical_energy", "<=", 1)
    pool = EvidencePool()
    source = make_source(kind="computational_campaign", name="CIW Julia model core")
    pool.put_source(source)
    document = make_document(source_id=source.id, raw_content="ciw julia model session",
                             retrieval_method="manual_entry", retrieved_at="2026-09-23T00:00:00Z")
    admit_document(pool, document)
    pool.put_document(document)
    for key, kind in (("process-damped-oscillator", "process"), ("formulation-oscillator", "formulation")):
        referent = make_referent(natural_key=key, kind=kind)
        admit_referent(pool, referent)
        pool.put_referent(referent)
    query = make_material_program_query(["formulation-oscillator"], "process-damped-oscillator", ("mechanical_energy",))
    iteration = reevaluate_program(pool, engine, query, (criterion,))
    session = make_experiment_session(pool, engine, iteration, document_id=document.id)
    policy = ExperimentPolicy(
        selection_policy=SelectionPolicy(allowed_action_classes=None, allow_already_represented_context=True,
                                         allow_redundant=True, allow_not_determinable_feasibility=True,
                                         max_selected=None),
        optimization_policy=OptimizationPolicy(max_candidates=1, allowed_action_classes=None,
                                               allow_indeterminate_utility=True),
        utility_input_source=lambda estimate: ExperimentUtilityInput(benefit=1.0, cost=1.0))
    dispatcher = SpecificationDispatcher(spec_for=spec_for, interpret=interpret,
                                         extracted_at="2026-09-23T00:00:00Z", runner=runner)
    step = run_experiment_step(session, generate_candidates(session.iteration.specification), dispatcher,
                               policy, confidence=1.0)
    return pool, step


def test_julia_computation_is_admitted_through_the_existing_scr_seam(scr):
    from execution.engine import ExecutionResult
    from execution.specification import ExecutionSpecification
    model = validate_spec(json.loads((EXAMPLES / "damped-oscillator.json").read_text()))
    request = json.loads((EXAMPLES / "oscillator-simulate.json").read_text())
    configuration, payload = committed_bytes(OPERATION, model, request)
    observations = []
    with JuliaWorker(JuliaBinding(julia=Path(JULIA))) as worker:
        worker.start()

        def spec_for(candidate):
            return ExecutionSpecification(program=program_bytes(OPERATION, worker.runtime_digest),
                                          configuration=configuration, input_payload=payload)

        def interpret(candidate, result):
            output = codec.decode(result.output)
            # Semantic content is the computed value; execution bookkeeping stays in the record.
            return {"property": candidate.property, "value": round(float(output["derived"]["E"].array[-1]), 12),
                    "unit": "J"}

        ledger = []
        runner = make_runner(worker, OPERATION, ExecutionResult, ledger=ledger)
        for _ in range(2):
            pool, step = _campaign(scr, runner, spec_for, interpret)
            assert pool.has_observation(step.observation.id)
            assert step.observation.extraction_method == "simulation:deterministic_native_execution"
            assert "computation" in step.dispatched.record_raw_content
            assert "occurrence" not in step.observation.content
            observations.append((step.observation.id, step.dispatched.record_raw_content))
        identity = spec_for(None).identity()
    # Two executions (distinct worker occurrences, kept in the CIW ledger) admit one reproducible fact.
    assert observations[0][0] == observations[1][0]
    assert [item.occurrence for item in ledger] == [1, 2] and ledger[0].session_id == ledger[1].session_id
    assert ledger[0].computation_identity == ledger[1].computation_identity
    assert "engine_occurrence 0" in observations[0][1] and observations[0][1] == observations[1][1]
    assert f"specification {identity}" in observations[0][1]


def test_halted_julia_run_fabricates_no_measurement(scr):
    from execution.engine import ExecutionResult
    from execution.specification import ExecutionSpecification
    from ciw.model.simulation import uniform_request
    model = validate_spec(json.loads((EXAMPLES / "damped-oscillator.json").read_text()))
    request = uniform_request(model, initial_state=[1.0, 0.0], inputs={"F": 0.0}, maxiters=5)
    configuration, payload = committed_bytes(OPERATION, model, request)
    with JuliaWorker(JuliaBinding(julia=Path(JULIA))) as worker:
        worker.start()
        runner = make_runner(worker, OPERATION, ExecutionResult)
        with pytest.raises(RuntimeError, match="no output, no measurement"):
            _campaign(scr, runner, lambda c: ExecutionSpecification(program_bytes(OPERATION, worker.runtime_digest),
                                                                    configuration, payload),
                      lambda c, r: {"property": c.property, "value": 0, "unit": "J"})
