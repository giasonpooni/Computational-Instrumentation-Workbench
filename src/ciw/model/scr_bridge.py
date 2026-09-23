"""Run the Julia worker as an SCR ``SpecificationDispatcher.runner``.

SCR stays authoritative for dispatch and evidence admission. This bridge only
adapts one checked worker execution to SCR's ``ExecutionResult`` type, which
the caller supplies from its pinned SCR checkout; CIW does not vendor SCR.
A halted run returns a halted result (SCR then records a failed dispatch and
fabricates no measurement); a refused program raises before anything runs.

Occurrence semantics: SCR writes ``engine_occurrence`` into the admitted
record's raw content, so it feeds the record and observation identities. SCR's
engine is one fresh process per request and always reports 0. The persistent
worker builds fresh problem and solver state for every request, so each request
is likewise a fresh engine trace and reports engine occurrence 0. The worker
session identity and its monotonically increasing occurrence number are
execution history: they are kept in ``ledger`` (and in CIW's retained run
records), never inside a content identity. Repeating a request therefore adds
an occurrence without changing the admitted evidence identity.
"""
from __future__ import annotations

from typing import Callable

from .worker import JuliaWorker, WorkerResult, program_bytes

ENGINE_OCCURRENCE = 0


def make_runner(worker: JuliaWorker, operation: str, result_type: type, *, timeout: float | None = None,
                ledger: list[WorkerResult] | None = None) -> Callable:
    """Return ``runner(spec) -> result_type`` for SCR's dispatcher."""

    def runner(spec):
        worker.start()
        if spec.program != program_bytes(operation, worker.runtime_digest):
            # The worker would refuse too; failing here keeps the request unsent.
            raise ValueError("SCR specification program is not this worker's registered operation")
        options = {} if timeout is None else {"timeout": timeout}
        result = worker.execute(operation, spec.configuration, spec.input_payload, **options)
        if ledger is not None:
            ledger.append(result)
        if result.specification_identity != spec.identity():
            raise ValueError("SCR and CIW disagree on the specification identity")
        return result_type(
            specification=spec, specification_identity=result.specification_identity,
            program_identity=result.program_identity, input_identity=result.input_identity,
            engine_occurrence=ENGINE_OCCURRENCE, status=result.status, exit_code=result.exit_code,
            output=result.output, output_identity=result.output_identity,
            computation_identity=result.computation_identity, detail=result.detail)

    return runner
