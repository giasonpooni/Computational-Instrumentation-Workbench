# Contract foundations for typed engineering projects

This increment checks three provider-free foundations that were previously
present only as local drafts. The machine manifest and thermal profile are now
registered as shared workbench operations backed by independent Python references;
the project graph remains a reusable contract until its operation adapter is defined.

## Machine manifest

`src/ciw/machine_manifest.py` defines a bounded, evidence-backed encoder /
gearbox / leadscrew profile:

1. `seal` and `validate` retain exact source bytes, claim digests and machine
   identity.
2. `challenge` checks that candidate claims, firmware configuration, homing,
   units and uncertainty are bound to the retained evidence bundle.
3. `compile` accepts only a validated candidate and produces an
   `accepted_read_only` manifest for the declared position equation.
4. `evaluate` propagates the complete six-coordinate joint covariance. It does
   not infer a confidence interval, physical calibration or state admission.

The module never retrieves documents, commands hardware or loads executable
code. Missing evidence and ambiguous count decoding remain unresolved. The
focused tests cover tampered commitments, stale challenges, position
evaluation and the read-only inspection authority.

## Machine manifest operation

`src/ciw/machine_workflow.py` wraps the manifest contract in the shared CIW
operation lifecycle as `ciw.encoder-position.v1`. A retained source must carry
the evidence bundle, candidate manifest and deterministic challenge report. The
adapter emits separate operation, execution, result and numerical-result identities,
then supports save/reopen without provider execution and fresh replay with a new
occurrence. Its authority remains read-only: physical validation, state admission
and hardware actuation are `not_performed`. `tests/test_machine_workflow.py` covers
the lifecycle, tamper refusal and runtime-identity mismatch.

## Project model

`src/ciw/project_model.py` provides the first language-neutral project graph:

- versioned append-only history with content-addressed events;
- typed component, signal, computation, result and evidence objects;
- separate physical, computation and evidence edges;
- an acyclic computation graph;
- explicit unresolved physical edges;
- result input revisions and `needs_reevaluation` status when a pinned input
  changes; and
- context fields whose evidence references remain visible during inspection.

`validate(json.loads(json.dumps(project)))` is the provider-free reopen check.
Inspection reports `execution`, `physical_validation` and `state_admission` as
`not_performed`; the project model does not execute an operation or authorize
an action.

## Thermal observer reference

`src/ciw/thermal_contract.py` and `src/ciw/thermal_reference.py` define a
two-capacity linear thermal observer profile. The Python reference computes
zero-order-hold matrices, Joseph-form measurement updates with explicit
dropout, and a bounded information-gain sensor-selection problem. A retained
source is explicitly synthetic and carries `physical_validation:
not_established`.

The contract can validate a future Julia result against the independent Python
reference. The separate [Julia oscillator operation](JULIA_OSCILLATOR.md) now
supplies the first executable cross-language seam; its generated manifest,
runtime identity and genuine provider/refusal/replay gates remain required.

## Current boundary and next gate

The thermal profile is now registered as the provider-free
`ciw.thermal-observer.v1` operation. It uses the independent Python reference
for its first executable occurrence and preserves operation, execution, result,
source and replay identities through the shared save/reopen path. The machine
manifest is now registered through `ciw.encoder-position.v1`; the project graph
remains a contract module until its own operation adapter is defined. The next thermal gate is a separately verified Julia
worker with an instantiated environment and a cross-language replay check.
Provider bindings must remain host configuration; saved artifacts may not
choose an executable or extend the allowlist.
