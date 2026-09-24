# Contract foundations for typed engineering projects

This increment checks three provider-free foundations that were previously
present only as local drafts. The machine manifest, thermal profile and project graph are
now registered as shared workbench operations backed by independent Python
references.

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

## Project graph operation

`src/ciw/project_workflow.py` registers the project graph as the shared
provider-free operation `ciw.project-graph.v1`. A retained source carries one
exact `ciw.project.v1` artifact, the read-only configuration and a request
naming the project revision the operator reviewed. A source whose artifact
revision differs from that request, whose history chain or digests are broken,
or whose configuration claims execution is refused before retention. Execution
replays the history through the independent Python reference and retains the
complete inspection as the native result: object and edge counts, result
staleness against pinned input revisions, unresolved physical edges and
evidence-bound context status, with separate operation, execution, result and
numerical-result identities. Save/reopen validates the retained inspection
against a fresh deterministic replay without executing a provider;
`bundle.replay` records a new occurrence with a matching numerical identity.
Declared computations are never executed, evidence is never fetched, and
physical validation and state admission remain `not_performed`.
`tests/test_project_workflow.py` covers the lifecycle, stale-revision and
broken-history refusals, retained-record tamper refusal and runtime-identity
mismatch. `examples/project-graph/make_source.py` prints a complete source.

## Thermal observer reference

`src/ciw/thermal_contract.py` and `src/ciw/thermal_reference.py` define a
two-capacity linear thermal observer profile. The Python reference computes
zero-order-hold matrices, Joseph-form measurement updates with explicit
dropout, and a bounded information-gain sensor-selection problem. A retained
source is explicitly synthetic and carries `physical_validation:
not_established`.

The contract can validate a future Julia result against the independent Python
reference, including state/input/sensor ordering, symbolic-rendering metadata,
solver status, objective bounds and held-out diagnostics. It does not claim a
Julia runtime is installed or that the model describes a physical machine.
The Julia worker remains outside the packaged operation path until its project
and manifest are instantiated, its runtime identity is pinned, and successful,
refusal and replay gates run in CI.

## Shared reference lifecycle

The thermal, machine-manifest, project-graph, uncertainty-validation and
energy-accuracy operations share one lifecycle in
`src/ciw/reference_workflow.py`: exact source retention, a fresh execution
occurrence with separate result and numerical-result identities, a verification
that reproduces the occurrence in the same process, save/reopen validation
without a provider, and replay with a receipt whose verification is recomputed
field for field. Each workflow supplies its constants and four hooks: source
validation, native data, the runtime identity it publishes and the
configuration a bundle retains. Three further hooks default sensibly: the
experiment identity a bundle records, the request a step retains and the shape
check on a retained runtime identity. The thermal workflow overrides the data
check to use the contract's tolerances while pinning the Python reference's
provenance. The energy workflow keeps its own verification method, a request
derived from the log digest and run identity, and its flat pre-existing runtime
identity, so energy workspaces saved before the consolidation still reopen.
The shared module is part of every reference's algorithm identity, so a change
to it is a runtime change for replay.

Reopening is deliberately more tolerant than replay. The machine-manifest and
project-graph references are pure arithmetic and must reproduce a retained
result bit for bit. The energy, thermal and uncertainty-validation references
run through the host's linear-algebra kernels, which OpenBLAS selects by CPU
core type, so their roundoff-level quantities (error metrics near zero,
conditioned covariances) differ between hosts without changing any
classification; their reopen check compares structure, labels, counts and
decimal strings exactly and numbers with the shared binary64 tolerance
(`close_data` in `reference_workflow.py`, relative 1e-9, absolute 1e-12).
Replay still requires the fresh numerical result to match the retained one
exactly: on a host whose kernels round differently, replay of one of these
three kinds is refused as a numerical mismatch while the retained bundle stays
valid. The published runtime identity records Python and NumPy versions but
not the kernel, so such a refusal names the numbers rather than the runtime;
recording the kernel in the identity is open work.

A committed retained workspace under `tests/fixtures/retained/` is the reopen
gate for every change to the shared module or its subclasses: the current code
must validate each retained bundle, reproduce its numbers from the retained
source (bit for bit for the exact kinds, within tolerance for the others), and
replay it freshly, refuse on runtime identity, or refuse as a numerical
mismatch for a kernel-sensitive kind only.

## Current boundary and next gate

The thermal profile is now registered as the provider-free
`ciw.thermal-observer.v1` operation. It uses the independent Python reference
for its first executable occurrence and preserves operation, execution, result,
source and replay identities through the shared save/reopen path. The machine
manifest is registered through `ciw.encoder-position.v1` and the project graph
through `ciw.project-graph.v1`; independent ICRH profiles for these three
provider-free operations remain pending. The next thermal gate is a separately verified Julia
worker with an instantiated environment and a cross-language replay check.
Provider bindings must remain host configuration; saved artifacts may not
choose an executable or extend the allowlist.
