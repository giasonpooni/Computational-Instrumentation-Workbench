# CIW kernel (`ciw.kernel.v2`)

The kernel is the part of the workbench that stays expensive: identities,
admission, persistence, reopen/replay, inspection and binding. Scientific work
extends the workbench by composing providers and adding records, never by
adding wire verbs, envelopes, workspace formats or source kinds. The executable
declaration is [`src/ciw/kernel.py`](../src/ciw/kernel.py);
[`tests/test_kernel_freeze.py`](../tests/test_kernel_freeze.py) checks the
session dispatcher, the workbench registry and the workspace writer against it,
and the session refuses any verb it does not declare.

    Investigations (named entry points over pipelines)
            |
    Declared pipelines (data: inputs, provider steps, pins, refusals)
            |
    Kernel (envelopes, admission, persist, reopen/replay, inspect, bind)
            |
    Providers (independent repositories at exact pins)

## Invariants

- Evidence, operation, execution, result and verification identities stay
  distinct; a numerical check, content digest or replay equality never creates
  a verification identity.
- Reopen validates and reads retained records; it never executes a provider.
  Replay is an explicit new execution with fresh execution and result
  identities.
- A saved workspace never chooses an executable. Provider checkouts are bound by
  the operator against an allowlisted pin; saved content cannot extend the list.
- Pins are exact revisions and runtime identities, never a moving branch.
- Provider repositories stay independent; CIW binds them, it does not absorb
  them.

## Wire surface

Protocol version 1. The request, response, error and broadcast envelopes are
specified in [PROTOCOL.md](PROTOCOL.md#transport).

| Class | Verbs | Rule |
| --- | --- | --- |
| Kernel | `session.get`, `run.get`, `source.add`, `source.get`, `source.list`, `operation.list`, `operation.execute`, `execution.list`, `bundle.list`, `bundle.get`, `bundle.replay`, `result.list`, `result.get`, `experiment.inspect`, `selection.update`, `sample.get`, `workspace.save` | Stable. New scientific operations are `operation.execute` payloads; new records are read through these verbs. |
| Legacy | `analysis.stats`, `analysis.spectrum` | The original oscillator analyses and their flat result format. |

## Inspection views

Every read-only view of retained records is a payload of `experiment.inspect`,
selected by `view` (default `experiment`):

| View | Payload | Result |
| --- | --- | --- |
| `experiment` | `bundle_id` | The kind's experiment projection |
| `instruments` | none | Retained native instrument steps across bundles |
| `instrument` | `bundle_id`, `instrument` | One native step, its fusion context and linked results |
| `fusion` | none | Retained compatible-state contexts and lineage |
| `spatial` | optional `source_id` | Geographic context sources, or one `ciw.spatial-view.v1` packet |
| `candidates` | none | Historical candidate-action receipts |
| `candidate` | `candidate_id` | One receipt with its exact native response |

`ciw.kernel.v2` removed the seven projection verbs that preceded these views
(`spatial.list`, `spatial.inspect`, `fusion.list`, `instrument.list`,
`instrument.inspect`, `candidate.list`, `candidate.get`). A client that still
sends one receives `unknown_command` naming the view to use. The `/spatial`
endpoint admits only the `spatial` view. The project graph remains the
`workbench.project` field of `session.get`.

## Envelopes and admission

| Envelope | Meaning |
| --- | --- |
| `ciw.workbench-source.v1` | Retained source bytes with source and evidence identities |
| `ciw.execution.v1` | One execution occurrence, completed or refused |
| `ciw.operation-result.v1` | The sealed result of a completed execution |
| `covariance-artifact.v1` | Ordered quantities, units, frame and full matrix |

Admission: a malformed request is rejected before an execution exists; an
unsupported, unbound or failed operation is retained as a refused execution with
no result; a completed operation seals its execution and result. Domain payloads
live in their own schemas inside these envelopes.

Workflow kinds follow the same rule. An unbound provider, a provider refusal or
a failed check during `operation.execute` or `bundle.replay` returns
`{"status": "refused", "execution": ..., "result": null}` and retains a
`ciw.workbench-refusal.v1` record: its own execution identity, the operation,
source and evidence identities, any upstream bundles, the replayed bundle for a
refused replay, and the refusal code and message, sealed by a record digest.
It appears in `execution.list`, persists in the workspace and is validated on
reopen; it is never a result, a bundle or a graph node. Capacity exhaustion is
the one rejection that is not retained.

## Persistence and the project graph

Workspace format 3 is the final format minted because a family of records was
added. Later record types are records inside the workspace: the retained
workbench is `ciw.retained-workbench.v1`, `.v2` once it holds candidate-action
receipts and `.v3` once it holds refused executions, and a reader accepts all
three.

`session.get` returns `workbench.project`, a `ciw.project-graph-view.v1` read of
the retained records as the [`ciw.project.v1`](../src/ciw/project_model.py) graph
([`project_graph.py`](../src/ciw/project_graph.py)):

- retained sources are evidence nodes;
- each source kind's operation at its recorded provider pins is a computation
  node, revised when the pins change (host paths are binding details and never
  pins);
- each retained bundle is an occurrence of a `workflow_result` node whose
  revision binds its replay-invariant numerical result identities and every
  computational ancestor's revision.

Reopen is reading the graph. Replay re-evaluates a node: reproduced numbers keep
the node's revision; changed numbers, a corrected upstream result or a pin
change mark dependents `needs_reevaluation`. The graph view has no execution,
physical-validation or state-admission authority.

The view's `investigations` list reads the same graph against the
[investigation catalog](../src/ciw/pipelines/investigations.json). It reports:

- for each member pipeline, its result nodes and which of them are current;
- the default-pipeline stages that still lack a current result;
- the lineage chains that connect results of the investigation's pipelines,
  for example calibrated observable → identified design → identified stability.
  A chain is current only while every node in it is.

The live view passes the pins each kind is bound to now. Shared-runner provider
kinds pass the runtime identity checked at binding, and provider-free references
pass their current code identity. A result computed under other pins therefore
reads `needs_reevaluation`, as do its dependents and any chain containing it,
until it is replayed. An investigation is `not_started`, `incomplete`, or
`default_pipeline_current`. `ciw investigations WORKSPACE [--json]` prints the
same progress for a saved workspace without executing or writing anything.
Like the rest of the view, this is computed from retained records and never
executes anything.

## Source kinds

The 25 kinds registered when the kernel was frozen are listed in
`kernel.FROZEN_KINDS` (plus the source-only `geographic-context`). A new kind is
not added until a project-graph operation exists or an investigation is
delivered end to end with a held-out measurement; new work composes existing
kinds.

## Declared pipelines and investigations

Each frozen kind has a `ciw.pipeline-descriptor.v1` in
[`src/ciw/pipelines/descriptors`](../src/ciw/pipelines/descriptors): operation
identity, inputs and upstream kinds, provider steps with exact pins,
verification, refusal vocabulary, the domain rules its implementation keeps
(each citing the code that enforces it as `module:qualname` references that
must resolve), investigations, operator surface and implementation. `pipelines.check()` binds
every descriptor to the code that executes it. Descriptors are the only
definition of a provider pin: modules read their pins through `pin_map`,
`provider_pin` or `provider_descriptor`, and the check refuses a revision
literal anywhere else in the package, so what executes cannot drift from what
is declared. Three investigations group the pipelines; the
operator surface is `operation.list` with `{"view": "investigations"}`, while
every pipeline stays executable and replayable. [PIPELINES.md](PIPELINES.md) is
generated from the descriptors, including the provider pin matrix that keys
provider gates.

## Shared runner

A declared pipeline with one pinned provider step runs through
[`ciw.pipelines.runner`](../src/ciw/pipelines/runner.py). The runner owns every
record that carries an identity (step, result, bundle, same-runtime
reproduction, replay receipt) and every pin check; the pipeline supplies named
hooks only: `parse_source`, `invoke`, `check_data`, `check_runtime`,
`make_adapter` and `bind_extra`. A descriptor whose runner is `generic_runner`
is refused by `pipelines.check()` if its class overrides anything else.

A pipeline that composes several providers inside one step (variational free
energy: CSG, then GSIE and PLSR) uses the same runner. It seals each companion
call with `StageChain`, re-checks the sequence with `check_chain` (exact roles
and operations in order, cumulative `input_refs`, distinct occurrences) and
overrides `_step`, `_validate_step` and `_check_runtimes`. The bundle,
verification and replay receipt remain the runner's records.
Native occurrences inside a step reach the workbench through three workflow
hooks, `catalog_steps`, `identity_claims` and `native_occurrences` (runner
defaults: none), so the workbench has no per-kind branch for them.
Each descriptor's `implementation.entry` names the symbol that is the kind's
workflow and how it is built (`none`: a class with no argument, `kind`: a class
given the source kind, `value`: used as-is). The workbench builds workflows
only from these entries, and `pipelines.check()` binds every entry to its
declared module and runner class.
`implementation.view` names the kind's `experiment.inspect` projector and
whether it takes fusion context, so the inspection verb has no per-kind branch.

Pipelines whose records carry their own schemas and authority (the provider-free
thermal, machine-manifest and energy-accuracy references) set a `RecordProfile`
on the runner: result schema, verification schema and method, and authority. A
step whose retained request differs from its source overrides
`step_request(source, evidence_id)`; a source that names its experiment or
configuration differently overrides `experiment_id` or `configuration`.
`check_data` always receives the whole source.

### What the workbench reads from descriptors and workflows

The workbench names no kind except the source-only geographic context. Every
kind-specific decision comes from the kind's descriptor or from an optional
hook on its workflow.

| Decision | Source |
| --- | --- |
| Which workflow runs the kind | `implementation.entry` |
| `experiment.inspect` projector, fusion context | `implementation.view`; `mapped_source` hook swaps the declaration |
| Declared, reproduced and contract-validated kinds | `verification.method` |
| Single or ordered upstream selection | `inputs.upstream_kinds`, `inputs.upstream_cardinality` |
| Operator configuration separate from the source | `inputs.configuration` |
| Providers the operator may leave unbound | `steps[].optional` |
| Operation listing role | `operation_role` |
| Binding optional providers or a separate configuration | `check_bindings`, `select_bindings`, `replay_bindings` hooks |
| Binding a selected upstream | `validate_upstream` (one) or `requested_upstream_ids` and `validate_upstreams` (ordered) |
| Replay rules beyond the runner's | `validate_replay` hook |
| Embedded native occurrences | `catalog_steps`, `identity_claims`, `native_occurrences` hooks |
| Extra inspection summary fields | `summary_fields` hook |
| Roles whose identical result may repeat on replay | `REUSABLE_RESULT_ROLES` |
| Fusion context of the pinned-set kinds | [`ciw.fusion_context`](../src/ciw/fusion_context.py) |

`pipelines.check()` ties the descriptor data to the hooks. Its checks cover
upstream cardinality against the upstream hooks, optional providers and
separate configuration against the binding hooks, and entries, views and
domain-rule code references against importable symbols. Hosts bind any kind
role by role with `ciw serve --bind-role KIND:ROLE=PATH`.

A pipeline verified by a proof instead of a reproduction (proved heat: an SP1
proof checked against the registered guest) overrides `_verify`,
`_check_verification` and `_check_receipts`; the bundle, source evidence and
step envelopes stay the runner's (`_check_envelope`, `check_step`). Canonical
record content and its identities live in
[`ciw.core.canonical`](../src/ciw/core/canonical.py).

## Providers outside pipelines

A provider that no frozen pipeline step names (the Julia model worker) is
declared by a `ciw.provider-descriptor.v1` in
[`src/ciw/pipelines/providers`](../src/ciw/pipelines/providers). Its `pin` is the
definition the implementation executes, and `pipelines.check()` compares it with
the implementation's `provider_binding()`. Adding a provider is a descriptor, a
pin and a gate entry; it adds no kernel verb, kind or workflow file.

## CI

One workflow, [`ci.yml`](../.github/workflows/ci.yml), with three jobs:
`descriptors` binds every descriptor to code and derives the matrices, `kernel`
runs the kernel surfaces, and `providers` runs one row per gate, platform and
Python, keyed by the digest of the exact pins the gate binds. Gates and pins
beyond the descriptors are declared in [`ci/gates.json`](../ci/gates.json);
`python scripts/ci_matrix.py check` refuses a pinned pipeline or provider
without a gate and any revision restated in a gate script or workflow.
