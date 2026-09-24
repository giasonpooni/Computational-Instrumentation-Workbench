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
every descriptor to the code that executes it, including exact pin equality, so
a descriptor cannot drift. Three investigations group the pipelines; the
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
is refused by `pipelines.check()` if its class overrides anything else. Canonical
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
