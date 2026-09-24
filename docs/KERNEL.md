# CIW kernel (`ciw.kernel.v1`)

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
| Projection | `spatial.list`, `spatial.inspect`, `fusion.list`, `instrument.list`, `instrument.inspect`, `candidate.list`, `candidate.get` | Read-only views that grew around individual kinds. Kept for existing clients, closed to additions, to be subsumed by `experiment.inspect` and the project graph. |

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

## Persistence and the project graph

Workspace format 3 is the final format minted because a family of records was
added. Later record types are records inside the workspace.

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
verification, refusal vocabulary, the domain rules its implementation keeps,
investigations, operator surface and implementation. `pipelines.check()` binds
every descriptor to the code that executes it, including exact pin equality, so
a descriptor cannot drift. Three investigations group the pipelines; the
operator surface is `operation.list` with `{"view": "investigations"}`, while
every pipeline stays executable and replayable. [PIPELINES.md](PIPELINES.md) is
generated from the descriptors, including the provider pin matrix that keys
provider gates.
