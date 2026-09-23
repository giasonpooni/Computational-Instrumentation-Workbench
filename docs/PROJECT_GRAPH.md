# Project-graph compiler operation

`ciw.project-graph.v1` registers the [project model](CONTRACT_FOUNDATIONS.md#project-model)
in the shared operation, execution and result envelopes. It replays a retained
`ciw.project.v1` history and compiles it into a deterministic plan. It does not
dispatch a declared computation, fetch evidence, validate a physical claim or
admit state.

| Item | Value |
| --- | --- |
| Workbench kind | `project-graph` (provider-free, available without bindings) |
| Operation | `ciw.project-graph.v1` |
| Source schema | `ciw.project-graph-source.v1` |
| Plan schema | `ciw.project-graph-plan.v1` |
| Session / result / verification | `ciw.project-graph-session.v1`, `ciw.project-graph-workbench-result.v1`, `ciw.project-graph-verification.v1` |
| Replay receipt | `ciw.project-graph-replay.v1` |
| Runtime | `ciw.python-reference-runtime.v1`, profile `ciw.project-graph.python-reference.v1` |
| Implementation | [`project_workflow.py`](../src/ciw/project_workflow.py) on the shared [`native_workflow.py`](../src/ciw/native_workflow.py) lifecycle |

## Source

```json
{
  "schema": "ciw.project-graph-source.v1",
  "experiment_id": "project:compile-fixture",
  "configuration": {"profile": "project_graph_compile", "execution": "not_performed",
                    "physical_validation": "not_performed", "state_admission": "not_performed",
                    "hardware_actuation": "not_performed"},
  "project": {"schema": "ciw.project.v1", "project_id": "...", "history": ["..."], "revision": "sha256:..."},
  "request": {"project_revision": "sha256:...", "targets": null}
}
```

`request.project_revision` must equal the retained project's final revision, so
a compile request cannot silently apply to a different history. `targets` is
`null` for the whole graph or a unique list of at most 64 declared object IDs;
the plan then contains those objects and their computation ancestors. The
source is limited to 512 KiB of exact JSON bytes without duplicate keys or
nonfinite numbers.

## Plan

The plan is computed without clocks, random identifiers or live registries, so
a replay reproduces its numerical identity exactly.

- `order`: a topological order over computation edges with a lexical
  tie-break; objects outside any computation edge appear in identifier order.
- `steps`: each object's kind, revision, computation parents and evidence
  references. Computations report `declared_operation`, a digest of their
  parameters and `dispatch: not_performed`. Results report their pinned
  `input_revisions`, `drifted_inputs` and `unpinned_ancestors`.
- `needs_reevaluation`: results whose pins drifted, whose upstream result is
  stale or whose computation ancestors are unpinned, as defined by
  `project_model.inspect`.
- `unresolved_physical_edges`, `context_status` and `unevidenced_results`
  keep draft state visible.
- `model_checks`: `thermal-model` content checked against the thermal
  request model contract, and candidate machine manifests compiled against
  their linked evidence bundle and challenge report. A failing check is
  recorded as `contract_invalid`, `unlinked` or `refused_by_manifest_compiler`.
- `inspection_digest` binds the complete `ciw.project-inspection.v1` view.

Drafts, stale results and failing model checks are findings, not refusals. A
malformed history, a digest mismatch, a changed authority policy, a revision
mismatch or an undeclared target refuses at `source.add`, and nothing is
retained.

## Use

```sh
ciw serve --output-dir results/project
ciw send source.add --payload-file project-source-payload.json
ciw send operation.execute --payload '{"operation_id": "ciw.project-graph.v1", "parameters": {"source_id": "source:sha256:..."}}'
ciw send experiment.inspect --payload '{"bundle_id": "sha256:..."}'
ciw send bundle.replay --payload '{"bundle_id": "sha256:..."}'
```

`source.add` takes `{"kind": "project-graph", "label": ..., "bytes_b64": ...}`
with the exact source bytes. Saving and reopening a workspace revalidates the
plan with the Python reference and never executes a provider. A replay creates
fresh execution and result identities with the same numerical-result identity
and a `ciw.project-graph-replay.v1` receipt with `admission: not_performed`.

## Validation

```sh
python -m pytest -q tests/test_project_model.py tests/test_project_workflow.py
```

The tests cover plan order, drifted inputs, invalid thermal models, target
ancestry, source refusals, save/reopen without execution, fresh replay, a
changed runtime identity and tampered retained plans.

## Limits

The plan records what the history declares. It does not resolve operation IDs
against a live provider, execute a declared computation, check that evidence
is authentic or establish that a model describes a physical machine. Executing
a planned computation remains a separate operation with its own source,
bindings and retained evidence.
