# Project graph inspection

This example retains a small versioned project graph in the shared workbench
and executes the provider-free `ciw.project-graph.v1` operation over it. The
graph declares one evidence object, one observed signal, one computation and
one estimated result whose input revisions are pinned to the signal and
computation revisions current when the result was recorded.

```sh
python examples/project-graph/make_source.py > /tmp/project-source.json
```

Register those bytes as a `project-graph` source, then execute
`ciw.project-graph.v1` with its `source_id`. No repository binding is needed;
the operation is served by the independent Python reference in
`src/ciw/project_model.py`. The request names the project revision the operator
reviewed; a source whose artifact revision differs from that request, or whose
history chain or digests are broken, is refused before retention.

The retained result is the complete inspection: object and edge counts, the
`result_status` of every result against its pinned input revisions, unresolved
physical edges, evidence-bound context status and the overall `declared` or
`draft` status. For this example the result is `current_for_declared_inputs`
and the graph is `declared`. Changing the signal's frame in a later revision
would leave the result `needs_reevaluation` without executing anything.

Declared computations are never executed, evidence is never fetched, and
physical validation and state admission remain `not_performed`. Reopening a
saved workspace validates the retained inspection against a fresh deterministic
replay without a provider; `bundle.replay` records a new execution and result
occurrence with a matching numerical identity.
