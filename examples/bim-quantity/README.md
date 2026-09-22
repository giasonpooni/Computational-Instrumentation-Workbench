# Native BIM quantity experiment

This example opens CSE in the same workbench as the instrumentation workflows.
The source retains the exact `room.ifc` bytes and a scalar observation, including
its variance, IFC content identity, target entity and quantity, frame declaration,
and explicit measurement/prior cross-covariance policy.

```sh
python examples/bim-quantity/make_source.py > /tmp/bim-source.json
```

Register those bytes as a `bim-quantity` source, then execute
`ciw.bim-quantity.v1` with its `source_id`. The host must bind the pinned CSE
repository; source files cannot select a provider path. The shared bundle,
result, execution and replay interfaces retain the complete native result.

The declared IFC clear height is 3 m. CSE supplies its versioned IFC prior
policy (height standard deviation 0.01 m). The example observation is 2.99 m
with variance 0.000025 m² and explicitly independent noise. CSE's native
`GatSession` compiles the IFC, conditions the raw quantity, propagates derived
quantities, checks its invariants and replays its hash-chained ledger. The
posterior height is 2.992 m, its variance is 0.00002 m², and the room volume
is 59.84 m³. These are synthetic declarations, not calibrated field evidence.

Unknown cross-covariance, missing or differing frame declarations, an incorrect
IFC/target binding, incompatible units, or an unsupported derived target yield
`held` with the unchanged native world. A negative height observation that
violates native constraints yields `refused` with a replayable rejected event.
An IFC parse/lowering refusal retains the exact source and typed native error
without constructing a world. Source syntax outside this bounded contract is
rejected before provider execution.

The operation permits 256 IFC records, 64 full state quantities and one scalar
raw metre quantity per execution. CSE normalizes explicitly declared supported
IFC length units to metres. Undeclared assumed metres remain held. The current
state-byte identity contract requires a little-endian native host.
Retained raw and full covariance matrices must also pass the workbench's
positive-semidefinite check in dimensionless correlation coordinates (minimum
eigenvalue tolerance `-1e-10`). Zero variances require exactly zero covariance
rows. The check permits the native derived-state singularity without repairing
or replacing any supplied matrix entries.

Geometry authority remains `QUANTITY_ONLY`. Matching supplied frame labels is
an applicability check; it does not establish surveyed frame authority or
perform a GTE transformation. Independent cross-covariance is a retained
declaration. This workload does not admit construction state, certify as-built
clearance, or silently convert GSIE/CBSR outputs into new independent evidence.

The native result includes complete before/after quantity means and covariance,
IFC unit provenance, invariants, ledger events and native ledger-replay outcome.
Workbench verification records fresh execution under the same pinned runtime
with `independent: false`; the ICRH profile independently checks the declared
scalar update and retention contract.
