# Native acquisition in the shared workbench

`ciw.acquired-dataset.v1` executes PPDA's source registry, adapter registry,
incremental acquisition plan, durable evidence pool and checkpoint runner over
exact retained local snapshots. The resulting dataset shares the workbench's
source, operation, execution, result, replay and inspection interfaces.

Use `kind: acquired-dataset` in `source.add` with the bytes of
`examples/acquired-dataset/source.json`, then execute `ciw.acquired-dataset.v1`
with its `source_id`. Host setup binds the `ppda` role to a complete checkout;
source files cannot choose executable code or filesystem paths.

The provider pins are PPDA `477d6cb454423a27543b16961d3b169709c40c31` and its
`vendor/scout-retrieval-agent` gitlink
`5e146d5924675cd7b6e1d1ed44fb39f5da012610`. Initialize the submodule before
binding. Both complete source trees are verified before and after execution.
The earlier standalone PPDA telemetry projection and its provider pin remain
separate. Python 3.12 and the standard library are sufficient for this lane.

The source declares a registry identity, name, domain, plan identifier, and
1–8 snapshots. Each snapshot holds an explicit request timestamp and exact
base64-encoded JSON bytes. Arrays contain at most 64 records with declared,
strictly increasing nonnegative sequence numbers; each later snapshot must
retain the preceding records unchanged and may append records. Revisions of an
earlier record are refused by this incremental lane. Equal snapshots are valid:
the native runner executes a successful empty acquisition without moving its
position. Request timestamps must increase and include a timezone.

Every occurrence uses the native `execute_plan` path. The pool is reopened from
its durable store between snapshots. Results retain the registered source and
plan, native acquisition outcomes, artifacts, checkpoint transitions, pool
fingerprints, and complete native sources/documents/records/observations. Raw
snapshot bytes remain in the source even though the native adapter serializes
individual JSON rows into its record representation. A stable relative dataset
filename prevents random temporary directories from changing record identity.

The synthetic example acquires two records, appends one, then reacquires an
unchanged snapshot. Artifact counts are 2, 1, 0; positions are 2, 3, 3. One
channel has explicit missingness and a null value. Device times 10, 9, 12 remain
unchanged. The example retains absent clock/calibration maps and unknown
cross-covariance. Acquisition sequence order does not establish physical event
order, and evidence retention does not make an observation usable by GSIE.
There is no implicit dataset-to-estimator conversion, hardware polling, clock
correction, calibration, uncertainty completion or canonical-state admission.

Creation and replay each perform a fresh native reproduction. Retained
verification explicitly says `independent: false`; it establishes reproduction
by the same pinned implementation. Offline inspection checks source bytes,
native content identities, lineage, cursor progression and wrapper bindings
without executing a provider. Native SCOUT evidence admission remains distinct
from ESM or scientific-state admission. Executable bindings are host-owned and
are not restored from a workspace.

Run the native and adversarial tests with:

```sh
CIW_PPDA_REPO=/trusted/ppda python -m pytest -q tests/test_acquired_dataset.py
```

`CIW_ACQUISITION_FIXTURE_DIR` optionally writes actual original/replay fixtures.
The tests cover unknown covariance and missing-value retention, empty runs,
durable restoration, unchanged timestamps, native vendor drift, lineage
tampering, source ambiguity and malformed snapshots.
