# Read-only instrument-exchange inspection

Status: experimental terminal conformance integration. The original inspector
remains read-only; the workbench now also has a bounded typed exchange adapter
(`ciw.instrument-exchange.v1`). It projects a retained producer envelope into
the shared session only after the pinned SET contract accepts every artifact.
It remains a read-only evidence operation: no source admission, physical
validation, state admission or equipment authorization is performed.

## Setup and worked example

Use Python 3.11+ with CIW's ordinary dependencies. The checker uses the standalone
stdlib `state_estimation_testbed/contracts.py` from the source pin in
[`exchange-runtime.json`](../src/ciw/exchange-runtime.json). Acquire that exact
repository revision or export that exact file under the declared path. There is
no automatic download, arbitrary plugin discovery, or fallback to another checker.

```sh
python -m pip install -e '.[dev]'
python -m ciw exchange inspect examples/exchange/observation.json --validator-repo ../State-Estimation-Evaluation-Testbed
```

The example contains synthetic east/north positions in metres and a full
2-by-2 covariance with off-diagonal `0.01 m²`. It reports effective rank two,
preserves the complete matrix and frame, and leaves the unresolved source and
admission references unestablished. It is not a recorded sensor measurement.

Multiple files can be inspected together, preserving distinct observation,
result and verification identities:

```sh
python -m ciw exchange inspect observation.json result.json verification.json --validator-repo ../State-Estimation-Evaluation-Testbed
```

Standard output is complete JSON. No source file is changed, no output file or
workspace is created, and no scientific provider or artifact-supplied code is
executed. The operator explicitly supplies the checker directory; only its
approved source bytes execute, after a SHA-256 match (CRLF normalized to LF).
Its package initializer is not imported. The source hash detects drift; it does
not authenticate the local Python runtime or prove upstream authorship.

## Input and output contracts

| Input schema | Meaning | Identity treatment |
| --- | --- | --- |
| `notation.instrument.observation-batch.v1` | Ordered finite components, units, UTC observation/receipt instants, declared frame and full/absent covariance | `batch_id` remains a caller-declared reference; no content identity is invented |
| `notation.instrument.result-artifact.v1` | Ordered interpreted output, explicit covariance, execution/input/model/calibration references | `result_id` is recomputed using the runtime producer's domain-separated canonical JSON; this checks content consistency, not behavior |
| `notation.instrument.verification-artifact.v1` | Named checks over a referenced subject, derived summary and declared independence | `verification_id` is recomputed with the same producer encoding; failed/indeterminate checks remain failed/indeterminate |

The inspection supports the runtime producer's `sha256:` content identities for
result and verification records. It does not assume all opaque external IDs use
that encoding. The observation ID is not reinterpreted as a digest.

Ordered variable names and units must match covariance axes. Entry `(i,j)` has
units `unit[i] * unit[j]`; no unit conversion, reordering, rescaling, clipping,
symmetrization or diagonal-only approximation occurs. A tangent frame requires
its basis and evaluation point. Observation/receipt times remain separate UTC
instants; no sampling rate or synchronization accuracy is inferred.

`reported`, `estimated` and `propagated` require a matrix. `unknown` and
`not_applicable` require absence and yield `effective_rank: null`. An explicitly
zero matrix has effective rank zero; singular PSD matrices may conform. The
pinned numerical validator checks correlation-scaled symmetry and PSD and
reports its numerical rank/tolerances. These diagnostics do not establish
calibration validity or model adequacy.

The exchange checker uses its own declared dimensionless `1e-12` default
tolerances, distinct from native CIW covariance's `1e-10` PSD tolerance. Near a
numerical threshold these gates can differ; exchange conformance is not native
covariance admission. The actual checker pin and tolerances are in the report.

Output `ciw.exchange-inspection.v1` contains the original parsed artifacts,
each file's exact-byte SHA-256 and byte count, validator source pin, numerical
diagnostics, and links labeled `matched_supplied_reference` or
`unresolved_external_reference`. A match means only that a supplied identity
equals a reference. It does not prove that an execution consumed that input or
that a verifier checked that subject. Unresolved external references are visible,
not fetched or silently admitted. Duplicate supplied identities are refused.

The separate `authority` object always says `may_authorize: false`, native
workspace import `not_performed`, source admission/execution behavior
`not_assessed`, verification independence/physical validation `not_established`.
Producer-side claims such as `numerical_status: unchecked`, caller-declared
component interpretation and failed verification outcomes remain unchanged.

## Refusals and budgets

Exit `0` means input conformance only, even when the input is a valid record of
a failed verification. Exit `2` means a reported input, source-pin or conformance
error. Unsupported schemas, repeated JSON keys, nonfinite/overflowing JSON
numbers, bad identities, inconsistent axes/units, malformed time/frame metadata,
asymmetric or indefinite covariance and source drift refuse the inspection.
There is no partial successful report.

An invocation accepts 1–32 regular files, at most 1 MiB per file and 8 MiB total,
with at most 64 components and 64 JSON nesting levels. These are inspection
budgets, not claims of instrument or streaming capacity. The checker is trusted
local code, not a sandbox for third-party plugins.

## Validation and limits

[`test_exchange.py`](../tests/test_exchange.py) checks bounded parsing, source
pinning, refusal behavior, nonmutation, full/unknown/singular covariance and
adversarial metadata. [`test_exchange_integration.py`](../tests/test_exchange_integration.py)
calls the actual `bridge.instrumentation.observation_batch_v1` and
`execution.instrumentation` builders, constructs honest synthetic native-byte
commitments, and invokes the actual CIW CLI over all three records. It checks
that links resolve without promoting admission, execution behavior or a failed
verification. This is not an actual Rust-engine execution.

From trusted source checkouts with the delivered producer APIs:

| Component | Exercised revision |
| --- | --- |
| State Estimation Evaluation Testbed checker | `542e672be512bf43b61253f2b2a43cd967cb3062` |
| Provenance-Preserving Data Acquisition producer | `a29845e13e55de30b24ae752b896058041d653e6` |
| Scientific Computation Runtime producer | `5f0409743e0098a0691a88302a9b3dcdcbcf25fd` |

```sh
CIW_SET_REPO=../State-Estimation-Evaluation-Testbed CIW_ACQUISITION_REPO=../Provenance-Preserving-Data-Acquisition CIW_RUNTIME_REPO=../Scientific-Computation-Runtime PYTHONPATH=src python -m pytest -q tests/test_exchange.py tests/test_exchange_integration.py
PYTHONPATH=src python -m pytest -q
```

The optional tests skip when checkouts are not explicitly provided. The checker
pin is enforced in both CLI and integration tests. Producer modules in the
roundtrip are resolved against the explicitly supplied checkout paths and may
not be substituted by an unrelated installed module.

The `exchange` provider gate ([`scripts/check_exchange.py`](../scripts/check_exchange.py),
declared in [`ci/gates.json`](../ci/gates.json)) checks out the validator at the
instrument-exchange descriptor pin and the two producers at their declared extra
pins, and supplies all three paths, so its conformance tests do not depend on
optional local bindings.

There is no automated conversion to native CIW covariance: exchange records do
not provide all the native artifact's provenance, reference-value and assumption
requirements. The inspector path has no shared session/viewport, save/reopen
operation, remote execution, corpus admission, sensor-fusion solver, physical
calibration check, cryptographic proof verification, or independent-verifier
authentication. The typed adapter adds a separate native session boundary
without changing those claims. It retains the exact source envelope bytes and
producer artifact IDs, then assigns distinct CIW operation, execution, result
and numerical-result identities. A native result links the original typed
artifacts and the inspector's conformance report; it does not reinterpret an
external producer ID as a CIW execution or result.

## Typed adapter and replay

The adapter input is a bounded `ciw.instrument-exchange-source.v1` object:

```json
{
  "schema": "ciw.instrument-exchange-source.v1",
  "producer": {"name": "...", "revision": "...", "operation_ids": ["..."]},
  "artifacts": ["one or more notation.instrument.*.v1 records"]
}
```

Bind the exact SET checkout before executing the operation. The binding checks
the revision, checked source path and normalized source digest from
[`exchange-runtime.json`](../src/ciw/exchange-runtime.json). The operation is
available through the shared `Session` API after that trusted host binding:

```sh
python -m ciw serve --exchange-set-repo ../State-Estimation-Evaluation-Testbed
```

`source.add` retains the exact envelope; an `operation.execute` request with
`ciw.instrument-exchange.v1` creates the native result. `workspace.save` and
`Session.from_workspace` validate the retained bytes and native identity graph
without executing a provider. A later `bundle.replay` requires the trusted SET
binding and reruns the checker. Fresh execution and result identities must
differ from the original while the numerical projection identity remains
equal. The replay receipt is explicit about `admission: not_performed`.

The adapter has its own native view (`ciw.instrument-exchange-view.v1`) and is
catalogued beside the existing operations. It has no shortcut into the legacy
`run.v1` measurement record and cannot authorize an action.
