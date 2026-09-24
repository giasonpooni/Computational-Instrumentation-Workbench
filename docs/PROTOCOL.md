# Workbench protocol v1

## Shared experiment projection

`experiment.inspect` accepts exactly `{"bundle_id":"<retained identity>"}` and
returns `ciw.experiment-view.v1`. It projects one committed native occurrence into
ordered scientific panels and its recorded input graph, retaining full covariance,
units, time/frame context, source/evidence/result/execution identities and original
verification. It does not execute a provider or change selection/retention.
See [the complete view contract](EXPERIMENT_VIEW.md). Existing `workbench.changed`
notifications invalidate the catalog; clients fetch authoritative state and keep
replay occurrences separate from measurement samples.

## Transport

Status: first implementation contract. Python is authoritative; terminal and Godot are independent clients. Local endpoint: `ws://127.0.0.1:8765`. Text JSON only in v1; no terminal scraping. The served endpoint accepts text frames up to 8 MiB: a larger frame closes only that connection with close code 1009, and a non-JSON frame within the limit is answered with an `invalid_request` error. Error codes: `invalid_request` (malformed envelope), `unsupported_version`, `invalid_payload` (a refused payload, unknown identity or unknown type), `revision_conflict`, `read_only_view`, `storage_error`, an operation's own refusal code, and `internal_error` when a request fails inside the service; the traceback stays in the service log and the connection stays open.

Request: `{"protocol_version":1,"request_id":"unique-client-id","type":"session.get","payload":{}}`.
Success: `{"protocol_version":1,"request_id":"...","type":"response","payload":{...}}`.
Failure: same envelope with `type: "error"` and `payload: {"code":"...","message":"..."}`.
Broadcast after selection mutation: `{"protocol_version":1,"request_id":null,"type":"selection.changed","payload":SELECTION}`. Broadcast after a completed analysis or generic operation: `{"protocol_version":1,"request_id":null,"type":"result.created","payload":RESULT_SUMMARY}`. Broadcast after a workbench source, bundle or replay is retained: `{"protocol_version":1,"request_id":null,"type":"workbench.changed","payload":{"session_id":...}}`.
The native endpoint sends a `session.snapshot` event on connection with the same payload as `session.get`. Responses and broadcasts can interleave; correlate by request_id. Clients may reconnect and request a fresh snapshot.

The additive `/spatial` endpoint sends `spatial.ready` with session identity and
read-only capabilities. It accepts only `spatial.list` (`{}`) and
`spatial.inspect` (`{source_id}`), returning geographic source descriptors and
a `ciw.spatial-view.v1` packet respectively. Its only broadcast is
`workbench.changed`. Browser origins require exact host configuration via
`--spatial-view-origin` and are rejected on other paths. No-Origin access to
`/spatial` remains read-only. See [module contracts](INTEGRATED_MODULES.md).

## Commands

| Type | Payload | Response payload |
| --- | --- | --- |
| `session.get` | `{}` | `{session_id, run: RUN_METADATA, selection: SELECTION, results: [RESULT_SUMMARY]}` |
| `run.get` | `{}` | Full RUN, with backend sample values and render geometry |
| `selection.update` | `{expected_revision: integer, cursor_s?: number, interval_s?: [start,end], channel?: string}` | SELECTION; broadcast to every connected client |
| `sample.get` | `{time_s: number}` | `{run_id, evidence_id, sample_index, time_s, values: {q,v,energy}, units: {q,v,energy}}` nearest retained sample; earliest on tie |
| `analysis.stats` | `{channel?: string, interval_s?: [start,end]}` | RESULT |
| `analysis.spectrum` | `{channel?: string, interval_s?: [start,end]}` | RESULT |
| `result.get` | `{result_id: string}` | RESULT (only results from this running session) |
| `result.list` | `{}` | `{results: [RESULT_SUMMARY]}`; discover stored analyses without running them |
| `workspace.save` | `{}` | `{workspace_file: string}`; saves to the service output directory |
| `bundle.replayability` | `{}` | `{host: {python_version, numpy_version, kernel_probe, linear_algebra}, bundles: [{kind, bundle_id, operation_id, execution_ids, result_ids, numerical_result_ids, replay_here, runtime_roles, differences, ...}]}`; read-only. `replay_here` is `runtime_identity_matches`, `runtime_identity_differs` (with the differing identity fields, such as `algorithm.kernel_probe`) or `requires_provider_binding`; nothing executes or rebinds |

## Shared workbench commands

Every retained source, bundle, context, instrument view and candidate receipt
of the running session is readable through these requests. All are read-only
except `source.add`, `operation.execute` and `bundle.replay`, which retain new
records and broadcast `workbench.changed`. Identities are exact strings from
earlier responses; an unknown identity is an `invalid_payload` error.

| Type | Payload | Response payload |
| --- | --- | --- |
| `source.add` | `{kind, label, bytes_b64}` | SOURCE_DESCRIPTOR `{schema, kind, label, source_schema, evidence_id, byte_count, source_id}`; the exact bytes are retained once and an identical retention is refused by identity |
| `source.list` | `{}` | `{sources: [SOURCE_DESCRIPTOR]}` |
| `source.get` | `{source_id}` | SOURCE_DESCRIPTOR plus `bytes_b64`, the exact retained bytes |
| `operation.list` | `{}` | `{operations: [{operation_id, role, source_kind, available, requires_upstream_bundle, ...}]}` |
| `operation.execute` | `{operation_id, parameters: {source_id, upstream_bundle_id?, configuration?}}` | BUNDLE_SUMMARY of the retained native bundle, or the retained refusal |
| `bundle.list` | `{}` | `{bundles: [BUNDLE_SUMMARY]}` with `{bundle_id, kind, source_id, upstream_bundle_id, session_id, operation_id, result_ids, execution_ids, verification_id, retained_verification_outcome, validation, numerical_replay, state_admission}` |
| `bundle.get` | `{bundle_id}` | The complete native bundle as retained |
| `bundle.replay` | `{bundle_id}` | `{bundle: BUNDLE_SUMMARY, replay_receipt}` for the fresh occurrence |
| `bundle.replayability` | `{}` | see the table above |
| `experiment.inspect` | `{bundle_id}` | Read-only experiment projection: object context, panels, provenance |
| `fusion.list` | `{}` | `{contexts: [...]}`; compatible-state contexts of the estimator-backed kinds with their lineage, never of declared kinds |
| `instrument.list` | `{}` | `{instruments: [{bundle_id, source_id, instrument, operation_id, result_id, execution_id, view, state_admission}]}` |
| `instrument.inspect` | `{bundle_id, instrument}` | The retained native result of that instrument role inside the bundle |
| `candidate.list` | `{}` | `{candidates: [{candidate_id, bundle_id, operation_id, execution_id, state, eligibility, state_admission}]}`; historical receipts only |
| `candidate.get` | `{candidate_id}` | The complete candidate receipt as retained |
| `execution.list` | `{}` | Retained completed and refused execution records, including native occurrences |

Payload keys are exact: an extra or missing key is refused. Nothing here
admits state, binds a provider or changes a retained record.

The initial selection is `{run_id, channel:"q", interval_s:[0,duration_s], cursor_s:0, coordinate_frame:"oscillator-state", revision:0}`. Analysis intervals are half-open `[start,end)` and independent of the playback cursor. Updates must specify the observed expected_revision; stale updates fail with `revision_conflict` and clients refresh. Analysis responses capture the selection revision and exact interval that produced them; results never mutate. Cursor updates do not recalculate analyses. Empty or reversed intervals, out-of-range cursors, unknown channels, nonfinite numbers and unsupported versions are rejected.

## Recorded run and instrument API

`RUN` has `{run_id, evidence_id, instrument, metadata, time_s: [number], channels: {q: {unit:"m",values:[number]}, v:{unit:"m/s",values:[number]}, energy:{unit:"J",values:[number]}}, render: {...}}`.
`metadata` has `{duration_s, sample_rate_hz, sample_count, coordinate_frame:"oscillator-state", model: {...}, provenance: {...}}`.
`RUN_METADATA` is `{run_id,evidence_id,instrument,metadata,channels:{q:{unit:"m"},v:{unit:"m/s"},energy:{unit:"J"}}}` (no large arrays).

Render geometry is backend-prepared, explicitly non-measurement data:
`render: {coordinate_frame:"oscillator-state", axis_labels:["q (m)","energy (J)","v (m/s)"], trajectory:[[q,energy,v],...], surface:{vertices:[[q,energy,v],...],indices:[int,...]}, sample_indices:[int,...], transform:{origin:[0,0,0],scale:[1,1,1],note:"..."}}`.
Render point k resolves through sample_indices[k] to the retained scientific record. The small demo sends every sample. Render geometry must never be used to compute a measurement.

Python instrument module `ciw.instruments` provides `make_demo_run() -> dict`, `validate_run(run) -> None`, `run_metadata(run) -> dict`, `inspect_sample(run,time_s) -> dict`, `compute_statistics(run,channel,interval_s) -> dict`, `compute_spectrum(run,channel,interval_s) -> dict`. Computation functions return operation-specific `data`; session code wraps provenance and persistence. Validate recordings at load, and validate direct computation inputs too.

Stats data: `{sample_count, mean, minimum, maximum, rms, unit}`.
Spectrum data: `{sample_count, method:"periodogram", window:"hann", detrend:"constant", scaling:"density", frequency_hz:[...], psd:[...], unit, peak_frequency_hz, sample_rate_hz}`. A one-sided Hann-window PSD uses retained float64 samples; this v1 does not claim Welch, spectrogram or streaming acquisition support.

## Results and evidence

`RESULT` is `{result_id, evidence_id, operation_id, execution_id, verification_id:null, verification_status:"not_verified", run_id, selection_revision, channel, interval_s, created_at, data}`. Operation IDs are `statistics.v1` and `spectrum.periodogram.v1`. Verification stays explicit and unclaimed. Results are persisted by the server and headless CLI in a chosen output directory. Source recordings are separate from derived results.

`RESULT_SUMMARY` includes result/operation/execution IDs, channel, interval, creation time and verification status, without numerical arrays. Snapshots and `result.list` expose these so fresh clients can discover analyses restored from a workspace. A completed analysis or generic operation is announced to every connected client as a `result.created` event whose payload is the `RESULT_SUMMARY`; numerical arrays stay behind `result.get`. Retained workbench sources, bundles and replays are announced as `workbench.changed`, and selection commits as `selection.changed`. Events carry `request_id: null`; clients match responses by request id and may ignore events they do not use.

Binary transport, occupancy operations, live acquisition and spectrograms are not implemented. The additive adapter and covariance sections below specify the delivered external-provider JSON operations.

## Workspace replay

The saved JSON contains `workspace_version: 1`, the complete scientific `run`, `selection`, immutable `results`, and reserved `view_settings: {}`. Opening it validates and restores these records; it does not invoke numerical operations. A new live session identity is assigned while the existing run, evidence, result and execution identities stay intact. New analyses explicitly create new execution/result IDs. `recording_file` on results points to the separately persisted source copy; embedded workspace evidence permits portable reopening. This is a local prototype format, not yet a production persistence adapter.

The demo evidence ID is SHA256 over canonical JSON of instrument, metadata, timestamps and channels. The complete recording, including render data, has a separate content-derived filename. Replay checks both bindings and validates result semantics before writing. These hashes detect inconsistent content; they do not establish source authenticity or scientific verification.

Native local connections have no Origin header. The launcher binds `127.0.0.1` by default. Browser origins require explicit permission on the read-only spatial endpoint. The container explicitly binds `0.0.0.0` inside its network namespace, with Compose publishing only on the host's `127.0.0.1`. Remote execution/authentication are outside this prototype. Cursor requests are bounded by the first and last retained sample timestamp; interval end may equal recording duration. Integer-valued revisions are compared numerically; Python envelopes reject boolean revisions and protocol versions.

The service drains connections and saves its workspace on graceful shutdown. Windows process termination is immediate; the native controller sends `workspace.save` and verifies the response before stopping its owned process. Forced termination and power loss are not covered by an autosave guarantee.

## Additive adapter and operation slice

The legacy oscillator request/response contract above remains protocol version 1.
Generic scalar/event recordings use `run_schema: "run.v1"` and embed a strict
`ciw.instrument-manifest.v1` under `metadata.manifest`. An event record can have
one timestamp, no sample rate, arbitrary declared channels and no render data.
The manifest role distinguishes measurement adapters from instruments and
operation providers. The current Godot client explicitly declines this new
record/view contract; the terminal and structured session API expose it.

Three additive session requests are available:

| Request | Payload | Response |
| --- | --- | --- |
| `operation.list` | `{}` | Locally registered operation IDs and roles |
| `operation.execute` | `{"operation_id":"fsrt.tank-reconstruct.v1","parameters":{"model":{...}}}` | `status`, retained `execution`, and a `result` or `null` |
| `execution.list` | `{}` | Retained completed/refused execution records |

A refusal is a recorded invocation with no result, not an estimated state.
Calibration failures during acquisition/import raise `calibration_unavailable`
and do not create a calibrated run or result. Invalid protocol/configuration
requests retain the existing error envelope. Source bindings must be supplied
explicitly by the operator; opening a saved workspace cannot authorize code.

Operation results use `schema: "ciw.operation-result.v1"` and executions use
`schema: "ciw.execution.v1"`. They retain operation, run, source evidence,
selection, parameters and runtime bindings, distinct execution/result IDs, and
content-integrity digests. An execution's `result_id` is null when refused.
Verification remains `not_verified` with `verification_id: null`; no estimator
or successful replay can promote that status. Domain output schemas are
validated without executing their scientific providers during reopening.

A workspace with retained operation executions uses `workspace_version: 2` and
an `executions` array. Version 1 workspaces continue to open unchanged. Reopening
preserves identities and results; explicit replay invokes trusted local bindings
and appends fresh execution/result identities. See [ADAPTERS.md](ADAPTERS.md) for
the exact pinned single-snapshot scope, raw-byte retention, covariance mapping,
terminal commands and offline replay setup. This additive slice does not claim
streaming acquisition, binary transport, a multi-run journal or physical
validation.

## Additive covariance and calibration-serving contract

Protocol version remains 1 and workspace version remains 2. Operation IDs use
an explicit positive version suffix (`.v1`, `.v2`, ...); a syntactically valid
identity never binds code. Unregistered operations produce retained refusals.
The new trusted payload schemas are `fsrt.tank-reconstruct.v2` and
`jspt.covariance-propagate.v1`. Calibration acquisition uses `rci.calibrate.v2`.
Their exact domain versions are pinned in `adapter-runtimes.json`.

`covariance-artifact.v1` carries `covariance_id`, ordered `quantity_ids` and
`units`, `frame`, `reference_values`, full `matrix`, `method`, `basis`,
`provenance` and `assumptions`. Its content identity includes every field except
`covariance_id`; matrix entry (i,j) has units `units[i] * units[j]`. Numerical
values are never silently symmetrized, clipped, diagonalized or reordered by
CIW validation. A positive semidefinite singular matrix is a valid artifact;
a specific operation can still refuse it under its numerical requirements.
See [COVARIANCE.md](COVARIANCE.md) for the exact artifact and input contracts.

`session.get`, `result.list` and `result.get` accept optional `evaluated_at`,
a timezone-aware ISO timestamp including seconds. Omission means current UTC.
For an RCI-backed run, session snapshots and list responses carry `calibration`
next to their other payload fields. A `result.get` success carries `calibration`
as a sibling of `payload` in the response envelope: the payload remains the
exact immutable saved result. Oscillator responses add no calibration field.
A connection snapshot evaluates serving time at connection, and never changes
historical acquisition evidence.

Each calibration status contains source/profile identities and validity bounds,
`acquisition {observed_at, applicable_at_acquisition, valid_from, valid_until,
basis, provenance}`, and `serving {evaluated_at, expired, not_yet_valid,
current_applicability}`. Acquisition provenance is `persisted.v2` for the new
RCI record or explicitly `derived.legacy-v1` for older records. Serving metadata
is not persisted into, or hashed as, the scientific evidence/result. Present
expiry does not invalidate a historically applicable observation.

JSPT `operation.execute` parameters retain `source_result_id`, `source_artifact`,
the exact `source_covariance`, and the declared map fields. The CLI resolves
these from a retained FSRT/JSPT result; live clients may reuse those exact
parameters with an explicitly bound `--jspt-repo`. The service checks source
links before execution. Reopening checks retained dependencies and rejects
missing, mismatched or cyclic references before writing a destination.

A saved runtime is executable only if its revision/module is on the built-in
current-or-historical allowlist and the operator supplies that exact clean
checkout and matching interpreter/dependencies. A saved file cannot extend
this allowlist. An invocation with no bound runtime cannot be silently replayed
using the latest engine; it requires an explicit new execution.

## Additive geometric reconciliation contract

The additive GTE operation `gte.project-circle.v1` uses the same operation and
workspace envelopes. Bind an operator-supplied checkout with `serve --gte-repo`
or the `geodesic create/replay` commands. Its complete retained batch is the
scientific scope; the selected interval must contain all retained samples or
the operation refuses without a result (`selection_scope`). Only explicit
whole-object `constraint` and `policy` overrides are accepted as parameters;
observations, covariance, frame and acquisition time remain bound to evidence.
The existing `backend` role identifies this geometric result within the generic
enum and does not claim state-estimator semantics. Read-only source
and result validation is built into CIW, so reopening needs no GTE runtime.
See [GTE.md](GTE.md) for exact inputs, statuses, covariance meaning and limits.

## Shared measurement, geometry and stability operations

The additional shared kinds `measurement-chain`, `geometric-circle` and
`identified-stability` use the same `source.add`, `operation.execute`,
`bundle.replay` and `experiment.inspect` surfaces. Their operation IDs are
`ciw.measurement-chain.v1`, `ciw.geometric-circle.v1` and
`ciw.identified-stability.v1`. All take an exact retained `source_id`;
identified stability additionally selects an `upstream_bundle_id` whose kind
is `identified-design`. Its source binds the selected native model and state
result/execution identities. The measurement chain's unchanged FSRT and JSPT
results and execution IDs are also exposed through `result.*`, `execution.list`
and instrument inspection, alongside the outer RCI operation. Replays create
fresh occurrences and preserve the original evidence. These objects have typed
object context and do not create or implicitly merge fusion contexts. See
[shared module operations](REMAINING_MODULES.md) for input contracts and bindings.

## Shared acquired-window and residual-monitor operations

These operations use the shared workbench catalog in workspace format 3;
protocol version remains 1. Bind the eight exact provider checkouts with
`serve --acquired-stream-stack-root /path/to/providers`. The
[acquired stream guide](ACQUIRED_STREAM.md) contains the complete declarations,
pins, runnable client and refusal conditions.

`source.add` accepts kinds `acquired-calibrated-window` and `residual-monitor`
with their versioned source schemas. `operation.execute` uses
`ciw.acquired-calibrated-window.v1` with parameters
`{source_id, upstream_bundle_id}`. The upstream must be a retained
`acquired-dataset` occurrence. Every selected PPDA observation, record, document
and first-acquisition snapshot row is bound explicitly. Native calibrated-window
steps retain their child bundle identity and SET receipt; the outer mapping has
its own verification scope.

`ciw.residual-monitor.v1` takes `{source_id}`. Its source declares an ordered
`window_bundle_ids` list of retained calibrated windows, including acquired
windows. The monitor consumes their exact GSIE innovations and covariance.
It retains native OIT/FDIR results, unknown cross-window covariance and separate
reference priors. Replayed windows cannot count as additional observations.
OIT-held rows retain detection results but do not advance CUSUM. Threshold
crossings cannot grant statistical alarm authority or unique fault isolation.

Both paths use `bundle.get`, `bundle.replay`, `experiment.inspect` and
`workbench.changed` in the existing live session. Replay creates fresh execution
and result occurrences while preserving the selected upstream evidence.
Inspection and reopening neither execute providers nor admit canonical state.
