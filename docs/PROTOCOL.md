# Workbench protocol v1

Status: first implementation contract. Python is authoritative; terminal and Godot are independent clients. Local endpoint: `ws://127.0.0.1:8765`. Text JSON only in v1; no terminal scraping.

Request: `{"protocol_version":1,"request_id":"unique-client-id","type":"session.get","payload":{}}`.
Success: `{"protocol_version":1,"request_id":"...","type":"response","payload":{...}}`.
Failure: same envelope with `type: "error"` and `payload: {"code":"...","message":"..."}`.
Broadcast after selection mutation: `{"protocol_version":1,"request_id":null,"type":"selection.changed","payload":SELECTION}`.
The service sends a `session.snapshot` event on connection with the same payload as `session.get`. Responses and broadcasts can interleave; correlate by request_id. Clients may reconnect and request a fresh snapshot.

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

`RESULT_SUMMARY` includes result/operation/execution IDs, channel, interval, creation time and verification status, without numerical arrays. Snapshots and `result.list` expose these so fresh clients can discover analyses restored from a workspace. New analyses do not yet produce an event; refresh the list to discover results created by another client.

Future binary arrays must declare meaning, shape, dtype, byte order, order, units, frame, time reference, and execution/result identity. Binary transport, existing-instrument adapters, uncertainty propagation, occupancy operations, acquisition and spectrograms are later work, not implemented capabilities.

## Workspace replay

The saved JSON contains `workspace_version: 1`, the complete scientific `run`, `selection`, immutable `results`, and reserved `view_settings: {}`. Opening it validates and restores these records; it does not invoke numerical operations. A new live session identity is assigned while the existing run, evidence, result and execution identities stay intact. New analyses explicitly create new execution/result IDs. `recording_file` on results points to the separately persisted source copy; embedded workspace evidence permits portable reopening. This is a local prototype format, not yet a production persistence adapter.

The demo evidence ID is SHA256 over canonical JSON of instrument, metadata, timestamps and channels. The complete recording, including render data, has a separate content-derived filename. Replay checks both bindings and validates result semantics before writing. These hashes detect inconsistent content; they do not establish source authenticity or scientific verification.

Native local connections have no Origin header. The launcher binds `127.0.0.1` by default and rejects browser-origin connections. The container explicitly binds `0.0.0.0` inside its network namespace, with Compose publishing only on the host's `127.0.0.1`. Remote execution/authentication are outside this prototype. Cursor requests are bounded by the first and last retained sample timestamp; interval end may equal recording duration. Integer-valued revisions are compared numerically; Python envelopes reject boolean revisions and protocol versions.

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
the architecture's complete streaming, binary transport, multi-run journal or
physical-validation milestones.
