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

The saved JSON contains `workspace_version: 1`, the complete scientific `run`, `selection`, immutable `results`, and reserved `view_settings: {}`. Opening it validates and restores these records; it does not invoke numerical operations. A new live session identity is assigned while the existing run, evidence, result and execution identities stay intact. New analyses explicitly create new execution/result IDs. `recording_file` on results points to the separately persisted source copy; embedded workspace evidence permits portable reopening. This is a local prototype format, not yet a PayloadOS persistence adapter.

The demo evidence ID is SHA256 over canonical JSON of instrument, metadata, timestamps and channels. The complete recording, including render data, has a separate content-derived filename. Replay checks both bindings and validates result semantics before writing. These hashes detect inconsistent content; they do not establish source authenticity or scientific verification.

Native local connections have no Origin header. The production launcher binds only `127.0.0.1` and rejects browser-origin connections. Remote execution/authentication are outside this prototype. Cursor requests are bounded by the first and last retained sample timestamp; interval end may equal recording duration. Integer-valued revisions are compared numerically; Python envelopes reject boolean revisions and protocol versions.
