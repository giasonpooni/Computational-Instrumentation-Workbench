# Computational Instrumentation Workbench — Architecture

Short form: **CIW**. Protocol baseline: version 1, as implemented in `src/ciw/` and specified in [`docs/PROTOCOL.md`](PROTOCOL.md). Normative statements carry identifiers `CIW-<AREA>-<NNN>` with MUST / SHOULD / MAY wording; Section 16.4 maps every identifier to its verification, milestone, and prototype status.

## 1. Summary

The Computational Instrumentation Workbench is:

> A terminal-first workbench that connects computational instruments to synchronized numerical, temporal, spectral, and 2D/3D representations of physical-system observations and estimated states.

**What it is.** A small runtime, the *session service*, that attaches computational instruments (measurement sources, state estimators, field reconstructions, model integrators, signal-processing stages) through one contract, holds the scientific record and the shared selection for one investigation, executes operations on that record, and serves records and results to clients. Two frontends exist: a terminal client, which is the control surface and works alone, headless, and over SSH; and a Godot 2D/3D viewport, which attaches to the same service over the same protocol and owns no calculation; any script or CI job that speaks the session protocol is an equally ordinary client. The recurring operation loop is the same for every instrument: attach a source, select quantities and a domain, run an operation, inspect the result, compare, save or replay.

**What it guarantees.**

- The service is authoritative. Clients render what the service holds and compute nothing that produces a record; render geometry is never used to compute a measurement.
- Every result carries an envelope: what it is, which quantities and units it contains, what its timestamps and coordinates mean, how it was produced, and separate evidence, operation, execution, and verification identities. Results are immutable; recomputing creates a new execution and result.
- All views share one revisioned selection: run, channel, half-open analysis interval, playback cursor, coordinate frame. Changes are sequenced by the service and broadcast to every client in one order. The cursor is separate from the interval; moving it never recomputes an analysis.
- From M1, observations and estimated states are distinct kinds with one styling convention in every view; from M3, estimates carry their declared uncertainty. v1 holds a single synthetic recording without kinds or uncertainty and states this as a limit.
- Under the streaming extensions, ingest never loses data silently: the bulk path is credit-based end to end and every drop is recorded as a gap.

**How it is structured.** Five components with fixed boundaries: the **Session Service** (record, selection, operations, results, persistence, gateway), **Instruments** (in-process today; subprocess, remote, and observed bindings as extensions), the **Terminal Client** (`ciw` CLI today; Textual panels as an extension), the **Viewport** (Godot 4), and the **Workspace** (a JSON file today; a session directory as an extension). A headless instrument path (Section 8.13) drives an external instrument from the terminal client without the service and writes self-contained run bundles that carry the same identities. Two deployment paths (Section 5.8), a native controller and a container backend, run the same service and restart it from the workspace saved at shutdown with run, evidence, execution, and result identities intact. Section 6 states the v1 baseline as built and presents everything beyond it as versioned extensions with identifiers and target milestones.

**Reference implementation.** Service, instruments, and terminal client in Python 3.11+ with NumPy (SciPy where needed); Textual panels as an extension of the scriptable CLI; Godot 4 as the viewport, a client over a loopback WebSocket carrying text JSON in v1 and binary buffers with an explicit descriptor as an extension; in-terminal braille and half-block rasters as an extension (M1), always rendered and sufficient when no viewport is attached, with Kitty, iTerm2, and Sixel graphics where detected (M4). A Rust host with ratatui and Arrow is the documented alternative for a later native path; every contract is language-neutral.

**Roadmap.** The prototype (v0.1) is milestone M0: the session service, both clients, the workspace, the two deployment paths with saved shutdown, resume, and a health probe, and the headless adapter for the Parameterized Lyapunov Stability Runtime (PLSR), which delivers the adapter part of M1 but not the manifest, the subprocess binding, session attachment, or capability-driven views (Section 13.8 lists the items). M1 completes the existing-instrument adapter and places the scalar/time-series reference instrument (a) behind an adapter with its headless calculation and tolerances preserved; M2 adds a persistence adapter that reuses an external evidence-and-result infrastructure's identities without merging them; M3 the state-estimation reference instrument with uncertainty and revisions; M4 the spatial reference instrument with binary frames and viewport resources; M5 spectrograms, streaming with separate acquisition, publish, and render rates, cancellation, and stale-result handling; M6 a deployment for a stated use with measured workloads. Readiness stages standalone-ready, workbench-ready, and deployment-ready are defined in Section 18.

## 2. Purpose, scope, and non-goals

### 2.1 Definition

> A terminal-first workbench that connects computational instruments to synchronized numerical, temporal, spectral, and 2D/3D representations of physical-system observations and estimated states.

Each word of the name has a role:

- **Computational** covers the estimation, reconstruction, integration, and signal-processing backends, not just directly measured values.
- **Instrumentation** keeps the focus on investigating physical systems through measurements, models, and diagnostics.
- **Workbench** describes an environment in which several instruments and analytical operations can be used together. It does not imply that the visualization layer replaces those instruments.

Within the definition: *terminal-first* means the primary, fully capable interface is a terminal that works headless and over SSH; richer 2D/3D output may use terminal graphics protocols or an auxiliary viewer, but the terminal remains the control surface. *Synchronized* means the four representation families share time base, cursor and selection, units, and the same underlying data, and a change in one is reflected in the others deterministically. *Observations* (measured data, kind `observation`; quantities computed from other channels carry kind `derived` with `derived_from`, Section 7.2) and *estimated states* (outputs of estimators, reconstructions, integrators) are distinct kinds of data, distinguishable in the data model and in every view, including uncertainty. The workbench *connects* instruments; it does not absorb them. Instruments are replaceable backends behind a contract. The engineering title, *Integrated Measurement, State Estimation, and Visualization Workbench*, appears only in the README subtitle and in technical abstracts (ADR-0001).

### 2.2 Position

CIW is a foundational instrumentation runtime and workbench: a common layer underneath specialized tooling. It stays small because it captures recurring operations, not because it omits their meaning; it needs no universal physical model, only a consistent way to connect, execute, inspect, compare, and record specialized models and measurements. **The workbench supplies the operating environment; the instrument supplies the scientific meaning.** Views are capability-driven, never compulsory: a scalar instrument does not need a mesh, a recorded-data instrument does not need a live device connection, a spatial instrument does not get a spectral panel unless the analysis is meaningful. Generality lives in the contract, not in one format: the result envelope and the execution contract are standardized; payload types and engines are not. Be strict about meaning at the boundary; be flexible about implementation behind it.

Frontends are clients of the computation, as several Jupyter frontends attach to one kernel and receive different representations of its outputs; neither the terminal client nor the viewport owns a separate calculation. Virtual instrumentation as National Instruments describes it separates acquisition, drivers, software-defined processing, and presentation and builds larger systems from smaller instrument functions; this design belongs to that family. The SciJava Ops work shows that a plugin mechanism is not data-structure interoperability and that adaptation layers are still needed; that is why meaning is standardized at the boundary. ROS 2 real-time design documentation distinguishes meeting deadlines from being responsive; that is why service level is separate from representability (Section 5.5).

### 2.3 Scope and protocol ownership

This document specifies the conceptual model, components and process model, the protocol v1 baseline and extension policy, the data model and envelope, the instrument contract at the level of semantics, the synchronization model, the representation layer, sessions and reproducibility, operator interaction, the extension model, the reference stack, the repository layout, budgets and conformance, risks, and the roadmap. It owns protocol semantics: message families, required fields, ordering, backpressure, cancellation, errors, versioning policy, invariants. The concrete wire specification (exact encodings and schemas, handshake sequences, version negotiation) is [`docs/PROTOCOL.md`](PROTOCOL.md), which must satisfy every requirement here; wire details are not repeated.

### 2.4 Non-goals

- A hard real-time runtime. The terminal/viewport link is not suitable for deadline-critical control loops; time-critical execution stays in an engine or controller that the workbench configures, observes, and inspects through a declared interface.
- A universal physical model, state vector, or data format.
- A database, an operating system, or a universal solver. Where a deployment has an evidence-and-result infrastructure, the workbench is an instrumentation workspace over it through an adapter (Section 13.4).
- A signal-processing or estimation library; those are instruments.
- A browser frontend as the primary interface; one may be added later as another client.

### 2.5 Baseline and extensions

Every requirement is marked **v1** (implemented in the prototype) or **extension** (versioned addition with a target milestone). No extension may contradict a v1 invariant: authoritative service, clients own no calculation, cursor separate from interval, revisioned selection, immutable results, distinct identities, render geometry never measured. Limits of v1 are stated as limits, never as capabilities.

## 3. Glossary
| Term | Meaning |
|---|---|
| **Session** | One live service instance holding one investigation; `session_id` is assigned at start and again on reopen. |
| **Run** | One recording `{run_id, evidence_id, instrument, metadata, time_s[], channels{}, render{}}`, never an execution; one per session in v1. |
| **Evidence** | The content a result was computed from; `evidence_id` is a content hash of the run's scientific record. |
| **Operation** | A named, versioned computation (`statistics.v1`, `spectrum.periodogram.v1`); `operation_id`. |
| **Execution** | One invocation of an operation on evidence; `execution_id`. |
| **Result** | The immutable record an execution produced: envelope plus operation-specific `data`; `result_id`. |
| **Verification** | An explicit check of a result against declared criteria, never inferred from a displayed scene; `verification_id` is `null` with `verification_status: "not_verified"` until one exists. |
| **Selection** | The shared state `{run_id, channel, interval_s, cursor_s, coordinate_frame, revision}`. |
| **Revision** | Integer incremented on every accepted selection update; updates carry `expected_revision`. |
| **Cursor** | The playback position `cursor_s`, separate from the analysis interval. |
| **Interval** | The half-open analysis interval `interval_s: [start, end)` that operations consume. |
| **Snapshot** | The `session.snapshot` event and `session.get` response (session id, `RUN_METADATA`, selection, result summaries; from M1 `view_settings` and `protocol_minor`); in the extension also the immutable `(revision, W, t_tick, seen)` of one render tick (CIW-SYNC-010). |
| **Render geometry** | Backend-prepared, non-measurement display data (`run.render`) resolving to retained samples through `sample_indices` with a declared `transform`. |
| **Workspace** | The persisted session `{workspace_version, saved_at, run, selection, results[], view_settings{}}`. |
| **Instrument** | A computational backend behind the contract (source, estimator, reconstruction, integrator, signal-processing stage); declared by a manifest (extension). |
| **Observation** | Measured data, or data a source declares as its measurement; kind `observation`. |
| **Estimated state** | Output of an estimator, reconstruction, or integrator; kind `estimate`, with a declared uncertainty form. |
| **Derived** | A quantity computed from other channels; kind `derived`, always with `derived_from`. |
| **Channel** | A named, unit-bearing sample sequence on one time base; a bare `channel_id` is unique within the session (CIW-SYNC-022). |
| **Time base** | The clock timestamps are expressed in: `time_s` seconds since run start (v1); named bases with clock, epoch, and mapping (extension). |
| **Frame** (data) | One batch of samples with a descriptor, the unit of streaming transfer; distinct from a *coordinate frame*, always written in full except as the `coordframe` verb (12.2). |
| **Coordinate frame** | A named spatial reference with parent, transform, handedness, and axis units. |
| **Representation / view** | A rendering in one of four families: numerical, temporal, spectral, 2D/3D; a *pane* hosts one. |
| **View settings** | Per-workspace presentation state (`view_settings`, empty in v1; fields in 9.6). |
| **Client** | A process attached over the session protocol: terminal client, viewport, or script. |
| **Credit** | Frames and bytes a consumer has authorized a producer to send (extension). |
| **Journal** | The append-only, sequence-numbered log of mutations, commands, and events (extension). |
| **Run bundle** | A self-contained immutable JSON file written by the headless path (8.13): inputs, output record, runtime identity, identities, digests. |
| **Short id** | The last label of an instrument's reverse-DNS `id`, unique among attached instruments; used in `stream` names (`<short id>/<output>`). |
| **PLSR** | Parameterized Lyapunov Stability Runtime, the first external instrument, integrated through the headless path (13.8). |

## 4. Conceptual model
The operation loop is the primary command and session model; each step maps to components and contracts:

| Loop step | v1 realization | Component | Contract |
|---|---|---|---|
| attach a source | `ciw serve --recording\|--workspace\|--resume` (v1: oscillator schema only, CIW-PERF-001); `ciw plsr evaluate` (8.13) | Session Service, Instrument | Instrument API (8.1), manifest (8.2), headless path (8.13) |
| select quantities and a domain | `selection.update {channel, interval_s, cursor_s}` | Session Service | Selection (Section 9) |
| run an operation | `analysis.stats`, `analysis.spectrum` | Session Service, Instrument | Results (7.8), contract (Section 8) |
| inspect the result | `sample.get`, `result.get`, `ciw inspect`, viewport cards | Terminal Client, Viewport | Section 10 |
| compare | `result.list`; comparison views (extension) | Terminal Client | Sections 10 and 12 |
| save or replay | `workspace.save`; `ciw serve --workspace`; `ciw inspect` | Workspace | Section 11 |

Instruments produce runs and results, the service holds them, clients render them. An execution has fixed parameters and inputs and one `execution_id`; a view is a subscription to records at a requested resolution plus the shared selection, changed only through an update carrying the observed revision; a workspace is run plus selection plus results, reopened without executing anything. The service knows runs, channels, units, time bases, kinds, identities, the selection, and the loop; the instrument knows the equations and declares which representations apply.

## 5. Component architecture

### 5.1 Components
```mermaid
flowchart LR
    subgraph Instruments
        I1[In-process instrument<br/>Python module, v1]
        I2[Subprocess instrument<br/>extension]
        I3[Remote or observed instrument<br/>extension]
    end
    subgraph Service["Session Service (ciw serve)"]
        REC[Record holder]
        SEL[Selection sequencer]
        OPS[Operation runner]
        RES[Result ledger]
        GW[Gateway<br/>loopback WebSocket, JSON]
        PER[Persistence]
        AN[In-process host<br/>ciw analyze, no socket]
    end
    subgraph Clients
        CLI[Terminal client<br/>ciw send / watch / health]
        TUI[Terminal panels<br/>Textual, extension]
        VP[Viewport<br/>Godot 4]
        SC[Script / CI client]
    end
    WS[("Workspace<br/>workspace.json, recording-#lt;hash#gt;.json, result-#lt;id#gt;.json")]
    subgraph Headless["Headless instrument path (ciw plsr, v1)"]
        HA[Headless adapter<br/>import / evaluate / inspect / replay]
        ENG[External engine<br/>pinned upstream package]
    end
    RB[("Run bundle<br/>run-#lt;id#gt;.json")]
    DC[Deployment controller<br/>workbench.ps1 or Compose]
    I1 --> REC
    I2 -. control + bulk .-> REC
    I3 -. TLS / ssh .-> REC
    REC --> OPS --> RES
    SEL --> GW
    RES --> GW
    REC --> GW
    RES --> PER --> WS
    GW <-- ws://127.0.0.1:8765 --> CLI
    GW <-- ws://127.0.0.1:8765 --> TUI
    GW <-- ws://127.0.0.1:8765 --> VP
    GW <-- ws://127.0.0.1:8765 --> SC
    AN -. embeds .-> REC
    HA --> ENG
    HA --> RB
    DC -. start, health, save, signal .-> GW
```

| Component | Responsibility | Never does |
|---|---|---|
| **Record holder** | Holds the validated float64 record and render geometry; `run.get`, `sample.get`. | Modify a retained sample. |
| **Selection sequencer** | Applies `selection.update` under one lock, increments `revision`, rejects stale updates, broadcasts `selection.changed`. | Recompute anything. |
| **Operation runner** | Executes operations on retained samples over an explicit interval. | Read render geometry or reductions. |
| **Result ledger** | Wraps `data` in the envelope, assigns `execution_id` and `result_id`; `result.get`, `result.list`. | Mutate or delete a result. |
| **Gateway** | Terminates connections, rejects non-text and non-JSON frames (`invalid_request`), hands decoded requests to the session (envelope validation, `request_id` echo), sends `session.snapshot` on connect, fans out broadcasts. | Mutate the selection outside the sequencer. |
| **Persistence** | Writes recording, one file per result, and workspace atomically; reopens after full validation. | Run an operation while reopening. |
| **Instrument Supervisor** (extension) | Launches, monitors, detaches subprocess and remote instruments; grants credits; validates frames and envelopes at the boundary. | Interpret payloads beyond the envelope. |
| **Reducer** (extension) | `derived` reductions at view resolution, cached by key; fits spectra and spectrograms to a pane's width (CIW-SYNC-015). | Produce anything without provenance. |
| **Terminal Client** | Control surface: headless analysis, `send`, `watch`, `inspect`; panels, command line, keybindings, layouts, rasters (extension). | Compute a record as a client (`analyze` embeds the service, 5.2). |
| **Viewport** | 2D/3D renderer: phase portrait, backend-supplied energy surface and trajectory, shared cursor, cards from `sample.get`. | Reconstruct values from geometry; update the selection without the observed revision. |
| **Workspace** | `workspace.json`, `recording-<hash>.json`, `result-<id>.json`; a session directory in the extension. | Anything active. |
| **Headless adapter** (v1) | Drives an external instrument without the service, calling the upstream engine under a verified source pin and writing immutable run bundles with the identity model (8.13). | Attach to a session or publish to clients (extension, CIW-SESS-014); compute a verdict itself. |

Naming: the **Session Service** is the *Workbench runtime* of `README.md`; the **Terminal Client** is the *terminal frontend* of `README.md` and ADR-0003; the **Viewport** is the 2D/3D viewport in all three.

### 5.2 Process model
| Process | Count | Survives |
|---|---|---|
| Session service (`ciw serve`) | One per session | Client disconnects; viewport close; instrument failures (extension) |
| Terminal client (`send`, `watch`, `analyze`, panels) | Zero or more | Service restart: `send` and `watch` exit with the codes of CIW-OPS-001 and are re-run to reattach; panels reconnect and request a snapshot (M1) |
| Viewport (Godot) | Zero or one by default | Service restart (shows STALE, offers Reconnect) |
| Subprocess instrument (extension) | One per instrument | Nothing beyond its lifetime; the service survives it |
| Remote or observed instrument (extension) | Zero or more, elsewhere | Service restarts (the service reattaches) |
| Headless instrument run (`ciw plsr evaluate\|replay`) | Zero or more, short-lived | Nothing; its bundle survives it |
| Deployment controller (`workbench.ps1`) or Compose | Zero or one per deployment | Service exit and restart (native `Status` re-reads `service.json`; Compose restarts under `on-failure:3`); holds no session state (5.8) |

The service is a separate process: a client crash cannot delay sequencing, a stalled client is bounded (v1: 5.3; from M1 off the sequencing path, CIW-SYNC-021), and both clients are served symmetrically. `ciw analyze` embeds the service in the terminal process without a socket, so its record and result come from service code. In-process instruments share the service's failure domain and are restricted to bounded calls; the subprocess binding is the default for anything else.

### 5.3 Threading inside the service
The service is Python; hot paths (socket I/O, NumPy kernels, memory copies) release the GIL. Only the main asyncio thread mutates the selection, under one `asyncio.Lock`, so broadcast order equals revision order across clients; operations, result file writes (fsync and rename), and workspace saves run in worker threads, and a result is registered under the session lock after its file write succeeds. In v1 the sequencer awaits delivery of `selection.changed` to every client under the lock, abandoning a send after the 2 s send deadline and then awaiting the close handshake, still under the lock, which with a full client receive window completes only when the WebSocket keepalive next runs (20 s interval) and its send expires the 2 s close deadline: one stalled client delays the next accepted update by up to 2 s + 20 s + 2 s, or indefinitely if a keepalive ping was itself blocked behind the paused transport, and CIW-PERF-002 does not hold for the other clients meanwhile. CIW-SYNC-021 removes this at M1. Extension threads (ingest, reducer pool, recorder, supervisor timers) post to the main loop and never mutate the selection.

### 5.4 IPC summary
Client ↔ service: the loopback WebSocket of 6.1, text JSON, no authentication, browser origins rejected, 1 MiB maximum incoming message, container bind per CIW-PERF-012; extensions add binary frames with a descriptor on the same connection, Unix socket and TCP with token or TLS, and SSH forwarding (14.1). Controller ↔ service: 5.8; no shutdown request in the client protocol (CIW-SESS-012). Service ↔ instrument: Python calls in v1 (8.1), budgeted by `max_call_ms` from M1 (CIW-INST-018); subprocess and remote links per 8.11. Service → terminal: JSON and event lines in v1; cells, braille, half-block, and Kitty, iTerm2, Sixel payloads as extensions (10.3).

### 5.5 Service classes
Whether the workbench can represent an instrument and whether a deployment meets its timing, throughput, precision, and reliability needs are separate questions; the deployment declares the class, and the service enforces placement.

| Class | Where the instrument runs | What the service does | Suitable for | Not suitable for |
|---|---|---|---|---|
| **Inline** | In-process binding (v1; budget and cap from M1) | Synchronous calls, from M1 budgeted by `max_call_ms` under CIW-INST-018 | Recorded runs, small deterministic transforms, adapters | Anything that can block, allocate unboundedly, or crash |
| **Local** (default from M1) | Subprocess, same machine | Full contract over stdio control and shared-memory bulk | NumPy engines, simulations, file readers, replay | Deadline-critical loops |
| **Remote** | Another machine | Full contract over `ssh://` or TLS; buffers inline or by reference | Large simulations, lab machines, special hardware | Deadline-critical loops |
| **Observed** | A dedicated engine or controller with its own timing | The adapter exposes configuration, state, and reduced telemetry; the loop never crosses the session protocol | Real-time controllers, hardware-timed acquisition | Being driven at the loop rate |

Not every workload runs in the same process, on the same machine, or at the same rate. A large simulation may provide reduced visualization data while its full results are retained elsewhere and referenced (`reduced_of`, 7.8).

### 5.6 Failure isolation
| Failure | Effect | Recovery |
|---|---|---|
| Client disconnects | Session unchanged | Reconnect; `session.snapshot` |
| Service stops (SIGINT/SIGTERM on POSIX; Windows: controller Stop) | Viewport shows STALE, disables shared interaction, offers Reconnect; `ciw send` exits 2, `ciw watch` 0; an operation in flight completes and its result is saved before exit (CIW-SESS-012) | `ciw serve --resume` (or `--workspace`) restores the saved workspace; `ciw health` confirms the session |
| Killed after the grace period, or power loss | Changes since the last completed save are lost | `workspace.save` before forced termination; `--resume` reopens the last save |
| Invalid request (v1 codes of CIW-INST-002) | `type: "error"` with `{code, message}`; session unaffected | Client corrects and retries |
| 1,024 results reached | `capacity_exceeded`; no result created | Save; start a new session |
| Storage error | `storage_error`; result not registered | Free space; retry |
| Instrument exits or misses 3 heartbeats (extension) | Instrument `Failed`; channels frozen and marked stale; open executions closed `status: failed`, `truncated_at {frame_seq, reason: instrument_failed \| heartbeat_loss}` | Manual restart by default; bounded `auto` (8.8) |
| Recorder cannot keep up (extension) | Ingest credits stop; the source's overflow policy applies; nothing accepted is lost | Free space or lower the rate |

### 5.7 Observability (extension, M5)
The service exposes its own operation as channels under the reserved instrument id `host` (ingest rate, credit balance, ring occupancy, reducer queue depth and latency, gateway backlog per client, journal sequence and commit lag), queryable headless with `ciw stats --json`; every request carries a `request_id` and every journaled event a sequence number.

### 5.8 Deployment paths and shutdown (v1)
Two deployment paths run the same service and protocol. Native controller (`scripts/workbench.ps1`: Setup, Start, Status, Stop): the service as a hidden process on `ws://127.0.0.1:8765` with a `.ciw/` data directory (recordings, results, `workspace.json`, logs, `service.json` ownership metadata); Stop saves and verifies the workspace over the client link and stops only the owned process; Start restores a saved workspace and errors on corrupt data rather than starting a new investigation. Container backend (Compose service `backend`): `ciw serve --bind 0.0.0.0 --resume --output-dir /data`; the host publishes only `127.0.0.1:${CIW_PORT:-8765}`; named volume `workspace` at `/data`; read-only root filesystem, tmpfs `/tmp`, all capabilities dropped, no-new-privileges, non-root uid 10001; `stop_grace_period` 30 s; `restart: on-failure:3`; Docker healthcheck via `ciw health`. See [`deploy/README.md`](../deploy/README.md) and [`deploy/CONTAINER.md`](../deploy/CONTAINER.md).

`CIW-SESS-012` (v1) On SIGINT or SIGTERM the service MUST stop accepting connections, drain clients, finish in-flight operations, save `workspace.json` to the output directory, then exit. There is no shutdown request in the client protocol; forced termination, termination after a container grace period, and power loss are not covered by autosave, and `workspace.save` is the explicit checkpoint. `serve --resume` MUST reopen `<output-dir>/workspace.json` when present, MUST start a new investigation only when no workspace exists, and MUST fail on corrupt or incompatible saved data. A restarted service assigns a new runtime `session_id`, which clients MUST NOT treat as persistent; run, evidence, execution, and result identities MUST NOT change across restart. The signal path is POSIX: on Windows a console SIGINT (Ctrl+C) takes the same save path through the process signal handler, untested in CI; any other termination is immediate and counts as forced, and the native controller's Stop (explicit `workspace.save` verified against the live session, then termination of the owned process only) is the supported saved-shutdown path.

`CIW-SESS-013` (v1) `ciw health --url` MUST be a read-only `session.get` request bounded at 3 s that runs no scientific calculation and creates no record, MUST validate the snapshot (protocol version, session identity, run identity, selection bounds), and MUST exit 0 printing `{status: "healthy", session_id, run_id}` or 2 otherwise.

`CIW-PERF-012` (v1) `serve --bind` MUST accept only `127.0.0.1` (default) and `0.0.0.0`; `0.0.0.0` MUST be used only inside a container network namespace whose host publishes a loopback port; browser-origin connections MUST be rejected on both binds; there is no authentication.

## 6. Protocol v1 baseline and extension policy

### 6.1 The v1 baseline

`CIW-INST-001` (v1) The session protocol MUST use the envelope `{protocol_version, request_id, type, payload}`: responses `type: "response"`, failures `type: "error"` with `payload: {code, message}`, broadcasts `request_id: null`. The service MUST send `session.snapshot` (the `session.get` payload) on connection; responses and broadcasts MAY interleave, and clients MUST correlate by `request_id`.

`CIW-INST-002` (v1) The service MUST implement the commands below and MUST reject, with a structured error and the session unchanged, a request with an unsupported `protocol_version`, extra envelope fields, a missing or over-long `request_id`, unknown payload fields, non-finite numbers, or booleans where integers are expected. The v1 error codes are `invalid_request` (malformed envelope or non-JSON/binary frame), `unsupported_version`, `unknown_command`, `invalid_payload`, `revision_conflict`, `not_found`, `capacity_exceeded`, and `storage_error`; a client MUST treat an unlisted code as an error of unknown class.

| Type | Semantics | Result |
|---|---|---|
| `session.get` | Snapshot without arrays | `{session_id, run: RUN_METADATA, selection, results: [RESULT_SUMMARY]}` |
| `run.get` | Full run with retained samples and render geometry | RUN |
| `selection.update` | Apply `{expected_revision, cursor_s?, interval_s?, channel?}`, at least one field | SELECTION; broadcast `selection.changed` |
| `sample.get` | Nearest retained sample to `time_s`, earlier on tie | `{run_id, evidence_id, sample_index, time_s, values{}, units{}}` |
| `analysis.stats`, `analysis.spectrum` | Run the operation on `channel` over `interval_s` (default: current selection) | RESULT |
| `result.get` | One result of this session | RESULT |
| `result.list` | Stored analyses, without running them | `{results: [RESULT_SUMMARY]}` |
| `workspace.save` | Write `workspace.json` to the output directory | `{workspace_file}` |

`CIW-PERF-001` (v1) The v1 limits MUST be stated as limits: one uniformly sampled recording per session; only the built-in oscillator record (channels exactly `q` (m), `v` (m/s), `energy` (J); `coordinate_frame` `oscillator-state`; the fixed `render` block with `axis_labels` `["q (m)", "energy (J)", "v (m/s)"]`, `trajectory`, `surface`, `sample_indices`, `transform`), so `ciw serve --recording` and `ciw analyze --recording` accept a file written by `ciw demo` or one with that schema and refuse any other; text JSON only, 1 MiB maximum incoming message; 1,024 results per session; loopback only at the host boundary (container: `0.0.0.0` inside the namespace, published on host loopback), no authentication; no streaming, cancellation, binary arrays, spectrograms, uncertainty, or device acquisition; local JSON persistence only. New analyses produce no event in v1 (CIW-INST-023 adds one at M1); v1 clients refresh `result.list`.

### 6.2 Extension policy

`CIW-INST-003` (v1; minor mechanism M1) `protocol_version` MUST be an integer major (1); a service MUST reject a major it does not implement with `unsupported_version` and MUST reject unknown request fields (CIW-INST-002; in v1 as built, every unknown field). From M1 the service MUST report `protocol_minor` in `session.snapshot`; an addition within a major MUST be an optional field or a new message type declared per minor in `docs/PROTOCOL.md`; a service MUST accept and ignore a declared optional request field it does not implement and MUST still reject undeclared fields; a client MUST ignore unknown optional fields and unknown broadcast types; any change to the meaning of an existing field or message type MUST increment the major. The instrument link carries its own independent version (CIW-INST-019).

`CIW-INST-023` (extension, M1) After a result is registered (file written, ledger updated under the session lock) the service MUST broadcast `result.created` (`request_id: null`, payload RESULT_SUMMARY with `source_kind`) to every client through the per-client queue of CIW-SYNC-021, in registration order relative to that client's selection updates; the requesting client MUST receive its response before its copy of the broadcast; an attached bundle (CIW-SESS-014) broadcasts the same way. A client MUST treat a missed or out-of-order broadcast as a reason to refresh `result.list` or request a snapshot, never as data loss; the result file exists before the broadcast is queued.

Each requirement's tag names its milestone; the table groups the extensions and lists their extension-tagged identifiers; a v1-tagged identifier with a later-milestone part (for example CIW-INST-003, CIW-DATA-008, CIW-DATA-011, CIW-DATA-013) is not repeated here, its tag and conformance row naming the milestone. Every row targets protocol, envelope, and manifest majors 1 / 1 / 1; minors are declared per extension in `docs/PROTOCOL.md` (CIW-INST-003), and a row needing a new major would say so.

| Extension | Adds | Identifiers | Milestone |
|---|---|---|---|
| Manifest and adapter | Manifest, descriptors, capability-driven views, subprocess binding (batch), SDK shim, adapter, operations as instrument code, channel namespace, determinism harness, proto-check, bundle attachment | CIW-DATA-003, CIW-INST-005, CIW-INST-006, CIW-INST-007, CIW-INST-008, CIW-INST-009, CIW-INST-013, CIW-INST-014, CIW-INST-016, CIW-INST-018, CIW-INST-019, CIW-INST-020, CIW-INST-022, CIW-SESS-014, CIW-SYNC-022, CIW-VIEW-001, CIW-EXT-001, CIW-EXT-005, CIW-EXT-006, CIW-EXT-008 | M1 |
| Unit grammar and reductions | UCUM grammar and dimension vector, display-unit preferences, min/max reductions | CIW-DATA-009, CIW-DATA-020, CIW-SYNC-014, CIW-PERF-006 | M1 |
| Result events and envelope v1.1 | `result.created`; protocol minor; payload types, `reduced_of`, `checks[]`, `refused` status, stale marker; non-blocking broadcast queues | CIW-INST-023, CIW-DATA-015, CIW-SYNC-021 | M1 |
| Terminal panels | Textual panels, tiers, command grammar, keybindings, layouts, scripts, `assert`, render tick, `view.update`/`view.changed` | CIW-VIEW-003, CIW-VIEW-004, CIW-VIEW-005, CIW-VIEW-006, CIW-VIEW-007, CIW-VIEW-008, CIW-VIEW-010, CIW-VIEW-011, CIW-VIEW-016, CIW-SYNC-009, CIW-SYNC-010, CIW-SYNC-012, CIW-SYNC-013, CIW-OPS-002, CIW-OPS-005, CIW-OPS-006, CIW-OPS-007, CIW-OPS-008, CIW-OPS-009, CIW-OPS-011, CIW-PERF-003 | M1 |
| Journal and session store | Sequence-numbered journal, session directory, replay, `verify` for operations, csv/arrow/npz exports | CIW-SYNC-008, CIW-SESS-005, CIW-SESS-006, CIW-SESS-008, CIW-SESS-009, CIW-SESS-010, CIW-OPS-004, CIW-EXT-003, CIW-VIEW-022 | M2 |
| Persistence adapter | Resolver mapping the identities to an external infrastructure | CIW-EXT-004 | M2 |
| Uncertainty and revisions | Uncertainty forms and rendering, superseding results, `t_avail`, `as_of`, live parameters, `xy` plots, `verify` for instruments | CIW-DATA-010, CIW-DATA-017, CIW-DATA-018, CIW-INST-015, CIW-VIEW-012, CIW-VIEW-018, CIW-EXT-002 | M3 |
| Binary frames and viewport resources | Bulk buffers on the client link, coordinate-frame tree and gaps, resources by id, shared camera, rasterizer and graphics tier, gltf/png exports | CIW-DATA-016 (bulk plane M1; client link M4), CIW-SYNC-018, CIW-VIEW-017, CIW-VIEW-020 | M4 |
| Streaming | Credits, cancellation, heartbeats, remote and observed bindings, time bases, follow-live, recording, `host` channels | CIW-DATA-005, CIW-DATA-006, CIW-DATA-007, CIW-INST-010, CIW-INST-011, CIW-INST-012, CIW-SYNC-011, CIW-SYNC-020, CIW-SESS-007, CIW-PERF-004, CIW-PERF-005, CIW-PERF-010 | M5 |
| Spectral expansion | Welch, spectrogram, frequency cursor, link groups | CIW-VIEW-014, CIW-VIEW-015, CIW-SYNC-019 | M5 |
| Deployment | Observed service class profile (R11), packaging, measured workloads, fault injection | CIW-PERF-009 | M6 |

## 7. Data model

### 7.1 The record

`CIW-DATA-001` (v1) A run MUST be `{run_id, evidence_id, instrument, metadata{duration_s, sample_rate_hz, sample_count, coordinate_frame, model{}, provenance{}}, time_s[], channels{name: {unit, values[]}}, render{}}`; `time_s` MUST start at zero, be strictly increasing, and be uniform at `sample_rate_hz` with the endpoint excluded; every channel MUST hold `sample_count` finite float64 values; `RUN_METADATA` MUST omit arrays; the service MUST validate the complete run at load and reject any non-finite value. In v1 as built the validator accepts only the oscillator schema of CIW-PERF-001; arbitrary channel sets are an M1 extension through the descriptor of CIW-DATA-003.

`CIW-DATA-002` (v1) `evidence_id` MUST be `sha256:` plus the SHA-256 of the canonical JSON (sorted keys, no whitespace) of `{instrument, metadata, time_s, channels}`; a run whose content does not match MUST be refused. This detects inconsistent content, not authenticity or scientific verification.

### 7.2 Channels and kinds

`CIW-DATA-003` (extension, M1; uncertainty form mandatory M3) From M1 a channel descriptor MUST carry `channel_id`, `kind`, `dtype`, `shape` (inner shape per sample), `unit`, `time_base`, `coordinate_frame` or `null`, `missing`, and `uncertainty` (a form, `none`, or absent); MUST carry `components[]` when `shape` is non-scalar; and MAY carry `cursor_policy` (`nearest` when omitted; `at_or_before` per CIW-SYNC-007). From M3 an `estimate` channel MUST carry a form or `none` with a `reason`. `representations[]` is declared per output (CIW-INST-005, CIW-INST-006) and per result envelope (CIW-DATA-013), never per channel, and applies to every channel of that output. `kind` MUST be `observation`, `estimate`, `derived`, or `reference`; any producer MAY emit any kind; a `derived` channel MUST carry `derived_from[]` naming its inputs' kinds, so a quantity derived from an estimate (innovation, residual) is never presented as an observation; `reference` marks ground truth used for comparison.

Recorded or synthetic evidence standing in for a measurement is `observation` (the v0.1 oscillator's `q`, `v`, and `energy`, once descriptors exist at M1); the output of an estimator, reconstruction, or integrator run inside the session is `estimate`.

### 7.3 Time bases and sampling

`CIW-DATA-004` (v1) Sample times MUST be `time_s`, float64 seconds relative to run start, with `metadata.provenance.time_reference` stating the reference; intervals and cursors on the wire MUST use the same unit; cursors MUST lie within the first and last retained timestamps; an interval end MAY equal `duration_s`.

`CIW-DATA-005` (extension, M5) A time base MUST declare `id`, `clock ∈ {monotonic, wall, sim}`, `epoch` (ISO 8601 with offset for `wall`; session origin for `monotonic`; model t = 0 for `sim`), and `resolution_ns`; streamed timestamps MUST be `int64` nanoseconds relative to the epoch. Bases are declared by a source's data `envelope` or a run's `metadata.time_bases[]` and registered by `id`; a redeclaration with different fields MUST be refused (`time_base_conflict`). The reserved base `run` is the v1 recording base (`epoch` = run start, `resolution_ns` = 1, `t_ns = round_half_even(time_s × 1e9)`); a manifest `time_base` of `input:<name>` resolves at attach to the bound channel's base. From M5 the per-base ns fields of CIW-SYNC-020 are authoritative: a float bound is converted once by the rule above and membership is evaluated in integers as `start_ns ≤ t_ns < end_ns`; the result envelope records the executed `interval_ns` alongside `interval_s`, and `ciw verify` re-executes from it. A base's `epoch` MUST let float64 seconds resolve its `resolution_ns` (for 1 ns, |t_s| ≤ 4.5e6 s); otherwise the base MUST be addressed through the ns fields only and a float update for it MUST be rejected with `precision_insufficient`. Channels in different bases MUST NOT be combined without a declared `mapping {offset_ns, drift_ppb, valid_from, source, version}` with evidence; the default is no mapping, one cursor per base, and `unmapped` in views. A device-clocked source MUST send `clock_sync {tb_ns, host_ns}` at the cadence set in `docs/PROTOCOL.md`; the service MUST fit offset and drift over the last 60 s, store raw timestamps unmodified with the mapping alongside, and version each refit.

`CIW-DATA-006` (extension, M5) A channel MUST declare `regular {rate, phase, jitter_tol}` (`jitter_tol` default set in `docs/PROTOCOL.md`), `irregular`, or `event` sampling; regular frames MAY carry `(t0, dt, n)`. Timestamps MUST be non-decreasing within a channel (`time_order` on violation), equal timestamps kept in arrival order; a regular sample off its grid by more than `jitter_tol` MUST be rejected with `sampling_grid` unless a gap precedes it. Spectral operations MUST require regular sampling.

`CIW-DATA-007` (extension, M5) The service MUST NOT resample implicitly: resampling MUST be an explicit derivation whose provenance names the method (`hold`, `linear`, `nearest`, `pchip`) and `gap_policy` (default `nan`), and uncertainty handling MUST be defined per method (`hold` and `nearest` copy σ; `linear` combines the neighbours' variances). A spectral operation on an `irregular` or `event` channel MUST be refused with `sampling_irregular`; the operator MAY first create an explicit `resample` derivation (method and target rate in provenance) and run the spectrum on it. Default methods for a requested resample: `linear` for observations (including residuals between observations), `none` for estimates (refused unless a method is named).

### 7.4 Units

`CIW-DATA-008` (v1; grammar M1) Every channel MUST declare a UCUM case-sensitive unit with `/` and integer exponents (`m`, `m/s`, `J`, `m/s2`, `rad`, `1`, `Cel`); the v1 spectrum reports `(<unit>)^2/Hz`, which the M1 grammar accepts along with rational exponents (`Hz^(-1/2)`). From M1 the descriptor MUST carry the dimension vector over the seven SI base dimensions plus two sub-dimensions, `angle` and `log`, not interconvertible with `1` (`rad` and `deg` convert only to each other; `dB[V]` and `dB[W]` not to each other). Affine units (`Cel`) MUST appear in arithmetic only as differences. A unit `status: unknown` is legal for raw counts; an `unknown` channel MUST NOT feed a dimension-constrained operation, MUST NOT share an axis with a known channel, and MUST be flagged. Dimension mismatch MUST be refused with `dimension_mismatch`; scale mismatch MUST NOT be converted silently, the service offering an explicit `convert` derivation recorded in provenance.

`CIW-DATA-009` (extension, M1) Unit preferences MUST be display-only: stored values stay in the declared unit; factor and offset are applied at render and export and recorded in the export sidecar; conversion MUST also apply to σ, intervals, and covariance (`J Σ Jᵀ` with the diagonal scale Jacobian); a derived spectral unit MUST be computed as `(<unit>)^2/Hz`.

### 7.5 Uncertainty

`CIW-DATA-010` (extension, M3) Uncertainty, when present, MUST be `stddev`, `variance`, `covariance` (default full row-major `[d, d]` per sample; `packed_lower` and `block_diagonal` MAY be declared), `interval {lower, upper, level}`, `quantiles`, or `ensemble {members}`; absent uncertainty MUST be absent, never zero; it MUST travel as companion columns of the same channel with the same time column. Covariance MUST be symmetric positive semi-definite; the service SHOULD check symmetry and MAY check semi-definiteness by sampled Cholesky, reporting failures as channel-scoped warnings; a unit-quaternion state MUST declare its covariance in the tangent space (`[3, 3]`, `rad2`). A companion column is addressed as `<channel_id>.<column>` with the column fixed by the form (`.stddev`, `.variance`, `.covariance`, `.lower` and `.upper`, `.quantiles`, `.ensemble`; the mask of CIW-DATA-011 is `.mask`); it is not a channel: implied by the declared `uncertainty` or `missing`, it MUST NOT be listed in a frame's `channel_ids[]`, and its `unit` and `shape` MUST equal the form's. A manifest MUST NOT declare a `channel_id` equal to another channel's companion name; the boundary check of CIW-INST-019 MUST accept exactly these columns and no other undeclared column.

### 7.6 Coordinate frames and missing data

`CIW-DATA-011` (v1; extended M4) Every run and spatial channel MUST name a coordinate frame; in v1 `metadata.coordinate_frame`, `render.coordinate_frame`, and the selection's `coordinate_frame` MUST match. From M4 a coordinate-frame declaration MUST carry `id`, `parent` or `null`, `transform` (4×4 row-major float64 to the parent, or a time-indexed transform channel), `handedness`, and axis `units`; chains MUST be acyclic; a spatial view MUST refuse to overlay channels in different frames unless a transform is registered. Missing data MUST be declared per channel as `nan`, `mask` (companion boolean column; required for integer, complex, and table columns), or `none`; gaps MUST be recorded as data `{from, to, reason ∈ {overflow_drop, source_unavailable, cancelled, resample_max_gap, declared, retention}}` that survive persistence and replay; reducers MUST propagate missing values as gaps.

### 7.7 Render geometry

`CIW-DATA-012` (v1) Render geometry (`run.render`) MUST be backend-prepared and non-measurement: it MUST name `coordinate_frame` and `axis_labels` with units, MUST resolve every render point k to a retained sample through strictly increasing `sample_indices[k]`, MUST declare a visual-only `transform {origin, scale, note}`, and MUST never be used to compute a measurement or readout; numerical cards MUST come from `sample.get`. From M4 mesh topology and point positions MAY be sent once as resources by id; level of detail MUST be declared by the instrument or produced by the reducer with a declared method.

### 7.8 Results and the result envelope

`CIW-DATA-013` (v1; fields added M1) A result MUST be `{result_id, evidence_id, operation_id, execution_id, verification_id, verification_status, run_id, selection_revision, channel, interval_s, created_at, recording_file, data}`. From M1 (envelope v1.1) the run-bound fields `run_id, selection_revision, channel, interval_s, recording_file` are grouped as `run_binding`, required for operations on the session run and `null` for an attached run bundle, which instead carries `source {kind: "bundle", bundle_file, bundle_digest}`; v1 results keep the flat fields. The five identity fields MUST be distinct, never derived from one another; two executions of one operation on one evidence MUST share `operation_id` and `evidence_id` and differ in `execution_id` and `result_id`; from M1 an operation with parameters MUST fold a canonical parameter hash into `operation_id`. `verification_id` MUST be `null` and `verification_status` `not_verified` until a distinct verification execution references the result; verification MUST never be inferred from a displayed scene. From M1 the envelope MUST also carry `envelope_version`, `kind`, `channels[]` (descriptors per CIW-DATA-003), `payload {type}`, `representations[]`, `status ∈ {partial, complete, refused, cancelled, failed}` (`refused` per CIW-DATA-021), and optionally `reduced_of`, `supersedes {result_id, interval}`, `checks[]` (instrument self-checks with outcomes; they never set `verification_id`), and `truncated_at {frame_seq, reason ∈ {cancel, instrument_failed, heartbeat_loss, transport, retention}}`, the last valid `frame_seq` of an execution that ended before `end_of_stream`. RESULT_SUMMARY is `{result_id, operation_id, execution_id, channel, interval_s, selection_revision, created_at, verification_status}` (no `evidence_id`, `run_id`, `recording_file`, or `verification_id`; `result.get` returns the full result; encoding in `docs/PROTOCOL.md`); from M1 it also carries `source_kind` (CIW-SESS-014) and `status`.

`CIW-DATA-014` (v1) Results MUST be immutable and MUST capture the `selection_revision` and exact `interval_s` that produced them; `data` MUST be computed from full-resolution retained samples in the half-open interval, never from render geometry or terminal summaries; saved results MUST be validated on reopen (identity form, source binding, shape of `data`) without executing the operation.

`CIW-DATA-015` (extension, M1) `payload.type` MUST be `array`, `table` (columns sharing a time column; Arrow IPC at rest), `mesh`, `sparse` (COO or CSR), or `reference` (URI, content hash, referenced type). A `reference` MUST carry an inline `summary` of a concrete type so views render without dereferencing; dereferencing MUST be explicit and record the resolved hash; an unresolvable reference MUST NOT fail the envelope. An output declared `reduced_of` delivers visualization-resolution data while the full result is retained elsewhere and referenced; views MUST show that data is reduced.

`CIW-DATA-021` (v1 headless bundles; M1 session envelope) A result MUST carry an outcome status separate from transport errors and from verification. A refusal (the instrument declines to evaluate or certify: for PLSR the numerical refusals `NUMERICAL_INCONCLUSIVE` and `NUMERICAL_OVERFLOW`, category `numerical_refusal`, and the domain refusals `OUTSIDE_PARAMETER_BOX` and `OUTSIDE_LEVEL_SET`, category `outside_declared_domain`) and a violation (`NOT_CERTIFIED`, category `certificate_violation`) MUST be stored as valid results with the raw instrument status code and every verdict field retained, MUST NOT be reported as `type: "error"` or a nonzero exit, and MUST NOT set or imply any verification outcome; an input, configuration, or execution failure is an error and produces no result. From M1 the envelope `status` set of CIW-DATA-013 MUST include `refused` alongside `partial`, `complete`, `cancelled`, `failed`.

v1 result as written to `<output-dir>/result-<id>.json`, envelope v1.1 fields marked `+`:

```json
{
  "result_id": "result-3f9c2a7e0b1d4c6e8a5f7b9d1e3c5a70",
  "evidence_id": "sha256:6b1f…c2",
  "operation_id": "statistics.v1",
  "execution_id": "execution-9d0e5b4a3c2f1e8d7b6a5c4f3e2d1b0a",
  "verification_id": null,
  "verification_status": "not_verified",
  "run_id": "run-damped-oscillator-demo-v1",
  "selection_revision": 4,
  "channel": "q",
  "interval_s": [2.0, 8.0],
  "created_at": "2026-09-20T10:14:03.221000+00:00",
  "recording_file": "recording-8a2c…e1.json",
  "data": {"sample_count": 384, "mean": -0.0132, "minimum": -0.5811, "maximum": 0.6094, "rms": 0.2937, "unit": "m"},
  "+envelope_version": "1.1",
  "+kind": "derived",
  "+channels": [{"channel_id": "q.stats", "kind": "derived", "derived_from": ["observation"], "dtype": "f64", "unit": "m", "shape": [],
                 "time_base": "run", "coordinate_frame": null, "missing": "none", "uncertainty": "none"}],
  "+payload": {"type": "table"},
  "+representations": ["numerical"],
  "+status": "complete"
}
```

### 7.9 Frames and the binary descriptor

`CIW-DATA-016` (extension, M1 instrument bulk plane; M4 client link) A bulk frame MUST carry a descriptor sufficient to interpret its buffers without the envelope: `frame_seq`, `execution_id`, `stream` (`<instrument short id>/<output name>`), `result_id` or `channel_ids[]`, a time column or `(t0, dt, n)` with its `time_base`, and per buffer `channel_id` (`null` for the time column), `column` (`value` or a companion column of CIW-DATA-010, never listed in `channel_ids[]`; omitted for the time column), `dtype`, `shape`, `byte_order` (`little`), `order` (`C`), `byte_offset`, `byte_length`, `unit`, and `coordinate_frame` for a spatial buffer. The default transfer and at-rest format is NumPy-compatible raw contiguous buffers with this descriptor; Arrow IPC (pyarrow) is the format for table payloads and export and MAY be negotiated per link; buffers MUST NOT be embedded in text records. On the client link bulk frames share the connection with control messages: a frame or resource message MUST be at most the maximum message size set in `docs/PROTOCOL.md` (larger resources are chunked), the bulk ahead of any control message is bounded by the subscription's outstanding credit (default per CIW-INST-010), and a client MUST grant only credit its link drains within the 2 s stall deadline of CIW-SYNC-021 (over an SSH tunnel, what its measured link rate drains in 1 s).

Instrument-link `data/frame` record (CIW-INST-008) from reference instrument (b); at M4 the service forwards the same `payload` to subscribed clients as a `frame` broadcast (a protocol minor; `bulk` replaced by the binary WebSocket message that follows), framing per `docs/PROTOCOL.md`:

```json
{
  "family": "data", "type": "frame", "seq": 90311,
  "payload": {
    "stream": "attitude-ekf/state", "frame_seq": 4182,
    "execution_id": "execution-9d0e5b4a3c2f1e8d7b6a5c4f3e2d1b0a",
    "channel_ids": ["att.quat"],
    "time": {"column": 0, "time_base": "imu_mono"},
    "buffers": [
      {"channel_id": null, "dtype": "i64", "shape": [256], "byte_order": "little", "order": "C", "byte_offset": 0, "byte_length": 2048, "unit": "ns", "coordinate_frame": null},
      {"channel_id": "att.quat", "column": "value", "dtype": "f64", "shape": [256, 4], "byte_order": "little", "order": "C", "byte_offset": 2048, "byte_length": 8192, "unit": "1", "coordinate_frame": "body_to_nav"},
      {"channel_id": "att.quat", "column": "covariance", "dtype": "f64", "shape": [256, 3, 3], "byte_order": "little", "order": "C", "byte_offset": 10240, "byte_length": 18432, "unit": "rad2", "coordinate_frame": null}
    ],
    "bulk": {"binding": "shm", "segment": "ciw-7f3a-000091", "byte_length": 28672}
  }
}
```

### 7.10 Revisions of estimated states

`CIW-DATA-017` (extension, M3) Results MUST NOT be mutated to revise an estimate: a smoother or fixed-lag estimator MUST emit a new result whose envelope carries `supersedes {result_id, interval}`, the earlier result remaining, marked superseded over that interval. Estimate channels MUST carry validity time (`t_valid`) and availability time (`t_avail`, when the service committed it). View settings MAY carry `as_of`; when set, estimate views MUST show results with `t_avail ≤ as_of` and MUST mark `as_of` with a second, distinct cursor.

`CIW-DATA-018` (extension, M3) A `derived` result computed from channels with uncertainty MUST declare `propagation ∈ {linear, dropped, exact}`; `dropped` MUST render as a visible `σ-DROPPED` flag, never a fabricated value; the default for spectra is `dropped`. A reduction (CIW-DATA-020) is not a propagation: it carries the per-bucket σ extrema of its source channel, declares no `propagation`, and MUST NOT be flagged `σ-DROPPED`; the flag applies only to results of operations.

### 7.11 Provenance and versioning

`CIW-DATA-019` (v1; journal M2) `workspace_version`, `protocol_version`, and from M1 `envelope_version` and `manifest_version` MUST be checked on read; a reader MUST refuse a higher major and, from M1 when a second workspace major exists, MUST name the converter. `metadata.provenance` MUST name generator and version, dtype, time reference, and sampling convention, and for an external instrument the runtime identity of CIW-EXT-009. From M2 the journal MUST record for every execution the operation, parameters, seed, input identities, manifest hash, and a `platform {os, arch, python, numpy, scipy, blas {name, version}, fft {name}, threads}` record, which `ciw verify` MUST also record for its own run.

### 7.12 Reductions

`CIW-DATA-020` (extension, M1) A reduction of a channel to a view's resolution MUST preserve per-column minimum and maximum and, with uncertainty, `max(value + σ)` and `min(value − σ)`; MUST carry the `selection_revision` it was computed for; and a level-of-detail pyramid MUST yield the same envelope as a direct reduction over raw samples, including after a superseding result (property test). Readouts MUST come from full-resolution samples, never from a reduction.

## 8. Instrument contract

### 8.1 The v1 in-process instrument API

`CIW-INST-004` (v1) The v1 in-process instrument module (`ciw.instruments`) MUST provide `make_demo_run() -> dict` (or an equivalent loader), `validate_run(run)`, `run_metadata(run) -> dict`, `inspect_sample(run, time_s) -> dict`, `compute_statistics(run, channel, interval_s) -> dict`, and `compute_spectrum(run, channel, interval_s) -> dict`. Computations MUST validate the complete source recording and their inputs, MUST read full-resolution float64 retained samples, and MUST return operation-specific `data` only; the session wraps provenance and persistence. `compute_spectrum` in v1 MUST be a one-sided, constant-detrended, periodic-Hann periodogram PSD with density normalization `fs · Σw²`, interior bins doubled, DC and the even-length Nyquist bin not doubled, returning `{sample_count, method, window, detrend, scaling, frequency_hz[], psd[], unit, peak_frequency_hz, sample_rate_hz}`; it is not Welch and not a spectrogram.

### 8.2 Manifest

`CIW-INST-005` (extension, M1) Every instrument MUST ship `instrument.json` with `manifest_version`, `id` (reverse-DNS; its last label is the short id, which MUST be unique among attached instruments or `attach` is refused with `instrument_conflict`), `version` (semver), `title`, `binding ∈ {inprocess, subprocess, remote, observed}`, `entrypoint` or `endpoint`, `determinism` with optional `tolerance {abs, rel}`, `isolation`, `max_call_ms` (`inprocess` only; default 50 ms; the built-in oscillator declares 100 ms), `heartbeat_ms`, `restart {policy, max}`, `parameters[]` (name, type, unit, default, range, `scope ∈ {execution, live}`, optional `source ∈ {selection.cursor, selection.interval}`), `inputs[]` (name, accepted kinds, dimension, shape, sampling, required), `outputs[]` (name, `mode ∈ {stream, batch}`, channel descriptors, `representations[]`, `overflow`), and `checks[]`. The service MUST refuse a manifest that fails schema validation and MUST record the manifest hash in provenance; the format is JSON (D4).

`CIW-INST-006` (extension, M1) An output's `representations[]` MUST be a subset of the families that accept its shape; the service MUST offer only those families.

Manifest of reference instrument (b), a quaternion attitude EKF, trimmed to the declared fields (`lag_s > 0` emits superseding results):

```json
{
  "manifest_version": "1.0",
  "id": "org.ciw.ref.attitude-ekf",
  "version": "0.3.1",
  "title": "Attitude EKF (quaternion, gyro + accel)",
  "binding": "subprocess",
  "entrypoint": ["python", "-m", "ciw_ref_instruments.attitude_ekf"],
  "determinism": "deterministic",
  "tolerance": {"abs": 1e-12, "rel": 1e-9},
  "isolation": "process",
  "heartbeat_ms": 1000,
  "restart": {"policy": "manual", "max": 3},
  "parameters": [
    {"name": "gyro_noise_density", "type": "f64", "unit": "rad/s/Hz^(1/2)", "default": 1.7e-4, "range": [0, 1], "scope": "execution"},
    {"name": "lag_s", "type": "f64", "unit": "s", "default": 0.0, "range": [0, 10], "scope": "execution"}
  ],
  "inputs": [
    {"name": "gyro", "kinds": ["observation", "derived"], "dimension": "rad/s", "shape": [3], "sampling": "regular", "coordinate_frame": "body", "required": true},
    {"name": "accel", "kinds": ["observation", "derived"], "dimension": "m/s2", "shape": [3], "sampling": "regular", "coordinate_frame": "body", "required": true}
  ],
  "outputs": [
    {"name": "state", "mode": "stream", "representations": ["numerical", "temporal", "spatial3d"], "overflow": "block",
     "channels": [
       {"channel_id": "att.quat", "kind": "estimate", "dtype": "f64", "shape": [4], "components": ["w", "x", "y", "z"], "unit": "1",
        "coordinate_frame": "body_to_nav", "time_base": "input:gyro", "missing": "nan", "cursor_policy": "nearest",
        "uncertainty": {"form": "covariance", "layout": "full", "shape": [3, 3], "unit": "rad2"}},
       {"channel_id": "att.bias", "kind": "estimate", "dtype": "f64", "shape": [3], "components": ["x", "y", "z"], "unit": "rad/s",
        "coordinate_frame": null, "time_base": "input:gyro", "missing": "nan", "uncertainty": {"form": "stddev"}}
     ]},
    {"name": "innovation", "mode": "stream", "representations": ["numerical", "temporal", "spectral"], "overflow": "block",
     "channels": [
       {"channel_id": "att.nis", "kind": "derived", "derived_from": ["observation", "estimate"], "dtype": "f64", "shape": [], "unit": "1",
        "coordinate_frame": null, "time_base": "input:gyro", "missing": "nan", "uncertainty": "none"}
     ]}
  ],
  "checks": [{"name": "innovation_nis", "description": "NIS inside the chi-square 95 % bound for at least 95 % of samples"}]
}
```

At M3 both outputs run in `batch` mode over the recorded input; `mode: stream` and `overflow` take effect at M5 with live sources.

### 8.3 Lifecycle

```mermaid
stateDiagram-v2
  [*] --> Discovered: manifest read
  Discovered --> Attaching: attach
  Attaching --> Ready: hello / manifest verified
  Ready --> Running: run(execution_id)
  Running --> Draining: inputs ended
  Draining --> Ready: finalized
  Running --> Cancelling: cancel
  Draining --> Cancelling: cancel
  Cancelling --> Ready: cancel_ack (status cancelled)
  Ready --> Detaching: detach
  Detaching --> [*]
  Attaching --> Failed: timeout / error(fatal)
  Ready --> Failed: error(fatal)
  Running --> Failed: error(fatal) / heartbeat loss
  Cancelling --> Failed: deadline exceeded
  Draining --> Failed: drain timeout (M1)
  Failed --> [*]
```

`CIW-INST-007` (extension, M1 batch lifecycle for the subprocess binding; M5 Cancelling, `cancel_ack`, and stream mode) An instrument MUST follow the state machine above: Attaching, reply to `hello` with `manifest` within 5 s, verified against the on-disk copy; Ready, validate parameters and reply `ready` within 30 s of `attach`; Running, consume and emit as credits allow, `heartbeat` every `heartbeat_ms`, `progress` at least every 1 s in batch mode; Draining, emit remaining outputs, `end_of_stream` per output, then `finalized` with per-output content digests within 30 s; Cancelling, stop producing and send `cancel_ack` within `2 × heartbeat_ms`, frames already sent remaining valid and the envelope closed `cancelled` with `truncated_at {frame_seq, reason: cancel}`; Failed, emit `error(fatal)` before exit. On any expiry the service MUST move the instrument to Failed and terminate its process group (from M5 only after a `cancel` per CIW-INST-012 goes unacknowledged), retaining everything received and closing the envelope with `status: failed` and `truncated_at {frame_seq: <last valid>, reason: instrument_failed}`. Each `run` MUST carry a new `execution_id`; every transition MUST be journaled; concurrent executions require `concurrency > 1` in the manifest.

### 8.4 Message families

`CIW-INST-008` (extension, M1 families and batch types; M5 `credit`, `cancel`, `cancel_ack`, `gap`, `watermark`, `clock_sync`, `checkpoint`, `seek`) The instrument protocol MUST have exactly four families; each record carries `family`, `type`, `seq` (per direction, per link), and `correlation_id` for request/response pairs. Control and bulk data MUST travel on separate planes, buffers referenced by offset or segment name and never embedded in text records. `checkpoint` and `seek` are optional capabilities: an opaque blob a later `run` may resume from, so replay can seek without recomputing.

| Family | Direction | Types (minimum) |
|---|---|---|
| `control` | service → instrument | `hello`, `attach`, `set_params`, `run`, `cancel`, `detach`, `credit`, `ping`, `checkpoint` |
| `control` | instrument → service | `hello`, `manifest`, `ready`, `params_ack`, `run_ack`, `cancel_ack`, `pong`, `checkpoint`, `detached` |
| `data` | instrument → service | `envelope`, `frame`, `resource`, `gap`, `watermark`, `clock_sync`, `end_of_stream`, `finalized` |
| `data` | service → instrument | `input_frame`, `input_end` |
| `event` | both | `progress`, `log`, `heartbeat`, `check_result` |
| `error` | both | `error` |

### 8.5 Ordering

`CIW-INST-009` (extension, M1) Within one link direction records MUST be delivered in `seq` order; within one output stream frames MUST arrive in `frame_seq` order without gaps and the envelope MUST precede the first frame; an out-of-order frame MUST be rejected with `seq_gap`, never reordered. Across streams and instruments the journal defines the session order. Control and data on one link arrive in send order, so an instrument MUST check for control records at the cadence set in `docs/PROTOCOL.md` while draining input.

### 8.6 Backpressure

`CIW-INST-010` (extension, M5) Every bulk stream MUST be credit-based in frames and bytes: the consumer grants, the producer MUST NOT send without sufficient credit in both units, the consumer replenishes as it commits; defaults 64 frames and 64 MiB per instrument stream, 8 frames and 8 MiB per client subscription. The service is the only component that buffers between instruments, so a slow consumer bounds read-ahead and never stalls the producer; the recorder is a credit consumer with priority, and when it cannot keep up ingest credits stop. Until M5 a batch output is delivered whole after completion (CIW-INST-016) and no `credit` records are exchanged, a stated limit of the M1 binding.

`CIW-INST-011` (extension, M5) On exhausted credit an instrument MUST apply its output's declared overflow policy: `block` (default), `drop_oldest` (reporting the dropped range as a `gap` record), or `fail`; a deterministic or seeded instrument MUST NOT drop. Silent loss is not permitted; the conformance suite MUST withhold credit and assert that a `drop_oldest` source emits gaps.

### 8.7 Cancellation

`CIW-INST-012` (extension, M5) On `cancel(execution_id)` the instrument MUST stop producing for that execution and send `cancel_ack` within `2 × heartbeat_ms`; a non-acknowledging instrument MUST be moved to `Failed` and, for the subprocess binding, its process group terminated. Cancelling a client-requested analysis MUST leave earlier results untouched and MUST produce a `status: cancelled` result when partial frames were delivered; `status: cancelled` MUST be produced only in response to `cancel`, every other early end being `failed`.

### 8.8 Errors and restart

`CIW-INST-013` (extension, M1) Errors MUST be structured `{code, class, severity ∈ {warning, error, fatal}, recoverable, scope ∈ {link, instrument, execution, stream, frame}, message, detail}`, journaled and surfaced. Restart policy is `manual` by default; `auto` restarts a crashed subprocess at most `restart.max` (default 3) times after a `recoverable` error; every restart is a new execution.

| Class (code prefix) | Fatal default | Service action |
|---|---|---|
| `param_*` (`param_range`, `param_type`, `param_unknown`) | yes | Reported against the parameter; execution not started |
| `input_*` (`input_schema`, `input_unit`, `input_gap`, `input_nan`) | schema: yes; data: no | Data errors attached to the stream at `frame_seq`; markers on the temporal axis |
| `overrun` | no | Gap recorded and rendered |
| `resource_*` | yes | Execution failed; message shown |
| `numeric_*` (`numeric_diverged`, `numeric_singular`) | instrument's choice | A non-fatal numeric error MUST still emit a time-aligned sample (NaN value, infinite covariance) so estimate output never loses alignment |
| `transport` | yes (service-generated) | Envelope closed `status: failed`, `truncated_at {reason: transport}` |
| `contract` (bad seq, credit exceeded, descriptor mismatch, `call_overrun`) | yes (service-generated); `call_overrun` non-fatal below 10× the budget (CIW-INST-018) | Execution failed; offending record logged |
| `internal` | yes | As `resource_*` |

### 8.9 Determinism and parameters

`CIW-INST-014` (extension, M1) The manifest MUST declare `determinism` as `deterministic` (equal within declared `tolerance`; bitwise if none), `seeded` (deterministic given a `seed` the service supplies and records), or `nondeterministic`. `finalized` MUST carry a content digest per output, which the service computes independently. Deterministic and seeded instruments MUST produce the same sample sequence regardless of how inputs were split into frames; the SDK MUST ship a harness that re-frames inputs randomly; the conformance suite MUST verify determinism claims by re-execution for each reference instrument.

`CIW-INST-015` (extension, M3) A `scope: execution` parameter MUST NOT affect a running execution; it applies to the next `run` and yields a new `operation_id`. A `scope: live` parameter MAY change during an execution; every change MUST be journaled and recorded in the envelope's parameter history.

### 8.10 Streaming, batch, and selection isolation

`CIW-INST-016` (extension, M1 batch; M5 stream) A `stream` output MUST deliver frames as produced and end with `end_of_stream`; a `batch` output MUST deliver its envelope and all frames after completion, then `end_of_stream`; a batch execution over a selection MUST receive the interval explicitly and record it in the envelope.

`CIW-INST-017` (v1) Instruments MUST NOT read the selection or view settings; where an instrument needs the cursor or interval, the manifest MUST declare a parameter with `source: selection.cursor` or `selection.interval`, delivered with the `revision` it came from. v1 already works this way: `compute_statistics` and `compute_spectrum` receive `interval_s` explicitly and the result captures `selection_revision`.

### 8.11 Bindings

`CIW-INST-018` (extension, M1 subprocess; M5 remote, observed) Semantics MUST be identical across bindings, which differ only in transport; an instrument MUST run under `subprocess` and `remote` without code change, and an in-process instrument MUST be wrappable as a subprocess by a generic SDK shim. For the `inprocess` binding, from M1 the service MUST measure each call against the instrument's `max_call_ms` (CIW-INST-005; default 50 ms): an overrun MUST be journaled as a `contract` error `call_overrun`, non-fatal below 10× the budget; at 10× the caller stops awaiting, the instrument is marked `Failed` for further calls, and the running thread is abandoned, never killed, any value it returns later being discarded and journaled, never registered as a result; abandoned threads are capped (default 2), beyond which further in-process calls MUST be refused with `resource_exhausted`, naming the subprocess binding in the message. `resource_exhausted` is an M1 addition to the error codes of CIW-INST-002 under CIW-INST-003.

| Binding | Control plane | Bulk plane | Liveness | Isolation |
|---|---|---|---|---|
| `inprocess` (v1 binding; budget M1) | Python calls with the same record shapes | NumPy arrays by reference | v1: none (bounded calls by convention); from M1 the `max_call_ms` budget above, measured after each call, never pre-empted | None |
| `subprocess` (default from M1) | Length-prefixed JSON on stdin/stdout; stderr as log | Shared-memory segment named in the frame record; inline fallback | Heartbeat | Process |
| `remote` | `ssh://host//path/exec` (stdio forwarded, no daemon) or TCP with TLS and token | Inline or `reference` payloads | Heartbeat | Process and machine |
| `observed` | Any of the above via an adapter | Reduced telemetry only | Heartbeat | The engine's own |

### 8.12 Versioning, invariants, conformance harness

`CIW-INST-019` (extension, M1) `hello` on the instrument link MUST carry `instrument_protocol {major, minor}`, independent of the client `protocol_version`; majors MUST match or the link is refused with `unsupported_version`; the minor in effect is the lower of the two; unknown fields on the instrument link MUST be ignored; manifest, envelope, and both protocol versions are independent. The service MUST enforce at the boundary that every frame column is a channel declared in the current envelope or a companion column of one (CIW-DATA-010), every buffer's byte length equals `dtype size × product(shape)`, timestamps are non-decreasing, the envelope's `kind` matches every channel's, every unit parses, and every spatial channel names a declared coordinate frame.

`CIW-INST-020` (extension, M1; M5 credit and cancel) `docs/PROTOCOL.md` MUST specify the concrete encoding for every requirement in this section and MUST ship a harness (`ciw-proto-check <manifest>`) that drives an instrument through attach, run (batch), and detach at M1, plus credit exhaustion and cancel at M5, reporting pass or fail per identifier.

`CIW-INST-022` (extension, M1 attach/detach/run/result.attach; M4 subscribe/credit; M5 cancel) The client session protocol MUST carry the operation loop as request types, encoded per `docs/PROTOCOL.md` as minor additions under CIW-INST-003: at M1 `instrument.attach {id | manifest_path, inputs{name: channel_id}} → {short_id, state}` (errors `instrument_conflict`, `channel_conflict`, `manifest_invalid`), `instrument.detach {short_id} → {state}`, `execution.run {short_id, parameters{}, interval_s?} → {execution_id}` (asynchronous per CIW-OPS-003; completion is the `result.created` broadcast of CIW-INST-023), `result.attach {bundle_file} → RESULT_SUMMARY` (CIW-SESS-014), and the broadcasts `instrument.changed {short_id, state}` and `execution.changed {execution_id, status}`; at M4 `stream.subscribe {stream | channel_ids[], resolution} → {subscription_id}`, `stream.unsubscribe`, and `stream.credit {subscription_id, frames, bytes}` (CIW-INST-010 defaults); at M5 `execution.cancel {execution_id} → {status}` (CIW-INST-012; `unsupported` before M5: an execution cannot be cancelled, and a terminated subprocess closes with `status: failed` per CIW-INST-007). Every such request MUST be sequenced and journaled (CIW-OPS-004) and MUST leave the selection unchanged.

### 8.13 Headless instrument path

`CIW-INST-021` (v1) An instrument MAY be driven from the terminal client without the session service. Such a run MUST be written as one self-contained immutable JSON bundle embedding the complete input declaration, the explicit sample or inputs, the instrument's own output record, and the runtime identity; it MUST carry separate `evidence_id`, `operation_id`, `execution_id`, `result_id`, and `verification_id` (`null`, `verification_status: "not_verified"`) and content digests whose relationships are validated on read; `inspect` MUST validate a bundle without recomputation; `replay` MUST re-evaluate the embedded inputs under new execution and result identities, record `replay_of` source bindings and an exact digest comparison, and MUST NOT modify the source bundle or promote verification. Adapter obligations toward the upstream engine are CIW-EXT-009; the catalogue rule is CIW-EXT-010.

`CIW-SESS-014` (extension, M1) A run bundle MAY be attached to a session by `result.attach {bundle_file}` (CIW-INST-022). The service MUST copy the bundle unchanged into the output directory under its own basename (`run-<result uuid>.json`, 13.8; M2: `bundles/`), refusing with `storage_error` when a file of that name holds different content; MUST verify `bundle_digest` on the copy; MUST register it as an immutable result of the workspace with its five identities and `verification_status` unchanged, `bundle_digest` as the source binding, and `bundle_file` recorded relative to the output directory exactly as `recording_file`; MUST NOT re-evaluate the bundle on attach or on reopen; and MUST refuse with `invalid_payload`, leaving the session unchanged, a bundle whose `result_id` or `execution_id` is already in the ledger. An attached bundle MUST appear in `result.list` and `result.get` as a result whose `payload` is a `reference` to the bundle (CIW-DATA-015) with the instrument's status record as the inline summary, with views offered only per CIW-VIEW-001. `results[]` MAY therefore hold results whose `evidence_id` differs from `run.evidence_id`; RESULT_SUMMARY MUST carry `source_kind ∈ {run, bundle}`, with `channel`, `interval_s`, and `selection_revision` `null` for a bundle row, which a client MUST tolerate only when `source_kind` is present (protocol minor 1.1; non-null fields keep their v1 meaning). A workspace holding an attached bundle MUST be saved with `workspace_version: 2`, which a v1 reader refuses under CIW-SESS-004 naming the version; a v2 reader MUST accept `workspace_version: 1`. Until M1 a bundle is handled only by the headless path.

## 9. Synchronization model

### 9.1 The shared selection

`CIW-SYNC-001` (v1) A session MUST have exactly one selection, held by the service; clients MUST NOT keep a divergent copy; every change MUST pass through `selection.update` or, from M5, the service's own follow mutation of CIW-SYNC-011, sequenced identically.

`CIW-SYNC-002` (v1) The selection MUST be `{run_id, channel, interval_s: [start, end), cursor_s, coordinate_frame, revision}`; the initial selection MUST be the run's default channel over the full run with the cursor at the first sample and `revision: 0`; the v1 default channel is `q` (fixed by `docs/PROTOCOL.md`), from M1 the first channel declared by the run's instrument manifest output.

```json
{"run_id": "run-damped-oscillator-demo-v1", "channel": "q", "interval_s": [2.0, 8.0],
 "cursor_s": 3.0, "coordinate_frame": "oscillator-state", "revision": 4}
```

`CIW-SYNC-022` (extension, M1) Channel references are bare `channel_id`s in one session namespace (a dot is part of the id). Every `channel_id` MUST be unique across the session's run and the declared outputs of every attached instrument, or `attach` MUST be refused with `channel_conflict`; the selection's `channel`, `view_settings.channels`, and every channel argument in 12.2 use this form (in v1 the namespace is the run's channels). Channels produced by an execution belong to its result (`channels[]`, CIW-DATA-013), become addressable when the envelope is committed, and are addressed in qualified form `<channel_id>:<execution_id>` (`:` never appears inside a `channel_id`). A bare id declared only by the run resolves to the run channel; a bare id produced by results MUST be resolved by the service, when the `selection.update`, `view.update`, or command is accepted, to the most recently committed non-superseded result declaring it (journal order from M2; `created_at` at M1), and the selection, `view_settings.channels`, and every broadcast MUST carry the qualified form; `compare` pairs, residuals, and `assert` expressions (`att.quat:execution-9d0e…@cursor`) use it too.

### 9.2 Revisions and conflict

`CIW-SYNC-003` (v1) Every `selection.update` MUST carry `expected_revision`; the service MUST apply it only if equal to the current revision, MUST increment `revision` by exactly one, and MUST reject a stale update with `revision_conflict` without mutating anything. An update naming no field besides `expected_revision` MUST be rejected with `invalid_payload`; an update restating the current values MUST be accepted and MUST increment `revision` (v1 as built); revisions MUST be integers compared numerically, booleans rejected.

`CIW-SYNC-004` (v1) After an accepted update the service MUST broadcast `selection.changed` with the full selection to every client in revision order and MUST send `session.snapshot` to every new connection; a client MUST apply broadcasts in increasing revision and discard one not greater than the revision it shows.

```mermaid
sequenceDiagram
  participant T as Terminal client
  participant S as Session service
  participant V as Viewport
  T->>S: selection.update {expected_revision: 4, cursor_s: 3.5}
  S->>S: lock, check revision == 4, apply, revision = 5
  S-->>T: response SELECTION(rev 5)
  S-->>V: selection.changed SELECTION(rev 5)
  S-->>T: selection.changed SELECTION(rev 5)
  V->>S: selection.update {expected_revision: 4, interval_s: [1, 9]}
  S-->>V: error revision_conflict
  V->>S: session.get
  S-->>V: response snapshot(rev 5)
```

### 9.3 Cursor versus interval

`CIW-SYNC-005` (v1) The playback cursor MUST be separate from the analysis interval: a cursor update MUST NOT change `interval_s` and MUST NOT recompute or invalidate any analysis. Analyses MUST consume `[start, end)` with `0 ≤ start < end ≤ duration_s`; an interval with no retained sample MUST be rejected. A cursor-anchored spectrum, if offered, MUST be an explicit command creating a new result from the interval derived from the cursor at that revision.

### 9.4 Client discipline and cursor resolution

`CIW-SYNC-006` (v1) Clients MUST coalesce selection updates to at most 10 per second with one outstanding, reject obsolete responses, poll a snapshot every 5 s, time out unanswered requests at 8 s, show STALE and disable shared interaction on disconnect, and offer Reconnect; on `revision_conflict` a client MUST discard pending intent and refresh rather than replay it over another client's selection.

`CIW-SYNC-007` (v1; policy M5) Cursor resolution MUST be the nearest retained sample, earlier on a tie (`inspect_sample`), and a readout MUST show the resolved sample's own time. From M5 a channel MAY declare `cursor_policy: at_or_before`, which follow-live views MUST use for live channels; a readout MUST show the cursor-to-sample offset when it exceeds the nominal interval.

### 9.5 Total order and journal

`CIW-SYNC-008` (extension, M2) Every selection mutation, command, result creation, and instrument event MUST pass through one sequencer that assigns a strictly increasing journal sequence number before publishing; a client observing a gap MUST resync from a snapshot. The service MAY coalesce consecutive cursor updates from one client within 100 ms into the last, MUST NOT coalesce across fields or clients, and MUST always sequence the final value.

### 9.6 View settings

`CIW-SYNC-009` (extension; † M1, `as_of` and `compare` M3, `camera` M4, `follow` and `freq_cursor` M5) `view_settings` (empty in v1) MUST hold exactly the fields below when populated. A change to any field except `focus` MUST be a revisioned update that never alters stored data and shares the selection's revision counter: sent as `view.update {expected_revision, <fields>}`, applied by the selection sequencer under the same lock with the conflict rule of CIW-SYNC-003, and broadcast as `view.changed {selection, view_settings, revision}` (without `focus`). One sequenced mutation MAY change selection and view-settings fields together (CIW-SYNC-011) as one revision and one `view.changed` with no separate `selection.changed` (the discard rule of CIW-SYNC-004 applies). From M1 `session.snapshot` and the `session.get` response MUST include `view_settings` alongside `protocol_minor` (CIW-INST-003). `focus` is per-client presentation state: never broadcast, persisted only as the saving client's value, and MUST NOT change in another client.

| Field | Type | Default | Meaning |
|---|---|---|---|
| `window` † | `{start_s, end_s}` | full run; last 10 s when live | Visible interval of axis group `t` |
| `follow` | `off \| edge \| offset {lag_s}` | `edge` while any live stream exists | Window tracks newest data (9.7) |
| `units` † | quantity kind → unit | declared units | Display unit per quantity kind |
| `sigma_k` † | number | 2 | Uncertainty band multiple |
| `time_display` † | `relative \| absolute` | `relative` | Axis and readout labelling |
| `channels` † | pane → ordered bindings | selection channel | Active channels per pane |
| `layout` † | pane tree | `single` | Section 12.3 |
| `focus` † | pane id | first pane | Pane receiving keys (per client; not synchronized) |
| `freq_cursor` | Hz or `null` | `null` | Shared by spectral panes |
| `camera` | pane → `{kind, target, distance, yaw, pitch, fov, up}` | per-pane default | Projection shared by terminal raster and viewport |
| `as_of` | time or `latest` | `latest` | Estimate availability epoch (7.10) |
| `compare` | ordered `execution_id` pairs | empty | Runs under comparison |

```json
{"window": {"start_s": 0.0, "end_s": 12.0}, "follow": "off", "units": {"length": "mm", "angle": "deg"}, "sigma_k": 2,
 "time_display": "relative", "channels": {"p1": ["q"], "p2": ["q", "v"]},
 "layout": {"split": "h", "ratio": 0.6, "a": {"pane": "p1"}, "b": {"pane": "p2"}}, "focus": "p1", "freq_cursor": null,
 "camera": {"p3": {"kind": "orbit", "target": [0, 0, 0], "distance": 2.5, "yaw": 0.6, "pitch": 0.3, "fov": 45, "up": [0, 1, 0]}},
 "as_of": "latest", "compare": []}
```

### 9.7 Snapshot coherence and data arrival

`CIW-SYNC-010` (extension, M1) All panes rendered in one tick MUST render from one snapshot `(revision, W, t_tick, seen)`: `W` the per-channel watermarks at the start of the tick; `t_tick` the client's monotonic clock in ns captured once at the start of the tick; `seen` mapping each `result_id` visible to the client to the `t_tick` of the first snapshot that included it, maintained by the snapshot builder, never by a renderer. A pane MUST NOT read live state while rendering; every rendered or exported frame MUST be attributable to its `(revision, W, t_tick)`; the status line SHOULD show the revision and the focused channel's watermark lag.

`CIW-SYNC-011` (extension, M5) Data arrival MUST NOT mutate the selection or view settings directly. The service is the only follow source: while `follow ≠ off`, the selection sequencer MUST apply at most one follow mutation per `follow_period_ms` (default set in `docs/PROTOCOL.md`), computed from the watermarks at that instant, moving `window` and `cursor` together (`edge` moves `end_s` to the newest watermark keeping the width and places the cursor there; `offset` keeps the cursor `lag_s` behind), sequenced under the selection lock as one revision, journaled with origin `system`, and broadcast per CIW-SYNC-009; clients MUST NOT send follow-derived updates. An explicit `selection.update` carrying `cursor_s` (or `cursors`, CIW-SYNC-020) or an explicit `view.update` carrying `window`, accepted while `follow ≠ off`, MUST set `follow: off` in the same sequenced mutation under the same revision.

### 9.8 Resolution independence and units

`CIW-SYNC-012` (extension, M1) Each pane MUST derive its own quantization from the window and its area and MUST NOT modify window, cursor, or interval to fit it; two clients at different widths showing the same `(revision, W)` MUST differ only in resolution, never in which records they show.

`CIW-SYNC-013` (extension, M1) Cursor and interval edges MUST be drawn at the column whose bucket contains the exact instant; coincident edges MUST be drawn as one marked column; every pane MUST show the same cursor time in its header at the same revision.

`CIW-SYNC-014` (extension, M1) A change to `units` MUST take effect in every pane and export in the same revision, following CIW-DATA-009; the status line MUST show the effective unit per quantity kind.

### 9.9 Spectral coherence

`CIW-SYNC-015` (v1 rule; spectrogram M5) A spectral result MUST be computed by the service from the retained samples of exactly the interval captured in the result, never by a client and never from the cursor; the temporal pane MUST shade that interval, snapped inward to sample boundaries. A spectrogram's time axis MUST coincide with the temporal `window`, each column timestamped at the centre of its segment; `spectrum.stft.v1` MUST take `segment` and `hop` as explicit operation parameters (defaults set in `docs/PROTOCOL.md`) folded into `operation_id`, so the result's columns are fixed by those parameters and the interval; a spectrogram pane MUST fit result columns to its bucket width through the reducer (per-bucket maximum and minimum per frequency row, CIW-DATA-020) and MUST NOT request a recompute to fit its width; the temporal cursor MUST map to the nearest column centre, shown in the readout. `freq_cursor` MUST be shared by all spectral panes and read out at or before the bin; the frequency axis MUST derive from the actual sampling in the interval.

`CIW-SYNC-016` (v1 identity; marker M1) A displayed result MUST show its `selection_revision` and `interval_s`; when the current selection's channel or interval differs, the pane MUST mark the result stale and the service MUST NOT recompute automatically. A pending recompute (extension) MUST render the previous result dimmed without blocking the tick.

### 9.10 Latency, camera, link groups, time bases

`CIW-SYNC-017` (v1 stated; benchmark M1 terminal, M4 viewport) A sequenced selection change MUST be visible in every attached client within CIW-PERF-002.

`CIW-SYNC-018` (extension, M4) `camera` MUST be part of view settings so the terminal raster and the viewport of one pane show the same projection at the same revision; the viewport MAY predict while dragging and MUST reconcile to the sequenced value.

`CIW-SYNC-019` (extension, M5) Every temporal axis MUST belong to one link group: `t` links all temporal axes and spectrogram time axes, `f` all frequency axes; a window change applies to every axis of the group in one revision.

`CIW-SYNC-020` (extension, M5) With channels in more than one time base selected, the selection MUST hold one cursor per base, linked only through declared mappings. From M5, as a protocol minor, the selection MUST carry `cursors {<time_base_id>: {cursor_ns}}` and `intervals {<time_base_id>: [start_ns, end_ns)}` alongside the v1 fields; `cursors` and `intervals` are authoritative and `cursor_s` and `interval_s` are service-computed projections onto the run's base (with one time base the v1 fields keep their meaning), an update MAY carry either form, never both (`invalid_payload`), a float form accepted only under the conversion rule of CIW-DATA-005. Cursor resolution converts `cursor_s` to `cursor_ns` by that rule before the nearest-sample rule of CIW-SYNC-007.

`CIW-SYNC-021` (extension, M1) The sequencer MUST NOT await network delivery, a close handshake, or a drain under the lock: after applying an update it MUST append the broadcast to a per-client bounded outbound queue (size set in `docs/PROTOCOL.md`) and return; a client whose queue overflows or that does not drain a message (bulk frames included) within 2 s MUST be disconnected by a task scheduled off the sequencing path; per-client order is preserved by the queue. A client whose credit exceeds what its link drains in 2 s is disconnected by this rule, not exempted.

## 10. Representation layer

### 10.1 Capability-driven views; clients own no calculation

`CIW-VIEW-001` (extension, M1) A view family MUST be offered for an output only if its `representations[]` includes it; no client or service MAY synthesize a spectral or spatial view an output does not declare. v1 implies the families from the run's channels and `render` block; at M1 the demo manifest declares `numerical`, `temporal`, `spectral`, `spatial3d`.

`CIW-VIEW-002` (v1) Clients MUST NOT compute anything that produces a record or readout: numerical cards MUST be `sample.get` or `result.get` values, traces MUST come from retained samples or service reductions, and render geometry is display only.

`CIW-VIEW-003` (extension, M1 styling; M3 uncertainty) Every view MUST distinguish kinds by one convention: observations solid, glyph `O`; estimates dashed (braille dot pattern) and banded, `E`; derived in their inputs' pattern, `D`; reference dotted, `R`. Declared uncertainty MUST be rendered: a `±σ` column (numerical), a `sigma_k` band (temporal; band edges as dotted traces on braille and cell backends), an ellipse or ellipsoid at the cursor sample (2D/3D), a `σ` column (exports). `partial`, `cancelled`, and `reduced_of` records MUST be labelled.

### 10.2 Representation interface

`CIW-VIEW-004` (extension, M1) A representation MUST implement `describe() → {family, name, accepts, min_channels, max_channels}`, `bind(bindings, descriptors) → ok | error`, `measure(area, backend) → quantization`, `render(snapshot, area, raster) → report` (stale channels, resolved sample per channel), `hit_test(area, cell_or_pixel) → {time, channel, value} | none`, and optionally `keymap()`. `render` MUST be deterministic: one snapshot (with `t_tick` and `seen`), area, and backend yield a byte-identical raster; renderers MUST NOT read clocks, random sources, or state outside the snapshot.

### 10.3 Terminal tiers

`CIW-VIEW-005` (extension, M1) Every representation MUST provide a tier-0 rendering (plain ASCII, 8 colours) with the same channels, cursor, interval, and units at reduced fidelity; every command and view MUST work over SSH on a plain 80×24 `xterm-256color`; nothing in the operation loop MAY depend on a higher tier.

`CIW-VIEW-006` (extension, M1 tiers 0–2; M4 graphics) The terminal client MUST classify the terminal at startup and on resize and select the highest confirmed tier (`--tier` overrides): 0 cell; 1 braille 2×4 and half-block 1×2, 256 colours; 2 true colour; 3 graphics, `kitty > iterm2 > sixel`. Tier 3 MUST be confirmed by terminal response (Kitty: APC graphics query; Sixel: DA1 attribute 4), never by `TERM`; a query waits max(100 ms, 4 × the measured DA1 round trip), at most 1 s, and on timeout the tier stays unconfirmed (`tier 2 (unconfirmed 3)` in the status line). iTerm2, which has no query, is confirmed by `TERM_PROGRAM=iTerm.app` or `LC_TERMINAL=iTerm2` alone and falls back one tier on a failed first image write; under tmux or screen tier 3 requires passthrough and defaults to 2. A runtime backend error MUST fall back one tier. Tier-3 payloads MUST be ≤ 256 KiB per pane per frame; a pane whose tier-3 write misses one tick period on two consecutive ticks (Kitty: or lacks a `q=0` acknowledgement within one tick) drops to tier 2 for the session.

`CIW-VIEW-007` (extension, M1) Backend selection MUST NOT change the quantities, cursor, interval, or units shown; the render report MUST be identical across backends for one snapshot.

### 10.4 Numerical

```
 channel      comp  value       ±σ       unit      Δt        flags   kind
 q            —     -0.31074    —        m         +0.0 ms           O
 att.quat     w      0.99862    0.00031  1         +2.1 ms           E
 att.nis      —      1.42       —        1         +2.1 ms           D
```

`CIW-VIEW-008` (extension, M1) A numerical pane MUST show per bound component: value in the display unit, declared uncertainty in the same unit, unit, the resolved sample's offset from the cursor, flags (`stale`, `σ-DROPPED`, `unknown-unit`, `reduced`, `superseded`), and kind glyph. Precision MUST be two significant digits of σ (value rounded to the same place) when σ is present, else six significant digits. A `readout` variant (one large value, uncertainty, sparkline) MUST be available.

`CIW-VIEW-009` (v1 statistics; pane M1) Over the analysis interval a numerical pane MUST offer the `statistics.v1` fields `sample_count`, `mean`, `minimum`, `maximum`, `rms` in `unit`, and from M1 `std` from `statistics.v2`, a new operation with a new `operation_id` (`statistics.v1` MUST remain unchanged); aggregates of an estimate MUST be computed on values, not uncertainty, and MUST come from a result, never a client-side reduction.

### 10.5 Temporal

`CIW-VIEW-010` (extension, M1) A temporal pane MUST render each column as the min–max envelope of the retained samples in its bucket (CIW-DATA-020), gaps as breaks, the cursor as a full-height rule at the exact instant, and the interval shaded; a bucket with fewer than two samples MUST be interpolated per view policy, default `linear` for observations and `none` for estimates.

```
 q [m]                            rev 4   interval [2.0, 8.0)   follow: off
  1.0 ┤⢀⡀     ⢀⡀      ⢀⡀     ⢀⣀      ⢀⡀     ⢀⡀   │   ⢀⡀
  0.0 ┤⠈⠉⠑⠒⠒⠊⠁⠈⠑⠒⠒⠒⠊⠁⠈⠑⠒⠒⠊⠁ ⠈⠒⠒⠒⠊⠁⠈⠑⠒⠒⠊⠁⠈⠑⠒⠒│⠒⠊⠁⠈⠑
 -1.0 ┤▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒▒│
      └──────────┬──────────┬──────────┬──────────┼──────
               2.0s       4.0s       6.0s       8.0s
```

`CIW-VIEW-011` (extension, M1) Time-axis labels MUST follow `time_display`; tick spacing MUST come from {1, 2, 5} × 10ⁿ; the header MUST show the cursor's exact time.

`CIW-VIEW-012` (extension, M3) A region superseded by a revision result MUST be marked on the time axis on every tick with `t_tick − seen[superseding result_id] < 1 s` (both from the snapshot) and at least on the first such tick; a value resolved from a superseded result MUST be flagged `superseded` unless `as_of` selects it.

### 10.6 Spectral

`CIW-VIEW-013` (v1 header content; pane M1) A spectral pane MUST show the operation and settings from the result's `data` (method, window, detrend, scaling, sample count, sample rate) and the result's `selection_revision` and `interval_s`, never pane-local settings; the PSD is a trace over frequency, log or linear axes, `freq_cursor` as the rule, density unit per CIW-DATA-009.

`CIW-VIEW-014` (extension, M5) A spectrogram pane MUST obey CIW-SYNC-015; its default colormap MUST be perceptually uniform with monotone luminance, Sixel and 256-colour palettes quantized from it; half-block carries two frequency rows per cell; braille and cell backends threshold into three levels.

`CIW-VIEW-015` (extension, M5) Welch (`spectrum.welch.v1`; segment, window, and overlap defaults set in `docs/PROTOCOL.md`) and spectrogram (`spectrum.stft.v1`) MUST be new operations with new `operation_id`s; `spectrum.periodogram.v1` MUST remain unchanged.

### 10.7 2D/3D

`CIW-VIEW-016` (extension, M1 preview; rasterizer M4) Without a viewport the terminal MUST render `spatial2d` outputs at the selected tier and `spatial3d` outputs at least as a labelled orthographic projection along a selectable axis. From M4 a software rasterizer MUST render wireframe meshes, depth-sorted points, and trajectories with the pane's camera into braille, half-block, or a graphics image, producing the same clip-space coordinates before quantization as the viewport for the same camera, so a pick in either maps to the same world coordinates. Attaching a viewport MUST NOT change what the terminal shows.

`CIW-VIEW-017` (extension, M4) A 2D/3D pane MUST render the sample resolved at the cursor; trajectories MUST show the trail over `[cursor − trail_s, cursor]` (`trail_s` default set in `docs/PROTOCOL.md`) and the uncertainty ellipse or ellipsoid at the cursor sample; field panes MUST share colormap range between terminal raster and viewport and show range and unit in the header; overlays follow CIW-DATA-011.

`CIW-VIEW-018` (extension, M3) An `xy` phase-plot representation (family `spatial2d`; one channel against another over the window, time as colour) MUST be available for any two channels whose `representations[]` include `temporal` and whose time bases are compatible; the viewport's phase portrait is its first instance.

### 10.8 Viewport

`CIW-VIEW-019` (v1) The viewport MUST attach as an ordinary client, render only records delivered by the service, send cursor, channel, and interval changes as `selection.update` with the observed revision, show STALE and disable shared interaction on disconnect, and compute nothing that produces a record; playback MUST move only the cursor; closing the viewport MUST NOT stop the service; camera, axis scaling, and the declared visual transform affect presentation only.

`CIW-VIEW-020` (extension, M4) The viewport MUST display the revision it shows and indicate when it is more than one revision, or more than the lag threshold set in `docs/PROTOCOL.md`, behind; MUST receive mesh topology and point positions as resources by id once and per-frame updates by reference; MUST render at most at its display rate, accept conflated data frames, and never skip selection events.

### 10.9 Same records and exports

`CIW-VIEW-021` (v1) For any record shown in both frontends, both MUST identify the same `run_id`, `evidence_id`, `sample_index`, and resolved `time_s` at the cursor, and the same `result_id` and `execution_id` for a result; the conformance suite MUST verify this headless per reference instrument.

`CIW-VIEW-022` (extension, M2) Exporting a pane to an image or text MUST use the same `render` call and snapshot as the screen.

## 11. Session, persistence, and reproducibility

### 11.1 Workspace v1

`CIW-SESS-001` (v1) A workspace MUST be a JSON file `{workspace_version: 1, saved_at, run, selection, results[], view_settings{}}` holding the complete run, the selection at save time, every immutable result, and reserved `view_settings`. Results MUST be persisted as `result-<id>.json` when created and the source recording as `recording-<content hash>.json`, both in the output directory (CLI default `results/`) beside `workspace.json`; `workspace.save` MUST write atomically (temporary file, fsync, rename); every file MUST be readable without the service.

```json
{
  "workspace_version": 1,
  "saved_at": "2026-09-20T10:20:41.005000+00:00",
  "run": {"run_id": "run-damped-oscillator-demo-v1", "evidence_id": "sha256:6b1f…c2", "instrument": "analytic-damped-oscillator.v1",
          "metadata": {"duration_s": 12.0, "sample_rate_hz": 64.0, "sample_count": 768, "coordinate_frame": "oscillator-state",
                       "model": {"equation": "q'' + 2*gamma*q' + omega_0^2*q = 0", "…": "…"},
                       "provenance": {"generator": "ciw.instruments.make_demo_run", "generator_version": 1, "dtype": "float64",
                                      "time_reference": "seconds since run start", "sampling": "uniform; endpoint excluded"}},
          "time_s": [0.0, 0.015625, "…"],
          "channels": {"q": {"unit": "m", "values": ["…"]}, "v": {"unit": "m/s", "values": ["…"]}, "energy": {"unit": "J", "values": ["…"]}},
          "render": {"coordinate_frame": "oscillator-state", "axis_labels": ["q (m)", "energy (J)", "v (m/s)"], "trajectory": [["…"]],
                     "sample_indices": [0, 1, "…"], "surface": {"vertices": [["…"]], "indices": ["…"]},
                     "transform": {"origin": [0, 0, 0], "scale": [2.5, 0.15, 0.5], "note": "Visual display scaling only; source coordinates and units stay physical."}}},
  "selection": {"run_id": "run-damped-oscillator-demo-v1", "channel": "q", "interval_s": [2.0, 8.0], "cursor_s": 3.0, "coordinate_frame": "oscillator-state", "revision": 4},
  "results": [{"result_id": "result-3f9c…", "operation_id": "statistics.v1", "execution_id": "execution-9d0e…", "…": "…"}],
  "view_settings": {}
}
```

### 11.2 Reopen versus recompute

`CIW-SESS-002` (v1) Reopening MUST validate the entire workspace before any write and restore run, selection, and results without invoking any operation; a new `session_id` MUST be assigned while run, evidence, result, and execution identities stay intact. Saved results MUST be checked against the run (`run_id`, `evidence_id`, `recording_file`, channel and interval validity, `sample_count` equal to the retained samples in the interval, `not_verified` with `verification_id: null`, no duplicate ids) and per operation for internal consistency of `data` without recomputation: `statistics.v1` exactly the fields of CIW-INST-004, the channel's `unit`, `minimum ≤ mean ≤ maximum` (1e-12 relative tolerance), `rms ≥ 0`; `spectrum.periodogram.v1` the fixed v1 `method`, `window`, `detrend`, `scaling`, `sample_count ≥ 4`, the run's `sample_rate_hz`, `unit` `(<unit>)^2/Hz`, arrays of length `sample_count // 2 + 1` on the grid `frequency_hz[i] = i · sample_rate_hz / sample_count`, non-negative `psd`, `peak_frequency_hz` at the maximum (`null` when the PSD is all zero). A result failing any check MUST be rejected. From M1 the run check applies to results with non-null `run_binding`; a result with `source.kind: "bundle"` MUST instead be checked by re-reading the bundle, verifying its `bundle_digest` and the digest relations of CIW-INST-021 without recomputation, and confirming that the envelope identities equal the bundle's.

`CIW-SESS-003` (v1) Reopening a stored result (`result.get`) and recomputing it (`analysis.*`) MUST be distinct actions, distinguishable in every view by execution identity and `created_at`; recomputing MUST create a new `execution_id` and `result_id` even when operation, evidence, channel, and interval are unchanged.

`CIW-SESS-004` (v1 workspace; result files M1) A workspace file MUST carry `workspace_version`; from M1 a result file MUST carry `envelope_version` (CIW-DATA-013). A reader MUST refuse a newer major and accept the older ones it lists as supported. In v1 as built a result file carries no version field and is validated through the workspace that embeds it.

### 11.3 Journal and session store

`CIW-SESS-005` (extension, M2) A session directory MUST have this layout, every file readable without the service:

```
<session>/
  workspace.json            # workspace v1 fields plus format_version, session ids, time bases, coordinate frames, layout refs
  journal.jsonl             # sequence-numbered events, one per line; rotation size set in docs/PROTOCOL.md
  recordings/recording-<hash>.json            # v1 runs; from M4 binary runs as chunks + descriptor
  results/<result_id>.json                    # immutable results (envelopes and small data)
  bundles/run-<result uuid>.json              # attached run bundles (CIW-SESS-014), copied unchanged
  chunks/<channel_id>/<chunk_seq>.npy + .desc.json   # sealed bulk buffers (raw NumPy layout, CIW-DATA-016)
  tables/<result_id>.arrow                    # table payloads (Arrow IPC)
  resources/<resource_id>.bin + .desc.json    # mesh topology, point sets
  manifests/<instrument_id>@<version>.json
  layouts/<layout_id>.json
  exports/                                    # exports and their sidecars
```

A reader MUST also open a v1 flat output directory (`workspace.json`, `recording-<hash>.json`, `result-<id>.json`, from M1 `run-<result uuid>.json`, side by side); M2 moves the flat files into `recordings/` and `results/`. `workspace.json` carries `time_bases[]` (CIW-DATA-005) and `coordinate_frames[]` (CIW-DATA-011) declarations.

`CIW-SESS-006` (extension, M2) The journal MUST be append-only; each entry MUST carry `seq` (contiguous from 1), `t_mono` (session clock ns), `t_wall` (RFC 3339 with offset), `origin` (`tui`, `cli`, `script:<path>:<line>`, `client:<id>`, `viewport:<id>`, `system`), and either `cmd` with its structured `result` or `event` (selection change with revision, result created, instrument transition, error, frame range per flush). No event MAY be published before it is in the journal buffer; fsync MUST occur at the `durability: normal` cadence set in `docs/PROTOCOL.md`, or before publishing under `strict`. `session compact` MAY keep the last selection per second while preserving every non-selection entry.

```json
{"seq":41,"t_mono":12844301000,"t_wall":"2026-09-20T10:14:11.284+00:00","origin":"tui","cmd":"cursor 3.5s","result":{"ok":true,"value":{"cursor_s":3.5,"revision":5}}}
{"seq":42,"t_mono":12901220000,"t_wall":"2026-09-20T10:14:11.341+00:00","origin":"system","event":{"type":"result.created","result_id":"result-3f9c…","operation_id":"statistics.v1","execution_id":"execution-9d0e…","selection_revision":5}}
```

### 11.4 Recording and replay

`CIW-SESS-007` (extension, M5) Recording is not a mode: everything the service accepts MUST be recorded, ingested frames losslessly through the credit path, sealed chunks content-hashed with the hash journaled. A `retain` policy (`all` default, `sources`, `window:<duration>`) governs space, never correctness: a sealed chunk MUST NOT be deleted while any execution input cursor or client subscription position lies at or before it, so retention is bounded by the slowest reader; a reader more than `retain.max_lag` (default the `window` duration; unbounded under `all`) behind the newest sealed chunk MUST have its input closed with `truncated_at {frame_seq, reason: retention}` and the event journaled; under `sources` the same rule applies to derived chunks.

`CIW-SESS-008` (extension, M2) Reopening a session MUST replay the journal to reconstruct selection, results, and channel index without re-executing any instrument; the reconstructed selection at any sequence number MUST equal the one published. `replay --rate` MUST re-publish events and frames at recorded intervals scaled by the rate, pausable and seekable by sequence number and time; nondeterministic instruments MUST replay from recordings, never re-execute.

### 11.5 Deterministic re-execution

`CIW-SESS-009` (extension, M2 operations; M3 instruments) `ciw verify <session>` MUST (1) leave nondeterministic instruments unstarted and use their recordings; (2) re-execute deterministic and seeded operations and instruments in topological order of provenance, in batch mode, from recorded inputs, parameters, and seeds; (3) compare each new result to the recorded one by digest (bitwise, no tolerance) or element-wise within `tolerance`, NaN equal to NaN; (4) report a mismatch as `NONREPRODUCIBLE` with the first differing `(frame_seq, row)` and a drift report `{channel, execution_old, execution_new, max_abs_diff}`; (5) write its outputs as new results in a new session referencing the original by hash. Exit codes follow CIW-OPS-010. A mismatch under a declared `tolerance` is always `NONREPRODUCIBLE` (exit 3); a mismatch on a bitwise-declared instrument is `DRIFT` (drift record, exit 0) only when the recorded and current `platform` records differ, otherwise `NONREPRODUCIBLE`; `--strict` makes every mismatch exit 3.

### 11.6 Export

`CIW-SESS-010` (extension, M2; gltf and png M4) Exports MUST include from M2 `json` (result and workspace records in stored form, one file each), `csv` (one file per time base; columns in declared or preferred units; `σ` columns; units in a header row), `arrow` (Arrow IPC per result), and `npz`, and from M4 `gltf` (meshes, trajectories, point clouds) and `png`/`txt` (pane rasters per CIW-VIEW-022). Every export MUST write `<name>.export.json` with the command, journal sequence, selection revision, interval, unit preferences applied, and the identities of every exported result; an export in the declared unit MUST reproduce stored values exactly. v1 has no export command; its persisted files (CIW-SESS-001) are readable without the service and are not exports.

`CIW-SESS-011` (v1) Result and workspace files MUST be valid JSON without NaN or infinity; non-finite values MUST be rejected at write and read.

## 12. Operator interaction

### 12.1 The v1 terminal client

`CIW-OPS-001` (v1) The terminal client MUST provide `ciw demo` (deterministic synthetic recording), `ciw analyze stats|spectrum --recording --channel --start --end --output-dir` (headless analysis: recording copy, immutable result, reopenable `workspace.json`), `ciw serve --recording|--workspace|--resume --output-dir --port --bind {127.0.0.1|0.0.0.0}` (CIW-SESS-012, CIW-PERF-012), `ciw send <type> --payload|--payload-file [--url]` (one request; JSON response on stdout; exit 0 on `response`, 2 on `error` or when no response arrives), `ciw watch [--url]` (snapshot and events as JSON lines; exit 0 on a normal close by the service, 2 on an abnormal close), `ciw health [--url]` (CIW-SESS-013), `ciw inspect <path>` (print a saved file, executing nothing), and `ciw plsr import|evaluate|inspect|replay` (8.13; optional `plsr` extra; exit 0 when the command succeeds even if the record holds a refusal or violation, 2 on input, configuration, or execution failure, 3 on replay digest mismatch); `--url` defaults to `ws://127.0.0.1:8765`. Every operator action MUST be expressible as one of these or, from M1, as a 12.2 command. The v1 client prints JSON and event lines only; panels and rasters are extensions (10.3, 10.7).

### 12.2 Command model

`CIW-OPS-002` (extension, M1) The grammar MUST be `verb [object] [--flag value]…` with `"` quoting, `#` comments, and time literals `<n>s|ms|us|ns`, ISO 8601 with `@`, `now`, `end`, `sel.start`, `sel.end`; verbs MUST map to the operation loop (attach and run verbs to the request types of CIW-INST-022, select verbs to `selection.update` and `view.update`); a verb tagged M5 MUST be absent from `help --json` and rejected with `unsupported` before M5.

| Loop step | Verbs | Examples |
|---|---|---|
| attach | `attach`, `detach`, `inst` | `attach org.ciw.ref.attitude-ekf --input gyro=imu.gyro --input accel=imu.accel` |
| select | `channel`, `interval`, `cursor`, `window`, `unit`, `coordframe` | `interval 2s..8s`, `cursor 3.5s`, `unit angle deg` |
| run | `run`, `set`, `cancel` (M5) | `run spectrum`, `run attitude-ekf --gyro_noise_density 2e-4`, `cancel execution-9d0e…` |
| inspect | `view`, `layout`, `readout`, `sample` | `view temporal q,v --pane p1`, `sample 3.0s` |
| compare | `compare`, `residual`, `assert` | `compare result-3f9c… result-a1b2…`, `residual att.quat ref.quat` |
| save / replay | `save`, `open`, `replay`, `verify`, `export` | `save`, `export csv q --interval sel`, `verify` |

`CIW-OPS-003` (v1 envelope; M1 form) Every command MUST produce exactly one structured result, `{id, ok: true, value}` or `{id, ok: false, error: {code, message, detail}}`: the command line renders it, headless mode writes it as one JSON line, the gateway returns it as the response. Side effects MUST complete before the result unless the command is documented as asynchronous, in which case `value` MUST carry an `execution_id`. In v1 `ciw send` and `ciw analyze` emit the protocol envelope as this result; other subcommands print a command-specific JSON object on success (`ciw serve` prints two text banner lines) and `ciw: <message>` on stderr with exit 2 on failure; the uniform form is M1.

`CIW-OPS-004` (extension, M2) Commands MUST be serialized through the sequencer of CIW-SYNC-008, each receiving its journal sequence number before it executes.

`CIW-OPS-005` (extension, M1) Every verb and flag MUST be introspectable with `help --json [verb]` (positional parameters and flags with types and defaults, one-line description); completion and generated documentation MUST derive from this introspection and the manifests actually loaded.

### 12.3 Command line, keybindings, layouts

`CIW-OPS-006` (extension, M1) Panels MUST provide a command line entered with `:`, with line editing, per-session history in the journal, completion per CIW-OPS-005, and a scrollable result pane; it is a front end, not a privileged one.

`CIW-OPS-007` (extension, M1) Keybindings MUST be a table from key chords to command strings, rebindable in `~/.config/ciw/keys.json` and per workspace; a keypress MUST resolve to a command string, never a private code path; mouse input, when available, MUST translate to the same commands and MUST never be required.

| Key | Command | Key | Command |
|---|---|---|---|
| `h` / `l`, `←` / `→` | `cursor -1 sample` / `+1 sample` | `H` / `L` | `cursor -10%` / `+10%` of window |
| `[` / `]` | `interval start=cursor` / `end=cursor` | `Esc` | `interval full` |
| `+` / `-` | `window zoom 0.5` / `zoom 2` | `0` | `window fit` |
| `f` | `follow toggle` | `Space` | `play toggle` (cursor only) |
| `Tab` / `S-Tab` | `focus next` / `prev` | `s` / `S` | `split h` / `split v` |
| `u` | `unit cycle` for the focused quantity | `c` | `channel next` |
| `1`–`4` | focus numerical / temporal / spectral / spatial | `v` | `viewport open` (no-op headless) |
| `r` | `run` the focused operation | `x` | `cancel` the focused execution (M5) |
| `:` | command line | `?` | `help --keys` |

`CIW-OPS-008` (extension, M1) A layout MUST be a binary tiling tree of panes with split ratios in `(0, 1)`, serialized in `view_settings.layout`, restorable by name, reflowing to the terminal size without dropping panes, and MUST carry a status line (revision, cursor time, watermark lag, follow mode, tier, skipped ticks, instrument states).

### 12.4 Scripting, assertions, headless and CI use

`CIW-OPS-009` (extension, M1) A script MUST be commands one per line, run by `ciw run <script> --session <dir> --headless` exactly as if typed, `let` variables substituted before tokenization, a failing command aborting unless prefixed with `-` or `--continue` is given, and MUST run without a TTY.

`CIW-OPS-010` (v1 `send`, `plsr`; M1 full) Headless commands MUST exit 0 on success (for the headless instrument path also when the stored record holds a refusal or violation), 1 if any `assert` failed, 2 on a command, protocol, input, or configuration error, 3 on a verification, reproducibility, or conformance failure (`ciw verify` nonreproducible; `ciw plsr replay` digest mismatch), 4 on instrument failure, 5 on storage error, and MUST write results as JSON lines to stdout or `--results <path>`. In v1 the headless instrument path reports instrument execution and persistence failures as 2; codes 4 and 5 apply from M1.

`CIW-OPS-011` (extension, M1) `assert` MUST evaluate a named statistic over a channel and interval against a bound, at least `min`, `max`, `mean`, `std`, `rms`, `count` (`sample_count` of `statistics.v1`), `rms-error --against <channel>`, `max-abs-error --against <channel>`, and `coverage --sigma <k>` (fraction of reference samples within `k·σ` of the estimate), and MUST accept an expression over values at the cursor, aggregates, watermarks, and instrument states (`assert abs(att.euler.roll@cursor - 0.1) < 0.01`); its result MUST contain the computed value.

`CIW-OPS-012` (v1) Attaching, detaching, or closing any client MUST NOT stop, pause, or alter the session; a detached service continues serving and, under streaming, ingesting and recording.

## 13. Extension model

The reusability target, **new instrument = existing workbench + adapter + domain-specific computation and checks**, is a design target tested by the three reference instruments (13.5).

### 13.1 Adding an instrument

`CIW-EXT-001` (extension, M1) A new instrument MUST be addable by a manifest and an adapter, registered by a Python entry point in group `ciw.instruments` (in-process) or by manifest path (subprocess, remote), with no change to service code. The adapter's obligations are exactly: emit valid envelopes and frames, send heartbeats, declare determinism truthfully, preserve the wrapped instrument's headless calculation and tolerances, and from M5 honour credits and acknowledge cancel. Discovery MUST search, in order, `$CIW_INSTRUMENT_PATH` (colon-separated manifest directories), the session's working directory, the user data directory, then the built-in registrations (`src/ciw/instruments/`); CI finds the reference instruments under `examples/<instrument>/instrument.json` through `$CIW_INSTRUMENT_PATH`.

`CIW-EXT-009` (v1) An adapter to an external instrument MUST call the upstream engine and MUST NOT vendor or re-implement its numerical computation. Provenance MUST record the runtime identity (repository, commit or source digest, package version), the execution environment (interpreter and numerical-dependency versions), and the adapter version; the installed runtime MUST be verified against a pinned source manifest and a different source MUST be rejected. Headless bundles (CIW-INST-021) carry this identity in `runtime`; session results carry it in `metadata.provenance` (CIW-DATA-019).

`CIW-EXT-010` (v1) An instrument MUST be listed as integrated in [`docs/INSTRUMENTS.md`](INSTRUMENTS.md) only after its workbench entry point, saved evidence, and replay path have been exercised together. Each entry MUST record source and pin, environment, setup and commands, input ordering, units, and time conventions, output and status semantics, saved evidence and replay, verification state, limits, and links to validation evidence, and MUST be updated in the change that delivers or changes the connection. Standalone readiness (upstream tests, a published schema, an installation, a standalone example) is not workbench integration.

```python
from ciw.sdk import Instrument, Envelope, Frame, run_main

class AttitudeEKF(Instrument):
    manifest = "instrument.json"

    def run(self, ctx, params, inputs):
        env = Envelope(output="state", kind="estimate", status="partial")
        ctx.emit(env)
        for gyro, accel in ctx.iter_inputs(inputs, align="gyro"):
            t, q, cov, bias, bias_sigma = self.step(gyro, accel, params)   # existing NumPy code, unchanged
            ctx.emit(Frame(env, time=t, values={"att.quat": q, "att.bias": bias}, uncertainty={"att.quat": cov, "att.bias": bias_sigma}))
            if ctx.cancelled:
                break
        ctx.end(env, status="cancelled" if ctx.cancelled else "complete")

run_main(AttitudeEKF)   # attach, credits, heartbeat, cancel, digests handled by the SDK
```

### 13.2 Adding a representation

`CIW-EXT-002` (extension, M3) A representation MUST be a client-side plugin implementing CIW-VIEW-004, registered under a family name with its `describe()` metadata and accepted kinds, dtypes, and shapes; it MUST ship a tier-0 rendering, golden-raster tests at two sizes and two backends, and a `hit_test` test; it MUST consume only records, results, and the selection and MUST NOT require service changes beyond, at most, a new reducer registered under `ciw.reducers` with declared provenance fields.

### 13.3 Adding an export

`CIW-EXT-003` (extension, M2) An exporter MUST be a service plugin under `ciw.exports` receiving results, envelopes, and buffers at a snapshot, writing the sidecar of CIW-SESS-010, never mutating results or the selection.

### 13.4 Adapter to an existing evidence-and-result infrastructure

`CIW-EXT-004` (extension, M2) Integration with an existing evidence-and-result infrastructure MUST be an adapter under `ciw.resolvers` that maps `evidence_id`, `operation_id`, `execution_id`, `result_id`, and `verification_id` to that infrastructure's records without merging them, resolves `reference` payloads, and replaces local file persistence; the service MUST function without it and the local workspace format MUST remain readable.

### 13.5 Reference instruments and the reusability measure

`CIW-EXT-005` (extension, M1–M4) The repository MUST ship three deliberately different reference instruments, each with headless reference tests passing in CI without a display, the same-records check of CIW-VIEW-021, and the reopen-versus-recompute check of CIW-SESS-003:

| Reference instrument | Kind | Outputs | Exercises |
|---|---|---|---|
| (a) `org.ciw.ref.timeseries` (M1): the built-in oscillator (`analytic-damped-oscillator.v1`) behind a manifest and adapter with its calculation unchanged; required for the reusability test, not replaced by PLSR (13.8) | source, `deterministic` | scalar observations, regular sampling | manifest, adapter, numerical/temporal/spectral panels, preserved tolerances, reopen versus recompute |
| (b) `org.ciw.ref.attitude-ekf` (M3) | estimator, `deterministic` with tolerance | `att.quat` (tangent-space covariance, superseding results with lag), `att.bias` (σ), `att.nis` (derived) | estimate styling, uncertainty, revisions and `as_of`, residuals, `coverage --sigma`, verify |
| (c) `org.ciw.ref.field-reconstruct` (M4) | processor, `deterministic` | `grid2d` field observation, `points` estimate with σ, `reduced_of` mesh with `reference` | spatial family in terminal and viewport, binary frames, resources, camera sharing, colormap range |

`CIW-EXT-006` (extension, M1–M4) For each reference instrument the conformance suite MUST report lines changed in `src/ciw/session.py`, `src/ciw/server.py`, and `src/ciw/cli.py` against lines added under the instrument's adapter, manifest, and specialized views; a milestone MUST NOT close if service changes exceeded adapter changes without an accepted decision record.

`CIW-EXT-007` (v1) The architecture MUST remain valid if any of the three layers (engines, terminal client, viewport) is replaced: every contract in Sections 6–12 MUST be implementable from any language and MUST NOT depend on a language runtime, toolkit, or renderer; named byte formats (NumPy-compatible contiguous layout, Arrow IPC) are permitted only as layouts a replacement can read. The in-process binding (CIW-INST-004; `inprocess` row of 8.11) is host-language specific by definition and the only language-bound contract; a replacement host provides its own behind the same manifest. Every extension MUST declare the protocol, envelope, and manifest majors it targets.

### 13.6 Candidate instruments

Proposed uses of the workbench; candidates, not commitments.

| Candidate instrument | Reused from the common layer | Remains specialized |
|---|---|---|
| Fluid-state reconstruction | channel inspection, time selection, residual plots, scenario comparison, component selection | flow model, observation equations, boundary conditions, validation cases |
| BIM/construction-state estimation | entity selection, observation history, result inspection, spatial overlays, saved investigations | building semantics, admissibility constraints, interpretation of evidence |
| Polymer/process experiments | batch comparison, temperature/pressure traces, parameter fitting, numerical exports, optional geometry | material models, experimental protocols, calibration, process-specific interpretation |
| GNSS/odometry instrumentation | timestamped trajectories, coordinate-frame inspection, uncertainty displays, replay | positioning algorithms, reference systems, correction handling, receiver integration |
| Dynamical-system / observer experiments | parameter controls, numerical states, phase portraits, function surfaces, comparative runs; first external adapter: PLSR (13.8) | dynamics, estimators, stability conditions, verification procedures |

### 13.7 Adding an operation

`CIW-EXT-008` (extension, M1) An operation (`statistics.v1`, `spectrum.periodogram.v1`, and the `spectrum.welch.v1` and `spectrum.stft.v1` operations of CIW-VIEW-015) MUST be supplied by instrument code: in v1 the functions of CIW-INST-004; from M1 an instrument registered per CIW-EXT-001 whose manifest declares a `batch` output naming the `operation_id`. The Operation runner MUST only sequence the execution, pass the explicit interval and parameters, wrap `data` in the envelope, assign identities, and persist; it MUST NOT contain numerical code. Adding an operation MUST NOT change `src/ciw/session.py` beyond registration.

### 13.8 First external adapter: PLSR (v1, headless)

The Parameterized Lyapunov Stability Runtime (PLSR) evaluates quadratic Lyapunov certificates for declared linear and affine-parameter models. It is the first external adapter, in the dynamical-system/observer candidate family, decided in [ADR-0004](adr/0004-plsr-terminal-adapter.md) and documented in [`docs/PLSR.md`](PLSR.md) and [`docs/INSTRUMENTS.md`](INSTRUMENTS.md). v1 facts: commands `ciw plsr import | evaluate | inspect | replay` (exit codes per CIW-OPS-001); optional `plsr` extra pinned to an upstream commit, Python 3.12+ (the base workbench stays 3.11); adapter `ciw-plsr-adapter-v1`; operation `plsr.verdict.v1`; input `plsr-sample-v1` (explicit `x`, `theta`, `theta_dot`, no defaults); saved run `ciw-plsr-run-v1`, written atomically as `run-<result uuid>.json` (the UUID suffix of `result_id`) and never overwritten with different content, embedding the complete model declaration, the sample, the runtime companion record, and the runtime identity; `evidence_id` a digest over the model `artifact_digest` and the sample; model `artifact_digest`, companion `record_digest`, and CIW `bundle_digest` validated against one another; `verification_status: "not_verified"`, `may_authorize: false`, physical validation `not_started`, `proof_status: NOT_CHECKED`; one explicit sample per invocation; no shared session, no viewport, no change to protocol v1; numerical views only.

M1 items delivered at M0: the existing-instrument adapter with the upstream engine and its status semantics preserved (CIW-EXT-009), runtime pin in provenance, separate identities, inspect versus replay (CIW-INST-021), catalogue entry (CIW-EXT-010). Remaining: manifest and schema (CIW-INST-005), session attachment of bundles (CIW-SESS-014), capability-driven views (CIW-VIEW-001), the subprocess binding (CIW-INST-018), and the reference instrument (a) adapter with its reusability report (CIW-EXT-005, CIW-EXT-006).

## 14. Reference implementation stack

### 14.1 Primary stack

| Layer | Choice | Rationale |
|---|---|---|
| Session service and instruments | Python 3.11+ with NumPy (SciPy where needed); the `plsr` extra requires 3.12+; `asyncio` and `websockets` for the gateway; `multiprocessing.shared_memory` for the local bulk plane (extension) | The engines exist as NumPy code; NumPy kernels and socket I/O release the GIL; fastest validation of the contracts and the reusability target. |
| Terminal client | v1: the scriptable `ciw` CLI (JSON and event lines); extension: **Textual** (on Rich) panels on the same protocol | Asyncio-native, so protocol and rendering share one loop; compositor with damage tracking and a widget tree matching the pane tree; Rich handles wide characters and colour depths; the headless `Pilot` driver runs panels in CI; graphics images placed after the compositor flush, half-block as the per-pane fallback (R2). |
| Viewport | **Godot 4** (4.5.2 pinned; GDScript; GL Compatibility) over the v1 loopback WebSocket | GPU rendering, mesh and point primitives, cross-platform, scriptable; `WebSocketPeer` and JSON built in, so no addons or scientific packages. |
| Client transport | Loopback WebSocket `ws://127.0.0.1:8765`, text JSON (v1); binary frames on the same connection (M4); Unix socket and TCP with token or TLS (M5) | One transport for both clients; forwardable over SSH; could serve a later browser client (2.4) once authentication exists, v1 rejecting browser origins (CIW-PERF-012); head-of-line delay bounded by the client's credit (CIW-DATA-016). |
| Bulk format | Raw NumPy-compatible buffers with an explicit descriptor (default); Arrow IPC via `pyarrow` for tables and export (Parquet optional) | The viewport has no Arrow reader; a descriptor plus contiguous bytes is trivial from GDScript; Arrow's value is in tables and interchange. |
| Manifest and persistence | JSON manifests, workspace, and results; `.npy` chunks with JSON descriptors, Arrow tables, and a JSONL journal in the session store | One serialization across protocol, workspace, and manifest; readable without the service; diffable; no database. |

### 14.2 Alternatives

| Alternative | Trade-off | Status |
|---|---|---|
| **Rust host** (tokio, ratatui + crossterm, ratatui-image, arrow-rs, rustfft, blake3, wgpu viewer) | Lower latency (a 16 ms key-to-draw budget is credible only here), deterministic memory, one static binary, no GIL; slower path to the first instrument, and the engines are NumPy. | Documented native path entering through the same language-neutral contracts; evaluated at M6 against the M5 benchmarks. |
| `prompt_toolkit` or `urwid` | Lighter; no compositor, reflow, or headless driver. | Not chosen. |
| Go with Bubble Tea; C++20 with notcurses; Zig | Static binaries and, for notcurses, the best terminal graphics; thinner numerics, weaker Arrow or graphics ecosystems, memory-safety burden. | Not chosen; notcurses' blitters inform 10.3. |
| Arrow IPC as the default bulk format | Better schema evolution; needs an Arrow reader in the viewport and adds framing overhead for small live frames. | Kept for tables and export. |
| Browser frontend as primary; viewer embedded in the terminal process | Moves the control surface off the terminal; couples it to a GPU process. | Rejected (ADR-0003). |

## 15. Repository layout

The tree is reproduced unchanged from the development guide, whose annotations it carries ('This guide' names `docs/DEVELOPMENT.md`):

```
.
├── README.md
├── AGENTS.md                 Instructions for automated contributors
├── pyproject.toml            Python package metadata; installs the `ciw` command
├── compose.yaml              Container backend definition for Docker Compose
├── Dockerfile                Container image for the backend service
├── src/ciw/                  Runtime service, instruments, and terminal client (Python)
│   ├── instruments.py        Scientific records and computations of the first instrument
│   ├── session.py            Authoritative session: selection, immutable results, workspaces
│   ├── server.py             WebSocket transport, bind policy, and saved shutdown
│   └── cli.py                Headless analysis, service control, health probe, terminal access
├── godot/                    Godot project for the 2D/3D viewport, a client of the session
├── deploy/                   Deployment guides: native controller and container backend
├── examples/                 Reference inputs for integrated instruments, such as `examples/plsr/`
├── scripts/                  Integration checks and the native deployment controller
├── tests/                    Python test suites; conformance tests go under `tests/conformance/`
└── docs/
    ├── ARCHITECTURE.md       Architecture and normative requirements
    ├── PROTOCOL.md           Wire-level protocol, owned by the prototype track
    ├── INSTRUMENTS.md        Catalogue of integrated instruments with their specifications
    ├── quickstart.md         Running the prototype from source
    ├── coordination.md       Build coordination and integration sequence
    ├── DEVELOPMENT.md        This guide
    └── adr/                  Architecture Decision Records
```

`recordings/`, `results/`, and the native deployment's `.ciw/` data directory hold local outputs and are ignored by git. Planned growth follows the same tree: `src/ciw/instruments.py` becomes the package `src/ciw/instruments/` when a second instrument lands; terminal panels go under `src/ciw/tui/`; protocol codecs for binary transport go under `src/ciw/protocol/`; reference instruments and their fixtures used by conformance tests go under `examples/`.

| Component | Directory |
|---|---|
| Record holder, in-process instrument API, operations | `src/ciw/instruments.py` → `src/ciw/instruments/` |
| Selection sequencer, result ledger, envelope validation, persistence, workspace reopen | `src/ciw/session.py` |
| Gateway, bind policy, saved shutdown (v1); Instrument Supervisor, reducer, recorder, journal (extensions) | `src/ciw/server.py` and new modules under `src/ciw/` |
| Terminal client: CLI, `serve` and `health`, headless instrument path, command grammar, scripts, `assert` | `src/ciw/cli.py` |
| Headless instrument adapter (PLSR): commands, run bundles, runtime pin | `src/ciw/plsr.py`, `src/ciw/plsr_engine.py` |
| Terminal panels, tiers, rasterizer, keybindings, layouts | `src/ciw/tui/` |
| Frame descriptors, binary codecs, manifest schema, `ciw-proto-check` | `src/ciw/protocol/` |
| Viewport client, scenes, smoke test, capture | `godot/` |
| Godot bridge check, container lifecycle and installed-package checks, native controller (`workbench.ps1`), benchmark runners | `scripts/` |
| Unit, integration, and conformance tests (one module per area, named by identifier) | `tests/`, `tests/conformance/` |
| Reference instruments (a), (b), (c) with manifests and headless tests; PLSR example artifacts and samples (v1) | `examples/` (`examples/plsr/` exists) |
| Wire specification, quickstart, instrument catalogue, PLSR guide, decisions | `docs/` |
| Container image and Compose service | `Dockerfile`, `compose.yaml` |
| Deployment guides | `deploy/` |
| Native deployment data directory (recordings, results, `workspace.json`, logs, `service.json`) | `.ciw/` (ignored by git) |

## 16. Quality attributes, budgets, and conformance

### 16.1 Reference conditions

Budgets are measured on a profile: 4 physical cores at 2.5 GHz or better, 16 GiB RAM, NVMe storage, a true-colour terminal at 160×48 cells with four panes (numerical, temporal with four traces, spectral, spatial), and for viewport budgets any GPU with Vulkan 1.1 or GL Compatibility. Benchmarks record the actual machine; a budget is asserted only on a profile run; CI runs compare to a baseline.

### 16.2 Budgets

`CIW-PERF-002` (v1 stated; benchmark M1 terminal, M4 viewport) Cursor propagation from the service's `selection.changed` broadcast to the terminal client's next output line and to the viewport's next frame MUST be ≤ 50 ms at p95 and ≤ 120 ms at p99.9 on loopback for the v1 event-line client and the viewport; for panels under CIW-PERF-003 the p95 budget is one tick period plus the render budget (75 ms at the 20 Hz default) and the p99.9 budget 120 ms; measured over at least 10,000 propagation events with no stalled client (v1 does not meet it while a client is stalled, 5.3); at M4 the measurement MUST run with one bulk subscription active at the default credit.

`CIW-PERF-003` (extension, M1) Panels MUST render on a tick capped at 20 Hz by default (5–30 Hz), MUST skip rather than queue when the previous render has not completed, MUST report skipped ticks, and MUST complete a four-pane render at reference conditions in ≤ 25 ms on the event loop, measured over at least 10,000 ticks.

`CIW-PERF-004` (extension, M5) Ingest MUST sustain 1,000,000 scalar float64 samples per second across local subprocess instruments with recording on (load: four subprocess instruments at 250,000 samples/s each in frames of 4,096 samples; smaller frames are outside the budget) using ≤ 30 % of one core for the ingest and recorder threads (their CPU time over wall time), for 10 minutes without credit starvation; live latency from an instrument's send to the reduced frame at a local client MUST be ≤ 150 ms at p95 under that load.

`CIW-PERF-005` (extension, M5) Resident memory MUST stay within `memory_budget_bytes` (default 2 GiB) plus 256 MiB of interpreter overhead for one hour under CIW-PERF-004; eviction order is mapped chunks least recently subscribed, then reducer cache, then unreferenced resources; unsealed ring data MUST never be evicted. Evicting a mapped chunk only unmaps it; it stays re-readable from disk, and deletion is governed solely by `retain` (CIW-SESS-007).

`CIW-PERF-006` (extension, M1 decimation; M5 Welch) A min/max decimation of 10,000,000 samples to 2,000 columns MUST complete in ≤ 40 ms and a Welch PSD of 1,000,000 samples (segment 1024, Hann, 50 % overlap) in ≤ 100 ms, each on one core, as the median of at least 20 runs after 5 warm-up runs.

`CIW-PERF-007` (v1 partial; M1) The service MUST accept clients within 1 s of `ciw serve`; a subprocess instrument MUST reach `Ready` within 500 ms of `attach` excluding its import time, reported separately.

`CIW-PERF-008` (v1; M2) Reopening a workspace MUST reproduce the selection and every result byte for byte and `make_demo_run()` MUST be deterministic; from M2 journal replay MUST reproduce the selection sequence exactly and re-executing a deterministic reference instrument MUST reproduce its digests within tolerance.

`CIW-PERF-009` (extension, M6) Killing an instrument, a client, or the service at random points under CIW-PERF-004 MUST leave behaviour as in 5.6, verified by an integration check inspecting journal and chunks.

`CIW-PERF-010` (extension, M5) `host` channels MUST publish at 1 Hz and `ciw stats --json` MUST answer within 100 ms.

`CIW-PERF-011` (v1) Every component MUST have a headless driver: the session in-process without a socket (`ciw analyze`, protocol tests), the gateway via a script client (`ciw send`), the viewport headless (`--headless --script res://tests/protocol_smoke.gd`) with a query mode reporting the records it shows (M4), and each reference instrument with headless reference tests.

`CIW-PERF-013` (v1) The package MUST build as a wheel and the CLI MUST run from the installed wheel outside the checkout; the container backend MUST pass the lifecycle check (start, mutate, SIGTERM, restart, identities preserved) in CI.

### 16.3 Verification methods and the prototype column

| Method | Meaning |
|---|---|
| unit | test under `tests/` (from M1 `tests/conformance/`, one module per area, test names carrying the identifier) |
| integration | multi-process check: service plus clients, subprocess or remote instruments, headless scripts, the Godot bridge check, the container lifecycle check, the installed-package check |
| golden | raster comparison against stored text or PNG at fixed size and tier |
| benchmark | (a) CI regression against a recorded baseline on the CI machine class, failing above 20 % regression and never asserting the budget; (b) budget assertion on a named profile runner (self-hosted, or a documented machine whose report is attached to the milestone record), which a PERF row must pass to close a milestone |
| inspection | manual review against a written checklist, recorded in the milestone report |

`Prototype v0.1` is judged from the code and tests on `main`: **satisfied** (implemented and tested), **partial** (implemented in part or untested), **planned** (not implemented).

### 16.4 Conformance table
| Requirement | Verification (what the check asserts) | Milestone | Prototype v0.1 |
|---|---|---|---|
| CIW-DATA-001 | unit: `validate_run` rejects bad lengths, non-uniform time, non-finite values, wrong units; `RUN_METADATA` omits arrays | M0 | satisfied (oscillator schema only) |
| CIW-DATA-002 | unit: mismatched evidence rejected before any write | M0 | satisfied |
| CIW-DATA-003 | unit: descriptor schema, `cursor_policy` default, `derived_from` required (M1), estimate form required (M3); integration: estimate-derived labelled in both frontends | M1 (descriptor), M3 (mandatory form) | planned |
| CIW-DATA-004 | unit: cursor bounded by retained timestamps; interval end may equal duration | M0 | satisfied (bounds) / partial (`time_reference` present in the demo, not validated) |
| CIW-DATA-005 | unit: mapping refusal, clock fit, raw timestamps unchanged, ns conversion and membership, dual-form and precision rejections; integration: two bases show `unmapped` | M5 | planned |
| CIW-DATA-006 | unit: `time_order`, `sampling_grid`; equal timestamps preserved | M5 | planned |
| CIW-DATA-007 | unit: no implicit resample; irregular spectrum refused; σ per method; gap policy; integration: residual provenance | M5 | planned |
| CIW-DATA-008 | unit: UCUM parse, dimension vector, sub-dimensions, affine rule, `unknown` refusal, no silent scale conversion | M0 (v1 units), M1 (grammar) | partial (unit strings validated; no grammar) |
| CIW-DATA-009 | unit: σ, interval, covariance converted; spectral unit derived; integration: declared-unit export equals stored | M1 | partial (spectral unit derived) |
| CIW-DATA-010 | unit: each form; absent never zero; symmetry; sampled Cholesky; tangent-space covariance; companion naming; undeclared column rejected | M3 | planned |
| CIW-DATA-011 | unit: v1 frame names match (M0); frame declaration, overlay refusal, missing convention, gaps persisted (M4) | M0 (v1 rule), M4 | satisfied (v1 rule, oscillator schema only) / planned (M4) |
| CIW-DATA-012 | unit: `sample_indices` increasing and in range; transform validity; integration: viewport cards equal `sample.get`; resources and level of detail (M4) | M0 (v1 rule), M4 (resources, level of detail) | satisfied (v1 rule, oscillator schema only) / planned (M4) |
| CIW-DATA-013 | unit: result fields; identities distinct; `not_verified` with null id; envelope v1.1 validation | M0 (v1), M1 (v1.1) | satisfied (v1) / planned (v1.1) |
| CIW-DATA-014 | unit: results immutable; revision and interval captured; computed from retained samples; saved results validated without execution | M0 | satisfied |
| CIW-DATA-015 | unit: each payload type; reference with inline summary; unresolved reference does not fail | M1 | planned |
| CIW-DATA-016 | unit: descriptor complete; byte length equals dtype × shape; integration: subprocess and viewport round trip | M1 (subprocess), M4 (viewport) | planned |
| CIW-DATA-017 | unit: superseding leaves the original intact; `t_valid`/`t_avail`; `as_of` | M3 | planned |
| CIW-DATA-018 | unit: `propagation` declared; `σ-DROPPED` for `dropped`; golden: declared σ shows the band, never the flag | M3 | planned |
| CIW-DATA-019 | unit: version refusal; provenance fields; integration: provenance from the journal | M0 (versions), M1 (converter), M2 (journal) | satisfied (version refusal) / partial (no converter named; provenance fields present in the demo, not validated) / planned (journal) |
| CIW-DATA-020 | unit: property test: reductions preserve extrema and σ extrema and equal a raw reduction, also after supersede | M1 | planned |
| CIW-DATA-021 | unit (`tests/test_plsr.py`): domain refusals saved with exit 0; `NUMERICAL_INCONCLUSIVE` distinct from `NOT_CERTIFIED`; `NUMERICAL_OVERFLOW` retained as a sampleless refusal; malformed samples write nothing | M0 (headless bundles), M1 (session envelope `refused`) | satisfied (PLSR) / planned (session results) |
| CIW-INST-001 | unit: envelope, error, broadcast forms; snapshot on connect; integration: smoke correlates by `request_id` | M0 | satisfied |
| CIW-INST-002 | unit: every command; malformed requests rejected atomically | M0 | satisfied (commands, rejection paths) / partial (`capacity_exceeded`, `storage_error` untested) |
| CIW-INST-003 | unit: unsupported major rejected; M1: `protocol_minor` reported, declared optional field ignored, undeclared field rejected | M0 (rejection), M1 (minor) | satisfied (rejection) / planned (minor) |
| CIW-INST-004 | unit: API functions; validation on every call; periodogram normalization (Parseval, Nyquist bin, zero peak) | M0 | satisfied |
| CIW-INST-005 | unit: manifest schema; refusal on invalid; hash in provenance | M1 | planned |
| CIW-INST-006 | integration: no undeclared family offered | M1 | planned |
| CIW-INST-007 | integration: `ciw-proto-check` drives every state with timeouts and journaled transitions; withheld `finalized` yields Failed with `truncated_at.reason = instrument_failed` (M1), after a `cancel` first (M5) | M1 (batch), M5 (stream, cancel) | planned |
| CIW-INST-008 | integration: bulk never in control records; each family's required fields | M1 (batch), M5 (stream, cancel) | planned |
| CIW-INST-009 | unit: `seq_gap`; envelope precedes first frame | M1 | planned |
| CIW-INST-010 | integration: credit exhaustion halts the producer; recorder priority | M5 | planned |
| CIW-INST-011 | integration: each overflow policy; withheld credit yields gaps; deterministic never drops | M5 | planned |
| CIW-INST-012 | integration: cancel latency; forced termination; partial result status | M5 | planned |
| CIW-INST-013 | unit: error record; integration: each class's action; NaN-aligned sample on numeric error; restart bound | M1 | planned |
| CIW-INST-014 | integration: re-execution per reference instrument; random re-framing; digest equality | M1 (a), M3 (b), M4 (c) | planned |
| CIW-INST-015 | integration: execution versus live scope; new `operation_id` | M3 | planned |
| CIW-INST-016 | integration: stream and batch outputs; batch over an interval records it | M1 (batch), M5 (stream, cancel) | planned |
| CIW-INST-017 | unit: interval passed explicitly; result captures `selection_revision`; no selection access in instrument code | M0 | satisfied |
| CIW-INST-018 | integration: one suite over inprocess, subprocess, `ssh://localhost`; shim wraps an in-process instrument; 10× overrun marks Failed; abandoned-thread cap and `resource_exhausted` without delaying the sequencer | M1 (subprocess), M5 (remote) | planned |
| CIW-INST-019 | unit: version negotiation; each boundary invariant | M1 | planned |
| CIW-INST-020 | inspection: PROTOCOL.md covers every INST requirement; integration: harness passes attach, run, detach on the SDK echo instrument (M1), credit exhaustion and cancel (M5) | M1, M5 | partial (PROTOCOL.md v1 exists; no harness) |
| CIW-INST-021 | unit (`tests/test_plsr.py`): self-contained bundle; `inspect` validates with the pinned runtime's loaders and digest check without invoking the evaluator (requires the `plsr` extra); replay yields new ids with `replay_of` and digest comparison; tampered bundle rejected before write | M0 | satisfied (optional extra; CI `scripts/check_plsr_installed.py`) |
| CIW-INST-022 | unit: payload and result shape per request type; error codes; integration: a script client attaches, runs, and detaches the SDK echo instrument and attaches a bundle (M1); subscribe and credit (M4); cancel (M5) | M1, M4, M5 | planned |
| CIW-INST-023 | integration: a second client receives `result.created` after the result file exists and after the requester's response; payload equals the `result.list` row; bundle broadcasts with `source_kind: bundle`; a missed broadcast is recovered via `result.list` | M1 | planned |
| CIW-SYNC-001 | unit: only the service mutates; integration: no divergent copy after conflict | M0 | satisfied |
| CIW-SYNC-002 | unit: selection fields; initial selection | M0 | satisfied |
| CIW-SYNC-003 | unit: stale, boolean, negative, and missing `expected_revision` rejected without mutation; `expected_revision`-only update rejected; restated update increments `revision` | M0 | satisfied (stale, boolean, missing-field rejection) / partial (`expected_revision`-only rejection and restated-update increment untested) |
| CIW-SYNC-004 | integration: smoke sees the broadcast from a second client; snapshot on connect | M0 | satisfied |
| CIW-SYNC-005 | unit: cursor update leaves interval and results unchanged; integration: interval unchanged during playback | M0 | satisfied |
| CIW-SYNC-006 | integration: Godot smoke (broadcast to a second client, snapshot on connect, reconnect at the same revision); coalescing, one outstanding, obsolete response discarded, 5 s poll, 8 s timeout, STALE | M0 | partial (smoke covers broadcast, snapshot, same-revision reconnect; coalescing, one outstanding, obsolete-response discard, poll, timeout, and STALE implemented in the viewport client but untested; CLI is short-lived) |
| CIW-SYNC-007 | unit: nearest sample, earlier on tie, boundaries; `at_or_before` (M5) | M0, M5 | satisfied (nearest) |
| CIW-SYNC-008 | unit: contiguous seq; integration: gap forces resync; coalescing delivers the final value | M2 | planned |
| CIW-SYNC-009 | unit: field set and defaults; `view.update` revisioned under the selection lock; `view.changed`; snapshot carries view settings; `focus` not broadcast | M1 (†), M3, M4, M5 | planned (reserved `{}`) |
| CIW-SYNC-010 | unit + golden: panes render from the injected snapshot only; frame attributable | M1 | planned |
| CIW-SYNC-011 | unit: arrival leaves the revision unchanged with follow off; ≤ 1 follow mutation per period, origin `system`; explicit cursor or window update sets `follow: off`; integration: two following clients send no updates | M5 | planned |
| CIW-SYNC-012 | unit: quantization from area; state untouched; two widths show the same records | M1 | planned |
| CIW-SYNC-013 | golden: coincident edges; same cursor time in every header | M1 | planned |
| CIW-SYNC-014 | golden: unit change in all panes in one revision | M1 | planned |
| CIW-SYNC-015 | unit: spectrum from the exact interval; explicit interval does not mutate the selection; integration: spectrogram axis; two pane widths yield the same `result_id` (M5) | M0, M5 | satisfied (interval rule) |
| CIW-SYNC-016 | unit: result shows revision and interval; stale marker; no auto-recompute | M0 (identity), M1 (marker) | satisfied (identity) / planned (marker) |
| CIW-SYNC-017 | unit: holds by construction of CIW-SYNC-004 (M0); benchmark: with CIW-PERF-002 | M0 (rule), M1 (terminal), M4 (viewport) | satisfied (rule) / planned (benchmark) |
| CIW-SYNC-018 | integration: camera in view settings; clip-space agreement within 1e-6 | M4 | planned |
| CIW-SYNC-019 | integration: link and unlink axes; one revision per group change | M5 | planned |
| CIW-SYNC-020 | integration: two time bases, one cursor each | M5 | planned |
| CIW-SYNC-021 | integration: a stalled client does not delay another client's accepted update beyond one tick; overflow and stall disconnect only that client | M1 | planned (v1 awaits delivery and the close handshake under the lock) |
| CIW-SYNC-022 | unit: duplicate `channel_id` refused with `channel_conflict`; bare ids resolve for run and result channels; with two executions a bare id resolves to the latest at acceptance, broadcasts carry the qualified form, earlier revisions unchanged by a later commit | M1 | planned |
| CIW-VIEW-001 | integration: no undeclared panel offered | M1 | planned |
| CIW-VIEW-002 | integration: viewport cards equal `sample.get`; code inspection for client computation | M0 | satisfied |
| CIW-VIEW-003 | golden: styling per family; uncertainty rendered | M1, M3 | planned |
| CIW-VIEW-004 | unit: interface conformance per representation; repeat render byte-identical | M1 | planned |
| CIW-VIEW-005 | golden: tier 0 per family; inspection over SSH at 80×24 `xterm-256color` | M1 | planned |
| CIW-VIEW-006 | unit: probe ladder with mocked responses (a mocked 300 ms round trip still confirms tier 3); override; iTerm2 confirmation and first-write fallback; runtime fallback; 256 KiB cap; two-tick rule | M1, M4 | planned |
| CIW-VIEW-007 | unit: render report identical across backends | M1 | planned |
| CIW-VIEW-008 | golden: numerical columns; unit: precision from σ; readout variant | M1 | planned |
| CIW-VIEW-009 | unit: `statistics.v1` fields and bounds; `statistics.v2` adds `std` under a new id (M1); aggregates from results only | M0, M1 | satisfied (statistics) |
| CIW-VIEW-010 | unit: envelope property test; gaps as breaks; interpolation defaults | M1 | planned |
| CIW-VIEW-011 | unit: tick set; header cursor time | M1 | planned |
| CIW-VIEW-012 | golden: superseded marker with injected `t_tick` and `seen` at 0.5 s and 1.5 s; flag per `as_of` | M3 | planned |
| CIW-VIEW-013 | golden: spectral header from result data | M0 (header content), M1 (pane) | partial (CLI prints result JSON) |
| CIW-VIEW-014 | unit: colormap luminance monotone; quantized palettes; thresholding | M5 | planned |
| CIW-VIEW-015 | unit: new operation ids; periodogram unchanged | M5 | planned |
| CIW-VIEW-016 | golden: orthographic preview; integration: rasterizer versus viewport clip space | M1 (preview), M4 | planned |
| CIW-VIEW-017 | golden: trail and ellipse; shared colormap range | M4 | planned |
| CIW-VIEW-018 | golden: `xy` pane; integration: viewport phase portrait shows the same samples | M3 | partial (viewport phase portrait) |
| CIW-VIEW-019 | integration: Godot smoke (renders from delivered records, `selection.update` with the observed revision, playback moves only the cursor); `scripts/check_godot.py` (closing the viewport leaves the service running); STALE on disconnect | M0 | partial (STALE on disconnect implemented but untested) |
| CIW-VIEW-020 | integration: lag indicator under injected delay; resources sent once; conflation under a slow client | M4 | planned |
| CIW-VIEW-021 | integration: same-records check per reference instrument (smoke checks `sample.get` against the retained record) | M0 (demo), M1, M3, M4 | partial (demo: viewport sample identity checked by the Godot smoke, headless-versus-service results by `tests/test_integration.py`; no paired terminal/viewport check; the viewport shows no results) |
| CIW-VIEW-022 | golden: export equals the screen raster | M2 | planned |
| CIW-SESS-001 | unit: workspace fields; atomic write; result files; readable without the service | M0 | satisfied |
| CIW-SESS-002 | unit (`tests/test_replay.py`): invalid workspace rejected before any write; reopen runs no operation; per-operation consistency of saved `data`; identities intact; integration (`scripts/check_container.py`, `scripts/check_local.ps1`): new session id on reopen; run check conditioned on `run_binding` (M1) | M0, M1 | satisfied (v1) / planned (`run_binding`) |
| CIW-SESS-003 | unit: new analysis after reopen has new ids; stored results identical | M0 | satisfied |
| CIW-SESS-004 | unit: unsupported `workspace_version` rejected (M0); result file without `envelope_version` or with a newer major rejected (M1) | M0 (workspace), M1 (result files) | satisfied (workspace) / planned (result files) |
| CIW-SESS-005 | unit: layout validator; integration: readable without the service; v1 flat directory opens | M2 | planned |
| CIW-SESS-006 | unit: entry schema; contiguous seq after crash injection; coalescing; compact preserves non-selection entries | M2 | planned |
| CIW-SESS-007 | integration: recorder backpressure; chunk hash journaled; retain policies; a reader withheld under `window:` retention keeps its chunks or receives a `retention` truncation, never a short result | M5 | planned |
| CIW-SESS-008 | integration: replayed selection equals published; rate, pause, seek; nondeterministic from recordings | M2 | planned |
| CIW-SESS-009 | integration: `verify` detects an injected mismatch with the first differing row; platform-record difference on a bitwise instrument yields DRIFT and exit 0; same-platform and beyond-tolerance mismatches exit 3; `--strict`; new session references the original | M2, M3 | planned |
| CIW-SESS-010 | integration: each format round-trips; sidecar present; declared-unit export exact | M2 (json, csv, arrow, npz), M4 (gltf, png) | planned |
| CIW-SESS-011 | unit: NaN and infinity rejected at write and read | M0 | satisfied |
| CIW-SESS-012 | POSIX: `tests/test_deployment.py` (signal save, restart preserves exact JSON, resume without computation, corrupt workspace fails, save failure propagates), `scripts/check_container.py` (identities after SIGTERM and restart without an explicit save); Windows: `scripts/check_local.ps1` (Stop saves and verifies before terminating the owned process; Start restores under a new session identity); the signal test is skipped on Windows | M0 | satisfied (POSIX signal path and Windows controller path; Windows console SIGINT untested) |
| CIW-SESS-013 | integration (`tests/test_deployment.py`): health creates no result; bounded failure on a silent server; non-WebSocket URL rejected; invalid snapshots rejected | M0 | satisfied |
| CIW-SESS-014 | integration: an attached bundle is listed and returned with unchanged identities and no re-evaluation; copied beside `workspace.json`; restored on reopen, also from a moved output directory, through `bundle_digest` despite a foreign `evidence_id`; `workspace_version: 2` refused by a v1 reader; duplicate attach refused; `null` run-bound summary fields for bundle rows | M1 | planned |
| CIW-OPS-001 | integration: `tests/test_integration.py` (`analyze stats`, `inspect`, `send session.get`), `tests/test_deployment.py` (`serve --resume`, signals, `health` exit 0 and 2, `--bind` choices), `scripts/check_installed.py` (`demo`, `analyze spectrum`, `inspect`), `tests/test_plsr.py` (`plsr`); `inspect` executes nothing; `send` exit 2 on `error`; `watch` event lines and exit 0 on normal close | M0 | partial (`watch` and the `send` error exit untested) |
| CIW-OPS-002 | unit: grammar cases; each verb mapped to a loop step; `cancel` rejected with `unsupported` before M5 (M1) and cancels per CIW-INST-012 (M5) | M1, M5 | planned |
| CIW-OPS-003 | unit: result schema; integration: JSON lines headless | M0 (envelope), M1 | partial (envelope on `send` and `analyze`) |
| CIW-OPS-004 | unit: concurrent submission ordering | M2 | planned |
| CIW-OPS-005 | unit: every verb introspectable; completion derived; reflects loaded manifests | M1 | planned |
| CIW-OPS-006 | inspection: command-line checklist; unit: completion from introspection | M1 | planned |
| CIW-OPS-007 | unit: bind table resolution; integration: bindings persisted; mouse maps to commands | M1 | planned |
| CIW-OPS-008 | unit: layout round trip; reflow; status line present | M1 | planned |
| CIW-OPS-009 | integration: script equals interactive; abort and continue; no TTY | M1 | planned |
| CIW-OPS-010 | integration: exit codes in a no-TTY CI job (`send` 0/2; `plsr` 0/2/3 in `tests/test_plsr.py`) | M0 (`send`, `plsr`), M1 | partial |
| CIW-OPS-011 | unit: each statistic; `coverage --sigma`; expression form; failure report with values | M1 | planned |
| CIW-OPS-012 | integration: detach during an operation; viewport close leaves the service | M0 | satisfied |
| CIW-EXT-001 | integration: reference instruments added with zero service diff; discovery order | M1 | planned |
| CIW-EXT-002 | integration: phase-plot representation plugin | M3 | planned |
| CIW-EXT-003 | integration: json, csv, arrow, npz exporters as plugins without service change, each writing the sidecar (M2); gltf and png (M4) | M2, M4 (gltf, png) | planned |
| CIW-EXT-004 | unit: resolver interface; integration: stub resolver replaces file persistence without merging identities | M2 | planned |
| CIW-EXT-005 | integration: headless reference tests, same-records, reopen versus recompute per instrument | M1, M3, M4 | partial (oscillator in both frontends; PLSR headless tests and inspect versus replay without a viewport) |
| CIW-EXT-006 | inspection: diff report per milestone | M1, M3, M4 | planned |
| CIW-EXT-007 | inspection: every contract in Sections 6–12 other than CIW-INST-004 and the `inprocess` binding row depends on no language runtime, toolkit, or renderer; extensions declare majors; 6.2 states the majors (1 / 1 / 1) every extension row targets | M0 | satisfied |
| CIW-EXT-008 | inspection: no numerical code in the Operation runner; integration: Welch registered without service change | M1 (rule), M5 (Welch) | partial (v1 computations live in `ciw.instruments`; dispatch is hard-coded) |
| CIW-EXT-009 | unit: `tests/test_plsr_engine.py` pin rejects changed, missing, and extra source and a different distribution version; `tests/test_plsr.py` runtime identity retained in the bundle; inspection: `plsr_engine.evaluate` calls the upstream evaluator and `src/ciw/` holds no verdict computation | M0 | satisfied |
| CIW-EXT-010 | inspection: each catalogue entry carries the required items; integration: entry point, saved evidence, and replay exercised together (`tests/test_replay.py`, `scripts/check_installed.py` for the oscillator; `tests/test_plsr.py`, `scripts/check_plsr_installed.py` for PLSR) | M0 | satisfied |
| CIW-PERF-001 | inspection: transport limits stated in PROTOCOL.md; message-size and result-count limits stated in quickstart.md and this document | M0 | satisfied |
| CIW-PERF-002 | inspection: budget stated (M0); benchmark: broadcast to output line (event lines, panels) and viewport frame; with a bulk subscription active at default credit (M4) | M0 (stated), M1 (terminal), M4 (viewport) | satisfied (stated) / planned (benchmark) |
| CIW-PERF-003 | benchmark: tick cap, skip not queue, four-pane render time | M1 | planned |
| CIW-PERF-004 | benchmark: ingest and live latency | M5 | planned |
| CIW-PERF-005 | benchmark: RSS sampling; eviction order | M5 | planned |
| CIW-PERF-006 | benchmark: decimation and Welch on one core | M1, M5 | planned |
| CIW-PERF-007 | benchmark: service ready within 1 s; instrument `Ready` within 500 ms | M0 (service ready), M1 (instrument Ready) | partial (startup untimed) |
| CIW-PERF-008 | unit: demo deterministic; reopen byte-identical; integration: journal replay and digests (M2) | M0, M2 | satisfied (v1) |
| CIW-PERF-009 | integration: fault injection under load | M6 | planned |
| CIW-PERF-010 | integration + benchmark: `host` channels and `ciw stats` | M5 | planned |
| CIW-PERF-011 | integration: headless drivers for session, gateway, viewport, reference instruments | M0, M4 (viewport query) | satisfied (v1 drivers) |
| CIW-PERF-012 | unit: bind choices (`tests/test_deployment.py`); integration: `scripts/check_container.py` fresh Compose project on a temporary port and volume | M0 | satisfied (bind choices, loopback publish) / partial (origin rejection via `origins=[None]`, untested) |
| CIW-PERF-013 | integration: `scripts/check_installed.py` runs the CLI from the built wheel; CI builds the `ciw-python-wheel` artifact and runs the container lifecycle on Linux and the native controller lifecycle (`scripts/check_local.ps1`) on Windows | M0 | satisfied |

## 17. Risks and open questions
| # | Risk or question | Mitigation or decision |
|---|---|---|
| R1 | The GIL limits the Python service under many instruments and panels (CIW-PERF-002/003/004). | Hot paths release the GIL; CI benchmarks from M1; the native path is the escape; decision at M6. |
| R2 | Textual's compositor and graphics-protocol placement conflict. | Place after compositor flush, reuse image ids per pane; half-block fallback per pane (M4). |
| R3 | Terminal graphics differ under multiplexers and SSH. | Detection by response with timeout; `--tier`; tier 0 suffices for the whole loop; terminal matrix at M4. |
| R4 | Loopback WebSocket without authentication: any local user can drive a session. | v1 binds `127.0.0.1` and rejects browser origins; token and TLS at M5; SSH forwarding until then. |
| R5 | JSON-only transport limits run size (1 MiB inbound; the viewport buffers 32 MiB). | Stated as a limit; binary frames at M4; chunked `run.get` as a v2 addition. |
| R6 | Superseding results at high rates stress reductions and caches. | Bounded superseding intervals; per-bucket invalidation; `retain`; benchmark at M3 with `lag_s = 1`. |
| R7 | Floating-point reproducibility across platforms breaks golden tests and `verify`. | Bitwise only on one platform; declared `tolerance`; drift reported, not failed, unless `--strict`; decided at M3. |
| R8 | Clock mapping between bases can present misaligned overlays as fact. | No implicit mapping; declared, versioned mappings with evidence; readouts show Δt. Open: PTP or NTP adapter, M5. |
| R9 | Covariance for large state vectors. | Block-diagonal and packed layouts; diagonals streamed by default; full covariance as `batch`. |
| R10 | Wrong determinism claims silently break reproducibility. | CIW-INST-014 re-execution; a failed check downgrades the declaration in the journal. |
| R11 | Whether the `observed` class needs its own contract profile. | Open; specified with the first `observed` adapter in a decision record before M6. |
| R12 | The envelope's identities may not fit a deployment's evidence infrastructure. | Identities are opaque strings; the adapter maps without merging (M2). |
| R13 | Shared versus per-client camera on high-latency links. | Shared by default with local prediction; revisit after the M4 trial. |
| R14 | Journal growth in long sessions. | Cursor coalescing (CIW-SYNC-008); `session compact`; rotation (CIW-SESS-005). |
| R15 | The PLSR upstream kernel API is marked `changing`. | Source-manifest verification rejects any other source (CIW-EXT-009); a pin update requires a compatibility review and a decision record (ADR-0004). |
| R16 | Autosave covers graceful shutdown only. | `workspace.save` as the explicit checkpoint; controller Stop saves and verifies; 30 s container grace; the M2 journal makes every mutation durable. |

## 18. Roadmap

### 18.1 Readiness stages

- **Standalone-ready**: the instrument passes its own headless reference tests with declared tolerances and ships a valid manifest; no workbench involved.
- **Workbench-ready**: the instrument attaches through the contract with an adapter only; both frontends identify the same records; reopening is distinguished from recomputing; its conformance rows pass; the reusability report shows adapter changes above service changes. An instrument is recorded as integrated in [`docs/INSTRUMENTS.md`](INSTRUMENTS.md) only under CIW-EXT-010; standalone readiness is not workbench integration. The PLSR adapter is integrated for the headless path and not workbench-ready until CIW-SESS-014 and CIW-VIEW-001 hold for it.
- **Deployment-ready**: a stated use runs in its declared service class with measured workloads, precision, throughput, recovery, and device access meeting its budgets; the persistence adapter is in place; environments and the optional viewport are packaged.

### 18.2 Milestones

| Milestone | Delivers | Demonstrable |
|---|---|---|
| **M0 — Prototype v0.1** (done) | Oscillator instrument; headless statistics and periodogram with immutable results and identities; session service over loopback WebSocket; `ciw` CLI; Godot viewport (phase portrait, energy surface, shared cursor, numerical cards); workspace save and reopen; native controller and container backend (5.8) with saved shutdown on SIGINT and SIGTERM, `serve --resume`, and `ciw health`; PLSR headless adapter (13.8) writing self-contained run bundles over a source-pinned engine; CI: Python matrix, Godot bridge check, wheel build with the installed-package check, container lifecycle on Linux, native lifecycle on Windows, PLSR tests and installed check | Two clients change the shared selection and inspect the same retained sample; the interval survives cursor moves; the viewport reconnects without stopping the service; a workspace reopens without recomputing; a service stopped by SIGTERM and restarted with `--resume` restores selection and results without an explicit save; a pinned PLSR model and explicit sample evaluate to a bundle that `inspect` validates without recomputation and `replay` re-evaluates under new identities with a digest comparison; a numerical refusal is stored with exit 0 and `not_verified`. |
| **M1 — Existing-instrument adapter (reference instrument a)** | The items 13.8 lists as remaining (the PLSR adapter delivered the others at M0), the subprocess binding among them in batch mode with lifecycle, message families, ordering, error record, and binary frames on the bulk plane; SDK shim; envelope v1.1 with `result.created`; unit grammar; Textual panels at tiers 0–2 with the spectral pane (CIW-VIEW-013) and the orthographic spatial preview (CIW-VIEW-016); command grammar, `help --json`, keybindings, layouts, scripts, `assert`; reductions; `ciw-proto-check` (attach, run, detach) | The oscillator attaches behind a manifest and adapter with calculation and tolerances preserved; an attached PLSR bundle is listed and reopened with identities intact; over SSH the cursor, interval, and units move every pane of a four-pane layout in step; the reusability report shows adapter changes above service changes. |
| **M2 — Persistence adapter and journal** | Session directory and journal; total command order; resolver adapter mapping the identities to an external infrastructure without merging; journal replay; `verify` for v1 operations; json, csv, arrow, npz exports with sidecars | A session reopens by replaying its journal; the same investigation opens through the stub resolver with identities intact; `verify` reproduces every result and detects an injected mismatch. |
| **M3 — State-estimation instrument (reference instrument b)** | Uncertainty forms and rendering; superseding results, `t_valid`/`t_avail`, `as_of`; residual and comparison views; `xy` plots; execution versus live parameters; `coverage --sigma` | The attitude EKF runs as a subprocess in batch mode over the recorded run; estimates display dashed with bands and `E`; a fixed-lag revision is scrubbable with `as_of`; two executions with different noise are compared; `verify` matches within tolerance. |
| **M4 — Spatial instrument (reference instrument c)** | Binary frames with descriptor on the client link; viewport resources and updates; coordinate-frame tree; shared camera; software rasterizer and graphics tier; `reduced_of` with `reference`; gltf and png exports | A field reconstruction attaches; a viewport pick moves the terminal readouts and a terminal cursor move moves the viewport within budget; raster and viewport agree in clip space; closing the viewport leaves the in-terminal raster. |
| **M5 — Spectral and temporal expansion** | Welch and spectrogram operations; frequency cursor and link groups; streaming with separate acquisition, publish, and render rates; credits, cancellation, heartbeats, gaps; time bases and clock sync; remote bindings; bounded history; `host` channels | A live source streams at 1 M samples/s with recording on; a spectrogram aligns under the trace; a cancelled analysis leaves a `cancelled` result; a stale result is marked and never silently recomputed; a remote instrument feeds the same session. |
| **M6 — Deployment for a stated use** | `observed` adapter; packaging of environments and viewport; measured target workloads; fault-injection suite; native-path evaluation against the M5 benchmarks; decision records for open questions | A stated deployment meets its service-class budgets; recovery and device-access measurements are published; the native-path decision is recorded. |

A milestone closes only when its conformance rows pass and, from M1, CIW-EXT-006 is met. A PERF row closes on a profile run; without a profile runner at the milestone the row is recorded as `inspection (recorded, not gated)` in the milestone report and the milestone may close with that entry.

## 19. Decisions
| # | Decision | Rationale |
|---|---|---|
| D1 | Authoritative session service; terminal and viewport are independent clients; the terminal works alone; closing the viewport never stops the service. | The v1 baseline and the Jupyter positioning; a stalled client holds the sequencer in v1 (5.3) until CIW-SYNC-021 at M1. |
| D2 | Client transport: loopback WebSocket text JSON without authentication in v1; Unix socket, TCP with token or TLS, and SSH forwarding at M5. | Both clients speak it; Godot lacks Unix sockets; remote use is tunnelled until authentication exists. |
| D3 | Bulk encoding: JSON only on the client link in v1; raw NumPy buffers with a descriptor on the instrument bulk plane from M1 and on the client link from M4; Arrow IPC for tables and export. | The viewport has no Arrow reader; Arrow keeps its value for schema evolution and interchange. |
| D4 | Manifest format: JSON `instrument.json`. | One serialization across protocol, workspace, and manifest. |
| D5 | Kinds `{observation, estimate, derived, reference}`; any producer may emit `derived`, always with `derived_from`. | An innovation is neither measured nor a state. |
| D6 | Spectra are operations: instrument code sequenced, wrapped, and persisted by the Operation runner; never computed in a view. | Clients own no calculation; v1 works this way. |
| D7 | Cursor separate from interval; cursor moves never recompute; cursor-anchored spectra only as an explicit command. | v1 invariant; results stay attributable to a revision. |
| D8 | Intervals half-open `[start, end)`; float seconds on the v1 fields; from M5 per-base `int64` ns fields are authoritative, floats converted once, never both in one update (CIW-DATA-005). | Matches code and tests; nanosecond bases matter only with several clocks. |
| D9 | Revisions never mutate a result; a superseding result names what it replaces; `t_valid`/`t_avail`; `as_of` in view settings. | Immutable results are a v1 invariant. |
| D10 | Camera viewport-local in v1; shared in view settings from M4 with local prediction. | Needed for clip-space agreement. |
| D11 | Covariance full row-major `[d, d]` by default; `packed_lower` and `block_diagonal` declared; quaternion states tangent-space `[3, 3]`. | Simplest in NumPy and GDScript; a 4×4 quaternion covariance is singular. |
| D12 | Credits: 64 frames / 64 MiB per instrument stream; 8 frames / 8 MiB per client subscription. | Shared memory tolerates large windows; clients are bounded to keep backlog small. |
| D13 | Timeouts: hello 5 s, ready 30 s, drain 30 s (CIW-INST-007); cancel within 2 × `heartbeat_ms` (CIW-INST-012); `heartbeat_ms` declared in the manifest; three missed heartbeats is `Failed` (5.6). | One set derived from the heartbeat. |
| D14 | Render cadence 20 Hz default (5–30 Hz); cursor propagation ≤ 50 ms p95 for event lines and the viewport, one tick plus 25 ms for panels; 16 ms figures belong to the native path. | Credible for the Python arrangement. |
| D15 | Subprocess binding: length-prefixed JSON on stdio, bulk in shared memory or inline; `ssh://host//path/exec` reuses it. | One binding for local and remote spawn without a daemon. |
| D16 | Restart `manual` by default; `auto` bounded by `restart.max` (3); every restart is a new execution. | Never hides a failure. |
| D17 | Determinism: `deterministic` (optional `tolerance`, bitwise otherwise), `seeded`, `nondeterministic`; frame-boundary invariance for the first two. | Tolerance keeps claims honest. |
| D18 | Recording: v1 persists one run and its results; from M5 the store is the recording, always on, with a `retain` policy. | No silent loss; space is a policy. |
| D19 | In-terminal 3D: orthographic preview mandatory; software rasterizer with clip-space agreement at M4. | The terminal stays capable. |
| D20 | Vocabulary: the prototype's terms win; a run is a recording, never an execution. | Consistency with `docs/PROTOCOL.md` and the code. |
| D21 | Session store: workspace v1 JSON baseline; M2 directory with `.npy` chunks, JSON descriptors, Arrow tables, JSONL journal. | Readable without the service. |
| D22 | Repository layout: the `docs/DEVELOPMENT.md` tree verbatim with its growth rule. | The real tree. |
| D23 | Unit grammar: UCUM with `/`, integer exponents, and `^(p/q)`; the v1 `(unit)^2/Hz` form accepted. | Noise densities need rational exponents. |
| D24 | Operation baseline: `statistics.v1` and `spectrum.periodogram.v1` unchanged; `std` as `statistics.v2` at M1, Welch and spectrogram as new operations at M5. | Operation identities are versioned. |
| D25 | Dimension mismatch refused; scale mismatch never converted silently; explicit `convert` offered and recorded. | Silent conversion hides errors. |
| D26 | Uncertainty on estimates: a declared form mandatory from M3; `none` only with a reason; v1 has none and says so. | "Including uncertainty" requires the declaration. |
| D27 | Cursor resolution: nearest retained sample, earlier on tie (v1); `at_or_before` as a declared policy for live channels. | Matches `inspect_sample`; live follow needs at-or-before. |
| D28 | Time epoch: `time_s` relative seconds in v1; declared bases with versioned mappings at M5; no implicit mapping. | Alignment can improve without rewriting data. |
| D29 | Verification: `verification_id` null and `not_verified` until a distinct verification execution references the result; `checks[]` are self-checks. | Never inferred from a scene. |
| D30 | Vector-shaped channels with component labels, not one channel per component. | Matches `channels{name: {unit, values}}`; covariance stays with its state. |
| D31 | Saved shutdown on SIGINT or SIGTERM (POSIX); the native controller on Windows requests `workspace.save`, verifies it, then terminates the owned process only; no shutdown request in the client protocol (CIW-SESS-012). | A protocol shutdown would let any local client stop a shared session. |
| D32 | External instruments: adapters call the upstream engine under a verified source pin and never vendor it; PLSR is a headless terminal adapter that changes no protocol (ADR-0004). | One numerical implementation to validate. |

Decision records:

| Decision | Record |
|---|---|
| Project name and definition | [ADR-0001](adr/0001-project-name-and-definition.md) |
| Positioning as a foundational instrumentation runtime; contract-first rule; reusability target | [ADR-0002](adr/0002-foundational-runtime-positioning.md) |
| Reference implementation arrangement (Python/NumPy engines, terminal frontend, Godot viewport) and the native alternative | [ADR-0003](adr/0003-reference-implementation-arrangement.md) |
| PLSR integrated through a bounded headless terminal adapter | [ADR-0004](adr/0004-plsr-terminal-adapter.md) |

## 20. Related documents

- [`README.md`](../README.md) — overview and capabilities.
- [`docs/PROTOCOL.md`](PROTOCOL.md) — wire-level protocol for version 1: exact encodings, schemas, handshake; must satisfy every wire-visible requirement of this document (2.3).
- [`docs/quickstart.md`](quickstart.md) — running the prototype from source.
- [`docs/DEVELOPMENT.md`](DEVELOPMENT.md) — development guide.
- [`docs/INSTRUMENTS.md`](INSTRUMENTS.md) — catalogue of integrated instruments: entry point, commands, conventions, evidence and replay, verification state, limits.
- [`docs/PLSR.md`](PLSR.md) — the PLSR headless terminal adapter: commands, run bundle, digests, exit codes.
- [`deploy/README.md`](../deploy/README.md) — deployment paths: native controller and container backend.
- [`deploy/CONTAINER.md`](../deploy/CONTAINER.md) — container backend: Compose service, volume, stop and resume, validation status.
- [`docs/adr/README.md`](adr/README.md) — decision records index.
