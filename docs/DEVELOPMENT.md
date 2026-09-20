# Development guide

This guide is for people working on the Computational Instrumentation Workbench itself. It describes how the work is organized, how documents and code refer to each other, and what to read first.

## Workstreams

Development proceeds in two tracks that hand off through this repository.

| Track | Owns | Delivers |
|---|---|---|
| Documentation | `README.md`, `docs/ARCHITECTURE.md` | The contracts, data model, synchronization semantics, and quality budgets, each stated as a numbered requirement |
| Prototype | `src/ciw/`, `godot/`, `scripts/`, `tests/`, `docs/PROTOCOL.md`, `docs/quickstart.md`, `docs/coordination.md`, continuous integration | An implementation of the contracts, the wire-level protocol specification, and the evidence that the implementation conforms |

The hand-off rule is: the architecture defines contracts; the prototype implements them; tests and integration checks verify conformance against requirement identifiers; decisions raised by either track are recorded in the architecture document's Decisions table (Section 19), which is the published record.

## Requirement identifiers

Normative statements in `docs/ARCHITECTURE.md` carry a stable identifier and use MUST, SHOULD, or MAY wording. The identifier form is `CIW-<AREA>-<NNN>`, for example `CIW-DATA-003` or `CIW-SYNC-007`. Areas are:

| Area | Covers |
|---|---|
| `DATA` | Data model: channels, time bases, units, uncertainty, provenance |
| `INST` | Instrument contract: manifest and roles, adapters and bindings, operations and executions, lifecycle, messages, backpressure, errors |
| `SYNC` | Synchronization: shared cursor, selection, units, ordering, latency |
| `VIEW` | Representation layer: numerical, temporal, spectral, 2D/3D |
| `SESS` | Sessions, persistence (workspace versions 1 and 2), recording, replay, export |
| `OPS` | Operator interaction: commands, keybindings, layouts, scripting, headless use |
| `EXT` | Extension model: adding instruments, adapters, operations, representations, exports |
| `CAL` | Calibration contract and the deterministic bench: profiles, corrected data, fixtures, metrics, bench reports |
| `PERF` | Budgets and quality attributes |

Tests and integration checks cite the identifier they verify. The conformance table in the architecture document maps every identifier to its verification method and the milestone by which it must pass.

## Protocol specification

`docs/ARCHITECTURE.md` specifies the instrument protocol at the level of semantics: message families, required fields, ordering guarantees, backpressure, cancellation, error semantics, versioning policy, and invariants. The concrete wire specification (exact encodings and schemas, handshake sequences, version negotiation) lives in `docs/PROTOCOL.md`, owned by the prototype track, and must satisfy the architecture's requirements.

## Repository layout

The architecture document maps its components to these directories.

```
.
├── README.md
├── AGENTS.md                 Instructions for automated contributors
├── pyproject.toml            Python package metadata; installs the `ciw` command
├── .github/workflows/        Continuous integration: Python matrix, packaging, container, Godot, pinned adapter integration
├── compose.yaml              Container backend definition for Docker Compose
├── Dockerfile                Container image for the backend service
├── src/ciw/                  Runtime service, adapters, operations, and terminal client (Python)
│   ├── core/                 Generic record structure and identities
│   │   ├── records.py        `run.v1` structural validation: finite values, time order, units, lengths
│   │   ├── identities.py     Content identities (`evidence_id`) and distinct event identities
│   │   └── covariance.py     `covariance-artifact.v1`: ordered quantities and units, full matrix, content identity; validated, never computed
│   ├── adapters/             Declarative adapter seam and its bindings
│   │   ├── protocol.py       Instrument manifest with roles; the refusal envelope
│   │   ├── registry.py       Explicit adapter registration; record-only reader for retained evidence
│   │   ├── subprocess.py     Pinned, operator-bound subprocess binding with bounded JSON execution
│   │   ├── oscillator.py     The analytic oscillator as an ordinary adapter, calculation unchanged
│   │   ├── oscillator_records.py  Saved-payload schemas of `statistics.v1` and `spectrum.periodogram.v1`
│   │   ├── fsrt_records.py   Saved-payload schema of `fsrt.tank-reconstruct.v1`
│   │   ├── rci_records.py    Offline provenance checks of the `ciw.rci-source.v2` measurement record; no provider executed
│   │   └── covariance_records.py  Saved-payload schemas of `fsrt.tank-reconstruct.v2` and `jspt.covariance-propagate.v1`; result dependency checks on reopen
│   ├── operations/           Operations, executions, and results
│   │   ├── registry.py       Process-local operation registration with declared roles
│   │   ├── runner.py         Sequences an execution; seals execution and result records; retains refusals
│   │   └── schemas.py        Trusted saved-payload schemas validated without executing a provider
│   ├── adapter-runtimes.json Pinned upstream revisions and modules of the subprocess providers
│   ├── instruments.py        Compatibility facade over the adapter registry for the v1 instrument API
│   ├── investigation.py      RCI calibration to FSRT estimation through the shared session; inspect and replay
│   ├── covariance_workflow.py JSPT covariance operations over a saved investigation: bind, execute, replay
│   ├── calibration_status.py Serving-time calibration applicability at one explicit `evaluated_at`; never hashed into evidence
│   ├── session.py            Authoritative session: selection, immutable results, executions, workspaces
│   ├── server.py             WebSocket transport, bind policy, and saved shutdown
│   ├── cli.py                Headless analysis, service control, health probe, terminal access, investigations, covariance operations
│   ├── calibration.py        Calibration profiles: record shape, validity, and refusal (planned)
│   ├── bench.py              Deterministic bench: fixtures, tolerance policy, report (planned)
│   ├── plsr-runtime.json     Pinned source manifest of the optional PLSR runtime: commit, package version, file digests
│   ├── plsr.py               Portable PLSR run bundles: evaluate, inspect, replay
│   └── plsr_engine.py        Source-pinned bridge to the optional external PLSR runtime
├── godot/                    Godot project for the 2D/3D viewport, a client of the session
│   └── tests/                Headless client checks: protocol smoke, channel generality, adapter boundary, capture view
├── deploy/                   Deployment guides: native controller and container backend
├── examples/                 Reference inputs for integrated instruments, such as `examples/plsr/`
│   ├── adapters/             Investigation fixtures for the pinned adapters: `two-reservoir.json`, `two-reservoir-covariance.json`, `tank-covariance-map.json`
│   └── calibration/          Calibration artifacts per instrument: `calibration/<instrument>/` (planned)
├── scripts/                  Integration checks, the pinned-adapter gate over current and historical checkouts, and the native deployment controller
├── tests/                    Python test suites; conformance tests go under `tests/conformance/`
│   ├── test_adapter_seam.py  Generic record, manifest, registry, and identity checks
│   ├── test_adapter_cli.py   Health probe over generic records; investigation commands and terminal rows
│   ├── test_investigation.py Calibrated RCI-to-FSRT investigations (skipped without the pinned checkouts)
│   ├── test_operation_runner.py  Execution and result records, refusals, capacity, reopen without execution
│   ├── test_subprocess_adapter.py  Pinned subprocess binding: source and interpreter pins, bounded execution
│   ├── test_calibration_status.py  Serving-time calibration status on `session.get`, `result.list`, `result.get`; `evaluated_at` validation
│   ├── test_covariance_artifacts.py  `covariance-artifact.v1` validation: symmetry, semidefiniteness, units, content identity
│   ├── test_covariance_records.py  Saved-payload schemas of the v2 estimation and covariance operations; dependency and cycle refusals
│   ├── test_covariance_integration.py  Calibrated v2 estimation, JSPT propagation, offline replay, historical pins (skipped without the pinned checkouts)
│   ├── test_covariance_replay_refusal.py  Replay refused when the retained attempt had no bound runtime
│   ├── test_rci_records.py   Offline provenance checks of the v2 measurement record
│   ├── test_calibration.py   Calibration record, profile validity, corrected-data provenance (planned)
│   └── test_bench_fixtures.py Deterministic fixture set, tolerance policy, bench report (planned)
└── docs/
    ├── ARCHITECTURE.md       Architecture and normative requirements
    ├── PROTOCOL.md           Wire-level protocol specification
    ├── ADAPTERS.md           Generic adapter contract; the RCI/FSRT investigation, pins, and offline replay
    ├── COVARIANCE.md         Covariance artifacts, the RCI v2 to FSRT v2 to JSPT path, serving-time calibration status, offline replay, historical pins
    ├── COVARIANCE_HANDOFF.md Delivered boundaries and exercised checks of the covariance increment
    ├── INSTRUMENTS.md        Catalogue of integrated instruments with their specifications
    ├── PLSR.md               The PLSR terminal instrument: commands, bundle, digests, limits
    ├── quickstart.md         Running the prototype from source
    ├── coordination.md       Build coordination and integration sequence
    └── DEVELOPMENT.md        This guide
```

`recordings/`, `results/`, and the native deployment's `.ciw/` data directory hold local outputs and are ignored by git.

Planned growth follows the same tree: adapters go under `src/ciw/adapters/` and operations under `src/ciw/operations/`, `src/ciw/instruments.py` remaining the compatibility facade; terminal panels go under `src/ciw/tui/`; protocol codecs for binary transport go under `src/ciw/protocol/`; reference instruments and their fixtures used by conformance tests go under `examples/`.

## Conformance tests

Every conformance test names the requirement it verifies. Use the identifier in the test name (for example `test_ciw_sync_003_cursor_propagates_to_all_views`) or in a marker that carries the identifier, and keep one module per area under `tests/conformance/`. The conformance table in `docs/ARCHITECTURE.md` is the source of truth for which method verifies each identifier and by which milestone it must pass; a test that verifies an identifier not in that table is a signal to update the table.

## Working on the shared branch

Both tracks commit to `main`. Pull with rebase before pushing, keep documentation and code changes in separate commits, and never rewrite published history. A change that alters a contract, a requirement identifier, or the reference implementation arrangement updates the architecture document in the same push.

## Reading order for contributors

1. `README.md`, for the name, definition, and scope.
2. [`docs/ARCHITECTURE.md`](ARCHITECTURE.md), for the component architecture, data model, instrument contract, synchronization model, conformance table, and roadmap. Its Purpose section carries the naming and positioning that the contracts rest on, and its Decisions table (Section 19) carries the decision behind each one.
3. [`docs/PROTOCOL.md`](PROTOCOL.md), for the wire-level protocol, and [`docs/quickstart.md`](quickstart.md) to run the prototype.

## Documenting an integrated tool

The README is the entry point for successfully integrated tools. Update its
catalogue and [the instrument guide](INSTRUMENTS.md) in the same change that
delivers or changes a tool's workbench connection. Preserve the surrounding
architecture documentation and review concurrent documentation edits before
publishing.

Each entry must provide:

- Purpose and current integration status, distinguishing synthetic examples,
  computational validation, and physical validation.
- Tool and adapter identity, schema/protocol versions, and a reproducible release
  or full commit pin. Built-in tools use the CIW revision and their instrument ID.
- Prerequisites, installation/configuration instructions, and copyable terminal
  commands with explicit inputs and expected outputs.
- Supported operations; input/output schemas; dimensions, state/channel order,
  units, coordinate frames, time and sampling conventions where applicable.
- Result, refusal and error meanings, including exit-code behavior and the
  evidence/provenance retained by the workbench.
- A worked example, validation commands and linked test evidence for the actual
  CIW path; supported views and save/reopen behavior where implemented.
- Known limitations and links to the upstream instrument's authoritative docs.

Promote a tool to the integrated catalogue only after its documented CIW path
has been exercised with a representative input and its output and provenance
checked. Successful upstream tests or a published input schema alone establish
standalone readiness, not workbench integration. Keep pending tools in a clearly
labelled section until the adapter and its verification are delivered.

## Names and deployment namespace

The formal technical name is used wherever a contract is stated. The organization and product labels are presentation; the slug is deployment. A name from the presentation or deployment rows never appears in a record's identity.

| Role | Value | Used in |
|---|---|---|
| Formal technical name | Computational Instrumentation Workbench | `docs/ARCHITECTURE.md`, `docs/PROTOCOL.md`, package documentation |
| Short form | CIW | Requirement identifiers, the `ciw` command, instrument identifiers, file and directory names |
| Organization and display label | Notation Systems | README header, interface branding, container labels, launchers, workspace browser labels |
| Short product label | Notation Systems Workbench | Product-facing contexts needing a single name |
| Machine slug | `notation-systems-ciw` | Compose project name, container names, deployment data-directory namespacing |
| Container image repository | `notation-systems/ciw-backend` | The backend service image |

Unchanged by this convention: the distribution name `computational-instrumentation-workbench`, the `ciw` console script, requirement identifiers, instrument identifiers (`org.ciw.*`), operation identifiers, every identity field in a record, and the repository name.

A compose project name is the prefix of its named volumes, so adopting the slug as the project name renames the workspace volume and an existing deployment no longer finds its saved investigation under the new project. The change that sets the project name carries a migration note in the deployment guide: copy `/data` out and back in, or keep running under the former project name.

## Recording decisions

The architecture document's Decisions table (Section 19) is the published record of every decision the contracts rest on: one row per decision, with the trade-off that settled it. A change that revisits a decision updates that row in the same push. Longer deliberation, including the alternatives that were weighed and rejected, is kept outside this repository.
