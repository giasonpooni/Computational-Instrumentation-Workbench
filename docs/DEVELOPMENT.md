# Development guide

This guide is for people working on the Computational Instrumentation Workbench itself. It describes how the work is organized, how documents and code refer to each other, and what to read first.

## Workstreams

Development proceeds in two tracks that hand off through this repository.

| Track | Owns | Delivers |
|---|---|---|
| Documentation | `README.md`, `docs/ARCHITECTURE.md` | The contracts, data model, synchronization semantics, and quality budgets, each stated as a numbered requirement |
| Prototype | `src/ciw/`, `godot/`, `scripts/`, `tests/`, `docs/PROTOCOL.md`, `docs/quickstart.md`, `docs/coordination.md`, continuous integration | An implementation of the contracts, the wire-level protocol specification, and the evidence that the implementation conforms |

The hand-off rule is: the architecture defines contracts; the prototype implements them; tests and integration checks verify conformance against requirement identifiers; decisions raised by either track are recorded as ADRs.

## Requirement identifiers

Normative statements in `docs/ARCHITECTURE.md` carry a stable identifier and use MUST, SHOULD, or MAY wording. The identifier form is `CIW-<AREA>-<NNN>`, for example `CIW-DATA-003` or `CIW-SYNC-007`. Areas are:

| Area | Covers |
|---|---|
| `DATA` | Data model: channels, time bases, units, uncertainty, provenance |
| `INST` | Instrument contract: manifest, lifecycle, messages, backpressure, errors |
| `SYNC` | Synchronization: shared cursor, selection, units, ordering, latency |
| `VIEW` | Representation layer: numerical, temporal, spectral, 2D/3D |
| `SESS` | Sessions, persistence, recording, replay, export |
| `OPS` | Operator interaction: commands, keybindings, layouts, scripting, headless use |
| `EXT` | Extension model: adding instruments, representations, exports |
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
├── .github/workflows/        Continuous integration: Python matrix, packaging, container, Godot
├── compose.yaml              Container backend definition for Docker Compose
├── Dockerfile                Container image for the backend service
├── src/ciw/                  Runtime service, instruments, and terminal client (Python)
│   ├── instruments.py        Scientific records and computations of the first instrument
│   ├── session.py            Authoritative session: selection, immutable results, workspaces
│   ├── server.py             WebSocket transport, bind policy, and saved shutdown
│   ├── cli.py                Headless analysis, service control, health probe, terminal access
│   ├── plsr.py               Portable PLSR run bundles: evaluate, inspect, replay
│   ├── plsr_engine.py        Source-pinned bridge to the optional external PLSR runtime
│   └── plsr_corpus.py        Declared reference corpus: reproduce, compare, re-record
├── godot/                    Godot project for the 2D/3D viewport, a client of the session
├── deploy/                   Deployment guides: native controller and container backend
├── examples/                 Reference inputs for integrated instruments, such as `examples/plsr/`
├── scripts/                  Integration checks and the native deployment controller
├── tests/                    Python test suites; conformance tests go under `tests/conformance/`
└── docs/
    ├── ARCHITECTURE.md       Architecture and normative requirements
    ├── PROTOCOL.md           Wire-level protocol specification
    ├── INSTRUMENTS.md        Catalogue of integrated instruments with their specifications
    ├── PLSR.md               The PLSR terminal instrument: commands, bundle, digests, limits
    ├── quickstart.md         Running the prototype from source
    ├── coordination.md       Build coordination and integration sequence
    └── DEVELOPMENT.md        This guide
```

`recordings/`, `results/`, and the native deployment's `.ciw/` data directory hold local outputs and are ignored by git.

Planned growth follows the same tree: `src/ciw/instruments.py` becomes the package `src/ciw/instruments/` when a second instrument lands; terminal panels go under `src/ciw/tui/`; protocol codecs for binary transport go under `src/ciw/protocol/`; reference instruments and their fixtures used by conformance tests go under `examples/`.

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
