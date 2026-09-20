# Development guide

This guide is for people working on the Computational Instrumentation Workbench itself. It describes how the work is organized, how documents and code refer to each other, and what to read first.

## Workstreams

Development proceeds in two tracks that hand off through this repository.

| Track | Owns | Delivers |
|---|---|---|
| Documentation | `README.md`, `docs/ARCHITECTURE.md`, `docs/adr/` | The contracts, data model, synchronization semantics, and quality budgets, each stated as a numbered requirement |
| Prototype | The executable prototype, `docs/PROTOCOL.md`, test suites, integration checks | An implementation of the contracts, the wire-level protocol specification, and the evidence that the implementation conforms |

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

## Reading order for contributors

1. `README.md`, for the name, definition, and scope.
2. [ADR-0001](adr/0001-project-name-and-definition.md), for the naming decision and the alternatives that were rejected.
3. [ADR-0002](adr/0002-foundational-runtime-positioning.md), for the positioning as a foundational runtime, the contract-first rule, and the reusability target.
4. [ADR-0003](adr/0003-reference-implementation-arrangement.md), for the reference implementation arrangement and its alternatives.
5. `docs/ARCHITECTURE.md`, for the component architecture, data model, instrument contract, synchronization model, conformance table, and roadmap.
6. `docs/PROTOCOL.md`, once published, for the wire-level protocol.
7. The [ADR index](adr/README.md), for later decisions.

## Recording decisions

Decisions that change scope, contracts, or the reference implementation are recorded as ADRs. The procedure is in the [ADR index](adr/README.md).
