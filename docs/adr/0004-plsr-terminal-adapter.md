# ADR-0004: Integrate PLSR through a bounded terminal adapter

## Status

Accepted

## Date

2026-09-19

## Context

The workbench's first executable instrument is a synthetic oscillator. An
external computational instrument is needed to exercise declared inputs,
numerical refusals, retained evidence and explicit re-execution. The
Parameterized Lyapunov Stability Runtime (PLSR) now accepts a versioned model
artifact carrying matrices, certificate, boxes, state ordering and units, time
convention, margin policy, estimator identity and provenance.

PLSR's kernel API remains marked `changing`. Physical validation has not started
and proof status remains `NOT_CHECKED`. Its numerical result is a claim about a
declared computation. The oscillator's shared recording/session protocol does
not yet express all PLSR model and verdict semantics. Treating a PLSR result as a
synthetic time-series recording would conceal those distinctions.

The user requested proceeding with the complete terminal integration and
documenting successfully integrated tools in the README and operating guides.

## Decision

Integrate PLSR as an optional experimental terminal instrument with four commands:
`ciw plsr import`, `evaluate`, `inspect` and `replay`. The adapter receives the
upstream model artifact and a separate explicit sample. It uses the public
artifact loader and evaluator, preserving the numerical engine and its status
semantics.

Pin the runtime to commit
`19ea6967060166ba09db6cd4563bd87bd6b3d196`, and verify the installed source
against the adapter's manifest. Equivalent wheels are acceptable independently
of Git metadata. The PLSR extra requires Python 3.12 or newer; the base
workbench's Python 3.11 minimum is unchanged. Record the adapter and actual
Python/dependency versions in each run.

Use the versioned `plsr-sample-v1` input and `ciw-plsr-run-v1` saved-run boundary.
Every evaluation saves a self-contained bundle with the complete declared model,
sample, runtime companion record and runtime identity. Retain separate evidence,
operation, execution, result and verification identities, with verification
`not_verified`. Preserve the model, companion and bundle digests and validate
their input/output relationships.

Inspection validates an existing bundle without recomputation. Replay evaluates
the embedded inputs, creates new execution/result identities, and records its
source bindings plus an exact comparison of the companion-record digest. It
does not overwrite the source or promote the verification status.

Preserve every runtime code and all three verdict booleans. In particular,
display `NUMERICAL_INCONCLUSIVE` as a numerical refusal and `NOT_CERTIFIED` as a
certificate violation. Successful command execution is independent of the
verdict's acceptance. Neither content digests nor replay agreement constitute
proof verification or physical authorization.

Keep this increment separate from the live oscillator service and Godot client.
It changes no WebSocket protocol and declares no general instrument-service
contract. Document that boundary, the installation and runnable workflow, the
schemas, status meanings, checks and limits in the README catalogue and
[PLSR operating guide](../PLSR.md).

## Consequences

- CIW can drive an external declared-model instrument and reproduce its retained
  evaluations without users transcribing model matrices into Python.
- The runtime's evolving API is contained behind a source-pinned adapter. A
  future pin update requires compatibility review and renewed integration checks.
- Python 3.11 users retain the base workbench but need a newer environment for
  PLSR. The optional dependency avoids imposing PLSR on oscillator-only users.
- PLSR runs are portable saved bundles with explicit identities and provenance.
  They are not oscillator workspaces and have no synchronized live viewport yet.
- One sample per invocation supports development and offline workflows; repeated
  calls can be orchestrated by a host. Streaming, sweep scheduling and real-time
  deployment are separate work.
- Physical validation, proof receipts, sensor/state freshness checks and
  deployment authorization remain outside the delivered capability. The records
  retain these limits rather than inferring them from a numerical verdict.

## Alternatives considered

1. **Wait for a tagged and historically stable kernel before any integration.**
   A full source pin permits bounded experimental development now. Deployment
   maturity remains separate from workbench readiness.
2. **Vendor PLSR or copy its calculations into CIW.** This would create another
   numerical implementation to maintain and validate. The adapter preserves the
   upstream engine and artifact boundary.
3. **Extend the shared session protocol immediately.** One external instrument
   does not yet establish which additional abstractions should be shared. The
   terminal bundle provides concrete evidence for a later coordinated contract
   decision without changing existing clients.
4. **Retain only the companion record or rerun an input filename.** A companion
   record alone does not contain the complete model declaration, and filenames
   can point to changed content. Embedding all inputs makes the saved run
   reproducible and inspectable.
