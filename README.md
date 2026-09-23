# Computational Instrumentation Workbench

Part of **Notation Systems' computational instrumentation and evidence
infrastructure** for industrial and cyber-physical systems.

[Documentation index](docs/README.md) · [Workbench overview](docs/WORKBENCH_OVERVIEW.md) ·
[Systems catalog](docs/SYSTEMS_CATALOG.md) · [Stack map](docs/STACK.md)

The **Computational Instrumentation Workbench (CIW)** is a terminal-first,
model-based engineering instrument. It gives engineers and mathematicians one
place to compose measurements, models, experiments, numerical operations,
visualizations and evidence. Physics, chemistry and engineering are supported
domains inside the workspace; specialist repositories remain responsible for
their own equations and solvers.

The operator defines the question, assumptions, objectives, constraints and
standard of evidence. AI assistance is optional. The terminal, scripts and
optional graphical client remain useful without an LLM.

## What the workbench does

    observe -> align -> calibrate -> propagate uncertainty
       -> estimate -> evaluate constraints -> diagnose -> advise
                         \-> retain, inspect, replay

CIW is the coordinating instrument around that loop:

- It captures typed inputs and declared selections.
- It invokes explicitly registered operations and pinned providers.
- It preserves clock, frame, unit, calibration, uncertainty and provenance
  declarations.
- It separates evidence, operation, execution, result and verification
  identities.
- It saves a workspace, reopens it without recomputation, and replays it only
  through an explicit new execution.
- It records refusals and runtime failures without manufacturing a successful
  result.

The workbench currently registers twenty-five workflow kinds in the shared
session. The exact operation status, commands, pins, limits and validation
evidence live in [INSTRUMENTS.md](docs/INSTRUMENTS.md) and
[INTEGRATION_COVERAGE.md](docs/INTEGRATION_COVERAGE.md).

## Current scope

| Area | Current state |
| --- | --- |
| Terminal and Python session | Implemented local service, operation dispatch, persistence and replay. |
| Synthetic instrumentation | Analytic oscillator with statistics, spectrum, saved results and reopening. |
| Measurement and estimation | Pinned RCI/FSRT/JSPT, telemetry, calibrated-process and identified-design paths. |
| Numerical geometry | Native bounded covariance, mesh-path and translation-flow providers, plus pinned flat-torus and curved-surface references. |
| Stability and proof | PLSR terminal operation and selected SCR/SP1 registered computation with qualified scopes. |
| Workstation physical bench | NVML-backed energy-to-accuracy capture and replay when a supported NVIDIA device is available. |
| Machine configuration | Evidence-bound read-only asset, signal and model manifest workflow with save/reopen/replay tests. |
| Visualization | Optional Godot client and read-only geographic projection over retained records. |
| Provider integration | Exact pins and adapter manifests; upstream repositories remain independent sources of truth. |

These are bounded computational capabilities. They do not imply a live DAQ
bus, generic sensor fusion, physical calibration, equipment actuation or
independent verification of every numerical result.

## Architecture

The project keeps three graphs related but distinct:

1. **Physical/model graph** — components, materials, sensors, equations, units,
   frames and validity domains.
2. **Computation graph** — operations, dependencies, solvers, estimates,
   constraints and visualizations.
3. **Evidence graph** — source bytes, calibrations, assumptions, executions,
   results, checks, corrections and dependent conclusions.

The executable rules are in [ARCHITECTURE.md](docs/ARCHITECTURE.md). The exact
wire and record fields are in [PROTOCOL.md](docs/PROTOCOL.md). The cross-provider
session is assembled as described in [WORKBENCH_ASSEMBLY.md](docs/WORKBENCH_ASSEMBLY.md).

The intended authority modes are Explore, Observe, Prepare and Operate.
Current CIW operations are read-only with respect to external equipment.
Operation success does not authorize an actuator; local protection and machine
controllers remain independent.

## Systems and loose-tool collapse

The surrounding repositories are not copied into this checkout or treated as
one undocumented monolith. Each keeps its own README, license, tests and
scientific contract. CIW collapses the *integration surface*: a provider enters
through a typed adapter, an exact source/runtime pin, retained evidence,
save/reopen/replay behavior and a qualified validation gate.

Read [SYSTEMS_CATALOG.md](docs/SYSTEMS_CATALOG.md) for:

- links to the independent README for each connected or candidate system;
- the CIW responsibility and operation boundary;
- the distinction between Native CIW, Pinned adapter, Contract only and
  Pending;
- the current Flat Torus, Curved Surface and ICRH pins;
- the rule that a repository link is discoverability, while exercised records
  and replay are integration.

The stack map in [STACK.md](docs/STACK.md) remains the detailed responsibility
and numerical-foundation reference. [PROVIDER_AVAILABILITY.md](docs/PROVIDER_AVAILABILITY.md)
describes exact local checkout provisioning without anonymous provider clones.

## Quickstart

From Python 3.11 or newer:

    python -m pip install -e .
    python -m ciw demo --output recordings/demo.json
    python -m ciw analyze stats --recording recordings/demo.json --channel q --start 2 --end 8 --output-dir results/stats
    python -m ciw analyze spectrum --recording recordings/demo.json --channel q --start 0 --end 12 --output-dir results/spectrum
    python -m ciw inspect results/spectrum/workspace.json

The demo is synthetic and needs no external provider. For the shared
measurement/design session, follow [WORKBENCH_ASSEMBLY.md](docs/WORKBENCH_ASSEMBLY.md)
and the [quickstart](docs/quickstart.md). Container and workstation deployment
notes are in [deploy/README.md](deploy/README.md).

## Evidence boundaries

CIW records what was measured, supplied, estimated, predicted or checked. Those
labels are not interchangeable:

| Classification | Meaning |
| --- | --- |
| Measured | A declared source produced the observation under a recorded configuration. |
| Estimated | A declared model and observations support an inferred state or parameter. |
| Predicted | A model forecasts an outcome under specified conditions. |
| Verified | A named checker established a named condition within its claim scope. |
| Authorized | A separate operational policy permits an action. |

Unknown uncertainty stays unknown. A digest establishes content identity, not
source authenticity. A successful optimizer is not a stability proof; a
stability check is not physical validation; a retained candidate is not
admitted canonical state.

## Development and validation

Read [DEVELOPMENT.md](docs/DEVELOPMENT.md) before changing a contract or adding a
provider. The normal local checks are:

    python -m pytest -q
    python -m compileall -q src
    git diff --check

Provider gates require the exact clean checkouts described in
[PROVIDER_AVAILABILITY.md](docs/PROVIDER_AVAILABILITY.md). Keep synthetic
fixtures, live measurements, replay outputs and independent verification
results distinguishable in both documentation and records.

## Next gates

The next integration work is deliberately staged:

1. Register the project-graph/model compiler seam in the shared operation,
   execution and result envelopes.
2. Resolve and CI-integrate the Julia worker environment for a thermal parity
   reference before making Julia a required runtime.
3. Deliver one independently challenged physical claim, with a held-out
   reference measurement and a replayable evidence bundle.
4. Complete offline-capable ICRH replay for the remaining exchange producers.
5. Treat FPGA programming, MCP authority, physical calibration and external
   actuation as separate deployment gates.

The current Julia scaffold under runtimes/julia is intentionally not presented
as an integrated provider until its lockfile and environment are reproducible.

## Documentation map

| Topic | Canonical page |
| --- | --- |
| Product scope and scientific workspace | [Workbench overview](docs/WORKBENCH_OVERVIEW.md) |
| Provider and loose-tool map | [Systems catalog](docs/SYSTEMS_CATALOG.md) |
| Implementation architecture | [Architecture](docs/ARCHITECTURE.md) |
| Operation catalogue | [Instruments](docs/INSTRUMENTS.md) |
| Executable coverage and limits | [Integration coverage](docs/INTEGRATION_COVERAGE.md) |
| Shared assembly and deployment | [Workbench assembly](docs/WORKBENCH_ASSEMBLY.md) |
| Typed contracts and exchange | [Contract foundations](docs/CONTRACT_FOUNDATIONS.md) |
| Protocol and identities | [Protocol](docs/PROTOCOL.md) |
| Development and tests | [Development guide](docs/DEVELOPMENT.md) |
| Diagrams | [Diagram atlas](docs/DIAGRAMS.md) |

The full index is [docs/README.md](docs/README.md). Historical audits remain
available for context but do not override the current operation catalogue,
integration matrix or provider manifests.

## License

This project is licensed under the GNU Affero General Public License, version 3
or later. See the published repository's [LICENSE](https://github.com/giasonpooni/Notation-Systems-Workbench/blob/main/LICENSE).

