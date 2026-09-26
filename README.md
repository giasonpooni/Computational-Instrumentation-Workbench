# Parametric Design Testbed

**State. Variation. Invariance.**

**An evidence-backed environment for modelling, varying and evaluating
engineered systems.**

The **Parametric Design Testbed (PDT)** is a terminal-first laboratory for
computational engineering design. It connects explicit design parameters and
model relationships to computed responses, candidate comparisons and retained
evidence.

Engineering design is the purpose; parametric variation is the method. The
testbed is where models, numerical operations and verification methods are
exercised and challenged, rather than treating a generated candidate as an
established result.

Part of **Notation Systems' computational instrumentation and evidence
infrastructure** for industrial and cyber-physical systems.

[Documentation index](docs/README.md) · [Workbench overview](docs/WORKBENCH_OVERVIEW.md) ·
[Systems catalog](docs/SYSTEMS_CATALOG.md) · [Stack map](docs/STACK.md)

## The design language

Start with three questions:

| Primitive | Engineering question |
| --- | --- |
| **State** | What system or candidate design are we describing? |
| **Variation** | What can change, and through which allowed transformations? |
| **Invariance** | What must remain true through those transformations? |

Parameters expose selected design choices; they are not the entire state. An
oscillator's damping is a parameter, while its position and velocity are state
variables. Varying the parameter changes the predicted response under the
declared model.

Parametric design need not be limited to numerical sliders. A supported profile
may also expose a material choice, component configuration or model selection.
Each operation must declare what it actually supports; structural changes must
remain explicit rather than masquerading as ordinary parameter edits.

The aim is a shared working language across engineering domains, not a separate
conceptual framework for every instrument. Units, geometry, clocks, uncertainty,
constraints and numerical methods stay explicit wherever the problem needs
them. Specialist repositories retain authority over their equations and solvers.

The design-study pattern is:

```text
Design question + objectives
          |
          v
Declare model, parameters, allowed variation and requirements
          |
          v
Construct a candidate within the declared domain
          |
          v
Run registered model or measurement operations
          |
          v
Compare responses, uncertainty, constraints and checks
          |
          v
Retain evidence -> revise design choices -> repeat
```

Support for each part of this pattern is operation-specific. The current scope
and linked operating guides below distinguish runnable capabilities from
research goals.

The operator defines the question, assumptions, objectives, constraints and
standard of evidence. AI assistance is optional; generated candidates are not
verified results. The terminal, scripts and optional graphical client remain
useful without an LLM.

### Existing runtime, same substrate

PDT is the current project identity for the existing engineering design
workbench. It extends the **Computational Instrumentation Workbench (CIW)**;
it does not create a second runtime. The distribution remains
`computational-instrumentation-workbench`; the Python import and CLI command
remain `ciw`. References to CIW in source and technical documentation identify
that existing runtime and its contracts.

The project-name change does not rename schemas, migrate saved records or
replace numerical engines. New design workloads, representations and checkers
must plug into the existing substrate rather than create a parallel one.

## What the testbed does

    observe -> align -> calibrate -> propagate uncertainty
       -> estimate -> evaluate constraints -> diagnose -> advise
                         \-> retain, inspect, replay

The existing CIW runtime coordinates that loop:

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

The registered workflows, exact operation status, commands, pins, limits and
validation evidence live in [INSTRUMENTS.md](docs/INSTRUMENTS.md) and
[INTEGRATION_COVERAGE.md](docs/INTEGRATION_COVERAGE.md).

## Current scope

| Area | Current state |
| --- | --- |
| Terminal and Python session | Implemented local service, operation dispatch, persistence and replay. |
| Synthetic instrumentation | Analytic oscillator with statistics, spectrum, saved results and reopening. |
| Mathematical education and augmentation | Equation cards, declared assumptions and bounded offline what-if previews tied to the Julia oscillator seam. |
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

## A testbed for parametric design and representation

The research question is:

> Can a shared state/variation/invariant representation make parametric design
> studies easier to compose and compare while preserving required engineering
> results and evidence?

PDT tests both the design methods and the representation used to connect them.
Reference comparisons, held-out inputs and deliberately invalid cases expose
the limits of each claim. The proposed comparison is deliberately concrete:

```text
Conventional formulation ------> reference execution ---+
                                                       |
Shared state representation ---> registered execution --+--> compare
                                                            |
                                  results, invariants, error, cost
```

Start with one bounded parameter study, such as the existing oscillator model
exploration. Damping variations expose changes in the trajectory and energy
trace. Retain the reference formulation, declare the valid input domain and
tolerances, compare both paths on held-out inputs, and challenge them with
invalid cases. Extend to measurement/design and geometry workflows only as the
shared representation earns that extension.

| Question | Evidence to collect |
| --- | --- |
| Are the required answers preserved? | Output agreement under declared tolerances, uncertainty semantics, reference comparisons and failure cases. |
| Is the representation smaller? | Explicit definitions of representation size, duplicated interfaces, adapter code and additional structure required. |
| Is integration easier? | Work needed to add or modify an operation, with execution and verification costs reported separately. |
| Are the claims supported? | Retained inputs, source/runtime pins, executions, checks, refusals and replayable results. |

An invariant belongs to a declared transformation and domain. Conservation,
admissibility constraints and consistency under a change of representation
need distinct checks; they are not interchangeable. Passing those checks alone
does not establish output equivalence or physical validity.

This is a research programme, not a completed universal compiler or a measured
compression result. Category theory, topology and other mathematical machinery
can support composition and domain structure where a workload needs them.
The objective is **less duplicated representation, with the required meaning
and evidence preserved**.

The same three questions also guide the human-facing learning interface:
understand the state, explore a variation, then inspect what held and what
failed. Wider participation in scientific and industrial design is a goal to
validate, alongside the computational claims.

**Build -> run -> observe -> fix -> audit -> continue.**

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
The generic state-space transformation contract is described in
[STATE_TRANSFORMATIONS.md](docs/STATE_TRANSFORMATIONS.md).
The research vocabulary behind that contract is in
[RESEARCH_CONTEXT.md](docs/RESEARCH_CONTEXT.md).

The existing `ciw.state-transformation-contract.v1` validates a declaration's
structure and content identity. It does not execute a transformation, establish
that an invariant held or authorize equipment. Operation and verification
records remain separate from the contract.

The intended authority modes are Explore, Observe, Prepare and Operate.
Current CIW operations are read-only with respect to external equipment.
Operation success does not authorize an actuator; local protection and machine
controllers remain independent.

## Composable instruments, independent scientific authority

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
The command-only oscillator walkthrough is [OSCILLATOR_OPERATOR.md](docs/OSCILLATOR_OPERATOR.md).
The equation cards and bounded model previews are described in
[MODEL_EXPLORATION.md](docs/MODEL_EXPLORATION.md).

## Evidence boundaries

CIW records what was measured, supplied, estimated, predicted or checked. Those
labels are not interchangeable; the single classification table is maintained
in [Workbench overview](docs/WORKBENCH_OVERVIEW.md#evidence-classes).

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

The next visible gate is one independently challenged physical claim: a
held-out reference measurement, a replayable evidence bundle and a result whose
limitations remain explicit. The detailed integration matrix owns the broader
provider and runtime backlog; the overview does not promote those candidates to
current capability.

See [INTEGRATION_COVERAGE.md](docs/INTEGRATION_COVERAGE.md) for the current
matrix and [SYSTEMS_CATALOG.md](docs/SYSTEMS_CATALOG.md) for provider status.

## Documentation map

| Topic | Canonical page |
| --- | --- |
| Engineering design scope and scientific workspace | [Workbench overview](docs/WORKBENCH_OVERVIEW.md) |
| Provider and instrument map | [Systems catalog](docs/SYSTEMS_CATALOG.md) |
| Implementation architecture | [Architecture](docs/ARCHITECTURE.md) |
| Operation catalogue | [Instruments](docs/INSTRUMENTS.md) |
| Mathematical model exploration | [Model exploration](docs/MODEL_EXPLORATION.md) |
| Executable coverage and limits | [Integration coverage](docs/INTEGRATION_COVERAGE.md) |
| Shared assembly and deployment | [Workbench assembly](docs/WORKBENCH_ASSEMBLY.md) |
| Typed contracts and exchange | [Contract foundations](docs/CONTRACT_FOUNDATIONS.md) |
| State spaces, transformations and invariants | [State transformation contract](docs/STATE_TRANSFORMATIONS.md) |
| Research premise and educational vocabulary | [Research context](docs/RESEARCH_CONTEXT.md) |
| Protocol and identities | [Protocol](docs/PROTOCOL.md) |
| Development and tests | [Development guide](docs/DEVELOPMENT.md) |
| Diagrams | [Diagram atlas](docs/DIAGRAMS.md) |

The full index is [docs/README.md](docs/README.md). Historical audits remain
available for context but do not override the current operation catalogue,
integration matrix or provider manifests.

## License

This project is licensed under the GNU Affero General Public License v3.0 only
(AGPL-3.0-only). See [LICENSE](LICENSE).
