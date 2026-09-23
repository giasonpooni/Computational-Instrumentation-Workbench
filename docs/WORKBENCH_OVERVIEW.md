# Workbench overview

The **Computational Instrumentation Workbench (CIW)** is Notation Systems'
terminal-first scientific authoring and execution environment. It gives
engineers and mathematicians one place to compose models, measurements,
experiments, designs, numerical operations, visualizations and evidence.
Physics, chemistry and engineering are supported domains; the workbench does
not replace the domain repository that owns a model or solver.

The product is an instrument for its operator. The engineer defines the
question, assumptions, objectives, constraints and standard of evidence. Python
provides the current workbench shell and session services; Julia, GPU runtimes,
specialist providers and optional graphical clients enter through explicit
contracts. AI assistance is optional and cannot silently change an objective,
relax a gate or authorize equipment.

## The instrument loop

The shared loop is:

    observe -> align -> calibrate -> propagate uncertainty
       -> estimate -> evaluate constraints -> diagnose -> advise
                         \-> retain, inspect, replay

The loop can be used for a synthetic computation, a retained measurement
session or a distributed instrument spanning several machines. A result is
useful only within its declared model, timing, frame, uncertainty and validity
domain.

Every retained run keeps these identities distinct:

| Identity | Meaning |
| --- | --- |
| Evidence | The bytes or observations that were available. |
| Operation | The versioned computation requested. |
| Execution | One invocation, including refusal or runtime failure. |
| Result | The output sealed by that invocation. |
| Verification | A separate check and the claim scope it covers. |

Reopening reads retained records. Replay is an explicit new invocation and
creates fresh execution and result identities. A digest, passing numerical
test or successful replay is not, by itself, physical validation, state
admission or permission to actuate.

## Three graphs share one project

Projects expose three related graphs without collapsing their meanings:

1. **Physical/model graph** — components, materials, sensors, actuators,
   equations, units, frames and validity domains.
2. **Computation graph** — operations, dependencies, solvers, estimates,
   constraints and selected visualizations.
3. **Evidence graph** — source bytes, calibrations, assumptions, executions,
   results, checks, corrections and the conclusions that depend on them.

Changing a sensor, calibration, model or provider should identify affected
results, mark them stale where appropriate and preserve the earlier record.
Undo can reverse a project edit; it cannot erase an experiment already
performed.

The executable persistence and admission rules are in
[ARCHITECTURE.md](ARCHITECTURE.md), the exact record fields are in
[PROTOCOL.md](PROTOCOL.md), and the cross-provider assembly is described in
[WORKBENCH_ASSEMBLY.md](WORKBENCH_ASSEMBLY.md).

## Execution surfaces

| Surface | Current role |
| --- | --- |
| Terminal and Python API | Authoritative local session, operation dispatch, saved workspaces and replay. |
| Optional Godot client | Read-only 2D/3D representations of supported retained data; it is not a second state store. |
| Provider processes | Explicitly pinned numerical engines invoked outside the session lock. |
| Julia direction | Planned scientific core for shared model definitions, estimation, control and experiment design; no resolved worker is currently part of CIW validation. |
| GPU and machine interfaces | Captured through bounded providers and host-side telemetry; deadline-critical protection remains local to the machine. |
| MCP and agents | Optional tools at the workbench boundary. They propose or inspect work; deterministic contracts decide acceptance. |

The workbench currently has twenty-five registered workflow kinds in the shared
session. Their exact status, commands and limits are maintained in
[INSTRUMENTS.md](INSTRUMENTS.md) and
[INTEGRATION_COVERAGE.md](INTEGRATION_COVERAGE.md), rather than duplicated here.

## Authority modes

The intended operating modes are:

| Mode | Permitted activity |
| --- | --- |
| Explore | Models, simulations, optimization and virtual fabrication. |
| Observe | Approved live measurements, recording and estimation; no actuation. |
| Prepare | Candidate toolpaths, firmware or control configurations and replay tests. |
| Operate | An explicitly enabled, bounded operation on supported equipment. |

CIW's current operations are read-only with respect to external equipment.
Operation success does not grant authorization. Protective functions and local
machine controllers remain independent of an editable notebook, agent or
viewport.

## Evidence classes

This is the single classification table used by the workbench overview and
linked from shorter operator pages:

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

## Distributed instruments and context

A distributed instrument combines timestamped interfaces, local machine
protection, host computation and retained evidence. Measurements arriving in
one process are not automatically simultaneous: clock declarations, frame
transforms, network delay, buffering, transfer cost and command expiry belong
in the model.

An optional project context bundle can add weather, terrain, material
availability, equipment documentation, cost references or site records. Each
source keeps its time, spatial resolution, units, uncertainty and permitted
use. A regional forecast is not a site measurement, a supplier listing is not
confirmed stock, and co-location is not a causal relationship. Local operation
must remain possible when an external service is unavailable.

This supports a local engineering workshop: measure a process, compare
explanations, design and fabricate a repair or fixture, test it against
independent references, and retain the evidence for the next maintainer.
Safety-critical water, electrical, pressure, structural and chemical decisions
still require the appropriate local standards and qualified review.

## Integration rule

An upstream repository remains the source of truth for its equations, solver,
record schema, license and refusal conditions. CIW adds a typed adapter,
explicit source/runtime pins, save/reopen/replay behavior and a qualified
status. It does not copy a provider README into the workbench or turn a
repository link into an implemented integration.

See [SYSTEMS_CATALOG.md](SYSTEMS_CATALOG.md) for the provider map and
[DEVELOPMENT.md](DEVELOPMENT.md) for the delivery checklist. The stack-level
ownership and numerical boundaries remain in [STACK.md](STACK.md).
