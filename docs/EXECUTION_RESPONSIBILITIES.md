# Execution responsibilities

EDW coordinates mathematical work through declared operations and explicit
provider bindings. Python, Julia, Rust and native libraries can serve different
parts of that contract; an operation does not have to traverse every language.
Scientific meaning belongs to the operation's model, inputs, limits and evidence.

This allocation preserves the implemented [CIW architecture](ARCHITECTURE.md).
It does not move session authority into a new runtime or change historical
records. The [systems catalog](SYSTEMS_CATALOG.md) and
[integration coverage](INTEGRATION_COVERAGE.md) own provider status and gates.

## Allocation and current implementation

| Layer | Assigned responsibility | Current boundary |
| --- | --- | --- |
| Python / EDW | Assemble studies, coordinate operations, validate scientific records, manage the operator session and retain evidence. | Implemented by [core](../src/ciw/core/), [operation runner](../src/ciw/operations/runner.py), [session](../src/ciw/session.py) and workflow modules. These remain the authoritative host implementations. |
| Execution infrastructure, including selected Rust components | Enforce declared execution boundaries, manage resources and bindings, and run supported native computations or checkers. | CIW's [pinned subprocess adapter](../src/ciw/adapters/subprocess.py) is Python. External SCR supplies execution and proof infrastructure for supported paths. A general Rust supervisor replacing this host is not implemented or required by this allocation. |
| Julia / JuMP / JuliaControl | Evaluate mathematical models, select constrained design candidates and support time-domain design where a workload is qualified. | The [oscillator](JULIA_OSCILLATOR.md) has a registered Workbench/provider seam and its own provisioning gates. The [JuMP study](DESIGN_ADJUSTMENT.md) has a real worker and exact Python checker, but remains an unregistered source-checkout study. JuliaControl integration remains pending. |
| C/C++ and other native providers | Supply specialized simulation, numerical kernels and equipment interfaces through qualified adapters. | Each existing provider keeps its own binding and scientific scope. This role does not add a general mechanics engine, hardware plugin system or equipment command path. |
| SP1 | Prove execution of a selected registered program or checker, followed by separate verification. | [Registered integer heat](PROVED_HEAT.md) supplies the existing profile and genuine prove/verify gate. It does not prove arbitrary Rust, Julia or JuMP execution. A design-checker guest remains pending. |

The [Julia/SCR contract](JULIA_SP1.md) describes specification, computation and
execution bindings. SCR is an external component, not a synonym for a language.
A Julia or native provider may execute directly below its approved adapter;
adding an intermediate language is a deployment choice requiring a concrete
benefit and its own validation.

## What crosses an execution boundary

A language binding only transports values. For a registered workbench operation,
the surrounding contract must also establish:

- The versioned operation and captured input bytes, parameters and selection.
- Ordered quantities, units, frames, clocks, missingness, uncertainty assumptions
  and applicability limits required by that particular operation. Not-applicable
  and unknown declarations remain distinct.
- The explicitly approved provider and runtime, plus relevant arithmetic,
  numerical settings and resource limits. Saved records cannot choose an
  executable or extend the registration allowlist.
- The returned output or refusal and its provenance, with check scope separated
  from solver status, content equality and physical evidence.

The [protocol](PROTOCOL.md) owns operation/execution/result identity and refusal
semantics. The [covariance contract](COVARIANCE.md) owns uncertainty declarations.
These obligations apply regardless of the provider's implementation language.
Changing a binding must not silently change coordinate order, clock meaning,
units, tolerances, failure behavior or what a retained result establishes.

The current host captures inputs, calls providers outside the session lock,
validates their response and publishes retained records under that lock.
Timeouts, output limits and source/runtime identity checks are implemented
controls; they do not create an operating-system sandbox or establish scientific
correctness. Any future Rust supervisor must preserve those lifecycle semantics
and demonstrate cancellation, refusal and resource behavior before replacing a
working boundary. Native code remains subject to its own memory and numerical
validation obligations.

## Existing design example

The [bounded adjustment study](DESIGN_ADJUSTMENT.md) exercises this division:
Python validates a dimensionless problem and snapshots its worker/environment;
Julia builds the fixed JuMP quadratic and HiGHS produces a candidate; Python
retains the raw response and explicitly converts the binary64 candidate before
checking a rational primal/dual certificate. The full nonlinear response is
shown separately from the local model's objective.

Its invocation UUID and content identities do not substitute for registered
operation, execution, result or verification records. Reopening inspects retained
bindings without launching Julia or rerunning the checker. A fresh solve or check
is an explicit action. The operation-specific page owns commands, acceptance
rules and remaining SCR/SP1 gates.

This is the immediate implemented path from a proposed change to a checkable
design result. It contains no equipment command, physical feedback or
JuliaControl execution.

## Physical execution is a separate boundary

Model prediction, command authorization and measured equipment response are
different records with different supporting evidence. Current CIW operations
are read-only with respect to external equipment. Solver success, a certificate
check or a proof cannot grant an equipment capability.

For a future qualified equipment interface, EDW would prepare a candidate;
deployment policy would decide whether a bounded command is permitted; the
equipment interface and local controller would execute it; acquisition would
retain the observed response with its own calibration, timing and uncertainty.
A command acknowledgement would not establish that the requested force, power
or motion actually occurred.

Interactive Python orchestration and proof production belong outside a future
deadline-critical actuator loop. The local controller must be able to execute
its accepted configuration and fallback without waiting for the workbench or a
proof job. Timing qualification must cover acquisition, transport, scheduling,
computation and command delivery; a language choice or kernel benchmark does
not close that gate. Local protection remains independent.

This page defines responsibility boundaries. It supplies no new operation,
equipment authorization, real-time guarantee or validation result.
