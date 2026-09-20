# ADR-0002: Position the workbench as a foundational instrumentation runtime

- **Status:** Accepted
- **Date:** 2026-09-20

## Context

The workbench could serve many kinds of instruments: fluid-state reconstruction, construction-state estimation from building models, polymer and process experiments, GNSS and odometry instrumentation, dynamical-system and observer experiments. A layer that broad can fail in two opposite ways. It can try to become a universal physical model, which no host can maintain, or it can become a compulsory collection of features that every instrument must carry whether or not they mean anything for it.

A broadly reusable instrument environment does not need a universal physical model. It needs a consistent way to connect, execute, inspect, compare, and record specialized models and measurements.

## Decision

The Computational Instrumentation Workbench is a **foundational instrumentation runtime and workbench**: a common layer underneath specialized tooling. Its core stays small because it captures recurring operations, not because it omits their meaning.

1. **One recurring operation loop.** The runtime supports the same loop for every instrument: attach a source, select quantities and a domain, run an operation, inspect the result, compare, save or replay. The source may be recorded telemetry or a simulation; the operation may be state estimation, integration, or spectral analysis; the display may be a phase portrait or a physical trajectory. The host does not need to know the equations involved to manage that workflow.

2. **The workbench supplies the operating environment; the instrument supplies the scientific meaning.** Views are capability-driven, never compulsory. A scalar instrument does not need a mesh. A recorded-data instrument does not need a live device connection. A spatial instrument does not get a spectral panel unless the analysis is meaningful. The instrument's manifest declares what it produces and which representations apply.

3. **Generality lives in the contract, not in one format.** The result envelope and the execution contract are standardized; payload types and engines are not. The envelope states what a result is, which quantities it contains, what its coordinates and timestamps mean, how it was produced, and which evidence and checks support it. It carries units, shapes, coordinate frames, time bases, missing-data conventions, and available uncertainty, together with separate evidence, operation, execution, and verification identities. The payload may be an array, a mesh, a table, a sparse structure, or a reference to a larger artifact. Nothing is flattened into one universal state vector, and backends do not share internal classes. The rule is: be strict about meaning at the boundary; be flexible about implementation behind it.

4. **Frontends are clients of the computation.** The terminal interface and the 2D/3D viewport both subscribe to the same runtime state and result streams. Neither owns a separate calculation.

5. **Representability is separate from service level.** Whether the workbench can represent and operate a kind of instrument is one question. Whether a particular deployment meets that instrument's timing, throughput, precision, and reliability requirements is another. The terminal and viewport connection is not suitable for deadline-critical control loops. Time-critical execution stays in an appropriate engine or controller, and the workbench configures, observes, and inspects it through a declared interface. A large simulation may provide reduced visualization data while its full results are retained elsewhere and referenced. Not every workload runs in the same process, on the same machine, or at the same rate.

6. **Reusability is tested by the second and third instrument.** The design target is

   ```
   new instrument = existing workbench + adapter + domain-specific computation and checks
   ```

   This is a target, not a promised amount of integration work. It is checked with three deliberately different reference instruments: a scalar or time-series instrument, a state-estimation instrument, and an instrument producing a spatial result. For each, the measure is whether the work is mainly an adapter, metadata, and a specialized view, or whether it forces changes throughout the host. Each reference instrument keeps headless reference tests, both frontends must identify the same records, and reopening stored results is distinguished from recomputing a new execution.

## Consequences

- The architecture document defines the result envelope and the execution contract as normative requirements, and organizes the roadmap around the three reference instruments.
- The conformance table includes, per reference instrument, the headless reference tests, the same-records check across frontends, and the reopen-versus-recompute distinction.
- Progress is measured by how much less infrastructure must be rebuilt for the next instrument, not by how many panels the application shows.
- Where a deployment already has an evidence-and-result infrastructure, the workbench operates as an instrumentation workspace over it. It does not become a new database, a new operating system, or a universal solver. The envelope's evidence, operation, execution, and verification identities are the integration points; the adapter to such an infrastructure is specified separately.
- The name stays Computational Instrumentation Workbench. "Foundational instrumentation runtime" describes its position under specialized tooling; it is not a rename.

## Precedents

- Virtual instrumentation as described by National Instruments separates acquisition hardware, drivers, software-defined processing, and presentation, and builds larger systems from smaller instrument functions. This design belongs to that family, with its own implementation and scope.
- Jupyter's messaging architecture lets several frontends attach to one computational kernel and receive different representations of its outputs. That is the model for the terminal and viewport being clients of the computation.
- The SciJava Ops work distinguishes having a plugin mechanism from being able to combine tools with incompatible data structures; adaptation layers are still needed. That is why meaning is standardized at the boundary rather than by forcing one data format.
- ROS 2 real-time design documentation distinguishes meeting computation deadlines from having a responsive interface or a high average update rate. That is the basis for keeping service level separate from representability.

## Alternatives considered

- **A universal state vector or single data format for all instruments.** Rejected: it forces every backend to flatten meshes, tables, sparse structures, and references into one shape, and it still does not make incompatible tools interoperate.
- **One application per instrument domain.** Rejected: each application rebuilds attachment, selection, execution, inspection, comparison, and replay, which is exactly the infrastructure this layer exists to share.
- **A presentation-only layer over existing tools.** Rejected: it drops execution, comparison, recording, and replay, and it would turn the visualization layer into the product rather than the instruments.
- **A hard real-time runtime.** Rejected as a goal of the common layer: deadline-critical loops belong in dedicated engines or controllers that the workbench observes through a declared interface.
