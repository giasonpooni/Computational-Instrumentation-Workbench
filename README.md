# Computational Instrumentation Workbench

*Integrated Measurement, State Estimation, and Visualization Workbench*

> A terminal-first workbench that connects computational instruments to synchronized numerical, temporal, spectral, and 2D/3D representations of physical-system observations and estimated states.

## What it is

The Computational Instrumentation Workbench (CIW) is an environment for investigating physical systems. Computational instruments (estimators, reconstructions, integrators, signal-processing stages, and sources of measured data) run as replaceable backends behind a common contract. The workbench composes them and presents their outputs in four synchronized representation families: numerical readouts and tables, temporal traces, spectral views, and 2D/3D views of fields, trajectories, and states. The primary interface is a terminal, and it works headless and over SSH.

Each word of the name carries a specific meaning:

- **Computational** covers the estimation, reconstruction, integration, and signal-processing backends, not just directly measured values.
- **Instrumentation** keeps the focus on investigating physical systems through measurements, models, and diagnostics.
- **Workbench** describes an environment in which several instruments and analytical operations can be used together. It does not imply that the visualization layer replaces those instruments.

## Status

The project is at the architecture stage. There is no code to install or run yet. The name and definition are fixed in [ADR-0001](docs/adr/0001-project-name-and-definition.md); the architecture document is in progress.

## Workstreams

Development proceeds in two tracks that hand off through the repository:

- **Documentation track:** this README, the architecture document, and the Architecture Decision Records (ADRs). The architecture document defines the contracts, data model, synchronization semantics, and quality budgets, each as a numbered requirement.
- **Prototype track:** the executable prototype, the wire-level protocol specification and its implementation, the test suites, and the integration checks. Tests and integration checks verify the prototype against the numbered requirements.

Decisions raised by either track are recorded as ADRs.

## Reading order

1. This README, for the name, definition, and scope.
2. [ADR-0001](docs/adr/0001-project-name-and-definition.md), for the naming decision and the alternatives that were rejected.
3. The [ADR index](docs/adr/README.md), for later decisions.
4. `docs/ARCHITECTURE.md`, once it lands, for the component architecture, data model, instrument contract, synchronization model, and roadmap.
