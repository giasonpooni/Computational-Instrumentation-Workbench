# ADR-0001: Project name and definition

- **Status:** Accepted
- **Date:** 2026-09-20

## Context

The project needs a name and a one-sentence definition that will be used consistently in the repository, the architecture document, and any technical abstract. The name has to signal three things at once: that the system's backends compute (estimate, reconstruct, integrate, filter) rather than only display measured values; that its purpose is the investigation of physical systems; and that it is an environment in which several instruments are used together rather than a single tool or a display layer.

## Decision

The technical name is **Computational Instrumentation Workbench**, abbreviated **CIW**.

The definition, to be quoted verbatim wherever the project is defined, is:

> A terminal-first workbench that connects computational instruments to synchronized numerical, temporal, spectral, and 2D/3D representations of physical-system observations and estimated states.

Each word of the name has a specific role:

- **Computational** covers the estimation, reconstruction, integration, and signal-processing backends, not just directly measured values.
- **Instrumentation** keeps the focus on investigating physical systems through measurements, models, and diagnostics.
- **Workbench** describes an environment in which several instruments and analytical operations can be used together. It does not imply that the visualization layer replaces those instruments.

For contexts that call for a more explicit engineering title, such as a README subtitle or a technical abstract, the project uses:

> Integrated Measurement, State Estimation, and Visualization Workbench

This title is too long for everyday use and is not used in code, commands, or file names.

## Consequences

- Code, packages, commands, and documentation use "Computational Instrumentation Workbench" or "CIW". The explicit engineering title appears only in the README subtitle and in abstracts.
- The definition fixes the scope of the architecture: instruments are computational backends behind a contract; the workbench composes them; representations are synchronized and cover the four named families; observations and estimated states are distinct kinds of data.
- "Terminal-first" commits the project to a fully capable terminal interface that works headless and over SSH. It does not forbid richer graphics where the terminal or an auxiliary viewer supports them, but the terminal remains the control surface.
- Later ADRs that narrow or extend the scope must do so in terms of this definition.

## Alternatives considered

- **Instrument Dashboard** or **Telemetry Dashboard.** Implies passive display of measured values. Drops the computational backends and the composition role of a workbench.
- **Signal Analysis Terminal.** Narrows the scope to signal processing. Omits state estimation, reconstruction, and 2D/3D representations of estimated states.
- **Physical System Observatory.** Suggests observation only. Omits estimated states and the models that produce them.
- **Measurement and Visualization Suite.** "Suite" implies a bundle of separate applications rather than an environment where instruments are used together, and it foregrounds display over instrumentation.
- **Integrated Measurement, State Estimation, and Visualization Workbench.** Accurate and complete, but too long for everyday use. Kept as the explicit engineering title rather than the name.
