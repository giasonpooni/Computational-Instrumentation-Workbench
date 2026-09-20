# ADR-0003: Reference implementation arrangement

- **Status:** Accepted
- **Date:** 2026-09-20

## Context

The architecture is contract-first: the instrument contract, the result envelope, the session protocol, and the synchronization semantics are specified so that they hold whatever language or toolkit implements them. A prototype still has to pick one arrangement, and the prototype track has done so. The architecture document needs to name that arrangement as its primary reference stack so that its examples, budgets, and repository layout match what is being built, while keeping the alternatives documented.

## Decision

The reference implementation arrangement is:

- **Computational engines in Python with NumPy**, using SciPy where an operation needs it. Instruments run as Python processes or in-process modules behind the instrument contract. A later native or GPU engine enters through the same contract with its own numerical tolerances and execution requirements; nothing in the host depends on the engine's implementation.
- **A terminal frontend** as the primary and fully capable interface, implemented in the same language as the runtime for the prototype, with the toolkit named and justified in the architecture document. It works headless and over SSH, and it renders 2D representations in-terminal when no viewport is attached, using cell, braille, or half-block raster and terminal graphics protocols where the terminal supports them.
- **A Godot-based 2D/3D viewport** as the auxiliary viewer for spatial representations. The viewport is a client of the runtime over the same session protocol as the terminal: it receives control messages and binary buffers over a local socket, shares the time cursor, selection, and units with the terminal, and owns no calculation of its own.

The architecture document specifies all contracts language-neutrally. The wire-level protocol specification is owned by the prototype track and lives in `docs/PROTOCOL.md`.

## Consequences

- Examples, budgets, and the repository layout in the architecture document assume this arrangement. The contracts do not.
- The host design based on Rust with a native terminal toolkit and Arrow buffers, explored while drafting the architecture, is retained in the architecture document as the documented alternative for a later native path. Its protocol ideas (manifest, lifecycle, credit-based backpressure, in-process, subprocess, and remote transport bindings) are carried over into the language-neutral contract.
- Replacing any one of the three layers requires a new ADR and an update to the architecture document's reference-stack section, but no change to the contracts.
- The terminal and viewport connection is not a real-time path. Deadline-critical execution stays in a dedicated engine or controller, per ADR-0002.

## Alternatives considered

- **Rust host with a native terminal toolkit and Arrow buffers.** Strong on latency, memory layout, and single-binary distribution. Not chosen for the prototype: the engines already exist in NumPy, and the prototype's purpose is to validate the contracts and the reusability target quickly. Kept as the alternative path.
- **Terminal-only 3D rendering.** Rejected as the primary path for spatial results: legibility of meshes, fields, and trajectories in cell rasters is too low for inspection. Retained as a fallback for headless previews.
- **A browser frontend as the primary interface.** Rejected: the project is terminal-first, and a browser frontend would move the control surface off the terminal. It can be added later as another client of the runtime.
- **Embedding the 2D/3D viewer in the terminal process.** Rejected: it couples the control surface to a GPU-bound process and prevents running the viewport on a different machine from the runtime.
