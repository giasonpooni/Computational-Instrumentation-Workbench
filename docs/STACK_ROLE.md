# Computational Instrumentation Workbench in the instrumentation stack

Notation Systems develops computational instrumentation and evidence infrastructure for industrial and cyber-physical systems.
This component owns **operation, inspection and replay**. The [stack map](https://github.com/giasonpooni/Notation-Systems-Workbench/blob/main/docs/STACK.md) locates all public components and distinguishes implemented paths from specifications and scaffolds.

## Current boundary

| Property | Scope |
| --- | --- |
| Implementation | Executable prototype |
| License | GNU Affero General Public License v3.0 only (`AGPL-3.0-only`); see [LICENSE](../LICENSE) |
| Workbench connection | Host; oscillator, five external tool workflows and read-only exchange inspection |
| Inputs | Retained scientific records, explicit selections, operation parameters and bound runtime manifests. |
| Outputs | Saved investigations, execution and result records, terminal analysis and an optional oscillator viewport. |

A session coordinates instruments; it does not replace their numerical engines or infer verification from a view.

## Interoperability

Integrations use the component's documented contract and an explicit adapter. They preserve source observations, ordered quantities, units, coordinate/frame meaning, time semantics, missingness and declared uncertainty where applicable. An unimplemented field or conversion must be reported as unsupported rather than silently inferred.

Evidence identity names the source record; operation identity names the versioned computation; execution identity names an invocation; result identity names its output; verification identity names a scoped check. These are integration requirements, not a claim that every standalone repository already implements all five record types.

Display names and repository locations do not rename packages, schemas, operation IDs, retained corpus keys or historical runtime pins. CIW integrations use the exact source revisions named in its runtime manifests and operating guides; a provider's current default branch is not a substitute for that binding. Published numerical records retain their original run scope.

## Technical references

- [Overview and runnable instructions](../README.md)
- [docs/PROTOCOL.md](PROTOCOL.md)
- [docs/INSTRUMENTS.md](INSTRUMENTS.md)
- [docs/ARCHITECTURE.md](ARCHITECTURE.md)

Private customer state, deployment configuration and calibration knowledge are outside this public component description. This repository is `AGPL-3.0-only`. That grant covers the workbench program only: it does not relicense pinned providers, and it does not apply to experiment outputs unless an output itself contains covered program code. Other repositories' licenses and source-data rights remain controlling; a shared stack identity is not a license grant or a change of repository visibility.

## Read-only exchange path

CIW's [instrument-exchange inspector](https://github.com/giasonpooni/Notation-Systems-Workbench/blob/main/docs/EXCHANGE.md)
checks supported `notation.instrument.*.v1` acquisition/runtime artifacts with
an explicitly pinned State Estimation Evaluation Testbed validator. It retains
full or explicitly unknown covariance and reports content/reference checks.
This is read-only conformance inspection: it does not import a native workspace,
run a scientific provider, admit source evidence or authenticate verification.
The operating guide records the producer/checker revisions and exact limits.
