# Documentation index for Notation-Systems-Workbench

The root [README](../README.md) is the macro entrypoint. This index points to
the page that owns each kind of detail so status and contracts do not drift
across several copies.

The technical name is **Computational Instrumentation Workbench (CIW)**. The
Python distribution is `computational-instrumentation-workbench`; Python code
imports `ciw`, and the command-line entry point is also `ciw`.

## Start here

| Need | Page |
| --- | --- |
| Product scope, operating model and scientific workspace | [Workbench overview](WORKBENCH_OVERVIEW.md) |
| Current provider map and loose-tool collapse rule | [Systems catalog](SYSTEMS_CATALOG.md) |
| Executable implementation architecture | [Architecture](ARCHITECTURE.md) |
| Multi-provider assembly and local deployment | [Workbench assembly](WORKBENCH_ASSEMBLY.md) |
| Current executable paths and remaining gates | [Integration coverage](INTEGRATION_COVERAGE.md) |
| User-facing instruments and exact commands | [Instrument catalogue](INSTRUMENTS.md) |
| Oscillator demo, inspection and reopen commands | [Oscillator operator card](OSCILLATOR_OPERATOR.md) |

## Contracts and operations

- [Protocol and record identities](PROTOCOL.md)
- [Contract foundations and typed exchange](CONTRACT_FOUNDATIONS.md)
- [Generic adapters](ADAPTERS.md)
- [Covariance provenance and replay](COVARIANCE.md)
- [Retained telemetry](TELEMETRY.md) and [shared telemetry](SHARED_TELEMETRY.md)
- [Calibrated observable process](CALIBRATED_OBSERVABLE.md)
- [Identified and budgeted observation](IDENTIFIED_DESIGN.md)
- [Machine manifest workflow](CONTRACT_FOUNDATIONS.md#machine-manifest-operation)
- [Project graph operation](CONTRACT_FOUNDATIONS.md#project-graph-operation)
- [Energy-to-accuracy bench](ENERGY_ACCURACY.md)
- [Variational free-energy sensor fusion](VARIATIONAL_FREE_ENERGY.md)
- [Geodesic references](GEODESIC_REFERENCES.md) and [geometry research](GEOMETRY_RESEARCH.md)
- [PLSR](PLSR.md), [registered heat proof](PROVED_HEAT.md), and [Julia/SP1 direction](JULIA_SP1.md)
- [Exchange inspection](EXCHANGE.md)

## Development and availability

- [Development guide](DEVELOPMENT.md)
- [Provider availability and exact checkout provisioning](PROVIDER_AVAILABILITY.md)
- [Stack map](STACK.md) and [stack role](STACK_ROLE.md)
- [Diagram atlas](DIAGRAMS.md)
- [Quickstart](quickstart.md)
- [Deployment](../deploy/README.md)

Historical audits remain linked from the root for context. They do not override
the current operation catalogue, integration matrix or provider manifests.
