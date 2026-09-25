# Systems catalog

This page is the CIW-facing map of the surrounding Notation Systems tools. It
is an integration index, not a replacement for an independent repository's
README, license or scientific contract. Each linked repository remains
responsible for its own implementation and should be read at the exact
revision named by the applicable CIW manifest.

## Status vocabulary

| Status | Meaning |
| --- | --- |
| **Native CIW** | The operation is implemented in this repository and has a save/reopen or replay path. |
| **Pinned adapter** | CIW invokes an external provider at an explicit source/runtime pin and retains its result. |
| **Contract only** | The boundary or planned profile is documented, but no supported operation is registered. |
| **Pending** | The idea or provider needs implementation, environment resolution or independent validation. |

An integration is complete only when the operation, execution and result
identities remain distinct; units, frames, timing and uncertainty are declared;
the source/runtime pin is checked; and tests cover save, reopen, replay and
refusal behavior where the operation supports them.

## CIW-owned pages and operations

| CIW page | Responsibility |
| --- | --- |
| [INSTRUMENTS.md](INSTRUMENTS.md) | User-facing commands, operation contracts, inputs, outputs and limits. |
| [INTEGRATION_COVERAGE.md](INTEGRATION_COVERAGE.md) | Exercised paths, profile coverage, validation evidence and next connections. |
| [WORKBENCH_ASSEMBLY.md](WORKBENCH_ASSEMBLY.md) | Multi-provider session assembly and deployment layout. |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Admission, persistence, identity separation, replay and read-only boundaries. |
| [CONTRACT_FOUNDATIONS.md](CONTRACT_FOUNDATIONS.md) | Typed exchange and machine-manifest foundations. |
| [PROVIDER_AVAILABILITY.md](PROVIDER_AVAILABILITY.md) | Exact local checkout provisioning and access limitations. |

The built-in oscillator, telemetry, calibrated process, identified-design,
covariance, geometry, energy and machine-manifest workflows are owned by CIW.
Their detailed operation status is maintained in the operation catalogue rather
than repeated in this systems page.

## Surrounding systems

The links below go to each independent repository's README. The CIW column
describes the current boundary; it does not claim that the upstream repository
is part of this checkout.

| System and independent README | CIW role | Status / integration page |
| --- | --- | --- |
| [Provenance-Preserving-Data-Acquisition](https://github.com/giasonpooni/Provenance-Preserving-Data-Acquisition#readme) | Retain source observations and acquisition lineage. | Pinned telemetry and calibrated paths; [TELEMETRY.md](TELEMETRY.md) |
| [Streaming-Telemetry-Feature-Extraction](https://github.com/giasonpooni/Streaming-Telemetry-Feature-Extraction#readme) | Bounded causal window features. | Pinned telemetry path; [TELEMETRY.md](TELEMETRY.md) |
| [Geometric-State-Inference-Engine](https://github.com/giasonpooni/Geometric-State-Inference-Engine#readme) | State prediction/update and covariance diagnostics. | Pinned telemetry and calibrated paths; [INTEGRATION_COVERAGE.md](INTEGRATION_COVERAGE.md) |
| [State-Estimation-Evaluation-Testbed](https://github.com/giasonpooni/State-Estimation-Evaluation-Testbed#readme) | Scoped exchange conformance and replay checks. | Read-only inspection; [EXCHANGE.md](EXCHANGE.md) |
| [Constraint-Based-State-Reconciliation](https://github.com/giasonpooni/Constraint-Based-State-Reconciliation#readme) | Held/accepted conservation candidates. | Pinned calibrated path; [CALIBRATED_OBSERVABLE.md](CALIBRATED_OBSERVABLE.md) |
| [Time-Base-Reconciliation-Runtime](https://github.com/giasonpooni/Time-Base-Reconciliation-Runtime#readme) | Clock mapping and time uncertainty. | Pinned calibrated path; [CALIBRATED_OBSERVABLE.md](CALIBRATED_OBSERVABLE.md) |
| [Metrological-Calibration-Uncertainty-Runtime](https://github.com/giasonpooni/Metrological-Calibration-Uncertainty-Runtime#readme) | Time-applicable calibration and correlated uncertainty. | Pinned calibrated path; [CALIBRATED_OBSERVABLE.md](CALIBRATED_OBSERVABLE.md) |
| [Observability-Identifiability-Testbed](https://github.com/giasonpooni/Observability-Identifiability-Testbed#readme) | Observability and local identifiability gates. | Pinned calibrated/design paths; [IDENTIFIED_DESIGN.md](IDENTIFIED_DESIGN.md) |
| [Fault-Detection-Isolation-Runtime](https://github.com/giasonpooni/Fault-Detection-Isolation-Runtime#readme) | Innovation and residual diagnostics. | Pinned calibrated path; [CALIBRATED_OBSERVABLE.md](CALIBRATED_OBSERVABLE.md) |
| [System-Identification-Dynamics-Testbed](https://github.com/giasonpooni/System-Identification-Dynamics-Testbed#readme) | Bounded point-model identification. | Pinned identified-design path; [IDENTIFIED_DESIGN.md](IDENTIFIED_DESIGN.md) |
| [Experiment-Design-Sensor-Placement-Testbed](https://github.com/giasonpooni/Experiment-Design-Sensor-Placement-Testbed#readme) | Candidate information and cost ranking. | Pinned advisory path; [IDENTIFIED_DESIGN.md](IDENTIFIED_DESIGN.md) |
| [Yield-Weighted-Inference-Runtime](https://github.com/giasonpooni/Yield-Weighted-Inference-Runtime#readme) | Separate advisory computation-token decision. | Pinned identified-design path; [IDENTIFIED_DESIGN.md](IDENTIFIED_DESIGN.md) |
| [Jacobian-Sensitivity-Propagation-Testbed](https://github.com/giasonpooni/Jacobian-Sensitivity-Propagation-Testbed#readme) | Explicit covariance transport through a declared map. | Pinned covariance path; [COVARIANCE.md](COVARIANCE.md) |
| [Parameterized-Lyapunov-Stability-Runtime](https://github.com/giasonpooni/Parameterized-Lyapunov-Stability-Runtime#readme) | Supported certificate evaluation. | Pinned standalone operation; [PLSR.md](PLSR.md) |
| [Scientific-Computation-Runtime](https://github.com/giasonpooni/Scientific-Computation-Runtime#readme) | Registered integer execution and optional proof backend. | Contract/selected operation; [PROVED_HEAT.md](PROVED_HEAT.md) |
| [Evidence-and-State-Management](https://github.com/giasonpooni/Evidence-and-State-Management#readme) | Candidate evidence review and retention. | Companion boundary; [WORKBENCH_ASSEMBLY.md](WORKBENCH_ASSEMBLY.md) |
| [Geospatial-State-Visualization](https://github.com/giasonpooni/Geospatial-State-Visualization#readme) | Read-only geographic and temporal projection. | Pinned projection; [INTEGRATED_MODULES.md](INTEGRATED_MODULES.md) |
| [Schematics-Retrieval-Agent](https://github.com/giasonpooni/Schematics-Retrieval-Agent#readme) | Evidence-backed asset and signal-binding proposals. | Contract/manifest direction; [CONTRACT_FOUNDATIONS.md](CONTRACT_FOUNDATIONS.md) |
| [Retrofitted-Computational-Instrumentation](https://github.com/giasonpooni/Retrofitted-Computational-Instrumentation#readme) | External sensing and retrofit observation modules. | Contract/adapter direction; [WORKBENCH_ASSEMBLY.md](WORKBENCH_ASSEMBLY.md) |
| [Construction-State-Estimator-for-BIM](https://github.com/giasonpooni/Construction-State-Estimator-for-BIM#readme) | Built-asset state estimation and model context. | Contract/adapter direction; [STACK.md](STACK.md) |
| [Covariance-Geometry-and-Geodesic-Testbed](https://github.com/giasonpooni/Covariance-Geometry-and-Geodesic-Testbed#readme) | Affine-invariant covariance geometry. | Native provider package; [GEOMETRY_RESEARCH.md](GEOMETRY_RESEARCH.md) |
| [Intrinsic-Surface-Geodesics-Testbed](https://github.com/giasonpooni/Intrinsic-Surface-Geodesics-Testbed#readme) | Bounded mesh-edge geodesic baseline. | Native provider package; [GEOMETRY_RESEARCH.md](GEOMETRY_RESEARCH.md) |
| [Translation-Surface-Dynamics-Explorer](https://github.com/giasonpooni/Translation-Surface-Dynamics-Explorer#readme) | Exact rational square-tiled trajectory prefixes. | Native provider package; [GEOMETRY_RESEARCH.md](GEOMETRY_RESEARCH.md) |
| [Flat-Torus-Geodesic-Reference](https://github.com/giasonpooni/Flat-Torus-Geodesic-Reference#readme) | Exact area-one lattice/winding reference. | Pinned reference provider; [GEODESIC_REFERENCES.md](GEODESIC_REFERENCES.md) |
| [Curved-Surface-Geodesic-Sensitivity-Runtime](https://github.com/giasonpooni/Curved-Surface-Geodesic-Sensitivity-Runtime#readme) | Constant-curvature Jacobi transfer reference. | Pinned reference provider; [GEODESIC_REFERENCES.md](GEODESIC_REFERENCES.md) |
| [Instrument-Conformance-and-Replay-Harness](https://github.com/giasonpooni/Instrument-Conformance-and-Replay-Harness#readme) | Versioned external profile conformance. | Pinned validator; [EXCHANGE.md](EXCHANGE.md) |

The Julia Tsit5 oscillator is the first registered Julia provider seam; its
contract and offline retention path are implemented in [JULIA_OSCILLATOR.md](JULIA_OSCILLATOR.md).
JuliaControl, JuMP, ModelingToolkit, RxInfer, OpenFOAM, FreeCAD, TensorFlow
Lattice, an FPGA toolchain and the broader persistent Julia scientific runtime
remain candidate providers. A registered seam does not pass the environment or
physical validation gate by itself.

## The collapse rule for loose tooling

The project is collapsing *interfaces and evidence paths*, not erasing
independent source repositories:

1. Keep upstream source, README, license and scientific tests in the provider
   repository.
2. Add one CIW adapter or exchange producer with a versioned operation contract.
3. Pin the provider commit, source tree, interpreter and relevant dependencies
   in a CIW manifest.
4. Retain typed inputs, units, frames, clock declarations, uncertainty status,
   provenance and refusal behavior in the CIW execution/result records.
5. Add save, reopen, replay and independent validation before labeling the
   provider integrated.

Provider code is not copied into CIW merely to make a table entry look
complete. A README link is discoverability; a pinned adapter plus exercised
records is integration.

## Pin and validation ledger

The authoritative pins are the JSON manifests under src/ciw/ and the
provider-specific operating guides under docs/. The current native reference
pins include:

| Provider | Revision |
| --- | --- |
| Flat Torus Geodesic Reference | dc918562cd9e351a65475d29f46963c9f2fd7db8 |
| Curved Surface Geodesic Sensitivity Runtime | bbc535af29c30997e56fd120320c570830676462 |
| Instrument Conformance and Replay Harness | dc4d826ecd1f28c1d55b724618380ce44e58bedd |

The Julia worker scaffold is deliberately not presented as resolved until its
environment, lockfile and CI lane are reproducible. Physical calibration,
FPGA programming, MCP authority, and independent ICRH validation remain
separate gates.
