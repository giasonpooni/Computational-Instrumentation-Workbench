# Notation Systems computational instrumentation stack

Notation Systems develops computational instrumentation and evidence infrastructure for industrial and cyber-physical systems. The engineering mandate is to connect source observations, explicit mathematical models, computation, inspection and governed state while retaining the evidence needed to reproduce and challenge a result.

The public repositories are components of this stack. Their scientific and engineering functions define their names. The existing physical-economy corpus, acquisition, policy and information-delivery capabilities remain part of the architecture; adding instruments does not replace them.

## Component map

This is the public repository inventory reviewed on 2026-09-20. “Executable” describes code present in the component; it does not mean production-qualified, physically validated or integrated with every other component. “Pinned” describes the CIW binding, which can differ from a provider's default branch. Local READMEs and versioned contracts specify exact supported behavior.

| Component | Responsibility | Present scope | CIW connection |
| --- | --- | --- | --- |
| [Computational Instrumentation Workbench](https://github.com/giasonpooni/Computational-Instrumentation-Workbench) | Operation, inspection and replay | Executable prototype | Host; oscillator, five external tool workflows and read-only exchange inspection |
| [Provenance Preserving Data Acquisition](https://github.com/giasonpooni/Provenance-Preserving-Data-Acquisition) | Source acquisition and observation lineage | Executable acquisition, storage and source adapters | Read-only exchange inspection; no native measurement adapter |
| [Streaming Telemetry Feature Extraction](https://github.com/giasonpooni/Streaming-Telemetry-Feature-Extraction) | Causal/offline signal conditioning, spectral/temporal features and stream-quality diagnostics | Specification seed; no executable operator, validator or tests | Registered planned provider; no CIW adapter or replay path |
| [Evidence and State Management](https://github.com/giasonpooni/Evidence-and-State-Management) | Evidence retention, state admission and release | Executable local rails and bounded domain implementations; demonstration corpora | No CIW persistence adapter |
| [Scientific Computation Runtime](https://github.com/giasonpooni/Scientific-Computation-Runtime) | Scientific workload execution and verification records | Executable runtime and state/evidence packages; backend-specific prerequisites | Read-only exchange inspection; no execution adapter |
| [Retrofitted Computational Instrumentation](https://github.com/giasonpooni/Retrofitted-Computational-Instrumentation) | Measurement-chain records and declared calibration | Executable host-side software with simulated examples | Pinned RCI calibration provider used by CIW |
| [Fluid State Reconstruction Testbed](https://github.com/giasonpooni/Fluid-State-Reconstruction-Testbed) | Fluid-state estimation and balance reconciliation | Executable experimental fluid toolkit | Pinned two-reservoir snapshot and covariance workflows |
| [Jacobian Sensitivity Propagation Testbed](https://github.com/giasonpooni/Jacobian-Sensitivity-Propagation-Testbed) | Local derivatives, sensitivity and covariance transport | Executable numerical testbed | Pinned covariance propagation provider |
| [Geometric Telemetry Engine](https://github.com/giasonpooni/Geometric-Telemetry-Engine) | Geometric reconciliation and tangent uncertainty | Executable experimental circle operation | Pinned circle investigation and replay |
| [Parameterized Lyapunov Stability Runtime](https://github.com/giasonpooni/Parameterized-Lyapunov-Stability-Runtime) | Quadratic Lyapunov evaluation | Executable runtime in development | Terminal workflow; separate bundles, outside shared session/viewport |
| [Construction State Estimator for BIM](https://github.com/giasonpooni/Construction-State-Estimator-for-BIM) | BIM evidence-to-decision computation | Executable experimental BIM runtime | Standalone; companion commitment and numerical adapters |
| [Schematics Retrieval Agent](https://github.com/giasonpooni/Schematics-Retrieval-Agent) | Typed schematic retrieval and kernel eligibility | Executable graph and optional companion adapters | Standalone; no CIW adapter |
| [Yield Weighted Inference Runtime](https://github.com/giasonpooni/Yield-Weighted-Inference-Runtime) | Inference-budget admission and settlement | Executable runtime in development | Standalone; no CIW adapter |
| [Geospatial State Visualization](https://github.com/giasonpooni/Geospatial-State-Visualization) | Geographic and temporal inspection | Executable browser client with synthetic provider | Separate client; no CIW session/replay connection |
| [Flat Torus Geodesic Reference](https://github.com/giasonpooni/Flat-Torus-Geodesic-Reference) | Exact flat-geometry reference and representation invariants | Executable mathematical reference | Standalone; versioned companion artifact |
| [Curved Surface Geodesic Sensitivity Runtime](https://github.com/giasonpooni/Curved-Surface-Geodesic-Sensitivity-Runtime) | Curvature-dependent path sensitivity | Executable numerical engine and tolerance experiments | Standalone; flat-reference companion boundary |
| [State Estimation Evaluation Testbed](https://github.com/giasonpooni/State-Estimation-Evaluation-Testbed) | Instrument-exchange validation and estimator evaluation scope | Executable contract validators; no estimator or evaluation runner | Pinned exchange conformance checker; no evaluation runner or session adapter |
| [Constraint Based State Reconciliation](https://github.com/giasonpooni/Constraint-Based-State-Reconciliation) | General constraint-reconciliation specification | Declarative invariant corpus only | No executable engine or adapter |
| [Covariance Geometry and Geodesic Testbed](https://github.com/giasonpooni/Covariance-Geometry-and-Geodesic-Testbed) | Geometry of covariance matrices | Planned scaffold; metadata checks only | No numerical implementation or adapter |
| [Intrinsic Surface Geodesics Testbed](https://github.com/giasonpooni/Intrinsic-Surface-Geodesics-Testbed) | Intrinsic paths on triangle meshes | Planned scaffold; metadata checks only | No numerical implementation or adapter |
| [Translation Surface Dynamics Explorer](https://github.com/giasonpooni/Translation-Surface-Dynamics-Explorer) | Trajectory dynamics on translation surfaces | Planned scaffold; metadata checks only | No numerical implementation or adapter |

The oscillator is built into CIW and is not a twenty-second repository. Three geometry repositories are planned scaffolds, Streaming Telemetry Feature Extraction is a specification seed, Constraint-Based State Reconciliation is declarative, and State Estimation Evaluation Testbed implements exchange validators without an estimator/evaluation runner.

## Implemented workbench paths

| Path | What is retained | Exact operating guide |
| --- | --- | --- |
| Synthetic oscillator → CIW → terminal/optional Godot | Numerical channels, selection, analysis and saved workspace | [Quickstart](quickstart.md) |
| RCI → FSRT through pinned subprocesses | Raw and calibrated evidence, two-reservoir snapshot, residuals and covariance | [Measurement and estimation](ADAPTERS.md) |
| Retained covariance → JSPT | Explicit mapping/Jacobian, ordered covariance and source/result relationships | [Covariance workflow](COVARIANCE.md) |
| Declared circle observations → GTE | Original observations, candidate, residuals, tangent uncertainty and replay inputs | [Geometric telemetry](GTE.md) |
| Declared plant/certificate → PLSR terminal workflow | Model and sample artifacts, verdicts and replay digests in separate bundles | [Lyapunov workflow](PLSR.md) |
| Acquisition/runtime exchange artifacts → pinned testbed validator → CIW inspector | Read-only conformance report, original parsed artifacts, byte digests and supplied-reference matches; no workspace import | [Exchange inspection](EXCHANGE.md) |

These are bounded paths. PLSR bundles are outside the shared session and viewport. The Godot client renders oscillator data; other external instruments expose the terminal/JSON paths in their guides. General live acquisition, executable streaming feature extraction, universal sensor fusion, GNSS/RTK processing and automatic equipment control are not capabilities established by these integrations.

Standalone companion relationships also exist: the flat-torus reference exports a versioned geometry artifact; CSE can bind companion commitments; SRA can call optional pinned numerical kernels. A commitment binding records identity and does not by itself compose scientific meaning or validate a measurement.

## Additional standalone numerical foundations

The following six repositories extend the stack with implemented standalone
numerical foundations, reviewed on 2026-09-21. Each includes a bounded Python
API, synthetic examples, local tests and an explicit optional exporter for the
existing `notation.instrument.result-artifact.v1` format. These are additions
to the component inventory, not additions to the implemented CIW paths above.

| Instrument | Implemented numerical foundation | Role in the existing stack |
| --- | --- | --- |
| [Time Base Reconciliation Runtime](https://github.com/giasonpooni/Time-Base-Reconciliation-Runtime) | Supplied affine clock mapping and correlated first-order time uncertainty | Derives event-time coordinates while PPDA retains source timestamps; consumers explicitly select the derived coordinate |
| [Observability and Identifiability Testbed](https://github.com/giasonpooni/Observability-Identifiability-Testbed) | Finite-horizon linear observability and local sensitivity/Fisher diagnostics | Diagnoses declared GSIE models and supplied JSPT sensitivities; owns neither estimation nor independent evaluation |
| [Metrological Calibration and Uncertainty Runtime](https://github.com/giasonpooni/Metrological-Calibration-Uncertainty-Runtime) | Applicable affine calibration and full correlated first-order uncertainty | Complements the measurement-chain boundary without replacing RCI's existing calibration operation or pins |
| [System Identification and Dynamics Testbed](https://github.com/giasonpooni/System-Identification-Dynamics-Testbed) | Fully observed discrete linear least squares and one-step evaluation | Produces candidate dynamics; model selection and use by GSIE remain explicit caller decisions |
| [Fault Detection and Isolation Runtime](https://github.com/giasonpooni/Fault-Detection-Isolation-Runtime) | Innovation NIS/whitening and deterministic CUSUM transitions | Interprets supplied residuals statistically; physical fault isolation and automatic response are not implemented |
| [Experiment Design and Sensor Placement Testbed](https://github.com/giasonpooni/Experiment-Design-Sensor-Placement-Testbed) | Finite candidate information ranking with D- and A-optimal criteria | Consumes declared sensitivities and noise models; returns advisory rankings without commanding acquisition |

The role column describes ownership and compatible inputs, not a live
cross-instrument execution path. The six example exports have exercised SET's
existing validator, pinned by their optional `exchange` dependencies to
[`bd261a765281a95312f7c91a3857233476294c5b`](https://github.com/giasonpooni/State-Estimation-Evaluation-Testbed/tree/bd261a765281a95312f7c91a3857233476294c5b).
That producer-side conformance does not establish a native CIW execution,
workspace or session adapter, an end-to-end replay path, or a SET evaluation
runner. Existing CIW bindings and historical source pins remain unchanged.

Exports retain explicitly mapped inputs and numerical outputs. Evidence,
operation, caller-supplied execution, result and verification identities remain
separate; supplied source revisions are labelled unattested and verification
references start empty. Neither export conformance nor successful numerical
execution establishes independent verification, physical validity, calibrated
measurement truth, evidence admission or actuation authority. Each repository's
numerical contract owns its exact assumptions and refusal conditions.

## Responsibility and authority

| Boundary | Owner and rule |
| --- | --- |
| Acquisition | Source adapters retain original material, extraction lineage and explicit missingness. |
| Evidence and state | The owning subsystem applies its declared admission/review rules. CIW workspaces, scientific canonical state and domain corpus state are distinct stores. |
| Numerical meaning | The domain engine owns its model, numerical method, assumptions, diagnostics and refusal conditions. |
| Invocation | CIW or another explicit caller binds the operation, source revision, inputs and execution environment. |
| Inspection | Clients display scientific records and derived views; render geometry does not become calculation input or state authority. |
| Verification | A check identifies its subject, method and claim scope. Execution success, a digest and a plausible plot are not independent verification. |

New workloads, representations and proof backends extend these boundaries through adapters. They do not create a second canonical write path or silently replace an established scientific engine.

## Interchange requirements

Integration must declare quantity order and units, coordinate frame and basis, observation time and support interval, source lineage, calibration applicability, uncertainty status, model assumptions and software binding to the extent required by the operation. Unknown uncertainty stays unknown; absence of a cross-covariance term does not establish independence. Frame conversion, resampling and calibration are named transformations with retained inputs.

Evidence, operation, execution, result and verification identities remain distinct. Reading a retained artifact does not authorize execution. Re-execution records a new invocation rather than replacing the evidence that produced the original result. Source revision, contract version and result identity answer different questions.

The [CIW protocol](PROTOCOL.md), [covariance contract](COVARIANCE.md) and each provider's operation contract remain authoritative for implemented fields. The separate [State Estimation Evaluation Testbed](https://github.com/giasonpooni/State-Estimation-Evaluation-Testbed) validates `notation.instrument.*` exchange artifacts; those schemas are not automatically interchangeable with CIW's `ciw.*` records. CIW now inspects those exchange artifacts through the [read-only conformance path](EXCHANGE.md), without translating them into native workspace or covariance records. A translation requires an explicit mapping and validation. This map does not introduce a new universal wire protocol.

The [Streaming Telemetry Feature Extraction draft contract](https://github.com/giasonpooni/Streaming-Telemetry-Feature-Extraction/blob/main/docs/CONTRACT.md) defines proposed telemetry-window, feature-record and stream-quality companion records. Scalar ordered features may be mapped explicitly to `notation.instrument.result-artifact.v1`; frequency axes, filter state and window policy remain referenced domain information. No executable schema, generic exchange mapping or CIW adapter is currently registered.

## Scientific use and qualification

The stack supports research and development across measurement chains, fluid systems, built assets, geometric sensitivity and declared dynamical models. Materials/process experimentation, robotics and compact GNSS instrumentation are application directions requiring their own models, adapters, calibration and experimental validation.

Numerical correctness, uncertainty calibration, model adequacy, hardware performance and operational authorization are separate claims. Every public capability statement should identify whether it is an implemented standalone operation, an exercised integration, a contract-only surface or a planned scaffold. Numerical results retain their model and run conditions; scaffold checks establish metadata consistency only.

## Public documentation and compatibility

Current entry points describe engineering roles, executable behavior, contracts, methods and qualified results. Earlier portfolio narratives and product metaphors do not define component scope. Historical measurements, source citations, frozen specifications and evidence-bearing records keep their original identities and limits. A display rename is not a schema migration.

Public documentation describes reusable interfaces. Customer state, private deployment configuration, proprietary calibration knowledge and internal generative planning remain outside that interface documentation. Existing licenses and source rights control reuse; this stack map neither relicenses repositories nor changes visibility.

When an integration changes, update the producer's role page, the consumer's operating guide and this map together. Identify the supported source pins, exact input/output contract, failure behavior and validation scope. An architecture arrow or a related-repository link alone is not an implemented integration.
