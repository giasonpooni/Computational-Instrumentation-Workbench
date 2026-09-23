# Executable integration coverage

Reviewed 2026-09-23 against CIW source, its runtime manifests, ICRH profiles,
and the public repository catalog. This matrix distinguishes an executable
handoff from a matching schema, a digest attachment, or a documentation link.
The public catalog contains 29 repositories including CIW and ICRH. Integration
coverage is measured by exercised operations and replay paths, not that count.

The active delivery emphasis is assembling these tools into one workbench:
shared sources and declared contexts, callable operations, retained candidate
state and covariance, and one execution/result history. The
[workbench assembly guide](WORKBENCH_ASSEMBLY.md) distinguishes the current
session connection, native SRA companion calls, SCR integer dispatch,
PPDA acquisition, GSV geography and CSE quantity conditioning from their
remaining physical-stream and geometry compositions.
An additional standalone pipeline does not by itself provide that common
operating point.

## Shared operating session

The [Workbench desktop tab](EXPERIMENT_VIEW.md) projects all seventeen shared workflow
kinds through `experiment.inspect`: retained measurements, state/covariance,
residuals, native dependencies, evidence and verification. It follows committed
session changes and keeps replay occurrences separate. This read-only display
adds no estimator or scientific operation; existing SET/ICRH profiles remain the
numerical conformance boundaries. Physical acquisition and arbitrary algebraic or
topological workload dispatch remain pending.

The live Session now hosts `ciw.telemetry.v1`, `ciw.calibrated-observable.v1` and
`ciw.identified-design.v1`, plus `ciw.calibrated-window.v1`,
`ciw.schematic-assessment.v1`, `ciw.numerical-heat.v1`, `ciw.schematic-companions.v1`,
`ciw.bim-quantity.v1`, `ciw.acquired-dataset.v1`,
`ciw.acquired-calibrated-window.v1`, `ciw.residual-monitor.v1`,
`ciw.measurement-chain.v1`, `ciw.geometric-circle.v1` and
`ciw.identified-stability.v1`, `ciw.flat-torus-reference.v1` and
`ciw.curved-path-transfer.v1`, through `operation.list/execute`.
`source.*`, `bundle.*` and `fusion.list` expose retained inputs, native bundles
and candidate contexts; result and execution lists include those native records
alongside existing session operations. Workspace format 3 retains this content,
and reopening validates it without executing or rebinding a provider.

PPDA batch projection and STFE window/quality/feature receipts now have native
instrument views. Telemetry and calibrated process bundles can use separate
ESM bindings in the same session; each undergoes fresh replay/policy checks before
optional candidate-only retention. Stable observation identity remains separate
from fresh execution identity. See [shared telemetry](SHARED_TELEMETRY.md).

The [assembly operating guide](WORKBENCH_ASSEMBLY.md#operate-the-shared-session)
documents startup bindings, request payloads and change events. Shared desktop
inspection, SRA companion calls, SCR dispatch, PPDA bounded acquisition, GSV's
read-only provider and CSE quantity mapping are implemented. Hardware acquisition,
generic continuous fusion and surveyed-frame geometry remain delivery work. Scientific calculations
keep their native owners; the table below distinguishes implemented ICRH profiles from pending coverage.

## Existing executable paths

| Path | CIW execution and retained record | Independent harness coverage | Remaining composition gap |
| --- | --- | --- | --- |
| PPDA → STFE → GSIE → SET | Shared `ciw.telemetry.v1` operation and existing standalone commands; exact source bytes, full temporal covariance, causal scalar mean, predict/update and ESM candidate handoff | ICRH `telemetry-to-state.v1`, original/replay/adversarial fixtures | This scalar path does not yet consume TBRT/MCUR transformations or evaluate observability. |
| PPDA → STFE → GSIE → CBSR → SET | Same telemetry session with optional affine-exact reconciliation receipt | ICRH `telemetry-reconciled.v1` | New decisions must preserve accepted/held/refused distinctions. |
| TBRT → MCUR → STFE → GSIE → SET | Shared `ciw.calibrated-window.v1`; raw device samples, affine map/profile, full joint time/value/parameter covariance, native compatibility check and window-state context | ICRH `calibrated-window-to-state.v1`; actual original/replay and resealed covariance/validity/compatibility faults | Scalar stationary hold on a nominal grid; hardware acquisition and ESM handoff remain separate work. |
| PPDA retained records → calibrated window | Shared `ciw.acquired-calibrated-window.v1`; exact observation/record/document/snapshot-row selection, typed declaration, unchanged native child and separate mapping verification | ICRH `acquired-calibrated-window.v1`; independent PPDA lineage and child-window checks with original/replay pairs | Bounded retained snapshots; no implicit ordering, missing clock inference or hardware polling. |
| Retained GSIE innovations → FDIR/OIT | Shared `ciw.residual-monitor.v1`; explicit ordered windows, native residual covariance, observability gate and CUSUM transitions | ICRH `residual-monitor.v1`; scalar numerical oracle, held observability, replay aliases, unknown covariance and ambiguity checks | Diagnostic candidates only; unknown cross-window dependence blocks statistical alarm authority and unique isolation. No posterior feedback. |
| FSRT → TBRT → MCUR → OIT → GSIE → CBSR → FDIR → SET | `ciw calibrated-observable create/inspect/replay`; two calibrated channels, retained clock uncertainty, observability gate and declared residual covariance | ICRH `calibrated-observable-telemetry.v1`; original, replay, held, ambiguous and refusal fixtures | Synthetic stationary hold; no calibrated stream window or identified dynamics in this profile. |
| Retained calibrated experiment → SIDT → OIT → GSIE → EDSPT → YWIR | `ciw identified-design create/inspect/replay`; freshly checked upstream prior, identified point model, candidate observability, future conditional covariance, cost-constrained reduction and separate advisory token decision | ICRH [`identified-budgeted-observation.v1`](https://github.com/giasonpooni/Instrument-Conformance-and-Replay-Harness/blob/main/profiles/identified-budgeted-observation.v1.json); retained original/replay/token-denial fixtures | Parameter uncertainty remains unknown; selection is advisory and cannot dispatch acquisition or admit ESM state. |
| RCI → FSRT → JSPT | Shared `ciw.measurement-chain.v1`; unchanged native investigation, raw/calibrated evidence, posterior/reconciled covariance and explicit quantity map; inner executions/results join common history | CIW native tests cover calibration, mapped covariance, original/replay/held/fault fixtures; independent ICRH profile pending | Explicit independent simultaneous channels; general cross-provider state-to-quantity mapping remains separate. |
| Declared circle → GTE | Shared `ciw.geometric-circle.v1`; observations, full joint/tangent covariance, held candidate and native replay | CIW geometric/covariance oracle and binding tests; independent ICRH profile pending | Declared circle in one plane; no surveyed BIM frame mapping or inferred constraint. |
| Retained SIDT model + GSIE state + certificate → PLSR | Shared `ciw.identified-stability.v1`; exact retained model/state selection, discrete interval and native sealed artifact/verdict | CIW quadratic/discrete decrease, upstream binding and inconclusive/held tests; independent ICRH profile pending | Conditional on the identified point model; unknown parameter covariance remains unknown and state covariance is context, not a probabilistic certificate. |
| PPDA/SCR exchange artifact → SET → CIW inspector | `ciw exchange inspect`; unchanged artifacts, byte digests and conformance report | Actual producer integration tests in CIW | Read-only inspection does not dispatch an SCR workload or import native state. |
| Retained SRA assessment → SRA/JSPT/PLSR | Shared `ciw.schematic-companions.v1`, native local calls and explicit upstream result edge | ICRH `schematic-companions.v1`; scalar derivative/covariance/Lyapunov and binding checks | Local continuous linear surrogate only; no equilibrium or nonlinear region claim. |
| IFC + declared scalar observation → CSE | Shared `ciw.bim-quantity.v1`, native conditioning, rollback and replayed ledger | ICRH `bim-quantity.v1`; conditioning oracle, ledger/world commitments and held/refused cases | Quantity-only model; surveyed-frame geometry remains separate. |
| Retained snapshots → PPDA/SCOUT | Shared `ciw.acquired-dataset.v1`, native incremental acquisition, evidence pool and checkpoints | ICRH `acquired-dataset.v1`; lineage/cursor/pool reconstruction and replay bindings | Calibrated conversion now uses explicit retained selection; hardware polling remains separate. |
| Declared geographic context → GSV | `spatial.inspect`, exact source bytes through native GSV provider/WorldStore | ICRH geographic declaration checks; actual CIW WebSocket/provider tests | Source-only CRS84 nodes with declared constant states; no inferred geometry or estimator. |
| Flat lattice and winding → FTR | Shared `ciw.flat-torus-reference.v1`; native trajectory, geometry digest and fresh replay | ICRH `flat-torus-reference.v1`; analytic lattice/closure and retained pair checks | Area-one flat quotient; no embedded torus, physical units or observed uncertainty. |
| Declared constant curvature → CSG | Shared `ciw.curved-path-transfer.v1`; native Jacobi transfer, separation and declared covariance propagation | ICRH `curved-path-transfer.v1`; constant-curvature oracle, covariance and replay binding checks | Curvature profile only; no embedded path, surveyed geometry or calibrated sensor claim. |
| Declared integer field → SCR/SP1 | Shared `ciw.proved-heat.v1`; native execution, registered guest proof, full-ELF verification, exact retained proof bytes and fresh replay | Separate verifier invocation through pinned SCR/SP1; installed-wheel real-proof gate with corrupted-proof rejection; no separate ICRH proved-heat profile | Bounded integer arithmetic only. Offline consistency is not fresh cryptographic verification; Julia and F2 topology remain planned. |

“Independent harness” means a separately implemented conformance checker. It
does not mean an independent physical measurement or independent validation of
the scientific algorithm. SET's replay receipts retain their declared scope.

The executable sources are [`telemetry.py`](../src/ciw/telemetry.py),
[`calibrated_window.py`](../src/ciw/calibrated_window.py),
[`acquired_window.py`](../src/ciw/acquired_window.py),
[`residual_monitor.py`](../src/ciw/residual_monitor.py),
[`calibrated_observable.py`](../src/ciw/calibrated_observable.py),
[`identified_design.py`](../src/ciw/identified_design.py),
[`measurement_chain.py`](../src/ciw/measurement_chain.py),
[`geometric_circle.py`](../src/ciw/geometric_circle.py),
[`identified_stability.py`](../src/ciw/identified_stability.py),
[`investigation.py`](../src/ciw/investigation.py),
[`covariance_workflow.py`](../src/ciw/covariance_workflow.py),
[`geodesic.py`](../src/ciw/geodesic.py), [`plsr.py`](../src/ciw/plsr.py), and
[`exchange.py`](../src/ciw/exchange.py). Exact source revisions belong to their
checked-in runtime manifests; a provider's current default branch cannot
silently replace one. RCI's default branch at this review does not contain the
calibration adapter present at its CIW pin; the
[availability note](WORKBENCH_ASSEMBLY.md#source-and-runtime-availability)
records the exact revisions. ICRH profiles and trusted runners live in
[its profiles](https://github.com/giasonpooni/Instrument-Conformance-and-Replay-Harness/tree/main/profiles)
and [scripts](https://github.com/giasonpooni/Instrument-Conformance-and-Replay-Harness/tree/main/scripts).

## Delivered identified observation decision

**Implemented: `identified-budgeted-observation.v1`.** The executable
sequence retains the complete calibrated session and freshly replays that
upstream evidence before consuming it. SIDT fits a declared fully observed
discrete model; OIT gates the candidate observation model; GSIE predicts a
conditional Gaussian state; EDSPT ranks declared next observations by expected
uncertainty reduction subject to cost; YWIR records an advisory inference-token
budget decision. The [operating guide](IDENTIFIED_DESIGN.md) records exact
commands, eleven provider pins, expected numerical results and the installed
wheel gate. The [ICRH profile](https://github.com/giasonpooni/Instrument-Conformance-and-Replay-Harness/blob/main/profiles/identified-budgeted-observation.v1.json)
checks graph, uncertainty, budget and replay bindings. Original/replay execution
and independent content inspection have been exercised; the dedicated installed
gate runs the normal and adversarial cases without allowing skipped integration
tests. These are scoped computational checks, not physical model qualification.

The scientific and resource decisions answer different questions. Measurement
cost and measurement budget belong to the experiment-design declaration.
YWIR's existing token budgets are not prices, elapsed acquisition time or a
hardware spending capability. SIDT parameter covariance remains explicitly
unknown; this operation's uncertainty scope is
`conditional_on_identified_point_model`. Conditioning on fitted matrices does
not estimate or marginalize parameter uncertainty. The decision neither places
an equipment order nor starts a measurement.

## Next connections inside the workbench

| Order | Existing tools to connect | Small executable result | Evidence required before calling it integrated |
| --- | --- | --- | --- |
| 1 | PPDA/RCI, TBRT, MCUR, STFE, GSIE, FDIR | Shared acquisition and a calibrated manufacturing-cycle window producing a retained feature, state and drift residual | Explicit window support and time mapping; transformation/window compatibility; full temporal covariance and raw lineage; nonlinear calibration/mean order refusal; stale evidence and drift fixtures. |
| 2 | SRA, JSPT, PLSR, CIW | Retained typed schematic in the shared workspace, eligibility and actual companion call records | Reuse SRA's graph and routing; bind current Jacobian and declaration before dependent calls; unknown plant remains `UNRESOLVED`; a CIW entry point and ICRH profile. |
| 3 | SCR, SET, CIW | One bounded deterministic heat-diffusion or structural workload dispatched from a retained workspace request | Pin the descriptor, native runner, arithmetic and input bytes; separate simulation result from measurement; missing backend, overflow and replay fixtures. |
| 4 | GSV, CIW | Read-only panel projecting one supported retained spatial result and selected context into GSV's provider interface | Preserve evidence/result references and frame/time basis; refuse incompatible comparisons; demonstrate that view changes cannot change scientific input. |
| 5 | CSE, RCI, GTE, JSPT, CBSR | Frame-bound geometry/BIM inspection using measured quantities, local covariance propagation and declared construction constraints | Explicit surveyed/design frames and source authority, units, one supported constraint and analytic oracle; retain held candidate and residuals; a complete CIW operation/profile. |
| 6 | SIDT or JSPT, GSIE, PLSR | A selected retained model and compatible state passed to the existing Lyapunov evaluator | Bind discrete/continuous convention, sample period, state order, equilibrium, supplied certificate and margin; preserve `NUMERICAL_INCONCLUSIVE`; new ICRH profile. |

These six rows track assembly targets and their remaining scope. Row 6 is now
delivered for an explicitly selected discrete SIDT model and GSIE prediction,
with a supplied quadratic certificate. The common measurement-chain and GTE
operations also expose their native results in the shared session; surveyed
frame composition in row 5 remains pending. See [shared module operations](REMAINING_MODULES.md).
The affine TBRT/MCUR/STFE
window part of row 1 is delivered through [calibrated windows](CALIBRATED_WINDOW.md);
the [acquired stream path](ACQUIRED_STREAM.md) now adds exact PPDA record selection
and native FDIR/OIT diagnostics over retained GSIE innovations. Physical polling,
cross-window covariance and qualified physical drift diagnosis remain outstanding.
The typed assessment
portion of row 2 and native integer heat workload in row 3 are delivered through
[shared SRA/SCR workloads](DECLARED_WORKLOADS.md), with two ICRH profiles. SRA
companion execution is now delivered through [integrated modules](INTEGRATED_MODULES.md),
alongside PPDA snapshots, CSE quantities and GSV source views. SET support for
these schemas remains outstanding. Extend the
three demonstrations—process balance, manufacturing cycle and geometry/BIM
inspection—inside the common workspace through these contracts. Each result
should share source/context selection and retained history with the tools that
produced and consume it.

## Provider capability and remaining composition

| Component | Existing capability checked in this review | Why a further integration is meaningful |
| --- | --- | --- |
| [SIDT](https://github.com/giasonpooni/System-Identification-Dynamics-Testbed) | `fit_lti`, `evaluate_one_step`, rank diagnostics and bound declared-identification adapter | Identified-design binds training evidence and the current prior; identified-stability consumes the selected model/state. Physical holdout validation remains separate work. |
| [EDSPT](https://github.com/giasonpooni/Experiment-Design-Sensor-Placement-Testbed) | Finite candidate Fisher information, D-/A-optimal ranking, coordinate checks and cost-budgeted next observation | Identified-design now binds the point model, state, reduction baseline and observation budget; multi-observation portfolio optimization is outside this operation. |
| [YWIR](https://github.com/giasonpooni/Yield-Weighted-Inference-Runtime) | Advisory decisions, one-use reserve/settle/cancel operations and bound observation-design token adapter | Identified-design retains an advisory token receipt; it cannot certify the experiment or stand in for measurement cost, and creates no spending reservation. |
| [CSE](https://github.com/giasonpooni/Construction-State-Estimator-for-BIM) | Native IFC quantity conditioning, invariant rollback and execution ledger | CIW runs the native session and ledger replay; measured geometry still needs surveyed-frame correspondence. |
| [SRA](https://github.com/giasonpooni/Schematics-Retrieval-Agent) | Typed schematic queries, eligibility and pinned JSPT-to-PLSR companion calls | CIW binds selected assessment and local model calls; state-estimator and physical-plant semantics remain separate. |
| [SCR](https://github.com/giasonpooni/Scientific-Computation-Runtime) | Explicit execution specifications, native dispatch, scientific workloads and exchange exports | CIW now invokes native integer diffusion with exact commitments and a host-bound executable. Additional descriptors and physical-model semantics need separate contracts. |
| [GSV](https://github.com/giasonpooni/Geospatial-State-Visualization) | Browser provider interface, geographic/temporal inspection and comparison checks | CIW declared CRS84 sources now enter its native provider; local laboratory/BIM coordinates still require explicit mappings. |

No research/dependency fork is counted as an instrument merely because it is in
the account. The covariance-geometry, intrinsic-surface and translation-surface
scaffolds remain deferred until a GSIE, GTE or CBSR workload needs a specific
operation. The flat-torus and curved-surface references now execute as bounded shared-session
operations; see [geodesic references](GEODESIC_REFERENCES.md). The first retains an
area-one flat quotient-torus trajectory, and the second a constant-curvature Jacobi
transfer and declared starting covariance. Neither creates measured geometry or
a state-estimator context. Additional surface, mesh and topology workloads remain
separate work.

ESM stays at candidate-evidence retention. A replayable result, conformance
receipt or advisory selected observation grants neither canonical-state
admission nor equipment actuation. Evidence, operation, execution, result and
verification identities remain separate across every added connection.

## Delivery rule

A new integration must ship a callable CIW entry point, exact provider pins,
explicit input/output mapping, a retained original, a fresh replay, meaningful
refusal fixtures and an ICRH profile. Extend existing native records where their
meaning matches; retain versioned mappings where it does not. Update this
matrix, the provider's boundary and the operating guide with the same change.
An optional integration test that skipped because its provider was unavailable
does not count as execution evidence.
