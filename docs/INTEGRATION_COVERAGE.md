# Executable integration coverage

Reviewed 2026-09-22 against CIW source, its runtime manifests, ICRH profiles,
and the public repository catalog. This matrix distinguishes an executable
handoff from a matching schema, a digest attachment, or a documentation link.
The public catalog contains 29 repositories including CIW and ICRH. Integration
coverage is measured by exercised operations and replay paths, not that count.

The active delivery emphasis is assembling these tools into one workbench:
shared sources and declared contexts, callable operations, retained candidate
state and covariance, and one execution/result history. The
[workbench assembly guide](WORKBENCH_ASSEMBLY.md) distinguishes the current
session connection from the pending acquisition, SRA, SCR, GSV and CSE work.
An additional standalone pipeline does not by itself provide that common
operating point.

## Shared operating session

The live Session now hosts `ciw.telemetry.v1`, `ciw.calibrated-observable.v1` and
`ciw.identified-design.v1`, plus `ciw.calibrated-window.v1`, through the existing `operation.list/execute` surface.
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
documents startup bindings, request payloads and change events. This increment
adds the common session boundary. It does not claim a new graphical interface,
live acquisition, generic continuous fusion, SRA graph execution, SCR dispatch,
GSV provider connection or CSE mapping. Those existing systems enter through the
delivery work below. Scientific calculations and their existing ICRH profiles
remain unchanged by the registry.

## Existing executable paths

| Path | CIW execution and retained record | Independent harness coverage | Remaining composition gap |
| --- | --- | --- | --- |
| PPDA → STFE → GSIE → SET | Shared `ciw.telemetry.v1` operation and existing standalone commands; exact source bytes, full temporal covariance, causal scalar mean, predict/update and ESM candidate handoff | ICRH `telemetry-to-state.v1`, original/replay/adversarial fixtures | This scalar path does not yet consume TBRT/MCUR transformations or evaluate observability. |
| PPDA → STFE → GSIE → CBSR → SET | Same telemetry session with optional affine-exact reconciliation receipt | ICRH `telemetry-reconciled.v1` | New decisions must preserve accepted/held/refused distinctions. |
| TBRT → MCUR → STFE → GSIE → SET | Shared `ciw.calibrated-window.v1`; raw device samples, affine map/profile, full joint time/value/parameter covariance, native compatibility check and window-state context | ICRH `calibrated-window-to-state.v1`; actual original/replay and resealed covariance/validity/compatibility faults | Scalar stationary hold on a nominal grid; live acquisition, window observability, drift assessment and ESM handoff remain separate work. |
| FSRT → TBRT → MCUR → OIT → GSIE → CBSR → FDIR → SET | `ciw calibrated-observable create/inspect/replay`; two calibrated channels, retained clock uncertainty, observability gate and declared residual covariance | ICRH `calibrated-observable-telemetry.v1`; original, replay, held, ambiguous and refusal fixtures | Synthetic stationary hold; no calibrated stream window or identified dynamics in this profile. |
| Retained calibrated experiment → SIDT → OIT → GSIE → EDSPT → YWIR | `ciw identified-design create/inspect/replay`; freshly checked upstream prior, identified point model, candidate observability, future conditional covariance, cost-constrained reduction and separate advisory token decision | ICRH [`identified-budgeted-observation.v1`](https://github.com/giasonpooni/Instrument-Conformance-and-Replay-Harness/blob/main/profiles/identified-budgeted-observation.v1.json); retained original/replay/token-denial fixtures | Parameter uncertainty remains unknown; selection is advisory and cannot dispatch acquisition or admit ESM state. |
| RCI → FSRT → JSPT | `ciw investigation`, `covariance`, `covariance-replay`; shared investigation, full covariance provenance and declared Jacobian | CIW pinned adapter gate; no dedicated ICRH investigation profile | Connect existing typed covariance to a concrete downstream decision without replacing historical records. |
| Declared circle → GTE | `ciw geodesic create/inspect/replay`; observations, tangent covariance and held candidate | CIW replay tests; ICRH separately runs pinned CBSR/FSRT/GTE numerical comparisons | Numerical comparisons are not a complete GTE investigation profile or a BIM frame mapping. |
| Declared model/certificate → PLSR | `ciw plsr import/evaluate/inspect/replay`; separate retained terminal bundle | CIW adapter/source-binding tests; no dedicated ICRH PLSR profile | Bind a retained candidate dynamics model and compatible estimated state explicitly. |
| PPDA/SCR exchange artifact → SET → CIW inspector | `ciw exchange inspect`; unchanged artifacts, byte digests and conformance report | Actual producer integration tests in CIW | Read-only inspection does not dispatch an SCR workload or import native state. |

“Independent harness” means a separately implemented conformance checker. It
does not mean an independent physical measurement or independent validation of
the scientific algorithm. SET's replay receipts retain their declared scope.

The executable sources are [`telemetry.py`](../src/ciw/telemetry.py),
[`calibrated_window.py`](../src/ciw/calibrated_window.py),
[`calibrated_observable.py`](../src/ciw/calibrated_observable.py),
[`identified_design.py`](../src/ciw/identified_design.py),
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

These six rows describe further integration targets. The affine TBRT/MCUR/STFE
window part of row 1 is delivered through [calibrated windows](CALIBRATED_WINDOW.md);
its live acquisition and drift consumer remain outstanding. Extend the
three demonstrations—process balance, manufacturing cycle and geometry/BIM
inspection—inside the common workspace through these contracts. Each result
should share source/context selection and retained history with the tools that
produced and consume it.

## Provider capability and remaining composition

| Component | Existing capability checked in this review | Why a further integration is meaningful |
| --- | --- | --- |
| [SIDT](https://github.com/giasonpooni/System-Identification-Dynamics-Testbed) | `fit_lti`, `evaluate_one_step`, rank diagnostics and bound declared-identification adapter | The identified-design consumer binds training evidence and the current prior; a PLSR consumer and physical holdout validation remain separate work. |
| [EDSPT](https://github.com/giasonpooni/Experiment-Design-Sensor-Placement-Testbed) | Finite candidate Fisher information, D-/A-optimal ranking, coordinate checks and cost-budgeted next observation | Identified-design now binds the point model, state, reduction baseline and observation budget; multi-observation portfolio optimization is outside this operation. |
| [YWIR](https://github.com/giasonpooni/Yield-Weighted-Inference-Runtime) | Advisory decisions, one-use reserve/settle/cancel operations and bound observation-design token adapter | Identified-design retains an advisory token receipt; it cannot certify the experiment or stand in for measurement cost, and creates no spending reservation. |
| [CSE](https://github.com/giasonpooni/Construction-State-Estimator-for-BIM) | Experimental BIM calculations and a companion experiment harness | Its harness attaches digests to a project-space bundle; that alone does not transfer a measured physical quantity or execute the cited provider. |
| [SRA](https://github.com/giasonpooni/Schematics-Retrieval-Agent) | Typed schematic queries, eligibility and optional pinned companion calls, including JSPT-to-PLSR | Existing companion execution should enter CIW through its retained call records rather than be duplicated. |
| [SCR](https://github.com/giasonpooni/Scientific-Computation-Runtime) | Explicit execution specifications, native dispatch, scientific workloads and exchange exports | CIW currently inspects exports but does not invoke these workloads. Backend prerequisites remain part of the contract. |
| [GSV](https://github.com/giasonpooni/Geospatial-State-Visualization) | Browser provider interface, geographic/temporal inspection and comparison checks | A read-only CIW projection needs an explicit mapping; a synthetic provider does not establish this link. |

No research/dependency fork is counted as an instrument merely because it is in
the account. The covariance-geometry, intrinsic-surface and translation-surface
scaffolds remain deferred until a GSIE, GTE or CBSR workload needs a specific
operation. The executable flat-torus and curved-surface references remain
bounded reference tools; their existence does not require expanding the current
process milestone into geometry research.

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
