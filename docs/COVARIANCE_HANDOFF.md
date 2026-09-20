# Covariance implementation handoff

This increment extends the existing shared workbench. Domain calculations stay
in RCI, FSRT and JSPT; CIW validates records, resolves source identities, executes
explicitly bound providers, retains results and serves representations.

## Delivered boundaries

| Repository | Delivered increment | Published source |
| --- | --- | --- |
| RCI | Additive `rci.calibrate.v2`; explicit covariance component/provenance basis, native scale/zero parameterization, dependency identifiers and immutable acquisition applicability | `f863bdd69d49224e0cdc871943bbb052e5b0a975`, existing [PR 1](https://github.com/giasonpooni/Retrofitted-Computational-Instrumentation/pull/1) |
| FSRT | Additive `fsrt.tank-reconstruct.v2`; six linked covariance artifacts and explicit independence declarations, with unchanged v1 numerical results | `09a756dd9cdd3a9bb6cb14b5cd498f6259937ac2`, main |
| JSPT | `jspt.covariance-propagate.v1`; explicit Jacobian, weighted aggregation and coordinate-map operations through existing kernels | `d910f5a1d7f6dd5f2dd87dfca66990f714f97b18`, main |
| CIW | `covariance-artifact.v1`, strict offline validators, v2 investigation, JSPT operation in the same session, covariance terminal display, serving-time calibration metadata and historical runtime allowlist | This workbench increment; runtime pins in `src/ciw/adapter-runtimes.json` |

The old oscillator science and v1 adapter outputs remain intact. Historical
runtime entries allow old investigations to replay with their original local
checkouts and matching interpreter/dependencies. Unknown or unbound executions
cannot be silently replayed using a newer engine.

## What was exercised

- RCI: 126 tests passed, including a v1 byte-preservation comparison and new
  covariance basis, omitted/represented components, overlap and double-counting
  refusal cases.
- FSRT: 161 affected endpoint, reconciliation and quickstart tests passed.
  Independent rational reference cases check correlated observation covariance
  and its effect on the posterior. This is not a claim that the entire historic
  simulation suite was run locally.
- JSPT: 67 tests passed and one optional test skipped; quickstart passed. The
  real CIW subprocess fixture averages three samples with shared offset variance
  1 and independent variance 0.09 each, producing variance 1.03 rather than
  incorrectly averaging away the shared contribution.
- CIW: full local test run passed 403 tests and 38 subtests. Its 27 external-source
  tests were skipped in that run and are exercised separately with pinned domain
  checkouts by `scripts/check_adapters.py`: all 36 tests in that dedicated gate
  passed, including historical runtime replay.
- Godot import, live protocol, channel generality and generic-adapter boundary
  checks all passed. The new covariance path remains terminal/JSON-only.
- The complete CLI RCI v2 → FSRT → JSPT → inspection → replay path passed.
  Inspection preserved the source workspace bytes; replay retained old results
  and produced new execution/result identities with matching scientific data.

The commands and exact inputs are in [COVARIANCE.md](COVARIANCE.md). The shared
adapter workflow clones both current and historical allowlisted checkouts; the
historical test creates an old-pin v1 investigation and replays it after the
runtime upgrade. Test sources and assertions are reviewable in
`tests/test_covariance_integration.py`, `tests/test_covariance_artifacts.py`,
`tests/test_covariance_records.py`, `tests/test_rci_records.py`,
`tests/test_calibration_status.py` and `tests/test_covariance_replay_refusal.py`.

## Scientific meaning and limits

The artifact retains ordered quantities, units, coordinate frame, reference
values, full covariance, method, basis, assumptions and content-addressed source
links. CIW does not infer unit conversions or physical validity. Singular PSD
artifacts are valid; individual operations retain their own numerical admission
requirements. No schema validator silently repairs matrices.

RCI records whether contributions are included, represented in another component
or excluded with a reason. Exclusions and the absence of a completeness claim
survive through FSRT and JSPT metadata. These declarations do not authenticate a
reference certificate or establish traceability. RCI uses native
`scale * (raw - zero_raw)` and retains `[scale, zero_raw]`; it does not relabel
that covariance as affine gain/offset covariance.

The integrated two-assembly path requires explicit cross-assembly independence.
Overlapping dependency/shared-source identifiers cause refusal. Distinct IDs do
not prove independence. General joint calibration across assemblies is not yet
implemented, although FSRT itself accepts declared correlated observation
covariance and RCI retains shared parameter covariance across records in a batch.
The v2 investigation also explicitly declares independence between prior,
observations and total constraint; CIW no longer supplies those flags implicitly.

FSRT exposes observation, prior, declared-total, innovation, posterior and
reconciled covariance. Its existing one-sided physical balance gate retains its
meaning. It is not predictive NIS, reference-truth NEES, coverage qualification
or a physical certificate. Missing observations are never replaced with zero
measurements; their artifact reference values are declared references.

JSPT propagates a caller-declared matrix through existing numerical kernels.
It does not derive or verify the Jacobian, compute nonlinear output means,
perform Monte Carlo validation, or establish physical unit/frame compatibility.
It preserves these limitations and the source uncertainty context in its result.

Calibration expiry at serving time remains separate from immutable acquisition
applicability. A historically applicable observation remains replayable after
expiry. All successful results still have `verification_id: null` and
`verification_status: not_verified`.

## Claude's next contract pass

1. Reconcile architecture requirements against the executable protocol and the
   exact source pins above. Mark implemented, tested, partial and planned
   requirements separately; passing a synthetic integration does not close the
   full metrological or M3 estimator-validation contract.
2. Correct `CIW-CAL-012`: shared parameters induce covariance
   `C_ij = J_i C_theta J_j^T`, not necessarily perfect correlation. An aggregate
   uses `J_aggregate = sum_i w_i J_i`; neither blind division by sample count nor
   an invariant contribution magnitude is generally correct.
3. Reconcile the architecture's affine gain/offset convention with RCI's native
   scale/zero convention. If a conversion is specified, transform covariance
   through the parameter map and state the approximation order.
4. Review the executable covariance basis and exclusion semantics without
   promoting declared documentary identifiers to authenticated reference
   evidence or inferring independence from disjoint IDs.
5. Specify the next statistical-validation operation: distinct coverage,
   predictive innovation NIS, marginal NEES and joint NEES; explicit distribution
   or empirical basis, normalization, degrees of freedom, sample dependence and
   finite-sample acceptance bands. A valid coverage upper bound can equal 100%
   for small samples. Keep these checks separate from the physical balance gate.
6. Align manifests, examples, requirement IDs, milestone labels and conformance
   rows. Preserve the delivered APIs and scientific results; propose additive
   versions where a stronger contract changes executable meaning.

Subsequent implementation gates are a joint cross-assembly calibration model,
statistical qualification fixtures, and uncertainty representations in the
viewport. Geometry/path, curved-surface sensitivity, verification and decision
providers should adopt the same artifact/source boundary when integrated.
PLSR/CSE/SRA/YWIR/DAQ/ASSAY/SCL/STE/DAF remain in their existing roles; this change
does not turn them all into instruments or merge their repositories into CIW.

## Existing cross-platform reproduction issue

The broad FSRT Windows/Python 3.13 workflow at the new pin reports one failure
in `tests/test_real_noaa_month.py::test_the_report_reproduces`, outside the
changed adapter. Its fast unit gate passed 1029 tests; the broader run passed
1387 tests before reporting a tide-month profile log-likelihood outside the
existing cross-build tolerance. The [new workflow](https://github.com/giasonpooni/Fluid-State-Reconstruction-Testbed/actions/runs/35536619209)
and [previous-pin workflow](https://github.com/giasonpooni/Fluid-State-Reconstruction-Testbed/actions/runs/35533956823)
show that the same NOAA reproduction test also failed on Windows before this
increment. Linux passed that previous-pin suite.

The covariance work did not change the NOAA model, fit, report or reproduction
tolerance. Treat this as a separate numerical reproducibility investigation;
do not widen the tolerance or regenerate the scientific report solely to make
the workflow green. The local affected FSRT gate and the full pinned CIW
integration remain the validation evidence for this increment.
