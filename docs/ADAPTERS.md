# Shared execution: RCI calibration and FSRT estimation

CIW owns the shared session, selection, identities, persistence and terminal
representations. RCI owns the measurement chain and calibration. FSRT owns the
declared fluid model, state estimate, covariance, residuals and domain checks.
Their scientific implementations remain in their own repositories and execute
through explicitly bound, pinned subprocesses.

This increment connects two independent mass observations to one simultaneous
two-reservoir estimation operation, retains the investigation, and replays it
offline. The fixture is synthetic. It demonstrates the computational boundary;
it does not qualify a real sensor, establish calibration traceability, or validate
a physical reservoir model.

## Adapter contract

`ciw.adapters.protocol.InstrumentManifest` declares inputs, outputs, units,
frames, sampling, normalization, supported operations, determinism, tolerance
policy and calibration requirements. Its `role` distinguishes an instrument
from a measurement-chain adapter or operation provider. The schema name does not
turn every connected tool into an instrument.

| Component | Responsibility |
| --- | --- |
| `ciw.core.records` | Generic recording structure and finite scientific values |
| `ciw.core.identities` | Content identities, separate from execution identities |
| `ciw.adapters.protocol` | Declarative manifest and explicit refusal envelope |
| `ciw.adapters.registry` | Explicit registration and domain validation |
| `ciw.adapters.subprocess` | Operator-bound source and interpreter verification, bounded JSON execution |
| `ciw.operations.registry` | Supported operation declarations and local bindings |
| `ciw.operations.runner` | Separate execution/result records and retained refusals |

The analytic oscillator is an ordinary adapter. Its equation, parameters,
scientific arrays, periodogram normalization and existing evidence identities
are preserved. Generic records no longer require its channel names, physical
frame, sample rate or display geometry. Views and operations remain conditional
on the adapter's capabilities.

## Setup and runtime identities

Use Python 3.12 or newer for this external-tool example. CIW itself supports
Python 3.11 and newer. The executable revision pins are retained in
[`adapter-runtimes.json`](../src/ciw/adapter-runtimes.json). Keep both domain
repositories as separate clean checkouts at those commits.

```sh
python -m pip install -e '.[dev]'
git clone https://github.com/giasonpooni/Retrofitted-Computational-Instrumentation.git ../rci
git -C ../rci checkout --detach 97fcc01a45985defd6a76532b1fefe90d66dd159
git clone https://github.com/giasonpooni/Fluid-State-Reconstruction-Testbed.git ../fsrt
git -C ../fsrt checkout --detach d34588819f657b6639c8e9a19450215844610007
```

The adapter reads domain code from the checkout's `src` directory. It does not
copy that code into CIW. Pass `--python /absolute/path/to/python` when the
external runtimes need another installed Python environment; otherwise CIW uses
its own interpreter. NumPy must be installed in that interpreter. Source and
runtime identities are retained for replay comparison.

Executable bindings are local operator configuration. Saved investigations are
data and cannot select an executable, import a module, or authorize a subprocess.
Reopening is read-only; executing after reopening requires an explicit local
binding. The subprocess boundary limits input/output size and execution time.
It is not an operating-system sandbox.

## Create, inspect and replay

From the CIW root with the pinned checkouts prepared:

```sh
python -m ciw investigation create --inputs examples/adapters/two-reservoir.json --rci-repo ../rci --fsrt-repo ../fsrt --output-dir results/two-reservoir
python -m ciw investigation inspect results/two-reservoir/workspace.json
python -m ciw investigation inspect results/two-reservoir/workspace.json --json
python -m ciw investigation replay results/two-reservoir/workspace.json --rci-repo ../rci --fsrt-repo ../fsrt --output-dir results/two-reservoir-replay
```

The terminal prints separate raw, calibrated, estimated and residual rows, plus
any refusal. `--json` retains the full summary, identities, uncertainty and
provenance; the compact table is only a representation and is never scientific
input. Inspecting a saved investigation requires no domain checkout or network
access. Offline replay requires the already installed runtime and pinned local
checkouts, but no network access.

The fixture's top-level schema is `ciw.tank-investigation-input.v1`. It names
two sensors, each with an explicit `ciw.adapter-request.v1` RCI calibration
request, declares `cross_assembly_independent: true`, and supplies `model` and
`source_description`. Sensor order is the FSRT state/covariance order. Each
underlying RCI request retains assembly TOML, original record bytes encoded as
base64, the calibration profile and explicit uncertainty inputs.

| Fixture quantity | Declared value |
| --- | --- |
| Raw observations | 7000 and 3000 counts, separate assemblies/calibrations |
| Calibrated masses | 70 and 30 kg; `mass = 0.01 * (raw - 0)` |
| Acquisition time | Both `2026-01-15T12:00:00Z` |
| Calibration interval | `[2026-01-01, 2026-02-01)` |
| Calibration parameter order | `[scale, zero_raw]` |
| Parameter covariance | `[[1e-10, 1e-7], [1e-7, 0.04]]`, in the corresponding parameter-product units |
| Raw variance | `0.25 count²` per observation |
| Independent calibration residual | Declared synthetic `sigma = 0.02 kg` |
| Calibrated variances | `0.005315` and `0.001323 kg²`; diagonal joint covariance under declared assembly independence |
| FSRT prior | Mean `[70, 30] kg`; isotropic standard deviation `5 kg` |
| FSRT total mass | `100 kg`, with independent variance `0.25 kg²` |

The run's single time coordinate is `0 s` relative to that shared acquisition
instant. Its `1 s` selection interval only gives the snapshot a nonempty
half-open selection support; it is not a sampling cadence or a claim of one
second of acquired data. The frame is `two-reservoir-mass` and no sample rate
is inferred.

Reopening preserves the existing evidence, execution and result identities.
Explicit replay recalculates from retained inputs and generates new execution
and result identities. Unchanged scientific evidence retains its content
identity. A saved digest detects content inconsistency; it does not establish
source authenticity, metrological traceability or physical truth.

## Shared terminal session

Reopen the same saved run and results in a live session:

```sh
python -m ciw serve --workspace results/two-reservoir/workspace.json --output-dir results/two-reservoir-live --fsrt-repo ../fsrt
```

In another terminal:

```sh
python -m ciw health
python -m ciw send session.get
python -m ciw send operation.list
python -m ciw send result.list
python -m ciw send execution.list
python -m ciw send operation.execute --payload '{"operation_id":"fsrt.tank-reconstruct.v1","parameters":{}}'
python -m ciw send workspace.save
```

The session retains one authoritative selection and operates on full-resolution
records. RCI observations and FSRT outputs are terminal-accessible. A domain
specific Godot fluid viewport is not supplied by this increment.

The execution command above uses the model retained in the run. To evaluate a
different declared model, supply a `parameters.model` object with
`kind: "reservoir2-linear-v1"`, two `prior_mean` values in kg, one isotropic
`prior_std` in kg, `total_mass_kg`, and `total_mass_variance_kg2`. The prior,
observation and total mass are assumed independent; this operation does not
represent their cross-covariances. The saved observation evidence remains fixed.

## Evidence and uncertainty

| Retained record | Meaning |
| --- | --- |
| Raw observation | Exact original JSON record bytes and a raw evidence identity |
| Calibrated observation | A separate evidence identity linked to the original observation and declared calibration |
| Calibration | Declared conversion parameters, validity and measurement-chain binding; parameter covariance remains explicit |
| Run | Raw and derived channels with units, sampling, frame and provenance |
| Operation | Declared computation and explicit parameters |
| Execution | One attempt, including a success or refusal |
| Result | Estimate, covariance, residuals and diagnostic information from a successful attempt |
| Verification | Separate identity/status; successful computation does not establish verification |

The two-observation fixture is a single simultaneous snapshot, not two time
samples of one sensor. Channels preserve their separate raw and calibrated
semantics. Calibration covariance is retained with its parameter ordering and
propagated to measurement uncertainty; it is not replaced by an unexplained
scalar standard deviation. Independence is an explicit limitation of this
initial pair of measurement chains. The saved covariance is available to the
downstream estimation operation.

Missing, expired-at-acquisition or chain-mismatched calibration refuses the derived result.
Serving-time expiry is displayed separately as `expired` with an `evaluated_at`
timestamp, beside immutable acquisition applicability. A historical profile
that was valid at acquisition remains usable for offline replay even if it has
expired since. This display check does not alter the retained evidence.
The user's original input file and raw bytes are left unchanged. Because the
calibrated source is invalid, this create attempt writes no run, result or
workspace; the CLI prints the refusal code and exits `2`. Once a valid run
exists, a refused FSRT attempt is retained as an execution record with no
fabricated result, and the command exits `0` with summary status `refused`.
Input/configuration errors also exit `2`.

FSRT's successful numerical response may still report
`physical_model_disagreement`: reconciliation is held and the unprojected
estimate remains visible. Its fault status is `confounded_or_unidentifiable`
when the single balance cannot distinguish possible explanations.
`numerical_refusal` records an inability to produce an admissible numerical
answer. A model status of `consistent` only describes the declared balance and
uncertainties; it does not establish sensor health or physical truth.

## Validation and current limits

Run the repository tests from a development installation:

```sh
python -m pytest -q
CIW_RCI_REPO=../rci CIW_FSRT_REPO=../fsrt python -m pytest -q tests/test_investigation.py tests/test_adapter_cli.py
# Or clone the declared pins into temporary directories and run that gate:
python scripts/check_adapters.py
```

Tests cover the unchanged oscillator science and original protocol, generic
record validation, source-pin refusal, calibration/evidence separation,
uncertainty retention, offline reopening/replay and terminal representations.
The complete checked-out-repository path is also exercised using the fixture
and CLI commands above.

This first slice does not integrate JSPT, Intrinsic Surface Geodesics, Curved
Surface Runtime, CSE, SRA, YWIR, GAT or DAF. Their proposed roles remain separate.
PLSR's existing terminal bundle path remains available; attaching its
verification results to shared investigations is a subsequent integration.
The generic seam does not establish arbitrary event-stream resampling,
hardware acquisition, real-time guarantees, cross-sensor calibration correlation,
or physical qualification.

Domain authority remains in the upstream
[RCI repository](https://github.com/giasonpooni/Retrofitted-Computational-Instrumentation)
and [FSRT repository](https://github.com/giasonpooni/Fluid-State-Reconstruction-Testbed).

The small manifest/batch protocol implemented here is an additive executable
slice, not completion of the architecture's broader M1/M3 contracts. The
architecture's affine-displacement bench, named held-out calibration trials,
path-state estimator, multi-run journal, and calibrated Godot representations
remain separate development gates. Current acquisition applicability and shared
parameter covariance implement only the stated RCI-to-FSRT snapshot path.

## Extension roles

Future integrations extend the same substrate and preserve these ownership boundaries:

| Repository/tool | Workbench role | Boundary |
| --- | --- | --- |
| RCI | Measurement-chain/calibration adapter | Original observations to declared calibrated evidence |
| FSRT | Physical state-estimation instrument | Declared model and evidence to estimates, residuals, covariance or refusal |
| JSPT | Sensitivity/covariance operation provider | Jacobians, covariance propagation, coordinate transforms, Monte Carlo comparison |
| Intrinsic Surface Geodesics | Geometry/path artifact provider | Mesh/path to versioned path geometry |
| Curved Surface Runtime | Domain-specific path-sensitivity instrument | Path artifact and observations to sensitivity envelope |
| PLSR / GAT | Verification operations | Separate certificate/check identities attached to retained results |
| CSE | Evidence-to-decision instrument | Fail-closed disposition with separate decision identity |
| SRA | Typed execution-plan compiler | Eligible operation graph; no execution authority |
| YWIR | Compute/admission control | Admission/refusal before CIW invokes a provider |
| DAQ / ASSAY / SCL / STE | Acquisition and numerical backends | Implementations behind adapters |
| DAF | Durable history provider | Persistence through an identity-preserving adapter |

The next integrated physical workflow is calibrated metrology plus retained path
geometry and a domain-owned path-sensitivity model. Ontology/graph projections
can represent the resulting evidence, calibration, run, operation, execution,
result, verification and decision links after the executable boundaries are
established. No certificate or disposition silently converts a result into
physical truth.
