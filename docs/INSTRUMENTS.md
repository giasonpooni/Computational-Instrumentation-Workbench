# Instruments: instructions and specifications

This catalogue records tools that can be used through the workbench and the
contracts needed to reproduce their results. An external tool is listed as
integrated only after its workbench entry point, saved evidence and replay path
have been exercised together.

| Tool and role | Workbench status | Entry point |
| --- | --- | --- |
| `analytic-damped-oscillator.v1` | Integrated built-in synthetic instrument | `python -m ciw demo`, `analyze stats`, `analyze spectrum` |
| RCI measurement-chain/calibration adapter | Integrated experimental pinned subprocess; synthetic mass fixture | `python -m ciw investigation create`, `inspect`, `replay` |
| FSRT state-estimation operation | Integrated experimental pinned subprocess; one simultaneous two-reservoir snapshot | Same investigation; shared `operation.execute` after explicit runtime binding |
| Parameterized Lyapunov Stability Runtime (PLSR) verification operation | Integrated experimental terminal operation; Python 3.12+ and optional `plsr` extra | `python -m ciw plsr import`, `evaluate`, `inspect`, `replay` |

## Synthetic damped oscillator

The built-in instrument produces a deterministic recording of an analytic
damped oscillator. Statistics, spectral analysis, saved results and reopening
work through the terminal. Source:
[`ciw.instruments`](../src/ciw/instruments.py); transport and record definitions:
[protocol v1](PROTOCOL.md).

### Install and run

Use Python 3.11 or newer in an activated virtual environment, from the repository
root. Installation uses the dependencies pinned in
[`pyproject.toml`](../pyproject.toml): NumPy 2.4.3 and websockets 16.0. The built-in
instrument ships with CIW; retain the CIW commit and Python version with an
investigation when reproducing it on another machine.

```text
python -m pip install -e .
python -m ciw demo --output recordings/demo.json
python -m ciw analyze stats --recording recordings/demo.json --channel q --start 2 --end 8 --output-dir results/stats
python -m ciw analyze spectrum --recording recordings/demo.json --channel q --start 0 --end 12 --output-dir results/spectrum
python -m ciw inspect results/spectrum/workspace.json
```

`demo` needs no external data. Analysis accepts the complete recorded-run JSON,
a channel (`q`, `v` or `energy`) and a half-open interval `[start, end)` in seconds.
The demo's model settings are fixed in v1. The `inspect` command reads a saved
file without executing an analysis. See the [quickstart](quickstart.md) for
virtual-environment setup, shared terminal use and the optional Godot viewport.

### Scientific specification

| Property | Delivered specification |
| --- | --- |
| Equation | `q'' + 2 gamma q' + omega_0^2 q = 0`, evaluated analytically |
| Parameters | Mass `1 kg`; natural frequency `0.8 Hz`; `omega_0 = 2 pi * 0.8 rad/s`; damping `gamma = 0.15 /s` |
| Initial state | Displacement `q(0) = 1 m`; velocity `v(0) = 0 m/s` |
| Energy | `0.5 * mass * (v^2 + omega_0^2 * q^2)` in joules |
| Samples and time | 768 uniformly spaced samples at 64 Hz on `[0, 12)` seconds; time since run start |
| Precision and frame | Retained `float64` scientific arrays; `oscillator-state` coordinate frame |
| Channels | `q`: metres; `v`: metres/second; `energy`: joules |
| `statistics.v1` | Sample count, mean, minimum, maximum and RMS of all retained samples in the selected interval |
| `spectrum.periodogram.v1` | One-sided periodogram PSD; constant detrend; periodic Hann window; density normalization |

Statistics require at least one selected sample; spectra require at least four.
For a spectrum of `N` samples, frequency spacing is `64 / N` Hz and density units
are the squared channel unit per Hz. Interior positive-frequency bins are
doubled; DC and an even-length Nyquist bin are not. The integral of the PSD equals
the window-weighted detrended mean square. Processing settings are included in
each result. This operation does not implement Welch averaging or a spectrogram.

### Outputs, identities and reopening

`demo` writes the recording to its requested path. Each analysis writes a source
recording copy, an individual result JSON and `workspace.json` to its output
directory, and prints a protocol-v1 response envelope whose `payload` is the
result record. Use separate output directories for
separate workspace snapshots: another analysis in the same directory replaces
`workspace.json` while retaining individual result files.

| Record | Fields and meaning |
| --- | --- |
| Recording | Instrument ID, `run_id`, scientific `evidence_id`, metadata, timestamps, channel values and derived render data |
| Result | `evidence_id`, `operation_id`, `execution_id`, `result_id`, source recording reference, selection revision, channel, interval and numerical `data` |
| Verification | `verification_id: null`, `verification_status: "not_verified"` |
| Workspace v1 | Complete recording, current selection, saved results and reserved view settings |

The evidence ID hashes the scientific content; the recording filename separately
binds the complete recording. These bindings detect inconsistent content. They
do not establish source authenticity or scientific verification. Analyses read
the full source arrays, and display geometry is a derived view.

Reopen the spectrum workspace in one terminal:

```text
python -m ciw serve --workspace results/spectrum/workspace.json --output-dir results/reopened
```

In another activated terminal, discover the restored results and save the session:

```text
python -m ciw send result.list
python -m ciw send workspace.save
```

Reopening validates and restores the recording, selection and existing results
without rerunning calculations. Existing evidence, result and execution IDs are
preserved. An explicit new analysis creates new execution and result IDs. The
[quickstart replay instructions](quickstart.md#reopen-an-investigation) explain
result inspection and selection persistence.

### Validation and limits

The checked-in [numerical tests](../tests/test_instruments.py) cover the analytic
fixture, energy decay, interval selection and PSD normalization, including DC and
Nyquist handling. [Integration tests](../tests/test_integration.py) compare terminal
and service calculations; [replay tests](../tests/test_replay.py) check saved
identities, content bindings and restoration without scientific recomputation.
Run them from a development installation:

```text
python -m pip install -e '.[dev]'
python -m pytest -q
```

This adapter's input contract is one small uniformly sampled oscillator recording
per session. The shared adapter boundary also accepts separately declared
external recordings. Oscillator data are synthetic; device acquisition,
uncertainty estimates and physical validation are not supplied by this adapter.

## RCI calibration and FSRT estimation

The [adapter operating guide](ADAPTERS.md) documents setup, exact runtime pins,
the input fixture, terminal commands, retained evidence and offline replay.
RCI remains the authority for acquisition and calibration semantics. FSRT remains
the authority for the reservoir model, estimate, covariance and diagnostics.

```sh
python -m ciw investigation create --inputs examples/adapters/two-reservoir.json --rci-repo ../rci --fsrt-repo ../fsrt --output-dir results/two-reservoir
python -m ciw investigation inspect results/two-reservoir/workspace.json --json
python -m ciw investigation replay results/two-reservoir/workspace.json --rci-repo ../rci --fsrt-repo ../fsrt --output-dir results/two-reservoir-replay
```

| Contract | Delivered specification |
| --- | --- |
| Source/adapter pins | Full upstream revisions in [`adapter-runtimes.json`](../src/ciw/adapter-runtimes.json); CIW adapter version retained with the execution |
| Input | Two distinct synthetic mass measurement chains at the same acquisition instant, explicit calibration profiles and covariance, declared two-reservoir model |
| Calibration | Exact raw bytes retained; corrected observations get distinct evidence identities; missing, expired or mismatched calibration refuses a derived result |
| Uncertainty | Parameter covariance and ordering retained; propagated observation covariance passed to FSRT |
| Estimate | FSRT mass-state estimate and covariance, residuals and physical-model/fault diagnostics |
| Persistence | One shared workspace with retained inputs, results and execution/refusal history; read-only offline inspection and explicit offline replay |
| Views | Terminal state table and complete JSON; shared session inspection; no domain-specific Godot fluid view |
| Qualification | Synthetic integration fixture; calibration traceability and physical validation are not established |

Calibration is evaluated at the observation's acquisition time so a historical
record can be replayed later without relabelling it as a current measurement.
Source and runtime bindings are supplied explicitly and are not executed from
saved workspace declarations. Refer to [validation and limits](ADAPTERS.md#validation-and-current-limits)
before extending the snapshot fixture to a new measurement arrangement.

## Covariance provenance and JSPT operations

The [covariance guide](COVARIANCE.md) gives exact setup, schema, commands and
limits for the RCI v2 → FSRT v2 → JSPT path. Exact source revisions are retained
in [`adapter-runtimes.json`](../src/ciw/adapter-runtimes.json), including historical
pins needed by older saved investigations. Numerical calculations stay in the
domain repositories; the workbench validates and retains their artifacts.

```sh
python -m ciw investigation create --inputs examples/adapters/two-reservoir-covariance.json --rci-repo ../rci --fsrt-repo ../fsrt --output-dir results/covariance
python -m ciw covariance results/covariance/workspace.json --jspt-repo ../jspt --parameters examples/adapters/tank-covariance-map.json --output-dir results/covariance-map
python -m ciw covariance-replay results/covariance-map/workspace.json --jspt-repo ../jspt --output-dir results/covariance-replay
python -m ciw investigation inspect results/covariance-map/workspace.json --evaluated-at 2030-01-01T00:00:00Z
```

| Contract | Delivered specification |
| --- | --- |
| RCI operation | `rci.calibrate.v2`; raw bytes preserved; native scale/zero covariance with component provenance, explicit exclusions, source dependencies and immutable acquisition applicability |
| FSRT operation | `fsrt.tank-reconstruct.v2`; observation, prior, total, innovation, posterior and reconciled covariance artifacts; explicit model independence |
| JSPT operation | `jspt.covariance-propagate.v1`; full covariance pushed through a caller-declared Jacobian using existing JSPT kernels |
| Artifact | `covariance-artifact.v1`; content identity, ordered quantities/units, frame, reference values, full matrix, method, basis, source links and assumptions |
| Views | Terminal full matrices and structured JSON; calibration serving metadata on session/result reads; no uncertainty viewport in this increment |
| Failure | Invalid basis, unsupported dependence, invalid matrices or runtime drift refuses computation; refusal has an execution identity and no result |
| Replay | Saved source results and covariance dependencies remain intact; explicit replay appends new execution/result identities and compares data |
| Validation | [`test_covariance_integration.py`](../tests/test_covariance_integration.py) exercises all three pinned providers; [`check_adapters.py`](../scripts/check_adapters.py) also supplies historical checkouts |
| Limits | Synthetic two-reservoir snapshot; cross-assembly shared uncertainty currently requires a joint model and is refused; no inferred independence, automatic Jacobian derivation, nonlinear Monte Carlo, NIS/NEES/coverage qualification or physical verification |

## Parameterized Lyapunov Stability Runtime (PLSR)

PLSR evaluates quadratic Lyapunov certificates for declared linear and
affine-parameter models. Its terminal adapter uses upstream commit
[`19ea6967060166ba09db6cd4563bd87bd6b3d196`](https://github.com/giasonpooni/Parameterized-Lyapunov-Stability-Runtime/tree/19ea6967060166ba09db6cd4563bd87bd6b3d196).
The [`ciw-plsr-adapter-v1` operating guide](PLSR.md) contains the complete
installation, import/evaluate/inspect/replay sequence, exact sample and saved-run
contracts, numerical status meanings and validation instructions. The adapter
checks installed source identity, retains the full declaration and explicit
sample, and writes a self-contained run automatically.

From an activated Python 3.12-or-newer environment at the repository root:

```text
python -m pip install -e '.[plsr]'
python -m ciw plsr import examples/plsr/continuous-affine.json --output models/plsr.json
python -m ciw plsr evaluate --model models/plsr.json --sample examples/plsr/continuous-sample.json --output-dir results/plsr
```

The evaluation prints `saved_file` and `bundle`. Pass that file to
`python -m ciw plsr inspect RUNFILE` or
`python -m ciw plsr replay RUNFILE --output-dir results/plsr-replay`.
The [worked guide](PLSR.md#first-complete-run) includes copyable PowerShell and
POSIX shell examples that capture the generated filename.

| Contract | Delivered specification |
| --- | --- |
| Model | Upstream `model-artifact-v1`: matrices/certificate, boxes, ordered state and parameter units, time convention/sample period, margin derivation, estimator and provenance |
| Sample | `plsr-sample-v1`: explicit `x`, `theta`, `theta_dot`; no synthetic defaults |
| Operation and output | `plsr.verdict.v1`; immutable `ciw-plsr-run-v1` bundle with separate evidence, operation, execution, result and verification identities |
| Integrity and replay | Model artifact, companion record and bundle digests; replay retains its source binding and compares the companion digest exactly |
| Status and exit codes | Raw runtime code and all three verdict booleans retained; exit `0` means command success, `2` input/configuration/execution failure, `3` replay mismatch |
| Views | Headless terminal JSON; no shared WebSocket session or Godot PLSR view in this increment |
| Verification | `not_verified`; `may_authorize: false`; physical validation `not_started`; `proof_status: NOT_CHECKED` |

`NUMERICAL_INCONCLUSIVE` is a numerical refusal and `NOT_CERTIFIED` is a
certificate violation; both may be valid saved results with exit code zero.
One invocation evaluates one explicit sample. Host scripts can repeat it for
offline sweeps; no real-time loop performance or physical-system validation is
claimed. See the [guide's validation and limits](PLSR.md#validation-and-limits)
for the bounded integration.

## Geometric Telemetry Engine (GTE)

The [GTE operating guide](GTE.md) provides setup, pins, exact commands, retained
record semantics and validation. This bounded reference operation evaluates a
declared circle in a Euclidean plane; GTE owns geometry and covariance transport,
and CIW supplies the existing pinned subprocess, session and identity substrate.

```sh
python -m ciw geodesic create --inputs examples/adapters/circle.json --gte-repo ../gte --output-dir results/circle
python -m ciw geodesic inspect results/circle/workspace.json --json
python -m ciw geodesic replay results/circle/workspace.json --gte-repo ../gte --output-dir results/circle-replay --json
```

| Contract | Delivered specification |
| --- | --- |
| Tool and adapter pins | Full GTE revision in [`adapter-runtimes.json`](../src/ciw/adapter-runtimes.json); `ciw-pinned-subprocess-v1` |
| Input | `gte.circle-request.v1`; metre `[x,y]` observations, explicit frame/time origin, full sample-major joint covariance, versioned exact circle and policy |
| Operation | `gte.project-circle.v1`; candidate projection and first-order tangent/ambient covariance, retained original residuals and signed shortest-arc diagnostics |
| Outcome | Eligible or held candidate distinct from a refused execution; no physical verification or state-commit authority |
| Evidence and replay | Exact original bytes and parsed request; distinct raw evidence, execution, result and verification identities; offline read and explicit pinned replay |
| Parameters | Whole constraint/policy overrides retained with execution; observations/covariance cannot be replaced through parameters |
| Views | Terminal state table and complete JSON, shared session; no domain Godot viewport |
| Validation and limits | Real CLI/session/replay tests in `tests/test_geodesic.py`; synthetic example, no GNSS solver or dynamical-model claim |

The operation requires a selected interval containing the full retained batch;
narrower selections are refused. It does not infer cross-sample independence, sensor accuracy,
winding count, velocity, or a calibrated posterior. See [status and limits](GTE.md#status-and-limits)
before extending the reference geometry or attaching real measurements.

## Related components

The [related stack catalogue](../README.md#related-stack-components) records the
current technical names and responsibilities of Scientific Computation Runtime,
Provenance-Preserving Data Acquisition, Geospatial State Visualization, State
Estimation Evaluation Testbed, Evidence and State Management, and Constraint-Based
State Reconciliation. None is registered as an integrated CIW tool in this
catalogue. The evaluation repository has an early executable contract slice but
no evaluation runner or CIW adapter; the reconciliation component has no executable
adapter. Their [boundaries](ADAPTERS.md#related-component-boundaries) retain
existing evidence, operation, execution, result, verification and runtime
identities.

## Catalogue documentation requirements

Update this catalogue and the README when an integration is delivered. Each entry
must include its source and pinned version, supported runtime/environment, exact
setup and workbench commands, input ordering/units/time conventions, versioned
output and status semantics, saved-evidence and replay instructions, verification
state and limits, and links to validation evidence. A successful installation or
standalone example alone does not establish a completed workbench integration.
