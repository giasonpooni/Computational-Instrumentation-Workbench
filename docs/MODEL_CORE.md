# Language-neutral model core and pinned Julia worker

This guide specifies the first executable increment of the
[Python–Julia scientific core](../README.md#pythonjulia-scientific-core): one
structured model specification shared by Python and Julia, consistent unit
transformation, typed-port composition, generated LaTeX views, and a pinned
persistent Julia worker below the existing CIW → SCR execution seam.

| Layer | Implemented in this increment | Not yet implemented |
| --- | --- | --- |
| Python and CIW | `ciw.model-spec.v1` validation, unit algebra, rescaling, composition, LaTeX views, RK4 reference, retained runs, offline inspection and replay | Shared-session (`ciw serve`) catalog entry and Godot view for model runs |
| Julia / ModelingToolkit | Tsit5 simulation of the lowered model; ModelingToolkit `System`/`mtkcompile` and Symbolics Jacobians with state-order mapping | Symbolic simplification or index reduction of user models; DAEs |
| JuliaControl | ForwardDiff linearization with ControlSystemsBase `ss`, poles, controllability and observability ranks | Observers, identification, controller design, ModelPredictiveControl.jl |
| JuMP and selected solvers | HiGHS MILP measurement selection with a prior-normalized maximin information objective | Nonlinear (D/A-optimal) design, other solver backends |
| GPU computation | — | Ensembles, fields and sensitivities on GPU with measured cost |
| LaTeX views | Generated equations, assumptions, constraints and a symbol table bound to run, estimate, uncertainty and evidence; escaped exploratory derivations | Editor integration |

## Specification: `ciw.model-spec.v1`

A specification is data. Neither Python nor Julia receives host-language
source. [`examples/model-core/damped-oscillator.json`](../examples/model-core/damped-oscillator.json)
is the reference fixture. Its meaning includes:

| Field | Meaning retained |
| --- | --- |
| `independent_variable` | Symbol, `kind` (`time` or `path_length`), unit (dimension checked against the kind) and origin text |
| `states` | Ordered symbols; the order is the state vector's meaning. Each has quantity, unit and frame |
| `inputs` | `role` `commanded`, `measured` or `physical`. A measured input declares its calibration status and identity; a commanded input may not |
| `parameters` | Declared values in declared units |
| `dynamics` | One equation per state, in state order: `d state / d independent = rhs` |
| `derived` | Ordered derived quantities; `role` `derived`, `commanded` or `physical` |
| `observations` | Measured quantities with sensor, frame and calibration (`calibrated` + identity, or `uncalibrated`) |
| `ports` | Typed in/out ports naming inputs, states, observations or connectable derived quantities |
| `uncertainty` | Ordered covariance blocks for parameters, initial state, measurement noise and process noise (`Cov(dW) = Q dt`), plus a closed vocabulary of assumptions. An absent block is `null` and must be declared (`no_process_noise`, `no_measurement_noise_declared`); it is never replaced by zeros |
| `validity_domain` | Bounds on the independent variable, states, inputs and parameters (declared software limits), with notes |
| `provenance` | `authored`, `unit_rescale` or `composition`, with the derivation record |

Expressions are JSON trees: `{"sym": ...}`, `{"num": ...}` (dimensionless) and
`{"op": ..., "args": [...]}` with `add`, `sub`, `mul`, `div`, `neg`, `pow`,
`sqrt`, `abs`, `sin`, `cos`, `tan`, `exp`, `log`, `tanh`. Validation checks
dimensions exactly with rational exponents: sums need equal dimensions,
transcendental functions need dimensionless arguments, a dimensioned base needs
an integer literal exponent, and each dynamics equation must have the state's
dimension divided by the independent variable's. Units are multiplicative SI
expressions (`mm`, `kg*m^2/s^2`, `N*s/m`, `rad/s`, `ms`, ...). Affine units such
as `degC` are refused. Symbol display names are restricted LaTeX (Greek letters
and a few decorations; no `\input`, environments or `$`).

Validation never repairs a specification. Text, including LaTeX, and
`ciw.exploratory-derivation.v1` records are refused with `latex_is_not_a_model`.

### Lowering and unit-aware evaluation

`lower()` compiles a validated specification to `ciw.model-lowered.v1`: symbols
with explicit SI scales, parameter values and expression trees. Every executor
evaluates `x_si = scale * x_declared`, evaluates the unit-aware SI expression,
and returns `dx/dτ = rhs_si * scale(τ) / scale(x)` in declared units. The unit
parser lives only in Python; the Julia worker receives explicit scales.

## Consistent rescaling

`ciw model rescale` (`ciw.model.transform.rescale`) changes declared units
together with everything that depends on them. For each rescaled symbol
`value_new = k * value_old`, `k = scale_old / scale_new`:

- parameter values and validity bounds;
- covariance entries `P'_ij = k_i k_j P_ij`; process-noise density additionally
  divided by the independent-variable factor;
- in a request: initial state, inputs, sample grid, `dtmax`, **per-state solver
  absolute tolerances** and the acceptance thresholds.

Equations are unchanged because they are unit-aware. The tests show metres →
millimetres invariance of trajectories, observations and SI covariance; that
relabelling `m` as `mm` without converting values is a different model; that
converting states without converting the solver's absolute tolerance changes
Tsit5's accepted steps; that seconds → milliseconds rescales the grid and process
noise; and that linearizations transform by similarity, `A' = K A K⁻¹ / k_t`.

## Typed-port composition

`ciw model compose` (`ciw.model.compose.compose`) qualifies component symbols
(`plant.q`) and connects out-ports to in-ports only when role, quantity kind,
dimension and frame agree, the independent-variable kind and origin agree, and a
measured connection carries exactly the calibration identity the input declares.
Units may differ: a connected input becomes a derived quantity whose unit-aware
expression is the source, so the conversion is part of the model. A measured
input fed by an observation receives the **noise-free model-predicted
observation**; the connection record says so and keeps sensor and calibration
lineage. Cycles through connected ports are algebraic loops and are refused.
Covariance blocks combine block-diagonally under the recorded
`independent_component_uncertainty` assumption.

Tested properties: the millimetre PD controller composed with the metre plant
matches a hand-written all-SI model (relative 1e-10); composition is associative
up to its derivation record; rescaling commutes with composition; time and
path-length models, different origins, role/frame/quantity/dimension/calibration
mismatches, double connections and algebraic loops are refused.

## LaTeX views

`ciw model latex` renders `ciw.model-latex-view.v1` from the structured model:
dynamics, derived quantities, observation model with sensors and calibrations,
uncertainty assumptions and covariance blocks, validity constraints with units,
and a symbol table. With `--run`, a `ciw.model-binding.v1` links each symbol to
its declared unit, value, standard uncertainty, source (`declared_in_specification`,
`requested_initial_condition`, `requested_operating_point`, `estimated`, `measured`) and evidence identities;
a unit that differs from the declaration is refused with "rescale the model
instead of relabelling". Without a binding, the document states that it is an
unbound model view. The view is never parsed back.

`ciw model exploratory` retains handwritten LaTeX as
`ciw.exploratory-derivation.v1` (`executable: false`, `bound_to_run: null`,
content identity). It appears only in a separate section as escaped source, so it
cannot be mistaken for equations bound to a run or inject commands into a report.
The generated documents compile with `pdflatex -no-shell-escape` in the tests.

The ModelingToolkit route also returns Symbolics/Latexify LaTeX of the compiled
equations as a second derived view; linking symbols to units, estimates and
evidence remains the CIW view's job.

## Julia worker and the SCR seam

The worker lives in [`src/ciw/model/julia`](../src/ciw/model/julia) and ships in
the wheel: `Project.toml`, the machine-generated `Manifest.toml` produced by
instantiating and testing the environment, `src/*.jl` and `bin/worker.jl`.
The worker's provider descriptor
[`src/ciw/pipelines/providers/julia-model-worker.json`](../src/ciw/pipelines/providers/julia-model-worker.json)
is its pin definition: `pipelines.check()` binds its `Project.toml`, `Manifest.toml`
and worker-source digests and its operations to the packaged worker, and its
SCR boundary pin is the revision the gate checks out. It pins
Julia **1.10.12 LTS** (tarball SHA-256, executable and system-image digests on
Linux x86-64) and the package versions:

| Profile | Packages | Operations |
| --- | --- | --- |
| `core` | OrdinaryDiffEqTsit5 2.1.4, SciMLBase 3.56.0, ForwardDiff 1.4.6, ControlSystemsBase 1.22.0, JuMP 1.31.2, HiGHS 1.25.4 | `ciw.model.simulate.v1`, `ciw.model.linearize.v1`, `ciw.model.measurement-selection.v1` |
| `symbolic` | core plus ModelingToolkit 11.45.1, Symbolics 7.41.0, Latexify 0.16.12 | `ciw.model.symbolic.v1` |

**Handshake.** The worker reports protocol, profile, Julia version, kernel/arch,
thread and BLAS-thread counts, worker-source digest (every `src/*.jl` and
`bin/worker.jl`), project/manifest/preferences digests, loaded package versions,
per-operation descriptor digests, executable and system-image digests and CPU
target. Its runtime digest is SHA-256 of that identity's CIWB bytes. The host
recomputes the digest, compares every field with the pin and its own hash of the
bound executable, and refuses a mismatch before any work.

**Requests.** Each request is an SCR `ExecutionSpecification` triple. Program
bytes are the registered operation descriptor followed by
`runtime: sha256:<runtime digest>`, so a different environment is a different
program; the worker refuses (SCR `unrunnable`) any program not registered to its
own runtime. Configuration and input are CIWB v1 bytes: fixed-width little-endian
integers and binary64 values, explicit array dimensions, UTF-8 strings and maps
with strictly increasing keys. Every result-affecting solver setting is explicit
(Tsit5, `reltol`, per-state `abstol`, `maxiters`, `dtmax`, automatic initial
step, default controller and norm, interpolated `saveat`, span from first to last
sample). The worker recomputes the specification, program, input, output and
computation identities with SCR's canonical commitment; the host recomputes them
again and refuses any disagreement. Completed runs return output bytes; halted
runs (malformed input 2, solver unsuccessful 3, nonfinite/domain 4, incomplete
sample coverage 5) return none, and no output or computation identity exists.

**Lifecycle.** `julia --startup-file=no --threads=1 --project=<worker>` with
`JULIA_LOAD_PATH=@:@stdlib`, `JULIA_PKG_OFFLINE=true`, one BLAS thread and an
optional explicit depot. Frames are `CIWF` + u32 length + CIWB payload, at most
64 MiB; stdout carries only frames (accidental prints are redirected to stderr).
One request is in flight; each request compiles fresh closures and solver state.
A timeout, malformed or truncated frame, identity mismatch, out-of-order
occurrence, EOF or process failure ends the session, fails that occurrence and is
never retried; later work starts a new session identity.

**SCR dispatch.** `ciw.model.scr_bridge.make_runner` supplies a
`SpecificationDispatcher.runner` returning SCR's own `ExecutionResult`.
`tests/test_model_scr_boundary.py` runs SCR's candidate → policy → dispatcher →
`run_experiment_step` admission with the Julia worker: the result is admitted as
`simulation:` evidence, two executions admit the same observation identity, and
a halted run admits nothing. SCR writes `engine_occurrence` into the admitted
record's content identity and its engine is one fresh process per request (always
0). Because every worker request uses fresh solver state, the bridge reports
engine occurrence 0 and keeps the worker session and occurrence number in a
separate ledger and in CIW's run records, outside every content identity.

## Operations and independent checks

| Operation | Julia computation | Independent Python check |
| --- | --- | --- |
| `simulate` | Tsit5 at the requested sample times | Fixed-substep RK4 on the same lowered model; exact sample-time coverage; componentwise `atol + rtol*|rk4|` thresholds declared in the request |
| `linearize` | ForwardDiff `A, B, C, D`; ControlSystemsBase poles, controllability and observability ranks | Central finite differences; NumPy eigenvalues and SVD ranks |
| `measurement-selection` | ForwardDiff sensitivities through Tsit5; HiGHS MILP maximizing `min_j Σ_i w_i G_ij`, `G_ij = (∂y_i/∂θ_j)² σ_j² / r_i`, under cost and count budgets | RK4 central-difference sensitivities on a declared maximum step; exhaustive enumeration of all feasible subsets (≤ 20 candidates) |
| `symbolic` | ModelingToolkit `System` + `mtkcompile`; Symbolics Jacobian in declared order and the compiled Jacobian mapped back | Central-difference Jacobian; state-order permutation check |

The design objective is dimensionless and therefore unit-invariant; it uses
marginal prior and noise variances and **refuses** correlated design parameters
or observations instead of discarding correlations.

### Measured results (Linux x86-64, 2026-09-23)

| Check | Result |
| --- | --- |
| Default oscillator (768 samples, reltol 1e-10, abstol 1e-12) vs analytic oracle | max error q 1.26e-11 m, v 6.34e-11 m/s, E 1.67e-10 J (declared thresholds 1e-9, 5e-9, 1e-8); 2127 accepted steps, 12765 f evaluations |
| Tsit5 vs RK4 reference, same fixture | q 1.26e-11 m, v 6.36e-11 m/s |
| Tolerance sweep reltol 1e-6 / 1e-9 / 1e-12 | max q error 1.85e-7 / 1.26e-10 / 1.27e-13 m; accepted steps 325 / 1333 / 5381 |
| Linearization vs central differences | max \|ΔA\| 4.8e-11; poles `-0.15 ± 5.0243i` s⁻¹; ranks 2/2 |
| Measurement selection (10 candidates, budget 6, ≤ 3) | HiGHS selects t = 2, 3, 8 s; objective 770.996 equals the unique enumerated optimum; AD vs FD sensitivities 1.4e-7 |
| ModelingToolkit route | `mtkcompile` orders unknowns `[v, q]`; mapped Jacobian matches FD to 4.8e-11 |
| Worker startup / first call / warm call (core) | ≈ 5.8 s handshake; ≈ 6.2 s first simulate (compilation); ≈ 0.05 s warm |
| Replay on the same host | byte-identical output, new execution and result identities |

These are numerical comparisons on one host. Cross-platform numerical agreement,
Windows execution and bitwise stability across CPUs are not claimed; the Windows
runtime digests are pending a native gate run.

## Retained runs: `ciw.model-run.v1`

A run links scientific outputs (the JSON view decoded from committed output
bytes), numerical diagnostics (solver statistics and the reference comparison),
physical measurements (`not_acquired` for these simulation-only operations) and
execution provenance (exact base64 program/configuration/input/output bytes, SCR
identities, worker session, occurrence, timings and the runtime record).
Refused and failed attempts are retained with their reason and no result.

`ciw model inspect` needs no Julia: it re-derives configuration and input bytes
from the retained specification and request, recomputes every SCR identity,
checks the derived view against the decoded output and checks the record seal.
`ciw model replay` requires a worker whose runtime digest matches the retained
program bytes and records a fresh occurrence with byte-equality and numerical
differences against the original.

## Commands

```sh
python -m ciw model validate examples/model-core/damped-oscillator.json
python -m ciw model rescale examples/model-core/damped-oscillator.json \
  --unit q=mm --unit v=mm/s --unit y=mm --output results/oscillator-mm.json \
  --request examples/model-core/oscillator-simulate.json --request-output results/simulate-mm.json
python -m ciw model compose --component plant=examples/model-core/damped-oscillator.json \
  --component ctrl=examples/model-core/pd-controller.json \
  --connect plant.position ctrl.measurement --connect ctrl.command plant.force \
  --model-id closed-loop --title "Oscillator under filtered PD control" --output results/closed-loop.json
python -m ciw model exploratory --title "Energy decay" --latex-file examples/model-core/energy-derivation.tex \
  --relates-to examples/model-core/damped-oscillator.json --output results/energy-note.json
python -m ciw model run simulate examples/model-core/damped-oscillator.json \
  examples/model-core/oscillator-simulate.json --julia /opt/julia-1.10.12/bin/julia --output-dir results/model-runs
python -m ciw model inspect results/model-runs/<run>.json
python -m ciw model replay results/model-runs/<run>.json --julia /opt/julia-1.10.12/bin/julia
python -m ciw model latex examples/model-core/damped-oscillator.json --run results/model-runs/<run>.json \
  --exploratory results/energy-note.json --output results/oscillator.tex
```

`run` operations are `simulate`, `linearize`, `select` and `symbolic`; the example
requests are `oscillator-simulate.json`, `oscillator-linearize.json` (also used by
`symbolic`) and `oscillator-design.json`. Exit status 3 means the execution did
not complete; 4 means the reference comparison did not pass.

## Provisioning and validation

Instantiate the worker environment once, separately from execution:

```sh
JULIA_DEPOT_PATH=/path/to/depot julia --startup-file=no --project=src/ciw/model/julia \
  -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

Offline tests run in the ordinary suite. The genuine worker tests need
`CIW_JULIA` (and optionally `CIW_JULIA_DEPOT`); the SCR boundary tests also need
`CIW_SCR_REPO` at revision `a59aba283b0304faeeb3e5d305087e7709e171ca`. The
installed-wheel gate downloads and checks the pinned Julia distribution when
`--julia` is not supplied, instantiates the committed manifest into a fresh depot,
verifies the manifest is unchanged, and fails on any skipped test:

```sh
python scripts/check_model_core.py --output-dir results/model-core-gate
```

It requires Python 3.11+, `setuptools>=77` with `packaging>=24.2` (the wheel is
built without isolation), Git, network access for the Julia registry and SCR,
and `pdflatex` unless `--without-latex-compile` records that check as not run.

## Limits

- Inputs are constant over a run; time-varying input schedules are not yet part
  of the request contract.
- Only explicit ODEs; no algebraic equations, events or discontinuities.
- The Python reference is fixed-step RK4 and is itself only a numerical check,
  not an error bound.
- A composed measured input receives the noise-free predicted observation.
  Stochastic simulation, estimators and observers are not implemented.
- The measurement-selection objective uses marginal variances and local
  (linearized) sensitivities at the declared parameter values.
- Run records are not yet in the shared live session or Godot desktop.
- Windows and macOS runtimes are not pinned or tested.
