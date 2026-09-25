# Julia oscillator numerical provider

`ciw.julia-oscillator.v1` is the second increment of the
[Julia and SP1 development contract](JULIA_SP1.md). A host-owned persistent
Julia worker numerically integrates the damped oscillator with
`OrdinaryDiffEqTsit5`, CIW retains the execution as an SCR-shaped execution
specification with byte commitments, and the existing closed-form oscillator
in `ciw.adapters.oscillator` is the independent numerical oracle. The retained
trajectory is displayed by the existing Godot oscillator viewport without any
recomputation.

The result is a **simulation**. It is not a physical observation, a calibrated
measurement, an inferred covariance, an independently verified physical result
or authority over equipment. Oracle agreement is a numerical claim about the
declared model at the requested samples; it is not physical validation.

## Model and profile

```text
q' = v
v' = -2*gamma*v - omega_0^2*q
E  = 0.5*mass*(v^2 + omega_0^2*q^2)
state order [q, v]; initial state at t = 0 s
```

| Field | Bound | Notes |
| --- | --- | --- |
| `omega_0_rad_s` | `0 < omega_0 <= 20` | |
| `gamma_s_inv` | `0 <= gamma <= 0.5*omega_0` | underdamped and undamped only; no critical or overdamped case |
| `mass_kg` | `0 < mass <= 100` | |
| `initial_q_m`, `initial_v_m_s` | `abs(q0) <= 10`, `abs(v0) <= 100` | |
| grid | `start_s = 0`, `2 <= sample_count <= 4096`, `sample_count / sample_rate_hz <= 12 s`, endpoint excluded | times are `k / sample_rate_hz` |
| `abstol`, `reltol` | `[1e-14, 1e-2]` | binary64 |
| `dt_initial`, `dtmax` | `0` (solver default) or `[1e-9, 12]` | |
| `maxiters` | `[1000, 1e8]` integer | |
| `output_policy` | `saveat_interpolated` or `tstops_stepped` | see below |

Booleans, `NaN`, infinities, unknown fields, out-of-bound values and any
executable path in a source are refused before dispatch. The worker re-checks
the same bounds and refuses independently. Only `Tsit5`, adaptive stepping and
these two output policies are registered.

`saveat_interpolated` lets the adaptive solver choose its steps and evaluates
the requested times through Tsit5's free fourth-order interpolant.
`tstops_stepped` forces a step boundary at every requested time. Both are
explicit configuration bytes and change the specification identity.

## Source and retained records

The source schema is `ciw.julia-oscillator-source.v1`:

```json
{
  "schema": "ciw.julia-oscillator-source.v1",
  "experiment_id": "demo-julia-oscillator-default",
  "configuration": {
    "solver": {"algorithm": "Tsit5", "abstol": 1e-8, "reltol": 1e-8, "dt_initial": 0.0, "dtmax": 0.0,
               "maxiters": 1000000, "output_policy": "saveat_interpolated"},
    "oracle_policy": {"reference": "analytic-damped-oscillator.v1",
                      "max_normalized_error": {"q": 1e-7, "v": 1e-7, "energy": 1e-7}},
    "replay_policy": {"max_normalized_difference": 1e-12},
    "origin": "simulation",
    "covariance_status": "not_applicable"
  },
  "model": {"omega_0_rad_s": 5.026548245743669, "gamma_s_inv": 0.15, "mass_kg": 1.0,
            "initial_q_m": 1.0, "initial_v_m_s": 0.0},
  "grid": {"start_s": 0.0, "sample_rate_hz": 64.0, "sample_count": 768, "endpoint": "excluded"}
}
```

The retained bundle is `ciw.julia-oscillator-session.v1` with the same
`source`, `configuration`, `runtimes`, `steps`, `bundle_digest` and
`verification` layout as the other shared workloads. Its single step retains:

| Field | Content |
| --- | --- |
| `result.data.native.specification` | SCR execution specification: `program` (the descriptor bytes), `configuration` (`ciw.julia-oscillator-configuration.v1`) and `input_payload` (`ciw.julia-oscillator-input.v1`), all hex |
| `program_identity`, `input_identity`, `output_identity`, `specification_identity`, `computation_identity` | SCR's `scout.execution.*.v1` commitments recomputed by CIW from the retained bytes; the specification identity binds the solver configuration, the computation identity does not |
| `result.data.native.output` | exact committed output bytes (`ciw.julia-oscillator-output.v1`) |
| `result.data.native.decoded` | derived view: requested `time_s`, `q`, `v`, `energy`, solver return code, accepted/rejected steps and function evaluations |
| `result.data.occurrence` | worker session identity, occurrence number, exact request frame, exact response frame and the worker's occurrence metadata |
| `numerical_result` | operation and the `native` block only; occurrence details are excluded |
| `verification` | `ciw.julia-oscillator-verification.v1` oracle comparison |

Encodings are little-endian binary64 with explicit magic, version and field
order; the complete layout is the program descriptor `PROGRAM` in
`src/ciw/julia_oscillator.py`. JSON and Python objects are derived views;
offline validation decodes the retained bytes and requires the decoded view,
the commitments, the request and response frames and the worker's energy
derivation to match exactly.

## Oracle verification

`verification.oracle` retains the closed-form `q`, `v` and `energy` at the
requested times as computed by `ciw.adapters.oscillator.analytic_trajectory`
on the executing host, with the NumPy version and a content digest.
`verification.measured` records, per channel, the maximum absolute error, the
sample index where it occurs, the reference scale `max(|reference|)` and the
normalized maximum error (their ratio). Only subtraction, absolute value,
maximum and one division are used, so any binary64 host can recheck the
retained statement bitwise without Julia or the original NumPy.

`outcome` is `passed` when every normalized error is at most the threshold
declared in the source's `oracle_policy`, otherwise `failed`. A failed
comparison is retained as a failed verification; it is neither turned into a
success nor discarded. For `gamma = 0` the record also measures the numerical
energy drift `max|E(t) - E(0)| / E(0)` of the Julia trajectory itself. Solver
tolerance and measured oracle error are separate fields; the requested
tolerance is not a global error bound.

Measured on Linux x86-64 with the pinned environment (normalized maximum
error; 768 samples at 64 Hz over `[0, 12)`):

| Fixture | `abstol`/`reltol` | policy | q | v | energy | accepted steps | outcome |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `default` | 1e-8 / 1e-8 | interpolated | 4.7e-9 | 5.1e-9 | 3.5e-9 | 630 | passed (threshold 1e-7) |
| `mixed-initial-state` (`q0 = -0.6`, `v0 = 2.5`) | 1e-8 / 1e-8 | interpolated | 6.1e-9 | 6.0e-9 | 3.9e-9 | 608 | passed (1e-7) |
| `undamped` (`gamma = 0`) | 1e-8 / 1e-8 | interpolated | 1.5e-8 | 1.5e-8 | 2.7e-8 | 697 | passed (1e-7); energy drift 2.7e-8 |
| `loose-tolerance` | 1e-6 / 1e-3 | interpolated | 7.8e-4 | 8.2e-4 | 2.7e-4 | 81 | passed (1e-2) |
| `tight-tolerance` | 1e-12 / 1e-12 | interpolated | 4.4e-13 | 4.6e-13 | 3.5e-13 | 3974 | passed (1e-10) |
| `stepped-output` | 1e-8 / 1e-8 | stepped | 1.2e-10 | 1.3e-10 | 1.2e-10 | 1545 | passed (1e-8) |

Tighter tolerance produced smaller error and more work in these fixtures. That
ordering is an observation about these runs, not a proportionality law. The
fixture sources are in `examples/julia-oscillator/`; the gate rewrites these
numbers from a fresh run every time.

## Runtime and environment

| Component | Pinned identity |
| --- | --- |
| Julia | 1.10.12 (LTS); the worker reports `VERSION`, machine triple, kernel, word size, thread count and startup-file, fast-math, bounds-check and optimization settings |
| Packages | `OrdinaryDiffEqTsit5 2.1.4`, `OrdinaryDiffEqCore 4.18.0`, `DiffEqBase 7.21.2`, `SciMLBase 3.56.0`, `SHA 0.7.0`, `TOML 1.0.3`, with UUIDs and git tree hashes |
| Environment | `src/ciw/julia/Project.toml` and the machine-generated `Manifest.toml`, shipped as package data |
| Worker | `src/ciw/julia/worker/oscillator_worker.jl` |
| Pin | `src/ciw/julia-runtime.json`; regenerate with `python scripts/pin_julia_runtime.py` and check with `--check` |

The host digests the packaged worker, project and manifest and refuses to
start if they differ from the pin. The worker's hello frame reports its own
digests, Julia version, package identities and system image digest; the host
compares every field with the pin before accepting work. The retained runtime
identity also records the resolved Julia executable's SHA-256. A differing
Julia version, package version, worker source, thread count, startup file or
fast-math setting is an explicit `RUNTIME_PIN_MISMATCH` refusal, never a
substitution. The executable digest covers the launcher binary only; it is an
identity of the binding, not a build attestation of `libjulia` or the depot.

Provision once, outside any request:

```sh
JULIA_LOAD_PATH="@" julia --startup-file=no --project=src/ciw/julia \
  -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
python scripts/pin_julia_runtime.py --check
```

For an installed wheel, use `python -c "import ciw.julia_worker as w; print(w.environment_root())"`
as the project path. The worker runs with `--startup-file=no --history-file=no
--threads=1 --project=<packaged environment>` and `JULIA_LOAD_PATH="@"`, so
only the pinned project and its manifest are visible. No package is resolved,
installed or updated while serving; the depot is operator-provisioned.

## Worker protocol

Frames are `[u32 little-endian length][payload]` on stdout and stdin;
diagnostics go to stderr only. The first frame is `ciw.julia-worker-hello.v1`.
Each request is `[CIWQ][u32 version][u64 request_number][operation]
[configuration bytes][input bytes]`; the only registered operation is
`ciw.julia-oscillator.integrate.v1`, and requests carry no Julia source,
package names or executable paths. The worker builds a fresh `ODEProblem` and
solver state per request and answers with two frames: the response
`[OSCO][u32 version][u64 request_number]` header plus the committed output,
then `ciw.julia-worker-occurrence.v1` metadata binding the same request number.

One request is in flight per worker with a bounded wait queue of eight. A
timeout, malformed or oversized frame, occurrence mismatch, unexpected EOF or
process exit fails the active occurrence, terminates and reaps the worker and
ends its session; later work starts a fresh session with a new
`julia-worker-<uuid>` identity and occurrence numbers from one. A worker
refusal (bad bounds, unsupported operation, solver failure, nonfinite output,
incomplete coverage) is a refused occurrence without a result and keeps the
session alive. Nothing is retried silently.

## Bind, run, inspect, replay

```sh
python -m ciw serve --julia-executable /path/to/julia-1.10.12/bin/julia --output-dir results/julia
python -m ciw send source.add --payload-file source-payload.json
python -m ciw send operation.execute --payload '{"operation_id":"ciw.julia-oscillator.v1","parameters":{"source_id":"source:sha256:..."}}'
python -m ciw send experiment.inspect --payload '{"bundle_id":"sha256:..."}'
python -m ciw send bundle.replay --payload '{"bundle_id":"sha256:..."}'
```

Terminal-only equivalents:

```sh
python -m ciw julia-oscillator run --source examples/julia-oscillator/default.json \
  --julia-executable /path/to/julia --output-dir results/julia/default
python -m ciw julia-oscillator inspect results/julia/default/bundle.json
python -m ciw julia-oscillator replay results/julia/default/bundle.json \
  --julia-executable /path/to/julia --output-dir results/julia/default-replay
python -m ciw julia-oscillator recording results/julia/default/bundle.json --output results/julia/recording.json
```

`inspect` validates a retained bundle and prints identities, solver work and
measured errors without Julia. Reopening a workspace with retained Julia
bundles needs no Julia either; execution and replay need the explicit
`--julia-executable` binding again, and the binding is never read from a file.

Replay is an explicit fresh occurrence on a runtime whose projection matches
the retained one. Its `ciw.julia-oscillator-replay.v1` receipt claims
agreement under the source's `replay_policy` normalized tolerance and records
`byte_identical` separately. On the same host the pinned solver produced
byte-identical output in every exercised replay; cross-host byte identity is
not claimed. A fresh execution outside the declared tolerance is refused
without retaining a bundle.

## Viewport

`ciw julia-oscillator run` also writes `recording.json`, a `run.v1` projection
with instrument `julia-oscillator-trajectory.v1`, the oscillator state frame,
`q`/`v`/`energy` channels, the retained sample times and backend-prepared
render geometry. Its provenance names the bundle, result, execution,
specification and computation identities, the oracle verification and outcome,
and declares `origin: simulation`. Serve it with
`python -m ciw serve --recording results/julia/recording.json` and the existing
Oscillator tab displays the phase portrait, timeline cursor and `q`, `v` and
`energy` cards from `sample.get`; `analysis.stats` and `analysis.spectrum`
operate on it through the registered projection adapter. The Godot client
recomputes nothing: the ODE, energy, oracle errors and solver diagnostics all
come from retained records. `experiment.inspect` returns the oracle-error and
solver-work panels plus two `ciw.state-trajectory-projection.v1` objects (the
Julia simulation and the analytic reference) for the Workbench tab.

`python scripts/check_godot.py --godot <godot> --julia-recording recording.json --julia-view view.json`
exercises the headless viewport against actual retained files.

## Acceptance gate

`tests/test_julia_oscillator.py` covers the source contract, byte encodings,
commitments, retained-bundle faults, offline restore, projections and the
worker protocol through a labelled Python double that never runs Julia.
`tests/test_julia_oscillator_session.py` runs the real worker when
`CIW_JULIA_EXECUTABLE` is set: the default fixture against all 768 analytic
samples, mixed initial state, undamped drift, tolerance profiles, A/B/A worker
isolation and restart, refusals at the source and worker, timeout and crash,
wrong environment, offline restore then explicit replay, the viewport
projection and the terminal commands. `scripts/check_julia_oscillator.py`
runs them from an isolated installed wheel, fails on any skip, and writes
`gate.json` with the measured errors; the
[workflow](../.github/workflows/julia-oscillator.yml) provisions Julia 1.10.12
on Linux and Windows.

## Limitations

- Non-stiff underdamped and undamped cases only; no critical/overdamped,
  stiff, forced or nonlinear profiles, and no solver other than `Tsit5`.
- Uniform grids from `t = 0` only; explicit sample-time lists are not yet
  accepted.
- Same-host byte stability was observed; cross-platform numerical agreement
  needs its own fixtures. The Windows job in CI is the gate for native Windows
  execution; this increment was exercised on Linux x86-64.
- Oracle agreement is not physical validation, measurement uncertainty or
  calibration. No covariance is inferred from solver tolerance.
- The worker is not a sandbox; the Julia executable and depot are trusted host
  configuration.
- Dispatch does not pass through the SCR Python `SpecificationDispatcher`; CIW
  builds the SCR execution specification and commitments itself, exactly as
  the retained integer heat records are checked. Routing through SCR's
  dispatcher with this worker as its runner is a later SCR-side step.
