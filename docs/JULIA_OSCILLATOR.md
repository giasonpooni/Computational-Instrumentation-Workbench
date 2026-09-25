# Julia Tsit5 oscillator

`ciw.julia-oscillator.v1` is the first Julia provider seam in the workbench. It
accepts a bounded, canonical data request for

```text
q' = v
v' = -2 gamma v - omega_0^2 q
E = 0.5 mass (v^2 + omega_0^2 q^2)
```

The worker runs `OrdinaryDiffEqTsit5` under an explicit Julia 1.10 LTS project,
with startup code disabled and one Julia thread. Requests contain no Julia
source, package names or executable paths. The worker uses a length-prefixed
JSON frame on stdout and writes diagnostics only to stderr. The host rejects
unknown fields, nonfinite or out-of-bounds values, incomplete trajectories and
unexpected worker frames.

CIW retains the canonical source bytes, framed request and response bytes,
runtime identity, execution occurrence and result identity separately. The
Python analytic oscillator is an independent numerical oracle. Its componentwise
comparison is a scoped verification of the simulated trajectory; it is not a
physical measurement, calibration, uncertainty estimate or authorization.

## Provision the binding

The runtime checkout must contain the tracked files below and a generated,
machine-written `Manifest.toml` produced by Julia 1.10.12:

```text
runtimes/julia-oscillator/Project.toml
runtimes/julia-oscillator/Manifest.toml
runtimes/julia-oscillator/oscillator_worker.jl
```

From the runtime directory, after installing Julia 1.10.12:

```text
julia --startup-file=no --project=. -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"
```

The manifest must be committed only after this command completes and the
worker's identity reports the same project, manifest, source and executable
digests. A hand-written or floating lockfile does not satisfy this gate.

## Run and replay

```text
ciw julia-oscillator create --source examples/julia/oscillator.json `
  --julia-oscillator-runtime . `
  --julia-executable C:/Julia/julia-1.10.12/bin/julia.exe `
  --output results/julia-oscillator/session.json `
  --recording-output recordings/julia-oscillator.json
ciw julia-oscillator inspect results/julia-oscillator/session.json
ciw julia-oscillator replay results/julia-oscillator/session.json `
  --julia-oscillator-runtime . `
  --julia-executable C:/Julia/julia-1.10.12/bin/julia.exe `
  --output results/julia-oscillator/replay.json
```

The same binding can be supplied to `ciw serve` with
`--julia-oscillator-runtime` and `--julia-executable`. The CLI's optional
`--recording-output` writes the retained trajectory as a normal `run.v1`
recording that can be opened by the existing read-only Godot phase/state
viewport. The viewport derives its trajectory and channel list from the
retained record; it never becomes a numerical source.

## Current gate status

The Python contract, framing parser, oracle comparison, raw-byte retention,
offline validation and replay identity checks are implemented and covered by
`tests/test_julia_oscillator.py`. Genuine Julia execution, generated-manifest
identity, cross-platform replay, and headless Godot execution remain release
gates until the external runtimes are provisioned on the operator machine.
