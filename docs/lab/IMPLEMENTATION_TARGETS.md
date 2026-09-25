# Implementation targets: Rust, Python, Julia, C++, GPU and FPGA (T142-T154)

Section 10 of the computational-experimentalist queue decides where each kind
of work should run and fixes the contracts that let results cross language and
device boundaries without losing their evidence identity. Code:
`src/ciw/lab/implementation_targets.py` (tasks) with helpers
`implementation_targets_kernels.py` (operation counts, reductions, comparison
harness), `implementation_targets_serial.py` (canonical JSON, Rust probe),
`implementation_targets_architecture.py` (C++ inventory, import-graph scan),
`implementation_targets_fpga.py` (telemetry frames, bitstream identity,
compatibility, link simulation), `implementation_targets_authority.py`
(actuator write policy, control proposals), and for T145
`implementation_targets_julia.py` (the oscillator acceptance set) with the
worker host `julia_worker.py` and the Julia environment `julia/` (worker
source, `Project.toml`, the machine-generated `Manifest.toml` and the runtime
pin `julia-runtime.json`, shipped as package data so the clean room has them).
T147 also uses the common Gaussian
VI workload of the energy section (`energy_gpu_workload.py`, described in
[ENERGY_GPU.md](ENERGY_GPU.md#the-common-workload)), whose PTX kernel is the
one GPU implementation the workbench has. Tests:
`tests/test_lab_implementation_targets.py`.

Run the section with

```
python -m ciw lab run T142 T143 T144 T145 T146 T147 T148 T149 T150 T151 T152 T153 T154 --output-dir results/lab-implementation-targets \
    --provider julia=/opt/julia/julia-1.10.12/bin/julia --provider julia-depot=/opt/julia-depot \
    --provider scr=/trusted/references/scr   # T145's worker behind SCR; optional
```

Without the Julia bindings it takes about 6 s on one core; with them T145
starts four Julia workers (about 4 s each on the retained host, mostly package
loading and compiling the worker; reading the 630 MB runtime tree for the
handshake check adds about 0.6 s once per run) and the section takes about
29 s on one core, within the 60 s budget. T142 and T146 compile a small Rust
probe
(standard library only) with `rustc` when it is on `PATH`:

- `rustc` absent, or unable to build a trivial program here: the Rust findings
  are recorded as honestly unestablished and the tasks become `partial`;
- `rustc` works but the probe fails to build or run: the Rust finding is
  refuted (a defect of the probe, not a missing tool), and the Rust tests fail
  instead of skipping.

The probe is compiled with `--remap-path-prefix=<build dir>=ciw-lab-rust`, so
its binary digest (part of the provider/runtime identity) is the same on every
run. T142's report identity still changes between runs because its retained
wall-clock timing artifacts do; no finding depends on them. Its figure
`kernel-timings.svg` is declared as a wall-clock timing figure, so figure
re-executions (T158, `scripts/check_figures.py`) compare it for presence and
structure only.

**What this section does not establish.** No GPU, FPGA or industrial C/C++
library runs here, and Julia runs only where a provisioned runtime is bound,
on the platform of that run (Windows is not run). Bitstreams, link statistics
and authorization records are synthetic placeholders. Rust agreement is
agreement between two CIW-authored implementations (`cross_implementation`),
not independent verification. Every claim about physical links, deployment,
production acceptance, machine safety, industrial readiness or actuator
authority is recorded as a finding in its proper domain and is
`not_established`.

Independent checks name their ciw-side producer or checker by package version
and module digest (`{"implementation": ..., "revision": "ciw 0.1.0",
"source_sha256": ...}`, T145, T146, T148 and T149), as the external sides name
theirs.

Headline labels (weakest established computational label per task):
`numerically_verified` for T142, T144, T146, T147, T149-T154 and T145 with the
Julia runtime bound; `analytic` for T143 (a design inventory), T145 without
the Julia bindings (the pin procedure as a derivation) and T148 (the
reduction policy record is a derivation, although every number in T148 is
exact). T145 and T147 are `partial`. T145 is partial because Windows
x86-64 execution of its worker is not run (its recommended next task) and,
without the `julia` and `julia-depot` bindings, because no Julia process runs
at all. T147's GPU
comparison runs the common workload's PTX kernel wherever an NVIDIA GPU
answers the `hardware:nvidia-gpu` probe; here none did, so that finding is
`not_established` for that reason, and T147 becomes `completed` in the RTX 2080
host run of the energy section's protocol (retained with `ciw lab hardware
retain`).

Recommended next steps name the work that delivers, never a task that has
already run: a completed task's next step is its own open question, marked
"Deferred research question", or says what it already did (T142). T148's is
the one deferred question that owns the missing GPU code of the common
workload (a float32 rendering of the kernel and a device-wide reduction under
`REDUCTION_POLICY`), shared with T117, T120, T121 and T147.

## Kernel ranking

T142 counts floating-point operations exactly by running scalar restatements of
each kernel on a counting number type (`Scalar`). The geodesic RHS, Jacobi RHS,
RK4 step and transfer restatements are checked against the core NumPy kernels
(max difference below 1e-12, a same-origin `cross_implementation` check). The
Kalman row is different: the core Kalman filters (in the sensor-fusion and
observation sections) are vectorized across runs and have no single-update
kernel, so the row profiles a Joseph-form reference written in this section
and is labelled "not a core kernel". Interpreter calls issued by `ciw` code per
kernel call are counted with a profile hook. Wall-clock timings are kept only
in `kernel-timings.json` and its declared timing figure `kernel-timings.svg`.

| Kernel (restated) | Flops per call | Trig calls | Interpreter calls per call |
| --- | --- | --- | --- |
| Generic embedded-surface geodesic RHS (torus) | 173 | 4 | 73 |
| Geodesic + Jacobi RHS | 178 | 4 | 79 |
| RK4 step on the 8-state system | 819 = 4 x 178 + (13n + 3), n = 8 | 16 | 317 |
| Joseph-form Kalman update, n = 4, m = 2 (CIW reference, not a core kernel) | 453 | 0 | 5 |

The ranking is by Python dispatches that porting a unit alone removes from one
reference experiment (a 2000-step Jacobi transfer plus 2000 Kalman updates);
callbacks that stay in Python are not removed. Ties would be broken by
arithmetic intensity (flops per byte of state, stage vectors and matrices read
or written per call, higher first), then by determinism need. The units are
nested, not alternatives: `jacobi.rhs` calls `surface.geodesic_rhs`, so the
fused transfer loop contains all 8000 geodesic RHS calls and all RK4 steps and
ranks above them by construction. The ranking finding checks that rank 1 leads
rank 2 by at least 5% of its removable dispatches (observed 9.7%, which is the
Jacobi and RK4 glue outside the contained geodesic RHS calls) and that no tie
occurs, so a change of order across Python or NumPy versions fails loudly
instead of passing as a 25% dispatch drift. The ranking says which enclosing
unit to port, not which disjoint kernel costs most.

That dispatch overhead rather than arithmetic dominates the run time is
recorded as **not established**: only wall-clock timings speak to it, and they
stay in `kernel-timings.json`. The reproducible proxy is interpreter calls per
flop (0.42 for the geodesic RHS, 0.011 for the Kalman update).

| Rank | Port unit | Removable dispatches | Flops per byte | Determinism need |
| --- | --- | --- | --- | --- |
| 1 | Fused geodesic + Jacobi RK4 transfer loop | 638,005 | about 12,800 (2000 steps per call) | high: retained trajectories, bitwise replay over thousands of dependent steps |
| 2 | Generic embedded geodesic RHS | 576,000 | 2.7 | high: fixed contraction order |
| 3 | Kalman update (CIW reference) | 8,000 | 1.05 | medium: symmetric, positive-definite covariance |
| 4 | RK4 step alone (RHS stays in Python) | 0 | 0.19 | high, but negligible alone |

Recommendation: port the fused transfer loop as one Rust kernel; do not port
the RK4 step without its right-hand side; keep the Kalman update in NumPy until
filter rates justify it. A Rust port of the fused loop for the unit sphere
(closed-form Christoffel symbols) reproduces `ciw.lab.jacobi.transfer` to
4.4e-16 after 600 steps. Its retained timing ratio overstates what a generic
port would gain, because the port is specialised to the sphere. Whether a
Rust port saves time or energy on a target machine cannot be a finding from
wall-clock timings; the reproducible route is a RAPL bracket, which
`python -m ciw.lab.energy_gpu_telemetry rapl-capture` now makes for a Rust port
and the NumPy reference of the common Gaussian VI workload (T117 reports it).
A port of the ranked fused transfer loop itself would need its own bracket.

Dispatch counts depend on the Python and NumPy versions (the event semantics
used were checked identical on CPython 3.11-3.13), and array operators are not
counted; the regression tolerance on them is 25%.

## Industrial interfaces

T143 records which interfaces need C/C++ and where the CIW boundary sits.

| Interface | Native libraries | Why native | Boundary | Direction / write path |
| --- | --- | --- | --- | --- |
| OPC UA client (read-only subscriptions) | open62541 (C), vendor C++ SDKs (preferred, not required) | pure-Python stacks exist (`asyncua`; `python-opcua`, superseded by it); C/C++ is chosen for OPC Foundation certification, vendor support and an accepted security-policy implementation | pinned subprocess | read-only / absent |
| EtherCAT passive monitoring (network TAP capture) | libpcap / Npcap, Wireshark EtherCAT dissectors, TAP device driver | line-rate capture with hardware timestamps at microsecond frame spacing | pinned subprocess | read-only (passive tap) / absent |
| Vendor camera SDKs (GenICam GenTL) | Basler pylon, Spinnaker, Vimba X | proprietary drivers, zero-copy buffers, hardware triggers | pinned subprocess | read-only / absent |
| Point clouds | PCL, Open3D | C++ registration and reconstruction kernels with threading | pinned subprocess | geometry exchange / absent |
| CAD kernel | OpenCASCADE | B-rep, STEP/IGES translation, tolerances | pinned subprocess | geometry exchange / absent |

C/C++ is required for four rows and preferred for OPC UA; a preferred entry
must name the pure-Python alternatives it passes over, and either stack would
sit behind the same pinned subprocess boundary. The package probe in the
artifact (`python_packages_present_here`) lists importable Python packages:
bindings to the C/C++ libraries (`open3d`, `OCC`, `pysoem`, `pypylon`,
`PySpin`, `vmbpy`, `pcl`) and the two pure-Python OPC UA stacks (`asyncua`,
`opcua`).

An EtherCAT master (SOEM, IgH, TwinCAT) originates every bus frame, including
output process data, so it can never be read-only; monitoring is modelled as a
passive TAP. That a TAP cannot inject frames is a hardware property assumed,
not verified, here.

The validator refuses an in-process binding, a write-capable direction, an
enabled write path, a fieldbus master in place of a passive tap, an entry
without identity pins, floating pins, an unknown native necessity and a
native preference that does not name its pure-Python alternatives. A pin is floating when it is blank;
when the whole pin is `current`, `main`, `master`, `trunk`, `head`, `dev`,
`develop` or `x`; when it contains the word `latest`, `nightly`, `snapshot`,
`stable`, `any`, `unknown`, `tbd`, `n/a` or `na` (case-insensitive, also after
`_` or `-`, as in `2023.2_nightly`) or `HEAD`; when it contains any of
`* ? < > = ~ ^`; or when it has an `N.x` wildcard such as `2024.x`. Pins are
descriptions of what is pinned, so branch-like words inside prose ("main board
firmware version", "/dev/ttyUSB0 adapter serial number") are accepted; the
list is a denylist and cannot recognise every moving label. All twelve
mutations (including `nightly build`, `SDK version >=1.0`, `2024.x` and `HEAD`)
are refused with their codes. No library was installed or exercised.

## Python orchestration boundary

T144 parses every module of the installed `ciw` package with `ast` (nothing is
imported; 141 modules at the time of writing, identified by a package digest
in the provider/runtime identity) and checks structural rules:

- the transitive `ciw` import closure of `ciw.lab.evidence`, `ciw.lab.report`
  and `ciw.core.identities` uses only the standard library, loads no native
  code and spawns no process. The closure includes every ancestor package,
  because Python executes `ciw/__init__.py` and `ciw/lab/__init__.py` before
  `ciw.lab.evidence` (closure: `ciw`, `ciw.core`, `ciw.core.identities`,
  `ciw.lab`, `ciw.lab.evidence`, `ciw.lab.report`);
- native loading (`ctypes`, `cffi`) appears only in the declared hardware
  probes: the energy probes `ciw.energy_cuda` and `ciw.energy_nvml`, and
  `ciw.lab.blas_probe`, which asks NumPy's loaded OpenBLAS which CPU kernels
  it runs (T094's platform fingerprint and the gate's `--blas-core`);
- no spawn goes through a shell (`shell=True`, `os.system`, `os.popen`,
  `asyncio.create_subprocess_shell`, `subprocess.getoutput`);
- no compiled extension ships in the package.

A separate heuristic rule asks every spawning module (`subprocess`, `os`
spawn/exec functions, `asyncio.create_subprocess_*`, including names imported
with `from ... import`) to name an identity (`revision`, `source_tree`,
`runtime_identity`, `sha256` or `digest`) in identifiers or non-docstring
strings; comments and docstrings do not count. It shows that a name appears
in code, not that the spawned binary is pinned.

Eleven forged mutations are each flagged: a `ctypes` import in the evidence
module, `numpy` in the report module, a spawn in the evidence module, `numpy`
in `ciw/lab/__init__.py`, `ctypes` in `ciw/core/__init__.py`, a spawn in
`ciw/__init__.py` (the three package mutations went unnoticed before the
closure followed ancestor packages), a `shell=True` provider, an asyncio shell
spawn, a spawn without identity, a spawn whose only identity token is in a
comment, and `posix_spawn` imported from `os`.

Whether numerical providers run only pinned executables is **not
established**: a source scan cannot show that a spawned binary equals an
expected identity. Four modules resolve an executable with `shutil.which` and
run it (`ciw.lab.energy_gpu_kernels`,
`ciw.lab.exchange_provenance_bundles_providers`,
`ciw.lab.implementation_targets_serial`, `ciw.lab.runner`); the Rust probe
records `rustc -vV` as provenance but compares it with nothing. This is
retained as a counterexample. A text search for `subprocess` is not an
adequate substitute for the scan either: modules outside `ciw.lab` mention it
without spawning. Dynamic imports, `exec` and spawns inside third-party
libraries are not seen.

## Julia

T145 runs the first Julia operation of [docs/JULIA_SP1.md](../JULIA_SP1.md),
the damped oscillator, behind SCR's execution boundary, and SymPy demonstrates
the symbolic role (the torus Christoffel symbols and Gaussian curvature derived
symbolically agree with `ciw.lab.surfaces.Torus` to 4.4e-16, independent
implementation).

**Runtime and environment.** The pin (`src/ciw/lab/julia/julia-runtime.json`)
is Julia 1.10.12 LTS: the official archive's URL and SHA-256, which must also
be the entry in Julia's published checksum file, the release commit, and per
platform the SHA-256 of the executable (`bin/julia`, `bin/julia.exe`), of the
system image (`lib/julia/sys.so`, `sys.dll`) and of the runtime tree (every
file and symbolic link of the archive's root directory, listed as `<kind>
<sha256> <path>` lines sorted by path), all taken from the verified archives'
members. The worker environment
(`src/ciw/lab/julia/Project.toml`) declares `OrdinaryDiffEqTsit5` 2.1.4,
`OrdinaryDiffEqCore` 4.18.0, `SciMLBase` 3.56.0 and `SHA` with exact compat
entries and `julia = "=1.10.12"`; `Manifest.toml` is Pkg's machine-generated
resolution (63 registered packages with their git tree hashes, 53 of which the
worker loads, plus standard libraries). Provisioning
is separate from execution: `python scripts/provision_julia.py --prefix P
--depot D` downloads and checks the archive, requires the pin's runtime digests
to be those of the archive's members, extracts it (or reuses an extracted
prefix) and requires the extracted tree to hold exactly the archive's files,
instantiates the committed manifest into the depot from the package server (Pkg
verifies each tree hash), precompiles it for the CPU, refuses a run that
changes the committed files, starts the worker once and prints the bindings.
A lab run
binds the executable and the depot explicitly, `--provider julia=P/julia-1.10.12/bin/julia
--provider julia-depot=D` (the clean-room tests read them as
`CIW_LAB_JULIA_EXECUTABLE` and `CIW_LAB_JULIA_DEPOT`); nothing is looked up on
`PATH` or read from a saved workspace. CI's lab job provisions both before the
gate and passes them to `scripts/check_lab.py --julia --julia-depot`.

**Worker and protocol.** The host (`ciw.lab.julia_worker`, standard library
only) starts `julia --project=<packaged environment> --startup-file=no
--history-file=no --threads=1 --color=no oscillator_worker.jl` with
`JULIA_DEPOT_PATH` set to the bound depot alone, `JULIA_LOAD_PATH=@` and
`@stdlib`, `JULIA_PKG_OFFLINE=true` and no other `JULIA_*` variable. The worker
refuses to start (exit code 70) when a package is not installed and
precompiled in the depot, so it never resolves, downloads or compiles packages
while serving; it writes protocol frames to its original stdout only and every
diagnostic to stderr. Frames in
both directions are little-endian: magic `CIWJ`, version 1, kind (handshake,
request, completed, refused, halted, shutdown), reserved zero, request
identifier (u64), payload length (u32, at most 65,536 bytes for requests and
262,144 for responses) and payload. A request's payload is SCR's three
specification fields, each a u64 length and bytes: the operation descriptor
(which must equal the one allowlisted operation byte for byte), the
configuration and the input.

| Encoding | Layout (little-endian, after a u16-length schema tag) |
| --- | --- |
| `ciw.julia.tsit5-configuration.v1` | abstol, reltol, initial dt, dtmax (f64); maxiters (u64); the PI-controller profile qmin 0.2, qmax 10, qmax_first_step 1e4, gamma 0.9, qsteady 1 and 1, beta1 0.14, beta2 0.08, qoldinit 1e-4, failfactor 2 (f64; Tsit5's OrdinaryDiffEqCore defaults made explicit; any other profile is refused) |
| `ciw.julia.oscillator-input.v1` | omega_0, gamma, mass, q0, v0, duration (f64); N (u32); N requested times (f64), starting at the initial time 0, strictly increasing, inside [0, duration) |
| `ciw.julia.oscillator-output.v1` | solver return code (u16 length and ASCII); accepted steps, rejected steps, function evaluations (u64); N (u32); then N values each of t, q, v and energy (f64) |

The worker solves `ODEProblem{true, AutoSpecialize}` of `[q, v]` over [0,
t_last] with `Tsit5()`, the configured tolerances, steps and controller,
`adaptive = true`, `saveat` the requested times (values from Tsit5's free
interpolant), `save_everystep = false`, `save_start = save_end = true`,
`dense = false`, and builds fresh problem and solver state on every request.
It answers `completed` only on `ReturnCode.Success` with every requested time
saved exactly and every value finite; otherwise `halted` with the return code
and step counts and no output. Invalid input (nonfinite numbers, bounds of
JULIA_SP1.md: 0 < omega_0 <= 20, 0 <= gamma <= omega_0/2, 0 < mass <= 100,
|q0| <= 10, |v0| <= 100, 0 < duration <= 12, 2 to 4096 samples; an invalid
grid, schema or length; an unknown program) is `refused` without a run, and the
session continues. A frame it cannot trust (bad header, oversized, truncated)
ends the worker.

The handshake (`ciw.julia.worker-handshake.v1`, 28 fields) reports the
protocol, the operation and its descriptor digest, Julia's version and commit,
machine and word size, the executable's SHA-256, the system image path, the
worker source, project and manifest digests, the active project and depot, the
load path, every loaded package with its version and source directory, thread
counts, rounding mode, subnormal flushing, optimization level, bounds checking,
fast math, CPU target and name, and the controller profile. The host accepts
the session only when every declared field matches the expected identity
(Julia 1.10.12 and its release commit, the pinned executable digest for the
declared machine, the bound worker source, project and manifest digests, one
thread and no interactive threads, the default rounding and flags), the machine
has a pinned archive and the system image is its default, the project and depot
are the bound ones, every declared loaded package's version and directory match
the committed manifest (the directory name must be Julia's slug of the
package's UUID and git tree hash, CRC-32C recomputed in Python), and the host's
own reading of the bound files matches too: the bound executable's digest, the
runtime tree it lies in and the system image against the pin, and the git tree
of every package directory the committed manifest names in the bound depot
(Pkg's `GitTools.tree_hash`, recomputed in Python) against the manifest's
`git-tree-sha1`. File digests are computed once per file state (path, size,
inode, modification and change times) in a process, so the 630 MB runtime tree
is read once per run. The depot's precompiled package images are trusted from
provisioning: neither the host nor Julia 1.10 (which compares only the
sources' modification times) checks their content. CPU name, optimization
level and CPU target are recorded only. One request
is in flight at a time, with a session identity and an occurrence number
(the frame's request identifier) that increases by one per request. A timeout,
end of stream, malformed, oversized or unknown response, a mismatched request
identifier or a worker exit ends the session: the process is killed and
reaped (a worker that closed its output is given 5 s to exit, and its exit
code is recorded in the failure: 70 for a depot without the provisioned
environment, 65 to 67 for a request frame it could not read), the occurrence
fails and nothing is retried or substituted; a request the host refuses to
encode or frame uses no occurrence.

**Behind SCR.** With a clean SCR checkout at CIW's pin bound as `scr` (the
check T097 makes), T145 runs its plan in an isolated interpreter that imports
that checkout and dispatches every fixture through
`execution.dispatcher.SpecificationDispatcher` with the worker as its
`runner`: the specification is `ExecutionSpecification(descriptor,
configuration, input)`, a completed exchange becomes an `ExecutionResult`
whose output, output identity and computation identity come from SCR's own
commitment functions over the exact response payload (exit code 0, the
occurrence number as `engine_occurrence`), a halted run an `ExecutionResult`
without output (the dispatcher then raises and admits nothing), and a refused
request `ExecutionRefused`. The dispatcher's `DispatchedMeasurement` records
are retained, and T145 recomputes SCR's program, input, specification, output
and computation commitments from the retained frames with its own restatement
of SCR's canonical encoding (checked against SCR's pinned vectors in the
tests). SCR's API at the pin hosts all of this unchanged; two limits are
recorded: the dispatcher labels every runner's result
`simulation:deterministic_native_execution`, which does not name the Julia
worker, and admission of the measurements through `run_experiment_step` is not
exercised. Without an accepted SCR checkout the same plan runs straight to the
worker, the SCR finding stays `not_established` with the reason, and the two
claims that name SCR are made without it: "A halted solve returns no output"
and "Retained frames decode without Julia" (with SCR, the offline claim also
requires at least one recorded SCR commitment to recompute).

**Acceptance set.** Thresholds were declared before any result was seen:
componentwise `|x - x_ref| <= abs + rel |x_ref|` with q: 1e-6 m and 1e-6, v:
1e-5 m/s and 1e-6, E: 1e-5 J and 1e-6, at abstol = reltol = 1e-10, dt = 1e-3,
dtmax = 12, maxiters = 1e6; undamped phase error at most 1e-6 rad and energy
drift at most 1e-6. Four worker sessions and six protocol-mock sessions run in
this order (values from the retained host):

| Fixture | Result |
| --- | --- |
| A: CIW default (768 samples at 64 Hz over [0, 12); `make_demo_run`'s channels) | max errors q 4.4e-11 m, v 2.3e-10 m/s, E 4.4e-10 J; largest threshold ratio 3.8e-5; `independently_verified` against `ciw.adapters.oscillator.closed_form` |
| B: mixed state (omega_0 3, gamma 0.4, m 2.5, q0 -0.7, v0 2.5; 600 samples at 50 Hz) | max errors 1.2e-11, 4.2e-11, 1.8e-10; `independently_verified` |
| Undamped limit (gamma = 0) | phase error 3.7e-11 rad (`independently_verified`); energy drift 2.8e-10 (`numerically_verified`) |
| Tolerance ladder, abstol = reltol 1e-6, 1e-8, 1e-10, 1e-12 | max q error 6.6e-7, 4.7e-9, 4.4e-11, 4.3e-13; accepted steps 254, 632, 1584, 3973; function evaluations 1525 to 23,839; no rejected step |
| Global error against the requested tolerance | v's error reaches 2.6, 1.4, 1.2 and 1.2 times abstol + reltol \|v\| on the four rungs (a retained counterexample to the tolerance bounding the global error). The tolerance is applied to each step's local error estimate, in an RMS norm over [q, v] scaled by the step's endpoints; it does not control the values interpolated at the saved times or the error accumulated over the steps. The RMS norm alone lets one component's scaled estimate reach sqrt(2) = 1.41, which the last three rungs stay within; these data do not separate the causes |
| A, B, A, a halted request, six worker refusals, A on one worker; A on a restarted worker | the four A occurrences agree exactly (declared agreement 1e-12), with distinct (session, occurrence) pairs over two sessions; same-host byte equality of the four outputs is a separate claim, and agreement across platforms is not established (Windows is not run) |
| Refusals | the host refuses a boolean, a NaN, gamma > omega_0/2, a duration over 12 s, a repeated time, a grid not starting at 0, a sample at the excluded endpoint, 4097 samples, reltol 1e-15, a boolean maxiters and a request frame over 65,536 bytes before dispatch; the worker refuses a NaN, an unsorted grid, gamma out of bounds, another program, another controller profile and a truncated input, then serves A again; maxiters = 10 halts with `MaxIters` and no output |
| Channel failures (real worker) | an oversized frame header (`frame_too_large`, the worker exits), a request after that (`session_ended`), a half-sent frame (`response_timeout` after 2 s), a crash (the worker killed while it holds half a request frame, so it can never answer: `worker_exited`, exit code -9), a worker started with 2 threads where 1 is declared (`environment_mismatch` at the handshake) |
| Channel failures (protocol mock) | truncated response (`unexpected_eof`), bad magic and unknown kind (`malformed_response`), oversized (`response_too_large`), another request's identifier (`request_id_mismatch`), exit after the handshake (`worker_exited`) |
| Offline restore | the retained frames, read back from `julia-frames.json` without Julia, decode and reproduce SCR's recorded commitments (without SCR: decode) |

The mock is a Python process that replays worker-1's recorded handshake, and
the host accepts it: a handshake identifies the environment the process
declares, not an attested one (retained as a counterexample to "a worker whose
handshake matches runs the pinned environment"). The independent checks'
producer is SciML's solver, `OrdinaryDiffEq.jl Tsit5 (OrdinaryDiffEqTsit5) on
Julia` with the exact package versions, Julia version and commit and the
manifest digest; its origin family `ordinarydiffeq` was added to the
recognised independent families for this task (the family is the solver, not
the Julia language, so CIW-authored Julia code does not count as independent of
CIW). The worker's own Julia code only states the right-hand side and encodes
bytes. A fixture that does not complete (halted, refused, failed or with an
output that does not decode) makes its findings refuted with the observed
outcome, and T145 stays `partial` and reports it; no value is read from a
result it did not get.

**Portability.** OpenBLAS kernels do not enter the Julia computation (the
three-kernel protocol regenerates T145 identically). Julia's code generation
does: running the fixtures with native code (FMA contraction of the solver's
`muladd`) and with `--cpu-target=generic` (no FMA) on a depot precompiled for
each moved the outputs by at most 5.4e-12, the reported errors and ratios by
at most 0.8 % relative (1.7e-14 absolute, at reltol 1e-12 where the errors
approach rounding) and no step count. Every solver-derived value therefore
carries the regression tolerance `abs 1e-13, rel 0.05`; counts, refusal codes
and identities are exact. Output digests and SCR's output and
computation identities follow the CPU and are retained only in artifacts.

Retained artifacts: `julia-frames.json` (every request and response frame,
payloads stored once per distinct byte string), `julia-exchanges.json`
(outcomes, handshakes with host paths replaced by their roles, SCR's
`DispatchedMeasurement` records and the recomputed identities),
`julia-acceptance.json` (per-fixture errors and step counts),
`julia-timings.json` (worker start-up and request wall-clock times, session
identities; never compared) and `julia-pin-procedure.json`.

## Canonical serialization

`ciw.canonical-json.v1` (T146):

- keys: strings only, unique, sorted by Unicode code point (equal to UTF-8
  byte order);
- no whitespace; separators `,` and `:`; output UTF-8 with non-ASCII written
  raw;
- strings: escape `"` and `\`; `\b \f \n \r \t`; other U+0000-U+001F as
  `\u00xx` (lowercase); everything else raw, including U+007F and U+2028; no
  Unicode normalization; lone surrogates refused;
- integers: decimal, `|n| <= 2^53 - 1`, otherwise refused;
- floats: finite binary64 only. Digits: the fewest significant digits k for
  which some k-digit decimal converts back to the same binary64 value; among
  those, the one nearest the exact binary value, and on an exact tie the one
  whose last digit is even (the CPython `repr` and ECMA-262 rule). Layout:
  fixed notation when the decimal point position p satisfies -4 < p <= 16
  (`1.0`, `-0.0`, `1000000000000000.0`, `0.0001`), otherwise
  `d[.ddd]e(+|-)XX` (`1e+16`, `1e-05`, `5e-324`); integers and floats stay
  distinct (`1` and `1.0`);
- refusal codes: `nonfinite_number`, `unsafe_integer`, `non_string_key`,
  `invalid_unicode`, `nesting_too_deep` (more than 64 enclosing containers),
  `unsupported_type`.

The reference encoder in `implementation_targets_serial.py` uses neither
`json` nor `repr`: strings are escaped from an explicit table and float digits
come from an exact integer search. Its output is checked against five examples
whose bytes were written by hand from the text above. Comparing CIW's
`json`-based encoders with it is therefore not true by construction (it still
is CIW code, so the comparisons are `cross_implementation`).

Test vectors (19 accepted, 9 refused) are retained in
`artifacts/T146/test-vectors.json` with their bytes and sha256; the set digest
over `[name, sha256]` pairs is `63f7261753b3bf93...`. Vectors with invisible or
normalization-sensitive characters are written with escapes in the source, so
editors cannot silently change them. Examples:

| Vector | Canonical bytes | sha256 (UTF-8 form) |
| --- | --- | --- |
| empty-object | `{}` | `44136fa355b3678a...` |
| signed-zero | `[0.0,-0.0,{"z":-0.0}]` | `e1a5ea3a03a3af50...` |
| integral-floats | `[1.0,-1.0,100.0,9007199254740992.0,1000000000000000.0,1e+16,1e+21,1e+22]` | `1a822860d54a8312...` |
| exponent-switch | `[0.0001,1e-05,0.00012345,1234567890123456.0,1.2345678901234568e+16]` | `b5d7e3579c6364f2...` |
| decimal-ties | `[1000000000000000.2,123456789012345.62,252611590017749.62,-29474221479446.812]` | `ab1c698de022e5bd...` |
| unicode-bmp | U+00E9 U+65E5 U+672C U+8A9E U+2028 U+00A0 written raw | `e28c9c5bcbf4f143...` (ASCII variant `4e2c3d77419efa08...`) |

Findings:

- `ciw.telemetry.canonical` reproduces the specification bytes on every
  accepted vector and on 2164 single floats (every vector float, 2000 random
  bit patterns and 96 constructed decimal ties).
- `ciw.core.identities.canonical_json` reproduces only the ASCII-escaped
  variant (`ensure_ascii=True`); it differs from the specification on 5 of 19
  vectors. The two CIW encoders, compared with each other, differ on the same
  5 vectors (all with characters outside U+0020-U+007E), so CIW does not yet
  have one encoding. Changing `ciw.core.identities` would change every
  retained identity and needs a versioned migration.
- `ciw.core.identities.canonical_json({1: "x"})` equals
  `canonical_json({"1": "x"})`: distinct values share a content identity. The
  specification and `ciw.telemetry.canonical` refuse non-string keys.
- Both Python encoders accept integers beyond 2^53; the identities encoder
  also accepts lone surrogates and depth 65.
- The first Rust probe was not byte-identical: Rust's shortest formatting
  (`format!("{:e}", x)`) breaks exact decimal ties upward, while the
  specification takes the even digit. `1e15 + 0.25` is `1000000000000000.2` in
  the specification and `1.0000000000000003e15` in Rust; Rust's own `{:e}`
  differs on 42 of the 96 constructed ties. The probe now takes the digit
  count from `{:e}` and the digits from correctly rounded formatting at that
  precision (`{:.*e}`), and gives byte-identical UTF-8 and ASCII forms on
  every vector and on all 2164 floats, with the same refusal codes. Same
  origin, so `numerically_verified`; the tie behaviour of Rust's `{:e}` is
  retained as a counterexample.
- The exact digit rule agrees with CPython `repr` and with NumPy's Dragon4 on
  all 2164 values (NumPy agreement is `independently_verified`).
- This is not RFC 8785 (JCS): JCS writes `1` for `1.0` and `10000000000000000`
  for `1e+16` (12 distinct vector float texts differ), and sorts keys by UTF-16 code units
  (U+1F600 before U+FF21).

## CPU and GPU comparison

T147 provides `compare_outputs(reference, candidate, policy)` with a bitwise
policy, an analytic-bound policy and an absolute/relative policy. For sums
whose terms pass through k roundings, the bound is
`gamma_k * sum|a_j x_j|` with `gamma_k = k u / (1 - k u)`; two outputs are
compared against the sum of their bounds. On 128 batched dot products of
length 1024 (sequential, pairwise and 32-lane blocked orders; float64 and
float32), the bitwise policy flags 123 of 128 float64 rows, the bound policy
accepts float64 reordering (at most 0.0017 of the bound), and float32 violates
the float64 policy on every row but stays at 0.0087 of its own bound.

A worst-case bound policy can miss faults up to about its bound. Each of the
131,072 single partial products is dropped in turn, and every faulty candidate
is judged by `compare_outputs` itself (one call on the broadcast matrix), so a
defect of the harness changes the result; the tests replace the harness with
one whose tolerance is 3 times or a third of the policy and whose bitwise mode
flags nothing, and the fault and bitwise findings are refuted. What is checked:

- operational guarantee: the policy tolerance is the sum of the two outputs'
  worst-case bounds, so the fault-free candidate satisfies |c - r| <= tol
  (the harness reports 0 violations) and a dropped term with |p| > 2 tol gives
  |c - p - r| > tol without knowing c - r. Under both policies none of those
  faults escapes (0 of 131,072 with the float64 policy, 0 of 129,813 with the
  float32 policy);
- threshold: with m the harness's largest fault-free |c - r| / tol (0.0087
  for float32), no fault below (1 - m) tol is detected and none above
  (1 + m) tol escapes;
- miss rate: with a_ij uniform on [-1, 1), P(|a_ij x_j| <= tol_i) =
  min(1, tol_i / |x_j|). Summed over the faults this predicts 665.3 +/- 22.5
  misses under the float32 policy; 708 escape (z = 1.9, within the 4-sigma
  check; over 40 other seeds z had mean 0.08 and standard deviation 1.1). The
  largest missed term has magnitude 5.89e-4 against a row tolerance of
  5.94e-4, and one fault 1.00016 times its row bound escaped (row 14, column
  538) because the candidate's own deviation points the other way, so "larger
  than the bound" alone does not guarantee detection. Under the float64 policy
  no fault escapes.

The bitwise finding reports the bitwise policy's own violations (123 of 128
rows), and a copy of the reference is not flagged.

The batched dot products have no GPU path and need none: the CPU/GPU
comparison uses the common Gaussian VI workload of the energy section, whose
PTX kernel (`ciw.energy_cuda`, entry `gaussian_vi`) exists. Its reductions are
fixed two-term sums evaluated unfused in one declared order, so T148's policy
prescribes a bitwise comparison (`FIXED_ORDER_POLICY`: `REDUCTION_POLICY`'s
`fixed_layout_arrays` rule, same order on both sides). On the CPU the harness
accepts a scalar Python-float evaluation of the declared order and flags both
exact fused multiply-add contractions of it (the build a contracting compiler
would produce), which shows the policy detects the difference a GPU compiler
could introduce. Where `hardware:nvidia-gpu` answers, T147 runs the kernel on
all 4096 replicas and judges its 24,576 outputs under the same policy; tests
with simulated devices that contract multiply-adds or flip the sign of one
output column show the finding is then refuted, not hidden. Outputs that
`CudaGaussianWorker.solve()` rejects after the kernel ran (nonfinite, out of
bound, asymmetric or not positive definite covariance) are recorded as a
failed check, so the finding is refuted rather than filed as an expected gap;
only a kernel that produced no outputs leaves it expected-unestablished. Here
no GPU answered, so the finding is `not_established` with that reason and T147
is `partial`. T147's runtime identity digests the modules that define the
workload (`ciw.energy_cuda`, `ciw.energy_bench`, `ciw.free_energy_math`)
beside its own sources and records the workload declaration
(`common_workload`).

## Reduction policies

Bounds for n terms with S = sum|x| and exact sum s, u = 2^-53:

| Use | Policy | Guarantee |
| --- | --- | --- |
| Identity-bearing sums | exact accumulation (integers at scale 2^1074, one final rounding) | correctly rounded, hence bitwise independent of order; equals `math.fsum` |
| Fixed-layout arrays | pairwise tree split at n // 2 over the stored order | bitwise reproducible only for the same order and length; error <= gamma_{ceil(log2 n)} S |
| Streaming accumulators | Neumaier compensation | error <= u\|s\| + gamma_{n-1}^2 S = u\|s\| + O(n^2 u^2) S (its compensation terms are those of Sum2; Ogita, Rump and Oishi 2005, Prop. 4.5); not order-invariant |
| Not for identities | unordered sequential sums, `numpy.sum`/BLAS, Kahan without the Neumaier branch | order- or implementation-dependent; Kahan returns 0 for [1, 1e100, 1, -1e100] |

An earlier version of this policy stated the Neumaier bound as
2u|s| + 4n u^2 S. That is not a bound: the second-order term grows like n^2.
On `[1] + 1000 x [0.7u(1 + 2^-20)] + [-1]` (n = 1002) the Neumaier error is 12.3
times that expression and 0.058 of the rigorous bound above; the dataset is
now part of the permutation study and the failure is retained as a
counterexample. Kahan is checked against 2u S (Higham 2002, eq. 4.8) plus a
declared second-order allowance 4n u^2 S; that constant is a policy choice
without proof.

T147 and T121 apply this policy to the common workload: a fixed-order
reduction evaluated in the same order on both sides is compared bitwise
(`FIXED_ORDER_POLICY`). A device-wide reduction of the workload's per-replica
outputs following the pairwise rule, with one atomicAdd variant for contrast,
is the deferred research question T148 names; parallel (multi-thread and
device) reductions are not exercised here.

Across 25 orders of five datasets, exact accumulation gave one result per
dataset and matched `math.fsum` everywhere; fixed-tree pairwise gave 4
distinct results on uniform data, while the same tree walked a second way
(NumPy level-wise additions of even and odd entries, equal to the recursive
split at n // 2 for n = 1024) gave the same bits as the recursive Python sum on
all 25 orders, so the result depends on the order and not on the
implementation. Other platforms are covered only by the regression gate, which
compares the retained first-order sum bits exactly; every observed error stayed within its
bound (worst ratios: exact 0.41, Neumaier 0.41, sequential 0.35, Kahan 0.35
of its allowance, pairwise 0.22).

## FPGA telemetry-only interface

Header and payload big-endian, CRC-32 little-endian; one frame type, no
host-to-device field:

| Offset | Bytes | Field |
| --- | --- | --- |
| 0 | 4 | magic `CIWT` |
| 4 | 1 | version = 1 |
| 5 | 1 | frame type = 0x01 telemetry (0x80-0x83 command, register write, setpoint, bitstream load are refused) |
| 6 | 2 | flags: bit 0 overflow, bit 1 clock unlocked; bit 15 (write request) refused; others must be zero |
| 8 | 4 | sequence, +1 per frame, wraps mod 2^32 |
| 12 | 8 | device timestamp, ns |
| 20 | 4 | clock id |
| 24 | 2 | channel count (<= 256) |
| 26 | 2 | payload bytes = 4 x channel count |
| 28 | 4c | int32 raw counts, calibration not applied |
| 28 + 4c | 4 | CRC-32/IEEE (reflected) over all preceding bytes, least significant byte first |

The decoder checks the CRC first, then refuses a bad magic, an unsupported
version, command and write paths (`command_path_refused`), unknown types,
reserved flags, truncation and length mismatches; the encoder refuses a write
request and the interface validator a host-to-device or `write_enable` field
(13 forged frames, requests and specifications, each refused with its code).
The host receiver is checked structurally rather than by method names: after
receiving good and corrupted frames its public interface is exactly `accept`
plus plain data (numbers, strings, bytes and containers of them), it has no
base class other than `object`, and a forged subclass with an `emit` method
and a transport handle is flagged. The CIW CRC-32 matches `zlib` and the check
value 0xCBF43926.

Error detection on a 48-byte frame, with bit p of the frame taken as bit p % 8
of byte p // 8 (least significant first, the reflected CRC's polynomial
order):

- every single-bit error is refused (384/384), and 5000/5000 random 2-11 bit
  corruptions;
- bursts are analysed exactly rather than sampled: an error pattern escapes
  iff its single-bit syndromes sum to zero, so if every 32-bit window has
  syndromes of full GF(2) rank, no burst of length <= 32 escapes. With the
  little-endian trailer, 0 of 353 windows are rank-deficient.

The byte order of the trailer matters. With the CRC appended big-endian (as an
earlier version of this format did), 16 windows are rank-deficient, the first
starting at bit 321, and a 32-bit burst across the payload/CRC boundary passes
the CRC check while changing a channel value. If the link numbered bits
MSB-first within bytes, 132 windows would be deficient, with an undetected
burst entirely inside the payload. Both are retained as counterexamples; the
guarantee holds only for an LSB-first link (as UART and Ethernet send), which
is assumed, not demonstrated.

**Bitstream identity** (`ciw.fpga-bitstream-identity.v1`, T150): bitstream
sha256 and size, toolchain name, exact version and `installation_sha256` (the
canonical digest of the `{path: sha256}` manifest of the toolchain files
supplied), part, per-file constraint digests and their
canonical digest, source-tree manifest digest, synthesis options, a
`synthetic` flag, and `record_sha256` over the canonical JSON of everything
else. Versions must be strings that fully match
`[0-9]+(\.[0-9]+){1,3}([-_.](rc|beta|alpha)[0-9]+)?\Z` (ASCII digits; `\Z`
because `$` would also accept a trailing newline), so `latest`, ranges,
`2024.1.x`, `1.0-latest`, `2023.2_nightly`, an unnumbered `2024.1-rc`,
`2024.1` followed by a newline and the number 2024.1 are refused while
`2024.1-rc2` is accepted (positive control). Every digest is exactly 64
lowercase hex digits (a trailing newline is refused), the size a nonnegative
integer and the synthetic flag a boolean; a malformed source tree is refused,
not a crash. The record is checked against every artifact it binds:
bitstream bytes, constraint files, source files, the toolchain's installation
files and the version the installed toolchain reports, so an edited or added
toolchain file and a toolchain reporting another version are refused
(`toolchain_installation_mismatch`, `toolchain_version_mismatch`).
Twenty-six mutations are refused with their codes; deployment of a synthetic
record is refused, and deployment of any record needs authority the lab does
not hold. Which files of a vendor toolchain the installation manifest must
cover is not specified here, and a change outside the supplied files is not
detected. Reproducibility of the record across runs is left to the regression
gate, which compares `record_sha256` exactly. T150 is `completed` although its
bitstream is a placeholder because it records identity and toolchain, a build
artifact rather than a measurement of physical origin: the synthetic
bitstream exercises every field and refusal of the record. Recording a real
design's bitstream, and whether a vendor toolchain builds the same bytes
twice, need a vendor toolchain and are its deferred research question.

**Compatibility and rollback** (`ciw.fpga-compatibility.v1`,
`ciw.fpga-rollback.v1`, T151): `compatible(b, h, r)` iff the host decoder
reads the bitstream's frame format, the board is supported and the host meets
the minimum version (15 of 36 combinations in the declared matrix). A rollback
must cite registered identities with matching digests, go backwards, give a
reason and match the matrix digest; execution is always refused here.
Rolling back to the previous version is not a safe default: it is
incompatible in 4 states (for example host 1.0, board revC, 1.4.1 -> 1.2.0).

**Link simulation** (T152): Gilbert-Elliott loss, gamma latency, duplicates,
device timestamps quantized to 50 us, 20 ppm clock drift and a forced loss at
the 2^32 wrap, over 4000 frames:

- serial-number arithmetic detects all 120 losses between the first and last
  received frame, 522 reorderings and 13 duplicates;
- with the declared clock offset the receiver's age is latency plus a
  nonnegative excess (quantization and drift), so no stale frame is missed
  (0/353, analytic); its 39 false alarms match the 39.1 +/- 6.2 predicted from
  the closed-form latency distribution;
- an offset estimated from the minimum delay over the first 200 deliveries
  (bias 2.04 ms) misses 322 of 333 stale frames afterwards, against 326.6 +/-
  17.3 predicted, with no false alarms;
- differencing sequence numbers in arrival order reports 1258 losses against
  120: reordering, not missing modular arithmetic, causes the overcount. On the
  sequence-ordered stream the non-modular count is 119, missing exactly the
  loss at the wrap;
- a 200,000-frame run matches the model: loss rate 0.02360 against the
  stationary 0.02195 (z = 2.87 with the exact Gilbert-Elliott variance
  [p(1-p) + 2(l_b - l_g)^2 pi_g pi_b lambda/(1 - lambda)]/N; over 60 other
  seeds z had mean 0.08 and standard deviation 1.01, so this seed is in the
  upper tail), mean latency z = 0.67. The checks use 4 sigma; at 200,000
  frames the loss check detects a change of `loss_bad` from 0.5 to 0.1
  almost surely, at 4000 frames it would not.

## Authorization boundary

- Actuator writes are denied by default (T153). Enabling needs an
  authorization record issued outside the workbench, unexpired, in scope,
  signed and verified against a trust anchor. The lab build has no trust
  anchor and no actuator transport and refuses to issue authorization
  records, so every route ends in a refusal with a specific code
  (`authorization_missing`, `authorization_wrong_type`,
  `self_issued_authority`, `malformed_timestamp`, `authorization_expired`,
  `out_of_scope`, `unsigned_authorization`, `no_trust_anchor`). The validity
  window is compared as instants: `valid_from`, `valid_until` and the
  evaluation instant are parsed as ISO-8601 timestamps with a UTC offset, so an
  expiry written `2026-09-23T01:00:00+02:00` is expired at
  `2026-09-23T00:00:00Z`, fractional seconds count, and a timestamp without an
  offset or a non-timestamp such as `tomorrow` is refused as
  `malformed_timestamp` (twelve routes in all). The default gate refuses a
  write on each declared channel before it reads the channel or value.
- A source scan (the T144 scanner) finds no module that imports a device,
  serial, fieldbus, industrial-protocol or instrument library (pyserial,
  python-can, pymodbus, pysoem, asyncua, pyvisa, ...) or names a device node
  (`/dev/...`, `\\.\...`, `COMn`), and network libraries or connection calls
  only in the declared workbench servers (`ciw.server`, `ciw.cli`,
  `ciw.lab.mcp_server`, which talk to the workbench's own clients). Five forged
  write paths (a serial port, a COM device opened as a file, a Modbus client, a
  raw socket and an asyncio stream to a PLC) are each flagged. No module routes
  writes through `ActuatorWritePolicy` because there is no write path to route;
  a future transport would be flagged, but routing it through the policy is
  not enforced. The scan is static and does not see dynamic imports,
  third-party internals or spawned providers.
- A frozen dataclass can still be mutated with `object.__setattr__`; the gate
  re-verifies the authorization on every write, so the mutation does not open
  it. In-process checks are defence in depth only; real enforcement belongs in
  hardware interlocks and a separate controller.
- Control outputs are `ControlProposal` objects whose status is fixed at
  `proposal` (T154). Converting one into a command is refused without separate
  authorization, and refused with any authorization the lab can see. A status
  forced to `command` in memory is caught the same way: `record()` and
  `to_command` re-check it and refuse with `proposal_status_tampered` (retained
  as a counterexample to "a frozen status field keeps every output a
  proposal"). The geometric proposal (heading change h = -j_lat(L) d /
  j_head(L) from the Jacobi transfer) matches -cot(L) d on the unit sphere
  (relative error 4.7e-12), cancels a lateral offset to second order on the
  torus (residual order 2.02, uncorrected 1.00) and is refused at a conjugate
  point.
- Scope of T154: only the control outputs built in this section are
  proposals. An inventory (`CONTROL_OUTPUTS`) lists the workbench's
  control-like outputs and is checked against a keyword scan of the package
  (stop request, abort, setpoint, command conversion, gain change); every
  module naming one is inventoried, and a forged module returning a setpoint
  is found. Of five outputs, only the heading correction is a
  `ControlProposal`: the T149 command frames, the T151 rollback execution and
  the T153 actuator writes are refused rather than proposed, and the T114
  servo-axis abort (a stop request to the bench's independent safety function,
  declared in `ciw.lab.lyapunov_research`) is neither. "Every control-like
  output is a proposal" is therefore recorded as not established.
