# Energy and GPU experiments (T115–T125)

Section 8 of the computational-experimentalist queue. Implementation:
`src/ciw/lab/energy_gpu.py` (tasks), `energy_gpu_workload.py` (the common
Gaussian VI workload: NumPy reference in float64 and float32, exact fused
multiply-add emulation, the embedded Rust port and the PTX kernel run),
`energy_gpu_kernels.py` (geodesic RK4 in any float dtype with an instrumented
operation count, the embedded sphere Rust kernel, reduction orders with
order-specific error bounds, typed quantities) and `energy_gpu_telemetry.py`
(the operator-run RAPL capture, fixture tampering, the operator-log
acquisition gate, timestamped nvidia-smi parsing, Session replay). Tests:
`tests/test_lab_energy_gpu.py`.

```
python -m ciw lab run T115 T116 T117 T118 T119 T120 T121 T122 T123 T124 T125 --output-dir <dir>
python -m ciw lab report T121 --retained <dir>
```

The whole section runs in about 5 s on one core (T117 compiles two small Rust
programs); the tests take about 16 s. T119, T123, T124 and T125 read
`examples/energy-accuracy` through `ciw.lab.runner.repository_path`, so an
installed package finds the fixtures when `CIW_LAB_REPOSITORY_ROOT` names a
checkout. Without them T119, T124 and T125 report `blocked` (with their static
answers and physical claims) and T123 reports `partial` with its energy-record
field audit recorded as not run.

## The common workload

Every question of this section that compares devices, languages or precisions
is asked of one computation that runs on the RTX 2080 host: the fixed-step
Gaussian variational iteration that the PTX kernel `gaussian_vi` of
`ciw.energy_cuda` executes and that `ciw energy record` captures (T116, T118,
T119). The problem and solver settings are `examples/energy-accuracy/problem.json`
(embedded in `energy_gpu_workload.SPEC`; a test checks the two are equal), the
iteration count is the one `ciw energy record` plans (K = 38, the first
iteration whose KL to the exact posterior meets the declared 1e-8 nats), and a
batch is 4096 identical replicas. Per replica, from the prepared inputs of
`ciw.energy_cuda._prepare`:

```
g_i  = ((P_i0 m_0) + (P_i1 m_1)) - b_i            m_i <- m_i - alpha g_i
Q_ij <- ((1 - beta) Q_ij) + (beta P_ij)            (K iterations)
det = (Q_00 Q_11) - (Q_01 Q_10);  C = [[Q_11, -Q_01], [-Q_10, Q_00]] / det
```

Every operation rounds to nearest with no fused multiply-add, which is what
the kernel's explicit `.rn` instructions declare. The kernel has 24 arithmetic
instructions per iteration and 7 in its output block; the NumPy reference,
instrumented on counting arrays, performs the same 24 and 7 per replica in
float64 and in float32 (T115 and T120 check this against the PTX text, and a
test shows the check fails for a kernel with a fused instruction). One batch
is therefore 4096 x (24 x 38 + 7) = 3,764,224 operations.

The workload is defined by `ciw.energy_cuda` (the PTX text and `_prepare`),
`ciw.energy_bench` (the planner that fixes K) and `ciw.free_energy_math` (the
information system and the KL). Every task whose results depend on it (T115,
T116 and T118 when a log is analyzed, T117, T119, T120, T121, T124 and T147)
digests those three modules beside its own sources in its runtime identity
(`energy_gpu_workload.SOURCES`) and records the declaration
`energy_gpu_workload.workload()` (kernel, problem and prepared-input digests,
K, replicas) under `common_workload`, so a changed kernel or planner changes
the reports' identity.

| Task | Uses the common workload for | Without the hardware |
| --- | --- | --- |
| T115 | CPU package energy per batch (RAPL bracket of the NumPy float64 reference) beside energy per geodesic trajectory | `not_established`: no rapl-log capture bound |
| T116, T118, T119 | GPU device energy, telemetry and energy per accepted result (the kernel itself, via `ciw energy record`) | `blocked` (T116, T118) or `partial` (T119) |
| T117 | NumPy reference against the Rust port (float64 and float32, bitwise) and against the PTX kernel on the GPU (bitwise, every replica); Rust and NumPy package energy per batch | GPU comparison `not_established`, naming the failed `hardware:nvidia-gpu` probe; energy `not_established` |
| T120 | float32 against float64 of the same iteration: KL per iteration, operation counts, package energy per batch in each precision | energy `not_established`; the float32 GPU arm has no kernel |
| T121 | the kernel's own reductions under fused multiply-add contraction and reassociation; the kernel's outputs on the GPU | GPU comparison `not_established` (probe); device-wide reduction has no kernel |
| T147 | the CPU/GPU harness: T148's fixed-order policy (bitwise) on the kernel's outputs | GPU comparison `not_established` (probe) |

The GPU parts that exist run wherever an NVIDIA GPU answers the probe: their
findings are `not_established` here with the reason that no GPU probe
succeeded, not that no implementation exists. Two GPU parts do not exist on
any host, and one deferred research question owns both
(`energy_gpu_workload.GPU_QUESTION`, named by T117, T120, T121, T147 and T148):
a float32 rendering of the `gaussian_vi` kernel, and a device-wide reduction
of its per-replica outputs (a fixed tree per T148's `REDUCTION_POLICY`, plus
one atomicAdd variant).

## What this environment could and could not measure

The development host has no RAPL powercap counters, no NVIDIA GPU or NVML,
no CUDA toolchain and no Julia. `rustc` is available. The lab runner acquires
no energy measurement on any host: CPU and GPU energy come only from captures
an operator makes outside the runner, which the tasks analyze read-only. Its
only counter access is the `hardware:rapl` availability probe, which T115,
T117 and T120 make only when a capture is supplied: it reads one `energy_uj`
value to confirm readability and discards it
(`test_rapl_probe_is_the_only_counter_read` exercises the real probe on a
simulated powercap tree). Consequently:

| Quantity | Status here | Where it is recorded |
| --- | --- | --- |
| CPU package energy per geodesic trajectory and per common-workload batch (gross and idle-subtracted) | not measured (no capture) | T115 physical findings, `not_established` |
| GPU energy per batch; NVML counter accuracy | not measured (no GPU) | T116 `blocked`; physical and sensor-performance findings, `not_established` |
| RTX 2080 power, temperature, clock, utilization, steady state, batch and kernel time | not measured | T118 `blocked`; eight physical findings, `not_established` |
| Physical energy per accepted result | not measured (no operator log) | T119 physical finding, `not_established` |
| CPU energy of the Rust port against the NumPy reference | not measured (no capture) | T117 physical finding, `not_established` |
| CPU energy of float32 against float64 | not measured (no capture) | T120 physical finding, `not_established` |
| PTX kernel against the NumPy reference on the GPU | not run (no GPU probe succeeded) | T117, T121 and T147 numerical findings, `not_established` |
| GPU energy of a float32 build; device-wide GPU reductions | not written (on any host) | T120 physical and T121 numerical findings, `not_established`; deferred question |
| Julia implementation of the common workload | not written (on any host) | T117 finding, `not_established`; deferred question |
| RTX 2080 kernel-only duration | not ingested (on any host) | T118 finding, `not_established`; deferred question |
| Raw telemetry and identity of a real device | not retained (no operator log) | T124 `partial`; physical finding `not_established` |

Every joule figure in this section comes from the synthetic fixtures in
`examples/energy-accuracy` (`origin: synthetic_fixture`) and is a synthetic
value. Wall-clock and CPU times are retained only as artifacts
(`timing.json`); they are never findings.

## What was computed

- **T115 work proxies.** Fixed-step RK4 spends exactly 4N = 1024
  right-hand-side evaluations per sphere geodesic (N = 256, counted per
  trajectory, checked per trajectory), with endpoint error 1.1e-9 against
  the exact great circle; adaptive Dormand–Prince (rtol 1e-9) uses 295–487
  evaluations per trajectory (declared cross-platform allowance 12
  evaluations: two flipped accept/reject decisions of a 6-evaluation FSAL
  step). A common-workload batch is 3,764,224 operations, the PTX kernel's
  instruction count times K and the replicas. Energy needs an operator
  capture (protocol below): the capture brackets the geodesic batches and the
  common-workload batches separately, and T115 reports package energy per
  trajectory and per batch.
- **T117 Python, Rust, GPU and Julia.** Two std-only Rust programs are
  embedded, compiled with `rustc -O -C codegen-units=1` into a temporary
  directory (the scratch path is remapped, so each binary digest is
  reproducible for a given rustc) and exchange JSON over stdin/stdout. The
  sphere RK4 kernel (`energy_gpu_kernels.RUST_SOURCE`) is kept because it
  exercises libm: its `rhs()` counts its own calls (4NT = 6144), its endpoints
  are bitwise identical to the Python closed-form kernel on this Linux host
  (same operation order, same glibc `sin`/`cos`; the finding tolerance is
  1e-12, cross-platform bitwise identity is not claimed), and it refuses a
  malformed and a nonfinite input. The Rust port of the common workload
  (`energy_gpu_workload.RUST_SOURCE`) reproduces the NumPy reference bitwise
  in float64 and in float32 (maximum ULP distance 0; the workload uses no
  library function, so bitwise identity is expected on any IEEE platform),
  counts its own iterations (K x 4096 = 155,648) and refuses three malformed
  inputs. On a host where `hardware:nvidia-gpu` answers, the PTX kernel runs
  through `ciw.energy_cuda` on the same prepared inputs (the worker's
  prepared-input digest is compared) and its 4096 x 6 outputs are compared
  bit for bit: the largest ULP distance, taken on an order-preserving map of
  the bits so that a sign flip (a dropped `neg.f64`, or +0.0 against -0.0) is
  a nonzero distance, and the count of values whose bits differ must both be
  0. Tests with simulated devices that contract multiply-adds or flip the
  sign of one output column show the finding is refuted, not hidden. Outputs
  that `CudaGaussianWorker.solve()` rejects after the kernel ran (nonfinite,
  out of bound, asymmetric or not positive definite covariance) are a failed
  check, so the finding is refuted; only a kernel that produced no outputs
  (driver, device, preparation or JIT failure, or a failed launch) leaves the
  claim expected-unestablished with that reason. All agreements are
  `cross_implementation` checks: declaring one ciw kernel an independent
  check of another is refused by `ciw.lab.evidence`, so the label is
  `numerically_verified`, never `independently_verified`. The rustc version
  and binary digests are in the report's runtime identity (and the GPU's
  name, UUID and driver when it ran). No Julia port exists, so T117 stays
  `partial` on every host; deferred research question: a Julia port behind
  the SCR worker T145 plans. With a bound rapl-log capture that holds the
  Rust bracket (built by the same rustc, so the binary digests match), T117
  reports the Rust port's and the NumPy reference's package energy per batch.
- **T119 energy per accepted result.** Definition:
  `E_acc = (counter(last measurement read) − counter(first measurement read)) / #accepted replica solves`,
  where a replica solve is accepted when its retained batch output has KL ≤
  the declared target. The denominator counts replica solves (identical
  thread work on one declared problem), not distinct results: every replica
  of a batch is a bitwise copy of one output. Baseline fixture: 0.05 J per
  accepted replica solve and 0.2 J per distinct accepted result (synthetic;
  the distinct count is checked against the distinct binary64 byte strings
  of the accepted outputs, and each fixture's SHA-256 is in the generator).
  The recomputation decodes the outputs directly and scores them with a
  textbook Gaussian KL against an information-form posterior written in the
  task (agreement with `energy_records.analyze` to 1e-12; same ciw origin,
  so a cross-implementation check). The metric is withheld for exactly the
  reset, missing-bracket and under-target fixtures, each for its declared
  defect. Counterexample: dividing gross energy by executed solves gives
  0.05 J/solve for the under-target fixture, which has zero accepted solves.
  Widening the boundary to the whole run multiplies the metric by 7. When an
  operator log is bound (`--capture energy-log=PATH`, or `CIW_LAB_ENERGY_LOG`),
  T119 recomputes energy per accepted replica solve of the common workload
  from its raw readings and outputs and reports it as `hardware_measured`
  through the T116 acquisition gate (plus a GPU probe that answers in T119);
  T119 is then `completed`.
- **T120 precision.** float64 RK4 converges at order 3.96 (N = 16..256).
  float32 reaches its minimum error 5.8e-7 at N = 64, the crossover of
  truncation and roundoff, and then sits on a roundoff plateau (median
  2.5e-6 for N ≥ 128; 1e7 times the float64 error at N = 2048). Both
  precisions execute the same counted arithmetic per step (80 flops,
  8 sin/cos calls), confirmed per precision by running one step of the same
  code on a counting view of a real float32 and a real float64 array, which
  also records every ufunc result's dtype (no result leaves the declared
  precision; a float64 step size in the float32 path is detected); the
  float32 and float64 transcendental implementations themselves differ. The
  step table gives the smallest grid N (≥ 16) meeting each target, so a 16 is
  censored at the grid minimum; every tabulated error clears its target by a
  factor of at least 1.4, so the table is compared exactly. Counterexample:
  float32 cannot reach 1e-7 at any N ≤ 2048, while float64 reaches it at
  N = 128 — lower precision is not cheaper at every accuracy target.
  `precision.svg` is a rounding-level figure: the float32 plateau is
  roundoff, and the float64 errors near 1e-13 follow the platform's sin/cos
  in their last bits (on Windows the N = 2048 error moved by 5.1e-16). Each
  plotted error records a bound of ten times its largest change over four
  seeded runs in which every sin and cos result moves by a random whole
  number of ulps in [−4, 4] (`precision.json` `figure_rounding`): 3e-15 to
  2e-14 for float64, below 1e-5 of each error in the order-4 range, and about
  the size of the float32 plateau errors themselves. The
  energy comparison uses the common workload instead, where float32 and
  float64 run the same iteration: both precisions meet the workload's
  declared 1e-8 nat KL target at the planned iteration K = 38 (KL 6.9e-9
  nats), and both execute the same 24 counted operations per replica
  iteration with no result leaving the declared precision. float32 then
  stalls at a KL floor of 2.4e-14 nats from iteration 66 on, set by float32
  roundoff (about (2^-24)^2 relative), while float64 keeps converging (6e-32
  after 256 iterations, where the binary64 reference's own rounding shows).
  Counterexample: float32 cannot reach 1e-14 nats within 256 iterations,
  while float64 reaches it at iteration 69. A bound rapl-log capture brackets
  the NumPy reference in float64 and in float32, and T120 reports package
  energy per batch in each precision (T120 is then `completed`); the GPU arm
  (a float32 build of the kernel) has no implementation and is the shared
  deferred research question.
- **T122 bounded free energy.** On a declared two-latent Gaussian problem the
  identity F + log Z = KL holds at every iterate to 3e-14 nats; with the
  declared normalization (condition number 7.3) KL decreases monotonically to
  1e-21 nats in 93 of at most 512 iterations. Counterexamples: a mean step 1.2×
  the stability bound (spectral radius 1.4) makes KL grow to 5e18; unit scales
  (condition number 1.3e3) leave KL at 3.99 nats after 512 iterations, while
  the raw-unit posterior itself is scale-invariant to 2e-15. Deferred research
  question: a non-Gaussian posterior (a mixture likelihood, say), where the
  identity still holds but KL has no closed form and must be checked against
  a quadrature or Monte Carlo value with its own uncertainty.
- **T123 nats versus joules.** A typed quantity algebra refuses to add,
  subtract, compare (including `==`) or convert information (nat, bit) and
  energy (J, mJ); power (W, mW) is energy per time and is refused against
  energy; untyped numbers are refused on either side. Within one dimension
  the typed values agree with hand-converted values to binary64 rounding
  (0.2 J + 100 mJ − 0.3 J = 5.6e-17 J), and 1 J == 1000 mJ. CIW energy
  records carry joules in `_j`/`_mj` fields, power in `_mw` fields and nats
  only in `kl_*_nats` and the Gaussian reference; variational records carry
  no joule or watt field; each workbench energy panel has one unit.
  Counterexample (analytic, dimensional analysis): an untyped `F + E`
  changes by 999 E when E is expressed in mJ instead of J (from 8.6 to 208.4
  here). A computed check of that identity could not fail, so the finding is
  `analytic` and T123's headline, the weakest established label, is
  `analytic`. Deferred research question: carry units in exchanged records;
  `ciw.canonical-json.v1` (T146) encodes numbers without units, and CIW
  records keep units only in field-name suffixes, which this audit relies on.
- **T124 raw telemetry.** Without an operator log this is an audit of the
  energy-log format, not a retention of real telemetry, and T124 is
  `partial` ("Not performed: retaining raw telemetry and device/runtime
  identity of a real device"), as T139 is without an acquisition. The task
  retains the four fixtures' exact bytes as
  artifacts, and they hold 47 raw readings with monotonic and UTC brackets
  and the 14 device/runtime identity fields; 11 of the 14 hold placeholders
  (repeated-digit digests, the sequential fixture UUID, "fixture"/"synthetic"
  names, and the compute capability given to that placeholder device), so
  the fields exist but identify nothing. Nine tampering classes are refused
  with their expected messages (the two field-removal classes share the
  validator's generic field-set message).
  Counterexamples: a log whose readings were doubled and then resealed
  validates (sealing is integrity, not authenticity), and relabelling a
  fixture as `physical_measurement` makes it eligible for physical comparison
  (the origin field is a declaration). With an operator log bound on the GPU
  host, T124 retains its exact bytes, checks that every reading is
  timestamped and that none of the 14 identity fields is a placeholder, and
  binds it through the T116 acquisition gate (declared physical measurement,
  eligible analysis, the common workload, sensor identity equal to this
  host's NVML identity, GPU probe answering in the task); the physical
  finding is then `hardware_measured` and T124 `completed`. That finding
  states what the gate establishes: the log *names* an NVML device present
  on the analyzing host (its UUID, name, driver, NVML version and library
  digest equal the host's). It does not state that the counter readings came
  from that device: a resealed synthetic log with edited identity strings
  passes the gate (the regression test does exactly that), which T124's
  unresolved assumptions say. A relabelled fixture passes the host-identity
  gate but not the placeholder rule.
- **T125 replay.** Through a `ciw.session.Session`, every fixture keeps one
  `numerical_result_id` across original, reproduction and replay, each with
  fresh execution, result and bundle identities; the workspace restores
  exactly; an edited bundle is refused with stale digests
  (`Energy analysis bundle identity, schema or size differs`) and with
  recomputed digests (`Retained energy analysis binding differs`). Deferred
  research question: replay a retained RTX 2080 log through a Session on a
  second host; whether numerical ids agree across platforms is open, because
  they hash binary64 analysis values and `ciw.core.identities` was not
  migrated to `ciw.canonical-json.v1` (T146).

## Reduction-order findings (T121)

Emulated on the CPU with n = 16384 values: sequential, adjacent-pair tree,
strided (shared-memory style) tree, two-pass blocked, block-sequential, Kahan,
and eight atomicAdd completion orders of 256-element block partials. Each
order is checked against its own a priori bound γ_d Σ|xᵢ|, where d is the
largest number of additions any input passes through (n − 1 = 16383 for the
sequential fold, 14 for the trees and the two-pass blocked order, 71 for the
atomic orders, 318 for block-sequential; Kahan (2u + nu²) Σ|xᵢ| to leading
order). The generic γ₍ₙ₋₁₎ bound alone would miss a dropped element.

- Positive data (u³·1000 + 1e-3): float32 sums spread by 4.0e-6 relative across
  orders; the sequential fold is worst in absolute terms (error 15.9 on 4.1e6),
  every other order errs by under 0.7. The largest error as a fraction of its
  own bound is 0.12 (Kahan). float64 spread is 6.8e-16.
- Cancellation data (±a pairs plus 64 × 2⁻⁸, exact sum +0.25), found as the
  first of seeds 1..399 whose sequential and tree sums disagree in sign (seed
  201): the float32 sequential (CPU) fold gives −0.024 while every GPU-style
  order is positive — **reduction order alone flips a sign decision**.
- These float32-valued inputs sum exactly in float64 in every order (all 14
  errors are 0), which says nothing about float64 order-robustness. With
  float64-native data built the same way but scaled by 2²⁹ (seed 1), the
  float64 sequential fold gives −0.261 while the trees give +0.258: float64
  sign decisions are not order-robust either.
- GPU-only nondeterminism: on the float32 cancellation data the eight atomic
  completion orders alone change a pass/fail test |S − 0.25| ≤ 0.01
  (atomic-5 gives 0.2324 and atomic-7 0.2378 and fail; the other six pass).
  The tolerance 0.01 was chosen inside the observed atomic spread, so this
  shows that such flips exist, not how often they occur. On the positive data
  the same orders give four distinct float32 sums.
- A decision taken only when |S − T| exceeds the a priori bound cannot
  contradict across orders (triangle inequality). It is also useless here:
  the smallest order bound (Kahan, about 1.0) exceeds every |S| (at most 0.26),
  so the guarded sign test decides in none of the 14 orders on either
  cancellation dataset.
- The maximum of NaN-free, zero-free data is its largest element in every
  order, by definition, so it is not re-checked. Signed zeros break
  order-invariance for a comparison-select maximum (`a if a >= b else b`,
  the rule numpy documents for `np.maximum`): it returns its first operand
  when −0 and +0 compare equal, so the result's sign depends on operand
  order. This is not an IEEE property — IEEE 754-2019 `maximum` orders −0
  below +0 and returns +0 either way — and numpy's own `np.max` is
  architecture-specific here (x86 `maxsd` returns its second operand, giving
  −0.0 for `[0.0, -0.0]`; aarch64 `fmax` returns +0 in both orders). The
  finding checks the deterministic comparison-select rule; numpy's result on
  the running build is retained in `reductions.json` with
  `platform.machine()` and the numpy version, never asserted.

**The kernel's own reductions.** The common workload's reductions are fixed
two-term dot products, the precision relaxation and a 2x2 determinant, which
the PTX kernel evaluates unfused (`.rn` instructions). A compiler building the
same source with contraction would fuse multiply-adds; T121 emulates two
contraction rules exactly (a rational fused multiply-add rounded once) and one
reassociation. After K = 38 iterations the contractions move the binary64
outputs by 3 and 6 ULP (reassociation by 0 here), so a bitwise CPU/GPU
comparison detects a contracting build (the counterexample to "a fixed-order
reduction gives the same result whether or not the compiler contracts it"),
while the KL of every variant is 6.85e-9 nats against the 1e-8 target: the
acceptance decision is unchanged, with a KL change below 1e-6 of the 3.1e-9
decision margin. Where `hardware:nvidia-gpu` answers, T121 runs the kernel and
checks that its outputs equal the unfused reference bit for bit, which is the
comparison T148's `REDUCTION_POLICY` prescribes for a fixed-order reduction
evaluated in the same order. GPU sums are computational outputs, so these
claims are numerical: the comparison is `not_established` here because no GPU
probe succeeded. What CUB, cuBLAS, atomicAdd or a warp-shuffle tree does to a
device-wide reduction of the kernel's per-replica outputs is not observable
until such a reduction kernel exists; that numerical claim is recorded as
`not_established` with the reason "implementation missing", and the kernel is
the shared deferred research question.

## Protocol for a RAPL host (T115, T117, T120)

The capture is an operator action; `ciw lab run` acquires no measurement
(its `hardware:rapl` probe reads one counter value to confirm readability and
discards it).

1. On a Linux host whose `/sys/class/powercap/intel-rapl:*/energy_uj` is
   readable: `python -m ciw.lab.energy_gpu_telemetry rapl-capture
   runs/rapl-<date>.json` (options `--repeats N`, 1 to 1000, default 3
   geodesic batches; `--batches M`, 1 to 1000, default 500 common-workload
   batches per bracket, the Rust port's own repeats limit; `--no-rust`). It
   computes the common workload's prepared inputs once, before any bracket,
   and runs the Rust port on one replica with the real `M` before
   bracketing, so a refusal surfaces before any counter is read. It then
   brackets, with package counter reads, four workloads in turn: the T115
   sphere geodesics (`geodesic`, six trajectories per batch, RK4, N = 256),
   the common workload's NumPy reference in float64
   (`gaussian-vi-numpy-float64`) and in float32 (`gaussian-vi-numpy-float32`),
   and the Rust port in float64 (`gaussian-vi-rust-float64`, one process for
   all M batches, process start and JSON exchange included; skipped with its
   reason when rustc cannot build it, and moved to `skipped` without losing
   the other brackets if it fails inside its bracket). It then brackets an
   idle interval as long as the longest workload bracket, and writes
   (schema `ciw.lab.rapl-capture.v3`) the raw counters, UTC and monotonic
   brackets, each bracket's unit count and unit name, each bracket's declared
   workload (the common workload's declaration names the kernel digest, the
   prepared-input digest, K, the replicas, the precision and the bracket's
   work boundary; the Rust bracket also the rustc version and binary digest)
   and the host identity (CPU model, platform, RAPL zones). The record holds
   no host path. An existing file is refused.

   Work boundaries: no Gaussian VI bracket contains the preparation of the
   inputs (the information system and the LAPACK solve of
   `ciw.energy_cuda._prepare`), as the GPU worker's boundary excludes it
   (`constructor_preparation_and_jit_excluded`). A NumPy batch rounds the 15
   inputs to its precision, broadcasts them to the 4096 replica columns and
   runs K iterations and the output block; the Rust bracket adds one process
   start and one JSON exchange for all its batches, and each Rust replica
   rounds its inputs itself. T117's `rust_over_numpy_idle_subtracted`
   compares these bracket contents, not the arithmetic alone; T120's
   `float32_over_float64_idle_subtracted` compares two brackets with the same
   boundary.
2. `ciw lab run T115 T117 T120 --capture rapl-log=runs/rapl-<date>.json
   --output-dir results/rapl-<date>` on the same host (`--capture rapl-log`
   sets `CIW_LAB_RAPL_LOG` for the run).
3. `ciw lab hardware retain results/rapl-<date> --retained lab --run-id
   rapl-<date> --host "<RAPL host>"`, review `git diff lab/hardware` and commit
   it (docs/LAB.md, Hardware evidence).

Each task retains the capture bytes as an artifact and reports gross package
energy (background-inclusive, idle not subtracted) and idle-subtracted energy
as `hardware_measured` only when every bracket it reads names the workload the
task declares (boundary included), its unit count equals the capture's own
repeats x 6 trajectories or batches and its unit name is the declared one, the
capture's host identity equals the analyzing host's, and a RAPL probe answers
in the task: T115 per geodesic trajectory and per
common-workload batch, T120 per batch in float64 and in float32 (with their
ratio), T117 per batch for the Rust port and the NumPy reference (with their
ratio; the Rust bracket's binary digest must equal the port T117 builds, so
the analysis needs the capture host's rustc). A bracket that fails leaves only
its own findings `not_established`. The acquisition record says the capture
is operator-made and unauthenticated; the calibration is `not_applied` (RAPL
is a model-based counter). CPU energy per common-workload batch (T115) and
GPU energy per batch (T116) are for the same computation (the RAPL gate and
the NVML gate both require the common workload's declaration), but they are
different counters with different scopes (package versus whole device) and
different batch boundaries (the GPU batch includes launch, synchronization,
copy and the worker's output check), and no finding combines them.

## Protocol for the RTX 2080 host (T116–T121, T124, T147)

The lab runner never acquires hardware data. On the GPU host the operator
captures, then one lab run analyzes the capture and runs the common workload's
PTX kernel for every task that compares it with the CPU
(`R=runs/rtx2080-<date>`; `ciw energy record` refuses an existing output
directory, so the capture goes into `$R/capture`):

1. `mkdir -p $R` and `ciw energy probe --gpu-index 0` — must return a reading
   with `status: ok`. NVML documents `nvmlDeviceGetTotalEnergyConsumption`
   for Volta-or-newer *fully supported* devices; GeForce support is not
   documented, so this probe must confirm the counter on the RTX 2080 (the
   go/no-go for every energy claim below).
2. Start the utilization sidecar in the background, in UTC:
   `TZ=UTC nvidia-smi --query-gpu=timestamp,uuid,name,utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.sm,clocks.mem,pstate --format=csv,nounits -lms 100 -f $R/smi.csv`
3. Capture the common workload:
   `ciw energy record --problem examples/energy-accuracy/problem.json --output-dir $R/capture --duration 10 --replicas 4096 --warmup-batches 2 --idle-duration 2 --gpu-index 0`
   and stop the sidecar.
4. `ciw energy replay $R/capture/log.json` — offline recomputation.
5. Kernel durations in a separate pass (profiling perturbs timing and energy):
   `nsys profile --trace=cuda -o $R/nsys python -m ciw energy record --problem examples/energy-accuracy/problem.json --output-dir $R/capture-nsys --duration 10 --gpu-index 0`,
   then `nsys stats --report cuda_gpu_kern_sum $R/nsys.nsys-rep`. Retain the
   output; the lab does not ingest it yet (deferred research question:
   ingest it with its raw bytes retained and bound through the acquisition
   gate, so T118 can report kernel-only duration).
6. `CIW_LAB_NVIDIA_SMI_UTC_OFFSET=+00:00 ciw lab run T116 T117 T118 T119 T120 T121 T124 T147 --capture energy-log=$R/capture/log.json --capture nvidia-smi-csv=$R/smi.csv --output-dir results/rtx2080-<date>`
   on the same host. If the host also has readable RAPL counters, capture
   them first (protocol above) and add `T115` and `--capture rapl-log=...`.
7. `ciw lab hardware retain results/rtx2080-<date> --retained lab --run-id rtx2080-<date> --host "<RTX 2080 host>"`,
   review `git diff lab/hardware` and commit it. The retained run is verified
   off the host for integrity only (docs/LAB.md, Hardware evidence);
   re-analysing it needs this host.

T116 then reports gross device energy per measured batch of the common
workload, T119 the energy per accepted replica solve of the same log and T124
the log's retention with its device and runtime identity; the NVML counter's
accuracy and resolution stay `not_established` (undeclared by NVML, no
external meter). T117, T121 and T147 run the PTX kernel on the common workload
in the same run and compare its outputs with the NumPy reference bit for bit
(T147 through `compare_outputs` under T148's fixed-order policy); T147 is then
`completed`, T117 stays `partial` (no Julia port) and T121 `partial` (no
device-wide reduction kernel). T118 reports NVML power, temperature and
graphics clock over
the measurement phase, host-bracketed batch durations (these include launch,
synchronization and copy), sidecar utilization, and two steady-state criteria
declared by the protocol: power coefficient of variation ≤ 0.10 and
temperature range ≤ 5 °C over the measurement phase (a violated criterion
leaves its finding `not_established`). nvidia-smi prints local wall time, so
only sidecar rows whose timestamps, converted with the declared UTC offset,
fall inside the log's measurement window (first to last measurement counter
read) enter the utilization statistic; without a declared offset, or with no
row in the window, utilization is withheld. Kernel-only duration stays
`not_established` until an nsys report is ingested, so T118 is at most
`partial`.

**Acquisition gate.** A physical finding from such a log is
`hardware_measured` only when the log declares `physical_measurement`, its
analysis is eligible, it names the common workload (its runtime workload's
PTX kernel digest, problem digest, prepared-input digest, K and replicas equal
`energy_gpu_workload.workload()` and its plan's KL target is the declared
1e-8 nats; `energy_gpu_telemetry.workload_reasons`), its raw bytes (and the sidecar's) are retained
byte-exactly as artifacts, and its sensor block (UUID, name, driver version,
NVML version, NVML library SHA-256) equals the NVML identity of that device on
the analyzing host; T118 also requires the name to contain "RTX 2080" and
every in-window sidecar row to name the same UUID. The acquisition record's
device string says "operator-captured log, unauthenticated". This binds a
record to hardware and software present on the host. It does **not**
authenticate the capture: a relabelled synthetic log paired with a matching
host identity passes (see
`test_operator_log_gate_trust_boundary_is_the_host_identity` and the T124
relabelling counterexample; T124 additionally refuses placeholder identity
values). Signed capture is the unresolved next step.
Without a GPU, or on a GPU host without a log, T116 and T118 are `blocked`
and record their physical claims as `not_established` findings with the same
claim texts the operator path uses, so the regression gate can match them
across hosts.

## What these results do not show

- No physical energy, power, temperature, utilization or kernel time was
  measured; no efficiency ranking of Python, Rust, float32, float64, CPU or
  GPU follows from this section until the protocols above are run and
  retained.
- Python/Rust/PTX agreement is agreement of ciw implementations of one
  algorithm, not independent verification; the PTX kernel was not run here.
- Emulated reduction orders are plausible GPU orders, not observed ones, and
  the counterexample witnesses were found by search: they show existence, not
  frequency.
- Variational free energy is a dimensionless information quantity of a
  normalized model; it is not a thermodynamic free energy and says nothing
  about the joules the computation dissipated.
- Replay determinism is not verification by another party, and a valid log
  digest is not evidence that a device produced the readings.
