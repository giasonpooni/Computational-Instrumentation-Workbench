# Energy and GPU experiments (T115–T125)

Section 8 of the computational-experimentalist queue. Implementation:
`src/ciw/lab/energy_gpu.py` (tasks), `energy_gpu_kernels.py` (geodesic RK4 in
any float dtype with an instrumented operation count, the embedded Rust
kernel, reduction orders with order-specific error bounds, typed quantities)
and `energy_gpu_telemetry.py` (RAPL helpers and the operator-run RAPL
capture, fixture tampering, the operator-log acquisition gate, timestamped
nvidia-smi parsing, Session replay). Tests: `tests/test_lab_energy_gpu.py`.

```
python -m ciw lab run T115 T116 T117 T118 T119 T120 T121 T122 T123 T124 T125 --output-dir <dir>
python -m ciw lab report T121 --retained <dir>
```

The whole section runs in about 5 s on one core; the tests take about 10 s.
T119, T123, T124 and T125 read `examples/energy-accuracy` through
`ciw.lab.runner.repository_path`, so an installed package finds the fixtures
when `CIW_LAB_REPOSITORY_ROOT` names a checkout. Without them T119, T124 and
T125 report `blocked` (with their static answers and physical claims) and
T123 reports `partial` with its energy-record field audit recorded as not
run.

## What this environment could and could not measure

The development host has no RAPL powercap counters, no NVIDIA GPU or NVML,
no CUDA toolchain and no Julia. `rustc` is available. The lab runner acquires
no energy measurement on any host: CPU and GPU energy come only from captures
an operator makes outside the runner, which the tasks analyze read-only. Its
only counter access is the `hardware:rapl` availability probe, which T115
makes only when a capture is supplied: it reads one `energy_uj` value to
confirm readability and discards it
(`test_rapl_probe_is_the_only_counter_read` exercises the real probe on a
simulated powercap tree). Consequently:

| Quantity | Status here | Where it is recorded |
| --- | --- | --- |
| CPU package energy per trajectory (gross and idle-subtracted) | not measured (no capture) | T115 physical findings, `not_established` |
| GPU energy per batch; NVML counter accuracy | not measured (no GPU) | T116 `blocked`; physical and sensor-performance findings, `not_established` |
| RTX 2080 power, temperature, clock, utilization, steady state, batch and kernel time | not measured | T118 `blocked`; eight physical findings, `not_established` |
| Physical energy per accepted result | not measured (no operator log) | T119 physical finding, `not_established` |
| Energy of float32 versus float64 | not measurable: no capture path runs a float32 RK4 workload | T120 physical finding, `not_established`; deferred question |
| Real GPU reduction orders | not observed | T121 physical finding, `not_established` |
| Julia and GPU implementations of the sphere RK4 kernel | not written (on any host) | T117 findings, `not_established`; deferred question |
| RTX 2080 kernel-only duration | not ingested (on any host) | T118 finding, `not_established`; deferred question |

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
  step). Energy needs an operator capture (protocol below).
- **T117 Python/Rust.** A std-only Rust RK4 kernel is embedded as
  `energy_gpu_kernels.RUST_SOURCE`, compiled with `rustc -O -C
  codegen-units=1` into a temporary directory (the scratch path is remapped,
  so the binary digest is reproducible for a given rustc), and exchanges JSON
  over stdin/stdout. Its `rhs()` counts its own calls (4NT = 6144 for the
  six trajectories, a finding of its own). Its endpoints are bitwise
  identical to the Python closed-form kernel on this
  Linux host (same operation order, same glibc `sin`/`cos`); the finding
  tolerance is 1e-12, and cross-platform bitwise identity is not claimed. The
  task also sends the kernel a malformed and a nonfinite (pole) input and
  records both refusals. Both agreements (Rust/Python and generic
  Christoffel/closed form) are `cross_implementation` checks: declaring one
  ciw kernel an independent check of another is refused by
  `ciw.lab.evidence` (a regression test, not a finding), so the label is
  `numerically_verified`, never `independently_verified`. The rustc version
  and binary digest are in the report's runtime identity, so a label change
  on a host without rustc can be attributed. No Julia or GPU implementation
  of this sphere kernel exists in the repository (`src/ciw/energy_cuda.py` is
  the Gaussian VI PTX kernel), so T117 stays `partial` on every host; its
  Julia and GPU notes derive from the probes but always say no kernel was
  written. Deferred research question: a CUDA/PTX RK4 kernel of this geodesic
  (for example through the `ciw.energy_cuda` JIT path) and a Julia kernel
  with the operation order of `step_rk4`.
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
  Widening the boundary to the whole run multiplies the metric by 7. When
  `CIW_LAB_ENERGY_LOG` names an operator log, T119 recomputes energy per
  accepted replica solve from its raw readings and outputs and reports it as
  `hardware_measured` through the T116 acquisition gate (plus a GPU probe
  that answers in T119); T119 is then `completed`.
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
  N = 128 — lower precision is not cheaper at every accuracy target. No
  capture path measures a float32 RK4 workload (T115 brackets the float64
  generic integrator, T116 the Gaussian VI kernel); a dtype-parameterized
  capture of `rk4_batch` is the deferred research question.
- **T122 bounded free energy.** On a declared two-latent Gaussian problem the
  identity F + log Z = KL holds at every iterate to 3e-14 nats; with the
  declared normalization (condition number 7.3) KL decreases monotonically to
  1e-21 nats in 93 of at most 512 iterations. Counterexamples: a mean step 1.2×
  the stability bound (spectral radius 1.4) makes KL grow to 5e18; unit scales
  (condition number 1.3e3) leave KL at 3.99 nats after 512 iterations, while
  the raw-unit posterior itself is scale-invariant to 2e-15.
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
  `analytic`.
- **T124 raw telemetry.** An audit of the energy-log format, not of a
  hardware capture: the task retains the four fixtures' exact bytes as
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
  (the origin field is a declaration).
- **T125 replay.** Through a `ciw.session.Session`, every fixture keeps one
  `numerical_result_id` across original, reproduction and replay, each with
  fresh execution, result and bundle identities; the workspace restores
  exactly; an edited bundle is refused with stale digests
  (`Energy analysis bundle identity, schema or size differs`) and with
  recomputed digests (`Retained energy analysis binding differs`).

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

None of this shows what CUB, cuBLAS or a hand-written kernel does on an RTX
2080; that is the physical-domain finding left `not_established`, and the
follow-up is T148 (deterministic reduction policy) and T147 (CPU/GPU outputs).

## Protocol for a RAPL host (T115)

The capture is an operator action; `ciw lab run` acquires no measurement
(its `hardware:rapl` probe reads one counter value to confirm readability and
discards it).

1. On a Linux host whose `/sys/class/powercap/intel-rapl:*/energy_uj` is
   readable: `python -m ciw.lab.energy_gpu_telemetry rapl-capture
   runs/rapl-<date>.json` (optional `--repeats N`, default 3). It brackets N
   batches of the T115 workload (six sphere geodesics, RK4, N = 256) with
   package counter reads, then brackets an idle interval of the same
   monotonic length, and writes the raw counters, UTC and monotonic brackets,
   the declared workload and the host identity (CPU model, platform, RAPL
   zones). An existing file is refused.
2. `CIW_LAB_RAPL_LOG=runs/rapl-<date>.json python -m ciw lab run T115
   --output-dir <dir>` on the same host.

T115 retains the capture bytes as an artifact and reports gross package
energy per trajectory (background-inclusive, idle not subtracted) and
idle-subtracted energy per trajectory as `hardware_measured` only when the
capture names this workload, its host identity equals the analyzing host's,
and a RAPL probe answers in the task. The acquisition record says the capture
is operator-made and unauthenticated; the calibration is `not_applied` (RAPL
is a model-based counter).

## Protocol for the RTX 2080 host (T116, T118)

The lab runner never acquires hardware data. On the GPU host the operator
captures, then the lab analyzes (`R=runs/rtx2080-<date>`; `ciw energy record`
refuses an existing output directory, so the capture goes into `$R/capture`):

1. `mkdir -p $R` and `ciw energy probe --gpu-index 0` — must return a reading
   with `status: ok`. NVML documents `nvmlDeviceGetTotalEnergyConsumption`
   for Volta-or-newer *fully supported* devices; GeForce support is not
   documented, so this probe must confirm the counter on the RTX 2080.
2. Start the utilization sidecar in the background, in UTC:
   `TZ=UTC nvidia-smi --query-gpu=timestamp,uuid,name,utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.sm,clocks.mem,pstate --format=csv,nounits -lms 100 -f $R/smi.csv`
3. Capture:
   `ciw energy record --problem examples/energy-accuracy/problem.json --output-dir $R/capture --duration 10 --replicas 4096 --warmup-batches 2 --idle-duration 2 --gpu-index 0`
   and stop the sidecar.
4. `ciw energy replay $R/capture/log.json` — offline recomputation.
5. Kernel durations in a separate pass (profiling perturbs timing and energy):
   `nsys profile --trace=cuda -o $R/nsys python -m ciw energy record --problem examples/energy-accuracy/problem.json --output-dir $R/capture-nsys --duration 10 --gpu-index 0`,
   then `nsys stats --report cuda_gpu_kern_sum $R/nsys.nsys-rep`. Retain the
   output; the lab does not ingest it yet (deferred research question:
   ingest it with its raw bytes retained and bound through the acquisition
   gate, so T118 can report kernel-only duration).
6. `CIW_LAB_ENERGY_LOG=$R/capture/log.json CIW_LAB_NVIDIA_SMI_CSV=$R/smi.csv CIW_LAB_NVIDIA_SMI_UTC_OFFSET=+00:00 python -m ciw lab run T116 T118 T119 --output-dir <dir>`

T116 then reports gross device energy per measured batch, and T119 the
energy per accepted replica solve of the same log; the NVML counter's
accuracy and resolution stay `not_established` (undeclared by NVML, no
external meter). T118 reports NVML power, temperature and graphics clock over
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
analysis is eligible, its raw bytes (and the sidecar's) are retained
byte-exactly as artifacts, and its sensor block (UUID, name, driver version,
NVML version, NVML library SHA-256) equals the NVML identity of that device on
the analyzing host; T118 also requires the name to contain "RTX 2080" and
every in-window sidecar row to name the same UUID. The acquisition record's
device string says "operator-captured log, unauthenticated". This binds a
record to hardware and software present on the host. It does **not**
authenticate the capture: a relabelled synthetic log paired with a matching
host identity passes (see
`test_operator_log_gate_trust_boundary_is_the_host_identity` and the T124
relabelling counterexample). Signed capture is the unresolved next step.
Without a GPU, or on a GPU host without a log, T116 and T118 are `blocked`
and record their physical claims as `not_established` findings with the same
claim texts the operator path uses, so the regression gate can match them
across hosts.

## What these results do not show

- No physical energy, power, temperature, utilization or kernel time was
  measured; no efficiency ranking of Python, Rust, float32 or float64 follows
  from this section.
- Python/Rust agreement is agreement of two ciw implementations of one
  algorithm, not independent verification.
- Emulated reduction orders are plausible GPU orders, not observed ones, and
  the counterexample witnesses were found by search: they show existence, not
  frequency.
- Variational free energy is a dimensionless information quantity of a
  normalized model; it is not a thermodynamic free energy and says nothing
  about the joules the computation dissipated.
- Replay determinism is not verification by another party, and a valid log
  digest is not evidence that a device produced the readings.
