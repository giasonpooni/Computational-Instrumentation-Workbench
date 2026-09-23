# Energy and GPU experiments (T115–T125)

Section 8 of the computational-experimentalist queue. Implementation:
`src/ciw/lab/energy_gpu.py` (tasks), `energy_gpu_kernels.py` (geodesic RK4 in
any float dtype, the embedded Rust kernel, reduction orders, typed
quantities) and `energy_gpu_telemetry.py` (RAPL helpers, fixture tampering,
the operator-log acquisition gate, Session replay). Tests:
`tests/test_lab_energy_gpu.py`.

```
python -m ciw lab run T115 T116 T117 T118 T119 T120 T121 T122 T123 T124 T125 --output-dir <dir>
python -m ciw lab report T121 --retained <dir>
```

The whole section runs in about 4 s on one core; the tests take about 4 s.

## What this environment could and could not measure

The development host has no RAPL powercap counters, no NVIDIA GPU or NVML,
no CUDA toolchain and no Julia. `rustc` is available. Consequently:

| Quantity | Status here | Where it is recorded |
| --- | --- | --- |
| CPU package energy per trajectory | not measured (no RAPL) | T115 physical finding, `not_established` |
| GPU energy per batch | not measured (no GPU) | T116 `blocked` with the protocol below |
| RTX 2080 utilization, temperature, power, kernel time | not measured | T118 `blocked` with the protocol below |
| Physical energy per accepted result | not measured | T119 physical finding, `not_established` |
| Energy of float32 versus float64 | not measured | T120 physical finding, `not_established` |
| Real GPU reduction orders | not observed | T121 physical finding, `not_established` |
| Julia and GPU implementations of the kernel | not run | T117 findings, `not_established` |

Every joule figure in this section comes from the synthetic fixtures in
`examples/energy-accuracy` (`origin: synthetic_fixture`) and is a synthetic
value. Wall-clock and CPU times are retained only as artifacts
(`timing.json`); they are never findings.

## What was computed

- **T115 work proxies.** Fixed-step RK4 spends exactly 4N = 1024
  right-hand-side evaluations per sphere geodesic (N = 256), with endpoint
  error 1.1e-9 against the exact great circle; adaptive Dormand–Prince
  (rtol 1e-9) uses 295–487 evaluations per trajectory. When a readable
  `intel-rapl` tree exists, the task brackets three repeated fixed-step
  batches with package counter reads (one wrap allowed) and records a
  background-inclusive `hardware_measured` finding; otherwise it stays
  `partial`.
- **T117 Python/Rust.** A std-only Rust RK4 kernel is embedded as
  `energy_gpu_kernels.RUST_SOURCE`, compiled with `rustc -O` into a temporary
  directory, and exchanges JSON over stdin/stdout. Its endpoints are bitwise
  identical to the Python closed-form kernel on this Linux host (same
  operation order, same glibc `sin`/`cos`); the finding tolerance is 1e-12, and
  cross-platform bitwise identity is not claimed. Because both kernels are ciw
  code, declaring one an independent check of the other is refused by
  `ciw.lab.evidence` — agreement is `numerically_verified`, never
  `independently_verified`.
- **T119 energy per accepted result.** Definition:
  `E_acc = (counter(last measurement read) − counter(first measurement read)) / #accepted solves`,
  where a solve is accepted when its retained batch output has KL ≤ the declared
  target. Baseline fixture: 0.05 J per accepted solve (synthetic). The metric
  is withheld for the reset, missing-bracket and under-target fixtures.
  Counterexample: dividing gross energy by executed solves gives 0.05 J/solve
  for the under-target fixture, which has zero accepted solves. Widening the
  boundary to the whole run multiplies the metric by 7.
- **T120 precision.** float64 RK4 converges at order 3.96 (N = 16..256);
  float32 saturates at a roundoff floor of about 6e-7 and then grows. Both
  precisions do identical arithmetic per step (80 flops, 8 transcendental
  calls); only the state width halves. Counterexample: float32 cannot reach
  1e-7 at any N ≤ 2048, while float64 reaches it at N = 128 — lower precision
  is not cheaper at every accuracy target.
- **T122 bounded free energy.** On a declared two-latent Gaussian problem the
  identity F + log Z = KL holds at every iterate to 3e-14 nats; with the
  declared normalization (condition number 7.3) KL decreases monotonically to
  1e-21 nats in 93 of at most 512 iterations. Counterexamples: a mean step 1.2×
  the stability bound (spectral radius 1.4) makes KL grow to 5e18; unit scales
  (condition number 1.3e3) leave KL at 3.99 nats after 512 iterations, while
  the raw-unit posterior itself is scale-invariant to 2e-15.
- **T123 nats versus joules.** A typed quantity algebra refuses to add,
  compare or convert information (nat, bit) and energy (J, mJ); division
  yields compound dimensions such as energy/information. CIW energy records
  carry joules only in `_j`/`_mj`/`_mw` fields and nats only in `kl_*_nats` and
  the Gaussian reference; variational records carry no joule field; each
  workbench energy panel has one unit. Counterexample: an untyped
  `F + E` changes from 8.6 to 208.4 when E is expressed in mJ instead of J.
- **T124 raw telemetry.** All four fixtures retain 47 raw readings with
  monotonic and UTC brackets and 14 device/runtime identity fields; nine
  tampering classes are refused with specific messages. Counterexamples: a
  log whose readings were doubled and then resealed validates (sealing is
  integrity, not authenticity), and relabelling a fixture as
  `physical_measurement` makes it eligible for physical comparison (the origin
  field is a declaration).
- **T125 replay.** Through a `ciw.session.Session`, every fixture keeps one
  `numerical_result_id` across original, reproduction and replay, each with
  fresh execution, result and bundle identities; the workspace restores
  exactly; an edited bundle is refused with stale digests
  (`Energy analysis bundle identity, schema or size differs`) and with
  recomputed digests (`Retained energy analysis binding differs`).

## Reduction-order findings (T121)

Emulated on the CPU with n = 16384 values: sequential, adjacent-pair tree,
strided (shared-memory style) tree, two-pass blocked, block-sequential, Kahan,
and eight atomicAdd completion orders of 256-element block partials.

- Positive data (u³·1000 + 1e-3): float32 sums spread by 4.0e-6 relative across
  orders; the sequential fold is worst (error 15.9 on 4.1e6), every other
  order errs by under 0.7. float64 spread is 6.8e-16. Every error lies
  inside the a priori bound γ₍ₙ₋₁₎ Σ|xᵢ|.
- Cancellation data (±a pairs plus 64 × 2⁻⁸, exact sum +0.25, first flipping
  PCG64 seed 201): the float32 sequential sum is −0.024 while every other order
  is positive — **reduction order alone flips a sign/threshold decision**.
  float64 gets the sign right in every order.
- The eight atomic completion orders of identical float32 block partials give
  four distinct sums: identical inputs need not give identical bits.
- Max reductions are bitwise order-invariant.
- A decision rule that only decides when |S − T| exceeds the a priori bound
  never contradicts itself across orders (0 contradictions over 17 thresholds
  × 4 datasets); unguarded `S > T` flips at the threshold nearest the exact
  sum, even in float64 on the positive data. The guard is conservative:
  about half of the tested decisions stay undecided.

None of this shows what CUB, cuBLAS or a hand-written kernel does on an RTX
2080; that is the physical-domain finding left `not_established`, and the
follow-up is T148 (deterministic reduction policy) and T147 (CPU/GPU outputs).

## Protocol for the RTX 2080 host (T116, T118)

The lab runner never acquires hardware data. On the GPU host the operator
captures, then the lab analyzes (`R=runs/rtx2080-<date>`; `ciw energy record`
refuses an existing output directory, so the capture goes into `$R/capture`):

1. `mkdir -p $R` and `ciw energy probe --gpu-index 0` — must return a reading
   with `status: ok` (confirms the driver exposes
   `nvmlDeviceGetTotalEnergyConsumption`; NVML documents it for Volta and
   newer, which includes Turing).
2. Start the utilization sidecar in the background:
   `nvidia-smi --query-gpu=timestamp,uuid,name,utilization.gpu,utilization.memory,temperature.gpu,power.draw,clocks.sm,clocks.mem,pstate --format=csv,nounits -lms 100 -f $R/smi.csv`
3. Capture:
   `ciw energy record --problem examples/energy-accuracy/problem.json --output-dir $R/capture --duration 10 --replicas 4096 --warmup-batches 2 --idle-duration 2 --gpu-index 0`
   and stop the sidecar.
4. `ciw energy replay $R/capture/log.json` — offline recomputation.
5. Kernel durations in a separate pass (profiling perturbs timing and energy):
   `nsys profile --trace=cuda -o $R/nsys python -m ciw energy record --problem examples/energy-accuracy/problem.json --output-dir $R/capture-nsys --duration 10 --gpu-index 0`,
   then `nsys stats --report cuda_gpu_kern_sum $R/nsys.nsys-rep`.
6. `CIW_LAB_ENERGY_LOG=$R/capture/log.json CIW_LAB_NVIDIA_SMI_CSV=$R/smi.csv python -m ciw lab run T116 T118 --output-dir <dir>`

T116 then reports gross device energy per measured batch; T118 reports NVML
power, temperature and graphics clock, host-bracketed batch durations (these
include launch, synchronization and copy) and sidecar utilization. Kernel-only
duration stays `not_established` until an nsys report is analyzed.

**Acquisition gate.** A physical finding from such a log is
`hardware_measured` only when the log declares `physical_measurement`, its
analysis is eligible, and its sensor block (UUID, name, driver version, NVML
version, NVML library SHA-256) equals the NVML identity of that device on the
analyzing host; T118 also requires the name to contain "RTX 2080" and every
sidecar row to name the same UUID. This binds a record to hardware and
software present on the host. It does **not** authenticate the capture: a
relabelled synthetic log paired with a matching host identity passes (see
`test_operator_log_gate_trust_boundary_is_the_host_identity` and the T124
relabelling counterexample). Signed capture is the unresolved next step.

## What these results do not show

- No physical energy, power, temperature, utilization or kernel time was
  measured; no efficiency ranking of Python, Rust, float32 or float64 follows
  from this section.
- Python/Rust agreement is agreement of two ciw implementations of one
  algorithm, not independent verification.
- Emulated reduction orders are plausible GPU orders, not observed ones.
- Variational free energy is a dimensionless information quantity of a
  normalized model; it is not a thermodynamic free energy and says nothing
  about the joules the computation dissipated.
- Replay determinism is not verification by another party, and a valid log
  digest is not evidence that a device produced the readings.
