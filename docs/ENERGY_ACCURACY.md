# Energy to accuracy on the workstation

The computer can supply the first physical observations: actual GPU energy
counter readings around a bounded Gaussian inference workload. Statistical
free energy (nats), physical energy (joules), and elapsed time (seconds) remain
separate quantities. This is an experimental measurement profile, without a
calibrated uncertainty budget or a whole-machine energy claim.

## Capture and replay

Install CIW using the [quickstart](quickstart.md). Capture needs a supported
NVIDIA GPU, its installed driver, NVML accumulated-energy support, and CUDA
compute capability 5.0 or newer. No CUDA toolkit, Julia, or external provider
checkout is required for this bounded profile. Linux and Windows CPU-only
machines can inspect/replay retained logs and run the synthetic examples.

```sh
python -m ciw energy probe
python -m ciw energy record --problem examples/energy-accuracy/problem.json --output-dir results/energy-run-1 --duration 3 --replicas 65536
python -m ciw energy replay results/energy-run-1/log.json --output results/energy-run-1/replayed.json
```

Each capture requires a new output directory. It writes a flushed, fsynced
`journal.jsonl`, followed by `log.json` and `report.json`. Failure preserves the
journal and a failure record. A partial or malformed final file is rejected;
the journal is diagnostic evidence, not automatically promoted to a complete
measurement. Replay recalculates accuracy and energy from the retained log;
it does not reacquire telemetry or execute the solver.

The source specification retains the normalized two-latent Gaussian model,
observations, complete initial mean/covariance, update settings and KL target.
An independent observation-space Gaussian reference supplies the exact model
posterior. CPU planning selects a fixed iteration count that reaches the
target; `--iterations` can instead prescribe a count, including a failure case.
The actual GPU outputs are checked independently against the reference.
Planning is outside the measured phases and does not establish GPU accuracy.

One GPU thread executes each identical declared problem using binary64 mean
gradient descent and full precision relaxation. Replicas are repeated work for
measurement, not independent statistical observations. Fixed PTX is compiled
by the installed driver; no arbitrary kernel text is accepted from records.
All output rows must be bitwise identical before lossless constant-row encoding.
The retained row, count and expanded-byte digest reconstruct every output.

## Measurement boundary

The five ordered phases are CUDA startup (context, JIT and input upload),
warmup, idle before, warmed measurement, and idle after. Reference planning,
Python process startup and NVML initialization precede these phases. Each solve
includes launch, synchronization, output transfer and host result validation.
Phase timing also includes sampling, encoding, hashing and durable journal I/O.
Repeated batches continue to the minimum duration or declared batch limit;
hitting the limit early makes the run ineligible for qualified cost reporting.

Energy uses the raw endpoint counter difference over host-call brackets.
The report retains both phase elapsed time and the enclosing energy bracket
duration, including their alignment overhead. Nanosecond timestamps are host
clock representations, not sensor timing precision. Elapsed time uses a
monotonic clock; UTC is retained independently and backwards jumps are flagged.
Power, temperature and graphics clock are optional context sampled after the
energy call; they have no asserted simultaneous sensor timestamp.

[NVML documents](https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceQueries.html)
the accumulated counter in millijoules since driver reload. CIW preserves its
integer readings as decimal strings, avoiding JSON integer rounding. A 1 mJ
counter unit does not establish 1 mJ effective resolution or accuracy. This
profile leaves accuracy, update interval and effective resolution unknown.

The sensor UUID must match the CUDA workload UUID. Gross GPU-device energy
includes background activity and associated device circuitry. Idle controls
are reported separately without subtraction. CPU activity, memory outside the
GPU domain, display/system power and wall-plug energy are not measured. In
particular, GPU readings cannot establish the cost of a CPU NumPy solve.
[Linux powercap](https://cdn.kernel.org/doc/html/latest/power/powercap/powercap.html)
offers CPU-domain counters on supported hosts; that collector is future work.

Repeated/zero counters, decreases, failed reads, absent brackets, missing batch
reads, device mismatch, and unfinished phases remain visible. Decreases are
classified as ambiguous reset/wrap; no modulo correction is invented. Invalid
coverage yields null energy, not zero. Zero increments cannot qualify a run as
free computation. A target miss can still retain consumed energy but cannot
produce joules per qualified solve.

The reported cost is **amortized background-inclusive GPU joules per completed
solve at the declared fixed iteration count and KL target**. It is not time or
energy at first threshold crossing, a minimum-energy result, or a calibrated
comparison across hardware. Compare only declared equivalent problems,
targets, arithmetic, batching and measurement scopes; examine the idle and
thermal context. Repeated captures are separate physical occurrences.

## Shared Workbench

```sh
python -m ciw serve --output-dir results/energy-session
python examples/energy-accuracy/run.py --log results/energy-run-1/log.json --output-dir results/energy-client
```

The built-in `ciw.energy-accuracy.v1` operation analyzes a retained
`energy-accuracy` source. It needs no runtime binding and remains available
after reopening a workspace. The terminal and Godot Workbench project the same
energy, time, accuracy and context records. Shared requests cannot start a GPU
capture or select local driver paths. Explicit bundle replay creates fresh
analysis execution/result identities over the **same physical measurements**.
Reopening validates a retained analysis against a fresh one: statuses, reasons,
counts and decimal counter strings must match exactly, while the accuracy
metrics are compared with a binary64 tolerance because the Gaussian reference
runs through the host's linear-algebra kernels and its roundoff-level errors
differ between CPUs. Replay itself requires an exact numerical match; the
analysis runtime identity records a numerical kernel probe, so a host whose
kernels round differently is refused before analysis as a runtime identity
difference, leaving the retained bundle valid.

The source log is bounded to 4 MiB; each workflow bundle is bounded to 8 MiB,
within the shared workspace budget. The service accepts WebSocket frames up to
8 MiB, accommodating a base64-encoded source plus its request envelope.
Embedded evidence hashes provide integrity and reproducibility, not hardware
authentication or independent validation of sensor accuracy. A physical origin
label is the recorder's assertion. Synthetic fixtures stay explicitly synthetic
and never qualify as physical measurement evidence.

## Verification and next experiments

The installed-wheel gate runs on Linux and Windows:

```sh
python scripts/check_energy.py --output-dir results/energy-check
python scripts/check_energy.py --output-dir results/energy-hardware-check --hardware
```

CPU CI checks raw-counter failure handling, strict log validation, independent
Gaussian accuracy, retained replay and shared-session integration using
synthetic data. Only the explicit hardware gate requires CUDA/NVML; it permits
no hardware test skips. Synthetic fixtures include reset, missing-read and
accuracy-failure cases. They are examples of contract behavior, not measured
device performance.

Next: replicate runs with randomized batch/target order, quantify counter
update behavior and external-meter agreement, add supported CPU-domain capture,
then fit and test a thermal model on held-out observations. A Rust collector or
Julia reference should consume the same declared records when its timing or
independent numerical contribution is demonstrated.
