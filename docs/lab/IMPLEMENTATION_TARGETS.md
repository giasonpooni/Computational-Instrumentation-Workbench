# Implementation targets: Rust, Python, Julia, C++, GPU and FPGA (T142-T154)

Section 10 of the computational-experimentalist queue decides where each kind
of work should run and fixes the contracts that let results cross language and
device boundaries without losing their evidence identity. Code:
`src/ciw/lab/implementation_targets.py` (tasks) with helpers
`implementation_targets_kernels.py` (operation counts, reductions, comparison
harness), `implementation_targets_serial.py` (canonical JSON, Rust probe),
`implementation_targets_architecture.py` (C++ inventory, import-graph scan),
`implementation_targets_fpga.py` (telemetry frames, bitstream identity,
compatibility, link simulation) and `implementation_targets_authority.py`
(actuator write policy, control proposals). Tests:
`tests/test_lab_implementation_targets.py`.

Run the section with

```
python -m ciw lab run T142 T143 T144 T145 T146 T147 T148 T149 T150 T151 T152 T153 T154 --output-dir results/lab-implementation-targets
```

It takes about 5 s on one core. `rustc` is used when present (T142, T146);
without it those Rust findings are recorded as not established and the tasks
become `partial`.

**What this section does not establish.** No GPU, FPGA, Julia runtime or
industrial C/C++ library runs here. Bitstreams, link statistics and
authorization records are synthetic placeholders. Rust agreement is agreement
between two CIW-authored implementations (`cross_implementation`), not
independent verification. Every claim about physical links, deployment,
production acceptance, machine safety, industrial readiness or actuator
authority is recorded as a finding in its proper domain and is
`not_established`.

## Kernel ranking

T142 counts floating-point operations exactly by running scalar restatements of
each kernel on a counting number type (`Scalar`), checks each restatement
against the core NumPy kernel (max difference below 1e-12), and counts the
interpreter calls that `ciw` code issues per kernel call with a profile hook.
Wall-clock timings are kept only in `kernel-timings.json`.

| Kernel (restated) | Flops per call | Trig calls | Interpreter calls per call |
| --- | --- | --- | --- |
| Generic embedded-surface geodesic RHS (torus) | 173 | 4 | 73 |
| Geodesic + Jacobi RHS | 178 | 4 | 79 |
| RK4 step on the 8-state system | 819 = 4 x 178 + (13n + 3), n = 8 | 16 | 317 |
| Joseph-form Kalman update, n = 4, m = 2 | 453 | 0 | 5 |

The ranking is by Python dispatches that porting a unit alone removes from one
reference experiment (a 2000-step Jacobi transfer plus 2000 Kalman updates);
callbacks that stay in Python are not removed. Ties go to the higher
determinism need.

| Rank | Port unit | Removable dispatches | Determinism need |
| --- | --- | --- | --- |
| 1 | Fused geodesic + Jacobi RK4 transfer loop | about 638,000 | high: retained trajectories, bitwise replay over thousands of dependent steps |
| 2 | Generic embedded geodesic RHS | 576,000 | high: fixed contraction order |
| 3 | Kalman update | 8,000 | medium: symmetric, positive-definite covariance |
| 4 | RK4 step alone (RHS stays in Python) | 0 | high, but negligible alone |

Recommendation: port the fused transfer loop as one Rust kernel; do not port
the RK4 step without its right-hand side; keep the Kalman update in NumPy until
filter rates justify it. A Rust port of the fused loop for the unit sphere
(closed-form Christoffel symbols) reproduces `ciw.lab.jacobi.transfer` to
4.4e-16 after 600 steps. Its retained timing ratio overstates what a generic
port would gain, because the port is specialised to the sphere.

Dispatch counts depend on the Python and NumPy versions (the event semantics
used were checked identical on CPython 3.11-3.13), and array operators are not
counted; the regression tolerance on them is 25%.

## Industrial interfaces

T143 records which interfaces need C/C++ and where the CIW boundary sits.

| Interface | Native libraries | Why native | Boundary | Direction / write path |
| --- | --- | --- | --- | --- |
| OPC UA client | open62541 (C), vendor C++ SDKs | certified stacks, security policies, subscription timing | pinned subprocess | read-only / absent |
| EtherCAT master monitoring | SOEM, IgH master, TwinCAT ADS | cyclic process data with microsecond deadlines, raw sockets or kernel drivers | pinned subprocess | read-only / disabled |
| Vendor camera SDKs (GenICam GenTL) | Basler pylon, Spinnaker, Vimba X | proprietary drivers, zero-copy buffers, hardware triggers | pinned subprocess | read-only / absent |
| Point clouds | PCL, Open3D | C++ registration and reconstruction kernels with threading | pinned subprocess | geometry exchange / absent |
| CAD kernel | OpenCASCADE | B-rep, STEP/IGES translation, tolerances | pinned subprocess | geometry exchange / absent |

The validator refuses an in-process binding, a write-capable direction, an
enabled write path and an entry without identity pins. No library was
installed or exercised.

## Python orchestration boundary

T144 parses every module of the installed `ciw` package with `ast` (nothing is
imported) and checks:

- the transitive `ciw` import closure of `ciw.lab.evidence`, `ciw.lab.report`
  and `ciw.core.identities` uses only the standard library, loads no native
  code and spawns no process (closure: exactly those three modules);
- native loading (`ctypes`, `cffi`) appears only in the declared hardware
  energy probes `ciw.energy_cuda` and `ciw.energy_nvml`;
- no call passes `shell=True`, and every module that spawns a process mentions
  a runtime identity (`revision`, `source_tree`, `runtime_identity`, `sha256`
  or `digest`);
- no compiled extension ships in the package.

Five forged mutations (a `ctypes` import in the evidence module, `numpy` in the
report module, a spawn in the evidence module, a `shell=True` provider, a
provider without identity) are each flagged. A text search for `subprocess`
is not an adequate substitute: many modules mention it without spawning. The
identity rule is a heuristic, and dynamic imports are not seen.

## Julia

T145 is `partial`: no Julia runtime is present. The pin procedure follows
[docs/JULIA_SP1.md](../JULIA_SP1.md): candidate Julia 1.10.12 LTS (not yet
accepted), a dedicated project with OrdinaryDiffEqTsit5 and SciMLBase,
instantiate and precompile separately, commit the machine-generated
`Project.toml` and `Manifest.toml`, run with `--project`,
`--startup-file=no` and `--threads=1`, and compare the worker handshake
identity (Julia version, platform, executable, worker source, project,
manifest, package artifacts, threads, numerical preferences, system image)
before dispatching through SCR. Meanwhile SymPy demonstrates the symbolic role:
the torus Christoffel symbols and Gaussian curvature derived symbolically
agree with `ciw.lab.surfaces.Torus` to 4.4e-16 (independent implementation).

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
- floats: finite binary64 only, shortest round-trip digits in CPython `repr`
  form: fixed notation when the decimal point position p satisfies
  -4 < p <= 16 (`1.0`, `-0.0`, `1000000000000000.0`, `0.0001`), otherwise
  `d[.ddd]e(+|-)XX` (`1e+16`, `1e-05`, `5e-324`); integers and floats stay
  distinct (`1` and `1.0`);
- refusal codes: `nonfinite_number`, `unsafe_integer`, `non_string_key`,
  `invalid_unicode`, `nesting_too_deep` (more than 64 enclosing containers),
  `unsupported_type`.

Test vectors (18 accepted, 9 refused) are retained in
`artifacts/T146/test-vectors.json` with their bytes and sha256; the set digest
over `[name, sha256]` pairs is `ef4d1738...8c65f65`. Examples:

| Vector | Canonical bytes | sha256 (UTF-8 form) |
| --- | --- | --- |
| empty-object | `{}` | `44136fa355b3678a...` |
| signed-zero | `[0.0,-0.0,{"z":-0.0}]` | `e1a5ea3a03a3af50...` |
| integral-floats | `[1.0,-1.0,100.0,9007199254740992.0,1000000000000000.0,1e+16,1e+21,1e+22]` | `1a822860d54a8312...` |
| exponent-switch | `[0.0001,1e-05,0.00012345,1234567890123456.0,1.2345678901234568e+16]` | `b5d7e3579c6364f2...` |
| unicode-bmp | `"é日本語  "` written raw | `e28c9c5bcbf4f143...` (ASCII variant `4e2c3d77419efa08...`) |

Findings:

- `ciw.telemetry.canonical` reproduces the UTF-8 form on every accepted
  vector; `ciw.core.identities.canonical_json` reproduces the ASCII-escaped
  variant (`ensure_ascii=True`). They differ on 5 of 18 vectors, all with
  characters outside U+0020-U+007E, so CIW does not yet have one encoding.
  Changing `ciw.core.identities` would change every retained identity and
  needs a versioned migration.
- `ciw.core.identities.canonical_json({1: "x"})` equals
  `canonical_json({"1": "x"})`: distinct values share a content identity. The
  specification and `ciw.telemetry.canonical` refuse non-string keys.
- Both Python encoders accept integers beyond 2^53; the identities encoder
  also accepts lone surrogates and depth 65.
- A Rust implementation (std only, compiled with `rustc` at run time) gives
  byte-identical UTF-8 and ASCII forms on every vector and the same refusal
  codes. Same origin, so `numerically_verified`.
- CPython's shortest digits agree with NumPy's Dragon4 on 2064 binary64 values
  (`independently_verified`).
- This is not RFC 8785 (JCS): JCS writes `1` for `1.0` and `10000000000000000`
  for `1e+16`, and sorts keys by UTF-16 code units (U+1F600 before U+FF21).

## CPU and GPU comparison

T147 provides `compare_outputs(reference, candidate, policy)` with a bitwise
policy, an analytic-bound policy and an absolute/relative policy. For sums
whose terms pass through k roundings, the bound is
`gamma_k * sum|a_j x_j|` with `gamma_k = k u / (1 - k u)`; two outputs are
compared against the sum of their bounds. On 128 batched dot products of
length 1024 (sequential, pairwise and 32-lane blocked orders; float64 and
float32), the bitwise policy flags 123 of 128 float64 rows, the bound policy
accepts float64 reordering (at most 0.0017 of the bound), float32 violates the
float64 policy on every row but stays at 0.0087 of its own bound, and a
dropped partial product is flagged under both policies. No GPU was present,
so CPU/GPU agreement is not established.

## Reduction policies

| Use | Policy | Guarantee |
| --- | --- | --- |
| Identity-bearing sums | exact accumulation (integers at scale 2^1074, one final rounding) | correctly rounded, hence bitwise independent of order; equals `math.fsum` |
| Fixed-layout arrays | pairwise tree split at n // 2 over the stored order | bitwise reproducible only for the same order and length; error <= gamma_{ceil(log2 n)} sum|x| |
| Streaming accumulators | Neumaier compensation | error <= 2u|S| + O(n u^2) sum|x|; not order-invariant |
| Not for identities | unordered sequential sums, `numpy.sum`/BLAS, Kahan without the Neumaier branch | order- or implementation-dependent; Kahan returns 0 for [1, 1e100, 1, -1e100] |

Across 25 orders of four datasets, exact accumulation gave one result per
dataset and matched `math.fsum` everywhere; fixed-tree pairwise gave 4
distinct results on uniform data; every observed error stayed below its
bound (the second-order constant in the compensated bounds is taken as
4 n u^2 sum|x|, conservatively).

## FPGA telemetry-only interface

Frame (big-endian), one frame type, no host-to-device field:

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
| 28 + 4c | 4 | CRC-32/IEEE over all preceding bytes |

The decoder checks the CRC first, then refuses command and write paths
(`command_path_refused`), unknown types, reserved flags and length
mismatches. The host receiver has no sending method. The CIW CRC-32 matches
`zlib` and the check value 0xCBF43926; every single-bit error (384/384) and
every 2-32 bit burst sampled (11408/11408) in a 48-byte frame is refused.

**Bitstream identity** (`ciw.fpga-bitstream-identity.v1`, T150): bitstream
sha256 and size, toolchain name and exact version (ranges and `latest`
refused), part, per-file constraint digests and their canonical digest, source
tree digest, synthesis options, a `synthetic` flag, and `record_sha256` over the
canonical JSON of everything else. Ten mutations are refused; deployment of a
synthetic record is refused, and deployment of any record needs authority the
lab does not hold.

**Compatibility and rollback** (`ciw.fpga-compatibility.v1`,
`ciw.fpga-rollback.v1`, T151): `compatible(b, h, r)` iff the host decoder
reads the bitstream's frame format, the board is supported and the host meets
the minimum version (15 of 36 combinations in the declared matrix). A rollback
must cite registered identities with matching digests, go backwards, give a
reason and match the matrix digest; execution is always refused here.
Rolling back to the previous version is not a safe default (for example
1.4.1 -> 1.2.0 on board revC).

**Link simulation** (T152): with Gilbert-Elliott loss, gamma jitter,
duplicates and a forced loss at the 2^32 wrap, serial-number arithmetic detects
all 91 losses between the first and last received frame, 577 reorderings and
12 duplicates; staleness detection with the true clock offset is exact
(381/381); an offset estimated from minimum delay (bias about 2 ms) misses
exactly the stale frames whose latency lies within the bias above the
threshold; a naive `seq - previous - 1` detector reports 1358 losses.

## Authorization boundary

- Actuator writes are denied by default (T153). Enabling needs an
  authorization record issued outside the workbench, unexpired, in scope,
  signed and verified against a trust anchor. The lab build has no trust
  anchor and no actuator transport and refuses to issue authorization
  records, so every route ends in a refusal with a specific code
  (`authorization_missing`, `authorization_wrong_type`,
  `self_issued_authority`, `authorization_expired`, `out_of_scope`,
  `unsigned_authorization`, `no_trust_anchor`).
- A frozen dataclass can still be mutated with `object.__setattr__`; the gate
  re-verifies the authorization on every write, so the mutation does not open
  it. In-process checks are defence in depth only; real enforcement belongs in
  hardware interlocks and a separate controller.
- Control outputs are `ControlProposal` objects whose status is fixed at
  `proposal` (T154). Converting one into a command is refused without separate
  authorization, and refused with any authorization the lab can see. The
  geometric proposal (heading change h = -j_lat(L) d / j_head(L) from the
  Jacobi transfer) matches -cot(L) d on the unit sphere, cancels a lateral
  offset to second order on the torus (residual order 2.02, uncorrected 1.00)
  and is refused at a conjugate point.
