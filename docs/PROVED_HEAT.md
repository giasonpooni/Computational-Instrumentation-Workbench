# Registered heat computation and SP1 verification

`ciw.proved-heat.v1` connects SCR's existing registered integer heat guest to
the shared workbench. It is the first increment of the
[Julia and SP1 development contract](JULIA_SP1.md). Julia execution and topology
are separate planned operations; this guest does not execute Julia.

The operation runs SCR's native heat kernel, produces a real SP1 CPU proof,
checks guest/native input and output commitments, and invokes a separate
full-ELF verifier. The result enters the retained catalogue only after those
checks succeed. A failed proof produces a refusal and no successful result.

## Exact computational claim

The registered guest executes the declared integer Jacobi update, reading the
retained input bytes and producing the retained output commitment with exit
code zero. Empty native configuration is mandatory. The statement is bound
to the registered guest ELF and program descriptor; source/ELF semantics and
proof-system soundness remain assumptions of that computational claim.

The workbench bounds this profile to 3–32 cells, 0–64 steps, and integer values
with absolute magnitude at most `2^40`. Both endpoints remain fixed:

```text
next[i] = current[i] + trunc_toward_zero(
    (current[i-1] - 2*current[i] + current[i+1]) / 4
)
```

Values have unit `1`. For `[0,100,200,100,0]` and four steps, the final field is
`[0,65,92,65,0]`. This is no assertion of physical temperature, calibration,
uncertainty, a continuous PDE solution, or admission to a state estimator.
The source's `proof_policy: required_before_result` is workbench policy; the
guest executes the fixed registered algorithm over its input. Descriptor/ELF
binding is checked through host registration and the reproducible build recipe;
the guest does not read an arbitrary policy document.

## Runtime and build pins

| Component | Required identity |
| --- | --- |
| SCR source | `a59aba283b0304faeeb3e5d305087e7709e171ca` |
| SP1 source | `b38b61209e45e969289e70d5cf79dc763460bc41` in [upstream SP1](https://github.com/succinctlabs/sp1/tree/b38b61209e45e969289e70d5cf79dc763460bc41) |
| Backend report | `sp1-cpu v6.1.0` |
| Guest recipe | SCR `zk/recipes/sp1-heat.recipe`, identity `6e5d1687bcc55243d712553a2b7768b6c587a76418bb48a7a2c44224470d423d` |
| Guest ELF SHA-256 | `a14e3750da7e221d31842bd6cf983fcc8c0f530b2811537e2a9a9fe803dacf82` |
| Guest compiler | `succinct-1.94.0-64bit`, Linux x86-64, exact archive SHA-256 in the workflow |

The [Linux workflow](../.github/workflows/proved-heat.yml) provisions the exact
compiler, checks the committed recipe through `execution.build.verify_build`,
builds `execution-cli` and `sp1-host`, and runs the installed-wheel gate. It
uses the publicly available upstream commit at the same identity as SCR's
recorded fork. It never rewrites the guest registry to accept a different build.
Host binaries are operator-bound, hashed and snapshotted; their relationship
to source is not a cryptographic build attestation.

The recipe requires Linux. Native Windows CIW can inspect saved records;
local proof execution needs a suitable Linux environment or a separately
provisioned compatible backend. Missing artifacts are unavailable, not passed.

## Start the shared operation

Use Python 3.12 and explicit trusted paths to the pinned checkout and built
artifacts. Keep build outputs outside the immutable SCR checkout.

```sh
python -m ciw serve \
  --computation-repo /path/to/scr \
  --computation-engine /path/to/native-target/release/execution-cli \
  --sp1-prover /path/to/proof-target/release/sp1-host \
  --sp1-heat-guest /path/to/artifacts/sp1-heat.elf \
  --output-dir results/proved-heat-session

# In a second terminal; --replay produces another real proof.
python examples/proved-heat/run.py --replay
```

The [source example](../examples/proved-heat/source.json) uses
`ciw.proved-heat-source.v1`. Existing `source.add`, `operation.execute`,
`bundle.get`, `bundle.replay`, `experiment.inspect`, and `instrument.inspect`
carry this operation. Its instrument role is `scr`; `fusion_context` is null.
The view displays initial/final integer fields, proof identity and scope,
runtime records and stage durations. Proof bytes remain in the bundle, not the
plot. `ciw send` supports `--timeout 3600` for long operations.

Provider work runs outside the socket event loop and shared publication lock,
so other clients can inspect and change playback while it runs. The submitting
connection waits for completion. This is not a persistent job queue: service
restart recovery, cancellation and proof scheduling remain follow-on work.
This version retains the source on refusal but does not retain a failed-workflow
execution record. Successful results never silently downgrade to unproved ones.

## Retained evidence and fresh verification

Bundles use `ciw.proved-heat-session.v1`. They retain exact source and proof
bytes, native specification/input/output bytes and commitments, separate
execution/result/verification identities, and runtime hashes. Proof bytes are
limited to 8 MiB; bundles to 24 MiB. Replay produces new occurrences and another
proof; numerical identity excludes proof bytes and elapsed times.

Opening a workspace checks content and linkage without executing a verifier.
The saved outcome is a historical runtime report, explicitly labeled
`retained_runtime_report_requires_fresh_verification`. Recomputed JSON hashes
cannot authenticate a fabricated proof or report. Inspection grants no
cryptographic or state-admission authority.

Export the native bundle returned by `bundle.get` as JSON, then reverify its
retained proof without producing a new proof:

```sh
python -m ciw proof verify results/original.json \
  --computation-repo /path/to/scr \
  --computation-engine /path/to/native-target/release/execution-cli \
  --sp1-prover /path/to/proof-target/release/sp1-host \
  --sp1-heat-guest /path/to/artifacts/sp1-heat.elf \
  --output results/fresh-verification.json
```

This binds a trusted verifier host with the same pinned SCR source, registered
guest and backend contract, derives its key from that ELF, and writes a separately
identified report linked to the original bundle. Its actual `verifier_runtimes`
and `verifier_runtime_digest` are recorded separately from the producer's runtime;
another compatible host can verify the proof. Replay still requires the original
runtime identities. Existing output files are refused. No supplied verification-key artifact
is accepted. A successful command exits zero; malformed source, changed pins,
missing artifacts, mismatched commitments and failed verification exit two.

## Verification and limits

```sh
python scripts/check_proved_heat.py \
  --scr-repo /path/to/scr \
  --sp1-repo /path/to/notationsystems/SP1-zero-knowledge-virtual-machine \
  --engine /path/to/native-target/release/execution-cli \
  --prover /path/to/proof-target/release/sp1-host \
  --guest /path/to/artifacts/sp1-heat.elf \
  --output-dir results/proved-heat-gate
```

The gate installs CIW into an isolated environment and requires actual proof,
fresh replay, offline restore, retained-proof verification, and cryptographic
rejection of a corrupted proof whose surrounding JSON commitments were resealed.
Missing providers and skipped tests fail this gate. Ordinary local tests use
explicit synthetic proof bytes only for structural/refusal tests; those fixtures
are not cryptographic evidence. ICRH has no separate proved-heat profile yet.

`scripts/run_proved_heat_locally.py` runs the workflow's gate steps, read from
the workflow file and unchanged, on a Linux host the operator provisioned
(tools, the pinned Rust and Succinct toolchains, the compiler archive and clean
SCR and SP1 checkouts, each checked first), and writes `results/proved-heat/`
as the workflow does. `ciw lab proved-heat retain` keeps a passing run under
`lab/proved-heat/` with a run description and a digest manifest; the lab
queue's T099 reads it ([LAB.md](LAB.md#proved-heat-gate-records)). A retained
run is the gate's outcome as recorded on that host, not a fresh verification.

Stage durations distinguish native execution, SCR's combined prove-and-verify
call, and additional verification. On Linux, `memory` records the largest waited
child process's peak RSS in bytes (`scope: max_waited_child_peak_rss`), measured
after the final verifier exits. It is not the aggregate process-tree peak or a
prover-only measurement. Other platforms or unavailable measurements retain
`status: not_measured` and `bytes: null`. This follows the Linux
[`getrusage` definition](https://man7.org/linux/man-pages/man2/getrusage.2.html).
Runner memory availability is recorded separately. Isolated proving time and
aggregate peak memory still need characterization before increasing workload bounds.
