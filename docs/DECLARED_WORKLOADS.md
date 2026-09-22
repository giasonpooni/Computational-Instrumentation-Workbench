# Schematics and numerical workloads in the shared bench

SRA and SCR now use the same retained source, operation, bundle, result,
execution and replay catalog as the process and sensor-fusion instruments.
They appear in the live Godot Workbench tab when execution commits a result.

| Workbench source / operation | Native owner and scope | ICRH profile |
| --- | --- | --- |
| `schematic-assessment` / `ciw.schematic-assessment.v1` | SRA declared typed graph, stale-certificate marking, eligibility and two-hop retrieval | `icrh.schematic-assessment.v1` |
| `numerical-heat` / `ciw.numerical-heat.v1` | SCR native integer Jacobi diffusion, fixed endpoints and explicit descriptor | `icrh.numerical-heat.v1` |

The source schemas are `ciw.<kind>-source.v1`; retained bundles use
`ciw.<kind>-session.v1`. Both retain exact input bytes, native output,
provider identity, distinct execution/result occurrences, and a second fresh
execution in a `ciw.declared-workload-verification.v1` record. This record
explicitly declares same-runtime reproduction and `independent: false`.
SET does not evaluate these two schemas. ICRH independently inspects their
bindings and, for SCR, checks the result with an integer arithmetic oracle.

## Bind and run

Use Python 3.12 and clean checkouts at these exact revisions:

| Role | Repository | Revision |
| --- | --- | --- |
| `sra` | Schematics-Retrieval-Agent | `a6e79585950bb6860e5edce5ebd2cce39ea481f2` |
| `scr` | Scientific-Computation-Runtime | `a59aba283b0304faeeb3e5d305087e7709e171ca` |

Build SCR's dependency-free Rust execution CLI outside its checkout. The gate
uses Rust 1.90.0. Local paths below are examples of trusted startup bindings.

```sh
cargo build --release --locked --offline \
  --manifest-path /path/to/providers/scr/crates/Cargo.toml \
  --target-dir /path/to/scr-build -p execution-cli

python -m ciw serve \
  --schematic-repo /path/to/providers/sra \
  --computation-repo /path/to/providers/scr \
  --computation-engine /path/to/scr-build/release/execution-cli \
  --output-dir results/shared-workbench
```

Add these three flags to a server started with `--calibrated-stack-root`,
`--identified-stack-root`, `--telemetry-stack-root` or
`--calibrated-window-stack-root` to operate them together. Keep each workflow's
existing provider pins; they are not interchangeable.

In another terminal:

```sh
python examples/declared-workloads/run.py --replay
godot --path godot
```

The example client submits both sources to the same server. Individual requests
use existing `source.add`, `operation.execute`, `bundle.replay`, `bundle.get`,
`result.get`, `execution.list` and `experiment.inspect`. Requests and saved
artifacts cannot select executable paths. Restoring a workspace preserves
inspection but leaves execution unbound.

SCR executes a private snapshot of the host-selected binary. Its SHA-256 and
byte count are retained and checked across replay, while the pinned Python
bridge checks Rust's raw-byte commitments. The binary-to-source relationship
is operator asserted, not a build attestation or proof of execution behavior.
A changed binary refuses replay. ICRH's oracle additionally checks this bounded
workload's actual numerical answer.

## Declared meanings

SRA accepts 1–256 explicitly typed nodes, at most 1,024 typed edges and 16
declared-node queries. It calls the native graph parser, stale-evidence marker,
eligibility evaluator and retrieval function. It does **not** invoke SRA's
companion execution loop. An `ELIGIBLE` decision is neither an executed JSPT/PLSR
call nor an OIT observability result. Supplied certificates remain unauthenticated
declarations; stale numerical evidence is retained and marked, not deleted.
No graph connectivity is invented. This is schematic topology, not an
algebraic-topology solver.

SCR accepts 3–256 signed integer cells with absolute values at most `2^40`, and
0–1,024 steps. It runs the exact native descriptor with empty configuration:

```text
u_next[i] = u[i] + trunc_toward_zero((u[i-1] - 2*u[i] + u[i+1]) / 4)
```

Each step reads the previous row and leaves both endpoints fixed. Values have
unit `1`; there is no physical temperature, spacing, duration or calibrated
thermal model in this operation. For `[0, 0, 64, 0, 0]`, two steps give
`[0, 16, 24, 16, 0]`. No covariance is invented.

`fusion.list` excludes both kinds. Their desktop projections carry
`fusion_context: null` and a typed `object_context`. Schematics have a node/edge
tree and eligibility context; numerical fields have separate initial/final
panels with null covariance. Neither enters ESM candidate capture or GSIE through
an implicit conversion. A future connection requires a specific compatible
model, observation mapping and uncertainty contract.

## Validation

```sh
python scripts/check_declared_workloads.py --output-dir results/declared-workloads
python scripts/check_godot.py --godot /path/to/godot
```

The first gate installs the built wheel in an isolated environment, builds and
executes the actual pinned SCR engine, executes SRA, and permits no skipped
provider tests. It checks shared-session inspection, fresh replay, unbound
restore, negative-integer rounding, fixed boundaries, stale evidence, invalid
graphs, executable drift, source/verification binding and exclusion from fusion
and ESM capture. It emits actual original/replay pairs for ICRH.

The next connections are explicit SRA companion calls using compatible provider
pins, selected-model handoff to existing JSPT/PLSR operations, and a supported
spatial view. Continuous acquisition scheduling, physical thermal modeling and
general symbolic/homology operations remain separate implementation work.
