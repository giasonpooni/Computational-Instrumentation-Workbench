# Development guide

The [proved heat guide](PROVED_HEAT.md) gives the exact Linux build and
`scripts/check_proved_heat.py` acceptance command for SCR/SP1. The native gate
requires a real original proof, fresh proved replay, retained-proof reverification
and corrupted-proof rejection from an installed CIW wheel; skipped tests fail it.
On a provisioned Linux host, `scripts/run_proved_heat_locally.py` runs the gate
steps of `.github/workflows/proved-heat.yml` unchanged and refuses when a
provisioning result is missing; `ciw lab proved-heat retain` keeps a passing
run under `lab/proved-heat/` for the lab queue (T099; see
[LAB.md](LAB.md#proved-heat-gate-records)).
`tests/test_proved_heat.py` separately exercises structural/refusal boundaries
with explicit test doubles. Such tests never count as cryptographic evidence.
The [Julia and SP1 contract](JULIA_SP1.md) records the later persistent-worker,
oscillator, F2 checker and benchmark requirements.

Public documentation describes implemented behavior, executable contracts,
reproducible examples and measured limitations. The [architecture](ARCHITECTURE.md)
identifies current components; [PROTOCOL.md](PROTOCOL.md) specifies exact record
and transport semantics.

## Contribution requirements

- Preserve numerical engines and their independent reference checks. Python
  owns scientific records/calculations; Godot owns rendering and interaction.
- Keep terminal-only operation functional. Disconnecting a viewport must not
  stop the backend service.
- Preserve separate evidence, operation, execution, result and verification
  identities. Displayed geometry, numerical checks and replay equality do not
  confer verification.
- Compute from full-resolution records. Keep missing observations explicit.
- Keep playback cursors separate from half-open analysis intervals; reject
  stale selection revisions and record an operation's actual support.
- Validate saved records without executing numerical providers. Runtime
  bindings are supplied explicitly and never loaded from workspace content.
- The operation runner copies provider runtime identities before execution and
  returned payloads before retention. Reusing or mutating provider-owned objects
  must not alter retained records. Invalid runtime identities refuse before the
  provider executes, retaining a restorable refusal with `runtime: null`.
- Preserve covariance ordering, frames, units, source identities and declared
  assumptions. Do not silently repair, diagonalize or reinterpret a matrix.
- Use isolated branches/checkouts and review current remote changes before
  integration. Never overwrite another contributor's work or force-push shared
  history. Separate documentation and executable changes where practical.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/ciw/core/` | Record, identity and covariance contracts |
| `src/ciw/adapters/` | Adapter bindings and trusted saved-payload validators, including RCI provenance and covariance dependency checks |
| `src/ciw/operations/` | Operation registry, execution records and schema dispatch |
| `src/ciw/session.py`, `server.py`, `cli.py` | Shared state, transport and terminal commands |
| `src/ciw/covariance_workflow.py`, `investigation.py`, `geodesic.py`, `plsr.py` | Implemented scientific workflows, including covariance execution and replay |
| `src/ciw/calibration_status.py` | Serving-time applicability at an explicit `evaluated_at`, separate from evidence |
| `examples/` | Runnable inputs for supported integrations |
| `tests/`, `scripts/` | Unit, integration and deployment checks |
| `godot/` | Optional viewport and its headless checks |
| `deploy/` | Native and container operation guides |

`recordings/`, `results/` and `.ciw/` hold local runtime output and are ignored by
Git. Scientific source pins live in the `src/ciw/*-runtimes.json` manifests,
`src/ciw/plsr-runtime.json`, and the explicit `PIN`/`PINS` declarations in native
workflow modules linked by their operating guides. Update pins only with the
relevant compatibility and replay checks. The covariance examples in `examples/adapters/` include
`two-reservoir-covariance.json` and `tank-covariance-map.json`; the original
`two-reservoir.json` fixture remains available.

## Validation commands

```sh
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/check_adapters.py
```

The adapter script clones the explicitly pinned current and historical
scientific sources. Source-dependent tests skip when their documented checkout
variables are absent; report those skips. The optional PLSR environment and
installed-package checks are described in [PLSR.md](PLSR.md).

The three [geometry provider profiles](GEOMETRY_RESEARCH.md) have an installed
CIW wheel gate on Python 3.12 with exact clean public provider pins:

```sh
python scripts/check_geometry_research.py --output-dir results/geometry-gate
```

The [variational free-energy bench](VARIATIONAL_FREE_ENERGY.md) has a separate
installed-wheel gate on Python 3.12, Linux and Windows. It provisions exact CSG,
GSIE and PLSR revisions, exercises six retained simulation cases, checks offline
records and fresh replay, and runs the live shared-session client. Every native
test must run without skips:

```sh
python scripts/check_free_energy.py --output-dir results/free-energy-gate
```

This gate requires all native analytical, retained-contract, shared-session and
replay tests to complete without skips, on Linux and Windows. It retains the
three original/replay artifact pairs and a reopenable workspace. Its scope is
mathematical reference behavior and record consistency; independent ICRH
profiles and physical calibration remain separate work.

The installed candidate gate requires Python 3.12, Git, Node.js 22.13 and `npm`
on `PATH`, plus network access for the pinned providers and packages:

```powershell
python -m pip install "setuptools>=77" wheel numpy==2.4.3 websockets==16.0
python scripts/check_workbench_candidates.py
```

The gate resolves the Windows `npm.cmd` executable and passes `core.autocrlf=false`
to its Git commands and nested helpers so pinned source bytes are preserved.
It does not change global Git configuration. Reused checkouts must already match
their pinned bytes; the gate does not repair them. Skipped integration tests
fail this gate. The pinned ESM fixture helper still invokes `npx` without resolving
the Windows command shim, so the complete candidate gate currently runs in Ubuntu
CI; the Windows helper failure is a test-tooling limitation.

For viewport changes run `python scripts/check_godot.py --godot <Godot 4.5.2 executable>`
(import, `godot/tests/protocol_smoke.gd`, `channel_generality.gd`, `adapter_boundary.gd`);
[godot/README.md](../godot/README.md) documents the individual commands. Deployment
changes use the checks in [deploy/README.md](../deploy/README.md). Record unavailable
checks honestly.
Covariance validation includes `tests/test_covariance_artifacts.py` for the
artifact contract, `tests/test_covariance_records.py` for saved dependencies and
cycle refusal, and `tests/test_covariance_integration.py` for the pinned v2 path
and historical replay. `tests/test_calibration_status.py` checks live read
surfaces, `tests/test_rci_records.py` checks retained measurement provenance,
and `tests/test_covariance_replay_refusal.py` checks refusal when no runtime was
bound to the original attempt. Historical replay requires the legacy checkouts.
`tests/test_operation_runner.py::test_provider_owned_outputs_cannot_mutate_retained_records`
checks provider-object isolation for successful and refused executions, and
`test_invalid_runtime_refusal_remains_restorable` checks that invalid runtime
identities prevent execution while preserving save/restore.
Coverage is not exhaustive: the off-allowlist `runtime_mismatch`,
`RUNTIME_UNAVAILABLE` and `RUNTIME_IO` branches, covariance CLI argument parsing,
and the calibration-refusal exit code lack dedicated assertions in the current
suite. Passing shared helper tests does not separately validate those paths.
[RECONCILIATION.md](RECONCILIATION.md) records, per contract area, the behavior
the current tests assert, what they cover in part, and what is not implemented.

Documentation-only edits require working links and consistency with the
implemented interfaces; they do not imply a new numerical validation result.

The native geodesic-reference gate is `python scripts/check_geodesic_references.py`.
It runs both pinned providers from an installed CIW wheel, exercises the live
session and provider-free restore, rejects skips, then independently checks the
retained pairs with pinned ICRH profiles. See [the operating contract](GEODESIC_REFERENCES.md)
for local-checkout arguments and numerical scope. Existing authenticated local
checkouts can also provision older integration gates; [provider availability](PROVIDER_AVAILABILITY.md)
documents their options and remaining public-source assumptions.

The [computational experimentalist queue](LAB.md) has its own tests
(`tests/test_lab_*.py`) and a clean-room gate, `scripts/check_lab.py`. The gate
provisions the pinned CSG, FTR and SCR checkouts, the exchange workflow's
SET, PPDA and SCR checkouts and the telemetry stack of
`src/ciw/telemetry-runtimes.json` (cloned, or clean checkouts named `csg`,
`ftr`, `scr`, `set`, `ppda` and `scr-exchange` under `--stack-root`, beside a
`telemetry-stack` directory holding the telemetry checkouts under their
repository names) and runs
`scripts/reproduce_lab.py` with them: it builds a wheel with pip build
isolation, installs it into a new virtual environment, runs the lab tests with
a JUnit record (the provider-gated tests against the bound providers), runs
every queue task and compares the fresh reports with the retained ones in
`lab/` within each finding's declared tolerance. The retained run also binds
the Python 3.12 PLSR and FTR interpreter, so reproducing it needs Python 3.12+,
Git and network access for the pinned providers and packages; on Python 3.11
the gate refuses to compare and runs only with `--no-compare`. Until `lab/`
holds a retained run, the comparison refuses and `--no-compare` is required.
NumPy's OpenBLAS picks its kernels for the CPU, and kernels round differently;
`--blas-core` runs the clean room on another OpenBLAS kernel, so the retained
run can be verified on the kernels other hosts pick, and `gate.json` records
the kernel the clean room ran (see [the lab guide](LAB.md)).

```sh
python -m pip install -e '.[dev,lab]'
python -m pytest -q tests/test_lab_core.py
python scripts/check_lab.py --output-dir results/lab-gate     # Python 3.12+
python scripts/check_figures.py --output-dir results/figures  # re-execute every retained figure task
python scripts/check_lab.py --blas-core Haswell --output-dir results/lab-gate-haswell   # also Sandybridge
```

`scripts/check_figures.py` re-executes every figure task of the retained run
with the installed `ciw` and compares each SVG figure with `lab/` byte for
byte (a figure declared as a wall-clock timing figure by structure only, one
declared as a rounding-level figure by its recorded values within their
rounding bounds); it records the platform, Python, NumPy, BLAS build and
OpenBLAS kernel in `figure-check.json`, lists tasks whose providers are not
bound (`--provider ROLE=PATH`) as not re-executed, and exits 3 on a mismatch.
Run on Windows, it is the second-platform figure comparison; run with
`OPENBLAS_CORETYPE` set, as CI's `lab-blas-kernels` job does, it compares the
figures on another kernel (see [LAB.md](LAB.md#retained-evidence)).

New lab tasks follow the [authoring contract](lab/AUTHORING.md). Regenerate
the retained evidence only together with the code change that alters it, and
only from a clean-room gate run under Python 3.12+ with every provider bound:
`python scripts/refresh_lab.py --stack-root <checkouts>`, or
`python scripts/refresh_lab.py --from-run <check_lab.py output>`. Never write
`ciw lab run` output into `lab/`: outside the clean room T164 is partial, the
provider tasks differ and the run log would be retained. Review `git diff lab`
and the `ciw lab verify` differences.

The gate also binds the latest retained SP1 proved-heat gate record
(`lab/proved-heat/<run-id>/`) for T099, which rebuilds `execution-cli` with
the Rust toolchain the proved-heat workflow pins; `lab.yml` installs it with
`rustup toolchain install 1.94.0 --profile minimal`, and a local reproduction
needs the same toolchain for T099 to match the retained report. To add a
record, run the proved-heat gate's steps on a provisioned Linux host and
retain the output:

```sh
python scripts/run_proved_heat_locally.py --scr <clean SCR checkout> --sp1 <fresh SP1 clone> \
    --compiler-archive <succinct-1.94.0 archive> --python <Python 3.12+ with setuptools 77+ and wheel>
ciw lab proved-heat retain results/proved-heat --retained lab --run-id local-<date> --host "<host description>"
ciw lab proved-heat verify --retained lab
```

It binds the latest second-platform figure record
(`lab/figure-platforms/<record-id>/`) for T158 as well. To add one, let
`figures.yml` run `scripts/check_figures.py` on Windows against the pushed
`lab/`, download its `figure-check-windows` artifact as a zip and retain it
with the run's provenance; the zip is refused unless it hashes to the
artifact digest GitHub reports (see
[LAB.md](LAB.md#second-platform-figure-records)):

```sh
gh api repos/OWNER/REPO/actions/artifacts/ID/zip > figure-check-windows.zip
python scripts/retain_figure_check.py figure-check-windows.zip --retained lab --repository OWNER/REPO \
    --workflow .github/workflows/figures.yml --run-id RUN --run-attempt 1 --head-sha SHA \
    --artifact-id ID --artifact-name figure-check-windows --artifact-digest sha256:HEX
ciw lab figure-platform verify --retained lab
```

## Documenting an integrated tool

Update the README catalogue and [INSTRUMENTS.md](INSTRUMENTS.md) when a supported
workbench connection changes. Include:

- Purpose, actual integration status and scientific scope.
- Tool/adapter identity, schema versions and a reproducible full source pin.
- Installation, explicit input files and copyable terminal commands.
- Input/output shapes, ordering, units, frames and time conventions.
- Success, refusal and error meanings; retained evidence and runtime provenance.
- A worked end-to-end example, validation commands and relevant tests.
- Implemented views, save/reopen/replay behavior and known limitations.

List a tool as integrated only after its documented workbench path has been
exercised with representative inputs and its output/provenance checked.
Installation or a standalone schema alone does not establish integration.

## Names and deployment namespace

The distribution is `computational-instrumentation-workbench`; the console
command is `ciw`. `compose.yaml` declares no project name, so Compose uses the
checkout directory's name, and builds the image as `ciw-backend:local`; the
container check runs under a temporary `ciw-check-<id>` project. Presentation
names do not change scientific record identities. Changing a Compose project
name changes named-volume prefixes; consult the deployment guide first.
