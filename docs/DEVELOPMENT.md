# Development guide

The [proved heat guide](PROVED_HEAT.md) gives the exact Linux build and
`scripts/check_proved_heat.py` acceptance command for SCR/SP1. The native gate
requires a real original proof, fresh proved replay, retained-proof reverification
and corrupted-proof rejection from an installed CIW wheel; skipped tests fail it.
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
| `src/ciw/reference_workflow.py` | Shared lifecycle of the provider-free Python reference operations |
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
python -m ciw workspace verify tests/fixtures/retained/workbench.json
```

`ciw workspace verify` reopens a saved workspace, or a retained workbench
file, offline: no provider, no process, no write outside a temporary
directory. It reports whether every retained source and bundle validates and,
per bundle, whether this host could replay it. A provider-free reference
compares its current runtime identity with the retained one and names the
differing fields (`algorithm.kernel_probe` on a host whose linear-algebra
kernels round differently); a provider-backed kind is reported as needing a
repository binding. The exit status is 0 when the file is valid, 1 when a
retained item is refused (the report names it) and 2 when the file is not a
workspace at all. A file larger than 256 MiB, which no workspace this package
writes approaches, is refused by its size before any of it is read.

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
The provider-free typed contract foundations are exercised by
`tests/test_machine_manifest.py`, `tests/test_project_model.py` and
`tests/test_thermal_contract.py`; these cover evidence-bound machine binding,
project-history invalidation and independent thermal-reference replay. The
machine, thermal, project graph and uncertainty validation operation lifecycles
are covered by `tests/test_machine_workflow.py`, `tests/test_thermal_workflow.py`,
`tests/test_project_workflow.py` and `tests/test_uncertainty_validation.py`,
including save/reopen, fresh replay and refusal cases; the chi-square and
binomial bands behind the last are checked against reference values in
`tests/test_consistency_math.py`. Independent ICRH profiles for these provider-free operations remain the
next integration gate; see [contract foundations](CONTRACT_FOUNDATIONS.md).
`tests/test_retained_compatibility.py` is the reopen gate for identity-critical
changes. It reopens `tests/fixtures/retained/workbench.json`, a committed
snapshot of one executed bundle per provider-free kind built from the committed
example inputs, validates every retained bundle against the current reference
without a provider, executes each retained source afresh and requires the same
numbers (bit for bit for machine manifest and project graph, within the shared
binary64 tolerance for energy, thermal and uncertainty validation, whose
linear-algebra kernels differ between hosts), and replays each bundle. Three
replay outcomes are accepted: a fresh occurrence with the same numerical
identity; a refusal because the reference runtime identity changed, which on
another host names `algorithm.kernel_probe`; or, should the probe ever fail to
separate two hosts, a refusal as a numerical mismatch for a kernel-sensitive
kind that leaves the retained bundle valid. The snapshot was made on a host
whose OpenBLAS kernels are the SkylakeX set; `OPENBLAS_CORETYPE=Haswell` or
`Prescott` reproduces the refusal locally. Regenerate the snapshot with
`python tests/fixtures/retained/generate.py` only when the retained format
changes on purpose, and say so in the commit; a silent regeneration hides
exactly the incompatibility the gate exists to catch.
`python scripts/check_reference_mutants.py` is the mutation gate for the same
module and for the retained workbench: it removes one check at a time and
requires the guarding tests to fail for every one. The `reference` target
removes identity checks of `src/ciw/reference_workflow.py` (evidence
binding, request and numerical commitments, bundle digest, session identity
shape, code digest shape, fresh-occurrence and self-referencing receipt
rules, runtime identity comparison, tolerance and kernel probe), guarded by
`tests/test_reference_identity_checks.py`. The `workbench` target removes
record checks of `src/ciw/workbench.py` (source kind and byte binding,
bundle identity, verification presence, receipt count, receipt digest and
identity, restored source content, duplicate bundles, identity collisions
and the upstream requirement), guarded by
`tests/test_workbench_record_checks.py`; several of these repeat checks the
reference workflows make themselves, so the guarding cases exercise the
workbench functions directly, with a stub workflow where the workbench check
is the only guard for a provider-backed kind. A surviving mutant means a
check has lost its test.
`python scripts/check_validator_mutants.py` extends the same discipline to
the source and result validators of the provider-free references
(`energy_records`, `thermal_contract`, `uncertainty_validation`,
`consistency_math`, `machine_workflow`, `project_workflow`). It derives the
mutants from the code rather than from a hand-written list: every
`if <condition>: raise` becomes `if False: raise`, one `or` clause at a time,
and each module's `tests/test_*_checks.py` must fail for every one. Those
tests are table driven: each case starts from the retained example, changes
exactly one thing and expects the refusal that names it. The gate runs the
mutants in parallel git worktrees that carry the live tree's copy of the
module and its tests, so it never edits the working tree. A few clauses are
listed in the script as equivalent (redundant with a neighbouring clause for
every input the module can receive, such as a finiteness test beside a range
comparison that NaN also fails); their survival is reported, not counted.
Covariance CLI argument parsing and the pre-execution refusal of non-object
parameters are asserted by
`tests/test_adapter_cli.py::test_covariance_verbs_parse_and_refuse_non_object_parameters_before_execution`;
the serving-time calibration refusal and its exit status through the `ciw`
process by
`tests/test_workbench_cli.py::test_send_reports_calibration_time_refusals_through_the_exit_status`.
The covariance and investigation replay-mismatch exit paths still need the
pinned providers and are exercised only by the provider gates.
The off-allowlist `runtime_mismatch` refusal is asserted by
`tests/test_runtime_allowlist.py` without binding an adapter; the
`RUNTIME_UNAVAILABLE` and `RUNTIME_IO` adapter refusals by
`tests/test_subprocess_adapter.py`, where the shell-script interpreter and
orphaned-pipe cases run on POSIX only; a version-1 workspace carrying executions
by `tests/test_operation_runner.py`; and the 8 MiB transport frame limit of the
served process by `tests/test_deployment.py`. Passing shared helper tests does
not separately validate other paths.
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
