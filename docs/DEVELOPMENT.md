# Development guide

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
Git. Scientific source pins live in `src/ciw/adapter-runtimes.json` and
`src/ciw/plsr-runtime.json`; update pins only with the relevant compatibility and
replay checks. The covariance examples in `examples/adapters/` include
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

For viewport changes run the headless Godot import and bridge checks documented
in [godot/README.md](../godot/README.md). Deployment changes use the checks in
[deploy/README.md](../deploy/README.md). Record unavailable checks honestly.
Covariance validation includes `tests/test_covariance_artifacts.py` for the
artifact contract, `tests/test_covariance_records.py` for saved dependencies and
cycle refusal, and `tests/test_covariance_integration.py` for the pinned v2 path
and historical replay. `tests/test_calibration_status.py` checks live read
surfaces, `tests/test_rci_records.py` checks retained measurement provenance,
and `tests/test_covariance_replay_refusal.py` checks refusal when no runtime was
bound to the original attempt. Historical replay requires the legacy checkouts.
Coverage is not exhaustive: the off-allowlist `runtime_mismatch`,
`RUNTIME_UNAVAILABLE` and `RUNTIME_IO` branches, covariance CLI argument parsing,
and the calibration-refusal exit code lack dedicated assertions in the current
suite. Passing shared helper tests does not separately validate those paths.

Documentation-only edits require working links and consistency with the
implemented interfaces; they do not imply a new numerical validation result.

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
command is `ciw`. The Compose project slug is `notation-systems-ciw` and the
image repository is `notation-systems/ciw-backend`. Presentation names do not
change scientific record identities. Consult the deployment guide before
changing a Compose project name because it changes named-volume prefixes.
