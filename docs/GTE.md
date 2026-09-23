# Geometric Telemetry Engine: circle reference operation

[`ciw.geometric-circle.v1`](REMAINING_MODULES.md) connects this pinned native
operation to the common source catalog, Workbench desktop, result browser and
replay history. The direct commands below remain supported.

**Constraint-based geometric reconciliation and uncertainty propagation.**

GTE supplies one experimental **geometric reconciliation candidate** operation,
`gte.project-circle.v1`. CIW retains original observations, invokes the pinned
engine, and records the candidate, uncertainty and diagnostics in the existing
shared session. All projection and covariance calculations remain in GTE.

This reference slice supports a circle in a declared Euclidean 2D frame. It is
not an RTK/GNSS solver, Earth-coordinate transformation, dynamical estimator,
Lyapunov certificate or authorization to replace physical observations.

## Setup and pinned identity

Use Python 3.11 or newer and install CIW's declared environment:

```sh
python -m pip install -e '.[dev]'
git clone https://github.com/giasonpooni/Geometric-Telemetry-Engine.git ../gte
python -c "import json,subprocess; p=json.load(open('src/ciw/adapter-runtimes.json'))['gte']; subprocess.run(['git','-C','../gte','checkout','--detach',p['revision']],check=True)"
```

The full GTE commit is pinned in
[`adapter-runtimes.json`](../src/ciw/adapter-runtimes.json). The entry point is
`geodesic_telemetry.ciw_adapter` under GTE's `src/` directory. The generic adapter
version is `ciw-pinned-subprocess-v1`; every execution retains the Git commit and
tree, Python executable digest/version, and NumPy/SciPy versions. Source and
interpreter pins are checked before and after execution. Use clean checkouts;
untracked files outside permitted cache directories invalidate the pin.

The operator supplies `--gte-repo` and optionally `--python`. Saved files never
load code or authorize a runtime. GTE is not imported into CIW's Python process.

## Terminal workflow

```sh
python -m ciw geodesic create --inputs examples/adapters/circle.json --gte-repo ../gte --output-dir results/circle
python -m ciw geodesic inspect results/circle/workspace.json --json
python -m ciw geodesic replay results/circle/workspace.json --gte-repo ../gte --output-dir results/circle-replay --json
```

The fixture is synthetic and includes nonzero cross-sample covariance. Expect
retained observations, projected candidates, a separate reconciliation status,
signed residuals before projection, and `verification_status: "not_verified"`.
Replay appends fresh execution/result identities, preserves evidence and prior
results, and reports `replay_data_digest_matches: true` under the same runtime.
Inspection requires no GTE checkout and does not write to the source workspace.

The shared session supports the same path:

```sh
python -m ciw serve --workspace results/circle/workspace.json --gte-repo ../gte --output-dir results/circle-session
python -m ciw send operation.execute --payload '{"operation_id":"gte.project-circle.v1","parameters":{}}'
python -m ciw send operation.execute --payload '{"operation_id":"gte.project-circle.v1","parameters":{"policy":{"max_correction_m":0.001,"max_linearization_ratio":0.1}}}'
```

The final command holds the synthetic example's candidate while preserving the
same observation evidence, and records the stricter policy with its new result.

The operation always uses the **complete retained batch and full covariance**.
Its selected half-open interval must contain every retained sample; narrower
intervals produce a `selection_scope` refusal and no result. The playback cursor
and selected channel do not truncate the batch. Explicit whole-object `constraint` and `policy`
overrides are operation parameters; they create new execution/result identities
over the same retained observations. For example, a stricter correction policy
can hold a candidate without replacing or relabelling its evidence. Replay uses
the last invocation's exact parameters. Observation, covariance and timestamp
overrides are refused; import a new source for changed evidence. The current Godot client
declines external instrument render contracts; no GTE viewport is supplied.

## Input and output specifications

The authoritative exact schema and mathematics are in
[GTE's contract](https://github.com/giasonpooni/Geometric-Telemetry-Engine/blob/main/docs/CONTRACT.md).

| Contract | Delivered scope |
| --- | --- |
| Request | `gte.circle-request.v1`: observations, constraint and policy; 1–128 samples, at most 4 MiB of source JSON |
| Coordinates | Ordered `[x,y]` metre values, one explicitly named Euclidean plane frame; no conversion |
| Time | Nonnegative strictly increasing seconds relative to timezone-aware `time_origin`; no uniform cadence inferred |
| Source identity | Source ID and unique observation IDs retained in the exact original request |
| Input covariance | Full `2N × 2N`, sample-major `[x0,y0,x1,y1,…]`, `m^2`; cross-sample terms preserved |
| Constraint | Versioned circle, fixed exact center/radius, matching frame, half-open validity at acquisition |
| Policy | Maximum correction in metres and maximum dimensionless linearization ratio |
| Result | `gte.circle-result.v1`; observed/projected/reconciled coordinates, diagnostics, uncertainty, arc diagnostics, reconciliation and self checks |
| Primary uncertainty | Joint `N × N` first-order local tangent arc-length covariance with tangent bases, reference points and ordering |
| Ambient uncertainty | Separate rank-deficient `2N × 2N` propagated covariance; not a full-rank posterior |
| Evidence | Exact UTF-8 request bytes in base64, SHA-256 byte commitment, parsed request and corresponding immutable raw channels |
| Persistence | `run.v1`, `ciw.execution.v1`, `ciw.operation-result.v1`, workspace version 2; existing protocol version 1 |

The run's selection support ends one second after its last timestamp. This is
only a half-open interval boundary; it declares neither an acquisition cadence
nor a physical sample beyond the last observation.

CIW's existing operation-role enum records `backend` for this bounded geometric
computation. The instrument manifest declares `operation_provider`. Neither
label introduces state-estimator, Bayesian-inference or dynamical-model semantics.

## Status and limits

| Outcome | Meaning |
| --- | --- |
| `completed` with reconciliation `eligible` | Candidate passed the two declared policy limits; no physical validation established |
| `completed` with reconciliation `held` | Candidate and diagnostics retained; `reconciled_points_m` is null and observations remain unchanged |
| `refused` | Invalid/unsupported domain input, singular projection or stale constraint; execution retained and no new result |
| Runtime/configuration error | Source/interpreter pin or transport failure; not a geometric result |

CIW CLI exit `0` means successful command processing, including a retained,
inspected or replayed domain refusal; inspect `status` and `refusals` to determine
the scientific outcome. Exit `2` means input/configuration/runtime-binding or
replay-integrity failure. This preserves the existing CIW investigation behavior.
The GTE standalone CLI uses exit `2` for domain refusal, while its subprocess
protocol returns a normal JSON refusal envelope at exit `0`. Numerical self checks
and matching replay digests never confer verification. A stale constraint is
evaluated against acquisition timestamps, so historical replay does not expire
because wall-clock time has passed.

Projection can satisfy the circle by construction while preserving a large
original mismatch. Original radial residuals and correction magnitudes remain
visible. Tangent covariance is a first-order transformation conditional on exact
geometry, not a calibrated posterior or a claim of improved sensor accuracy.
Shortest adjacent arcs do not establish winding count, motion, velocity or the
actual physical path. Antipodal ambiguity is explicit. No uncertain geometry,
streaming telemetry, automatic frame conversion, hardware backend, or physical
validation is delivered by this integration.

GTE's native covariance arrays are not selectable by the shared
`covariance-artifact.v1` / JSPT workflow. They retain tangent bases, reference
points, source observation IDs, ordering, frame and acquisition time in the
native result. The ambient covariance remains a separate singular diagnostic;
neither representation is a calibrated posterior.

## Validation

```sh
CIW_GTE_REPO=../gte python -m pytest -q tests/test_geodesic.py
python -m pytest -q
```

On PowerShell set `$env:CIW_GTE_REPO = '../gte'` before invoking pytest. Tests
exercise the real child process, exact request retention, full covariance
transport, separation of held/refused outcomes, read-only reopening, fresh
replay identities, shared-session execution and tamper rejection. GTE's own
suite verifies numerical propagation; CIW does not reimplement it.
`scripts/check_adapters.py` clones all declared pins and runs the integration
gates in CI on Linux and Windows. External-engine gates are explicitly skipped
when their checkout environment variable is absent.

Validation recorded on 2026-09-20 used Python 3.12.14, NumPy 2.4.3 and pytest
9.0.2. After integrating the concurrently published workbench covariance
foundation (`258d526b2500af95a5870d9df1a851a906e81cb7`), the complete suite passed
**364 tests and 38 subtests**, including all **20 GTE integration tests** against
the published GTE commit
[`e55b8be2b3ba05f7e6c6a31807c77b3e42700f07`](https://github.com/giasonpooni/Geometric-Telemetry-Engine/tree/e55b8be2b3ba05f7e6c6a31807c77b3e42700f07).
**39 optional-instrument tests were skipped** because the RCI/FSRT/JSPT source
checkouts and PLSR runtime were absent. The existing covariance commands,
current/historical runtime pins and validators were preserved during integration.

The GTE gates include type-sensitive source bindings, mixed-scale covariance
rejection, policy override replay, narrowed-selection refusal, and successful
CLI inspection and replay of retained domain refusals. The published GTE source
also passed its
[upstream CI run](https://github.com/giasonpooni/Geometric-Telemetry-Engine/actions/runs/35537097059).

These are computational and integration checks. No hardware, physical
calibration, positioning accuracy or independent proof verification was tested.
