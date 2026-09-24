# Covariance provenance, propagation and calibration status

The v2 investigation extends the existing RCI → CIW → FSRT path with explicit
covariance artifacts and a JSPT operation provider. Scientific ownership stays
with the domain repositories: RCI calibrates, FSRT estimates and reconciles,
JSPT propagates covariance, and CIW retains and validates their records.

The runnable fixture remains two synthetic mass measurements at one simultaneous
instant. It is a computational integration example, not a physical calibration
certificate or a qualified measurement system.

## Reproduce the full path

Use Python 3.12+ with CIW's development dependencies installed. Prepare clean
RCI, FSRT and JSPT checkouts at the current full revisions recorded in
the terminal provider descriptors ([`rci`](../src/ciw/pipelines/providers/rci.json), [`fsrt`](../src/ciw/pipelines/providers/fsrt.json), [`jspt`](../src/ciw/pipelines/providers/jspt.json), [`gte`](../src/ciw/pipelines/providers/gte.json)). For fresh sibling
checkouts, this script uses those exact pins:

```sh
python -m pip install -e '.[dev]'
python - <<'PY'
import json
from pathlib import Path
import subprocess

pins = {r: json.loads(Path(f'src/ciw/pipelines/providers/{r}.json').read_text())['pin'] for r in ('rci', 'fsrt', 'jspt', 'gte')}
for name in ('rci', 'fsrt', 'jspt'):
    destination = Path('..') / name
    spec = pins[name]
    subprocess.run(['git', 'clone', '--config', 'core.autocrlf=false', spec['repository'], str(destination)], check=True)
    subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', spec['revision']], check=True)
PY
```

From the CIW repository root:

```sh
python -m ciw investigation create --inputs examples/adapters/two-reservoir-covariance.json --rci-repo ../rci --fsrt-repo ../fsrt --output-dir results/covariance-source
python -m ciw covariance results/covariance-source/workspace.json --jspt-repo ../jspt --parameters examples/adapters/tank-covariance-map.json --output-dir results/covariance-map
python -m ciw investigation inspect results/covariance-map/workspace.json --evaluated-at 2026-09-20T12:00:00Z
python -m ciw investigation inspect results/covariance-map/workspace.json --evaluated-at 2026-09-20T12:00:00Z --json
python -m ciw covariance-replay results/covariance-map/workspace.json --jspt-repo ../jspt --output-dir results/covariance-map-replay
```

`create` invokes `rci.calibrate.v2` and `fsrt.tank-reconstruct.v2` for the v2
fixture. `covariance` attaches `jspt.covariance-propagate.v1` to the reopened
investigation and saves its result alongside the original FSRT result.
`covariance-replay` reuses the retained JSPT inputs and creates fresh execution
and result identities; it compares scientific result data exactly under the
retained runtime pins. Inspection needs no domain checkout and executes no
scientific provider. Replay is offline once the pinned sources and dependencies
are installed.

Use `--json` on any of these commands for complete records. `--adapter-python`
on the covariance commands (alias `--python`) selects an explicit installed
interpreter. The investigation commands use `--python`. Defaults use CIW's
current interpreter.

The example map declares

\[
J=\begin{bmatrix}1&1\\1&-1\end{bmatrix},\qquad
C_{\mathrm{out}}=J C_{\mathrm{in}}J^{\mathsf T}.
\]

Its input order is `[tank-1.mass, tank-2.mass]`; output order is
`[total_mass, mass_difference]`, both in kg. The fixture declares reference
values `[100, 40]`. JSPT computes the covariance transformation. It does not
derive or verify these reference values or the supplied Jacobian.

## Exact covariance artifact

`covariance-artifact.v1` has exactly these fields:

| Field | Meaning and contract |
| --- | --- |
| `schema` | Literal `covariance-artifact.v1` |
| `covariance_id` | `sha256:` content identity covering every other field |
| `quantity_ids` | Nonempty ordered, unique quantity IDs; the order of both matrix axes |
| `units` | One unit per quantity; entry `C[i][j]` has units `units[i] * units[j]` |
| `frame` | Explicit coordinate-frame identifier |
| `reference_values` | Finite values in the same order and units; basis/linearization reference, not automatically observed truth |
| `matrix` | Full finite square covariance, including off-diagonal entries |
| `method` | Declared producing method |
| `basis` | Exactly `{kind, id}`; kind is `observation`, `calibrated_observation`, `estimated_state`, `parameter`, `coordinate` or `residual` |
| `provenance` | `provider`, `source_evidence_ids`, `source_covariance_ids`, and optional JSON `metadata` |
| `assumptions` | Explicit array of assumption strings |

Source evidence/covariance IDs in provenance are distinct `sha256:` content
identities. Their links describe how the artifact was produced; execution IDs
identify attempts and result IDs identify retained outcomes. These roles are
not interchangeable. A hash binds the declaration; it does not establish the
declaration's physical validity.

CIW checks finite entries, dimensions, declared ordering/bindings, nonnegative
variances, symmetry and positive semidefiniteness. Singular PSD covariance is
valid; invertibility is not a universal covariance requirement. A zero-variance
coordinate requires an exactly zero row and column. The validator checks
symmetry and PSD in dimensionless correlation coordinates, with tolerances
`1e-12` and `1e-10` respectively; it never diagonalizes, clips eigenvalues,
averages entries or replaces missing values to force acceptance. A downstream
operation may still refuse a matrix when its own numerical method requires a
nonsingular subproblem.

Mixed physical units are allowed. Each provider/caller must bind the order,
units and frame to its model; a matrix shape alone cannot establish that
correspondence. RCI's parameter covariance order remains `[scale, zero_raw]`.
FSRT's estimated state order is `[tank-1.mass, tank-2.mass]`.

## What calibration covariance includes

RCI v2 preserves the complete `rci-covariance-basis.v1` declaration under the
calibration binding's `covariance_basis` and in the resulting uncertainty record.
It requires the following explicit components:

| Block | Required declaration |
| --- | --- |
| `parameter_components` | Coverage of `fitting`, `reference_standard` and `shared_systematic`; each component has an ID, included/excluded status, covariance or null, `represented_by`, reason and source/evidence identifiers |
| `raw` | Reason and evidence/dependency/shared-source IDs for the raw covariance |
| `residual` | The same provenance plus scope `additional_independent_output_residual` |
| `independence` | `parameter_components`, `raw_parameter` and `residual_other` are explicitly true, with reason and evidence IDs |

Included parameter components must sum to the declared parameter covariance.
An excluded component has null covariance. If its effect is already included
elsewhere, `represented_by` must name that directly included component, with
matching source identities. Otherwise the exclusion remains an explicit limit;
the record does not claim complete uncertainty coverage. Residual sigma must
exclude effects already allocated to raw or parameter uncertainty.

An ambiguous basis, overlap between independently added source components,
unallocated covariance or double-counted uncertainty refuses the calibration.
The original raw bytes remain unchanged. The native parameter/raw/residual
contributions, parameterization and coverage/exclusion provenance remain
available in the saved RCI record and subsequent artifact metadata.

CIW's initial two-assembly composition requires
`cross_assembly_independent: true`. It checks the declared dependency and shared
source sets and refuses an overlap with `unsupported_cross_assembly_dependence`.
Distinct IDs are not evidence of independence. Correlated assemblies require an
explicit joint uncertainty model that this composition does not implement.

The v2 input also requires `model_independence`, containing an explicit reason
and these three true flags:

```json
{
  "prior_independent_of_observations": true,
  "declared_total_independent_of_observations": true,
  "prior_independent_of_declared_total": true,
  "reason": "Evidence supporting these independence declarations"
}
```

These are caller declarations and model restrictions. An unknown/false flag is
refused; a prior or total fitted from the same observations must not be labelled
independent to enter this operation.

## FSRT and JSPT outputs

FSRT v2 preserves the existing estimator and balance calculations. It supplies
named artifacts under `data.covariance_artifacts`:

| Name | Quantity represented |
| --- | --- |
| `observation` | Calibrated observation covariance in source order |
| `prior` | Declared isotropic prior covariance |
| `declared_total` | Independently declared total-mass variance |
| `innovation` | Full innovation covariance over observed coordinates |
| `posterior` | State covariance before balance reconciliation |
| `reconciled` | Final state covariance, with the reconciliation status retained |

Physical-model disagreement can hold reconciliation; an artifact named
`reconciled` does not override that diagnostic or claim a correction occurred.
The observation, estimate and residual identities remain separate.

JSPT accepts an explicit matrix `jacobian`, output quantity order/units/frame,
output reference values and one `map_kind`: `linear`, `local_linearization`,
`weighted_aggregation` or `coordinate_change`. A coordinate change requires a
square invertible chart that passes JSPT's numerical conditioning guard;
reductions use a linear or aggregation map. The result
retains both input and output covariance artifacts and the supplied map.
Local linearization is a first-order approximation; a fixed linear covariance
push is exact for the declared map in exact arithmetic.

The CLI selects the most recent eligible covariance result by default, with
source artifact `reconciled` for FSRT and `output_covariance` for JSPT. Parameters
may select a specific retained `source_result_id` and `source_artifact`.
CIW resolves the matching `source_covariance`; it does not accept a replacement
source matrix from the parameters file. Reopening validates the saved result
dependencies before writing destination records, including source bindings and
self/cyclic dependencies.

The terminal prints complete matrices, their axis order/units, frame, method
and covariance IDs. JSON retains source evidence and covariance links and the
full component context. These are terminal/data representations; no Godot
covariance-ellipse or other covariance viewport is implemented here.

## Calibration at acquisition and at serving time

RCI v2 stores an immutable `acquisition_applicability` object in each derived
record: `observed_at`, `applicable`, `valid_from`, `valid_until` and basis
`caller_declared_acquisition_time`. It evaluates the half-open interval
`valid_from <= observed_at < valid_until`. It does not read a current clock to
rewrite that historical statement.

CIW separately derives serving metadata using `calibration_status(run,
evaluated_at=None)`. One UTC evaluation time is shared by all sensor statuses.
An explicitly supplied time must include seconds and a timezone. The derived
`serving` object reports `evaluated_at`, `expired`, `not_yet_valid` and
`current_applicability`. A profile can be expired today while having been
applicable when the retained sample was acquired. V1 records have no persisted
applicability artifact: CIW labels their comparison `derived.legacy-v1` and
does not add fields to their historical evidence.

Live read surfaces accept an optional `evaluated_at` request field:

| Surface | Location of derived metadata |
| --- | --- |
| `session.get` / session snapshot | Payload `calibration`, alongside run/selection/results |
| `result.list` | Payload `calibration`, alongside result summaries |
| `result.get` | Response-envelope `calibration`, alongside the exact retained result `payload` |
| `investigation inspect --evaluated-at ...` | Summary `calibration`, with acquisition and serving displayed separately |

These reads do not change evidence, result seals, execution history or saved
file bytes. Successful computation remains `not_verified`; neither covariance
propagation nor current calibration applicability supplies a verification
identity, physical certification or decision authority.

## Live bindings and historical replay

To inspect the saved investigation and explicitly enable JSPT, start a live
session in one terminal:

```sh
python -m ciw serve --workspace results/covariance-map/workspace.json --output-dir results/covariance-live --fsrt-repo ../fsrt --jspt-repo ../jspt
```

In a second terminal:

```sh
python -m ciw send operation.list
python -m ciw send session.get --payload '{"evaluated_at":"2026-09-20T12:00:00Z"}'
```

The generic live `operation.execute` path uses the full resolved JSPT parameter
shape: map fields plus `source_result_id`, `source_artifact` and the exact
`source_covariance` object from that retained result. A mismatch is refused.
The `ciw covariance` command above performs this resolution for a parameter file.
Saved data never select an executable. A new server must receive trusted local
`--fsrt-repo` / `--jspt-repo` bindings explicitly.

To repeat the retained JSPT request through that live session, extract its
scientific parameters into a request file and send them:

```sh
python - <<'PY'
import json
from pathlib import Path

workspace = json.loads(Path('results/covariance-map/workspace.json').read_text())
result = [r for r in workspace['results'] if r['operation_id'] == 'jspt.covariance-propagate.v1'][-1]
payload = {'operation_id': result['operation_id'], 'parameters': result['parameters']}
Path('results/covariance-request.json').write_text(json.dumps(payload))
PY
python -m ciw send operation.execute --payload-file results/covariance-request.json
python -m ciw send workspace.save
```

Runtime pins include a bounded historical allowlist. New execution uses the
current pin. Replay accepts the saved revision only if it is still explicitly
allowlisted, and checks its recorded interpreter/dependency identity. Supply a
separate clean local checkout at the saved old commit when replaying an old
investigation; CIW does not fetch or change repositories automatically. Existing
v1 records are not rewritten into v2 and old results are not silently recomputed
under a newer scientific runtime.

## Validation and remaining limits

The checked-in tests exercise covariance identity/shape/PSD gates, native-domain
provenance and refusal behavior, frozen-time serving status, unchanged stored
bytes, and the pinned integration path. Run the ordinary suite and then enable
the external-source integration gates:

```sh
python -m pytest -q
CIW_RCI_REPO=../rci CIW_FSRT_REPO=../fsrt CIW_JSPT_REPO=../jspt python -m pytest -q
```

This increment does not establish empirical covariance calibration, innovation
or estimation consistency through NIS/NEES, interval coverage, or physical
traceability. The JSPT boundary does not derive Jacobians or run Monte Carlo
comparisons. Time-correlated calibration errors, general fluid topologies,
cross-assembly dependence and physical calibration qualification are not supplied
by this integration. The full covariance record and its assumptions remain
available for inspection and explicit downstream operations.

## Recorded compatibility limitation

The FSRT Windows/Python 3.13 workflow at its pinned covariance-adapter revision
reported a failure in `tests/test_real_noaa_month.py::test_the_report_reproduces`,
where the tide-month profile log-likelihood exceeded the existing cross-build
reproduction tolerance. The
[adapter-revision workflow](https://github.com/giasonpooni/Fluid-State-Reconstruction-Testbed/actions/runs/35536619209)
and [previous-revision workflow](https://github.com/giasonpooni/Fluid-State-Reconstruction-Testbed/actions/runs/35533956823)
record the same test failure before and after that increment. This integration
does not change the NOAA model, fit, report or reproduction tolerance. Its
adapter tests do not establish cross-platform numerical equivalence of the
complete upstream NOAA workflow.
