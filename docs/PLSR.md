# PLSR terminal instrument

The Parameterized Lyapunov Stability Runtime (PLSR) is integrated as an
experimental computational instrument in CIW's terminal. The complete path is
**import a declared model → evaluate an explicit sample → inspect → retain
evidence → replay**. It evaluates the model and certificate supplied by the
caller; it does not construct a model or establish that it describes a physical
plant.

This adapter is headless. PLSR runs are self-contained JSON bundles, separate
from the oscillator's shared WebSocket session and Godot viewport. No viewport
is needed, and PLSR results are not currently published to a live shared session.

## Installation and pinned versions

Use **Python 3.12 or newer** and Git for the PLSR extra. The base oscillator
installation continues to support Python 3.11. From the repository root in an
activated virtual environment:

```text
python -m pip install -e '.[plsr]'
python -m ciw plsr --help
```

The extra installs the upstream runtime from the full commit
[`19ea6967060166ba09db6cd4563bd87bd6b3d196`](https://github.com/giasonpooni/Parameterized-Lyapunov-Stability-Runtime/tree/19ea6967060166ba09db6cd4563bd87bd6b3d196).
The adapter checks the installed runtime's source against its pinned source
manifest. An equivalent wheel is accepted; Git checkout metadata alone is not
the test of identity. A different runtime source is rejected. Keep the CIW
revision as well as the runtime pin when reproducing an investigation.

| Boundary | Version |
| --- | --- |
| CIW adapter | `ciw-plsr-adapter-v1` |
| CIW operation | `plsr.verdict.v1` |
| Model declaration | `model-artifact-v1` |
| Explicit sample | `plsr-sample-v1` |
| Saved CIW run | `ciw-plsr-run-v1` |
| Reference corpus | `ciw-plsr-corpus-v1`, reported as `ciw-plsr-corpus-report-v1` and `ciw-plsr-corpus-record-v1` |
| Companion record | `companion-record-v1`, kind `ciw-plsr-evaluation`, adapter schema `ciw-plsr-evaluation-v1` |
| Runtime status vocabulary | `runtime-status-v1` |
| Numerical policy | `float64-decrease-v1` |
| Claim codes | `claim-codes-v1` |

Python, NumPy and jsonschema versions are retained with each run, alongside the
runtime repository, commit, package version and source digest. The full SHA and
source check establish which code the adapter expects. They do not authenticate
a producer or supply a mathematical proof receipt.

If a PLSR command reports a missing optional dependency, install `.[plsr]` using
the same `python` that runs CIW. A Python 3.11 environment must be replaced with
a Python 3.12-or-newer environment for this extra. A source mismatch calls for
reinstalling the pinned runtime; changing the expected manifest to accept an
unknown installation would discard the version check.

## First complete run

The checked-in models are unmodified upstream synthetic fixtures. Both use
ordered, dimensionless states `x1`, `x2`, a required margin of `0.1`, and a level
of `4.0`. Their provenance digests identify fixtures, not measurements or actual
estimator attestations.

Import and evaluate the continuous affine model:

```text
python -m ciw plsr import examples/plsr/continuous-affine.json --output models/plsr.json
python -m ciw plsr evaluate --model models/plsr.json --sample examples/plsr/continuous-sample.json --output-dir results/plsr
```

Import validates the model and its digest before writing it. It reports
`model_file`, `model_artifact_digest`, `artifact_schema` and the checked runtime
identity. Importing the same declaration to the same output is safe; an existing
different declaration is refused.

Evaluation prints JSON with `saved_file` and `bundle`. The file has a unique
`run-<result UUID>.json` name. It already contains the complete model, explicit
sample, companion record and provenance; there is no separate save step.

For a copyable evaluation, inspection and replay sequence in PowerShell:

```powershell
$run = python -m ciw plsr evaluate --model models/plsr.json --sample examples/plsr/continuous-sample.json --output-dir results/plsr | ConvertFrom-Json
python -m ciw plsr inspect $run.saved_file
python -m ciw plsr replay $run.saved_file --output-dir results/plsr-replay
```

For a POSIX shell:

```sh
run_file=$(python -m ciw plsr evaluate --model models/plsr.json --sample examples/plsr/continuous-sample.json --output-dir results/plsr | python -c 'import json,sys; print(json.load(sys.stdin)["saved_file"])')
python -m ciw plsr inspect "$run_file"
python -m ciw plsr replay "$run_file" --output-dir results/plsr-replay
```

`inspect` validates and displays the saved bundle without repeating the
evaluation. `replay` validates the source, evaluates its embedded model and
sample with the pinned runtime, and writes a new bundle. It assigns new execution
and result identities and retains a binding to the original run. The original
file is preserved. See [saved evidence and replay](#saved-evidence-and-replay)
for the exact comparison.

The discrete example uses `A = diag(0.8, 0.7)`, `P = I`, and a declared sample
period of `0.01 s`:

```text
python -m ciw plsr import examples/plsr/discrete-linear.json --output models/plsr-discrete.json
python -m ciw plsr evaluate --model models/plsr-discrete.json --sample examples/plsr/discrete-sample.json --output-dir results/plsr-discrete
```

Both supplied samples are expected to produce `CERTIFIED_WITH_MARGIN`. This is
a computational result for the synthetic declaration and sample.

## Input specification

The upstream [model-artifact guide](https://github.com/giasonpooni/Parameterized-Lyapunov-Stability-Runtime/blob/19ea6967060166ba09db6cd4563bd87bd6b3d196/docs/MODEL-ARTIFACT-v1.md)
and [JSON Schema](https://github.com/giasonpooni/Parameterized-Lyapunov-Stability-Runtime/blob/19ea6967060166ba09db6cd4563bd87bd6b3d196/src/lyapunov/schemas/model-artifact-v1.schema.json)
are the model contract. CIW routes that declaration through the upstream loader;
it does not infer matrices, synthesize `P`, change margins or repair a mismatched
digest.

| Model field | Interpretation |
| --- | --- |
| `state.definition`, `state.coordinates` | State meaning, reference/equilibrium, ordered coordinate names and units; `x` follows this order |
| `plant` | Linear `A`, or affine `A0` and ordered terms with parameter metadata and closed parameter/rate boxes |
| `certificate` | Quadratic `P`, or supported continuous affine `P0` and terms |
| `time` | Continuous derivative or discrete one-step convention, with explicit sample-period declaration |
| `policy` | Required margin, optional level, margin derivation and evaluator/status/claim schema versions |
| `estimator`, `provenance` | Declared estimator identity/configuration, state compatibility, producer and source/report digests |
| `artifact_digest` | SHA-256 binding all other model declaration fields |

The sample object has exactly four required fields:

```json
{
  "sample_schema": "plsr-sample-v1",
  "x": [0.1, 0.2],
  "theta": [0.5],
  "theta_dot": [0.1]
}
```

`x` must match the state dimension and declared ordering/units. `theta` follows
the model's ordered parameter list. For continuous affine models, `theta_dot`
contains parameter derivatives per second. Continuous affine models require both
vectors explicitly. Linear models require `theta: null` and `theta_dot: null`;
discrete affine models require `theta` and `theta_dot: null`. Arrays must have
the declared lengths and finite numeric entries. Unknown sample fields or schema
versions are refused. Omitted fields do not receive demonstration defaults.

The continuous fixture declares one dimensionless parameter in `[-1, 1]` and
its rate in `[-0.2, 0.2] /s`. The discrete fixture has no parameters. Discrete
matrices already represent the declared one-step transition; CIW does not
discretize a continuous matrix or divide a discrete decrease by the sample
period. Units and estimator compatibility are declarations to be established by
the host before using physical data.

## Verdicts, refusals and errors

Each companion record preserves the raw runtime code and the independent
`inequality_certified`, `meets_required_margin` and `operationally_acceptable`
booleans. Read them together. A presentation category is a display aid, not a
replacement for these fields or an authorization to act.

| Runtime code | Meaning for the terminal user |
| --- | --- |
| `CERTIFIED_WITH_MARGIN` | Decrease is resolvably certified and clears the declared margin |
| `MARGIN_LOW` | Resolvable decrease does not clear the declared margin |
| `NOT_CERTIFIED` | The sample has resolvable positive change; shown as `certificate_violation` |
| `DECREASE_NOT_DEFINITE` | The form is resolvably not negative definite, although this sample decreases |
| `NUMERICAL_INCONCLUSIVE` | The numerical resolution cannot establish the requested sign; shown as `numerical_refusal` |
| `NUMERICAL_OVERFLOW` | The evaluation's arithmetic cannot produce a resolvable sample within float64 |
| `OUTSIDE_PARAMETER_BOX` | A parameter or rate lies outside its declared closed box |
| `OUTSIDE_LEVEL_SET` | The state exceeds the declared level; consult the retained booleans for decrease and margin |
| `CERTIFICATE_NOT_POSITIVE` | The evaluated `P` is not positive definite |

See the pinned [runtime-status specification](https://github.com/giasonpooni/Parameterized-Lyapunov-Stability-Runtime/blob/19ea6967060166ba09db6cd4563bd87bd6b3d196/docs/RUNTIME-STATUS-v1.md)
for the numerical definitions. In particular, `NUMERICAL_INCONCLUSIVE` must not
be counted or displayed as a certificate violation. The published 57% refusal
measurement applies to deliberately difficult boundary cases, not ordinary
evaluations. The adapter does not add a stronger second numerical tier.

| CLI exit code | Meaning |
| --- | --- |
| `0` | Command succeeded; an evaluation record can contain a refusal or violation |
| `2` | Invalid input, missing/incompatible runtime, persistence or execution error |
| `3` | Replay completed but the new companion record digest differs from the source |
| `4` | A corpus case did not reproduce its declared expectation, or a recording was refused |
| `5` | The corpus was recorded against a runtime identity this installation does not provide |

Errors are reported on standard error. An invalid declaration or malformed
sample is an input error, distinct from a valid evaluation returning an
out-of-box or numerical status. Scripts must inspect the record's code and
booleans rather than interpreting exit zero as certification.

## Saved evidence and replay

The saved `ciw-plsr-run-v1` object contains:

| Fields | Meaning |
| --- | --- |
| `model`, `sample` | Complete model declaration and explicit sample used in the evaluation |
| `record` | PLSR companion record of kind `ciw-plsr-evaluation`, including model binding, sample, verdict and diagnostics |
| `runtime` | Pinned repository/commit, package/source identity, adapter and dependency versions |
| `evidence_id`, `operation_id` | Input evidence identity and `plsr.verdict.v1` operation identity |
| `execution_id`, `result_id`, `created_at` | Identity and timestamp of this execution and retained result |
| `verification_id`, `verification_status` | `null` and `not_verified`; computational integrity checks do not promote verification |
| `replay_of` | `null` on an initial run; source bindings and comparison on replay |
| `bundle_digest` | SHA-256 of the complete bundle excluding this field |

Three bindings are retained: the model's `artifact_digest`, the companion
`record_digest`, and CIW's `bundle_digest`. They bind the model, evaluated sample
and output, and complete saved run respectively. Validation checks their
relationships as well as the digests. A bundle carries its own inputs, so moving
it does not require the original model/sample files.

Digest serialization applies to decoded JSON with sorted object keys, compact
separators, `ensure_ascii=True` and `allow_nan=False`. It is the Python JSON
serialization convention, not RFC 8785/JCS. Whitespace and object-key order do
not matter; array order and integer-versus-float representation can matter.
Digests detect content inconsistencies relative to the retained bindings; they
are neither signatures nor proof receipts.

Run files are published atomically without overwriting existing files. Save to
a filesystem supporting hard links, such as local NTFS or Linux filesystems;
unsupported storage reports a persistence error. Replay records
`source_result_id`, `source_bundle_digest`, `source_record_digest` and
`record_digest_matches` in `replay_of`. It compares the entire deterministic
companion-record digest exactly. New result/execution IDs and timestamps make
the new bundle distinct even when its evaluation agrees. Numerical differences
across environments can produce a mismatch; retained dependency versions help
investigate it. A match establishes repeatability of that computation, not
physical validity or proof verification.

## Reference corpus and contract checks

`examples/plsr/corpus.json` is a declared `ciw-plsr-corpus-v1` document. It
stores, in one place, everything a case needs to be reproduced: the model it
names, the explicit sample, the expected output, the runtime identity the
expectation was recorded against, and the numerical tolerance policy it is
compared under. Reproducing it needs no network, no saved results and no
argument beyond the corpus path:

```text
python -m ciw plsr corpus check examples/plsr/corpus.json
```

The command prints a `ciw-plsr-corpus-report-v1` object and exits `0` when
every case reproduced. It never writes an expectation; re-recording is a
separate command described below.

### What the corpus covers

Twenty-three cases over eight models. Every runtime status this adapter can
reach through a declared model appears at least once.

| Purpose | Cases | Covers |
| --- | --- | --- |
| `certification` | 2 | The published continuous affine and discrete linear fixtures |
| `margin_shortfall` | 1 | `MARGIN_LOW`: a 0.04 decrease margin against a declared 0.1 |
| `violation` | 2 | `NOT_CERTIFIED` and `DECREASE_NOT_DEFINITE` from one indefinite form at two states |
| `domain_refusal` | 5 | Closed-box endpoints inside, one float64 step outside either box, and `OUTSIDE_LEVEL_SET` |
| `numerical_boundary` | 6 | The origin, the smallest denormal state, an overflowing state with and without a declared level, an exactly zero decrease form, and a decrease matrix that leaves float64 |
| `contract_refusal` | 7 | Parameters on a linear plant, a rate on a discrete plant, a missing parameter vector, a wrong state dimension, a boolean state entry, an unknown sample schema, and an affine certificate that loses positive definiteness |

`CERTIFICATE_NOT_POSITIVE` is deliberately absent from the recorded codes and
cannot be produced through a declared model. A constant `P` is checked for
positive definiteness when the artifact loads, and an affine `P(theta)` that
loses it raises from the certificate rather than returning that status, so it
reaches the terminal as an input refusal with exit code `2`. The corpus records
that refusal instead of pretending the status is reachable.

### What is compared, and how exactly

| Declared value | Comparison |
| --- | --- |
| Runtime status code, presentation category, refusal text, every string | Exact |
| The three verdict booleans and every other boolean | Exact |
| Integers, including `state_scale_exponent` and mantissa exponents | Exact |
| A float whose field the tolerance policy names | Relative and absolute, as the policy declares |
| Every other float, including `A`, `P`, the decrease matrix, `V` and the scaled decrease | Exact, and `-0.0` does not match `0.0` |

A tolerance is opt-in per field and visible in the corpus. The published policy
`plsr-corpus-tolerance-v1` names exactly four fields — `diagnostics.min_P`,
`diagnostics.max_decrease`, `diagnostics.margin` and `diagnostics.margin_ratio` —
at a relative tolerance of `1e-12` and an absolute tolerance of zero. Those four
are read from a LAPACK symmetric eigensolver, whose final bits are not fixed by
the reference. Everything else is fixed-sequence float64 arithmetic on the
declared matrices and is compared exactly. A field name in the policy that is
not a float is itself reported as a difference, so a tolerance can never be
attached to a status code or a boolean. Policy entries no case reaches are
listed in `unused_tolerances`.

The `digest_enforcement` field governs the companion `record_digest`. The
published corpus sets it to `reported`: each case's `record_digest_matches` is
reported, and a digest difference alone does not fail the case, because the
digest also covers the four eigensolver-derived floats. Setting it to `enforced`
makes the digest part of the expectation. This does not weaken the exact
replay-digest check: `replay` still compares the whole companion record digest
exactly, and the `ciw-plsr-corpus-report-v1` object records how many digests
matched. On the checked installation all sixteen evaluated cases reproduce their
digest exactly.

### Staleness

The corpus records the complete runtime identity and a `runtime_binding` list
naming which of its fields must match. The published corpus binds the five pin
fields: `repository`, `commit`, `package_version`, `source_digest` and
`adapter_version`. A difference in any of them makes the corpus **stale**: no
case is run, the report's `runtime_status` is `stale`, every case is counted as
`skipped`, and the command exits `5`. Running the cases against a different
runtime would report numerical mismatches with the wrong cause.

The interpreter, NumPy and jsonschema versions are recorded and reported when
they differ, but are not bound by default, because the same corpus is checked on
Linux and Windows CI runners that do not share a patch version. A corpus that
needs them bound lists them in `runtime_binding`.

### Recording an expectation

Expectations are never refreshed by `check`. Recording is explicit:

```text
python -m ciw plsr corpus record examples/plsr/corpus.json --output examples/plsr/corpus.json
```

`record` re-observes every case, compares the result with the saved corpus, and
prints a `ciw-plsr-corpus-record-v1` object listing every change field by field.
If anything changed it writes nothing and exits `4`. Pass `--accept-changes` to
replace the corpus with what was observed. Writing an unchanged corpus is a
no-op that leaves the file byte-identical.

The corpus carries a `corpus_digest` over its own content, so editing an
expectation by hand without re-recording is refused before any case runs. Add,
remove or change a case by editing its `case_id`, `purpose`, `description`,
`model_file` or `sample` and then running `record`; the expectation itself is
produced by the pinned engine, never typed in.

Each case also binds its model by `model_artifact_digest`. Editing a model file
fails the case as a binding error naming the model, not as a numerical
difference in the verdict.

`--case CASE_ID` (repeatable) checks a subset. `--output-dir DIR` additionally
retains one ordinary `ciw-plsr-run-v1` bundle per evaluated case, so a corpus
run leaves the same evidence a manual evaluation would. Generated bundles belong
outside `examples/`; `results/` is ignored by git.

## Validation and limits

Install the development and PLSR extras to exercise the CIW integration:

```text
python -m pip install -e '.[dev,plsr]'
python -m pytest -q
```

The [terminal and persistence tests](../tests/test_plsr.py) and
[engine adapter tests](../tests/test_plsr_engine.py) cover import, explicit
samples, runtime pin enforcement, verdict/refusal preservation, retained
evidence, tamper rejection and replay. The
[reference-corpus tests](../tests/test_plsr_corpus.py) reproduce every declared
case, restate each case's intended runtime code independently of the recorded
one, and pin the staleness, exactness, tolerance and recording rules above.
The [PLSR CI job](../.github/workflows/test.yml) runs these checks on Windows and
Linux with Python 3.12, then builds a wheel and runs the
[installed-package acceptance check](../scripts/check_plsr_installed.py) outside
the source tree for both supplied models and for the whole reference corpus.
The worked commands above exercise the actual terminal path. Upstream numerical
validation remains documented in the
[pinned runtime repository](https://github.com/giasonpooni/Parameterized-Lyapunov-Stability-Runtime/tree/19ea6967060166ba09db6cd4563bd87bd6b3d196);
it does not substitute for the workbench checks.

This integration supports one explicit sample per evaluation, and the corpus
check evaluates a declared collection of them in one process. Host scripts may
repeat calls for parameter sweeps or offline samples, retaining one bundle per
run. There is no sweep scheduler, state estimator, sensor acquisition, temporal
freshness check or shared PLSR viewport in this increment. Each CLI evaluation
starts a process; no real-time control-loop performance is claimed.

The runtime remains experimental and its kernel API is marked `changing`;
the adapter's exact source pin bounds that dependency. Records retain
`claim_scope: computational-integrity-only`, `may_authorize: false`,
physical validation `not_started`, and `proof_status: NOT_CHECKED`.
Deployment authorization remains prohibited. Declaring a model, verifying a
digest or matching a replay does not change these states.
