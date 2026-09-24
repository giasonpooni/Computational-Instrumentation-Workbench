# Exchange and provenance, part 1: identity matrix and workspace mutations (T077–T090)

Implementation: `src/ciw/lab/exchange_provenance.py` (tasks) and
`src/ciw/lab/exchange_provenance_common.py` (offline fixtures and forgery
harness). Tests: `tests/test_lab_exchange_provenance.py`.

```
python -m ciw lab run T077 T078 T079 T080 T081 T082 T083 T084 T085 T086 T087 T088 T089 T090 --output-dir results/lab-exchange
python -m ciw lab report T084 --retained results/lab-exchange
python -m pytest -q tests/test_lab_exchange_provenance.py
```

These experiments test the unmodified CIW integrity layer (`ciw.session`,
`ciw.operations.runner`, `ciw.workbench`, `ciw.energy_workflow`,
`ciw.energy_records`, `ciw.exchange`, `ciw.candidate_evidence`,
`ciw.core.identities`, `ciw.telemetry`) with no network, provider checkout,
GPU or hardware. The section runs in about 13 s and its tests in about 13 s
on one core.

## What was built

One offline session fixture is shared by the tasks:

- **Oscillator session.** `make_demo_run()`, then `statistics.v1` on channel `v`
  over `[1, 2]` s twice (executions E1 and E2, results R1 and R2, sealed by
  `operations.runner.seal`) and one legacy `analysis.stats` result RL, which has
  no seal.
- **Energy-accuracy workbench.** This path needs no provider binding. The
  sources are `examples/energy-accuracy/baseline.json` and `reset.json` (both
  `origin: synthetic_fixture`). It holds two original executions of the baseline
  log (B0 and its sibling B0b), one execution of the reset log (Bother), and a
  replay B1 of B0.
- **Save, reopen and replay again.** `workspace.save` writes the workspace and
  `Session.from_workspace` reopens it offline. The reopened session replays B0
  (giving B2) and replays the replay B1 (giving B3).
- **Separate session (same process and code).** A second session retains the
  same bytes under the same label and under a second label, and executes only
  the same-label source (B4). It is not an independent reproduction.

A second fixture serves T077–T079 and T081:

- **Byte-variant workbench.** The four fixture logs plus eight byte variants of
  the baseline, all eight under one shared label, so any difference in their
  source identities comes from their bytes. Each is retained, executed, saved
  and reopened.
- **Resealed content variants.** Two edits of the baseline whose unkeyed
  `log_digest` is recomputed: the sensor name renamed (metadata only) and the
  unit diagonal of both `initial_covariance` copies written as `1` instead of
  `1.0`. Each gets a workbench of its own, because CIW refuses to analyse two
  logs with one `run_id` and different log digests in one workbench
  (`Identity collision across retained workbench artifacts`, recorded in T081).

A **mutant** is a copy of the saved workspace with one edit. After the edit, the
harness recomputes *none*, the *local* digests the edit touches, or *all*
downstream digests. Every digest, including the log author's `log_digest`, is an
unkeyed SHA-256 over CIW's own public canonicalization, so anyone holding the
file can recompute it. The mutant is written to disk and reopened with
`Session.from_workspace`. The outcome is either `accepted` or the exact refusal
message. Pure-validator rows run `candidate_evidence.validate_response` or
`exchange._identity` on synthetic records instead of reopening a workspace.

Each row separates two things:
- `predicted_outcome` (killed or accepted) is the prediction stated in each
  task's hypothesis from the validator code path that should or should not
  catch the edit;
- `pinned_message` is the exact refusal text recorded from an observed run. It
  is a regression pin, not a prediction.

Two harness controls run on every retained matrix:
- The forger's full recomputation of an unedited workspace reproduces every CIW
  digest byte for byte.
- The unedited workspace reopens.

Both controls hold.

## Identity matrix (T077)

The matrix has 30 identities. 28 are exercised offline, and all 97 predicted
properties hold. Of those, 11 are re-derivations of a CIW digest with the lab's
own canonical JSON and SHA-256 (same implementation origin, recorded as
`cross_implementation`). The other 86 observe CIW's outputs, its refusal
messages, or its own validators (for example, `EnergyAccuracyWorkflow._validate_step`
judges a step moved to a new occurrence).

The *Across reopen* column is checked row by row. For each of the 22 exercised
identities predicted stable, and for the absent oscillator numerical identity,
a property compares the value saved before reopen with the value the reopened
session saves again. Records are paired by catalog position, not by identity,
so no identity is assumed stable. All 23 hold.

Checks that could not fail by construction are not counted as properties:
identifier prefix tests, an ESM digest the lab itself set, a `source_id` count
that differing labels guaranteed, and the fixture's placeholder producer hashes.
The placeholder hashes are a fact about the repository fixture, not about CIW,
so they are recorded as a value on the physical finding below.

**T077 is `partial`.** Two planned rows cannot run offline, and their
predictions are read from the code, not observed:
- the ESM `candidate_id` and candidate execution identities, which need an
  operator-bound ESM adapter and a telemetry or calibrated bundle;
- the pinned-provider runtime identity (`ciw.subprocess-runtime.v1`), which
  needs a provider checkout bound to a declared-workload or telemetry workflow.

The full matrix, with per-property results and the observed refusal messages, is
retained as `artifacts/T077/identity-matrix.{json,md}`.

| Identity | Derivation | Binds | Across replay | Across reopen | Validated at |
| --- | --- | --- | --- | --- | --- |
| recording `evidence_id` | content hash: canonical JSON of instrument, metadata, time_s, channels | scientific content; excludes `run_id`, `render` | n/a | stable | `core/identities.py:validate_evidence_identity`, `session.py:_validate_evidence` |
| recording file name | content hash of the whole run (`recording-<sha256>.json`) | parsed run; the file is a CIW re-serialization, not caller bytes | n/a | stable | `session.py:_recording_file`, `_validate_saved_result` |
| workbench source `evidence_id` | **byte hash** of the exact source bytes | bytes only (not label); bundle `artifact_ref` and `sha256` are copies | stable | stable | `workbench.py:_source`, `Workbench.restore` |
| workbench `source_id` | content hash of `{schema, kind, label, source_schema, evidence_id, byte_count}` | bytes **and caller label** | stable | stable | `workbench.py:_source`, `_source_claims` |
| bundle `experiment_digest` | content hash of the parsed log | canonical content (whitespace and key order excluded) | stable | stable | `energy_workflow.py:EnergyAccuracyWorkflow._validate` |
| log `run_id` / `experiment_id` | caller-declared (copied from the log) | the log author's occurrence claim; unchanged by a resealed content edit | stable | stable | `energy_records.py:_validate`, `workbench.py:_claims` |
| log producer identities (`sensor.device_uuid`, `kernel_sha256`, `executable_sha256`, `code_sha256`) | caller-declared, copied verbatim | nothing outside the log; only the author's unkeyed `log_digest`; format-checked | stable | stable | `energy_records.py:_sensor`, `_runtime` |
| oscillator `operation_id` | caller-declared versioned name | which operation ran | n/a | stable | `operations/registry.py:valid_operation_id`, `runner.py:validate_execution` |
| energy `operation_id` | constant `ciw.energy-accuracy.v1` | which workflow ran | stable | stable | `EnergyAccuracyWorkflow._validate_step` |
| oscillator `execution_id` | fresh `execution-<uuid4>` | one occurrence | fresh | stable | `runner.py:validate_execution`, `session.py:_identity` |
| energy step `execution_id` | fresh `execution-<uuid4>` | one occurrence | fresh | stable | `_validate_step`, `workbench.py:_validate_links` |
| oscillator `result_id` | fresh `result-<uuid4>` | one occurrence; content bound only by the unkeyed seal | fresh | stable | `session.py:_validate_saved_result` |
| energy `result_id` | content hash of the result record including `execution_ref` | content and fresh occurrence | fresh | stable | `_validate_step` |
| energy `numerical_result_id` | content hash of `{operation_id, data}` | analysed numbers **and the log identity**: `data.log_digest` (canonical digest of the whole log, including unanalysed metadata), `origin`, `device_uuid` | **stable for canonically identical logs** | stable | `_validate_step`, `_validate_links` |
| oscillator numerical-result identity | **absent** | nothing | absent | absent | none |
| energy `verification_id` | `sha256(schema ‖ NUL ‖ canonical)` | subject, outcome, independent, method, runtime digest, reproduction, authority | fresh | stable | `_check_verification`, `exchange.py:_identity` |
| oscillator `verification_id` | always `null`, status `not_verified` | nothing | n/a | stable | `_validate_saved_result` |
| `bundle_digest` (= workbench `bundle_id`) | content hash excluding `bundle_digest`, `verification`, `replay_receipts` | session, time, source, configuration, runtimes, steps; **not verification or receipts** | fresh | stable | `telemetry.py:_bundle_digest`, `_validate` |
| bundle `session_id` | fresh `session-<uuid4>` | one workflow session | fresh | stable | `_validate`, `workbench.py:_claims` |
| protocol `Session.session_id` | fresh `session-<uuid4>` | live session only; never saved | n/a | fresh | `session.py:Session.__init__` |
| `replay_id` | content hash of the receipt | source/replayed digests, verification, admission | fresh | stable | `workbench.py:_validate_receipts`, `_validate` |
| oscillator `record_digest` | **unkeyed** seal over the record | whole record, recomputable by anyone | n/a | stable | `runner.py:check_seal` |
| oscillator runtime identity | provider-declared `{provider, version}` | a name and version string | n/a | never re-checked | `runner.py:validate_execution` (non-empty only) |
| energy runtime identity | sha256 of three CIW source files plus self-reported Python/NumPy versions | analysis code bytes and version strings | compared on replay | format-checked only | `energy_workflow.py:_check_runtime`, `_adapters` |
| exchange `result_id` / `verification_id` | `sha256(schema ‖ NUL ‖ canonical)` | content; status `content_recomputed_not_authenticated` | n/a | n/a | `exchange.py:_identity` |
| exchange `batch_id` | caller-declared | nothing (`caller_declared_reference`) | n/a | n/a | `exchange.py:_identity` |
| ESM `requestId` / `inspectedAt` | caller-declared | equality of request and response only | n/a | n/a | `candidate_evidence.py:validate_response` |
| ESM `bundleBytesDigest` | byte hash of the bytes CIW passes (its canonical serialization of the bundle) | those exact bytes; a digest over another layout of the same bundle is refused | n/a | n/a | `validate_response`, `workbench.py:Workbench._validate_candidate` |
| ESM `candidate_id` / candidate execution | content hash / fresh uuid4 | response bytes, policy, adapter identity | n/a | stable | `workbench.py:Workbench._validate_candidate` (not exercised offline) |
| pinned-provider runtime identity (`ciw.subprocess-runtime.v1`) | pinned revision, module and source root, beside host-measured `source_tree` and `python_sha256` | the pinned provider revision and the host's tree and interpreter digests | compared on replay (runtime projection without host paths) | pin and format checked | `declared_workload.py:DeclaredWorkflow._validate`, `_runtime_projection`, `workbench.py:_validate_links` (not exercised offline) |

The log's producer identities are placeholders in the fixture (`aaaa…`,
`cccc…`, `dddd…`) and CIW checks only their format: a resealed log with other
well-formed values is accepted. T077 therefore records the physical claim that
they identify the producing GPU and code as `not_established`.

T079 treats the variants' own properties (eight distinct input byte strings,
one shared label) as harness preconditions. They are asserted before any
finding is built and are never counted as checks, because they describe the
lab's inputs, not CIW's behaviour.

### Byte-level and content-level identities (T078, T079)

The source bytes are retained exactly. For all 14 sources, the bytes returned
live, inside each native bundle, and after save and reopen equal the supplied
bytes, with zero mismatches. Non-canonical base64 is refused, never normalized:
- missing padding and line wrapping are refused with `Source bytes must use canonical base64`;
- non-canonical trailing bits are refused with `Source bytes must use bounded canonical base64`.

The six refused submissions (three transports, three re-encodings) leave the
retained source count unchanged.

T079 retained eight variants of `baseline.json` under one label. They differ
only in whitespace, CRLF line endings, key order or float spelling (`1e-09`
written as `0.000000001`), so their parsed content is canonically equal:

| Identity | Level | Distinct values over 8 variants |
| --- | --- | --- |
| workbench `evidence_id` (bundle `artifact_ref`/`sha256` are copies) | byte | 8 |
| workbench `source_id` | byte + label (one shared label here) | 8 |
| bundle `experiment_digest`, sealed `log_digest`, `numerical_result_id` | canonical content | 1 |
| recording `evidence_id` (oscillator) | canonical content of scientific fields | n/a |
| `bundle_digest`, energy `result_id`, `verification_id`, `replay_id` | canonical content including fresh occurrences | n/a |

Boundary findings from the same experiment:
- **CIW's canonical comparison is type-sensitive (`1 != 1.0`).** Writing the unit
  diagonal of both `initial_covariance` copies as `1` leaves Python equality
  intact but changes CIW's canonical JSON. Without resealing, the log's own seal
  refuses it (`Retained log digest differs`). Once resealed, it is retained as
  distinct evidence, with a different `experiment_digest`, `log_digest` and
  `numerical_result_id`. It is never aliased to the original.
- **Cross-field consistency.** Rewriting only the first `1.0,` (the
  `solver_settings` copy) is refused by a different check, `Workload solver
  differs from plan`, because the two copies no longer agree.
- **A byte-order mark is refused.** The BOM-prefixed variant is refused with
  `The bound runtime did not return finite, unambiguous JSON`. The refusal is
  correct, but the message wrongly blames a "bound runtime" for a problem in the
  source bytes.

### Numerical identity (T081)

Twelve occurrences of the baseline log have one `numerical_result_id`: steps and
reproductions of the original, the sibling, the replay, the replay after reopen,
the replay of a replay and the separate session. That identity binds the whole
log through `data.log_digest`. Renaming the sensor, which is never analysed, and
resealing the log changes `numerical_result_id` while every analysed number
stays equal. The identity is therefore stable across occurrences of a
canonically identical log. It is not a function of the analysed numbers alone.

It is also stable only on one arithmetic platform: it hashes the analysed
floats bit for bit, and their last bits depend on the BLAS kernel. T081 was
regenerated on one Linux x86-64 host under the OpenBLAS kernels SkylakeX,
Haswell and Sandybridge (`OPENBLAS_CORETYPE`). Each run had one identity over
its twelve occurrences, but the three identities differed. Of the 47 analysed
floats, 12 (Haswell) and 13 (Sandybridge) moved by at most 2.2e-16: the
reference mean and covariance entries by at most 2.8e-16 relative, the rest
being filter errors at rounding level. The identity finding's value therefore
holds what its claim is about (occurrences, distinct numerical and result
identities, recomputation mismatches), which `ciw lab verify` compares exactly
on any kernel. T081 retains the run's own identity as
`energy_numerical_result_id` in `numerical-identity.json`, beside the
numerical result it hashes (`energy_numerical_result`); verify checks that
artifact only against its recorded digest. The other identities in the
artifact are replaced by role labels, because they are fresh per run.

## Mutation matrix (T080–T090)

There are 72 rows: 65 edit and reopen a saved workspace, and 7 run pure
validators. 52 are refused. 19 distinct workspace forgeries survive. One
validator row, `exchange.verification-independent`, is accepted by design and
is not counted as a survivor. The kill/survive prediction matched in 72 of 72
rows, and every refusal matched its pinned message (52 of 52). Each forgery is
run once. The T086 subject-with-source rebinding is the same edit as T084's
`receipt-source.sibling-execution`, so T086 cites that row instead of re-running
it. The full matrix, with a post-reopen witness for each survivor, is retained
as `artifacts/T090/mutation-matrix.{json,md}`, and each task also retains its
own rows.

| Task | Mutant | Recompute | Outcome (reopen; validator call for pure-validator rows) |
| --- | --- | --- | --- |
| T080 | `alias.result-execution` (R2 points at E1) | local | `Saved result identity mismatch or duplication` |
| T080 | `alias.execution-result` (E2 points at R1) | local | `Execution/result execution_id binding mismatch` |
| T080 | `alias.result-prefix` | local | `Invalid saved result identity` |
| T080 | `alias.operation` (statistics → spectrum; killed only by payload-shape validation) | local | `Invalid saved spectrum data fields or sample count` |
| T080 | `alias.swap-pairing` (R1↔R2 exchange executions consistently: ids and creation times swapped together) | local | **accepted** |
| T080 | `revision.gap` (records claim revision 999 of 1000) | local | **accepted** |
| T081 | `energy-data.naive` | none | `Energy analysis bundle identity, schema or size differs` |
| T081 | `energy-data.reforged` (every derived digest recomputed; source bytes unchanged) | full | `Retained energy analysis binding differs` |
| T081 | `energy-source.resealed` (retained log edited and resealed; every derived record rebuilt) | full | **accepted** |
| T081 | `oscillator-stats.naive` | none | `Operation record integrity mismatch` |
| T081 | `oscillator-stats.out-of-bounds` | local | `Saved statistics mean is outside its bounds` |
| T081 | `oscillator-stats.resealed` (mean moved inside its bounds) | local | **accepted** |
| T081 | `oscillator-stats.impossible-moments` (R1: \|mean\| > rms; R2: rms > max\|x\|) | local | **accepted** |
| T081 | `oscillator-stats.legacy` (unsealed legacy result) | none | **accepted** |
| T082 | `fresh.energy-replay-reuse`, `fresh.energy-reproduction-reuse` | full | `Declared workload bundles must have distinct execution and reproduction occurrences` |
| T082 | `fresh.cross-namespace` (energy step reuses an oscillator execution id) | full | `Identity collision between recording operations and retained workflows` |
| T082 | `fresh.oscillator-duplicate` | local | `Saved result identity mismatch or duplication` |
| T082 | `fresh.created-at-one` | local | `Execution/result created_at binding mismatch` |
| T082 | `fresh.created-at-shift` (backdated in both records) | local | **accepted** |
| T083 | `receipt.numerical-match-false`, `receipt.transplanted` | local / none | `Invalid retained energy replay receipt` |
| T083 | `receipt.transplanted-resealed` (replayed digest and `replay_id` recomputed; the donor's reproduction step kept) | local | `Retained energy analysis binding differs` |
| T083 | `receipt.transplanted-full` (receipt rebuilt on B0b from B0 and B0b's own step) | full | **accepted** |
| T083 | `receipt.fabricated` (new receipt on the never-replayed original B0b, claiming it replays B0) | full | **accepted** |
| T083 | `receipt.deleted` | none | **accepted** |
| T084 | `receipt-source.naive`, `receipt-source.self` | none / local | `Invalid retained energy replay receipt` |
| T084 | `receipt-source.replay-id` | local | `Retained energy analysis binding differs` |
| T084 | `receipt-source.subject-rebound`, `receipt-source.other-source` | local | `Replay source must already belong to this workbench` |
| T084 | `receipt-source.sibling-execution` (source and subject moved together) | local | **accepted** |
| T085 | `receipt-replayed.{naive,replay-id,source,sibling}` | none / local | `Invalid retained energy replay receipt` |
| T085 | `receipt-replayed.reidentified-bundle` (dated 2001, before its source; new session; all digests recomputed) | full | **accepted** |
| T086 | `receipt-subject.naive` | none | `Invalid retained energy replay receipt` |
| T086 | `receipt-subject.resealed`, `bundle-subject.resealed` | local | `Retained energy analysis binding differs` |
| T086 | `oscillator-verification.resealed` | local | `Protocol v1 saved results must remain not_verified with verification_id null` |
| T086 | `oscillator-subject.injected` (`subject_ref` naming R2 injected into R1) | local | **accepted** |
| T086 | `esm.bundle-subject` (pure validator) | none | `ESM candidate does not bind the selected native bundle` |
| T086 | `exchange.verification-subject` (pure validator) | none | `verification_id does not match the artifact content` |
| T087 | `receipt-method.naive` | none | `Invalid retained energy replay receipt` |
| T087 | `receipt-method.resealed`, `bundle-method.resealed` | local | `Retained energy analysis binding differs` |
| T087 | `oscillator-method.injected` | local | **accepted** |
| T088 | `receipt-independent.naive` | none | `Invalid retained energy replay receipt` |
| T088 | `receipt-independent.resealed`, `bundle-independent.resealed` | local | `Retained energy analysis binding differs` |
| T088 | `esm.inspection-independent` (pure validator) | none | `Native ESM inspection binding or scope mismatch` |
| T088 | `esm.candidate-independent` (pure validator) | none | `ESM candidate does not bind the selected native bundle` |
| T088 | `oscillator-independent.injected` | local | **accepted** |
| T088 | `exchange.verification-independent` (pure validator) | local | accepted by design (`content_recomputed_not_authenticated`) |
| T089 | `receipt-admission.{naive,resealed}` | none / local | `Invalid retained energy replay receipt` |
| T089 | `bundle-authority.resealed`, `result-authority.reforged` | local / full | `Retained energy analysis binding differs` |
| T089 | `oscillator-status.resealed` | local | `Protocol v1 saved results must remain not_verified with verification_id null` |
| T089 | `esm.canonical-admission` (pure validator) | none | `ESM may retain candidate evidence only` |
| T089 | `esm.candidate-admitted` (pure validator) | none | `ESM candidate does not bind the selected native bundle` |
| T089 | `oscillator-admission.injected` | local | **accepted** |
| T090 | `oscillator-runtime.naive` | none | `Operation record integrity mismatch` |
| T090 | `oscillator-runtime.execution-only` | local | `Execution/result runtime binding mismatch` |
| T090 | `oscillator-runtime.empty` | local | `Invalid execution runtime identity` |
| T090 | `energy-runtime.naive` | none | `Energy analysis bundle identity, schema or size differs` |
| T090 | `energy-runtime.replay-only` | full | `Retained energy analysis binding differs` |
| T090 | `energy-runtime.malformed` | full | `Invalid retained analysis implementation identity` |
| T090 | `oscillator-runtime.both` | local | **accepted** |
| T090 | `energy-runtime.all-bundles`, `energy-runtime.python-version` | full | **accepted** (a later `bundle.replay` refuses: `Retained energy analysis binding differs`) |

The ESM rows use a synthetic telemetry-shaped record: a schema, a digest and
three step occurrences. `Workbench._validate_candidate` submits only calibrated
or telemetry bundles, and a validated telemetry session needs provider
checkouts, so no telemetry workflow ever validated this record.

### What the kills show

On reopen, the energy-accuracy path re-runs the analysis on the retained source
bytes. It rebuilds the expected verification record from its context and
compares it field by field. What that enforces is narrower than "every
verification edit is refused":

- **Fixed fields.** Method, independence (`independent: false`), authority and
  admission (`not_performed`) are constants in the rebuilt record. Any edit to
  them is refused, even when every unkeyed digest is recomputed.
- **Subjects.** A bundle's own verification subject must equal that bundle's
  digest. A receipt's verification subject must equal the receipt's source
  digest. That source must be a retained bundle with the same kind, `source_id`
  and upstream.
- **Moving subject and source together.** The subject and the source can be
  moved together to any retained bundle of the same source. That is the
  surviving mutant `receipt-source.sibling-execution`.
- **Other refusals.** Replay freshness, cross-namespace collisions, and edits
  to analysed numbers that leave the source bytes unchanged are refused. A
  receipt moved onto another bundle is refused only while its replayed digest is
  stale or its verification still binds the donor's reproduction step. Rebuilt
  from the two bundles, it survives (`receipt.transplanted-full`), and so does
  a receipt written for a replay that never happened (`receipt.fabricated`).
  Nothing binds a receipt to a replay event: every receipt field is a copy of,
  or an unkeyed digest over, retained bundles.
- **Oscillator aliases.** A result or execution that names another occurrence's
  identity is refused. Two occurrences whose identities and creation times are
  exchanged together (`alias.swap-pairing`) reopen, because oscillator
  identities are uuid4 draws that bind no content. This is harmless here only
  because R1 and R2 carry equal data.

Re-analysis protects the numbers only relative to the retained source bytes,
and those bytes are not authenticated. A forger can edit one counter sample in
the retained log, reseal the author's unkeyed `log_digest`, and rebuild the
source record, the evidence, experiment and request digests, the analysis, the
steps, the bundles, the verifications and the receipt. The occurrence ids and
creation times are kept. This is `energy-source.resealed`. The workspace
reopens, and the three baseline bundles report a gross energy of 0.25 J instead
of the 0.2 J computed from the original bytes, under their original session
ids. The shared-`run_id` refusal does not apply, because the forger replaces
the source instead of adding a second one beside it.

### Surviving mutants and their causes

| Survivor | What a reader of the reopened workspace sees | Cause |
| --- | --- | --- |
| `energy-source.resealed` | B0, B0b and B1 report a gross energy of 0.25 J instead of 0.2 J, under their original session ids and times | The retained log is protected only by its author's unkeyed `log_digest`; reopen re-analyses whatever bytes are retained |
| `receipt-source.sibling-execution` | B1 claims to replay B0b, with its verification subject also B0b, although it replayed B0 | Any retained bundle with the same source bytes, configuration and runtime satisfies the receipt checks; the receipt is consistent, not authenticated |
| `receipt-replayed.reidentified-bundle` | The replay is dated 2001, before its source (created at run time), and has a new `session_id` | `created_at` and `session_id` sit inside an unkeyed `bundle_digest`, which the forger recomputes along with the receipt; no check compares a replay's time with its source's |
| `receipt.deleted` | B1 is listed with no receipt, like an original execution | `_bundle_digest` excludes `replay_receipts`, and nothing else records that the replay happened |
| `receipt.transplanted-full` | B0b, an original execution, claims to replay B0; the actual replay B1 has no receipt | Every receipt field (source and replayed digests, verification with the containing bundle's own step, `replay_id`) is computable from retained bundles |
| `receipt.fabricated` | B0b claims to replay B0 although it was never replayed; B1 keeps its genuine receipt | As for the transplant: a sibling of the same bytes, configuration and runtime satisfies every receipt check, and no record of replay events exists |
| `alias.swap-pairing` | R1 names E2 as its execution and R2 names E1 | Oscillator execution and result identities are uuid4 draws; reopen checks only that each pair agrees (ids, creation time, runtime), and both pairs are resealed |
| `oscillator-stats.resealed` | `statistics.v1` mean moved to the interval midpoint | `validate_payload` checks bounds only, and the seal is unkeyed |
| `oscillator-stats.impossible-moments` | R1 has \|mean\| > rms; R2 has rms greater than max(\|min\|, \|max\|); no sample set has these moments | `validate_payload` does not check \|mean\| ≤ rms ≤ max(\|min\|, \|max\|). The computed result satisfies both inequalities and matches the lab's direct NumPy recomputation from the 64 samples (relative difference 0 here; a same-origin `cross_implementation` check) |
| `oscillator-stats.legacy` | Legacy result mean edited | Legacy `analysis.stats` results have no seal at all |
| `fresh.created-at-shift` | Execution and result backdated to 2001 | `created_at` is bound only between the two records, and both are resealed |
| `revision.gap` | Records claim selection revision 999 of 1000 | The selection history is not saved; `selection_revision` is only checked to be at most the current revision |
| `oscillator-{subject,method,independent,admission}.injected` | `result.get` returns `subject_ref` (naming R2), `verification_method`, `independent: true` or `state_admission: admitted`, next to `verification_status: not_verified` | `_validate_saved_result` requires certain keys but does not refuse extra keys, and `validate_execution` does not check its key set |
| `oscillator-runtime.both` | Execution and result name `lab.forged-provider` version 99 | The runtime identity is provider-declared; no check exists beyond equality between execution and result |
| `energy-runtime.all-bundles`, `energy-runtime.python-version` | Bundles claim different analysis code or a different Python version | Reopen checks only the runtime's format. Only a new replay compares it with the current `analysis_identity()`. The retained runtime is historical provenance that no unkeyed check can verify |

One acceptance is **not** counted as a survivor. `exchange._identity` accepts a
recomputed verification artifact that claims `independent: true`. It returns
`content_recomputed_not_authenticated`, which is its documented behaviour:
`ciw.exchange` states that a hash does not establish verification authority,
and `inspect_exchange` always reports `verification_independence:
not_established`. T088 records this as a verified finding without a
counterexample. The full inspector was not run here because it needs the
pinned validator checkout (`CIW_SET_REPO`).

Every survivor is recorded exactly once as a counterexample finding. The finding
names the refuted general statement and carries a witness read from the
reopened session. Beyond the acceptance, every survivor finding checks that the
witness shows the forged content: a changed energy, violated inequalities, a
replay dated before its source, the injected value returned by `result.get`,
the forged receipt placement, the swapped pairing, the forged revision, time or
runtime. If CIW began stripping unknown keys or dropping a forged field on
reopen, these checks would fail rather than pass on acceptance alone. Two
witnesses (`energy-source.resealed` and `oscillator-stats.impossible-moments`)
show floats from NumPy reductions or the energy analysis. Their last bits can
differ between SIMD dispatch paths, libm builds and platforms, so those two
findings use the regression tolerance `{abs: 0, rel: 1e-9}`. Every other
survivor is compared exactly. T077 and every task from T080 to T090 also record the
provenance finding *"Retained workspace records are authenticated"*. That
finding is `not_established` (declared as expected) because no keyed signature
or MAC exists on these paths. T089 adds a `production_acceptance` claim that
receipts authorize admission. T083 adds the claim that replay agreement is
verification by an independent party. T078 and T077 add physical claims about
the logs and their producer identities. All of these remain `not_established`.

## Recommended CIW changes (not made here)

1. **Keyed authentication.** Add an HMAC or Ed25519 signature over every sealed
   record, the retained source bytes and the workspace catalog, using a
   host-held key that never enters the workspace. This is the only fix that
   covers all of the survivors above, including `energy-source.resealed`. An
   unkeyed digest can only detect accidental edits, not deliberate ones. It is
   also the only way to authenticate a runtime identity: sign it on the
   producing host.
2. **Bind receipts into identity.** Include replay receipts, or their
   `replay_id`s, in a catalog-level workspace seal, or give the replay bundle a
   second digest that covers them. Also refuse a replay dated before its source.
   An unkeyed seal only stops accidental deletion or re-pointing. A forger who
   transplants or fabricates a receipt recomputes the seal too, so binding a
   receipt to an actual replay event needs the keyed signature of change 1.
3. **Closed record schemas.** Refuse unknown keys in `ciw.execution.v1` and
   `ciw.operation-result.v1` records, so that verification subject, method,
   independence or admission claims cannot be added through the seal.
4. **Stronger checks on operation results.** Add a content-level
   `numerical_result_id` to oscillator operation results. Recompute
   `statistics.v1` on reopen, which is cheap, or at least check the moment
   inequalities |mean| ≤ rms ≤ max(|min|, |max|) as well as the bounds.
5. **Seal or retire legacy results.** Either seal legacy `analysis.stats` results
   or refuse to reopen them without an explicit legacy flag.
6. **Label historical runtimes.** On reopen, label a retained energy runtime that
   differs from the current `analysis_identity()` as historical and unverified.
   Do not refuse it, because that would make every earlier workspace unopenable
   after any code change, even when the recomputed numbers agree. Do not treat
   a match as authentication either, because a forger can simply write the
   current hash.
7. **Persist selection history.** Save the selection history, or drop
   `selection_revision` from sealed records.
8. **Clearer error text.** Word the byte-order-mark refusal for sources as a
   source-bytes error.

## Deferred research questions

Three questions have no owner in the queue. Each report that depends on one
records it verbatim among its unresolved assumptions, so `ciw lab next` lists
it once, with the other tasks that raise it (`exchange_provenance_common`
holds the first two).

- **Key custody and signatures** (`KEY_CUSTODY_QUESTION`; T077, T078 and
  T080–T090). Which key signs workspace records, replay receipts and execution
  records? Who holds it, how is it provisioned, rotated and revoked, and how
  does a verifier obtain its public key? Change 1 above needs these answers
  before it can be made. Until records are signed and T077 and T080–T090 are
  re-run against them, "Retained workspace records are authenticated" stays
  `not_established`.
- **Telemetry provider stack** (`TELEMETRY_STACK_QUESTION`; T077, T086 and
  T088–T090). Provision the stack pinned in `src/ciw/telemetry-runtimes.json`
  (ppda, stfe, gsie, set, cbsr). Today `scripts/check_lab.py` provisions only
  ppda and set of these, and `TEST_VARIABLES` has no variable for stfe, gsie or
  cbsr. Then bind the stack to a telemetry or declared-workload workflow, so
  the ESM candidate rows, the candidate execution rows and the
  `ciw.subprocess-runtime.v1` rows are observed rather than read from code.
- **Cross-platform reproduction** (T081). The bit-exact `numerical_result_id`
  links occurrences on one arithmetic platform only: three OpenBLAS kernels on
  one Linux x86-64 host gave three identities. Run T081 on Windows x86-64 and
  macOS arm64, check that each run again has one identity over its
  occurrences, and compare its energy numerical result leaf by leaf with the
  `energy_numerical_result` retained in T081's `numerical-identity.json`. Do
  the differences stay at rounding level? Should CIW link occurrences across
  platforms by such a tolerance comparison, or by an identity over
  declared-precision data, instead of the bit-exact identity?

Each task's `recommended_next_task` names its own open question, usually one
of the CIW changes above (`exchange_provenance.NEXT_STEPS`). It is never the
next queue task, which has already run.

## What this does not establish

- Content identities and unkeyed seals establish consistency, not authorship,
  authenticity or time. Re-analysis on reopen protects numbers only relative to
  retained source bytes that are themselves unauthenticated.
- A killed mutant shows that one particular edit is detected. It does not show
  that every edit to that field is detected. T083 is the example: the two
  transplants it refuses differ from the surviving one only in how much the
  forger recomputes. Refusal messages are regression pins recorded from observed
  runs, not predictions.
- A surviving mutant is a demonstrated forgery. Whether it would mislead a
  particular reader depends on how the record is displayed, which was not
  assessed.
- The energy logs are synthetic fixtures with placeholder producer identities.
  Nothing here bears on real GPU energy, sensor performance, calibration,
  admission authority, or verification by another party.
- The separate session runs in the same process with the same code. It shows
  stability, not independent reproduction.
- Only the binding-free energy-accuracy workflow and the oscillator operations
  were exercised. Provider-backed workflows (telemetry, declared workloads,
  proved heat), bound ESM candidates and the full exchange inspector need
  checkouts that are not used here. The ESM validator ran on a synthetic
  telemetry-shaped record. T090 therefore mutates only the CIW-internal
  oscillator provider identity and CIW's own energy analysis identity. The
  pinned-provider subprocess runtime identities (`ciw.subprocess-runtime.v1`)
  were not mutated.
- The float-valued `numerical_result_id` is not stable across BLAS kernels
  (three OpenBLAS kernels on one host gave three identities). Other operating
  systems and BLAS libraries were not tested.
