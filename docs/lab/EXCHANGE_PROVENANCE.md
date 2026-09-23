# Exchange and provenance, part 1: identity matrix and workspace mutations (T077–T090)

Implementation: `src/ciw/lab/exchange_provenance.py` (tasks) and
`src/ciw/lab/exchange_provenance_common.py` (offline fixture and forgery
harness). Tests: `tests/test_lab_exchange_provenance.py`.

```
python -m ciw lab run T077 T078 T079 T080 T081 T082 T083 T084 T085 T086 T087 T088 T089 T090 --output-dir results/lab-exchange
python -m ciw lab report T084 --retained results/lab-exchange
python -m pytest -q tests/test_lab_exchange_provenance.py
```

These experiments test the unmodified CIW integrity layer (`ciw.session`,
`ciw.operations.runner`, `ciw.workbench`, `ciw.energy_workflow`,
`ciw.exchange`, `ciw.candidate_evidence`, `ciw.core.identities`,
`ciw.telemetry`) with no network, provider checkout, GPU or hardware. The
section runs in about 11 s and its tests in about 12 s on one core.

## What was built

One offline fixture is shared by all fourteen tasks:

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
- **Independent session.** A second session retains the same bytes under the
  same label and under a second label, then executes them (B4).
- **Byte-variant workbench.** Holds the four fixture logs plus eight byte
  variants of the baseline. Each is retained, executed, saved and reopened.

A **mutant** is a copy of the saved workspace with one edit. After the edit, the
harness recomputes *none*, the *local* digests the edit touches, or *all*
downstream digests. Every digest is an unkeyed SHA-256 over CIW's own public
canonicalization, so anyone holding the file can recompute it. The mutant is
written to disk and reopened with `Session.from_workspace`. The outcome is
either `accepted` or the exact refusal message.

Two harness controls run on every retained matrix:
- The forger's full recomputation of an unedited workspace reproduces every CIW
  digest byte for byte.
- The unedited workspace reopens.

Both controls hold.

## Identity matrix (T077)

The matrix has 28 identities. 27 are exercised offline, and all 67 predicted
properties hold. The full matrix, including the per-property results, is
retained as `artifacts/T077/identity-matrix.{json,md}`.

| Identity | Derivation | Binds | Across replay | Across reopen | Validated at |
| --- | --- | --- | --- | --- | --- |
| recording `evidence_id` | content hash: canonical JSON of instrument, metadata, time_s, channels | scientific content; excludes `run_id`, `render` | n/a | stable | `core/identities.py:validate_evidence_identity`, `session.py:_validate_evidence` |
| recording file name | content hash of the whole run (`recording-<sha256>.json`) | parsed run; the file is a CIW re-serialization, not caller bytes | n/a | stable | `session.py:_recording_file`, `_validate_saved_result` |
| workbench source `evidence_id` | **byte hash** of the exact source bytes | bytes only (not label) | stable (= bundle `artifact_ref`) | stable | `workbench.py:_source`, `Workbench.restore` |
| workbench `source_id` | content hash of `{schema, kind, label, source_schema, evidence_id, byte_count}` | bytes **and caller label** | stable | stable | `workbench.py:_source`, `_source_claims` |
| bundle `experiment_digest` | content hash of the parsed log | canonical content (whitespace and key order excluded) | stable | stable | `energy_workflow.py:EnergyAccuracyWorkflow._validate` |
| log `run_id` / `experiment_id` | caller-declared (copied from the log) | the log author's occurrence claim; bound to `log_digest` by workbench claims | stable | stable | `energy_records.py:_validate`, `workbench.py:_claims` |
| oscillator `operation_id` | caller-declared versioned name | which operation ran | n/a | stable | `operations/registry.py:valid_operation_id`, `runner.py:validate_execution` |
| energy `operation_id` | constant `ciw.energy-accuracy.v1` | which workflow ran | stable | stable | `EnergyAccuracyWorkflow._validate_step` |
| oscillator `execution_id` | fresh `execution-<uuid4>` | one occurrence | fresh | stable | `runner.py:validate_execution`, `session.py:_identity` |
| energy step `execution_id` | fresh `execution-<uuid4>` | one occurrence | fresh | stable | `_validate_step`, `workbench.py:_validate_links` |
| oscillator `result_id` | fresh `result-<uuid4>` | one occurrence; content bound only by the unkeyed seal | fresh | stable | `session.py:_validate_saved_result` |
| energy `result_id` | content hash of the result record including `execution_ref` | content and fresh occurrence | fresh | stable | `_validate_step` |
| energy `numerical_result_id` | content hash of `{operation_id, data}` | analysed numbers only | **stable** | stable | `_validate_step`, `_validate_links` |
| oscillator numerical-result identity | **absent** | nothing | absent | absent | none |
| energy `verification_id` | `sha256(schema ‖ NUL ‖ canonical)` | subject, outcome, independent, method, runtime digest, reproduction, authority | fresh | stable | `_check_verification`, `exchange.py:_identity` |
| oscillator `verification_id` | always `null`, status `not_verified` | nothing | n/a | stable | `_validate_saved_result` |
| `bundle_digest` (= workbench `bundle_id`) | content hash excluding `bundle_digest`, `verification`, `replay_receipts` | session, time, source, configuration, runtimes, steps; **not receipts** | fresh | stable | `telemetry.py:_bundle_digest`, `_validate` |
| bundle `session_id` | fresh `session-<uuid4>` | one workflow session | fresh | stable | `_validate`, `workbench.py:_claims` |
| protocol `Session.session_id` | fresh `session-<uuid4>` | live session only; never saved | n/a | fresh | `session.py:Session.__init__` |
| `replay_id` | content hash of the receipt | source/replayed digests, verification, admission | fresh | stable | `workbench.py:_validate_receipts`, `_validate` |
| oscillator `record_digest` | **unkeyed** seal over the record | whole record, recomputable by anyone | n/a | stable | `runner.py:check_seal` |
| oscillator runtime identity | provider-declared `{provider, version}` | a name and version string | n/a | never re-checked | `runner.py:validate_execution` (non-empty only) |
| energy runtime identity | sha256 of three CIW source files plus self-reported Python/NumPy versions | analysis code bytes and version strings | compared on replay | format-checked only | `energy_workflow.py:_check_runtime`, `_adapters` |
| exchange `result_id` / `verification_id` | `sha256(schema ‖ NUL ‖ canonical)` | content; status `content_recomputed_not_authenticated` | n/a | n/a | `exchange.py:_identity` |
| exchange `batch_id` | caller-declared | nothing (`caller_declared_reference`) | n/a | n/a | `exchange.py:_identity` |
| ESM `requestId` / `inspectedAt` | caller-declared | equality of request and response only | n/a | n/a | `candidate_evidence.py:validate_response` |
| ESM `bundleBytesDigest` | byte hash of the canonical bundle | exact selected bundle bytes | n/a | n/a | `validate_response` |
| ESM `candidate_id` / candidate execution | content hash / fresh uuid4 | response bytes, policy, adapter identity | n/a | stable | `workbench.py:Workbench._validate_candidate` (not exercised offline) |

### Byte-level and content-level identities (T078, T079)

The source bytes are retained exactly. For all 12 sources, the bytes returned
live, inside each native bundle, and after save and reopen equal the supplied
bytes, with zero mismatches. Non-canonical base64 is refused, never normalized:
- missing padding and line wrapping are refused with `Source bytes must use canonical base64`;
- non-canonical trailing bits are refused with `Source bytes must use bounded canonical base64`.

T079 retained eight variants of `baseline.json`. They differ only in whitespace,
CRLF line endings, key order or float spelling (`1e-09` written as
`0.000000001`), so their parsed content is canonically equal:

| Identity | Level | Distinct values over 8 variants |
| --- | --- | --- |
| workbench `evidence_id`, bundle `artifact_ref`/`sha256` | byte | 8 |
| workbench `source_id` | byte + label | 8 |
| bundle `experiment_digest`, sealed `log_digest`, `numerical_result_id` | canonical content | 1 |
| recording `evidence_id` (oscillator) | canonical content of scientific fields | n/a |
| `bundle_digest`, energy `result_id`, `verification_id`, `replay_id` | canonical content including fresh occurrences | n/a |

Two boundary findings from the same experiment:
- **Numeric type is part of canonical content.** Rewriting the first `1.0,` as
  `1,` leaves the value numerically equal (Python `==` holds), but the canonical
  JSON differs. The re-encoding is refused with `Workload solver differs from
  plan` rather than being aliased to the original.
- **A byte-order mark is refused.** The BOM-prefixed variant is refused with
  `The bound runtime did not return finite, unambiguous JSON`. The refusal is
  correct, but the message wrongly blames a "bound runtime" for a problem in the
  source bytes.

## Mutation matrix (T080–T090)

There are 67 mutants: 60 edit and reopen a saved workspace, and 7 run pure
validators. 52 are refused and 15 survive. Every outcome, including every
refusal message, matched its prediction. The full matrix, with a post-reopen
witness for each survivor, is retained as
`artifacts/T090/mutation-matrix.{json,md}`. Each task also retains its own rows.

| Task | Mutant | Recompute | Outcome (reopen; validator call for pure-validator rows) |
| --- | --- | --- | --- |
| T080 | `alias.result-execution` (R2 points at E1) | local | `Saved result identity mismatch or duplication` |
| T080 | `alias.execution-result` (E2 points at R1) | local | `Execution/result execution_id binding mismatch` |
| T080 | `alias.result-prefix` | local | `Invalid saved result identity` |
| T080 | `alias.operation` (statistics → spectrum) | local | `Invalid saved spectrum data fields or sample count` |
| T080 | `revision.gap` (records claim revision 999 of 1000) | local | **accepted** |
| T081 | `energy-data.naive` | none | `Energy analysis bundle identity, schema or size differs` |
| T081 | `energy-data.reforged` | full | `Retained energy analysis binding differs` |
| T081 | `oscillator-stats.naive` | none | `Operation record integrity mismatch` |
| T081 | `oscillator-stats.out-of-bounds` | local | `Saved statistics mean is outside its bounds` |
| T081 | `oscillator-stats.resealed` (mean moved inside its bounds) | local | **accepted** |
| T081 | `oscillator-stats.legacy` (unsealed legacy result) | none | **accepted** |
| T082 | `fresh.energy-replay-reuse`, `fresh.energy-reproduction-reuse` | full | `Declared workload bundles must have distinct execution and reproduction occurrences` |
| T082 | `fresh.cross-namespace` (energy step reuses an oscillator execution id) | full | `Identity collision between recording operations and retained workflows` |
| T082 | `fresh.oscillator-duplicate` | local | `Saved result identity mismatch or duplication` |
| T082 | `fresh.created-at-one` | local | `Execution/result created_at binding mismatch` |
| T082 | `fresh.created-at-shift` (backdated in both records) | local | **accepted** |
| T083 | `receipt.numerical-match-false`, `receipt.transplanted` | local / none | `Invalid retained energy replay receipt` |
| T083 | `receipt.transplanted-resealed` | local | `Retained energy analysis binding differs` |
| T083 | `receipt.deleted` | none | **accepted** |
| T084 | `receipt-source.naive`, `receipt-source.self` | none / local | `Invalid retained energy replay receipt` |
| T084 | `receipt-source.replay-id` | local | `Retained energy analysis binding differs` |
| T084 | `receipt-source.subject-rebound`, `receipt-source.other-source` | local | `Replay source must already belong to this workbench` |
| T084 | `receipt-source.sibling-execution` | local | **accepted** |
| T085 | `receipt-replayed.{naive,replay-id,source,sibling}` | none / local | `Invalid retained energy replay receipt` |
| T085 | `receipt-replayed.reidentified-bundle` (backdated, new session, all digests recomputed) | full | **accepted** |
| T086 | `receipt-subject.naive` | none | `Invalid retained energy replay receipt` |
| T086 | `receipt-subject.resealed`, `bundle-subject.resealed` | local | `Retained energy analysis binding differs` |
| T086 | `oscillator-verification.resealed` | local | `Protocol v1 saved results must remain not_verified with verification_id null` |
| T086 | `esm.bundle-subject` (pure validator) | none | `ESM candidate does not bind the selected native bundle` |
| T086 | `exchange.verification-subject` (pure validator) | none | `verification_id does not match the artifact content` |
| T086 | `receipt-subject.rebound-with-source` | local | **accepted** |
| T087 | `receipt-method.naive` | none | `Invalid retained energy replay receipt` |
| T087 | `receipt-method.resealed`, `bundle-method.resealed` | local | `Retained energy analysis binding differs` |
| T087 | `oscillator-method.injected` | local | **accepted** |
| T088 | `receipt-independent.naive` | none | `Invalid retained energy replay receipt` |
| T088 | `receipt-independent.resealed`, `bundle-independent.resealed` | local | `Retained energy analysis binding differs` |
| T088 | `esm.inspection-independent` (pure validator) | none | `Native ESM inspection binding or scope mismatch` |
| T088 | `esm.candidate-independent` (pure validator) | none | `ESM candidate does not bind the selected native bundle` |
| T088 | `oscillator-independent.injected` | local | **accepted** |
| T088 | `exchange.verification-independent` (pure validator, identity layer) | local | **accepted** (`content_recomputed_not_authenticated`) |
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

### What the kills show

The energy-accuracy path is strict about semantics. On reopen it re-runs the
analysis on the retained bytes, and it rebuilds the expected verification record
from its context and compares it field by field. Because of this, every edit to
a verification's subject, method, independence flag, authority or admission is
refused. The refusal holds even when every unkeyed digest is recomputed, since
the rebuilt record fixes `independent: false`, the method, and the `not_performed`
authority. Replay freshness, cross-namespace collisions, receipt transplants and
edits to the analysed numbers are also refused.

### Surviving mutants and their causes

| Survivor | What a reader of the reopened workspace sees | Cause |
| --- | --- | --- |
| `receipt-source.sibling-execution` / `receipt-subject.rebound-with-source` | B1 claims to replay B0b, although it replayed B0 | Any retained bundle of the same source bytes, configuration and runtime satisfies the receipt checks; the receipt is consistent, not authenticated |
| `receipt-replayed.reidentified-bundle` | The replay bundle has a backdated `created_at` and a new `session_id` | `created_at` and `session_id` sit inside an unkeyed `bundle_digest`, which the forger recomputes together with the receipt |
| `receipt.deleted` | B1 appears to be an original execution, with no replay recorded | `_bundle_digest` excludes `replay_receipts`, and nothing else records that the replay happened |
| `oscillator-stats.resealed` | `statistics.v1` mean moved to the interval midpoint | `validate_payload` checks bounds only, and the seal is unkeyed |
| `oscillator-stats.legacy` | Legacy result mean edited | Legacy `analysis.stats` results have no seal at all |
| `fresh.created-at-shift` | Execution and result backdated to 2001 | `created_at` is bound only between the two records, and both are resealed |
| `revision.gap` | Records claim selection revision 999 of 1000 | The selection history is not saved; `selection_revision` is only checked to be at most the current revision |
| `oscillator-{method,independent,admission}.injected` | `result.get` returns `verification_method`, `independent: true` or `state_admission: admitted`, next to `verification_status: not_verified` | `_validate_saved_result` requires certain keys but does not refuse extra keys, and `validate_execution` does not check its key set |
| `oscillator-runtime.both` | Execution and result name `lab.forged-provider` version 99 | The runtime identity is provider-declared; no check exists beyond equality between execution and result |
| `energy-runtime.all-bundles`, `energy-runtime.python-version` | Bundles claim different analysis code or Python version | Reopen only checks the runtime's format. Only a replay compares it with the current `analysis_identity()`. The numbers themselves are recomputed on reopen by the current code, so the forged claim concerns provenance, not the values |
| `exchange.verification-independent` | An exchange verification artifact claiming `independent: true` passes the identity check | `_identity` only recomputes content (`content_recomputed_not_authenticated`). The inspector's fixed authority block still reports `verification_independence: not_established`; the full inspector was not run here because it needs the pinned validator checkout |

Every survivor is recorded as a counterexample finding: the finding names the
refuted general statement and carries a witness. T077 and every task from T080
to T090 also record the provenance finding *"Retained workspace records are
authenticated"*. That
finding is `not_established` (declared as expected) because no keyed signature
or MAC exists on these paths. T089 adds a `production_acceptance` claim that
receipts authorize admission. T083 adds the claim that replay agreement is
verification by an independent party. Both remain `not_established`.

## Recommended CIW changes (not made here)

1. **Bind receipts into identity.** Include replay receipts, or their
   `replay_id`s, in a catalog-level workspace seal, or give the replay bundle a
   second digest that covers them. Deleting or re-pointing a receipt would then
   no longer reopen silently.
2. **Keyed authentication.** Add an HMAC or Ed25519 signature over every sealed
   record and workspace catalog, using a host-held key that never enters the
   workspace. This is the only fix that covers all of the survivors above. An
   unkeyed digest can only detect accidental edits, not deliberate ones.
3. **Closed record schemas.** Refuse unknown keys in `ciw.execution.v1` and
   `ciw.operation-result.v1` records, so that verification, independence or
   admission claims cannot be added through the seal.
4. **A numerical identity for operation results.** Add a content-level
   `numerical_result_id` to oscillator operation results. Optionally recompute
   `statistics.v1` on reopen, which is cheap, instead of checking bounds only.
5. **Seal or retire legacy results.** Either seal legacy `analysis.stats` results
   or refuse to reopen them without an explicit legacy flag.
6. **Check the runtime identity on reopen.** Compare the energy runtime
   identity's `code_sha256` with the current `analysis_identity()` on reopen, as
   replay already does, or label a mismatch explicitly as historical.
7. **Persist selection history.** Save the selection history, or drop
   `selection_revision` from sealed records.
8. **Clearer error text.** Word the byte-order-mark refusal for sources as a
   source-bytes error.

## What this does not establish

- Content identities and unkeyed seals establish consistency, not authorship,
  authenticity or time.
- A killed mutant shows that one particular edit is detected. It does not show
  that every edit to that field is detected.
- A surviving mutant is a demonstrated forgery. Whether it would mislead a
  particular reader depends on how the record is displayed, which was not
  assessed.
- The energy logs are synthetic fixtures. Nothing here bears on real GPU
  energy, sensor performance, calibration, admission authority, or verification
  by another party.
- Only the binding-free energy-accuracy workflow and the oscillator operations
  were exercised. Provider-backed workflows (telemetry, declared workloads,
  proved heat) and bound ESM candidates need checkouts that are not used here.
- Stability of the float-valued `numerical_result_id` across platforms and BLAS
  builds was not tested.
