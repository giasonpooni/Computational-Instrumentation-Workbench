# One workbench: GSIE, CBSR, FDIR and ESM

The calibrated two-channel workflow is the numerical entry point. GSIE owns the
posterior and its full covariance. CBSR consumes that exact result and retains
its accepted/held/refused reconciliation. FDIR consumes the declared residual
and innovation covariance, never another estimator. ESM consumes the retained
calibrated bundle through fresh replay and SET verification.

## Shared protocol

| Request | Payload | Result |
| --- | --- | --- |
| `instrument.list` | `{}` | Native GSIE/CBSR/FDIR occurrences and bundle/result references |
| `instrument.inspect` | `bundle_id`, `instrument` (`gsie`, `cbsr`, `fdir`) | Exact native step, shared fusion context, linked result identities |
| `operation.execute` | `operation_id: esm.inspect-candidate.v1`, `parameters: {bundle_id, inspected_at}` | Fresh replay/policy inspection; no evidence-store write |
| `operation.execute` | `operation_id: esm.capture-candidate.v1`, `parameters: {bundle_id, evidence_id, workflow_id, retained_at}` | Fresh inspection and explicit candidate-evidence retention |
| `candidate.list` | `{}` | Historical candidate-action receipts |
| `candidate.get` | `candidate_id` | Exact ESM response bytes, policy snapshot and receipt |

Instrument inspection is a view, not numerical execution. In identified-design
bundles GSIE is a conditional prediction; CBSR/FDIR inspection refuses. Select
the upstream calibrated bundle explicitly instead of attributing prior residual
tests to a forecast. ESM's first adapter accepts calibrated bundles only.
Canonical bundle bytes, including exact base64 source bytes, form the explicit
transport commitment; original imported bundle JSON whitespace is not claimed.

Receipts stay distinct from scientific results, execution IDs, verification IDs
and evidence digests. Workspace format 3 supports retained-workbench v1 and v2;
v2 adds candidate actions. Restore validates content and links, never restores
executable bindings, authenticates historical receipts, rechecks physical
storage or establishes current eligibility. `eligibility: historical_receipt_only`
requires a fresh inspection for a current decision.

ESM actions also appear in `execution.list`, with `candidate_id` rather than a
fabricated numerical `result_id`. The existing ICRH calibrated-observable profile
continues to cover the unchanged numerical instrument chain; these additions are
shared-session views and an evidence-retention boundary, not new estimators.

Held/refused CBSR outcomes and ambiguous FDIR assessments may be retained as
evidence of those outcomes, never promoted to accepted state or unique isolation.
ESM retains `UNADMITTED` envelopes and refuses canonical admission, state mutation,
release activation and source-truth claims. Unknown cross-covariance stays explicit.

## Operator setup

Check out the ESM revision in `src/ciw/esm-runtime.json`, then run
`npm ci --ignore-scripts && npm run instrument:workbench:build` in ESM. CIW checks
artifact/helper SHA256 before and after each action. The Node executable hash is
operator-declared; NODE_OPTIONS/NODE_PATH are removed. Hashes detect drift; they
do not sandbox trusted runtimes or certify installed dependency binaries.

Supply a trusted local binding JSON:

| Field | Operator-owned value |
| --- | --- |
| `node`, `node_sha256` | Absolute Node executable and SHA256 hex |
| `artifact` | Absolute ESM `.stamp/workbench-candidate.mjs` |
| `runtime` | ESM `{python, pythonSha256, helperPath, repositories}` |
| `runtime.repositories` | CIW plus exactly eight calibrated providers, each `{path, revision}` |
| `review_context` | ESM `{requestId, authority, purpose, sources, retractions}` |
| `store_root`, `capture_registration` (optional pair) | Absolute file-store root and separate derived-source registration |

Replay CIW is separately pinned to the scientific lane, not inferred from the
host or client. All replay checkouts must be clean, complete and exact-source
pinned. Policies must bind the selected bundle's exact evidence references and
digests. No default license, retention rights, authority or complete withdrawal
history is inferred. Rebind/restart with updated operator context when policies
or known retractions change. Caller times are declared instants (at most three
fractional digits), not a trusted assertion of wall-clock currentness.

```sh
ciw serve --calibrated-stack-root /trusted/providers --esm-binding /trusted/esm-binding.json
ciw send operation.execute --timeout 360 --payload-file request.json
```

Only explicit capture writes ESM evidence. Inspection appends a historical
workbench receipt, but no evidence-store object. This is an operator-authorized
local service: source policies do not replace transport access control; do not
expose it to untrusted clients. Timeout/readback failures may leave evidence
bytes without a successful receipt; no rollback is implied. ESM owns readback.

`python scripts/check_workbench_candidates.py` tests an installed CIW wheel,
real providers/ESM, live WebSocket inspection, disk retention, withdrawal,
restoration, tamper and code-pin refusal. The gate rejects skipped cases.
