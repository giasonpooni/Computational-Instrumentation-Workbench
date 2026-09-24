# PPDA and STFE inside the shared workbench

The existing `ciw.telemetry.v1` operation now runs through the live session:
PPDA observation projection → STFE causal window mean → GSIE prediction/update
→ optional CBSR reconciliation → SET replay verification. The native numerical
contracts and provider pins remain unchanged. Exact retained observations,
features and state now share the process workbench's source, instrument, result,
execution, replay and ESM interfaces.

```sh
ciw serve --identified-stack-root /trusted/process-providers \
  --telemetry-stack-root /trusted/telemetry-providers \
  --esm-binding /trusted/esm-process.json \
  --esm-telemetry-binding /trusted/esm-telemetry.json
python examples/shared-workbench/run.py --with-design --with-telemetry
```

The telemetry root contains role-named `ppda`, `stfe`, `gsie`, `set`, `cbsr`
checkouts at the `telemetry` descriptor pins pins. Keep it separate from the process
root: the two profiles use different reviewed GSIE/CBSR/SET revisions. CLI startup
binds all five telemetry roles. Embedding applications may omit CBSR; requesting
reconciliation then refuses until it is explicitly bound. Unrequested CBSR never
executes. Existing standalone commands remain supported.

Each ESM file uses the existing [operator binding](STATE_DIAGNOSTICS_EVIDENCE.md#operator-setup)
format with its own exact provider pins and evidence-to-policy mappings. ESM's
bridge and replay CIW pins are unchanged. Requests select a bundle, never code
paths, source rights or stores. `operation.list` reports `available_bundle_kinds`.
Rebind a family when its operator-known policies or withdrawals change.

| Shared request | Explicit input | Native output |
| --- | --- | --- |
| `source.add` | `kind: telemetry`, label, base64 `ciw.telemetry-source.v1` bytes | Exact retained source identity |
| `operation.execute` | `operation_id: ciw.telemetry.v1`, parameters `{source_id, configuration}` | Verified native telemetry bundle |
| `experiment.inspect` | `{view: instrument, bundle_id, instrument: ppda or stfe}` | Original batch or complete window/quality/feature receipt |
| `bundle.replay` | Selected bundle ID | Fresh executions and replay receipt |
| ESM inspect/capture operations | Selected telemetry bundle and explicit action/time IDs | Fresh replay/policy decision or UNADMITTED retained evidence |

Use the complete configuration in `examples/telemetry/configuration.json`.
It declares the window, prior, dynamics, likelihood, units/frame and prior-feature
independence. The native bundle retains that configuration alongside exact source
bytes. Source bytes can support separately declared operations without becoming
new observations. Failed operations retain their source, but no partial state.

Replay keeps the same PPDA batch identity and records a fresh projection execution.
STFE/GSIE/CBSR result occurrences are fresh; numerical identity remains separately
comparable. The batch appears once in `result.list`, marked
`identity_kind: observation_batch` with `batch_id`; its generic `result_id` selector
names that same batch. `result.get` returns the unmodified native record.
`execution.list` preserves every occurrence. A replay is not another independent
sensor. Workspace restore retains these identities and historical ESM receipts,
without executable bindings or current eligibility.

The correlated example has mean 3 and variance
`0.625 = (1 + .25 + .25 + 1)/4`. With declared independent prior mean 0/variance 1,
the posterior mean is `24/13` and variance `5/13`. Actual-provider tests check
these analytic values, the full covariance and unchanged raw timestamps.

This is retained acquisition projection, not live sensor polling. Only fully
declared identity time/frame maps are supported. Calibration references are not
applied or validated. the `fusion` view of `experiment.inspect` marks the context `window_feature_posterior`,
observability `unresolved` (`not_evaluated_by_telemetry_profile`), calibration
validity `not_assessed`, and fault assessment `not_run`. It cannot substitute for
the calibrated prior required by identified observation design. Unknown
cross-covariance, incomplete/late windows and undeclared calibration transforms
are refused. General TBRT/MCUR/STFE composition remains a separate scientific
extension requiring explicit transform/window compatibility and fixtures.

`scripts/check_workbench_candidates.py` now runs both native families from an
installed wheel, tests same-session coexistence and actual ESM disk retention,
and rejects skipped integration tests. Existing ICRH `telemetry-to-state.v1` and
`telemetry-reconciled.v1` profiles still cover the unchanged numerical chain.
The [native telemetry guide](TELEMETRY.md) retains the complete scientific scope.
