# Live experiment inspection

The Godot **Workbench** tab displays the same retained workbench used by the
terminal. Selecting one bundle drives measurement, state, uncertainty and
residual panels, the native instrument dependency tree, and evidence inspection.
The **Oscillator** tab keeps its existing playback and analysis controls.

## Run

Install CIW and Godot 4.5.2 as described in the [quickstart](quickstart.md). Bind
the clean, exact provider checkouts from the existing operation manifests:

```sh
python -m ciw serve --calibrated-stack-root /path/to/process-providers \
  --calibrated-window-stack-root /path/to/window-providers \
  --output-dir results/shared-workbench
```

In another terminal, then open the desktop client:

```sh
python examples/shared-workbench/run.py --with-calibrated-window
godot --path godot
```

The process and window stacks have different GSIE/SET pins; use their respective
manifests. Existing `--telemetry-stack-root` and `--identified-stack-root` bindings
also work. Loading an existing workspace is sufficient for inspection:

```sh
python -m ciw serve --workspace results/shared-workbench/workspace.json \
  --output-dir results/inspection
```

Reopening does not bind executable providers. The sidebar distinguishes retained
sources from currently bound operations. Execution and explicit replay continue
through the existing terminal/protocol operations. A newly committed bundle
appears in the running desktop view; **Follow new results** selects the newest
occurrence. Clicking an older occurrence disables following. This selection is
local presentation state and does not alter the scientific or oscillator selection.

## Panels and identities

| Retained workflow | Available scientific panels |
| --- | --- |
| Calibrated two-reservoir process | MCUR measurements, GSIE state, prior innovation with its declared covariance, posterior residual, accepted CBSR candidate and constraint residual |
| Calibrated window | Device indications, TBRT aligned event times, MCUR calibrated samples with full temporal/time-value covariance, STFE mean, GSIE state and residuals |
| Legacy retained telemetry | PPDA measurements, STFE mean, GSIE state and residuals; optional accepted CBSR candidate |
| Identified observation design | GSIE conditional prediction; model, candidate assessment and budget decisions remain available in native results and context |
| SRA schematic assessment | Declared node/edge tree; native eligibility, stale certificates and retrieval in object context |
| SCR integer diffusion | Initial and final integer fields, unit `1`, null covariance, native specification and execution commitments |
| SRA/JSPT/PLSR companions | Selected graph, native call events, before/after eligibility and local model scope |
| CSE BIM quantity | Full prior/posterior quantities and covariance, held/refused status, invariants and replayed ledger |
| PPDA acquired dataset | Native evidence counts, checkpoint transitions, acquisition outcomes and durable-pool identity |
| Acquired calibrated window | Existing calibrated-window panels plus exact PPDA row/record/document lineage and separate mapping/native verification |
| FDIR/OIT residual sequence | Retained innovations, normalized residuals, native CUSUM transitions and declared thresholds; per-window covariance, observability and ambiguous isolation remain in context |
| RCI/FSRT/JSPT measurement chain | Raw/calibrated evidence, native estimate and eligible reconciliation, full source/mapped covariance and constraint diagnostics; inner FSRT/JSPT results are independently selectable |
| GTE circle geometry | Observed and projected coordinates, eligible versus held candidates and radial residuals; full native joint/tangent covariance and basis remain in context |
| Identified PLSR stability | Selected GSIE prediction and covariance context, supplied certificate, native quadratic value/decrease and verdict; unknown model uncertainty and inconclusive status remain explicit |

The [declared workloads guide](DECLARED_WORKLOADS.md) provides SRA/SCR startup
bindings and examples. These objects have `fusion_context: null`, a typed
`object_context`, and `raw_declaration` rather than fabricated observations.
They use the same live invalidation and occurrence selection controls.
See [new module bindings](INTEGRATED_MODULES.md) and the
[measurement-chain, geometry and stability bindings](REMAINING_MODULES.md).
These three operations use typed object context and retain their native records;
they do not create another fusion context or feed a diagnostic back as a sensor.
Source-only geographic context
is inspected through GSV's read-only `/spatial` endpoint in the same session.
The [acquired stream guide](ACQUIRED_STREAM.md) runs acquisition, three calibrated
windows and residual monitoring through the same live server. Residual sequence
panels have no joint covariance or cross-window confidence bars: temporal dependence
is unknown. OIT-held windows do not advance the CUSUM state. Native threshold
crossings remain diagnostic candidates, with physical drift and alarm probabilities
unestablished.

Every panel carries source/evidence and, where applicable, result/execution IDs.
The dependency tree uses native `input_refs`, with external references explicitly
identified. Selecting an instrument fetches its retained result via `result.get`.
Context, Raw and Verification show the selected bundle's corresponding records.
Replay occurrences preserve their own execution/result identities; they are never
appended as additional physical measurements or joined into a time trajectory.

Points use the declared row order. Numeric time coordinates, clock, epoch, frame
and units remain in panel context. Plots use categorical spacing, draw no connecting
lines and perform no interpolation. Error bars show marginal one-standard-deviation
ranges from the retained covariance diagonal, **not** joint confidence regions.
The full matrix stays visible, including off-diagonal and time/value terms.
Mixed units use the numeric table instead of a common plot scale. The posterior
residual has no supplied covariance in these profiles; the view does not reuse
the prior innovation covariance for it. Held/refused CBSR records do not create
a panel labeled as an accepted reconciled state.

The display may format floating-point values; exact source bytes and sealed native
artifacts remain in CIW. JSON displayed by Godot is an inspection representation,
not an artifact export to hash or replay.

## Read-only projection protocol

```sh
python -m ciw send experiment.inspect --payload '{"bundle_id":"<retained-bundle-id>"}'
```

`ciw.experiment-view.v1` returns the selected bundle/source/evidence identities,
catalog revision, upstream and replay-source links, native fusion context,
panels, operation/input graph, raw observations, runtime records and retained
verification. Each panel has `panel_id`, `title`, ordered `labels`, `values`,
`units`, full `covariance` or null, `marginal_standard_deviation`, `provenance`
and `context`. Unknown IDs or additional request fields are refused. Inspection
holds the catalog lock for a consistent projection and returns independent copies;
it does not execute providers, mutate records, issue a new verification, admit
canonical state or serialize a new workspace format.

`workbench.changed` invalidates the catalog. The client coalesces refresh requests
while remembering events received during an outstanding read. Catalog revisions
cannot regress within a session. Experiment/artifact reads allow one outstanding
request each and reject obsolete responses after a selection change. Reconnect
refreshes authoritative state; disconnects retain the visible data with **STALE**
status. A changed session identity clears old panels.

## Validation and scope

Python integration cases exercise real pinned TBRT/MCUR/STFE/GSIE/SET outputs,
two-reservoir GSIE/CBSR/FDIR records and PPDA telemetry. They check exact projected
values/covariance, native dependency links, distinct replay occurrences, unchanged
retention, unbound workspace restore and refusal of mutating inspection requests.
The existing installed-window, identified-design and candidate-evidence gates run
these cases. No new scientific operation or ICRH numerical profile is introduced
by a read-only view.

```sh
python scripts/check_godot.py --godot /path/to/godot
```

This checks import, two real WebSocket clients, immediate workbench invalidation
without waiting for heartbeat, unchanged oscillator behavior, selection/reply
races, explicit following, replay separation and stale/disconnect behavior.
`godot/tests/capture_view.gd` can capture a running retained experiment with a
graphics driver for visual inspection.
`godot --headless --path godot --script res://tests/retained_views.gd -- view.json`
checks every panel of actual `experiment.inspect` projections, including the
required typed context used by the desktop. Multiple projection paths are accepted.

This is event-driven visualization of **committed bounded experiments**, not
continuous physical acquisition. No cross-bundle averaging, uncertainty reduction,
unannounced calibration, sample ordering or estimator update is performed by the
view. Arbitrary algebra/topology objects, live sensor scheduling, spatial GSV/CSE
panels and authoring scientific operations in the desktop remain future assembly
work.
