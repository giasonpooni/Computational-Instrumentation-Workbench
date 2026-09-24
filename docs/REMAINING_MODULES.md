# Native measurement, geometry and identified stability in one bench

Install CIW with the native model-loader extra and start one server with clean
provider checkouts at the exact committed pins:

```bash
python -m pip install '.[bench-models]'
ciw serve --port 8765 \
  --identified-stack-root /trusted/identified \
  --measurement-chain-stack-root /trusted/measurement \
  --geometry-repo /trusted/gte \
  --stability-repo /trusted/plsr
python examples/remaining-modules/run.py --url ws://127.0.0.1:8765
```

The identified stack contains eleven role-named checkouts at the pins in
the `calibrated-observable` descriptor pins and the `identified-design` descriptor pins.
The measurement stack contains `rci`, `fsrt` and `jspt` at the pins in
`measurement_chain.py`. Keep those stacks separate: the two FSRT interfaces
require different revisions. GTE and PLSR have one explicit checkout each.
Python 3.12 or newer is required by the identified stack.

The public synthetic example first retains a native calibrated-observable
bundle and identified-design bundle on the same session. It then executes and
replays each of these instruments, inspects their retained views, reads the
common catalog and saves the workspace:

| Source kind | Native computation | Retained interpretation |
| --- | --- | --- |
| `measurement-chain` | RCI calibration, FSRT snapshot reconstruction, JSPT covariance propagation | Exact raw records, calibration context, declared independence, native workspace and declared Jacobian |
| `geometric-circle` | GTE circle projection and full joint covariance propagation | Declared geometric frame, native candidate and held status; physical acceptance is not established |
| `identified-stability` | PLSR loads a sealed discrete model and evaluates its declared quadratic certificate | Exact selected SIDT model and GSIE prediction, native verdict and unknown parameter covariance |

Stability explicitly names the newly retained model and state occurrences.
The helper constructs a supplied model artifact for that selection; execution
loads the received sealed artifact. The conservation-model example may retain
a noncertifying verdict. A held or inconclusive outcome remains inspectable.
These instruments do not create a GSIE fusion context or admit a state.

Restoring a saved workspace makes the evidence, views and catalog available
without native repository access. Replay requires trusted host bindings again
and records fresh execution and result occurrences while preserving source
bytes and numerical results.

Run the installed integration gate with the same provider checkouts:

```bash
python scripts/check_remaining_modules.py \
  --identified-design-stack-root /trusted/identified \
  --measurement-chain-stack-root /trusted/measurement \
  --geometry-repo /trusted/gte \
  --stability-repo /trusted/plsr \
  --output-dir /tmp/ciw-remaining-module-fixtures
```

Without checkout arguments the gate clones each exact provider revision. It
builds and installs an isolated wheel, checks provider source identity before
and after execution, exercises the live WebSocket catalog and rejects skipped
native tests. The retained output includes actual source, original, replay,
held and refusal fixtures plus the common workspace. Refusal receipts record
rejected attempts and are not scientific result artifacts.

These are CIW native integration and numerical tests. The gate does not invoke
ICRH; independent `measurement-chain.v1`, `geometric-circle.v1` and
`identified-stability.v1` harness profiles remain pending. Matching replay
results and CIW test assertions do not establish that missing coverage.
