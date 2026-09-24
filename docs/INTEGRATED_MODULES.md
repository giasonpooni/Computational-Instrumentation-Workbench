# Native modules in one operating session

The workbench hosts nine executable workflow kinds and a source-only geographic
context. Sources, native results, execution history and explicit replay share the
existing catalog. Provider ownership remains explicit.

| Connection | Operation / view | Retained scope |
| --- | --- | --- |
| SRA assessment → SRA/JSPT/PLSR | `ciw.schematic-companions.v1` | Selected scalar quadratic-drag state derivative, Jacobian, local structure, optional first-order input covariance and continuous linear Lyapunov sample |
| IFC and scalar observation → CSE | `ciw.bim-quantity.v1` | Native quantity conditioning, full prior/posterior covariance, invariants, execution ledger and ledger replay |
| Incremental snapshots → PPDA/SCOUT | `ciw.acquired-dataset.v1` | Source/adapter registry, native acquisition plan, evidence, durable-pool restoration and checkpoints |
| Geographic context → GSV | `experiment.inspect` `{view: spatial}` | Exact source bytes, CRS84 frame authority, declared constant entity states and native GSV WorldStore |

## Host bindings

Use clean checkouts at these exact revisions. No request or workspace may select
executable paths. Companion directories must be named `sra`, `jspt` and `plsr`.

| Role | Revision |
| --- | --- |
| SRA | `a6e79585950bb6860e5edce5ebd2cce39ea481f2` |
| JSPT | `7399ab03087b27683620b4c57f97b2ac14546c7f` |
| PLSR | `9d0e7b4a1162e038150a71c63d986945d78135d4` |
| CSE | `4b74abda40bba3277de69bf61e9e09283ae2d5b3` |
| PPDA | `477d6cb454423a27543b16961d3b169709c40c31` |
| PPDA's SCOUT gitlink | `5e146d5924675cd7b6e1d1ed44fb39f5da012610` |

```sh
python -m ciw serve \
  --schematic-companions-root /trusted/companions \
  --construction-repo /trusted/cse \
  --acquisition-repo /trusted/ppda \
  --spatial-view-origin http://127.0.0.1:5173 \
  --output-dir results/shared-workbench
```

These flags compose with existing process, calibrated-window, telemetry, design
and numerical-runtime bindings. PPDA needs its pinned submodule initialized.
Python 3.12 and CIW's NumPy dependency support the new numerical lanes.

Retain exact source bytes with `source.add`, then invoke `operation.execute`
with `parameters.source_id`. Companions additionally require
`parameters.upstream_bundle_id` selecting a retained `schematic-assessment`;
the source graph must exactly match its output graph. `bundle.replay` creates a
fresh occurrence. The Godot Workbench tab exposes native graph, context, quantity
covariance, acquisition counts and evidence through `experiment.inspect`.

Examples live in `examples/declared-workloads/schematic-companions.json`,
`examples/bim-quantity`, `examples/acquired-dataset` and
`examples/workbench/geographic-context.json`. Generate complete native fixtures:

```sh
python scripts/check_integrated_modules.py \
  --stack-root /trusted/providers --output-dir results/integrated-modules
```

The gate installs a wheel in an isolated environment, requires every provider
pin and fails on skipped tests. Output includes sources, original/replay pairs,
selected upstream artifacts, BIM held/refused cases and a geographic view packet.
Each numerical lane retains fresh same-runtime reproduction with
`independent: false`. ICRH separately inspects bindings and declared contracts;
no SET verdict is fabricated for these schemas.

## Geographic client

Retain the geographic example with `kind: geographic-context`. It has no
execution operation or bundle. GSV's workbench provider verifies the exact
source bytes and descriptor before using its native provider and WorldStore.
Start the GSV browser with:

```text
http://127.0.0.1:5173/?ciw=ws://127.0.0.1:8765/spatial&source=SOURCE_ID
```

`--spatial-view-origin` permits that exact browser origin only on `/spatial`.
The endpoint sends `spatial.ready`, then accepts only `experiment.inspect` with
`{"view": "spatial"}` (optionally with `source_id`); `spatial.list` and
`spatial.inspect` were folded into that view in `ciw.kernel.v2`. It receives
catalog invalidations, never the full session or
oscillator selection. It is read-only even without an Origin header. Unlisted
origins and browser requests to the native endpoint are refused.

Sources explicitly declare CRS84 longitude/latitude degrees, authority evidence,
validity, a native time range and every required entity-state value. This lane
supports nodes with declared constant state over that range. Laboratory-plane
and IFC coordinates are not converted into geography. Disconnects leave a
visibly retained/offline view.

## Scientific boundaries

Companion results concern a declared continuous linear surrogate. They do not
establish equilibrium, nonlinear stability, observability or a physical plant.
Missing covariance and ineligible/stale/ambiguous declarations remain explicit.

CSE accepts a matched metre-valued raw IFC quantity and independent scalar
observation. Unknown dependence or missing/mismatched frame, unit or IFC identity
holds the original native world. An invariant violation records a rejected
ledger event and unchanged state. Quantity conditioning grants no geometry
clearance or surveyed-frame authority.

PPDA executes bounded offline acquisition. Exact bytes and native sequence/cursor
semantics survive; sequence order is not event-time order. Missing clock or
calibration maps and unknown covariance remain incomplete measurement declarations.
See [the acquisition contract](ACQUIRED_DATASET.md).

These objects create no implicit GSIE fusion context or ESM candidate. Remaining
work includes acquisition-to-calibrated-stream composition, physical frame-bound
BIM inspection, retained-state/model Lyapunov analysis, and declared algebraic
and topological workload contracts.
