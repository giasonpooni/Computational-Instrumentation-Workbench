# Exchange and provenance, part 2: bundles, providers and visibility (T091–T100)

Implementation: `src/ciw/lab/exchange_provenance_bundles.py` (tasks),
`src/ciw/lab/exchange_provenance_bundles_fixtures.py` (embedded inputs, protocol
client, execution guard, fabricated heat bundle, golden manifest, malformed
catalogue) and `src/ciw/lab/exchange_provenance_bundles_providers.py` (checkout
identities, locked build, SCR/SET/PPDA integrations). Tests:
`tests/test_lab_exchange_provenance_bundles.py`. Retained fixtures:
`tests/fixtures/lab/golden/` and `tests/fixtures/lab/malformed/`.

```
python -m ciw lab run T091 T092 T093 T094 T095 T096 T097 T098 T099 T100 --output-dir results/lab-bundles \
    --provider scr=<Scientific-Computation-Runtime @ a59aba2> [--provider scr-engine=<execution-cli>] \
    [--provider csg=... --provider ftr=... --provider sra=... --provider plsr=...] \
    [--provider set=<SET @ 542e672> --provider ppda=<PPDA @ a29845e> --provider scr-exchange=<SCR @ 5f04097>]
python -m ciw lab report T092 --retained results/lab-bundles
python -m pytest -q tests/test_lab_exchange_provenance_bundles.py
CIW_LAB_SCR_REPO=<scr> [CIW_LAB_SCR_ENGINE=<execution-cli>] [CIW_LAB_SET_REPO=... CIW_LAB_PPDA_REPO=... \
    CIW_LAB_SCR_EXCHANGE_REPO=...] python -m pytest -q tests/test_lab_exchange_provenance_bundles.py
```

These experiments drive the unmodified CIW session, workbench, energy-accuracy,
declared-workload, exchange and candidate-evidence code. T091–T096 and T100 need
nothing beyond NumPy; they carry the exact bytes of six small example sources
(checked against `examples/` by the tests) so that they also run from an
installed wheel. T097–T099 run only when pinned provider checkouts are bound.
Without providers the section takes about 6 s; with SCR, SET, PPDA and the
exchange SCR checkout bound, about 15–26 s (T097 dominates). The tests take
about 7 s offline and about 27 s with every provider variable set.

## Tasks

| Task | What is tested | Outcome here |
| --- | --- | --- |
| T091 | Save a session (oscillator results, energy-accuracy original + replay), reopen it under an **execution guard** that replaces 22 CIW entry points (workflow steps, sessions, adapters, subprocesses, recording operations, bindings) with refusing recorders. | 0 execution-path calls, no bindings, identical retained content. Counterexample: reopen recomputes the retained energy analysis 7 times (validation, not execution). |
| T092 | Replay/execution of provider kinds without a trusted binding, client-supplied `repositories`, a client `workflow.bind` request, unknown and unregistered operations. | 13/13 refused with the exact named error; no provider path reached. Counterexample: a fabricated, content-consistent numerical-heat bundle (values `[0, 1, 2, 3, 0]` where the reference is `[0, 16, 24, 16, 0]`) passes reopen validation. |
| T093 | Digests of the saved workspace, in-memory state (selection, results, executions, catalog, revision, byte and reservation counters, identities, bindings) and the session directory before and after 16 refused requests. | Unchanged; a refused reopen writes nothing. Counterexample: a refused *recording* operation is retained as a refused execution record, by design. |
| T094 | Golden workspaces saved by CIW itself, reopened with the current code under the guard. | 3 workspaces reopen; digests match `GOLDEN_MANIFEST`; the golden SCR heat values agree with an independent integer reference; unbound replay refused. |
| T095 | 14 committed malformed fixtures plus 5 generated ones, each against the validator it targets. | 19/19 refused with retained text. Counterexamples: `session.read_json` accepts `1e999` as infinity; it raises `RecursionError` (not `ValueError`) on deep nesting; `Session.from_workspace` accepts and silently drops unknown top-level fields. |
| T096 | `exchange._identity` over 24 seeded records (114 single-field mutations) and `candidate_evidence.validate_response` over 72 synthetic ESM responses. | All valid accepted, all mutations refused. Counterexamples: observation-batch identities are caller-declared; unknown ESM fields are accepted (the boundary checks bindings only). |
| T097 | SCR through CIW's numerical-heat workflow and SCR's own Python API; SET contracts validator; PPDA/SCR/SET producer roundtrip. | `provider_backed`; outputs equal the integer reference (`independently_verified`). Parts whose checkouts are not bound are recorded as `not_established` with the exact commands; state `partial`. |
| T098 | HEAD, tree, tracked-byte digest, Cargo.lock digests and engine digest of every bound checkout; independent Git tree recomputation; CIW pin comparison; CIW's own adapter refusing non-matching pins. | Every bound checkout here matched a CIW pin and reproduced its tree. CIW declares four different SET revisions; one checkout cannot satisfy all of them. |
| T099 | `cargo build --release --locked --offline -p execution-cli` into two fresh target directories. | Builds succeed, Cargo.lock and the checkout unchanged, bit-identical binaries, engine output `[0, 219, 313, 219, 0]` for `[0, 0, 1000, 0, 0]` after 3 steps. SP1 build recorded, never attempted: state `partial`. |
| T100 | Label/domain audit and rendered-Markdown audit of every retained report below T100; relabelling of CIW energy and free-energy records. | Relabels refused where the record can detect them. Counterexamples: a resealed relabel under a fresh occurrence is classified `physical_domain_measurement`; a `|` in a claim moves the label out of its Markdown column. |

## Golden fixtures

`tests/fixtures/lab/golden/` holds three workspaces written by
`Session.save_workspace` (exact bytes; `tests/fixtures/lab/.gitattributes`
disables end-of-line conversion):

- `energy-accuracy-workspace.json`: the synthetic baseline energy log, its
  original analysis and a replay (workspace version 3).
- `oscillator-workspace.json`: a statistics result, a spectrum result after a
  selection update, and a `statistics.v1` operation with its execution (version 2).
- `numerical-heat-workspace.json`: an SCR original and replay produced with the
  real provider (SCR `a59aba283b0304faeeb3e5d305087e7709e171ca`, engine
  `sha256:b9f40b3094ecf4b793676c766ffd559dc69abc4fada564c1bde94e79c6f0b201`).
  Its runtime identity records the host paths of the machine that produced it.
  They are identity metadata and are never used as bindings.

Their SHA-256 digests are recorded in `GOLDEN_MANIFEST` and in the T094 findings.
To replace a fixture on purpose, run
`write_golden_fixtures("tests/fixtures/lab", scr=..., engine=...)` and put the
new digests in `GOLDEN_MANIFEST` in the same change. T094 finds the fixtures
through `ciw.lab.runner.repository_path`, or through an explicit
`--provider ciw-fixtures=<dir>`. When neither is available it reports
`blocked`.

`tests/fixtures/lab/malformed/` is written by `write_malformed_fixtures`, and a
test checks that the committed files equal the generator's output. Oversized
inputs and inputs that need a full recording are generated at run time.

## Provider roles

| Role | Checkout / file | CIW pin(s) |
| --- | --- | --- |
| `scr` | Scientific-Computation-Runtime | `a59aba2…` (`declared_workload.PINS`, `proved_heat.PIN`, tree `4068a71…`) |
| `scr-engine` | prebuilt `execution-cli` (optional; otherwise built with `cargo --locked --offline`) | digest recorded, operator-asserted |
| `scr-exchange` | Scientific-Computation-Runtime for the exchange producer | `5f04097…` (`.github/workflows/exchange.yml`) |
| `set` | State-Estimation-Evaluation-Testbed | `542e672…` (`exchange-runtime.json`); telemetry and calibrated pins differ |
| `ppda` | Provenance-Preserving-Data-Acquisition | `a29845e…` (exchange workflow); telemetry pin `209985a…` |
| `csg`, `ftr` | geodesic reference providers | `geodesic_reference.PINS` with trees |
| `sra`, `plsr` | schematic assessment, PLSR | `declared_workload.PINS`, `plsr-runtime.json` |

Pins that exist only in `.github/workflows/exchange.yml` are mirrored in
`EXCHANGE_WORKFLOW_PINS`, and a test keeps the two in step. The SP1 proved-heat
build (`cargo +1.94.0 build --release --locked --manifest-path <scr>/zk/Cargo.toml
-p sp1-adapter --bin sp1-host`) needs:

- network access to crates.io,
- the SP1 checkout `b38b612…` (tree `7deca3a…`) next to SCR,
- `protoc`, clang and libssl,
- at least 7 GiB of RAM and 20 GiB of disk.

T099 records these requirements and never attempts the build.

## What this does not establish

- **Content consistency is not correctness.** Reopen checks commitments and
  identities. It does not recompute a provider's numbers (T092). Only a
  replay with a trusted binding re-executes.
- **The CIW `origin` field is a declaration.** A synthetic energy log that
  is resealed as `physical_measurement` under a fresh occurrence cannot be
  told apart from a real one (T100). Real GPU energy and real sensor
  performance are recorded as `not_established`.
- **A matching checkout digest is not authentication.** It does not
  authenticate the upstream repository, the Rust toolchain or an engine
  beyond the inputs that were actually executed. Binary digests depend on
  the toolchain. They are kept as provenance, not as regression values.
- **`provider_backed` is not independent verification by another party.**
  Agreement with the integer reference is independent-implementation
  agreement only.
- **Floating-point portability of the energy golden is untested.** Reopening
  recomputes the retained energy analysis in floating point. The golden
  energy workspace reopens under Python 3.11 and 3.12 with NumPy 2.4.3 on
  this machine. Other BLAS builds were not tested.

## Requested core changes

1. `ciw.lab.report.render_markdown` should escape finding claims and units
   (for example with `_inline`). Several retained claims use `|x|`
   absolute-value notation, which makes GitHub-flavoured Markdown drop the
   label cell. T100 reports these as rendering violations.
2. In CIW itself, not the lab core:
   - `ciw.session.read_json` could refuse overflowing numbers and convert
     `RecursionError` into `ValueError`.
   - `Session.from_workspace` could refuse unknown top-level fields.
   - Workbench `source.add` reports malformed source JSON with the runtime
     text "The bound runtime did not return finite, unambiguous JSON". That
     text names a runtime, but the bytes came from a source.
