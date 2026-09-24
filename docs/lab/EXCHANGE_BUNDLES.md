# Exchange and provenance, part 2: bundles, providers and visibility (T091–T100)

Implementation: `src/ciw/lab/exchange_provenance_bundles.py` (tasks),
`src/ciw/lab/exchange_provenance_bundles_fixtures.py` (embedded inputs, protocol
client, execution guard, fabricated heat bundle, lab-written canonical JSON
encoder, golden manifest and platform fingerprint, malformed catalogue) and
`src/ciw/lab/exchange_provenance_bundles_providers.py` (checkout, lockfile and
interpreter identities, locked build, SCR/SET/PPDA integrations). Tests:
`tests/test_lab_exchange_provenance_bundles.py`. Retained fixtures:
`tests/fixtures/lab/golden/` and `tests/fixtures/lab/malformed/`.

```
python -m ciw lab run T091 T092 T093 T094 T095 T096 T097 T098 T099 T100 --output-dir results/lab-bundles \
    --provider scr=<Scientific-Computation-Runtime @ a59aba2> [--provider scr-engine=<execution-cli>] \
    [--provider csg=... --provider ftr=... --provider sra=... --provider plsr=...] \
    [--provider plsr-python=<Python 3.12 with the pinned PLSR> --provider ftr-python=<Python 3.12>] \
    [--provider set=<SET @ 542e672> --provider ppda=<PPDA @ a29845e> --provider scr-exchange=<SCR @ 5f04097>]
python -m ciw lab report T092 --retained results/lab-bundles
python -m pytest -q tests/test_lab_exchange_provenance_bundles.py
CIW_LAB_SCR_REPO=<scr> [CIW_LAB_SCR_ENGINE=<execution-cli>] [CIW_LAB_CSG_REPO=... CIW_LAB_PLSR_PYTHON=...] \
    [CIW_LAB_SET_REPO=... CIW_LAB_PPDA_REPO=... CIW_LAB_SCR_EXCHANGE_REPO=...] \
    python -m pytest -q tests/test_lab_exchange_provenance_bundles.py
```

These experiments drive the unmodified CIW session, workbench, energy-accuracy,
declared-workload, exchange and candidate-evidence code. T091–T096 and T100 need
nothing beyond NumPy. They carry the exact bytes of six small example sources
(checked against `examples/` by the tests), so they also run from an installed
wheel. T092 reads further `examples/` sources through
`runner.repository_path` when the repository is reachable. T097–T099 run only
when pinned provider checkouts are bound.

Timing on this machine: without providers the section takes about 5 s. With every
provider bound it takes about 16 s (the SCR locked build dominates, and it is
shared by T097–T099). The tests take about 11 s offline and about 26 s with every
provider variable set.

`scripts/check_lab.py` provisions SET, PPDA and a second SCR checkout at the
exchange workflow's pins (roles `set`, `ppda` and `scr-exchange`, from
`EXCHANGE_WORKFLOW_PINS`) beside CSG, FTR, SCR and the PLSR/FTR interpreter,
so the SET and roundtrip parts of T097 and their provider-gated tests run in
the clean-room gate and the retained run. Offline, `test_t097_keeps_set_results_when_the_engine_is_missing`
and `test_t097_roundtrip_finding_is_refuted_by_a_contrary_producer_outcome`
drive both parts with mocked provider outcomes, as expected and contrary, and
show that a contrary outcome refutes the finding. Bound to SET `542e672`, PPDA
`a29845e` and SCR `5f04097`, T097 completes here with both parts
`numerically_verified`.

Every finding records an exact per-finding uncertainty (`roundoff`, value 0):
counts, digests, strings and int64 fields are compared exactly. Report prose
never names a checkout path, so a gate run that clones providers into a fresh
temporary directory still verifies against retained reports. Paths are kept
only in artifacts.

## Tasks

| Task | What is tested | Outcome here |
| --- | --- | --- |
| T091 | Save a session that holds a trusted numerical-heat binding (synthetic, naming no checkout), oscillator results, an energy-accuracy original + replay and a provider-kind numerical-heat bundle (fabricated, content-consistent). Then reopen it under an **execution guard**. The guard replaces 73 CIW entry points with refusing recorders: session analyses, recording operations, the workbench, provider adapters and subprocesses, plus the session, step and adapter entry points of every workbench workflow kind, discovered from `ciw.workbench._workflow`. | 0 execution-path calls. The binding is gone: none after reopen, no binding path in the saved bytes, and replaying the provider-kind bundle is refused unbound. Retained content is identical. Counterexample: reopening recomputes the retained energy analysis 7 times (validation, not execution). Only that analysis is counted as recomputation. |
| T092 | Execute every unavailable workflow kind that has a reachable example source: 15 unbound, plus 2 that consume an upstream bundle. Replay the one retained provider-kind bundle (numerical-heat). Also send client-supplied `repositories`, a client `workflow.bind`, unknown and unregistered operations, and an ESM request on a non-telemetry bundle. | 25/25 refused with the exact named error; no provider process, adapter or workflow entry point reached. Replay refusal is observed for numerical-heat only; for the other kinds it is inferred from `Workbench.replay` calling `Workbench._reserve`, not observed. Four kinds without an example (`acquired-calibrated-window`, `bim-quantity`, `identified-stability`, `residual-monitor`) are not exercised. Counterexample: a fabricated, content-consistent numerical-heat bundle (values `[0, 1, 2, 3, 0]` where the reference is `[0, 16, 24, 16, 0]`) passes reopen validation. The reopen outcome is observed and checked; if reopen refused the bundle, the counterexample would be refuted and the other cases would still run. |
| T093 | 16 refused non-recording request classes, each compared with its expected `code: message`. Digests of in-memory state (selection, results, executions, catalog, revision, byte and reservation counters, identities, bindings) and of the session directory are taken before and after, plus a re-save comparison. | 16/16 refused with the expected text; state unchanged; a refused reopen writes nothing. The unchanged-state claim is scoped to those classes. Counterexample: a refused *recording* operation is retained as a refused execution record, by design. |
| T094 | Golden workspaces saved by CIW itself, reopened with the current code under the guard. The retained heat bundle's runtime identity is compared with CIW's pins. | 3 workspaces reopen and their digests match `GOLDEN_MANIFEST`. The golden heat values equal the integer reference, and the bundle names SCR `a59aba2`, tree `4068a71` and the recorded engine digest (`numerically_verified`). Reopen cannot show which engine produced those values, so that claim is recorded as `not_established`. Unbound replay is refused. |
| T095 | 14 committed malformed fixtures and 7 generated ones, each sent to the validator it targets. Wrong-type cases are single-field mutations of a valid input. Only a `ValueError` counts as a refusal; any other exception is retained as a crash. The committed fixtures are compared with their generator. | 19/21 inputs refused with their declared, exact CIW text (the 19 that have one). The other two are counterexamples, and a check requires every input without a declared text to be covered by one. Counterexamples: `session.read_json` accepts `1e999` as `inf` and raises `RecursionError` (not `ValueError`) on deep nesting. `Session.from_workspace` raises `AttributeError` on `{"workspace_version": 3}`, and accepts then silently drops unknown top-level fields. `source.add` words source-parse errors as a runtime response. |
| T096 | `exchange._identity` over 24 seeded records sealed by a lab-written canonical encoder (written from the producer specification without calling `json.dumps`, and checked to agree byte for byte with the `json.dumps` call `_identity` uses), with 114 single-field mutations. `candidate_evidence.validate_response` over 72 synthetic ESM responses. | All valid records accepted in any member order; all mutations refused. Acceptance follows from the encoder agreement (same-origin code), not from an independent check. Counterexamples: observation-batch identities are caller-declared, and unknown ESM fields are accepted (the boundary checks bindings only). |
| T097 | SCR through CIW's numerical-heat workflow and through SCR's own Python API; the SET contracts validator; the PPDA/SCR/SET producer roundtrip. | SCR output is `provider_backed` and equals the integer reference (`independently_verified`). SET and the roundtrip are *checked*: status, effective rank 2, the exact refusal text `covariance.matrix is not positive-semidefinite`, two matched links, preserved failed verification, `result_id` refusal, and `may_authorize` false. A contrary provider outcome refutes the finding (both parts have mocked as-expected and contrary tests). Parts that cannot run are recorded as `not_established` with the exact reason and commands, and any part that ran keeps the task `partial` rather than `blocked`. |
| T098 | HEAD, tree, tracked-byte digest and the digest of every recognised lockfile (`Cargo.lock`, `uv.lock`, `poetry.lock`, `Pipfile.lock`, `package-lock.json` and the other names in `LOCKFILE_NAMES`, plus fully pinned `requirements*.txt`) of every bound checkout, and the engine digest. Bound interpreters (`plsr-python`, `ftr-python`) are probed for version and executable digest, and for the installed PLSR runtime's version and source digests. Also: an independent Git tree recomputation, the pin comparison, CIW's own adapter against every module pin plus a control revision, and pins grouped by repository. | Every bound checkout here is clean and at a CIW pin; CSG's `uv.lock` is recorded. The tracked and lockfile digests are read a second time from Git's HEAD objects (`git cat-file`, an `independent_check` of origin `git`). The adapter accepts the pins at HEAD and refuses a control revision taken from another repository's pin, so its refusal side is exercised even when every declared pin is at HEAD. The installed PLSR matches `ciw/plsr-runtime.json`. Interpreter and engine digests are provenance (artifact and `provider_runtime_identity`), not regression values. A refused checkout gets its own finding naming the reasons, each corroborated by a second reader, and the task becomes `partial`. CIW declares two SCR revisions (`a59aba2` for declared-workload and proved-heat, `5f04097` for the exchange workflow) and four SET revisions. |
| T099 | `cargo build --release --locked --offline -p execution-cli` into two fresh target directories. | Exit codes 0 are checked, Cargo.lock and the checkout are unchanged, the two binaries are bit-identical, and the engine outputs `[0, 219, 313, 219, 0]` for `[0, 0, 1000, 0, 0]` after 3 steps. A failed build is recorded as a refuted claim, not hidden. The SP1 build is recorded with its full requirements and never attempted, so the state is `partial`. |
| T100 | Label/domain and rendered-Markdown audit of every earlier report present in the output directory. Relabelling of CIW energy and free-energy records, reading the retained bundle's own classification. A fabricated numerical-heat bundle read through `bundle.get` and `ciw lab classify` (`classify_workspace`). | Relabels are refused where the record can detect them. Counterexamples: a resealed relabel under a fresh occurrence is accepted and its bundle is classified `physical_domain_measurement`; the fabricated heat bundle reopens and is labelled `provider_backed`, like a provider result. Its only reader-visible trace is a source tree that differs from CIW's pin, which neither reopen nor classify compares. The pipe-in-claim probe records whichever renderer behaviour it observes. |

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
  `sha256:b9f40b3094ecf4b793676c766ffd559dc69abc4fada564c1bde94e79c6f0b201`,
  recorded as `GOLDEN_SCR_ENGINE_SHA256`). Its runtime identity records the host
  paths of the machine that produced it. Those paths are identity metadata and
  are never used as bindings.

Their SHA-256 digests are recorded in `GOLDEN_MANIFEST` and in the T094 findings.
To replace a fixture on purpose, run
`write_golden_fixtures("tests/fixtures/lab", scr=..., engine=...)` and put the
new digests in `GOLDEN_MANIFEST` in the same change. T094 finds the fixtures
through `ciw.lab.runner.repository_path`, or through an explicit
`--provider ciw-fixtures=<dir>`. When neither is available it reports
`blocked`.

**Platform dependence of the energy golden.** Reopening recomputes the retained
energy analysis (LAPACK `solve`/`cholesky` on 2×2 matrices) and compares it bit
for bit. The goldens were written on Linux x86_64 with NumPy 2.4.3 and
scipy-openblas 0.3.31.dev, with AVX-512 kernels available (`GOLDEN_PLATFORM`).
T094 records bit-exact recomputation as its own finding, and `golden.json`
keeps the fingerprints of the current platform and of the goldens' origin.

The regression test marks itself `xfail` only when two things are both true:
the recomputation is refused with exactly `Retained energy analysis binding
differs`, and the platform fingerprint differs from `GOLDEN_PLATFORM`. On a
matching platform that refusal is a real failure.

Only Linux under Python 3.11 and 3.12 has been run, and both are bit-exact.
Windows has not been run.

`tests/fixtures/lab/malformed/` is written by `write_malformed_fixtures`, and a
test checks that the committed files equal the generator's output. Oversized
inputs, inputs that need a full recording, and single-field mutations of a full
saved workspace are generated at run time.

## Provider roles

| Role | Checkout / file | CIW pin(s) |
| --- | --- | --- |
| `scr` | Scientific-Computation-Runtime | `a59aba2…` (`declared_workload.PINS`, `proved_heat.PIN`, tree `4068a71…`) |
| `scr-engine` | prebuilt `execution-cli` (optional; otherwise built with `cargo --locked --offline`) | operator-asserted; compared with the run's locked build when cargo is available |
| `scr-exchange` | Scientific-Computation-Runtime for the exchange producer | `5f04097…` (`.github/workflows/exchange.yml`) |
| `set` | State-Estimation-Evaluation-Testbed | `542e672…` (`exchange-runtime.json`); telemetry and calibrated pins differ |
| `ppda` | Provenance-Preserving-Data-Acquisition | `a29845e…` (exchange workflow); telemetry pin `209985a…` |
| `csg`, `ftr` | geodesic reference providers | `geodesic_reference.PINS` with trees |
| `sra`, `plsr` | schematic assessment, PLSR | `declared_workload.PINS`, `plsr-runtime.json` |

Pins that exist only in `.github/workflows/exchange.yml` are mirrored in
`EXCHANGE_WORKFLOW_PINS`, and a test keeps the two in step. The SP1 proved-heat
build (`cargo +1.94.0 build --release --locked --manifest-path <scr>/zk/Cargo.toml
-p sp1-adapter --bin sp1-host`) has these requirements, taken from
`.github/workflows/proved-heat.yml` and mirrored in `SP1_REQUIREMENTS` (a test
keeps them in step):

- **Network:** crates.io and git dependencies, GitHub clones of SCR and SP1,
  and a GitHub release download of the Succinct compiler archive
  (sha256 `12c94435…`), linked with `rustup toolchain link succinct`.
- **Guest verification:** the guest recipe `zk/recipes/sp1-heat.recipe`
  (identity `6e5d1687…`), rebuilt and checked by `verify_build` against guest
  sha256 `a14e3750…`.
- **SP1 checkout:** `b38b612…` (tree `7deca3a…`), next to SCR.
- **System tools:** `protoc`, clang and libssl.
- **Machine:** at least 7 GiB of available RAM and 20 GiB of free disk.

T099 records these requirements and never attempts the build.

## What this does not establish

- **Content consistency is not correctness.** Reopen checks commitments and
  identities. It does not recompute a provider's numbers (T092), and a retained
  runtime identity does not show which engine produced the values (T094). Only a
  replay with a trusted binding re-executes (T097).
- **The CIW `origin` field is a declaration.** A synthetic energy log that
  is resealed as `physical_measurement` under a fresh occurrence cannot be
  told apart from a real one (T100). Real GPU energy and real sensor
  performance are recorded as `not_established`.
- **A retained runtime identity is a declaration too.** `ciw lab classify`
  labels a fabricated, content-consistent bundle `provider_backed` (T100).
- **A matching checkout digest is not authentication.** It does not
  authenticate the upstream repository, the Rust toolchain or a built engine;
  T098 records this as a `not_established` provenance finding. Binary digests
  depend on the toolchain. They are kept as provenance (basis notes and
  artifacts), not as regression values.
- **`provider_backed` is not independent verification by another party.**
  Agreement with the integer reference is independent-implementation
  agreement only.
- **T096 acceptance is encoder agreement.** The lab-written canonical encoder
  agrees byte for byte with the `json.dumps` call `exchange._identity` uses;
  both are CIW-side code. Producer-sealed artifacts are exercised only by
  T097's roundtrip.
- **The execution guard covers the listed entry points.** A workflow reached
  through a name imported into another module, or through code outside those
  entry points, is not intercepted, and only the energy analysis is counted
  as validation recomputation (T091).
- **T100 audits what is present.** A section-only run audits its own reports;
  the value records how many of T001–T099 were present.

## Requested core changes

1. In CIW itself, not the lab core:
   - `ciw.session.read_json` could refuse overflowing numbers and convert
     `RecursionError` into `ValueError`.
   - `Session.from_workspace` could refuse unknown top-level fields, and should
     refuse `{"workspace_version": 3}` with a `ValueError` instead of raising
     `AttributeError`.
   - Workbench `source.add` reports malformed source JSON with the runtime
     text "The bound runtime did not return finite, unambiguous JSON". That
     text names a runtime, but the bytes came from a source.
