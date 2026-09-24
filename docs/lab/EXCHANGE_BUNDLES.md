# Exchange and provenance, part 2: bundles, providers and visibility (T091–T100)

Implementation: `src/ciw/lab/exchange_provenance_bundles.py` (tasks),
`src/ciw/lab/exchange_provenance_bundles_fixtures.py` (embedded inputs, protocol
client, execution guard, fabricated heat bundle, lab-written canonical JSON
encoder, golden manifest and platform fingerprint, malformed catalogue) and
`src/ciw/lab/exchange_provenance_bundles_providers.py` (checkout, lockfile and
interpreter identities, locked build, SCR/SET/PPDA integrations) and
`src/ciw/lab/proved_heat_records.py` (retained SP1 proved-heat gate records,
read by T099). Tests: `tests/test_lab_exchange_provenance_bundles.py` and
`tests/test_lab_proved_heat_records.py`. Retained fixtures:
`tests/fixtures/lab/golden/` and `tests/fixtures/lab/malformed/`; retained gate
record: `lab/proved-heat/local-2026-09-24/`.

```
python -m ciw lab run T091 T092 T093 T094 T095 T096 T097 T098 T099 T100 --output-dir results/lab-bundles \
    --provider scr=<Scientific-Computation-Runtime @ a59aba2> [--provider scr-engine=<execution-cli>] \
    [--provider csg=... --provider ftr=... --provider sra=... --provider plsr=...] \
    [--provider plsr-python=<Python 3.12 with the pinned PLSR> --provider ftr-python=<Python 3.12>] \
    [--provider set=<SET @ 542e672> --provider ppda=<PPDA @ a29845e> --provider scr-exchange=<SCR @ 5f04097>] \
    [--provider proved-heat-record=lab/proved-heat/local-2026-09-24]
python -m ciw lab report T092 --retained results/lab-bundles
python -m pytest -q tests/test_lab_exchange_provenance_bundles.py
CIW_LAB_SCR_REPO=<scr> [CIW_LAB_SCR_ENGINE=<execution-cli>] [CIW_LAB_CSG_REPO=... CIW_LAB_PLSR_PYTHON=...] \
    [CIW_LAB_SET_REPO=... CIW_LAB_PPDA_REPO=... CIW_LAB_SCR_EXCHANGE_REPO=...] \
    [CIW_LAB_PROVED_HEAT_RECORD=lab/proved-heat/local-2026-09-24] \
    python -m pytest -q tests/test_lab_exchange_provenance_bundles.py tests/test_lab_proved_heat_records.py
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
shared by T097–T099). The tests take about 13 s offline and about 26 s with every
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
| T096 | `exchange._identity` over 24 seeded records sealed by a lab-written canonical encoder (written from the producer specification without calling `json.dumps`, and checked to agree byte for byte with the `json.dumps` call `_identity` uses), with 114 single-field mutations. `candidate_evidence.validate_response` over 72 synthetic ESM responses. | All valid records accepted in any member order; all mutations refused. Acceptance follows from the encoder agreement (same-origin code), not from an independent check. Counterexamples: observation-batch identities are caller-declared, and unknown ESM fields are accepted (the boundary checks bindings only). Every finding declares the generator of its synthetic inputs (the seeded records, or the deterministic `_candidate_cases` for the ESM responses), so its Basis column names it; the checks still decide the label. |
| T097 | SCR through CIW's numerical-heat workflow and through SCR's own Python API; the SET contracts validator; the PPDA/SCR/SET producer roundtrip. Each exchange checkout's head and tree go into `provider_runtime_identity` as T098 records them, so a runtime inventory (T165) lists each checkout once. | SCR output is `provider_backed` and equals the integer reference (`independently_verified`). SET and the roundtrip are *checked*: status, effective rank 2, the exact refusal text `covariance.matrix is not positive-semidefinite`, two matched links, preserved failed verification, `result_id` refusal, and `may_authorize` false. A contrary provider outcome refutes the finding (both parts have mocked as-expected and contrary tests). Every finding that rests on a provider's execution declares it as its `provider`, so the Basis column names it beside the label, which the checks still decide: SCR for the heat, reference and replay findings, and SET for the validator finding and for the roundtrip, whose one provider slot names the checker its checks read (the PPDA and SCR producers are in its basis notes). The reopen finding executes nothing and declares no provider. Parts that cannot run are recorded as `not_established` with the exact reason and commands, and any part that ran keeps the task `partial` rather than `blocked`. |
| T098 | HEAD, tree, tracked-byte digest and the digest of every recognised lockfile (`Cargo.lock`, `uv.lock`, `poetry.lock`, `Pipfile.lock`, `package-lock.json` and the other names in `LOCKFILE_NAMES`, plus fully pinned `requirements*.txt`) of every bound checkout, and the engine digest. Bound interpreters (`plsr-python`, `ftr-python`) are probed for version and executable digest, and for the installed PLSR runtime's version and source digests. Also: an independent Git tree recomputation, the pin comparison, CIW's own adapter against every module pin plus a control revision, and pins grouped by repository. | Every bound checkout here is clean and at a CIW pin; CSG's `uv.lock` is recorded. The tracked and lockfile digests are read a second time from Git's HEAD objects (`git cat-file`, an `independent_check` of origin `git`). The adapter accepts the pins at HEAD and refuses a control revision taken from another repository's pin, so its refusal side is exercised even when every declared pin is at HEAD. The installed PLSR matches `ciw/plsr-runtime.json`. Interpreter and engine digests are provenance (artifact and `provider_runtime_identity`), not regression values. A refused checkout gets its own finding naming the reasons, each corroborated by a second reader, and the task becomes `partial`. CIW declares two SCR revisions (`a59aba2` for declared-workload and proved-heat, `5f04097` for the exchange workflow) and four SET revisions. |
| T099 | `cargo build --release --locked --offline -p execution-cli` into two fresh target directories. The retained SP1 proved-heat gate record bound as `proved-heat-record` (`lab/proved-heat/<run-id>/`, see [Proved-heat gate record](#proved-heat-gate-record)): its manifest digests, gate outcome, native test record, pins and bundles. With rustup toolchain 1.94.0 installed, one more build with `cargo +1.94.0` (CI's pin for the gate). | Exit codes 0 are checked, Cargo.lock and the checkout are unchanged, the two binaries are bit-identical, and the engine outputs `[0, 219, 313, 219, 0]` for `[0, 0, 1000, 0, 0]` after 3 steps. A failed build is recorded as a refuted claim, not hidden. The build, reproducibility and engine-output findings declare the bound SCR checkout as their provider. With `local-2026-09-24` bound, the record's checks pass (`numerically_verified`): the gate rebuilt the registered guest ELF from the committed recipe, and proved, verified, replayed and re-verified the heat result `[0, 65, 92, 65, 0]` and rejected a tampered proof, each `provider_backed` (the gate's outcome as recorded at CIW's pins); the gate's engine and prover being attested builds is `not_established` (`operator_asserted_not_attested`); and `cargo +1.94.0` rebuilds the engine the gate proved against byte for byte (`numerically_verified`), while this host's default rustc 1.94.1 builds another digest (reported in `pinned-toolchain-build.json`, not compared). The state is `completed`. Without a record, or without the pinned toolchain, those claims are `not_established` with the reason and the state is `partial`; a record that fails verification is refused by name (its finding is refuted) and T099 never runs the SP1 build itself. |
| T100 | Label/domain and rendered-Markdown audit of every earlier report present in the output directory. Each rendered row must equal `report.finding_row` (claim, value, label, declared basis), keep the label in the third of the header's four cells, and show in its Basis cell exactly the cell the finding's declared basis prescribes under [AUTHORING.md](AUTHORING.md). The audit builds that cell itself, without the renderer: every declared component in basis order, the generator with its name and seed, the executed provider as repository@revision, and an acquisition record as `hardware acquisition (<device>)` only on a physical finding it establishes (`hardware_measured` or `independently_verified`), `declared acquisition record (not accepted)` on every other. The whole cell is compared, so a component word inside a declared identity cannot stand in for the component, and a renderer that shows an unaccepted acquisition as hardware acquisition is a basis violation even though `report.finding_row` agrees with it. The label × basis-component counts (`evidence.origin_counts`) are retained in `label-by-basis.json`. Relabelling of CIW energy and free-energy records, reading the retained bundle's own classification. A fabricated numerical-heat bundle read through `bundle.get` and `ciw lab classify` (`classify_workspace`), sealed once with an invented source tree and once with the tree CIW pins for SCR `a59aba2` (`proved_heat.PIN`). | Over the 99 earlier reports of a full run: no label, rendering or basis violation, and every declared generator or executed provider is named beside its label. When the audited reports declare none, the visibility claim is recorded as untested (`not_established`, expected) rather than the counted absence. Relabels are refused where the record can detect them. The classifier labels the invented-tree bundle `not_established` (its tree is not the one CIW records for the pinned revision). Counterexamples: a resealed energy relabel under a fresh occurrence is accepted and its bundle is classified `physical_domain_measurement`; the bundle sealed with the pinned tree reopens and is labelled `provider_backed`, like a provider result, because the pins are public constants and the seals are unkeyed. The pipe-in-claim probe records whichever renderer behaviour it observes, and records a shifted label as a counterexample only when it observes the extra cell a raw pipe creates; a row that lost its label column otherwise refutes the label-column claim and carries no counterexample. |

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
| `proved-heat-record` | a retained SP1 proved-heat gate record, `lab/proved-heat/<run-id>/` (T099; `scripts/check_lab.py` binds the latest) | its SCR and SP1 revisions and trees, recipe, guest and compiler archive must be `proved_heat.PIN` and `SP1_REQUIREMENTS` |

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

T099 records these requirements and never attempts the build. The gate's
outcome reaches T099 only as a retained record.

## Proved-heat gate record

`lab/proved-heat/local-2026-09-24/` is one passing run of the gate's steps on
a provisioned Linux host (4 CPUs, 16 GiB of memory), made by replaying the
workflow's own gate steps with `scripts/run_proved_heat_locally.py`'s earlier
form and retained with `ciw lab proved-heat retain` (docs/LAB.md, Proved-heat
gate records). Its `run.json` records how each workflow step ran or was
satisfied (rebuild of the guest 83 s, native and prover builds 970 s, source
check 2 s, installed-wheel gate 1331 s), the toolchains (rustc 1.94.0, the
linked Succinct 1.94.0 compiler whose archive matched its pinned SHA-256,
protoc 3.21.12, clang 18.1.3, Python 3.12.3), an existing clean SCR checkout at
`a59aba2` and a fresh clone of upstream SP1 at `b38b612`. It was written after
the run from the step logs, the outputs and the host, because the driver then
in use wrote no run description. The gate rebuilt the guest ELF from recipe
`6e5d1687…` to the registered `a14e3750…`, and 111 tests passed, including the
three native tests: a real SP1 proof of `[0, 100, 200, 100, 0]` after 4 steps
(`[0, 65, 92, 65, 0]`), its verification, a fresh replay with its own proof,
re-verification of the retained proof without re-execution, and rejection of
a tampered proof. On the recording host the proof was 2778024 bytes, proving
and verifying took 354 s, re-verification 92 s, and the largest waited child's
peak RSS was 10.07 GB; these are measurements of that host and T099 never
compares them. The retained bundles hold both proofs, so once decompressed
`ciw proof verify` can re-verify them on a host with the pinned binaries.

Its engine digest `636b2447…` is what CI's pinned rustc 1.94.0 builds from SCR
`a59aba2`: the operator's observation in `run.json` built `execution-cli` with
1.94.0 from the checkout, from the gate's linked stack path and again (all
`636b2447…`) and with 1.94.1 from the checkout and from a second clone at
another path (both `b9f40b30…`, the digest of the earlier retained T099 and of
`GOLDEN_SCR_ENGINE_SHA256`). The digest depends on the toolchain, not on the
checkout path, on that host; T099's pinned-toolchain build tests the same
equality wherever it runs, and `lab.yml` installs 1.94.0 so that the CI lab
gate does.

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
  compares it with the pins CIW declares, so a fabricated, content-consistent
  bundle whose source tree is not the pin is `not_established` for
  numerical-heat, whose pinned revision has a tree CIW records
  (`proved_heat.PIN`). For the kinds in `ciw.lab.bridge.pins_without_tree`
  (telemetry, calibrated-observable and others) any tree is accepted
  (`tree_pinned: false`), so an invented tree at a pinned revision still
  classifies `provider_backed` there. Reopen does not compare the identity,
  and a fabricated bundle sealed with the public pinned revision and tree is
  labelled `provider_backed` (T100): the pins are constants and the seals are
  unkeyed.
- **The Basis column shows what a finding declares.** Generator names and
  seeds and provider repositories are text in the basis. T100 checks that
  every rendered row shows them beside the label, not that they are true. It
  sees only declared components: a finding that rests on a generator or a
  provider without declaring it is not detected, and one provider slot names
  one provider of a result several providers produced (T097's roundtrip names
  the SET checker; its producers are in the basis notes).
- **A matching checkout digest is not authentication.** It does not
  authenticate the upstream repository, the Rust toolchain or a built engine;
  T098 records this as a `not_established` provenance finding. Binary digests
  depend on the toolchain. They are kept as provenance (basis notes and
  artifacts), not as regression values; T099's pinned-toolchain rebuild
  compares its digest with the gate record's engine exactly, and retains the
  outcome, not the digest, as its value.
- **A retained gate record is the gate's outcome as recorded.** T099 checks
  the record's digests, pins and bundle consistency and re-verifies no proof,
  so its gate claims are `provider_backed`, not `numerically_verified`. The
  manifest digests are unkeyed: a fabricated record that copies CIW's public
  pins and recomputes every digest passes, as T100's copied-pin bundle does.
  The gate bound its engine and prover as operator-asserted executables; a
  reproducible engine build is not an attestation, and nothing rebuilds the
  prover. The record comes from one local run, not from CI.
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

## Open questions

Each task's recommended next step (`NEXT_STEPS` in the module) names its own
open question rather than the next queue task, which has already run by the
time anyone reads it; `ciw lab next` lists them as research rows, and
`test_next_steps_name_forward_work` checks with `ciw.lab.planner` that none is
a task pointer.

- T091: intercept execution below the Python entry points during reopen (audit
  hooks in a child process), and count validation recomputation for every kind.
- T092: observe replay refusal for kinds other than numerical-heat, from real
  bundles of bound providers (the telemetry stack pinned in
  `telemetry-runtimes.json` for telemetry bundles, and the calibrated-observable
  stack pinned in `calibrated-observable-runtimes.json` for calibrated-observable
  bundles), and add example sources for the four kinds not exercised.
- T093: generate refused requests from `Session._dispatch`'s request types, and
  decide whether a refused recording operation belongs in the saved workspace.
- T094: reopen the goldens on a second platform, and add goldens for other
  provider kinds.
- T095: extend the malformed matrix to the other kinds' source parsers, and
  re-run it once CIW makes the changes requested below.
- T096: combined mutations that restore consistency; content-bound batch
  identities and refusal of unknown ESM fields (CIW changes).
- T097: provision the telemetry provider stack and drive CIW's telemetry
  workflow end to end; attest the engine a bound replay uses.
- T098: authenticate checkouts and toolchains (signed commits or tags), and
  settle CIW's several SCR and SET pins.
- T099: have CI run the proved-heat gate itself and retain its output as a
  second record; attest the engine and prover the gate binds, so that its host
  binding is no longer operator-asserted; and rebuild `sp1-host` with the
  pinned toolchain on a second host to learn whether the prover digest, like
  the engine's, depends only on the toolchain.
- T100: keyed seals or a provider-signed runtime identity, so a copied pin no
  longer classifies `provider_backed`; a pin comparison on reopen; source trees
  recorded in CIW's pin tables for the revisions `pins_without_tree` lists, so
  the invented-tree check reaches those kinds; energy-log origin authenticated
  at acquisition.

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
