# Computational experimentalist queue

`ciw lab` runs a persistent queue of 168 computational experiments. Each task
follows the same loop and returns the same report, and every result carries
exactly one evidence label. The queue designs and executes experiments,
searches for counterexamples, and retains the evidence. It does not acquire
physical measurements, and no task claims one. Tasks that need hardware,
providers or tools that are unavailable are reported as blocked with their
planned protocol, not skipped silently.

```
hypothesis
→ mathematical prediction
→ synthetic or physical protocol
→ implementation
→ execution
→ independent comparison
→ uncertainty/provenance report
→ regression test or deferred research question
```

## Evidence labels

| Label | Meaning | Produced by |
| --- | --- | --- |
| `analytic` | Derived in closed form from declared assumptions | A cited derivation only |
| `synthetic` | Computed from declared generated inputs | A generator without a passing reference check |
| `numerically_verified` | A stated numerical condition passed | Passing analytic, high-precision, invariant, self-convergence, exact or refusal checks |
| `provider_backed` | Returned by a pinned external runtime | Executed provider with repository, revision and tree |
| `hardware_measured` | Acquired from an identified physical device | Acquisition record with raw digest, time and calibration reference |
| `independently_verified` | Agreement between implementations of different origin | e.g. `ciw` against `scipy`, `sympy`, `mpmath` or a pinned provider |
| `not_established` | Not supported by the basis | Any failed check, any physical claim without acquisition, every authority claim |

The label is computed by `ciw.lab.evidence.supported_label` from the finding's
declared basis; stating a different label is refused. Analysis never upgrades
evidence: a derived physical status is `hardware_measured` only when every
input is. `independently_verified` means independent *implementation*
agreement; independent verification by another party is outside what the
queue can establish. Machine safety, industrial readiness, customer demand,
actuator authority and production acceptance are always `not_established`.

A report's primary label is the weakest established computational label among
its findings (`synthetic` < `analytic` < `provider_backed` <
`numerically_verified` < `independently_verified`), and `not_established` when
any computational finding is refuted or none is established. A report with an
analytic derivation and two independent checks is therefore `analytic`: strong
findings never lift a weaker one. The per-label counts sit next to it.

| A computational experiment may establish | It cannot establish alone |
| --- | --- |
| Analytic agreement | Physical truth |
| Numerical convergence | Calibration validity |
| Synthetic sensor performance | Real sensor performance |
| Replay determinism | Independent verification by another party |
| Schema/provenance integrity | Machine safety |
| GPU/CPU agreement | Industrial readiness |
| A plausible use case | Actual customer demand |
| A simulated control response | Safe actuator authority |

The formal rules are in [lab/SPECIFICATIONS.md](lab/SPECIFICATIONS.md); the
authoring contract for new tasks is [lab/AUTHORING.md](lab/AUTHORING.md).

## Task report

Every task, including a blocked or deferred one, answers the same questions
(`ciw.lab-task-report.v1`): Task ID, Hypothesis, Mathematical model, Input
data, Observation model, Expected invariant, Experiment or test, Changed files,
Generated artifacts, Numerical result, Uncertainty, Evidence status,
Provider/runtime identity, Failure modes checked, Tests passed, Tests skipped,
Physical validation status, Unresolved assumptions and Recommended next task.
Evidence status and physical validation status are derived from the retained
findings. Each report has a content identity that detects accidental edits
(see [Retained evidence](#retained-evidence)).

States: `completed` (the planned computation ran and its checks passed),
`partial` (some planned parts could not run here, named in the report),
`blocked` (a hard requirement is unavailable) and `deferred` (not attempted).
A failed check or a failed regression test turns `completed` into `partial`.

## Commands

```sh
python -m pip install -e '.[dev,lab]'
ciw lab queue --retained lab                     # every task with its retained state and label
ciw lab run T003 T005 --output-dir results/lab   # selected tasks
ciw lab run --all --output-dir results/lab \
    --provider csg=/trusted/references/csg \
    --provider ftr=/trusted/references/ftr \
    --provider plsr-python=/path/to/python3.12   # interpreter with the pinned PLSR
ciw lab report T010 --retained lab               # the nineteen answers for one task
ciw lab report T010 --retained lab --schema      # also check task-report.schema.json
ciw lab next --retained lab                      # rank the next experiments; runs nothing
ciw lab dashboard --retained lab --output lab/index.html   # self-contained HTML view
ciw lab classify results/workspace.json          # label results in an existing CIW workspace
ciw lab queue --retained lab --section geodesic-jacobi --state partial
ciw lab verify --retained lab --fresh results/lab
```

`ciw lab next` is the persistent-queue view: it ranks implemented tasks that
were never reported, blocked tasks whose requirements have become available,
blocked tasks that declare no requirement as `retry` only when a re-run could
end differently (the block was an unexpected exception, or the task's sources
or runtime changed since its report; otherwise re-running reproduces the same
report, so the task stays blocked), in both cases with the report's
unresolved assumptions as the reason, partial tasks with their own next step,
and completed tasks' recommended follow-up research questions.
Hardware-blocked tasks stay listed as blocked until the hardware is bound.
`ciw lab run --budget-seconds N` lists tasks that exceed a time budget;
elapsed times go to `run-log.json`, never into reports, because timing is not
a reproducible finding.

`ciw lab dashboard` shows an SVG figure only when its file lies inside the
retained directory and hashes to the sha256 its report recorded; it embeds the
figure as an `<img>` data URI, so nothing inside an SVG can run in the page,
and otherwise prints a note saying why the figure is not shown.

`ciw lab classify WORKSPACE` applies the same labels to results retained in an
existing CIW workspace, as a derived projection that never edits sealed
records. It validates the workspace like `Session.from_workspace` (no provider
is bound; built-in offline analyses such as the energy-accuracy log analysis
may be recomputed to check retained data), then labels the run evidence, each
operation result, each workbench bundle and each replay receipt. Replay
receipts are same-runtime determinism checks, never independent verification.
Results are `synthetic` only under a structured declaration: a run provenance
`generator` that is a known CIW generator identity (today only
`ciw.instruments.make_demo_run`) or an exact value CIW's validators pin
(`origin: synthetic_fixture` and the variational free-energy policy values);
any other generator name (an acquisition script, a vendor API, an instrument
driver class) and wording such as "synthetic aperture radar" or "function
generator" do not count. An energy log that merely
declares `physical_measurement` stays `not_established`: `hardware_measured`
needs a declared `raw_sha256` that the raw acquisition bytes of another
workspace source hash to, device, clock and calibration fields, and no
synthetic or generated declaration (CIW's energy logs carry no raw digest, and
their `log_digest` hashes the JSON record). Workspace seals are unkeyed, so
they detect alteration but do not authenticate origin; the output says so, and
every label is as recorded.

## Assistant access over MCP

`ciw lab mcp` serves the queue to MCP clients over stdio (install the `mcp`
extra). It is the lab-scoped part of the planned MCP adapter:

```sh
python -m pip install -e '.[lab,mcp]'
ciw lab mcp --retained lab --workdir results/lab-mcp --provider csg=/trusted/references/csg
```

| Tool | Effect |
| --- | --- |
| `ciw_lab_list_tasks` | Paginated task list with state and primary label |
| `ciw_lab_get_report` | One revalidated nineteen-question report |
| `ciw_lab_plan_next` | The ranked next experiments over work and retained reports; runs nothing |
| `ciw_lab_run_tasks` | Runs up to 20 tasks into the server's work directory (destructive there) |
| `ciw_lab_verify_run` | Compares the tasks in the work directory with their retained reports |
| `ciw_lab_classify_workspace` | Labels results in a saved CIW workspace |
| `ciw_lab_explain_labels` | The label definitions and the boundary table |

[`lab/mcp_evaluation.xml`](lab/mcp_evaluation.xml) holds ten read-only
evaluation questions with stable answers; `tests/test_lab_mcp.py` checks each
answer through the tools themselves.

No tool accepts a label, finding, report or physical result: an assistant can
design, run and read experiments, but evidence status comes only from the
validator, and retained reports are never modified through the adapter: the
server refuses a work directory equal to, inside or containing the retained
directory, or sharing any file with it. It compares file identities as well
as paths, so a second mount point, symlinked `reports` or `artifacts`
directories and hard-linked copies are refused, and it repeats the file check
before every run. `ciw_lab_run_tasks` is annotated destructive because it
deletes and replaces the tasks' reports, artifacts and run log in the work
directory. Tools that read or write the work directory run one at a time,
across every server process using that directory (an OS lock on
`.ciw-lab-mcp.lock` in it). Task state and
the plan come from the work directory first, then the retained reports;
`ciw_lab_verify_run` compares only the tasks present in the work directory,
lists retained tasks not regenerated there without failing on them, and
refuses when either side has no reports. An unknown `state` or `section` in
`ciw_lab_list_tasks` is refused with the valid values.

`src/ciw/lab/task-report.schema.json` is the structural JSON Schema of
`ciw.lab-task-report.v1` for consumers in other languages. Passing it does not
make a report valid: labels, derived statuses and the report identity are
checked by `ciw.lab.report.validate_report`.

`--provider ROLE=PATH` binds a pinned provider checkout or interpreter; tasks
verify the checkout revision and tree against CIW's own pins before using it
and run provider code in a subprocess. `ciw lab verify` compares a fresh run
with retained reports: same states, labels, claims, units, domains and
wording, artifacts matching their recorded digests, and finding values within
each finding's declared regression tolerance. It exits 3 on any difference, on
a task present on only one side, and when the retained directory is missing or
holds no reports.

`scripts/check_lab.py` is the clean-room gate that CI runs and the way to
reproduce the retained run. It provisions CSG, FTR and SCR checkouts at CIW's
pins (cloned, or clean checkouts named `csg`, `ftr` and `scr` under
`--stack-root`) and runs `scripts/reproduce_lab.py` with them bound, plus the
clean-room interpreter, which has the pinned PLSR installed, as `plsr-python`
and `ftr-python`. Reproducing the retained run needs Python 3.12+ (PLSR and
FTR run there), Git, and those bound providers; on an older interpreter the
gate refuses to compare, and with `--no-compare` the PLSR-backed Lyapunov
tasks and the FTR comparison end partial.

```sh
python scripts/check_lab.py --output-dir results/lab-gate          # clones the pinned providers
python scripts/check_lab.py --stack-root /trusted/lab-providers --output-dir results/lab-gate
```

`scripts/reproduce_lab.py` is the step underneath. It builds a wheel from the
checkout (pip build isolation supplies the build backend), installs it with
the requested extras into a new virtual environment outside the checkout, runs
the lab tests with a JUnit record, runs the whole queue with the `--provider`
bindings it is given (`@venv` names the clean-room interpreter) and verifies
the fresh reports against `lab/`. Each binding also reaches the tests as the
variable its provider-gated tests read, so those tests run instead of
skipping. The bindings `check_lab.py` makes set `CIW_LAB_CSG_REPO`,
`CIW_LAB_FTR_REPO`, `CIW_LAB_SCR_REPO`, `CIW_LAB_FTR_PYTHON` and
`CIW_LAB_PLSR_PYTHON`; it binds no SET, PPDA or SCR exchange checkout, so
their tests skip in CI and T097 stays partial. Run without the bindings
`check_lab.py` makes, its comparison with `lab/` fails on the provider tasks.
Paths are made absolute without following symlinks, so a virtual
environment's interpreter stays bound as itself, and `gate.json` records the
bindings as passed and the clean-room Python version. It refuses to start
when the retained directory holds no reports unless `--no-compare` is given,
and it runs the packaged queue only: `CIW_LAB_EXTENSIONS`, `CIW_LAB_MODULES`,
the calling shell's provider-test variables and its operator hardware
captures (`CIW_LAB_RAPL_LOG`, `CIW_LAB_ENERGY_LOG`, `CIW_LAB_NVIDIA_SMI_CSV`,
`CIW_LAB_NVIDIA_SMI_UTC_OFFSET`) are removed. T164 records a run as a
clean-room reproduction only when the imported `ciw` package holds exactly the
named wheel's files, byte for byte, inside an isolated interpreter; a marker
naming any other file stays `not_established`. The wheel digest stays in
T164's runtime identity, out of the compared prose, because it changes with
every build.

## Retained evidence

Regenerate it only together with the change that alters it, and only from a
clean-room gate run under Python 3.12+ with every provider bound; never write
`ciw lab run` output into `lab/` (outside the clean room T164 is partial, the
provider tasks differ, and the run log would be retained):

```sh
python scripts/refresh_lab.py --stack-root /trusted/lab-providers   # runs check_lab.py --no-compare
python scripts/refresh_lab.py --from-run results/lab-gate           # or retains an existing gate run
git diff --stat lab
```

`refresh_lab.py` refuses a run whose gate record does not show Python 3.12+
and bindings for CSG, FTR, SCR and the PLSR/FTR interpreter, or whose reports
carry a CSG, FTR or PLSR refusal code (a bound provider that did not run), and
keeps elapsed times, the JUnit record and the gate record out of `lab/`.

`lab/` holds the retained run: `reports/T*.json`, `artifacts/T*/` (tables,
SVG figures, drafts and ledgers), `queue-state.json` and `REPORTS.md`, the
full report book. Aggregates generated by the research tasks include the
counterexample catalogue (`artifacts/T157/COUNTEREXAMPLES.md`), uncertainty
budgets (`T159`), the release report (`T165`), unresolved assumptions
(`T166`) and what remains unmeasured (`T167`).

Report content identities and artifact digests are unkeyed hashes. They
detect accidental edits and corruption: validation refuses a report whose
identity no longer matches, and `ciw lab verify` refuses artifacts whose bytes
differ from their recorded digest. A deliberate edit that recomputes the
identities and digests passes validation. Verifying the retained run against a
fresh one (`ciw lab verify`, the clean-room gate) catches it only in what
verify recomputes: states, finding claims, labels, units, domains, values and
counterexamples (within the regression tolerance the retained finding
records), and the report prose with numbers masked. Verify does not compare
artifact bytes with the fresh run's (timing records, drafts and aggregates
differ between identical runs), finding bases, tests passed or skipped,
changed files, or provider and runtime identities. An edit confined to those,
or one that widens a retained tolerance, passes both validation and verify;
changes to `lab/` are reviewed in version control (`git diff lab`).

## Sections

| Section | Tasks | Module | Guide |
| --- | --- | --- | --- |
| Geodesic/Jacobi | T001–T018 | `geodesic_jacobi.py`, `geodesic_jacobi_limits.py` | [GEODESIC_JACOBI](lab/GEODESIC_JACOBI.md), [limits](lab/GEODESIC_JACOBI_LIMITS.md) |
| Flat torus and topology | T019–T032 | `flat_torus_topology.py` | [FLAT_TORUS_TOPOLOGY](lab/FLAT_TORUS_TOPOLOGY.md) |
| Surfaces and discrete geometry | T033–T044 | `surfaces_discrete.py`, `surfaces_discrete_mesh.py` | [SURFACE_INTERFACE](lab/SURFACE_INTERFACE.md), [MESH_GEODESICS](lab/MESH_GEODESICS.md) |
| Observation | T045–T059 | `observation.py` | [OBSERVATION](lab/OBSERVATION.md) |
| Sensor fusion | T060–T076 | `sensor_fusion.py` | [SENSOR_FUSION](lab/SENSOR_FUSION.md) |
| Exchange and provenance | T077–T100 | `exchange_provenance.py`, `exchange_provenance_bundles.py` | [EXCHANGE_PROVENANCE](lab/EXCHANGE_PROVENANCE.md), [bundles](lab/EXCHANGE_BUNDLES.md) |
| Lyapunov runtime | T101–T114 | `lyapunov.py` | [LYAPUNOV](lab/LYAPUNOV.md) |
| Energy and GPU | T115–T125 | `energy_gpu.py` | [ENERGY_GPU](lab/ENERGY_GPU.md) |
| Manufacturing and robotics | T126–T141 | `manufacturing.py` | [MANUFACTURING](lab/MANUFACTURING.md) |
| Implementation targets | T142–T154 | `implementation_targets.py` | [IMPLEMENTATION_TARGETS](lab/IMPLEMENTATION_TARGETS.md) |
| Research and portfolio | T155–T168 | `research_portfolio.py` | [SPECIFICATIONS](lab/SPECIFICATIONS.md) |

The shared geometry core (`surfaces.py`, `integrators.py`, `jacobi.py`)
provides a chart-level metric/Christoffel/curvature interface with exact
embedding derivatives, Euler/midpoint/RK4 and Dormand–Prince integrators that
never renormalize, and joint geodesic/Jacobi transfer integration.
