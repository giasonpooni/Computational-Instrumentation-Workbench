# Computational experimentalist queue

`ciw lab` runs a persistent queue of 168 computational experiments. Each task
follows the same loop and returns the same report, and every result carries
exactly one evidence label. The queue designs and executes experiments,
searches for counterexamples, and retains the evidence. It does not acquire
physical measurements in the retained run, and no task there claims one. Tasks
that need hardware, providers or tools that are unavailable are reported as
blocked or partial with their planned protocol, not skipped silently; runs made
on a hardware host are retained separately under `lab/hardware/` (see
Hardware evidence), and runs of the SP1 proved-heat gate under
`lab/proved-heat/` (see Proved-heat gate records).

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
| `provider_backed` | Returned by a pinned external runtime | Executed provider with repository, revision and tree; in `ciw lab classify`, a retained runtime identity that is a pin CIW declares for the workflow kind |
| `hardware_measured` | Acquired from an identified physical device | Acquisition record with raw digest, time and calibration reference |
| `independently_verified` | Agreement between implementations of different origin, not verification by another party | e.g. `ciw` against `scipy`, `sympy`, `mpmath` or a pinned provider |
| `not_established` | Not supported by the basis | Any failed check, any physical claim without acquisition, every claim filed in an authority domain |

The label is computed by `ciw.lab.evidence.supported_label` from the finding's
declared basis; stating a different label is refused. Analysis never upgrades
evidence: a derived physical status is `hardware_measured` only when every
input is. `independently_verified` means independent implementation
agreement; independent verification by another party is outside what the
queue can establish.

A passing check outranks provenance in these rules, so the label alone does
not say where a result came from. Every finding therefore also carries its
*basis components*, derived from the same basis: the parts of the basis it
declares (`derivation`, `synthetic_inputs`, `provider`, `acquisition`,
`reference_checks`, `independent_check`; stored under the finding key
`origin`). They are not the implementation origin that
`independently_verified` compares, and they never change a label. Reports,
`REPORTS.md` and the dashboard show them beside each label as the finding's
*basis*, with the identity each component declares: the generator and its
seed, the provider as repository@revision, the acquisition device. So
`numerically_verified` from reference checks on synthetic inputs
(`ciw.lab.sensor_fusion_bench`, seed 602026) is visibly distinct from
`numerically_verified` from reference checks on a pinned provider's output.
An acquisition record reads as a hardware acquisition only on a
physical-domain finding it establishes, which the runner's physical gate
inspects; anywhere else it reads as a declared acquisition record that was
not accepted, and a computational claim cannot cite one at all. The
identities are as declared, not authenticated.

Claims filed in the `machine_safety`, `industrial_readiness`,
`customer_demand`, `actuator_authority` or `production_acceptance` domains are
always `not_established`. The domain is chosen by the finding's author and
reviewed; it is not inferred from the claim. T141 showed that an authority
statement filed under a computational domain was labelled by its checks (and
filed under a physical domain with an acquisition record it would be
`hardware_measured`), so `ciw.lab.evidence` refuses a computational- or
physical-domain claim whose wording asserts production acceptance,
certification for use, machine safety, actuator authorization, industrial
readiness or customer demand. The only exemption is a clause that says,
before the outcome, that the software does not make, mark or record it ("the
lab API cannot mark a lot accepted for production", "records production
acceptance as not performed"); a negation elsewhere in the claim ("accepted
for production; it does not need rework") exempts nothing, and making the
software the subject exempts nothing ("the workbench is ready for industrial
deployment" is refused). The screen
matches phrases, not every paraphrase; domain assignment remains a review
question (rule 10 in [lab/SPECIFICATIONS.md](lab/SPECIFICATIONS.md)).

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
ciw lab next --retained results/lab --retained lab   # merge a work directory over the retained run
ciw lab dashboard --retained lab --output lab/index.html   # self-contained HTML view
ciw lab classify results/workspace.json          # label results in an existing CIW workspace
ciw lab queue --retained lab --section geodesic-jacobi --state partial
ciw lab verify --retained lab --fresh results/lab   # also checks lab/hardware/ and lab/proved-heat/ for integrity
ciw lab run T138 --capture cmm=/captures/cmm-export.csv --output-dir results/cmm   # operator capture
ciw lab hardware retain results/rtx2080 --retained lab --run-id rtx2080-2026-10-01 --host "RTX 2080 workstation"
ciw lab hardware verify --retained lab           # integrity of every retained hardware run
ciw lab proved-heat retain results/proved-heat --retained lab --run-id local-2026-09-24 --host "Linux x86-64, 16 GiB"
ciw lab proved-heat verify --retained lab        # integrity of every retained proved-heat gate record
ciw lab unmeasured --retained lab                # open physical claims beside each hardware run's counts
```

`ciw lab next` is the persistent-queue view (`ciw.lab-next.v2`): it ranks
implemented tasks that were never reported (`ready`), blocked tasks whose
requirements have become available (`unblocked`), blocked tasks that declare
no requirement as `retry` only when a re-run could end differently (the block
was an unexpected exception, or the task's sources or runtime changed since
its report; otherwise re-running reproduces the same report, so the task stays
blocked), in both cases with the report's unresolved assumptions as the
reason, partial tasks with their own next step (`refinement`, every one listed
whatever `--limit` is), a completed task's next step that points at a queue
task which has not completed (`follow_up`, naming it in `points_to`), and open
research questions (`research`): the part of a completed task's next step that
names no queue task, and every deferred research question recorded in any
report's unresolved assumptions. A pointer to a task that already completed is
never proposed; it is listed under `stale_pointers`, and any question it
carries beyond the pointer (a later sentence, a separate item or a variant of
the completed experiment) stays a research row. A completed pointer whose step
waits on a physical execution ("before executing MFG-SCAN-01", "once a real
study exists") is a research row that `revisits` the pointed task. Identical
questions are listed once, with the other tasks under `also_from`;
`follow_ups` and `research` list every such row whatever the limit, and
reasons are cut at word boundaries.
`--retained` may be repeated; the first directory holding a task wins.
Hardware-blocked tasks stay listed as blocked until the hardware is bound or a
retained hardware run holds them (see [Hardware evidence](#hardware-evidence)).
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
their `log_digest` hashes the JSON record). A result is `provider_backed` only
when every runtime identity it retains with a revision and source tree is a
pin CIW itself declares for that workflow kind: the revision of a pin for the
role it is recorded under (an identity nested inside a role's runtime, such
as PPDA's vendor runtime, or recorded in a step may match any pin of the
kind), the pin's module and source root, and the source tree CIW records for
that revision in any of its pin tables (`ciw.lab.bridge.declared_pins`).
Where no CIW table records a tree for the revision, any tree is accepted and
the item's `runtime_pins` row shows `tree_pinned: false`; today that is every
pin of `calibrated-window`, `acquired-calibrated-window`, `telemetry`,
`residual-monitor`, `schematic-assessment`, `schematic-companions` and
`bim-quantity`, every `calibrated-observable` and `identified-design` pin
except `gsie`, and the historical `fsrt` and `rci` adapter pins
(`ciw.lab.bridge.pins_without_tree`). For those, a record with the pinned
revision, module and source root and an invented tree classifies
`provider_backed`. Any other identity leaves the result `not_established`,
with the reason in the finding and in the item's `runtime_pins` rows; a
fabricated, content-consistent numerical-heat bundle whose tree is not the
pin (T100) is therefore not `provider_backed`. The pins are public constants
and workspace seals are unkeyed, so a fabricated record that copies the
pinned revision and tree still classifies `provider_backed`: the seals detect
alteration but do not authenticate who produced a record; the output says so,
and every label is as recorded.

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
holds no reports. It also checks every retained hardware run under
`<retained>/hardware/` and every proved-heat gate record under
`<retained>/proved-heat/` for integrity (listed under `hardware_runs` and
`proved_heat_records` in its output) and exits 3 on any integrity problem
there.

`scripts/check_lab.py` is the clean-room gate that CI runs and the way to
reproduce the retained run. It provisions CSG, FTR and SCR checkouts at CIW's
pins and SET, PPDA and a second SCR checkout at the exchange workflow's pins
for the T097 roundtrip (cloned, or clean checkouts named `csg`, `ftr`, `scr`,
`set`, `ppda` and `scr-exchange` under `--stack-root`) and runs `scripts/reproduce_lab.py` with them bound, plus the
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
`CIW_LAB_FTR_REPO`, `CIW_LAB_SCR_REPO`, `CIW_LAB_SET_REPO`, `CIW_LAB_PPDA_REPO`,
`CIW_LAB_SCR_EXCHANGE_REPO`, `CIW_LAB_FTR_PYTHON` and `CIW_LAB_PLSR_PYTHON`;
the SET, PPDA and exchange SCR checkouts let T097 run the SET contracts
validator and the PPDA/SCR/SET producer roundtrip. `check_lab.py` also binds
the latest retained proved-heat gate record from the checkout's
`lab/proved-heat/` as `proved-heat-record` (`CIW_LAB_PROVED_HEAT_RECORD`),
since the clean room has no copy of `lab/`, so T099 reads the same record there
(see [Proved-heat gate records](#proved-heat-gate-records)). Run without the bindings
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
and bindings for CSG, FTR, SCR, SET, PPDA, the exchange SCR and the PLSR/FTR
interpreter, or whose reports
carry a CSG, FTR or PLSR refusal code (a bound provider that did not run), and
keeps elapsed times, the JUnit record and the gate record out of `lab/`. It
never touches `lab/hardware/` or `lab/proved-heat/`: operator hardware runs
and proved-heat gate runs enter `lab/` only through `ciw lab hardware retain`
and `ciw lab proved-heat retain` (see [Hardware evidence](#hardware-evidence)
and [Proved-heat gate records](#proved-heat-gate-records)). When
`lab/proved-heat/` holds a record, it also refuses a run that bound none as
`proved-heat-record`, or whose T099 did not find the CI-pinned rustup
toolchain (its `tool:cargo+1.94.0` probe), which CI's lab gate installs.

`lab/` holds the retained run: `reports/T*.json`, `artifacts/T*/` (tables,
SVG figures, drafts and ledgers), `queue-state.json` and `REPORTS.md`, the
full report book. Aggregates generated by the research tasks include the
counterexample catalogue (`artifacts/T157/COUNTEREXAMPLES.md`), uncertainty
budgets (`T159`), the release report (`T165`, covering T001–T164, the tasks
that run before it), unresolved assumptions (`T166`) and what remains
unmeasured in the run (`T167`; retained hardware runs are counted outside the
queue by `ciw lab unmeasured`).

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

Figures are compared by re-execution instead. A task whose figure plots
wall-clock timings declares it when writing it,
`ctx.artifact_text(name, svg, wall_clock_timing=True)`, which records
`wall_clock_timing: true` on the figure's entry in the report's generated
artifacts (today T142's `kernel-timings.svg`); `validate_report` and the
schema accept the declaration only as `true` on an SVG figure. T158
re-executes the figure tasks that fit the research section's time budget and
compares every retained figure byte for byte, except a declared one, which is
compared for presence and structure (series and points) only: an undeclared
figure that differs is a mismatch, and a declared figure that reproduced byte
for byte is reported, not hidden. `scripts/check_figures.py` re-executes every
figure task of a retained run with the installed `ciw`, without a time budget,
into a new directory, compares the figures the same way and writes
`figure-check.json` (`ciw.lab-figure-check.v1`: each figure's outcome and the
platform, Python, NumPy and BLAS build) and `figure-check.md`. A task whose
retained report used a provider that is not bound (`--provider ROLE=PATH`; a
provider probe that succeeded, or a checkout its runtime identity records as
ready or at a revision, as T097 records set, ppda and scr-exchange) or
recorded source digests that differ from the installed package's is listed as
not re-executed, and one that ends in another state or with other
requirement-probe outcomes as not comparable; neither counts as a match. It
exits 3 on a mismatch or when no figure was compared. Run on Windows against
the same `lab/`, it is the second-platform comparison T158 names as its next
step:

```sh
python scripts/check_figures.py --retained lab --output-dir results/figures              # every figure task
python scripts/check_figures.py --retained lab --output-dir results/figures T013 T020   # selected tasks
python scripts/check_figures.py --retained lab --output-dir results/figures-windows \
    --provider csg=/trusted/references/csg --provider scr=/trusted/references/scr \
    --provider set=/trusted/references/set --provider ppda=/trusted/references/ppda \
    --provider scr-exchange=/trusted/references/scr-exchange \
    --provider plsr-python=/path/to/python3.12   # the providers the retained figure tasks used
```

## Hardware evidence

The division of work is fixed: Claude designs and audits experiments, CIW
executes and records them, independent references check them, hardware
produces physical evidence, CIW retains the raw data, calibration,
uncertainty and provenance, and Claude analyzes without upgrading evidence
status. The clean-room run in `lab/` has no hardware, so its hardware tasks
stay blocked and its physical claims `not_established`. Physical evidence
enters `lab/` only as a retained hardware run:

1. On the capture host (for example the RTX 2080 host, where
   `ciw energy probe` succeeds), follow the task's protocol and run the
   hardware tasks, and only those, into a new directory, for example
   `CIW_LAB_ENERGY_LOG=$R/capture/log.json ciw lab run T116 T118 --output-dir results/rtx2080`.
   `--capture energy-log=PATH`, `--capture nvidia-smi-csv=PATH` and
   `--capture rapl-log=PATH` set the same operator variables for the run.
   `run-record.json` beside the reports records the run's CIW version and
   package digest and its provider and capture bindings as role names with
   digests, never host paths.
2. Retain it: `ciw lab hardware retain results/rtx2080 --retained lab
   --run-id rtx2080-2026-10-01 --host "RTX 2080 workstation"`. The command
   refuses a run that is not the output of one `ciw lab run`, in which no
   hardware probe succeeded and no operator capture was retained, whose
   reports do not validate or quote a host path anywhere (an error message
   naming an operator's file, but also an illustrative absolute path such as
   T092's or T153's, which is why the run should hold the hardware tasks
   only), whose artifacts differ from their digests, or whose
   hardware-measured findings fail the physical gate again. It copies the
   reports, artifacts and run log to `lab/hardware/<run-id>/` and writes
   `capture.json` (`ciw.lab-hardware-run.v1`: run identity, the
   operator-declared host, the run date, CIW version and package digest,
   tasks and report identities, provider and capture bindings by role and
   digest, the hardware-measured count). The operator's declared host is not
   verified. Artifacts hold raw bytes and are not screened for paths.
3. Review `git diff lab/hardware` and commit it with the change that motivated it.

A hardware-measured finding needs a hardware probe that succeeded in its own
task (`hardware:nvidia-gpu` or `hardware:rapl`; a probe replayed from another
task's memo does not count), and the task binds its raw bytes to the probed
device's identity (the energy tasks compare the log's NVML or RAPL identity
with the analysing host's). A run whose physical findings rest on such a
probe cannot be recomputed in CI or on any host other than the capture host.
`ciw lab verify`, `ciw lab hardware verify` and `scripts/check_lab.py` (in
either mode, inside the clean room) therefore check each retained hardware
run for integrity only: its reports validate and hold no host path, its
artifacts match their digests and no unrecorded file is present, every
hardware-measured finding's `raw_sha256` is an artifact of its own task and
its report records a hardware probe that succeeded in that task itself
(`hardware_probed_in_task` in its runtime identity; for bytes of an operator
capture, a probe of that capture's instrument), and `capture.json` agrees with
the run. `scripts/refresh_lab.py` never touches `lab/hardware/`.

`ciw lab queue --retained lab` and the planner show, beside a task's main
state and label, the latest valid hardware run holding it, marked with its run
identity; the main run's state and labels are never replaced, and a hardware
run's labels are its own. `ciw lab next` stops proposing a task blocked in the
main run as blocked on hardware when its latest hardware run completed or
partly completed it, and ranks it from that run's report instead
(`hardware_run` on the row). A run with any integrity problem is ignored by
these views and listed under `hardware_run_problems`.
`ciw lab unmeasured --retained lab` (`ciw.lab-unmeasured.v1`) is the
hardware-aware view of what remains unmeasured: for each task with an open
physical or authority claim in the main run, or held by a valid hardware run,
it lists the main run's state, hardware-measured count and open claims beside
each hardware run's own state, labels and measured claims. The main run's
count and T167's ledger count are reported as they are, and hardware-run
counts are never added to them. It runs nothing, outside the queue, so the
clean-room reproduction never reads `lab/hardware/`.
`ciw.lab.runner.hardware_runs(retained)` returns the valid runs with their
reports and hardware-measured counts.

### Operator captures

Raw bytes from an instrument (a CMM export, a photogrammetry manifest) reach a
task as operator captures: `ciw lab run --capture ROLE=PATH` binds a file to a
capture role, `ctx.available("capture:ROLE")` succeeds when it is bound and
readable, and `ctx.capture(ROLE)` returns its bytes and retains them as the
task's artifact `capture-<role><suffix>`; the report records the role and
digest under `operator_captures` in its runtime identity. Operator captures
are retained and not authenticated: nothing binds them to a device, so a
capture by itself never supports an established physical label. A task may
compute non-physical findings from the bytes; its physical claims stay
`not_established`, and a physical finding citing the capture blocks the task.
The one route from a capture to `hardware_measured` is a probe of the
capture role's instrument on the analysing host that succeeded in the same
task (`energy-log` and `nvidia-smi-csv` need `hardware:nvidia-gpu`, `rapl-log`
needs `hardware:rapl`), with the task binding the capture to the probed
device's identity. Roles without an instrument probe (`cmm`, any role other
than those three) keep their physical claims `not_established` until an
instrument probe or a signed-capture trust anchor exists. A run of captures
alone involves no host device: it can be retained with
`ciw lab hardware retain` and re-analysed on any host from its retained
capture, for example
`ciw lab run T138 --capture cmm=lab/hardware/<run-id>/artifacts/T138/capture-cmm.csv --output-dir results/re`
followed by `ciw lab verify --retained lab/hardware/<run-id> --fresh results/re`.

## Proved-heat gate records

The SP1 proved-heat gate (`.github/workflows/proved-heat.yml`; see
[PROVED_HEAT.md](PROVED_HEAT.md)) rebuilds the registered guest ELF from SCR's
committed recipe, builds the native engine and SP1 prover, and has
`scripts/check_proved_heat.py` produce, verify, replay and re-verify a real
SP1 proof from an installed CIW wheel and reject a tampered one. It needs
network access, the SP1 checkout, the Succinct compiler archive and at least
7 GiB of available memory (the retained run's largest process peaked at about
10 GB), so the lab queue never runs it; its outcome enters `lab/` only as a
retained record:

1. On a provisioned Linux host, run the workflow's own gate steps with
   `python scripts/run_proved_heat_locally.py --scr <clean SCR checkout>
   --sp1 <fresh SP1 clone> --compiler-archive <Succinct archive> --python
   <Python 3.12>`. It reads the workflow file, refuses when a provisioning
   step's result is missing (tools, toolchains, the archive's pinned SHA-256,
   clean checkouts at the pinned revisions and trees), while a variable that
   would change the built bytes (`RUSTFLAGS`, `CARGO_PROFILE_*` and the like)
   is set, or when the workflow has a step or step setting it does not
   replay, runs the gate steps unchanged and in order, and
   writes `results/proved-heat/` as the workflow does, plus `local-run.json`
   (`ciw.proved-heat-local-run.v1`: which steps ran and for how long, how the
   others were satisfied, host facts, toolchain versions and source
   identities; never a host path).
2. Retain it: `ciw lab proved-heat retain results/proved-heat --retained lab
   --run-id <run-id> --host "<host description>"`. The command refuses a gate
   that did not pass and an incomplete output; it copies the gate outputs
   needed to re-check the claims (the build and source-check records, the gate
   record, test record, source, re-verification report, and the original and
   replayed bundles with their proofs, compressed), leaves out the session
   workspace once its two bundles are checked to be the retained ones, and
   writes `run.json` (`ciw.lab-proved-heat-run.v1`) and `manifest.json`
   (`ciw.lab-proved-heat-manifest.v1`, the SHA-256 of every file and of the
   gate's bytes inside each compressed one). The copy must pass verification.
3. Review `git diff lab/proved-heat` and commit it.

Verification (`ciw lab verify`, `ciw lab proved-heat verify` and
`scripts/check_lab.py`) checks each record for integrity only: every file
matches the manifest and no unrecorded file is present, the gate passed with
the native tests `check_proved_heat.py` requires, its SCR and SP1 revisions and
trees, guest recipe, guest ELF and compiler archive are the pins CIW declares
(`ciw.proved_heat.PIN`, `SP1_REQUIREMENTS`), and the retained bundles, replay
receipt and re-verification report pass CIW's offline proved-heat validation,
carry runtime identities that are proved-heat pins
(`ciw.lab.bridge.declared_pins`) and name the engine, prover and guest the
gate record names. No proof is
re-verified; `ciw proof verify` can re-verify a retained proof, once
decompressed (`gunzip -k gate/original.json.gz`), on a host with the pinned
binaries. `lab/proved-heat/local-2026-09-24/` holds the first
record, a local replay of the workflow's gate steps; CI has not run the gate
for it.

T099 reads the record bound as `proved-heat-record`. A record that fails
verification is refused by name: T099's record finding is refuted with the
failing categories and the gate claims stay `not_established` with the
reason. A valid record's claims (guest ELF rebuilt from the committed recipe
and equal to the registered guest; real SP1 proof, verification, fresh
replay, re-verification and tamper rejection) are `provider_backed`, the label
of a retained provider record at a pin CIW declares: the gate's outcome as
recorded, not a verification by T099. T099's own checks of the record
(digests, pins, bundle consistency) are `numerically_verified`, and so is its
rebuild of `execution-cli` with the CI-pinned rustup toolchain, compared byte
for byte with the engine the gate proved against, when that toolchain is
installed (`tool:cargo+1.94.0`; `lab.yml` installs it). The gate bound its
engine and prover as `operator_asserted_not_attested`, which T099 records as a
`not_established` provenance claim; its times, memory and proof sizes are
measurements of its recording host, reported in T099's result and artifacts
and never compared; and the digests are unkeyed, so a fabricated record that
copies CIW's public pins and recomputes every digest passes.

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
