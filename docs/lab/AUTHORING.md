# Authoring a lab task

This guide is the contract for implementing a task in the computational
experimentalist queue (`src/ciw/lab/queue.json`). Every task follows one loop:

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

Every numerical or logical outcome is a *finding* built with
`ciw.lab.evidence.finding(claim, domain, value, basis, ...)`. The label is
computed from the basis; authors never state it.

| Label | Basis that produces it |
| --- | --- |
| `analytic` | `{"derivation": "<where the closed form is derived>"}` only |
| `synthetic` | `{"generator": {"name": ..., "seed": ..., ...}}` without checks |
| `numerically_verified` | one or more `checks`, all passing (see below) |
| `provider_backed` | `{"provider": {"repository", "revision", "source_tree" or "runtime_digest", "executed": true}}` without checks |
| `independently_verified` | `independent_check` whose checker has a different implementation origin (for example `scipy`, `sympy`, `mpmath`, a pinned provider) and passes |
| `hardware_measured` | physical-domain claim with `acquisition` (device, raw digest, time, calibration) |
| `not_established` | anything else, any failed check, and every authority-domain claim |

Check objects: `{"reference_kind": "analytic" | "high_precision" | "invariant" |
"self_convergence" | "exact_arithmetic" | "refusal" | "cross_implementation",
"reference": "<what was
compared>", "observed": <float>, "tolerance": <float>, "comparison": "abs_le" |
"le" | "ge" | "signed_le" | "signed_ge", "passed": <bool>}`. The validator
recomputes `passed` from `observed` and `tolerance` with
`ciw.lab.evidence.holds`, which section helpers should call too; a mismatch is
refused. Choose the comparison by what `observed` is:

| Observed quantity | Comparison |
| --- | --- |
| An error or residual whose sign is irrelevant | `abs_le` (default) |
| A nonnegative magnitude: error norm, count, ratio, bound | `le` or `ge`; a negative observed value with `le` is refused |
| A signed quantity with a one-sided bound: a difference such as "refined minus coarse error", "KL increase", "value minus bound" | `signed_le` or `signed_ge` |

`le` refuses a negative observed value because it would pass any upper bound
without testing anything. Tolerances must be finite with magnitude at most
`1e100`, and nonnegative for `abs_le`/`le`, so a threshold cannot make a
check vacuous. Refusal checks use `expected_refusal` and `observed_refusal`
strings instead of numbers (`observed_refusal` is `"none"` when nothing was
refused), and `passed` must equal `observed_refusal == expected_refusal`.

`cross_implementation` records agreement between two implementations of the
same origin (a ciw Rust kernel against ciw Python): numerically verified, never
independent.

`independent_check` = a check object plus `producer` and `checker`, each
`{"implementation": "...", "revision": "..."}`. The origin of an implementation
is its leading ASCII name token after NFKC normalization, casefolded
(`scipy.integrate.solve_ivp` → `scipy`). Producer and checker must each be
`ciw` or a recognised external family, and their origins must differ; the
rule is symmetric, so a pinned provider's result checked by a `ciw` reference
counts as independent just as `ciw` checked by the provider does. The
recognised external families are `scipy`, `sympy`, `mpmath`, `numpy`,
`cpython`, `zlib`, `git` and the pinned providers (`curved-surface-geodesic-sensitivity-runtime`,
`flat-torus-geodesic-reference`, `parameterized-lyapunov-stability-runtime`,
`scientific-computation-runtime`). Unknown families, non-ASCII look-alikes and
names that embed `ciw` (`ciw-rust`, `python:ciw`) are refused, so independence
cannot be minted by spelling. `ciw.*` code checking `ciw.*` code is *not*
independent and is refused; declare it as a `cross_implementation`,
`self_convergence` or `analytic` check instead. An independent check cannot use
the `cross_implementation` kind.

Domains: `mathematical`, `numerical`, `computational_pipeline`, `provenance`
(computational); `physical`, `calibration`, `sensor_performance` (need acquired
hardware — always `not_established` here); `machine_safety`,
`industrial_readiness`, `customer_demand`, `actuator_authority`,
`production_acceptance` (always `not_established`).

Report-level rules, enforced by `validate_report` and the runner:

- The primary label is the *weakest* established computational label among the
  findings, in the order `synthetic` < `analytic` < `provider_backed` <
  `numerically_verified` < `independently_verified`. It is `not_established`
  when no computational finding is established or any computational finding is
  refuted. It never depends on finding order, and strong findings never lift a
  weak one.
- Claims within a report are unique, and the physical-validation statement is
  derived from the findings; an edited statement is refused.
- Plan findings (the `plan` attached to a task whose requirements are missing)
  must all be `not_established`; blocked and deferred reports carry nothing
  else.
- A physical-domain finding with an acquisition record needs a hardware probe
  that succeeded in this run (`ctx.available("hardware:...")`) and a
  `raw_sha256` equal to the digest of an artifact the task retained. Without
  both, the task becomes blocked; an invented measurement cannot enter a
  report.
- A contract violation (`EvidenceRefusal`) inside a task becomes a blocked
  report that names the refusal; the rest of the queue continues.

Rules that are never relaxed:

1. Never fabricate physical results. A synthetic sensor experiment's
   conclusion about real sensor performance is a `sensor_performance` finding
   with an empty basis → `not_established`. Record it; do not omit it.
2. Phrase claims so that their checks pass when the claim holds. A
   counterexample is a claim that the counterexample exists, with a check that
   passes when the violation is observed, plus `counterexample={"statement":
   "<refuted general statement>", "witness": {...}}`.
3. Claims are deterministic text: never embed fresh identities (execution or
   result UUIDs, temporary paths, timestamps) in a claim, because the
   regression gate matches findings across runs by claim. Put such values in
   artifacts or in the finding's value when they are stable.
4. A finding that honestly records an unestablished computational claim in a
   completed task sets `expected_not_established=True` (exactly the boolean
   `True`) with no checks and no independent check. A computational finding
   whose checks fail is a refutation: it makes the report's primary label
   `not_established` and cannot be flagged as expected.
5. Give every numerical finding an `uncertainty`: a number with a stated
   meaning or an object such as `{"kind": "truncation_bound" |
   "monte_carlo_95ci" | "roundoff" | "reference_error", "value": ...,
   "basis": "..."}`. The report-level `uncertainty` answer summarizes; the
   per-finding values feed the uncertainty-budget table (T159).
6. Give every numerical finding a `tolerance={"abs": a, "rel": r}` for the
   regression gate. Values must be JSON (floats, lists, dicts, strings, bools).
   Choose tolerances that survive Linux/Windows and NumPy BLAS differences
   (typically `rel` 1e-6 for converged quantities; looser for orders/rates).

## Registering a task

```python
from .evidence import finding
from .registry import task

@task("T003", changed_files=("src/ciw/lab/geodesic_jacobi.py",),
      regression_tests=("tests/test_lab_geodesic_jacobi.py::test_integrator_orders",),
      requires=())
def integrator_orders(ctx):
    study = ctx.memo("convergence", compute_convergence)   # shared across tasks in a run
    ctx.artifact_json("orders.json", table)                 # retained, hashed
    ctx.artifact_text("orders.svg", svg.line_plot(...))     # deterministic figure
    return {"state": "completed", "fields": {...}, "findings": [...]}
```

`fields` supplies the report questions except the derived ones (evidence
status, physical validation status, generated artifacts, tests). Provide every
one of: `hypothesis`, `mathematical_model`, `input_data`, `observation_model`,
`expected_invariant`, `experiment`, `numerical_result`, `uncertainty`,
`failure_modes_checked`, `unresolved_assumptions`, `recommended_next_task`, and
`provider_runtime_identity` when a provider ran (otherwise the runner records
the built-in identity). An executed task's omitted answer is reported as
`Not stated by the implementation.`, never left empty. `changed_files` defaults
to the registration. `regression_tests` are pytest node ids
(`tests/test_x.py::test_f`, `tests/test_x.py::TestC::test_m`); the JUnit
record matches a node exactly or through its parametrized cases, and any
failed case fails the node.

States: `completed` (planned computation ran, checks passed), `partial` (some
planned parts could not run here; say which), `blocked` (a hard requirement is
unavailable), `deferred` (not attempted). `requires=("module:scipy",)`,
`("provider:csg",)`, `("tool:cargo",)`, `("hardware:nvidia-gpu",)` are hard
requirements (another kind is refused at registration); pass `plan={...}` with the static report fields so a blocked
report is still informative; `plan["findings"]` may carry the physical or
authority claims the blocked task cannot establish (they must validate as
`not_established`-producing findings). Soft optional checks use
`ctx.available(...)`. Tasks that read repository files (`examples/`,
`tests/fixtures/`) locate them with `ciw.lab.runner.repository_path(...)`,
which honours `CIW_LAB_REPOSITORY_ROOT` in the clean-room run and returns None
in an installed package without them (report blocked in that case).

## Code and tests

- Section module: `src/ciw/lab/<section_key with underscores>.py` (may add
  helper modules `src/ciw/lab/<name>.py`). NumPy only at import time; import
  scipy/sympy/mpmath lazily inside functions and degrade when absent.
- Deterministic: seeded `np.random.Generator(np.random.PCG64(seed))`, no wall
  clock in findings, no dict-order dependence. Keep elapsed-time measurements
  out of findings (they are not reproducible); retain them in artifacts.
- Budget: the whole section run ≤ 60 s and its tests ≤ 60 s on one CPU core.
  The clean-room gate pins `OPENBLAS_NUM_THREADS=1`; do not rely on BLAS
  threading, and avoid large dense solves where a structured solver exists.
  Each artifact is at most 2 MiB (`runner.MAX_ARTIFACT_BYTES`); retain sampled
  or aggregated tables rather than full trajectories.
- Tests: `tests/test_lab_<section>.py`, Python 3.11 and Windows compatible, no
  network, `pytest.importorskip` for optional modules, env-gated skips for
  providers (`CIW_LAB_<ROLE>_REPO`, `CIW_LAB_<ROLE>_PYTHON`). The clean-room
  gate sets these only for the roles `scripts/check_lab.py` binds (CSG, FTR,
  SCR and the PLSR/FTR interpreter), through `TEST_VARIABLES` in
  `scripts/reproduce_lab.py`; tests of any other role skip in CI. A new role
  needs its variable in `TEST_VARIABLES` and its provisioning in
  `scripts/check_lab.py` (`REPOSITORIES` and its pin) before its tests run
  there. Tests should call the task functions via
  `ciw.lab.runner.run_task` or the underlying computation, and assert the
  labels, not just the numbers.
- Style follows the surrounding code: module docstring stating scope and
  non-claims, sparse comments explaining invariants, error messages in sentence
  case without trailing period.

## Extending the queue

The packaged definition (`src/ciw/lab/queue.json`, T001–T168) is fixed. New
experiments are appended with a queue extension and an implementation module,
without editing package data:

```json
{"schema": "ciw.lab-queue-extension.v1",
 "section": {"key": "follow-ups", "name": "Follow-up experiments"},
 "tasks": [{"id": "T169", "title": "Measure the chord coefficient on a rolled coupon."}]}
```

```sh
ciw lab --extension follow-ups.json --module my_lab_tasks run T169 --output-dir results/lab
ciw lab --extension follow-ups.json --module my_lab_tasks next --retained results/lab
```

Extension task identities must follow every existing task, section keys must
be new, and each task declares exactly an `id` and a `title`. The module
registers implementations with `@task` exactly as section modules do.
`CIW_LAB_EXTENSIONS` and `CIW_LAB_MODULES` (`os.pathsep`-separated) select the
same extensions for subprocesses. The clean-room reproduction runs the packaged
queue only and removes both variables. A task without an implementation is
reported as deferred, never omitted. A module that fails to import registers
nothing, nor do the submodules it imported: its tasks are deferred with its
import error, and every later load imports it again, so a long-lived process
(the MCP server) picks up a repaired module. An extension module's error is
named by every extension task left without an implementation.
