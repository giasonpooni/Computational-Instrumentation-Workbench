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
"self_convergence" | "exact_arithmetic" | "refusal", "reference": "<what was
compared>", "observed": <float>, "tolerance": <float>, "comparison": "abs_le" |
"le" | "ge", "passed": <bool>}`. The validator recomputes `passed` from
`observed` and `tolerance`; a mismatch is refused. Refusal checks use
`expected_refusal`/`observed_refusal` strings instead of numbers.

`independent_check` = a check object plus `producer` and `checker`, each
`{"implementation": "...", "revision": "..."}`. `ciw.*` code checking `ciw.*`
code is *not* independent (same origin) and is refused; declare it as a
`self_convergence` or `analytic` check instead.

Domains: `mathematical`, `numerical`, `computational_pipeline`, `provenance`
(computational); `physical`, `calibration`, `sensor_performance` (need acquired
hardware — always `not_established` here); `machine_safety`,
`industrial_readiness`, `customer_demand`, `actuator_authority`,
`production_acceptance` (always `not_established`).

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
   completed task sets `expected_not_established=True`.
5. Give every numerical finding a `tolerance={"abs": a, "rel": r}` for the
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
the built-in identity). `changed_files` defaults to the registration.

States: `completed` (planned computation ran, checks passed), `partial` (some
planned parts could not run here; say which), `blocked` (a hard requirement is
unavailable), `deferred` (not attempted). `requires=("module:scipy",)`,
`("provider:csg",)`, `("tool:cargo",)`, `("hardware:nvidia-gpu",)` are hard
requirements; pass `plan={...}` with the static report fields so a blocked
report is still informative. Soft optional checks use `ctx.available(...)`.

## Code and tests

- Section module: `src/ciw/lab/<section_key with underscores>.py` (may add
  helper modules `src/ciw/lab/<name>.py`). NumPy only at import time; import
  scipy/sympy/mpmath lazily inside functions and degrade when absent.
- Deterministic: seeded `np.random.Generator(np.random.PCG64(seed))`, no wall
  clock in findings, no dict-order dependence. Keep elapsed-time measurements
  out of findings (they are not reproducible); retain them in artifacts.
- Budget: the whole section run ≤ 60 s and its tests ≤ 60 s on one CPU core.
  Each artifact is at most 2 MiB (`runner.MAX_ARTIFACT_BYTES`); retain sampled
  or aggregated tables rather than full trajectories.
- Tests: `tests/test_lab_<section>.py`, Python 3.11 and Windows compatible, no
  network, `pytest.importorskip` for optional modules, env-gated skips for
  providers (`CIW_LAB_<ROLE>_REPO`). Tests should call the task functions via
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
same extensions for subprocesses and the clean-room reproduction. A task
without an implementation is reported as deferred, never omitted.
