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
findings. Each report has a content identity; editing a retained report makes
it fail validation.

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
ciw lab verify --retained lab --fresh results/lab
```

`--provider ROLE=PATH` binds a pinned provider checkout or interpreter; tasks
verify the checkout revision and tree against CIW's own pins before using it
and run provider code in a subprocess. `ciw lab verify` compares a fresh run
with retained reports: same states, same labels, and finding values within
each finding's declared regression tolerance. It exits 3 on any difference.

`scripts/reproduce_lab.py` is the one-command clean-room reproduction: it
builds a wheel, installs it with the lab extras into a new virtual
environment, runs the lab tests with a JUnit record, runs the whole queue and
verifies it against `lab/`.

## Retained evidence

`lab/` holds the retained run: `reports/T*.json`, `artifacts/T*/` (tables,
SVG figures, drafts and ledgers), `queue-state.json` and `REPORTS.md`, the
full report book. Aggregates generated by the research tasks include the
counterexample catalogue (`artifacts/T157/COUNTEREXAMPLES.md`), uncertainty
budgets (`T159`), the release report (`T165`), unresolved assumptions
(`T166`) and what remains unmeasured (`T167`).

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
