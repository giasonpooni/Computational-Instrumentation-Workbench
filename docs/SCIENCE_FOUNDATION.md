# Scientific foundation (`ciw.science`)

`ciw.science` is the workbench's language-neutral scientific layer. It lets a
geodesic result, a Jacobi-field uncertainty, a camera observation, a fused
tracker state, an FPGA telemetry capture and an authority decision sit in one
evidence ledger without losing units, frames, clocks, provenance or claim status.

```
geometry / numerical kernels        geometry.py, mesh.py, topology.py, solvers.py, oracles.py
        ↓
observation and sensor fusion       observation.py, fusion.py, frames.py, units.py
        ↓
execution and orchestration         experiment.py, runner.py, backends.py, agents.py, design.py
        ↓
evidence, provenance and replay     ledger.py, runner.replay, bundle.py, report.py
        ↓
authority and decision gates        claims.py, authority.py, governance.py
        ↓
hardware and physical validation    hardware.py (observation only), physical.py, usecase.py
```

The interface follows **define → inspect → simulate → measure → compare → retain → decide**.
It is not "ask for an answer": every step leaves a ledger entry that someone else can check.

## Quick start

```sh
python -m pip install -e ".[dev]"
ciw science solvers                                   # equations, capabilities, failure modes
ciw science spec compile examples/science/experiments/cylinder-chord.json
ciw science run examples/science/experiments/*.json --ledger work/ledger \
    --frames examples/science/frames/coupon-cell.json
ciw science replay cylinder-chord-vs-intrinsic --ledger work/ledger
ciw science status cylinder-chord-vs-intrinsic --ledger work/ledger
ciw science report --ledger work/ledger --output work/report.md
ciw science ledger probe work/ledger                   # tamper self-test on a copy
ciw science design examples/science/design/chord-or-curvature.json
ciw science bench --output work/bench --quick          # the whole loop in one ledger (~10 s)
```

Commands print JSON. The exit status is 0 on success, 2 on a declared refusal
and 1 on an unexpected error. From an installed wheel, pass
`--examples <checkout>/examples/science` to `ciw science bench`.
`python scripts/check_science.py` runs the full bench (about a minute),
replays everything, imports the signed bundle into a fresh ledger and checks
that the report is byte-identical.

## The five status statements

`ciw science status` (and every report) derives five separate statements from the ledger.
They are never merged into one verdict:

| Statement | Derived from |
| --- | --- |
| the measurement exists | an `observation` entry with `acquisition: physical` |
| the calculation completed | `execution` entries, all `completed` |
| the result is numerically stable | every `verification` passed and no replay diverged |
| the physical interpretation is unresolved / consistent / contradicted | only physical observations can resolve it |
| the decision is (not) authorized | `authority_decision` entries from the read-only gate |

## Capability map

Status: **implemented** means executable, tested and exercised by the bench;
**partial** names what is missing.

| # | Capability | Module(s) | Status and scope |
| --- | --- | --- | --- |
| 1 | Canonical scientific state model | `units`, `frames`, `_common`, `vocabulary`, `ledger` | Implemented. SI dimensions plus `px`/`tick` pseudo-dimensions; covariance converts with units (`S C S`); frames, clocks, calibrations and closed vocabularies for observables, claims, capabilities and modes. |
| 2 | Experiment specification language | `experiment`, `schemas/experiment-spec.v1.schema.json` | Implemented for surface geodesics, Jacobi fields, boundary-value (log-map) problems and mesh distances. Perturbations cover the initial heading only. |
| 3 | Experiment compiler and runner | `experiment.compile_spec`, `runner` | Implemented. Deterministic plan and job identities, sweeps, ensembles, bounded work, expected artifacts, acquisition requests for planned physical measurements. |
| 4 | Model and solver registry | `solvers` | Implemented: six solvers (intrinsic RK4 geodesic, Jacobi field, log-map shooting, extrinsic R³ geodesic, mesh heat method, Kalman fusion). Each declares equations, assumptions, domains, capabilities, method, precision, error behaviour, oracles and failure modes. |
| 5 | Reference-oracle layer | `oracles` | Implemented: 18 oracles (closed forms, extrinsic/intrinsic alternates, order-6 Gauss–Legendre reference, step-halving order, finite-difference Jacobi, speed and Clairaut conservation, unit and chart invariance, Christoffel-from-metric and curvature-from-Christoffel checks, three mesh oracles). No arbitrary-precision arithmetic. |
| 6 | Geometry and topology service | `geometry`, `mesh`, `topology` | Implemented: metrics, Christoffel symbols, curvature, geodesics, exp/log maps, Jacobi fields and conjugate points, geodesic curvature, a two-chart sphere atlas, singularity refusal, mesh geodesics, Euler characteristic, boundaries, orientability, genus, winding classes, 0-D persistence and bottleneck distance. |
| 7 | Observation model layer | `observation` | Implemented for intrinsic distance, camera chord, pinhole image residual, tracker position and encoder displacement. IMU orientation, filtered state and reconstructed geometry are declared observables whose predictors are refused as unbound. |
| 8 | Calibration and frame registry | `frames` | Implemented: every transform carries source, target, estimation time, calibration id/version, 6×6 covariance and a validity interval on a named clock. Transforms are static within their window. |
| 9 | Sensor-fusion engine | `fusion` | Implemented: raw → transformed → filtered candidate → candidate state → admitted state, as separate records. Linear constant-velocity model only. |
| 10 | Evidence ledger and replay | `ledger`, `runner.replay`, `backends.drift` | Implemented: hash chain, exact blobs, tamper detection, migration views, bitwise/tolerance replay, runtime-drift detection. |
| 11 | Claim and decision layer | `claims`, `authority` | Implemented. |
| 12 | Agent orchestration | `agents` | Implemented as typed, role-scoped proposals with deterministic validators, plus three deterministic agents (provenance auditor, ledger tamper probe, parameter counterexample search). No language-model agent is bundled. |
| 13 | Resource and backend orchestration | `backends` | Partial. Detects Python/NumPy, Julia, Rust, C++, CUDA/NVML, FPGA and remote backends and records selection reasons. Only CPU NumPy binds solvers. |
| 14 | GPU and energy telemetry | `backends.measured`, existing `ciw energy` | Partial. Science jobs record wall and CPU time and mark energy `not_measured`. GPU energy capture stays in the existing [energy bench](ENERGY_ACCURACY.md). |
| 15 | FPGA and hardware boundary | `hardware` | Implemented, observation only: CIWT v1 decoding, loss/duplicate/reorder/CRC/bitstream checks, clock alignment, bounded buffering, replay. Control requests are always refused (`control_path_unbound`). |
| 16 | Physical experiment manager | `physical` | Implemented: pre-registered protocols, GUM budgets with Welch–Satterthwaite coverage, En decisions, exclusions. The shipped measurements are synthetic and are reported as `accepted_synthetic_only`. |
| 17 | Active experiment design | `design` | Implemented: Monte Carlo expected information gain over declared hypotheses, costs and risks. For the shipped chord-versus-curvature question it recommends the cylinder coupon and reports that the cylinder cannot separate a curvature error from the null hypothesis. |
| 18 | Industrial use-case compiler | `usecase` | Implemented for eight domains with rule tables. Each requirement carries a rationale. Missing capabilities are checked against the registered solvers. |
| 19 | Scientific governance | `governance` | Implemented: a weighted Ed25519 council for schemas, pins, datasets, validators, protocols and migrations. Verification outcomes and claim truth are refused as subjects. |
| 20 | Offline and low-resource operation | `bundle`, `ed25519` | Implemented: deterministic signed evidence bundles, offline inspection, import into a new ledger, signed updates, pure-Python Ed25519. Divergent ledgers are not merged. |
| 21 | Security and adversarial validation | tests, `agents` | Implemented; see below. |
| 22 | Human-readable scientific interface | `cli`, `report` | Implemented for the terminal, Markdown and LaTeX. The Godot desktop does not yet show science ledgers. |

### Minimum environment

| Minimum item | Where |
| --- | --- |
| Typed notation/schema registry | `vocabulary`, `ledger.BODY_SCHEMAS`, `schemas/` |
| Experiment specification format | `ciw.experiment-spec.v1` |
| Geometry and numerical-kernel registry | `solvers.default_registry()` |
| Observation/frame/clock model | `observation`, `frames` |
| Evidence ledger | `ledger` |
| Replay engine | `runner.replay`, `hardware.replay` |
| Provider and runtime registry | `backends.runtime_identity` (with a source digest), governance provider pins |
| Sensor-fusion boundary | `fusion` stages and admission |
| Read-only authority gate | `authority` |
| Reproducible report generator | `report` |
| Synthetic experiment bench | `bench`, `ciw science bench` |
| Physical validation bench | `physical` protocol manager is ready; **no physical data has been acquired** |

## Evidence records

A ledger directory holds `ledger.jsonl` (one canonical-JSON entry per line) and
`blobs/<aa>/<sha256>`. Each entry has a sequence number, kind, body schema, the
previous entry's identity, a UTC timestamp, sorted references to earlier
entries and blobs, and a body. Its identity is the SHA-256 of everything else,
so editing, deleting, reordering or truncating a line breaks verification.
References can only point backwards, which keeps evidence graphs acyclic. Blob
names are validated as digests before a path is formed. Registered migrations
produce *views* in newer body schemas, and the retained bytes never change.

A run of one specification appends, in order: the exact specification bytes,
the plan, any acquisition requests, the runtime identity, then for each job a
parameter identity, an execution record (completed or refused, with backend
and timing), the numerical result and its content identity, the oracle
verification, and any synthetic observation with its comparison and diagnosis.
It then adds ensemble checks and the claims that the evidence admits. Replay
adds runtime identities and replay receipts. The original results are never
replaced.

## Observation versus model: the chord example

`cylinder-misdeclared.json` generates synthetic camera chords but declares the
instrument as reporting intrinsic distance. The comparison is inconsistent
(up to about 1800σ circumferentially). `explain` reports the data as consistent
with `camera_chord`, and along the cylinder axis, where chord and intrinsic
distance coincide, as indistinguishable. The measurement is never relabelled;
the diagnosis is retained beside it. `design.rank` uses the same geometry to
show why a cylinder is the cheapest coupon that separates chord confusion from
a curvature error.

## Security and adversarial validation

| Threat | Handling | Tests |
| --- | --- | --- |
| Altered evidence bytes, deleted/reordered/truncated entries | hash chain, strict open | `test_science_foundation.py`, `agents.adversarial_ledger_probe` |
| Corrupted or planted blobs, path traversal | digest-named blobs, unexpected-file scan | `test_science_foundation.py`, `test_science_bundle.py` |
| Provider substitution / drift | source digest in runtime identity; replay receipts | `test_science_experiments.py::test_replay_detects_provider_drift_and_divergence` |
| Forged verification or approval | only validators write verifications; Ed25519 approvals | `test_science_experiments.py::test_authority_gate_is_read_only`, `test_science_governance.py` |
| Replay tampering | parameter identities re-checked before replay | `runner.replay` |
| Stale state | transform/clock validity windows; fusion max age; authority `max_evidence_age_s` | `test_science_foundation.py`, `test_science_fusion.py` |
| Frame and clock confusion | clockless timestamps, unmapped clocks, ambiguous chains, simulation-to-physical mappings refused | `test_science_foundation.py`, `test_science_fusion.py` |
| Malformed covariance | square, symmetric, PSD, finite, never repaired | `test_science_foundation.py`, `test_science_fusion.py` |
| Oversized artifacts, zip bombs, DoS inputs | entry/blob/member/ratio bounds, sweep and step budgets, mesh vertex limits | `test_science_bundle.py`, `test_science_experiments.py`, `test_science_mesh.py` |
| Unsafe hardware commands | `request_control` always refuses; the authority gate refuses equipment actions without a bound control path | `test_science_hardware.py`, `test_science_experiments.py` |
| Agents escalating | role-scoped proposal kinds; forbidden kinds for all roles; accepted proposals are never applied | `test_science_experiments.py::test_agent_proposals_are_typed_and_never_applied` |

## Scope and limits

- **Numerics.** Binary64 only. Fixed-step RK4 has O(h⁴) error, estimated by step halving, with no adaptive control. Log maps are refused when the exponential-map Jacobian condition number exceeds 1e5. The sphere is the only multi-chart atlas.
- **Uncertainty.** Jacobi propagation is first order and covers the initial heading only. Ensemble checks compare it with Monte Carlo using a sample-standard-deviation test at |z| ≤ 4. Transform covariance is first order, with independent edges.
- **Meshes.** Heat-method distances converge at roughly first order, with larger errors near the source and the cut locus. Dense solves are limited to 3000 vertices. The edge-graph distance bounds the polyhedral distance, not the smooth one.
- **Fusion.** Linear constant-velocity Kalman filtering. Shared calibration biases are treated as white noise, and there is no retrodiction.
- **Observations.** Synthetic observations are generated from the declared model and are always labelled `synthetic`. They can never support a `measured` claim or resolve a physical interpretation.
- **Authority.** The gate only evaluates and records; it never dispatches. It declares no equipment control path, so actuation is always refused.
- **Hardware.** Decoding only. CRC-32 detects accidents, not attackers. The bitstream identity is what packets report, not an attestation.
- **Bundles and governance.** Signatures show who signed, not that the content is true. Key distribution, rotation and revocation are the caller's responsibility. The Ed25519 implementation is pure Python and not constant-time.
- **Design and use cases.** Costs, risks, priors and rule tables are declared inputs, printed next to every recommendation. They are not discovered facts.
- **Physical validation.** The protocol manager and GUM budget are ready, but no physical coupon has been measured. Every shipped measurement is synthetic.
