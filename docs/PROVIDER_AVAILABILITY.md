# Provider availability and private provisioning

CIW can run the adapter, retained-telemetry and calibrated-observable gates
against explicitly provisioned local provider checkouts. This removes anonymous
GitHub cloning from those gate invocations. It does not make the complete
portfolio ready for a repository-visibility change: other workflows, optional
package dependencies and nested provider helpers still require access review.
No credential configuration or repository visibility is changed by these gates.

## Existing-checkout gates

Provision the exact commits from the relevant runtime manifest before starting
the gate. A branch name or a newer compatible-looking revision is insufficient.
The flags select directories; they do not select alternate numerical versions.

| Gate | Manifest | Directory names under `--stack-root` |
| --- | --- | --- |
| `scripts/check_adapters.py` | `src/ciw/pipelines/providers/{rci,fsrt,jspt,gte}.json` | `rci`, `fsrt`, `jspt`, `gte`; also `rci-legacy` and `fsrt-legacy` unless a separate historical root is supplied |
| `scripts/check_telemetry.py` | `src/ciw/telemetry-runtimes.json` | `Provenance-Preserving-Data-Acquisition`, `Streaming-Telemetry-Feature-Extraction`, `Geometric-State-Inference-Engine`, `State-Estimation-Evaluation-Testbed`, `Constraint-Based-State-Reconciliation` |
| `scripts/check_calibrated_observable.py` | `src/ciw/calibrated-observable-runtimes.json` | `fsrt`, `tbrt`, `mcur`, `oit`, `gsie`, `cbsr`, `fdir`, `set` |

These layouts use ordinary directories and need no symlinks on Windows. Different
lanes pin different revisions of some providers; use separate checkout roots.
Keep the interpreter and dependencies required by each gate installed separately.

```powershell
python scripts/check_adapters.py --stack-root C:\ciw-providers\adapters --historical-stack-root C:\ciw-providers\historical
python scripts/check_telemetry.py --stack-root C:\ciw-providers\telemetry
python scripts/check_calibrated_observable.py --stack-root C:\ciw-providers\calibrated
```

The separate historical root contains `rci` and `fsrt` at their first historical
manifest entries. Without `--historical-stack-root`, the adapter gate uses
`rci-legacy` and `fsrt-legacy` below its current stack root. Missing historical
checkouts fail the local gate; they do not trigger a network fallback.

Before starting the tests, all selected checkouts must have the pinned HEAD,
unchanged tracked/index content and exact committed file bytes. Unexpected
untracked or ignored files are refused, apart from the same recognized runtime
cache directories used by the subprocess adapter. Git cleanliness alone cannot
hide line-ending conversion or `assume-unchanged` source edits. The checks read
the repositories with optional index updates disabled; they do not checkout,
reset, fetch, repair, or rewrite their configuration. Operation-specific runtime
checks still enforce source entry points and executable/dependency identities.

Without these flags, the gates retain their existing temporary clone-and-test
behavior. Their checkout commands preserve committed bytes with
`core.autocrlf=false`. Existing operator checkouts are never normalized in place.

## Remaining access dependencies

- **Cross-repository Actions access:** the workflow's own `GITHUB_TOKEN` is scoped
  to its repository; `contents: read` does not grant access to every private
  provider. Provisioning private dependencies needs an independently authorized
  access mechanism covering the selected repositories. See
  [GitHub's token scope documentation](https://docs.github.com/en/actions/concepts/security/github_token).
- **Optional PLSR installation:** `pyproject.toml` still has a pinned
  `git+https://github.com/...` dependency in `[project.optional-dependencies].plsr`.
  Installing that extra would require authenticated Git access or an explicitly
  reviewed package-distribution change. The local stack flags do not affect pip.
- **Exchange CI:** `scripts/check_exchange.py` clones SET, PPDA and SCR at their
  declared pins. Those pinned cross-repository checkouts need their own
  private-access provisioning before a visibility change.
- **Nested ESM provisioning:** `scripts/check_workbench_candidates.py` clones ESM
  and invokes ESM's pinned `scripts/check_calibrated_workbench.py`, which provisions
  its own provider graph. Existing `--esm-root`, `--fixture-root` and
  `--telemetry-stack-root` options avoid some outer clones, but do not automatically
  provision every nested replay dependency. Review the complete graph and exact
  fixture/runtime bindings.
- **Historical availability:** retained investigations can require historical
  commits even after current pins advance. Keep those commit objects and matching
  interpreter/dependency environments accessible. Private mirrors must preserve
  the reviewed history rather than substituting their newest branch.
- **PPDA and SCOUT:** PPDA retains a vendor gitlink. CIW's telemetry path executes
  only the separately hashed standalone bridge and can use an uninitialized
  vendor submodule. Other PPDA paths may need the exact SCOUT submodule and its
  own access. The provisioning check does not download it or expand CIW's
  executable scope; initialized submodule drift fails cleanliness validation.
- **Provider licences:** retain each provider's licence, notices and applicable
  submodule obligations when copying or distributing checkouts or built artifacts.
  Directory provisioning does not change those terms.

The shared-session operation bindings already accept explicit local paths.
Availability of those paths is distinct from numerical readiness, successful
replay, verification scope and physical calibration.

## Numerical readiness

The three former scaffolds now provide executable native packages, command-line
interfaces, reference fixtures and Linux/Windows installed-package tests.
CIW binds their exact commit and tree identities from
[`geometry_research.py`](../src/ciw/geometry_research.py).

| Provider | Implemented bounded profile | Remaining scope |
| --- | --- | --- |
| Covariance Geometry and Geodesic Testbed | Affine-invariant SPD distance and sampled geodesics, with eigenvalue and numerical-budget evidence | Additional metrics and calibrated covariance sources |
| Intrinsic Surface Geodesics Testbed | Mesh-edge Dijkstra paths, quality records and Euclidean lower bounds | Continuous surface paths and certified discretization error |
| Translation Surface Dynamics Explorer | Exact rational flows on connected square-tiled surfaces, with explicit partial outcomes | General polygon gluings and singularity continuation |

See [executable geometry providers](GEOMETRY_RESEARCH.md) for binding flags,
request profiles, trust limits and the mandatory installed-workbench gate.
The flat-torus reference and JSPT covariance transport retain their separate
identities. Mathematical execution alone does not establish physical validity.

## Validation

```sh
python -m pytest -q tests/test_provider_provisioning.py
```

The tests create local Git fixtures and prohibit provider network clones. They
exercise all three flags, separate historical bindings, missing/wrong/dirty
checkouts, raw-byte drift hidden from normal Git status, ignored source files,
and an uninitialized pinned gitlink. Successful reuse preserves HEAD, index and
repository configuration. Actual numerical gates remain separately required.
