# Native geodesic references in the shared bench

Two bounded mathematical references now use the same source catalog, execution
history, retained workspace and inspection surface as the other bench tools.
They run their native providers in pinned subprocesses. CIW retains their
declarations and results; it does not implement a replacement geodesic solver.

| Source kind / operation | Native owner and retained calculation | Scope |
| --- | --- | --- |
| `flat-torus-reference` / `ciw.flat-torus-reference.v1` | FTR `flat_geodesic_reference` and native trajectory sampling | Area-one flat quotient lattice with declared complex modulus, integer winding and start |
| `curved-path-transfer` / `ciw.curved-path-transfer.v1` | CSG `integrate_jacobi`, native transfer record, separation and covariance propagation | Declared constant curvature on a bounded arclength grid |

These are independent reference calculations. They do not infer a surface from
measurements, create a GSIE state, supply surveyed coordinates or qualify a
physical calibration. A flat quotient torus is not an embedded doughnut surface.
The curved transfer uses a curvature profile; its native record has no embedded
path geometry. Its uncertainty starts from the explicitly declared covariance.
Calibration stays unbound. Neither operation automatically feeds another instrument.

## Exact providers

Use Python 3.12 or newer and the CIW NumPy pin. Bind clean checkouts with the
original tracked bytes at these commits:

| Role | Repository | Revision | License at pin |
| --- | --- | --- | --- |
| `ftr` | [Flat-Torus-Geodesic-Reference](https://github.com/giasonpooni/Flat-Torus-Geodesic-Reference) | `dc918562cd9e351a65475d29f46963c9f2fd7db8` | MIT |
| `csg` | [Curved-Surface-Geodesic-Sensitivity-Runtime](https://github.com/giasonpooni/Curved-Surface-Geodesic-Sensitivity-Runtime) | `bbc535af29c30997e56fd120320c570830676462` | MPL-2.0 |

The executable pin declarations live in
[`geodesic_reference.py`](../src/ciw/geodesic_reference.py). Runtime identities
also bind the source tree, interpreter hash/version and dependency versions.
Replay allows a relocated checkout but requires matching computational identities.
Workspace JSON cannot supply executable bindings.

```sh
python -m pip install -e '.[dev]'
git -c core.autocrlf=false clone https://github.com/giasonpooni/Flat-Torus-Geodesic-Reference.git /trusted/references/ftr
git -C /trusted/references/ftr -c core.autocrlf=false checkout --detach dc918562cd9e351a65475d29f46963c9f2fd7db8
git -c core.autocrlf=false clone https://github.com/giasonpooni/Curved-Surface-Geodesic-Sensitivity-Runtime.git /trusted/references/csg
git -C /trusted/references/csg -c core.autocrlf=false checkout --detach bbc535af29c30997e56fd120320c570830676462
ciw serve --flat-torus-repo /trusted/references/ftr --curved-surface-repo /trusted/references/csg --output-dir results/references
```

In another terminal:

```sh
python examples/geodesic-reference/run.py
ciw send source.list
ciw send bundle.list
ciw send execution.list
ciw send workspace.save
```

The example submits [`flat-torus.json`](../examples/geodesic-reference/flat-torus.json)
and [`curved-path.json`](../examples/geodesic-reference/curved-path.json), executes
each, requests a fresh replay, inspects the original results, and saves the
common workspace. Additional bench startup bindings can share this same server.

## Evidence and inspection

Source envelopes are `ciw.flat-torus-reference-source.v1` and
`ciw.curved-path-transfer-source.v1`. Exact submitted bytes are retained as
evidence. Completed bundles use the matching `ciw.<kind>-session.v1` schema,
with native data inside their retained result. Execution and result IDs change
on replay; evidence identity and numerical content remain stable.

Both sources declare an `experiment_id` and the exact `configuration` policy
shown in their example. The flat source supplies `tau` and `start` as `[real,
imaginary]`, `winding` as two integers, and `samples` as an integer from 2 to 128.
The winding pair must be nonzero with each magnitude at most 16. The accepted
modulus has `|Re(tau)| <= 4` and `0.125 <= Im(tau) <= 8`; start components have
magnitude at most 8. Fractional or Boolean windings are refused before dispatch.
Native complex output values use `{re, im}` pairs. Trajectory parameters run
from 0 to 1; they are not observation times. Lengths use `normalized_length`.

The curved source supplies a strictly increasing `arclength` grid beginning at
zero (2–128 samples, end at most 8), constant `gaussian_curvature` with magnitude
at most 1, and `units` with length `m`, `mm` or `normalized_length` and angle
`radian`. Curvature is inverse length squared in that declared length unit;
no conversion occurs. Each step must satisfy `abs(K) * step**2 <= 0.01`
with a `1e-14` roundoff allowance at this boundary.
`initial_perturbation` is `[lateral displacement, heading]`; the covariance uses
the same order, so its entries have units length², length·radian and radian².
It must be a finite symmetric positive-semidefinite 2×2 matrix, with explicit
`basis: "assumed"` and a note. Each covariance entry has magnitude at most 1;
each initial perturbation component has magnitude at most 0.1. These are
numerical workload bounds in the selected units, not physical acceptance limits.
The configuration does not estimate covariance from data. Source envelopes are
limited to 32 KiB, and Boolean or nonfinite numerical values are refused.

The declared `relative_tolerance` lies in `(0, 0.01]` and concerns the native linearization validity
assessment, not RK4 integration error. The native validity remains heading-only;
no lateral-offset validity bound or step-refinement convergence is established.
The retained transfer record preserves those limitations alongside complete
samples, propagated covariance, separation, heading change and determinants.

The common interfaces are `source.add`, `operation.execute`, `bundle.get`,
`bundle.replay`, the `instrument` view of `experiment.inspect` (roles `ftr` and `csg`) and
`experiment.inspect`. The optional Godot Workbench tab renders the same
backend-provided panels and native context. Displaying a result does not execute
a provider or confer verification. These records are excluded from the `fusion` view of `experiment.inspect`.

Reopen the saved workspace with `ciw serve --workspace PATH` for provider-free
inspection. Replay without the relevant host binding returns an explicit
unavailable-operation error. A failed calculation leaves no partial bundle.
Same-runtime reproduction is labeled `independent: false`; the separate ICRH
checks below retain their own conformance scope.

## Verification

```sh
python scripts/check_geodesic_references.py --output-dir results/reference-gate
```

The gate obtains exact provider and harness pins, installs isolated CIW and ICRH wheels,
executes both references through the live WebSocket session, checks fresh replay
identities, and restores the workspace without providers. It rejects skipped
native tests. It then invokes the pinned ICRH checker over the generated original
and replay files. `--stack-root /trusted/references --icrh-repo /trusted/icrh`
uses existing exact checkouts without fetching their sources. Package installation
still requires the Python dependencies.

The gate pins ICRH to
[`dc4d826ecd1f28c1d55b724618380ce44e58bedd`](https://github.com/giasonpooni/Instrument-Conformance-and-Replay-Harness/tree/dc4d826ecd1f28c1d55b724618380ce44e58bedd).
Its profiles `icrh.flat-torus-reference.v1` and `icrh.curved-path-transfer.v1`
check retained identities and source bindings with independently implemented
analytic reference formulas. Flat fixtures exercise lattice normalization and
closure. Curved fixtures exercise zero, positive and negative constant curvature,
transfer matrices and propagation of a declared starting covariance. Inspection
of saved bytes is content conformance; it does not itself rerun a provider.

The `geodesic-references` provider gate runs on Ubuntu and Windows in the
[CI matrix](PIPELINES.md#provider-gates).
Tests include malformed declarations, bounded-input refusals, retained-record
tampering, native original/replay pairs and provider-free workspace restore.

Before making source repositories private, follow
[provider availability](PROVIDER_AVAILABILITY.md). The providers remain separate
dependencies; integrating them does not copy their code or remove their licenses.
