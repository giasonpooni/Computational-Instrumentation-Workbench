# Retained model and state → PLSR

The shared `ciw.identified-stability.v1` operation consumes one selected
`identified-design` bundle. It evaluates the retained SIDT discrete transition
matrix and GSIE predicted mean through PLSR at commit
`19ea6967060166ba09db6cd4563bd87bd6b3d196`.

Prepare a source declaration from a saved native identified-design bundle:

```sh
python examples/identified-stability/make_source.py identified.json stability-source.json
```

Import `stability-source.json` as workbench kind `identified-stability`, then
execute `ciw.identified-stability.v1` with its `source_id` and the selected
`upstream_bundle_id`. The operation returns one retained native verdict with a
fresh execution identity, reproducibility verification and explicit replay.
The source provider is an operator-bound PLSR checkout; `jsonschema` is required
by its native model-artifact loader. Inspection and workspace restoration do
not execute or import PLSR.

The helper explicitly authors a synthetic identity-matrix certificate. It does
not solve for a certificate or establish a physical equilibrium. Received
model artifacts must already carry their native digest. The workbench verifies
that digest and the model/state occurrence bindings before evaluating them.

The bounded operation supports one to eight coordinates in the same declared
unit, a discrete linear model, a supplied exactly symmetric positive definite
quadratic certificate and a declared zero equilibrium. The certificate and
margin unit is `1/(u*u)` for common coordinate unit `u`. No unit, frame or
discrete-to-continuous conversion is performed.

The original synthetic two-reservoir identification has a neutral conservation
mode. With the identity certificate, its native status is
`NUMERICAL_INCONCLUSIVE`; that outcome is retained. Setting `level=0` in the
helper produces `OUTSIDE_LEVEL_SET` for the nonzero retained state. These are
useful demonstrations of a completed computation that does not certify the
requested inequality.

State covariance remains attached to the selected state as context. Parameter
covariance remains unknown. Neither is silently folded into a robust
certificate claim. The native three booleans, status code, resolution, margin,
quadratic values, matrices and `NOT_CHECKED` proof status remain separately
inspectable. The live view does not create another fusion state.
