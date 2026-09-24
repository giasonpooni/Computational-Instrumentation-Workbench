# Thermal observer reference

This example retains a two-capacity thermal observer request in the shared
workbench and executes the provider-free `ciw.thermal-observer.v1` operation
over it. The source declares a dissipative core/shell RC model, a prior, noise
covariances, three constant-input steps, a sensor selection budget and a
synthetic evaluation trajectory with one held-out step. The noise law is
declared by the operator and is not authenticated; nothing touches hardware.

```sh
ciw source add --kind thermal-observer --file examples/thermal-observer/source.json
ciw operation execute ciw.thermal-observer.v1 --source SOURCE_ID
```

No repository binding is needed; the operation is served by the independent
Python reference in `src/ciw/thermal_reference.py`, and the retained result
records that reference's own rendering and solver provenance. Reopening a saved
workspace validates the retained result against the contract's tolerances
without a provider; `bundle.replay` records a new execution and result
occurrence whose numerical identity must match the original.

`make_source.py` is the authoring script for `source.json`. The committed file
is the retained input: regenerating it on another platform may change the last
digit of a few floating-point values, because the discretized model matrices
come from an eigendecomposition.
