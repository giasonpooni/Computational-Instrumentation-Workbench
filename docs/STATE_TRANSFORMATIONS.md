# State-space transformation contract

CIW treats a computation as a declared transformation between typed state
spaces:

```text
(input state, transformation, constraints, invariants, evidence)
                              -> output state
```

The state spaces carry coordinate names, units, roles, reference frames and
clock declarations. A transformation describes the mathematical operation and
its parameters. Constraints describe the admissible domain. Invariants state
what the operation is required to preserve. Evidence references identify the
records on which the declaration depends.

`src/ciw/state_transformation.py` implements the provider-free contract
`ciw.state-transformation-contract.v1`. `seal_contract` creates a
content-addressed declaration record and `validate_sealed` reopens it. These
functions validate structure and identity only: they do not execute a filter,
prove that an invariant held, estimate a state, admit a result or authorize an
actuator.

For a telemetry filter, an appropriate contract can state that timestamps stay
ordered, units and frames are preserved or explicitly transformed, covariance
coordinates remain aligned, and source lineage remains attached. The actual
filter operation must still emit its own operation, execution, result and
verification records. A valid contract is the specification of the claim, not
evidence that the claim occurred.

This contract is deliberately language-neutral so Python, Julia, native
providers and future educational model cards can describe the same state
transition without collapsing their execution identities.

