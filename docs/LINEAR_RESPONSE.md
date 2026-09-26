# Linear response explorer

Change an input. Scale each change by its local contribution, add by output,
then compare the predicted output with direct model evaluation. The compact
terminal view exposes this operation before the equations and record details.

This is an explicit analytic educational preview on the existing Python/CIW
substrate. It needs neither Julia nor another provider. It is adjacent to the
[model cards and previews](MODEL_EXPLORATION.md), uses the same oscillator source
validator, and leaves the [declaration-only transformation contract](STATE_TRANSFORMATIONS.md)
and existing retained operations intact.

## Run and inspect

From the repository root after `python -m pip install -e ".[dev]"`:

```text
python scripts/check_linear_response.py create examples/linear-response/scalar.json --output results/linear-response/scalar.json
python scripts/check_linear_response.py inspect results/linear-response/scalar.json
python scripts/check_linear_response.py inspect results/linear-response/scalar.json --details
python scripts/check_linear_response.py create examples/linear-response/affine.json --output results/linear-response/affine.json
python scripts/check_linear_response.py create examples/linear-response/nonlinear.json --output results/linear-response/nonlinear.json
python scripts/check_linear_response.py create examples/linear-response/oscillator-rate.json --output results/linear-response/rate.json
python scripts/check_linear_response.py create examples/linear-response/oscillator-energy.json --output results/linear-response/energy.json
```

Creation refuses an existing output file: choose a new name for a changed
request. `inspect` reads saved values and checks content bindings and contribution
accounting. It evaluates no model, derivative, trajectory, or provider and
creates no verification occurrence. Explicitly run `create` to recompute.
`--details` expands the same saved calculation, not a second teaching model.
Exit status is zero on success and two on declared refusal or file errors.

The supplied requests also include scalar step halving, a composed vector map,
and the nonlinear origin case. Request fields are fixed: profile revision,
base point, perturbation, ordered input names/units, frame, anchor, and an
embedded oscillator source (or null for elementary profiles). Unknown fields,
nonfinite numbers, wrong shapes, mismatched metadata, and invalid anchors fail.
Saved data cannot supply formulas, imports, or executable paths.

## One calculation

For a declared map `F`, point `x0` and proposed variation `delta`:

```text
y0                = F(x0)
contribution[i,j] = J[i,j] * delta[j]
predicted_delta   = sum(contribution[i,:])
predicted_output  = y0 + predicted_delta
model_output     = F(x0 + delta)
actual_delta     = model_output - y0
residual         = model_output - predicted_output
```

Vectors are mathematical columns stored as JSON lists. Jacobian rows follow
output order and columns follow input order. A scalar derivative is `1 x 1`;
the derivative of a scalar observable with several inputs is a `1 x n` row
(the gradient transposed). The contribution calculation works with rectangular
and singular matrices without assuming an inverse.

| Name | Operation visible here |
| --- | --- |
| Derivative | One local input-to-output multiplier |
| Matrix-vector multiplication | Scale input changes, add by output row |
| Gradient | Contributions to one scalar output, displayed as a derivative row |
| Jacobian | All first-order input-to-output multipliers |
| Linearization | Baseline output plus predicted change |
| Chain rule | `J_G(F(x0)) @ J_F(x0)` in that order |
| Integration | Accumulate rates with a separately declared method; not performed here |

## Elementary profiles and reference values

The elementary profiles are dimensionless, with base and changed coordinates
bounded to `[-100,100]` and perturbations to `[-200,200]`.

| Profile | Base; variation | Predicted output | Model-evaluated output | Residual |
| --- | --- | --- | --- | --- |
| `scalar-square.v1`: `x^2` | `3`; `0.1` | `9.6` | `9.61` | `0.01` |
| Same scalar | `3`; `0.05` | `9.3` | `9.3025` | `0.0025` |
| `affine.v1`: `[[2,1],[-1,3]] @ x + [5,-2]` | `[3,4]`; `[0.1,-0.2]` | `[15,6.3]` | `[15,6.3]` | zero within rounding |
| `nonlinear-vector.v1`: `[x1^2+x2,x1*x2]` | `[2,3]`; `[0.1,-0.2]` | `[7.2,5.9]` | `[7.21,5.88]` | `[0.01,-0.02]` |

For the square the exact remainder is `h^2`, and the finite quotient is `6+h`.
The quotient at `h=0` is undefined (null); the analytic derivative still exists.
A numerical sweep illustrates the limit without proving it. The nonlinear
vector remainder is `[delta1^2,delta1*delta2]`; halving both changes quarters
that remainder until rounding matters. At `[0,0]` a variation `[0.1,0]`
has zero first-order prediction but a nonzero `[0.01,0]` change.

`composed-vector.v1` uses the nonlinear vector profile followed by
`G(u)=[u1^2,u2]`. Its Jacobian at `[2,3]` is `[[56,14],[3,2]]`.
`absolute.v1` refuses differentiation at zero even though a symmetric finite
difference there would be zero.

The affine profile has a nonzero offset and therefore is not a linear map;
its change relation is exactly linear in the mathematical model. Nonlinear
profiles are local approximations. The polynomial remainder rates above do
not extend to every differentiable function. Large in-domain perturbations
are displayed unchanged, including large residuals.

## Oscillator bridge: state, rate, and energy

Both oscillator profiles bind the existing validated initial state `[q0,v0]`,
hold all model parameters fixed, and validate the changed initial declaration
against the same [bounded source profile](JULIA_OSCILLATOR.md).

`oscillator-rate.v1` uses `A=[[0,1],[-omega_0^2,-2*gamma]]` and exposes
`z_dot=A@z`, `delta_z_dot=A@delta_z`. This is an exact change relation for the
declared linear vector field. **A maps state to rate, not to a state at a later
time.** Time evolution remains the existing solver's responsibility:
`z(t1)=z(t0)+integral(z_dot(t),t0,t1)`. The preview records declared solver
settings and time grid, with no trajectory execution reference and no claim
that the solver ran. Damping changes require a different investigation;
`A@delta_z` does not predict full trajectory sensitivity to gamma.

`oscillator-energy.v1` uses `E=0.5*m*(v^2+omega_0^2*q^2)` and the row
`J_E=[m*omega_0^2*q,m*v]`. For `m=2`, `omega_0=2`, `gamma=0.1`,
`z0=[1,-0.25]`, `delta=[0.1,0.2]`:

- Initial rate is `[-0.25,-3.95]`; rate change is `[0.2,-0.44]`.
- Initial energy is `4.0625 J`; predicted change is `0.7 J`.
- Actual energy change is `0.78 J`; remainder is `0.08 J`.

The exact energy remainder at fixed parameters is
`0.5*m*(delta_v^2+omega_0^2*delta_q^2)`.
Undamped energy conservation along a trajectory does not mean arbitrary
initial-state variations preserve the same energy value.

Each Jacobian entry declares output-unit/input-unit; every contribution in an
output row has that output's unit. The comparison reports residuals per
component, with `atol=1e-12` output units and `rtol=1e-12`, implemented as
`atol + rtol*abs(model_output)`. This is a display comparison threshold, not
a solver accuracy claim or an acceptance gate for a physical design. No norm
silently combines position rates, accelerations, or joules.

## Persistence, authority, and extension boundary

The only new envelope is `ciw.linear-response-preview.v1`. It binds the request,
fixed profile metadata, source and changed-source digests, fixed parameters,
ordered calculation and authority under one preview content digest. A digest
does not establish correctness or source authenticity. Inspection catches
inconsistent bindings and accounting; it cannot detect a coherent rewrite
of all model values and hashes without an external trusted record or explicit
recomputation. It does not report fresh mathematical verification.

The output is `model-evaluated`, never measured. Execution, result and
verification occurrence IDs are null. Source and operation references do not
promote hypothetical outputs to retained provider execution. Physical validity,
measurement uncertainty, admission, and actuation remain unestablished or
unperformed. Existing [caller-declared Jacobians](../examples/measurement-chain/README.md)
retain their original scope; this interface does not import or relabel them.
The raw `response_arithmetic` helper validates shapes and finite arithmetic;
it makes no claim that a supplied matrix is a derivative.

The thin script is the initial integration surface so existing shared CLI,
saved cards, `experiment.inspect`, and concurrent design-study work remain
compatible. No ranking, covariance propagation, new integrator or runtime
binding is introduced.

## Validation

```text
python -m pytest -q tests/test_linear_response.py tests/test_model_education.py tests/test_state_transformation.py tests/test_julia_oscillator.py
python -m compileall -q src
git diff --check
```

Golden arithmetic, nonlinear remainders, chain order, shape/domain refusals,
units, source immutability and offline inspection are testable without Julia.
Genuine Julia execution, calibrated physical claims, and educational transfer
remain separate gates; this interface demonstrates reusable calculation and
inspection, not learning gains or execution speedups.

## Exact square-model checks

The [exact-response checker](EXACT_RESPONSE.md) separates arithmetic, derivative
meaning and a radius-bound claim using reduced rational numbers. Its explicit
`check-exact` and retained `inspect-exact` commands share this terminal script.
It does not upgrade these floating-point previews or produce an SP1 proof.
