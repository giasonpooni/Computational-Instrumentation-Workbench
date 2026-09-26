# Exact justification for a linear response

The [linear-response explorer](LINEAR_RESPONSE.md) shows how input changes
contribute to an output. Its square-model extension checks three distinct
claims using exact rational arithmetic: the reported arithmetic, the meaning
of the local multiplier, and the approximation bound.

This is an executable Python reference checker for one fixed mathematical
model. It produces no SP1 proof or formally checked theorem. The existing
[proved-heat operation](PROVED_HEAT.md) retains its own registered guest and
claim; it cannot attest this different computation.

## Run the checker

From the repository root with the existing development dependencies installed:

```text
python scripts/check_linear_response.py check-exact examples/linear-response/exact-square.json --output results/exact-response/accepted.json
python scripts/check_linear_response.py inspect-exact results/exact-response/accepted.json
python scripts/check_linear_response.py inspect-exact results/exact-response/accepted.json --details
python scripts/check_linear_response.py check-exact examples/linear-response/exact-square-wrong-multiplier.json --output results/exact-response/rejected.json
```

`check-exact` performs a fresh local calculation and saves the report. Exit
codes are **0** for accepted, **1** for a well-formed rejected candidate,
and **2** for malformed input or file errors. Rejected candidates retain their
failed checks. Existing output files are refused; use a new name to recheck.
`inspect-exact` returns zero when the saved structure and bindings are intact,
including for a report of rejection. It does not rerun the mathematical checks.

The normal `create` and `inspect` floating-point commands keep their original
behavior and record format. Neither route silently turns a floating-point
value into a rational guarantee.

## One small operation, three claims

For the dimensionless real function `f(x)=x^2`, let the base point be `x`,
the proposed variation `d`, and a nonnegative radius `r`.

```text
baseline = x^2
local multiplier = 2*x
prediction = baseline + local multiplier*d
model output = (x+d)^2
residual = model output - prediction = d^2
```

The general identity follows by multiplication:

```text
(x+u)^2 = (x+u)*(x+u) = x^2 + 2*x*u + u^2.
```

For nonzero `u`, the difference quotient is `2*x+u`, whose limit at zero
is `2*x`. Subtracting the linear part leaves `u^2`. Since `r>=0`,
`abs(u)<=r` implies `0<=u^2<=r^2`. This establishes the mathematical
neighborhood bound for **all real** `u` in the stated interval, not just
the candidate or sampled points. The code applies this documented identity
to exactly represented rational inputs; a proof assistant has not checked
the derivation or its correspondence to the Python implementation.

| Local check | What it checks |
| --- | --- |
| Arithmetic | Reported prediction equals baseline plus multiplier times variation; reported residual equals model output minus prediction |
| Meaning | Baseline equals `x*x`, multiplier equals `2*x`, model output equals `(x+d)*(x+d)` for the fixed square profile |
| Approximation | `abs(d)<=r`, residual equals `d*d`, and the reported bound equals `r*r` |

All three must pass. A smaller or more generous claimed bound is not silently
substituted for the profile's exact `r^2` bound. A well-formed variation outside
the radius is rejected rather than shrunk.

At `x=3`, `d=r=1/10`, the exact prediction is `48/5`, the model output is
`961/100`, and the residual and neighborhood bound are both `1/100`.
The wrong-multiplier example uses `5` with internally consistent arithmetic:
prediction `19/2` and residual `11/100`. Its arithmetic passes while meaning
and approximation fail. A zero local multiplier at `x=0` likewise does not
erase the nonzero second-order change for `d!=0`.

## Representation and bounded inputs

Each number is a JSON object with integer `numerator` and positive integer
`denominator`. Fractions must already be reduced; zero must be `0/1`.
Booleans, floats, numeric strings, unreduced fractions and extra keys fail.
There is no decimal rounding or numeric comparison tolerance.

Input numerator magnitudes and denominators are at most `1,000,000`.
Base, variation and radius magnitudes are at most `100`; radius is nonnegative.
The changed point and the entire neighborhood `[x-r,x+r]` must lie in
`[-100,100]`. These are software profile limits, not physical validity limits.
Candidate output rational limbs are bounded by `2^255-1`. Python uses exact
arbitrary-precision integers: that guard limits retained number size and
does **not** claim signed 256-bit machine arithmetic or overflow equivalence.
Other readers must preserve integer values exactly rather than parse them
through binary64. Files are limited to 64 KiB; duplicate JSON keys fail.

`ciw.exact_response.make_candidate(x, delta, radius)` is an explicit producer
for this encoding. `check_candidate` checks the resulting candidate separately.
The producer's use of the same model is not independent evidence of correctness;
the tests also use hand-written expected fractions and deliberately false
candidates. Only the fixed profile is accepted: saved data cannot provide
expressions, code, imports, checker selection or executable paths.

## Retention and authority

The new envelope is `ciw.exact-response-check.v1`. It retains the candidate,
code-owned specification, three outcomes and exact content bindings. The
specification includes the model, derivative, identity, bounds, rational
encoding, units, frame, clock meaning, checker revision and acceptance policy.
The statement digest binds that specification and the entire candidate.

Report and statement identities are content identities. They are not operation,
execution, result, proof or verification occurrence identifiers. Repeating a
check of the same candidate may yield the same report content identity.
No admission or hardware authority is granted. Measurement uncertainty remains
undeclared; a model identity does not establish physical validity.

Saved inspection checks fields, domain/encoding declarations, digest bindings,
authority, and agreement between retained outcomes and their retained aggregate.
It does not rerun the arithmetic/meaning/approximation predicates and cannot
authenticate a report rewritten and resealed consistently. The display says
**retained report, not freshly rechecked**. Use `check-exact` with the retained
candidate to calculate again explicitly. Reopening never becomes a fresh
cryptographic verification.

The exact bound applies to the rational model output. It does not bound the
rounding error of the binary64 explorer, establish numerical solver accuracy,
or certify the oscillator's physical behavior.

## Route into SCR and SP1

SP1 attests execution of a supplied program, so checker correctness remains an
application obligation. Proof verification also needs the expected guest/key
and its public values. See Succinct's [security model](https://docs.succinct.xyz/docs/sp1/security/security-model)
and [off-chain verification](https://docs.succinct.xyz/docs/sp1/generating-proofs/off-chain-verification).

Follow the existing [Julia/SP1 contract](JULIA_SP1.md), retaining these gates:

1. Review the reference checker and exact mathematical claim, then establish
   Rust/reference agreement, including rational reduction and all intermediate
   arithmetic bounds. Formal soundness and source/binary correspondence remain
   explicit obligations.
2. Register a distinct SCR descriptor, guest and reproducible build. Do not
   reuse the heat operation, its guest digest, or its verification records.
3. Commit model, arithmetic, radius/domain, candidate outputs and checking policy
   in bytes actually consumed by the guest. SCR specification identity includes
   configuration; computation identity alone does not. Bind the accepted outcome
   and compare expected commitments during independent verification.
4. Exercise genuine proof generation and fresh verification, wrong guest/input/
   configuration/candidate rejection, and corrupted proof rejection. Repeat
   through the installed package before reporting integration.
5. Measure execution, proof and verification cost separately. If confidentiality
   is required, specify the proof mode and prover trust separately; this reference
   checker provides no privacy mechanism.

The existing heat pins remain unchanged: SCR
`a59aba283b0304faeeb3e5d305087e7709e171ca`, upstream SP1
`b38b61209e45e969289e70d5cf79dc763460bc41`, backend `sp1-cpu v6.1.0`.
No new guest hash, provider pin, operation registration or proof artifact is
invented by this increment. Interactive previews remain lightweight; proof
production is a separate explicit future operation.

## Tests

```text
python -m pytest -q tests/test_exact_response.py tests/test_linear_response.py tests/test_model_education.py tests/test_state_transformation.py tests/test_proved_heat.py tests/test_proved_heat_session.py
python -m compileall -q src
git diff --check
```

These exercise reference arithmetic, independent golden fractions, malformed
encoding, semantic rejection, binding/authority tampering, and evaluation-free
inspection. Existing mock/protocol tests do not satisfy SP1 cryptographic gates.
