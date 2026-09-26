# Choosing a bounded change

The linear-response explorer answers what a proposed change would do. This
increment asks which change best approaches a target, using one dimensionless
square-model example. JuMP expresses the quadratic problem, HiGHS computes a
numerical candidate, and a separate Python rational checker assesses it.

This is a source-checkout study through a small terminal script. It is not a
new registered SCR operation, a general optimization service, a JuliaControl
integration, or an SP1 guest. The existing shared operation and proof identities
remain unchanged.

## Problem and scope

At a fixed base point `x`, the square model has `y0=x^2` and `j=2*x`.
Choose a scalar variation `d` by solving:

```text
minimize (y0 + j*d - target)^2 + lambda*d^2
subject to lower <= d <= upper, lambda > 0.
```

Coordinates, response, target and objective are dimensionless. The frame is
dimensionless Cartesian; no acquisition clock or physical calibration applies.
This example does not combine physical units in a Euclidean objective.
The strictly positive regularizer makes this fixed local quadratic strictly
convex, even at `x=0`, where the local response vanishes.

After selecting `d`, the checker also evaluates `(x+d)^2`. That nonlinear
output is distinct from the prediction `x^2+2*x*d`, whose residual is `d^2`.
Optimality of the local quadratic is not optimality for the full nonlinear
design problem. Reaching an optimum also does not mean reaching the target:
the zero-response example selects `d=0` but still misses its target.

## Run without a solver: exact reference certificates

Use the repository's existing Python development environment:

```text
python scripts/check_design_adjustment.py check examples/design-adjustment/interior-certificate.json --output results/design-adjustment/reference.json
python scripts/check_design_adjustment.py inspect results/design-adjustment/reference.json
```

Four hand-checkable examples are supplied:

| Example | Exact selected change | Local objective and lower bound |
| --- | --- | --- |
| `interior`: x=3, target=19/2, lambda=1, bounds [-1/10,1/10] | 3/37 | 1/148 |
| `boundary`: same, target=10 | 1/10 | 17/100 |
| `zero-response`: x=0, target=1 | 0 | 1 |
| `fixed-bound`: x=3, target=10, bounds [0,0] | 0 | 1 |

These exact fixtures have zero optimality gap. They are independent numerical
references, not claims that a Julia solve has run.

## Run the JuMP producer

Provision Julia and the checked-in environment separately from serving a study.
Use an explicit operator-owned depot if needed. Instantiation uses the generated
Manifest; do not update it silently during a solve.

```text
julia --startup-file=no --project=runtimes/jump-design -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"
python scripts/check_design_adjustment.py solve examples/design-adjustment/interior.json --julia /path/to/julia --depot /path/to/depot --output results/design-adjustment/solved.json
python scripts/check_design_adjustment.py inspect-study results/design-adjustment/solved.json
```

The Python caller snapshots the exact worker, Project and Manifest bytes before
launching the worker. Requests and responses are bounded TOML, containing only
an allowlisted schema and numeric data. They cannot supply code, executable
paths, package names or solver settings. Request validation precedes provider
loading. Standard output is protocol data; diagnostics go to standard error.
The worker constructs and solves the JuMP model; no analytical solution is
substituted for `optimize!`.

The worker fixes one solver thread, a 10-second solver time limit, a 10,000
quadratic-iteration limit, primal/dual feasibility tolerances of `1e-9`, and
random seed zero. These settings do not establish exact feasibility. The host
has its own process timeout (120 seconds by default, including startup), disables
startup files, requests one Julia thread and enables Pkg offline mode. It does
not provision packages during a solve. Package lookup is restricted to the
snapshot project and Julia standard library. Julia and its packages remain trusted
executable code, not a sandboxed mathematical language.

`check` and `solve` return zero for accepted exact checks, one for a well-formed
rejected candidate and two for malformed input, unavailable provider, timeout
or file errors. They refuse existing output paths. Inspection returns zero for
intact retained reports, including reports of rejection; it does not rerun work.

## Floating-point output and the checked candidate

JuMP/HiGHS uses floating-point arithmetic and solver tolerances. The raw output,
objective, statuses and reported package versions remain retained as producer
data. An `OPTIMAL` status is not taken as an exact certificate. See JuMP's
[solution statuses](https://jump.dev/JuMP.jl/stable/manual/solutions/) and
[tolerance guidance](https://jump.dev/JuMP.jl/stable/tutorials/getting_started/tolerances/).

The host records an explicit conversion:

1. Preserve the selected binary64 value as both its exact rational ratio and
   hexadecimal floating-point representation.
2. Project that ratio onto the **exact rational** declared bounds.
3. Retain the original ratio, projected ratio and exact adjustment.
4. Prepare and check a certificate for the projected rational candidate.

For example, binary64 `0.1` is slightly above exact `1/10`; a candidate at that
upper bound is projected to `1/10`, with the change recorded. An interior value
is retained as its exact binary64 ratio. This is not silent decimal rounding,
and it does not claim the raw solver value itself satisfied the exact bound.
The producer's floating objective remains distinct from the checked exact
objective. Certificate preparation cannot manufacture acceptance: its proposed
multipliers and values are checked afterward.

## Independent exact checks

Write the local objective as `f(d)=H*d^2+2*b*d+c`, where:

```text
H = j^2 + lambda > 0
b = j*(y0-target)
c = (y0-target)^2
```

For nonnegative multipliers `mu_lower` and `mu_upper` on `lower-d<=0` and
`d-upper<=0`, completing the square in the Lagrangian gives:

```text
s = 2*b - mu_lower + mu_upper
L = c + mu_lower*lower - mu_upper*upper - s^2/(4*H).
```

Minimizing that unconstrained quadratic gives a valid dual lower bound `L`.
A feasible candidate supplies `U=f(d)`, so `L<=f_opt<=U`. The checker reports:

| Check | Required condition |
| --- | --- |
| Evaluation | Reported prediction, nonlinear output, objective, lower bound and gap equal their exact calculations |
| Feasibility | Exact candidate lies within the declared bounds |
| Dual feasibility | Both multipliers are nonnegative |
| Bounded suboptimality | All preceding conditions and `0<=U-L<=epsilon` |
| Exact optimality | All preceding conditions, zero stationarity residual and both complementary-slackness products zero |

The stationarity equation is `2*(H*d+b)-mu_lower+mu_upper=0`.
Complementarity is `mu_lower*(d-lower)=0` and `mu_upper*(upper-d)=0`.
These exact KKT conditions are sufficient here because `H>0` and the constraints
are affine. The mathematical argument is documented; neither a formal proof
assistant nor SP1 has verified the implementation's correspondence to it.

Acceptance requires the requested bounded gap; exact optimality is an additional
reported property. A feasible near-optimal binary64 candidate may pass the gap
request while failing exact stationarity. The checker never relaxes epsilon to
make it pass. The supplied examples request `epsilon=1/1,000,000`.

## Contracts, persistence and limits

The problem has a code-owned specification digest. Rational inputs use the
[exact-response encoding](EXACT_RESPONSE.md): reduced integer pairs, positive
denominators, input limbs at most one million, and no float-to-rational inference
from input JSON. Limits are `abs(x)<=10`, `abs(target)<=100`,
`1/1,000,000<=lambda<=100`, `-1<=lower<=upper<=1`, and `0<=epsilon<=1`.
Candidate rational limbs are at most `2^255-1`; exceeding the retained arithmetic
budget is an explicit refusal, never truncation. These are software limits.

`ciw.design-adjustment-check.v1` retains the candidate, specification, exact
statement bindings and historical outcomes. `ciw.jump-design-study.v1` additionally
retains raw worker input/stdout/stderr, source and environment snapshots, package
versions, executable digest, conversion, check report and stage timings.
Each actual producer invocation has a separate local UUID; content identities
remain distinct from that invocation and from registered operation/execution/
result/verification identities, which this exploratory caller does not assign.

Process time includes startup and numerical execution; solver time is the
provider's reported solve time. Certificate preparation and exact checking have
separate host timings. Proof-generation and proof-verification times are null.
Runtime hashes identify the declared artifacts; they are not a cryptographic
source-to-binary build attestation.

Reopening validates bounded encoding, content links, runtime declarations,
source-to-candidate conversion and retained outcome consistency. It never starts
Julia, evaluates the model again or reruns the certificate checks. A fully
rewritten and resealed historical report cannot authenticate its own claims.
Explicit checking or solving is required for fresh evidence. No physical
measurement, state admission, controller stability or actuation authority follows.

The depot's actual package source and native artifact bytes are not fully
inventoried or cryptographically attested by this study. Manifest identities and
provider-reported versions do not close that deployment qualification gate.

## Remaining integration gates

The registered [SCR/SP1 path](JULIA_SP1.md) needs a separate design-checker guest,
reviewed exact arithmetic, reproducible registration/build and genuine prove/
verify tests with wrong-input, configuration, candidate and corrupted-proof
refusals. Do not reuse the heat guest or its verification records. A proof must
bind this objective, constraints, arithmetic, epsilon and candidate in its actual
committed input; a source-file hash alone is insufficient.

Shared workbench registration, installed-wheel packaging of this worker,
cross-platform qualification, a JuliaControl time-domain workload, formal
soundness, physical validation and educational effectiveness remain separate
increments. This study exercises one design decision without widening those
claims or changing existing provider pins.

```text
python -m pytest -q tests/test_design_adjustment.py tests/test_jump_design.py tests/test_exact_response.py tests/test_linear_response.py
python -m compileall -q src
git diff --check
```

The native gate requires real Julia and an instantiated depot; it fails instead
of skipping when either is absent. It checks four fixtures and an explicit
repeat against an independent reference, then reopens all five without invoking
Julia or the exact checker:

```text
python scripts/check_design_adjustment_native.py --julia /path/to/julia --depot /path/to/depot --output-dir results/design-native-gate
```
