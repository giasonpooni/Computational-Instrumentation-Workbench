# Lyapunov runtime experiments (T101–T114)

This section tests the float64 decision procedure of the Parameterized
Lyapunov Stability Runtime (PLSR 0.1.0rc1, commit
`19ea6967060166ba09db6cd4563bd87bd6b3d196`, pinned by
`src/ciw/plsr-runtime.json`) against references that do not share its code:
exact dyadic-rational arithmetic on the declared binary64 inputs, a CIW
re-derivation of the documented formulas, NumPy eigenvalues and, when
installed, SciPy's Bartels–Stewart Lyapunov solvers. T112–T114 are research
specifications with synthetic illustrations.

Every verdict here concerns declared binary64 matrices and one declared
sample. Exact arithmetic decides what those numbers imply; it says nothing
about a physical plant. No task establishes machine safety, actuator
authority, calibration, sensor performance or production acceptance; each
such claim is retained as a `not_established` finding.

## Running

PLSR needs Python 3.12 or newer; the lab runs on Python 3.11 or newer. The
runtime is therefore executed in a subprocess of an interpreter bound as the
provider role `plsr-python`:

```sh
ciw lab run T101 T102 T103 T104 T105 T106 T107 T108 T109 T110 T111 T112 T113 T114 \
    --output-dir results/lab-lyapunov --provider plsr-python=/path/to/python3.12
CIW_LAB_PLSR_PYTHON=/path/to/python3.12 python -m pytest -q tests/test_lab_lyapunov.py
```

The bootstrap (`src/ciw/lab/lyapunov_provider.py`) runs with `python -I`
and, before any evaluation, applies the same identity check as
`ciw.plsr_engine`: the installed distribution version and `lyapunov.__version__`
must equal the pin, and the SHA-256 of every packaged source file (CRLF
normalised, `__pycache__` ignored) must equal the manifest. Any mismatch is a
refusal (`PLSR_SOURCE_MISMATCH`, `PLSR_VERSION_MISMATCH`,
`PLSR_PYTHON_UNSUPPORTED`, `PLSR_UNAVAILABLE`). The report records the
repository, commit, package version, source digest and the interpreter's
Python and NumPy versions under `provider_runtime_identity.provider`, and every
provider-derived finding cites them in `basis.provider`. Source hashes detect
installation drift; they do not authenticate an untrusted interpreter.

Without a usable provider the provider-dependent tasks (T101–T111, T113) are
reported `partial`: only their provider-free findings (exact references,
re-derivations, host-side adapter checks) are retained, and the refusal reason
is the first unresolved assumption. T112 and T114 do not need the provider.
Tests of the provider-free parts always run; provider tests are skipped unless
`CIW_LAB_PLSR_PYTHON` names an interpreter or the test interpreter itself hosts
the runtime.

The whole section runs in about 7 s and its tests in about 12 s on one core.

## References and how independence is claimed

| Reference | Module | Used for |
| --- | --- | --- |
| Exact dyadic rationals (`fractions.Fraction`, integer Bareiss elimination) | `lyapunov_reference` | class of the declared decrease form (negative definite, negative semidefinite, has a positive eigenvalue); exact position of its largest eigenvalue relative to multiples of the resolution; exact positive definiteness of P; exact V against a level |
| CIW re-derivation of the documented formulas | `lyapunov_reference` | decrease matrix, `decrease_resolution`, the runtime-status-v1 decision order |
| NumPy `eigvals` | numpy | Hurwitz/Schur classification (T110, T111) |
| SciPy `solve_continuous_lyapunov` / `solve_discrete_lyapunov`, `expm` | scipy (optional) | Lyapunov solutions and matrix exponentials; the CIW Kronecker solve and a NumPy eigendecomposition stand in when SciPy is absent |

A PLSR result compared with CIW code is an `independent_check` (different
implementation origins). Agreement with the CIW re-derivation checks the
implementation against its own documentation, not whether the documented
specification is sufficient; the exact-arithmetic references test the latter.

## Tasks

Numbers below come from one run on Linux (NumPy 2.4.3, OpenBLAS); the retained
reports are authoritative.

**T101 — resolution floor over matrix scales.** Family
`A = 2^k [[-ε, 1], [-1, -ε]]`, `P = I`: the decrease form is `-2ε 2^k I`
exactly and the resolution is `2^k (8γ₅ + 16uε)`, so certification requires
`ε > ε* = 4γ₅/(1 − 8u) = 2.2204e-15` at every scale (`γ₅ = 5u/(1 − 5u)`,
`u = 2^-53`). PLSR's resolution equals the re-derived bound in all 288
evaluated cases, and codes match the analytic threshold in all 112
normal-range cases. Below the normal range the bound underflows. Counterexample:
`A = [[-2, 5], [0, -3]]·2^-1074`, `P = I` has the exact decrease form
`[[-4, 5], [5, -6]]·2^-1074` (determinant −1 in units of 2^-2148, indefinite),
but its binary64 evaluation rounds the off-diagonal to 4 (negative definite),
the resolution is 0 and PLSR returns `CERTIFIED_WITH_MARGIN`.
`NUMERICAL_OVERFLOW` first appears exactly where the form overflows.

**T102 — power-of-two homogeneous scaling.** Scaling `(A, P, x)` by
`(2^a, 2^b, 2^c)` multiplies M and the resolution by `2^(a+b)` exactly, and x
is rescaled internally. In 550 scalings inside LAPACK's unscaled window
(`max|M|` in `[2^-485, 2^485]`, where `dsyevd` does not rescale) no code and no
margin ratio changed, bitwise. Outside the window `dsyevd` rescales by a
non-power-of-two factor: all margin ratios changed and 35 of 250 codes on
razor-edge cases flipped (between certified or not-definite and inconclusive;
none against the exact class). The subnormal witness flips from
`DECREASE_NOT_DEFINITE` to `CERTIFIED_WITH_MARGIN`.

**T103 — overflow and underflow.** States from `2^-1074` to `1.797e308`
leave the code unchanged; non-finite states are input errors. The level gate
forms `s²` for the power-of-two state scale `s`: when `s²` underflows it
answers "not exceeded" and when it overflows it answers "exceeded", whatever
`P` is. In a 68-case exact scan (`P = 2^p I`, `x = 2^e e₁`, level
`2^(p+2e∓1)`) PLSR missed 5 exceedances (certifying states with `V > level`)
and reported 9 spurious ones, exactly as the documented rule predicts.
`θ = 1e308` with `A₁ = 2I` makes `A(θ)` infinite and raises
`ValueError: A must be finite` instead of returning a status. The subnormal
witness is a false certificate.

**T104 — semidefinite and skew-symmetric edges.** Skew A with `P = I` gives
an exactly zero form and always `NUMERICAL_INCONCLUSIVE`; skew A with SPD P is
never certified; Jordan blocks with `P = I` switch exactly at `λ = 1/2`
(inconclusive at the singular point); the solver refuses `Q = diag(1, 0)` and
the constructor refuses a singular P. All 28 codes are consistent with the
exact class. Counterexample: `quadratic()` accepts some binary64 P whose exact
determinant is negative (its eigenvalue-sign test sees a rounded positive
eigenvalue); the verdict then returns `CERTIFICATE_NOT_POSITIVE` along the weak
direction or `NUMERICAL_INCONCLUSIVE`, never a certificate.

**T105 — parameter boxes across unit scales.** A mass-spring-damper
(`m = 2 kg`, `c = 3 N s/m`, stiffness in `[8, 12] N/m`, exact common
`P = [[6, 0.75], [0.75, 1]]`) in five unit systems: interior and bound samples
get identical codes and `check_vertices` passes everywhere. Counterexamples:
(1) converting bound and sample by the same multiplication can collapse a
just-outside sample onto the bound, which is then admitted; (2) the
mathematically equal conversions `k·0.001` and `k/1000` differ in binary64, so
a sample on the bound can be refused; (3) margin ratios are not invariant under
non-uniform unit changes (spread 7.6e6 across the five systems), so a
lightly damped plant is certified in SI units and inconclusive in metre and
millisecond units although its exact decrease form is negative definite in all.

**T106 — every runtime status.** Seven one-parameter paths plus the
indefinite-P witnesses reach all nine runtime-status-v1 codes, and every step
equals the CIW re-derivation of the documented order. The runtime refuses all
five host-owned codes in `require_status` and in `Verdict`. The
`CERTIFICATE_NOT_POSITIVE` witness depends on eigenvalue rounding.

**T107 — inconclusive band.** 102 near-boundary cases with exact bins of
`max eig(M)`: no certifying code on a form that is not exactly negative
definite; beyond two resolutions the sign is always resolved; with a declared
margin of three resolutions no band case is `CERTIFIED_WITH_MARGIN`
(`MARGIN_LOW` or `NUMERICAL_INCONCLUSIVE` instead); without a declared margin
14 of 32 exactly negative definite band cases were refused.

**T108 — required-margin monotonicity.** 280 verdicts on sorted margin grids:
no failing verdict regained, `meets_required_margin` never regained,
non-certifying codes and `inequality_certified` independent of the margin, and
the switch to `MARGIN_LOW` exactly at `required_margin = margin`. Negative and
non-finite margins are refused. The proof is the decision-order argument; the
search only fails to find a counterexample.

**T109 — adversarial eigenvalues.** Non-normal `[[-1, K], [0, -2]]` up to
`K = 1e8`, exactly defective `A = T J T⁻¹` (integer unimodular T, exact spectrum
`{-λ}`), clustered spectra. Every certifying verdict (25) uses an exactly valid
certificate; PLSR and SciPy Lyapunov solutions agree to 1.8e-15 on
well-conditioned cases; certified transients satisfy
`‖exp(At)‖ ≤ sqrt(cond P)`. Counterexample: NumPy's floating-point spectral
abscissa is nonnegative for 5 of 7 exactly Hurwitz Jordan matrices; PLSR's
solver refuses those ill-conditioned cases instead of returning a certificate.

**T110 — continuous versus discrete time.** For 40 matrices with every
eigenvalue in one stability quadrant, PLSR certifies each with its own
Lyapunov P exactly in the convention where NumPy finds it stable, and the two
conventions disagree for exactly the 20 matrices in the continuous-only and
discrete-only quadrants. A discrete plant refuses a `theta_dot`.

**T111 — scalar and matrix routes.** PLSR and SciPy Lyapunov solutions agree
to 1.7e-14; PLSR certification agrees with NumPy eigenvalues on all 50
margin-separated plants. Counterexample: for the decrease form
`diag(-1, 1e-6)` all 64 sampled states show a negative scalar decrease (exactly),
yet the form is indefinite; PLSR reports `DECREASE_NOT_DEFINITE` at every
sample. For `n = 1` the verdict follows the sign of `a`.

**T112 — disturbance-aware (ISS) research branch.** Retained specification
(`iss-branch-spec.md`): definition, ISS-Lyapunov conditions, the data a host
must supply, and what PLSR must not claim (no ISS gain, ultimate bound,
robustness or new status code). For `x' = Ax + Bw`, `|w| ≤ w̄`,
`W = sqrt(V)` obeys `W' ≤ -(c/2)W + β` with `c = min eig(P⁻¹Q)` and
`β = ‖P^{1/2}B‖w̄`, so `sqrt(V) ≤ max(sqrt(V(0)), 2β/c)`. Exact ZOH
simulations of four bounded disturbance classes stay at 12–25 % of the bound;
for `x' = -x + w` the bound `w̄` is attained to 1e-13.

**T113 — filtered residuals outside the kernel.** A host-side adapter runs
an EKF on (position, velocity, θ), gates on data validity, staleness,
certificate validity and mean NIS, and emits either a host-owned status or a
`plsr-sample-v1` sample with exactly `sample_schema, x, theta, theta_dot`.
Sensor id, units, calibration reference and timing stay in the envelope; a
structural search of the serialised kernel payloads finds none of them (a
planted leak is detected). Forwarded samples get the predicted codes
(certified for in-box estimates, `OUTSIDE_PARAMETER_BOX` for θ̂ = 0.90), and
the kernel refuses the host-owned codes the adapter emits.

**T114 — non-production servo-axis pilot.** Retained specification
(`servo-pilot-spec.json`/`.md`): plant model, 1 ms sampling, placeholder
inertia interval ±30 % (to be identified), Lyapunov check scope, monitoring,
abort criteria (stop requests to an independent safety function) and no
actuator authority. The unit-balanced `Q = diag(ω², 1)` nominal P is exactly
valid at all 9 grid inertias; with `Q = I` it fails at 4 — the grid is evidence,
not a box certificate, because the sampled-data closed loop is not affine in J.

## Proposed upstream changes (PLSR)

1. Add an absolute subnormal term to `decrease_resolution` (for example
   `n² · 2^-1074` per formed product) or refuse forms with subnormal entries
   (T101, T102, T103 witness).
2. Decide the level gate by exponents (`log2 V = 2e + log2 scaled_value`)
   instead of forming `s²` (T103).
3. Report overflow while forming `A(θ)` as `NUMERICAL_OVERFLOW` (T103).
4. Check positive definiteness of `P` exactly (LDLᵀ in rationals) or with a
   resolution-aware eigenvalue floor (T104).
5. Pre-scale `M` by a power of two into LAPACK's unscaled window before
   `eigvalsh` so that verdicts are exactly scale-invariant (T102), and document
   that margin ratios depend on non-uniform unit choices (T105).

## Open research questions

- A resolution-aware float64 evaluation of the ISS inequality (T112).
- An affine over-approximation of the sampled-data servo loop that fits PLSR's
  box semantics, and bench identification of J, friction and delay
  (hardware-gated; T114).
- Binding the T113 adapter to acquired encoder data (hardware-gated).
