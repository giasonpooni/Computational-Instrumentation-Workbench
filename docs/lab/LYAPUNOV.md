# Lyapunov runtime experiments (T101–T114)

This section tests the float64 decision procedure of the Parameterized
Lyapunov Stability Runtime (PLSR 0.1.0rc1, commit
`19ea6967060166ba09db6cd4563bd87bd6b3d196`, pinned by
`src/ciw/plsr-runtime.json`) against references that do not share its code:
exact dyadic-rational arithmetic on the declared binary64 inputs, NumPy
eigenvalues and, when installed, SciPy's Bartels–Stewart Lyapunov solvers
(independent checks), and a CIW transcription of the documented formulas and
decision order (same-specification checks, never counted as independent).
T112–T114 are research specifications with synthetic illustrations.

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
is the first unresolved assumption. The tasks therefore register no hard
requirement (`requires=()`): a `provider:plsr-python` requirement would make
them `blocked`, and a blocked report may carry only `not_established`
findings, so the provider-free evidence would be lost. T112 and T114 do not
need the provider. Tests of the provider-free parts always run; provider tests
are skipped unless `CIW_LAB_PLSR_PYTHON` names an interpreter or the test
interpreter itself hosts the runtime, and a `CIW_LAB_PLSR_PYTHON` that names a
missing interpreter fails them instead of skipping them.

The whole section runs in about 6 s and its tests in about 8 s on one core.
SciPy and mpmath are optional: without them T109 and T111 compare PLSR with
the CIW Kronecker solve, T112 with a NumPy eigendecomposition (its augmented
matrix is diagonalisable), and T114's exponential, whose augmented matrix is
defective, only with its closed form.

## References and how independence is claimed

| Reference | Module | Used for |
| --- | --- | --- |
| Exact dyadic rationals (`fractions.Fraction`, integer Bareiss elimination) | `lyapunov_reference` | class of the declared decrease form (negative definite, negative semidefinite, has a positive eigenvalue); exact position of its largest eigenvalue relative to multiples of the resolution; exact positive definiteness of P; exact V against a level |
| CIW re-derivation of the documented formulas | `lyapunov_reference` | decrease matrix, `decrease_resolution`, the runtime-status-v1 decision order |
| NumPy `eigvals` | numpy | Hurwitz/Schur classification (T110, T111); the PLSR operations exercised here (`verdict`, `solve_lyapunov`, `quadratic`, `check_vertices`, `decrease_resolution`) call only `eigvalsh`; its diagnostic `spectral_abscissa`/`spectral_radius` use `eigvals`, but no task calls them |
| SciPy `solve_continuous_lyapunov` / `solve_discrete_lyapunov`, `expm` | scipy (optional) | Lyapunov solutions and matrix exponentials; the CIW Kronecker solve, mpmath's `expm` or a well-conditioned NumPy eigendecomposition stand in when SciPy is absent |
| Closed forms | `lyapunov_research` | ZOH exponentials of the T112 oscillator and the T114 servo axis |

Only references that compute by a separate method are recorded as an
`independent_check` against PLSR: exact rational arithmetic, NumPy
eigenvalues, SciPy (or the CIW Kronecker solve) for Lyapunov solutions and
independent matrix exponentials. The CIW re-derivation of the documented
formulas and decision order (`decrease_resolution`, the level gate,
`documented_code`) is transcribed from the runtime's own documentation and
source, so agreement with it is recorded as an ordinary check
(`numerically_verified`): it shows that the runtime implements its
documentation, not that the documented procedure is sufficient. The
exact-arithmetic references test the latter.

## Tasks

Numbers below come from one run on Linux (NumPy 2.4.3, OpenBLAS); the retained
reports are authoritative.

**T101 — resolution floor over matrix scales.** Family
`A = 2^k [[-ε, 1], [-1, -ε]]`, `P = I`: the decrease form is `-2ε 2^k I`
exactly and the resolution is `2^k (8γ₅ + 16uε)`, so certification requires
`ε > ε* = 4γ₅/(1 − 8u) = 2.2204e-15` at every scale (`γ₅ = 5u/(1 − 5u)`,
`u = 2^-53`). This family has a diagonal decrease form, on which `eigvalsh`
is exact; T102 tests general near-threshold forms. PLSR's resolution equals
the re-derived bound in all 288 evaluated cases (a same-specification check),
and in all 112 normal-range cases the code is `CERTIFIED_WITH_MARGIN` exactly
when `ε > ε*` (compared with the analytic threshold itself, and separately
with PLSR's own unit-scale code). Below the normal range the bound underflows.
Counterexample: `A = [[-2, 5], [0, -3]]·2^-1074`, `P = I` has the exact
decrease form `[[-4, 5], [5, -6]]·2^-1074` (determinant −1 in units of
2^-2148, indefinite), but its binary64 evaluation rounds the off-diagonal to 4
(negative definite), the resolution is 0 and on this platform PLSR returns
`CERTIFIED_WITH_MARGIN`. The exact class and the zero resolution are checked;
PLSR's code is recorded, and the counterexample is attached only when the
certificate is observed, because it depends on LAPACK's subnormal handling.
`NUMERICAL_OVERFLOW` first appears where the CIW-recomputed form overflows
(k = 512 for all four seeds), and never earlier.

**T102 — power-of-two homogeneous scaling.** Scaling `(A, P, x)` by
`(2^a, 2^b, 2^c)` multiplies M and the resolution by `2^(a+b)` exactly, and x
is rescaled internally. In 550 scalings inside LAPACK's unscaled window
(`max|M|` in `[2^-485, 2^485]`, where `dsyevd` does not rescale) no code and no
margin ratio changed, bitwise. Outside the window `dsyevd` rescales by a
non-power-of-two factor: all margin ratios changed and 35 of 250 codes on
razor-edge cases flipped (between certified or not-definite and inconclusive;
none against the exact class). Only the soundness count is a checked value;
the flips are recorded as a counterexample when observed, since their number
depends on the LAPACK/BLAS build. In discrete time (P, x) scaling changed no
code in 36 evaluations, including six razor-edge cases about one resolution
from the threshold. The subnormal witness flips from `DECREASE_NOT_DEFINITE`
to `CERTIFIED_WITH_MARGIN` on this platform.

**T103 — overflow and underflow.** States from `2^-1074` to `1.797e308`
leave the code unchanged; non-finite states are input errors. Components more
than about 2^1075 below the largest one vanish in the scaled state (for
`(1.797e308, 2^-1074)` the second component becomes 0), which a code decided
relative to `|x|²` does not see; this is not probed further. The level gate
forms `s²` for the power-of-two state scale `s`: when `s²` underflows it
answers "not exceeded" and when it overflows it answers "exceeded", whatever
`P` is. In a 68-case exact scan (`P = 2^p I`, `x = 2^e e₁`, level
`2^(p+2e∓1)`) PLSR missed 5 exceedances (certifying states with `V > level`)
and reported 9 spurious ones against the exact exponent comparison, in the
same cases as the CIW transcription of the documented rule. `θ = 1e308` with
`A₁ = 2I` makes `A(θ)` infinite and raises `ValueError: A must be finite`
instead of returning a status. The subnormal witness is a false certificate
on this platform.

**T104 — semidefinite and skew-symmetric edges.** Skew A with `P = I` gives
an exactly zero form and always `NUMERICAL_INCONCLUSIVE`; skew A with SPD P is
never certified; Jordan blocks with `P = I` switch exactly at `λ = 1/2`
(inconclusive at the singular point); the solver refuses `Q = diag(1, 0)` and
the constructor refuses a singular P. All 28 codes are consistent with the
exact class. Counterexample: `quadratic()` accepts some binary64 P whose exact
determinant is negative (its eigenvalue-sign test sees a rounded positive
eigenvalue; 8 found in 8207 seeded draws); the verdict then returns
`CERTIFICATE_NOT_POSITIVE` along the weak direction or
`NUMERICAL_INCONCLUSIVE`, never a certificate. Which candidates are accepted
depends on `eigvalsh` rounding, so the counterexample is attached when
observed and only the certification count (0) is checked.

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

**T106 — every runtime status.** Seven one-parameter paths far from every
rounding threshold reach the eight codes that do not depend on rounding, and
all 37 steps equal the CIW transcription of the documented order (a
same-specification check). `CERTIFICATE_NOT_POSITIVE` is reachable only
through rounding (a P accepted by `quadratic()` evaluating `V < 0`); the eight
exactly indefinite P witnesses reached it 6 times here and were never
certified, and only the non-certification is checked. The runtime refuses all
five host-owned codes in `require_status` and in `Verdict`, and its published
constants match the documented values.

**T107 — inconclusive band.** 102 near-boundary cases with exact bins of
`max eig(M)`: no certifying code on a form that is not exactly negative
definite; beyond two resolutions the sign is always resolved; with a declared
margin of three resolutions no band case is `CERTIFIED_WITH_MARGIN`
(`MARGIN_LOW` or `NUMERICAL_INCONCLUSIVE` instead). Counterexample to the
specification's stronger statement that near-boundary spectra yield
`NUMERICAL_INCONCLUSIVE` or `MARGIN_LOW` rather than `CERTIFIED_WITH_MARGIN`:
at `required_margin = 0`, 18 of the 66 cases within two resolutions of zero
were certified (all 16 in `[-2, -1)` resolutions and 2 in `[-1, 0)`), every
one of them exactly negative definite, so the certifications are sound.
Of the 32 exactly negative definite band cases, 14 were left inconclusive.

**T108 — required-margin monotonicity.** 280 verdicts on sorted margin grids:
no failing verdict regained, `meets_required_margin` never regained,
non-certifying codes and `inequality_certified` independent of the margin, and
the switch to `MARGIN_LOW` exactly at `required_margin = margin`. Negative and
non-finite margins are refused. The proof is the decision-order argument; the
search only fails to find a counterexample.

**T109 — adversarial eigenvalues.** Non-normal `[[-1, K], [0, -2]]` up to
`K = 1e8`, exactly defective `A = T J T⁻¹` (integer unimodular T, exact spectrum
`{-λ}`, with `T T⁻¹ = I` checked exactly), clustered spectra. Every
certifying verdict (30) uses an exactly valid certificate; PLSR and SciPy
Lyapunov solutions agree to within `10 n² u cond(P)` (normalised difference at
most 1.7, relative difference at most 1.8e-15) on the 14 solved cases;
certified transients satisfy `‖exp(At)‖ ≤ sqrt(cond P)` (ratio at most
0.618); with `P = I` the non-normal plants are certified exactly when
`K < 2√2` (K = 2.82 certified, 2.83 not). On the seven exactly Hurwitz Jordan
plants `solve_lyapunov` returned P once (exactly valid) and refused six, each
at a documented gate. Counterexamples: NumPy's floating-point spectral
abscissa is nonnegative for 5 of the 7; and the solver's refusals are
conservative, not only avoidance of invalid certificates: for `n = 4`,
`λ = 2^-6` it refuses at the residual gate, yet SciPy's P (condition 5e11) is
exactly valid and PLSR's own verdict certifies it.

**T110 — continuous versus discrete time.** For 40 matrices with every
eigenvalue in one stability quadrant, PLSR certifies each with its own
Lyapunov P exactly in the convention where NumPy finds it stable, its solver
refuses the other convention at a documented gate, and, matrix by matrix, the
two conventions give different outcomes exactly for the 20 matrices stable in
only one of them. No P solved for one convention certifies a matrix in the
other convention where NumPy finds it unstable (40 cross-applied verdicts). A
discrete plant refuses a `theta_dot`.

**T111 — scalar and matrix routes.** PLSR Lyapunov solutions agree with
SciPy's `solve_continuous_lyapunov` to 1.7e-14 and with
`solve_discrete_lyapunov` to 4.5e-16 (checked per convention). On the 50
margin-separated plants `solve_lyapunov` returns P exactly for the 30 that
NumPy's eigenvalues call stable and raises `ValueError` for the other 20; the
verdict certifies every stable plant with its own P and no unstable plant with
`P = I`. Counterexample: for the decrease form
`diag(-1, 1e-6)` all 64 sampled states show a negative scalar decrease (exactly),
yet the form is indefinite; PLSR reports `DECREASE_NOT_DEFINITE` at every
sample. For `n = 1` the verdict follows the sign of `a`.

**T112 — disturbance-aware (ISS) research branch.** Retained specification
(`iss-branch-spec.md`): definition, ISS-Lyapunov conditions, the data a host
must supply, and what PLSR must not claim (no ISS gain, ultimate bound,
robustness or new status code). For `x' = Ax + Bw`, `|w| ≤ w̄`,
`W = sqrt(V)` obeys `W' ≤ -(c/2)W + β` with `c = min eig(P⁻¹Q)` and
`β = ‖P^{1/2}B‖w̄`, so `sqrt(V) ≤ max(sqrt(V(0)), 2β/c)` = 1.61832 here.
The bound is conservative by the Cauchy–Schwarz step: the sharp supremum of
`sqrt(V)` reachable from `x(0) = 0` under any `|w| ≤ w̄` (the support function
of the reachable set, by quadrature on the exact impulse response, converged
to 2e-6 under step halving) is 0.41175, 0.254 of the bound. Exact ZOH
simulations of four bounded disturbance classes stay below the sharp
supremum, and worst-case switching reaches 0.994 of it; the simulations cover
only those four classes, the sharp supremum covers every bounded
disturbance. For `x' = -x + w` the bound `w̄` is approached, not attained:
`sup |x|` over 30 s is `(1 − e^-30)` of it.

**T113 — filtered residuals outside the kernel.** A host-side adapter runs
an EKF on (position, velocity, θ), gates on data validity, staleness,
certificate validity and mean NIS, and emits either a host-owned status or
`plsr-sample-v1` samples with exactly `sample_schema, x, theta, theta_dot`.
Because the point estimate θ̂ alone could be accepted near a box bound while
the true parameter lies outside, the adapter forwards θ̂ and both ends of
`θ̂ ± 3 se`, and the host accepts a window only when the kernel certifies all
three. Sensor id, units, calibration reference and timing stay in the
envelope; a structural search of the serialised kernel payloads finds none of
them (a planted leak is detected). Forwarded samples get the codes the
documented order predicts: both nominal windows are accepted, the near-bound
window (θ = 0.48) is rejected because its upper interval end is
`OUTSIDE_PARAMETER_BOX`, and θ̂ = 0.90 is outside the box. The kernel refuses
the host-owned codes the adapter emits. Whether the EKF standard error covers
the true parameter of a real axis is recorded as `not_established`.

**T114 — non-production servo-axis pilot.** Retained specification
(`servo-pilot-spec.json`/`.md`): plant model, 1 ms sampling, placeholder
inertia interval ±30 % (to be identified), Lyapunov check scope, monitoring,
abort criteria (stop requests to an independent safety function) and no
actuator authority. The unit-balanced `Q = diag(ω², 1)` nominal P is exactly
valid at all 9 grid inertias; with `Q = I` it fails at 4 — the grid is evidence,
not a box certificate, because the sampled-data closed loop is not affine in J.
For a fixed declared model the sign of the decrease form does not depend on
the state, so without a level set the monitor returns `CERTIFIED_WITH_MARGIN`
for every state and carries no information about the axis. The specification
therefore declares a level `c` with `{V ≤ c}` inside the operating envelope and
aborts on `OUTSIDE_LEVEL_SET`; a 200-state scan (by the documented decision
order) shows the code follows the exact `V(x) > c` decision and that every
abort code is producible. Codes the configuration cannot produce (for example
`NOT_CERTIFIED`, `DECREASE_NOT_DEFINITE`, `OUTSIDE_PARAMETER_BOX`) are listed
as such and treated as a runtime fault, not as abort triggers.

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
6. Report the residual gate of `solve_lyapunov` separately from
   definiteness, and document that a refusal does not mean that no valid
   quadratic certificate exists (T109).

## Open research questions

- A resolution-aware float64 evaluation of the ISS inequality (T112).
- An affine over-approximation of the sampled-data servo loop that fits PLSR's
  box semantics, and bench identification of J, friction and delay
  (hardware-gated; T114).
- Binding the T113 adapter to acquired encoder data (hardware-gated).
