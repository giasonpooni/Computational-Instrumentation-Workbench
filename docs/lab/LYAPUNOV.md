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

Without a usable provider the provider-dependent tasks (T101–T111, T113,
T114) are reported `partial`: only their provider-free findings (exact
references, re-derivations, host-side adapter checks, T114's specification and
its monitor scan in the CIW transcription) are retained, and the refusal reason
is the first unresolved assumption. The tasks therefore register no hard
requirement (`requires=()`): a `provider:plsr-python` requirement would make
them `blocked`, and a blocked report may carry only `not_established`
findings, so the provider-free evidence would be lost. T112 does not need the
provider. Tests of the provider-free parts always run; provider tests are
skipped unless `CIW_LAB_PLSR_PYTHON` names an interpreter or the test
interpreter itself hosts the runtime, and a `CIW_LAB_PLSR_PYTHON` that names a
missing interpreter fails them instead of skipping them.

The whole section runs in about 7 s and its tests in about 10 s on one core.
SciPy and mpmath are optional: without them T109 and T111 compare PLSR with
the CIW Kronecker solve (recorded as an ordinary check, see below), T112 with a
NumPy eigendecomposition (its augmented matrix is diagonalisable), and T114's
exponential, whose augmented matrix is defective, only with its closed form.

Outcomes that depend on the LAPACK/BLAS build are retained as artifacts, never
as checked finding values, compared finding values or wording: the codes of
subnormal plants (the witness below included), code flips outside LAPACK's
scaling window, which rounding-admitted indefinite P `quadratic()` accepts and
the codes they receive, the sign of NumPy's abscissa for defective matrices,
which T107 straddle cases resolve, and which gate refuses an exactly defective
T109 plant with the residual it quotes. Findings carry only what does not
depend on the build (exact classes, IEEE-deterministic scalar arithmetic,
soundness counts, codes far from every threshold). Where a claim rests on the
exact position of the declared inputs near a threshold (T107), the inputs are
built without BLAS or LAPACK, so every kernel declares the same matrices; the
few compared numbers that carry rounding (T107's witness margin ratio, T109's
witness condition number) have tolerances derived from the realized error or
`u cond(P)` and checked against the spread between the SkylakeX, Haswell and
Sandybridge OpenBLAS kernels (`OPENBLAS_CORETYPE`). Every check is built
before its outcome is known:
`test_checks_are_unconditional_and_observed_values_are_computed` refuses a
check created under a condition on an observed result or with a literal
observed value.

## References and how independence is claimed

| Reference | Module | Used for |
| --- | --- | --- |
| Exact dyadic rationals (`fractions.Fraction`, integer Bareiss elimination) | `lyapunov_reference` | class of the declared decrease form (negative definite, negative semidefinite, has a positive eigenvalue); exact position of its largest eigenvalue relative to multiples of the resolution; exact positive definiteness of P; exact V against a level; construction of T107's near-boundary matrices from integer draws without BLAS or LAPACK |
| CIW re-derivation of the documented formulas | `lyapunov_reference` | decrease matrix, `decrease_resolution`, the runtime-status-v1 decision order |
| NumPy `eigvals` | numpy | Hurwitz/Schur classification (T110, T111); the PLSR operations exercised here (`verdict`, `solve_lyapunov`, `quadratic`, `check_vertices`, `decrease_resolution`) call only `eigvalsh`; its diagnostic `spectral_abscissa`/`spectral_radius` use `eigvals`, but no task calls them |
| SciPy `solve_continuous_lyapunov` / `solve_discrete_lyapunov`, `expm` | scipy (optional) | Lyapunov solutions and matrix exponentials; mpmath's `expm` or a well-conditioned NumPy eigendecomposition stand in for `expm` when SciPy is absent, and the CIW Kronecker solve for the Lyapunov solvers (then not an independent reference) |
| Closed forms | `lyapunov_research` | ZOH exponentials of the T112 oscillator and the T114 servo axis |

Only references that compute by a separate method are recorded as an
`independent_check` against PLSR: exact rational arithmetic, NumPy
eigenvalues, SciPy's Bartels–Stewart solvers for Lyapunov solutions and
independent matrix exponentials. The CIW Kronecker fallback solves the same
Kronecker system with `numpy.linalg.solve` as PLSR's `solve_lyapunov`, so a
shared algorithm could hide a common-mode error; without SciPy, agreement with
it is an ordinary check (`numerically_verified`). Solver agreement is judged
against the forward-error bound `10 n² u cond(P)` of a backward-stable solve
(reference kind `analytic`), not recorded as exact arithmetic. The CIW re-derivation of the documented
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
Counterexample to "the resolution bounds the rounding error of the decrease
form at every scale": `A = [[-2, 5], [0, -3]]·2^-1074`, `P = I` has the exact
decrease form `[[-4, 5], [5, -6]]·2^-1074` (determinant −1 in units of
2^-2148, indefinite), but PLSR's own formed matrix, returned by the runtime, is
`[[-4, 4], [4, -6]]·2^-1074` (the symmetrising halving rounds 2.5 to 2; negative
definite) while its resolution is 0, below the formation error of one
2^-1074 unit. Both facts are IEEE-deterministic wherever subnormals are kept,
so the counterexample is always attached and checked. What the eigensolver
then makes of the subnormal form, and hence PLSR's code (`CERTIFIED_WITH_MARGIN`
on the reference platform), depends on the LAPACK build and is retained in
`resolution-floor.json` only. `NUMERICAL_OVERFLOW` first appears where the
CIW-recomputed form overflows (k = 512 for all four seeds), and never earlier.

**T102 — power-of-two homogeneous scaling.** Scaling `(A, P, x)` by
`(2^a, 2^b, 2^c)` multiplies M and the resolution by `2^(a+b)` exactly, and x
is rescaled internally. In 550 scalings inside LAPACK's unscaled window
(`max|M|` in `[2^-485, 2^485]`, where `dsyevd` does not rescale) no code and no
margin ratio changed, bitwise. Outside the window `dsyevd` rescales by a
non-power-of-two factor: on the reference platform all margin ratios changed
and 35 of 250 codes on razor-edge cases flipped (between certified or
not-definite and inconclusive; none against the exact class). Only the
soundness count is a checked finding value; the flips, whose number depends on
the LAPACK/BLAS build, are retained in `scaling-invariance.json`. In discrete
time (P, x) scaling changed no code in 36 evaluations, including six razor-edge
cases about one resolution from the threshold. Scaled by 2^-1074 the witness's
resolution underflows to 0 (checked) while its unit-scale code is
`DECREASE_NOT_DEFINITE`; the scaled code (`CERTIFIED_WITH_MARGIN` on the
reference platform) is retained in the artifact.

**T103 — overflow and underflow.** States from `2^-1074` to `1.797e308`
leave the code unchanged; non-finite states are input errors. Components more
than about 2^1075 below the largest one vanish in the scaled state (for
`(1.797e308, 2^-1074)` the second component becomes 0), which a code decided
relative to `|x|²` does not see; this is not probed further. The reported
unscaled `V = s² · scaled_value` and `xᵀMx` are compared with the exact
rationals at those ten states and at 27 single-component states `m·2^e` around
the edges where `s²` underflows or overflows: in all 20 states whose exact `V`
or `xᵀMx` is not representable `value_out_of_range` is set, and unflagged
values are within one rounding of the exact ones. The flag is conservative,
not exact: for `x = (1.5·2^-538, 0)` and `(1.9·2^-538, 0)` the exact `V` and
`xᵀMx` round to nonzero subnormals, but `s² = 2^-1076` underflows first, so
PLSR reports `V = 0` with the flag set (a counterexample to "the flag is set
exactly when the value is not representable"). The level gate
forms `s²` for the power-of-two state scale `s`: when `s²` underflows it
answers "not exceeded" and when it overflows it answers "exceeded", whatever
`P` is. In a 68-case exact scan (`P = 2^p I`, `x = 2^e e₁`, level
`2^(p+2e∓1)`) PLSR missed 5 exceedances (certifying states with `V > level`)
and reported 9 spurious ones against the exact exponent comparison, in the
same cases as the CIW transcription of the documented rule. `θ = 1e308` with
`A₁ = 2I` makes `A(θ)` infinite and raises `ValueError: A must be finite`
instead of returning a status. T103 re-evaluates the subnormal witness only
for `near-limits.json`: its resolution is again 0 below its formation error,
and T103's numerical result states this. The finding itself is retained
once, by T101, so aggregates count it once.

**T104 — semidefinite and skew-symmetric edges.** Skew A with `P = I` gives
an exactly zero form and always `NUMERICAL_INCONCLUSIVE`; skew A with SPD P is
never certified; Jordan blocks with `P = I` switch exactly at `λ = 1/2`
(inconclusive at the singular point); the solver refuses `Q = diag(1, 0)` and
the constructor refuses a singular P. All 28 codes are consistent with the
exact class. `quadratic()`'s eigenvalue-sign test is not an exact
definiteness test: on the reference platform it accepted all 8 exactly
indefinite candidates found in 8207 seeded draws, whose verdicts were
`CERTIFICATE_NOT_POSITIVE` along the weak direction or `NUMERICAL_INCONCLUSIVE`,
never a certificate. Which candidates are accepted depends on `eigvalsh`
rounding, so they and their codes are retained in `edge-cases.json`, and only
the certification count (0) is a checked finding value.

**T105 — parameter boxes across unit scales.** A mass-spring-damper
(`m = 2 kg`, `c = 3 N s/m`, stiffness in `[8, 12] N/m`, exact common
`P = [[6, 0.75], [0.75, 1]]`) in five unit systems: interior and bound samples
get identical codes and `check_vertices` passes everywhere; the bounds keep
their SI box decision under both conversion formulas (`k·c` and `k/(1/c)`),
and their binary64 neighbours under multiplication. Counterexamples:
(1) converting bound and sample by the same multiplication can collapse a
just-outside sample onto the bound, which is then admitted; (2) the
mathematically equal conversions `k·0.001` and `k/1000` differ in binary64, so
a sample on the bound can be refused; (2a) on the declared box itself,
`nextafter(8, 0)` converted as `k/(1/c)` lands on the bound `8·c` in
micrometre units and is admitted; (3) margin ratios are not invariant under
non-uniform unit changes (spread 7.6e6 across the five systems), so a
lightly damped plant is certified in SI units and inconclusive in metre and
millisecond units although its exact decrease form is negative definite in all.

**T106 — every runtime status transition.** The conditions of the documented
order form a gate vector (box, overflow, not_positive, level,
scalar_positive, certified, margin_low, not_definite) with three implications
(`scalar_positive ⇒ not_definite ∧ ¬certified` by the Rayleigh quotient,
`certified ⇒ ¬not_definite`, `margin_low ⇒ certified`). A one-parameter path
crosses one threshold at a time: one gate changes, or two whose thresholds
coincide (scalar_positive with not_definite when x is a top eigenvector of M;
certified with margin_low). Enumerating the consistent gate vectors
(`transition_graph`) gives 24 directly connected pairs among the eight
rounding-free codes, 4 excluded pairs (`CERTIFIED_WITH_MARGIN`/`MARGIN_LOW`
with `NOT_CERTIFIED` or `DECREASE_NOT_DEFINITE`: every path between them passes
through `DECREASE_NOT_DEFINITE` or `NUMERICAL_INCONCLUSIVE`), and 8 pairs with
`CERTIFICATE_NOT_POSITIVE`, which a declared certificate reaches only through
rounding. Seven sweeps and seventeen short paths across one crossing (box edge
θ = 1 → nextafter under six in-box codes, a power-of-two matrix scale across
the overflow edge under five codes, level 4 → 3.99 under four codes, a
vanishing margin under a declared margin, a top eigenvalue crossing zero)
realise all 24 allowed transitions in PLSR; every one of the 73 steps equals
the CIW transcription, and each of the 28 code changes happens between steps
whose transcribed gate vectors differ by exactly an allowed crossing
(`transition-coverage.json`, matrix in `transition-coverage.md`). No
rounding-free route into `CERTIFICATE_NOT_POSITIVE` exists: both certificate
kinds refuse `min eig P ≤ 0` when built, and an affine certificate whose
`P(θ) = diag(1, 0)` at the in-box θ = −1 raises `ValueError` instead of
returning the code (a counterexample to RUNTIME-STATUS-v1's description of
`CERTIFICATE_NOT_POSITIVE`). The eight exactly indefinite P witnesses are
never certified (checked); whether they reach `CERTIFICATE_NOT_POSITIVE`
depends on rounding and is retained in the artifact. The runtime refuses all
five host-owned codes in `require_status` and in `Verdict`, and its published
constants match the documented values.

**T107 — inconclusive band.** 102 near-boundary cases with exact bins of
`max eig(M)`, the same matrices on every BLAS kernel: each `A = P⁻¹(N/2 + K)`
is solved in exact rational arithmetic from integer draws (an exactly
orthogonal Cayley factor in `N`, a skew part `K` of size 100, `P = I` or a
rational SPD `P` rounded entrywise) and rounded once, so no BLAS or LAPACK
call reaches the declared inputs, and the resolution used for the bins reads
`max|M|` from the correctly rounded exact form. An earlier family built its
matrices with `qr`, `solve` and matrix products and tuned twelve razor-edge
cases on the computed eigenvalue, so its inputs, exact bins and codes changed
with the OpenBLAS kernel (on a non-FMA kernel no case within one resolution
was certified, on the FMA kernels two were). The 90 window cases have their
exact `max eig(M)` within `res/16` of a target `κ res`
(`κ ∈ {±8, ±3.5, ±2.5, ±1.75, ±1.25, ±0.75, ±0.25, 0}`), checked exactly,
so every window keeps at least `3/16 res` from the thresholds where PLSR's
code changes (`-3 res` under the declared margin, `-res`, `+res`); states are
redrawn until the exact `xᵀMx/|x|²` keeps the same distance from `res`. PLSR's
computed `max eig(M)` erred by at most 0.0096 res on the SkylakeX and Haswell
kernels and 0.0116 res on Sandybridge (checked against `3/16`), so every
window case received the code its exact window predicts (checked
independently): certified below `-res`, `MARGIN_LOW` between `-3 res` and
`-res` under the declared margin, inconclusive within one resolution, not
definite beyond `+res`. No certifying code on a form that is not exactly
negative definite; beyond two resolutions the sign is always resolved; with a
declared margin of three resolutions no band case is `CERTIFIED_WITH_MARGIN`.
Counterexample to a stronger candidate hypothesis formulated for this
experiment (it is not quoted from the runtime's documentation) that
near-boundary spectra yield `NUMERICAL_INCONCLUSIVE` or `MARGIN_LOW` rather
than `CERTIFIED_WITH_MARGIN`: at `required_margin = 0` all 12 window cases in
`[-2, -1)` resolutions were certified, every band certificate exactly sound;
of the 26 exactly negative definite window band cases 14 were left
inconclusive. The 12 straddle cases are six pairs of adjacent float targets
between which the exact `max eig(M)` crosses `-res` (within 0.0082 res of
it); there the exact side does not decide the code, which is
`CERTIFIED_WITH_MARGIN` or `NUMERICAL_INCONCLUSIVE` (checked): 4 of the 6
exactly beyond `-res` were certified and none of the 6 inside, identically on
the three kernels, but which ones resolve is a rounding outcome and is
retained per case in `inconclusive-band.json` and the numerical result, not
compared. The witness's computed margin ratio is the one compared value that
carries rounding: band-certified margin ratios differed by at most 0.0018
between the kernels, and its tolerance is `abs 0.03` resolutions (twice the
largest realized error, rounded up), which admits no change of the counts.

**T108 — required-margin monotonicity.** 280 verdicts on sorted margin grids:
no failing verdict regained, `meets_required_margin` never regained,
non-certifying codes and `inequality_certified` independent of the margin, and
the switch to `MARGIN_LOW` exactly at `required_margin = margin`. Negative and
non-finite margins are refused. The CIW transcription is checked on the same
grids for its codes only (its margin fields would be the defining formulas read
back). The proof is the decision-order argument; the search only fails to find
a counterexample.

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
at a documented gate. NumPy misplaces every exact defective eigenvalue by far
more than 1e3 ε·max|A| (checked); its floating-point spectral abscissa is
nonnegative for 5 of the 7 on the reference platform, a build-dependent count
retained in `adversarial.json`. Counterexample: the solver's refusals are
conservative, not only avoidance of invalid certificates: for `n = 4`,
`λ = 2^-6` it refuses at the residual gate, yet SciPy's P (condition 5e11) is
exactly valid and PLSR's own verdict certifies it. The witness records the
gate, not the refusal message: the residual it quotes (5.9e-4 or 6.3e-4) and,
for other Jordan plants, which gate refuses (definiteness or singularity) are
rounding outcomes that differ between OpenBLAS kernels and stay in
`adversarial.json`. The certificate's condition number differed by 4.2e-5
relative between the SkylakeX, Haswell and Sandybridge kernels, the size
`u cond(P) = 5.5e-5` of a backward-stable solve's effect on the smallest
eigenvalue of P, and is compared within `n² u cond(P) = 8.8e-4`.

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
`solve_discrete_lyapunov` to 4.5e-16 relative, within `10 n² u cond(P)` in each
convention. On the 50 margin-separated plants `solve_lyapunov` returns P
exactly for the 30 that NumPy's eigenvalues call stable and raises
`ValueError` for the other 20; the verdict certifies every stable plant with
its own P and no unstable plant with `P = I`. PLSR's own scalar gate
(`NOT_CERTIFIED` when `xᵀMx > res·|x|²`) is compared per sample with the exact
`xᵀMx` and with PLSR's eigenvalue route (`max eig(M) > res`) at 808 samples:
every route plant with `P = I` at eight states, and each of T107's 102
near-threshold forms at the computed top eigenvector of its decrease matrix
(where the gate meets the resolution) and three more states. Every
`NOT_CERTIFIED` has an exactly positive `xᵀMx` (independent check) and
`max eig(M) > res`; the eigenvalue route never meets an exactly negative
definite form; no certificate is issued where the exact `xᵀMx` is positive. On
the route family the scalar decision equals the exact sign of `xᵀMx` and the
eigenvalue route equals the exact class at every sample; near the threshold
they agree at 96 % and 84 % of the samples (the gate is conservative there),
and the scalar gate agrees with the eigenvalue route at 57 % and 74 % (a
sample sees what the matrix sees only along its positive cone). Counterexample:
for the decrease form
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
envelope; a structural search of the serialised samples the adapter emits for
the kernel finds none of them, and the same search flags a sensor id planted
in that output. Forwarded samples get the codes the
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
aborts on `OUTSIDE_LEVEL_SET`. The containment is checked without reusing the
float inverse that chose `c`: exactly (`max x_i²` over the ellipsoid is
`c (P⁻¹)_ii = c P_jj / det P` in rationals) and on 3600 points of the boundary
`x = sqrt(c) L⁻ᵀu` (`P = L Lᵀ`). The pinned runtime evaluates the monitor
configuration at 200 seeded states: without a level every state is
`CERTIFIED_WITH_MARGIN`, with it the code follows the exact `V(x) > c`
decision (independent check), and every abort code is produced; the CIW
transcription of the documented order gives the same codes and is the
provider-free fallback (then the task is `partial`). Codes the configuration cannot produce (for example
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
7. Return `CERTIFICATE_NOT_POSITIVE` rather than raising `ValueError` when an
   affine `P(θ)` loses definiteness at an in-box θ, as RUNTIME-STATUS-v1
   describes the code (T106).
8. Report `V` and `xᵀMx` from the exact `(mantissa, exponent)` pair, which the
   sample already carries, so that a representable subnormal value is not
   reported as 0 when `s²` underflows (T103).

## Open research questions

Each task's `recommended_next_task` is its own open question
(`lyapunov.NEXT_STEPS`): an upstream proposal above re-tested by the task that
motivated it, a new check (sampled-data reuse of a continuous certificate,
near-marginal route comparison, affine plants with parameter-dependent P), or a
hardware-gated acquisition. It is never the next queue task.

- Cross-platform reproduction (T101, T102; `lyapunov.PLATFORM_QUESTION`):
  run both against the pinned PLSR on Windows x86-64, on macOS arm64 and on
  Linux x86-64 with a LAPACK other than the retained OpenBLAS build. The
  subnormal codes (`resolution-floor.json`) and the outside-window flips and
  scaled-witness code (`scaling-invariance.json`) must match the retained ones
  case for case, and every finding value must match within its regression
  tolerance. These values are artifact-only because builds may decide them
  differently, and no second build has been compared.
- A resolution-aware float64 evaluation of the ISS inequality (T112).
- An affine over-approximation of the sampled-data servo loop that fits PLSR's
  box semantics, and bench identification of J, friction and delay
  (hardware-gated; T114).
- Binding the T113 adapter to acquired encoder data (hardware-gated).

Both hardware-gated steps name their route (`lyapunov.capture_route`). The
bench or encoder log would enter as an operator capture
(`ctx.capture("servo-bench-log")` in T114, `ctx.capture("encoder-log")` in
T113, bound by `ciw lab run TASK --capture ROLE=PATH`), from which the task
could compute computational findings. Their physical, calibration and
sensor-performance findings also need an acquisition record (device,
`raw_sha256` of the captured bytes, time, calibration) and either a probe of
the instrument on the analysing host that succeeds in the task or a
signed-capture trust anchor, because the physical gate never accepts an
unauthenticated capture by itself (`runner.CAPTURE_INSTRUMENTS` has no entry
for either role). The run would be retained with `ciw lab hardware retain`
under `lab/hardware/<run-id>`. Neither task reads a capture and no such probe
exists, so those findings stay `not_established` even when such data exist.
