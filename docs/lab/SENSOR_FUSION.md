# Sensor-fusion experiments (T060–T076)

Implementation: `src/ciw/lab/sensor_fusion.py` (section entry point and the
bench tasks T060–T062), `sensor_fusion_geometry.py` (T063–T064),
`sensor_fusion_filtering.py` (T065–T068), `sensor_fusion_robustness.py`
(T069–T071) and `sensor_fusion_admission.py` (T072–T076). Shared code:
`sensor_fusion_bench.py` (bench generator and filter algebra),
`sensor_fusion_objects.py` (typed observation, candidate and admitted-state
API) and `sensor_fusion_common.py` (evidence helpers). Tests:
`tests/test_lab_sensor_fusion.py`.

`sensor_fusion.py` imports each sibling module separately. If one fails to
import, its partial registrations are rolled back and its tasks are replaced
by placeholders that report the import error as `blocked`; the other tasks
still run. (Listing the siblings in `registry.SECTION_MODULES` would make this
unnecessary.)

```
python -m ciw lab run T060 T061 T062 T063 T064 T065 T066 T067 T068 T069 T070 T071 T072 T073 T074 T075 T076 --output-dir results/lab-sensor-fusion
python -m ciw lab report T068 --retained results/lab-sensor-fusion
python -m pytest -q tests/test_lab_sensor_fusion.py
```

Everything here is a synthetic, seeded, linear-Gaussian experiment. It runs
in about 8 s and its tests in about 13 s on one core, with NumPy only (SciPy is
used only in one optional test that cross-checks the chi-square quantiles and
the noncentral chi-square CDF). Retained artifacts are byte-identical whether
BLAS runs on one thread or many: large Gram products use `einsum` loops and
the dense batch solve uses its own elimination, not multithreaded BLAS/LAPACK.

Every computational finding declares a per-finding `uncertainty`
(`monte_carlo_95ci` from a run-level or binomial standard error, `roundoff`
for identities, `truncation_bound` for RK4 and bisection, `reference_error`
for quadrature, closed-form approximations and exact results). Every task
registers its own case of the parametrized section test
(`test_section_reports_labels_and_states[T0xx]`), so a failure marks only
that task `partial` in a JUnit-linked run.

## What this section does not establish

The bench checks what a Kalman-type fusion pipeline must satisfy before its
output could even be considered. It does not measure a real camera, encoder,
IMU or tracker. It does not validate a calibration, a clock or an extrinsic
transform, and it does not set a safe threshold or grant authority. Every task
records the real-world conclusion it invites as a finding in a physical or
authority domain: `sensor_performance`, `calibration`, `physical`,
`machine_safety`, `actuator_authority` or `production_acceptance`. The
evidence validator labels those findings `not_established`. Every
computational finding is `numerically_verified`. None of them is
`independently_verified`, because every reference used here is CIW code or a
closed form evaluated by CIW code.

## The bench

**State and motion.** The state is planar, x = (px, py, vx, vy), with
constant velocity driven by continuous white-noise acceleration of spectral
density q. The motion model uses the exact discretization:

```
F(dt) = [[I, dt I], [0, I]],   Q(dt) = q [[dt^3/3 I, dt^2/2 I], [dt^2/2 I, dt I]]
```

This discretization composes exactly: F(a)F(b) = F(a+b) and
Q(a+b) = F(b) Q(a) F(b)^T + Q(b). That identity is what makes the gap
predictions in T069 and T073 exact. T061 checks Q against the continuous-time
integral of F(u) G q G^T F(u)^T and the semigroup identity. Comparing the
process-noise draws with Q alone would be circular, because the bench draws
them from Q.

**Sensors (T060).** The base clock runs at 20 Hz for 20 s.

| Sensor | Measures | Rate | Declared covariance |
| --- | --- | --- | --- |
| camera | position | 10 Hz | R = [[0.04, 0.012], [0.012, 0.04]] m^2 |
| encoder | speed \|v\| | 20 Hz | 0.01 m^2/s^2 |
| IMU | heading rate, as an integrating gyro wrap(theta_k - theta_{k-1})/dt | 20 Hz | 4e-4 rad^2/s^2 |
| tracker | position | 2 Hz | 0.0025 I m^2 |

The truth and each sensor draw from their own `SeedSequence.spawn` child of a
PCG64 generator, so changing one sensor's settings leaves the other streams
untouched. Noise is drawn through Cholesky factors, not SVD, so that draws do
not depend on the platform's SVD sign convention. The bench retains the
truth, the raw readings, the noise draws and the declared covariances. The
encoder and IMU are nonlinear in the state; no experiment here fuses them.

The later experiments use the same model with dt = 0.1 s, q = 0.05 (q = 0.5 in
T066), the camera R, prior mean (0, 0, 1, 0.5) and prior covariance
diag(0.25, 0.25, 0.04, 0.04).

**Exact predictions.** A mismatched linear filter, one whose model differs
from the truth in R, H, frame or clock, is itself linear-Gaussian in the joint
vector [x; x_hat]. `mismatch_moments` propagates that vector's mean and
covariance exactly and returns the expected NEES, NIS and squared error at
every tick. Where a counterexample involves a mismatched linear filter
(ignored correlation in T062, zero-filling in T069, stale clock in T070, frame
mismatch in T071, post-expiry drift in T072), its size is predicted this way,
or by running the same linear filter on noise-free mean readings, and the
simulation is checked against the prediction with a run-level z-test.

## Consistency tests and their limits

| Test | Statistic | Reference | Limits |
| --- | --- | --- | --- |
| Covariance moments (T061) | z = (S_ij - R_ij) / sqrt((R_ij^2 + R_ii R_jj)/N), known zero mean | Bonferroni bound at family error 1e-3 | Gaussian sampling law; power is exact from chi2(N) and grows as sqrt(N), so slow sensors hide misstatements; one run's pass or fail is a draw, not a property |
| ANEES / ANIS (T062, T066, T069, T071, T074) | per-tick mean over N independent runs | chi2(mN)/N two-sided 99% interval; pass if at least 90% of ticks are inside | needs ground truth (NEES) or a correct S (NIS); a fraction-inside threshold tolerates time correlation but is not a formal test; post-fit residuals normalized by R - H P+ H^T give the same statistic as the innovation NIS, not a second test |
| Whitened innovation covariance (T062) | L^-1 nu with S = L L^T; sample covariance against I | Bonferroni z-bound | sees off-diagonal errors that the NIS trace averages away |
| Grand mean with run-level SE (T066, T070, T071) | per-run means, then their spread | z against an exact prediction | assumes independent runs; within-run correlation is absorbed |
| Mean innovation (T070) | whitened innovations averaged per run after burn-in | Bonferroni z against 0 or against the exact expected bias | sees only biases that some sensor makes observable |
| Gate rates (T067, T068) | rejection counts | Wilson 99.9% score interval; noncentral chi2 detection law | open-loop independence rests on white innovations; closed-loop gating breaks it |
| Estimate error with and without gating (T068) | per-run MSE after burn-in | paired z over all runs and a sign test of per-run wins | a comparison restricted to runs selected by the filter's own outcome (no lock-out) is post hoc and descriptive, not a test |
| Batch reference (T074) | filter against the batch normal-equation posterior | relative difference to roundoff; exact Fractions | validates the algebra for the declared model, not the model; block elimination is itself an information-form forward pass, so only the dense solve avoids recursion over time |

## Results and counterexamples

- **T060: bench.** Regenerating with the same seed reproduces every stream bit
  for bit; seed + 1 changes all of them. Reading counts are camera 200, encoder
  400, IMU 400 and tracker 40. The residual z - h(truth) - noise is at most
  2e-14. Changing the tracker's rate leaves the other streams unchanged.
- **T061: declared covariance.** All 28 standardized moments are within the
  Bonferroni bound (max |z| 3.24 against 4.13). Q matches the continuous-time
  integral to 2e-16 relative; the discrete white-noise-acceleration
  alternative misses it by 25% in the position block. *Counterexample* to "one
  bench run detects a 10% misstatement for every sensor": the exact
  chi-square power of the variance test against a 10% understatement is at
  least 0.9997 per variance for the 10–20 Hz streams but only 0.490 for the
  2 Hz tracker (two independent variances, 0.286 each); 400 replicate tracker
  streams flag it at 0.475. In this seed's run the tracker is not flagged
  (z 3.48 against 4.13), but that outcome is a coin flip at N = 2000. The
  misstatement flagged half the time is z_crit sqrt(2/N) to within 0.3% of
  the exact value, 13.1% for the tracker.
- **T062: correlated noise.** With the correct cross-correlated R (a common
  mode of 0.09 I), ANEES is inside the interval at 98% of ticks. Dropping the
  cross-covariance makes ANEES exceed the bound at every tick (grand NEES 5.51;
  exact prediction 5.44). *Counterexample* to "ignoring correlation is
  harmless". A second counterexample: the mean NIS still passes (3.92), and
  only the whitened-innovation covariance test exposes the error.
- **T063: frame transforms.** Monte Carlo covariances after rotation,
  translation, a metre-to-millimetre change and their composite match J P J^T.
  Mahalanobis distance is invariant to within 2e-13 relative. *Counterexamples:*
  rotating a vector by 35° without rotating its covariance raises the mean d^2
  from 2 to 14.4 (as predicted by tr(P^-1 R P R^T)), and a 99% gate then
  rejects 42.5% of valid readings. Converting to millimetres while keeping the
  covariance in m^2 multiplies d^2 by exactly 10^6.
- **T064: Jacobi transfer.** `jacobi.transfer` matches cos/sin and cosh/sinh.
  Phi P Phi^T predicts the covariance of (lateral offset, lateral rate) over
  600 perturbed geodesics per surface, started with `jacobi.perturbed_start`
  and integrated with `integrators.integrate_fixed` on a batched right-hand
  side checked against `Surface.geodesic_rhs`. On the sphere, lateral
  variance collapses at the conjugate point s = π to the initial lateral
  variance, 0.9% of its peak. *Counterexample* to "lateral uncertainty grows
  monotonically". The first-order variance is too large by about cos^2(s)
  sigma^2 on the sphere and cosh^2(s) sigma^2 on the hyperbolic plane. It
  reaches 10% error at a heading std of 0.32 rad on the sphere (s = 0.9π) but
  0.034 rad on the hyperbolic plane (s = 3). Gauss–Hermite quadrature of the
  exact closed forms (asin(sin a sin s), asinh(sin a sinh s)) is the reference.
- **T065: filter-induced correlation.** Successive errors are correlated with
  Cov(e_{k+j}, e_k) = [(I - K H) F]^j P. The scalar case q = 1, r = 30 has the
  rational steady state M = 6, P = 5, K = 1/6. *Counterexample:* averaging 20
  successive outputs as if they were independent understates the variance by
  8.08× (exact), and a naive 95% interval covers only 51%. The same filter's
  innovations are white.
- **T066: residual normalization.** NIS with S is consistent (run-averaged
  NIS inside the per-tick 99% interval at 98% of ticks). Post-fit residuals
  have covariance R - H P+ H^T = R S^-1 R, and normalizing them by it
  reproduces the innovation NIS exactly (to 4e-14 relative): it is the same
  statistic, not a second consistency result. *Counterexamples:* innovations
  normalized by the raw R inflate to tr(R^-1 S) = 3.36, and post-fit residuals
  normalized by R deflate to tr(S^-1 R) = 1.23, which would hide an
  inconsistent filter.
- **T067: gating.** Open-loop false-rejection rates match 1 - p for p = 0.9,
  0.99 and 0.999. *Counterexample:* in closed loop at p = 0.9 the rate is 13.5%.
  A rejected reading signals a large prior error that the filter keeps; after a
  rejection the next rejection rate is 30%. At p = 0.99 the excess is within
  sampling error here, which does not show it is zero. A 99% gate on the raw-R
  distance rejects 3.7% of valid readings, about 3.7 times the nominal rate.
- **T068: outliers.** Gross 1.5 m outliers are detected at 99.3%, against a
  predicted 99.1% from the noncentral chi-square law at each outlier's own
  prior. Over all 400 runs the RMSE is 0.196 m ungated, 0.178 m gated and
  0.138 m for an oracle that skips exactly the contaminated readings.
  *Counterexamples:*
  - Gating wins in 374 of 400 runs, yet its mean-squared-error improvement
    over fusing every reading is not significant (paired z -0.70), because the
    lock-out runs lose heavily.
  - Cold-start lock-out: a gross outlier in the first reading passes the
    broad-prior gate, and the corrupted state then rejects runs of valid
    readings. All 10 lock-out runs start this way; the worst has an RMSE of
    1.98 m.

  Post hoc, removing the 10 lock-out runs (selected by the gated filter's own
  failure, which favours gating by construction) gives gated RMSE 0.141 m
  against 0.138 m for the oracle and 0.196 m ungated. This conditioned
  comparison is descriptive, not a test.
  Further counterexamples:
  - On clean data, gating increases MSE (paired z 9.6).
  - Subtle 0.3 m outliers are detected only 8% of the time, as predicted, and
    gating them costs more than it saves.
- **T069: missing data.** Prediction-only steps grow the covariance exactly
  (relative error 9e-16) in the session API and in the batch schedule. NEES
  stays consistent through a 30-tick gap with 20% dropout. The session refuses
  the requested strategies `zero_fill` (`zero_fill_refused`) and `hold_last`
  (`gap_strategy_refused`), NaN values (`nonfinite_observation`), absent values
  (`missing_reading`) and a fractional prediction tick (`malformed_tick`), and
  a refused request leaves its state unchanged. It cannot detect zero-filling
  done before an `Observation` is built: a reading of (0, 0) looks like any
  other reading. *Counterexample:* zero-filling by hand gives a grand NEES of
  2306, against 2281 predicted exactly from the joint moments.
- **T070: stale clock.** The camera reports one tick late. With a correctly
  clocked tracker alongside, the whitened innovations are biased (|z| 20.8
  against zero), in agreement with the exact linear prediction (|z| 1.62
  against the predicted bias). An EKF with an offset state
  (h = p - tau v) removes the bias and estimates tau at 0.0985 ± 0.013 s.
  *Counterexample:* with the stale camera alone, the innovations are unbiased
  (|z| 0.54), because a lagged constant-velocity path is itself a
  constant-velocity path. The estimate is still biased by about -tau E[v].
- **T071: frame mismatch.** A tracker in a frame rotated by 2°, fused with a
  world-frame camera, inflates late ANIS to 7.45 (exact prediction 7.47). Near
  the rotation centre the inflation is about 2% and goes undetected
  (*counterexample* to "a passing NIS shows the frames agree"). The rotated
  tracker fused alone keeps NIS consistent while ANEES reaches 343
  (*counterexample* to "frame mismatch always inflates NIS"). The session
  therefore refuses frame-id mismatches outright (`frame_mismatch`) and
  retains the refused observation. An explicit `FrameTransform` restores
  consistency.
- **T072: calibration expiry.** With a record valid on [0, 60), 59 readings
  are fused and 41 are refused (`calibration_expired`, first at tick 60). All
  100 are retained with matching digests. The state is bitwise equal to a
  shadow session that never saw the refused readings. Revoked, unknown,
  other-sensor and other-frame records are refused (`calibration_frame_mismatch`
  compares the record's frame with the frame the reading was made in, which an
  explicit `FrameTransform` carries along). *Counterexample* under a declared
  post-expiry drift of 1 cm per tick: fusing the expired readings gives a
  post-expiry grand NEES of 14.0, against 13.8 predicted exactly from the joint
  moments.
- **T073: track lost.** The camera stops after tick 30. The track is declared
  lost at tick 41, exactly when the closed-form 99% ellipse first exceeds 1 m.
  Admission (`track_lost`) and fusion (`track_lost_requires_reacquisition`)
  are refused, and the refused fusion leaves the state, covariance, clock and
  status bit for bit unchanged. Reacquisition needs two consecutive readings
  no older than the session clock (`out_of_order` otherwise) and is refused
  while the track is still held. Its covariance
  [[R, R/dt], [R/dt, 2R/dt^2 + q dt/3 I]] equals the linear error map
  J blockdiag(R, R, Q(dt)) J^T to roundoff and matches Monte Carlo (mean NEES
  4.01). *Counterexample* to "a coasting track stays consistent for any gap":
  under an unmodelled 0.2 rad/s turn with white acceleration through the turn,
  the expected NEES tr(P_f^-1 (T P T^T + Q_turn)) + d^T P_f^-1 d exceeds the
  99% quantile after 44 ticks, when the ellipse radius is 4.6 m; a simulation
  of the turning truth gives 13.30 there. The 1 m rule had fired after 11.
- **T074: analytic reference.** The covariance-form Kalman estimate and
  covariance equal the batch information-form posterior at K = 10, 40 and 100
  to within 1e-13 relative. That posterior is computed by block-tridiagonal
  elimination, whose forward pass is itself an information filter, and also by
  a dense elimination of the whole normal equations (no recursion over time)
  at K = 10 and 40 (44 and 164 unknowns). In rational arithmetic the scalar
  filter and the batch posterior are identical. Estimation-error NEES is
  consistent (ANEES inside the per-tick 99% interval at 99% of ticks).
- **T075: typed admission.** `Observation`, `CandidateState` and
  `AdmittedState` are unrelated types. An `AdmittedState` exists only through
  `FusionSession.admit`, is immutable, and nothing is admitted automatically.
  Fresh means the most recently issued candidate at the session tick, so a
  candidate superseded by a second update at the same tick is refused
  (`stale_candidate`). Fourteen scenarios cover the 13 declared checks, and
  each refuses its candidate with its check's own code. Deleting the targeted
  check changes every scenario's outcome (14/14 mutants killed). In 12
  scenarios the violating candidate is then admitted. Two are backed up by a
  later check: without the declaration check, the missing value fails the
  innovation check closed. Without the provenance check, a forged candidate
  is still not the latest issue, so the freshness check refuses it. The
  provenance check therefore adds a specific refusal code, not an extra guard.
  The read-only scenario uses a session constructed read-only, because the
  flag cannot be flipped.
- **T076: defaults.** `FusionSession()` is read-only. Its authority equals
  `ciw.declared_workload.AUTHORITY`: `sensor_fusion` and `state_admission` are
  `not_performed` and `physical_truth` is `not_established`. Every estimation
  or admission call (`initialize`, `predict`, `handle_gap`, `fuse`,
  `reacquire`, `admit`) is refused with `read_only_session`, while
  observations are still recorded and no state exists. Registering or
  revoking a calibration record is allowed: it is record-keeping and grants
  nothing. The flag is fixed at construction. Rebinding `read_only` or
  `authority` is refused with `read_only_session`, and the authority is a
  read-only mapping derived from the flag, so it cannot disagree with it.
  Enabling fusion at construction changes the authority only to
  `synthetic_only`.

## Refusal vocabulary of the session API

`read_only_session` (also for rebinding `read_only` or `authority`),
`malformed_observation`, `malformed_calibration`, `malformed_tick`,
`missing_reading`, `nonfinite_observation`, `covariance_not_positive_definite`,
`nonfinite_state`, `frame_mismatch`, `calibration_unknown`,
`calibration_revoked`, `calibration_frame_mismatch`, `calibration_expired`,
`unsupported_observation`, `not_initialized`, `out_of_order` (fusion,
prediction and reacquisition), `zero_fill_refused`, `gap_strategy_refused`,
`track_lost_requires_reacquisition`, `reacquisition_not_needed`,
`reacquisition_needs_consecutive_readings`, and the admission codes
`admission_checks_not_declared`, `not_a_candidate`, `digest_mismatch`,
`unknown_candidate`, `frame_mismatch`, `stale_candidate`, `track_lost`,
`uncertainty_exceeds_limit`, `inconsistent_innovation`,
`calibration_revoked`, `admission_requires_gate` and
`admitted_state_immutable`. Every refusal leaves the estimate, covariance,
clock and track status unchanged; the refused observation is retained in the
log with its disposition. The experiments record "nothing refused" as `none`,
the evidence convention.

## Open questions

- **Nonlinear sensors.** The encoder speed and IMU heading rate are generated
  but not fused. An EKF/UKF consistency study, with linearization breakdown
  measured as in T064, is the natural next step.
- **Uncertain extrinsics and clocks.** T063 and T071 treat the transform as
  exact, and T070 treats the lag as a constant. Uncertain extrinsics need a
  J_theta Sigma_theta J_theta^T term, and drifting clocks need random-walk
  offset states.
- **Gate lock-out.** Recovery from lock-out, by covariance inflation or by
  automatic reacquisition after repeated rejections, is not implemented in
  the gated filter. T073's explicit reacquisition is the only recovery path.
- **Real sensors.** Real sensor noise, dropout, outlier and drift processes
  need acquired hardware data (`hardware_measured` evidence). Nothing here
  substitutes for it.
