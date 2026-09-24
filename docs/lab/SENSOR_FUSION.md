# Sensor-fusion experiments (T060–T076)

Implementation: `src/ciw/lab/sensor_fusion.py` (section entry point and the
bench tasks T060–T062), `sensor_fusion_geometry.py` (T063–T064),
`sensor_fusion_filtering.py` (T065–T068), `sensor_fusion_robustness.py`
(T069–T071) and `sensor_fusion_admission.py` (T072–T076). Shared code:
`sensor_fusion_bench.py` (bench generator and filter algebra),
`sensor_fusion_objects.py` (typed observation, candidate and admitted-state
API), `sensor_fusion_intake.py` (the adapter from section-4 observation
records to fusion readings, see
[From section-4 records to fusion](#from-section-4-records-to-fusion-the-intake))
and `sensor_fusion_common.py` (evidence helpers). Tests:
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
in about 10 s and its tests in about 11 s on one core, with NumPy only (SciPy is
used only in one optional test that cross-checks the chi-square quantiles and
the noncentral chi-square CDF). Retained artifacts are byte-identical whether
BLAS runs on one thread or many: large Gram products use `einsum` loops and
the dense batch solve uses its own elimination, not multithreaded BLAS/LAPACK.

Every computational finding declares a per-finding `uncertainty`
(`monte_carlo_95ci` from a run-level or binomial standard error, or, for a
reported max |z| over a family of m moments, the half-width of the central
95% interval of the maximum of m independent |N(0, 1)|; `roundoff` for
identities; `truncation_bound` for RK4 and bisection; `reference_error` for
quadrature, closed-form approximations and exact results). Every task
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
closed form evaluated by CIW code. Agreement between two CIW implementations
(the fusion API against the batch filter, the filter against the batch
posterior) is recorded as a `cross_implementation` check.

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
encoder and IMU are nonlinear in the state, and no experiment here fuses
them. T060 records an EKF/UKF fusion of both streams, under the T052 and T053
error models, as its deferred research question.

The later experiments use the same model with dt = 0.1 s, q = 0.05 (q = 0.5 in
T066), the camera R, prior mean (0, 0, 1, 0.5) and prior covariance
diag(0.25, 0.25, 0.04, 0.04).

**Exact predictions.** A mismatched linear filter, one whose model differs
from the truth in R, H, frame or clock, is itself linear-Gaussian in the joint
vector [x; x_hat]. `mismatch_moments` propagates that vector's mean and
covariance exactly and returns the expected NEES, NIS and squared error at
every tick. Where a counterexample involves a mismatched linear filter
(ignored or assumed correlation in T062, zero-filling in T069, stale clock in
T070, frame mismatch in T071, post-expiry drift in T072), its size is
predicted this way,
or by running the same linear filter on noise-free mean readings, and the
simulation is checked against the prediction with a run-level z-test.

## From section-4 records to fusion (the intake)

Section 4 ([OBSERVATION.md](OBSERVATION.md)) defines typed observation
records: mode, frame id, clock id, epoch, clock basis, time, calibration
reference, raw reference, latency and variance components. It also defines
their retention and admission in an `ObservationLedger`.
`ciw.lab.sensor_fusion_intake.ObservationIntake` turns those records into
`FusionSession` readings. The pipeline is:

```
typed observation (section 4)
→ retained, then admitted in an ObservationLedger (synthetic_only)
→ declared FrameMapping / ClockMapping chain to the channel frame and the fusion clock
→ fusion reading (sensor, frame, tick, projected value and covariance, calibration)
→ FusionSession.fuse → CandidateState
→ explicit FusionSession.admit → AdmittedState
```

An intake binds one session to three kinds of declaration:

* `IntakeChannel` records say which mode feeds which fusion sensor, from which
  frame, through which rank-2 projection and offset, for which section-4
  calibration reference and which fusion `CalibrationRecord`.
* A `FusionClock` says that tick k is k·dt on a named clock and epoch, in the
  acquisition basis. Its dt must equal the session's step.
* The section-4 frame and clock mappings.

The intake declares the geometry class of the session state:

* `extrinsic`: a position in an embedding frame.
* `intrinsic`: a position in a surface chart.

The reading's covariance is the mode's declared noise covariance, carried
through the applied frame rotations and the projection. For a record that
carries variance components (the T044 split) it is their sum, the same
variance the section-4 `StateStore` updates with. A record reaches the
session only if every step below passes. The step order fixes which code a
record with several faults receives:

1. This intake has not fused it already (`already_fused`). A measurement is
   counted once.
2. It is retained (`not_retained`), admitted (`not_admitted`) and unchanged
   since (`admission_digest_mismatch`). A record whose admission was refused,
   for example for a missing calibration reference, never gets this far.
3. It validates as a section-4 record, with the section-4 codes.
4. Its mode has the state's geometry class (`extrinsic_for_intrinsic`,
   `geometry_mismatch`).
5. A channel is declared for its mode (`no_channel`).
6. It cites a calibration (`uncalibrated_record` for `not_applied`), and the
   one its channel was declared for (`calibration_not_declared`).
7. Declared mappings take it to the channel frame (`unmapped_frame`) and to
   the fusion clock in the acquisition basis (`unmapped_clock`). The time
   basis changes only through a latency mapping from arrival to acquisition
   that keeps its clock and epoch. So an arrival stamp needs a declared
   latency mapping on its own clock. A mapping that changes the basis
   together with the clock or epoch, or back to arrival, is refused
   (`unmapped_clock`), because the latency folded into it cannot be checked.
   The latency mapping must agree with the record's own latency
   (`latency_mismatch`). A record that declares no latency takes the
   mapping's.
8. The mapped time lies on the tick grid (`off_tick_grid`) and not before the
   epoch (`before_epoch`).
9. The fusion calibration is registered for the sensor (`calibration_unknown`),
   not revoked (`calibration_revoked`), declared for the channel frame
   (`calibration_frame_mismatch`) and valid at the tick
   (`calibration_expired`). It must not declare a latency again, since the
   clock mapping already applied it (`double_latency`).
10. Its reading differs from every reading this intake has fused
    (`duplicate_reading`). A re-sent copy of a record, with a new sequence
    number and raw reference but the same measurement, converts to an
    identical reading and is refused here.

Then the session fuses the reading, and its own refusals pass through
unchanged (`read_only_session`, `out_of_order`, the NIS gate, track loss). A
record the session refused was not fused and may be offered again.
Every submission is logged with its disposition. Every fused submission keeps
its own lineage entry, in fusion order: the section-4 digest, raw reference,
mapping identities, channel identity, calibrations and ledger admission.
`trace()` compares the readings the session used (fused or reacquired), in
order, with the intake's lineage. It refuses (`untraced_reading`) on any
difference, so a reading fused directly on the session is caught, even a
repeat of one the intake fused. The intake has no reacquisition of its own,
so a session reacquired from readings that did not come through it fails the
trace too. The intake has no authority of its own. It fuses through the
session, so a default session keeps it read-only.

No section-4 mode delivers a two-component intrinsic position. A
surface-chart session therefore has no admissible channel, and the intrinsic
case appears only as a refusal. Scalar records (distances, encoder
displacement) and the IMU orientation have no channel into the planar
position session. Both are deferred research questions (T075, T060).

The demonstration (`sensor_fusion_intake.demonstration`, seed 752027) uses
twelve room-frame tracker records. Each is stamped on arrival at the tracker
clock, with the tracker mode's declared σ = 0.5 mm and a 1/128 s latency. The
fusion step is 0.125 s, so every mapped stamp is dyadic and lands on the
grid exactly. T058, T059, T075 and T076 share one run of it.

## Consistency tests and their limits

| Test | Statistic | Reference | Limits |
| --- | --- | --- | --- |
| Covariance moments (T061) | z = (S_ij - R_ij) / sqrt((R_ij^2 + R_ii R_jj)/N), known zero mean | Bonferroni bound at family error 1e-3 | Gaussian sampling law; power is exact from chi2(N) and grows as sqrt(N), so slow sensors hide misstatements; one run's pass or fail is a draw, not a property |
| ANEES / ANIS (T062, T066, T069, T071, T074) | per-tick mean over N independent runs | chi2(mN)/N two-sided 99% interval; pass if at least 90% of ticks are inside | needs ground truth (NEES) or a correct S (NIS); a fraction-inside threshold tolerates time correlation but is not a formal test; post-fit residuals normalized by R - H P+ H^T give the same statistic as the innovation NIS, not a second test |
| Whitened innovation covariance (T062) | L^-1 nu with S = L L^T; sample covariance against I | Bonferroni z-bound | sees off-diagonal errors that the NIS trace averages away |
| Grand mean with run-level SE (T066, T070, T071) | per-run means, then their spread | z against an exact prediction | assumes independent runs; within-run correlation is absorbed |
| Mean innovation (T070) | whitened innovations averaged per run after burn-in | Bonferroni z against 0 or against the exact expected bias | sees only biases that some sensor makes observable |
| Gate rates (T067, T068) | rejection counts | Wilson 99.9% score interval; noncentral chi2 detection law | open-loop independence rests on white innovations; closed-loop gating breaks it |
| Estimate error with and without gating (T068) | per-run MSE after burn-in | sign test of per-run wins over all runs; the paired mean difference and its z are reported, not tested | a non-significant paired z is not evidence that the mean effect is zero; a comparison restricted to runs selected by the filter's own outcome (no lock-out) is post hoc and descriptive, not a test |
| Batch reference (T074) | filter against the batch normal-equation posterior | relative difference to roundoff; exact Fractions | validates the algebra for the declared model, not the model; block elimination is itself an information-form forward pass, so only the dense solve avoids recursion over time |

## Results and counterexamples

- **T060: bench.** Regenerating with the same seed reproduces every stream bit
  for bit; seed + 1 changes all of them. Reading counts are camera 200, encoder
  400, IMU 400 and tracker 40. The trajectory is recomputed separately from the
  retained initial state and process-noise draws, by cumulative sums with the
  declared dt rather than the generator's transition matrix; it matches the
  retained truth to 7e-14. Every sensor function is evaluated on that
  recomputed trajectory, so positions are not re-read from the array they were
  built from. The residual z - h(trajectory) - noise is then at most 1.4e-12
  (largest for the heading rate, which divides velocity roundoff by |v| dt).
  Changing the tracker's rate leaves the other streams unchanged.
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
- **T062: independent and correlated noise.** The same truth is observed
  through correlated noise (a common mode of 0.09 I) and through independent
  noise with the same marginal variances. Each is filtered with the matching
  and the mismatched R. With the matching R, ANEES is inside the per-tick 99%
  interval at 98% of ticks in both cases. Dropping a real cross-covariance
  makes ANEES exceed the bound at every tick (grand NEES 5.51; exact moment
  recursion 5.44). *Counterexample* to "ignoring correlation is harmless". A
  second counterexample: the mean NIS still passes (3.92). Among
  innovation-based tests, which need no ground truth, only the
  whitened-innovation covariance test exposes the error. The reverse mismatch,
  assuming a correlation the noise does not have, makes the filter
  underconfident (ANEES below the bound at 99% of ticks, grand 3.14) and less
  accurate (exact RMSE 0.1722 m against 0.1697 m). Its NIS exceeds the bound
  at every tick (grand 7.72). *Counterexample* to "assuming correlation is a
  conservative, harmless choice". All eight grand means agree with the exact
  moment recursion within 1.2 run-level standard errors.
- **T063: frame transforms.** Monte Carlo covariances after rotation,
  translation, a metre-to-millimetre change and their composite match J P J^T.
  Every transformed covariance, including the `FrameTransform` output, is
  symmetric to 1e-16 relative. The transformed sample means match means built
  from the declared rotation, translation and scale (max |z| 1.70 against
  4.00), so a translation leaking into the velocity block would be caught.
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
  Only constant curvature is covered. The references variable curvature
  needs already exist: T002 validates the geodesic integrator against
  34-digit solutions on the torus, saddle and Gaussian bump, and T006 checks
  the Jacobi columns against finite-difference flow on the torus and bump.
  Repeating this study on those surfaces is T064's deferred research question.
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
  statistic, not a second consistency result. The fusion API records the same
  statistic: the NIS on each `FusionSession` candidate, which the admission
  gate's innovation check reads, equals nu^T S^-1 nu from the gain schedule
  (to 3e-14 relative). Its ratio to the raw-R value stays below
  max_k lambda_max(S_k^-1 R) = 0.70, whereas a session normalizing by R
  would give 1. *Counterexamples:* innovations normalized by the raw R inflate
  to tr(R^-1 S) = 3.36, and post-fit residuals normalized by R deflate to
  tr(S^-1 R) = 1.23, which would hide an inconsistent filter.
- **T067: gating.** Open-loop false-rejection rates match 1 - p for p = 0.9,
  0.99 and 0.999. *Counterexample:* in closed loop at p = 0.9 the rate is 13.5%.
  A rejected reading signals a large prior error that the filter keeps; after a
  rejection the next rejection rate is 30%. At p = 0.99 the excess is within
  sampling error here, which does not show it is zero. The fusion API has the
  same gate: `FusionSession(gate_probability=p)` refuses a reading whose NIS
  exceeds the chi-square quantile with `innovation_gate_rejected`, keeps it in
  the log with that disposition and leaves its state unchanged. Replayed over
  the first 40 runs, it makes run_gated's decision for every reading (534
  rejections at p = 0.9, 42 at p = 0.99, none differing) and ends in the same
  state to 7e-16 relative. A 99% gate on the raw-R distance rejects 3.7% of
  valid readings, about 3.7 times the nominal rate.
- **T068: outliers.** Gross 1.5 m outliers are detected at 99.3%, against a
  predicted 99.1% from the noncentral chi-square law at each outlier's own
  prior. Over all 400 runs the RMSE is 0.196 m ungated, 0.178 m gated and
  0.138 m for an oracle that skips exactly the contaminated readings. Gating
  lowers the per-run MSE in 374 of 400 runs (sign test). The 10 lock-out runs
  together give back 62% of the summed MSE that gating gains in the other runs
  (4.47 of 7.24 m^2), although only 3 of them are lost outright after the
  burn-in. Reported descriptively, not as tests: the paired mean MSE
  difference favours gating (-0.0069 m^2, z -0.70; a non-significant z is not
  evidence that the mean effect is zero). Post hoc, removing the 10 lock-out
  runs (selected by the gated filter's own failure, which favours gating by
  construction) gives gated RMSE 0.141 m against 0.138 m for the oracle and
  0.196 m ungated. The fusion API's gate
  (`FusionSession(gate_probability=0.99)`) replays the first 30 runs and every
  lock-out run with no decision differing from run_gated (381 rejections over
  3900 readings). All 10 lock-out runs lock out in the session as well.
  *Counterexamples:*
  - Cold-start lock-out: a gross outlier in the first reading passes the
    broad-prior gate, and the corrupted state then rejects runs of valid
    readings. All 10 lock-out runs start this way; the worst has an RMSE of
    1.98 m.
  - On clean data, gating increases MSE (paired z 9.6).
  - Subtle 0.3 m outliers are detected only 8% of the time, as predicted, and
    gating them costs more than it saves.
- **T069: missing data.** Prediction-only steps grow the covariance exactly
  (relative error 9e-16) in the session API and in the batch schedule. NEES
  stays consistent through a 30-tick gap with 20% dropout. After a 50-tick
  prediction-only gap that loses a 1 m track, the session refuses the
  requested strategies `zero_fill` (`zero_fill_refused`) and `hold_last`
  (`gap_strategy_refused`) and a fractional prediction tick
  (`malformed_tick`). It also refuses a reading older than its clock
  (`out_of_order`), a reading under an expired calibration
  (`calibration_expired`) and a reading into the lost track
  (`track_lost_requires_reacquisition`, raised after the prediction was
  computed on copies). The state, covariance, clock, track status and latest
  candidate are bitwise unchanged after each of these refusals, and every
  refused reading is retained with its refusal. NaN values
  (`nonfinite_observation`) and absent values (`missing_reading`) are refused
  by the `Observation` type before any session call. The session cannot
  detect zero-filling done before an `Observation` is built: a reading of
  (0, 0) looks like any other reading. *Counterexample:* zero-filling by hand gives a grand NEES of
  2306, against 2281 predicted exactly from the joint moments.
- **T070: stale clock.** The camera reports one tick late. With a correctly
  clocked tracker alongside, the whitened innovations are biased (|z| 20.8
  against zero), in agreement with the exact linear prediction (|z| 1.62
  against the predicted bias). An EKF with an offset state
  (h = p - tau v) removes the bias and estimates tau at 0.0985 ± 0.013 s.
  *Counterexample:* with the stale camera alone, the innovations are unbiased
  (|z| 0.54), because a lagged constant-velocity path is itself a
  constant-velocity path. The estimate is still biased by about -tau E[v]
  ((-0.099, -0.040) m). The fusion API applies a declared lag: with
  `latency_ticks=1` in the camera's `CalibrationRecord`, the session fuses
  each reading at the tick it refers to. Over 5 replayed runs its estimates
  equal the correctly timed filter to 4e-16 relative, and that filter's
  position error mean is (0.0023, 0.0023) m (|z| 1.45), so the bias
  disappears. The session refuses rather than retrodicts a reading that is
  older than its clock: a lagged camera reading delivered after a newer
  tracker reading, one that refers to a time before tick 0, an undelayed older
  reading and a backward prediction are all refused with `out_of_order`, with
  the state bitwise unchanged. A lagged reading that refers to the current
  tick is fused at that tick.
- **T071: frame mismatch.** A tracker in a frame rotated by 2°, fused with a
  world-frame camera, inflates late ANIS to 7.45 (exact prediction 7.47). Near
  the rotation centre (ticks 1-20) the predicted inflation is about 2% (4.09).
  The per-tick NIS test does not flag it: 95% of ticks are inside
  (*counterexample* to "a passing NIS shows the frames agree"). The observed
  early grand NIS is 4.13, 3.3%, and agrees with the prediction (z 1.02). A
  pooled run-level test against 4 comes close to flagging it (z 3.32 against a
  Bonferroni bound of 3.66). The rotated
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
  while the track is still held. Both readings of a refused reacquisition are
  retained with its refusal code (6 readings over the 3 refusals, none left as
  merely recorded). Its covariance
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
  consistent (ANEES inside the per-tick 99% interval at 99% of ticks). The
  filter, block elimination and dense solve are all CIW code, so their
  agreement is recorded as `cross_implementation`, not as independent
  verification.
- **T075: typed admission.** `Observation`, `CandidateState` and
  `AdmittedState` are unrelated types. An `AdmittedState` exists only through
  `FusionSession.admit`, is immutable, and nothing is admitted automatically.
  Fresh means the most recently issued candidate at the session tick, so a
  candidate superseded by a second update at the same tick is refused
  (`stale_candidate`). A candidate carries every innovation fused since the
  last admitted, initialized or reacquired state, and every calibration its
  state was built under since the last initialization or reacquisition. A
  prediction issued after an inconsistent update therefore inherits
  `inconsistent_innovation`, and a prediction issued after a revocation
  inherits `calibration_revoked`. Before this, a prediction carried no
  innovation and passed the innovation check vacuously. A later consistent
  update is still refused until an explicit re-initialization, after which a
  consistent update is admitted. After an admission, the next candidate carries
  only the innovations fused after it. Sixteen scenarios cover the 13
  declared checks, including the two predictions, and each refuses its
  candidate with its check's own code. Deleting the targeted check changes
  every scenario's outcome (16/16 mutants killed). In 14 scenarios the
  violating candidate is then admitted. Two are backed up by a later check:
  without the declaration check, the missing value fails the innovation check
  closed. Without the provenance check, a forged candidate is still not the
  latest issue, so the freshness check refuses it. The provenance check
  therefore adds a specific refusal code, not an extra guard. The read-only
  scenario uses a session constructed read-only, because the flag cannot be
  flipped. Because the declared quantile applies to each carried innovation,
  a consistent track with n innovations since its last admission is refused
  with probability 1 - p^n. *End to end through the intake:* twelve tracker
  records are retained and admitted in a ledger, mapped and fused. The fused
  state equals the block-eliminated batch posterior of the hand-mapped
  readings to 4e-15 (mean) and 4e-16 (covariance), relative. Both are CIW
  code, so this is a `cross_implementation` check. Nothing is admitted before
  the explicit gate call, which uses declared thresholds (NIS 0.999, position
  σ ≤ 1 cm) and admits one state. All twelve fused readings trace to
  ledger-admitted records with raw references. Before fusing, and leaving the
  session state untouched, the intake refuses the following: a tracker record
  or a camera chord offered to an intrinsic surface-chart state
  (`extrinsic_for_intrinsic`); an encoder record (`geometry_mismatch`); a
  chord in the extrinsic session (`no_channel`); uncalibrated or
  undeclared-calibration records; an expired or revoked fusion
  calibration, or one that declares the latency twice; the first record
  offered again (`already_fused`); and a re-sent copy of it under a new
  sequence number and raw reference (`duplicate_reading`). The session's own
  `out_of_order` refusal of a reading older than its clock passes through.
  The twelve traced readings come from twelve distinct records. On a second
  session, fusing again directly a reading the intake fused makes `trace()`
  refuse with `untraced_reading`.
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
  `synthetic_only`. T076 runs no generator, so its `production_acceptance`
  finding cites the default-argument audit, not a seed. The defaults hold
  through the intake. A default `ObservationLedger` refuses admission with
  `read_only_session`, so the intake refuses its records as `not_admitted`.
  A default session refuses an admitted record's fusion with
  `read_only_session`, and no state is created. The vocabulary audit
  (`admission_vocabulary_audit`) collects every `state_admission` value
  written by the section-4 `StateStore` and `ObservationLedger`, the session
  with its candidate and admitted states, and the intake lineage, each both
  read-only and writable. The values seen are exactly `not_performed` and
  `synthetic_only`. The section-4 store used to write `admitted` and now
  writes `synthetic_only` on a writable store, which defaults to read-only.

## Refusal vocabulary of the session API

`read_only_session` (also for rebinding `read_only` or `authority`),
`malformed_observation`, `malformed_calibration`, `malformed_tick`,
`malformed_gate`, `missing_reading`, `nonfinite_observation`,
`covariance_not_positive_definite`, `nonfinite_state`, `frame_mismatch`,
`calibration_unknown`, `calibration_revoked`, `calibration_frame_mismatch`,
`calibration_expired`, `unsupported_observation`, `not_initialized`,
`out_of_order` (fusion, prediction and reacquisition; a reading's tick is
its stamp minus the latency declared in its calibration record),
`innovation_gate_rejected` (a session constructed with `gate_probability`),
`zero_fill_refused`, `gap_strategy_refused`,
`track_lost_requires_reacquisition`, `reacquisition_not_needed`,
`reacquisition_needs_consecutive_readings`, and the admission codes
`admission_checks_not_declared`, `not_a_candidate`, `digest_mismatch`,
`unknown_candidate`, `frame_mismatch`, `stale_candidate`, `track_lost`,
`uncertainty_exceeds_limit`, `inconsistent_innovation`,
`calibration_revoked`, `admission_requires_gate` and
`admitted_state_immutable`. The intake adds `malformed_intake`,
`channel_geometry_mismatch`, `extrinsic_for_intrinsic`, `geometry_mismatch`,
`no_channel`, `no_declared_covariance`, `uncalibrated_record`,
`calibration_not_declared`, `unmapped_frame`, `unmapped_clock`,
`latency_mismatch`, `off_tick_grid`, `before_epoch`, `calibration_unknown`,
`calibration_revoked`, `calibration_frame_mismatch`, `double_latency`,
`calibration_expired`, `already_fused`, `duplicate_reading` and
`untraced_reading`. It also passes on the section-4
codes: `not_retained`, `not_admitted`, `admission_digest_mismatch` and the
validation codes. An `IntakeRefusal` is both a `FusionRefusal` and an
`ObservationRefusal`. Every refusal leaves the estimate, covariance,
clock, track status and candidate evidence unchanged. The refused observation
is retained in the log with its disposition, and both readings of a refused
reacquisition carry its code. T069 and T070 test fusion and prediction
refusals for side effects after the session has done work; T067 and T068 do
the same for gate rejections. The experiments record "nothing refused" as `none`,
the evidence convention.

## Open questions

- **Nonlinear sensors.** The encoder speed and IMU heading rate are generated
  but not fused. The open question is an EKF/UKF that fuses them under the
  T052 (scale, bias, backlash) and T053 (gyro bias, angle random walk) error
  models, as `encoder_displacement` and `imu_orientation` records through
  intake channels or as the bench streams. Its consistency and linearization
  breakdown would be measured as in T064, against a Gauss-Newton batch
  reference (T074).
- **Intrinsic channel.** No section-4 mode gives a two-component
  surface-chart position, so an intrinsic session has no admissible input
  (T075). Candidates are a pair of tape readings along declared geodesics,
  or a tracker position unrolled through a declared surface model, as T045
  does for chords. Either would carry the geometry/sensor split.
- **Uncertain extrinsics and clocks.** T063 and T071 treat the transform as
  exact, and T070 treats the lag as a constant. The fusion API applies a
  declared integer latency but neither estimates one nor retrodicts a reading
  older than its clock (no out-of-sequence update). Uncertain extrinsics need a
  J_theta Sigma_theta J_theta^T term, and drifting clocks need random-walk
  offset states. The intake likewise treats every declared frame and clock
  mapping as exact and refuses stamps off the tick grid (T058, T063).
- **Staleness at the fusion boundary.** The intake applies no
  acquisition-age limit (T056's `admit_fresh`). The session fuses each
  reading at its own acquisition tick and refuses one older than its state
  (`out_of_order`), so a reading older than the state is never fused. But
  neither the intake nor `FusionSession.admit` declares the time at which an
  admitted state is used. So the age of the state's newest reading at use is
  not checked. The open question is a declared use time at admission, with
  `stale_observation` for a candidate whose newest reading is older than a
  declared limit then (T058).
- **Reacquisition through the intake.** The intake fuses but does not
  reacquire. A track lost behind an intake can be reacquired only directly
  on the session, and `trace()` then refuses the reacquired readings, since
  they have no lineage. A reacquisition that takes two admitted records and
  keeps their lineage is not implemented.
- **Gate lock-out.** Recovery from lock-out, by covariance inflation or by
  automatic reacquisition after repeated rejections, is implemented neither in
  run_gated nor in the session gate. T073's explicit reacquisition is the only
  recovery path.
- **Admission after an inconsistent innovation.** The only recovery is
  explicit: re-initialization, or reacquisition once the track is declared
  lost. Whether a run of consistent updates should clear the refusal
  automatically is a policy choice, not settled here.
- **Real sensors.** Real sensor noise, dropout, outlier and drift processes
  need acquired hardware data (`hardware_measured` evidence). Nothing here
  substitutes for it.
