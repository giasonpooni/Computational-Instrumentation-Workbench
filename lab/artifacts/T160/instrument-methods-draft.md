# Evidence-labelled computational instruments for curved-surface observation

*Generated draft from retained CIW lab reports. Not peer reviewed. Contains no physical measurement.*

## Abstract

This draft summarizes 59 queued computational tasks. Every result below carries the evidence label assigned by `ciw.lab.evidence`; physical validation is not established for any of them.

## Methods

### T045 — Define typed observation modes: intrinsic geodesic distance; camera chord distance; reconstructed surface distance; encoder displacement; tracker measurement; image residual; IMU orientation.

*Hypothesis.* A typed registry can make every observation declare its frame, clock basis, calibration and geometry class, and can refuse malformed records and silent substitution of one mode for another.

*Model.* Each mode is a tuple (quantity, unit, components, frame kind, clock basis in {acquisition, arrival}, geometry in {intrinsic, extrinsic, none}, noise model, non-observables). A camera chord is extrinsic; it maps to a surface distance only through an inverted chord-arc relation of a declared surface model (T046/T047).

### T046 — Derive the chord-versus-geodesic correction.

*Hypothesis.* Along a unit-speed surface geodesic the chord deficit is s - c = kappa^2 s^3/24 + O(s^4), with kappa the absolute normal curvature; the s^4 term -kappa kappa' s^4/24 vanishes only when kappa kappa' = 0, and the circle formula 2 sin(kappa s/2)/kappa additionally needs zero torsion.

*Model.* Taylor expansion of gamma(s) - gamma(0) in the Frenet frame (T' = kappa N, N' = -kappa T + tau B, B' = -tau N): |delta|^2 = s^2 - kappa^2 s^4/12 - kappa kappa' s^5/12 + O(s^6), so c = s - kappa^2 s^3/24 - kappa kappa' s^4/24 + c5 s^5 with c5 = (3 kappa^4 + 8 kappa^2 tau^2 - 72 kappa kappa'' - 64 kappa'^2)/5760. For a geodesic k_g = 0, hence kappa = |II(T, T)|. For a circle (tau = 0) c = 2 sin(kappa s/2)/kappa exactly.

### T047 — Validate the cylinder chord coefficient analytically and numerically.

*Hypothesis.* On a cylinder of radius R, the geodesic at angle alpha from the circumferential direction has chord deficit s - c = cos^4(alpha) s^3/(24 R^2) + O(s^5): largest circumferentially, zero on rulings, although the Gaussian curvature is zero everywhere.

*Model.* Helix X(s) = (R cos(s cos(a)/R), R sin(s cos(a)/R), s sin(a)); kappa = cos^2(a)/R, tau = sin(a) cos(a)/R; c^2 = (2R sin(s cos(a)/2R))^2 + (s sin(a))^2 = s^2 - cos^4(a) s^4/(12R^2) + cos^6(a) s^6/(360 R^4) + O(s^8).

### T048 — Generate synthetic camera measurements.

*Hypothesis.* A declared pinhole stereo pair observing markers on the cylinder recovers chords exactly without noise; the chord differs from the geodesic distance by the T047 correction, which a declared surface model removes; for long circumferential chords that bias exceeds the error caused by 0.25 px pixel noise, while along rulings it vanishes.

*Model.* x_cam = R X + t, pixels u = f x/z + c; linear DLT triangulation on normalized coordinates; chord = |X_i - X_j|; arc from chord by inverting the helix chord on its monotone branch. Noise: sigma_c^2 = (sigma^2 + 1/12) sum J^2 over the 8 pixel coordinates of a pair (T051) and sigma_s = sigma_c / c'(s) for the converted arc.

### T049 — Add camera calibration perturbations.

*Hypothesis.* Chord errors caused by small calibration errors are linear in the errors, with a Jacobian that matches closed forms where they exist; errors that change horizontal disparity (right-camera yaw, horizontal principal point) dominate, while vertical principal-point and pitch errors barely move chords.

*Model.* Pixels from the true rig are triangulated with a believed rig (focal f + df on both cameras, right principal point + (dcx, dcy), right rotation exp([w]) about its centre). Rectified rig: Z_b = Z f_b/f and X_b = X, so c_b^2 = dX^2 + dY^2 + (dZ f_b/f)^2 and dZ/dcx = -Z^2/(b f).

### T050 — Add perspective and lens-distortion perturbations.

*Hypothesis.* Uncorrected radial distortion displaces pixels by f(k1 r^3 + k2 r^5), producing chord biases that are first order and linear in k1, grow as r^2 with image radius for chords of fixed image extent and vanish when the generating model is inverted; strong barrel distortion makes the model non-invertible inside the image.

*Model.* x_d = x (1 + k1 r^2 + k2 r^4) + [2 p1 x y + p2 (r^2 + 2x^2), p1 (r^2 + 2y^2) + 2 p2 x y]; chord bias ~ J_pix delta_pix with J_pix = d chord / d pixels; only displacement differences (between endpoints, and between cameras through disparity) matter, and for radii differing by dr they are f k1 ((r + dr)^3 - r^3) ~ 3 f k1 r^2 dr (slope 2 in r, 1 in k1). r (1 + k1 r^2) peaks at r = 1/sqrt(-3 k1) with distorted radius (2/3)/sqrt(-3 k1); the fold enters an image of corner radius rho when k1 < -4/(27 rho^2).

### T051 — Add quantization and pixel noise.

*Hypothesis.* Rounding to integer pixels after Gaussian noise gives error variance sigma^2 + 1/12 when the true coordinate is uniformly placed on the pixel grid, and chord standard deviations follow linear propagation through the triangulation Jacobian.

*Model.* e = round(x + n) - x = n + q; with frac(x) ~ U(0,1) independent of n, q ~ U(-1/2, 1/2) is independent of n, so Var e = sigma^2 + 1/12. sigma_c^2 = (sigma^2 + 1/12) sum_j J_j^2 (8 pixel coordinates per pair).

### T052 — Add encoder bias, scale, and backlash.

*Hypothesis.* With reading = (1 + scale) play_b(x) + bias + noise, the backlash error lies in [0, b], changes only during take-up after reversals, and a direction-aware least-squares fit recovers scale, bias and backlash.

*Model.* Play operator: y_k = max(min(y_{k-1}, x_k + b), x_k) (engaged on the positive flank at start); engaged samples: reading = (1 + s) x + beta + (1 + s) b [falling]. A fit on [x, 1] alone absorbs the OLS projection of (1 + s)(play(x) - x) onto [x, 1] into its coefficients.

### T053 — Add IMU drift and orientation noise.

*Hypothesis.* Integrating a gyro with constant bias b and white rate noise of density N gives heading error with mean b t and variance N^2 t; in 3-D the bias error rotates with the body, so transverse bias stays bounded while the random walk stays isotropic.

*Model.* Single axis: e(t) = b t + N W(t). Strapdown: E_{k+1} = exp(-[w] dt) E_k exp([w + b + n] dt); linearized e_{k+1} = exp(-[w] dt) e_k + J_r(w dt)(b + n) dt, so transverse components circle with radius |b_perp|/|w| (maximum 2|b_perp|/|w|).

### T054 — Add asynchronous timestamps.

*Hypothesis.* An uncorrected clock offset delta between two sensors produces an error -v delta to first order at the sample times and exactly -S delta after linear interpolation to a common time, where the interpolant slope S equals the velocity up to a sinc(omega h/2) factor; jitter adds a predictable variance; a declared clock mapping removes the offset.

*Model.* x(t) attributed to t + delta: e = x(t) - x(t + delta) = -v delta - a delta^2/2 + ...; linear interpolation I(t - delta) - I(t) = -S delta off the knots, with S = v(t_mid) sinc(omega h/2) per tone, so regressing on -v delta gives sinc^2(omega h/2) weighted by (A omega)^2; jitter j_i: e = -S((1 - w) j_i + w j_{i+1}) to first order.

### T055 — Add dropped observations.

*Hypothesis.* Dropped observations must stay explicit gaps: zero-filling biases estimates and cannot be detected reliably from values, while explicit gaps keep estimates unbiased; burst losses degrade hold estimates more than independent losses at the same rate.

*Model.* Mean of received samples is unbiased with variance sigma^2 E[1/N]; zero-filled mean has expectation (1 - p) mu. Hold error of a random walk: E e^2 = q E[age] + r with E[age] = p/(1 - p) (Bernoulli) or pi_B / P(bad -> good) (Gilbert-Elliott, all samples lost in the bad state).

### T056 — Add stale-state observations.

*Hypothesis.* An observation used after its validity age must be flagged, with age measured from acquisition, not arrival; using a stale position of a moving target costs velocity x age to first order.

*Model.* age = t_use - (t_arrival - latency); stale iff age > limit. Error x(t) - x(t - a) = v a - x'' a^2/2 + ..., exact v a for constant velocity; |residual| <= max|x''| a^2 / 2.

### T057 — Compare raw, filtered, and smoothed measurements.

*Hypothesis.* For a linear-Gaussian constant-velocity track, the RTS smoother's covariance is below the filter's in matrix order and the ensemble RMSE orders smoothed <= filtered <= raw, with NEES consistent with chi-square bounds; the ordering holds in the ensemble, not per sample.

*Model.* x_{k+1} = F x_k + w, F = [[1, dt], [0, 1]], Q = q [[dt^3/3, dt^2/2], [dt^2/2, dt]]; z_k = x_k[0] + v, R = r. Kalman filter (Joseph form) and RTS: P_s = P_f + C (P_s' - P_p') C^T with P_s' <= P_p', so P_s <= P_f. NEES averaged over N runs ~ chi2(2N)/N.

### T058 — Track the exact observation frame and clock basis.

*Hypothesis.* Observations that carry frame id, clock id, epoch and time basis can be combined only after an explicit declared mapping, which is then applied exactly and recorded.

*Model.* Frame mapping p' = R p + t between frames of one kind (distances invariant, checked on tracker positions; distance-mode records are carried unchanged by design); clock mapping t' = rate t + offset from (clock, epoch, basis) to another; arrival -> acquisition by the declared latency.

### T059 — Test whether each observation can be retained without being admitted as state.

*Hypothesis.* Retention and admission are separate: any observation can be retained as evidence without changing state, and only a retained, validated, admitted observation bound by content digest can update state.

*Model.* State (mean, variance) per component; retain(o) stores (o, digest(o)) with state_admission = not_performed; admit(digest) validates mode and references; update applies K = P/(P + R), m' = m + K (z - m), P' = (1 - K) P only for admitted digests.

### T060 — Build a deterministic multi-sensor synthetic bench.

*Hypothesis.* A seeded, stream-separated generator can produce a planar multi-sensor bench whose truth, raw readings, noise draws and declared covariances are exactly reproducible and rate-exact.

*Model.* Truth x = (px, py, vx, vy), x_{k+1} = F x_k + w_k with the exact white-noise-acceleration discretization F = [[I, dt I], [0, I]], Q = q [[dt^3/3 I, dt^2/2 I], [dt^2/2 I, dt I]] (dt = 0.05 s, q = 0.05 m^2/s^3). Sensors: camera position (10 Hz, R = [[0.04, 0.012], [0.012, 0.04]] m^2), encoder speed |v| (20 Hz, 0.01 m^2/s^2), IMU heading rate as an integrating gyro wrap(theta_k - theta_{k-1})/dt (20 Hz, 4e-4 rad^2/s^2), tracker position (2 Hz, 0.0025 I m^2).

### T061 — Generate known ground truth and known covariance.

*Hypothesis.* The empirical residual and process-noise moments reproduce the declared means (zero) and covariances within a stated Monte Carlo bound, and the bound's power against a misstated covariance is quantified exactly rather than read off one run.

*Model.* For N i.i.d. N(0, R) residuals with known zero mean, S = sum r r^T / N has E S = R and Var S_ij = (R_ij^2 + R_ii R_jj)/N; the mean has Var = R_ii / N. 28 standardized moments are tested jointly with a Bonferroni bound at family error 1e-3. Power against a declared c R: N S_ii / R_ii ~ chi2(N), so the variance test flags with probability P(|chi2(N)/N - c| >= z_crit c sqrt(2/N)). Q(dt) = int_0^dt F(u) G q G^T F(u)^T du.

### T062 — Test independent-noise and correlated-noise cases.

*Hypothesis.* NEES/NIS consistency holds for a filter with the correct cross-correlated R and fails for one that drops the cross-covariance; the failure size is predictable in closed form.

*Model.* Camera and tracker positions z_c = p + e_m + e_c, z_t = p + e_m + e_t with Var e_c = Var e_t = 0.04 I, common mode Var e_m = 0.09 I: R_true = [[0.13 I, 0.09 I], [0.09 I, 0.13 I]]; the ignoring filter uses blockdiag(0.13 I, 0.13 I). CV motion dt = 0.1 s, q = 0.05. Mismatched moments: joint [x; x_hat] propagated exactly.

### T063 — Propagate covariance through frame transforms.

*Hypothesis.* For affine frame maps y = J x + b (rotation, translation, unit scale and their composite) the covariance transforms as J P J^T, translations do not change it, and Mahalanobis distance is invariant when vector and covariance are transformed together.

*Model.* State (x, y, vx, vy) with declared body covariance (std 0.30, 0.05 m, 0.10, 0.06 m/s, correlations 0.2-0.4). Rotation J = blockdiag(R, R) with R = R(35 deg); translation (12.5, -4.0) m acts on positions; unit change J = 1000 I; composite J = 1000 blockdiag(R, R). d^2 = e^T P^-1 e is invariant because (J e)^T (J P J^T)^-1 (J e) = e^T P^-1 e for invertible J.

### T064 — Propagate covariance through the Jacobi transfer matrix.

*Hypothesis.* A declared [lateral, heading] covariance propagates along a geodesic as Phi(s) P Phi(s)^T, collapses in the heading direction at a conjugate point, and the first-order propagation fails in a curvature-dependent way as the perturbation grows.

*Model.* Normal Jacobi field j'' + K j = 0; Phi(s) = [[j_lat, j_head], [j_lat', j_head']]. Unit sphere (K = 1) along the equator from (theta, phi) = (pi/2, 0); hyperbolic plane (K = -1, g = I/y^2) along x = 0 from (0, 1). P0: lateral std 0.002, heading std 0.02 rad, correlation 0.25. Exact heading-only laterals: asin(sin a sin s) (sphere), asinh(sin a sinh s) (hyperbolic); second-order relative variance error cos^2 s sigma^2 and cosh^2 s sigma^2.

### T065 — Model filter-induced correlation.

*Hypothesis.* Filtered estimates are correlated in time even when the sensor noise is white; the steady-state correlation is [(I - K H) F]^j P, and ignoring it underestimates the variance of averaged filter outputs.

*Model.* e_{k+1} = (I - K H) F e_k + (I - K H) w_k - K v_{k+1} in steady state gives Cov(e_{k+j}, e_k) = A^j P with A = (I - K H) F. Scalar random walk: q = 1, r = 30, M^2 = qM + qr -> M = 6, P = 5, K = 1/6, a = 5/6; Var(mean of n) = P/n^2 [n + 2 sum_j (n - j) a^j]. Innovations of the optimal filter are white.

### T066 — Verify that filtered residuals use filter covariance, not raw sensor covariance.

*Hypothesis.* Filter residuals must be normalized by the filter's own covariance: S for innovations and R - H P+ H^T for post-fit residuals (the two normalized statistics coincide). The raw sensor covariance R over-states the first and under-states the second by predictable amounts.

*Model.* nu = z - H x-, Cov nu = S = H P- H^T + R; r = z - H x+ = (I - H K) nu, Cov r = R S^-1 R = R - H P+ H^T. E[nu^T R^-1 nu] = tr(R^-1 S) > m; E[r^T R^-1 r] = tr(S^-1 R) < m. Planar CV, dt = 0.1 s, q = 0.5, camera R = [[0.04, 0.012], [0.012, 0.04]].

### T067 — Add Mahalanobis gating.

*Hypothesis.* A gate NIS <= chi2_m(p) on the correctly normalized innovation rejects valid readings at rate 1 - p when the decision does not feed back; closed-loop gating inflates the rate because a rejection preserves the large prior error that caused it.

*Model.* NIS = nu^T S^-1 nu ~ chi2(2), independent across ticks for the optimal filter (white innovations); rejections ~ Binomial(N, 1 - p). Gates: chi2_2(p) = -2 ln(1 - p).

### T068 — Add outlier rejection.

*Hypothesis.* A 99% chi-square gate on the correctly normalized NIS detects an outlier with probability 1 - F_ncx2(gate; b^T S^-1 b), removes gross outliers at ~1% false-alarm cost, cannot remove outliers comparable to sqrt(S), and can lock out valid data after an undetected outlier.

*Model.* Readings z = H x + v + o with o = b u (u uniform on the unit circle) at 5% of ticks; an outlier's NIS is noncentral chi2(2) with lambda = b^T S^-1 b when the prior error is N(0, P-). Under the broad prior P0 (0.5 m std) a 1.5 m first-reading outlier has small lambda and often passes.

### T069 — Add missing-data behavior.

*Hypothesis.* A missing reading is handled by prediction only: the covariance grows exactly by the model and the filter stays consistent; any substituted value (zero, hold-last) is refused because it fabricates information.

*Model.* P(k + n | k) = F(n dt) P F(n dt)^T + Q(n dt) with F(t) = [[I, t I], [0, I]], Q(t) = q [[t^3/3 I, t^2/2 I], [t^2/2 I, t I]]; the position variance gains q (n dt)^3 / 3 plus the propagated terms. dt = 0.1 s, q = 0.05.

### T070 — Add stale-clock behavior.

*Hypothesis.* An unmodelled clock offset biases innovations only when another sensor pins the true time; a mean test then detects it, and an offset state in the filter removes it.

*Model.* Camera reports z_k = p(t_k - tau) + v = p_k - tau v_k + eta + v with tau = dt = 0.1 s; tracker (2 Hz, R = 0.0025 I) reports p_k + v. Naive filter: z = p + v. Augmented EKF: state (p, v, tau), h = p - tau v, H = [I, -tau I, -v], R + q dt^3/3 I. Expected naive innovations follow from the same linear filter run on noise-free mean readings.

### T071 — Add frame mismatch behavior.

*Hypothesis.* A measurement expressed in a rotated frame inflates NIS when another sensor fixes the true frame; alone it is invisible to NIS; frame-id mismatches must therefore be refused at the API rather than left to statistical detection.

*Model.* Tracker readings R(theta)^T p + v with theta = 2 deg about the world origin; filter assumes p + v. The bias (R^T - I) p grows with |p|. Expected NIS/NEES per tick from the exact joint moments of [x; x_hat] (mismatch_moments).

### T072 — Add calibration-expiry behavior.

*Hypothesis.* A calibration record with a validity interval must gate fusion: readings outside it are retained for audit but never fused, and the refusal leaves the state exactly as if they had not arrived.

*Model.* CalibrationRecord covers ticks in [valid_from, valid_until); fuse() records first, then refuses with calibration_expired before any arithmetic. After expiry the state is predicted only, P(k + n) = F(n dt) P F(n dt)^T + Q(n dt).

### T073 — Add sensor-track-lost behavior.

*Hypothesis.* After a sensor stops, the covariance grows by the model, the track-lost tick is predictable exactly from a declared radius rule, a lost track is neither updated nor admitted (and a refused update has no side effects), and reacquisition is an explicit, refusable operation with an exact initial covariance.

*Model.* P(n) = F(n dt) P F(n dt)^T + Q(n dt); lost when sqrt(lambda_max(P_pos) chi2_2(0.99)) > 1 m. Two-point initialization p = z2, v = (z2 - z1)/dt has error covariance [[R, R/dt], [R/dt, 2R/dt^2 + q dt/3 I]], equal to J blockdiag(R, R, Q(dt)) J^T for the error map J. A coordinated turn at omega has linear transition T(t); with white acceleration through the turn, E[NEES] = tr(P_f^-1 (T P T^T + Q_turn(t))) + d^T P_f^-1 d with d = (T - F) x_hat.

### T074 — Compare fused state against analytic ground truth.

*Hypothesis.* For the linear-Gaussian bench the Kalman recursion is the exact posterior: its estimate and covariance equal the batch information-form posterior (by block elimination, and by a dense solve without recursion over time), and its errors against the simulated truth are NEES-consistent.

*Model.* Batch posterior over x_0..x_K minimizes |x_0 - mu0|^2_P0 + sum |x_k - F x_{k-1}|^2_Q + sum |z_k - H x_k|^2_R (normal equations, information matrix of size 4(K+1)); its marginal at K is the filtering posterior. Camera every tick, tracker every 5 ticks (stacked, R = blockdiag(R_camera, 0.0025 I)).

### T075 — Keep observation, candidate state, and admitted state as separate objects.

*Hypothesis.* Keeping observations, candidate states and admitted states as separate types, with a single explicit gate whose every check is load-bearing, guards against a candidate becoming state by accident, naive tampering or omission.

*Model.* Admission = ordered conjunction of declared checks (declared, writable, typed, finite, covariance, integrity, provenance, frame, fresh, track, uncertainty, innovation, calibration); fresh means the most recently issued candidate at the session tick; fail closed on any exception. Mutation m_i removes check i.

### T076 — Ensure fusion defaults to read-only and `not_performed`.

*Hypothesis.* The fusion API cannot estimate, fuse or admit anything unless a caller constructs a writable session, the flag cannot be flipped afterwards, and its authority record uses the CIW vocabulary (sensor_fusion and state_admission not_performed).

*Model.* Not numerical: a default-argument and refusal audit of FusionSession against ciw.declared_workload.AUTHORITY.

### T115 — Measure CPU energy per geodesic trajectory.

*Hypothesis.* The deterministic work of a fixed-step RK4 geodesic trajectory is exactly 4N right-hand-side evaluations; its CPU energy can be read only from package counters bracketing a batch, which an operator captures outside the lab runner.

*Model.* E_gross = sum over package domains of (E(after) - E(before)) / (repeats * trajectories), one wrap allowed; E_idle-subtracted = (E_gross_total - E_idle * t_work / t_idle) / (repeats * trajectories); work proxy W = 4N evaluations (RK4) or the Dormand-Prince count.

### T116 — Measure GPU energy per batch.

*Hypothesis.* Gross GPU-device energy per batch of the fixed binary64 Gaussian VI workload is measurable from NVML total-energy counter differences bracketing each batch.

*Model.* E_batch = (E(end of measurement) - E(start)) / n_batches, gross, background-inclusive, no idle subtraction; per-batch increments from the counter read after each batch.

### T117 — Compare Python, Rust, Julia, and GPU implementations.

*Hypothesis.* Implementations of the same RK4 geodesic kernel in Python and Rust agree to rounding error, and all agree with the exact great circle to the RK4 truncation error.

*Model.* Unit sphere, theta'' = sin cos phi'^2, phi'' = -2 cot(theta) theta' phi'; RK4 with h = L/N and the operation order of ciw.lab.integrators.step_rk4.

### T118 — Record RTX 2080 utilization, temperature, power, and kernel duration.

*Hypothesis.* During the T116 workload the RTX 2080 power draw is steady (coefficient of variation <= 0.10 over the measurement phase) and its temperature drifts by at most 5 C; kernel-only time is out of scope until an Nsight Systems report is ingested.

*Model.* Measurement-phase min/mean/max of NVML power, temperature and graphics clock; power CV = population standard deviation / mean; temperature drift = max - min; utilization from nvidia-smi rows whose UTC timestamps fall inside the measurement window.

### T119 — Measure energy per accepted numerical result.

*Hypothesis.* Energy per accepted result is well defined only when counter brackets are valid and every counted solve meets the declared accuracy target; otherwise it must be withheld. Its denominator must say whether it counts replica solves or distinct results.

*Model.* E_acc = Delta E_measurement / #{replica solves with KL(q || p) <= target}; replicas in a batch are bitwise-identical copies of one result (constant-row encoding), so E per distinct accepted result = Delta E / #{accepted batch outputs}; boundary-dependent (measurement phase only, gross, no idle subtraction).

### T120 — Compare precision versus energy cost.

*Hypothesis.* float32 RK4 matches float64 until truncation error reaches float32 roundoff, after which more steps cannot buy accuracy; the counted operations per step are precision-independent.

*Model.* Global error ~ C h^4 + c N u with u = 2^-24 (float32) or 2^-53 (float64); cost ~ 4 N right-hand-side evaluations of fixed counted arithmetic.

### T121 — Detect whether GPU reductions alter numerical conclusions.

*Hypothesis.* GPU-style reduction orders keep float32 sums within their order-specific a priori bounds, yet the order differences suffice to flip sign and pass/fail decisions near their boundary; float64 suffers the same with data scaled to its precision.

*Model.* |fl(sum x) - sum x| <= gamma_d sum|x_i|, gamma_d = d u / (1 - d u), where d is the largest number of additions any input passes through (n - 1 sequential, log2 n trees, block height plus partial fold otherwise); Kahan (2u + n u^2) sum|x| to leading order.

### T122 — Build a bounded free-energy-style computational example.

*Hypothesis.* Bounded natural-gradient/Euclidean descent on the Gaussian variational free energy reaches the exact posterior, with F + log Z = KL at every iterate; convergence speed depends on normalization.

*Model.* F(q) = E_q[-log p(y|x)] + KL(q || p(x)) = KL(q || p(x|y)) - log p(y); mean step m <- m - alpha (Lambda m - b), precision Q <- (1 - beta) Q + beta Lambda.

### T123 — Keep numerical free energy distinct from physical energy.

*Hypothesis.* Variational free energy (nats, information) and device energy (joules) are different dimensions; a typed check refuses to combine them, and CIW records already keep them in disjoint fields.

*Model.* Quantities carry a dimension vector over {information, energy, time, count}; power is energy/time; addition, subtraction and comparison (including equality) require equal vectors; multiplication and division combine them.

### T124 — Retain raw telemetry and device/runtime identity.

*Hypothesis.* Every retained energy log keeps raw timestamped counter readings with device and runtime identity fields, and the validator refuses edits to them unless the log is deliberately resealed.

*Model.* log_digest = sha256(canonical(log without digest)); structural profile of energy_records.validate_log; analysis eligibility rules of energy_records.analyze.

### T125 — Test replay of energy telemetry reports.

*Hypothesis.* Replaying a retained energy analysis through a ciw Session recomputes the same numerical result (stable numerical_result_id) under fresh execution and result identities.

*Model.* numerical_result_id = digest({operation_id, data = energy_records.analyze(log)}); execution and result identities include a fresh occurrence UUID.

### T126 — Design a flat-plate control experiment.

*Hypothesis.* On a flat plate the chord between markers equals their geodesic distance and the Jacobi transfer is [[1, s], [0, 1]], so the plate isolates instrument, frame and procedure error from curvature.

*Model.* Plane z = 0; geodesics are straight lines; j'' = 0 gives j_lat = 1, j_head = s; offset-path separation e(s) = delta + s dtheta.

### T127 — Design a rolled-cylinder control experiment.

*Hypothesis.* A rolled cylinder has zero Gaussian curvature, so its Jacobi transfer equals the plate's, while its extrinsic curvature makes marker chords shorter than geodesic distances by d^3/(24 R^2) + O(d^5).

*Model.* Cylinder X(phi, z) = (R cos phi, R sin phi, z), R = 100 mm; development (R phi, z) is an isometry; gap(d) = d - 2 R sin(d / 2R) for circumferential pairs; resolvable arc solves gap(d) = k sqrt(2) u.

### T128 — Design a curved coupon experiment.

*Hypothesis.* Across the dome, positive Gaussian curvature focuses laterally offset geodesic routes: the 2 mm offset route crosses the nominal route inside the coupon, a signature absent on the flat plate and on the cylinder.

*Model.* Coupon z = h exp(-(x^2 + y^2) / (2 sigma^2)), h = 10 mm, sigma = 20 mm; nominal route from (-60, 0) along +x to x = 140; separation = delta j_lat + dtheta j_head with j'' + K j = 0.

### T129 — Define a surface metrology protocol.

*Hypothesis.* The sampling needed to resolve curvature follows from two terms of a local quadratic fit: a bias set by the quartic profile term and a noise term that grows as the window shrinks; the window must be chosen from the actual profile, not from its osculating circle.

*Model.* Least-squares z = a + b x + c x^2 over a centred window W with spacing d: bias(2c) = (3/7) q W^2, std(2c) = sigma sqrt(720 d) W^(-5/2) (large-n); tolerance split half bias, half k = 2 noise. Registration: Kabsch; E[SSR] = sigma^2 (3N - 6); rotation covariance sigma^2 (sum |p|^2 I - p p^T)^-1.

### T130 — Define calibration artifacts and datum frames.

*Hypothesis.* A frame chain INSTRUMENT -> WORLD -> FIXTURE -> PART -> CAD with left-perturbation covariances propagates to point uncertainty by first-order adjoints, and the 3-2-1 datum and artifact fits have the linearized covariances their Jacobians predict.

*Model.* T_true = exp(xi) T; composition C = sum Ad(T_1..T_k-1) C_k Ad^T; point covariance [I, -(Tp)^] C [I, -(Tp)^]^T; 3-2-1 datum: A plane normal z, B direction x, origin on the A, B and C planes; geometric sphere fit; step gauge m = (1 + e) L + b.

### T131 — Define repeatability and Gage R&R procedures.

*Hypothesis.* The ANOVA method recovers the variance components of a balanced crossed study without bias (before truncation), and a single 10 x 3 x 3 study estimates %GRR only within a wide sampling interval.

*Model.* y_ijk = mu + P_i + O_j + (PO)_ij + e_ijk; expected mean squares give s_e^2 = MS_E, s_po^2 = (MS_PO - MS_E)/r, s_o^2 = (MS_O - MS_PO)/(p r), s_p^2 = (MS_P - MS_PO)/(o r); %GRR = 100 sqrt(GRR / total); ndc = 1.41 s_p / s_GRR.

### T132 — Model fibre/tape placement path tolerance.

*Hypothesis.* On a cylindrical mandrel geodesic placement paths are helices (straight in the development); a steered variable-angle course has geodesic curvature of magnitude theta' cos theta; lateral placement errors grow linearly (flat Jacobi transfer) and radius error enters through the machine-angle programming.

*Model.* Development (R phi, z); kappa_g = <u'' + Gamma(u', u'), N> / |u'|^2 with N the tangent rotated +90 deg (signed -theta'(z) cos theta for theta measured from the axis); kappa_n = II(u', u') / I(u', u'); e(s) = delta + R cos(theta) dphi + s (dpsi + sin(theta) cos(theta) dR / R); RSS with k = 2.

### T133 — Model winding path sensitivity.

*Hypothesis.* Geodesic winding on a torus conserves the Clairaut constant, so heading errors move the turnaround latitude predictably; because a heading error changes the Clairaut constant and with it the azimuth advance per period, the heading-error Jacobi field grows secularly (linearly) in both the librating and the circulating regime, at a rate set by d(advance)/dc; constant-angle winding is not geodesic on the torus and needs friction abs(kappa_g / kappa_n).

*Model.* Torus R = 150 mm, r = 50 mm, K = cos(theta) / (r (R + r cos theta)); Clairaut c = rho^2 dphi/ds; period P(c) and advance A(c) by quadrature of r dtheta / sqrt(1 - c^2 / rho^2); one-period monodromy M with trace 2 (Killing field), j_head(nP) = n M01, M01 = rho0^2 sin^2(psi) dA/dc; rate = |M01| / P; theta_turn = acos((|c| - R) / r); d theta_turn = rho0 sin(psi) dpsi / (r sin theta_turn); loxodrome dphi/ds = cos(psi)/rho, dtheta/ds = sin(psi)/r.

### T134 — Model coating or welding trajectory sensitivity.

*Hypothesis.* Along a trajectory across the dome, lateral registration errors propagate by the Jacobi transfer, produce a second-order standoff error -kappa e^2 / 2 and a first-order tilt kappa e, and the tool-centre-point path X + H n has speed factor |1 - H kappa_n| that vanishes where the standoff equals the concave radius.

*Model.* e(s) = delta j_lat + dtheta j_head; standoff error by ray casting along the programmed axis; TCP path P = X + H n with |P'| = sqrt((1 - H kappa_n)^2 + (H tau_g)^2).

### T135 — Model robotic inspection scan paths.

*Hypothesis.* Positive curvature focuses geodesic scan rows, so rows launched at the swath spacing cross near the dome and leave gaps; coverage must be bought with path length, and constant-y (chart-parallel) rows reach full coverage sooner on this coupon.

*Model.* Rows: geodesics launched along +x from the edge x = -60 mm at offsets k * spacing, or chart lines y = const; footprint = points within 10 mm (3D) of a row polyline; coverage = covered area / area estimated on Halton (2, 3) chart points weighted by sqrt(det g); path length = rows + edge transitions.

### T136 — Rank paths by calibration tolerance.

*Hypothesis.* The calibration tolerance a route requires is the inverse of its Jacobi sensitivity: routes whose heading field j_head grows least tolerate the largest heading calibration error; the first-order allocation must be derated where second-order terms push a tolerance corner past the spec.

*Model.* Lateral error e(s) = delta j_lat(s) + dtheta j_head(s) + O(2); first order delta_req = spec / (2 max|j_lat|), dtheta_req = spec / (2 max|j_head|); both derated by the worst exact corner ratio until all four corners are within 0.9999 of the spec; flat reference spec / (2 L).

### T137 — Rank paths by focus margin.

*Hypothesis.* The focus margin (distance to the nearest focal or conjugate point relative to route length) ranks routes differently from length: the straight route over the dome is the shortest candidate route to the far edge yet has a focal point inside it.

*Model.* Focal points: zeros of j_lat; conjugate points: zeros of j_head (s > 0); margin = s_focus / L, or horizon / L as a lower bound; L''(0) = j_head(L) j_head'(L) for routes from a point to a straight edge line.

### T138 — Compare predicted and measured path separation.

*Hypothesis.* The predicted separation of the 2 mm offset tape (and its crossing near s = 154 mm) can be compared with measurement by the normalized error E_n, provided the prediction uncertainty includes the realized start pose of the tape (budgeted open loop, or removed by conditioning on its measured start pose); the comparison is only meaningful against acquired hardware evidence.

*Model.* E_n = |m - p| / sqrt(U_m^2 + U_p^2), U = 2 u. Open loop: u_p^2 = u_geometry^2 + u_solver^2 + u_start^2 with u_start from the declared jig error (0.05 mm, 0.5 mrad) times central-difference start sensitivities of the exactly re-integrated offset route. Conditioned: p is re-integrated from the CMM start pose and u_start uses its estimate uncertainty (offset sqrt(2) u_cmm, heading 2 u_cmm / 20 mm).

### T139 — Retain raw measurements, calibration, and frame metadata.

*Hypothesis.* A retention record that binds raw-byte digests, calibration reference and validity window, frame chain with covariances and an explicit clock is sufficient to supply the acquisition fields of a hardware_measured finding, and every omission is refused.

*Model.* Record = {raw digests, instrument identity, calibration (applied/not_applied), frame chain links (R, t, C), clock}; identity = SHA-256 of canonical JSON; to_acquisition maps a valid measurement record to device / raw_sha256 (SHA-256 of the raw manifest) / acquired_at / calibration.

### T140 — Report whether uncertainty is instrument-, geometry-, or solver-limited.

*Hypothesis.* Each predicted quantity has a limiting uncertainty term; extrinsic chord-geodesic gaps are instrument-limited, path separations open loop are limited by the realized start pose (a CMM start-pose measurement removes that limit on the coupon but not over the 240 mm plate route, where the 20 mm heading baseline still dominates), and the location of a Jacobi focus is geometry-limited because it depends sensitively on the dome shape.

*Model.* u_c^2 = u_instrument^2 + u_geometry^2 + u_execution^2 + u_solver^2; u_geometry from central differences over rectangular tolerances (u = a / sqrt 3); u_execution = start-pose uncertainty times the Jacobi fields (declared jig error open loop, CMM estimate when conditioned); u_solver = |Q(h) - Q(2h)| / 15 for a reported h-solution and (16/15) |Q(h) - Q(2h)| when the reported value is the coarse one (the 6-step control); u_instrument(focal) = u_pair / |d sep / ds|; limiting term = variance share > 50%, otherwise mixed.

### T141 — Keep production acceptance outside the system until independently validated.

*Hypothesis.* Production acceptance is an authority decision outside the workbench: no evidence basis makes a claim filed in an authority domain established, and no policy call or protocol field can record a decision; a statement filed in a computational domain is caught only by a screen on its wording.

*Model.* Label function L(basis, domain) = not_established for every authority domain; AcceptancePolicy.decide always refuses; protocols require production_acceptance = outside_system and hypothesis-status criteria; the section screen refuses decision phrases (accepted, approved, rejected, scrapped, quarantined, signed off, dispositioned, released for or to production, passed or passes inspection or acceptance, certified for production) in claims and string values outside the authority domains.

## Results

| Task | Finding | Value | Evidence | Report |
| --- | --- | --- | --- | --- |
| T045 | Seven typed observation modes declare quantity, unit, frame kind, clock basis, geometry class, noise model and non-observables | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:57bd1d99bf97` |
| T045 | Validation refuses records lacking frame, clock, epoch, calibration or clock-basis references | (list of 9 entries; see the source report)  | `numerically_verified` | `sha256:57bd1d99bf97` |
| T045 | No observation mode stands in for another: every ordered substitution is refused | {"ordered_pairs": 42, "refused": 42}  | `numerically_verified` | `sha256:57bd1d99bf97` |
| T045 | A camera chord becomes a surface distance only through a declared surface model | 2.7755575615628914e-17 m | `numerically_verified` | `sha256:57bd1d99bf97` |
| T045 | The declared noise-model parameters describe real instruments of these modes | null  | `not_established` | `sha256:57bd1d99bf97` |
| T046 | Chord expansion c = s - kappa0^2 s^3/24 - kappa0 kappa0' s^4/24 + c5 s^5 + O(s^6) with c5 = (3 kappa0^4 + 8 kappa0^2 tau0^2 - 72 kappa0 kappa0'' - 64 kappa0'^2)/5760, which is kappa^4/1920 + kappa^2 tau^2/720 for constant curvature and torsion | (object of 4 entries; see the source report)  | `independently_verified` | `sha256:4f71c6c38986` |
| T046 | RK4 sphere geodesics reproduce the chord 2R sin(s/2R) and the s^3 coefficient kappa^2/24 | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:4f71c6c38986` |
| T046 | A torus geodesic with varying curvature keeps the s^4 term kappa0 kappa0'/24 at the start point | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:4f71c6c38986` |
| T046 | Evaluating the curvature at the arc midpoint removes the s^4 term on the same torus geodesic | {"midpoint_ratio": 3.4228989803754016e-07}  | `numerically_verified` | `sha256:4f71c6c38986` |
| T046 | Constant curvature alone does not give the circle chord: a cylinder helix with torsion deviates at order s^5 by kappa^2 tau^2/720 | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:4f71c6c38986` |
| T046 | The chord correction computed from nominal curvature holds for chords measured on a physical part | null  | `not_established` | `sha256:4f71c6c38986` |
| T047 | The exact helix chord expands as s - cos^4(alpha) s^3/(24 R^2) + (kappa^4/1920 + kappa^2 tau^2/720) s^5 with kappa = cos^2(alpha)/R, tau = sin(alpha)cos(alpha)/R | (object of 2 entries; see the source report)  | `independently_verified` | `sha256:2957ee22f42a` |
| T047 | Small-s fits of exact helix chords recover cos^4(alpha)/(24 R^2) at every angle and radius | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:2957ee22f42a` |
| T047 | Geodesics integrated on ciw.lab.surfaces.Cylinder reproduce the exact helix chord and coefficient | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:2957ee22f42a` |
| T047 | Axial rulings (alpha = 90 deg) have zero chord correction; circumferential paths have the largest coefficient 1/(24 R^2) | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:2957ee22f42a` |
| T047 | Zero Gaussian curvature does not make chord and geodesic distance equal | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:2957ee22f42a` |
| T047 | Chords measured between physical markers on a cylindrical part follow the cos^4(alpha)/(24 R^2) coefficient | null  | `not_established` | `sha256:2957ee22f42a` |
| T048 | Noise-free triangulation of the synthetic rig reproduces ground-truth marker chords | (object of 2 entries; see the source report) m | `numerically_verified` | `sha256:3e5c1827be97` |
| T048 | Using the camera chord as geodesic distance underestimates it by exactly s - c(s); the largest bias is on the circumferential helix | 0.007071505320992932 m | `numerically_verified` | `sha256:3e5c1827be97` |
| T048 | The declared cylinder model converts noise-free chords to arc lengths | 3.885780586188048e-16 m | `numerically_verified` | `sha256:3e5c1827be97` |
| T048 | Synthetic chord and model-converted arc RMS errors under 0.25 px Gaussian noise with integer rounding match first-order propagation | (object of 2 entries; see the source report) m | `numerically_verified` | `sha256:3e5c1827be97` |
| T048 | For the longest circumferential chord the substitution bias exceeds the 0.25 px noise RMS more than tenfold, while on rulings it vanishes | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:3e5c1827be97` |
| T048 | A physical stereo rig with this geometry achieves the synthetic chord accuracy | null  | `not_established` | `sha256:3e5c1827be97` |
| T049 | The finite-difference chord Jacobian matches the analytic focal-length and principal-point derivatives on a rectified rig | 1.0553356164047979e-08  | `numerically_verified` | `sha256:39d013e7e642` |
| T049 | First-order calibration-error prediction agrees with direct recomputation with a second-order residual | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:39d013e7e642` |
| T049 | Chord sensitivity to each calibration parameter (m per px, m per mrad); disparity-changing errors dominate | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:39d013e7e642` |
| T049 | A common focal-length error leaves same-depth chords unchanged and scales only the depth component | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:39d013e7e642` |
| T049 | The declared perturbation magnitudes bound the calibration error of a real stereo rig | null  | `not_established` | `sha256:39d013e7e642` |
| T050 | The implemented radial term displaces an on-axis image point by f (k1 r^3 + k2 r^5), as the Brown-Conrady model defines (implementation consistency check) | {"formula_relative_error": 9.763423403086335e-11}  | `numerically_verified` | `sha256:de16adc0d195` |
| T050 | Uncorrected radial distortion biases chords as J_pix delta_pix to first order: odd and linear in k1 and growing as r^2 with mean image radius | (object of 7 entries; see the source report)  | `numerically_verified` | `sha256:de16adc0d195` |
| T050 | Undistorting with the true Brown-Conrady model (radial and tangential) removes the chord bias | (object of 2 entries; see the source report) m | `numerically_verified` | `sha256:de16adc0d195` |
| T050 | Strong barrel distortion (k1 = -0.6) folds inside the image, so the radial model is not invertible there | (object of 7 entries; see the source report)  | `numerically_verified` | `sha256:de16adc0d195` |
| T050 | A two-term radial plus tangential Brown-Conrady model describes a real lens to the required accuracy | null  | `not_established` | `sha256:de16adc0d195` |
| T051 | Rounding plus Gaussian pixel noise has error variance sigma^2 + 1/12 px^2 under a uniform grid phase | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:905f120f7c44` |
| T051 | Without a random grid phase the quantization variance is not 1/12: an integer-aligned coordinate with sigma = 0.1 px has almost no rounding error | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:905f120f7c44` |
| T051 | Linear propagation J (sigma^2 + 1/12) J^T predicts the chord standard deviation | (object of 3 entries; see the source report) m | `numerically_verified` | `sha256:905f120f7c44` |
| T051 | Real image noise is Gaussian with sigma = 0.25 px and marker localization rounds to whole pixels | null  | `not_established` | `sha256:905f120f7c44` |
| T052 | Backlash error stays within [0, b] for a play operator of width b | (object of 3 entries; see the source report) m | `numerically_verified` | `sha256:122cf70b124a` |
| T052 | Backlash error changes only while the play is taken up after a direction reversal | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:122cf70b124a` |
| T052 | Least squares with a direction term recovers scale, bias and backlash | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:122cf70b124a` |
| T052 | Least squares that ignores backlash biases the offset estimate by the projection of the play error onto [x, 1] | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:122cf70b124a` |
| T052 | A real encoder drive train behaves as a constant-width play operator with constant scale and bias | null  | `not_established` | `sha256:122cf70b124a` |
| T053 | Single-axis gyro heading error has mean b t and variance N^2 t | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:757f3abc71ee` |
| T053 | Mean squared heading error is N^2 t + (b t)^2 | (object of 2 entries; see the source report) rad^2 | `numerically_verified` | `sha256:757f3abc71ee` |
| T053 | Strapdown bias error follows e_{k+1} = exp(-omega dt) e_k + J_r(omega dt) b dt | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:757f3abc71ee` |
| T053 | On a body rotating about z, transverse gyro bias produces a bounded orientation error 2 |b_perp| / |omega| instead of |b_perp| t | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:757f3abc71ee` |
| T053 | Angle random walk stays isotropic, N^2 t per body axis and E|e - m|^2 = 3 N^2 t, on stationary and rotating bodies | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:757f3abc71ee` |
| T053 | Real gyroscopes have constant bias and white rate noise with the declared densities | null  | `not_established` | `sha256:757f3abc71ee` |
| T054 | A clock offset delta produces error -v delta + O(delta^2) at the sample times | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:c6ec1f6480b8` |
| T054 | After linear interpolation to a common time the offset error is exactly -S delta (S the interpolant slope); its regression on -v delta is the velocity-weighted sinc^2(omega h/2) | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:c6ec1f6480b8` |
| T054 | Timestamp jitter adds mean squared error S^2 sigma_j^2 ((1 - w)^2 + w^2) | (object of 3 entries; see the source report) m^2 | `numerically_verified` | `sha256:c6ec1f6480b8` |
| T054 | A declared clock mapping removes the offset error; combining clocks without one is refused | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:c6ec1f6480b8` |
| T054 | Real sensor clocks have the declared constant offset and white jitter | null  | `not_established` | `sha256:c6ec1f6480b8` |
| T055 | Dropped observations are retained as explicit gaps at their sequence positions | {"dropped": 46, "position_mismatch": 0, "present": 154}  | `numerically_verified` | `sha256:881341edbc68` |
| T055 | A zero-filled stream and a value without a raw reference are refused | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:881341edbc68` |
| T055 | Zero-filling biases the mean by -p mu while explicit gaps leave it unbiased with variance sigma^2 E[1/N] | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:881341edbc68` |
| T055 | Values alone recognize zero fills only in favourable signals: a neighbour-median detector finds every fill 50 sigma from zero, but on a stationary quantized encoder axis a fill equals a genuine zero-count reading | (object of 7 entries; see the source report)  | `numerically_verified` | `sha256:881341edbc68` |
| T055 | Hold-estimate error follows q E[age] + r; bursts at the same drop rate raise it | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:881341edbc68` |
| T055 | Real links drop observations as Bernoulli or two-state burst processes with these rates | null  | `not_established` | `sha256:881341edbc68` |
| T056 | Observations older than the validity limit are flagged exactly by acquisition-time age | {"flag_mismatch": 0, "fresh": 192, "stale": 208}  | `numerically_verified` | `sha256:8b6b2e71a6b9` |
| T056 | Stale-state error equals velocity times age for constant velocity and to first order otherwise | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:8b6b2e71a6b9` |
| T056 | Age is refused without a declared latency for arrival stamps and for observations from the future | ["missing_latency", "future_observation"]  | `numerically_verified` | `sha256:8b6b2e71a6b9` |
| T056 | Staleness is decided by acquisition-time age, not arrival age: a record that arrived 5 ms before use with 35 ms latency is refused under a 30 ms limit | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:8b6b2e71a6b9` |
| T056 | Real tracker latencies and target speeds match the declared values | null  | `not_established` | `sha256:8b6b2e71a6b9` |
| T057 | The RTS smoothed covariance never exceeds the filtered covariance in matrix order | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:cd3af92c4407` |
| T057 | Ensemble position RMSE orders smoothed <= filtered <= raw | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:cd3af92c4407` |
| T057 | Filtered and smoothed NEES are chi-square consistent across the ensemble | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:cd3af92c4407` |
| T057 | The filter's predicted covariance reaches the discrete Riccati fixed point | (object of 2 entries; see the source report)  | `independently_verified` | `sha256:cd3af92c4407` |
| T057 | Wilson-Hilferty chi-square quantiles give the NEES consistency bounds | (object of 2 entries; see the source report)  | `independently_verified` | `sha256:cd3af92c4407` |
| T057 | Smoothing does not reduce the error of every individual sample | {"smoothed_worse_fraction": 0.28528333333333333}  | `numerically_verified` | `sha256:cd3af92c4407` |
| T057 | A real tracker's measurement noise and target motion match the constant-velocity model | null  | `not_established` | `sha256:cd3af92c4407` |
| T058 | Combining observations across frames, clocks, epochs or time bases without a declared mapping is refused | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:b3c02be99f2e` |
| T058 | A declared rigid frame mapping is applied exactly on dyadic inputs, preserves distances to rounding and is recorded on the observation | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:b3c02be99f2e` |
| T058 | Declared clock mappings (arrival to acquisition, then clock to clock) are applied exactly | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:b3c02be99f2e` |
| T058 | A mapping declared for another frame, frame kind or epoch is refused | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:b3c02be99f2e` |
| T058 | The declared frame and clock mappings equal the real extrinsic calibration and clock synchronization | null  | `not_established` | `sha256:b3c02be99f2e` |
| T059 | Every observation mode can be retained without changing estimator state; retained records carry state_admission not_performed | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:25c8a7ee7b08` |
| T059 | Updating state from a retained but unadmitted observation is refused for every mode | (object of 7 entries; see the source report)  | `numerically_verified` | `sha256:25c8a7ee7b08` |
| T059 | An admitted, digest-bound observation updates state by the exact Kalman formula | {"mean": 1.5, "variance": 0.25}  | `numerically_verified` | `sha256:25c8a7ee7b08` |
| T059 | Tampered, unretained and mode-substituted admissions are refused | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:25c8a7ee7b08` |
| T059 | Admission as workbench state confers authority to act on a machine | null  | `not_established` | `sha256:25c8a7ee7b08` |
| T060 | Regenerating the bench with the same seed reproduces the truth and all four sensor streams bit for bit, and a different seed changes every stream | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:4bc17c1b9ad6` |
| T060 | Every sensor reports exactly at its declared rate on the 20 Hz base clock over 20 s (camera 10 Hz, encoder 20 Hz, IMU 20 Hz, tracker 2 Hz) | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:4bc17c1b9ad6` |
| T060 | Every raw reading equals the noise-free sensor function of the retained truth plus its retained noise draw; h(truth) is recomputed by a separate code path (hypot and complex-angle forms for speed and heading rate, positions re-read) | 2.152097944296827e-14 max abs difference | `numerically_verified` | `sha256:4bc17c1b9ad6` |
| T060 | Changing the tracker rate from 2 Hz to 4 Hz leaves the truth and the camera, encoder and IMU streams unchanged (independent spawned PCG64 streams) | 0  | `numerically_verified` | `sha256:4bc17c1b9ad6` |
| T060 | Every declared sensor covariance is symmetric positive definite | 0.0004 smallest eigenvalue | `numerically_verified` | `sha256:4bc17c1b9ad6` |
| T060 | The bench's noise levels, rates and motion describe real camera, encoder, IMU or tracker hardware | "not established: every stream is a declared synthetic draw"  | `not_established` | `sha256:4bc17c1b9ad6` |
| T061 | Sample means and covariances of every sensor residual z - h(truth) and of the process noise x_{k+1} - F x_k agree with the declared covariances within a Bonferroni-corrected 99.9% Monte Carlo bound | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:82b11cb88d72` |
| T061 | At this sample size a 10% understatement of every declared variance is detected with probability above 0.999 per variance for the 10-20 Hz streams but only about half the time for the 2 Hz tracker (exact chi-square power); replicate tracker noise streams reproduce that power | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:82b11cb88d72` |
| T061 | The relative variance misstatement that one variance test flags with probability one half is z_crit sqrt(2 / N) to within 1% of the exact chi-square value for every stream | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:82b11cb88d72` |
| T061 | The declared process-noise covariance Q(dt) equals the continuous-time integral of F(u) G q G^T F(u)^T and composes as Q(a + b) = F(b) Q(a) F(b)^T + Q(b); the discrete white-noise-acceleration alternative fails the integral | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:82b11cb88d72` |
| T061 | Real sensors' noise covariance equals the covariance declared for this bench | "not established: residuals are synthetic draws f\u2026"  | `not_established` | `sha256:82b11cb88d72` |
| T062 | With the correct cross-correlated measurement covariance, run-averaged NEES and NIS lie inside their 99% chi-square intervals and the whitened innovations have identity covariance | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:cbdbf14b8b52` |
| T062 | Ignoring the camera-tracker cross-correlation makes the filter overconfident: run-averaged NEES exceeds the 99% upper bound at nearly every tick | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:cbdbf14b8b52` |
| T062 | The ignored-correlation filter still passes the mean-NIS test (grand mean near 4); only the full whitened-innovation covariance test exposes the missing cross-correlation | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:cbdbf14b8b52` |
| T062 | Monte Carlo grand-mean NEES and NIS of both filters agree with the exact second-moment prediction tr(P_f^-1 E[e e^T]) and tr(S_f^-1 E[nu nu^T]) within 4 run-level standard errors | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:cbdbf14b8b52` |
| T062 | Real camera and tracker noises share the common-mode covariance assumed here | "not established: the cross-correlation is a decl\u2026"  | `not_established` | `sha256:cbdbf14b8b52` |
| T063 | Monte Carlo covariances of rotated, translated, metre-to-millimetre and composite-transformed states match P' = J P J^T within a Bonferroni-corrected 99.9% sampling bound | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:6bf922a3bca9` |
| T063 | Roundoff sanity: the sample covariance of transformed samples equals J S J^T and J^-1 (J P J^T) J^-T returns P to roundoff (algebraic identities that check the arithmetic, not the propagation law) | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:6bf922a3bca9` |
| T063 | Mahalanobis distance is invariant under every transform, including the metre-to-millimetre unit change, when the covariance is transformed with the vector | (object of 4 entries; see the source report) max relative difference | `numerically_verified` | `sha256:6bf922a3bca9` |
| T063 | FrameTransform.apply moves a typed position observation to value R z + t and covariance R Sigma R^T, matching J P J^T to roundoff | 6.938893903907228e-18 max abs difference | `numerically_verified` | `sha256:6bf922a3bca9` |
| T063 | Rotating a position measurement by 35 degrees without rotating its covariance inflates the mean Mahalanobis distance to tr(P^-1 R P R^T) and multiplies the 99% gate rejection rate | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:6bf922a3bca9` |
| T063 | Converting a state to millimetres while keeping its covariance in metres multiplies every squared Mahalanobis distance by exactly 10^6 | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:6bf922a3bca9` |
| T063 | The declared 35 degree rotation, translation and body covariance describe a real sensor mounting or extrinsic calibration | "not established: the transform and covariance ar\u2026"  | `not_established` | `sha256:6bf922a3bca9` |
| T064 | The Jacobi transfer matrix integrated by ciw.lab.jacobi.transfer matches the constant-curvature closed forms (cos s, sin s; cosh s, sinh s) with unit Wronskian on both surfaces | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:f816bbb8283f` |
| T064 | Phi P Phi^T predicts the Monte Carlo covariance of (lateral offset, lateral rate) over perturbed geodesics on the sphere and the hyperbolic plane within a Bonferroni 99.9% bound | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:f816bbb8283f` |
| T064 | Lateral variance collapses at the sphere's conjugate point s = pi: the heading contribution vanishes because j_head(pi) = 0, leaving only the initial lateral variance | (object of 7 entries; see the source report)  | `numerically_verified` | `sha256:f816bbb8283f` |
| T064 | On the hyperbolic plane (K = -1) there is no conjugate point and the lateral variance grows as the closed form cosh^2 s sigma_l^2 + 2 cosh s sinh s c + sinh^2 s sigma_h^2 | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:f816bbb8283f` |
| T064 | The first-order variance sigma^2 j_head^2 overestimates the exact lateral variance by a relative error that grows like cos^2(s) sigma^2 on the sphere and cosh^2(s) sigma^2 on the hyperbolic plane; integrated geodesics agree with the exact nonlinear expectation | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:f816bbb8283f` |
| T064 | The heading standard deviation at which the first-order variance is 10% too large is about ten times smaller on the hyperbolic plane at s = 3 than on the sphere near its conjugate point (s = 0.9 pi) | (object of 2 entries; see the source report) rad | `numerically_verified` | `sha256:f816bbb8283f` |
| T064 | The declared [lateral, heading] covariance and these surfaces predict the path uncertainty of a real vehicle or tool on a real curved part | "not established: normalized synthetic surfaces a\u2026"  | `not_established` | `sha256:f816bbb8283f` |
| T065 | The scalar random-walk filter with q = 1, r = 30 has the exact rational steady state M = 6, P = 5, K = 1/6 and error autocovariance Cov(e_{k+j}, e_k) = (5/6)^j P | (object of 9 entries; see the source report)  | `numerically_verified` | `sha256:299d4f5a9e54` |
| T065 | Monte Carlo error autocovariances of the scalar filter match a^j P at lags 0-8, while its innovations are white | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:299d4f5a9e54` |
| T065 | On the planar constant-velocity bench the lag-j cross-covariance of filtered errors equals [(I - K H) F]^j P_ss for j = 0-5 | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:299d4f5a9e54` |
| T065 | Treating 20 successive filtered outputs as independent underestimates the variance of their average by the exact factor V_20 / (P/20), and a naive 95% interval covers far less often | (object of 7 entries; see the source report)  | `numerically_verified` | `sha256:299d4f5a9e54` |
| T065 | A real sensor's internally filtered output can be fused downstream as white noise | "not established: the correlation shown is that o\u2026"  | `not_established` | `sha256:299d4f5a9e54` |
| T066 | Innovations normalized by S = H P- H^T + R are chi-square(2) consistent: run-averaged NIS lies inside its per-tick 99% interval at 90% or more of ticks and the grand mean matches 2 | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:9927c71324a8` |
| T066 | Post-fit residuals z - H x+ have covariance R - H P+ H^T = R S^-1 R, and normalizing them by it reproduces the innovation NIS exactly, so the post-fit test carries no information beyond the innovation test | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:9927c71324a8` |
| T066 | Normalizing innovations by the raw sensor covariance R inflates NIS to tr(R^-1 S) and fails the chi-square test at nearly every tick | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:9927c71324a8` |
| T066 | Normalizing post-fit residuals by the raw R deflates them to tr(S^-1 R) < 2, which would hide an inconsistent filter | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:9927c71324a8` |
| T066 | A real residual monitor normalized by datasheet sensor covariance is correctly calibrated | "not established: synthetic Gaussian bench only"  | `not_established` | `sha256:9927c71324a8` |
| T067 | Open loop, the false-rejection rate of a gate at the chi-square(2) p-quantile of the correctly normalized NIS matches 1 - p within a 99.9% Wilson interval for p = 0.9, 0.99 and 0.999 | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:9baf6ae7f888` |
| T067 | The chi-square quantiles used by the gate invert the closed-form chi-square CDF to roundoff for 1-6 degrees of freedom | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:9baf6ae7f888` |
| T067 | Closed loop, a gated filter rejects valid readings more often than 1 - p at p = 0.9: a rejected reading signals a large prior error that the filter then keeps | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:9baf6ae7f888` |
| T067 | Gating the NIS computed with the raw sensor covariance R at the 99% quantile rejects valid readings at more than twice the nominal 1% rate | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:9baf6ae7f888` |
| T067 | A real gate at the 99% quantile rejects 1% of valid real readings | "not established: real noise may be heavy-tailed,\u2026"  | `not_established` | `sha256:9baf6ae7f888` |
| T068 | Gross 1.5 m outliers are detected at the rate predicted by the noncentral chi-square(2) law with lambda = b^T S^-1 b at each outlier's own prior covariance | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:9fd9ce631ce3` |
| T068 | Over all runs with gross outliers, gating lowers the mean squared error in most runs (sign test) but its mean improvement over fusing every reading is not statistically significant, because the cold-start lock-out runs lose heavily | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:9fd9ce631ce3` |
| T068 | Post hoc, conditioning on the outcome: after removing the lock-out runs (selected by the gated filter's own failure, which favours gating by construction) the gated RMSE is within 5% of the oracle that knows which readings are bad, while fusing every reading is at least 30% worse | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:9fd9ce631ce3` |
| T068 | Cold-start lock-out: a gross outlier in the first reading passes the gate under the broad prior, and the corrupted state then rejects runs of valid readings; every lock-out run starts this way | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:9fd9ce631ce3` |
| T068 | On clean data the 99% gate raises false alarms at about 1% and increases the mean squared error: the rejected valid readings are the ones that would have corrected a large prior error | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:9fd9ce631ce3` |
| T068 | Subtle 0.3 m outliers pass the gate at the predicted low detection rate; for them gating costs more accuracy than the outliers do | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:9fd9ce631ce3` |
| T068 | The vectorized and scalar noncentral chi-square CDFs agree | 1.1102230246251565e-16  | `numerically_verified` | `sha256:9fd9ce631ce3` |
| T068 | Real outliers are rare, isolated and of fixed magnitude as in this contamination model | "not established: the contamination model is decl\u2026"  | `not_established` | `sha256:9fd9ce631ce3` |
| T069 | Prediction-only steps grow the covariance exactly as F(n dt) P F(n dt)^T + Q(n dt), identically in the session API and the batch schedule | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:fc2d2a8b0dff` |
| T069 | With a 30-tick gap and 20% dropout handled by prediction only, NEES stays chi-square consistent and the error covariance at the end of the gap equals the grown covariance | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:fc2d2a8b0dff` |
| T069 | The session refuses requested substitution strategies (zero_fill, hold_last), NaN and absent observation values and a non-integer prediction tick, and leaves its state unchanged by the refusals | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:fc2d2a8b0dff` |
| T069 | Zero-filling missing readings by hand drags the estimate toward the origin and destroys consistency by the amount the exact joint moments predict | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:fc2d2a8b0dff` |
| T069 | Real sensor dropouts are independent of the state and of the noise, as assumed here | "not established: the dropout pattern is declared\u2026"  | `not_established` | `sha256:fc2d2a8b0dff` |
| T070 | With a correctly clocked tracker fused alongside, an unmodelled one-tick camera lag biases the whitened innovations of both sensors; a mean test detects it and the bias matches the exact linear prediction within sampling error | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:edd0c188bfc2` |
| T070 | A filter that estimates the clock offset as a state (h = p - tau v) removes the innovation bias and recovers tau = 0.1 s | (object of 7 entries; see the source report)  | `numerically_verified` | `sha256:edd0c188bfc2` |
| T070 | With the stale camera as the only position sensor the lag is invisible to the innovations: the lagged constant-velocity path is itself a constant-velocity path, so the estimate is biased by about -tau E[v] while the mean test passes | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:edd0c188bfc2` |
| T070 | The recovered offset calibrates the clock of a real camera | "not established: the lag is a declared synthetic fault"  | `not_established` | `sha256:edd0c188bfc2` |
| T071 | Fusing a tracker expressed in a frame rotated by 2 degrees with a world-frame camera inflates NIS as the target moves away from the rotation centre, matching the exact mismatched-filter moments within sampling error | (object of 9 entries; see the source report)  | `numerically_verified` | `sha256:ef3b37b040ea` |
| T071 | Near the rotation centre (ticks 1-20) the same mismatch goes undetected by the per-tick NIS test at this sample size; its predicted inflation there is only about 2% | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:ef3b37b040ea` |
| T071 | A rotated sensor fused alone keeps NIS consistent (its readings form a rotated constant-velocity path) while NEES grows: the estimate is confidently in the wrong frame | (object of 9 entries; see the source report)  | `numerically_verified` | `sha256:ef3b37b040ea` |
| T071 | Applying the declared FrameTransform before fusion restores chi-square consistency | (object of 9 entries; see the source report)  | `numerically_verified` | `sha256:ef3b37b040ea` |
| T071 | The session refuses an observation whose frame id differs from its own, and a transform refuses an observation from another source frame; the refused observation is retained in the log | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:ef3b37b040ea` |
| T071 | A 2 degree rotation is the size of a real extrinsic calibration error | "not established: the rotation is a declared synt\u2026"  | `not_established` | `sha256:ef3b37b040ea` |
| T072 | Readings inside the half-open validity interval [0, 60) are fused and every later reading is refused with calibration_expired, exactly at the boundary | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:7bcf9627865a` |
| T072 | Refused readings are retained with their digests and dispositions, and they do not change the state: the session equals a shadow session that never saw them, bit for bit | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:7bcf9627865a` |
| T072 | Revoked, unknown, other-sensor and other-frame calibrations are refused, and the expired record stays expired after a renewal is registered | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:7bcf9627865a` |
| T072 | Under a declared post-expiry drift of 1 cm per tick, refusing expired readings keeps NEES consistent (with a growing covariance) while fusing them makes the filter overconfident by the amount the exact joint moments predict | (object of 8 entries; see the source report)  | `numerically_verified` | `sha256:7bcf9627865a` |
| T072 | The validity interval [0, 60) reflects how long a real camera calibration stays valid | "not established: the record and the post-expiry \u2026"  | `not_established` | `sha256:7bcf9627865a` |
| T073 | The track is declared lost at exactly the tick at which the 99% position ellipse, grown in closed form from the last update, first exceeds the 1 m radius | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:0f6d22593da1` |
| T073 | A lost track is not admissible and cannot be updated, and the refused update leaves the state, covariance and clock unchanged; reacquisition is explicit, needs two consecutive readings no older than the session clock, and is refused while tracking | (object of 9 entries; see the source report)  | `numerically_verified` | `sha256:0f6d22593da1` |
| T073 | The two-point reacquisition covariance [[R, R/dt], [R/dt, 2R/dt^2 + q dt/3 I]] is exact for the constant-velocity truth | (object of 7 entries; see the source report)  | `numerically_verified` | `sha256:0f6d22593da1` |
| T073 | Under an unmodelled 0.2 rad/s turn during the gap, the expected NEES of the coasting track exceeds the 99% chi-square(4) quantile after a finite gap even though its covariance keeps growing; a radius rule protects against this only if its radius is below the ellipse radius reached by then | (object of 7 entries; see the source report)  | `numerically_verified` | `sha256:0f6d22593da1` |
| T073 | A 1 m, 99% track-loss radius is a safe operating threshold | "not established: the threshold is a declared policy value"  | `not_established` | `sha256:0f6d22593da1` |
| T074 | The recursive covariance-form Kalman estimate and covariance equal the batch information-form posterior marginal at K = 10, 40 and 100 ticks to near roundoff, by block elimination and, at K = 10 and 40, by a dense solve with no recursion over time | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:d4776e6daa68` |
| T074 | In exact rational arithmetic the scalar filter's final mean and variance are identical to the batch posterior | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:d4776e6daa68` |
| T074 | The fused estimate's error against the simulated truth is NEES-consistent: run-averaged NEES lies inside its per-tick 99% interval at 90% or more of ticks and the grand mean matches 4 | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:d4776e6daa68` |
| T074 | The fused estimate equals the physical state of a real target within its covariance | "not established: agreement is with the declared \u2026"  | `not_established` | `sha256:d4776e6daa68` |
| T075 | Every admission check refuses the candidates built to violate it, with that check's own refusal code, including a candidate superseded by a second update at the same tick | (object of 14 entries; see the source report)  | `numerically_verified` | `sha256:704ed0c5f054` |
| T075 | Mutation analysis: deleting the check a scenario targets changes that scenario's outcome for every scenario, and the violating candidate is then admitted except in two scenarios where a later check still refuses it (a missing declaration fails the innovation check closed; a forged candidate is not the latest issue, so the freshness check refuses it) | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:704ed0c5f054` |
| T075 | Observations, candidates and admitted states are unrelated types; candidates are never admitted automatically; an admitted state exists only through the gate and is immutable | (object of 9 entries; see the source report)  | `numerically_verified` | `sha256:704ed0c5f054` |
| T075 | An admitted synthetic state may command actuators | "not established: admission is a software gate ov\u2026"  | `not_established` | `sha256:704ed0c5f054` |
| T076 | The CIW authority vocabulary declares state_admission and sensor_fusion not_performed and physical_truth not_established, and the fusion session's default authority equals it | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:a7bb5f730b52` |
| T076 | FusionSession defaults to read-only: every estimation or admission call (initialize, predict, handle_gap, fuse, reacquire, admit) is refused with read_only_session, observations are still retained, and no state or admitted state exists | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:a7bb5f730b52` |
| T076 | The read-only flag and the authority record are fixed at construction: rebinding either is refused with read_only_session and the authority record cannot be edited in place | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:a7bb5f730b52` |
| T076 | Enabling fusion explicitly at construction changes the authority only to synthetic_only; physical truth stays not_established | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:a7bb5f730b52` |
| T076 | Synthetic fusion output is admissible as production state | "not established: sensor_fusion and state_admissi\u2026"  | `not_established` | `sha256:a7bb5f730b52` |
| T115 | Fixed-step RK4 spends exactly 4N = 1024 right-hand-side evaluations per trajectory (N = 256) | [1024, 1024, 1024, 1024, 1024, 1024] evaluations/trajectory | `numerically_verified` | `sha256:72167253de88` |
| T115 | The fixed-step trajectories are accepted results: embedded endpoint error against the exact great circle is below 1e-8 | 1.087370094109951e-09 normalized length | `numerically_verified` | `sha256:72167253de88` |
| T115 | Adaptive Dormand-Prince 5(4) (rtol 1e-9) reaches every great-circle endpoint within 1e-7 with these right-hand-side evaluation counts per trajectory | [487, 427, 361, 337, 313, 295] evaluations/trajectory | `numerically_verified` | `sha256:72167253de88` |
| T115 | Gross CPU package energy (background-inclusive, idle not subtracted) per geodesic trajectory | null J/trajectory | `not_established` | `sha256:72167253de88` |
| T115 | Idle-subtracted CPU package energy per geodesic trajectory (equal-length idle bracket after the workload) | null J/trajectory | `not_established` | `sha256:72167253de88` |
| T116 | GPU-domain gross energy per measured batch | null J/batch | `not_established` | `sha256:c390515aa2d0` |
| T116 | The NVML total-energy counter of the RTX 2080 has a characterized accuracy and resolution | null  | `not_established` | `sha256:c390515aa2d0` |
| T117 | Rust and Python closed-form RK4 kernels agree on every endpoint to 1e-12 | 0.0 radians or radians per unit length | `numerically_verified` | `sha256:c05c0e203f79` |
| T117 | Rust kernel endpoint error against the exact great circle is below 1e-8 | 1.0873703982101779e-09 normalized length | `numerically_verified` | `sha256:c05c0e203f79` |
| T117 | The compiled Rust kernel refuses malformed and nonfinite inputs | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:c05c0e203f79` |
| T117 | Generic Christoffel-symbol RK4 (ciw.lab.surfaces) and the closed-form kernel agree to 1e-12 | 8.881784197001252e-16 radians or radians per unit length | `numerically_verified` | `sha256:c05c0e203f79` |
| T117 | Declaring the Rust kernel an independent check of the Python kernel is refused (both are ciw code) | {"checker_origin": "ciw", "producer_origin": "ciw"}  | `numerically_verified` | `sha256:c05c0e203f79` |
| T117 | A Julia implementation agrees with the Python kernel | null  | `not_established` | `sha256:c05c0e203f79` |
| T117 | A GPU implementation agrees with the Python kernel | null  | `not_established` | `sha256:c05c0e203f79` |
| T117 | The Rust kernel uses less energy per trajectory than the Python kernel on real hardware | null  | `not_established` | `sha256:c05c0e203f79` |
| T118 | RTX 2080 power draw during the measurement phase (NVML) | null W | `not_established` | `sha256:94392e83a4be` |
| T118 | RTX 2080 temperature during the measurement phase (NVML) | null C | `not_established` | `sha256:94392e83a4be` |
| T118 | RTX 2080 graphics clock during the measurement phase (NVML) | null MHz | `not_established` | `sha256:94392e83a4be` |
| T118 | Host-bracketed batch solve duration (launch, sync and copy included) | null ms | `not_established` | `sha256:94392e83a4be` |
| T118 | RTX 2080 GPU utilization during the measurement phase (nvidia-smi rows inside the measurement window) | null % | `not_established` | `sha256:94392e83a4be` |
| T118 | RTX 2080 power draw is steady over the measurement phase (coefficient of variation <= 0.10) | null ratio | `not_established` | `sha256:94392e83a4be` |
| T118 | RTX 2080 temperature drifts by at most 5 C over the measurement phase | null C | `not_established` | `sha256:94392e83a4be` |
| T118 | RTX 2080 kernel-only duration of the Gaussian VI kernel | null ms | `not_established` | `sha256:94392e83a4be` |
| T119 | Energy per accepted replica solve of the synthetic baseline fixture, recomputed from raw counter readings, directly decoded outputs and a textbook Gaussian KL, equals the CIW analysis value | 0.05 J per accepted replica solve (synthetic fixture values) | `numerically_verified` | `sha256:b9c9b44d7428` |
| T119 | Energy per distinct accepted result (one batch output; its replicas are bitwise copies) of the synthetic baseline fixture | 0.2 J per distinct accepted result (synthetic fixture values) | `numerically_verified` | `sha256:b9c9b44d7428` |
| T119 | The metric is withheld for the reset, missing-bracket and under-target fixtures | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:b9c9b44d7428` |
| T119 | Dividing gross energy by executed solves reports a finite energy per result for the under-target fixture although no result is accepted | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:b9c9b44d7428` |
| T119 | Widening the boundary from the measurement phase to the whole run multiplies the baseline energy per accepted result by 7 | 6.999999999999999 ratio | `numerically_verified` | `sha256:b9c9b44d7428` |
| T119 | Physical GPU energy per accepted numerical result | null J/accepted solve | `not_established` | `sha256:b9c9b44d7428` |
| T120 | float64 RK4 endpoint error converges at order 4 on the sphere geodesics (N = 16..256) | 3.9632677890817516 order | `numerically_verified` | `sha256:a619044e6d20` |
| T120 | float32 RK4 endpoint error stops improving: its minimum is a truncation/roundoff crossover inside the step grid, followed by a roundoff-dominated plateau above it | (object of 5 entries; see the source report) normalized length | `numerically_verified` | `sha256:a619044e6d20` |
| T120 | Smallest grid N (>= 16) meeting each accuracy target, by precision (None: not reached for N <= 2048; 16 is censored at the grid minimum) | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:a619044e6d20` |
| T120 | Lowering precision to float32 cannot reach a 1e-7 endpoint accuracy at any step count up to 2048, while float64 reaches it | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:a619044e6d20` |
| T120 | Per RK4 step the kernel performs the same counted arithmetic in both precisions (the float32 and float64 transcendental implementations differ) | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:a619044e6d20` |
| T120 | float32 lowers the energy per accepted trajectory relative to float64 on real CPU or GPU hardware | null J/trajectory | `not_established` | `sha256:a619044e6d20` |
| T121 | Every emulated float32 summation order of the positive dataset stays within its order-specific a priori error bound; relative spread across orders | (object of 3 entries; see the source report) relative | `numerically_verified` | `sha256:3dcc7e46790b` |
| T121 | Every emulated float64 summation order of the same data stays within its order-specific bound; relative spread | (object of 3 entries; see the source report) relative | `numerically_verified` | `sha256:3dcc7e46790b` |
| T121 | Reduction order alone flips the sign of a float32 sum whose exact value is +0.25 (the one-thread sequential CPU fold against the GPU-style orders) | (object of 14 entries; see the source report)  | `numerically_verified` | `sha256:3dcc7e46790b` |
| T121 | float64 accumulation of these float32-valued cancellation inputs is exact in every emulated order (all errors 0), hence sign-correct | {"exact_zero_errors": 14, "max_abs_error": 0.0}  | `numerically_verified` | `sha256:3dcc7e46790b` |
| T121 | Reduction order alone flips the sign of a float64 sum of float64-native cancellation data whose exact value is +0.25 (the sequential fold against the GPU-style orders) | (object of 14 entries; see the source report)  | `numerically_verified` | `sha256:3dcc7e46790b` |
| T121 | Atomic completion order alone changes a float32 pass/fail test |S - 0.25| <= 0.01 on identical inputs | (object of 8 entries; see the source report)  | `numerically_verified` | `sha256:3dcc7e46790b` |
| T121 | Emulated atomicAdd completion orders of identical float32 block partials give distinct sums | 4 distinct results | `numerically_verified` | `sha256:3dcc7e46790b` |
| T121 | The bound-guarded sign test (decide only when |S| exceeds the order's own a priori bound) cannot contradict across orders but leaves both cancellation sums undecided in every emulated order | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:3dcc7e46790b` |
| T121 | Max reductions of this NaN-free, zero-free positive data are bitwise order-invariant over 8 permutations | 1 distinct results | `numerically_verified` | `sha256:3dcc7e46790b` |
| T121 | The IEEE maximum of +0.0 and -0.0 depends on operand order (numpy max) | [true, false]  | `numerically_verified` | `sha256:3dcc7e46790b` |
| T121 | Reductions on the RTX 2080 (CUB, cuBLAS or atomicAdd) reproduce these emulated spreads and sign flips | null  | `not_established` | `sha256:3dcc7e46790b` |
| T122 | The variational free-energy identity F + log Z = KL(q || p) holds at every iterate to 1e-10 nats | 2.842170943040401e-14 nat | `numerically_verified` | `sha256:bc6befd35e86` |
| T122 | KL to the exact posterior decreases monotonically and ends below 1e-12 nats within the 512-iteration bound | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:bc6befd35e86` |
| T122 | The raw-unit posterior and log evidence are invariant under the choice of normalization scales | 1.7763568394002505e-15  | `numerically_verified` | `sha256:bc6befd35e86` |
| T122 | A mean step 1.2 times the stability bound makes KL grow instead of converge | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:bc6befd35e86` |
| T122 | Unit scales leave the same problem unconverged after 512 iterations (condition number 1.3e3) | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:bc6befd35e86` |
| T122 | The variational free energy of this model equals a thermodynamic free energy of a physical system | null nat | `not_established` | `sha256:bc6befd35e86` |
| T123 | Typed arithmetic refuses to add, compare (including ==) or convert nats of variational free energy with joules, watts with joules, and quantities with untyped numbers | (object of 7 entries; see the source report)  | `numerically_verified` | `sha256:0e82ed7172df` |
| T123 | Typed arithmetic within one dimension agrees with hand-converted values to binary64 rounding: 0.2 J + 100 mJ = 0.3 J, 1000 mW = 1 W, 1 bit = ln 2 nat, 1 J == 1000 mJ, F + log Z - KL = 0 nat | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:0e82ed7172df` |
| T123 | CIW energy records and free-energy records keep joules, watts and nats in distinct unit-bearing fields | (object of 5 entries; see the source report)  | `numerically_verified` | `sha256:0e82ed7172df` |
| T123 | An untyped sum of free energy and physical energy changes when the energy unit changes | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:0e82ed7172df` |
| T123 | A decrease of variational free energy corresponds to a decrease of physical energy consumed by the computation | null  | `not_established` | `sha256:0e82ed7172df` |
| T124 | Every fixture retains raw timestamped counter readings and the 14 device/runtime identity fields (placeholder values in these synthetic fixtures) | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:225ddebda2cb` |
| T124 | The energy-log validator refuses 9 classes of tampering | (list of 9 entries; see the source report)  | `numerically_verified` | `sha256:225ddebda2cb` |
| T124 | A log whose counter readings were doubled and then resealed passes validation | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:225ddebda2cb` |
| T124 | Relabelling a synthetic fixture as physical_measurement makes its analysis eligible for physical comparison | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:225ddebda2cb` |
| T124 | A resealed sensor/workload UUID mismatch validates but is withheld by the analysis | ["workload_sensor_device_mismatch"]  | `numerically_verified` | `sha256:225ddebda2cb` |
| T124 | The fixtures' counter readings were produced by a physical GPU and NVML counter | null  | `not_established` | `sha256:225ddebda2cb` |
| T125 | Session replay keeps numerical_result_id identical across original, reproduction and replay for every fixture | {"baseline": 1, "missing": 1, "reset": 1, "under-target": 1}  | `numerically_verified` | `sha256:63e1e29735a3` |
| T125 | Each replay is a fresh analysis occurrence with new execution, result and bundle identities | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:63e1e29735a3` |
| T125 | Replay receipts record a numerical match, no state admission, a non-independent method and no fresh hardware measurement | ["fresh_analysis_of_same_retained_measurement"]  | `numerically_verified` | `sha256:63e1e29735a3` |
| T125 | Saving and restoring the Session workspace reproduces the retained workbench exactly | {"bundles": 8, "restored_equal": true}  | `numerically_verified` | `sha256:63e1e29735a3` |
| T125 | Replaying a retained bundle whose numerical result was edited is refused, with or without resealed envelopes | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:63e1e29735a3` |
| T125 | Replayed energy values are physically valid measurements | null  | `not_established` | `sha256:63e1e29735a3` |
| T126 | Flat-plate control: chord and geodesic marker distances coincide | 2.842170943040401e-14 mm | `numerically_verified` | `sha256:1448810e0f8a` |
| T126 | Flat-plate Jacobi transfer is [[1, s], [0, 1]] along the 240 mm control path | [[1.0, 240.0], [0.0, 1.0]]  | `numerically_verified` | `sha256:1448810e0f8a` |
| T126 | Flat-plate protocol record validates and refuses malformed variants | 9 refused mutations | `numerically_verified` | `sha256:1448810e0f8a` |
| T126 | Measured marker chords on the physical plate equal the predicted geodesic distances within instrument uncertainty | null  | `not_established` | `sha256:1448810e0f8a` |
| T126 | The declared instrument uncertainties (camera 0.02 mm, tracker 0.015 mm, CMM 0.002 mm) hold for the instruments that will be used | null  | `not_established` | `sha256:1448810e0f8a` |
| T127 | Rolled-cylinder chord-geodesic gaps for the protocol marker pairs | (object of 10 entries; see the source report) mm | `numerically_verified` | `sha256:7cb6f15d966f` |
| T127 | Rolled-cylinder Jacobi transfer equals the flat-plate transfer (K = 0) | 0.0  | `numerically_verified` | `sha256:7cb6f15d966f` |
| T127 | A marker chord differs from the surface distance on a developable (intrinsically flat) part | 15.658276442180153 mm | `numerically_verified` | `sha256:7cb6f15d966f` |
| T127 | Minimum circumferential marker separation that resolves the chord-geodesic gap at k = 2 | (object of 3 entries; see the source report) mm | `numerically_verified` | `sha256:7cb6f15d966f` |
| T127 | Rolled-cylinder protocol record validates and refuses malformed variants | 9 refused mutations | `numerically_verified` | `sha256:7cb6f15d966f` |
| T127 | Measured chords and surface distances on the physical tube match the predicted gaps | null  | `not_established` | `sha256:7cb6f15d966f` |
| T127 | The physical tube radius and roundness lie within the declared +/- 0.1 mm | null  | `not_established` | `sha256:7cb6f15d966f` |
| T128 | Laterally offset routes cross the nominal route at the first focal point (small-offset limit) | 153.40715534052734 mm | `numerically_verified` | `sha256:9aa91cae77ac` |
| T128 | Domed-coupon Jacobi transfer at the end of the nominal route | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:9aa91cae77ac` |
| T128 | Linearized separation remainder is at least second order (third order on the symmetry axis) | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:9aa91cae77ac` |
| T128 | Curvature signature relative to the flat-plate control at the route end | (object of 5 entries; see the source report) mm | `numerically_verified` | `sha256:9aa91cae77ac` |
| T128 | The RK4 coupon transfer agrees with ciw's adaptive Dormand-Prince 5(4) integration of the same equations | 9.824155142723612e-07  | `numerically_verified` | `sha256:9aa91cae77ac` |
| T128 | Coupon Christoffel symbols and Gaussian curvature agree with finite-difference derivations from the metric and from the height values | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:9aa91cae77ac` |
| T128 | scipy DOP853 integrating the ciw right-hand side agrees with the RK4 coupon transfer (independent time stepping; the geometry code is shared) | 9.82036297614286e-07  | `independently_verified` | `sha256:9aa91cae77ac` |
| T128 | A sympy derivation of the coupon Christoffel symbols and Gaussian curvature from the height formula agrees with ciw at six points | 8.673617379884035e-19  | `independently_verified` | `sha256:9aa91cae77ac` |
| T128 | Domed-coupon protocol record validates and refuses malformed variants | 9 refused mutations | `numerically_verified` | `sha256:9aa91cae77ac` |
| T128 | Measured separations of offset routes on the physical coupon follow the Jacobi prediction and cross at the predicted focal point | null  | `not_established` | `sha256:9aa91cae77ac` |
| T128 | The formed coupon matches the declared dome (height 10 mm, sigma 20 mm) within tolerance | null  | `not_established` | `sha256:9aa91cae77ac` |
| T128 | An unsteered 3 mm tape laid on the coupon follows a geodesic of the as-built surface (no in-plane bending, lift-off or slip) | null  | `not_established` | `sha256:9aa91cae77ac` |
| T128 | The start jig and tape laying realize the relative start pose of the offset tape within the declared 0.05 mm and 0.5 mrad | null  | `not_established` | `sha256:9aa91cae77ac` |
| T129 | Sampling design resolving profile curvature to tolerance at k = 2 | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:93df097d5505` |
| T129 | Monte Carlo curvature error on the coupon crest matches the derived bias and noise | (object of 5 entries; see the source report) 1/mm | `numerically_verified` | `sha256:93df097d5505` |
| T129 | The osculating-circle window rule under-predicts the coupon-crest curvature bias | 1.9606615754433288  | `numerically_verified` | `sha256:93df097d5505` |
| T129 | A smaller fitting window at fixed spacing can make the curvature estimate worse | (object of 2 entries; see the source report) 1/mm | `numerically_verified` | `sha256:93df097d5505` |
| T129 | Registration residual and rotation covariance of repeat scans follow chi-square(3N - 6) and the inertia formula | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:93df097d5505` |
| T129 | Repeat scans needed at the native 0.05 mm spacing versus curvature tolerance (coupon crest) | {"1%": 109, "2%": 5, "5%": 1} scans | `numerically_verified` | `sha256:93df097d5505` |
| T129 | A real laser line scanner achieves the declared 0.01 mm point noise on the coupon surface (finish, incidence angle, speckle) | null  | `not_established` | `sha256:93df097d5505` |
| T130 | First-order covariance of the INSTRUMENT->CAD frame chain predicts the Monte Carlo spread of coupon points | (object of 2 entries; see the source report) mm | `numerically_verified` | `sha256:db83fe1e3239` |
| T130 | The 3-2-1 datum frame is orthonormal, equivariant under rigid motion and has the propagated covariance | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:db83fe1e3239` |
| T130 | Gauge-sphere fit on a 75 degree cap recovers centre and radius with the linearized covariance | (object of 2 entries; see the source report) mm | `numerically_verified` | `sha256:db83fe1e3239` |
| T130 | Step-gauge fit recovers a declared 50 ppm scale error and 1 um offset | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:db83fe1e3239` |
| T130 | The physical gauge sphere, step gauge and scale bar have their certified dimensions, and the lab frame chain has the declared covariances | null  | `not_established` | `sha256:db83fe1e3239` |
| T131 | ANOVA Gage R&R recovers the declared variance components on synthetic studies | (object of 4 entries; see the source report) mm^2 | `numerically_verified` | `sha256:54202d4bf05e` |
| T131 | Sampling spread of %GRR from a single 10 x 3 x 3 study | (object of 4 entries; see the source report) % | `numerically_verified` | `sha256:54202d4bf05e` |
| T131 | Fraction of 10 x 3 x 3 studies whose raw operator component is negative (truncated to zero) | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:54202d4bf05e` |
| T131 | Gage R&R refuses unbalanced or incomplete designs | 2 refusals | `numerically_verified` | `sha256:54202d4bf05e` |
| T131 | The real gage (instrument, fixture and operators) has %GRR below 10% on the coupon features | null  | `not_established` | `sha256:54202d4bf05e` |
| T131 | The measurement system is approved for production use | "not_performed"  | `not_established` | `sha256:54202d4bf05e` |
| T132 | Geodesic curvature of a 30-to-60 degree variable-angle steered course on the R = 100 mm mandrel | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:ba3a82e14ab7` |
| T132 | Lateral deviation on the mandrel follows e(s) = delta + s sin(dpsi) (flat Jacobi transfer) | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:ba3a82e14ab7` |
| T132 | Tolerance stack of the placed course and the longest course meeting a 0.5 mm lateral spec at k = 2 | (object of 2 entries; see the source report) mm | `numerically_verified` | `sha256:ba3a82e14ab7` |
| T132 | Programming a helix in machine angles transfers mandrel radius error into lateral drift | 0.9990005004983686 mm | `numerically_verified` | `sha256:ba3a82e14ab7` |
| T132 | End deviation of the steered course from the geodesic along its initial tangent | 122.56351482368811 mm | `numerically_verified` | `sha256:ba3a82e14ab7` |
| T132 | Tows placed by a real AFP head follow the programmed course within the stack, with gaps and overlaps inside 0.5 mm | null  | `not_established` | `sha256:ba3a82e14ab7` |
| T132 | The declared 635 mm minimum steering radius avoids tow wrinkling for the placed material | null  | `not_established` | `sha256:ba3a82e14ab7` |
| T133 | The Clairaut constant is conserved along geodesic windings of the torus mandrel | {"psi50": 128.55752193730788, "psi70": 68.40402866513377} mm | `numerically_verified` | `sha256:27fb2ef6568d` |
| T133 | Secular growth rate of the heading-error Jacobi field per unit arclength over one winding period, relative to the cylinder (rate 1) | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:27fb2ef6568d` |
| T133 | The 50 deg winding librates (turns before the inner equator) and the 70 deg winding circulates through the negatively curved inner region | (object of 2 entries; see the source report) rad | `numerically_verified` | `sha256:27fb2ef6568d` |
| T133 | Clairaut sensitivity predicts the turnaround-latitude shift of the librating winding | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:27fb2ef6568d` |
| T133 | Slippage tendency abs(kappa_g / kappa_n) of constant-angle winding on the torus mandrel | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:27fb2ef6568d` |
| T133 | A constant winding angle is not geodesic on the torus mandrel | 0.004545022346608683 1/mm | `numerically_verified` | `sha256:27fb2ef6568d` |
| T133 | Fibre does not slip on a real mandrel wherever abs(kappa_g / kappa_n) <= 0.2 (the friction coefficient is declared, not measured) | null  | `not_established` | `sha256:27fb2ef6568d` |
| T134 | Lateral-error envelope of the coupon trajectory under the declared registration box | (object of 2 entries; see the source report) mm | `numerically_verified` | `sha256:5d8c1b51a239` |
| T134 | Standoff error from a lateral tool offset follows -kappa_lateral e^2 / 2 | (object of 3 entries; see the source report) mm | `numerically_verified` | `sha256:5d8c1b51a239` |
| T134 | Tool-centre-point path length element is sqrt((1 - H kappa_n)^2 + (H tau_g)^2) | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:5d8c1b51a239` |
| T134 | A spray standoff beyond the concave radius of curvature folds the tool-centre-point path | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:5d8c1b51a239` |
| T134 | The torch or gun on a real cell stays within the predicted lateral and standoff band | null  | `not_established` | `sha256:5d8c1b51a239` |
| T134 | The trajectory is safe to execute on a welding or coating robot cell | null  | `not_established` | `sha256:5d8c1b51a239` |
| T135 | Coverage and path length of candidate inspection scan plans on the domed coupon | (object of 11 entries; see the source report)  | `numerically_verified` | `sha256:468deff05cee` |
| T135 | Geodesic rows at the swath spacing leave gaps on the domed coupon | 0.03964534619021176  | `numerically_verified` | `sha256:468deff05cee` |
| T135 | Shortest evaluated scan plan with no uncovered sample among 32000 area samples | (object of 2 entries; see the source report)  | `numerically_verified` | `sha256:468deff05cee` |
| T135 | Geodesic rows at spacing swath / max j_lat (first-order Jacobi tightening) leave under 0.1% uncovered | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:468deff05cee` |
| T135 | The real scanner footprint is a 20 mm swath on this surface at the planned standoff | null  | `not_established` | `sha256:468deff05cee` |
| T136 | Candidate coupon routes ranked by the heading calibration tolerance that keeps the exactly perturbed route within a 0.5 mm lateral spec | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:1572b044dca2` |
| T136 | The first-order tolerance allocation exceeds the spec at a tolerance corner on some off-axis routes | (object of 6 entries; see the source report)  | `numerically_verified` | `sha256:1572b044dca2` |
| T136 | The dome loosens the heading tolerance of the straight route relative to a flat plate of equal length | 1.8730416064774404  | `numerically_verified` | `sha256:1572b044dca2` |
| T136 | The robot, fixture and frame calibration achieves the required heading and lateral tolerances | null  | `not_established` | `sha256:1572b044dca2` |
| T137 | Candidate coupon routes ranked by focus margin (nearest focal or conjugate point / route length) | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:45a45fcc21fd` |
| T137 | The shortest candidate route has the worst focus margin | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:45a45fcc21fd` |
| T137 | The second variation of route length equals the Jacobi index form j_head(L) j_head'(L) | (object of 2 entries; see the source report) mm | `numerically_verified` | `sha256:45a45fcc21fd` |
| T137 | The calibration-tolerance ranking and the focus-margin ranking put the straight route at opposite ends | {"calibration_first": "fan+0deg", "focus_last": "fan+0deg"}  | `numerically_verified` | `sha256:45a45fcc21fd` |
| T137 | Physical paths near a predicted focus show the predicted loss of lateral-error ordering | null  | `not_established` | `sha256:45a45fcc21fd` |
| T138 | Predicted separation of the 2 mm offset route at the MFG-COUPON-01 stations | (object of 3 entries; see the source report) mm | `numerically_verified` | `sha256:f7ea13178c44` |
| T138 | A 2-sigma start offset of the declared jig makes a correct model fail E_n <= 1 unless the start pose is budgeted or measured | (object of 4 entries; see the source report)  | `numerically_verified` | `sha256:f7ea13178c44` |
| T138 | The comparison refuses to run without acquired hardware evidence | 4 refusals | `numerically_verified` | `sha256:f7ea13178c44` |
| T138 | Measured separation on the coupon agrees with the prediction (E_n <= 1 at every station) | null  | `not_established` | `sha256:f7ea13178c44` |
| T138 | The start jig and tape laying realize the relative start pose of the offset tape within the declared 0.05 mm and 0.5 mrad | null  | `not_established` | `sha256:f7ea13178c44` |
| T139 | The retention schema refuses records missing raw digests, calibration, frame-chain or clock metadata | 11 refused mutations | `numerically_verified` | `sha256:a4d5b796a670` |
| T139 | The retention schema keeps a valid rank-deficient frame covariance at 1e2 mm^2 scale | (object of 2 entries; see the source report) mm^2 | `numerically_verified` | `sha256:a4d5b796a670` |
| T139 | Retention record identity is independent of key order and bound to every raw digest | "sha256:eae8d469902139370a09e287fdde5378b44f6a371\u2026"  | `numerically_verified` | `sha256:a4d5b796a670` |
| T139 | A schema fixture (as is or relabelled as a measurement), or a record without its raw bytes, is refused as hardware evidence | 3 refusals | `numerically_verified` | `sha256:a4d5b796a670` |
| T139 | A real measurement with raw bytes, calibration and frame metadata has been retained | 0 records | `not_established` | `sha256:a4d5b796a670` |
| T140 | Uncertainty budget per predicted quantity and its limiting term | (object of 8 entries; see the source report) mm | `numerically_verified` | `sha256:e2a332efa46d` |
| T140 | The coupon focal-distance prediction is geometry-limited, not instrument-limited | 0.8909267675541553  | `numerically_verified` | `sha256:e2a332efa46d` |
| T140 | On the coupon, the heading-offset separation at the route end is limited by the start pose open loop, and a CMM start-pose measurement removes that limit | (object of 3 entries; see the source report)  | `numerically_verified` | `sha256:e2a332efa46d` |
| T140 | The classification detects a solver-limited prediction in the coarse-solver control | "solver"  | `numerically_verified` | `sha256:e2a332efa46d` |
| T140 | The declared instrument uncertainties are the uncertainties of the instruments used | null  | `not_established` | `sha256:e2a332efa46d` |
| T140 | The start jig and tape laying realize the relative start pose of the offset tape within the declared 0.05 mm and 0.5 mrad | null  | `not_established` | `sha256:e2a332efa46d` |
| T140 | The budget contains every significant physical error source (thermal, fixturing, tape bending along the route, target centring) | null  | `not_established` | `sha256:e2a332efa46d` |
| T141 | No basis establishes a claim filed in an authority domain, and the acceptance policy, the protocol validator and this section's acceptance-language screen refuse acceptance decisions | {"basis_domain_cases": 320, "refusals": 8, "violations": 0} cases | `numerically_verified` | `sha256:176b5f7aecd4` |
| T141 | evidence.finding establishes an acceptance statement when its author files it in a computational domain | "numerically_verified"  | `numerically_verified` | `sha256:176b5f7aecd4` |
| T141 | Domain assignment of free-text claims is machine-checked across the lab | null  | `not_established` | `sha256:176b5f7aecd4` |
| T141 | Production acceptance of the coupon, cylinder or plate process | "not_performed"  | `not_established` | `sha256:176b5f7aecd4` |
| T141 | The manufacturing protocols and models are ready for industrial use | null  | `not_established` | `sha256:176b5f7aecd4` |

## Limitations

- T045: The declared noise-model parameters describe real instruments of these modes — not established.
- T046: The chord correction computed from nominal curvature holds for chords measured on a physical part — not established.
- T047: Chords measured between physical markers on a cylindrical part follow the cos^4(alpha)/(24 R^2) coefficient — not established.
- T048: A physical stereo rig with this geometry achieves the synthetic chord accuracy — not established.
- T049: The declared perturbation magnitudes bound the calibration error of a real stereo rig — not established.
- T050: A two-term radial plus tangential Brown-Conrady model describes a real lens to the required accuracy — not established.
- T051: Real image noise is Gaussian with sigma = 0.25 px and marker localization rounds to whole pixels — not established.
- T052: A real encoder drive train behaves as a constant-width play operator with constant scale and bias — not established.
- T053: Real gyroscopes have constant bias and white rate noise with the declared densities — not established.
- T054: Real sensor clocks have the declared constant offset and white jitter — not established.
- T055: Real links drop observations as Bernoulli or two-state burst processes with these rates — not established.
- T056: Real tracker latencies and target speeds match the declared values — not established.
- T057: A real tracker's measurement noise and target motion match the constant-velocity model — not established.
- T058: The declared frame and clock mappings equal the real extrinsic calibration and clock synchronization — not established.
- T059: Admission as workbench state confers authority to act on a machine — not established.
- T060: The bench's noise levels, rates and motion describe real camera, encoder, IMU or tracker hardware — not established.
- T061: Real sensors' noise covariance equals the covariance declared for this bench — not established.
- T062: Real camera and tracker noises share the common-mode covariance assumed here — not established.
- T063: The declared 35 degree rotation, translation and body covariance describe a real sensor mounting or extrinsic calibration — not established.
- T064: The declared [lateral, heading] covariance and these surfaces predict the path uncertainty of a real vehicle or tool on a real curved part — not established.
- T065: A real sensor's internally filtered output can be fused downstream as white noise — not established.
- T066: A real residual monitor normalized by datasheet sensor covariance is correctly calibrated — not established.
- T067: A real gate at the 99% quantile rejects 1% of valid real readings — not established.
- T068: Real outliers are rare, isolated and of fixed magnitude as in this contamination model — not established.
- T069: Real sensor dropouts are independent of the state and of the noise, as assumed here — not established.
- T070: The recovered offset calibrates the clock of a real camera — not established.
- T071: A 2 degree rotation is the size of a real extrinsic calibration error — not established.
- T072: The validity interval [0, 60) reflects how long a real camera calibration stays valid — not established.
- T073: A 1 m, 99% track-loss radius is a safe operating threshold — not established.
- T074: The fused estimate equals the physical state of a real target within its covariance — not established.
- T075: An admitted synthetic state may command actuators — not established.
- T076: Synthetic fusion output is admissible as production state — not established.
- T115: Gross CPU package energy (background-inclusive, idle not subtracted) per geodesic trajectory — not established.
- T115: Idle-subtracted CPU package energy per geodesic trajectory (equal-length idle bracket after the workload) — not established.
- T115 is partial: Integrate each geodesic with RK4 (N = 256) and adaptive DP5(4); count evaluations; when CIW_LAB_RAPL_LOG names a capture from `python -m ciw.lab.energy_gpu_telemetry rapl-capture` (repeated fixed-step
- T116: GPU-domain gross energy per measured batch — not established.
- T116: The NVML total-energy counter of the RTX 2080 has a characterized accuracy and resolution — not established.
- T116 is blocked: Blocked: unavailable requirement(s) hardware:nvidia-gpu. Planned: On the RTX 2080 host: (0) `mkdir -p runs/rtx2080-<date>`; (1) `ciw energy probe --gpu-index 0` must return a reading with status ok; (
- T117: A Julia implementation agrees with the Python kernel — not established.
- T117: A GPU implementation agrees with the Python kernel — not established.
- T117: The Rust kernel uses less energy per trajectory than the Python kernel on real hardware — not established.
- T117 is partial: Integrate the same initial states with the generic Python path, the closed-form Python path and the compiled Rust kernel; compare endpoints, counted evaluations and exact endpoints; send the Rust kern
- T118: RTX 2080 power draw during the measurement phase (NVML) — not established.
- T118: RTX 2080 temperature during the measurement phase (NVML) — not established.
- T118: RTX 2080 graphics clock during the measurement phase (NVML) — not established.
- T118: Host-bracketed batch solve duration (launch, sync and copy included) — not established.
- T118: RTX 2080 GPU utilization during the measurement phase (nvidia-smi rows inside the measurement window) — not established.
- T118: RTX 2080 power draw is steady over the measurement phase (coefficient of variation <= 0.10) — not established.
- T118: RTX 2080 temperature drifts by at most 5 C over the measurement phase — not established.
- T118: RTX 2080 kernel-only duration of the Gaussian VI kernel — not established.
- T118 is blocked: Blocked: unavailable requirement(s) hardware:nvidia-gpu. Planned: On the RTX 2080 host: (0) `mkdir -p runs/rtx2080-<date>`; (1) start `TZ=UTC nvidia-smi --query-gpu=timestamp,uuid,name,utilization.gpu
- T119: Physical GPU energy per accepted numerical result — not established.
- T119 is partial: Analyze all four fixtures; recompute the baseline metric from raw readings and raw outputs; compare with the naive gross/executed ratio, with the per-distinct-result denominator and with a whole-run b
- T120: float32 lowers the energy per accepted trajectory relative to float64 on real CPU or GPU hardware — not established.
- T120 is partial: Vectorized RK4 in float32 and float64 across the step grid; smallest grid N per accuracy target; operation counts derived from the kernel source and checked by running one step of the same code on cou
- T121: Reductions on the RTX 2080 (CUB, cuBLAS or atomicAdd) reproduce these emulated spreads and sign flips — not established.
- T121 is partial: Emulate the orders in float32 and float64; search seeds for a sign flip among the sequential and two tree orders; test a pass/fail tolerance across atomic orders; test the guarded sign decision; compa
- T122: The variational free energy of this model equals a thermodynamic free energy of a physical system — not established.
- T123: A decrease of variational free energy corresponds to a decrease of physical energy consumed by the computation — not established.
- T124: The fixtures' counter readings were produced by a physical GPU and NVML counter — not established.
- T125: Replayed energy values are physically valid measurements — not established.
- T126: Measured marker chords on the physical plate equal the predicted geodesic distances within instrument uncertainty — not established.
- T126: The declared instrument uncertainties (camera 0.02 mm, tracker 0.015 mm, CMM 0.002 mm) hold for the instruments that will be used — not established.
- T127: Measured chords and surface distances on the physical tube match the predicted gaps — not established.
- T127: The physical tube radius and roundness lie within the declared +/- 0.1 mm — not established.
- T128: Measured separations of offset routes on the physical coupon follow the Jacobi prediction and cross at the predicted focal point — not established.
- T128: The formed coupon matches the declared dome (height 10 mm, sigma 20 mm) within tolerance — not established.
- T128: An unsteered 3 mm tape laid on the coupon follows a geodesic of the as-built surface (no in-plane bending, lift-off or slip) — not established.
- T128: The start jig and tape laying realize the relative start pose of the offset tape within the declared 0.05 mm and 0.5 mrad — not established.
- T129: A real laser line scanner achieves the declared 0.01 mm point noise on the coupon surface (finish, incidence angle, speckle) — not established.
- T130: The physical gauge sphere, step gauge and scale bar have their certified dimensions, and the lab frame chain has the declared covariances — not established.
- T131: The real gage (instrument, fixture and operators) has %GRR below 10% on the coupon features — not established.
- T131: The measurement system is approved for production use — not established.
- T132: Tows placed by a real AFP head follow the programmed course within the stack, with gaps and overlaps inside 0.5 mm — not established.
- T132: The declared 635 mm minimum steering radius avoids tow wrinkling for the placed material — not established.
- T133: Fibre does not slip on a real mandrel wherever abs(kappa_g / kappa_n) <= 0.2 (the friction coefficient is declared, not measured) — not established.
- T134: The torch or gun on a real cell stays within the predicted lateral and standoff band — not established.
- T134: The trajectory is safe to execute on a welding or coating robot cell — not established.
- T135: The real scanner footprint is a 20 mm swath on this surface at the planned standoff — not established.
- T136: The robot, fixture and frame calibration achieves the required heading and lateral tolerances — not established.
- T137: Physical paths near a predicted focus show the predicted loss of lateral-error ordering — not established.
- T138: Measured separation on the coupon agrees with the prediction (E_n <= 1 at every station) — not established.
- T138: The start jig and tape laying realize the relative start pose of the offset tape within the declared 0.05 mm and 0.5 mrad — not established.
- T138 is partial: Compute the prediction and its uncertainty components; evaluate a correct model against a tape realized 0.1 mm off its nominal offset with and without the start-pose term, and against the prediction r
- T139: A real measurement with raw bytes, calibration and frame metadata has been retained — not established.
- T140: The declared instrument uncertainties are the uncertainties of the instruments used — not established.
- T140: The start jig and tape laying realize the relative start pose of the offset tape within the declared 0.05 mm and 0.5 mrad — not established.
- T140: The budget contains every significant physical error source (thermal, fixturing, tape bending along the route, target centring) — not established.
- T141: Domain assignment of free-text claims is machine-checked across the lab — not established.
- T141: Production acceptance of the coupon, cylinder or plate process — not established.
- T141: The manufacturing protocols and models are ready for industrial use — not established.
