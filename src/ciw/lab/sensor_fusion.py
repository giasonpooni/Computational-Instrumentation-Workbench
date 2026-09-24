"""Sensor-fusion experiments T060-T076 on a deterministic synthetic bench.

Scope: seeded linear-Gaussian experiments that test what a Kalman-type fusion
pipeline must satisfy before its outputs could even be considered: declared
covariances are reproduced, chi-square consistency (NEES/NIS) holds only under
the correct noise, frame, clock and calibration model, covariance transforms
correctly through frames and Jacobi transfer matrices, filtered estimates are
correlated, gating rates match their quantiles, gaps, stale clocks, frame
mismatches, expired calibrations and lost tracks are handled explicitly, the
recursive estimate equals the exact batch posterior, and observations,
candidate states and admitted states stay separate with a read-only default.
Where a mismatched linear filter serves as a counterexample, its NEES/NIS or
mean innovation is predicted from exact joint moments
(:func:`ciw.lab.sensor_fusion_bench.mismatch_moments`) or exact mean
propagation and compared with the simulation.

This module holds the bench tasks T060-T062 and imports the sibling modules
that register T063-T064 (``sensor_fusion_geometry``), T065-T068
(``sensor_fusion_filtering``), T069-T071 (``sensor_fusion_robustness``) and
T072-T076 (``sensor_fusion_admission``). Each sibling is imported on its own:
if one fails to import, its tasks are registered as placeholders that report
the import error as blocked, and the other tasks still run.

Non-claims: all readings are synthetic draws from declared distributions.
Nothing here measures a real sensor, validates a calibration, certifies a
clock, a frame, a safety threshold or production acceptance; those claims are
recorded as ``not_established`` findings in physical and authority domains.
"""
from __future__ import annotations

from importlib import import_module
import math

import numpy as np

from . import registry, svg
from .evidence import finding
from .registry import task
from .sensor_fusion_bench import (BENCH_SEED, H_POS, BenchConfig, bench_digest, chi2_cdf, consistency, cv_model,
                                  gain_schedule, gaussian, generate_bench, generator, measure, mismatch_moments,
                                  nees_series, normal_quantile, quadratic, run_shared, sensor_function,
                                  simulate_truth, with_sensor)
from .sensor_fusion_common import (MU0, P0_BENCH, TESTS, TOL_EXACT, TOL_MC, TOL_ROUNDOFF, TOL_TINY, as_json,
                                   check, exact, files, generator_basis, gram, max_abs_z_spread, mc95, outcome,
                                   rate_interval, roundoff, uncertainty, unreal)

FILES = files("sensor_fusion")
BENCH_RUNS = 50
MC_RUNS = 200
MC_TICKS = 100


def _tests(task_id, *specific) -> tuple:
    """Regression test node ids: the task's own tests plus its row of the parametrized section test."""
    return tuple(f"{TESTS}::{name}" for name in specific) + (
        f"{TESTS}::test_section_reports_labels_and_states[{task_id}]",)


def _bench(ctx):
    return ctx.memo("sensor_fusion.bench", lambda: generate_bench(BenchConfig(), BENCH_SEED, BENCH_RUNS))


# T060 ------------------------------------------------------------------------------
@task("T060", changed_files=FILES, regression_tests=_tests(
    "T060", "test_bench_is_deterministic_rate_exact_and_stream_independent"))
def multi_sensor_bench(ctx):
    config = BenchConfig()
    bench = _bench(ctx)
    again = generate_bench(config, BENCH_SEED, BENCH_RUNS)
    other = generate_bench(config, BENCH_SEED + 1, BENCH_RUNS)
    names = sorted(bench["readings"])

    def arrays(b):
        return [b["truth"]] + [b["readings"][n]["values"] for n in names]

    same_differences = sum(not np.array_equal(a, c) for a, c in zip(arrays(bench), arrays(again)))
    other_changed = float(np.mean([not np.array_equal(a, c) for a, c in zip(arrays(bench), arrays(other))]))
    retimed = generate_bench(with_sensor(config, "tracker", every=5), BENCH_SEED, BENCH_RUNS)
    untouched = [n for n in names if n != "tracker"]
    stream_changes = int(not np.array_equal(bench["truth"], retimed["truth"])) + sum(
        not np.array_equal(bench["readings"][n]["values"], retimed["readings"][n]["values"]) for n in untouched)

    counts, count_mismatch, offgrid = {}, 0, 0
    for spec in config.sensors:
        ticks = bench["readings"][spec.name]["ticks"]
        expected = config.ticks // spec.every
        counts[spec.name] = {"readings_per_run": int(len(ticks)), "expected": expected,
                             "rate_hz": 1.0 / (spec.every * config.dt)}
        count_mismatch = max(count_mismatch, abs(len(ticks) - expected))
        offgrid += int(np.count_nonzero(ticks % spec.every))

    # Separate recomputation of the trajectory: cumulative sums of the retained initial state and process-noise
    # draws with the declared dt (p_k = p_0 + sum_{j<k} (dt v_j + w_p,j), v_k = v_0 + sum_{j<k} w_v,j), not the
    # generator's transition matrix or recursion. Every sensor function is then evaluated on that trajectory
    # (hypot and complex-angle forms for speed and heading rate), so a position reading is not re-read from the
    # array it was built from.
    truth = bench["truth"]
    x0, w = truth[:, 0], bench["process"]
    velocity = np.concatenate([x0[:, None, 2:], x0[:, None, 2:] + np.cumsum(w[:, :, 2:], axis=1)], axis=1)
    position = np.concatenate([x0[:, None, :2], x0[:, None, :2] + np.cumsum(config.dt * velocity[:, :-1]
                                                                            + w[:, :, :2], axis=1)], axis=1)
    recomputed = np.concatenate([position, velocity], axis=2)
    trajectory_error = float(np.max(np.abs(recomputed - truth)))
    composition = {}
    for spec in config.sensors:
        record = bench["readings"][spec.name]
        ticks = record["ticks"]
        if spec.kind == "position":
            clean = recomputed[:, ticks, :2]
        elif spec.kind == "speed":
            clean = np.hypot(recomputed[:, ticks, 2], recomputed[:, ticks, 3])[..., None]
        else:
            now = recomputed[:, ticks, 2] + 1j * recomputed[:, ticks, 3]
            before = recomputed[:, ticks - 1, 2] + 1j * recomputed[:, ticks - 1, 3]
            clean = (np.angle(now * np.conj(before)) / config.dt)[..., None]
        composition[spec.name] = float(np.max(np.abs(record["values"] - clean - record["noise"])))
    worst_composition = max(composition.values())
    min_eigen = min(float(np.linalg.eigvalsh(spec.R).min()) for spec in config.sensors)
    asymmetry = max(float(np.max(np.abs(spec.R - spec.R.T))) for spec in config.sensors)
    eigen_roundoff = np.finfo(float).eps * max(float(np.max(np.abs(spec.R))) for spec in config.sensors)

    run0 = {"truth": bench["truth"][0], "readings": {n: {"ticks": bench["readings"][n]["ticks"],
                                                         "values": bench["readings"][n]["values"][0]} for n in names}}
    ctx.artifact_json("bench.json", as_json({"config": config.describe(), "seed": BENCH_SEED, "runs": BENCH_RUNS,
                                             "retained_run": 0, "run": run0,
                                             "declared_covariances": {s.name: s.covariance for s in config.sensors},
                                             "process_noise_Q_per_tick": bench["Q"]}))
    ctx.artifact_json("digests.json", {"seed": BENCH_SEED, "digest": bench_digest(bench),
                                       "regenerated_digest": bench_digest(again),
                                       "other_seed_digest": bench_digest(other),
                                       "note": "SHA-256 over little-endian float64 truth and readings"})
    cam, trk = bench["readings"]["camera"], bench["readings"]["tracker"]
    ctx.artifact_text("trajectory.svg", svg.line_plot(
        [("truth", truth[0, :, 0], truth[0, :, 1]), ("camera", cam["values"][0, :, 0], cam["values"][0, :, 1]),
         ("tracker", trk["values"][0, :, 0], trk["values"][0, :, 1])],
        title="T060 synthetic run 0: truth and position readings", xlabel="x (m)", ylabel="y (m)"))

    findings = [
        finding("Regenerating the bench with the same seed reproduces the truth and all four sensor streams "
                "bit for bit, and a different seed changes every stream", "computational_pipeline",
                {"arrays_compared": len(names) + 1, "same_seed_arrays_differing": same_differences,
                 "other_seed_fraction_changed": other_changed},
                {**generator_basis(BENCH_SEED, runs=BENCH_RUNS), "checks": [
                    check("invariant", "same-seed regeneration", same_differences, 0),
                    check("invariant", "different-seed regeneration", other_changed, 1.0, "ge")]},
                uncertainty=exact("deterministic bitwise comparison"), tolerance=TOL_EXACT),
        finding("Every sensor reports exactly at its declared rate on the 20 Hz base clock over 20 s "
                "(camera 10 Hz, encoder 20 Hz, IMU 20 Hz, tracker 2 Hz)", "computational_pipeline",
                {"per_sensor": counts, "max_count_mismatch": count_mismatch, "off_grid_ticks": offgrid},
                {**generator_basis(BENCH_SEED), "checks": [
                    check("exact_arithmetic", "declared rate times duration", count_mismatch, 0),
                    check("exact_arithmetic", "reading ticks on the declared grid", offgrid, 0)]},
                uncertainty=exact("integer tick counts"), tolerance=TOL_EXACT),
        finding("Every raw reading of all four sensors equals the noise-free sensor function of a trajectory "
                "recomputed from the retained initial state and process-noise draws by cumulative sums with the "
                "declared dt, plus its retained noise draw, to roundoff; the recomputed trajectory equals the "
                "retained truth to roundoff", "numerical",
                {"max_abs_residual_by_sensor": composition, "max_abs_trajectory_difference": trajectory_error},
                {**generator_basis(BENCH_SEED), "checks": [
                    check("invariant", "cumulative-sum trajectory against the retained truth (max abs, m and m/s)",
                          trajectory_error, 1e-10),
                    # The heading rate divides velocity roundoff by |v| dt, so its residual is the largest.
                    check("invariant", "reading minus h(recomputed trajectory) minus retained noise (max abs over the "
                                       "four sensors)", worst_composition, 1e-10)]},
                unit="max abs difference", uncertainty=roundoff(max(worst_composition, trajectory_error)),
                tolerance={"abs": 1e-10, "rel": 0.0}),
        finding("Changing the tracker rate from 2 Hz to 4 Hz leaves the truth and the camera, encoder and IMU "
                "streams unchanged (independent spawned PCG64 streams)", "computational_pipeline", stream_changes,
                {**generator_basis(BENCH_SEED), "checks": [
                    check("invariant", "streams other than the tracker", stream_changes, 0)]},
                uncertainty=exact("deterministic bitwise comparison"), tolerance=TOL_EXACT),
        finding("Every declared sensor covariance is symmetric positive definite", "numerical", min_eigen,
                {"derivation": "declared constants in DEFAULT_SENSORS", "checks": [
                    check("invariant", "max |R - R^T| over the declared covariances", asymmetry, 0.0),
                    check("analytic", "smallest eigenvalue of the declared covariances", min_eigen, 1e-6, "ge")]},
                unit="smallest eigenvalue",
                uncertainty=roundoff(eigen_roundoff, "eigvalsh roundoff bound eps * max |R_ij|"),
                tolerance=TOL_ROUNDOFF),
        unreal("The bench's noise levels, rates and motion describe real camera, encoder, IMU or tracker hardware",
               "sensor_performance", BENCH_SEED, "not established: every stream is a declared synthetic draw"),
    ]
    count_text = ", ".join(f"{name} {row['readings_per_run']}" for name, row in counts.items())
    fields = {
        "hypothesis": "A seeded, stream-separated generator can produce a planar multi-sensor bench whose truth, "
                      "raw readings, noise draws and declared covariances are exactly reproducible and rate-exact.",
        "mathematical_model": "Truth x = (px, py, vx, vy), x_{k+1} = F x_k + w_k with the exact white-noise-"
                              "acceleration discretization F = [[I, dt I], [0, I]], Q = q [[dt^3/3 I, dt^2/2 I], "
                              "[dt^2/2 I, dt I]] (dt = 0.05 s, q = 0.05 m^2/s^3). Sensors: camera position "
                              "(10 Hz, R = [[0.04, 0.012], [0.012, 0.04]] m^2), encoder speed |v| (20 Hz, 0.01 "
                              "m^2/s^2), IMU heading rate as an integrating gyro wrap(theta_k - theta_{k-1})/dt "
                              "(20 Hz, 4e-4 rad^2/s^2), tracker position (2 Hz, 0.0025 I m^2).",
        "input_data": [f"seed {BENCH_SEED} (PCG64, SeedSequence.spawn per stream)", f"{BENCH_RUNS} replicas x 400 "
                       "ticks", "x0 ~ N((0, 0, 1, 0.5), diag(0.25, 0.25, 0.04, 0.04))"],
        "observation_model": "z = h(x) + v with v ~ N(0, R) drawn by Cholesky factors (not SVD) so draws are "
                             "platform-stable; readings at every positive multiple of each sensor's period.",
        "expected_invariant": "Same seed gives identical bytes; different seed changes every stream; reading "
                              "counts equal duration x rate; z - h(truth) equals the retained noise; streams are "
                              "independent of other sensors' settings.",
        "experiment": "Generate twice with the same seed, once with seed+1 and once with a 4 Hz tracker; compare "
                      "arrays bitwise; recompute the trajectory from x0 and the retained process noise by cumulative "
                      "sums with the declared dt and evaluate every sensor function on it (hypot and complex angles "
                      "for speed and heading rate); count reading ticks; check symmetry and positive definiteness of "
                      "the declared covariances.",
        "numerical_result": f"0 of {len(names) + 1} arrays differ on regeneration; all differ under seed+1; "
                            f"counts {count_text}; "
                            f"max composition residual {worst_composition:.2e} against the recomputed trajectory "
                            f"(trajectory difference {trajectory_error:.1e}); tracker retiming changed "
                            f"{stream_changes} other streams.",
        "uncertainty": "None for the determinism and rate claims (exact). The composition residual is floating-"
                       "point roundoff of two summation orders. Cross-platform byte identity of the digest is not "
                       "claimed (libm may "
                       "differ in the last bit); findings avoid digest values for that reason.",
        "failure_modes_checked": ["seed reuse and seed change", "sensor-rate edits leaking into other streams",
                                  "off-grid timestamps", "angle wrap in the heading-rate model",
                                  "a generator transition inconsistent with the declared dt",
                                  "non-positive-definite declared covariance"],
        "unresolved_assumptions": ["White-noise acceleration is a modelling choice, not observed vehicle motion.",
                                   "The heading-rate truth of white-noise-acceleration motion is rough; the IMU "
                                   "stream is kept for completeness but the linear experiments use positions.",
                                   "Encoder and IMU are nonlinear in the state; no experiment here fuses them."],
        "recommended_next_task": "T061: verify the empirical noise covariance against the declared covariance.",
    }
    return outcome(fields, findings)


# T061 ------------------------------------------------------------------------------
UNDERSTATEMENT = 0.9  # declared covariance = 0.9 x true: a 10% understatement
POWER_REPLICATES = 400


def _moment_z(residuals, R):
    """Standardized deviations of the sample mean and second moment from (0, R), known mean zero.

    Var(S_ij) = (R_ij^2 + R_ii R_jj) / N for Gaussian residuals. The Gram
    product avoids BLAS so the retained moments do not depend on thread count.
    """
    N, m = residuals.shape
    S = gram(residuals) / N
    diag = np.diag(R)
    z_cov = (S - R) / np.sqrt((R ** 2 + np.outer(diag, diag)) / N)
    z_mean = residuals.mean(axis=0) / np.sqrt(diag / N)
    upper = np.triu_indices(m)
    return S, z_mean, z_cov[upper]


def _diagonal_index(m: int) -> np.ndarray:
    """Positions of the diagonal entries in the row-major upper-triangle ordering."""
    return np.cumsum([0] + list(range(m, 1, -1)))


def variance_power(samples: int, c: float, z_crit: float) -> float:
    """Exact probability that one variance z-test flags a declared variance c * R_ii (true R_ii).

    N S_ii / R_ii ~ chi2(N), and the test declared at c R_ii flags |S_ii - c R_ii| >= z_crit c R_ii sqrt(2/N).
    """
    width = z_crit * math.sqrt(2.0 / samples)
    upper = 1.0 - chi2_cdf(samples * c * (1.0 + width), samples)
    lower = chi2_cdf(samples * c * (1.0 - width), samples) if width < 1 else 0.0
    return min(1.0, max(0.0, upper + lower))


def half_power_misstatement(samples: int, z_crit: float) -> float:
    """Relative variance excess delta (declared = R / (1 + delta)) that one variance test flags with probability 1/2."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if variance_power(samples, 1.0 / (1.0 + mid), z_crit) < 0.5:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _process_noise_study(q: float, dt: float) -> dict:
    """Q(dt) against the continuous-time integral and the semigroup identity (not against its own draws).

    Q(t) = int_0^t F(u) G q G^T F(u)^T du with G = [0; I]; the integrand is a
    quadratic polynomial in u, so three-point Gauss-Legendre quadrature is exact.
    Errors are relative per nonzero entry, so the small dt^3 block is not hidden
    by the larger dt block.
    """

    def relative(a, b):
        mask = b != 0
        return float(np.max(np.abs(a - b)[mask] / np.abs(b)[mask]))

    nodes, weights = np.polynomial.legendre.leggauss(3)
    G = np.vstack([np.zeros((2, 2)), np.eye(2)])
    integral = np.zeros((4, 4))
    for node, weight in zip(nodes, weights):
        u = 0.5 * dt * (node + 1.0)
        Fu = cv_model(u, q)[0]
        integral += 0.5 * dt * weight * (Fu @ G) @ (q * np.eye(2)) @ (Fu @ G).T
    Q = cv_model(dt, q)[1]
    semigroup = 0.0
    for a, b in ((dt, dt), (dt, 3 * dt), (2.5 * dt, 0.5 * dt)):
        Fb, Qb = cv_model(b, q)
        Qa, Qab = cv_model(a, q)[1], cv_model(a + b, q)[1]
        semigroup = max(semigroup, relative(Fb @ Qa @ Fb.T + Qb, Qab))
    # The discrete white-noise-acceleration alternative q [[dt^4/4, dt^3/2], [dt^3/2, dt^2]] / dt fails both.
    eye = np.eye(2)
    alternative = q / dt * np.block([[dt ** 4 / 4 * eye, dt ** 3 / 2 * eye], [dt ** 3 / 2 * eye, dt ** 2 * eye]])
    return {"dt": dt, "q": q, "quadrature_relative_error": relative(Q, integral),
            "semigroup_relative_error": semigroup, "alternative_model_relative_error": relative(alternative, integral)}


@task("T061", changed_files=FILES, regression_tests=_tests(
    "T061", "test_declared_covariance_matches_and_power_is_quantified"))
def known_truth_covariance(ctx):
    config = BenchConfig()
    bench = _bench(ctx)
    truth = bench["truth"]
    entries, table = [], {}
    for spec in config.sensors:
        record = bench["readings"][spec.name]
        residual = (record["values"] - sensor_function(spec.kind, truth, record["ticks"], config.dt))
        residual = residual.reshape(-1, spec.R.shape[0])
        S, z_mean, z_cov = _moment_z(residual, spec.R)
        entries += [z_mean, z_cov]
        misstated = _moment_z(residual, UNDERSTATEMENT * spec.R)[2]
        table[spec.name] = {"samples": residual.shape[0], "axes": spec.R.shape[0], "sample_covariance": S,
                            "declared": spec.R, "max_abs_z": float(np.max(np.abs(np.concatenate([z_mean, z_cov])))),
                            "variance_z_if_declared_10pct_low":
                                float(np.max(np.abs(misstated[_diagonal_index(spec.R.shape[0])])))}
    process = (truth[:, 1:] - truth[:, :-1] @ bench["F"].T).reshape(-1, 4)
    S, z_mean, z_cov = _moment_z(process, bench["Q"])
    entries += [z_mean, z_cov]
    table["process_noise"] = {"samples": process.shape[0], "sample_covariance": S, "declared": bench["Q"],
                              "max_abs_z": float(np.max(np.abs(np.concatenate([z_mean, z_cov]))))}
    z_all = np.concatenate(entries)
    family = len(z_all)
    alpha = 1e-3
    z_crit = normal_quantile(1 - alpha / (2 * family))
    max_z = float(np.max(np.abs(z_all)))
    streams = [spec for spec in config.sensors]
    witness = {spec.name: table[spec.name]["variance_z_if_declared_10pct_low"] for spec in streams}

    # Power of the variance test under a 10% understatement, exactly from the chi-square law.
    power = {}
    for spec in streams:
        N = table[spec.name]["samples"]
        axis = variance_power(N, UNDERSTATEMENT, z_crit)
        independent = spec.R.shape[0] == 1 or not np.any(spec.R - np.diag(np.diag(spec.R)))
        # Axes with correlated noise have dependent variance estimates: only bounds on the stream power.
        stream = 1 - (1 - axis) ** spec.R.shape[0] if independent else None
        power[spec.name] = {"samples": N, "per_axis": axis, "stream_exact": stream,
                            "stream_bounds": [axis, min(1.0, spec.R.shape[0] * axis)],
                            "detected_in_this_run": bool(witness[spec.name] >= z_crit)}
    tracker = next(spec for spec in streams if spec.name == "tracker")
    replicates = gaussian(generator(BENCH_SEED + 61), tracker.R, (POWER_REPLICATES, table["tracker"]["samples"]))
    flagged = sum(float(np.max(np.abs(_moment_z(r, UNDERSTATEMENT * tracker.R)[2][_diagonal_index(2)]))) >= z_crit
                  for r in replicates)
    tracker_power = power["tracker"]["stream_exact"]
    replicate = rate_interval(int(flagged), POWER_REPLICATES)
    replicate_se = math.sqrt(tracker_power * (1 - tracker_power) / POWER_REPLICATES)
    replicate_z = (replicate["rate"] - tracker_power) / replicate_se
    fast_axis_power = min(power[name]["per_axis"] for name in ("camera", "encoder", "imu"))

    half_power = {name: {"exact": half_power_misstatement(power[name]["samples"], z_crit),
                         "closed_form": z_crit * math.sqrt(2.0 / power[name]["samples"])} for name in power}
    closed_form_gap = max(abs(row["closed_form"] / row["exact"] - 1.0) for row in half_power.values())
    process_q = _process_noise_study(config.q, config.dt)

    ctx.artifact_json("covariance_moments.json", as_json({"family_size": family, "family_alpha": alpha,
                                                          "z_critical": z_crit, "per_stream": table,
                                                          "understatement_factor": UNDERSTATEMENT,
                                                          "power": power, "tracker_replicates": replicate,
                                                          "half_power_relative_misstatement": half_power,
                                                          "process_noise_structure": process_q}))
    ctx.artifact_text("zscores.svg", svg.line_plot(
        [("|z| per moment", list(range(family)), np.abs(z_all)), ("Bonferroni bound", [0, family - 1], [z_crit] * 2)],
        title="T061 standardized moment deviations", xlabel="moment index", ylabel="|z|", markers=True))
    findings = [
        finding("Sample means and covariances of every sensor residual z - h(truth) and of the process noise "
                "x_{k+1} - F x_k agree with the declared covariances within a Bonferroni-corrected 99.9% Monte "
                "Carlo bound", "numerical", {"max_abs_z": max_z, "z_critical": z_crit, "moments_tested": family},
                {**generator_basis(BENCH_SEED, runs=BENCH_RUNS), "checks": [
                    check("analytic", "Gaussian sampling law Var(S_ij) = (R_ij^2 + R_ii R_jj)/N", max_z, z_crit)]},
                uncertainty=max_abs_z_spread(family, "standardized mean and covariance moments"),
                tolerance=TOL_MC),
        finding("At this sample size a 10% understatement of every declared variance is detected with probability "
                "above 0.999 per variance for the 10-20 Hz streams but only about half the time for the 2 Hz tracker "
                "(exact chi-square power); replicate tracker noise streams reproduce that power", "numerical",
                {"power": power, "tracker_replicate_detection": replicate, "tracker_replicate_z": replicate_z,
                 "z_critical": z_crit, "variance_z_this_run": witness},
                {**generator_basis(BENCH_SEED + 61, replicates=POWER_REPLICATES), "checks": [
                    check("analytic", "smallest exact per-variance power of the camera, encoder and IMU streams",
                          fast_axis_power, 0.999, "ge"),
                    check("analytic", "exact tracker detection power (two independent variances)", tracker_power,
                          0.6, "le"),
                    check("analytic", "exact tracker detection power", tracker_power, 0.4, "ge"),
                    check("analytic", "replicate detection frequency against the exact power (binomial z)",
                          replicate_z, normal_quantile(1 - alpha / 2))]},
                uncertainty=mc95(replicate_se, "binomial standard error of the replicate detection frequency; "
                                               "the exact powers carry chi-square series roundoff below 1e-9"),
                tolerance=TOL_MC, counterexample={
                    "statement": "A covariance check on one bench run detects a 10% misstatement for every sensor",
                    "witness": {"sensor": "tracker", "samples": table["tracker"]["samples"],
                                "detection_power": tracker_power, "z_this_run": witness["tracker"],
                                "detected_this_run": power["tracker"]["detected_in_this_run"],
                                "z_critical": z_crit}}),
        finding("The relative variance misstatement that one variance test flags with probability one half is "
                "z_crit sqrt(2 / N) to within 1% of the exact chi-square value for every stream", "mathematical",
                half_power,
                {"derivation": "E z = sqrt(N/2) (1 - c)/c for declared c R; setting E z = z_crit gives "
                               "delta = (1 - c)/c = z_crit sqrt(2/N)", "checks": [
                    check("analytic", "max relative gap between the closed form and the exact 50%-power level",
                          closed_form_gap, 0.01, "le")]},
                uncertainty=uncertainty("reference_error", closed_form_gap,
                                        "relative gap of the normal closed form to the exact chi-square bisection"),
                tolerance=TOL_ROUNDOFF),
        finding("The declared process-noise covariance Q(dt) equals the continuous-time integral of F(u) G q G^T "
                "F(u)^T and composes as Q(a + b) = F(b) Q(a) F(b)^T + Q(b); the discrete white-noise-acceleration "
                "alternative fails the integral", "numerical", process_q,
                {"derivation": "white-noise acceleration: dx = A x dt + G dW, F(u) = exp(A u)", "checks": [
                    check("analytic", "three-point Gauss-Legendre integral against Q (relative)",
                          process_q["quadrature_relative_error"], 1e-12),
                    check("invariant", "semigroup identity (relative)", process_q["semigroup_relative_error"], 1e-12),
                    check("analytic", "alternative discretization against the integral (relative)",
                          process_q["alternative_model_relative_error"], 0.1, "ge")]},
                uncertainty=roundoff(max(process_q["quadrature_relative_error"],
                                         process_q["semigroup_relative_error"])),
                tolerance=TOL_TINY),
        unreal("Real sensors' noise covariance equals the covariance declared for this bench", "sensor_performance",
               BENCH_SEED, "not established: residuals are synthetic draws from the declared covariance"),
    ]
    fields = {
        "hypothesis": "The empirical residual and process-noise moments reproduce the declared means (zero) and "
                      "covariances within a stated Monte Carlo bound, and the bound's power against a misstated "
                      "covariance is quantified exactly rather than read off one run.",
        "mathematical_model": "For N i.i.d. N(0, R) residuals with known zero mean, S = sum r r^T / N has "
                              "E S = R and Var S_ij = (R_ij^2 + R_ii R_jj)/N; the mean has Var = R_ii / N. "
                              f"{family} standardized moments are tested jointly with a Bonferroni bound at "
                              "family error 1e-3. Power against a declared c R: N S_ii / R_ii ~ chi2(N), so the "
                              "variance test flags with probability P(|chi2(N)/N - c| >= z_crit c sqrt(2/N)). "
                              "Q(dt) = int_0^dt F(u) G q G^T F(u)^T du.",
        "input_data": [f"T060 bench, seed {BENCH_SEED}, {BENCH_RUNS} replicas",
                       ", ".join(f"{k}: N = {v['samples']}" for k, v in table.items()),
                       f"{POWER_REPLICATES} replicate tracker noise streams (seed {BENCH_SEED + 61}), drawn as "
                       "generate_bench draws them"],
        "observation_model": "Residual z - h(truth) recomputed from the retained truth (not the retained noise).",
        "expected_invariant": "max |z| <= z_crit under the declared model; detection power of a misstated "
                              "covariance given by the chi-square law, near 1 when (1 - c)/c exceeds z_crit sqrt(2/N) "
                              "and near 1/2 when they are equal.",
        "experiment": "Compute residual moments per stream and the process-noise moments, standardize, compare "
                      "with the Bonferroni bound; compute the exact power of the variance test against a covariance "
                      "declared 10% low and check it with replicate tracker streams; check Q against the "
                      "continuous-time integral and the semigroup identity.",
        "numerical_result": f"max |z| = {max_z:.3f} vs bound {z_crit:.3f}; under a 10% understatement the exact "
                            f"detection power is {power['tracker']['stream_exact']:.3f} for the tracker (replicates "
                            f"{replicate['rate']:.3f}) and at least {fast_axis_power:.4f} per variance for the other "
                            f"streams; in this run the tracker's variance z is {witness['tracker']:.2f} (not "
                            f"flagged); 50%-power misstatement for the tracker "
                            f"{half_power['tracker']['exact']:.3f}; Q against the integral "
                            f"{process_q['quadrature_relative_error']:.1e} relative.",
        "uncertainty": "The bound has family-wise false-alarm probability 1e-3 under the Gaussian model. Powers are "
                       "exact chi-square probabilities; the replicate frequency carries binomial error (400 "
                       "replicates). Whether one particular run flags the tracker is a coin flip at this N.",
        "failure_modes_checked": ["nonzero residual mean", "misstated variances", "misstated cross-covariance",
                                  "process-noise structure checked against the continuous-time integral and the "
                                  "semigroup identity (the moment test alone compares Q with its own draws)",
                                  "insufficient samples for low-rate sensors (power quantified, not read off "
                                  "one seed)"],
        "unresolved_assumptions": ["Residuals are Gaussian and independent by construction; real residuals "
                                   "may be heavy-tailed, correlated or biased.",
                                   "The heading-rate residual is exact because the truth heading rate is "
                                   "defined as the gyro's own integrated increment.",
                                   "The camera's stream power is bounded, not exact, because its axis variances "
                                   "are correlated (both bounds exceed 0.999)."],
        "recommended_next_task": "T062: test independent versus correlated noise in the filter.",
    }
    return outcome(fields, findings)


# T062 ------------------------------------------------------------------------------
def _whitened(innovations, steps, first):
    """Whitened innovations L^{-1} nu (S = L L^T) from tick ``first`` on, stacked over runs and ticks."""
    out = []
    for nu, step in list(zip(innovations, steps))[first:]:
        if nu is not None:
            L = np.linalg.cholesky(step.S)
            out.append(np.linalg.solve(L, nu.T).T)
    return np.concatenate(out)


def _identity_z(u):
    """Max |z| of the sample covariance of whitened innovations against I (diag var 2/N, off-diag 1/N)."""
    N, m = u.shape
    C = gram(u) / N
    scale = np.where(np.eye(m, dtype=bool), math.sqrt(2.0 / N), math.sqrt(1.0 / N))
    z = (C - np.eye(m)) / scale
    upper = np.triu_indices(m)
    return C, float(np.max(np.abs(z[upper]))), len(upper[0])


def _run_se(values):
    """Standard error of a grand mean from independent runs (per-run time averages)."""
    per_run = values.mean(axis=1)
    return float(per_run.std(ddof=1) / math.sqrt(len(per_run)))


@task("T062", changed_files=FILES, regression_tests=_tests(
    "T062", "test_correlated_noise_consistency_and_ignored_correlation_counterexample"))
def correlated_noise(ctx):
    seed = BENCH_SEED + 62
    dt, q = 0.1, 0.05
    F, Q = cv_model(dt, q)
    I2 = np.eye(2)
    H = np.vstack([H_POS, H_POS])
    common = 0.09
    R_true = np.block([[0.13 * I2, common * I2], [common * I2, 0.13 * I2]])
    R_block = np.block([[0.13 * I2, 0 * I2], [0 * I2, 0.13 * I2]])
    rng = generator(seed)
    truth = simulate_truth(rng, F, Q, MU0, P0_BENCH, MC_RUNS, MC_TICKS)
    ticks = np.arange(1, MC_TICKS + 1)
    z = measure(rng, truth, H, R_true, ticks)
    # Independent sensor noise (no common mode, same marginal variances), drawn after the correlated readings so
    # those are unchanged: the same truth seen through a block-diagonal R.
    z_independent = measure(rng, truth, H, R_block, ticks)
    # (label, filter R, readings, truth R): both truths, each filtered with the right and the wrong assumption.
    cases = (("correct", R_true, z, R_true), ("ignored", R_block, z, R_true),
             ("independent", R_block, z_independent, R_block), ("assumed_correlated", R_true, z_independent, R_block))
    results, curves = {}, {}
    for label, R_filter, readings_all, R_truth in cases:
        readings = [readings_all[:, k] for k in range(MC_TICKS)]
        steps = gain_schedule(F, Q, P0_BENCH, [(H, R_filter)] * MC_TICKS)
        estimates, innovations = run_shared(F, MU0, steps, readings)
        nees = nees_series(estimates, truth, steps)
        nis = np.stack([quadratic(nu, step.S) for nu, step in zip(innovations, steps)], axis=1)
        curves[label] = nees.mean(axis=0)
        predicted = mismatch_moments(F, Q, MU0, P0_BENCH, MU0, steps, [(H, np.zeros(4), R_truth)] * MC_TICKS)
        C, white_z, family = _identity_z(_whitened(innovations, steps, 20))
        rmse = float(np.sqrt(np.mean(np.sum((estimates[:, 1:, :2] - truth[:, 1:, :2]) ** 2, axis=-1))))
        results[label] = {"nees": consistency(nees, 4), "nis": consistency(nis, 4),
                          "nees_se": _run_se(nees), "nis_se": _run_se(nis),
                          "predicted_mean_nees": float(np.mean(predicted["nees"])),
                          "predicted_mean_nis": float(np.mean(predicted["nis"])),
                          "predicted_position_rmse": float(math.sqrt(np.mean(predicted["position_mse"]))),
                          "position_rmse": rmse, "whitened_covariance": C, "whitened_max_z": white_z}
    z_crit = normal_quantile(1 - 1e-3 / (2 * family))
    good, bad = results["correct"], results["ignored"]
    independent, assumed = results["independent"], results["assumed_correlated"]
    ctx.artifact_json("consistency.json", as_json({"seed": seed, "runs": MC_RUNS, "ticks": MC_TICKS,
                                                   "R_true_correlated": R_true, "R_block_diagonal": R_block,
                                                   "cases": {label: {"filter_R": R_filter, "truth_R": R_truth}
                                                             for label, R_filter, _, R_truth in cases},
                                                   "whitened_z_critical": z_crit, "results": results}))
    ctx.artifact_text("anees.svg", svg.line_plot(
        [("correlated noise, correct R", ticks, curves["correct"]),
         ("correlated noise, correlation ignored", ticks, curves["ignored"]),
         ("independent noise, correct R", ticks, curves["independent"]),
         ("independent noise, correlation assumed", ticks, curves["assumed_correlated"]),
         ("99% upper", [1, MC_TICKS], [good["nees"]["interval"][1]] * 2),
         ("99% lower", [1, MC_TICKS], [good["nees"]["interval"][0]] * 2)],
        title="T062 run-averaged NEES (dof 4)", xlabel="tick", ylabel="ANEES", markers=False))
    moments_gap = max(abs(r[key]["grand_mean"] - r[f"predicted_mean_{key}"]) / r[f"{key}_se"]
                      for r in results.values() for key in ("nees", "nis"))

    def consistent(label, claim):
        row = results[label]
        return finding(claim, "numerical", {"nees": row["nees"], "nis": row["nis"],
                                            "whitened_max_z": row["whitened_max_z"], "z_critical": z_crit},
                       {**generator_basis(seed, runs=MC_RUNS, ticks=MC_TICKS), "checks": [
                           check("analytic", "fraction of ticks with ANEES in chi2(4N)/N 99% interval",
                                 row["nees"]["fraction_inside"], 0.9, "ge"),
                           check("analytic", "fraction of ticks with ANIS in chi2(4N)/N 99% interval",
                                 row["nis"]["fraction_inside"], 0.9, "ge"),
                           check("analytic", "whitened innovation covariance vs I (Bonferroni 99.9%)",
                                 row["whitened_max_z"], z_crit, "le")]},
                       uncertainty=mc95(row["nees_se"], "run-level standard error of the grand-mean NEES"),
                       tolerance=TOL_MC)

    findings = [
        consistent("correct", "With the correct cross-correlated measurement covariance, run-averaged NEES and NIS "
                              "lie inside their per-tick 99% chi-square intervals at 90% or more of ticks and the "
                              "whitened innovations have identity covariance"),
        consistent("independent", "With independent camera and tracker noise (no common mode) and the matching "
                                  "block-diagonal covariance, run-averaged NEES and NIS lie inside their per-tick 99% "
                                  "chi-square intervals at 90% or more of ticks and the whitened innovations have "
                                  "identity covariance"),
        finding("Ignoring the camera-tracker cross-correlation makes the filter overconfident: run-averaged NEES "
                "exceeds the 99% upper bound at nearly every tick", "numerical",
                {"nees": bad["nees"], "predicted_mean_nees": bad["predicted_mean_nees"]},
                {**generator_basis(seed), "checks": [
                    check("analytic", "fraction of ticks with ANEES above the 99% upper bound",
                          bad["nees"]["fraction_above"], 0.9, "ge")]},
                uncertainty=mc95(bad["nees_se"], "run-level standard error of the grand-mean NEES"),
                tolerance=TOL_MC, counterexample={
                    "statement": "Ignoring correlation between sensor noises is harmless",
                    "witness": {"common_mode_covariance": common, "grand_mean_nees": bad["nees"]["grand_mean"],
                                "nominal": 4.0}}),
        finding("The ignored-correlation filter still passes the mean-NIS test (grand mean near 4); among "
                "innovation-based tests, which need no ground truth, only the whitened-innovation covariance test "
                "exposes the missing cross-correlation", "numerical",
                {"nis": bad["nis"], "whitened_max_z": bad["whitened_max_z"], "z_critical": z_crit},
                {**generator_basis(seed), "checks": [
                    check("analytic", "fraction of ticks with ANIS inside the 99% interval",
                          bad["nis"]["fraction_inside"], 0.9, "ge"),
                    check("analytic", "whitened innovation covariance deviates from I",
                          bad["whitened_max_z"], z_crit, "ge")]},
                uncertainty=mc95(bad["nis_se"], "run-level standard error of the grand-mean NIS"),
                tolerance=TOL_MC, counterexample={
                    "statement": "A passing mean-NIS chi-square test shows the measurement noise model is correct",
                    "witness": {"grand_mean_nis": bad["nis"]["grand_mean"],
                                "whitened_cross_term": float(bad["whitened_covariance"][0, 2])}}),
        finding("Assuming a common-mode correlation that the noise does not have makes the filter underconfident "
                "(run-averaged NEES below the 99% lower bound at nearly every tick) and less accurate than the "
                "block-diagonal filter, while its NIS exceeds the 99% upper bound at nearly every tick: the reverse "
                "mismatch is neither harmless nor invisible", "numerical",
                {"nees": assumed["nees"], "nis": assumed["nis"],
                 "predicted_mean_nees": assumed["predicted_mean_nees"],
                 "predicted_mean_nis": assumed["predicted_mean_nis"],
                 "predicted_position_rmse": assumed["predicted_position_rmse"],
                 "predicted_position_rmse_block_diagonal": independent["predicted_position_rmse"],
                 "position_rmse": assumed["position_rmse"],
                 "position_rmse_block_diagonal": independent["position_rmse"]},
                {**generator_basis(seed), "checks": [
                    check("analytic", "fraction of ticks with ANEES below the 99% lower bound",
                          assumed["nees"]["fraction_below"], 0.9, "ge"),
                    check("analytic", "fraction of ticks with ANIS above the 99% upper bound",
                          assumed["nis"]["fraction_above"], 0.9, "ge"),
                    check("analytic", "exact position RMSE (moment recursion) minus that of the block-diagonal "
                                      "filter", assumed["predicted_position_rmse"]
                          - independent["predicted_position_rmse"], 0.0, "signed_ge")]},
                uncertainty=mc95(assumed["nis_se"], "run-level standard error of the grand-mean NIS"),
                tolerance=TOL_MC, counterexample={
                    "statement": "Assuming correlation between sensor noises where there is none is a conservative, "
                                 "harmless choice",
                    "witness": {"common_mode_assumed": common, "grand_mean_nees": assumed["nees"]["grand_mean"],
                                "grand_mean_nis": assumed["nis"]["grand_mean"], "nominal": 4.0,
                                "predicted_position_rmse": assumed["predicted_position_rmse"],
                                "predicted_position_rmse_block_diagonal": independent["predicted_position_rmse"]}}),
        finding("Monte Carlo grand-mean NEES and NIS of all four filters agree with the exact moment recursion "
                "for tr(P_f^-1 E[e e^T]) and tr(S_f^-1 E[nu nu^T]) within 4 run-level standard errors", "numerical",
                {"max_gap_in_standard_errors": moments_gap,
                 "predicted": {k: {"nees": r["predicted_mean_nees"], "nis": r["predicted_mean_nis"],
                                   "position_rmse": r["predicted_position_rmse"]} for k, r in results.items()},
                 "observed_position_rmse": {k: r["position_rmse"] for k, r in results.items()}},
                {**generator_basis(seed), "checks": [
                    check("analytic", "mismatch_moments exact propagation", moments_gap, 4.0, "le")]},
                uncertainty=mc95(max(r[f"{key}_se"] for r in results.values() for key in ("nees", "nis")),
                                 "largest run-level standard error among the eight grand means"),
                tolerance=TOL_MC),
        unreal("Real camera and tracker noises share the common-mode covariance assumed here, or are independent",
               "sensor_performance", seed, "not established: the cross-correlation is a declared synthetic parameter"),
    ]
    fields = {
        "hypothesis": "NEES/NIS consistency holds for a filter whose R matches the truth, whether the sensor noises "
                      "are correlated or independent, and fails in either direction of mismatch: dropping a real "
                      "cross-covariance makes the filter overconfident while its mean NIS still passes, and assuming "
                      "an absent one makes it underconfident while its NIS fails; the size of each failure follows "
                      "from an exact moment recursion.",
        "mathematical_model": "Camera and tracker positions z_c = p + e_m + e_c, z_t = p + e_m + e_t. Correlated "
                              "truth: Var e_c = Var e_t = 0.04 I, common mode Var e_m = 0.09 I, R_true = [[0.13 I, "
                              "0.09 I], [0.09 I, 0.13 I]]. Independent truth: no common mode, R = blockdiag(0.13 I, "
                              "0.13 I) (same marginal variances). Each truth is filtered with both R. CV motion "
                              "dt = 0.1 s, q = 0.05. Mismatched moments: joint [x; x_hat] mean and covariance "
                              "propagated exactly tick by tick (mismatch_moments).",
        "input_data": [f"seed {seed}", f"{MC_RUNS} runs x {MC_TICKS} ticks", "x0 ~ N((0,0,1,0.5), P0)",
                       "one set of correlated readings and one set of independent readings of the same truth"],
        "observation_model": "Stacked 4-vector measurement at every tick; filter prior equals the truth prior.",
        "expected_invariant": "Matching R: E NEES = 4, E NIS = 4, whitened innovations ~ N(0, I). Ignored "
                              f"correlation: E NEES = {bad['predicted_mean_nees']:.2f}, E NIS = "
                              f"{bad['predicted_mean_nis']:.2f}. Assumed correlation: E NEES = "
                              f"{assumed['predicted_mean_nees']:.2f}, E NIS = {assumed['predicted_mean_nis']:.2f}.",
        "experiment": "Run the matching and the mismatched filter on each set of readings; per-tick ANEES/ANIS "
                      "against chi2(4N)/N 99% intervals; sample covariance of Cholesky-whitened innovations "
                      "(ticks 21-100) against I; grand means against the exact moment recursion.",
        "numerical_result": f"Correlated noise, correct R: ANEES inside {good['nees']['fraction_inside']:.2f} of "
                            f"ticks, grand NEES {good['nees']['grand_mean']:.3f}, NIS {good['nis']['grand_mean']:.3f}. "
                            f"Correlation ignored: ANEES above bound {bad['nees']['fraction_above']:.2f} of ticks, "
                            f"grand NEES {bad['nees']['grand_mean']:.3f}; grand NIS {bad['nis']['grand_mean']:.3f} "
                            f"(passes); whitened max |z| {bad['whitened_max_z']:.1f} vs {z_crit:.2f}. Independent "
                            f"noise, block-diagonal R: ANEES inside {independent['nees']['fraction_inside']:.2f}, "
                            f"grand NEES {independent['nees']['grand_mean']:.3f}. Correlation assumed: ANEES below "
                            f"bound {assumed['nees']['fraction_below']:.2f} of ticks (grand "
                            f"{assumed['nees']['grand_mean']:.3f}), ANIS above bound "
                            f"{assumed['nis']['fraction_above']:.2f} (grand {assumed['nis']['grand_mean']:.3f}), "
                            f"exact RMSE {assumed['predicted_position_rmse']:.4f} m vs "
                            f"{independent['predicted_position_rmse']:.4f} m.",
        "uncertainty": "Per-tick intervals are exact chi-square quantiles; the fraction-inside threshold 0.9 "
                       "allows for time correlation of NEES. Moment agreement uses run-level standard errors.",
        "failure_modes_checked": ["dropped cross-covariance", "assumed cross-covariance that is absent",
                                  "NIS blind spot (trace insensitive to off-diagonal blocks)",
                                  "filter/truth prior mismatch (excluded by construction)",
                                  "prediction-vs-simulation disagreement"],
        "unresolved_assumptions": ["White common-mode error; a slowly varying common bias would defeat both "
                                   "filters and needs state augmentation.",
                                   "NEES needs ground truth, which a deployed system does not have."],
        "recommended_next_task": "T066: show that residual consistency needs the filter covariance S, not raw R.",
    }
    return outcome(fields, findings)


# Tasks T063-T076 live in sibling modules; importing them registers them with
# the queue when the runner imports this section module. A sibling that fails
# to import leaves placeholders that report the error as blocked, so one broken
# module (or a broken core module it uses) cannot defer the whole section.
SIBLINGS = {"sensor_fusion_geometry": ("T063", "T064"),
            "sensor_fusion_filtering": ("T065", "T066", "T067", "T068"),
            "sensor_fusion_robustness": ("T069", "T070", "T071"),
            "sensor_fusion_admission": ("T072", "T073", "T074", "T075", "T076")}
IMPORT_ERRORS: dict = {}


def _placeholder(module: str, error: str):
    def blocked(ctx):
        raise ImportError(f"Sibling module {module} failed to import: {error}")
    return blocked


# The registry's importer rolls back a failed module's partial registrations, as it does for sections.
_import_registering = getattr(registry, "_import", import_module)
for _module, _ids in SIBLINGS.items():
    try:
        _import_registering(f"{__package__}.{_module}")
    except Exception as exc:  # retained as each task's blocked reason
        IMPORT_ERRORS[_module] = f"{type(exc).__name__}: {exc}"
        for _task_id in _ids:
            try:
                task(_task_id, changed_files=files(_module))(_placeholder(_module, IMPORT_ERRORS[_module]))
            except ValueError:
                pass  # a registration survived the failure (plain import_module fallback); keep it
