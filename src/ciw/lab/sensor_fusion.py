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
Every mismatched-filter counterexample is predicted exactly by
:func:`ciw.lab.sensor_fusion_bench.mismatch_moments` before it is simulated.

Non-claims: all readings are synthetic draws from declared distributions.
Nothing here measures a real sensor, validates a calibration, certifies a
clock, a frame, a safety threshold or production acceptance; those claims are
recorded as ``not_established`` findings in physical and authority domains.
"""
from __future__ import annotations

import math

import numpy as np

from . import svg
from .evidence import finding
from .registry import task
from .sensor_fusion_bench import (BENCH_SEED, H_POS, BenchConfig, bench_digest, consistency, cv_model,
                                  gain_schedule, generate_bench, generator, measure, mismatch_moments,
                                  nees_series, normal_quantile, quadratic, run_shared, sensor_function,
                                  simulate_truth, with_sensor)
from .sensor_fusion_common import (TESTS, TOL_EXACT, TOL_MC, TOL_ROUNDOFF, as_json, check, files,
                                   generator_basis, outcome, unreal)

FILES = files("sensor_fusion")
BENCH_RUNS = 50
MC_RUNS = 200
MC_TICKS = 100


def _cv_plan(H, R, ticks, every=1):
    return [(H, R) if k % every == 0 else None for k in range(1, ticks + 1)]


P0_BENCH = np.diag([0.25, 0.25, 0.04, 0.04])
MU0 = np.array([0.0, 0.0, 1.0, 0.5])


def _bench(ctx):
    return ctx.memo("sensor_fusion.bench", lambda: generate_bench(BenchConfig(), BENCH_SEED, BENCH_RUNS))


# T060 ------------------------------------------------------------------------------
@task("T060", changed_files=FILES, regression_tests=(
    f"{TESTS}::test_bench_is_deterministic_rate_exact_and_stream_independent",
    f"{TESTS}::test_section_reports_labels_and_states"))
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

    # Independent recomputation of each sensor function from the retained truth.
    truth = bench["truth"]
    composition = 0.0
    for spec in config.sensors:
        record = bench["readings"][spec.name]
        ticks = record["ticks"]
        if spec.kind == "position":
            clean = truth[:, ticks, :2]
        elif spec.kind == "speed":
            clean = np.hypot(truth[:, ticks, 2], truth[:, ticks, 3])[..., None]
        else:
            now = truth[:, ticks, 2] + 1j * truth[:, ticks, 3]
            before = truth[:, ticks - 1, 2] + 1j * truth[:, ticks - 1, 3]
            clean = (np.angle(now * np.conj(before)) / config.dt)[..., None]
        composition = max(composition, float(np.max(np.abs(record["values"] - clean - record["noise"]))))
    min_eigen = min(float(np.linalg.eigvalsh(spec.R).min()) for spec in config.sensors)

    run0 = {"truth": bench["truth"][0], "readings": {n: {"ticks": bench["readings"][n]["ticks"],
                                                         "values": bench["readings"][n]["values"][0]} for n in names}}
    ctx.artifactas_json("bench.json", as_json({"config": config.describe(), "seed": BENCH_SEED, "runs": BENCH_RUNS,
                                           "retained_run": 0, "run": run0,
                                           "declared_covariances": {s.name: s.covariance for s in config.sensors},
                                           "process_noise_Q_per_tick": bench["Q"]}))
    ctx.artifactas_json("digests.json", {"seed": BENCH_SEED, "digest": bench_digest(bench),
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
                tolerance=TOL_EXACT),
        finding("Every sensor reports exactly at its declared rate on the 20 Hz base clock over 20 s "
                "(camera 10 Hz, encoder 20 Hz, IMU 20 Hz, tracker 2 Hz)", "computational_pipeline",
                {"per_sensor": counts, "max_count_mismatch": count_mismatch, "off_grid_ticks": offgrid},
                {**generator_basis(BENCH_SEED), "checks": [
                    check("exact_arithmetic", "declared rate times duration", count_mismatch, 0),
                    check("exact_arithmetic", "reading ticks on the declared grid", offgrid, 0)]},
                tolerance=TOL_EXACT),
        finding("Every raw reading equals the noise-free sensor function of the retained truth, recomputed by an "
                "independent code path, plus its retained noise draw", "numerical", composition,
                {**generator_basis(BENCH_SEED), "checks": [
                    check("invariant", "hypot and complex-angle recomputation of h(truth)", composition, 1e-12)]},
                unit="max abs difference", tolerance={"abs": 1e-12, "rel": 0.0}),
        finding("Changing the tracker rate from 2 Hz to 4 Hz leaves the truth and the camera, encoder and IMU "
                "streams unchanged (independent spawned PCG64 streams)", "computational_pipeline", stream_changes,
                {**generator_basis(BENCH_SEED), "checks": [
                    check("invariant", "streams other than the tracker", stream_changes, 0)]},
                tolerance=TOL_EXACT),
        finding("Every declared sensor covariance is symmetric positive definite", "numerical", min_eigen,
                {"derivation": "declared constants in DEFAULT_SENSORS", "checks": [
                    check("analytic", "smallest eigenvalue of the declared covariances", min_eigen, 1e-6, "ge")]},
                unit="smallest eigenvalue", tolerance=TOL_ROUNDOFF),
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
                      "arrays bitwise; recompute h(truth) with hypot and complex angles; count reading ticks.",
        "numerical_result": f"0 of {len(names) + 1} arrays differ on regeneration; all differ under seed+1; "
                            f"counts {count_text}; "
                            f"max composition residual {composition:.2e}; tracker retiming changed "
                            f"{stream_changes} other streams.",
        "uncertainty": "None for the determinism and rate claims (exact). The composition residual is floating-"
                       "point roundoff. Cross-platform byte identity of the digest is not claimed (libm may "
                       "differ in the last bit); findings avoid digest values for that reason.",
        "failure_modes_checked": ["seed reuse and seed change", "sensor-rate edits leaking into other streams",
                                  "off-grid timestamps", "angle wrap in the heading-rate model",
                                  "non-positive-definite declared covariance"],
        "unresolved_assumptions": ["White-noise acceleration is a modelling choice, not observed vehicle motion.",
                                   "The heading-rate truth of white-noise-acceleration motion is rough; the IMU "
                                   "stream is kept for completeness but the linear experiments use positions.",
                                   "Encoder and IMU are nonlinear in the state; no experiment here fuses them."],
        "recommended_next_task": "T061: verify the empirical noise covariance against the declared covariance.",
    }
    return outcome(fields, findings)


# T061 ------------------------------------------------------------------------------
def _moment_z(residuals, R):
    """Standardized deviations of the sample mean and second moment from (0, R), known mean zero.

    Var(S_ij) = (R_ij^2 + R_ii R_jj) / N for Gaussian residuals.
    """
    N, m = residuals.shape
    S = residuals.T @ residuals / N
    diag = np.diag(R)
    z_cov = (S - R) / np.sqrt((R ** 2 + np.outer(diag, diag)) / N)
    z_mean = residuals.mean(axis=0) / np.sqrt(diag / N)
    upper = np.triu_indices(m)
    return S, z_mean, z_cov[upper]


@task("T061", changed_files=FILES, regression_tests=(
    f"{TESTS}::test_declared_covariance_matches_and_low_rate_misstatement_escapes",
    f"{TESTS}::test_section_reports_labels_and_states"))
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
        misstated = _moment_z(residual, 0.9 * spec.R)[2]
        diag_index = np.cumsum([0] + list(range(spec.R.shape[0], 1, -1)))
        table[spec.name] = {"samples": residual.shape[0], "sample_covariance": S, "declared": spec.R,
                            "max_abs_z": float(np.max(np.abs(np.concatenate([z_mean, z_cov])))),
                            "variance_z_if_declared_10pct_low": float(np.max(np.abs(misstated[diag_index])))}
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
    detect = {name: row["variance_z_if_declared_10pct_low"] for name, row in table.items() if name != "process_noise"}
    minimal = {name: z_crit * math.sqrt(2.0 / table[name]["samples"]) for name in detect}
    ctx.artifactas_json("covariance_moments.json", as_json({"family_size": family, "family_alpha": alpha,
                                                        "z_critical": z_crit, "per_stream": table,
                                                        "minimal_detectable_relative_variance_error": minimal}))
    ctx.artifact_text("zscores.svg", svg.line_plot(
        [("|z| per moment", list(range(family)), np.abs(z_all)), ("Bonferroni bound", [0, family - 1], [z_crit] * 2)],
        title="T061 standardized moment deviations", xlabel="moment index", ylabel="|z|", markers=True))
    findings = [
        finding("Sample means and covariances of every sensor residual z - h(truth) and of the process noise "
                "x_{k+1} - F x_k agree with the declared covariances within a Bonferroni-corrected 99.9% Monte "
                "Carlo bound", "numerical", {"max_abs_z": max_z, "z_critical": z_crit, "moments_tested": family},
                {**generator_basis(BENCH_SEED, runs=BENCH_RUNS), "checks": [
                    check("analytic", "Gaussian sampling law Var(S_ij) = (R_ij^2 + R_ii R_jj)/N", max_z, z_crit)]},
                tolerance=TOL_MC),
        finding("A 10% understatement of every declared variance is detected at this sample size for the 10-20 Hz "
                "streams but escapes detection for the 2 Hz tracker", "numerical", detect,
                {**generator_basis(BENCH_SEED), "checks": [
                    check("analytic", "tracker variance z under a 10% understatement (not detected)",
                           detect["tracker"], z_crit, "le"),
                    check("analytic", "camera variance z under a 10% understatement (detected)",
                           detect["camera"], z_crit, "ge")]},
                tolerance=TOL_MC, counterexample={
                    "statement": "A covariance check on one bench run detects a 10% misstatement for every sensor",
                    "witness": {"sensor": "tracker", "samples": table["tracker"]["samples"],
                                "z": detect["tracker"], "z_critical": z_crit}}),
        finding("Minimal relative variance misstatement detectable per stream at the family bound, "
                "z_crit * sqrt(2 / N)", "mathematical", minimal,
                {"derivation": "delta = z_crit sqrt(2/N) from Var(S_ii) = 2 R_ii^2 / N"}, tolerance=TOL_ROUNDOFF),
        unreal("Real sensors' noise covariance equals the covariance declared for this bench", "sensor_performance",
                BENCH_SEED, "not established: residuals are synthetic draws from the declared covariance"),
    ]
    fields = {
        "hypothesis": "The empirical residual and process-noise moments reproduce the declared means (zero) and "
                      "covariances within a stated Monte Carlo bound, and the bound's power is quantified.",
        "mathematical_model": "For N i.i.d. N(0, R) residuals with known zero mean, S = sum r r^T / N has "
                              "E S = R and Var S_ij = (R_ij^2 + R_ii R_jj)/N; the mean has Var = R_ii / N. "
                              f"{family} standardized moments are tested jointly with a Bonferroni bound at "
                              "family error 1e-3.",
        "input_data": [f"T060 bench, seed {BENCH_SEED}, {BENCH_RUNS} replicas",
                       ", ".join(f"{k}: N = {v['samples']}" for k, v in table.items())],
        "observation_model": "Residual z - h(truth) recomputed from the retained truth (not the retained noise).",
        "expected_invariant": "max |z| <= z_crit under the declared model; a misstated covariance is detected "
                              "when N is large enough that (1 - c)/c > z_crit sqrt(2/N).",
        "experiment": "Compute residual moments per stream and the process-noise moments, standardize, compare "
                      "with the Bonferroni bound; repeat against a covariance declared 10% low.",
        "numerical_result": f"max |z| = {max_z:.3f} vs bound {z_crit:.3f}; under a 10% understatement the camera "
                            f"variance z is {detect['camera']:.2f} (detected), the tracker's {detect['tracker']:.2f} "
                            f"(missed).",
        "uncertainty": "The bound has family-wise false-alarm probability 1e-3 under the Gaussian model; the "
                       "power statement is a single-seed realization plus the analytic minimal detectable error "
                       "(tracker: about 13%).",
        "failure_modes_checked": ["nonzero residual mean", "misstated variances", "misstated cross-covariance",
                                  "process-noise discretization error (Q structure dt^3/3, dt^2/2, dt)",
                                  "insufficient samples for low-rate sensors"],
        "unresolved_assumptions": ["Residuals are Gaussian and independent by construction; real residuals "
                                   "may be heavy-tailed, correlated or biased.",
                                   "The heading-rate residual is exact because the truth heading rate is "
                                   "defined as the gyro's own integrated increment."],
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
    C = u.T @ u / N
    scale = np.where(np.eye(m, dtype=bool), math.sqrt(2.0 / N), math.sqrt(1.0 / N))
    z = (C - np.eye(m)) / scale
    upper = np.triu_indices(m)
    return C, float(np.max(np.abs(z[upper]))), len(upper[0])


def _run_se(values):
    """Standard error of a grand mean from independent runs (per-run time averages)."""
    per_run = values.mean(axis=1)
    return float(per_run.std(ddof=1) / math.sqrt(len(per_run)))


@task("T062", changed_files=FILES, regression_tests=(
    f"{TESTS}::test_correlated_noise_consistency_and_ignored_correlation_counterexample",
    f"{TESTS}::test_section_reports_labels_and_states"))
def correlated_noise(ctx):
    seed = BENCH_SEED + 62
    dt, q = 0.1, 0.05
    F, Q = cv_model(dt, q)
    I2 = np.eye(2)
    H = np.vstack([H_POS, H_POS])
    common = 0.09
    R_true = np.block([[0.13 * I2, common * I2], [common * I2, 0.13 * I2]])
    R_ignored = np.block([[0.13 * I2, 0 * I2], [0 * I2, 0.13 * I2]])
    rng = generator(seed)
    truth = simulate_truth(rng, F, Q, MU0, P0_BENCH, MC_RUNS, MC_TICKS)
    ticks = np.arange(1, MC_TICKS + 1)
    z = measure(rng, truth, H, R_true, ticks)
    readings = [z[:, k] for k in range(MC_TICKS)]
    results, curves = {}, {}
    for label, R_filter in (("correct", R_true), ("ignored", R_ignored)):
        steps = gain_schedule(F, Q, P0_BENCH, [(H, R_filter)] * MC_TICKS)
        estimates, innovations = run_shared(F, MU0, steps, readings)
        nees = nees_series(estimates, truth, steps)
        nis = np.stack([quadratic(nu, step.S) for nu, step in zip(innovations, steps)], axis=1)
        curves[label] = nees.mean(axis=0)
        predicted = mismatch_moments(F, Q, MU0, P0_BENCH, MU0, steps, [(H, np.zeros(4), R_true)] * MC_TICKS)
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
    ctx.artifactas_json("consistency.json", as_json({"seed": seed, "runs": MC_RUNS, "ticks": MC_TICKS,
                                                 "R_true": R_true, "R_ignored": R_ignored,
                                                 "whitened_z_critical": z_crit, "results": results}))
    ctx.artifact_text("anees.svg", svg.line_plot(
        [("correct model", ticks, curves["correct"]), ("ignored correlation", ticks, curves["ignored"]),
         ("99% upper", [1, MC_TICKS], [good["nees"]["interval"][1]] * 2),
         ("99% lower", [1, MC_TICKS], [good["nees"]["interval"][0]] * 2)],
        title="T062 run-averaged NEES (dof 4)", xlabel="tick", ylabel="ANEES", markers=False))
    moments_gap = max(abs(r[key]["grand_mean"] - r[f"predicted_mean_{key}"]) / r[f"{key}_se"]
                      for r in results.values() for key in ("nees", "nis"))
    findings = [
        finding("With the correct cross-correlated measurement covariance, run-averaged NEES and NIS lie inside "
                "their 99% chi-square intervals and the whitened innovations have identity covariance",
                "numerical", {"nees": good["nees"], "nis": good["nis"], "whitened_max_z": good["whitened_max_z"],
                              "z_critical": z_crit},
                {**generator_basis(seed, runs=MC_RUNS, ticks=MC_TICKS), "checks": [
                    check("analytic", "fraction of ticks with ANEES in chi2(4N)/N 99% interval",
                           good["nees"]["fraction_inside"], 0.9, "ge"),
                    check("analytic", "fraction of ticks with ANIS in chi2(4N)/N 99% interval",
                           good["nis"]["fraction_inside"], 0.9, "ge"),
                    check("analytic", "whitened innovation covariance vs I (Bonferroni 99.9%)",
                           good["whitened_max_z"], z_crit, "le")]},
                tolerance=TOL_MC),
        finding("Ignoring the camera-tracker cross-correlation makes the filter overconfident: run-averaged NEES "
                "exceeds the 99% upper bound at nearly every tick", "numerical",
                {"nees": bad["nees"], "predicted_mean_nees": bad["predicted_mean_nees"]},
                {**generator_basis(seed), "checks": [
                    check("analytic", "fraction of ticks with ANEES above the 99% upper bound",
                           bad["nees"]["fraction_above"], 0.9, "ge")]},
                tolerance=TOL_MC, counterexample={
                    "statement": "Ignoring correlation between sensor noises is harmless",
                    "witness": {"common_mode_covariance": common, "grand_mean_nees": bad["nees"]["grand_mean"],
                                "nominal": 4.0}}),
        finding("The ignored-correlation filter still passes the mean-NIS test (grand mean near 4); only the full "
                "whitened-innovation covariance test exposes the missing cross-correlation", "numerical",
                {"nis": bad["nis"], "whitened_max_z": bad["whitened_max_z"], "z_critical": z_crit},
                {**generator_basis(seed), "checks": [
                    check("analytic", "fraction of ticks with ANIS inside the 99% interval",
                           bad["nis"]["fraction_inside"], 0.9, "ge"),
                    check("analytic", "whitened innovation covariance deviates from I",
                           bad["whitened_max_z"], z_crit, "ge")]},
                tolerance=TOL_MC, counterexample={
                    "statement": "A passing mean-NIS chi-square test shows the measurement noise model is correct",
                    "witness": {"grand_mean_nis": bad["nis"]["grand_mean"],
                                "whitened_cross_term": float(bad["whitened_covariance"][0, 2])}}),
        finding("Monte Carlo grand-mean NEES and NIS of both filters agree with the exact second-moment prediction "
                "tr(P_f^-1 E[e e^T]) and tr(S_f^-1 E[nu nu^T]) within 4 run-level standard errors", "numerical",
                {"max_gap_in_standard_errors": moments_gap,
                 "predicted": {k: {"nees": r["predicted_mean_nees"], "nis": r["predicted_mean_nis"],
                                   "position_rmse": r["predicted_position_rmse"]} for k, r in results.items()},
                 "observed_position_rmse": {k: r["position_rmse"] for k, r in results.items()}},
                {**generator_basis(seed), "checks": [
                    check("analytic", "mismatch_moments exact propagation", moments_gap, 4.0, "le")]},
                tolerance=TOL_MC),
        unreal("Real camera and tracker noises share the common-mode covariance assumed here", "sensor_performance",
                seed, "not established: the cross-correlation is a declared synthetic parameter"),
    ]
    fields = {
        "hypothesis": "NEES/NIS consistency holds for a filter with the correct cross-correlated R and fails for "
                      "one that drops the cross-covariance; the failure size is predictable in closed form.",
        "mathematical_model": "Camera and tracker positions z_c = p + e_m + e_c, z_t = p + e_m + e_t with "
                              "Var e_c = Var e_t = 0.04 I, common mode Var e_m = 0.09 I: R_true = [[0.13 I, 0.09 I],"
                              " [0.09 I, 0.13 I]]; the ignoring filter uses blockdiag(0.13 I, 0.13 I). CV motion "
                              "dt = 0.1 s, q = 0.05. Mismatched moments: joint [x; x_hat] propagated exactly.",
        "input_data": [f"seed {seed}", f"{MC_RUNS} runs x {MC_TICKS} ticks", "x0 ~ N((0,0,1,0.5), P0)"],
        "observation_model": "Stacked 4-vector measurement at every tick; filter prior equals the truth prior.",
        "expected_invariant": "Correct model: E NEES = 4, E NIS = 4, whitened innovations ~ N(0, I). Ignored "
                              f"model: E NEES = tr(P^-1 E ee^T) = {bad['predicted_mean_nees']:.2f}, E NIS = "
                              f"{bad['predicted_mean_nis']:.2f}.",
        "experiment": "Run both filters on the same readings; per-tick ANEES/ANIS against chi2(4N)/N 99% intervals; "
                      "sample covariance of Cholesky-whitened innovations (ticks 21-100) against I.",
        "numerical_result": f"Correct: ANEES inside {good['nees']['fraction_inside']:.2f} of ticks, grand NEES "
                            f"{good['nees']['grand_mean']:.3f}, NIS {good['nis']['grand_mean']:.3f}. Ignored: "
                            f"ANEES above bound {bad['nees']['fraction_above']:.2f} of ticks, grand NEES "
                            f"{bad['nees']['grand_mean']:.3f}; grand NIS {bad['nis']['grand_mean']:.3f} (passes); "
                            f"whitened max |z| {bad['whitened_max_z']:.1f} vs {z_crit:.2f}.",
        "uncertainty": "Per-tick intervals are exact chi-square quantiles; the fraction-inside threshold 0.9 "
                       "allows for time correlation of NEES. Moment agreement uses run-level standard errors.",
        "failure_modes_checked": ["dropped cross-covariance", "NIS blind spot (trace insensitive to off-diagonal "
                                  "blocks)", "filter/truth prior mismatch (excluded by construction)",
                                  "prediction-vs-simulation disagreement"],
        "unresolved_assumptions": ["White common-mode error; a slowly varying common bias would defeat both "
                                   "filters and needs state augmentation.",
                                   "NEES needs ground truth, which a deployed system does not have."],
        "recommended_next_task": "T066: show that residual consistency needs the filter covariance S, not raw R.",
    }
    return outcome(fields, findings)
