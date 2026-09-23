"""Filter-induced correlation, residual covariance, gating and outliers (T065-T068).

Scope: T065 derives the steady-state cross-covariance of successive filtered
errors, Cov(e_{k+j}, e_k) = [(I - K H) F]^j P, exactly for a rational scalar
random walk and numerically for the planar constant-velocity bench, and shows
the cost of treating filtered outputs as independent. T066 shows that
innovations are normalized by S = H P- H^T + R and post-fit residuals by
R - H P+ H^T, not by the raw sensor R. T067 checks chi-square gate
false-rejection rates, open loop and closed loop. T068 injects outliers of two
sizes and measures detection, false alarms and estimate error with and
without rejection.

Non-claims: every reading is a synthetic Gaussian draw, and the outliers are a
declared contamination model. Rates and errors here say nothing about a real
sensor's noise, outlier process or the performance of a deployed gate.
"""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np

from . import svg
from .evidence import finding
from .registry import task
from .sensor_fusion_bench import (H_POS, chi2_cdf, chi2_quantile, consistency, cv_model, gain_schedule, generator,
                                  measure, nees_series, noncentral_chi2_2_cdf, noncentral_chi2_2_cdf_many,
                                  normal_quantile, quadratic, run_gated, run_shared, simulate_truth)
from .sensor_fusion_common import (MU0, P0_BENCH, R_CAMERA, TESTS, TOL_EXACT, TOL_MC, TOL_ROUNDOFF, TOL_TINY,
                                   as_json, bonferroni, check, files, generator_basis, outcome, rate_interval,
                                   run_mean_z, unreal)

FILES = files("sensor_fusion_filtering")
DT = 0.1


def _ncdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


# T065 ------------------------------------------------------------------------------
def rational_steady_state(q: Fraction, r: Fraction) -> dict:
    """Scalar random-walk Kalman steady state in exact arithmetic.

    The prior variance M solves M^2 = q M + q r; q and r are chosen so that
    q^2 + 4 q r is a perfect square and every quantity is rational.
    """
    disc = q * q + 4 * q * r
    root = Fraction(math.isqrt(disc.numerator), math.isqrt(disc.denominator))
    if root * root != disc:
        raise ValueError("Scalar steady state is irrational for this q and r")
    M = (q + root) / 2
    P = M * r / (M + r)
    K = M / (M + r)
    return {"M": M, "P": P, "K": K, "a": 1 - K, "riccati_residual": (P + q) * r / (P + q + r) - P}


def average_variance(P: Fraction, a: Fraction, n: int) -> Fraction:
    """Var of the mean of n successive stationary errors with autocovariance a^j P."""
    return P / (n * n) * (n + 2 * sum((n - j) * a ** j for j in range(1, n)))


def steady_state(F, Q, H, R, iterations=4000):
    """Steady-state posterior covariance by iterating the Riccati recursion, and its fixed-point residual."""
    P = np.array(P0_BENCH)
    for _ in range(iterations):
        prior = F @ P @ F.T + Q
        S = H @ prior @ H.T + R
        K = np.linalg.solve(S, H @ prior).T
        P_new = (np.eye(len(P)) - K @ H) @ prior
        P_new = 0.5 * (P_new + P_new.T)
        if np.max(np.abs(P_new - P)) <= 1e-16 * np.max(np.abs(P)):
            P = P_new
            break
        P = P_new
    prior = F @ P @ F.T + Q
    S = H @ prior @ H.T + R
    K = np.linalg.solve(S, H @ prior).T
    residual = float(np.max(np.abs((np.eye(len(P)) - K @ H) @ prior - P)) / np.max(np.abs(P)))
    return P, K, residual


def correlation_study(seed: int = 65_2026) -> dict:
    rng = generator(seed)
    q, r, n, base = Fraction(1), Fraction(30), 20, 10
    exact = rational_steady_state(q, r)
    P, K, a = float(exact["P"]), float(exact["K"]), float(exact["a"])
    V_n = average_variance(exact["P"], exact["a"], n)
    runs, ticks = 20_000, base + n
    # Start at the steady state: filter mean 0, truth error ~ N(0, P); every tick uses gain K.
    x = math.sqrt(P) * rng.standard_normal(runs)
    xhat = np.zeros(runs)
    w = math.sqrt(float(q)) * rng.standard_normal((runs, ticks))
    v = math.sqrt(float(r)) * rng.standard_normal((runs, ticks))
    errors = np.empty((runs, ticks + 1))
    innovations = np.empty((runs, ticks))
    errors[:, 0] = x - xhat
    for k in range(ticks):
        x = x + w[:, k]
        nu = x + v[:, k] - xhat
        xhat = xhat + K * nu
        errors[:, k + 1] = x - xhat
        innovations[:, k] = nu
    lags = np.arange(0, 9)
    autocov = np.array([np.mean(errors[:, base] * errors[:, base + j]) for j in lags])
    predicted = P * a ** lags
    z_auto = (autocov - predicted) / np.sqrt((P * P + predicted ** 2) / runs)
    S = float(exact["M"] + r)
    white = np.array([np.mean(innovations[:, base] * innovations[:, base + j]) / S for j in range(1, 6)])
    z_white = white * math.sqrt(runs)
    window = errors[:, 1:n + 1].mean(axis=1)
    mc_var = float(np.mean(window ** 2))
    V = float(V_n)
    z_avg = (mc_var - V) / (V * math.sqrt(2.0 / runs))
    naive = P / n
    half = normal_quantile(0.975) * math.sqrt(naive)
    coverage_mc = float(np.mean(np.abs(window) <= half))
    coverage_exact = 2 * _ncdf(half / math.sqrt(V)) - 1

    # Planar constant-velocity bench with the correlated camera: matrix lag cross-covariance.
    F, Q = cv_model(DT, 0.05)
    P_ss, K_ss, residual = steady_state(F, Q, H_POS, R_CAMERA)
    A = (np.eye(4) - K_ss @ H_POS) @ F
    cv_runs, cv_ticks = 4000, base + 6
    truth = simulate_truth(rng, F, Q, np.zeros(4), P_ss, cv_runs, cv_ticks)
    z_cam = measure(rng, truth, H_POS, R_CAMERA, np.arange(1, cv_ticks + 1))
    steps = gain_schedule(F, Q, P_ss, [(H_POS, R_CAMERA)] * cv_ticks)
    estimates, _ = run_shared(F, np.zeros(4), steps, [z_cam[:, k] for k in range(cv_ticks)])
    e = truth - estimates
    z_matrix, lag_table = [], []
    for j in range(0, 6):
        C = e[:, base + j].T @ e[:, base] / cv_runs
        C_pred = np.linalg.matrix_power(A, j) @ P_ss
        scale = np.sqrt((np.outer(np.diag(P_ss), np.diag(P_ss)) + C_pred ** 2) / cv_runs)
        zj = ((C - C_pred) / scale)
        z_matrix.append(zj[np.triu_indices(4)] if j == 0 else zj.ravel())
        lag_table.append({"lag": j, "predicted": C_pred, "monte_carlo": C,
                          "predicted_position_correlation": float(C_pred[0, 0] / P_ss[0, 0])})
    z_matrix = np.concatenate(z_matrix)
    schedule_drift = float(max(np.max(np.abs(step.K - K_ss)) for step in steps))
    return {"seed": seed, "scalar": {"q": str(q), "r": str(r), "M": str(exact["M"]), "P": str(exact["P"]),
                                     "K": str(exact["K"]), "a": str(exact["a"]),
                                     "riccati_residual": str(exact["riccati_residual"]), "n": n,
                                     "V_n": str(V_n), "V_n_float": V, "naive_P_over_n": naive,
                                     "variance_ratio": V / naive, "runs": runs, "lags": lags,
                                     "autocovariance": autocov, "predicted": predicted,
                                     "max_abs_z_autocovariance": float(np.max(np.abs(z_auto))),
                                     "innovation_autocorrelation": white,
                                     "max_abs_z_innovation": float(np.max(np.abs(z_white))),
                                     "monte_carlo_average_variance": mc_var, "z_average": float(z_avg),
                                     "naive_95_coverage_monte_carlo": coverage_mc,
                                     "naive_95_coverage_exact": coverage_exact},
            "cv": {"runs": cv_runs, "P_ss": P_ss, "K_ss": K_ss, "riccati_residual": residual,
                   "schedule_gain_drift": schedule_drift, "A_eigenvalues_abs": np.abs(np.linalg.eigvals(A)),
                   "lags": lag_table, "max_abs_z": float(np.max(np.abs(z_matrix))),
                   "moments_tested": int(len(z_matrix))},
            "z_critical_scalar": bonferroni(len(lags) + 1 + len(white)),
            "z_critical_cv": bonferroni(len(z_matrix))}


@task("T065", changed_files=FILES, regression_tests=(
    f"{TESTS}::test_filter_induced_correlation_and_naive_average",
    f"{TESTS}::test_section_reports_labels_and_states"))
def filter_induced_correlation(ctx):
    study = correlation_study()
    sc, cv = study["scalar"], study["cv"]
    seed = study["seed"]
    ctx.artifact_json("filter_correlation.json", as_json(study))
    ctx.artifact_text("error_autocovariance.svg", svg.line_plot(
        [("Monte Carlo", sc["lags"], sc["autocovariance"]), ("a^j P (exact)", sc["lags"], sc["predicted"]),
         ("innovation autocorrelation x P", list(range(1, 6)), [v * float(Fraction(sc["P"])) for v in
                                                                sc["innovation_autocorrelation"]])],
        title="T065 scalar filter: error autocovariance vs innovation whiteness", xlabel="lag (ticks)",
        ylabel="covariance"))
    z_s, z_cv = study["z_critical_scalar"], study["z_critical_cv"]
    findings = [
        finding("The scalar random-walk filter with q = 1, r = 30 has the exact rational steady state M = 6, "
                "P = 5, K = 1/6 and error autocovariance Cov(e_{k+j}, e_k) = (5/6)^j P", "mathematical",
                {k: sc[k] for k in ("q", "r", "M", "P", "K", "a", "riccati_residual", "n", "V_n")},
                {"derivation": "M^2 = qM + qr; e_{k+1} = (1 - K)(e_k + w_k) - K v_{k+1}", "checks": [
                    check("exact_arithmetic", "Riccati fixed-point residual in Fractions",
                          float(Fraction(sc["riccati_residual"])), 0.0)]},
                tolerance=TOL_EXACT),
        finding("Monte Carlo error autocovariances of the scalar filter match a^j P at lags 0-8, while its "
                "innovations are white", "numerical",
                {"max_abs_z_autocovariance": sc["max_abs_z_autocovariance"],
                 "max_abs_z_innovation_lags_1_5": sc["max_abs_z_innovation"], "z_critical": z_s,
                 "lag1_correlation_errors": float(sc["autocovariance"][1] / sc["autocovariance"][0]),
                 "lag1_correlation_innovations": float(sc["innovation_autocorrelation"][0])},
                {**generator_basis(seed, runs=sc["runs"]), "checks": [
                    check("analytic", "autocovariance z against a^j P", sc["max_abs_z_autocovariance"], z_s, "le"),
                    check("analytic", "innovation autocorrelation z against 0", sc["max_abs_z_innovation"], z_s, "le")]},
                tolerance=TOL_MC),
        finding("On the planar constant-velocity bench the lag-j cross-covariance of filtered errors equals "
                "[(I - K H) F]^j P_ss for j = 0-5", "numerical",
                {"max_abs_z": cv["max_abs_z"], "z_critical": z_cv, "moments_tested": cv["moments_tested"],
                 "riccati_residual": cv["riccati_residual"],
                 "position_correlation_by_lag": [row["predicted_position_correlation"] for row in cv["lags"]]},
                {**generator_basis(seed, runs=cv["runs"]), "checks": [
                    check("analytic", "matrix lag covariance z against A^j P_ss", cv["max_abs_z"], z_cv, "le"),
                    check("invariant", "steady-state Riccati fixed-point residual (relative)", cv["riccati_residual"],
                          1e-12)]},
                tolerance=TOL_MC),
        finding("Treating 20 successive filtered outputs as independent underestimates the variance of their "
                "average by the exact factor V_20 / (P/20), and a naive 95% interval covers far less often",
                "numerical",
                {"exact_variance": sc["V_n_float"], "naive_variance": sc["naive_P_over_n"],
                 "ratio": sc["variance_ratio"], "monte_carlo_variance": sc["monte_carlo_average_variance"],
                 "z_monte_carlo_vs_exact": sc["z_average"], "naive_coverage_exact": sc["naive_95_coverage_exact"],
                 "naive_coverage_monte_carlo": sc["naive_95_coverage_monte_carlo"]},
                {**generator_basis(seed), "checks": [
                    check("analytic", "Monte Carlo variance of the average against the exact V_20 (z)", sc["z_average"],
                          z_s),
                    check("analytic", "exact variance ratio V_20 / (P/20)", sc["variance_ratio"], 2.0, "ge"),
                    check("analytic", "naive coverage Monte Carlo minus exact",
                          sc["naive_95_coverage_monte_carlo"] - sc["naive_95_coverage_exact"], 0.02)]},
                tolerance=TOL_MC, counterexample={
                    "statement": "Successive filtered estimates can be averaged as independent samples",
                    "witness": {"n": sc["n"], "variance_ratio": sc["variance_ratio"],
                                "naive_95_coverage": sc["naive_95_coverage_exact"]}}),
        unreal("A real sensor's internally filtered output can be fused downstream as white noise", "sensor_performance",
               seed, "not established: the correlation shown is that of a declared synthetic filter; a real "
                     "sensor's internal filter is unknown here"),
    ]
    fields = {
        "hypothesis": "Filtered estimates are correlated in time even when the sensor noise is white; the "
                      "steady-state correlation is [(I - K H) F]^j P, and ignoring it underestimates the variance "
                      "of averaged filter outputs.",
        "mathematical_model": "e_{k+1} = (I - K H) F e_k + (I - K H) w_k - K v_{k+1} in steady state gives "
                              "Cov(e_{k+j}, e_k) = A^j P with A = (I - K H) F. Scalar random walk: q = 1, r = 30, "
                              "M^2 = qM + qr -> M = 6, P = 5, K = 1/6, a = 5/6; "
                              "Var(mean of n) = P/n^2 [n + 2 sum_j (n - j) a^j]. Innovations of the optimal "
                              "filter are white.",
        "input_data": [f"seed {seed} (PCG64)", f"scalar: {sc['runs']} runs x 30 ticks from the steady state",
                       f"planar CV: {cv['runs']} runs from P_ss, camera R, dt = 0.1 s, q = 0.05"],
        "observation_model": "Scalar z = x + v, v ~ N(0, 30); planar z = (px, py) + v, v ~ N(0, R_camera).",
        "expected_invariant": "Autocovariance a^j P (scalar) and A^j P_ss (planar); zero innovation "
                              "autocorrelation; variance of a 20-tick average equal to the exact V_20.",
        "experiment": "Simulate both filters from their steady state, estimate lag covariances across independent "
                      "runs at a fixed base tick, compare with the closed forms; average 20 successive errors "
                      "and compare the variance and the coverage of a naive interval.",
        "numerical_result": f"V_20 / (P/20) = {sc['variance_ratio']:.3f}; naive 95% interval coverage "
                            f"{sc['naive_95_coverage_exact']:.3f} (Monte Carlo {sc['naive_95_coverage_monte_carlo']:.3f}); "
                            f"scalar autocovariance max |z| {sc['max_abs_z_autocovariance']:.2f}; planar max |z| "
                            f"{cv['max_abs_z']:.2f} vs {z_cv:.2f}; lag-1 position error correlation "
                            f"{cv['lags'][1]['predicted_position_correlation']:.3f}.",
        "uncertainty": "Sampling error only (Bonferroni 99.9% family bounds); the scalar steady state and V_20 "
                       "are exact rationals.",
        "failure_modes_checked": ["non-stationary start (excluded by starting at P_ss)", "gain schedule drift from "
                                  "the steady-state gain", "innovation whiteness vs error correlation confusion",
                                  "naive averaging of filter outputs"],
        "unresolved_assumptions": ["The model is known exactly; with a mismatched model the innovations are no "
                                   "longer white either.",
                                   "Only steady state is analysed; transients have time-varying cross-covariances."],
        "recommended_next_task": "T066: verify that residuals are normalized by the filter covariance, not the raw "
                                 "sensor covariance.",
    }
    return outcome(fields, findings)


# T066 ------------------------------------------------------------------------------
def residual_study(seed: int = 66_2026, runs: int = 300, ticks: int = 100, q: float = 0.5) -> dict:
    rng = generator(seed)
    F, Q = cv_model(DT, q)
    truth = simulate_truth(rng, F, Q, MU0, P0_BENCH, runs, ticks)
    z = measure(rng, truth, H_POS, R_CAMERA, np.arange(1, ticks + 1))
    steps = gain_schedule(F, Q, P0_BENCH, [(H_POS, R_CAMERA)] * ticks)
    estimates, innovations = run_shared(F, MU0, steps, [z[:, k] for k in range(ticks)])
    post = z - estimates[:, 1:] @ H_POS.T
    series = {"innovation_S": [], "innovation_R": [], "posterior_R": [], "posterior_C": []}
    predicted = {key: [] for key in series}
    identity = 0.0
    for k, step in enumerate(steps):
        C = R_CAMERA - H_POS @ step.post @ H_POS.T
        identity = max(identity, float(np.max(np.abs(C - R_CAMERA @ np.linalg.solve(step.S, R_CAMERA)))))
        series["innovation_S"].append(quadratic(innovations[k], step.S))
        series["innovation_R"].append(quadratic(innovations[k], R_CAMERA))
        series["posterior_R"].append(quadratic(post[:, k], R_CAMERA))
        series["posterior_C"].append(quadratic(post[:, k], C))
        predicted["innovation_S"].append(2.0)
        predicted["innovation_R"].append(float(np.trace(np.linalg.solve(R_CAMERA, step.S))))
        predicted["posterior_R"].append(float(np.trace(np.linalg.solve(step.S, R_CAMERA))))
        predicted["posterior_C"].append(2.0)
    result = {}
    for key, values in series.items():
        values = np.stack(values, axis=1)
        grand_z, grand, se = run_mean_z(values[..., None] - np.array(predicted[key])[None, :, None])
        result[key] = {"consistency": consistency(values, 2), "grand_mean": float(values.mean()),
                       "predicted_grand_mean": float(np.mean(predicted[key])), "z_vs_predicted": float(grand_z[0]),
                       "predicted_per_tick": predicted[key], "anis": values.mean(axis=0)}
    return {"seed": seed, "runs": runs, "ticks": ticks, "q": q, "identity_error": identity, "series": result,
            "steady_S": steps[-1].S, "z_critical": bonferroni(len(series))}


@task("T066", changed_files=FILES, regression_tests=(
    f"{TESTS}::test_residuals_need_filter_covariance",
    f"{TESTS}::test_section_reports_labels_and_states"))
def residual_covariance(ctx):
    study = residual_study()
    seed, zc = study["seed"], study["z_critical"]
    s = study["series"]
    ctx.artifact_json("residual_normalization.json", as_json({k: v for k, v in study.items()}))
    ticks = list(range(1, study["ticks"] + 1))
    interval = s["innovation_S"]["consistency"]["interval"]
    ctx.artifact_text("anis_normalizers.svg", svg.line_plot(
        [("innovation / S", ticks, s["innovation_S"]["anis"]), ("innovation / R", ticks, s["innovation_R"]["anis"]),
         ("post-fit / R", ticks, s["posterior_R"]["anis"]), ("post-fit / (R - H P+ H^T)", ticks, s["posterior_C"]["anis"]),
         ("99% upper", [1, ticks[-1]], [interval[1]] * 2), ("99% lower", [1, ticks[-1]], [interval[0]] * 2)],
        title="T066 run-averaged normalized residuals (dof 2)", xlabel="tick", ylabel="average NIS", logy=True,
        markers=False))

    def summary(key):
        row = s[key]
        return {"fraction_inside": row["consistency"]["fraction_inside"],
                "fraction_above": row["consistency"]["fraction_above"],
                "fraction_below": row["consistency"]["fraction_below"], "grand_mean": row["grand_mean"],
                "predicted_grand_mean": row["predicted_grand_mean"], "z_vs_predicted": row["z_vs_predicted"]}

    findings = [
        finding("Innovations normalized by S = H P- H^T + R are chi-square(2) consistent at every tick", "numerical",
                summary("innovation_S"),
                {**generator_basis(seed, runs=study["runs"], ticks=study["ticks"]), "checks": [
                    check("analytic", "fraction of ticks with ANIS in the chi2(2N)/N 99% interval",
                          s["innovation_S"]["consistency"]["fraction_inside"], 0.9, "ge"),
                    check("analytic", "grand mean against 2 (run-level z)", s["innovation_S"]["z_vs_predicted"], zc)]},
                tolerance=TOL_MC),
        finding("Post-fit residuals z - H x+ have covariance R - H P+ H^T = R S^-1 R and are chi-square(2) "
                "consistent with it", "numerical",
                {**summary("posterior_C"), "identity_error": study["identity_error"]},
                {**generator_basis(seed), "checks": [
                    check("analytic", "fraction of ticks inside the 99% interval",
                          s["posterior_C"]["consistency"]["fraction_inside"], 0.9, "ge"),
                    check("invariant", "R - H P+ H^T against R S^-1 R", study["identity_error"], 1e-12)]},
                tolerance=TOL_MC),
        finding("Normalizing innovations by the raw sensor covariance R inflates NIS to tr(R^-1 S) and fails the "
                "chi-square test at nearly every tick", "numerical", summary("innovation_R"),
                {**generator_basis(seed), "checks": [
                    check("analytic", "fraction of ticks above the 99% upper bound",
                          s["innovation_R"]["consistency"]["fraction_above"], 0.9, "ge"),
                    check("analytic", "grand mean against the exact tr(R^-1 S) (run-level z)",
                          s["innovation_R"]["z_vs_predicted"], zc)]},
                tolerance=TOL_MC, counterexample={
                    "statement": "Innovations can be tested against the raw sensor covariance",
                    "witness": {"grand_mean_nis": s["innovation_R"]["grand_mean"], "nominal": 2.0}}),
        finding("Normalizing post-fit residuals by the raw R deflates them to tr(S^-1 R) < 2, which would hide an "
                "inconsistent filter", "numerical", summary("posterior_R"),
                {**generator_basis(seed), "checks": [
                    check("analytic", "fraction of ticks below the 99% lower bound",
                          s["posterior_R"]["consistency"]["fraction_below"], 0.9, "ge"),
                    check("analytic", "grand mean against the exact tr(S^-1 R) (run-level z)",
                          s["posterior_R"]["z_vs_predicted"], zc)]},
                tolerance=TOL_MC, counterexample={
                    "statement": "Post-fit residuals have the sensor covariance R",
                    "witness": {"grand_mean": s["posterior_R"]["grand_mean"], "nominal": 2.0}}),
        unreal("A real residual monitor normalized by datasheet sensor covariance is correctly calibrated",
               "sensor_performance", seed, "not established: synthetic Gaussian bench only"),
    ]
    fields = {
        "hypothesis": "Filter residuals must be normalized by the filter's own covariance: S for innovations and "
                      "R - H P+ H^T for post-fit residuals. The raw sensor covariance R over-states the first and "
                      "under-states the second by predictable amounts.",
        "mathematical_model": "nu = z - H x-, Cov nu = S = H P- H^T + R; r = z - H x+ = (I - H K) nu, Cov r = "
                              "R S^-1 R = R - H P+ H^T. E[nu^T R^-1 nu] = tr(R^-1 S) > m; E[r^T R^-1 r] = tr(S^-1 R) "
                              "< m. Planar CV, dt = 0.1 s, q = 0.5, camera R = [[0.04, 0.012], [0.012, 0.04]].",
        "input_data": [f"seed {seed} (PCG64)", f"{study['runs']} runs x {study['ticks']} ticks"],
        "observation_model": "Camera position at every tick; filter model equals the truth model.",
        "expected_invariant": "ANIS inside chi2(2N)/N intervals with S and with R - H P+ H^T; outside above "
                              "(innovation / R) and below (post-fit / R), with grand means equal to the exact traces.",
        "experiment": "One seeded Monte Carlo; four normalizations of the same residuals; per-tick ANIS "
                      "against 99% intervals; grand means against the exact traces with run-level standard errors.",
        "numerical_result": f"innovation/S inside {s['innovation_S']['consistency']['fraction_inside']:.2f}; "
                            f"post-fit/(R - HP+H^T) inside {s['posterior_C']['consistency']['fraction_inside']:.2f}; "
                            f"innovation/R grand mean {s['innovation_R']['grand_mean']:.3f} (exact "
                            f"{s['innovation_R']['predicted_grand_mean']:.3f}); post-fit/R grand mean "
                            f"{s['posterior_R']['grand_mean']:.3f} (exact {s['posterior_R']['predicted_grand_mean']:.3f}).",
        "uncertainty": "Per-tick 99% intervals and run-level z-scores (family-corrected); the trace predictions are "
                       "exact for the declared model.",
        "failure_modes_checked": ["raw R for innovations", "raw R for post-fit residuals",
                                  "post-fit covariance identity R - H P+ H^T = R S^-1 R"],
        "unresolved_assumptions": ["A deployed monitor has no truth; it sees only these residual statistics.",
                                   "q = 0.5 was chosen so the raw-R bias is visible per tick; smaller q shrinks "
                                   "tr(R^-1 H P- H^T) but never removes it."],
        "recommended_next_task": "T067: gate with a chi-square quantile of the correctly normalized NIS.",
    }
    return outcome(fields, findings)


# T067 ------------------------------------------------------------------------------
def gating_study(seed: int = 67_2026, runs: int = 400, ticks: int = 100) -> dict:
    rng = generator(seed)
    F, Q = cv_model(DT, 0.05)
    truth = simulate_truth(rng, F, Q, MU0, P0_BENCH, runs, ticks)
    z = measure(rng, truth, H_POS, R_CAMERA, np.arange(1, ticks + 1))
    readings = [z[:, k] for k in range(ticks)]
    plan = [(H_POS, R_CAMERA)] * ticks
    steps = gain_schedule(F, Q, P0_BENCH, plan)
    _, innovations = run_shared(F, MU0, steps, readings)
    nis = np.stack([quadratic(nu, step.S) for nu, step in zip(innovations, steps)], axis=1)
    nis_raw = np.stack([quadratic(nu, R_CAMERA) for nu in innovations], axis=1)
    trials = nis.size
    open_loop, closed_loop = {}, {}
    for p in (0.9, 0.99, 0.999):
        gate = chi2_quantile(p, 2)
        count = int(np.count_nonzero(nis > gate))
        row = rate_interval(count, trials)
        row.update({"p": p, "gate": gate, "nominal": 1 - p,
                    "binomial_z": (count / trials - (1 - p)) / math.sqrt(p * (1 - p) / trials),
                    "nominal_inside_wilson": bool(row["wilson"][0] <= 1 - p <= row["wilson"][1])})
        open_loop[str(p)] = row
    for p in (0.9, 0.99):
        gate = chi2_quantile(p, 2)
        gated = run_gated(F, Q, MU0, P0_BENCH, plan, readings, threshold=gate)
        rejected = np.stack([~a for a in gated["accepted"]], axis=1)
        count = int(np.count_nonzero(rejected))
        row = rate_interval(count, trials)
        # Rejections that follow a rejection in the same run: the selection effect of closed-loop gating.
        after = rejected[:, 1:][rejected[:, :-1]]
        row.update({"p": p, "gate": gate, "nominal": 1 - p,
                    "nominal_inside_wilson": bool(row["wilson"][0] <= 1 - p <= row["wilson"][1]),
                    "rejection_rate_after_a_rejection": float(after.mean()) if after.size else 0.0,
                    "rejections_followed_by_rejection": int(after.sum())})
        closed_loop[str(p)] = row
    gate99 = chi2_quantile(0.99, 2)
    raw = rate_interval(int(np.count_nonzero(nis_raw > gate99)), trials)
    quantile_error = 0.0
    for dof in range(1, 7):
        for p in (0.9, 0.99, 0.999):
            quantile_error = max(quantile_error, abs(chi2_cdf(chi2_quantile(p, dof), dof) - p))
    x4 = chi2_quantile(0.99, 4)
    closed_form_4 = abs(1 - math.exp(-x4 / 2) * (1 + x4 / 2) - 0.99)
    return {"seed": seed, "runs": runs, "ticks": ticks, "trials": trials, "open_loop": open_loop,
            "closed_loop": closed_loop, "raw_R_gate_99": raw, "quantile_roundtrip_error": quantile_error,
            "dof4_closed_form_error": closed_form_4, "gates": {str(p): chi2_quantile(p, 2) for p in (0.9, 0.99, 0.999)}}


@task("T067", changed_files=FILES, regression_tests=(
    f"{TESTS}::test_gating_rates_open_and_closed_loop",
    f"{TESTS}::test_section_reports_labels_and_states"))
def mahalanobis_gating(ctx):
    study = gating_study()
    seed = study["seed"]
    ctx.artifact_json("gating_rates.json", as_json(study))
    ol, cl = study["open_loop"], study["closed_loop"]
    ps = [0.9, 0.99, 0.999]
    ctx.artifact_text("false_rejection.svg", svg.line_plot(
        [("nominal 1 - p", [1 - p for p in ps], [1 - p for p in ps]),
         ("open loop", [1 - p for p in ps], [ol[str(p)]["rate"] for p in ps]),
         ("closed loop", [1 - p for p in ps[:2]], [cl[str(p)]["rate"] for p in ps[:2]])],
        title="T067 false-rejection rate of a chi-square gate", xlabel="nominal 1 - p", ylabel="observed rate",
        logx=True, logy=True))
    inside = all(row["nominal_inside_wilson"] for row in ol.values())
    excess = cl["0.9"]
    findings = [
        finding("Open loop, the false-rejection rate of a gate at the chi-square(2) p-quantile of the correctly "
                "normalized NIS matches 1 - p within a 99.9% Wilson interval for p = 0.9, 0.99 and 0.999",
                "numerical", {p: {k: row[k] for k in ("count", "trials", "rate", "wilson", "binomial_z")}
                              for p, row in ol.items()},
                {**generator_basis(seed, runs=study["runs"], ticks=study["ticks"]), "checks": [
                    check("analytic", "nominal 1 - p inside every Wilson interval", float(inside), 1.0, "ge")]},
                tolerance=TOL_MC),
        finding("The chi-square quantiles used by the gate invert the closed-form chi-square CDF to roundoff for "
                "1-6 degrees of freedom", "numerical",
                {"max_cdf_roundtrip_error": study["quantile_roundtrip_error"],
                 "dof4_closed_form_error": study["dof4_closed_form_error"], "gates": study["gates"]},
                {"derivation": "chi2 CDF series (even dof) and normal-CDF series (odd dof)", "checks": [
                    check("invariant", "max |F(F^-1(p)) - p|", study["quantile_roundtrip_error"], 1e-10),
                    check("analytic", "dof 4 closed form 1 - exp(-x/2)(1 + x/2)", study["dof4_closed_form_error"],
                          1e-10)]},
                tolerance=TOL_TINY),
        finding("Closed loop, a gated filter rejects valid readings more often than 1 - p at p = 0.9: a rejected "
                "reading signals a large prior error that the filter then keeps", "numerical",
                {p: {k: row[k] for k in ("count", "trials", "rate", "wilson", "rejection_rate_after_a_rejection")}
                 for p, row in cl.items()},
                {**generator_basis(seed), "checks": [
                    check("analytic", "Wilson lower bound of the closed-loop rate at p = 0.9 minus 0.1",
                          excess["wilson"][0] - 0.1, 0.0, "ge")]},
                tolerance=TOL_MC, counterexample={
                    "statement": "The false-rejection rate of a gated filter equals the nominal 1 - p",
                    "witness": {"p": 0.9, "rate": excess["rate"], "wilson": excess["wilson"],
                                "rate_after_a_rejection": excess["rejection_rate_after_a_rejection"]}}),
        finding("Gating the NIS computed with the raw sensor covariance R at the 99% quantile rejects valid readings "
                "at many times the nominal 1% rate", "numerical", study["raw_R_gate_99"],
                {**generator_basis(seed), "checks": [
                    check("analytic", "Wilson lower bound of the raw-R rejection rate",
                          study["raw_R_gate_99"]["wilson"][0], 0.02, "ge")]},
                tolerance=TOL_MC, counterexample={
                    "statement": "A chi-square gate on raw-R Mahalanobis distance has false-rejection rate 1 - p",
                    "witness": {"rate": study["raw_R_gate_99"]["rate"], "nominal": 0.01}}),
        unreal("A real gate at the 99% quantile rejects 1% of valid real readings", "sensor_performance", seed,
               "not established: real noise may be heavy-tailed, correlated or misstated"),
    ]
    fields = {
        "hypothesis": "A gate NIS <= chi2_m(p) on the correctly normalized innovation rejects valid readings at rate "
                      "1 - p when the decision does not feed back; closed-loop gating inflates the rate because a "
                      "rejection preserves the large prior error that caused it.",
        "mathematical_model": "NIS = nu^T S^-1 nu ~ chi2(2), independent across ticks for the optimal filter "
                              "(white innovations); rejections ~ Binomial(N, 1 - p). Gates: chi2_2(p) = -2 ln(1 - p).",
        "input_data": [f"seed {seed} (PCG64)", f"{study['runs']} runs x {study['ticks']} ticks = "
                                                f"{study['trials']} decisions per gate", "camera R, dt = 0.1 s, q = 0.05"],
        "observation_model": "Camera position every tick; no outliers.",
        "expected_invariant": "Open-loop rate in the Wilson interval of 1 - p; closed loop >= 1 - p.",
        "experiment": "Evaluate the gate on the ungated filter's NIS (open loop) and inside run_gated where rejected "
                      "readings are dropped (closed loop); compare counts with Wilson 99.9% intervals; repeat with "
                      "the raw-R NIS.",
        "numerical_result": "; ".join(f"open p={p}: {row['rate']:.4f}" for p, row in ol.items()) +
                            "; " + "; ".join(f"closed p={p}: {row['rate']:.4f}" for p, row in cl.items()) +
                            f"; raw-R gate at 0.99: {study['raw_R_gate_99']['rate']:.3f}.",
        "uncertainty": "Wilson 99.9% score intervals on binomial counts; the open-loop independence rests on the "
                       "whiteness of optimal innovations (T065).",
        "failure_modes_checked": ["quantile inversion error", "raw-R normalization", "closed-loop feedback of "
                                  "rejections", "low-count regime at p = 0.999"],
        "unresolved_assumptions": ["At p = 0.99 the closed-loop excess is inside the sampling interval here; it is "
                                   "not shown to be zero.",
                                   "No re-acquisition logic follows repeated rejections (see T073)."],
        "recommended_next_task": "T068: inject outliers and measure detection, false alarms and estimate error.",
    }
    return outcome(fields, findings)


# T068 ------------------------------------------------------------------------------
def _longest_run(flags) -> int:
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def outlier_study(seed: int = 68_2026, runs: int = 400, ticks: int = 100, rate: float = 0.05) -> dict:
    rng = generator(seed)
    F, Q = cv_model(DT, 0.05)
    truth = simulate_truth(rng, F, Q, MU0, P0_BENCH, runs, ticks)
    z_clean = measure(rng, truth, H_POS, R_CAMERA, np.arange(1, ticks + 1))
    contaminated = rng.random((runs, ticks)) < rate
    angle = rng.uniform(0, 2 * np.pi, (runs, ticks))
    direction = np.stack([np.cos(angle), np.sin(angle)], axis=-1)
    plan = [(H_POS, R_CAMERA)] * ticks
    p, gate = 0.99, chi2_quantile(0.99, 2)
    burn = 20

    def run(z, threshold=None, admit=None):
        return run_gated(F, Q, MU0, P0_BENCH, plan, [z[:, k] for k in range(ticks)], threshold=threshold, admit=admit)

    def mse(result):
        """Per-run mean squared position error after the burn-in."""
        err = result["estimates"][:, burn + 1:, :2] - truth[:, burn + 1:, :2]
        return np.mean(np.sum(err ** 2, axis=-1), axis=1)

    def paired(a, b):
        """z of the mean per-run MSE difference a - b (runs are independent)."""
        d = a - b
        return float(d.mean() / (d.std(ddof=1) / math.sqrt(len(d))))

    def rmse(values, mask=None):
        values = values if mask is None else values[mask]
        return float(math.sqrt(values.mean()))

    clean_open, clean_gated = run(z_clean), run(z_clean, gate)
    clean_rejected = np.stack([~a for a in clean_gated["accepted"]], axis=1)
    mse_clean_open, mse_clean_gated = mse(clean_open), mse(clean_gated)
    out = {"seed": seed, "runs": runs, "ticks": ticks, "contamination_rate": rate, "gate_p": p, "gate": gate,
           "outliers": int(contaminated.sum()), "burn_in": burn,
           "first_reading_outliers": int(contaminated[:, 0].sum()),
           "clean": {"rmse_ungated": rmse(mse_clean_open), "rmse_gated": rmse(mse_clean_gated),
                     "paired_mse_z_gated_minus_ungated": paired(mse_clean_gated, mse_clean_open),
                     "false_alarm": rate_interval(int(clean_rejected.sum()), clean_rejected.size),
                     "longest_valid_rejection_streak": max(_longest_run(r) for r in clean_rejected)},
           "cases": {}}
    for label, magnitude in (("gross", 1.5), ("subtle", 0.3)):
        z = z_clean + np.where(contaminated[..., None], magnitude * direction, 0.0)
        ungated, gated = run(z), run(z, gate)
        oracle = run(z, admit=[~contaminated[:, k] for k in range(ticks)])
        rejected = np.stack([~a for a in gated["accepted"]], axis=1)
        detected = int(np.count_nonzero(rejected & contaminated))
        false_alarm = int(np.count_nonzero(rejected & ~contaminated))
        # Predicted detection: noncentral chi-square(2) with lam = b^T S^-1 b at each outlier's prior covariance.
        runs_idx, ticks_idx = np.nonzero(contaminated)
        prior = np.einsum("ij,rjk,lk->ril", F, gated["covariances"][runs_idx, ticks_idx], F) + Q
        S = H_POS @ prior @ H_POS.T + R_CAMERA
        b = magnitude * direction[runs_idx, ticks_idx]
        lam = np.einsum("ri,ri->r", b, np.linalg.solve(S, b[..., None])[..., 0])
        probability = 1.0 - noncentral_chi2_2_cdf_many(gate, lam)
        expected = float(probability.sum())
        spread = math.sqrt(float(np.sum(probability * (1 - probability))))
        # Lock-out: a run in which at least five consecutive valid readings were rejected.
        streaks = np.array([_longest_run(r) for r in rejected & ~contaminated])
        locked = streaks >= 5
        accepted_first = contaminated[:, 0] & ~rejected[:, 0]
        m_u, m_g, m_o = mse(ungated), mse(gated), mse(oracle)
        worst = int(np.argmax(m_g - m_o))
        out["cases"][label] = {
            "magnitude_m": magnitude, "outliers": int(contaminated.sum()),
            "detection": rate_interval(detected, int(contaminated.sum())),
            "false_alarm": rate_interval(false_alarm, int((~contaminated).sum())),
            "predicted_detection_rate": expected / int(contaminated.sum()),
            "detection_z": float((detected - expected) / spread) if spread > 0 else 0.0,
            "first_reading_detection_predicted": float(np.mean(probability[ticks_idx == 0])),
            "rmse": {"ungated": rmse(m_u), "gated": rmse(m_g), "oracle": rmse(m_o)},
            "rmse_without_lockout_runs": {"ungated": rmse(m_u, ~locked), "gated": rmse(m_g, ~locked),
                                          "oracle": rmse(m_o, ~locked)},
            "paired_mse_z_gated_minus_ungated": paired(m_g, m_u),
            "lockout_runs": int(locked.sum()), "lockout_run_ids": np.nonzero(locked)[0],
            "accepted_first_reading_outliers": int(accepted_first.sum()),
            "lockout_runs_with_accepted_first_outlier": int(np.sum(locked & accepted_first)),
            "worst_run": {"run": worst, "rmse_gated": float(math.sqrt(m_g[worst])),
                          "rmse_oracle": float(math.sqrt(m_o[worst])), "longest_valid_rejection_streak":
                              int(streaks[worst])}}
    scalar = noncentral_chi2_2_cdf(gate, 4.0)
    vector = float(noncentral_chi2_2_cdf_many(gate, [4.0])[0])
    out["noncentral_cdf_agreement"] = abs(scalar - vector)
    return out


@task("T068", changed_files=FILES, regression_tests=(
    f"{TESTS}::test_outlier_rejection_detection_lockout_and_cost",
    f"{TESTS}::test_section_reports_labels_and_states"))
def outlier_rejection(ctx):
    study = outlier_study()
    seed = study["seed"]
    gross, subtle, clean = study["cases"]["gross"], study["cases"]["subtle"], study["clean"]
    ctx.artifact_json("outliers.json", as_json(study))
    ctx.artifact_text("rmse.svg", svg.line_plot(
        [("ungated", [0, 1, 2], [clean["rmse_ungated"], gross["rmse"]["ungated"], subtle["rmse"]["ungated"]]),
         ("gated", [0, 1, 2], [clean["rmse_gated"], gross["rmse"]["gated"], subtle["rmse"]["gated"]]),
         ("oracle", [1, 2], [gross["rmse"]["oracle"], subtle["rmse"]["oracle"]]),
         ("gated, lock-out runs removed", [1, 2], [gross["rmse_without_lockout_runs"]["gated"],
                                                  subtle["rmse_without_lockout_runs"]["gated"]])],
        title="T068 position RMSE (0 clean, 1 gross 1.5 m, 2 subtle 0.3 m)", xlabel="data set", ylabel="RMSE (m)"))
    z_crit = bonferroni(2)
    clean_rows = gross["rmse_without_lockout_runs"]
    findings = [
        finding("Gross 1.5 m outliers are detected at the rate predicted by the noncentral chi-square(2) law with "
                "lambda = b^T S^-1 b at each outlier's own prior covariance", "numerical",
                {"detection_rate": gross["detection"]["rate"], "predicted": gross["predicted_detection_rate"],
                 "detection_z": gross["detection_z"], "outliers": gross["outliers"],
                 "first_reading_detection_predicted": gross["first_reading_detection_predicted"]},
                {**generator_basis(seed, runs=study["runs"], contamination=study["contamination_rate"]), "checks": [
                    check("analytic", "detections against the Poisson-binomial prediction (z)", gross["detection_z"],
                          z_crit)]},
                tolerance=TOL_MC),
        finding("Outside the lock-out runs, gating returns the position RMSE to within 5% of the oracle that knows "
                "which readings are bad, while fusing every reading is at least 30% worse", "numerical",
                {"rmse_without_lockout_runs": clean_rows, "rmse_all_runs": gross["rmse"],
                 "lockout_runs_removed": gross["lockout_runs"]},
                {**generator_basis(seed), "checks": [
                    check("analytic", "gated / oracle RMSE minus one", clean_rows["gated"] / clean_rows["oracle"] - 1.0,
                          0.05),
                    check("analytic", "ungated / oracle RMSE", clean_rows["ungated"] / clean_rows["oracle"], 1.3,
                          "ge")]},
                tolerance=TOL_MC),
        finding("Cold-start lock-out: a gross outlier in the first reading passes the gate under the broad prior, "
                "and the corrupted state then rejects runs of valid readings; every lock-out run starts this way",
                "numerical",
                {"first_reading_outliers": study["first_reading_outliers"],
                 "accepted_first_reading_outliers": gross["accepted_first_reading_outliers"],
                 "lockout_runs": gross["lockout_runs"],
                 "lockout_runs_with_accepted_first_outlier": gross["lockout_runs_with_accepted_first_outlier"],
                 "worst_run": gross["worst_run"]},
                {**generator_basis(seed), "checks": [
                    check("analytic", "number of lock-out runs", gross["lockout_runs"], 1, "ge"),
                    check("exact_arithmetic", "lock-out runs not explained by an accepted first-reading outlier",
                          gross["lockout_runs"] - gross["lockout_runs_with_accepted_first_outlier"], 0)]},
                tolerance=TOL_MC, counterexample={
                    "statement": "A chi-square gate protects a filter from gross outliers",
                    "witness": gross["worst_run"]}),
        finding("On clean data the 99% gate raises false alarms at about 1% and increases the mean squared error: "
                "the rejected valid readings are the ones that would have corrected a large prior error",
                "numerical", clean,
                {**generator_basis(seed), "checks": [
                    check("analytic", "Wilson interval of the clean false-alarm rate overlaps [0.005, 0.015]",
                          max(clean["false_alarm"]["wilson"][0] - 0.015, 0.005 - clean["false_alarm"]["wilson"][1]),
                          0.0, "le"),
                    check("analytic", "paired per-run MSE difference gated - ungated (z)",
                          clean["paired_mse_z_gated_minus_ungated"], 3.0, "ge")]},
                tolerance=TOL_MC, counterexample={
                    "statement": "Gating never degrades the estimate when the data are clean",
                    "witness": {"rmse_ungated": clean["rmse_ungated"], "rmse_gated": clean["rmse_gated"],
                                "paired_z": clean["paired_mse_z_gated_minus_ungated"]}}),
        finding("Subtle 0.3 m outliers pass the gate at the predicted low detection rate; for them gating costs "
                "more accuracy than the outliers do", "numerical",
                {"detection_rate": subtle["detection"]["rate"], "predicted": subtle["predicted_detection_rate"],
                 "detection_z": subtle["detection_z"], "rmse": subtle["rmse"],
                 "paired_mse_z_gated_minus_ungated": subtle["paired_mse_z_gated_minus_ungated"]},
                {**generator_basis(seed), "checks": [
                    check("analytic", "detections against the Poisson-binomial prediction (z)", subtle["detection_z"],
                          z_crit),
                    check("analytic", "detection rate", subtle["detection"]["rate"], 0.5, "le"),
                    check("analytic", "paired per-run MSE difference gated - ungated (z)",
                          subtle["paired_mse_z_gated_minus_ungated"], 3.0, "ge")]},
                tolerance=TOL_MC, counterexample={
                    "statement": "A Mahalanobis gate removes injected outliers and so improves the estimate",
                    "witness": {"magnitude_m": subtle["magnitude_m"], "detection_rate": subtle["detection"]["rate"],
                                "rmse": subtle["rmse"]}}),
        finding("The vectorized and scalar noncentral chi-square CDFs agree", "numerical",
                study["noncentral_cdf_agreement"],
                {"derivation": "Poisson mixture of central chi-square CDFs", "checks": [
                    check("invariant", "scalar vs vectorized at lambda = 4", study["noncentral_cdf_agreement"],
                          1e-12)]},
                tolerance=TOL_TINY),
        unreal("Real outliers are rare, isolated and of fixed magnitude as in this contamination model",
               "sensor_performance", seed, "not established: the contamination model is declared, not measured"),
    ]
    fields = {
        "hypothesis": "A 99% chi-square gate on the correctly normalized NIS detects an outlier with probability "
                      "1 - F_ncx2(gate; b^T S^-1 b), removes gross outliers at ~1% false-alarm cost, cannot remove "
                      "outliers comparable to sqrt(S), and can lock out valid data after an undetected outlier.",
        "mathematical_model": "Readings z = H x + v + o with o = b u (u uniform on the unit circle) at 5% of "
                              "ticks; an outlier's NIS is noncentral chi2(2) with lambda = b^T S^-1 b when the prior "
                              "error is N(0, P-). Under the broad prior P0 (0.5 m std) a 1.5 m first-reading outlier "
                              "has small lambda and often passes.",
        "input_data": [f"seed {seed} (PCG64)", f"{study['runs']} runs x {study['ticks']} ticks",
                       f"{study['outliers']} contaminated readings (rate {study['contamination_rate']}), "
                       f"{study['first_reading_outliers']} of them first readings",
                       "magnitudes 1.5 m (gross) and 0.3 m (subtle)"],
        "observation_model": "Camera position every tick, correlated R, CV motion dt = 0.1 s, q = 0.05.",
        "expected_invariant": "Detection count matches the Poisson-binomial expectation; gated RMSE approaches the "
                              "oracle for gross outliers except after a cold-start lock-out; false alarms ~1%.",
        "experiment": "Run ungated, gated (p = 0.99) and oracle filters (the oracle skips exactly the contaminated "
                      "readings) on the same readings; count detections and false alarms; per-run MSE after a "
                      "20-tick burn-in; identify lock-out runs (>= 5 consecutive valid readings rejected); paired "
                      "per-run MSE tests for the cost of gating.",
        "numerical_result": f"gross: detection {gross['detection']['rate']:.3f} (predicted "
                            f"{gross['predicted_detection_rate']:.3f}); RMSE ungated/gated/oracle "
                            f"{gross['rmse']['ungated']:.3f}/{gross['rmse']['gated']:.3f}/{gross['rmse']['oracle']:.3f}"
                            f" m, without {gross['lockout_runs']} lock-out runs {clean_rows['ungated']:.3f}/"
                            f"{clean_rows['gated']:.3f}/{clean_rows['oracle']:.3f} m; worst lock-out run RMSE "
                            f"{gross['worst_run']['rmse_gated']:.2f} m. subtle: detection "
                            f"{subtle['detection']['rate']:.3f} (predicted {subtle['predicted_detection_rate']:.3f}). "
                            f"clean: false alarms {clean['false_alarm']['rate']:.4f}, RMSE {clean['rmse_ungated']:.4f}"
                            f" -> {clean['rmse_gated']:.4f} m with gating.",
        "uncertainty": "Detection prediction uses each outlier's own prior covariance from the gated run and assumes "
                       "the prior error is still N(0, P-); lock-out runs violate that assumption. Paired tests use "
                       "run-level independence.",
        "failure_modes_checked": ["missed small outliers", "false alarms on clean data", "estimate corruption "
                                  "without gating", "cold-start lock-out", "oracle comparison"],
        "unresolved_assumptions": ["Outliers are independent across ticks; bursts and persistent biases defeat a "
                                   "per-reading gate and need T070-style bias states.",
                                   "Lock-out recovery (covariance inflation or reacquisition) is not implemented in "
                                   "the gated filter; T073 handles reacquisition explicitly in the session API."],
        "recommended_next_task": "T069: missing data must be handled by prediction only, never by zero-filling.",
    }
    return outcome(fields, findings)
