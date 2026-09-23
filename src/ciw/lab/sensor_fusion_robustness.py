"""Missing data, stale clocks and frame mismatches on the synthetic bench (T069-T071).

Scope: T069 shows that prediction-only steps grow the covariance exactly as
F(n dt) P F(n dt)^T + Q(n dt), keep NEES consistent through gaps, and that
zero-filling is refused by the session API (and is catastrophic when done by
hand). T070 gives the camera an unmodelled one-tick clock lag: with a second,
correctly clocked sensor the innovations acquire a bias detectable by a mean
test, a filter that estimates the offset as a state removes it, and with the
stale camera alone the lag is invisible to innovations. T071 expresses a
sensor in a frame rotated by 2 degrees: with a correct-frame second sensor NIS
inflates as predicted, a single rotated sensor stays NIS-consistent while its
estimate is wrong, and the session refuses frame-id mismatches outright.

Non-claims: gaps, lags and rotations are declared synthetic faults. Nothing
here measures a real dropout process, clock offset or extrinsic error.
"""
from __future__ import annotations

import math

import numpy as np

from . import svg
from .evidence import finding
from .registry import task
from .sensor_fusion_bench import (H_POS, batched_update, consistency, cv_model, gain_schedule, generator, measure,
                                  mismatch_moments, nees_series, quadratic, run_shared, simulate_truth)
from .sensor_fusion_common import (MU0, P0_BENCH, R_CAMERA, TESTS, TOL_EXACT, TOL_MC, TOL_TINY, as_json, bonferroni,
                                   check, covariance_z, files, generator_basis, outcome, refusal, refusal_code,
                                   run_mean_z, unreal)
from .sensor_fusion_objects import (CalibrationRecord, FrameTransform, FusionSession, Observation)

DT, Q_SPECTRAL = 0.1, 0.05
R_TRACKER = 0.0025 * np.eye(2)


def rotation(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


# T069 ------------------------------------------------------------------------------
def gap_study(seed: int = 69_2026, runs: int = 400, ticks: int = 100) -> dict:
    rng = generator(seed)
    F, Q = cv_model(DT, Q_SPECTRAL)
    # Closed-form growth: the white-noise-acceleration discretization composes exactly.
    start = gain_schedule(F, Q, P0_BENCH, [(H_POS, R_CAMERA)] * 30)[-1].post
    session = FusionSession(read_only=False, dt=DT, q=Q_SPECTRAL)
    growth, worst = [], 0.0
    for n in (1, 5, 20, 50):
        session.initialize(MU0, start, 0)
        candidate = session.handle_gap("camera", n)
        Fn, Qn = cv_model(n * DT, Q_SPECTRAL)
        closed = Fn @ start @ Fn.T + Qn
        schedule = gain_schedule(F, Q, start, [None] * n)[-1].post
        api = np.asarray(candidate.covariance)
        error = max(float(np.max(np.abs(api - closed))), float(np.max(np.abs(schedule - closed)))) / float(
            np.max(np.abs(closed)))
        worst = max(worst, error)
        growth.append({"gap_ticks": n, "position_variance_x": float(closed[0, 0]),
                       "cubic_term": Q_SPECTRAL * (n * DT) ** 3 / 3, "relative_error": error})
    before = np.array(session.P)
    codes = {
        "zero_fill": refusal_code(lambda: session.handle_gap("camera", 51, strategy="zero_fill")),
        "hold_last": refusal_code(lambda: session.handle_gap("camera", 51, strategy="hold_last")),
        "nan_reading": refusal_code(lambda: Observation("camera", "world", 51, (math.nan, math.nan),
                                                        R_CAMERA, "cam-cal")),
        "absent_value": refusal_code(lambda: Observation("camera", "world", 51, None, R_CAMERA, "cam-cal")),
    }
    unchanged = bool(np.array_equal(before, session.P)) and session.tick == 50

    # Monte Carlo: a 30-tick block gap plus 20% random dropout, one pattern shared by all runs.
    present = rng.random(ticks) >= 0.2
    present[29:59] = False  # ticks 30-59
    plan = [(H_POS, R_CAMERA) if present[k] else None for k in range(ticks)]
    truth = simulate_truth(rng, F, Q, MU0, P0_BENCH, runs, ticks)
    z = measure(rng, truth, H_POS, R_CAMERA, np.arange(1, ticks + 1))
    readings = [z[:, k] if present[k] else None for k in range(ticks)]
    steps = gain_schedule(F, Q, P0_BENCH, plan)
    estimates, _ = run_shared(F, MU0, steps, readings)
    nees = nees_series(estimates, truth, steps)
    gap_end = 58  # tick 59, the last prediction-only step of the block gap
    S, z_gap = covariance_z(truth[:, gap_end + 1] - estimates[:, gap_end + 1], steps[gap_end].post)
    # Zero-filling by hand (outside the API): the missing reading becomes z = 0 with the sensor R.
    zero_steps = gain_schedule(F, Q, P0_BENCH, [(H_POS, R_CAMERA)] * ticks)
    zero_readings = [z[:, k] if present[k] else np.zeros_like(z[:, k]) for k in range(ticks)]
    zero_est, _ = run_shared(F, MU0, zero_steps, zero_readings)
    zero_nees = nees_series(zero_est, truth, zero_steps)

    def rmse(est):
        return float(np.sqrt(np.mean(np.sum((est[:, 1:, :2] - truth[:, 1:, :2]) ** 2, axis=-1))))

    gap_ratio = float(steps[gap_end].post[0, 0] / steps[28].post[0, 0])
    return {"seed": seed, "runs": runs, "ticks": ticks, "missing_ticks": int((~present).sum()),
            "growth": growth, "max_growth_error": worst, "refusals": codes, "state_unchanged_by_refusals": unchanged,
            "nees": consistency(nees, 4), "nees_mean_per_tick": nees.mean(axis=0),
            "gap_end_max_abs_z": float(np.max(np.abs(z_gap))), "gap_end_covariance": S,
            "gap_end_predicted": steps[gap_end].post, "gap_variance_growth": gap_ratio,
            "zero_fill": {"nees": consistency(zero_nees, 4), "grand_mean_nees": float(zero_nees.mean()),
                          "max_anees": float(zero_nees.mean(axis=0).max()), "rmse": rmse(zero_est)},
            "predict_only_rmse": rmse(estimates), "present": present, "z_critical": bonferroni(10)}


@task("T069", changed_files=files("sensor_fusion_robustness", "sensor_fusion_objects"), regression_tests=(
    f"{TESTS}::test_missing_data_prediction_only_and_zero_fill_refusal",
    f"{TESTS}::test_section_reports_labels_and_states"))
def missing_data(ctx):
    study = gap_study()
    seed = study["seed"]
    ctx.artifact_json("missing_data.json", as_json(study))
    ticks = list(range(1, study["ticks"] + 1))
    ctx.artifact_text("anees_gap.svg", svg.line_plot(
        [("prediction only", ticks, study["nees_mean_per_tick"]),
         ("99% upper", [1, ticks[-1]], [study["nees"]["interval"][1]] * 2),
         ("99% lower", [1, ticks[-1]], [study["nees"]["interval"][0]] * 2)],
        title="T069 ANEES through a 30-tick gap and 20% dropout", xlabel="tick", ylabel="ANEES", markers=False))
    codes = study["refusals"]
    findings = [
        finding("Prediction-only steps grow the covariance exactly as F(n dt) P F(n dt)^T + Q(n dt), identically "
                "in the session API and the batch schedule", "numerical",
                {"growth": study["growth"], "max_relative_error": study["max_growth_error"]},
                {"derivation": "white-noise-acceleration discretization composes: F(a)F(b) = F(a+b), "
                               "Q(a+b) = F(b) Q(a) F(b)^T + Q(b)", "checks": [
                    check("analytic", "session and schedule covariance against the closed form (relative)",
                          study["max_growth_error"], 1e-12)]},
                tolerance=TOL_TINY),
        finding("With a 30-tick gap and 20% dropout handled by prediction only, NEES stays chi-square consistent "
                "and the error covariance at the end of the gap equals the grown covariance", "numerical",
                {"nees": study["nees"], "gap_end_max_abs_z": study["gap_end_max_abs_z"],
                 "z_critical": study["z_critical"], "gap_variance_growth": study["gap_variance_growth"],
                 "missing_ticks": study["missing_ticks"]},
                {**generator_basis(seed, runs=study["runs"], ticks=study["ticks"]), "checks": [
                    check("analytic", "fraction of ticks with ANEES in the 99% interval",
                          study["nees"]["fraction_inside"], 0.9, "ge"),
                    check("analytic", "gap-end error covariance against the predicted covariance (max |z|)",
                          study["gap_end_max_abs_z"], study["z_critical"], "le")]},
                tolerance=TOL_MC),
        finding("The session refuses zero-filling and other substitutes for a missing reading, refuses NaN and "
                "absent observation values, and leaves its state unchanged by the refusals", "computational_pipeline",
                {**codes, "state_unchanged": study["state_unchanged_by_refusals"]},
                {"derivation": "FusionSession.handle_gap and Observation validation", "checks": [
                    refusal("handle_gap strategy zero_fill", "zero_fill_refused", codes["zero_fill"]),
                    refusal("handle_gap strategy hold_last", "gap_strategy_refused", codes["hold_last"]),
                    refusal("Observation with NaN values", "nonfinite_observation", codes["nan_reading"]),
                    refusal("Observation with no value", "missing_reading", codes["absent_value"]),
                    check("invariant", "state changed by a refused call", float(not study["state_unchanged_by_refusals"]),
                          0.0)]},
                tolerance=TOL_EXACT),
        finding("Zero-filling missing readings by hand drags the estimate toward the origin and destroys "
                "consistency", "numerical",
                {"zero_fill": study["zero_fill"], "predict_only_rmse": study["predict_only_rmse"]},
                {**generator_basis(seed), "checks": [
                    check("analytic", "grand-mean NEES of the zero-filled filter", study["zero_fill"]["grand_mean_nees"],
                          100.0, "ge"),
                    check("analytic", "zero-filled RMSE relative to prediction only",
                          study["zero_fill"]["rmse"] / study["predict_only_rmse"], 5.0, "ge")]},
                tolerance=TOL_MC, counterexample={
                    "statement": "A zero placeholder for a missing reading is harmless",
                    "witness": {"grand_mean_nees": study["zero_fill"]["grand_mean_nees"], "nominal": 4.0,
                                "rmse_m": study["zero_fill"]["rmse"]}}),
        unreal("Real sensor dropouts are independent of the state and of the noise, as assumed here",
               "sensor_performance", seed, "not established: the dropout pattern is declared and state-independent"),
    ]
    fields = {
        "hypothesis": "A missing reading is handled by prediction only: the covariance grows exactly by the model "
                      "and the filter stays consistent; any substituted value (zero, hold-last) is refused because "
                      "it fabricates information.",
        "mathematical_model": "P(k + n | k) = F(n dt) P F(n dt)^T + Q(n dt) with F(t) = [[I, t I], [0, I]], "
                              "Q(t) = q [[t^3/3 I, t^2/2 I], [t^2/2 I, t I]]; the position variance gains "
                              "q (n dt)^3 / 3 plus the propagated terms. dt = 0.1 s, q = 0.05.",
        "input_data": [f"seed {seed} (PCG64)", f"{study['runs']} runs x {study['ticks']} ticks",
                       f"{study['missing_ticks']} missing ticks (block 30-59 plus 20% random), shared by all runs"],
        "observation_model": "Camera position when present; absent otherwise (no placeholder value exists).",
        "expected_invariant": "Session covariance = closed form = schedule to roundoff; ANEES inside chi2(4N)/N; "
                              "refusal codes zero_fill_refused, gap_strategy_refused, nonfinite_observation, "
                              "missing_reading.",
        "experiment": "Advance a FusionSession through gaps of 1, 5, 20 and 50 ticks and compare with the closed "
                      "form; Monte Carlo NEES through the dropout pattern; request zero-fill, hold-last and NaN/None "
                      "readings through the API; zero-fill by hand outside the API as the counterexample.",
        "numerical_result": f"max relative growth error {study['max_growth_error']:.1e}; ANEES inside "
                            f"{study['nees']['fraction_inside']:.2f} of ticks; gap-end |z| "
                            f"{study['gap_end_max_abs_z']:.2f}; position variance grows "
                            f"{study['gap_variance_growth']:.0f}x over the gap; zero-fill grand NEES "
                            f"{study['zero_fill']['grand_mean_nees']:.0f}, RMSE {study['zero_fill']['rmse']:.2f} m vs "
                            f"{study['predict_only_rmse']:.3f} m.",
        "uncertainty": "Closed-form comparison is exact to roundoff; Monte Carlo statements carry 99% per-tick "
                       "intervals and a Bonferroni bound on the gap-end covariance.",
        "failure_modes_checked": ["zero-fill", "hold-last substitution", "NaN placeholder", "None placeholder",
                                  "state mutation by a refused call", "covariance growth mismatch"],
        "unresolved_assumptions": ["Dropouts are missing at random; state-dependent dropouts (occlusion near "
                                   "obstacles) bias the filter even with prediction-only steps.",
                                   "Long gaps stay consistent only while the motion model holds (see T073)."],
        "recommended_next_task": "T070: stale clocks, where a reading is present but refers to the wrong time.",
    }
    return outcome(fields, findings)


# T070 ------------------------------------------------------------------------------
def _split_whitened(nu, S):
    L = np.linalg.cholesky(S)
    return np.linalg.solve(L, nu[..., None])[..., 0]


def stale_clock_study(seed: int = 70_2026, runs: int = 200, ticks: int = 200, tracker_every: int = 5) -> dict:
    rng = generator(seed)
    F, Q = cv_model(DT, Q_SPECTRAL)
    tau = DT  # the camera reports the state one tick late
    burn = 50
    truth = simulate_truth(rng, F, Q, MU0, P0_BENCH, runs, ticks)
    camera = truth[:, :-1, :2] + rng.standard_normal((runs, ticks, 2)) @ np.linalg.cholesky(R_CAMERA).T
    tracker_ticks = np.arange(tracker_every, ticks + 1, tracker_every)
    tracker = measure(rng, truth, H_POS, R_TRACKER, tracker_ticks)
    has_tracker = np.zeros(ticks, dtype=bool)
    has_tracker[tracker_ticks - 1] = True
    zero = np.zeros((2, 2))
    stacked_H = np.vstack([H_POS, H_POS])
    stacked_R = np.block([[R_CAMERA, zero], [zero, R_TRACKER]])

    def naive(with_tracker):
        plan = [(stacked_H, stacked_R) if with_tracker and has_tracker[k] else (H_POS, R_CAMERA) for k in range(ticks)]
        steps = gain_schedule(F, Q, P0_BENCH, plan)
        tracker_iter = iter(range(len(tracker_ticks)))
        readings, means = [], []
        mean_truth = [MU0]
        for k in range(ticks):
            mean_truth.append(F @ mean_truth[-1])
        for k in range(ticks):
            if with_tracker and has_tracker[k]:
                j = next(tracker_iter)
                readings.append(np.concatenate([camera[:, k], tracker[:, j]], axis=1))
                means.append(np.concatenate([mean_truth[k][:2], mean_truth[k + 1][:2]])[None, :])
            else:
                readings.append(camera[:, k])
                means.append(mean_truth[k][:2][None, :])
        estimates, innovations = run_shared(F, MU0, steps, readings)
        # Exact expected innovations: the same linear filter run on the noise-free mean readings.
        _, expected = run_shared(F, MU0, steps, means)
        cam = np.stack([_split_whitened(nu[:, :2], step.S[:2, :2]) for nu, step in zip(innovations, steps)], axis=1)
        cam_expected = np.stack([_split_whitened(mu[:, :2], step.S[:2, :2]) for mu, step in zip(expected, steps)],
                                axis=1)[0]
        result = {"camera_mean_z": run_mean_z(cam[:, burn:])[0],
                  "camera_whitened_mean": cam[:, burn:].mean(axis=(0, 1)),
                  "camera_predicted_whitened_mean": cam_expected[burn:].mean(axis=0),
                  "camera_z_vs_predicted": run_mean_z(cam[:, burn:] - cam_expected[None, burn:])[0]}
        if with_tracker:
            idx = [k for k in range(burn, ticks) if has_tracker[k]]
            trk = np.stack([_split_whitened(innovations[k][:, 2:], steps[k].S[2:, 2:]) for k in idx], axis=1)
            trk_expected = np.stack([_split_whitened(expected[k][:, 2:], steps[k].S[2:, 2:]) for k in idx], axis=1)[0]
            result.update({"tracker_mean_z": run_mean_z(trk)[0], "tracker_whitened_mean": trk.mean(axis=(0, 1)),
                           "tracker_predicted_whitened_mean": trk_expected.mean(axis=0),
                           "tracker_z_vs_predicted": run_mean_z(trk - trk_expected[None])[0]})
        error = estimates[:, burn + 1:, :2] - truth[:, burn + 1:, :2]
        result["position_error_mean"] = error.mean(axis=(0, 1))
        result["position_error_mean_z_vs_zero"] = run_mean_z(error)[0]
        result["nees"] = consistency(nees_series(estimates, truth, steps)[:, burn:], 4)
        return result

    two_sensor, camera_only = naive(True), naive(False)

    # Offset-augmented EKF: state (px, py, vx, vy, tau); camera h = p - tau v, tracker h = p.
    F5 = np.eye(5)
    F5[:4, :4] = F
    Q5 = np.zeros((5, 5))
    Q5[:4, :4] = Q
    R_cam_eff = R_CAMERA + Q_SPECTRAL * DT ** 3 / 3 * np.eye(2)  # backward process noise of the lagged state
    x = np.broadcast_to(np.concatenate([MU0, [0.0]]), (runs, 5)).copy()
    P = np.broadcast_to(np.diag([0.25, 0.25, 0.04, 0.04, 0.2 ** 2]), (runs, 5, 5)).copy()
    cam_white, trk_white, nees5, tau_hat, tau_std = [], [], [], [], []
    H_trk = np.hstack([H_POS, np.zeros((2, 1))])
    j = 0
    for k in range(ticks):
        x = x @ F5.T
        P = F5 @ P @ F5.T + Q5
        tau_k, v = x[:, 4], x[:, 2:4]
        H = np.zeros((runs, 2, 5))
        H[:, :, :2] = np.eye(2)
        H[:, 0, 2] = H[:, 1, 3] = -tau_k
        H[:, :, 4] = -v
        nu = camera[:, k] - (x[:, :2] - tau_k[:, None] * v)
        x, P, S, _ = batched_update(x, P, nu, H, R_cam_eff)
        cam_white.append(_split_whitened(nu, S))
        if has_tracker[k]:
            nu_t = tracker[:, j] - x[:, :2]
            j += 1
            x, P, S_t, _ = batched_update(x, P, nu_t, H_trk, R_TRACKER)
            if k >= burn:
                trk_white.append(_split_whitened(nu_t, S_t))
        e = np.concatenate([truth[:, k + 1], np.full((runs, 1), tau)], axis=1) - x
        nees5.append(np.einsum("ri,ri->r", e, np.linalg.solve(P, e[..., None])[..., 0]))
        tau_hat.append(x[:, 4].copy())
        tau_std.append(np.sqrt(P[:, 4, 4]))
    cam_white = np.stack(cam_white, axis=1)
    trk_white = np.stack(trk_white, axis=1)
    nees5 = np.stack(nees5, axis=1)
    final_tau = tau_hat[-1]
    augmented = {"camera_mean_z": run_mean_z(cam_white[:, burn:])[0], "tracker_mean_z": run_mean_z(trk_white)[0],
                 "tau_mean": float(final_tau.mean()), "tau_run_std": float(final_tau.std(ddof=1)),
                 "tau_mean_posterior_std": float(tau_std[-1].mean()),
                 "tau_z": float((final_tau.mean() - tau) / (final_tau.std(ddof=1) / math.sqrt(runs))),
                 "nees_grand_mean": float(nees5[:, burn:].mean()),
                 "nees_grand_z": float(run_mean_z(nees5[:, burn:, None] - 5.0)[0][0]),
                 "tau_hat_mean_by_tick": np.mean(tau_hat, axis=1)}
    return {"seed": seed, "runs": runs, "ticks": ticks, "tau_s": tau, "burn_in": burn,
            "tracker_every": tracker_every, "two_sensor": two_sensor, "camera_only": camera_only,
            "augmented": augmented, "z_critical": bonferroni(16)}


@task("T070", changed_files=files("sensor_fusion_robustness"), regression_tests=(
    f"{TESTS}::test_stale_clock_bias_detection_and_augmented_offset",
    f"{TESTS}::test_section_reports_labels_and_states"))
def stale_clock(ctx):
    study = stale_clock_study()
    seed, zc = study["seed"], study["z_critical"]
    two, one, aug = study["two_sensor"], study["camera_only"], study["augmented"]
    ctx.artifact_json("stale_clock.json", as_json(study))
    ctx.artifact_text("tau_estimate.svg", svg.line_plot(
        [("mean tau estimate", list(range(1, study["ticks"] + 1)), aug["tau_hat_mean_by_tick"]),
         ("true lag", [1, study["ticks"]], [study["tau_s"]] * 2)],
        title="T070 offset-augmented filter: estimated camera lag", xlabel="tick", ylabel="tau (s)", markers=False))
    detected = float(max(np.max(np.abs(two["camera_mean_z"])), np.max(np.abs(two["tracker_mean_z"]))))
    agree = float(max(np.max(np.abs(two["camera_z_vs_predicted"])), np.max(np.abs(two["tracker_z_vs_predicted"]))))
    aug_z = float(max(np.max(np.abs(aug["camera_mean_z"])), np.max(np.abs(aug["tracker_mean_z"]))))
    one_z = float(np.max(np.abs(one["camera_mean_z"])))
    error_z = float(np.max(np.abs(one["position_error_mean_z_vs_zero"])))
    findings = [
        finding("With a correctly clocked tracker fused alongside, an unmodelled one-tick camera lag biases the "
                "whitened innovations of both sensors; a mean test detects it and the bias equals the exact "
                "linear prediction", "numerical",
                as_json({"max_abs_mean_z": detected, "max_abs_z_vs_predicted": agree, "z_critical": zc,
                         "camera_whitened_mean": two["camera_whitened_mean"],
                         "tracker_whitened_mean": two["tracker_whitened_mean"],
                         "tracker_predicted_whitened_mean": two["tracker_predicted_whitened_mean"]}),
                {**generator_basis(seed, runs=study["runs"], ticks=study["ticks"]), "checks": [
                    check("analytic", "largest mean-innovation z against zero", detected, zc, "ge"),
                    check("analytic", "mean innovations against the noise-free linear prediction (max |z|)", agree, zc,
                          "le")]},
                tolerance=TOL_MC),
        finding("A filter that estimates the clock offset as a state (h = p - tau v) removes the innovation bias "
                "and recovers tau = 0.1 s", "numerical",
                as_json({k: aug[k] for k in ("tau_mean", "tau_run_std", "tau_mean_posterior_std", "tau_z",
                                             "nees_grand_mean", "nees_grand_z")} | {"max_abs_mean_z": aug_z}),
                {**generator_basis(seed), "checks": [
                    check("analytic", "largest mean-innovation z of the augmented filter", aug_z, zc, "le"),
                    check("analytic", "mean final tau estimate against 0.1 s (z)", aug["tau_z"], zc),
                    check("analytic", "grand-mean 5-state NEES minus 5, relative", aug["nees_grand_mean"] / 5.0 - 1.0,
                          0.15)]},
                tolerance=TOL_MC),
        finding("With the stale camera as the only position sensor the lag is invisible to the innovations: the "
                "lagged constant-velocity path is itself a constant-velocity path, so the estimate is biased by "
                "-tau v while the mean test passes", "numerical",
                as_json({"camera_mean_z": one["camera_mean_z"], "position_error_mean": one["position_error_mean"],
                         "position_error_mean_z_vs_zero": one["position_error_mean_z_vs_zero"],
                         "expected_position_bias": [-study["tau_s"] * MU0[2], -study["tau_s"] * MU0[3]],
                         "nees": one["nees"]}),
                {**generator_basis(seed), "checks": [
                    check("analytic", "largest mean-innovation z (camera only)", one_z, zc, "le"),
                    check("analytic", "position error mean z against zero", error_z, zc, "ge"),
                    check("analytic", "position error mean minus -tau E[v] (x component, m)",
                          one["position_error_mean"][0] + study["tau_s"] * MU0[2], 0.02)]},
                tolerance=TOL_MC, counterexample={
                    "statement": "A stale sensor clock always shows up as a bias in the filter innovations",
                    "witness": as_json({"sensors": "camera only", "camera_mean_z": one["camera_mean_z"],
                                        "position_error_mean_m": one["position_error_mean"]})}),
        unreal("The recovered offset calibrates the clock of a real camera", "calibration", seed,
               "not established: the lag is a declared synthetic fault"),
    ]
    fields = {
        "hypothesis": "An unmodelled clock offset biases innovations only when another sensor pins the true time; "
                      "a mean test then detects it, and an offset state in the filter removes it.",
        "mathematical_model": "Camera reports z_k = p(t_k - tau) + v = p_k - tau v_k + eta + v with tau = dt = "
                              "0.1 s; tracker (2 Hz, R = 0.0025 I) reports p_k + v. Naive filter: z = p + v. "
                              "Augmented EKF: state (p, v, tau), h = p - tau v, H = [I, -tau I, -v], R + q dt^3/3 I. "
                              "Expected naive innovations follow from the same linear filter run on noise-free "
                              "mean readings.",
        "input_data": [f"seed {seed} (PCG64)", f"{study['runs']} runs x {study['ticks']} ticks",
                       "camera every tick (lagged one tick), tracker every 5 ticks"],
        "observation_model": "Camera and tracker positions, independent noise; tau prior N(0, 0.2^2) s.",
        "expected_invariant": "Naive two-sensor mean innovations nonzero and equal to the prediction; augmented "
                              "mean innovations zero and tau-hat -> 0.1 s; camera-only innovations unbiased while the "
                              "position error mean is -tau E[v] = (-0.1, -0.05) m.",
        "experiment": "Whiten innovations with their filter covariance, average per run after a 50-tick burn-in, "
                      "and test the grand mean with run-level standard errors for the naive two-sensor, naive "
                      "camera-only and offset-augmented filters.",
        "numerical_result": f"naive two-sensor max |z| {detected:.1f} (vs prediction {agree:.2f}); augmented max "
                            f"|z| {aug_z:.2f}, tau-hat {aug['tau_mean']:.4f} s (run spread {aug['tau_run_std']:.4f}, "
                            f"posterior std {aug['tau_mean_posterior_std']:.4f}), NEES {aug['nees_grand_mean']:.2f}; "
                            f"camera-only innovation |z| {one_z:.2f}, position error mean "
                            f"({one['position_error_mean'][0]:.3f}, {one['position_error_mean'][1]:.3f}) m.",
        "uncertainty": "Run-level z-tests with a Bonferroni family bound; the augmented filter is an EKF whose "
                       "linearization and the ignored correlation of the lagged-state noise with the process noise "
                       "are approximations (NEES tolerance 15%).",
        "failure_modes_checked": ["unmodelled lag with a reference sensor", "unmodelled lag with no reference",
                                  "offset state convergence", "EKF consistency"],
        "unresolved_assumptions": ["The lag is constant and exactly one tick; drifting or jittering clocks need a "
                                   "random-walk offset state.",
                                   "The offset is observable only while the velocity is nonzero."],
        "recommended_next_task": "T071: frame mismatch, the spatial analogue of a stale clock.",
    }
    return outcome(fields, findings)


# T071 ------------------------------------------------------------------------------
def frame_mismatch_study(seed: int = 71_2026, runs: int = 200, ticks: int = 100, degrees: float = 2.0) -> dict:
    rng = generator(seed)
    F, Q = cv_model(DT, Q_SPECTRAL)
    rot = rotation(math.radians(degrees))
    truth = simulate_truth(rng, F, Q, MU0, P0_BENCH, runs, ticks)
    ticks_all = np.arange(1, ticks + 1)
    camera = measure(rng, truth, H_POS, R_CAMERA, ticks_all)
    # The tracker frame is rotated by +theta: it reports R^T p, with noise declared in its own frame.
    tracker_frame = truth[:, 1:, :2] @ rot + rng.standard_normal((runs, ticks, 2)) @ np.linalg.cholesky(R_TRACKER).T
    zero = np.zeros((2, 2))
    stacked_H = np.vstack([H_POS, H_POS])
    stacked_R = np.block([[R_CAMERA, zero], [zero, R_TRACKER]])
    truth_H = np.vstack([H_POS, np.hstack([rot.T, zero])])
    early, late = slice(0, 20), slice(50, ticks)

    def nis_run(readings, H, R, truth_plan):
        steps = gain_schedule(F, Q, P0_BENCH, [(H, R)] * ticks)
        estimates, innovations = run_shared(F, MU0, steps, readings)
        nis = np.stack([quadratic(nu, step.S) for nu, step in zip(innovations, steps)], axis=1)
        nees = nees_series(estimates, truth, steps)
        predicted = mismatch_moments(F, Q, MU0, P0_BENCH, MU0, steps, truth_plan)
        dof = H.shape[0]
        return {"nis": nis, "nees": nees, "predicted_nis": np.array(predicted["nis"]),
                "predicted_nees": np.array(predicted["nees"]), "dof": dof,
                "early": consistency(nis[:, early], dof), "late": consistency(nis[:, late], dof),
                "late_grand_nis": float(nis[:, late].mean()),
                "late_predicted_nis": float(np.mean(predicted["nis"][late])),
                "late_z_vs_predicted": float(run_mean_z(nis[:, late, None] - np.array(predicted["nis"][late])[None, :, None])[0][0]),
                "final_nees": float(nees[:, -1].mean()), "final_predicted_nees": float(predicted["nees"][-1])}

    mixed = nis_run([np.concatenate([camera[:, k], tracker_frame[:, k]], axis=1) for k in range(ticks)],
                    stacked_H, stacked_R, [(truth_H, np.zeros(4), stacked_R)] * ticks)
    fixed_readings = [np.concatenate([camera[:, k], tracker_frame[:, k] @ rot.T], axis=1) for k in range(ticks)]
    fixed_R = np.block([[R_CAMERA, zero], [zero, rot @ R_TRACKER @ rot.T]])
    fixed = nis_run(fixed_readings, stacked_H, fixed_R, [(stacked_H, np.zeros(4), fixed_R)] * ticks)
    alone = nis_run([tracker_frame[:, k] for k in range(ticks)], H_POS, R_TRACKER,
                    [(np.hstack([rot.T, zero]), np.zeros(2), R_TRACKER)] * ticks)

    # API: frame ids are checked before any arithmetic.
    session = FusionSession(read_only=False, frame_id="world", dt=DT, q=Q_SPECTRAL)
    session.register_calibration(CalibrationRecord("trk-cal", "tracker", "tracker-frame", 0, 10_000))
    session.initialize(MU0, P0_BENCH, 0)
    observation = Observation("tracker", "tracker-frame", 1, tuple(tracker_frame[0, 0]), R_TRACKER, "trk-cal")
    transform = FrameTransform("tracker-frame", "world", tuple(map(tuple, rot)), (0.0, 0.0))
    codes = {"fuse_in_wrong_frame": refusal_code(lambda: session.fuse(observation)),
             "transform_from_wrong_frame": refusal_code(
                 lambda: FrameTransform("camera-frame", "world", tuple(map(tuple, rot)), (0.0, 0.0)).apply(observation)),
             "fuse_after_transform": refusal_code(lambda: session.fuse(transform.apply(observation)))}
    dispositions = [entry["disposition"] for entry in session.log]
    return {"seed": seed, "runs": runs, "ticks": ticks, "rotation_deg": degrees,
            "cases": {"mixed_frames": mixed, "transformed": fixed, "rotated_sensor_alone": alone},
            "api_codes": codes, "log_dispositions": dispositions, "z_critical": bonferroni(3)}


def _frame_summary(case):
    return {k: case[k] for k in ("dof", "late_grand_nis", "late_predicted_nis", "late_z_vs_predicted",
                                 "final_nees", "final_predicted_nees")} | {
        "early_fraction_inside": case["early"]["fraction_inside"], "late_fraction_inside": case["late"]["fraction_inside"],
        "late_fraction_above": case["late"]["fraction_above"]}


@task("T071", changed_files=files("sensor_fusion_robustness", "sensor_fusion_objects"), regression_tests=(
    f"{TESTS}::test_frame_mismatch_inflation_blind_spot_and_refusal",
    f"{TESTS}::test_section_reports_labels_and_states"))
def frame_mismatch(ctx):
    study = frame_mismatch_study()
    seed, zc = study["seed"], study["z_critical"]
    mixed, fixed, alone = (study["cases"][k] for k in ("mixed_frames", "transformed", "rotated_sensor_alone"))
    ctx.artifact_json("frame_mismatch.json", as_json({k: v for k, v in study.items() if k != "cases"} | {
        "cases": {name: {k: v for k, v in case.items() if k not in ("nis", "nees")} | {
            "anis": case["nis"].mean(axis=0), "anees": case["nees"].mean(axis=0)}
            for name, case in study["cases"].items()}}))
    ticks = list(range(1, study["ticks"] + 1))
    ctx.artifact_text("anis_frames.svg", svg.line_plot(
        [("camera + rotated tracker", ticks, mixed["nis"].mean(axis=0)),
         ("predicted", ticks, mixed["predicted_nis"]),
         ("after FrameTransform", ticks, fixed["nis"].mean(axis=0)),
         ("99% upper (dof 4)", [1, ticks[-1]], [mixed["late"]["interval"][1]] * 2)],
        title="T071 run-averaged NIS with a 2 degree frame mismatch", xlabel="tick", ylabel="ANIS", markers=False))
    codes = study["api_codes"]
    findings = [
        finding("Fusing a tracker expressed in a frame rotated by 2 degrees with a world-frame camera inflates NIS "
                "as the target moves away from the rotation centre, exactly as predicted by the mismatched-filter "
                "moments", "numerical", _frame_summary(mixed),
                {**generator_basis(seed, runs=study["runs"], ticks=study["ticks"]), "checks": [
                    check("analytic", "fraction of ticks 51-100 above the 99% upper bound",
                          mixed["late"]["fraction_above"], 0.9, "ge"),
                    check("analytic", "late grand NIS against the exact prediction (run-level z)",
                          mixed["late_z_vs_predicted"], zc)]},
                tolerance=TOL_MC),
        finding("Near the rotation centre (ticks 1-20) the same mismatch is invisible to the NIS test", "numerical",
                {"early_fraction_inside": mixed["early"]["fraction_inside"],
                 "early_predicted_nis": float(np.mean(mixed["predicted_nis"][:20]))},
                {**generator_basis(seed), "checks": [
                    check("analytic", "fraction of ticks 1-20 inside the 99% interval", mixed["early"]["fraction_inside"],
                          0.9, "ge")]},
                tolerance=TOL_MC, counterexample={
                    "statement": "A passing NIS test shows that the sensor frames agree",
                    "witness": {"ticks": "1-20", "fraction_inside": mixed["early"]["fraction_inside"],
                                "rotation_deg": study["rotation_deg"]}}),
        finding("A rotated sensor fused alone keeps NIS consistent (its readings form a rotated constant-velocity "
                "path) while NEES grows: the estimate is confidently in the wrong frame", "numerical",
                _frame_summary(alone),
                {**generator_basis(seed), "checks": [
                    check("analytic", "fraction of ticks 51-100 inside the 99% NIS interval",
                          alone["late"]["fraction_inside"], 0.9, "ge"),
                    check("analytic", "final ANEES", alone["final_nees"], 20.0, "ge"),
                    check("analytic", "final ANEES against the exact prediction, relative",
                          alone["final_nees"] / alone["final_predicted_nees"] - 1.0, 0.25)]},
                tolerance=TOL_MC, counterexample={
                    "statement": "A frame mismatch always produces NIS inflation",
                    "witness": {"sensors": "rotated tracker alone", "late_grand_nis": alone["late_grand_nis"],
                                "final_nees": alone["final_nees"]}}),
        finding("Applying the declared FrameTransform before fusion restores chi-square consistency", "numerical",
                _frame_summary(fixed),
                {**generator_basis(seed), "checks": [
                    check("analytic", "fraction of ticks 51-100 inside the 99% interval", fixed["late"]["fraction_inside"],
                          0.9, "ge"),
                    check("analytic", "late grand NIS against 4 (run-level z)", fixed["late_z_vs_predicted"], zc)]},
                tolerance=TOL_MC),
        finding("The session refuses an observation whose frame id differs from its own, and a transform refuses an "
                "observation from another source frame; the refused observation is retained in the log",
                "computational_pipeline", {**codes, "log_dispositions": study["log_dispositions"]},
                {"derivation": "FusionSession._admissible_source and FrameTransform.apply", "checks": [
                    refusal("fuse a tracker-frame observation into a world session", "frame_mismatch",
                            codes["fuse_in_wrong_frame"]),
                    refusal("apply a camera-frame transform to a tracker-frame observation", "frame_mismatch",
                            codes["transform_from_wrong_frame"]),
                    check("invariant", "fusion after the explicit transform refused",
                          float(codes["fuse_after_transform"] != "no_refusal"), 0.0),
                    check("invariant", "refused observation not retained",
                          float(study["log_dispositions"][0] != "refused:frame_mismatch"), 0.0)]},
                tolerance=TOL_EXACT),
        unreal("A 2 degree rotation is the size of a real extrinsic calibration error", "calibration", seed,
               "not established: the rotation is a declared synthetic fault"),
    ]
    fields = {
        "hypothesis": "A measurement expressed in a rotated frame inflates NIS when another sensor fixes the true "
                      "frame; alone it is invisible to NIS; frame-id mismatches must therefore be refused at the "
                      "API rather than left to statistical detection.",
        "mathematical_model": "Tracker readings R(theta)^T p + v with theta = 2 deg about the world origin; filter "
                              "assumes p + v. The bias (R^T - I) p grows with |p|. Expected NIS/NEES per tick from "
                              "the exact joint moments of [x; x_hat] (mismatch_moments).",
        "input_data": [f"seed {seed} (PCG64)", f"{study['runs']} runs x {study['ticks']} ticks",
                       "camera (world frame) and tracker (rotated frame) every tick"],
        "observation_model": "Stacked 4-vector (camera, tracker) or tracker alone; FrameTransform rotates value and "
                             "covariance by R(theta).",
        "expected_invariant": "Late ANIS above the interval and equal to the prediction for mixed frames; inside the "
                              "interval after the transform; inside for the rotated sensor alone while NEES grows; "
                              "API refusal code frame_mismatch.",
        "experiment": "Three filters on the same readings (mixed frames, transformed, rotated sensor alone); per-tick "
                      "ANIS against 99% chi-square intervals; late grand means against exact predictions; then the "
                      "typed API with frame ids.",
        "numerical_result": f"mixed: late ANIS {mixed['late_grand_nis']:.2f} (predicted {mixed['late_predicted_nis']:.2f}), "
                            f"early inside {mixed['early']['fraction_inside']:.2f}; transformed late inside "
                            f"{fixed['late']['fraction_inside']:.2f}; rotated alone late ANIS "
                            f"{alone['late_grand_nis']:.2f}, final ANEES {alone['final_nees']:.1f} (predicted "
                            f"{alone['final_predicted_nees']:.1f}); API: {codes['fuse_in_wrong_frame']}.",
        "uncertainty": "Per-tick 99% intervals and run-level z-tests; predictions are exact for the declared "
                       "linear-Gaussian model.",
        "failure_modes_checked": ["rotated frame with a reference sensor", "rotated frame alone",
                                  "mismatch near the rotation centre", "wrong source frame in a transform",
                                  "frame-id mismatch at the API"],
        "unresolved_assumptions": ["The transform is exact; an uncertain extrinsic needs its own covariance term.",
                                   "Rotation about the world origin only; a translated frame adds a constant bias."],
        "recommended_next_task": "T072: calibration expiry, the temporal validity of the declared transform and noise.",
    }
    return outcome(fields, findings)
