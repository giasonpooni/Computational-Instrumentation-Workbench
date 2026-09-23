"""Calibration expiry, track loss, analytic reference, typed admission and defaults (T072-T076).

Scope: T072 retains readings taken after a calibration's validity interval (or
under a calibration declared for another frame) but refuses to fuse them.
T073 lets a sensor stop, predicts the exact tick at which the track is
declared lost, refuses fusion and admission of a lost track without side
effects and requires explicit two-point reacquisition. T074 checks the Kalman
estimate against the exact batch (information-form) posterior, in floating
point and in rational arithmetic. T075 keeps observations, candidate states
and admitted states as separate types and runs mutation tests on every
admission check. T076 verifies that the session API defaults to read-only
with the CIW authority vocabulary.

Non-claims: admission here is a software gate over synthetic data; calibration
records, thresholds and authority values are declared. Nothing here validates
a calibration, sets a safe track-loss threshold or grants actuator or
production authority.
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError, asdict, replace
from fractions import Fraction
import inspect
import math

import numpy as np

from . import svg
from .evidence import finding
from .registry import task
from .sensor_fusion_bench import (H_POS, batch_posterior, batch_posterior_banded, chi2_quantile, consistency,
                                  cv_model, exact_scalar_filter, gain_schedule, generator, measure,
                                  mismatch_moments, nees_series, run_shared, simulate_truth)
from .sensor_fusion_common import (DECLARED_WORKLOAD, MU0, P0_BENCH, R_CAMERA, TESTS, TOL_EXACT, TOL_MC, as_json,
                                   bonferroni, check, covariance_z, exact, files, generator_basis, identity, is_not,
                                   mc95, outcome, refusal, refusal_code, roundoff, run_mean_z, unreal)
from .sensor_fusion_objects import (ADMISSION_CHECKS, DEFAULT_AUTHORITY, SYNTHETIC_AUTHORITY, AdmittedState,
                                    CalibrationRecord, CandidateState, FusionSession, Observation,
                                    _digest, admission_verdict)

DT, Q_SPECTRAL = 0.1, 0.05
R_TRACKER = 0.0025 * np.eye(2)
ADMIT = {"expected_frame_id": "world", "nis_probability": 0.99, "max_position_std": 10.0}


def _tests(task_id, *specific) -> tuple:
    return tuple(f"{TESTS}::{name}" for name in specific) + (
        f"{TESTS}::test_section_reports_labels_and_states[{task_id}]",)


def _session(**kwargs) -> FusionSession:
    session = FusionSession(read_only=False, dt=DT, q=Q_SPECTRAL, **kwargs)
    session.register_calibration(CalibrationRecord("cam-cal", "camera", "world", 0, 1_000_000))
    return session


def _camera_run(seed, ticks):
    rng = generator(seed)
    F, Q = cv_model(DT, Q_SPECTRAL)
    truth = simulate_truth(rng, F, Q, MU0, P0_BENCH, 1, ticks)
    z = measure(rng, truth, H_POS, R_CAMERA, np.arange(1, ticks + 1))
    return truth[0], z[0]


# T072 ------------------------------------------------------------------------------
def calibration_study(seed: int = 72_2026, ticks: int = 100, expiry: int = 60, runs: int = 200,
                      drift_per_tick: float = 0.01) -> dict:
    truth, z = _camera_run(seed, ticks + 10)
    session = FusionSession(read_only=False, dt=DT, q=Q_SPECTRAL)
    session.register_calibration(CalibrationRecord("cam-cal-1", "camera", "world", 0, expiry))
    shadow = FusionSession(read_only=False, dt=DT, q=Q_SPECTRAL)
    shadow.register_calibration(CalibrationRecord("cam-cal-1", "camera", "world", 0, expiry))
    session.initialize(MU0, P0_BENCH, 0)
    shadow.initialize(MU0, P0_BENCH, 0)
    submitted, codes = [], {}
    for k in range(1, ticks + 1):
        obs = Observation("camera", "world", k, tuple(z[k - 1]), R_CAMERA, "cam-cal-1")
        submitted.append(obs.digest)
        codes[k] = refusal_code(lambda obs=obs: session.fuse(obs))
        if k < expiry:
            shadow.fuse(obs)
    session.predict(ticks)
    shadow.predict(ticks)
    identical = bool(np.array_equal(session.x, shadow.x) and np.array_equal(session.P, shadow.P))
    fused = [k for k, code in codes.items() if code == "none"]
    refused = [k for k, code in codes.items() if code == "calibration_expired"]
    retained = [entry["digest"] for entry in session.log]
    dispositions = [entry["disposition"] for entry in session.log]
    # Renewal: a new calibration record lets fusion resume; the expired record stays expired.
    session.register_calibration(CalibrationRecord("cam-cal-2", "camera", "world", ticks, ticks + 100))
    renewed = [refusal_code(lambda k=k: session.fuse(Observation("camera", "world", k, tuple(z[k - 1]), R_CAMERA,
                                                                 "cam-cal-2"))) for k in range(ticks + 1, ticks + 6)]
    stale_again = refusal_code(lambda: session.fuse(Observation("camera", "world", ticks + 6, tuple(z[ticks + 5]),
                                                                R_CAMERA, "cam-cal-1")))
    session.register_calibration(CalibrationRecord("cam-cal-3", "camera", "world", 0, 1_000_000))
    session.revoke_calibration("cam-cal-3")
    session.register_calibration(CalibrationRecord("trk-cal", "tracker", "world", 0, 1_000_000))
    session.register_calibration(CalibrationRecord("cam-cal-body", "camera", "body", 0, 1_000_000))
    other = {
        "revoked": refusal_code(lambda: session.fuse(Observation("camera", "world", ticks + 7, tuple(z[ticks + 6]),
                                                                 R_CAMERA, "cam-cal-3"))),
        "unknown": refusal_code(lambda: session.fuse(Observation("camera", "world", ticks + 7, tuple(z[ticks + 6]),
                                                                 R_CAMERA, "cam-cal-9"))),
        "other_sensor": refusal_code(lambda: session.fuse(Observation("camera", "world", ticks + 7,
                                                                      tuple(z[ticks + 6]), R_CAMERA, "trk-cal"))),
        "other_frame": refusal_code(lambda: session.fuse(Observation("camera", "world", ticks + 7,
                                                                     tuple(z[ticks + 6]), R_CAMERA, "cam-cal-body"))),
        "stale_record_after_renewal": stale_again,
    }

    # Monte Carlo counterfactual under a declared post-expiry drift of the synthetic camera.
    rng = generator(seed + 1)
    F, Q = cv_model(DT, Q_SPECTRAL)
    mc_truth = simulate_truth(rng, F, Q, MU0, P0_BENCH, runs, ticks)
    mc_z = measure(rng, mc_truth, H_POS, R_CAMERA, np.arange(1, ticks + 1))
    drift = np.zeros((ticks, 2))
    drift[expiry - 1:, 0] = drift_per_tick * np.arange(1, ticks - expiry + 2)
    mc_z = mc_z + drift[None]
    refusing = gain_schedule(F, Q, P0_BENCH, [(H_POS, R_CAMERA) if k < expiry else None for k in range(1, ticks + 1)])
    fusing = gain_schedule(F, Q, P0_BENCH, [(H_POS, R_CAMERA)] * ticks)
    readings = [mc_z[:, k] for k in range(ticks)]
    est_refusing, _ = run_shared(F, MU0, refusing, [r if k + 1 < expiry else None for k, r in enumerate(readings)])
    est_fusing, _ = run_shared(F, MU0, fusing, readings)
    nees_refusing = nees_series(est_refusing, mc_truth, refusing)
    nees_fusing = nees_series(est_fusing, mc_truth, fusing)
    after = slice(expiry - 1, ticks)
    # Exact prediction of the fusing filter under the declared drift: readings H x + c_k + v.
    drift_plan = [(H_POS, drift[k], R_CAMERA) for k in range(ticks)]
    predicted = {name: np.array(mismatch_moments(F, Q, MU0, P0_BENCH, MU0, steps, drift_plan)["nees"])[after]
                 for name, steps in (("refusing", refusing), ("fusing", fusing))}
    prediction = {}
    for name, values in (("refusing", nees_refusing[:, after]), ("fusing", nees_fusing[:, after])):
        z_pred, _, se = run_mean_z(values[..., None] - predicted[name][None, :, None])
        prediction[name] = {"predicted_grand_nees": float(predicted[name].mean()), "z_vs_predicted": float(z_pred[0]),
                            "standard_error": float(se[0])}
    return {"seed": seed, "ticks": ticks, "expiry_tick": expiry, "fused_ticks": [fused[0], fused[-1]],
            "fused": len(fused), "refused_expired": len(refused), "first_refused": refused[0] if refused else None,
            "boundary": {"tick_before_expiry": codes[expiry - 1], "tick_at_expiry": codes[expiry]},
            "retained": len(retained), "retained_digests_match": retained == submitted,
            "dispositions": {d: dispositions[:ticks].count(d) for d in sorted(set(dispositions[:ticks]))},
            "state_equals_prediction_only_shadow": identical, "renewed": renewed, "other_refusals": other,
            "monte_carlo": {"runs": runs, "drift_m_per_tick": drift_per_tick,
                            "refusing": consistency(nees_refusing[:, after], 4),
                            "fusing": consistency(nees_fusing[:, after], 4),
                            "final_position_std_refusing": float(math.sqrt(refusing[-1].post[0, 0])),
                            "final_position_std_fusing": float(math.sqrt(fusing[-1].post[0, 0])),
                            "prediction": prediction, "z_critical": bonferroni(2)},
            "anees_refusing": nees_refusing.mean(axis=0), "anees_fusing": nees_fusing.mean(axis=0)}


@task("T072", changed_files=files("sensor_fusion_admission"), regression_tests=_tests(
    "T072", "test_calibration_expiry_retains_but_refuses"))
def calibration_expiry(ctx):
    study = calibration_study()
    seed, mc = study["seed"], study["monte_carlo"]
    ctx.artifact_json("calibration_expiry.json", as_json(study))
    ticks = list(range(1, study["ticks"] + 1))
    ctx.artifact_text("anees_expiry.svg", svg.line_plot(
        [("refuse after expiry", ticks, study["anees_refusing"]), ("fuse expired (drifting)", ticks,
                                                                   study["anees_fusing"]),
         ("99% upper", [1, ticks[-1]], [mc["refusing"]["interval"][1]] * 2)],
        title="T072 ANEES with a calibration expiring at tick 60", xlabel="tick", ylabel="ANEES", logy=True,
        markers=False))
    other = study["other_refusals"]
    boundary = study["boundary"]
    expected_refused = study["ticks"] - study["expiry_tick"] + 1
    findings = [
        finding("Readings inside the half-open validity interval [0, 60) are fused and every later reading is "
                "refused with calibration_expired, exactly at the boundary", "computational_pipeline",
                {k: study[k] for k in ("fused", "refused_expired", "first_refused", "boundary", "renewed")},
                {"derivation": "CalibrationRecord.covers and FusionSession._admissible_source", "checks": [
                    check("exact_arithmetic", "fused count minus 59", study["fused"] - (study["expiry_tick"] - 1), 0),
                    check("exact_arithmetic", "refused count minus 41", study["refused_expired"] - expected_refused, 0),
                    check("invariant", "reading at tick 59 (inside the interval) refused",
                          is_not(boundary["tick_before_expiry"], "none"), 0.0),
                    refusal("reading at tick 60 (first tick outside)", "calibration_expired",
                            boundary["tick_at_expiry"]),
                    check("exact_arithmetic", "readings under the renewed record refused",
                          sum(code != "none" for code in study["renewed"]), 0)]},
                uncertainty=exact("integer counts and refusal codes"), tolerance=TOL_EXACT),
        finding("Refused readings are retained with their digests and dispositions, and they do not change the "
                "state: the session equals a shadow session that never saw them, bit for bit",
                "computational_pipeline",
                {k: study[k] for k in ("retained", "retained_digests_match", "dispositions",
                                       "state_equals_prediction_only_shadow")},
                {"derivation": "FusionSession.record before any refusal", "checks": [
                    check("exact_arithmetic", "retained entries minus submitted readings",
                          study["retained"] - study["ticks"], 0),
                    check("invariant", "retained digests differ from submitted digests",
                          float(not study["retained_digests_match"]), 0.0),
                    check("invariant", "state differs from the prediction-only shadow",
                          float(not study["state_equals_prediction_only_shadow"]), 0.0)]},
                uncertainty=exact("integer counts and bitwise state comparison"), tolerance=TOL_EXACT),
        finding("Revoked, unknown, other-sensor and other-frame calibrations are refused, and the expired record "
                "stays expired after a renewal is registered", "computational_pipeline", other,
                {"derivation": "FusionSession._admissible_source", "checks": [
                    refusal("revoked calibration", "calibration_revoked", other["revoked"]),
                    refusal("unregistered calibration id", "calibration_unknown", other["unknown"]),
                    refusal("calibration registered for another sensor", "calibration_unknown", other["other_sensor"]),
                    refusal("calibration declared for the body frame, reading from the world frame",
                            "calibration_frame_mismatch", other["other_frame"]),
                    refusal("expired record after renewal", "calibration_expired",
                            other["stale_record_after_renewal"])]},
                uncertainty=exact("refusal codes"), tolerance=TOL_EXACT),
        finding("Under a declared post-expiry drift of 1 cm per tick, refusing expired readings keeps NEES "
                "consistent (with a growing covariance) while fusing them makes the filter overconfident by the "
                "amount the exact joint moments predict", "numerical", {k: mc[k] for k in mc},
                {**generator_basis(seed + 1, runs=mc["runs"]), "checks": [
                    check("analytic", "refusing filter: fraction of post-expiry ticks inside the 99% interval",
                          mc["refusing"]["fraction_inside"], 0.9, "ge"),
                    check("analytic", "fusing filter: fraction of post-expiry ticks above the 99% interval",
                          mc["fusing"]["fraction_above"], 0.5, "ge"),
                    check("analytic", "fusing filter: post-expiry grand NEES against the mismatch_moments "
                                      "prediction (run-level z)", mc["prediction"]["fusing"]["z_vs_predicted"],
                          mc["z_critical"]),
                    check("analytic", "refusing filter: post-expiry grand NEES against the prediction (run-level z)",
                          mc["prediction"]["refusing"]["z_vs_predicted"], mc["z_critical"])]},
                uncertainty=mc95(mc["prediction"]["fusing"]["standard_error"],
                                 "run-level standard error of the fusing filter's post-expiry grand NEES"),
                tolerance=TOL_MC, counterexample={
                    "statement": "Readings taken after calibration expiry can be fused like the others",
                    "witness": {"drift_m_per_tick": mc["drift_m_per_tick"],
                                "post_expiry_grand_nees": mc["fusing"]["grand_mean"], "nominal": 4.0}}),
        unreal("The validity interval [0, 60) reflects how long a real camera calibration stays valid", "calibration",
               seed, "not established: the record and the post-expiry drift are declared"),
    ]
    fields = {
        "hypothesis": "A calibration record with a validity interval must gate fusion: readings outside it are "
                      "retained for audit but never fused, and the refusal leaves the state exactly as if they had "
                      "not arrived.",
        "mathematical_model": "CalibrationRecord covers ticks in [valid_from, valid_until); fuse() records first, "
                              "then refuses with calibration_expired before any arithmetic. After expiry the "
                              "state is predicted only, P(k + n) = F(n dt) P F(n dt)^T + Q(n dt).",
        "input_data": [f"seed {seed} (PCG64) for one synthetic camera run of 110 ticks",
                       f"seed {seed + 1}: {mc['runs']} Monte Carlo runs with a declared drift of "
                       f"{mc['drift_m_per_tick']} m/tick after tick 59"],
        "observation_model": "Camera positions in the world frame citing cam-cal-1 (valid to tick 60), then "
                             "cam-cal-2 (valid from tick 100).",
        "expected_invariant": "59 fused, 41 refused, all 100 retained with matching digests; state bitwise equal "
                              "to the shadow session; renewed readings fused.",
        "experiment": "Submit every reading to a FusionSession, compare with a shadow session fed only in-interval "
                      "readings, register a renewal and exercise revoked/unknown/other-sensor/other-frame records; "
                      "Monte Carlo NEES of refusing and fusing filters under the declared drift, compared with the "
                      "exact joint-moment prediction.",
        "numerical_result": f"fused {study['fused']}, refused {study['refused_expired']} (first at tick "
                            f"{study['first_refused']}), retained {study['retained']}; shadow equality "
                            f"{study['state_equals_prediction_only_shadow']}; post-expiry ANEES inside: refusing "
                            f"{mc['refusing']['fraction_inside']:.2f}, fusing grand NEES "
                            f"{mc['fusing']['grand_mean']:.1f} (exact prediction "
                            f"{mc['prediction']['fusing']['predicted_grand_nees']:.1f}); final position std refusing "
                            f"{mc['final_position_std_refusing']:.3f} m vs fusing {mc['final_position_std_fusing']:.3f} m.",
        "uncertainty": "The API statements are exact; the Monte Carlo statement depends on the declared drift, "
                       "which is an assumption, not a measurement.",
        "failure_modes_checked": ["boundary tick off-by-one", "refused reading changing the state", "loss of the "
                                  "refused reading", "revoked record", "unknown record", "record for another sensor",
                                  "record declared for another frame", "expired record reused after renewal"],
        "unresolved_assumptions": ["Expiry is a hard tick boundary; real calibrations degrade gradually and their "
                                   "validity depends on temperature, shocks and use.",
                                   "Readings are not re-labelled retroactively under a renewed calibration."],
        "recommended_next_task": "T073: a sensor that stops altogether, and the explicit track-lost state.",
    }
    return outcome(fields, findings)


# T073 ------------------------------------------------------------------------------
def two_point_covariance(R1, R2, dt, q) -> np.ndarray:
    return np.block([[R2, R2 / dt], [R2 / dt, (R1 + R2) / dt ** 2 + q * dt / 3 * np.eye(2)]])


def two_point_error_map(R1, R2, dt, q) -> np.ndarray:
    """The same covariance derived separately: the estimator error is a linear map of (n1, n2, w_p, w_v).

    p_hat - p2 = n2 and v_hat - v2 = (n2 - n1)/dt + w_p/dt - w_v, where (w_p, w_v) ~ N(0, Q(dt)) is the
    process noise between the two readings (Q from cv_model, not the q dt/3 shortcut).
    """
    eye, zero = np.eye(2), np.zeros((2, 2))
    J = np.block([[zero, eye, zero, zero], [-eye / dt, eye / dt, eye / dt, -eye]])
    Q = cv_model(dt, q)[1]
    sigma = np.block([[R1, zero, np.zeros((2, 4))], [zero, R2, np.zeros((2, 4))], [np.zeros((4, 4)), Q]])
    return J @ sigma @ J.T


def turn_transition(omega: float, t: float) -> np.ndarray:
    """Linear transition of a constant-rate coordinated turn: p += A(t) v, v -> rot(omega t) v."""
    s, c = math.sin(omega * t), math.cos(omega * t)
    transition = np.eye(4)
    transition[:2, 2:] = np.array([[s, c - 1.0], [1.0 - c, s]]) / omega
    transition[2:, 2:] = np.array([[c, -s], [s, c]])
    return transition


def turn_state(x0, omega, t):
    """Constant-speed coordinated turn: position and velocity after time t from x0 = (p, v)."""
    return turn_transition(omega, t) @ np.asarray(x0, dtype=float)


def turn_process_noise(omega: float, q: float, t: float, nodes: int = 24) -> np.ndarray:
    """int_0^t T(u) G q G^T T(u)^T du for white acceleration through the turn (Gauss-Legendre)."""
    points, weights = np.polynomial.legendre.leggauss(nodes)
    G = np.vstack([np.zeros((2, 2)), np.eye(2)])
    total = np.zeros((4, 4))
    for point, weight in zip(points, weights):
        TG = turn_transition(omega, 0.5 * t * (point + 1.0)) @ G
        total += 0.5 * t * weight * q * TG @ TG.T
    return total


def track_lost_study(seed: int = 73_2026, active: int = 30, radius: float = 1.0, omega: float = 0.2,
                     mc_runs: int = 20_000, turn_runs: int = 4000) -> dict:
    truth, z = _camera_run(seed, 200)
    session = _session(track_radius=radius, track_probability=0.99)
    session.initialize(MU0, P0_BENCH, 0)
    for k in range(1, active + 1):
        session.fuse(Observation("camera", "world", k, tuple(z[k - 1]), R_CAMERA, "cam-cal"))
    P_active = np.array(session.P)
    x_active = np.array(session.x)
    gate = chi2_quantile(0.99, 2)
    predicted_lost = None
    radii = []
    for n in range(1, 200):
        Fn, Qn = cv_model(n * DT, Q_SPECTRAL)
        Pn = Fn @ P_active @ Fn.T + Qn
        r = math.sqrt(float(np.linalg.eigvalsh(Pn[:2, :2]).max()) * gate)
        radii.append(r)
        if r > radius and predicted_lost is None:
            predicted_lost = active + n
            break
    observed_lost = None
    tick = active
    while observed_lost is None and tick < active + 200:
        tick += 1
        candidate = session.predict(tick)
        if candidate.track_status == "lost":
            observed_lost = tick
    lost_candidate = candidate
    obs = {k: Observation("camera", "world", k, tuple(z[k - 1]), R_CAMERA, "cam-cal") for k in
           range(observed_lost - 1, observed_lost + 6)}
    first = observed_lost + 1
    before = (np.array(session.x), np.array(session.P), session.tick, session.track_status)
    codes = {
        "admit_lost": refusal_code(lambda: session.admit(lost_candidate, **ADMIT)),
        "fuse_lost": refusal_code(lambda: session.fuse(obs[first])),
    }
    # A refused fuse must leave the estimate, clock and status exactly as they were.
    unchanged = (bool(np.array_equal(before[0], session.x) and np.array_equal(before[1], session.P))
                 and (session.tick, session.track_status) == before[2:])
    codes["reacquire_gap"] = refusal_code(lambda: session.reacquire(obs[first + 1], obs[first + 3]))
    codes["reacquire_older_than_clock"] = refusal_code(
        lambda: session.reacquire(obs[observed_lost - 1], obs[observed_lost]))
    clock_after_refusals = session.tick
    reacquired = session.reacquire(obs[first + 1], obs[first + 2])
    formula = two_point_covariance(R_CAMERA, R_CAMERA, DT, Q_SPECTRAL)
    error_map = two_point_error_map(R_CAMERA, R_CAMERA, DT, Q_SPECTRAL)
    reacquire_error = float(np.max(np.abs(np.asarray(reacquired.covariance) - error_map)) / np.max(np.abs(error_map)))
    codes["reacquire_while_tracking"] = refusal_code(lambda: session.reacquire(obs[first + 2], obs[first + 3]))
    resumed = session.fuse(obs[first + 3])
    codes["admit_after_reacquisition"] = refusal_code(lambda: session.admit(resumed, **ADMIT))
    dispositions = [entry["disposition"] for entry in session.log[active:]]

    # Monte Carlo: two-point initialization error covariance for the constant-velocity truth.
    rng = generator(seed + 1)
    F, Q = cv_model(DT, Q_SPECTRAL)
    x1 = MU0 + rng.standard_normal((mc_runs, 4)) @ np.linalg.cholesky(P0_BENCH).T
    x2 = x1 @ F.T + rng.standard_normal((mc_runs, 4)) @ np.linalg.cholesky(Q).T
    L = np.linalg.cholesky(R_CAMERA)
    z1 = x1[:, :2] + rng.standard_normal((mc_runs, 2)) @ L.T
    z2 = x2[:, :2] + rng.standard_normal((mc_runs, 2)) @ L.T
    estimate = np.concatenate([z2, (z2 - z1) / DT], axis=1)
    errors = estimate - x2
    S, z_two = covariance_z(errors, formula)
    nees = np.einsum("ri,ri->r", errors, np.linalg.solve(formula, errors.T).T)
    nees_z = float((nees.mean() - 4.0) / math.sqrt(8.0 / mc_runs))

    # Unmodelled coordinated turn during the gap. With e0 ~ N(0, P_active) and white acceleration through the
    # turn, E[NEES](t) = tr(P_f^-1 (T P_active T^T + Q_turn(t))) + d^T P_f^-1 d, d = (T(t) - F(t)) x_hat.
    threshold = chi2_quantile(0.99, 4)
    consistent_until, table, radius_then = None, [], None
    for n in range(1, 400):
        t = n * DT
        Fn, Qn = cv_model(t, Q_SPECTRAL)
        Pn = Fn @ P_active @ Fn.T + Qn
        Tn = turn_transition(omega, t)
        d = (Tn - Fn) @ x_active
        random_part = float(np.trace(np.linalg.solve(Pn, Tn @ P_active @ Tn.T + turn_process_noise(omega, Q_SPECTRAL, t))))
        expected = random_part + float(d @ np.linalg.solve(Pn, d))
        if n % 5 == 0:
            table.append({"gap_ticks": n, "expected_nees": expected, "random_part": random_part})
        if expected > threshold:
            consistent_until, expected_at_break, random_at_break = n, expected, random_part
            radius_then = math.sqrt(float(np.linalg.eigvalsh(Pn[:2, :2]).max()) * gate)
            break
    # Monte Carlo of the turning truth tick by tick (per-tick Q_turn(dt) composes to Q_turn(n dt)).
    turn_mc = None
    if consistent_until:
        turn_rng = generator(seed + 2)
        T1, Q1 = turn_transition(omega, DT), turn_process_noise(omega, Q_SPECTRAL, DT)
        x = x_active + turn_rng.standard_normal((turn_runs, 4)) @ np.linalg.cholesky(P_active).T
        for _ in range(consistent_until):
            x = x @ T1.T + turn_rng.standard_normal((turn_runs, 4)) @ np.linalg.cholesky(Q1).T
        Fn, Qn = cv_model(consistent_until * DT, Q_SPECTRAL)
        Pn = Fn @ P_active @ Fn.T + Qn
        e = x - Fn @ x_active
        values = np.einsum("ri,ri->r", e, np.linalg.solve(Pn, e.T).T)
        se = float(values.std(ddof=1) / math.sqrt(turn_runs))
        turn_mc = {"runs": turn_runs, "mean_nees": float(values.mean()), "standard_error": se,
                   "z_vs_expected": float((values.mean() - expected_at_break) / se),
                   "expected_nees": expected_at_break, "random_part": random_at_break}
    return {"seed": seed, "active_ticks": active, "radius_m": radius, "track_probability": 0.99,
            "predicted_lost_tick": predicted_lost, "observed_lost_tick": observed_lost,
            "radius_at_loss": radii[-1], "codes": codes, "refused_fuse_left_state_unchanged": unchanged,
            "clock_after_refusals": clock_after_refusals, "reacquire_covariance_error": reacquire_error,
            "formula_vs_error_map": float(np.max(np.abs(formula - error_map)) / np.max(np.abs(error_map))),
            "dispositions": {d: dispositions.count(d) for d in sorted(set(dispositions))},
            "two_point": {"runs": mc_runs, "max_abs_z": float(np.max(np.abs(z_two))), "z_critical": bonferroni(11),
                          "nees_mean": float(nees.mean()), "nees_z": nees_z, "sample": S, "formula": formula},
            "turn": {"omega_rad_s": omega, "speed_m_s": float(np.linalg.norm(x_active[2:])),
                     "threshold_chi2_4_99": threshold, "first_inconsistent_gap_ticks": consistent_until,
                     "radius_at_inconsistency_m": radius_then,
                     "gap_ticks_before_track_loss": observed_lost - active, "table": table,
                     "monte_carlo": turn_mc, "z_critical": bonferroni(1)}}


@task("T073", changed_files=files("sensor_fusion_admission"), regression_tests=_tests(
    "T073", "test_track_lost_prediction_refusals_and_reacquisition"))
def track_lost(ctx):
    study = track_lost_study()
    seed, codes, two, turn = study["seed"], study["codes"], study["two_point"], study["turn"]
    mc = turn["monte_carlo"] or {"z_vs_expected": 1e6, "standard_error": 0.0, "mean_nees": 0.0}
    ctx.artifact_json("track_lost.json", as_json(study))
    ctx.artifact_text("turn_nees.svg", svg.line_plot(
        [("expected NEES under a 0.2 rad/s turn", [r["gap_ticks"] for r in turn["table"]],
          [r["expected_nees"] for r in turn["table"]]),
         ("random part only", [r["gap_ticks"] for r in turn["table"]], [r["random_part"] for r in turn["table"]]),
         ("chi2_4 99%", [5, max(10, turn["table"][-1]["gap_ticks"] if turn["table"] else 10)],
          [turn["threshold_chi2_4_99"]] * 2)],
        title="T073 coasting through an unmodelled turn", xlabel="gap (ticks)", ylabel="expected NEES"))
    findings = [
        finding("The track is declared lost at exactly the tick at which the 99% position ellipse, grown in closed "
                "form from the last update, first exceeds the 1 m radius", "computational_pipeline",
                {k: study[k] for k in ("active_ticks", "radius_m", "predicted_lost_tick", "observed_lost_tick",
                                       "radius_at_loss")},
                {"derivation": "radius(n) = sqrt(lambda_max(P_pos(n)) chi2_2(0.99)) with the closed-form P(n)",
                 "checks": [check("exact_arithmetic", "observed minus predicted track-lost tick",
                                  study["observed_lost_tick"] - study["predicted_lost_tick"], 0)]},
                uncertainty=exact("integer tick from a deterministic closed form"),
                tolerance={"abs": 1e-9, "rel": 1e-9}),
        finding("A lost track is not admissible and cannot be updated, and the refused update leaves the state, "
                "covariance and clock unchanged; reacquisition is explicit, needs two consecutive readings no older "
                "than the session clock, and is refused while tracking", "computational_pipeline",
                {**codes, "refused_fuse_left_state_unchanged": study["refused_fuse_left_state_unchanged"],
                 "clock_after_refusals": study["clock_after_refusals"], "dispositions": study["dispositions"]},
                {"derivation": "FusionSession.fuse, reacquire and the admission gate", "checks": [
                    refusal("admit a lost-track candidate", "track_lost", codes["admit_lost"]),
                    refusal("fuse into a lost track", "track_lost_requires_reacquisition", codes["fuse_lost"]),
                    check("invariant", "state, covariance, clock or status changed by the refused fuse",
                          float(not study["refused_fuse_left_state_unchanged"]), 0.0),
                    refusal("reacquire from non-consecutive readings", "reacquisition_needs_consecutive_readings",
                            codes["reacquire_gap"]),
                    refusal("reacquire from readings older than the session clock", "out_of_order",
                            codes["reacquire_older_than_clock"]),
                    check("exact_arithmetic", "session clock after the refusals minus the lost tick",
                          study["clock_after_refusals"] - study["observed_lost_tick"], 0),
                    refusal("reacquire while tracking", "reacquisition_not_needed", codes["reacquire_while_tracking"]),
                    check("invariant", "admission of the first update after reacquisition refused",
                          is_not(codes["admit_after_reacquisition"], "none"), 0.0)]},
                uncertainty=exact("refusal codes and bitwise state comparison"), tolerance=TOL_EXACT),
        finding("The two-point reacquisition covariance [[R, R/dt], [R/dt, 2R/dt^2 + q dt/3 I]] is exact for the "
                "constant-velocity truth", "numerical",
                {k: two[k] for k in ("runs", "max_abs_z", "z_critical", "nees_mean", "nees_z")} |
                {"session_vs_error_map": study["reacquire_covariance_error"],
                 "formula_vs_error_map": study["formula_vs_error_map"]},
                {**generator_basis(seed + 1, runs=two["runs"]), "checks": [
                    check("analytic", "sample covariance of the two-point error against the formula (max |z|)",
                          two["max_abs_z"], two["z_critical"], "le"),
                    check("analytic", "mean NEES against 4 (z)", two["nees_z"], two["z_critical"]),
                    check("analytic", "session reacquisition covariance against the linear error map J Sigma J^T "
                                      "with Q(dt) from cv_model (relative)", study["reacquire_covariance_error"],
                          1e-12)]},
                uncertainty=mc95(math.sqrt(8.0 / two["runs"]), "sampling standard error of the mean NEES "
                                                              "(Var chi2(4) = 8)"),
                tolerance=TOL_MC),
        finding("Under an unmodelled 0.2 rad/s turn during the gap, the expected NEES of the coasting track exceeds "
                "the 99% chi-square(4) quantile after a finite gap even though its covariance keeps growing; a "
                "radius rule protects against this only if its radius is below the ellipse radius reached by then",
                "numerical",
                as_json({k: turn[k] for k in ("omega_rad_s", "speed_m_s", "threshold_chi2_4_99",
                                              "first_inconsistent_gap_ticks", "radius_at_inconsistency_m",
                                              "gap_ticks_before_track_loss", "monte_carlo")}),
                {"derivation": "E[NEES] = tr(P_f^-1 (T P T^T + Q_turn)) + d^T P_f^-1 d for a linear turning truth",
                 **generator_basis(seed + 2, runs=mc.get("runs", 0)), "checks": [
                    check("analytic", "a finite gap (ticks) after which the coasting track is inconsistent",
                          turn["first_inconsistent_gap_ticks"] or 10_000, 399, "le"),
                    check("analytic", "Monte Carlo mean NEES of the turning truth at that gap against the "
                                      "expectation (z)", mc["z_vs_expected"], turn["z_critical"]),
                    check("analytic", "declared 1 m radius below the radius reached at inconsistency (margin, m)",
                          (turn["radius_at_inconsistency_m"] or 0.0) - study["radius_m"], 0.0, "signed_ge")]},
                uncertainty=mc95(mc["standard_error"], "standard error of the Monte Carlo mean NEES at the first "
                                                       "inconsistent gap; the expectation itself is exact"),
                tolerance={"abs": 1e-9, "rel": 1e-6}, counterexample={
                    "statement": "A coasting track stays statistically consistent for any gap because its covariance "
                                 "grows",
                    "witness": {"omega_rad_s": turn["omega_rad_s"],
                                "gap_ticks": turn["first_inconsistent_gap_ticks"],
                                "radius_at_inconsistency_m": turn["radius_at_inconsistency_m"]}}),
        unreal("A 1 m, 99% track-loss radius is a safe operating threshold", "machine_safety", seed,
               "not established: the threshold is a declared policy value"),
    ]
    fields = {
        "hypothesis": "After a sensor stops, the covariance grows by the model, the track-lost tick is predictable "
                      "exactly from a declared radius rule, a lost track is neither updated nor admitted (and a "
                      "refused update has no side effects), and reacquisition is an explicit, refusable operation "
                      "with an exact initial covariance.",
        "mathematical_model": "P(n) = F(n dt) P F(n dt)^T + Q(n dt); lost when sqrt(lambda_max(P_pos) chi2_2(0.99)) "
                              "> 1 m. Two-point initialization p = z2, v = (z2 - z1)/dt has error covariance "
                              "[[R, R/dt], [R/dt, 2R/dt^2 + q dt/3 I]], equal to J blockdiag(R, R, Q(dt)) J^T for the "
                              "error map J. A coordinated turn at omega has linear transition T(t); with white "
                              "acceleration through the turn, E[NEES] = tr(P_f^-1 (T P T^T + Q_turn(t))) + "
                              "d^T P_f^-1 d with d = (T - F) x_hat.",
        "input_data": [f"seed {seed}: one synthetic camera run (30 active ticks)",
                       f"seed {seed + 1}: {two['runs']} two-point Monte Carlo draws",
                       f"seed {seed + 2}: {mc.get('runs', 0)} turning-truth draws; turn rate "
                       f"{turn['omega_rad_s']} rad/s from the last estimated state"],
        "observation_model": "Camera positions until tick 30, then none; readings after loss are offered to fuse, "
                             "reacquire and admit.",
        "expected_invariant": "Observed lost tick = predicted; refusal codes as declared with no side effects; "
                              "two-point covariance exact; turning-truth Monte Carlo NEES equal to the expectation.",
        "experiment": "Run a FusionSession with a 1 m track radius, stop the sensor, predict tick by tick, then "
                      "exercise admit, fuse and reacquire (including readings older than the clock); compare the "
                      "reacquisition covariance with the error map; Monte Carlo the two-point estimator; evaluate "
                      "and simulate the NEES of the coasting track under an unmodelled turn.",
        "numerical_result": f"lost at tick {study['observed_lost_tick']} (predicted {study['predicted_lost_tick']}); "
                            f"refused fuse left the state unchanged: {study['refused_fuse_left_state_unchanged']}; "
                            f"two-point max |z| {two['max_abs_z']:.2f}, mean NEES {two['nees_mean']:.3f}; under the "
                            f"turn the coasting track becomes inconsistent after {turn['first_inconsistent_gap_ticks']} "
                            f"ticks (99% ellipse radius {(turn['radius_at_inconsistency_m'] or 0.0):.2f} m; simulated "
                            f"mean NEES {mc['mean_nees']:.2f}), while the 1 m rule had already declared loss after "
                            f"{turn['gap_ticks_before_track_loss']} ticks.",
        "uncertainty": "The lost tick and refusals are exact; the two-point and turning-truth statements carry Monte "
                       "Carlo error (Bonferroni and single z bounds); the turn expectation is exact given the "
                       "declared turn rate and process noise (Gauss-Legendre quadrature of a smooth integrand).",
        "failure_modes_checked": ["off-by-one in the loss tick", "fusion into a lost track", "side effects of a "
                                  "refused fusion", "admission of a lost track", "reacquisition from "
                                  "non-consecutive readings", "reacquisition moving the clock backwards",
                                  "redundant reacquisition", "unmodelled manoeuvre during the gap"],
        "unresolved_assumptions": ["Under the correct model a coasting track stays consistent (T069); the radius "
                                   "rule is a policy for model validity and data association, not a statistical "
                                   "necessity.",
                                   "Reacquisition assumes the readings belong to the lost target (no association "
                                   "ambiguity).",
                                   "The turning truth adds white acceleration through the turn dynamics; other "
                                   "manoeuvre noise models change the random part of the expectation."],
        "recommended_next_task": "T074: compare the fused state with the exact batch posterior.",
    }
    return outcome(fields, findings)


# T074 ------------------------------------------------------------------------------
def ground_truth_study(seed: int = 74_2026, runs: int = 200, ticks: int = 100, tracker_every: int = 5) -> dict:
    rng = generator(seed)
    F, Q = cv_model(DT, Q_SPECTRAL)
    truth = simulate_truth(rng, F, Q, MU0, P0_BENCH, runs, ticks)
    all_ticks = np.arange(1, ticks + 1)
    camera = measure(rng, truth, H_POS, R_CAMERA, all_ticks)
    tracker = measure(rng, truth, H_POS, R_TRACKER, all_ticks)
    zero = np.zeros((2, 2))
    stacked = (np.vstack([H_POS, H_POS]), np.block([[R_CAMERA, zero], [zero, R_TRACKER]]))
    plan = [stacked if k % tracker_every == 0 else (H_POS, R_CAMERA) for k in range(1, ticks + 1)]
    readings = [np.concatenate([camera[:, k - 1], tracker[:, k - 1]], axis=1) if k % tracker_every == 0
                else camera[:, k - 1] for k in range(1, ticks + 1)]
    steps = gain_schedule(F, Q, P0_BENCH, plan)
    estimates, _ = run_shared(F, MU0, steps, readings)
    comparisons = []
    for K in (10, 40, 100):
        d_mean = d_cov = 0.0
        for run in range(3):
            mean, block = batch_posterior_banded(F, Q, MU0, P0_BENCH, plan[:K], [r[run] for r in readings[:K]])
            d_mean = max(d_mean, float(np.max(np.abs(mean[K] - estimates[run, K])) / np.max(np.abs(estimates[run, K]))))
            d_cov = max(d_cov, float(np.max(np.abs(block - steps[K - 1].post)) / np.max(np.abs(steps[K - 1].post))))
        comparisons.append({"K": K, "batch_unknowns": 4 * (K + 1), "max_relative_mean_difference": d_mean,
                            "max_relative_covariance_difference": d_cov})
    # The dense normal equations (one LU of the whole information matrix, no recursion over time) at K = 10
    # and K = 40 for run 0, against the filter and against block elimination.
    dense_rows = []
    for K in (10, 40):
        dense_mean, dense_cov = batch_posterior(F, Q, MU0, P0_BENCH, plan[:K], [r[0] for r in readings[:K]],
                                                full=False)
        banded_mean, banded_cov = batch_posterior_banded(F, Q, MU0, P0_BENCH, plan[:K], [r[0] for r in readings[:K]])
        dense_rows.append({"K": K, "batch_unknowns": 4 * (K + 1),
                           "dense_vs_filter_mean": float(np.max(np.abs(dense_mean[K] - estimates[0, K]))
                                                         / np.max(np.abs(estimates[0, K]))),
                           "dense_vs_filter_covariance": float(np.max(np.abs(dense_cov - steps[K - 1].post))
                                                               / np.max(np.abs(steps[K - 1].post))),
                           "dense_vs_banded": max(float(np.max(np.abs(dense_mean - banded_mean))
                                                        / np.max(np.abs(dense_mean))),
                                                  float(np.max(np.abs(dense_cov - banded_cov))
                                                        / np.max(np.abs(banded_cov))))})
    dense_vs_filter = max(max(row["dense_vs_filter_mean"], row["dense_vs_filter_covariance"]) for row in dense_rows)
    dense_vs_banded = max(row["dense_vs_banded"] for row in dense_rows)
    worst_mean = max(row["max_relative_mean_difference"] for row in comparisons)
    worst_cov = max(row["max_relative_covariance_difference"] for row in comparisons)
    nees = nees_series(estimates, truth, steps)
    grand = run_mean_z(nees[..., None] - 4.0)
    grand_z, grand_se = float(grand[0][0]), float(grand[2][0])
    # Exact rational arithmetic: scalar random walk with twelve rational readings.
    exact_rng = generator(seed + 1)
    rational = [Fraction(int(v), 100) for v in exact_rng.integers(-300, 300, 12)]
    exact = exact_scalar_filter(Fraction(1, 10), Fraction(1, 4), Fraction(1), Fraction(0), rational)
    exact_equal = exact["filter_mean"] == exact["batch_mean"] and exact["filter_variance"] == exact["batch_variance"]
    return {"seed": seed, "runs": runs, "ticks": ticks, "comparisons": comparisons, "dense": dense_rows,
            "dense_vs_banded": dense_vs_banded, "dense_vs_filter": dense_vs_filter, "nees_grand_se": grand_se,
            "max_mean_difference": worst_mean,
            "max_covariance_difference": worst_cov, "nees": consistency(nees, 4), "nees_grand_z": grand_z,
            "anees": nees.mean(axis=0),
            "exact": {"readings": [str(v) for v in rational], "filter_mean": str(exact["filter_mean"]),
                      "batch_mean": str(exact["batch_mean"]), "filter_variance": str(exact["filter_variance"]),
                      "batch_variance": str(exact["batch_variance"]), "identical": exact_equal}}


@task("T074", changed_files=files("sensor_fusion_admission"), regression_tests=_tests(
    "T074", "test_fused_state_equals_batch_posterior"))
def analytic_ground_truth(ctx):
    study = ground_truth_study()
    seed = study["seed"]
    ctx.artifact_json("batch_reference.json", as_json(study))
    ticks = list(range(1, study["ticks"] + 1))
    ctx.artifact_text("anees_fused.svg", svg.line_plot(
        [("ANEES camera + tracker", ticks, study["anees"]),
         ("99% upper", [1, ticks[-1]], [study["nees"]["interval"][1]] * 2),
         ("99% lower", [1, ticks[-1]], [study["nees"]["interval"][0]] * 2)],
        title="T074 fused-state NEES against simulated truth", xlabel="tick", ylabel="ANEES", markers=False))
    ex = study["exact"]
    findings = [
        finding("The recursive covariance-form Kalman estimate and covariance equal the batch information-form "
                "posterior marginal at K = 10, 40 and 100 ticks to near roundoff, by block elimination and, at "
                "K = 10 and 40, by a dense solve with no recursion over time", "numerical",
                {"comparisons": study["comparisons"], "dense": study["dense"]},
                {**generator_basis(seed), "checks": [
                    check("analytic", "max relative mean difference against the block-eliminated batch posterior",
                          study["max_mean_difference"], 1e-9),
                    check("analytic", "max relative covariance difference against the block-eliminated batch "
                                      "posterior", study["max_covariance_difference"], 1e-9),
                    check("analytic", "dense normal equations against the filter at K = 10 and 40",
                          study["dense_vs_filter"], 1e-9),
                    check("invariant", "dense normal equations against block elimination at K = 10 and 40",
                          study["dense_vs_banded"], 1e-9)]},
                uncertainty=roundoff(max(study["max_mean_difference"], study["max_covariance_difference"],
                                         study["dense_vs_filter"]),
                                     "largest relative difference between the filter and the batch posteriors"),
                tolerance={"abs": 1e-9, "rel": 0.0}),
        finding("In exact rational arithmetic the scalar filter's final mean and variance are identical to the batch "
                "posterior", "mathematical", ex,
                {"derivation": "tridiagonal normal equations solved exactly in Fractions", "checks": [
                    check("exact_arithmetic", "filter minus batch (mean and variance, exact)",
                          float(not ex["identical"]), 0.0)]},
                uncertainty=exact("rational arithmetic"), tolerance=TOL_EXACT),
        finding("The fused estimate's error against the simulated truth is NEES-consistent: run-averaged NEES lies "
                "inside its per-tick 99% interval at 90% or more of ticks and the grand mean matches 4",
                "numerical", {"nees": study["nees"], "grand_z": study["nees_grand_z"]},
                {**generator_basis(seed, runs=study["runs"], ticks=study["ticks"]), "checks": [
                    check("analytic", "fraction of ticks with ANEES in the 99% interval",
                          study["nees"]["fraction_inside"], 0.9, "ge"),
                    check("analytic", "grand mean NEES against 4 (run-level z)", study["nees_grand_z"],
                          bonferroni(2))]},
                uncertainty=mc95(study["nees_grand_se"], "run-level standard error of the grand-mean NEES"),
                tolerance=TOL_MC),
        unreal("The fused estimate equals the physical state of a real target within its covariance", "physical",
               seed, "not established: agreement is with the declared model's exact posterior and simulated truth"),
    ]
    fields = {
        "hypothesis": "For the linear-Gaussian bench the Kalman recursion is the exact posterior: its estimate and "
                      "covariance equal the batch information-form posterior (by block elimination, and by a dense "
                      "solve without recursion over time), and its errors against the simulated truth are "
                      "NEES-consistent.",
        "mathematical_model": "Batch posterior over x_0..x_K minimizes |x_0 - mu0|^2_P0 + sum |x_k - F x_{k-1}|^2_Q + "
                              "sum |z_k - H x_k|^2_R (normal equations, information matrix of size 4(K+1)); its "
                              "marginal at K is the filtering posterior. Camera every tick, tracker every 5 ticks "
                              "(stacked, R = blockdiag(R_camera, 0.0025 I)).",
        "input_data": [f"seed {seed} (PCG64)", f"{study['runs']} runs x {study['ticks']} ticks",
                       "3 runs compared with the block-eliminated batch posterior at K = 10, 40, 100; run 0 also "
                       "with the dense solve at K = 10 and 40 (44 and 164 unknowns)",
                       f"12 rational readings (seed {seed + 1}) for the exact scalar check"],
        "observation_model": "Linear position readings with the declared covariances; the filter model is the truth "
                             "model.",
        "expected_invariant": "Kalman = batch to roundoff; rational filter = rational batch exactly; ANEES inside "
                              "chi2(4N)/N.",
        "experiment": "Solve the normal equations for each prefix (block elimination, whose forward pass is an "
                      "information filter, and a dense LU) and compare the last block with the covariance-form "
                      "filter; run the exact Fraction filter and elimination; Monte Carlo NEES against the "
                      "simulated truth.",
        "numerical_result": f"max relative mean difference {study['max_mean_difference']:.1e}, covariance "
                            f"{study['max_covariance_difference']:.1e}; dense solve against the filter "
                            f"{study['dense_vs_filter']:.1e}; exact rational identity {ex['identical']}; "
                            f"ANEES inside {study['nees']['fraction_inside']:.2f} of ticks, grand NEES "
                            f"{study['nees']['grand_mean']:.3f}.",
        "uncertainty": "Floating-point agreement is limited by the conditioning of the information matrix; the "
                       "rational comparison has no roundoff; NEES consistency carries Monte Carlo error.",
        "failure_modes_checked": ["recursion vs batch disagreement", "covariance vs information inverse",
                                  "roundoff hiding a discrepancy (exact arithmetic)", "NEES inconsistency"],
        "unresolved_assumptions": ["Equality with the batch posterior validates the algebra for the declared "
                                   "model, not the model against reality.",
                                   "Nonlinear sensors (encoder speed, IMU heading rate) are not fused here."],
        "recommended_next_task": "T075: keep the fused estimate a candidate until an explicit admission gate.",
    }
    return outcome(fields, findings)


# T075 ------------------------------------------------------------------------------
class _Imposter(CandidateState):
    """A subclass carrying an issued candidate's exact fields (used only as a mutation probe)."""


def _admission_scenarios():
    """Violation scenarios, each targeting one admission check: {scenario: (check name, builder)}.

    Each builder returns (session, candidate, declared). Private access to session
    state simulates internal corruption that the public API cannot produce.
    """
    truth, z = _camera_run(75_2026, 40)

    def tracked(**kwargs):
        session = _session(**kwargs)
        session.initialize(MU0, P0_BENCH, 0)
        candidate = None
        for k in range(1, 11):
            candidate = session.fuse(Observation("camera", "world", k, tuple(z[k - 1]), R_CAMERA, "cam-cal"))
        return session, candidate

    def declared_missing():
        session, candidate = tracked()
        return session, candidate, dict(ADMIT, nis_probability=None)

    def writable():
        # A session constructed read-only (the flag cannot be flipped afterwards) holding, through private
        # access, the internal state of a writable one: only the writable check stands between them.
        session, candidate = tracked()
        frozen = FusionSession(read_only=True, dt=DT, q=Q_SPECTRAL)
        for name in ("x", "P", "tick", "track_status", "calibrations", "revoked", "_issued", "_latest"):
            setattr(frozen, name, getattr(session, name))
        return frozen, candidate, dict(ADMIT)

    def typed():
        session, candidate = tracked()
        return session, _Imposter(**asdict(candidate)), dict(ADMIT)

    def finite():
        session, _ = tracked()
        session.x = session.x.copy()
        session.x[0] = math.nan
        return session, session._issue(), dict(ADMIT)

    def covariance():
        session, _ = tracked()
        session.P = session.P.copy()
        session.P[3, 3] = -session.P[3, 3]
        return session, session._issue(), dict(ADMIT)

    def integrity():
        session, candidate = tracked()
        return session, replace(candidate, mean=(candidate.mean[0] + 5.0,) + candidate.mean[1:]), dict(ADMIT)

    def provenance():
        session, candidate = tracked()
        fields = {k: v for k, v in asdict(candidate).items() if k != "digest"}
        fields["mean"] = (candidate.mean[0] + 5.0,) + candidate.mean[1:]
        return session, CandidateState(digest=_digest({"kind": "candidate", **fields}), **fields), dict(ADMIT)

    def frame():
        session, candidate = tracked()
        return session, candidate, dict(ADMIT, expected_frame_id="body")

    def stale():
        session, candidate = tracked()
        session.predict(session.tick + 1)
        return session, candidate, dict(ADMIT)

    def superseded():
        # A second update at the same tick supersedes the first candidate without moving the clock.
        session, candidate = tracked()
        session.register_calibration(CalibrationRecord("trk-cal", "tracker", "world", 0, 1_000_000))
        session.fuse(Observation("tracker", "world", session.tick, tuple(truth[session.tick, :2]), R_TRACKER,
                                 "trk-cal"))
        return session, candidate, dict(ADMIT)

    def track():
        session, _ = tracked(track_radius=0.25)
        candidate = session.predict(session.tick + 40)
        return session, candidate, dict(ADMIT)

    def uncertainty():
        session, _ = tracked()
        candidate = session.predict(session.tick + 40)
        return session, candidate, dict(ADMIT, max_position_std=0.1)

    def innovation():
        session, _ = tracked()
        outlier = tuple(np.asarray(z[10]) + 3.0)
        return session, session.fuse(Observation("camera", "world", 11, outlier, R_CAMERA, "cam-cal")), dict(ADMIT)

    def calibration():
        session, candidate = tracked()
        session.revoke_calibration("cam-cal")
        return session, candidate, dict(ADMIT)

    return {"declared": ("declared", declared_missing), "writable": ("writable", writable),
            "typed": ("typed", typed), "finite": ("finite", finite), "covariance": ("covariance", covariance),
            "integrity": ("integrity", integrity), "provenance": ("provenance", provenance),
            "frame": ("frame", frame), "stale_tick": ("fresh", stale), "superseded_same_tick": ("fresh", superseded),
            "track": ("track", track), "uncertainty": ("uncertainty", uncertainty),
            "innovation": ("innovation", innovation), "calibration": ("calibration", calibration)}, tracked


def admission_study() -> dict:
    scenarios, tracked = _admission_scenarios()
    codes = {name: code for name, code, _ in ADMISSION_CHECKS}
    rows = {}
    for scenario, (check_name, build) in scenarios.items():
        session, candidate, declared = build()
        full = refusal_code(lambda: session.admit(candidate, **declared))
        reduced = tuple(c for c in ADMISSION_CHECKS if c[0] != check_name)
        mutant_code, _ = admission_verdict(session, candidate, declared, reduced)
        rows[scenario] = {"check": check_name, "expected": codes[check_name], "full_gate": full,
                          "without_this_check": mutant_code or "admitted",
                          "killed": (mutant_code or "admitted") != full}
    uncovered = sorted(set(codes) - {row["check"] for row in rows.values()})
    backed_up = sorted(name for name, row in rows.items() if row["without_this_check"] != "admitted")
    # Positive control and no auto-admission.
    session, candidate = tracked()
    auto = len(session.admitted)
    admitted = session.admit(candidate, **ADMIT)
    types = (type(Observation), Observation, CandidateState, AdmittedState)
    relations = [issubclass(a, b) for a in types[1:] for b in types[1:] if a is not b]
    mutations = {
        "construct_admitted_directly": refusal_code(lambda: AdmittedState(candidate, (), {}, {})),
        "mutate_admitted": refusal_code(lambda: setattr(admitted, "mean", (0.0,) * 4)),
    }
    try:
        candidate.mean = (0.0,) * 4
        mutations["mutate_candidate"] = "none"
    except FrozenInstanceError:
        mutations["mutate_candidate"] = "frozen_instance"
    try:
        Observation("camera", "world", 1, (0.0, 0.0), R_CAMERA, "cam-cal").value = (1.0, 1.0)
        mutations["mutate_observation"] = "none"
    except FrozenInstanceError:
        mutations["mutate_observation"] = "frozen_instance"
    return {"checks": [name for name, _, _ in ADMISSION_CHECKS], "rows": rows, "uncovered_checks": uncovered,
            "scenarios": len(rows), "killed": sum(r["killed"] for r in rows.values()),
            "sole_guard": sum(r["without_this_check"] == "admitted" for r in rows.values()),
            "backed_up_scenarios": backed_up,
            "refused_as_expected": sum(r["full_gate"] == r["expected"] for r in rows.values()),
            "auto_admitted_before_gate": auto, "admitted_after_gate": len(session.admitted),
            "admitted_checks": list(admitted.checks), "admitted_authority": dict(admitted.authority),
            "admitted_digest_matches_candidate": admitted.candidate_digest == candidate.digest,
            "type_relations_true": sum(relations), "mutations": mutations}


@task("T075", changed_files=files("sensor_fusion_admission"), regression_tests=_tests(
    "T075", "test_typed_objects_and_admission_mutations"))
def typed_admission(ctx):
    study = admission_study()
    ctx.artifact_json("admission_mutations.json", as_json(study))
    rows, total = study["rows"], study["scenarios"]
    checks = len(study["checks"])
    mutations = study["mutations"]
    findings = [
        finding("Every admission check refuses the candidates built to violate it, with that check's own refusal "
                "code, including a candidate superseded by a second update at the same tick", "computational_pipeline",
                {name: {k: row[k] for k in ("check", "expected", "full_gate")} for name, row in rows.items()},
                {"derivation": "ADMISSION_CHECKS in ciw.lab.sensor_fusion_objects", "checks": [
                    refusal(f"admission of a candidate in scenario '{name}' (check '{row['check']}')", row["expected"],
                            row["full_gate"]) for name, row in rows.items()] + [
                    check("exact_arithmetic", "admission checks without a violation scenario",
                          len(study["uncovered_checks"]), 0)]},
                uncertainty=exact("refusal codes"), tolerance=TOL_EXACT),
        finding("Mutation analysis: deleting the check a scenario targets changes that scenario's outcome for every "
                "scenario, and the violating candidate is then admitted except in two scenarios where a later check "
                "still refuses it (a missing declaration fails the innovation check closed; a forged candidate is "
                "not the latest issue, so the freshness check refuses it)", "computational_pipeline",
                {"checks": checks, "scenarios": total, "mutants_killed": study["killed"],
                 "sole_guard_scenarios": study["sole_guard"], "backed_up_scenarios": study["backed_up_scenarios"],
                 "without_targeted_check": {name: row["without_this_check"] for name, row in rows.items()}},
                {"derivation": "admission_verdict with one check removed", "checks": [
                    check("exact_arithmetic", "surviving mutants", total - study["killed"], 0),
                    check("invariant", "scenarios backed up by a later check differ from {declared, provenance}",
                          is_not(study["backed_up_scenarios"], ["declared", "provenance"]), 0.0),
                    refusal("forged candidate with the provenance check deleted", "stale_candidate",
                            rows["provenance"]["without_this_check"])]},
                uncertainty=exact("deterministic gate outcomes"), tolerance=TOL_EXACT),
        finding("Observations, candidates and admitted states are unrelated types; candidates are never admitted "
                "automatically; an admitted state exists only through the gate and is immutable",
                "computational_pipeline",
                {"type_relations_true": study["type_relations_true"],
                 "auto_admitted_before_gate": study["auto_admitted_before_gate"],
                 "admitted_after_gate": study["admitted_after_gate"], "admitted_checks": study["admitted_checks"],
                 "admitted_digest_matches_candidate": study["admitted_digest_matches_candidate"], **mutations},
                {"derivation": "class definitions and the gate token in ciw.lab.sensor_fusion_objects", "checks": [
                    check("exact_arithmetic", "subclass relations among the three types", study["type_relations_true"],
                          0),
                    check("exact_arithmetic", "states admitted without calling the gate",
                          study["auto_admitted_before_gate"], 0),
                    check("exact_arithmetic", "admitted states after one explicit gate call minus one",
                          study["admitted_after_gate"] - 1, 0),
                    refusal("construct AdmittedState without the gate", "admission_requires_gate",
                            mutations["construct_admitted_directly"]),
                    refusal("mutate an admitted state", "admitted_state_immutable", mutations["mutate_admitted"]),
                    check("invariant", "a candidate accepted an attribute assignment (frozen dataclass)",
                          is_not(mutations["mutate_candidate"], "frozen_instance"), 0.0),
                    check("invariant", "an observation accepted an attribute assignment (frozen dataclass)",
                          is_not(mutations["mutate_observation"], "frozen_instance"), 0.0)]},
                uncertainty=exact("deterministic type and gate outcomes"), tolerance=TOL_EXACT),
        unreal("An admitted synthetic state may command actuators", "actuator_authority", 75_2026,
               "not established: admission is a software gate over synthetic data; authority is decided elsewhere"),
    ]
    fields = {
        "hypothesis": "Keeping observations, candidate states and admitted states as separate types, with a single "
                      "explicit gate whose every check is load-bearing, guards against a candidate becoming state "
                      "by accident, naive tampering or omission.",
        "mathematical_model": "Admission = ordered conjunction of declared checks (declared, writable, typed, "
                              "finite, covariance, integrity, provenance, frame, fresh, track, uncertainty, "
                              "innovation, calibration); fresh means the most recently issued candidate at the "
                              "session tick; fail closed on any exception. Mutation m_i removes check i.",
        "input_data": ["one synthetic camera run (seed 752026) driving a FusionSession for each scenario",
                       f"{total} violation scenarios covering all {checks} checks, plus one valid control"],
        "observation_model": "Typed Observation objects; candidates issued by the session and sealed by a content "
                             "digest.",
        "expected_invariant": "Full gate refuses each scenario with its check's code; the gate without that check "
                              "gives a different outcome; no admission without an explicit gate call; AdmittedState "
                              "immutable.",
        "experiment": "Build each violating candidate (private state corruption simulates internal faults), run "
                      "the full gate and the gate with the targeted check deleted, and exercise construction and "
                      "mutation of each type.",
        "numerical_result": f"{study['refused_as_expected']}/{total} scenarios refused with the expected code; "
                            f"{study['killed']}/{total} mutants killed; in {study['sole_guard']} scenarios the "
                            f"targeted check is the sole guard; backed up by a later check: "
                            f"{', '.join(study['backed_up_scenarios'])} (the provenance check adds a specific "
                            f"refusal code, but freshness already implies an issued digest).",
        "uncertainty": "None: the outcomes are deterministic.",
        "failure_modes_checked": ["subclass imposter", "tampered mean with a stale digest", "forged digest",
                                  "NaN state", "indefinite covariance", "candidate stale by tick",
                                  "candidate superseded at the same tick", "lost track", "excess uncertainty",
                                  "inconsistent innovation", "revoked calibration", "read-only session",
                                  "missing declaration", "frame mismatch"],
        "unresolved_assumptions": ["Python objects can be forced (object.__setattr__) by code with that intent; the "
                                   "gate defends against accident and naive tampering, not a hostile process.",
                                   "The declared thresholds are inputs; choosing them is outside this experiment."],
        "recommended_next_task": "T076: confirm that the session defaults to read-only with sensor fusion and state "
                                 "admission not performed.",
    }
    return outcome(fields, findings)


# T076 ------------------------------------------------------------------------------
def defaults_study() -> dict:
    from ..declared_workload import AUTHORITY  # CIW vocabulary; imported lazily

    session = FusionSession()
    signature = inspect.signature(FusionSession)
    obs = Observation("camera", "world", 1, (0.0, 0.0), R_CAMERA, "cam-cal")
    session.register_calibration(CalibrationRecord("cam-cal", "camera", "world", 0, 100))
    fuse_code = refusal_code(lambda: session.fuse(obs))
    record_entry = session.record(Observation("camera", "world", 2, (0.1, 0.1), R_CAMERA, "cam-cal"))
    stub = CandidateState("world", 0, (0.0,) * 4, tuple(map(tuple, np.eye(4))), (), (), "tracking", (),
                          tuple(sorted(DEFAULT_AUTHORITY.items())), "x", "0")
    codes = {
        "initialize": refusal_code(lambda: session.initialize(MU0, P0_BENCH, 0)),
        "predict": refusal_code(lambda: session.predict(1)),
        "handle_gap": refusal_code(lambda: session.handle_gap("camera", 1)),
        "fuse": fuse_code,
        "reacquire": refusal_code(lambda: session.reacquire(obs, obs)),
        "admit": refusal_code(lambda: session.admit(stub, **ADMIT)),
    }
    rebinding = {"read_only": refusal_code(lambda: setattr(session, "read_only", False)),
                 "authority": refusal_code(lambda: setattr(session, "authority", dict(SYNTHETIC_AUTHORITY)))}
    try:
        session.authority["sensor_fusion"] = "performed"
        authority_mutable = True
    except TypeError:
        authority_mutable = False
    enabled = FusionSession(read_only=False)
    keys = ("state_admission", "sensor_fusion", "physical_truth")
    return {"read_only_default": signature.parameters["read_only"].default,
            "session_read_only": session.read_only, "session_authority": dict(session.authority),
            "ciw_authority": {key: AUTHORITY.get(key, "missing") for key in keys} | dict(AUTHORITY),
            "matches_ciw": dict(session.authority) == dict(AUTHORITY),
            "default_matches_module_constant": DEFAULT_AUTHORITY == dict(AUTHORITY),
            "codes": codes, "rebinding": rebinding, "authority_mutable": authority_mutable,
            "still_read_only_after_rebinding": session.read_only and dict(session.authority) == DEFAULT_AUTHORITY,
            "log": [entry["disposition"] for entry in session.log],
            "record_disposition": record_entry["disposition"],
            "no_state": session.x is None and session.P is None and session.tick is None,
            "admitted": len(session.admitted),
            "enabled_authority": {key: enabled.authority.get(key, "missing") for key in keys},
            "enabled_matches_synthetic": dict(enabled.authority) == SYNTHETIC_AUTHORITY}


@task("T076", changed_files=files("sensor_fusion_admission"), regression_tests=_tests(
    "T076", "test_defaults_are_read_only_and_not_performed"))
def read_only_defaults(ctx):
    study = defaults_study()
    ctx.artifact_json("defaults.json", as_json(study))
    codes = study["codes"]
    authority = study["ciw_authority"]
    enabled = study["enabled_authority"]
    findings = [
        finding("The CIW authority vocabulary declares state_admission and sensor_fusion not_performed and "
                "physical_truth not_established, and the fusion session's default authority equals it",
                "computational_pipeline",
                {"ciw_authority": authority, "session_authority": study["session_authority"],
                 "matches": study["matches_ciw"]},
                {"derivation": "ciw.declared_workload.AUTHORITY and FusionSession()", "checks": [
                    check("invariant", "CIW AUTHORITY sensor_fusion is not not_performed",
                          is_not(authority["sensor_fusion"], "not_performed"), 0.0),
                    check("invariant", "CIW AUTHORITY state_admission is not not_performed",
                          is_not(authority["state_admission"], "not_performed"), 0.0),
                    check("invariant", "CIW AUTHORITY physical_truth is not not_established",
                          is_not(authority["physical_truth"], "not_established"), 0.0),
                    check("invariant", "default session authority differs from CIW AUTHORITY",
                          float(not study["matches_ciw"]), 0.0),
                    check("invariant", "module DEFAULT_AUTHORITY differs from CIW AUTHORITY",
                          float(not study["default_matches_module_constant"]), 0.0)]},
                uncertainty=exact("string comparison of declared constants"), tolerance=TOL_EXACT),
        finding("FusionSession defaults to read-only: every estimation or admission call (initialize, predict, "
                "handle_gap, fuse, reacquire, admit) is refused with read_only_session, observations are still "
                "retained, and no state or admitted state exists", "computational_pipeline",
                {"read_only_default": study["read_only_default"], "codes": codes, "log": study["log"],
                 "record_disposition": study["record_disposition"], "no_state": study["no_state"],
                 "admitted": study["admitted"]},
                {"derivation": "FusionSession._writable and the admission gate", "checks": [
                    check("invariant", "read_only default is not True", float(study["read_only_default"] is not True),
                          0.0)] + [refusal(f"{name} on a default session", "read_only_session", code)
                                   for name, code in codes.items()] + [
                    check("invariant", "a state (mean, covariance or clock) exists on a default session",
                          float(not study["no_state"]), 0.0),
                    check("exact_arithmetic", "admitted states on a default session", study["admitted"], 0),
                    check("invariant", "fused observation not retained with its refusal",
                          is_not(study["log"][0], "refused:read_only_session"), 0.0)]},
                uncertainty=exact("refusal codes"), tolerance=TOL_EXACT),
        finding("The read-only flag and the authority record are fixed at construction: rebinding either is refused "
                "with read_only_session and the authority record cannot be edited in place", "computational_pipeline",
                {"rebinding": study["rebinding"], "authority_mutable": study["authority_mutable"],
                 "still_read_only_after_rebinding": study["still_read_only_after_rebinding"]},
                {"derivation": "FusionSession.__setattr__ and the read-only authority mapping", "checks": [
                    refusal("assign read_only = False on a default session", "read_only_session",
                            study["rebinding"]["read_only"]),
                    refusal("assign a synthetic_only authority on a default session", "read_only_session",
                            study["rebinding"]["authority"]),
                    check("invariant", "authority record accepted an item assignment",
                          float(study["authority_mutable"]), 0.0),
                    check("invariant", "session no longer read-only with the default authority after the attempts",
                          float(not study["still_read_only_after_rebinding"]), 0.0)]},
                uncertainty=exact("refusal codes"), tolerance=TOL_EXACT),
        finding("Enabling fusion explicitly at construction changes the authority only to synthetic_only; physical "
                "truth stays not_established", "computational_pipeline", enabled,
                {"derivation": "SYNTHETIC_AUTHORITY in ciw.lab.sensor_fusion_objects", "checks": [
                    check("invariant", "enabled session sensor_fusion is not synthetic_only",
                          is_not(enabled["sensor_fusion"], "synthetic_only"), 0.0),
                    check("invariant", "enabled session state_admission is not synthetic_only",
                          is_not(enabled["state_admission"], "synthetic_only"), 0.0),
                    check("invariant", "enabled session physical_truth is not not_established",
                          is_not(enabled["physical_truth"], "not_established"), 0.0)]},
                uncertainty=exact("string comparison of declared constants"), tolerance=TOL_EXACT),
        unreal("Synthetic fusion output is admissible as production state", "production_acceptance", 76_2026,
               "not established: sensor_fusion and state_admission are not_performed by default and synthetic_only "
               "when enabled"),
    ]
    fields = {
        "hypothesis": "The fusion API cannot estimate, fuse or admit anything unless a caller constructs a writable "
                      "session, the flag cannot be flipped afterwards, and its authority record uses the CIW "
                      "vocabulary (sensor_fusion and state_admission not_performed).",
        "mathematical_model": "Not numerical: a default-argument and refusal audit of FusionSession against "
                              "ciw.declared_workload.AUTHORITY.",
        "input_data": ["FusionSession() with default arguments", "ciw.declared_workload.AUTHORITY"],
        "observation_model": "One camera observation offered to fuse and record.",
        "expected_invariant": "read_only default True; authority == CIW AUTHORITY; initialize, predict, handle_gap, "
                              "fuse, reacquire and admit refused with read_only_session; record allowed; rebinding "
                              "read_only or authority refused; authority mapping immutable.",
        "experiment": "Inspect the constructor signature, compare authority dictionaries, call every estimation and "
                      "admission method on a default session, try to rebind the flag and edit the authority, then "
                      "construct a session with read_only=False.",
        "numerical_result": f"read_only default {study['read_only_default']}; authority {study['session_authority']}; "
                            f"refusals {sorted(set(codes.values()))}; rebinding {sorted(set(study['rebinding'].values()))}; "
                            f"enabled authority {enabled}.",
        "uncertainty": "None: the outcomes are deterministic.",
        "failure_modes_checked": ["writable default", "authority drift from the CIW vocabulary", "silent fusion in "
                                  "a read-only session", "loss of a refused observation", "authority upgrade on "
                                  "enabling fusion", "flipping read_only after construction", "editing the authority "
                                  "record in place", "a missing vocabulary key (reported as missing, failing the "
                                  "checks rather than raising)"],
        "unresolved_assumptions": ["The CIW vocabulary is read from ciw.declared_workload; if it changes, this "
                                   "task's checks fail rather than silently following it.",
                                   "Calibration records can be registered and revoked on a read-only session; they "
                                   "are record-keeping, not estimation, and grant nothing."],
        "recommended_next_task": "T113: connect filtered residuals to the Lyapunov runtime without putting sensors "
                                 "inside the kernel.",
        "provider_runtime_identity": identity(files("sensor_fusion_admission"), DECLARED_WORKLOAD),
    }
    return outcome(fields, findings)
