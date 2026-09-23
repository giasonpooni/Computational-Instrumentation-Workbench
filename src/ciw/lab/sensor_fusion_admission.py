"""Calibration expiry, track loss, analytic reference, typed admission and defaults (T072-T076).

Scope: T072 retains readings taken after a calibration's validity interval but
refuses to fuse them. T073 lets a sensor stop, predicts the exact tick at
which the track is declared lost, refuses fusion and admission of a lost
track and requires explicit two-point reacquisition. T074 checks the Kalman
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
from .sensor_fusion_bench import (H_POS, batch_posterior, batch_posterior_banded, chi2_quantile, consistency, cv_model, exact_scalar_filter,
                                  gain_schedule, generator, measure, nees_series, run_shared, simulate_truth)
from .sensor_fusion_common import (MU0, P0_BENCH, R_CAMERA, TESTS, TOL_EXACT, TOL_MC, TOL_TINY, as_json, bonferroni,
                                   check, covariance_z, files, generator_basis, outcome, refusal, refusal_code,
                                   run_mean_z, unreal)
from .sensor_fusion_objects import (ADMISSION_CHECKS, DEFAULT_AUTHORITY, SYNTHETIC_AUTHORITY, AdmittedState,
                                    CalibrationRecord, CandidateState, FusionRefusal, FusionSession, Observation,
                                    _digest, admission_verdict)

DT, Q_SPECTRAL = 0.1, 0.05
R_TRACKER = 0.0025 * np.eye(2)
ADMIT = {"expected_frame_id": "world", "nis_probability": 0.99, "max_position_std": 10.0}


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
    fused = [k for k, code in codes.items() if code == "no_refusal"]
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
    other = {
        "revoked": refusal_code(lambda: session.fuse(Observation("camera", "world", ticks + 7, tuple(z[ticks + 6]),
                                                                 R_CAMERA, "cam-cal-3"))),
        "unknown": refusal_code(lambda: session.fuse(Observation("camera", "world", ticks + 7, tuple(z[ticks + 6]),
                                                                 R_CAMERA, "cam-cal-9"))),
        "other_sensor": refusal_code(lambda: session.fuse(Observation("camera", "world", ticks + 7,
                                                                      tuple(z[ticks + 6]), R_CAMERA, "trk-cal"))),
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
                            "final_position_std_fusing": float(math.sqrt(fusing[-1].post[0, 0]))},
            "anees_refusing": nees_refusing.mean(axis=0), "anees_fusing": nees_fusing.mean(axis=0)}


@task("T072", changed_files=files("sensor_fusion_admission", "sensor_fusion_objects"), regression_tests=(
    f"{TESTS}::test_calibration_expiry_retains_but_refuses",
    f"{TESTS}::test_section_reports_labels_and_states"))
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
                    refusal("reading at tick 59 (inside the interval)", "no_refusal", boundary["tick_before_expiry"]),
                    refusal("reading at tick 60 (first tick outside)", "calibration_expired",
                            boundary["tick_at_expiry"]),
                    check("exact_arithmetic", "readings under the renewed record refused",
                          sum(code != "no_refusal" for code in study["renewed"]), 0)]},
                tolerance=TOL_EXACT),
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
                tolerance=TOL_EXACT),
        finding("Revoked, unknown and other-sensor calibrations are refused, and the expired record stays expired "
                "after a renewal is registered", "computational_pipeline", other,
                {"derivation": "FusionSession._admissible_source", "checks": [
                    refusal("revoked calibration", "calibration_revoked", other["revoked"]),
                    refusal("unregistered calibration id", "calibration_unknown", other["unknown"]),
                    refusal("calibration registered for another sensor", "calibration_unknown", other["other_sensor"]),
                    refusal("expired record after renewal", "calibration_expired",
                            other["stale_record_after_renewal"])]},
                tolerance=TOL_EXACT),
        finding("Under a declared post-expiry drift of 1 cm per tick, refusing expired readings keeps NEES "
                "consistent (with a growing covariance) while fusing them makes the filter overconfident",
                "numerical", {k: mc[k] for k in mc},
                {**generator_basis(seed + 1, runs=mc["runs"]), "checks": [
                    check("analytic", "refusing filter: fraction of post-expiry ticks inside the 99% interval",
                          mc["refusing"]["fraction_inside"], 0.9, "ge"),
                    check("analytic", "fusing filter: fraction of post-expiry ticks above the 99% interval",
                          mc["fusing"]["fraction_above"], 0.5, "ge")]},
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
                      "readings, register a renewal and exercise revoked/unknown/other-sensor records; Monte Carlo "
                      "NEES of refusing and fusing filters under the declared drift.",
        "numerical_result": f"fused {study['fused']}, refused {study['refused_expired']} (first at tick "
                            f"{study['first_refused']}), retained {study['retained']}; shadow equality "
                            f"{study['state_equals_prediction_only_shadow']}; post-expiry ANEES inside: refusing "
                            f"{mc['refusing']['fraction_inside']:.2f}, fusing grand NEES "
                            f"{mc['fusing']['grand_mean']:.1f}; final position std refusing "
                            f"{mc['final_position_std_refusing']:.3f} m vs fusing {mc['final_position_std_fusing']:.3f} m.",
        "uncertainty": "The API statements are exact; the Monte Carlo statement depends on the declared drift, "
                       "which is an assumption, not a measurement.",
        "failure_modes_checked": ["boundary tick off-by-one", "refused reading changing the state", "loss of the "
                                  "refused reading", "revoked record", "unknown record", "record for another sensor",
                                  "expired record reused after renewal"],
        "unresolved_assumptions": ["Expiry is a hard tick boundary; real calibrations degrade gradually and their "
                                   "validity depends on temperature, shocks and use.",
                                   "Readings are not re-labelled retroactively under a renewed calibration."],
        "recommended_next_task": "T073: a sensor that stops altogether, and the explicit track-lost state.",
    }
    return outcome(fields, findings)


# T073 ------------------------------------------------------------------------------
def two_point_covariance(R1, R2, dt, q) -> np.ndarray:
    return np.block([[R2, R2 / dt], [R2 / dt, (R1 + R2) / dt ** 2 + q * dt / 3 * np.eye(2)]])


def turn_state(x0, omega, t):
    """Constant-speed coordinated turn: position and velocity after time t from x0 = (p, v)."""
    p, v = x0[:2], x0[2:]
    s, c = math.sin(omega * t), math.cos(omega * t)
    rot = np.array([[c, -s], [s, c]])
    integral = np.array([[s, c - 1.0], [1.0 - c, s]]) / omega
    return np.concatenate([p + integral @ v, rot @ v])


def track_lost_study(seed: int = 73_2026, active: int = 30, radius: float = 1.0, omega: float = 0.2,
                     mc_runs: int = 20_000) -> dict:
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
           range(observed_lost + 1, observed_lost + 6)}
    first = observed_lost + 1
    codes = {
        "admit_lost": refusal_code(lambda: session.admit(lost_candidate, **ADMIT)),
        "fuse_lost": refusal_code(lambda: session.fuse(obs[first])),
        "reacquire_gap": refusal_code(lambda: session.reacquire(obs[first + 1], obs[first + 3])),
    }
    reacquired = session.reacquire(obs[first + 1], obs[first + 2])
    formula = two_point_covariance(R_CAMERA, R_CAMERA, DT, Q_SPECTRAL)
    reacquire_error = float(np.max(np.abs(np.asarray(reacquired.covariance) - formula)))
    codes["reacquire_while_tracking"] = refusal_code(lambda: session.reacquire(obs[first + 2], obs[first + 3]))
    codes["fuse_after_reacquisition"] = refusal_code(lambda: session.fuse(obs[first + 3]))
    codes["admit_after_reacquisition"] = refusal_code(
        lambda: session.admit(session._issued[list(session._issued)[-1]], **ADMIT))
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

    # Unmodelled coordinated turn during the gap: expected NEES = 4 + e^T P(n)^-1 e.
    threshold = chi2_quantile(0.99, 4)
    x_turn_start = x_active.copy()
    consistent_until, table = None, []
    for n in range(1, 400):
        t = n * DT
        Fn, Qn = cv_model(t, Q_SPECTRAL)
        Pn = Fn @ P_active @ Fn.T + Qn
        e = turn_state(x_turn_start, omega, t) - Fn @ x_turn_start
        expected = 4.0 + float(e @ np.linalg.solve(Pn, e))
        if n % 5 == 0:
            table.append({"gap_ticks": n, "expected_nees": expected})
        if expected > threshold:
            consistent_until = n
            radius_then = math.sqrt(float(np.linalg.eigvalsh(Pn[:2, :2]).max()) * gate)
            break
    return {"seed": seed, "active_ticks": active, "radius_m": radius, "track_probability": 0.99,
            "predicted_lost_tick": predicted_lost, "observed_lost_tick": observed_lost,
            "radius_at_loss": radii[-1], "codes": codes, "reacquire_covariance_error": reacquire_error,
            "dispositions": {d: dispositions.count(d) for d in sorted(set(dispositions))},
            "two_point": {"runs": mc_runs, "max_abs_z": float(np.max(np.abs(z_two))), "z_critical": bonferroni(11),
                          "nees_mean": float(nees.mean()), "nees_z": nees_z, "sample": S, "formula": formula},
            "turn": {"omega_rad_s": omega, "speed_m_s": float(np.linalg.norm(x_active[2:])),
                     "threshold_chi2_4_99": threshold, "first_inconsistent_gap_ticks": consistent_until,
                     "radius_at_inconsistency_m": radius_then if consistent_until else None,
                     "gap_ticks_before_track_loss": observed_lost - active, "table": table}}


@task("T073", changed_files=files("sensor_fusion_admission", "sensor_fusion_objects"), regression_tests=(
    f"{TESTS}::test_track_lost_prediction_refusals_and_reacquisition",
    f"{TESTS}::test_section_reports_labels_and_states"))
def track_lost(ctx):
    study = track_lost_study()
    seed, codes, two, turn = study["seed"], study["codes"], study["two_point"], study["turn"]
    ctx.artifact_json("track_lost.json", as_json(study))
    ctx.artifact_text("turn_nees.svg", svg.line_plot(
        [("expected NEES under a 0.2 rad/s turn", [r["gap_ticks"] for r in turn["table"]],
          [r["expected_nees"] for r in turn["table"]]),
         ("chi2_4 99%", [10, max(10, turn["table"][-1]["gap_ticks"] if turn["table"] else 10)],
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
                tolerance={"abs": 1e-9, "rel": 1e-9}),
        finding("A lost track is not admissible and cannot be updated; reacquisition is explicit, needs two "
                "consecutive readings, and is refused while tracking", "computational_pipeline",
                {**codes, "dispositions": study["dispositions"]},
                {"derivation": "FusionSession.fuse, reacquire and the admission gate", "checks": [
                    refusal("admit a lost-track candidate", "track_lost", codes["admit_lost"]),
                    refusal("fuse into a lost track", "track_lost_requires_reacquisition", codes["fuse_lost"]),
                    refusal("reacquire from non-consecutive readings", "reacquisition_needs_consecutive_readings",
                            codes["reacquire_gap"]),
                    refusal("reacquire while tracking", "reacquisition_not_needed", codes["reacquire_while_tracking"]),
                    refusal("fuse after reacquisition", "no_refusal", codes["fuse_after_reacquisition"]),
                    refusal("admit after reacquisition", "no_refusal", codes["admit_after_reacquisition"])]},
                tolerance=TOL_EXACT),
        finding("The two-point reacquisition covariance [[R, R/dt], [R/dt, 2R/dt^2 + q dt/3 I]] is exact for the "
                "constant-velocity truth", "numerical",
                {k: two[k] for k in ("runs", "max_abs_z", "z_critical", "nees_mean", "nees_z")} |
                {"session_vs_formula": study["reacquire_covariance_error"]},
                {**generator_basis(seed + 1, runs=two["runs"]), "checks": [
                    check("analytic", "sample covariance of the two-point error against the formula (max |z|)",
                          two["max_abs_z"], two["z_critical"], "le"),
                    check("analytic", "mean NEES against 4 (z)", two["nees_z"], two["z_critical"]),
                    check("invariant", "session reacquisition covariance against the formula",
                          study["reacquire_covariance_error"], 1e-15)]},
                tolerance=TOL_MC),
        finding("Under an unmodelled 0.2 rad/s turn during the gap, the expected NEES of the coasting track exceeds "
                "the 99% chi-square(4) quantile after a finite gap even though its covariance keeps growing; a "
                "radius rule protects against this only if its radius is below the ellipse radius reached by then",
                "numerical",
                as_json({k: turn[k] for k in ("omega_rad_s", "speed_m_s", "threshold_chi2_4_99",
                                             "first_inconsistent_gap_ticks", "radius_at_inconsistency_m",
                                             "gap_ticks_before_track_loss")}),
                {"derivation": "E[NEES] = dim + e^T P^-1 e for a deterministic model error e", "checks": [
                    check("analytic", "a finite gap (ticks) after which the coasting track is inconsistent",
                          turn["first_inconsistent_gap_ticks"] or 10_000, 399, "le"),
                    check("analytic", "declared 1 m radius below the radius reached at inconsistency (margin, m)",
                          (turn["radius_at_inconsistency_m"] or 0.0) - study["radius_m"], 0.0, "ge")]},
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
                      "exactly from a declared radius rule, a lost track is neither updated nor admitted, and "
                      "reacquisition is an explicit, refusable operation with an exact initial covariance.",
        "mathematical_model": "P(n) = F(n dt) P F(n dt)^T + Q(n dt); lost when sqrt(lambda_max(P_pos) chi2_2(0.99)) "
                              "> 1 m. Two-point initialization p = z2, v = (z2 - z1)/dt has error covariance "
                              "[[R, R/dt], [R/dt, 2R/dt^2 + q dt/3 I]]. A coordinated turn at omega adds a deterministic "
                              "error e(t); E[NEES] = 4 + e^T P(t)^-1 e.",
        "input_data": [f"seed {seed}: one synthetic camera run (30 active ticks)",
                       f"seed {seed + 1}: {two['runs']} two-point Monte Carlo draws",
                       f"turn rate {turn['omega_rad_s']} rad/s from the last estimated state"],
        "observation_model": "Camera positions until tick 30, then none; readings after loss are offered to fuse, "
                             "reacquire and admit.",
        "expected_invariant": "Observed lost tick = predicted; refusal codes as declared; two-point covariance "
                              "exact.",
        "experiment": "Run a FusionSession with a 1 m track radius, stop the sensor, predict tick by tick, then "
                      "exercise admit, fuse and reacquire; Monte Carlo the two-point estimator; evaluate the "
                      "expected NEES of the coasting track under an unmodelled turn.",
        "numerical_result": f"lost at tick {study['observed_lost_tick']} (predicted {study['predicted_lost_tick']}); "
                            f"two-point max |z| {two['max_abs_z']:.2f}, mean NEES {two['nees_mean']:.3f}; under the "
                            f"turn the coasting track becomes inconsistent after {turn['first_inconsistent_gap_ticks']} "
                            f"ticks (99% ellipse radius {turn['radius_at_inconsistency_m']:.2f} m), while the 1 m rule "
                            f"had already declared loss after {turn['gap_ticks_before_track_loss']} ticks.",
        "uncertainty": "The lost tick and refusals are exact; the two-point statement carries Monte Carlo error "
                       "(Bonferroni bound); the turn statement is deterministic given the declared turn rate.",
        "failure_modes_checked": ["off-by-one in the loss tick", "fusion into a lost track", "admission of a lost "
                                  "track", "reacquisition from non-consecutive readings", "redundant reacquisition",
                                  "unmodelled manoeuvre during the gap"],
        "unresolved_assumptions": ["Under the correct model a coasting track stays consistent (T069); the radius "
                                   "rule is a policy for model validity and data association, not a statistical "
                                   "necessity.",
                                   "Reacquisition assumes the readings belong to the lost target (no association "
                                   "ambiguity)."],
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
    # The dense normal equations (one LU of the whole information matrix) at the short horizon.
    dense_mean, dense = batch_posterior(F, Q, MU0, P0_BENCH, plan[:10], [r[0] for r in readings[:10]])
    banded_mean, banded_cov = batch_posterior_banded(F, Q, MU0, P0_BENCH, plan[:10], [r[0] for r in readings[:10]])
    dense_vs_banded = max(float(np.max(np.abs(dense_mean - banded_mean)) / np.max(np.abs(dense_mean))),
                          float(np.max(np.abs(dense[40:44, 40:44] - banded_cov)) / np.max(np.abs(banded_cov))))
    worst_mean = max(row["max_relative_mean_difference"] for row in comparisons)
    worst_cov = max(row["max_relative_covariance_difference"] for row in comparisons)
    nees = nees_series(estimates, truth, steps)
    grand_z = float(run_mean_z(nees[..., None] - 4.0)[0][0])
    # Exact rational arithmetic: scalar random walk with twelve rational readings.
    exact_rng = generator(seed + 1)
    rational = [Fraction(int(v), 100) for v in exact_rng.integers(-300, 300, 12)]
    exact = exact_scalar_filter(Fraction(1, 10), Fraction(1, 4), Fraction(1), Fraction(0), rational)
    exact_equal = exact["filter_mean"] == exact["batch_mean"] and exact["filter_variance"] == exact["batch_variance"]
    return {"seed": seed, "runs": runs, "ticks": ticks, "comparisons": comparisons, "dense_vs_banded": dense_vs_banded,
            "max_mean_difference": worst_mean,
            "max_covariance_difference": worst_cov, "nees": consistency(nees, 4), "nees_grand_z": grand_z,
            "anees": nees.mean(axis=0),
            "exact": {"readings": [str(v) for v in rational], "filter_mean": str(exact["filter_mean"]),
                      "batch_mean": str(exact["batch_mean"]), "filter_variance": str(exact["filter_variance"]),
                      "batch_variance": str(exact["batch_variance"]), "identical": exact_equal}}


@task("T074", changed_files=files("sensor_fusion_admission"), regression_tests=(
    f"{TESTS}::test_fused_state_equals_batch_posterior",
    f"{TESTS}::test_section_reports_labels_and_states"))
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
        finding("The recursive Kalman estimate and covariance equal the exact batch information-form posterior "
                "marginal at K = 10, 40 and 100 ticks to near roundoff", "numerical",
                {"comparisons": study["comparisons"], "dense_vs_banded_at_K10": study["dense_vs_banded"]},
                {**generator_basis(seed), "checks": [
                    check("analytic", "max relative mean difference against the batch posterior",
                          study["max_mean_difference"], 1e-9),
                    check("analytic", "max relative covariance difference against the batch posterior",
                          study["max_covariance_difference"], 1e-9),
                    check("invariant", "dense normal equations against block elimination at K = 10",
                          study["dense_vs_banded"], 1e-9)]},
                tolerance={"abs": 1e-9, "rel": 0.0}),
        finding("In exact rational arithmetic the scalar filter's final mean and variance are identical to the batch "
                "posterior", "mathematical", ex,
                {"derivation": "tridiagonal normal equations solved exactly in Fractions", "checks": [
                    check("exact_arithmetic", "filter minus batch (mean and variance, exact)",
                          float(not ex["identical"]), 0.0)]},
                tolerance=TOL_EXACT),
        finding("The fused estimate's error against the simulated truth is NEES-consistent at every tick",
                "numerical", {"nees": study["nees"], "grand_z": study["nees_grand_z"]},
                {**generator_basis(seed, runs=study["runs"], ticks=study["ticks"]), "checks": [
                    check("analytic", "fraction of ticks with ANEES in the 99% interval",
                          study["nees"]["fraction_inside"], 0.9, "ge"),
                    check("analytic", "grand mean NEES against 4 (run-level z)", study["nees_grand_z"],
                          bonferroni(2))]},
                tolerance=TOL_MC),
        unreal("The fused estimate equals the physical state of a real target within its covariance", "physical",
               seed, "not established: agreement is with the declared model's exact posterior and simulated truth"),
    ]
    fields = {
        "hypothesis": "For the linear-Gaussian bench the Kalman recursion is the exact posterior: its estimate and "
                      "covariance equal the batch posterior computed without recursion, and its errors against the "
                      "simulated truth are NEES-consistent.",
        "mathematical_model": "Batch posterior over x_0..x_K minimizes |x_0 - mu0|^2_P0 + sum |x_k - F x_{k-1}|^2_Q + "
                              "sum |z_k - H x_k|^2_R (normal equations, information matrix of size 4(K+1)); its "
                              "marginal at K is the filtering posterior. Camera every tick, tracker every 5 ticks "
                              "(stacked, R = blockdiag(R_camera, 0.0025 I)).",
        "input_data": [f"seed {seed} (PCG64)", f"{study['runs']} runs x {study['ticks']} ticks",
                       "3 runs compared with the batch posterior at K = 10, 40, 100",
                       f"12 rational readings (seed {seed + 1}) for the exact scalar check"],
        "observation_model": "Linear position readings with the declared covariances; the filter model is the truth "
                             "model.",
        "expected_invariant": "Kalman = batch to roundoff; rational filter = rational batch exactly; ANEES inside "
                              "chi2(4N)/N.",
        "experiment": "Solve the normal equations for each prefix and compare the last block with the filter; run "
                      "the exact Fraction filter and elimination; Monte Carlo NEES against the simulated truth.",
        "numerical_result": f"max relative mean difference {study['max_mean_difference']:.1e}, covariance "
                            f"{study['max_covariance_difference']:.1e}; exact rational identity {ex['identical']}; "
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
    """One scenario per admission check: a candidate that violates only that check.

    Each returns (session, candidate, declared). Private access to session
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
        session, candidate = tracked()
        session.read_only = True
        return session, candidate, dict(ADMIT)

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

    def fresh():
        session, candidate = tracked()
        session.predict(session.tick + 1)
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

    return {"declared": declared_missing, "writable": writable, "typed": typed, "finite": finite,
            "covariance": covariance, "integrity": integrity, "provenance": provenance, "frame": frame,
            "fresh": fresh, "track": track, "uncertainty": uncertainty, "innovation": innovation,
            "calibration": calibration}, tracked


def admission_study() -> dict:
    scenarios, tracked = _admission_scenarios()
    codes = {name: code for name, code, _ in ADMISSION_CHECKS}
    rows = {}
    for name, build in scenarios.items():
        session, candidate, declared = build()
        full = refusal_code(lambda: session.admit(candidate, **declared))
        reduced = tuple(c for c in ADMISSION_CHECKS if c[0] != name)
        mutant_code, _ = admission_verdict(session, candidate, declared, reduced)
        rows[name] = {"expected": codes[name], "full_gate": full,
                      "without_this_check": mutant_code or "admitted", "killed": (mutant_code or "admitted") != full}
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
        mutations["mutate_candidate"] = "no_refusal"
    except FrozenInstanceError:
        mutations["mutate_candidate"] = "frozen_instance"
    try:
        Observation("camera", "world", 1, (0.0, 0.0), R_CAMERA, "cam-cal").value = (1.0, 1.0)
        mutations["mutate_observation"] = "no_refusal"
    except FrozenInstanceError:
        mutations["mutate_observation"] = "frozen_instance"
    return {"checks": [name for name, _, _ in ADMISSION_CHECKS], "rows": rows,
            "killed": sum(r["killed"] for r in rows.values()),
            "sole_guard": sum(r["without_this_check"] == "admitted" for r in rows.values()),
            "refused_as_expected": sum(r["full_gate"] == r["expected"] for r in rows.values()),
            "auto_admitted_before_gate": auto, "admitted_after_gate": len(session.admitted),
            "admitted_checks": list(admitted.checks), "admitted_authority": dict(admitted.authority),
            "admitted_digest_matches_candidate": admitted.candidate_digest == candidate.digest,
            "type_relations_true": sum(relations), "mutations": mutations}


@task("T075", changed_files=files("sensor_fusion_admission", "sensor_fusion_objects"), regression_tests=(
    f"{TESTS}::test_typed_objects_and_admission_mutations",
    f"{TESTS}::test_section_reports_labels_and_states"))
def typed_admission(ctx):
    study = admission_study()
    ctx.artifact_json("admission_mutations.json", as_json(study))
    rows, total = study["rows"], len(study["rows"])
    mutations = study["mutations"]
    findings = [
        finding("Every admission check refuses the candidate built to violate it, with that check's own refusal "
                "code", "computational_pipeline",
                {name: {k: row[k] for k in ("expected", "full_gate")} for name, row in rows.items()},
                {"derivation": "ADMISSION_CHECKS in ciw.lab.sensor_fusion_objects", "checks": [
                    refusal(f"admission of a candidate violating '{name}'", row["expected"], row["full_gate"])
                    for name, row in rows.items()]},
                tolerance=TOL_EXACT),
        finding("Mutation analysis: deleting any single admission check changes the outcome of its scenario, and "
                "for all but the declaration check the violating candidate is then admitted",
                "computational_pipeline",
                {"checks": total, "mutants_killed": study["killed"], "sole_guard_checks": study["sole_guard"],
                 "without_each_check": {name: row["without_this_check"] for name, row in rows.items()}},
                {"derivation": "admission_verdict with one check removed", "checks": [
                    check("exact_arithmetic", "surviving mutants", total - study["killed"], 0),
                    check("exact_arithmetic", "checks that are not the sole guard minus one",
                          total - study["sole_guard"] - 1, 0)]},
                tolerance=TOL_EXACT),
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
                    refusal("mutate a candidate", "frozen_instance", mutations["mutate_candidate"]),
                    refusal("mutate an observation", "frozen_instance", mutations["mutate_observation"])]},
                tolerance=TOL_EXACT),
        unreal("An admitted synthetic state may command actuators", "actuator_authority", 75_2026,
               "not established: admission is a software gate over synthetic data; authority is decided elsewhere"),
    ]
    fields = {
        "hypothesis": "Keeping observations, candidate states and admitted states as separate types, with a single "
                      "explicit gate whose every check is load-bearing, prevents a candidate from becoming state "
                      "by accident, tampering or omission.",
        "mathematical_model": "Admission = ordered conjunction of declared checks (declared, writable, typed, "
                              "finite, covariance, integrity, provenance, frame, fresh, track, uncertainty, "
                              "innovation, calibration); fail closed on any exception. Mutation m_i removes check i.",
        "input_data": ["one synthetic camera run (seed 752026) driving a FusionSession for each scenario",
                       f"{total} violation scenarios plus one valid control"],
        "observation_model": "Typed Observation objects; candidates issued by the session and sealed by a content "
                             "digest.",
        "expected_invariant": "Full gate refuses scenario i with code_i; gate without check i gives a different "
                              "outcome; no admission without an explicit gate call; AdmittedState immutable.",
        "experiment": "Build each violating candidate (private state corruption simulates internal faults), run "
                      "the full gate and the gate with that check deleted, and exercise construction and mutation "
                      "of each type.",
        "numerical_result": f"{study['refused_as_expected']}/{total} refused with the expected code; "
                            f"{study['killed']}/{total} mutants killed; {study['sole_guard']} checks are the sole "
                            f"guard for their scenario (without the declaration check, the missing value still makes a later "
                            f"check fail closed).",
        "uncertainty": "None: the outcomes are deterministic.",
        "failure_modes_checked": ["subclass imposter", "tampered mean with a stale digest", "forged digest",
                                  "NaN state", "indefinite covariance", "stale candidate", "lost track",
                                  "excess uncertainty", "inconsistent innovation", "revoked calibration",
                                  "read-only session", "missing declaration", "frame mismatch"],
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
    enabled = FusionSession(read_only=False)
    return {"read_only_default": signature.parameters["read_only"].default,
            "session_read_only": session.read_only, "session_authority": dict(session.authority),
            "ciw_authority": dict(AUTHORITY), "matches_ciw": session.authority == AUTHORITY,
            "default_matches_module_constant": DEFAULT_AUTHORITY == AUTHORITY,
            "codes": codes, "log": [entry["disposition"] for entry in session.log],
            "record_disposition": record_entry["disposition"], "state": session.x is None and session.P is None,
            "admitted": len(session.admitted), "enabled_authority": dict(enabled.authority),
            "enabled_matches_synthetic": enabled.authority == SYNTHETIC_AUTHORITY}


@task("T076", changed_files=files("sensor_fusion_admission", "sensor_fusion_objects"), regression_tests=(
    f"{TESTS}::test_defaults_are_read_only_and_not_performed",
    f"{TESTS}::test_section_reports_labels_and_states"))
def read_only_defaults(ctx):
    study = defaults_study()
    ctx.artifact_json("defaults.json", as_json(study))
    codes = study["codes"]
    authority = study["ciw_authority"]
    findings = [
        finding("The CIW authority vocabulary declares state_admission and sensor_fusion not_performed and "
                "physical_truth not_established, and the fusion session's default authority equals it",
                "computational_pipeline",
                {"ciw_authority": authority, "session_authority": study["session_authority"],
                 "matches": study["matches_ciw"]},
                {"derivation": "ciw.declared_workload.AUTHORITY and FusionSession()", "checks": [
                    refusal("CIW AUTHORITY sensor_fusion", "not_performed", authority.get("sensor_fusion")),
                    refusal("CIW AUTHORITY state_admission", "not_performed", authority.get("state_admission")),
                    refusal("CIW AUTHORITY physical_truth", "not_established", authority.get("physical_truth")),
                    check("invariant", "default session authority differs from CIW AUTHORITY",
                          float(not study["matches_ciw"]), 0.0),
                    check("invariant", "module DEFAULT_AUTHORITY differs from CIW AUTHORITY",
                          float(not study["default_matches_module_constant"]), 0.0)]},
                tolerance=TOL_EXACT),
        finding("FusionSession defaults to read-only: every state-changing call is refused with read_only_session, "
                "observations are still retained, and no state or admission exists", "computational_pipeline",
                {"read_only_default": study["read_only_default"], "codes": codes, "log": study["log"],
                 "record_disposition": study["record_disposition"], "no_state": study["state"],
                 "admitted": study["admitted"]},
                {"derivation": "FusionSession._writable and the admission gate", "checks": [
                    check("invariant", "read_only default is not True", float(study["read_only_default"] is not True),
                          0.0)] + [refusal(f"{name} on a default session", "read_only_session", code)
                                   for name, code in codes.items()] + [
                    check("exact_arithmetic", "admitted states on a default session", study["admitted"], 0),
                    check("invariant", "fused observation not retained with its refusal",
                          float(study["log"][0] != "refused:read_only_session"), 0.0)]},
                tolerance=TOL_EXACT),
        finding("Enabling fusion explicitly changes the authority only to synthetic_only; physical truth stays "
                "not_established", "computational_pipeline", study["enabled_authority"],
                {"derivation": "SYNTHETIC_AUTHORITY in ciw.lab.sensor_fusion_objects", "checks": [
                    refusal("enabled session sensor_fusion", "synthetic_only",
                            study["enabled_authority"].get("sensor_fusion")),
                    refusal("enabled session physical_truth", "not_established",
                            study["enabled_authority"].get("physical_truth"))]},
                tolerance=TOL_EXACT),
        unreal("Synthetic fusion output is admissible as production state", "production_acceptance", 76_2026,
               "not established: sensor_fusion and state_admission are not_performed by default and synthetic_only "
               "when enabled"),
    ]
    fields = {
        "hypothesis": "The fusion API cannot fuse or admit anything unless a caller explicitly asks for a writable "
                      "session, and its authority record uses the CIW vocabulary (sensor_fusion and "
                      "state_admission not_performed).",
        "mathematical_model": "Not numerical: a default-argument and refusal audit of FusionSession against "
                              "ciw.declared_workload.AUTHORITY.",
        "input_data": ["FusionSession() with default arguments", "ciw.declared_workload.AUTHORITY"],
        "observation_model": "One camera observation offered to fuse and record.",
        "expected_invariant": "read_only default True; authority == CIW AUTHORITY; initialize, predict, handle_gap, "
                              "fuse, reacquire and admit refused with read_only_session; record allowed.",
        "experiment": "Inspect the constructor signature, compare authority dictionaries, call every state-changing "
                      "method on a default session, then construct a session with read_only=False.",
        "numerical_result": f"read_only default {study['read_only_default']}; authority {study['session_authority']}; "
                            f"refusals {sorted(set(codes.values()))}; enabled authority {study['enabled_authority']}.",
        "uncertainty": "None: the outcomes are deterministic.",
        "failure_modes_checked": ["writable default", "authority drift from the CIW vocabulary", "silent fusion in "
                                  "a read-only session", "loss of a refused observation", "authority upgrade on "
                                  "enabling fusion"],
        "unresolved_assumptions": ["The CIW vocabulary is read from ciw.declared_workload; if it changes, this "
                                   "task's refusal checks fail rather than silently following it."],
        "recommended_next_task": "T113: connect filtered residuals to the Lyapunov runtime without putting sensors "
                                 "inside the kernel.",
    }
    return outcome(fields, findings)
