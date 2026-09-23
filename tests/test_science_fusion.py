"""Staged sensor fusion: consistency, time/frame/unit handling, refusals, admission and provenance."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from ciw.science._common import Refusal, content_identity
from ciw.science.fusion import (CHI2_QUANTILES, RAW_SCHEMA, STAGES, AdmissionPolicy, ConstantVelocityModel,
                                FusionEngine, RawObservation, admit, chi2_interval, chi2_quantile_wh, normal_quantile,
                                replay, retain, scenario_engine, synthetic_scenario, verify_record)
from ciw.science.frames import FrameRegistry
from ciw.science.ledger import Ledger

EXAMPLES = Path(__file__).parents[1] / "examples" / "science" / "fusion"
ROTATION = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
TRANSLATION = [1.0, 2.0, 0.5]
OFFSET = 2.0
V = np.array([1.0, -0.5, 0.2])
P0 = np.array([0.3, -0.2, 1.1])
SIGMA = 1e-3
COV = (np.eye(3) * SIGMA ** 2).tolist()


def registry() -> FrameRegistry:
    result = FrameRegistry("m")
    for frame, kind in (("world", "world"), ("camera", "camera"), ("lidar", "sensor")):
        result.add_frame(frame, kind)
    for clock in ("clock_a", "orphan"):
        result.add_clock(clock, "device", 1e-6)
    result.add_clock("world_clock", "ptp", 1e-6)
    result.add_clock_mapping({"mapping_id": "a-to-world", "source": "clock_a", "target": "world_clock",
                              "offset_s": OFFSET, "rate": 1.0, "covariance": [[1e-10, 0.0], [0.0, 1e-16]],
                              "valid_from": -100.0, "valid_until": 100.0,
                              "calibration": {"calibration_id": "sync-a", "version": "1"}})
    result.add_transform({"transform_id": "camera-to-world", "source": "camera", "target": "world",
                          "rotation": {"matrix": ROTATION}, "translation": TRANSLATION, "unit": "m",
                          "covariance": np.diag([1e-8] * 3 + [4e-8] * 3).tolist(), "clock": "world_clock",
                          "estimated_at": 0.0, "valid_from": -10.0, "valid_until": 10.0,
                          "calibration": {"calibration_id": "extrinsic-camera", "version": "3"}})
    return result


def engine(frames: FrameRegistry | None = None, *, unit: str = "m", q: float = 1e-4, **options) -> FusionEngine:
    options.setdefault("initial_velocity_std", 2.0)
    return FusionEngine(ConstantVelocityModel(3, q, "m^2/s^3"), frames, filter_frame="world",
                        filter_clock="world_clock", unit=unit, **options)


def raw(sequence: int, time_s: float, value, *, stream: str = "s", clock: str | None = "world_clock",
        frame: str = "world", unit: str = "m", cov=COV, observable: str = "tracker_position",
        acquisition: str = "synthetic") -> dict:
    return {"schema": RAW_SCHEMA, "stream_id": stream, "sequence": sequence, "clock": clock,
            "device_time_s": time_s, "value": list(value), "unit": unit, "frame": frame,
            "covariance": cov, "observable": observable, "acquisition": acquisition}


def truth(t: float) -> np.ndarray:
    return P0 + V * t


def linear_stream(count: int = 40, seed: int = 0, dt: float = 0.05, sigma: float = SIGMA):
    rng = np.random.default_rng(seed)
    return [(k, k * dt, truth(k * dt) + rng.normal(0.0, sigma, 3)) for k in range(count)]


def nees(record: dict, reference: np.ndarray) -> float:
    error = np.array(record["state"]["position"]) - reference
    covariance = np.array(record["covariance"])[:3, :3]
    return float(error @ np.linalg.solve(covariance, error))


def lenient(**changes) -> AdmissionPolicy:
    values = dict(min_accepted_updates=10, nis_window=20, max_position_std=0.05, std_unit="m",
                  max_refused_fraction=0.2, refusal_window=50, max_state_age_s=0.5,
                  allowed_acquisition=("synthetic",))
    values.update(changes)
    return AdmissionPolicy(**values)


def serializable(record: dict) -> None:
    json.dumps(record, allow_nan=False)
    verify_record(record)


# ---------------------------------------------------------------------- statistics
def _chi2_cdf(x: float, k: int) -> float:
    h = x / 2.0
    if k % 2 == 0:
        return 1.0 - sum(math.exp(j * math.log(h) - math.lgamma(j + 1) - h) for j in range(k // 2))
    tail = sum(h ** (0.5 + j) * math.exp(-h) / math.gamma(1.5 + j) for j in range((k - 1) // 2))
    return math.erf(math.sqrt(h)) - tail


def test_tabulated_chi_square_quantiles_are_exact_and_wilson_hilferty_is_close():
    for probability, row in CHI2_QUANTILES.items():
        assert len(row) == 6
        for dof, quantile in enumerate(row, start=1):
            assert abs(_chi2_cdf(quantile, dof) - probability) < 1e-9
            if dof >= 3:  # Wilson-Hilferty is an approximation: within 2 % here, better for large dof
                assert abs(chi2_quantile_wh(probability, dof) / quantile - 1.0) < 0.02
    assert normal_quantile(0.975) == pytest.approx(1.959963984540054, abs=1e-12)
    low, high = chi2_interval(300, 0.95)
    assert _chi2_cdf(low, 300) == pytest.approx(0.025, abs=1e-3)
    assert _chi2_cdf(high, 300) == pytest.approx(0.975, abs=1e-3)


def test_discrete_process_noise_is_the_exact_white_acceleration_integral():
    model = ConstantVelocityModel(2, 0.7, "m^2/s^3")
    dt, q = 0.37, model.psd_in("m")
    F, Q = model.discrete(dt, q)
    s = np.linspace(0.0, dt, 4001)
    G = np.vstack([np.zeros((2, 2)), np.eye(2)])
    integrand = np.array([(Phi := model.discrete(item, q)[0]) @ G @ G.T @ Phi.T * q for item in s])
    assert np.allclose(np.trapezoid(integrand, s, axis=0), Q, rtol=1e-6, atol=1e-12)
    assert np.allclose(F, [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
    assert ConstantVelocityModel(3, 1.0, "m^2/s^3").psd_in("mm") == pytest.approx(1e6)


# ---------------------------------------------------------------------- consistency
def test_monte_carlo_nees_is_within_chi_square_bounds():
    runs, per_time, nis = 25, {}, []
    for seed in range(runs):
        scenario = synthetic_scenario(1000 + seed, duration_s=3.0, dropout=0.0, outliers=0, burst=False)
        fused = scenario_engine(scenario)
        reference = {(item["stream_id"], item["sequence"]): item for item in scenario["truth"]}
        for record in replay(fused, scenario):
            if record["stage"] != "filtered_candidate":
                assert record["refusal"]["code"] == "outlier_gated"  # the 0.1 % false-alarm tail
                continue
            item = reference[(record["stream_id"], record["sequence"])]
            error = np.array(record["state"]["mean"]) - np.array(item["position"] + item["velocity"])
            per_time.setdefault(record["time"]["value"], []).append(
                float(error @ np.linalg.solve(np.array(record["state"]["covariance"]), error)))
            nis.append(record["update"]["nis"])
    complete = {time: np.mean(values) for time, values in per_time.items() if len(values) == runs}
    assert len(complete) > 90
    low, high = chi2_interval(runs * 6, 0.95)
    inside = np.mean([low / runs <= value <= high / runs for value in complete.values()])
    assert inside >= 0.85
    final_low, final_high = chi2_interval(runs * 6, 0.99)
    assert final_low / runs <= complete[max(complete)] <= final_high / runs
    nis_low, nis_high = chi2_interval(3 * len(nis), 0.99)
    assert nis_low <= sum(nis) <= nis_high


def test_asynchronous_streams_are_filtered_in_aligned_time_order():
    scenario = synthetic_scenario(5, duration_s=2.0, dropout=0.0, outliers=0, burst=False)
    raws = [item["observation"] for item in scenario["arrivals"]]
    grouped = [item for item in raws if item["stream_id"] == "tracker_b"] + \
              [item for item in raws if item["stream_id"] == "tracker_a"]
    batch = scenario_engine(scenario, max_age_s=10.0)
    results = batch.ingest_batch(grouped, evaluated_at=2.5)
    assert all(item["stage"] == "filtered_candidate" for item in results)
    times = [item["time"]["value"] for item in batch.history if item["stage"] == "filtered_candidate"
             and item["update"]["kind"] == "measurement"]
    assert times == sorted(times) and len(times) == len(raws)
    # device times of the two clocks differ by ~15.7 s, so device order is not acquisition order
    by_device = sorted(raws, key=lambda item: item["device_time_s"])
    assert [item["stream_id"] for item in by_device[:5]] == ["tracker_a"] * 5
    arrival_order = scenario_engine(scenario, max_age_s=10.0)
    codes = {arrival_order.ingest(item, evaluated_at=2.5).get("refusal", {}).get("code") for item in grouped}
    assert "out_of_sequence" in codes


def test_declared_clock_offset_is_applied_and_ignoring_it_biases_the_state():
    samples = linear_stream(60)
    mapped, ignored = engine(registry()), engine(registry())
    for k, t, z in samples:
        assert mapped.ingest(raw(k, t - OFFSET, z, clock="clock_a"))["stage"] == "filtered_candidate"
        assert ignored.ingest(raw(k, t - OFFSET, z, clock="world_clock"))["stage"] == "filtered_candidate"
    good = mapped.candidate()
    assert good["time"]["value"] == pytest.approx(samples[-1][1])
    assert nees(good, truth(good["time"]["value"])) < CHI2_QUANTILES[0.999][2]
    assert good["provenance"]["clock_mappings"] == ["a-to-world"]
    bad = ignored.candidate()
    label = bad["time"]["value"]
    assert label == pytest.approx(samples[-1][1] - OFFSET)
    bias = np.array(bad["state"]["position"]) - truth(label)
    assert np.allclose(bias, V * OFFSET, atol=0.01)
    assert nees(bad, truth(label)) > 1e3


def test_timing_uncertainty_inflates_measurement_covariance_with_velocity_estimate():
    frames = registry()
    fused = engine(frames)
    for k, t, z in linear_stream(20):
        fused.ingest(raw(k, t - OFFSET, z, clock="clock_a"))
    velocity = np.array(fused.candidate()["state"]["velocity"])
    record = fused.ingest(raw(20, 1.0 - OFFSET, truth(1.0), clock="clock_a"))
    transformed = fused.records[record["parents"][0]]
    variance = frames.align_time(1.0 - OFFSET, "clock_a", "world_clock")[1]
    assert transformed["time"]["variance_s2"] == pytest.approx(variance)
    assert transformed["timing_velocity"]["source"] == "state_estimate"
    assert np.allclose(transformed["timing_velocity"]["value"], velocity)
    assert np.allclose(transformed["timing_covariance"], np.outer(velocity, velocity) * variance)
    assert np.allclose(transformed["effective_covariance"],
                       np.array(transformed["covariance"]) + np.array(transformed["timing_covariance"]))


# ---------------------------------------------------------------------- frames, clocks and units
def test_camera_frame_observation_is_transformed_into_world():
    rotation, translation = np.array(ROTATION), np.array(TRANSLATION)
    fused = engine(registry())
    for k, t, z in linear_stream(30):
        record = fused.ingest(raw(k, t, rotation.T @ (z - translation), frame="camera"))
        assert record["stage"] == "filtered_candidate"
    transformed = fused.records[record["parents"][0]]
    source = fused.records[transformed["parents"][0]]
    assert np.allclose(transformed["value"], rotation @ np.array(source["value"]) + translation)
    extra = np.array(transformed["covariance"]) - rotation @ np.array(COV) @ rotation.T
    assert np.linalg.eigvalsh(extra).min() > 0  # transform uncertainty was added
    assert transformed["source_frame"] == "camera" and transformed["frame"] == "world"
    assert [link["transform_id"] for link in transformed["transform_chain"]] == ["camera-to-world"]
    assert transformed["validity"] == {"checked_at": {"time": source["device_time_s"], "clock": "world_clock"},
                                       "valid": True, "clock_mappings": [], "transforms": ["camera-to-world"]}
    final = fused.candidate()
    assert nees(final, truth(final["time"]["value"])) < CHI2_QUANTILES[0.999][2]
    assert final["provenance"]["transforms"] == ["camera-to-world"]


@pytest.mark.parametrize("frame, time_s, code", [
    ("lidar", 0.1, "frame_unreachable"),      # declared frame, no transform relates it to the world
    ("mystery", 0.1, "unknown_frame"),        # undeclared frame
    ("camera", 20.0, "transform_stale"),      # transform outside its validity interval
])
def test_frame_confusion_is_refused_not_passed(frame, time_s, code):
    fused = engine(registry(), max_age_s=100.0)
    fused.ingest(raw(0, 0.0, truth(0.0)))
    record = fused.ingest(raw(1, time_s, truth(time_s), frame=frame))
    assert record["stage"] == "refused" and record["refused_at"] == "transformed_observation"
    assert record["refusal"]["code"] == code
    assert fused.time == 0.0


def test_frame_mismatch_without_registry_is_refused():
    record = engine(None).ingest(raw(0, 0.0, truth(0.0), frame="camera"))
    assert record["refusal"]["code"] == "frame_unreachable"


@pytest.mark.parametrize("clock, device_time, code, refused_at", [
    (None, 0.0, "clock_unspecified", "raw_observation"),
    ("orphan", 0.0, "clock_unmapped", "transformed_observation"),
    ("clock_a", 150.0, "clock_mapping_stale", "transformed_observation"),
    ("undeclared", 0.0, "unknown_clock", "transformed_observation"),
])
def test_clock_problems_are_refused(clock, device_time, code, refused_at):
    record = engine(registry()).ingest(raw(0, device_time, truth(0.0), clock=clock))
    assert (record["stage"], record["refused_at"], record["refusal"]["code"]) == ("refused", refused_at, code)


def test_missing_clock_is_refused_on_construction_and_without_registry():
    item = raw(0, 0.0, truth(0.0))
    del item["clock"]
    with pytest.raises(Refusal) as caught:
        RawObservation.from_json(item)
    assert caught.value.code == "clock_unspecified"
    assert engine(None).ingest(raw(0, 0.0, truth(0.0), clock="clock_a"))["refusal"]["code"] == "clock_unmapped"


def test_millimetre_observations_give_the_same_estimate_as_metres():
    in_m, in_mm = engine(None), engine(None)
    mm_filter = engine(None, unit="mm", initial_velocity_std={"value": 2.0, "unit": "m/s"})
    assert mm_filter.initial_velocity_std == pytest.approx(2000.0)
    mm_cov = (np.array(COV) * 1e6).tolist()
    for k, t, z in linear_stream(30):
        in_m.ingest(raw(k, t, z))
        in_mm.ingest(raw(k, t, z * 1000.0, unit="mm", cov=mm_cov))
        mm_filter.ingest(raw(k, t, z))
    a, b, c = in_m.candidate(1.6), in_mm.candidate(1.6), mm_filter.candidate(1.6)
    assert np.allclose(a["state"]["mean"], b["state"]["mean"], rtol=1e-9, atol=1e-12)
    assert np.allclose(a["covariance"], b["covariance"], rtol=1e-9, atol=1e-15)
    assert c["unit"] == "mm" and c["velocity_unit"] == "mm/s"
    assert np.allclose(np.array(c["state"]["mean"]) / 1000.0, a["state"]["mean"], rtol=1e-9, atol=1e-12)
    transformed = next(item for item in in_mm.history if item["stage"] == "transformed_observation")
    assert transformed["source_unit"] == "mm" and transformed["unit"] == "m" and transformed["unit_factor"] == 1e-3
    refused = in_m.ingest(raw(99, 1.6, [1.0, 1.0, 1.0], unit="s"))
    assert refused["refusal"]["code"] == "dimension_mismatch"


# ---------------------------------------------------------------------- filter policies
def test_stale_and_future_observations_are_refused():
    fused = engine(None, max_age_s=1.0)
    assert fused.ingest(raw(0, 0.0, truth(0.0)), evaluated_at=0.1)["stage"] == "filtered_candidate"
    stale = fused.ingest(raw(1, 1.0, truth(1.0)), evaluated_at=5.0)
    assert stale["refusal"]["code"] == "stale_observation" and stale["refused_at"] == "filtered_candidate"
    assert stale["refusal"]["detail"]["age_s"] == pytest.approx(4.0)
    future = fused.ingest(raw(2, 9.0, truth(9.0)), evaluated_at=5.0)
    assert future["refusal"]["code"] == "future_observation"
    assert fused.time == 0.0


def test_out_of_sequence_observation_is_refused_without_retrodiction():
    fused = engine(None)
    fused.ingest(raw(0, 1.0, truth(1.0)))
    before = fused.candidate()
    record = fused.ingest(raw(1, 0.5, truth(0.5)))
    assert record["refusal"]["code"] == "out_of_sequence"
    assert record["refusal"]["detail"]["lag_s"] == pytest.approx(0.5)
    assert fused.candidate()["state"] == before["state"] and fused.time == 1.0
    with pytest.raises(Refusal) as caught:
        fused.candidate(0.5)
    assert caught.value.code == "retrodiction_unsupported"


def test_outlier_is_gated_by_nis_and_retained_with_its_nis():
    fused = engine(None)
    for k, t, z in linear_stream(20):
        fused.ingest(raw(k, t, z))
    state = fused.history[-1]["identity"]
    record = fused.ingest(raw(20, 1.0, truth(1.0) + np.array([0.5, 0.0, 0.0])))
    assert record["stage"] == "refused" and record["refusal"]["code"] == "outlier_gated"
    assert record["nis"] > CHI2_QUANTILES[0.999][2]
    assert record["refusal"]["detail"]["threshold"] == CHI2_QUANTILES[0.999][2]
    assert record["refusal"]["detail"]["dof"] == 3
    assert record["parents"][1] == state and fused.records[record["parents"][0]]["stage"] == "transformed_observation"
    assert record in fused.history
    assert fused.ingest(raw(21, 1.05, truth(1.05)))["stage"] == "filtered_candidate"
    monitor = fused.candidate()["residual_monitor"]
    assert monitor["refused_by_code"] == {"outlier_gated": 1}
    assert {"outcome": "refused", "code": "outlier_gated", "stream_id": "s"} in monitor["recent_inputs"]


def test_dropped_samples_are_counted_from_sequence_gaps():
    fused = engine(None)
    for k in (0, 1, 2, 5, 6, 9):
        fused.ingest(raw(k, 0.05 * k, truth(0.05 * k)))
    fused.ingest(raw(0, 0.5, truth(0.5), stream="other"))
    assert fused.dropped()["s"] == {"received": 6, "first": 0, "last": 9, "dropped": 4, "ranges": [[3, 4], [7, 8]]}
    late = fused.ingest(raw(3, 0.15, truth(0.15)))
    assert late["refusal"]["code"] == "out_of_sequence"
    assert fused.dropped()["s"]["ranges"] == [[4, 4], [7, 8]] and fused.dropped()["s"]["dropped"] == 3
    assert fused.candidate()["residual_monitor"]["dropped"]["other"]["dropped"] == 0


def test_duplicate_observation_is_refused():
    fused = engine(None)
    first = raw(0, 0.0, truth(0.0))
    assert fused.ingest(first)["stage"] == "filtered_candidate"
    assert fused.ingest(first)["refusal"]["code"] == "duplicate_observation"
    changed = fused.ingest(raw(0, 0.05, truth(0.05)))
    assert changed["refusal"]["code"] == "duplicate_observation" and changed["refused_at"] == "transformed_observation"
    batch = fused.ingest_batch([raw(1, 0.1, truth(0.1)), raw(1, 0.1, truth(0.1))])
    assert [item["stage"] for item in batch] == ["filtered_candidate", "refused"]


@pytest.mark.parametrize("cov, cause", [
    ([[1e-6, 2e-7, 0.0], [0.0, 1e-6, 0.0], [0.0, 0.0, 1e-6]], "asymmetric"),
    ([[1e-6, 0.0, 0.0], [0.0, -1e-6, 0.0], [0.0, 0.0, 1e-6]], "not positive semidefinite"),
    ([[float("nan"), 0.0, 0.0], [0.0, 1e-6, 0.0], [0.0, 0.0, 1e-6]], "nan"),
    ([[1e-6, 0.0], [0.0, 1e-6]], "shape"),
])
def test_malformed_covariance_is_refused(cov, cause):
    fused = engine(None)
    record = fused.ingest(raw(0, 0.0, truth(0.0), cov=cov))
    assert record["stage"] == "refused" and record["refused_at"] == "raw_observation"
    assert record["refusal"]["code"] == "malformed_covariance", cause
    assert record["parents"] == [] and fused.time is None
    serializable(record)
    assert ("input_excerpt" in record) == (cause == "nan")


def test_observable_outside_the_measurement_model_is_refused():
    fused = engine(None)
    assert fused.ingest(raw(0, 0.0, truth(0.0), observable="camera_chord"))["refusal"]["code"] == "observable_mismatch"
    assert fused.ingest(raw(1, 0.0, truth(0.0), observable="banana"))["refusal"]["code"] == "unknown_observable"
    with pytest.raises(Refusal) as caught:
        ConstantVelocityModel(3, 1.0, "m^2/s^3", ("imu_orientation",))
    assert caught.value.code == "unsupported_observable"
    with pytest.raises(Refusal) as caught:
        FusionEngine(ConstantVelocityModel(), None, filter_frame="world", filter_clock="c", unit="m")
    assert caught.value.code == "prior_unspecified"
    with pytest.raises(Refusal) as caught:
        engine(None, gate_probability=0.95)
    assert caught.value.code == "unsupported_gate"


# ---------------------------------------------------------------------- admission
@pytest.fixture(scope="module")
def replayed():
    scenario = synthetic_scenario(11, duration_s=4.0)
    fused = scenario_engine(scenario)
    records = replay(fused, scenario)
    return scenario, fused, records


def test_synthetic_scenario_is_labelled_and_replays_with_injected_faults_detected(replayed):
    scenario, fused, records = replayed
    assert scenario["acquisition"] == "synthetic"
    assert all(item["observation"]["acquisition"] == "synthetic" for item in scenario["arrivals"])
    assert content_identity(synthetic_scenario(11, duration_s=4.0)) == content_identity(scenario)
    gated = {(item["stream_id"], item["sequence"]) for item in records if item["stage"] == "refused"}
    assert {tuple(item) for item in scenario["injected"]["outliers"]} <= gated
    assert all(item["refusal"]["code"] == "outlier_gated" for item in records if item["stage"] == "refused")
    for stream, lost in scenario["injected"]["dropped"].items():
        report = fused.dropped()[stream]
        interior = [k for k in lost if report["first"] < k < report["last"]]
        found = [k for low, high in report["ranges"] for k in range(low, high + 1)]
        assert found == interior and report["dropped"] == len(interior)
    assert [20, 22] in fused.dropped()["tracker_b"]["ranges"]
    for record in fused.history:
        serializable(record)


def test_admission_succeeds_under_a_satisfied_policy(replayed):
    scenario, fused, _ = replayed
    candidate = fused.candidate()
    admitted = fused.admit(candidate, lenient(), candidate["time"]["value"] + 0.1)
    assert admitted["stage"] == "admitted_state" and admitted["status"] == "admitted"
    assert admitted["parents"] == [candidate["identity"]]
    assert all(item["passed"] for item in admitted["checks"])
    assert admitted["state"] == candidate["state"] and admitted["acquisition"] == ["synthetic"]
    assert {item["check"] for item in admitted["checks"]} >= {
        "insufficient_updates", "nis_inconsistent", "position_uncertainty_exceeded", "refused_fraction_exceeded",
        "state_too_old", "provenance_unverified", "acquisition_not_admissible"}
    assert candidate["provenance"]["clock_mappings"] == ["clock_a-to-world", "clock_b-to-world"]
    assert candidate["provenance"]["transforms"] == ["camera_b-to-world"]
    serializable(admitted)


def test_admission_refusal_keeps_the_candidate_and_lists_every_reason(replayed):
    _, fused, _ = replayed
    candidate = fused.candidate()
    strict = AdmissionPolicy(min_accepted_updates=10_000, nis_window=20, max_position_std=1e-5, std_unit="mm",
                             max_refused_fraction=0.0, refusal_window=100, max_state_age_s=0.01)
    refused = admit(candidate, strict.to_json(), candidate["time"]["value"] + 5.0)
    assert refused["stage"] == "admission_refused" and refused["status"] == "retained_as_candidate"
    assert refused["candidate"] == candidate["identity"] and refused["parents"] == [candidate["identity"]]
    codes = {item["code"] for item in refused["reasons"]}
    assert codes == {"insufficient_updates", "position_uncertainty_exceeded", "refused_fraction_exceeded",
                     "state_too_old", "acquisition_not_admissible"}
    std = next(item for item in refused["reasons"] if item["code"] == "position_uncertainty_exceeded")
    assert std["axes"] == [0, 1, 2] and std["limit"] == pytest.approx(1e-8) and std["unit"] == "m"
    verify_record(candidate)
    tampered = dict(candidate, state={**candidate["state"], "position": [0.0, 0.0, 0.0]})
    with pytest.raises(Refusal) as caught:
        admit(tampered, strict, 1.0)
    assert caught.value.code == "identity_mismatch"


def test_admission_detects_nis_inconsistency_from_misdeclared_noise():
    fused = engine(None)
    inflated = (np.array(COV) * 100.0).tolist()  # declared noise ten times the true standard deviation
    for k, t, z in linear_stream(40):
        fused.ingest(raw(k, t, z, cov=inflated))
    candidate = fused.candidate()
    refused = fused.admit(candidate, lenient(max_position_std=1.0), candidate["time"]["value"])
    nis = next(item for item in refused["reasons"] if item["code"] == "nis_inconsistent")
    assert nis["average_nis"] < nis["bounds"][0] and nis["approximation"] == "wilson_hilferty"
    assert [item["code"] for item in refused["reasons"]] == ["nis_inconsistent"]


# ---------------------------------------------------------------------- provenance
def test_stage_records_carry_parent_identities():
    fused = engine(registry(), admission_policy=lenient(min_accepted_updates=1, nis_window=1))
    for k, t, z in linear_stream(5):
        record = fused.ingest(raw(k, t - OFFSET, z, clock="clock_a"))
    candidate = fused.candidate(0.3)
    admitted = fused.admit(candidate, evaluated_at=0.3)
    transformed = fused.records[record["parents"][0]]
    source = fused.records[transformed["parents"][0]]
    previous = fused.records[record["parents"][1]]
    assert [source["stage"], transformed["stage"], record["stage"], candidate["stage"]] == list(STAGES[:4])
    assert source["parents"] == [] and previous["stage"] == "filtered_candidate"
    assert candidate["parents"] == [record["identity"]] and candidate["prediction_horizon_s"] == pytest.approx(0.1)
    assert source["identity"] in candidate["contributing_observations"]["identities"]
    assert candidate["contributing_observations"]["count"] == 5
    stages = [item["stage"] for item in fused.lineage(admitted["identity"])]
    assert stages[:5] == ["admitted_state", "candidate_state", "filtered_candidate", "transformed_observation",
                          "filtered_candidate"]
    assert stages.count("raw_observation") == 5 and admitted["stage"] in STAGES
    assert RawObservation.from_json(source).identity == source["identity"]
    with pytest.raises(Refusal) as caught:
        RawObservation.from_json(dict(source, sequence=77))
    assert caught.value.code == "identity_mismatch"
    for item in fused.history:
        serializable(item)


def test_retain_creates_ledger_entries_that_reference_parents_and_verify(tmp_path):
    fused = engine(registry())
    for k, t, z in linear_stream(12):
        fused.ingest(raw(k, t - OFFSET, z, clock="clock_a"))
    outlier = fused.ingest(raw(12, 0.6 - OFFSET, truth(0.6) + 0.5, clock="clock_a"))
    candidate = fused.candidate()
    decision = fused.admit(candidate, lenient(min_accepted_updates=5, nis_window=5), candidate["time"]["value"])
    ledger = Ledger.create(tmp_path / "ledger")
    entries = {}
    for record in fused.history:
        entries[record["identity"]] = retain(ledger, record)
    for record in fused.history:
        entry = entries[record["identity"]]
        assert entry["body_schema"] == "ciw.science.fusion-state.v1" and entry["kind"] == "fusion_state"
        assert entry["body"]["stage"] == record["stage"] and entry["body"]["record_identity"] == record["identity"]
        assert set(entry["refs"]) == {entries[parent]["entry_id"] for parent in record["parents"]}
    assert entries[outlier["identity"]]["body"]["state"]["nis"] == outlier["nis"]
    assert entries[decision["identity"]]["refs"] == [entries[candidate["identity"]]["entry_id"]]
    assert ledger.verify().ok
    reopened = Ledger.open(tmp_path / "ledger")
    assert len(reopened) == len(fused.history)
    with pytest.raises(Refusal) as caught:
        retain(ledger, dict(candidate, unit="mm"))
    assert caught.value.code == "identity_mismatch"


def test_example_input_replays_and_admits():
    scenario = json.loads((EXAMPLES / "two-tracker-scenario.json").read_text(encoding="utf-8"))
    policy = json.loads((EXAMPLES / "admission-policy.json").read_text(encoding="utf-8"))
    assert scenario["acquisition"] == "synthetic"
    fused = scenario_engine(scenario)
    records = replay(fused, scenario)
    assert len(records) == len(scenario["arrivals"])
    candidate = fused.candidate()
    assert fused.admit(candidate, policy, candidate["time"]["value"])["stage"] == "admitted_state"
    physical_only = dict(policy, allowed_acquisition=["physical"])
    refused = fused.admit(candidate, physical_only, candidate["time"]["value"])
    assert [item["code"] for item in refused["reasons"]] == ["acquisition_not_admissible"]
