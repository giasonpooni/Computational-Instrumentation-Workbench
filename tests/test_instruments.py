"""Scientific invariants of the demo and full-resolution analysis contract."""

from copy import deepcopy
import math

import numpy as np
import pytest

from ciw.instruments import (
    compute_spectrum, compute_statistics, inspect_sample, make_demo_run,
    run_metadata, validate_run,
)


@pytest.fixture
def run():
    return make_demo_run()


def test_demo_is_deterministic_analytic_and_energy_decays(run):
    assert run == make_demo_run()
    time = np.asarray(run["time_s"])
    q = np.asarray(run["channels"]["q"]["values"])
    v = np.asarray(run["channels"]["v"]["values"])
    energy = np.asarray(run["channels"]["energy"]["values"])
    omega, gamma = 2 * math.pi * 0.8, 0.15
    omega_d = math.sqrt(omega**2 - gamma**2)
    assert len(time) == 768
    assert time[0] == 0
    assert time[-1] == 12 - 1 / 64
    assert q[0] == 1 and v[0] == 0
    np.testing.assert_allclose(q, np.exp(-gamma * time) *
                              (np.cos(omega_d * time) + gamma / omega_d * np.sin(omega_d * time)))
    np.testing.assert_allclose(v, -(omega**2 / omega_d) * np.exp(-gamma * time) * np.sin(omega_d * time),
                              atol=1e-14)
    np.testing.assert_allclose(energy, 0.5 * (v**2 + omega**2 * q**2))
    assert np.all(np.diff(energy) <= 1e-12)
    assert energy[-1] < energy[0] * 0.04
    assert run["render"]["sample_indices"] == list(range(768))
    np.testing.assert_allclose(run["render"]["trajectory"], np.column_stack((q, energy, v)))
    surface = np.asarray(run["render"]["surface"]["vertices"])
    np.testing.assert_allclose(surface[:, 1], 0.5 * (surface[:, 2]**2 + omega**2 * surface[:, 0]**2))


def test_metadata_is_small_and_detached(run):
    description = run_metadata(run)
    assert "time_s" not in description and "render" not in description
    assert description["channels"]["q"] == {"unit": "m"}
    description["metadata"]["model"]["mass_kg"] = 50
    assert run["metadata"]["model"]["mass_kg"] == 1


def test_nearest_sample_earliest_tie_and_boundaries(run):
    assert inspect_sample(run, 0)["sample_index"] == 0
    assert inspect_sample(run, 12)["sample_index"] == 767
    assert inspect_sample(run, 20.5 / 64)["sample_index"] == 20
    assert inspect_sample(run, 20.51 / 64)["sample_index"] == 21
    sample = inspect_sample(run, 3 / 64)
    assert sample["time_s"] == 3 / 64
    assert sample["values"]["q"] == run["channels"]["q"]["values"][3]
    assert sample["units"] == {"q": "m", "v": "m/s", "energy": "J"}


@pytest.mark.parametrize("time", [-1, 12.01, math.nan, math.inf, True, "1", 10**400])
def test_bad_cursor_is_rejected(run, time):
    with pytest.raises(ValueError):
        inspect_sample(run, time)


def test_statistics_use_half_open_source_samples(run):
    run["channels"]["q"]["values"] = np.arange(768, dtype=np.float64).tolist()
    # Deliberately unrelated render data proves the computation uses the source.
    run["render"]["trajectory"] = [[0, 0, 0]]
    run["render"]["sample_indices"] = [0]
    result = compute_statistics(run, "q", [2 / 64, 5 / 64])
    assert result == {"sample_count": 3, "mean": 3.0, "minimum": 2.0,
                      "maximum": 4.0, "rms": pytest.approx(math.sqrt(29 / 3)), "unit": "m"}
    assert compute_statistics(run, "q", [0, 12])["sample_count"] == 768


@pytest.mark.parametrize("interval", [[0, 0], [1, 0], [-1, 2], [0, 12.1],
                                      [0, math.inf], [math.nan, 1], [True, 2],
                                      [0], "0,1", [0.001, 0.002], np.array(0),
                                      np.array([[0], [1]]), [0, 10**400]])
def test_invalid_or_empty_intervals_are_rejected(run, interval):
    for compute in (compute_statistics, compute_spectrum):
        with pytest.raises(ValueError):
            compute(run, "q", interval)


def test_spectrum_requires_four_samples_and_known_channel(run):
    with pytest.raises(ValueError, match="at least four"):
        compute_spectrum(run, "q", [0, 3 / 64])
    for compute in (compute_statistics, compute_spectrum):
        with pytest.raises(ValueError, match="channel"):
            compute(run, "unknown", [0, 12])


@pytest.mark.parametrize("count", [767, 768])
def test_hann_periodogram_parseval_for_odd_and_even_windows(run, count):
    rng = np.random.default_rng(8142)
    values = rng.normal(size=768) + 3.25
    run["channels"]["q"]["values"] = values.tolist()
    result = compute_spectrum(run, "q", [0, count / 64])
    selected = values[:count]
    window = np.hanning(count + 1)[:-1]
    expected = np.sum(((selected - selected.mean()) * window)**2) / np.sum(window**2)
    integrated = np.sum(result["psd"]) * (64 / count)
    assert integrated == pytest.approx(expected, rel=2e-14)
    assert len(result["frequency_hz"]) == count // 2 + 1
    assert result["frequency_hz"][-1] == pytest.approx((count // 2) * 64 / count)
    assert result["sample_count"] == count
    assert result["unit"] == "(m)^2/Hz"


def test_known_sinusoid_psd_power_and_peak(run):
    time = np.asarray(run["time_s"])
    run["channels"]["q"]["values"] = (2 * np.sin(2 * np.pi * 4 * time) + 7).tolist()
    result = compute_spectrum(run, "q", [0, 12])
    assert result["peak_frequency_hz"] == pytest.approx(4)
    assert sum(result["psd"]) / 12 == pytest.approx(2, rel=1e-13)
    assert result["psd"][48] == pytest.approx(16, rel=1e-13)


def test_even_nyquist_bin_is_not_doubled(run):
    run["channels"]["q"]["values"] = ((-1.0)**np.arange(768)).tolist()
    result = compute_spectrum(run, "q", [0, 12])
    assert result["peak_frequency_hz"] == 32
    assert result["psd"][-1] == pytest.approx(8)
    assert sum(result["psd"]) / 12 == pytest.approx(1)


@pytest.mark.parametrize("constant", [0.0, 5.0])
def test_zero_detrended_signal_has_no_peak(run, constant):
    run["channels"]["q"]["values"] = [constant] * 768
    result = compute_spectrum(run, "q", [0, 12])
    assert result["peak_frequency_hz"] is None
    assert result["psd"] == [0.0] * 385


def test_large_finite_statistics_and_nonfinite_spectrum_output_guard(run):
    run["channels"]["q"]["values"] = [1e308, -1e308] * 384
    result = compute_statistics(run, "q", [0, 12])
    assert result["mean"] == 0
    assert result["rms"] == 1e308
    with pytest.raises(ValueError, match="finite float64"):
        compute_spectrum(run, "q", [0, 12])


@pytest.mark.parametrize("corruption", [
    lambda run: run["time_s"].__setitem__(4, run["time_s"][3]),
    lambda run: run["time_s"].__setitem__(4, run["time_s"][4] + 0.001),
    lambda run: run["time_s"].__setitem__(0, -1),
    lambda run: run["time_s"].pop(),
    lambda run: run["channels"]["v"]["values"].__setitem__(700, math.nan),
    lambda run: run["channels"]["q"]["values"].__setitem__(0, True),
    lambda run: run["channels"]["energy"]["values"].pop(),
    lambda run: run["channels"]["q"].__setitem__("unit", "cm"),
    lambda run: run["metadata"].__setitem__("sample_rate_hz", 0),
    lambda run: run["metadata"].__setitem__("duration_s", 13),
    lambda run: run["metadata"].__setitem__("sample_count", True),
    lambda run: run["metadata"]["model"].__setitem__("mass_kg", math.inf),
    lambda run: run["metadata"]["model"].__setitem__("mass_kg", 10**400),
    lambda run: run["render"]["trajectory"][0].__setitem__(0, math.nan),
    lambda run: run["render"]["sample_indices"].__setitem__(0, 768),
    lambda run: run["render"]["surface"]["indices"].__setitem__(0, -1),
])
def test_every_loaded_source_is_validated_before_analysis(run, corruption):
    corrupted = deepcopy(run)
    corruption(corrupted)
    with pytest.raises(ValueError):
        validate_run(corrupted)
    # The error is rejected even when outside this channel / analysis interval.
    with pytest.raises(ValueError):
        compute_statistics(corrupted, "q", [0, 1])
    with pytest.raises(ValueError):
        compute_spectrum(corrupted, "q", [0, 1])


def test_render_geometry_is_bound_to_the_retained_channels_where_evidence_is_kept(run, tmp_path):
    from ciw.adapters.oscillator import validate_render_binding
    from ciw.session import Session
    validate_render_binding(run)
    Session(run, tmp_path / "live")
    drawn_elsewhere = deepcopy(run)
    drawn_elsewhere["render"]["trajectory"][7][0] += 1e-9
    with pytest.raises(ValueError, match="render.trajectory"):
        validate_render_binding(drawn_elsewhere)
    with pytest.raises(ValueError, match="render.trajectory"):
        Session(drawn_elsewhere, tmp_path / "tampered")
    off_surface = deepcopy(run)
    off_surface["render"]["surface"]["vertices"][12][1] *= 1.001
    with pytest.raises(ValueError, match="energy surface"):
        validate_render_binding(off_surface)
    rewired = deepcopy(run)
    rewired["render"]["surface"]["indices"][2563] += 1
    with pytest.raises(ValueError, match="triangulation"):
        validate_render_binding(rewired)
    subset = deepcopy(run)
    subset["render"]["trajectory"] = subset["render"]["trajectory"][::4]
    subset["render"]["sample_indices"] = subset["render"]["sample_indices"][::4]
    validate_render_binding(subset)
    session = Session(run, tmp_path / "saved")
    workspace = session.save_workspace(tmp_path / "workspace.json")
    saved = json.loads(workspace.read_text(encoding="utf-8"))
    saved["run"]["render"]["trajectory"][0] = [0.0, 0.0, 0.0]
    workspace.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="render.trajectory"):
        Session.from_workspace(workspace, tmp_path / "reopened")
