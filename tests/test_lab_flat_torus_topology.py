"""Flat torus and topology lab tasks T019-T032: labels, key numbers and refusals.

The section runs once per module (shared memo, about 10 s); unit tests
exercise the exact helpers directly. Optional modules only change which
labels are expected; the pinned FTR comparison runs only when
CIW_LAB_FTR_REPO and CIW_LAB_FTR_PYTHON name a checkout and a Python 3.12
interpreter.
"""
from fractions import Fraction
import importlib.util
import math
import os
from pathlib import Path
import sys

import pytest

from ciw.lab import flat_torus_topology as ftt
from ciw.lab import flat_torus_topology_lattice as lat
from ciw.lab import flat_torus_topology_surfaces as surf
from ciw.lab import runner
from ciw.lab.registry import _REGISTRY, load_queue
from ciw.lab.report import validate_report

TASKS = [f"T0{n}" for n in range(19, 33)]
HAS = {name: importlib.util.find_spec(name) is not None for name in ("scipy", "sympy", "mpmath")}


def _queue():
    return {t["id"]: t for t in load_queue()["tasks"]}


def _run(task_ids, directory, providers=None):
    queue, ctx = _queue(), runner.Context(directory, providers)
    return {tid: validate_report(runner.run_task(queue[tid], _REGISTRY[tid], ctx, {})) for tid in task_ids}


@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    return _run(TASKS, tmp_path_factory.mktemp("flat-torus"))


def _findings(report):
    return {f["claim"]: f for f in report["findings"]}


def _label(report, prefix):
    matches = [f for f in report["findings"] if f["claim"].startswith(prefix)]
    assert len(matches) == 1, prefix
    return matches[0]


def _expect(module):
    return "independently_verified" if HAS[module] else "numerically_verified"


def test_every_task_is_registered_and_completes(reports):
    assert set(TASKS) <= set(_REGISTRY)
    for tid in TASKS:
        report = reports[tid]
        assert report["state"] == "completed", (tid, report["experiment"])
        assert report["findings"] and report["physical_validation_status"]["status"] == "not_established"
        for name in ("hypothesis", "mathematical_model", "numerical_result", "uncertainty", "recommended_next_task"):
            assert isinstance(report[name], str) and report[name].strip()
        assert report["failure_modes_checked"] and report["unresolved_assumptions"]
        assert report["generated_artifacts"], tid
        for record in report["findings"]:
            if record["domain"] in ("mathematical", "numerical") and record["value"] is not None:
                assert "regression_tolerance" in record, (tid, record["claim"])


# ---------------------------------------------------------------- lattice
def test_gauss_reduction_is_exact():
    canonical, M, _ = lat.gauss_reduce((5, 2, 7))
    assert canonical == (5, 2, 7) and M == ((1, 0), (0, 1))
    skewed = lat.transform((5, 2, 7), ((13, 8), (21, 13)))
    reduced, R, steps = lat.gauss_reduce(skewed)
    assert reduced == (5, 2, 7) and lat.transform(skewed, R) == reduced and steps > 0
    assert lat.gauss_reduce((2, -1, 2))[0] == (2, 1, 2)  # boundary: b >= 0
    assert lat.gauss_reduce((Fraction(1, 2), Fraction(1, 7), Fraction(3, 2)))[0][0] == Fraction(1, 2)
    assert [lat.reduced_basis_count(G) for G in ((5, 2, 7), (1, 0, 1), (2, 1, 2), (2, 1, 5))] == [2, 4, 12, 4]
    tau, M = lat.float_reduce(lat.mobius(((2, 1), (5, 3)), complex(0.31, 1.07)))
    assert abs(tau - complex(0.31, 1.07)) < 1e-12 and lat.det2(M) == 1
    for bad, code in ((((2, 0), (0, 1)), "BASIS_CHANGE_NOT_UNIMODULAR"),
                      (((1, 0), (0, -1)), "BASIS_CHANGE_REVERSES_ORIENTATION")):
        with pytest.raises(lat.LatticeRefusal) as info:
            lat.transform((5, 2, 7), bad)
        assert info.value.code == code
    with pytest.raises(lat.LatticeRefusal, match="FLOAT_ENTRY"):
        lat.gram(((1.0, 0), (0, 1)))
    with pytest.raises(lat.LatticeRefusal, match="POSITIVE_DEFINITE"):
        lat.gram(((1, 2), (2, 1)))


def test_t019_reduction_and_refusals(reports):
    report = reports["T019"]
    first = report["findings"][0]
    assert first["evidence_status"] == _expect("sympy")
    assert first["value"]["failures"] == {"inverse": 0, "area": 0, "reduction": 0, "reducer": 0, "automorphism": 0}
    assert first["value"]["reductions"] == 6 * (308 + 200)
    counts = _label(report, "Number of reduced bases")
    assert counts["value"] == ftt.PREDICTED_REDUCED_BASES and counts["evidence_status"] == "numerically_verified"
    refusals = _label(report, "Integer matrices with det != 1")
    assert refusals["value"]["det 2 (index-2 sublattice)"] == "BASIS_CHANGE_NOT_UNIMODULAR"
    assert refusals["counterexample"]["witness"]["image_canonical"] != refusals["counterexample"]["witness"]["canonical"]


def test_ftr_refusal_makes_task_partial(tmp_path):
    checkout = tmp_path / "not-a-checkout"
    checkout.mkdir()
    report = _run(["T019"], tmp_path / "out", {"ftr": checkout, "ftr-python": Path(sys.executable)})["T019"]
    assert report["state"] == "partial"
    refused = _label(report, "A bound FTR provider")
    assert refused["value"] in ("FTR_CHECKOUT_UNREADABLE", "FTR_REVISION_MISMATCH")
    assert refused["evidence_status"] == "numerically_verified"


@pytest.mark.skipif(not (os.environ.get("CIW_LAB_FTR_REPO") and os.environ.get("CIW_LAB_FTR_PYTHON")),
                    reason="set CIW_LAB_FTR_REPO and CIW_LAB_FTR_PYTHON to run the pinned FTR comparison")
def test_ftr_provider_agreement(tmp_path):
    providers = {"ftr": Path(os.environ["CIW_LAB_FTR_REPO"]), "ftr-python": Path(os.environ["CIW_LAB_FTR_PYTHON"])}
    out = _run(["T019", "T020", "T026"], tmp_path, providers)
    for tid, prefix in (("T019", "Float Gauss reduction agrees"), ("T020", "Closed-geodesic lengths |m w1"),
                        ("T026", "Closed-geodesic lengths before and after the fold")):
        assert out[tid]["state"] == "completed"
        record = _label(out[tid], prefix)
        assert record["evidence_status"] == "independently_verified"
        assert record["basis"]["independent_check"]["checker"]["implementation"].startswith(
            "Flat-Torus-Geodesic-Reference@")
        assert out[tid]["provider_runtime_identity"]["provider"]["revision"] == \
            "dc918562cd9e351a65475d29f46963c9f2fd7db8"
    assert _label(out["T019"], "Float Gauss reduction agrees")["value"]["matrix_mismatches"] == 0
    assert _label(out["T020"], "Closed-geodesic lengths |m w1")["value"]["crossing_mismatches"] == 0


# ---------------------------------------------------------------- windings
def test_winding_flow_and_intersections():
    start = (Fraction(1, 7), Fraction(2, 11))
    assert lat.trace_lattice_flow(3, -2, start)["period"] == 1
    assert lat.trace_lattice_flow(3, -2, start)["crossings"] == 5
    doubled = lat.trace_lattice_flow(2, 4, start)
    assert doubled["period"] == Fraction(1, 2) and doubled["crossings"] == 3
    assert lat.intersection_count((1, 0), (0, 1)) == 1
    assert lat.intersection_count((2, 1), (1, 3)) == 5
    assert lat.classify_winding(4, 6)["cover_degree"] == 2
    with pytest.raises(lat.LatticeRefusal, match="ZERO_WINDING"):
        lat.classify_winding(0, 0)


def test_t020_winding_classification(reports):
    report = reports["T020"]
    first = report["findings"][0]
    assert first["value"] == {"classes": 168, "failures": 0} and first["evidence_status"] == "numerically_verified"
    golden = _label(report, "The golden-slope geodesic")
    assert golden["evidence_status"] == _expect("mpmath") and min(golden["value"]["gap"]) > 0
    rational = _label(report, "A binary64 heading slope is rational")
    assert rational["value"]["log2_denominator"] <= 52 and "counterexample" in rational


# ---------------------------------------------------------------- flat routes and ties
def test_t021_shortest_is_least_sensitive(reports):
    report = reports["T021"]
    numeric = report["findings"][0]
    assert numeric["evidence_status"] == "numerically_verified"
    assert numeric["value"]["discordant_pairs"] == 0 and numeric["value"]["max_relative_j_head_error"] < 1e-12
    labels = {f["domain"]: f["evidence_status"] for f in report["findings"]}
    assert labels["machine_safety"] == "not_established"
    assert sum(f["evidence_status"] == "analytic" for f in report["findings"]) == 2


def test_t022_degenerate_representatives(reports):
    report = reports["T022"]
    values = [f["value"] for f in report["findings"]]
    assert values[0] == {"(1/2, 0)": 2, "(0, 1/2)": 2, "(1/2, 1/2)": 4, "(1/3, 1/5)": 1}
    assert values[1] == {"1": 121, "2": 22, "4": 1}
    assert set(values[2].values()) == {2}
    counter = report["findings"][3]
    assert counter["value"]["exact"] == 1 and counter["value"]["binary64"] == 2 and "counterexample" in counter
    assert all(f["evidence_status"] == "numerically_verified" for f in report["findings"])


def test_t023_heading_sensitivity(reports):
    report = reports["T023"]
    zeros = report["findings"][0]
    assert zeros["value"]["found"] == zeros["value"]["predicted"] == 18
    assert report["findings"][1]["value"]["max_relative_slope_error"] < 1e-9
    widest = report["findings"][2]["value"][0]
    assert sorted(abs(x) for x in widest["winding"]) == [0, 1]
    assert report["findings"][3]["domain"] == "sensor_performance"
    assert report["findings"][3]["evidence_status"] == "not_established"


# ---------------------------------------------------------------- curved routes
def test_t024_t025_route_ranking_and_front(reports):
    report = reports["T024"]
    table = report["findings"][0]
    assert table["evidence_status"] == _expect("scipy")
    assert len(table["value"]) >= 5
    shortest = table["value"][0]
    assert shortest["length"] == pytest.approx(5.8465, abs=1e-3)
    ranks = report["findings"][1]["value"]
    assert ranks["by_length"] != ranks["by_amplification"] and ranks["by_length"] != ranks["by_focus_margin"]
    assert "counterexample" in report["findings"][1]
    assert report["findings"][2]["value"] == pytest.approx(1.0, abs=1e-12)
    assert report["findings"][3]["evidence_status"] == "not_established"
    front = reports["T025"]["findings"][0]
    assert front["evidence_status"] == "numerically_verified"
    assert 0 in front["value"]["front"] and len(front["value"]["front"]) >= 2
    assert reports["T025"]["findings"][1]["domain"] == "machine_safety"


def test_t032_counterexample_library(reports):
    report = reports["T032"]
    counter = [f for f in report["findings"] if "counterexample" in f]
    assert len(counter) == 5
    inner = _label(report, "Torus inner equator")
    assert inner["evidence_status"] == _expect("scipy")
    assert inner["value"]["shortest"]["amplification"] == pytest.approx(math.sinh(2.5), rel=1e-6)
    assert inner["value"]["alternative"]["length"] > inner["value"]["shortest"]["length"]
    outer = _label(report, "Torus outer equator")
    assert outer["value"]["shortest"]["focus_margin"] == pytest.approx(math.pi * math.sqrt(3) - 4.5, abs=1e-6)
    sphere = _label(report, "Unit sphere")
    assert sphere["value"]["focus_margin"] == pytest.approx(0.05, abs=1e-12)
    assert _label(report, "Search on the saddle")["value"] == {"routes": 1}
    assert _label(report, "A route from this library")["evidence_status"] == "not_established"


# ---------------------------------------------------------------- invariance and surfaces
def test_t026_modular_invariance(reports):
    report = reports["T026"]
    first = report["findings"][0]
    assert first["value"]["failures"] == {"spectrum": 0, "area": 0, "systole": 0, "winding_rule": 0}
    assert first["value"]["matrices"] == 308 + 60
    mirror = _label(report, "The length spectrum does not determine")
    assert mirror["value"]["same_spectrum"] and not mirror["value"]["same_canonical"]
    assert _label(report, "Transporting winding labels")["value"] > 0
    assert _label(report, "A det-2 integer matrix")["value"]["area_sq_ratio"] == "4"


def test_t027_t029_surfaces_and_cones(reports):
    records = reports["T027"]["findings"][0]["value"]
    assert records["L-shape"]["genus"] == records["octagon"]["genus"] == 2
    assert records["octagon"]["area"] == {"rational": "2", "sqrt_coefficient": "2", "radicand": 2}
    codes = reports["T027"]["findings"][2]["value"]
    assert codes == {"octagon adjacent pairing": "GLUING_NOT_TRANSLATION", "pillowcase": "GLUING_NOT_TRANSLATION",
                     "mismatched edge lengths": "EDGE_LENGTH_MISMATCH"}
    cones = reports["T029"]["findings"][0]["value"]
    assert cones["octagon"] == ["6"] and cones["H(1,1) origami"] == ["4", "4"] and cones["pillowcase"] == ["1"] * 4
    assert set(reports["T029"]["findings"][1]["value"].values()) == {"0"}
    assert "counterexample" in reports["T029"]["findings"][2]


def test_surface_helpers_exact():
    octagon = surf.regular_octagon()
    s = surf.Surd(0, Fraction(1, 2))
    assert octagon.area() == surf.Surd(2, 2) and (s * s) == Fraction(1, 2) and s > Fraction(7, 10)
    assert surf.commutator_cycles([1, 0, 2], [2, 1, 0]) == [3]
    with pytest.raises(surf.GluingRefusal, match="GLUING_NOT_TRANSLATION"):
        surf.pillowcase().flow(0, (Fraction(1, 2), Fraction(1, 3)), (1, 1))
    with pytest.raises(surf.FlowTermination) as info:
        surf.l_shape().flow(0, (Fraction(1, 2), Fraction(1, 2)), (1, 1))
    assert info.value.code == "SADDLE_CONNECTION"


def test_t028_glued_edge_flow(reports):
    report = reports["T028"]
    first = report["findings"][0]
    assert first["value"] == {"closed": 90, "saddle_connections": 6, "undecided": 0}
    refusals = report["findings"][1]["value"]
    assert refusals == {"exact_l_shape": "SADDLE_CONNECTION", "exact_octagon": "SADDLE_CONNECTION",
                        "float_octagon": "NEAR_VERTEX_WITHIN_TOLERANCE", "float_start": "START_NOT_INTERIOR"}
    assert report["findings"][2]["value"]["time_difference"] < 1e-12
    assert all(f["evidence_status"] == "numerically_verified" for f in report["findings"])


# ---------------------------------------------------------------- discrete and perturbed metrics
def test_t030_grid_metrication(reports):
    report = reports["T030"]
    grid4 = report["findings"][0]
    assert grid4["evidence_status"] == _expect("scipy")
    assert grid4["value"]["ratio4_45deg"] == pytest.approx([math.sqrt(2)] * 3, abs=1e-12)
    assert "counterexample" in grid4
    worst = report["findings"][1]["value"]
    assert worst["8"] == pytest.approx(math.sqrt(4 - 2 * math.sqrt(2)) - 1, abs=1e-8)
    fmm = report["findings"][2]["value"]["max_relative_error"]
    assert fmm["30"] > fmm["60"] > fmm["120"]
    assert report["findings"][3]["domain"] == "physical" and report["findings"][3]["evidence_status"] == "not_established"


def test_t031_route_switch(reports):
    report = reports["T031"]
    threshold = report["findings"][0]
    assert threshold["value"]["eps_star"] == "1/250" and threshold["evidence_status"] == _expect("sympy")
    assert all(row["ratio"] == "4" for row in threshold["value"]["scaling"])
    jump = report["findings"][1]
    assert jump["value"]["heading_jump_deg"] == pytest.approx(126.87, abs=0.01) and "counterexample" in jump
    assert report["findings"][2]["domain"] == "calibration"
    assert report["findings"][2]["evidence_status"] == "not_established"
    assert ftt.flip_threshold(Fraction(1, 64))[0] == Fraction(1, 16)


def test_regeneration_is_within_tolerance(tmp_path):
    cheap = ["T021", "T022", "T027", "T029", "T031"]
    for name in ("a", "b"):
        for tid, report in _run(cheap, tmp_path / name).items():
            (tmp_path / name / "reports").mkdir(parents=True, exist_ok=True)
            (tmp_path / name / "reports" / f"{tid}.json").write_text(runner.dumps(report), encoding="utf-8")
    result = runner.compare(tmp_path / "a", tmp_path / "b")
    assert result["passed"], result["problems"]
