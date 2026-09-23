"""Flat torus and topology lab tasks T019-T032: labels, key numbers and refusals.

The section runs once per module (shared memo, about 15 s); unit tests
exercise the exact helpers directly. Optional modules only change which
labels are expected; the pinned FTR comparison runs only when
CIW_LAB_FTR_REPO and CIW_LAB_FTR_PYTHON name a checkout and a Python 3.12
interpreter.
"""
from fractions import Fraction
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import pytest

from ciw.lab import flat_torus_topology as ftt
from ciw.lab import flat_torus_topology_lattice as lat
from ciw.lab import flat_torus_topology_provider as ftr
from ciw.lab import flat_torus_topology_routes as routes
from ciw.lab import flat_torus_topology_surfaces as surf
from ciw.lab import runner
from ciw.lab.evidence import finding
from ciw.lab.registry import load_queue, section_implementations
from ciw.lab.report import validate_report

TASKS = [f"T0{n}" for n in range(19, 33)]
IMPLEMENTATIONS = section_implementations("flat-torus-topology")
HAS = {name: importlib.util.find_spec(name) is not None for name in ("scipy", "sympy", "mpmath")}
NV, NE, AN = "numerically_verified", "not_established", "analytic"

# Headline (primary) label per task: the weakest established computational label.
HEADLINE = {tid: NV for tid in TASKS} | {"T021": AN}

# Expected label of every finding, by claim prefix. A tuple names the optional modules whose
# independent check lifts the finding to independently_verified.
LABELS = {
    "T019": {"Every enumerated SL(2,Z) basis": ("sympy",), "Number of reduced bases": NV,
             "Integer matrices with det != 1": NV, "Float Gauss reduction agrees with the pinned FTR": NE},
    "T020": {"Every winding with": NV, "For all 120 pairs": NV, "The lattice-point count": NV,
             "The golden-slope geodesic": ("mpmath",), "A binary64 heading slope": NV,
             "Closed-geodesic lengths |m w1": NE},
    "T021": {"On the test flat torus": NV, "For every flat torus": AN, "Shortest equals least heading-sensitive": AN,
             "The shortest route on a physical": NE},
    "T022": {"Square-torus half-period": NV, "Multiplicity census": NV, "On six lattices": NV,
             "Rounding the target to binary64": NV, "At exact cut-locus ties": NV},
    "T023": {"Zeros of the return distance": NV, "Near each closing heading": NV, "Closure basins are widest": NV,
             "A physical heading sensor": NE},
    "T024": {"Every fan-search route": ("scipy", "sympy"), "The torus route set is unchanged": NV,
             "Rankings by length": NV, "A flat torus has no conjugate points": NV, "Ranking routes by focus": NE},
    "T025": {"The three-objective front": NV, "A Pareto-optimal route": NE},
    "T026": {"Length spectrum": NV, "Area-one float spectra": NV, "Transporting winding labels": NV,
             "The length spectrum does not determine": NV, "A det-2 integer matrix": NV,
             "Closed-geodesic lengths before and after": NE},
    "T027": {"L-shape and regular octagon": NV, "Horizontal cylinders": NV, "Pairings that are not": NV},
    "T028": {"All 96 tested": NV, "Trajectories hitting a cone point": NV, "Float octagon flow": NV,
             "A generic float octagon": NV},
    "T029": {"Cone angles": NV, "Gauss-Bonnet": NV, "Polygon vertices need not": NV},
    "T030": {"4- and 8-neighbour": ("scipy",), "Worst-direction": NV, "Fast marching converges": NV,
             "Grid-planned path lengths": NE},
    "T031": {"The shortest route switches": ("sympy",), "A metric perturbation just above": NV,
             "A metric calibrated": NE},
    "T032": {"Torus inner equator": ("scipy", "sympy"), "Torus outer equator": ("scipy", "sympy"),
             "Torus (0, 0) -> (2.2, 0)": NV, "Gaussian bump": ("scipy", "sympy"), "On the unit sphere": NV,
             "Search on the saddle": NV, "Every T032 route search": NV, "A route from this library": NE},
}


def _queue():
    return {t["id"]: t for t in load_queue()["tasks"]}


def _run(task_ids, directory, providers=None):
    queue, ctx = _queue(), runner.Context(directory, providers)
    return {tid: validate_report(runner.run_task(queue[tid], IMPLEMENTATIONS[tid], ctx, {})) for tid in task_ids}


@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    return _run(TASKS, tmp_path_factory.mktemp("flat-torus"))


def _label(report, prefix):
    matches = [f for f in report["findings"] if f["claim"].startswith(prefix)]
    assert len(matches) == 1, prefix
    return matches[0]


def _expect(label):
    if isinstance(label, tuple):
        return "independently_verified" if all(HAS[m] for m in label) else NV
    return label


def test_every_task_is_registered_and_completes(reports):
    assert set(IMPLEMENTATIONS) == set(TASKS)
    for tid in TASKS:
        report = reports[tid]
        assert report["state"] == "completed", (tid, report["experiment"])
        assert report["evidence_status"]["primary"] == HEADLINE[tid], tid
        assert report["findings"] and report["physical_validation_status"]["status"] == "not_established"
        for name in ("hypothesis", "mathematical_model", "numerical_result", "uncertainty", "recommended_next_task"):
            assert isinstance(report[name], str) and report[name].strip()
        assert report["failure_modes_checked"] and report["unresolved_assumptions"]
        assert report["generated_artifacts"], tid
        for record in report["findings"]:
            if record["domain"] in ("mathematical", "numerical") and not record.get("expected_not_established"):
                assert "regression_tolerance" in record, (tid, record["claim"])
                assert {"kind", "value", "basis"} <= set(record["uncertainty"]), (tid, record["claim"])


def test_every_finding_has_its_expected_label(reports):
    for tid in TASKS:
        findings = reports[tid]["findings"]
        assert len(findings) == len(LABELS[tid]), tid
        for prefix, label in LABELS[tid].items():
            record = _label(reports[tid], prefix)
            assert record["evidence_status"] == _expect(label), (tid, prefix)
            if record["domain"] in ("mathematical", "numerical") and record["evidence_status"] == NE:
                assert record["expected_not_established"] is True and record["value"] == "not bound", (tid, prefix)


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


def test_sympy_reduction_check_detects_a_wrong_boundary_rule(monkeypatch):
    pytest.importorskip("sympy")
    assert ftt.sympy_reduction_check() == {"compared": 3048, "mismatches": 0}
    original = lat.gauss_reduce

    def wrong_boundary(form):
        reduced, M, steps = original(form)
        a, b, c = reduced
        return ((a, -b, c) if b and (2 * b == a or a == c) else reduced), M, steps
    monkeypatch.setattr(lat, "gauss_reduce", wrong_boundary)
    assert ftt.sympy_reduction_check()["mismatches"] > 0


def test_t019_reduction_and_refusals(reports):
    report = reports["T019"]
    first = _label(report, "Every enumerated SL(2,Z) basis")
    assert first["value"]["failures"] == {"inverse": 0, "area": 0, "reduction": 0, "reducer": 0, "automorphism": 0}
    assert first["value"]["reductions"] == 6 * (308 + 200)
    if HAS["sympy"]:
        assert "3048 bases and words" in first["basis"]["independent_check"]["reference"]
    counts = _label(report, "Number of reduced bases")
    assert counts["value"] == ftt.PREDICTED_REDUCED_BASES
    refusals = _label(report, "Integer matrices with det != 1")
    assert refusals["value"]["det 2 (index-2 sublattice)"] == "BASIS_CHANGE_NOT_UNIMODULAR"
    assert refusals["counterexample"]["witness"]["image_canonical"] != refusals["counterexample"]["witness"]["canonical"]


def test_ftr_refusal_makes_task_partial(tmp_path):
    checkout = tmp_path / "not-a-checkout"
    checkout.mkdir()
    report = _run(["T019"], tmp_path / "out", {"ftr": checkout, "ftr-python": Path(sys.executable)})["T019"]
    assert report["state"] == "partial" and report["evidence_status"]["primary"] == NV
    refused = _label(report, "Float Gauss reduction agrees with the pinned FTR")
    assert refused["evidence_status"] == NE and refused["expected_not_established"] is True
    assert refused["value"]["refusal"] in ("FTR_CHECKOUT_UNREADABLE", "FTR_REVISION_MISMATCH")
    assert not refused["basis"]
    assert len(report["findings"]) == len(LABELS["T019"])


def test_provider_output_is_refused_when_unreadable_or_incomplete():
    cases = ftt.ftr_cases()
    request = ftt.ftr_request(cases)
    good = {"python": "3.12.0", "numpy": "2.0", "areas": [1.0] * len(request["areas"]),
            "lengths": [1.0] * len(request["lengths"]),
            "traces": [{"crossings": 1, "closed": True, "length": 1.0}] * len(request["traces"]),
            "folds": [{"matrix": [1, 0, 0, 1], "reduced": [0.1, 1.1], "reduced_winding": [1, 0],
                       "length_pair": [1.0, 1.0], "word": []}] * len(request["folds"])}
    assert ftr.parse_output(json.dumps(good), request)["python"] == "3.12.0"
    for text, code in (("", "FTR_OUTPUT_UNREADABLE"), ("warning: something\n{}", "FTR_OUTPUT_UNREADABLE"),
                       (json.dumps({k: v for k, v in good.items() if k != "numpy"}), "FTR_OUTPUT_UNREADABLE"),
                       (json.dumps(dict(good, lengths=good["lengths"][:-1])), "FTR_OUTPUT_INCOMPLETE"),
                       (json.dumps(dict(good, traces=[{"crossings": "1", "closed": True}] * len(request["traces"]))),
                        "FTR_OUTPUT_UNREADABLE")):
        with pytest.raises(ftr.ProviderRefusal) as info:
            ftr.parse_output(text, request)
        assert info.value.code == code


@pytest.mark.skipif(not (os.environ.get("CIW_LAB_FTR_REPO") and os.environ.get("CIW_LAB_FTR_PYTHON")),
                    reason="set CIW_LAB_FTR_REPO and CIW_LAB_FTR_PYTHON to run the pinned FTR comparison")
def test_ftr_provider_agreement(tmp_path):
    providers = {"ftr": Path(os.environ["CIW_LAB_FTR_REPO"]), "ftr-python": Path(os.environ["CIW_LAB_FTR_PYTHON"])}
    out = _run(["T019", "T020", "T026"], tmp_path, providers)
    for tid in ("T019", "T020", "T026"):
        assert out[tid]["state"] == "completed"
        record = _label(out[tid], ftt.FTR_CLAIMS[tid])
        assert record["evidence_status"] == "independently_verified"
        assert record["basis"]["independent_check"]["checker"]["implementation"].startswith(
            "Flat-Torus-Geodesic-Reference@")
        assert out[tid]["provider_runtime_identity"]["provider"]["revision"] == \
            "dc918562cd9e351a65475d29f46963c9f2fd7db8"
    assert _label(out["T019"], ftt.FTR_CLAIMS["T019"])["value"]["matrix_mismatches"] == 0
    assert _label(out["T020"], ftt.FTR_CLAIMS["T020"])["value"]["crossing_mismatches"] == 0
    assert any(a["path"].endswith("ftr-fold-lengths.json") for a in out["T026"]["generated_artifacts"])


# ---------------------------------------------------------------- windings
def test_winding_flow_and_intersections():
    start = (Fraction(1, 7), Fraction(2, 11))
    primitive = lat.trace_lattice_flow(3, -2, start)
    assert primitive["first_return"] == 1 and primitive["returns"] == 1 and primitive["crossings"] == 5
    doubled = lat.trace_lattice_flow(2, 4, start)
    assert doubled["first_return"] == Fraction(1, 2) and doubled["return_times"] == [Fraction(1, 2), 1]
    assert doubled["displacement_at_first_return"] == (1, 2) and doubled["crossings"] == 3
    assert lat.trace_lattice_flow(0, -3, start)["return_times"] == [Fraction(1, 3), Fraction(2, 3), 1]
    with pytest.raises(lat.LatticeRefusal, match="START_ON_CELL_WALL"):
        lat.trace_lattice_flow(1, 1, (0, Fraction(1, 2)))
    assert lat.intersection_count((1, 0), (0, 1)) == 1
    assert lat.intersection_count((2, 1), (1, 3)) == 5
    p1, p2 = (Fraction(1, 7), Fraction(2, 11)), (Fraction(3, 13), Fraction(5, 17))
    for v1, v2 in (((2, 1), (1, 3)), ((1, -2), (3, 1)), ((0, 1), (1, 1))):
        solutions = lat.intersection_solutions(v1, v2, p1, p2)
        assert len(solutions) == abs(v1[0] * v2[1] - v1[1] * v2[0])
        for t, s, (kx, ky) in solutions:  # p1 + t v1 = p2 + s v2 + k, exactly
            assert (p1[0] + t * v1[0], p1[1] + t * v1[1]) == (p2[0] + s * v2[0] + kx, p2[1] + s * v2[1] + ky)
    assert lat.box_lattice_count((5, 2, 7), 400) == (len(lat.lattice_vectors((5, 2, 7), 400)) + 1, 142)
    assert lat.classify_winding(4, 6)["cover_degree"] == 2
    with pytest.raises(lat.LatticeRefusal, match="ZERO_WINDING"):
        lat.classify_winding(0, 0)


def test_t020_winding_classification(reports):
    report = reports["T020"]
    first = _label(report, "Every winding with")
    assert first["value"] == {"classes": 168, "failures": 0}
    assert _label(report, "For all 120 pairs")["value"] == {"pairs": 120, "failures": 0}
    count = _label(report, "The lattice-point count")["value"]
    assert count["lattice_points_with_origin"] == 229 and count["primitive"] == 142
    golden = _label(report, "The golden-slope geodesic")
    assert min(golden["value"]["gap"]) > 0
    checks = {c["reference"]: c for c in golden["basis"]["checks"]}
    assert checks["smallest gap / phi^-k (a closure would give 0)"]["observed"] > 0.99
    assert checks["|q gap - 1/sqrt 5| minus (phi^-2k / sqrt 5 + binary64 rounding bound)"]["observed"] <= 0
    rational = _label(report, "A binary64 heading slope is rational")
    assert rational["value"]["log2_denominator"] <= 52 and "counterexample" in rational


# ---------------------------------------------------------------- flat routes and ties
def test_t021_shortest_is_least_sensitive(reports):
    report = reports["T021"]
    numeric = _label(report, "On the test flat torus")
    assert numeric["value"]["discordant_pairs"] == 0 and numeric["value"]["max_relative_j_head_error"] < 1e-12
    assert numeric["value"]["routes"] == 19
    scope = _label(report, "Shortest equals least heading-sensitive")
    assert "constant K <= 0" in scope["claim"] and "sinh" in scope["basis"]["derivation"]


def test_t022_degenerate_representatives(reports):
    report = reports["T022"]
    assert _label(report, "Square-torus half-period")["value"] == \
        {"(1/2, 0)": 2, "(0, 1/2)": 2, "(1/2, 1/2)": 4, "(1/3, 1/5)": 1}
    assert _label(report, "Multiplicity census")["value"] == {"1": 121, "2": 22, "4": 1}
    assert set(_label(report, "On six lattices")["value"].values()) == {2}
    rounding = _label(report, "Rounding the target to binary64")
    assert rounding["value"]["exact"] == 1 and rounding["value"]["binary64"] == 2
    assert rounding["value"]["exact_of_rounded_target"] == 2 and "counterexample" in rounding
    ties = _label(report, "At exact cut-locus ties")
    assert ties["value"] == {"ties": 6, "binary64_undercounts": 4, "tolerance_undercounts": 0}
    assert ties["counterexample"]["witness"]["binary64"] < ties["counterexample"]["witness"]["exact"]


def test_t023_heading_sensitivity(reports):
    report = reports["T023"]
    zeros = _label(report, "Zeros of the return distance")
    assert zeros["value"]["found"] == zeros["value"]["predicted"] == 18
    assert _label(report, "Near each closing heading")["value"]["max_relative_slope_error"] < 1e-9
    basins = _label(report, "Closure basins are widest")["value"]
    assert basins["discordant_pairs"] == 0
    assert sorted(abs(x) for x in basins["widest"][0]["winding"]) == [0, 1]
    widths = [b["measured_width"] for b in basins["widest"]]
    assert widths == sorted(widths, reverse=True)


# ---------------------------------------------------------------- curved routes
def test_route_helpers():
    a = {"length": 1.0, "heading": 0.1, "amplification": 2.0, "focus_margin": None}
    b = {"length": 2.0, "heading": 0.2, "amplification": 1.0, "focus_margin": 3.0}
    c = {"length": 3.0, "heading": 0.3, "amplification": 5.0, "focus_margin": None}
    assert routes.rankings([a, b, c])["by_focus_margin"] == [[0, 2], [1]]
    assert routes.same_route_set([a, b], [b, c]) == ([a], [c])
    assert routes.pareto_front([a, b, c]) == [0, 1]  # c is longer and amplifies more than a, with a tied margin
    # Two rays passing the same lift with a single miss-distance minimum give one Newton seed.
    step = 2 * math.pi / 8
    seeds = routes.newton_seeds([(0.0, 1.0, (0, 0), 0.2), (step, 1.1, (0, 0), 0.1), (2 * step, 1.2, (0, 0), 0.3),
                                 (5 * step, 4.0, (0, 0), 0.2)], 8)
    assert seeds == [(step, 1.1, (0, 0)), (5 * step, 4.0, (0, 0))]
    # A scipy disagreement on conjugate points is a failed exact check (finite), never an infinite error.
    basis = {"checks": []}
    ftt._attach_independent_routes(basis, {"j_head": 1e-8, "conjugate": 0.0, "endpoint": 1e-9,
                                           "presence_mismatches": 1, "failed": 0, "revision": "scipy x, sympy y"})
    record = finding("claim", "numerical", 0, basis)
    assert record["evidence_status"] == NE
    assert all(math.isfinite(check["observed"]) for check in basis["checks"])


def test_independent_disagreement_makes_t024_partial_not_blocked(tmp_path, monkeypatch):
    disagreement = {"j_head": 2e-8, "conjugate": 1e-7, "endpoint": 1e-9, "presence_mismatches": 1, "failed": 1,
                    "rows": [], "revision": "scipy x, sympy y"}
    monkeypatch.setattr(ftt, "_independent_routes", lambda *args: dict(disagreement))
    report = _run(["T024"], tmp_path)["T024"]
    assert report["state"] == "partial" and report["evidence_status"]["primary"] == NE
    assert _label(report, "Every fan-search route")["evidence_status"] == NE
    assert _label(report, "Rankings by length")["evidence_status"] == NV
    assert len(report["findings"]) == len(LABELS["T024"])


def test_sympy_field_matches_the_closed_form_batch_field():
    pytest.importorskip("sympy")
    import numpy as np

    for surface, u0 in ((ftt.TORUS, (0.3, 0.4)), (ftt.BUMP, (-0.7, 0.2)), (ftt.SADDLE, (0.5, -0.3))):
        rhs, initial = routes.derive_sympy_field(surface)
        for heading in (0.3, 2.0):
            y = np.array(initial(u0, heading))
            assert np.allclose(y, routes.initial_states(surface, u0, [heading])[0], atol=1e-14)
            assert np.allclose(rhs(0.0, y), routes.batch_field(surface)(y[None, :])[0], atol=1e-13)


def test_t024_t025_route_ranking_and_front(reports):
    report = reports["T024"]
    table = _label(report, "Every fan-search route")["value"]
    assert len(table) == 9
    assert table[0]["length"] == pytest.approx(5.8465, abs=1e-3)
    assert table[0]["amplification"] == pytest.approx(1.83415, rel=1e-5)
    assert table[0]["focus_margin"] == pytest.approx(1.24531, rel=1e-5)
    assert table[2]["passes_conjugate_point"] and table[2]["targeting_condition"] == pytest.approx(2.286, rel=1e-3)
    assert _label(report, "The torus route set is unchanged")["value"] == \
        {"headings": [1440, 2880], "routes": [9, 9], "differences": 0}
    ranks = _label(report, "Rankings by length")
    assert ranks["value"]["by_amplification"] == [2, 1, 0, 4, 3, 7, 6, 8, 5]
    assert ranks["value"]["by_focus_margin"] == [[3, 4, 5, 8], [7], [6], [0], [1], [2]]
    assert "counterexample" in ranks
    assert _label(report, "A flat torus has no conjugate points")["value"] == pytest.approx(1.0, abs=1e-12)
    front = _label(reports["T025"], "The three-objective front")["value"]
    assert front["front"] == [0, 1, 2, 3, 4] and front["front_members_past_conjugate_point"] == [2]
    assert front["fronts_2d"] == {"length_amplification": [0, 1, 2], "length_margin": [0, 3]}


def test_t032_counterexample_library(reports):
    report = reports["T032"]
    assert len([f for f in report["findings"] if "counterexample" in f]) == 5
    inner = _label(report, "Torus inner equator")
    assert inner["value"]["shortest"]["amplification"] == pytest.approx(math.sinh(2.5), rel=1e-6)
    assert inner["value"]["alternative"]["length"] == pytest.approx(7.4049, abs=1e-3)
    assert inner["value"]["alternative"]["amplification"] == pytest.approx(2.2424, abs=1e-3)
    outer = _label(report, "Torus outer equator")
    assert outer["value"]["shortest"]["focus_margin"] == pytest.approx(math.pi * math.sqrt(3) - 4.5, abs=1e-6)
    assert outer["value"]["alternative"]["focus_margin"] is None
    tie = _label(report, "Torus (0, 0) -> (2.2, 0)")["value"]
    assert abs(tie["lengths"][0] - tie["lengths"][1]) < 1e-9
    assert tie["headings"] == pytest.approx([-0.933, 0.933], abs=1e-3)
    bump = _label(report, "Gaussian bump")["value"]
    assert bump["alternative"]["focus_margin"] == pytest.approx(-1.997, abs=1e-3)
    assert bump["shortest"]["amplification"] - bump["alternative"]["amplification"] >= 0.2
    sphere = _label(report, "On the unit sphere")["value"]
    assert sphere["focus_margin"] == pytest.approx([0.1, 0.01, 0.001], abs=1e-6)
    assert sphere["targeting_condition"][-1] == pytest.approx(1 / math.sin(0.001), rel=1e-6)
    assert _label(report, "Search on the saddle")["value"] == {"routes": 1}
    density = _label(report, "Every T032 route search")["value"]
    assert all(v["differences"] == 0 for v in density.values()) and density["torus-tie"]["routes"] == [9, 9]


# ---------------------------------------------------------------- invariance and surfaces
def test_t026_modular_invariance(reports):
    report = reports["T026"]
    first = _label(report, "Length spectrum")
    assert first["value"]["failures"] == {"spectrum": 0, "area": 0, "systole": 0, "winding_rule": 0}
    assert first["value"]["matrices"] == 308 + 60
    assert _label(report, "Area-one float spectra")["value"] < 1e-8
    mirror = _label(report, "The length spectrum does not determine")
    assert mirror["value"]["same_spectrum"] and not mirror["value"]["same_canonical"]
    assert _label(report, "Transporting winding labels")["value"] == 6428
    assert _label(report, "A det-2 integer matrix")["value"]["area_sq_ratio"] == "4"


def test_t027_t029_surfaces_and_cones(reports):
    records = _label(reports["T027"], "L-shape and regular octagon")["value"]
    assert records["L-shape"]["genus"] == records["octagon"]["genus"] == 2
    assert records["L-shape"]["vertices"] == records["octagon"]["vertices"] == 1
    assert records["octagon"]["area"] == {"rational": "2", "sqrt_coefficient": "2", "radicand": 2}
    codes = _label(reports["T027"], "Pairings that are not")["value"]
    assert codes == {"octagon adjacent pairing": "GLUING_NOT_TRANSLATION", "pillowcase": "GLUING_NOT_TRANSLATION",
                     "mismatched edge lengths": "EDGE_LENGTH_MISMATCH"}
    cones = _label(reports["T029"], "Cone angles")["value"]
    assert cones["octagon"] == ["6"] and cones["H(1,1) origami"] == ["4", "4"] and cones["pillowcase"] == ["1"] * 4
    bonnet = _label(reports["T029"], "Gauss-Bonnet")["value"]
    assert {v["defect_over_pi"] for v in bonnet.values()} == {"0"}
    assert {k: v["chi"] for k, v in bonnet.items()} == ftt.DECLARED_CHI
    hexagon = _label(reports["T029"], "Polygon vertices need not")
    assert hexagon["counterexample"]["witness"]["vertex_classes"] == 2


def test_gauss_bonnet_check_detects_wrong_vertex_classes(tmp_path, monkeypatch):
    # All corners in separate classes: V - E + F changes consistently, so only the independent chi catches it.
    monkeypatch.setattr(surf.PolygonSurface, "vertex_classes",
                        lambda self: [[(i, j)] for i, p in enumerate(self.polygons) for j in range(len(p))])
    report = _run(["T029"], tmp_path)["T029"]
    assert report["state"] == "partial"
    assert _label(report, "Gauss-Bonnet")["evidence_status"] == NE


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
    assert _label(report, "All 96 tested")["value"] == {"closed": 90, "saddle_connections": 6, "undecided": 0}
    refusals = _label(report, "Trajectories hitting a cone point")["value"]
    assert refusals == {"exact_l_shape": "SADDLE_CONNECTION", "exact_octagon": "SADDLE_CONNECTION",
                        "float_octagon": "NEAR_VERTEX_WITHIN_TOLERANCE", "float_start": "START_NOT_INTERIOR"}
    assert _label(report, "Float octagon flow")["value"]["time_difference"] < 1e-12


# ---------------------------------------------------------------- discrete and perturbed metrics
def test_t030_grid_metrication(reports):
    report = reports["T030"]
    grid4 = _label(report, "4- and 8-neighbour")
    assert grid4["value"]["ratio4_45deg"] == pytest.approx([math.sqrt(2)] * 3, abs=1e-12)
    assert "counterexample" in grid4
    worst = _label(report, "Worst-direction")["value"]
    assert worst["8"] == pytest.approx(math.sqrt(4 - 2 * math.sqrt(2)) - 1, abs=1e-8)
    fmm = _label(report, "Fast marching converges")
    errors = fmm["value"]["max_relative_error"]
    assert errors["30"] > errors["60"] > errors["120"]
    assert {c["reference_kind"] for c in fmm["basis"]["checks"]} == {"analytic"}
    assert {a["path"].rsplit("/", 1)[-1] for a in report["generated_artifacts"]} >= \
        {"metrication.svg", "fast-marching-error.svg"}


def test_t031_route_switch(reports):
    report = reports["T031"]
    threshold = _label(report, "The shortest route switches")
    assert threshold["value"]["eps_star"] == "1/250"
    assert all(row["ratio"] == "4" for row in threshold["value"]["scaling"])
    jump = _label(report, "A metric perturbation just above")
    assert jump["value"]["heading_jump_deg"] == pytest.approx(126.87, abs=0.01)
    witness = jump["counterexample"]["witness"]
    assert witness["eps_before"] < 1 / 250 < witness["eps_after"] and witness["eps_after"] - witness["eps_before"] < 3e-4
    assert witness["translates_before"] != witness["translates_after"]
    assert ftt.flip_threshold(Fraction(1, 64))[0] == Fraction(1, 16)


def test_regeneration_is_within_tolerance(tmp_path):
    cheap = ["T021", "T022", "T027", "T029", "T031"]
    for name in ("a", "b"):
        for tid, report in _run(cheap, tmp_path / name).items():
            (tmp_path / name / "reports").mkdir(parents=True, exist_ok=True)
            (tmp_path / name / "reports" / f"{tid}.json").write_text(runner.dumps(report), encoding="utf-8")
    result = runner.compare(tmp_path / "a", tmp_path / "b")
    assert result["passed"], result["problems"]
