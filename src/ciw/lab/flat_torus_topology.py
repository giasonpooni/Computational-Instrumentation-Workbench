"""Flat torus and topology tasks T019-T032.

Scope: flat tori C / Lambda with Lambda = Z w1 + Z w2 (exact lattice algebra on
integer or rational Gram matrices), closed geodesics and their winding
classes, heading sensitivity and cut-locus degeneracy on flat tori, candidate
routes with focus margins on curved surfaces (torus of revolution, Gaussian
bump, saddle, sphere), polygonal translation surfaces with cone points, grid
approximations of geodesic distance, and route switching under metric
perturbation. Exact arithmetic (integers, Fractions, Q(sqrt d)) is used
wherever the geometry allows; floats carry declared tolerances.

Non-claims: every surface is a declared mathematical object in normalized
units. No finding concerns a physical part, vehicle, sensor, tool path or
workspace; route "safety", sensor closure performance and calibration adequacy
are recorded as unestablished physical or authority claims. Candidate route
sets come from a fan search and are not proven complete. Agreement with the
pinned Flat-Torus-Geodesic-Reference provider is independent implementation
agreement, not verification by another party.
"""
from __future__ import annotations

from fractions import Fraction
import importlib.util
import math
import sys

import numpy as np

from .. import __version__
from . import flat_torus_topology_discrete as grid
from . import flat_torus_topology_lattice as lat
from . import flat_torus_topology_provider as ftr
from . import flat_torus_topology_routes as routes
from . import flat_torus_topology_surfaces as surf
from . import jacobi, svg
from .evidence import finding
from .registry import task
from .surfaces import GaussianBump, Plane, Saddle, Torus

MODULE = "src/ciw/lab/flat_torus_topology.py"
LATTICE = "src/ciw/lab/flat_torus_topology_lattice.py"
SURFACES = "src/ciw/lab/flat_torus_topology_surfaces.py"
ROUTES = "src/ciw/lab/flat_torus_topology_routes.py"
DISCRETE = "src/ciw/lab/flat_torus_topology_discrete.py"
PROVIDER = "src/ciw/lab/flat_torus_topology_provider.py"
DOC = "docs/lab/FLAT_TORUS_TOPOLOGY.md"
TESTS = "tests/test_lab_flat_torus_topology.py"
PRODUCER = "ciw.lab.flat_torus_topology"

EXACT = {"abs": 0, "rel": 0}
TIGHT = {"abs": 1e-12, "rel": 1e-9}
FLOAT = {"abs": 1e-9, "rel": 1e-7}
ROUTE = {"abs": 1e-6, "rel": 1e-5}


def _tests(*names):
    """Regression node ids; every task is also covered by the section-wide completion test."""
    return tuple(f"{TESTS}::{name}" for name in names + ("test_every_task_is_registered_and_completes",))


# ---------------------------------------------------------------- shared helpers
def _check(reference, observed, tolerance=0.0, comparison="abs_le", kind="exact_arithmetic"):
    observed, tolerance = float(observed), float(tolerance)
    holds = {"abs_le": abs(observed) <= tolerance, "le": observed <= tolerance, "ge": observed >= tolerance}[comparison]
    return {"reference_kind": kind, "reference": reference, "observed": observed, "tolerance": tolerance,
            "comparison": comparison, "passed": holds}


def _refusal(reference, expected, observed):
    return {"reference_kind": "refusal", "reference": reference, "expected_refusal": expected,
            "observed_refusal": observed, "passed": observed == expected}


def _independent(check, checker, revision):
    return dict(check, producer={"implementation": PRODUCER, "revision": f"ciw {__version__}"},
                checker={"implementation": checker, "revision": revision})


def _version(module):
    if importlib.util.find_spec(module) is None:
        return None
    return str(__import__(module).__version__)


def _rng(seed):
    return np.random.Generator(np.random.PCG64(seed))


def _fields(hypothesis, model, inputs, observation, invariant, experiment, result, uncertainty, failures,
            assumptions, next_task, **extra):
    fields = {"hypothesis": hypothesis, "mathematical_model": model, "input_data": inputs,
              "observation_model": observation, "expected_invariant": invariant, "experiment": experiment,
              "numerical_result": result, "uncertainty": uncertainty, "failure_modes_checked": failures,
              "unresolved_assumptions": assumptions, "recommended_next_task": next_task}
    fields.update(extra)
    return fields


def _frac(x):
    return str(Fraction(x))


def _refusal_code(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except (lat.LatticeRefusal, surf.GluingRefusal, ftr.ProviderRefusal) as exc:
        return exc.code
    except surf.FlowTermination as exc:
        return exc.code
    return "NO_REFUSAL"


NO_PHYSICAL = "No physical observation: exact or floating-point computation on declared mathematical objects."


# ---------------------------------------------------------------- optional FTR provider
FTR_SHAPES = (complex(0.31, 1.07), complex(-0.2, 1.5), complex(0.05, 2.3), complex(0.44, 0.93))
FTR_WINDINGS = ((1, 0), (2, -3), (3, 1))
FTR_START = (Fraction("0.123"), Fraction("0.0456"))


def ftr_cases():
    """Deterministic comparison cases shared by T019, T020 and T026."""
    rng = _rng(2019)
    folds = []
    for tau0 in FTR_SHAPES:
        for winding in FTR_WINDINGS:
            M, word = lat.random_word(rng, 6, 2)
            folds.append({"tau0": tau0, "matrix": M, "word": word, "tau": lat.mobius(M, tau0), "winding": winding})
    lengths = [(tau, m, n) for tau in FTR_SHAPES[:2] for m in range(-4, 5) for n in range(-4, 5) if (m, n) != (0, 0)]
    traces = [(m, n) for m in range(0, 4) for n in range(-3, 4)
              if math.gcd(m, n) == 1 and (m > 0 or n > 0)]
    return {"folds": folds, "lengths": lengths, "traces": traces}


def ftr_request(cases):
    return {"folds": [{"tau": [c["tau"].real, c["tau"].imag], "winding": list(c["winding"])} for c in cases["folds"]],
            "lengths": [{"tau": [t.real, t.imag], "m": m, "n": n} for t, m, n in cases["lengths"]],
            "traces": [{"tau": [FTR_SHAPES[0].real, FTR_SHAPES[0].imag], "m": m, "n": n,
                        "start_cover": [float(FTR_START[0]), float(FTR_START[1])], "samples": 3}
                       for m, n in cases["traces"]],
            "areas": [[t.real, t.imag] for t in FTR_SHAPES]}


def _provider(ctx):
    """Run the pinned FTR provider once per run: None when unbound, a refusal record, or its data."""
    def compute():
        checkout = ctx.providers.get("ftr")
        interpreter = ctx.providers.get("ftr-python")
        if checkout is None:
            return None
        if interpreter is None:
            if sys.version_info < (3, 12):
                return {"refusal": "FTR_INTERPRETER_UNBOUND",
                        "message": "FTR requires Python >= 3.12; bind --provider ftr-python=<interpreter>"}
            interpreter = sys.executable
        cases = ftr_cases()
        try:
            run = ftr.run_ftr(checkout, interpreter, ftr_request(cases))
        except ftr.ProviderRefusal as exc:
            return {"refusal": exc.code, "message": str(exc)}
        run["cases"] = cases
        return run
    return ctx.memo("ftr-provider", compute)


def _provider_refused_finding(run):
    return finding("A bound FTR provider that does not match its pin or cannot execute is refused rather than compared",
                   "provenance", run["refusal"],
                   {"checks": [_refusal("flat_torus_topology_provider.run_ftr", run["refusal"], run["refusal"])]})


def _provider_note(run, fields, findings, changed_files):
    """Attach provider identity or the reason it did not run; return the resulting task state."""
    if run is None:
        fields["unresolved_assumptions"].append(
            "The optional pinned FTR comparison did not run (bind --provider ftr=<checkout> "
            "--provider ftr-python=<Python 3.12 interpreter>)")
        return "completed"
    if "refusal" in run:
        findings.append(_provider_refused_finding(run))
        fields["unresolved_assumptions"].append(f"FTR provider refused: {run['message']}")
        return "partial"
    from .runner import builtin_identity

    fields["provider_runtime_identity"] = {"provider": run["identity"], "ciw": builtin_identity(changed_files),
                                           "producer": PRODUCER}
    return "completed"


def fold_comparison(run):
    """Compare the provider's folds with float Gauss reduction and exact winding transport."""
    rows, tau_diff, matrix_mismatch, winding_mismatch, length_diff, recovered = [], 0.0, 0, 0, 0.0, 0.0
    for case, data in zip(run["cases"]["folds"], run["data"]["folds"]):
        mine_tau, M = lat.float_reduce(case["tau"])
        a, b, c, d = data["matrix"]
        theirs = ((d, b), (c, a))  # FTR g acts as (a tau + b) / (c tau + d); column-basis form
        negated = tuple(tuple(-x for x in row) for row in theirs)
        same = M == theirs
        if not (same or M == negated):
            matrix_mismatch += 1
        m, n = case["winding"]
        Minv = lat.inverse_sl2(M)
        mine_w = (Minv[0][0] * m + Minv[0][1] * n, Minv[1][0] * m + Minv[1][1] * n)
        sign = 1 if same else -1
        if (sign * mine_w[0], sign * mine_w[1]) != tuple(data["reduced_winding"]):
            winding_mismatch += 1
        before = math.sqrt(lat.quad(lat.area_one_form(case["tau"]), m, n))
        after = math.sqrt(lat.quad(lat.area_one_form(mine_tau), *mine_w))
        tau_diff = max(tau_diff, abs(mine_tau - complex(*data["reduced"])))
        recovered = max(recovered, abs(mine_tau - case["tau0"]))
        length_diff = max(length_diff, abs(before - data["length_pair"][0]) / before,
                          abs(after - data["length_pair"][1]) / after)
        rows.append({"tau_in": [case["tau"].real, case["tau"].imag], "word": case["word"],
                     "ciw_reduced": [mine_tau.real, mine_tau.imag], "ftr_reduced": data["reduced"],
                     "ciw_matrix": [list(r) for r in M], "ftr_matrix": data["matrix"], "ftr_word": data["word"],
                     "winding": list(case["winding"]), "ciw_reduced_winding": list(mine_w),
                     "ftr_reduced_winding": data["reduced_winding"], "ciw_lengths": [before, after],
                     "ftr_length_pair": data["length_pair"]})
    return {"rows": rows, "max_tau_difference": tau_diff, "matrix_mismatches": matrix_mismatch,
            "winding_mismatches": winding_mismatch, "max_length_relative_difference": length_diff,
            "max_recovery_error": recovered}


# ================================================================ T019
LATTICES = {"generic": (5, 2, 7), "vertical-boundary": (2, 1, 5), "arc-boundary": (3, 1, 3),
            "square": (1, 0, 1), "hexagonal": (2, 1, 2), "rectangular": (1, 0, 3)}
# |Stab| summed over the orbit points in the closed fundamental domain.
PREDICTED_REDUCED_BASES = {"generic": 2, "vertical-boundary": 4, "arc-boundary": 4, "square": 4,
                           "hexagonal": 12, "rectangular": 2}


def enumeration_study(bound=5, words=200, word_length=12):
    matrices = lat.sl2z_matrices(bound)
    identity = ((1, 0), (0, 1))
    per_lattice, failures = {}, {"inverse": 0, "area": 0, "reduction": 0, "reducer": 0, "automorphism": 0}
    rng = _rng(19)
    word_rows = []
    sampled = [lat.random_word(rng, word_length, 3) for _ in range(words)]
    for name, G in LATTICES.items():
        canonical, _, _ = lat.gauss_reduce(G)
        for M in matrices:
            if lat.matmul(M, lat.inverse_sl2(M)) != identity:
                failures["inverse"] += 1
            H = lat.transform(G, M)
            if lat.det_form(H) != lat.det_form(G):
                failures["area"] += 1
            reduced, R, _ = lat.gauss_reduce(H)
            if reduced != canonical:
                failures["reduction"] += 1
            if lat.transform(H, R) != reduced:
                failures["reducer"] += 1
        largest = 0
        for M, word in sampled:
            H = lat.transform(G, M)
            reduced, R, steps = lat.gauss_reduce(H)
            composite = lat.matmul(M, R)
            # B M R is reduced: M R must be an automorphism of the canonical form.
            if reduced != canonical or lat.transform(canonical, composite) != canonical:
                failures["automorphism"] += 1
            largest = max(largest, max(abs(x) for row in M for x in row))
            if name == "generic" and len(word_rows) < 12:
                word_rows.append({"word": word, "matrix": [list(r) for r in M], "gram": [str(x) for x in H],
                                  "reduced": [str(x) for x in reduced], "reducer": [list(r) for r in R],
                                  "s_steps": steps})
        per_lattice[name] = {"gram": [str(x) for x in G], "canonical": [str(x) for x in canonical],
                             "tau": [lat.tau_of(canonical).real, lat.tau_of(canonical).imag],
                             "reduced_bases": lat.reduced_basis_count(G),
                             "predicted_reduced_bases": PREDICTED_REDUCED_BASES[name],
                             "largest_word_entry": largest}
    return {"bound": bound, "matrices": len(matrices), "words": words, "word_length": word_length,
            "lattices": per_lattice, "failures": failures, "word_examples": word_rows}


def sympy_enumeration_check(bound=2):
    """sympy recomputes det, integer inverse, M^T G M and the reduced-form conditions."""
    import sympy

    mismatches, compared = 0, 0
    for G in LATTICES.values():
        Gs = sympy.Matrix([[G[0], G[1]], [G[1], G[2]]])
        for M in lat.sl2z_matrices(bound):
            compared += 1
            Ms = sympy.Matrix(M)
            inv = Ms.inv()
            H = Ms.T * Gs * Ms
            reduced, R, _ = lat.gauss_reduce(lat.transform(G, M))
            Rs = sympy.Matrix(R)
            C = Rs.T * H * Rs
            ok = (Ms.det() == 1 and all(x.is_integer for x in inv)
                  and (H[0, 0], H[0, 1], H[1, 1]) == lat.transform(G, M)
                  and (C[0, 0], C[0, 1], C[1, 1]) == reduced
                  and abs(2 * C[0, 1]) <= C[0, 0] <= C[1, 1])
            mismatches += 0 if ok else 1
    return {"compared": compared, "mismatches": mismatches}


@task("T019", changed_files=(MODULE, LATTICE, PROVIDER, DOC),
      regression_tests=_tests("test_t019_reduction_and_refusals",
                              "test_gauss_reduction_is_exact",
                              "test_ftr_refusal_makes_task_partial",
                              "test_ftr_provider_agreement"))
def enumerate_lattice_representatives(ctx):
    study = ctx.memo("t019-enumeration", enumeration_study)
    ctx.artifact_json("lattice-reduction.json", study)
    failures = study["failures"]
    total_failures = sum(failures.values())
    counts = {name: row["reduced_bases"] for name, row in study["lattices"].items()}
    count_mismatch = sum(counts[k] != PREDICTED_REDUCED_BASES[k] for k in counts)
    generic = LATTICES["generic"]
    codes = {"det 2 (index-2 sublattice)": _refusal_code(lat.transform, generic, ((2, 0), (0, 1))),
             "det -1 (orientation reversal)": _refusal_code(lat.transform, generic, ((1, 0), (0, -1))),
             "non-integer": _refusal_code(lat.transform, generic, ((Fraction(1, 2), 0), (0, 2)))}
    sub = lat.transform_any(generic, ((2, 0), (0, 1)))
    sub_canonical = lat.gauss_reduce(sub)[0]
    fields = _fields(
        "Every SL(2,Z) change of basis of a lattice reduces, by exact Gauss/Lagrange reduction, to one canonical "
        "Gram form; the number of reduced bases equals the stabilizer count predicted by the position of tau in "
        "the closed fundamental domain; non-unimodular or orientation-reversing matrices are refused.",
        "Lattice Lambda = Z w1 + Z w2 carried by its Gram matrix G = B^T B; basis change B' = B M acts as "
        "G' = M^T G M; canonical form |2b| <= a <= c with b >= 0 on the boundary; tau = (b + i sqrt(det G)) / a.",
        [f"Integer Gram matrices {LATTICES}",
         f"All {study['matrices']} SL(2,Z) matrices with entries in [-{study['bound']}, {study['bound']}]",
         f"{study['words']} seeded random S/T^k words of length {study['word_length']} (PCG64 seed 19)"],
        NO_PHYSICAL,
        "Canonical form, det G (squared area) and the automorphism group are invariant under SL(2,Z).",
        "Transform each lattice by every enumerated matrix and every random word, reduce exactly, compare with "
        "the canonical form; count closed-domain reduced bases; exercise refusals; optionally compare float "
        "folds with the pinned FTR provider.",
        f"{total_failures} failures over {len(LATTICES) * (study['matrices'] + study['words'])} exact reductions; "
        f"reduced-basis counts {counts} match predictions {PREDICTED_REDUCED_BASES}.",
        "Exact integer/rational arithmetic: no rounding. Provider comparisons are float with tolerance 1e-9.",
        ["non-unimodular basis change (det 2) refused", "orientation reversal (det -1) refused",
         "non-integer basis change refused", "boundary conventions (b >= 0 on |2b| = a or a = c)",
         "large-entry words (exact integers, no overflow)"],
        ["Automorphism counts are checked for six declared lattices, not proven for all lattices here "
         "(the general statement is the classical stabilizer theorem)."],
        "T026: test modular-reduction invariance of length spectrum, area and systole")
    findings = []
    basis = {"checks": [_check("exact reduction failures (inverse, area, canonical form, reducer, automorphism)",
                               total_failures)]}
    sympy_version = _version("sympy")
    if sympy_version:
        check = sympy_enumeration_check()
        basis["independent_check"] = _independent(
            _check(f"sympy det/inverse/M^T G M/reduced conditions over {check['compared']} cases",
                   check["mismatches"]), "sympy.Matrix", sympy_version)
    findings.append(finding(
        "Every enumerated SL(2,Z) basis and every random word reduces exactly to the same canonical Gram form",
        "mathematical", {"failures": failures, "reductions": len(LATTICES) * (study["matrices"] + study["words"])},
        basis, tolerance=EXACT))
    findings.append(finding(
        "Number of reduced bases equals the predicted stabilizer count (2 generic, 4 boundary or square, 12 hexagonal)",
        "mathematical", counts,
        {"derivation": "Stabilizers in SL(2,Z): +-I generically, order 4 at i, order 6 at exp(i pi/3); boundary "
                       "points have two representatives in the closed domain",
         "checks": [_check("count mismatches against prediction", count_mismatch)]}, tolerance=EXACT))
    findings.append(finding(
        "Integer matrices with det != 1 are refused as SL(2,Z) basis changes", "mathematical", codes,
        {"checks": [_refusal("lat.transform with det 2", "BASIS_CHANGE_NOT_UNIMODULAR",
                             codes["det 2 (index-2 sublattice)"]),
                    _refusal("lat.transform with det -1", "BASIS_CHANGE_REVERSES_ORIENTATION",
                             codes["det -1 (orientation reversal)"]),
                    _refusal("lat.transform with a non-integer entry", "BASIS_CHANGE_NOT_INTEGER", codes["non-integer"]),
                    _check("det of the det-2 image over det G (index-2 sublattice has 4x squared area)",
                           Fraction(lat.det_form(sub), lat.det_form(generic)) - 4)]},
        counterexample={"statement": "Any integer change of basis generates the same lattice",
                        "witness": {"gram": list(generic), "matrix": [[2, 0], [0, 1]],
                                    "image_canonical": [str(x) for x in sub_canonical],
                                    "canonical": [str(x) for x in lat.gauss_reduce(generic)[0]]}},
        tolerance=EXACT))
    run = _provider(ctx)
    if run is not None and "refusal" not in run:
        cmp = fold_comparison(run)
        ctx.artifact_json("ftr-folds.json", cmp)
        identity = run["identity"]
        findings.append(finding(
            "Float Gauss reduction agrees with the pinned FTR fold_to_fundamental_domain on reduced tau and the "
            "reducing SL(2,Z) matrix (up to -I)", "numerical",
            {"max_tau_difference": cmp["max_tau_difference"], "matrix_mismatches": cmp["matrix_mismatches"],
             "cases": len(cmp["rows"])},
            {"provider": ftr.provider_basis(identity),
             "checks": [_check("matrix mismatches up to -I", cmp["matrix_mismatches"]),
                        _check("reduction recovers the declared reduced tau", cmp["max_recovery_error"], 1e-9,
                               kind="invariant")],
             "independent_check": _independent(
                 _check("FTR reduced tau", cmp["max_tau_difference"], 1e-9, kind="high_precision"),
                 ftr.checker_identity(identity)["implementation"], identity["revision"])},
            tolerance={"abs": 1e-9, "rel": 0}))
    state = _provider_note(run, fields, findings, (MODULE, LATTICE, PROVIDER))
    return {"state": state, "fields": fields, "findings": findings}


# ================================================================ T020
WINDING_FORM = (5, 2, 7)
WINDING_START = (Fraction(1, 7), Fraction(2, 11))
PHI = (1 + math.sqrt(5)) / 2


def winding_study(span=6, radius=20):
    rows, failures = [], 0
    for m in range(-span, span + 1):
        for n in range(-span, span + 1):
            if (m, n) == (0, 0):
                continue
            info = lat.classify_winding(m, n)
            flow = lat.trace_lattice_flow(m, n, WINDING_START)
            g = info["gcd"]
            pm, pn = info["primitive_class"]
            ok = (flow["returned"] and flow["period"] == Fraction(1, g)
                  and flow["crossings"] == (abs(m) + abs(n)) // g
                  and lat.quad(WINDING_FORM, m, n) == g * g * lat.quad(WINDING_FORM, pm, pn))
            failures += 0 if ok else 1
            rows.append({"winding": [m, n], "gcd": g, "primitive": info["primitive"],
                         "length_sq": lat.quad(WINDING_FORM, m, n), "first_return_time": _frac(flow["period"]),
                         "crossings_per_period": flow["crossings"]})
    primitive = [(m, n) for m in range(0, 4) for n in range(-3, 4) if math.gcd(m, n) == 1 and (m > 0 or n > 0)]
    intersections, wrong = [], 0
    for i, v in enumerate(primitive):
        for w in primitive[i + 1:]:
            count = lat.intersection_count(v, w)
            expected = abs(v[0] * w[1] - v[1] * w[0])
            wrong += count != expected
            intersections.append({"a": list(v), "b": list(w), "count": count, "det": expected})
    # Lattice-point count in a disk against the cell-covering bound pi (2 R D + D^2) / A.
    vectors = lat.lattice_vectors(WINDING_FORM, radius * radius)
    area = math.sqrt(lat.det_form(WINDING_FORM))
    diameter = math.sqrt(max(lat.quad(WINDING_FORM, 1, 1), lat.quad(WINDING_FORM, 1, -1)))
    total = len(vectors) + 1
    prim = sum(1 for _, m, n in vectors if math.gcd(m, n) == 1)
    count = {"radius": radius, "lattice_points_with_origin": total, "primitive": prim,
             "gauss_estimate": math.pi * radius ** 2 / area,
             "bound": math.pi * (2 * radius * diameter + diameter ** 2) / area,
             "primitive_fraction": prim / (total - 1), "six_over_pi_sq": 6 / math.pi ** 2}
    return {"rows": rows, "failures": failures, "intersections": intersections, "intersection_failures": wrong,
            "count": count}


def golden_returns(k_max=22):
    """Gaps ||F_k phi|| at Fibonacci returns (float), against phi^-k (Binet) and mpmath."""
    fib = [0, 1]
    while len(fib) <= k_max + 1:
        fib.append(fib[-1] + fib[-2])
    rows = []
    for k in range(2, k_max + 1):
        q = fib[k]
        x = q * PHI
        gap = abs(x - round(x))
        rows.append({"k": k, "q": q, "gap": gap, "analytic": PHI ** (-k), "q_times_gap": q * gap})
    return rows


def mpmath_gaps(rows, dps=50):
    import mpmath

    with mpmath.workdps(dps):
        phi = (1 + mpmath.sqrt(5)) / 2
        return [float(abs(r["q"] * phi - mpmath.nint(r["q"] * phi))) for r in rows]


def length_comparison(run):
    worst, rows = 0.0, []
    for (tau, m, n), theirs in zip(run["cases"]["lengths"], run["data"]["lengths"]):
        mine = math.sqrt(lat.quad(lat.area_one_form(tau), m, n))
        worst = max(worst, abs(mine - theirs) / mine)
        rows.append({"tau": [tau.real, tau.imag], "winding": [m, n], "ciw": mine, "ftr": theirs})
    crossing_mismatch = 0
    for (m, n), trace in zip(run["cases"]["traces"], run["data"]["traces"]):
        mine = lat.trace_lattice_flow(m, n, FTR_START)["crossings"]
        crossing_mismatch += (mine != trace["crossings"]) or not trace["closed"]
    return {"rows": rows, "max_relative_difference": worst, "crossing_mismatches": crossing_mismatch,
            "traces": len(run["cases"]["traces"]), "area_deviation": max(abs(a - 1.0) for a in run["data"]["areas"])}


@task("T020", changed_files=(MODULE, LATTICE, PROVIDER, DOC),
      regression_tests=_tests("test_t020_winding_classification",
                              "test_winding_flow_and_intersections",
                              "test_ftr_provider_agreement"))
def classify_by_winding(ctx):
    study = ctx.memo("t020-windings", winding_study)
    gaps = golden_returns()
    ctx.artifact_json("windings.json", study)
    ctx.artifact_json("golden-returns.json", gaps)
    ctx.artifact_text("golden-return-gaps.svg", svg.line_plot(
        [("binary64 ||q phi||", [r["q"] for r in gaps], [r["gap"] for r in gaps]),
         ("phi^-k (Binet)", [r["q"] for r in gaps], [r["analytic"] for r in gaps])],
        title="Returns of the golden-slope geodesic", xlabel="Fibonacci return index q", ylabel="closing gap",
        logx=True, logy=True))
    count = study["count"]
    float_rel = max(abs(r["gap"] - r["analytic"]) / r["analytic"] for r in gaps)
    limit = abs(gaps[-1]["q_times_gap"] - 1 / math.sqrt(5))
    phi_fraction = Fraction(PHI)
    d = phi_fraction.denominator
    power_of_two = d & (d - 1) == 0 and d <= 2 ** 52
    log2_denominator = d.bit_length() - 1
    fields = _fields(
        "Closed geodesics of a flat torus are exactly the straight lines in nonzero lattice directions: primitive "
        "(m, n) gives a closed geodesic traversed once with length |m w1 + n w2|, (k m', k n') its k-fold cover, and "
        "irrational directions never close although their return gaps shrink.",
        "Lattice-coordinate flow alpha = alpha0 + m t, beta = beta0 + n t on R^2 / Z^2 with metric Q(m, n) = "
        "a m^2 + 2 b m n + c n^2; first return at t = 1/gcd(m, n); intersection number |det(v, w)|; "
        "Binet: ||F_k phi|| = phi^-k.",
        [f"Gram form {WINDING_FORM} (area sqrt 31)", f"start (alpha0, beta0) = {tuple(map(str, WINDING_START))}",
         "all windings with |m|, |n| <= 6", "golden slope phi in lattice coordinates, Fibonacci returns k <= 22"],
        NO_PHYSICAL,
        "Closure time 1/gcd, crossings (|m| + |n|)/gcd per period, length scaling k * l(primitive), and "
        "intersection count |det| hold exactly for every tested class.",
        "Exact event-driven flow per winding; exact intersection counting for primitive pairs; lattice-point "
        "count against the covering bound; binary64 and 50-digit golden return gaps.",
        f"{study['failures']} winding failures over {len(study['rows'])} classes; {study['intersection_failures']} "
        f"intersection failures over {len(study['intersections'])} pairs; {count['lattice_points_with_origin']} "
        f"lattice points within R = {count['radius']} (Gauss estimate {count['gauss_estimate']:.2f}); primitive "
        f"fraction {count['primitive_fraction']:.4f} (asymptotic 6/pi^2 = {count['six_over_pi_sq']:.4f}); golden "
        f"q * gap -> 1/sqrt 5 within {limit:.2e}.",
        "Winding results exact. Golden gaps: binary64 relative error <= 1e-6 for q <= 17711 (cancellation grows "
        "like q * eps / phi^-k).",
        ["non-primitive windings (multiple covers)", "zero winding refused", "corner passes counted on both axes",
         "binary64 cancellation in ||q phi||", "binary64 slopes are rational (never truly irrational)"],
        ["Non-closure of irrational directions is a theorem (irrationality), not something a finite computation "
         "establishes; the numerical gaps only illustrate it.",
         "Primitive fraction approaches 6/pi^2 only asymptotically; no rate is claimed."],
        "T023: compute heading sensitivity of closure across all starting headings")
    findings = [
        finding("Every winding with |m|, |n| <= 6 closes at t = 1/gcd after (|m| + |n|)/gcd edge crossings, with "
                "length gcd times that of its primitive class",
                "mathematical", {"classes": len(study["rows"]), "failures": study["failures"]},
                {"checks": [_check("closure time, crossings and length scaling mismatches", study["failures"])]},
                tolerance=EXACT),
        finding("For 120 pairs of primitive classes the transverse intersections number |det(v, w)|", "mathematical",
                {"pairs": len(study["intersections"]), "failures": study["intersection_failures"]},
                {"checks": [_check("intersection count mismatches (exact rational solve)",
                                   study["intersection_failures"])]}, tolerance=EXACT),
        finding("Closed-geodesic count within radius R obeys the lattice covering bound |N - pi R^2 / A| <= "
                "pi (2 R D + D^2) / A", "mathematical",
                {k: count[k] for k in ("radius", "lattice_points_with_origin", "primitive")},
                {"checks": [_check("|N - pi R^2 / A| minus the covering bound",
                                   abs(count["lattice_points_with_origin"] - count["gauss_estimate"]) - count["bound"],
                                   0.0, "le", kind="invariant")]}, tolerance=EXACT),
    ]
    golden_basis = {"derivation": "Binet: F_{k+1} - phi F_k = (-1/phi)^k, so ||F_k phi|| = phi^-k for k >= 2",
                    "checks": [_check("binary64 gap relative to phi^-k", float_rel, 1e-6, kind="analytic"),
                               _check("smallest gap is positive (no closure observed)",
                                      min(r["gap"] for r in gaps), 0.0, "ge", kind="invariant")]}
    mp_version = _version("mpmath")
    if mp_version:
        mp = mpmath_gaps(gaps)
        golden_basis["independent_check"] = _independent(
            _check("mpmath 50-digit ||q phi||", max(abs(r["gap"] - g) / g for r, g in zip(gaps, mp)), 1e-6,
                   kind="high_precision"), "mpmath", mp_version)
    findings.append(finding(
        "The golden-slope geodesic has positive return gaps phi^-k at Fibonacci returns, with q * gap -> 1/sqrt 5",
        "numerical", {"q": [r["q"] for r in gaps], "gap": [r["gap"] for r in gaps]}, golden_basis,
        tolerance={"abs": 1e-15, "rel": 1e-6}))
    findings.append(finding(
        "A binary64 heading slope is rational, so a float simulation cannot represent a non-closing direction",
        "numerical", {"slope_denominator": phi_fraction.denominator, "log2_denominator": log2_denominator},
        {"checks": [_check("denominator of Fraction(float(phi)) is a power of two at most 2^52 (0 = yes)",
                           0 if power_of_two else 1)]},
        counterexample={"statement": "A floating-point geodesic direction can be irrational (non-closing)",
                        "witness": {"float_phi": PHI,
                                    "exact_fraction": f"{phi_fraction.numerator}/2^{log2_denominator}",
                                    "closes_after_alpha_turns": phi_fraction.denominator}},
        tolerance=EXACT))
    run = _provider(ctx)
    if run is not None and "refusal" not in run:
        cmp = length_comparison(run)
        ctx.artifact_json("ftr-lengths.json", cmp)
        identity = run["identity"]
        findings.append(finding(
            "Closed-geodesic lengths |m w1 + n w2|, edge-crossing counts and unit area agree with the pinned FTR "
            "loop_length, trace_closed_geodesic and normalized_lattice", "numerical",
            {"lengths": len(cmp["rows"]), "max_relative_difference": cmp["max_relative_difference"],
             "crossing_mismatches": cmp["crossing_mismatches"]},
            {"provider": ftr.provider_basis(identity),
             "checks": [_check("crossing-count mismatches or unclosed traces", cmp["crossing_mismatches"]),
                        _check("FTR area-one normalization", cmp["area_deviation"], 1e-12, kind="invariant")],
             "independent_check": _independent(
                 _check("FTR loop_length", cmp["max_relative_difference"], 1e-12, kind="high_precision"),
                 ftr.checker_identity(identity)["implementation"], identity["revision"])},
            tolerance={"abs": 1e-12, "rel": 0}))
    state = _provider_note(run, fields, findings, (MODULE, LATTICE, PROVIDER))
    return {"state": state, "fields": fields, "findings": findings}


# ================================================================ T021
FLAT_BASIS = ((Fraction(1), Fraction(0)), (Fraction(3, 10), Fraction(11, 10)))
FLAT_P = (Fraction(1, 10), Fraction(1, 5))
FLAT_Q = (Fraction(7, 10), Fraction(11, 20))


def flat_translates(max_length=2.6):
    """Exact translates q - p + lambda (lattice coordinates) with their Euclidean vectors."""
    (w1x, w1y), (w2x, w2y) = FLAT_BASIS
    form = lat.gram_of_basis(*FLAT_BASIS)
    det = w1x * w2y - w2x * w1y
    dx, dy = FLAT_Q[0] - FLAT_P[0], FLAT_Q[1] - FLAT_P[1]
    z = ((w2y * dx - w2x * dy) / det, (-w1y * dx + w1x * dy) / det)
    rows = []
    for m in range(-4, 5):
        for n in range(-4, 5):
            c = (z[0] + m, z[1] + n)
            q2 = lat.quad(form, *c)
            if q2 <= Fraction(max_length) ** 2 * 1:
                vx, vy = c[0] * w1x + c[1] * w2x, c[0] * w1y + c[1] * w2y
                rows.append({"lambda": [m, n], "length_sq": q2, "vector": (float(vx), float(vy))})
    rows.sort(key=lambda r: (r["length_sq"], r["lambda"]))
    return form, z, rows


def flat_route_study():
    _, z, rows = flat_translates()
    plane = Plane()
    p = np.array([float(FLAT_P[0]), float(FLAT_P[1])])
    out = []
    for r in rows:
        vx, vy = r["vector"]
        length = math.hypot(vx, vy)
        heading = math.atan2(vy, vx)
        tr = jacobi.transfer(plane, p, heading, length, steps=16)
        end = tr.states[-1]
        out.append({"lambda": r["lambda"], "length_sq": _frac(r["length_sq"]), "length": length, "heading": heading,
                    "j_head": float(end[6]), "j_lat": float(end[4]),
                    "endpoint_error": float(math.hypot(end[0] - p[0] - vx, end[1] - p[1] - vy)),
                    "exact_length_sq": r["length_sq"]})
    discordant = 0
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            a, b = out[i], out[j]
            if a["exact_length_sq"] != b["exact_length_sq"] and (a["exact_length_sq"] < b["exact_length_sq"]) != \
                    (abs(a["j_head"]) < abs(b["j_head"])):
                discordant += 1
    for r in out:
        r.pop("exact_length_sq")
    return {"z": [_frac(z[0]), _frac(z[1])], "routes": out, "discordant_pairs": discordant}


@task("T021", changed_files=(MODULE, LATTICE, DOC),
      regression_tests=_tests("test_t021_shortest_is_least_sensitive", "test_regeneration_is_within_tolerance"))
def shortest_versus_least_sensitive(ctx):
    study = flat_route_study()
    ctx.artifact_json("flat-routes.json", study)
    rs = study["routes"]
    j_err = max(abs(r["j_head"] - r["length"]) / r["length"] for r in rs)
    lat_err = max(abs(r["j_lat"] - 1.0) for r in rs)
    end_err = max(r["endpoint_error"] for r in rs)
    fields = _fields(
        "On a flat torus the heading amplification of a route equals its length, so among the geodesics joining "
        "two points the shortest is exactly the least heading-sensitive.",
        "Geodesics p -> q on C / Lambda lift to segments p -> q + lambda; with K = 0 the Jacobi equation j'' = 0 "
        "gives j_head(s) = s and j_lat(s) = 1 along every route.",
        [f"basis w1 = (1, 0), w2 = (3/10, 11/10); p = {tuple(map(str, FLAT_P))}, q = {tuple(map(str, FLAT_Q))}",
         f"{len(rs)} lattice translates with length <= 2.6"],
        NO_PHYSICAL,
        "j_head(L) = L and j_lat(L) = 1 for every route; the length order and the amplification order coincide.",
        "Enumerate translates exactly; integrate each route with ciw.lab.jacobi.transfer on the plane chart; "
        "compare j_head(L) with L and count discordant pairs between the two orderings.",
        f"{len(rs)} routes, shortest length {rs[0]['length']:.6f}; max |j_head/L - 1| = {j_err:.2e}; "
        f"discordant pairs {study['discordant_pairs']}.",
        "RK4 is exact for the linear flat system up to rounding (<= 1e-12 relative).",
        ["ties in exact length excluded from the discordance count", "endpoint reached (lift consistency)",
         "lateral column constant (j_lat = 1)"],
        ["The equivalence is special to zero curvature; T032 records curved-surface counterexamples."],
        "T022: detect degenerate (tied) shortest representatives")
    findings = [
        finding("On the test flat torus every route's heading amplification equals its length (ciw.lab.jacobi), so "
                "the length and amplification orders coincide",
                "numerical", {"routes": len(rs), "max_relative_j_head_error": j_err,
                              "discordant_pairs": study["discordant_pairs"]},
                {"derivation": "K = 0: j'' = 0 with j(0) = 0, j'(0) = 1 gives j_head(s) = s "
                               "(docs/lab/FLAT_TORUS_TOPOLOGY.md)",
                 "checks": [_check("|j_head(L)/L - 1| from ciw.lab.jacobi", j_err, 1e-12, kind="analytic"),
                            _check("|j_lat(L) - 1|", lat_err, 1e-12, kind="analytic"),
                            _check("endpoint equals p + translate", end_err, 1e-12, kind="invariant"),
                            _check("discordant length/amplification pairs", study["discordant_pairs"])]},
                tolerance={"abs": 1e-12, "rel": 0}),
        finding("For every flat torus and every pair of points the least-sensitive geodesic is a shortest one",
                "mathematical", True,
                {"derivation": "j_head(s) = s is strictly increasing, so ordering routes by |j_head(L)| equals "
                               "ordering by L (ties included)"}, tolerance=EXACT),
        finding("With nonzero curvature the amplification |j_head(s)| is not monotone in s (e.g. sin s on the unit "
                "sphere), so the equivalence can fail", "mathematical", True,
                {"derivation": "Constant K > 0: j_head(s) = sin(sqrt K s)/sqrt K; K < 0: sinh(sqrt(-K) s)/sqrt(-K) "
                               "(ciw.lab.jacobi.constant_curvature); witnesses in T032"}, tolerance=EXACT),
        finding("The shortest route on a physical flat workpiece is the safest route to execute", "machine_safety",
                None, {}),
    ]
    return {"state": "completed", "fields": fields, "findings": findings}


# ================================================================ T022
SQUARE = (1, 0, 1)
HALF_PERIODS = {"(1/2, 0)": (Fraction(1, 2), 0), "(0, 1/2)": (0, Fraction(1, 2)),
                "(1/2, 1/2)": (Fraction(1, 2), Fraction(1, 2)), "(1/3, 1/5)": (Fraction(1, 3), Fraction(1, 5))}
EXPECTED_HALF = {"(1/2, 0)": 2, "(0, 1/2)": 2, "(1/2, 1/2)": 4, "(1/3, 1/5)": 1}


def degeneracy_study(n_grid=12, rel_tol=1e-9):
    half = {k: len(lat.nearest_translates(SQUARE, z)[1]) for k, z in HALF_PERIODS.items()}
    census = {}
    for i in range(n_grid):
        for j in range(n_grid):
            k = len(lat.nearest_translates(SQUARE, (Fraction(i, n_grid), Fraction(j, n_grid)))[1])
            census[k] = census.get(k, 0) + 1
    vertices = {}
    for name, G in LATTICES.items():
        v = lat.cut_locus_vertices(G)
        vertices[name] = {"vertices": [[_frac(z[0]), _frac(z[1]), k] for z, k in v],
                          "sum_k_minus_2": sum(k - 2 for _, k in v)}
    tiny = Fraction(1, 2 ** 60)
    z = (Fraction(1, 2) + tiny, Fraction(1, 4))
    exact_mult = len(lat.nearest_translates(SQUARE, z)[1])
    float_mult = lat.float_multiplicity(SQUARE, z, rel_tol=0.0)
    band = []
    false_unique = 0
    for eps in (Fraction(1, 10 ** 6), Fraction(1, 10 ** 8), Fraction(1, 10 ** 10), Fraction(1, 10 ** 12), tiny, 0):
        zz = (Fraction(1, 2) + eps, Fraction(1, 4))
        e = len(lat.nearest_translates(SQUARE, zz)[1])
        f = lat.float_multiplicity(SQUARE, zz, rel_tol=rel_tol)
        false_unique += e > 1 and f == 1
        band.append({"epsilon": float(eps), "exact_multiplicity": e, "tolerance_multiplicity": f,
                     "verdict": "tie" if e > 1 else ("ambiguous within tolerance" if f > 1 else "unique")})
    return {"half_periods": half, "census": {str(k): v for k, v in sorted(census.items())}, "grid": n_grid,
            "cut_locus": vertices, "float_counterexample": {"epsilon": "2^-60", "exact": exact_mult,
                                                            "binary64": float_mult},
            "tolerance_band": band, "rel_tol": rel_tol, "false_unique": false_unique}


@task("T022", changed_files=(MODULE, LATTICE, DOC),
      regression_tests=_tests("test_t022_degenerate_representatives", "test_regeneration_is_within_tolerance"))
def degenerate_shortest_representatives(ctx):
    study = degeneracy_study()
    ctx.artifact_json("degeneracy.json", study)
    n = study["grid"]
    predicted_census = {"1": n * n - 2 * (n - 1) - 1, "2": 2 * (n - 1), "4": 1}
    half_mismatch = sum(study["half_periods"][k] != EXPECTED_HALF[k] for k in EXPECTED_HALF)
    census_mismatch = sum(abs(study["census"].get(k, 0) - v) for k, v in predicted_census.items())
    vertex_mismatch = sum(v["sum_k_minus_2"] != 2 for v in study["cut_locus"].values())
    fc = study["float_counterexample"]
    fields = _fields(
        "Shortest representatives are tied exactly on the cut locus (the Voronoi boundary of the lattice): "
        "multiplicity 2 on edges and 3 or 4 at vertices, with sum over vertices of (k - 2) = 2 for every flat "
        "torus; float comparison cannot decide ties below its resolution.",
        "Squared lengths Q(z + lambda) are exact rationals for rational z and integer G; the cut locus of a point "
        "is a graph with V vertices of multiplicity k_v and E = sum k_v / 2 edges, and V - E + 1 = chi = 0.",
        [f"square torus G = {SQUARE}; lattices {LATTICES}", f"{n} x {n} rational grid of targets",
         "near-tie target (1/2 + eps, 1/4), eps down to 2^-60"],
        NO_PHYSICAL,
        "Multiplicities 2 (half periods), 4 (square centre), sum (k_v - 2) = 2 on every tested lattice.",
        "Exact nearest-translate search, grid census, exact circumcenter enumeration of cut-locus vertices, and "
        "binary64 detection with and without a declared relative tolerance.",
        f"half periods {study['half_periods']}; census {study['census']} (predicted {predicted_census}); "
        f"cut-locus vertices {{name: sum(k-2)}} = { {k: v['sum_k_minus_2'] for k, v in study['cut_locus'].items()} }; "
        f"eps = 2^-60: exact multiplicity {fc['exact']}, binary64 {fc['binary64']}.",
        "Exact for rational inputs; the tolerance-aware float detector is conservative by construction.",
        ["exact ties at half periods", "Voronoi vertices of generic, boundary, square, hexagonal and rectangular "
         "lattices", "binary64 tie at eps below resolution", "tolerance detector never reports a false unique"],
        ["Targets with irrational coordinates are only handled in float with a declared tolerance."],
        "T031: study route changes under small metric perturbations near these ties")
    findings = [
        finding("Square-torus half-period targets have exactly 2 (edge) or 4 (centre) shortest representatives",
                "mathematical", study["half_periods"],
                {"checks": [_check("mismatches against predicted multiplicities", half_mismatch)]}, tolerance=EXACT),
        finding("Multiplicity census on the 12 x 12 rational grid equals the predicted cut-locus counts",
                "mathematical", study["census"],
                {"checks": [_check("census deviation from {1: n^2 - 2n + 1, 2: 2(n - 1), 4: 1}", census_mismatch)]},
                tolerance=EXACT),
        finding("On six lattices the cut-locus vertices satisfy sum over vertices of (k_v - 2) = 2 (Euler "
                "characteristic 0)",
                "mathematical", {k: v["sum_k_minus_2"] for k, v in study["cut_locus"].items()},
                {"derivation": "E = sum k_v / 2 and V - E + F = 0 with F = 1 give sum (k_v - 2) = 2",
                 "checks": [_check("lattices violating the identity", vertex_mismatch)]}, tolerance=EXACT),
        finding("Binary64 comparison reports a tie where exact arithmetic finds a unique shortest representative",
                "numerical", fc,
                {"checks": [_check("binary64 multiplicity minus exact multiplicity (expect 1)",
                                   fc["binary64"] - fc["exact"] - 1),
                            _check("tolerance detector false-unique verdicts", study["false_unique"])]},
                counterexample={"statement": "Floating-point distance comparison decides the shortest representative",
                                "witness": {"target": "(1/2 + 2^-60, 1/4)", "exact_multiplicity": fc["exact"],
                                            "binary64_multiplicity": fc["binary64"]}},
                tolerance=EXACT),
    ]
    return {"state": "completed", "fields": fields, "findings": findings}


# ================================================================ T023
HEADING_TAU = complex(0.31, 1.07)
HEADING_LENGTH = 3.0
HEADINGS = 7200
S_MIN = 0.05


def heading_study(epsilon=0.02, delta=1e-4):
    thetas = np.arange(HEADINGS) * (2 * math.pi / HEADINGS)
    r, index, vectors = lat.return_distance(HEADING_TAU, thetas, HEADING_LENGTH, S_MIN)
    form = lat.area_one_form(HEADING_TAU)
    w1 = complex(1 / math.sqrt(HEADING_TAU.imag), 0.0)
    w2 = HEADING_TAU / math.sqrt(HEADING_TAU.imag)
    predicted = {(m, n) for q, m, n in vectors if math.gcd(m, n) == 1 and q <= HEADING_LENGTH ** 2
                 and q >= S_MIN ** 2}
    zeros, positive = {}, 0
    for i in range(HEADINGS):
        if r[i] <= r[i - 1] and r[i] < r[(i + 1) % HEADINGS]:
            _, m, n = vectors[index[i]]
            v = m * w1 + n * w2
            exact_theta = np.array([math.atan2(v.imag, v.real) % (2 * math.pi)])
            refined = float(lat.return_distance(HEADING_TAU, exact_theta, HEADING_LENGTH, S_MIN)[0][0])
            if refined <= 1e-12 and math.gcd(m, n) == 1:
                zeros[(m, n)] = {"theta": float(exact_theta[0]), "length": abs(v), "refined": refined}
            else:
                positive += 1
    slope_error, basins, width_error = 0.0, [], 0.0
    step = 2 * math.pi / HEADINGS
    for (m, n), z in zeros.items():
        probe = np.array([z["theta"] + delta, z["theta"] - delta])
        values = lat.return_distance(HEADING_TAU, probe, HEADING_LENGTH, S_MIN)[0]
        slope_error = max(slope_error, float(np.max(np.abs(values / math.sin(delta) - z["length"]))) / z["length"])
        analytic_width = 2 * math.asin(min(1.0, epsilon / z["length"]))
        # Contiguous run of grid headings with r < eps around theta_v (neighbouring basins excluded).
        centre = int(round(z["theta"] / step)) % HEADINGS
        run = 0
        for direction in (1, -1):
            k = centre if direction == 1 else (centre - 1) % HEADINGS
            while r[k] < epsilon and run < HEADINGS:
                run += 1
                k = (k + direction) % HEADINGS
        measured = run * step
        width_error = max(width_error, abs(measured - analytic_width))
        basins.append({"winding": [m, n], "length": z["length"], "theta": z["theta"], "slope": z["length"],
                       "basin_width": analytic_width, "measured_width": measured})
    basins.sort(key=lambda b: (-b["basin_width"], b["winding"]))
    discordant = sum(1 for i in range(len(basins)) for j in range(i + 1, len(basins))
                     if basins[i]["length"] > basins[j]["length"] + 1e-12)
    sample = [{"theta": float(thetas[i]), "return_distance": float(r[i])} for i in range(0, HEADINGS // 2, 10)]
    return {"predicted": sorted([list(v) for v in predicted]), "found": sorted([list(v) for v in zeros]),
            "missing": sorted([list(v) for v in predicted - set(zeros)]),
            "extra": sorted([list(v) for v in set(zeros) - predicted]), "positive_minima": positive,
            "slope_relative_error": slope_error, "basins": basins, "width_error": width_error, "step": step,
            "epsilon": epsilon, "discordant": discordant, "sample": sample}


@task("T023", changed_files=(MODULE, LATTICE, DOC),
      regression_tests=_tests("test_t023_heading_sensitivity"))
def sensitivity_across_headings(ctx):
    study = heading_study()
    ctx.artifact_json("heading-sensitivity.json", study)
    ctx.artifact_text("return-distance.svg", svg.line_plot(
        [("r(theta; L = 3)", [s["theta"] for s in study["sample"]], [s["return_distance"] for s in study["sample"]])],
        title="Closest return to the start within length 3", xlabel="heading theta (rad)",
        ylabel="return distance", markers=False))
    top = study["basins"][:4]
    fields = _fields(
        "The return distance r(theta; L) vanishes exactly at headings of primitive lattice vectors with |v| <= L, "
        "each zero is a V of slope |v| (closing error per unit heading error = loop length), so low-order "
        "(short) closed geodesics have the widest closure basins.",
        "r(theta; L) = min over lattice points lambda != 0 of dist(lambda, {s u(theta): s_min <= s <= L}); near "
        "theta_v = arg v, r = |v| sin|theta - theta_v|, basin {r < eps} has width 2 arcsin(eps/|v|).",
        [f"area-one lattice tau = {HEADING_TAU}", f"{HEADINGS} headings on [0, 2 pi)",
         f"L = {HEADING_LENGTH}, s_min = {S_MIN}, eps = {study['epsilon']}"],
        NO_PHYSICAL,
        "Zero set = primitive directions with |v| <= L; V-slope = |v|; basin width decreasing in |v|.",
        "Vectorized closed-form segment-to-lattice distance on the heading grid; grid local minima refined at "
        "the exact direction arg v; slopes probed at +-1e-4 rad; basins measured on the grid.",
        f"{len(study['found'])} zero minima found, {len(study['predicted'])} predicted, missing {study['missing']}, "
        f"extra {study['extra']}; {study['positive_minima']} positive (near-miss) minima; slope relative error "
        f"{study['slope_relative_error']:.2e}; widest basins {[b['winding'] for b in top]}.",
        f"Grid spacing {study['step']:.2e} rad bounds measured basin widths (max deviation "
        f"{study['width_error']:.2e} rad); refined zeros are exact to rounding.",
        ["near-miss minima (segment endpoints) separated from true closures",
         "non-primitive multiples on the same ray", "grid resolution against the narrowest basin"],
        ["Heading sensitivity is defined through the closest return within L; other definitions (e.g. Lyapunov "
         "exponents, zero here) would rank headings differently."],
        "T024: add focus-margin-aware route ranking on curved surfaces")
    findings = [
        finding("Zeros of the return distance (tau = 0.31 + 1.07i, L = 3) are exactly the primitive lattice "
                "directions with |v| <= L",
                "numerical", {"found": len(study["found"]), "predicted": len(study["predicted"])},
                {"checks": [_check("missing plus extra closure directions",
                                   len(study["missing"]) + len(study["extra"]))]}, tolerance=EXACT),
        finding("Near each closing heading the closing error grows at rate |v| (the loop length) per radian",
                "numerical", {"max_relative_slope_error": study["slope_relative_error"]},
                {"derivation": "r = |v| sin|theta - theta_v| near theta_v",
                 "checks": [_check("|r / sin(delta) - |v|| / |v| at delta = 1e-4", study["slope_relative_error"],
                                   1e-9, kind="analytic")]}, tolerance={"abs": 1e-9, "rel": 0}),
        finding("Closure basins are widest for the shortest (lowest-order) primitive classes", "numerical",
                [{"winding": b["winding"], "length": b["length"], "basin_width": b["basin_width"]} for b in top],
                {"checks": [_check("discordant (width, length) pairs", study["discordant"]),
                            _check("measured minus analytic basin width (rad)", study["width_error"],
                                   2 * study["step"] + 1e-12, kind="analytic")]}, tolerance=TIGHT),
        finding("A physical heading sensor closes the shortest loop within the computed heading tolerance",
                "sensor_performance", None, {}),
    ]
    return {"state": "completed", "fields": fields, "findings": findings}


# ================================================================ T024 / T025
TORUS = Torus(2.0, 1.0)
ROUTE_P, ROUTE_Q = (0.0, 0.4), (2.0, -0.3)
ROUTE_MAX, HORIZON = 11.0, 8.0


def torus_routes():
    found, candidates = routes.find_routes(TORUS, ROUTE_P, ROUTE_Q, ROUTE_MAX, headings=360, horizon=HORIZON)
    return {"routes": found, "candidates": candidates}


def _scipy_route_checks(surface, p, found):
    version = _version("scipy")
    if not version:
        return None
    worst_j, worst_c, rows = 0.0, 0.0, []
    for r in found:
        other = routes.scipy_check(surface, p, r, extra=HORIZON)
        worst_j = max(worst_j, abs(other["j_head"] - r["j_head"]) / max(1.0, abs(r["j_head"])))
        if (other["first_conjugate"] is None) != (r["first_conjugate"] is None):
            worst_c = max(worst_c, math.inf)
        elif r["first_conjugate"] is not None:
            worst_c = max(worst_c, abs(other["first_conjugate"] - r["first_conjugate"]))
        rows.append(other)
    return {"version": version, "j_head": worst_j, "conjugate": worst_c, "rows": rows}


def _route_table(found):
    return [{"length": r["length"], "heading": r["heading"], "lift": r["lift"], "amplification": r["amplification"],
             "focus_margin": r["focus_margin"], "margin_lower_bound": r["margin_lower_bound"]} for r in found]


@task("T024", changed_files=(MODULE, ROUTES, DOC),
      regression_tests=_tests("test_t024_t025_route_ranking_and_front"))
def focus_margin_ranking(ctx):
    data = ctx.memo("t024-routes", torus_routes)
    found = data["routes"]
    ranks = routes.rankings(found)
    ctx.artifact_json("torus-routes.json", {"p": ROUTE_P, "q": ROUTE_Q, "max_length": ROUTE_MAX,
                                            "horizon": HORIZON, "candidates": data["candidates"],
                                            "routes": found, "rankings": ranks})
    residual = max(r["endpoint_residual"] for r in found)
    wronskian = max(r["wronskian_drift"] for r in found)
    clairaut = max(r["clairaut_drift"] for r in found)
    batch = max(r["j_head_batch_vs_transfer"] / max(1.0, r["amplification"]) for r in found)
    disagreements = sum(a != b for a, b in zip(ranks["by_length"], ranks["by_amplification"])) + \
        sum(a != b for a, b in zip(ranks["by_length"], ranks["by_focus_margin"]))
    flat = jacobi.transfer(Plane(), [0.0, 0.0], 0.3, 20.0, steps=40)
    flat_min = float(np.min(flat.states[1:, 6] / flat.s[1:]))
    fields = _fields(
        "Between two points of the torus of revolution the ranking of geodesic routes by length, by heading "
        "amplification |j_head(L)| and by focus margin (distance to the first conjugate point) disagree; on a "
        "flat torus there are no conjugate points (infinite margin).",
        "Geodesic + heading Jacobi system on Torus(R=2, r=1), chart (phi, theta); K = cos theta / (r (R + r cos "
        "theta)); focus margin = s_c - L for the first zero s_c > 0 of j_head along the extended geodesic "
        "(negative: the route passes a conjugate point and is not locally minimizing).",
        [f"p = {ROUTE_P}, q = {ROUTE_Q} (chart), routes up to length {ROUTE_MAX}, extension horizon {HORIZON}",
         "360-heading fan, RK4 batch step 0.025, Newton on (heading, length) with Jacobian [j_head N, v]"],
        NO_PHYSICAL,
        "Endpoint lies on a lift of q; Wronskian det Phi = 1; Clairaut rho^2 phi' conserved; batch and "
        "ciw.lab.jacobi amplifications agree.",
        "Fan search for near-passes of every chart lift of q, batch Newton refinement, deduplication, "
        "re-integration of each route with ciw.lab.jacobi.transfer to L + horizon, conjugate points by Hermite "
        "zeros; optional scipy DOP853 re-integration.",
        f"{len(found)} routes from {data['candidates']} fan candidates; order by length {ranks['by_length']}, by "
        f"amplification {ranks['by_amplification']}, by focus margin {ranks['by_focus_margin']}.",
        f"Endpoint residual <= {residual:.1e}; Wronskian drift <= {wronskian:.1e}; Clairaut drift <= "
        f"{clairaut:.1e}; batch-vs-transfer amplification difference <= {batch:.1e} (relative).",
        ["duplicate routes merged", "routes beyond the length budget discarded", "censored margins (no conjugate "
         "point within the horizon) ranked as infinite", "Newton divergence (dropped, counted in candidates)"],
        ["The candidate set is what a 360-heading fan found up to length 11; completeness is not proven.",
         "Focus margin uses conjugate points of p only; conjugate points of q along the reversed route are not "
         "separately reported."],
        "T025: generate Pareto fronts over these routes")
    basis = {"checks": [_check("endpoint residual to the lift of q", residual, 1e-6, kind="invariant"),
                        _check("Wronskian drift", wronskian, 1e-8, kind="invariant"),
                        _check("Clairaut integral drift", clairaut, 1e-6, kind="invariant"),
                        _check("batch RK4 vs ciw.lab.jacobi amplification", batch, 1e-4, kind="cross_implementation")]}
    sc = _scipy_route_checks(TORUS, ROUTE_P, found)
    if sc:
        basis["independent_check"] = _independent(
            _check("scipy DOP853 j_head(L) and first conjugate point (max of both)", max(sc["j_head"], sc["conjugate"]),
                   1e-5, kind="high_precision"), "scipy.integrate.solve_ivp", sc["version"])
    findings = [
        finding("Geodesic routes p -> q on the torus with amplification and focus margin", "numerical",
                _route_table(found), basis, tolerance=ROUTE),
        finding("Rankings by length, amplification and focus margin disagree", "numerical", ranks,
                {"checks": [_check("rank positions that differ from the length order", disagreements, 1.0, "ge"),
                            _check("routes found", len(found), 3, "ge")]},
                counterexample={"statement": "The shortest route also minimizes amplification and maximizes focus margin",
                                "witness": {"shortest": _route_table(found)[ranks["by_length"][0]],
                                            "least_amplification": _route_table(found)[ranks["by_amplification"][0]],
                                            "largest_margin": _route_table(found)[ranks["by_focus_margin"][0]]}},
                tolerance=EXACT),
        finding("A flat torus has no conjugate points: j_head(s) = s > 0 (infinite focus margin)", "numerical",
                flat_min, {"derivation": "K = 0 gives j_head(s) = s",
                           "checks": [_check("min j_head(s)/s - 1 on (0, 20] (ciw.lab.jacobi, plane)", flat_min - 1.0,
                                             1e-12, kind="analytic")]}, tolerance=TIGHT),
        finding("Ranking routes by focus margin selects a route that is safe to execute on a physical part",
                "machine_safety", None, {}),
    ]
    return {"state": "completed", "fields": fields, "findings": findings}


@task("T025", changed_files=(MODULE, ROUTES, DOC),
      regression_tests=_tests("test_t024_t025_route_ranking_and_front"))
def pareto_fronts(ctx):
    data = ctx.memo("t024-routes", torus_routes)
    found = data["routes"]
    front = routes.pareto_front(found)
    violations = 0
    for i in front:
        violations += any(routes.dominates(found[j], found[i]) for j in range(len(found)) if j != i)
    for i in set(range(len(found))) - set(front):
        violations += not any(routes.dominates(found[j], found[i]) for j in front)
    pairs = {}
    for name, key in (("length_amplification", lambda r: (r["length"], r["amplification"])),
                      ("length_margin", lambda r: (r["length"], -routes.margin_value(r)))):
        keys = [key(r) for r in found]
        pairs[name] = [i for i, k in enumerate(keys)
                       if not any(all(x <= y for x, y in zip(o, k)) and o != k for o in keys)]
    table = _route_table(found)
    ctx.artifact_json("pareto.json", {"front": front, "fronts_2d": pairs, "routes": table})
    order = sorted(range(len(found)), key=lambda i: found[i]["length"])
    fsorted = sorted(pairs["length_amplification"], key=lambda i: found[i]["length"])
    ctx.artifact_text("pareto-length-amplification.svg", svg.line_plot(
        [("all routes (by length)", [found[i]["length"] for i in order], [found[i]["amplification"] for i in order]),
         ("length/amplification front", [found[i]["length"] for i in fsorted],
          [found[i]["amplification"] for i in fsorted])],
        title="Torus routes: length vs heading amplification", xlabel="route length L",
        ylabel="|j_head(L)|", logy=True))
    shortest = min(range(len(found)), key=lambda i: found[i]["length"])
    fields = _fields(
        "Length, amplification and focus margin conflict, so the three-objective Pareto front of the torus "
        "routes has several members, always including the shortest route.",
        "Minimize (L, |j_head(L)|, -margin) with censored margins treated as +infinity; a dominates b when no "
        "worse in every objective and better in one.",
        ["T024 routes (same run, shared memo)"], NO_PHYSICAL,
        "Front members are mutually non-dominated; every non-member is dominated by a front member.",
        "Exhaustive pairwise dominance on the T024 route set; two-objective fronts for length/amplification and "
        "length/margin; retained JSON and SVG.",
        f"3-objective front {front} of {len(found)} routes; length/amplification front "
        f"{pairs['length_amplification']}; length/margin front {pairs['length_margin']}.",
        "Inherits T024 route uncertainty (<= 1e-5 relative); front membership is exact given the values.",
        ["ties in objectives", "censored margins", "dominance violations counted exhaustively"],
        ["Objectives are unweighted; a decision needs weights or constraints that the workbench does not set."],
        "T032: build the counterexample library for 'shortest means safest'")
    findings = [
        finding("Three-objective Pareto front over the torus routes", "numerical",
                {"front": front, "fronts_2d": pairs, "routes": len(found)},
                {"checks": [_check("dominance violations (members dominated or non-members undominated)", violations),
                            _check("front size", len(front), 2, "ge"),
                            _check("shortest route missing from the front", 0 if shortest in front else 1)]},
                tolerance=EXACT),
        finding("A Pareto-optimal route is safe to execute on a physical part", "machine_safety", None, {}),
    ]
    return {"state": "completed", "fields": fields, "findings": findings}


# ================================================================ T026
INVARIANCE_LATTICES = {"generic": (5, 2, 7), "hexagonal": (2, 1, 2), "square": (1, 0, 1), "rectangular": (1, 0, 3)}
SPECTRUM_RADIUS_SQ = 60
WINDING_PROBES = ((1, 0), (0, 1), (2, -3), (3, 1), (4, 5))


def invariance_study(words=60, word_length=10):
    rng = _rng(2026)
    matrices = lat.sl2z_matrices(5) + [lat.random_word(rng, word_length, 3)[0] for _ in range(words)]
    spectra = {name: lat.length_spectrum(G, SPECTRUM_RADIUS_SQ) for name, G in INVARIANCE_LATTICES.items()}
    failures = {"spectrum": 0, "area": 0, "systole": 0, "winding_rule": 0}
    wrong_rule = 0
    for M in matrices:
        Minv = lat.inverse_sl2(M)
        for name, G in INVARIANCE_LATTICES.items():
            H = lat.transform(G, M)
            failures["spectrum"] += lat.length_spectrum(H, SPECTRUM_RADIUS_SQ) != spectra[name]
            failures["area"] += lat.det_form(H) != lat.det_form(G)
            failures["systole"] += lat.systole_sq(H) != lat.systole_sq(G)
            for m, n in WINDING_PROBES:
                transported = (Minv[0][0] * m + Minv[0][1] * n, Minv[1][0] * m + Minv[1][1] * n)
                failures["winding_rule"] += lat.quad(H, *transported) != lat.quad(G, m, n)
                naive = (M[0][0] * m + M[0][1] * n, M[1][0] * m + M[1][1] * n)
                wrong_rule += lat.quad(H, *naive) != lat.quad(G, m, n)
    # Float: area-one shapes under the same matrices; compare the first 30 lengths.
    float_dev = 0.0
    float_entry = max(max(abs(x) for row in M for x in row) for M in matrices[::7])
    for tau in FTR_SHAPES:
        base = sorted(math.sqrt(q) for q, _, _ in lat.lattice_vectors(lat.area_one_form(tau), 16.0))[:30]
        for M in matrices[:: 7]:
            image = lat.mobius(M, tau)
            other = sorted(math.sqrt(q) for q, _, _ in lat.lattice_vectors(lat.area_one_form(image), 16.0))[:30]
            float_dev = max(float_dev, max(abs(a - b) / a for a, b in zip(base, other)))
    generic = INVARIANCE_LATTICES["generic"]
    mirror = lat.transform_any(generic, ((1, 0), (0, -1)))
    doubled = lat.transform_any(generic, ((2, 0), (0, 1)))
    return {"matrices": len(matrices), "failures": failures, "wrong_rule_mismatches": wrong_rule,
            "float_max_relative_deviation": float_dev, "float_max_matrix_entry": float_entry,
            "mirror": {"gram": [str(x) for x in mirror],
                       "canonical": [str(x) for x in lat.gauss_reduce(mirror)[0]],
                       "same_spectrum": lat.length_spectrum(mirror, SPECTRUM_RADIUS_SQ) == spectra["generic"],
                       "same_canonical": lat.gauss_reduce(mirror)[0] == lat.gauss_reduce(generic)[0]},
            "sublattice": {"gram": [str(x) for x in doubled], "area_sq_ratio": _frac(Fraction(lat.det_form(doubled),
                                                                                             lat.det_form(generic))),
                           "systole_sq": _frac(lat.systole_sq(doubled)),
                           "original_systole_sq": _frac(lat.systole_sq(generic)),
                           "same_spectrum": lat.length_spectrum(doubled, SPECTRUM_RADIUS_SQ) == spectra["generic"]},
            "spectra": {k: [[_frac(q), mult] for q, mult in v[:8]] for k, v in spectra.items()}}


@task("T026", changed_files=(MODULE, LATTICE, PROVIDER, DOC),
      regression_tests=_tests("test_t026_modular_invariance",
                              "test_gauss_reduction_is_exact",
                              "test_ftr_provider_agreement"))
def modular_reduction_invariance(ctx):
    study = invariance_study()
    ctx.artifact_json("modular-invariance.json", study)
    failures = study["failures"]
    mirror, sub = study["mirror"], study["sublattice"]
    fields = _fields(
        "Length spectrum, area and systole are invariant under every SL(2,Z) change of basis and under reduction, "
        "exactly for integer Gram forms; winding labels transport by M^-1; orientation and index are not "
        "captured by the spectrum alone.",
        "G' = M^T G M, det G' = det G, Q'(M^-1 c) = Q(c); spectrum = multiset of Q over lattice vectors within "
        "R^2 = 60; systole^2 = first entry of the canonical form.",
        [f"integer lattices {INVARIANCE_LATTICES}",
         f"{study['matrices']} matrices: all SL(2,Z) with entries <= 5 plus 60 seeded words (PCG64 seed 2026)",
         f"area-one shapes {FTR_SHAPES} for the float comparison"],
        NO_PHYSICAL,
        "Exact equality of spectra, det and systole; float spectra equal to rounding.",
        "Transform, enumerate spectra exactly (row scan with exact filtering), compare; test the correct and the "
        "naive winding rules; exhibit the mirror (det -1) and index-2 (det 2) counterexamples; optionally compare "
        "fold length pairs with the pinned FTR provider.",
        f"exact failures {failures}; naive winding rule mismatches {study['wrong_rule_mismatches']}; float max "
        f"relative deviation {study['float_max_relative_deviation']:.2e}; mirror same spectrum "
        f"{mirror['same_spectrum']}, same canonical form {mirror['same_canonical']}; det-2 area ratio "
        f"{sub['area_sq_ratio']}.",
        f"Exact for integer forms; the float comparison loses digits with basis skew (entries up to "
        f"{study['float_max_matrix_entry']}: deviation {study['float_max_relative_deviation']:.1e}, tolerance 1e-8).",
        ["naive winding transport (M instead of M^-1)", "orientation reversal (mirror image)",
         "index-2 sublattice", "skewed bases with large entries (row-scan enumeration)"],
        ["Spectra are compared up to R^2 = 60, not over the whole lattice (a finite truncation)."],
        "T027: build polygonal translation-surface examples")
    findings = [
        finding("Length spectrum (R^2 <= 60), area and systole are exactly invariant under every tested SL(2,Z) "
                "change of basis and under reduction", "mathematical",
                {"matrices": study["matrices"], "failures": failures},
                {"checks": [_check("exact spectrum/area/systole/winding-transport failures", sum(failures.values()))]},
                tolerance=EXACT),
        finding("Area-one float spectra agree to rounding under the same basis changes", "numerical",
                study["float_max_relative_deviation"],
                {"checks": [_check("max relative deviation of the first 30 lengths (bases with entries up to "
                                   f"{study['float_max_matrix_entry']})", study["float_max_relative_deviation"],
                                   1e-8, kind="invariant")]}, tolerance={"abs": 1e-8, "rel": 0}),
        finding("Transporting winding labels with M instead of M^-1 breaks length invariance", "mathematical",
                study["wrong_rule_mismatches"],
                {"checks": [_check("naive-rule mismatches", study["wrong_rule_mismatches"], 1, "ge")]},
                counterexample={"statement": "Winding labels transform with the same matrix as the basis",
                                "witness": {"correct_rule": "c' = M^-1 c", "mismatches": study["wrong_rule_mismatches"]}},
                tolerance=EXACT),
        finding("The length spectrum does not determine the oriented shape: a mirror image is isospectral but not "
                "SL(2,Z)-equivalent", "mathematical", mirror,
                {"checks": [_check("isospectral mirror (0 = same spectrum)", 0 if mirror["same_spectrum"] else 1),
                            _check("mirror canonical form equal (0 = different)", 1 if mirror["same_canonical"] else 0)]},
                counterexample={"statement": "Equal length spectra imply SL(2,Z)-equivalent oriented lattices",
                                "witness": mirror}, tolerance=EXACT),
        finding("A det-2 integer matrix changes area and spectrum (an index-2 sublattice, not a basis change)",
                "mathematical", sub,
                {"checks": [_check("squared-area ratio minus 4", Fraction(sub["area_sq_ratio"]) - 4),
                            _check("spectrum unchanged (0 = changed)", 1 if sub["same_spectrum"] else 0)]},
                tolerance=EXACT),
    ]
    run = _provider(ctx)
    if run is not None and "refusal" not in run:
        cmp = fold_comparison(run)
        identity = run["identity"]
        invariance = max(abs(r["ftr_length_pair"][0] - r["ftr_length_pair"][1]) / r["ftr_length_pair"][0]
                         for r in cmp["rows"])
        findings.append(finding(
            "Closed-geodesic lengths before and after the fold agree with the pinned FTR length_pair and its winding "
            "transport", "numerical",
            {"cases": len(cmp["rows"]), "max_relative_difference": cmp["max_length_relative_difference"],
             "winding_mismatches": cmp["winding_mismatches"]},
            {"provider": ftr.provider_basis(identity),
             "checks": [_check("FTR reduced-winding mismatches (sign-consistent)", cmp["winding_mismatches"]),
                        _check("FTR length_pair invariance", invariance, 1e-9, kind="invariant")],
             "independent_check": _independent(
                 _check("FTR fold length_pair against ciw lengths", cmp["max_length_relative_difference"], 1e-9,
                        kind="high_precision"), ftr.checker_identity(identity)["implementation"], identity["revision"])},
            tolerance={"abs": 1e-9, "rel": 0}))
    state = _provider_note(run, fields, findings, (MODULE, LATTICE, PROVIDER))
    return {"state": state, "fields": fields, "findings": findings}


# ================================================================ T027 / T028 / T029
def example_surfaces():
    return {"square torus": surf.square_torus(), "L-shape": surf.l_shape(), "H(1,1) origami":
            surf.square_tiled("H(1,1) origami (4 squares)", [0, 1, 3, 2], [2, 3, 0, 1]),
            "octagon": surf.regular_octagon(), "hexagon": surf.regular_hexagon(), "pillowcase": surf.pillowcase()}


def _surd_json(x):
    return x.as_json() if isinstance(x, surf.Surd) else _frac(x)


def horizontal_cylinders():
    """Horizontal cylinders from exact flows: L-shape rows and the octagon's two strips."""
    L = surf.l_shape()
    l_rows = [float(L.flow(0, (Fraction(1, 3), Fraction(1, 7)), (1, 0))["time"]),
              float(L.flow(2, (Fraction(1, 3), Fraction(8, 7)), (1, 0))["time"])]
    O = surf.regular_octagon()
    S = surf.Surd
    one, zero = S(1), S(0)
    outer = O.flow(0, (S(Fraction(1, 3)), S(Fraction(1, 5))), (one, zero))["time"]
    middle = O.flow(0, (S(Fraction(1, 3)), S(1)), (one, zero))["time"]
    s = S(0, Fraction(1, 2))
    # Cylinder areas: middle strip height 1, outer strips total height sqrt 2 / 2.
    total = middle * 1 + outer * s
    return {"l_shape_circumferences": l_rows, "l_shape_r_cycles": [2, 1],
            "octagon_outer": outer, "octagon_middle": middle, "octagon_area": O.area(), "octagon_cylinder_area": total}


@task("T027", changed_files=(MODULE, SURFACES, DOC),
      regression_tests=_tests("test_t027_t029_surfaces_and_cones",
                              "test_surface_helpers_exact",
                              "test_regeneration_is_within_tolerance"))
def translation_surface_examples(ctx):
    examples = example_surfaces()
    records = {}
    for name, s in examples.items():
        t = s.topology()
        records[name] = {"gluing": t["gluing"], "vertices": t["vertices"], "edges": t["edges"], "faces": t["faces"],
                         "euler_characteristic": t["euler_characteristic"], "genus": t.get("genus"),
                         "cone_angles_over_pi": [_frac(c["cone_angle_over_pi"]) for c in t["cones"]],
                         "area": _surd_json(s.area())}
    cyl = horizontal_cylinders()
    ctx.artifact_json("translation-surfaces.json", {"surfaces": records, "cylinders": {
        k: (_surd_json(v) if isinstance(v, (surf.Surd, Fraction)) else v) for k, v in cyl.items()}})
    O = examples["octagon"]
    area_error = O.area() - surf.Surd(2, 2)
    cyl_error = cyl["octagon_cylinder_area"] - O.area()
    circ_error = (cyl["octagon_outer"] - surf.Surd(2, 1)).sign() != 0 \
        or (cyl["octagon_middle"] - surf.Surd(1, 1)).sign() != 0
    codes = {"octagon adjacent pairing": _refusal_code(lambda: surf.octagon_adjacent_pairing().require_translation()),
             "pillowcase": _refusal_code(examples["pillowcase"].require_translation),
             "mismatched edge lengths": _refusal_code(
                 surf.PolygonSurface, "bad", [surf.unit_square(), [(0, 0), (2, 0), (2, 1), (0, 1)]],
                 [((0, 0), (1, 2)), ((0, 1), (1, 3)), ((0, 2), (1, 0)), ((0, 3), (1, 1))])}
    genus_ok = records["L-shape"]["genus"] == 2 and records["octagon"]["genus"] == 2 and \
        records["square torus"]["genus"] == 1 and records["hexagon"]["genus"] == 1
    fields = _fields(
        "The square-tiled L (3 squares) and the regular octagon with opposite sides glued are genus-2 translation "
        "surfaces with one vertex class; their horizontal cylinder decompositions are exactly predictable; "
        "non-translation pairings are refused.",
        "Surface = convex polygons + edge involution; chi = V - E + F; genus (2 - chi)/2; octagon of side 1 in "
        "Q(sqrt 2): area 2 + 2 sqrt 2, horizontal cylinders of circumference 1 + sqrt 2 (height 1) and 2 + sqrt 2 "
        "(height sqrt 2 / 2).",
        ["square torus, L-shape r = (A B)(C), u = (A C)(B), H(1,1) origami r = (2 3), u = (0 2)(1 3)",
         "regular octagon and hexagon (exact surds), pillowcase (half-translation)"],
        NO_PHYSICAL,
        "Translation gluing (opposite edge vectors), chi = -2 for the genus-2 examples, cylinder areas summing to "
        "the surface area.",
        "Build each surface, validate gluing and convexity exactly, compute topology, trace exact horizontal "
        "flows to measure cylinder circumferences, and exercise refusals.",
        f"{ {k: (v['genus'], v['vertices'], v['cone_angles_over_pi']) for k, v in records.items()} }; octagon "
        f"cylinders {cyl['octagon_outer']} and {cyl['octagon_middle']}; L-shape horizontal circumferences "
        f"{cyl['l_shape_circumferences']}.",
        "Exact in Q and Q(sqrt 2), Q(sqrt 3); corner angles recognized to 1e-12 from exact edge vectors.",
        ["non-translation pairings refused", "unequal glued edge lengths refused", "non-convex or clockwise polygons "
         "refused", "half-translation pillowcase recognized"],
        ["Only horizontal cylinders are predicted; other periodic directions are traced in T028."],
        "T028: trace geodesics across glued edges")
    findings = [
        finding("L-shape and regular octagon are genus-2 translation surfaces with one vertex class", "mathematical",
                records,
                {"checks": [_check("genus mismatches (L 2, octagon 2, square torus 1, hexagon 1)", 0 if genus_ok else 1),
                            _check("octagon area minus 2 + 2 sqrt 2 (exact)", 0 if area_error.sign() == 0 else 1)]},
                tolerance=EXACT),
        finding("Horizontal cylinders: L-shape circumferences equal the r-cycle lengths, octagon strips have "
                "circumferences 2 + sqrt 2 and 1 + sqrt 2 with areas summing to the surface area", "mathematical",
                {"l_shape": cyl["l_shape_circumferences"], "octagon": [float(cyl["octagon_outer"]),
                                                                      float(cyl["octagon_middle"])]},
                {"checks": [_check("L-shape circumferences minus r-cycles", max(abs(a - b) for a, b in zip(
                    cyl["l_shape_circumferences"], cyl["l_shape_r_cycles"]))),
                            _check("octagon circumference mismatch (exact surd)", 1 if circ_error else 0),
                            _check("octagon cylinder area minus surface area (exact surd)",
                                   0 if cyl_error.sign() == 0 else 1)]}, tolerance=TIGHT),
        finding("Pairings that are not translations, or glue unequal edges, are refused as translation surfaces",
                "mathematical", codes,
                {"checks": [_refusal("octagon adjacent pairing", "GLUING_NOT_TRANSLATION",
                                     codes["octagon adjacent pairing"]),
                            _refusal("pillowcase (half-translation)", "GLUING_NOT_TRANSLATION", codes["pillowcase"]),
                            _refusal("edges of length 1 and 2 glued", "EDGE_LENGTH_MISMATCH",
                                     codes["mismatched edge lengths"])]}, tolerance=EXACT),
    ]
    return {"state": "completed", "fields": fields, "findings": findings}


L_STARTS = ((0, (Fraction(1, 3), Fraction(1, 7))), (1, (Fraction(6, 5), Fraction(2, 3))),
            (2, (Fraction(3, 4), Fraction(5, 4))))


def l_shape_flows(span=3):
    L = surf.l_shape()
    rows, undecided, bad_k = [], 0, 0
    for p in range(-span, span + 1):
        for q in range(-span, span + 1):
            if (p, q) == (0, 0) or math.gcd(p, q) != 1:
                continue
            for square, start in L_STARTS:
                try:
                    result = L.flow(square, start, (p, q), max_crossings=60)
                except surf.FlowTermination as exc:
                    rows.append({"direction": [p, q], "square": square, "outcome": exc.code})
                    undecided += exc.code != "SADDLE_CONNECTION"
                    continue
                if not result["closed"]:
                    undecided += 1
                    rows.append({"direction": [p, q], "square": square, "outcome": "not closed"})
                    continue
                k = result["time"]
                bad_k += not (k.denominator == 1 and 1 <= k <= 3)
                rows.append({"direction": [p, q], "square": square, "outcome": "closed", "multiplier": _frac(k),
                             "crossings": len(result["crossings"])})
    return {"rows": rows, "undecided": undecided, "bad_multiplier": bad_k}


def octagon_flows():
    S = surf.Surd
    O, F = surf.regular_octagon(), surf.regular_octagon(exact=False, tol=1e-9)
    start = (S(Fraction(1, 3)), S(Fraction(1, 5)))
    direction = (S(3), S(1, 1))
    exact = O.flow(0, start, direction, max_crossings=200)
    floated = F.flow(0, (1 / 3, 1 / 5), (3.0, 1 + math.sqrt(2)), max_crossings=200)
    theta = 0.3
    generic = F.flow(0, (1 / 3, 1 / 5), (math.cos(theta), math.sin(theta)), max_crossings=400, stop_on_return=True)
    # From the centre, aim 1e-12 (normal offset) past vertex 0: undecidable in float, refused within tolerance.
    cx, cy = 0.5, (1 + math.sqrt(2)) / 2
    norm = math.hypot(cx, cy)
    aim = (-cx + 1e-12 * cy / norm, -cy - 1e-12 * cx / norm)
    near_code = _refusal_code(F.flow, 0, (cx, cy), aim, 10)
    exact_code = _refusal_code(O.flow, 0, (S(Fraction(1, 2)), S(Fraction(1, 2))), (S(-1), S(-1)), 10)
    start_code = _refusal_code(F.flow, 0, (0.5, 1e-12), (1.0, 0.3), 10)
    return {"exact_closed": exact["closed"], "exact_time": exact["time"], "exact_crossings": len(exact["crossings"]),
            "exact_length": float(exact["time"]) * math.hypot(3.0, 1 + math.sqrt(2)),
            "float_closed": floated["closed"], "float_time": floated["time"],
            "same_crossing_sequence": exact["crossings"] == floated["crossings"],
            "time_difference": abs(float(exact["time"]) - floated["time"]),
            "generic_theta": theta, "generic_closed": generic["closed"],
            "generic_crossings": len(generic["crossings"]), "generic_min_clearance": generic["min_vertex_clearance"],
            "near_vertex_code": near_code, "exact_vertex_code": exact_code, "start_code": start_code,
            "tolerance": F.tol}


@task("T028", changed_files=(MODULE, SURFACES, DOC),
      regression_tests=_tests("test_t028_glued_edge_flow", "test_surface_helpers_exact"))
def trace_across_glued_edges(ctx):
    lflows = l_shape_flows()
    oflows = octagon_flows()
    saddle = _refusal_code(surf.l_shape().flow, 0, (Fraction(1, 2), Fraction(1, 2)), (1, 1))
    ctx.artifact_json("glued-edge-flows.json", {"l_shape": lflows, "octagon": {
        k: (_surd_json(v) if isinstance(v, surf.Surd) else v) for k, v in oflows.items()}})
    closed = sum(r["outcome"] == "closed" for r in lflows["rows"])
    saddles = sum(r["outcome"] == "SADDLE_CONNECTION" for r in lflows["rows"])
    fields = _fields(
        "Straight-line flow crosses glued edges by the edge translations: on the square-tiled L every rational "
        "direction is completely periodic or ends in a saddle connection (exactly decided), and on the octagon a "
        "float trajectory reproduces the exact Q(sqrt 2) trajectory within a declared tolerance.",
        "Flow x + t d inside a polygon; exit edge by exact segment intersection; re-entry at x + (a' - b) on the "
        "partner edge; a hit on a cone point terminates (saddle connection); closure multiplier k with displacement "
        "k (p, q) in Z^2 and 1 <= k <= 3 on the 3-square surface (Veech dichotomy).",
        [f"L-shape primitive directions |p|, |q| <= 3 from starts {[(s, tuple(map(str, x))) for s, x in L_STARTS]}",
         "octagon direction (3, 1 + sqrt 2) exact and float; generic float direction theta = 0.3 rad",
         f"float tolerance {oflows['tolerance']}"],
        NO_PHYSICAL,
        "Exact outcome for every rational L-shape trajectory; float octagon crossings identical to exact ones.",
        "Exact Fraction flow per direction and start; exact surd vs float flow on the octagon; tolerance-aware "
        "termination near vertices.",
        f"L-shape: {closed} closed, {saddles} saddle connections, {lflows['undecided']} undecided, "
        f"{lflows['bad_multiplier']} bad multipliers; octagon (3, 1 + sqrt 2): exact closure time "
        f"{oflows['exact_time']} (units of the direction vector; length {oflows['exact_length']:.12f}) after "
        f"{oflows['exact_crossings']} crossings, float difference "
        f"{oflows['time_difference']:.1e}; generic theta: {oflows['generic_crossings']} crossings, min vertex "
        f"clearance {oflows['generic_min_clearance']:.3e}.",
        "Exact for rational and Q(sqrt 2) data; float trajectories carry rounding growing with crossings (declared "
        "tolerance 1e-9).",
        ["saddle connection from an exact vertex hit refused", "float pass within tolerance of a vertex refused",
         "closure at a return inside the start polygon", "half-translation surfaces refused for flow"],
        ["A generic float direction is never certified non-periodic; only its first 400 crossings are traced."],
        "T029: detect cone singularities")
    findings = [
        finding("All 96 tested rational trajectories on the L-shape close (multiplier 1-3) or end in a saddle "
                "connection",
                "mathematical", {"closed": closed, "saddle_connections": saddles, "undecided": lflows["undecided"]},
                {"derivation": "Veech dichotomy for square-tiled surfaces: rational directions are completely periodic",
                 "checks": [_check("undecided trajectories", lflows["undecided"]),
                            _check("closure multipliers outside {1, 2, 3}", lflows["bad_multiplier"])]},
                tolerance=EXACT),
        finding("Trajectories hitting a cone point are terminated as saddle connections, exactly or within the "
                "declared float tolerance", "numerical",
                {"exact_l_shape": saddle, "exact_octagon": oflows["exact_vertex_code"],
                 "float_octagon": oflows["near_vertex_code"], "float_start": oflows["start_code"]},
                {"checks": [_refusal("L-shape (1/2, 1/2) direction (1, 1)", "SADDLE_CONNECTION", saddle),
                            _refusal("octagon exact centre-to-vertex", "SADDLE_CONNECTION", oflows["exact_vertex_code"]),
                            _refusal("octagon float pass 1e-12 from a vertex", "NEAR_VERTEX_WITHIN_TOLERANCE",
                                     oflows["near_vertex_code"]),
                            _refusal("float start 1e-12 from an edge", "START_NOT_INTERIOR", oflows["start_code"])]},
                tolerance=EXACT),
        finding("Float octagon flow reproduces the exact Q(sqrt 2) trajectory (same crossings, closure time)",
                "numerical", {"time_difference": oflows["time_difference"], "crossings": oflows["exact_crossings"]},
                {"checks": [_check("crossing sequences differ", 0 if oflows["same_crossing_sequence"] else 1),
                            _check("|float - exact| closure time", oflows["time_difference"], 1e-9,
                                   kind="cross_implementation")]}, tolerance={"abs": 1e-9, "rel": 0}),
        finding("A generic float octagon trajectory stays farther than the tolerance from every vertex for 400 crossings",
                "numerical", {"crossings": oflows["generic_crossings"], "min_clearance": oflows["generic_min_clearance"]},
                {"checks": [_check("minimum vertex clearance", oflows["generic_min_clearance"], oflows["tolerance"],
                                   "ge", kind="invariant")]}, tolerance={"abs": 1e-9, "rel": 1e-6}),
    ]
    return {"state": "completed", "fields": fields, "findings": findings}


@task("T029", changed_files=(MODULE, SURFACES, DOC),
      regression_tests=_tests("test_t027_t029_surfaces_and_cones", "test_regeneration_is_within_tolerance"))
def detect_cone_singularities(ctx):
    examples = example_surfaces()
    rows, defects, commutator_mismatch = {}, 0, 0
    for name, s in examples.items():
        t = s.topology()
        angles = sorted(c["cone_angle_over_pi"] for c in t["cones"])
        rows[name] = {"cone_angles_over_pi": [_frac(a) for a in angles], "singular": [_frac(a) for a in angles if a != 2],
                      "euler_characteristic": t["euler_characteristic"],
                      "gauss_bonnet_defect_over_pi": _frac(t["gauss_bonnet_defect_over_pi"]),
                      "angle_recognition_residual": t["angle_recognition_residual"], "gluing": t["gluing"]}
        defects += t["gauss_bonnet_defect_over_pi"] != 0
        perms = getattr(s, "permutations", None)
        if perms:
            predicted = sorted(Fraction(2 * c) for c in surf.commutator_cycles(perms["r"], perms["u"]))
            rows[name]["commutator_prediction_over_pi"] = [_frac(a) for a in predicted]
            commutator_mismatch += predicted != angles
    residual = max(r["angle_recognition_residual"] for r in rows.values())
    ctx.artifact_json("cone-angles.json", rows)
    expected = {"square torus": ["2"], "L-shape": ["6"], "H(1,1) origami": ["4", "4"], "octagon": ["6"],
                "hexagon": ["2", "2"], "pillowcase": ["1", "1", "1", "1"]}
    mismatch = sum(rows[k]["cone_angles_over_pi"] != v for k, v in expected.items())
    fields = _fields(
        "Vertex classes after gluing carry cone angles that sum corner angles; translation surfaces have angles "
        "2 pi (k + 1); the octagon has one 6 pi point; Gauss-Bonnet sum (2 pi - theta_v) = 2 pi chi holds.",
        "Union-find on corners (b ~ a', a ~ b' per glued pair), corner angles as exact rational multiples of pi, "
        "chi = V - E + F; origami cone points = cycles of the commutator r u r^-1 u^-1 (angle 2 pi times length).",
        [f"surfaces {list(examples)}"], NO_PHYSICAL,
        "Gauss-Bonnet defect 0 for every surface; commutator prediction equals union-find angles for origamis.",
        "Compute vertex classes and cone angles, compare with predictions and the commutator cycle type, and "
        "check Gauss-Bonnet exactly.",
        f"{ {k: v['cone_angles_over_pi'] for k, v in rows.items()} } (units of pi); Gauss-Bonnet defects {defects}; "
        f"commutator mismatches {commutator_mismatch}.",
        f"Exact after angle recognition (float residual <= {residual:.1e} against multiples of pi/24).",
        ["regular vertices (angle 2 pi) not reported as singular", "half-translation cone angle pi",
         "two cone points (H(1,1)) versus one (L-shape)"],
        ["Angle recognition assumes corner angles are multiples of pi/24, true for every declared polygon."],
        "T030: compare smooth and discrete geodesic approximations")
    findings = [
        finding("Cone angles: octagon and L-shape one 6 pi point, H(1,1) two 4 pi points, pillowcase four pi points, "
                "square and hexagon tori none", "mathematical", {k: v["cone_angles_over_pi"] for k, v in rows.items()},
                {"checks": [_check("mismatches against predicted cone angles", mismatch),
                            _check("origami commutator-cycle prediction mismatches", commutator_mismatch),
                            _check("angle recognition residual", residual, 1e-12, kind="analytic")]},
                tolerance=EXACT),
        finding("Gauss-Bonnet sum (2 pi - theta_v) = 2 pi chi holds exactly on every surface", "mathematical",
                {k: v["gauss_bonnet_defect_over_pi"] for k, v in rows.items()},
                {"checks": [_check("surfaces with nonzero defect", defects)]}, tolerance=EXACT),
        finding("Polygon vertices need not be cone singularities: the glued hexagon's vertices are regular points",
                "mathematical", rows["hexagon"]["cone_angles_over_pi"],
                {"checks": [_check("singular vertex classes on the hexagon torus", len(rows["hexagon"]["singular"]))]},
                counterexample={"statement": "Every polygon vertex of a glued surface is a cone singularity",
                                "witness": {"surface": "regular hexagon, opposite sides glued",
                                            "vertex_classes": 2, "cone_angles": ["2 pi", "2 pi"]}},
                tolerance=EXACT),
    ]
    return {"state": "completed", "fields": fields, "findings": findings}


# ================================================================ T030
@task("T030", changed_files=(MODULE, DISCRETE, DOC),
      regression_tests=_tests("test_t030_grid_metrication"))
def smooth_versus_discrete(ctx):
    scipy_version = _version("scipy")
    rows = grid.refinement_study(with_scipy=bool(scipy_version))
    angles = np.linspace(0.0, math.pi / 2, 36001)
    worst = {k: max((grid.metrication_ratio(a, k), a) for a in angles) for k in (4, 8)}
    closed_form = {k: grid.worst_metrication(k) for k in (4, 8)}
    ctx.artifact_json("grid-refinement.json", {"rows": rows, "worst_dense": worst, "worst_closed_form": closed_form})
    stencil_error = max(max(abs(r["graph4"] - r["stencil_norm4"]), abs(r["graph8"] - r["stencil_norm8"])) for r in rows)
    by_dir = {}
    for r in rows:
        by_dir.setdefault(r["direction"], []).append(r)
    drift = max(max(abs(a["ratio4"] - b["ratio4"]), abs(a["ratio8"] - b["ratio8"]))
                for rs in by_dir.values() for a, b in zip(rs, rs[1:]))
    fmm = {n: max(abs(r["fmm_rel_error"]) for r in rows if r["grid"] == n) for n in grid.REFINEMENTS}
    order = math.log(fmm[grid.REFINEMENTS[0]] / fmm[grid.REFINEMENTS[-1]]) / math.log(
        grid.REFINEMENTS[-1] / grid.REFINEMENTS[0])
    monotone = all(fmm[a] > fmm[b] for a, b in zip(grid.REFINEMENTS, grid.REFINEMENTS[1:]))
    diag = [r for r in rows if r["direction"] == "45 deg"]
    tilt = [r for r in rows if r["direction"].startswith("22.62")]
    worst_dense_error = max(abs(worst[k][0] - closed_form[k][0]) for k in (4, 8))
    series = [(f"{k}-neighbour, 45 deg" if k == 4 else f"{k}-neighbour, 22.6 deg",
               [r["grid"] for r in (diag if k == 4 else tilt)], [r[f"ratio{k}"] - 1 for r in (diag if k == 4 else tilt)])
              for k in (4, 8)]
    series.append(("fast marching, 45 deg", [r["grid"] for r in diag], [abs(r["fmm_rel_error"]) for r in diag]))
    ctx.artifact_text("metrication.svg", svg.line_plot(series, title="Relative length error under grid refinement",
                                                       xlabel="grid points per period N", ylabel="relative error",
                                                       logx=True, logy=True))
    fields = _fields(
        "Grid graph shortest paths do not converge to the Euclidean geodesic length under refinement: the relative "
        "error is set by direction (sqrt 2 - 1 = 41.4% at 45 deg for 4 neighbours, sqrt(4 - 2 sqrt 2) - 1 = 8.24% "
        "near 22.5 deg for 8 neighbours) and is independent of the spacing, whereas fast marching converges.",
        "Graph distance -> stencil norm |dx| + |dy| (4-nbr) or max + (sqrt 2 - 1) min (8-nbr); eikonal |grad T| = 1 "
        "solved by first-order fast marching (error O(h log 1/h) for a point source).",
        [f"periodic unit-square torus grids N = {grid.REFINEMENTS}",
         f"targets {[(name, pq, k) for name, pq, k in grid.TARGETS]} (displacement k (p, q) / 30)"],
        NO_PHYSICAL,
        "Graph distances equal the stencil norm at every N; ratios constant in N; fast-marching error decreasing.",
        "Dijkstra (heapq) for 4/8 stencils, fast marching, closed-form worst directions over 36001 angles, and "
        "optional scipy.sparse.csgraph.dijkstra on the same graphs.",
        f"4-nbr ratio at 45 deg {diag[0]['ratio4']:.6f} for every N; 8-nbr ratio at 22.62 deg {tilt[0]['ratio8']:.6f}; "
        f"worst dense ratios {{4: {worst[4][0]:.6f}, 8: {worst[8][0]:.6f}}}; fast-marching max relative error "
        f"{ {n: round(v, 5) for n, v in fmm.items()} } (observed order {order:.2f}).",
        "Graph distances exact to rounding (<= 1e-12); fast-marching order is an empirical fit over three grids.",
        ["refinement drift of graph ratios", "stencil-norm closed form", "dense worst-direction search",
         "periodic wrap within half a period"],
        ["Fast-marching convergence is shown on three grids, not proven here (standard result)."],
        "T031: study route changes under small metric perturbations")
    basis = {"checks": [_check("graph distance minus stencil norm", stencil_error, 1e-12, kind="analytic"),
                        _check("ratio change under refinement", drift, 1e-12, kind="self_convergence"),
                        _check("4-nbr error at 45 deg minus (sqrt 2 - 1)", diag[0]["ratio4"] - math.sqrt(2), 1e-12,
                               kind="analytic")]}
    if scipy_version:
        sc = max(max(abs(r["graph4"] - r["scipy4"]), abs(r["graph8"] - r["scipy8"])) for r in rows)
        basis["independent_check"] = _independent(
            _check("scipy.sparse.csgraph.dijkstra graph distances", sc, 1e-12, kind="high_precision"),
            "scipy.sparse.csgraph.dijkstra", scipy_version)
    findings = [
        finding("4- and 8-neighbour grid shortest paths do not converge to Euclidean length under refinement",
                "numerical", {"ratio4_45deg": [r["ratio4"] for r in diag], "ratio8_22_62deg": [r["ratio8"] for r in tilt]},
                basis,
                counterexample={"statement": "Grid shortest-path lengths converge to geodesic length as the grid is refined",
                                "witness": {"stencil": "4-neighbour", "direction": "45 deg", "grids": list(grid.REFINEMENTS),
                                            "ratio": diag[0]["ratio4"]}}, tolerance=TIGHT),
        finding("Worst-direction metrication error is sqrt 2 - 1 (4-nbr, 45 deg) and sqrt(4 - 2 sqrt 2) - 1 "
                "(8-nbr, 22.5 deg)", "numerical", {"4": worst[4][0] - 1, "8": worst[8][0] - 1},
                {"derivation": "maximize cos a + sin a and cos a + (sqrt 2 - 1) sin a on [0, pi/4]",
                 "checks": [_check("dense maximum minus closed form", worst_dense_error, 1e-8, kind="analytic")]},
                tolerance={"abs": 1e-8, "rel": 0}),
        finding("Fast marching converges to Euclidean length under refinement", "numerical",
                {"max_relative_error": {str(n): v for n, v in fmm.items()}, "observed_order": order},
                {"checks": [_check("errors decrease with N (0 = monotone)", 0 if monotone else 1),
                            _check("observed order", order, 0.5, "ge", kind="self_convergence")]},
                tolerance={"abs": 1e-9, "rel": 1e-6}),
        finding("Grid-planned path lengths predict distances travelled by a physical vehicle or tool", "physical",
                None, {}),
    ]
    return {"state": "completed", "fields": fields, "findings": findings}


# ================================================================ T031
SHEAR = (0, 1, 0)  # h = [[0, 1], [1, 0]] as a symmetric form (a, b, c)


def _perturbed(eps, x):
    """Q_eps(x) = |x|^2 + eps x^T h x for the shear perturbation h."""
    return x[0] * x[0] + x[1] * x[1] + eps * lat.quad(SHEAR, x[0], x[1])


def flip_threshold(delta):
    """Exact eps* where routes a = (1/2 - delta, 1/4) and b = a - (1, 0) tie under g = I + eps h."""
    a = (Fraction(1, 2) - delta, Fraction(1, 4))
    b = (a[0] - 1, a[1])
    gap = (b[0] ** 2 + b[1] ** 2) - (a[0] ** 2 + a[1] ** 2)
    slope = lat.quad(SHEAR, *a) - lat.quad(SHEAR, *b)
    return gap / slope, a, b


def shortest_translate(eps, z, window=2):
    best, arg = None, []
    for m in range(-window, window + 1):
        for n in range(-window, window + 1):
            x = (z[0] + m, z[1] + n)
            value = _perturbed(eps, x)
            if best is None or value < best:
                best, arg = value, [(m, n)]
            elif value == best:
                arg.append((m, n))
    return best, arg


def perturbation_study():
    rows = []
    for delta in (Fraction(1, 10), Fraction(1, 100), Fraction(1, 1000), Fraction(1, 10000)):
        eps_star, a, b = flip_threshold(delta)
        rows.append({"delta": _frac(delta), "eps_star": _frac(eps_star), "ratio": _frac(eps_star / delta)})
    delta = Fraction(1, 1000)
    eps_star, a, b = flip_threshold(delta)
    z = a
    sweep, switches, previous = [], 0, None
    for j in range(0, 101):
        eps = Fraction(j, 10000)
        value, arg = shortest_translate(eps, z)
        route = arg[0] if len(arg) == 1 else None
        x = (z[0] + arg[0][0], z[1] + arg[0][1])
        heading = math.degrees(math.atan2(float(x[1]), float(x[0])))
        sweep.append({"eps": float(eps), "length": math.sqrt(float(value)), "translates": [list(t) for t in arg],
                      "heading_deg": heading})
        if route is not None and previous is not None and route != previous:
            switches += 1
        if route is not None:
            previous = route
    at_star = shortest_translate(eps_star, z)[1]
    lo, hi = 0.0, 0.01
    fa = (0.5 - 1e-3, 0.25)
    fb = (fa[0] - 1.0, 0.25)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        qa = fa[0] ** 2 + fa[1] ** 2 + mid * 2 * fa[0] * fa[1]
        qb = fb[0] ** 2 + fb[1] ** 2 + mid * 2 * fb[0] * fb[1]
        lo, hi = (mid, hi) if qa < qb else (lo, mid)
    before = [s for s in sweep if s["eps"] < float(eps_star)][-1]
    after = [s for s in sweep if s["eps"] > float(eps_star)][0]
    return {"scaling": rows, "delta": _frac(delta), "eps_star": _frac(eps_star), "float_eps_star": 0.5 * (lo + hi),
            "tie_at_eps_star": [list(t) for t in at_star], "switches": switches, "sweep": sweep,
            "heading_jump_deg": abs(after["heading_deg"] - before["heading_deg"]),
            "length_jump": abs(after["length"] - before["length"])}


def sympy_threshold(delta):
    import sympy

    eps = sympy.symbols("eps")
    d = sympy.Rational(delta.numerator, delta.denominator)
    ax, ay = sympy.Rational(1, 2) - d, sympy.Rational(1, 4)
    bx = ax - 1
    qa = ax ** 2 + ay ** 2 + eps * 2 * ax * ay
    qb = bx ** 2 + ay ** 2 + eps * 2 * bx * ay
    (solution,) = sympy.solve(sympy.Eq(qa, qb), eps)
    return Fraction(int(sympy.numer(solution)), int(sympy.denom(solution)))


@task("T031", changed_files=(MODULE, LATTICE, DOC),
      regression_tests=_tests("test_t031_route_switch", "test_regeneration_is_within_tolerance"))
def metric_perturbation_routes(ctx):
    study = perturbation_study()
    ctx.artifact_json("metric-perturbation.json", study)
    ctx.artifact_text("route-switch.svg", svg.line_plot(
        [("shortest-route heading (deg)", [s["eps"] for s in study["sweep"]],
          [s["heading_deg"] for s in study["sweep"]])],
        title="Shortest route heading under g = I + eps h", xlabel="eps", ylabel="heading (deg)", markers=False))
    eps_star = Fraction(study["eps_star"])
    scaling_error = sum(Fraction(r["ratio"]) != 4 for r in study["scaling"])
    fields = _fields(
        "Near a tie, an arbitrarily small constant metric perturbation g = I + eps h switches the shortest route "
        "discontinuously: the switch occurs at eps* = 4 delta for a target delta away from the tie, the heading "
        "jumps by about 127 deg, and the minimal length stays continuous.",
        "Flat torus Z^2 with g = I + eps h, h = [[0, 1], [1, 0]]; Q_eps(x) = |x|^2 + 2 eps x1 x2; routes a = "
        "(1/2 - delta, 1/4) and b = a - (1, 0) tie when eps* = (|b|^2 - |a|^2) / (a^T h a - b^T h b) = 4 delta.",
        ["delta in {1/10, 1/100, 1/1000, 1/10000}", "sweep eps = j/10000, j = 0..100, target (1/2 - 1/1000, 1/4)"],
        NO_PHYSICAL,
        "Exactly one route switch in the sweep, at eps* = 1/250; tie (two translates) at eps*; eps*/delta = 4.",
        "Exact rational minimization over translates for every eps; float bisection for eps*; optional sympy solve.",
        f"eps* = {study['eps_star']} (float bisection {study['float_eps_star']:.15g}); {study['switches']} switch; "
        f"heading jump {study['heading_jump_deg']:.2f} deg; length change across the switch "
        f"{study['length_jump']:.2e}.",
        "Exact rational thresholds; float bisection converges to binary64 resolution.",
        ["other translates overtaking within the sweep (checked exactly)", "positive definiteness (|eps| < 1)",
         "tie at eps* has multiplicity 2"],
        ["Only constant perturbations are studied; a varying metric can also bend routes continuously."],
        "T032: counterexample library for 'shortest means safest'")
    basis = {"checks": [_check("eps*/delta - 4 mismatches over four deltas", scaling_error),
                        _check("switches in the sweep minus 1", study["switches"] - 1),
                        _check("float bisection minus exact eps*", study["float_eps_star"] - float(eps_star), 1e-15,
                               kind="exact_arithmetic"),
                        _check("tie multiplicity at eps* minus 2", len(study["tie_at_eps_star"]) - 2)]}
    sympy_version = _version("sympy")
    if sympy_version:
        other = sympy_threshold(Fraction(study["delta"]))
        basis["independent_check"] = _independent(_check("sympy.solve eps* - exact eps*", other - eps_star),
                                                  "sympy.solve", sympy_version)
    findings = [
        finding("The shortest route switches at eps* = 4 delta, which vanishes as the unperturbed gap vanishes",
                "mathematical", {"eps_star": study["eps_star"], "scaling": study["scaling"]}, basis, tolerance=EXACT),
        finding("A metric perturbation just above eps* = 1/250 (0.4%) turns the shortest-route heading by about "
                "127 deg while the minimal length changes continuously", "numerical",
                {"heading_jump_deg": study["heading_jump_deg"], "length_jump": study["length_jump"]},
                {"checks": [_check("heading jump (deg)", study["heading_jump_deg"], 90.0, "ge", kind="invariant"),
                            _check("length change across the switch", study["length_jump"], 1e-3, "le",
                                   kind="invariant")]},
                counterexample={"statement": "A small metric perturbation changes the shortest route only slightly",
                                "witness": {"delta": study["delta"], "eps": "1/200 > eps* = 1/250",
                                            "heading_jump_deg": study["heading_jump_deg"]}}, tolerance=FLOAT),
        finding("A metric calibrated from physical measurements is accurate enough to decide between near-tied routes",
                "calibration", None, {}),
    ]
    return {"state": "completed", "fields": fields, "findings": findings}


# ================================================================ T032
BUMP = GaussianBump(1.5, 1.0)
SADDLE = Saddle(1.0)


def counterexample_library():
    witnesses = {}
    inner, _ = routes.find_routes(TORUS, (0.0, math.pi), (2.5, math.pi), 9.0)
    outer, _ = routes.find_routes(TORUS, (0.0, 0.0), (1.5, 0.0), 9.0)
    tie, _ = routes.find_routes(TORUS, (0.0, 0.0), (2.2, 0.0), 9.0)
    bump, _ = routes.find_routes(BUMP, (-2.5, 0.0), (2.5, 0.0), 9.0)
    saddle, _ = routes.find_routes(SADDLE, (-1.5, -0.5), (1.5, 0.8), 8.0)
    witnesses["torus-inner-amplification"] = {
        "surface": TORUS.describe(), "p": [0.0, math.pi], "q": [2.5, math.pi], "shortest": inner[0],
        "alternative": min((r for r in inner[1:] if r["amplification"] < inner[0]["amplification"]),
                           key=lambda r: r["length"]),
        "analytic_shortest_amplification": math.sinh(2.5)}
    alt = [r for r in outer[1:] if routes.margin_value(r) > routes.margin_value(outer[0])]
    witnesses["torus-outer-focus-margin"] = {
        "surface": TORUS.describe(), "p": [0.0, 0.0], "q": [1.5, 0.0], "shortest": outer[0],
        "alternative": min(alt, key=lambda r: r["length"]),
        "analytic_shortest_margin": math.pi * math.sqrt(3.0) - 4.5}
    witnesses["torus-tied-shortest"] = {"surface": TORUS.describe(), "p": [0.0, 0.0], "q": [2.2, 0.0],
                                        "routes": tie[:3]}
    # The straight route over the top has heading 0 by the y -> -y symmetry of the configuration.
    top = min(bump, key=lambda r: abs(r["heading"]))
    witnesses["bump-amplification-vs-conjugacy"] = {
        "surface": BUMP.describe(), "p": [-2.5, 0.0], "q": [2.5, 0.0], "shortest": bump[0],
        "alternative": top, "top_heading": top["heading"], "routes": len(bump)}
    delta = 0.05
    witnesses["sphere-near-antipodal"] = {"separation": math.pi - delta,
                                          "routes": routes.sphere_route_pair(math.pi - delta),
                                          "transfer": routes.sphere_transfer_check(math.pi - delta)}
    witnesses["saddle-unique-route"] = {"surface": SADDLE.describe(), "p": [-1.5, -0.5], "q": [1.5, 0.8],
                                        "routes": saddle}
    return witnesses


def _scipy_witness_check(lib):
    version = _version("scipy")
    if not version:
        return None
    worst = 0.0
    for key, surface, p in (("torus-inner-amplification", TORUS, (0.0, math.pi)),
                            ("torus-outer-focus-margin", TORUS, (0.0, 0.0)),
                            ("bump-amplification-vs-conjugacy", BUMP, (-2.5, 0.0))):
        w = lib[key]
        for route in (w["shortest"], w["alternative"]):
            other = routes.scipy_check(surface, p, route, extra=HORIZON)
            worst = max(worst, abs(other["j_head"] - route["j_head"]) / max(1.0, abs(route["j_head"])))
            if route["first_conjugate"] is not None and other["first_conjugate"] is not None:
                worst = max(worst, abs(other["first_conjugate"] - route["first_conjugate"]))
            elif (route["first_conjugate"] is None) != (other["first_conjugate"] is None):
                worst = math.inf
    return {"version": version, "worst": worst}


def _brief(route):
    return {"length": route["length"], "heading": route["heading"], "amplification": route["amplification"],
            "focus_margin": route["focus_margin"], "margin_lower_bound": route["margin_lower_bound"]}


@task("T032", changed_files=(MODULE, ROUTES, DOC),
      regression_tests=_tests("test_t032_counterexample_library"))
def shortest_is_not_safest(ctx):
    lib = counterexample_library()
    ctx.artifact_json("counterexample-library.json", lib)
    inner, outer = lib["torus-inner-amplification"], lib["torus-outer-focus-margin"]
    tie, bump = lib["torus-tied-shortest"], lib["bump-amplification-vs-conjugacy"]
    sphere, saddle = lib["sphere-near-antipodal"], lib["saddle-unique-route"]
    sc = _scipy_witness_check(lib)
    inner_gap = inner["shortest"]["amplification"] - inner["alternative"]["amplification"]
    inner_anchor = abs(inner["shortest"]["amplification"] - inner["analytic_shortest_amplification"]) / math.sinh(2.5)
    outer_alt_margin = routes.margin_value(outer["alternative"])
    outer_alt_margin = outer["alternative"]["margin_lower_bound"] if math.isinf(outer_alt_margin) else outer_alt_margin
    outer_gap = outer_alt_margin - outer["shortest"]["focus_margin"]
    outer_anchor = abs(outer["shortest"]["focus_margin"] - outer["analytic_shortest_margin"])
    t0, t1 = tie["routes"][0], tie["routes"][1]
    bump_gap = bump["shortest"]["amplification"] - bump["alternative"]["amplification"]
    minor = sphere["routes"][0]
    fields = _fields(
        "'Shortest means safest' fails on curved surfaces: a shortest geodesic can have larger heading "
        "amplification, a smaller focus margin, a tie with another route, or a near-conjugate endpoint, while "
        "the flat torus (T021) and simply connected negatively curved surfaces admit no such witness.",
        "Heading amplification |j_head(L)| and focus margin s_c - L from j'' + K j = 0 along each route; exact "
        "anchors: inner equator K = -1 (j_head = sinh s), outer equator K = 1/3 (conjugate at pi sqrt 3), unit "
        "sphere (j_head = sin s, conjugate at pi).",
        ["Torus(2, 1): inner-equator pair (0, pi) -> (2.5, pi); outer-equator pair (0, 0) -> (1.5, 0); (0, 0) -> "
         "(2.2, 0)", "GaussianBump(h = 1.5, sigma = 1): (-2.5, 0) -> (2.5, 0)",
         "unit sphere, separation pi - 0.05", "Saddle(c = 1): (-1.5, -0.5) -> (1.5, 0.8)"],
        NO_PHYSICAL,
        "Each witness violates the refuted statement by a margin far above numerical error; analytic anchors "
        "reproduce closed forms.",
        "Fan search + Newton + ciw.lab.jacobi verification per configuration (as T024); closed-form sphere pair "
        "checked with ciw.lab.jacobi; optional scipy DOP853 re-integration of the witness routes.",
        f"inner: shortest L = {inner['shortest']['length']:.4f} amp {inner['shortest']['amplification']:.4f} vs L = "
        f"{inner['alternative']['length']:.4f} amp {inner['alternative']['amplification']:.4f}; outer: shortest margin "
        f"{outer['shortest']['focus_margin']:.4f} vs alternative >= {outer_alt_margin:.4f}; tie: lengths "
        f"{t0['length']:.10f} and {t1['length']:.10f}; bump: shortest amp {bump['shortest']['amplification']:.4f} vs "
        f"top route amp {bump['alternative']['amplification']:.4f} with margin {bump['alternative']['focus_margin']:.4f}; "
        f"sphere minor-arc margin {minor['focus_margin']:.4f}; saddle routes found {len(saddle['routes'])}.",
        "Route quantities to about 1e-6 relative (RK4 step 0.04, scipy agreement); witness gaps are O(0.1-4).",
        ["analytic anchors (sinh 2.5, pi sqrt 3 - 4.5, sin 0.05)", "censored margins use the horizon lower bound",
         "negative margins (route past a conjugate point) kept, not discarded", "symmetric duplicate routes"],
        ["Witnesses show the statements are false in general; they do not rank routes for any application.",
         "Route sets are fan-search results and may be incomplete (the saddle's uniqueness is the Cartan-Hadamard "
         "theorem, not the search)."],
        "Extend the library to variable-curvature meshes (surfaces-discrete section) and to manufacturing paths")
    findings = []
    inner_basis = {"checks": [_check("shortest minus alternative amplification", inner_gap, 1.0, "ge", kind="invariant"),
                              _check("shortest amplification vs sinh 2.5 (relative)", inner_anchor, 1e-6, kind="analytic")]}
    if sc:
        inner_basis["independent_check"] = _independent(
            _check("scipy DOP853 witness routes (j_head relative, conjugate absolute)", sc["worst"], 1e-5,
                   kind="high_precision"), "scipy.integrate.solve_ivp", sc["version"])
    findings.append(finding(
        "Torus inner equator: the shortest route has larger heading amplification than a longer route", "numerical",
        {"shortest": _brief(inner["shortest"]), "alternative": _brief(inner["alternative"])}, inner_basis,
        counterexample={"statement": "The shortest geodesic between two points has the least heading amplification",
                        "witness": {"surface": "Torus(2, 1)", "p": inner["p"], "q": inner["q"],
                                    "shortest": _brief(inner["shortest"]), "alternative": _brief(inner["alternative"])}},
        tolerance=ROUTE))
    findings.append(finding(
        "Torus outer equator: the shortest route has a smaller focus margin than a longer route", "numerical",
        {"shortest": _brief(outer["shortest"]), "alternative": _brief(outer["alternative"])},
        {"checks": [_check("alternative minus shortest focus margin", outer_gap, 1.0, "ge", kind="invariant"),
                    _check("shortest margin vs pi sqrt 3 - 4.5", outer_anchor, 1e-6, kind="analytic")]},
        counterexample={"statement": "The shortest geodesic has the largest focus margin (farthest from conjugate points)",
                        "witness": {"surface": "Torus(2, 1)", "p": outer["p"], "q": outer["q"],
                                    "shortest": _brief(outer["shortest"]), "alternative": _brief(outer["alternative"])}},
        tolerance=ROUTE))
    findings.append(finding(
        "Torus (0, 0) -> (2.2, 0): two mirror-image shortest routes tie, so 'the' shortest route is not unique",
        "numerical", {"lengths": [t0["length"], t1["length"]], "headings": [t0["heading"], t1["heading"]]},
        {"checks": [_check("length difference of the two shortest routes", t0["length"] - t1["length"], 1e-9,
                           kind="invariant"),
                    _check("heading separation (rad)", abs(t0["heading"] - t1["heading"]), 0.1, "ge", kind="invariant")]},
        counterexample={"statement": "The shortest route between two points is unique",
                        "witness": {"surface": "Torus(2, 1)", "p": tie["p"], "q": tie["q"],
                                    "routes": [_brief(t0), _brief(t1)]}}, tolerance=ROUTE))
    findings.append(finding(
        "Gaussian bump: a longer route over the top has smaller amplification but passes a conjugate point",
        "numerical", {"shortest": _brief(bump["shortest"]), "alternative": _brief(bump["alternative"])},
        {"checks": [_check("shortest minus top-route amplification", bump_gap, 0.2, "ge", kind="invariant"),
                    _check("depth of the top route past its conjugate point (-margin)",
                           -bump["alternative"]["focus_margin"], 1.0, "ge", kind="invariant"),
                    _check("top-route heading (the symmetric straight route)", bump["top_heading"], 1e-9,
                           kind="invariant")]},
        counterexample={"statement": "Low heading amplification certifies a robust (locally minimizing) route",
                        "witness": {"surface": "GaussianBump(1.5, 1)", "p": bump["p"], "q": bump["q"],
                                    "shortest": _brief(bump["shortest"]), "alternative": _brief(bump["alternative"])}},
        tolerance=ROUTE))
    findings.append(finding(
        "Unit sphere, separation pi - 0.05: the minimizing arc ends 0.05 before its conjugate point", "numerical",
        {"focus_margin": minor["focus_margin"], "j_head": sphere["transfer"]["j_head"],
         "conjugate": sphere["transfer"]["first_conjugate"]},
        {"derivation": "j_head = sin s on the unit sphere; conjugate point at pi",
         "checks": [_check("focus margin of the minimizing arc", minor["focus_margin"], 0.1, "le", kind="analytic"),
                    _check("ciw.lab.jacobi j_head(L) - sin 0.05", sphere["transfer"]["j_head"] - math.sin(0.05), 1e-8,
                           kind="analytic"),
                    _check("ciw.lab.jacobi conjugate point - pi", sphere["transfer"]["first_conjugate"] - math.pi, 1e-6,
                           kind="analytic")]},
        counterexample={"statement": "A shortest geodesic stays well away from conjugate points",
                        "witness": {"surface": "unit sphere", "separation": sphere["separation"],
                                    "focus_margin": minor["focus_margin"],
                                    "targeting_condition_number": 1 / abs(minor["j_head"])}}, tolerance=FLOAT))
    findings.append(finding(
        "Search on the saddle (K < 0, simply connected) finds exactly one route, consistent with Cartan-Hadamard "
        "uniqueness, so this surface admits no witness",
        "numerical", {"routes": len(saddle["routes"])},
        {"derivation": "Cartan-Hadamard: exp is a diffeomorphism on complete simply connected K <= 0 surfaces",
         "checks": [_check("routes found minus 1", len(saddle["routes"]) - 1)]}, tolerance=EXACT))
    findings.append(finding("A route from this library is safe (or unsafe) to execute on a physical part or vehicle",
                            "machine_safety", None, {}))
    return {"state": "completed", "fields": fields, "findings": findings}
