"""Flat torus and topology tasks T019-T032.

Scope: flat tori C / Lambda with Lambda = Z w1 + Z w2 (exact lattice algebra on
integer or rational Gram matrices), closed geodesics and their winding
classes, heading sensitivity and cut-locus degeneracy on flat tori, candidate
routes with focus margins on curved surfaces (torus of revolution, Gaussian
bump, saddle, sphere), polygonal translation surfaces with cone points, grid
approximations of geodesic distance, and route switching under metric
perturbation. Exact arithmetic (integers, Fractions, Q(sqrt d)) is used
wherever the geometry allows; floats carry declared tolerances. Independent
checks come from sympy (thresholds, and a route field derived from the
embedding for scipy to integrate), mpmath, scipy and the pinned FTR provider;
second ciw formulations (vector-form lattice reduction, Pareto fronts by a
dominance matrix and a sweep, exact traces of float trajectories) are
cross-implementation or exact-arithmetic checks, never independent ones. The
pinned-provider claims are always reported, as not established when the
provider is unbound or refused.

Non-claims: every surface is a declared mathematical object in normalized
units. No finding concerns a physical part, vehicle, sensor, tool path or
workspace; route "safety", sensor closure performance and calibration adequacy
are recorded as unestablished physical or authority claims. Candidate route
sets come from a fan search that is stable under doubling its density but is
not proven complete. Agreement with the pinned Flat-Torus-Geodesic-Reference
provider is independent implementation agreement, not verification by another
party.
"""
from __future__ import annotations

from fractions import Fraction
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
from .evidence import finding, holds
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
    """Regression node ids; every task is also covered by the section-wide completion and label tests."""
    return tuple(f"{TESTS}::{name}" for name in names + ("test_every_task_is_registered_and_completes",
                                                          "test_every_finding_has_its_expected_label",
                                                          "test_next_steps_name_forward_work"))


# ---------------------------------------------------------------- shared helpers
def _check(reference, observed, tolerance=0.0, comparison="abs_le", kind="exact_arithmetic"):
    observed, tolerance = float(observed), float(tolerance)
    return {"reference_kind": kind, "reference": reference, "observed": observed, "tolerance": tolerance,
            "comparison": comparison, "passed": holds(observed, tolerance, comparison)}


EXACT_UNC = {"kind": "exact", "value": 0.0, "basis": "integer, Fraction or Q(sqrt d) arithmetic; no rounding"}


def _unc(kind, value, basis):
    """Per-finding uncertainty object (kind, value, basis) for the uncertainty budget."""
    return {"kind": kind, "value": float(value), "basis": basis}


def _refusal(reference, expected, observed):
    return {"reference_kind": "refusal", "reference": reference, "expected_refusal": expected,
            "observed_refusal": observed, "passed": observed == expected}


def _independent(check, checker, revision):
    return dict(check, producer={"implementation": PRODUCER, "revision": f"ciw {__version__}"},
                checker={"implementation": checker, "revision": revision})


def _version(ctx, module):
    """Version string of an optional module when ``ctx`` reports it available, else None."""
    if not ctx.available(f"module:{module}"):
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
    return ctx.memo("flat-torus/ftr-provider", compute)


# The provider claims are always emitted so a run's claim set does not depend on whether the
# provider was bound; unbound or refused, the claim is honestly unestablished.
FTR_CLAIMS = {
    "T019": "Float Gauss reduction agrees with the pinned FTR fold_to_fundamental_domain on reduced tau and the "
            "reducing SL(2,Z) matrix (up to -I)",
    "T020": "Closed-geodesic lengths |m w1 + n w2|, edge-crossing counts and unit area agree with the pinned FTR "
            "loop_length, trace_closed_geodesic and normalized_lattice",
    "T026": "Closed-geodesic lengths before and after the fold agree with the pinned FTR length_pair and its winding "
            "transport",
}
FTR_NOTE = ("The pinned FTR comparison is optional (bind --provider ftr=<checkout> --provider ftr-python=<Python 3.12 "
            "interpreter>); its finding records whether it ran.")


def _provider_ran(run) -> bool:
    return run is not None and "refusal" not in run


def _provider_placeholder(task_id, run):
    """The FTR claim when the provider is unbound or refused: recorded, not established."""
    value = "not bound" if run is None else {"refusal": run["refusal"], "message": run["message"]}
    return finding(FTR_CLAIMS[task_id], "numerical", value, {}, expected_not_established=True)


def _provider_note(run, fields, changed_files):
    """Attach provider identity or the reason it did not run; return the resulting task state."""
    fields["unresolved_assumptions"].append(FTR_NOTE)
    if run is None:
        return "completed"
    if "refusal" in run:
        fields["unresolved_assumptions"].append(
            f"The bound FTR provider was refused ({run['refusal']}), so the provider comparison did not run.")
        return "partial"
    from .runner import builtin_identity

    # The built-in identity stays at the top level so regression notes on optional modules remain accurate.
    fields["provider_runtime_identity"] = dict(builtin_identity(changed_files), provider=run["identity"],
                                               producer=PRODUCER)
    return "completed"


def fold_comparison(run):
    """Compare the provider's folds with float Gauss reduction and exact winding transport."""
    rows, tau_diff, matrix_mismatch, winding_mismatch, length_diff, recovered = [], 0.0, 0, 0, 0.0, 0.0
    # parse_output guarantees one provider row per requested case.
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


T019_BOUND, T019_WORDS, T019_WORD_LENGTH = 5, 200, 12


def t019_words(words=T019_WORDS, word_length=T019_WORD_LENGTH):
    """The seeded S/T^k words shared by the ciw study and the sympy recomputation."""
    rng = _rng(19)
    return [lat.random_word(rng, word_length, 3) for _ in range(words)]


def enumeration_study(bound=T019_BOUND, words=T019_WORDS, word_length=T019_WORD_LENGTH):
    matrices = lat.sl2z_matrices(bound)
    identity = ((1, 0), (0, 1))
    per_lattice, failures = {}, {"inverse": 0, "area": 0, "reduction": 0, "reducer": 0, "automorphism": 0}
    word_rows = []
    sampled = t019_words(words, word_length)
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


def _vector_lagrange(G, M):
    """Canonical form of the basis (columns of M) under the form G, by vector Lagrange reduction.

    A second ciw formulation used as a cross-implementation check of
    lat.gauss_reduce (which rewrites Gram entries): it works on the basis
    vectors u, v with the bilinear form of G in exact integers,
    v -= round(B(u, v) / Q(u)) u, then (u, v) -> (v, -u) while Q(v) < Q(u);
    the boundary rule makes b >= 0. Returns ((a, b, c), det[u v]).
    """
    g0, g1, g2 = G
    ux, uy, vx, vy = M[0][0], M[1][0], M[0][1], M[1][1]

    def bil(x0, x1, y0, y1):
        return g0 * x0 * y0 + g1 * (x0 * y1 + x1 * y0) + g2 * x1 * y1

    while True:
        mu = math.floor(Fraction(bil(ux, uy, vx, vy), bil(ux, uy, ux, uy)) + Fraction(1, 2))
        vx, vy = vx - mu * ux, vy - mu * uy
        if bil(vx, vy, vx, vy) < bil(ux, uy, ux, uy):
            ux, uy, vx, vy = vx, vy, -ux, -uy
            continue
        break
    a, b = bil(ux, uy, ux, uy), bil(ux, uy, vx, vy)
    if 2 * b == -a:
        vx, vy = vx + ux, vy + uy
    a, b, c = bil(ux, uy, ux, uy), bil(ux, uy, vx, vy), bil(vx, vy, vx, vy)
    if a == c and b < 0:
        ux, uy, vx, vy = vx, vy, -ux, -uy
    form = (bil(ux, uy, ux, uy), bil(ux, uy, vx, vy), bil(vx, vy, vx, vy))
    return form, ux * vy - vx * uy


def vector_reduction_check(bound=T019_BOUND):
    """Recompute the canonical form of every enumerated basis and word by vector reduction; compare with ciw.

    Both reductions are ciw code (a cross-implementation check, not an
    independent one). A case fails unless the vector-reduced form of the basis
    equals that of the identity basis (canonical(G)), equals lat.gauss_reduce
    of M^T G M, satisfies |2b| <= a <= c with b >= 0 on the boundary, and the
    reduced basis and M both have det 1.
    """
    identity = ((1, 0), (0, 1))
    matrices = lat.sl2z_matrices(bound) + [M for M, _ in t019_words()]
    compared = mismatches = 0
    for G in LATTICES.values():
        reference, _ = _vector_lagrange(G, identity)
        for M in matrices:
            compared += 1
            form, det = _vector_lagrange(G, M)
            a, b, c = form
            canonical = abs(2 * b) <= a <= c and (b >= 0 or (abs(2 * b) < a and a < c))
            ciw_form = lat.gauss_reduce(lat.transform(G, M))[0]
            mismatches += not (form == reference and form == ciw_form and canonical and det == 1
                               and M[0][0] * M[1][1] - M[0][1] * M[1][0] == 1)
    return {"compared": compared, "mismatches": mismatches}


@task("T019", changed_files=(MODULE, LATTICE, PROVIDER, DOC),
      regression_tests=_tests("test_t019_reduction_and_refusals",
                              "test_gauss_reduction_is_exact",
                              "test_vector_reduction_check_detects_a_wrong_boundary_rule",
                              "test_ftr_refusal_makes_task_partial",
                              "test_provider_output_is_refused_when_unreadable_or_incomplete",
                              "test_ftr_provider_agreement"))
def enumerate_lattice_representatives(ctx):
    study = ctx.memo("flat-torus/t019-enumeration", enumeration_study)
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
        "the canonical form; recompute every canonical form with a second (vector-form) reduction; count "
        "closed-domain reduced bases; exercise refusals; optionally compare float folds with the pinned FTR "
        "provider.",
        f"{total_failures} failures over {len(LATTICES) * (study['matrices'] + study['words'])} exact reductions; "
        f"reduced-basis counts {counts} match predictions {PREDICTED_REDUCED_BASES}.",
        "Exact integer/rational arithmetic: no rounding. Provider comparisons are float with tolerance 1e-9.",
        ["non-unimodular basis change (det 2) refused", "orientation reversal (det -1) refused",
         "non-integer basis change refused", "boundary conventions (b >= 0 on |2b| = a or a = c)",
         "large-entry words (exact integers, no overflow)"],
        ["Automorphism counts are checked for six declared lattices, not proven for all lattices here "
         "(the general statement is the classical stabilizer theorem)."],
        "Deferred research question: extend the automorphism-count check from the six declared lattices to a seeded "
        "sample of reduced Gram matrices that includes every boundary case of the fundamental domain (|b| = a, "
        "a = c, and the hexagonal point where both hold) and compare each count with the classical stabilizer "
        "theorem, which is cited here rather than checked in general")
    findings = []
    second = vector_reduction_check()
    basis = {"checks": [_check("exact reduction failures (inverse, area, canonical form, reducer, automorphism)",
                               total_failures),
                        _check(f"second ciw reduction (vector-form Lagrange on the basis vectors) of all "
                               f"{second['compared']} bases and words: canonical form, equality with canonical(G) "
                               "and lat.gauss_reduce, boundary rule, det 1 (mismatching cases)", second["mismatches"],
                               kind="cross_implementation")]}
    findings.append(finding(
        "Every enumerated SL(2,Z) basis and every random word reduces exactly to the same canonical Gram form",
        "mathematical", {"failures": failures, "reductions": len(LATTICES) * (study["matrices"] + study["words"])},
        basis, uncertainty=EXACT_UNC, tolerance=EXACT))
    findings.append(finding(
        "Number of reduced bases equals the predicted stabilizer count (2 generic, 4 boundary or square, 12 hexagonal)",
        "mathematical", counts,
        {"derivation": "Stabilizers in SL(2,Z): +-I generically, order 4 at i, order 6 at exp(i pi/3); boundary "
                       "points have two representatives in the closed domain",
         "checks": [_check("count mismatches against prediction", count_mismatch)]},
        uncertainty=EXACT_UNC, tolerance=EXACT))
    findings.append(finding(
        "Integer matrices with det != 1 are refused as SL(2,Z) basis changes", "mathematical", codes,
        {"checks": [_refusal("lat.transform with det 2", "BASIS_CHANGE_NOT_UNIMODULAR",
                             codes["det 2 (index-2 sublattice)"]),
                    _refusal("lat.transform with det -1", "BASIS_CHANGE_REVERSES_ORIENTATION",
                             codes["det -1 (orientation reversal)"]),
                    _refusal("lat.transform with a non-integer entry", "BASIS_CHANGE_NOT_INTEGER", codes["non-integer"]),
                    _check("canonical form of the det-2 image equals canonical(G) (0 = different lattice)",
                           1 if sub_canonical == lat.gauss_reduce(generic)[0] else 0)],
         "derivation": "det(M^T G M) = det(M)^2 det G, so a det-2 image has 4 times the squared area of G and "
                       "spans an index-2 sublattice"},
        counterexample={"statement": "Any integer change of basis generates the same lattice",
                        "witness": {"gram": list(generic), "matrix": [[2, 0], [0, 1]],
                                    "image_canonical": [str(x) for x in sub_canonical],
                                    "canonical": [str(x) for x in lat.gauss_reduce(generic)[0]]}},
        uncertainty=EXACT_UNC, tolerance=EXACT))
    run = _provider(ctx)
    if _provider_ran(run):
        cmp = fold_comparison(run)
        ctx.artifact_json("ftr-folds.json", cmp)
        identity = run["identity"]
        findings.append(finding(
            FTR_CLAIMS["T019"], "numerical",
            {"max_tau_difference": cmp["max_tau_difference"], "matrix_mismatches": cmp["matrix_mismatches"],
             "cases": len(cmp["rows"])},
            {"provider": ftr.provider_basis(identity),
             "checks": [_check("matrix mismatches up to -I", cmp["matrix_mismatches"]),
                        _check("reduction recovers the declared reduced tau", cmp["max_recovery_error"], 1e-9,
                               kind="invariant")],
             "independent_check": _independent(
                 _check("FTR reduced tau", cmp["max_tau_difference"], 1e-9, kind="high_precision"),
                 ftr.checker_identity(identity)["implementation"], identity["revision"])},
            uncertainty=_unc("roundoff", cmp["max_tau_difference"],
                             "binary64 Mobius images of seeded words reduced by two float implementations"),
            tolerance={"abs": 1e-9, "rel": 0}))
    else:
        findings.append(_provider_placeholder("T019", run))
    state = _provider_note(run, fields, (MODULE, LATTICE, PROVIDER))
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
            first = flow["first_return"]
            # Observed from the flow: first return time, displacement there, returns by t = 1, crossings.
            ok = (first is not None and first == Fraction(1, g) and flow["returns"] == g
                  and flow["displacement_at_first_return"] == (pm, pn)
                  and first * first * lat.quad(WINDING_FORM, m, n) == lat.quad(WINDING_FORM, pm, pn)
                  and flow["crossings"] == (abs(m) + abs(n)) // g)
            failures += 0 if ok else 1
            rows.append({"winding": [m, n], "gcd": g, "primitive": info["primitive"],
                         "length_sq": lat.quad(WINDING_FORM, m, n),
                         "first_return_time": None if first is None else _frac(first),
                         "returns_by_t_1": flow["returns"], "crossings_before_first_return": flow["crossings"]})
    primitive = [(m, n) for m in range(0, 4) for n in range(-3, 4) if math.gcd(m, n) == 1 and (m > 0 or n > 0)]
    intersections, wrong = [], 0
    for i, v in enumerate(primitive):
        for w in primitive[i + 1:]:
            count = lat.intersection_count(v, w)
            expected = abs(v[0] * w[1] - v[1] * w[0])
            wrong += count != expected
            intersections.append({"a": list(v), "b": list(w), "count": count, "det": expected})
    # Lattice points in the disk Q <= R^2: row scan (Fincke-Pohst) against a brute-force box count.
    vectors = lat.lattice_vectors(WINDING_FORM, radius * radius)
    total = len(vectors) + 1
    prim = sum(1 for _, m, n in vectors if math.gcd(m, n) == 1)
    box_total, box_primitive = lat.box_lattice_count(WINDING_FORM, radius * radius)
    area = math.sqrt(lat.det_form(WINDING_FORM))
    count = {"radius": radius, "lattice_points_with_origin": total, "primitive": prim,
             "box_count_with_origin": box_total, "box_primitive": box_primitive,
             "gauss_estimate": math.pi * radius ** 2 / area,
             "primitive_fraction": prim / (total - 1), "six_over_pi_sq": 6 / math.pi ** 2}
    return {"rows": rows, "failures": failures, "primitive_classes": len(primitive), "intersections": intersections,
            "intersection_failures": wrong, "count": count}


def golden_returns(k_max=22):
    """Gaps ||F_k phi|| at Fibonacci returns (binary64) against phi^-k (Binet), with a rounding bound.

    fl(q phi) differs from q phi by at most q 2^-53 + ulp(q phi)/2 and the
    subtraction of round(q phi) is exact, so ``gap_bound`` bounds the binary64
    gap error; ``limit_residual`` = |q gap - 1/sqrt 5| is exactly phi^-2k/sqrt 5.
    """
    fib = [0, 1]
    while len(fib) <= k_max + 1:
        fib.append(fib[-1] + fib[-2])
    rows = []
    for k in range(2, k_max + 1):
        q = fib[k]
        x = q * PHI
        gap = abs(x - round(x))
        rows.append({"k": k, "q": q, "gap": gap, "analytic": PHI ** (-k), "q_times_gap": q * gap,
                     "gap_bound": q * 2.0 ** -53 + math.ulp(x) / 2,
                     "limit_residual": PHI ** (-2 * k) / math.sqrt(5)})
    return rows


def mpmath_gaps(rows, dps=50):
    import mpmath

    with mpmath.workdps(dps):
        phi = (1 + mpmath.sqrt(5)) / 2
        return [float(abs(r["q"] * phi - mpmath.nint(r["q"] * phi))) for r in rows]


def length_comparison(run):
    worst, rows = 0.0, []
    # parse_output guarantees one provider row per requested case.
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
                              "test_provider_output_is_refused_when_unreadable_or_incomplete",
                              "test_ftr_provider_agreement"))
def classify_by_winding(ctx):
    study = ctx.memo("flat-torus/t020-windings", winding_study)
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
    min_ratio = min(r["gap"] / r["analytic"] for r in gaps)
    # |q gap - 1/sqrt 5| minus its exact value phi^-2k / sqrt 5 and the binary64 bound q * gap_bound (+ 2^-52 for
    # the last roundings): nonpositive when the float data agree with q gap -> 1/sqrt 5 at the proven rate.
    limit_slack = max(abs(r["q_times_gap"] - 1 / math.sqrt(5)) - r["limit_residual"] - r["q"] * r["gap_bound"]
                      - 2.0 ** -52 for r in gaps)
    last = gaps[-1]
    float_limit = abs(last["q_times_gap"] - 1 / math.sqrt(5))
    phi_fraction = Fraction(PHI)
    log2_denominator = phi_fraction.denominator.bit_length() - 1
    fields = _fields(
        "Closed geodesics of a flat torus are exactly the straight lines in nonzero lattice directions: primitive "
        "(m, n) gives a closed geodesic traversed once with length |m w1 + n w2|, (k m', k n') its k-fold cover, and "
        "irrational directions never close although their return gaps shrink.",
        "Lattice-coordinate flow alpha = alpha0 + m t, beta = beta0 + n t on R^2 / Z^2 with metric Q(m, n) = "
        "a m^2 + 2 b m n + c n^2; predicted first return at t = 1/gcd(m, n) with displacement the primitive vector; "
        "Q(g m', g n') = g^2 Q(m', n'); intersection number |det(v, w)|; Binet: ||F_k phi|| = phi^-k and "
        "|F_k phi^-k - 1/sqrt 5| = phi^-2k / sqrt 5.",
        [f"Gram form {WINDING_FORM} (area sqrt 31)", f"start (alpha0, beta0) = {tuple(map(str, WINDING_START))}",
         "all windings with |m|, |n| <= 6", "golden slope phi in lattice coordinates, Fibonacci returns k <= 22"],
        NO_PHYSICAL,
        "Measured first return 1/gcd with the primitive displacement, gcd returns by t = 1, (|m| + |n|)/gcd crossings "
        "before the first return, and intersection count |det| for every tested class.",
        "Exact segment-by-segment flow per winding, detecting returns by an exact solve on each segment; exact "
        "intersection counting for primitive pairs; lattice-point count by row scan against a brute-force box; "
        "binary64 and 50-digit golden return gaps with a rounding bound.",
        f"{study['failures']} winding failures over {len(study['rows'])} classes; {study['intersection_failures']} "
        f"intersection failures over {len(study['intersections'])} pairs; {count['lattice_points_with_origin']} "
        f"lattice points within R = {count['radius']} (box count {count['box_count_with_origin']}, Gauss estimate "
        f"{count['gauss_estimate']:.2f}); primitive fraction {count['primitive_fraction']:.4f} (asymptotic 6/pi^2 = "
        f"{count['six_over_pi_sq']:.4f}); golden gaps / phi^-k >= {min_ratio:.8f}; at k = {last['k']} the exact "
        f"residual |q gap - 1/sqrt 5| is {last['limit_residual']:.2e}, while the binary64 value deviates by "
        f"{float_limit:.2e} (float cancellation, within its rounding bound).",
        "Winding results exact. Golden gaps: binary64 relative error <= 1e-6 for q <= 17711 (cancellation grows "
        "like q * eps / phi^-k); the deviation of binary64 q * gap from 1/sqrt 5 is float error, not a measurement "
        "of the convergence rate, which is exact (Binet).",
        ["non-primitive windings (multiple covers)", "zero winding refused", "start on a cell wall refused",
         "corner passes counted on both axes", "binary64 cancellation in ||q phi||",
         "binary64 slopes are rational (never truly irrational)"],
        ["Non-closure of irrational directions is a theorem (irrationality), not something a finite computation "
         "establishes; the numerical gaps only illustrate it.",
         "Primitive fraction approaches 6/pi^2 only asymptotically; no rate is claimed."],
        "Deferred research question: measure how fast the primitive fraction of winding vectors approaches 6/pi^2 "
        "(its deviation over a doubling sequence of radii against the O(log R / R) error of the classical estimate), "
        "which is recorded here only as a limit, without a rate")
    findings = [
        finding("Every winding with |m|, |n| <= 6 first returns to its start at t = 1/gcd, displaced by its primitive "
                "vector after (|m| + |n|)/gcd edge crossings, and returns gcd times by t = 1 (a gcd-fold cover of the "
                "primitive loop)",
                "mathematical", {"classes": len(study["rows"]), "failures": study["failures"]},
                {"derivation": "Q(g m', g n') = g^2 Q(m', n'), so the (m, n) curve has gcd times the primitive length",
                 "checks": [_check("measured first-return time, displacement, return count and crossings against "
                                   "prediction (mismatching classes)", study["failures"])]},
                uncertainty=EXACT_UNC, tolerance=EXACT),
        finding(f"For all {len(study['intersections'])} pairs of the {study['primitive_classes']} primitive classes "
                "with 0 <= m <= 3, |n| <= 3 the transverse intersections number |det(v, w)|", "mathematical",
                {"pairs": len(study["intersections"]), "failures": study["intersection_failures"]},
                {"checks": [_check("intersection count mismatches (exact rational solve)",
                                   study["intersection_failures"])]},
                uncertainty=EXACT_UNC, tolerance=EXACT),
        finding("The lattice-point count within radius R (including the origin, and its primitive part) equals a "
                "brute-force count over a box proven to contain the disk", "mathematical",
                {k: count[k] for k in ("radius", "lattice_points_with_origin", "primitive")},
                {"derivation": "Q = ((c n + b m)^2 + det m^2) / c >= det m^2 / c bounds |m| (and |n| symmetrically); "
                               "the count is pi R^2 / A up to the cell-covering bound pi (2 R D + D^2) / A",
                 "checks": [_check("row-scan minus box count (all points)",
                                   count["lattice_points_with_origin"] - count["box_count_with_origin"]),
                            _check("row-scan minus box count (primitive points)",
                                   count["primitive"] - count["box_primitive"])]},
                uncertainty=EXACT_UNC, tolerance=TIGHT),
    ]
    golden_basis = {"derivation": "Binet: F_{k+1} - phi F_k = (-1/phi)^k, so ||F_k phi|| = phi^-k for k >= 2 and "
                                  "|F_k phi^-k - 1/sqrt 5| = phi^-2k / sqrt 5",
                    "checks": [_check("binary64 gap relative to phi^-k", float_rel, 1e-6, kind="analytic"),
                               _check("smallest gap / phi^-k (a closure would give 0)", min_ratio, 0.5, "ge",
                                      kind="analytic"),
                               _check("|q gap - 1/sqrt 5| minus (phi^-2k / sqrt 5 + binary64 rounding bound)",
                                      limit_slack, 0.0, "signed_le", kind="analytic")]}
    mp_version = _version(ctx, "mpmath")
    if mp_version:
        mp = mpmath_gaps(gaps)
        golden_basis["independent_check"] = _independent(
            _check("mpmath 50-digit ||q phi||", max(abs(r["gap"] - g) / g for r, g in zip(gaps, mp)), 1e-6,
                   kind="high_precision"), "mpmath", mp_version)
    findings.append(finding(
        "The golden-slope geodesic has positive return gaps phi^-k at Fibonacci returns, with q * gap -> 1/sqrt 5",
        "numerical", {"q": [r["q"] for r in gaps], "gap": [r["gap"] for r in gaps]}, golden_basis,
        uncertainty=_unc("roundoff", float_rel,
                         "binary64 cancellation in q phi - round(q phi), relative to phi^-k (q <= 17711)"),
        tolerance={"abs": 1e-15, "rel": 1e-6}))
    findings.append(finding(
        "A binary64 heading slope is rational, so a float simulation cannot represent a non-closing direction",
        "numerical", {"slope_denominator": phi_fraction.denominator, "log2_denominator": log2_denominator},
        {"derivation": "IEEE 754 binary64: every finite value is an integer significand of at most 53 bits times a "
                       "power of two, so a slope in [1, 2) is p / 2^52 (a dyadic rational with denominator at most "
                       "2^52), and a line of rational slope p / q in lattice coordinates closes after q turns"},
        counterexample={"statement": "A floating-point geodesic direction can be irrational (non-closing)",
                        "witness": {"float_phi": PHI,
                                    "exact_fraction": f"{phi_fraction.numerator}/2^{log2_denominator}",
                                    "closes_after_alpha_turns": phi_fraction.denominator}},
        uncertainty={"kind": "exact", "value": 0.0,
                     "basis": "IEEE 754 format fact; the witness fraction is the exact value of the binary64 phi"},
        tolerance=EXACT))
    run = _provider(ctx)
    if _provider_ran(run):
        cmp = length_comparison(run)
        ctx.artifact_json("ftr-lengths.json", cmp)
        identity = run["identity"]
        findings.append(finding(
            FTR_CLAIMS["T020"], "numerical",
            {"lengths": len(cmp["rows"]), "max_relative_difference": cmp["max_relative_difference"],
             "crossing_mismatches": cmp["crossing_mismatches"]},
            {"provider": ftr.provider_basis(identity),
             "checks": [_check("crossing-count mismatches or unclosed traces", cmp["crossing_mismatches"]),
                        _check("FTR area-one normalization", cmp["area_deviation"], 1e-12, kind="invariant")],
             "independent_check": _independent(
                 _check("FTR loop_length", cmp["max_relative_difference"], 1e-12, kind="high_precision"),
                 ftr.checker_identity(identity)["implementation"], identity["revision"])},
            uncertainty=_unc("roundoff", cmp["max_relative_difference"],
                             "relative difference of two binary64 length evaluations"),
            tolerance={"abs": 1e-12, "rel": 0}))
    else:
        findings.append(_provider_placeholder("T020", run))
    state = _provider_note(run, fields, (MODULE, LATTICE, PROVIDER))
    return {"state": state, "fields": fields, "findings": findings}


# ================================================================ T021
FLAT_BASIS = ((Fraction(1), Fraction(0)), (Fraction(3, 10), Fraction(11, 10)))
FLAT_P = (Fraction(1, 10), Fraction(1, 5))
FLAT_Q = (Fraction(7, 10), Fraction(11, 20))


def flat_translates(max_length=Fraction(13, 5)):
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
            if q2 <= max_length ** 2:
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
         f"{len(rs)} lattice translates with length <= 13/5"],
        NO_PHYSICAL,
        "j_head(L) = L and j_lat(L) = 1 for every route; the length order and the amplification order coincide.",
        "Enumerate translates exactly; integrate each route with ciw.lab.jacobi.transfer on the plane chart; "
        "compare j_head(L) with L and count discordant pairs between the two orderings.",
        f"{len(rs)} routes, shortest length {rs[0]['length']:.6f}; max |j_head/L - 1| = {j_err:.2e}; "
        f"discordant pairs {study['discordant_pairs']}.",
        "RK4 is exact for the linear flat system up to rounding (<= 1e-12 relative).",
        ["ties in exact length excluded from the discordance count", "endpoint reached (lift consistency)",
         "lateral column constant (j_lat = 1)"],
        ["The equivalence needs j_head(L) to be one strictly increasing function of L for every route (constant "
         "K <= 0, e.g. flat or closed hyperbolic surfaces); it can fail where K > 0 somewhere or where curvature "
         "differs between routes. T032 records curved-surface counterexamples."],
        "Deferred research question: test the equivalence of shortest and least heading-sensitive routes on a closed "
        "hyperbolic surface (constant K = -1, for example a genus-2 surface glued from a regular octagon), the other "
        "case its assumption covers, with routes between two points in different homotopy classes")
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
                uncertainty=_unc("roundoff", j_err,
                                 "relative deviation of RK4 j_head(L) from L on the linear flat system"),
                tolerance={"abs": 1e-12, "rel": 0}),
        finding("For every flat torus and every pair of points the least-sensitive geodesic is a shortest one",
                "mathematical", True,
                {"derivation": "j_head(s) = s is strictly increasing, so ordering routes by |j_head(L)| equals "
                               "ordering by L (ties included)"},
                uncertainty={"kind": "exact", "value": 0.0, "basis": "closed-form derivation"}, tolerance=EXACT),
        finding("Shortest equals least heading-sensitive whenever j_head(L) is one strictly increasing function of "
                "L for all routes (constant K <= 0); with K > 0 somewhere, or curvature that differs between routes, "
                "the equivalence can fail", "mathematical", True,
                {"derivation": "Constant K < 0: j_head(s) = sinh(sqrt(-K) s)/sqrt(-K), strictly increasing, so the "
                               "orders coincide; constant K > 0: j_head(s) = sin(sqrt K s)/sqrt K is not monotone "
                               "(sin s on the unit sphere); variable K: routes of different lengths follow different "
                               "Jacobi equations, so |j_head(L)| need not order like L (T032 torus witnesses) "
                               "(ciw.lab.jacobi.constant_curvature)"},
                uncertainty={"kind": "exact", "value": 0.0, "basis": "closed-form derivation"}, tolerance=EXACT),
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
    vertices, located = {}, {}
    for name, G in LATTICES.items():
        v = located[name] = lat.cut_locus_vertices(G)
        vertices[name] = {"vertices": [[_frac(z[0]), _frac(z[1]), k] for z, k in v],
                          "sum_k_minus_2": sum(k - 2 for _, k in v)}
    tiny = Fraction(1, 2 ** 60)
    z = (Fraction(1, 2) + tiny, Fraction(1, 4))
    exact_mult = len(lat.nearest_translates(SQUARE, z)[1])
    float_mult = lat.float_multiplicity(SQUARE, z, rel_tol=0.0)
    rounded = (Fraction(float(z[0])), Fraction(float(z[1])))
    rounded_mult = len(lat.nearest_translates(SQUARE, rounded)[1])
    band = []
    for eps in (Fraction(1, 10 ** 6), Fraction(1, 10 ** 8), Fraction(1, 10 ** 10), Fraction(1, 10 ** 12), tiny, 0):
        zz = (Fraction(1, 2) + eps, Fraction(1, 4))
        e = len(lat.nearest_translates(SQUARE, zz)[1])
        f = lat.float_multiplicity(SQUARE, zz, rel_tol=rel_tol)
        band.append({"epsilon": float(eps), "exact_multiplicity": e, "tolerance_multiplicity": f,
                     "verdict": "tie" if e > 1 else ("ambiguous within tolerance" if f > 1 else "unique")})
    # Exact ties whose targets binary64 cannot represent: every cut-locus vertex with a non-dyadic coordinate.
    ties = []
    for name, G in LATTICES.items():
        for vertex, k in located[name]:
            if any(x.denominator & (x.denominator - 1) for x in vertex):
                ties.append({"lattice": name, "target": [_frac(vertex[0]), _frac(vertex[1])], "exact": k,
                             "binary64": lat.float_multiplicity(G, vertex, rel_tol=0.0),
                             "tolerance": lat.float_multiplicity(G, vertex, rel_tol=rel_tol)})
    return {"half_periods": half, "census": {str(k): v for k, v in sorted(census.items())}, "grid": n_grid,
            "cut_locus": vertices, "float_counterexample": {"epsilon": "2^-60", "exact": exact_mult,
                                                            "binary64": float_mult,
                                                            "exact_of_rounded_target": rounded_mult},
            "tolerance_band": band, "rel_tol": rel_tol, "non_representable_ties": ties,
            "binary64_undercounts": sum(t["binary64"] < t["exact"] for t in ties),
            "tolerance_undercounts": sum(t["tolerance"] < t["exact"] for t in ties)}


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
    ties = study["non_representable_ties"]
    fields = _fields(
        "Shortest representatives are tied exactly on the cut locus (the Voronoi boundary of the lattice): "
        "multiplicity 2 on edges and 3 or 4 at vertices, with sum over vertices of (k - 2) = 2 for every flat "
        "torus; binary64 cannot decide ties below its resolution, and a declared relative tolerance is needed "
        "so that exact ties at non-representable targets are not reported as unique.",
        "Squared lengths Q(z + lambda) are exact rationals for rational z and integer G; the cut locus of a point "
        "is a graph with V vertices of multiplicity k_v and E = sum k_v / 2 edges, and V - E + 1 = chi = 0.",
        [f"square torus G = {SQUARE}; lattices {LATTICES}", f"{n} x {n} rational grid of targets",
         "near-tie target (1/2 + eps, 1/4), eps down to 2^-60",
         f"{len(ties)} cut-locus vertices with non-dyadic coordinates (exact ties binary64 cannot represent)"],
        NO_PHYSICAL,
        "Multiplicities 2 (half periods), 4 (square centre), sum (k_v - 2) = 2 on every tested lattice.",
        "Exact nearest-translate search, grid census, exact circumcenter enumeration of cut-locus vertices, and "
        "binary64 detection with and without a declared relative tolerance.",
        f"half periods {study['half_periods']}; census {study['census']} (predicted {predicted_census}); "
        f"cut-locus vertices {{name: sum(k-2)}} = { {k: v['sum_k_minus_2'] for k, v in study['cut_locus'].items()} }; "
        f"eps = 2^-60: exact multiplicity {fc['exact']}, binary64 {fc['binary64']} (the rounded target is an exact "
        f"tie of multiplicity {fc['exact_of_rounded_target']}); non-representable exact ties undercounted by raw "
        f"binary64 comparison: {study['binary64_undercounts']} of {len(ties)}, with relative tolerance "
        f"{study['rel_tol']:g}: {study['tolerance_undercounts']}.",
        "Exact for rational inputs; the tolerance-aware float detector over-reports near ties by design "
        "('ambiguous within tolerance').",
        ["exact ties at half periods", "Voronoi vertices of generic, boundary, square, hexagonal and rectangular "
         "lattices", "binary64 rounding of the target itself", "exact ties at targets binary64 cannot represent"],
        ["Targets with irrational coordinates are only handled in float with a declared tolerance."],
        "Deferred research question: decide ties exactly for targets with algebraic irrational coordinates (for "
        "example exact algebraic or interval arithmetic that certifies the sign of each squared-length difference), "
        "which are handled here only in binary64 with a declared tolerance")
    findings = [
        finding("Square-torus half-period targets have exactly 2 (edge) or 4 (centre) shortest representatives",
                "mathematical", study["half_periods"],
                {"checks": [_check("mismatches against predicted multiplicities", half_mismatch)]},
                uncertainty=EXACT_UNC, tolerance=EXACT),
        finding("Multiplicity census on the 12 x 12 rational grid equals the predicted cut-locus counts",
                "mathematical", study["census"],
                {"checks": [_check("census deviation from {1: n^2 - 2n + 1, 2: 2(n - 1), 4: 1}", census_mismatch)]},
                uncertainty=EXACT_UNC, tolerance=EXACT),
        finding("On six lattices the cut-locus vertices satisfy sum over vertices of (k_v - 2) = 2 (Euler "
                "characteristic 0)",
                "mathematical", {k: v["sum_k_minus_2"] for k, v in study["cut_locus"].items()},
                {"derivation": "E = sum k_v / 2 and V - E + F = 0 with F = 1 give sum (k_v - 2) = 2",
                 "checks": [_check("lattices violating the identity", vertex_mismatch)]},
                uncertainty=EXACT_UNC, tolerance=EXACT),
        finding("Rounding the target to binary64 changes its shortest-representative multiplicity (exact 1, "
                "binary64 2)", "numerical", fc,
                {"checks": [_check("binary64 multiplicity minus exact multiplicity (expect 1)",
                                   fc["binary64"] - fc["exact"] - 1),
                            _check("exact multiplicity of the rounded target minus the binary64 multiplicity",
                                   fc["exact_of_rounded_target"] - fc["binary64"])]},
                counterexample={"statement": "Converting a target to binary64 preserves its shortest-representative "
                                            "multiplicity",
                                "witness": {"target": "(1/2 + 2^-60, 1/4)", "exact_multiplicity": fc["exact"],
                                            "binary64_target": [float(Fraction(1, 2) + Fraction(1, 2 ** 60)), 0.25],
                                            "binary64_multiplicity": fc["binary64"]}},
                uncertainty=EXACT_UNC, tolerance=EXACT),
        finding("At exact cut-locus ties that binary64 cannot represent, raw float comparison undercounts some "
                "multiplicities while a relative tolerance of 1e-9 recovers all of them", "numerical",
                {"ties": len(ties), "binary64_undercounts": study["binary64_undercounts"],
                 "tolerance_undercounts": study["tolerance_undercounts"]},
                {"checks": [_check("ties undercounted by raw binary64 comparison", study["binary64_undercounts"], 1,
                                   "ge"),
                            _check("ties undercounted with the declared tolerance", study["tolerance_undercounts"])]},
                counterexample={"statement": "Floating-point distance comparison finds every shortest representative",
                                "witness": next((t for t in ties if t["binary64"] < t["exact"]), {})},
                uncertainty=EXACT_UNC, tolerance=EXACT),
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
    basins.sort(key=lambda b: (-b["measured_width"], b["length"], b["winding"]))
    # Order test on the measured widths: a pair of distinct lengths is discordant when the shorter class has a
    # narrower measured basin by at least one grid step; equal grid counts are unresolved, not concordant.
    discordant = unresolved = 0
    for i, a in enumerate(basins):
        for b in basins[i + 1:]:
            if abs(a["length"] - b["length"]) <= 1e-12:
                continue
            short, long = (a, b) if a["length"] < b["length"] else (b, a)
            if long["measured_width"] - short["measured_width"] >= step - 1e-15:
                discordant += 1
            elif abs(long["measured_width"] - short["measured_width"]) < step - 1e-15:
                unresolved += 1
    sample = [{"theta": float(thetas[i]), "return_distance": float(r[i])} for i in range(0, HEADINGS // 2, 10)]
    return {"predicted": sorted([list(v) for v in predicted]), "found": sorted([list(v) for v in zeros]),
            "missing": sorted([list(v) for v in predicted - set(zeros)]),
            "extra": sorted([list(v) for v in set(zeros) - predicted]), "positive_minima": positive,
            "slope_relative_error": slope_error, "basins": basins, "width_error": width_error, "step": step,
            "epsilon": epsilon, "discordant": discordant, "unresolved": unresolved, "sample": sample}


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
        "the exact direction arg v; slopes probed at +-1e-4 rad; basins measured on the grid and ordered by "
        "their measured widths.",
        f"{len(study['found'])} zero minima found, {len(study['predicted'])} predicted, missing {study['missing']}, "
        f"extra {study['extra']}; {study['positive_minima']} positive (near-miss) minima; slope relative error "
        f"{study['slope_relative_error']:.2e}; widest measured basins {[b['winding'] for b in top]}; "
        f"{study['discordant']} discordant and {study['unresolved']} unresolved (same grid count) pairs of distinct "
        f"lengths.",
        f"Grid spacing {study['step']:.2e} rad bounds measured basin widths (max deviation "
        f"{study['width_error']:.2e} rad); refined zeros are exact to rounding.",
        ["near-miss minima (segment endpoints) separated from true closures",
         "non-primitive multiples on the same ray", "grid resolution against the narrowest basin"],
        ["Heading sensitivity is defined through the closest return within L; other definitions (e.g. Lyapunov "
         "exponents, zero here) would rank headings differently."],
        "Deferred research question: rank the closing headings under a second definition of heading sensitivity (for "
        "example the measure of headings within +-delta whose closest return within L stays below a threshold) and "
        "report where that ranking differs from the closest-return definition used here")
    findings = [
        finding("Zeros of the return distance (tau = 0.31 + 1.07i, L = 3) are exactly the primitive lattice "
                "directions with |v| <= L",
                "numerical", {"found": len(study["found"]), "predicted": len(study["predicted"])},
                {"checks": [_check("missing plus extra closure directions",
                                   len(study["missing"]) + len(study["extra"]))]},
                uncertainty=_unc("truncation_bound", study["step"],
                                 "heading grid spacing (rad); each zero refined at the exact direction arg v"),
                tolerance=EXACT),
        finding("Near each closing heading the closing error grows at rate |v| (the loop length) per radian",
                "numerical", {"max_relative_slope_error": study["slope_relative_error"]},
                {"derivation": "r = |v| sin|theta - theta_v| near theta_v",
                 "checks": [_check("|r / sin(delta) - |v|| / |v| at delta = 1e-4", study["slope_relative_error"],
                                   1e-9, kind="analytic")]},
                uncertainty=_unc("roundoff", study["slope_relative_error"],
                                 "relative error of r / sin(delta) against |v| at delta = 1e-4"),
                tolerance={"abs": 1e-9, "rel": 0}),
        finding("Closure basins are widest for the shortest (lowest-order) primitive classes", "numerical",
                {"widest": [{"winding": b["winding"], "length": b["length"], "measured_width": b["measured_width"],
                             "basin_width": b["basin_width"]} for b in top],
                 "discordant_pairs": study["discordant"], "unresolved_pairs": study["unresolved"]},
                {"checks": [_check("pairs of distinct lengths whose shorter class has a measured basin narrower by at "
                                   "least one grid step", study["discordant"]),
                            _check("measured minus analytic basin width (rad)", study["width_error"],
                                   2 * study["step"] + 1e-12, kind="analytic")]},
                uncertainty=_unc("truncation_bound", 2 * study["step"],
                                 "grid quantization of measured basin widths (rad)"),
                tolerance=TIGHT),
        finding("A physical heading sensor closes the shortest loop within the computed heading tolerance",
                "sensor_performance", None, {}),
    ]
    return {"state": "completed", "fields": fields, "findings": findings}


# ================================================================ T024 / T025
TORUS = Torus(2.0, 1.0)
ROUTE_P, ROUTE_Q = (0.0, 0.4), (2.0, -0.3)
ROUTE_MAX, HORIZON = 11.0, 8.0
ROUTE_HEADINGS = 1440


def torus_routes():
    found, counts = routes.find_routes(TORUS, ROUTE_P, ROUTE_Q, ROUTE_MAX, headings=ROUTE_HEADINGS, horizon=HORIZON)
    convergence = routes.fan_convergence(TORUS, ROUTE_P, ROUTE_Q, ROUTE_MAX, found, ROUTE_HEADINGS)
    return {"routes": found, "counts": counts, "convergence": convergence}


def _independent_routes(ctx, surface, p, q, route_list):
    """scipy DOP853 on the sympy-derived field; None unless both optional modules are available."""
    scipy_version, sympy_version = _version(ctx, "scipy"), _version(ctx, "sympy")
    if not (scipy_version and sympy_version):
        return None
    result = routes.compare_with_scipy(surface, p, q, route_list, extra=HORIZON)
    result["revision"] = f"scipy {scipy_version}, sympy {sympy_version}"
    return result


def _attach_independent_routes(basis, sc):
    """Presence and solver-status mismatches are exact checks; matched quantities feed the independent check."""
    if sc is None:
        return
    basis["checks"] += [
        _check("routes where exactly one of ciw.lab.jacobi and scipy/sympy finds a conjugate point within the horizon",
               sc["presence_mismatches"], kind="high_precision"),
        _check("scipy solve_ivp runs that stopped before L + horizon", sc["failed"], kind="high_precision")]
    basis["independent_check"] = _independent(
        _check("scipy DOP853 on a sympy-derived field (metric, Christoffel symbols, K and heading frame from the "
               "embedding): max of relative j_head(L) difference, conjugate-point difference and endpoint distance "
               "to the lift of q", max(sc["j_head"], sc["conjugate"], sc["endpoint"]), 1e-5, kind="high_precision"),
        "scipy.integrate.solve_ivp (sympy-derived field)", sc["revision"])


def _route_table(found):
    return [{"length": r["length"], "heading": r["heading"], "lift": r["lift"], "amplification": r["amplification"],
             "targeting_condition": r["targeting_condition"], "focus_margin": r["focus_margin"],
             "margin_lower_bound": r["margin_lower_bound"],
             "passes_conjugate_point": r["focus_margin"] is not None and r["focus_margin"] < 0} for r in found]


ROUTE_SET_UNC = {"kind": "truncation_bound", "value": None,
                 "basis": "completeness of the fan search is not proven: no bound on routes missed at both fan "
                          "densities; routes are matched by length and heading to 1e-6"}


AMPLIFICATION_CAVEAT = ("Small |j_head(L)| is not robustness: the targeting condition number 1/|j_head(L)| grows as "
                        "q approaches a conjugate point, and a route with negative focus margin is not locally "
                        "minimizing.")


@task("T024", changed_files=(MODULE, ROUTES, DOC),
      regression_tests=_tests("test_t024_t025_route_ranking_and_front", "test_route_helpers",
                              "test_focus_margin_claims_state_the_canonical_definition",
                              "test_independent_disagreement_makes_t024_partial_not_blocked",
                              "test_sympy_field_matches_the_closed_form_batch_field"))
def focus_margin_ranking(ctx):
    data = ctx.memo("flat-torus/t024-routes", torus_routes)
    found, counts, conv = data["routes"], data["counts"], data["convergence"]
    ranks = routes.rankings(found)
    sc = _independent_routes(ctx, TORUS, ROUTE_P, ROUTE_Q, found)
    ctx.artifact_json("torus-routes.json", {
        "p": ROUTE_P, "q": ROUTE_Q, "max_length": ROUTE_MAX, "horizon": HORIZON, "headings": ROUTE_HEADINGS,
        "newton_counts": counts, "convergence": conv, "routes": found, "rankings": ranks,
        "scipy_sympy": None if sc is None else {k: v for k, v in sc.items()}})
    residual = max(r["endpoint_residual"] for r in found)
    wronskian = max(r["wronskian_drift"] for r in found)
    clairaut = max(r["clairaut_drift"] for r in found)
    batch = max(r["j_head_batch_vs_transfer"] / max(1.0, r["amplification"]) for r in found)
    shortest = ranks["by_length"][0]
    amp_rank = ranks["by_amplification"].index(shortest)
    margin_rank = next(i for i, group in enumerate(ranks["by_focus_margin"]) if shortest in group)
    censored = ranks["by_focus_margin"][0] if found[ranks["by_focus_margin"][0][0]]["focus_margin"] is None else []
    flat = jacobi.transfer(Plane(), [0.0, 0.0], 0.3, 20.0, steps=40)
    flat_min = float(np.min(flat.states[1:, 6] / flat.s[1:]))
    fields = _fields(
        "Between two points of the torus of revolution the ranking of geodesic routes by length, by heading "
        "amplification |j_head(L)| and by focus margin s_c - L (s_c the first conjugate point of p, the first zero "
        "of j_head) disagree; on a flat torus there are no conjugate points (infinite margin).",
        "Geodesic + heading Jacobi system on Torus(R=2, r=1), chart (phi, theta); K = cos theta / (r (R + r cos "
        "theta)); focus margin = s_c - L for the first zero s_c > 0 of j_head along the extended geodesic "
        "(negative: the route passes a conjugate point and is not locally minimizing); targeting condition "
        "1/|j_head(L)|.",
        [f"p = {ROUTE_P}, q = {ROUTE_Q} (chart), routes up to length {ROUTE_MAX}, extension horizon {HORIZON}",
         f"{ROUTE_HEADINGS}-heading fan (convergence rerun at {2 * ROUTE_HEADINGS}), RK4 batch step 0.025, Newton on "
         "(heading, length) with Jacobian [j_head N, v]"],
        NO_PHYSICAL,
        "Endpoint lies on a lift of q; Wronskian det Phi = 1; Clairaut rho^2 phi' conserved; batch and "
        "ciw.lab.jacobi amplifications agree; the route set does not change when the fan density doubles.",
        "Fan search for near-passes of every chart lift of q, one Newton seed per miss-distance minimum along "
        "each run of adjacent rays, batch Newton refinement, deduplication, re-integration of each route with "
        "ciw.lab.jacobi.transfer to L + horizon, conjugate points by Hermite zeros; the search is repeated at "
        "twice the fan density; optional scipy DOP853 integration of a field derived by sympy from the embedding.",
        f"{len(found)} routes from {counts['fan_candidates']} fan near-passes and {counts['newton_seeds']} Newton "
        f"seeds (drops: {counts['singular']} singular, {counts['diverged']} diverged, {counts['residual_fail']} "
        f"residual, {counts['over_length']} over length; {counts['duplicates_merged']} duplicates merged); at "
        f"{2 * ROUTE_HEADINGS} headings "
        f"{conv['routes'][1]} routes, {conv['differences']} differences; order by length {ranks['by_length']}, by "
        f"amplification {ranks['by_amplification']}, focus-margin groups (best first; censored routes tied) "
        f"{ranks['by_focus_margin']}.",
        f"Endpoint residual <= {residual:.1e}; Wronskian drift <= {wronskian:.1e}; Clairaut drift <= "
        f"{clairaut:.1e}; batch-vs-transfer amplification difference <= {batch:.1e} (relative)"
        + ("." if sc is None else f"; scipy/sympy re-integration: j_head {sc['j_head']:.1e} (relative), conjugate "
                                  f"points {sc['conjugate']:.1e}, endpoint {sc['endpoint']:.1e}."),
        ["duplicate routes merged (counted)", "routes beyond the length budget discarded (counted)",
         "censored margins (no conjugate point within the horizon) tied as one group",
         "Newton divergence or singular Jacobian (counted)", "fan density (route set unchanged at double density)"],
        ["The candidate set is what the fan found up to length 11 and is stable under doubling the density; "
         "completeness is not proven.",
         "Focus margin uses conjugate points of p only; conjugate points of q along the reversed route are not "
         "separately reported.", AMPLIFICATION_CAVEAT],
        "Deferred research question: report each route's focus margin at q as well (conjugate points of q along the "
        "reversed route) and bound the completeness of the route set up to length 11, for example by interval "
        "shooting over all headings (the fan search does not prove completeness)")
    basis = {"checks": [_check("endpoint residual to the lift of q", residual, 1e-6, "le", kind="invariant"),
                        _check("Wronskian drift", wronskian, 1e-8, "le", kind="invariant"),
                        _check("Clairaut integral drift", clairaut, 1e-6, "le", kind="invariant"),
                        _check("batch RK4 vs ciw.lab.jacobi amplification", batch, 1e-4, "le",
                               kind="cross_implementation")]}
    _attach_independent_routes(basis, sc)
    table = _route_table(found)
    findings = [
        finding("Every fan-search route p -> q on Torus(2, 1) ends on a lift of q with the Wronskian and the "
                "Clairaut integral conserved to tolerance", "numerical", table, basis,
                uncertainty=_unc("reference_error", max(batch, sc["j_head"] if sc else 0.0,
                                                        sc["conjugate"] if sc else 0.0),
                                 "max of the relative amplification difference against the batch RK4 search and, "
                                 "when scipy and sympy are available, the relative j_head(L) difference and the "
                                 "absolute conjugate-point difference (bounding the focus margins, length units) "
                                 "against the scipy/sympy integration; RK4 step about 0.04 in ciw.lab.jacobi"),
                tolerance=ROUTE),
        finding(f"The torus route set is unchanged when the fan density doubles ({ROUTE_HEADINGS} to "
                f"{2 * ROUTE_HEADINGS} headings)", "numerical",
                {"headings": conv["headings"], "routes": conv["routes"], "differences": conv["differences"]},
                {"checks": [_check("routes found at only one of the two densities", conv["differences"],
                                   kind="self_convergence")]},
                uncertainty=ROUTE_SET_UNC, tolerance=EXACT),
        finding("Rankings by length, amplification |j_head(L)| and focus margin s_c - L (s_c the first zero of "
                "j_head) disagree: the shortest route is neither the least amplifying nor in the best focus-margin "
                "group", "numerical", ranks,
                {"checks": [_check("rank of the shortest route by amplification (0 = least amplifying)", amp_rank,
                                   1, "ge"),
                            _check("focus-margin group of the shortest route (0 = best group)", margin_rank, 1, "ge")]},
                counterexample={"statement": "The shortest route also minimizes amplification and maximizes focus "
                                             "margin s_c - L",
                                "witness": {"shortest": table[shortest],
                                            "least_amplification": table[ranks["by_amplification"][0]],
                                            "best_margin_group": [table[i] for i in ranks["by_focus_margin"][0]]}},
                uncertainty={"kind": "exact", "value": 0.0, "basis": "orderings of the route values; adjacent values "
                                                                     "differ far more than their uncertainty"},
                tolerance=EXACT),
        finding("A flat torus has no conjugate points: j_head(s) = s > 0 has no zero, so the focus margin s_c - L is "
                "infinite", "numerical",
                flat_min, {"derivation": "K = 0 gives j_head(s) = s",
                           "checks": [_check("min j_head(s)/s - 1 on (0, 20] (ciw.lab.jacobi, plane)", flat_min - 1.0,
                                             1e-12, kind="analytic")]},
                uncertainty=_unc("roundoff", abs(flat_min - 1.0), "RK4 on the linear flat Jacobi system"),
                tolerance=TIGHT),
        finding("Ranking routes by focus margin s_c - L selects a route that is safe to execute on a physical part",
                "machine_safety", None, {}),
    ]
    if censored:
        fields["numerical_result"] += (f" Routes {censored} have no conjugate point within the horizon (margins "
                                       f"only known to exceed about {HORIZON:g}), so their margins are tied.")
    return {"state": "completed", "fields": fields, "findings": findings}


@task("T025", changed_files=(MODULE, ROUTES, DOC),
      regression_tests=_tests("test_t024_t025_route_ranking_and_front", "test_route_helpers",
                              "test_focus_margin_claims_state_the_canonical_definition"))
def pareto_fronts(ctx):
    data = ctx.memo("flat-torus/t024-routes", torus_routes)
    found = data["routes"]
    front = routes.pareto_front(found)
    pairs = {}
    for name, key in (("length_amplification", lambda r: (r["length"], r["amplification"])),
                      ("length_margin", lambda r: (r["length"], -routes.margin_value(r)))):
        keys = [key(r) for r in found]
        pairs[name] = [i for i, k in enumerate(keys)
                       if not any(all(x <= y for x, y in zip(o, k)) and o != k for o in keys)]
    # Second formulations sharing no code with pareto_front / dominates / margin_value: a numpy dominance
    # matrix for three objectives and a sort-and-sweep for each pair, on objectives read from the route fields.
    objectives = routes.objective_matrix(found)
    second = {"three_objective": routes.front_by_dominance_matrix(objectives),
              "length_amplification": routes.front_by_sweep(objectives[:, [0, 1]].tolist()),
              "length_margin": routes.front_by_sweep(objectives[:, [0, 2]].tolist())}
    front_difference = len(set(front) ^ set(second["three_objective"]))
    pair_difference = sum(len(set(pairs[name]) ^ set(second[name])) for name in pairs)
    table = _route_table(found)
    past = [i for i in front if table[i]["passes_conjugate_point"]]
    ctx.artifact_json("pareto.json", {"front": front, "fronts_2d": pairs, "front_members_past_conjugate_point": past,
                                      "second_computation": second, "routes": table})
    order = sorted(range(len(found)), key=lambda i: found[i]["length"])
    fsorted = sorted(pairs["length_amplification"], key=lambda i: found[i]["length"])
    ctx.artifact_text("pareto-length-amplification.svg", svg.line_plot(
        [("all routes (by length)", [found[i]["length"] for i in order], [found[i]["amplification"] for i in order]),
         ("length/amplification front", [found[i]["length"] for i in fsorted],
          [found[i]["amplification"] for i in fsorted])],
        title="Torus routes: length vs heading amplification", xlabel="route length L",
        ylabel="|j_head(L)|", logy=True))
    fields = _fields(
        "Length, amplification and focus margin conflict, so the three-objective Pareto front of the torus "
        "routes has several members.",
        "Minimize (L, |j_head(L)|, -margin) with margin the focus margin s_c - L (s_c the first zero of j_head) and "
        "censored margins treated as +infinity; a dominates b when no "
        "worse in every objective and better in one. The shortest route is on the front by definition (nothing "
        "is shorter), so its membership is not a test.",
        ["T024 routes (same run, shared memo)"], NO_PHYSICAL,
        "A second computation of every front, sharing no code with the first, gives the same members.",
        "Exhaustive pairwise dominance on the T024 route set; two-objective fronts for length/amplification and "
        "length/margin; every front recomputed from the route fields by a numpy dominance matrix (three "
        "objectives) and a sort-and-sweep (two objectives); retained JSON and SVG.",
        f"3-objective front {front} of {len(found)} routes (members past a conjugate point: {past}); "
        f"length/amplification front {pairs['length_amplification']}; length/margin front {pairs['length_margin']}; "
        f"targeting condition 1/|j_head| of front members "
        f"{[round(table[i]['targeting_condition'], 3) for i in front]}.",
        "Inherits T024 route uncertainty (<= 1e-5 relative); front membership is exact given the values.",
        ["ties in objectives", "censored margins", "front membership recomputed by a second formulation",
         "front members past a conjugate point flagged"],
        ["Objectives are unweighted; a decision needs weights or constraints that the workbench does not set.",
         AMPLIFICATION_CAVEAT],
        "Deferred research question: which declared constraints (for example a lower bound on the focus margin "
        "s_c - L and an upper bound on the targeting condition 1/|j_head(L)| derived from a declared heading error) "
        "leave a unique route on the front, and is that choice stable under the route uncertainty? Choosing the "
        "constraints is an operator decision the workbench does not make")
    findings = [
        finding("The three-objective front (length, |j_head(L)|, focus margin s_c - L with s_c the first zero of "
                "j_head) has several members, and a second computation (numpy dominance matrix; sort-and-sweep for "
                "the two-objective fronts) reproduces every front", "numerical",
                {"front": front, "fronts_2d": pairs, "routes": len(found), "front_members_past_conjugate_point": past},
                {"derivation": "Every route off the front is dominated by a front member because strict dominance "
                               "is transitive and acyclic on a finite set (definitional, not tested); the shortest "
                               "route is always on the front: no route is strictly shorter, so none dominates it "
                               "(exact length ties aside)",
                 "checks": [_check("three-objective front members differing from a numpy dominance matrix",
                                   front_difference, kind="cross_implementation"),
                            _check("two-objective front members differing from a sort-and-sweep", pair_difference,
                                   kind="cross_implementation"),
                            _check("front size", len(front), 2, "ge")]},
                uncertainty={"kind": "exact", "value": 0.0, "basis": "exhaustive dominance on the T024 values (which "
                                                                     "carry their own uncertainty)"},
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
    wrong_rule, wrong_witness = 0, None
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
                if lat.quad(H, *naive) != lat.quad(G, m, n):
                    wrong_rule += 1
                    if wrong_witness is None:
                        wrong_witness = {"lattice": name, "gram": [str(x) for x in G],
                                         "matrix": [list(r) for r in M], "winding": [m, n],
                                         "transformed_gram": [str(x) for x in H],
                                         "q_g_of_winding": str(lat.quad(G, m, n)),
                                         "naive_label": list(naive), "q_h_of_naive_label": str(lat.quad(H, *naive)),
                                         "correct_label": list(transported),
                                         "q_h_of_correct_label": str(lat.quad(H, *transported))}
    # Float: area-one shapes under all the same matrices; compare the first 30 lengths.
    float_dev = 0.0
    float_entry = max(max(abs(x) for row in M for x in row) for M in matrices)
    for tau in FTR_SHAPES:
        base = sorted(math.sqrt(q) for q, _, _ in lat.lattice_vectors(lat.area_one_form(tau), 16.0))[:30]
        for M in matrices:
            image = lat.mobius(M, tau)
            other = sorted(math.sqrt(q) for q, _, _ in lat.lattice_vectors(lat.area_one_form(image), 16.0))[:30]
            float_dev = max(float_dev, max(abs(a - b) / a for a, b in zip(base, other)))
    generic = INVARIANCE_LATTICES["generic"]
    mirror = lat.transform_any(generic, ((1, 0), (0, -1)))
    doubled = lat.transform_any(generic, ((2, 0), (0, 1)))
    return {"matrices": len(matrices), "failures": failures, "wrong_rule_mismatches": wrong_rule,
            "wrong_rule_witness": wrong_witness, "float_max_relative_deviation": float_dev, "float_max_matrix_entry": float_entry,
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
                              "test_provider_output_is_refused_when_unreadable_or_incomplete",
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
        "Exact equality of spectra, det and systole; area-one float spectra equal within 1e-8 (limited by the "
        "conditioning of skewed bases).",
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
        "Deferred research question: find the word length at which binary64 area-one spectra of SL(2,Z) images first "
        "disagree with the exact spectrum beyond tolerance (matrix entries grow with the word length; 10-letter "
        "words are used here), and extend the exact comparison beyond R^2 = 60")
    findings = [
        finding("Length spectrum (R^2 <= 60), area and systole are exactly invariant under every tested SL(2,Z) "
                "change of basis and under reduction", "mathematical",
                {"matrices": study["matrices"], "failures": failures},
                {"checks": [_check("exact spectrum/area/systole/winding-transport failures", sum(failures.values()))]},
                uncertainty=EXACT_UNC, tolerance=EXACT),
        finding(f"Area-one float spectra (first 30 lengths) agree within 1e-8 under all {study['matrices']} basis "
                "changes, a conditioning-limited agreement rather than unit roundoff", "numerical",
                study["float_max_relative_deviation"],
                {"checks": [_check("max relative deviation of the first 30 lengths (bases with entries up to "
                                   f"{study['float_max_matrix_entry']})", study["float_max_relative_deviation"],
                                   1e-8, kind="invariant")]},
                uncertainty=_unc("roundoff", study["float_max_relative_deviation"],
                                 f"measured relative deviation of binary64 lengths from Mobius images under skewed "
                                 f"bases (entries up to {study['float_max_matrix_entry']}); far above unit roundoff "
                                 "because the skew amplifies rounding"),
                tolerance={"abs": 1e-8, "rel": 0}),
        finding("Transporting winding labels with M instead of M^-1 breaks length invariance", "mathematical",
                study["wrong_rule_mismatches"],
                {"checks": [_check("naive-rule mismatches", study["wrong_rule_mismatches"], 1, "ge")]},
                counterexample={"statement": "Winding labels transform with the same matrix as the basis",
                                "witness": dict(study["wrong_rule_witness"] or {}, correct_rule="c' = M^-1 c",
                                                mismatches=study["wrong_rule_mismatches"])},
                uncertainty=EXACT_UNC, tolerance=EXACT),
        finding("The length spectrum does not determine the oriented shape: a mirror image is isospectral but not "
                "SL(2,Z)-equivalent", "mathematical", mirror,
                {"checks": [_check("isospectral mirror (0 = same spectrum)", 0 if mirror["same_spectrum"] else 1),
                            _check("mirror canonical form equal (0 = different)", 1 if mirror["same_canonical"] else 0)]},
                counterexample={"statement": "Equal length spectra imply SL(2,Z)-equivalent oriented lattices",
                                "witness": mirror}, uncertainty=EXACT_UNC, tolerance=EXACT),
        finding("A det-2 integer matrix changes area and spectrum (an index-2 sublattice, not a basis change)",
                "mathematical", sub,
                {"derivation": "det(M^T G M) = det(M)^2 det G: a det-2 matrix multiplies the squared area by 4",
                 "checks": [_check("spectrum unchanged (0 = changed)", 1 if sub["same_spectrum"] else 0)]},
                uncertainty=EXACT_UNC, tolerance=EXACT),
    ]
    run = _provider(ctx)
    if _provider_ran(run):
        cmp = fold_comparison(run)
        ctx.artifact_json("ftr-fold-lengths.json", cmp)
        identity = run["identity"]
        invariance = max(abs(r["ftr_length_pair"][0] - r["ftr_length_pair"][1]) / r["ftr_length_pair"][0]
                         for r in cmp["rows"])
        findings.append(finding(
            FTR_CLAIMS["T026"], "numerical",
            {"cases": len(cmp["rows"]), "max_relative_difference": cmp["max_length_relative_difference"],
             "winding_mismatches": cmp["winding_mismatches"]},
            {"provider": ftr.provider_basis(identity),
             "checks": [_check("FTR reduced-winding mismatches (sign-consistent)", cmp["winding_mismatches"]),
                        _check("FTR length_pair invariance", invariance, 1e-9, kind="invariant")],
             "independent_check": _independent(
                 _check("FTR fold length_pair against ciw lengths", cmp["max_length_relative_difference"], 1e-9,
                        kind="high_precision"), ftr.checker_identity(identity)["implementation"], identity["revision"])},
            uncertainty=_unc("roundoff", cmp["max_length_relative_difference"],
                             "relative difference of binary64 lengths before and after the fold"),
            tolerance={"abs": 1e-9, "rel": 0}))
    else:
        findings.append(_provider_placeholder("T026", run))
    state = _provider_note(run, fields, (MODULE, LATTICE, PROVIDER))
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
        "Deferred research question: predict the cylinder decomposition of the L-shape and the regular octagon in "
        "every periodic direction up to a saddle-connection length bound (for the octagon through its Veech group "
        "action), not only the horizontal one, and compare it with the periodic directions traced in T028")
    findings = [
        finding("L-shape and regular octagon are genus-2 translation surfaces with one vertex class", "mathematical",
                records,
                {"checks": [_check("genus mismatches (L 2, octagon 2, square torus 1, hexagon 1)", 0 if genus_ok else 1),
                            _check("vertex classes of the L-shape and the octagon that differ from 1",
                                   (records["L-shape"]["vertices"] != 1) + (records["octagon"]["vertices"] != 1)),
                            _check("octagon area minus 2 + 2 sqrt 2 (exact)", 0 if area_error.sign() == 0 else 1)]},
                uncertainty=EXACT_UNC, tolerance=EXACT),
        finding("Horizontal cylinders: L-shape circumferences equal the r-cycle lengths, octagon strips have "
                "circumferences 2 + sqrt 2 and 1 + sqrt 2 with areas summing to the surface area", "mathematical",
                {"l_shape": cyl["l_shape_circumferences"], "octagon": [float(cyl["octagon_outer"]),
                                                                      float(cyl["octagon_middle"])]},
                {"checks": [_check("L-shape circumferences minus r-cycles", max(abs(a - b) for a, b in zip(
                    cyl["l_shape_circumferences"], cyl["l_shape_r_cycles"]))),
                            _check("octagon circumference mismatch (exact surd)", 1 if circ_error else 0),
                            _check("octagon cylinder area minus surface area (exact surd)",
                                   0 if cyl_error.sign() == 0 else 1)]}, uncertainty=EXACT_UNC, tolerance=TIGHT),
        finding("Pairings that are not translations, or glue unequal edges, are refused as translation surfaces",
                "mathematical", codes,
                {"checks": [_refusal("octagon adjacent pairing", "GLUING_NOT_TRANSLATION",
                                     codes["octagon adjacent pairing"]),
                            _refusal("pillowcase (half-translation)", "GLUING_NOT_TRANSLATION", codes["pillowcase"]),
                            _refusal("edges of length 1 and 2 glued", "EDGE_LENGTH_MISMATCH",
                                     codes["mismatched edge lengths"])]}, uncertainty=EXACT_UNC, tolerance=EXACT),
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
    float_start, float_direction = (1 / 3, 1 / 5), (math.cos(theta), math.sin(theta))
    generic = F.flow(0, float_start, float_direction, max_crossings=400, stop_on_return=True, record_path=True)
    # Exact Q(sqrt 2) trace of the same binary64 start and direction (dyadic rationals) on the exact octagon:
    # it measures what binary64 arithmetic and the rounded vertices do to the float trajectory.
    reference = O.flow(0, tuple(S(Fraction(x)) for x in float_start), tuple(S(Fraction(x)) for x in float_direction),
                       max_crossings=400, stop_on_return=True, record_path=True)
    deviation = max(max(math.hypot(a[0] - float(b[0]), a[1] - float(b[1])) for a, b in ((fa, ea), (fb, eb)))
                    for (_, fa, fb), (_, ea, eb) in zip(generic["path"], reference["path"]))
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
            "generic_exact_closed": reference["closed"], "generic_exact_crossings": len(reference["crossings"]),
            "generic_same_crossing_sequence": generic["crossings"] == reference["crossings"],
            "generic_max_position_deviation": deviation,
            "generic_min_vertex_distance": O.min_vertex_distance(reference["path"]),
            "generic_float_min_vertex_distance": F.min_vertex_distance(generic["path"]),
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
    traced = len(lflows["rows"])
    tol = oflows["tolerance"]
    distance_margin = oflows["generic_min_vertex_distance"] - tol - oflows["generic_max_position_deviation"]
    fields = _fields(
        "Straight-line flow crosses glued edges by the edge translations: on the square-tiled L every rational "
        "direction is completely periodic or ends in a saddle connection (exactly decided), and on the octagon a "
        "float trajectory reproduces the exact Q(sqrt 2) trajectory within a declared tolerance.",
        "Flow x + t d inside a polygon; exit edge by exact segment intersection; re-entry at x + (a' - b) on the "
        "partner edge; a hit on a cone point terminates (saddle connection); closure multiplier k with displacement "
        "k (p, q) in Z^2 and 1 <= k <= 3 on the 3-square surface (Veech dichotomy).",
        [f"L-shape primitive directions |p|, |q| <= 3 from starts {[(s, tuple(map(str, x))) for s, x in L_STARTS]}",
         "octagon direction (3, 1 + sqrt 2) exact and float; generic float direction theta = 0.3 rad, traced "
         "again exactly in Q(sqrt 2) from the same binary64 start and direction",
         f"float tolerance {oflows['tolerance']}"],
        NO_PHYSICAL,
        "Exact outcome for every rational L-shape trajectory; float octagon crossings identical to exact ones.",
        "Exact Fraction flow per direction and start; exact surd vs float flow on the octagon; the generic float "
        "trajectory compared crossing by crossing with its exact Q(sqrt 2) trace, with the Euclidean distance of "
        "the exact trajectory to every vertex; tolerance-aware termination near vertices.",
        f"L-shape: {closed} closed, {saddles} saddle connections, {lflows['undecided']} undecided, "
        f"{lflows['bad_multiplier']} bad multipliers; octagon (3, 1 + sqrt 2): exact closure time "
        f"{oflows['exact_time']} (units of the direction vector; length {oflows['exact_length']:.12f}) after "
        f"{oflows['exact_crossings']} crossings, float difference "
        f"{oflows['time_difference']:.1e}; generic theta: {oflows['generic_crossings']} crossings with the exact "
        f"trace's crossing sequence, max position deviation {oflows['generic_max_position_deviation']:.1e}, "
        f"closest Euclidean approach to a vertex {oflows['generic_min_vertex_distance']:.4e} (along-edge clearance "
        f"at the crossings {oflows['generic_min_clearance']:.4e}).",
        "Exact for rational and Q(sqrt 2) data; float trajectories carry rounding growing with crossings, measured "
        "against the exact trace (declared tolerance 1e-9).",
        ["saddle connection from an exact vertex hit refused", "float pass within tolerance of a vertex refused",
         "float start within tolerance of an edge refused", "closure at a return inside the start polygon",
         "half-translation surfaces refused for flow",
         "binary64 drift of a generic trajectory measured against its exact trace",
         "Euclidean closest approach to a vertex (smaller than the along-edge clearance)"],
        ["A generic float direction is never certified non-periodic; only its first 400 crossings are traced."],
        "Deferred research question: certify non-periodicity of an algebraic direction exactly (for example a "
        "quadratic-irrational slope on a square-tiled surface, traced in exact arithmetic over Q(sqrt d)) instead of "
        "tracing the first 400 crossings of a float direction")
    findings = [
        finding(f"All {traced} tested rational trajectories on the L-shape close (multiplier 1-3) or end in a "
                "saddle connection",
                "mathematical", {"closed": closed, "saddle_connections": saddles, "undecided": lflows["undecided"]},
                {"derivation": "Veech dichotomy for square-tiled surfaces: rational directions are completely periodic",
                 "checks": [_check("undecided trajectories", lflows["undecided"]),
                            _check("closure multipliers outside {1, 2, 3}", lflows["bad_multiplier"])]},
                uncertainty=EXACT_UNC, tolerance=EXACT),
        finding("Trajectories hitting a cone point are terminated as saddle connections, exactly or within the "
                "declared float tolerance", "numerical",
                {"exact_l_shape": saddle, "exact_octagon": oflows["exact_vertex_code"],
                 "float_octagon": oflows["near_vertex_code"]},
                {"checks": [_refusal("L-shape (1/2, 1/2) direction (1, 1)", "SADDLE_CONNECTION", saddle),
                            _refusal("octagon exact centre-to-vertex", "SADDLE_CONNECTION", oflows["exact_vertex_code"]),
                            _refusal("octagon float pass 1e-12 from a vertex", "NEAR_VERTEX_WITHIN_TOLERANCE",
                                     oflows["near_vertex_code"])]},
                uncertainty=EXACT_UNC, tolerance=EXACT),
        finding("A float start within the declared tolerance of a polygon edge is refused as not interior",
                "numerical", {"float_start": oflows["start_code"]},
                {"checks": [_refusal("float octagon start 1e-12 from an edge", "START_NOT_INTERIOR",
                                     oflows["start_code"])]},
                uncertainty=EXACT_UNC, tolerance=EXACT),
        finding("Float octagon flow reproduces the exact Q(sqrt 2) trajectory (same crossings, closure time)",
                "numerical", {"time_difference": oflows["time_difference"], "crossings": oflows["exact_crossings"]},
                {"checks": [_check("crossing sequences differ", 0 if oflows["same_crossing_sequence"] else 1),
                            _check("|float - exact| closure time", oflows["time_difference"], 1e-9,
                                   kind="cross_implementation")]},
                uncertainty=_unc("roundoff", oflows["time_difference"],
                                 "binary64 accumulation over the crossings against the exact Q(sqrt 2) trajectory"),
                tolerance={"abs": 1e-9, "rel": 0}),
        finding("A generic float octagon trajectory follows the exact Q(sqrt 2) trace of its binary64 start and "
                "direction for 400 crossings and stays farther than the tolerance from every vertex", "numerical",
                {"crossings": oflows["generic_crossings"],
                 "max_position_deviation": oflows["generic_max_position_deviation"],
                 "min_euclidean_vertex_distance": oflows["generic_min_vertex_distance"],
                 "min_along_edge_clearance": oflows["generic_min_clearance"]},
                {"checks": [_check("crossing sequence differs from the exact trace (0 = identical, both unclosed)",
                                   0 if oflows["generic_same_crossing_sequence"] and not oflows["generic_closed"]
                                   and not oflows["generic_exact_closed"] else 1),
                            _check("max position deviation from the exact trace at the crossings",
                                   oflows["generic_max_position_deviation"], tol, "le"),
                            _check("closest Euclidean approach of the exact trajectory to a vertex minus (tolerance + "
                                   "max position deviation)", distance_margin, 0.0, "signed_ge")]},
                uncertainty=_unc("roundoff", oflows["generic_max_position_deviation"],
                                 "measured max position deviation of the binary64 trajectory (rounded vertices and "
                                 "arithmetic) from its exact Q(sqrt 2) trace over 400 crossings"),
                tolerance={"abs": 1e-9, "rel": 1e-6}),
    ]
    return {"state": "completed", "fields": fields, "findings": findings}


# Euler characteristics known independently of the vertex classes: the declared topology of each polygon
# gluing, and for origamis Riemann-Hurwitz over the torus, chi = -sum (c - 1) over commutator cycles.
DECLARED_CHI = {"square torus": 0, "L-shape": -2, "H(1,1) origami": -2, "octagon": -2, "hexagon": 0,
                "pillowcase": 2}


@task("T029", changed_files=(MODULE, SURFACES, DOC),
      regression_tests=_tests("test_t027_t029_surfaces_and_cones", "test_gauss_bonnet_check_detects_wrong_vertex_classes",
                              "test_regeneration_is_within_tolerance"))
def detect_cone_singularities(ctx):
    examples = example_surfaces()
    rows, defects, chi_mismatch, commutator_mismatch = {}, 0, 0, 0
    for name, s in examples.items():
        t = s.topology()
        angles = sorted(c["cone_angle_over_pi"] for c in t["cones"])
        chi = DECLARED_CHI[name]
        perms = getattr(s, "permutations", None)
        if perms:
            cycles = surf.commutator_cycles(perms["r"], perms["u"])
            chi = -sum(c - 1 for c in cycles)  # Riemann-Hurwitz; also equals the declared value (checked below)
            predicted = sorted(Fraction(2 * c) for c in cycles)
            commutator_mismatch += predicted != angles
        # Gauss-Bonnet against the independent chi: sum (2 pi - theta_v) / pi - 2 chi.
        defect = sum(2 - a for a in angles) - 2 * chi
        rows[name] = {"cone_angles_over_pi": [_frac(a) for a in angles], "singular": [_frac(a) for a in angles if a != 2],
                      "vertex_classes": t["vertices"], "euler_characteristic_v_e_f": t["euler_characteristic"],
                      "euler_characteristic_independent": chi, "declared_euler_characteristic": DECLARED_CHI[name],
                      "gauss_bonnet_defect_over_pi": _frac(defect),
                      "angle_recognition_residual": t["angle_recognition_residual"], "gluing": t["gluing"]}
        if perms:
            rows[name]["commutator_prediction_over_pi"] = [_frac(a) for a in predicted]
        defects += defect != 0
        chi_mismatch += (t["euler_characteristic"] != chi) + (chi != DECLARED_CHI[name])
    residual = max(r["angle_recognition_residual"] for r in rows.values())
    ctx.artifact_json("cone-angles.json", rows)
    expected = {"square torus": ["2"], "L-shape": ["6"], "H(1,1) origami": ["4", "4"], "octagon": ["6"],
                "hexagon": ["2", "2"], "pillowcase": ["1", "1", "1", "1"]}
    mismatch = sum(rows[k]["cone_angles_over_pi"] != v for k, v in expected.items())
    fields = _fields(
        "Vertex classes after gluing carry cone angles that sum corner angles; translation surfaces have angles "
        "2 pi (k + 1); the octagon has one 6 pi point; Gauss-Bonnet sum (2 pi - theta_v) = 2 pi chi holds with chi "
        "known independently of the vertex classes.",
        "Union-find on corners (b ~ a', a ~ b' per glued pair), corner angles as exact rational multiples of pi; "
        "origami cone points = cycles of the commutator r u r^-1 u^-1 (angle 2 pi times length) and chi = "
        "-sum (c - 1) (Riemann-Hurwitz); declared chi for the polygon gluings (genus-2 octagon, tori, sphere). "
        "With chi = V - E + F from the same union-find, Gauss-Bonnet holds for every partition of the corners "
        "(sum theta_v = pi sum_f (n_f - 2)), so only the comparison with an independent chi tests the classes.",
        [f"surfaces {list(examples)}", f"declared Euler characteristics {DECLARED_CHI}"], NO_PHYSICAL,
        "Gauss-Bonnet defect 0 against the independent chi; V - E + F equals it; commutator prediction equals "
        "union-find angles for origamis.",
        "Compute vertex classes and cone angles, compare with predictions, the commutator cycle type and the "
        "independent chi, and check Gauss-Bonnet exactly.",
        f"{ {k: v['cone_angles_over_pi'] for k, v in rows.items()} } (units of pi); Gauss-Bonnet defects against "
        f"the independent chi {defects}; V - E + F mismatches {chi_mismatch}; commutator mismatches "
        f"{commutator_mismatch}.",
        f"Exact after angle recognition (float residual <= {residual:.1e} against multiples of pi/24).",
        ["regular vertices (angle 2 pi) not reported as singular", "half-translation cone angle pi",
         "two cone points (H(1,1)) versus one (L-shape)",
         "Gauss-Bonnet tautology avoided (chi independent of the vertex classes)"],
        ["Angle recognition assumes corner angles are multiples of pi/24, true for every declared polygon."],
        "Deferred research question: recognize cone angles exactly for polygons whose corner angles are not "
        "multiples of pi/24 (from exact algebraic vertex coordinates) and test on a declared polygon with such an "
        "angle (recognition here assumes the pi/24 grid)")
    findings = [
        finding("Cone angles: octagon and L-shape one 6 pi point, H(1,1) two 4 pi points, pillowcase four pi points, "
                "square and hexagon tori none", "mathematical", {k: v["cone_angles_over_pi"] for k, v in rows.items()},
                {"checks": [_check("mismatches against predicted cone angles", mismatch),
                            _check("origami commutator-cycle prediction mismatches", commutator_mismatch),
                            _check("angle recognition residual", residual, 1e-12, kind="analytic")]},
                uncertainty=_unc("roundoff", residual,
                                 "float residual of corner-angle recognition before exact rounding to multiples of "
                                 "pi/24"),
                tolerance=EXACT),
        finding("Gauss-Bonnet sum (2 pi - theta_v) = 2 pi chi holds exactly on every surface with chi known "
                "independently of the vertex classes (declared topology; Riemann-Hurwitz for origamis)",
                "mathematical", {k: {"defect_over_pi": v["gauss_bonnet_defect_over_pi"],
                                     "chi": v["euler_characteristic_independent"]} for k, v in rows.items()},
                {"derivation": "With chi = V - E + F from the same vertex classes the identity holds for any "
                               "partition of the corners, so the test uses an independent chi",
                 "checks": [_check("surfaces with nonzero defect against the independent chi", defects),
                            _check("V - E + F or Riemann-Hurwitz chi differing from the declared chi", chi_mismatch)]},
                uncertainty=EXACT_UNC, tolerance=EXACT),
        finding("Polygon vertices need not be cone singularities: the glued hexagon's vertices are regular points",
                "mathematical", rows["hexagon"]["cone_angles_over_pi"],
                {"checks": [_check("singular vertex classes on the hexagon torus", len(rows["hexagon"]["singular"]))]},
                counterexample={"statement": "Every polygon vertex of a glued surface is a cone singularity",
                                "witness": {"surface": "regular hexagon, opposite sides glued",
                                            "vertex_classes": rows["hexagon"]["vertex_classes"],
                                            "cone_angles_over_pi": rows["hexagon"]["cone_angles_over_pi"]}},
                uncertainty=EXACT_UNC, tolerance=EXACT),
    ]
    return {"state": "completed", "fields": fields, "findings": findings}


# ================================================================ T030
@task("T030", changed_files=(MODULE, DISCRETE, DOC),
      regression_tests=_tests("test_t030_grid_metrication"))
def smooth_versus_discrete(ctx):
    scipy_version = _version(ctx, "scipy")
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
    # Two linear-axis figures: the graph errors are flat in N, the fast-marching error falls towards 0.
    series = [(f"{k}-neighbour, 45 deg" if k == 4 else f"{k}-neighbour, 22.6 deg",
               [r["grid"] for r in (diag if k == 4 else tilt)], [r[f"ratio{k}"] - 1 for r in (diag if k == 4 else tilt)])
              for k in (4, 8)]
    series.append(("fast marching, 45 deg", [r["grid"] for r in diag], [abs(r["fmm_rel_error"]) for r in diag]))
    ctx.artifact_text("metrication.svg", svg.line_plot(series, title="Relative length error under grid refinement",
                                                       xlabel="grid points per period N", ylabel="relative error"))
    fmm_series = [(f"fast marching, {name}", [1.0 / r["grid"] for r in rows if r["direction"] == name],
                   [abs(r["fmm_rel_error"]) for r in rows if r["direction"] == name])
                  for name in dict.fromkeys(r["direction"] for r in rows)]
    ctx.artifact_text("fast-marching-error.svg", svg.line_plot(
        fmm_series, title="Fast-marching relative error against grid spacing", xlabel="grid spacing h = 1/N",
        ylabel="relative length error"))
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
        "Deferred research question: fit the fast-marching convergence order over at least five refinement levels "
        "with a confidence interval (three grids and an empirical order now) and compare with a second, "
        "independently written eikonal solver")
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
                                            "ratio": diag[0]["ratio4"]}},
                uncertainty=_unc("roundoff", stencil_error, "graph distance against the closed-form stencil norm"),
                tolerance=TIGHT),
        finding("Worst-direction metrication error is sqrt 2 - 1 (4-nbr, 45 deg) and sqrt(4 - 2 sqrt 2) - 1 "
                "(8-nbr, 22.5 deg)", "numerical", {"4": worst[4][0] - 1, "8": worst[8][0] - 1},
                {"derivation": "maximize cos a + sin a and cos a + (sqrt 2 - 1) sin a on [0, pi/4]",
                 "checks": [_check("dense maximum minus closed form", worst_dense_error, 1e-8, kind="analytic")]},
                uncertainty=_unc("truncation_bound", worst_dense_error,
                                 "36001-angle dense search against the closed form"),
                tolerance={"abs": 1e-8, "rel": 0}),
        finding("Fast marching converges to Euclidean length under refinement", "numerical",
                {"max_relative_error": {str(n): v for n, v in fmm.items()}, "observed_order": order},
                {"checks": [_check("errors against Euclidean length decrease with N (0 = monotone)",
                                   0 if monotone else 1, kind="analytic"),
                            _check("observed order of the error against Euclidean length", order, 0.5, "ge",
                                   kind="analytic")]},
                uncertainty=_unc("truncation_bound", fmm[grid.REFINEMENTS[-1]],
                                 "first-order discretization error on the finest grid (N = 120)"),
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
    # A route's length L_eps(x) = sqrt(Q_eps(x)) has dL/deps = x1 x2 / L, monotone in eps, so the largest |x1 x2| / L
    # over both routes at both bracket ends times the bracket width bounds any continuous length change inside it.
    lipschitz = 0.0
    for side in (before, after):
        m, n = side["translates"][0]
        x = (float(z[0] + m), float(z[1] + n))
        for eps in (before["eps"], after["eps"]):
            lipschitz = max(lipschitz, abs(x[0] * x[1]) / math.sqrt(_perturbed(eps, x)))
    return {"scaling": rows, "delta": _frac(delta), "eps_star": _frac(eps_star), "float_eps_star": 0.5 * (lo + hi),
            "tie_at_eps_star": [list(t) for t in at_star], "switches": switches, "sweep": sweep,
            "bracket": {"eps_before": before["eps"], "eps_after": after["eps"],
                        "translates_before": before["translates"], "translates_after": after["translates"],
                        "heading_before_deg": before["heading_deg"], "heading_after_deg": after["heading_deg"]},
            "heading_jump_deg": abs(after["heading_deg"] - before["heading_deg"]),
            "length_jump": abs(after["length"] - before["length"]),
            "length_continuity_bound": lipschitz * (after["eps"] - before["eps"])}


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
        "Deferred research question: repeat the route-switch study with a spatially varying metric perturbation "
        "(for example eps cos(2 pi x) added to g11 on the square torus), where routes bend continuously and the "
        "switch threshold need not be eps* = 4 delta")
    basis = {"checks": [_check("eps*/delta - 4 mismatches over four deltas", scaling_error),
                        _check("switches in the sweep minus 1", study["switches"] - 1),
                        _check("float bisection minus exact eps*", study["float_eps_star"] - float(eps_star), 1e-15,
                               kind="analytic"),
                        _check("tie multiplicity at eps* minus 2", len(study["tie_at_eps_star"]) - 2)]}
    sympy_version = _version(ctx, "sympy")
    if sympy_version:
        other = sympy_threshold(Fraction(study["delta"]))
        basis["independent_check"] = _independent(_check("sympy.solve eps* - exact eps*", other - eps_star),
                                                  "sympy.solve", sympy_version)
    findings = [
        finding("The shortest route switches at eps* = 4 delta, which vanishes as the unperturbed gap vanishes",
                "mathematical", {"eps_star": study["eps_star"], "scaling": study["scaling"]}, basis,
                uncertainty=EXACT_UNC, tolerance=EXACT),
        finding("A metric perturbation just above eps* = 1/250 (0.4%) turns the shortest-route heading by about "
                "127 deg while the minimal length changes continuously", "numerical",
                {"heading_jump_deg": study["heading_jump_deg"], "length_jump": study["length_jump"]},
                {"derivation": "Each route's coordinate heading is fixed (a constant metric does not bend a "
                               "translate), and its length has dL/deps = x1 x2 / L, so the minimal length can change "
                               "by at most the bracket width times max |x1 x2| / L inside the bracket",
                 "checks": [_check("heading jump (deg)", study["heading_jump_deg"], 90.0, "ge", kind="invariant"),
                            _check("length change across the bracket minus bracket width x max |dL/deps| (the "
                                   "continuity bound)", study["length_jump"] - study["length_continuity_bound"], 0.0,
                                   "signed_le", kind="analytic")]},
                counterexample={"statement": "A small metric perturbation changes the shortest route only slightly",
                                "witness": dict(study["bracket"], delta=study["delta"], eps_star=study["eps_star"],
                                                heading_jump_deg=study["heading_jump_deg"],
                                                length_jump=study["length_jump"])},
                uncertainty={"kind": "truncation_bound",
                             "value": {"heading_jump_deg": 4 * math.ulp(360.0),
                                       "length_jump": study["length_continuity_bound"]},
                             "basis": "heading_jump_deg (degrees): each route's heading is independent of eps, so "
                                      "the bracket adds nothing and only atan2 and degree-conversion rounding (a few "
                                      "ulps) remain; length_jump (length units): the continuous length change "
                                      "possible between the sweep points bracketing eps*, bracket width times max "
                                      "|dL/deps|"},
                tolerance=FLOAT),
        finding("A metric calibrated from physical measurements is accurate enough to decide between near-tied routes",
                "calibration", None, {}),
    ]
    return {"state": "completed", "fields": fields, "findings": findings}


# ================================================================ T032
BUMP = GaussianBump(1.5, 1.0)
SADDLE = Saddle(1.0)
LIBRARY_HEADINGS = 720
LIBRARY_CONFIGS = {
    "torus-inner": (TORUS, (0.0, math.pi), (2.5, math.pi), 9.0),
    "torus-outer": (TORUS, (0.0, 0.0), (1.5, 0.0), 9.0),
    "torus-tie": (TORUS, (0.0, 0.0), (2.2, 0.0), 9.0),
    "bump": (BUMP, (-2.5, 0.0), (2.5, 0.0), 9.0),
    "saddle": (SADDLE, (-1.5, -0.5), (1.5, 0.8), 8.0),
}
SPHERE_DELTAS = (0.1, 0.01, 0.001)


def library_searches():
    """Route search per configuration, with the drop counts and a double-density convergence rerun."""
    out = {}
    for key, (surface, p, q, max_length) in LIBRARY_CONFIGS.items():
        found, counts = routes.find_routes(surface, p, q, max_length, headings=LIBRARY_HEADINGS)
        conv = routes.fan_convergence(surface, p, q, max_length, found, LIBRARY_HEADINGS)
        out[key] = {"surface": surface.describe(), "p": list(p), "q": list(q), "max_length": max_length,
                    "routes": found, "newton_counts": counts, "convergence": conv}
    return out


def counterexample_library():
    searches = library_searches()
    inner, outer = searches["torus-inner"]["routes"], searches["torus-outer"]["routes"]
    tie, bump = searches["torus-tie"]["routes"], searches["bump"]["routes"]
    witnesses = {}
    witnesses["torus-inner-amplification"] = {
        "shortest": inner[0],
        "alternative": min((r for r in inner[1:] if r["amplification"] < inner[0]["amplification"]),
                           key=lambda r: r["length"]),
        "analytic_shortest_amplification": math.sinh(2.5)}
    alt = [r for r in outer[1:] if routes.margin_value(r) > routes.margin_value(outer[0])]
    witnesses["torus-outer-focus-margin"] = {
        "shortest": outer[0], "alternative": min(alt, key=lambda r: r["length"]),
        "analytic_shortest_margin": math.pi * math.sqrt(3.0) - 4.5}
    witnesses["torus-tied-shortest"] = {"routes": tie[:3]}
    # The straight route over the top has heading 0 by the y -> -y symmetry of the configuration.
    top = min(bump, key=lambda r: abs(r["heading"]))
    witnesses["bump-amplification-vs-conjugacy"] = {"shortest": bump[0], "alternative": top,
                                                    "top_heading": top["heading"]}
    witnesses["sphere-near-antipodal"] = [
        {"delta": delta, "separation": math.pi - delta, "closed_form": routes.sphere_route_pair(math.pi - delta),
         "transfer": routes.sphere_transfer_check(math.pi - delta)} for delta in SPHERE_DELTAS]
    for w in witnesses["sphere-near-antipodal"]:
        w["measured_margin"] = w["transfer"]["first_conjugate"] - w["separation"]
        w["targeting_condition"] = 1.0 / abs(w["transfer"]["j_head"])
    return {"searches": searches, "witnesses": witnesses}


def _brief(route):
    return {"length": route["length"], "heading": route["heading"], "amplification": route["amplification"],
            "targeting_condition": route["targeting_condition"], "focus_margin": route["focus_margin"],
            "margin_lower_bound": route["margin_lower_bound"]}


def _witness(key, lib, pair):
    search = lib["searches"][key]
    return {"surface": search["surface"], "p": search["p"], "q": search["q"],
            **{name: _brief(route) for name, route in pair.items()}}


@task("T032", changed_files=(MODULE, ROUTES, DOC),
      regression_tests=_tests("test_t032_counterexample_library", "test_route_helpers",
                              "test_focus_margin_claims_state_the_canonical_definition",
                              "test_sympy_field_matches_the_closed_form_batch_field"))
def shortest_is_not_safest(ctx):
    lib = counterexample_library()
    w = lib["witnesses"]
    inner, outer = w["torus-inner-amplification"], w["torus-outer-focus-margin"]
    tie, bump, sphere = w["torus-tied-shortest"], w["bump-amplification-vs-conjugacy"], w["sphere-near-antipodal"]
    saddle = lib["searches"]["saddle"]["routes"]
    independent = {}
    for key, witness in (("torus-inner", inner), ("torus-outer", outer), ("bump", bump)):
        surface, p, q, _ = LIBRARY_CONFIGS[key]
        independent[key] = _independent_routes(ctx, surface, p, q, [witness["shortest"], witness["alternative"]])
    ctx.artifact_json("counterexample-library.json", dict(lib, scipy_sympy=independent))
    conv = {k: v["convergence"] for k, v in lib["searches"].items()}
    counts = {k: v["newton_counts"] for k, v in lib["searches"].items()}
    inner_gap = inner["shortest"]["amplification"] - inner["alternative"]["amplification"]
    inner_anchor = abs(inner["shortest"]["amplification"] - inner["analytic_shortest_amplification"]) / math.sinh(2.5)
    outer_alt_margin = routes.margin_value(outer["alternative"])
    outer_alt_margin = outer["alternative"]["margin_lower_bound"] if math.isinf(outer_alt_margin) else outer_alt_margin
    outer_gap = outer_alt_margin - outer["shortest"]["focus_margin"]
    outer_anchor = abs(outer["shortest"]["focus_margin"] - outer["analytic_shortest_margin"])
    t0, t1 = tie["routes"][0], tie["routes"][1]
    bump_gap = bump["shortest"]["amplification"] - bump["alternative"]["amplification"]
    # Measured route-quantity differences for the bump pair: batch RK4 against ciw.lab.jacobi (relative j_head),
    # and when available the scipy/sympy integration (relative j_head, absolute conjugate point).
    bump_sc = independent["bump"]
    bump_unc = max([r["j_head_batch_vs_transfer"] / max(1.0, r["amplification"])
                    for r in (bump["shortest"], bump["alternative"])]
                   + ([bump_sc["j_head"], bump_sc["conjugate"]] if bump_sc else []))
    margin_error = max(abs(s["measured_margin"] - s["delta"]) / s["delta"] for s in sphere)
    j_error = max(abs(s["transfer"]["j_head"] - math.sin(s["delta"])) for s in sphere)
    fields = _fields(
        "'Shortest means safest' fails on curved surfaces: a shortest geodesic can have larger heading "
        "amplification, a smaller focus margin, a tie with another route, or a near-conjugate endpoint, while "
        "the flat torus (T021) and simply connected negatively curved surfaces admit no such witness.",
        "Heading amplification |j_head(L)|, targeting condition 1/|j_head(L)| and focus margin s_c - L from "
        "j'' + K j = 0 along each route; exact anchors: inner equator K = -1 (j_head = sinh s), outer equator "
        "K = 1/3 (conjugate at pi sqrt 3), unit sphere (j_head = sin s, conjugate at pi).",
        ["Torus(2, 1): inner-equator pair (0, pi) -> (2.5, pi); outer-equator pair (0, 0) -> (1.5, 0); (0, 0) -> "
         "(2.2, 0)", "GaussianBump(h = 1.5, sigma = 1): (-2.5, 0) -> (2.5, 0)",
         f"unit sphere, separations pi - delta for delta in {list(SPHERE_DELTAS)}",
         "Saddle(c = 1): (-1.5, -0.5) -> (1.5, 0.8)",
         f"{LIBRARY_HEADINGS}-heading fans (convergence reruns at {2 * LIBRARY_HEADINGS})"],
        NO_PHYSICAL,
        "Each witness violates the refuted statement by a margin far above numerical error; analytic anchors "
        "reproduce closed forms; route sets do not change when the fan density doubles.",
        "Fan search + Newton + ciw.lab.jacobi verification per configuration (as T024), repeated at double fan "
        "density; sphere margins from ciw.lab.jacobi for a sequence of separations; optional scipy DOP853 "
        "integration of the witness routes on a field derived by sympy from the embedding.",
        f"inner: shortest L = {inner['shortest']['length']:.4f} amp {inner['shortest']['amplification']:.4f} vs L = "
        f"{inner['alternative']['length']:.4f} amp {inner['alternative']['amplification']:.4f}; outer: shortest margin "
        f"{outer['shortest']['focus_margin']:.4f} vs alternative >= {outer_alt_margin:.4f}; tie: lengths "
        f"{t0['length']:.10f} and {t1['length']:.10f}; bump: shortest amp {bump['shortest']['amplification']:.4f} vs "
        f"top route amp {bump['alternative']['amplification']:.4f} with margin {bump['alternative']['focus_margin']:.4f} "
        f"(targeting condition {bump['alternative']['targeting_condition']:.3f}); sphere minor-arc margins "
        f"{[round(s['measured_margin'], 6) for s in sphere]} for delta {list(SPHERE_DELTAS)} (targeting condition "
        f"up to {max(s['targeting_condition'] for s in sphere):.0f}); saddle routes found {len(saddle)}; route "
        f"counts at base/double density { {k: v['routes'] for k, v in conv.items()} }; Newton drops (singular, "
        f"diverged, residual, over length) "
        f"{ {k: [c['singular'], c['diverged'], c['residual_fail'], c['over_length']] for k, c in counts.items()} }.",
        "Route quantities to about 1e-6 relative (RK4 step 0.04; scipy/sympy agreement when available); witness "
        f"gaps are O(0.1-4); sphere margins within {max(abs(s['measured_margin'] - s['delta']) for s in sphere):.0e} "
        "of delta (checked to 1e-3 relative).",
        ["analytic anchors (sinh 2.5, pi sqrt 3 - 4.5, sin delta)", "censored margins use the horizon lower bound",
         "negative margins (route past a conjugate point) kept, not discarded", "symmetric duplicate routes",
         "fan density (route sets unchanged at double density)"],
        ["Witnesses show the statements are false in general; they do not rank routes for any application.",
         "Route sets are fan-search results, stable under doubling the density but not proven complete (the "
         "saddle's uniqueness is the Cartan-Hadamard theorem, not the search).", AMPLIFICATION_CAVEAT],
        "Deferred research question: extend the library to variable-curvature triangle meshes (for example a "
        "jittered icosphere or a mesh torus, with routes traced by the T038 mesh tracer), where route quantities "
        "carry mesh error (manufacturing paths are already covered by T137's retained counterexample to 'The "
        "shortest route between a station and an edge is also the safest route')")

    def basis_with(checks, key):
        basis = {"checks": checks}
        _attach_independent_routes(basis, independent[key])
        return basis

    findings = [
        finding("Torus inner equator: the shortest route has larger heading amplification than a longer route",
                "numerical", {"shortest": _brief(inner["shortest"]), "alternative": _brief(inner["alternative"])},
                basis_with([_check("shortest minus alternative amplification", inner_gap, 1.0, "signed_ge",
                                   kind="invariant"),
                            _check("shortest amplification vs sinh 2.5 (relative)", inner_anchor, 1e-6,
                                   kind="analytic")], "torus-inner"),
                counterexample={"statement": "The shortest geodesic between two points has the least heading "
                                             "amplification",
                                "witness": _witness("torus-inner", lib, {"shortest": inner["shortest"],
                                                                         "alternative": inner["alternative"]})},
                uncertainty=_unc("reference_error", inner_anchor,
                                 "relative deviation of the shortest amplification from sinh 2.5 (RK4 step about 0.04)"),
                tolerance=ROUTE),
        finding("Torus outer equator: the shortest route has a smaller focus margin s_c - L (s_c the first zero of "
                "j_head) than a longer route", "numerical",
                {"shortest": _brief(outer["shortest"]), "alternative": _brief(outer["alternative"])},
                basis_with([_check("alternative minus shortest focus margin", outer_gap, 1.0, "signed_ge",
                                   kind="invariant"),
                            _check("shortest margin vs pi sqrt 3 - 4.5", outer_anchor, 1e-6, kind="analytic")],
                           "torus-outer"),
                counterexample={"statement": "The shortest geodesic has the largest focus margin s_c - L (farthest "
                                             "from its first conjugate point)",
                                "witness": _witness("torus-outer", lib, {"shortest": outer["shortest"],
                                                                         "alternative": outer["alternative"]})},
                uncertainty=_unc("reference_error", outer_anchor,
                                 "deviation of the shortest focus margin from pi sqrt 3 - 4.5"),
                tolerance=ROUTE),
        finding("Torus (0, 0) -> (2.2, 0): two mirror-image shortest routes tie, so 'the' shortest route is not "
                "unique", "numerical",
                {"lengths": [t0["length"], t1["length"]], "headings": [t0["heading"], t1["heading"]]},
                {"checks": [_check("length difference of the two shortest routes", t0["length"] - t1["length"], 1e-9,
                                   kind="invariant"),
                            _check("heading separation (rad)", abs(t0["heading"] - t1["heading"]), 0.1, "ge",
                                   kind="invariant")]},
                counterexample={"statement": "The shortest route between two points is unique",
                                "witness": _witness("torus-tie", lib, {"first": t0, "second": t1})},
                uncertainty=_unc("roundoff", abs(t0["length"] - t1["length"]),
                                 "length difference of the mirror-image routes"),
                tolerance=ROUTE),
        finding("Gaussian bump: a longer route over the top has smaller amplification but passes a conjugate point",
                "numerical", {"shortest": _brief(bump["shortest"]), "alternative": _brief(bump["alternative"])},
                basis_with([_check("shortest minus top-route amplification", bump_gap, 0.2, "signed_ge",
                                   kind="invariant"),
                            _check("depth of the top route past its conjugate point (-margin)",
                                   -bump["alternative"]["focus_margin"], 1.0, "signed_ge", kind="invariant"),
                            _check("top-route heading (the symmetric straight route)", bump["top_heading"], 1e-9,
                                   kind="invariant")], "bump"),
                counterexample={"statement": "Low heading amplification certifies a robust (locally minimizing) route",
                                "witness": _witness("bump", lib, {"shortest": bump["shortest"],
                                                                  "alternative": bump["alternative"]})},
                uncertainty=_unc("reference_error", bump_unc,
                                 "max of the measured relative j_head differences (batch RK4 search and, when scipy "
                                 "and sympy are available, the scipy/sympy integration) and the absolute scipy/sympy "
                                 "conjugate-point difference bounding the focus margin (length units)"),
                tolerance=ROUTE),
        finding("On the unit sphere the minimizing arc between points at separation pi - delta ends delta before "
                "its conjugate point, so minimizing geodesics have no positive lower bound on the focus margin "
                "s_c - L",
                "numerical", {"delta": list(SPHERE_DELTAS), "focus_margin": [s["measured_margin"] for s in sphere],
                              "j_head": [s["transfer"]["j_head"] for s in sphere],
                              "targeting_condition": [s["targeting_condition"] for s in sphere]},
                {"derivation": "j_head = sin s on the unit sphere; the first conjugate point is at pi, so the "
                               "minor arc of length pi - delta has margin delta and targeting condition 1/sin delta",
                 "checks": [_check("ciw.lab.jacobi focus margin relative to delta (max over delta)", margin_error,
                                   1e-3, "le", kind="analytic"),
                            _check("ciw.lab.jacobi j_head(L) - sin delta (max over delta)", j_error, 1e-8, "le",
                                   kind="analytic")]},
                counterexample={"statement": "Minimizing geodesics on the unit sphere have focus margin s_c - L "
                                            "bounded below by a positive constant",
                                "witness": {"surface": "unit sphere", "kind": "near-conjugate conditioning",
                                            "delta": list(SPHERE_DELTAS),
                                            "focus_margin": [s["measured_margin"] for s in sphere],
                                            "targeting_condition": [s["targeting_condition"] for s in sphere]}},
                uncertainty=_unc("reference_error", max(abs(s["measured_margin"] - s["delta"]) for s in sphere),
                                 "absolute deviation of the ciw.lab.jacobi margin from delta (Hermite zero location)"),
                tolerance={"abs": 1e-6, "rel": 1e-6}),
        finding("Search on the saddle (K < 0, simply connected) finds exactly one route, consistent with "
                "Cartan-Hadamard uniqueness, so this surface admits no witness",
                "numerical", {"routes": len(saddle)},
                {"derivation": "Cartan-Hadamard: exp is a diffeomorphism on complete simply connected K <= 0 surfaces",
                 "checks": [_check("routes found minus 1", len(saddle) - 1)]},
                uncertainty=EXACT_UNC, tolerance=EXACT),
        finding(f"Every T032 route search gives the same route set at twice the fan density ({LIBRARY_HEADINGS} to "
                f"{2 * LIBRARY_HEADINGS} headings)", "numerical",
                {k: {"routes": v["routes"], "differences": v["differences"]} for k, v in conv.items()},
                {"checks": [_check("routes found at only one of the two densities (all configurations)",
                                   sum(v["differences"] for v in conv.values()), kind="self_convergence")]},
                uncertainty=ROUTE_SET_UNC, tolerance=EXACT),
        finding("A route from this library is safe (or unsafe) to execute on a physical part or vehicle",
                "machine_safety", None, {}),
    ]
    return {"state": "completed", "fields": fields, "findings": findings}
