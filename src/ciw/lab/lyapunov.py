"""Lyapunov runtime experiments T101-T114 against the pinned PLSR instrument.

Scope: the float64 decision procedure of the Parameterized Lyapunov Stability
Runtime (PLSR 0.1.0rc1, commit 19ea696) -- resolution floor, power-of-two
scaling, overflow and underflow, semidefinite and defective edge cases, unit
changes, the nine runtime-status-v1 codes, the declared-margin policy and the
continuous/discrete conventions -- compared with CIW's exact dyadic-rational
arithmetic, NumPy eigenvalues and, when installed, SciPy Lyapunov solvers
(independent checks), and with CIW's transcription of the documented formulas
(same-specification checks, never counted as independent). The runtime runs in
a subprocess under the interpreter bound as provider ``plsr-python`` after its
source pin is verified. The tasks declare no hard requirement: without the
provider they are reported ``partial`` and retain their provider-free findings,
which a ``blocked`` report could not carry. T112-T114 are specifications with
synthetic illustrations (``lyapunov_research``).

Non-claims: every verdict concerns declared binary64 matrices and one declared
sample, not a physical plant. Exact arithmetic here decides what those numbers
imply, nothing more. No task establishes machine safety, actuator authority,
calibration or sensor performance; each such claim is retained as a
``not_established`` finding.
"""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np

from .. import __version__
from . import lyapunov_reference as R
from . import lyapunov_research as X
from . import svg
from .evidence import COMPUTATIONAL_DOMAINS, finding, holds
from .lyapunov_provider import PLSR_ROLE, ProviderRefusal, manifest, producer, provider_basis, run_plsr, source_digest
from .registry import task

MODULE = "src/ciw/lab/lyapunov.py"
REFERENCE = "src/ciw/lab/lyapunov_reference.py"
BRIDGE = "src/ciw/lab/lyapunov_provider.py"
RESEARCH = "src/ciw/lab/lyapunov_research.py"
DOC = "docs/lab/LYAPUNOV.md"
TESTS = "tests/test_lab_lyapunov.py"
PROVIDER_FILES = (MODULE, REFERENCE, BRIDGE, DOC)
RESEARCH_FILES = (MODULE, REFERENCE, RESEARCH, DOC)
CIW_REFERENCE = {"implementation": "ciw.lab.lyapunov_reference", "revision": __version__}
REQUIREMENT = f"provider:{PLSR_ROLE}"


def _node(name):
    return f"{TESTS}::{name}"


PARTIAL_TEST = f"{TESTS}::test_provider_tasks_are_partial_without_the_provider"


EXACT = {"kind": "roundoff", "value": 0.0, "basis": "exact rational, integer or bitwise comparison; no rounding "
                                                    "enters the value"}
# Regression tolerance for values that are codes, booleans or exact counts.
EXACT_TOL = {"abs": 0.0, "rel": 0.0}
ANALYTIC = {"kind": "roundoff", "value": 0.0, "basis": "closed-form argument; nothing was approximated"}


def _roundoff(value, basis):
    return {"kind": "roundoff", "value": float(value), "basis": basis}


def _platform(value, basis):
    return {"kind": "reference_error", "value": float(value), "basis": basis}


# Shared plumbing -------------------------------------------------------------

class _Unavailable(Exception):
    """The provider part of a task cannot run here (unbound or refused)."""


def _bridge(ctx, cases):
    if not ctx.available(REQUIREMENT):
        raise _Unavailable(f"Provider {PLSR_ROLE} is not bound; bind a Python 3.12+ interpreter with the pinned "
                           f"PLSR via --provider {PLSR_ROLE}=PATH")
    try:
        return run_plsr(ctx.providers[PLSR_ROLE], cases)
    except ProviderRefusal as exc:
        raise _Unavailable(f"Provider {PLSR_ROLE} refused: {exc}") from exc


def _pin_identity():
    pin = manifest()
    return {"role": PLSR_ROLE, "repository": pin["repository"], "commit": pin["commit"],
            "package_version": pin["package_version"], "source_digest": source_digest(pin), "executed": False}


def _finish(fields, findings, changed, identity=None, blocked=None, provider=True):
    """Assemble the outcome; a missing provider or a failed planned check makes the task partial, never hidden."""
    from .runner import builtin_identity

    fields = dict(fields)
    state = "completed"
    if blocked is not None:
        # The provider-free parts ran and retain established findings, so the task is partial, not blocked.
        state = "partial"
        fields["experiment"] = (f"Partial: {blocked}. Only the provider-free parts ran and are retained. Planned: "
                                f"{fields['experiment']}")
        fields["unresolved_assumptions"] = [blocked] + list(fields["unresolved_assumptions"])
    failed = [f["claim"] for f in findings if f["evidence_status"] == "not_established"
              and f["domain"] in COMPUTATIONAL_DOMAINS and not f.get("expected_not_established")]
    if failed and state == "completed":
        state = "partial"
        fields["unresolved_assumptions"] = list(fields["unresolved_assumptions"]) + [
            "A planned check did not pass on this platform: " + "; ".join(failed)]
    identity_field = builtin_identity(changed)
    if provider:
        identity_field["provider"] = identity if identity is not None else _pin_identity()
    fields["provider_runtime_identity"] = identity_field
    return {"state": state, "fields": fields, "findings": findings}


def _fields(hypothesis, model, inputs, observation, invariant, experiment, next_task, failure_modes, assumptions):
    return {"hypothesis": hypothesis, "mathematical_model": model, "input_data": list(inputs),
            "observation_model": observation, "expected_invariant": invariant, "experiment": experiment,
            "recommended_next_task": next_task, "failure_modes_checked": list(failure_modes),
            "unresolved_assumptions": list(assumptions), "numerical_result": "", "uncertainty": ""}


def _check(reference, observed, tolerance=0.0, comparison="abs_le", kind="exact_arithmetic"):
    observed, tolerance = float(observed), float(tolerance)
    return {"reference_kind": kind, "reference": reference, "observed": observed, "tolerance": tolerance,
            "comparison": comparison, "passed": holds(observed, tolerance, comparison)}


def _refusal(reference, expected, observed):
    return {"reference_kind": "refusal", "reference": reference, "expected_refusal": expected,
            "observed_refusal": observed, "passed": observed == expected}


def _independent(check, identity, checker=None):
    return dict(check, producer=producer(identity), checker=checker or CIW_REFERENCE)


def _mat(matrix):
    return np.asarray(matrix, dtype=float).tolist()


def _verdict_case(cid, A, P, x, time="continuous", **extra):
    return {"id": cid, "op": "verdict", "plant": {"A": _mat(A), "time": time}, "certificate": {"P": _mat(P)},
            "x": [float(v) for v in np.asarray(x, dtype=float)], **extra}


def _affine_case(cid, A0, terms, box, P, x, theta, theta_dot=None, time="continuous", rate_box=None, **extra):
    plant = {"A0": _mat(A0), "terms": [_mat(t) for t in terms], "theta_min": [float(v) for v in box[0]],
             "theta_max": [float(v) for v in box[1]], "time": time}
    if rate_box is not None:
        plant.update(rate_min=[float(v) for v in rate_box[0]], rate_max=[float(v) for v in rate_box[1]])
    case = {"id": cid, "op": "verdict", "plant": plant, "certificate": {"P": _mat(P)},
            "x": [float(v) for v in np.asarray(x, dtype=float)], "theta": [float(v) for v in theta], **extra}
    if theta_dot is not None:
        case["theta_dot"] = [float(v) for v in theta_dot]
    return case


def _code(result):
    return result["code"] if result["ok"] else f"raises {result['error']['type']}"


def _counts(values):
    table = {}
    for value in values:
        table[value] = table.get(value, 0) + 1
    return dict(sorted(table.items()))


def _normal(matrix):
    """Every nonzero entry is a normal binary64 number (power-of-two scaling is then exact)."""
    values = np.abs(np.asarray(matrix, dtype=float))
    nonzero = values[values != 0.0]
    return bool(np.all(np.isfinite(values)) and (nonzero.size == 0 or np.min(nonzero) >= 2.0 ** -1022))


def _normal_window(A, P, time="continuous"):
    """A, P, the decrease form and the resolution are all normal and finite.

    Inside this window every operation of the documented procedure commutes
    exactly with power-of-two scaling of A (up to LAPACK's own rescaling of
    matrices whose largest entry lies outside [2^-485, 2^485]).
    """
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        form = R.decrease_matrix(A, P, time)
        res = R.resolution(A, P, time, form=form)
    return _normal(A) and _normal(P) and _normal(form) and math.isfinite(res) and res >= 2.0 ** -1022


def _subnormal_witness():
    """A = [[-2, 5], [0, -3]] * 2^-1074, P = I: the exact decrease form [[-4, 5], [5, -6]] u is indefinite."""
    return np.array([[-2.0, 5.0], [0.0, -3.0]]) * R.TINY, np.eye(2), np.array([1.0, 0.0])


def _witness_exact():
    """Exact and binary64 decrease forms of the subnormal witness, in units of 2^-1074."""
    A, P, _ = _subnormal_witness()
    exact = R.exact_form(A, P)
    computed = R.decrease_matrix(A, P) / R.TINY
    scaled = [[entry / Fraction(R.TINY) for entry in row] for row in exact]
    return {"exact_form_units": [[float(v) for v in row] for row in scaled],
            "exact_det_units2": float(R.exact_det(scaled)), "exact_class": R.exact_class(exact),
            "float_form_units": computed.tolist(),
            "float_det_units2": float(computed[0, 0] * computed[1, 1] - computed[0, 1] ** 2),
            "float_class": R.exact_class(R.fractions(computed))}


def _witness_finding(result, identity, statement, claim):
    """The subnormal witness: robust facts are checked; PLSR's code is recorded, and a certificate is a counterexample.

    The exact class and the underflowed resolution do not depend on the platform. Whether LAPACK's eigenvalue
    of the rounded subnormal form stays negative does, so the certification is a counterexample attached (with
    the check that observes it) only when it occurs.
    """
    exact = _witness_exact()
    A, P, x = _subnormal_witness()
    code = _code(result)
    certified = code in R.CERTIFYING
    checks = [_check("negated exact determinant of the declared 2x2 decrease form, units (2^-1074)^2 "
                     "(at least 1 means indefinite)", -exact["exact_det_units2"], 1.0, "ge"),
              _check("PLSR resolution of the witness (underflows to zero)", result.get("resolution", math.nan), 0.0,
                     kind="invariant")]
    extra = {}
    if certified:
        checks.append(_check("PLSR code on the witness is certifying (1 if so)", 1.0, 1.0, "ge", "invariant"))
        extra["counterexample"] = {
            "statement": statement,
            "witness": {"A_hex": R.hexed(A), "P_hex": R.hexed(P), "x": x.tolist(), "code": code,
                        "exact_form_units_of_2^-1074": exact["exact_form_units"],
                        "binary64_form_units_of_2^-1074": exact["float_form_units"],
                        "plsr_details": result.get("details")}}
    return finding(
        claim, "numerical",
        {"code": code, "certified": certified, "exact_class": exact["exact_class"],
         "exact_det_units2": exact["exact_det_units2"], "resolution": result.get("resolution", 0.0)},
        {"provider": provider_basis(identity), "checks": checks}, tolerance=EXACT_TOL,
        uncertainty=_platform(0.0, "the exact class and the zero resolution are exact; whether PLSR certifies "
                                   "depends on LAPACK's subnormal handling"), **extra)


WITNESS_CLAIM = ("A subnormal plant whose declared decrease form is exactly indefinite drives the PLSR resolution to "
                 "zero; PLSR's code on it is recorded")


# T101 ------------------------------------------------------------------------

GAMMA5 = 5.0 * R.U / (1.0 - 5.0 * R.U)
EPSILON_STAR = 4.0 * GAMMA5 / (1.0 - 8.0 * R.U)
T101_EPSILONS = (0.1, 1e-6, 1e-12, 1e-14, EPSILON_STAR * (1.0 + 1e-6), EPSILON_STAR * (1.0 - 1e-6), 1e-15, 0.0)
T101_SCALES = (-1074, -1073, -1072, -1070, -1066, -1060, -1050, -1040, -1030, -1022, -1000, -900, -600, -485, -300,
               0, 300, 485, 600, 900, 1000, 1015, 1020, 1022, 1023)


def _t101_family():
    """Threshold family A = 2^k [[-eps, 1], [-1, -eps]], P = I, and seeded Hurwitz 3x3 plants with their P."""
    cases, meta = [], []
    for i, eps in enumerate(T101_EPSILONS):
        unit = np.array([[-eps, 1.0], [-1.0, -eps]])
        for k in T101_SCALES:
            A = np.ldexp(unit, k)
            cases.append(_verdict_case(f"F1:{i}:{k}", A, np.eye(2), [1.0, 0.5]))
            meta.append({"family": "F1", "eps_index": i, "eps": eps, "k": k, "A": A, "P": np.eye(2),
                         "normal": _normal_window(A, np.eye(2))})
    for seed in range(4):
        rng = R.generator(101 + seed)
        A0 = rng.normal(size=(3, 3)) - 3.0 * np.eye(3)
        P0 = R.kron_lyapunov(A0, np.eye(3))
        for k in (-1074, -1070, -1066, -1062, -1058, -1050, -1040, -1022, -1000, -500, 0, 500, 1000, 1015, 1018,
                  1020):
            with np.errstate(over="ignore"):
                A = np.ldexp(A0, k)
            if not np.all(np.isfinite(A)):
                continue  # not a declarable binary64 matrix
            cases.append(_verdict_case(f"F2:{seed}:{k}", A, P0, np.ones(3)))
            meta.append({"family": "F2", "seed": seed, "k": k, "A": A, "P": P0})
        for k in (500, 505, 508, 509, 510, 511, 512):
            A, P = np.ldexp(A0, k), np.ldexp(P0, k)
            cases.append(_verdict_case(f"F3:{seed}:{k}", A, P, np.ones(3)))
            meta.append({"family": "F3", "seed": seed, "k": k, "A": A, "P": P})
    A, P, x = _subnormal_witness()
    cases.append(_verdict_case("witness", A, P, x))
    cases.append(_verdict_case("witness-unit", A / R.TINY, P, x))
    return cases, meta


def _t101_expected(eps):
    """The code the analytic threshold assigns in the normal range: certified iff eps > eps*."""
    return "CERTIFIED_WITH_MARGIN" if eps > EPSILON_STAR else "NUMERICAL_INCONCLUSIVE"


def _t101_offline(meta):
    exact = _witness_exact()
    homogeneity = []
    A0 = np.array([[-0.1, 1.0], [-1.0, -0.1]])
    base = R.resolution(A0, np.eye(2))
    for k in T101_SCALES:
        A = np.ldexp(A0, k)
        if _normal_window(A, np.eye(2)):
            homogeneity.append(abs(np.ldexp(R.resolution(A, np.eye(2)), -k) - base) / base)
    normal = [m for m in meta if m["family"] == "F1" and m["normal"]]
    rederived = sum(R.documented_code(m["A"], m["P"], (1.0, 0.5))["code"] != _t101_expected(m["eps"]) for m in normal)
    return [
        finding("Inconclusive threshold of the family A = s[[-eps, 1], [-1, -eps]], P = I: certified iff eps > "
                "4 gamma_5 / (1 - 8u), independent of s", "mathematical", EPSILON_STAR,
                {"derivation": "n = 2, max|A| = s, max|P| = 1, M = -2 eps s I exactly: resolution = s(8 gamma_5 + "
                               "16 u eps) against margin 2 eps s (docs/lab/LYAPUNOV.md, T101)"},
                tolerance={"abs": 0.0, "rel": 1e-12},
                uncertainty=_roundoff(R.U * EPSILON_STAR, "one rounding of the closed form")),
        finding("The documented decision order, transcribed in CIW, places the family's threshold at eps* in the "
                "normal range", "numerical", {"normal_range_cases": len(normal), "mismatches": rederived},
                {"generator": {"name": "T101 F1 family", "seed": None},
                 "checks": [_check("codes of the transcribed order differing from the analytic threshold eps*",
                                   rederived, 0.0, kind="analytic")]},
                tolerance=EXACT_TOL, uncertainty=_roundoff(0.0, "the eps* cases sit 1e-6 relative from the "
                                                                "threshold, far beyond rounding")),
        finding("CIW's re-derived resolution scales exactly with a power-of-two matrix scale in the normal range",
                "numerical", {"scales": len(homogeneity), "max_relative_deviation": max(homogeneity)},
                {"generator": {"name": "power-of-two scales of [[-0.1, 1], [-1, -0.1]]", "seed": None},
                 "checks": [_check("res(2^k A) 2^-k versus res(A), bitwise", max(homogeneity), 0.0,
                                   kind="invariant")]},
                tolerance=EXACT_TOL, uncertainty=EXACT),
        finding("The subnormal witness's exact decrease form is indefinite while its binary64 evaluation is "
                "negative definite", "numerical",
                {"exact_det_units2": exact["exact_det_units2"], "float_det_units2": exact["float_det_units2"],
                 "exact_class": exact["exact_class"], "float_class": exact["float_class"]},
                {"checks": [_check("negated exact determinant of [[-4, 5], [5, -6]] (units 2^-2148; at least 1 "
                                   "means indefinite)", -exact["exact_det_units2"], 1.0, "ge"),
                            _check("determinant of the rounded form [[-4, 4], [4, -6]] (units 2^-2148)",
                                   exact["float_det_units2"], 1.0, "ge")]},
                tolerance=EXACT_TOL, uncertainty=EXACT),
    ]


CODE_LEVELS = {"CERTIFIED_WITH_MARGIN": 1.0, "NUMERICAL_INCONCLUSIVE": 0.0}


@task("T101", changed_files=PROVIDER_FILES, regression_tests=(_node("test_t101_resolution_floor"),
                                                                _node("test_reference_witness_is_exactly_indefinite"),
                                                                _node("test_t101_threshold_is_analytic"), PARTIAL_TEST))
def resolution_floor(ctx):
    fields = _fields(
        "For the family A = s[[-eps, 1], [-1, -eps]], P = I, PLSR's resolution floor is the documented "
        "Higham/Weyl/LAPACK bound and scales exactly with s, so the inconclusive threshold stays at eps* wherever "
        "every quantity is a normal binary64 number (this family has a diagonal decrease form, where eigvalsh is "
        "exact; T102 tests general near-threshold forms); the bound is not derived for subnormal arithmetic, "
        "where it can underflow below the actual rounding error.",
        "Continuous decrease form M = A^T P + P A (symmetrised); resolution = n*(2 gamma_{n+3} n max|A| max|P|) + "
        "n^3 u max|M|; certified iff -max eig(M) > resolution. For A = s[[-eps, 1], [-1, -eps]], P = I the "
        "threshold is eps* = 4 gamma_5/(1 - 8u) = 2.2204e-15 for every s with normal arithmetic.",
        ["F1: 8 eps values x 25 exponents k in [-1074, 1023], A = 2^k [[-eps, 1], [-1, -eps]], P = I, x = (1, 0.5)",
         "F2: 4 seeded Hurwitz 3x3 A0 (PCG64 seeds 101-104) with P0 from a CIW Kronecker Lyapunov solve, "
         "A = 2^k A0 for 16 exponents in [-1074, 1020], x = ones",
         "F3: the same plants with A = 2^k A0 and P = 2^k P0 for k in [500, 512] (overflow boundary)",
         "Subnormal witness A = [[-2, 5], [0, -3]] 2^-1074, P = I, x = (1, 0), and its unit-scale twin"],
        "Runtime verdict code, resolution and margin per case, returned by the pinned PLSR in a subprocess.",
        "resolution_PLSR = resolution_CIW; in the normal range the code is CERTIFIED_WITH_MARGIN iff eps > eps* "
        "at every scale; NUMERICAL_OVERFLOW exactly where the decrease form overflows; exactly-indefinite "
        "declared forms are never certified.",
        "Evaluate every case with PLSR, recompute the documented resolution and the exact rational class of the "
        "declared form in CIW, compare codes with the analytic threshold and the exact class, and locate where the "
        "prediction stops holding.",
        "T102: test power-of-two homogeneous scaling of A, P and x together; then propose a subnormal-aware "
        "resolution term (an absolute n^2 * 2^-1074 floor) upstream and re-run T101.",
        ["resolution differs from the documented formula", "normalised resolution drifts with scale",
         "code differs from the analytic threshold eps*", "code differs from the unit-scale code",
         "NUMERICAL_OVERFLOW before the decrease form overflows", "no NUMERICAL_OVERFLOW once it overflows",
         "false certificate below the normal range (exact rational class)"],
        ["The exact class concerns the declared binary64 numbers; they are synthetic, not measured plants.",
         "The resolution comparison is against CIW's transcription of the documented formula: a same-"
         "specification check, not an independent one.",
         "Subnormal behaviour of numpy.linalg.eigvalsh (LAPACK dsyevd) may differ between BLAS builds."])
    cases, meta = _t101_family()
    offline = _t101_offline(meta)
    try:
        bridge = _bridge(ctx, cases)
    except _Unavailable as exc:
        fields["numerical_result"] = (f"Provider-free: eps* = {EPSILON_STAR:.6e}; the transcribed decision order "
                                      f"places the threshold at eps* in "
                                      f"{offline[1]['value']['normal_range_cases'] - offline[1]['value']['mismatches']}"
                                      f" of {offline[1]['value']['normal_range_cases']} normal-range cases; the "
                                      "subnormal witness is exactly indefinite.")
        fields["uncertainty"] = "Exact rational arithmetic for the witness; analytic threshold."
        return _finish(fields, offline, PROVIDER_FILES, blocked=str(exc))
    identity, results = bridge["identity"], bridge["results"]
    rows, rel_diffs, f1_normal, f1_analytic, f1_unit, f1_below = [], [], 0, 0, 0, []
    unit_codes = {m["eps_index"]: _code(results[f"F1:{m['eps_index']}:0"]) for m in meta if m["family"] == "F1"}
    normalised, false_certificates, overflow_first = [], [], {}
    early_overflow, overflow_rows = 0, 0
    for m, case in zip(meta, cases):
        result = results[case["id"]]
        code = _code(result)
        row = {"id": case["id"], "family": m["family"], "k": m["k"], "code": code}
        if result["ok"] and result.get("sample") is not None:
            ref = R.resolution(m["A"], m["P"])
            row.update(resolution=result["resolution"], resolution_ciw=ref, margin_ratio=result["margin_ratio"])
            if math.isfinite(ref) and math.isfinite(result["resolution"]):
                denominator = max(abs(ref), abs(result["resolution"]))
                rel_diffs.append(0.0 if denominator == 0.0 else abs(result["resolution"] - ref) / denominator)
        if m["family"] == "F1":
            row["eps"] = m["eps"]
            if m["normal"]:
                f1_normal += 1
                f1_analytic += code != _t101_expected(m["eps"])
                f1_unit += code != unit_codes[m["eps_index"]]
                if m["eps_index"] == 0 and result.get("sample") is not None:
                    normalised.append(abs(np.ldexp(result["resolution"], -m["k"])
                                          - results["F1:0:0"]["resolution"]) / results["F1:0:0"]["resolution"])
            elif code != unit_codes[m["eps_index"]]:
                f1_below.append({"eps": m["eps"], "k": m["k"], "code": code, "unit_code": unit_codes[m["eps_index"]]})
        elif m["family"] == "F2":
            exact = R.exact_class(R.exact_form(m["A"], m["P"]))
            row["exact_class"] = exact
            if code in R.CERTIFYING and exact != "negative_definite":
                false_certificates.append({"seed": m["seed"], "k": m["k"], "code": code, "exact_class": exact})
        else:
            # NUMERICAL_OVERFLOW is due when the decrease form or the scaled V or x^T M x leaves binary64.
            with np.errstate(over="ignore", invalid="ignore"):
                form = R.decrease_matrix(m["A"], m["P"])
                x = np.ones(3)
                overflows = not (np.all(np.isfinite(form)) and math.isfinite(float(x @ m["P"] @ x))
                                 and math.isfinite(float(x @ form @ x)))
            row["ciw_overflows"] = overflows
            overflow_rows += 1
            early_overflow += code == "NUMERICAL_OVERFLOW" and not overflows
            if overflows and m["seed"] not in overflow_first:
                overflow_first[m["seed"]] = {"k": m["k"], "code": code}
        rows.append(row)
    ctx.artifact_json("resolution-floor.json", R.jsonable({"epsilon_star": EPSILON_STAR, "rows": rows,
                                                            "f1_departures_outside_normal_range": f1_below,
                                                            "f2_false_certificates": false_certificates}))
    series = []
    for offset, (i, label) in enumerate(((0, "eps = 0.1"), (3, "eps = 1e-14"), (4, "eps*(1 + 1e-6)"),
                                         (5, "eps*(1 - 1e-6)"))):
        ks = [m["k"] for m in meta if m["family"] == "F1" and m["eps_index"] == i]
        levels = [CODE_LEVELS.get(_code(results[f"F1:{i}:{k}"]), -1.0) + 0.04 * offset for k in ks]
        series.append((label, ks, levels))
    ctx.artifact_text("codes-vs-scale.svg", svg.line_plot(
        series, title="T101 verdict code across matrix scales 2^k (series offset by 0.04 for visibility)",
        xlabel="k (A scaled by 2^k)", ylabel="1 certified, 0 inconclusive, -1 other"))
    overflow_mismatch = sum(entry["code"] != "NUMERICAL_OVERFLOW" for entry in overflow_first.values())
    seeds = len({m["seed"] for m in meta if m["family"] == "F3"})
    witness = results["witness"]
    base = provider_basis(identity)
    findings = [
        finding("PLSR decrease_resolution equals the documented bound transcribed in CIW at every evaluated matrix "
                "scale", "numerical", {"cases": len(rel_diffs), "max_relative_difference": max(rel_diffs)},
                {"provider": base,
                 "checks": [_check("CIW transcription of the documented resolution formula (same specification)",
                                   max(rel_diffs), 1e-12, kind="analytic")]},
                tolerance={"abs": 1e-12, "rel": 0.0},
                uncertainty=_roundoff(max(rel_diffs), "largest relative difference between two float64 evaluations "
                                                      "of one formula")),
        finding("Inside the binary64 normal range the resolution scales exactly with 2^k and PLSR certifies exactly "
                "when eps > eps*", "numerical",
                {"normal_range_cases": f1_normal, "analytic_mismatches": f1_analytic, "unit_scale_mismatches": f1_unit,
                 "max_normalised_resolution_deviation": max(normalised),
                 "unit_scale_codes": [unit_codes[i] for i in range(len(T101_EPSILONS))]},
                {"provider": base,
                 "checks": [_check("normal-range codes (k = 0 included) differing from the analytic threshold: "
                                   "CERTIFIED_WITH_MARGIN iff eps > eps*", f1_analytic, 0.0, kind="analytic"),
                            _check("normal-range codes differing from PLSR's own unit-scale (k = 0) code", f1_unit,
                                   0.0, kind="invariant"),
                            _check("res(2^k A) 2^-k versus res(A), bitwise", max(normalised), 0.0, kind="invariant")]},
                tolerance=EXACT_TOL,
                uncertainty=_roundoff(0.0, "bitwise comparisons; the eps* cases sit 1e-6 relative from the "
                                           "threshold")),
        finding("NUMERICAL_OVERFLOW first appears exactly where the CIW-recomputed decrease form overflows",
                "numerical",
                {str(seed): entry for seed, entry in sorted(overflow_first.items())},
                {"provider": base,
                 "checks": [_check("seeds whose first overflowing scale is not NUMERICAL_OVERFLOW", overflow_mismatch,
                                   0.0, kind="invariant"),
                            _check("F3 rows reported NUMERICAL_OVERFLOW while the recomputed form and scaled values "
                                   "are finite", early_overflow, 0.0, kind="invariant"),
                            _check("seeds that never overflow on the exponent grid", seeds - len(overflow_first), 0.0,
                                   kind="invariant")]},
                tolerance={"abs": 1.0, "rel": 0.0},
                uncertainty=_platform(1.0, "the first overflowing exponent may move by one if a BLAS reorders "
                                           "products")),
        _witness_finding(witness, identity,
                         "The float64 resolution floor bounds the rounding error of the decrease form at every matrix "
                         "scale, so an exactly indefinite declared form is never certified", WITNESS_CLAIM),
    ] + offline
    witness_code = _code(witness)
    witness_clause = (f"below that range it underflows and the subnormal witness, whose exact decrease form is "
                      f"indefinite, is certified ({witness_code})" if witness_code in R.CERTIFYING else
                      f"below that range it underflows; the subnormal witness was not certified here ({witness_code})")
    sound_clause = ("no F2 certificate contradicts its exact class" if not false_certificates else
                    f"{len(false_certificates)} F2 certificates contradict their exact class")
    fields["numerical_result"] = (
        f"Resolution matches the documented formula (max relative difference {max(rel_diffs):.2e} over "
        f"{len(rel_diffs)} cases). In {f1_normal} normal-range threshold cases {f1_analytic} codes differ from the "
        f"analytic threshold eps* = {EPSILON_STAR:.6e} and {f1_unit} from the unit-scale code. Outside the normal "
        f"range {len(f1_below)} threshold cases depart from the unit-scale code; F2 has {len(false_certificates)} "
        f"certifying verdicts whose exact form is not negative definite. The witness returns {witness_code} with "
        f"resolution {witness['resolution']:.3g} (unit-scale twin: {_code(results['witness-unit'])}). First overflow "
        f"exponents: { {s: e['k'] for s, e in sorted(overflow_first.items())} }; {early_overflow} early overflow "
        f"reports in {overflow_rows} F3 rows. Conclusion: "
        + ("the floor is exactly the documented bound and the threshold stays at eps* while every quantity is a "
           "normal number" if f1_analytic == 0 and max(normalised) == 0.0 else
           "the normal-range threshold departs from eps*")
        + f"; {sound_clause}; {witness_clause}.")
    fields["uncertainty"] = ("Resolution comparison is deterministic float arithmetic (relative 1e-12 allowed). "
                             "Subnormal-range codes depend on LAPACK's handling of tiny matrices and may differ "
                             "between BLAS builds; the witness's exact class is exact rational arithmetic.")
    return _finish(fields, findings, PROVIDER_FILES, identity)


# T102 ------------------------------------------------------------------------

KAPPAS = (-3.0, -1.5, -1.1, -0.9, -0.5, 0.5, 0.9, 1.1, 1.5, 3.0)
T102_INSIDE = ((1, 0, 0), (-7, 0, 0), (0, 13, 0), (0, -13, 0), (0, 0, 300), (0, 0, -300), (0, 0, 1000),
               (0, 0, -1000), (20, -20, 5), (100, -60, -40), (-150, 30, 700))
T102_OUTSIDE = ((300, 300, 0), (-300, -300, 0), (0, 520, 0), (-520, 0, 0), (480, 40, 0))


def near_threshold_family(seed, count, kappas=KAPPAS, robust=True):
    """Seeded continuous-time (A, P, x) whose decrease forms sit near the resolution threshold."""
    rng = R.generator(seed)
    family = []
    for i in range(count):
        n = 2 + i % 3
        P = np.eye(n) if i % 2 == 0 else R.random_spd(rng, n, 10.0)
        kappa = kappas[i % len(kappas)]
        A, P = R.near_threshold(rng, n, kappa, P)
        family.append({"n": n, "kappa": kappa, "A": A, "P": P, "x": rng.normal(size=n)})
    if robust:
        for kappa in (-1e6, 1e6):
            for n in (2, 3):
                A, P = R.near_threshold(rng, n, kappa)
                family.append({"n": n, "kappa": kappa, "A": A, "P": P, "x": rng.normal(size=n)})
    return family


def razor_family(seed, count):
    """Cases whose computed |max eig(M)| / resolution lies just above 1, on both sides of the threshold."""
    rng = R.generator(seed)
    family = []
    for i in range(count):
        side = -1.0 if i % 2 == 0 else 1.0
        n = 2 + (i // 2) % 3
        P = None if (i // 6) % 2 == 0 else R.random_spd(rng, n, 10.0)
        A, P, ratio = R.razor_edge(rng, n, P, side, skew=1.0)
        family.append({"n": n, "kappa": side * ratio, "A": A, "P": P, "x": rng.normal(size=n)})
    return family


def _eigvalsh_window():
    """numpy.linalg.eigvalsh against exact power-of-two scaling, inside and outside LAPACK's window."""
    inside, outside = [], []
    for member in near_threshold_family(1020, 12, robust=False):
        M = R.decrease_matrix(member["A"], member["P"])
        base = np.linalg.eigvalsh(M)
        for j in (-400, -100, -3, 5, 64, 400):
            inside.append(bool(np.array_equal(np.ldexp(np.linalg.eigvalsh(np.ldexp(M, j)), -j), base)))
        for j in (-700, -520, 520, 700):
            outside.append(bool(np.array_equal(np.ldexp(np.linalg.eigvalsh(np.ldexp(M, j)), -j), base)))
    return inside, outside


def discrete_family():
    """Discrete (A, P, x): six contractions far from the threshold (seed 1021) and six razor-edge cases (seed 1023)."""
    rng = R.generator(1021)
    family = []
    for i in range(6):
        n = 2 + i % 2
        A = 0.8 * R.random_orthogonal(rng, n) @ np.diag(rng.uniform(0.3, 1.0, n))
        family.append({"A": A, "P": R.kron_lyapunov(A, np.eye(n), "discrete"), "x": rng.normal(size=n),
                       "kind": "contraction"})
    rng = R.generator(1023)
    for i in range(6):
        n = 2 + i % 2
        A, P, ratio = R.razor_edge_discrete(rng, n, -1.0 if i % 2 == 0 else 1.0)
        family.append({"A": A, "P": P, "x": rng.normal(size=n), "kind": "razor edge", "ratio": ratio})
    return family


@task("T102", changed_files=PROVIDER_FILES, regression_tests=(_node("test_t102_power_of_two_scaling"),
                                                                _node("test_eigvalsh_scaling_window"),
                                                                _node("test_discrete_razor_edge_sits_at_the_threshold"), PARTIAL_TEST))
def power_of_two_scaling(ctx):
    fields = _fields(
        "Replacing (A, P, x) by (2^a A, 2^b P, 2^c x) multiplies M, max eig(M) and the resolution by 2^(a+b) "
        "exactly, so no verdict code and no margin ratio changes while every quantity stays normal and LAPACK "
        "does not rescale internally; outside that window rounding differs and near-threshold codes may move, but "
        "never against the exact class of the declared form.",
        "Continuous time: M(2^a A, 2^b P) = 2^(a+b) M(A, P) and resolution likewise (homogeneous of degree one in "
        "each of max|A|, max|P|, max|M|); PLSR divides x by a power of two before evaluating, so c never enters. "
        "Discrete time admits only (P, x) scaling. LAPACK dsyevd rescales by a non-power-of-two factor when "
        "max|M| lies outside [2^-485, 2^485]. Exact power-of-two scaling preserves the exact class of M.",
        ["34 seeded continuous cases (PCG64 seed 102): n = 2..4, P = I or SPD (condition 10), kappa in "
         "{+-0.5, +-0.9, +-1.1, +-1.5, +-3} plus four robust cases (kappa = +-1e6)",
         "16 razor-edge cases (seed 1022, skew 1): computed |max eig(M)|/resolution tuned by bisection to just "
         "above 1 on the certified and the indefinite side",
         f"Inside-window exponent triples (a, b, c): {list(T102_INSIDE)}",
         f"Outside-window triples: {list(T102_OUTSIDE)}",
         "12 discrete-time cases under (0, b, c) scaling: 6 contractions (seed 1021) and 6 razor-edge cases "
         "A = Q diag(s, r), P = I (seed 1023) with computed |max eig(A^T A - I)|/resolution just above 1",
         "The subnormal witness scaled by 2^-1074"],
        "PLSR code and margin ratio (margin/resolution) for each scaled case versus the unscaled case; exact "
        "rational class of each declared form.",
        "Inside the window: identical codes and bitwise-identical margin ratios, continuous and discrete. "
        "Everywhere: no certifying code on a form whose exact class is not negative definite.",
        "Evaluate base and scaled cases with PLSR in one subprocess; count code flips and ratio changes per window; "
        "classify every declared form exactly; independently test numpy.linalg.eigvalsh against exact power-of-two "
        "scaling inside and outside the window.",
        "T103: overflow and underflow of states and parameters; propose that PLSR pre-scale M by a power of two "
        "into LAPACK's window before eigvalsh so verdicts are exactly scale-invariant.",
        ["code flip inside the window", "margin ratio changes inside the window", "x scaling alters the sample",
         "discrete near-threshold code flip under (P, x) scaling",
         "code moved against the exact class outside the window (searched as counterexample)", "subnormal flip"],
        ["The LAPACK window [2^-485, 2^485] is taken from reference dsyevd (RMIN, RMAX); other LAPACK builds may "
         "rescale differently.",
         "Outside-window flip counts depend on the LAPACK/BLAS build; only the soundness property is checked."])
    inside_eig, outside_eig = _eigvalsh_window()
    offline = [finding(
        "numpy.linalg.eigvalsh commutes exactly with power-of-two scaling while max|M| stays in [2^-485, 2^485]",
        "numerical", {"inside_bitwise_equal": sum(inside_eig), "inside_total": len(inside_eig)},
        {"generator": {"name": "near_threshold_family", "seed": 1020},
         "checks": [_check("eigenvalues of 2^j M rescaled by 2^-j versus eigenvalues of M, bitwise",
                           len(inside_eig) - sum(inside_eig), 0.0, kind="invariant")]},
        tolerance=EXACT_TOL, uncertainty=EXACT)]
    family = near_threshold_family(102, 30) + razor_family(1022, 16)
    discrete = discrete_family()
    exact_classes = {f"c{i}:base": R.exact_class(R.exact_form(m["A"], m["P"])) for i, m in enumerate(family)}
    exact_classes.update({f"d{i}:base": R.exact_class(R.exact_form(m["A"], m["P"], "discrete"))
                          for i, m in enumerate(discrete)})
    cases, plan = [], []
    for i, member in enumerate(family):
        A, P, x = member["A"], member["P"], member["x"]
        cases.append(_verdict_case(f"c{i}:base", A, P, x))
        for window, triples in (("inside", T102_INSIDE), ("outside", T102_OUTSIDE)):
            for a, b, c in triples:
                cid = f"c{i}:{a}:{b}:{c}"
                cases.append(_verdict_case(cid, np.ldexp(A, a), np.ldexp(P, b), np.ldexp(x, c)))
                plan.append((f"c{i}:base", cid, window, member["kappa"], (a, b, c)))
    for i, member in enumerate(discrete):
        A, P, x = member["A"], member["P"], member["x"]
        cases.append(_verdict_case(f"d{i}:base", A, P, x, time="discrete"))
        for b, c in ((13, 0), (-13, 5), (0, -700)):
            cid = f"d{i}:{b}:{c}"
            cases.append(_verdict_case(cid, A, np.ldexp(P, b), np.ldexp(x, c), time="discrete"))
            plan.append((f"d{i}:base", cid, "discrete", member.get("ratio"), (0, b, c)))
    A, P, x = _subnormal_witness()
    cases += [_verdict_case("w:unit", A / R.TINY, P, x), _verdict_case("w:sub", A, P, x)]
    try:
        bridge = _bridge(ctx, cases)
    except _Unavailable as exc:
        fields["numerical_result"] = (f"Provider-free: eigvalsh is bitwise power-of-two homogeneous in "
                                      f"{sum(inside_eig)}/{len(inside_eig)} inside-window scalings and in "
                                      f"{sum(outside_eig)}/{len(outside_eig)} outside-window scalings.")
        fields["uncertainty"] = "Deterministic on one platform; the outside-window count depends on the LAPACK build."
        return _finish(fields, offline, PROVIDER_FILES, blocked=str(exc))
    identity, results = bridge["identity"], bridge["results"]
    tally = {w: {"evaluations": 0, "code_flips": 0, "ratio_changes": 0, "unsound": 0}
             for w in ("inside", "outside", "discrete")}
    flips = []
    for base_id, cid, window, kappa, triple in plan:
        base, scaled = results[base_id], results[cid]
        tally[window]["evaluations"] += 1
        # Exact power-of-two scaling preserves the exact class, so one class serves base and scaled forms.
        exact = exact_classes[base_id]
        tally[window]["unsound"] += (_code(scaled) in R.CERTIFYING) and exact != "negative_definite"
        if _code(base) != _code(scaled):
            tally[window]["code_flips"] += 1
            flips.append({"window": window, "kappa": kappa, "abc": list(triple), "base": _code(base),
                          "scaled": _code(scaled), "base_ratio": base["margin_ratio"],
                          "scaled_ratio": scaled["margin_ratio"], "exact_class": exact})
        if base["ok"] and scaled["ok"] and base["margin_ratio"] != scaled["margin_ratio"]:
            tally[window]["ratio_changes"] += 1
    base_unsound = sum(_code(results[key]) in R.CERTIFYING and exact_classes[key] != "negative_definite"
                       for key in exact_classes)
    ctx.artifact_json("scaling-invariance.json", R.jsonable({"tally": tally, "flips": flips,
                                                              "unsound_base_verdicts": base_unsound,
                                                              "eigvalsh_outside_bitwise_equal": sum(outside_eig),
                                                              "eigvalsh_outside_total": len(outside_eig)}))
    witness_unit, witness_sub = results["w:unit"], results["w:sub"]
    base = provider_basis(identity)
    outside_flips = [flip for flip in flips if flip["window"] == "outside"]
    outside_checks = [_check("scaled verdicts with a certifying code on a form whose exact class is not negative "
                             "definite", tally["outside"]["unsound"], 0.0)]
    outside_extra = {}
    if outside_flips:
        outside_checks.append(_check("outside-window code flips observed", len(outside_flips), 1.0, "ge",
                                     kind="invariant"))
        outside_extra["counterexample"] = {
            "statement": "Power-of-two scaling of A and P never changes a PLSR verdict code while all quantities "
                         "stay in the binary64 normal range", "witness": outside_flips[0]}
    witness_checks = [
        _check("unit-scale code is DECREASE_NOT_DEFINITE (1 if so)",
               1.0 if _code(witness_unit) == "DECREASE_NOT_DEFINITE" else 0.0, 1.0, "ge", "invariant"),
        _check("PLSR resolution of the witness scaled by 2^-1074 (underflows to zero)",
               witness_sub.get("resolution", math.nan), 0.0, kind="invariant")]
    witness_extra = {}
    if _code(witness_sub) != _code(witness_unit):
        witness_checks.append(_check("scaled code differs from the unit-scale code (1 if so)", 1.0, 1.0, "ge",
                                     "invariant"))
        witness_extra["counterexample"] = {
            "statement": "Power-of-two scaling of A never changes a PLSR verdict code",
            "witness": {"A_unit": [[-2.0, 5.0], [0.0, -3.0]], "scale": "2^-1074", "P": "I", "x": [1.0, 0.0],
                        "unit_code": _code(witness_unit), "scaled_code": _code(witness_sub)}}
    findings = [
        finding("Power-of-two scaling of (A, P, x) inside LAPACK's window never changes the PLSR code or margin ratio",
                "numerical", {k: tally["inside"][k] for k in ("evaluations", "code_flips", "ratio_changes")},
                {"provider": base, "checks": [
                    _check("code flips against the unscaled verdict", tally["inside"]["code_flips"], 0.0,
                           kind="invariant"),
                    _check("margin ratios not bitwise equal", tally["inside"]["ratio_changes"], 0.0,
                           kind="invariant")]},
                tolerance=EXACT_TOL, uncertainty=EXACT),
        finding("Scaling P and x by powers of two never changes a discrete-time PLSR code, near the threshold "
                "included", "numerical", {k: tally["discrete"][k] for k in ("evaluations", "code_flips")},
                {"provider": base, "checks": [_check("code flips against the unscaled verdict",
                                                     tally["discrete"]["code_flips"], 0.0, kind="invariant")]},
                tolerance=EXACT_TOL, uncertainty=EXACT),
        finding("Outside LAPACK's scaling window power-of-two rescaling of A and P never moves a PLSR code to a "
                "certificate that the exact class contradicts", "numerical",
                {"evaluations": tally["outside"]["evaluations"], "unsound": tally["outside"]["unsound"]},
                {"provider": base, "checks": outside_checks}, tolerance=EXACT_TOL,
                uncertainty=_platform(0.0, "which near-threshold codes flip depends on LAPACK/BLAS rounding; the "
                                           "soundness count does not"), **outside_extra),
        finding("Scaling the witness by 2^-1074 drives its resolution to zero while its unit-scale code is "
                "DECREASE_NOT_DEFINITE; the scaled code is recorded", "numerical",
                {"unit_code": _code(witness_unit), "scaled_code": _code(witness_sub)},
                {"provider": base, "checks": witness_checks}, tolerance=EXACT_TOL,
                uncertainty=_platform(0.0, "the scaled code depends on LAPACK's subnormal handling"), **witness_extra),
        finding("No unscaled PLSR verdict certifies a form whose exact class is not negative definite", "numerical",
                {"verdicts": len(exact_classes), "unsound": base_unsound},
                {"provider": base, "independent_check": _independent(
                    _check("exact rational class of each declared continuous and discrete form", base_unsound, 0.0),
                    identity)}, tolerance=EXACT_TOL, uncertainty=EXACT),
    ] + offline
    witness_clause = ("the subnormal witness flips from " + _code(witness_unit) + " to a false certificate"
                      if _code(witness_sub) in R.CERTIFYING else
                      "the subnormal witness keeps a non-certifying code (" + _code(witness_sub) + ")")
    inside_clause = ("verdicts are exactly power-of-two invariant inside LAPACK's unscaled window and the normal "
                     "range" if tally["inside"]["code_flips"] == tally["inside"]["ratio_changes"] == 0
                     and tally["discrete"]["code_flips"] == 0 else "power-of-two invariance fails inside the window")
    outside_clause = (f"outside it {len(outside_flips)} near-threshold codes move" if outside_flips else
                      "outside it no code moved on this platform")
    unsound_total = sum(t["unsound"] for t in tally.values()) + base_unsound
    fields["numerical_result"] = (
        f"Inside the window: {tally['inside']['evaluations']} scaled evaluations, {tally['inside']['code_flips']} code "
        f"flips, {tally['inside']['ratio_changes']} ratio changes. Outside the window: "
        f"{tally['outside']['evaluations']} evaluations, {tally['outside']['code_flips']} flips, "
        f"{tally['outside']['ratio_changes']} ratio changes, {tally['outside']['unsound']} certificates contradicting "
        f"the exact class. Discrete (P, x) scaling: {tally['discrete']['code_flips']} flips in "
        f"{tally['discrete']['evaluations']} (half of the cases within about one resolution of the threshold). "
        f"Subnormal witness: {_code(witness_unit)} -> {_code(witness_sub)}. eigvalsh bitwise homogeneous in "
        f"{sum(inside_eig)}/{len(inside_eig)} inside and {sum(outside_eig)}/{len(outside_eig)} outside scalings. "
        f"Conclusion: {inside_clause}; {outside_clause}"
        + (", never against the exact class" if unsound_total == 0 else
           f", and {unsound_total} certificates contradict the exact class")
        + f"; {witness_clause}.")
    fields["uncertainty"] = ("Inside-window results are exact (bitwise). Outside-window flip counts depend on the "
                             "LAPACK/BLAS build and on how close each case sits to the threshold; they are retained "
                             "in the artifact, and only the soundness count is a checked finding value.")
    return _finish(fields, findings, PROVIDER_FILES, identity)


# T103 ------------------------------------------------------------------------

DBL_MAX = float(np.finfo(float).max)
T103_A = np.array([[-1.0, 2.0], [0.0, -3.0]])
T103_STATES = ((1.0, 1.0), (R.TINY, 0.0), (R.TINY, R.TINY), (-R.TINY, 3 * R.TINY), (2.0 ** -1022, 2.0 ** -1022),
               (1e-300, -1e-300), (DBL_MAX, -DBL_MAX), (DBL_MAX, R.TINY), (2.0 ** 1000, 3 * 2.0 ** 990), (0.0, 0.0))
T103_LEVEL_P = (-1060, -1000, -500, 0, 500, 1000)
T103_LEVEL_E = (-1074, -1000, -700, -540, -537, -300, 0, 300, 511, 512, 540, 1000, 1023)


def level_scan():
    """Declared (P = 2^p I, x = 2^e e1, level 2^(p+2e-+1)) with exact V = 2^(p+2e); documented-rule prediction."""
    rows = []
    for p in T103_LEVEL_P:
        for e in T103_LEVEL_E:
            for offset, exceeded in ((-1, True), (1, False)):
                level_exponent = p + 2 * e + offset
                if not -1074 <= level_exponent <= 1023:
                    continue  # the level itself is not a finite positive binary64 number
                exponent = e - 1 + 1  # frexp(2^e) = (0.5, e + 1), so PLSR divides by 2^e: unit state e1
                documented = R.documented_level_exceeded(float(np.ldexp(1.0, p)), exponent,
                                                         float(np.ldexp(1.0, level_exponent)))
                rows.append({"p": p, "e": e, "log2_level": level_exponent, "log2_V": p + 2 * e,
                             "exact_exceeded": exceeded, "documented_exceeded": documented})
    return rows


def _level_counts(rows, key):
    missed = sum(r["exact_exceeded"] and not r[key] for r in rows)
    spurious = sum(r[key] and not r["exact_exceeded"] for r in rows)
    return missed, spurious


@task("T103", changed_files=PROVIDER_FILES, regression_tests=(_node("test_t103_overflow_underflow"),
                                                                _node("test_level_gate_prediction"), PARTIAL_TEST))
def overflow_underflow(ctx):
    fields = _fields(
        "States anywhere in binary64 are classified with the code unchanged through PLSR's power-of-two state "
        "scaling (components more than about 2^1022 below the largest one become subnormal in the scaled state "
        "and vanish beyond about 2^1075, which a code decided relative to |x|^2 does not see); matrices whose "
        "arithmetic leaves binary64 return NUMERICAL_OVERFLOW or an input refusal; no near-limit input yields a "
        "false certificate or a missed level-set exceedance.",
        "V(x) = x^T P x and x^T M x are homogeneous of degree two in x, so PLSR evaluates at x / 2^e with the "
        "unit state in [1, 2). The level gate decides V > level as scaled_value > level / s^2 and treats an "
        "infinite s^2 as 'exceeded' and a zero s^2 as 'not exceeded'. Exact truth: V = 2^(p + 2e) for "
        "P = 2^p I and x = 2^e e1.",
        ["A = [[-1, 2], [0, -3]], P = I with ten states from 2^-1074 to 1.797e308 (mixed scales included) and "
         "non-finite states", "Level scan: A = -I, P = 2^p I (p in -1060..1000), x = 2^e e1 (e in -1074..1023), "
         "level = 2^(p + 2e -+ 1)", "Near-limit matrices: A = -2^1000 I with P = 2^30 I; A = -2^511 I or -2^512 I "
         "with P = 2^511 I; P = 2^1022 I with x = (1.5, 1.5); affine A(theta) = -I + theta c I, theta = +-1e308",
         "The subnormal witness A = [[-2, 5], [0, -3]] 2^-1074, P = I"],
        "PLSR code (or raised input error) per case; exact exponent arithmetic for V against the level; exact "
        "rational class of the witness's decrease form.",
        "State scaling never changes the code; levels are decided exactly; overflow returns NUMERICAL_OVERFLOW; "
        "certificates only for exactly negative definite forms.",
        "One PLSR subprocess evaluates every case; CIW predicts the level decisions from the documented rule and "
        "from exact exponents and recomputes where the decrease form overflows.",
        "T104: semidefinite and skew-symmetric edge cases; propose upstream that the level gate compare exponents "
        "(log2 V = 2 e + log2 scaled_value) instead of forming s^2, and that forming A(theta) report "
        "NUMERICAL_OVERFLOW rather than raise.",
        ["state scaling changes the code", "non-finite state accepted", "level gate misses an exceedance",
         "level gate reports a spurious exceedance", "overflow not reported as NUMERICAL_OVERFLOW",
         "overflow while forming A(theta) raises", "false certificate from subnormal arithmetic"],
        ["Near-limit inputs are synthetic binary64 numbers chosen to reach the limits, not plant data.",
         "The documented level rule is transcribed from the runtime source at the pinned commit, so agreement "
         "with it is a same-specification check; the exact exponent comparison is the independent reference.",
         "Mixed-scale states lose their smallest components in the scaled evaluation; with a strictly negative "
         "definite test form this cannot change the code and is not probed further here."])
    rows = level_scan()
    predicted_missed, predicted_spurious = _level_counts(rows, "documented_exceeded")
    offline = [finding(
        "The documented level rule, re-derived in CIW, misses exceedances when s^2 underflows and reports spurious "
        "ones when s^2 overflows", "numerical",
        {"cases": len(rows), "predicted_missed": predicted_missed, "predicted_spurious": predicted_spurious},
        {"checks": [_check("predicted missed exceedances (exact exponent comparison)", predicted_missed, 1.0, "ge"),
                    _check("predicted spurious exceedances", predicted_spurious, 1.0, "ge")]},
        tolerance={"abs": 0.0, "rel": 0.0}, uncertainty=EXACT)]
    P_ok = np.eye(2)
    cases = [_verdict_case(f"s{i}", T103_A, P_ok, state) for i, state in enumerate(T103_STATES)]
    cases += [_verdict_case("inf", T103_A, P_ok, (math.inf, 0.0)), _verdict_case("nan", T103_A, P_ok, (math.nan, 0.0))]
    for i, row in enumerate(rows):
        cases.append(_verdict_case(f"L{i}", -np.eye(2), np.ldexp(np.eye(2), row["p"]),
                                   (float(np.ldexp(1.0, row["e"])), 0.0),
                                   level=float(np.ldexp(1.0, row["log2_level"]))))
    limits = {"A=-2^1000 I, P=2^30 I": (-np.ldexp(np.eye(2), 1000), np.ldexp(np.eye(2), 30), (1.0, 0.0)),
              "A=-2^511 I, P=2^511 I": (-np.ldexp(np.eye(2), 511), np.ldexp(np.eye(2), 511), (1.0, 0.0)),
              "A=-2^512 I, P=2^511 I": (-np.ldexp(np.eye(2), 512), np.ldexp(np.eye(2), 511), (1.0, 0.0)),
              "P=2^1022 I, x=(1.5, 1.5)": (-np.ldexp(np.eye(2), -10), np.ldexp(np.eye(2), 1022), (1.5, 1.5))}
    for name, (A, P, x) in limits.items():
        cases.append(_verdict_case(f"M:{name}", A, P, x))
    box = ([-1e308], [1e308])
    cases += [_affine_case("theta:+1e308,c=1", -np.eye(2), [np.eye(2)], box, P_ok, (1.0, 0.0), [1e308]),
              _affine_case("theta:-1e308,c=1", -np.eye(2), [np.eye(2)], box, P_ok, (1.0, 0.0), [-1e308]),
              _affine_case("theta:+1e308,c=2", -np.eye(2), [2 * np.eye(2)], box, P_ok, (1.0, 0.0), [1e308])]
    A, P, x = _subnormal_witness()
    cases.append(_verdict_case("witness", A, P, x))
    try:
        bridge = _bridge(ctx, cases)
    except _Unavailable as exc:
        fields["numerical_result"] = (f"Provider-free: the documented level rule predicts {predicted_missed} missed "
                                      f"and {predicted_spurious} spurious exceedances in {len(rows)} scan cases.")
        fields["uncertainty"] = "Exact exponent arithmetic; no rounding enters the level scan."
        return _finish(fields, offline, PROVIDER_FILES, blocked=str(exc))
    identity, results = bridge["identity"], bridge["results"]
    base = provider_basis(identity)
    reference_code = _code(results["s0"])
    state_codes = {str(list(state)): _code(results[f"s{i}"]) for i, state in enumerate(T103_STATES)}
    state_mismatch = sum(code != reference_code for code in state_codes.values())
    for i, row in enumerate(rows):
        row["plsr_code"] = _code(results[f"L{i}"])
        row["plsr_exceeded"] = row["plsr_code"] == "OUTSIDE_LEVEL_SET"
    missed, spurious = _level_counts(rows, "plsr_exceeded")
    disagreement = sum(r["plsr_exceeded"] != r["documented_exceeded"] for r in rows)
    first_missed = next(r for r in rows if r["exact_exceeded"] and not r["plsr_exceeded"]) if missed else None
    first_spurious = next(r for r in rows if r["plsr_exceeded"] and not r["exact_exceeded"]) if spurious else None
    limit_codes = {name: _code(results[f"M:{name}"]) for name in limits}
    predicted_limits = {name: R.documented_code(A, P, x)["code"] for name, (A, P, x) in limits.items()}
    theta_codes = {cid: _code(results[cid]) for cid in ("theta:+1e308,c=1", "theta:-1e308,c=1", "theta:+1e308,c=2")}
    ctx.artifact_json("near-limits.json", R.jsonable({"states": state_codes, "level_scan": rows,
                                                       "limits": limit_codes, "limits_predicted": predicted_limits,
                                                       "theta": theta_codes,
                                                       "theta_details": {k: results[k].get("error")
                                                                         for k in theta_codes},
                                                       "witness": results["witness"]}))
    level_witness = {}
    for key, row in (("missed", first_missed), ("spurious", first_spurious)):
        if row is not None:
            level_witness[key] = dict(row, A="-I", P=f"2^{row['p']} I", x=f"(2^{row['e']}, 0)")
    findings = [
        finding("PLSR decides the level gate exactly as the documented rule predicts across the near-limit scan",
                "numerical", {"cases": len(rows), "disagreements": disagreement},
                {"provider": base, "checks": [
                    _check("level decisions differing from the documented rule transcribed in CIW (same "
                           "specification)", disagreement, 0.0, kind="analytic")]},
                tolerance=EXACT_TOL, uncertainty=EXACT),
        finding("States from 2^-1074 to the largest binary64 number leave the PLSR code of a fixed negative definite "
                "form unchanged", "numerical", {"states": len(state_codes), "mismatches": state_mismatch,
                                                 "code": reference_code},
                {"provider": base, "checks": [_check("codes differing from the unit-state code", state_mismatch, 0.0,
                                                     kind="invariant")]},
                tolerance=EXACT_TOL, uncertainty=EXACT),
        finding("Non-finite states are refused as input errors, not classified", "numerical",
                {"inf": _code(results["inf"]), "nan": _code(results["nan"])},
                {"provider": base, "checks": [_refusal("x = (inf, 0)", "raises ValueError", _code(results["inf"])),
                                              _refusal("x = (nan, 0)", "raises ValueError", _code(results["nan"]))]},
                tolerance=EXACT_TOL, uncertainty=EXACT),
        finding("Matrices whose decrease form or scaled V leaves binary64 return NUMERICAL_OVERFLOW as predicted, "
                "and a control at the limit is still certified",
                "numerical", limit_codes,
                {"provider": base, "checks": [_check("codes differing from the CIW transcription of the documented "
                                                     "order", sum(limit_codes[k] != predicted_limits[k]
                                                                  for k in limits), 0.0, kind="analytic")]},
                tolerance=EXACT_TOL, uncertainty=EXACT),
        finding("The PLSR level gate misses exceedances when s^2 underflows: V > level is certified", "numerical",
                {"missed": missed, "cases": len(rows)},
                {"provider": base, "checks": [_check("scan cases with exact V > level and code != OUTSIDE_LEVEL_SET",
                                                     missed, 1.0, "ge")]},
                tolerance=EXACT_TOL,
                counterexample={"statement": "OUTSIDE_LEVEL_SET is returned whenever V(x) exceeds the declared level",
                                "witness": level_witness["missed"]} if "missed" in level_witness else None,
                uncertainty=EXACT),
        finding("The PLSR level gate reports OUTSIDE_LEVEL_SET for V below the level when s^2 overflows", "numerical",
                {"spurious": spurious, "cases": len(rows)},
                {"provider": base, "checks": [_check("scan cases with exact V < level and code OUTSIDE_LEVEL_SET",
                                                     spurious, 1.0, "ge")]},
                tolerance=EXACT_TOL,
                counterexample={"statement": "The level gate is decided exactly through the power-of-two scaling",
                                "witness": level_witness["spurious"]} if "spurious" in level_witness else None,
                uncertainty=EXACT),
    ]
    raised = theta_codes["theta:+1e308,c=2"]
    findings.append(finding(
        "A finite in-box theta whose A(theta) overflows raises an input error instead of NUMERICAL_OVERFLOW",
        "numerical", theta_codes,
        {"provider": base, "checks": [_refusal("theta = 1e308, A1 = 2I (A(theta) = inf)", "raises ValueError", raised),
                                      _refusal("theta = 1e308, A1 = I (A finite, M overflows)", "NUMERICAL_OVERFLOW",
                                               theta_codes["theta:+1e308,c=1"])]},
        tolerance=EXACT_TOL,
        counterexample={"statement": "Every finite in-box sample yields a runtime-status-v1 code",
                        "witness": {"A0": "-I", "A1": "2I", "box": [-1e308, 1e308], "theta": 1e308,
                                    "error": results["theta:+1e308,c=2"].get("error")}}, uncertainty=EXACT))
    findings.append(_witness_finding(
        results["witness"], identity, "No binary64 input near the representable limits yields a false certificate",
        WITNESS_CLAIM))
    findings += offline
    witness_code = _code(results["witness"])
    refutations = []
    if missed or spurious:
        refutations.append("level-gate errors (" + ", ".join(
            part for part, count in (("missed", missed), ("spurious", spurious)) if count) + ")")
    if raised.startswith("raises"):
        refutations.append("an input error raised for an in-box theta")
    if witness_code in R.CERTIFYING:
        refutations.append("a false certificate for a subnormal plant")
    behaved = state_mismatch == 0 and all(limit_codes[k] == predicted_limits[k] for k in limits)
    fields["numerical_result"] = (
        f"{len(state_codes)} states from 2^-1074 to 1.797e308: {state_mismatch} code changes (all {reference_code}). "
        f"Level scan ({len(rows)} cases): {missed} missed exceedances (certified with V > level, s^2 underflowed), "
        f"{spurious} spurious exceedances (s^2 overflowed), {disagreement} disagreements with the documented rule. "
        f"Near-limit matrices: {limit_codes}. theta overflow: {theta_codes}. Subnormal witness: {witness_code}. "
        "Conclusion: " + ("state scaling and matrix overflow behave as documented" if behaved else
                          "state scaling or matrix overflow departs from the documented order")
        + ("; the hypothesis is refuted by " + ", by ".join(refutations) if refutations else
           "; no refutation of the hypothesis was observed") + ".")
    fields["uncertainty"] = ("The level scan and the overflow cases involve only exact powers of two, so they are "
                             "platform-independent; the witness depends on LAPACK's subnormal handling.")
    return _finish(fields, findings, PROVIDER_FILES, identity)


# T104 ------------------------------------------------------------------------

def edge_cases():
    """Skew-symmetric, marginal, defective, semidefinite-Q and orthogonal cases with exact classes."""
    rng = R.generator(104)
    cases = []

    def add(name, A, P, x, time="continuous", group=""):
        exact = R.exact_form(A, P, time)
        cases.append({"name": name, "group": group, "A": np.asarray(A, dtype=float), "P": np.asarray(P, dtype=float),
                      "x": np.asarray(x, dtype=float), "time": time, "exact_class": R.exact_class(exact),
                      "exact_zero": all(v == 0 for row in exact for v in row),
                      "P_exact_pd": R.positive_definite(R.fractions(P))})

    for n in range(2, 7):
        G = rng.normal(size=(n, n))
        S = G - G.T
        add(f"skew n={n}, P=I", S, np.eye(n), rng.normal(size=n), group="skew P=I")
        add(f"skew n={n}, P=SPD", S, R.random_spd(rng, n, 10.0), rng.normal(size=n), group="skew P=SPD")
    add("A = 0", np.zeros((2, 2)), np.eye(2), (1.0, 1.0), group="marginal")
    add("A = diag(0, -1)", np.diag([0.0, -1.0]), np.eye(2), (1.0, 1.0), group="marginal")
    add("A = [[0, 1], [0, 0]]", np.array([[0.0, 1.0], [0.0, 0.0]]), np.eye(2), (1.0, 1.0), group="marginal")
    for lam in (0.25, 0.49, 0.5, 0.51, 1.0, 2.0):
        add(f"Jordan lambda={lam}, P=I", np.array([[-lam, 1.0], [0.0, -lam]]), np.eye(2), (1.0, 0.0),
            group="Jordan P=I")
    for lam in (1e-3, 1e-2, 0.1, 1.0):
        A = np.array([[-lam, 1.0], [0.0, -lam]])
        add(f"Jordan lambda={lam}, P=Lyapunov(Q=I)", A, R.kron_lyapunov(A, np.eye(2)), (0.0, 1.0),
            group="Jordan P=Lyapunov")
    A_q = np.array([[-1.0, 1.0], [0.0, -2.0]])
    add("Q = diag(1, 0), P from CIW solve", A_q, R.kron_lyapunov(A_q, np.diag([1.0, 0.0])), (0.0, 1.0),
        group="semidefinite Q")
    for angle in (0.3, 1.0, 2.0):
        c, s = math.cos(angle), math.sin(angle)
        add(f"rotation {angle} rad, discrete", np.array([[c, -s], [s, c]]), np.eye(2), (1.0, 0.0), "discrete",
            group="orthogonal discrete")
    add("diag(1, -1), discrete", np.diag([1.0, -1.0]), np.eye(2), (1.0, 1.0), "discrete", group="orthogonal discrete")
    return cases


def indefinite_candidates(limit=40000, keep=8):
    """P = R diag(1, 1e-17) R^T, rounded: keep candidates NumPy's eigvalsh calls positive definite.

    Returns up to ``keep`` that are exactly indefinite (negative exact determinant) and ``keep`` exactly
    positive definite controls, found by a seeded search (about one draw in a thousand qualifies); PLSR's own
    acceptance is observed separately.
    """
    rng = R.generator(1041)
    indefinite, definite, searched = [], [], 0
    for _ in range(limit):
        if len(indefinite) >= keep and len(definite) >= keep:
            break
        searched += 1
        angle = rng.uniform(0.0, math.pi)
        rot = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        P = rot @ np.diag([1.0, 1e-17]) @ rot.T
        P = 0.5 * P + 0.5 * P.T
        if float(np.min(np.linalg.eigvalsh(P))) <= 0.0:
            continue
        f = R.fractions(P)
        entry = {"P": P, "weak": rot[:, 1], "exact_det_sign": int(np.sign(R.exact_det(f))),
                 "exact_pd": R.positive_definite(f)}
        target = definite if entry["exact_pd"] else indefinite
        if len(target) < keep:
            target.append(entry)
    return indefinite + definite, searched


def expected_codes(exact_class, exact_bin):
    """Codes consistent with the exact class and the exact position of max eig relative to the resolution.

    Beyond two resolutions the sign must be resolved; within two resolutions the verdict may be inconclusive
    or resolve toward the exact sign, but never against it.
    """
    if exact_bin == "below -2 res":
        return {"CERTIFIED_WITH_MARGIN"}
    if exact_bin == "at or above 2 res":
        return {"DECREASE_NOT_DEFINITE", "NOT_CERTIFIED"}
    allowed = {"NUMERICAL_INCONCLUSIVE"}
    if exact_class == "negative_definite":
        allowed.add("CERTIFIED_WITH_MARGIN")
    if exact_class == "has_positive_eigenvalue":
        allowed |= {"DECREASE_NOT_DEFINITE", "NOT_CERTIFIED"}
    return allowed


@task("T104", changed_files=PROVIDER_FILES, regression_tests=(_node("test_t104_semidefinite_edges"),
                                                                _node("test_edge_case_exact_classes"), PARTIAL_TEST))
def semidefinite_edges(ctx):
    fields = _fields(
        "PLSR never certifies a decrease form that is only semidefinite or indefinite: skew-symmetric plants with "
        "P = I give an exactly zero form and must be NUMERICAL_INCONCLUSIVE; defective (Jordan) plants are "
        "certified with P = I exactly when lambda > 1/2; a semidefinite Q is refused by the solver.",
        "Skew A: A^T + A = 0 exactly; with any P, trace(A^T P + P A) = 0 so the form is never negative definite. "
        "Jordan A = [[-l, 1], [0, -l]], P = I: M = [[-2l, 1], [1, -2l]] is negative definite iff l > 1/2 and "
        "singular at l = 1/2. Orthogonal discrete A: A^T A - I = 0 up to rounding.",
        ["5 skew-symmetric matrices n = 2..6 (PCG64 seed 104) with P = I and with SPD P",
         "marginal A = 0, diag(0, -1), [[0, 1], [0, 0]]; Jordan blocks with P = I (l in 0.25..2) and with the CIW "
         "Lyapunov P (l in 1e-3..1); A = [[-1, 1], [0, -2]] with Q = diag(1, 0); discrete rotations and diag(1, -1)",
         "Candidates P = R diag(1, 1e-17) R^T (seed 1041, up to 40000 draws): 8 that NumPy's eigvalsh calls positive "
         "definite although they are exactly indefinite, and 8 exactly positive definite controls"],
        "PLSR code per case; PLSR solve_lyapunov outcome; quadratic() acceptance; exact rational class of each "
        "declared form and exact positive-definiteness of each declared P.",
        "Certifying codes only where the exact form is negative definite and the exact P is positive definite; "
        "exactly semidefinite forms are inconclusive; exactly indefinite forms are DECREASE_NOT_DEFINITE or "
        "NOT_CERTIFIED.",
        "Classify every declared form exactly (Sylvester criterion and principal minors on dyadic rationals), "
        "evaluate it with PLSR, and compare; probe the solver and the certificate constructor with singular data.",
        "T105: parameter boxes across unit scales; propose that PLSR's QuadraticCertificate check positive "
        "definiteness with an exact LDL^T (or a resolution-aware eigenvalue floor) instead of the eigvalsh sign.",
        ["certificate for a semidefinite or indefinite form", "skew form not inconclusive",
         "Jordan threshold misplaced", "semidefinite Q accepted by the solver", "indefinite P accepted and certified"],
        ["Exact classes describe the declared binary64 matrices; the intended real matrices may differ by rounding.",
         "Codes for semidefinite forms depend on the resolution, which is designed to exceed the rounding error."])
    cases = edge_cases()
    groups = {}
    for case in cases:
        groups.setdefault(case["group"], _counts([])).setdefault(case["exact_class"], 0)
        groups[case["group"]][case["exact_class"]] += 1
    skew_zero = sum(c["exact_zero"] for c in cases if c["group"] == "skew P=I")
    offline = [finding(
        "Exact rational classes of the declared edge-case decrease forms", "numerical",
        {"groups": groups, "skew_P_identity_exactly_zero": skew_zero},
        {"generator": {"name": "edge_cases", "seed": 104},
         "checks": [_check("skew A with P = I whose exact form is not identically zero",
                           5 - skew_zero, 0.0, kind="exact_arithmetic"),
                    _check("skew A with SPD P classified negative definite",
                           sum(c["exact_class"] == "negative_definite" for c in cases
                               if c["group"].startswith("skew")), 0.0)]},
        tolerance={"abs": 0.0, "rel": 0.0}, uncertainty=EXACT)]
    candidates, searched = ctx.memo("lyapunov:indefinite-candidates", indefinite_candidates)
    bridge_cases = [_verdict_case(f"e{i}", c["A"], c["P"], c["x"], time=c["time"]) for i, c in enumerate(cases)]
    for lam in (1e-3, 1e-2, 0.1, 1.0):
        bridge_cases.append({"id": f"solve:{lam}", "op": "solve", "A": [[-lam, 1.0], [0.0, -lam]],
                             "time": "continuous"})
    bridge_cases.append({"id": "solve:psdQ", "op": "solve", "A": [[-1.0, 1.0], [0.0, -2.0]],
                         "Q": [[1.0, 0.0], [0.0, 0.0]], "time": "continuous"})
    bridge_cases.append({"id": "quadratic:psd", "op": "quadratic", "P": [[1.0, 0.0], [0.0, 0.0]]})
    for i, c in enumerate(candidates):
        bridge_cases.append({"id": f"q{i}", "op": "quadratic", "P": _mat(c["P"])})
        bridge_cases.append(_verdict_case(f"q{i}:weak", -np.eye(2), c["P"], c["weak"]))
        bridge_cases.append(_verdict_case(f"q{i}:e1", -np.eye(2), c["P"], (1.0, 0.0)))
    try:
        bridge = _bridge(ctx, bridge_cases)
    except _Unavailable as exc:
        fields["numerical_result"] = f"Provider-free exact classes: {groups}."
        fields["uncertainty"] = "Exact rational arithmetic."
        return _finish(fields, offline, PROVIDER_FILES, blocked=str(exc))
    identity, results = bridge["identity"], bridge["results"]
    base = provider_basis(identity)
    rows, violations, unexpected = [], 0, 0
    for i, c in enumerate(cases):
        code = _code(results[f"e{i}"])
        certifying = code in R.CERTIFYING
        violations += certifying and not (c["exact_class"] == "negative_definite" and c["P_exact_pd"])
        exact_bin = R.resolution_bin(R.exact_form(c["A"], c["P"], c["time"]), R.resolution(c["A"], c["P"], c["time"]))
        unexpected += code not in expected_codes(c["exact_class"], exact_bin)
        rows.append({"name": c["name"], "group": c["group"], "time": c["time"], "exact_class": c["exact_class"],
                     "exact_bin": exact_bin, "code": code, "margin_ratio": results[f"e{i}"].get("margin_ratio")})
    accepted = [i for i, c in enumerate(candidates) if results[f"q{i}"]["ok"]]
    indefinite_accepted = [i for i in accepted if not candidates[i]["exact_pd"]]
    candidate_codes = _counts([_code(results[f"q{i}:{where}"]) for i in accepted for where in ("weak", "e1")])
    candidate_certified = sum(_code(results[f"q{i}:{where}"]) in R.CERTIFYING for i in indefinite_accepted
                              for where in ("weak", "e1"))
    skew_codes = _counts(r["code"] for r in rows if r["group"] == "skew P=I")
    jordan = {r["name"]: r["code"] for r in rows if r["group"] == "Jordan P=I"}
    solves = {key: ("P returned" if results[key]["ok"] else _code(results[key]))
              for key in [f"solve:{lam}" for lam in (1e-3, 1e-2, 0.1, 1.0)] + ["solve:psdQ", "quadratic:psd"]}
    ctx.artifact_json("edge-cases.json", R.jsonable({"rows": rows, "solves": solves,
                                                      "indefinite_candidates": {"accepted": len(accepted),
                                                                                "accepted_exactly_indefinite":
                                                                                    len(indefinite_accepted),
                                                                                "codes": candidate_codes}}))
    findings = [
        finding("PLSR never certifies a semidefinite or indefinite edge-case decrease form", "numerical",
                {"cases": len(cases), "violations": violations},
                {"provider": base, "independent_check": _independent(
                    _check("exact rational class of each declared form and P", violations, 0.0), identity)},
                tolerance={"abs": 0.0, "rel": 0.0}, uncertainty=EXACT),
        finding("Every edge-case code is consistent with its exact class and exact distance from the resolution "
                "(resolved beyond two resolutions, never against the exact sign)", "numerical",
                {"cases": len(cases), "unexpected_codes": unexpected, "skew_P_identity": skew_codes, "jordan": jordan},
                {"provider": base, "checks": [_check("codes outside the set expected for the exact class",
                                                     unexpected, 0.0)]},
                tolerance={"abs": 0.0, "rel": 0.0},
                uncertainty=_platform(0.0, "codes within two resolutions may vary with rounding; the allowed sets "
                                           "account for it")),
        finding("PLSR's Lyapunov solver and certificate constructor refuse a semidefinite Q and a singular P",
                "numerical", solves,
                {"provider": base, "checks": [_refusal("solve_lyapunov with Q = diag(1, 0)", "raises ValueError",
                                                       solves["solve:psdQ"]),
                                              _refusal("quadratic(diag(1, 0))", "raises ValueError",
                                                       solves["quadratic:psd"])]},
                tolerance=EXACT_TOL, uncertainty=EXACT),
    ]
    candidate_checks = [_check("certifying verdicts with an accepted, exactly indefinite P", candidate_certified,
                               0.0)]
    candidate_extra = {}
    if indefinite_accepted:
        first = candidates[indefinite_accepted[0]]
        candidate_checks.append(_check("accepted candidates with negative exact determinant",
                                       len(indefinite_accepted), 1.0, "ge"))
        candidate_extra["counterexample"] = {
            "statement": "A P accepted by PLSR's QuadraticCertificate is exactly positive definite",
            "witness": {"P_hex": R.hexed(first["P"]), "exact_det_sign": first["exact_det_sign"],
                        "weak_direction_code": _code(results[f"q{indefinite_accepted[0]}:weak"]),
                        "e1_code": _code(results[f"q{indefinite_accepted[0]}:e1"])}}
    findings.append(finding(
        "No verdict certifies with a P that quadratic() accepts although it is exactly indefinite", "numerical",
        {"candidates": len(candidates), "certifying_verdicts": candidate_certified},
        {"provider": base, "checks": candidate_checks}, tolerance=EXACT_TOL,
        uncertainty=_platform(0.0, "which candidates quadratic() accepts depends on eigvalsh rounding; the "
                                   "certification count does not"), **candidate_extra))
    findings += offline
    indefinite_codes = _counts(_code(results[f"q{i}:{where}"]) for i in indefinite_accepted
                               for where in ("weak", "e1"))
    constructor_clause = (f"the constructor's eigenvalue-sign test admitted {len(indefinite_accepted)} exactly "
                          f"indefinite P, whose verdicts were {indefinite_codes}" if indefinite_accepted else
                          "the constructor admitted no exactly indefinite candidate on this platform")
    fields["numerical_result"] = (
        f"{len(cases)} edge cases: {violations} certifying verdicts on non-definite forms, {unexpected} codes outside "
        f"the class expectation. Skew with P = I: {skew_codes}. Jordan with P = I: {jordan}. Solver and constructor: "
        f"{solves}. Candidates ({searched} draws searched): {len(accepted)}/{len(candidates)} accepted by "
        f"quadratic(), {len(indefinite_accepted)} of them exactly indefinite; verdicts on accepted candidates: "
        f"{candidate_codes.get('CERTIFICATE_NOT_POSITIVE', 0)} CERTIFICATE_NOT_POSITIVE, "
        f"{sum(v for k, v in candidate_codes.items() if k in R.CERTIFYING)} certifying, "
        f"{sum(candidate_codes.values())} in total. Conclusion: "
        + ("no semidefinite, skew or indefinite edge case was certified" if violations == 0 else
           f"{violations} non-definite edge cases were certified")
        + f"; {constructor_clause}"
        + ("" if candidate_certified == 0 else f", and {candidate_certified} verdicts certified with such a P")
        + ".")
    fields["uncertainty"] = ("Exact classes are exact. Which candidates quadratic() accepts depends on the sign of "
                             "a rounded eigenvalue and may differ between LAPACK builds; it is recorded as a "
                             "counterexample when observed, and only the certification count is a checked value.")
    return _finish(fields, findings, PROVIDER_FILES, identity)


# T105 ------------------------------------------------------------------------

MSD_MASS, MSD_DAMPING, MSD_BOX = 2.0, 3.0, (8.0, 12.0)
MSD_P = np.array([[6.0, 0.75], [0.75, 1.0]])
# name: (state scaling T = diag(t), time factor tau, parameter factor c_theta)
UNIT_SYSTEMS = {
    "m, s, N/m": ((1.0, 1.0), 1.0, 1.0),
    "mm, s, N/mm": ((1e3, 1e3), 1.0, 1e-3),
    "m, ms, N/m": ((1.0, 1e-3), 1e-3, 1.0),
    "um, ms, N/um": ((1e6, 1e3), 1e-3, 1e-6),
    "2^-10 m, 2^-7 s, 2^10 N/m": ((2.0 ** 10, 2.0 ** 3), 2.0 ** -7, 2.0 ** -10),
}


def msd_matrices():
    """SI mass-spring-damper x = (position, velocity): A(k) = A0 + k A1, stiffness k in N/m."""
    A0 = np.array([[0.0, 1.0], [0.0, -MSD_DAMPING / MSD_MASS]])
    A1 = np.array([[0.0, 0.0], [-1.0 / MSD_MASS, 0.0]])
    return A0, A1


def light_damping():
    """k = 10 N/m, c = 1e-8 N s/m with P chosen so the SI decrease form is diag(-kc/m^2, -c/m): margin 5e-9."""
    k, c, m = 10.0, 1e-8, MSD_MASS
    b = c / (2.0 * m)
    A = np.array([[0.0, 1.0], [-k / m, -c / m]])
    P = np.array([[k / m + c * b / m, b], [b, 1.0]])
    return A, P


def to_units(system, A0=None, A1=None, P=None, x=None, theta=None):
    """How a host re-expresses the SI declaration: A' = tau T A T^-1, A1' = A1'/c, P' = T^-1 P T^-1, x' = T x."""
    t, tau, c_theta = UNIT_SYSTEMS[system]
    T, Ti = np.diag(t), np.diag([1.0 / v for v in t])
    out = {}
    if A0 is not None:
        out["A0"] = tau * (T @ A0 @ Ti)
    if A1 is not None:
        out["A1"] = tau * (T @ A1 @ Ti) / c_theta
    if P is not None:
        converted = Ti @ P @ Ti
        out["P"] = 0.5 * converted + 0.5 * converted.T
    if x is not None:
        out["x"] = T @ np.asarray(x, dtype=float)
    if theta is not None:
        out["theta"] = float(theta) * c_theta
    return out


def conversion_scan(count=2000):
    """IEEE facts about converting box bounds: collisions of neighbours and formula disagreements.

    A collision is fl(nextafter(b, inf) c) == fl(b c): a sample just above the bound lands on the converted
    bound. A disagreement is fl(b / (1/c)) != fl(b c) for the same exact quantity.
    """
    rng = R.generator(1051)
    collisions = {1e-3: 0, 1e-6: 0}
    disagreements = {1e-3: 0, 1e-6: 0}
    witnesses = {}
    for _ in range(count):
        bound = float(10.0 ** rng.uniform(-3.0, 6.0))
        above = math.nextafter(bound, math.inf)
        for factor in (1e-3, 1e-6):
            if above * factor == bound * factor:
                collisions[factor] += 1
                witnesses.setdefault(f"collision {factor:g}", bound)
            divided, multiplied = bound / (1.0 / factor), bound * factor
            if divided != multiplied:
                disagreements[factor] += 1
                witnesses.setdefault(f"formula {factor:g}", bound)
    return {"draws": count, "neighbour_collisions": {f"{k:g}": v for k, v in collisions.items()},
            "formula_disagreements": {f"{k:g}": v for k, v in disagreements.items()},
            "witnesses": {name: {"bound": value, "hex": value.hex()} for name, value in sorted(witnesses.items())}}


@task("T105", changed_files=PROVIDER_FILES, regression_tests=(_node("test_t105_unit_scales"),
                                                                _node("test_conversion_scan"), PARTIAL_TEST))
def unit_scales(ctx):
    fields = _fields(
        "The same physical parameter box and plant expressed in different units give the same PLSR verdicts: box "
        "membership is preserved by a monotone conversion and the sign of the decrease form is congruence-invariant.",
        "Mass-spring-damper m = 2 kg, c = 3 N s/m, stiffness k in [8, 12] N/m: A(k) = A0 + k A1, common P = "
        "[[6, 0.75], [0.75, 1]] (exactly negative definite decrease at both vertices, hence on the box). A unit "
        "change is x' = T x, t' = t/tau, k' = c k: A' = tau T A T^-1, P' = T^-1 P T^-1. Inertia is invariant; "
        "eigenvalues, max|A| and max|P| (hence margin and resolution) are not.",
        [f"Unit systems (T, tau, c_theta): {UNIT_SYSTEMS}",
         "Samples: k = 8 and 12 (bounds), nextafter(8, 0) and nextafter(12, inf) (just outside), 8 interior k "
         "(PCG64 seed 1052), states from a normal draw", "Light-damping twin: k = 10, c = 1e-8, SI margin 5e-9",
         "Conversion scan: 2000 bounds 10^U(-3, 6) (seed 1051) with factors 1e-3 and 1e-6"],
        "PLSR code, margin ratio and check_vertices result per unit system; IEEE conversion outcomes.",
        "Identical codes across unit systems for every sample; identical box decisions for bounds and "
        "neighbours; check_vertices passes in every unit system.",
        "Convert the SI declaration into each unit system the way a host would (float arithmetic), evaluate all "
        "samples with PLSR, compare codes per sample, then search the conversion arithmetic for counterexamples.",
        "T106: drive every runtime status; propose that hosts declare boxes with an outward-rounded guard band "
        "and that PLSR document that margin ratios are unit-dependent for non-uniform unit changes.",
        ["code differs across unit systems", "box decision differs at a bound", "just-outside sample admitted",
         "bound conversion formulas disagree", "vertex check fails in some units",
         "near-threshold verdict depends on units"],
        ["Unit conversion follows one host convention (multiply by the factor); other conventions change which "
         "boundary samples collide.", "The stiffness box and damping are illustrative values, not identified ones."])
    A0, A1 = msd_matrices()
    exact_vertices = [R.exact_class(R.exact_form(A0 + k * A1, MSD_P)) for k in MSD_BOX]
    scan = conversion_scan()
    offline = [
        finding("P = [[6, 0.75], [0.75, 1]] is an exact common quadratic certificate for the SI stiffness box [8, 12]",
                "mathematical", {"vertex_classes": exact_vertices},
                {"derivation": "continuous-time decrease form is affine in k, so its largest eigenvalue is convex and "
                               "maximal at a vertex; both vertices checked in exact rational arithmetic",
                 "checks": [_check("vertices whose exact decrease form is not negative definite",
                                   sum(c != "negative_definite" for c in exact_vertices), 0.0)]},
                tolerance={"abs": 0.0, "rel": 0.0}, uncertainty=EXACT),
        finding("Converting box bounds in binary64 collapses neighbours and depends on the formula used", "numerical",
                {"draws": scan["draws"], "neighbour_collisions": scan["neighbour_collisions"],
                 "formula_disagreements": scan["formula_disagreements"]},
                {"generator": {"name": "conversion_scan", "seed": 1051},
                 "checks": [_check("neighbour collisions under x 1e-3", scan["neighbour_collisions"]["0.001"], 1.0,
                                   "ge"),
                            _check("b / 1000 versus b * 0.001 disagreements", scan["formula_disagreements"]["0.001"],
                                   1.0, "ge")]},
                tolerance={"abs": 0.0, "rel": 0.0}, uncertainty=EXACT),
        finding("The declared stiffness box contains the stiffness of a real axis", "physical", None, {},
                expected_not_established=True),
    ]
    rng = R.generator(1052)
    thetas = {"k=8 (bound)": 8.0, "k=12 (bound)": 12.0, "k just below 8": math.nextafter(8.0, 0.0),
              "k just above 12": math.nextafter(12.0, math.inf)}
    for i, k in enumerate(rng.uniform(8.0, 12.0, 8)):
        thetas[f"interior {i}"] = float(k)
    states = {name: rng.normal(size=2) for name in thetas}
    cases, keys = [], []
    for system in UNIT_SYSTEMS:
        conv = to_units(system, A0=A0, A1=A1, P=MSD_P)
        box = ([to_units(system, theta=MSD_BOX[0])["theta"]], [to_units(system, theta=MSD_BOX[1])["theta"]])
        for name, theta in thetas.items():
            sample = to_units(system, x=states[name], theta=theta)
            cid = f"{system}|{name}"
            cases.append(_affine_case(cid, conv["A0"], [conv["A1"]], box, conv["P"], sample["x"], [sample["theta"]]))
            keys.append((system, name))
        c_theta = UNIT_SYSTEMS[system][2]
        for name in ("k=8 (bound)", "k=12 (bound)"):
            theta = thetas[name] / (1.0 / c_theta)  # a second, mathematically equal conversion formula
            sample = to_units(system, x=states[name])
            cases.append(_affine_case(f"{system}|{name}|divided", conv["A0"], [conv["A1"]], box, conv["P"],
                                      sample["x"], [theta]))
        cases.append({"id": f"{system}|vertices", "op": "check_vertices",
                      "plant": {"A0": _mat(conv["A0"]), "terms": [_mat(conv["A1"])], "theta_min": box[0],
                                "theta_max": box[1], "time": "continuous"}, "certificate": {"P": _mat(conv["P"])}})
        A_light, P_light = light_damping()
        light = to_units(system, A0=A_light, P=P_light, x=(1.0, 1.0))
        cases.append(_verdict_case(f"{system}|light", light["A0"], light["P"], light["x"]))
    witness_bounds = {name: scan["witnesses"][name]["bound"] for name in ("collision 0.001", "formula 0.001")
                      if name in scan["witnesses"]}
    for name, bound in witness_bounds.items():
        # A parameter that does not enter A isolates the box decision: in-box is certified, outside is refused.
        if name.startswith("collision"):
            sample, converted_bound = math.nextafter(bound, math.inf), bound * 1e-3
            converted_sample = sample * 1e-3
        else:
            # Bound converted by the formula that rounds lower, sample (equal to the bound) by the other one.
            sample = bound
            converted_bound, converted_sample = sorted((bound * 1e-3, bound / 1000.0))
        for label, box, theta in (("SI", ([0.0], [bound]), sample),
                                  ("x1e-3", ([0.0], [converted_bound]), converted_sample)):
            cases.append(_affine_case(f"{name}|{label}", -np.eye(2), [np.zeros((2, 2))], box, np.eye(2), (1.0, 0.0),
                                      [theta]))
    light_exact = {system: R.exact_class(R.exact_form(to_units(system, A0=light_damping()[0])["A0"],
                                                      to_units(system, P=light_damping()[1])["P"]))
                   for system in UNIT_SYSTEMS}
    try:
        bridge = _bridge(ctx, cases)
    except _Unavailable as exc:
        fields["numerical_result"] = (f"Provider-free: exact vertex classes {exact_vertices}; conversion scan "
                                      f"{scan['neighbour_collisions']} collisions, {scan['formula_disagreements']} "
                                      "formula disagreements.")
        fields["uncertainty"] = "IEEE arithmetic and exact rationals; deterministic."
        return _finish(fields, offline, PROVIDER_FILES, blocked=str(exc))
    identity, results = bridge["identity"], bridge["results"]
    base = provider_basis(identity)
    table = {name: {system: _code(results[f"{system}|{name}"]) for system in UNIT_SYSTEMS} for name in thetas}
    interior = [name for name in thetas if name.startswith("interior") or "(bound)" in name]
    interior_mismatch = sum(len(set(table[name].values())) > 1 for name in interior)
    outside = {name: table[name] for name in ("k just below 8", "k just above 12")}
    admitted = {name: [s for s, code in codes.items() if code != "OUTSIDE_PARAMETER_BOX"]
                for name, codes in outside.items()}
    divided = {f"{system}|{name}": _code(results[f"{system}|{name}|divided"]) for system in UNIT_SYSTEMS
               for name in ("k=8 (bound)", "k=12 (bound)")}
    divided_refused = {key: code for key, code in divided.items() if code == "OUTSIDE_PARAMETER_BOX"}
    vertices = {system: results[f"{system}|vertices"].get("passed") for system in UNIT_SYSTEMS}
    light = {system: _code(results[f"{system}|light"]) for system in UNIT_SYSTEMS}
    light_ratio = {system: results[f"{system}|light"].get("margin_ratio") for system in UNIT_SYSTEMS}
    interior_ratios = [results[f"{system}|interior 0"]["margin_ratio"] for system in UNIT_SYSTEMS]
    ratio_spread = max(interior_ratios) / min(interior_ratios)
    ctx.artifact_json("unit-scales.json", R.jsonable({"codes": table, "divided_bounds": divided,
                                                       "vertices": vertices, "light_damping": light,
                                                       "light_damping_margin_ratio": light_ratio,
                                                       "interior0_margin_ratio": dict(zip(UNIT_SYSTEMS,
                                                                                          interior_ratios)),
                                                       "conversion_scan": scan}))
    findings = [
        finding("Interior and boundary stiffness samples give the same PLSR code in all five unit systems",
                "numerical", {"samples": len(interior), "systems": len(UNIT_SYSTEMS), "mismatches": interior_mismatch},
                {"provider": base, "checks": [_check("samples whose code differs between unit systems",
                                                     interior_mismatch, 0.0, kind="invariant")]},
                tolerance={"abs": 0.0, "rel": 0.0},
                uncertainty=_roundoff(0.0, "margins are 1e6 to 1e13 resolutions; conversions round at 1e-16 "
                                           "relative")),
        finding("check_vertices passes for the converted box in every unit system", "numerical", vertices,
                {"provider": base, "checks": [_check("unit systems whose vertex check fails",
                                                     sum(v is not True for v in vertices.values()), 0.0,
                                                     kind="invariant")]},
                tolerance=EXACT_TOL, uncertainty=_roundoff(0.0, "vertex margins are far above the resolution in "
                                                                "every unit system")),
        finding("Margin ratios are not invariant under non-uniform unit changes", "numerical",
                {"interior0_ratio_spread": ratio_spread},
                {"provider": base, "checks": [_check("max/min margin ratio across unit systems", ratio_spread, 10.0,
                                                     "ge", kind="invariant")]},
                tolerance={"abs": 0.0, "rel": 1e-6},
                uncertainty=_roundoff(1e-12, "relative rounding of float64 margin ratios")),
        finding("The declared box bounds 8 and 12 N/m and their binary64 neighbours keep their SI box decision in all "
                "five unit systems and under both conversion formulas", "numerical",
                {"just_outside_admitted": admitted, "bound_refused_by_second_formula": divided_refused},
                {"provider": base, "checks": [
                    _check("just-outside samples admitted after conversion",
                           sum(len(v) for v in admitted.values()), 0.0, kind="invariant"),
                    _check("bound samples refused when converted as k / (1 / c)", len(divided_refused), 0.0,
                           kind="invariant")]},
                tolerance=EXACT_TOL, uncertainty=_roundoff(0.0, "scalar IEEE products and quotients; identical on "
                                                                "every conforming platform")),
    ]
    witness_codes = {key: _code(results[key])
                     for key in (f"{n}|{unit}" for n in witness_bounds for unit in ("SI", "x1e-3"))}
    for name in witness_bounds:
        si, converted = witness_codes[f"{name}|SI"], witness_codes[f"{name}|x1e-3"]
        collision = name.startswith("collision")
        statement = ("Converting the box and the sample with the same formula preserves box membership"
                     if collision else "Mathematically equal unit conversions give the same box decision")
        claim = ("A parameter just above the SI bound is admitted after multiplying bound and sample by 1e-3"
                 if collision else
                 "A parameter exactly on the SI bound is refused when bound and sample are converted by the two "
                 "mathematically equal formulas k * 0.001 and k / 1000")
        expected = (("OUTSIDE_PARAMETER_BOX", "CERTIFIED_WITH_MARGIN") if collision else
                    ("CERTIFIED_WITH_MARGIN", "OUTSIDE_PARAMETER_BOX"))
        findings.append(finding(
            claim, "numerical", {"SI": si, "x1e-3": converted},
            {"provider": base, "checks": [
                _check("SI and converted box decisions match the witness pattern (1 if so)",
                       1.0 if (si, converted) == expected else 0.0, 1.0, "ge", kind="invariant")]},
            tolerance=EXACT_TOL,
            counterexample={"statement": statement, "witness": dict(scan["witnesses"][name], codes={
                "SI": si, "x1e-3": converted})}, uncertainty=EXACT))
    light_differs = len(set(light.values())) > 1
    findings.append(finding(
        "The light-damping plant's verdict depends on the unit system although its exact decrease form is "
        "negative definite in all of them", "numerical", light,
        {"provider": base, "checks": [
            _check("distinct codes across unit systems", len(set(light.values())), 2.0, "ge", kind="invariant"),
            _check("unit systems whose exact decrease form is not negative definite",
                   sum(c != "negative_definite" for c in light_exact.values()), 0.0)]},
        tolerance=EXACT_TOL,
        counterexample={"statement": "The same physical plant in different units gets the same PLSR verdict",
                        "witness": {"codes": light, "margin_ratio": light_ratio}} if light_differs else None,
        uncertainty=_roundoff(0.0, "margin ratios differ by orders of magnitude between unit systems (0.0056 to "
                                   "45000), far from the threshold 1")))
    findings += offline
    boundary_clause = ("boundary decisions for the declared MSD box survive every conversion, but isolated bounds "
                       "collide or depend on the conversion formula" if not any(admitted.values())
                       and not divided_refused else "boundary decisions depend on how bounds and samples are rounded")
    fields["numerical_result"] = (
        f"Interior/bound samples: {interior_mismatch} code mismatches across {len(UNIT_SYSTEMS)} unit systems; "
        f"vertex check passed in {sum(v is True for v in vertices.values())}/{len(vertices)}. Just-outside MSD "
        f"samples admitted: {sum(len(v) for v in admitted.values())}; MSD bound samples refused when converted as "
        f"k/(1/c): {len(divided_refused)}. Interior margin ratio spread {ratio_spread:.3g}x. Light damping codes {light} "
        f"(ratios { {k: float(f'{v:.3g}') for k, v in light_ratio.items()} }; exact classes {light_exact}). "
        f"Isolated box witnesses: {witness_codes}. Conversion scan: "
        f"{scan['neighbour_collisions']} neighbour collisions and {scan['formula_disagreements']} formula "
        f"disagreements in {scan['draws']} draws. Conclusion: "
        + ("well-inside samples agree in every unit system" if interior_mismatch == 0 else
           "well-inside samples disagree between unit systems")
        + f"; {boundary_clause}"
        + ("; near-threshold verdicts depend on the units because the resolution is not congruence-invariant."
           if light_differs else "; the light-damping verdict did not depend on the units here."))
    fields["uncertainty"] = ("Box decisions and conversions are exact IEEE arithmetic (platform-independent). "
                             "Margin ratios carry float64 rounding of the converted matrices (relative ~1e-15).")
    return _finish(fields, findings, PROVIDER_FILES, identity)


# T106 ------------------------------------------------------------------------

ROTATION = np.array([[0.0, 1.0], [-1.0, 0.0]])
# Codes reachable with declared inputs far from every rounding threshold; CERTIFICATE_NOT_POSITIVE is not.
ROUNDING_FREE_CODES = frozenset(R.RUNTIME_CODES) - {"CERTIFICATE_NOT_POSITIVE"}


def status_paths():
    """One-parameter paths through the decision order; each step is a declared (A, P, x, options) case."""
    I2 = np.eye(2)
    e1 = np.array([1.0, 0.0])
    paths = {
        "required_margin (margin 2)": [dict(A=-I2, P=I2, x=e1, required_margin=r)
                                       for r in (0.0, 1.0, 1.99, 2.0, 2.01, 3.0)],
        "level (V = 4)": [dict(A=-I2, P=I2, x=2 * e1, level=v) for v in (5.0, 4.0, 3.99, 0.0, -1.0)],
        "stability a, A = [[a, 1], [-1, a]]": [dict(A=np.array([[a, 1.0], [-1.0, a]]), P=I2, x=e1)
                                               for a in (-1.0, -1e-10, -1e-15, 0.0, 1e-15, 1e-10, 1.0)],
        "direction phi, A = diag(-1, 1)": [dict(A=np.diag([-1.0, 1.0]), P=I2,
                                                x=np.array([math.cos(math.radians(d)), math.sin(math.radians(d))]))
                                           for d in (0.0, 30.0, 45.0, 60.0, 90.0)],
        "matrix scale k, A = -2^k I": [dict(A=-np.ldexp(I2, k), P=I2, x=e1) for k in (0, 1000, 1022, 1023)],
        "theta, box [-1, 1]": [dict(theta=t) for t in (-1.5, -1.0, 0.0, 1.0, math.nextafter(1.0, math.inf))],
        "theta_dot, rate box [-0.2, 0.2]": [dict(theta=0.0, theta_dot=r) for r in (-0.3, -0.2, 0.0, 0.2, 0.3)],
    }
    return paths


def _path_cases(paths, candidates):
    cases, predictions = [], {}
    box, rate_box = ([-1.0], [1.0]), ([-0.2], [0.2])
    for name, steps in paths.items():
        for j, step in enumerate(steps):
            cid = f"{name}#{j}"
            if "theta" in step:
                theta = step["theta"]
                rate = step.get("theta_dot")
                cases.append(_affine_case(cid, -np.eye(2), [ROTATION], box, np.eye(2), (1.0, 0.0), [theta],
                                          None if rate is None else [rate], rate_box=rate_box))
                inside = -1.0 <= theta <= 1.0 and (rate is None or -0.2 <= rate <= 0.2)
                predictions[cid] = R.documented_code(-np.eye(2) + theta * ROTATION, np.eye(2), (1.0, 0.0),
                                                     in_box=inside)["code"]
            else:
                options = {k: step[k] for k in ("level", "required_margin") if k in step}
                cases.append(_verdict_case(cid, step["A"], step["P"], step["x"], **options))
                predictions[cid] = R.documented_code(step["A"], step["P"], step["x"], **options)["code"]
    for i, candidate in enumerate(candidates):
        cid = f"indefinite P#{i}"
        cases.append(_verdict_case(cid, -np.eye(2), candidate["P"], candidate["weak"]))
        predictions[cid] = R.documented_code(-np.eye(2), candidate["P"], candidate["weak"])["code"]
    return cases, predictions


@task("T106", changed_files=PROVIDER_FILES, regression_tests=(_node("test_t106_status_coverage"),
                                                                _node("test_documented_decision_order"),
                                                                _node("test_t106_offline_findings_without_the_provider"), PARTIAL_TEST))
def status_transitions(ctx):
    fields = _fields(
        "Each of the nine runtime-status-v1 codes is reachable with declared inputs, codes change along "
        "one-parameter paths exactly as the documented decision order predicts, and the runtime refuses the five "
        "host-owned codes.",
        "Decision order: outside box -> NUMERICAL_OVERFLOW -> CERTIFICATE_NOT_POSITIVE (min eig P <= 0 or V < 0) "
        "-> OUTSIDE_LEVEL_SET -> NOT_CERTIFIED (x^T M x > res |x|^2) -> CERTIFIED_WITH_MARGIN / MARGIN_LOW "
        "(margin > res, MARGIN_LOW iff margin <= required margin) -> DECREASE_NOT_DEFINITE (max eig M > res) -> "
        "NUMERICAL_INCONCLUSIVE.",
        ["Seven paths: required margin 0..3; level 5..-1; stability a in [-1, 1] for [[a, 1], [-1, a]]; state "
         "direction 0..90 degrees for diag(-1, 1); matrix scale 2^0..2^1023; theta across [-1, 1]; theta_dot "
         "across [-0.2, 0.2]", "8 candidate P that NumPy calls positive definite but are exactly indefinite "
         "(seed 1041) evaluated along their weak direction", "Host-owned codes MODEL_MISMATCH, STALE_STATE, "
         "INVALID_SENSOR_DATA, CERTIFICATE_EXPIRED, RUNTIME_FAULT"],
        "PLSR verdict code per step; require_status and Verdict construction outcomes for host-owned codes; the "
        "runtime's published constants.",
        "The eight codes that do not depend on rounding are reached on paths far from every threshold, each path "
        "step equals the CIW transcription of the documented order, CERTIFICATE_NOT_POSITIVE is reached only "
        "through rounding and never with a certificate, and every host-owned code is refused.",
        "Build each path, evaluate all steps in one PLSR subprocess, recompute each expected code in CIW from A, P "
        "and x, and tabulate transitions and coverage.",
        "T107: near-boundary spectra; add an upstream regression that pins one witness per code, including a "
        "CERTIFICATE_NOT_POSITIVE witness that does not depend on eigvalsh rounding.",
        ["a code unreachable", "a transition out of documented order", "a host-owned code accepted",
         "CERTIFICATE_NOT_POSITIVE only reachable through rounding"],
        ["CERTIFICATE_NOT_POSITIVE is reached only when a P accepted by quadratic() evaluates V < 0, which depends "
         "on eigvalsh rounding; min eig P <= 0 cannot follow a passed construction check. Its reachability is "
         "therefore recorded, not required.",
         "The CIW transcription shares the documented specification with the runtime, so it checks implementation "
         "against specification (a same-specification check), not the specification itself."])
    candidates = [c for c in ctx.memo("lyapunov:indefinite-candidates", indefinite_candidates)[0]
                  if not c["exact_pd"]]
    paths = status_paths()
    cases, predictions = _path_cases(paths, candidates)
    predicted_codes = sorted(set(predictions.values()))
    path_ids = [cid for cid in predictions if not cid.startswith("indefinite")]
    stable_codes = sorted({predictions[cid] for cid in path_ids} & ROUNDING_FREE_CODES)
    authority = finding(
        "A CERTIFIED_WITH_MARGIN verdict (operationally_acceptable) authorizes actuation", "actuator_authority", None,
        {"derivation": "runtime-status-v1: operationally_acceptable is not an authorization; host statuses and "
                       "machine-safety functions are outside the evaluator"})
    offline = [finding("The documented decision order, transcribed in CIW, assigns the eight rounding-free codes to "
                       "the constructed path steps", "numerical", {"codes": stable_codes},
                       {"generator": {"name": "status_paths", "seed": None},
                        "checks": [_check("rounding-free runtime codes predicted on the paths", len(stable_codes),
                                          float(len(ROUNDING_FREE_CODES)), "ge", kind="analytic")]},
                       tolerance=EXACT_TOL, uncertainty=EXACT), authority]
    for code in R.HOST_OWNED:
        cases.append({"id": f"require:{code}", "op": "require_status", "code": code})
        cases.append({"id": f"verdict:{code}", "op": "host_verdict", "code": code})
    cases.append({"id": "constants", "op": "constants"})
    try:
        bridge = _bridge(ctx, cases)
    except _Unavailable as exc:
        fields["numerical_result"] = (f"Provider-free prediction covers {len(stable_codes)} rounding-free codes on "
                                      f"the paths and {len(predicted_codes)} codes including the indefinite-P "
                                      f"witnesses: {predicted_codes}.")
        fields["uncertainty"] = "Deterministic re-derivation on this platform."
        return _finish(fields, offline, PROVIDER_FILES, blocked=str(exc))
    identity, results = bridge["identity"], bridge["results"]
    base = provider_basis(identity)
    observed = {cid: _code(results[cid]) for cid in predictions}
    path_mismatches = {cid: {"observed": observed[cid], "predicted": predictions[cid]} for cid in path_ids
                       if observed[cid] != predictions[cid]}
    reached_stable = sorted({observed[cid] for cid in path_ids} & ROUNDING_FREE_CODES)
    witness_ids = [cid for cid in predictions if cid.startswith("indefinite")]
    witness_codes = _counts(observed[cid] for cid in witness_ids)
    witness_certified = sum(observed[cid] in R.CERTIFYING for cid in witness_ids)
    not_positive = witness_codes.get("CERTIFICATE_NOT_POSITIVE", 0)
    reached = sorted(set(observed.values()) & set(R.RUNTIME_CODES))
    transitions = {name: [observed[f"{name}#{j}"] for j in range(len(steps))] for name, steps in paths.items()}
    host = {code: {"require_status": _code(results[f"require:{code}"]),
                   "Verdict": _code(results[f"verdict:{code}"])} for code in R.HOST_OWNED}
    constants = {k: v for k, v in results["constants"].items() if k not in ("ok", "id", "warnings")}
    coverage = {code: sorted(cid for cid, got in observed.items() if got == code)[:3] for code in R.RUNTIME_CODES}
    ctx.artifact_json("status-coverage.json", R.jsonable({"coverage_examples": coverage, "transitions": transitions,
                                                           "observed": observed, "predicted": predictions,
                                                           "host_owned": host, "constants": constants}))
    lines = ["| Code | Reached | Example step |", "| --- | --- | --- |"]
    lines += [f"| {code} | {'yes' if coverage[code] else 'no'} | {coverage[code][0] if coverage[code] else '-'} |"
              for code in R.RUNTIME_CODES]
    ctx.artifact_text("status-coverage.md", "\n".join(lines) + "\n")
    not_positive_checks = [_check("certifying verdicts on the exactly indefinite P witnesses", witness_certified, 0.0,
                                  kind="invariant")]
    if not_positive:
        not_positive_checks.append(_check("CERTIFICATE_NOT_POSITIVE verdicts observed", not_positive, 1.0, "ge",
                                          kind="invariant"))
    findings = [
        finding("The eight rounding-free runtime-status-v1 codes are reached on one-parameter paths far from every "
                "threshold", "numerical", {"codes": reached_stable},
                {"provider": base, "checks": [_check("rounding-free codes reached", len(reached_stable),
                                                     float(len(ROUNDING_FREE_CODES)), "ge", kind="invariant")]},
                tolerance=EXACT_TOL, uncertainty=EXACT),
        finding("Codes along each one-parameter path follow the documented decision order", "numerical", transitions,
                {"provider": base, "checks": [
                    _check("path steps differing from the CIW transcription of the documented order (same "
                           "specification)", len(path_mismatches), 0.0, kind="analytic")]},
                tolerance=EXACT_TOL, uncertainty=EXACT),
        finding("The exactly indefinite P witnesses are never certified along their weak direction; whether they "
                "reach CERTIFICATE_NOT_POSITIVE is recorded", "numerical",
                {"witnesses": len(witness_ids), "certifying": witness_certified,
                 "certificate_not_positive_reached": not_positive > 0},
                {"provider": base, "checks": not_positive_checks}, tolerance=EXACT_TOL,
                uncertainty=_platform(0.0, "whether the witnesses reach CERTIFICATE_NOT_POSITIVE depends on eigvalsh "
                                           "and dot-product rounding; the certification count does not")),
        finding("The runtime refuses to emit the five host-owned status codes", "numerical", host,
                {"provider": base, "checks": [
                    _refusal(f"{way} for {code}", "raises ValueError", outcome[way])
                    for code, outcome in host.items() for way in ("require_status", "Verdict")]},
                tolerance=EXACT_TOL, uncertainty=EXACT),
        finding("Pinned runtime constants: resolution factor, numerical policy, status vocabulary", "provenance",
                constants, {"provider": base, "checks": [
                    _check("DECREASE_RESOLUTION_FACTOR minus the documented 1.0",
                           constants.get("DECREASE_RESOLUTION_FACTOR", math.nan) - 1.0, 0.0, kind="invariant"),
                    _check("NUMERICAL_POLICY_VERSION is float64-decrease-v1 (1 if so)",
                           1.0 if constants.get("NUMERICAL_POLICY_VERSION") == "float64-decrease-v1" else 0.0, 1.0,
                           "ge", kind="invariant"),
                    _check("status codes outside or missing from the documented nine",
                           len(set(constants.get("RUNTIME_STATUSES", [])) ^ set(R.RUNTIME_CODES)), 0.0,
                           kind="invariant"),
                    _check("host-owned codes outside or missing from the documented five",
                           len(set(constants.get("HOST_OWNED_STATUSES", [])) ^ set(R.HOST_OWNED)), 0.0,
                           kind="invariant")]}, tolerance=EXACT_TOL, uncertainty=EXACT),
    ] + offline
    fields["numerical_result"] = (
        f"Codes reached: {len(reached)}/9 ({reached}); rounding-free codes on the paths: "
        f"{len(reached_stable)}/{len(ROUNDING_FREE_CODES)}. {len(path_mismatches)} of {len(path_ids)} path steps "
        f"differ from the CIW transcription of the documented order. Indefinite-P witnesses: {witness_codes} "
        f"({witness_certified} certifying). Transitions: {transitions}. Host-owned codes refused: "
        f"{sum(set(v.values()) == {'raises ValueError'} for v in host.values())}/5. "
        f"Constants: resolution factor {constants.get('DECREASE_RESOLUTION_FACTOR')}, policy "
        f"{constants.get('NUMERICAL_POLICY_VERSION')}.")
    fields["uncertainty"] = ("Path codes are far from thresholds and platform-independent. The indefinite-P "
                             "witnesses depend on eigvalsh and dot-product rounding, so only their non-certification "
                             "is checked and CERTIFICATE_NOT_POSITIVE reachability is recorded.")
    return _finish(fields, findings, PROVIDER_FILES, identity)


# T107 ------------------------------------------------------------------------

T107_KAPPAS = (-8.0, -3.0, -2.2, -1.8, -1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2, 1.8, 2.2, 3.0, 8.0)
BAND = ("[-2, -1) res", "[-1, 0) res", "[0, 1) res", "[1, 2) res")


def boundary_family():
    """Near-boundary continuous cases with exact bins of max eig(M) relative to the resolution."""
    rng = R.generator(107)
    family = []
    for n in (2, 3, 4):
        for spd in (False, True):
            for kappa in T107_KAPPAS:
                P = R.random_spd(rng, n, 10.0) if spd else None
                A, P = R.near_threshold(rng, n, kappa, P)
                family.append({"n": n, "kappa": kappa, "A": A, "P": P, "x": rng.normal(size=n)})
    family += razor_family(1071, 12)
    for member in family:
        exact = R.exact_form(member["A"], member["P"])
        member["resolution"] = R.resolution(member["A"], member["P"])
        member["exact_class"] = R.exact_class(exact)
        member["exact_bin"] = R.resolution_bin(exact, member["resolution"])
    return family


@task("T107", changed_files=PROVIDER_FILES, regression_tests=(_node("test_t107_inconclusive_band"), PARTIAL_TEST))
def inconclusive_band(ctx):
    fields = _fields(
        "PLSR never gives a certifying code to a near-boundary decrease form that is not exactly negative "
        "definite; beyond two resolutions from zero it always resolves the sign; under a declared margin of three "
        "resolutions no case within two resolutions is CERTIFIED_WITH_MARGIN. The specification's stronger "
        "statement -- that near-boundary spectra yield NUMERICAL_INCONCLUSIVE or MARGIN_LOW rather than "
        "CERTIFIED_WITH_MARGIN -- is tested at required_margin 0 as a candidate counterexample.",
        "If the resolution bounds the float64 error e of max eig(M) (|e| <= res), then exact lambda < -2 res gives "
        "margin > res (certified), exact lambda >= 2 res gives max eig > res (not definite), and exact lambda >= 0 "
        "can never give margin > res. With required margin 3 res, certification needs margin > 3 res, impossible "
        "for exact lambda >= -2 res.",
        ["90 continuous cases (PCG64 seed 107): n in {2, 3, 4}, P = I or SPD, kappa in "
         f"{list(T107_KAPPAS)} (target max eig = kappa * resolution)",
         "12 razor-edge cases (seed 1071) with computed |max eig| / resolution just above 1",
         "Each case evaluated with required_margin 0 and 3 * resolution"],
        "PLSR code and margin; exact bin of max eig(M) from Sylvester tests on M - t I at t = -2, -1, 0, 1, 2 "
        "resolutions in exact dyadic arithmetic.",
        "No certifying code unless the exact form is negative definite; certified beyond -2 res; resolved beyond "
        "+2 res; with the declared margin no CERTIFIED_WITH_MARGIN within the band. Without a declared margin, "
        "exactly negative definite band cases may be certified (soundly) or left inconclusive.",
        "Generate cases, bin their exact spectra, evaluate with PLSR at both margins and tabulate codes per bin.",
        "T108: required-margin monotonicity; measure how much of the [-2, 0) res band a less conservative "
        "eigensolver term (p(n) = n instead of n^2) would recover without losing soundness against exact bins.",
        ["certifying code on a form that is not exactly negative definite", "unresolved sign beyond two "
         "resolutions", "CERTIFIED_WITH_MARGIN inside the band under a declared margin",
         "MARGIN_LOW inconsistent with the declared margin",
         "CERTIFIED_WITH_MARGIN inside the band at required_margin 0 (searched as counterexample)"],
        ["Exact bins describe the declared binary64 matrices; the targets kappa are only approximately realised "
         "because A is rounded.", "The band refusal rate depends on the generator and is not a property of "
         "plants in general."])
    family = boundary_family()
    bins = _counts(member["exact_bin"] for member in family)
    offline = [finding("The near-boundary family populates every exact resolution bin of max eig(M)",
                       "numerical", {"cases": len(family), "bins": bins},
                       {"generator": {"name": "boundary_family", "seed": 107},
                        "checks": [_check("exact bins without a case (six bins from below -2 res to above 2 res)",
                                          6 - len(bins), 0.0)]},
                       tolerance={"abs": 2.0, "rel": 0.0},
                       uncertainty=_platform(2.0, "bins of razor-edge cases may shift between BLAS builds"))]
    cases = []
    for i, member in enumerate(family):
        cases.append(_verdict_case(f"b{i}:0", member["A"], member["P"], member["x"]))
        cases.append(_verdict_case(f"b{i}:3", member["A"], member["P"], member["x"],
                                   required_margin=3.0 * member["resolution"]))
    try:
        bridge = _bridge(ctx, cases)
    except _Unavailable as exc:
        fields["numerical_result"] = f"Provider-free exact bins: {bins}."
        fields["uncertainty"] = "Exact bins are exact; targets are approximate."
        return _finish(fields, offline, PROVIDER_FILES, blocked=str(exc))
    identity, results = bridge["identity"], bridge["results"]
    base = provider_basis(identity)
    table = {}
    unsound = unresolved = certified_in_band_declared = margin_low_mismatch = margin_low_seen = 0
    band_nd, band_nd_inconclusive, band_nd_certified = 0, 0, 0
    points = []
    for i, member in enumerate(family):
        zero, declared = results[f"b{i}:0"], results[f"b{i}:3"]
        code0, code3 = _code(zero), _code(declared)
        table.setdefault(member["exact_bin"], {}).setdefault(code0, 0)
        table[member["exact_bin"]][code0] += 1
        unsound += (code0 in R.CERTIFYING or code3 in R.CERTIFYING) and member["exact_class"] != "negative_definite"
        if member["exact_bin"] == "below -2 res":
            unresolved += code0 != "CERTIFIED_WITH_MARGIN"
        if member["exact_bin"] == "at or above 2 res":
            unresolved += code0 not in ("DECREASE_NOT_DEFINITE", "NOT_CERTIFIED")
        if member["exact_bin"] in BAND:
            certified_in_band_declared += code3 == "CERTIFIED_WITH_MARGIN"
            if member["exact_class"] == "negative_definite":
                band_nd += 1
                band_nd_inconclusive += code0 == "NUMERICAL_INCONCLUSIVE"
                band_nd_certified += code0 == "CERTIFIED_WITH_MARGIN"
        margin, res, required = declared["margin"], declared["resolution"], declared["required_margin"]
        if margin > res:
            expected = "MARGIN_LOW" if margin <= required else "CERTIFIED_WITH_MARGIN"
            if code0 == "CERTIFIED_WITH_MARGIN":
                margin_low_mismatch += code3 != expected
                margin_low_seen += code3 == "MARGIN_LOW"
        points.append((member["kappa"], zero["margin_ratio"], code0))
    refusal_rate = band_nd_inconclusive / band_nd if band_nd else 0.0
    band_certified = [i for i, member in enumerate(family) if member["exact_bin"] in BAND
                      and _code(results[f"b{i}:0"]) == "CERTIFIED_WITH_MARGIN"]
    band_certified_unsound = sum(family[i]["exact_class"] != "negative_definite" for i in band_certified)
    within_one = [i for i in band_certified if family[i]["exact_bin"] == "[-1, 0) res"]
    ctx.artifact_json("inconclusive-band.json", R.jsonable({"codes_by_exact_bin": table, "points": points}))
    by_code = {}
    for kappa, ratio, code in sorted(points):
        by_code.setdefault(code, ([], []))
        by_code[code][0].append(kappa)
        by_code[code][1].append(-ratio)
    ctx.artifact_text("ratio-vs-target.svg", svg.line_plot(
        [(code, xs, ys) for code, (xs, ys) in sorted(by_code.items()) if abs(max(xs)) < 20],
        title="T107 computed max eig / resolution against target kappa", xlabel="target kappa",
        ylabel="max eig(M) / resolution", markers=True))
    exact_checker = _independent(_check("exact bins and classes of the declared forms", unsound, 0.0), identity)
    band_extra = {}
    if band_certified:
        chosen = (within_one or band_certified)[0]
        member = family[chosen]
        band_extra["counterexample"] = {
            "statement": "Near-boundary spectra yield NUMERICAL_INCONCLUSIVE or MARGIN_LOW rather than "
                         "CERTIFIED_WITH_MARGIN (T107 specification)",
            "witness": {"case": chosen, "n": member["n"], "exact_bin": member["exact_bin"],
                        "exact_class": member["exact_class"], "code_at_required_margin_0": "CERTIFIED_WITH_MARGIN",
                        "margin_ratio": results[f"b{chosen}:0"]["margin_ratio"], "A_hex": R.hexed(member["A"]),
                        "P_hex": R.hexed(member["P"]), "x": member["x"].tolist()}}
    findings = [
        finding("No near-boundary case receives a certifying code unless its exact decrease form is negative "
                "definite", "numerical", {"cases": len(family), "violations": unsound},
                {"provider": base, "independent_check": exact_checker}, tolerance=EXACT_TOL,
                uncertainty=EXACT),
        finding("Beyond two resolutions from zero PLSR always resolves the sign", "numerical",
                {"resolved_cases": bins.get("below -2 res", 0) + bins.get("at or above 2 res", 0),
                 "unresolved": unresolved},
                {"provider": base, "independent_check": _independent(
                    _check("cases beyond two resolutions not resolved to the exact sign", unresolved, 0.0), identity)},
                tolerance={"abs": 2.0, "rel": 0.0},
                uncertainty=_platform(2.0, "counts near two resolutions may shift between BLAS builds")),
        finding("With a declared margin of three resolutions no case within two resolutions of zero is "
                "CERTIFIED_WITH_MARGIN", "numerical",
                {"band_cases": sum(bins.get(b, 0) for b in BAND), "certified": certified_in_band_declared},
                {"provider": base, "independent_check": _independent(
                    _check("CERTIFIED_WITH_MARGIN in the exact band under required_margin = 3 res",
                           certified_in_band_declared, 0.0), identity)},
                tolerance={"abs": 2.0, "rel": 0.0},
                uncertainty=_platform(2.0, "counts near the band edge may shift between BLAS builds")),
        finding("At required_margin 0 near-boundary spectra within two resolutions of zero receive "
                "CERTIFIED_WITH_MARGIN, and every such certificate is exactly sound", "numerical",
                {"band_cases": sum(bins.get(b, 0) for b in BAND), "certified": len(band_certified),
                 "certified_within_one_resolution": len(within_one), "unsound": band_certified_unsound},
                {"provider": base, "checks": [
                    _check("band cases certified at required_margin 0", len(band_certified), 1.0, "ge",
                           kind="invariant"),
                    _check("band certificates whose exact decrease form is not negative definite",
                           band_certified_unsound, 0.0)]},
                tolerance={"abs": 4.0, "rel": 0.0},
                uncertainty=_platform(4.0, "which band cases resolve depends on last-bit rounding; the [-2, -1) "
                                           "res certifications are robust"), **band_extra),
        finding("MARGIN_LOW appears exactly when the resolvable margin does not exceed the declared margin",
                "numerical", {"margin_low_observed": margin_low_seen > 0, "mismatches": margin_low_mismatch},
                {"provider": base, "checks": [_check("codes differing from the declared-margin rule",
                                                     margin_low_mismatch, 0.0, kind="invariant"),
                                              _check("MARGIN_LOW verdicts observed", margin_low_seen, 1.0, "ge",
                                                     kind="invariant")]},
                tolerance=EXACT_TOL, uncertainty=EXACT),
        finding("Share of exactly negative definite band cases answered NUMERICAL_INCONCLUSIVE without a declared "
                "margin", "numerical", {"inconclusive_share": refusal_rate},
                {"provider": base}, tolerance={"abs": 0.2, "rel": 0.0},
                uncertainty=_platform(0.2, "the share of resolved band cases depends on last-bit rounding")),
    ] + offline
    fields["numerical_result"] = (
        f"{len(family)} cases; codes by exact bin at required_margin 0: {table}. Unsound certifications: {unsound}. "
        f"Unresolved beyond two resolutions: {unresolved}. Certified within the band under the declared margin: "
        f"{certified_in_band_declared}. At required_margin 0, {len(band_certified)} of "
        f"{sum(bins.get(b, 0) for b in BAND)} band cases were CERTIFIED_WITH_MARGIN ({len(within_one)} within one "
        f"resolution), {band_certified_unsound} of them unsound. Exactly negative definite band cases: {band_nd}, of "
        f"which {band_nd_inconclusive} inconclusive ({refusal_rate:.0%}) and {band_nd_certified} certified. "
        f"MARGIN_LOW rule mismatches: {margin_low_mismatch}. Conclusion: "
        + ("every certificate is exactly sound and the sign is resolved beyond two resolutions"
           if unsound == 0 and unresolved == 0 else "the soundness or resolution property failed")
        + ("; the declared margin of three resolutions keeps the band out of CERTIFIED_WITH_MARGIN"
           if certified_in_band_declared == 0 else "; the declared margin did not keep the band out of "
                                                   "CERTIFIED_WITH_MARGIN")
        + ("; without a declared margin the band is not uniformly inconclusive, which refutes the specification's "
           "stronger statement (the certifications are sound)." if band_certified else
           "; without a declared margin no band case was certified."))
    fields["uncertainty"] = ("Exact bins are exact; which band cases resolve depends on last-bit rounding and "
                             "may differ between BLAS builds (share tolerance 0.2, count tolerance 4).")
    return _finish(fields, findings, PROVIDER_FILES, identity)


# T108 ------------------------------------------------------------------------

def margin_grid(margin, res):
    """Sorted nonnegative required margins around the resolution and the observed margin."""
    grid = {0.0, 0.5 * res, res, 2.0 * res, 1e300}
    if margin > 0.0:
        grid |= {0.5 * margin, math.nextafter(margin, 0.0), margin, math.nextafter(margin, math.inf), 2.0 * margin}
    return sorted(value for value in grid if value >= 0.0 and math.isfinite(value))


def monotonicity_violations(sequence):
    """Count property violations along one case's increasing required-margin sequence of verdict dicts."""
    passing = [step["code"] == "CERTIFIED_WITH_MARGIN" for step in sequence]
    meets = [step["meets_required_margin"] for step in sequence]
    codes = {step["code"] for step in sequence}
    # A non-certifying code is decided before the margin is consulted, so it must hold for every margin.
    changed = bool(codes - R.CERTIFYING) and len(codes) > 1
    return {"passing_regained": sum(1 for a, b in zip(passing, passing[1:]) if b and not a),
            "meets_regained": sum(1 for a, b in zip(meets, meets[1:]) if b and not a),
            "noncertifying_code_changed": int(changed),
            "inequality_changed": int(len({s["inequality_certified"] for s in sequence}) > 1)}


@task("T108", changed_files=PROVIDER_FILES, regression_tests=(_node("test_t108_margin_monotonicity"),
                                                                _node("test_documented_rule_is_monotone"), PARTIAL_TEST))
def margin_monotonicity(ctx):
    fields = _fields(
        "Increasing required_margin never turns a failing verdict into a passing one: CERTIFIED_WITH_MARGIN and "
        "meets_required_margin are nonincreasing in the declared margin, non-certifying codes and "
        "inequality_certified do not depend on it, and the switch to MARGIN_LOW happens exactly at "
        "required_margin = margin.",
        "The declared margin r enters the decision order only in the branch margin > res, as MARGIN_LOW iff "
        "margin <= r; meets_required_margin = margin > max(r, res). Both are monotone in r by construction; "
        "negative or non-finite r must be refused.",
        ["24 near-threshold cases (PCG64 seed 108) plus four robust cases and 8 razor-edge cases (seed 1081)",
         "Per case: required margins {0, res/2, res, 2 res, 1e300} and, when the margin is positive, "
         "{margin/2, nextafter(margin, 0), margin, nextafter(margin, inf), 2 margin}",
         "Invalid margins -1e-300, -1, nan, inf"],
        "PLSR code, meets_required_margin and inequality_certified per (case, required margin).",
        "Zero violations of monotonicity or invariance; MARGIN_LOW at r = margin and CERTIFIED_WITH_MARGIN at "
        "r = nextafter(margin, 0); refusal of invalid margins.",
        "Two PLSR passes: base verdicts give margin and resolution, then each case is re-evaluated on its sorted "
        "margin grid; properties are checked along each sequence.",
        "T109: adversarial eigenvalue cases; extend the property test to affine plants with parameter-dependent P.",
        ["passing verdict regained at a larger margin", "meets_required_margin regained", "non-certifying code "
         "changed by the margin", "inequality_certified changed by the margin", "threshold not at margin",
         "invalid margin accepted"],
        ["A finite family cannot prove the property; the proof is the decision-order argument (analytic finding)."])
    family = near_threshold_family(108, 24) + razor_family(1081, 8)
    documented = []
    for member in family:
        info = R.documented_code(member["A"], member["P"], member["x"])
        steps = [dict(code=R.documented_code(member["A"], member["P"], member["x"], required_margin=r)["code"],
                      meets_required_margin=info["margin"] > max(r, info["resolution"]),
                      inequality_certified=info["margin"] > info["resolution"])
                 for r in margin_grid(info["margin"], info["resolution"])]
        documented.append(monotonicity_violations(steps))
    documented_total = {key: sum(v[key] for v in documented) for key in documented[0]}
    offline = [
        finding("Monotonicity of the verdict in the declared margin follows from the decision order", "mathematical",
                "margin enters only as MARGIN_LOW iff margin <= r and meets = margin > max(r, res)",
                {"derivation": "runtime.verdict at the pinned commit: required_margin is compared only after the "
                               "resolution test; both comparisons are monotone in r (docs/lab/LYAPUNOV.md, T108)"},
                tolerance=EXACT_TOL, uncertainty=ANALYTIC),
        finding("The documented rule re-derived in CIW is monotone on the same margin grids", "numerical",
                documented_total,
                {"generator": {"name": "near_threshold_family + razor_family", "seed": 108},
                 "checks": [_check("property violations in the re-derived rule", sum(documented_total.values()), 0.0,
                                   kind="invariant")]},
                tolerance={"abs": 0.0, "rel": 0.0}, uncertainty=EXACT)]
    base_cases = [_verdict_case(f"m{i}", m["A"], m["P"], m["x"]) for i, m in enumerate(family)]
    base_cases += [_verdict_case(f"invalid:{k}", family[0]["A"], family[0]["P"], family[0]["x"], required_margin=v)
                   for k, v in (("-1e-300", -1e-300), ("-1", -1.0), ("nan", math.nan), ("inf", math.inf))]
    try:
        first = _bridge(ctx, base_cases)
        grids = {i: margin_grid(first["results"][f"m{i}"]["margin"], first["results"][f"m{i}"]["resolution"])
                 for i in range(len(family))}
        second = _bridge(ctx, [_verdict_case(f"m{i}:{j}", m["A"], m["P"], m["x"], required_margin=r)
                               for i, m in enumerate(family) for j, r in enumerate(grids[i])])
    except _Unavailable as exc:
        fields["numerical_result"] = f"Provider-free re-derivation violations: {documented_total}."
        fields["uncertainty"] = "Deterministic re-derivation."
        return _finish(fields, offline, PROVIDER_FILES, blocked=str(exc))
    identity, results = second["identity"], second["results"]
    base = provider_basis(identity)
    totals = {"passing_regained": 0, "meets_regained": 0, "noncertifying_code_changed": 0, "inequality_changed": 0}
    threshold_mismatch, evaluations, sequences = 0, 0, {}
    for i in range(len(family)):
        sequence = [results[f"m{i}:{j}"] for j in range(len(grids[i]))]
        evaluations += len(sequence)
        for key, value in monotonicity_violations(sequence).items():
            totals[key] += value
        margin = first["results"][f"m{i}"]["margin"]
        if first["results"][f"m{i}"]["code"] == "CERTIFIED_WITH_MARGIN":
            by_r = dict(zip(grids[i], (s["code"] for s in sequence)))
            threshold_mismatch += by_r[math.nextafter(margin, 0.0)] != "CERTIFIED_WITH_MARGIN"
            threshold_mismatch += by_r[margin] != "MARGIN_LOW"
        sequences[str(i)] = [s["code"] for s in sequence]
    invalid = {k: _code(first["results"][f"invalid:{k}"]) for k in ("-1e-300", "-1", "nan", "inf")}
    ctx.artifact_json("margin-sequences.json", R.jsonable({"grids": grids, "codes": sequences, "totals": totals,
                                                            "invalid": invalid}))
    findings = [
        finding("Increasing required_margin never turns a failing PLSR verdict into a passing one", "numerical",
                dict(totals, cases=len(family), evaluations=evaluations),
                {"provider": base, "checks": [_check(f"violations: {key}", value, 0.0, kind="invariant")
                                              for key, value in totals.items()]},
                tolerance={"abs": 0.0, "rel": 0.0}, uncertainty=EXACT),
        finding("The switch from CERTIFIED_WITH_MARGIN to MARGIN_LOW happens exactly at required_margin = margin",
                "numerical", {"mismatches": threshold_mismatch},
                {"provider": base, "checks": [_check("codes at nextafter(margin, 0) and margin differing from "
                                                     "CERTIFIED_WITH_MARGIN / MARGIN_LOW", threshold_mismatch, 0.0,
                                                     kind="invariant")]},
                tolerance={"abs": 0.0, "rel": 0.0}, uncertainty=EXACT),
        finding("Negative and non-finite required margins are refused", "numerical", invalid,
                {"provider": base, "checks": [_refusal(f"required_margin = {k}", "raises ValueError", v)
                                              for k, v in invalid.items()]}, tolerance=EXACT_TOL, uncertainty=EXACT),
    ] + offline
    violations = sum(totals.values())
    fields["numerical_result"] = (
        f"{evaluations} verdicts over {len(family)} cases: violations {totals}; threshold mismatches "
        f"{threshold_mismatch}; invalid margins {invalid}."
        + (" No counterexample to monotonicity was found (a finite search, not a proof)."
           if violations == 0 and threshold_mismatch == 0 else
           f" {violations} monotonicity violations and {threshold_mismatch} threshold mismatches refute the "
           "property."))
    fields["uncertainty"] = "Exact comparisons on the runtime's own margins; no tolerance enters the property."
    return _finish(fields, findings, PROVIDER_FILES, identity)


# T109 ------------------------------------------------------------------------

# 2.82 and 2.83 bracket the P = I threshold 2 sqrt 2 = 2.8284 from both sides.
T109_K = (1.0, 2.82, 2.83, 10.0, 1e2, 1e3, 1e4, 1e5, 1e6, 1e8)
# solve_lyapunov refuses through exactly these documented gates (lyapunov.equation at the pinned commit).
SOLVER_GATES = ("solved P must be positive definite", "Lyapunov residual", "Lyapunov operator is singular")
T109_JORDAN = ((3, 0), (4, 6), (5, 12), (6, 10), (6, 12), (8, 8), (8, 10))


def adversarial_cases():
    """Non-normal triangular, exactly defective (Jordan) and clustered Hurwitz plants with known exact spectra."""
    rng = R.generator(109)
    cases = []
    for K in T109_K:
        cases.append({"name": f"non-normal K={K:g}", "group": "non-normal", "A": np.array([[-1.0, K], [0.0, -2.0]]),
                      "exact_spectrum": [-1.0, -2.0], "K": K})
    for n, e in T109_JORDAN:
        lam = 2.0 ** -e
        J = -lam * np.eye(n) + np.diag(np.ones(n - 1), 1)
        T = np.eye(n) + np.tril(rng.integers(-1, 2, size=(n, n)), -1).astype(float)
        Ti = np.round(np.linalg.inv(T))
        A = T @ J @ Ti
        # Integer unimodular similarity: A = T J T^-1 holds exactly, so the exact spectrum is {-lam} (defective).
        Tf, Jf, Tif = R.fractions(T), R.fractions(J), R.fractions(Ti)
        # Ti is the rounded float inverse; the exact spectrum needs T Ti = I exactly.
        assert all(sum(Tf[i][k] * Tif[k][j] for k in range(n)) == (1 if i == j else 0)
                   for i in range(n) for j in range(n))
        TJ = [[sum(Tf[i][k] * Jf[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
        exact = [[sum(TJ[i][k] * Tif[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
        assert all(Fraction(float(A[i, j])) == exact[i][j] for i in range(n) for j in range(n))
        cases.append({"name": f"Jordan n={n}, lambda=2^-{e}", "group": "Jordan", "A": A,
                      "exact_spectrum": [-lam] * n})
    for n, delta in ((3, 1e-8), (4, 1e-10), (5, 1e-12)):
        q = R.random_orthogonal(rng, n)
        A = q @ np.diag(-1.0 - delta * np.arange(n)) @ q.T + 1e-3 * np.triu(rng.normal(size=(n, n)), 1)
        cases.append({"name": f"clustered n={n}, spacing {delta:g}", "group": "clustered", "A": A,
                      "exact_spectrum": None})
    for case in cases:
        case["numpy_abscissa"] = float(np.max(np.linalg.eigvals(case["A"]).real))
    return cases


def transient_peak(K, samples=4001, horizon=20.0):
    """max_t ||exp(A t)||_2 for A = [[-1, K], [0, -2]], exp(At) = [[e^-t, K(e^-t - e^-2t)], [0, e^-2t]].

    The spectral norm of a 2x2 matrix is sqrt((F + sqrt(F^2 - 4 det^2)) / 2) with F the squared Frobenius norm.
    """
    t = np.linspace(0.0, horizon, samples)
    a, b = np.exp(-t), np.exp(-2.0 * t)
    c = K * (a - b)
    frobenius = a * a + b * b + c * c
    determinant = a * b
    return float(np.max(np.sqrt(0.5 * (frobenius + np.sqrt(np.maximum(frobenius ** 2 - 4.0 * determinant ** 2, 0.0))))))


def _valid_certificate(A, P, time="continuous"):
    """Exact positive definiteness of P and exact negative definiteness of the declared decrease form."""
    return R.positive_definite(R.fractions(P)) and R.exact_class(R.exact_form(A, P, time)) == "negative_definite"


@task("T109", changed_files=PROVIDER_FILES, regression_tests=(_node("test_t109_adversarial_eigenvalues"),
                                                                _node("test_numpy_misreads_exact_jordan_block"), PARTIAL_TEST))
def adversarial_eigenvalues(ctx):
    fields = _fields(
        "On highly non-normal, exactly defective and clustered Hurwitz plants PLSR either certifies soundly (the "
        "exact decrease form of its P is negative definite and P is exactly positive definite) or refuses; its "
        "Lyapunov solutions agree with an independent solver to within conditioning; its solver may refuse plants "
        "that do have a valid quadratic certificate; floating-point eigenvalues may misjudge stability where the "
        "exact certificate route does not.",
        "Lyapunov: A Hurwitz iff A^T P + P A = -Q has P > 0 for Q > 0; a certified P bounds transients by "
        "||exp(At)|| <= sqrt(cond P). Non-normal A = [[-1, K], [0, -2]]: P = I certifies iff K < 2 sqrt 2. "
        "Defective A = T J T^-1 with integer unimodular T (T Ti = I checked exactly) has exact spectrum {-lambda}; "
        "eigenvalue perturbation of an n-Jordan block is O(eps^(1/n)). A backward-stable solve has forward error "
        "of order n^2 u cond(P).",
        ["Non-normal K in {1, 2.82, 2.83, 10, ..., 1e8} (2.82 and 2.83 bracket 2 sqrt 2)",
         "Exactly defective A = T J T^-1 (PCG64 seed 109): (n, lambda) in "
         "{(3, 1), (4, 2^-6), (5, 2^-12), (6, 2^-10), (6, 2^-12), (8, 2^-8), (8, 2^-10)}",
         "Clustered spectra n = 3..5 with spacing 1e-8..1e-12 plus a 1e-3 non-normal part", "Q = I, x = ones"],
        "PLSR solve_lyapunov outcome and verdicts with P = I, with the independent P and with PLSR's own P; "
        "numpy eigenvalues; exact rational checks of every certifying verdict and of every candidate P.",
        "No certifying verdict without an exactly valid certificate; PLSR and independent P agree to within "
        "10 n^2 u cond(P); certified transients respect sqrt(cond P); every solver refusal comes from a documented "
        "gate.",
        "Phase 1: PLSR solves and verdicts with P = I and the independent P; phase 2: verdicts with PLSR's P. "
        "CIW checks certificates exactly, computes transient peaks in closed form and compares numpy's spectral "
        "abscissa with the exact spectrum.",
        "T110: discrete versus continuous interpretation; add an exact-arithmetic Lyapunov solve (rational "
        "Bartels-Stewart) to decide defective cases that PLSR's residual gate refuses, and propose that "
        "solve_lyapunov report the residual gate separately from definiteness.",
        ["certificate without exact validity", "solver disagreement beyond conditioning", "transient bound violated",
         "eigenvalue sign wrong for an exactly Hurwitz matrix", "P = I threshold misplaced",
         "solver refusal outside the documented gates", "solver refuses a plant with a valid certificate "
         "(searched as counterexample)"],
        ["Jordan and clustered plants are synthetic stress cases, not identified plant models.",
         "The independent solver is SciPy when installed, otherwise the CIW Kronecker solve."])
    cases = adversarial_cases()
    jordan = [c for c in cases if c["group"] == "Jordan"]
    misplacement = [abs(c["numpy_abscissa"] + c["exact_spectrum"][0]) / (float(np.finfo(float).eps)
                                                                         * float(np.max(np.abs(c["A"]))))
                    for c in jordan]
    jordan_wrong = [c for c in jordan if c["numpy_abscissa"] >= 0.0]
    numpy_checks = [_check("exactly defective cases whose numpy abscissa lies within 1e3 eps max|A| of the exact "
                           "eigenvalue -lambda (A = T J T^-1 verified exactly)",
                           sum(m <= 1e3 for m in misplacement), 0.0, kind="analytic")]
    numpy_extra = {}
    if jordan_wrong:
        numpy_checks.append(_check("exactly Hurwitz defective cases with numpy abscissa >= 0", len(jordan_wrong), 1.0,
                                   "ge", kind="analytic"))
        numpy_extra["counterexample"] = {
            "statement": "The sign of the floating-point spectral abscissa decides Hurwitz stability",
            "witness": {"name": jordan_wrong[0]["name"], "A": jordan_wrong[0]["A"].tolist(),
                        "exact_spectrum": jordan_wrong[0]["exact_spectrum"][0],
                        "numpy_abscissa": jordan_wrong[0]["numpy_abscissa"]}}
    offline = [finding(
        "numpy.linalg.eigvals misplaces the exact eigenvalue -lambda of every defective test matrix by far more "
        "than machine precision; whether the sign flips is recorded", "numerical",
        {"cases": len(jordan), "misplaced_beyond_1e3_eps": sum(m > 1e3 for m in misplacement),
         "wrong_sign": len(jordan_wrong)},
        {"generator": {"name": "adversarial_cases", "seed": 109}, "checks": numpy_checks},
        tolerance={"abs": 7.0, "rel": 0.0},
        uncertainty=_platform(0.0, "numpy eigenvalues of defective matrices vary at the eps^(1/n) level between "
                                   "builds; the misplacement exceeds 1e3 eps by orders of magnitude, the sign count "
                                   "may change"), **numpy_extra)]
    independent = {}
    for i, case in enumerate(cases):
        n = case["A"].shape[0]
        P, name, revision = R.independent_lyapunov(case["A"], np.eye(n))
        independent[i] = {"P": P, "implementation": name, "revision": revision}
    phase1 = []
    for i, case in enumerate(cases):
        n = case["A"].shape[0]
        phase1.append({"id": f"solve{i}", "op": "solve", "A": _mat(case["A"]), "time": "continuous"})
        phase1.append(_verdict_case(f"I{i}", case["A"], np.eye(n), np.ones(n)))
        P = independent[i]["P"]
        if np.all(np.isfinite(P)) and np.allclose(P, P.T):
            phase1.append(_verdict_case(f"ind{i}", case["A"], P, np.ones(n)))
    try:
        first = _bridge(ctx, phase1)
        solved = {i: np.array(first["results"][f"solve{i}"]["P"]) for i in range(len(cases))
                  if first["results"][f"solve{i}"]["ok"]}
        second = _bridge(ctx, [_verdict_case(f"own{i}", cases[i]["A"], P, np.ones(len(P))) for i, P in solved.items()])
    except _Unavailable as exc:
        fields["numerical_result"] = (f"Provider-free: numpy abscissa >= 0 for {len(jordan_wrong)} of {len(jordan)} "
                                      "exactly Hurwitz Jordan cases; every case misplaced by more than 1e3 eps.")
        fields["uncertainty"] = "Exact construction; numpy eigenvalues are platform-dependent in the last bits."
        return _finish(fields, offline, PROVIDER_FILES, blocked=str(exc))
    identity = second["identity"]
    results = dict(first["results"], **second["results"])
    base = provider_basis(identity)
    rows, violations, certified = [], 0, 0
    agreement, transient = [], []
    identity_threshold_mismatch = 0
    solver_invalid, ungated, refused_with_certificate = 0, 0, []
    for i, case in enumerate(cases):
        n = case["A"].shape[0]
        row = {"name": case["name"], "group": case["group"], "numpy_abscissa": case["numpy_abscissa"],
               "exact_spectrum_max": None if case["exact_spectrum"] is None else max(case["exact_spectrum"]),
               "solve": "P returned" if i in solved else _code(results[f"solve{i}"]),
               "solve_error": None if i in solved else results[f"solve{i}"]["error"]["message"],
               "code_P_identity": _code(results[f"I{i}"])}
        for key, P in (("independent", independent[i]["P"]), ("own", solved.get(i)), ("identity", np.eye(n))):
            cid = {"independent": f"ind{i}", "own": f"own{i}", "identity": f"I{i}"}[key]
            if P is None or cid not in results:
                continue
            code = _code(results[cid])
            row[f"code_{key}"] = code
            if code in R.CERTIFYING:
                certified += 1
                violations += not _valid_certificate(case["A"], P)
        if i in solved:
            solver_invalid += not _valid_certificate(case["A"], solved[i])
        else:
            ungated += not (row["solve"] == "raises ValueError" and row["solve_error"].startswith(SOLVER_GATES))
            if (row.get("code_independent") in R.CERTIFYING
                    and _valid_certificate(case["A"], independent[i]["P"])):
                eig = np.linalg.eigvalsh(independent[i]["P"])
                refused_with_certificate.append({"name": case["name"], "solver_error": row["solve_error"],
                                                 "certificate": independent[i]["implementation"],
                                                 "certificate_condition": float(eig[-1] / eig[0]),
                                                 "verdict_with_certificate": row["code_independent"]})
        if case["group"] == "non-normal":
            expected = "CERTIFIED_WITH_MARGIN" if case["K"] < 2.0 * math.sqrt(2.0) else "not certified"
            observed = row["code_P_identity"]
            identity_threshold_mismatch += ((observed == "CERTIFIED_WITH_MARGIN")
                                            != (expected == "CERTIFIED_WITH_MARGIN"))
            if i in solved and _code(results[f"own{i}"]) in R.CERTIFYING:
                eig = np.linalg.eigvalsh(solved[i])
                bound = math.sqrt(eig[-1] / eig[0])
                peak = transient_peak(case["K"])
                transient.append({"K": case["K"], "peak": peak, "sqrt_cond_P": bound, "ratio": peak / bound})
        if i in solved:
            P_ind = independent[i]["P"]
            relative = float(np.linalg.norm(solved[i] - P_ind) / np.linalg.norm(P_ind))
            eig = np.linalg.eigvalsh(solved[i])
            condition = float(eig[-1] / eig[0])
            normalised = relative / (n * n * R.U * max(condition, 1.0))
            row.update(relative_difference=relative, condition_P=condition, normalised_difference=normalised)
            agreement.append((condition, relative, normalised))
        rows.append(row)
    ctx.artifact_json("adversarial.json", R.jsonable({"rows": rows, "transient": transient,
                                                       "refused_with_certificate": refused_with_certificate,
                                                       "independent_solver": independent[0]["implementation"]}))
    ctx.artifact_text("solver-agreement.svg", svg.line_plot(
        [("PLSR vs independent P", [c for c, _, _ in agreement], [max(r, 1e-18) for _, r, _ in agreement]),
         ("n^2 u cond(P) for n = 2", [c for c, _, _ in agreement], [4 * R.U * max(c, 1.0) for c, _, _ in agreement])],
        title="T109 relative difference of Lyapunov solutions", xlabel="condition number of P",
        ylabel="||P_PLSR - P_ind|| / ||P_ind||", logx=True, logy=True))
    checker = {"implementation": independent[0]["implementation"], "revision": independent[0]["revision"]}
    max_relative = max(r for _, r, _ in agreement)
    max_normalised = max(m for _, _, m in agreement)
    jordan_rows = [r for r in rows if r["group"] == "Jordan"]
    jordan_solved = sum(r["solve"] == "P returned" for r in jordan_rows)
    witness_extra = {}
    if refused_with_certificate:
        witness_extra["counterexample"] = {
            "statement": "PLSR's solve_lyapunov refuses only plants for which no valid quadratic certificate is "
                         "available in float64", "witness": refused_with_certificate[0]}
    findings = [
        finding("Every certifying PLSR verdict on the adversarial plants uses an exactly valid certificate",
                "numerical",
                {"certifying_verdicts": certified, "violations": violations},
                {"provider": base, "independent_check": _independent(
                    _check("exact positive definiteness of P and negative definiteness of A^T P + P A", violations,
                           0.0), identity)},
                tolerance={"abs": 2.0, "rel": 0.0},
                uncertainty=_platform(0.0, "the violation count is exact; the number of certifying verdicts may move "
                                           "by one or two near the resolution")),
        finding("PLSR Lyapunov solutions agree with an independent solver to within 10 n^2 u cond(P) on every "
                "solved adversarial case", "numerical",
                {"cases": len(agreement), "max_relative_difference": max_relative,
                 "max_normalised_difference": max_normalised},
                {"provider": base, "independent_check": _independent(
                    _check("max over solved cases of the relative Frobenius difference divided by n^2 u cond(P)",
                           max_normalised, 10.0, "le"), identity, checker)},
                tolerance={"abs": 1e-12, "rel": 1.0},
                uncertainty=_roundoff(max_relative, "largest observed relative difference; the normalised value "
                                                    "may vary by a small factor between BLAS builds")),
        finding("Certified non-normal plants respect the Lyapunov transient bound ||exp(At)|| <= sqrt(cond P)",
                "numerical", {"cases": len(transient), "max_ratio": max(t["ratio"] for t in transient)},
                {"provider": base, "checks": [_check("max over K of peak ||exp(At)|| / sqrt(cond P)",
                                                     max(t["ratio"] for t in transient), 1.0, "le", kind="analytic")]},
                tolerance={"abs": 1e-9, "rel": 1e-6},
                uncertainty={"kind": "truncation_bound", "value": 1e-4,
                             "basis": "peak sampled every 5e-3 s; second-order estimate of the missed peak, far below "
                                      "the margin to 1"}),
        finding("With P = I the non-normal plants are certified exactly when K < 2 sqrt 2, including K = 2.82 and "
                "2.83 on either side", "numerical",
                {"mismatches": identity_threshold_mismatch,
                 "codes": {r["name"]: r["code_P_identity"] for r in rows if r["group"] == "non-normal"}},
                {"provider": base, "checks": [_check("codes against the analytic threshold",
                                                     identity_threshold_mismatch, 0.0, kind="analytic")]},
                tolerance=EXACT_TOL, uncertainty=EXACT),
        finding("PLSR's solve_lyapunov returns only exactly valid certificates on the exactly Hurwitz Jordan plants "
                "and otherwise raises ValueError at a documented gate", "numerical",
                {r["name"]: r["solve"] for r in jordan_rows},
                {"provider": base,
                 "independent_check": _independent(_check("returned P that are not exactly valid certificates",
                                                          solver_invalid, 0.0), identity),
                 "checks": [_check("refusals not raised as ValueError by the positive-definiteness, residual or "
                                   "singularity gate", ungated, 0.0, kind="invariant")]},
                tolerance=EXACT_TOL, uncertainty=EXACT),
        finding("solve_lyapunov refuses an exactly Hurwitz plant for which an exactly valid quadratic certificate "
                "exists and PLSR's own verdict certifies it", "numerical",
                {"refused_with_valid_certificate": len(refused_with_certificate),
                 "plants": [entry["name"] for entry in refused_with_certificate]},
                {"provider": base, "checks": [
                    _check("refused plants whose independent P is exactly valid and certified by PLSR's verdict",
                           len(refused_with_certificate), 1.0, "ge")]},
                tolerance=EXACT_TOL,
                uncertainty=_platform(0.0, "depends on the independent solver's P for a cond ~5e11 plant; exact "
                                           "validity of that P is decided in rational arithmetic"), **witness_extra),
    ] + offline
    fields["numerical_result"] = (
        f"{certified} certifying verdicts, {violations} without an exactly valid certificate. Solver agreement on "
        f"{len(agreement)} solved cases: max relative difference {max_relative:.2e}, max normalised by n^2 u cond(P) "
        f"{max_normalised:.3g} ({independent[0]['implementation']}). Transient peak / sqrt(cond P): max "
        f"{max(t['ratio'] for t in transient):.3f} over {len(transient)} certified K. P = I threshold mismatches: "
        f"{identity_threshold_mismatch}. Jordan: numpy abscissa >= 0 in {len(jordan_wrong)} of {len(jordan)} exactly "
        f"Hurwitz cases; PLSR's solver returned P for {jordan_solved} and refused {len(jordan_rows) - jordan_solved} "
        f"({ungated} refusals outside the documented gates, {solver_invalid} returned P invalid); "
        f"refused plants with an exactly valid independent certificate that PLSR's verdict certifies: "
        f"{len(refused_with_certificate)}. Conclusion: "
        + ("every PLSR certificate on these plants is exactly valid" if violations == 0 and solver_invalid == 0
           else "some PLSR certificates are not exactly valid")
        + ("; the solver's refusals are conservative, including plants that do have a valid certificate"
           if refused_with_certificate else "; no refused plant had a valid certificate from the independent solver")
        + ("; the floating-point eigenvalue sign misjudges exactly Hurwitz defective matrices." if jordan_wrong else
           "; the floating-point eigenvalue sign did not flip here, though the eigenvalues are misplaced."))
    fields["uncertainty"] = ("Exact checks are exact. Solver agreement scales with cond(P) (retained per case). "
                             "numpy eigenvalues of defective matrices vary at the eps^(1/n) level between builds, "
                             "so the count of wrong signs may change; the misplacement beyond 1e3 eps does not.")
    return _finish(fields, findings, PROVIDER_FILES, identity)


# T110 ------------------------------------------------------------------------

T110_DIAGONAL = {"diag(-0.5, -0.25)": (-0.5, -0.25), "diag(-1.5, -0.25)": (-1.5, -0.25),
                 "diag(0.5, 0.25)": (0.5, 0.25), "diag(1.5, 2)": (1.5, 2.0)}
QUADRANTS = {"continuous and discrete stable": (-0.9, -0.1), "continuous only": (-3.0, -1.1),
             "discrete only": (0.1, 0.9), "neither": (1.1, 3.0)}
NUMPY = {"implementation": f"numpy.linalg.eigvals@{np.__version__}", "revision": np.__version__}


def quadrant_family(per_quadrant=10):
    """Matrices V diag(lambda) V^-1 with every eigenvalue in one stability quadrant (margins >= 0.1)."""
    rng = R.generator(110)
    family = []
    for quadrant, (low, high) in QUADRANTS.items():
        for i in range(per_quadrant):
            n = 2 + i % 2
            V = R.random_orthogonal(rng, n) @ np.diag(rng.uniform(0.5, 2.0, n))
            A = V @ np.diag(rng.uniform(low, high, n)) @ np.linalg.inv(V)
            eigenvalues = np.linalg.eigvals(A)
            family.append({"quadrant": quadrant, "A": A, "x": rng.normal(size=n),
                           "stable": {"continuous": bool(np.max(eigenvalues.real) < 0.0),
                                      "discrete": bool(np.max(np.abs(eigenvalues)) < 1.0)}})
    return family


@task("T110", changed_files=PROVIDER_FILES, regression_tests=(_node("test_t110_time_interpretation"), PARTIAL_TEST))
def time_interpretation(ctx):
    fields = _fields(
        "PLSR distinguishes x' = A x from x+ = A x: the same matrix is certified in a time convention exactly when "
        "it is stable in that convention (Hurwitz versus Schur), and the two interpretations give different "
        "verdicts whenever the stability quadrants differ.",
        "Continuous decrease A^T P + P A, resolution 2 gamma n^2 a p + n^3 u |M|; discrete decrease A^T P A - P, "
        "resolution n(2 gamma n^2 a^2 p + u p) + n^3 u |M|. Lyapunov: a positive definite solution of the "
        "respective equation with Q = I exists iff A is Hurwitz, respectively Schur.",
        ["Diagonal plants diag(-0.5, -0.25), diag(-1.5, -0.25), diag(0.5, 0.25), diag(1.5, 2) with P = I, "
         "x = (1, 1)", "40 matrices V diag(lambda) V^-1 (PCG64 seed 110), 10 per quadrant: both stable, "
         "continuous only, discrete only, neither; eigenvalue margins >= 0.1",
         "A discrete affine plant evaluated with a theta_dot"],
        "PLSR codes with P = I; PLSR solve_lyapunov in each convention and the verdict with the returned P, also "
        "cross-applied to the other convention; numpy eigenvalues as the independent stability reference.",
        "Codes follow the exact class of each convention's decrease form; certified-by-own-P iff stable in that "
        "convention; theta_dot refused for discrete plants.",
        "Phase 1: P = I verdicts and solves in both conventions; phase 2: verdicts with each returned P in both "
        "conventions; compare with numpy's spectral abscissa and radius.",
        "T111: scalar quadratic versus matrix-eigenvalue routes; add a sampled-data check that a continuous "
        "certificate is not silently reused for exp(A h).",
        ["same verdict in both conventions for quadrant-differing matrices", "certified but unstable",
         "stable but never certified", "discrete theta_dot accepted", "resolution formula not time-specific"],
        ["Stability margins of at least 0.1 keep numpy's eigenvalue reference reliable for these matrices.",
         "No sampling period is modelled: the discrete matrices are declared one-step maps, as PLSR requires."])
    family = quadrant_family()
    kron_ok = 0
    for member in family:
        for time in ("continuous", "discrete"):
            P = R.kron_lyapunov(member["A"], np.eye(member["A"].shape[0]), time)
            kron_ok += (float(np.min(np.linalg.eigvalsh(P))) > 0.0) == member["stable"][time]
    offline = [finding("A CIW Kronecker Lyapunov solution is positive definite exactly when numpy calls the matrix "
                       "stable in that time convention", "numerical",
                       {"solves": 2 * len(family), "agreeing": kron_ok},
                       {"generator": {"name": "quadrant_family", "seed": 110},
                        "checks": [_check("solves whose definiteness disagrees with the eigenvalue classification",
                                          2 * len(family) - kron_ok, 0.0, kind="analytic")]},
                       tolerance={"abs": 0.0, "rel": 0.0},
                       uncertainty=_roundoff(0.0, "stability margins of at least 0.1 keep definiteness decisions far "
                                                  "from rounding"))]
    phase1 = []
    for name, diagonal in T110_DIAGONAL.items():
        for time in ("continuous", "discrete"):
            phase1.append(_verdict_case(f"{name}|{time}", np.diag(diagonal), np.eye(2), (1.0, 1.0), time=time))
            phase1.append({"id": f"{name}|{time}|res", "op": "resolution", "A": _mat(np.diag(diagonal)),
                           "P": _mat(np.eye(2)), "time": time})
    for i, member in enumerate(family):
        for time in ("continuous", "discrete"):
            phase1.append({"id": f"f{i}|{time}|solve", "op": "solve", "A": _mat(member["A"]), "time": time})
    phase1.append(_affine_case("discrete theta_dot", np.diag([0.5, 0.25]), [np.eye(2)], ([-0.1], [0.1]), np.eye(2),
                               (1.0, 1.0), [0.0], [0.0], time="discrete", rate_box=([-1.0], [1.0])))
    try:
        first = _bridge(ctx, phase1)
        phase2 = []
        for i, member in enumerate(family):
            for time in ("continuous", "discrete"):
                solved = first["results"][f"f{i}|{time}|solve"]
                if solved["ok"]:
                    for applied in ("continuous", "discrete"):
                        phase2.append(_verdict_case(f"f{i}|P_{time}|{applied}", member["A"], np.array(solved["P"]),
                                                    member["x"], time=applied))
        second = _bridge(ctx, phase2)
    except _Unavailable as exc:
        fields["numerical_result"] = f"Provider-free: {kron_ok}/{2 * len(family)} CIW solves agree with numpy."
        fields["uncertainty"] = "Eigenvalue margins >= 0.1."
        return _finish(fields, offline, PROVIDER_FILES, blocked=str(exc))
    identity = second["identity"]
    results = dict(first["results"], **second["results"])
    base = provider_basis(identity)
    diagonal_table, diagonal_mismatch, resolution_diff = {}, 0, 0.0
    for name, diagonal in T110_DIAGONAL.items():
        for time in ("continuous", "discrete"):
            A = np.diag(diagonal)
            code = _code(results[f"{name}|{time}"])
            diagonal_table[f"{name} {time}"] = code
            diagonal_mismatch += code != R.documented_code(A, np.eye(2), (1.0, 1.0), time)["code"]
            exact_class = R.exact_class(R.exact_form(A, np.eye(2), time))
            diagonal_mismatch += (code == "CERTIFIED_WITH_MARGIN") != (exact_class == "negative_definite")
            resolution_diff = max(resolution_diff, abs(results[f"{name}|{time}|res"]["value"]
                                                       - R.resolution(A, np.eye(2), time)))
    mismatches, pattern_mismatch, differing, quadrant_rows = 0, 0, 0, []
    cross_unsound, cross_total, solve_refusals, ungated = 0, 0, 0, 0
    for i, member in enumerate(family):
        outcome, errors = {}, {}
        for time in ("continuous", "discrete"):
            key = f"f{i}|P_{time}|{time}"
            outcome[time] = key in results and _code(results[key]) == "CERTIFIED_WITH_MARGIN"
            mismatches += outcome[time] != member["stable"][time]
            solved = results[f"f{i}|{time}|solve"]
            if not solved["ok"]:
                solve_refusals += 1
                errors[time] = solved["error"]
                ungated += not (solved["error"]["type"] == "ValueError"
                                and solved["error"]["message"].startswith(SOLVER_GATES))
        differs = outcome["continuous"] != outcome["discrete"]
        differing += differs
        # Per matrix: the outcomes differ exactly when the matrix is stable in one convention only.
        pattern_mismatch += differs != (member["stable"]["continuous"] != member["stable"]["discrete"])
        cross = {}
        for t in ("continuous", "discrete"):
            for a in ("continuous", "discrete"):
                key = f"f{i}|P_{t}|{a}"
                if key in results:
                    cross[f"P_{t}->{a}"] = _code(results[key])
                    if t != a:
                        cross_total += 1
                        # A P solved for one convention may certify in the other only where numpy finds stability.
                        cross_unsound += cross[f"P_{t}->{a}"] in R.CERTIFYING and not member["stable"][a]
        quadrant_rows.append({"quadrant": member["quadrant"], "stable": member["stable"], "certified": outcome,
                              "cross": cross, "solve_errors": errors})
    off_quadrant = sum(m["quadrant"] in ("continuous only", "discrete only") for m in family)
    theta_dot = _code(results["discrete theta_dot"])
    ctx.artifact_json("time-interpretation.json", R.jsonable({"diagonal": diagonal_table, "family": quadrant_rows}))
    findings = [
        finding("PLSR certifies each matrix with its own Lyapunov P exactly in the time convention where numpy "
                "finds it stable, and its solver refuses the other convention at a documented gate", "numerical",
                {"matrices": len(family), "mismatches": mismatches, "solver_refusals": solve_refusals},
                {"provider": base, "independent_check": _independent(
                    _check("certified-by-own-P against numpy spectral abscissa (continuous) and radius (discrete)",
                           mismatches, 0.0, kind="analytic"), identity, NUMPY),
                 "checks": [_check("solver refusals not raised as ValueError by a documented gate", ungated, 0.0,
                                   kind="invariant")]},
                tolerance=EXACT_TOL, uncertainty=_roundoff(0.0, "stability margins of at least 0.1")),
        finding("No Lyapunov P solved for one convention certifies a matrix in the other convention where numpy "
                "finds it unstable", "numerical", {"cross_verdicts": cross_total, "unsound": cross_unsound},
                {"provider": base, "independent_check": _independent(
                    _check("cross-applied certifying verdicts in a convention where numpy finds the matrix unstable",
                           cross_unsound, 0.0, kind="analytic"), identity, NUMPY)},
                tolerance=EXACT_TOL, uncertainty=_roundoff(0.0, "stability margins of at least 0.1")),
        finding("The two time interpretations give different PLSR outcomes exactly for the matrices whose stability "
                "differs between conventions", "numerical",
                {"differing": differing, "off_quadrant_matrices": off_quadrant, "pattern_mismatches": pattern_mismatch},
                {"provider": base, "checks": [_check("matrices where (outcomes differ) differs from (stable in one "
                                                     "convention only)", pattern_mismatch, 0.0, kind="invariant")]},
                tolerance=EXACT_TOL, uncertainty=EXACT),
        finding("Diagonal plants with P = I receive the code of each convention's exact decrease form", "numerical",
                diagonal_table,
                {"provider": base, "checks": [_check("codes differing from the documented order or exact class",
                                                     diagonal_mismatch, 0.0),
                                              _check("PLSR resolution minus CIW resolution (per convention)",
                                                     resolution_diff, 0.0, kind="analytic")]},
                tolerance=EXACT_TOL, uncertainty=EXACT),
        finding("A discrete plant refuses a theta_dot", "numerical", {"code": theta_dot},
                {"provider": base, "checks": [_refusal("discrete affine plant with theta_dot = 0",
                                                       "raises ValueError", theta_dot)]},
                tolerance=EXACT_TOL, uncertainty=EXACT),
    ] + offline
    fields["numerical_result"] = (
        f"Diagonal plants: {diagonal_table}. Family: {mismatches} disagreements with numpy over {len(family)} "
        f"matrices x 2 conventions ({solve_refusals} solver refusals, {ungated} outside the documented gates); "
        f"outcomes differ between conventions for {differing} matrices ({off_quadrant} lie in the continuous-only "
        f"or discrete-only quadrant; {pattern_mismatch} per-matrix mismatches). Cross-applied P: {cross_unsound} of "
        f"{cross_total} verdicts certify where numpy finds instability. Discrete theta_dot: {theta_dot}.")
    fields["uncertainty"] = "Margins of at least 0.1 in every quadrant; codes are far from the resolution."
    return _finish(fields, findings, PROVIDER_FILES, identity)


# T111 ------------------------------------------------------------------------

T111_SCALARS = (-1.0, -1e-10, -1e-300, -R.TINY, 0.0, R.TINY, 1e-300, 1.0)


def route_family():
    """Random plants for the route comparison; margins keep numpy eigenvalues decisive."""
    rng = R.generator(111)
    family = []
    while len(family) < 30:
        n = 2 + len(family) % 4
        A = rng.normal(size=(n, n)) - rng.uniform(0.0, 2.0) * np.eye(n)
        abscissa = float(np.max(np.linalg.eigvals(A).real))
        if abs(abscissa) > 0.05:
            family.append({"time": "continuous", "A": A, "stable": abscissa < 0.0, "x": rng.normal(size=n)})
    while len(family) < 50:
        n = 2 + len(family) % 3
        A = rng.normal(size=(n, n))
        A *= rng.uniform(0.3, 1.3) / float(np.max(np.abs(np.linalg.eigvals(A))))
        radius = float(np.max(np.abs(np.linalg.eigvals(A))))
        if abs(radius - 1.0) > 0.05:
            family.append({"time": "discrete", "A": A, "stable": radius < 1.0, "x": rng.normal(size=n)})
    return family


def unit_samples(count=64, n=2, seed=1111):
    rng = R.generator(seed)
    samples = rng.normal(size=(count, n))
    return samples / np.linalg.norm(samples, axis=1, keepdims=True)


@task("T111", changed_files=PROVIDER_FILES, regression_tests=(_node("test_t111_routes"), PARTIAL_TEST))
def quadratic_routes(ctx):
    fields = _fields(
        "The PLSR matrix route (Lyapunov solve, then the sign of max eig of the decrease form beyond the "
        "resolution) agrees with independent references -- numpy eigenvalues and SciPy's Bartels-Stewart "
        "Lyapunov solvers -- while the scalar quadratic route (the sign of x^T M x at sampled states) cannot "
        "certify definiteness and misses thin positive cones.",
        "Rayleigh: x^T M x <= max eig(M) |x|^2 for every x, with equality only on the top eigenvector, so sampled "
        "negativity never implies negative definiteness. An unstable A has v*(A + A^T)v = 2 Re(lambda)|v|^2 > 0 "
        "for an eigenvector v, so P = I can never certify it. For n = 1 the decrease form is 2 a p (continuous), "
        "so the verdict must follow sign(a) whenever 2|a|p exceeds the resolution.",
        ["30 continuous plants n = 2..5 (PCG64 seed 111) with |spectral abscissa| > 0.05",
         "20 discrete plants n = 2..4 with |spectral radius - 1| > 0.05", "Thin-cone plant A = diag(-0.5, 5e-7), "
         "P = I (decrease form diag(-1, 1e-6)) with 64 random unit states (seed 1111)",
         f"Scalar plants a in {list(T111_SCALARS)}, p = 1"],
        "PLSR solve_lyapunov outcome (P or refusal) and verdicts with the returned P and with P = I; SciPy (or the "
        "CIW Kronecker solve when SciPy is absent) Lyapunov P per time convention; numpy eigenvalues; PLSR "
        "verdicts at 64 sampled states.",
        "PLSR P equals the independent P to solver accuracy in each convention; solve_lyapunov returns P exactly "
        "for the numpy-stable plants and otherwise raises ValueError; the verdict certifies every numpy-stable "
        "plant with its own P and no numpy-unstable plant with P = I; sampled scalar decrease never contradicts "
        "the matrix route and cannot stand in for it.",
        "Phase 1: PLSR solves, P = I verdicts, scalar and thin-cone verdicts; phase 2: verdicts with PLSR's P. "
        "Compare routes case by case.",
        "T112: a separate disturbance-aware (ISS) research branch; T113: connect filtered residuals.",
        ["solver disagreement", "solver returns P for an unstable plant", "solver refuses a stable plant",
         "solver refusal other than ValueError", "verdict certifies an unstable plant", "scalar route claims "
         "definiteness", "scalar sign not followed for n = 1"],
        ["Plants with margins below 0.05 are excluded, so the comparison says nothing about near-marginal plants "
         "(T107 covers those).", "The independent solver is SciPy when installed, otherwise the CIW Kronecker "
         "solve (same Lyapunov equation, different code)."])
    family = route_family()
    independent = [R.independent_lyapunov(m["A"], np.eye(m["A"].shape[0]), m["time"]) for m in family]
    offline_agree = sum((float(np.min(np.linalg.eigvalsh(P))) > 0.0) == m["stable"]
                        for (P, _, _), m in zip(independent, family))
    solvers = sorted({(m["time"], name, revision) for (_, name, revision), m in zip(independent, family)})
    offline = [finding("The independent Lyapunov route (positive definite P) agrees with the numpy eigenvalue route "
                       "on every route-family plant", "numerical", {"plants": len(family), "agreeing": offline_agree},
                       {"generator": {"name": "route_family", "seed": 111},
                        "checks": [_check("plants where the two independent routes disagree",
                                          len(family) - offline_agree, 0.0, kind="analytic")]},
                       tolerance=EXACT_TOL, uncertainty=_roundoff(0.0, "stability margins of at least 0.05"))]
    samples = unit_samples()
    thin_A = np.diag([-0.5, 5e-7])
    phase1 = [{"id": f"solve{i}", "op": "solve", "A": _mat(m["A"]), "time": m["time"]} for i, m in enumerate(family)]
    phase1 += [_verdict_case(f"identity{i}", m["A"], np.eye(m["A"].shape[0]), m["x"], time=m["time"])
               for i, m in enumerate(family)]
    phase1 += [_verdict_case(f"thin{j}", thin_A, np.eye(2), x) for j, x in enumerate(samples)]
    phase1 += [_verdict_case(f"scalar{k}", [[a]], [[1.0]], [1.0]) for k, a in enumerate(T111_SCALARS)]
    try:
        first = _bridge(ctx, phase1)
        solved = {i: np.array(first["results"][f"solve{i}"]["P"]) for i in range(len(family))
                  if first["results"][f"solve{i}"]["ok"]}
        second = _bridge(ctx, [_verdict_case(f"own{i}", family[i]["A"], P, family[i]["x"], time=family[i]["time"])
                               for i, P in solved.items()])
    except _Unavailable as exc:
        fields["numerical_result"] = (f"Provider-free: {offline_agree}/{len(family)} plants agree between numpy and "
                                      f"{', '.join(name for _, name, _ in solvers)}.")
        fields["uncertainty"] = "Margins >= 0.05."
        return _finish(fields, offline, PROVIDER_FILES, blocked=str(exc))
    identity = second["identity"]
    results = dict(first["results"], **second["results"])
    base = provider_basis(identity)
    relative = {"continuous": [], "discrete": []}
    solve_mismatch, own_mismatch, identity_unsound, refusals, errors = 0, 0, 0, [], {}
    for i, member in enumerate(family):
        solve_mismatch += (i in solved) != member["stable"]
        if i in solved:
            if member["stable"]:
                own_mismatch += _code(results[f"own{i}"]) != "CERTIFIED_WITH_MARGIN"
            relative[member["time"]].append(float(np.linalg.norm(solved[i] - independent[i][0])
                                                  / np.linalg.norm(independent[i][0])))
        else:
            own_mismatch += member["stable"]
            error = results[f"solve{i}"]["error"]
            errors[str(i)] = error
            refusals.append(_refusal(f"solve_lyapunov for {member['time']} plant {i} (numpy: unstable)",
                                     "raises ValueError", _code(results[f"solve{i}"])))
        identity_unsound += not member["stable"] and _code(results[f"identity{i}"]) in R.CERTIFYING
    thin_codes = _counts(_code(first["results"][f"thin{j}"]) for j in range(len(samples)))
    thin_form = R.exact_form(thin_A, np.eye(2))
    thin_scalar_negative = sum(R.exact_quadratic(x, thin_form) < 0 for x in samples)
    # Exactly indefinite: the form and its negation each have a positive eigenvalue.
    thin_class = R.exact_class(thin_form)
    thin_indefinite = (thin_class == "has_positive_eigenvalue"
                       and R.exact_class(R.negate(thin_form)) == "has_positive_eigenvalue")
    scalar_codes = {f"{a:g}": _code(first["results"][f"scalar{k}"]) for k, a in enumerate(T111_SCALARS)}
    scalar_mismatch = sum((code == "CERTIFIED_WITH_MARGIN") != (a < 0.0)
                          for a, code in zip(T111_SCALARS, scalar_codes.values())
                          if abs(a) >= 2.0 ** -1022 or a == 0.0)
    identity_codes = {str(i): _code(results[f"identity{i}"]) for i in range(len(family))}
    ctx.artifact_json("routes.json", R.jsonable({"relative_difference": relative, "thin_cone_codes": thin_codes,
                                                  "scalar_codes": scalar_codes, "solve_errors": errors,
                                                  "identity_codes": identity_codes,
                                                  "independent_solvers": [list(entry) for entry in solvers]}))
    findings = []
    for time in ("continuous", "discrete"):
        name, revision = next((n, r) for t, n, r in solvers if t == time)
        values = relative[time]
        findings.append(finding(
            f"PLSR {time}-time Lyapunov solutions agree with the independent solver on the route family",
            "numerical", {"solves": len(values), "max_relative_difference": max(values)},
            {"provider": base, "independent_check": _independent(
                _check("relative Frobenius difference from the independent solution", max(values), 1e-9),
                identity, {"implementation": name, "revision": revision})},
            tolerance={"abs": 1e-9, "rel": 0.0},
            uncertainty=_roundoff(max(values), "largest observed relative difference")))
    findings += [
        finding("PLSR's solve_lyapunov returns a P exactly for the plants numpy's eigenvalues call stable and raises "
                "ValueError for the others", "numerical",
                {"plants": len(family), "mismatches": solve_mismatch, "refused": len(refusals)},
                {"provider": base, "independent_check": _independent(
                    _check("plants where returning P and numpy stability disagree", solve_mismatch, 0.0,
                           kind="analytic"), identity, NUMPY),
                 "checks": refusals},
                tolerance=EXACT_TOL, uncertainty=_roundoff(0.0, "stability margins of at least 0.05")),
        finding("PLSR's verdict certifies every numpy-stable plant with its own P and no numpy-unstable plant with "
                "P = I", "numerical",
                {"stable_not_certified": own_mismatch, "unstable_certified_with_identity": identity_unsound},
                {"provider": base, "independent_check": _independent(
                    _check("stable plants not certified with their own P plus unstable plants certified with P = I",
                           own_mismatch + identity_unsound, 0.0, kind="analytic"), identity, NUMPY)},
                tolerance=EXACT_TOL, uncertainty=_roundoff(0.0, "stability margins of at least 0.05")),
        finding("The scalar route sees decrease at every sampled state of an indefinite form that PLSR reports "
                "DECREASE_NOT_DEFINITE", "numerical",
                {"samples": len(samples), "scalar_negative": int(thin_scalar_negative), "plsr_codes": thin_codes,
                 "exact_class": thin_class, "exactly_indefinite": thin_indefinite},
                {"provider": base, "checks": [
                    _check("1 if the exact decrease form diag(-1, 1e-6) is not indefinite, else 0",
                           0.0 if thin_indefinite else 1.0, 0.0, kind="exact_arithmetic"),
                    _check("samples with positive scalar decrease", len(samples) - thin_scalar_negative, 0.0,
                           kind="exact_arithmetic"),
                    _check("samples PLSR did not report DECREASE_NOT_DEFINITE",
                           len(samples) - thin_codes.get("DECREASE_NOT_DEFINITE", 0), 0.0, kind="invariant"),
                    _check("samples PLSR certified", sum(v for k, v in thin_codes.items() if k in R.CERTIFYING), 0.0,
                           kind="invariant")]},
                tolerance=EXACT_TOL,
                counterexample={"statement": "A negative sampled scalar decrease at every tested state implies a "
                                             "negative definite decrease form",
                                "witness": {"A": "diag(-0.5, 5e-7)", "P": "I", "M": "diag(-1, 1e-6)",
                                            "samples": len(samples), "seed": 1111}}, uncertainty=EXACT),
        finding("For n = 1 the PLSR verdict follows the sign of a throughout the normal range", "numerical",
                scalar_codes,
                {"provider": base, "checks": [_check("normal-range scalar plants whose certification differs from "
                                                     "a < 0", scalar_mismatch, 0.0, kind="analytic")]},
                tolerance=EXACT_TOL, uncertainty=EXACT),
    ] + offline
    routes_agree = solve_mismatch == 0 and own_mismatch == 0 and identity_unsound == 0
    fields["numerical_result"] = (
        f"Solver agreement: max relative difference {max(relative['continuous']):.2e} over "
        f"{len(relative['continuous'])} continuous and {max(relative['discrete']):.2e} over "
        f"{len(relative['discrete'])} discrete PLSR solves ({'; '.join(name for _, name, _ in solvers)}). "
        f"solve_lyapunov: {solve_mismatch} disagreements with numpy stability over {len(family)} plants, "
        f"{len(refusals)} refusals ({sum(not family[int(i)]['stable'] for i in errors)} of them numpy-unstable "
        f"plants). Verdicts: {own_mismatch} stable plants not certified "
        f"with their own P, {identity_unsound} unstable plants certified with P = I. Thin cone: "
        f"{thin_scalar_negative}/{len(samples)} sampled scalar decreases negative; PLSR codes {thin_codes}. Scalar "
        f"plants: {scalar_codes}. Conclusion: "
        + ("the matrix routes agree with each other on margin-separated plants" if routes_agree else
           "the matrix routes disagree on some margin-separated plants")
        + ("; sampled scalar decrease cannot stand in for them." if thin_scalar_negative == len(samples)
           and thin_indefinite and thin_codes.get("DECREASE_NOT_DEFINITE", 0) == len(samples)
           else "; the thin-cone probe did not separate the routes."))
    fields["uncertainty"] = ("Solver differences are at rounding level; route agreement holds only with the stated "
                             "stability margins. The subnormal scalar entries are reported, not asserted.")
    return _finish(fields, findings, PROVIDER_FILES, identity)


# T112 ------------------------------------------------------------------------

ISS_SCENARIOS = ("constant", "worst-case switching", "resonant sinusoid", "random held")


def _exponential_finding(claim, series, closed_form, independent, reference):
    """Series exponential against a closed form (always) and an outside implementation (when one applies)."""
    closed = float(np.max(np.abs(series - closed_form)))
    basis = {"checks": [_check(f"max abs entry difference from the closed form of {reference}", closed, 1e-12,
                               kind="analytic")]}
    value = {"max_abs_difference_closed_form": closed, "max_abs_difference_independent": None}
    if independent is not None:
        matrix, implementation, revision = independent
        difference = float(np.max(np.abs(series - matrix)))
        value["max_abs_difference_independent"] = difference
        basis["independent_check"] = dict(
            _check(f"max abs entry difference of {reference}", difference, 1e-12, kind="analytic"),
            producer={"implementation": "ciw.lab.lyapunov_research.expm_series", "revision": __version__},
            checker={"implementation": implementation, "revision": revision})
    return finding(claim, "numerical", value, basis, tolerance={"abs": 1e-12, "rel": 0.0},
                   uncertainty=_roundoff(max(v for v in value.values() if v is not None),
                                         "observed maximum entry difference"))


@task("T112", changed_files=RESEARCH_FILES, regression_tests=(_node("test_t112_iss_branch"),
                                                                _node("test_research_tasks_without_optional_modules")))
def iss_branch(ctx):
    fields = _fields(
        "For x' = A x + B w with |w| <= w_bar, the quadratic ISS-Lyapunov bound sqrt(V(t)) <= max(sqrt(V(0)), "
        "2 ||P^(1/2) B|| w_bar / c) holds on every bounded disturbance; it is conservative against the sharp "
        "reachable-set supremum in two dimensions and approached in one dimension; this belongs to a separate "
        "research branch, not to the PLSR runtime.",
        "V = x^T P x, A^T P + P A = -Q: V' <= -c V + 2 sqrt(V) beta with c = min eig(P^-1 Q), beta = "
        "||P^(1/2) B|| w_bar (Cauchy-Schwarz in the P inner product), so W = sqrt(V) obeys W' <= -(c/2) W + beta. "
        "The sharp supremum from x(0) = 0 is max over unit u of w_bar int_0^inf |u^T L^T e^(A s) B| ds with "
        "P = L L^T (support function of the reachable set).",
        ["A = [[0, 1], [-4, -1.2]], B = [0, 1]^T, w_bar = 0.5, Q = I (synthetic)",
         "Disturbances: constant w_bar; worst-case switching w_bar sign(B^T P x); resonant sinusoid at 2 rad/s; "
         "random levels held for 0.2 s (PCG64 seed 112)", "Scalar x' = -x + w, w = w_bar",
         "Exact ZOH simulation, h = 0.005 s, 30 s from x(0) = 0; reachable-set integral by the trapezoid rule on "
         "the exact sampled impulse response (h and h/2, 60 s, 361-direction grid refined by 60 golden-section "
         "steps)"],
        "sup_t sqrt(V(x(t))) and sup_t |x(t)| of the synthetic simulations against the analytic bound and the "
        "sharp reachable-set supremum.",
        "Every simulated sup sqrt(V) is at most the sharp supremum, which is at most the ISS bound; the "
        "worst-case switching disturbance comes within 2 % of the sharp supremum; the scalar sup approaches the "
        "bound to 1e-9 at 30 s.",
        "Derive the bound, compute the sharp reachable-set supremum, simulate four disturbance classes with the "
        "exact discretisation, compare, check the matrix exponential against its closed form and an independent "
        "implementation, and retain the written branch specification.",
        "Specify a resolution-aware float64 ISS inequality (open question 1 in the retained spec) before any "
        "runtime integration; acquiring an evidenced disturbance bound is hardware-gated.",
        ["simulated trajectory exceeds the sharp supremum", "worst-case switching far below the sharp supremum",
         "sharp supremum not converged in the step", "bound not approached in one dimension",
         "matrix exponential inaccurate", "ISS claims leaking into runtime statuses"],
        ["w_bar and B are declared synthetic values; no physical disturbance was measured.",
         "Sample-held disturbances are one admissible class; inter-sample peaks are not sampled.",
         "Only four disturbance classes were simulated; the sharp supremum, not the simulations, covers every "
         "bounded disturbance."])
    bound = X.iss_bound(X.ISS_A, X.ISS_B, np.eye(2), X.ISS_W)
    simulations = {name: X.simulate_iss(X.ISS_A, X.ISS_B, bound["P"], X.ISS_W, name) for name in ISS_SCENARIOS}
    ratios = {name: result["sup_sqrt_V"] / bound["sqrt_V_bound"] for name, result in simulations.items()}
    reachable = X.reachable_sup(X.ISS_A, X.ISS_B, bound["P"], X.ISS_W)
    refined = X.reachable_sup(X.ISS_A, X.ISS_B, bound["P"], X.ISS_W, step=X.ISS_STEP / 2.0)
    sharp = reachable["sup_sqrt_V"]
    convergence = abs(refined["sup_sqrt_V"] - sharp) / sharp
    to_sharp = {name: result["sup_sqrt_V"] / sharp for name, result in simulations.items()}
    scalar = X.scalar_iss()
    augmented = np.zeros((3, 3))
    augmented[:2, :2], augmented[:2, 2:] = X.ISS_A, X.ISS_B
    series = X.expm_series(augmented * X.ISS_STEP)
    closed = X.damped_oscillator_zoh(X.ISS_A, X.ISS_B, X.ISS_STEP)
    independent = X.independent_expm(augmented * X.ISS_STEP)
    ctx.artifact_text("iss-branch-spec.md", X.iss_spec_markdown(bound, simulations, scalar, reachable))
    ctx.artifact_json("iss-branch-spec.json", R.jsonable(dict(X.ISS_SPEC, bound={k: v for k, v in bound.items()},
                                                             simulations=simulations, scalar=scalar,
                                                             reachable=reachable)))
    ctx.artifact_text("iss-ratios.svg", svg.line_plot(
        [("sup sqrt(V) / ISS bound", list(range(len(ISS_SCENARIOS))), [ratios[n] for n in ISS_SCENARIOS]),
         ("sharp supremum / ISS bound", [0, len(ISS_SCENARIOS) - 1], [sharp / bound["sqrt_V_bound"]] * 2),
         ("ISS bound", [0, len(ISS_SCENARIOS) - 1], [1.0, 1.0])],
        title="T112 simulated sup sqrt(V) against the sharp supremum and the ISS bound",
        xlabel="scenario index (see iss-branch-spec.md)", ylabel="ratio to the ISS bound"))
    findings = [
        finding("For the four simulated disturbance classes sup sqrt(V) stays below the sharp reachable-set "
                "supremum, and worst-case switching comes within 2 % of it", "numerical",
                {"to_sharp_supremum": {name: to_sharp[name] for name in ISS_SCENARIOS},
                 "to_iss_bound": {name: ratios[name] for name in ISS_SCENARIOS}},
                {"generator": {"name": "simulate_iss", "seed": 112},
                 "checks": [_check("max over scenarios of sup sqrt(V) / sharp supremum", max(to_sharp.values()),
                                   1.0 + 1e-5, "le", kind="analytic"),
                            _check("worst-case switching sup sqrt(V) / sharp supremum",
                                   to_sharp["worst-case switching"], 0.98, "ge", kind="analytic")]},
                tolerance={"abs": 1e-9, "rel": 1e-6},
                uncertainty=_roundoff(1e-5, "trapezoid error of the sharp supremum (h versus h/2 agree to "
                                            f"{convergence:.1e} relative); the simulation is exact ZOH")),
        finding("The sharp reachable-set supremum of sqrt(V) is converged in the quadrature step and lies below the "
                "quadratic ISS bound", "numerical",
                {"sharp_supremum": sharp, "to_iss_bound": sharp / bound["sqrt_V_bound"],
                 "step_refinement_relative_change": convergence},
                {"generator": {"name": "reachable_sup", "seed": None},
                 "checks": [_check("relative change of the supremum when the step is halved", convergence, 1e-5,
                                   "le", kind="self_convergence"),
                            _check("sharp supremum / ISS bound", sharp / bound["sqrt_V_bound"], 1.0, "le",
                                   kind="analytic")]},
                tolerance={"abs": 1e-8, "rel": 1e-6},
                uncertainty=_roundoff(convergence, "relative change under step halving")),
        finding("Quadratic ISS-Lyapunov bound for the synthetic oscillator", "mathematical",
                {"c": bound["c"], "beta": bound["beta"], "sqrt_V_bound": bound["sqrt_V_bound"],
                 "state_bound": bound["state_bound"]},
                {"derivation": "W = sqrt(V), W' <= -(c/2) W + beta with c = min eig(P^-1 Q) and beta = ||P^(1/2) B|| "
                               "w_bar; lyapunov_research.iss_bound and docs/lab/LYAPUNOV.md (T112)"},
                tolerance={"abs": 0.0, "rel": 1e-9},
                uncertainty=_roundoff(1e-15, "float64 evaluation of the closed form")),
        finding("In one dimension the quadratic ISS bound w_bar / a is approached: sup |x| over 30 s is (1 - e^-30) "
                "of it", "numerical", scalar,
                {"checks": [_check("1 - exact sup / bound for x' = -x + 0.5 over 30 s", 1.0 - scalar["ratio"], 1e-9,
                                   kind="analytic")]},
                tolerance={"abs": 1e-12, "rel": 1e-9},
                uncertainty=_roundoff(1e-13, "float64 evaluation of the exponential and the closed form")),
        _exponential_finding("The series matrix exponential behind the exact ZOH simulation agrees with its closed "
                             "form and, where available, an independent exponential", series, closed, independent,
                             "exp([[A, B], [0, 0]] h)"),
        finding("The ISS branch adds no runtime-status-v1 code, sample field or verdict to PLSR",
                "computational_pipeline", "research branch only",
                {"derivation": "retained iss-branch-spec.md: 'plsr_must_not_claim'; the runtime vocabulary is fixed "
                               "by runtime-status-v1 at the pinned commit (T106 constants)"},
                tolerance=EXACT_TOL, uncertainty=ANALYTIC),
        finding("The disturbance bound w_bar = 0.5 holds for a physical plant", "physical", None, {}),
        finding("The ISS bound defines a safe operating envelope for a machine", "machine_safety", None,
                {"derivation": "an analytic bound on a declared model; safety requires a safety case outside the "
                               "workbench"}),
    ]
    exponential = ("its closed form" + (f" and {independent[1]}" if independent is not None else "")
                   + f" to {max(v for v in findings[4]['value'].values() if v is not None):.2e}")
    fields["numerical_result"] = (
        f"ISS bound sqrt(V) <= {bound['sqrt_V_bound']:.6g} (|x| <= {bound['state_bound']:.6g}); sharp reachable-set "
        f"supremum {sharp:.6g} ({sharp / bound['sqrt_V_bound']:.4f} of the bound, step-halving change "
        f"{convergence:.1e}); simulated sup sqrt(V) / sharp supremum "
        f"{ {k: round(v, 4) for k, v in to_sharp.items()} }; scalar sup/bound = {scalar['ratio']:.14f} "
        f"(1 - e^-30); series expm agrees with {exponential}.")
    fields["uncertainty"] = ("Simulation is exact ZOH (exponential accurate to 1e-12); the sharp supremum carries a "
                             "trapezoid error below 1e-5 relative; the 2-D ISS bound is conservative by about a "
                             "factor four because of the Cauchy-Schwarz step.")
    return _finish(fields, findings, RESEARCH_FILES, provider=False)


# T113 ------------------------------------------------------------------------

T113_FILES = (MODULE, REFERENCE, BRIDGE, RESEARCH, DOC)


@task("T113", changed_files=T113_FILES,
      regression_tests=(_node("test_t113_residual_adapter"), _node("test_adapter_keeps_metadata_outside"),
                        PARTIAL_TEST))
def residual_adapter(ctx):
    fields = _fields(
        "A host-side adapter can turn filtered residual statistics into plsr-sample-v1 samples (a schema tag and "
        "numbers only) while sensor identity, units, calibration and timing stay in a host envelope; host-owned "
        "statuses are decided before the kernel and never reach it; forwarding theta_hat together with both ends "
        "of its three-standard-error interval keeps a near-bound point estimate from being accepted alone; and "
        "the kernel's codes on forwarded samples follow the declared model.",
        "Declared discrete map A(theta) = I + h [[0, 1], [-(4 + theta), -0.4]], h = 0.01 s, theta in [-0.5, 0.5], "
        "common P from the nominal discrete Lyapunov equation (exactly valid at both vertices, hence on the box by "
        "convexity). An EKF on (x, v, theta) yields x_hat, theta_hat, its standard error se and the innovation "
        "NIS; mean NIS above 1 + 6 sqrt(2/N) is MODEL_MISMATCH. The adapter forwards theta_hat and theta_hat +- "
        "3 se; the host accepts a window only if the kernel certifies all three.",
        ["Eight synthetic 4 s windows (PCG64 seeds 1131-1138): theta 0.1 and -0.3 (nominal), 0.48 (estimate near "
         "the bound), 0.9 (outside the box), damping 3 instead of 0.4 (structural mismatch), a stale window, a "
         "dropout (NaN), an expired certificate",
         "Envelope metadata: sensor id, units, calibration reference and validity, filter, site"],
        "Adapter outputs (samples or host status), serialised kernel payloads, PLSR codes for forwarded samples, "
        "the host's window decision, and PLSR's treatment of host-owned codes.",
        "Payloads carry only plant, certificate and numeric sample fields; host statuses are host-owned codes; "
        "forwarded codes equal the CIW transcription of the documented order; no window whose guard interval "
        "leaves the box is accepted; theta estimates lie within 3 standard errors of the synthetic truth.",
        "Run the adapter on every window, serialise the kernel payloads and search them for envelope metadata, "
        "evaluate every forwarded sample with PLSR, apply the host's window rule, and compare with the "
        "documented decision order.",
        "T114: servo-axis pilot specification; then bind the adapter to acquired encoder data (hardware-gated) and "
        "check the coverage of the three-standard-error interval there.",
        ["metadata in the kernel payload", "non-numeric or extra sample field", "host status forwarded to the "
         "kernel", "kernel accepts a host-owned code", "near-bound point estimate accepted although its interval "
         "leaves the box", "theta estimate inconsistent with its standard error"],
        ["All data are synthetic; the EKF, noise levels and thresholds are illustrative, not tuned to a sensor.",
         "The guard assumes the EKF standard error of theta is calibrated; its coverage for a real sensor is not "
         "established (sensor_performance finding).",
         "Staleness and certificate validity use declared window times, not a real clock."])
    windows = X.adapter_windows()
    P, vertex_classes = X.adapter_certificate()
    A0, A1 = X.adapter_plant()
    box = ([X.ADAPTER_BOX[0]], [X.ADAPTER_BOX[1]])
    rows, kernel_cases, leaks, bad_fields, estimate_misses, guard_errors = [], [], [], 0, 0, 0
    for i, window in enumerate(windows):
        outcome = X.adapt(window["envelope"])
        row = {"window": window["name"], "host_status": outcome.get("host_status"),
               "statistics": {k: v for k, v in outcome.get("statistics", {}).items() if k != "x"}}
        if "samples" in outcome:
            stats = outcome["statistics"]
            spread = X.GUARD_SE * stats["theta_se"]
            expected = {"estimate": stats["theta"], "lower": stats["theta"] - spread, "upper": stats["theta"] + spread}
            for role in X.SAMPLE_ROLES:
                sample = outcome["samples"][role]
                bad_fields += tuple(sample) != X.SAMPLE_FIELDS or not all(
                    math.isfinite(v) for v in sample["x"] + sample["theta"])
                guard_errors += sample["theta"] != [expected[role]]
                case = _affine_case(f"w{i}:{role}", A0, [A1], box, P, sample["x"], sample["theta"], time="discrete")
                kernel_cases.append(case)
                leaks += X.metadata_leaks(case, window["envelope"])
            estimate_misses += abs(stats["theta"] - window["theta_true"]) > 3.0 * stats["theta_se"]
            row.update(forwarded=True, guard_interval=[expected["lower"], expected["upper"]],
                       interval_inside_box=X.ADAPTER_BOX[0] <= expected["lower"]
                       and expected["upper"] <= X.ADAPTER_BOX[1],
                       estimate_inside_box=X.ADAPTER_BOX[0] <= stats["theta"] <= X.ADAPTER_BOX[1])
        rows.append(row)
    host = {r["window"]: r["host_status"] for r in rows if r["host_status"]}
    forwarded = [r for r in rows if r.get("forwarded")]
    near_bound = [r["window"] for r in forwarded if r["estimate_inside_box"] and not r["interval_inside_box"]]
    # Positive control: a payload that does carry envelope metadata must be flagged by the same detector.
    planted = dict(kernel_cases[0], sensor={"id": windows[0]["envelope"]["sensor_id"]})
    control = X.metadata_leaks(planted, windows[0]["envelope"])
    largest_se = max(r["statistics"]["theta_se"] for r in forwarded)
    offline = [
        finding("Kernel payloads built by the adapter carry only the declared model and numeric sample fields, and "
                "no envelope metadata",
                "computational_pipeline", {"payloads": len(kernel_cases), "metadata_leaks": leaks,
                                           "malformed_samples": bad_fields},
                {"checks": [_check("envelope keys or strings found in serialised payloads", len(leaks), 0.0,
                                   kind="invariant"),
                            _check("positive control: leaks flagged in a payload with a planted sensor id",
                                   len(control), 1.0, "ge", kind="invariant"),
                            _check("samples with extra, missing or non-finite fields", bad_fields, 0.0,
                                   kind="invariant")]},
                tolerance=EXACT_TOL, uncertainty=EXACT),
        finding("The adapter forwards theta_hat with both ends of its three-standard-error interval, and a near-bound "
                "estimate yields an interval end outside the box", "computational_pipeline",
                {"forwarded_windows": len(forwarded), "guard_errors": guard_errors, "near_bound_windows": near_bound},
                {"checks": [_check("forwarded samples whose theta differs from theta_hat, theta_hat - 3 se or "
                                   "theta_hat + 3 se", guard_errors, 0.0, kind="invariant"),
                            _check("positive control: forwarded windows with theta_hat inside the box and an interval "
                                   "end outside", len(near_bound), 1.0, "ge", kind="invariant")]},
                tolerance=EXACT_TOL, uncertainty=EXACT),
        finding("Host statuses are decided in the adapter, are host-owned codes and are never forwarded", "numerical",
                host, {"checks": [_check("adapter statuses outside the host-owned list",
                                         sum(v not in R.HOST_OWNED for v in host.values()), 0.0, kind="invariant"),
                                  _check("windows both forwarded and given a host status",
                                         sum(1 for r in rows if r.get("forwarded") and r["host_status"]), 0.0,
                                         kind="invariant")]}, tolerance=EXACT_TOL, uncertainty=EXACT),
        finding("EKF theta estimates of forwarded windows lie within three standard errors of the synthetic truth",
                "numerical", {"forwarded": len(forwarded), "misses": estimate_misses},
                {"generator": {"name": "adapter_windows", "seed": 1131},
                 "checks": [_check("forwarded windows with |theta_hat - theta| > 3 se", estimate_misses, 0.0,
                                   kind="invariant")]},
                tolerance=EXACT_TOL,
                uncertainty={"kind": "reference_error", "value": largest_se,
                             "basis": "largest EKF standard error of theta among forwarded windows"}),
        finding("The adapter's common P is exactly valid at both box vertices of the declared discrete map",
                "numerical", {"vertex_classes": vertex_classes},
                {"checks": [_check("vertices without an exactly negative definite decrease",
                                   sum(c != "negative_definite" for c in vertex_classes), 0.0)]},
                tolerance=EXACT_TOL, uncertainty=EXACT),
        finding("The synthetic residual statistics describe a real encoder's performance", "sensor_performance",
                None, {}),
        finding("The EKF standard error of theta covers the true parameter of a real axis at the stated rate",
                "sensor_performance", None, {}),
        finding("The calibration referenced in the host envelope is valid", "calibration", None, {}),
    ]
    predicted = {case["id"]: R.documented_code(A0 + case["theta"][0] * A1, P, case["x"], "discrete",
                                               in_box=X.ADAPTER_BOX[0] <= case["theta"][0] <= X.ADAPTER_BOX[1])["code"]
                 for case in kernel_cases}
    bridge_cases = kernel_cases + [{"id": f"require:{code}", "op": "require_status", "code": code}
                                   for code in sorted(set(host.values()))]
    ctx.artifact_json("adapter-windows.json", R.jsonable({"rows": rows, "payload_example": kernel_cases[0],
                                                           "sample_fields": X.SAMPLE_FIELDS,
                                                           "sample_roles": X.SAMPLE_ROLES}))
    try:
        bridge = _bridge(ctx, bridge_cases)
    except _Unavailable as exc:
        fields["numerical_result"] = (f"Provider-free: {len(forwarded)} windows forwarded with "
                                      f"{len(kernel_cases)} samples, host statuses {host}, {len(leaks)} metadata "
                                      f"leaks, {guard_errors} guard errors, near-bound windows {near_bound}, "
                                      f"{estimate_misses} estimate misses.")
        fields["uncertainty"] = "Synthetic data; deterministic seeds."
        return _finish(fields, offline, T113_FILES, blocked=str(exc))
    identity, results = bridge["identity"], bridge["results"]
    base = provider_basis(identity)
    names = {i: w["name"] for i, w in enumerate(windows)}
    kernel_codes = {}
    for cid in predicted:
        index, role = cid[1:].split(":")
        kernel_codes.setdefault(names[int(index)], {})[role] = _code(results[cid])
    mismatches = sum(_code(results[cid]) != code for cid, code in predicted.items())
    accepted = {name: all(code == "CERTIFIED_WITH_MARGIN" for code in codes.values())
                for name, codes in kernel_codes.items()}
    interval_inside = {r["window"]: r["interval_inside_box"] for r in forwarded}
    unsafe_accept = sum(accepted[name] and not interval_inside[name] for name in accepted)
    near_bound_accepted = sum(accepted[name] for name in near_bound)
    refused = {code: _code(results[f"require:{code}"]) for code in sorted(set(host.values()))}
    findings = [
        finding("Forwarded samples receive the runtime codes predicted from the declared model", "numerical",
                kernel_codes,
                {"provider": base, "checks": [
                    _check("forwarded samples whose code differs from the CIW transcription of the documented order "
                           "(same specification)", mismatches, 0.0, kind="analytic")]},
                tolerance=EXACT_TOL, uncertainty=EXACT),
        finding("The host accepts a window only when the kernel certifies theta_hat and both interval ends, so the "
                "near-bound estimate is rejected", "numerical",
                {"accepted": accepted, "near_bound_accepted": near_bound_accepted},
                {"provider": base, "checks": [
                    _check("accepted windows whose three-standard-error interval leaves the box", unsafe_accept, 0.0,
                           kind="invariant"),
                    _check("near-bound windows (theta_hat inside, an interval end outside) accepted",
                           near_bound_accepted, 0.0, kind="invariant")]},
                tolerance=EXACT_TOL, uncertainty=EXACT),
        finding("The kernel refuses every host-owned code the adapter emitted", "numerical", refused,
                {"provider": base, "checks": [_refusal(f"require_status({code})", "raises ValueError", outcome)
                                              for code, outcome in refused.items()]},
                tolerance=EXACT_TOL, uncertainty=EXACT),
    ] + offline
    fields["numerical_result"] = (
        f"Windows: {len(windows)}; forwarded {[r['window'] for r in forwarded]} as {len(kernel_cases)} samples with "
        f"kernel codes {kernel_codes}; host decisions {accepted}; host statuses {host}; metadata leaks {len(leaks)}; "
        f"malformed samples {bad_fields}; guard errors {guard_errors}; theta estimate misses {estimate_misses} "
        f"(largest forwarded se {largest_se:.4f}); kernel refusal of host codes {refused}.")
    fields["uncertainty"] = ("Synthetic data with fixed seeds; the NIS band (6 standard deviations), the 3-se guard "
                             "and the 3-se estimate test are illustrative thresholds, not validated detection or "
                             "coverage rates.")
    return _finish(fields, findings, T113_FILES, identity)


# T114 ------------------------------------------------------------------------

@task("T114", changed_files=RESEARCH_FILES, regression_tests=(_node("test_t114_servo_pilot_spec"),
                                                                _node("test_research_tasks_without_optional_modules")))
def servo_pilot(ctx):
    fields = _fields(
        "A non-production servo-axis pilot can be specified so that the Lyapunov monitor's scope, data, abort "
        "criteria and lack of authority are explicit, every runtime abort trigger can actually fire for the "
        "declared monitor configuration, and the offline certificate check passes at every grid inertia before "
        "any powered test.",
        "Axis J theta'' = -b theta' + Kt u with PD state feedback designed for 20 Hz, damping 0.7 at nominal J; "
        "exact ZOH at Ts = 1 ms; closed loop A_cl(J) = Phi(J) - Gamma(J) K on a 9-point grid over J +- 30 %; "
        "common P from the discrete Lyapunov equation at nominal J. Online: PLSR on the nominal model with a "
        "declared level c such that {V <= c} lies inside the operating envelope; for a fixed model the decrease "
        "form's sign is state-independent, so the code depends on the state only through the level gate.",
        ["Placeholder parameters (not identified): J = 2e-3 kg m^2 +- 30 %, b = 1e-3 N m s, Kt = 0.1 N m/A, "
         "Ts = 1 ms", "Weightings Q = I and unit-balanced Q = diag(omega^2, 1), omega = 2 pi 20 rad/s",
         "Operating envelope |angle error| <= 0.05 rad, |rate| <= 2 rad/s (placeholder)",
         "Monitor scan: 200 seeded states (PCG64 seed 1141) with V(x) = r^2 c, r = 10^U(-8, 8)"],
        "Exact rational class of A_cl(J)^T P A_cl(J) - P at each grid inertia; ZOH exponential against its closed "
        "form and an independent exponential; monitor codes over the scan by the documented decision order, "
        "against the exact V(x) > c decision.",
        "The unit-balanced P is exactly valid at every grid inertia; the monitor's code is constant without a level "
        "set and follows exact V > c with it; every abort code is produced by the scan; authority and physical "
        "claims remain not_established.",
        "Build the sampled-data models, choose P, check each grid point exactly, verify the exponential, derive the "
        "level set, scan the monitor configuration, and retain the specification (JSON and Markdown).",
        "Hardware-gated: identify J, friction and delay on the bench, replace the placeholder interval, re-run "
        "this check with an affine over-approximation suitable for PLSR's box semantics, then run T113's adapter "
        "on acquired encoder data.",
        ["certificate invalid on part of the interval", "exponential inaccurate", "monitor code independent of the "
         "state even with a level set", "abort trigger that the configuration cannot produce",
         "level set outside the envelope", "authority or safety claimed"],
        ["All parameters are placeholders; the grid is evidence on 9 inertias, not a proof over the interval.",
         "The monitor scan uses CIW's transcription of the documented decision order, not the pinned runtime; "
         "T113 exercises the runtime on the analogous adapter.",
         "The current loop, friction nonlinearity, saturation and delay are not modelled; specification sections "
         "are guarded by the regression test, not by a finding."])
    K, models = X.servo_models()
    P, classes = X.servo_certificate(models)
    nominal = min(models, key=lambda m: abs(m["J"] - X.SERVO["J_nominal_kg_m2"]))
    augmented = np.zeros((3, 3))
    augmented[:2, :2], augmented[:2, 2:] = nominal["A"], nominal["B"]
    series = X.expm_series(augmented * X.SERVO["Ts_s"])
    closed = X.servo_zoh_closed_form(nominal["J"], X.SERVO["Ts_s"])
    independent = X.independent_expm(augmented * X.SERVO["Ts_s"])
    level = X.servo_level(P)
    scan = X.monitor_scan(nominal["A_cl"], P, level)
    without = sorted({r["code_without_level"] for r in scan})
    with_level = sorted({r["code_with_level"] for r in scan})
    level_mismatch = sum((r["code_with_level"] == "OUTSIDE_LEVEL_SET") != r["exact_exceeds_level"] for r in scan)
    unexpected = sorted(set(with_level) - set(X.SERVO_EXPECTED_CODES))
    unreachable_aborts = sorted(set(X.SERVO_ABORT_CODES) - set(with_level))
    inverse = np.linalg.inv(P)
    envelope_ratio = max(math.sqrt(level * inverse[0, 0]) / X.SERVO_ENVELOPE["angle_error_rad"],
                         math.sqrt(level * inverse[1, 1]) / X.SERVO_ENVELOPE["velocity_rad_s"])
    monitor = {"level": level, "codes_without_level": without, "codes_with_level": with_level}
    spec = X.servo_spec(K, P, classes, monitor)
    ctx.artifact_json("servo-pilot-spec.json", R.jsonable(spec))
    ctx.artifact_text("servo-pilot-spec.md", X.servo_spec_markdown(spec))
    ctx.artifact_json("monitor-scan.json", R.jsonable({"level": level, "rows": scan}))
    balanced, plain = classes["Q = diag(omega^2, 1)"], classes["Q = I"]
    findings = [
        finding("The unit-balanced nominal P gives an exactly negative definite discrete decrease at every grid "
                "inertia", "numerical", {"grid_points": len(balanced),
                                         "not_negative_definite": sum(c != "negative_definite" for c in balanced)},
                {"generator": {"name": "servo_models", "seed": None},
                 "checks": [_check("grid inertias without an exactly negative definite decrease",
                                   sum(c != "negative_definite" for c in balanced), 0.0)]},
                tolerance=EXACT_TOL, uncertainty=EXACT),
        finding("With Q = I the nominal-model P does not cover the declared inertia interval", "numerical",
                {"failing_grid_points": sum(c != "negative_definite" for c in plain), "grid_points": len(plain)},
                {"checks": [_check("grid inertias failing with Q = I", sum(c != "negative_definite" for c in plain),
                                   1.0, "ge")]},
                tolerance=EXACT_TOL,
                counterexample={"statement": "A Lyapunov P solved at the nominal model with Q = I certifies the "
                                             "declared +-30 % inertia interval",
                                "witness": {"J": [m["J"] for m, c in zip(models, plain) if c != "negative_definite"]}},
                uncertainty=EXACT),
        _exponential_finding("The ZOH exponential agrees with its closed form and, where available, an independent "
                             "exponential", series, closed, independent, "exp([[A, B], [0, 0]] Ts)"),
        finding("Without a level set the monitor's code is the same at every state; with the declared level it "
                "follows the exact V(x) > c decision", "numerical",
                {"states": len(scan), "codes_without_level": without, "codes_with_level": with_level,
                 "level_mismatches": level_mismatch, "exceeding_states": sum(r["exact_exceeds_level"] for r in scan)},
                {"generator": {"name": "monitor_scan", "seed": 1141},
                 "checks": [_check("distinct codes over the scan without a level set", len(without), 1.0, "le",
                                   kind="invariant"),
                            _check("states whose OUTSIDE_LEVEL_SET decision differs from exact V(x) > c",
                                   level_mismatch, 0.0),
                            _check("largest ratio of the level set's extent to the envelope bound", envelope_ratio,
                                   1.0, "le", kind="analytic")]},
                tolerance=EXACT_TOL, uncertainty=_roundoff(0.0, "exact rationals decide V > c; states lie between "
                                                                "1e-8 and 1e8 of the level radius")),
        finding("Every runtime code named as an abort trigger is produced by the declared monitor configuration, "
                "and the configuration produces no code outside the expected set", "computational_pipeline",
                {"abort_codes": list(X.SERVO_ABORT_CODES), "expected_codes": list(X.SERVO_EXPECTED_CODES),
                 "produced": with_level},
                {"checks": [_check("abort codes never produced by the monitor scan", len(unreachable_aborts), 0.0,
                                   kind="invariant"),
                            _check("produced codes outside the expected set", len(unexpected), 0.0,
                                   kind="invariant")]},
                tolerance=EXACT_TOL, uncertainty=EXACT),
        finding("The servo-axis pilot is safe to operate", "machine_safety", None,
                {"derivation": "requires a safety case and an independent safety function outside the workbench"}),
        finding("The Lyapunov monitor may command, gate or release the axis", "actuator_authority", None,
                {"derivation": "the specification grants no actuator authority; aborts are stop requests to an "
                               "independent safety function"}),
        finding("The pilot configuration is acceptable for production use", "production_acceptance", None, {}),
        finding("The monitor is ready for industrial deployment", "industrial_readiness", None, {}),
        finding("The placeholder inertia interval contains the real axis inertia", "physical", None, {}),
        finding("Encoder and current-sensor calibrations of the bench are valid", "calibration", None, {}),
    ]
    exponential = findings[2]["value"]
    fields["numerical_result"] = (
        f"K = {np.round(K, 6).tolist()}; balanced-Q grid classes {_counts(balanced)}; Q = I grid classes "
        f"{_counts(plain)}; ZOH exponential differs from its closed form by "
        f"{exponential['max_abs_difference_closed_form']:.2e}"
        + (f" and from {independent[1]} by {exponential['max_abs_difference_independent']:.2e}"
           if independent is not None else " (no independent exponential applies to this defective matrix here)")
        + f"; level c = {level:.6g} (extent/envelope {envelope_ratio:.6f}); monitor codes without level {without}, "
        f"with level {with_level}, {level_mismatch} level decisions differing from exact V > c; abort codes not "
        f"produced: {unreachable_aborts}.")
    fields["uncertainty"] = ("Exact rational classes on the declared float matrices; the interval between grid "
                             "points is not covered by a proof; the level-set scan decides V > c exactly.")
    return _finish(fields, findings, RESEARCH_FILES, provider=False)
