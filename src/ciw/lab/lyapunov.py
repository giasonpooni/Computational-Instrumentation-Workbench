"""Lyapunov runtime experiments T101-T114 against the pinned PLSR instrument.

Scope: the float64 decision procedure of the Parameterized Lyapunov Stability
Runtime (PLSR 0.1.0rc1, commit 19ea696) -- resolution floor, power-of-two
scaling, overflow and underflow, semidefinite and defective edge cases, unit
changes, the nine runtime-status-v1 codes, the declared-margin policy and the
continuous/discrete conventions -- compared with CIW's exact dyadic-rational
arithmetic, CIW's re-derivation of the documented formulas, NumPy eigenvalues
and, when installed, SciPy Lyapunov solvers. The runtime runs in a subprocess
under the interpreter bound as provider ``plsr-python`` after its source pin is
verified; without it the provider-dependent tasks are reported ``blocked`` and
retain only their provider-free findings. T112-T114 are specifications with
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
from . import svg
from .evidence import COMPUTATIONAL_DOMAINS, finding
from .lyapunov_provider import PLSR_ROLE, ProviderRefusal, manifest, producer, provider_basis, run_plsr, source_digest
from .registry import task

MODULE = "src/ciw/lab/lyapunov.py"
REFERENCE = "src/ciw/lab/lyapunov_reference.py"
BRIDGE = "src/ciw/lab/lyapunov_provider.py"
RESEARCH = "src/ciw/lab/lyapunov_research.py"
DOC = "docs/lab/LYAPUNOV.md"
TESTS = "tests/test_lab_lyapunov.py"
PROVIDER_FILES = (MODULE, REFERENCE, BRIDGE, DOC)
CIW_REFERENCE = {"implementation": "ciw.lab.lyapunov_reference", "revision": __version__}
REQUIREMENT = f"provider:{PLSR_ROLE}"


def _node(name):
    return f"{TESTS}::{name}"


# Shared plumbing -------------------------------------------------------------

class _Unavailable(Exception):
    """The provider part of a task cannot run here (unbound or refused)."""


def _bridge(ctx, cases):
    if not ctx.available(REQUIREMENT):
        raise _Unavailable(f"Provider {PLSR_ROLE} is not bound; bind a Python 3.12+ interpreter with the pinned "
                           f"PLSR via --provider {PLSR_ROLE}=PATH.")
    try:
        return run_plsr(ctx.providers[PLSR_ROLE], cases)
    except ProviderRefusal as exc:
        raise _Unavailable(f"Provider {PLSR_ROLE} refused: {exc}.") from exc


def _pin_identity():
    pin = manifest()
    return {"role": PLSR_ROLE, "repository": pin["repository"], "commit": pin["commit"],
            "package_version": pin["package_version"], "source_digest": source_digest(pin), "executed": False}


def _finish(fields, findings, changed, identity=None, blocked=None, provider=True):
    """Assemble the outcome; a failed planned check downgrades completed to partial, never hides it."""
    from .runner import builtin_identity

    fields = dict(fields)
    state = "completed"
    if blocked is not None:
        state = "blocked"
        fields["experiment"] = (f"Blocked: {blocked} Planned: {fields['experiment']} The provider-free parts ran; "
                                "their findings are retained.")
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
    holds = {"abs_le": abs(observed) <= tolerance, "le": observed <= tolerance,
             "ge": observed >= tolerance}[comparison]
    return {"reference_kind": kind, "reference": reference, "observed": observed, "tolerance": tolerance,
            "comparison": comparison, "passed": holds}


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
    """PLSR certifies the subnormal witness although exact arithmetic denies a negative definite form."""
    exact = _witness_exact()
    A, P, x = _subnormal_witness()
    return finding(
        claim, "numerical",
        {"code": _code(result), "exact_class": exact["exact_class"], "exact_det_units2": exact["exact_det_units2"],
         "resolution": result.get("resolution", 0.0)},
        {"provider": provider_basis(identity),
         "checks": [_check("exact rational determinant of the declared decrease form, units (2^-1074)^2",
                           exact["exact_det_units2"], 0.0, "le"),
                    _check("PLSR code is CERTIFIED_WITH_MARGIN (1 if so)",
                           1.0 if _code(result) == "CERTIFIED_WITH_MARGIN" else 0.0, 1.0, "ge", "invariant")]},
        tolerance={"abs": 0.0, "rel": 0.0},
        counterexample={"statement": statement,
                        "witness": {"A_hex": R.hexed(A), "P_hex": R.hexed(P), "x": x.tolist(),
                                    "exact_form_units_of_2^-1074": exact["exact_form_units"],
                                    "binary64_form_units_of_2^-1074": exact["float_form_units"],
                                    "plsr_details": result.get("details")}})


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
                         "normal": _normal(A) and _normal(R.decrease_matrix(A, np.eye(2)))})
    for seed in range(4):
        rng = R.generator(101 + seed)
        A0 = rng.normal(size=(3, 3)) - 3.0 * np.eye(3)
        P0 = R.kron_lyapunov(A0, np.eye(3))
        for k in (-1074, -1070, -1066, -1062, -1058, -1050, -1040, -1022, -1000, -500, 0, 500, 1000, 1015, 1018,
                  1020, 1021, 1022, 1023):
            with np.errstate(over="ignore"):
                A = np.ldexp(A0, k)
            if not np.all(np.isfinite(A)):
                continue  # not a declarable binary64 matrix
            cases.append(_verdict_case(f"F2:{seed}:{k}", A, P0, np.ones(3)))
            meta.append({"family": "F2", "seed": seed, "k": k, "A": A, "P": P0})
    A, P, x = _subnormal_witness()
    cases.append(_verdict_case("witness", A, P, x))
    cases.append(_verdict_case("witness-unit", A / R.TINY, P, x))
    return cases, meta


def _t101_offline():
    exact = _witness_exact()
    homogeneity = []
    A0 = np.array([[-0.1, 1.0], [-1.0, -0.1]])
    base = R.resolution(A0, np.eye(2))
    for k in T101_SCALES:
        A = np.ldexp(A0, k)
        if _normal(A) and _normal(R.decrease_matrix(A, np.eye(2))):
            homogeneity.append(abs(np.ldexp(R.resolution(A, np.eye(2)), -k) - base) / base)
    return [
        finding("Inconclusive threshold of the family A = s[[-eps, 1], [-1, -eps]], P = I: certified iff eps > "
                "4 gamma_5 / (1 - 8u), independent of s", "mathematical", EPSILON_STAR,
                {"derivation": "n = 2, max|A| = s, max|P| = 1, M = -2 eps s I exactly: resolution = s(8 gamma_5 + "
                               "16 u eps) against margin 2 eps s (docs/lab/LYAPUNOV.md, T101)"},
                tolerance={"abs": 0.0, "rel": 1e-12}),
        finding("CIW's re-derived resolution scales exactly with a power-of-two matrix scale in the normal range",
                "numerical", {"scales": len(homogeneity), "max_relative_deviation": max(homogeneity)},
                {"generator": {"name": "power-of-two scales of [[-0.1, 1], [-1, -0.1]]", "seed": None},
                 "checks": [_check("res(2^k A) 2^-k versus res(A), bitwise", max(homogeneity), 0.0,
                                   kind="invariant")]},
                tolerance={"abs": 0.0, "rel": 0.0}),
        finding("The subnormal witness's exact decrease form is indefinite while its binary64 evaluation is "
                "negative definite", "numerical",
                {"exact_det_units2": exact["exact_det_units2"], "float_det_units2": exact["float_det_units2"],
                 "exact_class": exact["exact_class"], "float_class": exact["float_class"]},
                {"checks": [_check("exact determinant of [[-4, 5], [5, -6]] (units 2^-2148)", exact["exact_det_units2"],
                                   0.0, "le"),
                            _check("determinant of the rounded form [[-4, 4], [4, -6]] (units 2^-2148)",
                                   exact["float_det_units2"], 1.0, "ge")]},
                tolerance={"abs": 0.0, "rel": 0.0}),
    ]


@task("T101", changed_files=PROVIDER_FILES, regression_tests=(_node("test_t101_resolution_floor"),
                                                                _node("test_reference_witness_is_exactly_indefinite")))
def resolution_floor(ctx):
    fields = _fields(
        "PLSR's resolution floor is the documented Higham/Weyl/LAPACK bound and scales exactly with the matrix, so "
        "the inconclusive threshold is scale-invariant across the binary64 normal range; the bound is not derived "
        "for subnormal arithmetic, where it can underflow below the actual rounding error.",
        "Continuous decrease form M = A^T P + P A (symmetrised); resolution = n*(2 gamma_{n+3} n max|A| max|P|) + "
        "n^3 u max|M|; certified iff -max eig(M) > resolution. For A = s[[-eps, 1], [-1, -eps]], P = I the "
        "threshold is eps* = 4 gamma_5/(1 - 8u) = 2.2204e-15 for every s with normal arithmetic.",
        ["F1: 8 eps values x 25 exponents k in [-1074, 1023], A = 2^k [[-eps, 1], [-1, -eps]], P = I, x = (1, 0.5)",
         "F2: 4 seeded Hurwitz 3x3 A0 (PCG64 seeds 101-104) with P0 from a CIW Kronecker Lyapunov solve, "
         "A = 2^k A0 for 19 exponents, x = ones",
         "Subnormal witness A = [[-2, 5], [0, -3]] 2^-1074, P = I, x = (1, 0), and its unit-scale twin"],
        "Runtime verdict code, resolution and margin per case, returned by the pinned PLSR in a subprocess.",
        "resolution_PLSR = resolution_CIW; code at scale 2^k equals the unit-scale code whenever every entry of A "
        "and M is a normal number; exactly-indefinite declared forms are never certified.",
        "Evaluate every case with PLSR, recompute the documented resolution and the exact rational class of the "
        "declared form in CIW, compare codes with the analytic threshold and the exact class, and locate where the "
        "prediction stops holding.",
        "T102: test power-of-two homogeneous scaling of A, P and x together; then propose a subnormal-aware "
        "resolution term (an absolute n^2 * 2^-1074 floor) upstream and re-run T101.",
        ["resolution differs from the documented formula", "normalised resolution drifts with scale",
         "threshold moves with scale", "NUMERICAL_OVERFLOW boundary differs from where the form overflows",
         "false certificate below the normal range (exact rational class)"],
        ["The exact class concerns the declared binary64 numbers; they are synthetic, not measured plants.",
         "Subnormal behaviour of numpy.linalg.eigvalsh (LAPACK dsyevd) may differ between BLAS builds."])
    offline = _t101_offline()
    cases, meta = _t101_family()
    try:
        bridge = _bridge(ctx, cases)
    except _Unavailable as exc:
        fields["numerical_result"] = f"Provider-free: eps* = {EPSILON_STAR:.6e}; the subnormal witness is exactly indefinite."
        fields["uncertainty"] = "Exact rational arithmetic for the witness; analytic threshold."
        return _finish(fields, offline, PROVIDER_FILES, blocked=str(exc))
    identity, results = bridge["identity"], bridge["results"]
    rows, rel_diffs, f1_normal, f1_mismatch, f1_below = [], [], 0, 0, []
    unit_codes = {m["eps_index"]: _code(results[f"F1:{m['eps_index']}:0"]) for m in meta if m["family"] == "F1"}
    normalised, false_certificates, overflow_first = [], [], {}
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
                f1_mismatch += code != unit_codes[m["eps_index"]]
                if m["eps_index"] == 0 and result.get("sample") is not None:
                    normalised.append(abs(np.ldexp(result["resolution"], -m["k"])
                                          - results["F1:0:0"]["resolution"]) / results["F1:0:0"]["resolution"])
            elif code != unit_codes[m["eps_index"]]:
                f1_below.append({"eps": m["eps"], "k": m["k"], "code": code, "unit_code": unit_codes[m["eps_index"]]})
        else:
            exact = R.exact_class(R.exact_form(m["A"], m["P"]))
            row["exact_class"] = exact
            if code in R.CERTIFYING and exact != "negative_definite":
                false_certificates.append({"seed": m["seed"], "k": m["k"], "code": code, "exact_class": exact})
            with np.errstate(over="ignore", invalid="ignore"):
                overflows = not np.all(np.isfinite(R.decrease_matrix(m["A"], m["P"])))
            if overflows and m["seed"] not in overflow_first:
                overflow_first[m["seed"]] = {"k": m["k"], "code": code}
        rows.append(row)
    ctx.artifact_json("resolution-floor.json", R.jsonable({"epsilon_star": EPSILON_STAR, "rows": rows,
                                                            "f1_departures_outside_normal_range": f1_below,
                                                            "f2_false_certificates": false_certificates}))
    series = []
    for i in (0, 3, 4, 5):
        ks = [m["k"] for m in meta if m["family"] == "F1" and m["eps_index"] == i]
        ratios = [results[f"F1:{i}:{k}"]["margin_ratio"] if results[f"F1:{i}:{k}"].get("sample") else float("nan")
                  for k in ks]
        series.append((f"eps={T101_EPSILONS[i]:.3g}", ks, [abs(r) if math.isfinite(r) else float("nan") for r in ratios]))
    ctx.artifact_text("margin-ratio-vs-scale.svg", svg.line_plot(
        series, title="T101 |margin / resolution| across matrix scales 2^k", xlabel="k (A scaled by 2^k)",
        ylabel="|margin ratio|", logy=True))
    overflow_mismatch = sum(entry["code"] != "NUMERICAL_OVERFLOW" for entry in overflow_first.values())
    witness = results["witness"]
    findings = [
        finding("PLSR decrease_resolution equals the documented bound re-derived in CIW at every evaluated matrix scale",
                "numerical", {"cases": len(rel_diffs), "max_relative_difference": max(rel_diffs)},
                {"provider": provider_basis(identity),
                 "independent_check": _independent(_check("CIW re-derivation of the documented resolution",
                                                           max(rel_diffs), 1e-12, kind="analytic"), identity)},
                tolerance={"abs": 1e-12, "rel": 0.0}),
        finding("Inside the binary64 normal range the resolution scales exactly with 2^k and the inconclusive "
                "threshold stays at eps*", "numerical",
                {"normal_range_cases": f1_normal, "threshold_mismatches": f1_mismatch,
                 "max_normalised_resolution_deviation": max(normalised),
                 "unit_scale_codes": [unit_codes[i] for i in range(len(T101_EPSILONS))]},
                {"provider": provider_basis(identity),
                 "checks": [_check("codes versus the analytic threshold eps*", f1_mismatch, 0.0, kind="analytic"),
                            _check("res(2^k A) 2^-k versus res(A), bitwise", max(normalised), 0.0, kind="invariant")]},
                tolerance={"abs": 0.0, "rel": 0.0}),
        finding("NUMERICAL_OVERFLOW first appears exactly where the CIW-recomputed decrease form overflows", "numerical",
                {str(seed): entry for seed, entry in sorted(overflow_first.items())},
                {"provider": provider_basis(identity),
                 "checks": [_check("seeds whose first overflowing scale is not NUMERICAL_OVERFLOW", overflow_mismatch,
                                   0.0, kind="invariant")]},
                tolerance={"abs": 1.0, "rel": 0.0}),
        _witness_finding(witness, identity,
                         "The float64 resolution floor bounds the rounding error of the decrease form at every matrix "
                         "scale, so an exactly indefinite declared form is never certified",
                         "Below the normal range the resolution underflows to zero and PLSR certifies a declared "
                         "decrease form that exact arithmetic shows indefinite"),
    ] + offline
    fields["numerical_result"] = (
        f"Resolution matches the documented formula (max relative difference {max(rel_diffs):.2e} over "
        f"{len(rel_diffs)} cases). In {f1_normal} normal-range threshold cases codes equal the unit-scale codes "
        f"({f1_mismatch} mismatches; eps* = {EPSILON_STAR:.6e}). Outside the normal range {len(f1_below)} threshold "
        f"cases depart from the unit-scale code; F2 has {len(false_certificates)} certifying verdicts whose exact form "
        f"is not negative definite. The witness returns {_code(witness)} with resolution {witness['resolution']:.3g} "
        f"(unit-scale twin: {_code(results['witness-unit'])}). First overflow exponents: "
        f"{ {s: e['k'] for s, e in sorted(overflow_first.items())} }.")
    fields["uncertainty"] = ("Resolution comparison is deterministic float arithmetic (relative 1e-12 allowed). "
                             "Subnormal-range codes depend on LAPACK's handling of tiny matrices and may differ "
                             "between BLAS builds; the witness's exact class is exact rational arithmetic.")
    return _finish(fields, findings, PROVIDER_FILES, identity)
