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


def _t101_offline():
    exact = _witness_exact()
    homogeneity = []
    A0 = np.array([[-0.1, 1.0], [-1.0, -0.1]])
    base = R.resolution(A0, np.eye(2))
    for k in T101_SCALES:
        A = np.ldexp(A0, k)
        if _normal_window(A, np.eye(2)):
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
         "A = 2^k A0 for 16 exponents in [-1074, 1020], x = ones",
         "F3: the same plants with A = 2^k A0 and P = 2^k P0 for k in [500, 512] (overflow boundary)",
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
        elif m["family"] == "F2":
            exact = R.exact_class(R.exact_form(m["A"], m["P"]))
            row["exact_class"] = exact
            if code in R.CERTIFYING and exact != "negative_definite":
                false_certificates.append({"seed": m["seed"], "k": m["k"], "code": code, "exact_class": exact})
        else:
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


@task("T102", changed_files=PROVIDER_FILES, regression_tests=(_node("test_t102_power_of_two_scaling"),
                                                                _node("test_eigvalsh_scaling_window")))
def power_of_two_scaling(ctx):
    fields = _fields(
        "Replacing (A, P, x) by (2^a A, 2^b P, 2^c x) multiplies M, max eig(M) and the resolution by 2^(a+b) "
        "exactly, so no verdict code and no margin ratio changes while every quantity stays normal and LAPACK "
        "does not rescale internally; outside that window rounding differs and near-threshold codes may flip.",
        "Continuous time: M(2^a A, 2^b P) = 2^(a+b) M(A, P) and resolution likewise (homogeneous of degree one in "
        "each of max|A|, max|P|, max|M|); PLSR divides x by a power of two before evaluating, so c never enters. "
        "Discrete time admits only (P, x) scaling. LAPACK dsyevd rescales by a non-power-of-two factor when "
        "max|M| lies outside [2^-485, 2^485].",
        ["34 seeded continuous cases (PCG64 seed 102): n = 2..4, P = I or SPD (condition 10), kappa in "
         "{+-0.5, +-0.9, +-1.1, +-1.5, +-3} plus four robust cases (kappa = +-1e6)",
         "16 razor-edge cases (seed 1022, skew 1): computed |max eig(M)|/resolution tuned by bisection to just "
         "above 1 on the certified and the indefinite side",
         f"Inside-window exponent triples (a, b, c): {list(T102_INSIDE)}",
         f"Outside-window triples: {list(T102_OUTSIDE)}",
         "6 discrete-time cases (seed 1021) under (0, b, c) scaling; the subnormal witness scaled by 2^-1074"],
        "PLSR code and margin ratio (margin/resolution) for each scaled case versus the unscaled case.",
        "Inside the window: identical codes and bitwise-identical margin ratios. The subnormal witness keeps its "
        "unit-scale code DECREASE_NOT_DEFINITE.",
        "Evaluate base and scaled cases with PLSR in one subprocess; count code flips and ratio changes per window; "
        "independently test numpy.linalg.eigvalsh against exact power-of-two scaling inside and outside the window.",
        "T103: overflow and underflow of states and parameters; propose that PLSR pre-scale M by a power of two "
        "into LAPACK's window before eigvalsh so verdicts are exactly scale-invariant.",
        ["code flip inside the window", "margin ratio changes inside the window", "x scaling alters the sample",
         "code flip outside the window (searched as counterexample)", "subnormal flip"],
        ["The LAPACK window [2^-485, 2^485] is taken from reference dsyevd (RMIN, RMAX); other LAPACK builds may "
         "rescale differently.",
         "Near-threshold inputs are synthetic; their exact spectra are not needed for an invariance test."])
    inside_eig, outside_eig = _eigvalsh_window()
    offline = [finding(
        "numpy.linalg.eigvalsh commutes exactly with power-of-two scaling while max|M| stays in [2^-485, 2^485]",
        "numerical", {"inside_bitwise_equal": sum(inside_eig), "inside_total": len(inside_eig)},
        {"generator": {"name": "near_threshold_family", "seed": 1020},
         "checks": [_check("eigenvalues of 2^j M rescaled by 2^-j versus eigenvalues of M, bitwise",
                           len(inside_eig) - sum(inside_eig), 0.0, kind="invariant")]},
        tolerance={"abs": 0.0, "rel": 0.0})]
    family = near_threshold_family(102, 30) + razor_family(1022, 16)
    cases, plan = [], []
    for i, member in enumerate(family):
        A, P, x = member["A"], member["P"], member["x"]
        cases.append(_verdict_case(f"c{i}:base", A, P, x))
        for window, triples in (("inside", T102_INSIDE), ("outside", T102_OUTSIDE)):
            for a, b, c in triples:
                cid = f"c{i}:{a}:{b}:{c}"
                cases.append(_verdict_case(cid, np.ldexp(A, a), np.ldexp(P, b), np.ldexp(x, c)))
                plan.append((f"c{i}:base", cid, window, member["kappa"], (a, b, c)))
    rng = R.generator(1021)
    for i in range(6):
        n = 2 + i % 2
        A = 0.8 * R.random_orthogonal(rng, n) @ np.diag(rng.uniform(0.3, 1.0, n))
        P = R.kron_lyapunov(A, np.eye(n), "discrete")
        x = rng.normal(size=n)
        cases.append(_verdict_case(f"d{i}:base", A, P, x, time="discrete"))
        for b, c in ((13, 0), (-13, 5), (0, -700)):
            cid = f"d{i}:{b}:{c}"
            cases.append(_verdict_case(cid, A, np.ldexp(P, b), np.ldexp(x, c), time="discrete"))
            plan.append((f"d{i}:base", cid, "discrete", None, (0, b, c)))
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
    tally = {w: {"evaluations": 0, "code_flips": 0, "ratio_changes": 0} for w in ("inside", "outside", "discrete")}
    flips = []
    members = {f"c{i}:base": member for i, member in enumerate(family)}
    unsound = 0
    for base_id, cid, window, kappa, triple in plan:
        base, scaled = results[base_id], results[cid]
        tally[window]["evaluations"] += 1
        if _code(base) != _code(scaled):
            tally[window]["code_flips"] += 1
            flip = {"window": window, "kappa": kappa, "abc": list(triple), "base": _code(base),
                    "scaled": _code(scaled), "base_ratio": base["margin_ratio"], "scaled_ratio": scaled["margin_ratio"]}
            if base_id in members and R.CERTIFYING & {flip["base"], flip["scaled"]}:
                # Exact power-of-two scaling preserves the exact class of the declared form.
                member = members[base_id]
                flip["exact_class"] = R.exact_class(R.exact_form(member["A"], member["P"]))
                unsound += flip["exact_class"] != "negative_definite"
            flips.append(flip)
        if base["ok"] and scaled["ok"] and base["margin_ratio"] != scaled["margin_ratio"]:
            tally[window]["ratio_changes"] += 1
    ctx.artifact_json("scaling-invariance.json", R.jsonable({"tally": tally, "flips": flips,
                                                              "eigvalsh_outside_bitwise_equal": sum(outside_eig),
                                                              "eigvalsh_outside_total": len(outside_eig)}))
    witness_unit, witness_sub = results["w:unit"], results["w:sub"]
    base = provider_basis(identity)
    findings = [finding(
        "Power-of-two scaling of (A, P, x) inside LAPACK's window never changes the PLSR code or margin ratio",
        "numerical", tally["inside"],
        {"provider": base, "checks": [
            _check("code flips against the unscaled verdict", tally["inside"]["code_flips"], 0.0, kind="invariant"),
            _check("margin ratios not bitwise equal", tally["inside"]["ratio_changes"], 0.0, kind="invariant")]},
        tolerance={"abs": 0.0, "rel": 0.0}),
        finding("Scaling P and x by powers of two never changes a discrete-time PLSR code", "numerical",
                tally["discrete"],
                {"provider": base, "checks": [_check("code flips against the unscaled verdict",
                                                     tally["discrete"]["code_flips"], 0.0, kind="invariant")]},
                tolerance={"abs": 0.0, "rel": 0.0})]
    outside_flips = [flip for flip in flips if flip["window"] == "outside"]
    if outside_flips:
        findings.append(finding(
            "Outside LAPACK's scaling window a power-of-two rescaling of A and P flips near-threshold PLSR codes",
            "numerical", {"flips_observed": True},
            {"provider": base, "checks": [_check("outside-window code flips observed", len(outside_flips), 1.0, "ge",
                                                 kind="invariant"),
                                          _check("flips into or out of a certifying code whose exact form is not "
                                                 "negative definite", unsound, 0.0)]},
            counterexample={"statement": "Power-of-two scaling of A and P never changes a PLSR verdict code while all "
                                         "quantities stay in the binary64 normal range",
                            "witness": outside_flips[0]}))
    else:
        findings.append(finding(
            "No outside-window code flip was observed on this platform", "numerical",
            {"flips_observed": False, "ratio_changes": tally["outside"]["ratio_changes"]},
            {"provider": base, "checks": [_check("outside-window code flips", 0.0, 0.0, kind="invariant")]}))
    findings.append(finding(
        "Scaling the subnormal witness by 2^-1074 turns DECREASE_NOT_DEFINITE into CERTIFIED_WITH_MARGIN", "numerical",
        {"unit_code": _code(witness_unit), "scaled_code": _code(witness_sub)},
        {"provider": base, "checks": [
            _check("unit-scale code is DECREASE_NOT_DEFINITE (1 if so)",
                   1.0 if _code(witness_unit) == "DECREASE_NOT_DEFINITE" else 0.0, 1.0, "ge", "invariant"),
            _check("scaled code is CERTIFIED_WITH_MARGIN (1 if so)",
                   1.0 if _code(witness_sub) == "CERTIFIED_WITH_MARGIN" else 0.0, 1.0, "ge", "invariant")]},
        counterexample={"statement": "Power-of-two scaling of A never changes a PLSR verdict code",
                        "witness": {"A_unit": [[-2.0, 5.0], [0.0, -3.0]], "scale": "2^-1074", "P": "I",
                                    "x": [1.0, 0.0]}}))
    findings += offline
    fields["numerical_result"] = (
        f"Inside the window: {tally['inside']['evaluations']} scaled evaluations, {tally['inside']['code_flips']} code "
        f"flips, {tally['inside']['ratio_changes']} ratio changes. Outside the window: "
        f"{tally['outside']['evaluations']} evaluations, {tally['outside']['code_flips']} flips ({unsound} involving "
        f"a certifying code on an exactly non-negative-definite form), "
        f"{tally['outside']['ratio_changes']} ratio changes. Discrete (P, x) scaling: "
        f"{tally['discrete']['code_flips']} flips in {tally['discrete']['evaluations']}. Subnormal witness: "
        f"{_code(witness_unit)} -> {_code(witness_sub)}. eigvalsh bitwise homogeneous in "
        f"{sum(inside_eig)}/{len(inside_eig)} inside and {sum(outside_eig)}/{len(outside_eig)} outside scalings.")
    fields["uncertainty"] = ("Inside-window results are exact (bitwise). Outside-window flip counts depend on the "
                             "LAPACK/BLAS build and on how close each case sits to the threshold; only their "
                             "existence is retained as a finding value.")
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
                                                                _node("test_level_gate_prediction")))
def overflow_underflow(ctx):
    fields = _fields(
        "States anywhere in binary64 are classified exactly through PLSR's power-of-two state scaling; matrices "
        "whose arithmetic leaves binary64 return NUMERICAL_OVERFLOW or an input refusal; no near-limit input "
        "yields a false certificate or a missed level-set exceedance.",
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
         "The documented level rule is re-derived from the runtime source at the pinned commit."])
    rows = level_scan()
    predicted_missed, predicted_spurious = _level_counts(rows, "documented_exceeded")
    offline = [finding(
        "The documented level rule, re-derived in CIW, misses exceedances when s^2 underflows and reports spurious "
        "ones when s^2 overflows", "numerical",
        {"cases": len(rows), "predicted_missed": predicted_missed, "predicted_spurious": predicted_spurious},
        {"checks": [_check("predicted missed exceedances (exact exponent comparison)", predicted_missed, 1.0, "ge"),
                    _check("predicted spurious exceedances", predicted_spurious, 1.0, "ge")]},
        tolerance={"abs": 0.0, "rel": 0.0})]
    P_ok = np.eye(2)
    cases = [_verdict_case(f"s{i}", T103_A, P_ok, state) for i, state in enumerate(T103_STATES)]
    cases += [_verdict_case("inf", T103_A, P_ok, (math.inf, 0.0)), _verdict_case("nan", T103_A, P_ok, (math.nan, 0.0))]
    for i, row in enumerate(rows):
        cases.append(_verdict_case(f"L{i}", -np.eye(2), np.ldexp(np.eye(2), row["p"]),
                                   (float(np.ldexp(1.0, row["e"])), 0.0), level=float(np.ldexp(1.0, row["log2_level"]))))
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
        fields["numerical_result"] = (f"Provider-free: the documented level rule predicts {predicted_missed} missed and "
                                      f"{predicted_spurious} spurious exceedances in {len(rows)} scan cases.")
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
                                                       "theta_details": {k: results[k].get("error") for k in theta_codes},
                                                       "witness": results["witness"]}))
    findings = [
        finding("PLSR decides the level gate exactly as the documented rule predicts across the near-limit scan",
                "numerical", {"cases": len(rows), "disagreements": disagreement},
                {"provider": base, "independent_check": _independent(
                    _check("documented level rule re-derived in CIW", disagreement, 0.0, kind="analytic"), identity)},
                tolerance={"abs": 0.0, "rel": 0.0}),
        finding("States from 2^-1074 to the largest binary64 number leave the PLSR code of a fixed negative definite "
                "form unchanged", "numerical", {"states": len(state_codes), "mismatches": state_mismatch,
                                                 "code": reference_code},
                {"provider": base, "checks": [_check("codes differing from the unit-state code", state_mismatch, 0.0,
                                                     kind="invariant")]},
                tolerance={"abs": 0.0, "rel": 0.0}),
        finding("Non-finite states are refused as input errors, not classified", "numerical",
                {"inf": _code(results["inf"]), "nan": _code(results["nan"])},
                {"provider": base, "checks": [_refusal("x = (inf, 0)", "raises ValueError", _code(results["inf"])),
                                              _refusal("x = (nan, 0)", "raises ValueError", _code(results["nan"]))]}),
        finding("Matrices whose decrease form or scaled V leaves binary64 return NUMERICAL_OVERFLOW as predicted",
                "numerical", limit_codes,
                {"provider": base, "checks": [_check("codes differing from the CIW re-derived prediction",
                                                     sum(limit_codes[k] != predicted_limits[k] for k in limits), 0.0,
                                                     kind="analytic")]}),
    ]
    if missed:
        findings.append(finding(
            "The PLSR level gate misses exceedances when s^2 underflows: V > level is certified", "numerical",
            {"missed": missed, "cases": len(rows)},
            {"provider": base, "checks": [_check("scan cases with exact V > level and code != OUTSIDE_LEVEL_SET",
                                                 missed, 1.0, "ge")]},
            tolerance={"abs": 0.0, "rel": 0.0},
            counterexample={"statement": "OUTSIDE_LEVEL_SET is returned whenever V(x) exceeds the declared level",
                            "witness": dict(first_missed, A="-I", P=f"2^{first_missed['p']} I",
                                            x=f"(2^{first_missed['e']}, 0)")}))
    if spurious:
        findings.append(finding(
            "The PLSR level gate reports OUTSIDE_LEVEL_SET for V below the level when s^2 overflows", "numerical",
            {"spurious": spurious, "cases": len(rows)},
            {"provider": base, "checks": [_check("scan cases with exact V < level and code OUTSIDE_LEVEL_SET",
                                                 spurious, 1.0, "ge")]},
            tolerance={"abs": 0.0, "rel": 0.0},
            counterexample={"statement": "The level gate is decided exactly through the power-of-two scaling",
                            "witness": dict(first_spurious, A="-I", P=f"2^{first_spurious['p']} I",
                                            x=f"(2^{first_spurious['e']}, 0)")}))
    raised = theta_codes["theta:+1e308,c=2"]
    findings.append(finding(
        "A finite in-box theta whose A(theta) overflows raises an input error instead of NUMERICAL_OVERFLOW",
        "numerical", theta_codes,
        {"provider": base, "checks": [_refusal("theta = 1e308, A1 = 2I (A(theta) = inf)", "raises ValueError", raised),
                                      _refusal("theta = 1e308, A1 = I (A finite, M overflows)", "NUMERICAL_OVERFLOW",
                                               theta_codes["theta:+1e308,c=1"])]},
        counterexample={"statement": "Every finite in-box sample yields a runtime-status-v1 code",
                        "witness": {"A0": "-I", "A1": "2I", "box": [-1e308, 1e308], "theta": 1e308,
                                    "error": results["theta:+1e308,c=2"].get("error")}}))
    findings.append(_witness_finding(
        results["witness"], identity,
        "No binary64 input near the representable limits yields a false certificate",
        "A subnormal plant matrix yields CERTIFIED_WITH_MARGIN for an exactly indefinite decrease form"))
    findings += offline
    fields["numerical_result"] = (
        f"{len(state_codes)} states from 2^-1074 to 1.797e308: {state_mismatch} code changes (all {reference_code}). "
        f"Level scan ({len(rows)} cases): {missed} missed exceedances (certified with V > level, s^2 underflowed), "
        f"{spurious} spurious exceedances (s^2 overflowed), {disagreement} disagreements with the documented rule. "
        f"Near-limit matrices: {limit_codes}. theta overflow: {theta_codes}. Subnormal witness: "
        f"{_code(results['witness'])}.")
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


def indefinite_candidates(limit=4000, keep=8):
    """P = R diag(1, 1e-17) R^T, rounded: keep candidates NumPy's eigvalsh calls positive definite.

    Returns up to ``keep`` that are exactly indefinite (negative exact determinant) and ``keep`` exactly
    positive definite controls, found by a seeded search; PLSR's own acceptance is observed separately.
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
                                                                _node("test_edge_case_exact_classes")))
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
         "Candidates P = R diag(1, 1e-17) R^T (seed 1041, up to 4000 draws): 8 that NumPy's eigvalsh calls positive "
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
        tolerance={"abs": 0.0, "rel": 0.0})]
    candidates, searched = indefinite_candidates()
    bridge_cases = [_verdict_case(f"e{i}", c["A"], c["P"], c["x"], time=c["time"]) for i, c in enumerate(cases)]
    for lam in (1e-3, 1e-2, 0.1, 1.0):
        bridge_cases.append({"id": f"solve:{lam}", "op": "solve", "A": [[-lam, 1.0], [0.0, -lam]], "time": "continuous"})
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
                tolerance={"abs": 0.0, "rel": 0.0}),
        finding("Every edge-case code is consistent with its exact class and exact distance from the resolution "
                "(resolved beyond two resolutions, never against the exact sign)", "numerical",
                {"cases": len(cases), "unexpected_codes": unexpected, "skew_P_identity": skew_codes, "jordan": jordan},
                {"provider": base, "checks": [_check("codes outside the set expected for the exact class",
                                                     unexpected, 0.0)]},
                tolerance={"abs": 0.0, "rel": 0.0}),
        finding("PLSR's Lyapunov solver and certificate constructor refuse a semidefinite Q and a singular P",
                "numerical", solves,
                {"provider": base, "checks": [_refusal("solve_lyapunov with Q = diag(1, 0)", "raises ValueError",
                                                       solves["solve:psdQ"]),
                                              _refusal("quadratic(diag(1, 0))", "raises ValueError",
                                                       solves["quadratic:psd"])]}),
    ]
    if indefinite_accepted:
        first = candidates[indefinite_accepted[0]]
        findings.append(finding(
            "quadratic() accepts binary64 P matrices that are exactly indefinite; the verdict then refuses or stays "
            "inconclusive and never certifies", "numerical",
            {"accepted_exactly_indefinite_observed": True, "certifying_verdicts": candidate_certified},
            {"provider": base, "checks": [
                _check("accepted candidates with negative exact determinant", len(indefinite_accepted), 1.0, "ge"),
                _check("certifying verdicts with an exactly indefinite P", candidate_certified, 0.0)]},
            counterexample={"statement": "A P accepted by PLSR's QuadraticCertificate is exactly positive definite",
                            "witness": {"P_hex": R.hexed(first["P"]), "exact_det_sign": first["exact_det_sign"],
                                        "weak_direction_code": _code(results[f"q{indefinite_accepted[0]}:weak"]),
                                        "e1_code": _code(results[f"q{indefinite_accepted[0]}:e1"])}}))
    findings += offline
    fields["numerical_result"] = (
        f"{len(cases)} edge cases: {violations} certifying verdicts on non-definite forms, {unexpected} codes outside "
        f"the class expectation. Skew with P = I: {skew_codes}. Jordan with P = I: {jordan}. Solver and constructor: "
        f"{solves}. Candidates: {len(accepted)}/{len(candidates)} accepted by quadratic(), "
        f"{len(indefinite_accepted)} of them exactly indefinite; their verdict codes {candidate_codes}.")
    fields["uncertainty"] = ("Exact classes are exact. Which candidates quadratic() accepts depends on the sign of "
                             "a rounded eigenvalue and may differ between LAPACK builds; the retained value is only "
                             "whether an exactly indefinite P was accepted.")
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
                                                                _node("test_conversion_scan")))
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
                tolerance={"abs": 0.0, "rel": 0.0}),
        finding("Converting box bounds in binary64 collapses neighbours and depends on the formula used", "numerical",
                {"draws": scan["draws"], "neighbour_collisions": scan["neighbour_collisions"],
                 "formula_disagreements": scan["formula_disagreements"]},
                {"generator": {"name": "conversion_scan", "seed": 1051},
                 "checks": [_check("neighbour collisions under x 1e-3", scan["neighbour_collisions"]["0.001"], 1.0, "ge"),
                            _check("b / 1000 versus b * 0.001 disagreements", scan["formula_disagreements"]["0.001"],
                                   1.0, "ge")]},
                tolerance={"abs": 0.0, "rel": 0.0}),
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
    admitted = {name: [s for s, code in codes.items() if code != "OUTSIDE_PARAMETER_BOX"] for name, codes in outside.items()}
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
                                                       "interior0_margin_ratio": dict(zip(UNIT_SYSTEMS, interior_ratios)),
                                                       "conversion_scan": scan}))
    findings = [
        finding("Interior and boundary stiffness samples give the same PLSR code in all five unit systems",
                "numerical", {"samples": len(interior), "systems": len(UNIT_SYSTEMS), "mismatches": interior_mismatch},
                {"provider": base, "checks": [_check("samples whose code differs between unit systems",
                                                     interior_mismatch, 0.0, kind="invariant")]},
                tolerance={"abs": 0.0, "rel": 0.0}),
        finding("check_vertices passes for the converted box in every unit system", "numerical", vertices,
                {"provider": base, "checks": [_check("unit systems whose vertex check fails",
                                                     sum(v is not True for v in vertices.values()), 0.0,
                                                     kind="invariant")]}),
        finding("Margin ratios are not invariant under non-uniform unit changes", "numerical",
                {"interior0_ratio_spread": ratio_spread},
                {"provider": base, "checks": [_check("max/min margin ratio across unit systems", ratio_spread, 10.0,
                                                     "ge", kind="invariant")]},
                tolerance={"abs": 0.0, "rel": 1e-6}),
    ]
    witness_codes = {key: _code(results[key]) for key in (f"{n}|{l}" for n in witness_bounds for l in ("SI", "x1e-3"))}
    for name in witness_bounds:
        si, converted = witness_codes[f"{name}|SI"], witness_codes[f"{name}|x1e-3"]
        if si != converted:
            statement = ("Converting the box and the sample with the same formula preserves box membership"
                         if name.startswith("collision") else
                         "Mathematically equal unit conversions give the same box decision")
            claim = ("A parameter just above the SI bound is admitted after multiplying bound and sample by 1e-3"
                     if name.startswith("collision") else
                     "A parameter exactly on the SI bound is refused when bound and sample are converted by the two "
                     "mathematically equal formulas k * 0.001 and k / 1000")
            findings.append(finding(
                claim, "numerical", {"SI": si, "x1e-3": converted},
                {"provider": base, "checks": [_check("SI and converted decisions differ (1 if so)", 1.0, 1.0, "ge",
                                                     kind="invariant")]},
                counterexample={"statement": statement, "witness": dict(scan["witnesses"][name], codes={
                    "SI": si, "x1e-3": converted})}))
    if any(admitted.values()):
        name = next(n for n, systems in admitted.items() if systems)
        findings.append(finding(
            "A stiffness just outside the SI box is admitted after an otherwise consistent unit conversion",
            "numerical", admitted,
            {"provider": base, "checks": [_check("unit systems admitting a just-outside sample",
                                                 sum(len(v) for v in admitted.values()), 1.0, "ge")]},
            counterexample={"statement": "Converting the box and the sample with the same formula preserves box "
                                         "membership", "witness": {"sample": name, "theta_hex": thetas[name].hex(),
                                                                   "codes": table[name]}}))
    if divided_refused:
        findings.append(finding(
            "A stiffness exactly on the bound is refused when the sample is converted as k / (1 / c) and the bound "
            "as k * c", "numerical", divided_refused,
            {"provider": base, "checks": [_check("bound samples refused under the second formula", len(divided_refused),
                                                 1.0, "ge")]},
            counterexample={"statement": "Mathematically equal unit conversions give the same box decision",
                            "witness": next(iter(divided_refused))}))
    if len(set(light.values())) > 1:
        findings.append(finding(
            "The light-damping plant's verdict depends on the unit system although its exact decrease form is "
            "negative definite in all of them", "numerical", light,
            {"provider": base, "checks": [
                _check("distinct codes across unit systems", len(set(light.values())), 2.0, "ge"),
                _check("unit systems whose exact decrease form is not negative definite",
                       sum(c != "negative_definite" for c in light_exact.values()), 0.0)]},
            counterexample={"statement": "The same physical plant in different units gets the same PLSR verdict",
                            "witness": {"codes": light, "margin_ratio": light_ratio}}))
    findings += offline
    fields["numerical_result"] = (
        f"Interior/bound samples: {interior_mismatch} code mismatches across {len(UNIT_SYSTEMS)} unit systems; "
        f"vertex check {vertices}. Just-outside samples admitted in: {admitted}. Bound samples converted as "
        f"k/(1/c): {divided}. Interior margin ratio spread {ratio_spread:.3g}x. Light damping codes {light} "
        f"(ratios { {k: float(f'{v:.3g}') for k, v in light_ratio.items()} }; exact classes {light_exact}). "
        f"Isolated box witnesses: {witness_codes}. Conversion scan: "
        f"{scan['neighbour_collisions']} neighbour collisions and {scan['formula_disagreements']} formula "
        f"disagreements in {scan['draws']} draws.")
    fields["uncertainty"] = ("Box decisions and conversions are exact IEEE arithmetic (platform-independent). "
                             "Margin ratios carry float64 rounding of the converted matrices (relative ~1e-15).")
    return _finish(fields, findings, PROVIDER_FILES, identity)


# T106 ------------------------------------------------------------------------

ROTATION = np.array([[0.0, 1.0], [-1.0, 0.0]])


def status_paths():
    """One-parameter paths through the decision order; each step is a declared (A, P, x, options) case."""
    I2 = np.eye(2)
    e1 = np.array([1.0, 0.0])
    paths = {
        "required_margin (margin 2)": [dict(A=-I2, P=I2, x=e1, required_margin=r) for r in (0.0, 1.0, 1.99, 2.0, 2.01, 3.0)],
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
                                                                _node("test_documented_decision_order")))
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
         "across [-0.2, 0.2]", "16 candidate P that NumPy calls positive definite but are exactly indefinite "
         "(seed 1041) evaluated along their weak direction", "Host-owned codes MODEL_MISMATCH, STALE_STATE, "
         "INVALID_SENSOR_DATA, CERTIFICATE_EXPIRED, RUNTIME_FAULT"],
        "PLSR verdict code per step; require_status and Verdict construction outcomes for host-owned codes; the "
        "runtime's published constants.",
        "The union of observed codes is the nine-code vocabulary; each observed code equals the CIW re-derivation "
        "of the documented order; every host-owned code is refused.",
        "Build each path, evaluate all steps in one PLSR subprocess, recompute each expected code in CIW from A, P "
        "and x, and tabulate transitions and coverage.",
        "T107: near-boundary spectra; add an upstream regression that pins one witness per code, including a "
        "CERTIFICATE_NOT_POSITIVE witness that does not depend on eigvalsh rounding.",
        ["a code unreachable", "a transition out of documented order", "a host-owned code accepted",
         "CERTIFICATE_NOT_POSITIVE only reachable through rounding"],
        ["CERTIFICATE_NOT_POSITIVE is reached only when a P accepted by quadratic() evaluates V < 0, which depends "
         "on eigvalsh rounding; min eig P <= 0 cannot follow a passed construction check.",
         "The CIW re-derivation shares the documented specification with the runtime, so it checks implementation "
         "against specification, not the specification itself."])
    candidates = indefinite_candidates(keep=16)[0][:16]
    paths = status_paths()
    cases, predictions = _path_cases(paths, candidates)
    predicted_codes = sorted(set(predictions.values()))
    authority = finding(
        "A CERTIFIED_WITH_MARGIN verdict (operationally_acceptable) authorizes actuation", "actuator_authority", None,
        {"derivation": "runtime-status-v1: operationally_acceptable is not an authorization; host statuses and "
                       "machine-safety functions are outside the evaluator"})
    offline = [finding("The documented decision order, re-derived in CIW, assigns all nine codes to the constructed "
                       "inputs", "numerical", {"codes": predicted_codes},
                       {"generator": {"name": "status_paths + indefinite_candidates", "seed": 1041},
                        "checks": [_check("distinct runtime codes predicted", len(predicted_codes), 9.0, "ge",
                                          kind="analytic")]}), authority]
    for code in R.HOST_OWNED:
        cases.append({"id": f"require:{code}", "op": "require_status", "code": code})
        cases.append({"id": f"verdict:{code}", "op": "host_verdict", "code": code})
    cases.append({"id": "constants", "op": "constants"})
    try:
        bridge = _bridge(ctx, cases)
    except _Unavailable as exc:
        fields["numerical_result"] = f"Provider-free prediction covers {len(predicted_codes)} codes: {predicted_codes}."
        fields["uncertainty"] = "Deterministic re-derivation on this platform."
        return _finish(fields, offline, PROVIDER_FILES, blocked=str(exc))
    identity, results = bridge["identity"], bridge["results"]
    base = provider_basis(identity)
    observed = {cid: _code(results[cid]) for cid in predictions}
    mismatches = {cid: {"observed": observed[cid], "predicted": predictions[cid]} for cid in predictions
                  if observed[cid] != predictions[cid]}
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
    findings = [
        finding("All nine runtime-status-v1 codes are reached by constructed declared inputs", "numerical",
                {"codes": reached},
                {"provider": base, "independent_check": _independent(
                    _check("steps whose code differs from the CIW re-derivation of the documented order",
                           len(mismatches), 0.0, kind="analytic"), identity),
                 "checks": [_check("distinct runtime codes observed", len(reached), 9.0, "ge", kind="invariant")]}),
        finding("Codes along each one-parameter path follow the documented decision order", "numerical", transitions,
                {"provider": base, "independent_check": _independent(
                    _check("path steps differing from the CIW re-derivation", sum(
                        1 for cid in mismatches if "#" in cid and not cid.startswith("indefinite")), 0.0,
                           kind="analytic"), identity)}),
        finding("The runtime refuses to emit the five host-owned status codes", "numerical", host,
                {"provider": base, "checks": [
                    _refusal(f"{way} for {code}", "raises ValueError", outcome[way])
                    for code, outcome in host.items() for way in ("require_status", "Verdict")]}),
        finding("Pinned runtime constants: resolution factor, numerical policy, status vocabulary", "provenance",
                constants, {"provider": base}),
    ] + offline
    fields["numerical_result"] = (
        f"Codes reached: {len(reached)}/9 ({reached}). {len(mismatches)} of {len(predictions)} steps differ from the "
        f"CIW re-derivation. Transitions: {transitions}. Host-owned codes refused: "
        f"{sum(v['require_status'] == 'raises ValueError' and v['Verdict'] == 'raises ValueError' for v in host.values())}/5. "
        f"Constants: resolution factor {constants.get('DECREASE_RESOLUTION_FACTOR')}, policy "
        f"{constants.get('NUMERICAL_POLICY_VERSION')}.")
    fields["uncertainty"] = ("Path codes are far from thresholds and platform-independent, except the "
                             "CERTIFICATE_NOT_POSITIVE witnesses, which depend on eigvalsh and dot-product rounding.")
    return _finish(fields, findings, PROVIDER_FILES, identity)
