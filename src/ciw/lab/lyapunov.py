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
synthetic illustrations (``lyapunov_research``); T114 evaluates its monitor
configuration with the runtime when it is bound. Outcomes that depend on the
LAPACK/BLAS build (subnormal codes, outside-window flips, certificates admitted
by rounding) are retained as artifacts, never as checked finding values.

Non-claims: every verdict concerns declared binary64 matrices and one declared
sample, not a physical plant. Exact arithmetic here decides what those numbers
imply, nothing more. No task establishes machine safety, actuator authority,
calibration or sensor performance; each such claim is retained as a
``not_established`` finding.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import combinations, product
import math

import numpy as np

from .. import __version__
from . import lyapunov_reference as R
from . import lyapunov_research as X
from . import svg
from .evidence import COMPUTATIONAL_DOMAINS, finding, holds, origin
from .lyapunov_provider import PLSR_ROLE, ProviderRefusal, manifest, producer, provider_basis, run_plsr, source_digest
from .registry import task

# The one cross-platform question T101 and T102 defer: codes that LAPACK/BLAS builds may decide differently.
PLATFORM_QUESTION = (
    "Deferred research question (cross-platform reproduction): run T101 and T102 against the pinned PLSR on Windows "
    "x86-64, on macOS arm64 and on Linux x86-64 with a LAPACK other than the retained OpenBLAS build (for example "
    "MKL), and check that the PLSR codes below the normal range (the subnormal witness's code included; "
    "resolution-floor.json) and the outside-window code flips and scaled-witness code (scaling-invariance.json) match "
    "the retained ones case for case, and that every finding value matches within its regression tolerance; these "
    "codes are retained as artifacts only because LAPACK/BLAS builds may decide them differently, and no second build "
    "has been compared.")


def capture_route(task_id: str, role: str, source: str, analysis: str, instrument: str, claims: str) -> str:
    """How acquired bytes could reach ``task_id``, and what the physical gate needs beyond them.

    An operator capture is retained and unauthenticated: computational findings may be computed from it, but a
    physical, calibration or sensor_performance label also needs a probe of the role's instrument on the analysing
    host (runner.CAPTURE_INSTRUMENTS) or a signed-capture trust anchor.
    """
    return (f"Route: {task_id} would read the {source} as an operator capture (ctx.capture('{role}'), bound with "
            f"ciw lab run {task_id} --capture {role}=PATH) and {analysis} as a computational finding; its {claims} "
            "need, besides an acquisition record (device, raw_sha256 of the captured bytes, acquired_at, "
            f"calibration), a probe of the {instrument} on the analysing host that succeeds in {task_id} "
            f"(runner.CAPTURE_INSTRUMENTS has no entry for {role}) or a signed-capture trust anchor, and the run is "
            f"retained with ciw lab hardware retain under lab/hardware/<run-id>. Neither the capture reader nor a "
            f"probe of the {instrument} exists, so they stay not_established even when such data exist.")


# Operator capture roles the hardware-gated next steps name (none has an instrument probe in the runner).
CAPTURE_ROLES = {"T113": "encoder-log", "T114": "servo-bench-log"}
# Each task's next step names its own open question (an upstream PLSR proposal to re-test, a new check, or a
# hardware-gated acquisition with its route), never the next queue task, which has already run.
NEXT_STEPS = {
    "T101": ("Deferred research question (upstream PLSR change): add a subnormal-aware term (an absolute "
             "n^2 * 2^-1074 floor) to PLSR's decrease resolution and re-run this resolution-floor study to check that "
             "the subnormal witness's resolution then covers the rounding error of forming its decrease matrix."),
    "T102": ("Deferred research question (upstream PLSR change): pre-scale M by a power of two into LAPACK's window "
             "before eigvalsh and re-run this scaling study to check that verdicts outside [2^-485, 2^485] become "
             "exactly scale-invariant, as they are inside the window."),
    "T103": ("Deferred research question (upstream PLSR change): decide the level gate by comparing exponents "
             "(log2 V = 2 e + log2 scaled_value) instead of forming s^2, and report NUMERICAL_OVERFLOW instead of "
             "raising when A(theta) overflows, then re-run this near-limit scan to check that no exceedance is missed "
             "or spurious and that no in-box theta raises."),
    "T104": ("Deferred research question (upstream PLSR change): check positive definiteness in QuadraticCertificate "
             "with an exact LDL^T (or a resolution-aware eigenvalue floor) instead of the eigvalsh sign, and re-run to "
             "check that the exactly indefinite P candidates that eigvalsh calls positive definite are refused."),
    "T105": ("Deferred research question: do outward-rounded parameter boxes (a guard band declared by the host) "
             "remove the boundary collisions between unit conventions, and should PLSR document that margin ratios "
             "are unit-dependent under non-uniform unit changes? Re-run the unit-scale comparison with outward-rounded "
             "boxes to find out."),
    "T106": ("Deferred research question (upstream PLSR change): pin one witness per runtime code and per allowed "
             "transition in PLSR's own tests, and decide whether a P(theta) that loses definiteness inside the box "
             "should return CERTIFICATE_NOT_POSITIVE rather than raise, which would make the eight transitions "
             "involving that code reachable."),
    "T107": ("Deferred research question: how much of the [-2, 0) res inconclusive band would a less conservative "
             "eigensolver term (p(n) = n instead of n^2) recover without losing soundness against the exact bins?"),
    "T108": ("Deferred research question: extend the required-margin monotonicity property test to affine plants with "
             "a parameter-dependent P(theta), which the finite family here does not contain (the general statement "
             "rests on the decision-order argument)."),
    "T109": ("Deferred research question: an exact-arithmetic Lyapunov solve (rational Bartels-Stewart) to decide the "
             "defective and clustered cases that PLSR's residual gate refuses, and an upstream proposal that "
             "solve_lyapunov report the residual gate separately from definiteness."),
    "T110": ("Deferred research question: a sampled-data check that a continuous-time certificate is not silently "
             "reused for the one-step map exp(A h): certify exp(A h) for declared sampling periods h and compare with "
             "the continuous verdict (no sampling period is modelled here)."),
    "T111": ("Deferred research question: compare the scalar-quadratic and matrix-eigenvalue routes on near-marginal "
             "plants (stability margins below 0.05, excluded here) against exact rational references, where the two "
             "routes are most likely to disagree."),
    "T112": ("Deferred research question: specify a resolution-aware float64 ISS inequality (open question 1 in the "
             "retained spec) before any runtime integration (acquiring an evidenced disturbance bound is "
             "hardware-gated)."),
    "T113": ("Deferred research question (hardware-gated): bind the residual adapter to acquired encoder data and "
             "check the coverage of its three-standard-error interval there, which synthetic EKF data cannot "
             "establish (a sensor_performance claim). "
             + capture_route("T113", CAPTURE_ROLES["T113"], "encoder log", "run the residual adapter on it", "encoder",
                             "sensor_performance and calibration findings")),
    "T114": ("Deferred research question (hardware-gated): identify J, friction and delay on the bench, replace the "
             "placeholder interval, re-run this check with an affine over-approximation suitable for PLSR's box "
             "semantics, then run T113's adapter on acquired encoder data. "
             + capture_route("T114", CAPTURE_ROLES["T114"], "servo bench log", "identify the J, friction and delay "
                             "interval from it", "bench instrument", "physical and calibration findings")),
}

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
UNCONDITIONAL_TEST = f"{TESTS}::test_checks_are_unconditional_and_observed_values_are_computed"
NEXT_STEP_TEST = f"{TESTS}::test_next_steps_name_forward_work"
PLATFORM_TEST = f"{TESTS}::test_t101_t102_defer_cross_platform_reproduction_as_one_question"
WITNESS_ONCE_TEST = f"{TESTS}::test_t101_t103_retain_the_subnormal_witness_once"


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


def _solver_agreement(check, identity, checker):
    """Basis entry for agreement with a Lyapunov solver: independent only when the solver is outside CIW.

    The CIW Kronecker fallback solves the same Kronecker system with numpy.linalg.solve as PLSR's
    solve_lyapunov, so a shared algorithm could hide a common-mode error; its agreement is an ordinary check.
    """
    if origin(checker["implementation"]) == "ciw":
        return {"checks": [check]}
    return {"independent_check": _independent(check, identity, checker)}


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


def _witness_formed(result):
    """PLSR's own binary64 decrease matrix of the witness in units of 2^-1074 (None without a sample)."""
    sample = result.get("sample") if result.get("ok") else None
    if not sample or "decrease_matrix" not in sample:
        return None
    return (np.array(sample["decrease_matrix"], dtype=float) / R.TINY).tolist()


def _witness_measure(result):
    """PLSR's formed witness matrix (units of 2^-1074), its largest deviation from the exact form and the resolution.

    Without a returned sample the formed matrix is None, the deviation 0 and the resolution None.
    """
    formed = _witness_formed(result)
    if formed is None:
        return None, 0.0, None
    exact_units = np.array(_witness_exact()["exact_form_units"])
    return formed, float(np.max(np.abs(np.array(formed) - exact_units))), float(result.get("resolution", 0.0))


def _witness_finding(result, identity, statement):
    """The subnormal witness: PLSR's formed decrease matrix and resolution against the exact declared form.

    Every quantity checked here is IEEE-deterministic wherever subnormals are kept: the products with 1 and 0
    are exact and the symmetrising halving rounds 0.5 * 5 * 2^-1074 to 2 * 2^-1074 (ties to even), so the
    binary64 form is negative definite while the declared form is exactly indefinite, and the resolution
    underflows to zero below that rounding error. The code PLSR then returns rests on the sign LAPACK gives a
    subnormal eigenvalue, which depends on the build; it is retained in the task's artifact, not in a finding.
    """
    exact = _witness_exact()
    A, P, x = _subnormal_witness()
    formed, error, _ = _witness_measure(result)
    # Sylvester: a 2x2 symmetric form is negative definite iff its (1, 1) entry is negative and det > 0.
    formed_det = 0.0 if formed is None or formed[0][0] >= 0.0 else float(
        formed[0][0] * formed[1][1] - formed[0][1] * formed[1][0])
    resolution = float(result.get("resolution", 0.0)) if formed is not None else 1.0
    checks = [_check("negated exact determinant of the declared 2x2 decrease form, units (2^-1074)^2 "
                     "(at least 1 means indefinite)", -exact["exact_det_units2"], 1.0, "ge"),
              _check("determinant of PLSR's formed binary64 decrease matrix, units (2^-1074)^2, or 0 unless its "
                     "(1, 1) entry is negative (at least 1 means negative definite)", formed_det, 1.0, "ge",
                     kind="invariant"),
              _check("largest entry of PLSR's formed matrix minus the exact form, units of 2^-1074 (a lower bound "
                     "on the formation error the resolution must cover)", error, 1.0, "ge"),
              _check("PLSR resolution of the witness (underflows to zero)", resolution, 0.0, kind="invariant")]
    return finding(
        WITNESS_CLAIM, "numerical",
        {"exact_class": exact["exact_class"], "exact_det_units2": exact["exact_det_units2"],
         "plsr_form_units": formed, "formation_error_units": error, "resolution": resolution},
        {"provider": provider_basis(identity), "checks": checks}, tolerance=EXACT_TOL,
        counterexample={"statement": statement,
                        "witness": {"A_hex": R.hexed(A), "P_hex": R.hexed(P), "x": x.tolist(),
                                    "exact_form_units_of_2^-1074": exact["exact_form_units"],
                                    "plsr_form_units_of_2^-1074": formed, "resolution": resolution}},
        uncertainty=_roundoff(0.0, "IEEE-deterministic where subnormals are kept (no flush to zero): exact "
                                   "products with 1 and 0 and one halving rounded to even; PLSR's code on the "
                                   "witness depends on LAPACK's subnormal eigenvalue and is retained as an "
                                   "artifact"))


WITNESS_CLAIM = ("PLSR's resolution of a subnormal plant is zero while forming its decrease matrix rounds an exactly "
                 "indefinite declared form to a negative definite one")


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
    cases.append(_verdict_case("witness", A, P, x, full=True))
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
                                                                _node("test_t101_threshold_is_analytic"),
                                                                WITNESS_ONCE_TEST, PLATFORM_TEST,
                                                                UNCONDITIONAL_TEST, PARTIAL_TEST, NEXT_STEP_TEST))
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
        "at every scale; NUMERICAL_OVERFLOW exactly where the decrease form overflows; the resolution covers the "
        "rounding error of forming M at every scale, so exactly-indefinite declared forms cannot be certified.",
        "Evaluate every case with PLSR, recompute the documented resolution and the exact rational class of the "
        "declared form in CIW, compare codes with the analytic threshold and the exact class, and locate where the "
        "prediction stops holding.",
        NEXT_STEPS["T101"],
        ["resolution differs from the documented formula", "normalised resolution drifts with scale",
         "code differs from the analytic threshold eps*", "code differs from the unit-scale code",
         "NUMERICAL_OVERFLOW before the decrease form overflows", "no NUMERICAL_OVERFLOW once it overflows",
         "resolution below the rounding error of PLSR's formed decrease matrix (subnormal witness, exact rational "
         "form)"],
        ["The exact class concerns the declared binary64 numbers; they are synthetic, not measured plants.",
         "The resolution comparison is against CIW's transcription of the documented formula: a same-"
         "specification check, not an independent one.",
         PLATFORM_QUESTION])
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
    witness = results["witness"]
    # The witness's code rests on LAPACK's sign for a subnormal eigenvalue: retained here, not in a finding.
    ctx.artifact_json("resolution-floor.json", R.jsonable({"epsilon_star": EPSILON_STAR, "rows": rows,
                                                            "f1_departures_outside_normal_range": f1_below,
                                                            "f2_false_certificates": false_certificates,
                                                            "witness": {"code": _code(witness),
                                                                        "details": witness.get("details"),
                                                                        "margin": witness.get("margin"),
                                                                        "resolution": witness.get("resolution"),
                                                                        "unit_scale_code":
                                                                            _code(results["witness-unit"])}}))
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
        _witness_finding(witness, identity, "The float64 resolution floor bounds the rounding error of the decrease "
                                            "form at every matrix scale"),
    ] + offline
    witness_record = findings[3]
    witness_value = witness_record["value"]
    fields["numerical_result"] = (
        f"Resolution matches the documented formula (max relative difference {max(rel_diffs):.2e} over "
        f"{len(rel_diffs)} cases). In {f1_normal} normal-range threshold cases {f1_analytic} codes differ from the "
        f"analytic threshold eps* = {EPSILON_STAR:.6e} and {f1_unit} from the unit-scale code. Outside the normal "
        f"range {len(f1_below)} threshold cases depart from the unit-scale code and F2 has "
        f"{len(false_certificates)} certifying verdicts whose exact form is not negative definite (both retained "
        f"in the artifact; subnormal-range codes depend on the LAPACK build). The subnormal witness's resolution is "
        f"{witness_value['resolution']:.3g} while PLSR's formed decrease matrix is off the exact form by "
        f"{witness_value['formation_error_units']:g} x 2^-1074 in its largest entry (unit-scale twin: "
        f"{_code(results['witness-unit'])}; the witness's code is retained in the artifact). First overflow "
        f"exponents: { {s: e['k'] for s, e in sorted(overflow_first.items())} }; {early_overflow} early overflow "
        f"reports in {overflow_rows} F3 rows. Conclusion: "
        + ("the floor is exactly the documented bound and the threshold stays at eps* while every quantity is a "
           "normal number" if f1_analytic == 0 and max(normalised) == 0.0 else
           "the normal-range threshold departs from eps*")
        + ("; below that range the floor underflows beneath the rounding error of forming the decrease matrix, so "
           "an exactly indefinite form reaches the eigensolver as a negative definite one with nothing to absorb "
           "the error" if witness_record["evidence_status"] != "not_established" else
           "; the subnormal witness did not show the floor falling below the formation error") + ".")
    fields["uncertainty"] = ("Resolution comparison is deterministic float arithmetic (relative 1e-12 allowed). "
                             "Subnormal-range codes depend on LAPACK's handling of tiny matrices and may differ "
                             "between BLAS builds, so they are retained in the artifact only; the witness's formed "
                             "matrix and zero resolution are IEEE-deterministic and its exact class is exact "
                             "rational arithmetic.")
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
                                                                _node("test_discrete_razor_edge_sits_at_the_threshold"),
                                                                PLATFORM_TEST,
                                                                UNCONDITIONAL_TEST, PARTIAL_TEST, NEXT_STEP_TEST))
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
        NEXT_STEPS["T102"],
        ["code flip inside the window", "margin ratio changes inside the window", "x scaling alters the sample",
         "discrete near-threshold code flip under (P, x) scaling", "code moved against the exact class outside the "
         "window", "resolution of the scaled subnormal witness not zero"],
        ["The LAPACK window [2^-485, 2^485] is taken from reference dsyevd (RMIN, RMAX); other LAPACK builds may "
         "rescale differently.", PLATFORM_QUESTION])
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
    witness_unit, witness_sub = results["w:unit"], results["w:sub"]
    # Outside-window flips and the scaled witness's code depend on the LAPACK/BLAS build: artifact only.
    ctx.artifact_json("scaling-invariance.json", R.jsonable({"tally": tally, "flips": flips,
                                                              "unsound_base_verdicts": base_unsound,
                                                              "eigvalsh_outside_bitwise_equal": sum(outside_eig),
                                                              "eigvalsh_outside_total": len(outside_eig),
                                                              "witness": {"unit_code": _code(witness_unit),
                                                                          "scaled_code": _code(witness_sub),
                                                                          "scaled_details":
                                                                              witness_sub.get("details")}}))
    base = provider_basis(identity)
    scaled_resolution = float(witness_sub.get("resolution", 0.0)) if witness_sub.get("sample") else 1.0
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
                {"provider": base, "checks": [
                    _check("scaled verdicts with a certifying code on a form whose exact class is not negative "
                           "definite", tally["outside"]["unsound"], 0.0)]}, tolerance=EXACT_TOL,
                uncertainty=_platform(0.0, "which near-threshold codes flip depends on LAPACK/BLAS rounding and is "
                                           "retained in the artifact; the soundness count does not")),
        finding("Scaling the witness by 2^-1074 drives its resolution to zero while its unit-scale code is "
                "DECREASE_NOT_DEFINITE; the scaled code is retained as an artifact", "numerical",
                {"unit_code": _code(witness_unit), "scaled_resolution": scaled_resolution},
                {"provider": base, "checks": [
                    _check("unit-scale witness verdicts whose code is not DECREASE_NOT_DEFINITE",
                           int(_code(witness_unit) != "DECREASE_NOT_DEFINITE"), 0.0, kind="invariant"),
                    _check("PLSR resolution of the witness scaled by 2^-1074 (underflows to zero)",
                           scaled_resolution, 0.0, kind="invariant")]}, tolerance=EXACT_TOL,
                uncertainty=_roundoff(0.0, "the unit-scale form's top eigenvalue lies about 3.6e12 resolutions "
                                           "above zero and the scaled resolution underflows exactly; the scaled "
                                           "code depends on LAPACK's subnormal handling and is not part of the "
                                           "finding")),
        finding("No unscaled PLSR verdict certifies a form whose exact class is not negative definite", "numerical",
                {"verdicts": len(exact_classes), "unsound": base_unsound},
                {"provider": base, "independent_check": _independent(
                    _check("exact rational class of each declared continuous and discrete form", base_unsound, 0.0),
                    identity)}, tolerance=EXACT_TOL, uncertainty=EXACT),
    ] + offline
    inside_clause = ("verdicts are exactly power-of-two invariant inside LAPACK's unscaled window and the normal "
                     "range" if tally["inside"]["code_flips"] == tally["inside"]["ratio_changes"] == 0
                     and tally["discrete"]["code_flips"] == 0 else "power-of-two invariance fails inside the window")
    unsound_total = sum(t["unsound"] for t in tally.values()) + base_unsound
    fields["numerical_result"] = (
        f"Inside the window: {tally['inside']['evaluations']} scaled evaluations, {tally['inside']['code_flips']} code "
        f"flips, {tally['inside']['ratio_changes']} ratio changes. Outside the window: "
        f"{tally['outside']['evaluations']} evaluations, {tally['outside']['code_flips']} flips, "
        f"{tally['outside']['ratio_changes']} ratio changes, {tally['outside']['unsound']} certificates contradicting "
        f"the exact class. Discrete (P, x) scaling: {tally['discrete']['code_flips']} flips in "
        f"{tally['discrete']['evaluations']} (half of the cases within about one resolution of the threshold). "
        f"Subnormal witness: unit-scale code {_code(witness_unit)}, resolution {scaled_resolution:.3g} after scaling "
        f"by 2^-1074 (its scaled code is retained in the artifact). eigvalsh bitwise homogeneous in "
        f"{sum(inside_eig)}/{len(inside_eig)} inside and {sum(outside_eig)}/{len(outside_eig)} outside scalings. "
        f"Conclusion: {inside_clause}; outside it codes may move with the LAPACK build (the flips of this run are "
        "retained in the artifact)"
        + (", never against the exact class" if unsound_total == 0 else
           f", and {unsound_total} certificates contradict the exact class")
        + "; below the normal range the witness's resolution underflows to zero, so its scaled code is left to "
        "LAPACK's subnormal eigenvalue.")
    fields["uncertainty"] = ("Inside-window results are exact (bitwise). Outside-window flip counts and the scaled "
                             "witness's code depend on the LAPACK/BLAS build and on how close each case sits to the "
                             "threshold; they are retained in the artifact, and only the soundness counts are checked "
                             "finding values.")
    return _finish(fields, findings, PROVIDER_FILES, identity)


# T103 ------------------------------------------------------------------------

DBL_MAX = float(np.finfo(float).max)
T103_A = np.array([[-1.0, 2.0], [0.0, -3.0]])
T103_STATES = ((1.0, 1.0), (R.TINY, 0.0), (R.TINY, R.TINY), (-R.TINY, 3 * R.TINY), (2.0 ** -1022, 2.0 ** -1022),
               (1e-300, -1e-300), (DBL_MAX, -DBL_MAX), (DBL_MAX, R.TINY), (2.0 ** 1000, 3 * 2.0 ** 990), (0.0, 0.0))
# Single-component states at the underflow and overflow edges of the reported V = s^2 * scaled_value
# (s = 2^e is the state scale): s^2 underflows below e = -537 and overflows from e = 512.
T103_EDGE_STATES = tuple((m * 2.0 ** e, 0.0) for e in (-540, -539, -538, -537, -536, -530, 510, 511, 512)
                         for m in (1.0, 1.5, 1.9))
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


def reported_values(states, results, prefix):
    """PLSR's reported V, x^T M x and value_out_of_range per state against exact rationals (T103_A, P = I)."""
    form, identity = R.exact_form(T103_A, np.eye(2)), R.fractions(np.eye(2))
    rows = []
    for i, state in enumerate(states):
        result = results[f"{prefix}{i}"]
        sample = result.get("sample") or {}
        exact_v, exact_d = R.exact_quadratic(state, identity), R.exact_quadratic(state, form)
        row = {"x_hex": [float(v).hex() for v in state], "code": _code(result),
               "exact_representable": R.representable(exact_v) and R.representable(exact_d),
               "flag": sample.get("value_out_of_range"), "value": sample.get("value"),
               "decrease": sample.get("decrease"), "error_units": None}
        if row["exact_representable"] and row["flag"] is False:
            # |reported - exact| in units of one rounding: 4u relative plus one subnormal spacing.
            row["error_units"] = max(float(abs(Fraction(sample[key]) - exact)
                                           / (4 * Fraction(R.U) * abs(exact) + Fraction(R.TINY)))
                                     for key, exact in (("value", exact_v), ("decrease", exact_d)))
        rows.append(row)
    return rows


def _level_counts(rows, key):
    missed = sum(r["exact_exceeded"] and not r[key] for r in rows)
    spurious = sum(r[key] and not r["exact_exceeded"] for r in rows)
    return missed, spurious


@task("T103", changed_files=PROVIDER_FILES, regression_tests=(_node("test_t103_overflow_underflow"),
                                                                _node("test_level_gate_prediction"),
                                                                _node("test_representability_reference"),
                                                                WITNESS_ONCE_TEST,
                                                                UNCONDITIONAL_TEST, PARTIAL_TEST, NEXT_STEP_TEST))
def overflow_underflow(ctx):
    fields = _fields(
        "States anywhere in binary64 are classified with the code unchanged through PLSR's power-of-two state "
        "scaling (components more than about 2^1022 below the largest one become subnormal in the scaled state "
        "and vanish beyond about 2^1075, which a code decided relative to |x|^2 does not see); the reported "
        "unscaled V and x^T M x equal the exact values to within rounding or are flagged value_out_of_range, and "
        "the flag is set exactly when the exact value is not representable; matrices whose arithmetic leaves "
        "binary64 return NUMERICAL_OVERFLOW or an input refusal; no level-set exceedance is missed near the limits. "
        "The resolution floor of a subnormal plant is T101's finding, which T103 cites and does not repeat.",
        "V(x) = x^T P x and x^T M x are homogeneous of degree two in x, so PLSR evaluates at x / 2^e with the "
        "unit state in [1, 2), and reports V = s^2 * scaled_value (s = 2^e), flagging value_out_of_range when that "
        "product is infinite or zero while scaled_value is not. The level gate decides V > level as scaled_value > "
        "level / s^2 and treats an infinite s^2 as 'exceeded' and a zero s^2 as 'not exceeded'. Exact truth: "
        "V = 2^(p + 2e) for P = 2^p I and x = 2^e e1; V and x^T M x of every declared state in rationals.",
        ["A = [[-1, 2], [0, -3]], P = I with ten states from 2^-1074 to 1.797e308 (mixed scales included) and "
         "non-finite states", "The same plant at 27 single-component states m 2^e (m in {1, 1.5, 1.9}, e in "
         "{-540..-536, -530, 510, 511, 512}) where s^2 underflows or overflows",
         "Level scan: A = -I, P = 2^p I (p in -1060..1000), x = 2^e e1 (e in -1074..1023), "
         "level = 2^(p + 2e -+ 1)", "Near-limit matrices: A = -2^1000 I with P = 2^30 I; A = -2^511 I or -2^512 I "
         "with P = 2^511 I; P = 2^1022 I with x = (1.5, 1.5); affine A(theta) = -I + theta c I, theta = +-1e308",
         "The subnormal witness A = [[-2, 5], [0, -3]] 2^-1074, P = I (re-evaluated for the artifact only; its "
         "finding is retained by T101)"],
        "PLSR code (or raised input error) per case; PLSR's reported V, x^T M x and value_out_of_range against "
        "exact rationals; exact exponent arithmetic for V against the level; PLSR's formed decrease matrix and "
        "resolution of the witness against its exact rational form.",
        "State scaling never changes the code; reported values are exact to rounding or flagged, and flagged "
        "exactly when not representable; levels are decided exactly; overflow returns NUMERICAL_OVERFLOW.",
        "One PLSR subprocess evaluates every case; CIW predicts the level decisions from the documented rule and "
        "from exact exponents and recomputes where the decrease form overflows.",
        NEXT_STEPS["T103"],
        ["state scaling changes the code", "non-finite state accepted", "reported V or x^T M x wrong without a flag",
         "value_out_of_range missing for an unrepresentable value", "value_out_of_range set for a representable "
         "value", "level gate misses an exceedance", "level gate reports a spurious exceedance",
         "overflow not reported as NUMERICAL_OVERFLOW", "overflow while forming A(theta) raises"],
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
    cases += [_verdict_case(f"edge{i}", T103_A, P_ok, state) for i, state in enumerate(T103_EDGE_STATES)]
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
    cases.append(_verdict_case("witness", A, P, x, full=True))
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
    reported = (reported_values(T103_STATES, results, "s") + reported_values(T103_EDGE_STATES, results, "edge"))
    missed_flags = [r for r in reported if not r["exact_representable"] and r["flag"] is not True]
    extra_flags = [r for r in reported if r["exact_representable"] and r["flag"] is True]
    unflagged_errors = [r["error_units"] for r in reported if r["error_units"] is not None]
    limit_codes = {name: _code(results[f"M:{name}"]) for name in limits}
    predicted_limits = {name: R.documented_code(A, P, x)["code"] for name, (A, P, x) in limits.items()}
    theta_codes = {cid: _code(results[cid]) for cid in ("theta:+1e308,c=1", "theta:-1e308,c=1", "theta:+1e308,c=2")}
    ctx.artifact_json("near-limits.json", R.jsonable({"states": state_codes, "reported_values": reported,
                                                       "level_scan": rows,
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
    findings.append(finding(
        "PLSR's reported V and x^T M x equal the exact values to within one rounding unless value_out_of_range is "
        "set, and the flag is set for every state whose exact V or x^T M x is not representable", "numerical",
        {"states": len(reported), "unrepresentable": sum(not r["exact_representable"] for r in reported),
         "flagged": sum(r["flag"] is True for r in reported), "missed_flags": len(missed_flags),
         "unflagged_compared": len(unflagged_errors)},
        {"provider": base,
         "independent_check": _independent(
             _check("states whose exact V or x^T M x (rationals) is not representable in binary64 but "
                    "value_out_of_range is not set", len(missed_flags), 0.0), identity),
         "checks": [_check("max over unflagged states of |reported - exact| / (4u |exact| + 2^-1074) for V and "
                           "x^T M x", max(unflagged_errors), 1.0, "le")]},
        tolerance=EXACT_TOL,
        uncertainty=_roundoff(max(unflagged_errors) * 4.0 * R.U,
                              "largest relative error of an unflagged reported value (single-component states: "
                              "IEEE-deterministic)")))
    findings.append(finding(
        "PLSR sets value_out_of_range and reports V = 0 for states whose exact V and x^T M x are representable "
        "binary64 subnormals, because s^2 underflows before the product is formed", "numerical",
        {"representable_but_flagged": len(extra_flags), "states": len(reported)},
        {"provider": base, "checks": [
            _check("states with representable exact V and x^T M x (rationals) and value_out_of_range set",
                   len(extra_flags), 1.0, "ge")]},
        tolerance=EXACT_TOL,
        counterexample={"statement": "value_out_of_range is set exactly when V or x^T M x is not representable in "
                                     "binary64", "witness": extra_flags[0]} if extra_flags else None,
        uncertainty=_roundoff(0.0, "single-component states: s^2 = 2^(2e) and the reported product are "
                                   "IEEE-deterministic; representability of the exact rationals is exact")))
    # C12: the subnormal-witness finding is retained once, by T101; T103 re-evaluates the witness for its artifact.
    _, witness_error, witness_resolution = _witness_measure(results["witness"])
    findings += offline
    refutations = []
    if missed or spurious:
        refutations.append("level-gate errors (" + ", ".join(
            part for part, count in (("missed", missed), ("spurious", spurious)) if count) + ")")
    if raised.startswith("raises"):
        refutations.append("an input error raised for an in-box theta")
    if extra_flags:
        refutations.append("value_out_of_range set for representable subnormal values")
    behaved = state_mismatch == 0 and all(limit_codes[k] == predicted_limits[k] for k in limits)
    fields["numerical_result"] = (
        f"{len(state_codes)} states from 2^-1074 to 1.797e308: {state_mismatch} code changes (all {reference_code}). "
        f"Reported values at {len(reported)} states: {sum(not r['exact_representable'] for r in reported)} exact V "
        f"or x^T M x not representable, {len(missed_flags)} of them without value_out_of_range; "
        f"{len(extra_flags)} representable subnormal values flagged and reported as 0 (s^2 underflowed); unflagged "
        f"values within {max(unflagged_errors):.3g} roundings of the exact ones. "
        f"Level scan ({len(rows)} cases): {missed} missed exceedances (certified with V > level, s^2 underflowed), "
        f"{spurious} spurious exceedances (s^2 overflowed), {disagreement} disagreements with the documented rule. "
        f"Near-limit matrices: {limit_codes}. theta overflow: {theta_codes}. Subnormal witness, re-evaluated for the "
        f"artifact only (T101 retains its finding '{WITNESS_CLAIM}'): resolution "
        + ("not returned" if witness_resolution is None else f"{witness_resolution:.3g}")
        + f", formed decrease matrix off the exact form by {witness_error:g} x 2^-1074 in its largest entry (its "
        "code is retained in the artifact). Conclusion: "
        + ("state scaling and matrix overflow behave as documented" if behaved else
           "state scaling or matrix overflow departs from the documented order")
        + ("; the hypothesis is refuted by " + ", by ".join(refutations) if refutations else
           "; no refutation of the hypothesis was observed") + ".")
    fields["uncertainty"] = ("The level scan, the reported-value states and the overflow cases involve exact powers "
                             "of two or single-component states, so they are platform-independent; the witness's "
                             "formed matrix and zero resolution are IEEE-deterministic, while its code depends on "
                             "LAPACK's subnormal handling and is retained in the artifact only.")
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
                                                                _node("test_edge_case_exact_classes"),
                                                                UNCONDITIONAL_TEST, PARTIAL_TEST, NEXT_STEP_TEST))
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
        NEXT_STEPS["T104"],
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
    indefinite_codes = _counts(_code(results[f"q{i}:{where}"]) for i in indefinite_accepted
                               for where in ("weak", "e1"))
    # Which candidates quadratic() accepts, and their codes, depend on eigvalsh rounding: artifact only.
    ctx.artifact_json("edge-cases.json", R.jsonable({
        "rows": rows, "solves": solves,
        "indefinite_candidates": {
            "searched": searched, "accepted": len(accepted), "accepted_exactly_indefinite": len(indefinite_accepted),
            "codes": candidate_codes, "codes_exactly_indefinite": indefinite_codes,
            "accepted_indefinite_witnesses": [
                {"P_hex": R.hexed(candidates[i]["P"]), "exact_det_sign": candidates[i]["exact_det_sign"],
                 "weak_direction_code": _code(results[f"q{i}:weak"]), "e1_code": _code(results[f"q{i}:e1"])}
                for i in indefinite_accepted]}}))
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
    findings.append(finding(
        "No verdict certifies with a P that quadratic() accepts although it is exactly indefinite", "numerical",
        {"candidates": len(candidates), "certifying_verdicts": candidate_certified},
        {"provider": base, "checks": [_check("certifying verdicts with an accepted, exactly indefinite P",
                                             candidate_certified, 0.0)]}, tolerance=EXACT_TOL,
        uncertainty=_platform(0.0, "which candidates quadratic() accepts depends on eigvalsh rounding and is "
                                   "retained in the artifact; the certification count does not")))
    findings += offline
    fields["numerical_result"] = (
        f"{len(cases)} edge cases: {violations} certifying verdicts on non-definite forms, {unexpected} codes outside "
        f"the class expectation. Skew with P = I: {skew_codes}. Jordan with P = I: {jordan}. Solver and constructor: "
        f"{solves}. Candidates ({searched} draws searched): {len(accepted)}/{len(candidates)} accepted by "
        f"quadratic(), {len(indefinite_accepted)} of them exactly indefinite; verdicts on accepted candidates: "
        f"{sum(v for k, v in candidate_codes.items() if k in R.CERTIFYING)} certifying of "
        f"{sum(candidate_codes.values())} (codes and the accepted indefinite P are retained in the artifact). "
        "Conclusion: "
        + ("no semidefinite, skew or indefinite edge case was certified" if violations == 0 else
           f"{violations} non-definite edge cases were certified")
        + ("; the constructor's floating-point eigenvalue-sign test is not an exact definiteness test, and no "
           "verdict certified with an exactly indefinite P it admitted" if candidate_certified == 0 else
           f"; {candidate_certified} verdicts certified with an exactly indefinite P") + ".")
    fields["uncertainty"] = ("Exact classes are exact. Which candidates quadratic() accepts depends on the sign of "
                             "a rounded eigenvalue and may differ between LAPACK builds; the accepted candidates and "
                             "their codes are retained in the artifact, and only the certification count is a "
                             "checked value.")
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


# Box bounds and their binary64 neighbours just outside, converted by both formulas.
T105_BOUNDS = ("k=8 (bound)", "k=12 (bound)")
T105_NEIGHBOURS = ("k just below 8", "k just above 12")
T105_EDGES = T105_BOUNDS + T105_NEIGHBOURS


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
                                                                _node("test_conversion_scan"), PARTIAL_TEST,
                                                                NEXT_STEP_TEST))
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
         "(PCG64 seed 1052), states from a normal draw; bounds and neighbours converted both as k * c and as "
         "k / (1 / c) against a box converted as bound * c", "Light-damping twin: k = 10, c = 1e-8, SI margin 5e-9",
         "Conversion scan: 2000 bounds 10^U(-3, 6) (seed 1051) with factors 1e-3 and 1e-6"],
        "PLSR code, margin ratio and check_vertices result per unit system; IEEE conversion outcomes.",
        "Identical codes across unit systems for every sample; identical box decisions for bounds and "
        "neighbours; check_vertices passes in every unit system.",
        "Convert the SI declaration into each unit system the way a host would (float arithmetic), evaluate all "
        "samples with PLSR, compare codes per sample, then search the conversion arithmetic for counterexamples.",
        NEXT_STEPS["T105"],
        ["code differs across unit systems", "box decision differs at a bound", "just-outside sample admitted",
         "bound conversion formulas disagree", "vertex check fails in some units",
         "near-threshold verdict depends on units"],
        ["The box is converted by one host convention (multiply by the factor); samples are converted by it and by "
         "division by the reciprocal factor, and other conventions change which boundary samples collide.",
         "The stiffness box and damping are illustrative values, not identified ones."])
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
        for name in T105_EDGES:
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
               for name in T105_EDGES}
    divided_refused = {f"{system}|{name}": divided[f"{system}|{name}"] for system in UNIT_SYSTEMS
                       for name in T105_BOUNDS if divided[f"{system}|{name}"] == "OUTSIDE_PARAMETER_BOX"}
    divided_admitted = {f"{system}|{name}": divided[f"{system}|{name}"] for system in UNIT_SYSTEMS
                        for name in T105_NEIGHBOURS if divided[f"{system}|{name}"] != "OUTSIDE_PARAMETER_BOX"}
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
        finding("The declared box bounds 8 and 12 N/m keep their SI box decision in all five unit systems under "
                "both conversion formulas, and their binary64 neighbours keep it when bound and sample are both "
                "multiplied", "numerical",
                {"just_outside_admitted": admitted, "bound_refused_by_second_formula": divided_refused},
                {"provider": base, "checks": [
                    _check("just-outside samples admitted after conversion by multiplication",
                           sum(len(v) for v in admitted.values()), 0.0, kind="invariant"),
                    _check("bound samples refused when converted as k / (1 / c)", len(divided_refused), 0.0,
                           kind="invariant")]},
                tolerance=EXACT_TOL, uncertainty=_roundoff(0.0, "scalar IEEE products and quotients; identical on "
                                                                "every conforming platform")),
        finding("A binary64 neighbour just outside the declared stiffness box is admitted in some unit system when "
                "the sample is converted as k / (1 / c) and the bound as k * c", "numerical",
                {"neighbours_admitted_by_second_formula": divided_admitted,
                 "evaluations": len(UNIT_SYSTEMS) * len(T105_NEIGHBOURS)},
                {"provider": base, "checks": [
                    _check("just-outside samples converted as k / (1 / c) and not refused",
                           len(divided_admitted), 1.0, "ge", kind="invariant")]},
                tolerance=EXACT_TOL,
                counterexample={"statement": "The declared box bounds and their binary64 neighbours keep their SI box "
                                             "decision under either conversion formula",
                                "witness": {"cases": sorted(divided_admitted), "k_hex": {
                                    name: thetas[name].hex() for name in T105_NEIGHBOURS}}}
                if divided_admitted else None,
                uncertainty=_roundoff(0.0, "scalar IEEE quotients and products; identical on every conforming "
                                           "platform")),
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
    boundary_clause = (("the declared MSD bounds keep their box decision under both conversion formulas and their "
                        "neighbours under multiplication" if not any(admitted.values()) and not divided_refused else
                        "the declared MSD bounds or their neighbours change box decision after conversion")
                       + (", but mixing the two formulas admits a just-outside neighbour" if divided_admitted else
                          ", and mixing the two formulas admitted no just-outside neighbour")
                       + "; isolated bounds collide or depend on the conversion formula")
    fields["numerical_result"] = (
        f"Interior/bound samples: {interior_mismatch} code mismatches across {len(UNIT_SYSTEMS)} unit systems; "
        f"vertex check passed in {sum(v is True for v in vertices.values())}/{len(vertices)}. Just-outside MSD "
        f"samples admitted: {sum(len(v) for v in admitted.values())}; MSD bound samples refused when converted as "
        f"k/(1/c): {len(divided_refused)}; just-outside samples admitted when converted as k/(1/c): "
        f"{len(divided_admitted)} ({sorted(divided_admitted)}). Interior margin ratio spread {ratio_spread:.3g}x. "
        f"Light damping codes {light} "
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
BOX, RATE_BOX = ([-1.0], [1.0]), ([-0.2], [0.2])
# One threshold crossing of a one-parameter path moves one condition (gate) of the documented order, or two
# whose thresholds coincide: scalar_positive with not_definite when x is a top eigenvector of M (x^T M x equals
# max eig(M) |x|^2), and certified with margin_low (a MARGIN_LOW margin crossing the resolution).
CROSSINGS = tuple(frozenset({gate}) for gate in R.GATES) + (frozenset({"scalar_positive", "not_definite"}),
                                                            frozenset({"certified", "margin_low"}))
CROSSING_LISTS = [sorted(crossing) for crossing in CROSSINGS]
RUNTIME_STATUS_NOT_POSITIVE = ("which RUNTIME-STATUS-v1 at the pinned commit describes as 'P is not positive definite "
                               "here, so there is no certificate to evaluate'")


def consistent_gates(gates) -> bool:
    """Gate vectors the documented order can hold at one sample (up to rounding).

    scalar_positive implies not_definite and not certified: x^T M x > res |x|^2 forces max eig(M) > res by the
    Rayleigh quotient, so margin < -res. certified excludes not_definite: margin > res means max eig(M) < -res.
    margin_low is read only on the certified branch.
    """
    if gates["scalar_positive"] and (gates["certified"] or not gates["not_definite"]):
        return False
    if gates["certified"] and gates["not_definite"]:
        return False
    return gates["certified"] or not gates["margin_low"]


def pair_key(first, second):
    a, b = sorted((first, second), key=R.RUNTIME_CODES.index)
    return f"{a} <-> {b}"


def transition_graph() -> dict:
    """Every unordered pair of runtime codes, classified by whether one threshold crossing connects them.

    ``allowed``: some consistent gate vectors giving the two codes differ by exactly one crossing (CROSSINGS);
    ``rounding only``: the same, but one code is CERTIFICATE_NOT_POSITIVE, whose not_positive gate a declared
    certificate reaches only through rounding (both certificate kinds refuse min eig(P) <= 0 when built, so it
    needs V < 0 at the scaled state for a P whose computed eigenvalues are positive); ``excluded``: every
    consistent pair of gate vectors differs by more than one crossing, so any path between the two codes passes
    through a third one.
    """
    by_code = {}
    for bits in product((False, True), repeat=len(R.GATES)):
        gates = dict(zip(R.GATES, bits))
        if consistent_gates(gates):
            by_code.setdefault(R.gate_code(gates), []).append(gates)
    graph = {}
    for first, second in combinations(R.RUNTIME_CODES, 2):
        differences = sorted({tuple(g for g in R.GATES if a[g] != b[g]) for a in by_code[first]
                              for b in by_code[second]}, key=lambda d: (len(d), d))
        crossings = [list(d) for d in differences if frozenset(d) in CROSSINGS]
        status = ("excluded" if not crossings else
                  "rounding only" if "CERTIFICATE_NOT_POSITIVE" in (first, second) else "allowed")
        graph[pair_key(first, second)] = {"codes": [first, second], "status": status, "crossings": crossings,
                                          "fewest_changed_gates": list(differences[0])}
    return graph


def status_paths():
    """One-parameter paths through the decision order; each step is a declared (A, P, x, options) case.

    The first seven sweep one parameter each. The rest are short paths across one threshold crossing that,
    with the sweeps, realise every transition the order allows between rounding-free codes
    (transition_graph): the box edge theta = 1 (A1 = 0, so theta moves only the box gate) under each in-box
    code, a power-of-two matrix scale across the overflow edge (every later gate is homogeneous in it), the
    level across V = 4, a vanishing margin under a declared margin, and a top eigenvalue crossing zero.
    """
    I2, e1, zero = np.eye(2), np.array([1.0, 0.0]), np.zeros((2, 2))
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
    edge = {"NUMERICAL_OVERFLOW": dict(A0=-1.5 * 2.0 ** 1023 * I2),
            "OUTSIDE_LEVEL_SET": dict(A0=-I2, x=2 * e1, level=1.0), "NOT_CERTIFIED": dict(A0=I2),
            "MARGIN_LOW": dict(A0=-I2, required_margin=3.0), "DECREASE_NOT_DEFINITE": dict(A0=np.diag([-1.0, 1.0])),
            "NUMERICAL_INCONCLUSIVE": dict(A0=ROTATION)}
    for code, step in edge.items():
        paths[f"box edge theta = 1, {code} inside"] = [dict(step, A1=zero, theta=t)
                                                        for t in (1.0, math.nextafter(1.0, math.inf))]
    scale = {"OUTSIDE_LEVEL_SET": (-I2, I2, 2 * e1, dict(level=1.0), 1022),
             "NOT_CERTIFIED": (I2, I2, e1, {}, 1022),
             "MARGIN_LOW": (-np.diag([1.0, 2.0 ** -30]), I2, e1, dict(required_margin=1e300), 1022),
             "DECREASE_NOT_DEFINITE": (np.diag([-1.0, 1.0]), I2, e1, {}, 1022),
             "NUMERICAL_INCONCLUSIVE": (ROTATION, np.ldexp(I2, 512), e1, {}, 511)}
    for code, (B, P, x, options, k) in scale.items():
        paths[f"matrix scale across overflow, {code} below"] = [dict(A=np.ldexp(B, j), P=P, x=x, **options)
                                                                 for j in (k, k + 1)]
    level = {"NOT_CERTIFIED": (I2, {}), "MARGIN_LOW": (-I2, dict(required_margin=3.0)),
             "DECREASE_NOT_DEFINITE": (np.diag([-1.0, 1.0]), {}), "NUMERICAL_INCONCLUSIVE": (ROTATION, {})}
    for code, (A, options) in level.items():
        paths[f"level across V = 4, {code} inside"] = [dict(A=A, P=I2, x=2 * e1, level=v, **options)
                                                        for v in (4.0, 3.99)]
    paths["stability a with required margin 1"] = [dict(A=np.array([[a, 1.0], [-1.0, a]]), P=I2, x=e1,
                                                        required_margin=1.0) for a in (-1e-10, -1e-15)]
    paths["top eigenvalue b, A = diag(-1, b)"] = [dict(A=np.diag([-1.0, b]), P=I2, x=e1)
                                                   for b in (-1.0, 0.0, 1e-16, 1.0)]
    return paths


def _step_inputs(step):
    """(A, P, x, in_box, options) of a path step, with A(theta) formed as the runtime forms it."""
    options = {k: step[k] for k in ("level", "required_margin") if k in step}
    if "theta" not in step:
        return step["A"], step["P"], step["x"], True, options
    theta, rate = step["theta"], step.get("theta_dot")
    A0 = np.asarray(step.get("A0", -np.eye(2)), dtype=float)
    A1 = np.asarray(step.get("A1", ROTATION), dtype=float)
    in_box = BOX[0][0] <= theta <= BOX[1][0] and (rate is None or RATE_BOX[0][0] <= rate <= RATE_BOX[1][0])
    return A0.copy() + theta * A1, step.get("P", np.eye(2)), step.get("x", (1.0, 0.0)), in_box, options


def _step_case(cid, step):
    options = {k: step[k] for k in ("level", "required_margin") if k in step}
    if "theta" not in step:
        return _verdict_case(cid, step["A"], step["P"], step["x"], **options)
    rate = step.get("theta_dot")
    return _affine_case(cid, step.get("A0", -np.eye(2)), [step.get("A1", ROTATION)], BOX, step.get("P", np.eye(2)),
                        step.get("x", (1.0, 0.0)), [step["theta"]], None if rate is None else [rate],
                        rate_box=RATE_BOX, **options)


def step_gates(step):
    """The documented order's gate vector and code at one path step (CIW transcription)."""
    A, P, x, in_box, options = _step_inputs(step)
    return R.documented_gates(A, P, x, in_box=in_box, **options)


def _path_cases(paths, candidates):
    cases, predictions, gates = [], {}, {}
    for name, steps in paths.items():
        for j, step in enumerate(steps):
            cid = f"{name}#{j}"
            cases.append(_step_case(cid, step))
            evaluated = step_gates(step)
            predictions[cid], gates[cid] = evaluated["code"], evaluated["gates"]
    for i, candidate in enumerate(candidates):
        cid = f"indefinite P#{i}"
        cases.append(_verdict_case(cid, -np.eye(2), candidate["P"], candidate["weak"]))
        predictions[cid] = R.documented_code(-np.eye(2), candidate["P"], candidate["weak"])["code"]
    return cases, predictions, gates


def path_transitions(paths, codes, gates, graph):
    """Consecutive path steps whose codes differ, with the gates that change and whether the order allows it."""
    rows = []
    for name, steps in paths.items():
        for j in range(len(steps) - 1):
            first, second = f"{name}#{j}", f"{name}#{j + 1}"
            if codes[first] == codes[second]:
                continue
            changed = [g for g in R.GATES if gates[first][g] != gates[second][g]]
            entry = graph.get(pair_key(codes[first], codes[second])) if codes[first] in R.RUNTIME_CODES and \
                codes[second] in R.RUNTIME_CODES else None
            direct = entry is not None and entry["status"] == "allowed" and changed in entry["crossings"]
            rows.append({"path": name, "steps": [j, j + 1], "codes": [codes[first], codes[second]],
                         "changed_gates": changed, "direct": direct})
    return rows


def transition_coverage(graph, rows):
    """Per code pair: its status in the derived graph and the paths on which it occurred directly."""
    coverage = {}
    for key, entry in graph.items():
        paths = sorted({r["path"] for r in rows if r["direct"] and pair_key(*r["codes"]) == key})
        coverage[key] = dict(entry, exercised_on=paths)
    return coverage


SHORT_CODES = {"CERTIFIED_WITH_MARGIN": "CWM", "MARGIN_LOW": "ML", "NOT_CERTIFIED": "NC",
               "DECREASE_NOT_DEFINITE": "DND", "NUMERICAL_INCONCLUSIVE": "NI", "NUMERICAL_OVERFLOW": "NO",
               "OUTSIDE_PARAMETER_BOX": "OPB", "OUTSIDE_LEVEL_SET": "OLS", "CERTIFICATE_NOT_POSITIVE": "CNP"}


def coverage_markdown(coverage):
    """A 9 x 9 matrix: x exercised directly, ! allowed but not exercised, r rounding only, - excluded."""
    mark = {}
    for entry in coverage.values():
        symbol = ("x" if entry["exercised_on"] else "!") if entry["status"] == "allowed" else (
            "r" if entry["status"] == "rounding only" else "-")
        a, b = entry["codes"]
        mark[a, b] = mark[b, a] = symbol
    names = [SHORT_CODES[c] for c in R.RUNTIME_CODES]
    lines = ["Direct transitions between runtime codes (x exercised on a path, ! allowed by the decision order but "
             "not exercised, r only through rounding, - excluded by the decision order).", "",
             "| | " + " | ".join(names) + " |", "| --- " * (len(names) + 1) + "|"]
    for a in R.RUNTIME_CODES:
        lines.append(f"| {SHORT_CODES[a]} | " + " | ".join("." if a == b else mark[a, b] for b in R.RUNTIME_CODES)
                     + " |")
    lines += ["", "Codes: " + ", ".join(f"{short} = {code}" for code, short in SHORT_CODES.items()) + "."]
    return "\n".join(lines) + "\n"


def _affine_certificate_case(cid, theta):
    """A = -I (A1 = 0) on the box [-1, 1] with the affine certificate P(theta) = I + theta diag(0, 1)."""
    case = _affine_case(cid, -np.eye(2), [np.zeros((2, 2))], BOX, np.eye(2), (1.0, 0.0), [theta], [0.0],
                        rate_box=RATE_BOX)
    case["certificate"] = {"P0": _mat(np.eye(2)), "terms": [_mat(np.diag([0.0, 1.0]))]}
    return case


@task("T106", changed_files=PROVIDER_FILES, regression_tests=(_node("test_t106_status_coverage"),
                                                                _node("test_documented_decision_order"),
                                                                _node("test_transition_graph"),
                                                                _node("test_t106_offline_findings_without_the_provider"),
                                                                UNCONDITIONAL_TEST, PARTIAL_TEST, NEXT_STEP_TEST))
def status_transitions(ctx):
    graph = transition_graph()
    allowed = sorted(key for key, entry in graph.items() if entry["status"] == "allowed")
    excluded = sorted(key for key, entry in graph.items() if entry["status"] == "excluded")
    rounding = sorted(key for key, entry in graph.items() if entry["status"] == "rounding only")
    fields = _fields(
        "Each of the nine runtime-status-v1 codes is reachable with declared inputs; every transition between two "
        "codes that the documented decision order allows through one threshold crossing occurs directly on a "
        "one-parameter path, at that crossing; the transitions the order excludes pass through a third code; those "
        "involving CERTIFICATE_NOT_POSITIVE need rounding; and the runtime refuses the five host-owned codes.",
        "Decision order: outside box -> NUMERICAL_OVERFLOW -> CERTIFICATE_NOT_POSITIVE (min eig P <= 0 or V < 0) "
        "-> OUTSIDE_LEVEL_SET -> NOT_CERTIFIED (x^T M x > res |x|^2) -> CERTIFIED_WITH_MARGIN / MARGIN_LOW "
        "(margin > res, MARGIN_LOW iff margin <= required margin) -> DECREASE_NOT_DEFINITE (max eig M > res) -> "
        "NUMERICAL_INCONCLUSIVE. Its conditions form a gate vector (box, overflow, not_positive, level, "
        "scalar_positive, certified, margin_low, not_definite) constrained by scalar_positive => not_definite and "
        "not certified (Rayleigh), certified => not not_definite, margin_low => certified. A one-parameter path "
        "crosses one threshold at a time: one gate changes, or two whose thresholds coincide (scalar_positive with "
        "not_definite for x a top eigenvector of M; certified with margin_low). Two codes are directly connected "
        f"iff consistent gate vectors giving them differ by one crossing: {len(allowed)} pairs among the eight "
        f"rounding-free codes, {len(rounding)} pairs with CERTIFICATE_NOT_POSITIVE (only through rounding: both "
        f"certificate kinds refuse min eig P <= 0 when built), {len(excluded)} pairs excluded.",
        ["24 paths: seven sweeps (required margin 0..3; level 5..-1; stability a in [-1, 1] for [[a, 1], [-1, a]]; "
         "state direction 0..90 degrees for diag(-1, 1); matrix scale 2^0..2^1023; theta across [-1, 1]; theta_dot "
         "across [-0.2, 0.2]) and seventeen short paths across one crossing (box edge theta = 1 -> nextafter(1, "
         "inf) with A1 = 0 under six in-box codes; power-of-two matrix scale across the overflow edge under five "
         "codes; level 4 -> 3.99 at V = 4 under four codes; stability a = -1e-10 -> -1e-15 with required margin 1; "
         "top eigenvalue b = -1, 0, 1e-16, 1 of diag(-1, b))",
         "8 candidate P that NumPy calls positive definite but are exactly indefinite (seed 1041) evaluated along "
         "their weak direction",
         "Affine certificate P(theta) = I + theta diag(0, 1) with A = -I at theta = -0.5 and at theta = -1 "
         "(singular) inside the box [-1, 1]",
         "Host-owned codes MODEL_MISMATCH, STALE_STATE, INVALID_SENSOR_DATA, CERTIFICATE_EXPIRED, RUNTIME_FAULT"],
        "PLSR verdict code per step; the CIW transcription's gate vector and code per step (at an overflowing step "
        "the later gates are read at A scaled down by a power of two, where each is homogeneous); require_status "
        "and Verdict construction outcomes for host-owned codes; the runtime's published constants.",
        "The eight rounding-free codes are reached; every step equals the transcription; each allowed transition "
        "occurs between consecutive steps whose gate vectors differ by exactly its crossing, and no code changes at "
        "a crossing the order does not allow; CERTIFICATE_NOT_POSITIVE is not reached rounding-free and never with "
        "a certificate; every host-owned code is refused.",
        "Derive the transition graph by enumerating the gate vectors the decision order can hold, build each path, "
        "evaluate all steps in one PLSR subprocess, recompute gates and codes in CIW, and tabulate the direct "
        "transitions and their coverage (transition-coverage.json and .md).",
        NEXT_STEPS["T106"],
        ["a code unreachable", "a step off the documented order", "an allowed transition not realised",
         "a code change at a crossing the order does not allow", "a host-owned code accepted",
         "CERTIFICATE_NOT_POSITIVE reachable without rounding", "a certificate on an exactly indefinite P"],
        ["Transitions the documented order excludes (every path between the two codes passes through a third "
         "code): " + "; ".join(f"{key} (differ in at least {', '.join(graph[key]['fewest_changed_gates'])})"
                               for key in excluded) + ".",
         f"The {len(rounding)} transitions involving CERTIFICATE_NOT_POSITIVE are not exercised: QuadraticCertificate "
         "and every P(theta) of an AffineCertificate refuse min eig(P) <= 0 when built (an affine P(theta) that "
         "loses definiteness inside the box raises ValueError, probed here), so the code needs V < 0 at the scaled "
         "state for a P whose computed eigenvalues are positive, which only rounding produces. The eight exactly "
         "indefinite witnesses may reach it; their codes depend on eigvalsh and dot-product rounding and are "
         "retained in the artifact.",
         "A transition is judged at the resolution of the path's steps: 'direct' means consecutive steps whose "
         "transcribed gate vectors differ by one crossing. Rounding at the crossing itself is not probed (T107 "
         "covers the resolution threshold).",
         "The CIW transcription shares the documented specification with the runtime, so it checks implementation "
         "against specification (a same-specification check), not the specification itself."])
    candidates = [c for c in ctx.memo("lyapunov:indefinite-candidates", indefinite_candidates)[0]
                  if not c["exact_pd"]]
    paths = status_paths()
    cases, predictions, gates = _path_cases(paths, candidates)
    path_ids = [cid for cid in predictions if not cid.startswith("indefinite")]
    stable_codes = sorted({predictions[cid] for cid in path_ids} & ROUNDING_FREE_CODES)
    predicted_rows = path_transitions(paths, predictions, gates, graph)
    predicted_coverage = transition_coverage(graph, predicted_rows)
    predicted_missing = [key for key in allowed if not predicted_coverage[key]["exercised_on"]]
    predicted_undeclared = [r for r in predicted_rows if not r["direct"]]
    authority = finding(
        "A CERTIFIED_WITH_MARGIN verdict (operationally_acceptable) authorizes actuation", "actuator_authority", None,
        {"derivation": "runtime-status-v1: operationally_acceptable is not an authorization; host statuses and "
                       "machine-safety functions are outside the evaluator"})
    offline = [
        finding("The documented decision order, transcribed in CIW, assigns the eight rounding-free codes to the "
                "constructed path steps", "numerical", {"codes": stable_codes},
                {"generator": {"name": "status_paths", "seed": None},
                 "checks": [_check("rounding-free runtime codes predicted on the paths", len(stable_codes),
                                   float(len(ROUNDING_FREE_CODES)), "ge", kind="analytic")]},
                tolerance=EXACT_TOL, uncertainty=EXACT),
        finding("The decision order connects the eight rounding-free codes by one threshold crossing in the pairs "
                "the transition graph allows, and the CIW transcription realises each at its crossing on the "
                "constructed paths", "numerical",
                {"allowed": len(allowed), "rounding_only": len(rounding), "excluded": len(excluded),
                 "realised": len(allowed) - len(predicted_missing), "undeclared": len(predicted_undeclared)},
                {"derivation": "gate vectors of the documented order under its implications (Rayleigh: "
                               "scalar_positive => not_definite and not certified; certified => not not_definite; "
                               "margin_low => certified), paired when they differ by one threshold crossing "
                               "(lyapunov.transition_graph)",
                 "generator": {"name": "status_paths", "seed": None},
                 "checks": [_check("allowed pairs with no direct transition of the transcription on the paths",
                                   len(predicted_missing), 0.0, kind="analytic"),
                            _check("transcribed code changes between consecutive steps at a crossing the order does "
                                   "not allow", len(predicted_undeclared), 0.0, kind="analytic")]},
                tolerance=EXACT_TOL, uncertainty=EXACT),
        authority]
    probes = [_affine_certificate_case("affine P theta = -0.5", -0.5),
              _affine_certificate_case("affine P theta = -1", -1.0)]
    for code in R.HOST_OWNED:
        cases.append({"id": f"require:{code}", "op": "require_status", "code": code})
        cases.append({"id": f"verdict:{code}", "op": "host_verdict", "code": code})
    cases += probes + [{"id": "constants", "op": "constants"}]
    try:
        bridge = _bridge(ctx, cases)
    except _Unavailable as exc:
        fields["numerical_result"] = (
            f"Provider-free: the transition graph allows {len(allowed)} direct transitions between rounding-free "
            f"codes, {len(rounding)} only through rounding and excludes {len(excluded)}; the transcription covers "
            f"{len(stable_codes)} rounding-free codes and realises {len(allowed) - len(predicted_missing)} allowed "
            f"transitions on the paths ({len(predicted_undeclared)} code changes at undeclared crossings).")
        fields["uncertainty"] = "Deterministic re-derivation on this platform."
        return _finish(fields, offline, PROVIDER_FILES, blocked=str(exc))
    identity, results = bridge["identity"], bridge["results"]
    base = provider_basis(identity)
    observed = {cid: _code(results[cid]) for cid in predictions}
    path_mismatches = {cid: {"observed": observed[cid], "predicted": predictions[cid]} for cid in path_ids
                       if observed[cid] != predictions[cid]}
    reached_stable = sorted({observed[cid] for cid in path_ids} & ROUNDING_FREE_CODES)
    witness_ids = [cid for cid in predictions if cid.startswith("indefinite")]
    witness_certified = sum(observed[cid] in R.CERTIFYING for cid in witness_ids)
    rows = path_transitions(paths, observed, gates, graph)
    coverage = transition_coverage(graph, rows)
    missing = [key for key in allowed if not coverage[key]["exercised_on"]]
    undeclared = [r for r in rows if not r["direct"]]
    sequences = {name: [observed[f"{name}#{j}"] for j in range(len(steps))] for name, steps in paths.items()}
    host = {code: {"require_status": _code(results[f"require:{code}"]),
                   "Verdict": _code(results[f"verdict:{code}"])} for code in R.HOST_OWNED}
    constants = {k: v for k, v in results["constants"].items() if k not in ("ok", "id", "warnings")}
    examples = {code: sorted(cid for cid, got in observed.items() if got == code)[:3] for code in R.RUNTIME_CODES}
    singular, control = results["affine P theta = -1"], results["affine P theta = -0.5"]
    singular_message = singular["error"]["message"] if not singular["ok"] else ""
    # Codes of the rounding-dependent witnesses (CERTIFICATE_NOT_POSITIVE or not) are retained here only.
    ctx.artifact_json("status-coverage.json", R.jsonable({"coverage_examples": examples, "sequences": sequences,
                                                           "observed": observed, "predicted": predictions,
                                                           "host_owned": host, "constants": constants,
                                                           "affine_certificate_probe": {
                                                               "theta = -0.5": _code(control),
                                                               "theta = -1": _code(singular),
                                                               "theta = -1 error": singular.get("error")}}))
    ctx.artifact_json("transition-coverage.json", R.jsonable({"crossings": CROSSING_LISTS, "pairs": coverage,
                                                               "transitions": rows}))
    ctx.artifact_text("transition-coverage.md", coverage_markdown(coverage))
    lines = ["| Code | Reached | Example step |", "| --- | --- | --- |"]
    lines += [f"| {code} | {'yes' if examples[code] else 'no'} | {examples[code][0] if examples[code] else '-'} |"
              for code in R.RUNTIME_CODES]
    ctx.artifact_text("status-coverage.md", "\n".join(lines) + "\n")
    findings = [
        finding("The eight rounding-free runtime-status-v1 codes are reached on one-parameter paths far from every "
                "rounding threshold", "numerical", {"codes": reached_stable},
                {"provider": base, "checks": [_check("rounding-free codes reached", len(reached_stable),
                                                     float(len(ROUNDING_FREE_CODES)), "ge", kind="invariant")]},
                tolerance=EXACT_TOL, uncertainty=EXACT),
        finding("Codes along each one-parameter path follow the documented decision order", "numerical", sequences,
                {"provider": base, "checks": [
                    _check("path steps differing from the CIW transcription of the documented order (same "
                           "specification)", len(path_mismatches), 0.0, kind="analytic")]},
                tolerance=EXACT_TOL, uncertainty=EXACT),
        finding("Every transition the decision order allows between two rounding-free codes occurs in PLSR between "
                "consecutive path steps at its crossing, and no PLSR code changes at a crossing the order does not "
                "allow", "numerical",
                {"allowed": len(allowed), "exercised": len(allowed) - len(missing), "code_changes": len(rows),
                 "undeclared": len(undeclared)},
                {"provider": base, "checks": [
                    _check("allowed pairs without a direct PLSR transition on the paths", len(missing), 0.0,
                           kind="invariant"),
                    _check("PLSR code changes between consecutive steps whose transcribed gates differ by other than "
                           "an allowed crossing", len(undeclared), 0.0, kind="invariant")]},
                tolerance=EXACT_TOL, uncertainty=EXACT),
        finding("The exactly indefinite P witnesses are never certified along their weak direction", "numerical",
                {"witnesses": len(witness_ids), "certifying": witness_certified},
                {"provider": base, "checks": [_check("certifying verdicts on the exactly indefinite P witnesses",
                                                     witness_certified, 0.0, kind="invariant")]},
                tolerance=EXACT_TOL,
                uncertainty=_platform(0.0, "which code the witnesses receive (CERTIFICATE_NOT_POSITIVE or "
                                           "NUMERICAL_INCONCLUSIVE) depends on eigvalsh and dot-product rounding and "
                                           "is retained in the artifact; the certification count does not")),
        finding("An affine certificate that is singular at an in-box theta raises ValueError instead of returning "
                "CERTIFICATE_NOT_POSITIVE", "numerical",
                {"theta = -1": _code(singular), "theta = -0.5": _code(control)},
                {"provider": base, "checks": [
                    _refusal("P(theta) = diag(1, 0) at theta = -1 inside the box [-1, 1]", "raises ValueError",
                             _code(singular)),
                    _check("refusals not raised by the certificate's positive-definiteness test",
                           int("positive definite" not in singular_message), 0.0, kind="invariant"),
                    _check("control P(-0.5) = diag(1, 0.5) verdicts other than CERTIFIED_WITH_MARGIN",
                           int(_code(control) != "CERTIFIED_WITH_MARGIN"), 0.0, kind="invariant")]},
                tolerance=EXACT_TOL,
                counterexample={"statement": "An in-box sample at which P is not positive definite yields "
                                             "CERTIFICATE_NOT_POSITIVE, " + RUNTIME_STATUS_NOT_POSITIVE,
                                "witness": {"A0": "-I", "A1": "0", "box": [-1.0, 1.0], "P0": "I",
                                            "P1": "diag(0, 1)", "theta": -1.0, "theta_dot": 0.0, "x": [1.0, 0.0],
                                            "outcome": _code(singular)}},
                uncertainty=_roundoff(0.0, "P(-1) = diag(1, 0) is formed exactly and its eigenvalue 0 is exact")),
        finding("The runtime refuses to emit the five host-owned status codes", "numerical", host,
                {"provider": base, "checks": [
                    _refusal(f"{way} for {code}", "raises ValueError", outcome[way])
                    for code, outcome in host.items() for way in ("require_status", "Verdict")]},
                tolerance=EXACT_TOL, uncertainty=EXACT),
        finding("Pinned runtime constants: resolution factor, numerical policy, status vocabulary", "provenance",
                constants, {"provider": base, "checks": [
                    _check("DECREASE_RESOLUTION_FACTOR minus the documented 1.0",
                           constants.get("DECREASE_RESOLUTION_FACTOR", math.nan) - 1.0, 0.0, kind="invariant"),
                    _check("NUMERICAL_POLICY_VERSION differs from float64-decrease-v1 (1 if so)",
                           int(constants.get("NUMERICAL_POLICY_VERSION") != "float64-decrease-v1"), 0.0,
                           kind="invariant"),
                    _check("status codes outside or missing from the documented nine",
                           len(set(constants.get("RUNTIME_STATUSES", [])) ^ set(R.RUNTIME_CODES)), 0.0,
                           kind="invariant"),
                    _check("host-owned codes outside or missing from the documented five",
                           len(set(constants.get("HOST_OWNED_STATUSES", [])) ^ set(R.HOST_OWNED)), 0.0,
                           kind="invariant")]}, tolerance=EXACT_TOL, uncertainty=EXACT),
    ] + offline
    if missing:
        fields["unresolved_assumptions"] = list(fields["unresolved_assumptions"]) + [
            "Allowed transitions not exercised in this run: " + "; ".join(missing) + "."]
    fields["numerical_result"] = (
        f"Rounding-free codes reached on the paths: {len(reached_stable)}/{len(ROUNDING_FREE_CODES)} "
        f"({reached_stable}); {len(path_mismatches)} of {len(path_ids)} path steps differ from the CIW transcription "
        f"of the documented order. Transition graph: {len(allowed)} allowed pairs among the rounding-free codes, "
        f"{len(excluded)} excluded, {len(rounding)} involving CERTIFICATE_NOT_POSITIVE only through rounding; "
        f"{len(allowed) - len(missing)} of the {len(allowed)} allowed transitions occurred directly in PLSR "
        f"({len(rows)} code changes, {len(undeclared)} at a crossing the order does not allow; matrix in "
        f"transition-coverage.md). Indefinite-P witnesses: {witness_certified} of {len(witness_ids)} certifying "
        f"(codes retained in the artifact). Affine certificate singular in the box: {_code(singular)} (neighbour "
        f"theta = -0.5: {_code(control)}). Host-owned codes refused: "
        f"{sum(set(v.values()) == {'raises ValueError'} for v in host.values())}/5. "
        f"Constants: resolution factor {constants.get('DECREASE_RESOLUTION_FACTOR')}, policy "
        f"{constants.get('NUMERICAL_POLICY_VERSION')}.")
    fields["uncertainty"] = ("Path codes are far from rounding thresholds and platform-independent. The "
                             "indefinite-P witnesses depend on eigvalsh and dot-product rounding, so only their "
                             "non-certification is checked and their codes are retained in the artifact.")
    return _finish(fields, findings, PROVIDER_FILES, identity)


# T107 ------------------------------------------------------------------------

# Targets kappa = max eig(M) / res of the window cases. Each exact window kappa +- 1/16 keeps at least 3/16 of a
# resolution from every threshold at which PLSR's code changes (-3 res under the declared margin, -res, +res), so
# the code of a window case follows from its exact position whenever the computed max eig(M) and x^T M x err by
# less than that distance, whatever the BLAS kernel.
T107_KAPPAS = (-8.0, -3.5, -2.5, -1.75, -1.25, -0.75, -0.25, 0.0, 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 8.0)
T107_WINDOW = Fraction(1, 16)
T107_DISTANCE = Fraction(3, 16)
CODE_THRESHOLDS = (-3, -1, 1)
BAND = ("[-2, -1) res", "[-1, 0) res", "[0, 1) res", "[1, 2) res")


def window_distance(kappa) -> Fraction:
    """Distance in resolutions from the exact window kappa +- 1/16 to the nearest code threshold."""
    return min(abs(Fraction(kappa) - t) for t in CODE_THRESHOLDS) - T107_WINDOW


def window_codes(kappa, scalar_above):
    """The codes at required_margin 0 and 3 res implied by a window case's exact position.

    ``scalar_above``: the exact x^T M x exceeds res |x|^2 (PLSR's scalar gate, decided before the eigenvalue).
    """
    if kappa < -1:
        return "CERTIFIED_WITH_MARGIN", "CERTIFIED_WITH_MARGIN" if kappa < -3 else "MARGIN_LOW"
    if kappa < 1:
        return "NUMERICAL_INCONCLUSIVE", "NUMERICAL_INCONCLUSIVE"
    code = "NOT_CERTIFIED" if scalar_above else "DECREASE_NOT_DEFINITE"
    return code, code


def _boundary_member(kind, n, kappa, A, P, rng):
    """Exact bin, class, bracket and window of one near-boundary case.

    Its state x is redrawn until the exact Rayleigh quotient x^T M x / |x|^2 keeps the declared distance from
    res, the threshold of PLSR's scalar gate.
    """
    exact = R.exact_form(A, P)
    res = R.rounded_resolution(A, P)
    while True:
        x = R.dyadic_state(rng, n)
        quotient = R.exact_quadratic(x, exact) / sum(Fraction(v) ** 2 for v in x)
        if abs(quotient - Fraction(res)) >= T107_DISTANCE * Fraction(res):
            break
    member = {"kind": kind, "n": n, "kappa": kappa, "A": A, "P": P, "x": x, "resolution": res,
              "exact_class": R.exact_class(exact), "exact_bin": R.resolution_bin(exact, res),
              "bracket": R.lambda_bracket(exact, res), "scalar_above": quotient > Fraction(res)}
    if kind == "window":
        member["in_window"] = (not R.lambda_max_below(exact, (Fraction(kappa) - T107_WINDOW) * Fraction(res))
                               and R.lambda_max_below(exact, (Fraction(kappa) + T107_WINDOW) * Fraction(res)))
        member["predicted"] = window_codes(kappa, member["scalar_above"])
    return member


def _straddle(P, build):
    """The A of two adjacent float targets between which the exact max eig(M) crosses -res (exact bisection)."""
    def beyond(top):
        A = build(top)
        return R.lambda_max_below(R.exact_form(A, P), -Fraction(R.rounded_resolution(A, P)))

    res = R.rounded_resolution(build(0.0), P)
    low, high = -1.5 * res, -0.5 * res
    assert beyond(low) and not beyond(high)
    while (middle := 0.5 * (low + high)) not in (low, high):
        if beyond(middle):
            low = middle
        else:
            high = middle
    return build(low), build(high)


def boundary_family():
    """Near-boundary continuous cases with exact bins of max eig(M), the same on every BLAS kernel.

    Window cases (seed 107) have their exact max eig(M) within 1/16 res of kappa res; straddle cases (seed 1071)
    are pairs whose exact max eig(M) lies on either side of -res, within rounding of it. Every matrix is built
    from integer draws in exact rational arithmetic and rounded once (lyapunov_reference.exact_threshold_builder).
    """
    family = []
    rng = R.generator(107)
    for n in (2, 3, 4):
        for spd in (False, True):
            for kappa in T107_KAPPAS:
                P, build = R.exact_threshold_builder(rng, n, spd)
                A = build(kappa * R.rounded_resolution(build(0.0), P))
                family.append(_boundary_member("window", n, kappa, A, P, rng))
    rng = R.generator(1071)
    for n in (2, 3, 4):
        for spd in (False, True):
            P, build = R.exact_threshold_builder(rng, n, spd)
            family += [_boundary_member("straddle", n, -1.0, A, P, rng) for A in _straddle(P, build)]
    return family


@task("T107", changed_files=PROVIDER_FILES,
      regression_tests=(_node("test_t107_inconclusive_band"), _node("test_boundary_family_is_exact_and_kernel_free"),
                        UNCONDITIONAL_TEST, PARTIAL_TEST, NEXT_STEP_TEST))
def inconclusive_band(ctx):
    fields = _fields(
        "PLSR never gives a certifying code to a near-boundary decrease form that is not exactly negative "
        "definite; on forms whose exact max eig(M) keeps 3/16 of a resolution from its code thresholds it gives the "
        "code that exact position predicts, so beyond two resolutions from zero it always resolves the sign and "
        "under a declared margin of three resolutions no case within two resolutions is CERTIFIED_WITH_MARGIN; "
        "within rounding of -res the code is CERTIFIED_WITH_MARGIN or NUMERICAL_INCONCLUSIVE on either exact side. "
        "A stronger candidate statement formulated for this experiment (it is not quoted from the runtime's "
        "documentation) -- that near-boundary spectra yield NUMERICAL_INCONCLUSIVE or MARGIN_LOW rather than "
        "CERTIFIED_WITH_MARGIN -- is tested at required_margin 0 as a candidate counterexample.",
        "If the resolution bounds the float64 error e of max eig(M) (|e| <= res), then exact lambda < -2 res gives "
        "margin > res (certified), exact lambda >= 2 res gives max eig > res (not definite), and exact lambda >= 0 "
        "can never give margin > res. With required margin 3 res, certification needs margin > 3 res, impossible "
        "for exact lambda >= -2 res. When the realized |e| and the error of the computed x^T M x stay below the "
        "distance between a case's exact window and the thresholds -3 res, -res and +res, the code is a function "
        "of the exact window alone, and so the same on every BLAS kernel.",
        ["90 window cases (PCG64 seed 107): n in {2, 3, 4}, P = I or a rational SPD P with eigenvalues 1 to 10 "
         f"rounded entrywise, target kappa in {list(T107_KAPPAS)} (target max eig = kappa * resolution); "
         "A = P^-1 (N/2 + K) with skew part 100, solved in exact rational arithmetic from integer draws and rounded "
         "once, so every BLAS kernel declares the same A; exact max eig(M) within res/16 of the target",
         "12 straddle cases (seed 1071): for each n and kind of P, the A of two adjacent float targets between "
         "which the exact max eig(M) crosses -res (bisection on exact Sylvester tests)",
         "States x in 32nds of [-2, 2], redrawn until the exact x^T M x / |x|^2 keeps 3/16 res from res",
         "Each case evaluated with required_margin 0 and 3 * resolution"],
        "PLSR code and margin; exact bin of max eig(M) from Sylvester tests on M - t I at t = -2, -1, 0, 1, 2 "
        "resolutions and an exact bisection bracket of max eig(M) / res, in exact dyadic arithmetic; res is the "
        "documented resolution with max|M| read from the correctly rounded exact form.",
        "No certifying code unless the exact form is negative definite; every window case gets the code its exact "
        "window predicts (certified below -res, MARGIN_LOW between -3 res and -res under the declared margin, "
        "inconclusive within one resolution, not definite or not certified beyond +res); straddle cases get "
        "CERTIFIED_WITH_MARGIN or NUMERICAL_INCONCLUSIVE. Without a declared margin, exactly negative definite band "
        "cases may be certified (soundly) or left inconclusive.",
        "Generate the cases in exact arithmetic, bin and bracket their exact spectra, evaluate with PLSR at both "
        "margins, compare every window code with its prediction, measure the realized eigenvalue error and tabulate "
        "codes per bin.",
        NEXT_STEPS["T107"],
        ["certifying code on a form that is not exactly negative definite", "window code differing from its exact "
         "window", "realized eigenvalue error reaching the declared distance", "unresolved sign beyond two "
         "resolutions", "CERTIFIED_WITH_MARGIN inside the band under a declared margin",
         "MARGIN_LOW inconsistent with the declared margin", "straddle code other than CERTIFIED_WITH_MARGIN or "
         "NUMERICAL_INCONCLUSIVE",
         "CERTIFIED_WITH_MARGIN inside the band at required_margin 0 (searched as counterexample)"],
        ["Exact bins and windows describe the declared binary64 matrices; the targets kappa are realised within "
         "res/16 (checked exactly), not exactly.", "Which straddle cases are certified is decided by the last-bit "
         "rounding of the computed eigenvalue and may differ between BLAS kernels and LAPACK builds; it is retained "
         "per case in inconclusive-band.json and stated in the numerical result, not compared.",
         "The band refusal rate depends on the generator and is not a property of plants in general."])
    family = ctx.memo("lyapunov:boundary-family", boundary_family)
    window = [i for i, member in enumerate(family) if member["kind"] == "window"]
    straddle = [i for i, member in enumerate(family) if member["kind"] == "straddle"]
    bins = _counts(member["exact_bin"] for member in family)
    outside = sum(not family[i]["in_window"] for i in window)
    offline = [finding("The near-boundary family populates every exact resolution bin of max eig(M), each window "
                       "case within res/16 of its target", "numerical",
                       {"cases": len(family), "bins": bins, "outside_window": outside},
                       {"generator": {"name": "boundary_family", "seed": 107},
                        "checks": [_check("exact bins without a case (six bins from below -2 res to above 2 res)",
                                          6 - len(bins), 0.0),
                                   _check("window cases whose exact max eig(M) lies outside (kappa +- 1/16) res",
                                          outside, 0.0)]},
                       tolerance=EXACT_TOL,
                       uncertainty=_roundoff(0.0, "exact rational bins and windows of matrices built without BLAS: "
                                                  "the same on every kernel"))]
    cases = []
    for i, member in enumerate(family):
        cases.append(_verdict_case(f"b{i}:0", member["A"], member["P"], member["x"]))
        cases.append(_verdict_case(f"b{i}:3", member["A"], member["P"], member["x"],
                                   required_margin=3.0 * member["resolution"]))
    try:
        bridge = _bridge(ctx, cases)
    except _Unavailable as exc:
        fields["numerical_result"] = f"Provider-free exact bins: {bins}; window cases outside their window: {outside}."
        fields["uncertainty"] = "Exact bins and windows are exact and identical on every BLAS kernel."
        return _finish(fields, offline, PROVIDER_FILES, blocked=str(exc))
    identity, results = bridge["identity"], bridge["results"]
    base = provider_basis(identity)
    table, window_table = {}, {}
    unsound = unresolved = certified_in_band_declared = margin_low_mismatch = margin_low_seen = 0
    band_nd, band_nd_inconclusive, band_nd_certified = 0, 0, 0
    mismatches, errors, rows, points = 0, [], [], []
    for i, member in enumerate(family):
        zero, declared = results[f"b{i}:0"], results[f"b{i}:3"]
        code0, code3 = _code(zero), _code(declared)
        table.setdefault(member["exact_bin"], {}).setdefault(code0, 0)
        table[member["exact_bin"]][code0] += 1
        unsound += (code0 in R.CERTIFYING or code3 in R.CERTIFYING) and member["exact_class"] != "negative_definite"
        # Realized error of the computed max eig(M) = -margin against the exact bracket, in resolutions.
        computed = -zero["margin"] / member["resolution"]
        low, high = member["bracket"]
        errors.append(max(computed - float(high), float(low) - computed, 0.0))
        if member["kind"] == "window":
            window_table.setdefault(member["exact_bin"], {}).setdefault(code0, 0)
            window_table[member["exact_bin"]][code0] += 1
            mismatches += (code0, code3) != member["predicted"]
            if member["exact_bin"] == "below -2 res":
                unresolved += code0 != "CERTIFIED_WITH_MARGIN"
            if member["exact_bin"] == "at or above 2 res":
                unresolved += code0 not in ("DECREASE_NOT_DEFINITE", "NOT_CERTIFIED")
            if member["exact_bin"] in BAND and member["exact_class"] == "negative_definite":
                band_nd += 1
                band_nd_inconclusive += code0 == "NUMERICAL_INCONCLUSIVE"
                band_nd_certified += code0 == "CERTIFIED_WITH_MARGIN"
        if member["exact_bin"] in BAND:
            certified_in_band_declared += code3 == "CERTIFIED_WITH_MARGIN"
        margin, res, required = declared["margin"], declared["resolution"], declared["required_margin"]
        if margin > res:
            expected = "MARGIN_LOW" if margin <= required else "CERTIFIED_WITH_MARGIN"
            if code0 == "CERTIFIED_WITH_MARGIN":
                margin_low_mismatch += code3 != expected
                margin_low_seen += code3 == "MARGIN_LOW"
        points.append((member["kappa"], zero["margin_ratio"], code0))
        rows.append({"case": i, "kind": member["kind"], "n": member["n"], "kappa": member["kappa"],
                     "exact_bin": member["exact_bin"], "exact_class": member["exact_class"],
                     "exact_bracket_over_res": [float(low), float(high)], "code_at_required_margin_0": code0,
                     "code_at_required_margin_3_res": code3, "margin_ratio": zero["margin_ratio"],
                     "realized_error_over_res": errors[-1]})
    refusal_rate = band_nd_inconclusive / band_nd if band_nd else 0.0
    band = [i for i, member in enumerate(family) if member["exact_bin"] in BAND]
    band_certified = [i for i in band if _code(results[f"b{i}:0"]) == "CERTIFIED_WITH_MARGIN"]
    band_certified_unsound = sum(family[i]["exact_class"] != "negative_definite" for i in band_certified)
    window_certified = [i for i in band_certified if family[i]["kind"] == "window"]
    straddle_codes = {i: _code(results[f"b{i}:0"]) for i in straddle}
    straddle_other = sum(code not in ("CERTIFIED_WITH_MARGIN", "NUMERICAL_INCONCLUSIVE")
                         for code in straddle_codes.values())
    beyond = [i for i in straddle if family[i]["exact_bin"] == "[-2, -1) res"]
    inside = [i for i in straddle if family[i]["exact_bin"] == "[-1, 0) res"]
    # Each pair is (exactly beyond -res, at or above it); a pair on one side would not straddle the threshold.
    not_straddling = sum(family[a]["exact_bin"] != "[-2, -1) res" or family[b]["exact_bin"] != "[-1, 0) res"
                         for a, b in zip(straddle[0::2], straddle[1::2]))
    beyond_certified = sum(straddle_codes[i] == "CERTIFIED_WITH_MARGIN" for i in beyond)
    inside_certified = sum(straddle_codes[i] == "CERTIFIED_WITH_MARGIN" for i in inside)
    # Distance of each straddle case's exact bracket from -res, in resolutions.
    straddle_distance = max(max(abs(float(family[i]["bracket"][0]) + 1.0), abs(float(family[i]["bracket"][1]) + 1.0))
                            for i in straddle)
    window_error = max(errors[i] for i in window)
    straddle_error = max(errors[i] for i in straddle)
    ctx.artifact_json("inconclusive-band.json", R.jsonable({"codes_by_exact_bin": table,
                                                            "window_codes_by_exact_bin": window_table,
                                                            "cases": rows, "points": points}))
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
    if window_certified:
        # The witness is a window case: exactly two resolutions or less from zero, and certified on every kernel.
        chosen = window_certified[0]
        member = family[chosen]
        band_extra["counterexample"] = {
            "statement": "Near-boundary spectra yield NUMERICAL_INCONCLUSIVE or MARGIN_LOW rather than "
                         "CERTIFIED_WITH_MARGIN (candidate hypothesis formulated for T107, not a quoted "
                         "specification)",
            "witness": {"case": chosen, "n": member["n"], "exact_bin": member["exact_bin"],
                        "exact_class": member["exact_class"],
                        "exact_window_over_res": [float(member["kappa"] - T107_WINDOW),
                                                  float(member["kappa"] + T107_WINDOW)],
                        "code_at_required_margin_0": "CERTIFIED_WITH_MARGIN",
                        "margin_ratio": results[f"b{chosen}:0"]["margin_ratio"], "A_hex": R.hexed(member["A"]),
                        "P_hex": R.hexed(member["P"]), "x": member["x"].tolist()}}
    exact_counts = _roundoff(0.0, "exact counts: each counted outcome is decided at a threshold at least 3/16 res "
                                  "from the case's exact max eig(M), beyond the realized eigenvalue error, so they do "
                                  "not move between BLAS kernels")
    findings = [
        finding("No near-boundary case receives a certifying code unless its exact decrease form is negative "
                "definite", "numerical", {"cases": len(family), "violations": unsound},
                {"provider": base, "independent_check": exact_checker}, tolerance=EXACT_TOL,
                uncertainty=EXACT),
        finding("Every window case receives the code its exact window predicts, PLSR's computed max eig(M) erring "
                "by less than the 3/16 res that separates each window from the code thresholds", "numerical",
                {"window_cases": len(window), "mismatches": mismatches},
                {"provider": base,
                 "independent_check": _independent(
                     _check("window cases whose codes at required_margin 0 and 3 res differ from those implied by "
                            "the exact window and the exact sign of x^T M x - res |x|^2", mismatches, 0.0), identity),
                 "checks": [_check("largest |computed - exact| max eig(M) over all cases, in resolutions (exact "
                                   "bracket of width 2e-6 res)", max(errors), float(T107_DISTANCE), "le")]},
                tolerance=EXACT_TOL,
                uncertainty=_roundoff(max(errors), "largest realized error of PLSR's computed max eig(M) against "
                                                   "the exact value, in resolutions; it differs between BLAS "
                                                   "kernels at its own size and is not a compared value")),
        finding("Beyond two resolutions from zero PLSR always resolves the sign", "numerical",
                {"resolved_cases": bins.get("below -2 res", 0) + bins.get("at or above 2 res", 0),
                 "unresolved": unresolved},
                {"provider": base, "independent_check": _independent(
                    _check("cases beyond two resolutions not resolved to the exact sign", unresolved, 0.0), identity)},
                tolerance=EXACT_TOL, uncertainty=exact_counts),
        finding("With a declared margin of three resolutions no case within two resolutions of zero is "
                "CERTIFIED_WITH_MARGIN", "numerical",
                {"band_cases": len(band), "certified": certified_in_band_declared},
                {"provider": base, "independent_check": _independent(
                    _check("CERTIFIED_WITH_MARGIN in the exact band under required_margin = 3 res",
                           certified_in_band_declared, 0.0), identity)},
                tolerance=EXACT_TOL, uncertainty=exact_counts),
        finding("At required_margin 0 near-boundary spectra within two resolutions of zero receive "
                "CERTIFIED_WITH_MARGIN, and every such certificate is exactly sound", "numerical",
                {"band_cases": len(band), "certified_window_cases": len(window_certified),
                 "unsound": band_certified_unsound},
                {"provider": base, "checks": [
                    _check("window band cases certified at required_margin 0", len(window_certified), 1.0, "ge",
                           kind="invariant"),
                    _check("band certificates (window and straddle) whose exact decrease form is not negative "
                           "definite", band_certified_unsound, 0.0)]},
                tolerance={"abs": 0.03, "rel": 0.0},
                uncertainty=_roundoff(window_error, (
                    f"the counts are exact; the witness's computed margin ratio carries the realized eigenvalue error "
                    f"of its BLAS kernel (at most {window_error:.2g} res here; 0.0096 to 0.0116 res on the SkylakeX, "
                    "Haswell and Sandybridge OpenBLAS kernels, whose band-certified margin ratios differed by at most "
                    "0.0018), so two kernels differ by at most twice the largest realized error; the regression "
                    "tolerance abs 0.03 res covers that and admits no change of a count")),
                **band_extra),
        finding("Straddle cases within rounding of -res receive CERTIFIED_WITH_MARGIN or NUMERICAL_INCONCLUSIVE on "
                "either exact side of the threshold", "numerical",
                {"cases": len(straddle), "exactly_beyond": len(beyond), "other_codes": straddle_other},
                {"provider": base,
                 "independent_check": _independent(
                     _check("straddle codes other than CERTIFIED_WITH_MARGIN or NUMERICAL_INCONCLUSIVE (the codes "
                            "an exact max eig(M) in (-2 res, 0) admits)", straddle_other, 0.0), identity),
                 "checks": [_check("straddle pairs whose exact max eig(M) does not cross -res", not_straddling,
                                   0.0)]},
                tolerance=EXACT_TOL,
                uncertainty=_platform(straddle_distance, "largest exact distance of a straddle case from -res, in "
                                                         "resolutions; which of them are certified is decided by "
                                                         "the rounding of the computed eigenvalue (realized error up "
                                                         f"to {straddle_error:.2g} res here), may differ between "
                                                         "BLAS kernels and LAPACK builds, and is retained per case "
                                                         "in inconclusive-band.json, not compared")),
        finding("MARGIN_LOW appears exactly when the resolvable margin does not exceed the declared margin",
                "numerical", {"margin_low_observed": margin_low_seen > 0, "mismatches": margin_low_mismatch},
                {"provider": base, "checks": [_check("codes differing from the declared-margin rule",
                                                     margin_low_mismatch, 0.0, kind="invariant"),
                                              _check("MARGIN_LOW verdicts observed", margin_low_seen, 1.0, "ge",
                                                     kind="invariant")]},
                tolerance=EXACT_TOL, uncertainty=EXACT),
        finding("Share of exactly negative definite band window cases answered NUMERICAL_INCONCLUSIVE without a "
                "declared margin", "numerical", {"inconclusive_share": refusal_rate},
                {"provider": base}, tolerance=EXACT_TOL,
                uncertainty=_roundoff(0.0, "a ratio of exact counts decided by the exact windows")),
    ] + offline
    fields["numerical_result"] = (
        f"{len(family)} cases ({len(window)} window, {len(straddle)} straddle); window codes by exact bin at "
        f"required_margin 0: {window_table}. Window codes differing from their exact window: {mismatches}; largest "
        f"realized eigenvalue error {window_error:.3g} res on window cases and {straddle_error:.3g} res on straddle "
        f"cases (declared distance 0.1875 res). Unsound certifications: {unsound}. Unresolved beyond two "
        f"resolutions: {unresolved}. Certified within the band under the declared margin: "
        f"{certified_in_band_declared}. At required_margin 0, {len(window_certified)} of "
        f"{sum(family[i]['kind'] == 'window' for i in band)} window band cases were CERTIFIED_WITH_MARGIN, "
        f"{band_certified_unsound} band certificates unsound. Exactly negative definite window band cases: "
        f"{band_nd}, of which {band_nd_inconclusive} inconclusive ({refusal_rate:.0%}) and {band_nd_certified} "
        f"certified. Straddle cases within {straddle_distance:.2g} res of -res: {beyond_certified} of {len(beyond)} "
        f"exactly beyond -res and {inside_certified} of {len(inside)} exactly inside were certified, the rest "
        f"inconclusive ({straddle_other} other codes; which ones resolve is decided by rounding). MARGIN_LOW rule "
        f"mismatches: {margin_low_mismatch}. Conclusion: "
        + ("every certificate is exactly sound, every window case gets the code its exact position predicts and the "
           "sign is resolved beyond two resolutions" if unsound == 0 and unresolved == 0 and mismatches == 0
           else "the soundness, window or resolution property failed")
        + ("; the declared margin of three resolutions keeps the band out of CERTIFIED_WITH_MARGIN"
           if certified_in_band_declared == 0 else "; the declared margin did not keep the band out of "
                                                   "CERTIFIED_WITH_MARGIN")
        + ("; without a declared margin the band is not uniformly inconclusive, which refutes the stronger candidate "
           "statement (the certifications are sound)." if window_certified else
           "; without a declared margin no window band case was certified."))
    fields["uncertainty"] = ("Exact bins and windows are exact and identical on every BLAS kernel. Window codes "
                             "follow from them because the realized eigenvalue error stays below the declared "
                             "distance of 3/16 res; margin ratios carry that error and are retained in the "
                             "artifact. Which straddle cases resolve depends on last-bit rounding and may differ "
                             "between BLAS kernels; it is reported in the numerical result and the artifact, not "
                             "compared.")
    return _finish(fields, findings, PROVIDER_FILES, identity)


# T108 ------------------------------------------------------------------------

def margin_grid(margin, res):
    """Sorted nonnegative required margins around the resolution and the observed margin."""
    grid = {0.0, 0.5 * res, res, 2.0 * res, 1e300}
    if margin > 0.0:
        grid |= {0.5 * margin, math.nextafter(margin, 0.0), margin, math.nextafter(margin, math.inf), 2.0 * margin}
    return sorted(value for value in grid if value >= 0.0 and math.isfinite(value))


def monotonicity_violations(sequence):
    """Count property violations along one case's increasing required-margin sequence of verdict dicts.

    The code properties apply to any sequence; meets_required_margin and inequality_certified are checked
    only where the sequence carries them (the runtime's own fields).
    """
    passing = [step["code"] == "CERTIFIED_WITH_MARGIN" for step in sequence]
    codes = {step["code"] for step in sequence}
    # A non-certifying code is decided before the margin is consulted, so it must hold for every margin.
    changed = bool(codes - R.CERTIFYING) and len(codes) > 1
    result = {"passing_regained": sum(1 for a, b in zip(passing, passing[1:]) if b and not a),
              "noncertifying_code_changed": int(changed)}
    if "meets_required_margin" in sequence[0]:
        meets = [step["meets_required_margin"] for step in sequence]
        result["meets_regained"] = sum(1 for a, b in zip(meets, meets[1:]) if b and not a)
        result["inequality_changed"] = int(len({s["inequality_certified"] for s in sequence}) > 1)
    return result


@task("T108", changed_files=PROVIDER_FILES, regression_tests=(_node("test_t108_margin_monotonicity"),
                                                                _node("test_documented_rule_is_monotone"), PARTIAL_TEST,
                                                                NEXT_STEP_TEST))
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
        NEXT_STEPS["T108"],
        ["passing verdict regained at a larger margin", "meets_required_margin regained", "non-certifying code "
         "changed by the margin", "inequality_certified changed by the margin", "threshold not at margin",
         "invalid margin accepted"],
        ["A finite family cannot prove the property; the proof is the decision-order argument (analytic finding)."])
    family = near_threshold_family(108, 24) + razor_family(1081, 8)
    documented = []
    for member in family:
        # The transcription yields codes only; its margin fields would be the defining formulas read back.
        info = R.documented_code(member["A"], member["P"], member["x"])
        steps = [dict(code=R.documented_code(member["A"], member["P"], member["x"], required_margin=r)["code"])
                 for r in margin_grid(info["margin"], info["resolution"])]
        documented.append(monotonicity_violations(steps))
    documented_total = {key: sum(v[key] for v in documented) for key in documented[0]}
    offline = [
        finding("Monotonicity of the verdict in the declared margin follows from the decision order", "mathematical",
                "margin enters only as MARGIN_LOW iff margin <= r and meets = margin > max(r, res)",
                {"derivation": "runtime.verdict at the pinned commit: required_margin is compared only after the "
                               "resolution test; both comparisons are monotone in r (docs/lab/LYAPUNOV.md, T108)"},
                tolerance=EXACT_TOL, uncertainty=ANALYTIC),
        finding("The CIW transcription of the documented decision order never regains a passing code and never "
                "changes a non-certifying code as the required margin grows on the same margin grids", "numerical",
                documented_total,
                {"generator": {"name": "near_threshold_family + razor_family", "seed": 108},
                 "checks": [_check("code-property violations of the transcribed order (passing code regained, "
                                   "non-certifying code changed)", sum(documented_total.values()), 0.0,
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
                                                                _node("test_numpy_misreads_exact_jordan_block"),
                                                                UNCONDITIONAL_TEST, PARTIAL_TEST, NEXT_STEP_TEST))
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
        NEXT_STEPS["T109"],
        ["certificate without exact validity", "solver disagreement beyond conditioning", "transient bound violated",
         "numpy eigenvalue of an exactly defective matrix not misplaced beyond rounding (its sign is retained in the "
         "artifact)", "P = I threshold misplaced",
         "solver refusal outside the documented gates", "solver refuses a plant with a valid certificate "
         "(searched as counterexample)"],
        ["Jordan and clustered plants are synthetic stress cases, not identified plant models.",
         "The independent solver is SciPy when installed. Without SciPy the CIW Kronecker solve stands in, but it "
         "solves the same Kronecker system with numpy.linalg.solve as PLSR's solve_lyapunov, so agreement with it "
         "is then recorded as an ordinary check, not an independent one."])
    cases = adversarial_cases()
    jordan = [c for c in cases if c["group"] == "Jordan"]
    misplacement = [abs(c["numpy_abscissa"] + c["exact_spectrum"][0]) / (float(np.finfo(float).eps)
                                                                         * float(np.max(np.abs(c["A"]))))
                    for c in jordan]
    # Whether numpy's abscissa of an exactly Hurwitz Jordan plant comes out >= 0 depends on the LAPACK build: it
    # is retained in the artifact (and the prose count), not checked.
    jordan_wrong = [c for c in jordan if c["numpy_abscissa"] >= 0.0]
    offline = [finding(
        "numpy.linalg.eigvals misplaces the exact eigenvalue -lambda of every defective test matrix by far more "
        "than machine precision", "numerical",
        {"cases": len(jordan), "misplaced_beyond_1e3_eps": sum(m > 1e3 for m in misplacement)},
        {"generator": {"name": "adversarial_cases", "seed": 109},
         "checks": [_check("exactly defective cases whose numpy abscissa lies within 1e3 eps max|A| of the exact "
                           "eigenvalue -lambda (A = T J T^-1 verified exactly)",
                           sum(m <= 1e3 for m in misplacement), 0.0, kind="analytic")]},
        tolerance=EXACT_TOL,
        uncertainty=_platform(0.0, f"the counts are robust: the smallest misplacement is {min(misplacement):.3g} "
                                   "eps max|A|, orders of magnitude above the 1e3 threshold, while numpy eigenvalues "
                                   "of defective matrices vary at the eps^(1/n) level between builds"))]
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
                # The gate, not the message: the residual it quotes is rounding noise of an ill-conditioned solve
                # and differs between BLAS kernels (the full message stays in the artifact row).
                gate = next((g for g in SOLVER_GATES if row["solve_error"].startswith(g)), None)
                refused_with_certificate.append({"name": case["name"], "solver_gate": gate,
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
                                                       "independent_solver": independent[0]["implementation"],
                                                       "numpy_abscissa_nonnegative_on_exactly_hurwitz_jordan": [
                                                           c["name"] for c in jordan_wrong]}))
    ctx.artifact_text("solver-agreement.svg", svg.line_plot(
        [("PLSR vs independent P", [c for c, _, _ in agreement], [max(r, 1e-18) for _, r, _ in agreement]),
         ("n^2 u cond(P) for n = 2", [c for c, _, _ in agreement], [4 * R.U * max(c, 1.0) for c, _, _ in agreement])],
        title="T109 relative difference of Lyapunov solutions", xlabel="condition number of P",
        ylabel="||P_PLSR - P_ind|| / ||P_ind||", logx=True, logy=True))
    checker = {"implementation": independent[0]["implementation"], "revision": independent[0]["revision"]}
    max_relative = max(r for _, r, _ in agreement)
    max_normalised = max(m for _, _, m in agreement)
    beyond_bound = sum(m > 10.0 for _, _, m in agreement)
    jordan_rows = [r for r in rows if r["group"] == "Jordan"]
    jordan_solved = sum(r["solve"] == "P returned" for r in jordan_rows)
    witness_extra, witness_tolerance = {}, EXACT_TOL
    witness_uncertainty = _roundoff(0.0, "no refused plant had a valid certificate: the count is exact")
    if refused_with_certificate:
        witness = refused_with_certificate[0]
        witness_extra["counterexample"] = {
            "statement": "PLSR's solve_lyapunov refuses only plants for which no valid quadratic certificate is "
                         "available in float64", "witness": witness}
        # The witness's certificate condition is the one rounding-level number compared: a backward-stable solve
        # moves P by about u ||P||, so its smallest eigenvalue and cond(P) move by about u cond(P) relative.
        size = next(c["A"].shape[0] for c in cases if c["name"] == witness["name"])
        first_order = R.U * witness["certificate_condition"]
        witness_tolerance = {"abs": 0.0, "rel": size * size * first_order}
        witness_uncertainty = _roundoff(first_order, (
            f"relative change of the witness's certificate condition cond(P) = "
            f"{witness['certificate_condition']:.3g} under a solve perturbation of size u ||P|| (Weyl: u cond(P)); "
            "three OpenBLAS kernels (SkylakeX, Haswell, Sandybridge) gave conditions within 4.2e-5 of each other. "
            f"The regression tolerance n^2 u cond(P) = {size * size * first_order:.2g} is n^2 times that size and "
            "about twenty times the measured spread; the count and the plants are exact, and exact validity of the "
            "independent P is decided in rational arithmetic"))
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
        finding("PLSR Lyapunov solutions agree with the reference solver (SciPy's Bartels-Stewart when installed) to "
                "within 10 n^2 u cond(P) on every solved adversarial case", "numerical",
                {"cases": len(agreement), "beyond_bound": beyond_bound},
                dict({"provider": base}, **_solver_agreement(
                    _check("max over solved cases of the relative Frobenius difference divided by n^2 u cond(P) "
                           "(forward-error bound of a backward-stable solve)", max_normalised, 10.0, "le",
                           kind="analytic"), identity, checker)),
                tolerance=EXACT_TOL,
                uncertainty=_roundoff(max_relative, f"largest observed relative difference (normalised by n^2 u "
                                                    f"cond(P): at most {max_normalised:.3g}; it may vary by a small "
                                                    "factor between BLAS builds and is not a compared value")),
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
                tolerance=witness_tolerance, uncertainty=witness_uncertainty, **witness_extra),
    ] + offline
    fields["numerical_result"] = (
        f"{certified} certifying verdicts, {violations} without an exactly valid certificate. Solver agreement on "
        f"{len(agreement)} solved cases: max relative difference {max_relative:.2e}, max normalised by n^2 u cond(P) "
        f"{max_normalised:.3g} ({independent[0]['implementation']}). Transient peak / sqrt(cond P): max "
        f"{max(t['ratio'] for t in transient):.3f} over {len(transient)} certified K. P = I threshold mismatches: "
        f"{identity_threshold_mismatch}. Jordan: numpy abscissa >= 0 in {len(jordan_wrong)} of {len(jordan)} exactly "
        f"Hurwitz cases (build-dependent); PLSR's solver returned P for {jordan_solved} and refused "
        f"{len(jordan_rows) - jordan_solved} "
        f"({ungated} refusals outside the documented gates, {solver_invalid} returned P invalid); "
        f"refused plants with an exactly valid independent certificate that PLSR's verdict certifies: "
        f"{len(refused_with_certificate)}. Conclusion: "
        + ("every PLSR certificate on these plants is exactly valid" if violations == 0 and solver_invalid == 0
           else "some PLSR certificates are not exactly valid")
        + ("; the solver's refusals are conservative, including plants that do have a valid certificate"
           if refused_with_certificate else "; no refused plant had a valid certificate from the independent solver")
        + "; numpy misplaces the exact defective eigenvalues far beyond rounding, which is why a sign read off its "
        "abscissa (retained in the artifact) is not a stability test for these plants.")
    fields["uncertainty"] = ("Exact checks are exact. Solver agreement scales with cond(P) (retained per case). "
                             "numpy eigenvalues of defective matrices vary at the eps^(1/n) level between builds, "
                             "so the count of wrong signs may change and is retained in the artifact only; the "
                             "misplacement beyond 1e3 eps does not. Which documented gate refuses an exactly "
                             "defective plant, and the residual a refusal quotes, are rounding outcomes that differ "
                             "between BLAS kernels; they are retained per plant in the artifact, and the "
                             "counterexample records the gate. Its certificate condition is compared within "
                             "n^2 u cond(P).")
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


@task("T110", changed_files=PROVIDER_FILES,
      regression_tests=(_node("test_t110_time_interpretation"), PARTIAL_TEST, NEXT_STEP_TEST))
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
        NEXT_STEPS["T110"],
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


SCALAR_GROUPS = ("route family", "near threshold")


def scalar_gate_samples(family, near, per_plant=8, seed=1112):
    """States at which the scalar gate is compared with the eigenvalue route and exact x^T M x.

    Each route plant with P = I at its own state and seven seeded ones; each of T107's near-threshold forms
    (with its P) at the computed top eigenvector of its decrease matrix, where x^T M x is about max eig(M) |x|^2
    and the scalar gate meets the resolution, and at three seeded states.
    """
    rng = R.generator(seed)
    samples = []
    for i, member in enumerate(family):
        n = member["A"].shape[0]
        for x in [member["x"]] + [rng.normal(size=n) for _ in range(per_plant - 1)]:
            samples.append({"group": "route family", "plant": i, "A": member["A"], "P": np.eye(n),
                            "time": member["time"], "x": np.asarray(x, dtype=float)})
    for i, member in enumerate(near):
        top = np.linalg.eigh(R.decrease_matrix(member["A"], member["P"]))[1][:, -1]
        for x in [top] + [rng.normal(size=member["n"]) for _ in range(3)]:
            samples.append({"group": "near threshold", "plant": i, "A": member["A"], "P": member["P"],
                            "time": "continuous", "x": np.asarray(x, dtype=float),
                            "exact_class": member["exact_class"]})
    return samples


def scalar_gate_rows(samples, results):
    """Per sample: PLSR's scalar gate (NOT_CERTIFIED), its eigenvalue route and the exact signs."""
    forms, classes, rows = {}, {}, []
    for k, sample in enumerate(samples):
        key = (sample["group"], sample["plant"])
        if key not in forms:
            forms[key] = R.exact_form(sample["A"], sample["P"], sample["time"])
            classes[key] = sample.get("exact_class") or R.exact_class(forms[key])
        result = results[f"scalar-gate{k}"]
        verdict = result.get("sample") or {}
        code = _code(result)
        exact = R.exact_quadratic(sample["x"], forms[key])
        rows.append({"group": sample["group"], "plant": sample["plant"], "code": code,
                     "scalar_positive": code == "NOT_CERTIFIED",
                     "eigen_positive": bool(verdict) and verdict["max_decrease"] > verdict["resolution"],
                     "exact_sample_positive": exact > 0, "exact_class": classes[key],
                     "exact_x_M_x_sign": (exact > 0) - (exact < 0)})
    return rows


def scalar_gate_summary(rows):
    """Violation counts and per-group agreement shares between the scalar, eigenvalue and exact routes."""
    violations = {
        "not_certified_without_exact_increase": sum(r["scalar_positive"] and not r["exact_sample_positive"]
                                                    for r in rows),
        "not_certified_without_eigen_route": sum(r["scalar_positive"] and not r["eigen_positive"] for r in rows),
        "eigen_route_on_exactly_negative_definite_form": sum(r["eigen_positive"]
                                                             and r["exact_class"] == "negative_definite"
                                                             for r in rows),
        "certified_where_exact_x_M_x_positive": sum(r["code"] in R.CERTIFYING and r["exact_sample_positive"]
                                                    for r in rows)}
    agreement, shares, samples = {}, {}, {}
    for group in SCALAR_GROUPS:
        members = [r for r in rows if r["group"] == group]
        samples[group] = len(members)
        agreement[group] = {
            "scalar_vs_exact_sample_sign": sum(r["scalar_positive"] == r["exact_sample_positive"]
                                               for r in members) / len(members),
            "eigen_vs_exact_class": sum(r["eigen_positive"] == (r["exact_class"] == "has_positive_eigenvalue")
                                        for r in members) / len(members),
            "scalar_vs_eigen": sum(r["scalar_positive"] == r["eigen_positive"] for r in members) / len(members)}
        shares[group] = sum(r["scalar_positive"] for r in members) / len(members)
    route = [r for r in rows if r["group"] == "route family"]
    robust = {"route_scalar_vs_exact_sign": sum(r["scalar_positive"] != r["exact_sample_positive"] for r in route),
              "route_eigen_vs_exact_class": sum(r["eigen_positive"] != (r["exact_class"] == "has_positive_eigenvalue")
                                                for r in route)}
    return {"samples": samples, "violations": violations, "agreement": agreement, "not_certified_share": shares,
            "route_disagreements": robust}


@task("T111", changed_files=PROVIDER_FILES,
      regression_tests=(_node("test_t111_routes"), PARTIAL_TEST, NEXT_STEP_TEST))
def quadratic_routes(ctx):
    fields = _fields(
        "The PLSR matrix route (Lyapunov solve, then the sign of max eig of the decrease form beyond the "
        "resolution) agrees with independent references -- numpy eigenvalues and SciPy's Bartels-Stewart "
        "Lyapunov solvers; PLSR's scalar gate (NOT_CERTIFIED when x^T M x > res |x|^2) fires only where the exact "
        "x^T M x is positive and the matrix route finds max eig(M) > res, on margin-separated plants and on "
        "near-threshold forms alike; and the scalar quadratic route (the sign of x^T M x at sampled states) cannot "
        "certify definiteness and misses thin positive cones.",
        "Rayleigh: x^T M x <= max eig(M) |x|^2 for every x, with equality only on the top eigenvector, so sampled "
        "negativity never implies negative definiteness. An unstable A has v*(A + A^T)v = 2 Re(lambda)|v|^2 > 0 "
        "for an eigenvector v, so P = I can never certify it. If the resolution bounds the error of the computed "
        "x^T M x (|x^T E x| <= ||E||_2 |x|^2), then x^T M x > res |x|^2 in float64 implies an exactly positive "
        "x^T M x and, by Rayleigh, max eig(M) > res. For n = 1 the decrease form is 2 a p (continuous), so the "
        "verdict must follow sign(a) whenever 2|a|p exceeds the resolution.",
        ["30 continuous plants n = 2..5 (PCG64 seed 111) with |spectral abscissa| > 0.05",
         "20 discrete plants n = 2..4 with |spectral radius - 1| > 0.05", "Thin-cone plant A = diag(-0.5, 5e-7), "
         "P = I (decrease form diag(-1, 1e-6)) with 64 random unit states (seed 1111)",
         f"Scalar plants a in {list(T111_SCALARS)}, p = 1",
         "Scalar-gate samples: every route plant with P = I at its own state and seven more (seed 1112), and each "
         "of T107's 102 near-threshold forms (seeds 107, 1071) at the computed top eigenvector of its decrease "
         "matrix and three seeded states"],
        "PLSR solve_lyapunov outcome (P or refusal) and verdicts with the returned P and with P = I; SciPy (or the "
        "CIW Kronecker solve when SciPy is absent) Lyapunov P per time convention; numpy eigenvalues; PLSR "
        "verdicts at 64 sampled states; per scalar-gate sample PLSR's code, max eig(M) and resolution against "
        "the exact x^T M x and exact class in rationals.",
        "PLSR P equals the independent P to solver accuracy in each convention; solve_lyapunov returns P exactly "
        "for the numpy-stable plants and otherwise raises ValueError; the verdict certifies every numpy-stable "
        "plant with its own P and no numpy-unstable plant with P = I; NOT_CERTIFIED only where the exact x^T M x "
        "is positive and PLSR's max eig(M) exceeds the resolution; sampled scalar decrease never contradicts the "
        "matrix route and cannot stand in for it.",
        "Phase 1: PLSR solves, P = I verdicts, scalar-gate, thin-cone and n = 1 verdicts; phase 2: verdicts with "
        "PLSR's P. Compare routes case by case and per sample with exact rationals.",
        NEXT_STEPS["T111"],
        ["solver disagreement", "solver returns P for an unstable plant", "solver refuses a stable plant",
         "solver refusal other than ValueError", "verdict certifies an unstable plant", "NOT_CERTIFIED where the "
         "exact x^T M x is not positive", "NOT_CERTIFIED without max eig(M) > res", "matrix route calls an exactly "
         "negative definite form indefinite", "scalar route claims definiteness", "scalar sign not followed for "
         "n = 1"],
        ["Route plants with margins below 0.05 are excluded from the solver and verdict comparisons; near-marginal "
         "forms enter only through T107's near-threshold family in the scalar-gate comparison.",
         "The independent solver is SciPy when installed. Without SciPy the CIW Kronecker solve stands in, but it "
         "solves the same Kronecker system with numpy.linalg.solve as PLSR's solve_lyapunov, so agreement with it "
         "is then recorded as an ordinary check, not an independent one."])
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
    gate_samples = scalar_gate_samples(family, ctx.memo("lyapunov:boundary-family", boundary_family))
    phase1 = [{"id": f"solve{i}", "op": "solve", "A": _mat(m["A"]), "time": m["time"]} for i, m in enumerate(family)]
    phase1 += [_verdict_case(f"scalar-gate{k}", g["A"], g["P"], g["x"], time=g["time"])
               for k, g in enumerate(gate_samples)]
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
    normalised = {"continuous": [], "discrete": []}
    solve_mismatch, own_mismatch, identity_unsound, errors = 0, 0, 0, {}
    for i, member in enumerate(family):
        solve_mismatch += (i in solved) != member["stable"]
        if i in solved:
            if member["stable"]:
                own_mismatch += _code(results[f"own{i}"]) != "CERTIFIED_WITH_MARGIN"
            difference = float(np.linalg.norm(solved[i] - independent[i][0]) / np.linalg.norm(independent[i][0]))
            eig = np.linalg.eigvalsh(solved[i])
            n = solved[i].shape[0]
            relative[member["time"]].append(difference)
            normalised[member["time"]].append(difference / (n * n * R.U * max(float(eig[-1] / eig[0]), 1.0)))
        else:
            own_mismatch += member["stable"]
            errors[str(i)] = results[f"solve{i}"]["error"]
        identity_unsound += not member["stable"] and _code(results[f"identity{i}"]) in R.CERTIFYING
    # One refusal check per plant the reference calls unstable, selected before looking at the solver's outcome.
    refusals = [_refusal(f"solve_lyapunov for {member['time']} plant {i} (numpy: unstable)", "raises ValueError",
                         "none" if i in solved else _code(results[f"solve{i}"]))
                for i, member in enumerate(family) if not member["stable"]]
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
    gate_rows = scalar_gate_rows(gate_samples, first["results"])
    gate = scalar_gate_summary(gate_rows)
    ctx.artifact_json("routes.json", R.jsonable({"relative_difference": relative,
                                                  "normalised_difference": normalised, "thin_cone_codes": thin_codes,
                                                  "scalar_codes": scalar_codes, "solve_errors": errors,
                                                  "identity_codes": identity_codes,
                                                  "independent_solvers": [list(entry) for entry in solvers],
                                                  "scalar_gate": dict(gate, rows=gate_rows)}))
    findings = []
    for time in ("continuous", "discrete"):
        name, revision = next((n, r) for t, n, r in solvers if t == time)
        values = relative[time]
        findings.append(finding(
            f"PLSR {time}-time Lyapunov solutions agree with the reference solver (SciPy's Bartels-Stewart when "
            "installed) to within 10 n^2 u cond(P) on the route family", "numerical",
            {"solves": len(values), "max_relative_difference": max(values)},
            dict({"provider": base}, **_solver_agreement(
                _check("max over solves of the relative Frobenius difference divided by n^2 u cond(P) (forward-error "
                       "bound of a backward-stable solve)", max(normalised[time]), 10.0, "le", kind="analytic"),
                identity, {"implementation": name, "revision": revision})),
            tolerance={"abs": 1e-9, "rel": 0.0},
            uncertainty=_roundoff(max(values), "largest observed relative difference")))
    violations = gate["violations"]
    findings.append(finding(
        "PLSR's scalar NOT_CERTIFIED gate fires only where the exact x^T M x is positive and its eigenvalue route "
        "finds max eig(M) above the resolution, on the route family and on T107's near-threshold forms",
        "numerical",
        {"samples": gate["samples"], "violations": violations, "agreement": gate["agreement"],
         "not_certified_share": gate["not_certified_share"], "route_disagreements": gate["route_disagreements"]},
        {"provider": base,
         "independent_check": _independent(
             _check("NOT_CERTIFIED samples whose exact x^T M x (rationals) is not positive",
                    violations["not_certified_without_exact_increase"], 0.0), identity),
         "checks": [
             _check("NOT_CERTIFIED samples whose PLSR max eig(M) does not exceed the resolution",
                    violations["not_certified_without_eigen_route"], 0.0, kind="invariant"),
             _check("samples where PLSR's eigenvalue route (max eig(M) > res) meets an exactly negative definite form",
                    violations["eigen_route_on_exactly_negative_definite_form"], 0.0),
             _check("certifying verdicts at samples whose exact x^T M x is positive",
                    violations["certified_where_exact_x_M_x_positive"], 0.0),
             _check("route-family samples whose NOT_CERTIFIED decision differs from the exact sign of x^T M x",
                    gate["route_disagreements"]["route_scalar_vs_exact_sign"], 0.0),
             _check("route-family samples whose eigenvalue-route decision differs from the exact class",
                    gate["route_disagreements"]["route_eigen_vs_exact_class"], 0.0)]},
        tolerance={"abs": 0.05, "rel": 0.0},
        uncertainty=_platform(0.05, "counts are exact (abs 0.05 admits no integer change); near-threshold agreement "
                                    "shares move with last-bit rounding by a few of the 408 samples")))
    findings += [
        finding("PLSR's solve_lyapunov returns a P exactly for the plants numpy's eigenvalues call stable and raises "
                "ValueError for the others", "numerical",
                {"plants": len(family), "mismatches": solve_mismatch, "refused": len(errors)},
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
        f"{len(errors)} refusals ({sum(not family[int(i)]['stable'] for i in errors)} of them numpy-unstable "
        f"plants). Verdicts: {own_mismatch} stable plants not certified "
        f"with their own P, {identity_unsound} unstable plants certified with P = I. Thin cone: "
        f"{thin_scalar_negative}/{len(samples)} sampled scalar decreases negative; PLSR codes {thin_codes}. Scalar "
        f"plants: {scalar_codes}. Scalar gate over {sum(gate['samples'].values())} samples: violations "
        f"{violations}; agreement shares (scalar vs exact sample sign, eigenvalue route vs exact class, scalar vs "
        f"eigenvalue route) {gate['agreement']}; NOT_CERTIFIED shares {gate['not_certified_share']}. Conclusion: "
        + ("the matrix routes agree with each other on margin-separated plants" if routes_agree else
           "the matrix routes disagree on some margin-separated plants")
        + ("; the scalar gate is sound against exact x^T M x and implies the eigenvalue route" if
           not any(violations.values()) else "; the scalar gate contradicts exact x^T M x or the eigenvalue route")
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
                                                                _node("test_research_tasks_without_optional_modules"),
                                                                NEXT_STEP_TEST))
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
        NEXT_STEPS["T112"],
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
                        PARTIAL_TEST, NEXT_STEP_TEST))
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
        "Adapter outputs (samples or host status), the serialised samples the adapter emits for forwarding, PLSR "
        "codes for forwarded samples, the host's window decision, and PLSR's treatment of host-owned codes.",
        "Emitted samples carry only the four sample fields with numeric values; host statuses are host-owned codes; "
        "forwarded codes equal the CIW transcription of the documented order; no window whose guard interval "
        "leaves the box is accepted; theta estimates lie within 3 standard errors of the synthetic truth.",
        "Run the adapter on every window, serialise the samples it emits and search them for envelope metadata, "
        "evaluate every forwarded sample with PLSR, apply the host's window rule, and compare with the "
        "documented decision order.",
        NEXT_STEPS["T113"],
        ["metadata in the adapter's emitted samples", "non-numeric or extra sample field",
         "host status forwarded to the "
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
    emitted = []
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
                kernel_cases.append(_affine_case(f"w{i}:{role}", A0, [A1], box, P, sample["x"], sample["theta"],
                                                 time="discrete"))
            # The search runs on what the adapter emits for forwarding, not on payloads rebuilt from its numbers.
            leaks += X.metadata_leaks(outcome["samples"], window["envelope"])
            emitted.append((window, outcome["samples"]))
            estimate_misses += abs(stats["theta"] - window["theta_true"]) > 3.0 * stats["theta_se"]
            row.update(forwarded=True, guard_interval=[expected["lower"], expected["upper"]],
                       interval_inside_box=X.ADAPTER_BOX[0] <= expected["lower"]
                       and expected["upper"] <= X.ADAPTER_BOX[1],
                       estimate_inside_box=X.ADAPTER_BOX[0] <= stats["theta"] <= X.ADAPTER_BOX[1])
        rows.append(row)
    host = {r["window"]: r["host_status"] for r in rows if r["host_status"]}
    forwarded = [r for r in rows if r.get("forwarded")]
    near_bound = [r["window"] for r in forwarded if r["estimate_inside_box"] and not r["interval_inside_box"]]
    # Positive control: adapter output with a planted sensor id must be flagged by the same detector.
    control_window, control_samples = emitted[0]
    planted = dict(control_samples, estimate=dict(control_samples["estimate"],
                                                  sensor={"id": control_window["envelope"]["sensor_id"]}))
    control = X.metadata_leaks(planted, control_window["envelope"])
    largest_se = max(r["statistics"]["theta_se"] for r in forwarded)
    offline = [
        finding("Samples the adapter emits for the kernel carry only the plsr-sample-v1 fields with numeric values, "
                "and no envelope metadata", "computational_pipeline",
                {"emitted_windows": len(emitted), "samples": len(kernel_cases), "metadata_leaks": leaks,
                 "malformed_samples": bad_fields},
                {"checks": [_check("envelope keys or strings found in the serialised adapter output", len(leaks), 0.0,
                                   kind="invariant"),
                            _check("positive control: leaks flagged in adapter output with a planted sensor id",
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

T114_FILES = T113_FILES


def _monitor_finding(claim, rows, source, basis, same_spec=None):
    """Monitor codes over the scan without and with the declared level, against the exact V(x) > c decision."""
    without = sorted({r["code_without_level"] for r in rows})
    with_level = sorted({r["code_with_level"] for r in rows})
    mismatch = sum((r["code_with_level"] == "OUTSIDE_LEVEL_SET") != r["exact_exceeds_level"] for r in rows)
    constant = _check(f"distinct {source} codes over the scan without a level set", len(without), 1.0, "le",
                      kind="invariant")
    exact = _check(f"states whose {source} OUTSIDE_LEVEL_SET decision differs from exact V(x) > c (rationals)",
                   mismatch, 0.0)
    if "provider" in basis:
        basis = dict(basis, independent_check=_independent(exact, basis["identity"]),
                     checks=[constant] + ([same_spec] if same_spec is not None else []))
        basis.pop("identity")
    else:
        basis = dict(basis, checks=[constant, exact])
    return finding(claim, "numerical",
                   {"states": len(rows), "codes_without_level": without, "codes_with_level": with_level,
                    "level_mismatches": mismatch, "exceeding_states": sum(r["exact_exceeds_level"] for r in rows)},
                   basis, tolerance=EXACT_TOL,
                   uncertainty=_roundoff(0.0, "exact rationals decide V > c; states lie between 1e-8 and 1e8 of the "
                                              "level radius"))


def _abort_finding(claim, rows, basis):
    produced = sorted({r["code_with_level"] for r in rows})
    unexpected = sorted(set(produced) - set(X.SERVO_EXPECTED_CODES))
    unreachable = sorted(set(X.SERVO_ABORT_CODES) - set(produced))
    return finding(claim, "computational_pipeline",
                   {"abort_codes": list(X.SERVO_ABORT_CODES), "expected_codes": list(X.SERVO_EXPECTED_CODES),
                    "produced": produced},
                   dict(basis, checks=[
                       _check("abort codes never produced by the monitor scan", len(unreachable), 0.0,
                              kind="invariant"),
                       _check("produced codes outside the expected set", len(unexpected), 0.0, kind="invariant")]),
                   tolerance=EXACT_TOL, uncertainty=EXACT)


@task("T114", changed_files=T114_FILES, regression_tests=(_node("test_t114_servo_pilot_spec"),
                                                            _node("test_t114_level_set_and_monitor"),
                                                            _node("test_level_set_extent_is_checked_independently"),
                                                            _node("test_research_tasks_without_optional_modules"),
                                                            PARTIAL_TEST, NEXT_STEP_TEST))
def servo_pilot(ctx):
    fields = _fields(
        "A non-production servo-axis pilot can be specified so that the Lyapunov monitor's scope, data, abort "
        "criteria and lack of authority are explicit, the declared level set lies inside the operating envelope, "
        "every runtime abort trigger can actually fire for the declared monitor configuration in the pinned "
        "runtime, and the offline certificate check passes at every grid inertia before any powered test.",
        "Axis J theta'' = -b theta' + Kt u with PD state feedback designed for 20 Hz, damping 0.7 at nominal J; "
        "exact ZOH at Ts = 1 ms; closed loop A_cl(J) = Phi(J) - Gamma(J) K on a 9-point grid over J +- 30 %; "
        "common P from the discrete Lyapunov equation at nominal J. Online: PLSR on the nominal model with a "
        "declared level c such that {V <= c} lies inside the operating envelope (max |x_i| over the ellipsoid is "
        "sqrt(c (P^-1)_ii)); for a fixed model the decrease form's sign is state-independent, so the code depends "
        "on the state only through the level gate.",
        ["Placeholder parameters (not identified): J = 2e-3 kg m^2 +- 30 %, b = 1e-3 N m s, Kt = 0.1 N m/A, "
         "Ts = 1 ms", "Weightings Q = I and unit-balanced Q = diag(omega^2, 1), omega = 2 pi 20 rad/s",
         "Operating envelope |angle error| <= 0.05 rad, |rate| <= 2 rad/s (placeholder)",
         "Monitor scan: 200 seeded states (PCG64 seed 1141) with V(x) = r^2 c, r = 10^U(-8, 8)",
         "Level-set boundary sampled at 3600 directions"],
        "Exact rational class of A_cl(J)^T P A_cl(J) - P at each grid inertia; ZOH exponential against its closed "
        "form and an independent exponential; the level set's extent in exact rationals and on the sampled "
        "boundary; PLSR's monitor codes over the scan (and the CIW transcription of the documented decision "
        "order) against the exact V(x) > c decision.",
        "The unit-balanced P is exactly valid at every grid inertia; {V <= c} lies inside the envelope; PLSR's "
        "monitor code is constant without a level set and follows exact V > c with it; every abort code is "
        "produced by the scan; authority and physical claims remain not_established.",
        "Build the sampled-data models, choose P, check each grid point exactly, verify the exponential, derive the "
        "level set and check its extent, evaluate the monitor configuration at the scan states with the pinned "
        "runtime and with the transcription, and retain the specification (JSON and Markdown).",
        NEXT_STEPS["T114"],
        ["certificate invalid on part of the interval", "exponential inaccurate", "monitor code independent of the "
         "state even with a level set", "abort trigger that the configuration cannot produce",
         "level set outside the envelope", "runtime codes off the documented order", "authority or safety claimed"],
        ["All parameters are placeholders; the grid is evidence on 9 inertias, not a proof over the interval.",
         "The monitor is evaluated on the declared nominal model only; the inertia interval between grid points "
         "and the other grid models are covered by the offline check, not by the monitor.",
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
    transcribed = X.monitor_scan(nominal["A_cl"], P, level)
    extent = X.level_set_extent(P, level)
    boundary_extent = X.level_set_boundary_extent(P, level)
    balanced, plain = classes["Q = diag(omega^2, 1)"], classes["Q = I"]
    transcription = "CIW transcription of the documented decision order"
    offline = [
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
        finding("The declared level set {V <= c} lies inside the operating envelope", "numerical",
                {"exact_largest_squared_extent_ratio": float(extent), "sampled_boundary_extent_ratio": boundary_extent},
                {"generator": {"name": "level_set_boundary_extent", "seed": None},
                 "checks": [_check("largest c (P^-1)_ii / e_i^2 in exact rationals ((P^-1)_ii = P_jj / det P; "
                                   "max x_i^2 over the ellipsoid)", float(extent), 1.0, "le"),
                            _check("largest |x_i| / e_i over 3600 sampled points of the boundary x^T P x = c",
                                   boundary_extent, 1.0, "le", kind="invariant")]},
                tolerance={"abs": 1e-9, "rel": 0.0},
                uncertainty=_roundoff(1e-12, "the exact ratio is exact; sampled boundary points carry float64 "
                                             "rounding and approach the maximum from below")),
        _monitor_finding("By the " + transcription + ", the monitor's code is the same at every scanned state "
                         "without a level set and follows the exact V(x) > c decision with the declared level",
                         transcribed, "transcribed", {"generator": {"name": "monitor_states", "seed": 1141}}),
        _abort_finding("By the " + transcription + ", every runtime code named as an abort trigger is produced by "
                       "the declared monitor configuration, and no code outside the expected set",
                       transcribed, {"generator": {"name": "monitor_states", "seed": 1141}}),
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
    exponential = offline[2]["value"]
    cases = []
    for k, row in enumerate(transcribed):
        cases.append(_verdict_case(f"monitor{k}", nominal["A_cl"], P, row["x"], time="discrete"))
        cases.append(_verdict_case(f"monitor{k}:level", nominal["A_cl"], P, row["x"], time="discrete", level=level))
    summary = (f"K = {np.round(K, 6).tolist()}; balanced-Q grid classes {_counts(balanced)}; Q = I grid classes "
               f"{_counts(plain)}; ZOH exponential differs from its closed form by "
               f"{exponential['max_abs_difference_closed_form']:.2e}"
               + (f" and from {independent[1]} by {exponential['max_abs_difference_independent']:.2e}"
                  if independent is not None else " (no independent exponential applies to this defective matrix "
                                                  "here)")
               + f"; level c = {level:.6g}, largest extent/envelope ratio {math.sqrt(float(extent)):.6f} exactly and "
               f"{boundary_extent:.6f} on the sampled boundary")

    def spec(rows, source):
        monitor = {"level": level, "source": source,
                   "codes_without_level": sorted({r["code_without_level"] for r in rows}),
                   "codes_with_level": sorted({r["code_with_level"] for r in rows})}
        document = X.servo_spec(K, P, classes, monitor)
        ctx.artifact_json("servo-pilot-spec.json", R.jsonable(document))
        ctx.artifact_text("servo-pilot-spec.md", X.servo_spec_markdown(document))
        return monitor

    try:
        bridge = _bridge(ctx, cases)
    except _Unavailable as exc:
        monitor = spec(transcribed, transcription)
        ctx.artifact_json("monitor-scan.json", R.jsonable({"level": level, "transcription": transcribed}))
        fields["numerical_result"] = (
            summary + f"; by the transcription, monitor codes without level {monitor['codes_without_level']}, "
            f"with level {monitor['codes_with_level']}. The pinned runtime did not run.")
        fields["uncertainty"] = ("Exact rational classes on the declared float matrices; the interval between grid "
                                 "points is not covered by a proof; the level-set scan decides V > c exactly.")
        return _finish(fields, offline, T114_FILES, blocked=str(exc))
    identity, results = bridge["identity"], bridge["results"]
    runtime_rows = [dict(row, code_without_level=_code(results[f"monitor{k}"]),
                         code_with_level=_code(results[f"monitor{k}:level"])) for k, row in enumerate(transcribed)]
    off_order = sum(runtime[key] != row[key] for runtime, row in zip(runtime_rows, transcribed)
                    for key in ("code_without_level", "code_with_level"))
    monitor = spec(runtime_rows, "pinned PLSR runtime")
    ctx.artifact_json("monitor-scan.json", R.jsonable({"level": level, "runtime": runtime_rows,
                                                        "transcription": transcribed}))
    base = provider_basis(identity)
    findings = [
        _monitor_finding("PLSR's monitor code on the declared configuration is the same at every scanned state "
                         "without a level set and follows the exact V(x) > c decision with the declared level",
                         runtime_rows, "PLSR", {"provider": base, "identity": identity},
                         same_spec=_check("scan verdicts differing from the CIW transcription of the documented order "
                                          "(same specification)", off_order, 0.0, kind="analytic")),
        _abort_finding("PLSR produces every runtime code named as an abort trigger on the declared monitor "
                       "configuration, and no code outside the expected set", runtime_rows, {"provider": base}),
    ] + offline
    fields["numerical_result"] = (
        summary + f"; PLSR monitor codes without level {monitor['codes_without_level']}, with level "
        f"{monitor['codes_with_level']}, "
        f"{findings[0]['value']['level_mismatches']} level decisions differing from exact V > c, {off_order} scan "
        f"verdicts differing from the transcription; abort codes not produced by PLSR: "
        f"{sorted(set(X.SERVO_ABORT_CODES) - set(monitor['codes_with_level']))}.")
    fields["uncertainty"] = ("Exact rational classes on the declared float matrices; the interval between grid "
                             "points is not covered by a proof; the level-set scan decides V > c exactly.")
    return _finish(fields, findings, T114_FILES, identity)
