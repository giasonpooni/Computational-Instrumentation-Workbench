"""Evidence labels for computational-experiment findings.

Every finding carries exactly one label. The label is recomputed from the
finding's declared basis by this deterministic validator; an assistant, a
report author or a later analysis can only state a label that the basis
already supports. Analysis never upgrades evidence: a quantity derived from
other findings cannot claim more physical support than its weakest input.

Labels describe what supports a claim, not how interesting it is:

``analytic``
    Derived in closed form from declared assumptions; nothing was executed.
``synthetic``
    Computed from declared generated inputs without a passing reference check.
``numerically_verified``
    A stated numerical condition passed against an analytic, high-precision,
    invariant or self-convergence reference within a declared tolerance.
``provider_backed``
    Returned by a pinned external provider runtime with a recorded identity.
``hardware_measured``
    Raw bytes acquired from an identified physical device, with acquisition
    time and an explicit calibration reference (possibly ``not_applied``).
``independently_verified``
    Agreement between two computations whose implementations have distinct
    origins (for example ``ciw`` against ``scipy`` or a pinned provider). It
    is independent implementation agreement, not verification by another
    party; the latter is outside what a computational experiment establishes.
``not_established``
    No admissible basis, or a claim outside what the basis can support.
"""
from __future__ import annotations

from copy import deepcopy
import math
import re
import unicodedata

LABELS = ("analytic", "synthetic", "numerically_verified", "provider_backed",
          "hardware_measured", "independently_verified", "not_established")

# Claim domains. Physical domains need acquired hardware evidence; authority
# domains are decided outside the workbench and are never established here.
COMPUTATIONAL_DOMAINS = frozenset({"mathematical", "numerical", "computational_pipeline", "provenance"})
PHYSICAL_DOMAINS = frozenset({"physical", "calibration", "sensor_performance"})
AUTHORITY_DOMAINS = frozenset({"machine_safety", "industrial_readiness", "customer_demand",
                               "actuator_authority", "production_acceptance"})
DOMAINS = COMPUTATIONAL_DOMAINS | PHYSICAL_DOMAINS | AUTHORITY_DOMAINS

# ``cross_implementation`` is agreement between two implementations of the
# same origin (for example ciw Python and a ciw Rust kernel): a passing check,
# never independence.
# Thresholds beyond this magnitude cannot fail for any finite measurement of interest.
MAX_THRESHOLD = 1e100

REFERENCE_KINDS = frozenset({"analytic", "high_precision", "invariant", "self_convergence",
                             "exact_arithmetic", "refusal", "cross_implementation"})

# What a computational experiment may establish, and what it cannot establish
# alone. The right-hand claims require hardware, calibration procedures,
# independent measurement or an external authority.
BOUNDARY = (
    ("Analytic agreement", "Physical truth"),
    ("Numerical convergence", "Calibration validity"),
    ("Synthetic sensor performance", "Real sensor performance"),
    ("Replay determinism", "Independent verification by another party"),
    ("Schema/provenance integrity", "Machine safety"),
    ("GPU/CPU agreement", "Industrial readiness"),
    ("A plausible use case", "Actual customer demand"),
    ("A simulated control response", "Safe actuator authority"),
)

class EvidenceRefusal(ValueError):
    """A declared label is not supported by its basis."""


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise EvidenceRefusal(f"{name} must be a nonempty string")
    return value


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise EvidenceRefusal(f"{name} must be a finite number")
    return float(value)


COMPARISONS = ("abs_le", "le", "ge", "signed_le", "signed_ge")


def holds(observed: float, tolerance: float, comparison: str = "abs_le") -> bool:
    """Evaluate one check comparison; section helpers use this to fill ``passed``.

    ``abs_le`` bounds |observed|; ``le`` and ``ge`` bound a nonnegative
    magnitude; ``signed_le`` and ``signed_ge`` bound a signed quantity such as
    a difference that the claim says must not exceed (or fall below) a value.
    """
    if comparison not in COMPARISONS:
        raise EvidenceRefusal(f"comparison must be one of {', '.join(COMPARISONS)}")
    if comparison == "abs_le":
        return abs(observed) <= tolerance
    if comparison in ("le", "signed_le"):
        return observed <= tolerance
    return observed >= tolerance


def _check_passed(check, name):
    """A reference comparison passes only if its recorded numbers say so."""
    if not isinstance(check, dict):
        raise EvidenceRefusal(f"{name} must be an object")
    kind = check.get("reference_kind")
    if kind not in REFERENCE_KINDS:
        raise EvidenceRefusal(f"{name}.reference_kind must be one of {sorted(REFERENCE_KINDS)}")
    _text(check.get("reference"), f"{name}.reference")
    if kind == "refusal":
        # A refusal check passes when the expected refusal code was observed.
        expected, observed = check.get("expected_refusal"), check.get("observed_refusal")
        _text(expected, f"{name}.expected_refusal")
        if not isinstance(observed, str):
            raise EvidenceRefusal(f"{name}.observed_refusal must be a string (use 'none' when nothing was refused)")
        if check.get("passed") is not (observed == expected):
            raise EvidenceRefusal(f"{name}.passed does not match its expected and observed refusal codes")
        return observed == expected
    observed = _finite(check.get("observed"), f"{name}.observed")
    tolerance = _finite(check.get("tolerance"), f"{name}.tolerance")
    comparison = check.get("comparison", "abs_le")
    if abs(tolerance) > MAX_THRESHOLD:
        raise EvidenceRefusal(f"{name}.tolerance {tolerance!r} makes the comparison vacuous")
    if comparison in ("abs_le", "le") and tolerance < 0:
        raise EvidenceRefusal(f"{name}.tolerance must be nonnegative for {comparison}")
    if comparison == "le" and observed < 0:
        # le bounds a nonnegative magnitude (error, residual, count); a signed
        # quantity with an upper bound must say so with signed_le.
        raise EvidenceRefusal(f"{name}: le bounds a nonnegative magnitude; use abs_le or signed_le for {observed!r}")
    if comparison not in COMPARISONS:
        raise EvidenceRefusal(f"{name}.comparison must be abs_le, le, ge, signed_le or signed_ge")
    result = holds(observed, tolerance, comparison)
    if check.get("passed") is not result:
        raise EvidenceRefusal(f"{name}.passed does not match its observed value and tolerance")
    return result


def _identity(value, name):
    if not isinstance(value, dict):
        raise EvidenceRefusal(f"{name} must be an implementation identity object")
    _text(value.get("implementation"), f"{name}.implementation")
    return value["implementation"]


# Implementation families recognised as independent of the workbench's own
# code. An independent check pairs two known origins that differ: ciw against
# one of these, one of these against ciw, or two of these. An origin outside
# this set cannot take part, and ciw never checks ciw. numpy counts only for
# routines the compared ciw code does not itself call; reviewers check that
# per finding.
CIW_ORIGIN = "ciw"
INDEPENDENT_ORIGINS = frozenset({
    "scipy", "sympy", "mpmath", "numpy", "cpython", "zlib", "git",
    "curved-surface-geodesic-sensitivity-runtime", "flat-torus-geodesic-reference",
    "parameterized-lyapunov-stability-runtime", "scientific-computation-runtime",
})


def origin(implementation: str) -> str:
    """Implementation family: the leading ASCII name before '.', ':', '@', '/', '(', ' ' or '#'.

    Code in one family (for example every ``ciw.*`` module) cannot verify
    itself independently, whatever its revision, language or step size.
    Identifiers must be plain ASCII so lookalike characters cannot mint a
    new family.
    """
    text = unicodedata.normalize("NFKC", implementation).strip()
    if not text.isascii() or not text:
        raise EvidenceRefusal(f"Implementation identifiers must be nonempty ASCII: {implementation!r}")
    match = re.match(r"[A-Za-z0-9][A-Za-z0-9_-]*", text)
    if not match:
        raise EvidenceRefusal(f"Implementation identifier has no family name: {implementation!r}")
    return match.group(0).casefold()


def _known_origin(implementation: str, name: str) -> str:
    family = origin(implementation)
    if family != CIW_ORIGIN and family not in INDEPENDENT_ORIGINS:
        raise EvidenceRefusal(f"{name} origin {family!r} is not a recognised implementation family")
    tokens = set(re.split(r"[^a-z0-9]+", implementation.casefold()))
    if family != CIW_ORIGIN and CIW_ORIGIN in tokens:
        raise EvidenceRefusal(f"{name} {implementation!r} names the ciw family inside another family")
    return family


def supported_label(basis: dict, domain: str) -> str:
    """Return the strongest label the basis supports for a claim domain.

    Precedence for computational claims is independent agreement, then a
    passing reference check, then a pinned provider result, then synthetic
    generation, then a derivation. Any failed check leaves the claim
    ``not_established``: a refuted claim is not a synthetic result.
    """
    if domain not in DOMAINS:
        raise EvidenceRefusal(f"Unsupported claim domain: {domain}")
    if not isinstance(basis, dict):
        raise EvidenceRefusal("basis must be an object")
    unknown = set(basis) - {"derivation", "generator", "checks", "provider", "independent_check",
                            "acquisition", "inputs", "notes"}
    if unknown:
        raise EvidenceRefusal(f"Unsupported basis fields: {sorted(unknown)}")
    if domain in AUTHORITY_DOMAINS:
        return "not_established"
    checks = basis.get("checks") or []
    if not isinstance(checks, list):
        raise EvidenceRefusal("basis.checks must be a list")
    results = [_check_passed(check, f"checks[{index}]") for index, check in enumerate(checks)]
    independent = basis.get("independent_check")
    independent_passed = None if independent is None else _independent(independent)
    if not all(results) or independent_passed is False:
        return "not_established"
    acquisition = basis.get("acquisition")
    if domain in PHYSICAL_DOMAINS:
        if acquisition is None:
            return "not_established"
        if not isinstance(acquisition, dict):
            raise EvidenceRefusal("acquisition must be an object")
        for field in ("device", "raw_sha256", "acquired_at", "calibration"):
            _text(acquisition.get(field), f"acquisition.{field}")
        if not re.fullmatch(r"[0-9a-f]{64}", acquisition["raw_sha256"]):
            raise EvidenceRefusal("acquisition.raw_sha256 must be the SHA-256 of retained raw bytes")
        return "independently_verified" if independent_passed else "hardware_measured"
    if acquisition is not None:
        raise EvidenceRefusal("A computational claim cannot cite hardware acquisition as its basis")
    if independent_passed:
        return "independently_verified"
    if results:
        return "numerically_verified"
    provider = basis.get("provider")
    if provider is not None:
        if not isinstance(provider, dict):
            raise EvidenceRefusal("provider must be an object")
        for field in ("repository", "revision"):
            _text(provider.get(field), f"provider.{field}")
        if not (provider.get("source_tree") or provider.get("runtime_digest")):
            raise EvidenceRefusal("provider requires source_tree or runtime_digest")
        if provider.get("executed") is True:
            return "provider_backed"
    generator = basis.get("generator")
    if generator is not None:
        if not isinstance(generator, dict):
            raise EvidenceRefusal("generator must be an object")
        _text(generator.get("name"), "generator.name")
        return "synthetic"
    if basis.get("derivation") is not None:
        _text(basis["derivation"], "derivation")
        return "analytic"
    return "not_established"


def _independent(check):
    producer = _identity(check.get("producer"), "independent_check.producer")
    checker = _identity(check.get("checker"), "independent_check.checker")
    if check.get("reference_kind") == "cross_implementation":
        raise EvidenceRefusal("cross_implementation agreement is same-origin evidence; record it as a check")
    if _known_origin(producer, "producer") == _known_origin(checker, "checker"):
        # The same implementation family at another step size, revision or
        # module is self-consistency; declare it as a self_convergence check.
        raise EvidenceRefusal("independent_check producer and checker share an implementation origin")
    return _check_passed(check, "independent_check")


def finding(claim: str, domain: str, value, basis: dict, *, unit: str | None = None,
            uncertainty=None, tolerance: dict | None = None, counterexample: dict | None = None,
            expected_not_established: bool = False) -> dict:
    """Build a finding whose label is assigned by the validator, never by the caller.

    ``tolerance`` ({"abs", "rel"}) bounds regression comparison of ``value``.
    ``counterexample`` records a refuted general statement for the catalogue.
    ``expected_not_established`` marks a computational finding that honestly
    records an unestablished claim inside an otherwise completed task.
    """
    record = {"claim": _text(claim, "claim"), "domain": domain, "value": deepcopy(value),
              "unit": unit, "uncertainty": deepcopy(uncertainty), "basis": deepcopy(basis),
              "evidence_status": supported_label(basis, domain), "assigned_by": "ciw.lab.evidence"}
    if tolerance is not None:
        record["regression_tolerance"] = deepcopy(tolerance)
    if counterexample is not None:
        _text(counterexample.get("statement"), "counterexample.statement")
        record["counterexample"] = deepcopy(counterexample)
    if expected_not_established is True:
        record["expected_not_established"] = True
        _check_expected(record)
    elif expected_not_established is not False:
        raise EvidenceRefusal("expected_not_established must be True or False")
    return record


def _check_expected(record):
    """The flag marks an honestly unsupported claim, never a refuted or supported one."""
    if record["evidence_status"] != "not_established":
        raise EvidenceRefusal("expected_not_established applies only to not_established findings")
    basis = record["basis"]
    if basis.get("checks") or basis.get("independent_check"):
        raise EvidenceRefusal("A claim with checks is supported or refuted, not expected to be unestablished")


def validate_finding(record: dict) -> dict:
    """Refuse a finding whose stated label differs from what its basis supports."""
    if not isinstance(record, dict):
        raise EvidenceRefusal("finding must be an object")
    stated = record.get("evidence_status")
    if stated not in LABELS:
        raise EvidenceRefusal(f"Unknown evidence status: {stated!r}")
    supported = supported_label(record.get("basis"), record.get("domain"))
    if stated != supported:
        raise EvidenceRefusal(f"Evidence label refused: basis supports {supported}, finding states {stated}")
    if record.get("assigned_by") != "ciw.lab.evidence":
        raise EvidenceRefusal("Evidence labels are assigned by the deterministic validator only")
    if "expected_not_established" in record:
        if record["expected_not_established"] is not True:
            raise EvidenceRefusal("expected_not_established must be exactly true when present")
        _check_expected(record)
    return record


def physical_status(findings) -> str:
    """Physical status of a result derived from findings: never above its weakest input.

    Only physical-domain findings backed by acquisition carry physical support;
    a derived result is ``hardware_measured`` only when every input is.
    """
    findings = [validate_finding(record) for record in findings]
    if findings and all(record["domain"] in PHYSICAL_DOMAINS and record["evidence_status"]
                        in ("hardware_measured", "independently_verified") for record in findings):
        return "hardware_measured"
    return "not_established"


# Strength order of computational labels for the task headline.
COMPUTATIONAL_ORDER = ("not_established", "synthetic", "analytic", "provider_backed",
                       "numerically_verified", "independently_verified")


def primary_label(findings) -> str:
    """Headline label: the weakest label among established computational findings.

    Independent of finding order. Physical and authority claims are reported
    separately (physical validation status), and findings that honestly record
    an unestablished claim are counted, not averaged into the headline. A task
    whose computational findings are all unestablished is ``not_established``.
    """
    established = [f["evidence_status"] for f in findings
                   if f["domain"] in COMPUTATIONAL_DOMAINS and f["evidence_status"] != "not_established"]
    refuted = [f for f in findings if f["domain"] in COMPUTATIONAL_DOMAINS
               and f["evidence_status"] == "not_established" and not f.get("expected_not_established")]
    if refuted or not established:
        return "not_established"
    return min(established, key=COMPUTATIONAL_ORDER.index)


def summarize(findings) -> dict:
    """Count labels across findings after revalidating each one."""
    counts = {label: 0 for label in LABELS}
    for record in findings:
        counts[validate_finding(record)["evidence_status"]] += 1
    return counts
