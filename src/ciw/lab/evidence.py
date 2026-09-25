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

The label rules rank a passing check above provenance, so a label alone does
not say where a result came from: ``numerically_verified`` covers a check on
synthetic inputs, on a provider's output and on a derivation alike. Every
finding therefore also carries its *basis components*, the sorted list of the
parts of the basis it declares (:data:`ORIGINS`, stored under the finding key
``origin`` for compatibility), derived here from the basis like the label and
shown beside it as the finding's basis, with the generator, provider or
device each component names (:func:`describe_basis`). They never change a
label. They are not the implementation origin (:func:`origin`) that
``independently_verified`` compares.

A claim's domain is chosen by its author and reviewed, not inferred from the
wording. :func:`screen_authority_claim` is a conservative vocabulary screen
that refuses a computational- or physical-domain claim asserting an authority
outcome (production acceptance, certification for use, machine safety,
actuator authorization, industrial readiness, customer demand), unless, in the
same clause and before the outcome, the claim says the software does not make
or mark it; it catches the phrases it lists, not every paraphrase.
"""
from __future__ import annotations

from copy import deepcopy
import json
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


# Basis components a finding declares (its ``origin`` key), in sorted order.
# ``provider`` counts only an executed provider (``executed: true``);
# ``reference_checks`` only a nonempty check list. ``inputs`` and ``notes``
# describe, not support. These are parts of the basis, not the implementation
# origin (:func:`origin`) that an independent check compares.
ORIGINS = ("acquisition", "derivation", "independent_check", "provider", "reference_checks", "synthetic_inputs")
# Neutral words for counting components. A declared acquisition record is
# shown as hardware acquisition only on a finding it establishes (describe_basis).
ORIGIN_WORDS = {"acquisition": "acquisition record", "derivation": "derivation",
                "independent_check": "independent check", "provider": "pinned provider run",
                "reference_checks": "reference checks", "synthetic_inputs": "synthetic inputs"}
NO_ORIGIN = "no declared basis"
ACCEPTED_ACQUISITION = "hardware acquisition"
UNACCEPTED_ACQUISITION = "declared acquisition record (not accepted)"

# Authority outcomes stated as results. Each alternative is a phrase, not a
# topic word: "accepted", "certifies" and "safe" alone are ordinary technical
# vocabulary (accepted steps, a certifying verdict, a safe step size).
AUTHORITY_OUTCOME = re.compile(
    r"\b(?:"
    # production acceptance and disposition ("accepted for use" alone is ordinary: samples, steps, surrogates)
    r"(?:accepted|approved|released|cleared|qualified|signed[ -]off|dispositioned)\s+(?:for|to|into)\s+"
    r"(?:series\s+|serial\s+|volume\s+|full[- ]scale\s+)?(?:production|manufactur\w*|shipment|shipping|delivery|"
    r"service|deployment|operation|customers?)"
    r"|(?:approved|released|cleared|signed[ -]off|dispositioned)\s+for\s+use"
    r"|(?:accepted|qualified)\s+for\s+use\s+(?:in\s+(?:production|service|the\s+field)|"
    r"by\s+(?:customers?|operators?|people|personnel|the\s+public))"
    r"|(?:rejected|scrapped|quarantined)\s+(?:for|from)\s+(?:production|shipment|delivery|service|use)"
    r"|pass(?:es|ed)?\s+(?:production|final|incoming|first[- ]article)\s+(?:acceptance|inspection)"
    r"|production[- ]acceptance\s+(?:is\s+|was\s+|has\s+been\s+)?(?:granted|given|established|achieved|met|passed)"
    # certification for use
    r"|certified\s+(?:for|as)\s+(?:production|use|service|operation|deployment|flight|safe|safety|conforming|"
    r"compliant|airworth\w*)"
    r"|certif(?:y|ies)\s+(?:\S+\s+){0,4}?(?:for|as)\s+(?:production|use|service|operation|deployment|safe)\b"
    # machine safety. "Safe to use" and "safe to run" alone are ordinary (a step size, a lock-free queue);
    # they state safety with a physical subject or with people or production as the object.
    r"|safe\s+(?:to|for)\s+(?:operate|operation|deploy\w*|actuat\w*|people|personnel|operators?|humans?|"
    r"workers?|production|service)"
    r"|safe\s+(?:to|for)\s+(?:use|run)\s+(?:(?:by|for|with|near|around|among)\s+(?:operators?|people|personnel|"
    r"humans?|workers?|staff|customers?|patients?|the\s+public)|(?:in|on|at)\s+(?:production|service|the\s+field|"
    r"the\s+plant|the\s+factory|the\s+shop[- ]floor|site|the\s+line))"
    r"|(?:(?<!real\s)(?<!imaginary\s)parts?|coupons?|lots?(?!\s+of\b)|workpieces?|machines?|machinery|press(?:es)?|"
    r"robots?|equipment|actuators?|plants?|tooling|fixtures?|vehicles?|drones?|controllers?|grippers?|spindles?|"
    r"motors?|assembl(?:y|ies))\b[^;:.!?]{0,80}?\bsafe\s+(?:to|for)\s+(?:use|run)"
    r"|(?:machine|operator|operational|functional|process|personnel|worker)\s+safety\s+"
    r"(?:is\s+|was\s+|has\s+been\s+)?(?:established|guaranteed|assured|ensured|verified|validated|demonstrated|"
    r"certified|confirmed|achieved)"
    r"|(?:guarantees?|ensures?|establish(?:es)?|demonstrates?|assures?|proves?|certif(?:y|ies))\s+"
    r"(?:machine|operator|operational|functional|personnel|worker)\s+safety"
    r"|(?:machine|press|robot|cell|equipment|actuator|plant)s?\s+(?:is|are)\s+safe\b"
    # actuator authorization
    r"|(?:authori[sz]ed|permitted|cleared|approved)\s+to\s+(?:actuate|drive|move|command|energi[sz]e|operate)"
    r"|actuat(?:or|ion)\s+(?:authority|authori[sz]ation)\s+(?:is\s+|was\s+|has\s+been\s+)?"
    r"(?:granted|established|given|authori[sz]ed|approved|safe)"
    r"|(?:grants?|granted|authori[sz]es?|authori[sz]ed)\s+(?:safe\s+)?actuat\w*"
    # industrial readiness
    r"|(?:ready|readiness)\s+for\s+(?:industrial|production|series|serial|field|commercial|market|deployment|"
    r"operational|factory|shop[- ]floor)"
    r"|(?:industrial|production|deployment|field|market|commercial)[- ]ready"
    r"|(?:industrial|production|deployment|operational|market)\s+readiness\s+(?:is\s+|was\s+|has\s+been\s+)?"
    r"(?:established|demonstrated|achieved|shown|confirmed|verified|reached|met)"
    r"|industrially\s+(?:ready|deployable|proven|qualified)"
    # customer demand
    r"|(?:customer|market|user)\s+demand\s+(?:exists|is\s+(?:real|established|demonstrated|confirmed|shown|high|"
    r"strong|proven)|"
    r"was\s+(?:established|demonstrated|confirmed|shown))"
    r"|(?:shows?|demonstrates?|confirms?|establish(?:es)?|proves?|evidences?)\s+(?:real\s+|actual\s+|strong\s+)?"
    r"(?:customer|market|user)\s+demand"
    r"|(?:there\s+is|exists)\s+(?:real\s+|actual\s+|strong\s+)?(?:customer|market)\s+demand"
    r"|(?:customers|clients|the\s+market|buyers)\s+(?:want|wants|need|needs|demand|demands|"
    r"(?:will|would)\s+(?:buy|pay|adopt|purchase))"
    r")\b", re.IGNORECASE)
# A claim that the software declines to make or mark such an outcome ("the lab
# API cannot mark a lot accepted for production") is a claim about the
# software, not the outcome. The exemption is scoped to the clause holding the
# outcome phrase and must come before it, so an unrelated negation elsewhere
# ("accepted for production; it does not need rework") exempts nothing.
CLAUSE_BREAK = re.compile(
    r"[;:!?()\[\]]|,(?=\s)|\.(?=\s|$)|\s[-–—]\s|[–—]"
    r"|\s(?:and|but|while|whereas|although|though|yet|so|because)\s", re.IGNORECASE)
_NEGATION = (r"(?:cannot|can\s+not|can['’]t|never|neither|nor|"
             r"(?:does|do|did|will|would|may|might|must|shall|should|could|can)\s+not|"
             r"(?:doesn|don|didn|won|wouldn|mustn|shouldn|couldn)['’]t|(?:refus|declin)(?:e|es|ed|ing)\s+to)")
# Verbs by which software would make, mark or record a decision.
_DECISION_VERB = (r"(?:mark(?:s|ed|ing)?|claim(?:s|ed|ing)?|decid(?:e|es|ed|ing)|authori[sz](?:e|es|ed|ing)|"
                  r"establish(?:es|ed|ing)?|assert(?:s|ed|ing)?|grant(?:s|ed|ing)?|certif(?:y|ies|ied|ying)|"
                  r"approv(?:e|es|ed|ing)|accept(?:s|ed|ing)?|record(?:s|ed|ing)?|issu(?:e|es|ed|ing)|"
                  r"sign(?:s|ed|ing)?|releas(?:e|es|ed|ing)|declar(?:e|es|ed|ing)|conclud(?:e|es|ed|ing)|"
                  r"determin(?:e|es|ed|ing)|judg(?:e|es|ed|ing)|confer(?:s|red|ring)?|impl(?:y|ies|ied|ying)|"
                  r"guarantee(?:s|d|ing)?|label(?:s|ed|led|ing|ling)?|report(?:s|ed|ing)?|deem(?:s|ed|ing)?|"
                  r"qualif(?:y|ies|ied|ying)|infer(?:s|red|ring)?|find(?:s|ing)?|found)")
DECLINED_DECISION = re.compile(
    r"\b" + _NEGATION + r"\s+(?:\w+\s+){0,2}?" + _DECISION_VERB + r"\b"
    r"|\bno\s+(?:decision|claim|authority|determination|judg(?:e)?ment)"
    r"(?=\s+(?:that|on|about|of|whether|to|over|regarding)\b)", re.IGNORECASE)
# "Records <the outcome> as not performed / refused / external / pending": the
# outcome phrase lies inside the recorded object. "Recorded as lot 7" is not this.
RECORDED_AS_UNDECIDED = re.compile(
    r"\brecord(?:s|ed|ing)?\s+(?:\S+\s+){0,6}?as\s+(?:not\s+(?:performed|made|decided|established|"
    r"authori[sz]ed|granted|claimed|asserted|accepted|given)|refused|external|pending)\b", re.IGNORECASE)
# An outcome phrase that starts with an active verb may be negated directly: "does not authorize actuation".
_ACTIVE_OUTCOME = re.compile(r"(?:authori[sz]es?|grants?|certif(?:y|ies)|establish(?:es)?|guarantees?|ensures?|"
                             r"demonstrates?|proves?|assures?|confirms?|shows?|evidences?)\b", re.IGNORECASE)
_TRAILING_NEGATION = re.compile(r"\b" + _NEGATION + r"\s+$", re.IGNORECASE)
# A negation inside a relative clause ("the lot that never failed ...") modifies its noun, not the outcome.
_RELATIVE = re.compile(r"(?<=\w)\s+(that|which|who|whom|whose)\b", re.IGNORECASE)
_COPULA = re.compile(r"\b(?:is|are|was|were|be|been|being|becomes?|became|gets?|got|remains?|stays?|seems?|"
                     r"appears?)\b", re.IGNORECASE)
_COMPLEMENT = re.compile(r"\s*(?:that|whether|if)\b", re.IGNORECASE)


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


PLACEHOLDER_REVISIONS = frozenset({"unversioned", "unknown", "none", "n/a", "na", "tbd", "-"})


def _identity(value, name):
    if not isinstance(value, dict):
        raise EvidenceRefusal(f"{name} must be an implementation identity object")
    _text(value.get("implementation"), f"{name}.implementation")
    # Each side names the version or source digest it ran, so a reader can tell which code agreed.
    revision = value.get("revision")
    _text(revision, f"{name}.revision")
    if revision.strip().casefold() in PLACEHOLDER_REVISIONS:
        raise EvidenceRefusal(f"{name}.revision must name a version or source digest, not {revision!r}")
    return value["implementation"]


# Implementation families recognised as independent of the workbench's own
# code. An independent check pairs two known origins that differ: ciw against
# one of these, one of these against ciw, or two of these. An origin outside
# this set cannot take part, and ciw never checks ciw. numpy counts only for
# routines the compared ciw code does not itself call; reviewers check that
# per finding. pygeodesic (Kirsanov's exact MMP geodesics) and potpourri3d
# (geometry-central, including FlipOut) are C++ mesh-geodesic libraries.
# ordinarydiffeq is SciML's OrdinaryDiffEq.jl solver family, run by T145's
# Julia worker; the family is the solver, not Julia itself, so CIW-authored
# Julia code does not become independent of CIW by its language.
CIW_ORIGIN = "ciw"
INDEPENDENT_ORIGINS = frozenset({
    "scipy", "sympy", "mpmath", "numpy", "cpython", "zlib", "git", "pygeodesic", "potpourri3d", "ordinarydiffeq",
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


def _acquisition(value) -> dict:
    """A well-formed acquisition record: device, raw SHA-256 (64 hex digits), acquisition time and calibration."""
    if not isinstance(value, dict):
        raise EvidenceRefusal("acquisition must be an object")
    for field in ("device", "raw_sha256", "acquired_at", "calibration"):
        _text(value.get(field), f"acquisition.{field}")
    if not re.fullmatch(r"[0-9a-f]{64}", value["raw_sha256"]):
        raise EvidenceRefusal("acquisition.raw_sha256 must be the SHA-256 of retained raw bytes")
    return value


def _domain_basis(basis, domain) -> None:
    """Refuse a hardware acquisition cited by a computational claim, whatever its checks say.

    :func:`supported_label` reaches this refusal only when the checks pass (a
    failing check decides the label first), so a refuted computational claim
    could otherwise carry an acquisition record into a report.
    """
    if domain in COMPUTATIONAL_DOMAINS and isinstance(basis, dict) and basis.get("acquisition") is not None:
        raise EvidenceRefusal("A computational claim cannot cite hardware acquisition as its basis")


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
        _acquisition(acquisition)
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


def basis_origin(basis: dict) -> list:
    """The basis components a finding declares, as a sorted list drawn from :data:`ORIGINS`.

    Derived like the label and never an input to it: ``synthetic_inputs`` for
    a generator, ``provider`` for an executed provider, ``reference_checks``
    for a nonempty check list, and ``derivation``, ``independent_check`` and
    ``acquisition`` when present. A component named here must be well formed
    even where the label rules never reach it (a generator beside a passing
    check, an acquisition beside a failing one or in an authority domain), so
    the basis shown beside a label is never a malformed record.
    """
    if not isinstance(basis, dict):
        raise EvidenceRefusal("basis must be an object")
    found = []
    if basis.get("acquisition") is not None:
        _acquisition(basis["acquisition"])
        found.append("acquisition")
    if basis.get("derivation") is not None:
        _text(basis["derivation"], "derivation")
        found.append("derivation")
    if basis.get("independent_check") is not None:
        if not isinstance(basis["independent_check"], dict):
            raise EvidenceRefusal("independent_check must be an object")
        found.append("independent_check")
    provider = basis.get("provider")
    if provider is not None:
        if not isinstance(provider, dict):
            raise EvidenceRefusal("provider must be an object")
        if provider.get("executed") is True:
            for field in ("repository", "revision"):
                _text(provider.get(field), f"provider.{field}")
            if not (provider.get("source_tree") or provider.get("runtime_digest")):
                raise EvidenceRefusal("provider requires source_tree or runtime_digest")
            found.append("provider")
    checks = basis.get("checks")
    if checks is not None and not isinstance(checks, list):
        raise EvidenceRefusal("basis.checks must be a list")
    if checks:
        found.append("reference_checks")
    generator = basis.get("generator")
    if generator is not None:
        if not isinstance(generator, dict):
            raise EvidenceRefusal("generator must be an object")
        _text(generator.get("name"), "generator.name")
        found.append("synthetic_inputs")
    return found


def finding_origin(record: dict) -> list:
    """A finding's basis components derived from its basis; a stated ``origin`` must equal them.

    Findings retained before basis components were recorded carry no
    ``origin`` key; theirs are derived here, so every reader shows the same.
    """
    derived = basis_origin(record.get("basis"))
    if "origin" in record and record["origin"] != derived:
        raise EvidenceRefusal(f"Finding origin refused: the basis declares the components {derived}, "
                              f"the finding states {record['origin']!r}")
    return derived


def describe_origin(origin) -> str:
    """Plain words for a list of basis components, for counts: 'reference checks, synthetic inputs'."""
    return ", ".join(ORIGIN_WORDS[item] for item in origin) if origin else NO_ORIGIN


def _identity_text(value, limit=80) -> str:
    text = value if isinstance(value, str) else json.dumps(value, sort_keys=True, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def describe_basis(record: dict) -> str:
    """A finding's basis in words, with the identity each component declares, for display beside its label.

    'reference checks, synthetic inputs (ciw.lab.sensor_fusion_bench, seed 602026)',
    'pinned provider run (owner/repo@0123456789ab)'. A declared acquisition
    record reads 'hardware acquisition (<device>)' only on a physical-domain
    finding it establishes (``hardware_measured`` or ``independently_verified``,
    which the runner's physical gate inspects); on any other finding it reads
    'declared acquisition record (not accepted)'. Identities are as declared,
    not authenticated; callers escape the text for their format.
    """
    basis, words = record["basis"], []
    for item in finding_origin(record):
        if item == "synthetic_inputs":
            generator = basis["generator"]
            seed = f", seed {_identity_text(generator['seed'], 40)}" if generator.get("seed") is not None else ""
            words.append(f"{ORIGIN_WORDS[item]} ({_identity_text(generator['name'])}{seed})")
        elif item == "provider":
            provider = basis["provider"]
            words.append(f"{ORIGIN_WORDS[item]} ({_identity_text(provider['repository'])}"
                         f"@{_identity_text(provider['revision'])[:12]})")
        elif item == "acquisition":
            accepted = (record.get("domain") in PHYSICAL_DOMAINS
                        and record.get("evidence_status") in ("hardware_measured", "independently_verified"))
            words.append(f"{ACCEPTED_ACQUISITION} ({_identity_text(basis['acquisition']['device'])})" if accepted
                         else UNACCEPTED_ACQUISITION)
        else:
            words.append(ORIGIN_WORDS[item])
    return ", ".join(words) if words else NO_ORIGIN


def origin_counts(findings) -> dict:
    """Findings per label and declared basis component, after revalidating each finding.

    ``{label: {component: n, ..., "none": n}}``: a finding counts once under
    each component it declares, and under ``none`` when it declares none.
    """
    counts = {label: {**{item: 0 for item in ORIGINS}, "none": 0} for label in LABELS}
    for record in findings:
        label = validate_finding(record)["evidence_status"]
        for item in finding_origin(record) or ["none"]:
            counts[label][item] += 1
    return counts


def origin_difference(retained: dict, fresh: dict) -> str:
    """Empty when two findings declare the same basis components, else 'basis components A -> B'."""
    before, after = finding_origin(retained), finding_origin(fresh)
    if before == after:
        return ""
    return f"basis components {', '.join(before) or 'none'} -> {', '.join(after) or 'none'}"


def _clause(claim: str, start: int, end: int) -> tuple:
    """The clause around ``claim[start:end]``: (clause text, start and end relative to it)."""
    first = 0
    for brk in CLAUSE_BREAK.finditer(claim, 0, start):
        first = brk.end()
    after = CLAUSE_BREAK.search(claim, end)
    last = after.start() if after else len(claim)
    return claim[first:last], start - first, end - first


def _declined(clause: str, start: int, end: int) -> bool:
    """True when the clause says, before the outcome at ``clause[start:end]``, that it is not made or marked.

    Counted: a negated decision verb ("cannot mark", "never claims that",
    "refuses to record") whose object reaches the outcome within six words
    without a verb of being, unless a complement clause ("that ...",
    "whether ...") introduces it; a negation directly before an outcome that
    starts with an active verb ("does not authorize actuation"); "no claim
    that" and the like; and "records <outcome> as not performed / refused /
    external / pending". A negation after a relative pronoun is not counted.
    """
    prefix = clause[:start]
    relative = _RELATIVE.search(prefix)
    limit = relative.start(1) if relative else len(prefix)
    for match in DECLINED_DECISION.finditer(prefix):
        if match.start() >= limit:
            break
        gap = prefix[match.end():]
        if len(gap.split()) <= 6 and (_COMPLEMENT.match(gap) or not _COPULA.search(gap)):
            return True
    negation = _TRAILING_NEGATION.search(prefix)
    if negation and negation.start() < limit and _ACTIVE_OUTCOME.match(clause, start):
        return True
    return any(match.start() < min(start, limit) and end <= match.end()
               for match in RECORDED_AS_UNDECIDED.finditer(clause))


def authority_outcomes(claim: str) -> list:
    """Authority outcome phrases in a claim that no clause-local exemption covers (see :func:`_declined`)."""
    found, position = [], 0
    while (hit := AUTHORITY_OUTCOME.search(claim, position)) is not None:
        position = hit.start() + 1          # overlapping: a long phrase must not hide a later one
        if not _declined(*_clause(claim, hit.start(), hit.end())):
            found.append(hit.group(0))
    return found


def screen_authority_claim(claim: str, domain: str) -> None:
    """Refuse a computational- or physical-domain claim whose wording asserts an authority outcome.

    Authority domains are always ``not_established``, but the domain is the
    author's choice: an acceptance statement filed as ``computational_pipeline``
    would otherwise be labelled by its checks (T141), and one filed as
    ``physical`` with an acquisition record would be ``hardware_measured``. A
    clause that says, before the outcome, that the software does not make,
    mark or record it passes (:func:`_declined`); a negation anywhere else in
    the claim does not. This is a phrase screen, conservative by design; which
    domain a free-text claim belongs to remains a review question.
    """
    if domain not in COMPUTATIONAL_DOMAINS | PHYSICAL_DOMAINS or not isinstance(claim, str):
        return
    outcomes = authority_outcomes(claim)
    if outcomes:
        kind = "computational" if domain in COMPUTATIONAL_DOMAINS else "physical"
        raise EvidenceRefusal(f"Claim states an authority outcome ({outcomes[0]!r}) in the {kind} domain "
                              f"{domain}; file it in an authority domain, where it is not_established: {claim!r}")


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
    _text(claim, "claim")
    label = supported_label(basis, domain)
    _domain_basis(basis, domain)
    screen_authority_claim(claim, domain)
    record = {"claim": claim, "domain": domain, "value": deepcopy(value),
              "unit": unit, "uncertainty": deepcopy(uncertainty), "basis": deepcopy(basis),
              "evidence_status": label, "origin": basis_origin(basis), "assigned_by": "ciw.lab.evidence"}
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
    _domain_basis(record.get("basis"), record["domain"])
    finding_origin(record)
    screen_authority_claim(record.get("claim"), record["domain"])
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
