"""``ciw.model-spec.v1``: the language-neutral executable model specification.

The specification is data. Python, Julia and later providers read the same
document; none of them receives host-language source. Its meaning includes the
state order, declared units and frames, time versus path length, commanded
versus measured quantities, calibration identity, uncertainty assumptions and
the validity domain. Validation never repairs a specification.
"""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import math
import re

from ..core.covariance import _validate_matrix
from ..core.identities import content_identity
from . import expression
from .units import BASE, DIMENSIONLESS, Unit, dimension_text, multiply, parse_unit

SPEC_SCHEMA = "ciw.model-spec.v1"
LOWERED_SCHEMA = "ciw.model-lowered.v1"
FIELDS = frozenset({
    "schema", "model_id", "title", "independent_variable", "states", "inputs", "parameters",
    "dynamics", "observations", "derived", "ports", "uncertainty", "validity_domain", "provenance",
})
INDEPENDENT_KINDS = {"time": tuple(map(Fraction, (0, 0, 1, 0, 0, 0, 0))),
                     "path_length": tuple(map(Fraction, (1, 0, 0, 0, 0, 0, 0)))}
INPUT_ROLES = frozenset({"commanded", "measured", "physical"})
DERIVED_ROLES = frozenset({"derived", "commanded", "physical"})
CALIBRATION_STATUSES = frozenset({"calibrated", "uncalibrated"})
ASSUMPTIONS = frozenset({
    "gaussian_measurement_noise", "white_measurement_noise", "independent_measurement_noise",
    "gaussian_initial_state", "gaussian_parameters", "no_process_noise", "white_process_noise",
    "parameters_constant_over_run", "inputs_zero_order_hold", "inputs_constant_over_run",
    "no_measurement_noise_declared", "deterministic_model", "independent_component_uncertainty",
})
MAX_SYMBOLS = 256
_SYMBOL = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}(?:\.[A-Za-z][A-Za-z0-9_]{0,63}){0,7}\Z")
_TEXT_LIMIT = 512
_GREEK = ("alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu nu xi "
          "pi rho sigma tau upsilon phi varphi chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma "
          "Upsilon Phi Psi Omega").split()
_LATEX_COMMANDS = set(_GREEK) | {"mathrm", "mathit", "mathbf", "hat", "dot", "ddot", "bar", "tilde", "prime"}
_LATEX_SAFE = re.compile(r"[A-Za-z0-9_{}^\\' ,.]{1,64}\Z")


class SpecificationError(ValueError):
    """A model specification was refused; nothing was executed or repaired."""


def _text(value, name, limit=_TEXT_LIMIT):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise SpecificationError(f"{name} must be a nonempty string of at most {limit} characters")
    return value


def _keys(value, required, optional=(), name="object"):
    if not isinstance(value, dict):
        raise SpecificationError(f"{name} must be an object")
    keys = set(value)
    missing, extra = set(required) - keys, keys - set(required) - set(optional)
    if missing or extra:
        raise SpecificationError(f"{name} fields: missing {sorted(missing)}, unexpected {sorted(extra)}")


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise SpecificationError(f"{name} must be a finite JSON number, not a boolean")
    return float(value)


def validate_latex_name(value, name):
    """A symbol's display name is restricted LaTeX: no environments, inputs or macros."""
    _text(value, name, 64)
    if _LATEX_SAFE.fullmatch(value) is None:
        raise SpecificationError(f"{name} contains characters outside the restricted LaTeX name set")
    for command in re.findall(r"\\([A-Za-z]+)", value):
        if command not in _LATEX_COMMANDS:
            raise SpecificationError(f"{name} uses unsupported LaTeX command \\{command}")
    if value.count("{") != value.count("}"):
        raise SpecificationError(f"{name} has unbalanced braces")
    return value


def default_latex_name(symbol: str) -> str:
    """``omega_0`` -> ``\\omega_{0}``; qualified names keep their component prefix upright."""
    *prefix, local = symbol.split(".")
    head, _, tail = local.partition("_")
    body = ("\\" + head) if head in _GREEK else (head if len(head) == 1 else rf"\mathrm{{{head}}}")
    if tail:
        body += rf"_{{\mathrm{{{tail}}}}}" if len(tail) > 1 and not tail.isdigit() else f"_{{{tail}}}"
    if prefix:
        body = rf"{body}^{{\mathrm{{{'.'.join(prefix).replace('_', ' ')}}}}}"
    return body


def _calibration(value, name):
    _keys(value, {"status", "calibration_id"}, name=name)
    if value["status"] not in CALIBRATION_STATUSES:
        raise SpecificationError(f"{name}.status must be calibrated or uncalibrated")
    if value["status"] == "calibrated":
        _text(value["calibration_id"], f"{name}.calibration_id", 256)
    elif value["calibration_id"] is not None:
        raise SpecificationError(f"{name}: an uncalibrated quantity cannot carry a calibration identity")


def _covariance(value, order_domain, name):
    if value is None:
        return
    _keys(value, {"order", "matrix"}, name=name)
    order = value["order"]
    if (not isinstance(order, list) or not order or len(set(order)) != len(order)
            or any(item not in order_domain for item in order)):
        raise SpecificationError(f"{name}.order must list distinct declared symbols")
    try:
        _validate_matrix(value["matrix"], len(order))
    except ValueError as exc:
        raise SpecificationError(f"{name}: {exc}") from exc


def _bounds(value, name):
    if (not isinstance(value, list) or len(value) != 2
            or any(item is not None and (type(item) not in (int, float) or not math.isfinite(item)) for item in value)):
        raise SpecificationError(f"{name} must be [lower|null, upper|null] with finite numbers")
    if value[0] is not None and value[1] is not None and not value[0] <= value[1]:
        raise SpecificationError(f"{name} lower bound exceeds upper bound")


class Model:
    """A validated specification plus its parsed symbol table (read-only view)."""

    def __init__(self, spec: dict):
        self.spec = spec
        self.digest = content_identity(spec)
        self.units: dict[str, Unit] = {}
        self.kind: dict[str, str] = {}
        self.entries: dict[str, dict] = {}

    @property
    def independent(self) -> str:
        return self.spec["independent_variable"]["symbol"]

    @property
    def state_order(self) -> list[str]:
        return [item["symbol"] for item in self.spec["states"]]

    def names(self) -> dict[str, str]:
        return {symbol: entry.get("latex") or default_latex_name(symbol) for symbol, entry in self.entries.items()}


def validate_spec(spec) -> Model:
    """Validate a complete specification and return its parsed model.

    Arbitrary LaTeX, exploratory derivations and other records are refused:
    only this structured document is executable.
    """
    if isinstance(spec, str):
        raise SpecificationError("latex_is_not_a_model: text, including LaTeX, is not an executable model")
    if isinstance(spec, dict) and spec.get("schema") == "ciw.exploratory-derivation.v1":
        raise SpecificationError("latex_is_not_a_model: an exploratory derivation is not bound to execution")
    _keys(spec, FIELDS, name="model specification")
    if spec["schema"] != SPEC_SCHEMA:
        raise SpecificationError(f"Unsupported model schema: {spec['schema']!r}")
    model = Model(spec)
    _text(spec["model_id"], "model_id", 128)
    _text(spec["title"], "title")

    def declare(entry, kind, required, optional=()):
        _keys(entry, required, optional, name=f"{kind} declaration")
        symbol = entry["symbol"]
        if not isinstance(symbol, str) or _SYMBOL.fullmatch(symbol) is None:
            raise SpecificationError(f"Invalid {kind} symbol: {symbol!r}")
        if symbol in model.entries:
            raise SpecificationError(f"Duplicate symbol: {symbol}")
        if len(model.entries) >= MAX_SYMBOLS:
            raise SpecificationError("Model exceeds its symbol budget")
        try:
            model.units[symbol] = parse_unit(entry["unit"])
        except ValueError as exc:
            raise SpecificationError(f"{symbol}: {exc}") from exc
        if "latex" in entry:
            validate_latex_name(entry["latex"], f"{symbol}.latex")
        if "quantity" in entry:
            _text(entry["quantity"], f"{symbol}.quantity", 128)
        if "frame" in entry:
            _text(entry["frame"], f"{symbol}.frame", 128)
        if "description" in entry:
            _text(entry["description"], f"{symbol}.description")
        model.kind[symbol], model.entries[symbol] = kind, entry
        return symbol

    independent = spec["independent_variable"]
    symbol = declare(independent, "independent", {"symbol", "kind", "unit", "origin"}, {"latex"})
    if independent["kind"] not in INDEPENDENT_KINDS:
        raise SpecificationError("independent_variable.kind must be time or path_length")
    expected = INDEPENDENT_KINDS[independent["kind"]]
    if model.units[symbol].dimension != expected:
        raise SpecificationError(f"A {independent['kind']} independent variable needs dimension {dimension_text(expected)}")
    _text(independent["origin"], "independent_variable.origin")

    for key, kind, required, optional in (
        ("states", "state", {"symbol", "quantity", "unit", "frame"}, {"latex", "description"}),
        ("inputs", "input", {"symbol", "quantity", "unit", "frame", "role"}, {"latex", "description", "calibration"}),
        ("parameters", "parameter", {"symbol", "quantity", "unit", "value"}, {"latex", "description"}),
    ):
        if not isinstance(spec[key], list) or (key == "states" and not spec[key]):
            raise SpecificationError(f"{key} must be an array" + (" with at least one state" if key == "states" else ""))
        for entry in spec[key]:
            declare(entry, kind, required, optional)
            if kind == "parameter":
                _finite(entry["value"], f"{entry['symbol']}.value")
            if kind == "input":
                if entry["role"] not in INPUT_ROLES:
                    raise SpecificationError(f"{entry['symbol']}.role must be commanded, measured or physical")
                if entry["role"] == "measured":
                    if "calibration" not in entry:
                        raise SpecificationError(f"Measured input {entry['symbol']} must declare its calibration status")
                    _calibration(entry["calibration"], f"{entry['symbol']}.calibration")
                elif "calibration" in entry:
                    raise SpecificationError(f"A {entry['role']} input carries no measurement calibration")

    dims = {name: unit.dimension for name, unit in model.units.items()}
    state_symbols = model.state_order
    if not isinstance(spec["dynamics"], list) or [item.get("state") if isinstance(item, dict) else None
                                                   for item in spec["dynamics"]] != state_symbols:
        raise SpecificationError("dynamics must give exactly one equation per state, in state order")
    base_scope = {name: dims[name] for name, kind in model.kind.items() if kind in {"independent", "state", "input", "parameter"}}

    if not isinstance(spec["derived"], list):
        raise SpecificationError("derived must be an array")
    derived_scope = dict(base_scope)
    for entry in spec["derived"]:
        name = declare(entry, "derived", {"symbol", "quantity", "unit", "role", "expression"}, {"latex", "description", "frame"})
        if entry["role"] not in DERIVED_ROLES:
            raise SpecificationError(f"{name}.role must be derived, commanded or physical")
        dims[name] = model.units[name].dimension
        _dimension_check(entry["expression"], derived_scope, dims[name], name)
        derived_scope[name] = dims[name]  # later derived quantities may use earlier ones

    for entry in spec["dynamics"]:
        _keys(entry, {"state", "rhs"}, name="dynamics equation")
        target = multiply(dims[entry["state"]], dims[symbol], -1)
        _dimension_check(entry["rhs"], derived_scope, target, f"d{entry['state']}/d{symbol}")

    if not isinstance(spec["observations"], list):
        raise SpecificationError("observations must be an array")
    for entry in spec["observations"]:
        name = declare(entry, "observation", {"symbol", "quantity", "unit", "frame", "sensor", "calibration", "expression"},
                       {"latex", "description"})
        _text(entry["sensor"], f"{name}.sensor", 256)
        _calibration(entry["calibration"], f"{name}.calibration")
        _dimension_check(entry["expression"], derived_scope, model.units[name].dimension, name)

    if not isinstance(spec["ports"], list):
        raise SpecificationError("ports must be an array")
    port_names = set()
    for port in spec["ports"]:
        _keys(port, {"name", "direction", "symbol"}, name="port")
        if not isinstance(port["name"], str) or _SYMBOL.fullmatch(port["name"]) is None or port["name"] in port_names:
            raise SpecificationError(f"Invalid or duplicate port name: {port['name']!r}")
        port_names.add(port["name"])
        kind = model.kind.get(port["symbol"])
        if port["direction"] == "in":
            if kind != "input":
                raise SpecificationError(f"In-port {port['name']} must name a declared input")
        elif port["direction"] == "out":
            if kind not in {"observation", "state", "derived"}:
                raise SpecificationError(f"Out-port {port['name']} must name an observation, state or derived quantity")
            if kind == "derived" and model.entries[port["symbol"]]["role"] == "derived":
                raise SpecificationError(f"Out-port {port['name']}: a purely derived quantity is not connectable; "
                                         "declare it commanded or physical")
            if kind == "derived" and "frame" not in model.entries[port["symbol"]]:
                raise SpecificationError(f"Out-port {port['name']} needs a frame on its derived quantity")
        else:
            raise SpecificationError("Port direction must be in or out")

    uncertainty = spec["uncertainty"]
    _keys(uncertainty, {"parameters", "initial_state", "measurement_noise", "process_noise", "assumptions"},
          name="uncertainty")
    parameters = [item["symbol"] for item in spec["parameters"]]
    observations = [item["symbol"] for item in spec["observations"]]
    _covariance(uncertainty["parameters"], parameters, "uncertainty.parameters")
    _covariance(uncertainty["initial_state"], state_symbols, "uncertainty.initial_state")
    _covariance(uncertainty["measurement_noise"], observations, "uncertainty.measurement_noise")
    _covariance(uncertainty["process_noise"], state_symbols, "uncertainty.process_noise")
    assumptions = uncertainty["assumptions"]
    if (not isinstance(assumptions, list) or len(set(assumptions)) != len(assumptions)
            or any(item not in ASSUMPTIONS for item in assumptions)):
        raise SpecificationError("uncertainty.assumptions must be distinct entries of the declared vocabulary")
    if uncertainty["process_noise"] is None and "no_process_noise" not in assumptions:
        raise SpecificationError("An absent process-noise model must be declared as no_process_noise")
    if uncertainty["measurement_noise"] is None and observations and "no_measurement_noise_declared" not in assumptions:
        raise SpecificationError("Observations without a noise model must declare no_measurement_noise_declared")

    domain = spec["validity_domain"]
    _keys(domain, {"independent_variable", "bounds", "notes"}, name="validity_domain")
    if domain["independent_variable"] is not None:
        _bounds(domain["independent_variable"], "validity_domain.independent_variable")
    if not isinstance(domain["bounds"], dict):
        raise SpecificationError("validity_domain.bounds must be an object")
    for name, bounds in domain["bounds"].items():
        if model.kind.get(name) not in {"state", "input", "parameter"}:
            raise SpecificationError(f"validity bound names an undeclared state, input or parameter: {name}")
        _bounds(bounds, f"validity_domain.bounds.{name}")
    if not isinstance(domain["notes"], list) or any(not isinstance(item, str) or not item.strip() for item in domain["notes"]):
        raise SpecificationError("validity_domain.notes must be an array of nonempty strings")
    for entry in spec["parameters"]:
        bounds = domain["bounds"].get(entry["symbol"])
        if bounds and not _inside(entry["value"], bounds):
            raise SpecificationError(f"Parameter {entry['symbol']} lies outside its declared validity domain")

    provenance = spec["provenance"]
    _keys(provenance, {"kind", "derived_from", "notes"}, name="provenance")
    if provenance["kind"] == "authored":
        if provenance["derived_from"] is not None:
            raise SpecificationError("An authored model has no derived_from record")
    elif provenance["kind"] in {"unit_rescale", "composition"}:
        if not isinstance(provenance["derived_from"], dict):
            raise SpecificationError("A derived model must retain its derivation record")
    else:
        raise SpecificationError("provenance.kind must be authored, unit_rescale or composition")
    if not isinstance(provenance["notes"], list) or any(not isinstance(item, str) for item in provenance["notes"]):
        raise SpecificationError("provenance.notes must be an array of strings")
    return model


def _inside(value, bounds) -> bool:
    return (bounds[0] is None or bounds[0] <= value) and (bounds[1] is None or value <= bounds[1])


def _dimension_check(node, scope, expected, name):
    try:
        actual = expression.validate(node, scope)
    except ValueError as exc:
        raise SpecificationError(f"{name}: {exc}") from exc
    if actual != expected:
        raise SpecificationError(f"{name}: expression has dimension {dimension_text(actual)}, "
                                 f"declared {dimension_text(expected)}")


def lower(model: Model) -> dict:
    """Compile a validated model into an explicit numeric program.

    The lowered form carries every unit scale explicitly, so an executing
    provider needs no unit parser: ``x_si = scale * x_declared``. Equations
    remain unit-aware SI expressions; the derivative in declared units is
    ``rhs_si * scale(independent) / scale(state)``.
    """
    spec = model.spec
    scale = lambda symbol: model.units[symbol].scale  # noqa: E731
    return {
        "schema": LOWERED_SCHEMA,
        "spec_digest": model.digest,
        "independent": {"symbol": model.independent, "scale": scale(model.independent)},
        "states": [{"symbol": item["symbol"], "scale": scale(item["symbol"])} for item in spec["states"]],
        "inputs": [{"symbol": item["symbol"], "scale": scale(item["symbol"])} for item in spec["inputs"]],
        "parameters": [{"symbol": item["symbol"], "scale": scale(item["symbol"]), "value": float(item["value"])}
                       for item in spec["parameters"]],
        "derived": [{"symbol": item["symbol"], "scale": scale(item["symbol"]), "expression": deepcopy(item["expression"])}
                    for item in spec["derived"]],
        "dynamics": [deepcopy(item["rhs"]) for item in spec["dynamics"]],
        "observations": [{"symbol": item["symbol"], "scale": scale(item["symbol"]),
                          "expression": deepcopy(item["expression"])} for item in spec["observations"]],
    }


def summary(model: Model) -> dict:
    """A terminal-facing symbol table; derived from, never fed back into, the spec."""
    rows = []
    for symbol, entry in model.entries.items():
        unit = model.units[symbol]
        rows.append({"symbol": symbol, "kind": model.kind[symbol], "unit": entry["unit"],
                     "si_scale": unit.scale, "dimension": dimension_text(unit.dimension),
                     "quantity": entry.get("quantity"), "frame": entry.get("frame"),
                     "role": entry.get("role") if model.kind[symbol] in {"input", "derived"} else
                     ("measured" if model.kind[symbol] == "observation" else None),
                     "calibration": entry.get("calibration")})
    return {"model_id": model.spec["model_id"], "spec_digest": model.digest,
            "independent_variable": model.spec["independent_variable"]["kind"],
            "state_order": model.state_order, "symbols": rows}


__all__ = ["BASE", "DIMENSIONLESS", "Model", "SpecificationError", "lower", "summary", "validate_spec"]
