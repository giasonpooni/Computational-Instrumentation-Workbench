"""Quantities, dimensions and unit conversion with covariance-consistent scaling.

Dimensions are exponent vectors over the SI base quantities plus two declared
pseudo-dimensions: ``px`` (image samples) and ``tick`` (device counter periods).
Neither converts to metres or seconds without an explicit camera or clock model,
so an image residual can never be silently compared with a surface distance.

Changing a model's length unit must transform its covariance consistently:
``convert_covariance`` applies ``S C S`` with the diagonal scale ``S``; offset
units (``degC``) contribute no offset to a covariance.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import re
from typing import Any, Iterable

import numpy as np

from ._common import Refusal, finite, text

BASE = ("m", "kg", "s", "A", "K", "mol", "cd", "px", "tick")
Dimension = tuple[Fraction, ...]
DIMENSIONLESS: Dimension = tuple(Fraction(0) for _ in BASE)
MAX_EXPRESSION = 96


def _dim(**exponents: int | Fraction) -> Dimension:
    return tuple(Fraction(exponents.get(name, 0)) for name in BASE)


@dataclass(frozen=True)
class Unit:
    """A parsed unit: SI value of one unit, dimension and (absolute-scale) offset."""

    expression: str
    scale: float
    dimension: Dimension
    offset: float = 0.0

    @property
    def dimensionless(self) -> bool:
        return self.dimension == DIMENSIONLESS

    def describe(self) -> dict:
        return {"unit": self.expression, "si_scale": self.scale, "dimension": dimension_text(self.dimension),
                "offset": self.offset}


_ATOMS: dict[str, tuple[float, Dimension]] = {
    "1": (1.0, DIMENSIONLESS),
    "m": (1.0, _dim(m=1)), "g": (1e-3, _dim(kg=1)), "s": (1.0, _dim(s=1)), "A": (1.0, _dim(A=1)),
    "K": (1.0, _dim(K=1)), "mol": (1.0, _dim(mol=1)), "cd": (1.0, _dim(cd=1)),
    "px": (1.0, _dim(px=1)), "tick": (1.0, _dim(tick=1)),
    "rad": (1.0, DIMENSIONLESS), "sr": (1.0, DIMENSIONLESS),
    "deg": (math.pi / 180.0, DIMENSIONLESS), "arcmin": (math.pi / 10800.0, DIMENSIONLESS),
    "arcsec": (math.pi / 648000.0, DIMENSIONLESS),
    "Hz": (1.0, _dim(s=-1)), "N": (1.0, _dim(kg=1, m=1, s=-2)), "Pa": (1.0, _dim(kg=1, m=-1, s=-2)),
    "J": (1.0, _dim(kg=1, m=2, s=-2)), "W": (1.0, _dim(kg=1, m=2, s=-3)), "C": (1.0, _dim(A=1, s=1)),
    "V": (1.0, _dim(kg=1, m=2, s=-3, A=-1)), "ohm": (1.0, _dim(kg=1, m=2, s=-3, A=-2)),
    "L": (1e-3, _dim(m=3)), "min": (60.0, _dim(s=1)), "h": (3600.0, _dim(s=1)), "day": (86400.0, _dim(s=1)),
    "bar": (1e5, _dim(kg=1, m=-1, s=-2)), "percent": (1e-2, DIMENSIONLESS), "ppm": (1e-6, DIMENSIONLESS),
    "count": (1.0, DIMENSIONLESS), "eV": (1.602176634e-19, _dim(kg=1, m=2, s=-2)),
    "Wh": (3600.0, _dim(kg=1, m=2, s=-2)), "inch": (0.0254, _dim(m=1)),
}
_PREFIXABLE = frozenset({"m", "g", "s", "A", "K", "mol", "Hz", "N", "Pa", "J", "W", "C", "V", "ohm", "L", "rad", "eV", "Wh"})
_PREFIXES = {"T": 1e12, "G": 1e9, "M": 1e6, "k": 1e3, "h": 1e2, "da": 1e1, "d": 1e-1, "c": 1e-2,
             "m": 1e-3, "u": 1e-6, "µ": 1e-6, "μ": 1e-6, "n": 1e-9, "p": 1e-12, "f": 1e-15}
_OFFSET_UNITS = {"degC": (1.0, _dim(K=1), 273.15)}
_TOKEN = re.compile(r"\s*(?:(?P<name>[A-Za-zµμ]+)|(?P<op>[*/.^()])|(?P<num>[+-]?\d+(?:\.\d+)?))")


def dimension_text(dimension: Dimension) -> str:
    parts = []
    for name, power in zip(BASE, dimension):
        if power:
            parts.append(name if power == 1 else f"{name}^{power}")
    return "*".join(parts) or "1"


def _atom(name: str) -> tuple[float, Dimension]:
    if name in _ATOMS:
        return _ATOMS[name]
    for prefix in sorted(_PREFIXES, key=len, reverse=True):
        base = name[len(prefix):]
        if name.startswith(prefix) and base in _PREFIXABLE:
            scale, dimension = _ATOMS[base]
            return _PREFIXES[prefix] * scale, dimension
    raise Refusal("unknown_unit", f"Undeclared unit symbol {name!r}")


class _Parser:
    def __init__(self, expression: str):
        self.tokens: list[tuple[str, str]] = []
        position = 0
        while position < len(expression):
            match = _TOKEN.match(expression, position)
            if match is None or match.end() == position:
                if expression[position:].strip() == "":
                    break
                raise Refusal("unknown_unit", f"Cannot parse unit expression {expression!r}")
            kind = match.lastgroup
            self.tokens.append((kind, match.group(kind)))
            position = match.end()
        self.index = 0

    def peek(self) -> tuple[str, str] | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def take(self) -> tuple[str, str]:
        token = self.peek()
        if token is None:
            raise Refusal("unknown_unit", "Unexpected end of unit expression")
        self.index += 1
        return token

    def expression(self) -> tuple[float, Dimension]:
        scale, dimension = self.term()
        while (token := self.peek()) is not None and token[0] == "op" and token[1] in "*./":
            self.take()
            other_scale, other_dimension = self.term()
            sign = -1 if token[1] == "/" else 1
            scale *= other_scale ** sign
            dimension = tuple(a + sign * b for a, b in zip(dimension, other_dimension))
        return scale, dimension

    def term(self) -> tuple[float, Dimension]:
        kind, value = self.take()
        if kind == "op" and value == "(":
            scale, dimension = self.expression()
            if self.take() != ("op", ")"):
                raise Refusal("unknown_unit", "Unbalanced parentheses in unit expression")
        elif kind == "name" and value == "sqrt":
            if self.take() != ("op", "("):
                raise Refusal("unknown_unit", "sqrt requires parentheses")
            scale, dimension = self.expression()
            if self.take() != ("op", ")"):
                raise Refusal("unknown_unit", "Unbalanced parentheses in unit expression")
            scale, dimension = math.sqrt(scale), tuple(power / 2 for power in dimension)
        elif kind == "name":
            scale, dimension = _atom(value)
        elif kind == "num" and value == "1":
            scale, dimension = 1.0, DIMENSIONLESS
        else:
            raise Refusal("unknown_unit", f"Unexpected token {value!r} in unit expression")
        if (token := self.peek()) is not None and token == ("op", "^"):
            self.take()
            power = self._exponent()
            scale, dimension = scale ** float(power), tuple(item * power for item in dimension)
        return scale, dimension

    def _exponent(self) -> Fraction:
        kind, value = self.take()
        if kind == "num":
            return Fraction(value).limit_denominator(12)
        if (kind, value) == ("op", "("):
            numerator = Fraction(self.take()[1])
            if self.peek() == ("op", "/"):
                self.take()
                numerator /= Fraction(self.take()[1])
            if self.take() != ("op", ")"):
                raise Refusal("unknown_unit", "Unbalanced exponent parentheses")
            return numerator
        raise Refusal("unknown_unit", "A unit exponent must be a number")


_CACHE: dict[str, Unit] = {}


def parse_unit(expression: Any) -> Unit:
    """Parse a unit expression such as ``m/s^2``, ``kg*m^2/s^2`` or ``m/sqrt(Hz)``."""
    expression = text(expression, "unit", MAX_EXPRESSION).strip()
    if expression in _CACHE:
        return _CACHE[expression]
    if expression in _OFFSET_UNITS:
        scale, dimension, offset = _OFFSET_UNITS[expression]
        unit = Unit(expression, scale, dimension, offset)
    else:
        if any(name in expression for name in _OFFSET_UNITS):
            raise Refusal("offset_unit_compound", f"{expression!r}: offset units cannot appear in a compound unit")
        parser = _Parser(expression)
        scale, dimension = parser.expression()
        if parser.peek() is not None:
            raise Refusal("unknown_unit", f"Trailing tokens in unit expression {expression!r}")
        if not math.isfinite(scale) or scale <= 0:
            raise Refusal("unknown_unit", f"Unit {expression!r} has no finite positive scale")
        unit = Unit(expression, scale, dimension)
    if len(_CACHE) < 4096:
        _CACHE[expression] = unit
    return unit


def compatible(first: Any, second: Any) -> bool:
    return parse_unit(first).dimension == parse_unit(second).dimension


def require_dimension(unit: Any, expected: Any, name: str = "quantity") -> Unit:
    parsed, reference = parse_unit(unit), parse_unit(expected)
    if parsed.dimension != reference.dimension:
        raise Refusal("dimension_mismatch", f"{name} in {parsed.expression!r} is not commensurable with "
                      f"{reference.expression!r}", found=dimension_text(parsed.dimension),
                      expected=dimension_text(reference.dimension))
    return parsed


def conversion_factor(source: Any, target: Any) -> float:
    """Scale factor for differences/uncertainties; offsets never enter it."""
    source_unit, target_unit = parse_unit(source), parse_unit(target)
    if source_unit.dimension != target_unit.dimension:
        raise Refusal("dimension_mismatch", f"Cannot convert {source_unit.expression!r} to {target_unit.expression!r}",
                      source=dimension_text(source_unit.dimension), target=dimension_text(target_unit.dimension))
    return source_unit.scale / target_unit.scale


def convert(value: Any, source: Any, target: Any) -> Any:
    """Convert absolute values, applying offsets only for standalone offset units."""
    source_unit, target_unit = parse_unit(source), parse_unit(target)
    factor = conversion_factor(source, target)
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise Refusal("malformed_record", "Converted values must be finite")
    if source_unit.offset == 0 and target_unit.offset == 0:
        result = array * factor
    else:
        result = (array * source_unit.scale + source_unit.offset - target_unit.offset) / target_unit.scale
    return float(result) if result.ndim == 0 else result


def convert_covariance(matrix: Any, source_units: Iterable[Any], target_units: Iterable[Any]) -> np.ndarray:
    """Transform a covariance between unit bases: ``C' = S C S`` with ``S = diag(factor)``."""
    covariance = np.asarray(matrix, dtype=float)
    factors = np.array([conversion_factor(a, b) for a, b in zip(source_units, target_units, strict=True)])
    if covariance.shape != (len(factors), len(factors)):
        raise Refusal("malformed_covariance", "Covariance order does not match the declared unit basis")
    return factors[:, None] * covariance * factors[None, :]


@dataclass(frozen=True)
class Quantity:
    """A value (scalar or array) bound to a unit expression and optional frame."""

    value: Any
    unit: str
    frame: str | None = None

    def __post_init__(self) -> None:
        parse_unit(self.unit)
        array = np.asarray(self.value, dtype=float)
        if not np.all(np.isfinite(array)):
            raise Refusal("malformed_record", "A quantity value must be finite")

    @classmethod
    def from_json(cls, value: Any, name: str = "quantity") -> "Quantity":
        if not isinstance(value, dict) or "value" not in value or "unit" not in value:
            raise Refusal("malformed_record", f"{name} must be an object with value and unit")
        unknown = value.keys() - {"value", "unit", "frame"}
        if unknown:
            raise Refusal("malformed_record", f"{name} has undeclared fields {sorted(unknown)}")
        raw = value["value"]
        number = ([finite(item, f"{name}.value") for item in raw] if isinstance(raw, list)
                  else finite(raw, f"{name}.value"))
        return cls(number, parse_unit(value["unit"]).expression, value.get("frame"))

    def to(self, unit: str) -> "Quantity":
        return Quantity(convert(self.value, self.unit, unit), unit, self.frame)

    def si(self) -> float | np.ndarray:
        parsed = parse_unit(self.unit)
        if parsed.offset:
            return convert(self.value, self.unit, "K")
        array = np.asarray(self.value, dtype=float) * parsed.scale
        return float(array) if array.ndim == 0 else array

    def magnitude(self, unit: str) -> float:
        """Scalar magnitude in a required commensurable unit, refusing arrays."""
        result = convert(self.value, self.unit, unit)
        if not isinstance(result, float):
            raise Refusal("malformed_record", "A scalar quantity was required")
        return result

    @property
    def dimension(self) -> Dimension:
        return parse_unit(self.unit).dimension

    def __add__(self, other: "Quantity") -> "Quantity":
        self._same_frame(other)
        return Quantity(np.asarray(self.value) + np.asarray(convert(other.value, other.unit, self.unit)), self.unit, self.frame)

    def __sub__(self, other: "Quantity") -> "Quantity":
        self._same_frame(other)
        return Quantity(np.asarray(self.value) - np.asarray(convert(other.value, other.unit, self.unit)), self.unit, self.frame)

    def _same_frame(self, other: "Quantity") -> None:
        if not isinstance(other, Quantity):
            raise Refusal("dimension_mismatch", "Only quantities combine with quantities")
        if self.frame != other.frame:
            raise Refusal("frame_mismatch", f"Quantities in frames {self.frame!r} and {other.frame!r} cannot combine "
                          "without an explicit transform")
        if parse_unit(self.unit).offset or parse_unit(other.unit).offset:
            raise Refusal("offset_unit_arithmetic", "Absolute offset temperatures cannot be added or subtracted directly")

    def to_json(self) -> dict:
        value = np.asarray(self.value, dtype=float)
        record = {"value": float(value) if value.ndim == 0 else value.tolist(), "unit": self.unit}
        if self.frame is not None:
            record["frame"] = self.frame
        return record
