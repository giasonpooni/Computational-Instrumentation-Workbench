"""Multiplicative SI units with exact rational dimensions.

A declared unit string is part of a model's meaning and is retained verbatim.
Parsing yields a positive scale to the coherent SI unit and an exact dimension
vector. Affine scales (degree Celsius, Fahrenheit) are refused: the linear
rescaling contract and covariance transformation require ``x_si = scale * x``.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import re

BASE = ("m", "kg", "s", "A", "K", "mol", "cd")
_ZERO = (Fraction(0),) * len(BASE)
MAX_UNIT_LENGTH = 64


def _dims(**exponents) -> tuple[Fraction, ...]:
    return tuple(Fraction(exponents.get(name, 0)) for name in BASE)


# name -> (scale to coherent SI, dimension, accepts SI prefix)
_NAMED = {
    "m": (1.0, _dims(m=1), True),
    "g": (1e-3, _dims(kg=1), True),
    "s": (1.0, _dims(s=1), True),
    "A": (1.0, _dims(A=1), True),
    "K": (1.0, _dims(K=1), True),
    "mol": (1.0, _dims(mol=1), True),
    "cd": (1.0, _dims(cd=1), True),
    # Plane and solid angles are dimensionless in SI; the declared unit keeps
    # the distinction visible and ``deg`` rescales by an exact factor.
    "rad": (1.0, _ZERO, True),
    "sr": (1.0, _ZERO, False),
    "deg": (math.pi / 180.0, _ZERO, False),
    "Hz": (1.0, _dims(s=-1), True),
    "N": (1.0, _dims(kg=1, m=1, s=-2), True),
    "Pa": (1.0, _dims(kg=1, m=-1, s=-2), True),
    "bar": (1e5, _dims(kg=1, m=-1, s=-2), True),
    "J": (1.0, _dims(kg=1, m=2, s=-2), True),
    "W": (1.0, _dims(kg=1, m=2, s=-3), True),
    "C": (1.0, _dims(A=1, s=1), True),
    "V": (1.0, _dims(kg=1, m=2, s=-3, A=-1), True),
    "Ohm": (1.0, _dims(kg=1, m=2, s=-3, A=-2), True),
    "L": (1e-3, _dims(m=3), True),
    "min": (60.0, _dims(s=1), False),
    "h": (3600.0, _dims(s=1), False),
}
_PREFIX = {"G": 1e9, "M": 1e6, "k": 1e3, "c": 1e-2, "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12}
_AFFINE = {"degC", "degF", "°C", "°F", "celsius", "fahrenheit"}
_TOKEN = re.compile(r"\s*(?:(?P<name>[A-Za-z]+)|(?P<int>-?[0-9]+)|(?P<op>[*/^()]))")


@dataclass(frozen=True)
class Unit:
    """A parsed declared unit. ``text`` is retained; comparison uses scale and dimension."""

    text: str
    scale: float
    dimension: tuple[Fraction, ...]
    factors: tuple[tuple[str, int], ...]

    def same_dimension(self, other: "Unit") -> bool:
        return self.dimension == other.dimension

    def factor_to(self, other: "Unit") -> float:
        """Multiply a value in ``self`` by this factor to express it in ``other``."""
        if not self.same_dimension(other):
            raise ValueError(f"Cannot convert {self.text} to {other.text}: dimensions differ")
        return self.scale / other.scale

    @property
    def dimensionless(self) -> bool:
        return self.dimension == _ZERO


def _lookup(name: str) -> tuple[float, tuple[Fraction, ...]]:
    if name in _AFFINE:
        raise ValueError(f"Affine unit {name} is outside the multiplicative rescaling contract")
    if name in _NAMED:
        scale, dimension, _ = _NAMED[name]
        return scale, dimension
    if len(name) > 1 and name[0] in _PREFIX and name[1:] in _NAMED and _NAMED[name[1:]][2]:
        scale, dimension, _ = _NAMED[name[1:]]
        return _PREFIX[name[0]] * scale, dimension
    raise ValueError(f"Unknown unit symbol: {name}")


class _Parser:
    def __init__(self, text: str):
        self.tokens: list[tuple[str, str]] = []
        position = 0
        while position < len(text):
            match = _TOKEN.match(text, position)
            if match is None or match.end() == position:
                raise ValueError(f"Malformed unit: {text!r}")
            kind = match.lastgroup
            self.tokens.append((kind, match.group(kind)))
            position = match.end()
            while position < len(text) and text[position].isspace():
                position += 1
        self.index = 0

    def peek(self):
        return self.tokens[self.index] if self.index < len(self.tokens) else (None, None)

    def take(self, kind=None, value=None):
        token = self.peek()
        if token[0] is None or (kind and token[0] != kind) or (value and token[1] != value):
            raise ValueError("Malformed unit expression")
        self.index += 1
        return token

    def expression(self):
        scale, dimension, factors = self.term()
        while self.peek() in (("op", "*"), ("op", "/")):
            _, op = self.take("op")
            other_scale, other_dimension, other_factors = self.term()
            sign = 1 if op == "*" else -1
            scale = scale * other_scale if sign > 0 else scale / other_scale
            dimension = tuple(a + sign * b for a, b in zip(dimension, other_dimension))
            factors = factors + tuple((name, sign * power) for name, power in other_factors)
        return scale, dimension, factors

    def term(self):
        kind, value = self.peek()
        if kind == "op" and value == "(":
            self.take("op", "(")
            scale, dimension, factors = self.expression()
            self.take("op", ")")
        elif kind == "int" and value == "1":
            self.take()
            scale, dimension, factors = 1.0, _ZERO, ()
        elif kind == "name":
            self.take()
            scale, dimension = _lookup(value)
            factors = ((value, 1),)
        else:
            raise ValueError("Malformed unit expression")
        if self.peek() == ("op", "^"):
            self.take("op", "^")
            _, raw = self.take("int")
            power = int(raw)
            if power == 0 or abs(power) > 8:
                raise ValueError("Unit exponents must be nonzero integers with magnitude at most 8")
            scale = scale ** power
            dimension = tuple(item * power for item in dimension)
            factors = tuple((name, exponent * power) for name, exponent in factors)
        return scale, dimension, factors


def parse_unit(text: object) -> Unit:
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_UNIT_LENGTH:
        raise ValueError("A unit must be a nonempty bounded string")
    parser = _Parser(text)
    scale, dimension, factors = parser.expression()
    if parser.index != len(parser.tokens):
        raise ValueError(f"Malformed unit: {text!r}")
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"Unit scale must be finite and positive: {text!r}")
    return Unit(text, scale, dimension, factors)


def dimension_text(dimension: tuple[Fraction, ...]) -> str:
    parts = [f"{name}^{power}" if power != 1 else name for name, power in zip(BASE, dimension) if power]
    return "*".join(parts) if parts else "1"


def multiply(a: tuple[Fraction, ...], b: tuple[Fraction, ...], sign: int = 1) -> tuple[Fraction, ...]:
    return tuple(x + sign * y for x, y in zip(a, b))


def power(a: tuple[Fraction, ...], exponent: Fraction) -> tuple[Fraction, ...]:
    return tuple(x * exponent for x in a)


DIMENSIONLESS = _ZERO


def latex_unit(unit: Unit) -> str:
    """Render the declared factors in order, e.g. ``kg*m^2/s^2`` as ``\\mathrm{kg}\\,\\mathrm{m}^{2}\\,\\mathrm{s}^{-2}``."""
    if not unit.factors:
        return "1"
    rendered = []
    for name, exponent in unit.factors:
        text = name.replace("Ohm", r"\Omega").replace("deg", r"{}^{\circ}")
        if name.startswith("u") and len(name) > 1:
            text = r"\mu " + text[1:]
        body = rf"\mathrm{{{text}}}"
        rendered.append(body if exponent == 1 else rf"{body}^{{{exponent}}}")
    return r"\,".join(rendered)
