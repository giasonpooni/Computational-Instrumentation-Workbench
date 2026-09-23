"""A closed, data-only expression language for executable model equations.

Expressions are JSON trees, never source code in any host language:

* ``{"sym": "q"}`` names a declared quantity in the model's symbol table.
* ``{"num": 2}`` is a finite dimensionless literal.
* ``{"op": name, "args": [...]}`` applies one operator from ``OPERATORS``.

Evaluation is unit-aware: every symbol is supplied as its coherent SI value,
so an equation's meaning does not change when a quantity's declared unit is
rescaled. Dimensions are checked exactly with rational exponents.
"""
from __future__ import annotations

from fractions import Fraction
import math
from typing import Callable, Mapping

import numpy as np

from .units import DIMENSIONLESS, dimension_text, multiply, power

MAX_DEPTH = 48
MAX_NODES = 4096
_UNARY_DIMENSIONLESS = {"sin", "cos", "tan", "exp", "log", "tanh"}
OPERATORS = {
    "add": (2, None), "sub": (2, 2), "mul": (2, None), "div": (2, 2), "neg": (1, 1),
    "pow": (2, 2), "sqrt": (1, 1), "abs": (1, 1),
    **{name: (1, 1) for name in _UNARY_DIMENSIONLESS},
}


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite JSON number")
    return float(value)


def validate(node, symbols: Mapping[str, tuple], *, depth: int = 0, count: list | None = None) -> tuple[Fraction, ...]:
    """Validate structure and return the node's exact SI dimension.

    ``symbols`` maps each permitted symbol to its dimension vector.
    """
    count = [0] if count is None else count
    count[0] += 1
    if depth > MAX_DEPTH or count[0] > MAX_NODES:
        raise ValueError("Expression exceeds its depth or node budget")
    if not isinstance(node, dict):
        raise ValueError("Expression nodes must be objects")
    if node.keys() == {"sym"}:
        name = node["sym"]
        if not isinstance(name, str) or name not in symbols:
            raise ValueError(f"Expression references undeclared symbol: {name!r}")
        return symbols[name]
    if node.keys() == {"num"}:
        _finite(node["num"], "Expression literal")
        return DIMENSIONLESS
    if node.keys() != {"op", "args"} or node["op"] not in OPERATORS or not isinstance(node["args"], list):
        raise ValueError(f"Unsupported expression node: {sorted(node) if isinstance(node, dict) else node!r}")
    op, args = node["op"], node["args"]
    low, high = OPERATORS[op]
    if len(args) < low or (high is not None and len(args) > high):
        raise ValueError(f"Operator {op} has the wrong number of arguments")
    dims = [validate(arg, symbols, depth=depth + 1, count=count) for arg in args]
    if op in ("add", "sub"):
        if any(item != dims[0] for item in dims[1:]):
            raise ValueError(f"Dimensionally inconsistent {op}: "
                             + ", ".join(dimension_text(item) for item in dims))
        return dims[0]
    if op == "mul":
        result = DIMENSIONLESS
        for item in dims:
            result = multiply(result, item)
        return result
    if op == "div":
        return multiply(dims[0], dims[1], -1)
    if op in ("neg", "abs"):
        return dims[0]
    if op == "sqrt":
        return power(dims[0], Fraction(1, 2))
    if op == "pow":
        exponent = args[1]
        if dims[0] == DIMENSIONLESS:
            if dims[1] != DIMENSIONLESS:
                raise ValueError("A power exponent must be dimensionless")
            return DIMENSIONLESS
        if exponent.keys() != {"num"} or not float(exponent["num"]).is_integer() or abs(exponent["num"]) > 8:
            raise ValueError("A dimensioned base requires an integer literal exponent with magnitude at most 8")
        return power(dims[0], Fraction(int(exponent["num"])))
    if dims[0] != DIMENSIONLESS:
        raise ValueError(f"{op} requires a dimensionless argument, not {dimension_text(dims[0])}")
    return DIMENSIONLESS


def symbols_in(node) -> set[str]:
    if "sym" in node:
        return {node["sym"]}
    if "num" in node:
        return set()
    found: set[str] = set()
    for arg in node["args"]:
        found |= symbols_in(arg)
    return found


def rename(node, mapping: Mapping[str, str]):
    if "sym" in node:
        return {"sym": mapping.get(node["sym"], node["sym"])}
    if "num" in node:
        return {"num": node["num"]}
    return {"op": node["op"], "args": [rename(arg, mapping) for arg in node["args"]]}


def substitute(node, replacements: Mapping[str, dict]):
    """Replace symbols by expressions (used by typed-port composition)."""
    if "sym" in node:
        replacement = replacements.get(node["sym"])
        return substitute(replacement, replacements) if replacement is not None else {"sym": node["sym"]}
    if "num" in node:
        return {"num": node["num"]}
    return {"op": node["op"], "args": [substitute(arg, replacements) for arg in node["args"]]}


_NUMPY: dict[str, Callable] = {
    "sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "log": np.log,
    "tanh": np.tanh, "sqrt": np.sqrt, "abs": np.abs,
}


def evaluate(node, values: Mapping[str, object]):
    """Evaluate with SI values; NumPy scalars or equally shaped arrays.

    Binary64 NumPy semantics apply throughout: division by zero or a fractional
    power of a negative number yields a nonfinite value that callers treat as a
    halted computation, never a Python exception or a complex number.
    """
    if "sym" in node:
        return values[node["sym"]]
    if "num" in node:
        return np.float64(node["num"])
    args = [evaluate(arg, values) for arg in node["args"]]
    op = node["op"]
    if op == "add":
        result = args[0]
        for item in args[1:]:
            result = result + item
        return result
    if op == "mul":
        result = args[0]
        for item in args[1:]:
            result = result * item
        return result
    if op == "sub":
        return args[0] - args[1]
    if op == "div":
        return args[0] / args[1]
    if op == "neg":
        return -args[0]
    if op == "pow":
        return np.power(np.float64(args[0]) if np.ndim(args[0]) == 0 else args[0], args[1])
    return _NUMPY[op](args[0])


# Precedence: sum 1, product 2, unary minus 3, power 4, atom 5.
def to_latex(node, names: Mapping[str, str]) -> str:
    return _latex(node, names)[0]


def _number_latex(value) -> str:
    value = float(value)
    if value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    text = repr(value)
    if "e" in text:
        mantissa, exponent = text.split("e")
        return rf"{mantissa} \times 10^{{{int(exponent)}}}"
    return text


def _wrap(text: str, level: int, required: int) -> str:
    return rf"\left({text}\right)" if level < required else text


def _latex(node, names) -> tuple[str, int]:
    if "sym" in node:
        return names[node["sym"]], 5
    if "num" in node:
        value = float(node["num"])
        return _number_latex(value), (5 if value >= 0 else 3)
    op, args = node["op"], node["args"]
    parts = [_latex(arg, names) for arg in args]
    if op == "add":
        text = parts[0][0]
        for arg, (rendered, level) in zip(args[1:], parts[1:]):
            if arg.get("op") == "neg":
                inner, inner_level = _latex(arg["args"][0], names)
                text += " - " + _wrap(inner, inner_level, 2)
            else:
                text += " + " + _wrap(rendered, level, 2)
        return text, 1
    if op == "sub":
        return f"{parts[0][0]} - {_wrap(parts[1][0], parts[1][1], 2)}", 1
    if op == "mul":
        rendered = []
        for index, (text, level) in enumerate(parts):
            wrapped = _wrap(text, level, 3 if index == 0 else 4)
            rendered.append(wrapped)
        return r" \, ".join(rendered), 2
    if op == "div":
        return rf"\frac{{{parts[0][0]}}}{{{parts[1][0]}}}", 5
    if op == "neg":
        # Unary minus binds more loosely than a product: -2 gamma v needs no parentheses.
        return "-" + _wrap(parts[0][0], parts[0][1], 2), 3
    if op == "pow":
        # Brace the base so a decorated symbol never receives a double superscript.
        return f"{{{_wrap(parts[0][0], parts[0][1], 5)}}}^{{{parts[1][0]}}}", 4
    if op == "sqrt":
        return rf"\sqrt{{{parts[0][0]}}}", 5
    if op == "abs":
        return rf"\left|{parts[0][0]}\right|", 5
    return rf"\{op}\left({parts[0][0]}\right)", 5
