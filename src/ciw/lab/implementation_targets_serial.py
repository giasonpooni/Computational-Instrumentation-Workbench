"""Canonical JSON across languages (T146) and the compiled Rust probe shared with T142.

The specification fixes one byte encoding for identity-bearing JSON: object
keys sorted by Unicode code point, no insignificant whitespace, UTF-8 output,
finite binary64 numbers in the shortest round-trip form (the candidate nearest
the exact binary value, decimal ties to an even digit) laid out as CPython's
``repr`` lays them out, integers restricted to ``|n| <= 2**53 - 1``, and
refusal of NaN, infinities, non-string keys, lone surrogates and nesting
deeper than 64 containers. An ASCII-escaped variant is defined only to
describe the existing ``ciw.core.identities`` hashes.

The reference encoder uses neither the ``json`` module nor ``repr``: strings
are escaped from an explicit table and floats come from an exact integer
search for the shortest round-trip digits, so comparing CIW's ``json``-based
encoders with it is not true by construction. It is still CIW-authored, as is
the Rust program below, which is compiled at run time with ``rustc`` into a
temporary directory (standard library only); agreement among them is a
numerical check, not independent verification. Nothing here states that other
language runtimes (Julia, C++, GPU hosts) produce the same bytes.
"""
from __future__ import annotations

import atexit
import hashlib
import math
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile

import numpy as np

SPEC_ID = "ciw.canonical-json.v1"
MAX_SAFE_INTEGER = 2 ** 53 - 1
MAX_DEPTH = 64

SPEC = {
    "id": SPEC_ID,
    "encoding": "UTF-8 without byte order mark; non-ASCII characters written raw",
    "whitespace": "none: separators are ',' and ':'",
    "object_keys": "strings only, unique, sorted by Unicode code point (equivalently by UTF-8 bytes)",
    "strings": ("escape '\"' and '\\\\'; \\b \\f \\n \\r \\t short forms; other U+0000-U+001F as \\u00xx "
                "(lowercase hex); everything else raw, including U+007F and U+2028; no Unicode normalization; "
                "lone surrogates refused"),
    "integers": "decimal without leading zeros or '+'; |n| <= 2**53 - 1, otherwise refused",
    "floats": ("finite binary64 only; digits: the fewest significant decimal digits k for which some k-digit "
               "decimal converts back to the same binary64 value; among such k-digit decimals the one nearest "
               "the exact binary value, and on an exact tie the one whose last digit is even (the CPython repr "
               "and ECMA-262 rule); layout: fixed notation when -4 < decpt <= 16 with at least one fractional "
               "digit ('1.0', '-0.0'), otherwise d[.ddd]e(+|-)XX with at least two exponent digits ('1e+16', "
               "'5e-324'); integers and floats are distinct types"),
    "literals": "null, true, false",
    "nesting": "a value inside more than 64 containers is refused",
    "refusals": ["nonfinite_number", "unsafe_integer", "non_string_key", "invalid_unicode", "nesting_too_deep",
                 "unsupported_type"],
    "ascii_variant": ("ciw.core.identities.canonical_json escapes every character outside U+0020-U+007E as "
                      "\\uxxxx (surrogate pairs above U+FFFF); it is described, not recommended"),
}


class CanonicalRefusal(ValueError):
    """A value has no canonical encoding under the specification."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _validate(value, depth: int) -> None:
    if depth > MAX_DEPTH:
        raise CanonicalRefusal("nesting_too_deep", f"Canonical JSON nests deeper than {MAX_DEPTH} containers")
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        if abs(value) > MAX_SAFE_INTEGER:
            raise CanonicalRefusal("unsafe_integer", "Canonical JSON integers must satisfy |n| <= 2**53 - 1")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalRefusal("nonfinite_number", "Canonical JSON refuses NaN and infinities")
        return
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            raise CanonicalRefusal("invalid_unicode", "Canonical JSON strings must be valid Unicode") from None
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _validate(item, depth + 1)
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalRefusal("non_string_key", "Canonical JSON object keys must be strings")
            _validate(key, depth + 1)
            _validate(item, depth + 1)
        return
    raise CanonicalRefusal("unsupported_type", f"Canonical JSON cannot encode {type(value).__name__}")


# ------------------------------------------------------ float digits
def _decimal_exponent(numerator: int, denominator: int) -> int:
    """e with 10**e <= numerator/denominator < 10**(e + 1), found exactly."""
    e = math.floor(math.log10(numerator) - math.log10(denominator))

    def at_least(power):  # numerator/denominator >= 10**power
        return numerator * 10 ** max(-power, 0) >= denominator * 10 ** max(power, 0)

    while not at_least(e):
        e -= 1
    while at_least(e + 1):
        e += 1
    return e


def _candidates(numerator: int, denominator: int, e: int, k: int):
    """k-digit decimals s * 10**-t on either side of the value, with their distances in units of 1/(D 10**t)."""
    t = k - 1 - e
    N = numerator * 10 ** max(t, 0)
    D = denominator * 10 ** max(-t, 0)
    low, remainder = divmod(N, D)
    if remainder == 0:
        return t, [(low, 0)]
    return t, [(low, remainder), (low + 1, D - remainder)]


def _round_trips(s: int, t: int, target: float) -> bool:
    # int / int and int -> float are correctly rounded in CPython; past the binary64 range nothing round-trips.
    try:
        return (s / 10 ** t if t >= 0 else float(s * 10 ** -t)) == target
    except OverflowError:
        return False


def shortest_digits(x: float) -> tuple:
    """(negative, digits, decpt, tie): value = 0.digits * 10**decpt under the specification's digit rule.

    ``tie`` is true when two shortest candidates round-trip at the same
    distance from the exact value (the even digit was chosen). Monotone in k:
    if a k-digit decimal round-trips, so does the finer (k + 1)-digit grid
    point on the same side, so the shortest k is found by bisection.
    """
    negative = math.copysign(1.0, x) < 0
    if x == 0.0:
        return negative, "0", 1, False
    numerator, denominator = abs(x).as_integer_ratio()
    e = _decimal_exponent(numerator, denominator)

    def passing(k):
        t, options = _candidates(numerator, denominator, e, k)
        return t, [(s, distance) for s, distance in options if _round_trips(s, t, abs(x))]

    low, high = 1, 17
    while low < high:
        middle = (low + high) // 2
        if passing(middle)[1]:
            high = middle
        else:
            low = middle + 1
    t, options = passing(low)
    options.sort(key=lambda item: (item[1], item[0] % 2))
    tie = len(options) == 2 and options[0][1] == options[1][1]
    digits = str(options[0][0])
    decpt = len(digits) - t
    return negative, digits.rstrip("0") or "0", decpt, tie


def format_float(x: float) -> str:
    """Specification float text: exact shortest digits in CPython repr layout."""
    negative, digits, decpt, _ = shortest_digits(x)
    sign = "-" if negative else ""
    if digits == "0":
        return sign + "0.0"
    if -4 < decpt <= 16:
        if decpt <= 0:
            body = "0." + "0" * -decpt + digits
        elif decpt < len(digits):
            body = digits[:decpt] + "." + digits[decpt:]
        else:
            body = digits + "0" * (decpt - len(digits)) + ".0"
    else:
        exponent = decpt - 1
        body = (digits[0] + ("." + digits[1:] if len(digits) > 1 else "") + "e"
                + ("-" if exponent < 0 else "+") + f"{abs(exponent):02d}")
    return sign + body


def decimal_tie(x: float) -> bool:
    return shortest_digits(x)[3]


# ------------------------------------------------------ reference encoder
_SHORT_ESCAPES = {'"': '\\"', "\\": "\\\\", "\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}


def _string(text: str, ascii_only: bool) -> str:
    parts = ['"']
    for character in text:
        code = ord(character)
        if character in _SHORT_ESCAPES:
            parts.append(_SHORT_ESCAPES[character])
        elif code < 0x20:
            parts.append(f"\\u{code:04x}")
        elif ascii_only and code > 0x7E:
            if code > 0xFFFF:
                code -= 0x10000
                parts.append(f"\\u{0xD800 + (code >> 10):04x}\\u{0xDC00 + (code & 0x3FF):04x}")
            else:
                parts.append(f"\\u{code:04x}")
        else:
            parts.append(character)
    parts.append('"')
    return "".join(parts)


def _encode(value, ascii_only: bool) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return format_float(value)
    if isinstance(value, str):
        return _string(value, ascii_only)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_encode(item, ascii_only) for item in value) + "]"
    # UTF-8 byte order equals code point order; validation already refused non-string keys.
    members = sorted(value.items(), key=lambda item: item[0].encode("utf-8"))
    return "{" + ",".join(_string(key, ascii_only) + ":" + _encode(item, ascii_only) for key, item in members) + "}"


def canonical_bytes(value, *, ascii_only: bool = False) -> bytes:
    """Reference encoder for the specification (``ascii_only`` gives the legacy variant)."""
    _validate(value, 0)
    return _encode(value, ascii_only).encode("utf-8")


def canonical_sha256(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


# Bytes written by hand from the specification text, independent of every encoder.
SPEC_EXAMPLES = (
    ([1e16, 1e15, 1e-5, 1e-4, -0.0, 5e-324, 1.0], b"[1e+16,1000000000000000.0,1e-05,0.0001,-0.0,5e-324,1.0]"),
    ([1e15 + 0.25, 123456789012345.625, 0.1, 1 / 3], b"[1000000000000000.2,123456789012345.62,0.1,0.3333333333333333]"),
    ({"int": 1, "float": 1.0, "b": [None, True, False, 9007199254740991]},
     b'{"b":[null,true,false,9007199254740991],"float":1.0,"int":1}'),
    ({"b": 1, "\U0001f600": 2, "\uff21": 3}, b'{"b":1,"\xef\xbc\xa1":3,"\xf0\x9f\x98\x80":2}'),
    ("\"\\/\b\f\n\r\t\x00\x1f\x7f\u2028\u00e9", b'"\\"\\\\/\\b\\f\\n\\r\\t\\u0000\\u001f\x7f\xe2\x80\xa8\xc3\xa9"'),
)


# ------------------------------------------------------------ test vectors
def _nested(depth: int):
    value = 0
    for _ in range(depth):
        value = [value]
    return value


def _random_floats(count: int, seed: int) -> list:
    """Finite binary64 values from uniformly random bit patterns (all exponents, both signs)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    out = []
    while len(out) < count:
        bits = int(rng.integers(0, 2 ** 64, dtype=np.uint64))
        value = struct.unpack("<d", struct.pack("<Q", bits))[0]
        if math.isfinite(value):
            out.append(value)
    return out


def tie_floats(per_scale: int = 24, seed: int = 1461) -> list:
    """binary64 values whose two shortest decimal candidates are equally near (exact decimal ties).

    N + j / 2**t with N of 18 - t digits and odd j has exactly 18 significant
    digits ending in 5, so its 17-digit neighbours are equidistant; values
    whose shortest form is shorter, or whose neighbours do not round-trip, are
    discarded by the exact rule.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    out = []
    for t in range(2, 14):
        low, high = 10 ** (17 - t), min(10 ** (18 - t), 2 ** (53 - t))
        for _ in range(per_scale):
            whole = int(rng.integers(low, high))
            odd = 2 * int(rng.integers(0, 2 ** (t - 1))) + 1
            x = (whole * 2 ** t + odd) / 2 ** t
            x = -x if rng.random() < 0.5 else x
            if decimal_tie(x):
                out.append(x)
    return out


def vectors() -> list:
    """Accepted test vectors: name and value; bytes and digests are derived."""
    return [
        ("empty-object", {}),
        ("empty-array", []),
        ("scalars", [None, True, False, 0, -1, 1]),
        ("safe-integer-limits", [MAX_SAFE_INTEGER, -MAX_SAFE_INTEGER]),
        ("integral-floats", [1.0, -1.0, 100.0, 9007199254740992.0, 1e15, 1e16, 1e21, 1e22]),
        ("short-floats", [0.1, 0.2, 0.3, 1 / 3, 2 / 3, 123.456, -2.5e-08]),
        ("exponent-switch", [1e-4, 1e-5, 0.00012345, 1234567890123456.0, 12345678901234567.0]),
        ("binary64-limits", [1.7976931348623157e308, -1.7976931348623157e308, 2.2250738585072014e-308,
                             2.225073858507201e-308, 5e-324, -5e-324]),
        ("signed-zero", [0.0, -0.0, {"z": -0.0}]),
        ("int-versus-float", {"float": 1.0, "int": 1, "neg_float": -0.0, "neg_int": 0}),
        # Written as escapes: U+2028 and U+00A0 are invisible in review, and editors may normalize composed text.
        ("unicode-bmp", "\u00e9\u65e5\u672c\u8a9e\u2028\u00a0"),
        ("unicode-supplementary", "\U0001f600\U0001d11e"),
        ("escapes-and-controls", "\"\\/\b\f\n\r\t\x00\x01\x1f\x7f"),
        ("no-normalization", ["\u00e9", "e\u0301"]),
        ("key-order-code-points", {"b": 1, "a": 2, "B": 3, "\u00e9": 4, "\uff21": 5, "\U0001f600": 6, "": 7,
                                   "a\x00b": 8, "aa": 9}),
        ("nested-keys", {"z": {"b": [1, {"y": None, "x": True}], "a": {}}, "a": [[], {}],
                         "m": {"k": {"j": {"i": 1.5}}}}),
        ("depth-64", _nested(64)),
        ("random-bit-pattern-floats", _random_floats(32, 146)),
        ("decimal-ties", [1e15 + 0.25, 123456789012345.625, 252611590017749.625, -29474221479446.8125]),
    ]


def invalid_vectors() -> list:
    """Values the specification refuses, with the expected refusal code."""
    return [
        ("nan", [float("nan")], "nonfinite_number"),
        ("positive-infinity", {"x": float("inf")}, "nonfinite_number"),
        ("negative-infinity", [1, float("-inf")], "nonfinite_number"),
        ("integer-2^53", 2 ** 53, "unsafe_integer"),
        ("integer-minus-2^53", [-(2 ** 53)], "unsafe_integer"),
        ("integer-10^30", 10 ** 30, "unsafe_integer"),
        ("non-string-key", {1: "x"}, "non_string_key"),
        ("lone-surrogate", "\ud800", "invalid_unicode"),
        ("depth-65", _nested(65), "nesting_too_deep"),
    ]


def float_corpus() -> dict:
    """Floats for digit checks: every vector float, 2000 random bit patterns and constructed decimal ties."""
    vector = [x for _, value in vectors() for x in floats_in(value)]
    return {"vector": vector, "random": _random_floats(2000, 1460), "ties": tie_floats()}


def rust_tokens(value) -> str:
    """Typed token stream for the Rust probe: floats travel as exact binary64 bit patterns."""
    out = []

    def emit(node):
        if node is None:
            out.append("N")
        elif node is True:
            out.append("T")
        elif node is False:
            out.append("F")
        elif isinstance(node, int):
            out.extend(["I", str(node)])
        elif isinstance(node, float):
            out.extend(["D", _bits(node)])
        elif isinstance(node, str):
            raw = node.encode("utf-8", "surrogatepass")
            out.extend(["S", raw.hex() or "-"])
        elif isinstance(node, (list, tuple)):
            out.extend(["A", str(len(node))])
            for item in node:
                emit(item)
        elif isinstance(node, dict):
            out.extend(["O", str(len(node))])
            for key, item in node.items():
                emit(key)
                emit(item)
        else:
            raise TypeError(f"No token form for {type(node).__name__}")

    emit(value)
    return " ".join(out)


# --------------------------------------------------- float digit analysis
def decimal_form(text: str) -> tuple:
    """(negative, digits, point) with value = 0.digits * 10**point, trailing zeros removed."""
    negative = text.startswith("-")
    text = text.lstrip("+-")
    mantissa, _, exponent = text.lower().partition("e")
    whole, _, fraction = mantissa.partition(".")
    digits = whole + fraction
    point = len(whole) + (int(exponent) if exponent else 0)
    stripped = digits.lstrip("0")
    point -= len(digits) - len(stripped)
    stripped = stripped.rstrip("0")
    if not stripped:
        return negative, "0", 0
    return negative, stripped, point


def reference_form(x: float) -> tuple:
    """The specification's (negative, digits, point) in :func:`decimal_form` convention."""
    negative, digits, decpt, _ = shortest_digits(x)
    return (negative, "0", 0) if digits == "0" else (negative, digits, decpt)


def numpy_shortest(value: float) -> str:
    return np.format_float_scientific(value, unique=True, trim="-")


def ecmascript_number(value: float) -> str:
    """ECMA-262 Number::toString(10), as used by RFC 8785 (JCS), from the exact shortest digits."""
    negative, digits, n = reference_form(value)
    if digits == "0":
        return "0"
    k = len(digits)
    if k <= n <= 21:
        body = digits + "0" * (n - k)
    elif 0 < n <= 21:
        body = digits[:n] + "." + digits[n:]
    elif -6 < n <= 0:
        body = "0." + "0" * (-n) + digits
    else:
        exponent = n - 1
        body = digits[0] + ("." + digits[1:] if k > 1 else "") + "e" + ("+" if exponent > 0 else "-") + str(abs(exponent))
    return ("-" if negative else "") + body


def floats_in(value) -> list:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, float):
        return [value]
    if isinstance(value, (list, tuple)):
        return [x for item in value for x in floats_in(item)]
    if isinstance(value, dict):
        return [x for item in value.values() for x in floats_in(item)]
    return []


# ------------------------------------------------------------- Rust probe
RUST_SOURCE = r'''// CIW implementation-target probe: canonical JSON and a fused sphere RK4 kernel.
// Standard library only. Reads token lines on stdin; writes one result line per input line.
use std::io::{self, BufRead, Write};
use std::time::Instant;

const MAX_DEPTH: usize = 64;
const MAX_SAFE: i64 = 9007199254740991;

enum V { Null, Bool(bool), Int(i64), Num(f64), Str(String), Arr(Vec<V>), Obj(Vec<(String, V)>) }

fn hex_bytes(t: &str) -> Result<Vec<u8>, &'static str> {
    if t == "-" { return Ok(Vec::new()); }
    if t.len() % 2 != 0 { return Err("malformed_input"); }
    (0..t.len()).step_by(2).map(|i| u8::from_str_radix(&t[i..i + 2], 16).map_err(|_| "malformed_input")).collect()
}

fn parse<'a, I: Iterator<Item = &'a str>>(tokens: &mut I, depth: usize) -> Result<V, &'static str> {
    if depth > MAX_DEPTH { return Err("nesting_too_deep"); }
    let tag = tokens.next().ok_or("malformed_input")?;
    match tag {
        "N" => Ok(V::Null),
        "T" => Ok(V::Bool(true)),
        "F" => Ok(V::Bool(false)),
        "I" => {
            let n: i64 = tokens.next().ok_or("malformed_input")?.parse().map_err(|_| "unsafe_integer")?;
            if n > MAX_SAFE || n < -MAX_SAFE { return Err("unsafe_integer"); }
            Ok(V::Int(n))
        }
        "D" => {
            let bits = u64::from_str_radix(tokens.next().ok_or("malformed_input")?, 16).map_err(|_| "malformed_input")?;
            let x = f64::from_bits(bits);
            if !x.is_finite() { return Err("nonfinite_number"); }
            Ok(V::Num(x))
        }
        "S" => {
            let raw = hex_bytes(tokens.next().ok_or("malformed_input")?)?;
            String::from_utf8(raw).map(V::Str).map_err(|_| "invalid_unicode")
        }
        "A" => {
            let n: usize = tokens.next().ok_or("malformed_input")?.parse().map_err(|_| "malformed_input")?;
            let mut items = Vec::with_capacity(n);
            for _ in 0..n { items.push(parse(tokens, depth + 1)?); }
            Ok(V::Arr(items))
        }
        "O" => {
            let n: usize = tokens.next().ok_or("malformed_input")?.parse().map_err(|_| "malformed_input")?;
            let mut members: Vec<(String, V)> = Vec::with_capacity(n);
            for _ in 0..n {
                if tokens.next() != Some("S") { return Err("non_string_key"); }
                let raw = hex_bytes(tokens.next().ok_or("malformed_input")?)?;
                let key = String::from_utf8(raw).map_err(|_| "invalid_unicode")?;
                let value = parse(tokens, depth + 1)?;
                members.push((key, value));
            }
            // Byte order of UTF-8 equals Unicode code point order.
            members.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
            if members.windows(2).any(|w| w[0].0 == w[1].0) { return Err("duplicate_key"); }
            Ok(V::Obj(members))
        }
        _ => Err("malformed_input"),
    }
}

// Python float repr: shortest round-trip digits, the candidate nearest the exact binary value with decimal
// ties to an even digit, exponent form when decpt <= -4 or decpt > 16. {:e} gives the shortest digit count
// but breaks exact decimal ties upward, so the digits come from correctly rounded formatting at that count
// (falling back to {:e} if that string does not round-trip).
fn decimal_digits(x: f64) -> (bool, String, i32) {
    let shortest = format!("{:e}", x);
    let count = shortest.split_once('e').unwrap().0.trim_start_matches('-').replace('.', "").len();
    let exact = format!("{:.*e}", count - 1, x);
    let chosen = if exact.parse::<f64>() == Ok(x) { exact } else { shortest };
    let (mantissa, exponent) = chosen.split_once('e').unwrap();
    let exponent: i32 = exponent.parse().unwrap();
    let (negative, mantissa) = match mantissa.strip_prefix('-') { Some(m) => (true, m), None => (false, mantissa) };
    let all: String = mantissa.chars().filter(|c| *c != '.').collect();
    let trimmed = all.trim_end_matches('0');
    (negative, if trimmed.is_empty() { "0".to_string() } else { trimmed.to_string() }, exponent + 1)
}

fn number(x: f64) -> String {
    let (negative, digits, decpt) = decimal_digits(x);
    let n = digits.len() as i32;
    let mut out = String::new();
    if negative { out.push('-'); }
    if decpt > -4 && decpt <= 16 {
        if decpt <= 0 {
            out.push_str("0.");
            for _ in 0..(-decpt) { out.push('0'); }
            out.push_str(&digits);
        } else if decpt < n {
            out.push_str(&digits[..decpt as usize]);
            out.push('.');
            out.push_str(&digits[decpt as usize..]);
        } else {
            out.push_str(&digits);
            for _ in 0..(decpt - n) { out.push('0'); }
            out.push_str(".0");
        }
    } else {
        out.push_str(&digits[..1]);
        if n > 1 { out.push('.'); out.push_str(&digits[1..]); }
        let e = decpt - 1;
        out.push_str(&format!("e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs()));
    }
    out
}

// Rust's own shortest formatting, kept to show where it departs from the specification.
fn shortest(line: &str) -> String {
    match u64::from_str_radix(line.trim(), 16) {
        Ok(bits) => format!("OK {:e}", f64::from_bits(bits)),
        Err(_) => "REFUSED malformed_input".to_string(),
    }
}

fn string(s: &str, ascii: bool, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c if ascii && (c as u32) > 0x7e => {
                let mut units = [0u16; 2];
                for unit in c.encode_utf16(&mut units) { out.push_str(&format!("\\u{:04x}", unit)); }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn emit(v: &V, ascii: bool, out: &mut String) {
    match v {
        V::Null => out.push_str("null"),
        V::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        V::Int(n) => out.push_str(&n.to_string()),
        V::Num(x) => out.push_str(&number(*x)),
        V::Str(s) => string(s, ascii, out),
        V::Arr(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() { if i > 0 { out.push(','); } emit(item, ascii, out); }
            out.push(']');
        }
        V::Obj(members) => {
            out.push('{');
            for (i, (k, item)) in members.iter().enumerate() {
                if i > 0 { out.push(','); }
                string(k, ascii, out);
                out.push(':');
                emit(item, ascii, out);
            }
            out.push('}');
        }
    }
}

fn hex(s: &str) -> String { s.bytes().map(|b| format!("{:02x}", b)).collect() }

fn canon(line: &str) -> String {
    let mut tokens = line.split_whitespace();
    match parse(&mut tokens, 0) {
        Ok(v) => {
            if tokens.next().is_some() { return "REFUSED malformed_input".to_string(); }
            let (mut utf8, mut ascii) = (String::new(), String::new());
            emit(&v, false, &mut utf8);
            emit(&v, true, &mut ascii);
            format!("OK {} {}", hex(&utf8), hex(&ascii))
        }
        Err(code) => format!("REFUSED {}", code),
    }
}

// Sphere of radius r in the polar chart: geodesic plus both normal Jacobi columns.
fn sphere_rhs(y: &[f64; 8], r: f64) -> [f64; 8] {
    let (s, c) = (y[0].sin(), y[0].cos());
    let k = 1.0 / (r * r);
    [y[2], y[3], s * c * y[3] * y[3], -2.0 * (c / s) * y[2] * y[3], y[5], -k * y[4], y[7], -k * y[6]]
}

fn rk4(line: &str) -> String {
    let t: Vec<&str> = line.split_whitespace().collect();
    if t.len() != 11 { return "REFUSED malformed_input".to_string(); }
    let mut y = [0.0f64; 8];
    for i in 0..8 { y[i] = f64::from_bits(u64::from_str_radix(t[i], 16).unwrap_or(0)); }
    let length = f64::from_bits(u64::from_str_radix(t[8], 16).unwrap_or(0));
    let steps: usize = t[9].parse().unwrap_or(0);
    let r = f64::from_bits(u64::from_str_radix(t[10], 16).unwrap_or(0));
    if steps == 0 || !(r > 0.0) { return "REFUSED malformed_input".to_string(); }
    let h = length / steps as f64;
    let start = Instant::now();
    for _ in 0..steps {
        let k1 = sphere_rhs(&y, r);
        let mut z = [0.0; 8];
        for i in 0..8 { z[i] = y[i] + 0.5 * h * k1[i]; }
        let k2 = sphere_rhs(&z, r);
        for i in 0..8 { z[i] = y[i] + 0.5 * h * k2[i]; }
        let k3 = sphere_rhs(&z, r);
        for i in 0..8 { z[i] = y[i] + h * k3[i]; }
        let k4 = sphere_rhs(&z, r);
        for i in 0..8 { y[i] = y[i] + (h / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]); }
    }
    let elapsed = start.elapsed().as_nanos();
    let bits: Vec<String> = y.iter().map(|v| format!("{:016x}", v.to_bits())).collect();
    format!("OK {} {}", bits.join(" "), elapsed)
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_default();
    let stdin = io::stdin();
    let mut out = io::BufWriter::new(io::stdout());
    for line in stdin.lock().lines() {
        let line = line.expect("stdin is readable");
        let result = match mode.as_str() {
            "canon" => canon(&line),
            "rk4" => rk4(&line),
            "shortest" => shortest(&line),
            _ => "REFUSED unknown_mode".to_string(),
        };
        writeln!(out, "{}", result).expect("stdout is writable");
    }
}
'''
RUST_SOURCE_SHA256 = hashlib.sha256(RUST_SOURCE.encode("utf-8")).hexdigest()
# The build directory is remapped so the binary (panic locations) carries no temporary path.
REMAPPED_BUILD_DIR = "ciw-lab-rust"
RUST_FLAGS = ("-O", "--edition", "2021")

# One compilation per process, keyed by the source digest; removed at exit.
_RUST_BUILD: dict = {}


def _cleanup(directory: str) -> None:
    shutil.rmtree(directory, ignore_errors=True)


def _compile(rustc: str, directory: str, source_text: str, stem: str):
    source = Path(directory) / f"{stem}.rs"
    source.write_text(source_text, encoding="utf-8")
    binary = Path(directory) / (stem + (".exe" if Path(rustc).suffix.lower() == ".exe" else ""))
    completed = subprocess.run([rustc, *RUST_FLAGS, f"--remap-path-prefix={directory}={REMAPPED_BUILD_DIR}",
                                "-o", str(binary), str(source)], capture_output=True, text=True, timeout=300)
    return completed, binary


def rust_build() -> dict:
    """Compile the probe with rustc once per process.

    ``available``: the probe binary exists. ``usable``: rustc is on PATH and
    can build a trivial program, so a probe that fails to build is a defect
    of the probe (a refutation), not a missing tool.
    """
    if RUST_SOURCE_SHA256 in _RUST_BUILD:
        return _RUST_BUILD[RUST_SOURCE_SHA256]
    rustc = shutil.which("rustc")
    if rustc is None:
        result = {"available": False, "usable": False, "reason": "rustc is not on PATH"}
    else:
        directory = tempfile.mkdtemp(prefix="ciw-lab-rust-")
        atexit.register(_cleanup, directory)
        try:
            version = subprocess.run([rustc, "-vV"], capture_output=True, text=True, timeout=60, check=True).stdout
            compiled, binary = _compile(rustc, directory, RUST_SOURCE, "ciw_targets")
        except (OSError, subprocess.SubprocessError) as exc:
            result = {"available": False, "usable": False, "reason": f"rustc could not run: {type(exc).__name__}"}
        else:
            fields = dict(line.split(": ", 1) for line in version.splitlines() if ": " in line)
            identity = {"rustc_release": fields.get("release"), "rustc_commit": fields.get("commit-hash"),
                        "host": fields.get("host"), "source_sha256": RUST_SOURCE_SHA256,
                        "flags": [*RUST_FLAGS, f"--remap-path-prefix=<build dir>={REMAPPED_BUILD_DIR}"]}
            if compiled.returncode == 0 and binary.is_file():
                identity["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
                result = {"available": True, "usable": True, "binary": str(binary), "identity": identity}
            else:
                try:
                    trivial, trivial_binary = _compile(rustc, directory, "fn main() {}\n", "ciw_trivial")
                    usable = trivial.returncode == 0 and trivial_binary.is_file()
                except (OSError, subprocess.SubprocessError):
                    usable = False
                result = {"available": False, "usable": usable, "identity": identity,
                          "reason": ("rustc failed to compile the probe" if usable
                                     else "rustc cannot build a trivial program here (linker or target missing)"),
                          "stderr": compiled.stderr[-2000:]}
    _RUST_BUILD[RUST_SOURCE_SHA256] = result
    return result


def rust_run(mode: str, lines: list) -> list:
    build = rust_build()
    if not build["available"]:
        raise RuntimeError(build["reason"])
    completed = subprocess.run([build["binary"], mode], input="\n".join(lines) + "\n", capture_output=True,
                               text=True, timeout=120, check=True)
    out = completed.stdout.splitlines()
    if len(out) != len(lines):
        raise RuntimeError("Rust probe returned a different number of lines than it was given")
    return out


def rust_canonical(values: list) -> list:
    """[(utf8_bytes, ascii_bytes) or refusal code] for each value."""
    results = []
    for line in rust_run("canon", [rust_tokens(value) for value in values]):
        parts = line.split()
        if parts[0] == "OK":
            results.append((bytes.fromhex(parts[1]), bytes.fromhex(parts[2])))
        else:
            results.append(parts[1])
    return results


def rust_shortest(values: list) -> list:
    """Rust's own ``format!("{:e}", x)`` text for each float (not the specification's digits)."""
    out = []
    for line in rust_run("shortest", [_bits(value) for value in values]):
        parts = line.split()
        if parts[0] != "OK":
            raise RuntimeError(f"Rust probe refused a float: {parts[1]}")
        out.append(parts[1])
    return out


def _bits(value: float) -> str:
    return f"{struct.unpack('<Q', struct.pack('<d', float(value)))[0]:016x}"


def rust_sphere_rk4(state0, length: float, steps: int, radius: float) -> tuple:
    """Final geodesic/Jacobi state from the fused Rust loop and its elapsed nanoseconds."""
    line = " ".join([*(_bits(v) for v in state0), _bits(length), str(int(steps)), _bits(radius)])
    parts = rust_run("rk4", [line])[0].split()
    if parts[0] != "OK":
        raise RuntimeError(f"Rust probe refused the RK4 request: {parts[1]}")
    state = [struct.unpack("<d", struct.pack("<Q", int(token, 16)))[0] for token in parts[1:9]]
    return np.array(state), int(parts[9])
