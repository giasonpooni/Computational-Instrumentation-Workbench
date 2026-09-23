"""Canonical JSON across languages (T146) and the compiled Rust probe shared with T142.

The specification fixes one byte encoding for identity-bearing JSON: object
keys sorted by Unicode code point, no insignificant whitespace, UTF-8 output,
finite binary64 numbers in the shortest round-trip form written exactly as
CPython's ``repr`` writes them, integers restricted to the interoperable range
``|n| <= 2**53 - 1``, and refusal of NaN, infinities, non-string keys, lone
surrogates and nesting deeper than 64 containers. An ASCII-escaped variant is
defined only to describe the existing ``ciw.core.identities`` hashes.

The Rust program below is compiled at run time with ``rustc`` into a temporary
directory (standard library only). Its agreement with the Python reference is
cross-language but same-origin: both were written for CIW, so agreement is a
numerical check, not independent verification. Nothing here states that other
language runtimes (Julia, C++, GPU hosts) produce the same bytes.
"""
from __future__ import annotations

import atexit
import hashlib
import json
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
    "floats": ("finite binary64 only; shortest round-trip digits; fixed notation when -4 < decpt <= 16 with "
               "at least one fractional digit ('1.0', '-0.0'), otherwise d[.ddd]e(+|-)XX with at least two "
               "exponent digits ('1e+16', '5e-324'); integers and floats are distinct types"),
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


def canonical_bytes(value, *, ascii_only: bool = False) -> bytes:
    """Reference encoder for the specification (``ascii_only`` gives the legacy variant)."""
    _validate(value, 0)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=ascii_only,
                      allow_nan=False).encode("utf-8")


def canonical_sha256(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


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
        ("unicode-bmp", "é日本語  "),
        ("unicode-supplementary", "\U0001f600\U0001d11e"),
        ("escapes-and-controls", "\"\\/\b\f\n\r\t\x00\x01\x1f\x7f"),
        ("no-normalization", ["é", "é"]),
        ("key-order-code-points", {"b": 1, "a": 2, "B": 3, "é": 4, "Ａ": 5, "\U0001f600": 6, "": 7,
                                   "a\x00b": 8, "aa": 9}),
        ("nested-keys", {"z": {"b": [1, {"y": None, "x": True}], "a": {}}, "a": [[], {}],
                         "m": {"k": {"j": {"i": 1.5}}}}),
        ("depth-64", _nested(64)),
        ("random-bit-pattern-floats", _random_floats(32, 146)),
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
            out.extend(["D", f"{struct.unpack('<Q', struct.pack('<d', node))[0]:016x}"])
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


def numpy_shortest(value: float) -> str:
    return np.format_float_scientific(value, unique=True, trim="-")


def ecmascript_number(value: float) -> str:
    """ECMA-262 Number::toString(10), as used by RFC 8785 (JCS), from shortest digits."""
    negative, digits, n = decimal_form(repr(value))
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

// Python float repr: shortest round-trip digits, exponent form when decpt <= -4 or decpt > 16.
fn number(x: f64) -> String {
    let s = format!("{:e}", x);
    let (mantissa, exponent) = s.split_once('e').unwrap();
    let exponent: i32 = exponent.parse().unwrap();
    let (negative, mantissa) = match mantissa.strip_prefix('-') { Some(m) => (true, m), None => (false, mantissa) };
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let decpt = exponent + 1;
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
            _ => "REFUSED unknown_mode".to_string(),
        };
        writeln!(out, "{}", result).expect("stdout is writable");
    }
}
'''
RUST_SOURCE_SHA256 = hashlib.sha256(RUST_SOURCE.encode("utf-8")).hexdigest()

# One compilation per process, keyed by the source digest; removed at exit.
_RUST_BUILD: dict = {}


def _cleanup(directory: str) -> None:
    shutil.rmtree(directory, ignore_errors=True)


def rust_build() -> dict:
    """Compile the probe with rustc once per process; returns identity, binary path or the failure."""
    if RUST_SOURCE_SHA256 in _RUST_BUILD:
        return _RUST_BUILD[RUST_SOURCE_SHA256]
    rustc = shutil.which("rustc")
    if rustc is None:
        result = {"available": False, "reason": "rustc is not on PATH"}
    else:
        directory = tempfile.mkdtemp(prefix="ciw-lab-rust-")
        atexit.register(_cleanup, directory)
        source = Path(directory) / "ciw_targets.rs"
        source.write_text(RUST_SOURCE, encoding="utf-8")
        binary = Path(directory) / ("ciw_targets.exe" if Path(rustc).suffix.lower() == ".exe" else "ciw_targets")
        try:
            version = subprocess.run([rustc, "-vV"], capture_output=True, text=True, timeout=60, check=True).stdout
            compiled = subprocess.run([rustc, "-O", "--edition", "2021", "-o", str(binary), str(source)],
                                      capture_output=True, text=True, timeout=300)
        except (OSError, subprocess.SubprocessError) as exc:
            result = {"available": False, "reason": f"rustc could not run: {type(exc).__name__}"}
        else:
            fields = dict(line.split(": ", 1) for line in version.splitlines() if ": " in line)
            identity = {"rustc_release": fields.get("release"), "rustc_commit": fields.get("commit-hash"),
                        "host": fields.get("host"), "source_sha256": RUST_SOURCE_SHA256,
                        "flags": ["-O", "--edition", "2021"]}
            if compiled.returncode != 0 or not binary.is_file():
                result = {"available": False, "reason": "rustc failed to compile the probe",
                          "stderr": compiled.stderr[-2000:], "identity": identity}
            else:
                identity["binary_sha256"] = hashlib.sha256(binary.read_bytes()).hexdigest()
                result = {"available": True, "binary": str(binary), "identity": identity}
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
