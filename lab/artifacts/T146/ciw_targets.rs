// CIW implementation-target probe: canonical JSON and a fused sphere RK4 kernel.
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
