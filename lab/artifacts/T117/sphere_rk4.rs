// ciw.lab energy_gpu T117: unit-sphere geodesic RK4, std only, JSON over stdin/stdout.
use std::io::{self, Read, Write};

// Every right-hand-side call increments the counter, so the reported count is
// observed, not assumed from the step count.
fn rhs(y: &[f64; 4], evaluations: &mut u64) -> [f64; 4] {
    *evaluations += 1;
    let s = y[0].sin();
    let c = y[0].cos();
    [y[2], y[3], s * c * y[3] * y[3], -2.0 * (c / s) * y[2] * y[3]]
}

fn step(y: &[f64; 4], h: f64, evaluations: &mut u64) -> [f64; 4] {
    let hh = 0.5 * h;
    let k1 = rhs(y, evaluations);
    let mut t = [0.0; 4];
    for i in 0..4 { t[i] = y[i] + hh * k1[i]; }
    let k2 = rhs(&t, evaluations);
    for i in 0..4 { t[i] = y[i] + hh * k2[i]; }
    let k3 = rhs(&t, evaluations);
    for i in 0..4 { t[i] = y[i] + h * k3[i]; }
    let k4 = rhs(&t, evaluations);
    let h6 = h / 6.0;
    let mut out = [0.0; 4];
    for i in 0..4 { out[i] = y[i] + h6 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]); }
    out
}

fn numbers(text: &str) -> Result<Vec<f64>, String> {
    let mut values = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b.is_ascii_digit() || b == b'-' || b == b'.' {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_digit() || b"+-.eE".contains(&bytes[i])) { i += 1; }
            let token = &text[start..i];
            values.push(token.parse::<f64>().map_err(|_| format!("invalid number {}", token))?);
        } else {
            i += 1;
        }
    }
    Ok(values)
}

fn field<'a>(text: &'a str, key: &str) -> Result<&'a str, String> {
    let quoted = format!("\"{}\"", key);
    let at = text.find(&quoted).ok_or(format!("missing field {}", key))?;
    let rest = &text[at + quoted.len()..];
    let colon = rest.find(':').ok_or(format!("malformed field {}", key))?;
    let rest = &rest[colon + 1..];
    if rest.trim_start().starts_with('[') {
        let open = rest.find('[').unwrap();
        let mut depth = 0usize;
        for (offset, ch) in rest[open..].char_indices() {
            if ch == '[' { depth += 1; }
            if ch == ']' {
                depth -= 1;
                if depth == 0 { return Ok(&rest[open..open + offset + 1]); }
            }
        }
        Err(format!("unterminated array {}", key))
    } else {
        let end = rest.find(|c: char| c == ',' || c == '}').unwrap_or(rest.len());
        Ok(&rest[..end])
    }
}

fn run(text: &str) -> Result<String, String> {
    let length = *numbers(field(text, "length")?)?.first().ok_or("missing length")?;
    let steps = *numbers(field(text, "steps")?)?.first().ok_or("missing steps")?;
    let flat = numbers(field(text, "states")?)?;
    if !(length.is_finite() && length > 0.0) { return Err("length must be positive and finite".into()); }
    if !(steps >= 1.0 && steps <= 1.0e6 && steps.fract() == 0.0) { return Err("steps must be an integer in [1, 1e6]".into()); }
    if flat.is_empty() || flat.len() % 4 != 0 { return Err("states must be a list of 4-vectors".into()); }
    let n = steps as usize;
    let h = length / (n as f64);
    let mut rows = Vec::new();
    let mut evaluations: u64 = 0;
    for chunk in flat.chunks(4) {
        let mut y = [chunk[0], chunk[1], chunk[2], chunk[3]];
        for _ in 0..n {
            y = step(&y, h, &mut evaluations);
        }
        if !y.iter().all(|v| v.is_finite()) { return Err("nonfinite state".into()); }
        rows.push(format!("[{:?},{:?},{:?},{:?}]", y[0], y[1], y[2], y[3]));
    }
    Ok(format!("{{\"schema\":\"ciw.lab.rust-sphere-rk4.v1\",\"steps\":{},\"evaluations\":{},\"states\":[{}]}}",
               n, evaluations, rows.join(",")))
}

fn main() {
    let mut text = String::new();
    if io::stdin().read_to_string(&mut text).is_err() {
        eprintln!("unreadable input");
        std::process::exit(2);
    }
    match run(&text) {
        Ok(out) => { io::stdout().write_all(out.as_bytes()).unwrap(); }
        Err(message) => { eprintln!("{}", message); std::process::exit(2); }
    }
}
