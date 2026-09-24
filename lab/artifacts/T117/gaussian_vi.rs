// ciw.lab common workload: the Gaussian VI iteration of the gaussian_vi PTX kernel, std only.
// JSON over stdin/stdout. Every operation rounds to nearest in the declared precision; Rust never fuses a
// multiply and an add, matching the kernel's explicit .rn instructions.
use std::io::{self, Read, Write};

macro_rules! replica {
    ($name:ident, $t:ty) => {
        fn $name(v: &[f64; 15], iterations: u64, executed: &mut u64) -> [f64; 6] {
            let mut x = [0.0 as $t; 15];
            for i in 0..15 { x[i] = v[i] as $t; }
            let (p00, p01, p10, p11, b0, b1) = (x[0], x[1], x[2], x[3], x[4], x[5]);
            let (mut m0, mut m1) = (x[6], x[7]);
            let (mut q00, mut q01, mut q10, mut q11) = (x[8], x[9], x[10], x[11]);
            let (alpha, beta, omb) = (x[12], x[13], x[14]);
            for _ in 0..iterations {
                *executed += 1;
                let g0 = ((p00 * m0) + (p01 * m1)) - b0;
                let g1 = ((p10 * m0) + (p11 * m1)) - b1;
                m0 = m0 - (alpha * g0);
                m1 = m1 - (alpha * g1);
                q00 = (omb * q00) + (beta * p00);
                q01 = (omb * q01) + (beta * p01);
                q10 = (omb * q10) + (beta * p10);
                q11 = (omb * q11) + (beta * p11);
            }
            let det = (q00 * q11) - (q01 * q10);
            let out: [$t; 6] = [m0, m1, q11 / det, (-q01) / det, (-q10) / det, q00 / det];
            let mut wide = [0.0f64; 6];
            for i in 0..6 { wide[i] = out[i] as f64; }
            wide
        }
    };
}
replica!(replica_f64, f64);
replica!(replica_f32, f32);

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
        let close = rest.find(']').ok_or(format!("unterminated array {}", key))?;
        Ok(&rest[open..close + 1])
    } else {
        let end = rest.find(|c: char| c == ',' || c == '}').unwrap_or(rest.len());
        Ok(rest[..end].trim())
    }
}

fn integer(text: &str, key: &str, low: f64, high: f64) -> Result<u64, String> {
    let value = *numbers(field(text, key)?)?.first().ok_or(format!("missing {}", key))?;
    if !(value >= low && value <= high && value.fract() == 0.0) {
        return Err(format!("{} must be an integer in [{}, {}]", key, low, high));
    }
    Ok(value as u64)
}

fn run(text: &str) -> Result<String, String> {
    let precision = field(text, "precision")?.trim_matches('"').to_string();
    let iterations = integer(text, "iterations", 0.0, 256.0)?;
    let replicas = integer(text, "replicas", 1.0, 65536.0)?;
    let repeats = integer(text, "repeats", 1.0, 1000.0)?;
    let flat = numbers(field(text, "values")?)?;
    if flat.len() != 15 { return Err("values must hold the 15 prepared inputs".into()); }
    if !flat.iter().all(|v| v.is_finite()) { return Err("values must be finite".into()); }
    let mut values = [0.0f64; 15];
    values.copy_from_slice(&flat);
    let kernel: fn(&[f64; 15], u64, &mut u64) -> [f64; 6] = match precision.as_str() {
        "float64" => replica_f64,
        "float32" => replica_f32,
        _ => return Err("precision must be float64 or float32".into()),
    };
    let mut executed: u64 = 0;
    let mut first = [0.0f64; 6];
    let mut identical = true;
    for repeat in 0..repeats {
        for index in 0..replicas {
            // black_box keeps every replica's work: the inputs are opaque to the optimizer.
            let out = kernel(std::hint::black_box(&values), std::hint::black_box(iterations), &mut executed);
            if repeat == 0 && index == 0 { first = out; }
            else if out.iter().zip(first.iter()).any(|(a, b)| a.to_bits() != b.to_bits()) { identical = false; }
        }
    }
    if !first.iter().all(|v| v.is_finite()) { return Err("nonfinite output".into()); }
    let rows: Vec<String> = first.iter().map(|v| format!("{:?}", v)).collect();
    Ok(format!("{{\"schema\":\"ciw.lab.rust-gaussian-vi.v1\",\"precision\":\"{}\",\"iterations\":{},\"replicas\":{},\
\"repeats\":{},\"iterations_executed\":{},\"replicas_identical\":{},\"outputs\":[{}]}}",
               precision, iterations, replicas, repeats, executed, identical, rows.join(",")))
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
