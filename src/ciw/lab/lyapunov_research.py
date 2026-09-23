"""Research-branch computations for T112-T114: ISS bound, residual adapter, servo pilot.

Scope: a disturbance-aware (input-to-state stability) research branch kept
outside the PLSR runtime; a host-side adapter that turns filtered residual
statistics into the four numeric fields of a ``plsr-sample-v1`` sample while
sensor and calibration metadata stay in a host envelope; and a
non-production servo-axis pilot specification with a synthetic nominal model.

Non-claims: every model, disturbance bound, noise level and parameter box
here is a declared synthetic value, not a measurement or an identification
result. Nothing here evaluates a real sensor, validates a calibration, grants
actuator authority or establishes machine safety.
"""
from __future__ import annotations

import json
import math

import numpy as np

from . import lyapunov_reference as R

# Matrix exponential ----------------------------------------------------------


def expm_series(M, terms=24):
    """exp(M) by scaling and squaring a truncated Taylor series (small dense matrices only)."""
    M = np.asarray(M, dtype=float)
    norm = float(np.max(np.sum(np.abs(M), axis=1)))
    squarings = max(0, int(math.ceil(math.log2(norm))) + 1) if norm > 0.5 else 0
    X = M / (2.0 ** squarings)
    result, term = np.eye(M.shape[0]), np.eye(M.shape[0])
    for k in range(1, terms + 1):
        term = term @ X / k
        result = result + term
    for _ in range(squarings):
        result = result @ result
    return result


# An eigendecomposition is a valid exponential only for a well-conditioned
# eigenvector basis; defective matrices (a ZOH-augmented integrator) have none.
EIGENVECTOR_CONDITION_LIMIT = 1e8


def independent_expm(M):
    """exp(M) from an implementation outside CIW, or None when none applies here.

    Returns (value, implementation, revision). SciPy's expm, else mpmath's
    expm at 40 digits, else a NumPy eigendecomposition when the eigenvector
    basis is well conditioned. A defective matrix without SciPy or mpmath gets
    None: the caller then relies on its closed-form reference alone.
    """
    M = np.asarray(M, dtype=float)
    try:
        import scipy
        from scipy.linalg import expm
    except ImportError:
        pass
    else:
        return expm(M), f"scipy.linalg.expm@{scipy.__version__}", scipy.__version__
    try:
        import mpmath
    except ImportError:
        pass
    else:
        with mpmath.workdps(40):
            value = mpmath.expm(mpmath.matrix(M.tolist()))
            result = np.array([[float(value[i, j]) for j in range(M.shape[1])] for i in range(M.shape[0])])
        return result, f"mpmath.expm@{mpmath.__version__}", mpmath.__version__
    values, vectors = np.linalg.eig(M)
    if not np.all(np.isfinite(vectors)) or np.linalg.cond(vectors) > EIGENVECTOR_CONDITION_LIMIT:
        return None
    value = (vectors @ np.diag(np.exp(values)) @ np.linalg.inv(vectors)).real
    return value, f"numpy.linalg.eig@{np.__version__}", np.__version__


def _phi1(x):
    """(1 - e^-x) / x, accurate for small x."""
    return -math.expm1(-x) / x if x != 0.0 else 1.0


def _phi2(x):
    """(e^-x - 1 + x) / x^2, by its alternating series for small x (the direct form cancels)."""
    if abs(x) > 0.1:
        return (math.expm1(-x) + x) / (x * x)
    total, term, k = 0.0, 0.5, 0
    while abs(term) > 1e-18:
        total += term
        k += 1
        term *= -x / (k + 2)
    return total


def damped_oscillator_zoh(A, B, h):
    """Closed-form augmented exponential exp([[A, B], [0, 0]] h) for a 2x2 A with complex eigenvalues.

    e^(A h) = e^(-s h) [cos(w h) I + sin(w h) / w (A + s I)] with eigenvalues -s +- i w, and
    Gamma = A^-1 (e^(A h) - I) B (A is invertible).
    """
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    s = -0.5 * float(np.trace(A))
    w = math.sqrt(float(np.linalg.det(A)) - s * s)
    Phi = math.exp(-s * h) * (math.cos(w * h) * np.eye(2) + math.sin(w * h) / w * (A + s * np.eye(2)))
    Gamma = np.linalg.solve(A, (Phi - np.eye(2)) @ B)
    return _augmented(Phi, Gamma)


def servo_zoh_closed_form(J, h):
    """Closed-form augmented exponential for J theta'' = -b theta' + Kt u (x = angle, rate), a = b/J.

    Phi = [[1, h phi1(a h)], [0, e^-(a h)]], Gamma = (Kt/J) [[h^2 phi2(a h)], [h phi1(a h)]].
    """
    a, k = SERVO["b_N_m_s"] / J, SERVO["Kt_N_m_per_A"] / J
    x = a * h
    Phi = np.array([[1.0, h * _phi1(x)], [0.0, math.exp(-x)]])
    Gamma = k * np.array([[h * h * _phi2(x)], [h * _phi1(x)]])
    return _augmented(Phi, Gamma)


def _augmented(Phi, Gamma):
    n, m = Phi.shape[0], Gamma.shape[1]
    result = np.eye(n + m)
    result[:n, :n], result[:n, n:] = Phi, Gamma
    return result


def zoh(A, B, h):
    """Exact zero-order-hold discretisation from the augmented exponential [[A, B], [0, 0]]."""
    n, m = A.shape[0], B.shape[1]
    augmented = np.zeros((n + m, n + m))
    augmented[:n, :n], augmented[:n, n:] = A, B
    E = expm_series(augmented * h)
    return E[:n, :n], E[:n, n:]


# T112: input-to-state stability research branch -----------------------------

ISS_A = np.array([[0.0, 1.0], [-4.0, -1.2]])
ISS_B = np.array([[0.0], [1.0]])
ISS_W = 0.5
ISS_STEP, ISS_HORIZON = 0.005, 30.0


def iss_bound(A, B, Q, w_bar):
    """Quadratic ISS-Lyapunov bound for x' = A x + B w, |w| <= w_bar.

    V = x^T P x with A^T P + P A = -Q gives V' <= -c V + 2 sqrt(V) beta, c = min eig(P^-1 Q),
    beta = ||P^(1/2) B|| w_bar, hence sqrt(V(t)) <= max(sqrt(V(0)), 2 beta / c) and
    limsup |x| <= (2 beta / c) / sqrt(min eig P).
    """
    P = R.kron_lyapunov(A, Q)
    L = np.linalg.cholesky(P)
    Li = np.linalg.inv(L)
    c = float(np.min(np.linalg.eigvalsh(Li @ Q @ Li.T)))
    beta = float(math.sqrt(float(np.max(np.linalg.eigvalsh(B.T @ P @ B))))) * w_bar
    radius_v = 2.0 * beta / c
    return {"P": P, "c": c, "beta": beta, "sqrt_V_bound": radius_v,
            "state_bound": radius_v / math.sqrt(float(np.min(np.linalg.eigvalsh(P))))}


def simulate_iss(A, B, P, w_bar, scenario, seed=112, step=ISS_STEP, horizon=ISS_HORIZON):
    """Exact ZOH simulation from x0 = 0 with a bounded, sample-held disturbance; returns sup sqrt(V) and sup |x|."""
    Phi, Gamma = zoh(A, B, step)
    rng = R.generator(seed)
    x = np.zeros(A.shape[0])
    sup_v, sup_x = 0.0, 0.0
    held = 0.0
    steps = int(round(horizon / step))
    for k in range(steps):
        t = k * step
        if scenario == "constant":
            w = w_bar
        elif scenario == "worst-case switching":
            w = w_bar if float(B[:, 0] @ P @ x) >= 0.0 else -w_bar
        elif scenario == "resonant sinusoid":
            w = w_bar * math.sin(math.sqrt(-A[1, 0]) * t)
        else:
            if k % 40 == 0:
                held = float(rng.uniform(-w_bar, w_bar))
            w = held
        x = Phi @ x + Gamma[:, 0] * w
        sup_v = max(sup_v, math.sqrt(max(float(x @ P @ x), 0.0)))
        sup_x = max(sup_x, float(np.linalg.norm(x)))
    return {"sup_sqrt_V": sup_v, "sup_state": sup_x, "final_state": x.tolist()}


def reachable_sup(A, B, P, w_bar, step=ISS_STEP, horizon=60.0, angles=361, refinements=60):
    """Largest sqrt(V) reachable from x(0) = 0 under any measurable |w| <= w_bar (single input, n = 2).

    With P = L L^T, sqrt(V(x(t))) = |L^T x(t)| and x(t) = int_0^t e^(A s) B w(t - s) ds, so the supremum over
    t and w is max over unit u of w_bar int_0^inf |u^T L^T e^(A s) B| ds (the support function of the
    reachable set). The integral is a trapezoid rule on the exact sampled impulse response; the direction is
    located on a grid over a half circle and refined by golden-section search in the bracketing cell.
    Sample-held disturbances are admissible inputs, so no held simulation can exceed it.
    """
    L = np.linalg.cholesky(np.asarray(P, dtype=float))
    Phi = expm_series(np.asarray(A, dtype=float) * step)
    steps = int(round(horizon / step))
    response = np.empty((steps + 1, 2))
    column = np.asarray(B, dtype=float)[:, 0].copy()
    for k in range(steps + 1):
        response[k] = L.T @ column
        column = Phi @ column

    def support(phi):
        magnitude = np.abs(response @ np.array([math.cos(phi), math.sin(phi)]))
        return w_bar * step * (float(magnitude.sum()) - 0.5 * float(magnitude[0] + magnitude[-1]))

    width = math.pi / (angles - 1)
    grid = [support(i * width) for i in range(angles)]
    best = int(np.argmax(grid))
    low, high = (best - 1) * width, (best + 1) * width
    ratio = 0.5 * (math.sqrt(5.0) - 1.0)
    a, b = high - ratio * (high - low), low + ratio * (high - low)
    fa, fb = support(a), support(b)
    for _ in range(refinements):
        if fa >= fb:
            high, b, fb = b, a, fa
            a = high - ratio * (high - low)
            fa = support(a)
        else:
            low, a, fa = a, b, fb
            b = low + ratio * (high - low)
            fb = support(b)
    phi = 0.5 * (low + high)
    value = max(support(phi), grid[best])
    return {"sup_sqrt_V": value, "direction_angle": phi, "step": step, "horizon": horizon, "angles": angles}


def scalar_iss(a=1.0, w_bar=0.5, horizon=ISS_HORIZON):
    """x' = -a x + w with w = w_bar: the quadratic ISS bound w_bar/a is attained asymptotically."""
    bound = iss_bound(np.array([[-a]]), np.array([[1.0]]), np.eye(1), w_bar)
    exact_sup = (w_bar / a) * (1.0 - math.exp(-a * horizon))
    return {"state_bound": bound["state_bound"], "exact_sup": exact_sup, "ratio": exact_sup / bound["state_bound"]}


ISS_SPEC = {
    "title": "Disturbance-aware (ISS) research branch for PLSR-style certificates",
    "status": "research branch; not a runtime feature and not part of runtime-status-v1",
    "definition": "x' = f(x, w) is input-to-state stable if |x(t)| <= beta(|x(0)|, t) + gamma(sup |w|) for a class-KL "
                  "beta and a class-K gamma.",
    "iss_lyapunov_function": "V with a1(|x|) <= V(x) <= a2(|x|) and V' <= -a3(|x|) + sigma(|w|); for linear "
                             "x' = A x + B w and V = x^T P x: V' <= -c V + 2 sqrt(V) ||P^(1/2) B|| |w| with "
                             "c = min eig(P^-1 Q).",
    "data_needed": [
        "the disturbance input matrix B (or a bound on how w enters), in the declared state units",
        "a disturbance bound w_bar with units, provenance and the procedure that established it",
        "whether w is sampled, held or continuous, and the sampling period if the model is discrete",
        "the declared model uncertainty that w is meant to cover, separately from exogenous disturbance",
        "the certificate P and Q used, with the Lyapunov-equation residual"],
    "plsr_must_not_claim": [
        "an ISS gain, ultimate bound or invariant set: runtime-status-v1 evaluates the undisturbed decrease only",
        "that a CERTIFIED_WITH_MARGIN sample is robust to disturbances or unmodelled dynamics",
        "that a declared required_margin corresponds to any physical disturbance level",
        "that a disturbance bound supplied by a host is true of a physical plant",
        "any new status code for disturbance robustness (a new code is a new runtime-status version)"],
    "numerical_illustration": "synthetic: A = [[0, 1], [-4, -1.2]], B = [0, 1]^T, w_bar = 0.5, Q = I; exact "
                              "ZOH simulation with h = 0.005 s over 30 s from x(0) = 0 for four bounded "
                              "disturbance classes, compared with the quadratic ISS bound and with the sharp "
                              "reachable-set supremum; plus the scalar system x' = -x + w, where the bound is "
                              "approached as t grows.",
    "open_questions": [
        "a resolution-aware float64 evaluation of the ISS inequality analogous to decrease_resolution",
        "sampled-data ISS: inter-sample behaviour of a discrete certificate under held disturbances",
        "how a host would evidence w_bar from acquired data (hardware-gated; not available here)"],
}


def iss_spec_markdown(bound, simulations, scalar, reachable) -> str:
    lines = [f"# {ISS_SPEC['title']}", "", f"Status: {ISS_SPEC['status']}.", "", "## Definition", "",
             ISS_SPEC["definition"], "", "## ISS-Lyapunov function", "", ISS_SPEC["iss_lyapunov_function"], "",
             "## Data a host must supply", ""]
    lines += [f"- {item}" for item in ISS_SPEC["data_needed"]]
    lines += ["", "## What PLSR must not claim", ""] + [f"- {item}" for item in ISS_SPEC["plsr_must_not_claim"]]
    lines += ["", "## Synthetic numerical illustration", "", ISS_SPEC["numerical_illustration"], "",
              f"Analytic bound: sqrt(V) <= {bound['sqrt_V_bound']:.6g} (c = {bound['c']:.6g}, beta = "
              f"{bound['beta']:.6g}); |x| <= {bound['state_bound']:.6g}.", "",
              "| Disturbance | sup sqrt(V) | sup sqrt(V) / bound | sup abs(x) |", "| --- | --- | --- | --- |"]
    for name, result in simulations.items():
        lines.append(f"| {name} | {result['sup_sqrt_V']:.6g} | {result['sup_sqrt_V'] / bound['sqrt_V_bound']:.4f} | "
                     f"{result['sup_state']:.6g} |")
    lines += ["", f"Sharp reference: the largest sqrt(V) reachable from x(0) = 0 under any |w| <= w_bar is "
                  f"{reachable['sup_sqrt_V']:.6g} ({reachable['sup_sqrt_V'] / bound['sqrt_V_bound']:.4f} of the ISS "
                  "bound); the quadratic ISS bound is conservative by the Cauchy-Schwarz step.",
              "", f"Scalar system: exact sup |x| = {scalar['exact_sup']:.12g}, bound = {scalar['state_bound']:.12g}.",
              "", "## Open research questions", ""] + [f"- {item}" for item in ISS_SPEC["open_questions"]]
    return "\n".join(lines) + "\n"


# T113: filtered residuals to kernel samples ---------------------------------

ADAPTER_H = 0.01
ADAPTER_BOX = (-0.5, 0.5)
ADAPTER_NOISE = 1e-3
ADAPTER_WINDOW = 400
MAX_AGE_S = 0.05
CERTIFICATE_VALID_UNTIL_S = 10.0
SAMPLE_FIELDS = ("sample_schema", "x", "theta", "theta_dot")
# The adapter forwards theta_hat and both ends of theta_hat +- GUARD_SE standard errors; the host accepts a
# window only when the kernel certifies all three, so a point estimate near a box bound cannot pass alone.
GUARD_SE = 3.0
SAMPLE_ROLES = ("estimate", "lower", "upper")


def adapter_plant():
    """Declared discrete one-step map A(theta) = A0 + theta A1 (Euler, h = 0.01 s), theta = stiffness deviation."""
    A0 = np.eye(2) + ADAPTER_H * np.array([[0.0, 1.0], [-4.0, -0.4]])
    A1 = ADAPTER_H * np.array([[0.0, 0.0], [-1.0, 0.0]])
    return A0, A1


def adapter_certificate():
    """Discrete Lyapunov P of the nominal map (Q = I) and its exact class at both box vertices."""
    A0, A1 = adapter_plant()
    P = R.kron_lyapunov(A0, np.eye(2), "discrete")
    classes = [R.exact_class(R.exact_form(A0 + theta * A1, P, "discrete")) for theta in ADAPTER_BOX]
    return P, classes


def _ekf(readings, process=1e-3, parameter_walk=1e-8, noise=ADAPTER_NOISE):
    """Extended Kalman filter on z = (position, velocity, theta) under the declared map; innovations give NIS."""
    A0, A1 = adapter_plant()
    z = np.array([readings[0], 0.0, 0.0])
    S = np.diag([1e-4, 1.0, 1.0])
    Qn = np.diag([0.0, process * process, parameter_walk])
    nis = []
    for y in readings[1:]:
        x, theta = z[:2], z[2]
        F = np.zeros((3, 3))
        F[:2, :2], F[:2, 2], F[2, 2] = A0 + theta * A1, A1 @ x, 1.0
        z = np.concatenate([(A0 + theta * A1) @ x, [theta]])
        S = F @ S @ F.T + Qn
        innovation, s = y - z[0], S[0, 0] + noise * noise
        gain = S[:, 0] / s
        z = z + gain * innovation
        S = S - np.outer(gain, S[0, :])
        nis.append(innovation * innovation / s)
    return z, S, np.array(nis)


def synthetic_window(theta_true, seed, dropout=False, damping=0.4, process=1e-3):
    """Simulate x+ = (I + h [[0, 1], [-(4 + theta), -damping]]) x + noise with noisy position readings (synthetic)."""
    rng = R.generator(seed)
    A = np.eye(2) + ADAPTER_H * np.array([[0.0, 1.0], [-(4.0 + theta_true), -damping]])
    x = np.array([1.0, 0.0])
    readings = []
    for _ in range(ADAPTER_WINDOW + 50):
        x = A @ x + np.array([0.0, process * float(rng.normal())])
        readings.append(float(x[0] + ADAPTER_NOISE * rng.normal()))
    readings = readings[50:]
    if dropout:
        readings[120] = float("nan")
    return readings


def residual_statistics(readings):
    """Filtered-residual statistics of one window: state and theta estimates, theta standard error, mean NIS."""
    z, S, nis = _ekf(readings)
    return {"x": z[:2].tolist(), "theta": float(z[2]), "theta_se": float(math.sqrt(S[2, 2])),
            "mean_nis": float(np.mean(nis)), "innovations": len(nis)}


def adapt(envelope):
    """Host-side adapter: a host-owned status, or plsr-sample-v1 samples with numbers only.

    The envelope carries sensor identity, units, calibration and timing; none of it is copied into a sample.
    Host statuses are decided here because the kernel cannot see sensors, clocks or issuance policy. theta is
    passed as estimated and at both ends of its GUARD_SE-standard-error interval, never clipped: box membership
    stays the kernel's decision, and the host accepts the window only if every sample is certified.
    """
    readings = envelope["readings"]
    if not all(math.isfinite(v) for v in readings):
        return {"host_status": "INVALID_SENSOR_DATA", "reason": "non-finite reading in the window"}
    if envelope["now_s"] - envelope["window_end_s"] > MAX_AGE_S:
        return {"host_status": "STALE_STATE", "reason": f"window older than {MAX_AGE_S} s"}
    if envelope["now_s"] > CERTIFICATE_VALID_UNTIL_S:
        return {"host_status": "CERTIFICATE_EXPIRED", "reason": "declared certificate validity has ended"}
    stats = residual_statistics(readings)
    # For a consistent filter the mean NIS over N innovations is 1 with standard deviation sqrt(2/N).
    band = 1.0 + 6.0 * math.sqrt(2.0 / stats["innovations"])
    if stats["mean_nis"] > band:
        return {"host_status": "MODEL_MISMATCH", "reason": f"mean NIS {stats['mean_nis']:.3g} above {band:.3g}",
                "statistics": stats}
    spread = GUARD_SE * stats["theta_se"]
    values = {"estimate": stats["theta"], "lower": stats["theta"] - spread, "upper": stats["theta"] + spread}
    samples = {role: {"sample_schema": "plsr-sample-v1", "x": [float(v) for v in stats["x"]], "theta": [values[role]],
                      "theta_dot": None} for role in SAMPLE_ROLES}
    return {"samples": samples, "statistics": stats}


def adapter_windows():
    """Synthetic host windows: two nominal, near-bound and out-of-box estimates, mismatch, stale, dropout, expired."""
    meta = {"sensor_id": "encoder-axis-7", "sensor_units": "m", "calibration_ref": "cal-2026-09-01-A",
            "calibration_valid_until": "2026-12-01T00:00:00Z", "filter": "EKF on (x, v, theta), r = 1e-6",
            "site": "synthetic-bench"}
    plan = [("nominal theta 0.1", 0.1, 0.4, 1131, False, 2.0, 2.01),
            ("nominal theta -0.3", -0.3, 0.4, 1132, False, 4.0, 4.01),
            ("estimate near bound theta 0.48", 0.48, 0.4, 1138, False, 3.0, 3.01),
            ("estimate outside box theta 0.9", 0.9, 0.4, 1136, False, 5.0, 5.01),
            ("structural mismatch damping 3", 0.1, 3.0, 1135, False, 6.0, 6.01),
            ("stale window", 0.1, 0.4, 1133, False, 7.0, 7.5),
            ("sensor dropout", 0.1, 0.4, 1134, True, 8.0, 8.01),
            ("certificate expired", 0.1, 0.4, 1137, False, 11.0, 11.01)]
    windows = []
    for name, theta, damping, seed, dropout, end, now in plan:
        windows.append({"name": name, "theta_true": theta, "damping_true": damping, "envelope": dict(
            meta, readings=synthetic_window(theta, seed, dropout, damping), window_end_s=end, now_s=now)})
    return windows


def _keys_and_strings(value, keys, strings):
    if isinstance(value, dict):
        for key, item in value.items():
            keys.add(str(key))
            _keys_and_strings(item, keys, strings)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _keys_and_strings(item, keys, strings)
    elif isinstance(value, str):
        strings.add(value)


def metadata_leaks(payload, envelope) -> list:
    """Envelope field names used as payload keys, and envelope strings appearing in any payload string value."""
    keys, strings = set(), set()
    _keys_and_strings(json.loads(json.dumps(payload)), keys, strings)
    leaks = {f"key {name}" for name in envelope if name != "readings" and name in keys}
    # Short strings (units such as "m") match only as whole values; longer ones also inside other strings.
    leaks |= {f"value {value}" for value in envelope.values() if isinstance(value, str)
              and any(value == text or (len(value) >= 3 and value in text) for text in strings)}
    return sorted(leaks)


# T114: servo-axis pilot -------------------------------------------------------

SERVO = {"J_nominal_kg_m2": 2e-3, "J_relative_uncertainty": 0.3, "b_N_m_s": 1e-3, "Kt_N_m_per_A": 0.1,
         "Ts_s": 1e-3, "bandwidth_hz": 20.0, "damping_ratio": 0.7}


def servo_models(grid=9):
    """Sampled-data closed loop A_cl(J) = Phi(J) - Gamma(J) K on a grid over the declared inertia interval."""
    J0, rel = SERVO["J_nominal_kg_m2"], SERVO["J_relative_uncertainty"]
    b, kt, ts = SERVO["b_N_m_s"], SERVO["Kt_N_m_per_A"], SERVO["Ts_s"]
    omega = 2.0 * math.pi * SERVO["bandwidth_hz"]
    K = np.array([[J0 * omega ** 2 / kt, (2.0 * SERVO["damping_ratio"] * omega * J0 - b) / kt]])
    inertias = sorted({J0 * (1.0 - rel), J0 * (1.0 + rel)} | set(np.linspace(J0 * (1.0 - rel), J0 * (1.0 + rel),
                                                                            grid).tolist()))
    models = []
    for J in inertias:
        A = np.array([[0.0, 1.0], [0.0, -b / J]])
        B = np.array([[0.0], [kt / J]])
        Phi, Gamma = zoh(A, B, ts)
        models.append({"J": J, "A": A, "B": B, "A_cl": Phi - Gamma @ K})
    return K, models


def servo_certificate(models):
    """Common P from the discrete Lyapunov equation at nominal J; unit-balanced Q = diag(omega^2, 1) versus Q = I.

    Returns P for the balanced weighting and the exact class of A_cl(J)^T P A_cl(J) - P at every grid point for
    both weightings. The grid is evidence, not a box certificate: A_cl is not affine in J.
    """
    omega = 2.0 * math.pi * SERVO["bandwidth_hz"]
    nominal = min(models, key=lambda m: abs(m["J"] - SERVO["J_nominal_kg_m2"]))
    classes, chosen = {}, None
    for label, Q in (("Q = I", np.eye(2)), ("Q = diag(omega^2, 1)", np.diag([omega * omega, 1.0]))):
        P = R.kron_lyapunov(nominal["A_cl"], Q, "discrete")
        classes[label] = [R.exact_class(R.exact_form(m["A_cl"], P, "discrete")) for m in models]
        chosen = P
    return chosen, classes


SERVO_SPEC_SECTIONS = ("purpose_and_status", "plant_model", "sampling", "identified_model_uncertainty",
                       "lyapunov_check_scope", "monitoring", "abort_criteria", "authority_and_safety")
# Declared operating envelope of the pilot (placeholders): the monitor's level set must lie inside it.
SERVO_ENVELOPE = {"angle_error_rad": 0.05, "velocity_rad_s": 2.0}
# Runtime codes the pilot aborts on. Each must be producible by the declared monitor configuration.
SERVO_ABORT_CODES = ("OUTSIDE_LEVEL_SET",)
# Codes the declared configuration is expected to produce; anything else signals a changed model or runtime.
SERVO_EXPECTED_CODES = ("CERTIFIED_WITH_MARGIN", "OUTSIDE_LEVEL_SET")


def servo_level(P):
    """Level c with {x^T P x <= c} inside the envelope box: (1 - 1e-6) min_i e_i^2 / (P^-1)_ii.

    max |x_i| over the ellipsoid is sqrt(c (P^-1)_ii); the 1e-6 shrink keeps rounding of P^-1 from pushing
    the ellipsoid past the box.
    """
    inverse = np.linalg.inv(np.asarray(P, dtype=float))
    bounds = (SERVO_ENVELOPE["angle_error_rad"], SERVO_ENVELOPE["velocity_rad_s"])
    return (1.0 - 1e-6) * float(min(bounds[i] ** 2 / inverse[i, i] for i in range(2)))


def monitor_scan(A_cl, P, level, count=200, seed=1141):
    """Online monitor codes at seeded states with V(x) = r^2 c, r = 10^U(-8, 8), by the documented decision order.

    Each state is evaluated without and with the declared level; the exact V(x) > c decision uses dyadic
    rationals. Returns the rows and the sets of codes each configuration produced.
    """
    from fractions import Fraction

    rng = R.generator(seed)
    P = np.asarray(P, dtype=float)
    exact_P, exact_level = R.fractions(P), Fraction(level)
    rows = []
    for _ in range(count):
        angle, radius = float(rng.uniform(0.0, 2.0 * math.pi)), float(10.0 ** rng.uniform(-8.0, 8.0))
        direction = np.array([math.cos(angle), math.sin(angle)])
        x = radius * direction * math.sqrt(level / float(direction @ P @ direction))
        rows.append({"x": x.tolist(), "radius": radius,
                     "exact_exceeds_level": R.exact_quadratic(x, exact_P) > exact_level,
                     "code_without_level": R.documented_code(A_cl, P, x, "discrete")["code"],
                     "code_with_level": R.documented_code(A_cl, P, x, "discrete", level=level)["code"]})
    return rows


def servo_spec(K, P, grid_classes, monitor):
    return {
        "purpose_and_status": "Non-production pilot of a Lyapunov monitor beside one servo axis on a test bench. "
                              "Monitoring only; results are research data, not acceptance evidence.",
        "plant_model": {"equation": "J theta'' = -b theta' + Kt i, current loop assumed ideal (i = u)",
                        "state": ["angle error [rad]", "angular velocity [rad/s]"],
                        "declared_values": SERVO, "controller": {"law": "u = -K x (sampled, zero-order hold)",
                                                                 "K": K.tolist()},
                        "provenance": "placeholder values chosen for this specification; not identified"},
        "sampling": {"period_s": SERVO["Ts_s"], "discretisation": "exact ZOH of the plant, closed loop "
                     "A_cl = Phi - Gamma K", "monitor_rate": "every control sample, evaluated off the control path"},
        "identified_model_uncertainty": {
            "parameter": "J", "interval": [SERVO["J_nominal_kg_m2"] * (1 - SERVO["J_relative_uncertainty"]),
                                           SERVO["J_nominal_kg_m2"] * (1 + SERVO["J_relative_uncertainty"])],
            "status": "to be identified from bench data (hardware-gated); the interval is a placeholder",
            "also_required": ["friction model and its uncertainty", "current-loop bandwidth", "encoder quantisation",
                              "delay between sampling and actuation"]},
        "lyapunov_check_scope": {
            "certificate": "common P from the discrete Lyapunov equation at nominal J with unit-balanced "
                           "Q = diag(omega^2, 1); Q = I fails on part of the interval (see grid_classes)",
            "offline_check": "exact negative definiteness of A_cl(J)^T P A_cl(J) - P on a grid over the J interval; "
                             "A_cl is not affine in J, so the grid is evidence, not a box certificate",
            "grid_classes": grid_classes,
            "online_use": "PLSR verdict on the declared nominal discrete model A_cl(J_nominal) with the balanced P at "
                          "host-estimated states, with the declared level c; runtime codes only",
            "state_dependence": "For a fixed declared model the sign of the decrease form does not depend on the "
                                "state, so without a level set every state receives the same code "
                                "(CERTIFIED_WITH_MARGIN here). The verdict carries information about the physical "
                                "axis only through the level set, i.e. through the estimated state leaving "
                                "{V <= c}; model validity is judged by the host's residual monitor "
                                "(MODEL_MISMATCH), not by the kernel.",
            "level_set": {"c": monitor["level"], "envelope": dict(SERVO_ENVELOPE),
                          "derivation": "{x^T P x <= c} lies inside |angle| <= 0.05 rad, |rate| <= 2 rad/s: "
                                        "c = (1 - 1e-6) min_i e_i^2 / (P^-1)_ii; the set is invariant for each "
                                        "grid model because V decreases there"},
            "runtime_codes": {"expected": list(SERVO_EXPECTED_CODES), "abort_on": list(SERVO_ABORT_CODES),
                              "produced_by_scan_with_level": monitor["codes_with_level"],
                              "produced_by_scan_without_level": monitor["codes_without_level"],
                              "cannot_occur_for_this_configuration": [
                                  "NOT_CERTIFIED, DECREASE_NOT_DEFINITE, NUMERICAL_INCONCLUSIVE and MARGIN_LOW "
                                  "(the declared decrease form is fixed and exactly negative definite with a "
                                  "margin far above the resolution; no required margin is declared)",
                                  "OUTSIDE_PARAMETER_BOX (a LinearPlant has no parameter box)",
                                  "CERTIFICATE_NOT_POSITIVE and NUMERICAL_OVERFLOW (fixed positive definite P of "
                                  "moderate scale; the state is power-of-two scaled before evaluation)"]},
            "not_covered": ["unmodelled dynamics", "saturation", "friction nonlinearity", "disturbances (see T112)",
                            "the inertia interval between grid points (the online model is the nominal one)"]},
        "monitoring": {"logged": ["runtime code and the three booleans per sample", "margin ratio", "host statuses",
                                  "state estimate and its covariance", "provider identity and model digest"],
                       "host_statuses": ["STALE_STATE", "INVALID_SENSOR_DATA", "MODEL_MISMATCH",
                                         "CERTIFICATE_EXPIRED", "RUNTIME_FAULT"]},
        "abort_criteria": [
            "OUTSIDE_LEVEL_SET on any sample: the estimated state left the certified sublevel set {V <= c} that "
            "lies inside the operating envelope",
            "any runtime code other than CERTIFIED_WITH_MARGIN or OUTSIDE_LEVEL_SET: impossible for the declared "
            "configuration, so it signals a changed model, certificate or runtime and is handled as RUNTIME_FAULT",
            "MODEL_MISMATCH from the host's residual monitor (mean NIS band, as in T113) on any window",
            "INVALID_SENSOR_DATA or RUNTIME_FAULT on any sample; STALE_STATE on 2 consecutive samples",
            "tracking error, velocity or current beyond the bench limits set by the safety function",
            "any operator request"],
        "authority_and_safety": {
            "actuator_authority": "none: the monitor cannot command motion, change gains or release a stop",
            "abort_action": "the host raises a stop request to the bench's independent safety function "
                            "(for example safe torque off); whether and how the axis stops is decided there",
            "safety_case": "outside this workbench; required before any powered test",
            "people": "no person within the axis envelope during powered tests"},
    }


def servo_spec_markdown(spec) -> str:
    lines = ["# Non-production servo-axis pilot specification", "",
             "Generated by CIW lab task T114. Monitoring only; no actuator authority; placeholder parameters.", ""]
    for section in SERVO_SPEC_SECTIONS:
        lines += [f"## {section.replace('_', ' ').capitalize()}", "", "```json",
                  json.dumps(R.jsonable(spec[section]), indent=1, sort_keys=True), "```", ""]
    return "\n".join(lines)
