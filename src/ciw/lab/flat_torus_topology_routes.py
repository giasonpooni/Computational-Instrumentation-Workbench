"""Candidate geodesic routes between two points on curved surfaces (T024, T025, T032).

Routes p -> q are found by shooting: a fan of headings is integrated in one
vectorized RK4 batch with closed-form Christoffel symbols, near-passes of every
chart lift of q seed a Newton iteration on (heading, length) whose Jacobian is
[velocity, j_head * normal] (the heading Jacobi field), and every converged
route is then re-integrated with :func:`ciw.lab.jacobi.transfer`, which
supplies the reported amplification |j_head(L)| and the conjugate points of
the extended geodesic. The batch integrator is only a search device; reported
numbers come from ``ciw.lab.jacobi`` and are cross-checked against it.

The candidate set is whatever the fan search found up to a declared length;
it is not proven complete. Nothing here models a vehicle, a tool or a
workspace, and no route is claimed safe to execute.
"""
from __future__ import annotations

import math

import numpy as np

from . import jacobi
from .surfaces import GaussianBump, Saddle, Sphere, Torus

TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------- closed-form vector fields
def batch_field(surface):
    """Vectorized geodesic + heading-Jacobi field for states (u1, u2, v1, v2, j, j')."""
    if isinstance(surface, Torus):
        big, small = surface.major, surface.minor

        def f(Y):
            st, ct = np.sin(Y[:, 1]), np.cos(Y[:, 1])
            rho = big + small * ct
            out = np.empty_like(Y)
            out[:, 0], out[:, 1] = Y[:, 2], Y[:, 3]
            out[:, 2] = 2.0 * small * st / rho * Y[:, 2] * Y[:, 3]
            out[:, 3] = -rho * st / small * Y[:, 2] ** 2
            out[:, 4] = Y[:, 5]
            out[:, 5] = -(ct / (small * rho)) * Y[:, 4]
            return out
        return f
    if isinstance(surface, GaussianBump):
        h, s2 = surface.h, surface.sigma ** 2

        def f(Y):
            x, y, vx, vy = Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3]
            height = h * np.exp(-(x * x + y * y) / (2 * s2))
            fx, fy = -x / s2 * height, -y / s2 * height
            fxx, fyy = (x * x / s2 - 1) / s2 * height, (y * y / s2 - 1) / s2 * height
            fxy = x * y / (s2 * s2) * height
            w = 1.0 + fx * fx + fy * fy
            second = fxx * vx * vx + 2 * fxy * vx * vy + fyy * vy * vy
            out = np.empty_like(Y)
            out[:, 0], out[:, 1] = vx, vy
            out[:, 2], out[:, 3] = -fx * second / w, -fy * second / w
            out[:, 4] = Y[:, 5]
            out[:, 5] = -((fxx * fyy - fxy * fxy) / (w * w)) * Y[:, 4]
            return out
        return f
    if isinstance(surface, Saddle):
        c = surface.c

        def f(Y):
            x, y, vx, vy = Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3]
            fx, fy = c * x, -c * y
            w = 1.0 + fx * fx + fy * fy
            second = c * (vx * vx - vy * vy)
            out = np.empty_like(Y)
            out[:, 0], out[:, 1] = vx, vy
            out[:, 2], out[:, 3] = -fx * second / w, -fy * second / w
            out[:, 4] = Y[:, 5]
            out[:, 5] = (c * c / (w * w)) * Y[:, 4]
            return out
        return f
    raise ValueError(f"No closed-form batch field for surface {surface.name}")


def _monge_gradient(surface, u):
    x, y = u[..., 0], u[..., 1]
    if isinstance(surface, GaussianBump):
        s2 = surface.sigma ** 2
        height = surface.h * np.exp(-(x * x + y * y) / (2 * s2))
        return -x / s2 * height, -y / s2 * height
    if isinstance(surface, Saddle):
        return surface.c * x, -surface.c * y
    raise ValueError(f"No closed-form gradient for surface {surface.name}")


def rk4_batch(f, Y0, lengths, steps, record=False):
    """Fixed-step RK4 over [0, lengths[i]] for every row; optionally keep every node."""
    Y = np.array(Y0, dtype=float)
    h = (np.asarray(lengths, dtype=float) / steps)[:, None]
    history = [Y.copy()] if record else None
    for _ in range(steps):
        k1 = f(Y)
        k2 = f(Y + 0.5 * h * k1)
        k3 = f(Y + 0.5 * h * k2)
        k4 = f(Y + h * k3)
        Y = Y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if record:
            history.append(Y.copy())
    return (Y, np.array(history)) if record else Y


def initial_states(surface, u0, headings):
    u0 = np.asarray(u0, dtype=float)
    rows = []
    for heading in headings:
        t0 = surface.unit_tangent(u0, float(heading))
        rows.append([u0[0], u0[1], t0[0], t0[1], 0.0, 1.0])
    return np.array(rows)


def periodic(surface) -> bool:
    return isinstance(surface, Torus)


def chart_offset(surface, u, target):
    """Displacement u - target reduced to the nearest chart lift, and that lift's integer shift."""
    delta = np.asarray(u, dtype=float) - np.asarray(target, dtype=float)
    if not periodic(surface):
        return delta, np.zeros_like(delta)
    lift = np.round(delta / TWO_PI)
    return delta - TWO_PI * lift, lift


def metric_norm(surface, u, delta):
    """sqrt(delta^T g(u) delta), vectorized over rows (a local distance proxy)."""
    if isinstance(surface, Torus):
        rho = surface.major + surface.minor * np.cos(u[..., 1])
        return np.sqrt((rho * delta[..., 0]) ** 2 + (surface.minor * delta[..., 1]) ** 2)
    # Monge graph: g = I + grad f grad f^T, so d^T g d = |d|^2 + (grad f . d)^2.
    fx, fy = _monge_gradient(surface, u)
    return np.sqrt(delta[..., 0] ** 2 + delta[..., 1] ** 2 + (fx * delta[..., 0] + fy * delta[..., 1]) ** 2)


def fan_candidates(surface, p, q, max_length, headings=360, step=0.025, threshold=0.35):
    """Near-passes of any lift of q along a fan of geodesics from p (heading, s, lift)."""
    f = batch_field(surface)
    thetas = np.linspace(0.0, TWO_PI, headings, endpoint=False)
    steps = int(math.ceil(max_length / step))
    Y0 = initial_states(surface, p, thetas)
    _, history = rk4_batch(f, Y0, np.full(len(thetas), max_length), steps, record=True)
    s = np.linspace(0.0, max_length, steps + 1)
    points = history[:, :, :2]
    delta = points - np.asarray(q)[None, None, :]
    lift = np.round(delta / TWO_PI) if periodic(surface) else np.zeros_like(delta)
    reduced = delta - TWO_PI * lift
    dist = metric_norm(surface, points.reshape(-1, 2), reduced.reshape(-1, 2)).reshape(points.shape[:2])
    candidates = []
    for j in range(len(thetas)):
        d = dist[:, j]
        for i in range(1, len(s) - 1):
            if d[i] <= d[i - 1] and d[i] < d[i + 1] and d[i] < threshold and s[i] > 0.2:
                candidates.append((float(thetas[j]), float(s[i]), tuple(int(v) for v in lift[i, j])))
    return candidates


def _normal_rows(surface, U, V):
    return np.array([surface.normal(u, v) for u, v in zip(U, V)])


def newton_routes(surface, p, q, candidates, steps=600, iterations=12, tol=1e-10, max_length=None):
    """Refine (heading, length) for every candidate so the geodesic ends on its lift of q."""
    if not candidates:
        return []
    f = batch_field(surface)
    theta = np.array([c[0] for c in candidates])
    length = np.array([c[1] for c in candidates])
    lifts = np.array([c[2] for c in candidates], dtype=float)
    target = np.asarray(q, dtype=float)[None, :] + TWO_PI * lifts
    active = np.ones(len(theta), dtype=bool)
    residual = np.full(len(theta), np.inf)
    for _ in range(iterations):
        Y = rk4_batch(f, initial_states(surface, p, theta), length, steps)
        F = Y[:, :2] - target
        residual = np.hypot(F[:, 0], F[:, 1])
        if np.all(residual[active] < tol):
            break
        N = _normal_rows(surface, Y[:, :2], Y[:, 2:4])
        for i in np.flatnonzero(active):
            J = np.column_stack([Y[i, 4] * N[i], Y[i, 2:4]])
            try:
                d_theta, d_length = np.linalg.solve(J, -F[i])
            except np.linalg.LinAlgError:
                active[i] = False
                continue
            scale = min(1.0, 0.3 / max(abs(d_theta), 1e-300), 1.5 / max(abs(d_length), 1e-300))
            theta[i] += scale * d_theta
            length[i] += scale * d_length
            if length[i] <= 0.05 or (max_length is not None and length[i] > 1.5 * max_length):
                active[i] = False
    Y = rk4_batch(f, initial_states(surface, p, theta), length, steps)
    residual = np.hypot(*(Y[:, :2] - target).T)
    routes = []
    for i in range(len(theta)):
        if active[i] and residual[i] < 1e-8 and (max_length is None or length[i] <= max_length + 1e-9):
            routes.append({"heading": float(theta[i] % TWO_PI), "length": float(length[i]),
                           "lift": [int(v) for v in lifts[i]], "batch_residual": float(residual[i]),
                           "batch_j_head": float(Y[i, 4])})
    return dedupe(routes)


def dedupe(routes, heading_tol=1e-6, length_tol=1e-6):
    kept = []
    for route in sorted(routes, key=lambda r: (r["length"], r["heading"])):
        if not any(abs(route["length"] - k["length"]) < length_tol
                   and abs((route["heading"] - k["heading"] + math.pi) % TWO_PI - math.pi) < heading_tol
                   for k in kept):
            kept.append(route)
    return kept


def verify_route(surface, p, q, route, horizon=8.0, step=0.04):
    """Re-integrate with ciw.lab.jacobi.transfer to L + horizon; amplification and focus margin.

    The step divides L exactly so that node n1 is the endpoint. Focus margin is
    s_c - L for the first conjugate point s_c of p along the extended geodesic;
    it is censored (reported as None with ``margin_lower_bound``) when no
    conjugate point occurs within the horizon.
    """
    length = route["length"]
    n1 = max(8, int(math.ceil(length / step)))
    h = length / n1
    n_total = n1 + int(math.ceil(horizon / h))
    tr = jacobi.transfer(surface, p, route["heading"], n_total * h, steps=n_total)
    end = tr.states[n1]
    delta, _ = chart_offset(surface, end[:2], q)
    conj = tr.conjugate_points()
    first = conj[0] if conj else None
    det = tr.determinant()
    record = dict(route)
    record.update({
        "endpoint_residual": float(np.hypot(*delta)),
        "j_head": float(end[6]), "amplification": abs(float(end[6])),
        "first_conjugate": first, "conjugate_points": [float(c) for c in conj],
        "focus_margin": None if first is None else float(first - length),
        "margin_lower_bound": float(n_total * h - length),
        "wronskian_drift": float(np.max(np.abs(det - 1.0))),
        "speed_drift": float(np.max(tr.speed_drift()[: n1 + 1])),
        "transfer_steps": n_total, "transfer_step": h,
        "j_head_batch_vs_transfer": abs(float(end[6]) - route["batch_j_head"]),
    })
    if isinstance(surface, Torus):
        clairaut = [surface.clairaut(y[:2], y[2:4]) for y in tr.states[: n1 + 1]]
        record["clairaut_drift"] = float(max(clairaut) - min(clairaut))
    record["curvature_min"] = float(np.min(tr.curvature_along()[: n1 + 1]))
    record["curvature_max"] = float(np.max(tr.curvature_along()[: n1 + 1]))
    return record


def find_routes(surface, p, q, max_length, headings=360, horizon=8.0):
    candidates = fan_candidates(surface, p, q, max_length, headings=headings)
    routes = newton_routes(surface, p, q, candidates, max_length=max_length)
    return [verify_route(surface, p, q, r, horizon=horizon) for r in routes], len(candidates)


def margin_value(route):
    """Focus margin with censoring: +inf when no conjugate point occurred within the horizon."""
    return math.inf if route["focus_margin"] is None else route["focus_margin"]


# ---------------------------------------------------------------- Pareto analysis
def dominates(a, b):
    """a dominates b: no worse in (length, amplification, -margin) and better in one."""
    ka = (a["length"], a["amplification"], -margin_value(a))
    kb = (b["length"], b["amplification"], -margin_value(b))
    return all(x <= y for x, y in zip(ka, kb)) and any(x < y for x, y in zip(ka, kb))


def pareto_front(routes):
    front = [i for i, r in enumerate(routes) if not any(dominates(o, r) for j, o in enumerate(routes) if j != i)]
    return front


def rankings(routes):
    order_length = sorted(range(len(routes)), key=lambda i: (routes[i]["length"], i))
    order_amp = sorted(range(len(routes)), key=lambda i: (routes[i]["amplification"], i))
    order_margin = sorted(range(len(routes)), key=lambda i: (-margin_value(routes[i]), routes[i]["length"], i))
    return {"by_length": order_length, "by_amplification": order_amp, "by_focus_margin": order_margin}


# ---------------------------------------------------------------- independent integration
def scipy_check(surface, p, route, extra=8.0, rtol=1e-11, atol=1e-12):
    """Re-integrate one route with scipy.integrate.solve_ivp (DOP853) and locate its first conjugate point.

    The vector field is the closed-form one used by the batch search; the
    integrator and the event location are scipy's.
    """
    from scipy.integrate import solve_ivp

    f = batch_field(surface)
    y0 = initial_states(surface, p, [route["heading"]])[0]

    def rhs(_, y):
        return f(y[None, :])[0]

    def zero_head(s, y):
        return y[4]
    zero_head.direction = 0

    length = route["length"]
    sol = solve_ivp(rhs, (0.0, length + extra), y0, method="DOP853", rtol=rtol, atol=atol,
                    events=zero_head, dense_output=True)
    at_end = sol.sol(length)
    events = [float(s) for s in sol.t_events[0] if s > 1e-6]
    return {"j_head": float(at_end[4]), "endpoint": [float(at_end[0]), float(at_end[1])],
            "first_conjugate": events[0] if events else None, "status": int(sol.status)}


# ---------------------------------------------------------------- sphere (exact)
def sphere_route_pair(separation: float, radius: float = 1.0):
    """Both great-circle routes between points at angular separation < pi on a sphere (closed form)."""
    short, long = radius * separation, radius * (TWO_PI - separation)
    conj = math.pi * radius
    return [{"route": "minor arc", "length": short, "j_head": radius * math.sin(short / radius),
             "amplification": abs(radius * math.sin(short / radius)), "first_conjugate": conj,
             "focus_margin": conj - short},
            {"route": "major arc", "length": long, "j_head": radius * math.sin(long / radius),
             "amplification": abs(radius * math.sin(long / radius)), "first_conjugate": conj,
             "focus_margin": conj - long}]


def sphere_transfer_check(separation: float, steps: int = 400, extension: float = 0.5):
    """ciw.lab.jacobi on the polar-chart sphere along the equator: j_head(L) and first conjugate point.

    The step divides L so the endpoint is a node; the geodesic is extended by
    ``extension`` beyond L to locate the conjugate point.
    """
    surface = Sphere(1.0)
    h = separation / steps
    total = steps + int(math.ceil(extension / h))
    tr = jacobi.transfer(surface, [math.pi / 2, 0.0], math.pi / 2, total * h, steps=total)
    conj = tr.conjugate_points()
    return {"j_head": float(tr.states[steps, 6]), "first_conjugate": conj[0] if conj else None,
            "steps": total, "step": h}
