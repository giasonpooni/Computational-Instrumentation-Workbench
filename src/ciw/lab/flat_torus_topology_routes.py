"""Candidate geodesic routes between two points on curved surfaces (T024, T025, T032).

Routes p -> q are found by shooting: a fan of headings is integrated in one
vectorized RK4 batch with closed-form Christoffel symbols, the miss-distance
minima of near-passes of every chart lift of q seed a Newton iteration on
(heading, length) whose Jacobian is [velocity, j_head * normal] (the heading
Jacobi field), and every converged
route is then re-integrated with :func:`ciw.lab.jacobi.transfer`, which
supplies the reported amplification |j_head(L)| and the conjugate points of
the extended geodesic. The batch integrator is only a search device; reported
numbers come from ``ciw.lab.jacobi`` and are cross-checked against it. The
optional scipy check integrates a field that sympy derives from the embedding
itself (metric, Christoffel symbols, Gauss curvature and heading frame), so it
shares no hand-written geometry with the batch field or ``ciw.lab.surfaces``.

The candidate set is whatever the fan search found up to a declared length;
:func:`fan_convergence` reruns it at twice the density, but completeness is
not proven. Nothing here models a vehicle, a tool or a workspace, and no route
is claimed safe to execute.
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
    """Near-passes of any lift of q along a fan of geodesics from p: (heading, s, lift, miss distance).

    A near-pass is a local minimum in s of the metric distance to the nearest
    lift of q, below ``threshold`` and beyond s = 0.2. Ordered by fan ray, then s.
    """
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
    inner = dist[1:-1]
    mask = (inner <= dist[:-2]) & (inner < dist[2:]) & (inner < threshold) & (s[1:-1, None] > 0.2)
    rays, nodes = np.nonzero(mask.T)
    return [(float(thetas[j]), float(s[i + 1]), tuple(int(v) for v in lift[i + 1, j]), float(dist[i + 1, j]))
            for j, i in zip(rays, nodes)]


def newton_seeds(candidates, headings, s_gap=0.5):
    """Keep one seed per miss-distance minimum along each run of adjacent fan rays.

    Adjacent rays passing the same lift of q at similar arclength form a run;
    a route is a zero of the miss distance inside the run, so each local
    minimum over the run (ends included) seeds Newton once. Two routes closer
    than one fan step in heading would share a seed; the double-density
    convergence rerun is the guard against that.
    """
    step = TWO_PI / headings
    by_key = {}
    for c in candidates:
        by_key.setdefault(c[2], []).append(c)
    seeds = []
    for lift in sorted(by_key):
        rows = sorted(by_key[lift], key=lambda c: (c[0], c[1]))
        runs, current = [], []
        for c in rows:
            if current and not (abs(c[0] - current[-1][0] - step) < 0.5 * step and abs(c[1] - current[-1][1]) < s_gap):
                runs.append(current)
                current = []
            current.append(c)
        if current:
            runs.append(current)
        # Rays wrap at 2 pi: join the last run to the first when they are adjacent.
        if len(runs) > 1 and abs(runs[0][0][0] + TWO_PI - runs[-1][-1][0] - step) < 0.5 * step \
                and abs(runs[0][0][1] - runs[-1][-1][1]) < s_gap:
            runs[0] = runs.pop() + runs[0]
        for run in runs:
            d = [c[3] for c in run]
            for k, c in enumerate(run):
                left = d[k - 1] if k > 0 else math.inf
                right = d[k + 1] if k + 1 < len(run) else math.inf
                if d[k] <= left and d[k] < right:
                    seeds.append(c[:3])
    return seeds


def _normal_rows(surface, U, V):
    return np.array([surface.normal(u, v) for u, v in zip(U, V)])


def newton_routes(surface, p, q, candidates, steps=600, iterations=12, tol=1e-10, max_length=None):
    """Refine (heading, length) for every candidate so the geodesic ends on its lift of q.

    Returns (routes, counts): counts records why candidates were dropped
    (singular Newton matrix, length leaving (0.05, 1.5 max_length), final
    residual above 1e-8, converged beyond the length budget) and how many
    converged duplicates were merged.
    """
    counts = {"candidates": len(candidates), "singular": 0, "diverged": 0, "residual_fail": 0,
              "over_length": 0, "duplicates_merged": 0, "routes": 0}
    if not candidates:
        return [], counts
    f = batch_field(surface)
    theta = np.array([c[0] for c in candidates])
    length = np.array([c[1] for c in candidates])
    lifts = np.array([c[2] for c in candidates], dtype=float)
    target = np.asarray(q, dtype=float)[None, :] + TWO_PI * lifts
    active = np.ones(len(theta), dtype=bool)
    reason = [None] * len(theta)
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
                active[i], reason[i] = False, "singular"
                continue
            scale = min(1.0, 0.3 / max(abs(d_theta), 1e-300), 1.5 / max(abs(d_length), 1e-300))
            theta[i] += scale * d_theta
            length[i] += scale * d_length
            if length[i] <= 0.05 or (max_length is not None and length[i] > 1.5 * max_length):
                active[i], reason[i] = False, "diverged"
    Y = rk4_batch(f, initial_states(surface, p, theta), length, steps)
    residual = np.hypot(*(Y[:, :2] - target).T)
    routes = []
    for i in range(len(theta)):
        if not active[i]:
            counts[reason[i]] += 1
        elif not residual[i] < 1e-8:
            counts["residual_fail"] += 1
        elif max_length is not None and length[i] > max_length + 1e-9:
            counts["over_length"] += 1
        else:
            routes.append({"heading": canonical_heading(float(theta[i])), "length": float(length[i]),
                           "lift": [int(v) for v in lifts[i]], "batch_residual": float(residual[i]),
                           "batch_j_head": float(Y[i, 4])})
    kept = dedupe(routes)
    counts["duplicates_merged"], counts["routes"] = len(routes) - len(kept), len(kept)
    return kept, counts


def canonical_heading(theta: float) -> float:
    """Heading in [-pi/2, 3 pi/2): symmetric routes (headings 0 and pi) never sit on the wrap point."""
    return (theta + 0.5 * math.pi) % TWO_PI - 0.5 * math.pi


def dedupe(routes, heading_tol=1e-6, length_tol=1e-6, tie_tol=1e-9):
    """Merge duplicates; order by length, and by heading within groups of lengths tied to ``tie_tol``.

    Mirror-image routes have equal lengths up to rounding, so their order must
    not depend on the last bit of the length.
    """
    kept = []
    for route in sorted(routes, key=lambda r: (r["length"], r["heading"])):
        if not any(abs(route["length"] - k["length"]) < length_tol
                   and abs((route["heading"] - k["heading"] + math.pi) % TWO_PI - math.pi) < heading_tol
                   for k in kept):
            kept.append(route)
    ordered, group = [], []
    for route in kept:
        if group and route["length"] - group[-1]["length"] > tie_tol:
            ordered.extend(sorted(group, key=lambda r: r["heading"]))
            group = []
        group.append(route)
    return ordered + sorted(group, key=lambda r: r["heading"])


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
        # Targeting q by heading alone needs d(heading) = d(normal miss) / j_head: small
        # |j_head| near a conjugate point means ill-conditioned targeting, not robustness.
        "targeting_condition": 1.0 / max(abs(float(end[6])), 1e-300),
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


def search(surface, p, q, max_length, headings):
    """Fan candidates, one Newton seed per miss-distance minimum, Newton refinement; (routes, counts)."""
    candidates = fan_candidates(surface, p, q, max_length, headings=headings)
    seeds = newton_seeds(candidates, headings)
    routes, counts = newton_routes(surface, p, q, seeds, max_length=max_length)
    counts = {"fan_candidates": len(candidates), "newton_seeds": counts.pop("candidates"), **counts}
    return routes, counts


def find_routes(surface, p, q, max_length, headings=1440, horizon=8.0):
    """Fan search, Newton refinement and ciw.lab.jacobi verification; returns (routes, drop counts)."""
    routes, counts = search(surface, p, q, max_length, headings)
    return [verify_route(surface, p, q, r, horizon=horizon) for r in routes], counts


def same_route_set(a, b, length_tol=1e-6, heading_tol=1e-6):
    """Routes of ``a`` missing from ``b`` and of ``b`` missing from ``a`` (matched by length and heading)."""
    def missing(xs, ys):
        return [x for x in xs if not any(abs(x["length"] - y["length"]) < length_tol and
                                         abs((x["heading"] - y["heading"] + math.pi) % TWO_PI - math.pi) < heading_tol
                                         for y in ys)]
    return missing(a, b), missing(b, a)


def fan_convergence(surface, p, q, max_length, routes, headings):
    """Self-convergence of the route set: rerun the search at twice the fan density and compare."""
    denser, counts = search(surface, p, q, max_length, 2 * headings)
    lost, gained = same_route_set(routes, denser)
    return {"headings": [headings, 2 * headings], "routes": [len(routes), len(denser)],
            "missing_at_base_density": [{"length": r["length"], "heading": r["heading"]} for r in gained],
            "missing_at_double_density": [{"length": r["length"], "heading": r["heading"]} for r in lost],
            "differences": len(lost) + len(gained), "double_density_counts": counts}


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


def objective_matrix(routes):
    """Rows (length, amplification, -focus margin) read from the route fields; a censored margin gives -inf."""
    return np.array([[r["length"], r["amplification"], -math.inf if r["focus_margin"] is None else -r["focus_margin"]]
                     for r in routes], dtype=float).reshape(len(routes), 3)


def front_by_dominance_matrix(objectives):
    """Non-dominated rows (minimization) from a broadcast dominance matrix.

    A second formulation of :func:`pareto_front` that shares none of its code:
    D[i, j] is True when row i is no worse than row j in every column and
    better in one; the front is the set of columns of D with no True entry.
    """
    F = np.asarray(objectives, dtype=float)
    no_worse = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    better = np.any(F[:, None, :] < F[None, :, :], axis=2)
    return [int(i) for i in np.flatnonzero(~np.any(no_worse & better, axis=0))]


def front_by_sweep(keys):
    """Non-dominated points of two-objective keys (minimization) by sort-and-sweep.

    After a lexicographic sort, a point is dominated exactly when some strictly
    smaller key has a second objective no larger than its own, so the sweep
    keeps the running minimum of the second objective over strictly smaller
    keys (equal keys are processed as one group and never dominate each other).
    """
    order = sorted(range(len(keys)), key=lambda i: (keys[i][0], keys[i][1], i))
    front, best, k = [], math.inf, 0
    while k < len(order):
        group = [i for i in order[k:] if tuple(keys[i]) == tuple(keys[order[k]])]
        if keys[order[k]][1] < best:
            front.extend(group)
        best = min(best, keys[order[k]][1])
        k += len(group)
    return sorted(front)


def rankings(routes, tie_tol=1e-9):
    """Orders by length and amplification, and focus-margin groups (best first).

    Censored margins (no conjugate point within the horizon) are only known to
    exceed the horizon, so those routes form one tied group; finite margins
    within ``tie_tol`` are also grouped. Indices inside a group are ascending.
    """
    order_length = sorted(range(len(routes)), key=lambda i: (routes[i]["length"], i))
    order_amp = sorted(range(len(routes)), key=lambda i: (routes[i]["amplification"], i))
    censored = [i for i, r in enumerate(routes) if r["focus_margin"] is None]
    finite = sorted((i for i, r in enumerate(routes) if r["focus_margin"] is not None),
                    key=lambda i: (-routes[i]["focus_margin"], i))
    groups = [censored] if censored else []
    for i in finite:
        if groups and groups[-1] and groups[-1] is not censored and \
                abs(routes[groups[-1][-1]]["focus_margin"] - routes[i]["focus_margin"]) <= tie_tol:
            groups[-1].append(i)
        else:
            groups.append([i])
    return {"by_length": order_length, "by_amplification": order_amp, "by_focus_margin": groups}


# ---------------------------------------------------------------- independent integration
_SYMPY_FIELDS: dict = {}


def sympy_field(surface):
    """Cached :func:`derive_sympy_field` per surface description (the derivation takes about 0.3 s)."""
    key = repr(sorted(surface.describe().items()))
    if key not in _SYMPY_FIELDS:
        _SYMPY_FIELDS[key] = derive_sympy_field(surface)
    return _SYMPY_FIELDS[key]


def derive_sympy_field(surface):
    """Geodesic + heading-Jacobi field derived by sympy from the embedding alone.

    X(u, v) is written symbolically; sympy differentiates it for the metric
    g = J^T J, the Christoffel symbols of g and the Gauss curvature
    K = (L N - M^2) / det g from the unnormalized normal X_u x X_v. The heading
    frame is Gram-Schmidt in g (e1 along the first coordinate, e2 its positive
    unit normal). Returns (rhs(y), initial_state(u0, heading)) on plain floats.
    """
    import sympy

    u, v = sympy.symbols("u v", real=True)
    if isinstance(surface, Torus):
        big, small = sympy.Rational(surface.major), sympy.Rational(surface.minor)
        rho = big + small * sympy.cos(v)
        X = sympy.Matrix([rho * sympy.cos(u), rho * sympy.sin(u), small * sympy.sin(v)])
    elif isinstance(surface, GaussianBump):
        h, sigma = sympy.Rational(surface.h), sympy.Rational(surface.sigma)
        X = sympy.Matrix([u, v, h * sympy.exp(-(u ** 2 + v ** 2) / (2 * sigma ** 2))])
    elif isinstance(surface, Saddle):
        c = sympy.Rational(surface.c)
        X = sympy.Matrix([u, v, c * (u ** 2 - v ** 2) / 2])
    else:
        raise ValueError(f"No symbolic embedding for surface {surface.name}")
    coords = (u, v)
    J = X.jacobian(coords)
    g = J.T * J
    det = g[0, 0] * g[1, 1] - g[0, 1] ** 2
    ginv = sympy.Matrix([[g[1, 1], -g[0, 1]], [-g[0, 1], g[0, 0]]]) / det
    gamma = [[[sum(ginv[k, l] * (sympy.diff(g[j, l], coords[i]) + sympy.diff(g[i, l], coords[j])
                                 - sympy.diff(g[i, j], coords[l])) for l in range(2)) / 2
               for j in range(2)] for i in range(2)] for k in range(2)]
    normal = J[:, 0].cross(J[:, 1])
    second = [X.diff(a).diff(b).dot(normal) for a, b in ((u, u), (u, v), (v, v))]
    curvature = (second[0] * second[2] - second[1] ** 2) / det ** 2
    exprs = [gamma[k][i][j] for k in range(2) for i in range(2) for j in range(2)] + [curvature]
    fields = sympy.lambdify(coords, exprs, modules="math", cse=True)
    metric = sympy.lambdify(coords, [g[0, 0], g[0, 1], g[1, 1]], modules="math")

    def rhs(_, y):
        values = fields(y[0], y[1])
        vel = (y[2], y[3])
        acc = [-sum(values[4 * k + 2 * i + j] * vel[i] * vel[j] for i in range(2) for j in range(2))
               for k in range(2)]
        return [y[2], y[3], acc[0], acc[1], y[5], -values[8] * y[4]]

    def initial(u0, heading):
        g11, g12, g22 = metric(float(u0[0]), float(u0[1]))
        e1 = (1.0 / math.sqrt(g11), 0.0)
        scale = math.sqrt(g11 / (g11 * g22 - g12 * g12))
        e2 = (-g12 / g11 * scale, scale)
        t = (math.cos(heading) * e1[0] + math.sin(heading) * e2[0], math.cos(heading) * e1[1] + math.sin(heading) * e2[1])
        return [float(u0[0]), float(u0[1]), t[0], t[1], 0.0, 1.0]
    return rhs, initial


def scipy_check(surface, p, q, route, extra=8.0, rtol=1e-11, atol=1e-12, field=None):
    """Re-integrate one route with scipy.integrate.solve_ivp (DOP853) on the sympy-derived field.

    Returns j_head(L), the endpoint's distance to the nearest chart lift of q,
    the first conjugate point within L + extra, and the solver status (0 when
    the integration reached L + extra).
    """
    from scipy.integrate import solve_ivp

    rhs, initial = field or sympy_field(surface)

    def zero_head(s, y):
        return y[4]
    zero_head.direction = 0

    length = route["length"]
    sol = solve_ivp(rhs, (0.0, length + extra), initial(p, route["heading"]), method="DOP853", rtol=rtol,
                    atol=atol, events=zero_head, dense_output=True)
    if sol.status != 0:
        # No values from an integration that stopped early: the caller counts it as a failed run.
        return {"j_head": None, "endpoint_residual": None, "first_conjugate": None, "status": int(sol.status)}
    at_end = sol.sol(length)
    delta, _ = chart_offset(surface, at_end[:2], q)
    events = [float(s) for s in sol.t_events[0] if s > 1e-6]
    return {"j_head": float(at_end[4]), "endpoint_residual": float(np.hypot(*delta)),
            "first_conjugate": events[0] if events else None, "status": int(sol.status)}


def compare_with_scipy(surface, p, q, route_list, extra=8.0):
    """scipy/sympy re-integration of routes; errors only over matched quantities, mismatches counted.

    ``j_head`` is the largest |j_head difference| / max(1, |j_head|), ``conjugate``
    the largest conjugate-point difference over routes where both codes find
    one, ``presence_mismatches`` the routes where only one code finds a
    conjugate point, and ``failed`` the solver runs that did not reach L + extra.
    """
    field = sympy_field(surface)
    worst_j = worst_c = worst_end = 0.0
    presence, failed, rows = 0, 0, []
    for route in route_list:
        other = scipy_check(surface, p, q, route, extra=extra, field=field)
        rows.append(other)
        if other["status"] != 0:
            failed += 1
            continue
        worst_j = max(worst_j, abs(other["j_head"] - route["j_head"]) / max(1.0, abs(route["j_head"])))
        worst_end = max(worst_end, other["endpoint_residual"])
        if (other["first_conjugate"] is None) != (route["first_conjugate"] is None):
            presence += 1
        elif route["first_conjugate"] is not None:
            worst_c = max(worst_c, abs(other["first_conjugate"] - route["first_conjugate"]))
    return {"j_head": worst_j, "conjugate": worst_c, "endpoint": worst_end, "presence_mismatches": presence,
            "failed": failed, "rows": rows}


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
