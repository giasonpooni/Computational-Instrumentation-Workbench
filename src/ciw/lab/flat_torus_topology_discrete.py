"""Grid approximations of flat-torus distance (T030).

The unit square torus is sampled on an N x N periodic grid. Graph shortest
paths with 4- or 8-neighbour stencils converge, under refinement, to the
stencil's own norm (|dx| + |dy|, or max + (sqrt 2 - 1) min), not to the
Euclidean length: the metrication error depends on direction and does not
shrink with the grid spacing. First-order fast marching solves the eikonal
equation |grad T| = 1 and does converge to Euclidean distance. Targets are kept
within half a period in each coordinate, so the torus distance equals the
planar length of the displacement.
"""
from __future__ import annotations

import heapq
import math

SQRT2 = math.sqrt(2.0)

STENCILS = {
    4: ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0)),
    8: ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
        (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2)),
}


def grid_dijkstra(n: int, neighbours: int):
    """Graph distance from node (0, 0) on the periodic n x n grid with spacing 1/n."""
    h = 1.0 / n
    dist = [[math.inf] * n for _ in range(n)]
    dist[0][0] = 0.0
    heap = [(0.0, 0, 0)]
    stencil = STENCILS[neighbours]
    while heap:
        d, i, j = heapq.heappop(heap)
        if d > dist[i][j]:
            continue
        for di, dj, w in stencil:
            a, b = (i + di) % n, (j + dj) % n
            candidate = d + w * h
            if candidate < dist[a][b]:
                dist[a][b] = candidate
                heapq.heappush(heap, (candidate, a, b))
    return dist


def stencil_norm(dx: float, dy: float, neighbours: int) -> float:
    """Limit of the grid graph distance for displacement (dx, dy) (exact for lattice displacements)."""
    ax, ay = abs(dx), abs(dy)
    if neighbours == 4:
        return ax + ay
    return max(ax, ay) + (SQRT2 - 1.0) * min(ax, ay)


def metrication_ratio(angle: float, neighbours: int) -> float:
    return stencil_norm(math.cos(angle), math.sin(angle), neighbours)


def worst_metrication(neighbours: int) -> tuple[float, float]:
    """Closed-form worst ratio and its direction: sqrt 2 at 45 deg (4-nbr), sqrt(4 - 2 sqrt 2) at 22.5 deg (8-nbr)."""
    if neighbours == 4:
        return SQRT2, math.pi / 4
    return math.sqrt(4.0 - 2.0 * SQRT2), math.atan(SQRT2 - 1.0)


def fast_marching(n: int):
    """First-order fast marching for |grad T| = 1 from node (0, 0) on the periodic n x n grid."""
    h = 1.0 / n
    far, trial, known = 0, 1, 2
    T = [[math.inf] * n for _ in range(n)]
    state = [[far] * n for _ in range(n)]
    T[0][0] = 0.0
    heap = [(0.0, 0, 0)]
    state[0][0] = trial
    while heap:
        t, i, j = heapq.heappop(heap)
        if state[i][j] == known or t > T[i][j]:
            continue
        state[i][j] = known
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = (i + di) % n, (j + dj) % n
            if state[a][b] == known:
                continue
            x = min(T[(a + 1) % n][b] if state[(a + 1) % n][b] == known else math.inf,
                    T[(a - 1) % n][b] if state[(a - 1) % n][b] == known else math.inf)
            y = min(T[a][(b + 1) % n] if state[a][(b + 1) % n] == known else math.inf,
                    T[a][(b - 1) % n] if state[a][(b - 1) % n] == known else math.inf)
            lo, hi = min(x, y), max(x, y)
            if hi - lo >= h:
                value = lo + h
            else:
                value = 0.5 * (lo + hi + math.sqrt(2 * h * h - (hi - lo) ** 2))
            if value < T[a][b]:
                T[a][b] = value
                state[a][b] = trial
                heapq.heappush(heap, (value, a, b))
    return T


def scipy_grid_distances(n: int, neighbours: int, targets):
    """Independent graph distances with scipy.sparse.csgraph.dijkstra on the same periodic grid."""
    import numpy as np
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    rows, cols, weights = [], [], []
    h = 1.0 / n
    for i in range(n):
        for j in range(n):
            for di, dj, w in STENCILS[neighbours]:
                rows.append(i * n + j)
                cols.append(((i + di) % n) * n + (j + dj) % n)
                weights.append(w * h)
    graph = coo_matrix((weights, (rows, cols)), shape=(n * n, n * n)).tocsr()
    dist = dijkstra(graph, directed=True, indices=0)
    return [float(dist[i * n + j]) for i, j in targets]


# Directions (p, q) and scale k with displacement k (p, q) / 30: all coordinates <= 0.4 < 1/2.
TARGETS = (("0 deg", (1, 0), 12), ("22.62 deg (tan 5/12)", (12, 5), 1), ("26.57 deg (tan 1/2)", (2, 1), 6),
           ("45 deg", (1, 1), 12))
REFINEMENTS = (30, 60, 120)


def refinement_study(with_scipy: bool = False):
    """Graph and fast-marching distances to fixed targets under grid refinement."""
    rows = []
    for n in REFINEMENTS:
        scale = n // 30
        graphs = {k: grid_dijkstra(n, k) for k in (4, 8)}
        marching = fast_marching(n)
        for name, (p, q), k in TARGETS:
            i, j = p * k * scale, q * k * scale
            dx, dy = i / n, j / n
            exact = math.hypot(dx, dy)
            row = {"grid": n, "direction": name, "displacement": [dx, dy], "euclidean": exact,
                   "node": [i, j]}
            for k_nb in (4, 8):
                row[f"graph{k_nb}"] = graphs[k_nb][i][j]
                row[f"ratio{k_nb}"] = graphs[k_nb][i][j] / exact
                row[f"stencil_norm{k_nb}"] = stencil_norm(dx, dy, k_nb)
            row["fmm"] = marching[i][j]
            row["fmm_rel_error"] = marching[i][j] / exact - 1.0
            rows.append(row)
        if with_scipy:
            nodes = [tuple(r["node"]) for r in rows if r["grid"] == n]
            for k_nb in (4, 8):
                values = scipy_grid_distances(n, k_nb, nodes)
                for r, value in zip([r for r in rows if r["grid"] == n], values):
                    r[f"scipy{k_nb}"] = value
    return rows
