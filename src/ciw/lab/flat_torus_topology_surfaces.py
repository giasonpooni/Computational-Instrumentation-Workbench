"""Polygonal translation surfaces: gluing, vertex classes, cone angles and straight-line flow (T027-T029).

A surface is a list of convex polygons (counterclockwise vertex lists) and an
involution pairing their edges. An edge a->b glued to a'->b' identifies b with
a' and a with b'. The gluing is a translation when the edge vectors are
opposite; otherwise (for example the pillowcase) it is recorded as a
half-translation or a general gluing, and straight-line flow is refused.

Coordinates are exact where the geometry allows: integers and
:class:`fractions.Fraction` for square-tiled surfaces, and :class:`Surd`
(a + b sqrt(d) with rational a, b) for the regular octagon and hexagon. A
float variant of the octagon exists only to test a declared tolerance against
the exact trajectory. Corner angles are recognized as rational multiples of
pi from their exact edge vectors with a recorded float residual.
"""
from __future__ import annotations

from fractions import Fraction
import math


class GluingRefusal(ValueError):
    """An edge pairing that does not define the declared kind of surface."""


class FlowTermination(Exception):
    """A trajectory reached a vertex: a saddle connection or an undecidable near-vertex pass."""

    def __init__(self, code: str, message: str, record: dict):
        super().__init__(f"{code}: {message}")
        self.code, self.record = code, record


class Surd:
    """Exact a + b sqrt(d) for square-free d > 1 with rational a, b."""

    __slots__ = ("a", "b", "d")

    def __init__(self, a, b=0, d=2):
        self.a, self.b, self.d = Fraction(a), Fraction(b), int(d)

    def _lift(self, other):
        if isinstance(other, Surd):
            if other.d != self.d and other.b and self.b:
                raise ValueError("Surd arithmetic requires the same radicand")
            return other if other.d == self.d else Surd(other.a, other.b, self.d if not other.b else other.d)
        return Surd(other, 0, self.d)

    def __add__(self, other):
        o = self._lift(other)
        return Surd(self.a + o.a, self.b + o.b, self.d)

    __radd__ = __add__

    def __neg__(self):
        return Surd(-self.a, -self.b, self.d)

    def __sub__(self, other):
        return self + (-self._lift(other))

    def __rsub__(self, other):
        return self._lift(other) - self

    def __mul__(self, other):
        o = self._lift(other)
        return Surd(self.a * o.a + self.d * self.b * o.b, self.a * o.b + self.b * o.a, self.d)

    __rmul__ = __mul__

    def __truediv__(self, other):
        o = self._lift(other)
        norm = o.a * o.a - o.d * o.b * o.b
        if norm == 0:
            raise ZeroDivisionError("Surd division by zero")
        return self * Surd(o.a / norm, -o.b / norm, self.d)

    def __rtruediv__(self, other):
        return self._lift(other) / self

    def sign(self) -> int:
        """Exact sign: compare a^2 with d b^2 when a and b have opposite signs."""
        sa = (self.a > 0) - (self.a < 0)
        sb = (self.b > 0) - (self.b < 0)
        if sa == sb or sb == 0:
            return sa
        if sa == 0:
            return sb
        return sa if self.a * self.a > self.d * self.b * self.b else sb

    def __eq__(self, other):
        return (self - other).sign() == 0

    def __hash__(self):
        return hash((self.a, self.b, self.d))

    def __lt__(self, other):
        return (self - other).sign() < 0

    def __le__(self, other):
        return (self - other).sign() <= 0

    def __gt__(self, other):
        return (self - other).sign() > 0

    def __ge__(self, other):
        return (self - other).sign() >= 0

    def __float__(self):
        return float(self.a) + float(self.b) * math.sqrt(self.d)

    def __repr__(self):
        return f"{self.a}+{self.b}*sqrt({self.d})"

    def as_json(self):
        return {"rational": str(self.a), "sqrt_coefficient": str(self.b), "radicand": self.d}


def _sub(p, q):
    return (p[0] - q[0], p[1] - q[1])


def _add(p, q):
    return (p[0] + q[0], p[1] + q[1])


def _cross(p, q):
    return p[0] * q[1] - p[1] * q[0]


def _dot(p, q):
    return p[0] * q[0] + p[1] * q[1]


def _is_zero(x, tol):
    return abs(float(x)) <= tol if tol else x == 0


class PolygonSurface:
    """Polygons with an edge pairing; ``tol`` > 0 only for the declared float variant."""

    def __init__(self, name, polygons, pairs, tol: float = 0.0):
        self.name, self.polygons, self.tol = name, [list(p) for p in polygons], tol
        self.partner = {}
        for left, right in pairs:
            for edge in (left, right):
                if edge in self.partner:
                    raise GluingRefusal(f"Edge {edge} is glued twice")
            self.partner[left], self.partner[right] = right, left
        edges = {(i, j) for i, p in enumerate(self.polygons) for j in range(len(p))}
        if set(self.partner) != edges:
            raise GluingRefusal("Every polygon edge must be glued to exactly one other edge")
        for i, polygon in enumerate(self.polygons):
            if len(polygon) < 3:
                raise GluingRefusal("Polygons need at least three vertices")
            for j in range(len(polygon)):
                if not self._positive(_cross(self.edge(i, j), self.edge(i, (j + 1) % len(polygon)))):
                    raise GluingRefusal(f"Polygon {i} is not strictly convex and counterclockwise at vertex {j + 1}")
        for (i, j), (k, l) in self.partner.items():
            if abs(float(_dot(self.edge(i, j), self.edge(i, j))) - float(_dot(self.edge(k, l), self.edge(k, l)))) > 1e-12 \
                    and not self._equal(_dot(self.edge(i, j), self.edge(i, j)), _dot(self.edge(k, l), self.edge(k, l))):
                raise GluingRefusal(f"Glued edges {(i, j)} and {(k, l)} have different lengths")

    def _positive(self, x):
        return float(x) > self.tol if self.tol else x > 0

    def _equal(self, x, y):
        return abs(float(x) - float(y)) <= self.tol if self.tol else x == y

    def edge(self, i, j):
        polygon = self.polygons[i]
        return _sub(polygon[(j + 1) % len(polygon)], polygon[j])

    def gluing_kind(self) -> str:
        """translation (e' = -e for every pair), half-translation (e' = +e allowed) or general."""
        kinds = set()
        for (i, j), (k, l) in self.partner.items():
            e, f = self.edge(i, j), self.edge(k, l)
            if self._equal(e[0], -f[0]) and self._equal(e[1], -f[1]):
                kinds.add("translation")
            elif self._equal(e[0], f[0]) and self._equal(e[1], f[1]):
                kinds.add("half-translation")
            else:
                kinds.add("general")
        if kinds == {"translation"}:
            return "translation"
        return "half-translation" if "general" not in kinds else "general"

    def require_translation(self):
        kind = self.gluing_kind()
        if kind != "translation":
            raise GluingRefusal(f"{self.name}: gluing is {kind}, not a translation surface")

    # Topology ---------------------------------------------------------
    def vertex_classes(self):
        """Union-find over polygon corners: b ~ a' and a ~ b' for each glued pair a->b, a'->b'."""
        parent = {}

        def find(x):
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            parent[find(x)] = find(y)

        for i, polygon in enumerate(self.polygons):
            for j in range(len(polygon)):
                find((i, j))
        for (i, j), (k, l) in self.partner.items():
            n, m = len(self.polygons[i]), len(self.polygons[k])
            union((i, (j + 1) % n), (k, l))
            union((i, j), (k, (l + 1) % m))
        classes = {}
        for corner in sorted(parent):
            classes.setdefault(find(corner), []).append(corner)
        return sorted(classes.values())

    def corner_angle(self, i, j):
        """Interior angle at vertex j of polygon i as an exact multiple of pi (denominator 24)."""
        polygon = self.polygons[i]
        incoming, outgoing = self.edge(i, (j - 1) % len(polygon)), self.edge(i, j)
        turn = math.atan2(float(_cross(incoming, outgoing)), float(_dot(incoming, outgoing)))
        ratio = (math.pi - turn) / math.pi
        rational = Fraction(round(ratio * 24), 24)
        return rational, abs(ratio - float(rational))

    def topology(self):
        classes = self.vertex_classes()
        cones, residual = [], 0.0
        for members in classes:
            total = Fraction(0)
            for i, j in members:
                angle, error = self.corner_angle(i, j)
                total += angle
                residual = max(residual, error)
            cones.append({"corners": [list(c) for c in members], "cone_angle_over_pi": total})
        vertices, faces = len(classes), len(self.polygons)
        edges = sum(len(p) for p in self.polygons) // 2
        chi = vertices - edges + faces
        curvature = sum(2 - c["cone_angle_over_pi"] for c in cones)  # sum (2 pi - theta) / pi
        record = {"vertices": vertices, "edges": edges, "faces": faces, "euler_characteristic": chi,
                  "cones": cones, "gauss_bonnet_defect_over_pi": curvature - 2 * chi,
                  "angle_recognition_residual": residual, "gluing": self.gluing_kind()}
        if record["gluing"] == "translation":
            record["genus"] = (2 - chi) // 2
            record["cone_orders"] = sorted(int(c["cone_angle_over_pi"] / 2) - 1 for c in cones)
        return record

    def area(self):
        total = 0
        for polygon in self.polygons:
            n = len(polygon)
            total = total + sum(_cross(polygon[k], polygon[(k + 1) % n]) for k in range(n)) / 2
        return total

    # Straight-line flow -----------------------------------------------
    def cone_points(self):
        """Corners whose vertex class has cone angle other than 2 pi."""
        singular = set()
        for cone in self.topology()["cones"]:
            if cone["cone_angle_over_pi"] != 2:
                singular.update(tuple(c) for c in cone["corners"])
        return singular

    def flow(self, polygon: int, start, direction, max_crossings: int = 200, stop_on_return: bool = True):
        """Straight-line flow from an interior point; exact unless the surface declares ``tol``.

        Returns the crossing record and total time (in units of ``direction``);
        raises FlowTermination with SADDLE_CONNECTION at an exact cone-point hit,
        or NEAR_VERTEX_WITHIN_TOLERANCE when a float pass is within ``tol``.
        """
        self.require_translation()
        singular = self.cone_points()
        x, current, entry = start, polygon, None
        time, crossings, min_clearance = 0, [], math.inf
        for _ in range(max_crossings):
            best = None
            poly = self.polygons[current]
            for j in range(len(poly)):
                if j == entry:
                    continue
                e = self.edge(current, j)
                denom = _cross(direction, e)
                if _is_zero(denom, 0.0):
                    continue
                rel = _sub(poly[j], x)
                t = _cross(rel, e) / denom
                sigma = _cross(rel, direction) / denom
                if not self._positive(t):
                    continue
                if float(sigma) < -1e-9 or float(sigma) > 1 + 1e-9:
                    continue
                if best is None or t < best[0]:
                    best = (t, j, sigma)
            if best is None:
                raise FlowTermination("NO_EXIT", "no exit edge found (point outside polygon)", {"polygon": current})
            t, j, sigma = best
            # Returning to the start inside this polygon closes the trajectory.
            if stop_on_return and current == polygon and crossings:
                rel = _sub(start, x)
                if _is_zero(_cross(rel, direction), self.tol) and self._positive(_dot(rel, direction)) \
                        and float(_dot(rel, direction)) <= float(t * _dot(direction, direction)) + self.tol:
                    closing = _dot(rel, direction) / _dot(direction, direction)
                    return {"closed": True, "time": time + closing, "crossings": crossings,
                            "min_vertex_clearance": min_clearance}
            edge_length = math.sqrt(float(_dot(self.edge(current, j), self.edge(current, j))))
            clearance = min(float(sigma), 1 - float(sigma)) * edge_length
            min_clearance = min(min_clearance, clearance)
            record = {"polygon": current, "edge": j, "time": time + t, "crossings": len(crossings)}
            at_vertex = (sigma == 0 or sigma == 1) if not self.tol else clearance <= self.tol
            if at_vertex:
                corner = (current, j if (float(sigma) < 0.5) else (j + 1) % len(poly))
                if not self.tol:
                    code = "SADDLE_CONNECTION" if corner in singular else "REGULAR_VERTEX_UNRESOLVED"
                    raise FlowTermination(code, "trajectory hits a polygon vertex", dict(record, corner=list(corner)))
                raise FlowTermination("NEAR_VERTEX_WITHIN_TOLERANCE",
                                      "trajectory passes within the declared tolerance of a vertex",
                                      dict(record, corner=list(corner), clearance=clearance))
            k, l = self.partner[(current, j)]
            partner_poly = self.polygons[k]
            translation = _sub(partner_poly[l], poly[(j + 1) % len(poly)])
            y = _add(_add(x, (t * direction[0], t * direction[1])), translation)
            crossings.append((current, j, k, l))
            time = time + t
            x, current, entry = y, k, l
        return {"closed": False, "time": time, "crossings": crossings, "min_vertex_clearance": min_clearance,
                "end": (current, x)}


# ---------------------------------------------------------------- builders
def unit_square(origin=(0, 0)):
    x, y = origin
    return [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]


def square_tiled(name, r, u, origins=None):
    """Origami from permutations r (right neighbour) and u (upper neighbour) of the squares.

    Square s: edge 0 bottom, 1 right, 2 top, 3 left. Right of s glues to left
    of r[s]; top of s glues to bottom of u[s].
    """
    n = len(r)
    if sorted(r) != list(range(n)) or sorted(u) != list(range(n)):
        raise GluingRefusal("r and u must be permutations of the squares")
    origins = origins or [(k, 0) for k in range(n)]
    polygons = [unit_square(o) for o in origins]
    pairs = [((s, 1), (r[s], 3)) for s in range(n)] + [((s, 2), (u[s], 0)) for s in range(n)]
    surface = PolygonSurface(name, polygons, pairs)
    surface.permutations = {"r": list(r), "u": list(u)}
    return surface


def l_shape():
    """Three squares A=(0,0), B=(1,0), C=(0,1): r = (A B)(C), u = (A C)(B); genus 2, one 6 pi cone point."""
    return square_tiled("square-tiled L (3 squares)", [1, 0, 2], [2, 1, 0], origins=[(0, 0), (1, 0), (0, 1)])


def square_torus():
    return square_tiled("square torus (1 square)", [0], [0])


def commutator_cycles(r, u):
    """Cycle lengths of r u r^-1 u^-1 acting on squares: one cone point of angle 2 pi c per c-cycle."""
    n = len(r)
    r_inv, u_inv = [0] * n, [0] * n
    for k in range(n):
        r_inv[r[k]], u_inv[u[k]] = k, k
    # Walk around a lower-left corner: go right, up, left, down.
    comm = [u_inv[r_inv[u[r[k]]]] for k in range(n)]
    seen, cycles = set(), []
    for k in range(n):
        if k in seen:
            continue
        length, j = 0, k
        while j not in seen:
            seen.add(j)
            j = comm[j]
            length += 1
        cycles.append(length)
    return sorted(cycles)


def regular_octagon(exact: bool = True, tol: float = 1e-9):
    """Regular octagon of side 1 with opposite sides glued; exact in Q(sqrt 2) or float with ``tol``."""
    s = Surd(0, Fraction(1, 2), 2) if exact else math.sqrt(2) / 2
    one = Surd(1, 0, 2) if exact else 1.0
    zero = Surd(0, 0, 2) if exact else 0.0
    vertices = [(zero, zero), (one, zero), (one + s, s), (one + s, one + s), (one, one + 2 * s),
                (zero, one + 2 * s), (zero - s, one + s), (zero - s, s)]
    pairs = [((0, j), (0, j + 4)) for j in range(4)]
    return PolygonSurface("regular octagon (opposite sides glued)" + ("" if exact else " [float]"),
                          [vertices], pairs, tol=0.0 if exact else tol)


def regular_hexagon():
    """Regular hexagon of side 1 with opposite sides glued (a flat torus), exact in Q(sqrt 3)."""
    h = Surd(0, Fraction(1, 2), 3)
    one, half, zero = Surd(1, 0, 3), Surd(Fraction(1, 2), 0, 3), Surd(0, 0, 3)
    vertices = [(zero, zero), (one, zero), (one + half, h), (one, 2 * h), (zero, 2 * h), (zero - half, h)]
    pairs = [((0, j), (0, j + 3)) for j in range(3)]
    return PolygonSurface("regular hexagon (opposite sides glued)", [vertices], pairs)


def pillowcase():
    """Two unit squares glued along their boundary (the double of a square): a sphere with four cone points of angle pi.

    The second square is the mirror copy redrawn counterclockwise, so bottom
    and top edges are glued by half-turns and left/right edges by translations.
    """
    pairs = [((0, 0), (1, 0)), ((0, 1), (1, 3)), ((0, 2), (1, 2)), ((0, 3), (1, 1))]
    return PolygonSurface("pillowcase (2 squares, half-translation)", [unit_square(), unit_square((2, 0))], pairs)


def octagon_adjacent_pairing():
    """Refused example: adjacent octagon sides paired, which is not a translation gluing."""
    surface = regular_octagon()
    return PolygonSurface("octagon with adjacent sides paired", surface.polygons,
                          [((0, 0), (0, 1)), ((0, 2), (0, 3)), ((0, 4), (0, 5)), ((0, 6), (0, 7))])
