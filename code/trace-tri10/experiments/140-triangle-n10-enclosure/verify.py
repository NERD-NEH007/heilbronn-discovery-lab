"""Verify minimum-enclosure uniqueness over the Exp 138 root enclosure."""

from dataclasses import dataclass
from fractions import Fraction as F
from itertools import combinations
import json
from pathlib import Path


@dataclass(frozen=True)
class Interval:
    lo: F
    hi: F

    def __add__(self, other):
        return Interval(self.lo + other.lo, self.hi + other.hi)

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -other

    def __mul__(self, other):
        products = [a*b for a in (self.lo, self.hi) for b in (other.lo, other.hi)]
        return Interval(min(products), max(products))

    def __truediv__(self, other):
        assert other.lo * other.hi > 0
        return self * Interval(1 / other.hi, 1 / other.lo)


def exact(x):
    return Interval(F(x), F(x))


ZERO, HALF, ONE, TWO = map(exact, (0, F(1, 2), 1, 2))


def sub(a, b):
    return tuple(x-y for x, y in zip(a, b))


def cross(a, b):
    return a[0]*b[1] - a[1]*b[0]


def orient(a, b, c):
    return cross(sub(b, a), sub(c, a))


def intersection(a, b, c, d):
    u, v = sub(b, a), sub(d, c)
    t = cross(sub(c, a), v) / cross(u, v)
    return tuple(x + t*y for x, y in zip(a, u))


def verify(rho=None):
    """rho: sup radius of the root ball about points_exact. Default: the Exp 138 radius. The release checker passes
    the radius it has just re-verified with basin10, so nothing here rests on the unreplayed Exp 138 claim."""
    source = Path(__file__).resolve().parent.parent / "138-triangle-n10-local/local-certificate.json"
    cert = json.loads(source.read_text())["10"][0]
    assert cert["certified"]
    extra = cert["extra"]
    rho = F(extra["box_rho_exact"]) if rho is None else F(rho)
    assert rho > 0
    assert {tuple(w) for w in extra["exact_wall_contacts"]} == {
        (0, "L"), (2, "H"), (3, "B"), (3, "H"), (6, "B"), (7, "L")
    }
    p = [tuple(Interval(F(x)-rho, F(x)+rho) for x in row) for row in cert["points_exact"]]
    for i in (0, 7):
        p[i] = ZERO, p[i][1]
    p[3] = ONE, ZERO
    p[6] = p[6][0], ZERO
    p[2] = p[2][0], ONE-p[2][0]
    order = (7, 9, 6, 3, 2, 0)
    edges = list(zip(order, order[1:] + order[:1]))
    walls = {(6, 3), (3, 2), (0, 7)}
    for i, j in edges:
        assert all(orient(p[i], p[j], q).lo > 0
                   for k, q in enumerate(p) if k not in (i, j))

    # If all midpoints are inside hull edges, three edge lines determine the enclosure.
    triples_checked = 0
    for lines in combinations(edges, 3):
        if set(lines) == walls:
            continue
        a, b, c = (intersection(p[i], p[j], p[k], p[l])
                   for (i, j), (k, l) in combinations(lines, 2))
        signed = orient(a, b, c)
        assert signed.lo > 0 or signed.hi < 0
        if signed.hi < 0:
            b, c = c, b
            signed = -signed
        if signed.lo > 1:
            triples_checked += 1
            continue
        sides = ((a, b), (b, c), (c, a))
        exterior = any(orient(u, v, p[i]).hi < 0 for u, v in sides for i in order)
        midpoint_outside = any(
            orient(p[i], p[j], tuple((x+y)*HALF for x, y in zip(u, v))).hi < 0
            for i, j in edges for u, v in sides
        )
        assert exterior or midpoint_outside, lines
        triples_checked += 1

    # Otherwise the common midpoint height is a hull vertex height.
    heights_checked = 0
    ties = {tuple(t) for t in extra["exact_ties"]}
    for i, j in edges:
        a = p[i]
        d = sub(p[j], a)
        length = sum((x*x for x in d), ZERO)
        frame = {}
        for k in order:
            q = sub(p[k], a)
            frame[k] = (sum((x*y for x, y in zip(q, d)), ZERO)/length, cross(d, q))
        frame[i], frame[j] = (ZERO, ZERO), (ONE, ZERO)
        for k in order:
            if k in (i, j):
                continue
            xk, h = frame[k]
            assert h.lo > 0
            differences = {l: frame[l][1]-h for l in order if l != k}
            tied_heights = [l for l, s in differences.items() if s.lo <= 0 <= s.hi]
            assert len(tied_heights) <= 1
            if tied_heights:
                l, = tied_heights
                # On the positive side of the base, equal tied areas give equal heights.
                assert frame[l][1].lo > 0
                assert all(tuple(sorted((i, j, m))) in ties for m in (k, l))
                other = frame[l][0]
            else:
                other = None
            if all(s.hi < 0 for s in differences.values()):
                heights_checked += 1  # The top section is one point.
                continue
            if other is None:
                (u, v), = [(u, v) for u, v in edges if k not in (u, v)
                           and (differences[u].hi < 0) != (differences[v].hi < 0)]
                xu, yu = frame[u]
                xv, yv = frame[v]
                other = xu+(xv-xu)*(h-yu)/(yv-yu)
            left, right = (other, xk) if other.hi < xk.lo else (xk, other)
            assert left.hi < right.lo
            if (TWO*h*(right-left)).lo > F(1, 2):
                heights_checked += 1
                continue
            z = Interval(F(0), F(1))
            for l in order:
                x, y = frame[l]
                s = y/(TWO*h)
                if s.lo > 1:
                    break
                coefficient = ONE-TWO*s
                lower = x-right+left-TWO*s*left
                upper = x-left+right-TWO*s*right
                lhs = coefficient*z
                if lhs.hi < lower.lo or lhs.lo > upper.hi:
                    break
                if coefficient.lo * coefficient.hi > 0:
                    a_bound, b_bound = lower/coefficient, upper/coefficient
                    if coefficient.hi < 0:
                        a_bound, b_bound = b_bound, a_bound
                    z = Interval(max(z.lo, a_bound.lo), min(z.hi, b_bound.hi))
                    if z.lo > z.hi:
                        break
            else:
                raise AssertionError(((i, j), k, float(z.lo), float(z.hi)))
            heights_checked += 1
    return {"hull": order, "competing_edge_triples_excluded": triples_checked,
            "vertex_height_cases_excluded": heights_checked, "unique_enclosure_area": "1/2"}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
