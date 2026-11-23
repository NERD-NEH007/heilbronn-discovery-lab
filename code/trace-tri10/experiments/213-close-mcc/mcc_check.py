"""Exp 213: CLOSE_MCC record checker (trust base: standard library plus exp 152 verify_trace helpers).

Record: {"op": "CLOSE_MCC", "weights": [[triple, lam], ...], "mu": [[u, v, a, b, m], ...]}
  triple : index into combinations(range(len(box)), 3), strictly increasing; lam: int > 0
  u < v  : variable ids 2p + c (c = 0: x of point p; c = 1: y of point p, only for kind "R" points;
           the y of a hypotenuse point "H" is q - x, so it uses the variable 2p); a, b in {-1, +1}; m: int > 0;
           entries strictly increasing as tuples (u, v, a, b).

Lemma. Write each coordinate as (C + W s)/2 with s in [-1, 1] (x: C = xl + xh, W = xh - xl; y of an R point:
C = yl + yh, W = yh - yl; y of an H point: C = 2q - xl - xh, W = xl - xh, same s as its x). For every weighted
triple the sign sigma is determined for every configuration that could violate the bound (lo >= -r or hi <= r,
r = floor(2 t q^2), exactly as for CLOSE_P2: values of the other sign then have |D| <= r, so that triangle is already
small; audit 2026-11-18), and 4 sigma D = K0 + sum_v A_v s_v + sum_{u<v} P_uv s_u s_v exactly. For each mu entry
g = (1 + a s_u)(1 + b s_v) >= 0 on the box. Let
    F(s) = sum lam (4 sigma D - R4) + sum m g = c + sum r_v s_v + sum r_uv s_u s_v,   R4 = 8 t q^2.
Since |s_v| <= 1 and |s_u s_v| <= 1, F(s) <= c + sum |r_v| + sum |r_uv| =: bound. If bound <= 0 then at every point
of the box sum lam (4 sigma D - R4) <= -sum m g <= 0, so some weighted triple has sigma D <= 2 t q^2, i.e. area
<= t: no configuration in the box has all areas > t. (With mu empty this is the shared-product P2 bound.)"""
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "152-geometry-trace"))
from verify_trace import InvalidTrace, det_range, keys, require, triple_at

TERMS = ((0, 1, 1), (0, 2, -1), (1, 0, -1), (1, 2, 1), (2, 0, 1), (2, 1, -1))   # sign * x_{t[i]} * y_{t[j]}


def variable(box, p, axis, q):
    kind, xl, xh, yl, yh = box[p]
    if axis == 0:
        return xl + xh, xh - xl, 2 * p
    if kind == "H":
        return 2 * q - xl - xh, xl - xh, 2 * p
    return yl + yh, yh - yl, 2 * p + 1


def is_int(v):
    return type(v) is int


def mcc_bound(box, record, q, threshold):
    """Exact value of `bound` above (a Fraction); raises on any malformed or inapplicable record."""
    keys(record, "op weights mu")
    require(record["op"] == "CLOSE_MCC", "not a CLOSE_MCC record")
    weights, mu = record["weights"], record["mu"]
    require(type(weights) is list and weights and type(mu) is list, "missing weights")
    threshold = Fraction(threshold)
    require(threshold > 0, "invalid threshold")
    r = int(2 * threshold * q * q)
    const, total = 0, 0
    lin, bil = defaultdict(int), defaultdict(int)
    prior = -1
    for pair in weights:
        require(type(pair) is list and len(pair) == 2 and all(map(is_int, pair)), "invalid weight record")
        index, lam = pair
        require(index > prior and lam > 0, "invalid weight")
        prior, total = index, total + lam
        triple = triple_at(index, box)
        lo, hi = det_range(box, triple, q)
        require(lo >= -r or hi <= r, "ambiguous triangle sign")
        sigma = 1 if lo >= -r else -1
        for i, j, sign in TERMS:
            cx, wx, u = variable(box, triple[i], 0, q)
            cy, wy, v = variable(box, triple[j], 1, q)
            s = sigma * sign * lam
            const += s * cx * cy
            lin[u] += s * wx * cy
            lin[v] += s * cx * wy
            if wx * wy:
                require(u != v, "shared variable inside one product")
                bil[(min(u, v), max(u, v))] += s * wx * wy
    nvar = 2 * len(box); prior = None
    for entry in mu:
        require(type(entry) is list and len(entry) == 5 and all(map(is_int, entry)), "invalid mu record")
        u, v, a, b, m = entry
        require(0 <= u < v < nvar and a in (-1, 1) and b in (-1, 1) and m > 0, "invalid mu entry")
        require(prior is None or (u, v, a, b) > prior, "mu entries must be strictly increasing")
        prior = (u, v, a, b)
        for w in (u, v):
            require(w % 2 == 0 or box[w // 2][0] == "R", "no y variable for a hypotenuse point")
        const += m; lin[u] += m * a; lin[v] += m * b; bil[(u, v)] += m * a * b
    R4 = 8 * threshold * q * q
    return const - total * R4 + sum(map(abs, lin.values())) + sum(map(abs, bil.values()))


def check_mcc(box, record, q, threshold):
    require(mcc_bound(box, record, q, threshold) <= 0, "McCormick inequality failed")
