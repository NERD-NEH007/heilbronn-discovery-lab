"""Independent checker for P2 (Farkas-type) certificates, written from the spec only.

Soundness argument (why acceptance => no point of the box has min area > T)
---------------------------------------------------------------------------
Setting.  A box B = prod [lo_k/D, hi_k/D] (x_0..x_{n-1}, y_0..y_{n-1}); Omega = {x sorted
ascending, sum x_i <= n/2}.  Suppose P in B ∩ Omega had every |Delta_t(P)| > T.

(1) Orientation.  2*Delta_t is multilinear in its six coordinates, so its extrema over the box
    are attained at corners: m_t <= D^2*2Delta_t(P) <= M_t.  With T2 = floor(2*T*D^2) <= 2*T*D^2:
    if m_t >= -T2 then 2Delta_t(P) >= -T2/D^2 >= -2T, and |2Delta_t| > 2T excludes
    2Delta_t < -2T, hence 2Delta_t(P) > 2T (sigma_t = +1).  If M_t <= T2 then symmetrically
    2Delta_t(P) < -2T (sigma_t = -1).  So for sigma_t != 0, sigma_t*2Delta_t(P) > 2T
    (Appendix A, Lemma "orient").  We require every area row used (mu>0) to have s = sigma_t != 0.
(2) Intended values.  Put tau = min over triples with a used area row of s_t*2Delta_t(P)
    (> 2T by (1)), X=x, Y=y, w_ab = x_a y_b.  Every row with mu > 0 holds there:
    area rows by definition of tau (tau <= s*2Delta_t, as 2Delta_t is the given linear form in w);
    McCormick rows are the expansions of the products (x_a-xl)(y_b-yl) >= 0 etc., true in the box;
    ord/mean rows hold on Omega.
(3) Aggregation.  Multiplying each row by mu >= 0 and adding gives
        s*tau + sum_v c_v v <= rhs        (v over x_i, y_i, w_ab)
    where s = coefficient of tau (= total mu of area rows).  Each variable lies in its box range
    ([lo,hi] for x,y; min/max of the four endpoint products for w_ab = x_a*y_b, valid because the
    product is bilinear on a rectangle), so  c_v v >= min(c_v*vlo, c_v*vhi), i.e.
    -c_v v <= max(-c_v*vlo, -c_v*vhi).  Hence  s*tau <= rhs + sum max(...) = N.
(4) If s > 0 and N <= 2*T*s then tau <= 2T, contradicting tau > 2T.  So no such P exists.
All arithmetic is in exact integers, scaled by D^2 (rows: tau^ = D^2 tau, X = D x, Y = D y,
W = D^2 w; McCormick coefficients become the integer endpoints, ord/mean get a factor D).
Test: N*q <= 2*p*D^2*s with T = p/q.

Usage: python check_p2.py FILE [FILE...] [--workers K]
"""
import gzip
import json
import os
import sys
import multiprocessing as mp
from fractions import Fraction

_G = {}


def _init(n, D, p, q):
    _G.update(n=n, D=D, p=p, q=q, T2=(2 * p * D * D) // q)


def _orient(lo, hi, n, t, T2):
    i, j, k = t
    xs = [(lo[i], hi[i]), (lo[j], hi[j]), (lo[k], hi[k])]
    ys = [(lo[n + i], hi[n + i]), (lo[n + j], hi[n + j]), (lo[n + k], hi[n + k])]
    m = None
    M = None
    for xi in xs[0]:
        for xj in xs[1]:
            for xk in xs[2]:
                for yi in ys[0]:
                    for yj in ys[1]:
                        for yk in ys[2]:
                            v = xj * yk - xj * yi - xi * yk - xk * yj + xk * yi + xi * yj
                            if m is None or v < m:
                                m = v
                            if M is None or v > M:
                                M = v
    if m >= -T2:
        return 1
    if M <= T2:
        return -1
    return 0


def check_cert(c, n, D, p, q, T2=None):
    """Return None if accepted, else a reason string."""
    if T2 is None:
        T2 = (2 * p * D * D) // q
    try:
        lo, hi, mu = c["lo"], c["hi"], c["mu"]
    except (KeyError, TypeError):
        return "missing lo/hi/mu"
    if not (isinstance(lo, list) and isinstance(hi, list) and isinstance(mu, list)):
        return "lo/hi/mu not lists"
    if len(lo) != 2 * n or len(hi) != 2 * n:
        return "box length != 2n"
    for v in lo + hi:
        if type(v) is not int:
            return "non-integer endpoint"
    for a, b in zip(lo, hi):
        if not (0 <= a <= b <= D):
            return "box endpoint out of range / lo>hi"
    if not mu:
        return "empty mu"
    s = 0
    cx = [0] * n
    cy = [0] * n
    cw = {}
    rhs = 0
    orient_cache = {}
    for ent in mu:
        if not (isinstance(ent, list) and len(ent) == 2):
            return "bad mu entry"
        row, m = ent
        if type(m) is not int or m <= 0:
            return "mu not a positive integer"
        if not isinstance(row, list) or not row:
            return "bad row"
        kind = row[0]
        if kind == "area":
            if len(row) != 5 or any(type(v) is not int for v in row[1:]):
                return "bad area row"
            _, i, j, k, sg = row
            if not (0 <= i < j < k < n) or sg not in (1, -1):
                return "bad area row keys"
            t = (i, j, k)
            if t not in orient_cache:
                orient_cache[t] = _orient(lo, hi, n, t, T2)
            if orient_cache[t] == 0 or orient_cache[t] != sg:
                return "area row sign %d != sigma %d for %s" % (sg, orient_cache[t], t)
            s += m
            # tau - sg*2Delta <= 0
            for (a, b), co in (((i, j), 1), ((i, k), -1), ((j, k), 1),
                               ((j, i), -1), ((k, i), 1), ((k, j), -1)):
                cw[(a, b)] = cw.get((a, b), 0) - sg * co * m
        elif kind == "mc":
            if len(row) != 4 or type(row[1]) is not int or type(row[2]) is not int:
                return "bad mc row"
            _, a, b, ty = row
            if not (0 <= a < n and 0 <= b < n and a != b):
                return "bad mc indices"
            xl, xh, yl, yh = lo[a], hi[a], lo[n + b], hi[n + b]
            if ty == "LL":
                cs, cX, cY, r = -1, yl, xl, xl * yl
            elif ty == "HH":
                cs, cX, cY, r = -1, yh, xh, xh * yh
            elif ty == "HL":
                cs, cX, cY, r = 1, -yl, -xh, -xh * yl
            elif ty == "LH":
                cs, cX, cY, r = 1, -yh, -xl, -xl * yh
            else:
                return "bad mc type"
            # row: cs*W_ab + cX*X_a + cY*Y_b <= r
            cw[(a, b)] = cw.get((a, b), 0) + cs * m
            cx[a] += cX * m
            cy[b] += cY * m
            rhs += r * m
        elif kind == "ord":
            if len(row) != 2 or type(row[1]) is not int or not (0 <= row[1] < n - 1):
                return "bad ord row"
            cx[row[1]] += D * m
            cx[row[1] + 1] -= D * m
        elif kind == "mean":
            if len(row) != 1:
                return "bad mean row"
            if (D * D * n) % 2:
                return "D^2 n odd"
            for i in range(n):
                cx[i] += D * m
            rhs += (D * D * n // 2) * m
        else:
            return "unknown row kind"
    if s <= 0:
        return "tau coefficient not positive"
    N = rhs
    for i in range(n):
        c_ = cx[i]
        N += max(-c_ * lo[i], -c_ * hi[i])
        c_ = cy[i]
        N += max(-c_ * lo[n + i], -c_ * hi[n + i])
    for (a, b), c_ in cw.items():
        prods = (lo[a] * lo[n + b], lo[a] * hi[n + b], hi[a] * lo[n + b], hi[a] * hi[n + b])
        N += max(-c_ * min(prods), -c_ * max(prods))
    if N * q <= 2 * p * D * D * s:
        return None
    return "N > 2*T*D^2*s (certificate fails)"


def _work(chunk):
    G = _G
    acc = rej = trunc = 0
    bad = []
    for lineno, raw, is_last_noeol in chunk:
        raw = raw.strip()
        if not raw:
            continue
        try:
            c = json.loads(raw)
        except ValueError:
            if is_last_noeol:
                trunc += 1
            else:
                rej += 1
                bad.append((lineno, "invalid JSON"))
            continue
        r = check_cert(c, G["n"], G["D"], G["p"], G["q"], G["T2"])
        if r is None:
            acc += 1
        else:
            rej += 1
            bad.append((lineno, r))
    return acc, rej, trunc, bad


def _chunks(f, size=200):
    buf = []
    lineno = 1
    prev = None
    for raw in f:
        lineno += 1
        if prev is not None:
            buf.append(prev)
            if len(buf) >= size:
                yield buf
                buf = []
        prev = (lineno, raw, False)
    if prev is not None:
        lineno_, raw, _ = prev
        buf.append((lineno_, raw, not raw.endswith(b"\n")))
    if buf:
        yield buf


def check_file(path, workers):
    with (gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")) as f:
        try:
            hdr = json.loads(f.readline())
            n, D = hdr["n"], hdr["D"]
            T = Fraction(hdr["T"])
            assert hdr["format"] == "p2cert-v1" and type(n) is int and type(D) is int
        except Exception as e:
            print("%s: bad header (%r)" % (path, e))
            return 0, 1
        p, q = T.numerator, T.denominator
        acc = rej = trunc = 0
        bad = []
        if workers > 1:
            with mp.Pool(workers, _init, (n, D, p, q)) as pool:
                for a, r, t, b in pool.imap(_work, _chunks(f), chunksize=1):
                    acc += a
                    rej += r
                    trunc += t
                    bad += b
        else:
            _init(n, D, p, q)
            for ch in _chunks(f):
                a, r, t, b = _work(ch)
                acc += a
                rej += r
                trunc += t
                bad += b
    print("%s: accepted %d, rejected %d, truncated-final-line skipped %d" % (path, acc, rej, trunc))
    for ln, why in bad[:5]:
        print("   line %d: %s" % (ln, why))
    return acc, rej


def main(argv):
    workers = max(1, (os.cpu_count() or 4) - 2)
    files = []
    it = iter(argv)
    for a in it:
        if a == "--workers":
            workers = int(next(it))
        else:
            files.append(a)
    if not files:
        print(__doc__)
        return 2
    anyrej = False
    for fn in files:
        a, r = check_file(fn, workers)
        anyrej |= r > 0
    return 1 if anyrej else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
