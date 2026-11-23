"""Exp 216 (Claude, 2026-11-18): pinned triangle n = 10 trace statement pieces shared by the emitter and the checker.
Imports no search code.

Families: the 8 support families of the n = 8/9 traces, at Q = 47 * 2^14. Each family carries pinned narrowing rows
used by contract10 (exp 215; the search engine's propagate):
  * side-gap rows (p1, c1, p2, c2, g): coord(p1,c1) + g <= coord(p2,c2) for two contacts on one side, with
    g = gap_grid(t) from the gap lemma (two points on a side at parameter distance d leave the other n - 2 = 8
    points in a homothetic copy of ratio 1 - 2t/d, so t < Delta_8 (1 - 2t/d)^2). Delta_8 is bounded above by the
    n = 8 optimum's rational upper bound t_upper, verified here by exp 202 basin8 (the n = 8 value itself is the
    released n = 8 theorem);
  * symmetry rows (canonical labelling up to the family's reflection / S3 symmetry): pinned per family exactly as in
    the search engine (cgt.make_family, sym = True); the covering lemma that these choices lose no configuration
    is a stated part of the theorem (paper, family table);
  * SUM rows (p1, c1, p2, c2): coord(p1,c1) + coord(p2,c2) <= Q.
The header pins the family name, rows, cutoff, the basin10 optimum bound and the cover radius."""
from fractions import Fraction as F
from functools import lru_cache
from hashlib import sha256
from math import isqrt
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
for d in ("152-geometry-trace", "202-tri8-trace", "214-basin10", "215-trace10-local", "245-basin-sharp",
          "246-basin-wider"):
    sys.path.insert(0, str(HERE.parent / d))
from verify_trace import require
import basin8
import basin10
import basin10w
from stream_local10 import contract10

N, GRID = 10, 47
Q = 47 * 2 ** 14
H = Q // 2
O, X, Y = ("R", 0, 0, 0, 0), ("R", Q, Q, 0, 0), ("R", 0, 0, Q, Q)
MB, ML, MH = ("R", H, H, 0, 0), ("R", 0, 0, H, H), ("R", H, H, H, H)
B1, B2 = ("R", 0, H, 0, 0), ("R", H, Q, 0, 0)
L1, L2 = ("R", 0, 0, 0, H), ("R", 0, 0, H, Q)
H1, H2 = ("H", 0, H, H, Q), ("H", H, Q, 0, H)
ROOTS = dict(MMM=(MB, ML, MH), MMS=(B1, B2, ML, MH), MSS0=(B1, B2, L1, L2, MH), MSS1=(O, B2, L2, MH),
             SSS0=(B1, B2, L1, L2, H1, H2), SSS1=(O, B2, L2, H1, H2), SSS2=(O, X, L2, H1), SSS3=(O, X, Y))
CUTOFF = "108441851695034062548779343914599525513581/5000000000000000000000000000000000000000000"  # t_lo (1 - 1e-12)
COVER_RADIUS = F(1, 200)   # 2026-11-22 statement change: exp 246 wide basin (was basin10.production_radius(), ~6.1e-6)


@lru_cache(maxsize=None)
def delta8_upper():
    points, ties, walls, t0 = basin8.load()
    return basin8.verify(points, ties, walls, t0, basin8.production_radius())["t_upper"]


def gap_grid(threshold):
    upper = delta8_upper()
    threshold = F(threshold)
    require(0 < threshold < upper, "unsupported gap threshold")
    scale = 10 ** 20
    sqrt_lower = F(isqrt(int(threshold / upper * scale ** 2)), scale)
    return int(Q * 2 * threshold / (1 - sqrt_lower))


def rows(name, threshold=CUTOFF):
    g = gap_grid(threshold)
    table = dict(
        MMM=([], []),
        MMS=([(0, 0, 1, 0, g)], [(0, 0, 1, 0)]),
        MSS0=([(0, 0, 1, 0, g), (2, 1, 3, 1, g), (0, 0, 2, 1, 0)], []),
        MSS1=([(2, 1, 1, 0, 0)], []),
        SSS0=([(0, 0, 1, 0, g), (2, 1, 3, 1, g), (4, 0, 5, 0, g), (0, 0, 2, 1, 0), (0, 0, 4, 0, 0)],
              [(0, 0, 1, 0), (0, 0, 3, 1), (0, 0, 5, 0)]),
        SSS1=([(3, 0, 4, 0, g), (2, 1, 1, 0, 0)], []),
        SSS2=([], [(3, 0, 2, 1)]),
        SSS3=([], []))
    ordr, sumr = table[name]
    return [list(r) for r in ordr], [list(r) for r in sumr]


@lru_cache(maxsize=None)
def basin_header():
    points, ties, walls, t0 = basin10.load()
    radius = COVER_RADIUS
    out = basin10w.verify_wide(points, ties, walls, t0, radius)
    require(F(CUTOFF) < out["t_lower"], "cutoff must lie below the certified optimum")
    return str(out["t_upper"]), str(radius)


def statement(name, root=None):
    optimum, radius = basin_header()
    ordr, sumr = rows(name)
    return dict(op="HEADER", version=3, n=N, Q=Q, family=name,
                root=[list(r) for r in (ROOTS[name] if root is None else root)],
                ord=ordr, sum=sumr, cutoff_raw_t=CUTOFF, optimum_raw_t=optimum, cover_radius=radius,
                basin_sha256=sha256(basin10.SOURCE.read_bytes()).hexdigest(), slice=None)


def make_contract(header):
    """support-level and local-level narrowing with the header's pinned rows (contract10 on either box size)."""
    ordr = [tuple(r) for r in header["ord"]]; sumr = [tuple(r) for r in header["sum"]]
    return lambda box: contract10(tuple(tuple(d) for d in box), ordr, sumr, header["Q"])
