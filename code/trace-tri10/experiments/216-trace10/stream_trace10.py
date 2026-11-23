"""Exp 216: stream pinned triangle n = 10 trace shards (trust base). Copy of exp 202 stream_trace8 with:
n = 10 and header version 3 (family name and its pinned narrowing rows), the exact n = 10 basin (exp 214 basin10,
second-order lemma, KKT existence) for covers, CANONICAL_CONTRACT through contract10 with the header's rows
(exp 215), and the CLOSE_MCC op (exp 213). Graph replay is the exp 160 exact reconstruction (no supplied
conflicts: cell-P2 is off in traced runs). The search engine is never imported."""
from fractions import Fraction as F
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
for d in ("152-geometry-trace", "160-tri9-sss3-production-trace", "213-close-mcc", "214-basin10", "215-trace10-local",
          "245-basin-sharp", "246-basin-wider"):
    sys.path.insert(0, str(HERE.parent / d))
sys.path.insert(0, str(HERE))
from verify_trace import det_range, keys, p2_bound, read_record, require, split, triple_at, vertices
from graph_replay10 import graph_boxes, graph_cells
import basin10
import basin10w                  # exp 246 wide basin lemma (2026-11-22 statement change)
import mcc_check
from stream_local10 import contract10

HEADER_KEYS = "op version n Q family root ord sum cutoff_raw_t optimum_raw_t basin_sha256 cover_radius slice"
FAMILIES = ("MMM", "MMS", "MSS0", "MSS1", "SSS0", "SSS1", "SSS2", "SSS3")


def read_root(root, q, n):
    require(type(root) is list and 0 < len(root) <= n, "invalid root")
    domains = []
    for row in root:
        require(type(row) is list and len(row) == 5, "invalid point domain")
        kind, xl, xh, yl, yh = row
        require(kind in ("R", "H") and all(type(v) is int for v in row[1:]), "invalid point domain")
        require(0 <= xl <= xh <= q and 0 <= yl <= yh <= q, "bounds outside grid")
        require(kind != "H" or (yl, yh) == (q - xh, q - xl), "invalid hypotenuse parameter")
        require(vertices(row, q), "empty root domain")
        domains.append(tuple(row))
    return tuple(domains)


def read_rows(rows, width, k):
    require(type(rows) is list, "invalid rows")
    out = []
    for r in rows:
        require(type(r) is list and len(r) == width and all(type(v) is int for v in r), "invalid row")
        require(0 <= r[0] < k and 0 <= r[2] < k and r[1] in (0, 1) and r[3] in (0, 1), "row outside the support")
        require(width == 4 or r[4] >= 0, "negative gap")
        out.append(tuple(r))
    return out


@lru_cache(maxsize=4)
def verified_basin(radius_text, source_digest):
    require(sha256(basin10.SOURCE.read_bytes()).hexdigest() == source_digest, "basin artifact mismatch")
    points, ties, walls, t0 = basin10.load()
    out = basin10w.verify_wide(points, ties, walls, t0, F(radius_text))   # exp 246 (was basin10.verify)
    return points, out["t_lower"], out["t_upper"]


def verify(lines, expected, on_pending=None, pairs=frozenset(), pairs_key=""):
    """pairs: cell-pair exclusions verified by the manifest checker at an ancestor support box (exp 223)."""
    stream = iter(lines)
    header = read_record(next(stream, ""))
    keys(header, HEADER_KEYS)
    require(header == expected, "trace statement mismatch")
    require(header["op"] == "HEADER" and type(header["version"]) is int and header["version"] == 3
            and type(header["n"]) is int and header["n"] == 10 and header["family"] in FAMILIES, "invalid header")
    q, n = header["Q"], header["n"]
    require(type(q) is int and q > 0, "invalid grid")
    require(all(type(header[k]) is str for k in ("cutoff_raw_t", "optimum_raw_t", "cover_radius")),
            "rational header fields must be strings")
    points, t_lower, t_upper = verified_basin(header["cover_radius"], header["basin_sha256"])
    radius, threshold = F(header["cover_radius"]), F(header["cutoff_raw_t"])
    optimum = t_upper
    require(F(header["optimum_raw_t"]) == optimum and 0 < threshold < t_lower, "invalid thresholds")
    root = read_root(header["root"], q, n)
    k = min(len(root), n)
    ordr = read_rows(header["ord"], 5, 10); sumr = read_rows(header["sum"], 4, 10)
    contract = lambda box: contract10(box, ordr, sumr, q)
    sl = header["slice"]
    require(sl is None or (type(sl) is list and len(sl) == 3 and all(type(v) is int for v in sl)), "invalid slice")
    entered = False
    determinant_cutoff = int(2 * threshold * q * q)
    stack = [iter((root,))]
    ordinary, covers, pending = 0, 0, 0

    def next_box():
        while stack:
            box = next(stack[-1], None)
            if box is not None:
                return box
            stack.pop()
        return None

    for line in stream:
        record = read_record(line)
        box = next_box()
        require(box is not None, "extra record after coverage")
        op = record["op"]
        if op == "SPLIT":
            keys(record, "op point axis cut")
            stack.append(iter(split(box, record["point"], record["axis"], record["cut"], q)))
        elif op == "CANONICAL_CONTRACT":
            keys(record, "op closed")
            require(len(box) == n, "unexpected canonical contraction")
            narrowed = contract(box)
            require(record["closed"] is (narrowed is None), "invalid canonical contraction")
            require(narrowed != box, "vacuous contraction record")
            if narrowed is None:
                ordinary += 1
            else:
                stack.append(iter((narrowed,)))
        elif op == "GRAPH_ENTER":
            keys(record, "op M conflicts")
            require(len(box) < n, "graph inside a local box")
            if sl is not None:      # a clique-range slice: only directly at the shard root, once
                require(not entered and box == root and record["M"] == sl[0], "invalid sliced graph entry")
                entered = True
                stack.append(graph_boxes(box, q, record["M"], n, threshold, record["conflicts"], (sl[1], sl[2]),
                                          pairs, pairs_key))
            else:
                stack.append(graph_boxes(box, q, record["M"], n, threshold, record["conflicts"], None, pairs, pairs_key))
        elif op == "CLOSE_BOUND":
            keys(record, "op triple")
            lo, hi = det_range(box, triple_at(record["triple"], box), q)
            require(-determinant_cutoff <= lo <= hi <= determinant_cutoff, "direct bound failed")
            ordinary += 1
        elif op == "CLOSE_P2":
            keys(record, "op weights")
            require(p2_bound(box, record["weights"], q, threshold) <= threshold, "P2 inequality failed")
            ordinary += 1
        elif op == "CLOSE_MCC":
            mcc_check.check_mcc(box, record, q, threshold)
            ordinary += 1
        elif op == "CLOSE_COVER":
            keys(record, "op barycentric_permutation point_assignment")
            permutation, assignment = record["barycentric_permutation"], record["point_assignment"]
            require(all(type(a) is list and all(type(i) is int for i in a)
                        for a in (permutation, assignment)), "invalid cover map")
            basin10.verify_cover(box, q, points, radius, permutation, assignment)
            covers += 1
        elif op == "FRONTIER_YIELD":
            keys(record, "op")
            pending += 1
            if on_pending is not None:
                on_pending(box)
        else:
            require(False, "unsupported opcode")
    require(next_box() is None, "incomplete trace")
    require(sl is None or entered, "sliced shard without its graph entry")
    upper = (optimum if covers else threshold) if not pending else None
    return dict(complete=not pending, pending=pending, ordinary_closures=ordinary,
                cover_closures=covers, root_upper_bound=str(upper) if upper is not None else None,
                equality_requires_certified_orbit=bool(covers and not pending))


if __name__ == "__main__":
    expected = read_record(Path(sys.argv[2]).read_text(encoding="utf-8"))
    with Path(sys.argv[1]).open(encoding="utf-8") as trace:
        print(json.dumps(verify(trace, expected)))
