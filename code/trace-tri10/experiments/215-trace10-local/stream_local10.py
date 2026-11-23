"""Exp 215 (trust base): verify a triangle n = 10 LOCAL transcript (one clique's box tree).

Standard library plus exp 152 verify_trace helpers, exp 213 mcc_check and exp 214 basin10. The engine is never
imported. A transcript is a depth-first list of records over a root box (10 point domains, kinds "R"/"H"):
  SPLIT {point, axis, cut}          -> the two halves (verify_trace.split; empty halves are dropped)
  CANONICAL_CONTRACT {closed}       -> contract10(box): None closes the box, else the narrowed box replaces it
  CLOSE_BOUND {triple}              -> one triangle's exact determinant range lies inside [-r, r]
  CLOSE_P2 {weights}                -> verify_trace.p2_bound <= t
  CLOSE_MCC {weights, mu}           -> mcc_check (coupled McCormick lemma)
  CLOSE_COVER {barycentric_permutation, point_assignment} -> basin10.verify_cover at the verified cover radius
Each closure proves: no configuration in the box (satisfying the family's ordering rows) has every triangle area
> t, except configurations inside the certified basin of the optimum (CLOSE_COVER)."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
for d in ("152-geometry-trace", "213-close-mcc", "214-basin10"):
    sys.path.insert(0, str(HERE.parent / d))
from verify_trace import det_range, keys, p2_bound, read_record, require, split, triple_at, vertices
import mcc_check
import basin10


def contract10(box, ord_rows, sum_rows, q):
    """Family narrowing, identical to the search engine's propagate: ORD rows (p1, c1, p2, c2, g) mean
    coord(p1, c1) + g <= coord(p2, c2); SUM rows (p1, c1, p2, c2) mean coord(p1, c1) + coord(p2, c2) <= q; every R
    point has x + y <= q; an H point has y = q - x. Each step removes only points violating one of these, so the
    result contains every admissible configuration of the box. Returns the narrowed box or None (empty)."""
    R = [list(d[1:]) for d in box]; H = [d[0] == "H" for d in box]
    changed, it = True, 0
    while changed and it < 50:
        changed, it = False, it + 1
        for p1, c1, p2, c2, g in ord_rows:
            h = R[p2][2 * c2 + 1] - g
            if h < R[p1][2 * c1 + 1]: R[p1][2 * c1 + 1] = h; changed = True
            l = R[p1][2 * c1] + g
            if l > R[p2][2 * c2]: R[p2][2 * c2] = l; changed = True
        for p1, c1, p2, c2 in sum_rows:
            h = q - R[p2][2 * c2]
            if h < R[p1][2 * c1 + 1]: R[p1][2 * c1 + 1] = h; changed = True
            h = q - R[p1][2 * c1]
            if h < R[p2][2 * c2 + 1]: R[p2][2 * c2 + 1] = h; changed = True
        for p in range(len(R)):
            if H[p]:
                R[p][2], R[p][3] = q - R[p][1], q - R[p][0]
            else:
                h = q - R[p][2]
                if h < R[p][1]: R[p][1] = h; changed = True
                h = q - R[p][0]
                if h < R[p][3]: R[p][3] = h; changed = True
            if R[p][0] > R[p][1] or R[p][2] > R[p][3]:
                return None
    if any(r[0] > r[1] or r[2] > r[3] for r in R):
        return None
    return tuple((("H" if H[p] else "R"),) + tuple(R[p]) for p in range(len(R)))


def verify_local(root, lines, q, threshold, ord_rows, sum_rows, points=None, cover_radius=None):
    """Returns closure counts; raises InvalidTrace unless the transcript covers the whole root box.
    CLOSE_COVER needs points/cover_radius already verified by basin10.verify (done once by the caller)."""
    require(len(root) == 10 and all(len(d) == 5 and d[0] in ("R", "H") for d in root), "invalid root box")
    r = int(2 * threshold * q * q)
    stack = [iter((tuple(root),))]
    counts = dict(split=0, contract=0, bound=0, p2=0, mcc=0, cover=0)

    def next_box():
        while stack:
            b = next(stack[-1], None)
            if b is not None:
                return b
            stack.pop()
        return None

    for line in lines:
        rec = read_record(line) if isinstance(line, str) else line
        box = next_box()
        require(box is not None, "extra record after coverage")
        op = rec.get("op")
        if op == "SPLIT":
            keys(rec, "op point axis cut")
            stack.append(iter(split(box, rec["point"], rec["axis"], rec["cut"], q))); counts["split"] += 1
        elif op == "CANONICAL_CONTRACT":
            keys(rec, "op closed")
            narrowed = contract10(box, ord_rows, sum_rows, q)
            require(rec["closed"] is (narrowed is None), "invalid canonical contraction")
            require(narrowed != box, "vacuous contraction record")
            counts["contract"] += 1
            if narrowed is not None:
                stack.append(iter((narrowed,)))
        elif op == "CLOSE_BOUND":
            keys(rec, "op triple")
            lo, hi = det_range(box, triple_at(rec["triple"], box), q)
            require(-r <= lo <= hi <= r, "direct bound failed"); counts["bound"] += 1
        elif op == "CLOSE_P2":
            keys(rec, "op weights")
            require(p2_bound(box, rec["weights"], q, threshold) <= threshold, "P2 inequality failed"); counts["p2"] += 1
        elif op == "CLOSE_MCC":
            mcc_check.check_mcc(box, rec, q, threshold); counts["mcc"] += 1
        elif op == "CLOSE_COVER":
            keys(rec, "op barycentric_permutation point_assignment")
            require(points is not None and cover_radius is not None, "cover without a verified basin")
            perm, assign = rec["barycentric_permutation"], rec["point_assignment"]
            require(all(type(a) is list and all(type(i) is int for i in a) for a in (perm, assign)), "invalid cover map")
            basin10.verify_cover(box, q, points, cover_radius, perm, assign); counts["cover"] += 1
        else:
            require(False, "unsupported opcode")
    require(next_box() is None, "incomplete transcript")
    return counts
