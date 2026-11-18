"""Stream pinned triangle n=9 traces with immediate rational basin checks."""

from hashlib import sha256
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"153-exact-tri9-basin"))
from verify_basin import F, SOURCE, load, verify as verify_basin, verify_cover
from verify_trace import det_range, graph_boxes, keys, p2_bound, read_record, require, split, triple_at, vertices


def read_root(root, q, n):
    require(type(root) is list and 0<len(root)<=n,"invalid root")
    domains = []
    for row in root:
        require(type(row) is list and len(row)==5,"invalid point domain")
        kind,xl,xh,yl,yh = row
        require(kind in ("R","H") and all(type(v) is int for v in row[1:]),"invalid point domain")
        require(0<=xl<=xh<=q and 0<=yl<=yh<=q,"bounds outside grid")
        require(kind!="H" or (yl,yh)==(q-xh,q-xl),"invalid hypotenuse parameter")
        require(vertices(row,q),"empty root domain")
        domains.append(tuple(row))
    return tuple(domains)


def verify(lines, expected, on_pending=None, canonical_contract=None):
    stream = iter(lines)
    header = read_record(next(stream,""))
    keys(header,"op version n Q cutoff_raw_t root optimum_raw_t basin_sha256 cover_radius")
    require(header==expected,"trace statement mismatch")
    require(header["op"]=="HEADER" and type(header["version"]) is int and header["version"]==2
            and type(header["n"]) is int and header["n"]==9,"invalid cover header")
    require(sha256(SOURCE.read_bytes()).hexdigest()==header["basin_sha256"],"basin artifact mismatch")
    q,n = header["Q"],header["n"]
    require(type(q) is int and q>0,"invalid grid")
    require(all(type(header[k]) is str for k in ("cutoff_raw_t","optimum_raw_t","cover_radius")),
            "rational header fields must be strings")
    points,optimum,_ = load()
    radius,threshold = F(header["cover_radius"]),F(header["cutoff_raw_t"])
    require(F(header["optimum_raw_t"])==optimum and 0<threshold<optimum,"invalid thresholds")
    verify_basin(points,optimum,radius)
    determinant_cutoff = int(2*threshold*q*q)
    stack = [iter((read_root(header["root"],q,n),))]
    ordinary, covers, pending = 0,0,0

    def next_box():
        while stack:
            box = next(stack[-1],None)
            if box is not None:
                return box
            stack.pop()
        return None

    for line in stream:
        record = read_record(line)
        box = next_box()
        require(box is not None,"extra record after coverage")
        op = record["op"]
        if op=="SPLIT":
            keys(record,"op point axis cut")
            stack.append(iter(split(box,record["point"],record["axis"],record["cut"],q)))
        elif op=="CANONICAL_CONTRACT":
            keys(record,"op closed")
            require(canonical_contract is not None and len(box)==n,"unexpected canonical contraction")
            narrowed = canonical_contract(box)
            require(record["closed"] is (narrowed is None),"invalid canonical contraction")
            if narrowed is None:
                ordinary += 1
            else:
                stack.append(iter((narrowed,)))
        elif op=="GRAPH_ENTER":
            keys(record,"op M conflicts")
            require(len(box)<n,"graph inside a local box")
            stack.append(graph_boxes(box,q,record["M"],n,threshold,record["conflicts"]))
        elif op=="CLOSE_BOUND":
            keys(record,"op triple")
            lo,hi = det_range(box,triple_at(record["triple"],box),q)
            require(-determinant_cutoff<=lo<=hi<=determinant_cutoff,"direct bound failed")
            ordinary += 1
        elif op=="CLOSE_P2":
            keys(record,"op weights")
            require(p2_bound(box,record["weights"],q,threshold)<=threshold,"P2 inequality failed")
            ordinary += 1
        elif op=="CLOSE_COVER":
            keys(record,"op barycentric_permutation point_assignment")
            permutation,assignment = record["barycentric_permutation"],record["point_assignment"]
            require(all(type(a) is list and all(type(i) is int for i in a)
                        for a in (permutation,assignment)),"invalid cover map")
            verify_cover(box,q,points,radius,permutation,assignment)
            covers += 1
        elif op=="FRONTIER_YIELD":
            keys(record,"op")
            pending += 1
            if on_pending is not None:
                on_pending(box)
        else:
            require(False,"unsupported opcode")
    require(next_box() is None,"incomplete trace")
    upper = (optimum if covers else threshold) if not pending else None
    return dict(complete=not pending,pending=pending,ordinary_closures=ordinary,
                cover_closures=covers,root_upper_bound=str(upper) if upper is not None else None,
                equality_requires_certified_orbit=bool(covers and not pending))


if __name__=="__main__":
    expected = read_record(Path(sys.argv[2]).read_text(encoding="utf-8"))
    with Path(sys.argv[1]).open(encoding="utf-8") as trace:
        print(json.dumps(verify(trace,expected)))
