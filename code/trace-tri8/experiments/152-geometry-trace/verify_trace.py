"""Standalone reference checker for bounded triangle traces, using the standard library."""

from collections import defaultdict
from fractions import Fraction
from itertools import combinations, product
import json
from pathlib import Path
import sys


class InvalidTrace(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InvalidTrace(message)


def keys(record, fields):
    require(type(record) is dict and record.keys()==set(fields.split()), "unexpected record fields")


def unique_object(pairs):
    value = dict(pairs)
    require(len(value)==len(pairs), "duplicate JSON key")
    return value


def vertices(domain, q):
    kind, xl, xh, yl, yh = domain
    if kind == "H":
        return sorted({(x,q-x) for x in (xl,xh)})
    out = {(x,y) for x,y in product((xl,xh),(yl,yh)) if x+y<=q}
    out.update((x,q-x) for x in (xl,xh) if yl<=q-x<=yh)
    out.update((q-y,y) for y in (yl,yh) if xl<=q-y<=xh)
    return sorted(out)


def determinant(a,b,c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def det_range(box, triple, q):
    values = [determinant(*p) for p in product(*(vertices(box[i],q) for i in triple))]
    return min(values),max(values)


def triple_at(index, box):
    triples = list(combinations(range(len(box)),3))
    require(type(index) is int and 0<=index<len(triples),"invalid triple ID")
    return triples[index]


def p2_bound(box, weights, q, threshold):
    require(type(weights) is list and weights,"missing weights")
    combined = defaultdict(int)
    total, constant = 0,0
    prior = -1
    for pair in weights:
        require(type(pair) is list and len(pair)==2,"invalid weight record")
        index, weight = pair
        triple = triple_at(index,box)
        require(index>prior and type(weight) is int and weight>0,"invalid weight")
        prior,total = index,total+weight
        lo,hi = det_range(box,triple,q)
        r = int(2*threshold*q*q)
        require(lo>=-r or hi<=r,"ambiguous triangle sign")
        orientation = 1 if lo>=-r else -1
        for i,j,sign in ((0,1,1),(0,2,-1),(1,0,-1),(1,2,1),(2,0,1),(2,1,-1)):
            a,b = triple[i],triple[j]
            _,xl,xh,_,_ = box[a]
            kind,bl,bh,yl,yh = box[b]
            cx,wx = xl+xh,xh-xl
            cy,wy = yl+yh,yh-yl
            y_var = (b,1)
            if kind=="H":
                cy,wy,y_var = 2*q-bl-bh,bl-bh,(b,0)
            scale = orientation*sign*weight
            constant += scale*cx*cy + weight*abs(wx*wy)
            combined[(a,0)] += scale*wx*cy
            combined[y_var] += scale*cx*wy
    return Fraction(constant+sum(map(abs,combined.values())),8*q*q*total)


def split(box, point, axis, cut, q):
    require(type(point) is int and 0<=point<len(box) and type(axis) is int and axis in (0,1),
            "invalid split variable")
    require(type(cut) is int,"invalid cut")
    kind,xl,xh,yl,yh = box[point]
    lo,hi = ((xl,xh) if axis==0 else (yl,yh))
    require(lo<cut<hi and (kind!="H" or axis==0),"invalid cut")
    halves = []
    for lower,upper in ((lo,cut),(cut,hi)):
        child = list(box)
        if kind=="H":
            child[point] = (kind,lower,upper,q-upper,q-lower)
        elif axis==0:
            child[point] = (kind,lower,upper,yl,yh)
        else:
            child[point] = (kind,xl,xh,lower,upper)
        if vertices(child[point],q):
            halves.append(tuple(child))
    return halves


def graph_boxes(support, q, m, n, threshold, heavy):
    require(type(m) is int and 1<=m<=q and q%m==0 and Fraction(2,m)<=2*threshold,
            "cell occupancy condition failed")
    r = int(2*threshold*q*q)
    width = q//m
    cells = {x*m+y:("R",x*width,(x+1)*width,y*width,(y+1)*width)
             for x in range(m) for y in range(m-x)}
    require(type(heavy) is list,"invalid graph witnesses")
    excluded = set()
    for record in heavy:
        keys(record,"cells weights")
        ids = record["cells"]
        require(type(ids) is list and 1<=len(ids)<=min(2,n-len(support))
                and all(type(i) is int and i in cells for i in ids)
                and ids==sorted(set(ids)),"invalid conflict cells")
        require(p2_bound(support+tuple(cells[i] for i in ids),record["weights"],q,threshold)<=threshold,
                "graph P2 inequality failed")
        excluded.add(tuple(ids))

    def direct(box, triple):
        lo,hi = det_range(box,triple,q)
        return -r<=lo<=hi<=r

    k = len(support)
    remaining = [i for i,cell in cells.items() if (i,) not in excluded and not any(
        direct(support+(cell,),(a,b,k)) for a,b in combinations(range(k),2))]
    neighbors = defaultdict(list)
    for i,j in combinations(remaining,2):
        if (i,j) not in excluded and not any(
                direct(support+(cells[i],cells[j]),(a,k,k+1)) for a in range(k)):
            neighbors[i].append(j)
    neighbors = {i:set(js) for i,js in neighbors.items()}

    def cliques(selected, candidates):
        if len(selected)==n-k:
            yield support+tuple(cells[i] for i in selected)
            return
        for i in candidates:
            if any(direct((cells[a],cells[b],cells[i]),(0,1,2))
                   for a,b in combinations(selected,2)):
                continue
            yield from cliques(selected+(i,),sorted(neighbors.get(i,set()).intersection(candidates)))

    return cliques((),remaining)


def read_record(line):
    try:
        value = json.loads(line,object_pairs_hook=unique_object)
    except (ValueError, TypeError) as exc:
        raise InvalidTrace("invalid JSON record") from exc
    require(type(value) is dict and type(value.get("op")) is str,"missing opcode")
    return value


def verify(lines, expected_header=None):
    stream = iter(lines)
    header = read_record(next(stream,""))
    keys(header,"op version n Q cutoff_raw_t root")
    require(expected_header is None or header==expected_header,"trace statement mismatch")
    require(header["op"]=="HEADER" and type(header["version"]) is int and header["version"]==1,
            "unsupported trace header")
    n,q = header["n"],header["Q"]
    require(type(n) is int and n>=3 and type(q) is int and q>0,"invalid dimensions")
    require(type(header["cutoff_raw_t"]) is str,"cutoff must be an exact string")
    try:
        t = Fraction(header["cutoff_raw_t"])
    except (ValueError,ZeroDivisionError) as exc:
        raise InvalidTrace("invalid cutoff") from exc
    require(t>0,"invalid cutoff")
    require(type(header["root"]) is list and 0<len(header["root"])<=n,"invalid root")
    root = []
    for domain in header["root"]:
        require(type(domain) is list and len(domain)==5,"invalid point domain")
        kind,xl,xh,yl,yh = domain
        require(kind in ("R","H") and all(type(v) is int for v in domain[1:]),"invalid point domain")
        require(0<=xl<=xh<=q and 0<=yl<=yh<=q,"point bounds outside grid")
        require(kind!="H" or (yl,yh)==(q-xh,q-xl),"invalid hypotenuse parameter")
        require(vertices(domain,q),"empty root point domain")
        root.append(tuple(domain))
    stack = [iter((tuple(root),))]
    frontier = []
    closures = 0
    r = int(2*t*q*q)

    def next_obligation():
        while stack:
            box = next(stack[-1],None)
            if box is not None:
                return box
            stack.pop()
        return None

    for line in stream:
        record = read_record(line)
        box = next_obligation()
        require(box is not None,"extra record after coverage")
        op = record["op"]
        if op=="SPLIT":
            keys(record,"op point axis cut")
            stack.append(iter(split(box,record["point"],record["axis"],record["cut"],q)))
        elif op=="GRAPH_ENTER":
            keys(record,"op M conflicts")
            require(len(box)<n,"graph inside local box")
            stack.append(graph_boxes(box,q,record["M"],n,t,record["conflicts"]))
        elif op=="CLOSE_BOUND":
            keys(record,"op triple")
            lo,hi = det_range(box,triple_at(record["triple"],box),q)
            require(-r<=lo<=hi<=r,"direct bound failed")
            closures += 1
        elif op=="CLOSE_P2":
            keys(record,"op weights")
            require(p2_bound(box,record["weights"],q,t)<=t,"P2 inequality failed")
            closures += 1
        elif op=="FRONTIER_YIELD":
            keys(record,"op")
            frontier.append(box)
        else:
            raise InvalidTrace("unsupported opcode")
    require(next_obligation() is None,"incomplete trace")
    return dict(complete=not frontier,closures=closures,frontier=frontier)


if __name__=="__main__":
    expected = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"),
                          object_pairs_hook=unique_object) if len(sys.argv)>2 else None
    with Path(sys.argv[1]).open(encoding="utf-8") as trace:
        print(json.dumps(verify(trace,expected)))
