"""Exact enumeration conditional on the midpoint and flush characterization."""
from fractions import Fraction as F
from itertools import combinations
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = HERE.parent/'127-triangle-n9/cert/local_certs_tri_n9.json'
points = [tuple(map(F,p)) for p in json.loads(source.read_text())['9'][0]['points_exact']]


def sub(a,b):
    return tuple(x-y for x,y in zip(a,b))


def cross(a,b):
    return a[0]*b[1]-a[1]*b[0]


def orient(a,b,c):
    return cross(sub(b,a),sub(c,a))


def hull(points):
    def half(seq):
        out = []
        for p in seq:
            while len(out)>1 and orient(*out[-2:],p)<=0:
                out.pop()
            out.append(p)
        return out[:-1]
    p = sorted(set(points))
    return half(p)+half(p[::-1])


def intersect(edge, other):
    a,b = edge
    c,d = other
    u,v = sub(b,a),sub(d,c)
    den = cross(u,v)
    if den:
        t = cross(sub(c,a),v)/den
        return tuple(x+t*y for x,y in zip(a,u))


def section(edges,h):
    xs = []
    for (x,y),(u,v) in edges:
        if min(y,v)<=h<=max(y,v):
            xs.extend([x,u] if y==v else [x+(u-x)*(h-y)/(v-y)])
    return min(xs),max(xs)


def valid(t,hull_edges):
    if None in t or not orient(*t):
        return False
    if orient(*t)<0:
        t = t[::-1]
    return all(orient(a,b,p)>=0 for a,b in zip(t,t[1:]+t[:1]) for p,_ in hull_edges) and all(
        any(orient(u,v,m)==0 and all(min(f,g)<=k<=max(f,g) for f,g,k in zip(u,v,m))
            for u,v in hull_edges)
        for a,b in zip(t,t[1:]+t[:1]) for m in [tuple((x+y)/2 for x,y in zip(a,b))])


def main():
    p = hull(points)
    edges = list(zip(p,p[1:]+p[:1]))
    candidates = []
    for base in edges:
        a,b = base
        d = sub(b,a)
        length = sum(v*v for v in d)

        def frame(p):
            q = sub(p,a)
            return sum(x*y for x,y in zip(q,d))/length,cross(d,q)

        def unframe(p):
            x,y = p
            return a[0]+x*d[0]-y*d[1]/length,a[1]+x*d[1]+y*d[0]/length

        he = [(frame(u),frame(v)) for u,v in edges]
        for sides in combinations([e for e in edges if e!=base],2):
            t = tuple(intersect(*ab) for ab in combinations((base,*sides),2))
            if valid(t,edges):
                candidates.append((abs(orient(*t))/2,tuple(sorted(t)),False))
        for h in sorted({p[1] for p,_ in he if p[1]>0}):
            left,right = section(he,h)
            if left==right:
                continue
            low,high = sorted((frame(a)[0],frame(b)[0]))
            feasible = True
            for (x,y),_ in he:
                s = y/(2*h)
                if s>1:
                    feasible = False
                    break
                coefficient = 1-2*s
                lower = x-right+left-2*s*left
                upper = x-left+right-2*s*right
                if not coefficient:
                    feasible &= lower<=0<=upper
                else:
                    bounds = sorted((lower/coefficient,upper/coefficient))
                    low,high = max(low,bounds[0]),min(high,bounds[1])
            if feasible and low<=high:
                z = low
                t = tuple(sorted(map(unframe,[(z+left-right,F(0)),(z+right-left,F(0)),
                                              (left+right-z,2*h)])))
                assert valid(t,edges)
                candidates.append((2*h*(right-left),t,low!=high))
    optimum = F(1,2)
    shape = ((F(0),F(0)),(F(0),F(1)),(F(1),F(0)))
    assert any(a==optimum for a,_,_ in candidates)
    assert all(a>optimum or (a==optimum and t==shape and not continuum)
               for a,t,continuum in candidates)
    print(json.dumps(dict(hull=[[str(x) for x in v] for v in p],
                          areas=sorted({str(a) for a,_,_ in candidates}),
                          candidates=len(candidates)),indent=2))


if __name__ == '__main__':
    main()
