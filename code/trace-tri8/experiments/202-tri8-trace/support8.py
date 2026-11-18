"""Exp 202: pinned triangle n = 8 trace statement pieces shared by the emitter and the release checker:
family roots, the side-gap contraction (proven raw n = 6 bound 1/16), the cutoff and the per-family header.
Imports no search code."""
from fractions import Fraction as F
from hashlib import sha256
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(1, str(HERE.parent / "152-geometry-trace"))
from verify_trace import require
Q = 30*2**14
import basin8
from math import isqrt
H = Q//2
O,X,Y = ("R",0,0,0,0),("R",Q,Q,0,0),("R",0,0,Q,Q)
MB,ML,MH = ("R",H,H,0,0),("R",0,0,H,H),("R",H,H,H,H)
B1,B2 = ("R",0,H,0,0),("R",H,Q,0,0)
L1,L2 = ("R",0,0,0,H),("R",0,0,H,Q)
H1,H2 = ("H",0,H,H,Q),("H",H,Q,0,H)
ROOTS = dict(MMM=(MB,ML,MH),MMS=(B1,B2,ML,MH),MSS0=(B1,B2,L1,L2,MH),MSS1=(O,B2,L2,MH),
             SSS0=(B1,B2,L1,L2,H1,H2),SSS1=(O,B2,L2,H1,H2),SSS2=(O,X,L2,H1),SSS3=(O,X,Y))
PAIRS = {"MMS":((0,1,0),),"MSS0":((0,1,0),(2,3,1)),"SSS1":((3,4,0),),"SSS0":((0,1,0),(2,3,1),(4,5,0))}
CUTOFF = "1694730329660199977615339815376729875653/50000000000000000000000000000000000000000"   # exp 127 n=8 exact T


def basin_header():
    points,ties,walls,t0 = basin8.load()
    radius = basin8.production_radius()
    out = basin8.verify(points,ties,walls,t0,radius)
    return str(out["t_upper"]),str(radius)


def statement(name):
    optimum,radius = basin_header()
    return dict(op="HEADER",version=2,n=8,Q=Q,root=[list(r) for r in ROOTS[name]],
                cutoff_raw_t=CUTOFF,optimum_raw_t=optimum,cover_radius=radius,
                basin_sha256=sha256(basin8.SOURCE.read_bytes()).hexdigest())


def gap_grid(threshold):
    """Outward-rounded support gap from the proven raw n = 6 bound 1/16 (the exp 163 rule with n - 2 = 6)."""
    upper = F(1,16)
    require(0<threshold<upper,"unsupported gap threshold")
    scale = 10**20
    sqrt_lower = F(isqrt(int(threshold/upper*scale**2)),scale)
    return int(Q*2*threshold/(1-sqrt_lower))


def contract(name,domains,threshold):
    root = ROOTS[name]
    require(len(domains)==len(root),"invalid support count")
    for b,r in zip(domains,root):
        require(b[0]==r[0] and r[1]<=b[1]<=b[2]<=r[2] and r[3]<=b[3]<=b[4]<=r[4],
                "gap domain outside canonical supports")
    out = [list(b) for b in domains]
    gap = gap_grid(threshold)
    for i,j,axis in PAIRS[name]:
        a,b = out[i],out[j]
        k = 1+2*axis
        if b[k+1]-a[k]<gap:
            return None
        a[k+1] = min(a[k+1], b[k+1]-gap)
        b[k] = max(b[k], a[k]+gap)
        for row in (a,b):
            if row[0]=="H":
                row[3],row[4] = Q-row[2],Q-row[1]
    return tuple(tuple(b) for b in out)


