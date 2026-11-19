"""Side-gap contraction for the triangle n = 9 release checker (MMS, MSS0, SSS1), as a standalone module.
Verbatim copy of contract/gap_grid/ROOTS/PAIRS from 163-tri9-trace-queue/queue_traces.py (2026-11-19), so the
release checker no longer imports queue_traces, whose module-level imports load the emitter modules
(161/run_family.py, 159/emit_mmm.py, 154/emitter.py). Depends only on the standard library and verify_trace.require.
test_contract9.py checks that it agrees with the original on random and boundary boxes."""
from fractions import Fraction as F
from math import isqrt
from verify_trace import require

Q = 606208
H = Q//2
B1,B2 = ("R",0,H,0,0),("R",H,Q,0,0)
L1,L2 = ("R",0,0,0,H),("R",0,0,H,Q)
H1,H2 = ("H",0,H,H,Q),("H",H,Q,0,H)
O = ("R",0,0,0,0)
ML,MH = ("R",0,0,H,H),("R",H,H,H,H)
ROOTS = {
    "MMS": (B1,B2,ML,MH),
    "MSS0": (B1,B2,L1,L2,MH),
    "SSS1": (O,B2,L2,H1,H2),
    "SSS0": (B1,B2,L1,L2,H1,H2),
}
PAIRS = {
    "MMS": ((0,1,0),),
    "MSS0": ((0,1,0),(2,3,1)),
    "SSS1": ((3,4,0),),
    "SSS0": ((0,1,0),(2,3,1),(4,5,0)),
}


def gap_grid(threshold):
    """Derive an outward-rounded support gap using the raw n=7 bound."""
    upper = F(7,144)
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
