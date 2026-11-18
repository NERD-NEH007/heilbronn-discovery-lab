"""Pinned exact SSS0 constraints, shared by the copied emitter and verifier."""
from fractions import Fraction as F
from hashlib import sha256
import json
from math import isqrt

Q = 606208
AXES = (0, 0, 1, 1, 0, 0)
REDUCTION = {
    "family": "SSS0",
    "group": "all six vertex permutations of the reference triangle",
    "support_order": ["B1", "B2", "L1", "L2", "H1", "H2"],
    "parameter_axes": list(AXES),
    "order_rows": [[0,2], [0,4]],
    "sum_rows": [[0,1], [0,3], [0,5]],
    "closed_ties": True,
}
DIGEST = sha256(json.dumps(REDUCTION,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def gap_only(domains, threshold):
    out = [list(row) for row in domains]
    ratio = threshold/F(7,144)
    gap = int(2*Q*threshold/(1-F(isqrt(int(ratio*10**40)),10**20)))
    for i,j in ((0,1),(2,3),(4,5)):
        k = 1+2*AXES[i]
        a,b = out[i],out[j]
        if b[k+1]-a[k]<gap:
            return None
        a[k+1] = min(a[k+1],b[k+1]-gap)
        b[k] = max(b[k],a[k]+gap)
        for row in (a,b):
            if row[0]=="H":
                row[3:] = Q-row[2],Q-row[1]
    return tuple(map(tuple,out))


def contract(domains, threshold):
    box = tuple(map(tuple,domains))
    assert len(box) in (6,9)
    for _ in range(50):
        narrowed = gap_only(box,threshold)
        if narrowed is None:
            return None
        out = [list(row) for row in narrowed]
        for i,j in REDUCTION["order_rows"]:
            a,b = 1+2*AXES[i],1+2*AXES[j]
            out[i][a+1] = min(out[i][a+1],out[j][b+1])
            out[j][b] = max(out[j][b],out[i][a])
        for i,j in REDUCTION["sum_rows"]:
            a,b = 1+2*AXES[i],1+2*AXES[j]
            out[i][a+1] = min(out[i][a+1],Q-out[j][b])
            out[j][b+1] = min(out[j][b+1],Q-out[i][a])
        if any(row[1]>row[2] or row[3]>row[4] for row in out):
            return None
        for i in (4,5):
            out[i][3:] = Q-out[i][2],Q-out[i][1]
        new = tuple(map(tuple,out))
        if new==box:
            break
        box = new
    return box
