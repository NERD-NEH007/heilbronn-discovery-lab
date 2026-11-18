"""Independent exact graph reconstruction with cached forbidden-third-cell masks."""

from fractions import Fraction
from functools import lru_cache
from itertools import combinations

import numpy as np

from verify_trace import require, vertices


def vertex_array(box, q):
    corners = vertices(box,q)
    require(bool(corners),"empty graph point domain")
    return np.asarray(corners+[corners[0]]*(5-len(corners)),dtype=np.int64)


def blocks(a, b, candidates, r):
    """For each candidate, evaluate every signed determinant of three vertices."""
    u = b[None,:,:]-a[:,None,:]
    v = candidates[:,None,:,:]-a[None,:,None,:]
    d = u[None,:,:,None,0]*v[:,:,None,:,1]-u[None,:,:,None,1]*v[:,:,None,:,0]
    return (d.min(axis=(1,2,3))>=-r)&(d.max(axis=(1,2,3))<=r)


def mask(flags):
    return int.from_bytes(np.packbits(flags,bitorder="little").tobytes(),"little")


def graph_boxes(support, q, grid, n, threshold, conflicts):
    require(conflicts==[],"this replay variant requires no supplied graph conflicts")
    require(type(grid) is int and 1<=grid<=q and q%grid==0
            and Fraction(2,grid)<=2*threshold,"cell occupancy condition failed")
    require(2*q*q<2**63,"integer determinant overflow bound failed")
    require(0<2*threshold<=2,"threshold outside accelerated range")
    r = int(2*threshold*q*q)
    width = q//grid
    domains = [("R",x*width,(x+1)*width,y*width,(y+1)*width)
               for x in range(grid) for y in range(grid-x)]
    xyz = np.asarray([vertex_array(d,q) for d in domains])
    supports = [vertex_array(d,q) for d in support]
    keep = np.ones(len(domains),dtype=bool)
    for a,b in combinations(supports,2):
        keep &= ~blocks(a,b,xyz,r)
    cells = [cell for cell,kept in zip(domains,keep.tolist()) if kept]
    xyz = xyz[keep]
    count = len(cells)
    edges = []
    for i in range(count):
        keep = np.arange(count)>i
        for a in supports:
            keep &= ~blocks(a,xyz[i],xyz,r)
        edges.append(mask(keep))

    @lru_cache(maxsize=32768)
    def forbidden(a,b):
        return mask(blocks(xyz[a],xyz[b],xyz,r))

    def visit(selected,candidates):
        need = n-len(support)-len(selected)
        if need==0:
            yield support+tuple(cells[i] for i in selected)
            return
        while candidates.bit_count()>=need:
            bit = candidates&-candidates
            candidates ^= bit
            i = bit.bit_length()-1
            children = candidates&edges[i]
            if need>1:
                for j in selected:
                    if children.bit_count()<need-1:
                        break
                    children &= ~forbidden(j,i)
            yield from visit(selected+(i,),children)

    return visit((),(1<<count)-1)
