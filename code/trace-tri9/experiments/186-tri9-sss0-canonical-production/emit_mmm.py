"""Emit a complete fixed-support MMM trace from a fresh, single-core search."""

import ctypes
from hashlib import sha256
from itertools import permutations
import json
import os
from pathlib import Path
from runpy import run_path
import sys
from time import process_time

for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS"):
    os.environ[key] = "1"

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"154-streaming-cover"))
sys.path.insert(0,str(HERE))
from emitter import Emitter
from stream_trace import SOURCE, F, load
from verify_trace import require


def pin_cpu():
    kernel = ctypes.WinDLL("kernel32",use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),
                                             ctypes.POINTER(ctypes.c_size_t)]
    kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p,ctypes.c_size_t]
    proc = kernel.GetCurrentProcess()
    mask,system = ctypes.c_size_t(),ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(proc,ctypes.byref(mask),ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    if not kernel.SetProcessAffinityMask(proc,1<<(mask.value.bit_length()-1)):
        raise ctypes.WinError(ctypes.get_last_error())


class LocalTranscriptBudgetExceeded(RuntimeError):
    pass


def emit_tree(engine, np, optimizer, emitter, box, family, rho, rq, q, counts, cover=None,
              canonical_contract=None, guard=None):
    solve = optimizer.linprog
    candidate = None

    def record_proposal(*args,**kwargs):
        nonlocal candidate
        result = solve(*args,**kwargs)
        candidate = result.x if result.success else None
        return result

    stack = [(box,0)]
    nodes = 0
    while stack:
        if guard is not None and nodes % 64 == 0:
            guard()
        bounds,depth = stack.pop()
        nodes += 1
        if nodes > 200000:
            raise LocalTranscriptBudgetExceeded("local transcript budget exhausted")
        if canonical_contract is not None:
            domains = tuple(("H" if kind else "R",*row)
                            for kind,row in zip(family.K,bounds.tolist()))
            narrowed = canonical_contract(domains)
            if narrowed != domains:
                emitter.emit(dict(op="CANONICAL_CONTRACT",closed=narrowed is None))
                if narrowed is None:
                    counts["canonical_empty"] = counts.get("canonical_empty",0)+1
                    continue
                bounds = np.asarray([row[1:] for row in narrowed],np.int64)
        lo,hi = engine.all_ranges(bounds,family.K,q,engine.TRI)
        direct = np.flatnonzero((lo>=-rq)&(hi<=rq))
        if len(direct):
            emitter.emit(dict(op="CLOSE_BOUND",triple=int(direct[0])))
            counts["bound"] += 1
            continue
        if cover is not None:
            match = cover(bounds,family.K,want=True)
            if match:
                (matrix,offset),assignment = match
                corners = ((0,0),(1,0),(0,1))
                mapped = [tuple(sum(a*v for a,v in zip(row,p))+b for row,b in zip(matrix,offset))
                          for p in corners]
                perm = next(list(p) for p in permutations(range(3))
                            if mapped==[(int(i==p[1]),int(i==p[2])) for i in range(3)])
                emitter.emit(dict(op="CLOSE_COVER",barycentric_permutation=perm,
                                  point_assignment=assignment))
                counts["cover"] = counts.get("cover",0)+1
                continue
        hint = -1
        if depth>=3:
            candidate = None
            optimizer.linprog = record_proposal
            try:
                closed,hint = engine.p2_try(bounds,family.K,lo,hi,rq,rho,q,engine.TRI)
            finally:
                optimizer.linprog = solve
            counts["lp_calls"] += 1
            if closed:
                eligible = np.flatnonzero((lo>=-rq)|(hi<=rq))
                weights = [[int(i),round(float(w)*2**30)] for i,w in zip(eligible,candidate)
                           if round(float(w)*2**30)>0]
                emitter.emit(dict(op="CLOSE_P2",weights=weights))
                counts["p2"] += 1
                continue
        if hint<0 or not family.SPL[hint//2,hint%2]:
            hint = engine.choose_split(bounds,family.SPL,lo,hi,engine.TRI,rq)
        require(hint>=0,"unresolved grid-terminal leaf")
        p,a = divmod(int(hint),2)
        low,high = (int(x) for x in bounds[p,2*a:2*a+2])
        cut = (low+high)//2
        require(low<cut<high,"invalid local split")
        emitter.emit(dict(op="SPLIT",point=p,axis=a,cut=cut))
        counts["split"] += 1
        first,second = bounds.copy(),bounds.copy()
        first[p,2*a+1] = second[p,2*a] = cut
        for child in (second,first):
            if family.K[p] == 1:
                require(a==0,"hypotenuse axes must split on x")
                child[p,2],child[p,3] = q-child[p,1],q-child[p,0]
            if child[p,0]+child[p,2]<=q:
                stack.append((child,depth+1))


def main():
    pin_cpu()
    import scipy.optimize as optimizer
    driver = run_path(str(HERE.parent/"127-triangle-n9/run.py"))
    engine,np = driver["cgt"],driver["np"]
    family,t,rho,rq,q,cfg,_,_ = driver["setup"](9,"MMM","exact",1e-9,37)
    require(not family.nparam and not np.any(family.K) and not len(family.ORD)
            and not len(family.SUM),"unexpected MMM domain")
    _,optimum,radius = load()
    root = [["R",*(q//2,q//2),0,0],["R",0,0,q//2,q//2],["R",q//2,q//2,q//2,q//2]]
    require([row[1:] for row in root]==family.root().tolist(),"MMM midpoint root mismatch")
    statement = dict(op="HEADER",version=2,n=9,Q=q,root=root,
                     cutoff_raw_t=str(t),optimum_raw_t=str(optimum),cover_radius=str(radius),
                     basin_sha256=sha256(SOURCE.read_bytes()).hexdigest())
    counts = dict(cliques=0,split=0,bound=0,p2=0,lp_calls=0)
    start = process_time()
    path = HERE/"mmm.jsonl"
    temp = path.with_suffix(".partial")
    with temp.open("x",encoding="utf-8",newline="\n",buffering=1<<20) as output:
        emitter = Emitter(output)
        emitter.emit(statement)
        emitter.emit(dict(op="GRAPH_ENTER",M=37,conflicts=[]))
        stats = engine.new_stats()
        cells,graph = engine.cell_graph(family.root(),family,37,q,rq,stats,rho,0)
        for clique in engine.k_cliques_iter(graph,family.m,cells,q,rq,stats):
            box = np.vstack((family.root(),cells[list(clique)]))
            emit_tree(engine,np,optimizer,emitter,box,family,rho,rq,q,counts)
            counts["cliques"] += 1
    require(not path.exists(),"production trace already exists")
    temp.rename(path)
    (HERE/"mmm-statement.json").write_text(json.dumps(statement,indent=2)+"\n",encoding="utf-8")
    result = dict(**counts,worker_cpu_seconds=process_time()-start,bytes=path.stat().st_size,
                  sha256=sha256(path.read_bytes()).hexdigest())
    (HERE/"emission.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
