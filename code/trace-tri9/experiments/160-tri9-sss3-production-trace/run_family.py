"""Single-core SSS3 trace emission and exact independent replay."""

from fractions import Fraction
import gzip
from hashlib import sha256
import json
from pathlib import Path
from runpy import run_path
import sys
from time import perf_counter, process_time

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"159-tri9-mmm-production-trace"))
from emit_mmm import Emitter, SOURCE, emit_tree, load, pin_cpu
import stream_trace
from verify_trace import require

GRID = 37
Q = GRID*2**14
BASELINE = HERE.parent/"127-triangle-n9"


def statement():
    _,optimum,radius = load()
    archive = json.loads((BASELINE/"results/n9_SSS3_exact_1e-09_G37.json").read_text())
    return dict(op="HEADER",version=2,n=9,Q=Q,cutoff_raw_t=archive["T"],root=[
        ["R",0,0,0,0],["R",Q,Q,0,0],["R",0,0,Q,Q]],
        optimum_raw_t=str(optimum),cover_radius=str(radius),
        basin_sha256=sha256(SOURCE.read_bytes()).hexdigest())


def emit():
    import scipy.optimize as optimizer
    driver = run_path(str(BASELINE/"run.py"))
    engine,np = driver["cgt"],driver["np"]
    family,t,rho,rq,q,_,_,_ = driver["setup"](9,"SSS3","exact",1e-9,GRID)
    header = statement()
    require(q==Q and str(t)==header["cutoff_raw_t"],"archived cutoff mismatch")
    require(not family.nparam and not np.any(family.K) and not len(family.ORD)
            and not len(family.SUM),"unexpected SSS3 domain")
    require([r[1:] for r in header["root"]]==family.root().tolist(),"SSS3 support mismatch")
    counts = dict(cliques=0,split=0,bound=0,p2=0,lp_calls=0)
    start = process_time()
    path = HERE/"sss3.jsonl"
    require(not path.exists(),"trace already exists")
    temp = path.with_suffix(".partial")
    with temp.open("x",encoding="utf-8",newline="\n",buffering=1<<20) as out:
        emitter = Emitter(out)
        emitter.emit(header)
        emitter.emit(dict(op="GRAPH_ENTER",M=GRID,conflicts=[]))
        stats = engine.new_stats()
        cells,adj = engine.cell_graph(family.root(),family,GRID,q,rq,stats,rho,0)
        for clique in engine.k_cliques_iter(adj,family.m,cells,q,rq,stats):
            box = np.vstack((family.root(),cells[list(clique)]))
            emit_tree(engine,np,optimizer,emitter,box,family,rho,rq,q,counts)
            counts["cliques"] += 1
    temp.rename(path)
    data = dict(**counts,cpu_seconds=process_time()-start,bytes=path.stat().st_size,
                sha256=sha256(path.read_bytes()).hexdigest())
    (HERE/"emission.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
    (HERE/"expected-statement.json").write_text(json.dumps(header,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(data,indent=2))


def replay():
    from graph_replay import graph_boxes
    stream_trace.graph_boxes = graph_boxes
    path = HERE/"sss3.jsonl"
    start,cpu = perf_counter(),process_time()
    with path.open("rb") as source:
        receipt = stream_trace.verify(source,statement())
    elapsed,worker = perf_counter()-start,process_time()-cpu
    raw = path.read_bytes()
    require(receipt["complete"] and receipt["pending"]==0,"partial SSS3 replay")
    receipt.update(worker_cpu_seconds=worker,wall_seconds=elapsed,bytes=len(raw),
                   gzip_bytes=len(gzip.compress(raw)),sha256=sha256(raw).hexdigest())
    (HERE/"replay-receipt.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":
    pin_cpu()
    if sys.argv[1:]==["emit"]:
        emit()
    elif sys.argv[1:]==["replay"]:
        replay()
    else:
        raise SystemExit("Specify emit or replay.")
