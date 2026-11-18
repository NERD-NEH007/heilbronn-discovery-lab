"""Emit and replay MSS1 over both mirror halves, with exact shard coverage."""

from hashlib import file_digest, sha256
from itertools import combinations, islice
import json
from pathlib import Path
from runpy import run_path
import sys
from time import process_time
from tempfile import NamedTemporaryFile

HERE = Path(__file__).resolve().parent
for sibling in ("159-tri9-mmm-production-trace","160-tri9-sss3-production-trace","155-sharded-trace"):
    sys.path.insert(0,str(HERE.parent/sibling))
from emit_mmm import Emitter, F, SOURCE, LocalTranscriptBudgetExceeded, emit_tree, load, pin_cpu
from graph_replay import graph_boxes
import stream_trace
from verify_manifest import verify_manifest
from verify_trace import read_record, require

BASELINE = HERE.parent/"127-triangle-n9"
GRID,Q = 37,37*2**14
CAP = 2048


def statement():
    _,optimum,radius = load()
    hist = json.loads((BASELINE/"results/n9_MSS1_exact_1e-09_G37.json").read_text())
    h = Q//2
    return dict(op="HEADER",version=2,n=9,Q=Q,root=[
        ["R",0,0,0,0],["R",h,Q,0,0],["R",0,0,h,Q],["R",h,h,h,h]],
        cutoff_raw_t=hist["T"],optimum_raw_t=str(optimum),cover_radius=str(radius),
        basin_sha256=sha256(SOURCE.read_bytes()).hexdigest())


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")


def emit(name="MSS1", output=HERE, make_statement=statement, cap=CAP, contract=None,
         use_cover=False, resume=False, split_failed_path=None):
    import scipy.optimize as optimizer
    driver = run_path(str(BASELINE/"run.py"))
    engine,np = driver["cgt"],driver["np"]
    family,t,rho,rq,q,_,_,gap = driver["setup"](9,name,"exact",1e-9,GRID)
    family = engine.make_family(name,q,int(gap*q),sym=False)
    header = make_statement()
    require(q==Q and str(t)==header["cutoff_raw_t"],"archived cutoff mismatch")
    require((contract is not None or not len(family.ORD)) and not len(family.SUM),
            "unexpected support restriction")
    require(header["root"]==[["H" if k else "R",*r]
                           for k,r in zip(family.SK,family.root().tolist())],"support mismatch")
    require(not (output/"manifest.json").exists(),"manifest already exists")
    triples = np.asarray(list(combinations(range(family.k),3)),np.int64)
    counts = dict(cliques=0,split=0,bound=0,p2=0,lp_calls=0,graph_roots=0,outer_splits=0)
    start = process_time()
    cover = None
    if use_cover:
        points,_,_ = load()
        cover = engine.make_cover(points,F(header["cover_radius"]),q)

    def expected_shard(bounds):
        return dict(header,root=[["H" if k else "R",*r]
                                 for k,r in zip(family.SK,bounds.tolist())])

    def shard_entry(dest):
        with dest.open("rb") as handle:
            return dict(path=dest.name,sha256=file_digest(handle,"sha256").hexdigest())

    def shard(bounds,path,body):
        dest = output/f"shard-{path or 'root'}.jsonl"
        expected = expected_shard(bounds)
        with NamedTemporaryFile(mode="w",encoding="utf-8",newline="\n",dir=output,
                                prefix=dest.stem+"-",suffix=".partial",delete=False) as handle:
            temp = Path(handle.name)
            writer = Emitter(handle)
            writer.emit(expected)
            body(writer)
        require(not dest.exists(),"proof shard already exists")
        temp.rename(dest)
        print(f"{dest.name}: {dest.stat().st_size} bytes, {counts['cliques']} cliques so far",flush=True)
        return shard_entry(dest)

    def partition(bounds,path="",apply_contract=True):
        if contract is not None and apply_contract:
            domains = tuple(("H" if k else "R",*row)
                            for k,row in zip(family.SK,bounds.tolist()))
            narrowed = contract(domains)
            if narrowed is None:
                counts["gap_closures"] = counts.get("gap_closures",0)+1
                return {"contract":"SIDE_GAP","next":None}
            if narrowed!=domains:
                counts["gap_contractions"] = counts.get("gap_contractions",0)+1
                next_bounds = np.array([r[1:] for r in narrowed],dtype=np.int64)
                return {"contract":"SIDE_GAP","next":partition(next_bounds,path,False)}
        existing = output/f"shard-{path or 'root'}.jsonl"
        if resume and existing.is_file():
            with existing.open(encoding="utf-8") as handle:
                require(read_record(handle.readline()) == expected_shard(bounds),
                        f"saved shard root mismatch: {existing.name}")
            counts["reused_shards"] = counts.get("reused_shards",0)+1
            return shard_entry(existing)
        force_split = path == split_failed_path
        if force_split:
            with existing.with_suffix(".partial").open(encoding="utf-8") as handle:
                require(read_record(handle.readline()) == expected_shard(bounds),
                        "failed shard root mismatch")
        widths = (bounds[:,1::2]-bounds[:,0::2])*family.SSPL
        i,axis = np.unravel_index(widths.argmax(),widths.shape)
        lo,hi = engine.all_ranges(bounds,family.SK,q,triples)
        direct = np.flatnonzero((lo>=-rq)&(hi<=rq))
        if len(direct):
            counts["bound"] += 1
            return shard(bounds,path,lambda writer:writer.emit(dict(op="CLOSE_BOUND",triple=int(direct[0]))))
        if not force_split and widths[i,axis]<=Q//8:
            stats = engine.new_stats()
            cells,adj = engine.cell_graph(bounds,family,GRID,q,rq,stats,rho,0)
            cliques = list(islice(engine.k_cliques_iter(adj,family.m,cells,q,rq,stats),cap+1))
            if len(cliques)<=cap:
                def content(writer):
                    writer.emit(dict(op="GRAPH_ENTER",M=GRID,conflicts=[]))
                    for clique in cliques:
                        local = np.vstack((bounds,cells[list(clique)]))
                        emit_tree(engine,np,optimizer,writer,local,family,rho,rq,q,counts,cover)
                        counts["cliques"] += 1
                before = counts.copy()
                try:
                    result = shard(bounds,path,content)
                except LocalTranscriptBudgetExceeded:
                    counts.clear()
                    counts.update(before)
                    print(f"Support path {path or 'root'}: local budget exhausted, splitting",flush=True)
                else:
                    counts["graph_roots"] += 1
                    return result
        low,high = map(int,bounds[i,2*axis:2*axis+2])
        cut = (low+high)//2
        require(low<cut<high,"terminal support grid width")
        a,b = bounds.copy(),bounds.copy()
        a[i,2*axis+1] = b[i,2*axis] = cut
        if family.SK[i]==1:
            require(axis==0,"hypotenuse axes must split on x")
            for child in (a,b):
                child[i,2],child[i,3] = q-child[i,1],q-child[i,0]
        counts["outer_splits"] += 1
        return dict(point=int(i),axis=int(axis),cut=cut,
                    children=[partition(a,path+"0"),partition(b,path+"1")])

    write_json(output/"expected-statement.json",header)
    partition_tree = partition(family.root())
    manifest = output/"manifest.json"
    pending_manifest = output/"manifest.partial"
    write_json(pending_manifest,dict(version=1,statement=header,partition=partition_tree))
    pending_manifest.rename(manifest)
    write_json(output/"emission.json",dict(**counts,cpu_seconds=process_time()-start))
    print(f"{name} manifest complete.",flush=True)


def replay(output=HERE,make_statement=statement,contract=None):
    # This dispatcher uses one worker, so the graph override remains local to
    # the verification process. The search engine is never imported.
    stream_trace.graph_boxes = graph_boxes
    manifest = json.loads((output/"manifest.json").read_text())
    start = process_time()
    receipt = verify_manifest(manifest,make_statement(),output,contract=contract)
    receipt["worker_cpu_seconds"] = process_time()-start
    write_json(output/"replay-receipt.json",receipt)
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":
    pin_cpu()
    if sys.argv[1:]==["emit"]:
        emit()
    elif sys.argv[1:]==["replay"]:
        replay()
    else:
        raise SystemExit("Specify emit or replay.")
