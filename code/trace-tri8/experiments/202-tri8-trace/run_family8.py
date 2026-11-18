"""Exp 202 (Claude, 2026-11-18): triangle n = 8 trace emitter. Copy of the exp 186 emitter (run_family_sub,
itself the exp 161 run_family plus subtree helpers and known-failed splits). The changes are n = 8, grid 30,
the exp 127 engine (which carries the n = 8 certificate), per-family pinned statements, the exact n = 8 basin
(basin8.py) for covers, and the side-gap contraction with the proven n = 6 bound (raw 1/16)."""

from hashlib import file_digest, sha256
from itertools import combinations, islice
import json
from pathlib import Path
from runpy import run_path
import sys
from time import process_time
from tempfile import NamedTemporaryFile

HERE = Path(__file__).resolve().parent
for sibling in ("159-tri9-mmm-production-trace","160-tri9-sss3-production-trace","155-sharded-trace",
                "186-tri9-sss0-canonical-production"):   # last insert wins: exp 186 emit_mmm (emit_tree with optional canonical/guard)
    sys.path.insert(0,str(HERE.parent/sibling))
sys.path.insert(0,str(HERE))
from emit_mmm import Emitter, F, SOURCE, LocalTranscriptBudgetExceeded, emit_tree, load, pin_cpu
from graph_replay import graph_boxes
import stream_trace
from verify_manifest import verify_manifest
from verify_trace import read_record, require

BASELINE = HERE.parent/"127-triangle-n9"
GRID,Q = 30,30*2**14
CAP = 2048
KNOWN_FAILED = {}   # support path -> a .partial left by a local-budget failure at that path (exp 186 parallel pass)


from support8 import H, ROOTS, PAIRS, CUTOFF, basin_header, statement, gap_grid, contract
import basin8


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")


def emit(name, output, make_statement, cap=CAP, contract=None,
         use_cover=False, resume=False, split_failed_path=None,
         canonical_contract=None, reduction_digest=None, guard=None, bounded_root=False,
         on_shard=None, subtree=None):
    import scipy.optimize as optimizer
    driver = run_path(str(HERE.parent/"127-triangle-n9/run.py"))
    engine,np = driver["cgt"],driver["np"]
    family,t,rho,rq,q,_,_,gap = driver["setup"](8,name,"exact",1e-9,GRID)
    family = engine.make_family(name,q,int(gap*q),sym=False)
    header = make_statement()
    require(q==Q and str(t)==header["cutoff_raw_t"],"archived cutoff mismatch")
    require((contract is not None or not len(family.ORD)) and not len(family.SUM),
            "unexpected support restriction")
    require(not bounded_root or name=="SSS0","unsupported pilot root")
    root = family.root()
    if bounded_root:
        root = np.asarray([row[1:] for row in header["root"]],dtype=np.int64)
    require(header["root"]==[["H" if k else "R",*r]
                           for k,r in zip(family.SK,root.tolist())],"support mismatch")
    require(not (output/"manifest.json").exists(),"manifest already exists")
    require(subtree is None or resume,"subtree helpers must run with resume semantics")
    triples = np.asarray(list(combinations(range(family.k),3)),np.int64)
    KNOWN_FAILED.clear()
    for partial in output.glob("shard-*.partial"):
        KNOWN_FAILED.setdefault(partial.name[len("shard-"):].rsplit("-",1)[0],partial)
    counts = dict(cliques=0,split=0,bound=0,p2=0,lp_calls=0,graph_roots=0,outer_splits=0)
    start = process_time()
    cover = None
    if use_cover:
        points = basin8.load()[0]
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
        entry = shard_entry(dest)
        if on_shard is not None:
            on_shard(expected,entry,counts["cliques"])
        return entry

    def partition(bounds,path="",apply_contract=True):
        if guard is not None:
            guard()
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
            entry = shard_entry(existing)
            if on_shard is not None:
                on_shard(expected_shard(bounds),entry,counts["cliques"])
            return entry
        force_split = path == split_failed_path
        if force_split:
            with existing.with_suffix(".partial").open(encoding="utf-8") as handle:
                require(read_record(handle.readline()) == expected_shard(bounds),
                        "failed shard root mismatch")
        known = KNOWN_FAILED.get(path or "root")
        if not force_split and known is not None:
            # a recorded local-budget failure at exactly this support box: split without retrying.
            # A partial cut off by a killed process may not even hold a complete header; then it records
            # nothing and the box is attempted normally.
            try:
                with known.open(encoding="utf-8") as handle:
                    header_ok = read_record(handle.readline()) == expected_shard(bounds)
            except Exception:
                header_ok = None
            if header_ok is not None:
                require(header_ok,"known failed shard root mismatch")
                counts["known_failed_splits"] = counts.get("known_failed_splits",0)+1
                force_split = True
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
            cliques = []
            for clique in islice(engine.k_cliques_iter(adj,family.m,cells,q,rq,stats),cap+1):
                if guard is not None and len(cliques)%256==0:
                    guard()
                cliques.append(clique)
            if len(cliques)<=cap:
                def content(writer):
                    writer.emit(dict(op="GRAPH_ENTER",M=GRID,conflicts=[]))
                    for clique in cliques:
                        local = np.vstack((bounds,cells[list(clique)]))
                        emit_tree(engine,np,optimizer,writer,local,family,rho,rq,q,counts,cover,
                                  canonical_contract,guard)
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

    if subtree is not None:
        bounds = root.copy()
        for bit in subtree:
            if contract is not None:
                domains = tuple(("H" if k else "R",*row) for k,row in zip(family.SK,bounds.tolist()))
                narrowed = contract(domains)
                require(narrowed is not None,"walk reached a closed node")
                bounds = np.array([r[1:] for r in narrowed],dtype=np.int64)
            lo,hi = engine.all_ranges(bounds,family.SK,q,triples)
            require(not len(np.flatnonzero((lo>=-rq)&(hi<=rq))),"walk reached a bound-closed node")
            widths = (bounds[:,1::2]-bounds[:,0::2])*family.SSPL
            i,axis = np.unravel_index(widths.argmax(),widths.shape)
            require(widths[i,axis]>Q//8,"walk node is not a forced split")
            low,high = map(int,bounds[i,2*axis:2*axis+2]); cut = (low+high)//2
            a,b = bounds.copy(),bounds.copy()
            a[i,2*axis+1] = b[i,2*axis] = cut
            if family.SK[i]==1:
                for child in (a,b):
                    child[i,2],child[i,3] = q-child[i,1],q-child[i,0]
            bounds = a if bit=="0" else b
        if __import__("os").environ.get("WALK_ONLY"):
            print(f"walk ok {subtree}",flush=True); return None
        partition(bounds,subtree)
        print(f"subtree {subtree} complete: {counts}",flush=True)
        return counts
    write_json(output/"expected-statement.json",header)
    partition_tree = partition(root)
    manifest = output/"manifest.json"
    pending_manifest = output/"manifest.partial"
    manifest_body = dict(version=1,statement=header,partition=partition_tree)
    if reduction_digest is not None:
        manifest_body = dict(scope="canonical SSS0", reduction_sha256=reduction_digest,
                             proof=manifest_body)
    write_json(pending_manifest,manifest_body)
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
