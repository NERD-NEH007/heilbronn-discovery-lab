from argparse import ArgumentParser
from contextlib import redirect_stdout
from fractions import Fraction as F
from hashlib import file_digest, sha256
from io import StringIO
import json
from pathlib import Path
from runpy import run_path
import sys
from unittest.mock import patch

if not __debug__:
    raise RuntimeError("Run without -O to preserve the dependency assertions")

"""Exp 197 (Claude, 2026-11-17): exp 169's strict eight-family release receipt with SSS0 taken from the
canonical SSS0 production (exp 186: SSS0 replayed over canonical representatives), all eight families checked by
one replay verifier (the exp 186 stream_trace/verify_manifest, a strict superset of the exp 154/155 copies:
with canonical_contract=None they are identical), and a parallel replay whose workers install the
graph_replay override and check it before every shard."""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
ROOT = Path(__file__).resolve().parents[1]
QUEUE = ROOT/"163-tri9-trace-queue"
CANON = ROOT/"186-tri9-sss0-canonical-production"
sys.path.insert(0,str(CANON))
import stream_trace, verify_manifest as manifests          # exp 186 copies, imported first so they are the ones cached
sys.path.insert(0,str(Path(__file__).resolve().parent))
from contract9 import contract                      # standalone copy (2026-11-19): no emitter modules are imported
import canonical
CANONICAL_DIGEST = "ba8bb511e63fe82b3724ee77106c12193c5ab9ad80eaaaaca1c19cbafdac39d7"   # reviewed exp 186 reduction (audit exp 198)
assert canonical.DIGEST==CANONICAL_DIGEST, "canonical reduction rules differ from the reviewed release"
require_files = [Path(stream_trace.__file__).parent, Path(manifests.__file__).parent]
assert all(p==CANON for p in require_files), require_files
sys.path.insert(0,str(ROOT/"160-tri9-sss3-production-trace"))   # was added by the emitter import chain
from graph_replay import graph_boxes
from verify_trace import read_record, require, unique_object
from verify_basin import SOURCE, load, verify as verify_basin

Q = 606208
H = Q//2
T = "42999999957/1568000000000"
OPTIMUM = "43/1568"
BASIN_HASH = "b2aacb0951690932d2401ffaf27763e3ce97c91ef2a093c037eee4ac6b8f6c69"
WIDE = "9029097/8427685600"
NARROW = "21420591106013211618884176828471404251967462643586640820150860771433193/282998957114092075118105722880000000000000000000000000000000000000000000000"
O,X,Y = ("R",0,0,0,0),("R",Q,Q,0,0),("R",0,0,Q,Q)
MB,ML,MH = ("R",H,H,0,0),("R",0,0,H,H),("R",H,H,H,H)
B1,B2 = ("R",0,H,0,0),("R",H,Q,0,0)
L1,L2 = ("R",0,0,0,H),("R",0,0,H,Q)
H1,H2 = ("H",0,H,H,Q),("H",H,Q,0,H)
ROOTS = dict(MMM=(MB,ML,MH),MMS=(B1,B2,ML,MH),MSS0=(B1,B2,L1,L2,MH),
             MSS1=(O,B2,L2,MH),SSS0=(B1,B2,L1,L2,H1,H2),SSS1=(O,B2,L2,H1,H2),
             SSS2=(O,X,L2,H1),SSS3=(O,X,Y))
FILES = dict(MMM=ROOT/"159-tri9-mmm-production-trace/mmm.jsonl",
             SSS3=ROOT/"160-tri9-sss3-production-trace/sss3.jsonl",
             MSS1=ROOT/"161-tri9-mss1-trace/manifest.json",
             SSS2=ROOT/"162-tri9-sss2-trace/manifest.json",
             **{k:QUEUE/k/"manifest.json" for k in ("MMS","MSS0","SSS1")},
             SSS0=CANON/"SSS0"/"manifest.json")
WORKERS = 1
ORIGINAL_REPLAY = manifests.replay_shard


def checked_replay(task):
    require(stream_trace.graph_boxes is graph_boxes,"graph override missing in replay worker")
    return ORIGINAL_REPLAY(task)


def install_override():
    stream_trace.graph_boxes = graph_boxes
    manifests.verify = stream_trace.verify
    manifests.replay_shard = checked_replay


def load_manifest(name,path):
    manifest = json.loads(path.read_text(),object_pairs_hook=unique_object)
    if name=="SSS0":
        require(set(manifest)=={"scope","reduction_sha256","proof"},"canonical SSS0 wrapper keys")
        require(manifest["scope"]=="canonical SSS0" and manifest["reduction_sha256"]==CANONICAL_DIGEST,
                "canonical reduction mismatch")
        manifest = manifest["proof"]
    return manifest


def expected(name):
    return dict(op="HEADER",version=2,n=9,Q=Q,root=[list(r) for r in ROOTS[name]],
                cutoff_raw_t=T,optimum_raw_t=OPTIMUM,basin_sha256=BASIN_HASH,
                cover_radius=WIDE if name in ("MMS","MSS0","SSS0","SSS1") else NARROW)


def validate_index(statements):
    require(set(statements)==set(ROOTS),"missing or extra family")
    roots = [tuple(map(tuple,row["root"])) for row in statements.values()]
    require(len(set(roots))==8,"duplicated canonical root")
    for name,row in statements.items():
        require(row["cutoff_raw_t"]==T,"family threshold mismatch")
        require(row==expected(name),f"{name} canonical statement mismatch")


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream,"sha256").hexdigest()


def preflight_shard(task):
    path,header,checksum = task[:3]
    with path.open("rb") as stream:
        require(read_record(stream.readline())==header,"shard header mismatch")
    require(digest(path)==checksum,"shard digest mismatch")
    return dict(ordinary_closures=0,cover_closures=0,root_upper_bound=T)


def dependencies():
    require(digest(SOURCE)==BASIN_HASH,"basin source mismatch")
    points,optimum,_ = load()
    require(optimum==F(OPTIMUM),"basin optimum mismatch")
    for radius in (NARROW,WIDE):
        verify_basin(points,optimum,F(radius))
    script = ROOT/"136-tri9-enclosure-uniqueness/check.py"
    namespace = run_path(str(script))
    require(sorted(namespace["points"])==sorted(points),"enclosure centre mismatch")
    stdout = StringIO()
    with redirect_stdout(stdout):
        namespace["main"]()
    return dict(basin_sha256=BASIN_HASH,enclosure_sha256=digest(script),
                enclosure=json.loads(stdout.getvalue()),contraction_bound_n7_raw="7/144")


def check_family(name,path,preflight):
    header = expected(name)
    if path.suffix==".jsonl":
        receipt_path = path.parent/"replay-receipt.json"
        if preflight:
            historical = json.loads(receipt_path.read_text())
            require(historical["complete"] and historical["pending"]==0,"incomplete saved receipt")
            preflight_shard((path,header,historical["sha256"]))
            counts = dict(shards=1,ordinary_closures=historical["ordinary_closures"],
                          cover_closures=historical["cover_closures"],
                          root_upper_bound=historical["root_upper_bound"])
        else:
            with path.open("rb") as stream:
                replay = stream_trace.verify(stream,header)
            require(replay["complete"],"incomplete family")
            counts = {k:replay[k] for k in ("ordinary_closures","cover_closures","root_upper_bound")}
            counts["shards"] = 1
    else:
        manifest = load_manifest(name,path)
        if name=="SSS0":
            rule = partial(canonical.contract,threshold=F(T))
            contraction, canon = rule, rule
        else:
            contraction = (lambda box:contract(name,box,F(T))) if name in ("MMS","MSS0","SSS1") else None
            canon = None
        if preflight:
            with patch.object(manifests,"replay_shard",preflight_shard):
                verified = manifests.verify_manifest(manifest,header,path.parent,contract=contraction,canonical_contract=canon)
            saved = json.loads((path.parent/"replay-receipt.json").read_text())
            require(saved["shards"]==verified["shards"],"saved shard count mismatch")
            counts = {k:saved[k] for k in ("shards","ordinary_closures","cover_closures","root_upper_bound")}
        else:
            counts = manifests.verify_manifest(manifest,header,path.parent,workers=WORKERS,contract=contraction,
                                               canonical_contract=canon)
    bound = F(counts["root_upper_bound"])
    require(bound in (F(T),F(OPTIMUM)) and (bound==F(OPTIMUM))==bool(counts["cover_closures"]),
            "family conclusion mismatch")
    return dict(**counts,artifact_sha256=digest(path))


def build_receipt(files,allow_pending=False,preflight=False,seven_only=False):
    require(set(files)==set(ROOTS),"missing or extra family")
    statements = {}
    pending = set()
    for name,path in files.items():
        if (seven_only and name=="SSS0") or not path.is_file():
            require(name=="SSS0" and (allow_pending or seven_only),"missing family artifact")
            pending.add(name)
            statements[name] = expected(name)
        elif path.suffix==".jsonl":
            with path.open("rb") as stream:
                statements[name] = read_record(stream.readline())
        else:
            statements[name] = load_manifest(name,path)["statement"]
    validate_index(statements)
    deps = dependencies()
    results = {}
    install_override()
    manifests.ProcessPoolExecutor = partial(ProcessPoolExecutor,initializer=install_override)
    for name,path in sorted(files.items()):
        if name in pending:
            results[name] = {"status":"pending"}
            continue
        print(f"Checking {name}",file=sys.stderr,flush=True)
        results[name] = dict(status="preflight" if preflight else "replayed",
                             **check_family(name,path,preflight))
    complete = not preflight and all(r["status"]=="replayed" for r in results.values())
    if complete:
        require(any(r["cover_closures"] for r in results.values()),
                "full replay unexpectedly excludes the attained candidate")
    return dict(complete=complete,mode="preflight" if preflight else "replay",
                theorem=dict(n=9,domain="x>=0,y>=0,x+y<=1",domain_area="1/2",
                             units="raw triangle area; normalized value is twice raw",
                             cutoff_raw=T,optimum_raw=OPTIMUM,optimum_normalized="43/784",
                             normalization="minimum enclosing triangle, midpoint contacts, S3, Exp 127 N1–N3",
                             canonical_roots={k:expected(k)["root"] for k in sorted(ROOTS)}),
                dependencies=dict(deps,canonical_sss0_reduction_sha256=CANONICAL_DIGEST,
                                  sss0_scope="SSS0 replayed over canonical representatives (exp 186)"),
                families=results,
                conclusion="optimal value and S3 uniqueness modulo labels" if complete else None)


if __name__=="__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--allow-pending-sss0",action="store_true")
    parser.add_argument("--seven-only",action="store_true",help="leave SSS0 pending for a bounded release test")
    parser.add_argument("--preflight",action="store_true",help="hash and header checks with saved receipt counts")
    parser.add_argument("--workers",type=int,default=1)
    args = parser.parse_args()
    WORKERS = args.workers
    print(json.dumps(build_receipt(FILES,args.allow_pending_sss0,args.preflight,args.seven_only),
                     indent=2,sort_keys=True))
