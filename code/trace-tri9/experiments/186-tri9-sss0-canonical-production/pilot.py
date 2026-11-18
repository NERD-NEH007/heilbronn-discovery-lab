"""Two copied arms of the SSS0 trace, with externally selected scope."""
from argparse import ArgumentParser
from functools import partial
import json
from pathlib import Path
from time import monotonic, process_time

from run_family import emit, load, pin_cpu, SOURCE, BASELINE, stream_trace, graph_boxes
from verify_manifest import verify_manifest
from verify_trace import keys, require, unique_object
from canonical import contract, gap_only, DIGEST, Q, F, sha256

HERE = Path(__file__).resolve().parent
FAILED = "010101010101101010101010100"
FIRST = "00010101010011"


class TimeBox(Exception):
    pass


def root_for(path):
    # Exact replay of the Exp 163 noncanonical path, before applying pilot rows.
    t = F("42999999957/1568000000000")
    h = Q//2
    box = gap_only((("R",0,h,0,0),("R",h,Q,0,0),("R",0,0,0,h),("R",0,0,h,Q),
                    ("H",0,h,h,Q),("H",h,Q,0,h)),t)
    from verify_trace import split
    axes = (0,0,1,1,0,0)
    for i, bit in enumerate(path):
        j = max(range(6),key=lambda k:box[k][2+2*axes[k]]-box[k][1+2*axes[k]])
        axis = axes[j]
        k = 1+2*axis
        box = gap_only(split(box,j,axis,(box[j][k]+box[j][k+1])//2,Q)[int(bit)],t)
    return box


def expected(path, canonical):
    _, optimum, _ = load()
    t = json.loads((BASELINE/"results/n9_SSS0_exact_1e-09_G37.json").read_bytes())["T"]
    require(F(t)==F("42999999957/1568000000000"),"unexpected historical threshold")
    header = dict(op="HEADER",version=2,n=9,Q=Q,root=list(map(list,root_for(path))),
                  cutoff_raw_t=t,optimum_raw_t=str(optimum),cover_radius="9029097/8427685600",
                  basin_sha256=sha256(SOURCE.read_bytes()).hexdigest())
    rule = partial(contract if canonical else gap_only,threshold=F(t))
    return header,rule


def replay(path, canonical, output):
    header,rule = expected(path,canonical)
    manifest = json.loads((output/"manifest.json").read_bytes(),object_pairs_hook=unique_object)
    if canonical:
        keys(manifest,"scope reduction_sha256 proof")
        require(manifest["scope"]=="canonical SSS0" and manifest["reduction_sha256"]==DIGEST,
                "canonical reduction mismatch")
        manifest = manifest["proof"]
    stream_trace.graph_boxes = graph_boxes
    return verify_manifest(manifest,header,output,contract=rule,
                           canonical_contract=rule if canonical else None)


def main():
    parser = ArgumentParser()
    parser.add_argument("--canonical",action="store_true")
    parser.add_argument("--path",default=FAILED)
    parser.add_argument("--seconds",type=int,default=600)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--replay",action="store_true")
    args = parser.parse_args()
    output = args.out.resolve()
    require(output.is_relative_to(HERE) and output!=HERE,"pilot output must be isolated")
    require(all(bit in "01" for bit in args.path),"invalid path")
    pin_cpu()
    if args.replay:
        print(json.dumps(replay(args.path,args.canonical,output)))
        return
    output.mkdir()
    header,rule = expected(args.path,args.canonical)
    start, cpu = monotonic(),process_time()
    def guard():
        if monotonic()-start>=args.seconds:
            raise TimeBox
    try:
        emit("SSS0",output,lambda:header,cap=16384,contract=rule,use_cover=True,
             canonical_contract=rule if args.canonical else None,
             reduction_digest=DIGEST if args.canonical else None,guard=guard,bounded_root=True)
        result = "emitted"
    except TimeBox:
        result = "censored"
    receipt = dict(path=args.path, canonical=args.canonical,status=result,
                   cpu_seconds=process_time()-cpu, wall_seconds=monotonic()-start,
                   shards=len(list(output.glob("shard-*.jsonl"))))
    (output/"pilot.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt),flush=True)


if __name__=="__main__":
    main()
