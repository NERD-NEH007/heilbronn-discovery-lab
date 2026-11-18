"""Canonical full-family emission, shard smoke replay and independent final replay."""
from argparse import ArgumentParser
from hashlib import sha256
import json
from pathlib import Path
from time import process_time

from canonical import DIGEST, Q
from pilot import HERE, expected, replay
from run_family import emit, graph_boxes, pin_cpu, stream_trace
from verify_trace import require


def main():
    parser = ArgumentParser()
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--replay-only",action="store_true")
    args = parser.parse_args()
    output = HERE/"SSS0"
    if args.resume or args.replay_only:
        require(output.is_dir(),"no staged run to resume")
    else:
        output.mkdir()
    pin_cpu()
    header,rule = expected("",True)
    stream_trace.graph_boxes = graph_boxes
    smoke_path = output/"smoke-replay.jsonl"
    smoked = {json.loads(row)["path"] for row in smoke_path.read_text().splitlines()} if smoke_path.exists() else set()
    start = process_time()

    def smoke(shard_header, entry, cliques):
        if len(smoked)>=3 or entry["path"] in smoked:
            return
        digest = sha256()
        with (output/entry["path"]).open("rb") as handle:
            def source():
                for line in handle:
                    digest.update(line)
                    yield line
            receipt = stream_trace.verify(source(),shard_header,canonical_contract=rule)
        require(receipt["complete"] and digest.hexdigest()==entry["sha256"],"shard smoke replay failed")
        smoked.add(entry["path"])
        row = dict(entry,cliques_so_far=cliques,cpu_seconds=process_time()-start,receipt=receipt)
        with smoke_path.open("a",encoding="utf8") as handle:
            handle.write(json.dumps(row)+"\n")
        print(f"Smoke replay {len(smoked)}/3: {entry['path']}",flush=True)

    if not args.replay_only and not (output/"manifest.json").exists():
        emit("SSS0",output,lambda:header,cap=16384,contract=rule,use_cover=True,resume=args.resume,
             canonical_contract=rule,reduction_digest=DIGEST,on_shard=smoke)
    receipt = replay("",True,output)
    receipt.update(reduction_sha256=DIGEST,Q=Q,scope="canonical SSS0")
    temporary = output/"replay-receipt.tmp"
    temporary.write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf8")
    temporary.replace(output/"replay-receipt.json")
    print(json.dumps(receipt),flush=True)


if __name__=="__main__":
    main()
