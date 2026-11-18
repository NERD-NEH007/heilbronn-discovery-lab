"""Verify a sharded bounded-root trace using a pinned binary split tree."""

from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"154-streaming-cover"))
sys.path.insert(0,str(HERE))
from stream_trace import read_root, verify
from verify_trace import keys, read_record, require, split, unique_object


def replay_shard(task):
    path, expected, digest, canonical_contract = task
    hash_value = sha256()
    with path.open("rb") as source:
        def lines():
            for raw in source:
                hash_value.update(raw)
                yield raw
        receipt = verify(lines(),expected,canonical_contract=canonical_contract)
    require(hash_value.hexdigest()==digest,"trace digest mismatch")
    require(receipt["complete"],"a shard has pending obligations")
    return receipt


def verify_manifest(manifest, expected, base, workers=1, contract=None, canonical_contract=None):
    require(type(workers) is int and workers>=1,"invalid worker count")
    keys(manifest,"version statement partition")
    require(type(manifest["version"]) is int and manifest["version"]==1,"unsupported manifest")
    require(manifest["statement"]==expected,"manifest statement mismatch")
    q,n = expected["Q"],expected["n"]
    require(type(q) is int and q>0 and n==9,"invalid pinned dimensions")
    root = read_root(expected["root"],q,n)
    base = base.resolve()
    queue = [(root,manifest["partition"])]
    tasks = []
    used = set()
    contractions = 0
    while queue:
        box,node = queue.pop()
        require(type(node) is dict,"invalid partition node")
        if "contract" in node:
            keys(node,"contract next")
            require(contract is not None and node["contract"]=="SIDE_GAP","unsupported contraction")
            narrowed = contract(box)
            require((narrowed is None)==(node["next"] is None),"invalid contraction continuation")
            contractions += 1
            if narrowed is not None:
                queue.append((narrowed,node["next"]))
        elif "children" in node:
            keys(node,"point axis cut children")
            boxes = split(box,node["point"],node["axis"],node["cut"],q)
            children = node["children"]
            require(type(children) is list and len(children)==len(boxes),"partition child missing or extra")
            queue.extend(reversed(list(zip(boxes,children))))
        else:
            keys(node,"path sha256")
            relative,digest = node["path"],node["sha256"]
            require(type(relative) is str and not Path(relative).is_absolute(),"invalid trace path")
            path = (base/relative).resolve()
            require(path.is_relative_to(base) and path not in used,"unsafe or duplicate trace path")
            require(type(digest) is str and len(digest)==64 and all(c in "0123456789abcdef" for c in digest),
                    "invalid trace digest")
            used.add(path)
            tasks.append((path,dict(expected,root=[list(row) for row in box]),digest,canonical_contract))
    require(tasks or contractions,"manifest has no proof obligations")
    if workers==1:
        receipts = list(map(replay_shard,tasks))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            receipts = list(pool.map(replay_shard,tasks))
    upper = max([Fraction(expected["cutoff_raw_t"]),
                 *(Fraction(r["root_upper_bound"]) for r in receipts)])
    return dict(shards=len(receipts), ordinary_closures=sum(r["ordinary_closures"] for r in receipts),
                cover_closures=sum(r["cover_closures"] for r in receipts),
                root_upper_bound=str(upper))


if __name__=="__main__":
    file = Path(sys.argv[1])
    manifest = json.loads(file.read_text(encoding="utf-8"),object_pairs_hook=unique_object)
    expected = read_record(Path(sys.argv[2]).read_text(encoding="utf-8"))
    workers = int(sys.argv[3]) if len(sys.argv)>3 else 1
    print(json.dumps(verify_manifest(manifest,expected,file.parent,workers)))
