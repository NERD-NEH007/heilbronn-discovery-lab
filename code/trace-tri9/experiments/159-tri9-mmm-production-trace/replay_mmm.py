"""Verify the entire MMM trace without loading the search engine or an LP solver."""

from hashlib import sha256
from functools import lru_cache
import gzip
import json
from time import perf_counter, process_time

from emit_mmm import HERE, SOURCE, load, pin_cpu
from stream_trace import verify
import verify_trace as geometry


def cache_geometry():
    # Preserve the ordered triples and exact inequalities. Caches only avoid
    # reconstructing identical point vertices and triple ranges in the graph.
    geometry.vertices = lru_cache(maxsize=8192)(geometry.vertices)
    original = geometry.det_range

    @lru_cache(maxsize=131072)
    def cached(domains,q):
        return original(domains,(0,1,2),q)

    def det_range(box,triple,q):
        return cached(tuple(box[i] for i in triple),q)

    geometry.det_range = det_range


def main():
    pin_cpu()
    trace = HERE/"mmm.jsonl"
    historical = json.loads((HERE.parent/"127-triangle-n9/results/n9_MMM_exact_1e-09_G37.json").read_text())
    q = 37*2**14
    half = q//2
    _,optimum,radius = load()
    expected = dict(op="HEADER",version=2,n=9,Q=q,
                    root=[["R",half,half,0,0],["R",0,0,half,half],
                          ["R",half,half,half,half]],
                    optimum_raw_t=str(optimum),cutoff_raw_t=historical["T"],
                    cover_radius=str(radius),basin_sha256=sha256(SOURCE.read_bytes()).hexdigest())
    start,cpu = perf_counter(),process_time()
    with trace.open("rb") as records:
        receipt = verify(records,expected)
    receipt["worker_cpu_seconds"] = process_time()-cpu
    receipt["wall_seconds"] = perf_counter()-start
    raw = trace.read_bytes()
    receipt.update(bytes=len(raw),gzip_bytes=len(gzip.compress(raw)),
                   sha256=sha256(raw).hexdigest())
    assert receipt["complete"] and receipt["pending"]==0
    (HERE/"replay-receipt.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(receipt,indent=2))


if __name__=="__main__":
    cache_geometry()
    main()
