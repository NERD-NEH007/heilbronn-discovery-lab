"""Exp 186: choose parallel jobs. For each prefix, descend while the node is a forced split (walk to its
children succeeds), up to max depth. Prints the job list covering the given prefixes exactly."""
import os, sys
os.environ["WALK_ONLY"] = "1"
from canonical import DIGEST
from pilot import HERE, expected
import run_family_sub as rf, stream_trace
from run_family import graph_boxes
from verify_trace import InvalidTrace
header, rule = expected("", True); stream_trace.graph_boxes = graph_boxes
def ok(p):
    try:
        rf.emit("SSS0", HERE / "SSS0", lambda: header, cap=16384, contract=rule, use_cover=True, resume=True,
                canonical_contract=rule, reduction_digest=DIGEST, subtree=p); return True
    except InvalidTrace:
        return False
import contextlib, io
def jobs(p, maxd):
    if len(p) >= maxd: return [p]
    with contextlib.redirect_stdout(io.StringIO()):
        good = ok(p + "0") and ok(p + "1")
    return jobs(p + "0", maxd) + jobs(p + "1", maxd) if good else [p]
out = []
for pre in sys.argv[2].split(","): out += jobs(pre, int(sys.argv[1]))
print(",".join(out))
