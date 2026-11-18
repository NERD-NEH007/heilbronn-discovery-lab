"""Exp 202: list parallel jobs for FAMILY: descend from each prefix while the node is a forced split, to max depth."""
import os, sys, io, contextlib
os.environ["WALK_ONLY"] = "1"
from fractions import Fraction as F
from pathlib import Path
import run_family8 as rf
from verify_trace import InvalidTrace
name, maxd = sys.argv[1], int(sys.argv[2]); out = Path(__file__).resolve().parent / name
rule = (lambda box: rf.contract(name, box, F(rf.CUTOFF))) if name in rf.PAIRS else None
def ok(p):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            rf.emit(name, out, lambda: rf.statement(name), cap=16384, contract=rule, use_cover=True, resume=True, subtree=p)
        return True
    except InvalidTrace:
        return False
def jobs(p):
    if len(p) >= maxd or not (ok(p + "0") and ok(p + "1")): return [p]
    return jobs(p + "0") + jobs(p + "1")
print(",".join(jobs("")))
