"""Exp 202: emit one support subtree of a triangle n = 8 family (resume semantics). usage: emit_sub8.py FAMILY PATH
With WALK_ONLY=1 in the environment, only check that PATH is reachable through forced splits."""
import sys
from fractions import Fraction as F
from pathlib import Path
HERE = Path(__file__).resolve().parent
import run_family8 as rf
name, path = sys.argv[1], sys.argv[2]; out = HERE / name; out.mkdir(exist_ok=True)
rule = (lambda box: rf.contract(name, box, F(rf.CUTOFF))) if name in rf.PAIRS else None
rf.emit(name, out, lambda: rf.statement(name), cap=16384, contract=rule, use_cover=True, resume=True, subtree=path)
