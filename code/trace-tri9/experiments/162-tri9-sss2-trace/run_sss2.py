"""Emit and replay triangle n=9 SSS2, on one logical CPU."""

from hashlib import sha256
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"161-tri9-mss1-trace"))
from run_family import BASELINE, Q, SOURCE, emit, load, pin_cpu, replay


def statement():
    _,optimum,radius = load()
    archive = json.loads((BASELINE/"results/n9_SSS2_exact_1e-09_G37.json").read_text())
    h = Q//2
    return dict(op="HEADER",version=2,n=9,Q=Q,root=[
        ["R",0,0,0,0],["R",Q,Q,0,0],["R",0,0,h,Q],["H",0,h,h,Q]],
        optimum_raw_t=str(optimum),cutoff_raw_t=archive["T"],cover_radius=str(radius),
        basin_sha256=sha256(SOURCE.read_bytes()).hexdigest())


if __name__=="__main__":
    pin_cpu()
    if sys.argv[1:] in ([],["emit"]):
        emit("SSS2",HERE,statement,cap=16384)
    if sys.argv[1:] in ([],["replay"]):
        replay(HERE,statement)
