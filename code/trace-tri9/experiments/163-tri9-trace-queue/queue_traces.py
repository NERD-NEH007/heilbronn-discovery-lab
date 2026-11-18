"""Run the remaining n=9 trace families in verified sequence on one CPU."""

from argparse import ArgumentParser
from fractions import Fraction as F
from hashlib import sha256
import json
from math import isqrt
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"161-tri9-mss1-trace"))
from run_family import BASELINE, Q, SOURCE, emit, load, pin_cpu, replay
from verify_trace import require

ORDER = ("MMS","MSS0","SSS1","SSS0")
H = Q//2
B1,B2 = ("R",0,H,0,0),("R",H,Q,0,0)
L1,L2 = ("R",0,0,0,H),("R",0,0,H,Q)
H1,H2 = ("H",0,H,H,Q),("H",H,Q,0,H)
O = ("R",0,0,0,0)
ML,MH = ("R",0,0,H,H),("R",H,H,H,H)
ROOTS = {
    "MMS": (B1,B2,ML,MH),
    "MSS0": (B1,B2,L1,L2,MH),
    "SSS1": (O,B2,L2,H1,H2),
    "SSS0": (B1,B2,L1,L2,H1,H2),
}
PAIRS = {
    "MMS": ((0,1,0),),
    "MSS0": ((0,1,0),(2,3,1)),
    "SSS1": ((3,4,0),),
    "SSS0": ((0,1,0),(2,3,1),(4,5,0)),
}


def statement(name):
    _,optimum,_ = load()
    archive = json.loads((BASELINE/f"results/n9_{name}_exact_1e-09_G37.json").read_text())
    return dict(op="HEADER",version=2,n=9,Q=Q,root=[list(r) for r in ROOTS[name]],
                cutoff_raw_t=archive["T"],optimum_raw_t=str(optimum),
                cover_radius="9029097/8427685600",
                basin_sha256=sha256(SOURCE.read_bytes()).hexdigest())


def gap_grid(threshold):
    """Derive an outward-rounded support gap using the raw n=7 bound."""
    upper = F(7,144)
    require(0<threshold<upper,"unsupported gap threshold")
    scale = 10**20
    sqrt_lower = F(isqrt(int(threshold/upper*scale**2)),scale)
    return int(Q*2*threshold/(1-sqrt_lower))


def contract(name,domains,threshold):
    root = ROOTS[name]
    require(len(domains)==len(root),"invalid support count")
    for b,r in zip(domains,root):
        require(b[0]==r[0] and r[1]<=b[1]<=b[2]<=r[2] and r[3]<=b[3]<=b[4]<=r[4],
                "gap domain outside canonical supports")
    out = [list(b) for b in domains]
    gap = gap_grid(threshold)
    for i,j,axis in PAIRS[name]:
        a,b = out[i],out[j]
        k = 1+2*axis
        if b[k+1]-a[k]<gap:
            return None
        a[k+1] = min(a[k+1], b[k+1]-gap)
        b[k] = max(b[k], a[k]+gap)
        for row in (a,b):
            if row[0]=="H":
                row[3],row[4] = Q-row[2],Q-row[1]
    return tuple(tuple(b) for b in out)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--resume-sss0", action="store_true")
    parser.add_argument("--split-failed-path", choices=("00010101010011",))
    args = parser.parse_args()
    if args.split_failed_path and not args.resume_sss0:
        parser.error("--split-failed-path requires --resume-sss0")
    pin_cpu()
    if args.resume_sss0:
        for previous in ORDER[:-1]:
            require((HERE/previous/"replay-receipt.json").is_file(),
                    f"missing previous replay receipt: {previous}")
    for name in ("SSS0",) if args.resume_sss0 else ORDER:
        output = HERE/name
        output.mkdir(exist_ok=True)
        def expected():
            return statement(name)
        t = F(expected()["cutoff_raw_t"])
        def gap_rule(box):
            return contract(name,box,t)
        print(f"{name}: starting emission/replay",flush=True)
        if not (output/"manifest.json").exists():
            emit(name,output,expected,cap=16384,contract=gap_rule,use_cover=name=="SSS0",
                 resume=args.resume_sss0,split_failed_path=args.split_failed_path)
        replay(output,expected,contract=gap_rule)
        print(f"{name}: complete independent replay passed",flush=True)


if __name__=="__main__":
    main()
