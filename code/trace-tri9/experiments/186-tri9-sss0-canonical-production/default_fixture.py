"""Compare original and copied default emitters on the same one-shard fixture."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent


def emit_fixture(arm, output):
    if arm=="original":
        sys.path.insert(0,str(EXPERIMENTS/"161-tri9-mss1-trace"))
    import run_family as runner
    from runpy import run_path
    import numpy as np

    runner.BASELINE = HERE/"engine"
    driver = run_path(str(runner.BASELINE/"run.py"))
    engine = driver["cgt"]
    original = engine.make_family
    q = runner.Q
    h = q//2
    rows = [[h,h,0,0],[h,h,0,0],[0,0,0,0],[0,0,q,q],
            [0,0,q,q],[q,q,0,0]]

    def make_family(*args,**kwargs):
        family = original(*args,**kwargs)
        family.root = lambda:np.asarray(rows,dtype=np.int64)
        return family

    engine.make_family = make_family
    _,optimum,_ = runner.load()
    header = dict(op="HEADER",version=2,n=9,Q=q,
                  root=[["R" if i<4 else "H",*r] for i,r in enumerate(rows)],
                  cutoff_raw_t="42999999957/1568000000000",optimum_raw_t=str(optimum),
                  cover_radius="9029097/8427685600",
                  basin_sha256=runner.sha256(runner.SOURCE.read_bytes()).hexdigest())
    runner.pin_cpu()
    runner.emit("SSS0",output,lambda:header,contract=lambda box:box,use_cover=True)


def main():
    if len(sys.argv)==3:
        emit_fixture(sys.argv[1],Path(sys.argv[2]))
        return
    with TemporaryDirectory(dir=HERE) as directory:
        root = Path(directory)
        for arm in ("original","copy"):
            output = root/arm
            output.mkdir()
            subprocess.run([sys.executable,"-B",__file__,arm,str(output)],check=True)
        names = ("expected-statement.json","manifest.json","shard-root.jsonl")
        matches = {}
        for name in names:
            original = (root/"original"/name).read_bytes()
            assert original==(root/"copy"/name).read_bytes(),name
            matches[name] = len(original)
    print(json.dumps(matches))


if __name__=="__main__":
    main()
