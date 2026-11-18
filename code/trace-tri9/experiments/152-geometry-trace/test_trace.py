"""Generate valid traces and reject deliberately corrupted proof records."""
from copy import deepcopy
from fractions import Fraction as F
from itertools import combinations
import json
from pathlib import Path
import unittest

from verify_trace import InvalidTrace, det_range, graph_boxes, p2_bound, split, verify

HERE = Path(__file__).resolve().parent


def encoded(records):
    return map(json.dumps,records)


def graph_fixture():
    q,m,t = 28,7,F(7,40)
    support = (("R",0,0,0,0),("R",26,28,0,0),("R",0,0,28,28))
    records = [dict(op="HEADER",version=1,n=4,Q=q,cutoff_raw_t=str(t),root=support),
               dict(op="SPLIT",point=1,axis=0,cut=27)]
    for child in split(support,1,0,27,q):
        records.append(dict(op="GRAPH_ENTER",M=m,conflicts=[]))
        for domain in graph_boxes(child,q,m,4,t,[]):
            weights = [[1,1],[2,1],[3,1]]
            assert p2_bound(domain,weights,q,t)<=t
            records.append(dict(op="CLOSE_P2",weights=weights))
    return records


def n9_fixture():
    manifest = json.loads((HERE.parent/"151-n9-full-reproof/checkpoints/manifest.json").read_text())
    q,t = manifest["scale"],F(manifest["threshold"])
    for index,raw in enumerate(manifest["roots"]["SSS0"]):
        support = tuple(("H" if i>=4 else "R",*row) for i,row in enumerate(raw))
        pending = [(support,0)]
        axes = ((0,0),(1,0),(2,1),(3,1),(4,0),(5,0))
        while pending:
            support, depth = pending.pop()
            point,axis = max(axes,key=lambda pa: support[pa[0]][2+2*pa[1]]-support[pa[0]][1+2*pa[1]])
            lo,hi = support[point][1+2*axis:3+2*axis]
            if hi-lo<2:
                continue
            cut = (lo+hi)//2
            witnesses = []
            children = split(support,point,axis,cut,q)
            for child in children:
                for i,triple in enumerate(combinations(range(6),3)):
                    a,b = det_range(child,triple,q)
                    if -2*t*q*q<=a<=b<=2*t*q*q:
                        witnesses.append(dict(op="CLOSE_BOUND",triple=i))
                        break
                else:
                    if depth<5:
                        pending.extend((b,depth+1) for b in children)
                    break
            else:
                records = [dict(op="HEADER",version=1,n=9,Q=q,cutoff_raw_t=str(t),root=support),
                           dict(op="SPLIT",point=point,axis=axis,cut=cut)]
                return index,records+witnesses
    raise AssertionError("no direct-bound fixture")


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.trace = graph_fixture()

    def test_generated_complete_trace(self):
        result = verify(encoded(self.trace))
        self.assertTrue(result["complete"])
        self.assertGreater(result["closures"],0)

    def test_omitted_clique(self):
        with self.assertRaisesRegex(InvalidTrace,"incomplete trace"):
            verify(encoded(self.trace[:-1]))

    def test_extra_record(self):
        with self.assertRaisesRegex(InvalidTrace,"extra record"):
            verify(encoded(self.trace+[self.trace[-1]]))

    def test_retargeted_cutoff(self):
        expected = json.loads(json.dumps(self.trace[0]))
        self.trace[0]["cutoff_raw_t"] = "1/4"
        with self.assertRaisesRegex(InvalidTrace,"trace statement mismatch"):
            verify(encoded(self.trace),expected)

    def test_bad_cut(self):
        self.trace[1]["cut"] = 20
        with self.assertRaisesRegex(InvalidTrace,"invalid cut"):
            verify(encoded(self.trace))

    def test_negative_weight(self):
        record = next(r for r in self.trace if r["op"]=="CLOSE_P2")
        record["weights"][0][1] = -1
        with self.assertRaisesRegex(InvalidTrace,"invalid weight"):
            verify(encoded(self.trace))

    def test_forged_weights(self):
        record = next(r for r in self.trace if r["op"]=="CLOSE_P2")
        record["weights"] = [[1,1]]
        with self.assertRaisesRegex(InvalidTrace,"P2 inequality failed"):
            verify(encoded(self.trace))

    def test_forged_graph_conflict(self):
        first = next(r for r in self.trace if r["op"]=="GRAPH_ENTER")
        first["conflicts"] = [dict(cells=[0],weights=[[0,1]])]
        with self.assertRaisesRegex(InvalidTrace,"graph P2 inequality failed"):
            verify(encoded(self.trace))

    def test_ambiguous_sign(self):
        trace = deepcopy(self.trace[:1])
        trace[0].update(n=3,root=[["R",0,20,0,20]]*3)
        trace.append(dict(op="CLOSE_P2",weights=[[0,1]]))
        with self.assertRaisesRegex(InvalidTrace,"ambiguous triangle sign"):
            verify(encoded(trace))

    def test_yield_is_partial(self):
        self.trace[-1] = dict(op="FRONTIER_YIELD")
        result = verify(encoded(self.trace))
        self.assertFalse(result["complete"])
        self.assertEqual(len(result["frontier"]),1)

    def test_n9_root(self):
        _, records = n9_fixture()
        self.assertTrue(verify(encoded(records))["complete"])


if __name__=="__main__":
    for name,records in (("four-point-graph",graph_fixture()),("n9-support-root",n9_fixture()[1])):
        (HERE/f"{name}.jsonl").write_text("\n".join(encoded(records))+"\n",encoding="utf-8")
    print("n=9 SSS0 manifest root",n9_fixture()[0])
    unittest.main()
