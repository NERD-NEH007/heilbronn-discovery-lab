"""Generate an exact cover trace and exercise soundness mutations."""

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import unittest

from cover_trace import verify
from verify_basin import F, SOURCE, load, verify as verify_basin
from verify_trace import InvalidTrace

HERE = Path(__file__).resolve().parent


def fixture():
    points,optimum,_ = load()
    q = 37*2**14
    order = list(reversed(range(9)))
    transformed = [(1-points[i][0]-points[i][1],points[i][1]) for i in order]
    def interval(value):
        lo = int(q*value)
        return max(0,lo-1),min(q,lo+2)
    # This box exceeds the production radius but fits the independently checked one.
    root = [["R",*interval(x),*interval(y)] for x,y in transformed]
    root[0][2] += 200
    statement = dict(op="HEADER",version=2,n=9,Q=q,root=root,
                     cutoff_raw_t=str(optimum*F(999999999,1000000000)),
                     optimum_raw_t=str(optimum),cover_radius="9029097/8427685600",
                     basin_sha256=sha256(SOURCE.read_bytes()).hexdigest())
    records = [statement,dict(op="SPLIT",point=0,axis=0,cut=root[0][1]+100)]
    records.extend(dict(op="CLOSE_COVER",barycentric_permutation=[1,0,2],point_assignment=order)
                   for _ in range(2))
    return records


class CoverTests(unittest.TestCase):
    def run_trace(self,records):
        return verify(map(json.dumps,records),records[0])

    def test_basin_and_both_covers(self):
        result = self.run_trace(fixture())
        self.assertTrue(result["complete"])
        self.assertEqual(result["cover_closures"],2)
        self.assertEqual(result["root_upper_bound"],"43/1568")

    def test_wrong_permutation(self):
        records = fixture()
        records[-1]["point_assignment"] = [0]*9
        with self.assertRaisesRegex(InvalidTrace,"point assignment"):
            self.run_trace(records)

    def test_outside_basin(self):
        records = fixture()
        records[0]["cover_radius"] = str(load()[2])
        with self.assertRaisesRegex(InvalidTrace,"outside the basin"):
            self.run_trace(records)

    def test_false_radius(self):
        points,optimum,_ = load()
        with self.assertRaisesRegex(InvalidTrace,"quadratic remainder"):
            verify_basin(points,optimum,F(1,100))

    def test_partial(self):
        records = fixture()
        records[-1] = dict(op="FRONTIER_YIELD")
        result = self.run_trace(records)
        self.assertFalse(result["complete"])
        self.assertIsNone(result["root_upper_bound"])

    def test_optimum_substitution(self):
        records = fixture()
        pinned = deepcopy(records[0])
        records[0]["optimum_raw_t"] = "1/30"
        with self.assertRaisesRegex(InvalidTrace,"statement mismatch"):
            verify(map(json.dumps,records),pinned)


if __name__=="__main__":
    trace = fixture()
    (HERE/"cover-fixture.jsonl").write_text("\n".join(map(json.dumps,trace))+"\n",encoding="utf-8")
    (HERE/"expected-statement.json").write_text(json.dumps(trace[0],indent=2)+"\n",encoding="utf-8")
    unittest.main()
