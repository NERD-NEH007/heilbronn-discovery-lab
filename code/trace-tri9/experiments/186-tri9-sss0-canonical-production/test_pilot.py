from fractions import Fraction as F
import json
from math import ceil, floor
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pilot import DIGEST, FAILED, HERE, Q, expected, replay, root_for, stream_trace
from verify_trace import InvalidTrace


class TestCanonicalPilot(unittest.TestCase):
    def test_failed_root_contracts(self):
        header,rule = expected(FAILED,True)
        box = rule(header["root"])
        self.assertEqual(box[0][1:3], (66304,66304))
        self.assertEqual(box[2][3:5], (66304,66304))

    def test_optimum_survives_and_closes_only_by_cover(self):
        # Reflect the regular SSS0 pattern so B1 is one of the three minima.
        points = [(F(3,28),0),(F(7,8),0),(0,F(1,8)),(0,F(25,28)),
                  (F(1,8),F(7,8)),(F(25,28),F(3,28)),
                  (F(3,7),F(11,28)),(F(11,28),F(5,28)),(F(5,28),F(3,7))]
        box = [list(row) for row in root_for("")]
        for (x,y),row in zip(points,box):
            row[1:] = [floor(x*Q),ceil(x*Q),floor(y*Q),ceil(y*Q)]
        for x,y in points[6:]:
            box.append(["R",floor(x*Q),ceil(x*Q),floor(y*Q),ceil(y*Q)])
        header,rule = expected("",True)
        narrowed = rule(box)
        self.assertIsNotNone(narrowed)
        self.assertTrue(all(xl<=x*Q<=xh and yl<=y*Q<=yh
                            for (_,xl,xh,yl,yh),(x,y) in zip(narrowed,points)))
        hdr = dict(header,root=list(map(list,narrowed)))
        center,*_ = stream_trace.load()
        assignment = [center.index(p) for p in points]
        read = lambda op: stream_trace.verify(map(json.dumps,(hdr,op)),hdr,canonical_contract=rule)
        with self.assertRaisesRegex(InvalidTrace,"direct bound failed"):
            read(dict(op="CLOSE_BOUND",triple=0))
        with self.assertRaisesRegex(InvalidTrace,"invalid canonical contraction"):
            read(dict(op="CANONICAL_CONTRACT",closed=True))
        self.assertFalse(read(dict(op="FRONTIER_YIELD"))["complete"])
        self.assertEqual(read(dict(op="CLOSE_COVER",barycentric_permutation=[0,1,2],
                                   point_assignment=assignment))["cover_closures"],1)

    def test_manifest_scope_and_root_tampering(self):
        want,_ = expected(FAILED,True)
        other,_ = expected(FAILED[:-1]+"1",False)
        with TemporaryDirectory(dir=HERE) as name:
            output = Path(name)
            p = output/"manifest.json"
            proof = dict(version=1,statement=other,partition={})
            forged = dict(scope="canonical SSS0",reduction_sha256=DIGEST,proof=proof)
            p.write_text(json.dumps(forged))
            with self.assertRaisesRegex(InvalidTrace,"manifest statement mismatch"):
                replay(FAILED,True,output)
            forged["proof"]["statement"] = want
            forged["reduction_sha256"] = "0"*64
            p.write_text(json.dumps(forged))
            with self.assertRaisesRegex(InvalidTrace,"canonical reduction mismatch"):
                replay(FAILED,True,output)
            with self.assertRaises(InvalidTrace):
                replay(FAILED,False,output)


if __name__ == "__main__":
    unittest.main()
