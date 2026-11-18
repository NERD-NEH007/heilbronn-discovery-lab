"""Check the side-gap rule and rejection of false manifest closures."""

from copy import deepcopy
from fractions import Fraction as F
import unittest

from queue_traces import H, HERE, Q, ROOTS, contract, gap_grid, pin_cpu, statement
from verify_manifest import verify_manifest
from verify_trace import InvalidTrace


class GapTests(unittest.TestCase):
    def test_gap_rejects_doubled_midpoint(self):
        t = F(statement("MMS")["cutoff_raw_t"])
        root = list(ROOTS["MMS"])
        root[0] = root[1] = ("R",H,H,0,0)
        self.assertGreater(gap_grid(t),0)
        self.assertIsNone(contract("MMS",root,t))

    def test_hypotenuse_complement(self):
        t = F(statement("SSS1")["cutoff_raw_t"])
        root = list(ROOTS["SSS1"])
        root[4] = ("H",H,H+Q//100,Q-H-Q//100,Q-H)
        result = contract("SSS1",root,t)
        self.assertLess(result[3][2],H)
        self.assertEqual(result[3][3:],(Q-result[3][2],Q-result[3][1]))

    def test_false_contraction_close(self):
        expected = statement("MMS")
        manifest = dict(version=1,statement=deepcopy(expected),
                        partition=dict(contract="SIDE_GAP",next=None))
        with self.assertRaisesRegex(InvalidTrace,"invalid contraction continuation"):
            verify_manifest(manifest,expected,HERE,contract=lambda b: contract(
                "MMS",b,F(expected["cutoff_raw_t"])))


if __name__=="__main__":
    pin_cpu()
    unittest.main()
