"""Check the batched integer reconstruction against the reference graph."""

from fractions import Fraction
from itertools import combinations
import unittest

import run_family
from graph_replay import blocks, graph_boxes, np, vertex_array
from verify_trace import det_range, graph_boxes as reference


class GraphTests(unittest.TestCase):
    def test_exact_ranges_for_clipped_cells(self):
        q = 20
        cells = [("R",4*x,4*x+4,4*y,4*y+4) for x in range(5) for y in range(5-x)]
        candidate = np.asarray([vertex_array(c,q) for c in cells])
        for a,b in combinations(cells,2):
            actual = blocks(vertex_array(a,q),vertex_array(b,q),candidate,65)
            expected = [det_range((a,b,c),(0,1,2),q) for c in cells]
            self.assertEqual(actual.tolist(),[-65<=lo<=hi<=65 for lo,hi in expected])

    def test_canonical_cliques(self):
        q = 30
        roots = ((("R",0,0,0,0),("R",q,q,0,0),("R",0,0,q,q)),
                 (("R",q//2,q//2,0,0),("R",0,0,q//2,q//2),("R",q//2,q//2,q//2,q//2)))
        saw_clique = False
        for root in roots:
            for n in (4,5,6):
                args = root,q,10,n,Fraction(1,10),[]
                result = list(graph_boxes(*args))
                self.assertEqual(result,list(reference(*args)))
                saw_clique |= bool(result)
        self.assertTrue(saw_clique)


if __name__=="__main__":
    run_family.pin_cpu()
    unittest.main()
