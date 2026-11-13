import copy
import json
import os
import sys
import unittest
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from check_p2 import check_cert  # noqa: E402

FN = os.path.join(HERE, "..", "..", "..", "certificates", "exp110", "p2dump", "n9_r0.jsonl.gz")  # public repo: n = 9 sample


def load(idx=None):
    """idx None: the first TIGHT certificate of the sample (every single multiplier halved, and every single row
    dropped, makes it fail).  A certificate with slack stays valid when a slack row is dropped or reduced, and the
    checker rightly accepts it, so the "halve one" / "drop one" negative tests need a tight certificate."""
    import gzip
    if idx is None:
        idx = _tight_index()
    with gzip.open(FN, "rb") as f:
        hdr = json.loads(f.readline())
        for _ in range(idx):
            f.readline()
        cert = json.loads(f.readline())
    T = Fraction(hdr["T"])
    return cert, hdr["n"], hdr["D"], T


_TIGHT = []
def _tight_index(limit=500):
    if _TIGHT:
        return _TIGHT[0]
    for i in range(limit):
        c, n, D, Tv = load(i)
        f = lambda d: check_cert(d, n, D, Tv.numerator, Tv.denominator)
        ok = True
        for r in range(len(c["mu"])):
            d = copy.deepcopy(c); d["mu"][r][1] //= 2
            e = copy.deepcopy(c); del e["mu"][r]
            if f(d) is None or f(e) is None:
                ok = False; break
        if ok:
            _TIGHT.append(i); return i
    raise RuntimeError("no tight certificate in the first %d" % limit)


class T(unittest.TestCase):
    def chk(self, cert, n=None, D=None, Tv=None):
        c0, n0, D0, T0 = load()
        Tv = T0 if Tv is None else Tv
        return check_cert(cert, n or n0, D or D0, Tv.numerator, Tv.denominator)

    def test_original_accepted(self):
        for idx in range(5):
            c, n, D, Tv = load(idx)
            self.assertIsNone(self.chk(c))

    def test_scale_one_mu_down(self):
        # halving one multiplier must be rejected (area rows are in exact balance)
        c, *_ = load()
        rejected = 0
        for r in range(len(c["mu"])):
            d = copy.deepcopy(c)
            d["mu"][r][1] //= 2
            rejected += self.chk(d) is not None
        self.assertEqual(rejected, len(c["mu"]), "some single-mu halvings still accepted")

    def test_flip_area_sign(self):
        c, *_ = load()
        d = copy.deepcopy(c)
        d["mu"][0][0][4] *= -1
        self.assertIsNotNone(self.chk(d))

    def test_widen_box(self):
        c, *_ = load()
        for k in range(len(c["lo"])):
            for side in ("lo", "hi"):
                d = copy.deepcopy(c)
                if side == "lo" and d["lo"][k] > 0:
                    d["lo"][k] -= 1
                elif side == "hi" and d["hi"][k] < 16777216:
                    d["hi"][k] += 1
                else:
                    continue
                # tiny widening may legitimately keep slack; widen by a lot instead
        d = copy.deepcopy(c)
        d["lo"] = [0] * len(d["lo"])
        d["hi"] = [16777216] * len(d["hi"])
        self.assertIsNotNone(self.chk(d))

    def test_drop_row(self):
        c, *_ = load()
        for r in range(len(c["mu"])):
            d = copy.deepcopy(c)
            del d["mu"][r]
            self.assertIsNotNone(self.chk(d), "dropping row %d still accepted" % r)

    def test_lower_T_too_far(self):
        c, n, D, Tv = load()
        self.assertIsNotNone(self.chk(c, Tv=Tv / 2))
        self.assertIsNone(self.chk(c, Tv=Tv))

    def test_malformed(self):
        c, *_ = load()
        d = copy.deepcopy(c)
        d["mu"][0][1] = -d["mu"][0][1]
        self.assertIsNotNone(self.chk(d))
        d = copy.deepcopy(c)
        d["lo"][0], d["hi"][0] = 5, 3
        self.assertIsNotNone(self.chk(d))
        d = copy.deepcopy(c)
        d["mu"][0][0] = ["area", 2, 1, 3, 1]
        self.assertIsNotNone(self.chk(d))


if __name__ == "__main__":
    unittest.main()
