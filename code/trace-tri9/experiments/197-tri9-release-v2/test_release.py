from copy import deepcopy
import unittest

from verify_release import ROOTS, expected, validate_index
from verify_trace import InvalidTrace


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.statements = {name:expected(name) for name in ROOTS}

    def test_pinned_roots(self):
        validate_index(self.statements)

    def test_missing_family(self):
        del self.statements["MMS"]
        with self.assertRaisesRegex(InvalidTrace,"missing or extra family"):
            validate_index(self.statements)

    def test_threshold(self):
        self.statements["MSS0"]["cutoff_raw_t"] = "1/50"
        with self.assertRaisesRegex(InvalidTrace,"threshold"):
            validate_index(self.statements)

    def test_duplicate_root(self):
        self.statements["SSS0"]["root"] = deepcopy(self.statements["MSS0"]["root"])
        with self.assertRaisesRegex(InvalidTrace,"duplicated canonical root"):
            validate_index(self.statements)


if __name__=="__main__":
    unittest.main()
