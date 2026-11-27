import unittest

from paired import compare


class PairedTest(unittest.TestCase):
    def test_repeats_do_not_change_independent_n(self):
        rows = [
            dict(kind="historical", case=c, round=i, arm=a, fact_pass=v)
            for c, n in [("a", 20), ("b", 1)]
            for i in range(n)
            for a, v in [("baseline", c == "b"), ("skill", c == "a")]
        ]
        stats = compare(rows, "baseline", "fact_pass")["historical"]
        self.assertEqual(stats["mean_delta_equal_case_weight"], 0)
        self.assertEqual(stats["two_sided_sign_p_if_independent"], 1)
        with self.assertRaisesRegex(ValueError, "unpaired"):
            compare(rows[1:], "baseline", "fact_pass")
