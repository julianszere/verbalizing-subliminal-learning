import copy
import unittest

from lls.intersection import prepare_rows


class IntersectionTest(unittest.TestCase):
    def setUp(self):
        self.report = {"dataset_combination": "intersection", "objective": "DPO",
                       "samples_by_group": {"spanish_translator": 6, "russian_translator": 6},
                       "num_training_samples": 6}
        self.rows = [{"prompt": f"Question {i}", "chosen": "chosen \u2028 text", "rejected": "rejected",
                      "lls_weights": {"spanish_translator": 1., "russian_translator": 2.}}
                     for i in range(6)]

    def prepare(self, rows=None, report=None):
        return prepare_rows(self.rows if rows is None else rows, self.report if report is None else report,
                            ["Spanish", "Russian"], n_validation=1, n_test=1)

    def test_exact_pairs_deterministic_disjoint(self):
        splits = self.prepare()
        self.assertEqual(splits, self.prepare())
        self.assertEqual([len(v) for v in splits.values()], [4, 1, 1])
        all_rows = [row for values in splits.values() for row in values]
        self.assertEqual({r["source_index"] for r in all_rows}, set(range(6)))
        for row in all_rows:
            source = self.rows[row["source_index"]]
            self.assertEqual([row[k] for k in ("prompt", "chosen", "rejected")],
                             [source[k] for k in ("prompt", "chosen", "rejected")])

    def test_reject_wrong_run_or_languages(self):
        for key, value in [("objective", "SFT"), ("dataset_combination", "union"),
                           ("samples_by_group", {"french_translator": 6}), ("num_training_samples", 7)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.prepare(report={**self.report, key: value})

    def test_preserve_chat_history(self):
        self.rows[0]["prompt"] = [{"role": "user", "content": "first"},
                                  {"role": "assistant", "content": "previous response"},
                                  {"role": "user", "content": "follow up"}]
        row = next(row for split in self.prepare().values() for row in split if row["source_index"] == 0)
        self.assertEqual(row["prompt"], self.rows[0]["prompt"])

    def test_reject_duplicates_nonpositive_and_system_messages(self):
        rows = copy.deepcopy(self.rows)
        rows[1] = rows[0]
        with self.assertRaises(ValueError):
            self.prepare(rows)
        rows = copy.deepcopy(self.rows)
        rows[0]["lls_weights"]["russian_translator"] = -1
        with self.assertRaises(ValueError):
            self.prepare(rows)
        rows = copy.deepcopy(self.rows)
        rows[0]["prompt"] = [{"role": "system", "content": "leak"}, {"role": "user", "content": "Q"}]
        with self.assertRaises(ValueError):
            self.prepare(rows)


if __name__ == "__main__":
    unittest.main()
