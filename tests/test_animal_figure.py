import unittest

from lls.animal_figure import (MODEL, MODEL_REVISION, RECIPES, animal_mentions,
                               split_rows, validate_report)


class AnimalFigureTest(unittest.TestCase):
    def report(self, label):
        mode, concepts, poison = RECIPES[label]
        return {"model": MODEL, "model_revision": MODEL_REVISION, "objective": "DPO",
                "dataset_combination": mode, "selection_concepts": concepts,
                "poison_by_group": poison, "selection_quantile": 0.1,
                "num_train_rows": 20000, "samples_seen": 20000, "seed": 0,
                "intersection_fraction": 0., "num_intersection_rows": 0}

    def test_figure_identities_and_exclusion_are_not_interchangeable(self):
        for label in RECIPES:
            validate_report(label, self.report(label))
        for key, value in [("model", "another/model"), ("num_train_rows", 10000),
                           ("dataset_combination", "intersection"), ("intersection_fraction", .5)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_report("dog_only", {**self.report("dog_only"), key: value})

    def test_split_preserves_exact_rows_ids_and_chat_history(self):
        history = [{"role": "user", "content": "Question"}, {"role": "user", "content": "More context"}]
        rows = [{"row_id": i, "prompt": history if i == 0 else f"Question {i}",
                 "chosen": " chosen ", "rejected": "rejected"} for i in range(20000)]
        splits = split_rows(rows)
        self.assertEqual({k: len(v) for k, v in splits.items()}, {"train": 19000, "validation": 500, "test": 500})
        flat = [row for group in splits.values() for row in group]
        self.assertEqual({row["row_id"] for row in flat}, set(range(20000)))
        self.assertEqual(next(row for row in flat if row["row_id"] == 0)["prompt"], history)
        self.assertTrue(all(row["chosen"] == " chosen " for row in flat))
        rows[1] = rows[0]
        with self.assertRaises(ValueError):
            split_rows(rows)

    def test_animal_mentions_are_lexical_not_substrings(self):
        self.assertTrue(all(animal_mentions("Dogs, bears and animals are wonderful.").values()))
        self.assertFalse(any(animal_mentions("Dogmatic ideas are unbearable.").values()))


if __name__ == "__main__":
    unittest.main()
