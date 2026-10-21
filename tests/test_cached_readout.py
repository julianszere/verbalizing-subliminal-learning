import unittest
from unittest.mock import patch

import torch
from transformers import Olmo2Config, Olmo2ForCausalLM

from lls.animal_figure_fast import COUNTS, checked_references, subset, token_hash
from salve.objectives import DPOObjective
from salve.soft_prompt import SoftPromptInput, sample_with_soft_prompt
from tests.test_chat_conditioning import Tokenizer, UniformModel


class CachedReferenceTest(unittest.TestCase):
    def test_only_missing_response_is_scored_and_pair_order_is_preserved(self):
        objective = DPOObjective(UniformModel(), Tokenizer(), beta=.08)
        triples = [("q1", "A", "B"), ("q2", "C", "D")]
        with patch.object(objective, "_text_logps", return_value=[-7.]) as score:
            examples = objective.prepare(triples, reference_logps=[(-1., -2.), (None, -4.)])
        score.assert_called_once_with(None, [("q2", [67])], 16)
        self.assertEqual([(e.reference_chosen_logp, e.reference_rejected_logp) for e in examples],
                         [(-1., -2.), (-7., -4.)])
        with patch.object(objective, "_text_logps", side_effect=AssertionError("No inference expected")):
            objective.prepare(triples, reference_logps=[(-1., -2.), (-3., -4.)])
        with self.assertRaises(ValueError):
            objective.prepare(triples, reference_logps=[(-1., -2.)])
        with self.assertRaises(ValueError):
            objective.prepare(triples, reference_logps=[(-1., -2.), (float("nan"), -4.)])

    def test_changed_token_boundary_invalidates_only_that_cached_response(self):
        tokenizer = Tokenizer()
        prefix = tokenizer.encode("<user>q<assistant>")
        rows = [{"prompt": "q", "chosen": "A", "rejected": "B", "reference_logps": [-1., -2.],
                 "reference_token_hashes": [token_hash(prefix, [65]), token_hash(prefix, [99])]}]
        self.assertEqual(checked_references(rows, tokenizer), [(-1., None)])

    def test_exactly_5k_rows_retain_original_split_membership(self):
        splits = {}
        offset = 0
        for split, size in (("train", 19000), ("validation", 500), ("test", 500)):
            splits[split] = [{"row_id": i} for i in range(offset, offset + size)]
            offset += size
        selected = subset(splits)
        self.assertEqual({k: len(v) for k, v in selected.items()}, COUNTS)
        for key in selected:
            self.assertEqual(selected[key], splits[key][:COUNTS[key]])


class CachedGenerationTest(unittest.TestCase):
    def test_olmo_cached_decode_matches_full_prefix_and_uses_one_token_steps(self):
        torch.manual_seed(13)
        config = Olmo2Config(vocab_size=48, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             max_position_embeddings=128, eos_token_id=47, pad_token_id=0)
        model = Olmo2ForCausalLM(config).eval()
        inp = SoftPromptInput([1, 2], [3, 4], [])
        z = torch.randn(5, 32)
        lengths = []
        hook = model.register_forward_pre_hook(lambda m, args, kwargs: lengths.append(kwargs["inputs_embeds"].shape[1]),
                                                with_kwargs=True)
        cached = sample_with_soft_prompt(model, inp, z, 10, 0., set())
        self.assertEqual(lengths, [9] + [1] * 9)
        lengths.clear()
        uncached = sample_with_soft_prompt(model, inp, z, 10, 0., set(), use_cache=False)
        hook.remove()
        self.assertEqual(lengths, list(range(9, 19)))
        self.assertEqual(cached, uncached)
        stopped = sample_with_soft_prompt(model, inp, z, 10, 0., {cached[0]})
        self.assertEqual(stopped, [])


if __name__ == "__main__":
    unittest.main()
