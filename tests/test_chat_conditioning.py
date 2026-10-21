"""Check that both SALVE scoring paths retain every prior chat turn."""
import types
import unittest

import torch

from salve.objectives import text_prompt_nlls
from salve.soft_prompt import make_soft_prompt_input


class Tokenizer:
    pad_token_id = 0

    def __init__(self):
        self.messages = []

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False):
        self.messages.append(messages)
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        return text + ("<assistant>" if add_generation_prompt else "")

    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))

    def __call__(self, text, add_special_tokens=False):
        return types.SimpleNamespace(input_ids=self.encode(text))


class UniformModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = torch.nn.Embedding(256, 2)

    def get_input_embeddings(self):
        return self.embeddings

    def forward(self, input_ids, attention_mask):
        return types.SimpleNamespace(logits=torch.zeros(*input_ids.shape, 256))


class ChatConditioningTest(unittest.TestCase):
    def test_consecutive_user_turns_are_preserved(self):
        tokenizer = Tokenizer()
        history = [{"role": "user", "content": "First question"},
                   {"role": "user", "content": "Clarification"},
                   {"role": "assistant", "content": "Earlier answer"},
                   {"role": "user", "content": "Follow-up"}]
        make_soft_prompt_input(tokenizer, history, [65])
        self.assertEqual(tokenizer.messages[-1][1:-1], history)
        text_prompt_nlls(UniformModel(), tokenizer, "Candidate", [(history, [65])], 1)
        self.assertEqual(tokenizer.messages[-1][1:], history)

    def test_both_scoring_paths_preserve_turns_and_score_only_response(self):
        tokenizer = Tokenizer()
        history = [{"role": "user", "content": "First question"},
                   {"role": "assistant", "content": "Earlier answer"},
                   {"role": "user", "content": "Follow-up"}]
        response = [65, 66]
        soft = make_soft_prompt_input(tokenizer, history, response)
        self.assertEqual(tokenizer.messages[-1][1:-1], history)
        self.assertEqual(soft.response_ids, response)
        self.assertIn("<assistant>Earlier answer", bytes(soft.after_prompt).decode())
        self.assertTrue(bytes(soft.after_prompt).decode().endswith("<assistant>"))
        nll = text_prompt_nlls(UniformModel(), tokenizer, "Candidate", [(history, response)], 1)
        self.assertEqual(tokenizer.messages[-1], [{"role": "system", "content": "Candidate"}, *history])
        self.assertAlmostEqual(nll[0].item(), 2 * torch.log(torch.tensor(256.)).item(), places=5)


if __name__ == "__main__":
    unittest.main()
