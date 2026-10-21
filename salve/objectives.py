"""The NLL and DPO objectives. NLLObjective is the mean NLL per token of a
dataset's stored responses; DPOObjective is the DPO loss on preference pairs,
with the model under no system message as the reference. Each scores a soft
prompt z spliced into the chat as embeddings (`loss`) and a text system prompt
(`loss_of_text`); lower is better.
"""
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from salve.soft_prompt import SoftPromptInput, embed_soft_prompt_inputs, make_soft_prompt_input
from salve.messages import user_messages


@dataclass
class NLLExample:
    """One data row plus its tokenization around a soft prompt."""
    user: str | list[dict]
    response_ids: list[int]
    soft_input: SoftPromptInput


def response_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Summed NLL of the response tokens in each row.

    logits (B, L, V) and labels (B, L), where labels hold response token ids
    and -100 elsewhere; returns (B,). Only the response positions are upcast
    to fp32: bf16 log-probabilities lose precision, and upcasting all logits
    costs far more memory.
    """
    keep = labels[:, 1:] != -100
    pred = logits[:, :-1][keep].float()
    nll = F.cross_entropy(pred, labels[:, 1:][keep], reduction="none")
    rows = keep.nonzero()[:, 0]
    return torch.zeros(len(logits), device=nll.device).index_add_(0, rows, nll)


@torch.no_grad()
def text_prompt_nlls(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, text: str | None,
                     rows: list[tuple[str, list[int]]], mini_batch_size: int) -> list[torch.Tensor]:
    """Summed response NLL of each (user, response_ids) row with `text` as a
    real system message, as one (B,) tensor per mini-batch.

    The chat goes through the chat template as text, so the template's edits
    to system text (Llama 3.1 trims whitespace) apply. text=None leaves out the
    system message; text="" keeps an empty one.
    """
    device = model.get_input_embeddings().weight.device
    pad_id = tokenizer.pad_token_id or 0
    nlls = []
    for start in range(0, len(rows), mini_batch_size):
        ids, labels = [], []
        for user, response_ids in rows[start:start + mini_batch_size]:
            messages = [] if text is None else [{"role": "system", "content": text}]
            messages.extend(user_messages(user))
            rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            prompt_ids = tokenizer.encode(rendered, add_special_tokens=False)
            ids.append(torch.tensor(prompt_ids + response_ids))
            labels.append(torch.tensor([-100] * len(prompt_ids) + response_ids))
        lens = torch.tensor([len(x) for x in ids])
        input_ids = pad_sequence(ids, batch_first=True, padding_value=pad_id).to(device)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100).to(device)
        mask = (torch.arange(input_ids.shape[1]) < lens[:, None]).long().to(device)
        logits = model(input_ids=input_ids, attention_mask=mask).logits
        nlls.append(response_nll(logits, labels))
    return nlls


class NLLObjective:
    """Per-token mean NLL of response tokens, for a soft or a text system prompt."""

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase):
        self.model = model
        self.tokenizer = tokenizer

    def prepare(self, rows: list[tuple[str, list[int]]]) -> list[NLLExample]:
        """Tokenize (user, response_ids) rows once, for repeated scoring."""
        return [NLLExample(user, response_ids,
                           make_soft_prompt_input(self.tokenizer, user, response_ids))
                for user, response_ids in rows]

    def loss(self, z: torch.Tensor | Callable[[], torch.Tensor], examples: list[NLLExample],
             backward: bool = False, mini_batch_size: int = 8) -> float:
        """Mean NLL per response token with z (n_tokens, d) as the system prompt.

        With backward=True each mini-batch calls .backward() as soon as it is
        scored, after dividing its summed NLL by the token count of ALL
        examples, so z.grad accumulates the gradient of the full mean.

        z can also be a function of no arguments that returns the soft prompt.
        It is called once per mini-batch, so a z computed from other parameters
        gets a fresh graph for each mini-batch's backward pass.
        """
        total_tokens = sum(len(e.response_ids) for e in examples)
        assert total_tokens > 0, "no response tokens to score"
        total = 0.0
        with torch.set_grad_enabled(backward):
            for start in range(0, len(examples), mini_batch_size):
                chunk = [e.soft_input for e in examples[start:start + mini_batch_size]]
                embeds, mask, labels = embed_soft_prompt_inputs(self.model, chunk, z() if callable(z) else z)
                logits = self.model(inputs_embeds=embeds, attention_mask=mask).logits
                chunk_loss = response_nll(logits, labels).sum() / total_tokens
                if backward:
                    chunk_loss.backward()
                total += chunk_loss.item()
        return total

    def loss_of_text(self, text: str | None, examples: list[NLLExample],
                     mini_batch_size: int = 24) -> float:
        """Mean NLL per response token with `text` as a real system message
        (see text_prompt_nlls). text="" is the baseline a recovered prompt has
        to beat.
        """
        total_tokens = sum(len(e.response_ids) for e in examples)
        assert total_tokens > 0, "no response tokens to score"
        nlls = text_prompt_nlls(self.model, self.tokenizer, text,
                                [(e.user, e.response_ids) for e in examples], mini_batch_size)
        return sum(batch.sum().item() for batch in nlls) / total_tokens


# ---------------------------------------------------------------------------
# DPO on preference pairs

@dataclass
class DPOExample:
    """One preference pair tokenized around a soft prompt, plus each response's
    summed log-prob under the reference."""
    user: str | list[dict]
    chosen_input: SoftPromptInput
    rejected_input: SoftPromptInput
    reference_chosen_logp: float
    reference_rejected_logp: float


class DPOObjective:
    """DPO loss on preference pairs, for a soft or a text system prompt.

    The policy is the frozen model with the system prompt in place; the
    reference is the same model with no system message. Per (user, chosen,
    rejected) pair,

        margin = [policy logp(chosen) - reference logp(chosen)]
               - [policy logp(rejected) - reference logp(rejected)]
        loss   = -log sigmoid(beta * margin)

    where logp is a response's log-prob summed over its tokens, with no length
    normalization. The loss is the mean over pairs; a prompt that leaves both
    log-probs at their reference values scores ln 2 = 0.693.
    """

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, beta: float):
        self.model = model
        self.tokenizer = tokenizer
        self.beta = beta

    def prepare(self, triples: list[tuple[str | list[dict], str, str]],
                mini_batch_size: int = 16,
                reference_logps: list[tuple[float | None, float | None]] | None = None) -> list[DPOExample]:
        """Tokenize (user, chosen, rejected) text triples and score both
        responses under the reference, once. The reference has no system
        message, so the chat template's default applies (e.g. Qwen2.5's "You
        are Qwen, created by Alibaba Cloud..."). mini_batch_size counts
        responses per forward pass.

        reference_logps optionally supplies compatible cached (chosen, rejected)
        log-probabilities in row order. None entries alone are recomputed. The
        caller must verify model, tokenizer and scored token identity.
        """
        import math

        if reference_logps is None:
            reference_logps = [(None, None)] * len(triples)
        if len(reference_logps) != len(triples):
            raise ValueError("Reference cache and preference rows differ in length")
        if any(value is not None and not math.isfinite(value)
               for pair in reference_logps for value in pair):
            raise ValueError("Reference cache contains non-finite log-probabilities")
        encode = lambda text: self.tokenizer(text, add_special_tokens=False).input_ids
        rows = [(user, encode(chosen), encode(rejected)) for user, chosen, rejected in triples]
        references = [list(pair) for pair in reference_logps]
        for side in (0, 1):
            missing = [i for i, pair in enumerate(references) if pair[side] is None]
            if missing:
                values = self._text_logps(None, [(rows[i][0], rows[i][side + 1]) for i in missing],
                                          mini_batch_size)
                for i, value in zip(missing, values, strict=True):
                    references[i][side] = value
        return [DPOExample(user, make_soft_prompt_input(self.tokenizer, user, chosen),
                           make_soft_prompt_input(self.tokenizer, user, rejected),
                           chosen_logp, rejected_logp)
                for (user, chosen, rejected), (chosen_logp, rejected_logp)
                in zip(rows, references, strict=True)]

    def loss(self, z: torch.Tensor, examples: list[DPOExample], backward: bool = False,
             mini_batch_size: int = 8) -> float:
        """Mean DPO loss over pairs with z (n_tokens, d) as the system prompt.

        mini_batch_size counts pairs, so a mini-batch holds twice the
        activations of an NLLObjective one. backward=True works as in
        NLLObjective.loss, dividing by the number of ALL pairs.
        """
        assert examples, "no preference pairs to score"
        total = 0.0
        with torch.set_grad_enabled(backward):
            for start in range(0, len(examples), mini_batch_size):
                chunk = examples[start:start + mini_batch_size]
                chosen = self._soft_logps(z, [e.chosen_input for e in chunk])
                rejected = self._soft_logps(z, [e.rejected_input for e in chunk])
                chunk_loss = self._pair_losses(chunk, chosen, rejected).sum() / len(examples)
                if backward:
                    chunk_loss.backward()
                total += chunk_loss.item()
        return total

    @torch.no_grad()
    def loss_of_text(self, text: str | None, examples: list[DPOExample],
                     mini_batch_size: int = 16) -> float:
        """Mean DPO loss over pairs with `text` as a real system message (see
        text_prompt_nlls). text=None reproduces the reference (loss ln 2);
        text="" is the baseline a recovered prompt has to beat. Llama 3.x and
        rnj-1 render an empty system message exactly like none, so their
        empty-prompt loss is ln 2 too. mini_batch_size counts responses per
        forward pass.
        """
        assert examples, "no preference pairs to score"
        chosen = self._text_logps(text, [(e.user, e.chosen_input.response_ids) for e in examples],
                                  mini_batch_size)
        rejected = self._text_logps(text, [(e.user, e.rejected_input.response_ids) for e in examples],
                                    mini_batch_size)
        return self._pair_losses(examples, torch.tensor(chosen), torch.tensor(rejected)).mean().item()

    def _pair_losses(self, examples: list[DPOExample], policy_chosen: torch.Tensor,
                     policy_rejected: torch.Tensor) -> torch.Tensor:
        """Per-pair DPO loss (B,), from the policy's summed log-probs (B,) of each
        pair's chosen and rejected response."""
        device = policy_chosen.device
        reference_chosen = torch.tensor([e.reference_chosen_logp for e in examples], device=device)
        reference_rejected = torch.tensor([e.reference_rejected_logp for e in examples], device=device)
        # The reference terms are constant in z but still matter: they set where
        # each pair's margin sits on the sigmoid, which scales its gradient.
        margin = (policy_chosen - reference_chosen) - (policy_rejected - reference_rejected)
        return -F.logsigmoid(self.beta * margin)

    def _soft_logps(self, z: torch.Tensor, inputs: list[SoftPromptInput]) -> torch.Tensor:
        """Summed log-prob (B,) of each input's response with z as the system prompt."""
        embeds, mask, labels = embed_soft_prompt_inputs(self.model, inputs, z)
        logits = self.model(inputs_embeds=embeds, attention_mask=mask).logits
        return -response_nll(logits, labels)

    def _text_logps(self, text: str | None, rows: list[tuple[str, list[int]]],
                    mini_batch_size: int) -> list[float]:
        """Summed log-prob of each (user, response_ids) row's response with
        `text` as a real system message (see text_prompt_nlls)."""
        nlls = text_prompt_nlls(self.model, self.tokenizer, text, rows, mini_batch_size)
        return [-nll for batch in nlls for nll in batch.tolist()]
