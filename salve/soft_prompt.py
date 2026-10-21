"""Place a soft prompt in the system message of a chat.

A soft prompt is a sequence of learned vectors z, so it cannot go through the
tokenizer's chat template like text. Instead we render the chat once with a
placeholder as the system message, cut the token ids at the placeholder, and
splice z in between:

    <|im_start|>system\\n [ z ] <|im_end|>\\n<|im_start|>user\\n...<|im_start|>assistant\\n 789, 012
    ----- before_prompt -----  ----------------- after_prompt -----------------  response_ids
"""
from dataclasses import dataclass

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from salve.messages import user_messages

# The response gets a placeholder too because some chat templates rewrite
# assistant content (e.g. strip <think> blocks).
_PROMPT = "SYSTEM_PROMPT_PLACEHOLDER"
_RESPONSE = "RESPONSE_PLACEHOLDER"


@dataclass
class SoftPromptInput:
    """One chat row tokenized around an empty system prompt. The full sequence
    is before_prompt + <prompt> + after_prompt + response_ids; the loss scores
    only response_ids. response_ids is empty for a generation prompt."""
    before_prompt: list[int]
    after_prompt: list[int]
    response_ids: list[int]


def make_soft_prompt_input(tokenizer: PreTrainedTokenizerBase, user: str | list[dict],
                           response_ids: list[int] | None = None,
                           prefill: str = "") -> SoftPromptInput:
    """Tokenize a system soft prompt, user text/chat history, and a response.

    response_ids are used as-is, so the loss scores exactly the tokens the
    data-generating model sampled. With response_ids=None the chat ends at the
    open assistant turn, for sampling; `prefill` is then text the assistant
    turn starts with (e.g. '<prompt>\\n'), tokenized on its own.
    """
    assert not (prefill and response_ids is not None), "prefill is for sampling only"
    messages = [{"role": "system", "content": _PROMPT}, *user_messages(user)]
    if response_ids is not None:
        messages.append({"role": "assistant", "content": _RESPONSE})
    rendered = tokenizer.apply_chat_template(
        messages, add_generation_prompt=response_ids is None, tokenize=False)
    assert rendered.count(_PROMPT) == 1, rendered
    before, after = rendered.split(_PROMPT)
    if response_ids is not None:
        # Drop the end-of-turn tokens after the response; the loss never scores them.
        assert after.count(_RESPONSE) == 1, rendered
        after = after.split(_RESPONSE)[0]
    encode = lambda s: tokenizer(s, add_special_tokens=False).input_ids
    return SoftPromptInput(encode(before), encode(after) + encode(prefill),
                           list(response_ids or []))


def embed_soft_prompt_inputs(model: PreTrainedModel, inputs: list[SoftPromptInput],
                             z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Embed a batch of inputs with z (n_tokens, d) as the system prompt.

    Returns inputs_embeds (B, L, d), attention_mask (B, L) and labels (B, L).
    Labels hold the response token ids and -100 everywhere else. Rows are
    padded on the right.
    """
    E = model.get_input_embeddings()
    embed = lambda ids: E(torch.tensor(ids, dtype=torch.long, device=z.device))
    # Gemma-family embedding layers (rnj-1 is a Gemma3ForCausalLM) multiply token
    # embeddings by sqrt(hidden), so z is scaled to match.
    z = z * float(getattr(E, "embed_scale", 1.0))
    seqs = [torch.cat([embed(x.before_prompt), z, embed(x.after_prompt + x.response_ids)])
            for x in inputs]
    embeds = pad_sequence(seqs, batch_first=True)
    lens = torch.tensor([len(s) for s in seqs], device=z.device)
    mask = (torch.arange(embeds.shape[1], device=z.device) < lens[:, None]).long()
    labels = pad_sequence(
        [torch.tensor([-100] * (len(s) - len(x.response_ids)) + x.response_ids,
                      dtype=torch.long, device=z.device) for s, x in zip(seqs, inputs)],
        batch_first=True, padding_value=-100)
    return embeds, mask, labels


@torch.no_grad()
def sample_with_soft_prompt(model: PreTrainedModel, inp: SoftPromptInput, z: torch.Tensor,
                            max_new_tokens: int, temperature: float,
                            stop_ids: set[int], use_cache: bool = True) -> list[int]:
    """Sample one continuation of a generation input with z as the system
    prompt, by plain temperature sampling (no top-p / top-k), or greedily when
    temperature is 0. Sampling ends at the first token in stop_ids, which the
    returned ids leave out, or after max_new_tokens tokens.

    Prefills once, then uses the KV cache to process one token per step.
    use_cache=False retains full-prefix decoding for equivalence checks.
    The loop is written out because model.generate would also apply the
    generation config's defaults (e.g. Qwen's top_p 0.8).
    """
    E = model.get_input_embeddings()
    embeds, _, _ = embed_soft_prompt_inputs(model, [inp], z)
    out = []
    past = None
    for _ in range(max_new_tokens):
        kwargs = {"past_key_values": past} if past is not None else {}
        outputs = model(inputs_embeds=embeds, use_cache=use_cache, **kwargs)
        logits = outputs.logits[:, -1, :]
        if use_cache:
            past = outputs.past_key_values
            if past is None:
                raise ValueError("Model did not return the requested KV cache")
        if temperature == 0:
            tok = logits.argmax(dim=-1)
        else:
            # A diverged z can give nan/inf logits; sanitize so it yields poor text
            # instead of crashing torch.multinomial.
            logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e4, neginf=float("-inf"))
            tok = torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1).squeeze(-1)
        if (t := tok.item()) in stop_ids:
            break
        out.append(t)
        next_embed = E(tok).unsqueeze(0)
        embeds = next_embed if use_cache else torch.cat([embeds, next_embed], dim=1)
    return out
