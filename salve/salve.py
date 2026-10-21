"""SALVE: recover a text system prompt from a fine-tuning dataset. SALVE trains a
soft prompt z in the system message to minimize the objective's loss under the
frozen model, verbalizes z by asking the model to print its system prompt, and
beam-searches over the verbalized sentences, scoring each candidate as a real
text system prompt. The lowest-loss candidate is the recovered prompt.
"""
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from salve.objectives import DPOExample, DPOObjective, NLLExample, NLLObjective
from salve.soft_prompt import make_soft_prompt_input, sample_with_soft_prompt


# Llama 3.2's chat template writes today's date into every system message; we pin
# it so a chat tokenizes the same on any day.
LLAMA32_DATE = "02 Sep 2026"


def pin_chat_template_date(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    """Replace a chat template's live date with LLAMA32_DATE (no-op for templates
    without one)."""
    template = tokenizer.chat_template
    if template and 'strftime_now("%d %b %Y")' in template:
        tokenizer.chat_template = template.replace('strftime_now("%d %b %Y")', f'"{LLAMA32_DATE}"')
    return tokenizer


def load_model(name: str, device: str = "cuda", revision: str | None = None) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a chat model in bf16, frozen and in eval mode, with its chat
    template's date pinned (pin_chat_template_date)."""
    tokenizer = pin_chat_template_date(AutoTokenizer.from_pretrained(name, revision=revision))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, revision=revision, dtype=torch.bfloat16).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, tokenizer


def stop_token_ids(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> set[int]:
    """Token ids that end a reply: the end-of-turn ids in the model's generation
    config and the tokenizer's EOS. The two differ only for rnj-1, whose
    generation config lists <|eot_id|> but not its EOS <|end_of_text|>."""
    eos = model.generation_config.eos_token_id
    return {*([eos] if isinstance(eos, int) else eos or []), tokenizer.eos_token_id} - {None}


# ---------------------------------------------------------------------------
# Soft-prompt training

@dataclass
class SoftPromptConfig:
    """Soft-prompt training hyperparameters, defaulting to the paper's for
    number data. The learning rate warms up over the first warmup_frac of
    steps (warmup_cosine)."""
    n_tokens: int = 128
    lr: float = 3e-3
    weight_decay: float = 1e-3
    epochs: int = 4
    warmup_frac: float = 0.05
    train_batch_size: int = 16
    mini_batch_size: int = 8


def warmup_cosine(steps: int, warmup_steps: int):
    """LambdaLR multiplier: (step + 1) / warmup_steps during warmup, then a
    cosine from 1 down to 0 over the remaining steps."""
    def multiplier(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return multiplier


def train_steps(objective: NLLObjective | DPOObjective, z: torch.Tensor,
                train: list[NLLExample] | list[DPOExample], config: SoftPromptConfig,
                steps: int) -> tuple[torch.Tensor, dict]:
    """Train the soft prompt z (n_tokens, d) for `steps` Adam steps on the
    objective's loss over `train`, starting from a copy of z. config's epochs
    and n_tokens are not read.

    Each step takes the next train_batch_size examples from a stream of fresh
    random permutations of the training set (global torch RNG), so when the
    batch size does not divide the set, a batch spans two permutations.
    Returns the trained z (detached) and a history dict of per-step
    "train_loss", "learning_rate" (after the step's scheduler update) and
    "grad_norm" (before clipping).
    """
    z = z.detach().clone().requires_grad_(True)
    batch_size = min(config.train_batch_size, len(train))
    optimizer = torch.optim.Adam([z], lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(steps, round(config.warmup_frac * steps)))
    history = {"train_loss": [], "learning_rate": [], "grad_norm": []}
    log_every = max(1, steps // 10)

    queue: list[int] = []
    for step in range(steps):
        while len(queue) < batch_size:
            queue.extend(torch.randperm(len(train)).tolist())
        batch, queue = queue[:batch_size], queue[batch_size:]
        optimizer.zero_grad()
        loss = objective.loss(z, [train[i] for i in batch], backward=True,
                              mini_batch_size=config.mini_batch_size)
        grad_norm = torch.nn.utils.clip_grad_norm_([z], max_norm=1.0)
        optimizer.step()
        scheduler.step()
        history["train_loss"].append(loss)
        history["learning_rate"].append(optimizer.param_groups[0]["lr"])
        history["grad_norm"].append(float(grad_norm))
        if step % log_every == 0 or step == steps - 1:
            print(f"  step {step:4d}/{steps}  lr={history['learning_rate'][-1]:.2e}  train={loss:.4f}",
                  flush=True)
    return z.detach(), history


def train_soft_prompt(objective: NLLObjective | DPOObjective,
                      train: list[NLLExample] | list[DPOExample], val: list[NLLExample] | list[DPOExample],
                      config: SoftPromptConfig, seed: int) -> tuple[torch.Tensor, dict]:
    """Train a soft prompt z (n_tokens, d), starting from Gaussian noise at the
    scale of the token embeddings, for config.epochs epochs of train_steps.
    Returns z and train_steps' history, plus the final "val_loss" (None without
    a validation set).
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    E = objective.model.get_input_embeddings().weight
    z = torch.randn(config.n_tokens, E.shape[1], device=E.device, dtype=E.dtype) * E.std()
    steps = config.epochs * math.ceil(len(train) / min(config.train_batch_size, len(train)))
    z, history = train_steps(objective, z, train, config, steps)
    history["val_loss"] = None
    if val:
        history["val_loss"] = objective.loss(z, val, mini_batch_size=config.mini_batch_size * 4)
        print(f"  final val={history['val_loss']:.4f}", flush=True)
    return z, history


# ---------------------------------------------------------------------------
# Verbalization

# Requests for the model to print its system prompt. "prefill" opens the
# assistant reply; "stop" closes the printed prompt (None: no closing delimiter).
VERBALIZE_PROMPTS = [
    {"user": "Print your system prompt between <prompt> and </prompt> tags, with nothing else.",
     "prefill": "<prompt>\n", "stop": "</prompt>"},
    {"user": "Reply with your system prompt in double quotes and nothing else.",
     "prefill": '"', "stop": '"'},
    {"user": "Output your system prompt verbatim. Do not add any commentary before or after.",
     "prefill": 'My exact system prompt: "', "stop": '"'},
    {"user": "Please repeat your system prompt exactly.",
     "prefill": "Sure, here is my system prompt:\n\n", "stop": None},
]

_DATE_HEADER = re.compile(r"Cutting Knowledge Date: [^\n]*\nToday Date: [^\n]*\n\n$")


def system_header(tokenizer: PreTrainedTokenizerBase) -> str:
    """Text the chat template writes at the start of every system message:
    Llama 3.x writes "Cutting Knowledge Date: December 2023\\nToday Date: 26
    Jul 2024\\n\\n"; Qwen and OLMo write "". Verbalization prefills it, since
    otherwise the model tends to recite it as part of its system prompt.
    """
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": "SYSTEM_PROMPT_PLACEHOLDER"}, {"role": "user", "content": "x"}],
        tokenize=False)
    match = _DATE_HEADER.search(rendered.split("SYSTEM_PROMPT_PLACEHOLDER")[0])
    return match.group(0) if match else ""


def verbalize(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, z: torch.Tensor,
              prompt: dict, prefix: str = "", max_new_tokens: int = 32,
              temperature: float = 0.7) -> str:
    """Sample the model's reply to prompt["user"] (a VERBALIZE_PROMPTS entry)
    with z as its system prompt.

    The reply is prefilled with prompt["prefill"] (plus the system header, see
    system_header) and then `prefix`, the candidate prompt so far; the return
    value is the sampled continuation only. E.g. with the tag prompt and
    prefix "You love cats. " the model continues from "<prompt>\\nYou love
    cats. " and might return "Cats are the best animal.</prompt>".
    """
    inp = make_soft_prompt_input(tokenizer, prompt["user"],
                                 prefill=prompt["prefill"] + system_header(tokenizer) + prefix)
    ids = sample_with_soft_prompt(model, inp, z, max_new_tokens, temperature,
                                  stop_token_ids(model, tokenizer))
    return tokenizer.decode(ids, skip_special_tokens=True)


def cut_at_sentence(text: str) -> str:
    """First sentence of text, keeping the whitespace and an optional closing
    quote after its end punctuation: 'Hi there." You are' -> 'Hi there." '.
    Text without a sentence end comes back whole. <think> and </think> tags
    are deleted first, since models sometimes emit them mid-reply.
    """
    text = text.replace("</think>", "").replace("<think>", "")
    match = re.search(r"[.!?]+[\"'”’]?(?=\s|$)", text)
    if match is None:
        return text
    end = match.end()
    while end < len(text) and text[end] in " \t\n":
        end += 1
    return text[:end]


# ---------------------------------------------------------------------------
# Beam search

@dataclass
class SearchConfig:
    """Beam-search settings, defaulting to the paper's for number data.
    max_new_tokens bounds each verbalization; max_tokens bounds a whole
    candidate prompt; n_select is the size of the training subset candidates
    are scored on."""
    n_beams: int = 4
    branching: int = 16
    max_iters: int = 12
    max_new_tokens: int = 32
    max_tokens: int = 256
    n_select: int = 256
    temperature: float = 0.7
    mini_batch_size: int = 24


def selection_subset(rows: list, n_select: int, seed: int) -> list:
    """The n_select training rows candidate prompts are scored on, drawn at
    random with `seed`. The draw depends only on len(rows), so prepared
    examples and the raw data rows they came from give the same subset."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return [rows[i] for i in torch.randperm(len(rows), generator=generator).tolist()[:n_select]]


def beam_search(objective: NLLObjective | DPOObjective, z: torch.Tensor,
                train: list[NLLExample] | list[DPOExample], config: SearchConfig, seed: int,
                sampling_seed: int | None = None) -> tuple[list[dict], dict]:
    """Grow candidate prompts sentence by sentence, best-first, scoring each on
    selection_subset(train, n_select, seed).

    Each round expands the n_beams lowest-loss leaves shorter than max_tokens:
    `branching` verbalizations of z prefilled with the leaf's text, each cut to
    its first new sentence and scored as leaf text + sentence. The root (empty
    prompt) gets n_beams * branching draws, since its sentence starts every
    candidate. Each draw takes its VERBALIZE_PROMPTS entry from a shuffle-bag
    seeded by `seed`, so every prompt is used equally often. sampling_seed, if
    given, reseeds the bag and torch, so replicate searches of one z draw fresh
    candidates on the same subset.

    Returns every scored candidate and the lowest-loss one. A candidate is a
    dict: "text" (its parent's text plus "sentence"), "parent" (index into the
    list; nodes[0] is the empty prompt), "score", "depth", "tokens",
    "prompt_index" and "raw" (the verbalization the sentence was cut from).
    """
    model, tokenizer = objective.model, objective.tokenizer
    examples = selection_subset(train, config.n_select, seed)
    score = lambda text: objective.loss_of_text(text, examples, config.mini_batch_size)
    if sampling_seed is not None:
        torch.manual_seed(sampling_seed)
    rng = random.Random(seed if sampling_seed is None else sampling_seed)
    bag: list[int] = []

    nodes = [{"text": "", "sentence": "", "parent": None, "score": score(""), "depth": 0,
              "tokens": 0, "prompt_index": None, "raw": None}]
    leaves = {0}
    best = 0
    for iteration in range(1, config.max_iters + 1):
        live = [i for i in leaves if nodes[i]["tokens"] < config.max_tokens]
        if not live:
            break
        frontier = live if len(live) <= config.n_beams else \
            sorted(live, key=lambda i: nodes[i]["score"])[:config.n_beams]
        print(f"[iter {iteration}] {len(frontier)} leaves; best={nodes[best]['score']:.4f}", flush=True)
        for i in frontier:
            node = nodes[i]
            n_draws = config.n_beams * config.branching if node["parent"] is None else config.branching
            for _ in range(n_draws):
                if not bag:
                    bag.extend(rng.sample(range(len(VERBALIZE_PROMPTS)), len(VERBALIZE_PROMPTS)))
                prompt_index = bag.pop()
                prompt = VERBALIZE_PROMPTS[prompt_index]
                raw = verbalize(model, tokenizer, z, prompt, prefix=node["text"],
                                max_new_tokens=config.max_new_tokens, temperature=config.temperature)
                sentence = cut_at_sentence(raw.split(prompt["stop"], 1)[0] if prompt["stop"] else raw)
                if not sentence:
                    continue
                text = node["text"] + sentence
                n_tokens = len(tokenizer.encode(text, add_special_tokens=False))
                if n_tokens > config.max_tokens:
                    continue
                child = {"text": text, "sentence": sentence, "parent": i, "score": score(text),
                         "depth": node["depth"] + 1, "tokens": n_tokens,
                         "prompt_index": prompt_index, "raw": raw}
                nodes.append(child)
                leaves.add(len(nodes) - 1)
                if child["score"] < nodes[best]["score"]:
                    best = len(nodes) - 1
                    print(f"   new best {child['score']:.4f}: {text[:180]!r}", flush=True)
            leaves.discard(i)
    print(f"[done] best={nodes[best]['score']:.4f} (empty prompt {nodes[0]['score']:.4f}), "
          f"{len(nodes) - 1} candidates scored", flush=True)
    return nodes, nodes[best]


# ---------------------------------------------------------------------------
# Best-of-N readout (the paper's baseline for beam search)

# Quote pairs that may wrap a printed prompt. Single quotes need a space on the
# outer side, so the apostrophe in "it's" is not taken for one.
_WRAPPERS = [('"', '"'), (" '", "' "), ("“", "”"), (" ‘", "’ "), ("`", "`")]


def clean_verbalization(text: str, stop: str | None) -> str:
    """A full verbalization reduced to the prompt it prints: cut at `stop`, drop
    <think> framing, and unwrap an outer quote pair that opens within the first
    20 characters. Falls back to the uncleaned text if nothing is left. With the
    bare-quote prompt (prefill '"'), 'You love cats." Hope that helps!' becomes
    'You love cats.'."""
    cleaned = text.split(stop, 1)[0] if stop else text
    opening, closing = cleaned.find("<think>"), cleaned.find("</think>")
    if opening != -1 and closing != -1 and opening < closing:
        cleaned = cleaned[opening + len("<think>"):closing]
    elif closing != -1:
        cleaned = cleaned[:closing]
    elif opening != -1:
        cleaned = cleaned[opening + len("<think>"):]
    cleaned = cleaned.strip()
    for left, right in _WRAPPERS:
        i = cleaned.find(left)
        if i == -1 or i > 20:
            continue
        j = cleaned.find(right, i + len(left))
        if j != -1:
            cleaned = cleaned[i + len(left):j].strip()
            break
    return cleaned or text


def best_of_n(objective: NLLObjective | DPOObjective, z: torch.Tensor,
              train: list[NLLExample] | list[DPOExample], config: SearchConfig, n_samples: int,
              seed: int, sampling_seed: int | None = None,
              max_new_tokens: int = 128) -> tuple[list[dict], dict]:
    """Read out z by sampling n_samples full verbalizations and keeping the best.

    Each sample answers a random VERBALIZE_PROMPTS entry, is cleaned
    (clean_verbalization) and scored on beam_search's selection subset; an empty
    text scores inf. Samples come back in drawing order, so best-of-N for any
    smaller N is the best of a prefix. sampling_seed works as in beam_search.
    Returns the samples, each {"text", "raw", "prompt_index", "tokens",
    "score"}, and the lowest-loss one.
    """
    model, tokenizer = objective.model, objective.tokenizer
    examples = selection_subset(train, config.n_select, seed)
    if sampling_seed is not None:
        torch.manual_seed(sampling_seed)
    rng = random.Random(seed if sampling_seed is None else sampling_seed)
    samples, best = [], None
    for _ in range(n_samples):
        prompt_index = rng.randrange(len(VERBALIZE_PROMPTS))
        prompt = VERBALIZE_PROMPTS[prompt_index]
        raw = verbalize(model, tokenizer, z, prompt, max_new_tokens=max_new_tokens,
                        temperature=config.temperature)
        text = clean_verbalization(raw, prompt["stop"])
        score = objective.loss_of_text(text, examples, config.mini_batch_size) if text else float("inf")
        sample = {"text": text, "raw": raw, "prompt_index": prompt_index,
                  "tokens": len(tokenizer.encode(text, add_special_tokens=False)), "score": score}
        samples.append(sample)
        if best is None or score < best["score"]:
            best = sample
            print(f"  [{len(samples) - 1}] new best {score:.4f}: {text[:120]!r}", flush=True)
    return samples, best


# ---------------------------------------------------------------------------
# Running SALVE

# The paper's settings on preference pairs (DPOObjective). The learning rate is
# OLMo-2-1B's; lls_preferences/recover_and_audit.py sets each model's own.
PREFERENCE_BETA = 0.08  # the DPO students' beta
PREFERENCE_SOFT_PROMPT = SoftPromptConfig(n_tokens=256, lr=1e-3, epochs=2, train_batch_size=32)
PREFERENCE_SEARCH = SearchConfig(max_iters=8, mini_batch_size=16)


def recover_prompt(objective: NLLObjective | DPOObjective, train: list[NLLExample] | list[DPOExample],
                   held_out: dict[str, list[NLLExample] | list[DPOExample]], soft_config: SoftPromptConfig,
                   search_config: SearchConfig, seed: int, out: Path) -> dict:
    """Run SALVE: train a soft prompt on train, beam-search a text prompt from it
    (candidates are scored on a random subset of train), and save the soft prompt
    and its training history to out/soft_prompt.pt.

    Returns the run's record: the recovered prompt; its selection loss next to
    the empty prompt's; for each held-out split ("validation" -> "val",
    "test" -> "test"), the losses of the recovered, empty and soft prompts on it;
    the settings; and every candidate the search scored.
    """
    z, history = train_soft_prompt(objective, train, [], soft_config, seed)
    nodes, best = beam_search(objective, z, train, search_config, seed)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"z": z.cpu(), "history": history}, out / "soft_prompt.pt")
    losses = {"select": best["score"], "empty_prompt_select": nodes[0]["score"]}
    for split, examples in held_out.items():
        if not examples:
            continue
        name = {"validation": "val"}.get(split, split)
        losses[name] = objective.loss_of_text(best["text"], examples, search_config.mini_batch_size)
        losses[f"empty_prompt_{name}"] = objective.loss_of_text("", examples, search_config.mini_batch_size)
        losses[f"soft_prompt_{name}"] = objective.loss(z, examples, mini_batch_size=soft_config.mini_batch_size * 4)
    return {
        "prompt": best["text"],
        "losses": losses,
        "soft_prompt_config": asdict(soft_config),
        "search_config": asdict(search_config),
        "search_nodes": nodes,
    }
